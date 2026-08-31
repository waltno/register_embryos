"""Composite ("atlas") embryo built from a registered cohort.

Anchor points are taken from the reference embryo's registered nuclei (or a
random subsample of them, to hit a target size).  At each anchor, the k nearest
nuclei *pooled across every embryo in the cohort* are averaged, both in position
and in gene intensity.  The result is one point cloud carrying the cohort's
consensus expression pattern with per-embryo noise averaged out.

Choosing k is a real trade-off, not a formality.  With N embryos, k about equal
to N averages roughly one nucleus per embryo, which smooths between-embryo
variability while preserving spatial detail.  Much larger k blurs the boundaries
of an expression domain -- worth it if you only care about broad domains, harmful
if you are measuring a domain edge.  :func:`atlas_diagnostics` reports the
neighbour radius and how many distinct embryos actually contributed, which is the
honest way to tell whether an atlas point is a consensus or just one embryo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .registration import COORD_COLS, REG_COLS, isotropic_downsample, register_frames

__all__ = [
    "Atlas",
    "build_atlas",
    "atlas_diagnostics",
    "align_atlases",
]


@dataclass
class Atlas:
    """A composite embryo plus the parameters and diagnostics behind it."""

    points: pd.DataFrame
    label: str
    genes: List[str]
    k_neighbors: int
    n_source_embryos: int
    reference_embryo_id: str
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __len__(self) -> int:
        return len(self.points)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.points.to_csv(path, index=False)
        print(f"  [ATLAS] {len(self)} points -> {path}")
        return path

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Atlas({self.label!r}, {len(self)} points, k={self.k_neighbors}, "
            f"{self.n_source_embryos} embryos, genes={self.genes})"
        )


def build_atlas(
    registered: pd.DataFrame,
    reference_embryo_id: Optional[str] = None,
    genes: Optional[Sequence[str]] = None,
    k_neighbors: int = 5,
    n_points: Optional[int] = None,
    label: str = "atlas",
    coord_cols: Sequence[str] = REG_COLS,
    weight_by_distance: bool = False,
    exclude_self_embryo: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> Atlas:
    """Average coordinates and gene intensities over k nearest cohort neighbours.

    Args:
        registered: a registered table with ``x_reg, y_reg, z_reg`` and one row
            per nucleus per embryo.
        genes: gene columns to average.  Defaults to every numeric column that is
            not a coordinate or an id.
        k_neighbors: neighbours pooled per anchor.  Roughly the embryo count is a
            good starting point.
        n_points: target atlas size.  ``None`` keeps one anchor per reference
            nucleus; a number takes a spatially uniform subsample of anchors, so
            downsampling does not thin one region more than another.
        weight_by_distance: inverse-distance weighting instead of a flat mean.
            Sharpens domain boundaries slightly at the cost of more noise.
        exclude_self_embryo: drop neighbours belonging to the anchor's own
            embryo.  Makes the atlas a genuine leave-one-out consensus -- use it
            when you intend to ask how well the reference embryo agrees with the
            atlas, since otherwise the reference is partly predicting itself.
    """
    if registered.empty:
        raise ValueError("registered table is empty")
    coords_present = [c for c in coord_cols if c in registered.columns]
    if len(coords_present) != 3:
        raise ValueError(
            f"registered table needs columns {tuple(coord_cols)}; found {coords_present}"
        )

    if reference_embryo_id is None:
        reference_embryo_id = str(registered["embryo_id"].iloc[0])
    reference_rows = registered[registered["embryo_id"] == reference_embryo_id]
    if reference_rows.empty:
        raise ValueError(f"reference {reference_embryo_id!r} not in the registered table")

    if genes is None:
        skip = set(coord_cols) | set(COORD_COLS) | {
            "embryo_id", "nucleus_id", "n_voxels", "x_um", "y_um", "z_um",
        }
        genes = [
            column
            for column in registered.columns
            if column not in skip and pd.api.types.is_numeric_dtype(registered[column])
        ]
    gene_list = [gene for gene in genes if gene in registered.columns]

    # Anchors: uniform subsample rather than uniform-at-random, so a thinned
    # atlas keeps the same shape as the full one.
    anchors_df = reference_rows
    if n_points is not None and n_points < len(reference_rows):
        anchors_df = isotropic_downsample(
            reference_rows, n_target=n_points, coord_cols=coords_present, seed=seed
        )
    anchors = anchors_df[coords_present].to_numpy(dtype=float)

    pool = registered[coords_present].to_numpy(dtype=float)
    pool_values = registered[gene_list].fillna(0).to_numpy(dtype=float) if gene_list else None
    pool_embryos = registered["embryo_id"].to_numpy()

    k = int(min(k_neighbors, len(pool)))
    if exclude_self_embryo:
        # Over-query, then mask own-embryo neighbours out and keep the first k.
        query_k = int(min(len(pool), max(k * 4, k + 8)))
    else:
        query_k = k

    tree = cKDTree(pool)
    distances, indices = tree.query(anchors, k=query_k)
    distances = np.atleast_2d(distances)
    indices = np.atleast_2d(indices)

    if exclude_self_embryo:
        anchor_embryos = anchors_df["embryo_id"].to_numpy()
        keep_mask = pool_embryos[indices] != anchor_embryos[:, None]
        indices, distances = _take_first_k(indices, distances, keep_mask, k)

    if weight_by_distance:
        weights = 1.0 / np.maximum(distances, 1e-9)
    else:
        weights = np.ones_like(distances)
    valid = np.isfinite(distances) & (indices >= 0)
    weights = np.where(valid, weights, 0.0)
    weight_sums = np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

    safe_indices = np.where(indices >= 0, indices, 0)
    averaged_coords = (pool[safe_indices] * weights[:, :, None]).sum(axis=1) / weight_sums
    atlas = pd.DataFrame(averaged_coords, columns=["x", "y", "z"])

    if gene_list:
        averaged_values = (
            pool_values[safe_indices] * weights[:, :, None]
        ).sum(axis=1) / weight_sums
        for column_index, gene in enumerate(gene_list):
            atlas[gene] = averaged_values[:, column_index]

    atlas.insert(0, "embryo_id", label)
    atlas.insert(1, "atlas_point_id", np.arange(len(atlas)))

    diagnostics = _neighbourhood_diagnostics(
        distances, indices, pool_embryos, valid
    )
    atlas["n_source_embryos"] = diagnostics["n_embryos"].to_numpy()
    atlas["neighbor_radius"] = diagnostics["max_distance"].to_numpy()

    result = Atlas(
        points=atlas,
        label=label,
        genes=gene_list,
        k_neighbors=k,
        n_source_embryos=int(registered["embryo_id"].nunique()),
        reference_embryo_id=reference_embryo_id,
        diagnostics=diagnostics,
    )
    if verbose:
        print(
            f"  [ATLAS] {label}: {len(atlas):,} points | k={k} | "
            f"{result.n_source_embryos} embryos | genes={gene_list}"
        )
        print(
            f"          neighbour radius: median "
            f"{diagnostics['max_distance'].median():.2f}, p95 "
            f"{diagnostics['max_distance'].quantile(0.95):.2f}"
        )
        single = int((diagnostics["n_embryos"] <= 1).sum())
        if single:
            print(
                f"          [WARN] {single:,}/{len(atlas):,} points draw from a "
                f"single embryo -- not a consensus there; consider raising k"
            )
    return result


def _take_first_k(
    indices: np.ndarray, distances: np.ndarray, keep_mask: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep the first ``k`` kept-by-mask neighbours per row, padding with -1."""
    n_rows = indices.shape[0]
    out_indices = np.full((n_rows, k), -1, dtype=int)
    out_distances = np.full((n_rows, k), np.inf, dtype=float)
    for row in range(n_rows):
        kept = np.nonzero(keep_mask[row])[0][:k]
        out_indices[row, : len(kept)] = indices[row, kept]
        out_distances[row, : len(kept)] = distances[row, kept]
    return out_indices, out_distances


def _neighbourhood_diagnostics(
    distances: np.ndarray,
    indices: np.ndarray,
    pool_embryos: np.ndarray,
    valid: np.ndarray,
) -> pd.DataFrame:
    """Per-anchor neighbourhood size and embryo diversity."""
    rows = []
    for row in range(indices.shape[0]):
        keep = valid[row]
        row_distances = distances[row][keep]
        row_embryos = pool_embryos[indices[row][keep]] if keep.any() else np.array([])
        rows.append(
            {
                "n_neighbors": int(keep.sum()),
                "n_embryos": int(len(np.unique(row_embryos))),
                "mean_distance": float(row_distances.mean()) if keep.any() else np.nan,
                "max_distance": float(row_distances.max()) if keep.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def atlas_diagnostics(atlas: Atlas) -> pd.DataFrame:
    """One-row summary of an atlas's neighbourhood quality."""
    d = atlas.diagnostics
    return pd.DataFrame(
        [
            {
                "label": atlas.label,
                "n_points": len(atlas),
                "k_neighbors": atlas.k_neighbors,
                "n_source_embryos": atlas.n_source_embryos,
                "reference_embryo_id": atlas.reference_embryo_id,
                "median_neighbor_radius": float(d["max_distance"].median()),
                "p95_neighbor_radius": float(d["max_distance"].quantile(0.95)),
                "median_embryos_per_point": float(d["n_embryos"].median()),
                "pct_single_embryo_points": float(100.0 * (d["n_embryos"] <= 1).mean()),
            }
        ]
    )


def align_atlases(
    atlases: Dict[str, Atlas],
    reference_label: str,
    center_first: bool = True,
    output_root: Optional[str | Path] = None,
    verbose: bool = True,
    **icp_kwargs,
):
    """Point-to-point ICP between whole atlases, e.g. mutant atlas onto WT atlas.

    Puts several cohorts in one coordinate frame so their expression domains can
    be compared directly.  ``center_first`` translates onto the reference centroid
    before ICP, since two atlases built from different cohorts can sit in quite
    different coordinate ranges.
    """
    frames = {label: atlas.points.copy() for label, atlas in atlases.items()}
    if reference_label not in frames:
        raise ValueError(f"reference {reference_label!r} not among {list(frames)}")
    return register_frames(
        frames,
        reference_embryo_id=reference_label,
        n_downsample=None,
        center_first=center_first,
        output_root=output_root,
        verbose=verbose,
        **icp_kwargs,
    )
