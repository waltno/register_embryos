"""Assign signal pixels to their nearest nucleus, then reduce to a nucleus table.

HCR signal sits in cytoplasm and around nuclei, not inside the nuclear stain, so
measuring gene intensity inside the Cellpose mask alone throws most of it away.
The fix, kept from the original pipeline: threshold each gene channel to find
signal-bearing pixels, drop everything below threshold as background, and give
each surviving pixel the label of the nearest nucleus.  The per-nucleus intensity
is then the mean over that expanded territory.

One correctness note carried over deliberately: pixels dropped as background are
set to ``BACKGROUND_VALUE`` (0.3) rather than 0, and the per-nucleus mean excludes
exactly that sentinel.  A dropped pixel must not read as "measured zero" -- that
would drag every nucleus mean toward zero in proportion to how much empty space
its territory covers.  The sentinel is a named constant here instead of a bare
0.3 repeated in four places.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .segmentation import SegmentedEmbryo

__all__ = [
    "BACKGROUND_VALUE",
    "EmbryoResult",
    "build_signal_mask",
    "assign_signal_pixels_2d",
    "assign_signal_pixels_3d",
    "nucleus_table",
    "build_nucleus_table",
    "build_cohort_tables",
]

#: Sentinel written into pixels that carry no signal.  Chosen non-zero so it is
#: distinguishable from a genuine zero-intensity measurement.
BACKGROUND_VALUE = 0.3


@dataclass
class EmbryoResult:
    """One embryo's per-nucleus table plus the provenance to reproduce it."""

    embryo_id: str
    nucleus_df: pd.DataFrame
    gene_map: Dict[int, str]
    nd2_path: str = ""
    output_dir: str = ""
    mode: str = "2d"
    contrast_limits: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    voxel_um: Tuple[float, float] = (1.0, 1.0)
    masks_path: str = ""
    assigned_masks_path: str = ""
    params: Dict[str, object] = field(default_factory=dict)

    @property
    def genes(self) -> List[str]:
        return [gene for gene in self.gene_map.values() if gene in self.nucleus_df.columns]

    @property
    def n_nuclei(self) -> int:
        if self.nucleus_df.empty:
            return 0
        return int(self.nucleus_df["nucleus_id"].nunique())


def build_signal_mask(
    channels: Dict[int, np.ndarray],
    signal_threshold: float = 0.05,
    gene_channels_only: bool = True,
    nuclei_channel: int = 0,
    verbose: bool = True,
) -> np.ndarray:
    """Boolean mask of pixels where at least one channel clears the threshold.

    Args:
        gene_channels_only: exclude the nuclear channel from the union.  The
            nuclear stain is bright nearly everywhere a nucleus is, so including
            it makes the mask a nucleus mask and defeats the point of expanding
            territory into where the HCR signal actually is.
    """
    considered = [
        index
        for index in sorted(channels)
        if not (gene_channels_only and index == nuclei_channel)
    ]
    if not considered:
        raise ValueError("no channels left to build a signal mask from")

    mask = np.zeros(channels[considered[0]].shape, dtype=bool)
    for index in considered:
        mask |= channels[index] > signal_threshold
    if verbose:
        pct = 100.0 * mask.sum() / mask.size
        print(
            f"  [SIGNAL] threshold={signal_threshold} over channels {considered}: "
            f"{mask.sum():,}/{mask.size:,} px ({pct:.2f}%)"
        )
    return mask


def _apply_background(
    channels: Dict[int, np.ndarray], signal_mask: np.ndarray
) -> Dict[int, np.ndarray]:
    """Copy channels with no-signal pixels replaced by ``BACKGROUND_VALUE``."""
    filtered: Dict[int, np.ndarray] = {}
    for index, data in channels.items():
        copy = data.copy()
        copy[~signal_mask] = BACKGROUND_VALUE
        filtered[index] = copy
    return filtered


def assign_signal_pixels_2d(
    nuclear_masks: np.ndarray,
    signal_mask: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """Nearest-nucleus assignment within each z-slice independently.

    Matches per-slice 2D labels: a pixel can only join a nucleus present on its
    own slice, so a slice with no nuclei contributes nothing.
    """
    assigned = nuclear_masks.copy()
    total = 0
    for z in range(nuclear_masks.shape[0]):
        frame = nuclear_masks[z]
        if frame.max() <= 0:
            continue
        unassigned = signal_mask[z] & (frame == 0)
        if not unassigned.any():
            continue
        nucleus_points = np.column_stack(np.nonzero(frame))
        tree = cKDTree(nucleus_points)
        query_points = np.column_stack(np.nonzero(unassigned))
        _, indices = tree.query(query_points)
        nearest = nucleus_points[indices]
        assigned[z][unassigned] = frame[nearest[:, 0], nearest[:, 1]]
        total += len(query_points)
    if verbose:
        print(f"  [ASSIGN 2D] {total:,} signal pixels assigned to nearest in-slice nucleus")
    return assigned


def assign_signal_pixels_3d(
    nuclear_masks: np.ndarray,
    signal_mask: np.ndarray,
    voxel: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_distance: Optional[float] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Nearest-nucleus assignment in 3D, with physical voxel scaling.

    Distances are measured in micrometres, not voxels, so an anisotropic stack
    does not preferentially assign pixels along z.  With ``max_distance`` set,
    pixels further than that from any nucleus stay unassigned instead of being
    dragged across the embryo to a distant nucleus.

    Args:
        voxel: ``(z_um, y_um, x_um)`` spacing, in that axis order.
    """
    assigned = nuclear_masks.copy()
    nucleus_indices = np.nonzero(nuclear_masks)
    if nucleus_indices[0].size == 0:
        if verbose:
            print("  [ASSIGN 3D] no nuclei; nothing assigned")
        return assigned

    scale = np.asarray(voxel, dtype=float)
    nucleus_voxels = np.column_stack(nucleus_indices)
    labels = nuclear_masks[nucleus_indices]

    unassigned = signal_mask & (nuclear_masks == 0)
    if not unassigned.any():
        if verbose:
            print("  [ASSIGN 3D] every signal pixel already inside a nucleus")
        return assigned
    query_voxels = np.column_stack(np.nonzero(unassigned))

    tree = cKDTree(nucleus_voxels * scale)
    distances, indices = tree.query(query_voxels * scale)

    keep = np.ones(len(query_voxels), dtype=bool) if max_distance is None else distances <= max_distance
    kept_voxels = query_voxels[keep]
    assigned[kept_voxels[:, 0], kept_voxels[:, 1], kept_voxels[:, 2]] = labels[indices[keep]]

    if verbose:
        dropped = int((~keep).sum())
        print(
            f"  [ASSIGN 3D] {int(keep.sum()):,} pixels assigned"
            + (f", {dropped:,} beyond {max_distance} um dropped" if dropped else "")
            + f" | median distance {np.median(distances[keep]) if keep.any() else 0:.2f} um"
        )
    return assigned


def nucleus_table(
    embryo_id: str,
    assigned_masks: np.ndarray,
    channels_filtered: Dict[int, np.ndarray],
    gene_map: Dict[int, str],
    mode: str = "2d",
    voxel: Tuple[float, float] = (1.0, 1.0),
    verbose: bool = True,
) -> pd.DataFrame:
    """Reduce label volume + channels to one row per nucleus.

    In ``"2d"`` mode a row is one label on one z-slice, so ``nucleus_id`` repeats
    across z (the original behaviour, kept so existing tables stay comparable).
    In ``"3d"`` mode labels are already z-consistent, so a row is one nucleus with
    a true 3D centroid, and ``nucleus_id`` is unique.

    Columns: ``embryo_id, nucleus_id, x, y, z, n_voxels, <gene...>``, plus
    ``x_um, y_um, z_um`` when a voxel size is given.
    """
    genes = dict(gene_map)
    rows: List[Dict[str, object]] = []

    if mode == "3d":
        label_iter = [(None, assigned_masks)]
    else:
        label_iter = [(z, assigned_masks[z]) for z in range(assigned_masks.shape[0])]

    for z_slice, label_array in label_iter:
        labels = np.unique(label_array[label_array > 0])
        for label in labels:
            region = label_array == label
            coords = np.nonzero(region)
            if coords[0].size == 0:
                continue

            if z_slice is None:  # 3D: coords are (z, y, x)
                row: Dict[str, object] = {
                    "embryo_id": embryo_id,
                    "nucleus_id": int(label),
                    "x": float(coords[2].mean()),
                    "y": float(coords[1].mean()),
                    "z": float(coords[0].mean()),
                    "n_voxels": int(coords[0].size),
                }
                gene_region = region
            else:  # 2D: coords are (y, x) within slice z_slice
                row = {
                    "embryo_id": embryo_id,
                    "nucleus_id": int(label),
                    "x": float(coords[1].mean()),
                    "y": float(coords[0].mean()),
                    "z": float(z_slice),
                    "n_voxels": int(coords[0].size),
                }
                gene_region = region

            for channel, gene in genes.items():
                if channel not in channels_filtered:
                    row[gene] = np.nan
                    continue
                data = channels_filtered[channel]
                values = data[gene_region] if z_slice is None else data[z_slice][gene_region]
                # Exclude the background sentinel: a pixel with no signal was
                # never measured, and averaging it in as ~0.3 would bias every
                # nucleus toward the sentinel value.
                measured = values[values != BACKGROUND_VALUE]
                row[gene] = float(measured.mean()) if measured.size else 0.0

            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        if verbose:
            print(f"  [TABLE] {embryo_id}: no nuclei found")
        return table

    if voxel and voxel != (1.0, 1.0):
        xy_um, z_um = voxel
        table["x_um"] = table["x"] * xy_um
        table["y_um"] = table["y"] * xy_um
        table["z_um"] = table["z"] * z_um

    ordered = (
        ["embryo_id", "nucleus_id", "x", "y", "z", "n_voxels"]
        + [c for c in ("x_um", "y_um", "z_um") if c in table.columns]
        + [gene for gene in genes.values() if gene in table.columns]
    )
    table = table[[column for column in ordered if column in table.columns]]

    if verbose:
        unique_nuclei = table["nucleus_id"].nunique()
        print(
            f"  [TABLE] {embryo_id}: {len(table):,} rows, {unique_nuclei:,} unique "
            f"nucleus ids, genes={list(genes.values())}"
        )
    return table


def build_nucleus_table(
    segmented: SegmentedEmbryo,
    signal_threshold: float = 0.05,
    gene_volume: Optional[Dict[int, np.ndarray]] = None,
    max_assign_distance_um: Optional[float] = None,
    save: bool = True,
    verbose: bool = True,
) -> EmbryoResult:
    """Full assignment + table for one segmented embryo.

    Args:
        gene_volume: replacement channel arrays for the gene channels, letting
            contrast be re-tuned for channels 1+ without re-running Cellpose.
            The nuclear channel always comes from ``segmented``.
    """
    volume = segmented.volume
    channels = dict(segmented.adjusted_channels)
    if gene_volume:
        for channel, data in gene_volume.items():
            if channel == 0:
                continue
            channels[channel] = data
        if verbose:
            print(f"  [CHANNELS] gene channels overridden: {sorted(gene_volume)}")

    signal_mask = build_signal_mask(
        channels, signal_threshold=signal_threshold, verbose=verbose
    )
    channels_filtered = _apply_background(channels, signal_mask)

    binned_voxel = volume.binned_voxel
    if segmented.is_3d:
        assigned = assign_signal_pixels_3d(
            segmented.nuclear_masks,
            signal_mask,
            voxel=(binned_voxel.z_um, binned_voxel.xy_um, binned_voxel.xy_um),
            max_distance=max_assign_distance_um,
            verbose=verbose,
        )
    else:
        assigned = assign_signal_pixels_2d(
            segmented.nuclear_masks, signal_mask, verbose=verbose
        )

    table = nucleus_table(
        embryo_id=volume.embryo_id,
        assigned_masks=assigned,
        channels_filtered=channels_filtered,
        gene_map=volume.gene_map,
        mode="3d" if segmented.is_3d else "2d",
        voxel=(binned_voxel.xy_um, binned_voxel.z_um),
        verbose=verbose,
    )

    assigned_path = ""
    if save and segmented.output_dir is not None:
        segmented.output_dir.mkdir(parents=True, exist_ok=True)
        assigned_path = str(segmented.output_dir / f"{volume.embryo_id}_assigned_masks.npy")
        np.save(assigned_path, assigned)
        table.to_csv(segmented.output_dir / f"{volume.embryo_id}_nucleus_table.csv", index=False)

    return EmbryoResult(
        embryo_id=volume.embryo_id,
        nucleus_df=table,
        gene_map=volume.gene_map,
        nd2_path=volume.nd2_path,
        output_dir=str(segmented.output_dir) if segmented.output_dir else "",
        mode="3d" if segmented.is_3d else "2d",
        voxel_um=(binned_voxel.xy_um, binned_voxel.z_um),
        masks_path=str(segmented.masks_path) if segmented.masks_path else "",
        assigned_masks_path=assigned_path,
        params={"signal_threshold": signal_threshold, **segmented.params},
    )


def build_cohort_tables(
    segmented_sets: Sequence[SegmentedEmbryo],
    signal_threshold: float = 0.05,
    gene_volumes: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
    max_assign_distance_um: Optional[float] = None,
    output_root: Optional[str | Path] = None,
    verbose: bool = True,
) -> Tuple[List[EmbryoResult], pd.DataFrame]:
    """Nucleus tables for a whole cohort, plus the concatenated table.

    Returns ``(results, combined_df)``.  ``combined_nucleus_table.csv`` is written
    to ``output_root`` when given -- the same filename the existing notebooks
    read, so they keep working against this package's output.
    """
    results: List[EmbryoResult] = []
    for index, segmented in enumerate(segmented_sets, start=1):
        if verbose:
            print(f"\n[{index}/{len(segmented_sets)}] {segmented.embryo_id}")
        results.append(
            build_nucleus_table(
                segmented,
                signal_threshold=signal_threshold,
                gene_volume=(gene_volumes or {}).get(segmented.embryo_id),
                max_assign_distance_um=max_assign_distance_um,
                verbose=verbose,
            )
        )

    non_empty = [r.nucleus_df for r in results if not r.nucleus_df.empty]
    combined = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()

    if output_root is not None and not combined.empty:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        combined_path = output_root / "combined_nucleus_table.csv"
        combined.to_csv(combined_path, index=False)
        if verbose:
            print(f"\n  [SAVE] {len(combined):,} rows -> {combined_path}")

    if verbose and not combined.empty:
        print(f"\n  [COHORT] {len(results)} embryos, {len(combined):,} nucleus rows")
        for result in results:
            print(f"    {result.embryo_id}: {result.n_nuclei:,} nuclei")
    return results, combined
