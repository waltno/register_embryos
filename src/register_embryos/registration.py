"""Rigid registration of embryo nucleus clouds by point-to-point ICP.

Every embryo in a cohort is aligned to one reference embryo, adding ``x_reg,
y_reg, z_reg`` to its nucleus table.  Alignment is pairwise against the
reference, so excluding one embryo never changes how the others land -- it only
changes the consensus atlas built from them.

Two implementations, same interface: Open3D's ICP when it is installed, and a
NumPy SVD (Kabsch) ICP otherwise.  Both are preceded by a PCA coarse alignment,
which matters because ICP is a local method -- two embryos mounted at
90 degrees to each other will not find each other from a cold start.

A note on residuals, learned the hard way in the original notebooks: report
before/after with the SAME metric.  Mean nearest-neighbour distance compared
against RMS nearest-neighbour distance makes a good fit look like a regression,
because RMS of a positive quantity always exceeds its mean.
:func:`icp_residuals` computes both sides identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

__all__ = [
    "HAS_OPEN3D",
    "RegistrationResult",
    "isotropic_downsample",
    "pca_align",
    "icp_point_to_point",
    "register_cohort",
    "icp_residuals",
    "register_frames",
]

try:  # pragma: no cover - environment dependent
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:  # pragma: no cover
    HAS_OPEN3D = False

COORD_COLS = ("x", "y", "z")
REG_COLS = ("x_reg", "y_reg", "z_reg")


@dataclass
class RegistrationResult:
    """Registered nucleus table plus per-embryo fit quality."""

    registered: pd.DataFrame
    stats: pd.DataFrame
    reference_embryo_id: str
    transforms: Dict[str, np.ndarray]

    @property
    def embryo_ids(self) -> List[str]:
        return list(self.registered["embryo_id"].unique())

    def transform_of(self, embryo_id: str) -> np.ndarray:
        return self.transforms[embryo_id]


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------

def isotropic_downsample(
    df: pd.DataFrame,
    n_target: int = 5000,
    coord_cols: Sequence[str] = COORD_COLS,
    seed: int = 42,
) -> pd.DataFrame:
    """Spatially uniform downsample to about ``n_target`` rows via a voxel grid.

    Each axis is normalised to [0, 1] before voxelising, so z (a handful of bins)
    is weighted like x and y (a thousand pixels) instead of collapsing into a
    single voxel layer.  A binary search finds the voxel size giving roughly
    ``n_target`` occupied voxels, then one random row is kept per voxel.

    Uniform in space, not uniform at random: random sampling would keep the
    densest regions dense, and ICP would then fit the dense regions and ignore
    the sparse ones.
    """
    coords = df[list(coord_cols)].to_numpy(dtype=float)
    if len(coords) <= n_target:
        return df.copy()

    rng = np.random.default_rng(seed)
    mins = coords.min(axis=0)
    span = coords.max(axis=0) - mins
    span[span == 0] = 1.0
    normed = (coords - mins) / span

    lo, hi, voxel_size = 0.0, 1.0, 1.0
    for _ in range(60):
        voxel_size = (lo + hi) / 2.0
        if voxel_size < 1e-12:
            break
        occupied = len({tuple(v) for v in np.floor(normed / voxel_size).astype(int)})
        if occupied > n_target:
            lo = voxel_size
        else:
            hi = voxel_size

    voxel_ids = np.floor(normed / voxel_size).astype(int)
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for row_index, voxel_id in enumerate(voxel_ids):
        buckets.setdefault(tuple(int(v) for v in voxel_id), []).append(row_index)
    kept = [int(rng.choice(rows)) for rows in buckets.values()]
    if len(kept) > n_target:
        kept = rng.choice(kept, size=n_target, replace=False).tolist()
    return df.iloc[sorted(kept)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Coarse alignment
# ---------------------------------------------------------------------------

def _mean_nn_distance(source: np.ndarray, target_tree: cKDTree) -> float:
    return float(target_tree.query(source)[0].mean())


def pca_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """4x4 rigid transform aligning source principal axes onto target axes.

    A principal axis has no inherent sign, so all four sign flips that keep a
    proper rotation (det = +1) are tried and the one with the lowest mean
    nearest-neighbour distance wins.  Without this, roughly three quarters of
    PCA initialisations start the embryo mirrored or upside down.
    """
    source_centre = source.mean(axis=0)
    target_centre = target.mean(axis=0)

    def axes(points: np.ndarray) -> np.ndarray:
        centred = points - points.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        return vt  # rows are principal axes, descending variance

    source_axes = axes(source)
    target_axes = axes(target)
    target_tree = cKDTree(target)

    best_transform = np.eye(4)
    best_score = _mean_nn_distance(source, target_tree)

    for flip in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]):
        rotation = target_axes.T @ np.diag(flip) @ source_axes
        if np.linalg.det(rotation) < 0:
            continue
        candidate = (rotation @ (source - source_centre).T).T + target_centre
        score = _mean_nn_distance(candidate, target_tree)
        if score < best_score:
            best_score = score
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = target_centre - rotation @ source_centre
            best_transform = transform
    return best_transform


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


# ---------------------------------------------------------------------------
# ICP
# ---------------------------------------------------------------------------

def _icp_numpy(
    source: np.ndarray,
    target: np.ndarray,
    max_correspondence_distance: float,
    max_iteration: int,
    tolerance: float,
) -> np.ndarray:
    """Point-to-point ICP by repeated Kabsch fits on nearest-neighbour pairs."""
    target_tree = cKDTree(target)
    transform = np.eye(4)
    current = source.copy()
    previous_error = np.inf

    for _ in range(max_iteration):
        distances, indices = target_tree.query(current)
        keep = distances <= max_correspondence_distance
        if keep.sum() < 3:
            break
        paired_source = current[keep]
        paired_target = target[indices[keep]]

        source_centre = paired_source.mean(axis=0)
        target_centre = paired_target.mean(axis=0)
        covariance = (paired_source - source_centre).T @ (paired_target - target_centre)
        u, _, vt = np.linalg.svd(covariance)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:  # reflection guard
            vt[-1, :] *= -1
            rotation = vt.T @ u.T
        translation = target_centre - rotation @ source_centre

        step = np.eye(4)
        step[:3, :3] = rotation
        step[:3, 3] = translation
        transform = step @ transform
        current = _apply_transform(source, transform)

        error = float(distances[keep].mean())
        if abs(previous_error - error) < tolerance:
            break
        previous_error = error
    return transform


def _icp_open3d(
    source: np.ndarray,
    target: np.ndarray,
    max_correspondence_distance: float,
    max_iteration: int,
    relative_fitness: float,
    relative_rmse: float,
    init: np.ndarray,
) -> np.ndarray:  # pragma: no cover - optional dependency
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source)
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target)

    result = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        max_correspondence_distance,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=relative_fitness,
            relative_rmse=relative_rmse,
            max_iteration=max_iteration,
        ),
    )
    return np.asarray(result.transformation)


def icp_point_to_point(
    source: np.ndarray,
    target: np.ndarray,
    max_correspondence_distance: float = 500.0,
    max_iteration: int = 500,
    relative_fitness: float = 1e-6,
    relative_rmse: float = 1e-6,
    tolerance: float = 1e-7,
    pca_init: bool = True,
    backend: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """Register ``source`` onto ``target``; returns ``(transformed, transform4x4)``.

    Args:
        max_correspondence_distance: in the coordinate units of the clouds -- for
            an unconverted nucleus table that is xy pixels, so the default 500 is
            calibrated to a 1024x1024 frame.  Too small and ICP has nothing to
            latch onto before the coarse alignment finishes the job; too large
            and it will happily pair unrelated regions.
        pca_init: PCA coarse alignment before ICP.  Leave on unless the clouds
            are already roughly aligned.
        backend: ``"auto"``, ``"open3d"`` or ``"numpy"``.
    """
    if len(source) < 3 or len(target) < 3:
        raise ValueError("both clouds need at least 3 points")

    init = pca_align(source, target) if pca_init else np.eye(4)
    use_open3d = backend == "open3d" or (backend == "auto" and HAS_OPEN3D)
    if backend == "open3d" and not HAS_OPEN3D:
        raise ImportError("backend='open3d' requested but open3d is not installed")

    if use_open3d:
        try:
            transform = _icp_open3d(
                source, target, max_correspondence_distance, max_iteration,
                relative_fitness, relative_rmse, init,
            )
        except Exception as exc:  # pragma: no cover
            print(f"    [ICP] open3d failed ({str(exc)[:70]}); using numpy ICP")
            transform = _icp_numpy(
                _apply_transform(source, init), target,
                max_correspondence_distance, max_iteration, tolerance,
            ) @ init
    else:
        transform = _icp_numpy(
            _apply_transform(source, init), target,
            max_correspondence_distance, max_iteration, tolerance,
        ) @ init

    return _apply_transform(source, transform), transform


# ---------------------------------------------------------------------------
# Cohort-level driver
# ---------------------------------------------------------------------------

def icp_residuals(
    registered: pd.DataFrame,
    reference_embryo_id: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """Nearest-neighbour residual to the reference, before vs after, like for like.

    Both sides use the same metric on the same points: pre-registration coords
    against the reference's pre-registration coords, and registered against
    registered.  Mixing a mean on one side with an RMS on the other is what made
    good fits look bad in the original notebooks.
    """
    reference = registered[registered["embryo_id"] == reference_embryo_id]
    if reference.empty:
        raise ValueError(f"reference {reference_embryo_id!r} not present in the table")
    raw_tree = cKDTree(reference[list(COORD_COLS)].to_numpy(dtype=float))
    reg_tree = cKDTree(reference[list(REG_COLS)].to_numpy(dtype=float))

    rows = []
    for embryo_id in registered["embryo_id"].unique():
        if embryo_id == reference_embryo_id:
            continue
        subset = registered[registered["embryo_id"] == embryo_id]
        before = raw_tree.query(subset[list(COORD_COLS)].to_numpy(dtype=float))[0]
        after = reg_tree.query(subset[list(REG_COLS)].to_numpy(dtype=float))[0]
        rows.append(
            {
                "embryo_id": embryo_id,
                "reference_embryo_id": reference_embryo_id,
                "n_points": len(subset),
                "mean_before": float(before.mean()),
                "mean_after": float(after.mean()),
                "median_before": float(np.median(before)),
                "median_after": float(np.median(after)),
                "rms_before": float(np.sqrt((before**2).mean())),
                "rms_after": float(np.sqrt((after**2).mean())),
            }
        )

    stats = pd.DataFrame(rows)
    if verbose and not stats.empty:
        for _, row in stats.iterrows():
            improvement = (
                (1 - row["mean_after"] / row["mean_before"]) * 100 if row["mean_before"] else 0.0
            )
            print(
                f"    {row['embryo_id']}: mean NN {row['mean_before']:7.2f} -> "
                f"{row['mean_after']:7.2f} ({improvement:+5.1f}%) | "
                f"rms {row['rms_before']:7.2f} -> {row['rms_after']:7.2f}"
            )
        print(
            f"    cohort mean NN: {stats['mean_before'].mean():.2f} -> "
            f"{stats['mean_after'].mean():.2f}"
        )
    return stats


def register_frames(
    frames: Dict[str, pd.DataFrame],
    reference_embryo_id: Optional[str] = None,
    n_downsample: Optional[int] = 5000,
    center_first: bool = False,
    output_root: Optional[str | Path] = None,
    verbose: bool = True,
    **icp_kwargs,
) -> RegistrationResult:
    """Register a dict of ``embryo_id -> nucleus table`` onto one reference.

    Args:
        n_downsample: isotropic downsample applied per embryo before ICP.  Set
            ``None`` to register the full clouds (slower, rarely better -- ICP on
            uniformly sampled clouds is both faster and less biased toward dense
            regions).
        center_first: translate each cloud onto the reference centroid before
            ICP.  Useful for atlas-to-atlas alignment where the two clouds may
            sit in unrelated coordinate ranges.
    """
    if not frames:
        return RegistrationResult(pd.DataFrame(), pd.DataFrame(), "", {})

    tables = {
        embryo_id: (
            isotropic_downsample(df, n_target=n_downsample)
            if n_downsample and len(df) > n_downsample
            else df.reset_index(drop=True)
        )
        for embryo_id, df in frames.items()
    }
    if verbose and n_downsample:
        for embryo_id, df in tables.items():
            print(f"    {embryo_id}: {len(frames[embryo_id]):,} -> {len(df):,} nuclei")

    if reference_embryo_id is None:
        reference_embryo_id = next(iter(tables))
    if reference_embryo_id not in tables:
        raise ValueError(
            f"reference {reference_embryo_id!r} not among {list(tables)}"
        )

    reference_cloud = tables[reference_embryo_id][list(COORD_COLS)].to_numpy(dtype=float)
    backend = "open3d" if HAS_OPEN3D else "numpy"
    if verbose:
        print(
            f"  [ICP] point-to-point ({backend}) -> reference "
            f"{reference_embryo_id} ({len(reference_cloud):,} points)"
        )

    registered_frames: List[pd.DataFrame] = []
    transforms: Dict[str, np.ndarray] = {}

    for embryo_id, df in tables.items():
        cloud = df[list(COORD_COLS)].to_numpy(dtype=float)
        if len(cloud) < 3:
            print(f"    [SKIP] {embryo_id}: fewer than 3 nuclei")
            continue

        if embryo_id == reference_embryo_id:
            transformed, transform = cloud.copy(), np.eye(4)
        else:
            offset = np.zeros(3)
            if center_first:
                offset = reference_cloud.mean(axis=0) - cloud.mean(axis=0)
            transformed, transform = icp_point_to_point(
                cloud + offset, reference_cloud, **icp_kwargs
            )
            shift = np.eye(4)
            shift[:3, 3] = offset
            transform = transform @ shift

        out = df.copy()
        out[["x_reg", "y_reg", "z_reg"]] = transformed
        registered_frames.append(out)
        transforms[embryo_id] = transform

    registered = (
        pd.concat(registered_frames, ignore_index=True) if registered_frames else pd.DataFrame()
    )
    stats = (
        icp_residuals(registered, reference_embryo_id, verbose=verbose)
        if not registered.empty
        else pd.DataFrame()
    )

    if output_root is not None and not registered.empty:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        registered.to_csv(output_root / "registered_nucleus_table.csv", index=False)
        stats.to_csv(output_root / "registration_residuals.csv", index=False)
        np.savez(
            output_root / "registration_transforms.npz",
            **{eid: matrix for eid, matrix in transforms.items()},
        )
        if verbose:
            print(f"  [SAVE] registration -> {output_root}")

    return RegistrationResult(registered, stats, reference_embryo_id, transforms)


def register_cohort(
    results,
    reference_embryo_id: Optional[str] = None,
    n_downsample: Optional[int] = 5000,
    output_root: Optional[str | Path] = None,
    verbose: bool = True,
    **icp_kwargs,
) -> RegistrationResult:
    """Register a list of :class:`~register_embryos.assignment.EmbryoResult`.

    Accepts either those objects or a single combined nucleus table with an
    ``embryo_id`` column.
    """
    if isinstance(results, pd.DataFrame):
        frames = {
            embryo_id: group.dropna(axis=1, how="all").reset_index(drop=True)
            for embryo_id, group in results.groupby("embryo_id", sort=False)
        }
    else:
        frames = {
            result.embryo_id: result.nucleus_df.dropna(axis=1, how="all")
            for result in results
            if not result.nucleus_df.empty
        }
    return register_frames(
        frames,
        reference_embryo_id=reference_embryo_id,
        n_downsample=n_downsample,
        output_root=output_root,
        verbose=verbose,
        **icp_kwargs,
    )
