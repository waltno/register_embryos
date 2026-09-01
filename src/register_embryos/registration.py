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
    "rotation_angles",
    "sinkhorn_plan",
    "ot_refine",
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


def rotation_angles(transform: np.ndarray) -> Tuple[float, float]:
    """``(in_plane_deg, out_of_plane_deg)`` of a 4x4 rigid transform.

    In-plane is rotation about z -- the axis a dorsally-mounted embryo is free to
    spin around, and the one a near-circular nucleus cloud cannot pin down.
    """
    R = transform[:3, :3]
    in_plane = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    # How far the z axis is tipped away from vertical.
    z_axis = R @ np.array([0.0, 0.0, 1.0])
    out_of_plane = float(np.degrees(np.arccos(np.clip(z_axis[2], -1.0, 1.0))))
    return in_plane, out_of_plane


def _constrain_rotation(
    transform: np.ndarray,
    centroid: np.ndarray,
    max_rotation_deg: Optional[float],
    inplane_only: bool,
) -> np.ndarray:
    """Clamp a transform to the rotation a manually oriented cohort should need.

    Two facts about this data make unconstrained ICP the wrong tool:

    * A 12-somite dorsal nucleus cloud is nearly a disc of revolution -- principal
      extents about 180 x 155 x 4, an in-plane aspect of only 1.18. The in-plane
      angle is therefore almost unconstrained by nucleus positions, and mean
      nearest-neighbour distance barely distinguishes a correct fit from one rotated
      180 degrees. Optimising that residual will happily flip an embryo end for end.
    * The anterior-posterior orientation is not actually unknown. It was set by eye
      in the widget, from the image, where it is obvious. Letting ICP re-derive it
      from a symmetric cloud discards better information than it has.

    So the rotation is capped, and optionally restricted to the z axis: out-of-plane
    tilt is even less constrained (the cloud is ~4 units thick against ~180 wide) and
    a large one is always fitting noise.
    """
    if max_rotation_deg is None and not inplane_only:
        return transform

    R = transform[:3, :3]
    if inplane_only:
        angle = np.arctan2(R[1, 0], R[0, 0])
        R = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                      [np.sin(angle), np.cos(angle), 0.0],
                      [0.0, 0.0, 1.0]])

    if max_rotation_deg is not None:
        angle = np.arctan2(R[1, 0], R[0, 0])
        cap = np.radians(max_rotation_deg)
        if abs(angle) > cap:
            angle = np.sign(angle) * cap
            R = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                          [np.sin(angle), np.cos(angle), 0.0],
                          [0.0, 0.0, 1.0]])
        elif not inplane_only:
            _, out_of_plane = rotation_angles(transform)
            if out_of_plane > max_rotation_deg:
                R = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                              [np.sin(angle), np.cos(angle), 0.0],
                              [0.0, 0.0, 1.0]])

    # Recover the translation that keeps the clamped rotation centred the same way.
    constrained = np.eye(4)
    constrained[:3, :3] = R
    original_centre = _apply_transform(centroid[None, :], transform)[0]
    constrained[:3, 3] = original_centre - R @ centroid
    return constrained


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
    max_rotation_deg: Optional[float] = None,
    inplane_only: bool = False,
    backend: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """Register ``source`` onto ``target``; returns ``(transformed, transform4x4)``.

    Args:
        max_correspondence_distance: in the coordinate units of the clouds -- for
            an unconverted nucleus table that is xy pixels, so the default 500 is
            calibrated to a 1024x1024 frame.  Too small and ICP has nothing to
            latch onto before the coarse alignment finishes the job; too large
            and it will happily pair unrelated regions.
        pca_init: PCA coarse alignment before ICP.  **Turn this off when the
            embryos have already been oriented by hand.** PCA derives the axes from
            the nucleus cloud, which for a dorsal embryo is nearly circular in plane
            (aspect ~1.18), so its answer is close to arbitrary -- and it will
            happily overwrite a correct manual orientation with a 180 degree flip.
        max_rotation_deg: cap the in-plane rotation ICP may apply.  With a manually
            oriented cohort this is the parameter that matters: it stops the fit
            wandering off an orientation that was set from the image, where anterior
            is obvious, using a cloud where it is not.
        inplane_only: restrict rotation to the z axis.  Out-of-plane tilt is
            essentially unconstrained for a cloud ~4 units thick and ~180 wide.
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

    transform = _constrain_rotation(
        transform, source.mean(axis=0), max_rotation_deg, inplane_only
    )
    return _apply_transform(source, transform), transform


# ---------------------------------------------------------------------------
# Optimal-transport refinement
# ---------------------------------------------------------------------------

def sinkhorn_plan(
    cost: np.ndarray,
    epsilon: float,
    n_iter: int = 200,
    weights_source: Optional[np.ndarray] = None,
    weights_target: Optional[np.ndarray] = None,
    tol: float = 1e-9,
) -> np.ndarray:
    """Entropic optimal-transport plan for a cost matrix, in the log domain.

    Log-domain because the direct form computes ``exp(-cost/epsilon)``, which
    underflows to zero for the small ``epsilon`` that makes the plan informative --
    and then every row normalises to nan. Implemented here rather than pulled from
    POT to keep the dependency list short; it is a few lines of Sinkhorn iteration.

    Returns a plan whose rows sum to ``weights_source`` and columns to
    ``weights_target``.
    """
    from scipy.special import logsumexp

    n, m = cost.shape
    a = np.full(n, 1.0 / n) if weights_source is None else weights_source / weights_source.sum()
    b = np.full(m, 1.0 / m) if weights_target is None else weights_target / weights_target.sum()

    log_a, log_b = np.log(a + 1e-300), np.log(b + 1e-300)
    scaled = -cost / epsilon
    f = np.zeros(n)
    g = np.zeros(m)

    for _ in range(n_iter):
        f_prev = f
        f = epsilon * (log_a - logsumexp(scaled + g[None, :] / epsilon, axis=1))
        g = epsilon * (log_b - logsumexp(scaled + f[:, None] / epsilon, axis=0))
        if np.max(np.abs(f - f_prev)) < tol:
            break

    return np.exp(scaled + f[:, None] / epsilon + g[None, :] / epsilon)


def ot_refine(
    source: np.ndarray,
    target: np.ndarray,
    epsilon: Optional[float] = None,
    n_iter: int = 200,
    max_points: int = 2000,
    max_rotation_deg: Optional[float] = None,
    inplane_only: bool = False,
    transform_model: str = "rigid",
    max_scale: float = 1.15,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Refine an already-rigidly-aligned pair using soft OT correspondences.

    ICP pairs each source point with its single nearest target point. That is brittle
    in two ways this data actually exhibits: the embryos have different nucleus counts
    and densities (2.5k-4.3k here), so many source points can pile onto one target
    point and dominate the fit; and a hard assignment makes the objective piecewise
    constant, which is what lets ICP sit in a local minimum. An entropic OT plan
    replaces that with a soft, mass-balanced correspondence -- every target point
    receives its share -- and the rigid fit is then solved against each source point's
    barycentric image under the plan.

    **This refines the fit; it does not resolve the orientation ambiguity.** A
    near-circular cloud is just as symmetric under OT as under nearest neighbours, so
    OT will not tell anterior from posterior either. Constrain the rotation for that
    (see :func:`icp_point_to_point`); the constraint is honoured here too.

    ``transform_model`` controls how much freedom this stage gets. Every option is a
    single **global** matrix -- there is no per-point displacement field -- so none of
    them can invent local agreement between embryos that genuinely differ:

    ``"rigid"``
        Rotation and translation only. The safe default.
    ``"similarity"``
        Adds one uniform scale, clamped to ``max_scale``. The mildest useful
        loosening, and the one that matches the actual variation: embryos differ in
        size and stage.
    ``"affine"``
        A bounded linear map -- stretch and shear -- with singular values clamped to
        ``[1/max_scale, max_scale]`` so no axis can collapse or blow up.

    A free-form non-rigid warp is deliberately not offered. It would align almost
    anything to almost anything, manufacturing exactly the agreement a consensus
    atlas is supposed to measure.

    Args:
        epsilon: entropic regularisation, in squared distance units. ``None`` sets it
            from the data (a small multiple of the median squared nearest-neighbour
            distance), which is what keeps it scale-free across cohorts.
        max_points: both clouds are uniformly subsampled to this before building the
            cost matrix, which is dense and O(n*m).
        transform_model: ``"rigid"``, ``"similarity"`` or ``"affine"``; see above.
        max_scale: bound on any scaling introduced by the latter two.
    """
    rng = np.random.default_rng(seed)

    def subsample(points: np.ndarray) -> np.ndarray:
        if len(points) <= max_points:
            return points
        return points[rng.choice(len(points), max_points, replace=False)]

    src_s, tgt_s = subsample(source), subsample(target)

    cost = ((src_s[:, None, :] - tgt_s[None, :, :]) ** 2).sum(axis=-1)

    if epsilon is None:
        # Scale-free: a few times the typical nearest-neighbour separation, squared.
        nn = cKDTree(tgt_s).query(tgt_s, k=2)[0][:, 1]
        epsilon = float(2.0 * np.median(nn) ** 2)
        epsilon = max(epsilon, 1e-6)

    plan = sinkhorn_plan(cost, epsilon=epsilon, n_iter=n_iter)
    mass = plan.sum(axis=1)
    keep = mass > 1e-12
    if keep.sum() < 3:
        if verbose:
            print("    [OT] plan degenerate; leaving the transform unchanged")
        return source.copy(), np.eye(4)

    # Barycentric projection: where the plan sends each source point.
    projected = (plan[keep] @ tgt_s) / mass[keep, None]
    weights = mass[keep]

    src_c = np.average(src_s[keep], axis=0, weights=weights)
    tgt_c = np.average(projected, axis=0, weights=weights)
    centred_src = src_s[keep] - src_c
    centred_tgt = projected - tgt_c
    covariance = (centred_src * weights[:, None]).T @ centred_tgt
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:          # reflection guard
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    if transform_model == "rigid":
        linear = rotation
    elif transform_model == "similarity":
        # One uniform scale. Embryos differ in size and stage, and a global scale is
        # the mildest way to absorb that -- it cannot deform anything locally.
        variance = float((weights[:, None] * centred_src**2).sum() / weights.sum())
        scale = float(np.trace(rotation.T @ covariance) / max(
            variance * weights.sum(), 1e-12))
        scale = float(np.clip(scale, 1.0 / max_scale, max_scale))
        linear = scale * rotation
    elif transform_model == "affine":
        # Weighted least-squares linear map, then bounded: the singular values are
        # clamped so the map can stretch and shear a little but never collapse or
        # blow up an axis. Still one global matrix -- no per-point displacement --
        # so it cannot invent local agreement between embryos.
        gram = (centred_src * weights[:, None]).T @ centred_src
        linear = np.linalg.solve(
            gram + 1e-9 * np.eye(3), (centred_src * weights[:, None]).T @ centred_tgt
        ).T
        u_a, sv, vt_a = np.linalg.svd(linear)
        sv = np.clip(sv, 1.0 / max_scale, max_scale)
        linear = u_a @ np.diag(sv) @ vt_a
        if np.linalg.det(linear) < 0:
            u_a[:, -1] *= -1
            linear = u_a @ np.diag(sv) @ vt_a
    else:
        raise ValueError(
            f"unknown transform_model {transform_model!r} "
            f"(expected 'rigid', 'similarity' or 'affine')"
        )

    transform = np.eye(4)
    transform[:3, :3] = linear
    transform[:3, 3] = tgt_c - linear @ src_c
    # The rotation cap applies to the rotational part only; a similarity or affine
    # map is decomposed, capped, and reassembled with its scale intact.
    if max_rotation_deg is not None or inplane_only:
        u_c, sv_c, vt_c = np.linalg.svd(transform[:3, :3])
        pure_rotation = u_c @ vt_c
        if np.linalg.det(pure_rotation) < 0:
            u_c[:, -1] *= -1
            pure_rotation = u_c @ vt_c
        rot_only = np.eye(4)
        rot_only[:3, :3] = pure_rotation
        capped = _constrain_rotation(
            rot_only, source.mean(axis=0), max_rotation_deg, inplane_only
        )
        stretch = vt_c.T @ np.diag(sv_c) @ vt_c        # symmetric positive part
        linear = capped[:3, :3] @ stretch
        transform = np.eye(4)
        transform[:3, :3] = linear
        transform[:3, 3] = tgt_c - linear @ src_c

    if verbose:
        before = float(cKDTree(target).query(source)[0].mean())
        after = float(cKDTree(target).query(_apply_transform(source, transform))[0].mean())
        in_plane, tilt = rotation_angles(transform)
        scales = np.linalg.svd(transform[:3, :3], compute_uv=False)
        print(f"    [OT] {transform_model}, epsilon={epsilon:.2f}  "
              f"mean NN {before:.3f} -> {after:.3f}  "
              f"(extra in-plane {in_plane:+.2f} deg, tilt {tilt:.2f} deg, "
              f"scale {scales.min():.3f}-{scales.max():.3f})")
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
    refine_with_ot: bool = False,
    ot_max_rotation_deg: float = 5.0,
    ot_kwargs: Optional[Dict[str, object]] = None,
    coord_cols: Sequence[str] = COORD_COLS,
    verbose: bool = True,
    **icp_kwargs,
) -> RegistrationResult:
    """Register a dict of ``embryo_id -> nucleus table`` onto one reference.

    For a cohort whose embryos were oriented by hand, pass
    ``pca_init=False, inplane_only=True, max_rotation_deg=30``. See
    :func:`icp_point_to_point` for why: the nucleus cloud of a dorsal embryo is
    nearly circular in plane, so nothing in it pins the anterior-posterior angle,
    and an unconstrained fit will overwrite a correct manual orientation.

    Args:
        n_downsample: isotropic downsample applied per embryo before ICP.  Set
            ``None`` to register the full clouds (slower, rarely better -- ICP on
            uniformly sampled clouds is both faster and less biased toward dense
            regions).
        refine_with_ot: after ICP, refine with soft optimal-transport
            correspondences (:func:`ot_refine`).  Capped tightly by
            ``ot_max_rotation_deg`` -- it is a refinement, not a second chance to
            re-orient.
        coord_cols: which columns to register on.  The default ``("x", "y", "z")``
            mixes units -- xy in pixels, z in bin indices -- so with this project's
            geometry z contributes only about 1.5% of the cost and ICP effectively
            ignores it.  Pass ``("x_um", "y_um", "z_um")`` to register in
            micrometres, where z is ~16%; that is also the only version in which a
            rotation matrix means a physical rotation rather than a shear.
        center_first: translate each cloud onto the reference centroid before
            ICP.  Useful for atlas-to-atlas alignment where the two clouds may
            sit in unrelated coordinate ranges.
    """
    if not frames:
        return RegistrationResult(pd.DataFrame(), pd.DataFrame(), "", {})

    ot_kwargs = dict(ot_kwargs or {})
    tables = {
        embryo_id: (
            isotropic_downsample(df, n_target=n_downsample, coord_cols=coord_cols)
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

    reference_cloud = tables[reference_embryo_id][list(coord_cols)].to_numpy(dtype=float)
    backend = "open3d" if HAS_OPEN3D else "numpy"
    if verbose:
        print(
            f"  [ICP] point-to-point ({backend}) -> reference "
            f"{reference_embryo_id} ({len(reference_cloud):,} points)"
        )

    registered_frames: List[pd.DataFrame] = []
    transforms: Dict[str, np.ndarray] = {}

    for embryo_id, df in tables.items():
        cloud = df[list(coord_cols)].to_numpy(dtype=float)
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

            if refine_with_ot:
                if verbose:
                    print(f"    {embryo_id}: OT refinement")
                transformed, extra = ot_refine(
                    transformed, reference_cloud,
                    max_rotation_deg=ot_max_rotation_deg,
                    inplane_only=icp_kwargs.get("inplane_only", False),
                    verbose=verbose, **ot_kwargs,
                )
                transform = extra @ transform

        out = df.copy()
        out[["x_reg", "y_reg", "z_reg"]] = transformed
        registered_frames.append(out)
        transforms[embryo_id] = transform

        if verbose and embryo_id != reference_embryo_id:
            in_plane, out_of_plane = rotation_angles(transform)
            flag = ""
            if abs(in_plane) > 90:
                flag = ("   <- WARNING: this reverses anterior-posterior. If the "
                        "cohort was oriented by hand, this is ICP overriding it.")
            elif abs(in_plane) > 30:
                flag = "   <- large in-plane correction; check it against the image"
            print(f"    [ROT] {embryo_id}: in-plane {in_plane:+.1f} deg, "
                  f"tilt {out_of_plane:.1f} deg{flag}")

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
