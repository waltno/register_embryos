"""Assignment, orientation, registration, atlas and contrast, on synthetic data.

Synthetic rather than real ND2s so the tests run anywhere and assert on known
ground truth: a nucleus placed at a known spot with a known intensity, a cloud
rotated by a known angle.
"""

import numpy as np
import pandas as pd
import pytest

from register_embryos.assignment import (
    BACKGROUND_VALUE,
    assign_signal_pixels_2d,
    assign_signal_pixels_3d,
    build_signal_mask,
    nucleus_table,
)
from register_embryos.atlas import build_atlas
from register_embryos.contrast import (
    ContrastLimits,
    apply_contrast,
    auto_contrast_limits,
    log2_lift,
)
from register_embryos.imaging import VoxelSize, bin_volume_by_z, check_xy_shape, normalize_channel
from register_embryos.orientation import (
    Orientation,
    OrientationSet,
    apply_orientation_to_points,
    rotate_frame,
)
from register_embryos.registration import (
    icp_point_to_point,
    icp_residuals,
    isotropic_downsample,
    pca_align,
    register_frames,
)


# ---------------------------------------------------------------------------
# imaging
# ---------------------------------------------------------------------------

def test_bin_volume_uses_max_not_mean():
    """A single bright plane must survive binning; a mean would dilute it."""
    volume = np.zeros((10, 4, 4), dtype=np.float32)
    volume[3] = 1.0
    binned = bin_volume_by_z(volume, bin_size=5)
    assert binned.shape == (2, 4, 4)
    assert binned[0].max() == pytest.approx(1.0)
    assert binned[1].max() == pytest.approx(0.0)


def test_bin_size_one_is_a_copy():
    volume = np.random.default_rng(0).random((5, 3, 3)).astype(np.float32)
    assert np.array_equal(bin_volume_by_z(volume, 1), volume)


def test_bin_handles_a_ragged_last_bin():
    volume = np.ones((7, 2, 2), dtype=np.float32)
    assert bin_volume_by_z(volume, bin_size=3).shape == (3, 2, 2)


def test_normalize_channel_maps_onto_unit_range():
    volume = np.array([[[100.0, 200.0], [300.0, 400.0]]], dtype=np.float32)
    normalised = normalize_channel(volume)
    assert normalised.min() == pytest.approx(0.0)
    assert normalised.max() == pytest.approx(1.0)


def test_normalize_a_flat_channel_does_not_divide_by_zero():
    assert np.all(normalize_channel(np.full((2, 2, 2), 7.0)) == 0.0)


def test_voxel_anisotropy_and_binning():
    voxel = VoxelSize(xy_um=0.863, z_um=1.5)
    assert voxel.anisotropy == pytest.approx(1.738, abs=1e-3)
    # Binning 7 slices makes each z-bin 7x thicker.
    assert voxel.binned(7).anisotropy == pytest.approx(1.738 * 7, abs=1e-3)


def test_xy_shape_check_warns_but_can_raise():
    assert check_xy_shape((1024, 1024))
    assert not check_xy_shape((512, 512), strict=False)
    with pytest.raises(ValueError, match="1024x1024"):
        check_xy_shape((512, 512), strict=True)
    assert check_xy_shape((512, 512), expected=None)


# ---------------------------------------------------------------------------
# contrast
# ---------------------------------------------------------------------------

def test_apply_contrast_rescales_the_window():
    data = np.array([0.0, 0.2, 0.5, 0.8, 1.0], dtype=np.float32)
    out = apply_contrast(data, (0.2, 0.8))
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(0.5)
    assert out[4] == pytest.approx(1.0)


def test_apply_contrast_rejects_an_inverted_window():
    with pytest.raises(ValueError, match="vmax"):
        apply_contrast(np.zeros(3), (0.8, 0.2))


def test_log2_lift_preserves_the_unit_range():
    data = np.linspace(0, 1, 50, dtype=np.float32)
    lifted = log2_lift(data)
    assert lifted.min() == pytest.approx(0.0)
    assert lifted.max() == pytest.approx(1.0)
    # It lifts: a mid value moves up.
    assert lifted[25] > data[25]


def test_contrast_limits_round_trip(tmp_path):
    limits = ContrastLimits(transform="log2")
    limits.set("embryoA", 0, 0.05, 0.4)
    limits.set("embryoA", 1, 0.10, 0.6)
    path = limits.save(tmp_path / "contrast.json")
    reloaded = ContrastLimits.load(path)
    assert reloaded.transform == "log2"
    assert reloaded["embryoA"][1] == (0.10, 0.6)


def test_contrast_limits_reject_an_inverted_window():
    with pytest.raises(ValueError):
        ContrastLimits().set("e", 0, 0.7, 0.3)


# ---------------------------------------------------------------------------
# orientation
# ---------------------------------------------------------------------------

def test_rotating_by_360_is_a_no_op():
    frame = np.random.default_rng(1).random((16, 16))
    assert np.array_equal(rotate_frame(frame, 360.0), frame)


def test_rotate_frame_keeps_the_canvas_by_default():
    frame = np.zeros((32, 32))
    frame[8:24, 14:18] = 1.0
    assert rotate_frame(frame, 45.0).shape == (32, 32)
    assert rotate_frame(frame, 45.0, resize=True).shape[0] > 32


def test_orientation_identity_and_description():
    assert Orientation().is_identity
    assert not Orientation(xy_rotation=90).is_identity
    assert not Orientation(flip_x=True).is_identity
    assert Orientation(xy_rotation=360).is_identity
    assert "xy 90°" in Orientation(xy_rotation=90).describe()
    assert "flip xz" in Orientation(flip_x=True, flip_z=True).describe()


def test_out_of_plane_rotation_is_flagged():
    assert not Orientation(xy_rotation=90).needs_volumetric
    assert Orientation(xz_rotation=10).needs_volumetric


def test_orientation_set_round_trip_and_from_angles(tmp_path):
    ids = ["a", "b", "c", "d"]
    orientations = OrientationSet.from_angles(ids, [215, 155, 165, 70])
    assert orientations["b"].xy_rotation == 155
    path = orientations.save(tmp_path / "orientation.json")
    assert OrientationSet.load(path)["d"].xy_rotation == 70


def test_from_angles_length_mismatch_raises():
    with pytest.raises(ValueError, match="3 embryos but 2 angles"):
        OrientationSet.from_angles(["a", "b", "c"], [1, 2])


def test_point_rotation_is_exact_for_90_degrees():
    """Rotating a point cloud is exact, unlike interpolating pixels."""
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [0.0, 0.0], "z": [0.0, 0.0]})
    rotated = apply_orientation_to_points(
        df, Orientation(xy_rotation=90.0), center=(0.0, 0.0, 0.0)
    )
    assert rotated["x"].to_numpy() == pytest.approx([0.0, 0.0], abs=1e-9)
    assert rotated["y"].to_numpy() == pytest.approx([1.0, 2.0], abs=1e-9)


def test_point_rotation_by_360_returns_the_original():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.random((20, 3)), columns=["x", "y", "z"])
    rotated = apply_orientation_to_points(df, Orientation(xy_rotation=360.0))
    assert np.allclose(rotated[["x", "y", "z"]], df[["x", "y", "z"]])


# ---------------------------------------------------------------------------
# assignment
# ---------------------------------------------------------------------------

def _toy_channels():
    """One 4x4 frame: a nucleus at the top-left, gene signal beside it."""
    nuclei = np.zeros((1, 4, 4), dtype=np.float32)
    nuclei[0, 0:2, 0:2] = 1.0
    gene = np.zeros((1, 4, 4), dtype=np.float32)
    gene[0, 0, 2] = 0.8       # outside the nucleus, adjacent to it
    gene[0, 0, 0] = 0.4       # inside the nucleus
    return {0: nuclei, 1: gene}


def test_signal_mask_excludes_the_nuclear_channel():
    """Including nuclei would make the mask a nucleus mask, defeating the point."""
    channels = _toy_channels()
    mask = build_signal_mask(channels, signal_threshold=0.05, verbose=False)
    assert mask[0, 0, 2]        # gene signal outside the nucleus
    assert not mask[0, 1, 1]    # nuclear stain only -> not signal
    with_nuclei = build_signal_mask(
        channels, signal_threshold=0.05, gene_channels_only=False, verbose=False
    )
    assert with_nuclei[0, 1, 1]


def test_signal_pixels_are_assigned_to_the_nearest_nucleus():
    channels = _toy_channels()
    masks = np.zeros((1, 4, 4), dtype=int)
    masks[0, 0:2, 0:2] = 1
    mask = build_signal_mask(channels, signal_threshold=0.05, verbose=False)
    assigned = assign_signal_pixels_2d(masks, mask, verbose=False)
    assert assigned[0, 0, 2] == 1        # joined the nucleus
    assert assigned[0, 3, 3] == 0        # no signal there, still background


def test_3d_assignment_respects_a_distance_cap():
    masks = np.zeros((4, 8, 8), dtype=int)
    masks[0, 0, 0] = 1
    signal = np.zeros((4, 8, 8), dtype=bool)
    signal[0, 0, 1] = True     # right next to the nucleus
    signal[3, 7, 7] = True     # far away

    unlimited = assign_signal_pixels_3d(masks, signal, verbose=False)
    assert unlimited[0, 0, 1] == 1 and unlimited[3, 7, 7] == 1

    capped = assign_signal_pixels_3d(masks, signal, max_distance=2.0, verbose=False)
    assert capped[0, 0, 1] == 1 and capped[3, 7, 7] == 0


def test_3d_assignment_scales_z_by_voxel_size():
    """With a thick z step, an in-plane neighbour must win over a z neighbour."""
    masks = np.zeros((3, 5, 5), dtype=int)
    masks[1, 2, 0] = 1    # same slice as the query, 2 px away in x
    masks[0, 2, 2] = 2    # directly above the query, 1 z-bin away
    signal = np.zeros((3, 5, 5), dtype=bool)
    signal[1, 2, 2] = True

    # Isotropic: the z neighbour is closer (1 voxel vs 2).
    assert assign_signal_pixels_3d(
        masks, signal, voxel=(1.0, 1.0, 1.0), verbose=False
    )[1, 2, 2] == 2
    # z 10x thicker: the in-plane neighbour is closer (2 um vs 10 um).
    assert assign_signal_pixels_3d(
        masks, signal, voxel=(10.0, 1.0, 1.0), verbose=False
    )[1, 2, 2] == 1


def test_nucleus_table_excludes_the_background_sentinel():
    """A dropped pixel must not read as a measured zero.

    The nucleus covers 4 pixels but only one carries signal; the reported mean
    must be that one value, not diluted by three background sentinels.
    """
    assigned = np.zeros((1, 4, 4), dtype=int)
    assigned[0, 0:2, 0:2] = 1
    gene = np.full((1, 4, 4), BACKGROUND_VALUE, dtype=np.float32)
    gene[0, 0, 0] = 0.8
    table = nucleus_table(
        "e1", assigned, {1: gene}, {1: "geneA"}, mode="2d", verbose=False
    )
    assert len(table) == 1
    assert table["geneA"].iloc[0] == pytest.approx(0.8)
    assert table["n_voxels"].iloc[0] == 4


def test_nucleus_table_2d_repeats_ids_across_z_but_3d_does_not():
    masks = np.zeros((3, 4, 4), dtype=int)
    masks[:, 1:3, 1:3] = 1          # one object spanning all three slices
    channels = {1: np.full((3, 4, 4), 0.5, dtype=np.float32)}

    two_d = nucleus_table("e", masks, channels, {1: "g"}, mode="2d", verbose=False)
    assert len(two_d) == 3 and two_d["nucleus_id"].nunique() == 1

    three_d = nucleus_table("e", masks, channels, {1: "g"}, mode="3d", verbose=False)
    assert len(three_d) == 1
    assert three_d["z"].iloc[0] == pytest.approx(1.0)   # true 3D centroid


def test_nucleus_table_adds_micrometre_columns():
    masks = np.zeros((2, 4, 4), dtype=int)
    masks[0, 0, 0] = 1
    table = nucleus_table(
        "e", masks, {}, {}, mode="3d", voxel=(0.5, 10.0), verbose=False
    )
    assert table["x_um"].iloc[0] == pytest.approx(0.0)
    assert "z_um" in table.columns


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

def _blob(n=400, seed=0):
    rng = np.random.default_rng(seed)
    # Elongated so the principal axes are well determined.
    return rng.normal(0, 1, (n, 3)) * np.array([10.0, 4.0, 1.5])


def test_icp_recovers_a_known_rigid_transform():
    target = _blob()
    theta = np.deg2rad(25)
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    source = target @ rotation.T + np.array([15.0, -8.0, 3.0])

    registered, transform = icp_point_to_point(
        source, target, max_correspondence_distance=100.0, max_iteration=200
    )
    from scipy.spatial import cKDTree

    before = cKDTree(target).query(source)[0].mean()
    after = cKDTree(target).query(registered)[0].mean()
    assert after < before * 0.2
    assert transform.shape == (4, 4)


def test_pca_align_resolves_the_axis_sign_ambiguity():
    """A principal axis has no sign, so PCA alone can start an embryo backwards.

    A 180 degree rotation about z leaves both principal axes pointing along the
    same lines, so a sign-naive PCA alignment happily returns the flipped pose.
    pca_align must try the sign combinations and pick the one that actually fits.
    """
    from scipy.spatial import cKDTree

    target = _blob(seed=3)
    # Asymmetric along x so the flipped pose is genuinely a worse fit.
    target = target[target[:, 0] > -8]
    rotation = np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])  # 180 deg about z
    source = target @ rotation.T + np.array([40.0, 25.0, 0.0])

    transform = pca_align(source, target)
    aligned = (transform[:3, :3] @ source.T).T + transform[:3, 3]

    tree = cKDTree(target)
    assert tree.query(aligned)[0].mean() < tree.query(source)[0].mean() * 0.25
    # A proper rotation, not a reflection.
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1.0, abs=1e-6)


def test_icp_rejects_a_tiny_cloud():
    with pytest.raises(ValueError, match="at least 3"):
        icp_point_to_point(np.zeros((2, 3)), _blob())


def test_isotropic_downsample_hits_the_target_and_keeps_extent():
    rng = np.random.default_rng(4)
    # Deliberately non-uniform: a dense core plus a sparse shell.  Uniform-at-
    # random sampling would keep the core and lose the shell.
    dense = rng.normal(0, 1, (4000, 3))
    sparse = rng.normal(0, 12, (400, 3))
    df = pd.DataFrame(np.vstack([dense, sparse]), columns=["x", "y", "z"])

    out = isotropic_downsample(df, n_target=300)
    assert len(out) <= 300
    # The sparse shell survives: the retained extent is close to the original.
    assert out["x"].max() > df["x"].max() * 0.7
    assert out["x"].min() < df["x"].min() * 0.7


def test_downsample_below_target_is_a_passthrough():
    df = pd.DataFrame(np.random.default_rng(5).random((10, 3)), columns=["x", "y", "z"])
    assert len(isotropic_downsample(df, n_target=100)) == 10


def test_register_frames_and_residuals():
    target = _blob(seed=6)
    theta = np.deg2rad(30)
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    frames = {
        "ref": pd.DataFrame(target, columns=["x", "y", "z"]).assign(embryo_id="ref", g=0.5),
        "mov": pd.DataFrame(
            target @ rotation.T + 20.0, columns=["x", "y", "z"]
        ).assign(embryo_id="mov", g=0.5),
    }
    result = register_frames(frames, reference_embryo_id="ref", n_downsample=None, verbose=False)
    assert {"x_reg", "y_reg", "z_reg"} <= set(result.registered.columns)
    assert np.allclose(result.transform_of("ref"), np.eye(4))

    stats = icp_residuals(result.registered, "ref", verbose=False)
    row = stats.iloc[0]
    # Both sides use the same metric, so an improvement really is an improvement.
    assert row["mean_after"] < row["mean_before"]
    assert row["rms_after"] < row["rms_before"]


def test_residuals_reject_an_absent_reference():
    df = pd.DataFrame({
        "embryo_id": ["a"] * 3, "x": [0, 1, 2], "y": [0, 1, 2], "z": [0, 1, 2],
        "x_reg": [0, 1, 2], "y_reg": [0, 1, 2], "z_reg": [0, 1, 2],
    })
    with pytest.raises(ValueError, match="not present"):
        icp_residuals(df, "missing")


# ---------------------------------------------------------------------------
# atlas
# ---------------------------------------------------------------------------

def _registered_cohort(n_embryos=3, n_points=250, seed=7):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 5, (n_points, 3))
    frames = []
    for index in range(n_embryos):
        jitter = base + rng.normal(0, 0.2, base.shape)
        df = pd.DataFrame(jitter, columns=["x_reg", "y_reg", "z_reg"])
        df["x"], df["y"], df["z"] = df["x_reg"], df["y_reg"], df["z_reg"]
        df["embryo_id"] = f"e{index}"
        # A gene on in the +x half, with per-embryo offset noise.
        df["geneA"] = np.where(jitter[:, 0] > 0, 0.8, 0.0) + rng.normal(0, 0.02, n_points)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_atlas_averages_across_embryos_and_reduces_noise():
    registered = _registered_cohort()
    atlas = build_atlas(
        registered, reference_embryo_id="e0", genes=["geneA"], k_neighbors=3, verbose=False
    )
    assert len(atlas) == 250
    assert "geneA" in atlas.points.columns
    assert atlas.n_source_embryos == 3
    # The +x/-x split survives averaging.
    positive = atlas.points.loc[atlas.points["x"] > 1, "geneA"].mean()
    negative = atlas.points.loc[atlas.points["x"] < -1, "geneA"].mean()
    assert positive > 0.6 and negative < 0.2


def test_atlas_n_points_thins_the_cloud():
    atlas = build_atlas(
        _registered_cohort(), reference_embryo_id="e0", k_neighbors=3,
        n_points=60, verbose=False,
    )
    assert len(atlas) <= 60


def test_atlas_diagnostics_report_embryo_diversity():
    atlas = build_atlas(
        _registered_cohort(), reference_embryo_id="e0", k_neighbors=3, verbose=False
    )
    assert "n_source_embryos" in atlas.points.columns
    assert "neighbor_radius" in atlas.points.columns
    assert atlas.diagnostics["n_embryos"].median() >= 2


def test_atlas_can_exclude_the_anchors_own_embryo():
    """Leave-one-out: no atlas point may draw on its own embryo."""
    registered = _registered_cohort()
    atlas = build_atlas(
        registered, reference_embryo_id="e0", k_neighbors=2,
        exclude_self_embryo=True, verbose=False,
    )
    assert len(atlas) == 250
    assert not (atlas.points["n_source_embryos"] == 0).any()


def test_atlas_rejects_a_missing_reference():
    with pytest.raises(ValueError, match="not in the registered table"):
        build_atlas(_registered_cohort(), reference_embryo_id="nope", verbose=False)


def test_atlas_rejects_a_table_without_registered_coords():
    df = pd.DataFrame({"embryo_id": ["a"], "x": [0.0], "y": [0.0], "z": [0.0]})
    with pytest.raises(ValueError, match="needs columns"):
        build_atlas(df, reference_embryo_id="a", verbose=False)
