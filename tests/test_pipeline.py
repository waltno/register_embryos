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


# ---------------------------------------------------------------------------
# Automatic contrast on a realistic HCR intensity distribution
# ---------------------------------------------------------------------------

def _hcr_like_channel(seed=0, shape=(8, 160, 160)):
    """A channel with the intensity profile real HCR data actually has.

    Fitted to a real 20x dorsal wt1a channel after normalisation:

        percentile   p50      p90      p99.5    p99.9    max
        real         0.0037   0.0253   0.0984   0.1765   1.000
        this         0.0037   0.0173   0.0845   0.1899   1.000

    The *shape* is the point, not just the range: a long dim lognormal shoulder
    that is all background, plus a very sparse bright tail (0.05% of pixels) that
    is the actual signal.  On a tight, narrow background any percentile pair looks
    fine -- it is this dim shoulder that makes a low percentile of 1 disastrous,
    because p1 then sits at the very bottom of the background rather than above it.
    """
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    values = rng.lognormal(mean=np.log(0.0037), sigma=1.2, size=n)   # background
    bright = rng.choice(n, size=max(1, int(0.0005 * n)), replace=False)
    values[bright] = rng.uniform(0.3, 1.0, bright.size)              # sparse puncta
    return np.clip(values, 0, 1).astype(np.float32).reshape(shape)


def test_the_hcr_fixture_matches_the_real_intensity_profile():
    """Guard the fixture itself: if it drifts, the contrast tests stop meaning anything."""
    channel = _hcr_like_channel()
    assert np.percentile(channel, 50) == pytest.approx(0.0037, rel=0.35)
    assert np.percentile(channel, 90) == pytest.approx(0.025, rel=0.45)
    assert np.percentile(channel, 99.5) == pytest.approx(0.098, rel=0.45)
    assert channel.max() > 0.9


def _volume_with(channels):
    from register_embryos.imaging import EmbryoVolume, VoxelSize
    from register_embryos.naming import parse_embryo_name

    return EmbryoVolume(
        name=parse_embryo_name("20260410_1.5_wt_12s_dorsal_20X_hand2_tbx1_wt1a.nd2"),
        binned_channels=channels,
        voxel=VoxelSize(0.863, 1.5), bin_size=7,
        c_size=len(channels), z_size=42,
    )


def test_a_low_percentile_of_one_stretches_background_into_signal():
    """Why the default is p90, not p1.

    On real data p1/p99.5 put 36-41% of pixels above the 0.05 threshold, so every
    nucleus read as expressing. The dim shoulder is background, not signal.
    """
    channel = _hcr_like_channel()
    volume = _volume_with({0: channel, 1: channel})

    naive = auto_contrast_limits(
        [volume], low_percentile=1.0, high_percentile=99.5, verbose=False
    )
    sensible = auto_contrast_limits([volume], verbose=False)   # p90 / p99.9

    naive_positive = (apply_contrast(channel, naive.get(volume.embryo_id, 1)) > 0.05).mean()
    good_positive = (apply_contrast(channel, sensible.get(volume.embryo_id, 1)) > 0.05).mean()

    # On real data this pair gave 36-41%; the fixture reproduces ~44%.
    assert naive_positive > 0.30          # background swept wholesale into signal
    assert good_positive < 0.15           # a plausible HCR positive fraction
    assert good_positive < naive_positive / 3


def test_auto_contrast_warns_when_almost_everything_reads_positive(capsys):
    channel = _hcr_like_channel()
    volume = _volume_with({0: channel, 1: channel})
    auto_contrast_limits(
        [volume], low_percentile=1.0, high_percentile=99.5, verbose=True
    )
    out = capsys.readouterr().out
    assert "background stretched into signal" in out
    assert "raise low_percentile" in out


def test_auto_contrast_does_not_warn_about_the_nuclear_channel(capsys):
    """DAPI is broadly positive by design, so a high fraction there is not a fault."""
    channel = _hcr_like_channel()
    volume = _volume_with({0: channel})        # channel 0 only
    auto_contrast_limits(
        [volume], low_percentile=1.0, high_percentile=99.5, verbose=True
    )
    assert "background stretched into signal" not in capsys.readouterr().out


def test_auto_contrast_reports_the_positive_fraction(capsys):
    volume = _volume_with({0: _hcr_like_channel(), 1: _hcr_like_channel(seed=1)})
    auto_contrast_limits([volume], verbose=True)
    out = capsys.readouterr().out
    assert "ch0=(" in out and "ch1=(" in out
    assert "+]" in out                         # the [NN%+] diagnostic


# ---------------------------------------------------------------------------
# preview_contrast must show rotation, not just contrast
# ---------------------------------------------------------------------------

def _asymmetric_volume():
    """A volume with an off-centre blob, so a rotation is detectable in the pixels."""
    from register_embryos.imaging import EmbryoVolume, VoxelSize
    from register_embryos.naming import parse_embryo_name

    data = np.zeros((3, 64, 64), np.float32)
    data[:, 8:24, 8:40] = 0.8          # deliberately not centred or symmetric
    return EmbryoVolume(
        name=parse_embryo_name("20260410_1.5_wt_12s_dorsal_20X_hand2_tbx1_wt1a.nd2"),
        binned_channels={0: data, 1: data.copy()},
        voxel=VoxelSize(0.863, 1.5), bin_size=7, c_size=2, z_size=21,
    )


def _rendered_image(fig, row=0):
    """Pixels of the left-hand 'oriented raw' panel of a given row."""
    return np.asarray(fig.axes[row * 3].images[0].get_array())


def test_preview_contrast_applies_the_rotation_from_a_prep_config():
    """Regression: the preview showed contrast on the UNROTATED volume.

    A QC figure that omits the rotation does not show what segmentation receives,
    so a wrong rotation passes the check unnoticed.
    """
    import matplotlib
    matplotlib.use("Agg")
    from register_embryos.contrast import ContrastLimits, preview_contrast
    from register_embryos.orientation import Orientation, OrientationSet
    from register_embryos.widgets import PrepConfig

    volume = _asymmetric_volume()
    eid = volume.embryo_id
    config = PrepConfig(
        orientations=OrientationSet({eid: Orientation(xy_rotation=90.0)}),
        contrast=ContrastLimits.from_dict({eid: {0: (0.05, 0.5), 1: (0.05, 0.5)}}),
    )

    rotated = _rendered_image(preview_contrast(volume, config))
    upright = _rendered_image(
        preview_contrast(volume, config, orientation=Orientation())
    )
    assert not np.allclose(rotated, upright), "rotation was ignored"
    assert rotated.shape == upright.shape        # canvas kept by default


def test_preview_contrast_reports_the_rotation_in_the_title():
    """The figure has to be self-documenting, or a QC folder is unreadable later."""
    import matplotlib
    matplotlib.use("Agg")
    from register_embryos.contrast import ContrastLimits, preview_contrast
    from register_embryos.orientation import Orientation, OrientationSet
    from register_embryos.widgets import PrepConfig

    volume = _asymmetric_volume()
    eid = volume.embryo_id
    config = PrepConfig(
        orientations=OrientationSet({eid: Orientation(xy_rotation=155.0, flip_x=True)}),
        contrast=ContrastLimits.from_dict({eid: {0: (0.05, 0.5), 1: (0.05, 0.5)}}),
    )
    title = preview_contrast(volume, config)._suptitle.get_text()
    assert "155" in title and "flip x" in title
    identity = preview_contrast(volume, config, orientation=Orientation())
    assert "identity" in identity._suptitle.get_text()


def test_preview_contrast_still_accepts_a_bare_contrast_limits():
    """The older positional call style must keep working."""
    import matplotlib
    matplotlib.use("Agg")
    from register_embryos.contrast import ContrastLimits, preview_contrast

    volume = _asymmetric_volume()
    limits = ContrastLimits.from_dict(
        {volume.embryo_id: {0: (0.05, 0.5), 1: (0.05, 0.5)}}
    )
    assert preview_contrast(volume, limits) is not None


def test_preview_contrast_warns_about_double_application(capsys):
    """Previewing an already-adjusted volume with limits would clip it twice."""
    import matplotlib
    matplotlib.use("Agg")
    from register_embryos.contrast import (
        ContrastLimits, apply_contrast_to_volumes, preview_contrast,
    )

    volume = _asymmetric_volume()
    limits = ContrastLimits.from_dict(
        {volume.embryo_id: {0: (0.05, 0.5), 1: (0.05, 0.5)}}
    )
    adjusted = apply_contrast_to_volumes([volume], limits, verbose=False)[0]
    preview_contrast(adjusted, limits)
    assert "double-clip" in capsys.readouterr().out


def test_preview_contrast_warns_about_double_rotation(capsys):
    import matplotlib
    matplotlib.use("Agg")
    from register_embryos.contrast import preview_contrast
    from register_embryos.orientation import Orientation, apply_orientation

    volume = _asymmetric_volume()
    oriented = apply_orientation(volume, Orientation(xy_rotation=90.0), verbose=False)
    preview_contrast(oriented, orientation=Orientation(xy_rotation=90.0))
    assert "compound the rotation" in capsys.readouterr().out


def test_preview_contrast_rejects_a_config_of_the_wrong_type():
    import matplotlib
    matplotlib.use("Agg")
    from register_embryos.contrast import preview_contrast

    with pytest.raises(TypeError, match="PrepConfig or ContrastLimits"):
        preview_contrast(_asymmetric_volume(), config={"nope": 1})


def test_degenerate_percentiles_do_not_break_the_preview():
    """A near-uniform frame collapses p90 and p99.9 onto the same value.

    Happens on a nearly empty channel or a z-bin outside the sample. A QC function
    must render it, not raise.
    """
    import matplotlib
    matplotlib.use("Agg")
    from register_embryos.contrast import preview_contrast

    volume = _asymmetric_volume()
    # A channel that is entirely one value: every percentile is identical.
    volume.binned_channels[1] = np.full((3, 64, 64), 0.8, np.float32)
    assert preview_contrast(volume) is not None       # no contrast supplied


def test_widen_if_degenerate_behaviour():
    from register_embryos.contrast import _widen_if_degenerate

    varied = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    # A usable window is returned untouched.
    assert _widen_if_degenerate(0.1, 0.9, varied) == (0.1, 0.9)
    # Collapsed percentiles fall back to the array's own range.
    assert _widen_if_degenerate(0.5, 0.5, varied) == (0.0, 1.0)
    # A wholly uniform array still yields a nonzero width.
    flat = np.full((4, 4), 0.8, np.float32)
    lo, hi = _widen_if_degenerate(0.8, 0.8, flat)
    assert hi > lo


def test_auto_contrast_survives_an_empty_channel():
    """An all-zero channel must not abort a cohort's automatic contrast."""
    volume = _volume_with({0: _hcr_like_channel(), 1: np.zeros((8, 160, 160), np.float32)})
    limits = auto_contrast_limits([volume], verbose=False)
    lo, hi = limits.get(volume.embryo_id, 1)
    assert hi > lo


# ---------------------------------------------------------------------------
# Reloading a saved configuration
# ---------------------------------------------------------------------------

def test_prep_config_round_trips_through_the_cohort_directory(tmp_path):
    """The notebook's resume path: reopen a cohort and get the same settings back."""
    from register_embryos.contrast import ContrastLimits
    from register_embryos.orientation import Orientation, OrientationSet
    from register_embryos.widgets import PrepConfig

    cohort_dir = tmp_path / "wt_12s_dorsal_20X"
    original = PrepConfig(
        orientations=OrientationSet({"e0": Orientation(xy_rotation=155.0, flip_x=True)}),
        contrast=ContrastLimits.from_dict({"e0": {0: (0.05, 0.4)}}, transform="log2"),
        orientation_path=cohort_dir / "orientation.json",
        contrast_path=cohort_dir / "contrast_limits.json",
    )
    original.save()

    back = PrepConfig.load(cohort_dir)
    assert back.orientations.get("e0").xy_rotation == 155.0
    assert back.orientations.get("e0").flip_x is True
    assert back.contrast["e0"][0] == (0.05, 0.4)
    assert back.contrast.transform == "log2"      # or limits get re-applied wrongly
    # Paths are set, so a reloaded config keeps saving to the same place.
    assert back.orientation_path == cohort_dir / "orientation.json"


def test_prep_config_load_on_a_fresh_directory_is_empty_not_an_error(tmp_path):
    """The resume cell must be safe to run before anything has been set."""
    from register_embryos.widgets import PrepConfig

    config = PrepConfig.load(tmp_path / "nothing_here")
    assert len(config.orientations) == 0
    assert len(config.contrast) == 0


def test_a_reloaded_config_applies_to_volumes(tmp_path):
    """End of the resume path: reloaded settings must actually drive apply_prep."""
    from register_embryos.contrast import ContrastLimits, apply_contrast_to_volumes
    from register_embryos.orientation import (
        Orientation, OrientationSet, apply_orientation_to_volumes,
    )
    from register_embryos.widgets import PrepConfig

    volume = _asymmetric_volume()
    eid = volume.embryo_id
    cohort_dir = tmp_path / "cohort"
    PrepConfig(
        orientations=OrientationSet({eid: Orientation(xy_rotation=90.0)}),
        contrast=ContrastLimits.from_dict({eid: {0: (0.05, 0.5), 1: (0.05, 0.5)}}),
        orientation_path=cohort_dir / "orientation.json",
        contrast_path=cohort_dir / "contrast_limits.json",
    ).save()

    reloaded = PrepConfig.load(cohort_dir)
    oriented = apply_orientation_to_volumes([volume], reloaded.orientations, verbose=False)
    adjusted = apply_contrast_to_volumes(oriented, reloaded.contrast, verbose=False)

    assert any("oriented" in note for note in adjusted[0].history)
    assert any("contrast applied" in note for note in adjusted[0].history)
    for array in adjusted[0].binned_channels.values():
        assert 0.0 <= array.min() and array.max() <= 1.0


# ---------------------------------------------------------------------------
# 2d vs 2d+link vs 3d: what each actually produces
# ---------------------------------------------------------------------------

def _spanning_masks():
    """One nucleus present on all 3 planes, plus one present on a single plane.

    In a genuine 2D run the first would carry an unrelated label on each plane; here
    the labels are already globally consistent, which is what a linking pass (or a
    3D pass) produces.
    """
    masks = np.zeros((3, 20, 20), dtype=int)
    masks[:, 4:10, 4:10] = 1        # spans z = 0, 1, 2
    masks[1, 14:18, 14:18] = 2      # z = 1 only
    return masks


def _segmented(mode, masks=None, gene_value=0.0):
    """A SegmentedEmbryo with controllable gene signal.

    ``gene_value=0`` leaves the signal mask empty, so no pixels are reassigned and
    each nucleus keeps exactly its mask voxels.  That isolates the *reduction* --
    how label volumes become table rows -- from the assignment step, which is what
    these tests are about.  A nonzero value blankets the frame with signal and
    every nucleus's territory expands to fill it.
    """
    from register_embryos.imaging import EmbryoVolume, VoxelSize
    from register_embryos.naming import parse_embryo_name
    from register_embryos.segmentation import SegmentedEmbryo

    gene = np.full((3, 20, 20), gene_value, np.float32)
    volume = EmbryoVolume(
        name=parse_embryo_name("20260410_1.5_wt_12s_dorsal_20X_hand2_tbx1_wt1a.nd2"),
        binned_channels={0: np.ones((3, 20, 20), np.float32), 1: gene,
                         2: gene.copy(), 3: gene.copy()},
        voxel=VoxelSize(0.863, 1.5), bin_size=7, c_size=4, z_size=21,
    )
    return SegmentedEmbryo(
        volume=volume,
        nuclear_masks=_spanning_masks() if masks is None else masks,
        mode=mode,
    )


def test_labels_are_3d_is_not_the_same_question_as_is_3d():
    """2d+link has z-consistent labels without Cellpose having run in 3D."""
    assert not _segmented("2d").labels_are_3d
    assert not _segmented("2d").is_3d

    linked = _segmented("2d+link")
    assert linked.labels_are_3d          # ids mean one object across planes
    assert not linked.is_3d              # but the masks came from 2D passes

    volumetric = _segmented("3d")
    assert volumetric.labels_are_3d and volumetric.is_3d


def test_plain_2d_emits_one_row_per_plane():
    """Per-slice labels: a spanning nucleus is several rows with unrelated ids."""
    from register_embryos.assignment import build_nucleus_table

    df = build_nucleus_table(_segmented("2d"), save=False, verbose=False).nucleus_df
    assert len(df) == 4                                   # 3 planes + 1
    assert sorted(df[df["nucleus_id"] == 1]["z"]) == [0.0, 1.0, 2.0]


def test_linking_collapses_a_spanning_nucleus_to_one_row():
    """Regression: 2d+link produced a table identical to plain 2d.

    The linking pass made ids z-consistent, but the reduction keyed off is_3d
    (False for 2d+link) and so still went per-plane -- discarding the linking at
    exactly the step meant to benefit from it, and leaving the duplicate centroids
    that 2d+link exists to remove.
    """
    from register_embryos.assignment import build_nucleus_table

    plain = build_nucleus_table(_segmented("2d"), save=False, verbose=False).nucleus_df
    linked = build_nucleus_table(_segmented("2d+link"), save=False, verbose=False).nucleus_df

    assert len(linked) < len(plain)
    assert len(linked) == 2                       # one row per object
    assert linked["nucleus_id"].is_unique
    # A true 3D centroid: the mean of planes 0, 1 and 2.
    spanning = linked[linked["nucleus_id"] == 1].iloc[0]
    assert spanning["z"] == pytest.approx(1.0)
    # It accounts for every voxel of the object, not just one plane's worth.
    # (gene_value=0 so nothing is reassigned; the count is the mask itself.)
    assert spanning["n_voxels"] == 3 * 36


def test_linked_and_3d_reductions_agree_on_the_same_masks():
    """With no signal to reassign, the two z-consistent modes reduce identically.

    They still differ in general, because 2d+link assigns signal pixels within
    each plane while 3d assigns them through the volume in micrometres -- so this
    holds only once assignment is taken out of the picture.
    """
    from register_embryos.assignment import build_nucleus_table

    linked = build_nucleus_table(
        _segmented("2d+link"), save=False, verbose=False).nucleus_df
    volumetric = build_nucleus_table(
        _segmented("3d"), save=False, verbose=False).nucleus_df

    assert len(linked) == len(volumetric)
    for column in ("nucleus_id", "x", "y", "z", "n_voxels"):
        assert linked[column].tolist() == pytest.approx(volumetric[column].tolist())


def test_link_and_3d_assignment_differ_when_there_is_signal_to_spread():
    """The remaining real difference: per-plane vs through-volume assignment.

    2d+link keeps 2D assignment (each plane's signal joins a nucleus on that
    plane); 3d measures distance through the stack in micrometres. With signal
    everywhere the resulting territories are genuinely different, and neither is
    a bug.
    """
    from register_embryos.assignment import build_nucleus_table

    linked = build_nucleus_table(
        _segmented("2d+link", gene_value=0.6), save=False, verbose=False).nucleus_df
    volumetric = build_nucleus_table(
        _segmented("3d", gene_value=0.6), save=False, verbose=False).nucleus_df

    assert len(linked) == len(volumetric) == 2          # same objects
    assert linked["n_voxels"].tolist() != volumetric["n_voxels"].tolist()


def test_the_recorded_mode_distinguishes_the_three():
    """The EmbryoResult must say which mode produced it, for provenance."""
    from register_embryos.assignment import build_nucleus_table

    for mode in ("2d", "2d+link", "3d"):
        result = build_nucleus_table(_segmented(mode), save=False, verbose=False)
        assert result.mode == mode


# ---------------------------------------------------------------------------
# Constrained registration: a near-circular cloud cannot pin down its own angle
# ---------------------------------------------------------------------------

def _disc_cloud(n=1500, seed=0):
    """A flat, nearly circular cloud, like a 12-somite dorsal embryo.

    Real cohort geometry: principal extents ~180 x 155 x 4, an in-plane aspect of
    only 1.18. That near-symmetry is why nucleus positions cannot fix the
    anterior-posterior angle.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, 3)) * np.array([180.0, 155.0, 4.0])


def _z_rotation(degrees):
    theta = np.radians(degrees)
    matrix = np.eye(4)
    matrix[:3, :3] = [[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]]
    return matrix


def test_rotation_angles_decomposes_a_transform():
    from register_embryos.registration import rotation_angles

    in_plane, tilt = rotation_angles(_z_rotation(172.4))
    assert in_plane == pytest.approx(172.4, abs=0.1)
    assert tilt == pytest.approx(0.0, abs=1e-6)


def test_nn_residual_cannot_tell_a_180_degree_flip_from_a_correct_fit():
    """The reason a residual-driven fit goes wrong on this geometry.

    A near-circular disc looks almost the same rotated end for end, so optimising
    nearest-neighbour distance gives no reason to prefer the correct orientation.
    """
    from scipy.spatial import cKDTree

    cloud = _disc_cloud()
    tree = cKDTree(cloud)
    upright = tree.query(cloud)[0].mean()
    flipped = tree.query(_apply(cloud, _z_rotation(180.0)))[0].mean()
    # Within a factor of two -- nowhere near enough to identify the right pose.
    assert flipped < upright + 12.0


def _apply(points, transform):
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def test_max_rotation_caps_the_in_plane_correction():
    from register_embryos.registration import icp_point_to_point, rotation_angles

    target = _disc_cloud(seed=1)
    source = _apply(target, _z_rotation(170.0))     # grossly mis-oriented input

    _, unconstrained = icp_point_to_point(
        source, target, max_correspondence_distance=500, max_iteration=100
    )
    _, capped = icp_point_to_point(
        source, target, max_correspondence_distance=500, max_iteration=100,
        pca_init=False, inplane_only=True, max_rotation_deg=30.0,
    )
    assert abs(rotation_angles(capped)[0]) <= 30.0 + 1e-6
    # The cap is doing something: unconstrained is free to swing much further.
    assert abs(rotation_angles(capped)[0]) < abs(rotation_angles(unconstrained)[0]) + 1e-6


def test_inplane_only_removes_out_of_plane_tilt():
    """A cloud ~4 units thick against ~180 wide cannot constrain tilt."""
    from register_embryos.registration import icp_point_to_point, rotation_angles

    target = _disc_cloud(seed=2)
    source = _apply(target, _z_rotation(15.0)) + np.array([30.0, -20.0, 1.0])
    _, transform = icp_point_to_point(
        source, target, max_correspondence_distance=500, pca_init=False,
        inplane_only=True,
    )
    assert rotation_angles(transform)[1] == pytest.approx(0.0, abs=1e-6)


def test_constrained_fit_still_recovers_a_small_misalignment():
    """Capping rotation must not stop ICP doing its actual job."""
    from scipy.spatial import cKDTree

    from register_embryos.registration import icp_point_to_point

    target = _disc_cloud(seed=3)
    source = _apply(target, _z_rotation(12.0)) + np.array([40.0, 25.0, 2.0])
    registered, _ = icp_point_to_point(
        source, target, max_correspondence_distance=500, pca_init=False,
        inplane_only=True, max_rotation_deg=30.0,
    )
    tree = cKDTree(target)
    assert tree.query(registered)[0].mean() < tree.query(source)[0].mean() * 0.5


def test_workflow_register_trusts_a_manual_orientation_by_default(tmp_path):
    """If orientations were recorded, ICP must not re-derive them from the cloud."""
    from register_embryos.naming import CohortKey, parse_embryo_name
    from register_embryos.orientation import Orientation, OrientationSet
    from register_embryos.registration import rotation_angles
    from register_embryos.workflow import CohortWorkflow

    embryo = parse_embryo_name("20260410_1.5_wt_12s_dorsal_20X_a_b_c.nd2")
    wf = CohortWorkflow(CohortKey("wt", "12s", "dorsal", "20X"), [embryo], tmp_path)

    target = _disc_cloud(seed=4)
    source = _apply(target, _z_rotation(170.0))
    frames = []
    for name, cloud in (("ref", target), ("mov", source)):
        df = pd.DataFrame(cloud, columns=["x", "y", "z"])
        df["embryo_id"] = name
        df["geneA"] = 0.5
        frames.append(df)
    wf.combined = pd.concat(frames, ignore_index=True)
    wf.orientations = OrientationSet({"mov": Orientation(xy_rotation=170.0)})

    result = wf.register(reference_embryo_id="ref", n_downsample=None, verbose=False)
    in_plane, tilt = rotation_angles(result.transform_of("mov"))
    assert abs(in_plane) <= 30.0 + 1e-6      # capped, not flipped
    assert tilt == pytest.approx(0.0, abs=1e-6)


def test_trust_orientation_can_be_turned_off():
    from register_embryos.registration import register_frames, rotation_angles

    target = _disc_cloud(seed=5)
    source = _apply(target, _z_rotation(170.0))
    frames = {
        "ref": pd.DataFrame(target, columns=["x", "y", "z"]).assign(embryo_id="ref"),
        "mov": pd.DataFrame(source, columns=["x", "y", "z"]).assign(embryo_id="mov"),
    }
    result = register_frames(frames, reference_embryo_id="ref", n_downsample=None,
                            verbose=False)   # unconstrained, the old behaviour
    assert abs(rotation_angles(result.transform_of("mov"))[0]) > 30.0


# ---------------------------------------------------------------------------
# Optimal-transport refinement
# ---------------------------------------------------------------------------

def test_sinkhorn_plan_is_mass_balanced():
    from register_embryos.registration import sinkhorn_plan

    cost = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]])
    plan = sinkhorn_plan(cost, epsilon=0.5)
    assert plan.sum() == pytest.approx(1.0)
    assert plan.sum(axis=1) == pytest.approx(np.full(3, 1 / 3))
    assert plan.sum(axis=0) == pytest.approx(np.full(3, 1 / 3))
    # Mass concentrates on the cheap pairs.
    assert np.argmax(plan, axis=1).tolist() == [0, 1, 2]


def test_sinkhorn_survives_a_small_epsilon():
    """The direct form underflows here; the log-domain one must not."""
    from register_embryos.registration import sinkhorn_plan

    rng = np.random.default_rng(0)
    cost = ((rng.normal(0, 50, (60, 3))[:, None, :]
             - rng.normal(0, 50, (60, 3))[None, :, :]) ** 2).sum(-1)
    plan = sinkhorn_plan(cost, epsilon=1e-2, n_iter=50)
    assert np.isfinite(plan).all()
    assert plan.sum() == pytest.approx(1.0, abs=1e-6)


def test_sinkhorn_honours_supplied_marginals():
    from register_embryos.registration import sinkhorn_plan

    cost = np.abs(np.subtract.outer(np.arange(4.0), np.arange(3.0)))
    a = np.array([0.4, 0.3, 0.2, 0.1])
    b = np.array([0.5, 0.3, 0.2])
    plan = sinkhorn_plan(cost, epsilon=0.3, weights_source=a, weights_target=b)
    assert plan.sum(axis=1) == pytest.approx(a, abs=1e-6)
    assert plan.sum(axis=0) == pytest.approx(b, abs=1e-6)


def test_ot_refine_returns_a_rigid_transform():
    """Rigid by construction: a non-rigid warp would manufacture agreement."""
    from register_embryos.registration import ot_refine

    target = _disc_cloud(seed=6)
    source = _apply(target, _z_rotation(3.0)) + np.array([12.0, -8.0, 0.5])
    _, transform = ot_refine(source, target, max_points=400, verbose=False)
    R = transform[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)     # orthogonal
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)   # proper, not a reflection


def test_ot_refine_improves_a_small_residual_misalignment():
    from scipy.spatial import cKDTree

    from register_embryos.registration import ot_refine

    target = _disc_cloud(seed=7)
    source = _apply(target, _z_rotation(2.0)) + np.array([9.0, 6.0, 0.4])
    refined, _ = ot_refine(source, target, max_points=600, verbose=False)
    tree = cKDTree(target)
    assert tree.query(refined)[0].mean() < tree.query(source)[0].mean()


def test_ot_refine_respects_the_rotation_cap():
    """It is a refinement, not a second chance to re-orient."""
    from register_embryos.registration import ot_refine, rotation_angles

    target = _disc_cloud(seed=8)
    source = _apply(target, _z_rotation(60.0))
    _, transform = ot_refine(source, target, max_points=400,
                             max_rotation_deg=5.0, inplane_only=True, verbose=False)
    in_plane, tilt = rotation_angles(transform)
    assert abs(in_plane) <= 5.0 + 1e-6
    assert tilt == pytest.approx(0.0, abs=1e-6)


def test_register_frames_can_run_the_ot_stage():
    from register_embryos.registration import register_frames

    target = _disc_cloud(seed=9)
    source = _apply(target, _z_rotation(4.0)) + np.array([15.0, 10.0, 0.5])
    frames = {
        "ref": pd.DataFrame(target, columns=["x", "y", "z"]).assign(embryo_id="ref"),
        "mov": pd.DataFrame(source, columns=["x", "y", "z"]).assign(embryo_id="mov"),
    }
    plain = register_frames(frames, reference_embryo_id="ref", n_downsample=None,
                            pca_init=False, verbose=False)
    with_ot = register_frames(frames, reference_embryo_id="ref", n_downsample=None,
                              pca_init=False, refine_with_ot=True,
                              ot_kwargs={"max_points": 400}, verbose=False)
    assert not np.allclose(with_ot.transform_of("mov"), plain.transform_of("mov"))
    assert with_ot.stats["mean_after"].iloc[0] < with_ot.stats["mean_before"].iloc[0]


# ---------------------------------------------------------------------------
# Orientation consistency QC
# ---------------------------------------------------------------------------

def test_orientation_consistency_flags_an_off_axis_embryo():
    from register_embryos.orientation import orientation_consistency

    rng = np.random.default_rng(3)
    frames = []
    for name, degrees in (("a", 0.0), ("b", 3.0), ("c", 45.0)):
        # Elongated so the principal axis is actually determined.
        cloud = rng.normal(0, 1, (800, 3)) * np.array([200.0, 60.0, 4.0])
        cloud = _apply(cloud, _z_rotation(degrees))
        df = pd.DataFrame(cloud, columns=["x", "y", "z"])
        df["embryo_id"] = name
        frames.append(df)
    table = pd.concat(frames, ignore_index=True)

    out = orientation_consistency(table, verbose=False)
    deviations = dict(zip(out["embryo_id"], out["deviation_deg"].abs()))
    assert deviations["c"] > deviations["a"]
    assert deviations["c"] > 20            # the off-axis one stands out
    assert (out["elongation"] > 2.0).all()  # axis is meaningful for these


def test_orientation_consistency_reports_low_elongation_as_unreliable(capsys):
    """A near-circular cloud cannot settle orientation; say so rather than imply it can."""
    from register_embryos.orientation import orientation_consistency

    rng = np.random.default_rng(4)
    frames = []
    for name in ("a", "b"):
        cloud = rng.normal(0, 1, (600, 3)) * np.array([180.0, 155.0, 4.0])
        df = pd.DataFrame(cloud, columns=["x", "y", "z"])
        df["embryo_id"] = name
        frames.append(df)
    out = orientation_consistency(pd.concat(frames, ignore_index=True), verbose=True)
    assert (out["elongation"] < 1.3).all()
    assert "cannot settle orientation" in capsys.readouterr().out


def test_ot_similarity_absorbs_a_size_difference_within_its_cap():
    """The bounded loosening: one global scale, clamped, no local warp."""
    from register_embryos.registration import ot_refine

    target = _disc_cloud(seed=10)
    source = target * 1.08 + np.array([10.0, 5.0, 0.3])     # 8% larger

    _, rigid = ot_refine(source, target, max_points=500,
                         transform_model="rigid", verbose=False)
    _, similar = ot_refine(source, target, max_points=500,
                           transform_model="similarity", max_scale=1.15, verbose=False)
    rigid_scale = np.linalg.svd(rigid[:3, :3], compute_uv=False)
    similar_scale = np.linalg.svd(similar[:3, :3], compute_uv=False)

    assert rigid_scale.max() == pytest.approx(1.0, abs=1e-6)   # rigid cannot scale
    assert similar_scale.max() < 1.0                            # shrinks toward target
    # One uniform scale: every singular value the same.
    assert similar_scale.max() - similar_scale.min() < 1e-6


def test_ot_scale_is_clamped():
    from register_embryos.registration import ot_refine

    target = _disc_cloud(seed=11)
    source = target * 2.0            # far outside any sane bound
    _, transform = ot_refine(source, target, max_points=500,
                             transform_model="similarity", max_scale=1.05,
                             verbose=False)
    scales = np.linalg.svd(transform[:3, :3], compute_uv=False)
    assert scales.max() <= 1.05 + 1e-6
    assert scales.min() >= 1 / 1.05 - 1e-6


def test_ot_affine_singular_values_are_bounded():
    from register_embryos.registration import ot_refine

    target = _disc_cloud(seed=12)
    source = target * np.array([1.4, 0.7, 1.0])      # anisotropic distortion
    _, transform = ot_refine(source, target, max_points=500,
                             transform_model="affine", max_scale=1.10, verbose=False)
    scales = np.linalg.svd(transform[:3, :3], compute_uv=False)
    assert scales.max() <= 1.10 + 1e-6
    assert scales.min() >= 1 / 1.10 - 1e-6
    assert np.linalg.det(transform[:3, :3]) > 0      # no reflection


def test_ot_rejects_an_unknown_transform_model():
    from register_embryos.registration import ot_refine

    with pytest.raises(ValueError, match="unknown transform_model"):
        ot_refine(_disc_cloud(seed=13), _disc_cloud(seed=14),
                  max_points=200, transform_model="elastic", verbose=False)


def _units_cohort():
    rng = np.random.default_rng(15)
    frames = {}
    for name, shift in (("ref", 0.0), ("mov", 3.0)):
        cloud = rng.normal(0, 1, (900, 3)) * np.array([180.0, 155.0, 4.0])
        df = pd.DataFrame(cloud, columns=["x", "y", "z"])
        df["z"] += shift                       # a pure z offset, in bins
        df["embryo_id"] = name
        df["x_um"] = df["x"] * 0.863
        df["y_um"] = df["y"] * 0.863
        df["z_um"] = df["z"] * 10.5            # bin thickness
        frames[name] = df
    return frames


def test_coord_cols_selects_which_columns_are_registered():
    """Registering in physical units is a different problem, not a rescaled one.

    The default ("x", "y", "z") mixes units -- xy in pixels, z in bin indices -- so on
    this project's geometry z contributes ~1.5% of the cost and ICP nearly ignores it;
    in micrometres it is ~16%. That changes the fit. It does NOT follow that the
    micrometre fit is better, and on this fixture it is not (it recovers the z offset
    slightly less well), so this test pins the mechanism only.
    """
    from register_embryos.registration import register_frames

    frames = _units_cohort()
    in_px = register_frames(frames, reference_embryo_id="ref", n_downsample=None,
                            pca_init=False, verbose=False)
    in_um = register_frames(frames, reference_embryo_id="ref", n_downsample=None,
                            pca_init=False, coord_cols=("x_um", "y_um", "z_um"),
                            verbose=False)
    # Different cost function -> different transform.
    assert not np.allclose(in_px.transform_of("mov"), in_um.transform_of("mov"))
    # Both still recover most of the 3-bin z offset, in their own units.
    assert abs(in_px.transform_of("mov")[2, 3]) > 1.5
    assert abs(in_um.transform_of("mov")[2, 3]) / 10.5 > 1.5
    # Output columns are always x_reg/y_reg/z_reg, in whatever units were registered.
    assert {"x_reg", "y_reg", "z_reg"} <= set(in_um.registered.columns)


def test_z_share_of_the_cost_differs_by_the_anisotropy():
    """The measurable claim: what fraction of the coordinate range z accounts for."""
    frames = _units_cohort()
    df = frames["ref"]
    spans_px = np.array([df[c].max() - df[c].min() for c in ("x", "y", "z")])
    spans_um = np.array([df[c].max() - df[c].min() for c in ("x_um", "y_um", "z_um")])
    share_px = spans_px[2] / spans_px.sum()
    share_um = spans_um[2] / spans_um.sum()
    assert share_px < 0.05          # z is negligible in mixed units
    assert share_um > 0.10          # and material in physical ones
    assert share_um / share_px > 5


def test_union_signal_mask_couples_the_gene_channels():
    """One gene's contrast changes another gene's measured value, under "union".

    Observed on real data: raising the hand2 and wt1a floors moved tbx1 from 20.4%
    to 24.2% positive with tbx1's own window untouched. "per_channel" decouples them.
    """
    from register_embryos.assignment import BACKGROUND_VALUE, _apply_background

    # geneA is bright in the left half; geneB is background everywhere.
    bright = np.zeros((1, 4, 4), np.float32)
    bright[0, :, :2] = 0.8
    background = np.full((1, 4, 4), 0.02, np.float32)
    channels = {0: np.ones((1, 4, 4), np.float32), 1: bright, 2: background}

    from register_embryos.assignment import build_signal_mask

    mask = build_signal_mask(channels, signal_threshold=0.05, verbose=False)

    union = _apply_background(channels, mask, mode="union")
    per_channel = _apply_background(channels, mask, mode="per_channel",
                                    signal_threshold=0.05)

    # Under union, geneB's background is retained wherever geneA was bright, so it
    # counts as a measurement.
    assert (union[2][0, :, :2] != BACKGROUND_VALUE).all()
    # Under per_channel it is not, because geneB never cleared its own threshold.
    assert (per_channel[2] == BACKGROUND_VALUE).all()
    # geneA is unaffected either way.
    assert (union[1][0, :, :2] == per_channel[1][0, :, :2]).all()


def test_per_channel_mode_leaves_the_nuclear_channel_on_the_union():
    from register_embryos.assignment import BACKGROUND_VALUE, _apply_background

    channels = {0: np.full((1, 3, 3), 0.9, np.float32),
                1: np.full((1, 3, 3), 0.02, np.float32)}
    mask = np.ones((1, 3, 3), dtype=bool)
    out = _apply_background(channels, mask, mode="per_channel", signal_threshold=0.05)
    assert (out[0] != BACKGROUND_VALUE).all()      # channel 0 follows the mask
    assert (out[1] == BACKGROUND_VALUE).all()      # the dim gene does not


def test_unknown_signal_mask_mode_is_rejected():
    from register_embryos.assignment import _apply_background

    with pytest.raises(ValueError, match="unknown signal_mask_mode"):
        _apply_background({1: np.zeros((1, 2, 2), np.float32)},
                          np.ones((1, 2, 2), dtype=bool), mode="intersection")


def test_reload_segmentation_uses_masks_from_disk(tmp_path):
    """The fast path: new gene contrast, same masks, no Cellpose."""
    from register_embryos.naming import CohortKey, parse_embryo_name
    from register_embryos.workflow import CohortWorkflow

    volume = _asymmetric_volume()
    cohort = CohortKey("wt", "12s", "dorsal", "20X")
    wf = CohortWorkflow(cohort, [volume.name], tmp_path)
    wf.volumes = [volume]
    wf.adjusted = [volume]

    embryo_dir = wf.output_dir / "embryos" / volume.embryo_id
    embryo_dir.mkdir(parents=True, exist_ok=True)
    masks = np.zeros((3, 64, 64), dtype=int)
    masks[:, 10:20, 10:20] = 1
    np.save(embryo_dir / f"{volume.embryo_id}_nuclear_masks.npy", masks)

    segmented = wf.reload_segmentation(verbose=False)
    assert len(segmented) == 1
    assert np.array_equal(segmented[0].nuclear_masks, masks)
    assert segmented[0].params["reloaded"] is True
    assert wf.segmented == segmented


def test_reload_segmentation_reports_missing_masks(tmp_path):
    from register_embryos.naming import CohortKey
    from register_embryos.workflow import CohortWorkflow

    volume = _asymmetric_volume()
    wf = CohortWorkflow(CohortKey("wt", "12s", "dorsal", "20X"), [volume.name], tmp_path)
    wf.volumes = [volume]
    wf.adjusted = [volume]
    with pytest.raises(FileNotFoundError, match="no saved masks"):
        wf.reload_segmentation(verbose=False)


def test_reload_segmentation_refuses_mismatched_masks(tmp_path):
    """A different bin_size or a resized canvas makes the masks describe another image."""
    from register_embryos.naming import CohortKey
    from register_embryos.workflow import CohortWorkflow

    volume = _asymmetric_volume()                      # (3, 64, 64)
    wf = CohortWorkflow(CohortKey("wt", "12s", "dorsal", "20X"), [volume.name], tmp_path)
    wf.volumes = [volume]
    wf.adjusted = [volume]
    embryo_dir = wf.output_dir / "embryos" / volume.embryo_id
    embryo_dir.mkdir(parents=True, exist_ok=True)
    np.save(embryo_dir / f"{volume.embryo_id}_nuclear_masks.npy",
            np.zeros((3, 32, 32), dtype=int))          # wrong shape
    with pytest.raises(ValueError, match="cannot be reused"):
        wf.reload_segmentation(verbose=False)
