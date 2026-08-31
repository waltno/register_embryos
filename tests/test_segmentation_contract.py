"""What we pass to Cellpose, asserted without paying for Cellpose.

3D CPSAM on CPU is minutes per z-bin, so these tests substitute a fake model that
records its ``eval`` kwargs.  That keeps the suite fast while still pinning the
call contract -- which is where the real bugs were:

* Cellpose 4 raises ``z_axis must be specified when segmenting 3D images of
  ndim=3`` for a bare ``(Z, Y, X)`` array, so every 3D call failed until
  ``z_axis=0`` was passed.  It applies to the stitching path too, because Cellpose
  routes that through the same 3D image conversion.
* Cellpose 4 has no ``"nuclei"`` checkpoint; asking for one prints a single
  easily-missed line and silently substitutes CPSAM.
"""

import numpy as np
import pytest

from register_embryos import segmentation


class FakeModel:
    """Stands in for a Cellpose model; records how it was called."""

    def __init__(self):
        self.calls = []

    def eval(self, image, **kwargs):
        self.calls.append({"ndim": np.asarray(image).ndim, **kwargs})
        image = np.asarray(image)
        if image.ndim == 3:                      # 3D: one label filling the volume
            masks = np.ones(image.shape, dtype=int)
        else:                                    # 2D: one label per frame
            masks = np.ones(image.shape, dtype=int)
        return masks, None, None


@pytest.fixture
def fake(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(segmentation, "_load_model", lambda gpu, pretrained_model: model)
    return model


@pytest.fixture
def volume():
    return np.random.default_rng(0).random((4, 32, 32)).astype(np.float32)


def test_3d_passes_z_axis_or_cellpose_4_rejects_the_volume(fake, volume):
    """The regression: without z_axis=0 every 3D call raised."""
    segmentation.segment_3d(volume, anisotropy=3.0, verbose=False)
    call = fake.calls[0]
    assert call["ndim"] == 3
    assert call["z_axis"] == 0
    assert call["channel_axis"] is None
    assert call["do_3D"] is True
    assert call["anisotropy"] == 3.0


def test_the_stitching_path_also_needs_z_axis(fake, volume):
    """Cellpose routes stitching through the same 3D conversion, so it needs it too."""
    segmentation.segment_3d(volume, anisotropy=3.0, stitch_threshold=0.3, verbose=False)
    call = fake.calls[0]
    assert call["z_axis"] == 0
    assert call["channel_axis"] is None
    assert call["stitch_threshold"] == 0.3
    assert call["do_3D"] is False          # stitching instead of the 3D flow field


def test_2d_is_called_per_z_bin_on_single_channel_frames(fake, volume):
    segmentation.segment_2d(volume, verbose=False)
    assert len(fake.calls) == volume.shape[0]      # one call per z-bin
    assert all(call["ndim"] == 2 for call in fake.calls)
    assert all(call["channel_axis"] is None for call in fake.calls)
    assert all("do_3D" not in call for call in fake.calls)


def test_diameter_and_anisotropy_are_forwarded_independently(fake, volume):
    """Different knobs: diameter is xy pixels, anisotropy is the z:xy ratio."""
    segmentation.segment_3d(volume, anisotropy=7.5, diameter=23.0, verbose=False)
    call = fake.calls[0]
    assert call["diameter"] == 23.0
    assert call["anisotropy"] == 7.5


def test_2d_does_not_receive_anisotropy(fake, volume):
    """Anisotropy is meaningless per-slice; passing it would be a silent no-op."""
    segmentation.segment_2d(volume, diameter=15.0, verbose=False)
    assert all("anisotropy" not in call for call in fake.calls)


def test_segment_embryo_takes_anisotropy_from_the_binned_voxel_size(monkeypatch):
    """The binning factor must be folded in, or 3D z-linking is wrong by that factor."""
    from register_embryos.imaging import EmbryoVolume, VoxelSize
    from register_embryos.naming import parse_embryo_name

    model = FakeModel()
    monkeypatch.setattr(segmentation, "_load_model", lambda gpu, pretrained_model: model)

    name = parse_embryo_name("20260410_1.5_wt_12s_dorsal_20X_hand2_tbx5a_wt1a.nd2")
    vol = EmbryoVolume(
        name=name,
        binned_channels={0: np.zeros((4, 16, 16), np.float32)},
        voxel=VoxelSize(xy_um=0.5, z_um=1.0),      # raw anisotropy 2.0
        bin_size=7,                                 # binned anisotropy 14.0
        c_size=1, z_size=28,
    )
    result = segmentation.segment_embryo(vol, mode="3d", verbose=False)
    assert model.calls[0]["anisotropy"] == pytest.approx(14.0)
    assert result.params["anisotropy"] == pytest.approx(14.0)
    assert result.is_3d


def test_explicit_anisotropy_overrides_the_metadata(monkeypatch):
    from register_embryos.imaging import EmbryoVolume, VoxelSize
    from register_embryos.naming import parse_embryo_name

    model = FakeModel()
    monkeypatch.setattr(segmentation, "_load_model", lambda gpu, pretrained_model: model)
    vol = EmbryoVolume(
        name=parse_embryo_name("20260410_1.5_wt_12s_dorsal_20X_a_b_c.nd2"),
        binned_channels={0: np.zeros((4, 16, 16), np.float32)},
        voxel=VoxelSize(0.5, 1.0), bin_size=7, c_size=1, z_size=28,
    )
    segmentation.segment_embryo(vol, mode="3d", anisotropy=2.0, verbose=False)
    assert model.calls[0]["anisotropy"] == pytest.approx(2.0)


def test_unknown_mode_is_rejected(monkeypatch):
    from register_embryos.imaging import EmbryoVolume, VoxelSize
    from register_embryos.naming import parse_embryo_name

    vol = EmbryoVolume(
        name=parse_embryo_name("20260410_1.5_wt_12s_dorsal_20X_a_b_c.nd2"),
        binned_channels={0: np.zeros((2, 8, 8), np.float32)},
        voxel=VoxelSize(1.0, 1.0), bin_size=1, c_size=1, z_size=2,
    )
    with pytest.raises(ValueError, match="unknown mode"):
        segmentation.segment_embryo(vol, mode="4d", verbose=False)


def test_missing_nuclei_channel_is_rejected(monkeypatch):
    from register_embryos.imaging import EmbryoVolume, VoxelSize
    from register_embryos.naming import parse_embryo_name

    vol = EmbryoVolume(
        name=parse_embryo_name("20260410_1.5_wt_12s_dorsal_20X_a_b_c.nd2"),
        binned_channels={1: np.zeros((2, 8, 8), np.float32)},
        voxel=VoxelSize(1.0, 1.0), bin_size=1, c_size=2, z_size=2,
    )
    with pytest.raises(ValueError, match="nuclei channel 0 not present"):
        segmentation.segment_embryo(vol, mode="2d", verbose=False)


def test_cellpose_major_version_is_detected():
    """A real check, not mocked: the model-selection branch depends on it."""
    assert segmentation.cellpose_major_version() >= 3


def test_asking_for_nuclei_weights_on_cellpose_4_warns(capsys, monkeypatch):
    """Cellpose 4 substitutes CPSAM silently; we must say so."""
    if segmentation.cellpose_major_version() < 4:
        pytest.skip("only applies to Cellpose 4+")

    class DummyModels:
        class CellposeModel:
            def __init__(self, gpu=False, pretrained_model=None):
                self.pretrained_model = pretrained_model

    import sys, types
    fake_cellpose = types.ModuleType("cellpose")
    fake_cellpose.models = DummyModels
    monkeypatch.setitem(sys.modules, "cellpose", fake_cellpose)
    monkeypatch.setitem(sys.modules, "cellpose.models", DummyModels)
    monkeypatch.setattr(segmentation, "_MODEL_NOTE_SHOWN", False)

    segmentation._load_model(gpu=False, pretrained_model="nuclei")
    out = capsys.readouterr().out
    assert "no 'nuclei' checkpoint" in out or "falls back to CPSAM" in out


def test_relabel_links_a_nucleus_across_slices():
    """Per-slice labels give one nucleus several ids; linking must collapse them."""
    masks = np.zeros((3, 10, 10), dtype=int)
    masks[0, 2:6, 2:6] = 1        # same object, restarting at label 1 each slice
    masks[1, 2:6, 2:6] = 1
    masks[2, 2:6, 2:6] = 1
    masks[0, 7:9, 7:9] = 2        # a separate object present on one slice only

    linked = segmentation.relabel_3d_from_2d(masks, iou_threshold=0.25)
    # The stacked object is one id through all three slices.
    assert linked[0, 3, 3] == linked[1, 3, 3] == linked[2, 3, 3]
    # And it is not confused with the other object.
    assert linked[0, 7, 7] != linked[0, 3, 3]
    assert len(np.unique(linked[linked > 0])) == 2


def test_relabel_keeps_non_overlapping_objects_separate():
    masks = np.zeros((2, 12, 12), dtype=int)
    masks[0, 1:4, 1:4] = 1
    masks[1, 8:11, 8:11] = 1      # same label, no overlap -> must become a new id
    linked = segmentation.relabel_3d_from_2d(masks, iou_threshold=0.25)
    assert linked[0, 2, 2] != linked[1, 9, 9]
