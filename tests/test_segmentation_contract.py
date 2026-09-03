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


def test_anisotropy_folds_in_the_binning_factor(monkeypatch):
    """Whatever the bin size, the anisotropy passed must describe the ACTUAL planes.

    Normal 3D runs are unbinned, so this only bites via allow_binned -- but if the
    factor were dropped there, Cellpose would be told the planes are 7x thinner
    than they are and would try to link nuclei across z that do not touch.
    """
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
    result = segmentation.segment_embryo(
        vol, mode="3d", allow_binned=True, verbose=False
    )
    assert model.calls[0]["anisotropy"] == pytest.approx(14.0)
    assert result.params["anisotropy"] == pytest.approx(14.0)
    assert result.is_3d


def test_unbinned_anisotropy_is_the_raw_voxel_ratio(monkeypatch):
    """The normal 3D case: bin_size=1, so binned and raw anisotropy agree."""
    from register_embryos.imaging import EmbryoVolume, VoxelSize
    from register_embryos.naming import parse_embryo_name

    model = FakeModel()
    monkeypatch.setattr(segmentation, "_load_model", lambda gpu, pretrained_model: model)
    vol = EmbryoVolume(
        name=parse_embryo_name("20260410_1.5_wt_12s_dorsal_20X_a_b_c.nd2"),
        binned_channels={0: np.zeros((28, 16, 16), np.float32)},
        voxel=VoxelSize(0.5, 1.0), bin_size=1, c_size=1, z_size=28,
    )
    segmentation.segment_embryo(vol, mode="3d", verbose=False)
    assert model.calls[0]["anisotropy"] == pytest.approx(2.0)


def test_explicit_anisotropy_overrides_the_metadata(monkeypatch):
    from register_embryos.imaging import EmbryoVolume, VoxelSize
    from register_embryos.naming import parse_embryo_name

    model = FakeModel()
    monkeypatch.setattr(segmentation, "_load_model", lambda gpu, pretrained_model: model)
    vol = EmbryoVolume(
        name=parse_embryo_name("20260410_1.5_wt_12s_dorsal_20X_a_b_c.nd2"),
        binned_channels={0: np.zeros((28, 16, 16), np.float32)},
        voxel=VoxelSize(0.5, 1.0), bin_size=1, c_size=1, z_size=28,
    )
    segmentation.segment_embryo(vol, mode="3d", anisotropy=3.3, verbose=False)
    assert model.calls[0]["anisotropy"] == pytest.approx(3.3)


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


# ---------------------------------------------------------------------------
# 3D takes the whole z-stack: binned input is refused
# ---------------------------------------------------------------------------

def _volume(bin_size, z_um=1.5, n_z=3):
    from register_embryos.imaging import EmbryoVolume, VoxelSize
    from register_embryos.naming import parse_embryo_name

    return EmbryoVolume(
        name=parse_embryo_name("20260410_1.5_wt_12s_dorsal_20X_a_b_c.nd2"),
        binned_channels={0: np.zeros((n_z, 16, 16), np.float32)},
        voxel=VoxelSize(xy_um=0.863, z_um=z_um),
        bin_size=bin_size, c_size=1, z_size=n_z * bin_size,
    )


def test_z_span_shrinks_with_coarser_binning():
    # 1.5 um z-step, ~6 um nucleus.
    assert segmentation.nucleus_z_span_bins(1.5) == pytest.approx(4.0)    # unbinned
    assert segmentation.nucleus_z_span_bins(3.0) == pytest.approx(2.0)    # bin_size 2
    assert segmentation.nucleus_z_span_bins(10.5) == pytest.approx(0.571, abs=1e-3)  # 7


def test_unbinned_is_accepted_for_3d():
    segmentation.require_unbinned_for_3d(bin_size=1, z_um=1.5)   # must not raise


def test_binned_input_is_refused_for_3d():
    """3D wants the whole stack; binned 3D is a slow 2D run."""
    with pytest.raises(ValueError) as excinfo:
        segmentation.require_unbinned_for_3d(bin_size=7, z_um=1.5, label="e1")
    message = str(excinfo.value)
    assert "unbinned z-stack" in message
    assert "bin_size=7" in message
    assert "0.57 planes" in message        # says why, concretely
    assert "load(bin_size=1)" in message   # says what to do


def test_the_refusal_can_be_overridden_with_a_warning(capsys):
    segmentation.require_unbinned_for_3d(
        bin_size=7, z_um=1.5, label="e1", allow_binned=True
    )
    assert "unbinned z-stack" in capsys.readouterr().out


def test_segment_embryo_refuses_3d_on_a_binned_volume(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(segmentation, "_load_model", lambda gpu, pretrained_model: model)
    with pytest.raises(ValueError, match="unbinned z-stack"):
        segmentation.segment_embryo(_volume(bin_size=7), mode="3d", verbose=False)
    assert model.calls == []               # refused before any Cellpose work


def test_segment_embryo_accepts_3d_on_an_unbinned_volume(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(segmentation, "_load_model", lambda gpu, pretrained_model: model)
    result = segmentation.segment_embryo(_volume(bin_size=1), mode="3d", verbose=False)
    assert result.is_3d
    # Unbinned, so the anisotropy passed is the raw voxel ratio.
    assert model.calls[0]["anisotropy"] == pytest.approx(1.5 / 0.863, abs=1e-3)


def test_2d_is_happy_with_a_binned_volume(monkeypatch):
    """Binning exists for 2D, so it must not inherit the 3D restriction."""
    model = FakeModel()
    monkeypatch.setattr(segmentation, "_load_model", lambda gpu, pretrained_model: model)
    result = segmentation.segment_embryo(_volume(bin_size=7), mode="2d", verbose=False)
    assert not result.is_3d
    assert len(model.calls) == 3           # one per z-bin


def test_memory_report_scales_with_channels(capsys):
    segmentation.report_3d_memory((200, 1024, 1024), n_channels=4)
    out = capsys.readouterr().out
    assert "0.84 GB per channel" in out
    assert "3.36 GB for 4 channels" in out


# ---------------------------------------------------------------------------
# mode-aware bin_size defaults
# ---------------------------------------------------------------------------

def test_bin_size_defaults_per_mode():
    from register_embryos.workflow import default_bin_size

    assert default_bin_size("2d") == 7
    assert default_bin_size("2d+link") == 7
    assert default_bin_size("3d") == 1       # the whole stack


def test_workflow_segment_refuses_3d_on_a_binned_cohort(monkeypatch, tmp_path):
    from register_embryos.naming import CohortKey, parse_embryo_name
    from register_embryos.workflow import CohortWorkflow

    embryo = parse_embryo_name("20260410_1.5_wt_12s_dorsal_20X_a_b_c.nd2")
    wf = CohortWorkflow(CohortKey("wt", "12s", "dorsal", "20X"), [embryo], tmp_path)
    wf.volumes = [_volume(bin_size=7)]
    with pytest.raises(ValueError, match="unbinned z-stack"):
        wf.segment(mode="3d", verbose=False)


# ---------------------------------------------------------------------------
# CPU accounting
# ---------------------------------------------------------------------------

def test_available_cpus_respects_the_process_affinity(monkeypatch):
    """os.cpu_count() reports the machine; a scheduler slot may allow far fewer.

    Regression: dividing the machine count among workers set each PyTorch worker
    to ~96 threads on a 2-core allocation.
    """
    import os as os_module

    monkeypatch.setattr(segmentation.os, "cpu_count", lambda: 192)
    monkeypatch.setattr(
        segmentation.os, "sched_getaffinity", lambda pid: {0, 1}, raising=False
    )
    assert segmentation.available_cpus() == 2


def test_available_cpus_is_never_zero(monkeypatch):
    monkeypatch.setattr(segmentation.os, "cpu_count", lambda: None)
    monkeypatch.setattr(
        segmentation.os, "sched_getaffinity", lambda pid: set(), raising=False
    )
    assert segmentation.available_cpus() >= 1


def test_available_cpus_matches_reality_here():
    assert 1 <= segmentation.available_cpus() <= (segmentation.os.cpu_count() or 1)


def test_workflow_segment_forwards_the_channel_to_cellpose(monkeypatch, tmp_path):
    """The regression: ``channel`` used to index the *volume list*, never Cellpose.

    So ``wf.segment(channel=1)`` silently segmented channel 0 anyway -- and on a
    one-embryo cohort ``channel=1`` was an IndexError rather than a wrong answer,
    which is the only reason it never produced quietly bad masks.
    """
    from register_embryos.naming import CohortKey, parse_embryo_name
    from register_embryos import workflow as workflow_module
    from register_embryos.workflow import CohortWorkflow

    seen = {}

    def fake_segment_cohort(volumes, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(workflow_module, "segment_cohort", fake_segment_cohort)

    embryo = parse_embryo_name("20260410_1.5_wt_12s_dorsal_20X_a_b_c.nd2")
    wf = CohortWorkflow(CohortKey("wt", "12s", "dorsal", "20X"), [embryo], tmp_path)
    wf.volumes = [_volume(bin_size=7)]

    wf.segment(mode="2d+link", channel=1, verbose=False)
    assert seen["nuclei_channel"] == 1
    assert wf._params["segmentation_channel"] == 1

    wf.segment(mode="2d+link", verbose=False)
    assert seen["nuclei_channel"] == 0


def test_the_binning_check_does_not_depend_on_the_channel(monkeypatch, tmp_path):
    """A single-embryo cohort must still reach the 3D refusal with channel=1."""
    from register_embryos.naming import CohortKey, parse_embryo_name
    from register_embryos.workflow import CohortWorkflow

    embryo = parse_embryo_name("20260410_1.5_wt_12s_dorsal_20X_a_b_c.nd2")
    wf = CohortWorkflow(CohortKey("wt", "12s", "dorsal", "20X"), [embryo], tmp_path)
    wf.volumes = [_volume(bin_size=7)]
    with pytest.raises(ValueError, match="unbinned z-stack"):
        wf.segment(mode="3d", channel=1, verbose=False)


def test_the_segmented_channel_is_recorded_in_the_sidecar(monkeypatch, tmp_path):
    """A channel-1 run writes masks under the same name as a channel-0 run."""
    import json

    from register_embryos.imaging import EmbryoVolume, VoxelSize
    from register_embryos.naming import parse_embryo_name

    monkeypatch.setattr(
        segmentation, "segment_2d",
        lambda vol, **kw: np.ones(vol.shape, dtype=int),
    )
    vol = EmbryoVolume(
        name=parse_embryo_name("20260410_1.5_wt_12s_dorsal_20X_a_b_c.nd2"),
        binned_channels={0: np.zeros((2, 8, 8), np.float32),
                         1: np.zeros((2, 8, 8), np.float32)},
        voxel=VoxelSize(1.0, 1.0), bin_size=1, c_size=2, z_size=2,
    )
    segmentation.segment_embryo(
        vol, mode="2d", nuclei_channel=1, output_dir=tmp_path, verbose=False
    )
    sidecar = json.loads((tmp_path / f"{vol.embryo_id}_gene_map.json").read_text())
    assert sidecar["nuclei_channel"] == 1
