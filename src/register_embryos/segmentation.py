"""Nuclear segmentation: 2D per-z-bin, or true 3D with Cellpose.

Two modes, same output type:

``mode="2d"``
    Cellpose on each z-bin independently, as the original pipeline did.  Labels
    are per-slice, so one physical nucleus spanning two z-bins becomes two
    records with unrelated ids.  Fast, robust, and adequate when the binning is
    coarse enough that nuclei rarely span bins.

``mode="3d"``
    One Cellpose call over the whole binned volume with ``do_3D=True`` and the
    voxel anisotropy, giving labels that are consistent through z, so a nucleus
    is one object with one centroid.  Slower and more memory-hungry, but the
    right input for registration -- 3D ICP on per-slice centroids is fitting a
    cloud that has duplicate points stacked in z.

Anisotropy is the z:xy voxel aspect ratio and is read from the ND2 (times the
z-binning factor).  ``diameter`` is the separate nucleus-size-in-xy-pixels knob;
``None`` lets Cellpose estimate it.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .imaging import EmbryoVolume, VoxelSize

__all__ = [
    "available_cpus",
    "cellpose_major_version",
    "DEFAULT_NUCLEUS_DIAMETER_UM",
    "nucleus_z_span_bins",
    "require_unbinned_for_3d",
    "report_3d_memory",
    "SegmentedEmbryo",
    "segment_2d",
    "segment_3d",
    "segment_embryo",
    "segment_cohort",
    "load_segmented",
    "relabel_3d_from_2d",
]


@dataclass
class SegmentedEmbryo:
    """Contrast-adjusted channels plus the nuclear label volume."""

    volume: EmbryoVolume
    nuclear_masks: np.ndarray  # (Z_bins, Y, X) int labels
    mode: str
    output_dir: Optional[Path] = None
    masks_path: Optional[Path] = None
    params: Dict[str, object] = field(default_factory=dict)

    @property
    def embryo_id(self) -> str:
        return self.volume.embryo_id

    @property
    def adjusted_channels(self) -> Dict[int, np.ndarray]:
        return self.volume.binned_channels

    @property
    def gene_map(self) -> Dict[int, str]:
        return self.volume.gene_map

    @property
    def n_labels(self) -> int:
        """Distinct nonzero labels.

        In 2D mode this is the max per-slice label count, not a nucleus count,
        because labels restart at 1 on every slice.
        """
        return int(len(np.unique(self.nuclear_masks[self.nuclear_masks > 0])))

    @property
    def is_3d(self) -> bool:
        """Whether Cellpose ran a genuine 3D pass.

        Governs how signal pixels are assigned: 3D assignment measures distances
        in micrometres through the volume, 2D assignment works within each slice.
        """
        return self.mode == "3d"

    @property
    def labels_are_3d(self) -> bool:
        """Whether a label id means the same object on every z-plane it appears on.

        True for ``3d`` and for ``2d+link`` -- the whole point of the linking pass
        is that ids become globally consistent.  This is a *different* question
        from :attr:`is_3d`, and conflating them made ``2d+link`` pointless: the
        nucleus table was reduced per-slice, so a linked nucleus still produced one
        row per plane and the linking was discarded at exactly the step that was
        supposed to benefit from it.
        """
        return self.mode in ("3d", "2d+link")


def available_cpus() -> int:
    """CPUs this process may actually use.

    ``os.cpu_count()`` reports the machine's cores, not the process's allowance.
    Inside a cgroup-limited session or a scheduler slot those differ wildly -- a
    2-core allocation on a 192-core node -- and dividing the machine count among
    workers then sets each PyTorch worker to ~96 threads on 2 cores, so they spend
    their time contending instead of computing.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover - not Linux
        return max(1, os.cpu_count() or 1)


def cellpose_major_version() -> int:
    """Major version of the installed Cellpose, or 0 if it cannot be determined."""
    try:
        from importlib.metadata import version

        return int(version("cellpose").split(".")[0])
    except Exception:  # pragma: no cover - defensive
        return 0


_MODEL_NOTE_SHOWN = False


def _load_model(gpu: bool, pretrained_model: Optional[str]):
    """Build a Cellpose model across the v3/v4 API split.

    Cellpose 4 replaced the per-task weights ("nuclei", "cyto") with a single
    general model, CPSAM.  Passing ``pretrained_model="nuclei"`` there does NOT
    select nuclear weights: Cellpose fails to find a checkpoint by that name,
    prints one easily-missed line, and uses CPSAM anyway.  Rather than pass an
    argument that is quietly ignored, ``pretrained_model=None`` (the default) means
    "this Cellpose version's default nuclear-capable model", and the resolved
    choice is stated once per session.

    Pass an explicit string to override -- a v3 task name, or a path to your own
    checkpoint, which is forwarded untouched.
    """
    global _MODEL_NOTE_SHOWN
    from cellpose import models

    major = cellpose_major_version()

    if pretrained_model is None:
        if major >= 4:
            model = models.CellposeModel(gpu=gpu)
            resolved = f"CPSAM (Cellpose {major} default)"
        elif hasattr(models, "CellposeModel"):  # pragma: no cover - v3 path
            model = models.CellposeModel(gpu=gpu, pretrained_model="nuclei")
            resolved = "nuclei (Cellpose 3 weights)"
        else:  # pragma: no cover - v2 path
            model = models.Cellpose(gpu=gpu, model_type="nuclei")
            resolved = "nuclei (Cellpose 2 weights)"
    else:
        if hasattr(models, "CellposeModel"):
            model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)
        else:  # pragma: no cover
            model = models.Cellpose(gpu=gpu, model_type=pretrained_model)
        resolved = f"{pretrained_model!r} (explicit)"
        if major >= 4 and pretrained_model in ("nuclei", "cyto", "cyto2", "cyto3"):
            print(
                f"  [WARN] Cellpose {major} has no {pretrained_model!r} checkpoint; "
                f"it falls back to CPSAM. Pass pretrained_model=None to make that "
                f"explicit."
            )

    if not _MODEL_NOTE_SHOWN:
        print(f"  [MODEL] Cellpose {major}, weights: {resolved}, gpu={gpu}")
        _MODEL_NOTE_SHOWN = True
    return model


def _eval_masks(model, image, **kwargs) -> np.ndarray:
    """Call ``model.eval`` and return only the mask array.

    Cellpose returns 3 or 4 values depending on version and model class.
    """
    output = model.eval(image, **kwargs)
    return np.asarray(output[0])


def segment_2d(
    nuclei_volume: np.ndarray,
    gpu: bool = False,
    diameter: Optional[float] = None,
    pretrained_model: Optional[str] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    verbose: bool = True,
) -> np.ndarray:
    """Cellpose on each z-bin independently. Labels are per-slice."""
    if verbose:
        print(f"  [SEG 2D] {nuclei_volume.shape[0]} z-bins, diameter={diameter}")
    model = _load_model(gpu, pretrained_model)
    masks: List[np.ndarray] = []
    for z_index in range(nuclei_volume.shape[0]):
        frame = (np.clip(nuclei_volume[z_index], 0, 1) * 255).astype(np.uint8)
        mask = _eval_masks(
            model,
            frame,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            channel_axis=None,
        )
        masks.append(mask)
        if verbose:
            print(f"    z{z_index:>3}: {int(mask.max())} nuclei")
    stacked = np.stack(masks, axis=0)
    if verbose:
        print(f"  [SEG 2D] total per-slice labels: {int(stacked.max())} max")
    return stacked


def segment_3d(
    nuclei_volume: np.ndarray,
    anisotropy: float,
    gpu: bool = False,
    diameter: Optional[float] = None,
    pretrained_model: Optional[str] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_size: int = 15,
    stitch_threshold: float = 0.0,
    verbose: bool = True,
) -> np.ndarray:
    """One Cellpose pass over the whole volume, labels consistent through z.

    Args:
        anisotropy: z:xy voxel aspect ratio of the BINNED volume, i.e.
            ``volume.binned_voxel.anisotropy``.  Getting this wrong is the usual
            cause of nuclei that split along z (too low) or merge (too high).
        stitch_threshold: if > 0, Cellpose runs 2D per slice and stitches masks
            by IoU instead of running the 3D flow field.  Much faster and often
            adequate for coarse z-binning; ``do_3D`` is used when this is 0.
        min_size: labels smaller than this many voxels are dropped, which clears
            the sub-nuclear fragments 3D flow tends to leave at stack edges.
    """
    if verbose:
        mode = f"stitch_threshold={stitch_threshold}" if stitch_threshold > 0 else "do_3D=True"
        print(
            f"  [SEG 3D] volume {nuclei_volume.shape}, anisotropy={anisotropy:.2f}, "
            f"diameter={diameter}, {mode}"
        )
    model = _load_model(gpu, pretrained_model)
    stack = (np.clip(nuclei_volume, 0, 1) * 255).astype(np.uint8)

    # z_axis=0 and channel_axis=None are required, not optional: given a bare
    # (Z, Y, X) array Cellpose 4 cannot tell the z axis from a channel axis and
    # raises "z_axis must be specified when segmenting 3D images of ndim=3". It
    # applies to the stitching path too, since Cellpose routes that through the
    # same 3D image conversion.
    volume_axes = dict(z_axis=0, channel_axis=None)

    if stitch_threshold > 0:
        masks = _eval_masks(
            model,
            stack,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            stitch_threshold=stitch_threshold,
            do_3D=False,
            **volume_axes,
        )
    else:
        masks = _eval_masks(
            model,
            stack,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            do_3D=True,
            anisotropy=anisotropy,
            min_size=min_size,
            **volume_axes,
        )

    masks = np.asarray(masks)
    if masks.ndim != 3:  # pragma: no cover - defensive
        raise RuntimeError(f"expected a 3D label volume from Cellpose, got shape {masks.shape}")
    if verbose:
        n_labels = len(np.unique(masks[masks > 0]))
        spans = _label_z_spans(masks)
        print(
            f"  [SEG 3D] {n_labels} nuclei | median z-span "
            f"{np.median(spans) if len(spans) else 0:.1f} bins"
        )
    return masks


#: Rough nucleus diameter for the cohorts this package was built for.  Used only
#: for the informational z-span report; override per call.
DEFAULT_NUCLEUS_DIAMETER_UM = 6.0


def nucleus_z_span_bins(
    binned_z_um: float, nucleus_diameter_um: float = DEFAULT_NUCLEUS_DIAMETER_UM
) -> float:
    """How many z-planes a typical nucleus spans at this z spacing."""
    return float(nucleus_diameter_um) / float(binned_z_um) if binned_z_um else float("inf")


def require_unbinned_for_3d(
    bin_size: int,
    z_um: float,
    label: str = "",
    nucleus_diameter_um: float = DEFAULT_NUCLEUS_DIAMETER_UM,
    allow_binned: bool = False,
) -> None:
    """Refuse 3D segmentation on a z-binned volume.

    3D segmentation takes the **whole z-stack**, unbinned.  z-binning is a
    concession made for 2D segmentation: it max-projects several planes into one
    so each plane has enough signal to segment independently.  That is exactly the
    information 3D needs -- feeding it binned data both destroys the z resolution
    it works from and leaves a nucleus thinner than a single plane, so there is
    nothing to link across z and ``do_3D`` reduces to a slow 2D run.

    With a 1.5 um z-step and a ~6 um nucleus, an unbinned stack gives a nucleus a
    4-plane span; at the 2D-tuned ``bin_size=7`` it is 0.57 planes.

    Raises:
        ValueError: unless ``bin_size == 1`` or ``allow_binned=True``.
    """
    if bin_size == 1:
        return

    span = nucleus_z_span_bins(z_um * bin_size, nucleus_diameter_um)
    message = (
        f"{label or 'volume'}: 3D segmentation needs the unbinned z-stack, but this "
        f"volume was loaded with bin_size={bin_size} ({z_um * bin_size:.2f} um per "
        f"plane, so a ~{nucleus_diameter_um:g} um nucleus spans only {span:.2f} "
        f"planes). Reload with load(bin_size=1) for mode='3d', or use mode='2d' / "
        f"'2d+link' with the binned volume. Pass allow_binned=True to override."
    )
    if allow_binned:
        print(f"  [WARN] {message}")
        return
    raise ValueError(message)


def report_3d_memory(volume_shape: Tuple[int, int, int], n_channels: int = 1) -> None:
    """State the footprint of an unbinned 3D run, which is the usual failure mode."""
    z, y, x = volume_shape
    per_channel_gb = z * y * x * 4 / 1e9          # float32
    print(
        f"  [MEM]  unbinned volume {z}x{y}x{x}: {per_channel_gb:.2f} GB per channel "
        f"as float32 ({per_channel_gb * n_channels:.2f} GB for {n_channels} channels). "
        f"Cellpose needs several multiples of this; keep max_workers=1 for 3D."
    )


def _label_z_spans(masks: np.ndarray) -> np.ndarray:
    """Number of z-bins each label occupies -- a sanity check on anisotropy.

    Spans stuck at 1 mean the 3D pass did not link across z (anisotropy too low,
    or z-binning already coarser than a nucleus).
    """
    labels = np.unique(masks[masks > 0])
    if labels.size == 0:
        return np.array([])
    spans = []
    for label in labels:
        z_present = np.unique(np.nonzero(masks == label)[0])
        spans.append(z_present.size)
    return np.asarray(spans)


def relabel_3d_from_2d(masks_2d: np.ndarray, iou_threshold: float = 0.25) -> np.ndarray:
    """Link per-slice 2D labels into 3D objects by overlap.

    A dependency-free alternative to ``segment_3d`` when Cellpose 3D is too slow:
    walk z, and give a label the id of the label below it whenever their pixel
    IoU clears ``iou_threshold``.  Coarser than true 3D flow -- it cannot split a
    merged blob -- but it removes the duplicate-centroid problem that per-slice
    labels create for registration.
    """
    linked = np.zeros_like(masks_2d)
    next_id = 1
    previous_map: Dict[int, int] = {}

    for z in range(masks_2d.shape[0]):
        frame = masks_2d[z]
        current_map: Dict[int, int] = {}
        for label in np.unique(frame[frame > 0]):
            region = frame == label
            best_id, best_iou = 0, 0.0
            if z > 0:
                below = linked[z - 1]
                for candidate in np.unique(below[region]):
                    if candidate == 0:
                        continue
                    candidate_region = below == candidate
                    intersection = np.logical_and(region, candidate_region).sum()
                    union = np.logical_or(region, candidate_region).sum()
                    iou = intersection / union if union else 0.0
                    if iou > best_iou:
                        best_id, best_iou = int(candidate), float(iou)
            if best_iou >= iou_threshold:
                assigned = best_id
            else:
                assigned = next_id
                next_id += 1
            linked[z][region] = assigned
            current_map[int(label)] = assigned
        previous_map = current_map

    print(f"  [LINK] {next_id - 1} 3D objects from {masks_2d.shape[0]} slices")
    return linked


def segment_embryo(
    volume: EmbryoVolume,
    mode: str = "2d",
    nuclei_channel: int = 0,
    gpu: bool = False,
    diameter: Optional[float] = None,
    anisotropy: Optional[float] = None,
    output_dir: Optional[str | Path] = None,
    save_masks: bool = True,
    nucleus_diameter_um: float = DEFAULT_NUCLEUS_DIAMETER_UM,
    allow_binned: bool = False,
    verbose: bool = True,
    **kwargs,
) -> SegmentedEmbryo:
    """Segment one embryo's nuclear channel.

    Args:
        mode: ``"2d"``, ``"3d"``, or ``"2d+link"`` (2D Cellpose then IoU linking).
        anisotropy: override the value derived from ND2 metadata.  Only used in
            3D mode.
        output_dir: masks and a gene-map sidecar are written here.
        nucleus_diameter_um: physical nucleus size, used only in the message when
            3D is refused on a binned volume.  Unrelated to ``diameter``, which is
            in xy pixels.
        allow_binned: permit ``mode="3d"`` on a z-binned volume.  Off by default:
            3D wants the whole stack, and binned 3D is a slow 2D run.
    """
    if nuclei_channel not in volume.binned_channels:
        raise ValueError(
            f"{volume.embryo_id}: nuclei channel {nuclei_channel} not present "
            f"(have {sorted(volume.binned_channels)})"
        )
    nuclei_volume = volume.binned_channels[nuclei_channel]
    resolved_anisotropy = (
        anisotropy if anisotropy is not None else volume.binned_voxel.anisotropy
    )

    if mode == "2d":
        masks = segment_2d(
            nuclei_volume, gpu=gpu, diameter=diameter, verbose=verbose, **kwargs
        )
    elif mode == "2d+link":
        iou_threshold = kwargs.pop("iou_threshold", 0.25)
        masks = relabel_3d_from_2d(
            segment_2d(nuclei_volume, gpu=gpu, diameter=diameter, verbose=verbose, **kwargs),
            iou_threshold=iou_threshold,
        )
    elif mode == "3d":
        require_unbinned_for_3d(
            bin_size=volume.bin_size,
            z_um=volume.voxel.z_um,
            label=volume.embryo_id,
            nucleus_diameter_um=nucleus_diameter_um,
            allow_binned=allow_binned,
        )
        if verbose:
            report_3d_memory(nuclei_volume.shape, n_channels=len(volume.binned_channels))
        masks = segment_3d(
            nuclei_volume,
            anisotropy=resolved_anisotropy,
            gpu=gpu,
            diameter=diameter,
            verbose=verbose,
            **kwargs,
        )
    else:
        raise ValueError(f"unknown mode {mode!r} (expected '2d', '3d' or '2d+link')")

    output_path = Path(output_dir) if output_dir else None
    masks_path = None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        if save_masks:
            masks_path = output_path / f"{volume.embryo_id}_nuclear_masks.npy"
            np.save(masks_path, masks)
        with open(output_path / f"{volume.embryo_id}_gene_map.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "gene_map": {str(k): v for k, v in volume.gene_map.items()},
                    "mode": mode,
                    "anisotropy": resolved_anisotropy,
                    "bin_size": volume.bin_size,
                    "voxel_um": {"xy": volume.voxel.xy_um, "z": volume.voxel.z_um},
                },
                fh,
                indent=2,
            )

    return SegmentedEmbryo(
        volume=volume,
        nuclear_masks=masks,
        mode=mode,
        output_dir=output_path,
        masks_path=masks_path,
        params={
            "diameter": diameter,
            "anisotropy": resolved_anisotropy,
            "nuclei_channel": nuclei_channel,
            "gpu": gpu,
            **kwargs,
        },
    )


def load_segmented(
    embryo_dir: str | Path,
    volume: EmbryoVolume,
    verbose: bool = True,
) -> SegmentedEmbryo:
    """Rebuild a :class:`SegmentedEmbryo` from masks already on disk.

    Cellpose only ever looks at channel 0. So re-tuning the contrast of a *gene*
    channel does not invalidate the segmentation -- only the signal mask, the pixel
    assignment and the per-nucleus means downstream of it. Re-running Cellpose for
    that is hours of work for no change in its output.

    This reloads the saved masks and pairs them with a freshly contrasted volume, so
    the cheap half of the pipeline can be redone on its own::

        vol = load_embryo(nd2, bin_size=7)
        vol = apply_orientation(vol, orientations.get(vol.embryo_id))
        vol = apply_contrast_to_volumes([vol], new_contrast)[0]   # new gene windows
        seg = load_segmented(out / "embryos" / vol.embryo_id, vol)
        result = build_nucleus_table(seg)                          # minutes, not hours

    The nuclear channel in ``volume`` should still carry the contrast the masks were
    produced with; a mismatch there is flagged, since it means the masks no longer
    correspond to the channel-0 image they came from.

    Raises:
        FileNotFoundError: if no ``*_nuclear_masks.npy`` is present.
    """
    embryo_dir = Path(embryo_dir)
    masks_path = embryo_dir / f"{volume.embryo_id}_nuclear_masks.npy"
    if not masks_path.exists():
        candidates = sorted(embryo_dir.glob("*_nuclear_masks.npy"))
        if not candidates:
            raise FileNotFoundError(f"no saved nuclear masks in {embryo_dir}")
        masks_path = candidates[0]

    masks = np.load(masks_path)
    sidecar = embryo_dir / f"{volume.embryo_id}_gene_map.json"
    mode, anisotropy = "2d", volume.binned_voxel.anisotropy
    if sidecar.exists():
        with open(sidecar, encoding="utf-8") as handle:
            meta = json.load(handle)
        mode = meta.get("mode", mode)
        anisotropy = meta.get("anisotropy", anisotropy)
        saved_bin = meta.get("bin_size")
        if saved_bin is not None and saved_bin != volume.bin_size:
            print(
                f"  [WARN] {volume.embryo_id}: masks were made at bin_size="
                f"{saved_bin} but this volume is bin_size={volume.bin_size}. "
                f"They describe different z sampling and must not be combined."
            )

    nuclei_shape = volume.binned_channels[min(volume.binned_channels)].shape
    if tuple(masks.shape) != tuple(nuclei_shape):
        raise ValueError(
            f"{volume.embryo_id}: saved masks are {masks.shape} but the volume is "
            f"{nuclei_shape}. Re-loading with a different bin_size or a rotation that "
            f"resized the canvas will do this; the masks cannot be reused."
        )

    if verbose:
        n_labels = len(np.unique(masks[masks > 0]))
        print(f"  [LOAD] {volume.embryo_id}: {n_labels} labels from {masks_path.name} "
              f"(mode={mode})")

    return SegmentedEmbryo(
        volume=volume,
        nuclear_masks=masks,
        mode=mode,
        output_dir=embryo_dir,
        masks_path=masks_path,
        params={"anisotropy": anisotropy, "reloaded": True},
    )


def segment_cohort(
    volumes: Sequence[EmbryoVolume],
    output_root: Optional[str | Path] = None,
    mode: str = "2d",
    nuclei_channel: int = 0,
    gpu: bool = False,
    diameter: Optional[float] = None,
    anisotropy: Optional[float] = None,
    nucleus_diameter_um: float = DEFAULT_NUCLEUS_DIAMETER_UM,
    allow_binned: bool = False,
    max_workers: int = 1,
    verbose: bool = True,
    **kwargs,
) -> List[SegmentedEmbryo]:
    """Segment a whole cohort, optionally several embryos at a time.

    Cellpose releases the GIL during inference, so threads genuinely overlap.
    Per-worker torch threads are capped so the workers do not oversubscribe the
    CPU and end up slower than sequential.  Threads are only worth it in 2D mode;
    3D peak memory per embryo is high enough that ``max_workers=1`` is safer.
    """
    output_root = Path(output_root) if output_root else None
    n_threads = max(1, min(max_workers, len(volumes)))

    if mode == "3d" and n_threads > 1:
        print(
            f"  [SEG] mode=3d with max_workers={max_workers}: 3D Cellpose peak "
            f"memory is per-worker; reduce workers if this runs out of memory."
        )
    if n_threads > 1:
        per_worker = max(1, available_cpus() // n_threads)
        try:
            import torch

            torch.set_num_threads(per_worker)
        except ImportError:  # pragma: no cover
            pass
        print(f"  [SEG] parallel: {n_threads} threads x {per_worker} torch threads")

    def run(volume: EmbryoVolume) -> SegmentedEmbryo:
        return segment_embryo(
            volume,
            mode=mode,
            nuclei_channel=nuclei_channel,
            gpu=gpu,
            diameter=diameter,
            anisotropy=anisotropy,
            nucleus_diameter_um=nucleus_diameter_um,
            allow_binned=allow_binned,
            output_dir=output_root / volume.embryo_id if output_root else None,
            verbose=verbose,
            **kwargs,
        )

    if n_threads == 1:
        segmented = []
        for index, volume in enumerate(volumes, start=1):
            if verbose:
                print(f"\n[{index}/{len(volumes)}] {volume.embryo_id}")
            segmented.append(run(volume))
    else:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            segmented = list(executor.map(run, volumes))

    if verbose:
        print(f"\n  [SEG] segmented {len(segmented)} embryos (mode={mode})")
        for item in segmented:
            print(f"    {item.embryo_id}: {item.n_labels} labels")
    return segmented
