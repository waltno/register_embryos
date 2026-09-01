"""ND2 loading, voxel geometry, z-binning and per-channel normalisation.

This is Phase 1 of the original modular pipeline, reorganised so the loaded
volume, its physical voxel size and the binning factor travel together in one
:class:`EmbryoVolume`.  Downstream steps need the voxel geometry (3D Cellpose
needs the z:xy anisotropy; registration benefits from isotropic coordinates),
and reading it off the file once is more reliable than restating it per cohort.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .naming import EmbryoName, parse_embryo_name

__all__ = [
    "EXPECTED_XY",
    "check_xy_shape",
    "VoxelSize",
    "EmbryoVolume",
    "read_voxel_size",
    "load_nd2_volume",
    "bin_volume_by_z",
    "normalize_channel",
    "load_embryo",
    "load_cohort",
    "channel_intensity_stats",
]

#: The xy frame size this workflow is built around.  Every cohort is acquired at
#: 1024 x 1024, and several defaults downstream are calibrated to it: the nucleus
#: diameter Cellpose auto-estimates, the pixel-distance ICP correspondence
#: threshold, and the atlas downsample target.  A differently sized frame is not
#: refused -- it is flagged, because those defaults will need rescaling.
EXPECTED_XY: Tuple[int, int] = (1024, 1024)


def check_xy_shape(
    shape_yx: Tuple[int, int],
    label: str = "",
    expected: Optional[Tuple[int, int]] = EXPECTED_XY,
    strict: bool = False,
) -> bool:
    """Warn (or raise) when a frame is not the expected xy size.

    Returns True when the frame matches.  ``expected=None`` disables the check.
    """
    if expected is None:
        return True
    if tuple(shape_yx) == tuple(expected):
        return True
    message = (
        f"{label or 'image'}: frame is {shape_yx[0]}x{shape_yx[1]}, expected "
        f"{expected[0]}x{expected[1]}. Pixel-scaled defaults (ICP "
        f"max_correspondence_distance, nucleus diameter, downsample targets) are "
        f"calibrated for {expected[0]}x{expected[1]} and will need rescaling."
    )
    if strict:
        raise ValueError(message)
    print(f"  [WARN] {message}")
    return False


@dataclass(frozen=True)
class VoxelSize:
    """Physical voxel size in micrometres."""

    xy_um: float
    z_um: float

    @property
    def anisotropy(self) -> float:
        """z:xy aspect ratio of one voxel.

        This is what Cellpose's ``anisotropy`` argument wants: it rescales z
        internally so a spherical nucleus is spherical in voxel space.  It is
        unrelated to ``diameter``, which is the nucleus size in xy pixels.
        """
        return float(self.z_um) / float(self.xy_um)

    def binned(self, bin_size: int) -> "VoxelSize":
        """Voxel size after max-projecting ``bin_size`` z-slices into one."""
        return VoxelSize(self.xy_um, self.z_um * bin_size)


@dataclass
class EmbryoVolume:
    """One ND2 file: raw stack, z-binned channels, and voxel geometry.

    Attributes:
        binned_channels: channel index -> ``(Z_bins, Y, X)`` float array in [0, 1].
        stack_zcyx: the raw ``(Z, C, Y, X)`` volume, or ``None`` once dropped to
            free memory (see :meth:`drop_raw`).
    """

    name: EmbryoName
    binned_channels: Dict[int, np.ndarray]
    voxel: VoxelSize
    bin_size: int
    c_size: int
    z_size: int
    channel_names: Tuple[str, ...] = ()
    stack_zcyx: Optional[np.ndarray] = None
    history: List[str] = field(default_factory=list)
    #: The orientation these channels were rotated into, as
    #: :meth:`~register_embryos.orientation.Orientation.describe` renders it, or
    #: ``"identity"``.  Carried explicitly because a rotation inside a fixed canvas
    #: leaves the array *shape* untouched: saved masks from a different rotation
    #: would pass every other compatibility check and be silently wrong.
    orientation: str = "identity"

    @property
    def embryo_id(self) -> str:
        return self.name.embryo_id

    @property
    def nd2_path(self) -> str:
        return str(self.name.path) if self.name.path is not None else ""

    @property
    def gene_map(self) -> Dict[int, str]:
        return self.name.gene_map

    @property
    def binned_voxel(self) -> VoxelSize:
        return self.voxel.binned(self.bin_size)

    @property
    def shape(self) -> Tuple[int, int, int]:
        first = next(iter(self.binned_channels.values()))
        return tuple(first.shape)  # type: ignore[return-value]

    def drop_raw(self) -> "EmbryoVolume":
        """Release the raw ``(Z, C, Y, X)`` stack.

        A 182-slice 4-channel 1024x1024 stack is ~1.5 GB; a whole cohort of them
        will not fit alongside Cellpose, and nothing after binning reads it.
        """
        self.stack_zcyx = None
        return self

    def replace_channels(
        self, channels: Dict[int, np.ndarray], note: str = "",
        orientation: Optional[str] = None,
    ) -> "EmbryoVolume":
        """A copy carrying new channel arrays, keeping identity and geometry."""
        return EmbryoVolume(
            name=self.name,
            binned_channels=channels,
            voxel=self.voxel,
            bin_size=self.bin_size,
            c_size=self.c_size,
            z_size=self.z_size,
            channel_names=self.channel_names,
            stack_zcyx=self.stack_zcyx,
            history=[*self.history, note] if note else list(self.history),
            orientation=self.orientation if orientation is None else orientation,
        )


def read_voxel_size(
    nd2_path: str | Path,
    xy_um: Optional[float] = None,
    z_um: Optional[float] = None,
) -> VoxelSize:
    """Read xy pixel size and z step from ND2 metadata.

    Explicit ``xy_um`` / ``z_um`` override the file.  The z step is the median
    spacing of ``z_coordinates`` rather than the first difference, so a single
    glitched stage position does not set the scale for the stack.  Falls back to
    an isotropic 1.0/1.0 with a warning when the metadata carries neither.
    """
    from nd2reader import ND2Reader

    meta_xy: Optional[float] = None
    meta_z: Optional[float] = None
    with ND2Reader(str(nd2_path)) as nd2_file:
        metadata = nd2_file.metadata
        pixel_microns = metadata.get("pixel_microns")
        if pixel_microns:
            meta_xy = float(pixel_microns)
        z_coords = metadata.get("z_coordinates")
        if z_coords is not None and len(z_coords) > 1:
            steps = np.abs(np.diff(np.asarray(z_coords, dtype=float)))
            steps = steps[steps > 0]
            if steps.size:
                meta_z = float(np.median(steps))

    resolved_xy = xy_um if xy_um is not None else meta_xy
    resolved_z = z_um if z_um is not None else meta_z
    if resolved_xy is None or resolved_z is None:
        print(
            f"  [VOXEL] {Path(nd2_path).name}: incomplete voxel metadata "
            f"(xy={resolved_xy}, z={resolved_z}); assuming isotropic 1.0 um. "
            f"Pass xy_um/z_um to set it explicitly."
        )
        resolved_xy = resolved_xy or 1.0
        resolved_z = resolved_z or 1.0
    return VoxelSize(float(resolved_xy), float(resolved_z))


def load_nd2_volume(
    nd2_path: str | Path, verbose: bool = True
) -> Tuple[np.ndarray, Dict[str, int], Tuple[str, ...]]:
    """Load a full ND2 as ``(Z, C, Y, X)``.

    Returns the stack, a ``{"c", "z"}`` size dict, and the microscope's own
    channel names (e.g. ``("DAPI", "FITC", "TRITC", "Cy5")``).
    """
    from nd2reader import ND2Reader

    if verbose:
        print(f"  [LOAD] {Path(nd2_path).name}")
    with ND2Reader(str(nd2_path)) as nd2_file:
        c_size = nd2_file.sizes.get("c", 1)
        z_size = nd2_file.sizes.get("z", 1)
        channel_names = tuple(nd2_file.metadata.get("channels") or ())
        channel_stacks: List[np.ndarray] = []
        for channel_index in range(c_size):
            nd2_file.default_coords["c"] = channel_index
            nd2_file.iter_axes = "z"
            channel_stacks.append(np.stack([frame for frame in nd2_file], axis=0))

    stack_zcyx = np.stack(channel_stacks, axis=0).transpose(1, 0, 2, 3)
    if verbose:
        print(
            f"  [LOAD] {c_size} channels x {z_size} z-slices "
            f"-> {stack_zcyx.shape} {stack_zcyx.dtype}"
        )
    return stack_zcyx, {"c": c_size, "z": z_size}, channel_names


def bin_volume_by_z(channel_stack: np.ndarray, bin_size: int) -> np.ndarray:
    """Max-project consecutive groups of ``bin_size`` z-slices.

    Max rather than mean: HCR puncta are sparse and bright, and averaging a
    punctum over five mostly-empty slices dilutes it below the signal threshold.
    """
    if bin_size < 1:
        raise ValueError(f"bin_size must be >= 1, got {bin_size}")
    if bin_size == 1:
        return channel_stack.copy()
    n_z = channel_stack.shape[0]
    binned = [
        channel_stack[z : min(z + bin_size, n_z)].max(axis=0)
        for z in range(0, n_z, bin_size)
    ]
    return np.stack(binned, axis=0)


def normalize_channel(channel_stack: np.ndarray) -> np.ndarray:
    """Scale a channel to [0, 1] by its own min/max.

    Per channel, not per z-bin, so relative brightness between z-bins survives
    and a dim deep bin does not get stretched to look like the bright surface.
    """
    stack = channel_stack.astype(np.float32)
    lo = float(stack.min())
    hi = float(stack.max())
    if hi <= lo:
        return np.zeros_like(stack)
    return (stack - lo) / (hi - lo)


def channel_intensity_stats(channel_stack: np.ndarray) -> Dict[str, float]:
    """Percentile summary used to seed contrast limits."""
    flat = channel_stack.reshape(-1)
    p1, p50, p95, p99, p999 = np.percentile(flat, [1, 50, 95, 99, 99.9])
    return {
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "p1": float(p1),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "p99.9": float(p999),
    }


def load_embryo(
    nd2_path: str | Path,
    bin_size: int = 7,
    name: Optional[EmbryoName] = None,
    xy_um: Optional[float] = None,
    z_um: Optional[float] = None,
    keep_raw: bool = False,
    normalize: bool = True,
    expected_xy: Optional[Tuple[int, int]] = EXPECTED_XY,
    strict_xy: bool = False,
    verbose: bool = True,
) -> EmbryoVolume:
    """Load, z-bin and normalise one ND2 file.

    Args:
        keep_raw: retain the ``(Z, C, Y, X)`` stack on the returned object.  Off
            by default because a cohort of raw stacks is several GB.
        expected_xy: frame size to check against (``None`` disables the check).
        strict_xy: raise instead of warning when the frame size differs.
    """
    nd2_path = Path(nd2_path)
    if name is None:
        name = parse_embryo_name(nd2_path)

    voxel = read_voxel_size(nd2_path, xy_um=xy_um, z_um=z_um)
    stack_zcyx, sizes, channel_names = load_nd2_volume(nd2_path, verbose=verbose)
    check_xy_shape(
        stack_zcyx.shape[-2:], label=nd2_path.name, expected=expected_xy, strict=strict_xy
    )

    binned_channels: Dict[int, np.ndarray] = {}
    for channel_index in range(sizes["c"]):
        binned = bin_volume_by_z(stack_zcyx[:, channel_index], bin_size)
        binned_channels[channel_index] = normalize_channel(binned) if normalize else binned

    volume = EmbryoVolume(
        name=name,
        binned_channels=binned_channels,
        voxel=voxel,
        bin_size=bin_size,
        c_size=sizes["c"],
        z_size=sizes["z"],
        channel_names=channel_names,
        stack_zcyx=stack_zcyx if keep_raw else None,
        history=[f"binned z x{bin_size}"] + (["normalised [0,1]"] if normalize else []),
    )
    if verbose:
        print(
            f"  [BIN]  x{bin_size} -> {volume.shape} | voxel "
            f"{voxel.xy_um:.3f} x {voxel.xy_um:.3f} x {voxel.z_um:.3f} um "
            f"(binned anisotropy {volume.binned_voxel.anisotropy:.2f})"
        )
    return volume


def load_cohort(
    embryos: List[EmbryoName],
    bin_size: int = 7,
    xy_um: Optional[float] = None,
    z_um: Optional[float] = None,
    keep_raw: bool = False,
    expected_xy: Optional[Tuple[int, int]] = EXPECTED_XY,
    strict_xy: bool = False,
    verbose: bool = True,
) -> List[EmbryoVolume]:
    """Load every embryo in a cohort, in the order given."""
    volumes: List[EmbryoVolume] = []
    for index, embryo in enumerate(embryos, start=1):
        if verbose:
            print(f"\n[{index}/{len(embryos)}] {embryo.embryo_id}")
        volumes.append(
            load_embryo(
                embryo.path,
                bin_size=bin_size,
                name=embryo,
                xy_um=xy_um,
                z_um=z_um,
                keep_raw=keep_raw,
                expected_xy=expected_xy,
                strict_xy=strict_xy,
                verbose=verbose,
            )
        )
    return volumes
