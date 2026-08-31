"""Per-embryo orientation: in-plane rotation, flips, and out-of-plane rotation.

Embryos are mounted at whatever angle they happened to land, so before anything
downstream the cohort has to be brought into a common gross orientation.  ICP can
in principle recover a large rotation, but in practice it is a local method and
converges far more reliably from a start that is already roughly right -- and the
2D views used to judge contrast and segmentation are only readable when anterior
is consistently in the same direction.

The original notebook did this by eyeballing a max projection, editing a literal
``angles = [215, 155, 165, 70]``, and re-running.  Here the same choice is made in
a widget (:func:`register_embryos.widgets.prepare_widget`) and recorded as JSON.

Three levels, in order of how much they cost you:

``xy_rotation`` + flips
    In-plane rotation of each z-bin, and axis flips.  Lossless apart from
    interpolation, since z is untouched.  This is the routine case.

``xz_rotation`` / ``yz_rotation``
    True volumetric rotation, which resamples across z.  Available, but read the
    warning on :func:`apply_orientation`: after z-binning there are only tens of
    z-bins, so an out-of-plane rotation interpolates coarse data and visibly
    degrades it.  Prefer fixing gross out-of-plane tilt at the microscope, or let
    ICP handle the residual.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .imaging import EmbryoVolume

__all__ = [
    "Orientation",
    "OrientationSet",
    "rotate_frame",
    "apply_orientation",
    "apply_orientation_to_volumes",
    "apply_orientation_to_points",
    "clipping_fraction",
]


@dataclass
class Orientation:
    """One embryo's gross orientation correction.

    Angles are degrees counter-clockwise.  Flips are applied before rotation, so
    a flip does not change what a given angle means.
    """

    xy_rotation: float = 0.0
    flip_x: bool = False
    flip_y: bool = False
    flip_z: bool = False
    xz_rotation: float = 0.0
    yz_rotation: float = 0.0

    @property
    def is_identity(self) -> bool:
        return (
            self.xy_rotation % 360 == 0
            and self.xz_rotation % 360 == 0
            and self.yz_rotation % 360 == 0
            and not (self.flip_x or self.flip_y or self.flip_z)
        )

    @property
    def needs_volumetric(self) -> bool:
        """True when a rotation resamples across z."""
        return bool(self.xz_rotation % 360) or bool(self.yz_rotation % 360)

    def describe(self) -> str:
        parts = []
        if self.xy_rotation % 360:
            parts.append(f"xy {self.xy_rotation:g}°")
        if self.xz_rotation % 360:
            parts.append(f"xz {self.xz_rotation:g}°")
        if self.yz_rotation % 360:
            parts.append(f"yz {self.yz_rotation:g}°")
        flips = "".join(axis for axis, on in
                        (("x", self.flip_x), ("y", self.flip_y), ("z", self.flip_z)) if on)
        if flips:
            parts.append(f"flip {flips}")
        return ", ".join(parts) or "identity"


@dataclass
class OrientationSet:
    """``embryo_id -> Orientation``, with a JSON sidecar."""

    orientations: Dict[str, Orientation] = field(default_factory=dict)

    def __getitem__(self, embryo_id: str) -> Orientation:
        return self.orientations.setdefault(embryo_id, Orientation())

    def __contains__(self, embryo_id: object) -> bool:
        return embryo_id in self.orientations

    def __len__(self) -> int:
        return len(self.orientations)

    def get(self, embryo_id: str) -> Orientation:
        return self.orientations.get(embryo_id, Orientation())

    def set(self, embryo_id: str, orientation: Orientation) -> None:
        self.orientations[embryo_id] = orientation

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {eid: asdict(o) for eid, o in self.orientations.items()},
                handle, indent=2, sort_keys=True,
            )
        print(f"  [ORIENT] saved {len(self.orientations)} embryo(s) -> {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "OrientationSet":
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls({eid: Orientation(**values) for eid, values in payload.items()})

    @classmethod
    def from_angles(
        cls, embryo_ids: Sequence[str], angles: Sequence[float]
    ) -> "OrientationSet":
        """Build from a bare list of xy angles, the notebook's old ``angles = [...]``."""
        if len(embryo_ids) != len(angles):
            raise ValueError(
                f"{len(embryo_ids)} embryos but {len(angles)} angles"
            )
        return cls({eid: Orientation(xy_rotation=float(a)) for eid, a in zip(embryo_ids, angles)})

    def __repr__(self) -> str:  # pragma: no cover - display only
        lines = [f"OrientationSet({len(self.orientations)} embryos)"]
        for eid, orientation in self.orientations.items():
            lines.append(f"  {eid}: {orientation.describe()}")
        return "\n".join(lines)


def rotate_frame(
    frame: np.ndarray, angle: float, resize: bool = False, order: int = 1
) -> np.ndarray:
    """Rotate a single 2D frame counter-clockwise by ``angle`` degrees.

    Args:
        resize: grow the canvas so nothing is cut off.  Off by default: this
            workflow expects 1024x1024 frames, and a resized frame both breaks
            that and gives different embryos different array shapes, which then
            have to be reconciled before anything can be compared.  Check
            :func:`clipping_fraction` before accepting an angle.
    """
    from skimage.transform import rotate as sk_rotate

    if angle % 360 == 0:
        return frame.copy()
    return sk_rotate(frame, angle, resize=resize, preserve_range=True, order=order).astype(
        frame.dtype, copy=False
    )


def clipping_fraction(
    frame: np.ndarray, angle: float, threshold: float = 0.05
) -> float:
    """Fraction of above-threshold signal that a ``resize=False`` rotation would cut.

    Rotating inside a fixed square canvas discards the corners.  This reports how
    much signal that actually costs, so an angle can be accepted knowingly rather
    than losing a fin bud to a slider drag.
    """
    if angle % 360 == 0:
        return 0.0
    signal = frame > threshold
    total = float(signal.sum())
    if total == 0:
        return 0.0
    rotated = rotate_frame(signal.astype(np.float32), angle, resize=False, order=0)
    return float(max(0.0, 1.0 - (rotated > 0.5).sum() / total))


def _rotate_volume_plane(
    volume: np.ndarray, angle: float, axes: Tuple[int, int], order: int = 1
) -> np.ndarray:
    """Rotate a 3D volume within one plane, keeping the array shape."""
    from scipy.ndimage import rotate as nd_rotate

    if angle % 360 == 0:
        return volume
    return nd_rotate(
        volume, angle, axes=axes, reshape=False, order=order, mode="constant", cval=0.0
    )


def apply_orientation(
    volume: EmbryoVolume,
    orientation: Orientation,
    resize: bool = False,
    order: int = 1,
    warn_clipping_above: float = 0.01,
    verbose: bool = True,
) -> EmbryoVolume:
    """Apply an :class:`Orientation` to every channel of one embryo.

    Order of operations: flips, then in-plane rotation, then any out-of-plane
    rotation.  Every channel gets exactly the same transform, so channels stay
    registered to each other.

    Out-of-plane (``xz_rotation``/``yz_rotation``) resamples across z.  After
    z-binning there are only tens of z-bins, so interpolating between them loses
    real resolution -- a warning is printed, and the operation is still performed
    because a coarse correction can still beat a badly tilted embryo.
    """
    if orientation.is_identity:
        return volume

    channels: Dict[int, np.ndarray] = {}
    clipping_reported = False

    for index, data in volume.binned_channels.items():
        array = data
        if orientation.flip_z:
            array = array[::-1]
        if orientation.flip_y:
            array = array[:, ::-1]
        if orientation.flip_x:
            array = array[:, :, ::-1]

        if orientation.xy_rotation % 360:
            if (
                verbose
                and not clipping_reported
                and not resize
                and index == 0
            ):
                lost = clipping_fraction(array.max(axis=0), orientation.xy_rotation)
                if lost > warn_clipping_above:
                    print(
                        f"  [WARN] {volume.embryo_id}: rotating "
                        f"{orientation.xy_rotation:g}° inside the fixed "
                        f"{array.shape[1]}x{array.shape[2]} canvas would cut "
                        f"{lost*100:.1f}% of nuclear signal. Pass resize=True to "
                        f"grow the canvas instead."
                    )
                clipping_reported = True
            array = np.stack(
                [rotate_frame(array[z], orientation.xy_rotation, resize=resize, order=order)
                 for z in range(array.shape[0])]
            )

        if orientation.xz_rotation % 360:
            array = _rotate_volume_plane(array, orientation.xz_rotation, axes=(0, 2), order=order)
        if orientation.yz_rotation % 360:
            array = _rotate_volume_plane(array, orientation.yz_rotation, axes=(0, 1), order=order)

        channels[index] = np.ascontiguousarray(array)

    if verbose:
        if orientation.needs_volumetric:
            print(
                f"  [ORIENT] {volume.embryo_id}: {orientation.describe()} "
                f"— out-of-plane rotation resamples across only "
                f"{volume.shape[0]} z-bins; expect a resolution loss in z"
            )
        else:
            print(f"  [ORIENT] {volume.embryo_id}: {orientation.describe()}")

    return volume.replace_channels(channels, note=f"oriented ({orientation.describe()})")


def apply_orientation_to_volumes(
    volumes: Sequence[EmbryoVolume],
    orientations: OrientationSet,
    resize: bool = False,
    order: int = 1,
    verbose: bool = True,
) -> List[EmbryoVolume]:
    """Apply orientations across a cohort; embryos with no entry pass through."""
    return [
        apply_orientation(
            volume, orientations.get(volume.embryo_id), resize=resize, order=order, verbose=verbose
        )
        for volume in volumes
    ]


def apply_orientation_to_points(
    df: pd.DataFrame,
    orientation: Orientation,
    coord_cols: Sequence[str] = ("x", "y", "z"),
    center: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Apply an orientation to an existing nucleus table instead of to pixels.

    Cheaper than re-rotating volumes and re-segmenting, and exact rather than
    interpolated, so it is the better route when the orientation only needs fixing
    after the nucleus table already exists.  Rotation is about the cloud centroid
    unless ``center`` is given.
    """
    out = df.copy()
    coords = out[list(coord_cols)].to_numpy(dtype=float)
    origin = np.asarray(center, dtype=float) if center is not None else coords.mean(axis=0)
    coords = coords - origin

    for axis_index, flip in enumerate((orientation.flip_x, orientation.flip_y, orientation.flip_z)):
        if flip:
            coords[:, axis_index] *= -1

    def rotation_matrix(angle_deg: float, axis: int) -> np.ndarray:
        theta = np.deg2rad(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        matrix = np.eye(3)
        if axis == 2:      # about z: rotates x-y (in-plane)
            matrix[:2, :2] = [[cos_t, -sin_t], [sin_t, cos_t]]
        elif axis == 1:    # about y: rotates x-z
            matrix[0, 0], matrix[0, 2] = cos_t, sin_t
            matrix[2, 0], matrix[2, 2] = -sin_t, cos_t
        else:              # about x: rotates y-z
            matrix[1, 1], matrix[1, 2] = cos_t, -sin_t
            matrix[2, 1], matrix[2, 2] = sin_t, cos_t
        return matrix

    for angle, axis in (
        (orientation.xy_rotation, 2),
        (orientation.xz_rotation, 1),
        (orientation.yz_rotation, 0),
    ):
        if angle % 360:
            coords = coords @ rotation_matrix(angle, axis).T

    out[list(coord_cols)] = coords + origin
    return out
