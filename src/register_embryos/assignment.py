"""Assign signal pixels to their nearest nucleus, then reduce to a nucleus table.

HCR signal sits in cytoplasm and around nuclei, not inside the nuclear stain, so
measuring gene intensity inside the Cellpose mask alone throws most of it away.
The fix, kept from the original pipeline: pick out the pixels that count as
measured territory, drop everything else as background, and give each surviving
pixel the label of the nearest nucleus.  The per-nucleus intensity is then the mean
over that expanded territory.

What defines "measured territory" is a real choice -- ``mask_source="genes"``
follows the gene channels themselves, ``"nuclei"`` follows the nuclear stain, i.e.
tissue.  See :func:`build_signal_mask`; the short version is that the gene-channel
mask measures bright debris as expression and couples the channels together, while
the nuclear mask does neither but puts every value on a lower scale.

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
    "FATE_ORDER",
    "EmbryoResult",
    "PixelFate",
    "build_signal_mask",
    "assign_signal_pixels_2d",
    "assign_signal_pixels_3d",
    "nucleus_table",
    "build_nucleus_table",
    "build_cohort_tables",
    "pixel_fate",
]

#: Sentinel written into pixels that carry no signal.  Chosen non-zero so it is
#: distinguishable from a genuine zero-intensity measurement.
BACKGROUND_VALUE = 0.3

#: What can happen to a pixel between the raw frame and the nucleus table, in
#: code order.  The index into this list is the value stored in
#: :attr:`PixelFate.fate`, so the order is part of the contract -- append, never
#: reorder.  Codes 4 and 5 are the two ways a pixel that *has* signal is thrown
#: away, which is what :func:`pixel_fate` exists to show.
FATE_ORDER = [
    "background",                    # 0: below the cut and outside every nucleus
    "nucleus, below cut",            # 1: inside a nucleus, but not measured
    "measured: in nucleus",          # 2: inside a nucleus and above the cut
    "measured: assigned",            # 3: signal handed to the nearest nucleus
    "dropped: too far",              # 4: signal beyond max_assign_distance_um
    "dropped: no nuclei on plane",   # 5: signal on a plane Cellpose found nothing on
]

#: Convenience lookup, name -> code.
FATE_CODES = {name: index for index, name in enumerate(FATE_ORDER)}


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


@dataclass
class PixelFate:
    """What :func:`build_nucleus_table` would do to every pixel, without doing it.

    The nucleus table is a mean over surviving pixels, so the pixels it *discards*
    never appear in any downstream figure -- a bright speck assigned across half
    the frame and a real domain look identical once both are a number per nucleus.
    This is the view before that reduction: one code per pixel, from
    :data:`FATE_ORDER`.

    :attr:`distances_um` holds the nucleus distance of every signal pixel outside a
    nucleus, so :meth:`recut` can re-decide the cap without another KD-tree query.
    That makes sweeping ``max_assign_distance_um`` cheap, which is the point -- the
    cap is the one assignment parameter with no principled default, and it is chosen
    by looking.
    """

    embryo_id: str
    fate: np.ndarray                      # uint8 (Z, Y, X), values index FATE_ORDER
    distances_um: np.ndarray              # (N,) float32, one per queried signal pixel
    coords: np.ndarray                    # (N, 3) int16 (z, y, x), matching distances_um
    xy_um: float = 1.0
    z_um: float = 1.0
    is_3d: bool = False
    signal_threshold: float = 0.05
    max_assign_distance_um: Optional[float] = None
    params: Dict[str, object] = field(default_factory=dict)

    @property
    def n_pixels(self) -> int:
        return int(self.fate.size)

    @property
    def counts(self) -> Dict[str, int]:
        """Pixel count per fate, every category present even at zero."""
        tallied = np.bincount(self.fate.ravel(), minlength=len(FATE_ORDER))
        return {name: int(tallied[code]) for code, name in enumerate(FATE_ORDER)}

    @property
    def summary(self) -> pd.DataFrame:
        """One row per fate: pixels, % of the frame, % of the signal pixels.

        ``pct_of_signal`` is the column to read. ``background`` is most of any
        frame, so a percentage of the frame makes every other category look
        negligible whether or not it is.
        """
        counts = self.counts
        signal_total = sum(
            counts[name] for name in FATE_ORDER if name != "background"
            and name != "nucleus, below cut"
        )
        rows = []
        for name in FATE_ORDER:
            rows.append({
                "fate": name,
                "n_pixels": counts[name],
                "pct_of_frame": 100.0 * counts[name] / max(self.n_pixels, 1),
                "pct_of_signal": (
                    np.nan if name in ("background", "nucleus, below cut")
                    else 100.0 * counts[name] / max(signal_total, 1)
                ),
            })
        return pd.DataFrame(rows)

    @property
    def dropped_fraction(self) -> float:
        """Fraction of signal pixels outside nuclei that the cap throws away."""
        if self.distances_um.size == 0:
            return 0.0
        if self.max_assign_distance_um is None:
            return 0.0
        return float(
            (self.distances_um > self.max_assign_distance_um).mean()
        )

    def recut(self, max_assign_distance_um: Optional[float]) -> "PixelFate":
        """The same embryo under a different distance cap. No re-segmentation, no tree.

        Only codes 3 and 4 can move: the threshold decided everything else, and the
        cap cannot rescue a pixel that never had signal.
        """
        fate = self.fate.copy()
        if self.coords.size:
            z, y, x = self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]
            keep = (
                np.ones(len(self.distances_um), dtype=bool)
                if max_assign_distance_um is None
                else self.distances_um <= float(max_assign_distance_um)
            )
            fate[z, y, x] = np.where(
                keep, FATE_CODES["measured: assigned"], FATE_CODES["dropped: too far"]
            )
        return PixelFate(
            embryo_id=self.embryo_id, fate=fate, distances_um=self.distances_um,
            coords=self.coords, xy_um=self.xy_um, z_um=self.z_um, is_3d=self.is_3d,
            signal_threshold=self.signal_threshold,
            max_assign_distance_um=max_assign_distance_um,
            params={**self.params, "max_assign_distance_um": max_assign_distance_um},
        )


def build_signal_mask(
    channels: Dict[int, np.ndarray],
    signal_threshold: float = 0.05,
    source: str = "genes",
    nuclei_channel: int = 0,
    nuclei_threshold: Optional[float] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Boolean mask of the pixels that count as measured territory.

    Which channels define that territory is the choice here, and it is not a
    cosmetic one:

    ``"genes"``
        A pixel is kept where **any gene channel** clears ``signal_threshold``.
        The original behaviour. It follows the HCR signal wherever it is, including
        the perinuclear space the nuclear stain does not cover -- but it is defined
        by the same channels being measured, so bright debris in a gene channel
        creates its own territory and is measured as if it were expression, and
        the channels are coupled to each other (see :func:`_apply_background`).
    ``"nuclei"``
        A pixel is kept where the **nuclear channel** clears ``nuclei_threshold``,
        i.e. where there is tissue. Debris that is bright in a gene channel but
        carries no nuclear stain is excluded outright, which is the point. It also
        breaks the circularity: territory no longer depends on any gene's contrast
        window, so gene windows can be tuned one at a time.

        The trade is scale. A gene's per-nucleus value becomes a mean over all
        tissue pixels in its territory rather than over the signal-bearing ones,
        so every value falls and the positivity cut has to be re-picked from the
        data (:func:`~register_embryos.thresholds.positive_fraction`). Signal that
        genuinely sits outside the nuclear stain is also lost, so keep
        ``nuclei_threshold`` low enough that the mask is a tissue mask and not a
        nucleus mask -- check the reported percentage against the embryo's actual
        footprint in the frame.
    ``"all"``
        Union over every channel including the nuclear one.

    Args:
        nuclei_threshold: cut for the nuclear channel under ``source="nuclei"``.
            Defaults to ``signal_threshold``; kept separate because the nuclear
            stain and a gene channel have no reason to share a cut.
    """
    if source == "genes":
        considered = [index for index in sorted(channels) if index != nuclei_channel]
        cut = signal_threshold
    elif source == "nuclei":
        if nuclei_channel not in channels:
            raise ValueError(
                f"source='nuclei' needs channel {nuclei_channel}; have "
                f"{sorted(channels)}"
            )
        considered = [nuclei_channel]
        cut = signal_threshold if nuclei_threshold is None else nuclei_threshold
    elif source == "all":
        considered = sorted(channels)
        cut = signal_threshold
    else:
        raise ValueError(
            f"unknown mask source {source!r} (expected 'genes', 'nuclei' or 'all')"
        )
    if not considered:
        raise ValueError("no channels left to build a signal mask from")

    mask = np.zeros(channels[considered[0]].shape, dtype=bool)
    for index in considered:
        mask |= channels[index] > cut
    if verbose:
        pct = 100.0 * mask.sum() / mask.size
        print(
            f"  [SIGNAL] source={source!r} threshold={cut} over channels "
            f"{considered}: {mask.sum():,}/{mask.size:,} px ({pct:.2f}%)"
        )
        if source == "nuclei" and pct < 5.0:
            print(
                f"  [WARN] only {pct:.2f}% of pixels kept -- at this cut the nuclear "
                f"mask is closer to a nucleus mask than a tissue mask, and "
                f"perinuclear signal will be dropped. Lower nuclei_threshold."
            )
    return mask


def _apply_background(
    channels: Dict[int, np.ndarray],
    signal_mask: np.ndarray,
    mode: str = "union",
    signal_threshold: float = 0.05,
    nuclei_channel: int = 0,
) -> Dict[int, np.ndarray]:
    """Mark which pixels count as a measurement, per channel.

    ``mode="union"`` (the original behaviour) keeps a pixel for **every** channel
    wherever **any** gene channel had signal. That couples the channels: a nucleus
    sitting in a region that is bright in wt1a has its hand2 value averaged over
    those pixels too, whether or not hand2 is on there. Observed consequence --
    raising the hand2 and wt1a contrast floors on one real embryo moved *tbx1* from
    20.4% to 24.2% positive without tbx1's own window changing at all. So gene
    contrast cannot be tuned one channel at a time under this rule.

    ``mode="per_channel"`` keeps a pixel for a given gene only where that gene itself
    clears the threshold, which decouples them: each gene's numbers then depend only
    on its own window. The trade is that a gene's mean is taken over its own positive
    pixels only, which pushes low-level graded expression toward either the threshold
    or zero -- it reads more binary.

    The mask still decides *territory* (which nucleus a pixel belongs to) in both
    modes; this only governs what is treated as measured.

    Both readings change meaning under ``mask_source="nuclei"``. The mask is then a
    tissue mask that no gene's contrast can move, so ``"union"`` no longer couples the
    channels -- it simply measures every gene over the tissue in its territory, and
    ``signal_threshold`` is not consulted at all. ``"per_channel"`` then means
    "tissue **and** this gene above its cut", which is the intersection.
    """
    filtered: Dict[int, np.ndarray] = {}
    for index, data in channels.items():
        copy = data.copy()
        if mode == "union" or index == nuclei_channel:
            keep = signal_mask
        elif mode == "per_channel":
            keep = data > signal_threshold
        else:
            raise ValueError(
                f"unknown signal_mask_mode {mode!r} (expected 'union' or 'per_channel')"
            )
        copy[~keep] = BACKGROUND_VALUE
        filtered[index] = copy
    return filtered


def assign_signal_pixels_2d(
    nuclear_masks: np.ndarray,
    signal_mask: np.ndarray,
    xy_um: float = 1.0,
    max_distance: Optional[float] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Nearest-nucleus assignment within each z-slice independently.

    Matches per-slice 2D labels: a pixel can only join a nucleus present on its
    own slice, so a slice with no nuclei contributes nothing.

    Args:
        max_distance: in micrometres. A signal pixel further than this from any
            nucleus on its slice stays unassigned instead of being handed to the
            nearest one. **This is what excludes debris**: nearest-nucleus
            assignment has no notion of "too far", so one bright speck of debris
            off in the corner of the frame is measured into whichever nucleus
            happens to be closest, however many cell diameters away that is. The
            cap is the only thing standing between that and the nucleus table.
        xy_um: in-plane pixel size, so ``max_distance`` can be stated in
            micrometres rather than in pixels of whatever binning is in force.
    """
    assigned = nuclear_masks.copy()
    max_px = None if max_distance is None else float(max_distance) / float(xy_um)
    total, dropped = 0, 0
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
        distances, indices = tree.query(query_points)
        keep = (
            np.ones(len(query_points), dtype=bool) if max_px is None
            else distances <= max_px
        )
        kept = query_points[keep]
        nearest = nucleus_points[indices[keep]]
        assigned[z][kept[:, 0], kept[:, 1]] = frame[nearest[:, 0], nearest[:, 1]]
        total += int(keep.sum())
        dropped += int((~keep).sum())
    if verbose:
        print(
            f"  [ASSIGN 2D] {total:,} signal pixels assigned to nearest in-slice "
            f"nucleus"
            + (f", {dropped:,} beyond {max_distance} um dropped" if dropped else "")
        )
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


def pixel_fate(
    segmented: SegmentedEmbryo,
    signal_threshold: float = 0.05,
    max_assign_distance_um: Optional[float] = None,
    mask_source: str = "genes",
    nuclei_threshold: Optional[float] = None,
    gene_volume: Optional[Dict[int, np.ndarray]] = None,
    verbose: bool = True,
) -> PixelFate:
    """Classify every pixel the way :func:`build_nucleus_table` will, but keep the map.

    Same threshold, same mask, same nearest-nucleus query with the same cap -- the
    difference is that nothing is reduced to a mean, so what gets discarded stays
    visible and can be drawn (:func:`~register_embryos.plotting.plot_pixel_fate`).
    Run it *before* ``build_tables`` to pick ``signal_threshold`` and
    ``max_assign_distance_um`` by looking at the consequence rather than at the
    resulting nucleus counts, where a wrong cap is invisible.

    Two of the six fates are pixels with real signal being thrown away:

    ``dropped: too far``
        Beyond ``max_assign_distance_um`` of any nucleus. This is the debris
        defence working -- or, if a whole domain is red, the cap set too tight.
    ``dropped: no nuclei on plane``
        On a z-plane where Cellpose found nothing at all, so per-slice assignment
        skips it. ``build_tables`` never mentions these; they are not in its
        "dropped" count, they simply never enter the loop. Typically the top and
        bottom bins of a stack, which is fine -- anywhere else it means
        segmentation failed on that plane.

    Territory is decided by the signal mask under both ``signal_mask_mode``
    settings, so this map is exact for either. What ``"per_channel"`` changes is
    only which *genes* a kept pixel counts for, one channel at a time; the
    in-nucleus split here follows the union rule.

    Args:
        max_assign_distance_um: the cap to draw. ``None`` shows uncapped
            assignment, i.e. every signal pixel reaching its nearest nucleus
            however far away -- worth looking at once, because it is what the
            pipeline did by default before 2026-09-01.
        gene_volume: replacement gene-channel arrays, as in
            :func:`build_nucleus_table`, so a re-tuned contrast can be previewed
            without touching the masks.
    """
    channels = dict(segmented.adjusted_channels)
    if gene_volume:
        for channel, data in gene_volume.items():
            if channel != 0:
                channels[channel] = data

    signal = build_signal_mask(
        channels, signal_threshold=signal_threshold, source=mask_source,
        nuclei_threshold=nuclei_threshold, verbose=verbose,
    )
    masks = segmented.nuclear_masks
    if masks.shape != signal.shape:
        raise ValueError(
            f"masks {masks.shape} and channels {signal.shape} disagree -- the masks "
            f"were made from a different volume (a different rotation or bin size)"
        )
    if max(masks.shape) > np.iinfo(np.int16).max:
        raise ValueError(f"volume {masks.shape} too large for int16 coordinates")

    binned = segmented.volume.binned_voxel
    in_nucleus = masks > 0

    fate = np.zeros(masks.shape, dtype=np.uint8)
    fate[in_nucleus] = FATE_CODES["nucleus, below cut"]
    fate[in_nucleus & signal] = FATE_CODES["measured: in nucleus"]
    outside = signal & ~in_nucleus

    coord_blocks: List[np.ndarray] = []
    distance_blocks: List[np.ndarray] = []

    if segmented.is_3d:
        nucleus_indices = np.nonzero(in_nucleus)
        query = np.column_stack(np.nonzero(outside))
        if nucleus_indices[0].size == 0:
            fate[outside] = FATE_CODES["dropped: no nuclei on plane"]
        elif query.size:
            scale = np.asarray(
                (binned.z_um, binned.xy_um, binned.xy_um), dtype=float
            )
            tree = cKDTree(np.column_stack(nucleus_indices) * scale)
            distances, _ = tree.query(query * scale)
            coord_blocks.append(query.astype(np.int16))
            distance_blocks.append(distances.astype(np.float32))
    else:
        for z in range(masks.shape[0]):
            plane_outside = outside[z]
            if not plane_outside.any():
                continue
            frame = masks[z]
            if frame.max() <= 0:
                # assign_signal_pixels_2d skips this plane entirely; its signal is
                # lost without ever being counted as dropped.
                fate[z][plane_outside] = FATE_CODES["dropped: no nuclei on plane"]
                continue
            nucleus_points = np.column_stack(np.nonzero(frame))
            query = np.column_stack(np.nonzero(plane_outside))
            distances, _ = cKDTree(nucleus_points).query(query)
            coords = np.column_stack(
                [np.full(len(query), z, dtype=np.int16), query.astype(np.int16)]
            )
            coord_blocks.append(coords)
            distance_blocks.append((distances * binned.xy_um).astype(np.float32))

    coords = (
        np.concatenate(coord_blocks) if coord_blocks
        else np.zeros((0, 3), dtype=np.int16)
    )
    distances_um = (
        np.concatenate(distance_blocks) if distance_blocks
        else np.zeros(0, dtype=np.float32)
    )

    result = PixelFate(
        embryo_id=segmented.embryo_id, fate=fate, distances_um=distances_um,
        coords=coords, xy_um=binned.xy_um, z_um=binned.z_um,
        is_3d=segmented.is_3d, signal_threshold=signal_threshold,
        max_assign_distance_um=None,
        params={
            "signal_threshold": signal_threshold,
            "mask_source": mask_source,
            "nuclei_threshold": nuclei_threshold,
            "mode": segmented.mode,
        },
    ).recut(max_assign_distance_um)

    if verbose:
        counts = result.counts
        cap = (
            "uncapped" if max_assign_distance_um is None
            else f"cap {max_assign_distance_um} um"
        )
        print(
            f"  [FATE] {segmented.embryo_id} ({cap}): "
            f"{counts['measured: assigned']:,} assigned, "
            f"{counts['dropped: too far']:,} too far, "
            f"{counts['dropped: no nuclei on plane']:,} on planes with no nuclei"
        )
        blank = counts["dropped: no nuclei on plane"]
        if blank and blank > 0.02 * max(int(outside.sum()), 1):
            print(
                f"  [WARN] {100 * blank / max(int(outside.sum()), 1):.1f}% of signal "
                f"outside nuclei sits on planes Cellpose found no nuclei on -- check "
                f"the fate map before treating that as empty stack"
            )
    return result


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
    signal_mask_mode: str = "union",
    mask_source: str = "genes",
    nuclei_threshold: Optional[float] = None,
    save: bool = True,
    verbose: bool = True,
) -> EmbryoResult:
    """Full assignment + table for one segmented embryo.

    Args:
        gene_volume: replacement channel arrays for the gene channels, letting
            contrast be re-tuned for channels 1+ without re-running Cellpose.
            The nuclear channel always comes from ``segmented``.
        signal_mask_mode: ``"union"`` or ``"per_channel"``; see
            :func:`_apply_background`. Use ``"per_channel"`` to make each gene's
            values depend only on its own contrast window.
        max_assign_distance_um: drop signal pixels further than this from any
            nucleus, in micrometres. Applies in **both** 2D and 3D now. This is
            the direct defence against bright debris: without it, nearest-nucleus
            assignment has no upper bound and a speck in the corner of the frame is
            measured into whichever nucleus is nearest.
        mask_source: which channels define measured territory -- ``"genes"``,
            ``"nuclei"`` or ``"all"``; see :func:`build_signal_mask`.
            ``"nuclei"`` restricts territory to the nuclear stain, which also
            excludes gene-bright debris but changes the scale of every value; on
            this project's data a DAPI cut of 0.05 kept only 3.6% of pixels, i.e.
            it was a nucleus mask, so prefer the distance cap for debris.
        nuclei_threshold: the nuclear-channel cut under ``mask_source="nuclei"``.
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

    if verbose and mask_source == "nuclei" and signal_mask_mode == "union":
        # Easy to lose an afternoon to: under these two settings nothing consults
        # signal_threshold at all -- the nuclear cut decides territory and every
        # channel is measured over it.
        print(
            f"  [NOTE] mask_source='nuclei' with signal_mask_mode='union': "
            f"signal_threshold={signal_threshold} is not used. Territory is "
            f"DAPI > {signal_threshold if nuclei_threshold is None else nuclei_threshold}"
            f"; use signal_mask_mode='per_channel' to also require each gene to "
            f"clear its own cut."
        )
    signal_mask = build_signal_mask(
        channels, signal_threshold=signal_threshold, source=mask_source,
        nuclei_threshold=nuclei_threshold, verbose=verbose,
    )
    channels_filtered = _apply_background(
        channels, signal_mask, mode=signal_mask_mode,
        signal_threshold=signal_threshold,
    )

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
            segmented.nuclear_masks, signal_mask, xy_um=binned_voxel.xy_um,
            max_distance=max_assign_distance_um, verbose=verbose,
        )

    # Reduce by whether the LABELS are z-consistent, not by whether Cellpose ran
    # in 3D.  Assignment above is chosen by is_3d (a 2d+link volume still has
    # per-slice mask extents, so per-slice assignment is right and cheaper), but
    # the reduction must follow the labels: linked ids identify one object across
    # planes, so it gets one row and a true 3D centroid.
    table = nucleus_table(
        embryo_id=volume.embryo_id,
        assigned_masks=assigned,
        channels_filtered=channels_filtered,
        gene_map=volume.gene_map,
        mode="3d" if segmented.labels_are_3d else "2d",
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
        mode=segmented.mode,
        voxel_um=(binned_voxel.xy_um, binned_voxel.z_um),
        masks_path=str(segmented.masks_path) if segmented.masks_path else "",
        assigned_masks_path=assigned_path,
        params={
            "signal_threshold": signal_threshold,
            "mask_source": mask_source,
            "nuclei_threshold": (
                signal_threshold if nuclei_threshold is None else nuclei_threshold
            ),
            "signal_mask_mode": signal_mask_mode,
            **segmented.params,
        },
    )


def build_cohort_tables(
    segmented_sets: Sequence[SegmentedEmbryo],
    signal_threshold: float = 0.05,
    gene_volumes: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
    max_assign_distance_um: Optional[float] = None,
    signal_mask_mode: str = "union",
    mask_source: str = "genes",
    nuclei_threshold: Optional[float] = None,
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
                signal_mask_mode=signal_mask_mode,
                mask_source=mask_source,
                nuclei_threshold=nuclei_threshold,
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
