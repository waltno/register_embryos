"""Per-embryo, per-channel contrast limits -- picked in a widget, not in code.

The original notebook set contrast by editing literal dicts (``test_limits_0 =
{0: (0.05, 0.3), ...}``), re-running a plotting cell, and eyeballing the result.
That works but it is slow, unrecorded, and the numbers end up trapped in one
notebook.  Here the same choice is made with :func:`contrast_widget`, which
shows the live image, its histogram and the clipped result as you drag, and
writes the accepted numbers to a JSON sidecar that the rest of the workflow (and
the CLI) reads back.

Nothing in this module is required: :func:`auto_contrast_limits` produces usable
percentile-based limits without any interaction, and
:func:`ContrastLimits.from_dict` accepts hand-written numbers as before.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .imaging import EmbryoVolume, channel_intensity_stats

__all__ = [
    "ContrastLimits",
    "auto_contrast_limits",
    "apply_contrast",
    "apply_contrast_to_volumes",
    "contrast_widget",
    "preview_contrast",
    "log2_lift",
]

Limits = Tuple[float, float]


# ---------------------------------------------------------------------------
# The limits object
# ---------------------------------------------------------------------------

@dataclass
class ContrastLimits:
    """``embryo_id -> channel index -> (vmin, vmax)`` in normalised [0, 1] units.

    Also carries the transform that was applied before the limits were chosen
    (``"none"`` or ``"log2"``), because a limit of 0.3 means a different pixel
    under a log2 lift than without one, and applying limits under the wrong
    transform silently changes every downstream intensity.
    """

    limits: Dict[str, Dict[int, Limits]] = field(default_factory=dict)
    transform: str = "none"

    def __getitem__(self, embryo_id: str) -> Dict[int, Limits]:
        return self.limits[embryo_id]

    def __contains__(self, embryo_id: object) -> bool:
        return embryo_id in self.limits

    def __len__(self) -> int:
        return len(self.limits)

    def get(self, embryo_id: str, channel: int, default: Limits = (0.0, 1.0)) -> Limits:
        return self.limits.get(embryo_id, {}).get(channel, default)

    def set(self, embryo_id: str, channel: int, vmin: float, vmax: float) -> None:
        if vmax <= vmin:
            raise ValueError(f"vmax must exceed vmin, got ({vmin}, {vmax})")
        self.limits.setdefault(embryo_id, {})[int(channel)] = (float(vmin), float(vmax))

    def as_dict(self) -> Dict[str, Dict[int, Limits]]:
        """Plain nested dict, the shape the segmentation functions take."""
        return {eid: dict(channels) for eid, channels in self.limits.items()}

    def missing(self, volumes: Sequence[EmbryoVolume]) -> List[Tuple[str, int]]:
        """``(embryo_id, channel)`` pairs with no limit set yet."""
        gaps: List[Tuple[str, int]] = []
        for volume in volumes:
            for channel in sorted(volume.binned_channels):
                if channel not in self.limits.get(volume.embryo_id, {}):
                    gaps.append((volume.embryo_id, channel))
        return gaps

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "transform": self.transform,
            "limits": {
                eid: {str(ch): list(lim) for ch, lim in channels.items()}
                for eid, channels in self.limits.items()
            },
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print(f"  [CONTRAST] saved {len(self.limits)} embryo(s) -> {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ContrastLimits":
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        # Tolerate a bare {embryo: {channel: [lo, hi]}} file with no wrapper.
        raw = payload.get("limits", payload)
        transform = payload.get("transform", "none") if isinstance(payload, dict) else "none"
        limits = {
            eid: {int(ch): (float(lo), float(hi)) for ch, (lo, hi) in channels.items()}
            for eid, channels in raw.items()
        }
        return cls(limits=limits, transform=transform)

    @classmethod
    def from_dict(
        cls, mapping: Dict[str, Dict[int, Limits]], transform: str = "none"
    ) -> "ContrastLimits":
        return cls(
            limits={
                eid: {int(ch): (float(lo), float(hi)) for ch, (lo, hi) in channels.items()}
                for eid, channels in mapping.items()
            },
            transform=transform,
        )

    def __repr__(self) -> str:  # pragma: no cover - display only
        lines = [f"ContrastLimits(transform={self.transform!r}, {len(self.limits)} embryos)"]
        for eid, channels in self.limits.items():
            rendered = "  ".join(
                f"ch{ch}=({lo:.3f}, {hi:.3f})" for ch, (lo, hi) in sorted(channels.items())
            )
            lines.append(f"  {eid}\n    {rendered}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Non-interactive paths
# ---------------------------------------------------------------------------

def auto_contrast_limits(
    volumes: Sequence[EmbryoVolume],
    low_percentile: float = 90.0,
    high_percentile: float = 99.9,
    transform: str = "none",
    signal_threshold: float = 0.05,
    warn_above_positive_fraction: float = 0.25,
    verbose: bool = True,
) -> ContrastLimits:
    """Percentile contrast limits for every embryo and channel.

    A starting point, not a substitute for the widget.

    ``low_percentile`` defaults to 90, not 1, and the reason matters.  An HCR
    channel is mostly background: the great majority of pixels carry no signal, so
    the 1st percentile of the *image* is the bottom of the background, not the top
    of it.  Using it puts the low limit at ~0, floors nothing, and then the narrow
    high limit stretches faint background all the way to full scale -- after which
    essentially every nucleus reads as expressing.  Putting the low limit up where
    most of the background already lies is what actually separates signal from it.

    The right percentile depends on how much of the field is genuinely positive,
    which is exactly what the widget lets you judge by eye.  This function reports
    the resulting positive fraction per channel and warns when it is implausibly
    high, so a bad automatic choice is visible rather than silent.

    Args:
        low_percentile: intensity percentile mapped to 0.  Raise it for a sparser
            channel, lower it for a broadly expressed one.
        signal_threshold: threshold the positive-fraction diagnostic reports
            against; match it to what you pass to the assignment step.
        warn_above_positive_fraction: warn when more than this fraction of pixels
            would end up above ``signal_threshold``.  0.25 because the observed
            failure on real data sat at 36-44%: a gene channel with a quarter of
            its pixels positive is background contamination, not expression.
    """
    result = ContrastLimits(transform=transform)
    for volume in volumes:
        diagnostics = []
        for channel, data in sorted(volume.binned_channels.items()):
            array = _transform_array(data, transform)
            lo, hi = np.percentile(array, [low_percentile, high_percentile])
            if hi <= lo:
                lo, hi = float(array.min()), float(array.max()) or 1.0
            result.set(volume.embryo_id, channel, float(lo), float(hi))

            positive = float((apply_contrast(array, (lo, hi)) > signal_threshold).mean())
            diagnostics.append((channel, float(lo), float(hi), positive))

        if verbose:
            rendered = "  ".join(
                f"ch{ch}=({lo:.3f},{hi:.3f})[{positive:.0%}+]"
                for ch, lo, hi, positive in diagnostics
            )
            print(f"  {volume.embryo_id}: {rendered}")
            for channel, lo, hi, positive in diagnostics:
                if channel == 0:
                    continue          # the nuclear stain is broadly positive by design
                if positive > warn_above_positive_fraction:
                    print(
                        f"    [WARN] ch{channel} "
                        f"({volume.gene_map.get(channel, '?')}): {positive:.0%} of "
                        f"pixels land above the {signal_threshold} signal threshold. "
                        f"That is almost certainly background stretched into signal -- "
                        f"raise low_percentile (currently {low_percentile:g}) or set "
                        f"this channel in the widget."
                    )
    return result


def log2_lift(volume_or_array):
    """``log2(1 + x)`` on [0, 1] data, which maps back onto [0, 1] exactly.

    Compresses the bright end and lifts dim signal, so faint HCR puncta survive
    a contrast window that would otherwise clip them.  Accepts an
    :class:`EmbryoVolume`, a list of them, or a bare array.
    """
    if isinstance(volume_or_array, np.ndarray):
        return np.log2(1.0 + volume_or_array.astype(np.float32))
    if isinstance(volume_or_array, (list, tuple)):
        return [log2_lift(item) for item in volume_or_array]
    volume: EmbryoVolume = volume_or_array
    return volume.replace_channels(
        {ch: np.log2(1.0 + data.astype(np.float32)) for ch, data in volume.binned_channels.items()},
        note="log2(1+x) lift",
    )


def _transform_array(array: np.ndarray, transform: str) -> np.ndarray:
    if transform in ("none", None, ""):
        return array
    if transform == "log2":
        return np.log2(1.0 + array.astype(np.float32))
    raise ValueError(f"unknown transform {transform!r} (expected 'none' or 'log2')")


def apply_contrast(array: np.ndarray, limits: Limits) -> np.ndarray:
    """Clip to ``limits`` and rescale that window onto [0, 1]."""
    vmin, vmax = limits
    if vmax <= vmin:
        raise ValueError(f"contrast limits must satisfy vmax > vmin, got {limits}")
    return np.clip((array.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)


def apply_contrast_to_volumes(
    volumes: Sequence[EmbryoVolume],
    contrast: ContrastLimits,
    verbose: bool = True,
) -> List[EmbryoVolume]:
    """Apply a :class:`ContrastLimits` to a cohort, honouring its transform.

    Raises if any embryo/channel has no limit, rather than falling back to a
    default -- an unset channel is almost always an embryo that was skipped in
    the widget, and silently using (0, 1) makes it look processed.
    """
    gaps = contrast.missing(volumes)
    if gaps:
        rendered = ", ".join(f"{eid} ch{ch}" for eid, ch in gaps[:8])
        more = f" (+{len(gaps) - 8} more)" if len(gaps) > 8 else ""
        raise ValueError(f"no contrast limits set for: {rendered}{more}")

    adjusted: List[EmbryoVolume] = []
    for volume in volumes:
        channels = {
            ch: apply_contrast(
                _transform_array(data, contrast.transform),
                contrast[volume.embryo_id][ch],
            )
            for ch, data in volume.binned_channels.items()
        }
        adjusted.append(
            volume.replace_channels(
                channels,
                note=f"contrast applied (transform={contrast.transform})",
            )
        )
        if verbose:
            print(f"  {volume.embryo_id}: contrast applied to {len(channels)} channels")
    return adjusted


# ---------------------------------------------------------------------------
# Static preview (works in any notebook, no widgets needed)
# ---------------------------------------------------------------------------

def preview_contrast(
    volume: EmbryoVolume,
    contrast: Optional[ContrastLimits] = None,
    z_index: Optional[int] = None,
    transform: str = "none",
    channels: Optional[Iterable[int]] = None,
    figsize_per_panel: Tuple[float, float] = (3.6, 3.6),
    save_path: Optional[str | Path] = None,
):
    """Grid of ``raw | histogram | clipped`` rows, one row per channel.

    The non-interactive counterpart of :func:`contrast_widget`: use it to render
    the accepted limits into a figure for a lab notebook or a QC folder.
    """
    import matplotlib.pyplot as plt

    channel_list = sorted(channels) if channels is not None else sorted(volume.binned_channels)
    if z_index is None:
        z_index = volume.shape[0] // 2

    fig, axes = plt.subplots(
        len(channel_list),
        3,
        figsize=(figsize_per_panel[0] * 3, figsize_per_panel[1] * len(channel_list)),
        squeeze=False,
    )
    gene_map = volume.gene_map
    for row, channel in enumerate(channel_list):
        data = _transform_array(volume.binned_channels[channel], transform)
        frame = data[z_index]
        limits = (
            contrast.get(volume.embryo_id, channel)
            if contrast is not None and volume.embryo_id in contrast
            else tuple(np.percentile(frame, [1, 99.5]))
        )
        label = gene_map.get(channel, "nuclei" if channel == 0 else f"ch{channel}")

        axes[row][0].imshow(frame, cmap="gray")
        axes[row][0].set_title(f"ch{channel} {label} — raw", fontsize=9)
        axes[row][0].axis("off")

        axes[row][1].hist(frame.reshape(-1), bins=200, color="#4477aa")
        axes[row][1].axvline(limits[0], color="#cc3311", ls="--", lw=1.2, label=f"lo={limits[0]:.3f}")
        axes[row][1].axvline(limits[1], color="#009988", ls="--", lw=1.2, label=f"hi={limits[1]:.3f}")
        axes[row][1].set_yscale("log")
        axes[row][1].set_title("intensity histogram", fontsize=9)
        axes[row][1].legend(fontsize=7)

        axes[row][2].imshow(apply_contrast(frame, limits), cmap="gray", vmin=0, vmax=1)
        axes[row][2].set_title(f"clipped ({limits[0]:.3f}, {limits[1]:.3f})", fontsize=9)
        axes[row][2].axis("off")

    fig.suptitle(f"{volume.embryo_id} — z-bin {z_index} (transform={transform})", fontsize=10)
    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [SAVED] {save_path}")
    return fig


# ---------------------------------------------------------------------------
# The widget
# ---------------------------------------------------------------------------

def contrast_widget(
    volumes: Sequence[EmbryoVolume] | EmbryoVolume,
    contrast: Optional[ContrastLimits] = None,
    save_path: Optional[str | Path] = None,
    transform: str = "none",
    autosave: bool = True,
    max_projection: bool = False,
    figsize: Tuple[float, float] = (11.5, 3.6),
):
    """Interactive contrast picker: returns the live :class:`ContrastLimits`.

    Pick an embryo and channel, scrub z, drag the low/high sliders, and the three
    panels (raw with the window marked, histogram, clipped result) redraw.  The
    returned object is mutated in place as you accept limits, so the same handle
    is what you pass to :func:`apply_contrast_to_volumes` afterwards -- no need
    to re-run this cell to collect the values.

    Buttons:
        Accept        store the current sliders for this embryo+channel
        Accept all ch apply this embryo's current window to all its channels
        Auto (p1/p99) snap the sliders to percentiles of the current z-bin
        Copy to all   push this channel's window to every other embryo

    Args:
        save_path: JSON sidecar; written on every Accept when ``autosave``.
        transform: ``"none"`` or ``"log2"``; recorded on the returned object so
            the same lift is re-applied when the limits are used.
        max_projection: show the z max projection instead of a single z-bin.

    Raises:
        ImportError: if ipywidgets is not installed.  Use
            :func:`auto_contrast_limits` plus :func:`preview_contrast` instead.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "contrast_widget needs ipywidgets: pip install 'register-embryos[widgets]'. "
            "Without it, use auto_contrast_limits() + preview_contrast()."
        ) from exc

    import matplotlib.pyplot as plt

    if isinstance(volumes, EmbryoVolume):
        volumes = [volumes]
    volume_list = list(volumes)
    if not volume_list:
        raise ValueError("no volumes supplied")

    by_id = {volume.embryo_id: volume for volume in volume_list}
    result = contrast if contrast is not None else ContrastLimits(transform=transform)
    result.transform = transform

    # Pre-compute percentile seeds once; recomputing on every slider tick makes
    # dragging feel laggy on a 1024x1024x26 stack.
    seeds: Dict[Tuple[str, int], Tuple[float, float]] = {}
    for volume in volume_list:
        for channel, data in volume.binned_channels.items():
            array = _transform_array(data, transform)
            lo, hi = np.percentile(array, [1, 99.5])
            seeds[(volume.embryo_id, channel)] = (float(lo), float(max(hi, lo + 1e-6)))

    embryo_dd = widgets.Dropdown(
        options=[v.embryo_id for v in volume_list], description="Embryo:",
        layout=widgets.Layout(width="640px"),
        style={"description_width": "70px"},
    )
    channel_dd = widgets.Dropdown(description="Channel:", style={"description_width": "70px"})
    z_slider = widgets.IntSlider(
        description="z-bin:", min=0, max=0, value=0, continuous_update=False,
        layout=widgets.Layout(width="440px"), style={"description_width": "70px"},
    )
    proj_cb = widgets.Checkbox(value=max_projection, description="max projection", indent=False)
    lo_slider = widgets.FloatSlider(
        description="low:", min=0.0, max=1.0, step=0.005, value=0.05, readout_format=".3f",
        continuous_update=False, layout=widgets.Layout(width="440px"),
        style={"description_width": "70px"},
    )
    hi_slider = widgets.FloatSlider(
        description="high:", min=0.0, max=1.0, step=0.005, value=0.60, readout_format=".3f",
        continuous_update=False, layout=widgets.Layout(width="440px"),
        style={"description_width": "70px"},
    )
    accept_btn = widgets.Button(description="Accept", button_style="success", icon="check")
    accept_all_btn = widgets.Button(description="Accept all ch", button_style="success")
    auto_btn = widgets.Button(description="Auto (p1/p99)", button_style="info")
    copy_btn = widgets.Button(description="Copy to all embryos", button_style="warning")
    status = widgets.HTML()
    progress = widgets.HTML()
    plot_out = widgets.Output()

    state = {"guard": False}

    def current_volume() -> EmbryoVolume:
        return by_id[embryo_dd.value]

    def channel_label(volume: EmbryoVolume, channel: int) -> str:
        if channel == 0:
            return f"0 — nuclei ({volume.channel_names[0] if volume.channel_names else 'DAPI'})"
        return f"{channel} — {volume.gene_map.get(channel, f'ch{channel}')}"

    def refresh_progress() -> None:
        total = sum(len(v.binned_channels) for v in volume_list)
        done = sum(
            1
            for v in volume_list
            for ch in v.binned_channels
            if ch in result.limits.get(v.embryo_id, {})
        )
        colour = "#009988" if done == total else "#aa7700"
        progress.value = (
            f"<span style='color:{colour}'><b>{done}/{total}</b> embryo-channel "
            f"limits set</span> &nbsp;|&nbsp; transform=<code>{transform}</code>"
        )

    def frame_for_display() -> np.ndarray:
        volume = current_volume()
        data = _transform_array(volume.binned_channels[channel_dd.value], transform)
        return data.max(axis=0) if proj_cb.value else data[z_slider.value]

    def redraw(*_) -> None:
        if state["guard"]:
            return
        volume = current_volume()
        frame = frame_for_display()
        lo, hi = lo_slider.value, hi_slider.value
        if hi <= lo:
            hi = lo + 1e-3
        stored = result.limits.get(volume.embryo_id, {}).get(channel_dd.value)

        with plot_out:
            plot_out.clear_output(wait=True)
            fig, axes = plt.subplots(1, 3, figsize=figsize)

            axes[0].imshow(frame, cmap="gray")
            axes[0].set_title(
                f"raw — {'max proj' if proj_cb.value else f'z-bin {z_slider.value}'}", fontsize=9
            )
            axes[0].axis("off")

            axes[1].hist(frame.reshape(-1), bins=200, color="#4477aa")
            axes[1].axvline(lo, color="#cc3311", ls="--", lw=1.4)
            axes[1].axvline(hi, color="#009988", ls="--", lw=1.4)
            axes[1].axvspan(lo, hi, color="#ffdd55", alpha=0.18)
            axes[1].set_yscale("log")
            axes[1].set_xlim(0, 1)
            axes[1].set_title(f"histogram — window ({lo:.3f}, {hi:.3f})", fontsize=9)

            axes[2].imshow(apply_contrast(frame, (lo, hi)), cmap="gray", vmin=0, vmax=1)
            saturated = 100.0 * float((frame >= hi).mean())
            floored = 100.0 * float((frame <= lo).mean())
            axes[2].set_title(
                f"clipped — {saturated:.2f}% saturated, {floored:.1f}% floored", fontsize=9
            )
            axes[2].axis("off")

            accepted = (
                f"accepted ({stored[0]:.3f}, {stored[1]:.3f})" if stored else "not yet accepted"
            )
            fig.suptitle(
                f"{volume.embryo_id}  |  ch{channel_dd.value} "
                f"{volume.gene_map.get(channel_dd.value, 'nuclei')}  |  {accepted}",
                fontsize=10,
            )
            fig.tight_layout()
            plt.show()

    def sync_channel_options(*_) -> None:
        volume = current_volume()
        state["guard"] = True
        channel_dd.options = [
            (channel_label(volume, ch), ch) for ch in sorted(volume.binned_channels)
        ]
        if channel_dd.value not in volume.binned_channels:
            channel_dd.value = min(volume.binned_channels)
        z_slider.max = max(0, volume.shape[0] - 1)
        z_slider.value = min(z_slider.value, z_slider.max)
        state["guard"] = False
        sync_sliders()

    def sync_sliders(*_) -> None:
        volume = current_volume()
        key = (volume.embryo_id, channel_dd.value)
        stored = result.limits.get(volume.embryo_id, {}).get(channel_dd.value)
        lo, hi = stored if stored else seeds[key]
        state["guard"] = True
        lo_slider.value, hi_slider.value = float(lo), float(hi)
        state["guard"] = False
        redraw()

    def maybe_autosave() -> None:
        if autosave and save_path:
            result.save(save_path)

    def on_accept(_) -> None:
        volume = current_volume()
        result.set(volume.embryo_id, channel_dd.value, lo_slider.value, hi_slider.value)
        status.value = (
            f"<b style='color:#009988'>&check; {volume.embryo_id} ch{channel_dd.value}</b> "
            f"= ({lo_slider.value:.3f}, {hi_slider.value:.3f})"
        )
        refresh_progress()
        maybe_autosave()
        # Advance to the next unset channel so a cohort can be walked without
        # touching the dropdowns.
        volume_channels = sorted(volume.binned_channels)
        remaining = [ch for ch in volume_channels if ch not in result[volume.embryo_id]]
        if remaining:
            channel_dd.value = remaining[0]
        else:
            unset = [
                v.embryo_id
                for v in volume_list
                if any(ch not in result.limits.get(v.embryo_id, {}) for ch in v.binned_channels)
            ]
            if unset:
                embryo_dd.value = unset[0]
            else:
                redraw()

    def on_accept_all_channels(_) -> None:
        volume = current_volume()
        for channel in volume.binned_channels:
            result.set(volume.embryo_id, channel, lo_slider.value, hi_slider.value)
        status.value = (
            f"<b style='color:#009988'>&check; {volume.embryo_id}</b> — all "
            f"{len(volume.binned_channels)} channels = "
            f"({lo_slider.value:.3f}, {hi_slider.value:.3f})"
        )
        refresh_progress()
        maybe_autosave()
        redraw()

    def on_auto(_) -> None:
        frame = frame_for_display()
        lo, hi = np.percentile(frame, [1, 99])
        state["guard"] = True
        lo_slider.value = float(lo)
        hi_slider.value = float(max(hi, lo + 1e-3))
        state["guard"] = False
        redraw()

    def on_copy(_) -> None:
        channel = channel_dd.value
        applied = 0
        for volume in volume_list:
            if channel in volume.binned_channels:
                result.set(volume.embryo_id, channel, lo_slider.value, hi_slider.value)
                applied += 1
        status.value = (
            f"<b style='color:#aa7700'>copied</b> ch{channel} window "
            f"({lo_slider.value:.3f}, {hi_slider.value:.3f}) to {applied} embryo(s)"
        )
        refresh_progress()
        maybe_autosave()
        redraw()

    embryo_dd.observe(sync_channel_options, names="value")
    channel_dd.observe(sync_sliders, names="value")
    for control in (z_slider, lo_slider, hi_slider, proj_cb):
        control.observe(redraw, names="value")
    accept_btn.on_click(on_accept)
    accept_all_btn.on_click(on_accept_all_channels)
    auto_btn.on_click(on_auto)
    copy_btn.on_click(on_copy)

    sync_channel_options()
    refresh_progress()

    display(
        widgets.VBox(
            [
                widgets.HTML("<h4 style='margin:2px 0'>Contrast limits</h4>"),
                progress,
                widgets.HBox([embryo_dd, channel_dd]),
                widgets.HBox([z_slider, proj_cb]),
                lo_slider,
                hi_slider,
                widgets.HBox([accept_btn, accept_all_btn, auto_btn, copy_btn]),
                status,
                plot_out,
            ]
        )
    )
    return result
