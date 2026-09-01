"""One widget for the two steps that used to be hand-edited literals.

Before this, preparing a cohort meant editing ``angles = [215, 155, 165, 70]``,
re-running a plotting cell, then editing four ``test_limits_N = {0: (0.05, 0.3),
...}`` dicts and re-running again -- with the accepted numbers left behind in one
notebook and nowhere else.

:func:`prepare_widget` does both in one panel.  Rotation and contrast belong
together because they interact: contrast is judged on the rotated view, and a
rotation is judged on a contrast-adjusted view where you can actually see the
embryo.  Both are written to JSON sidecars, so the run is reproducible from disk
and a cohort can be re-processed without re-deciding anything.

The rotation preview is computed on one z-slice at a time, so dragging the angle
slider stays responsive on a full stack; the accepted angle is applied to all
channels and all z-bins later, by
:func:`register_embryos.orientation.apply_orientation_to_volumes`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .contrast import (
    ContrastLimits,
    _transform_array,
    _widen_if_degenerate,
    apply_contrast,
)
from .imaging import EmbryoVolume
from .orientation import Orientation, OrientationSet, clipping_fraction, rotate_frame

__all__ = ["PrepConfig", "prepare_widget", "orientation_widget"]


@dataclass
class PrepConfig:
    """What :func:`prepare_widget` collects: orientation and contrast per embryo.

    Mutated in place as you accept values, so the handle returned when the widget
    is created is the same object to pass downstream -- no need to re-run the cell
    to harvest the numbers.
    """

    orientations: OrientationSet
    contrast: ContrastLimits
    orientation_path: Optional[Path] = None
    contrast_path: Optional[Path] = None

    def save(self, verbose: bool = True) -> None:
        if self.orientation_path:
            self.orientations.save(self.orientation_path, verbose=verbose)
        if self.contrast_path:
            self.contrast.save(self.contrast_path, verbose=verbose)

    @classmethod
    def load(cls, output_dir: str | Path) -> "PrepConfig":
        """Reload a previously accepted configuration from a cohort directory."""
        output_dir = Path(output_dir)
        orientation_path = output_dir / "orientation.json"
        contrast_path = output_dir / "contrast_limits.json"
        return cls(
            orientations=(
                OrientationSet.load(orientation_path)
                if orientation_path.exists()
                else OrientationSet()
            ),
            contrast=(
                ContrastLimits.load(contrast_path) if contrast_path.exists() else ContrastLimits()
            ),
            orientation_path=orientation_path,
            contrast_path=contrast_path,
        )

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"PrepConfig(\n{self.orientations!r}\n{self.contrast!r}\n)"


def _require_widgets():
    try:
        import ipywidgets as widgets
        from IPython.display import display

        return widgets, display
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "this widget needs ipywidgets: pip install 'register-embryos[widgets]'"
        ) from exc


def prepare_widget(
    volumes: Sequence[EmbryoVolume] | EmbryoVolume,
    config: Optional[PrepConfig] = None,
    output_dir: Optional[str | Path] = None,
    transform: str = "none",
    autosave: bool = True,
    rotation_step: float = 1.0,
    figsize: Tuple[float, float] = (13.0, 4.2),
):
    """Interactive rotation + contrast for a cohort. Returns a live :class:`PrepConfig`.

    Layout, left to right: the rotated raw frame with the contrast window marked,
    the intensity histogram with the window shaded, and the rotated + clipped
    result -- which is exactly what segmentation will see.

    Rotation controls apply to the whole embryo (all channels, all z-bins);
    contrast controls apply to the selected channel only.  That asymmetry is
    deliberate and matches the physics: the embryo has one pose, but each
    fluorophore has its own dynamic range.

    Buttons:
        Accept rotation   store the angle and flips for this embryo
        Accept contrast   store the window for this embryo + channel, then jump
                          to the next channel or embryo that still needs one
        All channels      apply this window to every channel of this embryo
        Auto (p1/p99)     snap the window to percentiles of the current view
        Copy contrast     push this channel's window to every other embryo
        Nudge -90/+90     quarter turns, for getting anterior pointing the same
                          way before fine-tuning

    Args:
        output_dir: where ``orientation.json`` and ``contrast_limits.json`` go.
        transform: ``"none"`` or ``"log2"``; recorded with the contrast limits so
            the same lift is re-applied when they are used.
        rotation_step: slider granularity in degrees.

    Raises:
        ImportError: if ipywidgets is unavailable.  Fall back to
            :func:`~register_embryos.contrast.auto_contrast_limits` plus an
            explicit :class:`~register_embryos.orientation.OrientationSet`.
    """
    widgets, display = _require_widgets()
    import matplotlib.pyplot as plt

    if isinstance(volumes, EmbryoVolume):
        volumes = [volumes]
    volume_list = list(volumes)
    if not volume_list:
        raise ValueError("no volumes supplied")

    output_path = Path(output_dir) if output_dir else None
    if config is None:
        config = PrepConfig(
            orientations=OrientationSet(),
            contrast=ContrastLimits(transform=transform),
            orientation_path=output_path / "orientation.json" if output_path else None,
            contrast_path=output_path / "contrast_limits.json" if output_path else None,
        )
    config.contrast.transform = transform

    by_id = {volume.embryo_id: volume for volume in volume_list}

    # Percentile seeds precomputed once; recomputing per slider tick makes
    # dragging laggy on a full stack.
    seeds: Dict[Tuple[str, int], Tuple[float, float]] = {}
    for volume in volume_list:
        for channel, data in volume.binned_channels.items():
            array = _transform_array(data, transform)
            lo, hi = np.percentile(array, [1, 99.5])
            seeds[(volume.embryo_id, channel)] = (float(lo), float(max(hi, lo + 1e-6)))

    label_width = {"description_width": "88px"}
    wide = widgets.Layout(width="430px")

    embryo_dd = widgets.Dropdown(
        options=[v.embryo_id for v in volume_list], description="Embryo:",
        layout=widgets.Layout(width="620px"), style=label_width,
    )
    channel_dd = widgets.Dropdown(description="Channel:", style=label_width,
                                  layout=widgets.Layout(width="300px"))
    z_slider = widgets.IntSlider(description="z-bin:", min=0, max=0, value=0,
                                 continuous_update=False, layout=wide, style=label_width)
    proj_cb = widgets.Checkbox(value=True, description="max projection", indent=False)

    angle_slider = widgets.FloatSlider(
        description="xy rotate °:", min=-180.0, max=180.0, step=rotation_step, value=0.0,
        continuous_update=False, readout_format=".1f", layout=wide, style=label_width,
    )
    resize_cb = widgets.Checkbox(value=False, description="grow canvas (resize)", indent=False)
    flip_x_cb = widgets.Checkbox(value=False, description="flip x", indent=False)
    flip_y_cb = widgets.Checkbox(value=False, description="flip y", indent=False)
    flip_z_cb = widgets.Checkbox(value=False, description="flip z", indent=False)
    nudge_ccw = widgets.Button(description="−90°", layout=widgets.Layout(width="70px"))
    nudge_cw = widgets.Button(description="+90°", layout=widgets.Layout(width="70px"))
    accept_rot_btn = widgets.Button(description="Accept rotation", button_style="primary",
                                    icon="rotate-right")

    lo_slider = widgets.FloatSlider(
        description="contrast lo:", min=0.0, max=1.0, step=0.005, value=0.05,
        readout_format=".3f", continuous_update=False, layout=wide, style=label_width,
    )
    hi_slider = widgets.FloatSlider(
        description="contrast hi:", min=0.0, max=1.0, step=0.005, value=0.60,
        readout_format=".3f", continuous_update=False, layout=wide, style=label_width,
    )
    accept_con_btn = widgets.Button(description="Accept contrast", button_style="success",
                                    icon="check")
    all_channels_btn = widgets.Button(description="All channels", button_style="success")
    auto_btn = widgets.Button(description="Auto (p1/p99)", button_style="info")
    copy_btn = widgets.Button(description="Copy contrast to all embryos",
                              button_style="warning")

    status = widgets.HTML()
    progress = widgets.HTML()
    plot_out = widgets.Output()
    state = {"guard": False}

    def current_volume() -> EmbryoVolume:
        return by_id[embryo_dd.value]

    def current_orientation() -> Orientation:
        return Orientation(
            xy_rotation=angle_slider.value,
            flip_x=flip_x_cb.value,
            flip_y=flip_y_cb.value,
            flip_z=flip_z_cb.value,
        )

    def channel_options(volume: EmbryoVolume) -> List[Tuple[str, int]]:
        options = []
        for channel in sorted(volume.binned_channels):
            if channel == 0:
                stain = volume.channel_names[0] if volume.channel_names else "DAPI"
                options.append((f"0 — nuclei ({stain})", 0))
            else:
                options.append((f"{channel} — {volume.gene_map.get(channel, f'ch{channel}')}",
                                channel))
        return options

    def refresh_progress() -> None:
        total = sum(len(v.binned_channels) for v in volume_list)
        done = sum(
            1 for v in volume_list for ch in v.binned_channels
            if ch in config.contrast.limits.get(v.embryo_id, {})
        )
        rotated = sum(1 for v in volume_list if v.embryo_id in config.orientations)
        progress.value = (
            f"rotation: <b>{rotated}/{len(volume_list)}</b> embryos &nbsp;|&nbsp; "
            f"contrast: <b>{done}/{total}</b> embryo-channels &nbsp;|&nbsp; "
            f"transform=<code>{transform}</code>"
        )

    def oriented_frame() -> np.ndarray:
        """The frame exactly as apply_orientation would produce it.

        Order matters: rotation interpolates, so rotating a max projection is not
        the same image as max-projecting rotated planes -- differences of ~0.15 on
        a [0,1] image, i.e. clearly visible. The preview has to rotate per plane
        and project afterwards, the way the volume is actually transformed, or the
        contrast window is being judged against an image that never gets segmented.
        """
        volume = current_volume()
        data = _transform_array(volume.binned_channels[channel_dd.value], transform)
        if flip_z_cb.value:
            data = data[::-1]
        if flip_y_cb.value:
            data = data[:, ::-1]
        if flip_x_cb.value:
            data = data[:, :, ::-1]

        angle = angle_slider.value
        if not angle % 360:
            return data.max(axis=0) if proj_cb.value else data[z_slider.value]

        if not proj_cb.value:
            return rotate_frame(data[z_slider.value], angle, resize=resize_cb.value)
        return np.stack([
            rotate_frame(data[z], angle, resize=resize_cb.value)
            for z in range(data.shape[0])
        ]).max(axis=0)

    def redraw(*_) -> None:
        if state["guard"]:
            return
        volume = current_volume()
        frame = oriented_frame()
        lo, hi = lo_slider.value, hi_slider.value
        if hi <= lo:
            hi = lo + 1e-3

        stored_contrast = config.contrast.limits.get(volume.embryo_id, {}).get(channel_dd.value)
        stored_orientation = config.orientations.get(volume.embryo_id)

        with plot_out:
            plot_out.clear_output(wait=True)
            fig, axes = plt.subplots(1, 3, figsize=figsize)

            axes[0].imshow(frame, cmap="gray")
            view = "max proj" if proj_cb.value else f"z-bin {z_slider.value}"
            axes[0].set_title(
                f"rotated {angle_slider.value:g}° — {view} — {frame.shape[0]}x{frame.shape[1]}",
                fontsize=9,
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
                f"what segmentation sees — {saturated:.2f}% saturated, "
                f"{floored:.1f}% floored", fontsize=9,
            )
            axes[2].axis("off")

            lost = (
                clipping_fraction(frame if not resize_cb.value else frame, 0.0)
                if resize_cb.value
                else clipping_fraction(
                    _transform_array(
                        current_volume().binned_channels[channel_dd.value], transform
                    ).max(axis=0),
                    angle_slider.value,
                )
            )
            warning = (
                f"  |  ⚠ {lost*100:.1f}% of signal clipped by the fixed canvas"
                if lost > 0.01 else ""
            )
            fig.suptitle(
                f"{volume.embryo_id}  |  ch{channel_dd.value} "
                f"{volume.gene_map.get(channel_dd.value, 'nuclei')}\n"
                f"rotation stored: {stored_orientation.describe()}  |  contrast stored: "
                + (f"({stored_contrast[0]:.3f}, {stored_contrast[1]:.3f})"
                   if stored_contrast else "none")
                + warning,
                fontsize=9,
            )
            fig.tight_layout()
            plt.show()
            # Close it: pyplot keeps every figure alive until told otherwise, and
            # this redraws on each slider tick, dropdown change and Accept. A
            # 7-embryo, 4-channel session runs to hundreds of figures and the
            # notebook slows to a crawl long before the cohort is finished.
            plt.close(fig)

    def sync_embryo(*_) -> None:
        volume = current_volume()
        state["guard"] = True
        channel_dd.options = channel_options(volume)
        if channel_dd.value not in volume.binned_channels:
            channel_dd.value = min(volume.binned_channels)
        z_slider.max = max(0, volume.shape[0] - 1)
        z_slider.value = min(z_slider.value, z_slider.max)
        orientation = config.orientations.get(volume.embryo_id)
        angle_slider.value = orientation.xy_rotation
        flip_x_cb.value = orientation.flip_x
        flip_y_cb.value = orientation.flip_y
        flip_z_cb.value = orientation.flip_z
        state["guard"] = False
        sync_contrast()

    def sync_contrast(*_) -> None:
        volume = current_volume()
        stored = config.contrast.limits.get(volume.embryo_id, {}).get(channel_dd.value)
        lo, hi = stored if stored else seeds[(volume.embryo_id, channel_dd.value)]
        state["guard"] = True
        lo_slider.value, hi_slider.value = float(lo), float(hi)
        state["guard"] = False
        redraw()

    def maybe_autosave() -> None:
        # Quiet: this fires on every Accept, and two log lines per click buries
        # the widget under its own output. The status line confirms the action,
        # and the progress counter shows the totals.
        if autosave:
            config.save(verbose=False)

    def on_accept_rotation(_) -> None:
        volume = current_volume()
        orientation = current_orientation()
        config.orientations.set(volume.embryo_id, orientation)
        status.value = (
            f"<b style='color:#2255cc'>↻ {volume.embryo_id}</b> rotation = "
            f"{orientation.describe()}"
        )
        refresh_progress()
        maybe_autosave()
        redraw()

    def on_accept_contrast(_) -> None:
        volume = current_volume()
        config.contrast.set(volume.embryo_id, channel_dd.value, lo_slider.value, hi_slider.value)
        status.value = (
            f"<b style='color:#009988'>✓ {volume.embryo_id} ch{channel_dd.value}</b> = "
            f"({lo_slider.value:.3f}, {hi_slider.value:.3f})"
        )
        refresh_progress()
        maybe_autosave()
        # Advance so a cohort can be walked with one button.
        remaining = [
            ch for ch in sorted(volume.binned_channels)
            if ch not in config.contrast.limits.get(volume.embryo_id, {})
        ]
        if remaining:
            channel_dd.value = remaining[0]
        else:
            unset = [
                v.embryo_id for v in volume_list
                if any(ch not in config.contrast.limits.get(v.embryo_id, {})
                       for ch in v.binned_channels)
            ]
            if unset:
                embryo_dd.value = unset[0]
            else:
                redraw()

    def on_all_channels(_) -> None:
        volume = current_volume()
        for channel in volume.binned_channels:
            config.contrast.set(volume.embryo_id, channel, lo_slider.value, hi_slider.value)
        status.value = (
            f"<b style='color:#009988'>✓ {volume.embryo_id}</b> — all "
            f"{len(volume.binned_channels)} channels = "
            f"({lo_slider.value:.3f}, {hi_slider.value:.3f})"
        )
        refresh_progress()
        maybe_autosave()
        redraw()

    def on_auto(_) -> None:
        frame = oriented_frame()
        lo, hi = _widen_if_degenerate(*np.percentile(frame, [1, 99]), frame)
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
                config.contrast.set(
                    volume.embryo_id, channel, lo_slider.value, hi_slider.value
                )
                applied += 1
        status.value = (
            f"<b style='color:#aa7700'>copied</b> ch{channel} window to {applied} embryo(s)"
        )
        refresh_progress()
        maybe_autosave()

    def nudge(delta: float):
        def handler(_) -> None:
            angle_slider.value = ((angle_slider.value + delta + 180) % 360) - 180
        return handler

    embryo_dd.observe(sync_embryo, names="value")
    channel_dd.observe(sync_contrast, names="value")
    for control in (z_slider, proj_cb, angle_slider, resize_cb,
                    flip_x_cb, flip_y_cb, flip_z_cb, lo_slider, hi_slider):
        control.observe(redraw, names="value")
    accept_rot_btn.on_click(on_accept_rotation)
    accept_con_btn.on_click(on_accept_contrast)
    all_channels_btn.on_click(on_all_channels)
    auto_btn.on_click(on_auto)
    copy_btn.on_click(on_copy)
    nudge_ccw.on_click(nudge(-90))
    nudge_cw.on_click(nudge(+90))

    sync_embryo()
    refresh_progress()

    def section(title: str) -> "widgets.HTML":
        return widgets.HTML(
            f"<div style='margin:6px 0 2px;font-weight:600;"
            f"border-bottom:1px solid #ddd'>{title}</div>"
        )

    display(
        widgets.VBox([
            widgets.HTML("<h4 style='margin:2px 0'>Prepare cohort — rotation & contrast</h4>"),
            progress,
            widgets.HBox([embryo_dd, channel_dd]),
            widgets.HBox([z_slider, proj_cb]),
            section("Orientation (whole embryo, all channels)"),
            widgets.HBox([angle_slider, nudge_ccw, nudge_cw]),
            widgets.HBox([flip_x_cb, flip_y_cb, flip_z_cb, resize_cb]),
            accept_rot_btn,
            section("Contrast (this channel)"),
            lo_slider,
            hi_slider,
            widgets.HBox([accept_con_btn, all_channels_btn, auto_btn, copy_btn]),
            status,
            plot_out,
        ])
    )
    return config


def orientation_widget(
    volumes: Sequence[EmbryoVolume] | EmbryoVolume,
    orientations: Optional[OrientationSet] = None,
    save_path: Optional[str | Path] = None,
    **kwargs,
) -> OrientationSet:
    """Rotation only, when contrast is already settled.

    A thin wrapper over :func:`prepare_widget`; the contrast controls are still
    present but ignored, and only the :class:`OrientationSet` is returned.
    """
    config = PrepConfig(
        orientations=orientations or OrientationSet(),
        contrast=ContrastLimits(),
        orientation_path=Path(save_path) if save_path else None,
    )
    prepare_widget(volumes, config=config, **kwargs)
    return config.orientations
