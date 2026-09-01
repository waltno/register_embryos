"""3D and 2D views of nucleus clouds, in matched day and night themes.

Every plotting function takes ``mode="dark"`` or ``mode="light"`` and produces
the same figure against the opposite ground, so a figure explored on screen in
dark mode has a publication-ready light twin with no re-tuning.  The two themes
are not just an inverted background: the silent-nucleus grey, the outline colour
and the additive gain differ, because additive colour that reads as bright on
black reads as washed out on white.

Two colouring schemes:

:func:`plot_pointcloud_3d`
    One panel per gene, each nucleus coloured on a black-to-gene-hue ramp by that
    gene's normalised intensity.  Reads quantitatively -- best for asking where
    one gene is on.

:func:`plot_additive_3d` / :func:`plot_additive_2d`
    All genes at once, hue from the additive mix of the channels, dot size and
    opacity from total intensity.  Reads qualitatively -- best for co-expression,
    where the question is which combinations occur where.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Safe at module level: thresholds imports plotting only lazily, inside functions.
from .thresholds import DEFAULT_THRESHOLD, resolve_gene_cuts

__all__ = [
    "GENE_RGB",
    "XY_AXES",
    "Theme",
    "DARK",
    "LIGHT",
    "theme_for",
    "gene_color",
    "additive_style",
    "plot_pointcloud_3d",
    "plot_additive_3d",
    "plot_additive_2d",
    "plot_additive_gene_2d",
    "plot_registration_2d",
    "plot_gene_panels_2d",
    "plot_gene_by_embryo",
]

#: Default hues for the genes in this project's panels.  Chosen to stay
#: distinguishable when added together: cyan + yellow + magenta mix to visibly
#: different colours, which a set of neighbouring hues would not.
GENE_RGB: Dict[str, np.ndarray] = {
    "hand2": np.array([0.00, 1.00, 1.00]),
    "tbx5a": np.array([1.00, 1.00, 0.00]),
    "tbx1": np.array([1.00, 0.53, 0.00]),
    "pax2a": np.array([0.00, 1.00, 0.27]),
    "wt1a": np.array([1.00, 0.00, 1.00]),
    "osr1": np.array([0.30, 0.60, 1.00]),
    "prdm1a": np.array([1.00, 0.40, 0.40]),
    "fli1a": np.array([0.60, 1.00, 0.20]),
}

#: Fallback hues, cycled for genes not in :data:`GENE_RGB`.
FALLBACK_RGB = [
    np.array([0.00, 0.80, 1.00]),
    np.array([1.00, 0.80, 0.00]),
    np.array([1.00, 0.30, 0.70]),
    np.array([0.40, 1.00, 0.40]),
    np.array([1.00, 0.55, 0.10]),
    np.array([0.70, 0.50, 1.00]),
]

#: The pipeline's constant positivity cut, aliased from :mod:`register_embryos.thresholds`
#: rather than restated, so the two cannot drift apart.
INTENSITY_THRESH = DEFAULT_THRESHOLD


@dataclass(frozen=True)
class Theme:
    """One coherent colour scheme for both matplotlib and plotly output."""

    name: str
    paper: str
    scene: str
    grid: str
    font: str
    silent_rgb: np.ndarray
    stroke: str
    axis_label: str
    plotly_template: str

    @property
    def is_dark(self) -> bool:
        return self.name == "dark"


DARK = Theme(
    name="dark",
    paper="#000000",
    scene="#000000",
    grid="#2a2a2a",
    font="#e0e0e0",
    # Light grey rather than dark: on black, a silent nucleus still needs to be
    # visible as tissue context, otherwise the embryo has no shape.
    silent_rgb=np.array([0.82, 0.82, 0.82]),
    stroke="#000000",
    axis_label="#999999",
    plotly_template="plotly_dark",
)

LIGHT = Theme(
    name="light",
    paper="#ffffff",
    scene="#f7f7f7",
    grid="#cccccc",
    font="#1a1a1a",
    silent_rgb=np.array([0.86, 0.86, 0.86]),
    stroke="#333333",
    axis_label="#555555",
    plotly_template="plotly_white",
)


def theme_for(mode: str) -> Theme:
    """``"dark"``/``"night"`` -> :data:`DARK`; ``"light"``/``"day"`` -> :data:`LIGHT`."""
    key = (mode or "dark").lower()
    if key in ("dark", "night", "black"):
        return DARK
    if key in ("light", "day", "white"):
        return LIGHT
    raise ValueError(f"unknown mode {mode!r} (expected 'dark'/'night' or 'light'/'day')")


def gene_color(gene: str, index: int = 0) -> np.ndarray:
    """RGB triple for a gene, falling back to a cycled palette."""
    if gene in GENE_RGB:
        return GENE_RGB[gene]
    return FALLBACK_RGB[index % len(FALLBACK_RGB)]


def _hex(rgb: Sequence[float]) -> str:
    r, g, b = (int(np.clip(c, 0, 1) * 255) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def gene_hex(genes: Sequence[str]) -> Dict[str, str]:
    return {gene: _hex(gene_color(gene, i)) for i, gene in enumerate(genes)}


def _resolve_coords(df: pd.DataFrame, coords: Optional[Sequence[str]]) -> List[str]:
    """Prefer registered coordinates when present, unless told otherwise."""
    if coords is not None:
        missing = [c for c in coords if c not in df.columns]
        if missing:
            raise ValueError(f"missing coordinate columns: {missing}")
        return list(coords)
    if all(c in df.columns for c in ("x_reg", "y_reg", "z_reg")):
        return ["x_reg", "y_reg", "z_reg"]
    return ["x", "y", "z"]


def _resolve_genes(df: pd.DataFrame, genes: Optional[Sequence[str]]) -> List[str]:
    if genes is not None:
        return [g for g in genes if g in df.columns]
    skip = {
        "embryo_id", "nucleus_id", "atlas_point_id", "n_voxels", "n_source_embryos",
        "neighbor_radius", "x", "y", "z", "x_reg", "y_reg", "z_reg",
        "x_um", "y_um", "z_um",
    }
    return [
        c for c in df.columns
        if c not in skip and pd.api.types.is_numeric_dtype(df[c])
    ]


# ---------------------------------------------------------------------------
# Additive styling
# ---------------------------------------------------------------------------

def additive_style(
    df: pd.DataFrame,
    genes: Optional[Sequence[str]] = None,
    mode: str = "dark",
    gamma: float = 0.35,
    quantile_clip: float = 0.95,
    threshold=INTENSITY_THRESH,
    size_range: Tuple[float, float] = (1.0, 14.0),
    alpha_low: float = 0.15,
    gate_below_threshold: bool = False,
    color_scale: str = "intensity",
    color_gamma: float = 1.0,
    min_saturation: float = 0.18,
) -> Dict[str, object]:
    """Per-nucleus colour, size and opacity from the additive channel mix.

    Hue direction comes from the intensity-weighted channel mix, so a nucleus with
    a lot of one gene and a little of another lands near the strong gene's hue
    rather than halfway between.  How saturated that hue is drawn then tracks the
    summed intensity: a barely-positive nucleus is a pale wash of its gene's colour,
    a strongly positive one is the full colour.  Size and opacity follow the same
    total, so the three channels of the encoding agree instead of competing.

    A nucleus counts as expressing when at least ONE channel clears its
    ``threshold``.  Not the channel sum: summing lets a nucleus with three
    sub-threshold channels look positive while being positive for nothing, so
    "coloured" and "positive for some gene" would stop agreeing.

    ``threshold`` takes anything :func:`~register_embryos.thresholds.resolve_gene_cuts`
    takes: a number, a ``{gene: cut}`` mapping, a ``call_thresholds`` results dict,
    ``"otsu"``, or a positive rate like ``"q0.95"``. The data-driven cuts differ per
    gene and per embryo, and a fixed 0.05 on contrast-normalised intensity means a
    different brightness in every embryo.

    Args:
        gamma: < 1 lifts dim nuclei.  0.35 is a moderate lift; the original
            notebooks used 0.1, which pushes nearly everything to full size.
        quantile_clip: intensity at this quantile of the channel sum maps to
            maximum size, so a few saturated nuclei do not flatten the rest.
        color_scale: ``"intensity"`` (the default) fades the hue toward the theme's
            silent grey as the summed intensity falls, so brightness carries
            magnitude and a dim positive nucleus reads as continuous with the
            silent tissue around it rather than as a confident call.  ``"full"``
            restores the older behaviour -- every positive nucleus drawn at full
            saturation whatever its intensity, which makes a nucleus at 0.06 look
            exactly like one at 0.9.
        color_gamma: exponent on the normalised intensity *for colour only*, kept
            separate from ``gamma`` (which lifts dot size).  1.0 is linear, so the
            colour does not quietly flatter weak signal; below 1 lifts dim nuclei.
        min_saturation: how much of the hue the palest positive nucleus keeps.
            Without a floor a just-above-threshold nucleus would be
            indistinguishable from a silent one.
        gate_below_threshold: zero each sub-threshold channel *before* mixing, so
            only the genes a nucleus is actually positive for contribute to its hue
            and its size.  Off by default, which is what the original notebook did:
            the threshold then decides only *whether* to colour a nucleus, and every
            channel tints it regardless.  That is tolerable with three genes and
            actively misleading with eight -- and worst in an atlas, where kNN
            averaging leaves every point with a small non-zero value in every
            channel, so every hue is a blend and the figure comes out a wash of
            mixed colours no flat threshold can clean up.  Turn it on and a point
            positive for pax2a alone reads as pax2a green.
    """
    theme = theme_for(mode)
    gene_list = _resolve_genes(df, genes)
    n = len(df)
    if not gene_list:
        rgb = np.tile(theme.silent_rgb, (n, 1))
        return {
            "rgb": rgb,
            "hex": [_hex(c) for c in rgb],
            "sizes": np.full(n, size_range[0]),
            "alpha": np.full(n, alpha_low),
            "summed": np.zeros(n),
            "hi_mask": np.zeros(n, dtype=bool),
            "genes": [],
            "theme": theme,
        }

    values = df[gene_list].fillna(0).to_numpy(dtype=float)

    # One threshold vocabulary everywhere: a number, a {gene: cut} mapping, a
    # call_thresholds results dict, "otsu", or a rate like "q0.95".
    resolved = resolve_gene_cuts(df, gene_list, threshold)
    cuts = np.array([resolved[gene] for gene in gene_list])
    above = values >= cuts[None, :]
    contributing = np.where(above, values, 0.0) if gate_below_threshold else values

    mixed = np.zeros((n, 3))
    for column, gene in enumerate(gene_list):
        mixed += np.outer(contributing[:, column], gene_color(gene, column))

    summed = contributing.sum(axis=1)
    ceiling = float(np.quantile(summed, quantile_clip))
    normalised = np.clip(summed / (ceiling + 1e-9), 0, 1)
    intensity = normalised ** gamma

    clipped = np.clip(mixed, 0, 1)
    peak = clipped.max(axis=1, keepdims=True)
    bright = clipped / np.where(peak > 1e-9, peak, 1.0)

    if color_scale == "intensity":
        # Fade toward the theme's silent grey, not toward white or black: that is
        # the one target that reads as "lighter" in both themes, and it puts the
        # palest positive nucleus next to the silent ones on the same ramp.
        strength = min_saturation + (1.0 - min_saturation) * normalised ** color_gamma
        bright = (
            theme.silent_rgb[None, :] * (1.0 - strength[:, None])
            + bright * strength[:, None]
        )
    elif color_scale != "full":
        raise ValueError(
            f"color_scale must be 'intensity' or 'full', got {color_scale!r}"
        )

    hi_mask = above.any(axis=1)
    rgb = np.where(hi_mask[:, None], bright, theme.silent_rgb[None, :])
    size_min, size_max = size_range

    return {
        "rgb": rgb,
        "hex": [_hex(c) for c in rgb],
        "sizes": size_min + intensity * (size_max - size_min),
        "alpha": np.where(hi_mask, 1.0, alpha_low),
        "summed": summed,
        "hi_mask": hi_mask,
        "genes": gene_list,
        "theme": theme,
    }


# ---------------------------------------------------------------------------
# Plotly 3D
# ---------------------------------------------------------------------------

def _plotly_axis(theme: Theme, title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=theme.font)),
        backgroundcolor=theme.scene,
        gridcolor=theme.grid,
        showbackground=True,
        tickfont=dict(color=theme.font),
        zerolinecolor=theme.grid,
    )


def _plotly_scene(theme: Theme, labels: Sequence[str]) -> dict:
    return dict(
        bgcolor=theme.scene,
        xaxis=_plotly_axis(theme, labels[0]),
        yaxis=_plotly_axis(theme, labels[1]),
        zaxis=_plotly_axis(theme, labels[2]),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
        aspectmode="data",
    )


def _write_html(fig, save_path: Optional[str | Path]) -> Optional[Path]:
    if save_path is None:
        return None
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))
    print(f"  [SAVED] {save_path}")
    return save_path


def plot_pointcloud_3d(
    df: pd.DataFrame,
    genes: Optional[Sequence[str]] = None,
    mode: str = "dark",
    coords: Optional[Sequence[str]] = None,
    title: str = "",
    marker_size: float = 2.5,
    quantile_cap: float = 0.99,
    save_path: Optional[str | Path] = None,
    width_per_panel: int = 420,
    height: int = 640,
):
    """One interactive 3D panel per gene, coloured by that gene's intensity.

    The colour ramp runs from the theme's ground to the gene hue and is capped at
    the gene's ``quantile_cap`` quantile, so one saturated nucleus does not
    compress the whole scale.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    theme = theme_for(mode)
    coord_cols = _resolve_coords(df, coords)
    gene_list = _resolve_genes(df, genes)
    if not gene_list:
        raise ValueError("no gene columns found to plot")
    hexes = gene_hex(gene_list)
    base = theme.paper if theme.is_dark else "#ffffff"

    fig = make_subplots(
        rows=1,
        cols=len(gene_list),
        specs=[[{"type": "scatter3d"}] * len(gene_list)],
        subplot_titles=gene_list,
        horizontal_spacing=0.02,
    )
    for column, gene in enumerate(gene_list, start=1):
        cap = float(df[gene].quantile(quantile_cap)) or 1.0
        fig.add_trace(
            go.Scatter3d(
                x=df[coord_cols[0]], y=df[coord_cols[1]], z=df[coord_cols[2]],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=df[gene],
                    colorscale=[[0, base], [1, hexes[gene]]],
                    cmin=0, cmax=cap,
                    opacity=0.85,
                    showscale=True,
                    colorbar=dict(
                        len=0.5, thickness=10,
                        x=column / len(gene_list) - 0.012,
                        tickfont=dict(color=theme.font),
                        title=dict(text="", font=dict(color=theme.font)),
                    ),
                    line=dict(width=0),
                ),
                hovertemplate=f"<b>{gene}</b>: %{{marker.color:.3f}}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=column,
        )
        scene_key = "scene" if column == 1 else f"scene{column}"
        fig.update_layout(**{scene_key: _plotly_scene(theme, ("X", "Y", "Z"))})

    fig.update_layout(
        template=theme.plotly_template,
        paper_bgcolor=theme.paper,
        font=dict(color=theme.font),
        title=dict(text=title or f"{len(df):,} nuclei", font=dict(size=14, color=theme.font)),
        width=width_per_panel * len(gene_list) + 60,
        height=height,
    )
    _write_html(fig, save_path)
    return fig


def plot_additive_3d(
    df: pd.DataFrame,
    genes: Optional[Sequence[str]] = None,
    mode: str = "dark",
    coords: Optional[Sequence[str]] = None,
    title: str = "",
    style: Optional[Dict[str, object]] = None,
    save_path: Optional[str | Path] = None,
    width: int = 900,
    height: int = 760,
    **style_kwargs,
):
    """One interactive 3D view with every gene overlaid additively.

    Silent nuclei are drawn first and faint so they give the embryo a shape
    without occluding expressing nuclei, which are drawn opaque on top.
    """
    import plotly.graph_objects as go

    theme = theme_for(mode)
    coord_cols = _resolve_coords(df, coords)
    st = style or additive_style(df, genes, mode=mode, **style_kwargs)
    gene_list = st["genes"]
    hexes = gene_hex(gene_list)

    hex_array = np.asarray(st["hex"])
    sizes = np.asarray(st["sizes"])
    hi = np.asarray(st["hi_mask"])
    x, y, z = (df[c].to_numpy() for c in coord_cols)

    fig = go.Figure()
    for mask, opacity, name in ((~hi, 0.18, "silent"), (hi, 1.0, "expressing")):
        if not mask.any():
            continue
        fig.add_trace(
            go.Scatter3d(
                x=x[mask], y=y[mask], z=z[mask], mode="markers",
                marker=dict(
                    size=sizes[mask].tolist(),
                    color=hex_array[mask].tolist(),
                    opacity=opacity,
                    line=dict(width=0),
                ),
                name=name,
                hovertemplate="X %{x:.1f}<br>Y %{y:.1f}<br>Z %{z:.1f}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        template=theme.plotly_template,
        paper_bgcolor=theme.paper,
        font=dict(color=theme.font),
        title=dict(
            text=title or f"additive overlay — {int(hi.sum()):,}/{len(df):,} expressing",
            font=dict(size=14, color=theme.font),
        ),
        scene=_plotly_scene(theme, ("X", "Y", "Z")),
        width=width, height=height,
        annotations=[
            dict(
                text=f"■ {gene}", x=0.01, y=0.98 - i * 0.045,
                xref="paper", yref="paper", showarrow=False,
                font=dict(color=hexes[gene], size=12), align="left",
            )
            for i, gene in enumerate(gene_list)
        ],
    )
    _write_html(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Matplotlib 2D projections
# ---------------------------------------------------------------------------

PROJECTIONS = (("x", "y", "XY"), ("x", "z", "XZ"), ("y", "z", "YZ"))


def _projection_cols(coord_cols: Sequence[str]) -> List[Tuple[str, str, str]]:
    x, y, z = coord_cols
    return [(x, y, "XY"), (x, z, "XZ"), (y, z, "YZ")]


XY_AXES = frozenset({"x", "y", "x_reg", "y_reg", "x_um", "y_um"})


def _axis_unit(column: str) -> str:
    """Which physical unit an axis is in: xy pixels, z bins, or micrometres."""
    if column.endswith("_um"):
        return "um"
    base = column.replace("_reg", "")
    return "xy_px" if base in ("x", "y") else "z_bin"


def _style_axes(
    ax,
    theme: Theme,
    xlabel: str,
    ylabel: str,
    title: str,
    z_aspect: Optional[float] = None,
) -> None:
    """Label and colour a panel, and set an aspect ratio that does not lie.

    ``aspect="equal"`` is only meaningful when both axes are in the same unit.
    An XY panel is x-pixels against y-pixels, so equal is right.  An XZ panel is
    x-*pixels* against z-*bin indices*: forcing equal there squashes the embryo
    into a one-pixel-tall pancake and makes it look flat, which is an artefact of
    the units, not of the sample.

    Args:
        z_aspect: how many xy pixels one z step is worth -- i.e. the voxel
            anisotropy.  Given it, XZ/YZ panels are drawn in true proportion.
            Without it they are left to autoscale, which fills the panel and
            distorts nothing silently.
    """
    ax.set_facecolor(theme.paper)
    ax.set_title(title, color=theme.font, fontsize=9)

    units = (_axis_unit(xlabel), _axis_unit(ylabel))
    suffix = {"xy_px": " (px)", "z_bin": " (z-bins)", "um": " (um)"}
    ax.set_xlabel(
        xlabel.replace("_reg", "").replace("_um", "") + suffix[units[0]],
        color=theme.axis_label, fontsize=8,
    )
    ax.set_ylabel(
        ylabel.replace("_reg", "").replace("_um", "") + suffix[units[1]],
        color=theme.axis_label, fontsize=8,
    )
    ax.tick_params(colors=theme.axis_label, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(theme.grid)

    if units[0] == units[1]:
        ax.set_aspect("equal", adjustable="datalim")
    elif z_aspect:
        # Mixed units, but the conversion factor is known: 1 z step = z_aspect px.
        ratio = z_aspect if units[1] == "z_bin" else 1.0 / z_aspect
        ax.set_aspect(ratio, adjustable="datalim")
    else:
        ax.set_aspect("auto")


def _save_fig(fig, theme: Theme, save_path: Optional[str | Path]) -> Optional[Path]:
    if save_path is None:
        return None
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, facecolor=theme.paper, bbox_inches="tight")
    print(f"  [SAVED] {save_path}")
    return save_path


def plot_additive_2d(
    frames,
    coord_sets: Optional[Sequence[Tuple[str, str, str]]] = None,
    style_of=None,
    mode: str = "dark",
    genes: Optional[Sequence[str]] = None,
    coords: Optional[Sequence[str]] = None,
    suptitle: str = "",
    size_scale: float = 2.0,
    stroke_color: str = "black",
    stroke_width: float = 0.3,
    silent_alpha: float = 0.85,
    silent_stroke_alpha: float = 0.4,
    colored_stroke_alpha: float = 1.0,
    save_path: Optional[str | Path] = None,
    panel_size: float = 5.0,
    z_aspect: Optional[float] = None,
    n_cols: Optional[int] = None,
    **style_kwargs,
):
    """Additive-overlay projections, one row per entry in ``frames``.

    Args:
        frames: a dataframe, or ``[(row_label, dataframe), ...]`` for one row each --
            the shape to use for stacking per-embryo rows or comparing atlases.
        coord_sets: ``[(cx, cy, panel_label), ...]`` column triples. Defaults to the
            XY/XZ/YZ projections of whichever coordinate set is present.
        style_of: precomputed :func:`additive_style` output -- a dict keyed by row
            label, or a single style dict. Pass this to style several figures
            identically (so colours and dot sizes are comparable between them);
            omit it and each row is styled from its own data.
        stroke_color / stroke_width: outline drawn around every nucleus. A thin
            outline is what keeps overlapping dim nuclei countable rather than
            merging into a wash.
        silent_alpha: fill opacity for below-threshold (grey) nuclei.
        silent_stroke_alpha / colored_stroke_alpha: outline opacity per layer.
            Silent nuclei keep a solid-ish outline so they read as tissue context;
            expressing nuclei get a soft one so the fill colour dominates.
        z_aspect: xy pixels per z step, so XZ/YZ are drawn in true proportion.
        n_cols: reflow the frames into a grid this many panels wide. Only allowed
            with a single projection, where frames-as-rows would otherwise make a
            one-panel-wide column -- a seven-embryo cohort in one XY view is a
            35-inch strip as rows and a readable 4x2 grid here.

    Colours, sizes and opacities are written as explicit per-point RGBA arrays rather
    than passed as scalar ``alpha=``: a scalar would apply one opacity to the whole
    layer, and the point of the silent/expressing split is that they differ.
    """
    import matplotlib
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    theme = theme_for(mode)
    dark = theme.is_dark

    # Accept a bare dataframe (+ a bare style dict) for the single-atlas case.
    if isinstance(frames, pd.DataFrame):
        frames = [("", frames)]
        if style_of is not None and not isinstance(style_of, dict):
            style_of = {"": style_of}
        elif isinstance(style_of, dict) and "rgb" in style_of:
            style_of = {"": style_of}
    frame_list = list(frames)

    first = frame_list[0][1]
    if coord_sets is None:
        coord_sets = _projection_cols(_resolve_coords(first, coords))
    gene_list = _resolve_genes(first, genes)
    hexes = gene_hex(gene_list)
    stroke_rgb = matplotlib.colors.to_rgb(stroke_color)

    # Panels in frame-major order, so one loop serves both layouts.
    panels = [(row_label, df, coord_set)
              for row_label, df in frame_list for coord_set in coord_sets]
    if n_cols is not None and len(coord_sets) > 1:
        raise ValueError(
            f"n_cols reflows frames into a grid and needs exactly one projection; "
            f"got {len(coord_sets)}"
        )
    if n_cols is None:
        grid_cols = len(coord_sets)
    else:
        grid_cols = max(1, min(int(n_cols), len(panels)))
    grid_rows = int(np.ceil(len(panels) / grid_cols))

    fig, axes = plt.subplots(
        grid_rows, grid_cols,
        figsize=(panel_size * grid_cols, panel_size * grid_rows),
        facecolor=theme.paper, squeeze=False,
    )
    flat_axes = [ax for row in axes for ax in row]
    for ax in flat_axes[len(panels):]:
        ax.set_visible(False)

    styled: Dict[str, Dict[str, object]] = {}
    for row_label, df in frame_list:
        if style_of is not None:
            styled[row_label] = style_of[row_label] if isinstance(style_of, dict) else style_of
        else:
            styled[row_label] = additive_style(df, genes, mode=mode, **style_kwargs)

    for index, (row_label, df, (cx, cy, panel)) in enumerate(panels):
        st = styled[row_label]
        rgb = np.asarray(st["rgb"])
        sizes = np.asarray(st["sizes"]) * size_scale
        hi = np.asarray(st["hi_mask"])

        # Silent first (translucent) so expressing nuclei, drawn opaque on top, are
        # never occluded by the grey background layer.
        layers = [(~hi, silent_alpha, silent_stroke_alpha),
                  (hi, 1.0, colored_stroke_alpha)]

        ax = flat_axes[index]
        ax.set_facecolor(theme.paper)
        # Positional throughout: the style arrays are positional, and a frame
        # sliced out of a concatenated table keeps its original labels.
        xv, yv = df[cx].to_numpy(), df[cy].to_numpy()

        for mask, fill_alpha, stroke_alpha in layers:
            k = int(mask.sum())
            if not k:
                continue
            face = np.column_stack([rgb[mask], np.full(k, fill_alpha)])
            edge = np.column_stack([np.tile(stroke_rgb, (k, 1)),
                                    np.full(k, stroke_alpha)])
            ax.scatter(xv[mask], yv[mask], c=face, s=sizes[mask],
                       edgecolors=edge, linewidths=stroke_width,
                       rasterized=dark)

        if row_label and len(coord_sets) > 1:
            title = f"{row_label} — {panel}"
        else:
            title = row_label or panel
        _style_axes(ax, theme, cx, cy, title, z_aspect=z_aspect)
        # Spines off on black (the panel edge competes with the data), light
        # grey on white where the panel needs a boundary.
        for spine in ax.spines.values():
            spine.set_visible(not dark)
            spine.set_edgecolor("#cccccc")

    handles = [
        mpatches.Patch(facecolor=hexes[g],
                       edgecolor="none" if dark else "#555555",
                       linewidth=0.5, label=g)
        for g in gene_list
    ]
    if handles:
        flat_axes[grid_cols - 1].legend(handles=handles, fontsize=8, loc="upper right",
                           facecolor=theme.paper, labelcolor=theme.font,
                           edgecolor="#cccccc", framealpha=0.9)
    if suptitle:
        fig.suptitle(suptitle, color=theme.font, fontsize=13)
    fig.tight_layout()
    _save_fig(fig, theme, save_path)
    return fig


def plot_additive_gene_2d(
    source,
    threshold=INTENSITY_THRESH,
    genes: Optional[Sequence[str]] = None,
    mode: str = "dark",
    coords: Optional[Sequence[str]] = None,
    require: str = "any",
    keep_silent: bool = True,
    projections: Sequence[str] = ("XY",),
    n_cols: Optional[int] = None,
    label: Optional[str] = None,
    suptitle: str = "",
    gamma: float = 0.35,
    quantile_clip: float = 0.95,
    size_range: Tuple[float, float] = (1.0, 14.0),
    gate_below_threshold: bool = True,
    style_kwargs: Optional[Dict[str, object]] = None,
    save_path: Optional[str | Path] = None,
    verbose: bool = True,
    **plot_kwargs,
):
    """Additive gene overlay for **either** registered embryos or an atlas, thresholded.

    The same call works on both spaces:

    >>> plot_additive_gene_2d(registered, threshold=0.05)          # rows = embryos
    >>> plot_additive_gene_2d(atlas, threshold="q0.9")             # one composite

    and that is the point -- a threshold is only interpretable next to the same
    threshold applied the same way to the other space, and the two spaces do **not**
    want the same number. Single-embryo intensity is a per-nucleus mean over assigned
    signal pixels; an atlas point is a k-nearest-neighbour average of those means, so
    kNN averaging pulls every point toward its local mean. That raises the floor
    (nuclei near a domain inherit some of it) and lowers the peaks, which is exactly
    the direction that makes a cut tuned on embryos call far too much of an atlas
    positive. Adding genes to the panel pushes the same way, because positivity is a
    union over channels and the signal mask is too.

    Sub-threshold nuclei are drawn as grey context by default, which keeps the shape
    of the embryo on the page -- without them a strict cut leaves a handful of dots
    floating in space with no anatomy to place them against. Pass
    ``keep_silent=False`` to drop them instead, which is the view that makes a
    too-low threshold obvious: greyed-out nuclei still fill the panel, so an
    over-inclusive cut can look like a healthy dense embryo either way.

    Args:
        source: an :class:`~register_embryos.atlas.Atlas`, a registered nucleus table
            (split into one row per ``embryo_id``), a single frame, a
            ``{label: frame}`` mapping, or ``[(label, frame), ...]``.
        threshold: any spec :func:`~register_embryos.thresholds.resolve_gene_cuts`
            takes -- a number, a ``{gene: cut}`` mapping, the ``results`` dict from
            :func:`~register_embryos.thresholds.call_thresholds`, ``"otsu"``, or a
            positive-rate quantile like ``"q0.9"``. String specs are recomputed per
            frame, so each embryo (or the atlas) gets its own cut.
        require: ``"any"`` keeps nuclei positive for at least one gene -- "drop the
            nuclei that lack signal in every channel". ``"all"`` keeps only nuclei
            positive for every gene measured in that frame.
        projections: which panels to draw, from ``"XY"``/``"XZ"``/``"YZ"``. XY only by
            default: a 12s dorsal mount is a few z-bins thick, so XZ and YZ are mostly
            a statement about section thickness.
        n_cols: with one projection, reflow the frames into a grid this wide instead
            of a single tall column. Defaults to 4 for a multi-embryo cohort.
        gate_below_threshold: on by default here (and off in
            :func:`plot_additive_2d`, which keeps the original notebook behaviour):
            a nucleus is tinted only by the genes it is positive for. Without it the
            threshold decides only whether to colour a nucleus while all eight
            channels still tint it, which is what turns an eight-gene atlas into a
            wash of blended hues at every cut.
        gamma / quantile_clip / size_range: passed to :func:`additive_style`, which is
            computed on the **full** frame before dropping, so dot sizes stay
            comparable between a thresholded figure and its ``keep_silent=False`` twin.
        style_kwargs: anything else :func:`additive_style` takes -- ``color_scale``,
            ``color_gamma``, ``min_saturation``, ``alpha_low``. A dict rather than
            ``**kwargs`` because the loose keywords here go to
            :func:`plot_additive_2d`.

    Prints the positive fraction per frame and per gene;
    :func:`~register_embryos.thresholds.positive_fraction` returns the same table
    without drawing anything, which is the cheaper way to pick a cut.
    """
    from .thresholds import as_frames, positive_calls, resolve_gene_cuts

    style_kwargs = dict(style_kwargs or {})
    frames = as_frames(source, label=label)
    if not frames:
        raise ValueError("nothing to plot: no frames in source")

    wanted = {str(p).upper() for p in projections}
    coord_cols = _resolve_coords(frames[0][1], coords)
    coord_sets = [c for c in _projection_cols(coord_cols) if c[2] in wanted]
    if not coord_sets:
        raise ValueError(
            f"no panels left: projections={tuple(projections)} matched none of "
            f"'XY', 'XZ', 'YZ'"
        )
    gene_list = _resolve_genes(frames[0][1], genes)

    kept_frames: List[Tuple[str, pd.DataFrame]] = []
    styles: Dict[str, Dict[str, object]] = {}
    rows = []
    for frame_label, frame in frames:
        cuts = resolve_gene_cuts(frame, gene_list, threshold, embryo_id=frame_label)
        keep, per_gene, measured = positive_calls(frame, gene_list, cuts, require=require)
        if keep_silent:
            keep = np.ones(len(frame), dtype=bool)

        style = additive_style(
            frame, gene_list, mode=mode, threshold=cuts, gamma=gamma,
            quantile_clip=quantile_clip, size_range=size_range,
            gate_below_threshold=gate_below_threshold, **style_kwargs,
        )
        styles[frame_label] = {
            key: (np.asarray(value)[keep] if key in ("rgb", "hex", "sizes", "alpha",
                                                     "summed", "hi_mask")
                  else value)
            for key, value in style.items()
        }
        kept_frames.append((frame_label, frame.iloc[keep]))

        n = len(frame)
        for column, gene in enumerate(measured):
            n_pos = int(per_gene[:, column].sum())
            rows.append({"label": frame_label, "gene": gene, "threshold": cuts[gene],
                         "n": n, "n_positive": n_pos,
                         "positive_fraction": n_pos / n if n else np.nan})
        rows.append({"label": frame_label, "gene": "<kept>", "threshold": np.nan,
                     "n": n, "n_positive": int(keep.sum()),
                     "positive_fraction": keep.sum() / n if n else np.nan})

    if verbose:
        spec = threshold if isinstance(threshold, (str, float, int)) else "per-gene"
        print(f"  [ADDITIVE] threshold={spec}, require={require}, "
              f"{'keeping' if keep_silent else 'dropping'} sub-threshold nuclei")
        for row in rows:
            cut = "" if not np.isfinite(row["threshold"]) else f"thr={row['threshold']:.4f}  "
            print(f"    {str(row['label'])[:44]:44s} {row['gene']:8s} {cut}"
                  f"{row['n_positive']:7,d}/{row['n']:,} = {row['positive_fraction']:6.1%}")
        empty = [label for label, frame in kept_frames if frame.empty]
        if empty:
            print(f"    [WARN] threshold left nothing to draw for: {', '.join(empty)}")

    if n_cols is not None and len(coord_sets) > 1:
        raise ValueError(
            f"n_cols reflows frames into a grid and needs exactly one projection; "
            f"got {len(coord_sets)} ({', '.join(c[2] for c in coord_sets)})"
        )
    if n_cols is None and len(coord_sets) == 1 and len(kept_frames) > 1:
        n_cols = min(4, len(kept_frames))
    return plot_additive_2d(
        kept_frames, coord_sets=coord_sets, style_of=styles, mode=mode,
        genes=gene_list, suptitle=suptitle, save_path=save_path,
        n_cols=n_cols, **plot_kwargs,
    )


def plot_gene_panels_2d(
    df: pd.DataFrame,
    genes: Optional[Sequence[str]] = None,
    mode: str = "dark",
    coords: Optional[Sequence[str]] = None,
    projection: Tuple[int, int] = (0, 1),
    suptitle: str = "",
    threshold: float = INTENSITY_THRESH,
    save_path: Optional[str | Path] = None,
    panel_size: float = 4.0,
    z_aspect: Optional[float] = None,
):
    """One panel per gene in a single projection, silent nuclei as grey context."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    matplotlib.rcParams["pdf.fonttype"] = 42
    theme = theme_for(mode)
    coord_cols = _resolve_coords(df, coords)
    cx, cy = coord_cols[projection[0]], coord_cols[projection[1]]
    gene_list = _resolve_genes(df, genes)

    fig, axes = plt.subplots(
        1, len(gene_list),
        figsize=(panel_size * len(gene_list), panel_size),
        facecolor=theme.paper, squeeze=False,
    )
    for column, gene in enumerate(gene_list):
        ax = axes[0][column]
        cmap = LinearSegmentedColormap.from_list(
            f"{gene}_ramp", [_hex(theme.silent_rgb), _hex(gene_color(gene, column))]
        )
        positive = (df[gene].fillna(0) >= threshold).to_numpy()
        x_all, y_all = df[cx].to_numpy(), df[cy].to_numpy()
        values = df[gene].fillna(0).to_numpy()
        ax.scatter(
            x_all[~positive], y_all[~positive],
            c=[theme.silent_rgb], s=2, alpha=0.35, rasterized=True,
        )
        if positive.any():
            cap = float(np.quantile(values[positive], 0.99)) or 1.0
            scatter = ax.scatter(
                x_all[positive], y_all[positive],
                c=values[positive], cmap=cmap, vmin=threshold, vmax=cap,
                s=7, alpha=0.9, rasterized=True,
            )
            bar = fig.colorbar(scatter, ax=ax, shrink=0.75)
            bar.ax.tick_params(colors=theme.axis_label, labelsize=6)
        _style_axes(ax, theme, cx, cy, f"{gene} — {int(positive.sum()):,} positive",
                    z_aspect=z_aspect)

    if suptitle:
        fig.suptitle(suptitle, color=theme.font, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    _save_fig(fig, theme, save_path)
    return fig


def plot_gene_by_embryo(
    registered: pd.DataFrame,
    genes: Optional[Sequence[str]] = None,
    mode: str = "light",
    coords: Optional[Sequence[str]] = None,
    projection: Tuple[int, int] = (0, 1),
    threshold=INTENSITY_THRESH,
    suptitle: str = "",
    panel_size: float = 2.9,
    point_size: float = 5.0,
    quantile_cap: float = 0.99,
    shared_scale: bool = True,
    z_aspect: Optional[float] = None,
    save_path: Optional[str | Path] = None,
):
    """Grid of embryos (rows) x genes (columns) in registered space.

    The view for judging a registration by the thing you actually care about, one
    embryo at a time, *before* the atlas averages them together. An atlas hides
    disagreement by construction -- it takes a mean -- so a domain that lands in a
    different place in one embryo shows up here and nowhere later.

    A gene absent from an embryo's panel leaves its cell blank rather than an empty
    axis, which keeps the rows readable when panels differ across the cohort (as they
    do whenever a rotating-partner design is used).

    Args:
        threshold: a scalar, or a ``{gene: cut}`` mapping, or the ``results`` dict
            from :func:`~register_embryos.thresholds.call_thresholds` (per-embryo cuts
            are then used for the matching embryo).
        shared_scale: put every embryo on one colour scale per gene, so panels are
            comparable down a column. Off scales each panel to its own maximum, which
            shows the pattern in a dim embryo at the cost of comparability.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    matplotlib.rcParams["pdf.fonttype"] = 42
    theme = theme_for(mode)
    coord_cols = _resolve_coords(registered, coords)
    cx, cy = coord_cols[projection[0]], coord_cols[projection[1]]
    gene_list = _resolve_genes(registered, genes)
    embryos = list(registered["embryo_id"].unique())

    def cut_for(embryo_id: str, gene: str) -> float:
        if isinstance(threshold, dict):
            keyed = threshold.get((embryo_id, gene))
            if keyed is not None:
                return float(getattr(keyed, "threshold", keyed))
            plain = threshold.get(gene)
            if plain is not None:
                return float(getattr(plain, "threshold", plain))
            return INTENSITY_THRESH
        return float(threshold)

    caps = {}
    for gene in gene_list:
        values = registered[gene].dropna()
        caps[gene] = float(values.quantile(quantile_cap)) if len(values) else 1.0

    fig, axes = plt.subplots(
        len(embryos), len(gene_list),
        figsize=(panel_size * len(gene_list), panel_size * len(embryos)),
        facecolor=theme.paper, squeeze=False,
    )
    for row, embryo_id in enumerate(embryos):
        sub = registered[registered["embryo_id"] == embryo_id]
        for col, gene in enumerate(gene_list):
            ax = axes[row][col]
            ax.set_facecolor(theme.paper)
            values = sub[gene]
            if values.isna().all():
                # Not in this embryo's panel: say so instead of drawing an empty box.
                ax.text(0.5, 0.5, "not in panel", ha="center", va="center",
                        transform=ax.transAxes, color=theme.axis_label, fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color(theme.grid)
                if row == 0:
                    ax.set_title(gene, color=theme.font, fontsize=10)
                continue

            cut = cut_for(embryo_id, gene)
            filled = values.fillna(0).to_numpy()
            positive = filled >= cut
            x_all, y_all = sub[cx].to_numpy(), sub[cy].to_numpy()

            ax.scatter(x_all[~positive], y_all[~positive], c=[theme.silent_rgb],
                       s=point_size * 0.45, alpha=0.35, rasterized=True, linewidths=0)
            if positive.any():
                cmap = LinearSegmentedColormap.from_list(
                    f"{gene}_ramp",
                    [_hex(theme.silent_rgb), _hex(gene_color(gene, col))],
                )
                cap = caps[gene] if shared_scale else float(filled[positive].max())
                ax.scatter(x_all[positive], y_all[positive], c=filled[positive],
                           cmap=cmap, vmin=cut, vmax=max(cap, cut + 1e-6),
                           s=point_size, alpha=0.95, rasterized=True,
                           linewidths=0.15,
                           edgecolors=matplotlib.colors.to_rgba(theme.stroke, 0.5))

            _style_axes(ax, theme, cx, cy, "", z_aspect=z_aspect)
            ax.set_xlabel(""); ax.set_ylabel("")
            ax.set_xticks([]); ax.set_yticks([])
            n_pos = int(positive.sum())
            ax.text(0.02, 0.98, f"{n_pos:,}  ({n_pos/len(sub):.0%})",
                    transform=ax.transAxes, va="top", ha="left",
                    color=theme.font, fontsize=7)
            if row == 0:
                ax.set_title(gene, color=theme.font, fontsize=10)

        short = embryo_id.split("_")
        axes[row][0].set_ylabel("_".join(short[:2]), color=theme.font, fontsize=7)

    if suptitle:
        fig.suptitle(suptitle, color=theme.font, fontsize=12)
        # Reserve space, or the suptitle lands on top of the first row's titles.
        fig.tight_layout(rect=(0, 0, 1, 0.98 if len(embryos) > 3 else 0.95))
    else:
        fig.tight_layout()
    _save_fig(fig, theme, save_path)
    return fig


def plot_registration_2d(
    registered: pd.DataFrame,
    reference_embryo_id: str,
    mode: str = "light",
    embryo_ids: Optional[Iterable[str]] = None,
    suptitle: str = "",
    save_path: Optional[str | Path] = None,
    panel_size: float = 3.6,
    reference_color: str = "#1f1f1f",
    embryo_color: str = "#FF4500",
    z_aspect: Optional[float] = None,
):
    """QC view: each embryo against the reference, before and after ICP.

    Six panels per embryo -- XY/XZ/YZ raw, then XY/XZ/YZ registered.  This is the
    plot that tells you whether a fit actually worked; a residual number alone
    will not reveal an embryo that converged to a plausible-looking wrong pose.
    """
    import matplotlib.pyplot as plt

    theme = theme_for(mode)
    ids = list(embryo_ids) if embryo_ids is not None else [
        e for e in registered["embryo_id"].unique() if e != reference_embryo_id
    ]
    if not ids:
        raise ValueError("nothing to plot: only the reference embryo is present")

    reference = registered[registered["embryo_id"] == reference_embryo_id]
    fig, axes = plt.subplots(
        len(ids), 6,
        figsize=(panel_size * 6, panel_size * len(ids)),
        facecolor=theme.paper, squeeze=False,
    )
    for row, embryo_id in enumerate(ids):
        subset = registered[registered["embryo_id"] == embryo_id]
        panels = [
            *[(a, b, f"{name} raw", ("x", "y", "z")) for a, b, name in PROJECTIONS],
            *[
                (f"{a}_reg", f"{b}_reg", f"{name} registered", ("x_reg", "y_reg", "z_reg"))
                for a, b, name in PROJECTIONS
            ],
        ]
        for column, (cx, cy, panel_title, ref_cols) in enumerate(panels):
            ax = axes[row][column]
            ref_x, ref_y = (ref_cols[0], ref_cols[1]) if "XY" in panel_title else (
                (ref_cols[0], ref_cols[2]) if "XZ" in panel_title else (ref_cols[1], ref_cols[2])
            )
            ax.scatter(
                reference[ref_x], reference[ref_y],
                c=reference_color, s=1.5, alpha=0.25, rasterized=True, label="reference",
            )
            ax.scatter(
                subset[cx], subset[cy],
                c=embryo_color, s=1.5, alpha=0.45, rasterized=True, label=embryo_id,
            )
            _style_axes(ax, theme, cx, cy, panel_title, z_aspect=z_aspect)
        axes[row][0].set_ylabel(
            embryo_id.replace("_", "\n"), color=theme.font, fontsize=7
        )

    if suptitle:
        fig.suptitle(suptitle, color=theme.font, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    _save_fig(fig, theme, save_path)
    return fig
