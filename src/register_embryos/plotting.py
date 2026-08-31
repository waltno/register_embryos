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
    "plot_registration_2d",
    "plot_gene_panels_2d",
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

INTENSITY_THRESH = 0.05


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
    threshold: float = INTENSITY_THRESH,
    size_range: Tuple[float, float] = (1.0, 14.0),
    alpha_low: float = 0.15,
) -> Dict[str, object]:
    """Per-nucleus colour, size and opacity from the additive channel mix.

    Hue comes from the channel mix normalised to full brightness, so a nucleus
    expressing one dim gene still shows that gene's hue rather than fading to
    black; total intensity drives size and opacity instead.  Separating hue from
    brightness is what keeps a three-colour overlay readable.

    A nucleus counts as expressing when at least ONE channel clears
    ``threshold``.  Not the channel sum: summing lets a nucleus with three
    sub-threshold channels look positive while being positive for nothing, so
    "coloured" and "positive for some gene" would stop agreeing.

    Args:
        gamma: < 1 lifts dim nuclei.  0.35 is a moderate lift; the original
            notebooks used 0.1, which pushes nearly everything to full size.
        quantile_clip: intensity at this quantile of the channel sum maps to
            maximum size, so a few saturated nuclei do not flatten the rest.
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
    mixed = np.zeros((n, 3))
    for column, gene in enumerate(gene_list):
        mixed += np.outer(values[:, column], gene_color(gene, column))

    summed = values.sum(axis=1)
    ceiling = float(np.quantile(summed, quantile_clip))
    intensity = np.clip(summed / (ceiling + 1e-9), 0, 1) ** gamma

    clipped = np.clip(mixed, 0, 1)
    peak = clipped.max(axis=1, keepdims=True)
    bright = clipped / np.where(peak > 1e-9, peak, 1.0)

    hi_mask = (values >= threshold).any(axis=1)
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
    mode: str = "dark",
    genes: Optional[Sequence[str]] = None,
    coords: Optional[Sequence[str]] = None,
    suptitle: str = "",
    size_scale: float = 1.6,
    save_path: Optional[str | Path] = None,
    panel_size: float = 4.2,
    z_aspect: Optional[float] = None,
    **style_kwargs,
):
    """Additive overlay as XY/XZ/YZ projections, one row per frame.

    Args:
        frames: a dataframe, or ``[(label, dataframe), ...]`` for one row each --
            the shape to use for stacking per-embryo rows or comparing atlases.
        z_aspect: xy pixels per z step (the voxel anisotropy), so the XZ/YZ panels
            are drawn in true proportion.  Without it they autoscale -- x is in
            pixels and z in bin indices, and forcing equal aspect across those two
            makes the embryo look flat.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    theme = theme_for(mode)
    if isinstance(frames, pd.DataFrame):
        frames = [("", frames)]
    frame_list = list(frames)

    first = frame_list[0][1]
    coord_cols = _resolve_coords(first, coords)
    projections = _projection_cols(coord_cols)
    hexes = gene_hex(_resolve_genes(first, genes))

    fig, axes = plt.subplots(
        len(frame_list), len(projections),
        figsize=(panel_size * len(projections), panel_size * len(frame_list)),
        facecolor=theme.paper, squeeze=False,
    )
    for row, (label, df) in enumerate(frame_list):
        st = additive_style(df, genes, mode=mode, **style_kwargs)
        rgb = np.asarray(st["rgb"])
        sizes = np.asarray(st["sizes"]) * size_scale
        hi = np.asarray(st["hi_mask"])

        for column, (cx, cy, panel) in enumerate(projections):
            ax = axes[row][column]
            # Silent first (translucent) so expressing nuclei on top are never
            # hidden behind the grey background layer.
            # Positional indexing throughout: the style arrays are positional, and
            # a frame sliced out of a concatenated table keeps its original labels,
            # so .loc with positional integers raises (or worse, silently
            # mismatches rows against their colours).
            x_all = df[cx].to_numpy()
            y_all = df[cy].to_numpy()
            for mask, alpha, stroke_alpha in ((~hi, 0.55, 0.35), (hi, 1.0, 0.9)):
                if not mask.any():
                    continue
                ax.scatter(
                    x_all[mask], y_all[mask],
                    c=rgb[mask], s=sizes[mask], alpha=alpha,
                    linewidths=0.25,
                    edgecolors=matplotlib.colors.to_rgba(theme.stroke, stroke_alpha),
                    rasterized=True,
                )
            title = f"{label} — {panel}" if label else panel
            _style_axes(ax, theme, cx, cy, title, z_aspect=z_aspect)

    handles = [
        matplotlib.patches.Patch(color=colour, label=gene) for gene, colour in hexes.items()
    ]
    if handles:
        axes[0][-1].legend(
            handles=handles, fontsize=7, loc="upper right",
            facecolor=theme.scene, labelcolor=theme.font, framealpha=0.85,
        )
    if suptitle:
        fig.suptitle(suptitle, color=theme.font, fontsize=12)
    fig.tight_layout()
    _save_fig(fig, theme, save_path)
    return fig


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
    fig.tight_layout()
    _save_fig(fig, theme, save_path)
    return fig
