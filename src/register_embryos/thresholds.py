"""Deciding what counts as expressing, from the data rather than from a constant.

A fixed cut (the pipeline default is 0.05) is easy to reason about and wrong in a
specific way: it is applied to *contrast-normalised* intensity, and contrast is set
per embryo by eye. So the same number means a different physical brightness in every
embryo, and the fraction called positive drifts with how each window happened to be
placed. Adding a broadly-expressed gene to a panel makes this worse, because the
signal mask is a union over gene channels -- more territory is retained as "measured",
and every channel's per-nucleus mean shifts.

The alternative here is to find each channel's own off/on split from its per-nucleus
intensity distribution, per embryo and per gene:

``otsu``
    The value maximising between-group variance. Assumption-free, fast, and the same
    routine already used to find the midline gap.
``gmm``
    Two-component Gaussian mixture on log-intensity, thresholded where the posteriors
    cross. Gives a per-nucleus probability rather than only a hard call, which is what
    you want for a graded readout.
``quantile``
    A fixed positive rate. Only honest when you already know the rate; included
    because it is sometimes the right comparison to make across embryos.
``fixed``
    The constant, for reproducing earlier work.

Every method returns a :class:`ThresholdResult` carrying a **separation score**, so a
channel whose distribution is unimodal -- a gene that is simply off, or a channel that
is all background -- is flagged instead of being silently cut in half.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "ThresholdResult",
    "otsu_threshold",
    "gmm_threshold",
    "call_thresholds",
    "apply_thresholds",
    "threshold_sweep",
    "plot_threshold_diagnostics",
]

#: Below this valley depth a split is not trustworthy.
#:
#: Calibrated on this project's real cohort, where the scores run 0.0-0.75 with most
#: channels at 0.1-0.4.  That spread is itself the finding: HCR per-nucleus intensity
#: is **not** cleanly bimodal.  A large share of nuclei (21-98% depending on the
#: gene) sit at exactly zero because no signal pixel fell in their territory, and the
#: remainder is a graded continuum rather than an off population and an on population.
#: So "trustworthy" here means "there is at least a shallow valley to cut at", not
#: "this gene is cleanly on or off".  Report :func:`threshold_sweep` alongside any
#: threshold-dependent conclusion.
MIN_SEPARATION = 0.15


@dataclass
class ThresholdResult:
    """One channel's threshold, how it was reached, and whether to believe it."""

    embryo_id: str
    gene: str
    threshold: float
    method: str
    n: int
    positive_fraction: float
    separation: float
    fell_back: bool = False
    note: str = ""
    #: Share of nuclei with no measurement at all (no signal pixel in their
    #: territory). These are unambiguous negatives; a high value means the gene is
    #: sparse, and also that the intensity histogram is dominated by a spike at zero.
    zero_fraction: float = 0.0

    @property
    def trustworthy(self) -> bool:
        return self.separation >= MIN_SEPARATION and not self.fell_back

    def as_row(self) -> Dict[str, object]:
        return {
            "embryo_id": self.embryo_id, "gene": self.gene,
            "threshold": self.threshold, "method": self.method, "n": self.n,
            "positive_fraction": self.positive_fraction,
            "separation": self.separation, "zero_fraction": self.zero_fraction,
            "fell_back": self.fell_back,
            "trustworthy": self.trustworthy, "note": self.note,
        }


def _clean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def otsu_threshold(
    values: np.ndarray, n_bins: int = 256, ignore_zeros: bool = True
) -> Tuple[float, float]:
    """Otsu split of a 1-D distribution. Returns ``(threshold, separation)``.

    Args:
        ignore_zeros: drop values at (or below) zero before splitting. A nucleus reads
            exactly zero when no signal pixel fell in its territory -- an unambiguous
            negative, not a measurement -- and in this data that is 21-98% of nuclei
            depending on the gene. Left in, that spike is the dominant mode, and Otsu
            ends up separating "zero" from "everything else" rather than off from on.

    ``separation`` is the valley depth at the cut; see :func:`_valley_depth`.
    """
    values = _clean(values)
    if ignore_zeros:
        nonzero = values[values > 0]
        if nonzero.size >= 50:
            values = nonzero
    if values.size < 20 or np.allclose(values, values[0]):
        return float("nan"), 0.0

    counts, edges = np.histogram(values, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight0 = counts.cumsum()[:-1]
    weight1 = counts.sum() - weight0
    cumulative = (counts * centers).cumsum()
    mean0 = cumulative[:-1] / np.maximum(weight0, 1)
    mean1 = (cumulative[-1] - cumulative[:-1]) / np.maximum(weight1, 1)
    between = np.where((weight0 > 0) & (weight1 > 0),
                       weight0 * weight1 * (mean0 - mean1) ** 2, -1.0)
    if between.max() <= 0:
        return float("nan"), 0.0

    index = int(np.argmax(between))
    threshold = float(edges[1:-1][index])
    return threshold, _valley_depth(counts, index + 1)


def _valley_depth(counts: np.ndarray, index: int, smooth: int = 5) -> float:
    """How much of a valley sits at ``index`` -- 0 for one group, ~1 for two.

    Otsu's own between-group variance is NOT usable as a separation score: splitting
    any distribution at its best point explains a large share of its variance (about
    64% for a plain Gaussian), so a unimodal channel scores as highly as a genuinely
    bimodal one. Measured that way a background-only channel looked separable, which
    is exactly the case the score exists to catch.

    Valley depth asks the right question instead: at the chosen cut, how far does the
    density fall below the smaller of the two modes it is supposed to separate? A real
    off/on split has a near-empty valley; a cut through a single hump has the mode
    itself sitting on one side of it, so the depth collapses to zero.
    """
    if counts.size < 3 or index <= 0 or index >= counts.size:
        return 0.0
    kernel = np.ones(max(1, smooth)) / max(1, smooth)
    smoothed = np.convolve(counts.astype(float), kernel, mode="same")

    left_peak = smoothed[:index].max() if index > 0 else 0.0
    right_peak = smoothed[index:].max() if index < smoothed.size else 0.0
    valley = smoothed[index]
    shallower = min(left_peak, right_peak)
    if shallower <= 0:
        return 0.0
    return float(np.clip(1.0 - valley / shallower, 0.0, 1.0))


def gmm_threshold(
    values: np.ndarray, floor: float = 1e-4, seed: int = 0
) -> Tuple[float, float, Optional[object]]:
    """Two-component Gaussian mixture on log intensity.

    Intensities are heavily right-skewed with a spike at zero, so the mixture is fitted
    in log space on the nonzero values; the threshold is where the two posteriors
    cross. Returns ``(threshold, separation, model)``; separation is the standardised
    distance between component means, scaled into [0, 1].
    """
    values = _clean(values)
    positive = values[values > floor]
    if positive.size < 50:
        return float("nan"), 0.0, None
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return float("nan"), 0.0, None

    logged = np.log10(positive).reshape(-1, 1)
    model = GaussianMixture(n_components=2, random_state=seed, n_init=3).fit(logged)
    order = np.argsort(model.means_.ravel())
    low, high = model.means_.ravel()[order]
    sd_low, sd_high = np.sqrt(model.covariances_.ravel()[order])

    grid = np.linspace(logged.min(), logged.max(), 2000).reshape(-1, 1)
    posterior = model.predict_proba(grid)[:, order[1]]
    crossings = np.nonzero(np.diff(np.sign(posterior - 0.5)))[0]
    if crossings.size == 0:
        return float("nan"), 0.0, model
    threshold = float(10 ** grid[crossings[0], 0])

    # Standardised distance between components, mapped onto the same [0,1] scale
    # as the valley depth so MIN_SEPARATION means one thing whichever method ran.
    standardised = abs(high - low) / np.sqrt(0.5 * (sd_low**2 + sd_high**2))
    return threshold, float(np.clip((standardised - 1.0) / 3.0, 0, 1)), model


def call_thresholds(
    table: pd.DataFrame,
    genes: Optional[Sequence[str]] = None,
    method: str = "otsu",
    per_embryo: bool = True,
    fallback: float = 0.05,
    min_separation: float = MIN_SEPARATION,
    quantile: float = 0.9,
    ignore_zeros: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[Tuple[str, str], ThresholdResult], pd.DataFrame]:
    """A threshold per ``(embryo, gene)``, with a diagnostic table.

    Args:
        per_embryo: thresholds per embryo (the default, and the right choice while
            contrast is set per embryo by eye). ``False`` pools the cohort, which is
            only defensible once intensities are comparable across embryos.
        fallback: used, and flagged, when a channel's distribution does not separate.
        ignore_zeros: exclude nuclei with no measurement at all before splitting; see
            :func:`otsu_threshold`.

    Returns ``(results, diagnostics)``. **Read the diagnostics**: a channel with
    ``trustworthy=False`` had no usable split, which usually means the gene is simply
    off in that embryo -- a legitimate answer, but not one a data-driven cut can give
    you on its own.
    """
    skip = {"embryo_id", "nucleus_id", "atlas_point_id", "n_voxels", "n_source_embryos",
            "neighbor_radius", "x", "y", "z", "x_reg", "y_reg", "z_reg",
            "x_um", "y_um", "z_um"}
    gene_list = list(genes) if genes is not None else [
        c for c in table.columns
        if c not in skip and pd.api.types.is_numeric_dtype(table[c])
    ]

    groups = (
        table.groupby("embryo_id") if per_embryo and "embryo_id" in table.columns
        else [("<cohort>", table)]
    )

    results: Dict[Tuple[str, str], ThresholdResult] = {}
    for embryo_id, sub in groups:
        for gene in gene_list:
            if gene not in sub.columns:
                continue
            values = _clean(sub[gene].to_numpy())
            if values.size == 0:
                continue

            note, fell_back = "", False
            if method == "otsu":
                threshold, separation = otsu_threshold(values, ignore_zeros=ignore_zeros)
            elif method == "gmm":
                threshold, separation, _ = gmm_threshold(values)
                if not np.isfinite(threshold):
                    note = "gmm unavailable or did not converge; used otsu"
                    threshold, separation = otsu_threshold(
                        values, ignore_zeros=ignore_zeros
                    )
            elif method == "quantile":
                threshold = float(np.quantile(values, quantile))
                separation = 1.0
                note = f"fixed positive rate {1 - quantile:.0%}"
            elif method == "fixed":
                threshold, separation = float(fallback), 1.0
                note = "constant"
            else:
                raise ValueError(
                    f"unknown method {method!r} (expected 'otsu', 'gmm', "
                    f"'quantile' or 'fixed')"
                )

            if not np.isfinite(threshold) or separation < min_separation:
                note = note or (
                    f"separation {separation:.3f} < {min_separation}; distribution is "
                    f"not two groups (gene likely off here)"
                )
                threshold, fell_back = float(fallback), True

            results[(embryo_id, gene)] = ThresholdResult(
                embryo_id=str(embryo_id), gene=gene, threshold=float(threshold),
                method=method, n=int(values.size),
                positive_fraction=float((values >= threshold).mean()),
                separation=float(separation), fell_back=fell_back, note=note,
                zero_fraction=float((values <= 0).mean()),
            )

    diagnostics = pd.DataFrame([r.as_row() for r in results.values()])
    if verbose and not diagnostics.empty:
        print(f"  [THRESHOLD] method={method}, per_embryo={per_embryo}")
        for _, row in diagnostics.iterrows():
            flag = "" if row["trustworthy"] else "   <- not trustworthy"
            print(f"    {str(row['embryo_id'])[:44]:44s} {row['gene']:8s} "
                  f"thr={row['threshold']:.4f}  sep={row['separation']:.3f}  "
                  f"{row['positive_fraction']:6.1%} positive{flag}")
        untrustworthy = int((~diagnostics["trustworthy"]).sum())
        if untrustworthy:
            print(f"    [WARN] {untrustworthy}/{len(diagnostics)} channel(s) fell back "
                  f"to {fallback}; see the note column")
    return results, diagnostics


def apply_thresholds(
    table: pd.DataFrame,
    results: Dict[Tuple[str, str], ThresholdResult],
    suffix: str = "_pos",
) -> pd.DataFrame:
    """Add a boolean ``<gene>_pos`` column per gene using its own threshold."""
    out = table.copy()
    genes = sorted({gene for _, gene in results})
    for gene in genes:
        column = np.zeros(len(out), dtype=bool)
        if "embryo_id" in out.columns:
            for embryo_id in out["embryo_id"].unique():
                result = results.get((embryo_id, gene)) or results.get(("<cohort>", gene))
                if result is None or gene not in out.columns:
                    continue
                mask = (out["embryo_id"] == embryo_id).to_numpy()
                column |= mask & (out[gene].fillna(-1).to_numpy() >= result.threshold)
        else:
            result = results.get(("<cohort>", gene))
            if result is not None and gene in out.columns:
                column = out[gene].fillna(-1).to_numpy() >= result.threshold
        out[f"{gene}{suffix}"] = column
    return out


def threshold_sweep(
    table: pd.DataFrame,
    genes: Optional[Sequence[str]] = None,
    thresholds: Sequence[float] = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5),
    per_embryo: bool = False,
) -> pd.DataFrame:
    """Positive fraction across a range of cuts.

    The honest way to report a threshold-dependent number: show that the conclusion
    survives the choice, or say that it does not.
    """
    skip = {"embryo_id", "nucleus_id", "atlas_point_id", "n_voxels", "n_source_embryos",
            "neighbor_radius", "x", "y", "z", "x_reg", "y_reg", "z_reg",
            "x_um", "y_um", "z_um"}
    gene_list = list(genes) if genes is not None else [
        c for c in table.columns
        if c not in skip and pd.api.types.is_numeric_dtype(table[c])
    ]
    groups = (
        table.groupby("embryo_id") if per_embryo and "embryo_id" in table.columns
        else [("<cohort>", table)]
    )
    rows = []
    for embryo_id, sub in groups:
        for gene in gene_list:
            if gene not in sub.columns:
                continue
            values = _clean(sub[gene].to_numpy())
            if values.size == 0:
                continue
            for threshold in thresholds:
                rows.append({
                    "embryo_id": embryo_id, "gene": gene, "threshold": threshold,
                    "positive_fraction": float((values >= threshold).mean()),
                    "n_positive": int((values >= threshold).sum()), "n": int(values.size),
                })
    return pd.DataFrame(rows)


def plot_threshold_diagnostics(
    table: pd.DataFrame,
    results: Dict[Tuple[str, str], ThresholdResult],
    genes: Optional[Sequence[str]] = None,
    mode: str = "light",
    log_x: bool = True,
    save_path: Optional[str] = None,
    panel_size: float = 3.2,
):
    """Per-gene intensity histograms with the chosen cut drawn on.

    One row per embryo, one column per gene. Look for two humps with the line in the
    valley; a single hump with a line through it is the fallback case and the panel is
    titled to say so.
    """
    import matplotlib.pyplot as plt

    from .plotting import theme_for

    theme = theme_for(mode)
    gene_list = list(genes) if genes is not None else sorted({g for _, g in results})
    embryos = sorted({e for e, _ in results})

    fig, axes = plt.subplots(
        len(embryos), len(gene_list),
        figsize=(panel_size * len(gene_list), panel_size * len(embryos)),
        facecolor=theme.paper, squeeze=False,
    )
    for row, embryo_id in enumerate(embryos):
        sub = (
            table[table["embryo_id"] == embryo_id]
            if "embryo_id" in table.columns and embryo_id != "<cohort>" else table
        )
        for col, gene in enumerate(gene_list):
            ax = axes[row][col]
            ax.set_facecolor(theme.paper)
            result = results.get((embryo_id, gene))
            if result is None or gene not in sub.columns:
                ax.axis("off")
                continue
            values = _clean(sub[gene].to_numpy())
            plot_values = values[values > 1e-4] if log_x else values
            if plot_values.size:
                ax.hist(np.log10(plot_values) if log_x else plot_values,
                        bins=80, color="#4477aa")
            cut = np.log10(result.threshold) if (log_x and result.threshold > 0) else result.threshold
            ax.axvline(cut, color="#cc3311", ls="--", lw=1.5)
            ax.set_yscale("log")
            title = (f"{gene} — thr {result.threshold:.3f}\n"
                     f"{result.positive_fraction:.1%} pos, sep {result.separation:.2f}")
            if not result.trustworthy:
                title += "  (fallback)"
            ax.set_title(title, color=theme.font, fontsize=8)
            ax.tick_params(colors=theme.axis_label, labelsize=6)
            ax.set_xlabel("log10 intensity" if log_x else "intensity",
                          color=theme.axis_label, fontsize=7)
            for spine in ax.spines.values():
                spine.set_color(theme.grid)
        axes[row][0].set_ylabel(str(embryo_id)[:28], color=theme.font, fontsize=7)

    fig.tight_layout()
    if save_path:
        from pathlib import Path

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, facecolor=theme.paper, bbox_inches="tight")
        print(f"  [SAVED] {path}")
    return fig
