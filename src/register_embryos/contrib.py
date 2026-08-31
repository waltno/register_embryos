"""Project-specific analysis helpers -- NOT part of the standard pipeline.

Nothing here is called by :class:`~register_embryos.workflow.CohortWorkflow` or
:func:`~register_embryos.workflow.run_cohort`.  These are steps that depend on the
biology of a particular experiment rather than on the imaging, so baking them into
the generic workflow would silently apply an assumption that does not hold for
other panels or other tissues.

Import them explicitly when you want them::

    from register_embryos.contrib import midline_filter

    atlas_clean, bounds = midline_filter(atlas.points, marker="wt1a", axis="y")
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["midline_filter"]


def midline_filter(
    points: pd.DataFrame,
    marker: str = "wt1a",
    axis: str = "y",
    threshold: float = 0.05,
    genes: Optional[Sequence[str]] = None,
    trim_quantile: float = 0.01,
    min_gap: float = 5.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Drop apparent gene signal in the midline gap between two bilateral domains.

    A bilateral marker such as wt1a gives two domains along ``axis`` separated by
    a marker-negative gap at the midline.  A nucleus inside that gap that appears
    to express something is autofluorescence or non-target tissue, not a real
    co-expressing nucleus, and it inflates every co-expression statistic
    downstream.

    Two distinct roles, and conflating them makes the filter a no-op:

    ``marker``
        used only to FIND the gap.  Its positive coordinate values are split in
        two by an Otsu threshold and each domain's inner edge becomes a gap
        boundary.  So the gap is found from the data, not hard-coded, and the same
        call works on any atlas.
    ``genes``
        what gets JUDGED inside the gap.  Any nucleus in the gap expressing at
        least one of these is dropped.  Defaults to every gene column except the
        marker -- the marker is excluded because it defines the gap as
        marker-negative, so testing it there would only remove the handful of
        stragglers that set the boundary in the first place.

    Returns ``(filtered_points, boundaries)``.

    Raises:
        ValueError: if the two domains do not resolve, rather than silently
            filtering a band chosen from a unimodal distribution.
    """
    if marker not in points.columns:
        raise ValueError(f"marker {marker!r} not in the table")
    positive = points[marker].fillna(0) >= threshold
    values = points.loc[positive, axis].to_numpy(dtype=float)
    if values.size < 20:
        raise ValueError(f"only {values.size} {marker}+ points; too few to find a midline")

    split = _otsu_split(values)
    lower = values[values <= split]
    upper = values[values > split]
    if lower.size == 0 or upper.size == 0:
        raise ValueError(f"{marker}+ {axis} values did not split into two domains")

    gap_lo = float(np.quantile(lower, 1 - trim_quantile))
    gap_hi = float(np.quantile(upper, trim_quantile))
    if gap_hi - gap_lo < min_gap:
        raise ValueError(
            f"{marker}+ domains did not resolve along {axis}: gap "
            f"[{gap_lo:.1f}, {gap_hi:.1f}] is narrower than min_gap={min_gap}. "
            f"Either the marker is not bilateral along {axis}, or the two domains "
            f"overlap at the midline."
        )

    if genes is None:
        skip = {
            "embryo_id", "nucleus_id", "atlas_point_id", "n_voxels",
            "n_source_embryos", "neighbor_radius", marker,
            "x", "y", "z", "x_reg", "y_reg", "z_reg", "x_um", "y_um", "z_um",
        }
        genes = [
            column for column in points.columns
            if column not in skip and pd.api.types.is_numeric_dtype(points[column])
        ]
    gene_list = [gene for gene in genes if gene in points.columns]
    if not gene_list:
        raise ValueError(
            f"no gene columns to judge inside the gap (marker {marker!r} is "
            f"excluded by default; pass genes=[...] explicitly)"
        )

    in_gap = (points[axis] > gap_lo) & (points[axis] < gap_hi)
    expressing = (points[gene_list].fillna(0) >= threshold).any(axis=1)
    drop = in_gap & expressing
    filtered = points[~drop].reset_index(drop=True)

    boundaries = {"split": float(split), "gap_lo": gap_lo, "gap_hi": gap_hi}
    if verbose:
        print(
            f"  [MIDLINE] {axis} gap [{gap_lo:.1f}, {gap_hi:.1f}] found from "
            f"{marker}+ | judged on {gene_list} | dropped "
            f"{int(drop.sum()):,}/{len(points):,} points"
        )
    return filtered, boundaries


def _otsu_split(values: np.ndarray, n_bins: int = 128) -> float:
    """1-D Otsu threshold: the value maximising between-group variance."""
    counts, edges = np.histogram(values, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = counts.cumsum()[:-1]
    w1 = counts.sum() - w0
    cumulative = (counts * centers).cumsum()
    mu0 = cumulative[:-1] / np.maximum(w0, 1)
    mu1 = (cumulative[-1] - cumulative[:-1]) / np.maximum(w1, 1)
    between = np.where((w0 > 0) & (w1 > 0), w0 * w1 * (mu0 - mu1) ** 2, -1.0)
    return float(edges[1:-1][int(np.argmax(between))])
