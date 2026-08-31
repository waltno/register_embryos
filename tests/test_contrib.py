"""Project-specific helpers in register_embryos.contrib.

These are deliberately outside the standard pipeline, so they are tested
separately from it.
"""

import numpy as np
import pandas as pd
import pytest

from register_embryos.contrib import midline_filter


def _bilateral_atlas(seed=8):
    """Two wt1a+ domains at y = +/-30, plus autofluorescent junk at the midline.

    The junk is wt1a-NEGATIVE (so it does not disturb gap detection) but appears
    to express hand2 -- exactly the artefact the filter exists to remove.
    """
    rng = np.random.default_rng(seed)
    domains = pd.DataFrame({"y": np.concatenate([
        rng.normal(-30, 4, 400), rng.normal(30, 4, 400)
    ])})
    domains["wt1a"] = 0.5
    domains["hand2"] = rng.uniform(0, 0.6, len(domains))

    junk = pd.DataFrame({"y": rng.normal(0, 1.5, 40)})
    junk["wt1a"] = 0.0        # the midline gap is marker-negative
    junk["hand2"] = 0.9       # but looks like signal in another channel

    points = pd.concat([domains, junk], ignore_index=True)
    points["x"] = rng.normal(0, 50, len(points))
    points["z"] = 0.0
    return points


def test_midline_filter_drops_all_the_gap_junk():
    points = _bilateral_atlas()
    filtered, bounds = midline_filter(points, marker="wt1a", axis="y", verbose=False)

    assert bounds["gap_lo"] < 0 < bounds["gap_hi"]
    # Every one of the 40 junk nuclei (hand2 == 0.9, wt1a == 0) is gone.
    assert not ((filtered["hand2"] > 0.85) & (filtered["wt1a"] == 0.0)).any()
    # No apparently-expressing nucleus survives anywhere inside the gap.
    inside = filtered[(filtered["y"] > bounds["gap_lo"]) & (filtered["y"] < bounds["gap_hi"])]
    assert not (inside["hand2"] >= 0.05).any()


def test_midline_gap_spans_the_domain_inner_edges_not_just_the_midline():
    """The gap runs inner-edge to inner-edge, trimmed by trim_quantile.

    So it is much wider than the visible junk cluster, and the 1% of each domain
    nearest the midline is inside it by construction.  The core of each domain
    must still come through untouched.
    """
    points = _bilateral_atlas()
    filtered, bounds = midline_filter(points, marker="wt1a", axis="y", verbose=False)

    assert bounds["gap_lo"] < -10 and bounds["gap_hi"] > 10   # wide, as designed
    core = points["y"].abs() > 25
    assert (filtered["y"].abs() > 25).sum() == int(core.sum())

    # A larger trim widens the gap, so it can only remove more.
    wider, wide_bounds = midline_filter(
        points, marker="wt1a", axis="y", trim_quantile=0.10, verbose=False
    )
    assert wide_bounds["gap_lo"] < bounds["gap_lo"]
    assert len(wider) <= len(filtered)


def test_midline_filter_excludes_the_marker_from_judgement_by_default():
    """The marker defines the gap as marker-negative, so judging on it finds little.

    Judging on wt1a alone can only catch the domain-tail nuclei that set the
    boundary; the default (judging on hand2) additionally catches the 40 junk
    nuclei that are the actual artefact.
    """
    points = _bilateral_atlas()
    default_filtered, _ = midline_filter(points, marker="wt1a", axis="y", verbose=False)
    marker_only, _ = midline_filter(
        points, marker="wt1a", axis="y", genes=["wt1a"], verbose=False
    )
    dropped_default = len(points) - len(default_filtered)
    dropped_marker = len(points) - len(marker_only)
    assert dropped_default >= dropped_marker + 40
    # The junk survives a marker-only pass, which is why the default excludes it.
    assert ((marker_only["hand2"] > 0.85) & (marker_only["wt1a"] == 0.0)).sum() == 40


def test_midline_filter_needs_some_gene_to_judge():
    points = _bilateral_atlas()[["x", "y", "z", "wt1a"]]
    with pytest.raises(ValueError, match="no gene columns"):
        midline_filter(points, marker="wt1a", axis="y", verbose=False)


def test_midline_filter_refuses_a_unimodal_distribution():
    rng = np.random.default_rng(9)
    points = pd.DataFrame({
        "x": rng.normal(0, 10, 300), "y": rng.normal(0, 1, 300),
        "z": 0.0, "wt1a": 0.5, "hand2": 0.5,
    })
    with pytest.raises(ValueError, match="did not resolve"):
        midline_filter(points, marker="wt1a", axis="y", verbose=False)
