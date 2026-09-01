"""Data-driven expression thresholds."""

import numpy as np
import pandas as pd
import pytest

from register_embryos.thresholds import (
    MIN_SEPARATION,
    apply_thresholds,
    call_thresholds,
    otsu_threshold,
    threshold_sweep,
)


def _bimodal(n_off=4000, n_on=1000, off=0.01, on=0.4, seed=0):
    """A channel that really is off/on, the case a threshold should find."""
    rng = np.random.default_rng(seed)
    return np.concatenate([
        np.abs(rng.normal(off, 0.01, n_off)),
        np.abs(rng.normal(on, 0.08, n_on)),
    ])


def _unimodal(n=5000, seed=1):
    """A channel that is simply off -- no split exists to find."""
    rng = np.random.default_rng(seed)
    return np.abs(rng.normal(0.01, 0.008, n))


def test_otsu_finds_the_valley_of_a_bimodal_channel():
    threshold, separation = otsu_threshold(_bimodal())
    assert 0.05 < threshold < 0.35          # between the two modes
    assert separation > MIN_SEPARATION


def test_otsu_reports_low_separation_for_a_unimodal_channel():
    """The important negative case: a cut always exists, so the score is what warns."""
    _, separation = otsu_threshold(_unimodal())
    assert separation < MIN_SEPARATION


def test_otsu_handles_degenerate_input():
    assert np.isnan(otsu_threshold(np.zeros(500))[0])
    assert otsu_threshold(np.array([0.1, 0.2]))[1] == 0.0     # too few points


def _table():
    on = _bimodal(seed=2)
    off = _unimodal(seed=3)
    n = len(on)
    return pd.DataFrame({
        "embryo_id": ["e0"] * n + ["e1"] * n,
        "x": np.tile(np.arange(n), 2), "y": 0.0, "z": 0.0,
        # e0 expresses geneA; e1 does not. geneB is off in both.
        "geneA": np.concatenate([on, _unimodal(n, seed=4)]),
        "geneB": np.concatenate([off, _unimodal(n, seed=5)]),
    })


def test_thresholds_are_found_per_embryo():
    """Contrast is set per embryo, so one cut for the cohort is not defensible."""
    results, diagnostics = call_thresholds(_table(), method="otsu", verbose=False)
    assert ("e0", "geneA") in results and ("e1", "geneA") in results
    assert len(diagnostics) == 4                       # 2 embryos x 2 genes

    expressing = results[("e0", "geneA")]
    assert expressing.trustworthy
    assert 0.05 < expressing.threshold < 0.35


def test_a_channel_that_is_simply_off_falls_back_and_is_flagged():
    """Not an error -- a real answer the data cannot give, so it must be visible."""
    results, _ = call_thresholds(_table(), method="otsu", fallback=0.05, verbose=False)
    silent = results[("e1", "geneA")]
    assert silent.fell_back
    assert not silent.trustworthy
    assert silent.threshold == 0.05
    assert "not two groups" in silent.note


def test_fixed_method_reproduces_the_constant():
    results, _ = call_thresholds(_table(), method="fixed", fallback=0.05, verbose=False)
    assert all(r.threshold == 0.05 for r in results.values())


def test_quantile_method_hits_the_requested_rate():
    results, _ = call_thresholds(
        _table(), method="quantile", quantile=0.8, verbose=False
    )
    for result in results.values():
        assert result.positive_fraction == pytest.approx(0.2, abs=0.02)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        call_thresholds(_table(), method="magic", verbose=False)


def test_gmm_falls_back_to_otsu_without_sklearn():
    """sklearn is optional; asking for gmm must degrade, not crash."""
    results, _ = call_thresholds(_table(), method="gmm", verbose=False)
    assert results[("e0", "geneA")].threshold > 0


def test_apply_thresholds_uses_each_embryos_own_cut():
    table = _table()
    results, _ = call_thresholds(table, verbose=False)
    called = apply_thresholds(table, results)
    assert "geneA_pos" in called.columns
    # e0 expresses geneA, e1 does not.
    by_embryo = called.groupby("embryo_id")["geneA_pos"].mean()
    assert by_embryo["e0"] > 0.1
    assert by_embryo["e1"] < 0.1


def test_data_driven_cuts_exceed_the_fixed_default_on_real_shaped_data():
    """The motivating observation: 0.05 sits inside the background shoulder.

    On the real cohort Otsu chose 0.10-0.25 per embryo-gene where the fixed cut was
    0.05, roughly halving the positive fraction.
    """
    table = _table()
    results, _ = call_thresholds(table, verbose=False)
    expressing = results[("e0", "geneA")]
    assert expressing.threshold > 0.05
    at_fixed = (table[table.embryo_id == "e0"]["geneA"] >= 0.05).mean()
    assert expressing.positive_fraction < at_fixed


def test_sweep_is_monotonically_decreasing():
    sweep = threshold_sweep(_table(), genes=["geneA"])
    for _, group in sweep.groupby("embryo_id" if "embryo_id" in sweep else "gene"):
        fractions = group.sort_values("threshold")["positive_fraction"].to_numpy()
        assert np.all(np.diff(fractions) <= 1e-12)


def test_additive_style_accepts_a_per_gene_threshold_mapping():
    """Data-driven cuts differ per gene, so a scalar cannot express them."""
    from register_embryos.plotting import additive_style

    df = pd.DataFrame({
        "x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0],
        "geneA": [0.10, 0.02], "geneB": [0.02, 0.10],
    })
    # geneA needs 0.05, geneB needs 0.5 -> only row 0 is positive.
    style = additive_style(df, genes=["geneA", "geneB"], mode="dark",
                           threshold={"geneA": 0.05, "geneB": 0.5})
    assert style["hi_mask"].tolist() == [True, False]
    # A scalar 0.05 would call both positive.
    scalar = additive_style(df, genes=["geneA", "geneB"], mode="dark", threshold=0.05)
    assert scalar["hi_mask"].tolist() == [True, True]
