"""Thresholding a nucleus table, and the additive figure that draws the result.

The pipeline's constant 0.05 was set when a panel held three genes.  These tests pin
the three things that broke once panels started rotating across a cohort and the
atlas started carrying eight channels: which nuclei a cut keeps, that an atlas
intensity stays on the same scale as the per-embryo intensity it came from, and that
only the genes a nucleus is positive for tint it.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from register_embryos.atlas import build_atlas
from register_embryos.plotting import (
    additive_style,
    plot_additive_gene_2d,
)
from register_embryos.thresholds import (
    DEFAULT_THRESHOLD,
    ThresholdResult,
    as_frames,
    gene_columns,
    positive_calls,
    positive_fraction,
    resolve_gene_cuts,
)


@pytest.fixture
def cohort():
    """Two embryos with overlapping-but-different panels, as the real cohort has.

    ``hand2`` is in both; ``tbx1`` only in e0 and ``pax2a`` only in e1, so the other
    embryo's column is all-NaN -- the shape that makes "not measured" and "measured
    as zero" easy to confuse.
    """
    rng = np.random.default_rng(0)
    frames = []
    for embryo_id, own_gene in (("e0", "tbx1"), ("e1", "pax2a")):
        df = pd.DataFrame(rng.uniform(0, 100, (120, 3)), columns=["x_reg", "y_reg", "z_reg"])
        df["embryo_id"] = embryo_id
        df["nucleus_id"] = np.arange(len(df))
        df["hand2"] = np.where(df["x_reg"] > 50, 0.8, 0.01)
        df[own_gene] = np.where(df["y_reg"] > 50, 0.6, 0.0)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# -- resolving a spec into cuts ------------------------------------------------

def test_gene_columns_skips_bookkeeping(cohort):
    genes = gene_columns(cohort)
    assert set(genes) == {"hand2", "tbx1", "pax2a"}
    assert "nucleus_id" not in genes and "x_reg" not in genes


def test_scalar_spec_applies_to_every_gene(cohort):
    cuts = resolve_gene_cuts(cohort, ["hand2", "tbx1"], 0.2)
    assert cuts == {"hand2": 0.2, "tbx1": 0.2}


def test_mapping_spec_falls_back_for_unnamed_genes(cohort):
    cuts = resolve_gene_cuts(cohort, ["hand2", "tbx1"], {"hand2": 0.3})
    assert cuts["hand2"] == 0.3
    assert cuts["tbx1"] == DEFAULT_THRESHOLD


def test_call_thresholds_results_are_accepted_per_embryo(cohort):
    results = {
        ("e0", "hand2"): ThresholdResult(
            embryo_id="e0", gene="hand2", threshold=0.44, method="otsu", n=10,
            positive_fraction=0.5, separation=0.9,
        ),
        ("hand2"): 0.11,
    }
    assert resolve_gene_cuts(cohort, ["hand2"], results, embryo_id="e0")["hand2"] == 0.44
    # No entry for this embryo: the plain gene key is the next thing tried.
    assert resolve_gene_cuts(cohort, ["hand2"], results, embryo_id="e9")["hand2"] == 0.11


def test_quantile_spec_hits_the_requested_rate():
    # A graded channel, which is what real HCR intensity looks like -- the two-level
    # fixture would put the quantile on a mass point, where no cut gives that rate.
    graded = pd.DataFrame({"hand2": np.linspace(0, 1, 1000)})
    cuts = resolve_gene_cuts(graded, ["hand2"], "q0.8")
    assert (graded["hand2"] >= cuts["hand2"]).mean() == pytest.approx(0.2, abs=0.01)


def test_otsu_spec_separates_the_two_levels(cohort):
    e0 = cohort[cohort["embryo_id"] == "e0"]
    cut = resolve_gene_cuts(e0, ["hand2"], "otsu")["hand2"]
    assert 0.01 < cut < 0.8


def test_unknown_spec_names_the_alternatives(cohort):
    with pytest.raises(ValueError, match="expected a number"):
        resolve_gene_cuts(cohort, ["hand2"], "kmeans")


# -- who counts as positive ---------------------------------------------------

def test_unmeasured_gene_is_excluded_not_called_negative(cohort):
    """A gene missing from an embryo's panel must not dilute that embryo's rate."""
    e0 = cohort[cohort["embryo_id"] == "e0"]
    _, per_gene, measured = positive_calls(e0, ["hand2", "tbx1", "pax2a"], {}, require="any")
    assert measured == ["hand2", "tbx1"]
    assert per_gene.shape == (len(e0), 2)


def test_require_any_versus_all(cohort):
    e0 = cohort[cohort["embryo_id"] == "e0"]
    cuts = {"hand2": 0.05, "tbx1": 0.05}
    any_keep, _, _ = positive_calls(e0, ["hand2", "tbx1"], cuts, require="any")
    all_keep, _, _ = positive_calls(e0, ["hand2", "tbx1"], cuts, require="all")
    assert all_keep.sum() < any_keep.sum()
    # "all" is the co-expressing core, so it must be a subset of "any".
    assert not (all_keep & ~any_keep).any()


def test_require_rejects_other_words(cohort):
    with pytest.raises(ValueError, match="'any' or 'all'"):
        positive_calls(cohort, ["hand2"], {"hand2": 0.05}, require="some")


def test_raising_the_cut_can_only_shrink_the_kept_set(cohort):
    kept = [
        positive_fraction(cohort, threshold=cut)
        .query("gene == '<kept>'")["n_positive"].sum()
        for cut in (0.005, 0.05, 0.5, 0.9)
    ]
    assert kept == sorted(kept, reverse=True)


def test_positive_fraction_reports_per_embryo_and_a_kept_row(cohort):
    table = positive_fraction(cohort, threshold=0.05)
    assert set(table["label"]) == {"e0", "e1"}
    kept = table[table["gene"] == "<kept>"]
    assert len(kept) == 2
    for _, row in kept.iterrows():
        assert 0.0 <= row["positive_fraction"] <= 1.0


# -- shapes in, frames out ----------------------------------------------------

def test_as_frames_splits_a_cohort_and_keeps_one_frame_whole(cohort):
    assert [label for label, _ in as_frames(cohort)] == ["e0", "e1"]
    one = cohort[cohort["embryo_id"] == "e0"]
    assert [label for label, _ in as_frames(one)] == ["e0"]


def test_as_frames_takes_an_atlas_by_its_points(cohort):
    atlas = build_atlas(cohort, reference_embryo_id="e0", k_neighbors=3, verbose=False)
    frames = as_frames(atlas)
    assert len(frames) == 1 and frames[0][0] == "atlas"
    assert frames[0][1] is atlas.points


# -- the atlas keeps the per-embryo intensity scale ---------------------------

def test_masked_average_does_not_dilute_a_gene_carried_by_one_embryo(cohort):
    """The reason the atlas needed its own threshold: unmeasured counted as zero.

    ``pax2a`` is in one of two embryos here. Averaging over every neighbour drags its
    atlas value toward zero by roughly the fraction of neighbours that never imaged
    it; averaging over the measuring neighbours only leaves it on its original scale.
    """
    rng = np.random.default_rng(1)
    frames = []
    for index in range(4):
        df = pd.DataFrame(rng.uniform(0, 100, (150, 3)),
                          columns=["x_reg", "y_reg", "z_reg"])
        df["embryo_id"] = f"e{index}"
        df["hand2"] = np.where(df["x_reg"] > 50, 0.8, 0.01)
        # Only e0 carries pax2a: 1 embryo in 4, as osr1 is 1 in 7 in the real cohort.
        df["pax2a"] = np.where(df["y_reg"] > 50, 0.6, 0.0) if index == 0 else np.nan
        frames.append(df)
    rotating = pd.concat(frames, ignore_index=True)

    kwargs = dict(reference_embryo_id="e1", k_neighbors=8, verbose=False)
    masked = build_atlas(rotating, mask_unmeasured=True, **kwargs).points
    naive = build_atlas(rotating, mask_unmeasured=False, **kwargs).points
    source_p90 = rotating.loc[rotating["pax2a"].notna(), "pax2a"].quantile(0.9)

    assert naive["pax2a"].quantile(0.9) < 0.5 * masked["pax2a"].quantile(0.9)
    assert masked["pax2a"].quantile(0.9) == pytest.approx(source_p90, rel=0.25)


def test_support_is_recorded_off_to_the_side(cohort):
    """Per-gene support must not land in ``points``, where it would look like a gene."""
    atlas = build_atlas(cohort, reference_embryo_id="e0", k_neighbors=4, verbose=False)
    assert "support_pax2a" in atlas.diagnostics.columns
    assert "support_pax2a" not in atlas.points.columns
    assert "support_pax2a" not in gene_columns(atlas.points)


def test_a_point_with_no_measuring_neighbour_is_nan_not_zero(cohort):
    """Unknown and absent are different answers, and only NaN says unknown."""
    far = cohort.copy()
    # Push e1 (the only pax2a embryo) far away so e0's anchors never reach it.
    far.loc[far["embryo_id"] == "e1", ["x_reg", "y_reg", "z_reg"]] += 10_000
    atlas = build_atlas(far, reference_embryo_id="e0", k_neighbors=3, verbose=False)
    assert atlas.points["pax2a"].isna().all()


# -- the figure ---------------------------------------------------------------

def test_gating_keeps_a_single_positive_gene_pure(cohort):
    """Sub-threshold channels must not tint a nucleus once gating is on."""
    df = pd.DataFrame({
        "x": [0.0], "y": [0.0], "z": [0.0],
        "hand2": [0.8],     # cyan  (0, 1, 1)
        "wt1a": [0.04],     # magenta, below the cut
    })
    ungated = additive_style(df, ["hand2", "wt1a"], threshold=0.05)["rgb"][0]
    gated = additive_style(df, ["hand2", "wt1a"], threshold=0.05,
                           gate_below_threshold=True)["rgb"][0]
    assert ungated[0] > 0.0          # red channel picked up from sub-threshold wt1a
    assert gated[0] == pytest.approx(0.0)
    assert gated[1] == pytest.approx(1.0) and gated[2] == pytest.approx(1.0)


def test_sub_threshold_nuclei_are_dropped_and_can_be_kept(cohort, tmp_path):
    """The whole point of the new figure: a silent nucleus is gone, not grey."""
    drawn = {}
    for keep_silent in (False, True):
        fig = plot_additive_gene_2d(
            cohort, threshold=0.05, keep_silent=keep_silent, verbose=False,
            save_path=tmp_path / f"keep{keep_silent}.png",
        )
        drawn[keep_silent] = sum(
            int(collection.get_offsets().shape[0])
            for ax in fig.axes for collection in ax.collections
        )
        matplotlib.pyplot.close(fig)
    assert drawn[False] < drawn[True] == len(cohort)


def test_one_projection_reflows_into_a_grid(cohort, tmp_path):
    fig = plot_additive_gene_2d(
        cohort, n_cols=2, verbose=False, save_path=tmp_path / "grid.png",
    )
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 2          # two embryos, one projection, two columns
    matplotlib.pyplot.close(fig)


def test_n_cols_is_refused_with_several_projections(cohort):
    with pytest.raises(ValueError, match="exactly one projection"):
        plot_additive_gene_2d(
            cohort, projections=("XY", "XZ"), n_cols=2, verbose=False,
        )


def test_an_atlas_plots_from_the_same_call(cohort, tmp_path):
    atlas = build_atlas(cohort, reference_embryo_id="e0", k_neighbors=3, verbose=False)
    fig = plot_additive_gene_2d(
        atlas, threshold="q0.9", verbose=False, save_path=tmp_path / "atlas.png",
    )
    assert (tmp_path / "atlas.png").exists()
    matplotlib.pyplot.close(fig)


def test_unmatched_projection_names_are_an_error_not_an_empty_figure(cohort):
    with pytest.raises(ValueError, match="no panels left"):
        plot_additive_gene_2d(cohort, projections=("saggital",), verbose=False)
