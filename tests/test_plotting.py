"""Themes and additive styling. Figures are rendered to a temp dir, not compared."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from register_embryos.plotting import (
    DARK,
    LIGHT,
    additive_style,
    gene_color,
    plot_additive_2d,
    plot_gene_panels_2d,
    theme_for,
)


@pytest.fixture
def cloud():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 10, (200, 3)), columns=["x", "y", "z"])
    df["embryo_id"] = "e0"
    df["hand2"] = np.where(df["x"] > 0, 0.7, 0.0)
    df["wt1a"] = np.where(df["y"] > 0, 0.6, 0.0)
    return df


@pytest.mark.parametrize("alias,expected", [
    ("dark", DARK), ("night", DARK), ("light", LIGHT), ("day", LIGHT),
])
def test_theme_aliases(alias, expected):
    assert theme_for(alias) is expected


def test_unknown_theme_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        theme_for("sepia")


def test_the_two_themes_differ_where_it_matters():
    assert DARK.paper != LIGHT.paper
    assert DARK.is_dark and not LIGHT.is_dark
    # Silent nuclei must stay visible on both grounds, so the greys differ.
    assert not np.allclose(DARK.silent_rgb, LIGHT.silent_rgb)


def test_gene_color_falls_back_for_unknown_genes():
    assert np.allclose(gene_color("hand2"), [0.0, 1.0, 1.0])
    unknown = gene_color("myUnnamedGene", index=1)
    assert unknown.shape == (3,) and unknown.max() <= 1.0


def test_additive_style_thresholds_per_channel_not_on_the_sum(cloud):
    """Three sub-threshold channels must not add up to "expressing"."""
    df = pd.DataFrame({
        "x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0],
        "hand2": [0.04, 0.20], "wt1a": [0.04, 0.0], "tbx1": [0.04, 0.0],
    })
    style = additive_style(df, mode="dark", threshold=0.05)
    # Row 0 sums to 0.12 but no single channel clears 0.05.
    assert not style["hi_mask"][0]
    assert style["hi_mask"][1]


def test_additive_style_gives_silent_nuclei_the_theme_grey(cloud):
    silent = cloud.copy()
    silent[["hand2", "wt1a"]] = 0.0
    for mode, theme in (("dark", DARK), ("light", LIGHT)):
        style = additive_style(silent, mode=mode)
        assert np.allclose(style["rgb"][0], theme.silent_rgb)


def test_additive_style_hue_survives_a_dim_channel():
    """Brightness is carried by size/opacity, so hue must not fade with intensity."""
    bright = pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0], "hand2": [1.0]})
    dim = pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0], "hand2": [0.1]})
    assert np.allclose(
        additive_style(bright, mode="dark")["rgb"][0],
        additive_style(dim, mode="dark")["rgb"][0],
    )


def test_additive_style_with_no_gene_columns_is_all_silent():
    df = pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]})
    style = additive_style(df, mode="dark")
    assert style["genes"] == []
    assert not style["hi_mask"].any()


def test_2d_figures_render_in_both_themes(cloud, tmp_path):
    for mode in ("dark", "light"):
        for plotter, name in ((plot_additive_2d, "additive"), (plot_gene_panels_2d, "per_gene")):
            path = tmp_path / f"{name}_{mode}.png"
            plotter(cloud, mode=mode, save_path=path)
            assert path.exists() and path.stat().st_size > 0


def test_additive_2d_accepts_labelled_rows(cloud, tmp_path):
    path = tmp_path / "rows.png"
    plot_additive_2d([("a", cloud), ("b", cloud)], mode="dark", save_path=path)
    assert path.exists()


def test_plotting_prefers_registered_coordinates(cloud, tmp_path):
    registered = cloud.copy()
    registered[["x_reg", "y_reg", "z_reg"]] = registered[["x", "y", "z"]] + 100
    fig = plot_additive_2d(registered, mode="light", save_path=tmp_path / "reg.png")
    # The x axis label comes from the coordinate column that was used.
    assert "x" in fig.axes[0].get_xlabel()
    assert fig.axes[0].get_xlim()[1] > 50


# ---------------------------------------------------------------------------
# Indexing: style arrays are positional, dataframe labels are not
# ---------------------------------------------------------------------------

def _concatenated_cohort():
    """Two embryos concatenated then sliced, so labels do NOT start at 0.

    This is the shape every real call has -- `registered[registered.embryo_id == eid]`
    off a concatenated table -- and it is what broke label-based indexing.
    """
    rng = np.random.default_rng(1)
    frames = []
    for name in ("e0", "e1"):
        df = pd.DataFrame(rng.normal(0, 10, (120, 3)), columns=["x", "y", "z"])
        df["embryo_id"] = name
        df["hand2"] = np.where(df["x"] > 0, 0.7, 0.0)
        df["wt1a"] = np.where(df["y"] > 0, 0.6, 0.0)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_additive_2d_handles_a_non_zero_based_index(tmp_path):
    """Regression: the second embryo's slice has labels 120..239, not 0..119."""
    cohort = _concatenated_cohort()
    slice_ = cohort[cohort["embryo_id"] == "e1"]
    assert slice_.index[0] != 0        # the condition that triggered the bug

    path = tmp_path / "sliced.png"
    plot_additive_2d(slice_, mode="dark", save_path=path)
    assert path.exists() and path.stat().st_size > 0


def test_additive_2d_handles_labelled_rows_from_slices(tmp_path):
    """The per-embryo-row form used by CohortWorkflow.plot_all()."""
    cohort = _concatenated_cohort()
    rows = [(eid, cohort[cohort["embryo_id"] == eid]) for eid in ("e0", "e1")]
    path = tmp_path / "rows.png"
    plot_additive_2d(rows, mode="light", save_path=path)
    assert path.exists()


def test_additive_2d_survives_a_duplicated_index(tmp_path):
    """A concatenation without ignore_index repeats labels; .loc would fan out."""
    cohort = _concatenated_cohort()
    duplicated = pd.concat([cohort.iloc[:50], cohort.iloc[:50]])
    assert duplicated.index.duplicated().any()
    path = tmp_path / "dup.png"
    plot_additive_2d(duplicated, mode="dark", save_path=path)
    assert path.exists()


def test_gene_panels_2d_handles_a_non_zero_based_index(tmp_path):
    cohort = _concatenated_cohort()
    slice_ = cohort[cohort["embryo_id"] == "e1"]
    path = tmp_path / "genes_sliced.png"
    plot_gene_panels_2d(slice_, mode="light", save_path=path)
    assert path.exists()


def test_colours_stay_matched_to_their_rows_after_slicing():
    """The real risk of label/positional mixing: right plot, wrong colours.

    Every hand2-positive row must get the hand2 hue, whatever the index labels.
    """
    from register_embryos.plotting import GENE_RGB, additive_style

    cohort = _concatenated_cohort()
    slice_ = cohort[cohort["embryo_id"] == "e1"].copy()
    style = additive_style(slice_, genes=["hand2"], mode="dark", threshold=0.05)

    positive = (slice_["hand2"].to_numpy() >= 0.05)
    assert positive.any() and (~positive).any()
    # Positional alignment: mask[i] must describe row i of the slice.
    assert np.allclose(style["rgb"][positive][0], GENE_RGB["hand2"])
    assert style["hi_mask"].tolist() == positive.tolist()


# ---------------------------------------------------------------------------
# Per-embryo gene grid
# ---------------------------------------------------------------------------

def _mixed_panel_cohort():
    """A cohort where the gene panel differs per embryo, as a real one does."""
    rng = np.random.default_rng(2)
    frames = []
    panels = {"e0": ["hand2", "wt1a"], "e1": ["hand2", "tbx1"], "e2": ["wt1a", "pax2a"]}
    for name, genes in panels.items():
        df = pd.DataFrame(rng.normal(0, 100, (200, 3)), columns=["x", "y", "z"])
        df["embryo_id"] = name
        df[["x_reg", "y_reg", "z_reg"]] = df[["x", "y", "z"]]
        for gene in ("hand2", "wt1a", "tbx1", "pax2a"):
            # np.nan for a gene this embryo was not stained for.
            df[gene] = np.where(df["x"] > 0, 0.6, 0.0) if gene in genes else np.nan
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_gene_by_embryo_renders_a_mixed_panel_cohort(tmp_path):
    from register_embryos.plotting import plot_gene_by_embryo

    path = tmp_path / "grid.png"
    plot_gene_by_embryo(_mixed_panel_cohort(), mode="light", save_path=path)
    assert path.exists() and path.stat().st_size > 0


def test_gene_by_embryo_marks_absent_genes_rather_than_drawing_nothing():
    """A blank axis is ambiguous; 'not in panel' is not."""
    from register_embryos.plotting import plot_gene_by_embryo

    fig = plot_gene_by_embryo(_mixed_panel_cohort(), mode="light")
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert "not in panel" in texts


def test_gene_by_embryo_accepts_per_embryo_thresholds():
    """The cuts from call_thresholds are keyed (embryo, gene); both forms must work."""
    from register_embryos.plotting import plot_gene_by_embryo
    from register_embryos.thresholds import call_thresholds

    table = _mixed_panel_cohort()
    results, _ = call_thresholds(table, genes=["hand2", "wt1a"], verbose=False)
    assert plot_gene_by_embryo(table, genes=["hand2", "wt1a"], mode="light",
                               threshold=results) is not None
    assert plot_gene_by_embryo(table, genes=["hand2", "wt1a"], mode="light",
                               threshold={"hand2": 0.3, "wt1a": 0.1}) is not None
    assert plot_gene_by_embryo(table, genes=["hand2"], mode="light",
                               threshold=0.05) is not None


def test_gene_by_embryo_prefers_registered_coordinates():
    from register_embryos.plotting import plot_gene_by_embryo

    table = _mixed_panel_cohort()
    table[["x_reg", "y_reg", "z_reg"]] = table[["x", "y", "z"]] + 500
    fig = plot_gene_by_embryo(table, genes=["hand2"], mode="light")
    assert fig.axes[0].get_xlim()[1] > 300
