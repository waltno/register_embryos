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
