"""Filename parsing, cohort grouping and the rename plan."""

from pathlib import Path

import pytest

from register_embryos.naming import (
    CohortKey,
    apply_renames,
    format_embryo_name,
    group_into_cohorts,
    parse_embryo_name,
    plan_renames,
    timepoint_from_directory,
    undo_renames,
)

CANONICAL = "20260410_1.5_wt_12s_dorsal_20X_hand2_tbx5a_wt1a.nd2"


def test_parses_canonical_name():
    name = parse_embryo_name(CANONICAL)
    assert name.date == "20260410"
    assert name.embryo_num == "1.5"
    assert name.genotype == "wt"
    assert name.timepoint == "12s"
    assert name.view == "dorsal"
    assert name.magnification == "20X"
    assert name.channels == ("hand2", "tbx5a", "wt1a")
    assert name.gene_map == {1: "hand2", 2: "tbx5a", 3: "wt1a"}
    assert name.embryo_id == Path(CANONICAL).stem


def test_view_and_magnification_order_is_tolerated():
    swapped = parse_embryo_name("20260825_1.1_wt_12s_20X_dorsal_wt1a_osr1_pax2a.nd2")
    assert swapped.view == "dorsal"
    assert swapped.magnification == "20X"
    # Normalised back to view-then-magnification on output.
    assert "dorsal_20X" in format_embryo_name(swapped)


def test_missing_timepoint_uses_default():
    name = parse_embryo_name(
        "20260410_1.5_wt_dorsal_20X_hand2_tbx5a_wt1a.nd2", default_timepoint="12s"
    )
    assert name.timepoint == "12s"
    assert name.genotype == "wt"


def test_missing_timepoint_without_default_raises():
    with pytest.raises(ValueError, match="no timepoint"):
        parse_embryo_name("20260410_1.5_wt_dorsal_20X_hand2_tbx5a_wt1a.nd2")


def test_strict_rejects_an_inferred_timepoint():
    with pytest.raises(ValueError, match="strict"):
        parse_embryo_name(
            "20260410_1.5_wt_dorsal_20X_hand2_tbx5a_wt1a.nd2",
            default_timepoint="12s",
            strict=True,
        )


@pytest.mark.parametrize(
    "bad,message",
    [
        ("notadate_1.5_wt_12s_dorsal_20X_a_b_c.nd2", "8-digit"),
        ("20260410_1.5_wt_12s_20X_hand2_tbx5a_wt1a.nd2", "no view"),
        ("20260410_1.5_wt_12s_dorsal_hand2_tbx5a_wt1a.nd2", "magnification"),
        ("20260410_1.5_dorsal_20X_a_b_c.nd2", "genotype"),
        ("20260410_1.5_wt_12s_dorsal_20X.nd2", "channel"),
    ],
)
def test_malformed_names_raise(bad, message):
    with pytest.raises(ValueError, match=message):
        parse_embryo_name(bad, default_timepoint="12s")


def test_non_adjacent_view_and_magnification_raise():
    with pytest.raises(ValueError, match="adjacent"):
        parse_embryo_name("20260410_1.5_wt_12s_dorsal_junk_20X_a_b_c.nd2")


def test_timepoint_from_directory():
    assert timepoint_from_directory("wt_lpm_nd2_12s_dorsal") == "12s"
    assert timepoint_from_directory("some/path/wt_24hpf_lateral") == "24hpf"
    assert timepoint_from_directory("no_stage_here") is None


def test_cohort_key_ignores_channels():
    """Two embryos with different gene panels still share a cohort.

    The wt1a-plus-rotating-partner design means panels differ between embryos
    that belong in the same registered space.
    """
    a = parse_embryo_name("20260410_1.5_wt_12s_dorsal_20X_hand2_tbx5a_wt1a.nd2")
    b = parse_embryo_name("20260505_3.6_wt_12s_dorsal_20X_hand2_pax2a_wt1a.nd2")
    assert a.cohort == b.cohort == CohortKey("wt", "12s", "dorsal", "20X")
    assert a.cohort.name == "wt_12s_dorsal_20X"


def test_grouping_separates_genotype_and_view():
    names = [
        parse_embryo_name("20260410_1.5_wt_12s_dorsal_20X_a_b_c.nd2"),
        parse_embryo_name("20260410_2.5_wt_12s_dorsal_20X_a_b_c.nd2"),
        parse_embryo_name("20260410_3.5_pbx_12s_dorsal_20X_a_b_c.nd2"),
        parse_embryo_name("20260410_4.5_wt_12s_lateral_20X_a_b_c.nd2"),
        parse_embryo_name("20260410_5.5_wt_24hpf_dorsal_20X_a_b_c.nd2"),
        parse_embryo_name("20260410_6.5_wt_12s_dorsal_10X_a_b_c.nd2"),
    ]
    cohorts = group_into_cohorts(names)
    assert len(cohorts) == 5
    assert len(cohorts[CohortKey("wt", "12s", "dorsal", "20X")]) == 2


def test_rename_plan_and_undo_round_trip(tmp_path):
    (tmp_path / "20260410_1.5_wt_dorsal_20X_hand2_tbx5a_wt1a.nd2").touch()
    (tmp_path / "20260825_1.1_wt_20X_dorsal_wt1a_osr1_pax2a.nd2").touch()
    (tmp_path / "20260505_2.2_wt_12s_dorsal_20X_a_b_c.nd2").touch()  # already canonical

    planned = plan_renames(tmp_path, timepoint="12s")
    assert len(planned) == 2
    new_names = {new.name for _, new in planned}
    assert "20260410_1.5_wt_12s_dorsal_20X_hand2_tbx5a_wt1a.nd2" in new_names
    assert "20260825_1.1_wt_12s_dorsal_20X_wt1a_osr1_pax2a.nd2" in new_names

    apply_renames(planned, dry_run=True)
    assert (tmp_path / "20260410_1.5_wt_dorsal_20X_hand2_tbx5a_wt1a.nd2").exists()

    apply_renames(planned, dry_run=False)
    manifest = tmp_path / "rename_manifest.csv"
    assert manifest.exists()
    for _, new in planned:
        assert new.exists()

    undo_renames(manifest, dry_run=False)
    assert (tmp_path / "20260410_1.5_wt_dorsal_20X_hand2_tbx5a_wt1a.nd2").exists()


def test_rename_refuses_to_clobber(tmp_path):
    (tmp_path / "20260410_1.5_wt_dorsal_20X_a_b_c.nd2").touch()
    (tmp_path / "20260410_1.5_wt_12s_dorsal_20X_a_b_c.nd2").touch()
    planned = plan_renames(tmp_path, timepoint="12s")
    with pytest.raises(FileExistsError):
        apply_renames(planned, dry_run=False)


def test_rename_refuses_a_colliding_plan(tmp_path):
    """Two differently-ordered names collapsing onto one canonical name.

    A partial rename would split a cohort, so the whole batch is refused.
    """
    (tmp_path / "20260410_1.5_wt_dorsal_20X_a_b_c.nd2").touch()
    (tmp_path / "20260410_1.5_wt_20X_dorsal_a_b_c.nd2").touch()
    planned = plan_renames(tmp_path, timepoint="12s")
    with pytest.raises(ValueError, match="collides"):
        apply_renames(planned, dry_run=False)
