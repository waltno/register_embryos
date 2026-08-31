"""Command line interface: ``register-embryos <command>``.

Commands mirror the library, so anything the CLI does can be done step-by-step in
a notebook and vice versa::

    register-embryos scan    DIR                 # what cohorts are here?
    register-embryos rename  DIR --timepoint 12s # bring filenames onto the spec
    register-embryos run     DIR -o OUT          # one cohort, or all of them
    register-embryos atlas   TABLE -o OUT        # re-register / re-atlas a table
    register-embryos plot    TABLE -o OUT        # figures from an existing table

The interactive rotation and contrast steps have no CLI equivalent by design --
they need eyes.  Run them once in a notebook; they write ``orientation.json`` and
``contrast_limits.json`` into the cohort directory, and ``run`` picks those up
automatically on every later invocation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="ND2 -> nucleus table -> registered atlas, for multiplexed HCR embryo images.",
)


@app.command()
def scan(
    input_dir: Path = typer.Argument(..., help="Directory of ND2 files."),
    timepoint: Optional[str] = typer.Option(
        None, "--timepoint", "-t",
        help="Timepoint for files missing that field (default: inferred from the directory name).",
    ),
) -> None:
    """List the cohorts in a directory and the embryos in each."""
    from .workflow import scan as scan_cohorts

    cohorts = scan_cohorts(input_dir, timepoint=timepoint)
    if not cohorts:
        typer.secho(f"no parseable ND2 files in {input_dir}", fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command()
def rename(
    input_dir: Path = typer.Argument(..., help="Directory of ND2 files."),
    timepoint: Optional[str] = typer.Option(
        None, "--timepoint", "-t", help="Timepoint to insert, e.g. 12s."
    ),
    apply: bool = typer.Option(
        False, "--apply", help="Actually rename. Without this it is a dry run."
    ),
    manifest: Optional[Path] = typer.Option(
        None, "--manifest", help="Where to record the renames (default: rename_manifest.csv)."
    ),
) -> None:
    """Bring filenames onto the canonical spec, recording an undo manifest.

    Canonical is ``date_id_genotype_timepoint_view_magnification_ch1_ch2_ch3``.
    Inserts a missing timepoint and normalises view/magnification order.
    """
    from .naming import apply_renames, plan_renames

    planned = plan_renames(input_dir, timepoint=timepoint)
    if not planned:
        typer.secho("all filenames already canonical", fg=typer.colors.GREEN)
        return
    apply_renames(planned, manifest_path=manifest, dry_run=not apply)
    if not apply:
        typer.secho("\nre-run with --apply to perform these renames", fg=typer.colors.YELLOW)


@app.command("undo-rename")
def undo_rename(
    manifest: Path = typer.Argument(..., help="A rename_manifest.csv written by `rename`."),
    apply: bool = typer.Option(False, "--apply", help="Actually revert."),
) -> None:
    """Revert the renames recorded in a manifest."""
    from .naming import undo_renames

    undo_renames(manifest, dry_run=not apply)
    if not apply:
        typer.secho("\nre-run with --apply to perform this revert", fg=typer.colors.YELLOW)


@app.command()
def run(
    input_dir: Path = typer.Argument(..., help="Directory of ND2 files."),
    output_root: Path = typer.Option(..., "--output", "-o", help="Output root directory."),
    cohort: Optional[str] = typer.Option(
        None, "--cohort", "-c",
        help="Cohort to run (genotype_timepoint_view_magnification). Omit with --all.",
    ),
    run_all: bool = typer.Option(False, "--all", help="Run every cohort in the directory."),
    timepoint: Optional[str] = typer.Option(None, "--timepoint", "-t"),
    bin_size: Optional[int] = typer.Option(
        None, "--bin-size", "-b",
        help="Z-planes per bin (max projection). Default: 7 for 2D, 1 for 3D "
             "(3D takes the whole stack unbinned).",
    ),
    mode: str = typer.Option(
        "2d", "--mode", "-m",
        help="Segmentation: '2d' (per z-bin), '3d' (Cellpose do_3D), '2d+link' (2D then IoU link).",
    ),
    diameter: Optional[float] = typer.Option(
        None, "--diameter", help="Nucleus diameter in xy pixels. Omit to let Cellpose estimate."
    ),
    gpu: bool = typer.Option(False, "--gpu", help="Use GPU for Cellpose."),
    workers: int = typer.Option(
        1, "--workers", "-w", help="Embryos segmented in parallel (2D only; 3D is memory-bound)."
    ),
    signal_threshold: float = typer.Option(0.05, "--signal-threshold"),
    reference: Optional[str] = typer.Option(
        None, "--reference", "-r", help="Registration reference embryo_id (default: first)."
    ),
    downsample: Optional[int] = typer.Option(
        5000, "--downsample", help="Nuclei per embryo before ICP. 0 disables."
    ),
    k_neighbors: Optional[int] = typer.Option(
        None, "--k", help="Atlas neighbours per point (default: the embryo count)."
    ),
    atlas_points: Optional[int] = typer.Option(
        None, "--atlas-points", help="Atlas size (default: one point per reference nucleus)."
    ),
    orientation_json: Optional[Path] = typer.Option(
        None, "--orientation", help="orientation.json from the widget."
    ),
    contrast_json: Optional[Path] = typer.Option(
        None, "--contrast", help="contrast_limits.json from the widget."
    ),
    no_auto_contrast: bool = typer.Option(
        False, "--no-auto-contrast",
        help="Fail rather than filling missing contrast limits from percentiles.",
    ),
    no_plots: bool = typer.Option(False, "--no-plots"),
    plot_mode: List[str] = typer.Option(
        ["dark", "light"], "--plot-mode", help="Theme(s) for figures. Repeatable."
    ),
    skip_single: bool = typer.Option(
        False, "--skip-single", help="With --all, skip one-embryo cohorts."
    ),
) -> None:
    """Load, orient, segment, assign, register and atlas a cohort.

    Rotation and contrast come from JSON written by the notebook widget; when
    absent, no rotation is applied and contrast falls back to percentiles.

    --mode 3d loads the z-stack unbinned (bin_size=1), because binning is a
    concession for 2D segmentation and destroys the z information 3D works from.
    It is much heavier: use a GPU (see docs/qsub_gpu_segmentation.sh).
    """
    from .workflow import run_all_cohorts, run_cohort

    if not run_all and cohort is None:
        typer.secho(
            "specify --cohort NAME, or --all to run every cohort "
            "(`register-embryos scan DIR` lists them)",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    shared = dict(
        timepoint=timepoint,
        bin_size=bin_size,
        segmentation_mode=mode,
        diameter=diameter,
        gpu=gpu,
        max_workers=workers,
        signal_threshold=signal_threshold,
        n_downsample=downsample or None,
        k_neighbors=k_neighbors,
        n_atlas_points=atlas_points,
        orientation_json=orientation_json,
        contrast_json=contrast_json,
        auto_contrast=not no_auto_contrast,
        plot=not no_plots,
        plot_modes=tuple(plot_mode),
    )
    if run_all:
        run_all_cohorts(input_dir, output_root, skip_single_embryo=skip_single, **shared)
    else:
        run_cohort(input_dir, output_root, cohort=cohort, reference_embryo_id=reference, **shared)


@app.command()
def atlas(
    table: Path = typer.Argument(..., help="A combined_nucleus_table.csv."),
    output_root: Path = typer.Option(..., "--output", "-o"),
    reference: Optional[str] = typer.Option(None, "--reference", "-r"),
    downsample: Optional[int] = typer.Option(5000, "--downsample"),
    k_neighbors: Optional[int] = typer.Option(None, "--k"),
    atlas_points: Optional[int] = typer.Option(None, "--atlas-points"),
    label: str = typer.Option("atlas", "--label"),
    exclude: List[str] = typer.Option([], "--exclude", help="embryo_id to drop. Repeatable."),
) -> None:
    """Register and atlas an existing nucleus table -- no images needed.

    Use this to retry registration or a different k without re-running Cellpose,
    or on tables produced before this package existed.
    """
    import pandas as pd

    from .atlas import atlas_diagnostics, build_atlas
    from .registration import register_cohort

    df = pd.read_csv(table)
    if exclude:
        before = df["embryo_id"].nunique()
        df = df[~df["embryo_id"].isin(exclude)]
        typer.echo(f"  excluded {before - df['embryo_id'].nunique()} embryo(s)")
    if df.empty:
        typer.secho("nothing left after exclusions", fg=typer.colors.RED)
        raise typer.Exit(1)

    output_root.mkdir(parents=True, exist_ok=True)
    registration = register_cohort(
        df, reference_embryo_id=reference, n_downsample=downsample or None,
        output_root=output_root,
    )
    k = k_neighbors if k_neighbors is not None else max(2, len(registration.embryo_ids))
    result = build_atlas(
        registration.registered,
        reference_embryo_id=registration.reference_embryo_id,
        k_neighbors=k, n_points=atlas_points, label=label,
    )
    result.save(output_root / f"{label}.csv")
    atlas_diagnostics(result).to_csv(output_root / f"{label}_diagnostics.csv", index=False)


@app.command()
def plot(
    table: Path = typer.Argument(..., help="A nucleus table or atlas CSV."),
    output_root: Path = typer.Option(..., "--output", "-o"),
    mode: List[str] = typer.Option(["dark", "light"], "--mode", help="Theme(s). Repeatable."),
    genes: List[str] = typer.Option([], "--gene", help="Restrict to these genes. Repeatable."),
    title: str = typer.Option("", "--title"),
) -> None:
    """Write the standard figure set for a table, in both themes."""
    import pandas as pd

    from .plotting import (
        plot_additive_2d, plot_additive_3d, plot_gene_panels_2d, plot_pointcloud_3d,
    )

    df = pd.read_csv(table)
    output_root.mkdir(parents=True, exist_ok=True)
    gene_list = list(genes) or None
    stem = table.stem

    for theme in mode:
        plot_additive_2d(
            df, mode=theme, genes=gene_list, suptitle=title or stem,
            save_path=output_root / f"{stem}_additive_2d_{theme}.png",
        )
        plot_gene_panels_2d(
            df, mode=theme, genes=gene_list, suptitle=title or stem,
            save_path=output_root / f"{stem}_per_gene_2d_{theme}.png",
        )
        plot_additive_3d(
            df, mode=theme, genes=gene_list, title=title or stem,
            save_path=output_root / f"{stem}_additive_3d_{theme}.html",
        )
        plot_pointcloud_3d(
            df, mode=theme, genes=gene_list, title=title or stem,
            save_path=output_root / f"{stem}_per_gene_3d_{theme}.html",
        )
    typer.secho(f"figures -> {output_root}", fg=typer.colors.GREEN)


@app.command()
def version() -> None:
    """Print the installed version."""
    from .workflow import _package_version

    typer.echo(f"register-embryos {_package_version()}")


def main() -> None:  # pragma: no cover - console entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
