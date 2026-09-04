"""Cohort-level orchestration: ND2 directory in, registered atlas out.

A *cohort* is every embryo sharing ``(genotype, timepoint, view, magnification)``,
and it is the unit of everything here: contrast is per embryo but decided while
looking at the cohort, registration is within the cohort, and the atlas is the
cohort's consensus.  Outputs land in ``<output_root>/<cohort name>/``.

Two entry points, same steps:

:class:`CohortWorkflow`
    Step-at-a-time, holding intermediate state, for notebook use.  Rotation and
    contrast come from the widget; every step can be re-run without repeating the
    expensive ones.

:func:`run_cohort`
    One call, start to finish, for the CLI and batch runs.  Needs orientation and
    contrast to already exist as JSON (or falls back to percentile contrast and no
    rotation), because there is nobody to ask.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .assignment import EmbryoResult, PixelFate, build_cohort_tables
from .atlas import Atlas, atlas_diagnostics, build_atlas
from .contrast import (
    ContrastLimits,
    apply_contrast_to_volumes,
    auto_contrast_limits,
)
from .imaging import EmbryoVolume, load_cohort
from .naming import CohortKey, EmbryoName, discover_embryos, group_into_cohorts
from .orientation import OrientationSet, apply_orientation_to_volumes
from .registration import RegistrationResult, register_cohort, registration_report
from .segmentation import SegmentedEmbryo, load_segmented, segment_cohort

__all__ = [
    "CohortWorkflow", "CohortOutputs", "run_cohort", "run_all_cohorts", "scan",
    "DEFAULT_BIN_SIZE", "default_bin_size", "find_masks_dir", "find_cohort_dir",
]


@dataclass
class CohortOutputs:
    """Everything one cohort run produced, and where it went."""

    cohort: CohortKey
    output_dir: Path
    embryo_ids: List[str] = field(default_factory=list)
    combined_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    registration: Optional[RegistrationResult] = None
    atlas: Optional[Atlas] = None
    figures: List[Path] = field(default_factory=list)

    @property
    def n_embryos(self) -> int:
        return len(self.embryo_ids)

    def summary(self) -> Dict[str, object]:
        return {
            "cohort": self.cohort.name,
            "n_embryos": self.n_embryos,
            "n_nuclei": int(len(self.combined_table)),
            "reference_embryo_id": (
                self.registration.reference_embryo_id if self.registration else None
            ),
            "mean_nn_after": (
                float(self.registration.stats["mean_after"].mean())
                if self.registration is not None and not self.registration.stats.empty
                else None
            ),
            "n_atlas_points": len(self.atlas) if self.atlas else 0,
            "output_dir": str(self.output_dir),
        }


def scan(
    input_dir: str | Path,
    timepoint: Optional[str] = None,
    verbose: bool = True,
) -> Dict[CohortKey, List[EmbryoName]]:
    """Parse an ND2 directory and report the cohorts it contains.

    Always worth running first: it is the cheapest way to catch a misnamed file,
    which otherwise shows up as a one-embryo cohort that cannot be registered.
    """
    embryos = discover_embryos(input_dir, default_timepoint=timepoint)
    cohorts = group_into_cohorts(embryos)
    if verbose:
        print(f"\n{len(embryos)} ND2 file(s) in {input_dir} -> {len(cohorts)} cohort(s)\n")
        for cohort, members in cohorts.items():
            print(f"  {cohort.name}  ({len(members)} embryo{'s' if len(members) != 1 else ''})")
            for embryo in members:
                genes = ", ".join(embryo.channels)
                print(f"    {embryo.embryo_id}")
                print(f"      date={embryo.date} id={embryo.embryo_num} genes=[{genes}]")
            if len(members) == 1:
                print(
                    f"    [NOTE] single-embryo cohort: registration and atlas are "
                    f"trivial here. Check the filename fields if you expected company."
                )
            print()
    return cohorts


class CohortWorkflow:
    """Step-at-a-time cohort processing, for notebook use.

    Typical session::

        wf = CohortWorkflow.from_directory(nd2_dir, cohort="wt_12s_dorsal_20X",
                                           output_root=out)
        wf.load(bin_size=7)                 # slow: reads every ND2
        config = wf.prepare()               # widget: rotate + set contrast
        wf.apply_prep(config)               # bake in rotation + contrast
        wf.segment(mode="3d")               # slow: Cellpose
        wf.build_tables()
        wf.register(reference_embryo_id=...)
        wf.build_atlas(k_neighbors=4)
        wf.plot_all()
        wf.save_manifest()

    Each step stores its result on the instance, so a later step can be re-run
    with different parameters without repeating the expensive earlier ones -- the
    point of splitting it up at all.
    """

    def __init__(
        self,
        cohort: CohortKey,
        embryos: Sequence[EmbryoName],
        output_root: str | Path,
    ) -> None:
        self.cohort = cohort
        self.embryos = list(embryos)
        self.output_dir = Path(output_root) / cohort.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.volumes: List[EmbryoVolume] = []
        self.adjusted: List[EmbryoVolume] = []
        self.segmented: List[SegmentedEmbryo] = []
        self.results: List[EmbryoResult] = []
        self.combined: pd.DataFrame = pd.DataFrame()
        self.registration: Optional[RegistrationResult] = None
        self.atlas: Optional[Atlas] = None
        self.orientations = OrientationSet()
        self.registration_qc: Dict[str, object] = {}
        self.contrast = ContrastLimits()
        self.figures: List[Path] = []
        self._params: Dict[str, object] = {}

    # -- construction -------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        input_dir: str | Path,
        output_root: str | Path,
        cohort: Optional[str] = None,
        timepoint: Optional[str] = None,
    ) -> "CohortWorkflow":
        """Build for one cohort found in ``input_dir``.

        Args:
            cohort: cohort name (``genotype_timepoint_view_magnification``).  With
                one cohort in the directory it can be omitted; with several it is
                required, and the error lists what is available.
        """
        cohorts = group_into_cohorts(
            discover_embryos(input_dir, default_timepoint=timepoint)
        )
        if not cohorts:
            raise FileNotFoundError(f"no parseable ND2 files in {input_dir}")
        if cohort is None:
            if len(cohorts) > 1:
                raise ValueError(
                    f"{len(cohorts)} cohorts in {input_dir}; name one of "
                    f"{[c.name for c in cohorts]}"
                )
            key = next(iter(cohorts))
        else:
            matches = [c for c in cohorts if c.name == cohort]
            if not matches:
                raise ValueError(
                    f"cohort {cohort!r} not found; available: {[c.name for c in cohorts]}"
                )
            key = matches[0]
        return cls(key, cohorts[key], output_root)

    # -- steps --------------------------------------------------------------

    def load(
        self,
        bin_size: int = 7,
        xy_um: Optional[float] = None,
        z_um: Optional[float] = None,
        strict_xy: bool = False,
        verbose: bool = True,
    ) -> List[EmbryoVolume]:
        """Read, z-bin and normalise every ND2 in the cohort. The slow first step."""
        print(f"\n{'='*72}\nLOAD — {self.cohort.name} ({len(self.embryos)} embryos)\n{'='*72}")
        self.volumes = load_cohort(
            self.embryos, bin_size=bin_size, xy_um=xy_um, z_um=z_um,
            strict_xy=strict_xy, verbose=verbose,
        )
        self._params["bin_size"] = bin_size
        return self.volumes

    def prepare(self, transform: str = "none", reuse: bool = True, **kwargs):
        """Open the rotation + contrast widget for this cohort.

        Args:
            reuse: reload any previously accepted values from this cohort's
                output directory, so a re-opened widget starts where you left off.
        """
        from .widgets import PrepConfig, prepare_widget

        config = PrepConfig.load(self.output_dir) if reuse else None
        if config is not None and (len(config.orientations) or len(config.contrast)):
            print(
                f"  reusing {len(config.orientations)} stored orientation(s) and "
                f"{len(config.contrast)} stored contrast entr(ies)"
            )
        return prepare_widget(
            self.volumes, config=config, output_dir=self.output_dir,
            transform=transform, **kwargs,
        )

    def apply_prep(
        self,
        config=None,
        orientations: Optional[OrientationSet] = None,
        contrast: Optional[ContrastLimits] = None,
        resize: bool = False,
        auto_contrast: bool = False,
        verbose: bool = True,
    ) -> List[EmbryoVolume]:
        """Bake rotation and contrast into the loaded volumes.

        Args:
            config: the :class:`~register_embryos.widgets.PrepConfig` from
                :meth:`prepare`.  Alternatively pass ``orientations`` and
                ``contrast`` directly.
            auto_contrast: fill any missing contrast limits from percentiles
                rather than raising.  Convenient for a first pass; check the
                result before trusting it, since the high percentile is set by
                however much bright junk each embryo carries.
        """
        if not self.volumes:
            raise RuntimeError("call load() first")

        if config is not None:
            orientations = orientations or config.orientations
            contrast = contrast or config.contrast
        self.orientations = orientations or self.orientations
        self.contrast = contrast or self.contrast

        print(f"\n{'='*72}\nORIENT + CONTRAST — {self.cohort.name}\n{'='*72}")
        oriented = apply_orientation_to_volumes(
            self.volumes, self.orientations, resize=resize, verbose=verbose
        )

        if auto_contrast and self.contrast.missing(oriented):
            gaps = self.contrast.missing(oriented)
            print(f"  filling {len(gaps)} missing contrast limit(s) from percentiles")
            filled = auto_contrast_limits(
                oriented, transform=self.contrast.transform, verbose=False
            )
            for embryo_id, channel in gaps:
                lo, hi = filled.get(embryo_id, channel)
                self.contrast.set(embryo_id, channel, lo, hi)

        self.adjusted = apply_contrast_to_volumes(oriented, self.contrast, verbose=verbose)
        self.orientations.save(self.output_dir / "orientation.json")
        self.contrast.save(self.output_dir / "contrast_limits.json")
        return self.adjusted

    def segment(
        self,
        mode: str = "2d",
        gpu: bool = False,
        diameter: Optional[float] = None,
        max_workers: int = 1,
        channel: int = 0,
        verbose: bool = True,
        **kwargs,
    ) -> List[SegmentedEmbryo]:
        """Cellpose nuclear segmentation. The slow second step.

        Args:
            channel: which channel Cellpose segments. 0 is the nuclear stain and
                is the right answer nearly always. Point it at a gene channel
                only to test what that stain alone segments into -- the rest of
                the pipeline still calls channel 0 "nuclei" and channels 1+
                "genes", so the segmented channel is then also measured as a
                gene against its own masks, and the resulting table's per-nucleus
                means for it are not comparable with a channel-0 run.
        """
        volumes = self.adjusted or self.volumes
        if not volumes:
            raise RuntimeError("call load() (and usually apply_prep()) first")
        if not self.adjusted:
            print("  [NOTE] segmenting un-adjusted volumes; apply_prep() was not run")
        if mode == "3d" and volumes[0].bin_size != 1:
            raise ValueError(
                f"mode='3d' needs the unbinned z-stack, but this cohort was loaded "
                f"with bin_size={volumes[0].bin_size}. Re-run load(bin_size=1) "
                f"(and apply_prep again), or segment with mode='2d' / '2d+link'."
            )
        if channel != 0:
            gene = volumes[0].gene_map.get(channel, f"ch{channel}")
            print(
                f"  [NOTE] segmenting channel {channel} ({gene}), not the nuclear "
                f"channel 0. Masks describe that stain, and downstream steps still "
                f"treat channel 0 as nuclei."
            )

        print(f"\n{'='*72}\nSEGMENT ({mode}) — {self.cohort.name}\n{'='*72}")
        self.segmented = segment_cohort(
            volumes, output_root=self.output_dir / "embryos", mode=mode, gpu=gpu,
            diameter=diameter, nuclei_channel=channel, max_workers=max_workers,
            verbose=verbose, **kwargs,
        )
        self._params.update(
            {"segmentation_mode": mode, "diameter": diameter,
             "segmentation_channel": channel}
        )
        return self.segmented

    def reload_segmentation(
        self,
        masks_from: Optional[str | Path] = None,
        verbose: bool = True,
    ) -> List[SegmentedEmbryo]:
        """Take the nuclear masks off disk instead of re-running Cellpose.

        Use this in place of :meth:`segment` when only the **gene** contrast has
        changed. Cellpose only ever looks at channel 0, so a new window on a gene
        channel leaves the masks valid -- what changes is the signal mask, the pixel
        assignment and the per-nucleus means, all of which are minutes of work rather
        than hours.

        Call it after :meth:`apply_prep`, so the reloaded masks are paired with the
        newly contrasted volumes. Refuses masks whose shape or bin size no longer
        matches, since a different z sampling or a canvas-resizing rotation makes them
        describe a different image.

        Args:
            masks_from: where the masks live. Defaults to this run's own output
                directory, which only works while you keep re-running into it --
                and the convention here puts each day's outputs under
                ``data/hcr/<YYYYMMDD>/``, so a fresh notebook on a new day cannot
                see yesterday's masks and would silently want Cellpose again. Point
                this at the run that did segment and Cellpose never runs twice::

                    wf.reload_segmentation(
                        "/net/trapnell/vol1/home/waltno/lpm/data/hcr/20260831"
                    )

                Accepts the dated run root, the cohort directory inside it, or the
                ``embryos/`` directory itself -- see :func:`find_masks_dir`. The path
                used is recorded in the manifest, since a run whose masks came from
                another day is not reproducible from its own directory alone.

        Raises:
            FileNotFoundError: if a cohort embryo has no saved masks.
        """
        volumes = self.adjusted or self.volumes
        if not volumes:
            raise RuntimeError("call load() (and usually apply_prep()) first")
        if not self.adjusted:
            print("  [NOTE] pairing masks with un-adjusted volumes; "
                  "apply_prep() was not run")

        embryos_dir = (
            find_masks_dir(masks_from, cohort_name=self.cohort.name)
            if masks_from is not None else self.output_dir / "embryos"
        )

        print(f"\n{'='*72}\nRELOAD SEGMENTATION — {self.cohort.name}\n{'='*72}")
        print(f"  masks from {embryos_dir}")
        print("  Cellpose is not re-run; only the gene channels are re-measured.")

        segmented: List[SegmentedEmbryo] = []
        missing: List[str] = []
        for volume in volumes:
            try:
                segmented.append(
                    load_segmented(embryos_dir / volume.embryo_id, volume,
                                   verbose=verbose)
                )
            except FileNotFoundError:
                missing.append(volume.embryo_id)

        if missing:
            available = sorted(
                path.parent.name for path in embryos_dir.glob("*/*_nuclear_masks.npy")
            ) if embryos_dir.is_dir() else []
            raise FileNotFoundError(
                f"no saved masks for {len(missing)} embryo(s) under {embryos_dir}: "
                f"{missing}.\n  Masks present there: {available or 'none'}.\n"
                f"  Run segment() for those, or pass masks_from=<the run that "
                f"segmented them>."
            )

        self.segmented = segmented
        if segmented:
            self._params["segmentation_mode"] = segmented[0].mode
            self._params["masks_reloaded"] = True
            self._params["masks_from"] = str(embryos_dir)
        return segmented

    # -- resuming from finished tables -------------------------------------

    def load_tables(
        self,
        tables_from: Optional[str | Path] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Read a finished nucleus table instead of re-deriving it from images.

        Everything downstream of the table -- registration, the atlas, every figure --
        needs only the table, so this skips loading ND2s and running Cellpose
        entirely and takes seconds:

            wf = CohortWorkflow.from_directory(nd2_dir, output_root=out, cohort=name)
            wf.load_tables("…/data/hcr/20260831")     # no load(), no segment()
            wf.register(); wf.build_atlas(); wf.plot_all()

        ``from_directory`` only parses filenames, so the cohort and its embryo
        metadata are already known without touching an image.

        Args:
            tables_from: the run holding ``combined_nucleus_table.csv`` -- the dated
                run root, the cohort directory, or the CSV itself. Defaults to this
                run's own output directory.

        Raises:
            FileNotFoundError: if no combined table is found under that path.
        """
        cohort_dir = find_cohort_dir(
            tables_from if tables_from is not None else self.output_dir,
            cohort_name=self.cohort.name,
        )
        path = cohort_dir / "combined_nucleus_table.csv"
        self.combined = pd.read_csv(path)
        # results stays empty: register() falls back to the combined table, and an
        # EmbryoResult rebuilt from a CSV would claim provenance it does not have.
        self.results = []

        # Orientations matter here even though no image is loaded: register() decides
        # whether to trust a manual orientation by whether any is known, and without
        # this the resume path silently reverts to an unconstrained fit -- which flips
        # embryos end-for-end on a cloud that cannot distinguish the two.
        orientation_json = cohort_dir / "orientation.json"
        if orientation_json.exists() and not len(self.orientations):
            self.orientations = OrientationSet.load(orientation_json)
            if verbose:
                print(f"  {len(self.orientations)} orientation(s) from "
                      f"{orientation_json}")

        self._params["tables_from"] = str(path)
        manifest = cohort_dir / "run_manifest.json"
        if manifest.exists():
            with open(manifest, encoding="utf-8") as handle:
                self._params["source_parameters"] = json.load(handle).get("parameters", {})

        if verbose:
            print(f"\n{'='*72}\nLOAD TABLES — {self.cohort.name}\n{'='*72}")
            print(f"  {len(self.combined):,} nucleus rows from {path}")
            in_table = set(self.combined["embryo_id"].unique())
            expected = {embryo.embryo_id for embryo in self.embryos}
            for label, missing in (("not in the table", expected - in_table),
                                   ("not in the cohort", in_table - expected)):
                if missing:
                    print(f"  [WARN] {len(missing)} embryo(s) {label}: "
                          f"{sorted(missing)}")
            for embryo_id, group in self.combined.groupby("embryo_id", sort=False):
                print(f"    {embryo_id}: {len(group):,} nuclei")
        return self.combined

    def load_registration(
        self,
        tables_from: Optional[str | Path] = None,
        verbose: bool = True,
    ) -> RegistrationResult:
        """Restore a finished registration, so the atlas and figures can be redone.

        Reads ``registered_nucleus_table.csv`` plus the residuals and transforms
        beside it. The reference embryo comes from the residuals table rather than
        being guessed, since which embryo everything was aligned to is not
        recoverable from the coordinates.

        Use it to re-atlas or re-plot an existing registration; call
        :meth:`register` instead to redo the fit itself.
        """
        cohort_dir = find_cohort_dir(
            tables_from if tables_from is not None else self.output_dir,
            cohort_name=self.cohort.name,
            filename="registered_nucleus_table.csv",
        )
        registered = pd.read_csv(cohort_dir / "registered_nucleus_table.csv")

        stats_path = cohort_dir / "registration_residuals.csv"
        stats = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
        if "reference_embryo_id" in stats.columns and not stats.empty:
            reference_embryo_id = str(stats["reference_embryo_id"].iloc[0])
        else:
            reference_embryo_id = str(registered["embryo_id"].iloc[0])
            if verbose:
                print(f"  [WARN] no residuals table; assuming the reference was "
                      f"{reference_embryo_id}")

        transforms_path = cohort_dir / "registration_transforms.npz"
        transforms = (
            {key: value for key, value in np.load(transforms_path).items()}
            if transforms_path.exists() else {}
        )

        self.registration = RegistrationResult(
            registered=registered, stats=stats,
            reference_embryo_id=reference_embryo_id, transforms=transforms,
        )
        if self.combined.empty:
            self.combined = registered
        self._params["registration_from"] = str(cohort_dir)

        if verbose:
            print(f"\n{'='*72}\nLOAD REGISTRATION — {self.cohort.name}\n{'='*72}")
            print(f"  {len(registered):,} rows, {len(self.registration.embryo_ids)} "
                  f"embryos, reference {reference_embryo_id}")
            if not stats.empty and "mean_after" in stats.columns:
                moved = stats[stats["embryo_id"] != reference_embryo_id]
                if not moved.empty:
                    print(f"  mean NN residual: {moved['mean_before'].mean():.2f} "
                          f"-> {moved['mean_after'].mean():.2f} px "
                          f"(cohort mean, reference excluded)")
        return self.registration

    def view_masks(self, embryo_ids: Optional[Sequence[str]] = None, **kwargs):
        """Open the z-by-z mask viewer for this cohort's segmentation.

        Needs only ``segment()`` or ``reload_segmentation()`` -- no tables, so
        this is the first look at what Cellpose produced, before any threshold or
        assignment cap has been chosen.  See
        :func:`~register_embryos.widgets.segmentation_widget`; the saveable
        version is :func:`~register_embryos.plotting.plot_mask_planes`.
        """
        from .widgets import segmentation_widget

        if not self.segmented:
            raise RuntimeError("call segment() or reload_segmentation() first")
        wanted = set(embryo_ids) if embryo_ids is not None else None
        chosen = [
            embryo for embryo in self.segmented
            if wanted is None or embryo.embryo_id in wanted
        ]
        if not chosen:
            raise ValueError(
                f"no segmented embryo matches {sorted(wanted)}; have "
                f"{[embryo.embryo_id for embryo in self.segmented]}"
            )
        return segmentation_widget(chosen, **kwargs)

    def preview_assignment(
        self,
        signal_threshold: float = 0.05,
        max_assign_distance_um: Optional[float] = None,
        embryo_ids: Optional[Sequence[str]] = None,
        mask_source: str = "genes",
        nuclei_threshold: Optional[float] = None,
        plot: bool = True,
        modes: Sequence[str] = ("light",),
        save_dir: Optional[Path] = None,
        verbose: bool = True,
        **plot_kwargs,
    ) -> Dict[str, "PixelFate"]:
        """What ``build_tables`` would discard, per pixel, without building anything.

        The dry run for the two parameters that decide the measurement.
        ``signal_threshold`` and ``max_assign_distance_um`` are invisible in their
        own output -- a nucleus table built with a cap ten times too tight still
        has one row per nucleus and a plausible number in every column -- so they
        have to be judged on the pixels, before the mean.

        Returns ``{embryo_id: PixelFate}``.  Re-cutting the distance is free from
        there (``fate.recut(30.0)``); only a new ``signal_threshold`` needs this
        called again.

        Args:
            embryo_ids: default is every segmented embryo.  Pass one to iterate
                quickly, then confirm across the cohort.
            save_dir: default ``<run dir>/qc``, since this is run QC and not a
                curated figure.
        """
        from .assignment import pixel_fate
        from .plotting import plot_pixel_fate

        if not self.segmented:
            raise RuntimeError("call segment() or reload_segmentation() first")
        wanted = set(embryo_ids) if embryo_ids is not None else None
        chosen = [
            embryo for embryo in self.segmented
            if wanted is None or embryo.embryo_id in wanted
        ]
        if not chosen:
            raise ValueError(
                f"no segmented embryo matches {sorted(wanted)}; have "
                f"{[embryo.embryo_id for embryo in self.segmented]}"
            )
        if verbose:
            print(f"\n{'='*72}\nASSIGNMENT PREVIEW — {self.cohort.name}\n{'='*72}")

        directory = Path(save_dir) if save_dir is not None else self.output_dir / "qc"
        fates: Dict[str, "PixelFate"] = {}
        for embryo in chosen:
            fate = pixel_fate(
                embryo, signal_threshold=signal_threshold,
                max_assign_distance_um=max_assign_distance_um,
                mask_source=mask_source, nuclei_threshold=nuclei_threshold,
                verbose=verbose,
            )
            fates[embryo.embryo_id] = fate
            if plot:
                for mode in modes:
                    plot_pixel_fate(
                        fate, mode=mode,
                        save_path=directory / f"{embryo.embryo_id}_pixel_fate_{mode}.png",
                        **plot_kwargs,
                    )
        return fates

    def build_tables(
        self,
        signal_threshold: float = 0.05,
        mask_source: str = "genes",
        nuclei_threshold: Optional[float] = None,
        max_assign_distance_um: Optional[float] = None,
        verbose: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Assign signal pixels to nuclei and build the combined nucleus table.

        Args:
            mask_source: which channels define measured territory. ``"genes"`` (the
                original) follows the gene channels; ``"nuclei"`` follows the nuclear
                stain, which excludes debris that is bright in a gene channel but
                carries no DAPI, and decouples the gene windows from each other. See
                :func:`~register_embryos.assignment.build_signal_mask` -- switching
                puts every gene value on a lower scale, so re-pick the positivity cut
                with :func:`~register_embryos.thresholds.positive_fraction`.
            nuclei_threshold: the DAPI cut under ``mask_source="nuclei"``. Defaults to
                ``signal_threshold``.
            max_assign_distance_um: drop signal pixels further than this from any
                nucleus. The direct defence against bright debris -- without it,
                nearest-nucleus assignment is unbounded and a speck in the corner of
                the frame is measured into whichever nucleus is nearest.
        """
        if not self.segmented:
            raise RuntimeError("call segment() first")
        print(f"\n{'='*72}\nNUCLEUS TABLES — {self.cohort.name}\n{'='*72}")
        self.results, self.combined = build_cohort_tables(
            self.segmented, signal_threshold=signal_threshold,
            mask_source=mask_source, nuclei_threshold=nuclei_threshold,
            max_assign_distance_um=max_assign_distance_um,
            output_root=self.output_dir, verbose=verbose, **kwargs,
        )
        self._params.update({
            "signal_threshold": signal_threshold,
            "mask_source": mask_source,
            "max_assign_distance_um": max_assign_distance_um,
            "nuclei_threshold": (
                signal_threshold if nuclei_threshold is None else nuclei_threshold
            ),
        })
        return self.combined

    def register(
        self,
        reference_embryo_id: Optional[str] = None,
        n_downsample: Optional[int] = 5000,
        trust_orientation: Optional[bool] = None,
        max_rotation_deg: float = 30.0,
        method: str = "icp",
        rot_kwargs: Optional[Dict[str, object]] = None,
        verbose: bool = True,
        **icp_kwargs,
    ) -> RegistrationResult:
        """Register every embryo onto one reference: ICP, or robust optimal transport.

        Args:
            method: ``"icp"`` (default, unchanged) -- point-to-point ICP.
                ``"rot"`` -- iterated robust optimal transport
                (:func:`~register_embryos.registration.robot_refine`), a
                from-scratch registration rather than a refinement of an ICP
                fit. ``**icp_kwargs`` is only used by ``method="icp"``; pass
                ``rot_kwargs`` for ``method="rot"``. See
                :func:`~register_embryos.registration.register_frames` for the
                full contract.
            trust_orientation: treat the rotation set in the widget as correct and
                let registration only refine it -- no PCA re-derivation (ICP) /
                unconstrained search (RobOT), rotation about z only, capped at
                ``max_rotation_deg``. Defaults to True whenever orientations were
                recorded for this cohort.

                This default exists because the unconstrained fit is actively wrong
                here: a 12-somite dorsal nucleus cloud is nearly a disc of
                revolution, so its in-plane angle carries almost no signal, and
                nearest-neighbour residual scores a 180-degree-flipped fit about as
                well as a correct one. Anterior-posterior orientation is obvious in
                the image and was already set from it; re-deriving it from the cloud
                throws that away.
            max_rotation_deg: the largest in-plane correction registration may
                apply when ``trust_orientation``.
            rot_kwargs: forwarded to ``robot_refine`` when ``method="rot"``.
                ``trust_orientation`` sets ``inplane_only``/``max_rotation_deg``
                in here too, the same way it sets them in ``icp_kwargs`` for
                ``method="icp"``.
        """
        if self.combined.empty:
            raise RuntimeError("call build_tables() first")
        if method not in ("icp", "rot"):
            raise ValueError(f"method must be 'icp' or 'rot', got {method!r}")

        if trust_orientation is None:
            trust_orientation = len(self.orientations) > 0
        rot_kwargs = dict(rot_kwargs or {})
        if trust_orientation:
            if method == "icp":
                icp_kwargs.setdefault("pca_init", False)
                icp_kwargs.setdefault("inplane_only", True)
                icp_kwargs.setdefault("max_rotation_deg", max_rotation_deg)
            else:
                rot_kwargs.setdefault("inplane_only", True)
                rot_kwargs.setdefault("max_rotation_deg", max_rotation_deg)

        print(f"\n{'='*72}\nREGISTER — {self.cohort.name} ({method})\n{'='*72}")
        if trust_orientation:
            print(f"  trusting the manual orientation: no PCA re-derivation, "
                  f"in-plane only, capped at +/-{max_rotation_deg:g} deg")
        else:
            print(
                "  [WARN] UNCONSTRAINED fit: PCA re-derives the axes and the rotation\n"
                "         is uncapped. On a dorsal cohort the nucleus cloud is nearly\n"
                "         symmetric in plane, so this routinely flips embryos\n"
                "         anterior-posterior. No orientation is on record for this\n"
                "         cohort -- if the embryos were rotated by hand, point\n"
                "         load_tables() at the run holding orientation.json, or pass\n"
                "         trust_orientation=True explicitly."
            )
        self.registration = register_cohort(
            self.results or self.combined,
            reference_embryo_id=reference_embryo_id,
            n_downsample=n_downsample,
            output_root=self.output_dir,
            method=method,
            rot_kwargs=rot_kwargs,
            verbose=verbose,
            **icp_kwargs,
        )
        self._params["n_downsample"] = n_downsample

        # Run the discriminating check automatically. The residual printed above
        # cannot see an anterior-posterior flip -- on this cohort it scored flipped
        # fits 6.49-8.81 px and correct ones 6.61-8.80 -- so a registration that is
        # only checked against the residual is not checked at all.
        self.registration_qc = registration_report(
            self.registration,
            threshold=self._params.get("signal_threshold", 0.05),
            max_rotation_deg=max_rotation_deg,
            verbose=verbose,
        )
        return self.registration

    def build_atlas(
        self,
        k_neighbors: Optional[int] = None,
        n_points: Optional[int] = None,
        verbose: bool = True,
        **kwargs,
    ) -> Atlas:
        """Consensus atlas from the registered cohort.

        ``k_neighbors`` defaults to **twice** the embryo count. One nucleus per
        embryo per point sounds like the natural choice and is too few whenever the
        gene panel rotates: a gene carried by 2 of 7 embryos then has one or zero
        measuring neighbours at most points, so its atlas channel is a
        nearest-neighbour lookup rather than a consensus, and it looks like noise
        scattered over the whole embryo. On this project's 7-embryo cohort, spatially
        coherent domains only appeared from about k=15. Watch the per-gene support
        that :func:`~register_embryos.atlas.build_atlas` prints and raise ``k`` until
        the rarest gene has a few neighbours -- the cost is spatial smoothing, which
        the neighbour radius reports.
        """
        if self.registration is None:
            raise RuntimeError("call register() first")
        k = (
            k_neighbors if k_neighbors is not None
            else max(4, 2 * len(self.registration.embryo_ids))
        )
        print(f"\n{'='*72}\nATLAS — {self.cohort.name}\n{'='*72}")
        self.atlas = build_atlas(
            self.registration.registered,
            reference_embryo_id=self.registration.reference_embryo_id,
            k_neighbors=k, n_points=n_points, label=f"{self.cohort.name}_atlas",
            verbose=verbose, **kwargs,
        )
        self.atlas.save(self.output_dir / f"{self.cohort.name}_atlas.csv")
        atlas_diagnostics(self.atlas).to_csv(
            self.output_dir / "atlas_diagnostics.csv", index=False
        )
        self._params.update({"k_neighbors": k, "n_atlas_points": n_points})
        return self.atlas

    # -- figures and provenance --------------------------------------------

    def plot_all(
        self,
        modes: Sequence[str] = ("dark", "light"),
        threshold=0.05,
        atlas_threshold=None,
    ) -> List[Path]:
        """Write the standard figure set in each theme.

        Both themes by default: the dark ones are for looking at, the light ones
        for putting in a figure, and generating them together means they never
        drift apart.

        Args:
            threshold: positivity cut for the ``*_thresholded_*`` figures, which
                drop sub-threshold nuclei and tint each nucleus only by the genes
                it is positive for. Any spec
                :func:`~register_embryos.thresholds.resolve_gene_cuts` takes.
            atlas_threshold: the same for the atlas, which needs its own number
                once a cohort carries many genes -- positivity is a union over
                channels, so more genes means more nuclei clear *something*, and
                kNN averaging narrows the spread the cut has to land in. Defaults
                to ``threshold``; use
                :func:`~register_embryos.thresholds.positive_fraction` to pick one.
        """
        from .plotting import (
            plot_additive_2d, plot_additive_3d, plot_additive_gene_2d,
            plot_gene_panels_2d, plot_pointcloud_3d, plot_registration_2d,
        )

        if atlas_threshold is None:
            atlas_threshold = threshold

        figure_dir = self.output_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        print(f"\n{'='*72}\nFIGURES — {self.cohort.name}\n{'='*72}")

        for mode in modes:
            if self.registration is not None and not self.registration.registered.empty:
                registered = self.registration.registered
                if len(self.registration.embryo_ids) > 1:
                    path = figure_dir / f"registration_qc_{mode}.png"
                    plot_registration_2d(
                        registered, self.registration.reference_embryo_id, mode=mode,
                        suptitle=f"{self.cohort.name} — ICP QC", save_path=path,
                    )
                    written.append(path)

                path = figure_dir / f"embryos_additive_{mode}.png"
                plot_additive_2d(
                    [(eid, registered[registered["embryo_id"] == eid])
                     for eid in self.registration.embryo_ids],
                    mode=mode, suptitle=f"{self.cohort.name} — registered embryos",
                    save_path=path,
                )
                written.append(path)

                path = figure_dir / f"embryos_additive_thresholded_{mode}.png"
                plot_additive_gene_2d(
                    registered, threshold=threshold, mode=mode, panel_size=4.2,
                    verbose=(mode == modes[0]),
                    suptitle=f"{self.cohort.name} — positive nuclei only",
                    save_path=path,
                )
                written.append(path)

            if self.atlas is not None:
                points = self.atlas.points
                for name, plotter, kwargs in (
                    ("atlas_additive_2d", plot_additive_2d, {}),
                    ("atlas_per_gene_2d", plot_gene_panels_2d, {}),
                ):
                    path = figure_dir / f"{name}_{mode}.png"
                    plotter(
                        points, mode=mode, save_path=path,
                        suptitle=f"{self.cohort.name} atlas ({len(points):,} points)",
                        coords=("x", "y", "z"), **kwargs,
                    )
                    written.append(path)

                path = figure_dir / f"atlas_additive_thresholded_{mode}.png"
                plot_additive_gene_2d(
                    self.atlas, threshold=atlas_threshold, mode=mode, panel_size=5.0,
                    verbose=(mode == modes[0]),
                    suptitle=f"{self.cohort.name} atlas — positive points only",
                    save_path=path,
                )
                written.append(path)

                for name, plotter in (
                    ("atlas_additive_3d", plot_additive_3d),
                    ("atlas_per_gene_3d", plot_pointcloud_3d),
                ):
                    path = figure_dir / f"{name}_{mode}.html"
                    plotter(
                        points, mode=mode, save_path=path,
                        title=f"{self.cohort.name} atlas", coords=("x", "y", "z"),
                    )
                    written.append(path)

        self._params.update({
            "figure_threshold": threshold if isinstance(threshold, (str, int, float))
                                else "per-gene",
            "figure_atlas_threshold": atlas_threshold
                if isinstance(atlas_threshold, (str, int, float)) else "per-gene",
        })
        self.figures = written
        print(f"  {len(written)} figure(s) -> {figure_dir}")
        return written

    def save_manifest(self) -> Path:
        """Write a run manifest: inputs, parameters, outputs, versions.

        Enough to reconstruct what produced the tables in this directory, which
        matters once several cohorts have been run with different bin sizes,
        segmentation modes and k values.
        """
        manifest = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "package_version": _package_version(),
            "cohort": {
                "name": self.cohort.name,
                "genotype": self.cohort.genotype,
                "timepoint": self.cohort.timepoint,
                "view": self.cohort.view,
                "magnification": self.cohort.magnification,
            },
            "embryos": [
                {
                    "embryo_id": embryo.embryo_id,
                    "nd2_path": str(embryo.path),
                    "date": embryo.date,
                    "embryo_num": embryo.embryo_num,
                    "genes": list(embryo.channels),
                }
                for embryo in self.embryos
            ],
            "parameters": self._params,
            "orientations": {
                eid: orientation.describe()
                for eid, orientation in self.orientations.orientations.items()
            },
            "outputs": {
                "n_nuclei": int(len(self.combined)),
                "reference_embryo_id": (
                    self.registration.reference_embryo_id if self.registration else None
                ),
                "n_atlas_points": len(self.atlas) if self.atlas else 0,
                "figures": [str(p.relative_to(self.output_dir)) for p in self.figures],
            },
        }
        path = self.output_dir / "run_manifest.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, default=str)
        print(f"  [MANIFEST] {path}")
        return path

    def outputs(self) -> CohortOutputs:
        return CohortOutputs(
            cohort=self.cohort,
            output_dir=self.output_dir,
            embryo_ids=[embryo.embryo_id for embryo in self.embryos],
            combined_table=self.combined,
            registration=self.registration,
            atlas=self.atlas,
            figures=self.figures,
        )


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("register-embryos")
    except Exception:  # pragma: no cover
        return "unknown"


#: z-bins per segmentation mode.  3D takes the whole stack unbinned -- binning is
#: a concession for 2D, which needs each plane to carry enough signal to segment
#: on its own, and it destroys exactly the z information 3D works from.
DEFAULT_BIN_SIZE = {"2d": 7, "2d+link": 7, "3d": 1}


def find_masks_dir(
    path: str | Path, cohort_name: Optional[str] = None
) -> Path:
    """Resolve any sensible way of naming a previous run to its ``embryos/`` directory.

    Segmentation is the expensive step and its output is reusable, so the path a
    person has to hand is whatever they happen to be looking at -- the dated run
    root, the cohort directory inside it, or the ``embryos/`` directory itself. All
    three resolve here rather than each caller guessing::

        data/hcr/20260831                          -> .../wt_12s_dorsal_20X/embryos
        data/hcr/20260831/wt_12s_dorsal_20X        -> .../embryos
        data/hcr/20260831/wt_12s_dorsal_20X/embryos-> itself

    Raises:
        FileNotFoundError: with the candidates that were tried, since a wrong path
            here otherwise looks like missing masks.
    """
    path = Path(path)
    candidates = [path / "embryos", path]
    if cohort_name:
        candidates.insert(0, path / cohort_name / "embryos")
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*/*_nuclear_masks.npy")):
            return candidate
    raise FileNotFoundError(
        f"no saved nuclear masks under {path}. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def find_cohort_dir(
    path: str | Path,
    cohort_name: Optional[str] = None,
    filename: str = "combined_nucleus_table.csv",
) -> Path:
    """Resolve a path to the cohort directory holding ``filename``.

    Same tolerance as :func:`find_masks_dir`: the dated run root, the cohort
    directory, or the file itself all work, because which one you have to hand
    depends on which you happen to be looking at.
    """
    path = Path(path)
    if path.is_file():
        return path.parent
    candidates = [path / cohort_name] if cohort_name else []
    candidates.append(path)
    for candidate in candidates:
        if (candidate / filename).is_file():
            return candidate
    raise FileNotFoundError(
        f"no {filename} under {path}. Tried: "
        + ", ".join(str(c / filename) for c in candidates)
    )


def default_bin_size(segmentation_mode: str) -> int:
    """Sensible z-binning for a segmentation mode."""
    return DEFAULT_BIN_SIZE.get(segmentation_mode, 7)


def run_cohort(
    input_dir: str | Path,
    output_root: str | Path,
    cohort: Optional[str] = None,
    timepoint: Optional[str] = None,
    bin_size: Optional[int] = None,
    segmentation_mode: str = "2d",
    diameter: Optional[float] = None,
    gpu: bool = False,
    max_workers: int = 1,
    signal_threshold: float = 0.05,
    mask_source: str = "genes",
    nuclei_threshold: Optional[float] = None,
    max_assign_distance_um: Optional[float] = None,
    reference_embryo_id: Optional[str] = None,
    n_downsample: Optional[int] = 5000,
    k_neighbors: Optional[int] = None,
    n_atlas_points: Optional[int] = None,
    orientation_json: Optional[str | Path] = None,
    contrast_json: Optional[str | Path] = None,
    masks_from: Optional[str | Path] = None,
    auto_contrast: bool = True,
    plot: bool = True,
    plot_modes: Sequence[str] = ("dark", "light"),
    verbose: bool = True,
) -> CohortOutputs:
    """Run one cohort end to end, without interaction.

    Orientation and contrast are read from JSON when given, and otherwise
    defaulted (no rotation; percentile contrast).  For a first pass on new data
    that is fine for a look, but the accepted-by-eye values from the widget are
    what you want for anything you intend to interpret -- a percentile high limit
    is set by whatever bright debris an embryo happens to carry.

    Args:
        bin_size: z-planes per bin.  ``None`` picks the right value for
            ``segmentation_mode``: 7 for 2D, and **1 for 3D**, which takes the
            whole stack unbinned.
        masks_from: a previous run whose nuclear masks to reuse instead of running
            Cellpose. Cellpose only ever sees channel 0, so re-tuning gene contrast
            or changing anything downstream does not invalidate its output -- this
            is the difference between minutes and hours per cohort. The bin size and
            the nuclear window must match the run that produced them; a mismatch is
            refused rather than silently combined.
    """
    workflow = CohortWorkflow.from_directory(
        input_dir, output_root=output_root, cohort=cohort, timepoint=timepoint
    )
    if bin_size is None:
        bin_size = default_bin_size(segmentation_mode)
        if verbose:
            print(f"  bin_size={bin_size} (default for mode={segmentation_mode!r})")
    elif segmentation_mode == "3d" and bin_size != 1:
        raise ValueError(
            f"mode='3d' takes the unbinned z-stack, but bin_size={bin_size} was "
            f"requested. Pass bin_size=1 (or leave it unset), or use mode='2d'."
        )
    workflow.load(bin_size=bin_size, verbose=verbose)

    orientations = (
        OrientationSet.load(orientation_json) if orientation_json else
        (OrientationSet.load(workflow.output_dir / "orientation.json")
         if (workflow.output_dir / "orientation.json").exists() else OrientationSet())
    )
    contrast = (
        ContrastLimits.load(contrast_json) if contrast_json else
        (ContrastLimits.load(workflow.output_dir / "contrast_limits.json")
         if (workflow.output_dir / "contrast_limits.json").exists() else ContrastLimits())
    )
    workflow.apply_prep(
        orientations=orientations, contrast=contrast,
        auto_contrast=auto_contrast, verbose=verbose,
    )

    if masks_from is not None:
        workflow.reload_segmentation(masks_from=masks_from, verbose=verbose)
    else:
        workflow.segment(
            mode=segmentation_mode, gpu=gpu, diameter=diameter,
            max_workers=max_workers, verbose=verbose,
        )
    workflow.build_tables(
        signal_threshold=signal_threshold, mask_source=mask_source,
        nuclei_threshold=nuclei_threshold,
        max_assign_distance_um=max_assign_distance_um, verbose=verbose,
    )
    workflow.register(
        reference_embryo_id=reference_embryo_id, n_downsample=n_downsample, verbose=verbose
    )
    workflow.build_atlas(k_neighbors=k_neighbors, n_points=n_atlas_points, verbose=verbose)
    if plot:
        workflow.plot_all(modes=plot_modes)
    workflow.save_manifest()

    outputs = workflow.outputs()
    if verbose:
        print(f"\n{'='*72}\nDONE — {workflow.cohort.name}\n{'='*72}")
        for key, value in outputs.summary().items():
            print(f"  {key}: {value}")
    return outputs


def run_all_cohorts(
    input_dir: str | Path,
    output_root: str | Path,
    timepoint: Optional[str] = None,
    skip_single_embryo: bool = False,
    **kwargs,
) -> List[CohortOutputs]:
    """Run every cohort found in a directory, continuing past failures.

    A failing cohort is reported and skipped rather than aborting the batch, since
    a long run usually contains one embryo with a problem and re-running the whole
    directory to get past it is expensive.
    """
    cohorts = scan(input_dir, timepoint=timepoint)
    outputs: List[CohortOutputs] = []
    failures: List[Tuple[str, str]] = []

    for cohort, members in cohorts.items():
        if skip_single_embryo and len(members) < 2:
            print(f"  [SKIP] {cohort.name}: only {len(members)} embryo")
            continue
        try:
            outputs.append(
                run_cohort(input_dir, output_root, cohort=cohort.name,
                           timepoint=timepoint, **kwargs)
            )
        except Exception as exc:  # noqa: BLE001 - batch must survive one bad cohort
            print(f"\n  [FAIL] {cohort.name}: {type(exc).__name__}: {exc}\n")
            failures.append((cohort.name, f"{type(exc).__name__}: {exc}"))

    print(f"\n{'='*72}\nBATCH COMPLETE\n{'='*72}")
    print(f"  succeeded: {len(outputs)}   failed: {len(failures)}")
    for name, error in failures:
        print(f"    FAILED {name}: {error}")
    if outputs:
        print()
        print(pd.DataFrame([o.summary() for o in outputs]).to_string(index=False))
    return outputs
