"""register_embryos — ND2 to registered embryo atlas, for multiplexed HCR images.

The workflow, in order:

1. **Name**   every ND2 as
   ``date_id_genotype_timepoint_view_magnification_ch1_ch2_ch3.nd2``.  Embryos
   sharing ``(genotype, timepoint, view, magnification)`` form a *cohort* and are
   processed together into one output directory.
2. **Load**   read each ND2, max-project z into bins, normalise per channel, and
   read the voxel size from the file metadata (:mod:`~register_embryos.imaging`).
3. **Prepare** rotate each embryo into a common orientation and set per-channel
   contrast, in one widget (:func:`~register_embryos.widgets.prepare_widget`).
   Both are saved as JSON, so the run is reproducible and the CLI can repeat it.
4. **Segment** nuclei with Cellpose, either per z-bin (2D) or over the whole
   volume (3D, using the voxel anisotropy) (:mod:`~register_embryos.segmentation`).
5. **Assign**  threshold the gene channels, give every signal pixel to its nearest
   nucleus, and reduce to one row per nucleus
   (:mod:`~register_embryos.assignment`).
6. **Register** every embryo onto a cohort reference by point-to-point ICP
   (:mod:`~register_embryos.registration`).
7. **Atlas**   average the k nearest nuclei pooled across embryos at each anchor
   point to make a composite embryo (:mod:`~register_embryos.atlas`).
8. **Plot**    3D and 2D views in matched day and night themes
   (:mod:`~register_embryos.plotting`).

Notebook use::

    from register_embryos import CohortWorkflow

    wf = CohortWorkflow.from_directory(nd2_dir, output_root=out,
                                       cohort="wt_12s_dorsal_20X")
    wf.load(bin_size=7)
    config = wf.prepare()        # widget: rotate + contrast
    wf.apply_prep(config)
    wf.segment(mode="3d")
    wf.build_tables()
    wf.register()
    wf.build_atlas()
    wf.plot_all()

Command line::

    register-embryos scan  /path/to/nd2
    register-embryos run   /path/to/nd2 -o out --all --mode 3d
"""

from __future__ import annotations

__version__ = "0.1.0"

from .assignment import (
    BACKGROUND_VALUE,
    EmbryoResult,
    assign_signal_pixels_2d,
    assign_signal_pixels_3d,
    build_cohort_tables,
    build_nucleus_table,
    build_signal_mask,
    nucleus_table,
)
from .atlas import Atlas, align_atlases, atlas_diagnostics, build_atlas
from .contrast import (
    ContrastLimits,
    apply_contrast,
    apply_contrast_to_volumes,
    auto_contrast_limits,
    contrast_widget,
    log2_lift,
    preview_contrast,
)
from .imaging import (
    EXPECTED_XY,
    EmbryoVolume,
    VoxelSize,
    bin_volume_by_z,
    check_xy_shape,
    load_cohort,
    load_embryo,
    normalize_channel,
    read_voxel_size,
)
from .naming import (
    FILENAME_SPEC,
    CohortKey,
    EmbryoName,
    apply_renames,
    discover_embryos,
    format_embryo_name,
    group_into_cohorts,
    parse_embryo_name,
    plan_renames,
    undo_renames,
)
from .orientation import (
    Orientation,
    OrientationSet,
    apply_orientation,
    apply_orientation_to_points,
    apply_orientation_to_volumes,
)
from .plotting import (
    DARK,
    GENE_RGB,
    LIGHT,
    Theme,
    additive_style,
    plot_additive_2d,
    plot_additive_3d,
    plot_gene_panels_2d,
    plot_pointcloud_3d,
    plot_registration_2d,
    theme_for,
)
from .registration import (
    HAS_OPEN3D,
    RegistrationResult,
    icp_point_to_point,
    icp_residuals,
    isotropic_downsample,
    pca_align,
    register_cohort,
    register_frames,
)
from .segmentation import (
    SegmentedEmbryo,
    relabel_3d_from_2d,
    segment_2d,
    segment_3d,
    segment_cohort,
    segment_embryo,
)
from .widgets import PrepConfig, orientation_widget, prepare_widget
from .workflow import CohortOutputs, CohortWorkflow, run_all_cohorts, run_cohort, scan

__all__ = [
    "__version__",
    # naming / cohorts
    "FILENAME_SPEC", "EmbryoName", "CohortKey", "parse_embryo_name",
    "format_embryo_name", "discover_embryos", "group_into_cohorts",
    "plan_renames", "apply_renames", "undo_renames",
    # imaging
    "EXPECTED_XY", "check_xy_shape", "VoxelSize", "EmbryoVolume", "read_voxel_size",
    "load_embryo", "load_cohort", "bin_volume_by_z", "normalize_channel",
    # orientation
    "Orientation", "OrientationSet", "apply_orientation",
    "apply_orientation_to_volumes", "apply_orientation_to_points",
    # contrast
    "ContrastLimits", "auto_contrast_limits", "apply_contrast",
    "apply_contrast_to_volumes", "contrast_widget", "preview_contrast", "log2_lift",
    # widgets
    "PrepConfig", "prepare_widget", "orientation_widget",
    # segmentation
    "SegmentedEmbryo", "segment_2d", "segment_3d", "segment_embryo",
    "segment_cohort", "relabel_3d_from_2d",
    # assignment
    "BACKGROUND_VALUE", "EmbryoResult", "build_signal_mask",
    "assign_signal_pixels_2d", "assign_signal_pixels_3d", "nucleus_table",
    "build_nucleus_table", "build_cohort_tables",
    # registration
    "HAS_OPEN3D", "RegistrationResult", "isotropic_downsample", "pca_align",
    "icp_point_to_point", "icp_residuals", "register_cohort", "register_frames",
    # atlas
    "Atlas", "build_atlas", "atlas_diagnostics", "align_atlases",
    # plotting
    "Theme", "DARK", "LIGHT", "theme_for", "GENE_RGB", "additive_style",
    "plot_pointcloud_3d", "plot_additive_3d", "plot_additive_2d",
    "plot_gene_panels_2d", "plot_registration_2d",
    # workflow
    "CohortWorkflow", "CohortOutputs", "run_cohort", "run_all_cohorts", "scan",
]
