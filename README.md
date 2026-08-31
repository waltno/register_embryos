# register_embryos

Turn a folder of multiplexed HCR `.nd2` z-stacks into per-nucleus gene-intensity
tables, register the embryos into a common space, and build a composite "atlas"
embryo — as an interactive Python package or a command-line workflow.

Built for zebrafish lateral plate mesoderm HCR (DAPI + three gene channels at
20×, 1024×1024), but nothing in it is specific to that panel.

```
ND2 files ─┬─ parse names ──► cohorts (genotype × timepoint × view × magnification)
           │
           ├─ load + z-bin + normalise ──► EmbryoVolume  (voxel size read from the ND2)
           │
           ├─ WIDGET: rotate + set contrast ──► orientation.json, contrast_limits.json
           │
           ├─ Cellpose nuclei ── 2D per z-bin │ 3D over the volume │ 2D + IoU linking
           │
           ├─ threshold gene channels, assign each signal pixel to its nearest nucleus
           │        ──► one row per nucleus with a mean intensity per gene
           │
           ├─ point-to-point ICP onto a cohort reference ──► x_reg, y_reg, z_reg
           │
           ├─ k-nearest-neighbour consensus ──► atlas (composite embryo)
           │
           └─ 3D + 2D figures, day and night themes
```

## Install

```bash
git clone https://github.com/waltno/register_embryos.git
cd register_embryos
pip install -e ".[widgets]"          # add ,open3d for the faster ICP backend
```

Requires Python 3.10+. Cellpose brings in PyTorch; a GPU is optional
(`--gpu` / `gpu=True`) and mainly matters for 3D segmentation.

## The filename contract

Every ND2 must be named:

```
{date}_{id}_{genotype}_{timepoint}_{view}_{magnification}_{ch1}_{ch2}_{ch3}.nd2
20260410_1.5_wt_12s_dorsal_20X_hand2_tbx5a_wt1a.nd2
```

Channel 0 is always the nuclear stain; `ch1..ch3` name the gene channels in
acquisition order.

**Embryos are processed together when they share
`(genotype, timepoint, view, magnification)`.** That 4-tuple is the *cohort*, and
it names the output directory (`wt_12s_dorsal_20X/`). Gene panel is deliberately
*not* part of the key — two embryos of the same genotype and stage imaged the same
way belong in one registered space even when the partner genes differ, which is
exactly the "wt1a plus a rotating partner" design.

Filenames not yet on the spec can be migrated:

```bash
register-embryos rename data/hcr/wt/wt_lpm_nd2_12s_dorsal --timepoint 12s   # dry run
register-embryos rename data/hcr/wt/wt_lpm_nd2_12s_dorsal --timepoint 12s --apply
register-embryos undo-rename data/hcr/.../rename_manifest.csv --apply       # reversible
```

The parser tolerates `20X_dorsal` as well as `dorsal_20X` and normalises the
order, and `rename` refuses the whole batch rather than half-renaming a cohort.

## Interactive use

```python
from register_embryos import CohortWorkflow

wf = CohortWorkflow.from_directory(
    "data/hcr/wt/wt_lpm_nd2_12s_dorsal",
    output_root="out/hcr",
    cohort="wt_12s_dorsal_20X",
)

wf.load(bin_size=7)          # slow: reads every ND2, bins z, reads voxel size
config = wf.prepare()        # ← widget: rotate each embryo, set per-channel contrast
wf.apply_prep(config)        # bake both in
wf.segment(mode="3d")        # or "2d" / "2d+link"
wf.build_tables()
wf.register(reference_embryo_id="20260410_1.5_wt_12s_dorsal_20X_hand2_tbx5a_wt1a")
wf.build_atlas(k_neighbors=4)
wf.plot_all()                # figures in both themes
wf.save_manifest()
```

Each step stores its result on the instance, so you can re-run `build_atlas` with
a different `k` — or `register` with a different reference — without repeating
`load` or `segment`.

### The prepare widget

The two steps that used to be hand-edited literals — `angles = [215, 155, 165, 70]`
and four `test_limits_N = {0: (0.05, 0.3), ...}` dicts — are now one panel:

- **Orientation** (whole embryo, all channels): xy rotation slider, ±90° nudges,
  x/y/z flips, and a warning when rotating inside the fixed 1024×1024 canvas
  would clip signal. `xz`/`yz` rotation exists too, with a warning that it
  resamples across only tens of z-bins.
- **Contrast** (per channel): low/high sliders over the live image, histogram and
  the clipped result — *what segmentation will actually see* — plus saturated and
  floored percentages. `Accept` advances to the next channel or embryo, so a
  cohort is walked with one button.

Both are written to `orientation.json` and `contrast_limits.json` in the cohort
directory. The CLI picks them up automatically, so the interactive decision is
made once and every later run is reproducible.

No ipywidgets? `auto_contrast_limits()` + `preview_contrast()` and an explicit
`OrientationSet.from_angles(ids, [215, 155, 165, 70])` cover the same ground.

## Command line

```bash
register-embryos scan   data/hcr/wt/wt_lpm_nd2_12s_dorsal        # what cohorts are here?
register-embryos run    data/hcr/wt/wt_lpm_nd2_12s_dorsal -o out --all --mode 3d
register-embryos atlas  out/wt_12s_dorsal_20X/combined_nucleus_table.csv -o out/retry --k 6
register-embryos plot   out/wt_12s_dorsal_20X/wt_12s_dorsal_20X_atlas.csv -o figs
```

`atlas` re-registers and re-atlases an existing table with no images involved —
use it to try a different `k`, drop a badly-fitting embryo (`--exclude`), or work
on tables produced before this package existed.

## Output layout

```
out/wt_12s_dorsal_20X/
├── orientation.json                 # accepted rotations
├── contrast_limits.json             # accepted contrast windows
├── combined_nucleus_table.csv       # every nucleus, every embryo
├── registered_nucleus_table.csv     # + x_reg, y_reg, z_reg
├── registration_residuals.csv       # before/after NN distance, same metric both sides
├── registration_transforms.npz      # the 4×4 matrices
├── wt_12s_dorsal_20X_atlas.csv      # the composite embryo
├── atlas_diagnostics.csv            # neighbour radius, embryos per atlas point
├── run_manifest.json                # inputs, parameters, versions
├── embryos/<embryo_id>/             # masks, per-embryo table, gene map
└── figures/                         # *_dark.png/html and *_light.png/html
```

## Segmentation modes

| mode | what it does | when |
|---|---|---|
| `2d` | Cellpose on each z-bin independently | fast, robust; labels are per-slice so one nucleus spanning two bins becomes two rows |
| `3d` | one Cellpose pass with `do_3D=True` and the voxel anisotropy | labels consistent through z, so a nucleus is one row with a true 3D centroid — the right input for ICP |
| `2d+link` | 2D Cellpose, then link slices by mask IoU | middle ground; removes duplicate centroids without the cost of 3D flow |

Two separate knobs, often confused:

- **`anisotropy`** — the z:xy voxel aspect ratio, read from the ND2 (`pixel_microns`
  and the median `z_coordinates` step) and multiplied by the z-binning factor.
  Cellpose uses it to rescale z internally so a spherical nucleus looks spherical.
  Wrong value ⇒ nuclei split along z (too low) or merge (too high).
- **`diameter`** — nucleus size in xy pixels. `None` lets Cellpose estimate it.

## Notes on the method

**Why max-project the z bins.** HCR puncta are sparse and bright; averaging a
punctum over five mostly-empty slices dilutes it below the signal threshold.

**Why signal pixels are assigned outward.** HCR signal sits around nuclei, not
inside the nuclear stain, so measuring only inside the Cellpose mask throws most
of it away. Each above-threshold pixel is given to its nearest nucleus and the
per-nucleus value is the mean over that expanded territory.

**Why background is `0.3`, not `0`.** A pixel dropped as background was never
measured. Averaging it in as zero would drag every nucleus mean toward zero in
proportion to how much empty space its territory covers, so the sentinel is
excluded from the mean instead. In 3D mode assignment distances are in
micrometres, so an anisotropic stack does not preferentially assign along z, and
`max_assign_distance_um` stops a pixel being dragged across the embryo.

**Why registration downsamples uniformly in space.** Uniform-at-random sampling
keeps dense regions dense, and ICP then fits the dense regions and ignores the
sparse ones. `isotropic_downsample` normalises each axis to [0,1] first, so z (a
few dozen bins) is weighted like x and y (a thousand pixels).

**Why residuals are reported with one metric.** Mean nearest-neighbour distance
compared against RMS nearest-neighbour distance makes a good fit look like a
regression, because RMS of a positive quantity always exceeds its mean.
`icp_residuals` computes both sides identically.

**Choosing atlas `k`.** With N embryos, `k ≈ N` averages roughly one nucleus per
embryo — smoothing between-embryo variability while preserving spatial detail.
Larger `k` blurs domain boundaries. `atlas_diagnostics` reports the neighbour
radius and how many distinct embryos actually contributed, which is how you tell
whether an atlas point is a consensus or just one embryo. `exclude_self_embryo=True`
makes it a genuine leave-one-out consensus.

**Two colour schemes, two themes.** `plot_pointcloud_3d` colours one gene per
panel on a black-to-hue ramp — quantitative, for "where is this gene on".
`plot_additive_3d`/`plot_additive_2d` overlay all genes, hue from the additive
mix at full brightness with size and opacity carrying intensity — qualitative, for
co-expression. Every function takes `mode="dark"` or `mode="light"`; the greys,
outlines and additive gain differ between them, because colour that reads bright
on black reads washed out on white. A nucleus counts as expressing when *one*
channel clears threshold, not when the channel sum does — otherwise a nucleus
could be coloured while being positive for nothing.

## Running 3D segmentation on a GPU

3D CPSAM over a 1024×1024 stack is minutes per z-bin on CPU, so a whole cohort
runs for many hours. 2D and `2d+link` are fine on CPU; 3D wants a GPU.

`docs/qsub_gpu_segmentation.sh` is a ready SGE submission script for the UW GS
cluster. It requests one device, and **fails fast if CUDA is not actually visible**
rather than silently falling back to CPU and looking like a hung job:

```bash
qsub -q trapnell-short.q docs/qsub_gpu_segmentation.sh <ND2_DIR> <OUT_DIR> [COHORT]

# which nodes have a free device right now:
qstat -F gpgpu,cuda.devices | awk '/\.q@/{q=$1} /cuda.devices=[1-9]/{print q,$0}'
```

Run the widget once in a notebook first — the job reads `orientation.json` and
`contrast_limits.json` from the cohort directory, so the GPU run reproduces your
accepted rotation and contrast instead of falling back to percentiles.

## Tests

```bash
pytest                       # synthetic data only; no ND2 files needed
```

## License

MIT
