# register_embryos
<img width="1052" height="315" alt="image" src="https://github.com/user-attachments/assets/85e17895-2682-4919-bf63-4a751020c978" />

Turn a folder of multiplexed HCR `.nd2` z-stacks into per-nucleus gene-intensity
tables, register the embryos into a common space, and build a composite "atlas"
embryo — as an interactive Python package or a command-line workflow.

Built for zebrafish lateral plate mesoderm HCR (DAPI + three gene channels at 20×,
1024×1024), but nothing in it is specific to that panel.

```
ND2 files → cohorts → load + z-bin → WIDGET: rotate + contrast → Cellpose nuclei
         → assign signal pixels to nearest nucleus → per-nucleus table
         → ICP onto a reference → kNN consensus atlas → 2D/3D figures, day & night
```

## Install

```bash
git clone https://github.com/waltno/register_embryos.git
cd register_embryos
pip install -e ".[widgets]"          # add ,open3d for the faster ICP backend
```

Python 3.10+. Cellpose brings in PyTorch; a GPU is optional and mainly matters for 3D.

## The filename contract

```
{date}_{id}_{genotype}_{timepoint}_{view}_{magnification}_{ch1}_{ch2}_{ch3}.nd2
20260410_1.5_wt_12s_dorsal_20X_hand2_tbx5a_wt1a.nd2
```

Channel 0 is always the nuclear stain; `ch1..ch3` name the gene channels in
acquisition order. **Embryos are processed together when they share
`(genotype, timepoint, view, magnification)`** — that 4-tuple is the *cohort* and it
names the output directory. Gene panel is deliberately not part of the key, so a
"wt1a plus a rotating partner" design lands in one registered space.

`20X_dorsal` and `dorsal_20X` both parse. Migrating older names is reversible, and
`rename` refuses the whole batch rather than half-renaming a cohort:

```bash
register-embryos rename <dir> --timepoint 12s            # dry run
register-embryos rename <dir> --timepoint 12s --apply
register-embryos undo-rename <dir>/rename_manifest.csv --apply
```

## Interactive use

```python
from register_embryos import CohortWorkflow

wf = CohortWorkflow.from_directory("<nd2_dir>", output_root="out/hcr",
                                   cohort="wt_12s_dorsal_20X")
wf.load(bin_size=7)          # slow: reads every ND2, bins z, reads voxel size
config = wf.prepare()        # ← widget: rotate each embryo, set per-channel contrast
wf.apply_prep(config)
wf.segment(mode="2d+link")   # or "2d" / "3d"
wf.build_tables(max_assign_distance_um=20.0)
wf.register()
wf.build_atlas()
wf.plot_all()                # both themes
wf.save_manifest()
```

Each step stores its result on the instance, so `build_atlas` with a different `k`
or `register` with a different reference costs seconds — no reload, no re-segment.

**The widget** sets xy rotation (±90° nudges, x/y/z flips, clipping warning) and
per-channel contrast over the live image, showing the clipped result — what
segmentation actually sees. It writes `orientation.json` and `contrast_limits.json`
to the cohort directory; every later run reads them, so the decision is made once.
`PrepConfig.load(cohort_dir)` resumes. Without ipywidgets, `auto_contrast_limits()`
and `OrientationSet.from_angles()` cover the same ground.

`preview_contrast(volume, config)` is the QC view — pass the whole `config`, not
`config.contrast`, or you get contrast on an unrotated embryo and a wrong rotation
passes unnoticed.

## Command line

```bash
register-embryos scan  <nd2_dir>                                   # what cohorts are here?
register-embryos run   <nd2_dir> -o out --all --mode 2d+link
register-embryos atlas out/<cohort>/combined_nucleus_table.csv -o out/retry --k 6
register-embryos plot  out/<cohort>/<cohort>_atlas.csv -o figs
```

`atlas` re-registers and re-atlases an existing table with no images involved — a
different `k`, a dropped embryo (`--exclude`), or tables from before this package.

## Output layout

```
out/wt_12s_dorsal_20X/
├── orientation.json  contrast_limits.json    # accepted rotations and windows
├── combined_nucleus_table.csv                # every nucleus, every embryo
├── registered_nucleus_table.csv              # + x_reg, y_reg, z_reg
├── registration_residuals.csv  registration_transforms.npz
├── wt_12s_dorsal_20X_atlas.csv               # the composite embryo
├── atlas_diagnostics.csv                     # neighbour radius, embryos per point
├── run_manifest.json                         # inputs, parameters, versions
├── embryos/<embryo_id>/                      # masks, per-embryo table, gene map
└── figures/                                  # *_dark and *_light
```

## Segmentation modes

| mode | z input | Cellpose runs | one nucleus becomes | assignment |
|---|---|---|---|---|
| `2d` | binned (7) | per z-plane | one row **per plane**, ids unrelated between planes | within each plane |
| `2d+link` | binned (7) | per z-plane (identical to `2d`) | one row, true 3D centroid | within each plane |
| `3d` | **unbinned** | once over the volume (`do_3D`) | one row, true 3D centroid | through the volume, in µm |

`2d` and `2d+link` run *exactly the same segmentation*; `2d+link` then merges labels
across z by pixel IoU. Without it a nucleus spanning two planes becomes two points
stacked in z, which ICP fits as real structure. It cannot split a blob 2D already
merged. **Default to `2d+link`.**

`3d` takes the unbinned stack and this is enforced — z-binning max-projects away
exactly the information `do_3D` works from (at `bin_size=7` a 6 µm nucleus spans 0.57
planes, so `do_3D` degenerates into a slow 2D run).

Two knobs that get confused: **`anisotropy`** is the z:xy voxel ratio, read from the
ND2 — too low splits nuclei along z, too high merges them. **`diameter`** is nucleus
size in xy pixels; `None` lets Cellpose estimate it.

## Things that will bite you

Numbers below are measured on this project's 7-embryo, 8-gene 20× dorsal cohort.

**Debris gets assigned from anywhere.** Nearest-nucleus assignment has no notion of
"too far". Here 20–24% of signal pixels sit >20 µm from any nucleus and 5–18% sit
>80 µm (maxima 366–591 µm) — all of it previously averaged in. Pass
`max_assign_distance_um=20.0` to `build_tables`; it leaves real measurements' scale
untouched.

**`mask_source` decides what counts as measured.** `"genes"` (default) follows the
gene channels into the perinuclear space where most HCR signal sits. `"nuclei"`
follows the nuclear stain, which also excludes gene-bright debris — but here a DAPI
cut of 0.05 keeps only **3.6%** of pixels, so territory collapses onto the Cellpose
masks and p90 gene intensity falls ~30×. A different measurement, not a stricter one;
use the distance cap for debris.

**Gene channels are coupled under the default mask.** It is a union, so a pixel
counts for *every* gene wherever *any* gene had signal: raising the hand2 and wt1a
floors moved **tbx1 from 20.4% to 24.2% positive with its own window untouched**.
Re-check every channel after changing any, or use `signal_mask_mode="per_channel"`.

**Automatic contrast uses p90, not p1.** An HCR channel is mostly background (median
0.004, p99.5 0.098), so `p1/p99.5` put **36–41% of pixels above the 0.05 threshold**
where `p90/p99.9` gives 6.5–8.2%. `auto_contrast_limits` warns above 25%, but the
widget is still the answer.

**Segment once.** Cellpose only ever sees channel 0, so new gene contrast, a new
threshold or a new reference does not invalidate the masks. Outputs are filed per
day, so a notebook opened on a new day would otherwise want hours of Cellpose again:

```python
wf.reload_segmentation(masks_from="…/data/hcr/20260831/wt_12s_dorsal_20X")
```
```bash
register-embryos run <nd2_dir> -o <out> --masks-from <previous run>
```

The run root, the cohort dir and its `embryos/` dir all resolve. A mask whose shape,
bin size or **orientation** no longer matches is refused rather than combined — the
orientation check matters because Cellpose runs after rotation, and rotating inside a
fixed canvas leaves the array shape identical, so nothing else would catch it. Change
a rotation and you must re-segment; `reload_segmentation` is for gene contrast only.

**Resume without images at all.** Registration, the atlas and every figure need only
the nucleus table, so on a fresh kernel skip straight past loading and segmentation:

```python
wf = CohortWorkflow.from_directory(nd2_dir, output_root=out, cohort=name)  # names only
wf.load_tables("…/data/hcr/20260831")        # no load(), no segment()
wf.load_registration("…/data/hcr/20260831")  # optional: skip ICP too
wf.build_atlas(); wf.plot_all()
```

`load_registration` takes the reference embryo from `registration_residuals.csv`
rather than guessing — which embryo everything was aligned to is not recoverable from
the coordinates.

**Registration trusts your manual orientation.** A 12-somite dorsal cloud is nearly a
disc of revolution (in-plane aspect **1.18**), so nothing in the positions pins down
the AP angle — nearest-neighbour residual scored a 180°-flipped fit as well as a
correct one (7.40 vs 7.38 px), and unconstrained ICP flipped three of seven embryos
end-for-end. `register()` therefore defaults to `trust_orientation=True` when
orientations exist: rotation about z only, capped at ±30°. That cut cohort gene-domain
spread for wt1a 69 → 43 px and tbx1 118 → 65 px.

So an embryo set wrong in the widget **stays** wrong and no residual will tell you —
check `orientation_grid(volumes, orientations)` before segmenting.

The cap binds *during* the fit. Clamping a free fit afterwards returns the cap value:
a 173° fit clamped to 30° left an embryo worse aligned than the flip it came from
(mean NN 8.09 → 10.88), where projecting each iteration into the allowed set reached
−7.4° at 8.02. open3d's ICP cannot be constrained mid-fit, so a constrained request
runs on the numpy backend; the `[ICP]` line names the one actually used.

`trust_orientation` defaults on when orientations are known, so **check the header
line** — an unconstrained run says so with a `[WARN]`, and the residual will not
distinguish it (flipped fits scored 6.49–8.81 mean NN against 6.61–8.80 for correct
ones).

**Point-to-point, not point-to-plane.** The predecessor pipeline defaulted to
point-to-plane with normals estimated at radius 50–60. A nucleus-centroid cloud is
not a surface: it is a slab (812 × 734 × 24 here), so **100% of estimated normals
come out along z**. Point-to-plane measures residual along the normal, which makes
its objective nearly blind to in-plane error — the alignment that matters. There is
no `normal_radius` here, deliberately.

`refine_with_ot=True` adds an optional bounded second stage, always a single global
matrix. `"similarity"` was the best model (mean NN 7.13 vs 7.39); off by default, and
it does **not** resolve the orientation ambiguity.

**The atlas averages each gene only over neighbours that measured it.** With a
rotating panel, NaN means "not in this embryo's panel", not "absent" — averaging those
in as zeros made a gene carried by 1 of 7 embryos ~7× too dim. `mask_unmeasured=True`
(default) restores the per-embryo scale (osr1 p90 0.058 → 0.419), which is what lets a
threshold transfer between spaces.

**`k` defaults to twice the embryo count.** One neighbour per embryo is too few with a
rotating panel: at k=3 every channel's median support was 0–2 and no coherent domains
appeared until about k=15. `build_atlas` prints per-gene support and warns below 3.

**Thresholding positives.** A flat 0.05 keeps 60–88% of nuclei per embryo and 92% of
atlas points, because positivity is a **union** — adding genes raises the kept
fraction with every channel unchanged. Decide from the data:

```python
positive_fraction(atlas, threshold=0.15)               # pick a cut without plotting
plot_additive_gene_2d(registered, threshold=0.05)      # embryo grid
plot_additive_gene_2d(atlas, threshold="q0.95")        # same call, one composite
```

`threshold` takes a number, a `{gene: cut}` mapping, `call_thresholds` results,
`"otsu"`, or a per-gene rate like `"q0.95"`; string forms are recomputed per frame.
Prefer a per-gene cut — the same 0.05 means a different brightness in every embryo.
`"otsu"` is no rescue: on this atlas it fell back to the constant for all eight
channels, because HCR intensity is a graded continuum, not two groups.

**Smaller ones.** Background pixels carry a `0.3` sentinel, not `0`, and are excluded
from the mean — a dropped pixel was never measured. Registration downsamples
isotropically, so z (a few dozen bins) is weighted like x and y (a thousand pixels).
XZ/YZ panels autoscale rather than forcing equal aspect across pixels and bin
indices; pass `z_aspect=<anisotropy>` for true proportion.

## Figures

`plot_pointcloud_3d` / `plot_gene_panels_2d` colour one gene per panel on a
background-to-hue ramp — quantitative. `plot_additive_*` overlay all genes —
qualitative, for co-expression. Every function takes `mode="dark"` or `"light"`.

In `plot_additive_gene_2d`, hue is the intensity-weighted mix of the genes a nucleus
is positive for, and saturation and dot size track the summed intensity, so a
barely-positive nucleus is a pale small dot rather than a confident call.
`color_scale="full"` restores flat saturation; `min_saturation` and `color_gamma` tune
the ramp; `keep_silent=False` drops sub-threshold nuclei instead of greying them.

## Running 3D segmentation on a GPU

3D CPSAM over a 1024×1024 stack is minutes per z-bin on CPU. `docs/qsub_gpu_segmentation.sh`
is a ready SGE script for the UW GS cluster that **fails fast if CUDA is not visible**
rather than falling back to CPU and looking like a hung job:

```bash
qsub -q trapnell-short.q docs/qsub_gpu_segmentation.sh <ND2_DIR> <OUT_DIR> [COHORT]

# which nodes have a free device right now:
qstat -F gpgpu,cuda.devices | awk '/\.q@/{q=$1} /cuda.devices=[1-9]/{print q,$0}'
```

Run the widget once first — the job reads the two JSONs from the cohort directory, so
the GPU run reproduces your accepted rotation and contrast instead of percentiles.

## Tests

```bash
pytest                       # synthetic data only; no ND2 files needed
```

## License

MIT
