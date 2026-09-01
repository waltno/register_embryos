# register_embryos
<img width="1052" height="315" alt="image" src="https://github.com/user-attachments/assets/85e17895-2682-4919-bf63-4a751020c978" />


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

**Resuming, and QC.** `PrepConfig.load(cohort_dir)` rebuilds the config from the two
JSON sidecars, so a session can be picked up later (`wf.prepare()` reloads
automatically too). `preview_contrast(volume, config)` renders the accepted
**rotation and contrast together** — pass the whole `config`, not `config.contrast`,
or the figure shows contrast on an unrotated embryo and a wrong rotation passes QC
unnoticed. The third panel reports the fraction of pixels above the signal
threshold, which is the number worth checking.

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

| mode | z input | Cellpose runs | label ids | one nucleus becomes | signal assignment |
|---|---|---|---|---|---|
| `2d` | binned (`bin_size=7`) | once per z-plane | restart on every plane | **one row per plane**, ids unrelated between planes | within each plane |
| `2d+link` | binned | once per z-plane (identical to `2d`) | linked across planes by mask IoU | **one row**, true 3D centroid | within each plane |
| `3d` | **unbinned** (`bin_size=1`) | once over the whole volume (`do_3D`, voxel anisotropy) | consistent by construction | **one row**, true 3D centroid | through the volume, in µm |

`2d` and `2d+link` do *exactly the same segmentation* — same Cellpose calls, same
masks, same cost. The only difference is identity: `2d+link` then walks z and gives
a label the id of the label below it wherever their pixel IoU clears a threshold
(default 0.25). That matters because per-plane ids make a nucleus spanning two
planes look like two nuclei at nearly the same xy — duplicate points stacked in z,
which ICP then fits as if they were real structure. `2d+link` removes those for the
cost of one pass over the masks, with no extra Cellpose work.

What `2d+link` *cannot* do is split a blob that 2D already merged — only a genuine
3D pass can. And it keeps per-plane assignment, so it does not get 3D's
micrometre-aware territory either.

Choosing: `2d` when you only want per-plane intensities and will not register;
`2d+link` as the default when you will register but cannot spend GPU time;
`3d` when nucleus identity has to be right and you have a GPU.

**3D takes the whole z-stack, unbinned — this is enforced, not advisory.** z-binning
is a concession made for 2D: it max-projects several planes into one so each plane
carries enough signal to segment alone. That is exactly the information 3D works
from. At the 2D-tuned `bin_size=7` (1.5 µm z-step → 10.5 µm per plane) a ~6 µm
nucleus spans 0.57 planes, so there is nothing to link across z and `do_3D`
degenerates into a slow 2D run. `segment(mode="3d")` refuses a binned volume;
`bin_size` defaults per mode, so `--mode 3d` loads unbinned automatically.

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

**Why automatic contrast uses the 90th percentile, not the 1st.** An HCR channel
is mostly background — measured on a real 20× dorsal stack, the median normalised
intensity is 0.004 and the 99.5th percentile is only 0.098. The 1st percentile of
the *image* therefore sits at the bottom of the background, not above it: it floors
nothing, and the narrow high limit then stretches the dim background shoulder all
the way to full scale. On that real stack, `p1/p99.5` put **36–41 % of pixels above
the 0.05 signal threshold**, so essentially every nucleus read as expressing.
`p90/p99.9` gives 6.5–8.2 %. `auto_contrast_limits` reports the resulting positive
fraction per channel and warns above 25 %, so a bad automatic choice is visible
rather than silent — but this is still the reason to use the widget.

**Why registration downsamples uniformly in space.** Uniform-at-random sampling
keeps dense regions dense, and ICP then fits the dense regions and ignores the
sparse ones. `isotropic_downsample` normalises each axis to [0,1] first, so z (a
few dozen bins) is weighted like x and y (a thousand pixels).

**Why registration trusts your manual orientation.** A 12-somite dorsal nucleus
cloud is nearly a disc of revolution — measured on a real cohort, principal extents
of ~180 × 155 × 4, an in-plane aspect of only **1.18**. Nothing in the nucleus
positions pins down the anterior–posterior angle, and mean nearest-neighbour distance
scores a 180°-flipped fit about as well as a correct one (7.40 vs 7.38 px on that
cohort — indistinguishable). Left unconstrained, ICP flipped three of seven embryos
end-for-end and tilted a fourth by 31°, silently destroying the orientation that had
been set by eye in the widget, where anterior is obvious.

So `CohortWorkflow.register()` defaults to `trust_orientation=True` whenever
orientations were recorded: no PCA re-derivation, rotation about z only, capped at
±30°. Judged on gene-domain agreement — which nucleus positions cannot provide,
because positions are near-symmetric and expression domains are not — that cut the
cohort spread for wt1a from 69 → 43 px and tbx1 from 118 → 65 px. The applied
in-plane rotation is printed per embryo, and anything beyond 90° is flagged.

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

**Why XZ/YZ panels are not forced to equal aspect.** `x` is in xy pixels and `z` in
bin indices — incommensurate units. Forcing `aspect="equal"` across them squashes
the embryo into a flat pancake that looks like a property of the sample rather than
of the axes. XY panels (pixels against pixels) get equal aspect; XZ/YZ autoscale
unless you pass `z_aspect=<voxel anisotropy>`, which draws them in true proportion.
Axis labels state their unit.

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
