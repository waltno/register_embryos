#!/bin/bash
#$ -S /bin/bash
#$ -N re_seg3d
#$ -cwd
#$ -j y
#$ -o $HOME/tmp_files/nobackup/
#$ -l mfree=64G
#$ -l h_rt=23:59:59
#$ -l gpgpu=TRUE
#$ -l cuda=1
#
# 3D Cellpose nuclear segmentation on a GPU node (UW GS SGE cluster).
#
# 3D CPSAM over a 1024x1024 stack is not viable on CPU -- it is minutes per
# z-bin, so a 7-embryo cohort runs for many hours.  On a GPU it is tractable.
#
# GPU nodes visible from the trapnell queues, as of writing:
#   t001 (1 device), t005 (2), t008 (2), t010 (2), t011 (4), t012 (4)
# `-l gpgpu=TRUE -l cuda=1` requests one device; SGE sets CUDA_VISIBLE_DEVICES.
# Check current availability with:
#   qstat -F gpgpu,cuda.devices | awk '/\.q@/{q=$1} /cuda.devices=[1-9]/{print q,$0}'
#
# Submit:
#   qsub -q trapnell-short.q docs/qsub_gpu_segmentation.sh <ND2_DIR> <OUT_DIR> [COHORT]
# Long jobs (a big cohort) belong in trapnell-long.q instead.

set -euo pipefail

ND2_DIR="${1:?usage: qsub qsub_gpu_segmentation.sh ND2_DIR OUT_DIR [COHORT]}"
OUT_DIR="${2:?usage: qsub qsub_gpu_segmentation.sh ND2_DIR OUT_DIR [COHORT]}"
COHORT="${3:-}"

PY=/net/trapnell/vol1/home/waltno/miniconda3/envs/napari-env/bin/python

echo "**** host      : $(hostname)"
echo "**** started   : $(date)"
echo "**** nd2 dir   : $ND2_DIR"
echo "**** out dir   : $OUT_DIR"
echo "**** cohort    : ${COHORT:-<all>}"
echo "**** CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Fail early and loudly if the GPU is not actually usable: falling back to CPU
# here would look like a hung job rather than a misconfigured request.
"$PY" - <<'PYCHECK'
import sys
import torch
print(f"     torch {torch.__version__}, built for CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit("**** ERROR: no CUDA device visible. Did the job land on a GPU node? "
             "Re-submit with -l gpgpu=TRUE -l cuda=1 to a queue that has one.")
print(f"     GPU: {torch.cuda.get_device_name(0)} "
      f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
PYCHECK

if [ -n "$COHORT" ]; then
    SCOPE=(--cohort "$COHORT")
else
    SCOPE=(--all --skip-single)
fi

# --gpu is what makes this worth a GPU node at all.
# Orientation and contrast are read from <OUT_DIR>/<cohort>/{orientation,contrast_limits}.json
# if the widget has been run; otherwise no rotation and percentile contrast.
"$PY" -m register_embryos.cli run "$ND2_DIR" \
    --output "$OUT_DIR" \
    "${SCOPE[@]}" \
    --mode 3d \
    --gpu \
    --workers 1 \
    --bin-size 7

echo "**** finished  : $(date)"
