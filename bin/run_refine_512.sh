#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/xiongle/Hi3D
export PATH="$ROOT/env/bin:$PATH"
export HF_HOME="$ROOT/hf_cache"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset CUDA_HOME
cd "$ROOT/src"
exec python -u demo/refine_mesh.py \
  --model "$ROOT/models/Hi3Dv1.0" \
  --image assets/example.png \
  --coarse-mesh "$ROOT/outputs/coarse_sphere.glb" \
  --output "$ROOT/outputs/refined_512.glb" \
  --resolution 512 \
  --num-inference-steps 25 \
  --seed 0 \
  --dino-model facebook/dinov2-large \
  --clip-model openai/clip-vit-large-patch14 \
  "$@"
