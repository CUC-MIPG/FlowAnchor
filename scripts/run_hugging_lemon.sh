#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
"${PYTHON:-python}" scripts/edit_flowanchor.py \
  --model_path "${MODEL_PATH:-$ROOT/models/Wan2.1-T2V-1.3B-Diffusers}" \
  --video_path examples/hugging_lemon/input.mp4 \
  --mask_dir examples/hugging_lemon/masks \
  --output_path outputs/hugging_lemon.mp4 \
  --src_prompt "A woman in a pink sweater, seen from behind, walks forward to hug a person in a gray top." \
  --tar_prompt "A woman in a lemon sweater, seen from behind, walks forward to hug a person in a gray top." \
  --target_word lemon --T_steps 25 --n_max 23 --tar_guidance_scale 1.0 \
  --omega 1.0 --row_strength 0.3 --col_strength 0.3 --seed 42 --fps 16 "$@"
