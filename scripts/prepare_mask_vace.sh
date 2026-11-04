#!/usr/bin/env bash
# Use the official VACE annotator in its own environment, then convert its mask video.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" || $# -ne 3 ]]; then
  echo "Usage: VACE_ROOT=/path/to/VACE [VACE_PYTHON=python] [PYTHON=python] bash $0 INPUT_VIDEO FIRST_MASK OUTPUT_DIR"
  echo "OUTPUT_DIR must be empty. VACE_PYTHON runs VACE; PYTHON runs the PNG converter."
  [[ "${1:-}" == "--help" ]] && exit 0 || exit 2
fi
: "${VACE_ROOT:?Set VACE_ROOT to your official ali-vilab/VACE checkout}"
PYTHON="${PYTHON:-python}"
VACE_PYTHON="${VACE_PYTHON:-python}"
VIDEO="$("$PYTHON" -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).resolve())' "$1")"
MASK="$("$PYTHON" -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).resolve())' "$2")"
OUT="$("$PYTHON" -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).resolve())' "$3")"
[[ -f "$VIDEO" && -f "$MASK" ]] || { echo "Input video or first-frame mask is missing" >&2; exit 2; }
[[ -f "$VACE_ROOT/vace/vace_preproccess.py" ]] || { echo "VACE_ROOT does not contain vace/vace_preproccess.py" >&2; exit 2; }
if [[ -d "$OUT" && -n "$(ls -A "$OUT")" ]]; then
  echo "OUTPUT_DIR must be empty" >&2; exit 2
fi
mkdir -p "$OUT"
(
  cd "$VACE_ROOT"
  "$VACE_PYTHON" vace/vace_preproccess.py --task inpainting --mode masktrack \
    --mask "$MASK" --video "$VIDEO" --pre_save_dir "$OUT/vace"
)
"$PYTHON" "$ROOT/scripts/mask_video_to_frames.py" \
  --source_video "$VIDEO" --mask_video "$OUT/vace/src_mask-inpainting.mp4" --output_dir "$OUT/masks"
