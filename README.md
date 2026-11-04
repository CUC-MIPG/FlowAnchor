# FlowAnchor: Stabilizing the Editing Signal for Inversion-Free Video Editing

[Project page](https://cuc-mipg.github.io/FlowAnchor.github.io/) · [Paper](https://arxiv.org/abs/2604.22586) · [Hugging Face Papers](https://huggingface.co/papers/2604.22586)

**ACM Multimedia 2026 — development version for testing.**

FlowAnchor is a training-free framework for inversion-free video editing. Spatial-aware Attention Refinement (SAR) anchors the editing region through cross-attention modulation, and Adaptive Magnitude Modulation (AMM) strengthens the editing signal.

![FlowAnchor teaser](assets/teaser_01.png)

This version includes independent implementations and inference entry points for FlowAnchor and Wan-Edit+Mask. The bundled hugging example changes a pink sweater to a lemon-colored sweater.

## Installation

Tested with Python 3.10, PyTorch 2.6.0, CUDA 12.4, and one NVIDIA L40 (46 GB). The checkpoint download occupies approximately 29 GB. GPU memory depends on video resolution and length.

```bash
git clone --branch codex/dev-release https://github.com/CUC-MIPG/FlowAnchor.git
cd FlowAnchor
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Both inference entries support `--help`. Run the commands below from the repository root.

## Model download

```bash
python scripts/download_models.py
```

For the Hugging Face mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com python scripts/download_models.py
```

The destination defaults to `models/Wan2.1-T2V-1.3B-Diffusers`; override it with `--output_dir`. The downloader pins [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) to revision `0fad780a534b6463e45facd96134c9f345acfa5b`, used for the reference result. Standard interrupted downloads resume when rerun. Inference reads local model files and works offline.

Optional parallel range transfer:

```bash
python -m pip install hf_transfer==0.1.9
python scripts/download_models.py --endpoint https://hf-mirror.com --fast
```

## Hugging example

```bash
bash scripts/run_hugging_lemon.sh
```

To reuse a model in another directory:

```bash
MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers bash scripts/run_hugging_lemon.sh
```

The script uses `examples/hugging_lemon/input.mp4` and the 81 supplied PNG masks. It saves `outputs/hugging_lemon.mp4` and a JSON parameter record. Existing outputs are protected; pass `--overwrite` to replace them.

`reference.mp4` is the previously generated reference result; `reference_config.json` records its settings and model revision. The supplied masks are exactly those used for that run. VACE installation is optional for running this example.

| Setting | Value |
| --- | --- |
| Input | 81 frames, 832 × 480, 16 FPS |
| Total / active steps | 25 / 23, skipping the first two |
| Source / target CFG | 1.0 / 1.0 |
| AMM coefficient `omega` | 1.0 |
| SAR row / column strengths | 0.3 / 0.3 |
| SAR layers | All 30 cross-attention layers |
| SAR schedule in this dev | Zero-based indices 2–9, intersected with active editing steps |
| Seed / dtype / flow shift | 42 / float16 / 3.0 |

The reference editing stage took approximately 214 seconds on an L40, excluding model loading and VAE encoding/decoding. Runtime and exact numerical reproducibility depend on hardware and environment.

## Your own video

```bash
python scripts/edit_flowanchor.py \
  --model_path models/Wan2.1-T2V-1.3B-Diffusers \
  --video_path /path/to/input.mp4 --mask_dir /path/to/masks \
  --output_path outputs/my_edit.mp4 \
  --src_prompt "A woman in a pink sweater, seen from behind, walks forward to hug a person in a gray top." \
  --tar_prompt "A woman in a lemon sweater, seen from behind, walks forward to hug a person in a gray top." \
  --target_word lemon --T_steps 25 --n_max 23 --tar_guidance_scale 1.0 \
  --omega 1.0 --row_strength 0.3 --col_strength 0.3 --seed 42
```

- Supply one grayscale mask per source frame: `00000.png`, `00001.png`, etc. White means edit; black means background. Mask dimensions must match the source video.
- Use `4n+1` frames and height/width divisible by 16. Inputs are preserved without automatic resizing or frame resampling.
- `--target_word` must occur in the target prompt and may be a phrase. Missing target tokens are reported rather than silently disabling SAR.
- Source CFG is fixed at 1.0; `--tar_guidance_scale` controls target CFG.
- `--omega` is the explicit AMM coefficient in this dev. For 81 pixel-space frames, latent length `F=21` gives `omega=1` at `gamma=1`. For other lengths, the frame-dependent value is `gamma * log(F) / log(21)`, where `F=(num_frames-1)/4+1`.
- Output FPS defaults to the source FPS; `--fps` overrides it. `--dtype` accepts `float16`, `bfloat16`, or `float32`; float16 is the reference setting.
- `--save_latents` also writes a `.pt` file for regression checks. Every successful run writes its parameters and video dimensions to a JSON sidecar.

## Wan-Edit+Mask interface

The baseline has its own inference entry and source tree. At every active editing step, it gates the velocity difference with the edit mask:

```python
V_delta = (V_target - V_source) * mask
z_edit = z_edit + (t_next - t) * V_delta
```

```bash
python scripts/edit_wanedit_mask.py \
  --model_path models/Wan2.1-T2V-1.3B-Diffusers \
  --video_path /path/to/input.mp4 --mask_dir /path/to/masks \
  --output_path outputs/wanedit_mask.mp4 \
  --src_prompt "Source video description" --tar_prompt "Target video description" \
  --T_steps 25 --n_max 23 --tar_guidance_scale 7.5 --seed 42
```

Source CFG is fixed at 1.0. Target CFG defaults to 7.5 and can be changed explicitly. No SAR or AMM is applied. Both source trees retain their own mask-loading and latent-mask preprocessing functions. No baseline showcase or separate baseline shell runner is included.

## Prepare masks with VACE

Use the official [VACE repository](https://github.com/ali-vilab/VACE). Follow its [preprocessing installation instructions](https://github.com/ali-vilab/VACE#setup-for-preprocess-tools) to install annotator dependencies and [VACE-Annotators](https://huggingface.co/ali-vilab/VACE-Annotators) weights, preferably in a separate environment. The wrapper targets VACE commit `48eb44f1c4be87cc65a98bff985a26976841e9f3`.

Given a first-frame mask, run inside VACE:

```bash
python vace/vace_preproccess.py --task inpainting --mode masktrack \
  --mask /absolute/path/to/first_mask.png --video /absolute/path/to/input.mp4 \
  --pre_save_dir /absolute/path/to/processed
```

VACE writes `src_mask-inpainting.mp4`. Convert it to PNG masks from the FlowAnchor root:

```bash
python scripts/mask_video_to_frames.py \
  --mask_video /absolute/path/to/processed/src_mask-inpainting.mp4 \
  --source_video /absolute/path/to/input.mp4 --output_dir /absolute/path/to/masks
```

The converter checks matching frame counts, dimensions and FPS, then thresholds to 0/255 without resampling. Use `--invert` only when the mask video uses black for the editing region. The output directory must be empty.

The wrapper runs both steps:

```bash
VACE_ROOT=/path/to/VACE \
VACE_PYTHON=/path/to/vace-env/bin/python \
PYTHON=/path/to/flowanchor-env/bin/python \
bash scripts/prepare_mask_vace.sh /path/to/input.mp4 /path/to/first_mask.png /path/to/new_mask_workdir
```

The work directory must be empty. VACE outputs go into `new_mask_workdir/vace/`; PNG masks go into `new_mask_workdir/masks/`. The external tracker is not bundled with this repository.

## Layout

```text
scripts/                   Inference entries, demo, downloader and mask tools
src/flowanchor/            FlowAnchor pipeline, SAR, Wan backbone and utilities
src/wanedit_mask/          Independent Wan-Edit+Mask pipeline, backbone and utilities
examples/hugging_lemon/    Input video, masks, first-frame mask and reference result
assets/                   Project teaser
tests/                    CPU equation and mask-file contract checks
```

## Verification

Run the lightweight checks without a GPU or model download:

```bash
python -m unittest discover -s tests -v
```

These check SAR against a hand-calculated example, AMM channel reduction and normalization, mask-gated baseline updates, and PNG mask export/alignment errors. Run the hugging example for full GPU inference verification. VACE's external tracking model requires its own environment and weights; it is not exercised by these CPU checks.

Validation on 2026-11-04: the reorganized FlowAnchor entry reproduced all 81 reference frames exactly, including identical MP4 bytes, on the tested L40 environment. The entry-point readability refactor also preserved the edited latents exactly. The independent Wan-Edit+Mask entry passed a small GPU inference smoke check. See `examples/hugging_lemon/validation.json` for the measured results and scope.

## Acknowledgements

Built on [Wan2.1](https://github.com/Wan-Video/Wan2.1), [Diffusers](https://github.com/huggingface/diffusers), and the FlowEdit-style editing formulation. Optional mask tracking uses [VACE](https://github.com/ali-vilab/VACE). The hugging example and masks come from our [FiVE-Bench](https://huggingface.co/datasets/LIMinghan/FiVE-Fine-Grained-Video-Editing-Benchmark) experiment assets.

## Citation

```bibtex
@article{chen2026flowanchor,
  title={FlowAnchor: Stabilizing the Editing Signal for Inversion-Free Video Editing},
  author={Chen, Ze and Chen, Lan and Li, Yuanhang and Mao, Qi},
  journal={arXiv preprint arXiv:2604.22586},
  year={2026}
}
```
