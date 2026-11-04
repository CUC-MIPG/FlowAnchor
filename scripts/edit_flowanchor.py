#!/usr/bin/env python3
"""Run FlowAnchor on a video and its per-frame edit masks."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NEGATIVE_PROMPT = (
    "worst quality, low quality, blurry, absolute black, absolute white, low res, "
    "extra limbs, extra digits, misplaced objects, mutated anatomy, monochrome, horror"
)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path", type=Path, default=ROOT / "models/Wan2.1-T2V-1.3B-Diffusers"
    )
    parser.add_argument("--video_path", type=Path, required=True)
    parser.add_argument(
        "--mask_dir",
        type=Path,
        required=True,
        help="00000.png, 00001.png, ...; white denotes the edit region",
    )
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--src_prompt", required=True)
    parser.add_argument("--tar_prompt", required=True)
    parser.add_argument("--target_word", required=True)
    parser.add_argument("--negative_prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--T_steps", type=int, default=25)
    parser.add_argument(
        "--n_max",
        type=int,
        default=23,
        help="Active editing steps; 23 of 25 means skip the first two",
    )
    parser.add_argument(
        "--tar_guidance_scale",
        type=float,
        default=1.0,
        help="Source CFG is fixed at 1.0",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=1.0,
        help="AMM coefficient; 1.0 is the validated 81-frame setting",
    )
    parser.add_argument("--row_strength", type=float, default=0.3)
    parser.add_argument("--col_strength", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output FPS; defaults to the source video's FPS",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save_latents",
        action="store_true",
        help="Also save edited latents for regression checks",
    )
    return parser


def validate_args(args, parser):
    """Fail before loading a large model when paths or parameters are invalid."""
    if not 0 < args.n_max <= args.T_steps:
        parser.error("Require 0 < n_max <= T_steps")
    if (
        args.omega < 0
        or not 0 <= args.row_strength <= 1
        or not 0 <= args.col_strength <= 1
    ):
        parser.error("Require omega >= 0 and row/col strengths in [0, 1]")
    if args.fps is not None and args.fps <= 0:
        parser.error("fps must be positive")
    if (
        not args.video_path.is_file()
        or not args.mask_dir.is_dir()
        or not (args.model_path / "model_index.json").is_file()
    ):
        parser.error(
            "Check video_path, mask_dir and local model_path (model_index.json is required)"
        )
    if not args.overwrite and any(
        p.exists()
        for p in (
            args.output_path,
            args.output_path.with_suffix(".json"),
            args.output_path.with_suffix(".pt"),
        )
    ):
        parser.error(
            "Output already exists; use --overwrite or choose another output_path"
        )


def load_pipeline(args):
    """Load the local checkpoint using the validated initialization order."""
    import random
    import numpy as np
    import torch
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from src.flowanchor.pipeline_wan import WanPipeline
    from src.flowanchor.transformer_wan import WanTransformer3DModel

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dtype = getattr(torch, args.dtype)
    model = str(args.model_path.resolve())
    print("Loading local model:", model, flush=True)
    # Keep the loading order of the validated reference run, including replacement
    # with the custom transformer, to preserve random-number consumption.
    pipe = WanPipeline.from_pretrained(
        model, torch_dtype=dtype, local_files_only=True
    ).to(args.device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=3.0
    )
    pipe.transformer = WanTransformer3DModel.from_pretrained(
        model, subfolder="transformer", torch_dtype=dtype, local_files_only=True
    ).to(args.device)

    return pipe


def prepare_inputs(args, pipe):
    """Validate alignment, encode the source and prepare the method-specific masks."""
    import cv2
    import torch
    from PIL import Image
    from src.flowanchor.utils import (
        load_video_to_frames,
        encode_video_to_latents,
        load_masks,
        resize_mask_to_latent,
    )
    from src.flowanchor.ptp_utils import get_word_ids

    frames = load_video_to_frames(str(args.video_path))
    h, w = frames[0].shape[:2]
    if (len(frames) - 1) % 4 or h % 16 or w % 16:
        raise ValueError(
            "Use 4n+1 video frames and height/width divisible by 16; the example is 81 x 480 x 832"
        )
    masks = load_masks(str(args.mask_dir), len(frames))
    if tuple(masks.shape[-2:]) != (h, w):
        raise ValueError("Mask height/width must match the input video")
    ids = get_word_ids(args.tar_prompt, args.target_word, pipe.tokenizer)
    if not len(ids) or max(ids) >= 226:
        raise ValueError(
            "target_word must match target-prompt tokens within the encoded context"
        )
    cap = cv2.VideoCapture(str(args.video_path))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fps = args.fps or source_fps
    if fps <= 0:
        raise ValueError("Source FPS is unavailable; provide --fps")
    video_tensor = pipe.video_processor.preprocess_video(
        [Image.fromarray(frame) for frame in frames]
    )
    video_latents = encode_video_to_latents(pipe, video_tensor)
    if not torch.isfinite(video_latents).all():
        raise ValueError("Source VAE latents contain NaN/Inf")
    latent_mask = resize_mask_to_latent(masks, video_latents.shape)
    _, _, f, lh, lw = video_latents.shape
    pt, ph, pw = pipe.transformer.config.patch_size
    ca_mask = torch.nn.functional.interpolate(
        latent_mask,
        size=(f // pt, lh // ph, lw // pw),
        mode="trilinear",
        align_corners=False,
    )
    print(
        f"Input: {len(frames)} frames, {w}x{h}; latent shape: {tuple(video_latents.shape)}",
        flush=True,
    )
    video_info = dict(
        num_frames=len(frames), width=w, height=h, fps=fps, latent_frames=f
    )
    return video_latents, latent_mask, ca_mask, video_info


def save_result(args, pipe, edited, video_info, seconds, peak):
    """Decode the edited latents and save the video with its run metadata."""
    import torch
    from diffusers.utils import export_to_video

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    latent_hash = (
        hashlib.sha256(edited.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        if args.dtype != "bfloat16"
        else None
    )
    if args.save_latents:
        torch.save(edited.detach().cpu(), args.output_path.with_suffix(".pt"))
    with torch.no_grad():
        mean = torch.tensor(
            pipe.vae.config.latents_mean, device=edited.device, dtype=edited.dtype
        ).view(1, -1, 1, 1, 1)
        std = torch.tensor(
            pipe.vae.config.latents_std, device=edited.device, dtype=edited.dtype
        ).view(1, -1, 1, 1, 1)
        decoded = pipe.vae.decode(edited * std + mean, return_dict=False)[0]
    if not torch.isfinite(decoded).all():
        raise ValueError("Decoded video contains NaN/Inf")
    video = pipe.video_processor.postprocess_video(decoded, output_type="np")[0]
    export_to_video(video, str(args.output_path), fps=video_info["fps"])
    metadata = {
        k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    metadata.update(
        method="flowanchor",
        source_cfg=1.0,
        num_frames=video_info["num_frames"],
        width=video_info["width"],
        height=video_info["height"],
        fps=video_info["fps"],
        latent_frames=video_info["latent_frames"],
        latent_sha256=latent_hash,
        inference_seconds=seconds,
        peak_allocated_mib=peak,
        sar_zero_based_step_indices=[
            i
            for i in range(args.T_steps)
            if args.T_steps - i <= args.n_max and i in range(2, 10)
        ],
    )
    args.output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def main():
    parser = get_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    import torch
    from src.flowanchor.pipeline import FlowEditWan

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; check PyTorch and your GPU allocation")
    pipe = load_pipeline(args)
    video_latents, latent_mask, ca_mask, video_info = prepare_inputs(args, pipe)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)
    start = time.perf_counter()
    edited = FlowEditWan(
        pipe=pipe,
        scheduler=pipe.scheduler,
        x_src_video=video_latents,
        src_prompt=args.src_prompt,
        tar_prompt=args.tar_prompt,
        negative_prompt=args.negative_prompt,
        T_steps=args.T_steps,
        tar_guidance_scale=args.tar_guidance_scale,
        n_max=args.n_max,
        omega=args.omega,
        ca_mask=ca_mask,
        target_word=args.target_word,
        row_strength=args.row_strength,
        col_strength=args.col_strength,
        edit_mask=latent_mask,
    )
    if not torch.isfinite(edited).all():
        raise ValueError("Edited latents contain NaN/Inf")
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    seconds = time.perf_counter() - start
    peak = (
        torch.cuda.max_memory_allocated(args.device) / 1024**2
        if args.device.startswith("cuda")
        else None
    )
    save_result(args, pipe, edited, video_info, seconds, peak)
    print(f"Saved {args.output_path} ({seconds:.2f}s editing inference)", flush=True)


if __name__ == "__main__":
    main()
