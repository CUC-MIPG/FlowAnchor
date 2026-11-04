#!/usr/bin/env python3
"""Download the pinned Wan2.1 checkpoint, optionally through a Hugging Face mirror."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir", type=Path, default=ROOT / "models/Wan2.1-T2V-1.3B-Diffusers"
    )
    parser.add_argument(
        "--endpoint", help="For example https://hf-mirror.com; also accepts HF_ENDPOINT"
    )
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use optional hf_transfer (pip install hf_transfer==0.1.9)",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    if args.fast:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    # Import after configuring the endpoint: huggingface_hub reads it on import.
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=REPO,
        revision=args.revision,
        local_dir=str(args.output_dir),
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "text_encoder/*",
            "tokenizer/*",
            "transformer/*",
            "vae/*",
            "README.md",
        ],
        max_workers=args.workers,
    )
    print("Model ready:", path)


if __name__ == "__main__":
    main()
