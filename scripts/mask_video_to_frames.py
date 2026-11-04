#!/usr/bin/env python3
"""Convert VACE's mask video to a source-aligned sequence of binary PNG masks."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask_video", type=Path, required=True)
    parser.add_argument("--source_video", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert only if your mask video uses black for the edit region",
    )
    args = parser.parse_args()
    if not 0 <= args.threshold <= 255:
        parser.error("threshold must be in [0, 255]")
    if not args.mask_video.is_file() or not args.source_video.is_file():
        parser.error("Input videos must exist")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output_dir must be empty to avoid mixing mask sequences")
    import cv2

    source, mask = cv2.VideoCapture(str(args.source_video)), cv2.VideoCapture(
        str(args.mask_video)
    )
    frames = []
    try:
        if not source.isOpened() or not mask.isOpened():
            raise ValueError("Cannot open source or mask video")
        sfps, mfps = source.get(cv2.CAP_PROP_FPS), mask.get(cv2.CAP_PROP_FPS)
        if sfps > 0 and mfps > 0 and abs(sfps - mfps) > 0.01:
            raise ValueError(
                f"FPS mismatch: source={sfps}, mask={mfps}; use matching videos without resampling"
            )
        while True:
            ok_source, src = source.read()
            ok_mask, img = mask.read()
            if ok_source != ok_mask:
                raise ValueError("Mask and source video frame counts differ")
            if not ok_source:
                break
            if img.shape[:2] != src.shape[:2]:
                raise ValueError("Mask and source dimensions differ")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, args.threshold, 255, cv2.THRESH_BINARY)
            frames.append(cv2.bitwise_not(binary) if args.invert else binary)
    finally:
        source.release()
        mask.release()
    if not frames:
        raise ValueError("No frames decoded")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        path = args.output_dir / f"{index:05d}.png"
        if not cv2.imwrite(str(path), frame):
            raise OSError(f"Cannot write {path}")
    print(f"Saved {len(frames)} masks to {args.output_dir}; white = edit region")


if __name__ == "__main__":
    main()
