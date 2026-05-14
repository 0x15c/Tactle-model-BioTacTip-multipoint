from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import cv2

from temporal_flow.preprocess import PreprocessConfig, preprocess_frame_bgr


def main():
    parser = argparse.ArgumentParser(description="Convert a video into one temporal-flow sequence.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--seq-name", default=None)
    parser.add_argument("--width", type=int, default=350)
    parser.add_argument("--height", type=int, default=350)
    parser.add_argument("--crop-px", type=int, default=175)
    parser.add_argument("--crop-py", type=int, default=175)
    parser.add_argument("--crop-offset-x", type=int, default=0)
    parser.add_argument("--crop-offset-y", type=int, default=-8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    parser.add_argument("--ref-mode", choices=["first", "mean_first_n"], default="first")
    parser.add_argument("--mean-n", type=int, default=10)
    parser.add_argument("--no-radial-correction", action="store_true")
    parser.add_argument("--clahe", action="store_true")
    args = parser.parse_args()

    cfg = PreprocessConfig(
        width=args.width,
        height=args.height,
        crop_px=args.crop_px,
        crop_py=args.crop_py,
        crop_offset_x=args.crop_offset_x,
        crop_offset_y=args.crop_offset_y,
        radial_correction=not args.no_radial_correction,
        clahe=args.clahe,
    )

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.seq_name is None:
        existing = sorted([p for p in out_root.iterdir() if p.is_dir() and p.name.startswith("seq_")])
        idx = max([int(p.name.split("_")[-1]) for p in existing], default=-1) + 1
        seq_dir = out_root / f"seq_{idx:06d}"
    else:
        seq_dir = out_root / args.seq_name
    seq_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {args.video}")

    frames = []
    raw_idx = 0
    saved_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if raw_idx % args.stride == 0:
            gray = preprocess_frame_bgr(frame, cfg)
            frames.append(gray)
            cv2.imwrite(str(seq_dir / f"frame_{saved_idx:06d}.png"), gray)
            saved_idx += 1
            if args.max_frames and saved_idx >= args.max_frames:
                break
        raw_idx += 1
    cap.release()

    if not frames:
        raise RuntimeError("No frames were extracted")

    if args.ref_mode == "first":
        ref = frames[0]
    else:
        import numpy as np
        n = min(args.mean_n, len(frames))
        ref = np.mean(frames[:n], axis=0).clip(0, 255).astype("uint8")
    cv2.imwrite(str(seq_dir / "ref.png"), ref)
    print(f"Wrote {saved_idx} frames to {seq_dir}")


if __name__ == "__main__":
    main()
