from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import cv2

from temporal_flow.preprocess import PreprocessConfig, configure_camera, preprocess_frame_bgr


def next_seq_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("seq_")])
    idx = 0
    if existing:
        idx = max(int(p.name.split("_")[-1]) for p in existing) + 1
    seq = root / f"seq_{idx:06d}"
    seq.mkdir(parents=True, exist_ok=True)
    return seq


def main():
    parser = argparse.ArgumentParser(description="Capture temporal-flow training sequences from a camera.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--width", type=int, default=350)
    parser.add_argument("--height", type=int, default=350)
    parser.add_argument("--crop-px", type=int, default=175)
    parser.add_argument("--crop-py", type=int, default=175)
    parser.add_argument("--crop-offset-x", type=int, default=0)
    parser.add_argument("--crop-offset-y", type=int, default=-8)
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

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")
    configure_camera(cap)

    out_root = Path(args.out_root)
    seq_dir = next_seq_dir(out_root)
    ref = None
    recording = False
    count = 0

    print("Controls: r=set ref, s=start/stop recording, n=new sequence, q=quit")
    print(f"Current sequence: {seq_dir}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed")
            break
        gray = preprocess_frame_bgr(frame, cfg)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        status = f"seq={seq_dir.name} rec={recording} count={count} ref={'yes' if ref is not None else 'no'}"
        cv2.putText(vis, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.imshow("temporal-flow capture", vis)

        if recording:
            if ref is None:
                ref = gray.copy()
                cv2.imwrite(str(seq_dir / "ref.png"), ref)
            cv2.imwrite(str(seq_dir / f"frame_{count:06d}.png"), gray)
            count += 1
            if count >= args.num_frames:
                recording = False
                print(f"Finished sequence {seq_dir} with {count} frames")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            ref = gray.copy()
            cv2.imwrite(str(seq_dir / "ref.png"), ref)
            print(f"Saved reference: {seq_dir / 'ref.png'}")
        if key == ord("s"):
            recording = not recording
            print(f"Recording: {recording}")
        if key == ord("n"):
            seq_dir = next_seq_dir(out_root)
            ref = None
            recording = False
            count = 0
            print(f"New sequence: {seq_dir}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
