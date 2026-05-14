from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from temporal_flow.checkpoint import load_checkpoint
from temporal_flow.model import TemporalResidualUNet
from temporal_flow.preprocess import PreprocessConfig, configure_camera, preprocess_frame_bgr
from temporal_flow.viz import make_demo_panel, tensor_to_gray_u8
from temporal_flow.warp import warp_image


def gray_to_tensor(gray: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return x.to(device)


def main():
    parser = argparse.ArgumentParser(description="Real-time temporal residual flow demo.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", default=None, help="Use a video file instead of camera if provided")
    parser.add_argument("--width", type=int, default=350)
    parser.add_argument("--height", type=int, default=350)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--crop-px", type=int, default=175)
    parser.add_argument("--crop-py", type=int, default=175)
    parser.add_argument("--crop-offset-x", type=int, default=0)
    parser.add_argument("--crop-offset-y", type=int, default=-8)
    parser.add_argument("--no-radial-correction", action="store_true")
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-video", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = TemporalResidualUNet(in_channels=4, base_channels=args.base_channels).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    source = args.video if args.video is not None else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source {source}")
    if args.video is None:
        configure_camera(cap)

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

    ref_t = None
    ref_u8 = None
    u_prev = torch.zeros((1, 2, args.height, args.width), device=device)
    writer = None
    last = time.time()

    print("Controls: r=set reference/reset flow, q=quit")

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = preprocess_frame_bgr(frame, cfg)
            cur_t = gray_to_tensor(gray, device)

            if ref_t is None:
                ref_t = cur_t.clone()
                ref_u8 = gray.copy()
                u_prev.zero_()

            x = torch.cat([ref_t, cur_t, u_prev], dim=1)
            du = model(x)
            u_t = u_prev + du
            warped = warp_image(ref_t, u_t)

            warped_u8 = tensor_to_gray_u8(warped)
            panel = make_demo_panel(ref_u8, gray, warped_u8, u_t)
            now = time.time()
            fps = 1.0 / max(now - last, 1e-6)
            last = now
            mag = torch.sqrt((u_t * u_t).sum(dim=1)).mean().item()
            cv2.putText(panel, f"FPS={fps:.1f} mean|u|={mag:.3f}px", (10, panel.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            cv2.imshow("temporal residual flow", panel)

            if args.save_video and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save_video, fourcc, 30.0, (panel.shape[1], panel.shape[0]))
            if writer is not None:
                writer.write(panel)

            u_prev = u_t.detach()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                ref_t = cur_t.clone()
                ref_u8 = gray.copy()
                u_prev = torch.zeros_like(u_prev)
                print("Reference reset")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
