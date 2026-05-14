from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from temporal_flow.checkpoint import load_checkpoint
from temporal_flow.dataset import read_gray_tensor
from temporal_flow.model import TemporalResidualUNet
from temporal_flow.viz import flow_to_hsv, make_demo_panel, tensor_to_gray_u8
from temporal_flow.warp import warp_image


def main():
    parser = argparse.ArgumentParser(description="Run temporal residual flow on one saved sequence.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--width", type=int, default=350)
    parser.add_argument("--height", type=int, default=350)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seq = Path(args.seq)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(seq.glob("frame_*.png"))
    if not frame_paths:
        raise RuntimeError(f"No frame_*.png found in {seq}")
    ref_path = seq / "ref.png"
    if not ref_path.exists():
        ref_path = frame_paths[0]

    device = torch.device(args.device)
    model = TemporalResidualUNet(in_channels=4, base_channels=args.base_channels).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    ref = read_gray_tensor(ref_path, args.width, args.height).unsqueeze(0).to(device)
    ref_u8 = tensor_to_gray_u8(ref)
    u_prev = torch.zeros((1, 2, args.height, args.width), device=device)

    with torch.no_grad():
        for i, path in enumerate(tqdm(frame_paths)):
            cur = read_gray_tensor(path, args.width, args.height).unsqueeze(0).to(device)
            x = torch.cat([ref, cur, u_prev], dim=1)
            du = model(x)
            u_t = u_prev + du
            warped = warp_image(ref, u_t)

            cur_u8 = tensor_to_gray_u8(cur)
            warped_u8 = tensor_to_gray_u8(warped)
            panel = make_demo_panel(ref_u8, cur_u8, warped_u8, u_t)
            cv2.imwrite(str(out / f"panel_{i:06d}.png"), panel)
            cv2.imwrite(str(out / f"flow_hsv_{i:06d}.png"), flow_to_hsv(u_t))
            np.save(str(out / f"flow_{i:06d}.npy"), u_t[0].detach().cpu().numpy())
            u_prev = u_t.detach()

    print(f"Wrote outputs to {out}")


if __name__ == "__main__":
    main()
