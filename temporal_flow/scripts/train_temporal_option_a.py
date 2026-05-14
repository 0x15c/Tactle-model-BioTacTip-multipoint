from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from temporal_flow.checkpoint import save_checkpoint
from temporal_flow.dataset import TemporalFlowClipDataset
from temporal_flow.losses import photometric_loss, spatial_smoothness_loss, temporal_update_loss
from temporal_flow.model import TemporalResidualUNet
from temporal_flow.warp import warp_image


def train_one_epoch(model, loader, optimizer, device, args, epoch: int):
    model.train()
    running = {"loss": 0.0, "img": 0.0, "spatial": 0.0, "temporal": 0.0}
    pbar = tqdm(loader, desc=f"train {epoch}")
    for batch in pbar:
        ref = batch["ref"].to(device)              # Bx1xHxW
        frames = batch["frames"].to(device)        # BxKx1xHxW
        b, k, _, h, w = frames.shape
        u_prev = torch.zeros((b, 2, h, w), device=device, dtype=ref.dtype)

        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        stats = {"img": 0.0, "spatial": 0.0, "temporal": 0.0}

        for t in range(k):
            cur = frames[:, t]
            x = torch.cat([ref, cur, u_prev], dim=1)
            du = model(x)
            u_t = u_prev + du
            warped = warp_image(ref, u_t)

            l_img = photometric_loss(warped, cur, args.photo_loss)
            l_sp = spatial_smoothness_loss(u_t)
            l_tmp = temporal_update_loss(du)
            loss = l_img + args.lambda_spatial * l_sp + args.lambda_temporal * l_tmp
            total = total + loss
            stats["img"] += float(l_img.detach())
            stats["spatial"] += float(l_sp.detach())
            stats["temporal"] += float(l_tmp.detach())

            # Truncated BPTT: keep temporal state but do not backprop through all previous frames.
            u_prev = u_t.detach()

        total = total / k
        total.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        running["loss"] += float(total.detach())
        for key in stats:
            running[key] += stats[key] / k
        n = max(1, pbar.n + 1)
        pbar.set_postfix({key: running[key] / n for key in running})
    return {key: value / max(1, len(loader)) for key, value in running.items()}


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    running = {"loss": 0.0, "img": 0.0, "spatial": 0.0, "temporal": 0.0}
    for batch in tqdm(loader, desc="val"):
        ref = batch["ref"].to(device)
        frames = batch["frames"].to(device)
        b, k, _, h, w = frames.shape
        u_prev = torch.zeros((b, 2, h, w), device=device, dtype=ref.dtype)
        total = 0.0
        stats = {"img": 0.0, "spatial": 0.0, "temporal": 0.0}
        for t in range(k):
            cur = frames[:, t]
            x = torch.cat([ref, cur, u_prev], dim=1)
            du = model(x)
            u_t = u_prev + du
            warped = warp_image(ref, u_t)
            l_img = photometric_loss(warped, cur, args.photo_loss)
            l_sp = spatial_smoothness_loss(u_t)
            l_tmp = temporal_update_loss(du)
            loss = l_img + args.lambda_spatial * l_sp + args.lambda_temporal * l_tmp
            total += loss
            stats["img"] += float(l_img)
            stats["spatial"] += float(l_sp)
            stats["temporal"] += float(l_tmp)
            u_prev = u_t
        running["loss"] += float(total / k)
        for key in stats:
            running[key] += stats[key] / k
    return {key: value / max(1, len(loader)) for key, value in running.items()}


def main():
    parser = argparse.ArgumentParser(description="Train Option A temporal residual flow model.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--save-dir", default="temporal_flow/checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--width", type=int, default=350)
    parser.add_argument("--height", type=int, default=350)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--lambda-spatial", type=float, default=0.05)
    parser.add_argument("--lambda-temporal", type=float, default=0.02)
    parser.add_argument("--photo-loss", choices=["charbonnier", "l1", "mse"], default="charbonnier")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = TemporalFlowClipDataset(
        args.data_root,
        clip_length=args.clip_length,
        width=args.width,
        height=args.height,
        samples_per_epoch=args.samples_per_epoch if args.samples_per_epoch > 0 else None,
    )

    if args.samples_per_epoch > 0:
        train_set = dataset
        val_set = TemporalFlowClipDataset(args.data_root, args.clip_length, args.width, args.height, samples_per_epoch=max(8, args.batch_size * 2))
    else:
        val_len = max(1, int(len(dataset) * args.val_ratio)) if len(dataset) > 1 else 0
        train_len = len(dataset) - val_len
        if val_len > 0:
            train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
        else:
            train_set, val_set = dataset, None

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) if val_set else None

    model = TemporalResidualUNet(in_channels=4, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device, args, epoch)
        val_stats = validate(model, val_loader, device, args) if val_loader else train_stats
        print(f"epoch={epoch} train={train_stats} val={val_stats}")

        save_checkpoint(save_dir / "last.pt", model, optimizer, epoch, extra={"args": vars(args), "val": val_stats})
        if val_stats["loss"] < best:
            best = val_stats["loss"]
            save_checkpoint(save_dir / "best.pt", model, optimizer, epoch, extra={"args": vars(args), "val": val_stats})
            print(f"Saved new best: {best:.6f}")


if __name__ == "__main__":
    main()
