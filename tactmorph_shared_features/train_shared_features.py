import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tactmorph.loss import bending_energy_loss, similarity_loss  # noqa: E402
from tactmorph.preprocessing import preprocess_registration_image  # noqa: E402
from tactmorph_shared_features.model import SharedFeatureTactMorph2D  # noqa: E402


def parse_size(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    text = str(value).lower().replace("x", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) == 1:
        size = int(parts[0])
        return (size, size)
    if len(parts) == 2:
        return (int(parts[0]), int(parts[1]))
    raise argparse.ArgumentTypeError(f"Invalid size: {value}")


def parse_float_list(value):
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    parts = [part.strip() for part in str(value).replace(";", ",").split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected one or more comma-separated float values.")
    return [float(part) for part in parts]


def load_gray(path, resize_to=None):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    if resize_to is not None:
        image = preprocess_registration_image(image, mode="none", size=resize_to)
    return image.astype(np.float32)


def load_images_in_memory(image_dir, resize_to, preprocess_mode, model_input_size):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_dir = Path(image_dir)
    files = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in exts)
    if not files:
        raise ValueError(f"No images found in {image_dir}")

    names = []
    model_tensors = []
    for path in files:
        original = load_gray(path, resize_to=resize_to)
        if preprocess_mode != "none":
            model_img = preprocess_registration_image(
                original,
                mode=preprocess_mode,
                size=model_input_size,
            )
        else:
            model_img = original

        names.append(path.name)
        model_tensors.append(torch.from_numpy(model_img)[None, ...])

    return names, model_tensors


def build_batch(tensors, batch_indices, fixed_tensor=None):
    moving = torch.stack([tensors[i] for i in batch_indices], dim=0)
    if fixed_tensor is not None:
        fixed = fixed_tensor.expand(moving.shape[0], -1, -1, -1)
    else:
        fixed_indices = [random.randint(0, len(tensors) - 1) for _ in batch_indices]
        fixed = torch.stack([tensors[i] for i in fixed_indices], dim=0)
    return moving, fixed


def resize_to_square(img, square_size):
    return F.interpolate(
        img.unsqueeze(0),
        size=(square_size, square_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def augment_pairs(
    moving,
    fixed,
    hflip_prob,
    rotate_choices,
):
    model_square_size = int(np.min(moving.shape[-2:]))

    new_moving = []
    new_fixed = []

    for i in range(moving.shape[0]):
        m = moving[i]
        f = fixed[i]

        k = random.choice(rotate_choices)
        if k != 0:
            m = torch.rot90(m, k=k, dims=(1, 2))
            f = torch.rot90(f, k=k, dims=(1, 2))

        if random.random() < hflip_prob:
            m = torch.flip(m, dims=(2,))
            f = torch.flip(f, dims=(2,))

        new_moving.append(resize_to_square(m, model_square_size))
        new_fixed.append(resize_to_square(f, model_square_size))

    return (
        torch.stack(new_moving, dim=0),
        torch.stack(new_fixed, dim=0),
    )


def resize_flow_torch(flow, size_hw):
    src_h, src_w = flow.shape[-2:]
    dst_h, dst_w = size_hw
    if (src_h, src_w) == (dst_h, dst_w):
        return flow

    scale_x = dst_w / float(src_w)
    scale_y = dst_h / float(src_h)
    resized = F.interpolate(flow, size=size_hw, mode="bilinear", align_corners=True)
    scale = torch.tensor([scale_x, scale_y], dtype=resized.dtype, device=resized.device)
    return resized * scale.view(1, 2, 1, 1)


def maybe_init_wandb(args):
    if not args.use_wandb:
        return None
    import wandb

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )
    return wandb


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def make_training_preview(moving, fixed, warped, flow):
    moving_np = moving[0, 0].detach().cpu().numpy()
    fixed_np = fixed[0, 0].detach().cpu().numpy()
    warped_np = warped[0, 0].detach().cpu().numpy()
    flow_np = flow[0].detach().cpu().numpy()

    def gray_rgb(image):
        image_u8 = np.uint8(np.clip(image, 0.0, 1.0) * 255.0)
        return cv2.cvtColor(image_u8, cv2.COLOR_GRAY2RGB)

    mag = np.sqrt(flow_np[0] ** 2 + flow_np[1] ** 2)
    mag_norm = mag / (float(mag.max()) + 1e-6)
    mag_u8 = np.uint8(np.clip(mag_norm, 0.0, 1.0) * 255.0)
    mag_rgb = cv2.cvtColor(cv2.applyColorMap(mag_u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)

    return np.concatenate(
        [gray_rgb(moving_np), gray_rgb(fixed_np), gray_rgb(warped_np), mag_rgb],
        axis=1,
    )


def expand_stage_weights(weights, num_stages, name):
    if len(weights) == 1:
        return weights * num_stages
    if len(weights) != num_stages:
        raise ValueError(f"{name} must contain either 1 value or {num_stages} values; got {len(weights)}.")
    return weights


def staged_registration_loss(model, moving, fixed, flow_pyramid, args, similarity_weights, bending_weights):
    total = torch.zeros((), dtype=moving.dtype, device=moving.device)
    stage_metrics = []

    for stage_idx, flow_stage in enumerate(flow_pyramid):
        flow_full = resize_flow_torch(flow_stage, fixed.shape[-2:])
        warped_stage = model.transformer(moving, flow_full)
        sim_loss = similarity_loss(fixed, warped_stage, loss_type=args.similarity)
        bend_loss = bending_energy_loss(flow_stage)
        stage_loss = (
            sim_loss * args.model_similarity_weight * similarity_weights[stage_idx]
            + bend_loss * args.bending_weight * bending_weights[stage_idx]
        )
        total = total + stage_loss
        stage_metrics.append(
            {
                "similarity": sim_loss,
                "bending": bend_loss,
                "loss": stage_loss,
                "flow_magnitude_mean": torch.linalg.vector_norm(flow_full, dim=1).mean(),
                "shape": tuple(flow_stage.shape[-2:]),
            }
        )

    return total, stage_metrics


def save_checkpoint(path, model, optimizer, epoch, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    torch.save(
        {
            "epoch": epoch,
            "architecture": "shared_feature_tactmorph_2d_v2_flow_pyramid",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train shared-feature TactMorph registration.")
    parser.add_argument("--image-dir", type=Path, default=Path("../captured_frames"))
    parser.add_argument("--fixed-image-name", type=str, default="frame_000.jpg")
    parser.add_argument("--resize-to", type=parse_size, default=None)
    parser.add_argument("--model-preprocess", choices=("none", "area", "maxpool"), default="maxpool")
    parser.add_argument("--model-input-size", type=parse_size, default=(64, 64))
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--similarity", choices=("MSE", "NCC", "mi"), default="NCC")
    parser.add_argument("--model-similarity-weight", type=float, default=1.0)
    parser.add_argument("--bending-weight", type=float, default=1.0)
    parser.add_argument("--stage-similarity-weights", type=parse_float_list, default=[0.125, 0.25, 0.5, 1.0])
    parser.add_argument("--stage-bending-weights", type=parse_float_list, default=[0.125, 0.25, 0.5, 1.0])
    parser.add_argument("--hflip-prob", type=float, default=0.65)
    parser.add_argument("--rotate-choices", type=int, nargs="*", default=[0, 1, 3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("tactmorph_shared_features/ckpt"))
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="tactmorph_shared_features")
    parser.add_argument("--wandb-run-name", type=str, default="shared_feature_tactmorph")
    parser.add_argument("--wandb-log-images-every", type=int, default=10)
    parser.add_argument("--wandb-log-checkpoints", action="store_true")
    parser.add_argument("--wandb-watch", choices=("none", "gradients", "all"), default="none")
    parser.add_argument("--wandb-watch-freq", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, tensors = load_images_in_memory(
        args.image_dir,
        resize_to=args.resize_to,
        preprocess_mode=args.model_preprocess,
        model_input_size=args.model_input_size,
    )

    fixed_tensor = None
    if args.fixed_image_name:
        try:
            fixed_index = names.index(args.fixed_image_name)
        except ValueError as exc:
            raise ValueError(f"Fixed image '{args.fixed_image_name}' not found in {args.image_dir}") from exc
        fixed_tensor = tensors[fixed_index]

    model = SharedFeatureTactMorph2D(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    total_params, trainable_params = count_parameters(model)

    start_epoch = 1
    if args.resume is not None:
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint.get("architecture") != "shared_feature_tactmorph_2d_v2_flow_pyramid":
            raise RuntimeError(f"Unsupported checkpoint architecture: {checkpoint.get('architecture')}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    wandb = maybe_init_wandb(args)
    if wandb is not None:
        wandb.define_metric("train/global_step")
        wandb.define_metric("train/*", step_metric="train/global_step")
        wandb.define_metric("epoch")
        wandb.define_metric("epoch/*", step_metric="epoch")
        wandb.config.update(
            {
                "parameters/total": total_params,
                "parameters/trainable": trainable_params,
            },
            allow_val_change=True,
        )
        if args.wandb_watch != "none":
            wandb.watch(model, log=args.wandb_watch, log_freq=args.wandb_watch_freq)

    model.train()
    num_samples = len(tensors)
    global_step = 0

    print(f"Using device: {device}")
    print(f"Loaded {num_samples} images from {args.image_dir}")
    print(f"Model input: {args.model_input_size}, base_channels: {args.base_channels}")
    print(f"Parameters: total={total_params:,}, trainable={trainable_params:,}")

    with torch.no_grad():
        dummy = torch.zeros(1, 1, args.model_input_size[1], args.model_input_size[0], device=device)
        _, _, dummy_pyramid = model(dummy, dummy, return_pyramid=True)
    stage_similarity_weights = expand_stage_weights(args.stage_similarity_weights, len(dummy_pyramid), "stage similarity weights")
    stage_bending_weights = expand_stage_weights(args.stage_bending_weights, len(dummy_pyramid), "stage bending weights")
    print(f"Flow pyramid sizes: {[tuple(flow.shape[-2:]) for flow in dummy_pyramid]}")
    print(f"Stage similarity weights: {stage_similarity_weights}")
    print(f"Stage bending weights: {stage_bending_weights}")

    for epoch in range(start_epoch, args.epochs + 1):
        perm = list(range(num_samples))
        random.shuffle(perm)
        if args.drop_last:
            perm = perm[: (num_samples // args.batch_size) * args.batch_size]

        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, len(perm), args.batch_size):
            batch_indices = perm[start: start + args.batch_size]
            if len(batch_indices) < args.batch_size and args.drop_last:
                continue

            moving, fixed = build_batch(tensors, batch_indices, fixed_tensor=fixed_tensor)
            moving = moving.to(device)
            fixed = fixed.to(device)

            moving, fixed = augment_pairs(
                moving,
                fixed,
                hflip_prob=args.hflip_prob,
                rotate_choices=args.rotate_choices,
            )

            warped, flow, flow_pyramid = model(moving, fixed, return_pyramid=True)
            loss, stage_metrics = staged_registration_loss(
                model,
                moving,
                fixed,
                flow_pyramid,
                args,
                stage_similarity_weights,
                stage_bending_weights,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            metrics = {
                "train/loss": loss.item(),
                "train/final_similarity_loss": stage_metrics[-1]["similarity"].item(),
                "train/final_bending_loss": stage_metrics[-1]["bending"].item(),
                "train/flow_magnitude_mean": torch.linalg.vector_norm(flow, dim=1).mean().item(),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/global_step": global_step,
                "epoch": epoch,
            }
            for stage_idx, stage in enumerate(stage_metrics):
                metrics[f"train/stage_{stage_idx}_similarity_loss"] = stage["similarity"].item()
                metrics[f"train/stage_{stage_idx}_bending_loss"] = stage["bending"].item()
                metrics[f"train/stage_{stage_idx}_weighted_loss"] = stage["loss"].item()
                metrics[f"train/stage_{stage_idx}_flow_magnitude_mean"] = stage["flow_magnitude_mean"].item()
            if wandb is not None:
                wandb.log(metrics, step=global_step)
                if (
                    args.wandb_log_images_every > 0
                    and epoch % args.wandb_log_images_every == 0
                    and num_batches == 1
                ):
                    wandb.log(
                        {
                            "train/preview_moving_fixed_warped_flow": wandb.Image(
                                make_training_preview(moving, fixed, warped, flow),
                                caption="moving | fixed | warped moving | flow magnitude",
                            ),
                            "train/global_step": global_step,
                            "epoch": epoch,
                        },
                        step=global_step,
                    )
            global_step += 1

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch}/{args.epochs} - loss={avg_loss:.6f}")
        if wandb is not None:
            wandb.log(
                {
                    "epoch/loss": avg_loss,
                    "train/epoch_loss": avg_loss,
                    "train/global_step": global_step,
                    "epoch": epoch,
                },
                step=global_step,
            )

        if args.save_every > 0 and epoch % args.save_every == 0:
            checkpoint_path = args.checkpoint_dir / f"shared_feature_tactmorph2d_{epoch}.pt"
            save_checkpoint(checkpoint_path, model, optimizer, epoch, args)
            print(f"Saved checkpoint to {checkpoint_path}")
            if wandb is not None and args.wandb_log_checkpoints:
                artifact = wandb.Artifact(
                    f"shared_feature_tactmorph2d_epoch_{epoch}",
                    type="model",
                    metadata={"epoch": epoch},
                )
                artifact.add_file(str(checkpoint_path))
                wandb.log_artifact(artifact)

    final_path = args.checkpoint_dir / "shared_feature_tactmorph2d_final.pt"
    save_checkpoint(final_path, model, optimizer, args.epochs, args)
    config_path = args.checkpoint_dir / "last_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
        f.write("\n")
    print(f"Saved final checkpoint to {final_path}")

    if wandb is not None:
        if args.wandb_log_checkpoints:
            artifact = wandb.Artifact(
                "shared_feature_tactmorph2d_final",
                type="model",
                metadata={"epoch": args.epochs},
            )
            artifact.add_file(str(final_path))
            artifact.add_file(str(config_path))
            wandb.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
