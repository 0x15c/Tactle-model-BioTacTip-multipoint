import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from force_prediction.PointNetRegressor import PointNetRegressor  # noqa: E402


def load_dataset(path: Path):
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    features = data["features"].float()
    forces = data["forces"].float()
    return data, TensorDataset(features, forces)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            batch = x.shape[0]
            total_loss += loss.item() * batch
            total_count += batch
            preds.append(pred.cpu())
            targets.append(y.cpu())

    avg_loss = total_loss / max(total_count, 1)
    return avg_loss, torch.cat(preds, dim=0), torch.cat(targets, dim=0)


def plot_loss_curves(train_losses, val_losses, output_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("PointNet force regression loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=600)
    plt.close(fig)


def plot_validation_scatter(preds: torch.Tensor, targets: torch.Tensor, output_path: Path):
    names = ("Fx", "Fy", "Fz")
    pred_np = preds.numpy()
    target_np = targets.numpy()
    component_mae = abs(pred_np - target_np).mean(axis=0)
    overall_mae = float(component_mae.mean())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, (ax, name) in enumerate(zip(axes, names)):
        ax.scatter(target_np[:, i], pred_np[:, i], s=12, alpha=0.7)
        lo = min(float(target_np[:, i].min()), float(pred_np[:, i].min()))
        hi = max(float(target_np[:, i].max()), float(pred_np[:, i].max()))
        if lo == hi:
            lo -= 1.0
            hi += 1.0
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlabel(f"Ground truth {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(f"{name}  MAE={component_mae[i]:.6f}")
        ax.text(
            0.03,
            0.97,
            f"MAE: {component_mae[i]:.6f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Validation force prediction scatter, overall MAE={overall_mae:.6f}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train PointNet force regressor from sparse TactMorph flow samples.")
    parser.add_argument("--dataset", type=Path, default=Path("force_prediction/calibration_data_washed/pointnet_force_dataset.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("force_prediction/outputs/pointnet_force"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    metadata, dataset = load_dataset(args.dataset)
    total = len(dataset)
    val_size = max(1, int(round(total * args.val_ratio)))
    train_size = total - val_size
    if train_size <= 0:
        raise ValueError("Dataset is too small for the requested validation split.")

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = int(metadata["features"].shape[-1])
    model = PointNetRegressor(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch = x.shape[0]
            total_loss += loss.item() * batch
            total_count += batch

        train_loss = total_loss / max(total_count, 1)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(f"Epoch {epoch:04d}/{args.epochs} train={train_loss:.6f} val={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "pointnet_force_regressor.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "dataset": str(args.dataset),
            "feature_layout": metadata.get("feature_layout", ["x", "y", "u", "v"]),
            "xy_normalized": metadata.get("xy_normalized", True),
            "max_points": metadata.get("max_points"),
            "best_val_loss": best_val,
        },
        model_path,
    )

    val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
    plot_loss_curves(train_losses, val_losses, args.output_dir / "training_curve.png")
    plot_validation_scatter(val_preds, val_targets, args.output_dir / "validation_scatter.png")

    metrics = {
        "dataset": str(args.dataset),
        "num_samples": total,
        "train_samples": train_size,
        "val_samples": val_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "best_val_loss": best_val,
        "final_val_loss": val_loss,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    print(f"Saved model to {model_path}")
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
