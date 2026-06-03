import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from force_prediction.PointNetRegressor import PointNetRegressor  # noqa: E402


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_default_dataset_from_metrics(output_dir: Path) -> Path | None:
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    dataset = metrics.get("dataset")
    return Path(dataset) if dataset else None


@torch.no_grad()
def extract_latent_and_predictions(model, features: torch.Tensor, batch_size: int, device: torch.device):
    loader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False)
    latents = []
    preds = []
    model.eval()
    for (x,) in loader:
        x = x.to(device)
        point_feat = model.point_encoder(x)
        global_feat = point_feat.max(dim=1)[0]
        pred = model.regressor(global_feat)
        latents.append(global_feat.cpu())
        preds.append(pred.cpu())
    return torch.cat(latents, dim=0).numpy(), torch.cat(preds, dim=0).numpy()


def add_embedding_panel(ax, embedding, color, title, colorbar_label, cmap="viridis"):
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=color, s=14, alpha=0.8, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.25)
    cb = plt.colorbar(scatter, ax=ax)
    cb.set_label(colorbar_label)


def plot_umap_embedding(embedding, forces, preds, output_path: Path, title_suffix: str):
    names = ("Fx", "Fy", "Fz")
    force_mag = np.linalg.norm(forces, axis=1)
    pred_error = np.linalg.norm(preds - forces, axis=1)
    component_abs_error = np.abs(preds - forces)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for i, name in enumerate(names):
        add_embedding_panel(
            axes[0, i],
            embedding,
            forces[:, i],
            f"Ground truth {name}",
            name,
            cmap="coolwarm",
        )

    add_embedding_panel(
        axes[0, 3],
        embedding,
        force_mag,
        "Ground truth force magnitude",
        "|F|",
        cmap="viridis",
    )

    for i, name in enumerate(names):
        add_embedding_panel(
            axes[1, i],
            embedding,
            component_abs_error[:, i],
            f"Absolute error {name}",
            f"|pred-true| {name}",
            cmap="magma",
        )

    add_embedding_panel(
        axes[1, 3],
        embedding,
        pred_error,
        "Vector prediction error",
        "||pred-true||",
        cmap="magma",
    )

    fig.suptitle(f"PointNet latent UMAP {title_suffix}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_embedding_csv(path: Path, embedding, forces, preds):
    fieldnames = [
        "umap1",
        "umap2",
        "Fx",
        "Fy",
        "Fz",
        "pred_Fx",
        "pred_Fy",
        "pred_Fz",
        "force_magnitude",
        "prediction_error",
    ]
    force_mag = np.linalg.norm(forces, axis=1)
    pred_error = np.linalg.norm(preds - forces, axis=1)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(embedding.shape[0]):
            writer.writerow(
                {
                    "umap1": float(embedding[i, 0]),
                    "umap2": float(embedding[i, 1]),
                    "Fx": float(forces[i, 0]),
                    "Fy": float(forces[i, 1]),
                    "Fz": float(forces[i, 2]),
                    "pred_Fx": float(preds[i, 0]),
                    "pred_Fy": float(preds[i, 1]),
                    "pred_Fz": float(preds[i, 2]),
                    "force_magnitude": float(force_mag[i]),
                    "prediction_error": float(pred_error[i]),
                }
            )


def main():
    parser = argparse.ArgumentParser(description="Visualize PointNet latent force representation using UMAP.")
    parser.add_argument("--output-dir", type=Path, default=Path("force_prediction/outputs/pointnet_force"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    try:
        import umap
    except ImportError as exc:
        raise ImportError("UMAP visualization requires `umap-learn`. Install with: pip install umap-learn") from exc

    checkpoint_path = args.checkpoint or (args.output_dir / "pointnet_force_regressor.pt")
    dataset_path = args.dataset or load_default_dataset_from_metrics(args.output_dir)
    if dataset_path is None:
        raise ValueError("Dataset path was not provided and could not be inferred from metrics.json.")

    checkpoint = torch_load(checkpoint_path)
    dataset = torch_load(dataset_path)
    features = dataset["features"].float()
    forces = dataset["forces"].float().numpy()

    input_dim = int(checkpoint.get("input_dim", features.shape[-1]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = PointNetRegressor(input_dim=input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    latents, preds = extract_latent_and_predictions(model, features, args.batch_size, device)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
    )
    embedding = reducer.fit_transform(latents)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / "latent_umap.png"
    csv_path = args.output_dir / "latent_umap.csv"
    title_suffix = f"(n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, metric={args.metric})"
    plot_umap_embedding(embedding, forces, preds, plot_path, title_suffix)
    write_embedding_csv(csv_path, embedding, forces, preds)

    print(f"Saved UMAP plot to {plot_path}")
    print(f"Saved UMAP coordinates to {csv_path}")


if __name__ == "__main__":
    main()
