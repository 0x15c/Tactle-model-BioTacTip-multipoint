import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

if "--interactive" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
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
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    cb = plt.colorbar(scatter, ax=ax)
    cb.set_label(colorbar_label)


def plot_pca_embedding(embedding, forces, preds, explained, output_path: Path):
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

    fig.suptitle(
        "PointNet latent PCA "
        f"(explained variance: PC1={explained[0]:.3f}, PC2={explained[1]:.3f})"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def add_embedding_3d_panel(ax, embedding, color, title, colorbar_label, cmap="viridis"):
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        embedding[:, 2],
        c=color,
        s=14,
        alpha=0.8,
        cmap=cmap,
        depthshade=False,
    )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    cb = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.08)
    cb.set_label(colorbar_label)


def plot_pca_embedding_3d(embedding, forces, preds, explained, output_path: Path, interactive: bool = False):
    force_mag = np.linalg.norm(forces, axis=1)
    pred_error = np.linalg.norm(preds - forces, axis=1)

    panels = [
        ("Ground truth Fx", forces[:, 0], "Fx", "coolwarm"),
        ("Ground truth Fy", forces[:, 1], "Fy", "coolwarm"),
        ("Ground truth Fz", forces[:, 2], "Fz", "coolwarm"),
        ("Ground truth force magnitude", force_mag, "|F|", "viridis"),
        ("Vector prediction error", pred_error, "||pred-true||", "magma"),
    ]

    fig = plt.figure(figsize=(18, 9))
    for index, (title, color, label, cmap) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 3, index, projection="3d")
        add_embedding_3d_panel(ax, embedding, color, title, label, cmap=cmap)

    fig.suptitle(
        "PointNet latent 3D PCA "
        f"(explained variance: PC1={explained[0]:.3f}, "
        f"PC2={explained[1]:.3f}, PC3={explained[2]:.3f})"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    if interactive:
        plt.show()
    plt.close(fig)


def write_embedding_csv(path: Path, embedding, forces, preds):
    fieldnames = [
        "pc1",
        "pc2",
        "pc3",
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
                    "pc1": float(embedding[i, 0]),
                    "pc2": float(embedding[i, 1]),
                    "pc3": float(embedding[i, 2]) if embedding.shape[1] > 2 else 0.0,
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
    parser = argparse.ArgumentParser(description="Visualize PointNet latent force representation using PCA.")
    parser.add_argument("--output-dir", type=Path, default=Path("force_prediction/outputs/pointnet_force"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--interactive", action="store_true", help="open an interactive 3D PCA window after saving plots")
    args = parser.parse_args()

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
    pca = PCA(n_components=3)
    embedding = pca.fit_transform(latents)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / "latent_pca.png"
    plot_3d_path = args.output_dir / "latent_pca_3d.png"
    csv_path = args.output_dir / "latent_pca.csv"
    plot_pca_embedding(embedding, forces, preds, pca.explained_variance_ratio_, plot_path)
    plot_pca_embedding_3d(
        embedding,
        forces,
        preds,
        pca.explained_variance_ratio_,
        plot_3d_path,
        interactive=args.interactive,
    )
    write_embedding_csv(csv_path, embedding, forces, preds)

    print(f"Saved PCA plot to {plot_path}")
    print(f"Saved 3D PCA plot to {plot_3d_path}")
    print(f"Saved PCA coordinates to {csv_path}")
    print(
        "Explained variance ratio: "
        f"PC1={pca.explained_variance_ratio_[0]:.4f}, "
        f"PC2={pca.explained_variance_ratio_[1]:.4f}, "
        f"PC3={pca.explained_variance_ratio_[2]:.4f}"
    )


if __name__ == "__main__":
    main()
