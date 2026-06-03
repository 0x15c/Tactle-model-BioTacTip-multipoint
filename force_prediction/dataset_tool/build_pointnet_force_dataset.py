import argparse
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tactile_prediction_biotactip import (  # noqa: E402
    get_device,
    get_tactmorph_model,
    gray_to_tensor,
    infer_warped_and_flow,
    sample_flow_at_points,
)
from tactile_prediction_sensor2_centernet import (  # noqa: E402
    Sensor2CenterNetConfig,
    blur_flow_field,
    extract_probability_centroids,
    get_centernet_model,
    infer_marker_probability,
    load_config as load_sensor2_config,
)
from tactmorph.preprocessing import upsample_flow_to_shape  # noqa: E402


def load_records(labels_path: Path) -> list[dict]:
    records = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_bgr(path: Path) -> np.ndarray:
    image = cv.imread(str(path), cv.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def pad_or_trim_points(points: np.ndarray, max_points: int) -> tuple[np.ndarray, int]:
    points = points.astype(np.float32)
    valid_count = min(points.shape[0], max_points)
    out = np.zeros((max_points, points.shape[1]), dtype=np.float32)
    if valid_count:
        out[:valid_count] = points[:valid_count]
    return out, valid_count


@torch.no_grad()
def build_features_for_record(
    record: dict,
    dataset_dir: Path,
    cfg: Sensor2CenterNetConfig,
    centernet,
    tactmorph,
    device: torch.device,
    init_cache: dict,
    max_points: int,
    normalize_xy: bool,
) -> tuple[np.ndarray, int]:
    raw_path = dataset_dir / record["raw_image"]
    init_path = dataset_dir / record["init_image"]
    raw_bgr = read_bgr(raw_path)

    init_key = str(init_path)
    if init_key not in init_cache:
        init_bgr = read_bgr(init_path)
        init_prob = infer_marker_probability(
            centernet,
            init_bgr,
            cfg.centernet_input_size,
            init_bgr.shape[:2],
            device,
        )
        centroids, _, _ = extract_probability_centroids(
            init_prob,
            cfg.centroid_threshold,
            cfg.centroid_min_area,
            cfg.centroid_max_area,
        )
        init_tensor = gray_to_tensor(
            init_prob,
            device,
            preprocess_mode=cfg.model_preprocess,
            model_input_size=cfg.model_input_size,
        )
        init_cache[init_key] = (init_prob, centroids, init_tensor)

    init_prob, rest_centroids, rest_tensor = init_cache[init_key]
    moving_prob = infer_marker_probability(
        centernet,
        raw_bgr,
        cfg.centernet_input_size,
        raw_bgr.shape[:2],
        device,
    )
    moving_tensor = gray_to_tensor(
        moving_prob,
        device,
        preprocess_mode=cfg.model_preprocess,
        model_input_size=cfg.model_input_size,
    )

    _, flow_model = infer_warped_and_flow(tactmorph, moving_tensor, rest_tensor)
    flow = upsample_flow_to_shape(flow_model, moving_prob.shape)
    flow = blur_flow_field(flow, cfg.flow_blur_ksize)
    displacement = sample_flow_at_points(flow, rest_centroids, cfg.flow_sample_radius)

    if rest_centroids.size == 0:
        features = np.zeros((0, 4), dtype=np.float32)
    else:
        xy = rest_centroids.astype(np.float32).copy()
        if normalize_xy:
            height, width = init_prob.shape
            xy[:, 0] = xy[:, 0] / max(width - 1, 1)
            xy[:, 1] = xy[:, 1] / max(height - 1, 1)
        features = np.concatenate((xy, displacement.astype(np.float32)), axis=1)

    return pad_or_trim_points(features, max_points)


def main():
    parser = argparse.ArgumentParser(description="Build sparse PointNet force dataset from washed Sensor2 calibration data.")
    parser.add_argument("--washed-dir", type=Path, default=Path("force_prediction/calibration_data_washed"))
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("force_prediction/calibration_data_washed/pointnet_force_dataset.pt"))
    parser.add_argument("--sensor2-config", type=Path, default=Path("configs/tactile_prediction_sensor2_centernet.json"))
    parser.add_argument("--centernet-weights", type=str, default=None)
    parser.add_argument("--tactmorph-weights", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-points", type=int, default=128)
    parser.add_argument("--no-normalize-xy", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    labels_path = args.labels or (args.washed_dir / "labels.jsonl")
    records = load_records(labels_path)
    if args.limit is not None:
        records = records[: args.limit]

    cfg = load_sensor2_config(args.sensor2_config) if args.sensor2_config.exists() else Sensor2CenterNetConfig()
    if args.centernet_weights is not None:
        cfg.centernet_weights_path = args.centernet_weights
    if args.tactmorph_weights is not None:
        cfg.tactmorph_weights_path = args.tactmorph_weights
    if args.device is not None:
        cfg.device = args.device

    device = get_device(cfg.device)
    centernet = get_centernet_model(cfg.centernet_weights_path, device)
    tactmorph = get_tactmorph_model(cfg.tactmorph_weights_path, device)

    features = []
    forces = []
    valid_counts = []
    kept_records = []
    init_cache = {}

    for index, record in enumerate(records, start=1):
        try:
            point_features, valid_count = build_features_for_record(
                record,
                args.washed_dir,
                cfg,
                centernet,
                tactmorph,
                device,
                init_cache,
                args.max_points,
                normalize_xy=not args.no_normalize_xy,
            )
            force = record["force"]
            features.append(point_features)
            forces.append([force["Fx"], force["Fy"], force["Fz"]])
            valid_counts.append(valid_count)
            kept_records.append(record)
        except Exception as exc:
            print(f"Skipping {record.get('sample_id', index)}: {exc}")

        if index % 50 == 0:
            print(f"Processed {index}/{len(records)} records")

    if not features:
        raise RuntimeError("No valid samples were built.")

    output = {
        "features": torch.tensor(np.stack(features), dtype=torch.float32),
        "forces": torch.tensor(np.asarray(forces), dtype=torch.float32),
        "valid_counts": torch.tensor(valid_counts, dtype=torch.long),
        "records": kept_records,
        "max_points": args.max_points,
        "feature_layout": ["x", "y", "u", "v"],
        "xy_normalized": not args.no_normalize_xy,
        "sensor2_config": {
            "centernet_weights_path": cfg.centernet_weights_path,
            "tactmorph_weights_path": cfg.tactmorph_weights_path,
            "centernet_input_size": list(cfg.centernet_input_size),
            "model_preprocess": cfg.model_preprocess,
            "model_input_size": list(cfg.model_input_size),
            "centroid_threshold": cfg.centroid_threshold,
            "centroid_min_area": cfg.centroid_min_area,
            "centroid_max_area": cfg.centroid_max_area,
            "flow_blur_ksize": cfg.flow_blur_ksize,
            "flow_sample_radius": cfg.flow_sample_radius,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"Saved {len(features)} samples to {args.output}")
    print(f"Feature tensor: {tuple(output['features'].shape)}")
    print(f"Force tensor: {tuple(output['forces'].shape)}")


if __name__ == "__main__":
    main()
