import argparse
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional, Union

import cv2 as cv
import numpy as np
import torch

from centernet.centernet_model import CenterNetModel
from tactile_prediction_biotactip import (
    draw_displacement_vectors,
    get_device,
    get_voxelmorph_model,
    gray_to_tensor,
    infer_warped_and_flow,
    parse_capture_source,
    render_flow_heatmap,
    render_flow_phase,
    render_full_res_warp_comparison,
    render_registration_comparison,
    sample_flow_at_points,
    tensor_to_gray_u8,
)
from voxelmorph.preprocessing import upsample_flow_to_shape


@dataclass
class Sensor2CenterNetConfig:
    input_source: Union[int, str] = 0
    output_original_path: str = "videos/sensor2_centernet_probability.mp4"
    output_heatmap_path: str = "videos/sensor2_centernet_heatmaps.mp4"
    video_codec: str = "XVID"

    centernet_weights_path: str = "centernet/checkpoints/latest.pth"
    centernet_input_size: tuple[int, int] = (360, 360)  # width, height
    centernet_prob_threshold: float = 0.2
    centroid_threshold: float = 0.2
    centroid_min_area: int = 2
    centroid_max_area: int = 5000

    voxelmorph_weights_path: str = "voxelmorph/ckpt/new_sensor_voxelmorph2d_260.pt"
    device: Optional[str] = None
    model_preprocess: str = "maxpool"
    model_input_size: tuple[int, int] = (64,64)

    frame_skip_init: int = 10
    flow_blur_ksize: int = 3
    flow_sample_radius: int = 2
    vector_visual_scale: float = 1.0
    max_disp_viz_mag: float = 5.0
    overlay_alpha: float = 0.65
    heatmap_alpha: float = 0.35


def load_config(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_fields = {field.name for field in fields(Sensor2CenterNetConfig)}
    unknown = sorted(set(data) - valid_fields)
    if unknown:
        raise ValueError(f"Unknown config field(s) in {path}: {', '.join(unknown)}")

    cfg = Sensor2CenterNetConfig()
    for key, value in data.items():
        setattr(cfg, key, value)
    cfg.input_source = parse_capture_source(str(cfg.input_source)) if isinstance(cfg.input_source, str) else cfg.input_source
    cfg.centernet_input_size = tuple(cfg.centernet_input_size)
    cfg.model_input_size = tuple(cfg.model_input_size)
    return cfg


def save_config(path, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
        f.write("\n")


def get_centernet_model(weights_path, device):
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, dict) and "head.2.bias" in state:
        num_classes = state["head.2.bias"].shape[0]
        model = CenterNetModel(num_classes=num_classes)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
    elif isinstance(state, torch.nn.Module):
        model = state
    else:
        raise RuntimeError(f"Unsupported CenterNet checkpoint format: {weights_path}")

    model.to(device)
    model.eval()
    return model


def preprocess_frame_for_centernet(frame_bgr, input_size):
    frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
    if input_size is not None:
        frame_rgb = cv.resize(frame_rgb, input_size, interpolation=cv.INTER_LINEAR)

    image = frame_rgb.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image)[None]


@torch.no_grad()
def infer_marker_probability(model, frame_bgr, input_size, out_shape, device):
    x = preprocess_frame_for_centernet(frame_bgr, input_size).to(device)
    logits = model(x)
    prob = torch.sigmoid(logits)[0]
    heat = prob.max(dim=0).values if prob.shape[0] > 1 else prob[0]
    heat = heat.detach().cpu().numpy().astype(np.float32)

    out_h, out_w = out_shape
    heat = cv.resize(heat, (out_w, out_h), interpolation=cv.INTER_LINEAR)
    return np.clip(heat, 0.0, 1.0)


def probability_to_u8(prob):
    return np.uint8(np.clip(prob, 0.0, 1.0) * 255.0)


def render_probability_heatmap(prob, threshold):
    heat = prob.copy()
    if threshold > 0:
        heat[heat < threshold] = 0.0
    color = cv.applyColorMap(probability_to_u8(heat), cv.COLORMAP_JET)
    cv.putText(color, "CenterNet probability", (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv.LINE_AA)
    return color


def extract_probability_centroids(prob, threshold, min_area, max_area):
    mask = np.uint8(prob >= threshold) * 255
    count, _, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)

    points = []
    areas = []
    for label in range(1, count):
        area = int(stats[label, cv.CC_STAT_AREA])
        if min_area <= area <= max_area:
            points.append(centroids[label])
            areas.append(area)

    if not points:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32), mask

    points = np.asarray(points, dtype=np.float32)
    areas = np.asarray(areas, dtype=np.float32)
    order = np.lexsort((points[:, 0], points[:, 1]))
    return points[order], areas[order], mask


def render_centroid_overlay(frame, prob_heatmap, centroids, rest_centroids=None):
    overlay = cv.addWeighted(frame, 0.65, prob_heatmap, 0.35, 0.0)
    for x, y in centroids:
        cv.circle(overlay, (int(round(x)), int(round(y))), 4, (0, 0, 255), 1, cv.LINE_AA)
    if rest_centroids is not None:
        for x, y in rest_centroids:
            cv.circle(overlay, (int(round(x)), int(round(y))), 4, (0, 255, 0), 1, cv.LINE_AA)
    cv.putText(overlay, f"current centroids: {len(centroids)}", (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv.LINE_AA)
    if rest_centroids is not None:
        cv.putText(overlay, f"rest centroids: {len(rest_centroids)}", (10, 48), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)
    return overlay


def blur_flow_field(flow, ksize):
    if ksize <= 1:
        return flow
    if ksize % 2 == 0:
        ksize += 1
    u = cv.blur(flow[0], (ksize, ksize), borderType=cv.BORDER_REFLECT)
    v = cv.blur(flow[1], (ksize, ksize), borderType=cv.BORDER_REFLECT)
    return np.stack([u, v], axis=0)


def main(cfg):
    device = get_device(cfg.device)
    centernet = get_centernet_model(cfg.centernet_weights_path, device)
    voxelmorph = get_voxelmorph_model(cfg.voxelmorph_weights_path, device)
    print(f"Using device: {device}")
    print(f"CenterNet weights: {cfg.centernet_weights_path}")
    print(f"VoxelMorph weights: {cfg.voxelmorph_weights_path}")

    cap = cv.VideoCapture(cfg.input_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open capture source {cfg.input_source}")

    capture_fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    video_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*cfg.video_codec)

    Path(cfg.output_original_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_heatmap_path).parent.mkdir(parents=True, exist_ok=True)
    out_original = cv.VideoWriter(cfg.output_original_path, fourcc, capture_fps, (video_w, video_h), False)
    heatmap_output = cv.VideoWriter(cfg.output_heatmap_path, fourcc, capture_fps, (video_w * 3, video_h), True)

    frame_count = 0
    start_time = time.time()
    prev_time = time.time()
    rest_prob = None
    rest_tensor = None
    rest_model_gray = None
    rest_centroids = np.zeros((0, 2), dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break

        prob = infer_marker_probability(
            centernet,
            frame,
            cfg.centernet_input_size,
            (frame.shape[0], frame.shape[1]),
            device,
        )
        prob_u8 = probability_to_u8(prob)
        out_original.write(prob_u8)

        prob_heatmap = render_probability_heatmap(prob, cfg.centernet_prob_threshold)
        centroids, areas, mask = extract_probability_centroids(
            prob,
            cfg.centroid_threshold,
            cfg.centroid_min_area,
            cfg.centroid_max_area,
        )

        if frame_count == cfg.frame_skip_init:
            rest_prob = prob.copy()
            rest_tensor = gray_to_tensor(rest_prob, device, cfg.model_preprocess, cfg.model_input_size)
            rest_model_gray = tensor_to_gray_u8(rest_tensor)
            rest_centroids = centroids.copy()
            print(f"Captured rest frame with {len(rest_centroids)} CenterNet centroids.")

        if rest_tensor is None:
            displacement_heatmap = cv.cvtColor(prob_u8, cv.COLOR_GRAY2BGR)
            cv.putText(
                displacement_heatmap,
                f"Capturing rest frame in {max(0, cfg.frame_skip_init - frame_count)} frames",
                (10, 24),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1,
                cv.LINE_AA,
            )
        else:
            moving_tensor = gray_to_tensor(prob, device, cfg.model_preprocess, cfg.model_input_size)
            moving_model_gray = tensor_to_gray_u8(moving_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()
            warped_model, flow_model = infer_warped_and_flow(voxelmorph, moving_tensor, rest_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()

            flow = upsample_flow_to_shape(flow_model, prob.shape)
            flow = blur_flow_field(flow, cfg.flow_blur_ksize)

            displacement = sample_flow_at_points(flow, rest_centroids, cfg.flow_sample_radius)
            displacement_viz = displacement * cfg.vector_visual_scale
            mean_disp = displacement.mean(axis=0) if displacement.size else np.zeros(2, dtype=np.float32)

            displacement_heatmap = render_flow_heatmap(flow, cfg.max_disp_viz_mag)
            displacement_heatmap = draw_displacement_vectors(displacement_heatmap, rest_centroids, displacement_viz)

            overlay = render_centroid_overlay(frame, prob_heatmap, centroids, rest_centroids)
            overlay = draw_displacement_vectors(overlay, rest_centroids, displacement_viz)
            cv.imshow("Sensor2 CenterNet Displacement Field", overlay)
            cv.imshow("Sensor2 CenterNet Flow Direction Phase", render_flow_phase(flow, cfg.max_disp_viz_mag))
            cv.imshow("Sensor2 CenterNet Inverse Warp", render_full_res_warp_comparison(probability_to_u8(rest_prob), prob_u8, flow))
            cv.imshow(
                "Sensor2 CenterNet Registration Comparison",
                render_registration_comparison(
                    rest_model_gray,
                    moving_model_gray,
                    warped_model,
                    display_shape=prob.shape,
                ),
            )

            cv.putText(
                displacement_heatmap,
                f"mean dx={mean_disp[0]:.2f}, dy={mean_disp[1]:.2f}",
                (10, 60),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv.LINE_AA,
            )

        now = time.time()
        fps = 1.0 / max(1e-6, now - prev_time)
        prev_time = now
        elapsed = now - start_time
        cv.putText(displacement_heatmap, "Displacement", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
        cv.putText(displacement_heatmap, f"FPS={fps:.0f}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
        cv.putText(displacement_heatmap, f"T={elapsed:.2f}", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

        centroid_overlay = render_centroid_overlay(frame, prob_heatmap, centroids, rest_centroids if rest_tensor is not None else None)
        combined = np.concatenate((frame, displacement_heatmap, prob_heatmap), axis=1)
        cv.imshow("sensor2 centernet raw/probability", centroid_overlay)
        cv.imshow("sensor2 centernet mask", mask)
        cv.imshow("sensor2 centernet heatmaps", combined)
        heatmap_output.write(combined)

        key = cv.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        frame_count += 1

    cap.release()
    out_original.release()
    heatmap_output.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sensor2 CenterNet + VoxelMorph displacement visualization")
    parser.add_argument("--config", type=str, default=None, help="JSON config")
    parser.add_argument("--save-config", type=str, default=None, help="write resolved config and exit")
    parser.add_argument("--input", type=str, default=None, help="camera index or video path")
    parser.add_argument("--centernet-weights", type=str, default=None, help="CenterNet checkpoint path")
    parser.add_argument("--voxelmorph-weights", type=str, default=None, help="VoxelMorph checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cpu or cuda")
    parser.add_argument("--model-preprocess", type=str, default=None, choices=("none", "area", "maxpool"))
    parser.add_argument("--model-input-size", type=int, default=None)
    parser.add_argument("--vector-scale", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else Sensor2CenterNetConfig()
    if args.input is not None:
        cfg.input_source = parse_capture_source(args.input)
    if args.centernet_weights is not None:
        cfg.centernet_weights_path = args.centernet_weights
    if args.voxelmorph_weights is not None:
        cfg.voxelmorph_weights_path = args.voxelmorph_weights
    if args.device is not None:
        cfg.device = args.device
    if args.model_preprocess is not None:
        cfg.model_preprocess = args.model_preprocess
    if args.model_input_size is not None:
        cfg.model_input_size = (args.model_input_size, args.model_input_size)
    if args.vector_scale is not None:
        cfg.vector_visual_scale = args.vector_scale

    if args.save_config is not None:
        save_config(args.save_config, cfg)
        print(f"Saved config to: {args.save_config}")
        raise SystemExit(0)

    main(cfg)
