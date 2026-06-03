import argparse
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import sys
from typing import Union

import cv2 as cv
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tactile_prediction_biotactip import (
    Config as TactMorphConfig,
    blur_flow_field,
    create_circular_mask,
    create_radial_mask,
    crop_and_preprocess_frame,
    draw_displacement_vectors,
    find_rest_marker_centroids,
    get_device,
    get_tactmorph_model,
    gray_panel,
    gray_to_tensor,
    infer_warped_and_flow,
    parse_capture_source,
    render_flow_heatmap,
    render_flow_phase,
    sample_flow_at_points,
)
from tactmorph.preprocessing import upsample_flow_to_shape


@dataclass
class ComparisonConfig(TactMorphConfig):
    output_heatmap_path: str = "videos/tactmorph_vs_farneback_biotactip.mp4"
    magnitude_colormap_min: float = 0.0
    magnitude_colormap_max: float = 40.0

    # Farneback parameters. Defaults mirror the common GelSlim shear baseline.
    farneback_pyr_scale: float = 0.5
    farneback_levels: int = 4
    farneback_winsize: int = 45
    farneback_iterations: int = 3
    farneback_poly_n: int = 5
    farneback_poly_sigma: float = 1.2
    farneback_flags: int = 0


def load_config(path: Union[str, Path]) -> ComparisonConfig:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_fields = {field.name for field in fields(ComparisonConfig)}
    cfg = ComparisonConfig()
    for key, value in data.items():
        if key in valid_fields:
            setattr(cfg, key, value)

    cfg.input_source = parse_capture_source(str(cfg.input_source)) if isinstance(cfg.input_source, str) else cfg.input_source
    cfg.center = tuple(cfg.center)
    cfg.model_input_size = tuple(cfg.model_input_size)
    return cfg


def save_config(path: Union[str, Path], cfg: ComparisonConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
        f.write("\n")


def calculate_farneback_flow(rest_gray: np.ndarray, current_gray: np.ndarray, cfg: ComparisonConfig) -> np.ndarray:
    flow_hw2 = cv.calcOpticalFlowFarneback(
        rest_gray,
        current_gray,
        None,
        cfg.farneback_pyr_scale,
        cfg.farneback_levels,
        cfg.farneback_winsize,
        cfg.farneback_iterations,
        cfg.farneback_poly_n,
        cfg.farneback_poly_sigma,
        cfg.farneback_flags,
    )
    return np.moveaxis(flow_hw2, 2, 0).astype(np.float32)


def render_method_panel(
    label: str,
    gray: np.ndarray,
    flow: np.ndarray,
    rest_centroids: np.ndarray,
    cfg: ComparisonConfig,
) -> np.ndarray:
    displacement = sample_flow_at_points(flow, rest_centroids, cfg.flow_sample_radius)
    displacement_viz = displacement * cfg.vector_visual_scale
    mean_disp = displacement.mean(axis=0) if displacement.size else np.zeros(2, dtype=np.float32)

    overlay = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    overlay = draw_displacement_vectors(overlay, rest_centroids, displacement_viz)
    cv.putText(overlay, label, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
    cv.putText(
        overlay,
        f"mean dx={mean_disp[0]:.2f}, dy={mean_disp[1]:.2f}",
        (10, 45),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv.LINE_AA,
    )
    return overlay


def render_flow_magnitude_panel(label: str, flow: np.ndarray, cfg: ComparisonConfig) -> np.ndarray:
    flow_mag = np.linalg.norm(flow, axis=0)
    mag_min = float(cfg.magnitude_colormap_min)
    mag_max = max(float(cfg.magnitude_colormap_max), mag_min + 1e-6)
    flow_norm = np.clip((flow_mag - mag_min) / (mag_max - mag_min), 0.0, 1.0)
    magnitude_panel = cv.applyColorMap(
        np.uint8(flow_norm * 255),
        cv.COLORMAP_JET,
    )
    cv.putText(magnitude_panel, f"{label} magnitude", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
    cv.putText(
        magnitude_panel,
        f"mean={float(flow_mag.mean()):.2f}, max={float(flow_mag.max()):.2f}",
        (10, 45),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv.LINE_AA,
    )
    cv.putText(
        magnitude_panel,
        f"color range: {mag_min:.1f}-{mag_max:.1f} px",
        (10, 70),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv.LINE_AA,
    )
    return magnitude_panel


def render_grid(
    top_left: np.ndarray,
    bottom_left: np.ndarray,
    top_mid: np.ndarray,
    bottom_mid: np.ndarray,
    top_right: np.ndarray,
    bottom_right: np.ndarray,
) -> np.ndarray:
    top_row = np.concatenate((top_left, top_mid, top_right), axis=1)
    bottom_row = np.concatenate((bottom_left, bottom_mid, bottom_right), axis=1)
    return np.concatenate((top_row, bottom_row), axis=0)


def main(cfg: ComparisonConfig):
    device = get_device(cfg.device)
    print(f"Using device: {device}")
    print(f"TactMorph preprocessing: {cfg.model_preprocess}, input size: {cfg.model_input_size}")
    print(f"TactMorph weights: {cfg.model_weights_path}")
    print(
        "Farneback params: "
        f"pyr_scale={cfg.farneback_pyr_scale}, levels={cfg.farneback_levels}, "
        f"winsize={cfg.farneback_winsize}, iterations={cfg.farneback_iterations}, "
        f"poly_n={cfg.farneback_poly_n}, poly_sigma={cfg.farneback_poly_sigma}, "
        f"flags={cfg.farneback_flags}"
    )

    model = get_tactmorph_model(cfg.model_weights_path, device)
    cap = cv.VideoCapture(cfg.input_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open capture source {cfg.input_source}")

    capture_fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    video_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*cfg.video_codec)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, video_w)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, video_h)
    cap.set(cv.CAP_PROP_EXPOSURE, cfg.exposure)
    cap.set(cv.CAP_PROP_BRIGHTNESS, cfg.brightness)
    cap.set(cv.CAP_PROP_CONTRAST, cfg.contrast)
    cap.set(cv.CAP_PROP_SATURATION, cfg.saturation)
    cap.set(cv.CAP_PROP_HUE, cfg.hue)
    cap.set(cv.CAP_PROP_GAIN, cfg.gain)

    cnt = [int(video_w / 2), int(video_h / 2)]
    cropped_limits = [
        [cnt[0] - cfg.crop_px + cfg.crop_offset_x, cnt[1] - cfg.crop_py + cfg.crop_offset_y],
        [cnt[0] + cfg.crop_px + cfg.crop_offset_x, cnt[1] + cfg.crop_py + cfg.crop_offset_y],
    ]
    cropped_size = (2 * cfg.crop_px, 2 * cfg.crop_py)
    frame_shape = (cropped_size[1], cropped_size[0])

    radial_mask = create_radial_mask(frame_shape, max_value=245)
    circular_mask = create_circular_mask(frame_shape, cfg.center, cfg.radius)

    Path(cfg.output_heatmap_path).parent.mkdir(parents=True, exist_ok=True)
    combined_heatmap_size = (cropped_size[0] * 3, cropped_size[1] * 2)
    heatmap_output = cv.VideoWriter(cfg.output_heatmap_path, fourcc, capture_fps, combined_heatmap_size, True)

    frame_count = 0
    time_0 = time.time()
    prev_time = time.time()
    rest_gray = None
    rest_tensor = None
    rest_centroids = np.zeros((0, 2), dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break

        _, grey_frame_corrected = crop_and_preprocess_frame(
            frame,
            cropped_limits,
            circular_mask,
            radial_mask,
        )

        if frame_count == cfg.frame_skip_init:
            rest_gray = grey_frame_corrected.copy()
            rest_tensor = gray_to_tensor(
                rest_gray,
                device,
                preprocess_mode=cfg.model_preprocess,
                model_input_size=cfg.model_input_size,
            )
            rest_centroids = find_rest_marker_centroids(
                rest_gray,
                cfg.rest_marker_threshold,
                cfg.rest_marker_min_area,
                cfg.rest_marker_max_area,
            )
            print(f"Captured rest frame with {len(rest_centroids)} marker centroids.")
            if len(rest_centroids) == 0:
                print("No rest markers detected; lower --rest-threshold or area limits to show vectors.")

        if rest_gray is None or rest_tensor is None:
            current_panel = gray_panel(grey_frame_corrected, "Current preprocessed")
            rest_panel = gray_panel(grey_frame_corrected, "Initial preprocessed")
            wait_panel = current_panel.copy()
            cv.putText(
                wait_panel,
                f"Capturing rest frame in {max(0, cfg.frame_skip_init - frame_count)} frames",
                (10, 45),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv.LINE_AA,
            )
            combined = render_grid(
                rest_panel,
                current_panel,
                wait_panel,
                wait_panel,
                wait_panel,
                wait_panel,
            )
        else:
            moving_tensor = gray_to_tensor(
                grey_frame_corrected,
                device,
                preprocess_mode=cfg.model_preprocess,
                model_input_size=cfg.model_input_size,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            _, tact_flow_model = infer_warped_and_flow(model, moving_tensor, rest_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()

            tact_flow = upsample_flow_to_shape(tact_flow_model, grey_frame_corrected.shape)
            tact_flow = blur_flow_field(tact_flow, cfg.flow_blur_ksize)

            farneback_flow = calculate_farneback_flow(rest_gray, grey_frame_corrected, cfg)
            farneback_flow = blur_flow_field(farneback_flow, cfg.flow_blur_ksize)

            tact_panel = render_method_panel(
                "TactMorph",
                grey_frame_corrected,
                tact_flow,
                rest_centroids,
                cfg,
            )
            farneback_panel = render_method_panel(
                "Farneback optical flow",
                grey_frame_corrected,
                farneback_flow,
                rest_centroids,
                cfg,
            )
            tact_magnitude_panel = render_flow_magnitude_panel("TactMorph", tact_flow, cfg)
            farneback_magnitude_panel = render_flow_magnitude_panel("Farneback", farneback_flow, cfg)

            now = time.time()
            fps_now = 1.0 / max(1e-6, now - prev_time)
            prev_time = now
            elapsed = now - time_0
            rest_panel = gray_panel(rest_gray, "Initial preprocessed")
            current_panel = gray_panel(grey_frame_corrected, "Current preprocessed")
            cv.putText(rest_panel, f"FPS={fps_now:.0f}", (10, 45), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
            cv.putText(rest_panel, f"T={elapsed:.2f}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
            combined = render_grid(
                rest_panel,
                current_panel,
                tact_panel,
                tact_magnitude_panel,
                farneback_panel,
                farneback_magnitude_panel,
            )

            cv.imshow("TactMorph phase", render_flow_phase(tact_flow, cfg.max_disp_viz_mag))
            cv.imshow("Farneback phase", render_flow_phase(farneback_flow, cfg.max_disp_viz_mag))

        cv.imshow("raw", frame)
        cv.imshow("tactmorph_vs_farneback", combined)
        heatmap_output.write(combined)

        key = cv.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        frame_count += 1

    cap.release()
    heatmap_output.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Controlled TactMorph vs Farneback tactile displacement comparison")
    parser.add_argument("--config", type=str, default=None, help="JSON config file to load")
    parser.add_argument("--save-config", type=str, default=None, help="write the resolved config to this JSON path and exit")
    parser.add_argument("--input", type=str, default=None, help="capture device index or video file path")
    parser.add_argument("--heat-out", type=str, default=None, help="path to save comparison heatmap video")
    parser.add_argument("--weights", type=str, default=None, help="TactMorph checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cpu or cuda")
    parser.add_argument("--rest-threshold", type=int, default=None, help="threshold for one-time rest marker centroid detection")
    parser.add_argument("--vector-scale", type=float, default=None, help="visual-only multiplier for sampled displacement vectors")
    parser.add_argument("--model-preprocess", type=str, default=None, choices=("none", "area", "maxpool"))
    parser.add_argument("--model-input-size", type=int, default=None)
    parser.add_argument("--mag-min", type=float, default=None, help="minimum flow magnitude for magnitude colormap")
    parser.add_argument("--mag-max", type=float, default=None, help="maximum flow magnitude for magnitude colormap")
    parser.add_argument("--winsize", type=int, default=None, help="Farneback averaging window size")
    parser.add_argument("--levels", type=int, default=None, help="number of Farneback pyramid levels")
    parser.add_argument("--iterations", type=int, default=None, help="Farneback iterations at each pyramid level")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else ComparisonConfig()

    if args.input is not None:
        cfg.input_source = parse_capture_source(args.input)
    if args.heat_out is not None:
        cfg.output_heatmap_path = args.heat_out
    if args.weights is not None:
        cfg.model_weights_path = args.weights
    if args.device is not None:
        cfg.device = args.device
    if args.rest_threshold is not None:
        cfg.rest_marker_threshold = args.rest_threshold
    if args.vector_scale is not None:
        cfg.vector_visual_scale = args.vector_scale
    if args.model_preprocess is not None:
        cfg.model_preprocess = args.model_preprocess
    if args.model_input_size is not None:
        cfg.model_input_size = (args.model_input_size, args.model_input_size)
    if args.mag_min is not None:
        cfg.magnitude_colormap_min = args.mag_min
    if args.mag_max is not None:
        cfg.magnitude_colormap_max = args.mag_max
    if args.winsize is not None:
        cfg.farneback_winsize = args.winsize
    if args.levels is not None:
        cfg.farneback_levels = args.levels
    if args.iterations is not None:
        cfg.farneback_iterations = args.iterations

    if args.save_config is not None:
        save_config(args.save_config, cfg)
        print(f"Saved config to: {args.save_config}")
        raise SystemExit(0)

    main(cfg)
