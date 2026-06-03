import argparse
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2 as cv
import numpy as np


@dataclass
class FarnebackConfig:
    # Device / IO
    input_source: Union[int, str] = 0
    output_heatmap_path: str = "videos/farneback_biotactip_heatmaps.mp4"
    video_codec: str = "XVID"

    # Image geometry, matching tactile_prediction_biotactip.py
    image_size: int = 350
    radius: int = 160
    center: Tuple[int, int] = (175, 175)
    crop_px: int = 175
    crop_py: int = 175
    crop_offset_x: int = 0
    crop_offset_y: int = -8

    # Camera properties
    exposure: float = -7.8
    brightness: float = 0
    contrast: float = 64
    saturation: float = 60
    hue: float = 0
    gain: float = 0

    # Rest marker sampling and visualization
    frame_skip_init: int = 10
    rest_marker_threshold: int = 100
    rest_marker_min_area: int = 8
    rest_marker_max_area: int = 500
    flow_blur_ksize: int = 3
    flow_sample_radius: int = 2
    vector_visual_scale: float = 1.0
    max_disp_viz_mag: float = 5.0

    # Farneback parameters. Defaults mirror the common GelSlim shear baseline.
    farneback_pyr_scale: float = 0.5
    farneback_levels: int = 3
    farneback_winsize: int = 45
    farneback_iterations: int = 3
    farneback_poly_n: int = 5
    farneback_poly_sigma: float = 1.2
    farneback_flags: int = 0


def parse_capture_source(value: str) -> Union[int, str]:
    try:
        return int(value)
    except ValueError:
        return value


def load_config(path: Union[str, Path]) -> FarnebackConfig:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_fields = {field.name for field in fields(FarnebackConfig)}
    cfg = FarnebackConfig()
    for key, value in data.items():
        if key in valid_fields:
            setattr(cfg, key, value)

    cfg.input_source = parse_capture_source(str(cfg.input_source)) if isinstance(cfg.input_source, str) else cfg.input_source
    cfg.center = tuple(cfg.center)
    return cfg


def save_config(path: Union[str, Path], cfg: FarnebackConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
        f.write("\n")


def create_radial_mask(size, center=None, max_value=60, power=2):
    h, w = (size, size) if isinstance(size, int) else size
    cx, cy = (w // 2, h // 2) if center is None else center
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    normalized = distance / np.sqrt(cx**2 + cy**2)
    return np.clip((normalized**power) * max_value, 0, max_value).astype(np.uint8)


def create_circular_mask(shape: Tuple[int, int], center: Tuple[int, int], radius: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv.circle(mask, center, radius, 255, -1)
    return mask


def crop_and_preprocess_frame(
    frame: np.ndarray,
    cropped_limits: list[list[int]],
    circular_mask: np.ndarray,
    radial_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame_cropped = frame[
        cropped_limits[0][1] : cropped_limits[1][1],
        cropped_limits[0][0] : cropped_limits[1][0],
    ]
    cropped_frame = cv.bitwise_and(frame_cropped, frame_cropped, mask=circular_mask)
    grey_frame = cv.cvtColor(cropped_frame, cv.COLOR_BGR2GRAY)
    grey_frame_corrected = cv.subtract(grey_frame, radial_mask)
    return frame_cropped, grey_frame_corrected


def find_rest_marker_centroids(
    gray: np.ndarray,
    threshold: int,
    min_area: int,
    max_area: int,
) -> np.ndarray:
    _, binary = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
    count, _, stats, centroids = cv.connectedComponentsWithStats(binary, connectivity=8)

    points = []
    for label in range(1, count):
        area = int(stats[label, cv.CC_STAT_AREA])
        if min_area <= area <= max_area:
            points.append(centroids[label])

    if not points:
        return np.zeros((0, 2), dtype=np.float32)

    points = np.asarray(points, dtype=np.float32)
    order = np.lexsort((points[:, 0], points[:, 1]))
    return points[order]


def blur_flow_field(flow: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return flow
    if ksize % 2 == 0:
        ksize += 1
    u = cv.blur(flow[0], (ksize, ksize), borderType=cv.BORDER_REFLECT)
    v = cv.blur(flow[1], (ksize, ksize), borderType=cv.BORDER_REFLECT)
    return np.stack([u, v], axis=0)


def sample_flow_at_points(flow: np.ndarray, points: np.ndarray, radius: int = 0) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    height, width = flow.shape[1], flow.shape[2]
    pts = np.asarray(points, dtype=np.float32)
    xs = np.clip(np.round(pts[:, 0]).astype(np.int64), 0, width - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(np.int64), 0, height - 1)

    if radius <= 0:
        u = flow[0, ys, xs]
        v = flow[1, ys, xs]
    else:
        ksize = 2 * radius + 1
        u_blur = cv.blur(flow[0], (ksize, ksize), borderType=cv.BORDER_REFLECT)
        v_blur = cv.blur(flow[1], (ksize, ksize), borderType=cv.BORDER_REFLECT)
        u = u_blur[ys, xs]
        v = v_blur[ys, xs]

    return np.stack([u, v], axis=1).astype(np.float32)


def draw_displacement_vectors(
    image: np.ndarray,
    base_points: np.ndarray,
    displacement: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    origin_color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
    tip_length: float = 0.2,
    copy: bool = True,
) -> np.ndarray:
    img = image.copy() if copy else image
    base_points = np.asarray(base_points, dtype=np.float32)
    displacement = np.asarray(displacement, dtype=np.float32)

    if base_points.shape != displacement.shape or base_points.shape[1] != 2:
        raise ValueError("base_points and displacement must be shape (N, 2).")

    height, width = img.shape[:2]
    for (x, y), (dx, dy) in zip(base_points, displacement):
        start = (int(round(x)), int(round(y)))
        end = (int(round(x + dx)), int(round(y + dy)))
        if not (0 <= start[0] < width and 0 <= start[1] < height):
            continue
        end = (int(np.clip(end[0], 0, width - 1)), int(np.clip(end[1], 0, height - 1)))
        cv.arrowedLine(img, start, end, color, thickness, cv.LINE_AA, tipLength=tip_length)
        cv.circle(img, start, 1, origin_color, -1)

    return img


def gray_panel(gray: np.ndarray, label: str) -> np.ndarray:
    panel = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    cv.putText(panel, label, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
    return panel


def render_flow_heatmap(flow: np.ndarray, max_disp_viz_mag: float) -> np.ndarray:
    flow_mag = np.linalg.norm(flow, axis=0)
    denom = max(max_disp_viz_mag, float(flow_mag.max()), 1e-6)
    flow_norm = np.clip(flow_mag / denom, 0.0, 1.0)
    return cv.applyColorMap(np.uint8(flow_norm * 255), cv.COLORMAP_PLASMA)


def render_flow_phase(flow: np.ndarray, max_disp_viz_mag: float) -> np.ndarray:
    fx = flow[0]
    fy = flow[1]
    angle = np.arctan2(fy, fx)
    magnitude = np.sqrt(fx**2 + fy**2)

    hue = np.uint8(((angle + np.pi) / (2.0 * np.pi)) * 179.0)
    denom = max(max_disp_viz_mag, float(magnitude.max()), 1e-6)
    saturation = np.uint8(np.clip(magnitude / denom, 0.0, 1.0) * 255.0)
    value = np.full_like(hue, 255, dtype=np.uint8)

    hsv = cv.merge((hue, saturation, value))
    phase_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.putText(
        phase_bgr,
        "Farneback direction: hue=angle, saturation=magnitude",
        (10, 20),
        cv.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )
    return phase_bgr


def calculate_farneback_flow(rest_gray: np.ndarray, current_gray: np.ndarray, cfg: FarnebackConfig) -> np.ndarray:
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


def main(cfg: FarnebackConfig):
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
    combined_heatmap_size = (cropped_size[0] * 3, cropped_size[1])
    heatmap_output = cv.VideoWriter(cfg.output_heatmap_path, fourcc, capture_fps, combined_heatmap_size, True)

    frame_count = 0
    time_0 = time.time()
    prev_time = time.time()
    rest_gray = None
    rest_centroids = np.zeros((0, 2), dtype=np.float32)

    print("Farneback optical flow baseline")
    print(
        "params: "
        f"pyr_scale={cfg.farneback_pyr_scale}, levels={cfg.farneback_levels}, "
        f"winsize={cfg.farneback_winsize}, iterations={cfg.farneback_iterations}, "
        f"poly_n={cfg.farneback_poly_n}, poly_sigma={cfg.farneback_poly_sigma}, "
        f"flags={cfg.farneback_flags}"
    )

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
            rest_centroids = find_rest_marker_centroids(
                rest_gray,
                cfg.rest_marker_threshold,
                cfg.rest_marker_min_area,
                cfg.rest_marker_max_area,
            )
            print(f"Captured rest frame with {len(rest_centroids)} marker centroids.")
            if len(rest_centroids) == 0:
                print("No rest markers detected; lower --rest-threshold or area limits to show vectors.")

        if rest_gray is None:
            displacement_heatmap = cv.cvtColor(grey_frame_corrected, cv.COLOR_GRAY2BGR)
            cv.putText(
                displacement_heatmap,
                f"Capturing rest frame in {max(0, cfg.frame_skip_init - frame_count)} frames",
                (10, 24),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv.LINE_AA,
            )
        else:
            flow = calculate_farneback_flow(rest_gray, grey_frame_corrected, cfg)
            flow = blur_flow_field(flow, cfg.flow_blur_ksize)

            displacement = sample_flow_at_points(flow, rest_centroids, cfg.flow_sample_radius)
            displacement_viz = displacement * cfg.vector_visual_scale
            mean_disp = displacement.mean(axis=0) if displacement.size else np.zeros(2, dtype=np.float32)

            displacement_heatmap = render_flow_heatmap(flow, cfg.max_disp_viz_mag)
            displacement_heatmap = draw_displacement_vectors(displacement_heatmap, rest_centroids, displacement_viz)

            overlay = cv.cvtColor(grey_frame_corrected, cv.COLOR_GRAY2BGR)
            overlay = draw_displacement_vectors(overlay, rest_centroids, displacement_viz)
            cv.imshow("Farneback Displacement Field", overlay)
            cv.imshow("Farneback Flow Direction Phase", render_flow_phase(flow, cfg.max_disp_viz_mag))

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

        cv.putText(displacement_heatmap, "Farneback displacement", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)

        timenow = time.time() - time_0
        now = time.time()
        fps_now = 1.0 / max(1e-6, now - prev_time)
        prev_time = now
        cv.putText(displacement_heatmap, f"FPS={fps_now:.0f}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
        cv.putText(displacement_heatmap, f"T={float(timenow):.2f}", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

        input_panel = gray_panel(grey_frame_corrected, "Input")
        rest_panel = gray_panel(rest_gray if rest_gray is not None else grey_frame_corrected, "Rest")
        combined_heatmap = np.concatenate((input_panel, displacement_heatmap, rest_panel), axis=1)

        cv.imshow("raw", frame)
        cv.imshow("farneback_tactile_heatmaps", combined_heatmap)
        heatmap_output.write(combined_heatmap)

        key = cv.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        frame_count += 1

    cap.release()
    heatmap_output.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Farneback optical-flow tactile displacement visualization")
    parser.add_argument("--config", type=str, default=None, help="JSON config file to load")
    parser.add_argument("--save-config", type=str, default=None, help="write the resolved config to this JSON path and exit")
    parser.add_argument("--input", type=str, default=None, help="capture device index or video file path")
    parser.add_argument("--heat-out", type=str, default=None, help="path to save Farneback heatmap video")
    parser.add_argument("--rest-threshold", type=int, default=None, help="threshold for one-time rest marker centroid detection")
    parser.add_argument("--vector-scale", type=float, default=None, help="visual-only multiplier for sampled displacement vectors")
    parser.add_argument("--winsize", type=int, default=None, help="Farneback averaging window size")
    parser.add_argument("--levels", type=int, default=None, help="number of Farneback pyramid levels")
    parser.add_argument("--iterations", type=int, default=None, help="Farneback iterations at each pyramid level")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else FarnebackConfig()

    if args.input is not None:
        cfg.input_source = parse_capture_source(args.input)
    if args.heat_out is not None:
        cfg.output_heatmap_path = args.heat_out
    if args.rest_threshold is not None:
        cfg.rest_marker_threshold = args.rest_threshold
    if args.vector_scale is not None:
        cfg.vector_visual_scale = args.vector_scale
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
