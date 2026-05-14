import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import cv2 as cv
import numpy as np


@dataclass
class Config:
    # Device / IO
    input_source: Union[int, str] = 0
    output_original_path: str = "videos/farneback_original.mp4"
    output_flow_path: str = "videos/farneback_displacement.mp4"
    video_codec: str = "XVID"

    # Image geometry
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
    fixed_frame_number: int = 5
    rest_marker_threshold: int = 100
    rest_marker_min_area: int = 8
    rest_marker_max_area: int = 500
    flow_sample_radius: int = 2
    vector_visual_scale: float = 1.0
    max_disp_viz_mag: float = 5.0
    fallback_grid_step: int = 25

    # Farneback optical-flow parameters
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 25
    iterations: int = 5
    poly_n: int = 7
    poly_sigma: float = 1.5
    flags: int = 0


def create_radial_mask(size, center=None, max_value=60, power=2):
    h, w = (size, size) if isinstance(size, int) else size
    cx, cy = (w // 2, h // 2) if center is None else center
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    normalized = distance / np.sqrt(cx**2 + cy**2)
    return np.clip((normalized**power) * max_value, 0, max_value).astype(np.uint8)


def create_circular_mask(shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv.circle(mask, center, radius, 255, -1)
    return mask


def parse_capture_source(value: str) -> Union[int, str]:
    try:
        return int(value)
    except ValueError:
        return value


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


def sample_regular_grid(height: int, width: int, step: int) -> np.ndarray:
    ys = np.arange(step // 2, height, step, dtype=np.int32)
    xs = np.arange(step // 2, width, step, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)


def sample_flow_at_points(flow: np.ndarray, points: np.ndarray, radius: int = 0) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    height, width = flow.shape[:2]
    pts = np.asarray(points, dtype=np.float32)
    xs = np.clip(np.round(pts[:, 0]).astype(np.int64), 0, width - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(np.int64), 0, height - 1)

    if radius <= 0:
        sampled = flow[ys, xs]
    else:
        ksize = 2 * radius + 1
        u_blur = cv.blur(flow[:, :, 0], (ksize, ksize), borderType=cv.BORDER_REFLECT)
        v_blur = cv.blur(flow[:, :, 1], (ksize, ksize), borderType=cv.BORDER_REFLECT)
        sampled = np.stack([u_blur[ys, xs], v_blur[ys, xs]], axis=1)

    return sampled.astype(np.float32)


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


def render_flow_heatmap(flow: np.ndarray, max_disp_viz_mag: float) -> np.ndarray:
    flow_mag = np.linalg.norm(flow, axis=2)
    denom = max(max_disp_viz_mag, float(flow_mag.max()), 1e-6)
    flow_norm = np.clip(flow_mag / denom, 0.0, 1.0)
    return cv.applyColorMap(np.uint8(flow_norm * 255), cv.COLORMAP_PLASMA)


def calculate_farneback_flow(fixed_gray: np.ndarray, moving_gray: np.ndarray, cfg: Config) -> np.ndarray:
    return cv.calcOpticalFlowFarneback(
        fixed_gray,
        moving_gray,
        None,
        cfg.pyr_scale,
        cfg.levels,
        cfg.winsize,
        cfg.iterations,
        cfg.poly_n,
        cfg.poly_sigma,
        cfg.flags,
    )


def main(cfg: Config):
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

    Path(cfg.output_original_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_flow_path).parent.mkdir(parents=True, exist_ok=True)
    out_original = cv.VideoWriter(cfg.output_original_path, fourcc, capture_fps, cropped_size, False)
    flow_output = cv.VideoWriter(cfg.output_flow_path, fourcc, capture_fps, cropped_size, True)

    fixed_gray = None
    base_points = np.zeros((0, 2), dtype=np.float32)
    frame_number = 0
    time_0 = time.time()
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break

        frame_number += 1
        _, gray = crop_and_preprocess_frame(frame, cropped_limits, circular_mask, radial_mask)
        out_original.write(gray)

        if fixed_gray is None and frame_number < cfg.fixed_frame_number:
            waiting_frame = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
            cv.putText(
                waiting_frame,
                f"Waiting for fixed frame {cfg.fixed_frame_number}",
                (10, 24),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv.LINE_AA,
            )
            cv.imshow("raw", frame)
            cv.imshow("farneback_displacement_overlay", waiting_frame)
            cv.imshow("farneback_displacement_heatmap", waiting_frame)
            flow_output.write(waiting_frame)

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            continue

        if fixed_gray is None:
            fixed_gray = gray.copy()
            base_points = find_rest_marker_centroids(
                fixed_gray,
                cfg.rest_marker_threshold,
                cfg.rest_marker_min_area,
                cfg.rest_marker_max_area,
            )
            if base_points.size == 0:
                base_points = sample_regular_grid(frame_shape[0], frame_shape[1], cfg.fallback_grid_step)
                print(f"No markers detected in fixed frame {frame_number}; using {len(base_points)} regular grid points.")
            else:
                print(f"Captured frame {frame_number} as fixed image with {len(base_points)} marker centroids.")

        flow = calculate_farneback_flow(fixed_gray, gray, cfg)
        displacement = sample_flow_at_points(flow, base_points, cfg.flow_sample_radius)
        displacement_viz = displacement * cfg.vector_visual_scale
        mean_disp = displacement.mean(axis=0) if displacement.size else np.zeros(2, dtype=np.float32)

        flow_heatmap = render_flow_heatmap(flow, cfg.max_disp_viz_mag)
        flow_heatmap = draw_displacement_vectors(flow_heatmap, base_points, displacement_viz)

        overlay = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        overlay = draw_displacement_vectors(overlay, base_points, displacement_viz)

        timenow = time.time() - time_0
        now = time.time()
        fps_now = 1.0 / max(1e-6, now - prev_time)
        prev_time = now

        cv.putText(flow_heatmap, "Farneback displacement", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
        cv.putText(flow_heatmap, f"FPS={fps_now:.0f}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
        cv.putText(flow_heatmap, f"T={timenow:.2f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
        cv.putText(flow_heatmap, f"mean dx={mean_disp[0]:.2f}, dy={mean_disp[1]:.2f}", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)

        cv.imshow("raw", frame)
        cv.imshow("farneback_displacement_overlay", overlay)
        cv.imshow("farneback_displacement_heatmap", flow_heatmap)
        flow_output.write(flow_heatmap)

        key = cv.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
    cap.release()
    out_original.release()
    flow_output.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online Farneback optical-flow displacement visualization for tactile sensor images")
    parser.add_argument("--input", type=str, default="0", help="capture device index or video file path")
    parser.add_argument("--orig-out", type=str, default=Config.output_original_path, help="path to save grayscale capture")
    parser.add_argument("--flow-out", type=str, default=Config.output_flow_path, help="path to save Farneback displacement video")
    parser.add_argument("--fixed-frame", type=int, default=Config.fixed_frame_number, help="one-based frame number to use as fixed/rest image")
    parser.add_argument("--rest-threshold", type=int, default=Config.rest_marker_threshold, help="threshold for first-frame marker centroid detection")
    parser.add_argument("--vector-scale", type=float, default=Config.vector_visual_scale, help="visual-only multiplier for sampled displacement vectors")
    parser.add_argument("--max-viz-mag", type=float, default=Config.max_disp_viz_mag, help="flow magnitude mapped to the top of the color scale")
    parser.add_argument("--winsize", type=int, default=Config.winsize, help="Farneback averaging window size")
    parser.add_argument("--levels", type=int, default=Config.levels, help="Farneback pyramid levels")
    parser.add_argument("--iterations", type=int, default=Config.iterations, help="Farneback iterations per pyramid level")
    args = parser.parse_args()

    cfg = Config()
    cfg.input_source = parse_capture_source(args.input)
    cfg.output_original_path = args.orig_out
    cfg.output_flow_path = args.flow_out
    cfg.fixed_frame_number = max(1, args.fixed_frame)
    cfg.rest_marker_threshold = args.rest_threshold
    cfg.vector_visual_scale = args.vector_scale
    cfg.max_disp_viz_mag = args.max_viz_mag
    cfg.winsize = args.winsize
    cfg.levels = args.levels
    cfg.iterations = args.iterations

    main(cfg)
