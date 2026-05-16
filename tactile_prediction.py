import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2 as cv
import numpy as np
import torch
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN

from voxelmorph.model import VoxelMorph2D


@dataclass
class Config:
    # Device / IO
    input_source: Union[int, str] = 0
    output_original_path: str = "videos/output_original.mp4"
    output_heatmap_path: str = "videos/output_from_online_cap.mp4"
    video_codec: str = "XVID"

    # Learned displacement model
    model_weights_path: str = "voxelmorph/ckpt/biotactip_voxelmorph2d_115.pt"
    device: Optional[str] = None

    # Image geometry
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

    # Normal-force / marker-intensity heatmap
    threshold_bin: int = 100
    dbscan_eps_points: int = 3
    dbscan_min_samples_points: int = 8
    gaussian_sigma: float = 20
    maxima_eps: int = 30
    maxima_min_samples: int = 8
    maxima_seed_threshold: float = 25.0


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


def parse_capture_source(value: str) -> Union[int, str]:
    try:
        return int(value)
    except ValueError:
        return value


def get_device(device_name: Optional[str]) -> torch.device:
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def get_voxelmorph_model(weights_path: str, device: torch.device) -> VoxelMorph2D:
    checkpoint = Path(weights_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"VoxelMorph checkpoint not found: {checkpoint}")

    state = torch.load(checkpoint, map_location=device)
    model = VoxelMorph2D()
    if isinstance(state, torch.nn.Module):
        model = state
    else:
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        if isinstance(state, dict):
            state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


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


def gray_to_tensor(gray: np.ndarray, device: torch.device) -> torch.Tensor:
    image = gray.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    return torch.from_numpy(image)[None, None].to(device)


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


def dbscan_extractor(dbscan_result, points):
    labels = dbscan_result.labels_
    points = np.asarray(points)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]

    clusters = []
    for cluster_id in unique_labels:
        clusters.append(points[labels == cluster_id])
    return clusters


def centroids_calc(cluster_array):
    centroids = np.zeros((0, 2), dtype=np.float32)
    intensity = np.zeros((0,), dtype=np.float32)

    for cluster in cluster_array:
        centroid = np.mean(cluster, axis=0)
        # Matches the original intensity proxy: active bright-pixel count per marker.
        marker_intensity = float(cluster.shape[0])
        centroids = np.append(centroids, [centroid], axis=0)
        intensity = np.append(intensity, [marker_intensity], axis=0)

    return centroids, intensity


def marker_intensity_points(
    gray: np.ndarray,
    threshold: int,
    eps: int,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    _, binary = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
    pts = cv.findNonZero(binary)
    if pts is None:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    cluster_coordinates = pts.reshape(-1, 2)
    cluster_data = DBSCAN(eps=eps, min_samples=min_samples).fit(cluster_coordinates)
    clusters = dbscan_extractor(cluster_data, cluster_coordinates)
    return centroids_calc(clusters)


class GradDescent:
    def __init__(self, arr_z: np.ndarray, seeds: np.ndarray, cropped_size: tuple[int, int]):
        self.arr_z = arr_z
        self.iter_pts = np.asarray(seeds, dtype=np.float32)
        self._cropped_size = tuple(cropped_size)
        if self.iter_pts.size == 0:
            self.iter_pts = np.zeros((0, 2), dtype=np.int16)
            return

        alpha = 0.05
        beta = 0.5
        scale = 10
        weights = ((10, 0.2), (5, 0.3), (3, 0.5))

        try:
            for i in range(50):
                pts_i = self.iter_pts.astype(np.int16)
                limit_mask = self.valid_mask(pts_i, max(step for step, _ in weights))
                pts_i = pts_i[limit_mask]
                self.iter_pts = self.iter_pts[limit_mask]
                if pts_i.size == 0:
                    break
                grad = sum(self.grad_at(pts_i, step) * weight for step, weight in weights) * scale
                self.iter_pts = self.iter_pts + grad * np.exp(-alpha * i + beta)
            self.iter_pts = self.iter_pts.astype(np.int16)
        except Exception as exc:
            print(f"Gradient-descent maxima step failed: {exc}")

    def valid_mask(self, pts: np.ndarray, step: int) -> np.ndarray:
        return (
            (pts[:, 0] <= self._cropped_size[0] - step - 1)
            & (pts[:, 1] <= self._cropped_size[1] - step - 1)
            & (pts[:, 0] >= step)
            & (pts[:, 1] >= step)
        )

    def grad_at(self, pts: np.ndarray, step: int) -> np.ndarray:
        grad_y = (self.arr_z[pts[:, 1] + step, pts[:, 0]] - self.arr_z[pts[:, 1] - step, pts[:, 0]]) / (2 * step)
        grad_x = (self.arr_z[pts[:, 1], pts[:, 0] + step] - self.arr_z[pts[:, 1], pts[:, 0] - step]) / (2 * step)
        return np.stack([grad_x, grad_y], axis=1)


def interpolate_reg(data: np.ndarray, scale_factor: float = 5.0) -> np.ndarray:
    data = data * scale_factor
    data[data < 0] = 0
    data[data > 255] = 255
    return data.astype(np.uint8)


def render_intensity_heatmap(
    gray: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    cropped_size: tuple[int, int],
    cfg: Config,
) -> np.ndarray:
    centroids, intensity = marker_intensity_points(
        gray,
        cfg.threshold_bin,
        cfg.dbscan_eps_points,
        cfg.dbscan_min_samples_points,
    )

    if centroids.size:
        z_val = griddata(centroids, intensity, (xgrid, ygrid), method="linear", fill_value=0.0)
    else:
        z_val = np.zeros_like(xgrid)

    reg_z = gaussian_filter(interpolate_reg(z_val), sigma=cfg.gaussian_sigma)
    heatmap = cv.applyColorMap(reg_z, cv.COLORMAP_JET)

    pack = np.column_stack((centroids, intensity)) if centroids.size else np.zeros((0, 3), dtype=np.float32)
    seeds = pack[pack[:, 2] > cfg.maxima_seed_threshold, 0:2] if pack.size else np.zeros((0, 2), dtype=np.float32)
    gd = GradDescent(z_val, seeds, cropped_size)

    if gd.iter_pts.size != 0:
        try:
            maxima_cls = DBSCAN(eps=cfg.maxima_eps, min_samples=cfg.maxima_min_samples).fit(gd.iter_pts)
            maxima_clusters = dbscan_extractor(maxima_cls, gd.iter_pts)
            maxima, _ = centroids_calc(maxima_clusters)

            maxima_circular_mask = np.zeros_like(z_val, dtype=np.uint8)
            for maximum in maxima.astype(np.int16):
                cv.circle(maxima_circular_mask, tuple(maximum), 35, 255, -1)
                maxarr = cv.bitwise_and(reg_z.astype(np.uint8), maxima_circular_mask)
                pt = np.unravel_index(np.argmax(maxarr), maxarr.shape)
                cv.drawMarker(heatmap, [pt[1], pt[0]], (0, 255, 0), cv.MARKER_CROSS, 15, 1)
                cv.putText(
                    heatmap,
                    f"{z_val[pt]:.2f}",
                    [pt[1] + 5, pt[0] + 15],
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv.LINE_AA,
                )
                maxima_circular_mask.fill(0)
        except ValueError:
            pass

    cv.putText(heatmap, "Intensity", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
    return heatmap


@torch.no_grad()
def infer_warped_and_flow(
    model: VoxelMorph2D,
    moving_tensor: torch.Tensor,
    fixed_tensor: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    warped, flow = model(moving_tensor, fixed_tensor)
    warped_np = warped.squeeze().detach().cpu().numpy()
    flow_np = flow.squeeze(0).detach().cpu().numpy()
    return warped_np, flow_np


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


def render_flow_heatmap(flow: np.ndarray, max_disp_viz_mag: float) -> np.ndarray:
    flow_mag = np.linalg.norm(flow, axis=0)
    denom = max(max_disp_viz_mag, float(flow_mag.max()), 1e-6)
    flow_norm = np.clip(flow_mag / denom, 0.0, 1.0)
    return cv.applyColorMap(np.uint8(flow_norm * 255), cv.COLORMAP_PLASMA)


def gray_panel(gray: np.ndarray, label: str) -> np.ndarray:
    panel = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    cv.putText(panel, label, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
    return panel


def render_registration_comparison(
    rest_gray: np.ndarray,
    current_gray: np.ndarray,
    warped_current: np.ndarray,
) -> np.ndarray:
    warped_u8 = np.uint8(np.clip(warped_current, 0.0, 1.0) * 255.0)
    diff = cv.absdiff(rest_gray, warped_u8)
    diff_color = cv.applyColorMap(diff, cv.COLORMAP_INFERNO)
    cv.putText(diff_color, "absdiff(rest, warped)", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)

    return np.concatenate(
        (
            gray_panel(rest_gray, "rest/fixed"),
            gray_panel(current_gray, "current/moving"),
            gray_panel(warped_u8, "warped current -> rest"),
            diff_color,
        ),
        axis=1,
    )


def main(cfg: Config):
    device = get_device(cfg.device)
    print(f"Using device: {device}")
    model = get_voxelmorph_model(cfg.model_weights_path, device)

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
    Path(cfg.output_heatmap_path).parent.mkdir(parents=True, exist_ok=True)
    out_original = cv.VideoWriter(cfg.output_original_path, fourcc, capture_fps, cropped_size, False)
    combined_heatmap_size = (cropped_size[0] * 2, cropped_size[1])
    heatmap_output = cv.VideoWriter(cfg.output_heatmap_path, fourcc, capture_fps, combined_heatmap_size, True)

    xspace = np.linspace(0, cropped_size[0] - 1, cropped_size[0])
    yspace = np.linspace(0, cropped_size[1] - 1, cropped_size[1])
    xgrid, ygrid = np.meshgrid(xspace, yspace)

    frame_count = 0
    time_0 = time.time()
    prev_time = time.time()
    rest_tensor = None
    rest_gray = None
    rest_centroids = np.zeros((0, 2), dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break

        _, grey_frame_corrected = crop_and_preprocess_frame(
            frame, cropped_limits, circular_mask, radial_mask
        )
        out_original.write(grey_frame_corrected)
        intensity_heatmap = render_intensity_heatmap(grey_frame_corrected, xgrid, ygrid, cropped_size, cfg)

        if frame_count == cfg.frame_skip_init:
            rest_gray = grey_frame_corrected.copy()
            rest_tensor = gray_to_tensor(grey_frame_corrected, device)
            rest_centroids = find_rest_marker_centroids(
                grey_frame_corrected,
                cfg.rest_marker_threshold,
                cfg.rest_marker_min_area,
                cfg.rest_marker_max_area,
            )
            print(f"Captured rest frame with {len(rest_centroids)} marker centroids.")
            if len(rest_centroids) == 0:
                print("No rest markers detected; lower --rest-threshold or area limits to show vectors.")

        if rest_tensor is None:
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
            moving_tensor = gray_to_tensor(grey_frame_corrected, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            warped_current, flow = infer_warped_and_flow(model, moving_tensor, rest_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()

            flow = blur_flow_field(flow, cfg.flow_blur_ksize)
            displacement = sample_flow_at_points(flow, rest_centroids, cfg.flow_sample_radius)
            displacement_viz = displacement * cfg.vector_visual_scale
            mean_disp = displacement.mean(axis=0) if displacement.size else np.zeros(2, dtype=np.float32)

            displacement_heatmap = render_flow_heatmap(flow, cfg.max_disp_viz_mag)
            displacement_heatmap = draw_displacement_vectors(displacement_heatmap, rest_centroids, displacement_viz)

            overlay = cv.cvtColor(grey_frame_corrected, cv.COLOR_GRAY2BGR)
            overlay = draw_displacement_vectors(overlay, rest_centroids, displacement_viz)
            cv.imshow("Learned Displacement Field", overlay)
            cv.imshow("Registration Comparison", render_registration_comparison(rest_gray, grey_frame_corrected, warped_current))

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

        cv.putText(displacement_heatmap, "Displacement", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)

        timenow = time.time() - time_0
        now = time.time()
        fps_now = 1.0 / max(1e-6, now - prev_time)
        prev_time = now
        cv.putText(displacement_heatmap, f"FPS={fps_now:.0f}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
        cv.putText(displacement_heatmap, f"T={float(timenow):.2f}", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

        combined_heatmap = np.concatenate((displacement_heatmap, intensity_heatmap), axis=1)

        cv.imshow("raw", frame)
        cv.imshow("tactile_heatmaps", combined_heatmap)
        heatmap_output.write(combined_heatmap)

        key = cv.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        frame_count += 1

    cap.release()
    out_original.release()
    heatmap_output.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online learned displacement visualization for tactile sensor images")
    parser.add_argument("--input", type=str, default="0", help="capture device index or video file path")
    parser.add_argument("--orig-out", type=str, default="videos/output_original.mp4", help="path to save grayscale capture")
    parser.add_argument("--heat-out", type=str, default="videos/output_from_online_cap.mp4", help="path to save flow heatmap video")
    parser.add_argument("--weights", type=str, default=Config.model_weights_path, help="VoxelMorph checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cpu or cuda")
    parser.add_argument("--rest-threshold", type=int, default=Config.rest_marker_threshold, help="threshold for one-time rest marker centroid detection")
    parser.add_argument("--intensity-threshold", type=int, default=Config.threshold_bin, help="threshold for normal-force marker intensity heatmap")
    parser.add_argument("--intensity-sigma", type=float, default=Config.gaussian_sigma, help="Gaussian smoothing sigma for normal-force heatmap")
    parser.add_argument("--vector-scale", type=float, default=Config.vector_visual_scale, help="visual-only multiplier for sampled displacement vectors")
    args = parser.parse_args()

    cfg = Config()
    cfg.input_source = parse_capture_source(args.input)
    cfg.output_original_path = args.orig_out
    cfg.output_heatmap_path = args.heat_out
    cfg.model_weights_path = args.weights
    cfg.device = args.device
    cfg.rest_marker_threshold = args.rest_threshold
    cfg.threshold_bin = args.intensity_threshold
    cfg.gaussian_sigma = args.intensity_sigma
    cfg.vector_visual_scale = args.vector_scale

    main(cfg)
