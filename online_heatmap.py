import numpy as np
import cv2 as cv
import open3d as o3d
from scipy.interpolate import griddata
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter
import time
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from sklearn.datasets import make_blobs
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from pycpd import DeformableRegistration
from scipy.interpolate import NearestNDInterpolator
from vector_plotter import VectorFieldVisualizer as vfv
from vsp.vsp.tracker import NearestNeighbourTracker

# --- Configuration ---
import argparse
from dataclasses import dataclass

@dataclass
class Config:
    # Device / IO
    input_source: int = 0  # camera index or video file path
    output_original_path: str = 'videos/output_original.mp4'
    output_heatmap_path: str = 'videos/output_from_online_cap.mp4'
    video_codec: str = 'XVID'

    # Image geometry (magic numbers used in original script)
    image_size: int = 350
    radius: int = 160
    center: tuple = (175, 175)
    crop_px: int = 175
    crop_py: int = 175
    crop_offset_x: int = 0
    crop_offset_y: int = -8

    # Camera properties (set after opening capture)
    exposure: float = -7.8
    brightness: float = 0
    contrast: float = 64
    saturation: float = 60
    hue: float = 0
    gain: float = 0

    # Processing params
    threshold_bin: int = 100
    dbscan_eps_points: int = 3
    dbscan_min_samples_points: int = 8
    maxima_eps: int = 30
    maxima_min_samples: int = 8
    gaussian_sigma: float = 20
    cpd_scale_factor: float = 1/100
    frame_skip_init: int = 10


# illuminance correct mask, makes darker at edge
def create_radial_mask(size, center=None, max_value=60, power=2):
    h, w = (size, size) if isinstance(size, int) else size
    cx, cy = (w//2, h//2) if center is None else center
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    normalized = distance / np.sqrt(cx**2 + cy**2)
    return np.clip((normalized ** power) * max_value, 0, max_value).astype(np.uint8)


# ---------------- Helper functions ----------------
def dbscan_extractor(dbscan_result, points):
    labels = dbscan_result.labels_
    points = np.array(points)

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]

    if len(unique_labels) == 0:
        return []

    cluster_info = []
    for cluster_id in unique_labels:
        cluster_mask = (labels == cluster_id)
        cluster_points = points[cluster_mask]
        cluster_info.append(cluster_points)

    return cluster_info


def centroids_calc(cluster_array):
    result = np.zeros((0, 2))
    intsty = np.zeros((0)).astype(np.uint16)
    for cluster in cluster_array:
        centroid = np.mean(cluster, axis=0)
        n_pts = cluster.shape[0]
        intensity = n_pts
        result = np.append(result, [centroid], axis=0)
        intsty = np.append(intsty, [intensity], axis=0)
    return result, intsty


def draw_centroid_cv(centroids, image, color=(0, 0, 255), flip=True):
    if flip:
        image = cv.flip(image, 0)
    for centroid in centroids:
        x, y = int(centroid[0]), int(centroid[1])
        cv.circle(image, (x, y), 2, color, -1)
    return image


class grad_descent():
    def __init__(self, arrZ, seeds, cropped_size):
        self.arrZ = arrZ
        self.iter_pts = seeds
        alpha = 0.05
        beta = 0.5
        scale = 10
        iters = 50
        w1 = 0.2
        w2 = 0.3
        w3 = 0.5
        self._cropped_size = tuple(cropped_size)
        try:
            for i in range(iters):
                grad = (self.grad(self.iter_pts.astype(np.int16), step=10) * w1 +
                        self.grad(self.iter_pts.astype(np.int16), step=5) * w2 +
                        self.grad(self.iter_pts.astype(np.int16), step=3) * w3) * scale
                self.iter_pts = self.iter_pts + grad * np.exp(-alpha * i + beta)
            self.iter_pts = self.iter_pts.astype(np.int16)
        except Exception as e:
            print(f"Exception in class grad_descent: {e}")

    def grad(self, pts, step):
        try:
            pts_limit_mask = (pts[:, 0] <= self._cropped_size[0] - step) & (pts[:, 1] <= self._cropped_size[1] - step) & (pts[:, 0] >= 0) & (pts[:, 1] >= 0)
            pts = pts[pts_limit_mask]
            self.iter_pts = self.iter_pts[pts_limit_mask]
            gradY = 1 / (2 * step) * (self.arrZ[pts[:, 1] + step, pts[:, 0]] - self.arrZ[pts[:, 1] - step, pts[:, 0]])
            gradX = 1 / (2 * step) * (self.arrZ[pts[:, 1], pts[:, 0] + step] - self.arrZ[pts[:, 1], pts[:, 0] - step])
            return np.stack([gradX, gradY], axis=1)
        except Exception as e:
            print(f"Exception in class grad_descent, function grad: {e}")


def create_lines(origins, vectors, cmap=cm.viridis):
    points = np.vstack([origins, origins + vectors])
    lines = [[i, i + len(origins)] for i in range(len(origins))]
    lengths = np.linalg.norm(vectors, axis=1)
    norm = (lengths - lengths.min()) / (np.ptp(lengths) + 1e-9)
    colors_arr = cmap(norm)[:, :3]
    line_colors = [colors_arr[i] for i in range(len(origins))]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    return line_set


def interpolate_reg(data, mode='zero-to-one', adaptive=False, scaleFactor=5.0):
    if adaptive:
        mn, mx = np.min(data), np.max(data)
        data = data - mn
        data = data * (1.0 / (mx - mn))
        if mode == 'cv-image':
            return (data * 255).astype(np.uint8)
        return data
    else:
        if mode == 'cv-image':
            data = data * scaleFactor
            data[data < 0] = 0
            data[data > 255] = 255
            return data.astype(np.uint8)
        return data


# ---------------- Main processing ----------------

def main(cfg: Config):
    # Open capture
    cap = cv.VideoCapture(cfg.input_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open capture source {cfg.input_source}")

    # get capture properties
    capture_fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    video_w, video_h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*cfg.video_codec)

    # set camera properties that were originally set
    cap.set(cv.CAP_PROP_FRAME_WIDTH, video_w)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, video_h)
    cap.set(cv.CAP_PROP_EXPOSURE, cfg.exposure)
    cap.set(cv.CAP_PROP_BRIGHTNESS, cfg.brightness)
    cap.set(cv.CAP_PROP_CONTRAST, cfg.contrast)
    cap.set(cv.CAP_PROP_SATURATION, cfg.saturation)
    cap.set(cv.CAP_PROP_HUE, cfg.hue)
    cap.set(cv.CAP_PROP_GAIN, cfg.gain)

    # compute cropping and masks
    cnt = [int(video_w / 2), int(video_h / 2)]
    cropped_limits = [[cnt[0] - cfg.crop_px + cfg.crop_offset_x, cnt[1] - cfg.crop_py + cfg.crop_offset_y],
                      [cnt[0] + cfg.crop_px + cfg.crop_offset_x, cnt[1] + cfg.crop_py + cfg.crop_offset_y]]
    cropped_size = [2 * cfg.crop_px, 2 * cfg.crop_py]

    o_ring_mask = create_radial_mask((cfg.image_size, cfg.image_size), max_value=245)

    out_original = cv.VideoWriter(cfg.output_original_path, fourcc, capture_fps, tuple(cropped_size), False)
    heatmap_output = cv.VideoWriter(cfg.output_heatmap_path, fourcc, capture_fps, tuple(cropped_size), True)

    # circular mask
    c_mask = np.zeros((cfg.image_size, cfg.image_size), dtype=np.uint8)
    cv.circle(c_mask, cfg.center, cfg.radius, (255), -1)

    # precompute grid used for interpolation
    xspace = np.linspace(0, cfg.image_size, cfg.image_size)
    yspace = np.linspace(0, cfg.image_size, cfg.image_size)
    xgrid, ygrid = np.meshgrid(xspace, yspace)

    frame_count = 0
    time_0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break

        time_start = time.time()
        frame_cropped = frame[cropped_limits[0][1]:cropped_limits[1][1], cropped_limits[0][0]:cropped_limits[1][0]]
        cropped_frame = cv.bitwise_and(frame_cropped, frame_cropped, mask=c_mask)
        grey_frame = cv.cvtColor(cropped_frame, cv.COLOR_BGR2GRAY)
        grey_frame_corrected = cv.subtract(grey_frame, o_ring_mask)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        out_original.write(grey_frame_corrected)
        _, frame_binary = cv.threshold(grey_frame_corrected, cfg.threshold_bin, 255, cv.THRESH_BINARY)
        cv.imshow('binary_frame', frame_binary)

        # perform dbscan algorithm
        pts = cv.findNonZero(frame_binary)
        if pts is None:
            centroids = np.zeros((0, 2))
            intensity = np.zeros((0,))
        else:
            cluster_coordinates = pts.reshape(-1, 2)
            cluster_data = DBSCAN(eps=cfg.dbscan_eps_points, min_samples=cfg.dbscan_min_samples_points).fit(cluster_coordinates)
            clusters = dbscan_extractor(cluster_data, cluster_coordinates)
            centroids, intensity = centroids_calc(clusters)

        img = np.zeros_like(frame_cropped)

        if frame_count == cfg.frame_skip_init:
            centroids_init = centroids
            markerPtsZ = np.ones(centroids_init.shape[0]) * 20
            markerPts3D = np.column_stack((centroids_init, markerPtsZ))

        if frame_count >= cfg.frame_skip_init and centroids.size and centroids_init.size:
            try:
                reg = DeformableRegistration(**{'X': centroids * cfg.cpd_scale_factor, 'Y': centroids_init * cfg.cpd_scale_factor, 'low_rank': True, 'w': 0.8}, alpha=2.5, beta=2)
                tY, tfparam = reg.register()
                centroids_afterTransform = tY / cfg.cpd_scale_factor
                displacement2D = centroids_afterTransform - centroids_init

                for p, q in zip((centroids_init).astype(int), (centroids_afterTransform).astype(int)):
                    cv.line(img, tuple(p), tuple(q), (0, 255, 255), 1, cv.LINE_AA)
                    cv.circle(img, tuple(p), 1, (0, 255, 0), -1)
                    cv.imshow("CPD Displacement Field", img)

                interp = NearestNDInterpolator(centroids, intensity)
                marker_Z_intensity = interp(centroids_init)
                markerDisp3D = np.column_stack((displacement2D, -marker_Z_intensity))
                disp_vec_mag = np.linalg.norm(markerDisp3D, axis=1)
            except Exception as e:
                # sometimes registration fails if there are no markers; ignore and continue
                print(f"Registration step failed: {e}")

        # intensity interpolation
        if centroids.size:
            z_val = griddata(centroids, intensity, (xgrid, ygrid), method='linear', fill_value=0.0)
        else:
            z_val = np.zeros_like(xgrid)

        reg_z = gaussian_filter(interpolate_reg(z_val, 'cv-image'), sigma=cfg.gaussian_sigma)
        heatmap = cv.applyColorMap(reg_z, cv.COLORMAP_JET)

        timenow = time.time() - time_0
        fps_now = int(1.0 / max(1e-6, time.time() - time_start))
        cv.putText(heatmap, f"FPS={fps_now}", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
        cv.putText(heatmap, f"T={float(timenow):.2f}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

        # draw centroids with largest intensity and find maxima via gradient descent
        pack = np.column_stack((centroids, intensity)) if centroids.size else np.zeros((0, 3))
        if pack.size:
            gd = grad_descent(z_val, (pack[pack[:, 2] > 25])[:, 0:2], cropped_size)
        else:
            gd = grad_descent(z_val, np.zeros((0, 2)), cropped_size)

        if gd.iter_pts.size != 0:
            maxima_cls = DBSCAN(eps=cfg.maxima_eps, min_samples=cfg.maxima_min_samples).fit(gd.iter_pts)
            maxima_clusters = dbscan_extractor(maxima_cls, gd.iter_pts)
            maxima, _ = centroids_calc(maxima_clusters)

            maxima = maxima.astype(np.int16)
            maxima_circular_mask = np.zeros_like(z_val).astype(np.uint8)
            for x in maxima:
                cv.circle(maxima_circular_mask, x, 35, 255, -1)
                maxarr = cv.bitwise_and(reg_z.astype(np.uint8), maxima_circular_mask)
                pt = np.unravel_index(np.argmax(maxarr), maxarr.shape)
                cv.drawMarker(heatmap, [pt[1], pt[0]], (0, 255, 0), cv.MARKER_CROSS, 15, 1)
                cv.putText(heatmap, f"{z_val[pt]:.2f}", [pt[1] + 5, pt[0] + 15], cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv.LINE_AA)
                maxima_circular_mask = np.zeros_like(z_val).astype(np.uint8)

        cv.imshow('heatmap', heatmap)

        if frame_count > 0:
            color_cvt_frame = cv.cvtColor(grey_frame_corrected, cv.COLOR_GRAY2BGR)
            cluster_image = draw_centroid_cv(centroids, color_cvt_frame, color=(0, 0, 255), flip=False)
            cv.putText(cluster_image, f"Cluster Count={centroids.shape[0]}", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
            heatmap_output.write(heatmap)
            last_centroids = centroids
        else:
            last_centroids = centroids

        frame_count += 1

    cap.release()
    out_original.release()
    heatmap_output.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Online heatmap visualization for tactile sensor markers')
    parser.add_argument('--input', type=int, default=0, help='capture device index or file path')
    parser.add_argument('--orig-out', type=str, default='videos/output_original.mp4', help='path to save original grayscale capture')
    parser.add_argument('--heat-out', type=str, default='videos/output_from_online_cap.mp4', help='path to save heatmap video')
    args = parser.parse_args()

    cfg = Config()
    cfg.input_source = args.input
    cfg.output_original_path = args.orig_out
    cfg.output_heatmap_path = args.heat_out

    main(cfg)