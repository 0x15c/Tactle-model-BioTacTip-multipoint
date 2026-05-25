"""
Compare low-resolution representations for sparse tactile marker images.

The important point is that ordinary resize preserves local averages, not marker
peak brightness. For sparse bright dots, this can make markers look dim or lost.
This script writes several alternatives so you can inspect which representation
is best for a coarse carrier-flow network.
"""

from pathlib import Path

import cv2
import numpy as np


IMAGE_PATH = Path(__file__).with_name("frame_000.jpg")
OUT_DIR = Path(__file__).with_name("outputs")
LOW_SIZE = (32, 32)  # OpenCV order: (width, height)

# Blob preprocessing before downsampling. Increase DILATE_ITERS if marker dots
# still disappear at low resolution.
THRESHOLD = 35
DILATE_KERNEL = 5
DILATE_ITERS = 2
BLUR_KERNEL = 7


def load_gray_float(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img.astype(np.float32) / 255.0


def save_gray(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.clip(img * 255.0, 0, 255).astype(np.uint8))


def normalize_for_view(img):
    img = img.astype(np.float32)
    lo = float(img.min())
    hi = float(img.max())
    if hi - lo < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return (img - lo) / (hi - lo)


def resize_area_mean(img, size):
    # Best default for downsampling natural images. Preserves local average.
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def resize_area_sum(img, size):
    # Preserves approximate integrated brightness instead of average brightness.
    # This can exceed 1.0 for sparse bright pixels, so use it as a float input
    # channel or normalize only for visualization.
    src_h, src_w = img.shape
    dst_w, dst_h = size
    area_ratio = (src_h * src_w) / float(dst_h * dst_w)
    return resize_area_mean(img, size) * area_ratio


def resize_max_pool(img, size):
    # Preserves whether a bright marker existed in each low-res bin. This is
    # often better than averaging for sparse marker localization.
    src_h, src_w = img.shape
    dst_w, dst_h = size
    y_edges = np.linspace(0, src_h, dst_h + 1).round().astype(np.int32)
    x_edges = np.linspace(0, src_w, dst_w + 1).round().astype(np.int32)

    pooled = np.zeros((dst_h, dst_w), dtype=np.float32)
    for yy in range(dst_h):
        y0, y1 = y_edges[yy], max(y_edges[yy + 1], y_edges[yy] + 1)
        for xx in range(dst_w):
            x0, x1 = x_edges[xx], max(x_edges[xx + 1], x_edges[xx] + 1)
            pooled[yy, xx] = img[y0:y1, x0:x1].max()
    return pooled


def make_blob_image(img):
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(img_u8, THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (DILATE_KERNEL, DILATE_KERNEL),
    )
    blobs = cv2.dilate(binary, kernel, iterations=DILATE_ITERS)
    blobs = cv2.GaussianBlur(blobs, (BLUR_KERNEL, BLUR_KERNEL), 0)
    return blobs.astype(np.float32) / 255.0


def add_label(img, label):
    view = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(view, (0, 0), (view.shape[1], 24), (0, 0, 0), thickness=-1)
    cv2.putText(
        view,
        label,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return view


def make_panel(items, display_size=(256, 256)):
    panels = []
    for label, img in items:
        view = cv2.resize(img, display_size, interpolation=cv2.INTER_NEAREST)
        panels.append(add_label(view, label))
    return np.hstack(panels)


def print_stats(name, img):
    print(
        f"{name:18s} "
        f"shape={img.shape} "
        f"min={img.min():.6f} "
        f"max={img.max():.6f} "
        f"mean={img.mean():.6f} "
        f"sum={img.sum():.2f}"
    )


def main():
    img = load_gray_float(IMAGE_PATH)

    area_mean = resize_area_mean(img, LOW_SIZE)
    area_sum = resize_area_sum(img, LOW_SIZE)
    max_pool = resize_max_pool(img, LOW_SIZE)

    blobs = make_blob_image(img)
    blobs_area = resize_area_mean(blobs, LOW_SIZE)
    blobs_max = resize_max_pool(blobs, LOW_SIZE)

    save_gray(OUT_DIR / "original.png", img)
    save_gray(OUT_DIR / "area_mean_64.png", area_mean)
    save_gray(OUT_DIR / "area_sum_64_display_normalized.png", normalize_for_view(area_sum))
    save_gray(OUT_DIR / "max_pool_64.png", max_pool)
    save_gray(OUT_DIR / "blobs_full_res.png", blobs)
    save_gray(OUT_DIR / "blobs_area_64.png", blobs_area)
    save_gray(OUT_DIR / "blobs_max_64.png", blobs_max)

    panel = make_panel(
        [
            ("original", img),
            ("area mean", area_mean),
            ("area sum view", normalize_for_view(area_sum)),
            ("max pool", max_pool),
            ("blob + area", blobs_area),
            ("blob + max", blobs_max),
        ]
    )
    cv2.imwrite(str(OUT_DIR / "resize_comparison_panel.png"), panel)

    print_stats("original", img)
    print_stats("area mean", area_mean)
    print_stats("area sum", area_sum)
    print_stats("max pool", max_pool)
    print_stats("blob + area", blobs_area)
    print_stats("blob + max", blobs_max)
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
