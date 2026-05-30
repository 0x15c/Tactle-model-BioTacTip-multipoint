import cv2
import numpy as np


def ensure_float01(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if image.size and image.max() > 1.0:
        image = image / 255.0
    return image


def max_pool_resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """
    Downsample a sparse marker image by taking the maximum value in each bin.

    size is OpenCV-style (width, height). The returned image is float32 in the
    same value scale as the input.
    """
    src_h, src_w = image.shape
    dst_w, dst_h = size
    y_edges = np.linspace(0, src_h, dst_h + 1).round().astype(np.int32)
    x_edges = np.linspace(0, src_w, dst_w + 1).round().astype(np.int32)

    pooled = np.zeros((dst_h, dst_w), dtype=np.float32)
    for yy in range(dst_h):
        y0 = y_edges[yy]
        y1 = max(y_edges[yy + 1], y0 + 1)
        for xx in range(dst_w):
            x0 = x_edges[xx]
            x1 = max(x_edges[xx + 1], x0 + 1)
            pooled[yy, xx] = image[y0:y1, x0:x1].max()
    return pooled


def preprocess_registration_image(
    image: np.ndarray,
    mode: str = "none",
    size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Prepare an image for TactMorph registration.

    mode:
        none    Keep the original image unless size is provided.
        area    Downsample with INTER_AREA.
        maxpool Downsample with spatial max pooling.
    """
    image = ensure_float01(image)
    mode = (mode or "none").lower()

    if mode == "none":
        if size is None:
            return image.astype(np.float32)
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR).astype(np.float32)

    if size is None:
        raise ValueError(f"Preprocess mode '{mode}' requires a target size.")

    if mode == "area":
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA).astype(np.float32)

    if mode == "maxpool":
        return max_pool_resize(image, size).astype(np.float32)

    raise ValueError(f"Unknown registration preprocessing mode: {mode}")


def upsample_flow_to_shape(flow: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Resize a low-resolution pixel flow to a target image shape.

    The vector values are scaled from low-res pixels into target-image pixels.
    """
    _, src_h, src_w = flow.shape
    dst_h, dst_w = target_shape
    if (src_h, src_w) == (dst_h, dst_w):
        return flow.astype(np.float32)

    scale_x = dst_w / float(src_w)
    scale_y = dst_h / float(src_h)
    flow_x = cv2.resize(flow[0], (dst_w, dst_h), interpolation=cv2.INTER_LINEAR) * scale_x
    flow_y = cv2.resize(flow[1], (dst_w, dst_h), interpolation=cv2.INTER_LINEAR) * scale_y
    return np.stack([flow_x, flow_y], axis=0).astype(np.float32)
