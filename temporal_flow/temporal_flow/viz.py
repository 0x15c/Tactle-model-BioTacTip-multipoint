from __future__ import annotations

import cv2
import numpy as np
import torch


def tensor_to_gray_u8(x: torch.Tensor) -> np.ndarray:
    if x.dim() == 4:
        x = x[0]
    if x.dim() == 3:
        x = x[0]
    arr = x.detach().cpu().float().clamp(0, 1).numpy()
    return (arr * 255.0).astype(np.uint8)


def flow_to_hsv(flow: torch.Tensor, max_mag: float | None = None) -> np.ndarray:
    if flow.dim() == 4:
        flow = flow[0]
    f = flow.detach().cpu().float().numpy()
    fx, fy = f[0], f[1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=False)
    if max_mag is None:
        max_mag = float(np.percentile(mag, 99) + 1e-6)
    hsv = np.zeros((*mag.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((ang / (2 * np.pi)) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_flow_quiver(gray: np.ndarray, flow: torch.Tensor, step: int = 20, scale: float = 1.0) -> np.ndarray:
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()
    if flow.dim() == 4:
        flow = flow[0]
    f = flow.detach().cpu().float().numpy()
    h, w = f.shape[1:]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx, dy = f[0, y, x] * scale, f[1, y, x] * scale
            p1 = (int(x), int(y))
            p2 = (int(x + dx), int(y + dy))
            cv2.arrowedLine(out, p1, p2, (0, 255, 255), 1, tipLength=0.3)
    return out


def make_demo_panel(ref_u8: np.ndarray, cur_u8: np.ndarray, warped_u8: np.ndarray, flow: torch.Tensor) -> np.ndarray:
    flow_bgr = flow_to_hsv(flow)
    ref_bgr = cv2.cvtColor(ref_u8, cv2.COLOR_GRAY2BGR)
    cur_bgr = cv2.cvtColor(cur_u8, cv2.COLOR_GRAY2BGR)
    warped_bgr = cv2.cvtColor(warped_u8, cv2.COLOR_GRAY2BGR)
    err = cv2.absdiff(cur_u8, warped_u8)
    err_bgr = cv2.applyColorMap(err, cv2.COLORMAP_JET)
    top = np.hstack([ref_bgr, cur_bgr])
    bot = np.hstack([warped_bgr, flow_bgr])
    panel = np.vstack([top, bot])
    cv2.putText(panel, "ref", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
    cv2.putText(panel, "current", (ref_u8.shape[1] + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
    cv2.putText(panel, "warped ref", (10, ref_u8.shape[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
    cv2.putText(panel, "flow HSV", (ref_u8.shape[1] + 10, ref_u8.shape[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
    return panel
