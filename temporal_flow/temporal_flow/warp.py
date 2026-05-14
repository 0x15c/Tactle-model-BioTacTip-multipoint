from __future__ import annotations

import torch
import torch.nn.functional as F


def make_base_grid(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
    """Return a pixel-coordinate grid with shape BxHxWx2, order [x, y]."""
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack([xs, ys], dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    return grid


def normalize_grid(pixel_grid: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert pixel coordinates to grid_sample coordinates in [-1, 1]."""
    x = pixel_grid[..., 0]
    y = pixel_grid[..., 1]
    x_norm = 2.0 * x / max(width - 1, 1) - 1.0
    y_norm = 2.0 * y / max(height - 1, 1) - 1.0
    return torch.stack([x_norm, y_norm], dim=-1)


def warp_image(image: torch.Tensor, flow: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Warp image by a dense reference-to-current flow field.

    image: BxCxHxW, usually reference image I0.
    flow:  Bx2xHxW in pixel units. flow[:,0] is x displacement, flow[:,1] is y displacement.

    The convention used here is forward physical displacement sampled using backward warping:
    I0_warped(x + u(x)) should match It. In practice, grid_sample samples source pixels from
    coordinates x + flow(x). This convention is sufficient for self-supervised regularization, but
    keep it consistent when visualising/sampling marker positions.
    """
    if image.dim() != 4 or flow.dim() != 4:
        raise ValueError("image and flow must be BCHW tensors")
    b, _, h, w = image.shape
    base = make_base_grid(b, h, w, image.device, image.dtype)
    sample_grid = base + flow.permute(0, 2, 3, 1)
    sample_grid = normalize_grid(sample_grid, h, w)
    return F.grid_sample(image, sample_grid, mode=mode, padding_mode="border", align_corners=True)
