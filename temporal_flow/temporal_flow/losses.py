from __future__ import annotations

import torch
import torch.nn.functional as F


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps).mean()


def photometric_loss(warped: torch.Tensor, target: torch.Tensor, kind: str = "charbonnier") -> torch.Tensor:
    if kind == "l1":
        return F.l1_loss(warped, target)
    if kind == "mse":
        return F.mse_loss(warped, target)
    if kind == "charbonnier":
        return charbonnier(warped - target)
    raise ValueError(f"Unknown photometric loss kind: {kind}")


def spatial_smoothness_loss(flow: torch.Tensor, robust: bool = True) -> torch.Tensor:
    dx = flow[..., :, 1:] - flow[..., :, :-1]
    dy = flow[..., 1:, :] - flow[..., :-1, :]
    if robust:
        return charbonnier(dx) + charbonnier(dy)
    return (dx * dx).mean() + (dy * dy).mean()


def temporal_update_loss(delta_flow: torch.Tensor, robust: bool = True) -> torch.Tensor:
    if robust:
        return charbonnier(delta_flow)
    return (delta_flow * delta_flow).mean()


def flow_magnitude(flow: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((flow * flow).sum(dim=1, keepdim=True).clamp_min(1e-12))
