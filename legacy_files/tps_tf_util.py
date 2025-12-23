import numpy as np
def tps_kernel_2d(r):
    """Thin plate spline kernel in 2D: φ(r) = r^2 log r, with φ(0)=0."""
    out = np.zeros_like(r)
    mask = r > 0
    rm = r[mask]
    out[mask] = (rm**2) * np.log(rm)
    return out

def tps_transform(points, a, v, control_pts):
    """
    Apply 2D Thin Plate Spline transformation to multiple points.

    Args:
        points : (N, 2) ndarray
            Input coordinates.
        a : (3, 2) ndarray
            Affine coefficients.
        v : (k-3, 2) ndarray
            Non-rigid spline weights.
        control_pts : (k, 2) ndarray
            Control points (only first k-3 used for warping).

    Returns:
        transformed : (N, 2) ndarray
            Transformed coordinates.
    """
    points = np.asarray(points)  # (N, 2)
    N = points.shape[0]

    # affine part: [1, x, y] @ a
    basis = np.hstack([np.ones((N, 1)), points])  # (N, 3)
    affine_part = basis @ a  # (N, 2)

    # spline warp part
    M = v.shape[0]  # effective control points
    C = control_pts[:M, :]  # (M, 2)

    # pairwise distances (N, M)
    dists = np.linalg.norm(points[:, None, :] - C[None, :, :], axis=2)
    phi = tps_kernel_2d(dists)  # (N, M)

    spline_part = phi @ v  # (N, 2)

    return affine_part + spline_part