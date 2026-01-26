import numpy as np
import cv2

def project_color_to_points(
    pts_lidar,          # (N,3)
    image_bgr,          # HxWx3
    K,                  # (3,3)
    T_lidar_to_cam      # (4,4)
):
    N = pts_lidar.shape[0]

    colors = np.zeros((N, 3), dtype=np.float32)

    # --- LiDAR -> Camera ---
    pts_h = np.hstack([pts_lidar, np.ones((N, 1))])  # (N,4)
    pts_cam = (T_lidar_to_cam @ pts_h.T).T           # (N,4)

    X = pts_cam[:, 0]
    Y = pts_cam[:, 1]
    Z = pts_cam[:, 2]

    valid = Z > 0.1
    idx = np.where(valid)[0]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = (fx * X[valid] / Z[valid] + cx).astype(np.int32)
    v = (fy * Y[valid] / Z[valid] + cy).astype(np.int32)

    H, W = image_bgr.shape[:2]
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    idx = idx[inside]
    u = u[inside]
    v = v[inside]

    # BGR -> RGB
    rgb = image_bgr[v, u][:, ::-1] / 255.0

    colors[idx] = rgb
    return colors
