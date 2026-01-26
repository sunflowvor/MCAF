import os
import numpy as np
import json


import os
import numpy as np

def run_pointcloud_reproject(project_path):
    chunks_dir = os.path.join(project_path, "lidar_odometry", "chunks")

    out_pc_dir = os.path.join(project_path, "pointcloud_segmentation")
    out_box_dir = os.path.join(project_path, "DDD_boxes")

    os.makedirs(out_pc_dir, exist_ok=True)
    os.makedirs(out_box_dir, exist_ok=True)

    chunk_dirs = sorted(
        d for d in os.listdir(chunks_dir)
        if d.startswith("chunk_")
    )

    for chunk_name in chunk_dirs:
        print(f"[Chunk] {chunk_name}")

        chunk_path = os.path.join(chunks_dir, chunk_name)

        # ✅读取 frame.txt（每一帧对应原始 npy 文件名）
        frame_txt_path = os.path.join(chunk_path, "frame.txt")
        if not os.path.exists(frame_txt_path):
            raise FileNotFoundError(f"Missing frame.txt in {chunk_path}")

        with open(frame_txt_path, "r", encoding="utf-8") as f:
            frame_names = [line.strip() for line in f if line.strip()]

        poses = np.load(os.path.join(chunk_path, "lidar_poses.npy"))

        # ⚠️确保数量一致
        num_frames = poses.shape[0]
        if len(frame_names) != num_frames:
            raise ValueError(
                f"[Mismatch] {chunk_name}: poses={num_frames} but frame.txt={len(frame_names)}"
            )

        boxes_last = load_kitti_boxes(os.path.join(chunk_path, "000000.txt"))

        data = np.load(os.path.join(chunk_path, "global_map_with_meta.npy"))

        xyz_last = data[:, :3]
        cat = data[:, 3]
        inst = data[:, 4]
        color = data[:, 5:8]
        frame_ids = data[:, 8]  # chunk 内 frame id

        # ---------- 每一帧：保存点云 ----------
        for i in range(num_frames):
            mask = frame_ids == i
            if not mask.any():
                continue

            pts_last = xyz_last[mask]
            pts_h = np.hstack([pts_last, np.ones((len(pts_last), 1))])

            # chunk_last -> frame_i
            T = np.linalg.inv(poses[i])
            pts_i = (T @ pts_h.T).T[:, :3]

            labels = np.hstack([
                pts_i,
                cat[mask, None],
                inst[mask, None],
                color[mask]
            ])

            # ✅使用 frame.txt 中的原始文件名保存
            out_npy_name = frame_names[i]  # e.g. 1661922775.300000_labels.npy
            np.save(os.path.join(out_pc_dir, out_npy_name), labels)

        # ---------- boxes：保存 bbox ----------
        for i in range(num_frames):
            T = np.linalg.inv(poses[i])
            boxes_i = [transform_box(b, T) for b in boxes_last]

            # ✅bbox 文件名：1661922775.300000.txt
            npy_name = frame_names[i]
            base = npy_name.replace("_labels.npy", "")  # -> 1661922775.300000
            out_box_name = f"{base}.txt"

            save_kitti_boxes(
                boxes_i,
                os.path.join(out_box_dir, out_box_name)
            )

    return {
        "status": "ok",
        "frames": "use frame.txt names"
    }


# def run_pointcloud_reproject(project_path):
#     chunks_dir = os.path.join(
#         project_path,
#         "lidar_odometry",
#         "chunks"
#     )

#     out_pc_dir = os.path.join(
#         project_path,
#         "pointcloud_segmentation"
#     )
#     out_box_dir = os.path.join(
#         project_path,
#         "DDD_boxes"
#     )

#     os.makedirs(out_pc_dir, exist_ok=True)
#     os.makedirs(out_box_dir, exist_ok=True)

#     chunk_dirs = sorted(
#         d for d in os.listdir(chunks_dir)
#         if d.startswith("chunk_")
#     )

#     global_frame_offset = 0

#     for chunk_name in chunk_dirs:
#         print(f"[Chunk] {chunk_name}")

#         chunk_path = os.path.join(chunks_dir, chunk_name)

#         poses = np.load(os.path.join(chunk_path, "lidar_poses.npy"))
#         boxes_last = load_kitti_boxes(
#             os.path.join(chunk_path, "000000.txt")
#         )

#         data = np.load(
#             os.path.join(chunk_path, "global_map_with_meta.npy")
#         )

#         xyz_last = data[:, :3]
#         #cat = data[:, 3].astype(int)
#         #inst = data[:, 4].astype(int)
#         #frame_ids = data[:, 8].astype(int)  # chunk 内 frame id
#         cat = data[:, 3]
#         inst = data[:, 4]
#         color = data[:, 5:8]
#         frame_ids = data[:, 8] # chunk 内 frame id

#         num_frames = poses.shape[0]

#         # ---------- 每一帧 ----------
#         for i in range(num_frames):
#             mask = frame_ids == i
#             if not mask.any():
#                 continue

#             pts_last = xyz_last[mask]
#             pts_h = np.hstack([pts_last, np.ones((len(pts_last), 1))])

#             # chunk_last -> frame_i
#             T = np.linalg.inv(poses[i])
#             pts_i = (T @ pts_h.T).T[:, :3]

#             labels = np.hstack([
#                 pts_i,
#                 cat[mask, None],
#                 inst[mask, None],
#                 color[mask]
#             ])

#             frame_idx = global_frame_offset + i
#             np.save(
#                 os.path.join(
#                     out_pc_dir,
#                     f"{frame_idx:010d}_labels.npy"
#                 ),
#                 labels
#             )

#         # ---------- boxes ----------
#         for i in range(num_frames):
#             T = np.linalg.inv(poses[i])
#             boxes_i = [transform_box(b, T) for b in boxes_last]

#             frame_idx = global_frame_offset + i
#             save_kitti_boxes(
#                 boxes_i,
#                 os.path.join(
#                     out_box_dir,
#                     f"frame_{frame_idx:06d}.txt"
#                 )
#             )

#         global_frame_offset += num_frames

#     return {
#         "status": "ok",
#         "frames": global_frame_offset
#     }


# ======================
# 复用函数（与你原来一致）
# ======================
def load_kitti_boxes(txt_path):
    boxes = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            cate, x, y, z, l, w, h, yaw = parts
            boxes.append({
                "category": cate,
                "center": np.array([float(x), float(y), float(z)]),
                "size": (float(l), float(w), float(h)),
                "yaw": float(yaw)
            })
    return boxes


def transform_box(box, T):
    cate = box["category"]
    c_h = np.hstack([box["center"], 1.0])
    c_new = (T @ c_h)[:3]

    R = T[:3, :3]
    dir_vec = np.array([
        np.cos(box["yaw"]),
        np.sin(box["yaw"]),
        0.0
    ])
    dir_new = R @ dir_vec
    yaw_new = np.arctan2(dir_new[1], dir_new[0])

    return {
        "category": cate,
        "center": c_new,
        "size": box["size"],
        "yaw": yaw_new
    }


def save_kitti_boxes(boxes, path):
    with open(path, "w") as f:
        for b in boxes:
            cate = b["category"]
            x, y, z = b["center"]
            l, w, h = b["size"]
            yaw = b["yaw"]
            f.write(
                f"{cate} {x:.3f} {y:.3f} {z:.3f} "
                f"{l:.3f} {w:.3f} {h:.3f} {yaw:.6f}\n"
            )
