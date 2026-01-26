# backend/scripts/lidar_odometry.py
import os
import numpy as np
import open3d as o3d

import cv2
from scripts.project_color_from_image import project_color_to_points

# =======================
# 颜色表
# =======================
CATEGORY_BASE_COLORS = {
    1:(135,206,235), 2:(128,64,128), 3:(160,160,160), 4:(110,110,110),
    5:(244,35,232), 6:(107,142,35), 7:(180,0,0), 8:(150,120,90),
    9:(120,120,120), 10:(70,130,180), 11:(255,140,0), 12:(255,215,0),
    13:(255,99,71), 14:(178,34,34), 15:(220,20,60), 16:(255,0,0),
    17:(139,69,19), 18:(0,128,0), 19:(105,105,105), 20:(255,165,0),
    21:(169,169,169), 22:(255,255,0), 23:(184,134,11), 24:(0,0,255),
    25:(255,255,255), 26:(160,82,45), 27:(139,69,19), 28:(0,191,255),
    29:(128,0,128), 30:(0,0,142), 31:(0,64,128), 32:(0,60,100)
}

def color_from_cat_inst(cat_id, inst_id):
    base = np.array(
        CATEGORY_BASE_COLORS.get(int(cat_id), (200, 200, 200)),
        dtype=np.int32
    )

    if inst_id == 0:
        return base / 255.0

    seed = int(cat_id) * 1000 + int(inst_id)
    rng = np.random.RandomState(seed)
    noise = rng.randint(-40, 40, 3)
    color = np.clip(base + noise, 0, 255)
    return color / 255.0

# =======================
# 工具函数
# =======================
def copy_pointcloud(pcd):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    if pcd.has_colors():
        p.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
    if pcd.has_normals():
        p.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals))
    return p

# =======================
# 加载 npy 点云 + 元数据
# =======================
def load_pcd_from_npy(
    npy_path
):
    data = np.load(npy_path)

    pts = data[:, :3]
    cat_ids = data[:, 3].astype(int)
    inst_ids = data[:, 4].astype(int)
    colors = data[:, 5:]
    use_cat = False
    if use_cat:
        colors = np.zeros((len(pts), 3))
        for i in range(len(pts)):
            colors[i] = color_from_cat_inst(cat_ids[i], inst_ids[i])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    meta = {
        "cat": cat_ids,
        "inst": inst_ids
    }
    return pcd, meta

# =======================
# 特征提取
# =======================
def extract_features(pcd, k=20, edge_thresh=0.12, plane_thresh=0.01):
    pts = np.asarray(pcd.points)
    tree = o3d.geometry.KDTreeFlann(pcd)

    curvature = np.zeros(len(pts))

    for i in range(len(pts)):
        _, idx, _ = tree.search_knn_vector_3d(pts[i], k)
        neighbors = pts[idx]
        centroid = neighbors.mean(axis=0)
        diff = neighbors - centroid
        cov = diff.T @ diff / k
        eigvals = np.linalg.eigvalsh(cov)
        curvature[i] = eigvals[0] / (eigvals.sum() + 1e-6)

    edge_idx = curvature > edge_thresh
    plane_idx = curvature < plane_thresh

    edge = pcd.select_by_index(np.where(edge_idx)[0])
    plane = pcd.select_by_index(np.where(plane_idx)[0])

    return edge, plane

# =======================
# ICP 配准
# =======================
def register_features(src_edge, src_plane, tgt_edge, tgt_plane, init=np.eye(4)):
    src = src_edge + src_plane
    tgt = tgt_edge + tgt_plane

    if len(src.points) < 50 or len(tgt.points) < 50:
        return init

    src.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
    )
    tgt.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
    )

    result = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=1.5,
        init=init,
        estimation_method=o3d.pipelines.registration.
            TransformationEstimationPointToPlane()
    )
    return result.transformation

# =======================
# LiDAR 里程计
# =======================
def lidar_odometry(npy_dir, files, image_dir, K, T_LIDAR_TO_CAM):
    poses = [np.eye(4)]
    pcd_cache = []
    meta_cache = []

    prev_edge, prev_plane = None, None
    T_world = np.eye(4)

    for i, f in enumerate(files):
        npy_path = os.path.join(npy_dir, f)

        pcd, meta = load_pcd_from_npy(
            npy_path
        )

        edge, plane = extract_features(pcd)

        if prev_edge is not None:
            T = register_features(edge, plane, prev_edge, prev_plane)
            T_world = T_world @ T
            poses.append(T_world.copy())

        prev_edge, prev_plane = edge, plane
        pcd_cache.append(pcd)
        meta_cache.append(meta)

        print(f"[Frame {i}] processed")

    return poses, pcd_cache, meta_cache


# =======================
# 构建融合点云 + 全局元数据
# =======================
def build_map_to_last_frame(pcd_list, meta_list, poses):
    T_last_inv = np.linalg.inv(poses[-1])

    global_map = o3d.geometry.PointCloud()
    global_meta = {"frame": [], "cat": [], "inst": []}

    for frame_id, (pcd, meta, T) in enumerate(zip(pcd_list, meta_list, poses)):
        pcd_c = copy_pointcloud(pcd)
        pcd_c.transform(T_last_inv @ T)

        global_map += pcd_c

        n = len(pcd.points)
        global_meta["frame"].extend([frame_id] * n)
        global_meta["cat"].extend(meta["cat"])
        global_meta["inst"].extend(meta["inst"])

    for k in global_meta:
        global_meta[k] = np.asarray(global_meta[k])

    return global_map, global_meta


def save_global_map_and_meta(global_map, global_meta, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # ========= 1. 保存点云 =========
    pcd_path = os.path.join(out_dir, "global_map.pcd")
    o3d.io.write_point_cloud(pcd_path, global_map)
    print(f"[Saved] PointCloud -> {pcd_path}")

    # ========= 2. 保存元数据 =========
    # meta_path = os.path.join(out_dir, "global_meta.npz")
    # np.savez(
    #     meta_path,
    #     frame=global_meta["frame"],
    #     cat=global_meta["cat"],
    #     inst=global_meta["inst"]
    # )
    # print(f"[Saved] Meta -> {meta_path}")
    # ========= 3. 可选：保存为一个大 npy（方便训练） =========
    pts = np.asarray(global_map.points)
    cols = np.asarray(global_map.colors)

    merged = np.concatenate([
        pts,
        global_meta["cat"][:, None],
        global_meta["inst"][:, None],
        cols,
        global_meta["frame"][:, None],
    ], axis=1)

    merged_path = os.path.join(out_dir, "global_map_with_meta.npy")
    np.save(merged_path, merged)
    print(f"[Saved] Merged npy -> {merged_path}")

# def save_chunked_maps(
#     pcd_list,
#     meta_list,
#     poses,
#     out_dir,
#     chunk_size=20
# ):
#     chunks_dir = os.path.join(out_dir, "chunks")
#     os.makedirs(chunks_dir, exist_ok=True)

#     num_frames = len(pcd_list)

#     for start in range(0, num_frames, chunk_size):
#         end = min(start + chunk_size, num_frames)

#         chunk_name = f"chunk_{start:03d}_{end-1:03d}"
#         chunk_out = os.path.join(chunks_dir, chunk_name)
#         os.makedirs(chunk_out, exist_ok=True)

#         print(f"[Chunk] {chunk_name}")

#         # ⚠️ 切片
#         chunk_pcds  = pcd_list[start:end]
#         chunk_metas = meta_list[start:end]
#         chunk_poses = poses[start:end]

#         # === 以该 chunk 最后一帧为参考 ===
#         global_map, global_meta = build_map_to_last_frame(
#             chunk_pcds,
#             chunk_metas,
#             chunk_poses
#         )

#         # === 复用你已有的保存逻辑 ===
#         save_global_map_and_meta(global_map, global_meta, chunk_out)

#         # 保存该 chunk 的 poses（相对于 chunk 最后一帧）
#         np.save(
#             os.path.join(chunk_out, "lidar_poses.npy"),
#             np.stack(chunk_poses)
#         )

def save_chunked_maps(
    pcd_list,
    meta_list,
    poses,
    out_dir,
    files,              # ✅新增：传入 npy 文件名列表
    chunk_size=20
):
    chunks_dir = os.path.join(out_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    num_frames = len(pcd_list)

    for start in range(0, num_frames, chunk_size):
        end = min(start + chunk_size, num_frames)

        chunk_name = f"chunk_{start:03d}_{end-1:03d}"
        chunk_out = os.path.join(chunks_dir, chunk_name)
        os.makedirs(chunk_out, exist_ok=True)

        print(f"[Chunk] {chunk_name}")

        # ✅这个 chunk 对应的 npy 文件名
        chunk_files = files[start:end]

        # ✅写入 frame.txt（每行一个文件名）
        frame_txt_path = os.path.join(chunk_out, "frame.txt")
        with open(frame_txt_path, "w", encoding="utf-8") as f:
            for name in chunk_files:
                f.write(name + "\n")
        print(f"[Saved] frame.txt -> {frame_txt_path}")

        # ⚠️ 切片
        chunk_pcds  = pcd_list[start:end]
        chunk_metas = meta_list[start:end]
        chunk_poses = poses[start:end]

        # === 以该 chunk 最后一帧为参考 ===
        global_map, global_meta = build_map_to_last_frame(
            chunk_pcds,
            chunk_metas,
            chunk_poses
        )

        # === 保存逻辑 ===
        save_global_map_and_meta(global_map, global_meta, chunk_out)

        # 保存该 chunk 的 poses
        np.save(
            os.path.join(chunk_out, "lidar_poses.npy"),
            np.stack(chunk_poses)
        )



def run_lidar_odometry(
    npy_dir,
    output_dir,
    image_dir,
    K,
    T,
    max_frames=None,
    chunk_size=20
):

    files = sorted(f for f in os.listdir(npy_dir) if f.endswith("_labels.npy"))
    if max_frames:
        files = files[:max_frames]

    poses, pcd_list, meta_list = lidar_odometry(npy_dir, files, image_dir, K, T)

    global_out = os.path.join(output_dir, "global_merged")
    os.makedirs(global_out, exist_ok=True)

    global_map, global_meta = build_map_to_last_frame(
        pcd_list, meta_list, poses
    )

    save_global_map_and_meta(global_map, global_meta, global_out)

    np.save(
        os.path.join(global_out, "lidar_poses.npy"),
        np.stack(poses)
    )

    save_chunked_maps(
        pcd_list,
        meta_list,
        poses,
        output_dir,
        files=files, 
        chunk_size=chunk_size
    )

    return {
        "status": "ok",
        "frames": len(files),
        "output_dir": output_dir,
        "chunk_size": chunk_size
    }
