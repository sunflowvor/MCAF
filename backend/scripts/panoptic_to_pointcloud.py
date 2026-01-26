# backend/scripts/panoptic_to_pointcloud.py

import os
import re
import json
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw


def run_panoptic_to_pointcloud(
    panoptic_dir,
    lidar_dir,
    image_dir,
    output_dir,
    K,
    T_LIDAR_TO_CAM,
    width,
    height,
    category_base_colors,
    thing_class_ids,
    is_kitti
):
    os.makedirs(output_dir, exist_ok=True)

    def normalize_polygons(polys):
        """
        输入可能是：
        - [[x,y], ...]                    (单 polygon)
        - [ [[x,y],...], [[x,y],...], ... ]  (多个 polygon)
        - 更深层嵌套
        输出：
        - [ [(x,y),...], [(x,y),...], ... ]
        """
        if polys is None:
            return []

        # 如果直接就是 [[x,y],...]
        if (
            isinstance(polys, list) and len(polys) > 0
            and isinstance(polys[0], (list, tuple)) and len(polys[0]) == 2
            and all(isinstance(v, (int, float, np.integer, np.floating)) for v in polys[0])
        ):
            return [ [(int(x), int(y)) for x, y in polys] ]

        # 如果是多个 polygon: [poly0, poly1, ...]
        out = []
        if isinstance(polys, list):
            for p in polys:
                pts = normalize_polygon(p)
                if len(pts) >= 3:
                    out.append(pts)
        return out

    def panoptic_color(category_id, instance_id):
        base = np.array(category_base_colors.get(category_id, (255,255,255)))
        if category_id not in thing_class_ids or instance_id == 0:
            return base.astype(np.uint8)
        rng = np.random.RandomState(category_id * 1000 + instance_id)
        noise = rng.randint(-35, 35, 3)
        return np.clip(base + noise, 0, 255).astype(np.uint8)

    def extract_timestamp(name):
        m = re.match(r"(\d+\.\d+)_gtFine_panoptic\.json", name)
        return m.group(1) if m else None

    def extract_timestamp_kitti(name):
        # 修改点：将 (\d+\.\d+) 改为 (\d+)，以匹配纯数字文件名
        m = re.match(r"(\d+)_gtFine_panoptic\.json", name)
        return m.group(1) if m else None

    def normalize_polygon(poly):
        """
        支持各种嵌套格式：
        - [[x,y], ...]
        - [[[x,y], ...]]
        - [[[[x,y], ...]]]
        最终输出: [(x,y), (x,y), ...]
        """
        # 一直剥外层，直到元素看起来像 [x, y]
        while isinstance(poly, list) and len(poly) > 0:
            first = poly[0]
            # 目标形态：first 是 [number, number]
            if isinstance(first, (list, tuple)) and len(first) == 2 and all(
                isinstance(v, (int, float, np.integer, np.floating)) for v in first
            ):
                break
            # 否则继续降一层
            poly = first

        # 现在 poly 应该是 [[x,y], ...]
        pts = []
        for p in poly:
            # 有些数据点会是 [[x,y]]，再剥一次
            while isinstance(p, list) and len(p) == 1 and isinstance(p[0], list):
                p = p[0]
            if not (isinstance(p, (list, tuple)) and len(p) == 2):
                continue
            x, y = p
            pts.append((int(x), int(y)))

        return pts

    json_files = sorted(
        f for f in os.listdir(panoptic_dir)
        if f.endswith("_gtFine_panoptic.json")
    )
    for fname in json_files:
        if is_kitti:
            ts = extract_timestamp_kitti(fname)
        else:
            ts = extract_timestamp(fname)
        if ts is None:
            continue

        pcd_path = os.path.join(lidar_dir, f"{ts}.pcd")
        if not os.path.exists(pcd_path):
            print(f"[Skip] missing {pcd_path}")
            continue

        print(f"[PC] Processing {ts}")

        # ---------- 读点云 ----------
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        N = points.shape[0]

        point_cat = np.zeros(N, dtype=np.int16)
        point_inst = np.zeros(N, dtype=np.int16)
        colors = np.zeros((N,3), dtype=np.uint8)

        # ---------- 投影 ----------
        pts_h = np.hstack([points, np.ones((N,1))])
        cam = (T_LIDAR_TO_CAM @ pts_h.T).T
        Z = cam[:,2]
        valid = Z > 0

        X, Y, Z = cam[valid,0], cam[valid,1], cam[valid,2]
        u = (K[0,0]*X/Z + K[0,2]).astype(np.int32)
        v = (K[1,1]*Y/Z + K[1,2]).astype(np.int32)

        valid_idx = np.where(valid)[0]

        # ---------- 读 JSON ----------
        with open(os.path.join(panoptic_dir, fname), "r") as f:
            pano = json.load(f)
        instance_map = {}
        instance_next = {}  # class_id -> next local id
        # 需要一张“画布”大小（从 config 或 image 推导更好）
        # 这里假设你知道 image 尺寸，或者从 config 取
        # 示例：1920x1080
        H, W = height, width

        # ---------- 读原始图像 ----------
        img_path = os.path.join(image_dir, f"{ts}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, f"{ts}.png")

        if not os.path.exists(img_path):
            print(f"[Skip RGB] missing image for {ts}")
            img = None
        else:
            img = np.array(Image.open(img_path).convert("RGB"))
            img_h, img_w, _ = img.shape


        # ---------- 遍历每个 segment ----------
        # for seg in pano["segments_info"]:
        #     class_id = int(seg["class_id"])
        #     isthing = seg.get("isthing", False)
        #     instance_id = seg.get("instance_id", seg["id"]) if isthing else 0
        for seg in pano["segments_info"]:
            class_id = int(seg["class_id"])
            isthing = seg.get("isthing", False)

            raw_instance_id = seg.get("instance_id", seg["id"]) if isthing else 0

            if not isthing:
                instance_id = 0
            else:
                key = (class_id, int(raw_instance_id))
                if key not in instance_map:
                    instance_next.setdefault(class_id, 1)
                    instance_map[key] = instance_next[class_id]
                    instance_next[class_id] += 1
                instance_id = instance_map[key]

    # 后面保持你的 polygon->mask 投影不变



            # polygon → mask
            mask = Image.new("L", (W, H), 0)
            draw = ImageDraw.Draw(mask)

            #print(seg["polygon"])

            # poly = normalize_polygon(seg["polygon"])
            # if len(poly) < 3:
            #     continue
            # draw.polygon(poly, outline=1, fill=1)
            polys = normalize_polygons(seg["polygon"])
            if len(polys) == 0:
                continue

            for poly in polys:
                draw.polygon(poly, outline=1, fill=1)



            mask = np.array(mask, dtype=bool)

            # 判断投影点是否落在 mask 中
            for i_img, uu, vv in zip(valid_idx, u, v):
                if 0 <= uu < W and 0 <= vv < H and mask[vv, uu]:
                    point_cat[i_img] = class_id
                    point_inst[i_img] = instance_id
                    colors[i_img] = panoptic_color(class_id, instance_id)
        if img is not None:
            for i_img, uu, vv in zip(valid_idx, u, v):
                if 0 <= uu < img_w and 0 <= vv < img_h:
                    colors[i_img] = img[vv, uu]


        # ---------- 保存 ----------
        # label_data = np.hstack([
        #     points,
        #     point_cat[:,None],
        #     point_inst[:,None]
        # ])

        #np.save(os.path.join(output_dir, f"{ts}_labels.npy"), label_data)

        #pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        # o3d.visualization.draw_geometries([pcd])

        #color_dir = os.path.join(output_dir, "..", "pointcloud_segmentation_with_color")
        #os.makedirs(color_dir, exist_ok=True)

        rgb_data = np.hstack([
            points,
            point_cat[:,None],
            point_inst[:,None],
            colors.astype(np.uint8)
        ])

        np.save(os.path.join(output_dir, f"{ts}_labels.npy"), rgb_data)
        #pcd.colors = o3d.utility.Vector3dVector(rgb_colors / 255.0)
        #o3d.visualization.draw_geometries([pcd])

    return {"status": "ok", "frames": len(json_files)}
