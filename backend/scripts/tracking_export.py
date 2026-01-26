import os
import json
import numpy as np
from PIL import Image


def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    return [int(x_min), int(y_min), int(w), int(h)]


def coco_from_instance_only(
    instance_dir,
    output_json_path,
    width,
    height,
    id_to_name=None
):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "thing"}
        ]
    }

    category_ids = set()
    
    if id_to_name is None:
        id_to_name = {}

    files = sorted([
        f for f in os.listdir(instance_dir)
        if f.endswith("_gtFine_instanceIds.png")
    ])

    ann_id = 1

    for img_id, filename in enumerate(files, start=1):
        base = filename.replace("_gtFine_instanceIds.png", "")
        path = os.path.join(instance_dir, filename)

        inst = np.array(Image.open(path))

        coco["images"].append({
            "id": img_id,
            "file_name": base + ".jpg",
            "width": width,
            "height": height
        })

        inst_ids = np.unique(inst)
        inst_ids = inst_ids[inst_ids > 0]

        for tid in inst_ids:
            mask = (inst == tid).astype(np.uint8)
            bbox = mask_to_bbox(mask)
            if bbox is None:
                continue

            area = int(mask.sum())

            category_id = int(tid) // 1000
            category_ids.add(category_id)

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "track_id": int(tid)
            })
            ann_id += 1

        coco["categories"] = [
            {"id": cid, "name": id_to_name.get(cid, f"class_{cid}")}
            for cid in sorted(category_ids)
        ]


    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    return coco   # ✅ 新增：返回 coco

import cv2
import random

# -------------------------------------
# 给每个 track_id 分配固定颜色
# -------------------------------------
def get_color(track_id):
    random.seed(track_id)
    return (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255)
    )

# -------------------------------------
# 从 COCO tracking JSON 可视化
# -------------------------------------
def visualize_tracking_from_coco(
    coco,
    image_root,
    output_dir
):
    os.makedirs(output_dir, exist_ok=True)

    # image_id -> annotations
    ann_map = {}
    for ann in coco["annotations"]:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    for img_info in coco["images"]:
        img_id = img_info["id"]
        base = os.path.splitext(img_info["file_name"])[0]

        # 这里用你现有的 panoptic 可视化图
        img_path = os.path.join(
            image_root,
            base + ".jpg"
        )

        if not os.path.exists(img_path):
            print(f"⚠ image not found: {img_path}")
            continue

        img = cv2.imread(img_path)

        for ann in ann_map.get(img_id, []):
            x, y, w, h = map(int, ann["bbox"])
            track_id = ann["track_id"]

            color = get_color(track_id)

            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv2.putText(
                img,
                f"ID {track_id}",
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        out_path = os.path.join(output_dir, base + ".png")
        cv2.imwrite(out_path, img)
        print(f"✔ tracking vis saved: {out_path}")

def export_tracking_with_vis(
    instance_dir,
    original_vis_dir,
    output_dir,
    width,
    height,
    id_to_name
):
    output_json = os.path.join(
        output_dir,
        "coco_tracking_from_instance.json"
    )

    coco = coco_from_instance_only(
        instance_dir=instance_dir,
        output_json_path=output_json,
        width=width,
        height=height,
        id_to_name = id_to_name
    )

    visualize_tracking_from_coco(
        coco=coco,
        image_root=original_vis_dir,
        output_dir=os.path.join(output_dir, "vis")
    )

    return output_json
