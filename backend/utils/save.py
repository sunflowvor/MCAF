from PIL import Image
import numpy as np
import os
import json
import cv2


def polygons_to_mask(polygons, height, width):
    """将多边形转换为二值 mask"""
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        if not polygon:
            continue
        pts = np.array(polygon, dtype=np.int32)
        if pts.ndim == 2:
            pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
    return mask

def mask_to_polygons(mask):
    mask = mask.astype(np.uint8)  # 转为 0/1 图像
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        if contour.shape[0] >= 3:  # 至少三个点才构成多边形
            # flatten 并转为 list
            polygon = contour.squeeze().tolist()
            if isinstance(polygon[0], int):
                polygon = [polygon]  # 单个点特殊处理
            polygons.append(polygon)
    return polygons

def save_panoptic_label(selected_list, mask_attrs, masks, save_dir, image_name, THING_CLASSES):
    N, H, W = masks.shape
    label_img = np.zeros((H, W), dtype=np.uint32)

    assert N == len(selected_list)
    segments_info = []
    instance_counters = {}
    #{3: 'mining_truck', '3_instance_id': 7001, 2: 'car', '2_instance_id': 16001}
    for i in range(N):
        mask_id = selected_list[i]
        class_name = mask_attrs[mask_id]
        instance_id = mask_attrs[str(mask_id) + "_instance_id"]

        if instance_id > 1000:
            class_id = int(instance_id / 1000)
        else:
            class_id = int(instance_id)

        is_thing = class_name in THING_CLASSES

        mask = masks[i].astype(bool)
        label_img[mask] = instance_id

        polygons = mask_to_polygons(mask)

        segments_info.append({
            "id": int(mask_id),
            "instance_id": int(instance_id),
            "class_id": class_id,
            "category_name": class_name,
            "isthing": is_thing,
            "polygon": polygons
        })

    os.makedirs(save_dir, exist_ok=True)
    label_path = os.path.join(save_dir, f"{image_name}_gtFine_instanceIds.png")
    Image.fromarray(label_img.astype(np.uint32)).save(label_path)

    meta_path = os.path.join(save_dir, f"{image_name}_gtFine_panoptic.json")
    with open(meta_path, "w") as f:
        json.dump({"segments_info": segments_info}, f, indent=2, ensure_ascii=False)

    return label_path, meta_path