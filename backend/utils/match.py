import numpy as np
import cv2
from skimage.measure import regionprops, label
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from PIL import Image
from ultralytics import YOLO
import torch
from utils.save import polygons_to_mask
import json

def get_mask_center_and_area(mask):
    labeled = label(mask)
    props = regionprops(labeled)
    if not props:
        return (0, 0), 0
    center = props[0].centroid  # (y, x)
    area = props[0].area
    return (int(center[1]), int(center[0])), area  # return as (x, y)

# ------- 基础度量 -------
def mask_iou(A: np.ndarray, B: np.ndarray, eps: float = 1e-8) -> float:
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    return inter / (union + eps)

def _centers_areas(masks):
    centers, areas = [], []
    for m in masks:
        c, a = get_mask_center_and_area(m)
        centers.append(c)
        areas.append(a)
    return np.vstack(centers), np.asarray(areas, dtype=float)

# ------- 构建代价矩阵 -------
def _build_cost_matrix(
    attr_masks,
    sam_masks,
    img_shape=None,
    w_iou: float = 0.6,
    w_center: float = 0.3,
    w_area: float = 0.1,
    min_iou: float = 0.0,
    max_center_frac: float = 0.75,
    huge_cost: float = 1e6,
):
    """
    cost[i, j] = w_iou*(1 - IoU_ij) + w_center*(center_dist_ij/diag) + w_area*(rel_area_diff_ij)
    不满足门限（IoU过低或中心距离过大）的配对用 huge_cost 屏蔽。
    """
    A, S = len(attr_masks), len(sam_masks)
    cost = np.full((A, S), huge_cost, dtype=float)

    attr_centers, attr_areas = _centers_areas(attr_masks)
    sam_centers,  sam_areas  = _centers_areas(sam_masks)

    # 归一化用图像对角线；若未显式给出，则用掩码尺寸推断
    if img_shape is None:
        # 从任意非空掩码推断形状
        H, W = attr_masks[0].shape if len(attr_masks) else sam_masks[0].shape
    else:
        H, W = img_shape[:2]
    diag = float(np.hypot(H, W))

    # 中心距离（A x S）
    center_d = cdist(attr_centers, sam_centers) / (diag + 1e-8)

    # 逐对计算 IoU / 面积相对差
    for i in range(A):
        for j in range(S):
            # 门控：中心相距过远，直接略过
            if center_d[i, j] > max_center_frac:
                continue

            iou = mask_iou(attr_masks[i], sam_masks[j])
            if iou < min_iou:
                continue

            # 相对面积差（对尺度鲁棒）
            a_i, a_j = attr_areas[i], sam_areas[j]
            area_term = abs(a_i - a_j) / (max(a_i, a_j) + 1e-8)

            # 线性加权代价（越小越好）
            cost[i, j] = w_iou * (1.0 - iou) + w_center * center_d[i, j] + w_area * area_term

    return cost

# ------- 主函数：一一匹配（全局最优），未通过门限的返回 -1 -------
def match_masks(attr_masks, sam_masks,
                img_shape=None,
                w_iou=0.6, w_center=0.3, w_area=0.1,
                min_iou=0.0, max_center_frac=0.75,
                huge_cost=1e6):
    """
    返回：list[int]，长度=len(attr_masks)，每个元素是匹配到的 sam_masks 索引；若未匹配则为 -1。
    """
    if len(attr_masks) == 0 or len(sam_masks) == 0:
        return [-1] * len(attr_masks)

    cost = _build_cost_matrix(
        attr_masks, sam_masks, img_shape,
        w_iou=w_iou, w_center=w_center, w_area=w_area,
        min_iou=min_iou, max_center_frac=max_center_frac, huge_cost=huge_cost
    )

    # 匈牙利算法：全局最优一一匹配
    row_ind, col_ind = linear_sum_assignment(cost)

    # 默认 -1 表示未匹配（比如该行只有 huge_cost）
    matched = [-1] * len(attr_masks)
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < huge_cost * 0.1:  # 只接受未被门控屏蔽的有效成本
            matched[r] = int(c)

    return matched

# def match_masks(attr_masks, sam_masks):
#     matched = []

#     attr_centers = []
#     attr_areas = []

#     for attr_mask in attr_masks:
#         center, area = get_mask_center_and_area(attr_mask)
#         attr_centers.append(center)
#         attr_areas.append(area)

#     sam_centers = []
#     sam_areas = []
#     for sam_mask in sam_masks:
#         center, area = get_mask_center_and_area(sam_mask)
#         sam_centers.append(center)
#         sam_areas.append(area)

#     # compute distance matrices
#     center_dist_matrix = cdist(attr_centers, sam_centers)
#     area_diff_matrix = np.abs(np.subtract.outer(attr_areas, sam_areas))

#     # Normalize
#     center_dist_matrix = center_dist_matrix / (np.max(center_dist_matrix) + 1e-5)
#     area_diff_matrix = area_diff_matrix / (np.max(area_diff_matrix) + 1e-5)

#     cost_matrix = center_dist_matrix + area_diff_matrix

#     for i in range(len(attr_masks)):
#         j = np.argmin(cost_matrix[i])  # find closest SAM mask
#         matched.append(j)

#     return matched

def extract_masks_from_label_image(label_img):
    """提取每个唯一值对应的 mask（除了 0 背景）"""
    masks = []
    classes = []
    unique_vals = np.unique(label_img)
    for val in unique_vals:
        if val == 0:
            continue  # 忽略背景
        mask = (label_img == val).astype(np.uint8)
        masks.append(mask)
        classes.append(val)
    return masks, classes

def extract_masks_from_label_json(json_path, height, width):
    """
    从 JSON 格式的 polygon 数据中提取 masks 和对应的 class/instance ID
    :param json_path: panoptic JSON 文件路径
    :param height: mask 高度
    :param width: mask 宽度
    :return: masks, classes
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    masks = []
    classes = []

    for segment in data['segments_info']:
        polygons = segment.get('polygon', [])
        mask = polygons_to_mask(polygons, height, width)
        masks.append(mask)
        classes.append(segment['instance_id'])  # 或 segment['class_id']，按需替换

    return masks, classes

def apply_labels_to_image(attr_masks, sam_masks, attr_classes, CLASS_NAME_TO_ID):
    output_masks = []
    matched_indices = match_masks(attr_masks, sam_masks)
    #print("matched_indices:", matched_indices)

    mask_class_attrs = {}
    output_idx = []

    height, width = attr_masks[0].shape
    label_img = np.zeros((height, width), dtype=np.uint32)
    output_masks_list = []

    for instance_id, sam_idx in zip(attr_classes, matched_indices):
        matched_sam_mask = sam_masks[sam_idx]
        label_img[matched_sam_mask > 0] = instance_id

        if instance_id > 1000:
            class_id = int(instance_id / 1000)
        else:
            class_id = instance_id

        ID_TO_CLASS_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}
        selected_class = ID_TO_CLASS_NAME.get(class_id, 'Unknown')

        mask_class_attrs[sam_idx] = selected_class  # 更新mask的类
        mask_class_attrs[f"{sam_idx}_instance_id"] = instance_id
        output_idx.append(sam_idx)
        output_masks_list.append(matched_sam_mask)

    return label_img, mask_class_attrs, output_idx, np.array(output_masks_list)


def sort_results_by_mask_area(results, ascending=True):
    """
    根据每个 mask 的面积对 results 中的分割掩码重新排序。
    ascending=True 表示面积从小到大；False 表示从大到小。
    该函数会直接修改 results。
    """
    for r in results:
        if not hasattr(r, "masks") or r.masks is None:
            continue  # 跳过没有 mask 的结果

        masks = r.masks.data  # [num_masks, H, W]
        areas = masks.sum(dim=(1, 2))
        sorted_indices = torch.argsort(areas, descending=not ascending)

        # 按面积重新排序
        r.masks.data = masks[sorted_indices]

        # 同步 boxes、probs 等（若存在）
        if hasattr(r, 'boxes') and r.boxes is not None:
            r.boxes.data = r.boxes.data[sorted_indices]
        if hasattr(r, 'probs') and r.probs is not None:
            r.probs = r.probs[sorted_indices]

    return results
