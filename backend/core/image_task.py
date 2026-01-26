# backend/core/image_task.py
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
from utils.tools_gradio import fast_process
import io
import base64
from utils.match import sort_results_by_mask_area, apply_labels_to_image

from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as TF
from PIL import ImageDraw
from pathlib import Path
import json
import cv2
import yaml
from utils.tracking import XMem
from utils.inference.inference_core import InferenceCore, MaskMapper
import torch.nn.functional as F
import torchvision.transforms as transforms


class ImageTaskCore:
    def __init__(self, global_state: dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 确保路径正确
        self.model = YOLO('./weights/FastSAM-x.pt')

        self.GLOBAL_STATE = global_state

        # ===== tracking (XMem) =====
        self.tracking_processor = None
        self.tracking_mapper = None
        self._init_xmem_tracker()   # 👈 新增
        # 用于“跨帧持续追踪”的状态
        self.tracking_ready = False                 # 是否已经给过第一帧 mask
        self.tracking_prev_img = None               # 上一帧 PIL
        self.tracking_label_to_polyid = {}          # 1..K -> polygon_mask_id
        self.tracking_polyid_to_label = {}          # polygon_mask_id -> 1..K


        self.current_image_pil = None
        self.current_masks_np = None
        self.mask_id_to_class = {}
        self.mask_id_to_instance = {}       # mask_id -> instance_id
        self.class_to_instances = {}        # class_name -> [instance_id, ...]

        self.polygon_index = 1  # 和 SAM mask id 隔离
        self.polygon_masks = {}   # {mask_id: np.bool_(H,W)}

        self.mask_id_to_source = {}   # mask_id -> "sam" | "polygon"
        self.polygon_points = {}  # mask_id -> [[x,y], ...]

        self.tracking_poly_signature = None  # 用来检测 polygon 集合变化

    def _init_xmem_tracker(self):
        """
        初始化 XMem tracking（全生命周期一次）
        """
        config_path = Path("./config/config.yaml")
        if not config_path.exists():
            print("[XMem] config.yaml not found, tracking disabled")
            return

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config["enable_long_term"] = not config.get("disable_long_term", False)

        torch.autograd.set_grad_enabled(False)

        tracking_model = XMem(config, config.get("model")).cuda().eval()

        if config.get("model") is not None:
            model_weights = torch.load(config["model"])
            tracking_model.load_weights(
                model_weights,
                init_as_zero_if_needed=True
            )
        else:
            print("[XMem] No model weights loaded")

        self.tracking_processor = InferenceCore(tracking_model, config=config)
        self.tracking_mapper = MaskMapper()

        print("[XMem] Tracker initialized")

    def process_first_image(self, image_path):

        self.mask_id_to_class.clear()
        self.mask_id_to_instance.clear()
        self.class_to_instances.clear()

        img = Image.open(image_path).convert("RGB")
        self.current_image_pil = img

        input_size = 1024
        results = self.model(
            img,
            device=self.device,
            retina_masks=True,
            iou=0.7,
            conf=0.25,
            imgsz=input_size
        )
        results = sort_results_by_mask_area(results, ascending=True)

        # ✅ 保存 numpy mask，供后续点击使用
        self.current_masks_np = results[0].masks.data.cpu().numpy()

        for i in range(self.current_masks_np.shape[0]):
            self.mask_id_to_source[i] = "sam"

        # ✅ 关键：补上 scale
        vis_img = fast_process(
            annotations=results[0].masks.data,
            image=img,
            device=self.device,
            scale=(1024 // input_size),   # 👈 必须有
            better_quality=False,
            mask_random_color=True,
            bbox=None,
            use_retina=True,
            withContours=True
        )

        return self.pil_to_base64(img), self.pil_to_base64(vis_img)

    def pil_to_base64(self, img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def get_mask_at(self, x: int, y: int):

        for mask_id, mask in self.polygon_masks.items():
            if mask[y, x]:
                return mask_id

        if self.current_masks_np is None:
            return None

        N, H, W = self.current_masks_np.shape
        if x < 0 or y < 0 or x >= W or y >= H:
            return None

        for i in range(N):
            if self.current_masks_np[i, y, x]:
                return i

        return None

    # def set_mask_class(self, mask_id: int, class_name: str):
    #     self.mask_id_to_class[mask_id] = class_name
    #     return self.render_overlay()
    def set_mask_class(self, mask_id: int, class_name: str, is_thing: bool):
        self.mask_id_to_class[mask_id] = class_name

        # ✅ 如果是 SAM mask，没写 source 的默认当 sam
        if mask_id not in self.mask_id_to_source:
            self.mask_id_to_source[mask_id] = "sam"

        # 生成两个 overlay（方案一）
        sam_img = self.render_sam_overlay()
        poly_img = self.render_polygon_overlay()

        payload = {
            "sam_overlay": self.pil_to_base64(sam_img),
            "polygon_overlay": self.pil_to_base64(poly_img),
        }

        if not is_thing:
            return {
                **payload,
                "need_instance": False,
            }

        existing = self.class_to_instances.get(class_name, [])
        return {
            **payload,
            "need_instance": True,
            "existing_instances": existing,
        }



    def set_mask_instance(self, mask_id: int, class_name: str, instance_choice):
        """
        instance_choice: "new" 或者 具体 instance_id（int / str）
        instance_id = class_id * 1000 + index
        """

        # 1️⃣ class_id
        class_id = self.GLOBAL_STATE["class_name_to_id"][class_name]

        # 2️⃣ 已有实例（完整 instance_id）
        existing = self.class_to_instances.get(class_name, [])

        # 3️⃣ 选择 new
        if instance_choice == "new":
            # 找出当前 class 下已用的 index
            used_indices = [
                inst_id % 1000
                for inst_id in existing
                if inst_id // 1000 == class_id
            ]

            next_index = max(used_indices) + 1 if used_indices else 1
            instance_id = class_id * 1000 + next_index

            existing.append(instance_id)
            existing.sort()
            self.class_to_instances[class_name] = existing

        # 4️⃣ 选择已有实例
        else:
            instance_id = int(instance_choice)

            if instance_id not in existing:
                existing.append(instance_id)
                existing.sort()
                self.class_to_instances[class_name] = existing

        # 5️⃣ 绑定 mask → instance
        self.mask_id_to_instance[mask_id] = instance_id

        # 6️⃣ 返回两套 overlay（方案一）
        sam_img = self.render_sam_overlay()
        poly_img = self.render_polygon_overlay()

        return {
            "instance_id": instance_id,
            "sam_overlay": self.pil_to_base64(sam_img),
            "polygon_overlay": self.pil_to_base64(poly_img),
        }

    def class_to_rgb(self, name: str):
        """
        稳定 hash -> RGB (0-255)
        """
        h = abs(hash(name)) & 0xFFFFFFFF
        r = 50 + (h & 0xFF) * 205 // 255
        g = 50 + ((h >> 8) & 0xFF) * 205 // 255
        b = 50 + ((h >> 16) & 0xFF) * 205 // 255
        return (int(r), int(g), int(b))

    def render_sam_overlay(self):
        if self.current_image_pil is None or self.current_masks_np is None:
            return self.current_image_pil

        items = [(mid, cls) for mid, cls in self.mask_id_to_class.items()
                if self.mask_id_to_source.get(mid, "sam") == "sam"]

        if not items:
            return self.current_image_pil

        masks = []
        colors = []
        H, W = self.current_masks_np.shape[1], self.current_masks_np.shape[2]

        for mask_id, class_name in sorted(items):
            if 0 <= mask_id < self.current_masks_np.shape[0]:
                masks.append(torch.tensor(self.current_masks_np[mask_id], dtype=torch.bool))
                colors.append(self.class_to_rgb(class_name))

        if not masks:
            return self.current_image_pil

        mask_tensor = torch.stack(masks)
        img = self.current_image_pil.resize((W, H))
        img_tensor = pil_to_tensor(img)

        overlayed = draw_segmentation_masks(
            img_tensor, masks=mask_tensor, alpha=0.6, colors=colors
        )
        return TF.to_pil_image(overlayed)


    def render_polygon_overlay(self):
        if self.current_image_pil is None:
            return self.current_image_pil

        items = [(mid, cls) for mid, cls in self.mask_id_to_class.items()
                if self.mask_id_to_source.get(mid) == "polygon"]

        if not items:
            return self.current_image_pil

        W, H = self.current_image_pil.size

        masks = []
        colors = []

        for mask_id, class_name in sorted(items):
            m = self.polygon_masks.get(mask_id)
            if m is None:
                continue
            # ✅ polygon mask 确保 [H,W]
            if m.shape != (H, W):
                # 如果你 points 是原图坐标，这里一般不会发生
                m = np.array(Image.fromarray(m.astype("uint8")*255).resize((W, H))).astype(bool)

            masks.append(torch.tensor(m, dtype=torch.bool))
            colors.append(self.class_to_rgb(class_name))

        if not masks:
            return self.current_image_pil

        mask_tensor = torch.stack(masks)
        img_tensor = pil_to_tensor(self.current_image_pil)

        overlayed = draw_segmentation_masks(
            img_tensor, masks=mask_tensor, alpha=0.6, colors=colors
        )
        return TF.to_pil_image(overlayed)



    def _render_overlay_by_source(self, source: str):
        masks = []
        colors = []

        for mask_id, class_name in self.mask_id_to_class.items():
            if self.mask_id_to_source.get(mask_id) != source:
                continue

            masks.append(
                torch.tensor(self.current_masks_np[mask_id], dtype=torch.bool)
            )
            colors.append(self.class_to_rgb(class_name))

        if not masks:
            return self.current_image_pil

        mask_tensor = torch.stack(masks)

        img = self.current_image_pil.resize(
            (self.current_masks_np.shape[2], self.current_masks_np.shape[1])
        )
        img_tensor = pil_to_tensor(img)

        overlayed = draw_segmentation_masks(
            img_tensor,
            masks=mask_tensor,
            alpha=0.6,
            colors=colors
        )

        return TF.to_pil_image(overlayed)



    def add_polygon_mask(self, points, image_size):
        """
        points: [[x,y], ...]
        image_size: (W, H)
        """
        #import numpy as np
        #from PIL import Image, ImageDraw

        W, H = image_size

        # 1️⃣ PIL 需要 tuple[(x,y),...]
        points_xy = [(int(x), int(y)) for x, y in points]

        # 2️⃣ 创建空 mask
        mask = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask)

        # 3️⃣ 画多边形
        draw.polygon(points_xy, outline=1, fill=1)

        mask_np = np.array(mask, dtype=bool)

        # 5️⃣ polygon mask id（避开 SAM）
        polygon_mask_id = 1000000 + self.polygon_index
        self.polygon_index += 1
        self.polygon_masks[polygon_mask_id] = mask_np   # ✅ 必须有
        self.polygon_points[polygon_mask_id] = points
        self.mask_id_to_source[polygon_mask_id] = "polygon"

        self.reset_xmem_polygon_session()

        return polygon_mask_id

    def build_semantic_label(self):
        """
        return: np.uint8 [H, W]，所有已标注类别（SAM + polygon，thing + non-thing）
        """
        H, W = self.current_image_pil.size[1], self.current_image_pil.size[0]
        semantic = np.zeros((H, W), dtype=np.uint8)

        for mask_id, class_name in self.mask_id_to_class.items():
            class_id = self.GLOBAL_STATE["class_name_to_id"][class_name]

            if self.mask_id_to_source.get(mask_id) == "sam":
                if not (0 <= mask_id < self.current_masks_np.shape[0]):
                    continue
                m = self.current_masks_np[mask_id]
            else:
                m = self.polygon_masks.get(mask_id)
                if m is None:
                    continue

            semantic[m.astype(bool)] = class_id

        return semantic


    def build_instance_label(self):
        """
        return: np.uint32 [H, W]
        """
        H, W = self.current_image_pil.size[1], self.current_image_pil.size[0]
        label = np.zeros((H, W), dtype=np.uint32)

        for mask_id, class_name in self.mask_id_to_class.items():
            if mask_id not in self.mask_id_to_instance:
                continue

            instance_id = self.mask_id_to_instance[mask_id]

            if self.mask_id_to_source.get(mask_id) == "sam":
                if mask_id >= self.current_masks_np.shape[0]:
                    continue
                m = self.current_masks_np[mask_id]
            else:
                m = self.polygon_masks.get(mask_id)
                if m is None:
                    continue

            # ✅ 关键修复
            m = m.astype(bool)

            label[m] = instance_id

        return label


    def save_current_annotations(self):
        project_path = self.GLOBAL_STATE["project_path"]
        img_name = self.GLOBAL_STATE["current_image_name"]

        save_dir = Path(project_path) / "panoptic"
        save_dir.mkdir(exist_ok=True)

        instance_label = self.build_instance_label()

        Image.fromarray(instance_label).save(
            save_dir / f"{img_name}_gtFine_instanceIds.png"
        )

        #semantic = instance_label // 1000
        semantic = self.build_semantic_label()

        Image.fromarray(semantic.astype(np.uint8)).save(
            save_dir / f"{img_name}_gtFine_labelIds.png"
        )

        self.save_panoptic_json(save_dir, img_name)

    @staticmethod
    def mask_to_polygon(mask: np.ndarray):
        """
        mask: bool or uint8 [H, W]
        return: list of polygons (每个 polygon 是 [[x,y], ...])
        """
        mask = mask.astype(np.uint8)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        polygons = []
        for cnt in contours:
            if len(cnt) < 3:
                continue
            polygon = cnt.squeeze(1).tolist()
            polygons.append(polygon)

        return polygons


    def build_panoptic_segments(self):
        segments = []
        seg_id = 1

        for mask_id, class_name in self.mask_id_to_class.items():
            class_id = self.GLOBAL_STATE["class_name_to_id"][class_name]
            isthing = class_name in self.GLOBAL_STATE["thing_classes"]
            instance_id = self.mask_id_to_instance.get(mask_id)

            if self.mask_id_to_source.get(mask_id) == "polygon":
                mask = self.polygon_masks.get(mask_id)
                if mask is None:
                    continue
                polygons = self.mask_to_polygon(mask)
            else:
                mask = self.current_masks_np[mask_id].astype(bool)
                polygons = self.mask_to_polygon(mask)

            segment = {
                "id": seg_id,
                "class_id": int(class_id),
                "category_name": class_name,
                "isthing": bool(isthing),
                "polygon": polygons
            }

            # thing 才有 instance_id
            if isthing and instance_id is not None:
                segment["instance_id"] = int(instance_id)

            segments.append(segment)
            seg_id += 1

        return {"segments_info": segments}

    def save_panoptic_json(self, save_dir, img_name):
        data = self.build_panoptic_segments()
        json_path = save_dir / f"{img_name}_gtFine_panoptic.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def match_prev_panoptic_to_current_sam(self, prev_instance_label, CLASS_NAME_TO_ID):
        """
        使用你原来的 apply_labels_to_image，
        把上一帧 panoptic label 匹配到当前帧 SAM masks
        """

        if self.current_masks_np is None:
            return None

        # 1️⃣ 从 instance label 中提取 attr_masks / attr_classes
        attr_masks = []
        attr_classes = []

        for inst_id in np.unique(prev_instance_label):
            if inst_id == 0:
                continue
            m = (prev_instance_label == inst_id).astype(np.uint8)
            attr_masks.append(m)
            attr_classes.append(int(inst_id))

        if not attr_masks:
            return None

        # 2️⃣ 调用你原封不动的算法
        label_img, mask_class_attrs, output_idx, output_masks = apply_labels_to_image(
            attr_masks,
            self.current_masks_np,
            attr_classes,
            CLASS_NAME_TO_ID
        )

        return {
            "label_img": label_img,                  # uint32 instance label
            "mask_class_attrs": mask_class_attrs,    # sam_idx → class / instance
            "output_idx": output_idx,                # 被选中的 sam mask id
            "output_masks": output_masks             # [K,H,W]
        }

    def match_prev_stuff_masks_to_current_sam(self, prev_masks, iou_thresh=0.3):
        """
        prev_masks: list of {
            "mask": np.bool [H,W],
            "class_name": str
        }
        """
        if self.current_masks_np is None:
            return []

        sam_masks = self.current_masks_np.astype(bool)
        used = set()
        results = []

        for item in prev_masks:
            best_iou = 0
            best_idx = None

            for i, sam_mask in enumerate(sam_masks):
                if i in used:
                    continue

                inter = np.logical_and(item["mask"], sam_mask).sum()
                union = np.logical_or(item["mask"], sam_mask).sum()
                if union == 0:
                    continue

                iou = inter / union
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx is not None and best_iou >= iou_thresh:
                used.add(best_idx)
                results.append({
                    "sam_idx": best_idx,
                    "class_name": item["class_name"]
                })

        return results

    def get_sam_mask_ids(self):
        """
        返回当前帧 SAM mask 的 id 列表（只包含 source==sam 且在 current_masks_np 范围内的）
        """
        if self.current_masks_np is None:
            return []
        n = self.current_masks_np.shape[0]
        return [mid for mid, src in self.mask_id_to_source.items() if src == "sam" and 0 <= mid < n]


    def build_instance_label_sam_only(self):
        """
        只用 SAM masks 构建 instance label（给 apply_labels_to_image 做 thing match）
        return: np.uint32 [H, W]
        """
        if self.current_image_pil is None or self.current_masks_np is None:
            return None

        H, W = self.current_image_pil.size[1], self.current_image_pil.size[0]
        label = np.zeros((H, W), dtype=np.uint32)

        for mask_id, class_name in self.mask_id_to_class.items():
            # 只取 SAM
            if self.mask_id_to_source.get(mask_id) != "sam":
                continue
            if mask_id not in self.mask_id_to_instance:
                continue
            if not (0 <= mask_id < self.current_masks_np.shape[0]):
                continue

            instance_id = int(self.mask_id_to_instance[mask_id])
            m = self.current_masks_np[mask_id].astype(bool)
            label[m] = instance_id

        return label


    def collect_prev_sam_stuff(self):
        """
        收集上一帧 SAM 的 non-thing（stuff）mask，用于 stuff IoU 传播（只看 SAM）
        return: list of { "mask": bool[H,W], "class_name": str }
        """
        out = []
        if self.current_masks_np is None:
            return out

        for mask_id, class_name in self.mask_id_to_class.items():
            if self.mask_id_to_source.get(mask_id) != "sam":
                continue
            if not (0 <= mask_id < self.current_masks_np.shape[0]):
                continue
            # non-thing 才收
            if class_name in self.GLOBAL_STATE["thing_classes"]:
                continue

            out.append({
                "mask": self.current_masks_np[mask_id].astype(bool),
                "class_name": class_name
            })

        return out

    def _build_polygon_label_mask(self, polygon_items, H, W):
        """
        polygon_items: list of {
            "poly_id": int,
            "mask": np.bool_(H,W)
        }

        return:
        label_mask: np.int64(H,W)  (0=bg, 1..K=instance)
        label_to_polyid: dict[int,int]
        polyid_to_label: dict[int,int]
        """
        label_mask = np.zeros((H, W), dtype=np.int64)

        label_to_polyid = {}
        polyid_to_label = {}

        # 固定顺序：按 poly_id 排序，避免随机抖动
        polygon_items = sorted(polygon_items, key=lambda x: x["poly_id"])

        for i, item in enumerate(polygon_items, start=1):  # 1..K
            poly_id = item["poly_id"]
            m = item["mask"].astype(bool)

            label_mask[m] = i
            label_to_polyid[i] = poly_id
            polyid_to_label[poly_id] = i

        return label_mask, label_to_polyid, polyid_to_label

    def track_polygons_xmem(self, prev_img_pil, curr_img_pil, prev_polygon_items):
        """
        prev_img_pil: PIL.Image   上一帧
        curr_img_pil: PIL.Image   当前帧
        prev_polygon_items: list of {
            "poly_id": int,
            "mask": np.bool_(H,W),
            "class_name": str,
            "instance_id": Optional[int]
        }

        return:
        tracked_polygon_items: list of 同结构（mask 更新到当前帧）
        """
        if self.tracking_processor is None or self.tracking_mapper is None:
            # 没初始化 tracker，就直接不追踪
            return []

        if not prev_polygon_items:
            # ✅ 只要 polygon 集合变了（新增 / 删除），必须 reset 并重新喂 mask
            curr_sig = tuple(sorted([it["poly_id"] for it in prev_polygon_items]))
            if self.tracking_poly_signature != curr_sig:
                self.reset_xmem_polygon_session()
                self.tracking_poly_signature = curr_sig


            # 没有 polygon，不追踪，同时重置 tracker 状态
            self.tracking_ready = False
            self.tracking_prev_img = None
            self.tracking_label_to_polyid = {}
            self.tracking_polyid_to_label = {}
            return []

        # ---------- 0) 尺寸 ----------
        W, H = curr_img_pil.size

        # ---------- 1) 构造上一帧 label mask (1..K) ----------
        prev_label_mask, label_to_polyid, polyid_to_label = self._build_polygon_label_mask(
            [{"poly_id": it["poly_id"], "mask": it["mask"]} for it in prev_polygon_items],
            H, W
        )

        # 保存映射（用于输出还原）
        self.tracking_label_to_polyid = label_to_polyid
        self.tracking_polyid_to_label = polyid_to_label

        # ---------- 2) 构造 tracking_img ----------
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                transform = transforms.ToTensor()

                tracking_img = transform(curr_img_pil).cuda()    # [3,H,W]
                tracking_img_half = F.interpolate(
                    tracking_img.unsqueeze(0),
                    scale_factor=0.5,
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)  # [3,H/2,W/2]

                # ---------- 3) 第一次：需要喂 mask ----------
                if not self.tracking_ready:
                    # prev_label_mask: (H, W)  值域 0..K
                    prev_label_mask_np = prev_label_mask.astype(np.uint8)

                    # ✅ convert_mask 会把 label-map 转成 multi-object masks: (N, H, W)
                    #    tracking_labels 是对应 N 个目标的内部 remap label
                    tracking_msk_np, tracking_labels = self.tracking_mapper.convert_mask(prev_label_mask_np)
                    # tracking_msk_np: (N, H, W)
                    tracking_msk = torch.Tensor(tracking_msk_np).cuda()
                    #tracking_msk = torch.from_numpy(tracking_msk_np).float().cuda()  # (N,H,W)

                    # ✅ 设置当前所有 label（保持你原逻辑）
                    self.tracking_processor.set_all_labels(list(self.tracking_mapper.remappings.values()))

                    # ✅ 下采样：必须保持 (N,H/2,W/2)，不能 squeeze 成 (H/2,W/2)
                    tracking_msk_half = F.interpolate(
                        tracking_msk.unsqueeze(0),      # [1,N,H,W]
                        scale_factor=0.5,
                        mode="nearest"
                    ).squeeze(0)                        # [N,H/2,W/2]

                    self.tracking_ready = True
                else:
                    tracking_msk_half = None
                    tracking_labels = None
                assert tracking_img_half.dim() == 3, tracking_img_half.shape
                if tracking_msk_half is not None:
                    assert tracking_msk_half.dim() == 3, f"mask must be (N,H,W), got {tracking_msk_half.shape}"
                    assert isinstance(tracking_labels, (list, tuple)) and len(tracking_labels) == tracking_msk_half.shape[0], \
                        f"labels mismatch: {len(tracking_labels)} vs {tracking_msk_half.shape[0]}"


                # ---------- 4) XMem step ----------
                tracking_prob = self.tracking_processor.step(tracking_img_half, tracking_msk_half, tracking_labels)
                tracking_out_mask = torch.max(tracking_prob, dim=0).indices  # [H/2,W/2]

                # ---------- 5) upscale 回原图 ----------
                tracking_out_mask = F.interpolate(
                    tracking_out_mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(H, W),
                    mode="nearest"
                ).squeeze(0).squeeze(0).long()

                tracking_out_mask = tracking_out_mask.detach().cpu().numpy().astype(np.uint8)

                # mapper.remap_index_mask: 把内部 remap 的 index 映射回输入 label (1..K)
                tracking_out_mask = self.tracking_mapper.remap_index_mask(tracking_out_mask)

        # ---------- 6) tracking_out_mask -> 每个 polygon 的 bool mask ----------
        tracked = []
        for it in prev_polygon_items:
            poly_id = it["poly_id"]
            label = self.tracking_polyid_to_label.get(poly_id, None)
            if label is None:
                continue

            m = (tracking_out_mask == label)
            if m.sum() == 0:
                # 追踪不到就直接丢（或你也可以保留空 mask）
                continue

            tracked.append({
                **it,
                "mask": m.astype(bool)
            })

        # 保存 prev_img（如果你后续要做“基于 prev_img 的逻辑”）
        self.tracking_prev_img = curr_img_pil
        return tracked

    def reset_xmem_polygon_session(self):
        """当 polygon 集合变化（新增/删除）时，必须重置 XMem 的 memory + mapper"""
        if self.tracking_processor is None:
            return

        # InferenceCore 通常有清 memory 的接口，不同实现名字可能不同
        for fn in ["clear_memory", "reset", "restart", "reset_memory"]:
            if hasattr(self.tracking_processor, fn):
                getattr(self.tracking_processor, fn)()
                break

        # mapper 也必须重置（remappings 不然会错）
        self.tracking_mapper = MaskMapper()

        self.tracking_ready = False
        self.tracking_label_to_polyid = {}
        self.tracking_polyid_to_label = {}
        self.tracking_poly_signature = None
