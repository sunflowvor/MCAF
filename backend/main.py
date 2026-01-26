import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.image_task import ImageTaskCore
from core.panoptic_task import PanopticTaskCore
from fastapi import HTTPException
import json
import torch
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as TF

from scripts.tracking_export import export_tracking_with_vis
from PIL import Image
from fastapi.responses import FileResponse

from scripts.panoptic_to_pointcloud import run_panoptic_to_pointcloud
import numpy as np

from scripts.lidar_odometry import run_lidar_odometry

from scripts.pointcloud_reproject import run_pointcloud_reproject


app = FastAPI()

# 全局状态，记录当前操作的数据路径
GLOBAL_STATE = {
    "project_path": None,
    "config": None,              # ← 项目 config.json
    "class_name_to_id": None,
    "thing_classes": None,
    "sam_model": None,
    "current_image_path": None,
    "current_image_name": None,
    "current_image_index": None,
    "image_files": [],
    "category_base_colors": None,
    "is_kitti": False,
    "chunks_number":None
}

class PathRequest(BaseModel):
    path: str

# 1. 获取工作流配置 (由前端渲染 Dashboard)
@app.get("/api/pipeline")
def get_pipeline():
    config_path = Path(__file__).parent / "pipeline.json"
    return json.loads(config_path.read_text(encoding="utf-8"))

# 2. 校验并保存路径 (核心步骤)
@app.post("/api/import-path")
def import_path(req: PathRequest):
    base = Path(req.path).resolve()

    details = {
        "lidar": (base / "lidar").is_dir(),
        "image": (base / "image").is_dir(),
        "config": (base / "config.json").is_file()
    }

    if not all(details.values()):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "details": details}
        )

    # ===== 读取 config.json =====
    config_path = base / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config.json: {e}")

    # ===== 写入全局状态 =====
    GLOBAL_STATE["project_path"] = str(base)
    GLOBAL_STATE["config"] = config
    GLOBAL_STATE["class_name_to_id"] = config["class_name_to_id"]
    GLOBAL_STATE["thing_classes"] = set(config["thing_classes"])
    GLOBAL_STATE["sam_model"] = config.get("SAM_model")
    GLOBAL_STATE["is_kitti"] = config.get("IS_KITTI")
    GLOBAL_STATE["chunks_number"] = config.get("CHUNKS_NUMBER")

    cfg = GLOBAL_STATE["config"]

    # 1️⃣ 从 config 取颜色
    raw_colors = cfg.get("category_base_colors", {})

    # 2️⃣ 转成 int -> tuple(int,int,int)
    CATEGORY_BASE_COLORS = {
        int(k): tuple(v)
        for k, v in raw_colors.items()
    }


    app.mount(
        "/project_image",
        StaticFiles(directory=base / "image"),
        name="project_image"
    )

    return {
        "status": "success",
        "details": details,
        "classes": list(config["class_name_to_id"].keys())
    }

@app.get("/api/config")
def get_project_config():
    if not GLOBAL_STATE["config"]:
        raise HTTPException(status_code=400, detail="Project not initialized")

    return {
        "class_name_to_id": GLOBAL_STATE["class_name_to_id"],
        "thing_classes": list(GLOBAL_STATE["thing_classes"])
    }

# 3. 提供给其他页面/脚本读取路径
@app.get("/api/get-context")
def get_context():
    return GLOBAL_STATE

image_task_core = ImageTaskCore(GLOBAL_STATE)
#image_task_core = None

@app.get("/api/image/init-first")
async def init_first_image():
    project_path = GLOBAL_STATE.get("project_path")
    if not project_path:
        raise HTTPException(status_code=400, detail="Path not set")
    
    img_dir = Path(project_path) / "image"
    # 获取第一张图
    img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
    if not img_files:
        raise HTTPException(status_code=404, detail="No images found")

    img_path = img_files[0]

    # 👇 写入全局状态（这是关键）
    GLOBAL_STATE["current_image_path"] = str(img_path)
    GLOBAL_STATE["current_image_name"] = img_path.stem
    GLOBAL_STATE["current_image_index"] = 0
    GLOBAL_STATE["image_files"] = img_files
    
    ori_b64, seg_b64 = image_task_core.process_first_image(img_path)
    
    return {
        "filename": img_path.name,
        "ori_image": ori_b64,
        "seg_image": seg_b64
    }

@app.post("/api/image/query-mask")
def query_mask(data: dict):
    x = int(data.get("x", -1))
    y = int(data.get("y", -1))

    mask_id = image_task_core.get_mask_at(x, y)
    if mask_id is None:
        return {"mask_id": None}

    return {
        "mask_id": mask_id,
        "current_class": image_task_core.mask_id_to_class.get(mask_id),
        "class_options": list(GLOBAL_STATE["class_name_to_id"].keys())
    }

@app.post("/api/image/set-mask-class")
def set_mask_class(payload: dict = Body(...)):
    mask_id = int(payload["mask_id"])
    class_name = payload["class_name"]

    if not GLOBAL_STATE.get("class_name_to_id"):
        raise HTTPException(400, "Project not initialized")

    if class_name not in GLOBAL_STATE["class_name_to_id"]:
        raise HTTPException(400, "Invalid class")

    class_id = GLOBAL_STATE["class_name_to_id"][class_name]
    is_thing = class_name in GLOBAL_STATE["thing_classes"]

    out = image_task_core.set_mask_class(mask_id, class_name, is_thing=is_thing)

    # ✅ 把 class_id 放回去（你前端要用）
    out["class_id"] = class_id
    return out

@app.post("/api/image/set-mask-instance")
def set_mask_instance(payload: dict = Body(...)):
    mask_id = int(payload["mask_id"])
    class_name = payload["class_name"]
    instance = payload["instance"]

    out = image_task_core.set_mask_instance(mask_id, class_name, instance)
    return out

@app.post("/api/image/delete-mask")
def delete_mask(payload: dict = Body(...)):
    mask_id = int(payload["mask_id"])

    # 1) 删绑定
    image_task_core.mask_id_to_class.pop(mask_id, None)
    image_task_core.mask_id_to_instance.pop(mask_id, None)

    # 2) 如果是 polygon，删本体
    if image_task_core.mask_id_to_source.get(mask_id) == "polygon":
        image_task_core.reset_xmem_polygon_session()
        
        image_task_core.polygon_masks.pop(mask_id, None)
        image_task_core.polygon_points.pop(mask_id, None)

    # 3) 删 source 记录
    image_task_core.mask_id_to_source.pop(mask_id, None)

    # 4) 重绘
    sam_img = image_task_core.render_sam_overlay()
    poly_img = image_task_core.render_polygon_overlay()

    return {
        "sam_overlay": image_task_core.pil_to_base64(sam_img),
        "polygon_overlay": image_task_core.pil_to_base64(poly_img),
    }



@app.post("/api/image/polygon-create-mask")
def polygon_create_mask(payload: dict = Body(...)):
    points = payload["polygon_points"]

    img = image_task_core.current_image_pil
    if img is None:
        raise HTTPException(400, "No image loaded")

    W, H = img.size
    mask_id = image_task_core.add_polygon_mask(points, (W, H))

    return {
        "mask_id": mask_id,
        "class_options": list(GLOBAL_STATE["class_name_to_id"].keys())
    }

def collect_prev_polygon_items(core):
    """
    只收 polygon，不碰 SAM
    """
    items = []
    for mask_id, class_name in core.mask_id_to_class.items():
        if core.mask_id_to_source.get(mask_id) != "polygon":
            continue

        m = core.polygon_masks.get(mask_id)
        if m is None:
            continue

        items.append({
            "poly_id": mask_id,
            "mask": m,
            "class_name": class_name,
            "instance_id": core.mask_id_to_instance.get(mask_id)  # thing 才有
        })
    return items

def apply_tracked_polygons(core, tracked_items):
    """
    把 XMem 输出的 polygon masks 写回 core
    """
    for it in tracked_items:
        poly_id = it["poly_id"]

        core.polygon_masks[poly_id] = it["mask"]
        core.mask_id_to_source[poly_id] = "polygon"
        core.mask_id_to_class[poly_id] = it["class_name"]

        if it.get("instance_id") is not None:
            core.mask_id_to_instance[poly_id] = it["instance_id"]

            # 维护实例池（跨 SAM / polygon）
            core.class_to_instances.setdefault(it["class_name"], [])
            if it["instance_id"] not in core.class_to_instances[it["class_name"]]:
                core.class_to_instances[it["class_name"]].append(it["instance_id"])
                core.class_to_instances[it["class_name"]].sort()

@app.post("/api/image/save-and-next")
def save_and_next():
    core = image_task_core
    state = GLOBAL_STATE

    # ===== 0️⃣ 记录实例池（跨 SAM / polygon）=====
    instance_pool = {k: sorted(set(v)) for k, v in core.class_to_instances.items()}

    # ===== 1️⃣ 收集上一帧 polygon（⚠️ 在 save 前）=====
    prev_polygon_items = collect_prev_polygon_items(core)
    prev_img = core.current_image_pil

    # ===== 2️⃣ 保存当前帧 =====
    core.save_current_annotations()

    # ===== 3️⃣ 上一帧 SAM-only 数据（你已有）=====
    prev_thing_label = core.build_instance_label_sam_only()
    prev_stuff = core.collect_prev_sam_stuff()

    # ===== 4️⃣ 移动到下一帧 =====
    idx = state["current_image_index"] + 1
    image_files = state["image_files"]
    if idx >= len(image_files):
        return {"done": True}

    next_image_path = image_files[idx]
    state["current_image_index"] = idx
    state["current_image_name"] = next_image_path.stem

    # ===== 5️⃣ SAM 推理下一帧 =====
    ori_b64, seg_b64 = core.process_first_image(next_image_path)

    # ⚠️ 清空 per-frame 状态，但保留实例池
    core.mask_id_to_class.clear()
    core.mask_id_to_instance.clear()
    core.class_to_instances = instance_pool

    labels = []

    # ===== 6️⃣ SAM thing tracking（你已有）=====
    matched_thing = core.match_prev_panoptic_to_current_sam(prev_thing_label, GLOBAL_STATE["class_name_to_id"])
    if matched_thing:
        for k, v in matched_thing["mask_class_attrs"].items():
            if isinstance(k, int):
                core.mask_id_to_class[k] = v
            elif isinstance(k, str) and k.endswith("_instance_id"):
                sam_idx = int(k.replace("_instance_id", ""))
                instance_id = int(v)
                core.mask_id_to_instance[sam_idx] = instance_id
                class_name = core.mask_id_to_class.get(sam_idx)

                if class_name:
                    core.class_to_instances.setdefault(class_name, [])
                    if instance_id not in core.class_to_instances[class_name]:
                        core.class_to_instances[class_name].append(instance_id)
                        core.class_to_instances[class_name].sort()
                    #print(class_name)

                    labels.append({
                        "mask_id": sam_idx,
                        "class_name": class_name,
                        "class_id": state["class_name_to_id"][class_name],
                        "instance_id": instance_id
                    })

    # ===== 7️⃣ SAM stuff tracking（你已有）=====
    matched_stuff = core.match_prev_stuff_masks_to_current_sam(prev_stuff)
    for m in matched_stuff:
        sam_idx = m["sam_idx"]
        class_name = m["class_name"]
        if sam_idx in core.mask_id_to_class:
            continue
        core.mask_id_to_class[sam_idx] = class_name
        labels.append({
            "mask_id": sam_idx,
            "class_name": class_name,
            "class_id": state["class_name_to_id"][class_name],
            "instance_id": None
        })

    # ===== 8️⃣ 🔥 Polygon XMem tracking（新加的）=====
    curr_img = core.current_image_pil
    tracked_polygons = core.track_polygons_xmem(
        prev_img_pil=prev_img,
        curr_img_pil=curr_img,
        prev_polygon_items=prev_polygon_items
    )

    # 进入新帧：先清空本帧 polygon 容器，再写入 tracking 结果
    core.polygon_masks.clear()
    core.polygon_points.clear()

    for it in tracked_polygons:
        poly_id = it["poly_id"]

        # 1️⃣ 写回 mask（用于可视化）
        core.polygon_masks[poly_id] = it["mask"]

        # 2️⃣ 标记来源
        core.mask_id_to_source[poly_id] = "polygon"

        # 3️⃣ class / instance 继承
        if it.get("class_name") is not None:
            core.mask_id_to_class[poly_id] = it["class_name"]

        if it.get("instance_id") is not None:
            core.mask_id_to_instance[poly_id] = it["instance_id"]

            # 维护实例池（跨 SAM / polygon）
            cls = it["class_name"]
            core.class_to_instances.setdefault(cls, [])
            if it["instance_id"] not in core.class_to_instances[cls]:
                core.class_to_instances[cls].append(it["instance_id"])
                core.class_to_instances[cls].sort()

    # 同步 polygon labels 到列表（前端用）
    for it in tracked_polygons:
        labels.append({
            "mask_id": it["poly_id"],
            "class_name": it["class_name"],
            "class_id": state["class_name_to_id"][it["class_name"]],
            "instance_id": it.get("instance_id")
        })


    # ===== 9️⃣ 可视化 =====
    matched_overlay = core.pil_to_base64(core.render_sam_overlay())
    polygon_overlay = core.pil_to_base64(core.render_polygon_overlay())

    return {
        "filename": next_image_path.name,
        "ori_image": ori_b64,
        "sam_overlay": seg_b64,
        "matched_overlay": matched_overlay,
        "polygon_overlay": polygon_overlay,
        "labels": labels
    }

@app.post("/api/tracking/generate")
def generate_2d_tracking():
    project_path = GLOBAL_STATE.get("project_path")
    if not project_path:
        raise HTTPException(400, "Project not initialized")

    project_path = Path(project_path)

    instance_dir = project_path / "panoptic"
    output_dir = project_path / "DDTracking"
    output_json = output_dir / "coco_tracking_from_instance.json"

    if not instance_dir.is_dir():
        raise HTTPException(
            400,
            f"panoptic folder not found under project path: {instance_dir}"
        )

    files = sorted(
        f for f in os.listdir(instance_dir)
        if f.endswith("_gtFine_instanceIds.png")
    )
    if not files:
        raise HTTPException(400, "No *_gtFine_instanceIds.png found in panoptic")

    # 自动读取尺寸
    sample = Image.open(instance_dir / files[0])
    width, height = sample.size

    output_dir.mkdir(parents=True, exist_ok=True)

    id_to_name = {v: k for k, v in GLOBAL_STATE["class_name_to_id"].items()}

    export_tracking_with_vis(
        instance_dir=project_path / "panoptic",
        original_vis_dir=project_path / "image",
        output_dir=project_path / "DDTracking",
        width=width,
        height=height,
        id_to_name=id_to_name,
    )


    return {
        "status": "success",
        "project_path": str(project_path),
        "input_dir": str(instance_dir),
        "output": str(output_json),
        "num_images": len(files)
    }

@app.get("/project_image/{filename}")
def get_project_image(filename: str):
    project_path = GLOBAL_STATE.get("project_path")
    if not project_path:
        raise HTTPException(400, "Project not initialized")

    img_path = Path(project_path) / "image" / filename
    if not img_path.exists():
        raise HTTPException(404, f"Image not found: {filename}")

    return FileResponse(img_path)

@app.get("/api/tracking/load")
def load_tracking():
    json_path = Path(GLOBAL_STATE["project_path"]) / "DDTracking/coco_tracking_from_instance.json"
    return json.loads(json_path.read_text())

@app.post("/api/tracking/save")
def save_tracking(payload: dict):
    json_path = Path(GLOBAL_STATE["project_path"]) / "DDTracking/coco_tracking_from_instance.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    return {"status": "ok"}

@app.post("/api/pointcloud/run")
def run_pointcloud_segmentation():
    project_path = Path(GLOBAL_STATE["project_path"])
    cfg = GLOBAL_STATE["config"]
    IS_KITTI = GLOBAL_STATE["is_kitti"]

    panoptic_dir = project_path / "panoptic"
    lidar_dir = project_path / "lidar"
    image_dir = project_path / "image"
    output_dir = project_path / "pointcloud_segmentation"

    # ===== 从 config 读取相机参数 =====
    cam_cfg = cfg["camera"]
    K = np.array(cam_cfg["K"], dtype=np.float32)
    T = np.array(cam_cfg["T_LIDAR_TO_CAM"], dtype=np.float32)

    width = int(cam_cfg["width"])
    height = int(cam_cfg["height"])

    # ===== thing class ids =====
    class_name_to_id = cfg["class_name_to_id"]
    thing_class_ids = {
        class_name_to_id[name]
        for name in cfg["thing_classes"]
    }

    # ===== 颜色表 =====
    raw_colors = cfg.get("category_base_colors", {})
    category_base_colors = {
        int(k): tuple(v)
        for k, v in raw_colors.items()
    }

    return run_panoptic_to_pointcloud(
        panoptic_dir=str(panoptic_dir),
        lidar_dir=str(lidar_dir),
        image_dir=str(image_dir),
        output_dir=str(output_dir),
        K=K,
        T_LIDAR_TO_CAM=T,
        width=width,                 # ✅ 新增
        height=height,               # ✅ 新增
        category_base_colors=category_base_colors,
        thing_class_ids=thing_class_ids,
        is_kitti=IS_KITTI
    )

@app.post("/api/pointcloud/odometry")
def run_lidar_odometry_api():
    project_path = GLOBAL_STATE.get("project_path")
    if not project_path:
        raise HTTPException(400, "Project not initialized")

    project_path = Path(project_path)

    npy_dir = project_path / "pointcloud_segmentation"
    output_dir = project_path / "lidar_odometry"

    if not npy_dir.is_dir():
        raise HTTPException(
            400,
            "pointcloud_segmentation not found, run pointcloud segmentation first"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = GLOBAL_STATE["config"]
    cam_cfg = cfg["camera"]
    K = np.array(cam_cfg["K"], dtype=np.float32)
    T = np.array(cam_cfg["T_LIDAR_TO_CAM"], dtype=np.float32)

    result = run_lidar_odometry(
        npy_dir=str(npy_dir),
        output_dir=str(output_dir),
        image_dir = project_path / "image",
        K = K,
        T = T,
        max_frames = None,
        chunk_size = GLOBAL_STATE["chunks_number"],
    )

    return result

@app.post("/api/pointcloud/reproject")
def run_pointcloud_reproject_api():
    project_path = GLOBAL_STATE.get("project_path")
    if not project_path:
        raise HTTPException(400, "Project not initialized")

    result = run_pointcloud_reproject(project_path)
    return {
        "status": "ok",
        **result
    }

# ===============================
# Panoptic Viewer APIs
# ===============================
panoptic_task_core = PanopticTaskCore(GLOBAL_STATE)
def _find_image_file(img_dir: Path, frame: str):
    for ext in [".png", ".jpg", ".jpeg"]:
        p = img_dir / f"{frame}{ext}"
        if p.exists():
            return p.name
    raise HTTPException(404, f"Image not found for frame {frame}")


def _get_panoptic_dir():
    project_path = GLOBAL_STATE.get("project_path")
    if not project_path:
        raise HTTPException(400, "Project not initialized")

    p = Path(project_path) / "panoptic"
    if not p.is_dir():
        raise HTTPException(404, "panoptic folder not found")
    return p


def _get_image_dir():
    project_path = GLOBAL_STATE.get("project_path")
    return Path(project_path) / "image"

# 挂载 panoptic png（非常重要）
@app.on_event("startup")
def mount_panoptic_png():
    project_path = GLOBAL_STATE.get("project_path")
    if not project_path:
        return

    panoptic_dir = Path(project_path) / "panoptic"
    if panoptic_dir.is_dir():
        app.mount(
            "/panoptic_png",
            StaticFiles(directory=panoptic_dir),
            name="panoptic_png"
        )


@app.get("/api/panoptic/list")
def list_panoptic():
    pan_dir = _get_panoptic_dir()

    files = sorted(
        f.name.replace("_gtFine_panoptic.json", "")
        for f in pan_dir.glob("*_gtFine_panoptic.json")
    )

    return {"files": files, "total": len(files)}


@app.get("/api/panoptic/by-index")
def get_panoptic_by_index(index: int):
    project_path = GLOBAL_STATE.get("project_path")
    files = sorted((Path(project_path) / "panoptic").glob("*_gtFine_panoptic.json"))

    if index < 0 or index >= len(files):
        raise HTTPException(status_code=400, detail="Index out of range")

    path = files[index]

    with open(path, "r", encoding="utf-8") as f:
        panoptic = json.load(f)

    return {
        "index": index,
        "json_name": path.name,          # ⭐ 关键
        "frame": path.name.split("_")[0],
        "image_name": path.name.replace("_gtFine_panoptic.json", ".jpg"),
        "panoptic": panoptic,
        "total": len(files)
    }


@app.post("/api/panoptic/save")
def save_panoptic(payload: dict):
    json_name = payload.get("json_name")
    panoptic = payload.get("panoptic")
    project_path = GLOBAL_STATE.get("project_path")

    if not json_name or panoptic is None:
        raise HTTPException(status_code=400, detail="Invalid payload")

    panoptic_dir = Path(project_path) / "panoptic"
    path = panoptic_dir / json_name

    if not path.exists():
        raise HTTPException(status_code=404, detail="Panoptic file not found")

    # ⭐ 可选：备份
    backup = path.with_suffix(".bak")
    path.replace(backup)
    backup.replace(path)  # 或直接 copy

    # 覆盖写入
    with open(path, "w", encoding="utf-8") as f:
        json.dump(panoptic, f, indent=2, ensure_ascii=False)

    return {
        "status": "ok",
        "file": json_name
    }

@app.post("/api/panoptic/fastsam")
def run_fastsam_for_panoptic():
    project_path = GLOBAL_STATE.get("project_path")
    if not project_path:
        raise HTTPException(status_code=400, detail="Path not set")

    img_path = GLOBAL_STATE.get("current_image_path")
    if not img_path:
        raise HTTPException(status_code=400, detail="No current image")

    img_path = Path(img_path)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    # ⭐ 直接复用你已有的 Fast-SAM 推理
    ori_b64, sam_b64 = panoptic_task_core.process_first_image(img_path)

    img = Image.open(img_path)

    return {
        "image_size": {
            "width": img.width,
            "height": img.height
        },
        "ori_image": ori_b64,      # ⭐ 原图
        "sam_overlay": sam_b64     # ⭐ Fast-SAM 结果图
    }
@app.post("/api/panoptic/fastsam/query")
def query_fastsam_mask(payload: dict = Body(...)):
    x = int(payload["x"])
    y = int(payload["y"])

    masks = panoptic_task_core.current_masks_np
    if masks is None:
        return {"mask_id": None}

    # masks: [N, H, W]
    for i in range(masks.shape[0]):
        if masks[i, y, x]:
            return {"mask_id": i}

    return {"mask_id": None}



# 挂载前端
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")