// frontend/app.js 核心逻辑
import { nav, getRoute } from "./router.js";

let currentMode = "select"; // select | polygon
let polygonPoints = [];
let polygonCanvas = null;
let polygonCtx = null;
let lastMaskSource = null;



const labeledMasks = {};

window.nav = nav; // 暴露给 HTML 内部脚本

// ... 之前的 getRoute, nav 等保持不变 ...

async function render() {
  const route = getRoute();
  const appEl = document.getElementById("app");

  if (route === "/import") {
      await loadPage("./pages/import.html");
      if (typeof bindImportEvents === 'function') bindImportEvents(); 
  } 
  else if (route === "/image") {
      // 图像标注任务
      await loadPage("./pages/image.html");
      if (typeof bindImageAnnotationEvents === 'function') {
          bindImageAnnotationEvents(); // 这里会执行你刚拆解的 SAM 逻辑
      }
  }
  else if (route === "/panoptic-viewer") {
    await loadPage("./pages/panoptic_viewer.html");
    if (typeof initPanopticViewer === "function") {
      initPanopticViewer(); // 先留空，后面实现
    }
  }
  else if (route === "/tracking-editor") {
    await loadPage("./pages/tracking_editor.html");
    if (typeof initTrackingEditor === "function") {
      initTrackingEditor();
    }
  }  
  else if (route === "/pointcloud") {
      renderTaskPage("点云标注");
  } 
  else {
      renderDashboard();
  }
}

async function loadPage(url) {
  const res = await fetch(url);
  const html = await res.text();
  app.innerHTML = html;

  // 修复：手动提取并运行 HTML 中的 script 标签
  const scripts = app.querySelectorAll("script");
  scripts.forEach(oldScript => {
      const newScript = document.createElement("script");
      newScript.textContent = oldScript.textContent;
      document.body.appendChild(newScript).parentNode.removeChild(newScript);
  });
}

async function renderDashboard() {
    const res = await fetch("/api/get-context");
    const state = await res.json();
    window.currentPath = state.project_path;

    document.getElementById("app").innerHTML = `
        <div class="dashboard">
            <header class="header-banner">
                <h1>MCAF: A Multi-task Closed-loop Annotation Framework for Autonomous Driving v1.0</h1>
                <div class="path-badge">${state.project_path || 'Please select a data source'}</div>
            </header>
            <div class="task-grid">
                <div class="card" onclick="nav('/import')">
                    <h3>📂 Data Import</h3>
                    <p>Validate and configure project root directory</p>
                </div>
                <div class="card ${!state.project_path ? 'disabled' : ''}" onclick="nav('/image')">
                    <h3>🖼️ Image Segmentation</h3>
                    <p>Support 2D semantic/instance/panoptic sengentation</p>
                </div>
                <div class="card ${!state.project_path ? 'disabled' : ''}" onclick="nav('/panoptic-viewer')">
                    <h3>🧩 Panoptic Viewer</h3>
                    <p>Support revising panoptic sengentation & loading pretrained results</p>
                </div>
                <div
                  class="card ${!state.project_path ? 'disabled' : ''}"
                  onclick="runTrackingExport()"
                >
                    <h3>🎯 2D Tracking Output</h3>
                    <p>Transfer the panoptic results to COCO tracking format</p>
                </div>
                <div 
                  class="card ${!state.project_path ? 'disabled' : ''}"
                  onclick="runPointCloudSeg()"
                >
                    <h3>☁️ Point segmentation Output</h3>
                    <p>Support 3D point segmentation/object detection</p>
                </div>
                <div class="card" onclick="runLidarOdometry()">
                  <h3>🧭 Odometry Generation</h3>
                  <p>Generate LiDAR odometry results based on point segmentaion</p>
                </div>
                <div class="card" onclick="location.href='/pages/pc.html'">
                  <h3>🛠 Point Segmentation Revision</h3>
                  <p>Revise the group point segmentation/instances/bounding boxes</p>
                </div>
                <div class="card ${!state.project_path ? 'disabled' : ''}"
                    onclick="runPointcloudReproject()">
                  <h3>🔁 Point Segmentation/3D boxes one-by-one</h3>
                  <p>Apply labels from the last frame to all previous frames</p>
                </div>
                <div class="card" onclick="location.href='/pages/pce.html'">
                  <h3>🛠 Each Point Seg & 3D Box Revision</h3>
                  <p>Revise the each point segmentation/bounding boxes</p>
                </div>
            </div>
        </div>
    `;
}

window.addEventListener("hashchange", render);
window.onload = render;

// 绑定导入页面的逻辑
function bindImportEvents() {
  const btn = document.getElementById('btnVerify');
  const input = document.getElementById('pathInput');
  const result = document.getElementById('checkResult');

  if (!btn) return;

  btn.onclick = async () => {
      const path = input.value.trim();
      const res = await fetch('/api/import-path', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path })
      });
      
      const data = await res.json();
      if (res.ok) {
          result.innerHTML = `<div class="badge done">✓ Import Successfully: ${data.path}</div>`;
          setTimeout(() => nav("/"), 1500); // 成功后 1.5 秒自动回首页
      } else {
          result.innerHTML = `<div class="badge error">✘ Error: ${data.detail}</div>`;
      }
  };
}

// frontend/app.js 里的核心逻辑片段
async function bindImageAnnotationEvents() {
  const fileNameEl = document.getElementById('currentFileName');
  const baseImage = document.getElementById('baseImage');
  const segOverlay = document.getElementById('segOverlay');
  const btnPolygon = document.getElementById("btnPolygon");

  const polygonImage = document.getElementById("polygonImage");

  polygonCanvas = document.getElementById("polygonCanvas");
  if (!polygonCanvas) {
    console.warn("polygonCanvas not found");
    return;
  }
  
  polygonCtx = polygonCanvas.getContext("2d");

  polygonCanvas.addEventListener("click", (e) => {
    console.log("🔥 polygonCanvas clicked", e.clientX, e.clientY);
    if (currentMode !== "polygon") return;

    const rect = polygonCanvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * polygonCanvas.width / rect.width;
    const y = (e.clientY - rect.top) * polygonCanvas.height / rect.height;

    polygonPoints.push([Math.round(x), Math.round(y)]);
    redrawPolygon();
  });

  polygonCanvas.addEventListener("dblclick", (e) => {
    if (currentMode !== "polygon") return;
    e.preventDefault();   // 👈 很重要，防止 click 再触发一次
    finishPolygon();
  });



  if (!baseImage || !segOverlay) {
    console.error("Image DOM not ready");
    return;
  }

  // 初始模式
  setMode("select");

  document.getElementById("btnCursor").onclick = () => {
    setMode("select");
  };

  document.getElementById("btnPolygon").onclick = () => {
    setMode("polygon");
  };

  // document.addEventListener("click", e => {
  //   console.log(
  //     "GLOBAL CLICK:",
  //     e.target,
  //     "id=", e.target.id,
  //     "class=", e.target.className
  //   );
  // });
  

  // ===== 1️⃣ 绑定点击事件（查 mask）=====
  baseImage.addEventListener("click", async (e) => {
      const rect = baseImage.getBoundingClientRect();
      const x = Math.floor(
        (e.clientX - rect.left) * baseImage.naturalWidth / rect.width
      );
      const y = Math.floor(
        (e.clientY - rect.top) * baseImage.naturalHeight / rect.height
      );
      //console.log("GLOBAL CLICK:", e.target.id || e.target);
    
      // ===============================
      // 🖱️ 选择模式 → 查 SAM mask
      // ===============================
      if (currentMode === "select") {
        const res = await fetch("/api/image/query-mask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ x, y })
        });
    
        const data = await res.json();
        if (data.mask_id == null) {
          console.warn("❌ no mask");
          return;
        }
        lastMaskSource = "sam";
        showClassDialog(data.mask_id, data.class_options);
        return;
      }
    
      // ===============================
      // ⬢ 多边形模式 → 什么都不做（由 polygonCanvas 处理）
      // ===============================
      if (currentMode === "polygon") {
        console.log("polygon mode: ignore baseImage click");
        return;
      }
    });    

  // ===== 2️⃣ 加载第一张图 =====
  try {
    const res = await fetch('/api/image/init-first');
    if (!res.ok) throw new Error("Failed to load SAM results");

    const data = await res.json();

    fileNameEl.textContent = `Current File: ${data.filename}`;
    baseImage.src = `data:image/png;base64,${data.ori_image}`;
    segOverlay.src = `data:image/png;base64,${data.seg_image}`;
    if (polygonImage) {
      polygonImage.onload = () => {
        initPolygonCanvas();
      };
      polygonImage.src = baseImage.src;
      
    } else {
      console.warn("polygonImage not found, polygon disabled");
    }
  } catch (err) {
    console.error(err);
    alert("Loading failed. Please check backend logs and weight paths.");
  }

  btnPolygon.onclick = () => {
    //showPolygonPanel();
    setMode("polygon");
  };

  const btnSaveNext = document.getElementById("btnSaveNext");
  if (!btnSaveNext) {
    console.warn("btnSaveNext not found");
    return;
  }

  btnSaveNext.onclick = async () => {
    const res = await fetch("/api/image/save-and-next", {
      method: "POST"
    });

    const data = await res.json();
    if (!res.ok) {
      alert(data.detail || "Save Failure!");
      return;
    }
    
    fileNameEl.textContent = `Current File: ${data.filename}`;

    document.getElementById("baseImage").src =
      "data:image/png;base64," + data.ori_image;

    document.getElementById("segOverlay").src =
      "data:image/png;base64," + data.sam_overlay;
      
    document.getElementById("samResultImage").src =
      "data:image/png;base64," + data.matched_overlay;

    document.getElementById("polygonImage").src =
      "data:image/png;base64," + data.ori_image;
    
    document.getElementById("polygonResultImage").src =
      "data:image/png;base64," + data.polygon_overlay;

    // ⚠️ 清空旧帧所有标注
    Object.keys(labeledMasks).forEach(k => delete labeledMasks[k]);

    // 用后端给的 labels 重建
    data.labels.forEach(item => {
      labeledMasks[item.mask_id] = {
        class_id: item.class_id,
        class_name: item.class_name,
        instance_id: item.instance_id
      };
    });

    // 刷新右侧列表
    renderLabelList();


    console.log("✅ saved and next:", data);
  };

}

function showClassDialog(maskId, classOptions) {
  // 如果已经存在对话框，先移除
  const old = document.getElementById("class-dialog");
  if (old) old.remove();

  // 创建容器
  const dialog = document.createElement("div");
  dialog.id = "class-dialog";
  dialog.style.cssText = `
    position: fixed;
    top: 30%;
    left: 50%;
    transform: translateX(-50%);
    background: #1e1e1e;
    color: #fff;
    padding: 16px;
    border-radius: 8px;
    z-index: 9999;
    min-width: 260px;
    font-family: sans-serif;
    box-shadow: 0 0 12px rgba(0,0,0,0.5);
  `;

  // 构建下拉列表
  const optionsHtml = classOptions
    .map(c => `<option value="${c}">${c}</option>`)
    .join("");

  dialog.innerHTML = `
    <div style="margin-bottom: 8px; font-weight: bold;">
      选择类别（mask ${maskId}）
    </div>

    <select id="class-select" size="8"
      style="
        width: 100%;
        background: #2b2b2b;
        color: #fff;
        border: 1px solid #555;
        padding: 4px;
      ">
      ${optionsHtml}
    </select>

    <div style="margin-top: 10px; text-align: right;">
      <button id="class-ok" style="margin-right: 6px;">确定</button>
      <button id="class-cancel">取消</button>
    </div>
  `;

  document.body.appendChild(dialog);

  // 绑定按钮
  document.getElementById("class-ok").onclick = () => {
    const cls = document.getElementById("class-select").value;
    applyClass(maskId, cls);
    dialog.remove();
  };

  document.getElementById("class-cancel").onclick = () => {
    dialog.remove();
  };
}

// 在 app.js 中找个位置添加
function renderTaskPage(title) {
  const appEl = document.getElementById("app");
  appEl.innerHTML = `
      <div class="container">
          <nav style="margin-bottom: 20px;">
              <button class="btn" onclick="nav('/')">← 返回仪表盘</button>
          </nav>
          <div class="card">
              <h2>${title}</h2>
              <p>正在开发中...</p>
          </div>
      </div>
  `;
}

async function applyClass(maskId, className) {
  const res = await fetch("/api/image/set-mask-class", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mask_id: maskId,
      class_name: className
    })
  });

  const data = await res.json();
  if (!res.ok) {
    alert(data.detail || "设置失败");
    return;
  }

  if (data.sam_overlay) {
    updateResultImage(data.sam_overlay);
  }
  if (data.polygon_overlay) {
    updatePolygonResultImage(data.polygon_overlay);
  }
  // ===============================
  // ✅ 维护前端标注状态（第一步）
  // ===============================
  labeledMasks[maskId] = {
    class_id: data.class_id ?? null,
    class_name: className,
    instance_id: data.instance_id ?? null
  };

  renderLabelList();

  // ===============================
  // 情况 1：非 thing 类 → 直接完成
  // ===============================
  if (!data.need_instance) {
    //updateResultImage(data.overlay_image);
    //renderLabelList();   // ✅ 刷新右上角列表
    return;
  }

  // ===============================
  // 情况 2：thing 类 → 选实例
  // ===============================
  showInstanceDialog(data.existing_instances, async (choice) => {
    const res2 = await fetch("/api/image/set-mask-instance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mask_id: maskId,
        class_name: className,
        instance: choice
      })
    });

    const data2 = await res2.json();
    if (!res2.ok) {
      alert(data2.detail || "实例设置失败");
      return;
    }

    // ===============================
    // ✅ 更新 instance id
    // ===============================
    labeledMasks[maskId].instance_id = data2.instance_id;

    if (data2.sam_overlay) {
      updateResultImage(data2.sam_overlay);
    }
    if (data2.polygon_overlay) {
      updatePolygonResultImage(data2.polygon_overlay);
    }
    
    renderLabelList();   // ✅ 再次刷新列表
  });
}

function showInstanceDialog(existingInstances, onConfirm) {
  // 移除旧的
  const old = document.getElementById("instance-dialog");
  if (old) old.remove();

  const dialog = document.createElement("div");
  dialog.id = "instance-dialog";
  dialog.style.cssText = `
    position: fixed;
    top: 35%;
    left: 50%;
    transform: translateX(-50%);
    background: #1e1e1e;
    color: #fff;
    padding: 16px;
    border-radius: 8px;
    z-index: 10000;
    min-width: 260px;
    box-shadow: 0 0 12px rgba(0,0,0,0.5);
    font-family: sans-serif;
  `;

  const options = existingInstances
    .map(id => `<option value="${id}">${id}</option>`)
    .join("");

  dialog.innerHTML = `
    <div style="margin-bottom:8px;font-weight:bold;">
      选择实例 ID
    </div>

    <select id="instance-select" size="6"
      style="
        width:100%;
        background:#2b2b2b;
        color:#fff;
        border:1px solid #555;
        padding:4px;
      ">
      ${options}
      <option value="new">➕ new instance</option>
    </select>

    <div style="margin-top:10px;text-align:right;">
      <button id="instance-ok">确定</button>
      <button id="instance-cancel" style="margin-left:6px;">取消</button>
    </div>
  `;

  document.body.appendChild(dialog);

  document.getElementById("instance-ok").onclick = () => {
    const value = document.getElementById("instance-select").value;
    dialog.remove();
    onConfirm(value);
  };

  document.getElementById("instance-cancel").onclick = () => {
    dialog.remove();
  };
}


function updateResultImage(b64) {
  const img = document.getElementById("samResultImage");
  if (!img) {
    console.error("samResultImage not found in DOM");
    return;
  }

  img.src = `data:image/png;base64,${b64}`;
  img.style.display = "block";
}

function updatePolygonResultImage(b64) {
  const img = document.getElementById("polygonResultImage");
  if (!img) {
    console.error("polygonResultImage not found in DOM");
    return;
  }

  img.src = `data:image/png;base64,${b64}`;
  img.style.display = "block";
}


function renderLabelList() {
  const ul = document.getElementById("labelList");
  if (!ul) return;

  ul.innerHTML = "";

  Object.entries(labeledMasks).forEach(([maskId, info]) => {
    const li = document.createElement("li");

    li.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div>
          <b>mask ${maskId}</b><br/>
          class: ${info.class_id} (${info.class_name})<br/>
          inst: ${info.instance_id ?? "-"}
        </div>
        <button class="del-btn" title="删除">✖</button>
      </div>
    `;

    // ===== 绑定删除 =====
    li.querySelector(".del-btn").onclick = () => {
      deleteMask(maskId);
    };

    ul.appendChild(li);
  });
}

async function deleteMask(maskId) {
  const ok = confirm(`确定删除 mask ${maskId} 的标注？`);
  if (!ok) return;

  const res = await fetch("/api/image/delete-mask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mask_id: maskId })
  });

  const data = await res.json();
  if (!res.ok) {
    alert(data.detail || "删除失败");
    return;
  }

  // ===== 更新前端状态 =====
  delete labeledMasks[maskId];
  renderLabelList();

  // ===== 更新 overlay =====
  if (data.sam_overlay) {
    updateResultImage(data.sam_overlay);
  }
  if (data.polygon_overlay) {
    updatePolygonResultImage(data.polygon_overlay);
  }
  

  // ===== 更新右上角列表 =====
  renderLabelList();
}

// document.getElementById("btnPolygon").onclick = () => {
//   currentMode = "polygon";
//   showPolygonPanel();
// };

function showPolygonPanel() {
  const panel = document.getElementById("polygonPanel");
  panel.style.display = "block";

  const canvas = document.getElementById("polygonCanvas");
  const img = document.getElementById("baseImage");

  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  canvas.style.width = "300px";
  canvas.style.height = `${300 * img.naturalHeight / img.naturalWidth}px`;

  polygonPoints = [];

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0,0,canvas.width,canvas.height);
}

function redrawPolygon(close = false) {
  if (!polygonCtx || !polygonCanvas) return;

  if (polygonCtx && polygonCanvas) {
    polygonCtx.clearRect(0, 0, polygonCanvas.width, polygonCanvas.height);
  }

  if (polygonPoints.length === 0) return;

  // 画线
  polygonCtx.strokeStyle = "#22c55e";
  polygonCtx.lineWidth = 2;
  polygonCtx.beginPath();

  polygonPoints.forEach(([x, y], i) => {
    if (i === 0) polygonCtx.moveTo(x, y);
    else polygonCtx.lineTo(x, y);
  });

  if (close) {
    polygonCtx.lineTo(polygonPoints[0][0], polygonPoints[0][1]);
  }

  polygonCtx.stroke();

  // 画点
  polygonCtx.fillStyle = "#22c55e";
  polygonPoints.forEach(([x, y]) => {
    polygonCtx.beginPath();
    polygonCtx.arc(x, y, 4, 0, Math.PI * 2);
    polygonCtx.fill();
  });
}

async function finishPolygon() {
  if (polygonPoints.length < 3) {
    alert("至少需要 3 个点");
    return;
  }

  // 视觉闭环
  redrawPolygon(true);

  const res = await fetch("/api/image/polygon-create-mask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      polygon_points: polygonPoints
    })
  });

  const data = await res.json();
  if (!res.ok) {
    alert("Polygon mask 创建失败");
    return;
  }

  // ✅ 立刻清空绘制区，准备下一个
  polygonPoints = [];
  if (polygonCtx && polygonCanvas) {
    polygonCtx.clearRect(0, 0, polygonCanvas.width, polygonCanvas.height);
  }

  // ✅ 和 SAM 一样：弹类别 / 实例
  lastMaskSource = "polygon";
  showClassDialog(data.mask_id, data.class_options);
}

function initPolygonCanvas() {
  const polygonImg = document.getElementById("polygonImage");
  const polygonCanvas = document.getElementById("polygonCanvas");

  const rect = polygonImg.getBoundingClientRect();

  polygonCanvas.width  = polygonImg.naturalWidth;
  polygonCanvas.height = polygonImg.naturalHeight;

  polygonCanvas.style.width  = rect.width + "px";
  polygonCanvas.style.height = rect.height + "px";

  console.log(
    "✅ polygon canvas ready:",
    polygonCanvas.width,
    polygonCanvas.height,
    "display:",
    rect.width,
    rect.height
  );
}

function setMode(mode) {
  currentMode = mode;

  const btnCursor = document.getElementById("btnCursor");
  const btnPolygon = document.getElementById("btnPolygon");

  btnCursor.classList.toggle("active", mode === "select");
  btnPolygon.classList.toggle("active", mode === "polygon");

  // ✅ 永远显示 polygonPanel
  const polygonPanel = document.getElementById("polygonPanel");
  if (polygonPanel) {
    polygonPanel.style.display = "block";
  }

  // ✅ 只控制是否响应鼠标
  if (polygonCanvas) {
    polygonCanvas.style.pointerEvents =
      mode === "polygon" ? "auto" : "none";
  }

  console.log("🔧 mode switched to:", mode);
}

async function runTrackingExport() {
  if (!confirm("将从 panoptic 结果生成 2D Tracking 数据，是否继续？")) {
    return;
  }

  try {
    const res = await fetch("/api/tracking/generate", {
      method: "POST"
    });

    const data = await res.json();

    if (!res.ok) {
      alert(data.detail || "Tracking 生成失败");
      return;
    }

    alert(
      `✅ Tracking 生成完成\n\n` +
      `输出文件：${data.output}\n` +
      `图像数量：${data.num_images}`
    );
    nav("/tracking-editor");
  } catch (e) {
    console.error(e);
    alert("请求失败，请检查后端日志");
  }
}
window.runTrackingExport = runTrackingExport;

// pc.js
async function runPointCloudSeg() {
  const ok = confirm("将使用 panoptic 结果生成 3D 点云标注，是否继续？");
  if (!ok) return;

  const res = await fetch("/api/pointcloud/run", { method: "POST" });
  const data = await res.json();

  alert("✅ 点云分割完成\nFrames: " + data.frames);
}
window.runPointCloudSeg = runPointCloudSeg;

async function runLidarOdometry() {
  if (!confirm("将运行 LiDAR 里程计并生成全局地图，是否继续？")) {
    return;
  }

  try {
    const res = await fetch("/api/pointcloud/odometry", {
      method: "POST"
    });

    const data = await res.json();

    if (!res.ok) {
      alert(data.detail || "里程计生成失败");
      return;
    }

    alert(
      "✅ 里程计生成完成\n\n" +
      `帧数: ${data.frames}\n` +
      `输出目录:\n${data.output_dir}`
    );
  } catch (e) {
    console.error(e);
    alert("请求失败，请检查后端日志");
  }
}
window.runLidarOdometry = runLidarOdometry;

async function runPointcloudReproject() {
  if (!confirm("将执行点云回投与 3D 框还原，是否继续？")) return;

  try {
    const res = await fetch("/api/pointcloud/reproject", {
      method: "POST"
    });
    const data = await res.json();

    if (!res.ok) {
      alert(data.detail || "执行失败");
      return;
    }

    alert(
      `✅ 回投完成\n\n` +
      `Frames: ${data.frames}\n` +
      `Boxes: ${data.boxes_dir}\n` +
      `Points: ${data.points_dir}`
    );
  } catch (e) {
    console.error(e);
    alert("请求失败，请查看后端日志");
  }
}
window.runPointcloudReproject = runPointcloudReproject;


// ===============================
// Panoptic Viewer State
// ===============================

let panopticFiles = [];
let currentPanopticIndex = 0;
let currentPanopticData = null;

let editingSegment = null;     // 当前编辑的 segment
let editingPolygon = null;     // 当前编辑的 polygon（引用）
let editingPolygonIndex = null; 
let controlPoints = [];        // [{x,y, idx}]
let activePointIndex = null;   // 当前拖动的点 index

// ===== Draw mode =====
let drawMode = false;
let drawingPoints = [];   // [[x,y], ...]
let hoverPoint = null;    // 当前鼠标位置

let undoStack = [];
const UNDO_LIMIT = 20;
let panopticClassConfig = null;
let panopticThingClasses = new Set();

let PROJECT_CONFIG = null;

let currentPanopticJsonName = null;

let fastSamResult = null;
let selectedFastSamMask = null;

// ===== Fast-SAM Preview =====
let canvasFastSam = null;
let ctxFastSam = null;

let fastSamMasks = [];     // [{ mask, polygon, mask_id }]
let fastSamImageSize = null;
let fastSamBaseImage = null;


async function initPanopticViewer() {
  console.log("🧩 initPanopticViewer called");

    // ===============================
  // Load project config
  // ===============================
  if (!PROJECT_CONFIG) {
    try {
      const res = await fetch("/api/config");
      if (!res.ok) throw new Error("failed to load config");
      PROJECT_CONFIG = await res.json();
      console.log("✅ project config loaded:", PROJECT_CONFIG);
    } catch (e) {
      alert("Failed to load project config");
      console.error(e);
      return;
    }
  }


  const img = document.getElementById("pvImage");
  const canvas = document.getElementById("pvCanvas");
  const ctx = canvas.getContext("2d");
  const SMOOTH_RADIUS = 30;   // 影响的点数量（越大越柔）
  const SIGMA = 0.4;         // 平滑程度（0.3~0.6 很好）
  const CONTROL_POINT_COUNT = 40;

  //const ctxFastSam = canvasFastSam?.getContext("2d");
  
  canvasFastSam = document.getElementById("fastSamCanvas");

  if (canvasFastSam) {
    ctxFastSam = canvasFastSam.getContext("2d");
  }

  if (!img || !canvas) {
    console.error("❌ panoptic viewer DOM not ready");
    return;
  }

  // 1️⃣ 拉取 panoptic 文件列表
  const listRes = await fetch("/api/panoptic/list");
  const listData = await listRes.json();

  panopticFiles = listData.files;
  if (!panopticFiles.length) {
    alert("No panoptic json files found");
    return;
  }

  // 2️⃣ 绑定按钮
  document.getElementById("btnPrev").onclick = () => {
    if (currentPanopticIndex > 0) {
      loadPanopticByIndex(currentPanopticIndex - 1);
    }
  };

  document.getElementById("btnNext").onclick = () => {
    if (currentPanopticIndex < panopticFiles.length - 1) {
      loadPanopticByIndex(currentPanopticIndex + 1);
    }
  };

  document.getElementById("btnSave").onclick = () => {
    saveCurrentPanoptic();
  };

  document.getElementById("btnDeleteMask").onclick = () => {
    deleteCurrentMask();
  };

  document.getElementById("btnDraw").onclick = () => {
    drawMode = !drawMode;
    drawingPoints = [];
    hoverPoint = null;
  
    document.getElementById("btnDraw").classList.toggle("active", drawMode);
    console.log("Draw mode:", drawMode);
  
    drawPanoptic(ctx, canvas);
  };
  

  document.addEventListener("keydown", e => {
    if (e.key === "Delete" || e.key === "Backspace") {
      deleteCurrentMask();
    }
  });  

  document.addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "z") {
      e.preventDefault();
      undoLast();
    }
  });
  

  // 3️⃣ 默认加载第一帧
  await loadPanopticByIndex(0);

  async function loadPanopticByIndex(index) {
    const res = await fetch(`/api/panoptic/by-index?index=${index}`);
    const data = await res.json();
  
    currentPanopticIndex = data.index;
    currentPanopticData = data.panoptic;

    currentPanopticJsonName = data.json_name;
    renderMaskList();
  
    document.getElementById("frameInfo").textContent =
      `Frame ${data.frame} (${index + 1}/${data.total})`;
  
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.style.width = img.clientWidth + "px";
      canvas.style.height = img.clientHeight + "px";
      drawPanoptic(ctx, canvas);

      if (canvasFastSam) {
        canvasFastSam.width  = img.naturalWidth;
        canvasFastSam.height = img.naturalHeight;
      
        canvasFastSam.style.width  = img.clientWidth + "px";
        canvasFastSam.style.height = img.clientHeight + "px";
      }
      fastSamBaseImage = img;   // ⭐ 右侧复用同一张图
      loadFastSamPreview();
    };
  
    img.src = `/project_image/${data.image_name}`;
  }
  
  function pushUndoState() {
    if (!currentPanopticData) return;
  
    // 深拷贝（非常重要）
    const snapshot = JSON.parse(JSON.stringify(currentPanopticData));
  
    undoStack.push(snapshot);
  
    // 限制长度，防止爆内存
    if (undoStack.length > UNDO_LIMIT) {
      undoStack.shift();
    }
  
    console.log("🕘 undo push, depth =", undoStack.length);
  }

  function undoLast() {
    if (!undoStack.length) {
      alert("Nothing to undo");
      return;
    }
  
    const prev = undoStack.pop();
    currentPanopticData = prev;
  
    // 清空编辑状态
    editingSegment = null;
    editingPolygon = null;
    controlPoints = [];
    activePointIndex = null;
  
    drawPanoptic(
      document.getElementById("pvCanvas").getContext("2d"),
      document.getElementById("pvCanvas")
    );
    renderMaskList();
  
    console.log("↩️ undo");
  }
  
  function drawPanoptic(ctx, canvas) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  
    if (!currentPanopticData) return;
  
    const segments = currentPanopticData.segments_info || [];
  
    segments.forEach(seg => {
      // seg.polygon: [poly1, poly2, ...]
      // poly: [[x,y], [x,y], ...]
      const polysRaw = seg.polygon;
  
      if (!Array.isArray(polysRaw) || polysRaw.length === 0) return;
  
      const isEditing = (editingSegment === seg);
      const color = idToColor(seg.id);
  
      // 收集所有点，用于算 centroid（支持多个不连通区域）
      const allPoints = [];
  
      polysRaw.forEach(polyRaw => {
        const poly = normalizePolygon(polyRaw);
        if (!poly || poly.length < 3) return;
  
        poly.forEach(p => allPoints.push(p));
  
        // ===== 画 mask =====
        ctx.beginPath();
        poly.forEach(([x, y], i) => {
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.closePath();
  
        ctx.fillStyle = isEditing
          ? color.replace("rgb", "rgba").replace(")", ",0.55)")
          : color.replace("rgb", "rgba").replace(")", ",0.35)");
        ctx.fill();
  
        ctx.strokeStyle = isEditing ? "#00ffff" : color;
        ctx.lineWidth = isEditing ? 3 : 2;
        ctx.stroke();
      });
  
      // ===============================
      // ✅ 仅编辑状态：显示类别 / 实例信息
      // ===============================
      if (isEditing && allPoints.length >= 3) {
        const { x, y } = polygonCentroid(allPoints);
  
        const className = seg.category_name ?? "unknown";
        const classId = seg.class_id ?? "-";
        const instId = seg.instance_id ?? "–";
        const label = `${className}  (cid:${classId}, iid:${instId})`;
  
        ctx.font = "bold 36px system-ui";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
  
        // 背景
        const padding = 6;
        const metrics = ctx.measureText(label);
        const w = metrics.width + padding * 2;
        const h = 40;
  
        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillRect(x - w / 2, y - h / 2, w, h);
  
        // 文字
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, x, y);
      }
    });
  
    // ===== 控制点（只依赖 editingPolygon）=====
    if (editingPolygon && controlPoints.length) {
      controlPoints.forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = "#00ffff";
        ctx.fill();
        ctx.strokeStyle = "#000";
        ctx.stroke();
      });
    }
  
    // ===============================
    // ✏️ Draw mode preview
    // ===============================
    if (drawMode && drawingPoints.length) {
      ctx.save();
  
      ctx.beginPath();
      drawingPoints.forEach(([x, y], i) => {
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
  
      // 预览到鼠标
      if (hoverPoint) {
        ctx.lineTo(hoverPoint.x, hoverPoint.y);
      }
  
      ctx.strokeStyle = "#00ff88";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.stroke();
  
      // ===== ⭐ 显示“即将闭合”的边 =====
      if (drawingPoints.length >= 2) {
        const [x0, y0] = drawingPoints[0];
        const [xLast, yLast] = drawingPoints[drawingPoints.length - 1];
  
        ctx.beginPath();
        ctx.moveTo(xLast, yLast);
        ctx.lineTo(x0, y0);
        ctx.strokeStyle = "rgba(0,255,136,0.5)";
        ctx.setLineDash([4, 6]);
        ctx.stroke();
      }
  
      // 控制点
      drawingPoints.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = "#00ff88";
        ctx.fill();
      });
  
      ctx.restore();
    }
  }
  
  
  function polygonCentroid(poly) {
    let x = 0, y = 0;
    poly.forEach(p => {
      x += p[0];
      y += p[1];
    });
    return {
      x: x / poly.length,
      y: y / poly.length
    };
  }
  
  function idToColor(id) {
    const r = (id * 37) % 255;
    const g = (id * 59) % 255;
    const b = (id * 83) % 255;
    return `rgb(${r},${g},${b})`;
  }

  async function saveCurrentPanoptic() {
    if (!currentPanopticData || !currentPanopticJsonName) {
      alert("No panoptic data to save");
      return;
    }
  
    try {
      const res = await fetch("/api/panoptic/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          json_name: currentPanopticJsonName,   // ⭐ 核心
          panoptic: currentPanopticData
        })
      });
  
      const data = await res.json();
  
      if (!res.ok) {
        alert(data.detail || "Save failed");
        return;
      }
  
      // 保存成功反馈
      alert(`✅ Saved: ${currentPanopticJsonName}`);
      undoStack = []; // ⭐ 保存后清空 undo
  
    } catch (e) {
      console.error(e);
      alert("Save error, check backend");
    }
  }

  async function showPanopticClassDialog() {
    if (!panopticClassConfig) return null;
  
    const categories = Object.keys(panopticClassConfig);
  
    // ===== 1️⃣ 选择类别 =====
    const category = prompt(
      `Choose category:\n\n${categories.join("\n")}`,
      categories[0]
    );
    if (!category) return null;
  
    const cfg = panopticClassConfig[category];
    const class_id = cfg.class_id;
    const isthing = cfg.isthing;
  
    // ===== 2️⃣ 非 thing：直接返回 =====
    if (!isthing) {
      return {
        category_name: category,
        class_id,
        isthing: false,
        instance_id: null
      };
    }
  
    // ===== 3️⃣ thing：统计已有 instance =====
    const segments = currentPanopticData.segments_info || [];
    const used = segments
      .filter(s => s.category_name === category && s.isthing)
      .map(s => s.instance_id)
      .filter(v => v != null);
  
    // ===== 4️⃣ 新类别：自动分配 =====
    if (used.length === 0) {
      return {
        category_name: category,
        class_id,
        isthing: true,
        instance_id: class_id * 1000 + 1
      };
    }
  
    // ===== 5️⃣ 已有类别：让用户选 =====
    const maxId = Math.max(...used);
    const choice = prompt(
      `Category "${category}" already has instances:\n\n` +
      `Existing: ${used.join(", ")}\n\n` +
      `Enter:\n` +
      `- "new" → new instance (${maxId + 1})\n` +
      `- or an existing instance id`,
      "new"
    );
  
    if (choice === null) return null;
  
    let instance_id;
    if (choice === "new") {
      instance_id = maxId + 1;
    } else {
      const v = parseInt(choice);
      if (!used.includes(v)) {
        alert("Invalid instance id");
        return null;
      }
      instance_id = v;
    }
  
    return {
      category_name: category,
      class_id,
      isthing: true,
      instance_id
    };
  }
  
  async function askCategoryAndInstance() {
    const cat = await showPanopticCategoryDialog();
    if (!cat) return null;
  
    if (!cat.isthing) {
      return {
        category_name: cat.category_name,
        class_id: cat.class_id,
        isthing: false,
        instance_id: null
      };
    }
  
    const instance_id = await showPanopticInstanceDialog(
      cat.category_name,
      cat.class_id
    );
    if (instance_id == null) return null;
  
    return {
      category_name: cat.category_name,
      class_id: cat.class_id,
      isthing: true,
      instance_id
    };
  }

  canvas.addEventListener("dblclick", e => {
    const { x, y } = getCanvasXY(e, canvas);
  
    const hit = findHitSegment(x, y);
    if (!hit) return;
  
    enterEditMode(hit);
  });
  // ===============================
  // 拖拽控制点
  // ===============================
  canvas.addEventListener("mousedown", e => {
    if (!editingPolygon || !controlPoints.length) return;

    const { x, y } = getCanvasXY(e, canvas);

    activePointIndex = null;
    controlPoints.forEach((p, i) => {
      const d = Math.hypot(p.x - x, p.y - y);
      if (d < 8) {
        activePointIndex = i;
      }
    });
  });

  canvas.addEventListener("mousemove", e => {
    if (activePointIndex === null) return;
  
    const { x, y } = getCanvasXY(e, canvas);
    const cp = controlPoints[activePointIndex];
  
    const dx = x - cp.x;
    const dy = y - cp.y;
  
    cp.x = x;
    cp.y = y;
  
    const centerIdx = cp.polyIndex;
    smoothDeform(centerIdx, dx, dy);
  
    drawPanoptic(ctx, canvas);
  });
  
  

  canvas.addEventListener("mouseup", () => {
    if (activePointIndex !== null) {
      pushUndoState();
    }
    activePointIndex = null;
    renderMaskList();
  });

  canvas.addEventListener("click", e => {
    if (!drawMode) return;
  
    const { x, y } = getCanvasXY(e, canvas);
    drawingPoints.push([x, y]);
    drawPanoptic(ctx, canvas);
  });

  canvas.addEventListener("mousemove", e => {
    if (!drawMode) return;
  
    hoverPoint = getCanvasXY(e, canvas);
    drawPanoptic(ctx, canvas);
  });

  canvas.addEventListener("dblclick", e => {
    if (!drawMode) return;
  
    e.preventDefault();
  
    if (drawingPoints.length < 3) {
      alert("Polygon needs at least 3 points");
      return;
    }
  
    finishDrawPolygon();
  });
    
  async function finishDrawPolygon() {
    drawMode = false;
    hoverPoint = null;
    document.getElementById("btnDraw").classList.remove("active");
  
    if (drawingPoints.length < 3) {
      drawingPoints = [];
      drawPanoptic(ctx, canvas);
      return;
    }
  
    // ⭐ 从 config + 现有帧逻辑中选择类别 / 实例
    const result = await askCategoryAndInstance();
    if (!result) {
      drawingPoints = [];
      drawPanoptic(ctx, canvas);
      return;
    }
  
    // ✅ 这里一定要包含 class_id
    const { category_name, class_id, isthing, instance_id } = result;
  
    const newId = generateNewSegmentId();
  
    const poly = drawingPoints.map(p => [p[0], p[1]]);
    const [x0, y0] = poly[0];
    const [xN, yN] = poly[poly.length - 1];
    if (x0 !== xN || y0 !== yN) poly.push([x0, y0]);
  
    const newSeg = {
      id: newId,
      category_name,
      class_id,                 // ✅ 现在是合法变量
      isthing,
      instance_id,
      polygon: poly
    };
  
    pushUndoState();
    currentPanopticData.segments_info.push(newSeg);
  
    editingSegment = newSeg;
    editingPolygon = newSeg.polygon;
    controlPoints = [];
    activePointIndex = null;
    drawingPoints = [];
  
    drawPanoptic(ctx, canvas);
    renderMaskList();
  }

  function showPanopticCategoryDialog() {
    return new Promise(resolve => {
      const old = document.getElementById("panoptic-class-dialog");
      if (old) old.remove();
  
      const { class_name_to_id, thing_classes } = PROJECT_CONFIG;
      const classes = Object.keys(class_name_to_id);
  
      const dialog = document.createElement("div");
      dialog.id = "panoptic-class-dialog";
      dialog.style.cssText = `
        position: fixed;
        top: 30%;
        left: 50%;
        transform: translateX(-50%);
        background: #1e1e1e;
        color: #fff;
        padding: 16px;
        border-radius: 8px;
        z-index: 10000;
        min-width: 280px;
        max-height: 420px;
        overflow: auto;
        box-shadow: 0 0 12px rgba(0,0,0,0.5);
        font-family: system-ui;
      `;
  
      dialog.innerHTML = `
        <div style="font-weight:bold;margin-bottom:8px;">
          Select Category
        </div>
        <ul id="panoptic-class-list"
            style="list-style:none;padding:0;margin:0;">
          ${classes.map(c => `
            <li data-name="${c}"
                style="
                  padding:6px 8px;
                  cursor:pointer;
                  border-bottom:1px solid #333;
                ">
              ${c}
              ${thing_classes.includes(c) ? " 🟢" : ""}
            </li>
          `).join("")}
        </ul>
        <div style="text-align:right;margin-top:8px;">
          <button id="panoptic-class-cancel">Cancel</button>
        </div>
      `;
  
      document.body.appendChild(dialog);
  
      // 点击类别
      dialog.querySelectorAll("li").forEach(li => {
        li.onclick = () => {
          const category = li.dataset.name;
          dialog.remove();
  
          resolve({
            category_name: category,
            class_id: class_name_to_id[category],
            isthing: thing_classes.includes(category)
          });
        };
      });
  
      document.getElementById("panoptic-class-cancel").onclick = () => {
        dialog.remove();
        resolve(null);
      };
    });
  }
  
  function showPanopticInstanceDialog(category, class_id) {
    return new Promise(resolve => {
      const segments = currentPanopticData.segments_info || [];
      const used = segments
        .filter(s => s.category_name === category && s.instance_id != null)
        .map(s => s.instance_id);
  
      // ⭐ 如果是这个类别的第一个 instance
      if (used.length === 0) {
        resolve(class_id * 1000 + 1);
        return;
      }
  
      const old = document.getElementById("panoptic-instance-dialog");
      if (old) old.remove();
  
      const dialog = document.createElement("div");
      dialog.id = "panoptic-instance-dialog";
      dialog.style.cssText = `
        position: fixed;
        top: 35%;
        left: 50%;
        transform: translateX(-50%);
        background: #1e1e1e;
        color: #fff;
        padding: 16px;
        border-radius: 8px;
        z-index: 10001;
        min-width: 260px;
        box-shadow: 0 0 12px rgba(0,0,0,0.5);
        font-family: system-ui;
      `;
  
      const maxId = Math.max(...used);
  
      dialog.innerHTML = `
        <div style="font-weight:bold;margin-bottom:8px;">
          Select Instance (${category})
        </div>
  
        <ul style="list-style:none;padding:0;margin:0;">
          ${used.map(id => `
            <li data-id="${id}"
                style="padding:6px;cursor:pointer;">
              Instance ${id}
            </li>
          `).join("")}
  
          <li data-id="new"
              style="padding:6px;cursor:pointer;color:#00ff88;">
            ➕ New instance (${maxId + 1})
          </li>
        </ul>
  
        <div style="text-align:right;margin-top:8px;">
          <button id="instance-cancel">Cancel</button>
        </div>
      `;
  
      document.body.appendChild(dialog);
  
      dialog.querySelectorAll("li").forEach(li => {
        li.onclick = () => {
          const v = li.dataset.id;
          dialog.remove();
          resolve(v === "new" ? maxId + 1 : parseInt(v));
        };
      });
  
      document.getElementById("instance-cancel").onclick = () => {
        dialog.remove();
        resolve(null);
      };
    });
  }  
  
  function generateNewSegmentId() {
    const ids = currentPanopticData.segments_info.map(s => s.id);
    return ids.length ? Math.max(...ids) + 1 : 1;
  }
  
  function polygonCentroid(poly) {
    let x = 0, y = 0;
    poly.forEach(p => {
      x += p[0];
      y += p[1];
    });
    return {
      x: x / poly.length,
      y: y / poly.length
    };
  }
  

  function getCanvasXY(e, canvas) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * canvas.width / rect.width,
      y: (e.clientY - rect.top) * canvas.height / rect.height
    };
  }
  
  function normalizePolygon(polygon) {
    if (!polygon) return null;
    if (Array.isArray(polygon[0][0])) return polygon[0];
    return polygon;
  }
  
  function findHitSegment(x, y) {
    const segments = currentPanopticData.segments_info || [];
  
    for (const seg of segments) {
      const poly = normalizePolygon(seg.polygon);
      if (!poly || poly.length < 3) continue;
  
      if (pointInPolygon(x, y, poly)) {
        return { seg, poly };
      }
    }
    return null;
  }
  
  function pointInPolygon(x, y, poly) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const xi = poly[i][0], yi = poly[i][1];
      const xj = poly[j][0], yj = poly[j][1];
  
      const intersect =
        ((yi > y) !== (yj > y)) &&
        (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
  
      if (intersect) inside = !inside;
    }
    return inside;
  }
  function enterEditMode(hit) {
    editingSegment = hit.seg;
    editingPolygon = hit.poly;
    editingPolygonIndex = hit.polyIndex ?? 0; 
  
    // 生成控制点（最多 20 个，或原始点数）
    const step = Math.max(1, Math.floor(editingPolygon.length / CONTROL_POINT_COUNT));
  
    controlPoints = editingPolygon
      .filter((_, i) => i % step === 0)
      .map((p, idx) => ({
        x: p[0],
        y: p[1],
        polyIndex: idx * step
      }));
  
    drawPanoptic(ctx, canvas);
    renderMaskList(); 
  }

  function smoothDeform(centerIdx, dx, dy) {
    const poly = editingPolygon;
    const N = poly.length;
  
    for (let offset = -SMOOTH_RADIUS; offset <= SMOOTH_RADIUS; offset++) {
      const idx = (centerIdx + offset + N) % N;
  
      const t = offset / SMOOTH_RADIUS;  // -1 → 1
      const w = Math.exp(- (t * t) / (2 * SIGMA * SIGMA));
      // ↑ Gaussian 权重（非常平滑）
  
      poly[idx][0] += dx * w;
      poly[idx][1] += dy * w;
    }
  }

  function renderMaskList() {
    const ul = document.getElementById("maskList");
    if (!ul || !currentPanopticData) return;
  
    ul.innerHTML = "";
  
    const segments = currentPanopticData.segments_info || [];
  
    segments.forEach(seg => {
      const polysRaw = seg.polygon;
      if (!Array.isArray(polysRaw) || polysRaw.length === 0) return;
  
      polysRaw.forEach((polyRaw, polyIndex) => {
        const poly = normalizePolygon(polyRaw);
        if (!poly || poly.length < 3) return;
  
        const li = document.createElement("li");
  
        // ⭐ 注意：编辑状态现在要同时匹配 seg + polyIndex
        const isEditing =
          editingSegment === seg && editingPolygonIndex === polyIndex;
  
        if (isEditing) li.classList.add("active");
  
        li.innerHTML = `
          <div><b>ID:</b> ${seg.id}</div>
          <div><b>Class:</b> ${seg.category_name || "-"}</div>
          <div><b>Poly:</b> ${polyIndex + 1}/${polysRaw.length}</div>
          <div><b>Instance:</b> ${seg.isthing ? (seg.instance_id ?? "-") : "-"}</div>
          <div><b>Points:</b> ${poly.length}</div>
          ${isEditing ? `<div style="color:#00ffff">[editing]</div>` : ""}
        `;
  
        // 👉 点击列表 = 进入编辑模式（编辑指定的 polygon）
        li.onclick = () => {
          enterEditMode({ seg, poly, polyIndex });
        };
  
        ul.appendChild(li);
      });
    });
  }
  
  
  // function deleteCurrentMask() {
  //   if (!editingSegment || !currentPanopticData) {
  //     alert("No mask selected");
  //     return;
  //   }

  //   pushUndoState();
  
  //   const ok = confirm(
  //     `Delete mask ID ${editingSegment.id} (${editingSegment.category_name}) ?`
  //   );
  //   if (!ok) return;
  
  //   // 1️⃣ 从 segments_info 中移除
  //   const segments = currentPanopticData.segments_info;
  //   const idx = segments.indexOf(editingSegment);
  
  //   if (idx >= 0) {
  //     segments.splice(idx, 1);
  //   }
  
  //   // 2️⃣ 清空编辑状态
  //   editingSegment = null;
  //   editingPolygon = null;
  //   controlPoints = [];
  //   activePointIndex = null;
  
  //   // 3️⃣ 重新绘制
  //   drawPanoptic(ctx, canvas);
  
  //   // 4️⃣ 更新右侧列表
  //   renderMaskList();
  // }

  function deleteCurrentMask() {
    if (!editingSegment || !currentPanopticData) {
      alert("No mask selected");
      return;
    }
  
    pushUndoState();
  
    const seg = editingSegment;
    const segments = currentPanopticData.segments_info;
  
    // seg.polygon 必须是数组： [poly0, poly1, ...]
    if (!Array.isArray(seg.polygon) || seg.polygon.length === 0) {
      alert("Segment has no polygon data");
      return;
    }
  
    const polyCount = seg.polygon.length;
    const pidx = editingPolygonIndex ?? 0;
  
    // 防止越界
    if (pidx < 0 || pidx >= polyCount) {
      alert(`Invalid polygon index: ${pidx}`);
      return;
    }
  
    // ✅提示信息：删的是整个 segment 还是某个 polygon
    const msg =
      polyCount > 1
        ? `Delete polygon ${pidx + 1}/${polyCount} of mask ID ${seg.id} (${seg.category_name}) ?`
        : `Delete mask ID ${seg.id} (${seg.category_name}) ?`;
  
    const ok = confirm(msg);
    if (!ok) return;
  
    // ==========================
    // ✅ case 1: 多 polygon → 只删当前 polygon
    // ==========================
    if (polyCount > 1) {
      seg.polygon.splice(pidx, 1);
  
      // 清空编辑状态（也可以切到剩下的某个 polygon）
      editingPolygon = null;
      editingPolygonIndex = null;
      controlPoints = [];
      activePointIndex = null;
  
    } else {
      // ==========================
      // ✅ case 2: 只有 1 个 polygon → 删除整个 segment
      // ==========================
      const idx = segments.indexOf(seg);
      if (idx >= 0) segments.splice(idx, 1);
  
      editingSegment = null;
      editingPolygon = null;
      editingPolygonIndex = null;
      controlPoints = [];
      activePointIndex = null;
    }
  
    // 重新绘制 + 更新列表
    drawPanoptic(ctx, canvas);      // ✅推荐别传 ctx/canvas，内部获取
    renderMaskList();
  }
  

  function drawFastSamPreview() {
    if (!canvasFastSam || !ctxFastSam) return;
    if (!fastSamImageSize || !fastSamBaseImage) return;
  
    const { width, height } = fastSamImageSize;
  
    canvasFastSam.width = width;
    canvasFastSam.height = height;
  
    ctxFastSam.clearRect(0, 0, width, height);
  
    // ✅ 1️⃣ 先画原图
    ctxFastSam.drawImage(fastSamBaseImage, 0, 0, width, height);
  
    // ✅ 2️⃣ 再画 Fast-SAM masks
    fastSamMasks.forEach(m => {
      const poly = m.polygon;
      if (!poly || poly.length < 3) return;
  
      ctxFastSam.beginPath();
      poly.forEach(([x, y], i) => {
        i === 0 ? ctxFastSam.moveTo(x, y) : ctxFastSam.lineTo(x, y);
      });
      ctxFastSam.closePath();
  
      ctxFastSam.fillStyle = "rgba(0,255,136,0.25)";
      ctxFastSam.strokeStyle = "#00ff88";
      ctxFastSam.lineWidth = 1;
      ctxFastSam.fill();
      ctxFastSam.stroke();
    });
  }
  

  function hitFastSamMask(e) {
    if (!canvasFastSam || !fastSamMasks.length) return null;
  
    const rect = canvasFastSam.getBoundingClientRect();
    const x = (e.clientX - rect.left) * canvasFastSam.width / rect.width;
    const y = (e.clientY - rect.top) * canvasFastSam.height / rect.height;
  
    for (const m of fastSamMasks) {
      if (pointInPolygon(x, y, m.polygon)) {
        return m;
      }
    }
    return null;
  }
  
  if (canvasFastSam) {
    canvasFastSam.addEventListener("dblclick", async e => {
      const hit = hitFastSamMask(e);
      if (!hit) return;
  
      // 深拷贝 polygon，避免污染 Fast-SAM 结果
      const poly = hit.polygon.map(p => [p[0], p[1]]);
  
      confirmFastSamPolygon(poly);
    });
  }
  //SAM
  async function loadFastSamPreview() {
    if (!canvasFastSam || !ctxFastSam) return;
  
    const res = await fetch("/api/panoptic/fastsam", {
      method: "POST"
    });
    const data = await res.json();
  
    // 统一存结构
    fastSamMasks = data.masks || [];
    fastSamImageSize = data.image_size;
  
    drawFastSamPreview();
  }

  async function confirmFastSamPolygon(polygon) {
    const result = await askCategoryAndInstance();
    if (!result) return;
  
    pushUndoState();
  
    currentPanopticData.segments_info.push({
      id: generateNewSegmentId(),
      category_name: result.category_name,
      class_id: result.class_id,
      isthing: result.isthing,
      instance_id: result.instance_id,
      polygon
    });
  
    drawPanoptic(
      document.getElementById("pvCanvas").getContext("2d"),
      document.getElementById("pvCanvas")
    );
    renderMaskList();
  }

  async function loadFastSamPreview() {
    if (!canvasFastSam || !ctxFastSam) return;
  
    const res = await fetch("/api/panoptic/fastsam", { method: "POST" });
    const data = await res.json();
  
    const { width, height } = data.image_size;
  
    canvasFastSam.width = width;
    canvasFastSam.height = height;
  
    // === 1. 画原图 ===
    const img = new Image();
    img.src = "data:image/png;base64," + data.ori_image;
  
    img.onload = () => {
      ctxFastSam.clearRect(0, 0, width, height);
      ctxFastSam.drawImage(img, 0, 0);
  
      // === 2. 叠加 SAM overlay（半透明）===
      const samImg = new Image();
      samImg.src = "data:image/png;base64," + data.sam_overlay;
  
      samImg.onload = () => {
        ctxFastSam.globalAlpha = 0.55;   // ⭐ 半透明
        ctxFastSam.drawImage(samImg, 0, 0);
        ctxFastSam.globalAlpha = 1.0;
      };
    };
  }

  canvasFastSam.addEventListener("dblclick", async e => {
    const rect = canvasFastSam.getBoundingClientRect();
  
    const x = Math.round(
      (e.clientX - rect.left) * canvasFastSam.width / rect.width
    );
    const y = Math.round(
      (e.clientY - rect.top) * canvasFastSam.height / rect.height
    );
  
    // 🔍 向后端查询 Fast-SAM mask_id
    const res = await fetch("/api/panoptic/fastsam/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x, y })
    });
  
    const { mask_id } = await res.json();
    if (mask_id == null) return;
  
    // 🚀 把这个 mask 转成 polygon
    importFastSamMask(mask_id);
  });

  async function importFastSamMask(maskId) {
    const result = await askCategoryAndInstance();
    if (!result) return;
  
    // 后端已有 polygon-create-mask 思路
    const res = await fetch("/api/image/set-mask-class", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mask_id: maskId,
        class_name: result.category_name
      })
    });
  
    const data = await res.json();
  
    // TODO: 如果是 thing，再 set instance
    if (result.isthing) {
      await fetch("/api/image/set-mask-instance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mask_id: maskId,
          class_name: result.category_name,
          instance: result.instance_id
        })
      });
    }
  
    alert(`Fast-SAM mask ${maskId} imported`);
  }
  
  
  
}