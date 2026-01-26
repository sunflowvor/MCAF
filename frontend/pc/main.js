import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.164/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.164/examples/jsm/controls/OrbitControls.js";
//import { PCDLoader } from "https://cdn.jsdelivr.net/npm/three@0.164/examples/jsm/loaders/PCDLoader.js";

/* =======================
   UI
======================= */
// const ui = {
//   pcdFile: document.getElementById("pcdFile"),
//   metaFile: document.getElementById("metaFile"),
//   stats: document.getElementById("stats"),
//   pointInfo: document.getElementById("pointInfo"),
// };
//const ui = { pcdFile:null, metaFile:null, stats:null, pointInfo:null };
const ui = { stats:null, pointInfo:null };

/* =======================
   Global State
======================= */
let scene, camera, renderer, controls;
let pointsObj = null;
let geom = null;
let posAttr = null;
let colorAttr = null;
let rgbArr = null;
let colorMode = "category";

let categoryArr = null;
let instanceArr = null;
let metaData = null;
let raycaster = new THREE.Raycaster();
let mouseNDC = new THREE.Vector2();
let selectedMask = null;   // Uint8Array, 0/1
let outputDirHandle = null;
let instanceBoxes = []; // THREE.Mesh[]

let activeBox = null;
let editMode = null; // "move" | "rotate" | "resize"
let dragStartPoint = new THREE.Vector3();
let dragStartYaw = 0;

// ===== box edit temp state =====
let dragStartPlanePoint = new THREE.Vector3();
let dragStartMouseX = 0;

// resize related
let resizeAxisLocal = new THREE.Vector3();
let resizeAxisWorld = new THREE.Vector3();
let resizeSign = 1;

let resizeStartSize = { dx: 0, dy: 0, dz: 0 };
let resizeStartCenter = { x: 0, y: 0, z: 0 };
let resizeStartLocalHit = new THREE.Vector3();

let visibilityMask = null;  // 1=显示, 0=隐藏
let currentFilter = { cat: null, inst: null }; // 当前选择

let metaFileHandle = null; // 🔑 关键

// ===== Category Mapping =====
let classNameToId = {};   // name -> id
let classIdToName = {};   // id -> name

const boxRaycaster = new THREE.Raycaster();
const arrowRaycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

const MOVE_STEP = 0.2;    // XY 平移步长（米）
const MOVE_STEP_Z = 0.1; // Z 平移步长
const ROTATE_STEP = 2 * Math.PI / 180; // 每次旋转 2°

/* =======================
   Init
======================= */
init();
animate();

function init() {

  ui.pcdFile = document.getElementById("pcdFile");
  //ui.metaFile = document.getElementById("metaFile");

  ui.stats = document.getElementById("stats");
  
  // === 强制创建 pointInfo（兜底） ===
  let el = document.getElementById("pointInfo");
  if (!el) {
    console.warn("pointInfo not found, creating one dynamically");

    el = document.createElement("div");
    el.id = "pointInfo";
    el.style.position = "fixed";
    el.style.top = "12px";
    el.style.right = "12px";
    el.style.zIndex = "20";
    el.style.background = "rgba(0,0,0,0.65)";
    el.style.color = "#fff";
    el.style.padding = "10px 12px";
    el.style.borderRadius = "10px";
    el.style.fontSize = "13px";
    el.style.pointerEvents = "none";
    el.innerHTML = "<b>Point Info</b><br/>Ctrl + Click a point";

    const uiPanel = document.getElementById("hint") || document.getElementById("ui");
    uiPanel.appendChild(el);
  }

  ui.pointInfo = el;

  console.log("pointInfo ready:", ui.pointInfo);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111111);

  camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.01,
    1e7
  );
  camera.position.set(0, 0, 5);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
  renderer.domElement.addEventListener("mousedown", onCtrlPickPoint);
  document.body.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = false;

  scene.add(new THREE.AxesHelper(1));

  window.addEventListener("resize", onResize);

  //ui.pcdFile.addEventListener("change", onLoadPCD);
  //ui.metaFile.addEventListener("change", onLoadMeta);
  //setupRectangleSelect();
  setupLassoSelect();
  document.getElementById("apply").addEventListener("click", applySelected);
  document.getElementById("saveMeta").addEventListener("click", saveMetaNPY);
  document.getElementById("genBox").addEventListener("click", generateInstanceBoxes);
  document.getElementById("delBox").addEventListener("click", deleteAllBoxes);
  //renderer.domElement.addEventListener("mousedown", onPickBoxFace);
  renderer.domElement.addEventListener("mousedown", onPickBox);
  renderer.domElement.addEventListener("dblclick", onDoubleClickBox);
  renderer.domElement.addEventListener("mousedown", onEditMouseDown);
  window.addEventListener("mousemove", onEditMouseMove);
  window.addEventListener("mouseup", onEditMouseUp);

  document.getElementById("exportKitti").addEventListener("click", exportBoxesToKitti);
  window.addEventListener("keydown", onBoxKeyDown);
  window.addEventListener("keydown", onKeyDown);

  document.getElementById("viewFrontX").onclick = () => setFrontView("x");
  document.getElementById("viewFrontY").onclick = () => setFrontView("y");
  document.getElementById("viewTop")?.addEventListener("click", setTopView);
  document.getElementById("viewReset").onclick = resetView; 

  // Help panel close
  const helpPanel = document.getElementById("helpPanel");
  const closeHelp = document.getElementById("closeHelp");

  if (closeHelp && helpPanel) {
    closeHelp.addEventListener("click", () => {
      helpPanel.style.display = "none";
    });
  }

  document.getElementById("openMeta").onclick = async () => {
    const [handle] = await window.showOpenFilePicker({
      types: [{
        description: "Meta NPY",
        accept: { "application/octet-stream": [".npy"] }
      }]
    });
  
    metaFileHandle = handle;           // 🔑 记住这个文件
    const file = await handle.getFile();
  
    await onLoadMetaFromFile(file);    // 🔥 直接显示
  }; 
  document.getElementById("openClassJson").onclick = async () => {
    const [handle] = await window.showOpenFilePicker({
      types: [{
        description: "Class Mapping JSON",
        accept: { "application/json": [".json"] }
      }]
    });
  
    const file = await handle.getFile();
    const text = await file.text();
    const json = JSON.parse(text);
  
    if (!json.class_name_to_id) {
      alert("Invalid class json: missing class_name_to_id");
      return;
    }
  
    classNameToId = json.class_name_to_id;
    classIdToName = {};
  
    // 🔁 反转映射
    for (const [name, id] of Object.entries(classNameToId)) {
      classIdToName[id] = name;
    }
  
    console.log("✅ Loaded class mapping:", classIdToName);
  };
  
  console.log("init ok");
}

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

async function onLoadMetaFromFile(file) {
  const { data, shape } = await loadNPY(file);

  if (shape.length !== 2 || shape[1] !== 9) {
    throw new Error("meta.npy must have shape (N, 9)");
  }

  const N = shape[0];
  metaData = data;

  // ===== 解析 =====
  const positions = new Float32Array(N * 3);
  rgbArr = new Uint8Array(N * 3);
  categoryArr = new Uint16Array(N);
  instanceArr = new Uint16Array(N);

  for (let i = 0; i < N; i++) {
    const b = i * 9;

    positions[i*3+0] = data[b+0];
    positions[i*3+1] = data[b+1];
    positions[i*3+2] = data[b+2];

    categoryArr[i] = data[b+3] | 0;
    instanceArr[i] = data[b+4] | 0;

    rgbArr[i*3+0] = data[b+5] | 0;
    rgbArr[i*3+1] = data[b+6] | 0;
    rgbArr[i*3+2] = data[b+7] | 0;
  }

  // ===== 构建点云 =====
  buildPointCloudFromArrays(positions, N);

  selectedMask = new Uint8Array(N);
  visibilityMask = new Uint8Array(N);
  visibilityMask.fill(1);

  colorMode = "category";
  recolorAll();
  fitCameraToPoints();
  updateStats();
  buildInstanceFilter();

  console.log("✅ Meta loaded:", N);
}


/* =======================
   NPY Loader (browser)
======================= */
async function loadNPY(file) {
  const buf = await file.arrayBuffer();
  const u8 = new Uint8Array(buf);
  const view = new DataView(buf);

  // magic string
  if (String.fromCharCode(...u8.slice(0, 6)) !== "\x93NUMPY") {
    throw new Error("Not a npy file");
  }

  const major = u8[6];
  const headerLen =
    major === 1
      ? view.getUint16(8, true)
      : view.getUint32(8, true);

  const headerStart = major === 1 ? 10 : 12;
  const headerText = new TextDecoder().decode(
    u8.slice(headerStart, headerStart + headerLen)
  );

  /* =========================
     ✅ 正确解析 shape
     ========================= */
  const shapeMatch = headerText.match(/'shape': *\(([^)]*)\)/);
  if (!shapeMatch) {
    throw new Error("Cannot parse npy shape");
  }

  const shape = shapeMatch[1]
    .split(",")
    .map(s => s.trim())
    .filter(s => s.length > 0)
    .map(Number);

  /* =========================
     解析 dtype
     ========================= */
  const descrMatch = headerText.match(/'descr': *'([^']+)'/);
  if (!descrMatch) {
    throw new Error("Cannot parse npy dtype");
  }
  const dtype = descrMatch[1];

  const offset = headerStart + headerLen;

  let array;
  if (dtype === "<f4") array = new Float32Array(buf, offset);
  else if (dtype === "<f8") array = new Float64Array(buf, offset);
  else if (dtype === "<u2") array = new Uint16Array(buf, offset);
  else if (dtype === "<i4") array = new Int32Array(buf, offset);
  else throw new Error("Unsupported dtype: " + dtype);

  return { data: array, shape };
}

/* =======================
   Coloring
======================= */
function recolor() {
  if (colorMode === "rgb") {
    recolorByRGB();
  } else {
    recolorByCategory();
  }
}

function recolorByCategory() {
  if (!colorAttr || !categoryArr) return;

  const colors = colorAttr.array;
  const n = categoryArr.length;

  for (let i = 0; i < n; i++) {
    const [r, g, b] = catToRGB(categoryArr[i]);
    colors[i * 3] = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;
  }

  colorAttr.needsUpdate = true;
}

function recolorByRGB() {
  if (!colorAttr || !rgbArr) return;

  const colors = colorAttr.array;
  const n = posAttr.count;

  for (let i = 0; i < n; i++) {
    colors[i * 3 + 0] = rgbArr[i * 3 + 0] / 255.0;
    colors[i * 3 + 1] = rgbArr[i * 3 + 1] / 255.0;
    colors[i * 3 + 2] = rgbArr[i * 3 + 2] / 255.0;
  }

  colorAttr.needsUpdate = true;
}


function catToRGB(cat) {
  let x = (cat * 2654435761) >>> 0;
  return [
    0.2 + 0.8 * ((x & 255) / 255),
    0.2 + 0.8 * (((x >> 8) & 255) / 255),
    0.2 + 0.8 * (((x >> 16) & 255) / 255),
  ];
}

/* =======================
   Camera / Stats
======================= */
function fitCameraToPoints() {
  geom.computeBoundingBox();
  const box = geom.boundingBox;

  const center = new THREE.Vector3();
  box.getCenter(center);

  const size = new THREE.Vector3();
  box.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z);

  controls.target.copy(center);
  camera.position.copy(
    center.clone().add(new THREE.Vector3(0, 0, maxDim * 1.5))
  );
  camera.updateProjectionMatrix();
}

function updateStats() {
  ui.stats.innerHTML = `
    <b>Points:</b> ${posAttr ? posAttr.count.toLocaleString() : "-"}<br/>
    <b>Meta:</b> ${categoryArr ? "loaded" : "none"}
  `;
}

/* =======================
   Rectangle Selection
   Shift + Left Drag
======================= */
function setupLassoSelect() {
  const canvas = renderer.domElement;
  const overlay = document.getElementById("overlay");

  // 用 canvas 画套索
  const drawCanvas = document.createElement("canvas");
  drawCanvas.width = window.innerWidth;
  drawCanvas.height = window.innerHeight;
  overlay.appendChild(drawCanvas);

  const ctx = drawCanvas.getContext("2d");

  let drawing = false;
  let lasso = [];

  canvas.addEventListener("mousedown", (e) => {
    if (!e.shiftKey || e.button !== 0) return;
    if (!posAttr) return;

    drawing = true;
    lasso = [{ x: e.clientX, y: e.clientY }];

    ctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    controls.enabled = false;
  });

  window.addEventListener("mousemove", (e) => {
    if (!drawing) return;

    lasso.push({ x: e.clientX, y: e.clientY });

    ctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    ctx.beginPath();
    ctx.moveTo(lasso[0].x, lasso[0].y);

    for (let i = 1; i < lasso.length; i++) {
      ctx.lineTo(lasso[i].x, lasso[i].y);
    }

    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.lineWidth = 2;
    ctx.stroke();
  });

  window.addEventListener("mouseup", () => {
    if (!drawing) return;

    drawing = false;
    controls.enabled = true;

    ctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);

    performLassoSelection(lasso);
  });

  window.addEventListener("resize", () => {
    drawCanvas.width = window.innerWidth;
    drawCanvas.height = window.innerHeight;
  });
}

function pointInPolygon(x, y, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x, yi = polygon[i].y;
    const xj = polygon[j].x, yj = polygon[j].y;

    const intersect =
      yi > y !== yj > y &&
      x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;

    if (intersect) inside = !inside;
  }
  return inside;
}

function performLassoSelection(lasso) {
  if (!posAttr || !selectedMask) return;

  const v = new THREE.Vector3();
  let count = 0;

  for (let i = 0; i < posAttr.count; i++) {
    if (visibilityMask[i] === 0) continue;
    if (selectedMask[i] === 1) continue; // 👈 不再选已选过的点

    v.set(
      posAttr.array[i * 3 + 0],
      posAttr.array[i * 3 + 1],
      posAttr.array[i * 3 + 2]
    );

    v.project(camera);
    if (v.z < -1 || v.z > 1) continue;

    const sx = (v.x * 0.5 + 0.5) * window.innerWidth;
    const sy = (-v.y * 0.5 + 0.5) * window.innerHeight;

    if (pointInPolygon(sx, sy, lasso)) {
      selectedMask[i] = 1;
      count++;
    }
  }

  recolorAll(); // 统一刷新颜色
  console.log("Lasso selected:", count);
}

function recolorAll() {
  if (!colorAttr || !categoryArr) return;

  const colors = colorAttr.array;
  const n = categoryArr.length;

  for (let i = 0; i < n; i++) {

    if (visibilityMask && visibilityMask[i] === 0) {
      // 完全隐藏
      colors[i * 3 + 0] = 0;
      colors[i * 3 + 1] = 0;
      colors[i * 3 + 2] = 0;
      continue;
    }

    if (selectedMask && selectedMask[i] === 1) {
      colors[i * 3 + 0] = 1.0;
      colors[i * 3 + 1] = 0.3;
      colors[i * 3 + 2] = 0.3;
    } else {
      const [r, g, b] =
        colorMode === "rgb"
          ? [
              rgbArr[i * 3] / 255,
              rgbArr[i * 3 + 1] / 255,
              rgbArr[i * 3 + 2] / 255,
            ]
          : catToRGB(categoryArr[i]);

      colors[i * 3 + 0] = r;
      colors[i * 3 + 1] = g;
      colors[i * 3 + 2] = b;
    }
  }

  colorAttr.needsUpdate = true;
}

function onCtrlPickPoint(e) {
  if (!e.ctrlKey || e.button !== 0) return;
  if (!pointsObj || !posAttr || !metaData) return;

  const rect = renderer.domElement.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  // 允许点击误差（像素半径）
  const radiusPx = 8;
  const radius2 = radiusPx * radiusPx;

  const v = new THREE.Vector3();

  let bestIdx = -1;
  let bestD2 = Infinity;
  let bestZ = Infinity; // 同距离时优先更靠近相机（可选）

  for (let i = 0; i < posAttr.count; i++) {
    v.set(
      posAttr.array[i * 3 + 0],
      posAttr.array[i * 3 + 1],
      posAttr.array[i * 3 + 2]
    );

    // 投影到 NDC
    v.project(camera);

    // 裁剪：在屏幕外/背后就跳过
    if (v.z < -1 || v.z > 1) continue;

    // NDC -> 像素坐标（相对于 canvas）
    const sx = (v.x * 0.5 + 0.5) * rect.width;
    const sy = (-v.y * 0.5 + 0.5) * rect.height;

    const dx = sx - mx;
    const dy = sy - my;
    const d2 = dx * dx + dy * dy;

    // 只考虑鼠标半径内
    if (d2 > radius2) continue;

    // 选择 2D 距离最小；若相同，选更靠近相机（z 更小）
    if (d2 < bestD2 || (d2 === bestD2 && v.z < bestZ)) {
      bestD2 = d2;
      bestIdx = i;
      bestZ = v.z;
    }
  }

  if (bestIdx < 0) return; // 没找到半径范围内的点
  showPointInfo(bestIdx);
}

function showPointInfo(i) {
  console.log("DEBUG ui.pointInfo =", ui.pointInfo);

  if (!ui.pointInfo) {
    console.error("pointInfo is missing. Check HTML id=pointInfo and init order.");
    return;
  }
  const b = i * 9;

  const frame = metaData[b + 6];
  const cat = metaData[b + 7];
  const inst = metaData[b + 8];

  ui.pointInfo.innerHTML = `
    <b>Point Info</b><br/>
    <b>Index:</b> ${i}<br/>
    <b>Frame:</b> ${frame}<br/>
    <b>Category:</b> ${cat}<br/>
    <b>Instance:</b> ${inst}
  `;
  console.log("inst:", inst);

  // 可选：高亮该点
  colorAttr.array[i * 3 + 0] = 0.0;
  colorAttr.array[i * 3 + 1] = 1.0;
  colorAttr.array[i * 3 + 2] = 1.0;
  colorAttr.needsUpdate = true;
}

function applySelected() {
  if (!selectedMask || !categoryArr || !instanceArr) return;

  const newCat = parseInt(document.getElementById("category").value, 10) | 0;
  const newInst = parseInt(document.getElementById("instance").value, 10) | 0;

  let changed = 0;

  for (let i = 0; i < selectedMask.length; i++) {
    if (selectedMask[i] === 1) {
      categoryArr[i] = newCat;
      instanceArr[i] = newInst;
      selectedMask[i] = 0;   // 👈 释放选择
      changed++;
    }
  }

  recolorAll();
  updateStats();

  console.log(`Applied cat=${newCat}, inst=${newInst} to ${changed} points`);
}


function syncMetaDataFromArrays() {
  if (!metaData || !categoryArr || !instanceArr) return;

  const N = categoryArr.length;

  for (let i = 0; i < N; i++) {
    const b = i * 9;

    // ✅ 写回到真正的 cat / inst 列
    metaData[b + 3] = categoryArr[i];
    metaData[b + 4] = instanceArr[i];
  }
}

async function openMetaNPY() {
  const [handle] = await window.showOpenFilePicker({
    types: [
      {
        description: "NPY file",
        accept: { "application/octet-stream": [".npy"] }
      }
    ]
  });

  metaFileHandle = handle; // 🔴 保存句柄（写权限来源）

  const file = await handle.getFile();
  await onLoadMetaFile(file); // 你现有的 onLoadMeta 内容
}

async function saveMetaNPY() {
  if (!metaFileHandle) {
    alert("No meta file opened");
    return;
  }

  syncMetaDataFromArrays();

  const N = metaData.length / 9;
  const buffer = buildNPYBuffer(metaData, [N, 9], "<f8");

  const writable = await metaFileHandle.createWritable();
  await writable.write(buffer);
  await writable.close();

  console.log("✅ 原始 meta.npy 已被直接替换");
}


function buildNPYBuffer(data, shape, descr) {
  const magic = "\x93NUMPY";
  const major = 1;
  const minor = 0;

  const shapeStr =
    "(" + shape.join(", ") + (shape.length === 1 ? "," : "") + ")";

  let header =
    "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " +
    shapeStr + ", }";

  const headerLen =
    Math.ceil((magic.length + 2 + 2 + header.length + 1) / 16) * 16 -
    (magic.length + 2 + 2);

  header = header.padEnd(headerLen - 1, " ") + "\n";
  const headerBuf = new TextEncoder().encode(header);

  const totalLen =
    magic.length + 2 + 2 + headerBuf.length + data.byteLength;

  const buf = new ArrayBuffer(totalLen);
  const view = new DataView(buf);
  let offset = 0;

  // magic
  for (let i = 0; i < magic.length; i++) {
    view.setUint8(offset++, magic.charCodeAt(i));
  }

  view.setUint8(offset++, major);
  view.setUint8(offset++, minor);
  view.setUint16(offset, headerBuf.length, true);
  offset += 2;

  new Uint8Array(buf, offset, headerBuf.length).set(headerBuf);
  offset += headerBuf.length;

  // ✅ 正确拷贝 TypedArray 数据
  const src = new Uint8Array(
    data.buffer,
    data.byteOffset,
    data.byteLength
  );

  new Uint8Array(buf, offset, data.byteLength).set(src);

  return buf;
}

//3D Box
function collectPointsByCategoryInstance() {
  const map = new Map(); 
  // key: "cat_inst" -> { cat, inst, points: [[x,y,z], ...] }

  for (let i = 0; i < posAttr.count; i++) {
    const cat = categoryArr[i];
    const inst = instanceArr[i];

    // 可选：跳过背景
    if (cat === 0 || inst === 0) continue;

    const key = `${cat}_${inst}`;

    if (!map.has(key)) {
      map.set(key, {
        cat,
        inst,
        points: [],
      });
    }

    map.get(key).points.push([
      posAttr.array[i * 3 + 0],
      posAttr.array[i * 3 + 1],
      posAttr.array[i * 3 + 2],
    ]);
  }

  return map;
}


function computeYawFromXY(points) {
  let mx = 0, my = 0;
  for (const [x, y] of points) {
    mx += x; my += y;
  }
  mx /= points.length;
  my /= points.length;

  let sxx = 0, syy = 0, sxy = 0;
  for (const [x, y] of points) {
    const dx = x - mx;
    const dy = y - my;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
  }

  return 0.5 * Math.atan2(2 * sxy, sxx - syy);
}

function computeXYBounds(points, yaw) {
  const c = Math.cos(-yaw);
  const s = Math.sin(-yaw);

  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;

  for (const [x, y] of points) {
    const rx = x * c - y * s;
    const ry = x * s + y * c;

    minX = Math.min(minX, rx);
    maxX = Math.max(maxX, rx);
    minY = Math.min(minY, ry);
    maxY = Math.max(maxY, ry);
  }

  return { minX, maxX, minY, maxY };
}

function computeZBounds(points) {
  let minZ = Infinity, maxZ = -Infinity;
  for (const [, , z] of points) {
    minZ = Math.min(minZ, z);
    maxZ = Math.max(maxZ, z);
  }
  return { minZ, maxZ };
}

function computeInstanceBox(points) {
  const yaw = computeYawFromXY(points);
  const { minX, maxX, minY, maxY } = computeXYBounds(points, yaw);
  const { minZ, maxZ } = computeZBounds(points);

  const cxLocal = (minX + maxX) / 2;
  const cyLocal = (minY + maxY) / 2;

  const c = Math.cos(yaw);
  const s = Math.sin(yaw);

  const cx = cxLocal * c - cyLocal * s;
  const cy = cxLocal * s + cyLocal * c;
  const cz = (minZ + maxZ) / 2;

  return {
    center: [cx, cy, cz],
    size: [maxX - minX, maxY - minY, maxZ - minZ],
    yaw,
  };
}

function createBoxMesh(box) {
  const [dx, dy, dz] = box.size;

  const geom = new THREE.BoxGeometry(dx, dy, dz);
  const mat = new THREE.MeshBasicMaterial({
    color: 0x00ffff,
    wireframe: true,
  });

  const mesh = new THREE.Mesh(geom, mat);
  mesh.position.set(...box.center);
  mesh.rotation.set(0, 0, box.yaw); // 只绕 Z

  return mesh;
}

function generateInstanceBoxes() {
  if (!posAttr || !instanceArr) {
    alert("Please load PCD and meta first");
    return;
  }

  // 先清掉旧 box
  for (const m of instanceBoxes) {
    scene.remove(m);
    m.geometry.dispose();
    m.material.dispose();
  }
  instanceBoxes.length = 0;

  const map = collectPointsByCategoryInstance();

  for (const { cat, inst, points } of map.values()) {
    if (points.length < 10) continue;
  
    const box = computeInstanceBox(points);
    const mesh = createBoxMesh(box);
  
    // 🔴 关键：cat / inst 必须挂在 mesh.userData
    mesh.userData = {
      cat,
      inst,
      center: {
        x: box.center[0],
        y: box.center[1],
        z: box.center[2],
      },
      size: {
        dx: box.size[0],
        dy: box.size[1],
        dz: box.size[2],
      },
      yaw: box.yaw,   // PCA 算出来的 raw yaw
    };
    
    scene.add(mesh);
    instanceBoxes.push(mesh);
    
    // 👇 画车头箭头
    createYawArrow(mesh);    
  }

  console.log("Generated boxes:", instanceBoxes.length);
}

function deleteAllBoxes() {
  if (!instanceBoxes || instanceBoxes.length === 0) {
    console.log("No boxes to delete");
    return;
  }

  for (const mesh of instanceBoxes) {
    scene.remove(mesh);

    if (mesh.geometry) mesh.geometry.dispose();
    if (mesh.material) mesh.material.dispose();
  }

  instanceBoxes.length = 0;

  console.log("All instance boxes deleted");
}
//show box info
function onPickBox(e) {
  // 只用左键，且不按 Ctrl（避免和点云 Ctrl+点击冲突）
  if (e.button !== 0 || e.ctrlKey) return;
  if (!instanceBoxes || instanceBoxes.length === 0) return;

  const rect = renderer.domElement.getBoundingClientRect();

  mouseNDC.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouseNDC.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

  boxRaycaster.setFromCamera(mouseNDC, camera);

  const hits = boxRaycaster.intersectObjects(instanceBoxes, false);
  if (hits.length === 0) return;

  const box = hits[0].object;
  showBoxInfo(box);
}

function showBoxInfo(box) {
  const d = box.userData;
  if (!d || !d.size) {
    ui.pointInfo.innerHTML = "<b>Box Info</b><br/>Invalid box";
    return;
  }

  const { center, size, yaw } = d;

  ui.pointInfo.innerHTML = `
    <b>Box Info</b><br/>
    <b>Category:</b> ${d.cat}
    (${classIdToName[d.cat] ?? "unknown"})<br/>
    <b>Instance:</b> ${d.inst}<br/><br/>

    <b>Center (LiDAR)</b><br/>
    x: ${center.x.toFixed(2)}<br/>
    y: ${center.y.toFixed(2)}<br/>
    z: ${center.z.toFixed(2)}<br/><br/>

    <b>Size (meters)</b><br/>
    dx: <input id="box_dx" type="number" step="0.01" value="${size.dx.toFixed(2)}"><br/>
    dy: <input id="box_dy" type="number" step="0.01" value="${size.dy.toFixed(2)}"><br/>
    dz: <input id="box_dz" type="number" step="0.01" value="${size.dz.toFixed(2)}"><br/><br/>

    <button id="applyBoxSize">Apply Size</button><br/><br/>

    <b>Yaw</b><br/>
    rad: ${yaw.toFixed(3)}<br/>
    deg: ${(yaw * 180 / Math.PI).toFixed(1)}
  `;

  // 绑定按钮事件
  document
    .getElementById("applyBoxSize")
    .addEventListener("click", () => applyBoxSize(box));
}


function highlightBox(box) {
  for (const b of instanceBoxes) {
    b.material.color.set(0x00ffff); // 默认色
  }
  box.material.color.set(0xffaa00); // 当前选中
}

function createYawArrow(mesh) {
  const { center, size, yaw } = mesh.userData;

  if (yaw == null) return;

  // 1️⃣ 车头方向（XY 平面）
  const dir = new THREE.Vector3(
    Math.cos(yaw),
    Math.sin(yaw),
    0
  ).normalize();

  // 2️⃣ 箭头起点：box 前表面中心
  // 前表面 = center + dir * (length / 2)
  const origin = new THREE.Vector3(
    center.x,
    center.y,
    center.z
  ).addScaledVector(dir, size.dx * 0.5);

  // 3️⃣ 箭头长度
  const length = size.dx * 0.6;

  const arrow = new THREE.ArrowHelper(
    dir,
    origin,
    length,
    0xffaa00,   // 橙色
    length * 0.25,
    length * 0.15
  );

  arrow.userData.type = "yawArrow";
  arrow.userData.parentBox = mesh;

  scene.add(arrow);
  mesh.userData.arrow = arrow;
}

function updateYawArrow(mesh) {
  const arrow = mesh.userData.arrow;
  if (!arrow) return;

  const { center, size, yaw } = mesh.userData;

  const dir = new THREE.Vector3(
    Math.cos(yaw),
    Math.sin(yaw),
    0
  ).normalize();

  const origin = new THREE.Vector3(
    center.x,
    center.y,
    center.z
  ).addScaledVector(dir, size.dx * 0.5);

  arrow.position.copy(origin);
  arrow.setDirection(dir);
}

//change 3D Box

function onDoubleClickBox(e) {
  if (!instanceBoxes || instanceBoxes.length === 0) return;

  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

  boxRaycaster.setFromCamera(mouse, camera);

  const hits = boxRaycaster.intersectObjects(instanceBoxes, false);
  if (!hits.length) return;

  activeBox = hits[0].object;

  // 确保 userData 完整
  if (!activeBox.userData?.center || !activeBox.userData?.size) {
    console.warn("Box userData missing center/size");
    return;
  }

  highlightBox(activeBox);
  console.log("Selected box:", activeBox.userData.cat, activeBox.userData.inst);
}

// ---------- helpers ----------
function setMouseNDC(e) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
}

// mouse ray hit a fixed Z plane -> point
function getMousePointOnZPlane(e, z) {
  setMouseNDC(e);
  const ray = new THREE.Raycaster();
  ray.setFromCamera(mouse, camera);

  // plane: 0x + 0y + 1z + d = 0  => z = -d
  const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -z);
  const out = new THREE.Vector3();
  ray.ray.intersectPlane(plane, out);
  return out;
}

// raycast hit point on activeBox (world)
function raycastBoxHitPointWorld(e, mesh) {
  setMouseNDC(e);
  boxRaycaster.setFromCamera(mouse, camera);
  const hits = boxRaycaster.intersectObject(mesh, false);
  if (!hits.length) return null;
  return hits[0]; // { point, face, ... }
}

// ---------- mousedown ----------
function onEditMouseDown(e) {
  if (!activeBox) return;

  // Alt + Left => move on fixed Z plane
  if (e.altKey && e.button === 0) {
    editMode = "move";
    controls.enabled = false;

    const z = activeBox.userData.center.z;
    dragStartPlanePoint.copy(getMousePointOnZPlane(e, z));
    return;
  }

  // Alt + Right => rotate yaw around Z
  if (e.altKey && e.button === 2) {
    editMode = "rotate";
    controls.enabled = false;

    dragStartMouseX = e.clientX;
    dragStartYaw = activeBox.userData.yaw || 0;
    return;
  }

  // Ctrl + Left => resize by picking face
  if (e.ctrlKey && e.button === 0) {
    const hit = raycastBoxHitPointWorld(e, activeBox);
    if (!hit || !hit.face) return;

    editMode = "resize";
    controls.enabled = false;

    // face normal is in geometry local space
    const n = hit.face.normal.clone(); // local
    // choose axis by largest abs component
    const ax = Math.abs(n.x), ay = Math.abs(n.y), az = Math.abs(n.z);

    resizeAxisLocal.set(0, 0, 0);
    if (ax >= ay && ax >= az) resizeAxisLocal.set(1, 0, 0);
    else if (ay >= ax && ay >= az) resizeAxisLocal.set(0, 1, 0);
    else resizeAxisLocal.set(0, 0, 1);

    // sign: which side
    // (n.x>0 => +X face, etc)
    resizeSign =
      resizeAxisLocal.x ? (n.x >= 0 ? 1 : -1) :
      resizeAxisLocal.y ? (n.y >= 0 ? 1 : -1) :
                          (n.z >= 0 ? 1 : -1);

    // world axis = local axis rotated by mesh quaternion
    resizeAxisWorld.copy(resizeAxisLocal).applyQuaternion(activeBox.quaternion).normalize();

    // store start size & center
    resizeStartSize.dx = activeBox.userData.size.dx;
    resizeStartSize.dy = activeBox.userData.size.dy;
    resizeStartSize.dz = activeBox.userData.size.dz;

    resizeStartCenter.x = activeBox.userData.center.x;
    resizeStartCenter.y = activeBox.userData.center.y;
    resizeStartCenter.z = activeBox.userData.center.z;

    // store start hit in LOCAL space (important)
    const localHit = hit.point.clone();
    activeBox.worldToLocal(localHit);
    resizeStartLocalHit.copy(localHit);

    return;
  }
}

// ---------- mousemove ----------
function onEditMouseMove(e) {
  if (!activeBox || !editMode) return;

  // 5) MOVE
  if (editMode === "move") {
    const z = activeBox.userData.center.z;
    const p = getMousePointOnZPlane(e, z);
    const delta = p.clone().sub(dragStartPlanePoint);

    activeBox.userData.center.x += delta.x;
    activeBox.userData.center.y += delta.y;

    dragStartPlanePoint.copy(p);

    syncMeshFromUserData(activeBox);
    return;
  }

  // 6) ROTATE (yaw)
  if (editMode === "rotate") {
    const dx = e.clientX - dragStartMouseX;
    const sensitivity = 0.005;

    activeBox.userData.yaw = normalizeAngle(dragStartYaw + dx * sensitivity);

    syncMeshFromUserData(activeBox);
    return;
  }

  // 7) RESIZE (no scale, change dx/dy/dz individually)
  if (editMode === "resize") {
    const hit = raycastBoxHitPointWorld(e, activeBox);
    if (!hit) return;

    // current hit in LOCAL space
    const localHit = hit.point.clone();
    activeBox.worldToLocal(localHit);

    const deltaLocal = localHit.clone().sub(resizeStartLocalHit);

    // movement along chosen local axis
    const delta =
      resizeAxisLocal.x ? deltaLocal.x :
      resizeAxisLocal.y ? deltaLocal.y :
                          deltaLocal.z;

    // dragging one face:
    // size change = delta * sign
    // center shift = (delta/2)*sign along WORLD axis
    const minSize = 0.05;

    let newDx = resizeStartSize.dx;
    let newDy = resizeStartSize.dy;
    let newDz = resizeStartSize.dz;

    if (resizeAxisLocal.x) newDx = Math.max(minSize, resizeStartSize.dx + delta * resizeSign);
    if (resizeAxisLocal.y) newDy = Math.max(minSize, resizeStartSize.dy + delta * resizeSign);
    if (resizeAxisLocal.z) newDz = Math.max(minSize, resizeStartSize.dz + delta * resizeSign);

    // compute actual applied delta (after clamp)
    const appliedDelta =
      resizeAxisLocal.x ? (newDx - resizeStartSize.dx) :
      resizeAxisLocal.y ? (newDy - resizeStartSize.dy) :
                          (newDz - resizeStartSize.dz);

    // center shift by half the applied delta
    const shift = (appliedDelta * 0.5) * resizeSign;

    activeBox.userData.size.dx = newDx;
    activeBox.userData.size.dy = newDy;
    activeBox.userData.size.dz = newDz;

    activeBox.userData.center.x = resizeStartCenter.x + resizeAxisWorld.x * shift;
    activeBox.userData.center.y = resizeStartCenter.y + resizeAxisWorld.y * shift;
    activeBox.userData.center.z = resizeStartCenter.z + resizeAxisWorld.z * shift;

    syncMeshFromUserData(activeBox);
    return;
  }
}

// ---------- mouseup ----------
function onEditMouseUp() {
  if (!editMode) return;

  editMode = null;
  controls.enabled = true;
}

// ---------- 8) sync mesh from userData (NO SCALE) ----------
function syncMeshFromUserData(mesh) {
  const ud = mesh.userData;
  if (!ud?.center || !ud?.size) return;

  // position + yaw
  mesh.position.set(ud.center.x, ud.center.y, ud.center.z);
  mesh.rotation.set(0, 0, ud.yaw || 0);

  // rebuild geometry with exact dx/dy/dz
  if (mesh.geometry) mesh.geometry.dispose();
  mesh.geometry = new THREE.BoxGeometry(ud.size.dx, ud.size.dy, ud.size.dz);
}

function normalizeAngle(a) {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

function kittiTypeFromCat(cat) {
  if (!classIdToName || classIdToName[cat] == null) {
    return "Unknown";
  }

  // KITTI 通常首字母大写，可选
  const name = classIdToName[cat];
  return name.charAt(0).toUpperCase() + name.slice(1);
}


function boxToKittiLine(mesh) {
  const d = mesh.userData;
  if (!d) return null;

  const { center, size, yaw, cat } = d;

  const type = kittiTypeFromCat(cat);

  const x = center.x;
  const y = center.y;
  const z = center.z - size.dz * 0.5; // ⚠️ 底中心

  const l = size.dx;
  const w = size.dy;
  const h = size.dz;

  return [
    type,
    x.toFixed(3),
    y.toFixed(3),
    z.toFixed(3),
    l.toFixed(3),
    w.toFixed(3),
    h.toFixed(3),
    yaw.toFixed(6),
  ].join(" ");
}

function exportBoxesToKitti() {
  if (!instanceBoxes.length) {
    alert("No boxes to export");
    return;
  }

  const lines = [];

  for (const mesh of instanceBoxes) {
    const line = boxToKittiLine(mesh);
    if (line) lines.push(line);
  }

  const content = lines.join("\n");
  const blob = new Blob([content], { type: "text/plain" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "000000.txt"; // KITTI 风格
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);

  URL.revokeObjectURL(url);

  console.log("Exported KITTI labels:", lines.length);
}

function onBoxKeyDown(e) {
  if (!activeBox) return;

  const ud = activeBox.userData;
  if (!ud?.center) return;

  let moved = false;

  // 当前 box 的朝向（yaw）
  const yaw = ud.yaw || 0;

  // 前方向（车头）
  const forward = new THREE.Vector3(
    Math.cos(yaw),
    Math.sin(yaw),
    0
  );

  // 左方向
  const left = new THREE.Vector3(
    -Math.sin(yaw),
     Math.cos(yaw),
     0
  );

  switch (e.key.toLowerCase()) {
    case "w": // 前
      ud.center.x += forward.x * MOVE_STEP;
      ud.center.y += forward.y * MOVE_STEP;
      moved = true;
      break;

    case "s": // 后
      ud.center.x -= forward.x * MOVE_STEP;
      ud.center.y -= forward.y * MOVE_STEP;
      moved = true;
      break;

    case "a": // 左
      ud.center.x += left.x * MOVE_STEP;
      ud.center.y += left.y * MOVE_STEP;
      moved = true;
      break;

    case "d": // 右
      ud.center.x -= left.x * MOVE_STEP;
      ud.center.y -= left.y * MOVE_STEP;
      moved = true;
      break;

    case "arrowup": // Z+
      ud.center.z += MOVE_STEP_Z;
      moved = true;
      break;

    case "arrowdown": // Z-
      ud.center.z -= MOVE_STEP_Z;
      moved = true;
      break;

    case "q": // 逆时针旋转（+yaw）
      ud.yaw = normalizeAngle((ud.yaw || 0) + ROTATE_STEP);
      moved = true;
      break;

    case "e": // 顺时针旋转（-yaw）
      ud.yaw = normalizeAngle((ud.yaw || 0) - ROTATE_STEP);
      moved = true;
      break;
      }

  if (moved) {
    syncMeshFromUserData(activeBox);
    updateYawArrow(activeBox);
    showBoxInfo(activeBox);
    highlightBox(activeBox);

    e.preventDefault();
  }
}

function applyBoxSize(box) {
  if (!box || !box.userData?.size) return;

  const dx = parseFloat(document.getElementById("box_dx").value);
  const dy = parseFloat(document.getElementById("box_dy").value);
  const dz = parseFloat(document.getElementById("box_dz").value);

  const MIN_SIZE = 0.05;

  if (
    !isFinite(dx) || !isFinite(dy) || !isFinite(dz) ||
    dx < MIN_SIZE || dy < MIN_SIZE || dz < MIN_SIZE
  ) {
    alert("Invalid box size");
    return;
  }

  // ✅ 只改 size，不动 center / yaw
  box.userData.size.dx = dx;
  box.userData.size.dy = dy;
  box.userData.size.dz = dz;

  // 同步 mesh
  syncMeshFromUserData(box);
  updateYawArrow(box);
  highlightBox(box);

  console.log("Box size updated:", dx, dy, dz);
}

function onKeyDown(e) {
  // Delete / Backspace 都支持
  if ((e.key === "Delete" || e.key === "Backspace") && activeBox) {
    deleteActiveBox();
    e.preventDefault();
  }
}

function deleteActiveBox() {
  if (!activeBox) return;

  const box = activeBox;

  console.log(
    "Deleting box:",
    box.userData?.cat,
    box.userData?.inst
  );

  // 1️⃣ 删除 yaw arrow（如果有）
  if (box.userData?.arrow) {
    scene.remove(box.userData.arrow);
    box.userData.arrow.geometry?.dispose?.();
    box.userData.arrow.material?.dispose?.();
    box.userData.arrow = null;
  }

  // 2️⃣ 从 scene 移除 box
  scene.remove(box);

  // 3️⃣ 释放资源
  if (box.geometry) box.geometry.dispose();
  if (box.material) box.material.dispose();

  // 4️⃣ 从 instanceBoxes 数组移除
  const idx = instanceBoxes.indexOf(box);
  if (idx >= 0) instanceBoxes.splice(idx, 1);

  // 5️⃣ 清空 UI
  ui.pointInfo.innerHTML = "<b>Box Info</b><br/>Deleted";

  // 6️⃣ 清空选中状态
  activeBox = null;

  console.log("Box deleted");
}

function setTopView() {
  if (!camera || !controls) return;

  // 1️⃣ 目标点：用当前 controls.target（最合理）
  const target = controls.target.clone();

  // 2️⃣ 计算一个合适的高度
  let height = 10;
  if (geom && geom.boundingBox) {
    const size = new THREE.Vector3();
    geom.boundingBox.getSize(size);
    height = Math.max(size.x, size.y) * 1.5;
  }

  // 3️⃣ 相机放在 Z 轴正上方
  camera.position.set(
    target.x,
    target.y,
    target.z + height
  );

  // 4️⃣ 非常重要：up 方向
  // 这样保证 X 向右，Y 向上
  camera.up.set(0, 1, 0);

  // 5️⃣ 看向目标
  camera.lookAt(target);

  // 6️⃣ 同步 OrbitControls
  controls.target.copy(target);
  controls.update();

  console.log("📐 Switched to Top View (Z+)");
}

function resetView() {
  if (!(camera instanceof THREE.PerspectiveCamera)) {
    camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      0.01,
      1e7
    );
    controls.object = camera;
  }
  fitCameraToPoints();
}

function setFrontView(axis = "x") {
  if (!geom) return;

  geom.computeBoundingBox();
  const box = geom.boundingBox;

  const center = new THREE.Vector3();
  box.getCenter(center);

  const size = new THREE.Vector3();
  box.getSize(size);

  const dist = Math.max(size.x, size.y, size.z) * 1.5;

  if (axis === "x") {
    camera.position.set(center.x + dist, center.y, center.z);
  } else if (axis === "y") {
    camera.position.set(center.x, center.y + dist, center.z);
  }

  camera.up.set(0, 0, 1); // Z 轴向上
  camera.lookAt(center);
  controls.target.copy(center);
  controls.update();
}

const btnColorMode = document.getElementById("toggleColorMode");

btnColorMode.addEventListener("click", () => {
  if (colorMode === "category") {
    colorMode = "rgb";
    btnColorMode.textContent = "Color: Category";
  } else {
    colorMode = "category";
    btnColorMode.textContent = "Color: RGB";
  }

  recolor();
});

function buildInstanceFilter() {
  const sel = document.getElementById("instanceFilter");
  sel.innerHTML = `<option value="all">All</option>`;

  const map = new Map();
  for (let i = 0; i < categoryArr.length; i++) {
    const cat = categoryArr[i];
    const inst = instanceArr[i];
    const key = `${cat}_${inst}`;
    map.set(key, { cat, inst });
  }

  [...map.values()]
    .sort((a, b) => a.cat - b.cat || a.inst - b.inst)
    .forEach(({ cat, inst }) => {
      const opt = document.createElement("option");
      opt.value = `${cat}_${inst}`;
      opt.textContent =
        inst === 0 ? `Category ${cat}` : `Category ${cat} / Inst ${inst}`;
      sel.appendChild(opt);
    });

  sel.onchange = () => {
    applyInstanceFilter(sel.value);
  };
}

function applyInstanceFilter(val) {
  if (!visibilityMask) return;

  if (val === "all") {
    visibilityMask.fill(1);
    currentFilter = { cat: null, inst: null };
  } else {
    const [cat, inst] = val.split("_").map(Number);
    currentFilter = { cat, inst };

    for (let i = 0; i < visibilityMask.length; i++) {
      visibilityMask[i] =
        categoryArr[i] === cat &&
        instanceArr[i] === inst
          ? 1
          : 0;
    }
  }

  selectedMask.fill(0); // 清空选择
  recolorAll();
}

function buildPointCloudFromArrays(positions, N) {
  if (pointsObj) {
    scene.remove(pointsObj);
    pointsObj.geometry.dispose();
    pointsObj.material.dispose();
  }

  geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));

  const colors = new Float32Array(N * 3);
  geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  posAttr = geom.getAttribute("position");
  colorAttr = geom.getAttribute("color");

  const mat = new THREE.PointsMaterial({
    size: 2.0,
    vertexColors: true,
    sizeAttenuation: false,
  });

  pointsObj = new THREE.Points(geom, mat);
  scene.add(pointsObj);
}