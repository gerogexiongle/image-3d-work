import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = (id) => document.getElementById(id);

const imageInput = $("image");
const meshInput = $("mesh");
const imagePreview = $("image-preview");
const meshName = $("mesh-name");
const statusEl = $("status");
const statsEl = $("stats");
const downloadsEl = $("downloads");
const downloadEl = $("download");
const downloadCoarseEl = $("download-coarse");
const downloadRefinedEl = $("download-refined");
const reloadPreviewBtn = $("reload-preview");
const runBtn = $("run-btn");
const exampleBtn = $("example-btn");
const backendSelect = $("coarse-backend");
const backendHint = $("backend-hint");
let lastPreviewUrl = "";

let exampleImageBlob = null;
let useExampleImage = false;

async function loadBackends() {
  const res = await fetch("/api/backends");
  const data = await res.json();
  backendSelect.innerHTML = "";
  const items = data.backends || [];
  let firstAvail = data.default || "triposr";
  for (const b of items) {
    const opt = document.createElement("option");
    opt.value = b.id;
    opt.disabled = !b.available;
    const mark = b.available ? (b.recommended ? "★ " : "") : "（不可用）";
    opt.textContent = `${mark}${b.label}`;
    opt.title = b.available
      ? `${b.description} | ${b.license}`
      : `${b.reason || "不可用"}`;
    backendSelect.appendChild(opt);
    if (b.available && b.id === data.default) firstAvail = b.id;
    if (b.available && b.recommended) firstAvail = b.id;
  }
  // Prefer available default; if recommended hunyuan available keep it
  const avail = items.filter((b) => b.available).map((b) => b.id);
  if (avail.includes("triposr")) firstAvail = "triposr";
  if (avail.includes(data.default)) firstAvail = data.default;
  // Prefer recommended among available for first paint if user wants compare — keep TripoSR as default for baseline
  backendSelect.value = avail.includes("triposr") ? "triposr" : avail[0] || firstAvail;
  const cur = items.find((b) => b.id === backendSelect.value);
  backendHint.textContent = cur
    ? `${cur.description}${cur.available ? "" : " — " + cur.reason}`
    : "";
  backendSelect.addEventListener("change", () => {
    const b = items.find((x) => x.id === backendSelect.value);
    backendHint.textContent = b
      ? `${b.description}${b.available ? "" : " — " + b.reason}`
      : "";
  });
}

imageInput.addEventListener("change", () => {
  useExampleImage = false;
  const file = imageInput.files?.[0];
  if (!file) {
    imagePreview.hidden = true;
    return;
  }
  imagePreview.src = URL.createObjectURL(file);
  imagePreview.hidden = false;
});

meshInput.addEventListener("change", () => {
  const file = meshInput.files?.[0];
  meshName.textContent = file ? file.name : "";
});

function setStatus(text, kind = "") {
  statusEl.textContent = text;
  statusEl.className = "status" + (kind ? ` ${kind}` : "");
}

/* --- Three.js viewer --- */
const viewer = $("viewer");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0e0c);
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
camera.position.set(1.8, 1.2, 2.2);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
viewer.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.85));
const key = new THREE.DirectionalLight(0xffffff, 1.25);
key.position.set(3, 5, 2);
scene.add(key);
const fill = new THREE.DirectionalLight(0xffffff, 0.55);
fill.position.set(-3, 1, -2);
scene.add(fill);
const rim = new THREE.DirectionalLight(0xffffff, 0.35);
rim.position.set(0, 2, -4);
scene.add(rim);

const grid = new THREE.GridHelper(3, 12, 0x2a3a32, 0x1a2420);
grid.position.y = -1;
scene.add(grid);

let currentRoot = null;
const loader = new GLTFLoader();

function resize() {
  const w = viewer.clientWidth;
  const h = viewer.clientHeight;
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}
window.addEventListener("resize", resize);
resize();

(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();

function clearModel() {
  if (!currentRoot) return;
  scene.remove(currentRoot);
  currentRoot.traverse((obj) => {
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
      const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
      mats.forEach((m) => m.dispose?.());
    }
  });
  currentRoot = null;
}

function fitCamera(object) {
  const box = new THREE.Box3().setFromObject(object);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  object.position.sub(center);
  grid.position.y = -size.y / 2 - 0.02;
  camera.position.set(maxDim * 1.4, maxDim * 0.9, maxDim * 1.6);
  controls.target.set(0, 0, 0);
  controls.update();
}

function _pickMap(mat) {
  if (!mat) return null;
  return mat.map || mat.emissiveMap || mat.normalMap || null;
}

function _applyPreviewMaterial(obj) {
  const srcMats = Array.isArray(obj.material) ? obj.material : [obj.material];
  const maps = srcMats.map(_pickMap).filter(Boolean);
  const hasUv = obj.geometry?.attributes?.uv != null;
  const hasVertexColors = obj.geometry?.attributes?.color != null;

  if (maps.length || (hasUv && srcMats.some((m) => m && m.isMeshStandardMaterial))) {
    const next = srcMats.map((m, i) => {
      const map = maps[i] || _pickMap(m) || maps[0] || null;
      if (map) {
        map.colorSpace = THREE.SRGBColorSpace;
        map.flipY = false;
        map.needsUpdate = true;
      }
      // Rebuild a simple lit material so texture always shows
      // (avoids glTF default metalness=1 looking black/grey).
      return new THREE.MeshStandardMaterial({
        map,
        color: 0xffffff,
        metalness: 0.0,
        roughness: 0.85,
        vertexColors: false,
        side: THREE.DoubleSide,
      });
    });
    obj.material = next.length === 1 ? next[0] : next;
    obj.material.needsUpdate = true;
    return true;
  }

  obj.material = new THREE.MeshStandardMaterial({
    color: hasVertexColors ? 0xffffff : 0xc5d8cc,
    metalness: 0.05,
    roughness: 0.55,
    vertexColors: hasVertexColors,
    side: THREE.DoubleSide,
  });
  return false;
}

function loadGlbUrl(url) {
  clearModel();
  // bust HTTP cache for repeated job downloads
  const sep = url.includes("?") ? "&" : "?";
  const fetchUrl = `${url}${sep}t=${Date.now()}`;
  setStatus("下载并解析 GLB 预览…", "running");
  let lastPct = -1;
  return new Promise((resolve, reject) => {
    loader.load(
      fetchUrl,
      (gltf) => {
        currentRoot = gltf.scene;
        let hasTexture = false;
        currentRoot.traverse((obj) => {
          if (!obj.isMesh) return;
          if (_applyPreviewMaterial(obj)) hasTexture = true;
        });
        scene.add(currentRoot);
        fitCamera(currentRoot);
        setStatus(
          hasTexture
            ? "完成：已加载贴图模型，可旋转预览或下载 GLB"
            : "完成：当前为无贴图几何（白模），可旋转预览或下载 GLB",
          "done"
        );
        resolve();
      },
      (ev) => {
        if (!ev.lengthComputable || ev.total <= 0) return;
        const pct = Math.min(99, Math.round((ev.loaded / ev.total) * 100));
        if (pct === lastPct) return;
        lastPct = pct;
        const mb = (ev.loaded / (1024 * 1024)).toFixed(1);
        const totalMb = (ev.total / (1024 * 1024)).toFixed(1);
        setStatus(`加载预览 ${pct}%（${mb}/${totalMb} MB）…`, "running");
      },
      (err) => {
        setStatus("预览加载失败: " + err.message, "error");
        reject(err);
      }
    );
  });
}

async function loadExamples() {
  setStatus("加载示例…");
  const imgRes = await fetch("/api/examples/image");
  if (!imgRes.ok) throw new Error("示例图不可用");
  exampleImageBlob = await imgRes.blob();
  useExampleImage = true;
  imageInput.value = "";
  meshInput.value = "";
  meshName.textContent = "";
  imagePreview.src = URL.createObjectURL(exampleImageBlob);
  imagePreview.hidden = false;
  setStatus("已加载示例图，点「生成 3D」即可");
}

exampleBtn.addEventListener("click", () => {
  loadExamples().catch((e) => setStatus(String(e.message || e), "error"));
});

async function pollJob(jobId) {
  for (; ;) {
    const res = await fetch(`/api/jobs/${jobId}`);
    const job = await res.json();
    if (job.status === "queued" || job.status === "running") {
      setStatus(job.message || job.status, "running");
      await new Promise((r) => setTimeout(r, 1500));
      continue;
    }
    if (job.status === "error") throw new Error(job.message || "生成失败");
    return job;
  }
}

runBtn.addEventListener("click", async () => {
  try {
    const imageFile = useExampleImage
      ? new File([exampleImageBlob], "example.jpg", {
        type: exampleImageBlob.type || "image/jpeg",
      })
      : imageInput.files?.[0];
    const meshFile = meshInput.files?.[0];

    if (!imageFile) {
      setStatus("请先选择参考图像，或点「加载示例图」", "error");
      return;
    }

    runBtn.disabled = true;
    exampleBtn.disabled = true;
    downloadsEl.hidden = true;
    statsEl.hidden = true;
    setStatus("提交任务…", "running");

    const fd = new FormData();
    fd.append("image", imageFile);
    fd.append("resolution", $("resolution").value);
    fd.append("steps", $("steps").value);
    fd.append("seed", $("seed").value);

    let endpoint = "/api/generate";
    if (meshFile) {
      endpoint = "/api/refine";
      fd.append("coarse_mesh", meshFile);
    } else {
      fd.append("remove_bg", $("remove-bg").checked ? "true" : "false");
      fd.append("remove_slabs", $("remove-slabs").checked ? "true" : "false");
      fd.append(
        "apply_texture",
        $("apply-texture").checked ? "true" : "false"
      );
      fd.append("coarse_backend", backendSelect.value || "triposr");
    }

    const res = await fetch(endpoint, { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    const { job_id } = await res.json();
    const job = await pollJob(job_id);

    const url = job.output || `/api/jobs/${job_id}/download`;
    lastPreviewUrl = url;
    downloadEl.href = url;
    downloadEl.download = job.textured
      ? `hi3d_textured_${job_id}.glb`
      : `hi3d_refined_${job_id}.glb`;
    downloadEl.textContent = job.textured ? "下载贴图 GLB" : "下载几何 GLB";
    if (job.refined && job.textured) {
      downloadRefinedEl.href = job.refined;
      downloadRefinedEl.download = `hi3d_refined_${job_id}.glb`;
      downloadRefinedEl.hidden = false;
    } else if (downloadRefinedEl) {
      downloadRefinedEl.hidden = true;
    }
    if (job.coarse) {
      downloadCoarseEl.href = job.coarse;
      downloadCoarseEl.download = `coarse_${job_id}.glb`;
      downloadCoarseEl.hidden = false;
    } else {
      downloadCoarseEl.hidden = true;
    }
    if (reloadPreviewBtn) reloadPreviewBtn.hidden = false;
    downloadsEl.hidden = false;

    if (job.stats) {
      statsEl.hidden = false;
      const s = job.stats;
      const t = s.timings || job.timings || {};
      const rembg = t.rembg != null ? Number(t.rembg).toFixed(2) : "-";
      const tri =
        t.triposr_infer != null
          ? (
            Number(t.triposr_infer) + Number(t.triposr_extract_mesh || 0)
          ).toFixed(2)
          : "-";
      const hi3dKey = Object.keys(t).find((k) => k.startsWith("hi3d_refine_"));
      const hi3d = hi3dKey != null ? Number(t[hi3dKey]).toFixed(2) : "-";
      const backendLabel = s.coarse_backend_label || s.coarse_backend || "-";
      const coarseKey = Object.keys(t).find((k) => k.startsWith("coarse_"));
      const coarseSec =
        coarseKey != null
          ? Number(t[coarseKey]).toFixed(2)
          : t.coarse_backend_seconds != null
            ? Number(t.coarse_backend_seconds).toFixed(2)
            : tri;
      statsEl.innerHTML = `
        <span>粗模 ${backendLabel}</span>
        <span>顶点 ${s.vertices.toLocaleString()}</span>
        <span>三角面 ${s.faces.toLocaleString()}</span>
        <span>总耗时 ${s.seconds}s</span>
        <span>大小 ${s.size_mb} MB</span>
        <span>抠图 ${rembg}s</span>
        <span>粗模耗时 ${coarseSec}s</span>
        <span>剔平面 ${t.coarse_remove_slabs != null ? Number(t.coarse_remove_slabs).toFixed(2) : "-"}s</span>
        <span>精修 ${hi3d}s</span>
        <span>贴图 ${t.paint_infer != null ? Number(t.paint_infer).toFixed(2) : (s.textured ? "ok" : "关")}s</span>
        <span>${s.textured ? "已上色" : "仅几何"}</span>
        <span>剔面数 ${s.slabs_removed_faces != null ? s.slabs_removed_faces : "-"}</span>`;
    }
    await loadGlbUrl(url);
  } catch (e) {
    setStatus(String(e.message || e), "error");
  } finally {
    runBtn.disabled = false;
    exampleBtn.disabled = false;
  }
});

if (reloadPreviewBtn) {
  reloadPreviewBtn.addEventListener("click", () => {
    const url = lastPreviewUrl || downloadEl.href;
    if (!url) {
      setStatus("没有可预览的结果", "error");
      return;
    }
    loadGlbUrl(url).catch((e) => setStatus(String(e.message || e), "error"));
  });
}

loadBackends().catch((e) => {
  backendHint.textContent = "后端列表加载失败: " + (e.message || e);
});

fetch("/api/health")
  .then((r) => r.json())
  .then((h) => {
    const ready =
      h.pipeline_loaded || (h.hi3d_loaded && h.triposr_loaded)
        ? "模型已就绪"
        : "模型加载中（首次较慢）";
    const n = (h.coarse_backends || []).filter((b) => b.available).length;
    setStatus(`${ready} · GPU ${h.gpu} · 可用粗模 ${n} 个`);
  })
  .catch(() => setStatus("服务健康检查失败", "error"));
