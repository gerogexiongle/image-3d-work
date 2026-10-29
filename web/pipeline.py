#!/usr/bin/env python3
"""Image → coarse → Hi3D refine → [Hunyuan Paint] unified pipeline."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import trimesh
from PIL import Image

ROOT = Path(os.environ.get("HI3D_ROOT", "/data1/xiongle/Hi3D")).resolve()
TRIPOSR_DIR = ROOT / "tools" / "TripoSR"
HUNYUAN_DIR = ROOT / "tools" / "Hunyuan3D-2"
CUSTOM_RASTER_DIR = HUNYUAN_DIR / "hy3dgen" / "texgen" / "custom_rasterizer"
MODEL_DIR = Path(os.environ.get("HI3D_MODEL", ROOT / "models" / "Hi3Dv1.0"))
DEVICE = os.environ.get("HI3D_DEVICE", "cuda")
HF_CACHE = ROOT / "hf_cache"
TRIPOSR_MODEL = os.environ.get("TRIPOSR_MODEL", "stabilityai/TripoSR")
HUNYUAN_PAINT_REPO = os.environ.get("HUNYUAN_PAINT_REPO", "tencent/Hunyuan3D-2")
HUNYUAN_PAINT_SUBFOLDER = os.environ.get(
    "HUNYUAN_PAINT_SUBFOLDER", "hunyuan3d-paint-v2-0-turbo"
)
PAINT_MAX_FACES = int(os.environ.get("HI3D_PAINT_MAX_FACES", "50000"))

_hi3d: Any = None
_triposr: Any = None
_paint: Any = None
_hi3d_lock = threading.Lock()
_triposr_lock = threading.Lock()
_paint_lock = threading.Lock()


def _ensure_hunyuan_path() -> None:
    for p in (str(HUNYUAN_DIR), str(CUSTOM_RASTER_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _ensure_triposr_path() -> None:
    path = str(TRIPOSR_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class StageTimer:
    """Collect per-stage seconds and print structured logs."""

    def __init__(self, job_tag: str = "") -> None:
        self.job_tag = job_tag
        self.timings: dict[str, float] = {}
        self._t0 = time.perf_counter()

    def _prefix(self) -> str:
        return f"[pipeline][{self.job_tag}]" if self.job_tag else "[pipeline]"

    def span(self, name: str):
        return _StageSpan(self, name)

    def add(self, name: str, seconds: float) -> None:
        self.timings[name] = round(float(seconds), 3)

    def total(self) -> float:
        return round(time.perf_counter() - self._t0, 3)

    def log(self, msg: str) -> None:
        print(f"{self._prefix()} {msg}", flush=True)

    def summary(self) -> dict[str, float]:
        total = self.total()
        parts = ", ".join(f"{k}={v:.3f}s" for k, v in self.timings.items())
        self.log(f"TIMING summary total={total:.3f}s | {parts}")
        out = dict(self.timings)
        out["total"] = total
        return out


class _StageSpan:
    def __init__(self, timer: StageTimer, name: str) -> None:
        self.timer = timer
        self.name = name
        self._start = 0.0

    def __enter__(self) -> "_StageSpan":
        _sync_cuda()
        self._start = time.perf_counter()
        self.timer.log(f"START {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _sync_cuda()
        elapsed = time.perf_counter() - self._start
        self.timer.add(self.name, elapsed)
        status = "FAIL" if exc_type else "DONE"
        self.timer.log(f"{status} {self.name} elapsed={elapsed:.3f}s")


def prepare_image(
    image: Image.Image,
    remove_bg: bool = True,
    timer: Optional[StageTimer] = None,
) -> Image.Image:
    """Return RGB image suitable for TripoSR + Hi3D (gray bg if rembg)."""
    t = timer or StageTimer()

    with t.span("preprocess_load_convert"):
        image = image.convert("RGBA")

    if remove_bg:
        with t.span("rembg"):
            from rembg import remove

            image = remove(image)

    with t.span("preprocess_composite"):
        arr = np.asarray(image).astype(np.float32) / 255.0
        if arr.shape[-1] == 4:
            rgb = arr[:, :, :3] * arr[:, :, 3:4] + (1.0 - arr[:, :, 3:4]) * 0.5
        else:
            rgb = arr[:, :, :3]
        out = Image.fromarray((np.clip(rgb, 0, 1) * 255.0).astype(np.uint8), mode="RGB")
    return out


def load_triposr() -> Any:
    global _triposr
    if _triposr is not None:
        return _triposr
    with _triposr_lock:
        if _triposr is not None:
            return _triposr
        t0 = time.perf_counter()
        _ensure_triposr_path()
        from tsr.system import TSR

        os.environ.setdefault("HF_HOME", str(HF_CACHE))
        model = TSR.from_pretrained(
            TRIPOSR_MODEL,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.renderer.set_chunk_size(8192)
        model.to(DEVICE)
        model.eval()
        _triposr = model
        print(
            f"[pipeline] TripoSR ready on {DEVICE} load={time.perf_counter()-t0:.3f}s",
            flush=True,
        )
        return _triposr


def load_hi3d() -> Any:
    global _hi3d
    if _hi3d is not None:
        return _hi3d
    with _hi3d_lock:
        if _hi3d is not None:
            return _hi3d
        t0 = time.perf_counter()
        from sparc3d import Sparc3DPipeline

        cache_dir = HF_CACHE / "hub"
        pipe = Sparc3DPipeline.from_pretrained(
            str(MODEL_DIR),
            cache_dir=cache_dir if cache_dir.exists() else None,
            local_files_only=True,
            dino_model_id="facebook/dinov2-large",
            clip_model_id="openai/clip-vit-large-patch14",
        )
        pipe.to(DEVICE)
        _hi3d = pipe
        print(
            f"[pipeline] Hi3D ready on {DEVICE} load={time.perf_counter()-t0:.3f}s",
            flush=True,
        )
        return _hi3d


def refine_mesh(
    image_rgb: Image.Image,
    coarse: trimesh.Trimesh,
    out_path: Path,
    resolution: int = 512,
    steps: int = 25,
    seed: int = 0,
    timer: Optional[StageTimer] = None,
) -> trimesh.Trimesh:
    t = timer or StageTimer()
    with t.span("hi3d_ensure_model"):
        pipe = load_hi3d()

    with t.span(f"hi3d_refine_{resolution}"):
        result = pipe(
            image_rgb,
            coarse,
            resolution=resolution,
            num_inference_steps=steps,
            seed=seed,
        )

    with t.span("hi3d_export_glb"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.mesh.export(out_path)
    return result.mesh


def _decimate_for_paint(
    mesh: trimesh.Trimesh, max_faces: int = PAINT_MAX_FACES
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Reduce face count before UV unwrap / paint (Hi3D refine can be ~1M faces)."""
    n = int(len(mesh.faces))
    meta: dict[str, Any] = {"faces_before": n, "max_faces": max_faces}
    if n <= max_faces:
        meta.update({"applied": False, "faces_after": n})
        return mesh, meta
    from fast_simplification import simplify

    verts, faces = simplify(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
        target_count=int(max_faces),
    )
    out = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    meta.update({"applied": True, "faces_after": int(len(out.faces))})
    return out, meta


def load_paint() -> Any:
    global _paint
    if _paint is not None:
        return _paint
    with _paint_lock:
        if _paint is not None:
            return _paint
        t0 = time.perf_counter()
        os.environ.setdefault("HF_HOME", str(HF_CACHE))
        _ensure_hunyuan_path()
        # torch must be imported before custom_rasterizer (shared libs)
        import torch as _torch  # noqa: F401
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        pipe = Hunyuan3DPaintPipeline.from_pretrained(
            HUNYUAN_PAINT_REPO, subfolder=HUNYUAN_PAINT_SUBFOLDER
        )
        _paint = pipe
        print(
            f"[pipeline] Hunyuan Paint ready "
            f"({HUNYUAN_PAINT_SUBFOLDER}) load={time.perf_counter()-t0:.3f}s",
            flush=True,
        )
        return _paint


def _fix_paint_material(mesh: Any) -> Any:
    """Ensure exported GLB is viewable: metallicFactor default 1 looks black."""
    try:
        from trimesh.visual.material import PBRMaterial, SimpleMaterial
    except Exception:  # noqa: BLE001
        return mesh

    def _fix_one(m: Any) -> None:
        vis = getattr(m, "visual", None)
        if vis is None:
            return
        mat = getattr(vis, "material", None)
        if isinstance(mat, PBRMaterial):
            mat.metallicFactor = 0.05
            if mat.roughnessFactor is None:
                mat.roughnessFactor = 0.7
            mat.doubleSided = True
        elif isinstance(mat, SimpleMaterial):
            # Upgrade to PBR for more predictable GLB viewers
            img = getattr(mat, "image", None)
            vis.material = PBRMaterial(
                baseColorTexture=img,
                metallicFactor=0.05,
                roughnessFactor=0.7,
                doubleSided=True,
            )

    if isinstance(mesh, trimesh.Scene):
        for geom in mesh.geometry.values():
            _fix_one(geom)
    else:
        _fix_one(mesh)
    return mesh


def apply_hunyuan_paint(
    image_rgb: Image.Image,
    mesh: trimesh.Trimesh,
    out_path: Path,
    timer: Optional[StageTimer] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[Any, dict[str, Any]]:
    """Geometry + reference image → textured GLB via Hunyuan3D Paint."""
    t = timer or StageTimer()

    def _p(msg: str) -> None:
        if progress:
            progress(msg)
        t.log(msg)

    with t.span("paint_decimate"):
        paint_mesh, dec_meta = _decimate_for_paint(mesh)
    _p(
        f"贴图前减面 {dec_meta.get('faces_before')}→{dec_meta.get('faces_after')} "
        f"(max={dec_meta.get('max_faces')})"
    )

    with t.span("paint_ensure_model"):
        pipe = load_paint()

    # Prefer RGBA prompt when available; Paint recenters internally.
    prompt = image_rgb.convert("RGBA") if image_rgb.mode != "RGBA" else image_rgb

    _p("Hunyuan Paint 生成贴图…")
    with t.span("paint_infer"):
        textured = pipe(paint_mesh, image=prompt)

    with t.span("paint_export_glb"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        textured = _fix_paint_material(textured)
        textured.export(out_path)

    meta = {
        "backend": "hunyuan_paint",
        "subfolder": HUNYUAN_PAINT_SUBFOLDER,
        "decimate": dec_meta,
        "output": str(out_path),
    }
    return textured, meta


def run_image_to_mesh(
    image_path: Path,
    job_dir: Path,
    resolution: int = 512,
    steps: int = 25,
    seed: int = 0,
    remove_bg: bool = True,
    coarse_backend: str = "triposr",
    remove_slabs: bool = True,
    apply_texture: bool = True,
    progress: Optional[Callable[[str], None]] = None,
    job_tag: str = "",
) -> dict[str, Any]:
    """Full pipeline: image → coarse → [slabs] → Hi3D → [Hunyuan Paint]."""
    import coarse_backends as cb

    timer = StageTimer(job_tag=job_tag or job_dir.name)

    def _p(msg: str) -> None:
        if progress:
            progress(msg)
        timer.log(msg)

    _p("预处理图像…")
    with timer.span("preprocess_open"):
        raw = Image.open(image_path)
    image_rgb = prepare_image(raw, remove_bg=remove_bg, timer=timer)
    with timer.span("preprocess_save"):
        prep_path = job_dir / "prepared.png"
        image_rgb.save(prep_path)

    _p(f"生成粗网格（{coarse_backend}）…")
    coarse_raw_path = job_dir / "coarse_raw.glb"
    coarse_path = job_dir / "coarse.glb"
    with timer.span(f"coarse_{coarse_backend}"):
        coarse, coarse_meta = cb.generate_coarse(
            coarse_backend,
            image_rgb,
            coarse_raw_path,
            seed=seed,
            progress=_p,
        )

    cleanup_meta: dict[str, Any] = {"applied": False, "skipped": True}
    if remove_slabs:
        _p("粗模后处理：剔除大平面伪影…")
        from mesh_cleanup import remove_large_planar_slabs

        with timer.span("coarse_remove_slabs"):
            coarse, cleanup_meta = remove_large_planar_slabs(coarse)
            coarse.export(coarse_path)
        _p(
            "剔平面完成 "
            f"faces {cleanup_meta.get('faces_before')}→{cleanup_meta.get('faces_after')} "
            f"removed={cleanup_meta.get('removed_faces', 0)}"
        )
    else:
        with timer.span("coarse_export"):
            coarse.export(coarse_path)
        cleanup_meta = {"applied": False, "enabled": False}

    _p(f"Hi3D 精修 @ {resolution}（仅几何，无贴图）…")
    refined_path = job_dir / "refined.glb"
    refined = refine_mesh(
        image_rgb,
        coarse,
        refined_path,
        resolution=resolution,
        steps=steps,
        seed=seed,
        timer=timer,
    )

    textured_path: Optional[Path] = None
    texture_meta: dict[str, Any] = {"applied": False, "enabled": bool(apply_texture)}
    output_mesh = refined
    output_path = refined_path
    if apply_texture:
        _p("贴图阶段：Hunyuan Paint（几何→上色）…")
        textured_path = job_dir / "textured.glb"
        try:
            textured, texture_meta = apply_hunyuan_paint(
                image_rgb,
                refined,
                textured_path,
                timer=timer,
                progress=_p,
            )
            texture_meta["applied"] = True
            texture_meta["enabled"] = True
            output_mesh = textured
            output_path = textured_path
        except Exception as exc:  # noqa: BLE001
            texture_meta = {
                "applied": False,
                "enabled": True,
                "error": str(exc),
            }
            _p(f"贴图失败，回退几何预览: {exc}")
            textured_path = None

    timings = timer.summary()
    timings["coarse_backend_seconds"] = coarse_meta.get("seconds", 0)
    stage_keys = [
        "rembg",
        f"coarse_{coarse_backend}",
        "coarse_remove_slabs",
        f"hi3d_refine_{resolution}",
        "paint_infer",
        "total",
    ]
    _p(
        "完成 | "
        + f"backend={coarse_backend} | textured={bool(texture_meta.get('applied'))} | "
        + " | ".join(f"{k}={v:.2f}s" for k, v in timings.items() if k in stage_keys)
    )
    return {
        "prepared": prep_path,
        "coarse_raw": coarse_raw_path if coarse_raw_path.exists() else None,
        "coarse": coarse_path,
        "refined": refined_path,
        "textured": textured_path,
        "output": output_path,
        "textured_applied": bool(texture_meta.get("applied")),
        "coarse_backend": coarse_backend,
        "coarse_backend_label": coarse_meta.get("backend_label", coarse_backend),
        "coarse_vertices": len(coarse.vertices),
        "coarse_faces": len(coarse.faces),
        "vertices": len(output_mesh.vertices)
        if hasattr(output_mesh, "vertices")
        else len(refined.vertices),
        "faces": len(output_mesh.faces)
        if hasattr(output_mesh, "faces")
        else len(refined.faces),
        "timings": timings,
        "coarse_meta": coarse_meta,
        "cleanup_meta": cleanup_meta,
        "texture_meta": texture_meta,
        "remove_slabs": remove_slabs,
        "apply_texture": apply_texture,
    }


def warm_all() -> None:
    t0 = time.perf_counter()
    # Warm default coarse + Hi3D; paint loads on first textured job.
    try:
        load_triposr()
    except Exception as exc:  # noqa: BLE001
        print("[pipeline] TripoSR warm skip:", exc, flush=True)
    load_hi3d()
    print(f"[pipeline] warm_all done elapsed={time.perf_counter()-t0:.3f}s", flush=True)


def status() -> dict[str, Any]:
    import coarse_backends as cb

    return {
        "triposr_loaded": _triposr is not None,
        "hi3d_loaded": _hi3d is not None,
        "paint_loaded": _paint is not None,
        "paint_repo": HUNYUAN_PAINT_REPO,
        "paint_subfolder": HUNYUAN_PAINT_SUBFOLDER,
        "device": DEVICE,
        "model": str(MODEL_DIR),
        "triposr_model": TRIPOSR_MODEL,
        "coarse_backends": cb.list_backends(),
        "pipeline_stages": [
            "rembg",
            "coarse",
            "remove_slabs",
            "hi3d_refine",
            "hunyuan_paint",
        ],
    }
