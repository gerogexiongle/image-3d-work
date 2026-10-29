#!/usr/bin/env python3
"""Hi3D web service: image → coarse → Hi3D refine → Hunyuan Paint."""

from __future__ import annotations

import os
import shutil
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import trimesh
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import pipeline as pipe

ROOT = Path(os.environ.get("HI3D_ROOT", "/data1/xiongle/Hi3D")).resolve()
WEB_DIR = ROOT / "web"
STATIC_DIR = WEB_DIR / "static"
JOBS_DIR = WEB_DIR / "jobs"
EXAMPLE_IMAGE = ROOT / "src" / "assets" / "example.png"
# Prefer character demo if present, else official example
DEMO_IMAGE = ROOT / "outputs" / "user_demo" / "character.jpg"
if not DEMO_IMAGE.exists():
    DEMO_IMAGE = EXAMPLE_IMAGE

JOBS_DIR.mkdir(parents=True, exist_ok=True)

_job_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def _warm() -> None:
    try:
        pipe.warm_all()
        print("[Hi3D] models ready", pipe.status(), flush=True)
    except Exception as exc:  # noqa: BLE001
        print("[Hi3D] warm-up failed:", exc, flush=True)
        traceback.print_exc()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="Hi3D Image-to-3D", version="2.0", lifespan=lifespan)


@app.middleware("http")
async def _no_cache_static(request, call_next):
    response = await call_next(request)
    path = request.url.path or ""
    if path.startswith("/static/") and path.endswith((".js", ".css", ".html")):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _save_upload(upload: UploadFile, dest: Path) -> None:
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)


def _validate_params(resolution: int, steps: int) -> None:
    if resolution not in (512, 1024, 1536):
        raise HTTPException(400, "resolution must be 512, 1024, or 1536")
    if steps < 1 or steps > 100:
        raise HTTPException(400, "steps out of range")


def _new_job(**extra: Any) -> str:
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "message": "已入队",
        "created_at": time.time(),
        **extra,
    }
    return job_id


def _run_generate(
    job_id: str,
    image_path: Path,
    resolution: int,
    steps: int,
    seed: int,
    remove_bg: bool,
    coarse_backend: str,
    remove_slabs: bool,
    apply_texture: bool,
) -> None:
    job = _jobs[job_id]
    job["status"] = "running"
    job["started_at"] = time.time()
    job_dir = JOBS_DIR / job_id

    def progress(msg: str) -> None:
        job["message"] = msg

    try:
        with _job_lock:
            result = pipe.run_image_to_mesh(
                image_path=image_path,
                job_dir=job_dir,
                resolution=resolution,
                steps=steps,
                seed=seed,
                remove_bg=remove_bg,
                coarse_backend=coarse_backend,
                remove_slabs=remove_slabs,
                apply_texture=apply_texture,
                progress=progress,
                job_tag=job_id,
            )
        out_path = Path(result.get("output") or result["refined"])
        timings = result.get("timings") or {}
        cleanup = result.get("cleanup_meta") or {}
        texture_meta = result.get("texture_meta") or {}
        textured_ok = bool(result.get("textured_applied"))
        wall = round(time.time() - job["started_at"], 3)
        coarse_key = f"coarse_{coarse_backend}"
        coarse_s = timings.get(coarse_key, timings.get("coarse_backend_seconds", 0))
        slab_s = timings.get("coarse_remove_slabs", 0)
        paint_s = timings.get("paint_infer", 0)
        job["status"] = "done"
        job["timings"] = timings
        job["cleanup_meta"] = cleanup
        job["texture_meta"] = texture_meta
        job["textured"] = textured_ok
        job["coarse_backend"] = result.get("coarse_backend", coarse_backend)
        job["coarse_backend_label"] = result.get("coarse_backend_label", coarse_backend)
        tex_note = "贴图OK" if textured_ok else ("几何-only" if not apply_texture else "贴图失败→几何")
        job["message"] = (
            f"完成 [{job['coarse_backend_label']}|{tex_note}] "
            f"total={timings.get('total', wall):.2f}s "
            f"(rembg={timings.get('rembg', 0):.2f}s, "
            f"coarse={coarse_s:.2f}s, "
            f"slabs={slab_s:.2f}s, "
            f"hi3d={timings.get(f'hi3d_refine_{resolution}', 0):.2f}s, "
            f"paint={paint_s:.2f}s)"
        )
        job["output"] = f"/api/jobs/{job_id}/download"
        job["refined"] = f"/api/jobs/{job_id}/refined"
        job["coarse"] = f"/api/jobs/{job_id}/coarse"
        job["coarse_raw"] = f"/api/jobs/{job_id}/coarse_raw"
        job["prepared"] = f"/api/jobs/{job_id}/prepared"
        if textured_ok:
            job["textured_url"] = f"/api/jobs/{job_id}/textured"
        job["finished_at"] = time.time()
        job["stats"] = {
            "vertices": result["vertices"],
            "faces": result["faces"],
            "coarse_vertices": result["coarse_vertices"],
            "coarse_faces": result["coarse_faces"],
            "coarse_backend": job["coarse_backend"],
            "coarse_backend_label": job["coarse_backend_label"],
            "remove_slabs": remove_slabs,
            "apply_texture": apply_texture,
            "textured": textured_ok,
            "slabs_removed_faces": cleanup.get("removed_faces", 0),
            "seconds": round(timings.get("total", wall), 1),
            "wall_seconds": wall,
            "size_mb": round(out_path.stat().st_size / (1024 * 1024), 2),
            "timings": timings,
            "cleanup_meta": cleanup,
            "texture_meta": texture_meta,
        }
        print(
            f"[job {job_id}] DONE backend={coarse_backend} textured={textured_ok} "
            f"wall={wall:.3f}s timings={timings}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        job["status"] = "error"
        job["message"] = str(exc)
        job["traceback"] = traceback.format_exc()
        job["finished_at"] = time.time()
        print(
            f"[job {job_id}] ERROR after {time.time()-job['started_at']:.3f}s: {exc}",
            flush=True,
        )


def _run_refine_only(
    job_id: str,
    image_path: Path,
    mesh_path: Path,
    resolution: int,
    steps: int,
    seed: int,
) -> None:
    job = _jobs[job_id]
    job["status"] = "running"
    job["message"] = "Hi3D 精修中…"
    job["started_at"] = time.time()
    out_path = JOBS_DIR / job_id / "refined.glb"
    try:
        with _job_lock:
            hi3d = pipe.load_hi3d()
            image = Image.open(image_path).convert("RGB")
            coarse = trimesh.load(mesh_path, force="mesh", process=False)
            result = hi3d(
                image,
                coarse,
                resolution=resolution,
                num_inference_steps=steps,
                seed=seed,
            )
            result.mesh.export(out_path)
            shutil.copy2(mesh_path, JOBS_DIR / job_id / "coarse.glb")
        job["status"] = "done"
        job["message"] = "完成"
        job["output"] = f"/api/jobs/{job_id}/download"
        job["coarse"] = f"/api/jobs/{job_id}/coarse"
        job["finished_at"] = time.time()
        job["stats"] = {
            "vertices": len(result.mesh.vertices),
            "faces": len(result.mesh.faces),
            "seconds": round(job["finished_at"] - job["started_at"], 1),
            "size_mb": round(out_path.stat().st_size / (1024 * 1024), 2),
        }
    except Exception as exc:  # noqa: BLE001
        job["status"] = "error"
        job["message"] = str(exc)
        job["traceback"] = traceback.format_exc()
        job["finished_at"] = time.time()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict[str, Any]:
    st = pipe.status()
    return {
        "ok": True,
        "mode": "image_to_3d",
        "pipeline_loaded": st["hi3d_loaded"] and st["triposr_loaded"],
        "busy": _job_lock.locked(),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        **st,
    }


@app.get("/api/backends")
def backends() -> dict[str, Any]:
    import coarse_backends as cb

    items = cb.list_backends()
    return {
        "default": "triposr",
        "backends": items,
        "available": [b["id"] for b in items if b["available"]],
    }


@app.get("/api/examples")
def examples() -> dict[str, Any]:
    return {
        "image": "/api/examples/image" if DEMO_IMAGE.exists() else None,
    }


@app.get("/api/examples/image")
def example_image() -> FileResponse:
    if not DEMO_IMAGE.exists():
        raise HTTPException(404, "example image missing")
    return FileResponse(DEMO_IMAGE, filename=DEMO_IMAGE.name)


@app.post("/api/generate")
async def generate(
    image: UploadFile = File(...),
    resolution: int = Form(512),
    steps: int = Form(25),
    seed: int = Form(0),
    remove_bg: str = Form("true"),
    coarse_backend: str = Form("triposr"),
    remove_slabs: str = Form("true"),
    apply_texture: str = Form("true"),
) -> dict[str, Any]:
    """Unified: image → coarse + slabs + Hi3D refine + Hunyuan Paint → GLB."""
    import coarse_backends as cb

    _validate_params(resolution, steps)
    remove_bg_flag = str(remove_bg).strip().lower() in ("1", "true", "yes", "on")
    remove_slabs_flag = str(remove_slabs).strip().lower() in ("1", "true", "yes", "on")
    apply_texture_flag = str(apply_texture).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    backend_id = (coarse_backend or "triposr").strip()
    try:
        cb.get_backend(backend_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc

    job_id = _new_job(
        mode="generate",
        resolution=resolution,
        steps=steps,
        seed=seed,
        remove_bg=remove_bg_flag,
        coarse_backend=backend_id,
        remove_slabs=remove_slabs_flag,
        apply_texture=apply_texture_flag,
    )
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(image.filename or "image.png").suffix or ".png"
    image_path = job_dir / f"input{suffix}"
    _save_upload(image, image_path)

    threading.Thread(
        target=_run_generate,
        args=(
            job_id,
            image_path,
            resolution,
            steps,
            seed,
            remove_bg_flag,
            backend_id,
            remove_slabs_flag,
            apply_texture_flag,
        ),
        daemon=True,
    ).start()
    return {
        "job_id": job_id,
        "status": "queued",
        "mode": "generate",
        "coarse_backend": backend_id,
        "remove_slabs": remove_slabs_flag,
        "apply_texture": apply_texture_flag,
    }


@app.post("/api/refine")
async def refine(
    image: UploadFile = File(...),
    coarse_mesh: UploadFile = File(...),
    resolution: int = Form(512),
    steps: int = Form(25),
    seed: int = Form(0),
) -> dict[str, Any]:
    """Advanced: refine with a provided coarse mesh."""
    _validate_params(resolution, steps)
    job_id = _new_job(mode="refine", resolution=resolution, steps=steps, seed=seed)
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    image_path = job_dir / f"input{Path(image.filename or 'image.png').suffix or '.png'}"
    mesh_path = job_dir / f"coarse_upload{Path(coarse_mesh.filename or 'mesh.glb').suffix or '.glb'}"
    _save_upload(image, image_path)
    _save_upload(coarse_mesh, mesh_path)

    threading.Thread(
        target=_run_refine_only,
        args=(job_id, image_path, mesh_path, resolution, steps, seed),
        daemon=True,
    ).start()
    return {"job_id": job_id, "status": "queued", "mode": "refine"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {k: v for k, v in job.items() if k != "traceback" or job["status"] == "error"}


@app.get("/api/jobs/{job_id}/download")
def job_download(job_id: str) -> FileResponse:
    """Prefer textured GLB when available; else geometry-only refined."""
    textured = JOBS_DIR / job_id / "textured.glb"
    refined = JOBS_DIR / job_id / "refined.glb"
    path = textured if textured.exists() else refined
    if not path.exists():
        raise HTTPException(404, "result not ready")
    suffix = "textured" if path.name == "textured.glb" else "refined"
    return FileResponse(
        path,
        media_type="model/gltf-binary",
        filename=f"hi3d_{suffix}_{job_id}.glb",
    )


@app.get("/api/jobs/{job_id}/refined")
def job_refined(job_id: str) -> FileResponse:
    path = JOBS_DIR / job_id / "refined.glb"
    if not path.exists():
        raise HTTPException(404, "refined mesh not ready")
    return FileResponse(
        path,
        media_type="model/gltf-binary",
        filename=f"hi3d_refined_{job_id}.glb",
    )


@app.get("/api/jobs/{job_id}/textured")
def job_textured(job_id: str) -> FileResponse:
    path = JOBS_DIR / job_id / "textured.glb"
    if not path.exists():
        raise HTTPException(404, "textured mesh not ready")
    return FileResponse(
        path,
        media_type="model/gltf-binary",
        filename=f"hi3d_textured_{job_id}.glb",
    )


@app.get("/api/jobs/{job_id}/coarse")
def job_coarse(job_id: str) -> FileResponse:
    path = JOBS_DIR / job_id / "coarse.glb"
    if not path.exists():
        raise HTTPException(404, "coarse mesh not ready")
    return FileResponse(
        path,
        media_type="model/gltf-binary",
        filename=f"coarse_{job_id}.glb",
    )


@app.get("/api/jobs/{job_id}/coarse_raw")
def job_coarse_raw(job_id: str) -> FileResponse:
    path = JOBS_DIR / job_id / "coarse_raw.glb"
    if not path.exists():
        raise HTTPException(404, "raw coarse mesh not ready")
    return FileResponse(
        path,
        media_type="model/gltf-binary",
        filename=f"coarse_raw_{job_id}.glb",
    )


@app.get("/api/jobs/{job_id}/prepared")
def job_prepared(job_id: str) -> FileResponse:
    path = JOBS_DIR / job_id / "prepared.png"
    if not path.exists():
        raise HTTPException(404, "prepared image not ready")
    return FileResponse(path, media_type="image/png", filename=f"prepared_{job_id}.png")


if __name__ == "__main__":
    host = os.environ.get("HI3D_HOST", "0.0.0.0")
    port = int(os.environ.get("HI3D_PORT", "17860"))
    uvicorn.run(app, host=host, port=port)
