#!/usr/bin/env python3
"""Pluggable coarse mesh backends for Hi3D image-to-3D A/B testing."""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import trimesh
from PIL import Image

ROOT = Path(os.environ.get("HI3D_ROOT", "/data1/xiongle/Hi3D")).resolve()
TOOLS = ROOT / "tools"
HF_CACHE = ROOT / "hf_cache"
DEVICE = os.environ.get("HI3D_DEVICE", "cuda")

ProgressFn = Optional[Callable[[str], None]]


@dataclass
class BackendInfo:
    id: str
    label: str
    description: str
    license: str
    available: bool = False
    reason: str = ""
    recommended: bool = False
    tags: list[str] = field(default_factory=list)


class CoarseBackend:
    id: str = "base"
    label: str = "Base"
    description: str = ""
    license: str = ""
    recommended: bool = False
    tags: list[str] = []

    def probe(self) -> BackendInfo:
        ok, reason = self._probe()
        return BackendInfo(
            id=self.id,
            label=self.label,
            description=self.description,
            license=self.license,
            available=ok,
            reason=reason,
            recommended=self.recommended,
            tags=list(self.tags),
        )

    def _probe(self) -> tuple[bool, str]:
        return True, ""

    def generate(
        self,
        image_rgb: Image.Image,
        out_path: Path,
        seed: int = 0,
        progress: ProgressFn = None,
    ) -> trimesh.Trimesh:
        raise NotImplementedError


class TripoSRBackend(CoarseBackend):
    id = "triposr"
    label = "TripoSR"
    description = "最快最轻；MIT；对照基线"
    license = "MIT"
    tags = ["fast", "mit", "baseline"]

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def _probe(self) -> tuple[bool, str]:
        if not (TOOLS / "TripoSR" / "tsr").exists():
            return False, "未找到 tools/TripoSR"
        return True, ""

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            path = str(TOOLS / "TripoSR")
            if path not in sys.path:
                sys.path.insert(0, path)
            from tsr.system import TSR

            os.environ.setdefault("HF_HOME", str(HF_CACHE))
            model = TSR.from_pretrained(
                os.environ.get("TRIPOSR_MODEL", "stabilityai/TripoSR"),
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            model.renderer.set_chunk_size(8192)
            model.to(DEVICE)
            model.eval()
            self._model = model
            return self._model

    def generate(self, image_rgb, out_path, seed=0, progress=None):
        if progress:
            progress("TripoSR 推理…")
        model = self._load()
        mc_res = int(os.environ.get("TRIPOSR_MC_RESOLUTION", "256"))
        with torch.no_grad():
            scene_codes = model([image_rgb], device=DEVICE)
            meshes = model.extract_mesh(scene_codes, True, resolution=mc_res)
        mesh = meshes[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(out_path)
        return mesh


class HunyuanBackend(CoarseBackend):
    def __init__(
        self,
        backend_id: str,
        label: str,
        repo: str,
        subfolder: str,
        description: str,
        recommended: bool = False,
        steps: int = 30,
        octree: int = 256,
    ) -> None:
        self.id = backend_id
        self.label = label
        self.description = description
        self.license = "Tencent Hunyuan Community (商用请自查)"
        self.recommended = recommended
        self.tags = ["official-hi3d-demo", "shape"]
        self.repo = repo
        self.subfolder = subfolder
        self.steps = steps
        self.octree = octree
        self._pipe = None
        self._lock = threading.Lock()

    def _probe(self) -> tuple[bool, str]:
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401

            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, f"hy3dgen 不可用: {e}"

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        with self._lock:
            if self._pipe is not None:
                return self._pipe
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

            os.environ.setdefault("HF_HOME", str(HF_CACHE))
            print(f"[coarse] loading Hunyuan {self.repo}/{self.subfolder}", flush=True)
            pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                self.repo,
                subfolder=self.subfolder,
                use_safetensors=True,
                device=DEVICE,
            )
            self._pipe = pipe
            return self._pipe

    def generate(self, image_rgb, out_path, seed=0, progress=None):
        if progress:
            progress(f"{self.label} 推理…")
        pipe = self._load()
        image = image_rgb.convert("RGBA")
        mesh = pipe(
            image=image,
            num_inference_steps=self.steps,
            octree_resolution=self.octree,
            generator=torch.Generator(device="cpu").manual_seed(int(seed)),
            output_type="trimesh",
        )[0]
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.Trimesh(
                vertices=getattr(mesh, "vertices"),
                faces=getattr(mesh, "faces"),
                process=False,
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(out_path)
        return mesh


class InstantMeshBackend(CoarseBackend):
    id = "instantmesh"
    label = "InstantMesh"
    description = "腾讯 ARC；质量通常优于 TripoSR（需独立依赖）"
    license = "Apache-2.0 (请核验权重条款)"
    tags = ["quality"]

    def _probe(self) -> tuple[bool, str]:
        if not (TOOLS / "InstantMesh").exists():
            return False, "未克隆 tools/InstantMesh"
        if os.environ.get("ENABLE_INSTANTMESH", "0") != "1":
            return False, "默认关闭；设 ENABLE_INSTANTMESH=1 并装依赖后启用"
        return False, "进程内未完整接入，请用高级上传或后续独立环境"

    def generate(self, image_rgb, out_path, seed=0, progress=None):
        raise RuntimeError("InstantMesh 尚未在本环境启用")


class Trellis2Backend(CoarseBackend):
    id = "trellis2"
    label = "TRELLIS.2"
    description = "微软；几何强，建议独立 conda（setup.sh）"
    license = "MIT"
    tags = ["sota", "mit", "heavy"]

    def __init__(self) -> None:
        self._pipe = None
        self._lock = threading.Lock()

    def _probe(self) -> tuple[bool, str]:
        if not (TOOLS / "TRELLIS.2").exists():
            return False, "未克隆 tools/TRELLIS.2"
        if os.environ.get("ENABLE_TRELLIS2", "0") != "1":
            return False, "默认关闭；按 TRELLIS.2/setup.sh 安装后设 ENABLE_TRELLIS2=1"
        try:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa: F401

            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, f"trellis2 未安装: {e}"

    def generate(self, image_rgb, out_path, seed=0, progress=None):
        if progress:
            progress("TRELLIS.2 推理…")
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        with self._lock:
            if self._pipe is None:
                self._pipe = Trellis2ImageTo3DPipeline.from_pretrained(
                    "microsoft/TRELLIS.2-4B"
                )
                self._pipe.cuda()
            pipe = self._pipe
        mesh = pipe.run(image_rgb)[0]
        try:
            mesh.simplify(16_777_216)
        except Exception:
            pass
        if not isinstance(mesh, trimesh.Trimesh):
            verts = mesh.vertices
            faces = mesh.faces
            if hasattr(verts, "detach"):
                verts = verts.detach().cpu().numpy()
            if hasattr(faces, "detach"):
                faces = faces.detach().cpu().numpy()
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(out_path)
        return mesh


_REGISTRY: dict[str, CoarseBackend] = {}
_DEFAULT = "triposr"


def _build_registry() -> dict[str, CoarseBackend]:
    backends: list[CoarseBackend] = [
        TripoSRBackend(),
        HunyuanBackend(
            backend_id="hunyuan_mini",
            label="Hunyuan3D-2 mini",
            repo=os.environ.get("HUNYUAN_MINI_REPO", "tencent/Hunyuan3D-2mini"),
            subfolder=os.environ.get(
                "HUNYUAN_MINI_SUBFOLDER", "hunyuan3d-dit-v2-mini-turbo"
            ),
            description="Hi3D 官方 Demo 粗模；推荐对照",
            recommended=True,
        ),
        HunyuanBackend(
            backend_id="hunyuan_v2",
            label="Hunyuan3D-2",
            repo=os.environ.get("HUNYUAN_V2_REPO", "tencent/Hunyuan3D-2"),
            subfolder=os.environ.get("HUNYUAN_V2_SUBFOLDER", "hunyuan3d-dit-v2-0"),
            description="完整 Hunyuan 形状模型（更重）",
        ),
        InstantMeshBackend(),
        Trellis2Backend(),
    ]
    return {b.id: b for b in backends}


def registry() -> dict[str, CoarseBackend]:
    global _REGISTRY
    if not _REGISTRY:
        _REGISTRY = _build_registry()
    return _REGISTRY


def list_backends() -> list[dict[str, Any]]:
    return [
        {
            "id": i.id,
            "label": i.label,
            "description": i.description,
            "license": i.license,
            "available": i.available,
            "reason": i.reason,
            "recommended": i.recommended,
            "tags": i.tags,
        }
        for i in (b.probe() for b in registry().values())
    ]


def get_backend(backend_id: Optional[str] = None) -> CoarseBackend:
    rid = (backend_id or os.environ.get("HI3D_COARSE_BACKEND") or _DEFAULT).strip()
    reg = registry()
    if rid not in reg:
        raise KeyError(f"未知粗模后端: {rid}; 可选: {list(reg)}")
    info = reg[rid].probe()
    if not info.available:
        raise RuntimeError(f"后端 {rid} 不可用: {info.reason}")
    return reg[rid]


def generate_coarse(
    backend_id: str,
    image_rgb: Image.Image,
    out_path: Path,
    seed: int = 0,
    progress: ProgressFn = None,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    backend = get_backend(backend_id)
    t0 = time.perf_counter()
    if progress:
        progress(f"粗模后端: {backend.label}")
    try:
        mesh = backend.generate(image_rgb, out_path, seed=seed, progress=progress)
    except Exception:
        print(f"[coarse] backend={backend.id} FAILED\n{traceback.format_exc()}", flush=True)
        raise
    elapsed = round(time.perf_counter() - t0, 3)
    meta = {
        "backend": backend.id,
        "backend_label": backend.label,
        "seconds": elapsed,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
    }
    print(
        f"[coarse] backend={backend.id} DONE {elapsed:.3f}s "
        f"verts={meta['vertices']} faces={meta['faces']}",
        flush=True,
    )
    return mesh, meta
