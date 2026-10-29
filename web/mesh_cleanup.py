#!/usr/bin/env python3
"""Post-process coarse meshes: strip large planar slab artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh


def remove_large_planar_slabs(
    mesh: trimesh.Trimesh,
    *,
    area_frac_thresh: float = 0.05,
    normal_quant: float = 0.15,
    align_thresh: float = 0.98,
    keep_largest_component: bool = True,
    min_faces_keep: int = 500,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Remove dominant near-planar face clusters (background slabs).

    Hunyuan coarse meshes often contain two huge ±X (or similar) walls. We
    bucket faces by quantized normals and drop buckets whose total area is a
    large fraction of the mesh and whose normals are highly aligned.
    """
    if mesh is None or len(mesh.faces) == 0:
        return mesh, {"skipped": True, "reason": "empty"}

    work = mesh.copy()
    # ensure face data
    areas = np.asarray(work.area_faces, dtype=np.float64)
    normals = np.asarray(work.face_normals, dtype=np.float64)
    total_area = float(areas.sum()) + 1e-12

    # quantize normals (use absolute for opposite sides of same wall plane family
    # but keep sign so +X and -X are separate slabs — both should go)
    q = np.round(normals / max(normal_quant, 1e-6)) * normal_quant
    # snap tiny values to 0 for stabler keys
    q[np.abs(q) < normal_quant * 0.5] = 0.0

    buckets: dict[tuple[float, float, float], list[int]] = {}
    for i, n in enumerate(q):
        key = (float(n[0]), float(n[1]), float(n[2]))
        buckets.setdefault(key, []).append(i)

    remove = np.zeros(len(work.faces), dtype=bool)
    removed_buckets = []
    for key, idxs in buckets.items():
        idx = np.asarray(idxs, dtype=np.int64)
        area = float(areas[idx].sum())
        frac = area / total_area
        if frac < area_frac_thresh:
            continue
        ns = normals[idx]
        mean = ns.mean(axis=0)
        norm = np.linalg.norm(mean) + 1e-12
        mean = mean / norm
        align = float(np.abs(ns @ mean).mean())
        if align < align_thresh:
            continue
        remove[idx] = True
        removed_buckets.append(
            {
                "normal_key": key,
                "faces": int(len(idx)),
                "area_frac": round(frac, 4),
                "align": round(align, 5),
            }
        )

    faces_before = int(len(work.faces))
    verts_before = int(len(work.vertices))
    if not remove.any():
        meta = {
            "applied": True,
            "removed_faces": 0,
            "faces_before": faces_before,
            "faces_after": faces_before,
            "verts_before": verts_before,
            "verts_after": verts_before,
            "removed_buckets": [],
            "area_frac_thresh": area_frac_thresh,
        }
        return work, meta

    keep_faces = work.faces[~remove]
    if len(keep_faces) < min_faces_keep:
        # safety: don't destroy the mesh
        return work, {
            "applied": False,
            "reason": "would_remove_too_much",
            "would_keep_faces": int(len(keep_faces)),
            "removed_buckets": removed_buckets,
            "faces_before": faces_before,
            "verts_before": verts_before,
        }

    cleaned = trimesh.Trimesh(
        vertices=work.vertices.copy(),
        faces=keep_faces.copy(),
        process=False,
    )
    cleaned.remove_unreferenced_vertices()

    if keep_largest_component and len(cleaned.faces) > 0:
        try:
            components = cleaned.split(only_watertight=False)
            if len(components) > 1:
                components = sorted(components, key=lambda m: len(m.faces), reverse=True)
                cleaned = components[0]
        except Exception:
            pass

    meta = {
        "applied": True,
        "removed_faces": int(remove.sum()),
        "faces_before": faces_before,
        "faces_after": int(len(cleaned.faces)),
        "verts_before": verts_before,
        "verts_after": int(len(cleaned.vertices)),
        "removed_buckets": removed_buckets,
        "area_frac_thresh": area_frac_thresh,
        "components_kept": 1,
    }
    return cleaned, meta
