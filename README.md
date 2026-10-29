# image-3d-work · Hi3D 图生 3D 工作台

单卡部署的 **图像 → 3D** Web 服务：粗模（多后端对照）→ Hi3D 几何精修 → Hunyuan Paint 贴图，浏览器 Three.js 预览。

默认服务端口：`17860`（`http://<host>:17860`）。内网直连或经跳板/反代均可；静态资源与 API 使用相对路径。

若贴图预览空白，请对页面 **强制刷新**（避免缓存旧 `app.js`），并确认反代对 `/api/jobs/*/download` 的超时与 body 大小足够（贴图 GLB 约 5MB+）。

## 能力流水线

```
抠图(rembg) → 粗模(TripoSR / Hunyuan mini / …)
         → [可选] 剔大平面伪影
         → Hi3D 精修（仅几何）
         → [可选] Hunyuan Paint 贴图
         → GLB 预览 / 下载
```

- **粗模 / Hi3D 精修本身没有颜色**；勾选「生成贴图」后才会按参考图上色。
- 前端会保留贴图材质；未贴图时显示 clay 白模。

## 仓库结构

| 路径 | 说明 |
|------|------|
| `web/` | FastAPI 后端 + 静态前端（`app.py` / `pipeline.py` / `static/`） |
| `bin/run_web.sh` | 启停 Web（`--restart` / `--status` / `--stop`） |
| `tools/` | 开源粗模 / 贴图依赖源码（TripoSR、Hunyuan3D-2、InstantMesh、TRELLIS.2） |
| `src/` | Hi3D / Sparc3D 相关源码 |
| `models/` / `hf_cache/` / `env/` | **本地权重与 conda 环境**（已 gitignore，需自行准备） |

## 本地准备（权重 & 环境）

本仓库不包含：

- micromamba/conda 环境（`env/`）
- HuggingFace 缓存（`hf_cache/`）
- Hi3D / Hunyuan 等大模型权重（`models/`）

在已有环境的机器上：

```bash
export HI3D_ROOT=/path/to/image-3d-work
export PATH="$HI3D_ROOT/env/bin:$PATH"
export HF_HOME="$HI3D_ROOT/hf_cache"
export PYTHONPATH="$HI3D_ROOT/tools/Hunyuan3D-2:$HI3D_ROOT/tools/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer${PYTHONPATH:+:$PYTHONPATH}"

# 启动
bash bin/run_web.sh
# 或重启
bash bin/run_web.sh --restart
```

Hunyuan Paint 的 `custom_rasterizer` CUDA 扩展需按本机 GPU 架构编译（例如 L20D / sm_100+）。仓库可附带预编译 `.so`，换卡时请在 `tools/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer` 下重新 `build_ext --inplace`。

## API 摘要

- `GET /api/health` — 健康检查与后端列表
- `GET /api/backends` — 粗模后端可用性
- `POST /api/generate` — 图生 3D（表单：`image`, `coarse_backend`, `resolution`, `remove_slabs`, `apply_texture`, …）
- `GET /api/jobs/{id}` — 任务状态
- `GET /api/jobs/{id}/download` — 优先贴图 GLB，否则几何精修

## 许可说明

- 本工作台编排代码按仓库 `LICENSE` / `NOTICE`（若有）使用。
- **Hunyuan3D**、**Hi3D** 等第三方权重与代码遵循各自上游许可证（含腾讯 Hunyuan 社区协议等），商用前请自行核验。
