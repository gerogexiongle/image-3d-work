#!/usr/bin/env bash
# Optional heavy coarse backends (avoid installing into Hi3D main env by default).
set -euo pipefail
ROOT=/data1/xiongle/Hi3D
echo "InstantMesh / TRELLIS.2 建议使用独立 conda/micromamba 环境，避免冲掉 Hi3D(torch/triton)。"
echo
echo "1) TRELLIS.2:"
echo "   cd $ROOT/tools/TRELLIS.2"
echo "   # 按官方 README: . ./setup.sh --new-env --basic --flash-attn ..."
echo "   # 装好后启动 Web 前: export ENABLE_TRELLIS2=1"
echo
echo "2) InstantMesh:"
echo "   cd $ROOT/tools/InstantMesh && pip install -r requirements.txt  # 建议独立 env"
echo "   # 启动 Web 前: export ENABLE_INSTANTMESH=1"
echo
echo "当前主环境已可用: triposr / hunyuan_mini / hunyuan_v2"
echo "对照接口: POST /api/generate 字段 coarse_backend=triposr|hunyuan_mini|hunyuan_v2"
