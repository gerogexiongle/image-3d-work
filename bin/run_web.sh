#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/xiongle/Hi3D
export PATH="$ROOT/env/bin:$PATH"
export HF_HOME="$ROOT/hf_cache"
export HI3D_ROOT="$ROOT"
export HI3D_MODEL="${HI3D_MODEL:-$ROOT/models/Hi3Dv1.0}"
export HI3D_HOST="${HI3D_HOST:-0.0.0.0}"
export HI3D_PORT="${HI3D_PORT:-17860}"
# Hunyuan Paint needs hy3dgen + compiled custom_rasterizer on PYTHONPATH
export PYTHONPATH="$ROOT/tools/Hunyuan3D-2:$ROOT/tools/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer${PYTHONPATH:+:$PYTHONPATH}"
# Default single-card deploy on GPU0 unless caller overrides explicitly before launch.
export CUDA_VISIBLE_DEVICES="${HI3D_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export HI3D_DEVICE="${HI3D_DEVICE:-cuda}"
unset CUDA_HOME

mkdir -p "$ROOT/web/jobs" "$ROOT/web/logs"
PID_FILE="$ROOT/web/logs/web.pid"
LOG_FILE="$ROOT/web/logs/web.log"

usage() {
  echo "Usage: $0 [--restart|--stop|--status]"
}

port_pids() {
  ss -lntp 2>/dev/null | awk -v p=":${HI3D_PORT}" '
    index($4, p) {
      while (match($0, /pid=[0-9]+/)) {
        print substr($0, RSTART+4, RLENGTH-4)
        $0 = substr($0, RSTART+RLENGTH)
      }
    }' | sort -u
}

is_healthy() {
  curl -fsS "http://127.0.0.1:${HI3D_PORT}/api/health" >/dev/null 2>&1
}

stop_web() {
  local pids
  pids="$(port_pids)"
  if [[ -z "${pids}" && -f "$PID_FILE" ]]; then
    pids="$(cat "$PID_FILE" 2>/dev/null || true)"
  fi
  if [[ -z "${pids}" ]]; then
    echo "Hi3D web: not running"
    return 0
  fi
  echo "Hi3D web: stopping pid(s): ${pids}"
  # shellcheck disable=SC2086
  kill ${pids} 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    [[ -z "$(port_pids)" ]] && break
    sleep 0.5
  done
  if [[ -n "$(port_pids)" ]]; then
    # shellcheck disable=SC2086
    kill -9 $(port_pids) 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "Hi3D web: stopped"
}

status_web() {
  local pids
  pids="$(port_pids)"
  if [[ -z "${pids}" ]]; then
    echo "Hi3D web: not running (port ${HI3D_PORT} free)"
    return 1
  fi
  echo "Hi3D web: running pid=${pids} port=${HI3D_PORT}"
  if is_healthy; then
    curl -sS "http://127.0.0.1:${HI3D_PORT}/api/health"
    echo
    echo "URL: http://10.0.3.253:${HI3D_PORT}"
  else
    echo "health check failed"
    return 1
  fi
}

cmd="${1:-}"
case "${cmd}" in
  -h|--help) usage; exit 0 ;;
  --stop|stop) stop_web; exit 0 ;;
  --status|status) status_web; exit $? ;;
  --restart|restart) stop_web; shift || true ;;
  "") ;;
  *) echo "unknown arg: ${cmd}"; usage; exit 2 ;;
esac

existing="$(port_pids)"
if [[ -n "${existing}" ]]; then
  if is_healthy; then
    echo "Hi3D web already running (pid=${existing}) on :${HI3D_PORT}"
    echo "URL: http://10.0.3.253:${HI3D_PORT}"
    echo "Use: $0 --restart   # 重启"
    echo "     $0 --stop      # 停止"
    echo "     $0 --status    # 状态"
    exit 0
  fi
  echo "port ${HI3D_PORT} busy but unhealthy; restarting…"
  stop_web
fi

cd "$ROOT/web"
echo "Hi3D web: starting on ${HI3D_HOST}:${HI3D_PORT} (GPU ${CUDA_VISIBLE_DEVICES})"
# foreground when attached to TTY; otherwise daemonize for convenience
if [[ -t 1 ]]; then
  echo $$ > "$PID_FILE"
  exec python -u app.py
else
  nohup python -u app.py >>"$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  sleep 1
  status_web || true
fi
