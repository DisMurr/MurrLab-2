#!/usr/bin/env bash
set -euo pipefail

# Launch the GPT-OSS Responses API
# Honors optional environment variables:
#  - BACKEND: transformers|stub (default: transformers)
#  - BACKEND_DEVICE: gpu|cpu (default: gpu for transformers)
#  - CHECKPOINT: path to the model (default: ../gpt-oss/gpt-oss-20b)
#  - PORT: server port (default: 8000)
#  - GPU_MEM_CAP: VRAM cap for model load (e.g., 18GiB)
#  - CPU_MEM_CAP: CPU RAM cap (e.g., 48GiB)
#  - OFFLOAD_DIR: folder for disk offload (default: ./offload)
#  - BACKEND_DEVICE: set to 'cpu' to force CPU-only backend load

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
BACKEND_DIR="${SCRIPT_DIR}/../gpt-oss/gpt_oss/responses_api"
CHECKPOINT="${CHECKPOINT:-${SCRIPT_DIR}/../gpt-oss/gpt-oss-20b}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
BACKEND="${BACKEND:-transformers}"
BACKEND_DEVICE="${BACKEND_DEVICE:-gpu}"

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export GPU_MEM_CAP=${GPU_MEM_CAP:-18GiB}
export CPU_MEM_CAP=${CPU_MEM_CAP:-48GiB}
export OFFLOAD_DIR=${OFFLOAD_DIR:-${ROOT_DIR}/offload}
export BACKEND_DEVICE=${BACKEND_DEVICE:-}
mkdir -p "${OFFLOAD_DIR}"

export PYTHONPATH="${SCRIPT_DIR}/../gpt-oss:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}/.."

# Determine backend command
if [[ "${BACKEND}" == "stub" ]]; then
  CMD=(python -m gpt_oss.responses_api.serve
       --checkpoint "${CHECKPOINT}"
    --port "${PORT}"
    --host "${HOST}"
       --inference-backend stub)
  DESC="stub"
else
  if [[ "${BACKEND_DEVICE}" == "cpu" ]]; then
    export BACKEND_DEVICE=cpu
    export CUDA_VISIBLE_DEVICES=""
    echo "Forcing CPU-only backend load (BACKEND_DEVICE=cpu)"
  fi
  CMD=(python -m gpt_oss.responses_api.serve
       --checkpoint "${CHECKPOINT}"
    --port "${PORT}"
    --host "${HOST}"
       --inference-backend transformers)
  DESC="transformers"
fi

echo "Launching Responses API (${DESC}) on port ${PORT} in background..."
# Ensure logs directory
LOG_FILE="${SCRIPT_DIR}/backend.log"
nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
echo "Backend PID=$!  logs at ${LOG_FILE}"
# Detach script
exit 0
