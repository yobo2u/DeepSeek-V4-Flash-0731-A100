#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/work/models/dsv4-runtime/deepseek-v4-a100-sglang-v0516"
SGLANG_ROOT="/root/work/models/dsv4-runtime/venv-sglang0516"  # SGLang 0.5.16 installed here
MODEL_PATH="/root/work/models/DeepSeek-V4-Flash-0731-BF16-MXFP4"
VENV="/root/work/models/dsv4-runtime/venv-sglang0516"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${PROJECT_ROOT}:${SGLANG_ROOT}/lib/python3.12/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export ENABLE_SGLANG_DSV4_A100_PATCH=1
export SGLANG_DSV4_A100_DIRECT_ATTENTION=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_TOPK_TRANSFORM_512_TORCH=0
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_DSV4_A100_INT8_INDEXER=1
export SGLANG_DSV4_INDEXER_QUERY_CP_PREFILL=1
export SGLANG_DSV4_FP4_EXPERTS=1
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_DSV4_MXFP4_MOE_BACKEND=mxfp4_int8

# Compatibility check
"${VENV}/bin/python" "${PROJECT_ROOT}/scripts/check_compatibility.py" \
  --sglang-root "${SGLANG_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --allow-sglang-commit-mismatch

# Log to file (needed by benchmark.py for accept rate/len parsing)
LOG_DIR="/root/work/models/dsv4-runtime/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/server_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${LOG_FILE}"

# Launch (tee to log file)
"${VENV}/bin/python" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --quantization fp8 \
  --moe-runner-backend marlin \
  --tp-size 8 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 \
  --port 8082 \
  --served-model-name deepseek-v4-flash-0731 \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --speculative-algorithm DSPARK \
  --chunked-prefill-size 32768 \
  --max-running-requests 16 \
  --watchdog-timeout 1800 \
  --context-length 1048576 \
  2>&1 | tee "${LOG_FILE}"