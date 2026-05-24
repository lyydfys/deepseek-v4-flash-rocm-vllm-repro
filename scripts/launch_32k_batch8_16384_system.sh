#!/usr/bin/env bash
set -euo pipefail

export MODEL_DIR=${MODEL_DIR:-/home/deepseek_hybrid/DeepSeek-V4-Flash}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-deepseek-v4-flash-amd-32k-batch8-16384}
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=600000
export VLLM_LOG_STATS_INTERVAL=30
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER=1
export AITER_FP8_MQA_BLOCK_KV=64
export AITER_PA_MQA_STAGE1_CHUNK_Q=32
export AITER_PA_MQA_STAGE1_CHUNK_K=64
export AITER_PA_MQA_CHUNK_K=64
export VLLM_DSV4_MHC_TORCH_FALLBACK=1
export VLLM_DSV4_FP8_EINSUM_TORCH_FALLBACK=1
export VLLM_DSV4_QNORM_ROPE_KV_TORCH_FALLBACK=1
export VLLM_DSV4_FLASHMLA_PREFILL_TORCH_FALLBACK=1
export VLLM_DSV4_FLASHMLA_DECODE_TORCH_FALLBACK=1

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --async-scheduling \
  --no-disable-hybrid-kv-cache-manager \
  --moe-backend triton \
  --disable-uvicorn-access-log
