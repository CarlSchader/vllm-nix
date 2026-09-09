#!/usr/bin/env bash

# MODEL_NAME="Qwen3.8-27B-INT4"

# uv run vllm serve RedHatAI/Qwen3.8-27B-INT4 \
#   --served-model-name="$MODEL_NAME" \
#   --limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0 \
#   --max-num-seqs 16 \
#   --max-num-batched-tokens 16384 \
#   --gpu-memory-utilization 0.85 \
#   --tensor-parallel-size 1 \
#   --enable-auto-tool-choice \
#   --tool-call-parser qwen3_coder \
#   --reasoning-parser qwen3 \
#   --mm-encoder-tp-mode data \
#   --host 0.0.0.0

# vllm serve Qwen/Qwen3.8-27B-FP8 \
#   --tensor-parallel-size 1 \
#   --enable-auto-tool-choice \
#   --tool-call-parser qwen3_coder \
#   --reasoning-parser qwen3 \
#   --mm-encoder-tp-mode data \
#   --host 0.0.0.0

vllm serve Inferact/Qwen3.8-27B-NVFP4 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --attention-backend TRITON_ATTN \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
# NOTE: MTP speculative decoding disabled — the eagle draft head needs a
# FlashInfer head_dim=256 ragged-prefill kernel that fails to build on this
# Blackwell (sm_121a) / flashinfer 0.6.16.post3 combo (undefined symbol).
# --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
