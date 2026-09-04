#!/usr/bin/env bash

MODEL_NAME="Qwen3.8-27B-INT4"
CONTEXT_SIZE=32768

uv run vllm serve RedHatAI/Qwen3.8-27B-INT4 \
  --served-model-name="$MODEL_NAME" \
  --limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0 \
  --max-num-seqs 8 \
  --max-model-len "$CONTEXT_SIZE" \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --host 0.0.0.0
