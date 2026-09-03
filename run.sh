#!/usr/bin/env bash

uv run vllm serve RedHatAI/Qwen3.8-27B-INT4 \
  --limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0 \
  --max-num-seqs 8 \
  --max-model-len 32768 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data # --kv-cache-dtype fp8 \
