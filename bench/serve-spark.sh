#!/usr/bin/env bash
# Run inside `nix develop` on dgx-spark. No changes to the production service.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Keep the EXACT production checkpoint: FP8 attention + NVFP4 MLP/lm_head.
model=RadixArk/Qwen3.8-27B-NVFP4
revision=52d1adc5f38aa5ebf099c29ed7025ba34cfbb854
profile=${1:-baseline}
if (($#)); then shift; fi
if [[ $(hostname) != dgx-spark ]]; then
  echo 'Run this on dgx-spark, not the workstation.' >&2; exit 2
fi
args=(
  serve "$model" --revision "$revision"
  --served-model-name qwen3.8-27b
  --host 127.0.0.1 --port "${PORT:-8000}"
  --tensor-parallel-size 1 --dtype bfloat16
  --max-model-len "${MAX_MODEL_LEN:-262144}"
  --max-num-seqs "${MAX_NUM_SEQS:-8}"
  --max-num-batched-tokens "${BATCHED_TOKENS:-4096}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.50}"
  --attention-backend "${ATTENTION_BACKEND:-TRITON_ATTN}"
  # SGLang's auto resolves to FP8 target KV (confirmed in its startup log).
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}"
  --mamba-ssm-cache-dtype bfloat16
  --enable-prefix-caching
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
)
case "$profile" in
  baseline) ;;
  mtp) args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS:-2},\"revision\":\"$revision\",\"attention_backend\":\"TRITON_ATTN\",\"kv_cache_dtype\":\"bfloat16\"}") ;;
  dflash) args+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"z-lab/Qwen3.8-27B-DFlash2\",\"revision\":\"50307d4c4cde6860d4eee73e2547cd786fe8e8a4\",\"num_speculative_tokens\":${SPEC_TOKENS:-7},\"attention_backend\":\"TRITON_ATTN\",\"kv_cache_dtype\":\"bfloat16\"}") ;;
  *) echo "Unknown profile: $profile (baseline, mtp, dflash)" >&2; exit 2 ;;
esac
# Dedicated persistent compiler caches, separate from SGLang and old experiments.
venv=${VLLM_ENV_DIR:-.venv}
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm-spark-experiment/${venv##*/}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$VLLM_CACHE_ROOT/triton}"
export MAX_JOBS="${MAX_JOBS:-8}"
printf 'Launching: '; printf '%q ' "${venv}/bin/vllm" "${args[@]}" "$@"; printf '\n'
exec "${venv}/bin/vllm" "${args[@]}" "$@"
