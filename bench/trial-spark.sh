#!/usr/bin/env bash
# One trial inside exclusive-spark.sh; save logs even on startup failure.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $(hostname) != dgx-spark ]]; then
  echo 'Run this on dgx-spark, not the workstation.' >&2; exit 2
fi
if (($# < 2)); then echo "Usage: $0 LABEL PROFILE [vllm options ...]" >&2; exit 2; fi
: "${SPARK_EXPERIMENT_UNIT:?Run inside exclusive-spark.sh}"
: "${SPARK_EXPERIMENT_OWNER:?Run inside exclusive-spark.sh}"
label=$1; profile=$2; shift 2
[[ $label =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid label' >&2; exit 2; }
if [[ $profile == sglang && $# != 0 ]]; then echo 'SGLang baseline does not accept overrides' >&2; exit 2; fi
if systemctl is-active --quiet sglang.service; then
  echo 'SGLang is still running; use exclusive-spark.sh for isolated trials.' >&2; exit 1
fi
unit=$SPARK_EXPERIMENT_UNIT
if [[ $(systemctl --user show "$unit" -p LoadState --value) != not-found ]]; then
  echo "$unit already exists" >&2; exit 1
fi
# Reject existing listeners, including non-HTTP servers, before trusting /health.
python3 - "${PORT:-8000}" <<'PY'
import socket, sys
with socket.socket() as s:
    s.bind(("127.0.0.1", int(sys.argv[1])))
PY
out="experiments/$label"
mkdir -p "$out"
[[ ! -e $out/launch.txt ]] || { echo "Refusing to overwrite $out" >&2; exit 1; }
source bench/spark-unit.sh
invocation=''
start_time=$(date '+%Y-%m-%d %H:%M:%S')
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -z $invocation ]]; then
    invocation=$(systemctl --user show "$unit" -p InvocationID --value) || true
  fi
  if ! spark_stop_owned_unit; then status=1; fi
  local log_status=0
  if [[ -n $invocation ]]; then
    journalctl --user "_SYSTEMD_INVOCATION_ID=$invocation" --no-pager > "$out/server.log" || log_status=$?
  else
    journalctl --user -u "$unit" --since "$start_time" --no-pager > "$out/server.log" || log_status=$?
  fi
  if ((log_status)); then
    echo "Journal export failed (status $log_status); see user journal for $unit" >&2
    if ((status == 0)); then status=1; fi
  fi
  printf '%s\n' "$status" > "$out/exit-status"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
{
  date -Is
  uname -a
  nvidia-smi
  printf 'unit=%s\nprofile=%q\n' "$unit" "$profile"
  printf 'extra_args='; printf '%q ' "$@"; printf '\n'
  for key in VLLM_ENV_DIR PORT MAX_MODEL_LEN MAX_NUM_SEQS BATCHED_TOKENS GPU_MEMORY_UTILIZATION ATTENTION_BACKEND KV_CACHE_DTYPE SPEC_TOKENS; do
    printf '%s=%s\n' "$key" "${!key-<default>}"
  done
} > "$out/launch.txt"
env_args=()
for key in VLLM_ENV_DIR PORT MAX_MODEL_LEN MAX_NUM_SEQS BATCHED_TOKENS GPU_MEMORY_UTILIZATION ATTENTION_BACKEND KV_CACHE_DTYPE SPEC_TOKENS; do
  if [[ -v $key ]]; then env_args+=("--setenv=$key=${!key}"); fi
done
if [[ $profile == sglang ]]; then
  launch=(python3 bench/sglang-baseline.py serve)
else
  launch=(nix develop --command bash bench/serve-spark.sh "$profile" "$@")
fi
# Do NOT --collect: failed units must survive long enough to retrieve their logs.
# RemainAfterExit also retains a process that unexpectedly exits successfully.
systemd-run --user --unit="$unit" --description="$SPARK_EXPERIMENT_OWNER" \
  --working-directory="$PWD" --service-type=exec \
  --property=RemainAfterExit=yes --property=KillMode=control-group --property=TimeoutStopSec=60 \
  --property=MemoryMax=100G --property=MemorySwapMax=0 \
  "${env_args[@]}" "${launch[@]}"
invocation=$(systemctl --user show "$unit" -p InvocationID --value)
ready=0
for ((i=0; i<180; i++)); do
  if ! systemctl --user is-active --quiet "$unit" || [[ $(systemctl --user show "$unit" -p MainPID --value) == 0 ]]; then
    echo 'Server failed before readiness' >&2; exit 1
  fi
  if curl -fsS --max-time 3 "http://127.0.0.1:${PORT:-8000}/health" >/dev/null 2>&1; then
    if [[ $(systemctl --user show "$unit" -p InvocationID --value) == "$invocation" ]] && systemctl --user is-active --quiet "$unit"; then
      ready=1; break
    fi
  fi
  sleep 5
done
((ready)) || { echo 'Server startup timed out' >&2; exit 1; }
curl -fsS "http://127.0.0.1:${PORT:-8000}/v1/models" > "$out/models.json"
python3 - "$out/models.json" <<'PY'
import json, sys
assert "qwen3.8-27b" in {m["id"] for m in json.load(open(sys.argv[1]))["data"]}, "Unexpected model alias"
PY
python_bin="${VLLM_ENV_DIR:-.venv}/bin/python"
if [[ $profile == sglang ]]; then
  cp experiments/sglang-production.json "$out/sglang-production.json"
else
  "$python_bin" -c 'import json,importlib.metadata as m; print(json.dumps({p:m.version(p) for p in ["vllm","torch","flashinfer-python","transformers"]},indent=2))' > "$out/versions.json"
fi
for concurrency in ${CONCURRENCIES:-1 4}; do
  "$python_bin" bench/benchmark.py \
    --base-url "http://127.0.0.1:${PORT:-8000}/v1" --model qwen3.8-27b \
    --label "$label-c$concurrency" --output "$out/c$concurrency.json" \
    --concurrency "$concurrency" --requests "${REQUESTS:-4}" \
    --max-tokens "${OUTPUT_TOKENS:-512}" --prompt-repeat "${PROMPT_REPEAT:-1}" \
    --seed "$((42 + concurrency))"
done
