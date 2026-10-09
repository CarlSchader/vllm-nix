#!/usr/bin/env bash
# Execute experiments on the Spark, with bounded shutdown and SGLang rollback.
# Prefer a systemd user session unit so SSH disconnects are harmless.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $(hostname) != dgx-spark ]]; then
  echo 'Run this on dgx-spark, not the workstation.' >&2; exit 2
fi
if (($# == 0)); then echo "Usage: $0 COMMAND [ARG ...]" >&2; exit 2; fi
exec 9>"${XDG_RUNTIME_DIR:?}/vllm-spark-experiment.lock"
flock -n 9 || { echo 'Another Spark experiment holds the lock.' >&2; exit 1; }
ctl=/run/current-system/sw/bin/systemctl
session=$(mktemp -d "$XDG_RUNTIME_DIR/vllm-spark-experiment.XXXXXXXX")
export SPARK_EXPERIMENT_UNIT="${session##*/}.service"
export SPARK_EXPERIMENT_OWNER="Spark experiment ${session##*/}"
# An absent unit is important: cleanup never commandeers a pre-existing one.
if [[ $(systemctl --user show "$SPARK_EXPERIMENT_UNIT" -p LoadState --value) != not-found ]]; then
  rmdir "$session"; echo 'Unexpected unit collision' >&2; exit 1
fi
source bench/spark-unit.sh
restore=0
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if ! spark_stop_owned_unit; then
    echo 'ROLLBACK BLOCKED: experimental GPU processes may remain; production NOT restarted.' >&2
    echo "Inspect: systemctl --user status $SPARK_EXPERIMENT_UNIT" >&2
    exit 1
  fi
  rmdir "$session"
  if ((restore)); then
    if ! sudo -n "$ctl" start sglang.service; then
      echo 'RESTORE FAILED: run sudo systemctl start sglang.service' >&2; exit 1
    fi
    echo 'SGLang restart requested; waiting up to 15 minutes for readiness.'
    local ready=0
    for ((i=0; i<180; i++)); do
      if curl -fsS --max-time 3 http://127.0.0.1:30000/health >/dev/null; then ready=1; break; fi
      sleep 5
    done
    if ((!ready)); then
      echo 'SGLang did not become healthy; inspect journalctl -u sglang.' >&2; exit 1
    fi
    echo 'SGLang restored and healthy.'
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if "$ctl" is-active --quiet sglang.service; then
  # Check BOTH permissions before interrupting the server.
  sudo -n -l "$ctl" start sglang.service >/dev/null
  sudo -n -l "$ctl" stop sglang.service >/dev/null
  restore=1
  sudo -n "$ctl" stop sglang.service
fi
"$@"
