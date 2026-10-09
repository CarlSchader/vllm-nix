#!/usr/bin/env bash
# Shared cleanup for the unique unit owned by an exclusive-spark.sh session.
# Return failure rather than starting production beside an unconfirmed shutdown.
spark_stop_owned_unit() {
  local load owner state
  load=$(systemctl --user show "$SPARK_EXPERIMENT_UNIT" -p LoadState --value) || return 1
  [[ $load == not-found ]] && return 0
  owner=$(systemctl --user show "$SPARK_EXPERIMENT_UNIT" -p Description --value) || return 1
  if [[ $owner != "$SPARK_EXPERIMENT_OWNER" ]]; then
    echo "Refusing to stop non-owned unit $SPARK_EXPERIMENT_UNIT" >&2
    return 1
  fi
  # TimeoutStopSec=60 and KillMode=control-group bound normal shutdown. If
  # systemctl fails/times out, escalate only against this uniquely owned unit.
  if ! timeout 90 systemctl --user stop "$SPARK_EXPERIMENT_UNIT"; then
    systemctl --user kill --signal=SIGKILL --kill-whom=all "$SPARK_EXPERIMENT_UNIT" || return 1
    timeout 30 systemctl --user stop "$SPARK_EXPERIMENT_UNIT" || return 1
  fi
  load=$(systemctl --user show "$SPARK_EXPERIMENT_UNIT" -p LoadState --value) || return 1
  [[ $load == not-found ]] && return 0
  state=$(systemctl --user show "$SPARK_EXPERIMENT_UNIT" -p ActiveState --value) || return 1
  case "$state" in
    inactive|failed) systemctl --user reset-failed "$SPARK_EXPERIMENT_UNIT" 2>/dev/null || true ;;
    *) echo "Experimental unit remains $state; NOT safe to restart SGLang" >&2; return 1 ;;
  esac
}
