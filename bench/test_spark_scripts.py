"""Offline launcher/ownership regression tests. Never starts a GPU server."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent


class CleanupTests(unittest.TestCase):
    def run_cleanup(self, load="loaded", owner="ours", fail=False, remains=False):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace"
            program = r'''
set -u
SPARK_EXPERIMENT_UNIT=unique.service
SPARK_EXPERIMENT_OWNER=ours
systemctl() {
  echo "$*" >> "$TRACE"
  case "$*" in
    *"-p LoadState --value") echo "$LOAD" ;;
    *"-p Description --value") echo "$OWNER" ;;
    *"-p ActiveState --value") echo "$STATE" ;;
    "--user stop unique.service")
      if [[ $FAIL == 1 ]]; then return 1; fi
      if [[ $REMAINS != 1 ]]; then LOAD=not-found; STATE=inactive; fi ;;
    "--user kill "*) [[ $FAIL != 1 ]] ;;
    "--user reset-failed "*) return 0 ;;
    *) return 99 ;;
  esac
}
timeout() { shift; "$@"; }
source "$HELPER"
spark_stop_owned_unit
'''
            result = subprocess.run(["bash", "-c", program], capture_output=True, text=True,
                env={**os.environ, "TRACE": str(trace), "HELPER": str(ROOT / "bench/spark-unit.sh"),
                     "LOAD": load, "OWNER": owner, "STATE": "active",
                     "FAIL": str(int(fail)), "REMAINS": str(int(remains))})
            return result, trace.read_text()

    def test_absent_unit_is_noop(self):
        result, trace = self.run_cleanup(load="not-found")
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("--user stop", trace)

    def test_foreign_unit_is_never_stopped(self):
        result, trace = self.run_cleanup(owner="someone else")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("--user stop", trace)
        self.assertNotIn("--user kill", trace)

    def test_owned_unit_stops(self):
        result, trace = self.run_cleanup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--user stop unique.service", trace)

    def test_failed_stop_escalates_and_reports_failure(self):
        result, trace = self.run_cleanup(fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--user kill --signal=SIGKILL --kill-whom=all unique.service", trace)

    def test_still_active_is_not_safe(self):
        result, _ = self.run_cleanup(remains=True)
        self.assertNotEqual(result.returncode, 0)


class LaunchTests(unittest.TestCase):
    def launch(self, profile, hostname="dgx-spark"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            host = root / "bin/hostname"
            host.write_text(f"#!/usr/bin/env bash\necho {hostname}\n")
            host.chmod(0o755)
            vllm = root / "bin/vllm"
            vllm.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
            vllm.chmod(0o755)
            return subprocess.run(["bash", str(ROOT / "bench/serve-spark.sh"), profile],
                capture_output=True, text=True,
                env={**os.environ, "PATH": str(root / "bin") + ":" + os.environ["PATH"],
                     "VLLM_ENV_DIR": str(root), "SPEC_TOKENS": "2", "KV_CACHE_DTYPE": "fp8"})

    def test_pinned_mtp_and_draft_attention(self):
        result = self.launch("mtp")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(args[1], "RadixArk/Qwen3.8-27B-NVFP4")
        revision = args[args.index("--revision") + 1]
        self.assertEqual(revision, "52d1adc5f38aa5ebf099c29ed7025ba34cfbb854")
        spec = json.loads(args[args.index("--speculative-config") + 1])
        self.assertEqual(spec["revision"], revision)
        self.assertEqual(spec["attention_backend"], "TRITON_ATTN")
        self.assertEqual(spec["kv_cache_dtype"], "bfloat16")
        self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")

    def test_pinned_dflash(self):
        result = self.launch("dflash")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout.splitlines()[-1])
        spec = json.loads(args[args.index("--speculative-config") + 1])
        self.assertEqual(spec["model"], "z-lab/Qwen3.8-27B-DFlash2")
        self.assertEqual(spec["revision"], "50307d4c4cde6860d4eee73e2547cd786fe8e8a4")
        self.assertNotIn("quantization", spec)

    def test_wrong_host_and_unknown_profile_rejected(self):
        self.assertEqual(self.launch("baseline", "rtx4090-tower").returncode, 2)
        self.assertEqual(self.launch("typo").returncode, 2)

    def test_sglang_flag_replacement_is_exact(self):
        spec = importlib.util.spec_from_file_location("sglang_baseline", ROOT / "bench/sglang-baseline.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original = ["sglang", "--host", "0.0.0.0", "--port", "30000"]
        changed = module.replace_flag(original, "--host", "127.0.0.1")
        self.assertEqual(changed[2], "127.0.0.1")
        self.assertEqual(original[2], "0.0.0.0")
        with self.assertRaises(ValueError):
            module.replace_flag(["--port", "1", "--port", "2"], "--port", "8000")
        with self.assertRaises(ValueError):
            module.replace_flag(["--port"], "--port", "8000")


if __name__ == "__main__":
    unittest.main()
