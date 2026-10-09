#!/usr/bin/env python3
"""Capture production SGLang's launch; replay it privately while production is stopped."""
import argparse
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import urllib.request

MODEL = "RadixArk/Qwen3.8-27B-NVFP4"
REVISION = "52d1adc5f38aa5ebf099c29ed7025ba34cfbb854"
ENV_KEYS = {
    "CUDA_HOME", "LD_LIBRARY_PATH", "LIBRARY_PATH", "NVCC_PREPEND_FLAGS",
    "PATH", "TRITON_LIBCUDA_PATH", "LOCALE_ARCHIVE",
}


def replace_flag(argv, flag, value):
    result = list(argv)
    positions = [i for i, arg in enumerate(result) if arg == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(result):
        raise ValueError(f"Expected exactly one {flag} VALUE")
    result[positions[0] + 1] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["capture", "serve"])
    parser.add_argument("--snapshot", default="experiments/sglang-production.json")
    args = parser.parse_args()
    if socket.gethostname() != "dgx-spark":
        parser.error("Run on dgx-spark")
    path = Path(args.snapshot)
    if args.action == "capture":
        pid = int(subprocess.check_output(["systemctl", "show", "sglang.service", "-p", "MainPID", "--value"]))
        if pid <= 0:
            raise RuntimeError("SGLang must be running to capture its configuration")
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().rstrip("\0").split("\0")
        # The exact immutable interpreter/env is reused; no dependency upgrades.
        if not argv[0].startswith("/nix/store/") or "sglang" not in " ".join(argv):
            raise RuntimeError("Unexpected production launch")
        if argv[argv.index("--model-path") + 1] != MODEL or argv[argv.index("--revision") + 1] != REVISION:
            raise RuntimeError("Production checkpoint changed; re-evaluate the experiment")
        raw_env = subprocess.check_output(["systemctl", "show", "sglang.service", "-p", "Environment", "--value"], text=True)
        env = dict(item.split("=", 1) for item in shlex.split(raw_env))
        env = {key: value for key, value in env.items() if key in ENV_KEYS}
        with urllib.request.urlopen("http://127.0.0.1:30000/get_server_info", timeout=10) as response:
            info = json.load(response)
        snapshot = {"argv": argv, "environment": env, "version": info["version"],
                    "model": MODEL, "revision": REVISION}
        path.parent.mkdir(parents=True, exist_ok=True)
        # Capturing is explicit and never silently overwrites provenance.
        with path.open("x", encoding="utf-8") as output:
            os.chmod(path, 0o600)
            json.dump(snapshot, output, indent=2)
            output.write("\n")
        print(path)
        return
    if subprocess.run(["systemctl", "is-active", "--quiet", "sglang.service"]).returncode == 0:
        raise RuntimeError("Stop production via exclusive-spark.sh before replaying")
    snapshot = json.loads(path.read_text())
    argv = replace_flag(snapshot["argv"], "--host", "127.0.0.1")
    argv = replace_flag(argv, "--port", os.environ.get("PORT", "8000"))
    home = Path.home()
    cache = home / ".cache/sglang-vllm-comparison"
    env = {**os.environ, **snapshot["environment"],
           "HOME": str(home), "HF_HOME": str(home / ".cache/huggingface"),
           "XDG_CACHE_HOME": str(cache), "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
           "TRITON_CACHE_DIR": str(cache / "triton"), "HF_HUB_OFFLINE": "1"}
    print("Replaying production SGLang:", shlex.join(argv), flush=True)
    os.execve(argv[0], argv, env)


if __name__ == "__main__":
    main()
