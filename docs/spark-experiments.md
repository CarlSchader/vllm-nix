# DGX Spark: controlled Qwen engine experiments

## Status (2026-10-09)

Preparation only so far. **No vLLM speedup has been measured.** Production
SGLang has not been stopped or reconfigured. The harness passed 19 offline
tests and a 32-token live SGLang smoke test; the latter is **not** a performance
baseline because other clients were active. See [HANDOFF.md](../HANDOFF.md)
for the continuation checklist and remote checkout/pull precautions.

The controlling checkout is on `rtx4090-tower`; the GPU host is reached with
`ssh dgx-spark`. Do not run Spark workloads on the tower's RTX 4090.

## Preserve the model, not just its name

Production uses:

- `RadixArk/Qwen3.8-27B-NVFP4`
- Revision `52d1adc5f38aa5ebf099c29ed7025ba34cfbb854`
- Architecture `Qwen3_5ForConditionalGeneration`
- **FP8 attention projections, NVFP4 MLP and LM head**, unquantized MTP weights.
- DFlash2 draft `z-lab/Qwen3.8-27B-DFlash2`, revision
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` (unquantized).
- API alias `qwen3.8-27b`, native context 262144.

Although production passes KV dtype `auto`, its startup journal confirms
**FP8 E4M3 target KV and BF16 draft KV**. On the inspected boot: target KV
11.714 GiB / 383852 tokens, consistent with FP8 storage. Mamba SSM state is
BF16. The experiment defaults explicitly match these precisions; BF16 target
KV can be tested separately. Do not force a uniform weight quantization mode.

The [linked discussion](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/34)
uses a different Unsloth checkpoint, FP8 KV and two-token MTP. Its current
Unsloth config is not lower precision in the inspected weight groups (last
8 MLP layers and LM head use FP8 rather than RadixArk's NVFP4), but different
export/calibration is a confounder. **No checkpoint substitution is made.**

## Installed environments on the Spark

- Existing `.venv`: vLLM 0.28.0, torch 2.13.0, FlashInfer 0.6.16.post3.
- Separate `.venv-v031`: vLLM 0.31.0, torch 2.13.0, FlashInfer 0.7.0.post1.
  Its dependency freeze is tracked as `bench/requirements-v031.txt` and also
  lives at `experiments/v031-requirements.txt` on the Spark. These are version
  pins, not wheel hashes. This is an experimental uv environment,
  **not an updated Nix/uv lock**.
- Production: SGLang 0.5.20, immutable Nix environment captured from systemd.
- Hardware: GB10, aarch64 NixOS, NVIDIA driver 595.104.02.

Both vLLM versions pass `vllm --version` inside `nix develop`. This is not a
model-loading test. `uv pip check` reports one platform-tag incompatibility in
both environments: `nvidia-cusparselt-cu13==0.8.1` declares
`py3-none-manylinux2014_sbsa`. The new environment's actual library is AArch64
(ELF machine 183) and loads with `ctypes.CDLL` after importing torch;
`torch.backends.cusparselt.is_available()` is true. This looks like vendor tag
metadata, but **do not claim clean dependency checks** or silently suppress it. The exact target and draft snapshots have been downloaded
to the user's HF cache; `/var/lib/sglang` remains private and untouched.

## Runbook

Copy `bench/` to the Spark checkout. All commands below run **on the Spark**.
Production clients lose service temporarily. Stop their workloads first.

The session needs permission only for these two commands:

```sh
sudo /run/current-system/sw/bin/systemctl stop sglang.service
sudo /run/current-system/sw/bin/systemctl start sglang.service
```

For unattended experiments, an administrator can grant a temporary narrow
sudo rule; no general passwordless sudo is needed:

```sh
echo 'carl ALL=(root) NOPASSWD: /run/current-system/sw/bin/systemctl stop sglang.service, /run/current-system/sw/bin/systemctl start sglang.service' | sudo tee /etc/sudoers.d/vllm-benchmark
sudo chmod 440 /etc/sudoers.d/vllm-benchmark
sudo visudo -cf /etc/sudoers.d/vllm-benchmark
```

Capture production before stopping it (already done for this session; refuses
to overwrite an existing capture):

```sh
cd ~/code/vllm-nix
python3 bench/sglang-baseline.py capture
```

The private SGLang baseline reuses its immutable interpreter, CUDA build
environment, and actual argv, changing only host/port and user-writable cache
paths. This keeps live clients on port 30000 from contaminating measurements.
Its compilation cache is initially cold, so allow a long first startup.

Example **single trial**, automatically restoring production afterward:

```sh
systemd-run --user --unit=spark-comparison --collect \
  --working-directory="$HOME/code/vllm-nix" \
  bash bench/exclusive-spark.sh \
  bash bench/trial-spark.sh sglang-isolated sglang
journalctl --user -u spark-comparison -f
```

Use unique labels (existing artifacts are never overwritten). Profiles:
`sglang`, `baseline` (no speculation), `mtp`, `dflash`. To amortize production
restarts, pass a shell script containing sequential trials to
`exclusive-spark.sh`. A failure should stop the sequence and restore SGLang.

```sh
# Examples for such a script, not established best settings:
VLLM_ENV_DIR=.venv-v031 bash bench/trial-spark.sh v031-base baseline
VLLM_ENV_DIR=.venv-v031 SPEC_TOKENS=2 bash bench/trial-spark.sh v031-mtp2 mtp
VLLM_ENV_DIR=.venv-v031 SPEC_TOKENS=7 bash bench/trial-spark.sh v031-dflash7 dflash
```

`serve-spark.sh` accepts additional vLLM CLI arguments. Environment knobs:
`VLLM_ENV_DIR`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `BATCHED_TOKENS`,
`GPU_MEMORY_UTILIZATION`, `ATTENTION_BACKEND`, `KV_CACHE_DTYPE`, `SPEC_TOKENS`,
`PORT`. Defaults: native 262144 context, 8 sequences, 4096 batched tokens,
0.50 memory utilization, Triton attention, FP8 target cache.

**MTP workaround candidate:** specify `attention_backend: TRITON_ATTN` inside
`--speculative-config`, not just on the target. Installed vLLM source explicitly
selects draft attention independently. This avoids the old recipe's accidental
FlashInfer draft selection; it has not yet been verified by model startup.
The draft's revision and BF16 KV dtype are explicit as well.

The user-unit limit is 100G with swap disabled. This is a guardrail, not a
complete bound on unified CUDA allocations. Do not raise memory utilization
aggressively: this hardware has previously frozen under transient allocations.

## Measurement plan

1. Isolated production-equivalent SGLang baseline.
2. vLLM 0.28 and 0.31 non-speculative baselines, then MTP and DFlash2.
3. Sweep speculative token count, then latency/throughput kernel policy,
   attention backend, graph sizes, batching; change one factor at a time.
4. Re-test finalists with more samples, concurrency 1/4/8, thinking on/off,
   longer prompts, and repeated-prefix workloads. Verify parsers and output
   correctness separately. A small synthetic test is not a quality evaluation.
5. Restore SGLang; only recommend a persistent deployment after measurements.

The default trial runs four requests of 512 tokens at concurrency 1 and 4,
with a separate warmup. Different fixed seeds per concurrency prevent the
second batch from inheriting the first batch's exact cached prefixes. The same
concurrency uses the same requests across engines. Re-running a benchmark on
the same live server/seed is **warm-cache**, not cold-cache.

`bench/benchmark.py` writes exact request payloads/hashes, raw SSE, text,
server-reported usage, TTFT, request latency, approximate streaming decode
rate, and aggregate output tokens per wall-second. It never counts SSE events
as tokens. Fixed-length tests use `ignore_eos=true`; they can continue after a
natural answer and therefore measure throughput, not answer quality. Use
`--allow-eos` for natural termination, `--thinking` for reasoning mode.
`--prompt-repeat` creates longer synthetic inputs, not a target token count.

The decode rate is explicitly approximate: all completion tokens divided by
time from first to last nonempty delta. First-chunk/speculative token batching
can bias that value, especially with short outputs. Use aggregate/e2e rates
alongside it. Failures, missing usage, truncated SSE, and short fixed-length
outputs yield nonzero exit status and remain recorded in the JSON.

Artifacts live under `experiments/LABEL/` on the Spark: launch metadata,
package versions (or captured SGLang provenance), model API response, JSON
measurements, full server journal, and exit status. They are ignored by Git. Do not publish a winner from failed
runs or compare them to the old SGLang README numbers (different prompts and
thinking settings).

## Rollback and cleanup

`exclusive-spark.sh` locks against parallel sessions, allocates a unique
owned server unit, and restores SGLang on normal exit/error/INT/TERM, including
a health check. Cleanup refuses foreign units and blocks production restart
if experimental shutdown cannot be confirmed. These paths have mocked offline
tests, **not a live stop/start validation yet**. Running the wrapper as a user
unit protects against SSH disconnects. SIGKILL, host crashes, or lost sudo permissions
cannot be recovered by a shell trap; manual recovery:

```sh
systemctl --user list-units --all 'vllm-spark-experiment.*.service'
# Stop the exact experiment unit shown above; confirm its GPU processes exit.
systemctl --user stop vllm-spark-experiment.EXACT_SESSION_ID.service
sudo /run/current-system/sw/bin/systemctl start sglang.service
curl -f http://127.0.0.1:30000/health
sudo rm /etc/sudoers.d/vllm-benchmark  # after all experiments finish
```

Production NixOS configuration, clients, Open WebUI, model weights and service
cache are not changed by these scripts.

Offline tests:

```sh
python3 -B -m unittest discover -s bench -p 'test_*.py'
for script in bench/*.sh; do bash -n "$script" || exit; done
```
