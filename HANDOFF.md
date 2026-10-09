# Continue Qwen/vLLM tuning on dgx-spark

**Written 2026-10-09. This is a handoff, NOT a completed optimization report.**

## Request and authorizations

The user wants the **same Qwen model currently served by SGLang** to perform
as well as possible with vLLM on the DGX Spark. They explicitly approved:

- Temporary SGLang downtime for isolated tests, with restoration afterward.
- FP8 KV-cache experiments, reported separately if precision differs.

They require asking **before lowering weight quantization**. Preserve the
exact production checkpoint by default. Do not substitute a different
"NVFP4" model just because its name looks equivalent.

The latest instruction was to write down everything done/planned so the user
can push this checkout and start a **new agent directly on the Spark**. No
production migration, commits, pushes, or NixOS rebuilds have been performed.

## Start here — important checkout caveat

This work was done in `/home/carl/code/vllm-nix` on **rtx4090-tower**, not on
the Spark. Remote work used `ssh dgx-spark` as `carl`.

The Spark already has `/home/carl/code/vllm-nix`, and an **older partial copy
of `bench/` is untracked there**, copied over for the smoke test. It does NOT
contain all final review fixes. A pull adding those tracked files may refuse
to overwrite the untracked copies.

On the Spark, inspect first, preserve those files outside the repo, then pull
the user's pushed revision:

```sh
cd ~/code/vllm-nix
hostname                         # must be dgx-spark
uname -m                         # must be aarch64
git status --short
# Only if bench/ is still the untracked copy described above:
mv bench "$HOME/vllm-bench-prepull-$(date +%Y%m%d-%H%M%S)"
git pull --ff-only
python3 -B -m unittest discover -s bench -p 'test_*.py'
for script in bench/*.sh; do bash -n "$script" || exit; done
```

Do not overwrite any new user edits. **Keep** `experiments/`, `.venv`,
`.venv-v031`, and the HF cache: they are useful remote-only state. The new
`.gitignore` ignores experiment artifacts, virtualenvs, and Python bytecode.

Then read [the runbook](docs/spark-experiments.md) and inspect the scripts.
The original root `Qwen3.8-27B-dgx-spark.sh` is unchanged and uses a different
Inferact checkpoint; **do not use it for this exact-checkpoint comparison**.

## Verified production state

Hardware/OS:

- NVIDIA GB10, unified memory, approximately 121 GiB usable host RAM.
- aarch64 NixOS, kernel 6.17.13, NVIDIA driver 595.104.02 (CUDA driver API 13.2).
- Production SGLang is a native NixOS systemd service, not Docker.
- `sglang.service` runs as `sglang`, port **30000**, alias **qwen3.8-27b**.
- Final check: service active and `/health` returned 200. One preceding
  five-second health probe timed out while the service was busy; a 30-second
  retry succeeded. Other clients were active throughout preparation.

Exact target:

```text
RadixArk/Qwen3.8-27B-NVFP4
52d1adc5f38aa5ebf099c29ed7025ba34cfbb854
```

This is **mixed precision**, not uniformly four-bit:

- FP8 attention projections.
- NVFP4 MLP projections and LM head, group size 16.
- Unquantized MTP weights are included in the same checkpoint.
- Architecture: `Qwen3_5ForConditionalGeneration` / `qwen3_5`.
- Quantization metadata: `modelopt`, `MIXED_PRECISION`; let vLLM detect it.
- Native context: 262144.

Exact unquantized speculative draft:

```text
z-lab/Qwen3.8-27B-DFlash2
50307d4c4cde6860d4eee73e2547cd786fe8e8a4
```

**Important correction made during investigation:** SGLang passes KV dtype
`auto`, but it already resolves to FP8 E4M3 target KV, NOT BF16. Startup logs
at 2026-10-07 21:30:18 show target `torch.float8_e4m3fn` (383852 tokens,
K/V 5.86 GiB each) and draft `torch.bfloat16` (K/V 3.66 GiB each). Mamba state
is explicitly BF16. The experiment defaults match these precisions.

Production launch:

```text
sglang serve --model-path RadixArk/Qwen3.8-27B-NVFP4
 --revision 52d1adc5f38aa5ebf099c29ed7025ba34cfbb854
 --served-model-name qwen3.8-27b --host 0.0.0.0 --port 30000
 --context-length 262144 --mem-fraction-static 0.50 --tp-size 1
 --speculative-algorithm DFLASH
 --speculative-draft-model-path z-lab/Qwen3.8-27B-DFlash2
 --speculative-draft-model-revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4
 --speculative-num-draft-tokens 8 --speculative-draft-model-quantization unquant
 --enable-torch-compile --torch-compile-max-bs 4 --max-running-requests 8
 --chunked-prefill-size 4096 --tokenizer-worker-num 4 --detokenizer-worker-num 2
 --attention-backend flashinfer --tool-call-parser qwen3_coder --reasoning-parser qwen3
 --trust-remote-code --enable-mixed-chunk --disable-prefill-cuda-graph
 --cuda-graph-max-bs-decode 8 --disable-flashinfer-autotune
 --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype bfloat16
 --max-mamba-cache-size 96 --num-continuous-decode-steps 2 --sleep-on-idle
```

The service has `MemoryMax=100G`, `Restart=always`, persistent state under
`/var/lib/sglang` (private to the service). Never kill its scheduler directly:
systemd would restart it and invalidate experiments.

## Repository findings

Read-only investigation of the sibling repositories found:

- `~/code/mono/system/nixos-configurations/dgx-spark/default.nix` imports
  SGLang and its Spark preset; it also wires the separate Open WebUI service.
- `~/code/sglang-nix/modules/dgx-spark-qwen38.nix` is the authoritative preset.
  Its root shell launcher is stale (8192 prefill, obsolete graph argument,
  missing the newer worker/mixed-chunk settings). Use captured live argv.
- `~/code/mono/pi/pi-models-options.nix` clients use
  `http://dgx-spark:30000/v1`; Open WebUI uses local port 30000. Preserve the
  alias, tool/reasoning support and advertised 262144 context if migrating.
- `~/code/sglang-nix/docs/usage.md` reports code 31–35 tok/s, math 55–61,
  eight-stream aggregate 155. These are historical, different-workload
  measurements, **not our baseline**. Its `bench.sh` is sequential, thinking
  enabled, max 800 output tokens; no raw concurrent results were found.
- Its TODO mentions poor long-prefix cache reuse and long-prefill stalls.
  These matter more for agent usability than a cherry-picked decode number.
- Current `vllm-nix/nix/vllm-env.nix` exposes packaged environments only for
  `x86_64-linux`. The aarch64 dev shell works; the existing Spark `.venv`
  works for CLI/imports. An eventual NixOS deployment needs ARM packaging
  work and checks, not just replacing the host's SGLang import.

No files in `mono` or `sglang-nix` were edited.

## What was installed/created on the Spark

Pre-existing `.venv` was left unchanged:

```text
vLLM 0.28.0 / torch 2.13.0 / FlashInfer 0.6.16.post3
```

Created separate `.venv-v031`, with `uv venv --python .venv/bin/python` and
`uv pip install --python .venv-v031/bin/python 'vllm==0.31.0'`:

```text
vLLM 0.31.0 / torch 2.13.0+cu130 / FlashInfer 0.7.0.post1
transformers 5.17.0 / Python 3.12.14
```

Full version freeze: **`bench/requirements-v031.txt`** (199 packages), copied
from remote `experiments/v031-requirements.txt`. It is not a hash-locked uv/Nix
update. To reconstruct if needed, use a separate venv and `uv pip sync` with
that file; do not upgrade the production SGLang environment.

Validation:

- Both `nix develop --command <venv>/bin/vllm --version` succeed.
- New env imports torch; cuSPARSELt reports available and its `.so` loads.
- **Both environments fail `uv pip check` on one vendor platform tag**:
  `nvidia-cusparselt-cu13==0.8.1` has `py3-none-manylinux2014_sbsa`.
  Actual ELF machine is 183 (AArch64), not x86_64. This appears to be metadata,
  but packaging is not cleanly validated; assess it before final Nix work.
- No vLLM model startup, GPU generation, MTP, DFlash2, or newer-version speed
  has been validated. CLI/import success is not enough.

Downloaded exact target (~21 GiB) and draft (~3.6 GiB) under:

```text
~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/
~/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/
```

Download/setup jobs used transient user units `vllm-model-download`,
`vllm-draft-download`, `vllm-v031-prepare`; all finished. Final inspection
showed **no loaded `vllm-*` user units**. No experiment server is running.

Remote artifacts already present:

| Path under `~/code/vllm-nix/` | Contents |
|---|---|
| `experiments/sglang-production.json` | Actual immutable production interpreter/argv, selected CUDA environment and SGLang version; captured before any downtime. Private permissions. |
| `experiments/sglang-harness-smoke-live.json` | Two synthetic 32-token requests (warmup + one measurement), raw SSE/usage. Validates client only; NOT isolated throughput evidence. |
| `experiments/v031-requirements.txt` | New environment version freeze. |
| `experiments/vllm-help.txt` | vLLM 0.28 complete CLI help. |

## New files in this change

- `bench/benchmark.py`: stdlib streaming OpenAI benchmark, fixed-length or
  natural EOS, thinking toggle, bounded concurrency, payload hashes, raw SSE,
  usage, TTFT/latency/aggregate throughput, explicitly approximate decode rate.
- `bench/serve-spark.sh`: pinned RadixArk `baseline`, `mtp`, `dflash` launch
  profiles, loopback only, conservative memory, same KV/state precision.
- `bench/sglang-baseline.py`: captures production or replays its exact runtime
  privately, changing host/port and cache/home locations only.
- `bench/exclusive-spark.sh`: exclusive session lock, narrow sudo preflight,
  production stop/restore, unique owned experiment unit and health wait.
- `bench/trial-spark.sh`: start one isolated profile, readiness/identity checks,
  concurrency 1/4 benchmark batches, logs/artifacts, fail-fast cleanup.
- `bench/spark-unit.sh`: shared owned-unit shutdown; blocks production restart
  if experimental shutdown cannot be verified.
- `bench/test_benchmark.py`, `bench/test_spark_scripts.py`: **19 offline tests**,
  including terminal SSE validation, mocked cleanup/ownership and launch args.
- `bench/requirements-v031.txt`: frozen experimental dependencies.
- `docs/spark-experiments.md`: detailed runbook/measurement limitations.
- `.gitignore`, `README.md`: artifact ignores and discovery links.

Offline tests and shell syntax checks passed. Previously captured real SGLang
SSE was replayed offline through the stricter terminal/usage validation and
passed. Read-only review identified safety/fairness issues, which were fixed:
per-concurrency seeds prevent accidental warm-cache comparison; unique unit
ownership prevents unrelated shutdown; failed shutdown blocks restore;
occupied ports are rejected; failed startup logs retained; missing terminal
finish and excess/short token counts rejected.

**Remaining verification:** these orchestration paths are still not tested
against a real stopped/started systemd GPU server. Test a single small trial
first, including rollback. Mocked tests are not a substitute for that.

## Why the linked recipe matters

Source: https://huggingface.co/Qwen/Qwen3.8-27B/discussions/34

The readable web fetch omitted the actual discussion; its JSON API worked:
`https://huggingface.co/api/models/Qwen/Qwen3.8-27B/discussions/34`.
It uses Unsloth NVFP4, FP8 KV, 0.50 memory, max sequences 4, native 262144
context, two-token MTP, and a pinned vLLM image.

The currently inspected Unsloth config is not lower precision in the compared
weight groups: its last eight MLP layers and LM head are FP8 rather than
RadixArk's NVFP4. However export/scaling/calibration differ; **we did not
switch or download it**. Keeping RadixArk makes the engine comparison fair.

The existing root vLLM recipe says MTP failed with a FlashInfer head_dim=256
undefined symbol on this hardware. Installed vLLM source explains a likely
workaround: **draft attention does not inherit target attention**.
`bench/serve-spark.sh` explicitly sets `attention_backend: TRITON_ATTN` in
speculative config as well as the target, plus draft revision and BF16 KV.
This is source-justified but still an **untested workaround**, not a fix claim.

vLLM 0.28 already registers DFlash2 and automatically selects its newer model
runner; method is `dflash`, not `dflash2`. Installed modelopt mixed loader
recognizes the RadixArk per-layer formats. v0.31 adds newer FlashInfer and
DFlash2 grouped-convolution/Qwen GDN fixes, making it worth testing separately.
Kernel support in source does not prove GB10 execution works.

## Blocking access issue

SSH `sudo -n` required a password; neither stop nor start permission was
granted in this session. An attempted noninteractive root SSH connection timed
out and was not used. No privilege escalation or system mutation occurred.

The user was shown a narrow temporary sudoers rule for **only**:

```text
/run/current-system/sw/bin/systemctl stop sglang.service
/run/current-system/sw/bin/systemctl start sglang.service
```

It has **not** been installed by us. The runbook includes the administrator
commands. Running the next agent locally may make authentication easier, but
does not itself grant permission. Ensure automatic restart permission works
before stopping production. Never put a password in files or chat.

## Next-agent execution plan

1. **Pull the final files safely**, run tests, verify host/GPU, live model
   revision, service status, downloaded snapshots and available RAM. Re-capture
   SGLang if production changed, preserving the old capture. Read any applicable
   repository instructions before changes to siblings.
2. **Establish service-control access** and ask the user to pause other clients.
   Do not benchmark SGLang on the busy production port. Capture is already done.
3. **Run one private SGLang baseline trial** using the captured immutable runtime
   on loopback port 8000, through the exclusive wrapper. First compile may take
   9+ minutes in the new user cache. Verify cleanup and production restore.
4. **Validate vLLM startup and semantics before sweeping**. Start conservative
   non-speculative native-context FP8-cache baseline. If diagnosing failures,
   use shorter context/eager mode and label those runs diagnostic, not equivalent
   deployment candidates. Inspect resolved quantization, target/draft dtype,
   chosen attention/GEMM kernels and any fallback. Do not force uniform NVFP4.
5. **Compare v0.28 vs v0.31, then MTP vs DFlash2** with identical inputs/settings.
   Start MTP k=1/2/3; DFlash2 k=7 corresponds to an 8-token block including the
   bonus token (verify exact implementation semantics; don't equate token-count
   knobs across engines blindly). Check generated text, reasoning, tools,
   structured JSON and errors before accepting throughput.
6. **Tune promising configurations**, changing one factor at a time:
   - Speculation token count, acceptance and verification cost, especially c=4/8.
   - `--performance-mode interactivity` vs balanced/throughput (if supported).
   - Graph capture sizes, compile policy, Triton vs working FlashInfer attention;
     NVFP4 GEMM backend is separate from attention. Identify actual kernels.
   - Batched tokens 2048/4096/8192 and max sequences 4/8; measure TTFT and
     interference from long prefills, not just decode throughput.
   - BF16 vs existing FP8 target KV as a separately labelled precision/capacity
     comparison, keeping draft BF16. Never silently lower weights/state.
   - Hybrid/Mamba prefix caching and repeated long context, subject to actual
     model-runner support. Avoid disabling caching just to win tiny benchmarks.
   Keep memory utilization at 0.50 initially. Prior SGLang work documents
   25–40 GB transient unified-memory allocations and hard freezes above safe
   limits; do not casually raise it above 0.70. No overclock/power changes made.
7. **Confirm finalists**, rather than declaring a winner from four requests:
   multiple repeats, same prompt sets, concurrency 1/4/8, 512+ output tokens,
   thinking on/off, long prompts (e.g. 8K/32K/128K/200K actual tokenizer counts),
   cold/repeated prefixes, mixed prefill/decode traffic. Synthetic repeated
   instructions are not a representative long coding context. Record temperature,
   GPU load/clocks, memory, startup time and server/client version provenance.
   Compare end-to-end rates and TTFT alongside approximate decode rates.
8. **Restore SGLang and record honest results**. If vLLM loses, report that.
   Preserve failed runs, OOMs, fallback information and untested areas. Do not
   compare to historical README numbers as though the workloads were identical.
9. **Only after a measured winner**, make deployment reproducible:
   - Pin winning vLLM/dependencies in pyproject/uv.lock, add aarch64 environment
     support and necessary narrowly justified wheel fixes in `nix/vllm-env.nix`.
   - Build/import/module-evaluate natively; resolve/document the vendor SBSA tag.
   - Add a reviewed Spark preset with pinned target/draft revisions, parsers,
     context, cache precision and measured tuning; include module-eval coverage.
   - If user elects to migrate, update `mono` host service/UI/client wiring
     deliberately, keeping API alias/port contract or updating all consumers.
     Keep SGLang rollback available. No such migration is authorized by a
     benchmark result alone without coordinating downtime and user preference.

Example first session from the runbook:

```sh
systemd-run --user --unit=spark-comparison --collect \
  --working-directory="$HOME/code/vllm-nix" \
  bash bench/exclusive-spark.sh \
  bash bench/trial-spark.sh sglang-isolated sglang
journalctl --user -u spark-comparison -f
```

The wrapper is designed to restore SGLang on error/normal exit/INT/TERM, but
not SIGKILL, a machine crash or lost permissions. Manual recovery: identify and
stop the exact `vllm-spark-experiment.*.service`, verify no experiment GPU
processes remain, then
`sudo /run/current-system/sw/bin/systemctl start sglang.service` and wait for health.

**Bottom line: groundwork and instrumentation are ready; actual vLLM loading,
benchmarking, tuning, Nix packaging, and any production switch remain to do.**
