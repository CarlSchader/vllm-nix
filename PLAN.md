# Plan: `vllm-nix` as a NixOS Module + systemd Service

Goal: package the current vLLM server setup (vllm 0.28.0, Qwen3.8-27B-INT4 presets,
open-webui front-end) into a **NixOS module** that installs a working, auto-restarting
systemd service stack on **any NixOS machine with a CUDA-capable NVIDIA GPU** (managed
by nixpkgs, e.g. `hardware.nvidia.packages = [ pkgs.nvidia-open ]`).

Consumers would do:

```nix
# in their NixOS configuration (flake input `vllm-nix`)
{ imports = [ vllm-nix.nixosModules.vllm ];

  services.vllm = {
    enable = true;
    package = vllm-nix.packages.${pkgs.system}.vllmEnv;
    model = { hfId = "RedHatAI/Qwen3.8-27B-INT4"; servedModelName = "Qwen3.8-27B-INT4"; maxModelLen = 32768; };
    ui.enable = true;   # configures nixpkgs' services.open-webui
  };
}
```

---

## 1. Current state (what we are packaging)

| Piece | Today | Notes |
| --- | --- | --- |
| Python env | `uv` project, `uv.lock` pins **vllm 0.28.0**, torch 2.13.0, flashinfer-python 0.6.16.post3, triton 3.7.1 | Prebuilt PyPI wheels; python 3.12; lock also carries **154 sdist entries** (most have wheel alternatives) |
| Launch | `Qwen3.8-27B-INT4.sh` / `Qwen3.8-27B-INT4-FP8Cache.sh` → `uv run vllm serve … --host 0.0.0.0` (port 8000) | Two presets = same HF model, different flags: 32k ctx vs 65k ctx + `--kv-cache-dtype fp8 --attention-backend TRITON_ATTN`. Both also pass `--limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0 --mm-encoder-tp-mode data` |
| UI | `run-ui.sh` → `uvx --with ddgs open-webui serve` (port 8080) with web-search env vars + `.webui_secret_key` file | **Superseded**: nixpkgs ships `pkgs.open-webui` (0.11.0 on our channel) **and** a `services.open-webui` NixOS module with stateDir/env/secret handling |
| Dev shell | `nix/shells.nix`: `cudaPackages_13.cudatoolkit`, `LD_LIBRARY_PATH = stdenv.cc.cc.lib + zlib + ${driverLink}/lib`, `CUDA_HOME`, `TRITON_LIBCUDA_PATH` | Confirms runtime needs **more than libcuda**: manylinux wheels also want `libstdc++`/`zlib` from the system |
| Flakes | `flake.nix` melds `nix/{default,models,shells}.nix`; nixpkgs-unstable, nix-community cachix | No `packages` or `nixosModules` yet |
| Target hw | RTX 4090 24 GB, nixpkgs-built open driver 595.91.07 | `/run/opengl-driver/lib` exposes `libcuda.so.1` (verified on this machine) |

Key facts verified against the locked nixpkgs rev:

- `python312Packages.vllm` exists at **0.24.0** (heavy source build) — two versions behind
  our lock; possibly cache-substitutable via nix-community, but still not the setup we
  validated. Not used.
- `pkgs.uv` = **0.12.5** — reads our `uv.lock` (revision 3) fine.
- `addDriverRunpath.driverLink` = `/run/opengl-driver` — the portable, NixOS-generic
  way to reach the active driver's `libcuda.so.1`.
- `pkgs.open-webui` **exists** (0.11.0) with a full NixOS module
  (`nixos/modules/services/misc/open-webui.nix`) — we reuse it instead of shipping our own.
- A plain `stdenv.mkDerivation` running `uv sync` **cannot reach the network** in the Nix
  sandbox ⇒ the env must be built with uv2nix (per-wheel fixed-output derivations using
  the hashes already in `uv.lock`).

---

## 2. Goals / non-goals

**Goals**

- `nixosModules.vllm`: declarative, typed, documented module (`services.vllm.*`),
  including a thin `services.vllm.ui.*` layer over nixpkgs' `services.open-webui`.
- `packages.<sys>.vllmEnv`: a reproducible python env built **from the pinned wheels in
  `uv.lock`** via **uv2nix** (pure eval, no whole-venv FOD, ELF patching handled).
- systemd: `vllm.service` (+ `vllm-models.service` one-shot downloader), dedicated
  `vllm` system user, auto-restart, persistent HF cache via `StateDirectory`.
- `openFirewall` option (default false).
- Works on any x86_64-linux NixOS box with a nixpkgs-managed NVIDIA driver (no GPU in
  CI/QEMU ⇒ test via import-check + manual deploy, see §8).
- `checks.*` so CI can build the env and eval the module.

**Non-goals (v1)**

- Multi-GPU / multi-node serving, ROCm/AMD, CPU fallback.
- NixOS config for this repo itself.
- Upstreaming to nixpkgs (later option, §11).
- TLS/reverse-proxy/auth in front of the API.
- Pinning open-webui's exact version (it now tracks the consumer's nixpkgs).

---

## 3. Key design decisions

### D1 — Python env: **uv2nix** ✅

| Option | Verdict | Why |
| --- | --- | --- |
| nixpkgs `python312Packages.vllm` 0.24 | Rejected | 2 versions behind validated setup; heavy source build (cachix may help, but behaviour drift is the real risk) |
| Plain `stdenv.mkDerivation` + `uv sync` | **Impossible** | No network in the Nix sandbox |
| Whole-venv fixed-output derivation | Rejected | Multi-GB output is not byte-reproducible (`.pyc`, timestamps, uv drift) → hash churn |
| Runtime `uv sync` in a oneshot unit | Rejected | Impure, env not a store path, network at service start |
| **uv2nix (pyproject.nix ecosystem)** | **Chosen** | Parses `uv.lock`; each wheel is its own FOD using the lock's existing hashes; venv composed purely; pyproject.nix build infra handles autoPatchelf for manylinux wheels |

Implementation shape (`nix/vllm-env.nix`):

- New flake inputs: `pyproject-nix`, `uv2nix`, `pyproject-build-systems` (standard trio).
- `workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; }` →
  `overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; }`
  (**wheel preference** enforces the "no source builds" promise; any dep with no wheel
  fails loudly at build → fix by pinning a wheel-capable version).
- `pythonSet = (pkgs.callPackage pyproject-nix.build.packages { python = pkgs.python312; })
  .overrideScope (lib.composeManyExtensions [ pyproject-build-systems.overlays.default overlay fixups ])`.
- `vllmEnv = pythonSet.mkVirtualEnv "vllm-env" workspace.deps.default` →
  `$out/bin/vllm`, `$out/bin/hf`, `$out/bin/python`.
- `fixups` overlay: expected small — torch/nvidia-* wheels may need
  `autoPatchelfIgnoreMissingDeps` for `libcuda.so.1` (driver-provided at runtime) and
  added lib paths for `libstdc++`/`zlib` (mirroring what `nix/shells.nix` already does).
  Budget real time here; this is where all the unknowns live (M1).
- **No changes to `pyproject.toml`/`uv.lock` needed** (open-webui stays out of the lock).

### D2 — Module API: `services.vllm`

Typed options for the flags the presets actually use + `extraArgs` escape hatch:

| Option | Type | Default | Maps to |
| --- | --- | --- | --- |
| `enable` | bool | false | gates everything |
| `package` | package | — (required) | env derivation (`…/bin/vllm`) |
| `listenHost` | str | `"0.0.0.0"` | `--host` |
| `listenPort` | port | 8000 | `--port` |
| `openFirewall` | bool | false | `networking.firewall.allowedTCPPorts` (vllm port; ui port when ui enabled) |
| `driverLibPath` | str | `"/run/opengl-driver/lib"` | part of `LD_LIBRARY_PATH` |
| `model.hfId` | str | `"RedHatAI/Qwen3.8-27B-INT4"` | `vllm serve <id>` |
| `model.servedModelName` | str | same as hfId | `--served-model-name` |
| `model.maxModelLen` | int | 32768 | `--max-model-len` |
| `maxNumSeqs` | int | 8 | `--max-num-seqs` |
| `maxNumBatchedTokens` | int | 2048 | `--max-num-batched-tokens` |
| `gpuMemoryUtilization` | float | 0.95 | `--gpu-memory-utilization` |
| `tensorParallelSize` | int | 1 | `--tensor-parallel-size` |
| `kvCacheDtype` | str | `"auto"` | `--kv-cache-dtype` (fp8 for FP8Cache preset) |
| `attentionBackend` | nullOr str | null | `--attention-backend` (TRITON_ATTN for FP8Cache) |
| `enableAutoToolChoice` | bool | true | `--enable-auto-tool-choice` |
| `toolCallParser` | nullOr str | `"qwen3_coder"` | `--tool-call-parser` |
| `reasoningParser` | nullOr str | `"qwen3"` | `--reasoning-parser` |
| `disableMultimodal` | bool | true | `--limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0` (matches current scripts) |
| `mmEncoderTpMode` | nullOr str | `"data"` | `--mm-encoder-tp-mode` (matches current scripts) |
| `extraArgs` | listOf str | [] | appended verbatim |
| `environment` | attrsOf str | {} | merged into unit `environment` |
| `timeoutStartSec` | str | `"20m"` | unit `TimeoutStartSec` (model load) |
| `ui.enable` | bool | false | enables + configures `services.open-webui` |
| `ui.port` | port | 8080 | `services.open-webui.port` |
| `ui.webSearch.enable` | bool | true | `ENABLE_WEB_SEARCH` etc. (defaults from `run-ui.sh`) |
| `ui.extraEnvironment` | attrsOf str | {} | merged into `services.open-webui.environment` |

Defaults reproduce the `Qwen3.8-27B-INT4` (32k) preset exactly, including the
multimodal-off flags the scripts pass. The FP8Cache preset = three option overrides
(`maxModelLen`, `kvCacheDtype`, `attentionBackend`), documented as a copy-paste block.
`nix/models.nix` stays as data; a preset-name helper is v2.

State/user options intentionally **removed** vs. earlier draft: state lives in
`StateDirectory=vllm` (`/var/lib/vllm`), user is a module-created `vllm` system user
(D5) — no `dataDir`/`user` knobs in v1 (fewer footguns; add later if requested).

### D3 — Service topology

```
network-online.target
        │
vllm-models.service (oneshot, RemainAfterExit=yes)
  └─ ${vllmEnv}/bin/hf download <hfId>   (HF_HOME=/var/lib/vllm/hf)
        │ After= / Wants=
vllm.service (Type=exec, Restart=on-failure)

open-webui.service   ← nixpkgs module; After=vllm.service (ordering only)
```

- **`vllm-models.service`** — first-boot model fetch (≈16 GB) in its own unit: own long
  timeout (`TimeoutStartSec=12h`), visible progress in journal, idempotent no-op re-runs.
- **`vllm.service`** — the server (§6.3).
- **`open-webui.service`** — produced by **nixpkgs' `services.open-webui`**. Our module
  just sets: `enable`, `port`, `environment` (`OPENAI_API_BASE_URL` →
  `http://127.0.0.1:<vllm port>/v1`, web-search vars from `run-ui.sh`), and adds
  `After=vllm.service` ordering. No health-wait `ExecStartPre` — open-webui is an
  OpenAI-API client and tolerates the backend being down; no secret-key shell hackery —
  the nixpkgs module already manages `WEBUI_SECRET_KEY` and its state dir.

### D4 — Driver / CUDA wiring

- System-side runtime needs: the driver's `libcuda.so.1` **plus** `libstdc++`/`zlib`
  for manylinux wheels (exactly what `nix/shells.nix` already provides in the dev shell).
- Unit sets `LD_LIBRARY_PATH = lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]
  - ":" + cfg.driverLibPath` — unless uv2nix's autoPatchelf makes the first part
  unnecessary (verify in M1; keep whichever is needed).
- `CUDA_HOME` / `TRITON_LIBCUDA_PATH` were **build-time** dev-shell needs (nvcc for
  flashinfer JIT); not set by default. Documented: add via `services.vllm.environment`
  if triton JIT errors appear at serve time.
- Sanity gate in `ExecStartPre`: `test -e ${driverLibPath}/libcuda.so.1` with a clear
  failure message (via `${pkgs.runtimeShell}`), so a missing driver fails fast instead
  of a cryptic vllm traceback.

### D5 — State, user, security

- **User**: module creates a dedicated **`vllm` system user/group**
  (`users.users.vllm = { isSystemUser = true; group = "vllm"; }`) with
  `SupplementaryGroups = [ "video" "render" ]` for `/dev/nvidia*` access.
- **State**: `StateDirectory = "vllm"` → systemd creates/chowns `/var/lib/vllm`
  (`HF_HOME=/var/lib/vllm/hf`, `WorkingDirectory=/var/lib/vllm`). No tmpfiles rules
  needed. open-webui state is handled by its nixpkgs module.
- **Hardening** (safe with GPU access): `NoNewPrivileges=true`, `ProtectHome=true`,
  `ProtectSystem="full"`, `PrivateTmp=true`. Deliberately **not** set: `PrivateDevices`,
  `RestrictAddressFamilies`, heavy `ProtectKernel*` — GPU device nodes and CUDA IPC need
  the default device namespace. `DynamicUser` not used (GPU groups + big persistent cache).
- **Restart policy**: `Restart=on-failure`, `RestartSec=10`; in `unitConfig`:
  `StartLimitIntervalSec=120`, `StartLimitBurst=3` (no hot-loop on OOM).
  `TimeoutStartSec` generous (model load + first triton compile).
- **Firewall**: `openFirewall` (default false) opens `listenPort` (+ `ui.port` when
  `ui.enable`).

### D6 — Version pinning & updates

- vllm env is reproducible via `uv.lock` (hashes consumed directly by uv2nix). Updating
  vllm = `uv lock --upgrade-package vllm` → commit → env rebuilds automatically (no
  manual version bump needed; the derivation content-addresses off the lock).
- Dev shell and service env share the same lock ⇒ "works in the shell" == service env.
- open-webui version now tracks the consumer's nixpkgs channel (documented tradeoff).

---

## 4. Planned repo layout

```
vllm-nix/
├── flake.nix                 # + inputs: pyproject-nix, uv2nix, pyproject-build-systems
├── nix/
│   ├── default.nix           # melds: models.nix, shells.nix, vllm-env.nix
│   ├── models.nix            # (unchanged) preset data
│   ├── shells.nix            # (unchanged)
│   └── vllm-env.nix          # NEW: uv2nix python set + vllmEnv package + checks
├── modules/
│   ├── default.nix           # NEW: nixosModules.vllm export
│   └── vllm.nix              # NEW: services.vllm module (D2–D5)
├── docs/
│   └── usage.md              # NEW: consumer-facing install/usage doc
├── pyproject.toml            # UNCHANGED (no open-webui/ddgs added)
├── uv.lock                   # UNCHANGED
└── PLAN.md                   # this file
```

---

## 5. Flake wiring

```nix
inputs = {
  nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  flake-utils = { … };
  pyproject-nix = { url = "github:pyproject-nix/pyproject.nix"; inputs.nixpkgs.follows = "nixpkgs"; };
  uv2nix = { url = "github:pyproject-nix/uv2nix"; inputs = { pyproject-nix.follows = "pyproject-nix"; nixpkgs.follows = "nixpkgs"; }; };
  pyproject-build-systems = { url = "github:pyproject-nix/build-system-pkgs"; inputs = { … follows … }; };
};

outputs = { flake-utils, ... } @ inputs:
  flake-utils.lib.meld inputs [
    ./nix               # existing shells/models + NEW packages.vllmEnv + checks
    ./modules           # NEW nixosModules.vllm
  ];
```

New attributes:

- `packages.x86_64-linux.vllmEnv` — the uv2nix virtualenv.
- `nixosModules.vllm` — self-contained module (receives `pkgs` from the consumer;
  the env comes in via `services.vllm.package`, so the module needs no flake inputs).
- `checks.x86_64-linux.vllmEnvImport` — `${vllmEnv}/bin/python -c "import vllm; print(vllm.__version__)"`
  (import only — no GPU needed; catches ELF/patching breakage).
- `checks.x86_64-linux.vllmModuleEval` — minimal `lib.nixosSystem`/`evalModules` config
  with `services.vllm.enable = true`, assert on
  `config.systemd.services.vllm.serviceConfig` (pure eval, no QEMU).

---

## 6. Implementation sketches

### 6.1 `nix/vllm-env.nix` (uv2nix)

```nix
{ pkgs, lib, pyproject-nix, uv2nix, pyproject-build-systems, ... }:
let
  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ../.; };

  overlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";   # enforce "no source builds"
  };

  # Fixups for GPU wheels: expect autoPatchelfIgnoreMissingDeps for libcuda.so.1
  # (driver-provided at runtime) on torch / nvidia-* / flashinfer; add lib deps
  # (stdenv.cc.cc.lib, zlib) where autoPatchelf reports missing. Fill in at M1.
  fixups = final: prev: { };

  pythonSet =
    (pkgs.callPackage pyproject-nix.build.packages { python = pkgs.python312; })
      .overrideScope (lib.composeManyExtensions [
        pyproject-build-systems.overlays.default
        overlay
        fixups
      ]);
in {
  packages.vllmEnv = pythonSet.mkVirtualEnv "vllm-env" workspace.deps.default;
}
```

### 6.2 `modules/default.nix`

```nix
_: { nixosModules.vllm = ./vllm.nix; }
```

(Flake-level: `nixosModules` is system-independent, exported outside `eachDefaultSystem`.)

### 6.3 `modules/vllm.nix` (core; corrected sketch)

```nix
{ config, lib, pkgs, ... }:
let
  cfg = config.services.vllm;

  serveArgs = lib.escapeShellArgs (
    [ cfg.model.hfId
      "--host" cfg.listenHost
      "--port" (toString cfg.listenPort)
      "--served-model-name" cfg.model.servedModelName
      "--max-model-len" (toString cfg.model.maxModelLen)
      "--max-num-seqs" (toString cfg.maxNumSeqs)
      "--max-num-batched-tokens" (toString cfg.maxNumBatchedTokens)
      "--gpu-memory-utilization" (toString cfg.gpuMemoryUtilization)
      "--tensor-parallel-size" (toString cfg.tensorParallelSize)
    ]
    ++ lib.optionals cfg.disableMultimodal
         [ "--limit-mm-per-prompt.image" "0" "--limit-mm-per-prompt.video" "0" ]
    ++ lib.optionals (cfg.mmEncoderTpMode != null) [ "--mm-encoder-tp-mode" cfg.mmEncoderTpMode ]
    ++ lib.optionals (cfg.kvCacheDtype != "auto") [ "--kv-cache-dtype" cfg.kvCacheDtype ]
    ++ lib.optionals (cfg.attentionBackend != null) [ "--attention-backend" cfg.attentionBackend ]
    ++ lib.optional cfg.enableAutoToolChoice "--enable-auto-tool-choice"
    ++ lib.optionals (cfg.toolCallParser != null) [ "--tool-call-parser" cfg.toolCallParser ]
    ++ lib.optionals (cfg.reasoningParser != null) [ "--reasoning-parser" cfg.reasoningParser ]
    ++ cfg.extraArgs
  );

  runtimeLibs = lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ];
in {
  options.services.vllm = { /* full D2 table; package = lib.mkOption { type = lib.types.package; }; */ };

  config = lib.mkIf cfg.enable {
    users.users.vllm = { isSystemUser = true; group = "vllm"; };
    users.groups.vllm = { };

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall
      ([ cfg.listenPort ] ++ lib.optional cfg.ui.enable cfg.ui.port);

    systemd.services.vllm-models = {
      description = "Download vLLM model weights";
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      environment.HF_HOME = "/var/lib/vllm/hf";
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        User = "vllm";  Group = "vllm";
        StateDirectory = "vllm";
        ExecStart = "${cfg.package}/bin/hf download ${lib.escapeShellArg cfg.model.hfId}";
        TimeoutStartSec = "12h";
      };
    };

    systemd.services.vllm = {
      description = "vLLM OpenAI-compatible inference server";
      after = [ "network-online.target" "vllm-models.service" ];
      wants = [ "network-online.target" "vllm-models.service" ];
      wantedBy = [ "multi-user.target" ];
      environment = {
        HF_HOME = "/var/lib/vllm/hf";
        LD_LIBRARY_PATH = "${runtimeLibs}:${cfg.driverLibPath}";
      } // cfg.environment;
      unitConfig = { StartLimitIntervalSec = 120; StartLimitBurst = 3; };
      serviceConfig = {
        Type = "exec";
        User = "vllm";  Group = "vllm";
        SupplementaryGroups = [ "video" "render" ];
        StateDirectory = "vllm";
        WorkingDirectory = "/var/lib/vllm";
        ExecStartPre = pkgs.writeShellScript "vllm-preflight" ''
          if [ ! -e ${cfg.driverLibPath}/libcuda.so.1 ]; then
            echo "vllm: ${cfg.driverLibPath}/libcuda.so.1 not found — is the NVIDIA driver installed via nixpkgs?" >&2
            exit 1
          fi
        '';
        ExecStart = "${cfg.package}/bin/vllm serve ${serveArgs}";
        Restart = "on-failure";
        RestartSec = "10s";
        TimeoutStartSec = cfg.timeoutStartSec;
        NoNewPrivileges = true;
        ProtectHome = true;
        ProtectSystem = "full";
        PrivateTmp = true;
      };
    };

    # ---- UI: reuse nixpkgs' module -------------------------------------
    services.open-webui = lib.mkIf cfg.ui.enable {
      enable = true;
      port = cfg.ui.port;
      environment = {
        OPENAI_API_BASE_URL = "http://127.0.0.1:${toString cfg.listenPort}/v1";
        WEBUI_AUTH = "False";   # confirm desired default at impl time
      } // lib.optionalAttrs cfg.ui.webSearch.enable {
        ENABLE_WEB_SEARCH = "true";
        WEB_SEARCH_ENGINE = "duckduckgo";
        WEB_SEARCH_RESULT_COUNT = "5";
        WEB_SEARCH_CONCURRENT_REQUESTS = "10";
        WEB_SEARCH_ENABLED_BY_DEFAULT = "true";
      } // cfg.ui.extraEnvironment;
    };
    systemd.services.open-webui = lib.mkIf cfg.ui.enable {
      after = [ "vllm.service" ];   # ordering only; no hard dependency
    };
  };
}
```

Implementation notes:

- `environment` on a NixOS systemd unit is an **attrset** (`environment.HF_HOME = …`).
- `StartLimit*` lives in `unitConfig`, not `serviceConfig`.
- All shell snippets via `pkgs.writeShellScript` / `${pkgs.runtimeShell}` (no `/usr/bin/sh`).
- Args built as a list + `lib.escapeShellArgs` (no manual quoting).
- `hf download` respects `HF_HOME` — no `--cache-dir` needed.
- Check exact env-var names accepted by the pinned open-webui module
  (`OPENAI_API_BASE_URL` vs `OPENAI_API_BASE_URLS`) at impl time.

### 6.4 Unit behaviour summary

| Unit | Type | Start | Restart | Timeouts |
| --- | --- | --- | --- | --- |
| `vllm-models` | oneshot | boot (wanted by vllm) | no (RemainAfterExit) | 12h (first-boot download) |
| `vllm` | exec | after vllm-models | on-failure /10s, burst-limited | `TimeoutStartSec=20m` default |
| `open-webui` | (nixpkgs module) | ordered after vllm | nixpkgs defaults | nixpkgs defaults |

Readiness: vllm has no native systemd readiness; v1 relies on `Restart=on-failure`.
open-webui tolerates a not-yet-ready backend, so no health-gating needed. Optional v2:
`ExecStartPost` curl loop against `/health`.

---

## 7. Milestones

**M1 — `vllmEnv` via uv2nix (½–1 day; the risk lives here)**

- Add flake inputs; write `nix/vllm-env.nix`; iterate on the `fixups` overlay until
  `nix build .#vllmEnv` succeeds with `sourcePreference = "wheel"`.
- Accept: `result/bin/python -c 'import vllm, torch'` works on the target box;
  `result/bin/vllm --version` → 0.28.0; `result/bin/hf --help` works.
- Watch-outs: torch/nvidia-*/flashinfer/triton wheels needing
  `autoPatchelfIgnoreMissingDeps` (libcuda) or extra lib deps (libstdc++, zlib);
  any dep with **no wheel** (lock has 154 sdist entries — most have wheel
  alternatives; if one doesn't, pin a wheel-capable version in `pyproject.toml`).

**M2 — Module + `vllm.service` on the real box (2 h)**

- Write `modules/vllm.nix` (+ `modules/default.nix`), add module eval check.
- Rehearsed rollout: back up current config, `nixos-rebuild test` first, then `switch`.
- Accept: `vllm-models` downloads (watch `journalctl -fu vllm-models`), `vllm` binds
  :8000 as user `vllm`; `curl localhost:8000/v1/models` lists `Qwen3.8-27B-INT4`;
  `kill -9` the process → systemd restarts within ~10s; `/var/lib/vllm` owned by vllm.
- Flip to FP8Cache preset options and re-switch to validate both presets.

**M3 — UI wiring (½ h)**

- `ui.enable = true` → nixpkgs `services.open-webui` comes up on :8080, talks to vllm,
  web-search env applied. Verify env-var names against the module's open-webui version.

**M4 — Docs + checks + polish (1 h)**

- `docs/usage.md`: consumer snippet, both preset blocks, options table, update
  procedure (`uv lock --upgrade-package vllm`), troubleshooting (missing libcuda, OOM
  restart loop, open-webui version drift), `openFirewall` note.
- `nix flake check` green; README pointer.

Total estimate: **~1.5–2 days**, dominated by M1 fixups + model-download wall time.

---

## 8. Test & rollout plan

1. **No-GPU CI**: `nix flake check` (env import check catches ELF/patchelf breakage
   without a GPU; module eval check catches option regressions). QEMU cannot exercise
   CUDA — accepted limitation.
2. **Local (this box)**: staged rollout per M2/M3; old shell scripts untouched, so
   rollback = `nixos-rebuild switch` to previous generation.
3. **Smoke suite (manual, documented in docs/usage.md)**:
   - `systemctl status vllm vllm-models open-webui`
   - `curl -s localhost:8000/v1/models | jq`
   - one `chat/completions` request via curl
   - `journalctl -u vllm | tail` after a simulated crash
4. **Other-machine matrix (v1.1)**: a nixos-26.05-channel box (module must not use
   unstable-only attrs; note `services.open-webui` availability/version there) and a
   closed-`nvidia`-driver box (verify `/run/opengl-driver/lib/libcuda.so.1`).

---

## 9. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| uv2nix fixups for torch/flashinfer wheels are fiddly | **medium-high** | M1 takes longer | Known-territory: uv2nix docs + community torch overlays; import-check pins down breakage early; timeboxed in M1 |
| A locked dep has no wheel (`sourcePreference = "wheel"` fails) | low-medium | build failure | Pin a wheel-capable version in pyproject; or allow sdist for that one pkg with build inputs |
| `/run/opengl-driver` absent (non-nixpkgs driver) | medium | service fails | `driverLibPath` override + clear preflight message |
| First-boot 16 GB download interrupted | medium | delayed start | `vllm-models` idempotent; `systemctl restart vllm-models` documented |
| open-webui env-var names differ across nixpkgs versions | low-medium | UI misconfigured | M3 verifies against pinned module; `ui.extraEnvironment` escape hatch |
| open-webui version drifts with consumer nixpkgs | medium (by design) | UI behaviour changes | Documented tradeoff of reusing the nixpkgs module |
| 0.95 GPU mem util OOM → restart loop | low | flapping | burst-limited restarts; `gpuMemoryUtilization` option |
| `vllm` user lacks `/dev/nvidia*` access on some setups | low | startup failure | `SupplementaryGroups = [ video render ]`; preflight message; documented |
| Consumer nixpkgs differs from ours (module evals against theirs) | high (by design) | minor | Module only needs standard NixOS options; env is prebuilt via our flake's locked inputs |

---

## 10. Open questions (answer during M2/M3)

1. **open-webui auth**: default `WEBUI_AUTH=False` (single-user LAN box, matches current
   usage) or leave open-webui's default auth on? (Proposed: leave nixpkgs default,
   document the env var.)
2. **`ui.webSearch` defaults**: keep the `run-ui.sh` duckduckgo settings as module
   defaults? (Proposed: yes, behind `ui.webSearch.enable`.)

---

## 11. Future ideas (out of scope for v1)

- `nixosConfigurations.example-vllm` turnkey VM example in this repo.
- Upstream the module to nixpkgs.
- Preset helper: `services.vllm.preset = "Qwen3.8-27B-INT4-FP8Cache"` mapping
  `nix/models.nix` entries → option overrides.
- Multi-model `vllm@name.service` template units.
- Prometheus metrics scrape + `ExecStartPost` readiness gate.
- Cachix publishing of `vllmEnv` so consumers skip wheel downloads.
