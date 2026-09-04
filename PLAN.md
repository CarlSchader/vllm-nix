# Plan: `vllm-nix` as a NixOS Module + systemd Service

Goal: package the current vLLM server setup (vllm 0.28.0, Qwen3.8-27B-INT4 presets,
open-webui front-end) into a **NixOS module** that installs a working, auto-restarting
systemd service stack on **any NixOS machine with a CUDA-capable NVIDIA GPU** (managed
by nixpkgs, e.g. `hardware.nvidia.packages = [ pkgs.nvidia-open ]`).

Consumers would do:

```nix
# in their NixOS configuration
{ imports = [ (builtins.getFlake "github:carlschader/vllm-nix").nixosModules.vllm ];

  services.vllm = {
    enable = true;
    package = (builtins.getFlake "github:carlschader/vllm-nix").packages.${system}.vllmEnv;
    model = { hfId = "RedHatAI/Qwen3.8-27B-INT4"; servedModelName = "Qwen3.8-27B-INT4"; maxModelLen = 32768; };
  };
}
```

---

## 1. Current state (what we are packaging)

| Piece | Today | Notes |
|---|---|---|
| Python env | `uv` project, `uv.lock` pins **vllm 0.28.0**, torch 2.13.0, flashinfer-python 0.6.16.post3, triton 3.7.1 | All prebuilt PyPI wheels; python 3.12 |
| Launch | `Qwen3.8-27B-INT4.sh` / `Qwen3.8-27B-INT4-FP8Cache.sh` → `uv run vllm serve … --host 0.0.0.0` (port 8000) | Two presets = same HF model, different flags: 32k ctx vs 65k ctx + `--kv-cache-dtype fp8 --attention-backend TRITON_ATTN` |
| UI | `run-ui.sh` → `uvx --with ddgs open-webui serve` (port 8080) with web-search env vars + `.webui_secret_key` file | `open-webui` is **not** in nixpkgs |
| Dev shell | `nix/shells.nix`: `cudaPackages_13.cudatoolkit`, `addDriverRunpath` driver link in `LD_LIBRARY_PATH`, `CUDA_HOME`, `TRITON_LIBCUDA_PATH` | Build-time env only; runtime only needs the driver's `libcuda.so.1` |
| Flakes | `flake.nix` melds `nix/{default,models,shells}.nix`; nixpkgs-unstable (rev `e8be781`), nix-community cachix | No `packages` or `nixosModules` yet |
| Target hw | RTX 4090 24 GB, nixpkgs-built open driver 595.91.07 | `/run/opengl-driver/lib` exposes `libcuda.so.1` (verified on this machine) |

Key facts verified against the locked nixpkgs rev:

- `python312Packages.vllm` exists at **0.24.0** (full source build w/ flash-attn, cutlass,
  deepgemm, flashmla — heavy build) — two versions behind our lock.
- `pkgs.uv` = **0.12.5** — reads our `uv.lock` (revision 3) fine.
- `addDriverRunpath.driverLink` = `/run/opengl-driver` — the portable, NixOS-generic
  way to reach the active driver's `libcuda.so.1`.
- `open-webui` has **no** nixpkgs python package → we ship it from the PyPI lock.

---

## 2. Goals / non-goals

**Goals**
- `nixosModules.vllm`: declarative, typed, documented module (`services.vllm.*`).
- `packages.<sys>.vllmEnv`: a relocatable, reproducible python env built **only from the
  pinned PyPI wheels** in `uv.lock` (no source builds).
- systemd: `vllm.service` (+ `vllm-models.service` one-shot downloader, + optional
  `open-webui.service`), auto-restart, persistent HF cache, clean first-boot model download.
- Works on any x86_64-linux NixOS box with a nixpkgs-managed NVIDIA driver (no GPU in
  CI/QEMU ⇒ test via import-check + manual deploy, see §8).
- `checks.*` so CI can build the env and eval the module.

**Non-goals (v1)**
- Multi-GPU / multi-node serving, ROCm/AMD, CPU fallback.
- NixOS config for this repo itself (we don't own the machine's config here).
- Upstreaming to nixpkgs (kept as a later option, §11).
- TLS/reverse-proxy/auth in front of the API.

---

## 3. Key design decisions

### D1 — How to package the Python env (the big one)

| Option | What | Pros | Cons | Verdict |
|---|---|---|---|---|
| **A. nixpkgs `python312Packages.vllm`** | Use 0.24.0 as-is | Fully nix-native, reproducible, maintained | 2 versions **behind** our working 0.28.0; **very heavy** source build (compiles flash-attention/cutlass/deepgemm/flashmla against cutlass v4.4.2); behaviour may diverge from the setup we validated | Rejected for v1 |
| **B. `uv sync --frozen` derivation from `uv.lock`** ✅ | stdenv derivation that runs `uv venv` + `uv sync --frozen --no-dev` against the repo's lock | **Exact same pinned wheels** as the working `.venv` (vllm 0.28.0); fast build (wheel download only, no CUDA compilation); version bumps = update `uv.lock` locally and commit; lock has content hashes ⇒ reproducible | Env derivation is "custom" (not a python3Packages derivation); platform pinned to x86_64-linux (fine); `uv` must match lock rev (0.12.5 ✓) | **Chosen** |
| C. `buildPythonApplication` per package | Reproduce the whole dep tree as nix expressions | nix-idiomatic | Hand-porting a ~100-dep tree incl. torch/flashinfer = huge maintenance, drift from lock | Rejected |

Chosen implementation (sketch in §6.1):

```nix
pkgs.stdenv.mkDerivation {
  pname = "vllm-env";
  src = lib.fileset.toSourcePath { root = ./.; fileset = ./: [ "/uv.lock" "/pyproject.toml" ]; };
  nativeBuildInputs = [ pkgs.uv ];
  env.UV_PROJECT_ENVIRONMENT = ".venv";
  buildPhase = ''
    uv venv --relocatable --python ${pkgs.python312} .venv
    uv sync --frozen --no-dev
  '';
  installPhase = ''
    mv .venv $out   # => $out/bin/vllm, $out/bin/hf, $out/bin/open-webui
  '';
}
```

Notes:
- Add `open-webui` + `ddgs` to the uv project (`uv add open-webui ddgs`) so one env
  provides **both** binaries. This keeps `pyproject.toml` the single source of truth.
- `uv sync` uses only the lock (offline of the build machine's site-packages), wheels are
  prebuilt ⇒ build is a few GB of downloads, minutes of CPU.
- The venv lives at a fixed store path ⇒ `--relocatable` is belt-and-braces.
- Runtime deps of the torch stack: torch wheels bundle the CUDA **runtime** via `nvidia-*`
  pip deps; the only system-side requirement is the driver's `libcuda.so.1` (D4).
- Build machine constraint: must build on x86_64-linux (wheels are platform-specific).

### D2 — Module API: `services.vllm`

Typed submodule; structured options for the flags we actually use, plus an `extraArgs`
escape hatch for anything else. Full option table:

| Option | Type | Default | Maps to |
|---|---|---|---|
| `services.vllm.enable` | bool | false | gates everything |
| `services.vllm.package` | str (store path) | — (required when enabled) | env derivation path |
| `services.vllm.listenHost` | str | `"0.0.0.0"` | `--host` |
| `services.vllm.listenPort` | port | 8000 | `--port` |
| `services.vllm.user` | str | `"root"` | unit `User=` (see D5) |
| `services.vllm.dataDir` | str | `"/var/lib/vllm"` | `WorkingDirectory`, `HF_HOME`, vllm state |
| `services.vllm.driverLibPath` | str | `"/run/opengl-driver/lib"` | prepended to `LD_LIBRARY_PATH` |
| `services.vllm.model.hfId` | str | `"RedHatAI/Qwen3.8-27B-INT4"` | `vllm serve <id>` |
| `services.vllm.model.servedModelName` | str | same as hfId | `--served-model-name` |
| `services.vllm.model.maxModelLen` | int | 32768 | `--max-model-len` |
| `services.vllm.maxNumSeqs` | int | 8 | `--max-num-seqs` |
| `services.vllm.maxNumBatchedTokens` | int | 2048 | `--max-num-batched-tokens` |
| `services.vllm.gpuMemoryUtilization` | float | 0.95 | `--gpu-memory-utilization` |
| `services.vllm.tensorParallelSize` | int | 1 | `--tensor-parallel-size` |
| `services.vllm.kvCacheDtype` | str | `"auto"` | `--kv-cache-dtype` (set `fp8` for the FP8Cache preset) |
| `services.vllm.attentionBackend` | str? | null | `--attention-backend` (set `TRITON_ATTN`) |
| `services.vllm.enableAutoToolChoice` | bool | true | `--enable-auto-tool-choice` |
| `services.vllm.toolCallParser` | str? | `"qwen3_coder"` | `--tool-call-parser` |
| `services.vllm.reasoningParser` | str? | `"qwen3"` | `--reasoning-parser` |
| `services.vllm.extraArgs` | listOf str | [] | appended verbatim (also `--limit-mm-per-prompt.image 0`-style pairs) |
| `services.vllm.environment` | listOf str | [] | extra `Environment=` lines |
| `services.vllm.timeoutStartSec` | str | `"20m"` | unit `TimeoutStartSec` (model load) |
| `services.vllm.ui.enable` | bool | false | `open-webui.service` |
| `services.vllm.ui.port` | port | 8080 | |
| `services.vllm.ui.dataDir` | str | `"/var/lib/open-webui"` | data dir + secret key location |
| `services.vllm.ui.webSearch` | attrs | see `run-ui.sh` env | `ENABLE_WEB_SEARCH` etc. |

Preset support (keeps `nix/models.nix` alive without coupling the module to it):
the flake exposes a `models` attribute set (derived from `nix/models.nix`). The module
itself stays preset-agnostic; the README documents both presets as copy-paste option
blocks (the two presets differ only in `model.maxModelLen`, `kvCacheDtype`,
`attentionBackend`). Later, a small `vllmPresets` helper could map a preset name →
overrides — not in v1 scope.

### D3 — Service topology

```
network-online.target
        │
vllm-models.service (oneshot, RemainAfterExit=yes)
  └─ hf download <hfId>  (HF_HOME=/var/lib/vllm/hf)
        │ After= / Wants=
vllm.service (Type=exec, Restart=on-failure)
        │ After= / Wants=  (only when ui.enable)
open-webui.service (Type=exec, Restart=on-failure)
```

- **`vllm-models.service`** — first-boot model fetch (≈16 GB) as its own unit so it has
  its own (long) timeout and visible progress, and so vllm start-time is not gated by
  downloads after the first boot. `ExecStart=${pkg}/bin/hf download $HF_ID --cache-dir
  $HF_HOME/hub`, `After=network-online.target`, `Wants=network-online.target`.
  Idempotent: re-runs are cheap no-ops.
- **`vllm.service`** — the server. Details in §6.4.
- **`open-webui.service`** — `ExecStart=${pkg}/bin/open-webui serve`, with an
  `ExecStartPre` that (a) blocks until `http://127.0.0.1:8000/health` is OK (bounded
  curl retry loop) and (b) generates `SECRET_KEY` in `/var/lib/open-webui/.secret_key`
  once if missing (replaces the hand-made `.webui_secret_key`; open-webui reads the
  `SECRET_KEY` env var). `OPENWEBUI_DATA_DIR=/var/lib/open-webui` keeps the SQLite DB
  persistent.

### D4 — Driver / CUDA wiring

- Runtime needs exactly one thing from the driver: `libcuda.so.1`.
- NixOS-generic location: `/run/opengl-driver/lib` (`addDriverRunpath.driverLink`),
  verified present on the target box. ⇒ unit `Environment=LD_LIBRARY_PATH=/run/opengl-driver/lib`
  (overridable via `services.vllm.driverLibPath`; set it to `""` on boxes where the
  system default path already finds libcuda).
- `CUDA_HOME` / `TRITON_LIBCUDA_PATH` were **build-time** needs in the dev shell (nvcc
  for flashinfer JIT); they are **not** required at serve-time. The module does not set
  them by default (documented; add via `environment` if triton JIT errors appear).
- Sanity gate in `vllm.service` `ExecStartPre`:
  `test -e ${driverLibPath}/libcuda.so.1` and `nvidia-smi -L` (from nixpkgs
  `nvidia-smi`… actually the driver provides `nvidia-smi` under `/run/opengl-driver/bin`? —
  use `command -v nvidia-smi || ${driverLibPath%/lib}/bin/nvidia-smi`), failing the unit
  with a clear `journalctl` message instead of a cryptic vllm traceback.

### D5 — State, user, security

- **State**: `dataDir` (default `/var/lib/vllm`) holds `hf/` (HF cache), vllm logs go to
  journald, nothing else persistent. open-webui state in `/var/lib/open-webui`
  (DB + secret key).
- **User**: default `root` (matches current interactive use, zero permission surprises
  with `/dev/nvidia*`). Option `user` documented for dedicated-user setups: the user must
  be in the group owning `/dev/nvidia*` (on nixpkgs open-driver NixOS boxes that's
  typically `render`/`video` — add `SupplementaryGroups` when `user` is set; v1 sets
  `SupplementaryGroups = [ "render" "video" ]` when `user != "root"`).
- **Hardening** (things that are safe *and* don't break GPU access):
  `NoNewPrivileges=true`, `ProtectHome=true`, `ProtectSystem=full` + `ReadWritePaths=${dataDir}`,
  `PrivateTmp=true`. Deliberately **not** set: `PrivateDevices`, `RestrictAddressFamilies`,
  `ProtectKernel*`-heavy profiles — GPU device nodes and CUDA IPC need the default
  device namespace.
- **Restart policy**: `Restart=on-failure`, `RestartSec=10`,
  `StartLimitIntervalSec=120`/`StartLimitBurst=3` (avoid hot-loop on OOM/crash),
  `TimeoutStartSec` generous (model load + first-time triton compilation).

### D6 — Version pinning & updates

- The env is reproducible **only** via `uv.lock` (hashes included). Updating vllm =
  `uv add 'vllm==X.Y.Z'` (or `uv lock --upgrade-package vllm`) locally → commit the lock →
  new derivation version (bump a `envVersion` string or use the lock's mtime/rev).
- Document that the dev shell (`nix/shells.nix`) and the service env intentionally share
  the same lock, so "works in the shell" == "what the service runs".

---

## 4. Planned repo layout

```
vllm-nix/
├── flake.nix                 # + nixosModules, packages (see §6.2)
├── nix/
│   ├── default.nix           # melds: models.nix, shells.nix, vllm-env.nix (lib)
│   ├── models.nix            # (unchanged) preset data
│   ├── shells.nix            # (unchanged)
│   └── vllm-env.nix          # NEW: vllm-env derivation (D1)
├── modules/
│   ├── default.nix           # NEW: exports all modules (for flake nixosModules)
│   └── vllm.nix              # NEW: services.vllm NixOS module (D2–D5)
├── docs/
│   └── usage.md              # NEW: consumer-facing install/usage doc
├── pyproject.toml            # + open-webui, ddgs deps (D1)
├── uv.lock                   # updated by `uv add open-webui ddgs`
└── PLAN.md                   # this file
```

---

## 5. Flake wiring

```nix
outputs = { flake-utils, ... } @ inputs:
  flake-utils.lib.meld inputs [
    ./nix               # existing: models, shells, + vllmEnv package
    ./modules           # NEW: nixosModules.default -> modules/vllm.nix
    ./checks            # NEW: env import test + module eval test
  ];
```

New attributes produced:

- `packages.<system>.vllmEnv` — the uv env (option A of `D1`; `vllm-env` alias optional).
- `nixosModules.vllm` — the module (self-contained: imports only `nixpkgs` via the
  flake's `inputs`/passthru — modules receive `pkgs` from the consuming config, so the
  module itself needs no flake inputs).
- `checks.<system>.vllmEnvImport` — build env, run `bin/python -c "import vllm; print(vllm.__version__)"`
  (import only — no GPU needed).
- `checks.<system>.vllmModuleEval` — `import nixpkgs ... (import <module> { config=…; })`
  in a minimal test config with `services.vllm.enable = true`, assert
  `config.systemd.services.vllm.serviceConfig.Type == "exec"` (pure Nix eval, no QEMU).

---

## 6. Implementation sketches

### 6.1 `nix/vllm-env.nix`

```nix
{ lib, stdenv, uv, python312 }:

stdenv.mkDerivation {
  pname = "vllm-env";
  version = "2026-09-04";  # bump when uv.lock changes

  src = lib.fileset.toSourcePath {
    root = ./.;
    fileset = ./: [ "/uv.lock" "/pyproject.toml" ];
  };

  nativeBuildInputs = [ uv ];

  env = {
    UV_PROJECT_ENVIRONMENT = ".venv";
    UV_CACHE_DIR = "$TMPDIR/uv-cache";   # don't pollute /nix/store with cache cruft
  };

  buildPhase = ''
    runHook preBuild
    uv venv --relocatable --python ${python312} .venv
    uv sync --frozen --no-dev
  '';

  installPhase = ''
    runHook preInstall
    mv .venv $out
    # smoke test: python + vllm import must work in this env (no CUDA init)
    $out/bin/python -c 'import vllm, torch; assert vllm.__version__'
    runHook postInstall
  '';
}
```

### 6.2 `flake.nix` / `nix/default.nix` additions

```nix
# nix/default.nix (extend the existing per-system let)
in {
  devShells.default = /* unchanged */;
  packages.vllmEnv = pkgs.callPackage ../vllm-env.nix {};
}

# modules/default.nix
{ inputs }: {
  nixosModules.vllm = ./vllm.nix;
}
```

### 6.3 `modules/vllm.nix` (core of the module)

```nix
{ config, lib, pkgs, ... }:

let
  cfg = config.services.vllm;

  # build CLI args from structured options (D2 table)
  modelArgs = lib.optionalStrings (
    [ "--served-model-name" cfg.model.servedModelName
      "--max-model-len" (toString cfg.model.maxModelLen)
      "--max-num-seqs" (toString cfg.maxNumSeqs)
      "--max-num-batched-tokens" (toString cfg.maxNumBatchedTokens)
      "--gpu-memory-utilization" (toString cfg.gpuMemoryUtilization)
      "--tensor-parallel-size" (toString cfg.tensorParallelSize)
    ]
    ++ lib.optionals (cfg.kvCacheDtype != "auto") [ "--kv-cache-dtype" cfg.kvCacheDtype ]
    ++ lib.optionals (cfg.attentionBackend != null) [ "--attention-backend" cfg.attentionBackend ]
    ++ lib.optional cfg.enableAutoToolChoice "--enable-auto-tool-choice"
    ++ lib.optionals (cfg.toolCallParser != null) [ "--tool-call-parser" cfg.toolCallParser ]
    ++ lib.optionals (cfg.reasoningParser != null) [ "--reasoning-parser" cfg.reasoningParser ]
    ++ cfg.extraArgs
  );

in
{
  options.services.vllm = {
    enable = lib.mkOption { type = lib.types.bool; default = false; description = "…"; };
    package = lib.mkOption { type = lib.types.str; description = "Store path of the vllmEnv derivation (…/bin/vllm)."; };
    # …full table from D2, each with description + sensible default…
    ui.enable = lib.mkOption { type = lib.types.bool; default = false; };
    # …
  };

  config = lib.mkIf cfg.enable {

    # ---- model prefetch (D3) --------------------------------------------
    systemd.services."vllm-models" = {
      description = "Download vLLM model weights (first boot)";
      wantedBy = [ "vllm.service" ];
      after  = [ "network-online.target" ];
      wants  = [ "network-online.target" ];
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        User = cfg.user;
        Environment = [ "HF_HOME=${cfg.dataDir}/hf" ];
        ExecStart = lib.fmt "${cfg.package}/bin/hf download %s --cache-dir %s/hub" cfg.model.hfId cfg.dataDir;
        TimeoutStartSec = "12h";
      };
    };

    # ---- the server -------------------------------------------------------
    systemd.services.vllm = {
      description = "vLLM OpenAI-compatible inference server";
      after  = [ "network-online.target" "vllm-models.service" ];
      wants  = [ "network-online.target" "vllm-models.service" ];
      wantedBy = [ "multi-user.target" ];
      environment = [
        "HF_HOME=${cfg.dataDir}/hf"
      ]
      ++ lib.optionals (cfg.driverLibPath != "") [ "LD_LIBRARY_PATH=${cfg.driverLibPath}" ]
      ++ cfg.environment;
      serviceConfig = {
        Type = "exec";
        User = cfg.user;
        Group = cfg.user;
        SupplementaryGroups = lib.optional (cfg.user != "root") "render";
        WorkingDirectory = cfg.dataDir;
        ExecStartPre = [
          (lib.fmt "test -e %s/libcuda.so.1" cfg.driverLibPath)   # fail fast, clear message
        ];
        ExecStart = lib.fmt "${cfg.package}/bin/vllm serve %s %s --host %s --port %s"
          cfg.model.hfId (lib.unsafeSeq modelArgs) cfg.listenHost (toString cfg.listenPort);
        # (implementation note: join modelArgs with a space in the real code)
        Restart = "on-failure";
        RestartSec = "10s";
        StartLimitIntervalSec = 120;
        StartLimitBurst = 3;
        TimeoutStartSec = cfg.timeoutStartSec;
        # hardening (D5)
        NoNewPrivileges = true;
        ProtectHome = true;
        ProtectSystem = "full";
        ReadWritePaths = [ cfg.dataDir ];
        PrivateTmp = true;
      };
    };

    # ---- optional web ui (D3) ---------------------------------------------
    systemd.services."open-webui" = lib.mkIf cfg.ui.enable {
      description = "Open WebUI for vLLM";
      after  = [ "network-online.target" "vllm.service" ];
      wants  = [ "vllm.service" ];
      wantedBy = [ "multi-user.target" ];
      environment = [
        "OPENWEBUI_DATA_DIR=${cfg.ui.dataDir}"
        "OPENAI_API_BASE=http://127.0.0.1:${toString cfg.listenPort}/v1"
      ] ++ lib.optionalStrings ( # web-search env from run-ui.sh
        [ "ENABLE_WEB_SEARCH=true" "WEB_SEARCH_ENGINE=duckduckgo"
          "WEB_SEARCH_RESULT_COUNT=5" "WEB_SEARCH_CONCURRENT_REQUESTS=10"
          "WEB_SEARCH_ENABLED_BY_DEFAULT=true" ]
      );
      serviceConfig = {
        Type = "exec";
        User = cfg.user;
        WorkingDirectory = cfg.ui.dataDir;
        EnvironmentFiles = [ "${cfg.ui.dataDir}/env" ];   # SECRET_KEY (generated once)
        ExecStartPre = [
          # wait for vllm /health (bounded)
          "/usr/bin/sh -c 'for i in $(seq 1 60); do curl -fsS http://127.0.0.1:${toString cfg.listenPort}/health && break; sleep 5; done'"
          # create secret key + env file once
          (lib.fmt "test -f %s/.secret_key || { head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \\\n' > %s/.secret_key; printf 'SECRET_KEY=%%s' \\"$(cat %s/.secret_key)\\" > %s/env; }"
            cfg.ui.dataDir cfg.ui.dataDir cfg.ui.dataDir cfg.ui.dataDir)
        ];
        ExecStart = "${cfg.package}/bin/open-webui serve --port ${toString cfg.ui.port}";
        # open-webui CLI: confirm exact flags at impl time (serve may take PORT env)
        Restart = "on-failure";
        RestartSec = "10s";
        NoNewPrivileges = true;
        ProtectHome = true;
        ProtectSystem = "full";
        ReadWritePaths = [ cfg.ui.dataDir ];
        PrivateTmp = true;
      };
    };

    # keep state dirs around
    users.users.vllm-state ? { } # not needed; systemd creates dataDir? No — see note
    # NOTE: add a small helper unit or use systemd.tmpfiles for /var/lib/vllm
  };
}
```

**Implementation notes / open items from the sketch:**
- `ExecStart` needs the args list joined (`lib.concatStringsSep " "`); model ids with
  special chars get quoted via `lib.escape` — use `lib.escape` on hfId and
  served-model-name defensively.
- `Environment=` in a nix systemd module is a list of `KEY=VAL` strings (as written).
- `/var/lib/vllm` creation: prefer a `systemd.tmpfiles.rules."vllm"` entry
  (`d /var/lib/vllm 0770 vllm:render -`) or `ForceFlags`... simplest:
  `ExecStartPre` mkdir is ugly; tmpfiles rule is cleanest (module adds it).
- `nvidia-smi` presence check: fold into the same `ExecStartPre` test line; the driver
  ships the binary — locate via `command -v` in a `/usr/bin/sh -c` wrapper if not on PATH.
- `User = cfg.user; Group = cfg.user` — root:root is fine; for a custom user, create the
  user in the module (`users.users.<cfg.user> = isSystemUser or ensureExistingUser`) —
  v1: assert the user exists instead of creating (documented).

### 6.4 Unit behaviour summary (what the module produces)

| Unit | Type | Start | Restart | Timeouts |
|---|---|---|---|---|
| `vllm-models` | oneshot | boot (wanted by vllm) | no (RemainAfterExit) | 12h (first-boot download) |
| `vllm` | exec | after vllm-models | on-failure /10s, burst-limited | `TimeoutStartSec=20m` default |
| `open-webui` | exec | after vllm healthy | on-failure /10s | default |

Health/readiness: vllm has no native readiness for systemd; v1 relies on
`Restart=on-failure` + the open-webui `ExecStartPre` curl-wait (which doubles as the
integration check). Optional v2: `ExecStartPost` curl loop + `Type=simple` stays.

---

## 7. Milestones

**M0 — Lock the env (½ h)**
- `uv add open-webui ddgs` → update `pyproject.toml` + `uv.lock`.
- Accept: `uv run open-webui serve --help` works; `uv run vllm serve --help` unchanged.

**M1 — `vllmEnv` derivation (2–4 h)**
- Write `nix/vllm-env.nix`; expose `packages.<sys>.vllmEnv`; add import `check`.
- Accept: `nix build .#packages.x86_64-linux.vllmEnv` succeeds from a clean cache miss
  scenario (wheel fetches from PyPI only); `nix run .#vllmEnv-...`? No — run
  `/nix/store/<env>/bin/python -c 'import vllm'`; `vllm --version` prints 0.28.0.
- Watch-outs: uv cache dir inside build; `--no-dev`; python must be 3.12 (lock's
  requires-python); verify no `pip`-build of any dep (any sdist in the lock would break
  the "no source build" promise — if `uv sync` tries to build one, add `buildInputs`
  (gcc, cudaPackages_13) or pin to a wheel-only version).

**M2 — Module + `vllm.service` on the real box (2 h)**
- Write `modules/vllm.nix` (server + models units only; UI off by default).
- Add the module to this machine's NixOS config in a **rehearsed** way:
  `cp` the current `configuration.nix` to a `.bak`, apply,
  `nixos-rebuild switch` (or `test` first for a dry eval).
- Accept: `systemctl enable --now vllm` pulls in `vllm-models` first (watch the 16 GB
  download in `journalctl -fu vllm-models`), then `vllm` binds :8000;
  `curl localhost:8000/v1/models` lists `Qwen3.8-27B-INT4`; `kill -9` the vllm process →
  systemd restarts it within ~10s.
- Validate both presets by flipping options (`kvCacheDtype="fp8"`,
  `attentionBackend="TRITON_ATTN"`, `maxModelLen=65536`) and re-switching.

**M3 — open-webui service (1 h)**
- `ui.enable = true`; accept: `open-webui` unit starts only after `/health` passes;
  first-boot generates `.secret_key` + `env` file; `http://localhost:8080` serves;
  model reachable from the UI; `systemctl daemon-reload`-safe across rebuilds.

**M4 — Docs + checks + polish (1 h)**
- `docs/usage.md`: consumer snippet, preset snippets, options table, update procedure,
  troubleshooting (missing libcuda, OOM restart loops, GPU-in-QEMU caveat).
- `checks.*` green; flake output names final; commit `uv.lock`; README pointer.

Total estimate: **~1–1.5 days** including build-time and model-download wall time.

---

## 8. Test & rollout plan

1. **No-GPU CI**: `nix flake check` (env import check + module eval check). QEMU cannot
   exercise CUDA — accepted limitation.
2. **Local (this box)**: staged rollout as M2/M3; keep the old shell scripts untouched so
   rollback = `nixos-rebuild switch` back to the previous config (store keeps both).
3. **Smoke suite (manual, documented in docs/usage.md)**:
   - `systemctl status vllm vllm-models open-webui`
   - `curl -s localhost:8000/v1/models | jq`
   - one completion via `curl` `chat/completions` (sanity of the whole stack)
   - `journalctl -u vllm | tail` after a simulated crash
4. **Other-machine matrix (v1.1)**: one box on a different nixpkgs channel (e.g.
   nixos-26.05 — flake has that line commented already) to confirm the module doesn't
   depend on unstable-only attrs; one box with `nvidia` (closed) driver instead of
   `nvidia-open` (verify `/run/opengl-driver/lib/libcuda.so.1` still present).

---

## 9. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `uv sync` hits an sdist (source build) in the lock | low | build needs cuda/gcc toolchain | M1 watch-out; add `buildInputs` or bump pins if so |
| nixpkgs `uv` lags lock-file format on a *consumer's* nixpkgs channel | low | `uv sync` fails for consumers on old channels | Document min nixpkgs (unstable 2026-08+); module could take a `uv` package override option |
| `/run/opengl-driver` absent (non-nix managed driver) | medium | service fails | `driverLibPath = ""` override + clear `ExecStartPre` message in journald |
| 16 GB first-boot download interrupted (no `Wants=network-online` guarantee) | medium | vllm starts, re-download on next boot | `vllm-models` is idempotent; document `systemctl restart vllm-models` |
| open-webui CLI flags differ from `uvx` invocation | medium | unit won't start | M3: verify `open-webui serve --help` from the built env before writing the unit |
| 0.95 GPU mem util + both presets toggling | low | OOM kills → restart loop | burst-limited restart; `gpuMemoryUtilization` is an option |
| Module pulls in `pkgs` from consumer's nixpkgs (different rev than our env) | high (by design) | env derivation builds against *their* python312/uv | That's fine — env only needs uv + python; document; optional: flake-provided `packages.vllmEnv` always built against *our* locked nixpkgs (recommended consumer path) |
| `Type=exec` + long model-load timeout across nixos versions | low | unit marked failed | `TimeoutStartSec` option; document `systemd-analyze` |

---

## 10. Open questions (answer before/during M2)

1. **Default user**: root (proposed) vs auto-created `vllm` system user? (Pro: root
   matches current behaviour and GPU group pain; Con: not great hygiene.)
2. **Should the module create `users.users.vllm`** when `user` is set, or assert-exists?
   (Proposed: assert, document.)
3. **Firewall**: auto-open 8000/8080 via `networking.firewall` when enabled?
   (Proposed: no — document; machines often put the UI behind tailscale/tun0.)
4. **Env versioning**: hardcode `version = "2026-09-04"` in the derivation or derive from
   `uv.lock` hash? (Proposed: date string, bumped per lock change — simpler store names.)
5. **open-webui extras**: `ddgs` is only for web-search; keep the web-search defaults
   from `run-ui.sh` as the module defaults? (Proposed: yes, with a `ui.webSearch = {}`
   override.)

---

## 11. Future ideas (out of scope for v1)

- `nixosConfigurations.example-vllm` in this repo for a turnkey VM.
- Upstream `nixosModules` to nixpkgs (vllm module is self-contained enough once the env
  is parameterized).
- Multi-model: `services.vllm.models.<name>` list → `vllm@name.service` template unit
  (needs multiple GPUs or strict scheduling; memory math).
- Prometheus scrape port + `systemd` `ExecStartPost` readiness gate.
- ROCm variant of the env derivation (`uv lock` with `--resolution`/markers).
- NUR/cachix publishing of `vllmEnv` so consumers skip the ~10 GB wheel download.
