# Using vllm-nix on a NixOS host

`vllm-nix` packages a pinned vLLM 0.28.0 environment (built from `uv.lock`
with uv2nix — prebuilt wheels only, no source builds) and a NixOS module that
runs it as a hardened, auto-restarting systemd service, optionally fronted by
Open WebUI.

Requirements: an x86_64-linux NixOS host with a CUDA-capable NVIDIA GPU
managed by nixpkgs (`hardware.nvidia`, driver libs on `/run/opengl-driver/lib`).

## Quick start

```nix
{
  inputs.vllm-nix.url = "github:carlschader/vllm-nix";

  outputs = { nixpkgs, vllm-nix, ... }: {
    nixosConfigurations.gpu-box = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        vllm-nix.nixosModules.vllm
        ({ pkgs, ... }: {
          nixpkgs.config.allowUnfree = true;

          services.vllm = {
            enable = true;
            package = vllm-nix.packages.${pkgs.system}.vllmEnv;
            model = {
              hfId = "RedHatAI/Qwen3.8-27B-INT4";
              servedModelName = "Qwen3.8-27B-INT4";
              maxModelLen = 32768;
            };
            ui.enable = true;
          };
        })
      ];
    };
  };
}
```

The API serves OpenAI-compatible endpoints on `0.0.0.0:8000`; Open WebUI
listens on `127.0.0.1:8080`.

## Presets

The two launch scripts this repo grew out of map to:

### 32k context (default preset)

```nix
services.vllm = {
  enable = true;
  package = vllm-nix.packages.${pkgs.system}.vllmEnv;
  model = {
    hfId = "RedHatAI/Qwen3.8-27B-INT4";
    servedModelName = "Qwen3.8-27B-INT4";
    maxModelLen = 32768;
  };
};
```

### 65k context with FP8 KV cache

```nix
services.vllm = {
  enable = true;
  package = vllm-nix.packages.${pkgs.system}.vllmEnv;
  model = {
    hfId = "RedHatAI/Qwen3.8-27B-INT4";
    servedModelName = "Qwen3.8-27B-INT4-FP8Cache";
    maxModelLen = 65536;
  };
  kvCacheDtype = "fp8";
  attentionBackend = "TRITON_ATTN";
};
```

## Options

| Option | Default | Description |
| --- | --- | --- |
| `enable` | `false` | Enable the vLLM service. |
| `package` | — (required) | Python env providing `bin/vllm`; use `vllm-nix.packages.<system>.vllmEnv`. |
| `host` / `port` | `"0.0.0.0"` / `8000` | API listen address. |
| `openFirewall` | `false` | Open the API port. |
| `model.hfId` | — (required) | Hugging Face model id. |
| `model.servedModelName` | basename of `hfId` | Name exposed on the API. |
| `model.maxModelLen` | `32768` | Context window. |
| `maxNumSeqs` | `8` | `--max-num-seqs`. |
| `maxNumBatchedTokens` | `2048` | `--max-num-batched-tokens`. |
| `gpuMemoryUtilization` | `"0.95"` | `--gpu-memory-utilization`. |
| `tensorParallelSize` | `1` | `--tensor-parallel-size`. |
| `kvCacheDtype` | `null` | e.g. `"fp8"`. |
| `attentionBackend` | `null` | e.g. `"TRITON_ATTN"`. |
| `enableAutoToolChoice` | `true` | `--enable-auto-tool-choice`. |
| `toolCallParser` | `"qwen3_coder"` | `--tool-call-parser` (`null` to omit). |
| `reasoningParser` | `"qwen3"` | `--reasoning-parser` (`null` to omit). |
| `disableMultimodalInputs` | `true` | Disable image/video inputs, data-parallel mm encoder. |
| `extraArgs` | `[]` | Extra `vllm serve` arguments. |
| `environment` | `{}` | Extra env vars for the service. |
| `environmentFile` | `null` | Secrets file, e.g. `HF_TOKEN=...` for gated models. |
| `cudaToolkit.enable` | `true` | CUDA toolkit + gcc on the unit PATH for FlashInfer JIT. |
| `cudaToolkit.package` | `cudaPackages_13.cudatoolkit` | Toolkit used for JIT. |
| `ui.enable` | `false` | Enable Open WebUI wired to this vLLM. |
| `ui.host` / `ui.port` | `"127.0.0.1"` / `8080` | UI listen address. |
| `ui.openFirewall` | `false` | Open the UI port. |
| `ui.webSearch.enable` | `false` | Enable web search (`duckduckgo` by default). |
| `ui.webSearch.engine` | `"duckduckgo"` | Search engine. |
| `ui.webSearch.resultCount` | `5` | Results per query. |
| `ui.webSearch.concurrentRequests` | `10` | Concurrent search requests. |
| `ui.environment` | `{}` | Extra env vars for Open WebUI (wins over module-set ones). |

## Operational notes

- **State** lives in `/var/lib/vllm`; model weights are cached under
  `/var/lib/vllm/huggingface`. First start downloads the weights
  (`TimeoutStartSec` is 60 min for that reason).
- **Gated models**: put `HF_TOKEN=...` in a root-owned file and point
  `services.vllm.environmentFile` at it.
- **Restart behaviour**: `Restart=always`, `RestartSec=5`.
- **Logs**: `journalctl -u vllm -f` (and `journalctl -u open-webui -f`).
- **UI ↔ API wiring**: the module sets `OPENAI_API_BASE_URL` to
  `http://127.0.0.1:<port>/v1` for Open WebUI and disables the Ollama API.

## Updating vLLM

```console
uv lock --upgrade          # or edit pyproject.toml constraints first
git add uv.lock
nix flake check            # rebuilds the env, runs import + module checks
```

The Nix side needs no changes unless new wheels fail `autoPatchelf`
(add them to `gpuWheels` in `nix/vllm-env.nix`) or a wheel gains a new
system-library dependency (add it to that package's `buildInputs`, as done
for numba/oneTBB).

## Troubleshooting

- `Failed to infer device type` — the process cannot see the NVIDIA driver.
  Check `hardware.nvidia` is configured and `/run/opengl-driver/lib` exists;
  the unit's `LD_LIBRARY_PATH` already points there.
- FlashInfer JIT errors mentioning `nvcc`/`gcc` — keep
  `services.vllm.cudaToolkit.enable = true` (default).
- CUDA OOM on the 65k preset — it needs the FP8 KV cache flags shown above;
  otherwise lower `model.maxModelLen` or `gpuMemoryUtilization`.
