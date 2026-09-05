# vllm-nix

A pinned [vLLM](https://github.com/vllm-project/vllm) 0.28.0 environment
(uv.lock → Nix via [uv2nix](https://github.com/pyproject-nix/uv2nix), prebuilt
wheels only) plus a NixOS module that runs it as a hardened, auto-restarting
systemd service — optionally fronted by Open WebUI.

```nix
imports = [ vllm-nix.nixosModules.vllm ];

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
```

See [docs/usage.md](docs/usage.md) for the full option reference, the two
Qwen presets (32k / 65k+FP8 KV cache), update procedure, and troubleshooting.

## Flake outputs

- `packages.x86_64-linux.vllmEnv` — the pinned python env (`bin/vllm`, `bin/hf`, `bin/python`)
- `nixosModules.vllm` (also `default`) — the `services.vllm` module
- `checks.x86_64-linux.{vllmEnvImport,vllmModuleEval}` — env import + module eval smoke tests
- `devShells.x86_64-linux.default` — uv/CUDA dev shell for working on the lock file

## Development

```console
nix build .#vllmEnv     # build the environment
nix flake check         # import + module-eval checks
```
