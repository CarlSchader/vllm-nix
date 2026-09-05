{
  nixpkgs,
  flake-utils,
  pyproject-nix,
  uv2nix,
  pyproject-build-systems,
  ...
}:
flake-utils.lib.eachSystem ["x86_64-linux"] (system: let
  inherit (nixpkgs) lib;

  pkgs = import nixpkgs {
    inherit system;
    config.allowUnfree = true;
  };

  # Read the uv.lock pinning vllm 0.28.0 (and friends) from the workspace root.
  workspace = uv2nix.lib.workspace.loadWorkspace {
    workspaceRoot = ../.;
  };

  # One derivation per pinned wheel, using the hashes already in uv.lock.
  # "wheel" preference enforces the no-source-builds promise: a dependency
  # that has no prebuilt wheel fails the build loudly instead of compiling.
  overlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };

  # Extra fixes for the GPU wheels. The manylinux wheels in this lock are
  # self-contained except for libraries that are intentionally absent from
  # the venv at patch time:
  #   - libcuda.so.1 / libnvidia-ml.so.1: provided by the NVIDIA driver at
  #     runtime (/run/opengl-driver/lib on NixOS);
  #   - libtorch*.so / libc10*.so and the nvidia-* CUDA libraries: live in
  #     sibling site-packages and are loaded by python (`import torch`)
  #     before any extension that needs them.
  # autoPatchelf must not fail the build over these, so they are ignored for
  # every GPU wheel in the lock.
  gpuWheels = [
    "apache-tvm-ffi"
    "cuda-bindings"
    "cuda-core"
    "cuda-python"
    "cuda-tile"
    "cuda-toolkit"
    "fastsafetensors"
    "flashinfer-python"
    "humming-kernels"
    "nccl4py"
    "nvidia-cublas"
    "nvidia-cuda-cccl"
    "nvidia-cuda-crt"
    "nvidia-cuda-cupti"
    "nvidia-cuda-nvcc"
    "nvidia-cuda-nvdisasm"
    "nvidia-cuda-nvrtc"
    "nvidia-cuda-runtime"
    "nvidia-cudnn-cu13"
    "nvidia-cudnn-frontend"
    "nvidia-cufft"
    "nvidia-cufile"
    "nvidia-curand"
    "nvidia-cusolver"
    "nvidia-cusparse"
    "nvidia-cusparselt-cu13"
    "nvidia-cutlass-dsl"
    "nvidia-cutlass-dsl-libs-base"
    "nvidia-cutlass-dsl-libs-core"
    "nvidia-cutlass-dsl-libs-cu12"
    "nvidia-cutlass-dsl-libs-cu13"
    "nvidia-nccl-cu13"
    "nvidia-nvjitlink"
    "nvidia-nvshmem-cu13"
    "nvidia-nvtx"
    "nvidia-nvvm"
    "nvtx"
    "pynvvideocodec"
    "quack-kernels"
    "tilelang"
    "tokenspeed-mla"
    "tokenspeed-triton"
    "torch"
    "torch-c-dlpack-ext"
    "torchaudio"
    "torchcodec"
    "torchvision"
    "triton"
    "vllm"
    "xgrammar"
  ];

  fixups = final: prev: let
    relaxed = lib.genAttrs (builtins.filter (n: builtins.hasAttr n prev) gpuWheels) (
      name:
        prev.${name}.overrideAttrs (old: {
          autoPatchelfIgnoreMissingDeps = true;
        })
    );

    # Both wheels accidentally ship a top-level site-packages/build_backend.py
    # (packaging junk), which collides during venv assembly.
    dropBuildBackend = drv:
      drv.overrideAttrs (old: {
        postInstall =
          (old.postInstall or "")
          + ''
            rm -f $out/lib/python*/site-packages/build_backend.py
          '';
      });
  in
    relaxed
    // {
      flashinfer-python = dropBuildBackend relaxed.flashinfer-python;
      torch-c-dlpack-ext = dropBuildBackend relaxed.torch-c-dlpack-ext;

      # numba's wheel links against oneTBB (libtbb.so.12).
      numba = prev.numba.overrideAttrs (old: {
        buildInputs = (old.buildInputs or []) ++ [pkgs.tbb];
      });
    };

  pythonSet =
    (pkgs.callPackage pyproject-nix.build.packages {
      python = pkgs.python312;
    }).overrideScope (lib.composeManyExtensions [
      pyproject-build-systems.overlays.default
      overlay
      fixups
    ]);

  vllmEnv = pythonSet.mkVirtualEnv "vllm-env" workspace.deps.default;

  # Pure-eval smoke test of the NixOS module: a full nixosSystem evaluation
  # with the module enabled (both presets' knobs exercised) that materialises
  # the rendered vllm ExecStart. No VM, no GPU.
  moduleEval = nixpkgs.lib.nixosSystem {
    inherit system;
    modules = [
      ../modules/vllm.nix
      {
        nixpkgs.hostPlatform = system;
        nixpkgs.config.allowUnfree = true;
        boot.loader.grub.enable = false;
        fileSystems."/" = {
          device = "none";
          fsType = "tmpfs";
        };
        system.stateVersion = "25.05";

        services.vllm = {
          enable = true;
          package = vllmEnv;
          openFirewall = true;
          model = {
            hfId = "RedHatAI/Qwen3.8-27B-INT4";
            servedModelName = "Qwen3.8-27B-INT4-FP8Cache";
            maxModelLen = 65536;
          };
          kvCacheDtype = "fp8";
          attentionBackend = "TRITON_ATTN";
          ui = {
            enable = true;
            webSearch.enable = true;
          };
        };
      }
    ];
  };
in {
  # Reproducible python env for `vllm serve` built from the pinned wheels.
  # Exposes $out/bin/vllm, $out/bin/hf, $out/bin/python.
  packages.vllmEnv = vllmEnv;

  checks = {
    # Prove the env is a coherent, importable python environment (no GPU).
    vllmEnvImport = pkgs.stdenv.mkDerivation {
      name = "vllm-env-import-check";
      nativeBuildInputs = [vllmEnv];
      dontUnpack = true;
      dontConfigure = true;
      buildPhase = ''
        runHook preBuild
        # `vllm --version` initialises platform/device detection and needs a
        # visible GPU driver, so assert on the python metadata instead: this
        # sandbox has no GPU.
        python - <<'EOF'
        import torch
        import vllm
        print("vllm", vllm.__version__)
        print("torch", torch.__version__)
        assert vllm.__version__ == "0.28.0", vllm.__version__
        EOF
        hf --help > /dev/null
        runHook postBuild
      '';
      installPhase = ''
        touch $out
      '';
    };

    # Evaluate the NixOS module against a minimal nixosSystem and pin down
    # the rendered unit. Fails at eval time if the module or its interplay
    # with services.open-webui regresses.
    vllmModuleEval = pkgs.writeText "vllm-module-eval" (builtins.toJSON {
      execStart = moduleEval.config.systemd.services.vllm.serviceConfig.ExecStart;
      openWebuiEnabled = moduleEval.config.services.open-webui.enable;
      firewallPorts = moduleEval.config.networking.firewall.allowedTCPPorts;
    });
  };
})
