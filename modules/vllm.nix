# NixOS module: services.vllm
#
# Runs `vllm serve` from a pinned python environment (see
# vllm-nix.packages.<system>.vllmEnv) as a hardened, auto-restarting systemd
# service, and optionally fronts it with nixpkgs' services.open-webui.
#
# Requires a CUDA-capable NVIDIA GPU managed by nixpkgs
# (hardware.nvidia / /run/opengl-driver/lib).
{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.vllm;

  driverLib = "${pkgs.addDriverRunpath.driverLink}/lib";

  stateDir = "/var/lib/vllm";

  serveArgs =
    [
      "serve"
      cfg.model.hfId
      "--served-model-name"
      cfg.model.servedModelName
      "--host"
      cfg.host
      "--port"
      (toString cfg.port)
      "--max-model-len"
      (toString cfg.model.maxModelLen)
      "--max-num-seqs"
      (toString cfg.maxNumSeqs)
      "--max-num-batched-tokens"
      (toString cfg.maxNumBatchedTokens)
      "--gpu-memory-utilization"
      cfg.gpuMemoryUtilization
      "--tensor-parallel-size"
      (toString cfg.tensorParallelSize)
    ]
    ++ lib.optionals (cfg.kvCacheDtype != null) ["--kv-cache-dtype" cfg.kvCacheDtype]
    ++ lib.optionals (cfg.attentionBackend != null) ["--attention-backend" cfg.attentionBackend]
    ++ lib.optional cfg.enableAutoToolChoice "--enable-auto-tool-choice"
    ++ lib.optionals (cfg.toolCallParser != null) ["--tool-call-parser" cfg.toolCallParser]
    ++ lib.optionals (cfg.reasoningParser != null) ["--reasoning-parser" cfg.reasoningParser]
    ++ lib.optionals cfg.disableMultimodalInputs [
      "--limit-mm-per-prompt.image"
      "0"
      "--limit-mm-per-prompt.video"
      "0"
      "--mm-encoder-tp-mode"
      "data"
    ]
    ++ cfg.extraArgs;
in {
  options.services.vllm = {
    enable = lib.mkEnableOption "vLLM OpenAI-compatible inference server";

    package = lib.mkOption {
      type = lib.types.package;
      description = ''
        Python environment providing {file}`bin/vllm`. Normally
        `vllm-nix.packages.''${pkgs.system}.vllmEnv`, the environment pinned
        by this flake's uv.lock.
      '';
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "0.0.0.0";
      description = "Address the OpenAI-compatible API listens on.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8000;
      description = "Port the OpenAI-compatible API listens on.";
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Open the API port in the firewall.";
    };

    model = {
      hfId = lib.mkOption {
        type = lib.types.str;
        example = "RedHatAI/Qwen3.8-27B-INT4";
        description = "Hugging Face model id passed to `vllm serve`.";
      };

      servedModelName = lib.mkOption {
        type = lib.types.str;
        default = lib.last (lib.splitString "/" cfg.model.hfId);
        defaultText = lib.literalExpression ''lib.last (lib.splitString "/" cfg.model.hfId)'';
        description = "Model name exposed on the OpenAI API.";
      };

      maxModelLen = lib.mkOption {
        type = lib.types.ints.positive;
        default = 32768;
        example = 65536;
        description = "Context window (`--max-model-len`).";
      };
    };

    maxNumSeqs = lib.mkOption {
      type = lib.types.ints.positive;
      default = 8;
      description = "`--max-num-seqs`.";
    };

    maxNumBatchedTokens = lib.mkOption {
      type = lib.types.ints.positive;
      default = 2048;
      description = "`--max-num-batched-tokens`.";
    };

    gpuMemoryUtilization = lib.mkOption {
      type = lib.types.str;
      default = "0.95";
      description = "`--gpu-memory-utilization` (string to avoid float formatting surprises).";
    };

    tensorParallelSize = lib.mkOption {
      type = lib.types.ints.positive;
      default = 1;
      description = "`--tensor-parallel-size`.";
    };

    kvCacheDtype = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "fp8";
      description = "`--kv-cache-dtype`; set to `\"fp8\"` for the FP8-cache preset.";
    };

    attentionBackend = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "TRITON_ATTN";
      description = "`--attention-backend`; the FP8-cache preset uses `TRITON_ATTN`.";
    };

    enableAutoToolChoice = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Pass `--enable-auto-tool-choice`.";
    };

    toolCallParser = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = "qwen3_coder";
      description = "`--tool-call-parser` (null to omit).";
    };

    reasoningParser = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = "qwen3";
      description = "`--reasoning-parser` (null to omit).";
    };

    disableMultimodalInputs = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Disable image/video inputs and use data-parallel multimodal encoding
        (`--limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0
        --mm-encoder-tp-mode data`), matching the original launch scripts.
      '';
    };

    extraArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [];
      example = ["--enable-prefix-caching"];
      description = "Extra arguments appended to `vllm serve`.";
    };

    environment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = {};
      example = {
        VLLM_LOGGING_LEVEL = "DEBUG";
      };
      description = "Extra environment variables for the vllm service.";
    };

    environmentFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      example = "/run/secrets/vllm.env";
      description = ''
        Environment file for secrets (e.g. `HF_TOKEN=...` for gated models).
      '';
    };

    cudaToolkit = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = ''
          Put a CUDA toolkit and gcc on the service's PATH and set
          {env}`CUDA_HOME`. Required for FlashInfer's runtime JIT
          compilation; disable only if your configuration never touches
          FlashInfer.
        '';
      };

      package = lib.mkOption {
        type = lib.types.package;
        default = pkgs.cudaPackages_13.cudatoolkit;
        defaultText = lib.literalExpression "pkgs.cudaPackages_13.cudatoolkit";
        description = "CUDA toolkit used for runtime JIT compilation.";
      };
    };

    ui = {
      enable = lib.mkEnableOption "Open WebUI frontend wired to this vLLM instance";

      host = lib.mkOption {
        type = lib.types.str;
        default = "127.0.0.1";
        example = "0.0.0.0";
        description = "Address Open WebUI listens on.";
      };

      port = lib.mkOption {
        type = lib.types.port;
        default = 8080;
        description = "Port Open WebUI listens on.";
      };

      openFirewall = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = "Open the UI port in the firewall.";
      };

      webSearch = {
        enable = lib.mkEnableOption "web search in Open WebUI";

        engine = lib.mkOption {
          type = lib.types.str;
          default = "duckduckgo";
          description = "Open WebUI web search engine.";
        };

        resultCount = lib.mkOption {
          type = lib.types.ints.positive;
          default = 5;
          description = "Search results per query.";
        };

        concurrentRequests = lib.mkOption {
          type = lib.types.ints.positive;
          default = 10;
          description = "Concurrent web search requests.";
        };
      };

      environment = lib.mkOption {
        type = lib.types.attrsOf lib.types.str;
        default = {};
        description = ''
          Extra environment variables for Open WebUI (merged last, wins over
          the ones set by this module).
        '';
      };
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.model.hfId != "";
        message = "services.vllm.model.hfId must be set.";
      }
    ];

    systemd.services.vllm = {
      description = "vLLM OpenAI-compatible inference server (${cfg.model.servedModelName})";
      wantedBy = ["multi-user.target"];
      wants = ["network-online.target"];
      after = ["network-online.target"];

      environment =
        {
          HOME = stateDir;
          HF_HOME = "${stateDir}/huggingface";
          XDG_CACHE_HOME = "${stateDir}/.cache";
          # The wheels are patched to find everything except the driver and
          # the system C++/zlib libraries; provide those here (mirrors the
          # dev shell this module was extracted from).
          LD_LIBRARY_PATH =
            lib.makeLibraryPath [pkgs.stdenv.cc.cc.lib pkgs.zlib]
            + ":${driverLib}";
          TRITON_LIBCUDA_PATH = driverLib;
        }
        // lib.optionalAttrs cfg.cudaToolkit.enable {
          CUDA_HOME = "${cfg.cudaToolkit.package}";
          # gcc needs to find the CUDA runtime stubs when FlashInfer JITs.
          LIBRARY_PATH = lib.concatStringsSep ":" [
            "${cfg.cudaToolkit.package}/lib"
            "${cfg.cudaToolkit.package}/lib/stubs"
          ];
        }
        // cfg.environment;

      path = lib.optionals cfg.cudaToolkit.enable [
        pkgs.gcc
        cfg.cudaToolkit.package
      ];

      serviceConfig = {
        ExecStart = "${cfg.package}/bin/vllm ${lib.escapeShellArgs serveArgs}";
        User = "vllm";
        Group = "vllm";
        StateDirectory = "vllm";
        WorkingDirectory = stateDir;

        Restart = "always";
        RestartSec = 5;
        # First start may download tens of GB of weights.
        TimeoutStartSec = "60min";

        EnvironmentFile = lib.optional (cfg.environmentFile != null) cfg.environmentFile;

        # GPU access
        SupplementaryGroups = ["video" "render"];
        DeviceAllow = [
          "char-nvidiactl"
          "char-nvidia-caps"
          "char-nvidia-frontend"
          "char-nvidia-uvm"
          "char-drm"
        ];

        # Hardening (kept compatible with CUDA + JIT compilation)
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ProtectKernelLogs = true;
        ProtectKernelModules = true;
        ProtectKernelTunables = true;
        ProtectControlGroups = true;
        ProtectHostname = true;
        RestrictRealtime = true;
        RestrictSUIDSGID = true;
        LockPersonality = true;
        SystemCallArchitectures = "native";
        RestrictAddressFamilies = ["AF_INET" "AF_INET6" "AF_UNIX" "AF_NETLINK"];
        UMask = "0077";
      };
    };

    users.users.vllm = {
      isSystemUser = true;
      group = "vllm";
      home = stateDir;
      description = "vLLM service user";
    };
    users.groups.vllm = {};

    networking.firewall.allowedTCPPorts = lib.optional cfg.openFirewall cfg.port;

    services.open-webui = lib.mkIf cfg.ui.enable {
      enable = true;
      host = cfg.ui.host;
      port = cfg.ui.port;
      openFirewall = cfg.ui.openFirewall;
      environment =
        {
          # Defaults from the nixpkgs module (replaced when we set this option).
          SCARF_NO_ANALYTICS = "True";
          DO_NOT_TRACK = "True";
          ANONYMIZED_TELEMETRY = "False";
          # Point at the local vLLM OpenAI endpoint.
          OPENAI_API_BASE_URL = "http://127.0.0.1:${toString cfg.port}/v1";
          OPENAI_API_KEY = "EMPTY";
          ENABLE_OLLAMA_API = "False";
        }
        // lib.optionalAttrs cfg.ui.webSearch.enable {
          ENABLE_WEB_SEARCH = "True";
          WEB_SEARCH_ENGINE = cfg.ui.webSearch.engine;
          WEB_SEARCH_RESULT_COUNT = toString cfg.ui.webSearch.resultCount;
          WEB_SEARCH_CONCURRENT_REQUESTS = toString cfg.ui.webSearch.concurrentRequests;
        }
        // cfg.ui.environment;
    };

    # Start the UI after the API it fronts.
    systemd.services.open-webui = lib.mkIf cfg.ui.enable {
      after = ["vllm.service"];
      wants = ["vllm.service"];
    };
  };
}
