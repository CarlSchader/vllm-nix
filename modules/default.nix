{...}: {
  nixosModules = rec {
    vllm = import ./vllm.nix;
    default = vllm;
  };
}
