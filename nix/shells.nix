{
  nixpkgs,
  flake-utils,
  ...
}:
flake-utils.lib.eachDefaultSystem (system: let
  pkgs = import nixpkgs {
    inherit system;
    config = {
      allowUnfree = true;
      # cudaSupport = true;
    };
  };
in {
  devShells.default = pkgs.mkShell {
    env = {
      LD_LIBRARY_PATH =
        pkgs.lib.makeLibraryPath (with pkgs; [
          stdenv.cc.cc.lib
          zlib
        ])
        + ":${pkgs.addDriverRunpath.driverLink}/lib";
      # This is so gcc can find nvcc for FlashInfer
      LIBRARY_PATH = pkgs.lib.concatStringsSep ":" [
        "${pkgs.cudaPackages_13.cudatoolkit}/lib"
        "${pkgs.cudaPackages_13.cudatoolkit}/lib/stubs"
      ];
      TRITON_LIBCUDA_PATH = "${pkgs.addDriverRunpath.driverLink}/lib";
      CUDA_HOME = "${pkgs.cudaPackages_13.cudatoolkit}";
    };
    buildInputs = with pkgs; [
      # vllm
      uv
      gcc
      cudaPackages_13.cudatoolkit
    ];
  };
})
