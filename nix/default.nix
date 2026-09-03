{flake-utils, ...} @ inputs:
flake-utils.lib.meld inputs [
  ./shells.nix
]
