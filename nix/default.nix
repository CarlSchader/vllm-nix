{flake-utils, ...} @ inputs:
flake-utils.lib.meld inputs [
  ./models.nix
  ./shells.nix
]
