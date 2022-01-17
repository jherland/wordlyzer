{
  pkgs ? import (builtins.fetchGit {
    url = "https://github.com/NixOS/nixpkgs/";
    ref = "nixos-21.05";
  }) {}
}:

pkgs.mkShell {
  name = "wordlyzer";
  buildInputs = with pkgs; [
    python310
    python310Packages.venvShellHook
  ];
  venvDir = "./.venv";
  postShellHook = ''
    unset SOURCE_DATE_EPOCH
    pip install --upgrade pip
    pip install -r requirements.txt
  '';
}
