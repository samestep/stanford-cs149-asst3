{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };
  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      cudaScanRef = pkgs.stdenv.mkDerivation {
        name = "cudaScan_ref";
        src = ./scan/cudaScan_ref_x86;
        dontUnpack = true;
        nativeBuildInputs = [ pkgs.autoPatchelfHook ];
        buildInputs = [ pkgs.cudatoolkit ];
        installPhase = ''
          mkdir -p $out/bin
          cp $src $out/bin/cudaScan_ref
        '';
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          pkgs.cudatoolkit
          pkgs.llvmPackages_17.clang-tools
          pkgs.nixfmt-rfc-style
          pkgs.python3
          cudaScanRef
        ];
      };
    };
}
