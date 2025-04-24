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
      renderRef = pkgs.stdenv.mkDerivation {
        name = "render_ref";
        src = ./render/render_ref_x86;
        dontUnpack = true;
        nativeBuildInputs = [ pkgs.autoPatchelfHook ];
        buildInputs = [
          pkgs.cudatoolkit
          pkgs.freeglut
          pkgs.libGL
        ];
        installPhase = ''
          mkdir -p $out/bin
          cp $src $out/bin/render_ref
        '';
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          pkgs.cudatoolkit
          pkgs.freeglut
          pkgs.libGL
          pkgs.libGLU
          pkgs.llvmPackages_17.clang-tools
          pkgs.nixfmt-rfc-style
          pkgs.python3
          pkgs.uv
          cudaScanRef
          renderRef
        ];
        LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib"; # For Python deps.
      };
    };
}
