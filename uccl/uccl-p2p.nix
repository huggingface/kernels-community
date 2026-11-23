{
  lib,
  manylinux_2_28,
  callPackage,
  python3,
  rdma-core,
  rocmPackages,

  # `cuda` or `rocm`.
  backend,

  version,

  # Unique ID per SDK build variant.
  sdkId,
  src,
}:

let
  isCuda = backend == "cuda";

  stdenv = if isCuda then manylinux_2_28.cudaBackendStdenv else manylinux_2_28.stdenv;

  cudaHome = callPackage ./cuda-home.nix { };

  hipHome = rocmPackages.clr;

  pythonEnv = python3.withPackages (ps: [ ps.nanobind ]);
in
stdenv.mkDerivation {
  pname = "uccl-p2p";
  inherit version src;

  sourceRoot = "${src.name}/p2p";

  # Give the extension a nanobind domain that is unique per SDK buil.
  postPatch = ''
    substituteInPlace Makefile Makefile.rocm \
      --replace-fail '-DNB_STABLE_ABI=1' '-DNB_STABLE_ABI=1 -DNB_DOMAIN=${sdkId}'
  '';

  nativeBuildInputs = [ pythonEnv ];

  buildInputs = [
    (lib.getDev rdma-core)
  ]
  ++ lib.optionals (!isCuda) [ hipHome ];

  enableParallelBuilding = true;

  makeFlags = [
    "EFA_HOME=${lib.getDev rdma-core}"
    "PYTHON=python3"
    # An extension that uses the stable ABI does not link libpython.
    "PYTHON_LDFLAGS="
    "PYTHON_LIB="
  ]
  ++ (
    if isCuda then
      [ "CUDA_HOME=${cudaHome}" ]
    else
      [
        "-f"
        "Makefile.rocm"
        "HIP_HOME=${hipHome}"
      ]
  );

  buildFlags = [ "all" ];

  preBuild =
    if isCuda then
      ''
        makeFlagsArray+=(
          "LIBS=-L${cudaHome}/lib64 -lcudart -lcuda -lpthread -ldl"
          'LDFLAGS=-L${cudaHome}/lib64 -lcudart -lcuda -lpthread -ldl $(IBV_WRAP_FLAGS)'
        )
      ''
    else
      ''
        # The Makefile adds `$CONDA_LIB_HOME/../include` as an include directory
        # and `$CONDA_LIB_HOME` as a library directory and RPATH.
        mkdir -p $TMPDIR/conda/lib $TMPDIR/conda/include
        makeFlagsArray+=(
          "CONDA_LIB_HOME=$TMPDIR/conda/lib"
          'LDFLAGS=-L${hipHome}/lib -lamdhip64 -lpthread -ldl $(IBV_WRAP_FLAGS)'
        )
      '';

  installPhase = ''
    runHook preInstall
    mkdir -p $out/lib
    cp p2p.abi3.so $out/
    cp libuccl_p2p.so $out/lib/
    runHook postInstall
  '';

  # The result is post-processed by the SDK derivation.
  dontFixup = true;
}
