{
  lib,
  manylinux_2_28,
  runCommand,
  fetchFromGitHub,
  callPackage,
  rdma-core,
  rocmPackages,

  # `cuda` or `rocm`.
  backend,

  version,
  src,
}:

let
  isCuda = backend == "cuda";

  stdenv = if isCuda then manylinux_2_28.cudaBackendStdenv else manylinux_2_28.stdenv;

  cudaHome = callPackage ./cuda-home.nix { };

  hipHome = rocmPackages.clr;

  # The plugin is built against the NCCL/RCCL net plugin headers of the version
  # that upstream pins (submodules `thirdparty/nccl` and `thirdparty/rccl`).
  # Both are NCCL 2.23.4.
  ncclSrc =
    if isCuda then
      fetchFromGitHub {
        owner = "NVIDIA";
        repo = "nccl";
        rev = "2ea4ee94bfb04c886c79ccae60ac9961000fdee2";
        hash = "sha256-DlMxlLO2F079fBkhORNPVN/ASYiVIRfLJw7bDoiClHw=";
      }
    else
      fetchFromGitHub {
        owner = "ROCm";
        repo = "rccl";
        rev = "532f54c2444501b3655e65fbce6d00d4bfc19c0b";
        hash = "sha256-eQBfoGfyCzfSDlHdxgivmOb52zOMcegctdtaRUbRzkM=";
      };

  # NCCL/RCCL generate `nccl.h` from a template during their build. The
  # template variables are named differently.
  versionVars =
    if isCuda then
      {
        major = "nccl:Major";
        minor = "nccl:Minor";
        patch = "nccl:Patch";
        suffix = "nccl:Suffix";
        version = "nccl:Version";
      }
    else
      {
        major = "NCCL_MAJOR";
        minor = "NCCL_MINOR";
        patch = "NCCL_PATCH";
        suffix = "NCCL_SUFFIX";
        version = "NCCL_VERSION";
      };

  ncclHeaders = runCommand "nccl-headers" { } ''
    mkdir -p $out/include
    cp ${ncclSrc}/src/include/{nccl_net,nccl_common,net_device}.h $out/include/
    sed \
      -e "s/[\$]{${versionVars.major}}/2/" \
      -e "s/[\$]{${versionVars.minor}}/23/" \
      -e "s/[\$]{${versionVars.patch}}/4/" \
      -e "s/[\$]{${versionVars.suffix}}//" \
      -e "s/[\$]{${versionVars.version}}/22304/" \
      ${ncclSrc}/src/nccl.h.in > $out/include/nccl.h
  '';

  pluginName = if isCuda then "libnccl-net-uccl.so" else "librccl-net-uccl.so";
in
stdenv.mkDerivation {
  pname = "uccl-nccl-plugin";
  inherit version src;

  sourceRoot = "${src.name}/collective/rdma";

  buildInputs = [
    rdma-core
    (lib.getDev rdma-core)
  ]
  ++ lib.optionals (!isCuda) [ hipHome ];

  enableParallelBuilding = true;

  makeFlags =
    if isCuda then
      [ "CUDA_HOME=${cudaHome}" ]
    else
      [
        "-f"
        "Makefile.rocm"
        "HIP_HOME=${hipHome}"
      ];

  buildFlags = [ pluginName ];

  preBuild =
    if isCuda then
      ''
        makeFlagsArray+=(
          "LIBS_SHARED=-libverbs"
          "NCCL_INC=-I${ncclHeaders}/include -I${cudaHome}/include"
        )
      ''
    else
      ''
        # The Makefile adds `$CONDA_LIB_HOME/../include` as an include directory
        # and `$CONDA_LIB_HOME` as a library directory.
        mkdir -p $TMPDIR/conda/lib $TMPDIR/conda/include
        makeFlagsArray+=(
          "CONDA_LIB_HOME=$TMPDIR/conda/lib"
          "LIBS_SHARED=-libverbs -lpthread -ldl"
          "NCCL_INC=-I${ncclHeaders}/include -I${hipHome}/include"
        )
      '';

  installPhase = ''
    runHook preInstall
    mkdir -p $out/lib
    cp ${pluginName} $out/lib/
    runHook postInstall
  '';

  # The result is post-processed by the SDK derivation.
  dontFixup = true;
}
