{
  lib,
  manylinux_2_28,
  python3,
  rdma-core,
  numactl,
  cudaPackages,
  rocmPackages,

  torch,

  backend,
  version,

  # Unique ID per SDK build variant.
  sdkId,
  src,

  # The ROCm architectures to build for.
  offloadArchs,
}:

let
  isCuda = backend == "cuda";

  stdenv = if isCuda then manylinux_2_28.cudaBackendStdenv else manylinux_2_28.stdenv;

  # CUDA architectures. Upstream's releases only use 9.0 (with PTX).
  gencode = "-gencode=arch=compute_90,code=[sm_90,compute_90]";

  cuda_nvcc = cudaPackages.cuda_nvcc.override { backendStdenv = stdenv; };

  # CLR that uses the manylinux stdenv, so that the device runtime does not
  # depend on a newer glibc/libstdc++ than the rest of the build.
  clr = rocmPackages.clr.override {
    clang = rocmPackages.llvm.clang.override {
      inherit stdenv;
      bintools = rocmPackages.llvm.bintools.override { libc = stdenv.cc.libc; };
      glibc = stdenv.cc.libc;
    };
  };

  # libnuma is bundled with the SDK (it may not be installed on the host), so
  # it must be compatible with manylinux_2_28.
  libnuma = numactl.override { inherit stdenv; };

  nanobind = "${python3.pkgs.nanobind}/${python3.sitePackages}/nanobind";

  hipifyPython = python3.withPackages (_: [ torch ]);

  commonDefs = [
    "-DNB_DOMAIN=${sdkId}"
    "-DPy_LIMITED_API=0x030C0000"
    "-DNB_STABLE_ABI=1"
    "-DDISABLE_AGGRESSIVE_PTX_INSTRS"
  ]
  ++ lib.optionals (!isCuda) [
    "-D__HIP_PLATFORM_AMD__=1"
    "-DUSE_ROCM=1"
    "-DDISABLE_SM90_FEATURES"
    "-DDISABLE_BUILTIN_SHLF_SYNC"
  ];

  includes = [
    "-Iinclude"
    "-I../include"
    "-I${nanobind}/include"
    "-I${nanobind}/ext/robin_map/include"
    "-I${python3}/include/${python3.libPrefix}"
  ]
  ++ (
    if isCuda then
      [
        "-I${lib.getDev cudaPackages.cuda_cudart}/include"
        "-I${lib.getDev cudaPackages.cccl}/include"
      ]
    else
      [ "-I${clr}/include" ]
  );

  deviceCompile =
    if isCuda then
      "nvcc -ccbin $CXX -std=c++17 -O3 ${gencode} -Xcompiler -fPIC -Xcompiler -O3 --expt-relaxed-constexpr"
    else
      "hipcc -x hip -std=c++17 -O3 -fPIC ${
        lib.concatMapStringsSep " " (arch: "--offload-arch=${arch}") offloadArchs
      }";

  deviceLinkFlags =
    if isCuda then
      "-L${lib.getLib cudaPackages.cuda_cudart}/lib -lcudart"
    else
      "-L${clr}/lib -lamdhip64";
in
stdenv.mkDerivation {
  pname = "uccl-ep";
  inherit version src;

  sourceRoot = "${src.name}/ep";

  nativeBuildInputs = if isCuda then [ cuda_nvcc ] else [ clr hipifyPython ];

  buildInputs = [
    (lib.getDev rdma-core)
    rdma-core
    (lib.getDev libnuma)
  ]
  ++ lib.optionals (isCuda && lib.versionAtLeast cudaPackages.cudaMajorMinorVersion "13.0") [
    (lib.getDev cudaPackages.cuda_crt)
  ];

  # hipify writes next to the sources.
  postPatch = lib.optionalString (!isCuda) ''
    chmod -R u+w ..
  '';

  buildPhase = ''
    runHook preBuild

    ${
      if isCuda then
        ''
          ls src/*.cu src/*.cpp src/*.cc > sources.txt
        ''
      else
        ''
          python3 - <<'EOF'
          import glob
          import os

          from torch.utils.hipify.hipify_python import hipify

          here = os.getcwd()
          sources = sorted(
              glob.glob("src/*.cu") + glob.glob("src/*.cpp") + glob.glob("src/*.cc")
          )
          result = hipify(
              project_directory=here,
              output_directory=here,
              header_include_dirs=[os.path.join(here, "include"), os.path.join(here, "..", "include")],
              includes=[os.path.join(here, "*")],
              extra_files=[os.path.abspath(s) for s in sources],
              show_detailed=False,
              is_pytorch_extension=True,
              hipify_extra_files_only=True,
          )
          with open("sources.txt", "w") as out:
              for source in sources:
                  path = os.path.abspath(source)
                  entry = result.get(path)
                  if entry is not None and entry.hipified_path is not None:
                      path = entry.hipified_path
                  print(path, file=out)
          EOF
        ''
    }
    echo ${nanobind}/src/nb_combined.cpp >> sources.txt

    export commonDefs="${lib.concatStringsSep " " commonDefs}"
    export includes="${lib.concatStringsSep " " includes}"

    compile() {
      case $1 in
        *.cu|*.hip)
          ${deviceCompile} $commonDefs $includes -c $1 -o objs/$(basename $1).o
          ;;
        *)
          g++ -std=c++17 -fPIC -fvisibility=hidden -O3 $commonDefs \
            -Wno-deprecated-declarations -Wno-unused-variable -Wno-sign-compare -Wno-reorder \
            -Wno-attributes -Wno-unused-result -Wno-unused-function \
            $includes -c $1 -o objs/$(basename $1).o
          ;;
      esac
    }
    export -f compile

    mkdir objs
    xargs -a sources.txt -P $NIX_BUILD_CORES -I{} bash -c 'compile {}'

    g++ -shared -o ep.abi3.so objs/*.o \
      ${deviceLinkFlags} -L${lib.getLib libnuma}/lib -libverbs -lnuma

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p $out/lib
    cp ep.abi3.so $out/
    cp -L ${lib.getLib libnuma}/lib/libnuma.so.1 $out/lib/
    runHook postInstall
  '';

  # The result is post-processed by the SDK derivation.
  dontFixup = true;
}
