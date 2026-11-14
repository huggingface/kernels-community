{
  lib,
  manylinux_2_28,
  fetchFromGitHub,
  cmake,
  ninja,
  python3,

  flydslSrc,
}:

let
  # Keep in sync with `thirdparty/llvm-build-info.json` in FlyDSL.
  llvmRev = "941a04e69ee8fe4c7a162b2f1e215aa8df867534";

  # `upstream.required_patches` from `thirdparty/llvm-build-info.json`, in order.
  patchNames = [
    "rocdl-lld-argv0.patch"
    "rocdl-waves-per-eu-range.patch"
    "xnack-any-codegen-override.patch"
    "amdgpu-initial-unclaused-vmem-sequence.patch"
    "amdgpu-improve-unclaused-vmem-sequence.patch"
    "codegen-avoid-regpressure-underflow.patch"
    "gfx1250-strict.patch"
    "schedule-bank-and-gpr.patch"
    "xnack-xcnt-replay-codegen.patch"
  ];

  pythonEnv = python3.withPackages (ps: [
    ps.nanobind
    ps.numpy
    ps.pybind11
    ps.pyyaml
  ]);
in
manylinux_2_28.stdenv.mkDerivation {
  pname = "llvm-mlir-flydsl";
  version = "git-${builtins.substring 0 8 llvmRev}";

  src = fetchFromGitHub {
    owner = "llvm";
    repo = "llvm-project";
    rev = llvmRev;
    hash = "sha256-se7P3JD7B8gs5qUz/lylLyRziHWb+0wnr4Mc2hTGydc=";
  };

  patches = map (name: "${flydslSrc}/thirdparty/llvm-extensions/${name}") patchNames;

  sourceRoot = "source";
  cmakeDir = "../llvm";

  # ld in manylinux 2.28 does not support `-z undefs`, which MLIR uses to negate
  # the `-z defs` that LLVM passes for all shared libraries. Disable both.
  postPatch = ''
    substituteInPlace llvm/cmake/modules/HandleLLVMOptions.cmake \
      --replace-fail ' -Wl,-z,defs")' '")'
    substituteInPlace mlir/cmake/modules/AddMLIRPython.cmake \
      --replace-fail '"LINKER:-z,undefs"' '"LINKER:-z,nodelete"'
  '';

  nativeBuildInputs = [
    cmake
    ninja
    pythonEnv
  ];

  # Mirrors the `amd-minimal` profile of FlyDSL's `scripts/build_llvm.sh`,
  # except that optional compression libraries are disabled, so that we do not
  # have to ship them.
  cmakeFlags = [
    (lib.cmakeFeature "LLVM_ENABLE_PROJECTS" "mlir")
    (lib.cmakeFeature "LLVM_TARGETS_TO_BUILD" "X86;AMDGPU")
    (lib.cmakeFeature "LLVM_ENABLE_RUNTIMES" "")
    (lib.cmakeFeature "CMAKE_BUILD_TYPE" "Release")
    (lib.cmakeFeature "CMAKE_CXX_STANDARD" "17")
    (lib.cmakeBool "LLVM_ENABLE_ASSERTIONS" true)
    (lib.cmakeBool "LLVM_INSTALL_UTILS" true)
    (lib.cmakeBool "MLIR_ENABLE_BINDINGS_PYTHON" true)
    (lib.cmakeFeature "MLIR_BINDINGS_PYTHON_NB_DOMAIN" "mlir")
    (lib.cmakeFeature "Python3_EXECUTABLE" "${pythonEnv}/bin/python")
    (lib.cmakeFeature "Python_EXECUTABLE" "${pythonEnv}/bin/python")
    (lib.cmakeBool "BUILD_SHARED_LIBS" false)
    (lib.cmakeBool "LLVM_BUILD_LLVM_DYLIB" false)
    (lib.cmakeBool "LLVM_LINK_LLVM_DYLIB" false)
    (lib.cmakeBool "MLIR_INCLUDE_TESTS" false)
    (lib.cmakeBool "LLVM_INCLUDE_TESTS" false)
    (lib.cmakeBool "LLVM_INCLUDE_BENCHMARKS" false)
    (lib.cmakeBool "LLVM_INCLUDE_EXAMPLES" false)
    (lib.cmakeBool "LLVM_ENABLE_ZLIB" false)
    (lib.cmakeBool "LLVM_ENABLE_ZSTD" false)
    (lib.cmakeBool "LLVM_ENABLE_LIBXML2" false)
    (lib.cmakeBool "LLVM_ENABLE_TERMINFO" false)
    (lib.cmakeFeature "HIP_PLATFORM" "amd")
    # Stub generation imports the extension modules that were just built with
    # the manylinux toolchain, which cannot be loaded by the Python in the build
    # environment.
    (lib.cmakeBool "MLIR_PYTHON_STUBGEN_ENABLED" false)
    # Build with abi3 support, requires >= 3.12 due to nanobind.
    (lib.cmakeBool "MLIR_ENABLE_PYTHON_STABLE_ABI" true)
  ];

  # The RPATH of the installed libraries/modules is relative.
  preConfigure = ''
    cmakeFlagsArray+=("-DCMAKE_INSTALL_RPATH=\$ORIGIN")
  '';

  meta = {
    description = "LLVM/MLIR as pinned and patched by FlyDSL, with Python bindings";
    homepage = "https://github.com/llvm/llvm-project";
    license = lib.licenses.ncsa;
    platforms = [ "x86_64-linux" ];
  };
}
