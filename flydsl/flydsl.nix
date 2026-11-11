{
  lib,
  manylinux_2_28,
  cmake,
  ninja,
  python3,
  rocmPackages,

  llvm,
  flydslSrc,
}:

let
  # CLR that uses the manylinux stdenv, so that the runtime library does not
  # depend on a newer glibc/libstdc++ than the rest of the build.
  clr = rocmPackages.clr.override {
    clang = rocmPackages.llvm.clang.override {
      stdenv = manylinux_2_28.stdenv;
      bintools = rocmPackages.llvm.bintools.override { libc = manylinux_2_28.stdenv.cc.libc; };
      glibc = manylinux_2_28.stdenv.cc.libc;
    };
  };

  pythonEnv = python3.withPackages (ps: [
    ps.nanobind
    ps.numpy
    ps.pybind11
  ]);
in
manylinux_2_28.stdenv.mkDerivation {
  pname = "flydsl-native";
  version = "0.5.0";

  src = flydslSrc;

  nativeBuildInputs = [
    cmake
    ninja
    pythonEnv
  ];

  # The source is copied from the Nix store, but the build copies files from it
  # and modifies the copies.
  postPatch = ''
    chmod -R u+w .
    # The runtime libraries are copied from the (read-only) LLVM install,
    # and then have their RPATH changed.
    substituteInPlace python/mlir_flydsl/CMakeLists.txt \
      --replace-fail 'COMMAND bash -c "patchelf' 'COMMAND bash -c "chmod u+w ''${_MLIR_LIBS_DIR}/lib*; patchelf'
    # Stub generation imports the extension modules, which are built with the
    # manylinux toolchain and cannot be loaded by the Python of the build
    # environment (it also needs /bin/bash).
    substituteInPlace python/mlir_flydsl/CMakeLists.txt \
      --replace-fail 'COMMAND /bin/bash -c "\' 'COMMAND true "\'
  '';

  buildInputs = [ clr ];

  # The RPATHs reference the build directory. They are replaced when the kernel is
  # assembled.
  dontFixup = true;

  # `find_package(hip)` uses ROCM_PATH.
  env.ROCM_PATH = "${clr}";

  cmakeFlags = [
    (lib.cmakeFeature "MLIR_DIR" "${llvm}/lib/cmake/mlir")
    (lib.cmakeFeature "Python3_EXECUTABLE" "${pythonEnv}/bin/python")
    (lib.cmakeFeature "Python_EXECUTABLE" "${pythonEnv}/bin/python")
    (lib.cmakeFeature "HIP_PLATFORM" "amd")
    # Build for ABI3 (>= 3.12) due to nanobind. LLVM build is not exported,
    # so it has to be repeated here.
    (lib.cmakeBool "MLIR_ENABLE_PYTHON_STABLE_ABI" true)
    (lib.cmakeFeature "CMAKE_BUILD_TYPE" "Release")
  ];

  # The Python package that `setup.py` would embed is not installed by
  # CMake, take it from the build tree.
  installPhase = ''
    runHook preInstall
    mkdir -p $out
    # Dereference symlinks. They point to the build directory and to the MLIR
    # Python sources in the LLVM install.
    cp -rL python_packages $out/
    rm -f $out/python_packages/flydsl/_mlir/_mlir_libs/libFlyPythonCAPI.so
    runHook postInstall
  '';
}
