{
  lib,
  symlinkJoin,
  cudaPackages,
}:

# The upstream Makefiles expect a single CUDA installation directory (with
# `lib64`) that also contains the stub `libcuda.so`. `crt/*` headers come from
# nvcc.
symlinkJoin {
  name = "cuda-home";
  paths = [
    (lib.getDev cudaPackages.cuda_cudart)
    (lib.getLib cudaPackages.cuda_cudart)
    (lib.getOutput "stubs" cudaPackages.cuda_cudart)
    (lib.getDev cudaPackages.cuda_nvcc)
    (lib.getDev cudaPackages.cccl)
  ]
  # CUDA 13 and later ship `crt/*` in a separate package.
  ++ lib.optionals (lib.versionAtLeast cudaPackages.cudaMajorMinorVersion "13.0") [
    (lib.getDev cudaPackages.cuda_crt)
  ];
  postBuild = ''
    ln -s lib $out/lib64
    ln -s stubs/libcuda.so $out/lib/libcuda.so
  '';
}
