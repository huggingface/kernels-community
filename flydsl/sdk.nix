{
  lib,
  stdenv,

  closureInfo,
  jq,
  patchelf-old,
  removeReferencesTo,

  # The native FlyDSL build (Python package with the extensions), and the LLVM it
  # was built against.
  native,
  llvm,
  version,

  # The build variant to produce, e.g. `sdk-rocm-7.0-cpython-3.12-abi3-x86_64-linux`.
  variant,

  # Unique ID for the SDK build variant.
  sdkId,
}:

let
  kernelName = "flydsl";

  buildClosure = closureInfo {
    rootPaths = [
      native
      llvm
    ];
  };
in
stdenv.mkDerivation {
  pname = "${kernelName}-kernel-${variant}";
  inherit version;

  dontUnpack = true;

  # Keep the binaries as shipped, except for the RPATH changes below.
  dontFixup = true;

  nativeBuildInputs = [
    jq
    patchelf-old
    removeReferencesTo
  ];

  allowedReferences = [ ];

  inherit variant sdkId;

  installPhase = ''
        runHook preInstall

        variantDir=$out/$variant
        mkdir -p $variantDir
        cp -r ${native}/python_packages/flydsl/. $variantDir/
        chmod -R u+w $variantDir

        libsDir=$variantDir/_mlir/_mlir_libs
        rm $libsDir/.stubs_generated

        # Needed by libmlir_c_runner_utils.so, not copied by the FlyDSL build.
        cp ${llvm}/lib/libmlir_apfloat_wrappers.so.* $libsDir/
        chmod -R u+w $variantDir

        # The package is the variant directory, so it cannot import itself by its
        # upstream name. Use relative imports instead.
        substituteInPlace $variantDir/utils/smem_allocator.py \
          --replace-fail "from flydsl.expr.gpu import" "from ..expr.gpu import"
        substituteInPlace $variantDir/expr/rocdl/__init__.py \
          --replace-fail "from flydsl._mlir.ir import" "from ..._mlir.ir import"
        substituteInPlace $variantDir/autotune.py \
          --replace-fail "            import flydsl
    " "            import importlib

                flydsl = importlib.import_module(\".\", __package__)
    "
        substituteInPlace $variantDir/compiler/_aot.py \
          --replace-fail "        import flydsl
    " "        import importlib

            flydsl = importlib.import_module(\"..\", __package__)
    "
        substituteInPlace $variantDir/compiler/jit_function.py \
          --replace-fail "    import flydsl

        contents = []" "    import importlib

        flydsl = importlib.import_module(\"..\", __package__)

        contents = []"

        # Hub snapshots are symlinks into `blobs/`, resolving them loses the SDK
        # directory structure. Use absolute paths instead.
        substituteInPlace $variantDir/runtime/_libraries.py \
          --replace-fail "    import flydsl

        root = Path(flydsl.__file__).resolve().parent" "    root = Path(__file__).absolute().parent.parent" \
          --replace-fail "[d.resolve() for d in candidates if d.is_dir()]" "[d.absolute() for d in candidates if d.is_dir()]" \
          --replace-fail "return path.resolve()" "return path.absolute()"
        substituteInPlace $variantDir/compiler/jit_executor.py \
          --replace-fail "Path(__file__).resolve().parent.parent" "Path(__file__).absolute().parent.parent"

        # When used as an SDK, we need to load modules eagerly, so that they are
        # accessible through `get_kernel(...)`.
        cat >> $variantDir/__init__.py <<'EOF'

    from . import compiler, expr, runtime, utils  # noqa: E402,F401
    from .compiler import protocol  # noqa: E402,F401
    from .compiler.backends import rocm  # noqa: E402,F401
    from .expr import meta, typing  # noqa: E402,F401
    from .expr.utils import arith  # noqa: E402,F401  (rebound below)
    from .runtime import device  # noqa: E402,F401
    from ._mlir import ir, passmanager  # noqa: E402,F401
    from ._mlir.extras import types  # noqa: E402,F401
    from ._mlir._mlir_libs import _mlirDialectsFly  # noqa: E402,F401
    from ._mlir.dialects import arith, builtin, fly, gpu, llvm, rocdl  # noqa: E402,F401
    EOF

        # Replace the RPATHs, which reference the build directory and the Nix store,
        # by one that is relative to the library. Also strip the libraries.
        for lib in $libsDir/*.so $libsDir/*.so.*; do
          patchelf --set-rpath '$ORIGIN' $lib
          strip --strip-unneeded $lib
        done

        # Remove any paths from the toolchain, LLVM, Python, etc. that are
        # embedded in the binaries (e.g. in asserts). The SDK cannot refer
        # to anything in the Nix store, or future fetches will fail.
        removeRefs=()
        while read -r path; do
          removeRefs+=(-t "$path")
        done < ${buildClosure}/store-paths
        for lib in $libsDir/*.so $libsDir/*.so.*; do
          remove-references-to "''${removeRefs[@]}" $lib
        done

        # FlyDSL generates device code at runtime. These are the GPUs that its
        # README lists as tested (MI300X, MI350/MI355X, Radeon AI PRO R9700, gfx1250).
        archs='["gfx942", "gfx950", "gfx1201", "gfx1250"]'

        jq -n \
          --arg id "$sdkId" \
          --argjson archs "$archs" \
          '{
            name: "${kernelName}",
            id: $id,
            version: 1,
            license: "Apache-2.0",
            upstream: "https://github.com/ROCm/FlyDSL",
            "python-depends": [],
            "kernel-depends": [],
            backend: { type: "rocm", archs: $archs }
          }' > $variantDir/metadata.json

        runHook postInstall
  '';

  meta = {
    description = "FlyDSL SDK (build variant ${variant})";
    homepage = "https://github.com/ROCm/FlyDSL";
    license = lib.licenses.asl20;
    sourceProvenance = with lib.sourceTypes; [ fromSource ];
    platforms = [ "x86_64-linux" ];
  };
}
