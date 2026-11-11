{
  lib,
  stdenv,
  writeText,

  hash-kernel-hook,
  jq,
  patchelf-old,
  remove-bytecode-hook,


  get-kernel-check,
  generate-symbols-hook,

  python3,
  torch,
  overrideTorch,

  # The native FlyDSL build (Python package with the extensions), and the LLVM it
  # was built against.
  native,
  llvm,
  version,

  # The build variant to produce, e.g. `sdk-rocm-7.0-abi3-x86_64-linux`.
  variant,

  # The (short) Git revision of the kernel, with dashes replaced by underscores
  # (e.g. `7a4e5a7` or `7a4e5a7_dirty`). Used to make the kernel id unique.
  rev,
}:

let
  kernelName = "flydsl";

  kernelId = "_${kernelName}_${lib.replaceStrings [ "-" "." ] [ "_" "_" ] variant}_${rev}";

  # Torch is not used by FlyDSL, it is only needed in the environment of the
  # load check, which provides the ROCm libraries.
  dependencies = [ torch ];
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
    # Order matters: remove bytecode, then hash.
    remove-bytecode-hook
    hash-kernel-hook
    (get-kernel-check.override {
      python3 = python3.withPackages (_: dependencies);
      kernels = overrideTorch python3.pkgs.kernels;
    })
    (generate-symbols-hook.override {
      python3 = python3.withPackages (_: dependencies);
      kernels = overrideTorch python3.pkgs.kernels;
    })
  ];

  # Required by the get-kernel-check and generate-symbols hooks.
  moduleName = kernelName;
  kernelDeps = writeText "kernel-deps" "[]";
  inherit variant kernelId;

  doInstallCheck = true;

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
    substituteInPlace $variantDir/runtime/_libraries.py \
      --replace-fail "    import flydsl

    root = Path(flydsl.__file__).resolve().parent" "    root = Path(__file__).resolve().parent.parent"

    # Replace the RPATHs, which reference the build directory and the Nix store,
    # by one that is relative to the library. Also strip the libraries.
    for lib in $libsDir/*.so $libsDir/*.so.*; do
      patchelf --set-rpath '$ORIGIN' $lib
      strip --strip-unneeded $lib
    done

    # FlyDSL generates device code at runtime. These are the GPUs that its
    # README lists as tested (MI300X, MI350/MI355X, Radeon AI PRO R9700, gfx1250).
    archs='["gfx942", "gfx950", "gfx1201", "gfx1250"]'

    jq -n \
      --arg id "$kernelId" \
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
