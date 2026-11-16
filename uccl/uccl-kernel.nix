{
  lib,
  stdenv,

  jq,
  patchelf-old,
  unzip,

  python3,

  # The upstream wheel (a fetchurl result) for this variant's backend.
  wheel,
  version,

  # The build variant to produce, e.g. `sdk-cuda-13.0-cpython-3.12-abi3-x86_64-linux`.
  variant,

  # The backend (`cuda` or `rocm`) and, for CUDA, the major CUDA version.
  backend,
  cudaMajorVersion,

  # The (short) Git revision of the kernel, with dashes replaced by underscores
  # (e.g. `7a4e5a7` or `7a4e5a7_dirty`). Used to make the kernel id unique.
  rev,

  # The CUDA capabilities to declare in the metadata of CUDA variants.
  cudaArchs,
}:

let
  cudaSupport = backend == "cuda";

  kernelName = "uccl";

  kernelId = "_${kernelName}_${lib.replaceStrings [ "-" "." ] [ "_" "_" ] variant}_${rev}";

  intervaltree = python3.pkgs.intervaltree;
  sortedcontainers = python3.pkgs.sortedcontainers;

  # The CUDA capabilities are hardcoded (see `cudaArchs`). The ROCm
  # architectures are the ones that the wheel has device code for.
  archsCommand =
    if cudaSupport then
      ''
        archs='${builtins.toJSON cudaArchs}'
      ''
    else
      ''
        archs=$(
          grep -a -h -o -E 'amdgcn-amd-amdhsa--gfx[0-9a-f]+' $variantDir/*.so $variantDir/lib/*.so \
            | sed 's/.*--//' \
            | sort -u \
            | jq -R . | jq -s -c .
        )
      '';
in
stdenv.mkDerivation {
  pname = "${kernelName}-sdk-${variant}";
  inherit version;

  dontUnpack = true;

  # Keep the binaries as shipped, except for the RPATH changes below.
  dontFixup = true;

  nativeBuildInputs = [
    jq
    patchelf-old
    unzip
  ];

  inherit variant kernelId;

  installPhase = ''
    runHook preInstall

    unzip -q ${wheel} -d wheel

    variantDir=$out/$variant
    mkdir -p $variantDir
    cp -r wheel/uccl/. $variantDir/
    rm $variantDir/_platform_tag_stub.abi3.so

    # Use libnuma that is bundled in the wheel.
    mkdir $variantDir/uccl.libs
    cp wheel/uccl.libs/libnuma-*.so* $variantDir/uccl.libs/

    # Vendor intervaltree and its dependency sortedcontainers.
    sitePackages=${python3.sitePackages}
    mkdir $variantDir/_vendor
    touch $variantDir/_vendor/__init__.py
    cp -r ${intervaltree}/$sitePackages/intervaltree $variantDir/_vendor/
    cp -r ${sortedcontainers}/$sitePackages/sortedcontainers $variantDir/_vendor/
    chmod -R u+w $variantDir
    cp ${intervaltree}/$sitePackages/intervaltree-*.dist-info/licenses/LICENSE.txt \
      $variantDir/_vendor/intervaltree/
    cp ${sortedcontainers}/$sitePackages/sortedcontainers-*.dist-info/licenses/LICENSE \
      $variantDir/_vendor/sortedcontainers/
    find $variantDir/_vendor -type d -name __pycache__ -exec rm -rf {} +
    chmod -R u+w $out

    substituteInPlace $variantDir/utils.py \
      --replace-fail "from intervaltree import" "from ._vendor.intervaltree import"
    # Only use relative imports, drop the fallback to top-level modules.
    substituteInPlace $variantDir/collective.py \
      --replace-fail $'try:\n    from . import p2p\n    from . import utils\nexcept ImportError:\n    import p2p\n    import utils\n' $'from . import p2p\nfrom . import utils\n'
    substituteInPlace $variantDir/_vendor/intervaltree/intervaltree.py \
      --replace-fail "from sortedcontainers import" "from ..sortedcontainers import"

    ${lib.optionalString cudaSupport ''
      # Use the libcudart that Torch loads instead of a bundled copy.
      for lib in $variantDir/*.so $variantDir/lib/*.so; do
        for needed in $(patchelf --print-needed $lib); do
          case $needed in
            libcudart-*)
              patchelf --replace-needed $needed libcudart.so.${cudaMajorVersion} $lib
              ;;
          esac
        done
      done
    ''}

    # Replace impure RPATHs (e.g. /usr/local/cuda) by paths relative to the kernel.
    for lib in $variantDir/*.so; do
      patchelf --set-rpath '$ORIGIN/uccl.libs' $lib
    done
    for lib in $variantDir/lib/*.so; do
      patchelf --set-rpath '$ORIGIN/../uccl.libs' $lib
    done

    ${archsCommand}
    if [ "$archs" = "[]" ]; then
      echo "Could not determine the device architectures of the wheel"
      exit 1
    fi


    jq -n \
      --arg id "$kernelId" \
      --arg backend "${backend}" \
      --argjson archs "$archs" \
      '{
        name: "${kernelName}",
        id: $id,
        version: 1,
        license: "Apache-2.0",
        upstream: "https://github.com/uccl-project/uccl",
        "python-depends": [],
        "kernel-depends": [],
        backend: { type: $backend, archs: $archs }
      }' > $variantDir/metadata.json

    runHook postInstall
  '';

  meta = {
    description = "UCCL repackaged as an SDK (build variant ${variant})";
    homepage = "https://github.com/uccl-project/uccl";
    license = lib.licenses.asl20;
    sourceProvenance = with lib.sourceTypes; [ binaryNativeCode ];
    platforms = [ "x86_64-linux" ];
  };
}
