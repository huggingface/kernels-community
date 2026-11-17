{
  lib,
  stdenv,

  jq,
  patchelf-old,

  python3,

  # Native components from the same source tree.
  p2p,
  ncclPlugin,
  ep,

  # The upstream source tree.
  src,
  version,

  # The build variant to produce, e.g. `sdk-cuda-13.0-cpython-3.12-abi3-x86_64-linux`.
  variant,

  # The backend (`cuda` or `rocm`).
  backend,

  # Unique ID for the SDK build variant.
  sdkId,

  # The capabilities/architectures to declare in the metadata.
  archs,
}:

let
  kernelName = "uccl";

  intervaltree = python3.pkgs.intervaltree;
  sortedcontainers = python3.pkgs.sortedcontainers;
in
stdenv.mkDerivation {
  pname = "${kernelName}-sdk-${variant}";
  inherit version;

  dontUnpack = true;

  # Keep the binaries as built, except for the RPATH changes below.
  dontFixup = true;

  nativeBuildInputs = [
    jq
    patchelf-old
  ];

  inherit variant sdkId;

  installPhase = ''
    runHook preInstall

    variantDir=$out/$variant
    mkdir -p $variantDir/lib $variantDir/uccl.libs

    # Python files. The upstream packaging copies `collective.py` and
    # `utils.py` into the package.
    cp --no-preserve=mode ${src}/uccl/__init__.py ${src}/p2p/collective.py ${src}/p2p/utils.py $variantDir/

    cp --no-preserve=mode ${p2p}/p2p.abi3.so $variantDir/
    cp --no-preserve=mode ${p2p}/lib/libuccl_p2p.so $variantDir/lib/
    cp --no-preserve=mode ${ep}/ep.abi3.so $variantDir/
    cp --no-preserve=mode ${ncclPlugin}/lib/*.so $variantDir/lib/
    # libnuma is bundled, the host may not have it.
    cp --no-preserve=mode ${ep}/lib/libnuma.so.1 $variantDir/uccl.libs/

    # Vendor intervaltree and its dependency sortedcontainers.
    sitePackages=${python3.sitePackages}
    mkdir $variantDir/_vendor
    touch $variantDir/_vendor/__init__.py
    cp --no-preserve=mode -r ${intervaltree}/$sitePackages/intervaltree $variantDir/_vendor/
    cp --no-preserve=mode -r ${sortedcontainers}/$sitePackages/sortedcontainers $variantDir/_vendor/
    cp --no-preserve=mode ${intervaltree}/$sitePackages/intervaltree-*.dist-info/licenses/LICENSE.txt \
      $variantDir/_vendor/intervaltree/
    cp --no-preserve=mode ${sortedcontainers}/$sitePackages/sortedcontainers-*.dist-info/licenses/LICENSE \
      $variantDir/_vendor/sortedcontainers/
    find $variantDir/_vendor -type d -name __pycache__ -exec rm -rf {} +

    substituteInPlace $variantDir/utils.py \
      --replace-fail "from intervaltree import" "from ._vendor.intervaltree import"
    # Only use relative imports, drop the fallback to top-level modules.
    substituteInPlace $variantDir/collective.py \
      --replace-fail $'try:\n    from . import p2p\n    from . import utils\nexcept ImportError:\n    import p2p\n    import utils\n' $'from . import p2p\nfrom . import utils\n'
    substituteInPlace $variantDir/_vendor/intervaltree/intervaltree.py \
      --replace-fail "from sortedcontainers import" "from ..sortedcontainers import"

    # The binaries only depend on system libraries, with the exception of the
    # bundled libnuma that is used by ep.
    for lib in $variantDir/*.so $variantDir/lib/*.so; do
      patchelf --remove-rpath $lib
    done
    patchelf --set-rpath '$ORIGIN/uccl.libs' $variantDir/ep.abi3.so

    jq -n \
      --arg id "$sdkId" \
      --arg backend "${backend}" \
      --argjson archs '${builtins.toJSON archs}' \
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
    description = "UCCL built as an SDK (build variant ${variant})";
    homepage = "https://github.com/uccl-project/uccl";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
