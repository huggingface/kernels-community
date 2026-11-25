{
  lib,
  stdenv,
  kernel-port,

  einopsSrc,
  version,

  # The build variant to produce, `sdk`.
  variant,

  # Unique ID for the SDK build variant.
  sdkId,

  backend,
}:

stdenv.mkDerivation {
  pname = "einops-kernel-${variant}";
  inherit version;

  dontUnpack = true;
  dontFixup = true;

  inherit variant sdkId backend;

  nativeBuildInputs = [ kernel-port ];

  installPhase = ''
    runHook preInstall

    mkdir -p $out
    variantDir=$out/$variant
    kernel-port ${./port}/source.kdl --dir ${einopsSrc} --out $variantDir
    rm $variantDir/port-provenance.json

    # Torch is optional.
    cat >> $variantDir/__init__.py <<'EOF2'

    try:
        from .layers import torch  # noqa: E402,F401
    except ImportError:
        pass
    EOF2

    cat > $variantDir/metadata.json <<EOF2
    {
      "name": "einops",
      "id": "$sdkId",
      "version": 1,
      "license": "MIT",
      "upstream": "https://github.com/arogozhnikov/einops",
      "python-depends": [],
      "kernel-depends": [],
      "backend": { "type": "$backend" }
    }
    EOF2

    runHook postInstall
  '';

  meta = {
    description = "einops SDK (build variant ${variant})";
    homepage = "https://github.com/arogozhnikov/einops";
    license = lib.licenses.mit;
    sourceProvenance = with lib.sourceTypes; [ fromSource ];
  };
}
