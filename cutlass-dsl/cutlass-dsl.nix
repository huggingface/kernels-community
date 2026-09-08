{
  stdenv,

  variant,

  get-kernel-check,
  python,
  rsync,

  nvidia-cutlass-dsl-libs,
  nvidia-cutlass-dsl-libs-core,
  nvidia-cutlass-dsl-libs-cu,
}:

stdenv.mkDerivation (finalAttrs: {
  name = "cutlass-dsl";

  dontUnpack = true;

  nativeBuildInputs = [ get-kernel-check rsync ];

  env = {
    inherit variant;
    kernelDeps = "${./kernel-deps.json}";
    moduleName = "cutlass_dsl";
  };

  installPhase = ''
    mkdir -p $out/${variant}

    for d in ${nvidia-cutlass-dsl-libs} ${nvidia-cutlass-dsl-libs-core} ${nvidia-cutlass-dsl-libs-cu}; do
      rsync -a $d/${python.sitePackages}/nvidia_cutlass_dsl/ $out/${variant}/
    done

    chmod -R a+w $out/${variant}

    mv $out/${variant}/dsl_packages/* $out/${variant}

    echo "from . import cutlass" > $out/${variant}/__init__.py

    cp ${./metadata.json} $out/${variant}/metadata.json
  '';

  doInstallCheck = true;
})
