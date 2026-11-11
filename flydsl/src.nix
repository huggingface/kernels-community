{
  runCommand,
  fetchFromGitHub,
  version,
}:

let
  src = fetchFromGitHub {
    owner = "ROCm";
    repo = "FlyDSL";
    rev = "v${version}";
    hash = "sha256-GSs/HkbW7SGJH5ggRXMS/6L2YR/BMe3A6yvsqQamyis=";
  };

  # The submodules of the FlyDSL repository (`.gitmodules`), at the commits
  # that the tag points to.
  dlpack = fetchFromGitHub {
    owner = "dmlc";
    repo = "dlpack";
    rev = "84d107bf416c6bab9ae68ad285876600d230490d";
    hash = "sha256-kIHBgTYaHEmweRBFtRl1pXhOyQ5TEwU8dLUssTMEnpc=";
  };
  tvm-ffi = fetchFromGitHub {
    owner = "apache";
    repo = "tvm-ffi";
    rev = "ed067c17b259774f4ddc23ba7de937a90642bbb1";
    hash = "sha256-tSGkLRGNZ7iZeaBJSq5pOtHrXtcf9r+ZBVWmzZKwRno=";
  };
in
runCommand "flydsl-src-${version}" { } ''
  cp -r ${src} $out
  chmod -R u+w $out
  rmdir $out/thirdparty/dlpack $out/thirdparty/tvm-ffi
  cp -r ${dlpack} $out/thirdparty/dlpack
  cp -r ${tvm-ffi} $out/thirdparty/tvm-ffi
''
