{
  runCommand,
  kernel-port,
  einopsSrc,
}:

runCommand "einops-tests" { nativeBuildInputs = [ kernel-port ]; } ''
  kernel-port ${./port}/tests.kdl --dir ${einopsSrc} --out $out
  rm $out/port-provenance.json
''
