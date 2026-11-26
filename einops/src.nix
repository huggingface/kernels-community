{ fetchFromGitHub, version }:

fetchFromGitHub {
  owner = "arogozhnikov";
  repo = "einops";
  rev = "v${version}";
  hash = "sha256-d5Vbtkw/MChS2j2IC6j97wfVoKWZT9mU4OeXyEjm6ys=";
}
