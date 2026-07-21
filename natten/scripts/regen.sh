#!/usr/bin/env bash
# Regenerates the committed NATTEN autogen kernel instantiations and the
# build.toml src lists. Run this when bumping the vendored NATTEN version
# (after re-vendoring csrc/ and scripts/autogen_*.py from upstream).
#
# The split counts below are upstream's "default" autogen policy
# (NUM_SPLITS["default"] in NATTEN's setup.py). Note that the vendored
# autogen scripts are patched to not emit `#include <torch/extension.h>`
# (pybind11 is incompatible with the Python limited API that kernel-builder
# targets); re-apply that patch when re-vendoring the scripts.

set -euo pipefail

cd "$(dirname "$0")/.."

rm -rf csrc/autogen

python3 scripts/autogen_reference_fna.py --num-splits 2 -o csrc
python3 scripts/autogen_fna.py --num-splits 64 -o csrc
python3 scripts/autogen_fmha.py --num-splits 6 -o csrc
python3 scripts/autogen_hopper_fna.py --num-splits 8 -o csrc
python3 scripts/autogen_hopper_fna_bwd.py --num-splits 4 -o csrc
python3 scripts/autogen_hopper_fmha.py --num-splits 5 -o csrc
python3 scripts/autogen_hopper_fmha_bwd.py --num-splits 5 -o csrc
python3 scripts/autogen_blackwell_fna.py --num-splits 28 -o csrc
python3 scripts/autogen_blackwell_fna_bwd.py --num-splits 14 -o csrc
python3 scripts/autogen_blackwell_fmha.py --num-splits 4 -o csrc
python3 scripts/autogen_blackwell_fmha_bwd.py --num-splits 4 -o csrc

if grep -rl "torch/extension.h" csrc/autogen >/dev/null; then
    echo "error: autogen output includes torch/extension.h (pybind11);" >&2
    echo "the vendored autogen scripts lost the limited-API patch." >&2
    exit 1
fi

python3 scripts/generate_build_toml.py

echo "Done. Review changes to csrc/autogen and build.toml, then commit."
