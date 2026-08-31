"""The Metal type table against the upstream code it was transcribed from.

`gguf_metal/ggml_dispatch.mm` re-implements, for the two ops here, what ggml's Metal backend does in
`ggml_metal_library_get_pipeline_mul_mv` and `ggml_metal_op_mul_mat`: which types have a gemv, how
much threadgroup memory each wants, and which of the two grids it is dispatched on. Those files are
vendored, so `vendor.py --rev` replaces them, and a pin bump can change any of it.

Most of that drift is caught by the compiler -- the nsg/nr0 values are used through upstream's own
`N_SG_*`/`N_R0_*` macros, so a rename or removal fails the build. What is not caught is a changed
`smem` expression or a changed grid rule: both still compile and both silently return wrong numbers.
This parses upstream and fails loudly instead.

No device and no built extension needed, so it runs anywhere.
"""

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
DEVICE_CPP = ROOT / "vendor/src/ggml-metal/ggml-metal-device.cpp"
OPS_CPP = ROOT / "vendor/src/ggml-metal/ggml-metal-ops.cpp"
DISPATCH_MM = ROOT / "gguf_metal/ggml_dispatch.mm"

pytestmark = pytest.mark.skipif(not DEVICE_CPP.exists(), reason="vendor/ is not checked out")


def upstream_mul_mv_table():
    """{type name: smem expression} from ggml_metal_library_get_pipeline_mul_mv's switch."""
    body = DEVICE_CPP.read_text()
    start = body.index("ggml_metal_library_get_pipeline_mul_mv(")
    end = body.index("ggml_metal_library_get_pipeline_mul_mm_id_map0", start)
    switch = body[start:end]

    table = {}
    for case in re.finditer(
        r"case GGML_TYPE_(\w+):\s*\{(.*?)\}\s*break;", switch, re.S
    ):
        name, arm = case.group(1), case.group(2)
        smem = re.search(r"smem\s*=\s*([^;]+);", arm)
        table[name] = " ".join(smem.group(1).split()) if smem else "0"
    return table


def ours():
    """{type name: smem expression} from our table, keyed the way upstream names its types."""
    body = DISPATCH_MM.read_text()
    start = body.index("const std::unordered_map<int, TypeInfo> &type_table()")
    end = body.index("const TypeInfo *lookup(", start)

    table = {}
    for entry in re.finditer(
        r"\{GGML_(\w+),\s*\{\"[\w]+\",\s*\d+,\s*\d+,\s*N_SG_\w+,\s*N_R0_\w+,\s*(true|false),\s*([^}]+)\}\}",
        body[start:end],
    ):
        name, reduce_flag, smem = entry.group(1), entry.group(2), entry.group(3)
        table[name] = (" ".join(smem.split()).rstrip(","), reduce_flag == "true")
    return table


def normalize(expr: str) -> str:
    """`32 * sizeof(float) * N_R0_Q8_0` and `32*sizeof(float)*N_R0_Q8_0` are the same expression."""
    return expr.replace(" ", "").replace("(float)", "(float)")


def test_every_type_we_claim_still_exists_upstream():
    upstream = upstream_mul_mv_table()
    missing = sorted(set(ours()) - set(upstream))
    assert not missing, (
        f"{missing} are in our table but no longer in upstream's mul_mv switch; the pin moved under "
        "gguf_metal/ggml_dispatch.mm"
    )


def test_threadgroup_memory_matches_upstream():
    upstream = upstream_mul_mv_table()
    mismatched = {
        name: (smem, upstream[name])
        for name, (smem, _) in ours().items()
        if normalize(smem) != normalize(upstream[name])
    }
    assert not mismatched, (
        "upstream changed the threadgroup memory for these types (ours, theirs): "
        f"{mismatched}. Too little is a fault, too much silently costs occupancy."
    )


def test_grid_rule_still_singles_out_q8_0():
    """The one place a type's grid divisor is decided, and the one field we cannot get from a macro.

    Upstream dispatches the dense types and q8_0 on `(ne01 + nr0 - 1)/nr0` and everything else on
    `(ne01 + nr0*nsg - 1)/(nr0*nsg)`. Our `mv_reduce_across_sgs` is exactly that predicate, so if
    upstream ever moves a quantized type across, this catches it.
    """
    body = OPS_CPP.read_text()
    start = body.index("ggml_metal_library_get_pipeline_mul_mv(lib, op)")
    branch = body[start : body.index("return 1;", start)]

    condition = re.search(r"if \(op->src\[0\]->type ==(.*?)\) \{", branch, re.S)
    assert condition, "upstream's mul_mv grid branch is no longer shaped as a type comparison"
    quantized_on_nr0_grid = {
        t for t in re.findall(r"GGML_TYPE_(\w+)", condition.group(1))
    } - {"F32", "F16", "BF16"}

    ours_on_nr0_grid = {name for name, (_, reduce_flag) in ours().items() if reduce_flag}
    assert ours_on_nr0_grid == quantized_on_nr0_grid, (
        f"upstream now puts {quantized_on_nr0_grid} on the undivided grid, we assume "
        f"{ours_on_nr0_grid}; mv_reduce_across_sgs in gguf_metal/ggml_dispatch.mm has to follow"
    )
