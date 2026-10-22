# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-TMA descriptors for the kernels' descriptor memory arms. ``maybe_descriptor`` builds one
where the operand layout can back it; the ``rebind_*`` functions are the autotuner per-config
``pre_hook``s that size each descriptor's box to the tuned tile (they MUST mutate ``block_shape`` in
place — a rebind never reaches the launch) and are no-ops for the pointer arms, whose descriptor
argument is a dead placeholder."""

import torch
from triton.tools.tensor_descriptor import TensorDescriptor


def rebind_operand_box(nargs, mode_key, desc_key, rows, cols):
    """Set one operand's host-TMA box to ``[rows, cols]`` in place — MUST mutate (a rebind never
    reaches the launch). No-op for a pointer config, whose descriptor is a dead int placeholder."""
    desc = nargs[desc_key]
    # None = the operand's layout cannot back a descriptor (see `maybe_descriptor`); those
    # configs are pruned, so reaching here with one would mean a pruner gap, not a rebind target
    if nargs.get(mode_key, "pointer") != "pointer" and not isinstance(desc, int) and desc is not None:
        desc.block_shape = [rows, cols]


def maybe_descriptor(t: torch.Tensor, box: list[int]) -> TensorDescriptor | None:
    """A host-TMA descriptor over ``t``, or None when the layout cannot back one — a TMA box
    needs a unit-stride innermost dim, which a transposed view (as a backward pass hands in to
    contract over the other axis) does not have. ``None`` is the same signal the pointer arms
    already take for an unused descriptor, and ``descriptor_box_pruner`` drops the configs that
    would read it, so the tuner is left with the pointer arms the kernel serves stride-generally."""
    if t is None or (t.ndim >= 2 and t.stride(-1) != 1):
        return None
    return TensorDescriptor.from_tensor(t, box)


def rebind_bd_descriptors(nargs):
    """Per-config pre_hook: set the A and B host-TMA boxes to the tuned tile over the
    ``(rows, K)`` matrices — ``[BLOCK_SIZE_M, block_k]`` and ``[BLOCK_SIZE_N, block_k]``."""
    rebind_operand_box(nargs, "A_MEMORY_MODE", "ADescriptor", nargs["BLOCK_SIZE_M"], nargs["block_k"])
    rebind_operand_box(nargs, "B_MEMORY_MODE", "BDescriptor", (2 if nargs.get("GATE") else 1) * nargs["BLOCK_SIZE_N"], nargs["block_k"])


def rebind_weight_only_descriptors(nargs):
    """Per-config pre_hook for the weight-only kernels — set the A/B host-TMA boxes to the tuned tile:
    ``[BM, BK]`` over the (M, K) bf16 activation and ``[BN, BK // WEIGHT_VALUES_PER_BYTE]`` over the
    packed (N, K_bytes) weight (bytes: uint8 = packed E2M1, two values/byte). No scale boxes — weight-only
    scales are always affine (never swizzled), read through the pointer arm."""
    wvpb = 2 if nargs["B"].dtype == torch.uint8 else 1
    bk = nargs["BLOCK_SIZE_K"]
    rebind_operand_box(nargs, "A_MEMORY_MODE", "ADescriptor", nargs["BLOCK_SIZE_M"], bk)
    rebind_operand_box(nargs, "B_MEMORY_MODE", "BDescriptor", (2 if nargs.get("GATE") else 1) * nargs["BLOCK_SIZE_N"], bk // wvpb)


def rebind_mx_descriptors(nargs):
    """Per-config pre_hook for the MX kernel — set the A/B host-TMA boxes to the tuned tile
    in BYTES over the (rows, K_bytes) packed matrices: ``[BM, BK // ACT_VALUES_PER_BYTE]`` and
    ``[BN, BK // WEIGHT_VALUES_PER_BYTE]`` (values-per-byte read off the operand dtype; uint8 =
    packed E2M1), plus the SWIZZLE_32_4_4 scale boxes on the offline (pre-quantized A) path."""
    avpb = 2 if nargs["A"].dtype == torch.uint8 else 1
    wvpb = 2 if nargs["B"].dtype == torch.uint8 else 1
    bk = nargs["BLOCK_SIZE_K"]
    rebind_operand_box(nargs, "A_MEMORY_MODE", "ADescriptor", nargs["BLOCK_SIZE_M"], bk // avpb)
    rebind_operand_box(nargs, "B_MEMORY_MODE", "BDescriptor", (2 if nargs.get("GATE") else 1) * nargs["BLOCK_SIZE_N"], bk // wvpb)
    # SWIZZLE_32_4_4 scale boxes: [1, BLOCK//128, (BK // SCALE_GROUP_K) // 4, 2, 256]. Only where the
    # scale is actually swizzled — else the SA/SB descriptor is a dummy aliased to the operand
    # descriptor, and stamping a scale box would clobber its [BM, BK] operand box. The weight is
    # swizzled iff SWIZZLED_SCALES; the act only when it was also offline-quantized (E4M3 / packed
    # E2M1) — inline (raw bf16 A) stays affine even under a swizzled weight.
    if nargs["SWIZZLED_SCALES"]:
        # Sub-128 tiles read scales via the per-row pointer gather, leaving the descriptor
        # unread — clamp its box to one block so it keeps a valid, non-degenerate shape
        # (a 0-block box traps the descriptor-encoding pass; same clamp as the grouped hook).
        rep_k = max((nargs["BLOCK_SIZE_K"] // nargs["SCALE_GROUP_K"]) // 4, 1)
        bn_blocks = max(1, ((2 if nargs.get("GATE") else 1) * nargs["BLOCK_SIZE_N"]) // 128)
        nargs["BSDescriptor"].block_shape = [1, bn_blocks, rep_k, 2, 256]
        if nargs["A"].dtype in (torch.float8_e4m3fn, torch.uint8):
            bm_blocks = max(nargs["BLOCK_SIZE_M"] // 128, 1)
            nargs["ASDescriptor"].block_shape = [1, bm_blocks, rep_k, 2, 256]
    # swizzled requant output (Cs is a descriptor): store tile [1, 1, rep_n, 2, 256], rep_n per config
    if nargs["CSDescriptor"] is not None:
        rep_n = (nargs["BLOCK_SIZE_N"] // nargs["SCALE_GROUP_K"]) // 4
        nargs["CSDescriptor"].block_shape = [1, 1, rep_n, 2, 256]


def rebind_batched_mx_bs_descriptor(nargs):
    """Per-config pre_hook: size the swizzled weight-scale descriptor box to the tile's 128-row
    blocks (doubled under GATE, whose tile spans 2*BN interleaved rows). BN<128 (fp8 scalar)
    pointer-gathers instead and never reads the descriptor. Only under SWIZZLED_SCALES; the
    un-swizzled path keeps its dummy box."""
    if not nargs.get("SWIZZLED_SCALES"):
        return
    rep = max(1, ((2 if nargs.get("GATE") else 1) * nargs["BLOCK_SIZE_N"]) // 128)
    rep_k = (nargs["BLOCK_SIZE_K"] // nargs["SCALE_GROUP_K"]) // 4
    nargs["BSDescriptor"].block_shape = [1, rep, rep_k, 2, 256]


def rebind_grouped_weight_descriptor(nargs):
    """Per-config pre_hook: set the MX weight descriptor box to the tuned
    ``[1, (2 if GATE else 1) * BLOCK_SIZE_N, BLOCK_SIZE_K // values_per_byte]`` over the
    ``(E, 2N|N, K_bytes)`` weight view. MUST mutate ``block_shape`` in place — a rebind never reaches
    the launch. No-op for pointer configs (they never read the descriptor)."""
    if nargs.get("B_MEMORY_MODE", "pointer") == "pointer" or isinstance(
        nargs["BDescriptor"], int
    ):
        return
    values_per_byte = 2 if nargs["B"].dtype == torch.uint8 else 1
    nargs["BDescriptor"].block_shape = [
        1,
        (2 if nargs.get("GATE") else 1) * nargs["BLOCK_SIZE_N"],
        nargs["BLOCK_SIZE_K"] // values_per_byte,
    ]


def rebind_grouped_act_descriptor(nargs):
    """Per-config pre_hook: set the activation descriptor box to the tuned
    ``[BLOCK_SIZE_M, BLOCK_SIZE_K // act_values_per_byte]`` over the ``(rows, K_bytes)``
    activation matrix. In-place mutate; no-op for pointer-A configs."""
    if nargs.get("A_MEMORY_MODE", "pointer") == "pointer" or isinstance(
        nargs["ADescriptor"], int
    ):
        return
    act_values_per_byte = 2 if nargs["A"].dtype == torch.uint8 else 1
    # tma gather4 loads N independent rows: descriptor_gather requires a 1-row box;
    # the contiguous (no-gather) arm loads the whole [BM, BK_bytes] tile in one box
    nargs["ADescriptor"].block_shape = [
        1 if nargs.get("GatherIdx") is not None else nargs["BLOCK_SIZE_M"],
        nargs["BLOCK_SIZE_K"] // act_values_per_byte,
    ]


def rebind_grouped_descriptors(nargs):
    """Composite pre_hook: both weight and activation descriptor boxes."""
    rebind_grouped_weight_descriptor(nargs)
    rebind_grouped_act_descriptor(nargs)


def build_grouped_operand_descriptors(a_operand, b_operand):
    """Operand host-TMA descriptors for a grouped launch: A box ``[16, 64]``, B box
    ``[1, 128, 64]`` over the ``(E, 2N|N, K_bytes)`` weight view. Placeholder boxes, re-bound to
    the tuned tile per config by ``rebind_grouped_descriptors``."""
    return (
        TensorDescriptor.from_tensor(a_operand, block_shape=[16, 64]),
        TensorDescriptor.from_tensor(b_operand, block_shape=[1, 128, 64]),
    )


def rebind_grouped_mx_descriptors(nargs):
    """MX composite pre_hook: the operand boxes plus the two SWIZZLE_32_4_4 scale boxes
    ``[1, BLOCK // 128, (BK // SCALE_GROUP_K) // 4, 2, 256]`` over the swizzled
    ``(1, rows // 128, cols // 4, 2, 256)`` views. Activation box is BM // 128 (BM pinned 128);
    the weight box is ``(2 if GATE else 1) * BN // 128`` — the stacked gate|up tile is one 2*BN
    block (BN pinned 128 under GATE by ``swizzled_scales_bm_pruner``). Mutate in place. Both scale
    descriptors are None on the un-swizzled arm (affine read — one SWIZZLED_SCALES flag governs both
    operands), so their boxes are skipped."""
    rebind_grouped_descriptors(nargs)
    # BK below 4 scale groups (BK=64 rows the pruner rejects) still passes through here when a
    # descriptor is unread — clamp like bn_blocks below so it keeps a valid, non-degenerate
    # shape (a 0-block box traps the descriptor-encoding pass).
    rep_k = max((nargs["BLOCK_SIZE_K"] // nargs["SCALE_GROUP_K"]) // 4, 1)
    if nargs["ASDescriptor"] is not None:
        nargs["ASDescriptor"].block_shape = [1, nargs["BLOCK_SIZE_M"] // 128, rep_k, 2, 256]
    if nargs["BSDescriptor"] is not None:
        # One bulk-load spans (2 if GATE) * BN//128 blocks: GATE reads the block-interleaved gate|up
        # pair ([g,u] adjacent) as one 2*BN tile. BN<128 (non-gate, non-128 N) reads via the per-row
        # pointer gather instead — clamp the box to one block so the (unread) descriptor keeps a valid,
        # non-degenerate shape (a 0-block box traps the descriptor-encoding pass).
        bn_blocks = max(1, ((2 if nargs.get("GATE") else 1) * nargs["BLOCK_SIZE_N"]) // 128)
        nargs["BSDescriptor"].block_shape = [1, bn_blocks, rep_k, 2, 256]
    # Swizzled requant output (Cs is a descriptor): the store tile is [1, 1, rep_n, 2, 256], and
    # rep_n = (BN // SCALE_GROUP_K) // 4 depends on the tuned BLOCK_SIZE_N and the group size — so
    # rebind per config, else nvfp4 (group-16 -> rep_n=2) mismatches the [.,.,1,.,.] build default.
    if nargs["CSDescriptor"] is not None:
        rep_n = (nargs["BLOCK_SIZE_N"] // nargs["SCALE_GROUP_K"]) // 4
        nargs["CSDescriptor"].block_shape = [1, 1, rep_n, 2, 256]


def rebind_dgrad_descriptors(nargs):
    """Per-config pre_hook: set the dgrad boxes to the tuned tile — ``[BM, BN]`` over the (M, N)
    gradient and ``[BN, BK // WEIGHT_VALUES_PER_BYTE]`` over the (N, K_bytes) weight. Both are
    natural contiguous windows in the FORWARD-oriented operands, which is why dgrad can take a
    descriptor at all: a transposed view has no unit-stride innermost dim (see
    ``maybe_descriptor``). Scales stay affine on the pointer arm, as weight-only does."""
    wvpb = 2 if nargs["B"].dtype == torch.uint8 else 1
    rebind_operand_box(
        nargs, "A_MEMORY_MODE", "ADescriptor", nargs["BLOCK_SIZE_M"], nargs["BLOCK_SIZE_N"]
    )
    rebind_operand_box(
        nargs, "B_MEMORY_MODE", "BDescriptor",
        nargs["BLOCK_SIZE_N"], nargs["BLOCK_SIZE_K"] // wvpb,
    )


def rebind_dgrad_grouped_descriptors(nargs):
    """Grouped dgrad boxes. The activation box is 1 row when the pass gathers (tma gather4 needs a
    1-row box) and ``[BM, BN]`` otherwise — ScatterIdx is the gather map here, since dgrad reads
    dY at the forward's SCATTER destination. The weight box is ``[1, BN, BK_bytes]`` over
    (E, N, K_bytes): the expert leads, so this descriptor takes THREE offsets."""
    wvpb = 2 if nargs["B"].dtype == torch.uint8 else 1
    gathering = nargs.get("ScatterIdx") is not None
    rebind_operand_box(
        nargs, "A_MEMORY_MODE", "ADescriptor",
        1 if gathering else nargs["BLOCK_SIZE_M"], nargs["BLOCK_SIZE_N"],
    )
    desc = nargs["BDescriptor"]
    if nargs.get("B_MEMORY_MODE", "pointer") != "pointer" and not isinstance(desc, int) and desc is not None:
        desc.block_shape = [1, nargs["BLOCK_SIZE_N"], nargs["BLOCK_SIZE_K"] // wvpb]
