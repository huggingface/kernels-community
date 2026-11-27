"""Pieces the persistent grouped Gluon kernels share: the item schedule, the token-tile load (a box, or a row gather
straight from the unsorted activations) and the tuning hooks."""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from ..scheduling import fill_packed_schedule_rows

_GL_DTYPES = {torch.float8_e4m3fn: gl.float8e4nv, torch.bfloat16: gl.bfloat16, torch.float16: gl.float16}


def token_tile_descriptor(A: torch.Tensor, block_t: int, block_k: int, gather: bool) -> TensorDescriptor:
    """The activation operand's descriptor: a ``[block_t, block_k]`` box over expert-sorted rows, or a one-row box
    (laid out as the full tile) that ``tma.async_gather`` repeats over a tile's gathered rows."""
    layout = gl.NVMMASharedLayout.get_default_for([block_t, block_k], _GL_DTYPES[A.dtype])
    return TensorDescriptor.from_tensor(A, [1 if gather else block_t, block_k], layout)


@gluon.jit
def schedule_item(it, Schedule, NB: gl.constexpr):
    """Item ``it``'s (expert, first sorted row, row end, weight block) from the packed schedule."""
    m = it // NB
    entry = Schedule + 1 + m * 3
    return gl.load(entry), gl.load(entry + 1), gl.load(entry + 2), it % NB


@gluon.jit
def gather_rows(GatherIdx, m_start, row_end, num_rows, BT: gl.constexpr):
    """An item's source rows of the activations, for ``tma.async_gather``: rows past the expert point past the end,
    which TMA fills with zeros. The gather wants 4 contiguous offsets per thread."""
    layout: gl.constexpr = gl.SliceLayout(0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]))
    r = m_start + gl.arange(0, BT, layout=layout)
    return gl.load(GatherIdx + r, mask=r < row_end, other=num_rows)


@gluon.jit
def load_token_tile(a_desc, src_rows, m_start, k_off, bar, buf, GATHER: gl.constexpr):
    """The ``[BT, BK]`` token tile at ``k_off``: gathered rows (``src_rows``), or the box at sorted row ``m_start``."""
    if GATHER:
        tma.async_gather(a_desc, src_rows, k_off, bar, buf)
    else:
        tma.async_load(a_desc, [m_start, k_off], bar, buf)


# Tuned token tiles: 16 rows fill the 64-token batches (a few routed rows per expert), 32 halve the weight re-reads
# once experts hold more.
TOKEN_TILES = (16, 32)
# B200's 227 KB per CTA, less room for barriers and the epilogues' layout conversions.
SMEM_BUDGET = 216 * 1024


def grouped_pre_hook(desc_name: str):
    """Per config, as at launch: the packed schedule of the config's ``BT``-row tiles (into ``Schedule``, sized for the
    smallest tile) and the token tile's box over ``desc_name``."""
    def hook(nargs):
        bt = nargs["BT"]
        fill_packed_schedule_rows(nargs["ExpertStart"], nargs["Schedule"], bt)
        desc = nargs[desc_name]
        block_k = desc.block_shape[-1]
        desc.block_shape = [1 if nargs["GatherIdx"] is not None else bt, block_k]
        desc.layout = gl.NVMMASharedLayout.get_default_for([bt, block_k], _GL_DTYPES[desc.base.dtype])
    return hook


def smem_fit_pruner(stage_bytes):
    """``early_config_prune`` keeping the configs whose ``NS``-stage ring fits shared memory; ``stage_bytes(kwargs,
    gate)`` is one stage of the kernel's ring."""
    def prune(configs, named_args, **kwargs):
        gate = kwargs.get("GATE", named_args.get("GATE"))
        return [c for c in configs if c.kwargs["NS"] * stage_bytes(c.kwargs, gate) <= SMEM_BUDGET]
    return prune
