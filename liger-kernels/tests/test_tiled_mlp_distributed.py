"""Run with torchrun --standalone --nproc-per-node=2 -m pytest <this file>.

Four ranks also exercise composed two-dimensional meshes. The same tests can
be launched with one rank per node to check cross-node collectives.
"""

import copy
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint

import kernels

liger_kernels = kernels.get_kernel("kernels-community/liger-kernels", version=4)
apply_tiled_mlp = liger_kernels.layers.apply_tiled_mlp


@pytest.fixture(scope="module")
def distributed():
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("launch with torchrun and at least two ranks")
    if not torch.cuda.is_available():
        pytest.skip("distributed tests need CUDA or ROCm GPUs")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    yield torch.device("cuda", local_rank)
    dist.destroy_process_group()


class MLP(nn.Module):
    def __init__(self, tiled):
        super().__init__()
        self.gate_proj = nn.Linear(16, 32, bias=False)
        self.up_proj = nn.Linear(16, 32, bias=False)
        self.down_proj = nn.Linear(32, 16, bias=False)
        self.tiled = tiled

    def forward(self, x):
        forward = liger_kernels.layers.LigerSwiGLUMLP.forward
        if self.tiled:
            # 6 tokens / 4 requested chunks is an adversarial torch.chunk case:
            # it produces only 3 chunks, unlike tensor_split's exact 4.
            return apply_tiled_mlp(forward, self, x, num_shards=4)
        return forward(self, x)


@pytest.mark.parametrize("layout", ["ddp", "fsdp2", "tp", "cp", "fsdp2_tp", "fsdp2_cp", "tp_cp", "dp_cp"])
@pytest.mark.parametrize("outer_checkpoint", [False, True])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_distributed_tiled_mlp(distributed, layout, outer_checkpoint, requires_grad):
    device = distributed
    world_size = dist.get_world_size()
    if "_" in layout:
        if world_size < 4 or world_size % 2:
            pytest.skip("composed meshes require an even world size of at least four")
        names = tuple(layout.replace("fsdp2", "fsdp").split("_"))
        shape = (2, world_size // 2)
    else:
        names = ({"ddp": "dp", "fsdp2": "fsdp"}.get(layout, layout),)
        shape = (world_size,)
    mesh = init_device_mesh("cuda", shape, mesh_dim_names=names)
    torch.manual_seed(0)
    reference = MLP(tiled=False).to(device)
    tiled = copy.deepcopy(reference)
    tiled.tiled = True
    if "tp" in names:
        parallelize_module(
            tiled,
            mesh["tp"],
            {"gate_proj": ColwiseParallel(), "up_proj": ColwiseParallel(), "down_proj": RowwiseParallel()},
        )
    if "fsdp" in names:
        from torch.distributed.fsdp import fully_shard

        for child in tiled.children():
            fully_shard(child, mesh=mesh["fsdp"])
        fully_shard(tiled, mesh=mesh["fsdp"])
    wrapped = tiled
    if "dp" in names:
        wrapped = DistributedDataParallel(tiled, device_ids=[device.index], process_group=mesh["dp"].get_group())
    coordinate = dict(zip(names, mesh.get_coordinate()))
    data_rank = coordinate.get("dp", coordinate.get("fsdp", 0))
    torch.manual_seed(100 + data_rank)
    if "cp" in names:
        # The MLP portion of CP has no token communication. Its weight gradients
        # are reduced across the CP group after local-sequence backward.
        full_input = torch.randn(2, 13, 16, device=device)
        x = torch.tensor_split(full_input, mesh["cp"].size(), dim=1)[coordinate["cp"]].contiguous()
    else:
        # TP peers receive the same inputs; data-sharded peers can have unequal
        # sequence lengths and must still enter the same tile collectives.
        x = torch.randn(2, 6 + data_rank % 2, 16, device=device)
    # Two iterations verify hooks are ready once, optimizers receive gradients,
    # and no parameter state is accidentally carried across iterations.
    optimizer_ref = torch.optim.SGD(reference.parameters(), lr=0.01)
    optimizer_tiled = torch.optim.SGD(wrapped.parameters(), lr=0.01)
    for _ in range(2):
        optimizer_ref.zero_grad(set_to_none=True)
        optimizer_tiled.zero_grad(set_to_none=True)
        x_ref = x.detach().clone().requires_grad_(requires_grad)
        x_tiled = x.detach().clone().requires_grad_(requires_grad)
        expected = reference(x_ref)
        actual = checkpoint(wrapped, x_tiled, use_reentrant=False) if outer_checkpoint else wrapped(x_tiled)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        expected.square().sum().backward()
        actual.square().sum().backward()
        if requires_grad:
            torch.testing.assert_close(x_tiled.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
        for (name, expected_param), actual_param in zip(reference.named_parameters(), tiled.parameters()):
            assert actual_param.grad is not None, name
            if "cp" in names:
                local_grad = (
                    actual_param.grad.to_local() if isinstance(actual_param.grad, DTensor) else actual_param.grad
                )
                dist.all_reduce(local_grad, group=mesh["cp"].get_group())
                local_grad.div_(mesh["cp"].size())
            # All ranks' dense gradients include identical TP contributions, so
            # averaging the world also gives the DP/FSDP/CP reference reduction.
            dist.all_reduce(expected_param.grad)
            expected_param.grad.div_(world_size)
            actual_grad = (
                actual_param.grad.full_tensor() if isinstance(actual_param.grad, DTensor) else actual_param.grad
            )
            torch.testing.assert_close(actual_grad, expected_param.grad, atol=1e-5, rtol=1e-5, msg=name)
        optimizer_ref.step()
        optimizer_tiled.step()
