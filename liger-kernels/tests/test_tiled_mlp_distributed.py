"""Run with torchrun --standalone --nproc-per-node=2 -m pytest <this file>.

Four ranks also exercise composed two-dimensional meshes. The same tests can
be launched with one rank per node to check cross-node collectives.
"""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint

import kernels

liger_kernels = kernels.get_kernel("kernels-community/liger-kernels", version=4)
apply_tiled_mlp = liger_kernels.layers.apply_tiled_mlp

pytestmark = [
    pytest.mark.skipif(
        int(os.environ.get("WORLD_SIZE", "1")) < 2,
        reason="launch with torchrun and at least two ranks",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="distributed tests need CUDA or ROCm GPUs"
    ),
]


@pytest.fixture(scope="module")
def distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    yield torch.device("cuda", local_rank)
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def meshes(distributed):
    return {}


class MLP(nn.Module):
    def __init__(self, variant="swiglu"):
        super().__init__()
        self.gate_proj = nn.Linear(16, 32, bias=False)
        self.up_proj = nn.Linear(16, 32, bias=False)
        self.down_proj = nn.Linear(32, 16, bias=False)
        self.variant = variant
        self.num_shards = 4
        self.gate_multiplier = 0.8838834764831844
        self.down_multiplier = 0.5859375
        self.swiglu_limit = 0.5

    def forward(self, x):
        if self.variant == "swiglu":
            return liger_kernels.layers.LigerSwiGLUMLP.forward(self, x)

        gate, up = self.gate_proj(x), self.up_proj(x)
        if self.variant == "scaled":
            gate = gate * self.gate_multiplier
        else:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)

        output = self.down_proj(nn.functional.silu(gate) * up)
        return output * self.down_multiplier if self.variant == "scaled" else output


class TiledMLP(MLP):
    def forward(self, x):
        if self.variant == "swiglu":
            # 6 tokens / 4 requested chunks produces only 3 torch.chunk outputs.
            return apply_tiled_mlp(
                liger_kernels.layers.LigerSwiGLUMLP.forward,
                self,
                x,
                num_shards=self.num_shards,
            )

        adapter = (
            liger_kernels.layers.LigerTiledScaledSwiGLUMLP
            if self.variant == "scaled"
            else liger_kernels.layers.LigerTiledClampedSwiGLUMLP
        )
        return adapter.forward(self, x)


@pytest.mark.parametrize(
    "layout", ["ddp", "fsdp2", "tp", "cp", "fsdp2_tp", "fsdp2_cp", "tp_cp", "dp_cp"]
)
@pytest.mark.parametrize("outer_checkpoint", [False, True])
@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("mixed_precision", [False, True], ids=["fp32", "bf16"])
@pytest.mark.parametrize("variant", ["swiglu", "scaled", "clamped"])
def test_distributed_tiled_mlp(
    distributed,
    meshes,
    layout,
    outer_checkpoint,
    requires_grad,
    mixed_precision,
    variant,
):
    device = distributed
    pending_collectives = getattr(
        torch._C._distributed_c10d, "_get_work_registry_size", None
    )
    pending_before = pending_collectives() if pending_collectives else None
    world_size = dist.get_world_size()
    if "_" in layout:
        if world_size < 4 or world_size % 2:
            pytest.skip("composed meshes require an even world size of at least four")
        names = tuple(layout.replace("fsdp2", "fsdp").split("_"))
        shape = (2, world_size // 2)
    else:
        names = ({"ddp": "dp", "fsdp2": "fsdp"}.get(layout, layout),)
        shape = (world_size,)
    if layout not in meshes:
        meshes[layout] = init_device_mesh("cuda", shape, mesh_dim_names=names)
    mesh = meshes[layout]

    torch.manual_seed(0)
    reference = MLP(variant=variant).to(device)
    tiled = TiledMLP(variant=variant).to(device)
    tiled.load_state_dict(reference.state_dict())

    if "tp" in names:
        parallelize_module(
            tiled,
            mesh["tp"],
            {
                "gate_proj": ColwiseParallel(),
                "up_proj": ColwiseParallel(),
                "down_proj": RowwiseParallel(),
            },
        )
    if "fsdp" in names:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        policy = (
            MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            if mixed_precision
            else MixedPrecisionPolicy()
        )
        for child in tiled.children():
            fully_shard(child, mesh=mesh["fsdp"], mp_policy=policy)
        fully_shard(tiled, mesh=mesh["fsdp"], mp_policy=policy)

    wrapped = tiled
    if "dp" in names:
        wrapped = DistributedDataParallel(
            tiled, device_ids=[device.index], process_group=mesh["dp"].get_group()
        )

    coordinate = dict(zip(names, mesh.get_coordinate()))
    data_rank = coordinate.get("dp", coordinate.get("fsdp", 0))
    torch.manual_seed(100 + data_rank)
    if "cp" in names:
        # The MLP portion of CP has no token communication. Its weight gradients
        # are reduced across the CP group after local-sequence backward.
        full_input = torch.randn(2, 13, 16, device=device)
        x = torch.tensor_split(full_input, mesh["cp"].size(), dim=1)[
            coordinate["cp"]
        ].contiguous()
    else:
        # TP peers receive the same inputs; data-sharded peers can have unequal
        # sequence lengths and must still enter the same tile collectives.
        x = torch.randn(2, 6 + data_rank % 2, 16, device=device)

    # Two iterations verify hooks are ready once, optimizers receive gradients,
    # and no parameter state is accidentally carried across iterations.
    optimizer_ref = torch.optim.SGD(reference.parameters(), lr=0.01)
    optimizer_tiled = torch.optim.SGD(wrapped.parameters(), lr=0.01)
    atol, rtol = (0.002, 0.03) if mixed_precision else (1e-5, 1e-5)

    for _ in range(2):
        optimizer_ref.zero_grad(set_to_none=True)
        optimizer_tiled.zero_grad(set_to_none=True)

        x_ref = x.detach().clone().requires_grad_(requires_grad)
        x_tiled = x.detach().clone().requires_grad_(requires_grad)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed_precision):
            expected = reference(x_ref)
            actual = (
                checkpoint(wrapped, x_tiled, use_reentrant=False)
                if outer_checkpoint
                else wrapped(x_tiled)
            )

        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

        expected.square().sum().backward()
        actual.square().sum().backward()
        if requires_grad:
            torch.testing.assert_close(x_tiled.grad, x_ref.grad, atol=atol, rtol=rtol)

        for (name, expected_param), actual_param in zip(
            reference.named_parameters(), tiled.parameters()
        ):
            assert actual_param.grad is not None, name
            if "cp" in names:
                local_grad = (
                    actual_param.grad.to_local()
                    if isinstance(actual_param.grad, DTensor)
                    else actual_param.grad
                )
                dist.all_reduce(local_grad, group=mesh["cp"].get_group())
                local_grad.div_(mesh["cp"].size())
            # All ranks' dense gradients include identical TP contributions, so
            # averaging the world also gives the DP/FSDP/CP reference reduction.
            dist.all_reduce(expected_param.grad)
            expected_param.grad.div_(world_size)
            actual_grad = (
                actual_param.grad.full_tensor()
                if isinstance(actual_param.grad, DTensor)
                else actual_param.grad
            )
            torch.testing.assert_close(
                actual_grad,
                expected_param.grad,
                atol=0.02 if mixed_precision else atol,
                rtol=rtol,
                msg=name,
            )

        optimizer_ref.step()
        optimizer_tiled.step()

    if pending_collectives:
        assert pending_collectives() == pending_before, (
            "checkpoint recomputation left an unwaited collective"
        )


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("outer_checkpoint", [False, True])
@pytest.mark.parametrize("variant", ["swiglu", "scaled", "clamped"])
def test_tiled_zero3(distributed, requires_grad, outer_checkpoint, variant):
    deepspeed = pytest.importorskip("deepspeed")
    from deepspeed.utils import safe_get_full_grad

    device = distributed

    torch.manual_seed(0)
    reference = MLP(variant=variant).to(device)
    tiled = TiledMLP(variant=variant).to(device)
    tiled.load_state_dict(reference.state_dict())
    optimizer_ref = torch.optim.SGD(reference.parameters(), lr=0.01)

    engine, _, _, _ = deepspeed.initialize(
        model=tiled,
        optimizer=torch.optim.SGD(tiled.parameters(), lr=0.01),
        config={
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "gradient_clipping": 0.0,
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {"stage": 3, "stage3_param_persistence_threshold": 0},
        },
        dist_init_required=False,
    )

    torch.manual_seed(100 + dist.get_rank())
    x = torch.randn(2, 6 + dist.get_rank() % 2, 16, device=device)

    for _ in range(2):
        optimizer_ref.zero_grad(set_to_none=True)

        x_ref = x.detach().clone().requires_grad_(requires_grad)
        x_tiled = x.detach().clone().requires_grad_(requires_grad)

        expected = reference(x_ref)
        actual = (
            checkpoint(engine, x_tiled, use_reentrant=False)
            if outer_checkpoint
            else engine(x_tiled)
        )

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

        expected.square().sum().backward()
        engine.backward(actual.square().sum())
        if requires_grad:
            torch.testing.assert_close(x_tiled.grad, x_ref.grad, atol=1e-5, rtol=1e-5)

        for (name, expected_param), actual_param in zip(
            reference.named_parameters(), tiled.parameters()
        ):
            dist.all_reduce(expected_param.grad)
            expected_param.grad.div_(dist.get_world_size())
            actual_grad = safe_get_full_grad(actual_param)
            assert actual_grad is not None, name
            torch.testing.assert_close(
                actual_grad, expected_param.grad, atol=1e-5, rtol=1e-5, msg=name
            )

        optimizer_ref.step()
        engine.step()
