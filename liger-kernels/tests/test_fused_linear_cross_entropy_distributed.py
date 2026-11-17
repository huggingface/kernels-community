"""Run with torchrun on four GPUs to include composed parallel meshes."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
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

pytestmark = [
    pytest.mark.skipif(
        int(os.environ.get("WORLD_SIZE", "1")) < 2,
        reason="launch with torchrun and at least two ranks",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA or ROCm"),
]


class LossModel(nn.Module):
    def __init__(self, loss_mode, outer_checkpoint=False):
        super().__init__()
        self.up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(16, 8, bias=False)
        self.weight = nn.Parameter(torch.randn(17, 8) * 0.2)
        self.bias = nn.Parameter(torch.randn(17) * 0.2)
        self.loss_mode = loss_mode
        self.outer_checkpoint = outer_checkpoint

    def forward(self, x, target):
        # FSDP input casting and all-gather run before the checkpoint boundary.
        if self.outer_checkpoint:
            return checkpoint(self.compute_loss, x, target, use_reentrant=False)
        return self.compute_loss(x, target)

    def compute_loss(self, x, target):
        hidden = self.down_proj(F.silu(self.up_proj(x)))
        return self.loss(hidden, target)

    def loss(self, hidden, target):
        reduction = "none" if self.loss_mode == "none" else "mean"
        logits = F.linear(hidden, self.weight, self.bias)
        if self.loss_mode == "causal":
            logits = 0.5 * (logits / 0.5).tanh()
        return F.cross_entropy(logits, target, reduction=reduction)


class FusedLossModel(LossModel):
    def loss(self, hidden, target):
        if self.loss_mode == "none":
            return liger_kernels.layers.liger_fused_linear_cross_entropy(
                hidden, self.weight, target, bias=self.bias, reduction="none"
            )

        return liger_kernels.layers.LigerForCausalLMLossLayer.forward(
            self,
            None,
            target,
            17,
            num_items_in_batch=target.numel(),
            shift_labels=target,
            hidden_states=hidden,
            lm_head_weight=self.weight,
            lm_head_bias=self.bias,
            hidden_size=8,
            final_logit_softcapping=0.5,
        )


def model_pair(device, loss_mode, outer_checkpoint=False):
    torch.manual_seed(0)
    reference = LossModel(loss_mode).to(device)
    fused = FusedLossModel(loss_mode, outer_checkpoint=outer_checkpoint).to(device)
    fused.load_state_dict(reference.state_dict())
    return reference, fused


@pytest.mark.parametrize(
    "layout", ["ddp", "fsdp2", "tp", "cp", "fsdp2_tp", "fsdp2_cp", "tp_cp", "dp_cp"]
)
@pytest.mark.parametrize("outer_checkpoint", [False, True])
@pytest.mark.parametrize("mixed_precision", [False, True], ids=["fp32", "bf16"])
@pytest.mark.parametrize("loss_mode", ["none", "causal"])
def test_distributed_fused_loss(
    distributed, meshes, layout, outer_checkpoint, mixed_precision, loss_mode
):
    device = distributed
    world_size = dist.get_world_size()
    if "_" in layout:
        if world_size < 4 or world_size % 2:
            pytest.skip("composed meshes need an even world size of at least four")
        names = tuple(layout.replace("fsdp2", "fsdp").split("_"))
        shape = (2, world_size // 2)
    else:
        names = ({"ddp": "dp", "fsdp2": "fsdp"}.get(layout, layout),)
        shape = (world_size,)

    if layout not in meshes:
        meshes[layout] = init_device_mesh("cuda", shape, mesh_dim_names=names)
    mesh = meshes[layout]
    reference, fused = model_pair(device, loss_mode, outer_checkpoint)

    # TP shards the preceding projections. The loss receives replicated hidden
    # states and a complete vocabulary head, as required by the existing API.
    if "tp" in names:
        parallelize_module(
            fused,
            mesh["tp"],
            {"up_proj": ColwiseParallel(), "down_proj": RowwiseParallel()},
        )
    if "fsdp" in names:
        policy = (
            MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            if mixed_precision
            else MixedPrecisionPolicy()
        )
        fully_shard(fused, mesh=mesh["fsdp"], mp_policy=policy)

    wrapped = fused
    if "dp" in names:
        wrapped = DistributedDataParallel(
            fused, device_ids=[device.index], process_group=mesh["dp"].get_group()
        )

    coordinate = dict(zip(names, mesh.get_coordinate()))
    data_rank = coordinate.get("dp", coordinate.get("fsdp", 0))
    torch.manual_seed(100 + data_rank)
    tokens = 13 if "cp" in names else 6 + data_rank % 2
    x = torch.randn(tokens, 8, device=device)
    target = torch.randint(17, (tokens,), device=device)
    if "cp" in names:
        x = torch.tensor_split(x, mesh["cp"].size())[coordinate["cp"]].contiguous()
        target = torch.tensor_split(target, mesh["cp"].size())[
            coordinate["cp"]
        ].contiguous()

    optimizer_ref = torch.optim.SGD(reference.parameters(), lr=0.01, foreach=False)
    optimizer_fused = torch.optim.SGD(wrapped.parameters(), lr=0.01, foreach=False)
    atol, rtol = (0.002, 0.03) if mixed_precision else (1e-5, 1e-5)

    for _ in range(2):
        optimizer_ref.zero_grad(set_to_none=True)
        optimizer_fused.zero_grad(set_to_none=True)
        x_ref = x.detach().clone().requires_grad_()
        x_fused = x.detach().clone().requires_grad_()

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed_precision):
            expected = reference(x_ref, target)
            actual = wrapped(x_fused, target)

        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        grad_output = (
            torch.linspace(0.1, 1.5, target.numel(), device=device)
            if loss_mode == "none"
            else None
        )
        expected.backward(grad_output)
        actual.backward(grad_output)
        torch.testing.assert_close(x_fused.grad, x_ref.grad, atol=atol, rtol=rtol)

        for (name, expected_param), actual_param in zip(
            reference.named_parameters(), fused.parameters()
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

            dist.all_reduce(expected_param.grad)
            expected_param.grad.div_(world_size)
            actual_grad = (
                actual_param.grad.full_tensor()
                if isinstance(actual_param.grad, DTensor)
                else actual_param.grad
            )
            # AMP rounds PyTorch linear's bias in BF16; fused CE adds an FP32
            # bias after matmul. Match the existing tiled MLP gradient tolerance.
            torch.testing.assert_close(
                actual_grad,
                expected_param.grad,
                atol=0.02 if mixed_precision else atol,
                rtol=rtol,
                msg=lambda message: f"{name}: {message}",
            )

        optimizer_ref.step()
        optimizer_fused.step()


@pytest.mark.parametrize("outer_checkpoint", [False, True])
@pytest.mark.parametrize("loss_mode", ["none", "causal"])
def test_fused_loss_zero3(distributed, outer_checkpoint, loss_mode):
    deepspeed = pytest.importorskip("deepspeed")
    from deepspeed.utils import safe_get_full_grad

    device = distributed
    reference, fused = model_pair(device, loss_mode, outer_checkpoint)
    optimizer_ref = torch.optim.SGD(reference.parameters(), lr=0.01, foreach=False)
    engine, _, _, _ = deepspeed.initialize(
        model=fused,
        optimizer=torch.optim.SGD(fused.parameters(), lr=0.01, foreach=False),
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
    tokens = 6 + dist.get_rank() % 2
    x = torch.randn(tokens, 8, device=device)
    target = torch.randint(17, (tokens,), device=device)

    for _ in range(2):
        optimizer_ref.zero_grad(set_to_none=True)
        x_ref = x.detach().clone().requires_grad_()
        x_fused = x.detach().clone().requires_grad_()

        expected = reference(x_ref, target)
        actual = (
            checkpoint(engine, x_fused, target, use_reentrant=False)
            if outer_checkpoint
            else engine(x_fused, target)
        )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

        # DeepSpeed's backward takes a scalar; nonuniform token weights exercise
        # the same loss-gradient path as backward(grad_output) above.
        token_weights = (
            torch.linspace(0.1, 1.5, tokens, device=device)
            if loss_mode == "none"
            else 1.0
        )
        (expected * token_weights).sum().backward()
        engine.backward((actual * token_weights).sum())
        torch.testing.assert_close(x_fused.grad, x_ref.grad, atol=1e-5, rtol=1e-5)

        for (name, expected_param), actual_param in zip(
            reference.named_parameters(), fused.parameters()
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
