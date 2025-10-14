import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os


def test_megablocks_moe_mlp_import():
    from megablocks.layers import MegaBlocksMoeMLP

    assert MegaBlocksMoeMLP is not None, "MegaBlocksMoeMLP import failed."


def run_distributed_test(rank, world_size):
    from megablocks.layers import MegaBlocksMoeMLP

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )

    expert_parallel_group = torch.distributed.new_group(
        range(torch.distributed.get_world_size())
    )

    model = MegaBlocksMoeMLP()
    model.expert_parallel_group = expert_parallel_group

    class Experts:
        def __init__(self):
            self.gate_up_proj = None
            self.gate_up_proj_bias = None
            self.down_proj = None
            self.down_proj_bias = None
            self.hidden_size = None

    model.experts = Experts()

    num_experts = 128
    hidden_size = 1152
    intermediate_size = 3072

    ne, hs, isz = num_experts, hidden_size, intermediate_size

    experts_per_rank = ne // world_size

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.router = torch.nn.Linear(hs, ne).to(device)
    model.router.weight.data.fill_(1)

    e = model.experts
    e.gate_up_proj = torch.nn.Parameter(
        torch.ones(experts_per_rank, hs, isz, device=device)
    )
    e.gate_up_proj_bias = torch.nn.Parameter(
        torch.zeros(experts_per_rank, isz, device=device)
    )
    e.down_proj = torch.nn.Parameter(
        torch.ones(experts_per_rank, 1536, hs, device=device)
    )
    e.down_proj_bias = torch.nn.Parameter(
        torch.zeros(experts_per_rank, hs, device=device)
    )
    e.hidden_size = hs

    x = torch.randn(1, 1, 1152).to(device)
    output, expert_weights_out = model(x)

    assert output.shape == (1, 1, 1152), f"Output shape mismatch on rank {rank}."

    print(f"Rank {rank}: Test passed! Output shape: {output.shape}")

    dist.destroy_process_group()


def test_megablocks_moe_mlp_functionality():
    world_size = 2

    mp.spawn(run_distributed_test, args=(world_size,), nprocs=world_size, join=True)

    print("Multi-process test completed successfully!")


if __name__ == "__main__":
    test_megablocks_moe_mlp_import()
    print("Import test passed!")

    test_megablocks_moe_mlp_functionality()
