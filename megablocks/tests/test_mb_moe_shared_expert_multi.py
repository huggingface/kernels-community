import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import pytest
from megablocks.layers import MegaBlocksMoeMLPWithSharedExpert, create_shared_expert_weights


def run_distributed_shared_expert_test(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )

    model = MegaBlocksMoeMLPWithSharedExpert()

    hidden_size = 128
    shared_expert_hidden_size = 192
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def simple_init(tensor):
        torch.nn.init.xavier_uniform_(tensor)

    shared_up_proj_weight, shared_down_proj_weight, shared_up_proj_bias, shared_down_proj_bias = create_shared_expert_weights(
        hidden_size=hidden_size,
        shared_expert_hidden_size=shared_expert_hidden_size,
        device=torch.device(device),
        dtype=torch.float32,
        init_method=simple_init
    )

    model.set_shared_expert_weights(
        up_proj_weight=shared_up_proj_weight,
        down_proj_weight=shared_down_proj_weight,
        up_proj_bias=shared_up_proj_bias,
        down_proj_bias=shared_down_proj_bias,
        weighted_sum=True,
        activation_fn=torch.nn.functional.gelu
    )

    assert model.shared_up_proj_weight is not None, f"Shared up proj weight not set on rank {rank}"
    assert model.shared_down_proj_weight is not None, f"Shared down proj weight not set on rank {rank}"
    assert model.shared_expert_weighted_sum == True, f"Weighted sum not set correctly on rank {rank}"
    
    print(f"Rank {rank}: Shared expert setup test passed!")

    dist.destroy_process_group()


def run_distributed_shared_expert_weighted_sum_test(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )

    model = MegaBlocksMoeMLPWithSharedExpert()

    hidden_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def simple_init(tensor):
        torch.nn.init.xavier_uniform_(tensor)

    shared_up_proj_weight, shared_down_proj_weight, _, _ = create_shared_expert_weights(
        hidden_size=hidden_size,
        shared_expert_hidden_size=96,
        device=torch.device(device),
        dtype=torch.float32,
        init_method=simple_init
    )

    model.set_shared_expert_weights(
        up_proj_weight=shared_up_proj_weight,
        down_proj_weight=shared_down_proj_weight,
        weighted_sum=False,
        activation_fn=torch.nn.functional.relu
    )

    assert model.shared_up_proj_weight is not None, f"Shared up proj weight not set on rank {rank}"
    assert model.shared_down_proj_weight is not None, f"Shared down proj weight not set on rank {rank}"
    assert model.shared_expert_weighted_sum == False, f"Weighted sum not set correctly on rank {rank}"
    assert model.shared_activation_fn == torch.nn.functional.relu, f"Activation function not set correctly on rank {rank}"
    
    print(f"Rank {rank}: Weighted sum setup test passed!")

    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_shared_expert_distributed_functionality(world_size):
    if world_size == 1:
        # Single process test
        model = MegaBlocksMoeMLPWithSharedExpert()
        
        hidden_size = 128
        shared_expert_hidden_size = 192
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def simple_init(tensor):
            torch.nn.init.xavier_uniform_(tensor)

        shared_up_proj_weight, shared_down_proj_weight, shared_up_proj_bias, shared_down_proj_bias = create_shared_expert_weights(
            hidden_size=hidden_size,
            shared_expert_hidden_size=shared_expert_hidden_size,
            device=torch.device(device),
            dtype=torch.float32,
            init_method=simple_init
        )

        model.set_shared_expert_weights(
            up_proj_weight=shared_up_proj_weight,
            down_proj_weight=shared_down_proj_weight,
            up_proj_bias=shared_up_proj_bias,
            down_proj_bias=shared_down_proj_bias,
            weighted_sum=True,
            activation_fn=torch.nn.functional.gelu
        )

        assert model.shared_up_proj_weight is not None, "Shared up proj weight not set"
        assert model.shared_down_proj_weight is not None, "Shared down proj weight not set"
        assert model.shared_expert_weighted_sum == True, "Weighted sum not set correctly"
        
        print("Single process shared expert setup test passed!")
    else:
        # Multi-process test
        mp.spawn(run_distributed_shared_expert_test, args=(world_size,), nprocs=world_size, join=True)
        print("Multi-process shared expert test completed successfully!")


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_shared_expert_distributed_weighted_sum(world_size):
    if world_size == 1:
        # Single process test
        model = MegaBlocksMoeMLPWithSharedExpert()

        hidden_size = 64
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def simple_init(tensor):
            torch.nn.init.xavier_uniform_(tensor)

        shared_up_proj_weight, shared_down_proj_weight, _, _ = create_shared_expert_weights(
            hidden_size=hidden_size,
            shared_expert_hidden_size=96,
            device=torch.device(device),
            dtype=torch.float32,
            init_method=simple_init
        )

        model.set_shared_expert_weights(
            up_proj_weight=shared_up_proj_weight,
            down_proj_weight=shared_down_proj_weight,
            weighted_sum=False,
            activation_fn=torch.nn.functional.relu
        )

        assert model.shared_up_proj_weight is not None, "Shared up proj weight not set"
        assert model.shared_down_proj_weight is not None, "Shared down proj weight not set"
        assert model.shared_expert_weighted_sum == False, "Weighted sum not set correctly"
        assert model.shared_activation_fn == torch.nn.functional.relu, "Activation function not set correctly"
        
        print("Single process weighted sum setup test passed!")
    else:
        # Multi-process test
        mp.spawn(run_distributed_shared_expert_weighted_sum_test, args=(world_size,), nprocs=world_size, join=True)
        print("Multi-process shared expert weighted sum test completed successfully!")


def test_shared_expert_single_process():
    model = MegaBlocksMoeMLPWithSharedExpert()
    
    assert model.shared_up_proj_weight is None
    assert model.shared_down_proj_weight is None
    assert hasattr(model, 'set_shared_expert_weights')
    
    print("Single process shared expert basic test passed!")


if __name__ == "__main__":
    test_shared_expert_single_process()
    print("Single process test passed!")
    
    os.environ['WORLD_SIZE'] = '2'
    test_shared_expert_distributed_functionality()
    print("Distributed functionality test passed!")
    
    test_shared_expert_distributed_weighted_sum()
    print("Distributed weighted sum test passed!")