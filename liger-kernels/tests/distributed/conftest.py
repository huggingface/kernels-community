import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist


@pytest.fixture(scope="session")
def distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))

    yield torch.device("cuda", local_rank)

    dist.destroy_process_group()


@pytest.fixture(scope="session")
def meshes(distributed):
    return {}
