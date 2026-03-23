import pytest
import torch

from sonic_moe import count_cumsum


@pytest.mark.parametrize("size", [128, 1024, 4096, 2097152])
@pytest.mark.parametrize("num_experts", [4, 8, 256, 2048])
@pytest.mark.parametrize("do_cumsum", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.long])
def test_count_cumsum(size, num_experts, do_cumsum, dtype):
    x = torch.randint(0, num_experts, (size,), device="cuda", dtype=dtype)

    result = count_cumsum(x=x, E=num_experts, do_cumsum=do_cumsum)

    expected_count = x.view(-1).bincount(minlength=num_experts).to(torch.int32)

    if do_cumsum:
        count_output, cumsum_output = result
        assert torch.equal(count_output, expected_count)

        expected_cumsum = expected_count.cumsum(-1)
        assert torch.equal(cumsum_output, expected_cumsum)
    else:
        count_output = result
        assert torch.equal(count_output, expected_count)
