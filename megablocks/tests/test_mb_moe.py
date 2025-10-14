import torch
import megablocks

def test_import():
    """Simple test to check if the module can be imported."""
    print("megablocks_moe module imported successfully.")
    print("Available functions:", dir(megablocks))

    expected_functions = [
        "Arguments", "MLP", "MoE", "ParallelDroplessMLP", "ParallelMLP",
        "SparseGLU", "SparseMLP", "argsort",
        "backend", "cumsum", "dMoE", "exclusive_cumsum",
        "get_load_balancing_loss", "grouped_gemm_util", "histogram",
        "inclusive_cumsum", "indices", "layers", "ops", "replicate_backward",
        "replicate_forward", "sort", "torch"
    ]

    # Check if all expected functions are available
    for func in expected_functions:
        assert func in dir(megablocks), f"Missing function: {func}" 

# exclusive_cumsum
def test_exclusive_cumsum():
    """Test exclusive cumulative sum."""
    x = torch.tensor([1, 2, 3, 4], dtype=torch.int16).cuda()
    out = torch.empty_like(x)
    megablocks.exclusive_cumsum(x, 0, out)
    expected = torch.tensor([0, 1, 3, 6], dtype=torch.float32).cuda()
    assert torch.equal(out, expected), f"Expected {expected}, got {out}"
    print("cumsum output:", out)

# inclusive_cumsum
def test_inclusive_cumsum():
    """Test inclusive cumulative sum."""
    x = torch.tensor([1, 2, 3, 4], dtype=torch.int16).cuda()
    out = torch.empty_like(x)
    megablocks.inclusive_cumsum(x, dim=0, out=out)
    expected = torch.tensor([1, 3, 6, 10], dtype=torch.float32).cuda()
    assert torch.equal(out, expected), f"Expected {expected}, got {out}"

# histogram
def test_histogram():
    """Test histogram operation."""
    x = torch.tensor([0, 1, 1, 2, 2, 2], dtype=torch.int16).cuda()
    num_bins = 3
    hist = megablocks.histogram(x, num_bins)
    expected_hist = torch.tensor([1, 2, 3], dtype=torch.int32).cuda()
    assert torch.equal(hist, expected_hist), f"Expected {expected_hist}, got {hist}"
