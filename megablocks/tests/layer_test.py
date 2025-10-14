import torch

from collections import namedtuple


def test_megablocks_moe_mlp_import():
    """Test if MegaBlocksMoeMLP can be imported."""
    from megablocks.layers import MegaBlocksMoeMLP

    assert MegaBlocksMoeMLP is not None, "MegaBlocksMoeMLP import failed."


def test_megablocks_moe_mlp_functionality():
    """Test the functionality of MegaBlocksMoeMLP."""
    from megablocks.layers import MegaBlocksMoeMLP

    # Create a simple instance of MegaBlocksMoeMLP
    model = MegaBlocksMoeMLP()

    # add experts attribute to the model
    model.experts = namedtuple(
        "Experts",
        [
            "gate_up_proj",
            "gate_down_proj",
            "down_proj",
            "hidden_size",
        ],
    )

    num_experts = 128
    hidden_size = 1152
    intermediate_size = 3072

    # Shorter names for reading convenience
    ne, hs, isz = num_experts, hidden_size, intermediate_size

    model.router = torch.nn.Linear(hs, ne).cuda()
    model.router.weight.data.fill_(1)

    e = model.experts
    e.gate_up_proj = torch.nn.Parameter(torch.ones(ne, hs, isz, device="cuda"))
    e.gate_up_proj_bias = torch.nn.Parameter(torch.zeros(ne, isz, device="cuda"))
    e.down_proj = torch.nn.Parameter(torch.ones(ne, 1536, hs, device="cuda"))
    e.down_proj_bias = torch.nn.Parameter(torch.zeros(ne, hs, device="cuda"))
    e.hidden_size = hs

    # Create dummy input data
    x = torch.randn(1, 1, 1152).to(torch.device("cuda"))
    output, expert_weights_out = model(x)

    # print("Output shape:", output.shape)
    assert output.shape == (1, 1, 1152), "Output shape mismatch."
