import torch
import megablocks
from megablocks.layers import MegaBlocksMoeMLPWithSharedExpert, create_shared_expert_weights


def test_megablocks_moe_mlp_with_shared_expert_import():
    mlp = MegaBlocksMoeMLPWithSharedExpert()
    assert hasattr(mlp, 'shared_up_proj_weight')
    assert hasattr(mlp, 'shared_down_proj_weight')
    assert hasattr(mlp, 'set_shared_expert_weights')


def test_set_shared_expert_weights():
    mlp = MegaBlocksMoeMLPWithSharedExpert()
    
    hidden_size = 128
    shared_expert_hidden_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    up_proj_weight = torch.randn(shared_expert_hidden_size, hidden_size, device=device, dtype=dtype)
    down_proj_weight = torch.randn(hidden_size, shared_expert_hidden_size, device=device, dtype=dtype)
    up_proj_bias = torch.randn(shared_expert_hidden_size, device=device, dtype=dtype)
    down_proj_bias = torch.randn(hidden_size, device=device, dtype=dtype)
    
    mlp.set_shared_expert_weights(
        up_proj_weight=up_proj_weight,
        down_proj_weight=down_proj_weight,
        up_proj_bias=up_proj_bias,
        down_proj_bias=down_proj_bias,
        weighted_sum=True,
        activation_fn=torch.nn.functional.gelu
    )
    
    assert torch.equal(mlp.shared_up_proj_weight, up_proj_weight)
    assert torch.equal(mlp.shared_down_proj_weight, down_proj_weight)
    assert torch.equal(mlp.shared_up_proj_bias, up_proj_bias)
    assert torch.equal(mlp.shared_down_proj_bias, down_proj_bias)
    assert mlp.shared_expert_weighted_sum == True
    assert mlp.shared_activation_fn == torch.nn.functional.gelu


def test_create_shared_expert_weights():
    hidden_size = 128
    shared_expert_hidden_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    def init_method(tensor):
        torch.nn.init.xavier_uniform_(tensor)
    
    up_proj_weight, down_proj_weight, up_proj_bias, down_proj_bias = create_shared_expert_weights(
        hidden_size=hidden_size,
        shared_expert_hidden_size=shared_expert_hidden_size,
        device=device,
        dtype=dtype,
        init_method=init_method
    )
    
    assert up_proj_weight.shape == (shared_expert_hidden_size, hidden_size)
    assert down_proj_weight.shape == (hidden_size, shared_expert_hidden_size)
    assert up_proj_weight.device.type == device.type
    assert down_proj_weight.device.type == device.type
    assert up_proj_weight.dtype == dtype
    assert down_proj_weight.dtype == dtype
    assert up_proj_bias is None
    assert down_proj_bias is None


def test_shared_expert_weights_none_by_default():
    mlp = MegaBlocksMoeMLPWithSharedExpert()
    
    assert mlp.shared_up_proj_weight is None
    assert mlp.shared_down_proj_weight is None
    assert mlp.shared_up_proj_bias is None
    assert mlp.shared_down_proj_bias is None
    assert mlp.shared_expert_weighted_sum == False
    assert mlp.shared_activation_fn is None


def test_inheritance_from_megablocks_moe_mlp():
    mlp = MegaBlocksMoeMLPWithSharedExpert()
    
    from megablocks.layers import MegaBlocksMoeMLP
    assert isinstance(mlp, MegaBlocksMoeMLP)
    assert hasattr(mlp, 'forward')


def test_shared_expert_weights_custom_init():
    hidden_size = 64
    shared_expert_hidden_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16
    
    def custom_init(tensor):
        torch.nn.init.constant_(tensor, 0.5)
    
    def custom_output_init(tensor):
        torch.nn.init.constant_(tensor, 0.1)
    
    up_proj_weight, down_proj_weight, up_proj_bias, down_proj_bias = create_shared_expert_weights(
        hidden_size=hidden_size,
        shared_expert_hidden_size=shared_expert_hidden_size,
        device=device,
        dtype=dtype,
        init_method=custom_init,
        output_layer_init_method=custom_output_init
    )
    
    assert torch.all(up_proj_weight == 0.5)
    assert torch.all(down_proj_weight == 0.1)
    assert up_proj_weight.dtype == dtype
    assert down_proj_weight.dtype == dtype


def test_shared_expert_weights_dimensions():
    mlp = MegaBlocksMoeMLPWithSharedExpert()
    
    batch_size = 4
    seq_len = 16
    hidden_size = 128
    shared_expert_hidden_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    up_proj_weight = torch.randn(shared_expert_hidden_size, hidden_size, device=device)
    down_proj_weight = torch.randn(hidden_size, shared_expert_hidden_size, device=device)
    
    mlp.set_shared_expert_weights(
        up_proj_weight=up_proj_weight,
        down_proj_weight=down_proj_weight
    )
    
    x = torch.randn(seq_len, batch_size, hidden_size, device=device)
    
    expected_up_output_shape = (seq_len, batch_size, shared_expert_hidden_size)
    expected_down_output_shape = (seq_len, batch_size, hidden_size)
    
    assert up_proj_weight.shape[1] == x.shape[-1]
    assert down_proj_weight.shape[0] == x.shape[-1]