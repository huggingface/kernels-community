def test_import_cross_entropy():
    from flash_attn_triton.losses.cross_entropy import CrossEntropyLoss
    from flash_attn_triton.ops.triton.cross_entropy import cross_entropy_loss
    assert CrossEntropyLoss is not None

def test_import_layer_norm():
    from flash_attn_triton.ops.triton.layer_norm import layer_norm_fn
    assert layer_norm_fn is not None
