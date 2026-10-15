import copy
from dataclasses import dataclass

import kernels
import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

transformers = pytest.importorskip("transformers")
try:
    from transformers import (
        DeformableDetrConfig,
        DeformableDetrForObjectDetection,
        GroundingDinoConfig,
        GroundingDinoForObjectDetection,
        RTDetrConfig,
        RTDetrForObjectDetection,
    )
except ValueError as e:
    # Older Transformers releases construct their kernel mapping without a
    # version at import time, which recent `kernels` releases reject.
    pytest.skip(
        f"Transformers is incompatible with kernels: {e}", allow_module_level=True
    )


LAYER = "MultiScaleDeformableAttention"


@dataclass(frozen=True)
class LoadedLayer:
    """The layer of the kernel, resolved like Transformers does (LOCAL_KERNELS in CI)."""

    layer_name: str

    def load(self):
        kernel = kernels.get_kernel("kernels-community/deformable-detr", version=1)
        return getattr(kernel.layers, self.layer_name)


def resnet_config():
    return transformers.ResNetConfig(
        embedding_size=8,
        hidden_sizes=[8, 16, 16, 16],
        depths=[1, 1, 1, 1],
        out_features=["stage2", "stage3", "stage4"],
    )


COMMON = dict(
    d_model=32,
    encoder_layers=1,
    decoder_layers=2,
    encoder_attention_heads=2,
    decoder_attention_heads=2,
    encoder_ffn_dim=32,
    decoder_ffn_dim=32,
    num_queries=12,
    dropout=0.0,
    attention_dropout=0.0,
    activation_dropout=0.0,
)
DETR_COMMON = dict(
    COMMON,
    backbone_config=resnet_config(),
    use_timm_backbone=False,
    use_pretrained_backbone=False,
    num_feature_levels=3,
    encoder_n_points=2,
    decoder_n_points=2,
)


def deformable_detr():
    return DeformableDetrForObjectDetection(DeformableDetrConfig(**DETR_COMMON))


def deformable_detr_two_stage():
    # Two-stage with box refinement uses 4-coordinate reference points.
    return DeformableDetrForObjectDetection(
        DeformableDetrConfig(
            **DETR_COMMON,
            two_stage=True,
            with_box_refine=True,
            two_stage_num_proposals=12,
        )
    )


def rt_detr():
    backbone = transformers.RTDetrResNetConfig(
        embedding_size=8,
        hidden_sizes=[8, 16, 16, 16],
        depths=[1, 1, 1, 1],
        out_features=["stage2", "stage3", "stage4"],
    )
    return RTDetrForObjectDetection(
        RTDetrConfig(
            backbone_config=backbone,
            d_model=32,
            encoder_hidden_dim=32,
            encoder_in_channels=[16, 16, 16],
            decoder_in_channels=[32, 32, 32],
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            decoder_layers=2,
            num_queries=12,
            num_denoising=0,
            decoder_n_points=2,
            decoder_attention_heads=2,
            dropout=0.0,
        )
    )


def grounding_dino():
    text = transformers.BertConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    return GroundingDinoForObjectDetection(
        GroundingDinoConfig(
            **DETR_COMMON,
            text_config=text,
            max_text_len=16,
            fusion_dropout=0.0,
        )
    )


# Only Deformable DETR runs in kernels_ci, and only in eager mode: compiling
# even this tiny model takes over 30s per test. torch.compile is covered in
# kernels_ci by the layer tests in test_compile.py.
MODELS = [
    pytest.param(deformable_detr, id="deformable_detr", marks=pytest.mark.kernels_ci),
    pytest.param(deformable_detr_two_stage, id="deformable_detr_two_stage"),
    pytest.param(rt_detr, id="rt_detr"),
    pytest.param(grounding_dino, id="grounding_dino"),
]
COMPILE_MODELS = [pytest.param(*m.values, id=m.id) for m in MODELS]

# (dtype of the model, autocast to bfloat16)
PRECISIONS = [
    pytest.param(torch.float32, False, id="fp32"),
    pytest.param(torch.bfloat16, False, id="bf16"),
    pytest.param(torch.float32, True, id="autocast_bf16"),
]


def build(factory):
    torch.manual_seed(0)
    model = factory()
    # Some models initialise the deformable attention projections so that its
    # output is (close to) zero, which would hide differences in the kernel.
    for module in model.modules():
        if type(module).__name__.endswith("MultiscaleDeformableAttention"):
            for param in module.parameters():
                torch.nn.init.normal_(param, std=0.05)
    return model.cuda().eval()


def kernelize(model):
    # The entry point users take. With LOCAL_KERNELS set, Transformers' own
    # kernel mapping (with its pinned version) resolves to the local build.
    model.set_use_kernels(True)
    layers = [m for m in model.modules() if type(m).__name__ == LAYER]
    assert layers, f"Transformers must expose `{LAYER}` for kernelization"
    expected = LoadedLayer(LAYER).load().forward
    for layer in layers:
        assert layer.forward.__func__ is expected, (
            "Test must not silently use a fallback"
        )
    return model


def inputs(model, dtype, batch=2, size=64):
    generator = torch.Generator(device="cuda").manual_seed(0)
    pixel_values = torch.randn(
        batch, 3, size, size, device="cuda", generator=generator, dtype=dtype
    )
    # Padding in the last image exercises the masked value path.
    pixel_mask = torch.ones(batch, size, size, dtype=torch.long, device="cuda")
    pixel_mask[-1, :, size * 5 // 8 :] = 0
    kwargs = dict(pixel_values=pixel_values, pixel_mask=pixel_mask)
    if isinstance(model, GroundingDinoForObjectDetection):
        kwargs["input_ids"] = torch.randint(
            1, 64, (batch, 8), device="cuda", generator=generator
        )
    return kwargs


def run(model, kwargs, autocast, backward=False):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        out = model(**kwargs)
    outputs = [out.logits, out.pred_boxes]
    if backward:
        # Masked logits are -inf in Grounding DINO.
        sum(
            o.float().nan_to_num(0.0, 0.0, 0.0).square().mean() for o in outputs
        ).backward()
    return outputs


def assert_close(actual, expected, dtype, autocast=False):
    low = dtype == torch.bfloat16 or autocast
    torch.testing.assert_close(
        actual,
        expected,
        equal_nan=True,
        rtol=3e-2 if low else 1e-4,
        atol=3e-2 if low else 1e-4,
    )


def capture_layer_calls(model, kwargs, autocast):
    """The arguments of every call of the deformable attention layer."""
    calls = []
    handles = [
        m.register_forward_hook(lambda m, args, out: calls.append(args))
        for m in model.modules()
        if type(m).__name__ == LAYER
    ]
    try:
        with torch.no_grad():
            run(model, kwargs, autocast)
    finally:
        for handle in handles:
            handle.remove()
    assert calls
    return calls


def layer_forward_backward(forward, args, autocast, dtype=None):
    """Output and value/attention weight gradients of one layer call."""
    value, sampling_loc, attn_weight = (
        args[i].detach().to(dtype or args[i].dtype).requires_grad_() for i in (0, 4, 5)
    )
    call = (value, args[1], args[2], args[3], sampling_loc, attn_weight, args[6])
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        out = forward(*call)
    out.float().square().mean().backward()
    # The sampling location gradient is piecewise and can differ near pixel
    # boundaries in reduced precision, see test_deformable_detr.py.
    return [out, value.grad, attn_weight.grad]


@pytest.mark.parametrize("factory", MODELS)
@pytest.mark.parametrize("dtype, autocast", PRECISIONS)
def test_layer_matches_transformers(factory, dtype, autocast):
    """Forward and backward of every layer call, on the inputs the model produces."""
    if factory is grounding_dino and dtype == torch.bfloat16:
        pytest.skip(
            "Transformers' Grounding DINO does not run in bfloat16 (dtype mismatch in its text fusion)"
        )
    model = build(factory).to(dtype)
    torch_layer = next(m for m in model.modules() if type(m).__name__ == LAYER)
    kernel_layer = LoadedLayer(LAYER).load()()

    def torch_forward(*args):
        return type(torch_layer).forward(torch_layer, *args)

    for args in capture_layer_calls(model, inputs(model, dtype), autocast):
        actual = layer_forward_backward(kernel_layer, args, autocast)
        expected = layer_forward_backward(torch_forward, args, autocast)
        assert actual[0].dtype == expected[0].dtype
        if dtype == torch.float32:
            assert_close(actual, expected, dtype, autocast)
            continue
        # Both implementations compute in bfloat16 and round differently.
        # Compare their errors against a float32 computation instead.
        exact = layer_forward_backward(
            torch_forward, args, autocast, dtype=torch.float32
        )
        for a, e, x in zip(actual, expected, exact):
            kernel_error = (a.float() - x.float()).abs().max().item()
            torch_error = (e.float() - x.float()).abs().max().item()
            scale = x.float().abs().max().item()
            eps = torch.finfo(dtype).eps
            assert kernel_error <= 2 * torch_error + 2 * eps * scale, (
                kernel_error,
                torch_error,
            )


@pytest.mark.parametrize("factory", MODELS)
@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "autocast_bf16"])
@torch.no_grad()
def test_model_matches_transformers(factory, autocast):
    reference = build(factory)
    model = kernelize(copy.deepcopy(reference))
    kwargs = inputs(model, torch.float32)
    assert_close(
        run(model, kwargs, autocast),
        run(reference, kwargs, autocast),
        torch.float32,
        autocast,
    )


def compile_model(model):
    compiled = copy.deepcopy(model)
    compiled.forward = torch.compile(compiled.forward, fullgraph=not model.training)
    return compiled


def max_error(actual, expected):
    return max(
        (a.float() - e.float()).nan_to_num(0.0, 0.0, 0.0).abs().max().item()
        for a, e in zip(actual, expected)
    )


def assert_compile_matches_baseline(kernel_results, torch_results):
    """Compiling the kernelized model must not deviate more than compiling Transformers' own.

    Compiled and eager results of these models can differ substantially, even
    without the kernel: Inductor's fused reductions round differently, and
    under autocast this can flip the top-k query selection of the two-stage
    models. So compare the deviation against the same model without the kernel.
    """
    (kernel_compiled, kernel_eager), (torch_compiled, torch_eager) = (
        kernel_results,
        torch_results,
    )
    for i, results in enumerate(
        zip(kernel_compiled, kernel_eager, torch_compiled, torch_eager)
    ):
        # Compiled training returns zero gradients for some parameters that
        # eager mode leaves as None (with or without the kernel).
        like = next((t for t in results if t is not None), None)
        if like is None:
            continue
        kc, ke, tc, te = (torch.zeros_like(like) if t is None else t for t in results)
        kernel_error = max_error([kc], [ke])
        torch_error = max_error([tc], [te])
        scale = ke.float().nan_to_num(0.0, 0.0, 0.0).abs().max().item()
        assert kernel_error <= 2 * torch_error + 1e-5 * max(scale, 1.0), (
            i,
            kernel_error,
            torch_error,
        )


@pytest.mark.parametrize("factory", COMPILE_MODELS)
@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "autocast_bf16"])
@torch.no_grad()
@torch._dynamo.config.patch(error_on_recompile=True)
def test_model_compile_inference(factory, autocast):
    # bfloat16 models are covered by autocast: Inductor cannot compile the
    # position embeddings of these models in bfloat16 (`isinf` in Triton).
    reference = build(factory)
    kwargs = inputs(reference, torch.float32)
    results = []
    for eager in (kernelize(copy.deepcopy(reference)), reference):
        compiled = compile_model(eager)
        try:
            # The second call reuses the compiled graph.
            for _ in range(2):
                outputs = (
                    run(compiled, kwargs, autocast),
                    run(eager, kwargs, autocast),
                )
        finally:
            torch._dynamo.reset()
        results.append(outputs)
    assert_compile_matches_baseline(*results)


def graph_break_count(model, kwargs, autocast):
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            return torch._dynamo.explain(model.forward)(**kwargs).graph_break_count
    finally:
        torch._dynamo.reset()


def run_training(model, kwargs, autocast):
    model.zero_grad(set_to_none=True)
    outputs = run(model, kwargs, autocast, backward=True)
    return outputs + [p.grad for p in model.parameters()]


# Rounding flips the query selection of the tiny Grounding DINO in training
# mode: even in eager mode without the kernel, its float32 gradients differ
# from float64 ones by more than their own magnitude, so they cannot be
# compared. The layer forward and backward on its inputs are covered by
# test_layer_matches_transformers.
TRAINING_MODELS = [m for m in COMPILE_MODELS if m.id != "grounding_dino"]


@pytest.mark.parametrize("factory", TRAINING_MODELS)
@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "autocast_bf16"])
def test_model_compile_training(factory, autocast):
    # In training mode the encoder layers check `torch.isfinite(...).all()` to
    # clamp their output, a data-dependent branch, so these models cannot be
    # compiled with fullgraph=True even without the kernel. The graph breaks
    # also make Dynamo recompile Transformers' output-capturing wrapper, which
    # is called with keyword arguments by the model and positional ones by
    # the resumed frames, so error_on_recompile cannot be used either. Check
    # that the kernel adds no graph breaks and that gradients deviate no more
    # than without the kernel.
    reference = build(factory).train()
    kernelized = kernelize(copy.deepcopy(reference))
    kwargs = inputs(reference, torch.float32)
    assert graph_break_count(kernelized, kwargs, autocast) == graph_break_count(
        reference, kwargs, autocast
    )

    results = []
    for eager in (kernelized, reference):
        compiled = compile_model(eager)
        try:
            for _ in range(2):
                outputs = (
                    run_training(compiled, kwargs, autocast),
                    run_training(eager, kwargs, autocast),
                )
        finally:
            torch._dynamo.reset()
        results.append(outputs)
    assert_compile_matches_baseline(*results)


@pytest.mark.kernels_ci
@torch.no_grad()
def test_model_batch_larger_than_im2col_step():
    # Transformers always passes im2col_step=64; the batch need not divide it.
    reference = build(deformable_detr)
    model = kernelize(copy.deepcopy(reference))
    kwargs = inputs(model, torch.float32, batch=66, size=32)
    assert_close(
        run(model, kwargs, False), run(reference, kwargs, False), torch.float32
    )


@pytest.mark.kernels_ci
@pytest.mark.parametrize(
    "mode",
    [
        kernels.Mode.INFERENCE | kernels.Mode.TORCH_COMPILE,
        kernels.Mode.TRAINING | kernels.Mode.TORCH_COMPILE,
    ],
    ids=["inference", "training"],
)
def test_kernelize_torch_compile_mode(mode):
    # kernels falls back to the original layer under Mode.TORCH_COMPILE unless
    # the kernel layer declares that it supports torch.compile.
    model = build(deformable_detr).train(kernels.Mode.TRAINING in mode)
    with kernels.use_kernel_mapping(
        {LAYER: {"cuda": LoadedLayer(LAYER)}}, inherit_mapping=False
    ):
        kernels.kernelize(model, mode=mode, device="cuda")
    expected = LoadedLayer(LAYER).load().forward
    layers = [m for m in model.modules() if type(m).__name__ == LAYER]
    assert layers and all(layer.forward.__func__ is expected for layer in layers)
