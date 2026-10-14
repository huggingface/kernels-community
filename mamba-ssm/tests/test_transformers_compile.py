import copy
from dataclasses import dataclass

import kernels
import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@dataclass(frozen=True)
class LoadedLayer:
    """Resolve through get_kernel so LOCAL_KERNELS also works before publication."""

    layer_name: str

    def load(self):
        kernel = kernels.get_kernel("kernels-community/mamba-ssm", version=4)
        return getattr(kernel.layers, self.layer_name)


# Mamba2 Triton compilation is expensive; its model cases and the direct SSD tests
# in test_compile.py run in the full suite, outside kernels_ci.
@pytest.fixture(params=[pytest.param("mamba", marks=pytest.mark.kernels_ci), "mamba2"])
def model(request):
    transformers = pytest.importorskip("transformers")
    common = dict(
        vocab_size=64, hidden_size=32, state_size=16, num_hidden_layers=1,
        expand=2, conv_kernel=4, use_bias=True,
    )
    if request.param == "mamba":
        config = transformers.MambaConfig(**common)
        cls = transformers.MambaForCausalLM
    else:
        config = transformers.Mamba2Config(
            **common, num_heads=8, head_dim=8, n_groups=1, chunk_size=16,
        )
        cls = transformers.Mamba2ForCausalLM
    torch.manual_seed(0)
    return cls(config).cuda()


def kernelize(model):
    kernel = kernels.get_kernel("kernels-community/mamba-ssm", version=4)
    mapping = {
        name: {"cuda": LoadedLayer(name)}
        for name in kernel.layers.__all__
    }
    # Deliberately do not pass TORCH_COMPILE: users compile after kernelization.
    mode = kernels.Mode.TRAINING if model.training else kernels.Mode.INFERENCE
    with kernels.use_kernel_mapping(mapping, inherit_mapping=False):
        # Leave unrelated layers such as SiLU on their normal Torch implementation.
        kernels.kernelize(model, mode=mode)
    funcs = [fn for module in model.modules()
             for fn in getattr(module, "_kernel_funcs", {}).values()]
    assert funcs, "Transformers must expose the Mamba Hub kernel integration"
    for fn in funcs:
        expected = getattr(kernel.layers, type(fn).kernel_layer_name).forward
        assert fn.forward.__func__ is expected, "Test must not silently use a fallback"
    return model


def assert_close(actual, expected, dtype):
    torch.testing.assert_close(
        actual, expected, rtol=3e-2 if dtype == torch.bfloat16 else 2e-3,
        atol=3e-3 if dtype == torch.bfloat16 else 2e-5,
    )


@torch._dynamo.config.patch(error_on_recompile=True)
def check_compiled_training(model, forward, dtype):
    eager = kernelize(model.train())
    compiled = copy.deepcopy(eager)
    compiled.forward = torch.compile(compiled.forward, fullgraph=True)
    try:
        # Reuse the compiled callable to exercise cached launches as well as autotuning.
        for _ in range(2):
            ids = torch.randint(0, 64, (2, 33), device="cuda")
            compiled.zero_grad(set_to_none=True)
            eager.zero_grad(set_to_none=True)

            actual = forward(compiled, ids)
            actual.float().square().mean().backward()

            expected = forward(eager, ids)
            expected.float().square().mean().backward()

            assert_close(actual, expected, dtype)
            for (name, param), (_, ref) in zip(compiled.named_parameters(), eager.named_parameters()):
                assert param.grad is not None and ref.grad is not None, name
                assert_close(param.grad, ref.grad, dtype)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_cache", [False, True])
def test_transformers_compile_training(model, dtype, use_cache):
    # A cache exercises separate convolution and scan ops instead of the fused path.
    def forward(model, ids):
        return model(ids, use_cache=use_cache).logits

    check_compiled_training(model.to(dtype), forward, dtype)


def test_transformers_compile_training_autocast(model):
    # FP32 parameters with BF16 autocast exercise mixed bias/weight dtypes.
    def forward(model, ids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(ids, use_cache=False).logits

    check_compiled_training(model.float(), forward, torch.bfloat16)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_transformers_compile_cached_decode(model, dtype):
    eager = kernelize(model.to(dtype).eval())
    compiled = copy.deepcopy(eager)
    compiled.forward = torch.compile(compiled.forward, fullgraph=True)
    actual_cache = expected_cache = None
    try:
        # Prefill followed by repeated single-token updates of both cache tensors.
        for length in (33, 1, 1, 1):
            ids = torch.randint(0, 64, (2, length), device="cuda")
            actual = compiled(ids, cache_params=actual_cache, use_cache=True)
            expected = eager(ids, cache_params=expected_cache, use_cache=True)
            assert_close(actual.logits, expected.logits, dtype)
            actual_cache, expected_cache = actual.cache_params, expected.cache_params
            assert actual_cache is not None and expected_cache is not None
            for layer, ref in zip(actual_cache.layers, expected_cache.layers):
                for field in ("conv_states", "recurrent_states"):
                    states, ref_states = getattr(layer, field), getattr(ref, field)
                    assert states.keys() == ref_states.keys()
                    for key in states:
                        assert_close(states[key], ref_states[key], dtype)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_transformers_generate_compile_config(model, dtype):
    transformers = pytest.importorskip("transformers")
    eager = kernelize(model.to(dtype).eval())
    compiled = copy.deepcopy(eager)
    ids = torch.randint(0, 64, (2, 9), device="cuda")
    kwargs = dict(max_new_tokens=4, do_sample=False, output_logits=True, return_dict_in_generate=True)
    try:
        expected = eager.generate(ids, disable_compile=True, **kwargs)
        # Only a static cache opts generate into compiling: Mamba's default cache is a DynamicCache.
        actual = compiled.generate(
            ids, cache_implementation="static",
            compile_config=transformers.CompileConfig(fullgraph=True), **kwargs,
        )
        assert hasattr(compiled, "_compiled_call"), "generate must have compiled the decode steps"
        for step, (actual_logits, expected_logits) in enumerate(zip(actual.logits, expected.logits)):
            assert_close(actual_logits, expected_logits, dtype)
        torch.testing.assert_close(actual.sequences, expected.sequences)
    finally:
        torch._dynamo.reset()
