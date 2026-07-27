import torch
import torch.nn as nn

from .functional import gemm
from ._ops import ops
from ._pack import PackedWeight


class NVFP4Linear(nn.Module):
    can_torch_compile = True
    has_backward = False

    def forward(self, x):
        pw = PackedWeight(
            qweight=self.weight,
            sf=self.weight_sf,
            global_scale=self.weight_global_scale,
            n=self.out_features,
            k=self.in_features,
            sf_rowmajor=getattr(self, "weight_sf_rowmajor", None),
        )
        return gemm(pw, x)


def _ntokens(x):
    return x.numel() // x.shape[-1]


def _fused_gemv(module, x):
    return ops.nvfp4_gemv(
        x.reshape(-1, x.shape[-1]).to(torch.bfloat16),
        module.nvfp4_fused_qweight,
        module.nvfp4_fused_sf,
        module.nvfp4_fused_alpha,
    )


class NVFP4FusedSwiGLUMLP(nn.Module):
    can_torch_compile = True
    has_backward = False

    def forward(self, x):
        if _ntokens(x) <= 2 and hasattr(self, "nvfp4_fused_qweight"):
            x2 = x.reshape(-1, x.shape[-1]).to(torch.bfloat16)
            h = ops.nvfp4_gemv_swiglu(
                x2,
                self.nvfp4_fused_qweight,
                self.nvfp4_fused_sf,
                self.nvfp4_fused_alpha,
            )
            return self.down_proj(h.reshape(*x.shape[:-1], -1))
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class NVFP4GatedAttention(nn.Module):
    can_torch_compile = True
    has_backward = False

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        og = None
        if _ntokens(hidden_states) <= 2 and hasattr(self, "nvfp4_fused_qweight"):
            fused = _fused_gemv(self, hidden_states)
            q, k, v, *rest = torch.split(fused, self.nvfp4_fused_splits, dim=-1)
            og = rest[0] if rest else None
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        query_states = q.reshape(hidden_shape).transpose(1, 2)
        key_states = k.reshape(hidden_shape).transpose(1, 2)
        value_states = v.reshape(hidden_shape).transpose(1, 2)

        norm = getattr(self, "qk_norm", None)
        if norm is not None:
            query_states = norm(query_states) * getattr(self, "qk_scale_factor", 1.0)
            key_states = norm(key_states)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = self.nvfp4_rope(
                query_states, key_states, cos, sin
            )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, self.nvfp4_eager
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None),
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if og is None and getattr(self, "gate_proj", None) is not None:
            og = self.gate_proj(hidden_states)
        o = self.o_proj
        if og is not None and _ntokens(attn_output) <= 2 and _has_nvfp4(o):
            a2 = attn_output.reshape(-1, attn_output.shape[-1])
            og2 = og.reshape(a2.shape).contiguous()
            alpha = (1.0 / o.weight_global_scale.float()).reshape(1)
            y = ops.nvfp4_gemv_gated(a2, og2, o.weight, o.weight_sf_rowmajor, alpha)
            return y.reshape(*input_shape, -1), attn_weights
        if og is not None:
            attn_output = attn_output * torch.sigmoid(og.reshape(attn_output.shape))
        return o(attn_output), attn_weights


def _has_nvfp4(m):
    return m is not None and hasattr(m, "weight_sf_rowmajor")


def _is_silu(act):
    return isinstance(act, nn.SiLU) or type(act).__name__ == "SiLUActivation"


def _attach_fused(module, mods, interleave=False):
    def join(ts):
        if interleave:
            return torch.stack(ts, 1).reshape(-1, *ts[0].shape[1:]).contiguous()
        return torch.cat(ts, 0).contiguous()

    module.nvfp4_fused_qweight = join([m.weight for m in mods])
    module.nvfp4_fused_sf = join([m.weight_sf_rowmajor for m in mods])
    module.nvfp4_fused_alpha = join(
        [
            (1.0 / m.weight_global_scale.float())
            .reshape(1)
            .expand(m.out_features)
            .contiguous()
            for m in mods
        ]
    )
    module.nvfp4_fused_splits = [m.out_features for m in mods]


def _capture_helpers(cls, rope_fn):
    if getattr(cls, "nvfp4_eager", None) is not None:
        return
    fwd = cls.forward
    names = ("apply_rotary_pos_emb_interleave", "apply_rotary_pos_emb")
    rope = rope_fn or next(
        (fwd.__globals__[n] for n in names if n in fwd.__code__.co_names), None
    )
    cls.nvfp4_rope = staticmethod(rope) if rope is not None else None
    cls.nvfp4_eager = staticmethod(fwd.__globals__["eager_attention_forward"])


def _mark(cls, layer_name):
    if getattr(cls, "kernel_layer_name", None) is not None:
        return
    try:
        from kernels import use_kernel_forward_from_hub

        use_kernel_forward_from_hub(layer_name)(cls)
    except Exception:
        cls.kernel_layer_name = layer_name


def fuse_decode_projections(model, rope_fn=None):
    n = 0
    for mod in model.modules():
        qkv = [getattr(mod, p, None) for p in ("q_proj", "k_proj", "v_proj")]
        if (
            all(_has_nvfp4(p) for p in qkv)
            and hasattr(mod, "o_proj")
            and hasattr(mod, "layer_idx")
        ):
            gate = getattr(mod, "gate_proj", None)
            _attach_fused(mod, qkv + ([gate] if _has_nvfp4(gate) else []))
            _capture_helpers(type(mod), rope_fn)
            _mark(type(mod), "NVFP4GatedAttention")
            n += 1
            continue
        gu = [getattr(mod, p, None) for p in ("gate_proj", "up_proj")]
        if (
            all(_has_nvfp4(p) for p in gu)
            and hasattr(mod, "down_proj")
            and _is_silu(getattr(mod, "act_fn", None))
            and gu[0].out_features == gu[1].out_features
        ):
            _attach_fused(mod, gu, interleave=True)
            _mark(type(mod), "NVFP4SwiGLUMLP")
            n += 1
    return n


def kernel_mapping():
    from pathlib import Path

    from kernels import LocalLayerRepository

    root = next(
        p for p in Path(__file__).resolve().parents if (p / "build.toml").exists()
    )
    return {
        "NVFP4GatedAttention": {
            "cuda": LocalLayerRepository(repo_path=root, layer_name="NVFP4GatedAttention")
        },
        "NVFP4SwiGLUMLP": {
            "cuda": LocalLayerRepository(repo_path=root, layer_name="NVFP4FusedSwiGLUMLP")
        },
    }
