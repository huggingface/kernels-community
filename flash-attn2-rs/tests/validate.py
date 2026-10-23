# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "apache-tvm-ffi==0.1.14.post1",
#     "kernels==0.17.2",
#     "torch==2.13.0",
# ]
# ///
# Use a local build with LOCAL_KERNELS=kernels-community/flash-attn2-rs=result.
import torch
from kernels import get_kernel

SRC = get_kernel("kernels-community/flash-attn2-rs", version=1)
REF = get_kernel("kernels-community/flash-attn2", version=2)


def src_fwd(
    q,
    k,
    v,
    *,
    alibi_slopes=None,
    softmax_scale=None,
    is_causal=False,
    window_size_left=-1,
    window_size_right=-1,
    softcap=0.0,
):
    """The tvm-ffi op is positional-only; adapt the reference kwargs."""
    return SRC.fwd_explicit(
        q,
        k,
        v,
        alibi_slopes,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
    )


CASES = {
    "base": {},
    "bf16": {"dtype": torch.bfloat16},
    "causal": {"is_causal": True},
    "cross": {"sq": 512, "sk": 1024},
    "gqa": {"h": 8, "hk": 2},
    "headdim80": {"d": 80},
    "window": {"window_size_left": 128},
    "softcap": {"softcap": 30.0},
    "alibi": {"alibi": True, "is_causal": True},
    "combined": {"is_causal": True, "h": 8, "hk": 4, "d": 96, "dtype": torch.bfloat16},
}


def run(name: str, spec: dict) -> bool:
    spec = dict(spec)
    dtype = spec.pop("dtype", torch.float16)
    b = spec.pop("b", 2)
    h = spec.pop("h", 4)
    hk = spec.pop("hk", h)
    d = spec.pop("d", 128)
    sq = spec.pop("sq", 1024)
    sk = spec.pop("sk", sq)

    torch.manual_seed(0)
    q = torch.randn(b, sq, h, d, device="cuda", dtype=dtype)
    k = torch.randn(b, sk, hk, d, device="cuda", dtype=dtype)
    v = torch.randn(b, sk, hk, d, device="cuda", dtype=dtype)
    if spec.pop("alibi", False):
        spec["alibi_slopes"] = torch.rand(h, device="cuda", dtype=torch.float32)

    out_src, lse_src = src_fwd(q, k, v, **spec)
    out_ref, lse_ref = REF.fwd(q, k, v, **spec)[:2]
    torch.cuda.synchronize()

    out_atol = 2e-2 if dtype is torch.bfloat16 else 2e-3
    ok = True
    for label, got, want, atol in (
        ("out", out_src.float(), out_ref.float(), out_atol),
        ("lse", lse_src.float(), lse_ref.float(), 2e-3),
    ):
        err = (got - want).abs().max().item()
        good = torch.allclose(got, want, atol=atol, rtol=2.5e-2)
        ok &= good
        print(f"{name:>10} {label} max_abs_err={err:.3e} {'ok' if good else 'FAIL'}")
    return ok


def main() -> None:
    failed = [name for name, spec in CASES.items() if not run(name, spec)]
    if failed:
        raise SystemExit(f"FAILED: {', '.join(failed)}")
    print(f"all {len(CASES)} cases match kernels-community/flash-attn2")


if __name__ == "__main__":
    main()
