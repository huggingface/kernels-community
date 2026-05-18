"""Runtime policy hooks for live attention throttling.

This is intentionally small and deterministic. The policy can clamp an
existing sliding-window request before CSR construction / kernel launch. It can
also opt into converting dense causal attention to sliding-window attention,
but that is disabled by default because it changes model semantics.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import os
from typing import Any, Mapping


_MIB = 1024 * 1024


def _parse_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _parse_int(value: object, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    return int(value)


def _parse_int_tuple(value: object) -> tuple[int, ...] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",")]
        items = [int(part) for part in parts if part]
    else:
        items = [int(part) for part in value]  # type: ignore[arg-type]
    if not items:
        return None
    if any(item <= 0 for item in items):
        raise ValueError(f"layer windows must be positive integers, got {value!r}")
    return tuple(items)


def _parse_float(value: object, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _parse_bytes(value: object, default: int) -> int:
    if value is None or value == "":
        return default
    text = str(value).strip().lower()
    scale = 1
    for suffix, multiplier in (
        ("gib", 1024**3),
        ("gb", 1000**3),
        ("mib", 1024**2),
        ("mb", 1000**2),
        ("kib", 1024),
        ("kb", 1000),
    ):
        if text.endswith(suffix):
            scale = multiplier
            text = text[: -len(suffix)]
            break
    return int(float(text) * scale)


@dataclass(frozen=True)
class RuntimePolicy:
    """Policy for request-time attention throttling."""

    mode: str = "off"
    max_window: int | None = None
    min_window: int = 128
    reserve_bytes: int = 512 * _MIB
    utilization: float = 0.85
    allow_dense_to_window: bool = False
    fail_closed: bool = False
    align_to_block: bool = True
    layer_windows: tuple[int, ...] | None = None

    @property
    def enabled(self) -> bool:
        return self.mode not in {"", "0", "off", "disabled", "none"}

    @classmethod
    def from_env(cls) -> "RuntimePolicy":
        raw_mode = os.environ.get("HYDRA_POLICY", "off").strip().lower()
        if raw_mode in {"1", "true", "yes", "on", "enabled"}:
            raw_mode = "adaptive"
        return cls(
            mode=raw_mode,
            max_window=_parse_int(os.environ.get("HYDRA_MAX_WINDOW")),
            min_window=int(os.environ.get("HYDRA_MIN_WINDOW", "128")),
            reserve_bytes=_parse_bytes(
                os.environ.get("HYDRA_RESERVE_BYTES"),
                int(float(os.environ.get("HYDRA_RESERVE_MB", "512")) * _MIB),
            ),
            utilization=_parse_float(os.environ.get("HYDRA_UTILIZATION"), 0.85),
            allow_dense_to_window=_parse_bool(
                os.environ.get("HYDRA_ALLOW_DENSE_THROTTLE"), False
            ),
            fail_closed=_parse_bool(os.environ.get("HYDRA_FAIL_CLOSED"), False),
            align_to_block=_parse_bool(os.environ.get("HYDRA_ALIGN_WINDOW"), True),
            layer_windows=_parse_int_tuple(
                os.environ.get("HYDRA_LAYER_WINDOWS")
            ),
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RuntimePolicy":
        return cls(
            mode=str(data.get("mode", "adaptive")).strip().lower(),
            max_window=_parse_int(data.get("max_window")),
            min_window=int(data.get("min_window", 128)),
            reserve_bytes=_parse_bytes(data.get("reserve_bytes"), 512 * _MIB),
            utilization=float(data.get("utilization", 0.85)),
            allow_dense_to_window=_parse_bool(data.get("allow_dense_to_window"), False),
            fail_closed=_parse_bool(data.get("fail_closed"), False),
            align_to_block=_parse_bool(data.get("align_to_block"), True),
            layer_windows=_parse_int_tuple(data.get("layer_windows")),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyDecision:
    enabled: bool
    action: str
    reason: str
    requested_window: int | None
    effective_window: int | None
    estimated_bytes: int | None = None
    budget_bytes: int | None = None
    free_bytes: int | None = None
    total_bytes: int | None = None
    layer_idx: int | None = None
    layer_window: int | None = None

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


_POLICY_OVERRIDE: RuntimePolicy | None = None
_HISTORY: deque[PolicyDecision] = deque(maxlen=64)


def set_runtime_policy(policy: RuntimePolicy | Mapping[str, Any] | None) -> None:
    """Install a process-local policy override.

    Passing ``None`` returns policy selection to environment variables.
    """

    global _POLICY_OVERRIDE
    if policy is None:
        _POLICY_OVERRIDE = None
    elif isinstance(policy, RuntimePolicy):
        _POLICY_OVERRIDE = policy
    else:
        _POLICY_OVERRIDE = RuntimePolicy.from_mapping(policy)


def get_runtime_policy() -> RuntimePolicy:
    return _POLICY_OVERRIDE or RuntimePolicy.from_env()


def last_policy_decision() -> PolicyDecision | None:
    return _HISTORY[-1] if _HISTORY else None


def policy_history() -> list[dict[str, Any]]:
    return [item.to_mapping() for item in _HISTORY]


def _record(decision: PolicyDecision) -> PolicyDecision:
    if decision.enabled:
        _HISTORY.append(decision)
    return decision


def estimate_attention_bytes(
    *,
    batch_size: int,
    query_heads: int,
    query_tokens: int,
    key_tokens: int,
    head_dim: int,
    dtype_bytes: int,
    block_size: int,
    sliding_window: int | None,
    decode: bool = False,
) -> int:
    """Conservative estimate of extra bytes allocated by this attention call."""

    output = batch_size * query_heads * query_tokens * head_dim * dtype_bytes
    lse = batch_size * query_heads * query_tokens * 4
    if decode:
        return output + lse

    q_blocks = math.ceil(query_tokens / block_size)
    k_blocks = math.ceil(key_tokens / block_size)
    row_ptr = batch_size * query_heads * (q_blocks + 1) * 4
    seq_lens = batch_size * 4

    if sliding_window is None or sliding_window >= key_tokens:
        nnz_per_head = q_blocks * (q_blocks + 1) // 2
    else:
        blocks_per_row = min(k_blocks, math.ceil(sliding_window / block_size) + 1)
        nnz_per_head = q_blocks * blocks_per_row
    col_idx = batch_size * query_heads * nnz_per_head * 4
    return output + lse + row_ptr + col_idx + seq_lens


def _cuda_mem_info(device) -> tuple[int | None, int | None]:
    if getattr(device, "type", None) != "cuda":
        return None, None
    try:
        import torch

        free, total = torch.cuda.mem_get_info(device)
        return int(free), int(total)
    except Exception:
        return None, None


def _budget(policy: RuntimePolicy, free_bytes: int | None) -> int | None:
    if free_bytes is None:
        return None
    usable = max(0, free_bytes - policy.reserve_bytes)
    return int(usable * policy.utilization)


def _align_window(window: int, block_size: int, policy: RuntimePolicy) -> int:
    window = max(policy.min_window, window)
    if policy.align_to_block:
        window = max(block_size, (window // block_size) * block_size)
    return window


def _layer_window_for(
    policy: RuntimePolicy, layer_idx: int | None, block_size: int
) -> int | None:
    if layer_idx is None or not policy.layer_windows:
        return None
    if layer_idx < 0:
        return None
    raw_window = policy.layer_windows[layer_idx % len(policy.layer_windows)]
    return _align_window(raw_window, block_size, policy)


def _fit_window_for_budget(
    *,
    budget_bytes: int,
    fixed_bytes: int,
    batch_size: int,
    query_heads: int,
    query_tokens: int,
    block_size: int,
    policy: RuntimePolicy,
) -> int:
    q_blocks = math.ceil(query_tokens / block_size)
    bytes_per_block_per_row = batch_size * query_heads * q_blocks * 4
    if bytes_per_block_per_row <= 0:
        return policy.min_window
    remaining = max(0, budget_bytes - fixed_bytes)
    # ``fixed_bytes`` is estimated with a one-token window, which still maps to
    # two CSR blocks per row: one reach block plus the diagonal block. Add any
    # remaining block budget to that floor, then subtract the diagonal to get
    # token-window reach.
    blocks_per_row = 2 + max(0, remaining // bytes_per_block_per_row)
    # CSR includes the diagonal block in addition to window reach.
    window_blocks = max(1, int(blocks_per_row) - 1)
    return _align_window(window_blocks * block_size, block_size, policy)


def apply_runtime_policy(
    q,
    k,
    v,
    *,
    sliding_window: int | None,
    block_size: int,
    head_dim: int,
    layer_idx: int | None = None,
) -> PolicyDecision:
    """Apply the active policy and return the effective sliding window."""

    policy = get_runtime_policy()
    requested = sliding_window
    if not policy.enabled:
        return PolicyDecision(False, "off", "policy disabled", requested, requested)

    batch_size, query_heads, query_tokens, dim = q.shape
    key_tokens = k.shape[2]
    decode = query_tokens == 1
    dtype_bytes = q.element_size()
    free, total = _cuda_mem_info(q.device)
    budget = _budget(policy, free)
    effective = requested
    layer_window = _layer_window_for(policy, layer_idx, block_size)
    action = "pass"
    reason = "within policy"

    if policy.max_window is not None:
        max_window = _align_window(policy.max_window, block_size, policy)
        if effective is None and policy.allow_dense_to_window:
            effective = max_window
            action = "dense_to_window"
            reason = f"dense attention converted to sw={effective}"
        elif effective is not None and effective > max_window:
            effective = max_window
            action = "clamp_max_window"
            reason = f"sliding_window clamped to max_window={effective}"

    if layer_window is not None:
        if effective is None and policy.allow_dense_to_window:
            effective = layer_window
            action = "layer_window"
            reason = f"dense attention converted to layer_window={effective}"
        elif effective is not None and effective > layer_window:
            effective = layer_window
            action = "layer_window"
            reason = f"sliding_window clamped to layer_window={effective}"

    estimate = estimate_attention_bytes(
        batch_size=batch_size,
        query_heads=query_heads,
        query_tokens=query_tokens,
        key_tokens=key_tokens,
        head_dim=dim,
        dtype_bytes=dtype_bytes,
        block_size=block_size,
        sliding_window=effective,
        decode=decode,
    )

    if policy.mode == "adaptive" and budget is not None and estimate > budget and not decode:
        fixed = estimate_attention_bytes(
            batch_size=batch_size,
            query_heads=query_heads,
            query_tokens=query_tokens,
            key_tokens=key_tokens,
            head_dim=dim,
            dtype_bytes=dtype_bytes,
            block_size=block_size,
            sliding_window=1,
            decode=False,
        )
        if effective is None and policy.allow_dense_to_window:
            effective = policy.max_window or key_tokens
        if effective is not None:
            fit_window = _fit_window_for_budget(
                budget_bytes=budget,
                fixed_bytes=fixed,
                batch_size=batch_size,
                query_heads=query_heads,
                query_tokens=query_tokens,
                block_size=block_size,
                policy=policy,
            )
            if fit_window < effective:
                effective = fit_window
                action = "budget_throttle"
                reason = f"estimated attention bytes exceeded budget; sw={effective}"
                estimate = estimate_attention_bytes(
                    batch_size=batch_size,
                    query_heads=query_heads,
                    query_tokens=query_tokens,
                    key_tokens=key_tokens,
                    head_dim=dim,
                    dtype_bytes=dtype_bytes,
                    block_size=block_size,
                    sliding_window=effective,
                    decode=False,
                )
        elif policy.fail_closed:
            decision = PolicyDecision(
                True,
                "reject",
                "dense attention exceeds policy budget and dense throttling is disabled",
                requested,
                effective,
                estimate,
                budget,
                free,
                total,
                layer_idx,
                layer_window,
            )
            _record(decision)
            raise MemoryError(decision.reason)

    if policy.fail_closed and budget is not None and estimate > budget:
        decision = PolicyDecision(
            True,
            "reject",
            "attention estimate still exceeds policy budget after throttling",
            requested,
            effective,
            estimate,
            budget,
            free,
            total,
            layer_idx,
            layer_window,
        )
        _record(decision)
        raise MemoryError(decision.reason)

    return _record(
        PolicyDecision(
            True,
            action,
            reason,
            requested,
            effective,
            estimate,
            budget,
            free,
            total,
            layer_idx,
            layer_window,
        )
    )
