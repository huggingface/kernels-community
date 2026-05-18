from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")


def test_public_imports():
    import hydra

    assert callable(hydra.hydra)
    assert hydra.hydra_attention is hydra.hydra
    assert hydra.flash_attn_blackwell is hydra.hydra


def test_policy_defaults():
    from hydra.policy import RuntimePolicy

    policy = RuntimePolicy()
    assert policy.mode == "off"
    assert policy.enabled is False
