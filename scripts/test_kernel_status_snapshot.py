import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kernel_status_snapshot as snap  # noqa: E402

NOW = datetime(2026, 10, 27, tzinfo=timezone.utc)
TORCH = ["2.14", "2.15"]


def _kernel(**overrides):
    kernel = {
        "version": 1,
        "platforms": ["linux", "mac"],
        "builder": {"rev": "b2", "date": "2026-10-20T00:00:00Z", "behind_days": 7},
        "tree_sha": "t2",
        "ci_tests": 2,
        "hub": {"branch": "v1", "variants": [], "torch_versions": TORCH, "stable_abi": None, "noarch": False},
        "abi_check": {"result": "pass", "checked_at": "2026-10-27T02:00:00Z"},
    }
    kernel.update(overrides)
    return kernel


def _event(platform, result, finished, builder="b2", tree="t2", mode="release"):
    return {
        "kernel": "relu",
        "workflow": platform,
        "mode": mode,
        "run_id": 99,
        "finished_at": finished,
        "trigger": {"pr_number": None, "requested_by": None, "upload": True, "repo_prefix": "kernels-community"},
        "source": {"builder_rev": builder, "kernel_tree_sha": tree},
        "result": {
            "overall": result,
            "jobs": [{"id": 11, "name": "build-kernel (cuda, x86_64-linux)", "backend": "cuda",
                      "arch": "x86_64", "conclusion": result, "failed_step": "Build kernel" if result == "failure" else None}],
        },
    }


def _history(*events):
    return [snap.history_row(f"events/{i}.json", e) for i, e in enumerate(events)]


def test_latest_release_per_platform_drives_release_failed():
    history = _history(
        _event("linux", "success", "2026-10-01T00:00:00Z"),
        _event("linux", "failure", "2026-10-20T00:00:00Z"),
        _event("mac", "success", "2026-10-21T00:00:00Z"),
    )
    status = snap.add_health(_kernel(), "relu", TORCH, history)
    assert status["release"]["linux"]["result"] == "failure"
    assert status["released"]["finished_at"] == "2026-10-21T00:00:00Z"
    assert list(status["flags"]) == ["release-failed"]
    assert status["health"] == "error"


def test_platforms_no_longer_built_are_ignored():
    history = _history(_event("windows", "failure", "2026-10-20T00:00:00Z"), _event("linux", "success", "2026-10-21T00:00:00Z"))
    assert "windows" not in snap.add_health(_kernel(), "relu", TORCH, history)["release"]


def test_pr_builds_never_count_as_releases():
    status = snap.add_health(_kernel(), "relu", TORCH, _history(_event("linux", "failure", "2026-10-20T00:00:00Z", mode="pr")))
    assert status["release"] == {} and status["released"] is None
    assert status["flags"] == {} and status["health"] == "ok"


def test_rebuild_and_unreleased_compare_against_main():
    history = _history(_event("linux", "success", "2026-10-01T00:00:00Z", builder="b1", tree="t1"))
    flags = snap.add_health(_kernel(), "relu", TORCH, history)["flags"]
    assert flags["needs-rebuild"] == "released with b1, main pins b2"
    assert "unreleased" in flags


def test_stable_abi_and_noarch_cover_newer_torch():
    hub = {"torch_versions": [], "stable_abi": "2.10", "noarch": False}
    assert snap.torch_covered(hub, "2.15")
    assert not snap.torch_covered({**hub, "stable_abi": None}, "2.15")
    assert snap.torch_covered({**hub, "stable_abi": None, "noarch": True}, "2.15")


def test_torch_only_in_an_open_hub_pr_is_pending_not_missing():
    pr = {"num": 6, "created_at": "2026-10-19T11:01:14Z", "torch_versions": ["2.14"], "stable_abi": None, "noarch": False}
    hub = {"branch": "v1", "torch_versions": ["2.13"], "stable_abi": None, "noarch": False, "pending": [pr]}
    flags = snap.add_health(_kernel(repo_id="ggml-org/ggml-attn", hub=hub), "ggml-attn", TORCH, [])["flags"]
    assert flags["missing-torch"] == "no 2.15 on v1"
    assert flags["pending-merge"] == "2.14 in https://huggingface.co/kernels/ggml-org/ggml-attn/discussions/6 (2026-10-19)"


def test_history_picks_up_events_newer_than_history_jsonl(tmp_path):
    old = snap.history_row("events/2026/10/1-1-linux.json", _event("linux", "success", "2026-10-01T00:00:00Z"))
    (tmp_path / "history.jsonl").write_text(json.dumps(old) + "\n")
    new = tmp_path / "events/2026/10/2-1-linux.json"
    new.parent.mkdir(parents=True)
    new.write_text(json.dumps(_event("linux", "failure", "2026-10-20T00:00:00Z")))
    (tmp_path / "events/2026/10/1-1-linux.json").write_text(json.dumps(_event("linux", "success", "2026-10-01T00:00:00Z")))

    rows = snap.load_history(snap.Dataset(str(tmp_path)), NOW)
    assert [r["result"] for r in rows] == ["success", "failure"]
