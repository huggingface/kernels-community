import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import emit_build_event  # noqa: E402

SOURCES = {
    "source": {"sha": "abc", "kernel_tree_sha": "def", "builder_rev": "123", "builder_date": None},
    "config": {"version": 2, "repo_id": "kernels-community/relu", "backends": ["cuda"]},
}


def _job(name, conclusion="success", failed_step=None):
    steps = [{"name": "Checkout", "conclusion": "success"}]
    if failed_step:
        steps.append({"name": failed_step, "conclusion": "failure"})
    return {
        "id": 7,
        "name": name,
        "conclusion": conclusion,
        "started_at": "2026-10-27T10:00:00Z",
        "completed_at": "2026-10-27T10:30:00Z",
        "steps": steps,
    }


def _env(**overrides):
    env = {
        "KERNEL": "relu",
        "MODE": "release",
        "GITHUB_REPOSITORY": "huggingface/kernels-community",
        "GITHUB_RUN_ID": "42",
        "NEEDS_JSON": json.dumps({"build-kernel": {"result": "success"}}),
    }
    env.update(overrides)
    return env


def test_matrix_job_name_yields_backend_and_arch():
    job = emit_build_event.summarize_job(
        _job("build-kernel (rocm, aarch64-linux, aws-r8g-8xl-plus-nix, 2, 12)", "failure", "Build kernel"),
        "linux",
    )
    assert (job["backend"], job["arch"]) == ("rocm", "aarch64")
    assert job["failed_step"] == "Build kernel"
    assert job["duration_s"] == 1800
    assert job["id"] == 7


def test_mac_job_defaults_to_metal():
    job = emit_build_event.summarize_job(_job("build-kernel"), "mac")
    assert (job["backend"], job["arch"]) == ("metal", "aarch64")


def test_overall_prefers_failure_over_cancelled():
    needs = {"a": {"result": "cancelled"}, "b": {"result": "failure"}, "c": {"result": "skipped"}}
    assert emit_build_event.overall_result(needs) == "failure"
    assert emit_build_event.overall_result({"a": {"result": "skipped"}}) == "success"


def test_build_event_drops_own_job_and_records_trigger():
    jobs = [_job("build-kernel (cuda, x86_64-linux, r, 2, 12)"), _job("emit-status-event")]
    event = emit_build_event.build_event(
        "linux", _env(PR_NUMBER="1092", REQUESTED_BY="someone", UPLOAD="false"), jobs, SOURCES, []
    )
    assert [j["backend"] for j in event["result"]["jobs"]] == ["cuda"]
    assert event["trigger"]["pr_number"] == 1092
    assert event["trigger"]["requested_by"] == "someone"
    assert event["trigger"]["upload"] is False
    assert event["source"]["builder_rev"] == "123"
    assert event["result"]["overall"] == "success"
    assert event["run_url"].endswith("/actions/runs/42")
