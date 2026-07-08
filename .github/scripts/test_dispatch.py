import io
import os
import sys
import urllib.error
from unittest import mock

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import dispatch  # noqa: E402


def _plan(kernel="somekernel", backends=("cuda",), **kwargs):
    with mock.patch.object(dispatch, "read_backends", return_value=list(backends)):
        return dispatch.plan_dispatch(kernel, **kwargs)


def _kinds(plan):
    return sorted({a.kind for a in plan.actions})


def _workflows(actions):
    return sorted(a.workflow for a in actions)


def _select(backends):
    return dispatch.select_workflows("somekernel", backends, notes=[])


def _build_actions(plan):
    return [a for a in plan.actions if a.kind == "build"]


def _backends_csv(action):
    return sorted(b for b in action.body["inputs"]["backends"].split(",") if b)


# Fallback routing when build.toml is unreadable or declares unknown backends.
def test_unreadable_backends_falls_back_to_all_builds():
    assert _select(None) == set(dispatch.WORKFLOWS["build"])


def test_unknown_backends_falls_back_to_all_builds():
    assert _select(["totally-made-up"]) == set(dispatch.WORKFLOWS["build"])


# Pure planning: backends and flags -> the intended dispatch actions.
def test_status_context_planned_only_with_head_sha():
    with_sha = [a for a in _plan(head_sha="abc").actions if a.kind == "build"]
    without_sha = [a for a in _plan().actions if a.kind == "build"]
    assert all(a.status_context for a in with_sha)
    assert all(a.status_context is None for a in without_sha)


def test_run_security_adds_security_actions():
    plan = _plan(pr_number="7", run_security=True)
    assert _kinds(plan) == ["build", "security"]


def test_security_only_plans_only_security():
    plan = dispatch.plan_dispatch(
        security_only=True,
        pr_number="42",
        head_sha="deadbeef",
        dispatch_key_prefix="pr42-",
    )
    assert _kinds(plan) == ["security"]
    assert _workflows(plan.actions) == sorted(dispatch.WORKFLOWS["security"])
    for action in plan.actions:
        assert action.dispatch_key.startswith("pr42-security-")
        assert action.body["inputs"]["pr_number"] == "42"
        assert action.body["inputs"]["head_sha"] == "deadbeef"


# Orchestration: kernel-name validation and the dry-run no-I/O contract.
def test_invalid_kernel_name_marks_all_builds_failed():
    result = dispatch.dispatch("bad name!", token="", repo="owner/repo")
    assert sorted(wf for wf, _ in result.failed) == sorted(dispatch.WORKFLOWS["build"])
    assert result.dispatched == []


def test_dry_run_projects_plan_and_performs_no_io():
    forbidden = mock.Mock(side_effect=AssertionError("dry run must not hit the API"))
    with (
        mock.patch.object(dispatch, "read_backends", return_value=["cuda"]),
        mock.patch.object(dispatch, "github_api_request", forbidden),
    ):
        result = dispatch.dispatch(
            "somekernel",
            token="",
            repo="owner/repo",
            pr_number="7",
            run_security=True,
            dry_run=True,
        )
    forbidden.assert_not_called()
    assert result.dispatched
    assert result.security_dispatched
    assert len(result.dry_run_payloads) == len(result.dispatched) + len(
        result.security_dispatched
    )


# Execution: GitHub API dispatch calls and HTTP-failure handling.
def test_posts_dispatch_and_pending_status():
    plan = _plan(backends=["cuda"], head_sha="abc")
    api = mock.Mock(return_value=(204, ""))
    with mock.patch.object(dispatch, "github_api_request", api):
        result = dispatch.execute_plan(plan, token="t", repo="owner/repo")
    # One build action -> one dispatch POST + one pending-status POST.
    assert api.call_count == 2
    assert [wf for wf, _ in result.dispatched] == ["build.yaml"]
    assert result.failed == []


def test_records_http_failure():
    plan = _plan(backends=["cuda"])  # no head_sha -> no pending-status POST
    err = urllib.error.HTTPError("u", 500, "err", {}, io.BytesIO(b"boom"))
    with mock.patch.object(dispatch, "github_api_request", side_effect=err):
        result = dispatch.execute_plan(plan, token="t", repo="owner/repo")
    assert result.dispatched == []
    assert [code for _, code in result.failed] == [500]


# parse_kernel_arg: the `kernel` / `kernel[b1,b2]` argument grammar.
@pytest.mark.parametrize(
    "token, expected",
    [
        ("flash-attn2", ("flash-attn2", None)),
        ("flash-attn2[xpu,cpu]", ("flash-attn2", ["xpu", "cpu"])),
        ("relu[cpu]", ("relu", ["cpu"])),
        # Unknown backends parse here; the caller rejects them by name.
        ("flash-attn2[bogus]", ("flash-attn2", ["bogus"])),
        ("bad/name", (None, None)),
        ("kernel[]", (None, None)),  # empty scope is not valid
        ("kernel[,]", (None, None)),  # dangling comma is not valid
        ("kernel[cpu,]", (None, None)),
    ],
)
def test_parse_kernel_arg(token, expected):
    assert dispatch.parse_kernel_arg(token) == expected


# requested_backends: user-supplied filter narrows workflows and the scoped CSV.
def test_requested_backends_filter_narrows_workflows_and_scope():
    plan = _plan(
        backends=["cpu", "cuda", "xpu"], requested_backends=["xpu", "cpu"]
    )
    build = _build_actions(plan)
    # cuda dropped -> Windows gated off for the non-allowlisted kernel -> Linux only.
    assert _workflows(build) == ["build.yaml"]
    assert _backends_csv(build[0]) == ["cpu", "xpu"]


def test_requested_backends_undeclared_are_dropped():
    plan = _plan(backends=["cpu", "cuda"], requested_backends=["cpu", "rocm"])
    build = _build_actions(plan)
    assert _backends_csv(build[0]) == ["cpu"]  # rocm not declared -> dropped


def test_requested_backends_none_match_skips_all_builds():
    plan = _plan(backends=["cpu", "cuda"], requested_backends=["rocm"])
    assert _build_actions(plan) == []
    assert plan.skipped == sorted(dispatch.WORKFLOWS["build"])
    assert any("skipping build" in n for n in plan.notes)


def test_requested_backends_thread_through_dispatch_dry_run():
    with mock.patch.object(
        dispatch, "read_backends", return_value=["cpu", "cuda", "xpu"]
    ):
        result = dispatch.dispatch(
            "somekernel",
            token="",
            repo="owner/repo",
            dry_run=True,
            requested_backends=["xpu"],
        )
    csvs = [body["inputs"]["backends"] for _, body in result.dry_run_payloads]
    assert csvs, "expected at least one dispatched build"
    assert all("cuda" not in c.split(",") for c in csvs)
    assert any("xpu" in c.split(",") for c in csvs)


# select_workflows: backend-union and Windows-gate cases (single-backend rows omitted).
ROUTING_TRUTH_TABLE = [
    (["cuda"], False, {"build.yaml"}),
    (["cuda"], True, {"build.yaml", "build-windows.yaml"}),
    (["metal"], False, {"build-mac.yaml"}),
    (["cuda", "metal"], False, {"build.yaml", "build-mac.yaml"}),
    (["cuda", "metal"], True, {"build.yaml", "build-mac.yaml", "build-windows.yaml"}),
]


@pytest.mark.parametrize("backends, windows_allowed, expected", ROUTING_TRUTH_TABLE)
def test_truth_table(backends, windows_allowed, expected):
    kernel = "k"
    allowlist = {kernel} if windows_allowed else set()
    with mock.patch.object(dispatch, "WINDOWS_KERNELS", allowlist):
        assert dispatch.select_workflows(kernel, backends, notes=[]) == expected


# Invariants over every real kernel: assert properties, not exact per-kernel output.
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))


def _discover_kernels():
    kernels = []
    for name in sorted(os.listdir(REPO_ROOT)):
        if not dispatch.KERNEL_NAME_RE.match(name):
            continue
        if os.path.exists(os.path.join(REPO_ROOT, name, "build.toml")):
            kernels.append(name)
    return kernels


@pytest.fixture
def kernels():
    # read_backends/select_workflows resolve build.toml relative to cwd.
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield _discover_kernels()
    finally:
        os.chdir(cwd)


def test_kernels_were_discovered(kernels):
    # Guards against the sweep silently testing nothing (wrong cwd, etc.).
    assert len(kernels) > 0


def test_every_build_toml_parses(kernels):
    for kernel in kernels:
        # Raises on malformed TOML; None (no backends declared) is valid.
        dispatch.read_backends(kernel)


def test_every_kernel_routes_to_at_least_one_known_build(kernels):
    build_workflows = set(dispatch.WORKFLOWS["build"])
    for kernel in kernels:
        workflows = dispatch.select_workflows(kernel, dispatch.read_backends(kernel), notes=[])
        assert workflows, f"{kernel} routes to no build workflow"
        assert workflows <= build_workflows, (
            f"{kernel} routes to unknown workflow(s): {workflows - build_workflows}"
        )


def test_no_kernel_declares_an_unknown_backend(kernels):
    # An unmapped backend would silently never build; add it to BACKEND_TO_WORKFLOWS.
    known = set(dispatch.BACKEND_TO_WORKFLOWS)
    for kernel in kernels:
        backends = dispatch.read_backends(kernel)
        if backends is None:
            continue
        unknown = set(backends) - known
        assert unknown == set(), f"{kernel} declares unmapped backend(s): {unknown}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
