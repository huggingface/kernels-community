import io
import os
import sys
import unittest
import urllib.error
from unittest import mock

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


# Fallback routing when build.toml is unreadable or declares unknown backends.
class SelectWorkflowsFallbackTest(unittest.TestCase):
    def _select(self, backends):
        with mock.patch.object(dispatch, "read_backends", return_value=backends):
            return dispatch.select_workflows("somekernel", notes=[])

    def test_unreadable_backends_falls_back_to_all_builds(self):
        self.assertEqual(self._select(None), set(dispatch.WORKFLOWS["build"]))

    def test_unknown_backends_falls_back_to_all_builds(self):
        self.assertEqual(
            self._select(["totally-made-up"]), set(dispatch.WORKFLOWS["build"])
        )


# Pure planning: backends and flags -> the intended dispatch actions.
class PlanDispatchTest(unittest.TestCase):
    def test_status_context_planned_only_with_head_sha(self):
        with_sha = [a for a in _plan(head_sha="abc").actions if a.kind == "build"]
        without_sha = [a for a in _plan().actions if a.kind == "build"]
        self.assertTrue(all(a.status_context for a in with_sha))
        self.assertTrue(all(a.status_context is None for a in without_sha))

    def test_run_security_adds_security_actions(self):
        plan = _plan(pr_number="7", run_security=True)
        self.assertEqual(_kinds(plan), ["build", "security"])

    def test_security_only_plans_only_security(self):
        plan = dispatch.plan_dispatch(
            security_only=True,
            pr_number="42",
            head_sha="deadbeef",
            dispatch_key_prefix="pr42-",
        )
        self.assertEqual(_kinds(plan), ["security"])
        self.assertEqual(
            _workflows(plan.actions), sorted(dispatch.WORKFLOWS["security"])
        )
        for action in plan.actions:
            self.assertTrue(action.dispatch_key.startswith("pr42-security-"))
            self.assertEqual(action.body["inputs"]["pr_number"], "42")
            self.assertEqual(action.body["inputs"]["head_sha"], "deadbeef")


# Orchestration: kernel-name validation and the dry-run no-I/O contract.
class DispatchOrchestratorTest(unittest.TestCase):
    def test_invalid_kernel_name_marks_all_builds_failed(self):
        result = dispatch.dispatch("bad name!", token="", repo="owner/repo")
        self.assertEqual(
            sorted(wf for wf, _ in result.failed), sorted(dispatch.WORKFLOWS["build"])
        )
        self.assertEqual(result.dispatched, [])

    def test_dry_run_projects_plan_and_performs_no_io(self):
        forbidden = mock.Mock(
            side_effect=AssertionError("dry run must not hit the API")
        )
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
        self.assertTrue(result.dispatched)
        self.assertTrue(result.security_dispatched)
        self.assertEqual(
            len(result.dry_run_payloads),
            len(result.dispatched) + len(result.security_dispatched),
        )


# Execution: GitHub API dispatch calls and HTTP-failure handling.
class ExecutePlanTest(unittest.TestCase):
    def test_posts_dispatch_and_pending_status(self):
        plan = _plan(backends=["cuda"], head_sha="abc")
        api = mock.Mock(return_value=(204, ""))
        with mock.patch.object(dispatch, "github_api_request", api):
            result = dispatch.execute_plan(plan, token="t", repo="owner/repo")
        # One build action -> one dispatch POST + one pending-status POST.
        self.assertEqual(api.call_count, 2)
        self.assertEqual([wf for wf, _ in result.dispatched], ["build.yaml"])
        self.assertEqual(result.failed, [])

    def test_records_http_failure(self):
        plan = _plan(backends=["cuda"])  # no head_sha -> no pending-status POST
        err = urllib.error.HTTPError("u", 500, "err", {}, io.BytesIO(b"boom"))
        with mock.patch.object(dispatch, "github_api_request", side_effect=err):
            result = dispatch.execute_plan(plan, token="t", repo="owner/repo")
        self.assertEqual(result.dispatched, [])
        self.assertEqual([code for _, code in result.failed], [500])


# select_workflows: backend-union and Windows-gate cases (single-backend rows omitted).
ROUTING_TRUTH_TABLE = [
    (["cuda"], False, {"build.yaml"}),
    (["cuda"], True, {"build.yaml", "build-windows.yaml"}),
    (["metal"], False, {"build-mac.yaml"}),
    (["cuda", "metal"], False, {"build.yaml", "build-mac.yaml"}),
    (["cuda", "metal"], True, {"build.yaml", "build-mac.yaml", "build-windows.yaml"}),
]


class RoutingTruthTableTest(unittest.TestCase):
    def test_truth_table(self):
        for backends, windows_allowed, expected in ROUTING_TRUTH_TABLE:
            kernel = "k"
            allowlist = {kernel} if windows_allowed else set()
            with self.subTest(backends=backends, windows_allowed=windows_allowed):
                with (
                    mock.patch.object(dispatch, "read_backends", return_value=backends),
                    mock.patch.object(dispatch, "WINDOWS_KERNELS", allowlist),
                ):
                    self.assertEqual(
                        dispatch.select_workflows(kernel, notes=[]), expected
                    )


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


class AllKernelsPropertyTest(unittest.TestCase):
    def setUp(self):
        self._cwd = os.getcwd()
        os.chdir(REPO_ROOT)
        self.kernels = _discover_kernels()

    def tearDown(self):
        os.chdir(self._cwd)

    def test_kernels_were_discovered(self):
        # Guards against the sweep silently testing nothing (wrong cwd, etc.).
        self.assertGreater(len(self.kernels), 0)

    def test_every_build_toml_parses(self):
        for kernel in self.kernels:
            with self.subTest(kernel=kernel):
                # Raises on malformed TOML; None (no backends declared) is valid.
                dispatch.read_backends(kernel)

    def test_every_kernel_routes_to_at_least_one_known_build(self):
        build_workflows = set(dispatch.WORKFLOWS["build"])
        for kernel in self.kernels:
            with self.subTest(kernel=kernel):
                workflows = dispatch.select_workflows(kernel, notes=[])
                self.assertTrue(workflows, f"{kernel} routes to no build workflow")
                self.assertTrue(
                    workflows <= build_workflows,
                    f"{kernel} routes to unknown workflow(s): "
                    f"{workflows - build_workflows}",
                )

    def test_no_kernel_declares_an_unknown_backend(self):
        # An unmapped backend would silently never build; add it to BACKEND_TO_WORKFLOWS.
        known = set(dispatch.BACKEND_TO_WORKFLOWS)
        for kernel in self.kernels:
            backends = dispatch.read_backends(kernel)
            if backends is None:
                continue
            with self.subTest(kernel=kernel):
                unknown = set(backends) - known
                self.assertEqual(
                    unknown, set(), f"{kernel} declares unmapped backend(s): {unknown}"
                )


if __name__ == "__main__":
    unittest.main(buffer=True)
