import os
import subprocess
import sys
import unittest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import pr_comment_kernel_bot as bot  # noqa: E402


class ParseCommandTest(unittest.TestCase):
    def test_build_with_kernels(self):
        parsed = bot.parse_command("/kernel-bot build activation relu")
        self.assertIsNone(parsed.error)
        self.assertEqual(parsed.command, "build")
        self.assertEqual(parsed.kernels, ["activation", "relu"])
        self.assertIsNone(parsed.branch)

    def test_security_needs_no_kernels(self):
        parsed = bot.parse_command("/kernel-bot security")
        self.assertIsNone(parsed.error)
        self.assertEqual(parsed.command, "security")
        self.assertEqual(parsed.kernels, [])

    def test_build_without_kernels_is_error(self):
        parsed = bot.parse_command("/kernel-bot build")
        self.assertIsNotNone(parsed.error)
        self.assertIsNone(parsed.command)

    def test_unknown_command_is_error(self):
        self.assertIsNotNone(bot.parse_command("/kernel-bot frobnicate foo").error)

    def test_wrong_prefix_is_error(self):
        self.assertIsNotNone(bot.parse_command("/not-kernel-bot build foo").error)

    def test_too_few_tokens_is_error(self):
        self.assertIsNotNone(bot.parse_command("/kernel-bot").error)

    def test_branch_at_end_is_parsed(self):
        parsed = bot.parse_command("/kernel-bot build foo --branch dev")
        self.assertIsNone(parsed.error)
        self.assertEqual(parsed.kernels, ["foo"])
        self.assertEqual(parsed.branch, "dev")

    def test_branch_not_at_end_is_error(self):
        parsed = bot.parse_command("/kernel-bot build --branch dev foo")
        self.assertIsNotNone(parsed.error)

    def test_security_with_branch_only(self):
        parsed = bot.parse_command("/kernel-bot security --branch dev")
        self.assertIsNone(parsed.error)
        self.assertEqual(parsed.command, "security")
        self.assertEqual(parsed.kernels, [])
        self.assertEqual(parsed.branch, "dev")

    def test_duplicate_kernels_are_deduped(self):
        parsed = bot.parse_command("/kernel-bot build foo foo bar")
        self.assertEqual(parsed.kernels, ["foo", "bar"])

    def test_invalid_kernel_name_is_error(self):
        self.assertIsNotNone(bot.parse_command("/kernel-bot build bad/name").error)

    def test_invalid_branch_name_is_error(self):
        parsed = bot.parse_command("/kernel-bot build foo --branch bad~branch")
        self.assertIsNotNone(parsed.error)


class CommentHelpersTest(unittest.TestCase):
    def test_supported_characters(self):
        self.assertTrue(
            bot.comment_has_only_supported_characters("/kernel-bot build activation")
        )
        self.assertFalse(
            bot.comment_has_only_supported_characters("/kernel-bot build `rm -rf`")
        )

    def test_make_dispatch_key_format(self):
        key = bot.make_dispatch_key(42, "activation")
        self.assertTrue(key.startswith("pr42-activation-"))

    def test_pending_comment_lists_security_workflows(self):
        text = bot.format_pending_comment(
            "/kernel-bot security",
            "security audit only",
            "pr-42",
            "abc123",
            include_security=True,
        )
        self.assertIn("Security audit", text)
        for wf in bot.WORKFLOWS["security"]:
            self.assertIn(wf, text)

    def test_result_comment_renders_security_dispatch_and_failure(self):
        dispatched = bot.DispatchResult(
            kernel_name="security (security-audit.yml)",
            dispatch_key="pr42-security-x",
            action_url="https://example/run/1",
        )
        text = bot.format_result_comment(
            "/kernel-bot security",
            "security audit only",
            "pr-42",
            "abc123",
            security_dispatches=[dispatched],
            security_failed=[("security-audit.yml", 500)],
        )
        self.assertIn("https://example/run/1", text)
        self.assertIn("Security audit failed", text)
        self.assertIn("security-audit.yml (HTTP 500)", text)


class DryRunSmokeTest(unittest.TestCase):
    def _run(self, comment, *extra):
        return subprocess.run(
            [
                sys.executable,
                os.path.join(SCRIPT_DIR, "pr_comment_kernel_bot.py"),
                "--dry-run",
                comment,
                "--pr-number",
                "42",
                "--repo",
                "owner/repo",
                "--head-sha",
                "abc123",
                *extra,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_security_only_dry_run(self):
        proc = self._run("/kernel-bot security")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("security-audit.yml", proc.stdout)

    def test_parse_error_dry_run_does_not_crash(self):
        proc = self._run("/kernel-bot frobnicate")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Parse error", proc.stderr)


if __name__ == "__main__":
    unittest.main(buffer=True)
