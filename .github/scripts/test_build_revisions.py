import os
from pathlib import Path
import subprocess

import pytest
import yaml


WORKFLOW_DIR = Path(__file__).resolve().parents[1] / "workflows"
BUILD_WORKFLOWS = ("build.yaml", "build-mac.yaml", "build-windows.yaml")
PR_REF = "${{ inputs.head_sha }}"
WORKFLOW_REF = "${{ github.sha }}"
SOURCE_REF = "${{ inputs.pr_number != '' && inputs.head_sha || github.sha }}"


@pytest.fixture(params=BUILD_WORKFLOWS)
def workflow(request):
    return yaml.safe_load((WORKFLOW_DIR / request.param).read_text())


def step_named(workflow, name):
    matches = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == name
    ]
    assert len(matches) == 1
    return matches[0]


def run_step(step, directory, **env):
    return subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=directory,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
    )


def test_source_checkouts_use_consistent_revisions(workflow):
    pr_checkouts = 0
    for job_name, job in workflow["jobs"].items():
        for step in job.get("steps", []):
            if not step.get("uses", "").startswith("actions/checkout@"):
                continue
            options = step.get("with", {})
            if "repository" in options:
                # The kernel-builder dependency has its own revision in flake.lock.
                continue
            if step.get("name") == "Checkout PR branch":
                assert step["if"] == "inputs.pr_number != ''"
                assert options["ref"] == PR_REF
                pr_checkouts += 1
            elif step.get("name") == "Checkout default branch":
                assert step["if"] == "inputs.pr_number == ''"
                assert options["ref"] == WORKFLOW_REF
            elif job_name in ("report-status", "emit-status-event"):
                assert options["ref"] == WORKFLOW_REF
            else:
                assert options["ref"] == SOURCE_REF
                pr_checkouts += 1
            assert options["persist-credentials"] is False
    assert pr_checkouts > 0


def test_pr_revision_is_validated_before_checkout(workflow):
    validation = step_named(workflow, "Validate PR revision")
    assert validation["if"] == "inputs.pr_number != ''"
    assert validation["shell"] == "bash"
    assert validation["env"]["HEAD_SHA"] == PR_REF
    jobs = workflow["jobs"]
    initial_job = jobs.get("setup", jobs["build-kernel"])
    steps = initial_job["steps"]
    checkout_index = next(
        i for i, step in enumerate(steps) if step.get("name") == "Checkout PR branch"
    )
    assert steps.index(validation) < checkout_index
    for name, job in jobs.items():
        if "setup" in jobs and name not in ("setup", "report-status", "emit-status-event"):
            needs = job.get("needs", [])
            needs = [needs] if isinstance(needs, str) else needs
            assert "setup" in needs


@pytest.mark.parametrize(
    "head_sha, valid",
    [
        ("a" * 40, True),
        ("", False),
        ("a" * 39, False),
        ("a" * 41, False),
        ("g" * 40, False),
        ("refs/heads/main", False),
        ("a" * 40 + "\n", False),
    ],
)
def test_pr_revision_validation(workflow, tmp_path, head_sha, valid):
    result = run_step(
        step_named(workflow, "Validate PR revision"), tmp_path, HEAD_SHA=head_sha
    )
    assert (result.returncode == 0) is valid, result.stderr


def git(directory, *args):
    return subprocess.check_output(
        ["git", "-C", str(directory), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def test_metadata_keeps_dispatched_revision_after_pr_update(workflow, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "--quiet", "--initial-branch=main")
    git(source, "config", "user.name", "CI test")
    git(source, "config", "user.email", "ci@example.com")
    git(source, "config", "core.hooksPath", "/dev/null")
    manifest = source / "build.toml"
    manifest.write_text('revision = "dispatched"\n')
    git(source, "add", "build.toml")
    git(source, "commit", "--quiet", "-m", "Initial revision")
    dispatched_sha = git(source, "rev-parse", "HEAD")
    manifest.write_text('revision = "updated"\n')
    git(source, "commit", "--quiet", "-am", "Update revision")
    git(source, "update-ref", "refs/pull/7/head", "HEAD")
    updated_sha = git(source, "rev-parse", "HEAD")
    assert updated_sha != dispatched_sha

    work = tmp_path / "work"
    subprocess.run(["git", "clone", "--quiet", str(source), str(work)], check=True)
    output = tmp_path / "output"
    result = run_step(
        step_named(workflow, "Fetch PR head as data"),
        work,
        PR_NUMBER="7",
        HEAD_SHA=dispatched_sha,
        GITHUB_OUTPUT=str(output),
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text() == "ref=FETCH_HEAD\n"
    assert git(work, "rev-parse", "FETCH_HEAD") == dispatched_sha
    assert git(work, "show", "FETCH_HEAD:build.toml") == 'revision = "dispatched"'


@pytest.mark.parametrize("head_sha", ["", "abc123", "refs/heads/main"])
def test_metadata_rejects_invalid_pr_revision_before_fetch(workflow, tmp_path, head_sha):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "git-called"
    fake_git = bin_dir / "git"
    fake_git.write_text('#!/bin/sh\ntouch "$GIT_MARKER"\nexit 0\n')
    fake_git.chmod(0o755)
    result = run_step(
        step_named(workflow, "Fetch PR head as data"),
        tmp_path,
        PR_NUMBER="7",
        HEAD_SHA=head_sha,
        GITHUB_OUTPUT=str(tmp_path / "output"),
        GIT_MARKER=str(marker),
        PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
    )
    assert result.returncode != 0
    assert not marker.exists()


def test_release_metadata_uses_workflow_checkout(workflow, tmp_path):
    output = tmp_path / "output"
    result = run_step(
        step_named(workflow, "Fetch PR head as data"),
        tmp_path,
        PR_NUMBER="",
        HEAD_SHA="",
        GITHUB_OUTPUT=str(output),
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text() == "ref=HEAD\n"
