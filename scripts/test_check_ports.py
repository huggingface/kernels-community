import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

COMMIT = "a" * 40
PINS = {"upstream": COMMIT, "dep": COMMIT}
PROVENANCE = '{"tree": "sha256:aa"}\n'
BUILD_TOML = '[general]\nname = "k"\nversion = 1\n\n[general.hub]\nrepo-id = "org/k"\n'


@pytest.fixture
def ports(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "check_ports_under_test", Path(__file__).with_name("check_ports.py")
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def lock(rev, vendor=COMMIT):
    return json.dumps(
        {
            "nodes": {
                "root": {"inputs": {"upstream": "upstream", "dep": "dep"}},
                "upstream": {"locked": {"rev": rev}},
                "dep": {"locked": {"rev": vendor}},
            }
        }
    )


@pytest.fixture
def srcless(tmp_path):
    kernel = tmp_path / "repo" / "k"
    kernel.mkdir(parents=True)
    (kernel / "build.toml").write_text(BUILD_TOML)
    (kernel / "port-provenance.json").write_text(PROVENANCE)
    (kernel / "flake.lock").write_text(lock(COMMIT))

    generated = tmp_path / "out" / "k"
    (generated / "tests").mkdir(parents=True)
    (generated / "build.toml").write_text(BUILD_TOML)
    (generated / "port-provenance.json").write_text(PROVENANCE)
    (generated / "tests" / "test_k.py").write_text(
        'get_kernel("org/k", version=1)\n'
    )
    return kernel, generated


def test_srcless_port_in_sync(ports, srcless):
    assert ports.check_srcless(*srcless, PINS) == []


def test_srcless_port_stale_build_toml(ports, srcless):
    kernel, generated = srcless
    (kernel / "build.toml").write_text(BUILD_TOML.replace("version = 1", "version = 2"))
    assert ports.check_srcless(kernel, generated, PINS) == ["differs: build.toml"]


def test_srcless_port_output_changed(ports, srcless):
    kernel, generated = srcless
    (generated / "port-provenance.json").write_text('{"tree": "sha256:bb"}\n')
    assert ports.check_srcless(kernel, generated, PINS) == [
        "differs: port-provenance.json"
    ]


def test_srcless_port_needs_provenance(ports, srcless):
    kernel, generated = srcless
    (kernel / "port-provenance.json").unlink()
    assert ports.check_srcless(kernel, generated, PINS) == [
        "no src/ and no committed port-provenance.json"
    ]


def test_srcless_port_needs_build_toml(ports, srcless):
    kernel, generated = srcless
    (kernel / "build.toml").unlink()
    assert ports.check_srcless(kernel, generated, PINS) == [
        "no src/ and no committed build.toml"
    ]


def test_srcless_port_lock_drifts_from_recipe(ports, srcless):
    kernel, generated = srcless
    (kernel / "flake.lock").write_text(lock("b" * 40))
    (problem,) = ports.check_srcless(kernel, generated, PINS)
    assert "flake.lock pins upstream" in problem


def test_srcless_port_vendor_lock_drifts_from_recipe(ports, srcless):
    kernel, generated = srcless
    (kernel / "flake.lock").write_text(lock(COMMIT, vendor="b" * 40))
    (problem,) = ports.check_srcless(kernel, generated, PINS)
    assert "flake.lock pins dep" in problem


def test_parse_vendors(ports, tmp_path):
    recipe = tmp_path / "port.kdl"
    recipe.write_text(
        'vendor \\\n    name="dep" \\\n    repo="https://e.com/d.git" \\\n'
        f'    commit="{COMMIT}" path="" to="d"\n'
        f'vendor name="dep" repo="https://e.com/d.git" commit="{COMMIT}" path="x" to="y"\n'
    )
    assert ports.parse_vendors(recipe) == [("dep", "https://e.com/d.git", COMMIT)]


def test_generate_passes_vendors(ports, tmp_path, monkeypatch):
    recipe = tmp_path / "port.kdl"
    recipe.write_text(
        f'source repo="https://e.com/u.git" commit="{COMMIT}"\n'
        f'vendor name="dep" repo="https://e.com/d.git" commit="{"b" * 40}" path="" to="d"\n'
    )
    monkeypatch.setattr(ports, "checkout", lambda repo, commit, dest: dest.mkdir())
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ports.subprocess, "run", run)
    assert ports.generate(recipe, tmp_path / "out" / "k", tmp_path) == []
    (cmd,) = calls
    assert cmd[cmd.index("--vendor") + 1] == f"dep={tmp_path}/upstream-{'b' * 12}"


def test_srcless_port_checks_generated_test_versions(ports, srcless):
    kernel, generated = srcless
    (generated / "tests" / "test_k.py").write_text('get_kernel("org/k", version=2)\n')
    (problem,) = ports.check_srcless(kernel, generated, PINS)
    assert "expected get_kernel" in problem


def test_changed_ports_counts_srcless_files(ports, tmp_path, monkeypatch):
    for name in ("a", "b", "c"):
        (tmp_path / name / "port").mkdir(parents=True)
        (tmp_path / name / "port" / "port.kdl").touch()

    class Diff:
        stdout = "a/flake.lock\nb/port/port.kdl\nc/README.md\nother/flake.lock\n"

    monkeypatch.setattr(ports.subprocess, "run", lambda *a, **k: Diff)
    assert ports.changed_ports(tmp_path, "base") == ["a", "b"]
