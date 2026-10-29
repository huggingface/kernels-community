import importlib.util
from pathlib import Path
from subprocess import CompletedProcess



spec = importlib.util.spec_from_file_location(
    "check_ports", Path(__file__).resolve().parents[2] / "scripts/check_ports.py"
)
ports = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ports)


def test_vendor_pins_include_continuations(tmp_path):
    recipe = tmp_path / "port.kdl"
    recipe.write_text(
        '// vendor name="ignored" repo="bad" commit="bad"\n'
        'vendor name="python" repo="upstream" commit="abc" path="package"\n'
        'vendor \\\n    name="dependency" \\\n    repo="other" commit="def"\n'
    )
    assert ports.parse_vendors(recipe) == [
        ("python", "upstream", "abc"),
        ("dependency", "other", "def"),
    ]


def test_check_supplies_vendors_and_compares_complete_tree(tmp_path, monkeypatch):
    root, work = tmp_path / "repo", tmp_path / "work"
    kernel = root / "example"
    port = kernel / "port"
    port.mkdir(parents=True)
    work.mkdir()
    (port / "port.kdl").write_text(
        'source repo="upstream" commit="abc"\n'
        'vendor name="python" repo="upstream" commit="abc"\n'
        'vendor name="dependency" repo="other" commit="def"\n'
    )
    committed = kernel / "src"
    committed.mkdir(exist_ok=True)
    (committed / "build.toml").write_text("manifest")
    (committed / "port-provenance.json").write_text("provenance")
    checkouts = []

    def checkout(repo, commit, dest):
        checkouts.append((repo, commit))
        dest.mkdir()

    def run(command, **kwargs):
        assert command[0] == "test-kernel-port"
        upstream = command[command.index("--dir") + 1]
        vendors = [command[i + 1] for i, arg in enumerate(command) if arg == "--vendor"]
        assert vendors[0] == f"python={upstream}"
        assert vendors[1].startswith("dependency=")
        assert vendors[1].split("=", 1)[1] != upstream
        output = Path(command[command.index("--out") + 1])
        output.mkdir()
        (output / "build.toml").write_text("manifest")
        (output / "port-provenance.json").write_text("provenance")
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(ports, "checkout", checkout)
    monkeypatch.setattr(ports.subprocess, "run", run)
    monkeypatch.setenv("KERNEL_PORT", "test-kernel-port")
    assert ports.check("example", root, work) == []
    assert checkouts == [("upstream", "abc"), ("other", "def")]

    (committed / "port-provenance.json").write_text("stale")
    (committed / "extra.py").touch()
    assert ports.check("example", root, work) == [
        "committed but not generated: extra.py",
        "differs: port-provenance.json",
    ]
