import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def check(monkeypatch):
    # Git access is outside these tests; only satisfy the checker's import.
    monkeypatch.setitem(sys.modules, "pygit2", SimpleNamespace())
    spec = importlib.util.spec_from_file_location(
        "check_public_api_under_test", Path(__file__).with_name("check_public_api.py")
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("private_name", ["_private_for_testing", "_internal"])
@pytest.mark.parametrize("private_path", ["{name}.py", "{name}/__init__.py"])
def test_private_module_changes_do_not_change_public_api(check, private_name, private_path):
    files = {
        "torch-ext/k/__init__.py": "def run(x): pass\n__all__ = ['run']\n",
    }

    def api():
        return check.python_api(check.Source(files, lambda content: content))

    baseline = api()
    files["torch-ext/k/__init__.py"] += (
        f"from . import {private_name}\n__all__ = ['run', '{private_name}']\n"
    )
    module = "torch-ext/k/" + private_path.format(name=private_name)
    files[module] = "def helper(x): pass\n__all__ = ['helper']\n"
    assert api() == baseline

    files[module] = "def replacement(x, y): pass\n__all__ = ['replacement']\n"
    assert api() == baseline

    files["torch-ext/k/__init__.py"] = "def run(x): pass\n__all__ = ['run']\n"
    del files[module]
    assert api() == baseline


def test_nested_private_modules_are_excluded(check):
    files = {
        "torch-ext/k/__init__.py": "from . import layers\n__all__ = ['layers']\n",
        "torch-ext/k/layers/__init__.py": (
            "from . import _internal\n"
            "def run(x): pass\n"
            "__all__ = ['run', '_internal']\n"
        ),
        "torch-ext/k/layers/_internal.py": "def helper(x): pass\n",
    }
    assert check.python_api(check.Source(files, lambda content: content)) == {
        "py k __all__": "['layers']",
        "py k.layers __all__": "['run']",
        "py k.layers.run": "(x)",
    }


@pytest.mark.parametrize("exports", ["", "__all__ = ['run', 'helpers']\n"])
def test_public_reexports_from_private_modules_are_checked(check, exports):
    files = {
        "torch-ext/k/__init__.py": (
            "from ._internal import run\nfrom . import _internal as helpers\n" + exports
        ),
        "torch-ext/k/_internal.py": "def run(x): pass\n__all__ = ['run']\n",
    }
    source = check.Source(files, lambda content: content)
    baseline = check.python_api(source)
    assert baseline["py k.run"] == "(x)"
    assert baseline["py k.helpers.run"] == "(x)"

    files["torch-ext/k/_internal.py"] = "def run(x, y): pass\n__all__ = ['run']\n"
    updated = check.python_api(source)
    assert updated["py k.run"] == "(x, y)"
    assert updated["py k.helpers.run"] == "(x, y)"
    assert check.report("k", baseline, updated).breaking


@pytest.mark.parametrize(
    "old,new,expected",
    [
        pytest.param(
            "(q, mask=None, scale=None)",
            "(q, mask=None, scale=None, sinks=None, softcap=0.0)",
            True,
            id="appended-optional-arguments",
        ),
        pytest.param(
            "(q, *, scale=None)",
            "(q, *, scale=None, sinks=None)",
            False,
            id="added-keyword-only-argument",
        ),
        pytest.param(
            "(q, *, scale=None)",
            "(q, sinks=None, *, scale=None)",
            True,
            id="appended-before-keyword-only-arguments",
        ),
        pytest.param(
            "(q, /)",
            "(q, /, sinks=None)",
            True,
            id="appended-after-positional-only-arguments",
        ),
        pytest.param(
            "(q, /)",
            "(q, sinks=None, /)",
            True,
            id="appended-optional-positional-only-argument",
        ),
        pytest.param(
            "(q, /, scale=1)",
            "(q, sinks=1, /, scale=1)",
            False,
            id="inserted-positional-only-argument",
        ),
        pytest.param(
            "(q, *args)",
            "(q, *args, sinks=None)",
            False,
            id="added-keyword-only-argument-with-varargs",
        ),
        pytest.param(
            "(q)",
            "(q, sinks)",
            False,
            id="added-required-positional-argument",
        ),
        pytest.param(
            "(q)",
            "(q, sinks: Tensor | None)",
            False,
            id="nullable-argument-without-default",
        ),
        pytest.param(
            "(q)",
            "(q, *, sinks)",
            False,
            id="added-required-keyword-only-argument",
        ),
        pytest.param(
            "(q, scale=1)",
            "(q, scale=2, sinks=None)",
            False,
            id="changed-default-with-optional-addition",
        ),
        pytest.param(
            "(q, scale=1)",
            "(q, scale=2)",
            False,
            id="changed-default",
        ),
        pytest.param(
            "(q, scale)",
            "(q, scale=1, sinks=None)",
            False,
            id="added-default-to-existing-argument",
        ),
        pytest.param(
            "(q, scale=1)",
            "(q, scale, sinks=None)",
            False,
            id="removed-default",
        ),
        pytest.param(
            "(q, scale=1)",
            "(q, sinks=None, scale=1)",
            False,
            id="inserted-positional-argument",
        ),
        pytest.param(
            "(q, scale=1)",
            "(q, *, scale=1, sinks=None)",
            False,
            id="existing-argument-made-keyword-only",
        ),
        pytest.param(
            "(q: Tensor) -> Tensor",
            "(q: int, sinks=None) -> Tensor",
            False,
            id="changed-argument-type",
        ),
        pytest.param(
            "(q) -> Tensor",
            "(q, sinks=None) -> tuple",
            False,
            id="changed-return-type",
        ),
        pytest.param(
            "(q, *args)",
            "(q, sinks=None, *args)",
            False,
            id="new-argument-captures-varargs",
        ),
        pytest.param(
            "(q, **kwargs)",
            "(q, sinks=None, **kwargs)",
            False,
            id="new-argument-captures-kwargs",
        ),
        pytest.param("(q, scale=1)", "(q)", False, id="removed-argument"),
        pytest.param(
            "(q)",
            "(x, sinks=None)",
            False,
            id="renamed-argument",
        ),
        pytest.param("(q)", "(q)", False, id="unchanged-signature"),
        pytest.param(
            "<value>",
            "<external>",
            False,
            id="non-function-export",
        ),
    ],
)
def test_python_optional_argument_compatibility(check, old, new, expected):
    assert check._python_optional_added(old, new) == expected


def argument(name, *, dtype="Tensor", optional=False, default=None):
    return SimpleNamespace(
        name=name,
        type=dtype,
        real_type=dtype,
        N=None,
        kwarg_only=False,
        has_default_value=lambda: optional,
        default_value=default,
        alias_info=None,
    )


@pytest.fixture
def schemas(monkeypatch):
    # Stub Torch's parser output, not the compatibility logic being tested.
    old = SimpleNamespace(
        arguments=[
            argument("q"),
            argument("scale", dtype="float", optional=True, default=1.0),
        ],
        returns=[argument("")],
    )
    new = deepcopy(old)
    new.arguments += [
        argument("sinks", dtype="Tensor?", optional=True),
        argument("softcap", dtype="float", optional=True, default=0.0),
    ]
    new.is_backward_compatible_with = lambda previous: True
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            _C=SimpleNamespace(parse_schema={"old": old, "new": new}.__getitem__)
        ),
    )
    return old, new


def test_torch_accepts_appended_optional_arguments(check, schemas):
    assert check._torch_optional_added("old", "new")
    diff = check.report("k", {"op f": "old"}, {"op f": "new"})
    assert not diff.breaking and not diff.additive


@pytest.mark.parametrize(
    "field,value",
    [
        pytest.param("name", "other", id="renamed-argument"),
        pytest.param("type", "int", id="changed-argument-type"),
        pytest.param("real_type", "int", id="changed-underlying-type"),
        pytest.param("N", 2, id="changed-list-size-constraint"),
        pytest.param("kwarg_only", True, id="existing-argument-made-keyword-only"),
        pytest.param(
            "has_default_value",
            lambda: False,
            id="removed-default",
        ),
        pytest.param(
            "default_value",
            2.0,
            id="changed-default",
        ),
        pytest.param(
            "alias_info",
            SimpleNamespace(is_write=True, before_set={"a"}, after_set={"a"}),
            id="changed-mutation-contract",
        ),
    ],
)
def test_torch_rejects_existing_argument_changes(check, schemas, field, value):
    _, new = schemas
    setattr(new.arguments[1], field, value)
    assert not check._torch_optional_added("old", "new")


@pytest.mark.parametrize(
    "field,value",
    [
        pytest.param(
            "has_default_value",
            lambda: False,
            id="nullable-argument-without-default",
        ),
        pytest.param(
            "kwarg_only",
            True,
            id="added-keyword-only-argument",
        ),
    ],
)
def test_torch_rejects_required_or_keyword_only_additions(check, schemas, field, value):
    _, new = schemas
    setattr(new.arguments[2], field, value)
    assert not check._torch_optional_added("old", "new")


@pytest.mark.parametrize(
    "change",
    [
        pytest.param("insert", id="inserted-argument"),
        pytest.param("reorder", id="reordered-arguments"),
        pytest.param("remove", id="removed-argument"),
        pytest.param("unchanged", id="unchanged-schema"),
        pytest.param("return", id="changed-return-type"),
        pytest.param("incompatible", id="incompatible-schema"),
    ],
)
def test_torch_rejects_other_schema_changes(check, schemas, change):
    old, new = schemas
    if change == "insert":
        new.arguments.insert(1, new.arguments.pop(2))
    elif change == "reorder":
        new.arguments[:2] = reversed(new.arguments[:2])
    elif change == "remove":
        new.arguments.pop(0)
    elif change == "unchanged":
        new.arguments = old.arguments
    elif change == "return":
        new.returns = [argument("", dtype="int")]
    else:
        new.is_backward_compatible_with = lambda previous: False
    assert not check._torch_optional_added("old", "new")


def test_non_schema_exports_are_not_optional_additions(check, monkeypatch):
    def parse_schema(text):
        raise RuntimeError("not a schema")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(_C=SimpleNamespace(parse_schema=parse_schema)),
    )
    assert not check._torch_optional_added("bound_function", "other_function")


@pytest.mark.parametrize(
    "old,new,expected",
    [
        pytest.param(
            {"py k.f": "(q)"},
            {"py k.f": "(q, sinks=None)"},
            (False, False),
            id="optional-addition-needs-no-bump",
        ),
        pytest.param(
            {"py k __all__": "['a']"},
            {"py k __all__": "['a', 'b']"},
            (False, True),
            id="expanded-exports-remain-additive",
        ),
        pytest.param(
            {"py k.f": "(q)"},
            {},
            (True, False),
            id="removed-function-remains-breaking",
        ),
        pytest.param(
            {},
            {"py k.f": "(q)"},
            (False, True),
            id="new-function-remains-additive",
        ),
        pytest.param(
            {"py k.f": "(q)"},
            {"py k.f": "(q, sinks)"},
            (True, False),
            id="required-addition-remains-breaking",
        ),
        pytest.param(
            {"py k.f": "(q)"},
            {"py k.f": "(q, sinks=None)", "py k.g": "(q)"},
            (False, True),
            id="optional-addition-with-new-function",
        ),
        pytest.param(
            {"py k.f": "(q)", "py k.g": "(q, scale=1)"},
            {"py k.f": "(q, sinks=None)", "py k.g": "(q, scale=2)"},
            (True, False),
            id="optional-addition-with-changed-default",
        ),
    ],
)
def test_api_change_classification(check, old, new, expected):
    diff = check.report("k", old, new)
    assert (diff.breaking, diff.additive) == expected
