import pr_kernel_build_guidance as guidance


def test_detects_existing_kernel_source(tmp_path):
    (tmp_path / "relu").mkdir()
    (tmp_path / "relu" / "build.toml").touch()
    files = [{"filename": "relu/torch-ext/relu/activation.py", "status": "modified"}]

    assert guidance.touched_source_kernels(
        files, guidance.kernel_directories(tmp_path)
    ) == ["relu"]


def test_ignores_kernel_docs_and_non_kernel_files(tmp_path):
    (tmp_path / "relu").mkdir()
    (tmp_path / "relu" / "build.toml").touch()
    files = [
        {"filename": "relu/README.md", "status": "modified"},
        {"filename": "relu/CARD.md", "status": "modified"},
        {"filename": ".github/scripts/tool.py", "status": "modified"},
    ]

    assert (
        guidance.touched_source_kernels(files, guidance.kernel_directories(tmp_path))
        == []
    )


def test_unknown_extensions_and_test_changes_are_build_relevant(tmp_path):
    (tmp_path / "relu").mkdir()
    (tmp_path / "relu" / "build.toml").touch()
    files = [
        {"filename": "relu/src/kernel.futurelang", "status": "added"},
        {"filename": "relu/tests/test_relu.py", "status": "modified"},
    ]

    assert guidance.touched_source_kernels(
        files, guidance.kernel_directories(tmp_path)
    ) == ["relu"]


def test_detects_new_kernel_from_added_manifest():
    files = [
        {"filename": "new-kernel/build.toml", "status": "added"},
        {"filename": "new-kernel/torch-ext/new_kernel/__init__.py", "status": "added"},
    ]

    assert guidance.touched_source_kernels(files, set()) == ["new-kernel"]


def test_modified_manifest_is_build_relevant_for_existing_kernel():
    files = [{"filename": "relu/build.toml", "status": "modified"}]

    assert guidance.touched_source_kernels(files, {"relu"}) == ["relu"]


def test_comment_contains_one_command_for_all_kernels():
    comment = guidance.format_comment(["activation", "relu"])

    assert guidance.COMMENT_MARKER in comment
    assert "`/kernel-bot build activation relu`" in comment
    assert "Thanks for your patience." in comment
