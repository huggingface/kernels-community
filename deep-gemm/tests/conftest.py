# Mega MoE depends on `get_symm_buffer_size_for_mega_moe` returning a Python
# closure (`std::function<...>`), which can't be exposed through TORCH_LIBRARY
# under Py_LIMITED_API. The op binding itself is unavailable in this build, so
# the test can't be collected.
collect_ignore = [
    "test_mega_moe.py",
]
