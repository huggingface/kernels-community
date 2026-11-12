# The upstream tests in the subdirectories are multi-process scripts (run them
# as e.g. `python tests/ep/test_ep.py --num-processes 2`) that need several
# Hopper/Blackwell GPUs; their `test_*` functions are not pytest tests. Only
# collect the tests in this directory.
collect_ignore_glob = ["*/*.py"]
