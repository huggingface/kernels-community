import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "kernels_ci: the cheap subset kernels-community CI runs")
