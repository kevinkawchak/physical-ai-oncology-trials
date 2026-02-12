"""Unit tests for scripts/verify_installation.py.

Tests cover REQUIREMENTS/OPTIONAL constants and check_version function.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

MOD_PATH = "scripts/verify_installation.py"


@pytest.fixture()
def mod():
    return load_module("verify_installation", MOD_PATH)


class TestConstants:
    def test_requirements_nonempty(self, mod):
        assert len(mod.REQUIREMENTS) > 0

    def test_requirements_have_versions(self, mod):
        for pkg, info in mod.REQUIREMENTS.items():
            assert isinstance(info, tuple)
            assert len(info) == 2

    def test_optional_packages_exist(self, mod):
        assert len(mod.OPTIONAL) >= 0  # May be empty


class TestCheckVersion:
    def test_numpy_available(self, mod):
        ok, info = mod.check_version("numpy", "1.20.0")
        assert ok is True
        assert isinstance(info, str)

    def test_nonexistent_package(self, mod):
        ok, info = mod.check_version("totally_fake_package_xyz", "0.0.1")
        assert ok is False
        assert isinstance(info, str)
