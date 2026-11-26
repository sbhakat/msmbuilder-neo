"""Run the BLAS wrapper tests that live in the compiled test_cyblas module."""
import pytest

from msmbuilder.tests import test_cyblas as cyblas_module

CYBLAS_TESTS = sorted(name for name in dir(cyblas_module) if name.startswith('test'))


def test_found_cyblas_tests():
    assert CYBLAS_TESTS


@pytest.mark.parametrize('name', CYBLAS_TESTS)
def test_cyblas_function(name):
    getattr(cyblas_module, name)()
