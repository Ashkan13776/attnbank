"""Every runnable example in a docstring is executed here.

The library promises "docstrings with runnable examples on every public function".
A promise that is not tested decays within a release, so the examples are the test:
:mod:`doctest` runs them over the public modules, and a second test asserts that every
name in ``attnbank.__all__`` actually has an ``Examples`` section.

The examples gallery under ``examples/`` is executed by ``test_examples.py``.
"""
from __future__ import annotations

import doctest
import importlib

import pytest

import attnbank as ab

#: modules whose docstrings are executed
DOCTEST_MODULES = [
    "attnbank",
    "attnbank.core",
    "attnbank.registry",
    "attnbank.errors",
    "attnbank.info",
    "attnbank.conformance",
    "attnbank.metrics",
    "attnbank.reference",
]

OPTIONS = (doctest.ELLIPSIS | doctest.IGNORE_EXCEPTION_DETAIL
           | doctest.NORMALIZE_WHITESPACE)


@pytest.mark.parametrize("modname", DOCTEST_MODULES)
def test_docstring_examples_run(modname):
    mod = importlib.import_module(modname)
    results = doctest.testmod(mod, optionflags=OPTIONS, verbose=False, report=True)
    assert results.failed == 0, f"{modname}: {results.failed} doctest failure(s)"


# names that are types, submodules or re-exported building blocks documented elsewhere
_NO_EXAMPLE_NEEDED = {"__version__", "metrics"}


@pytest.mark.parametrize("name", sorted(set(ab.__all__) - _NO_EXAMPLE_NEEDED))
def test_public_names_are_documented(name):
    """Every public name has a docstring, and the callables have runnable examples."""
    obj = getattr(ab, name)
    doc = obj.__doc__
    assert doc and doc.strip(), f"attnbank.{name} has no docstring"
    if isinstance(obj, type) and issubclass(obj, BaseException):
        return                      # error classes document themselves in one line
    assert ">>>" in doc, f"attnbank.{name} has no runnable example in its docstring"


def _public_functions(module):
    import inspect

    return sorted(name for name, obj in vars(module).items()
                  if inspect.isfunction(obj) and not name.startswith("_")
                  and obj.__module__ == module.__name__)


@pytest.mark.parametrize("name", _public_functions(ab.metrics))
def test_metrics_functions_have_examples(name):
    """``attnbank.metrics`` is public API, so its functions carry examples too."""
    fn = getattr(ab.metrics, name)
    assert fn.__doc__ and ">>>" in fn.__doc__, \
        f"attnbank.metrics.{name} has no runnable example in its docstring"


def test_design_contract_is_shipped():
    """DESIGN.md is the binding spec; the docs site and CONTRIBUTING both point at it."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("DESIGN.md", "README.md", "CONTRIBUTING.md", "CHANGELOG.md", "LICENSE"):
        assert (root / name).exists(), f"{name} is missing from the distribution root"
