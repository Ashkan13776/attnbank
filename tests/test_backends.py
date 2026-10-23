"""The backend layer: detection, selection, announcement, and the study pin.

Equivalence to the ``reference`` path is tested in ``test_conformance.py``
(``test_backend_equivalence``), because it is a property of every method rather than of
the backend machinery. This file tests the machinery.
"""
from __future__ import annotations

import subprocess
import sys
import warnings

import pytest
import torch

import attnbank as ab
from attnbank.backends import DISPATCHABLE, REFERENCE, resolve_backend


def qkv(b=1, h=2, m=6, n=10, d=8, dv=6, dtype=torch.float64):
    g = torch.Generator().manual_seed(0)
    mk = lambda *s: torch.randn(*s, generator=g, dtype=dtype)
    return mk(b, h, m, d), mk(b, h, n, d), mk(b, h, n, dv)


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------
def test_reference_is_always_available():
    assert ab.is_backend_available(REFERENCE)
    assert ab.backends()[REFERENCE].available


@pytest.mark.parametrize("name", ["reference", "sdpa", "flash_attn", "xformers", "triton"])
def test_every_backend_explains_itself(name):
    """An unavailable backend must say why, and how to fix it if that is possible."""
    info = ab.backend_info(name)
    assert info.description
    if not info.available:
        assert info.reason, f"{name} is unavailable without saying why"


def test_the_backend_table_prints_a_report():
    text = str(ab.backends())
    assert "attnbank backends on this machine" in text
    assert "reference" in text and "available" in text
    assert f"default: {ab.get_default_backend()!r}" in text


def test_unknown_backend_names_raise():
    with pytest.raises(KeyError, match="unknown backend"):
        ab.backend_info("cudnn_flash_v9")


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
def test_the_default_is_reference_and_every_build_uses_it():
    assert ab.get_default_backend() == REFERENCE
    for name in ab.available():
        assert ab.build(name).backend == REFERENCE


def test_an_explicit_request_is_honoured():
    assert ab.build("softmax", backend="sdpa").backend == "sdpa"
    assert ab.build("softmax", backend="reference").backend == "reference"


def test_an_undeclared_backend_raises_instead_of_downgrading():
    """An explicit request is an instruction; a silent downgrade voids a benchmark."""
    with pytest.raises(ab.BackendNotSupportedError) as e:
        ab.build("topk", backend="sdpa")
    msg = str(e.value)
    assert "does not support backend 'sdpa'" in msg
    assert "'reference'" in msg          # names what it does declare
    assert "backend_matrix" in msg       # and where to see the whole picture


def test_a_declared_but_uninstalled_backend_falls_back_loudly():
    name = "flash_attn"
    if ab.is_backend_available(name):
        pytest.skip("flash_attn is installed here; the fallback path cannot be reached")
    with pytest.warns(ab.BackendFallbackWarning, match="falling back"):
        meth = ab.build("softmax", backend=name)
    assert meth.backend == REFERENCE
    assert meth.requested_backend == name


def test_the_default_is_a_preference_not_an_instruction():
    """Falling back from the *default* is quiet; falling back from a request is not."""
    with ab.default_backend("sdpa"), warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ab.build("topk").backend == REFERENCE     # topk has no sdpa path
        assert ab.build("softmax").backend == "sdpa"


def test_default_backend_context_manager_restores():
    before = ab.get_default_backend()
    with ab.default_backend("sdpa"):
        assert ab.get_default_backend() == "sdpa"
    assert ab.get_default_backend() == before


def test_set_default_backend_rejects_the_undispatchable():
    for name in ("xformers", "triton"):
        with pytest.raises(ab.BackendNotSupportedError):
            ab.set_default_backend(name)
    assert ab.get_default_backend() == REFERENCE


def test_resolve_backend_contract():
    assert resolve_backend(None, ("reference", "sdpa"), "softmax") == REFERENCE
    assert resolve_backend("sdpa", ("reference", "sdpa"), "softmax") == "sdpa"
    with pytest.raises(ab.BackendNotSupportedError):
        resolve_backend("sdpa", ("reference",), "nmf_oracle")


# ---------------------------------------------------------------------------
# the matrix
# ---------------------------------------------------------------------------
def test_the_matrix_covers_every_method_and_starts_at_reference():
    matrix = ab.backend_matrix()
    assert set(matrix) == set(ab.available())
    assert all(b[0] == REFERENCE for b in matrix.values())
    assert all(set(b) <= set(DISPATCHABLE) for b in matrix.values())


def test_most_of_the_bank_is_reference_only_and_says_so():
    """Honesty check: a reference-only declaration is the common, correct answer."""
    matrix = ab.backend_matrix()
    reference_only = [n for n, b in matrix.items() if b == (REFERENCE,)]
    assert len(reference_only) > len(matrix) // 2
    assert "nmf_oracle" in reference_only and "merge_kv" in reference_only
    assert "reference only" in str(ab.describe("nmf_oracle"))


def test_the_matrix_filter():
    assert set(ab.backend_matrix("sdpa")) == {"softmax", "flash", "gqa", "mla"}
    assert set(ab.backend_matrix("flash_attn")) == {"softmax"}


def test_backends_appear_in_the_discovery_surface():
    assert "backends" in str(ab.list_methods("exact"))
    assert ab.describe("softmax").backends == ("reference", "sdpa", "flash_attn")


# ---------------------------------------------------------------------------
# the study pin  (see attnbank/studies/__init__.py)
# ---------------------------------------------------------------------------
def test_importing_studies_pins_the_backend_to_reference():
    """A published number must not be able to move because a faster path appeared."""
    code = (
        "import attnbank as ab; "
        "ab.set_default_backend('sdpa'); "
        "import attnbank.studies; "
        "assert ab.get_default_backend() == 'reference', ab.get_default_backend(); "
        "assert ab.build('softmax').backend == 'reference'; "
        "print('pinned')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "pinned" in proc.stdout


def test_no_study_module_requests_a_backend():
    """None of the eight studies overrides the pin; a grep is the clearest assertion."""
    from pathlib import Path

    studies = Path(ab.__file__).parent / "studies"
    offenders = [p.name for p in sorted(studies.glob("S*.py"))
                 if "backend=" in p.read_text() and 'backend="reference"' not in p.read_text()]
    assert not offenders, f"these studies ask for a non-reference backend: {offenders}"


# ---------------------------------------------------------------------------
# the sdpa component
# ---------------------------------------------------------------------------
def test_sdpa_component_uses_the_end_alignment_convention():
    """`is_causal=True` would align top-left; with m < n that is a different operator."""
    from attnbank.components import sdpa_attend

    q, k, v = qkv(m=4, n=10)
    ours = sdpa_attend(q, k, v, scale=q.shape[-1] ** -0.5, causal=True)
    ref = ab.build("softmax", backend="reference")(q, k, v, causal=True).y
    assert torch.allclose(ours, ref, atol=1e-12)
    wrong = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, scale=q.shape[-1] ** -0.5)
    assert not torch.allclose(wrong, ref, atol=1e-6), \
        "if these agree the test no longer proves anything about alignment"


def test_sdpa_kernel_must_be_a_known_name():
    from attnbank.components import sdpa_attend

    q, k, v = qkv()
    with pytest.raises(ValueError, match="sdpa_kernel must be one of"):
        sdpa_attend(q, k, v, scale=1.0, kernel="flashattention3")
