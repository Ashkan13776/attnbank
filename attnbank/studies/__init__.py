"""The eight studies of ``STUDIES.md``, one module each, on a shared harness.

``_harness`` owns the sweep runner, the resumable cache, the result files and the
median/IQR summariser; ``attnbank.data`` owns the activations each study runs on.

**Execution backend: pinned to ``reference``.**
Every number in the monograph came out of these modules, and a published figure must
not be able to move because the library grew a faster execution path. Importing
``attnbank.studies`` therefore pins the process-wide default backend to ``reference``
(:func:`attnbank.set_default_backend`) -- the plain-PyTorch implementations that the
conformance suite anchors everything else to. The pin is deliberate, visible and
asserted in ``tests/test_backends.py``; a study that wanted a different backend would
have to say so in its own ``ab.build(..., backend=...)`` call, and none does.

``reference`` is also the library's out-of-the-box default, so this line is a belt and
braces against a future change of that default, not a correction of the current one.
"""
from ..backends import REFERENCE, set_default_backend

#: pinned before any study module is imported; see the note above
set_default_backend(REFERENCE)

from ._harness import (  # noqa: E402  (the pin above must run first)
    DEFAULT_DTYPE,
    Sweep,
    SweepContext,
    default_device,
    device_info,
    generator,
    grid,
    load_results,
    provenance,
    results_dir,
    save_results,
    set_seed,
    summarize,
    to_rows,
    zip_grid,
)

__all__ = ["REFERENCE", "set_default_backend", "DEFAULT_DTYPE", "Sweep",
           "SweepContext", "default_device", "device_info", "generator", "grid",
           "load_results", "provenance", "results_dir", "save_results", "set_seed",
           "summarize", "to_rows", "zip_grid"]
