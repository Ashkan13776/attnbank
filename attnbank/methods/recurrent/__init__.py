"""Methods in category 'recurrent': finite-state operators.

Modules are discovered rather than listed: dropping a new ``<name>.py`` into
this directory registers it. The explicit import list this replaces was a file
every contributor had to edit, which made it the one guaranteed merge conflict
when several methods were added at once.
"""
from __future__ import annotations

import importlib as _importlib
import pkgutil as _pkgutil

__all__: list[str] = []

for _info in _pkgutil.iter_modules(__path__):
    if _info.name.startswith("_"):
        continue
    _importlib.import_module(f"{__name__}.{_info.name}")
    __all__.append(_info.name)

__all__.sort()
del _importlib, _pkgutil, _info
