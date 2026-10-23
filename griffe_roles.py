"""Translate Sphinx roles in docstrings as griffe loads them, for the API reference.

The library's docstrings are RST: they say ``:func:`attnbank.build``` and
``:class:`~attnbank.core.MethodInfo```, and they are read against the monograph and
quoted in the paper, so they stay that way.

``docs/gen_methods.py`` translates them for the pages it writes. The API reference
pages do not go through it -- mkdocstrings renders those from the source, and
``docstring_style: numpy`` does not know what an RST role is, so ``:func:`` and its
siblings reached the built page verbatim. This extension applies
:func:`gen_methods.strip_roles` -- the same substitution, not a second one -- to every
docstring in the object tree, before anything parses or renders it. Nothing on disk
changes; the rewrite lives for the length of the build.

Doing it here rather than in a ``hooks:`` entry on the rendered HTML is what makes it
cover every docstring, present and future: it runs on the docstring while it is still
the text the author wrote, so the one regex in ``gen_methods`` is the whole rule. By
the time the page is HTML the role has become ``:func:<code>...</code>`` and matching
it would need a second, different pattern.

Wired up in ``mkdocs.yml``:

    plugins:
      - mkdocstrings:
          handlers:
            python:
              options:
                extensions:
                  - docs/griffe_roles.py:SphinxRoles
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from griffe import Extension, Object

# griffe imports this file by path, so its own directory is not necessarily importable
# -- and ``gen_methods`` lives next to it.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from gen_methods import strip_roles  # noqa: E402


class SphinxRoles(Extension):
    """Rewrite ``:role:`target``` to a code span in every docstring griffe loads."""

    def on_object(self, *, obj: Object, **kwargs: Any) -> None:  # noqa: ARG002
        """Run once the object tree is complete, so every docstring is seen exactly once.

        ``Docstring.parsed`` is a cached property, so rewriting ``value`` here lands
        before the numpy parser ever looks at it.
        """
        docstring = getattr(obj, "docstring", None)
        if docstring is not None and docstring.value:
            docstring.value = strip_roles(docstring.value)
