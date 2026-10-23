"""mkdocs hook: carry every display formula's LaTeX source into the HTML.

``pymdownx.arithmatex`` with ``generic: true`` renders a display formula as

    <div class="arithmatex">\\[ ... \\]</div>

and MathJax then replaces the contents with SVG, taking the source with it. The copy
control in ``docs/javascripts/clip.js`` wants that source, so it is stamped onto the
element here, while the page is being built -- not scraped in the browser, where it is
a race against the typesetter and loses it on the pages that typeset fastest.

Display formulas only. An inline ``\\(n\\)`` is part of a sentence; hanging a control
off every one of them would be noise, and the sentence is the thing to copy anyway.

Registered from ``mkdocs.yml`` under ``hooks:``. It reads nothing and writes nothing
outside the page it is handed.
"""
from __future__ import annotations

import html
import re

#: What arithmatex emits for a display formula, non-greedy so two in a row stay two.
_DISPLAY = re.compile(r'<div class="arithmatex">\\\[(.*?)\\\]</div>', re.S)


def _stamp(match: "re.Match[str]") -> str:
    body = match.group(1)
    # the attribute wants the TeX as the author typed it: undo the HTML escaping
    # markdown applied, then escape once for an attribute value
    tex = html.escape(html.unescape(body).strip(), quote=True)
    return f'<div class="arithmatex" data-tex="{tex}">\\[{body}\\]</div>'


def on_page_content(html_: str, **_kwargs) -> str:
    """Add ``data-tex`` to every display formula on the page."""
    return _DISPLAY.sub(_stamp, html_)
