/* Copy controls for the two things a reader of this site wants to take away: the
 * BibTeX of a method's references, and the LaTeX source of a formula.
 *
 * Both sources are already in the HTML before this file runs. The bibliography is
 * written verbatim into a <details> by docs/gen_methods.py; a display formula carries
 * its source in `data-tex`, stamped at build time by docs/tex_source.py, because by
 * the time MathJax has turned it into SVG the source is gone from the DOM. So with
 * scripting off there is no button and nothing is missing: the BibTeX is text you can
 * open and select, and the formula is still the TeX MathJax never got to.
 *
 * Nothing is fetched. The clipboard is the platform's, with the old textarea trick
 * behind it for the browsers (and the insecure origins) where it is not available.
 */
(function () {
  "use strict";

  var HOLD = 1600;        // ms the confirmation stays up

  function write(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var pad = document.createElement("textarea");
      pad.value = text;
      pad.setAttribute("readonly", "");
      pad.style.cssText = "position:fixed;top:-1000px;opacity:0";
      document.body.appendChild(pad);
      pad.select();
      var ok = false;
      try { ok = document.execCommand("copy"); } catch (err) { ok = false; }
      document.body.removeChild(pad);
      ok ? resolve() : reject(new Error("copy refused"));
    });
  }

  /** Select a node's text, so that Ctrl+C is a real instruction and not a shrug. */
  function select(node) {
    if (!node) return false;
    var range = document.createRange();
    range.selectNodeContents(node);
    var sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    return true;
  }

  /** A button that copies ``source()`` and says so, briefly, where it stands.
   *
   * ``fallback`` is the visible node holding the same text. When the clipboard is
   * refused outright the button selects it rather than telling the reader to press
   * Ctrl+C over nothing.
   */
  function button(label, title, source, fallback) {
    var el = document.createElement("button");
    el.type = "button";
    el.className = "ab-copy";
    el.title = title;
    var text = document.createElement("span");
    text.className = "ab-copy__label";
    text.textContent = label;
    el.appendChild(text);
    var said = null;
    el.addEventListener("click", function () {
      write(source()).then(function () {
        el.dataset.state = "done";
        text.textContent = "copied";
      }, function () {
        el.dataset.state = "failed";
        text.textContent = select(fallback && fallback())
          ? "selected — press ⌘/Ctrl+C"
          : "copy refused";
      }).then(function () {
        if (said) clearTimeout(said);
        said = setTimeout(function () {
          delete el.dataset.state;
          text.textContent = label;
        }, HOLD);
      });
    });
    return el;
  }

  /* ---- BibTeX ------------------------------------------------------------- */

  function initBib(box) {
    if (box.dataset.ready === "1") return;
    box.dataset.ready = "1";

    var entries = Array.prototype.slice.call(box.querySelectorAll(".attn-bib__entry"));
    var summary = box.querySelector("summary");
    var sources = {};

    entries.forEach(function (entry) {
      var src = entry.querySelector(".attn-bib__src");
      if (!src) return;
      var text = src.textContent.replace(/\s+$/, "");
      sources[entry.dataset.bibkey] = text;
      var bar = document.createElement("div");
      bar.className = "attn-bib__bar";
      bar.appendChild(button("copy", "copy this BibTeX entry",
        function () { return text; }, function () { return src; }));
      entry.insertBefore(bar, entry.firstChild);
    });

    if (summary && entries.length > 1) {
      var all = entries.map(function (e) { return sources[e.dataset.bibkey]; })
        .filter(Boolean).join("\n\n");
      summary.appendChild(button("copy all", "copy every entry cited on this page",
        function () { return all; }));
    }

    // The citation list above says `key`; that is where a reader is looking when
    // they want the entry, so that is where the button goes too.
    Array.prototype.forEach.call(document.querySelectorAll(".attn-cite[data-bibkey]"),
      function (cite) {
        var text = sources[cite.dataset.bibkey];
        if (!text || cite.dataset.ready === "1") return;
        cite.dataset.ready = "1";
        var key = cite.dataset.bibkey;
        cite.appendChild(button("BibTeX", "copy the BibTeX entry for " + key,
          function () { return text; },
          function () {
            box.open = true;            // nothing to select inside a closed details
            return box.querySelector('[data-bibkey="' + key + '"] .attn-bib__src');
          }));
      });
  }

  /* ---- LaTeX -------------------------------------------------------------- */

  function initTex(node) {
    if (node.dataset.ready === "1") return;
    var tex = node.getAttribute("data-tex");
    if (!tex) return;
    node.dataset.ready = "1";
    var bar = document.createElement("div");
    bar.className = "ab-tex-bar";
    bar.appendChild(button("LaTeX", "copy the LaTeX source of this formula",
      function () { return tex; }));
    node.parentNode.insertBefore(bar, node.nextSibling);
    node.classList.add("ab-has-tex");
  }

  function init() {
    Array.prototype.forEach.call(document.querySelectorAll("details.attn-bib"), initBib);
    Array.prototype.forEach.call(
      document.querySelectorAll("div.arithmatex[data-tex], .op-formula[data-tex]"),
      initTex);
  }

  // mkdocs-material's instant navigation swaps the <body> without a page load, so the
  // theme republishes the document on every navigation; fall back to the normal event
  // when the feature is off.
  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(init);
  } else if (document.readyState !== "loading") {
    init();
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
