// MathJax configuration for the vendored SVG build.
//
// SVG output rather than CHTML: CHTML needs ~20 MB of woff2 webfonts, whereas
// the SVG renderer embeds glyph outlines in the bundle, so tex-svg.js is the
// whole dependency and the site stays self-contained.
//
// pymdownx.arithmatex (generic: true) emits \(...\) and \[...\] inside
// <span class="arithmatex">, so only those delimiters are configured and only
// that class is scanned.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// navigation.instant swaps the DOM without a reload, so re-typeset per page.
if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    if (window.MathJax && MathJax.startup && MathJax.startup.output) {
      MathJax.startup.output.clearCache();
      MathJax.typesetClear();
      MathJax.texReset();
      MathJax.typesetPromise();
    }
  });
}
