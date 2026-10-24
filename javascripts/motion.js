/* Motion and microinteractions, site-wide.
 *
 * Owned by one aspect of the site redesign; see lib/SITE_DESIGN.md.
 * Everything here must be inert under `prefers-reduced-motion: reduce`.
 *
 * Two jobs, both progressive -- with scripting off the site simply has no
 * category ticks in the nav and no page-entrance motion:
 *
 *   1. Stamp `data-cat` on the seven category groups in the left nav, so
 *      chrome.css can draw each group's hue tick and the active page's rule.
 *      The nav is plain theme markup with no category data, so the label text
 *      is matched against the seven registry names.
 *
 *   2. Re-trigger the page-entrance animation (motion.css) when
 *      navigation.instant swaps a page in. Never on the first render -- that
 *      page is already on screen -- and never under reduced motion.
 */
(function () {
  "use strict";

  var CATS = ["exact", "interaction", "restrict", "compress",
              "factorize", "coupling", "recurrent"];
  var seen = false;

  function stampNav() {
    var links = document.querySelectorAll(
      ".md-nav--primary .md-nav__item--nested > .md-nav__link");
    Array.prototype.forEach.call(links, function (link) {
      var name = (link.textContent || "").trim().toLowerCase();
      if (CATS.indexOf(name) !== -1) {
        link.parentElement.setAttribute("data-cat", name);
      }
    });
  }

  function enter() {
    if (!seen) { seen = true; return; }
    if (window.matchMedia &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      return;
    }
    var root = document.documentElement;
    var content = document.querySelector(".md-content");
    if (!content) return;
    root.classList.remove("ab-page-enter");
    void content.offsetWidth; /* restart the animation from frame zero */
    root.classList.add("ab-page-enter");
    content.addEventListener("animationend", function () {
      root.classList.remove("ab-page-enter");
    }, { once: true });
  }

  function init() {
    stampNav();
    enter();
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
