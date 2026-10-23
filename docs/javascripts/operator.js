/* The operator plate: the formula and the seven category keys.
 *
 * The page is complete before this file runs. The formula and everything each
 * category changes are in the HTML that mkdocs renders; this file only builds the
 * key row -- which is why it is marked `hidden` in the source and revealed here --
 * and lights the symbol of the operator that the chosen category changes.
 *
 * There is no data table in here on purpose. Which symbols a category touches is a
 * fact about the library, so it lives on the element as `data-syms` and is written by
 * docs/gen_taxonomy.py from the registry -- never duplicated in JavaScript.
 *
 * `figure` and `symsOf` are published on `window.AttnOperator` because the taxonomy
 * tree drives the same plate; taxonomy.js consumes them and must load after this.
 */
(function () {
  "use strict";

  var SYMS = ["y", "int", "v", "pi", "g", "mu", "Z"];

  /* ---- the operator figure ------------------------------------------------
   * One plate per page. `syms` is the set of symbols to light; an empty set with
   * `frame` asks for the whole operator to be outlined instead, which is how
   * `exact` -- the category that changes nothing -- says so.
   */
  function figure(root) {
    var formula = root.querySelector(".op-formula");
    var replacement = root.querySelector(".op-replacement");
    var cells = formula ? formula.querySelectorAll(".sym") : [];

    return function show(cat, syms, frame, replace) {
      if (!formula) return;
      formula.classList.toggle("is-focused", !!cat && !frame);
      formula.classList.toggle("is-framed", !!cat && !!frame);
      formula.setAttribute("data-cat", cat || "");
      Array.prototype.forEach.call(cells, function (cell) {
        cell.classList.toggle("is-lit", syms.indexOf(cell.dataset.s) !== -1);
      });
      if (replacement) replacement.hidden = !replace;
    };
  }

  function symsOf(el) {
    return (el.getAttribute("data-syms") || "").split(/\s+/).filter(function (s) {
      return SYMS.indexOf(s) !== -1;
    });
  }

  /* ---- home: the key row over the readout -------------------------------- */

  function initPlate(root) {
    if (root.dataset.ready === "1") return;
    var panels = Array.prototype.slice.call(root.querySelectorAll(".op-panel"));
    var keys = root.querySelector(".op-keys");
    var prompt = root.querySelector(".op-prompt");
    if (!panels.length || !keys) return;
    root.dataset.ready = "1";

    var show = figure(root);
    var buttons = panels.map(function (panel) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "op-key";
      button.textContent = panel.dataset.label || panel.dataset.cat;
      button.setAttribute("data-cat", panel.dataset.cat);
      button.setAttribute("aria-pressed", "false");
      keys.appendChild(button);
      return button;
    });
    keys.hidden = false;

    var active = null;

    function select(cat) {
      active = active === cat ? null : cat;
      buttons.forEach(function (button) {
        button.setAttribute("aria-pressed",
          String(button.getAttribute("data-cat") === active));
      });
      panels.forEach(function (panel) {
        panel.hidden = panel.dataset.cat !== active;
      });

      var panel = panels.filter(function (p) { return p.dataset.cat === active; })[0];
      show(active, panel ? symsOf(panel) : [],
           panel && panel.hasAttribute("data-frame"),
           panel && panel.hasAttribute("data-replace"));
    }

    buttons.forEach(function (button, i) {
      button.addEventListener("click", function () {
        select(button.getAttribute("data-cat"));
      });
      // a key row is a single widget; arrows walk it, as they do in a toolbar
      button.addEventListener("keydown", function (event) {
        var step = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: 1, ArrowUp: -1 }[event.key];
        if (!step) return;
        event.preventDefault();
        buttons[(i + step + buttons.length) % buttons.length].focus();
      });
    });

    if (prompt) prompt.hidden = false;
    // Open on `exact`, which is not an arbitrary pick: it is the category that
    // changes nothing, so the plate's first statement is the operator intact.
    select("exact");
  }

  /* ---- taxonomy: the tree ------------------------------------------------
   * Every branch is open in the HTML. Here each heading becomes the control that
   * closes it, so with scripting off the page is the whole taxonomy, in order,
   * with nothing to click and nothing missing.
   */

  window.AttnOperator = { SYMS: SYMS, figure: figure, symsOf: symsOf };

  function init() {
    Array.prototype.forEach.call(document.querySelectorAll(".op-readout"),
      function (el) { initPlate(el.closest("[data-op]") || el); });
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
