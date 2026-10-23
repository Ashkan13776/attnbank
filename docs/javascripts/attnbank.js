/* The operator plate and the taxonomy tree.
 *
 * Both pages are complete before this file runs. The formula, the seven categories,
 * everything each of them changes, and the full method table are all in the HTML that
 * mkdocs renders; this file only (a) builds the key row and collapses the branches,
 * which is why both are marked `hidden` / open in the source and revealed here, and
 * (b) lights the symbol of the operator that the chosen category changes.
 *
 * There is no data table in here on purpose. Which symbols a category touches is a
 * fact about the library, so it lives on the element as `data-syms` and is written by
 * docs/gen_taxonomy.py from the registry -- never duplicated in JavaScript.
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
  function initTree(root) {
    if (root.dataset.ready === "1") return;
    var nodes = Array.prototype.slice.call(root.querySelectorAll(".tx-node"));
    if (!nodes.length) return;
    root.dataset.ready = "1";

    var plate = root.closest("[data-op]") || document;
    var show = figure(plate);
    var items = [];

    nodes.forEach(function (node, i) {
      var heading = node.querySelector("h2");
      var body = node.querySelector(".tx-body");
      if (!heading || !body) return;

      if (!body.id) body.id = "tx-body-" + (heading.id || i);
      var button = document.createElement("button");
      button.type = "button";
      button.className = "tx-toggle";
      button.setAttribute("aria-expanded", "false");
      button.setAttribute("aria-controls", body.id);
      // Material appends its own permalink to the heading; it stays a link, outside
      // the button, so the anchor is still copyable.
      Array.prototype.slice.call(heading.childNodes).forEach(function (child) {
        if (child.nodeType === 1 && child.classList.contains("headerlink")) return;
        button.appendChild(child);
      });
      heading.insertBefore(button, heading.firstChild);
      node.dataset.open = "false";
      items.push({ node: node, button: button, body: body });
    });

    function open(which, focus) {
      items.forEach(function (item) {
        var on = item === which;
        item.node.dataset.open = String(on);
        item.button.setAttribute("aria-expanded", String(on));
      });
      var cat = which ? which.node.getAttribute("data-cat") : null;
      show(cat, which ? symsOf(which.node) : [],
           which && which.node.hasAttribute("data-frame"),
           which && which.node.hasAttribute("data-replace"));
      if (which && focus) which.button.focus();
    }

    items.forEach(function (item, i) {
      item.button.addEventListener("click", function () {
        open(item.node.dataset.open === "true" ? null : item, false);
      });
      item.button.addEventListener("keydown", function (event) {
        var step = { ArrowDown: 1, ArrowUp: -1 }[event.key];
        if (!step) return;
        event.preventDefault();
        items[(i + step + items.length) % items.length].button.focus();
      });
    });

    // `taxonomy.md#restrict` is linked from every method page and from the home
    // page, so an incoming hash has to open the branch it names.
    function fromHash() {
      var id = (location.hash || "").slice(1);
      if (!id) return false;
      var hit = items.filter(function (item) {
        return item.node.querySelector("h2").id === id || item.node.id === id;
      })[0];
      if (hit) open(hit, false);
      return !!hit;
    }

    if (!fromHash()) open(null, false);
    window.addEventListener("hashchange", fromHash);
  }

  function init() {
    Array.prototype.forEach.call(document.querySelectorAll(".op-readout"),
      function (el) { initPlate(el.closest("[data-op]") || el); });
    Array.prototype.forEach.call(document.querySelectorAll(".tx-tree"), initTree);
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
