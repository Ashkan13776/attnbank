/* The taxonomy tree on docs/taxonomy.md.
 *
 * The seven branches and the full method table are rendered by docs/gen_taxonomy.py;
 * this file collapses the branches and keeps the operator plate above them in step
 * with whichever branch is open. It drives that plate through `window.AttnOperator`,
 * published by operator.js, which must load first.
 */
(function () {
  "use strict";

  function figure(root) { return window.AttnOperator.figure(root); }
  function symsOf(el) { return window.AttnOperator.symsOf(el); }

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

    // the plate lights the symbols of `which`; `current` is the pinned branch that
    // the plate falls back to when a hover or focus preview ends
    var current = null;

    function light(which) {
      var cat = which ? which.node.getAttribute("data-cat") : null;
      show(cat, which ? symsOf(which.node) : [],
           which && which.node.hasAttribute("data-frame"),
           which && which.node.hasAttribute("data-replace"));
    }

    function open(which, focus) {
      items.forEach(function (item) {
        var on = item === which;
        item.node.dataset.open = String(on);
        item.button.setAttribute("aria-expanded", String(on));
      });
      current = which;
      light(current);
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

      // Pointing at or tabbing through a branch previews its symbol on the plate;
      // leaving restores whichever branch is pinned open. A state change, not an
      // animation, so there is nothing to still under prefers-reduced-motion.
      item.node.addEventListener("mouseenter", function () { light(item); });
      item.node.addEventListener("mouseleave", function () { light(current); });
      item.node.addEventListener("focusin", function () { light(item); });
      item.node.addEventListener("focusout", function (event) {
        if (!item.node.contains(event.relatedTarget)) light(current);
      });
    });

    // `taxonomy.md#restrict` is linked from every method page and from the home
    // page, so an incoming hash has to open the branch it names.
    function fromHash() {
      // instant navigation replaces the page without unloading it, so a listener
      // from a previous visit may still be here, holding a detached tree
      if (!root.isConnected) {
        window.removeEventListener("hashchange", fromHash);
        return false;
      }
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
