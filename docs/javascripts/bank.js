/* Client-side filtering and sorting for the method browser.
 *
 * The page it runs on is complete without it: docs/gen_methods.py renders all of the
 * rows into the HTML, sorted by category, with everything searchable already on the
 * row as data-* attributes. That is the index -- there is no JSON to fetch, nothing to
 * hydrate, and no framework. This file only hides rows and reorders them, and it
 * reveals the controls itself, so a reader with scripting off never sees a dead widget.
 */
(function () {
  "use strict";

  function init() {
    var root = document.getElementById("bank");
    if (!root || root.dataset.ready === "1") return;
    root.dataset.ready = "1";

    var controls = root.querySelector(".bank-controls");
    var input = root.querySelector("#bank-q");
    var count = root.querySelector("#bank-count");
    var empty = root.querySelector("#bank-empty");
    var tbody = root.querySelector("#bank-rows");
    var rows = Array.prototype.slice.call(tbody.querySelectorAll("tr"));
    var chips = Array.prototype.slice.call(root.querySelectorAll(".bank-chip"));
    var headers = Array.prototype.slice.call(root.querySelectorAll("th[data-sort]"));
    var total = rows.length;

    rows.forEach(function (row, i) { row.dataset.order = i; });
    controls.hidden = false;

    /* ---- filtering ------------------------------------------------------ */

    // A chip is a (field, value) pair. Chips on the same field are OR-ed, different
    // fields are AND-ed: "restrict or compress, and causal" is the thing people mean.
    function active() {
      var by = {};
      chips.forEach(function (chip) {
        if (chip.getAttribute("aria-pressed") !== "true") return;
        var field = chip.dataset.filter;
        (by[field] = by[field] || []).push(chip.dataset.value);
      });
      return by;
    }

    function terms() {
      return (input.value || "").toLowerCase().split(/\s+/).filter(Boolean);
    }

    function apply() {
      var by = active();
      var words = terms();
      var shown = 0;
      rows.forEach(function (row) {
        var ok = Object.keys(by).every(function (field) {
          return by[field].indexOf(row.dataset[field]) !== -1;
        }) && words.every(function (w) {
          return row.dataset.search.indexOf(w) !== -1;
        });
        row.hidden = !ok;
        if (ok) shown += 1;
      });
      count.textContent = shown === total
        ? total + " of " + total + " methods"
        : shown + " of " + total + " methods" + (shown ? "" : " — nothing matches");
      empty.hidden = shown !== 0;
    }

    chips.forEach(function (chip) {
      chip.setAttribute("aria-pressed", "false");
      chip.addEventListener("click", function () {
        var on = chip.getAttribute("aria-pressed") === "true";
        if (!on) {
          // "oracles" and "exclude oracles" are the same field: turn the other off.
          chips.forEach(function (other) {
            if (other !== chip && other.dataset.filter === chip.dataset.filter &&
                other.dataset.value !== chip.dataset.value) {
              other.setAttribute("aria-pressed", "false");
            }
          });
        }
        chip.setAttribute("aria-pressed", on ? "false" : "true");
        apply();
      });
    });

    input.addEventListener("input", apply);
    input.addEventListener("keydown", function (event) {
      if (event.key === "Escape") { input.value = ""; apply(); }
      event.stopPropagation();          // do not hand "/" and friends to the theme
    });

    function reset() {
      input.value = "";
      chips.forEach(function (chip) { chip.setAttribute("aria-pressed", "false"); });
      apply();
    }
    Array.prototype.forEach.call(root.querySelectorAll(".bank-reset"), function (b) {
      b.addEventListener("click", reset);
    });

    /* ---- sorting -------------------------------------------------------- */

    function key(row, field) {
      if (field === "name") return row.dataset.name;
      if (field === "backends") return row.dataset.backends;
      return (row.dataset[field] || "") + " " + row.dataset.name;
    }

    headers.forEach(function (header) {
      header.setAttribute("tabindex", "0");
      header.setAttribute("role", "button");
      function sort() {
        var field = header.dataset.sort;
        var desc = header.getAttribute("aria-sort") === "ascending";
        headers.forEach(function (h) { h.removeAttribute("aria-sort"); });
        header.setAttribute("aria-sort", desc ? "descending" : "ascending");
        rows.slice()
          .sort(function (a, b) {
            var x = key(a, field), y = key(b, field);
            if (x === y) return a.dataset.order - b.dataset.order;
            return (x < y ? -1 : 1) * (desc ? -1 : 1);
          })
          .forEach(function (row) { tbody.appendChild(row); });
      }
      header.addEventListener("click", sort);
      header.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); sort(); }
      });
    });

    apply();
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
