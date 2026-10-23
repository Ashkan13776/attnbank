/* The method browser on docs/methods/index.md: filtering, sorting, side-by-side
 * comparison, and the URL that holds all three.
 *
 * The page it runs on is complete without it: docs/gen_methods.py renders all of the
 * rows into the HTML, sorted by category, with everything searchable already on the
 * row as data-* attributes. That is the index -- there is no JSON to fetch, nothing to
 * hydrate, and no framework. This file hides rows, reorders them, and reads one
 * server-rendered JSON block for the comparison; it reveals every control it owns, and
 * inserts the checkboxes itself, so a reader with scripting off never sees a dead
 * widget.
 *
 * The comparison shows *differences*. Rows the selected methods agree on are folded
 * into a disclosure rather than repeated across four columns, and a property that is
 * not declared by all of the selected methods is not shown at all -- there is no cell
 * in here that the registry did not write.
 */
(function () {
  "use strict";

  var MIN_PICK = 2;
  var MAX_PICK = 4;

  /* Chip fields, in URL order. Each is a data-* attribute on the row and a
   * `data-filter` on its chips; the two are the same name on purpose. */
  var FIELDS = ["category", "objective", "causal", "trainable", "fused", "oracle"];

  /* What the side-by-side view can show, in order, grouped. Nothing here is a value:
   * every value comes from the #bank-data block, which gen_methods.py writes from
   * ab.describe() and the method's own cost(). A key missing from a method's entry is
   * a property that method does not declare, and the row is dropped rather than
   * guessed at. */
  var GROUPS = [
    { title: "what it is", props: [
      { key: "category", label: "category", kind: "code" },
      { key: "objective", label: "objective", kind: "code" },
      { key: "paper_section", label: "monograph", kind: "code" }
    ] },
    { title: "what it declares",
      sub: "every row is a commitment the conformance suite tests",
      props: [
        { key: "supports_causal", label: "supports_causal", kind: "flag" },
        { key: "expresses_weights", label: "expresses_weights", kind: "flag" },
        { key: "row_stochastic", label: "row_stochastic", kind: "flag" },
        { key: "head_independent", label: "head_independent", kind: "flag" },
        { key: "trainable", label: "trainable", kind: "flag" },
        { key: "oracle", label: "oracle", kind: "flag" },
        { key: "backends", label: "backends", kind: "list" },
        { key: "anchor", label: "correctness anchor", kind: "text" },
        { key: "config", label: "configuration", kind: "text" }
      ] },
    { title: "cost model", cost: true, props: [
      { key: "flops", label: "flops", kind: "num" },
      { key: "state_bytes", label: "state_bytes", kind: "num" },
      { key: "bytes_moved", label: "bytes_moved", kind: "num" }
    ] },
    { title: "references", props: [
      { key: "references", label: "BibTeX keys", kind: "list" }
    ] }
  ];

  /* ---- small helpers ------------------------------------------------------ */

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  /** Python's ``format(x, '.3g')``, so a ratio here reads as it does on the pages. */
  function g3(x) {
    if (!isFinite(x)) return "—";
    var s = Number(x).toPrecision(3);
    if (s.indexOf("e") === -1 && s.indexOf(".") !== -1) s = s.replace(/\.?0+$/, "");
    return s;
  }

  function num(x) {
    return Number(x).toLocaleString("en-US", { maximumFractionDigits: 0 });
  }

  /** The string a row's agreement is decided on. One definition, used everywhere. */
  function canon(value) {
    if (Array.isArray(value)) return value.join(", ");
    if (value === true) return "yes";
    if (value === false) return "no";
    return String(value);
  }

  /* ---- the "/" shortcut ---------------------------------------------------
   * Registered once on the document, which instant navigation does not replace, and
   * resolved against the DOM at keypress time -- so it is inert on every page that
   * has no bank, and the theme keeps "/" to itself there.
   */
  if (!window.__attnbankSlash) {
    window.__attnbankSlash = true;
    document.addEventListener("keydown", function (event) {
      if (event.key !== "/" || event.ctrlKey || event.metaKey || event.altKey) return;
      var box = document.getElementById("bank-q");
      if (!box || box.offsetParent === null) return;
      var target = event.target;
      if (target && (target.isContentEditable ||
          /^(input|textarea|select)$/i.test(target.tagName || ""))) return;
      event.preventDefault();
      event.stopPropagation();        // the theme binds "/" to its own search box
      box.focus();
      box.select();
    }, true);
  }

  /* ---- the browser -------------------------------------------------------- */

  function init() {
    var root = document.getElementById("bank");
    if (!root || root.dataset.ready === "1") return;
    root.dataset.ready = "1";

    var controls = root.querySelector(".bank-controls");
    var input = root.querySelector("#bank-q");
    var count = root.querySelector("#bank-count");
    var empty = root.querySelector("#bank-empty");
    var tbody = root.querySelector("#bank-rows");
    var rows = Array.prototype.slice.call(tbody.querySelectorAll("tr[data-name]"));
    /* the category section rows: server-rendered, so the taxonomy grouping is there
     * with scripting off; this file only hides them while a non-default sort is on
     * and keeps their counts honest while filters are */
    var groupRows = Array.prototype.slice.call(tbody.querySelectorAll("tr.bank-group"));
    var kids = Array.prototype.slice.call(tbody.children);   // server order, groups interleaved
    var catOrder = groupRows.map(function (g) { return g.dataset.cat; });
    var chips = Array.prototype.slice.call(root.querySelectorAll(".bank-chip[data-filter]"));
    var headers = Array.prototype.slice.call(root.querySelectorAll("th[data-sort]"));
    var total = rows.length;

    var panel = root.querySelector("#bank-compare");
    var panelTable = root.querySelector("#bank-compare-table");
    var panelHint = root.querySelector("#bank-compare-hint");
    var panelNote = root.querySelector("#bank-compare-note");
    var copyLink = root.querySelector("#bank-compare-link");
    var clearPicks = root.querySelector("#bank-compare-clear");
    var pickHint = root.querySelector("#bank-pick-hint");
    var blob = root.querySelector("#bank-data");

    var data = { shape: {}, methods: {} };
    if (blob) {
      try { data = JSON.parse(blob.textContent); } catch (err) { data = { methods: {} }; }
    }

    var byName = {};
    rows.forEach(function (row, i) {
      row.dataset.order = i;
      byName[row.dataset.name] = row;
    });
    controls.hidden = false;

    var picked = [];              // method names, in the order they were ticked
    var boxes = {};               // name -> checkbox
    var showSame = false;
    var sortField = null;
    var sortDesc = false;
    var restoring = false;
    var urlTimer = null;

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

    /* The section rows show whenever the table is in the server's taxonomy order --
     * no sort, or category ascending, which is the same order. Any other sort
     * interleaves the categories, so the sections would lie, and they hide. */
    function groupsShown() {
      return !sortField || (sortField === "category" && !sortDesc);
    }

    function syncGroups() {
      var show = groupsShown();
      groupRows.forEach(function (g) {
        var cat = g.dataset.cat;
        var all = Number(g.dataset.n) || 0;
        var vis = 0;
        rows.forEach(function (row) {
          if (row.dataset.category === cat && !row.hidden) vis += 1;
        });
        g.hidden = !show || vis === 0;
        var label = g.querySelector(".bank-group__n");
        if (label) {
          label.textContent = (vis === all ? all : vis + " of " + all) + " methods";
        }
      });
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
      syncGroups();
      pushUrl();
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
      applySort(null, false);          // back to the taxonomy order, sections shown
      apply();
    }
    Array.prototype.forEach.call(root.querySelectorAll(".bank-reset[data-reset], #bank-reset"),
      function (b) { b.addEventListener("click", reset); });

    /* ---- sorting -------------------------------------------------------- */

    function key(row, field) {
      // the first column leads with the human name, so that is what sorting it means
      if (field === "name") return row.dataset.title || row.dataset.name;
      if (field === "backends") return row.dataset.backends;
      if (field === "category") {
        // taxonomy order, not alphabetical: "category ascending" means the order the
        // taxonomy gives the categories, which is also the server's default order
        var i = catOrder.indexOf(row.dataset.category);
        return i < 0 ? "99" : ("0" + i).slice(-2);
      }
      return (row.dataset[field] || "") + " " + row.dataset.name;
    }

    function applySort(field, desc) {
      sortField = field || null;
      sortDesc = !!desc;
      headers.forEach(function (h) {
        if (field && h.dataset.sort === field) {
          h.setAttribute("aria-sort", desc ? "descending" : "ascending");
        } else {
          h.removeAttribute("aria-sort");
        }
      });
      if (groupsShown()) {
        // the server's order, section rows interleaved where they belong
        kids.forEach(function (node) { tbody.appendChild(node); });
      } else {
        rows.slice()
          .sort(function (a, b) {
            var x = key(a, field), y = key(b, field);
            if (x === y) return a.dataset.order - b.dataset.order;
            return (x < y ? -1 : 1) * (desc ? -1 : 1);
          })
          .forEach(function (row) { tbody.appendChild(row); });
      }
      syncGroups();
      pushUrl();
    }

    headers.forEach(function (header) {
      header.setAttribute("tabindex", "0");
      header.setAttribute("role", "button");
      function sort() {
        applySort(header.dataset.sort,
                  header.getAttribute("aria-sort") === "ascending");
      }
      header.addEventListener("click", sort);
      header.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); sort(); }
      });
    });

    /* ---- selection ------------------------------------------------------ */

    rows.forEach(function (row) {
      var name = row.dataset.name;
      if (!data.methods[name]) return;              // nothing to put in a column
      var cell = row.querySelector("th");
      var box = el("input", "bank-pick");
      box.type = "checkbox";
      box.setAttribute("aria-label", "compare " + (row.dataset.title || name));
      box.addEventListener("change", function () {
        if (box.checked) {
          if (picked.indexOf(name) === -1) picked.push(name);
        } else {
          picked = picked.filter(function (n) { return n !== name; });
        }
        syncPicks();
      });
      cell.insertBefore(box, cell.firstChild);
      boxes[name] = box;
    });

    function syncPicks() {
      picked = picked.slice(0, MAX_PICK);
      Object.keys(boxes).forEach(function (name) {
        var on = picked.indexOf(name) !== -1;
        boxes[name].checked = on;
        boxes[name].disabled = !on && picked.length >= MAX_PICK;
        if (byName[name]) byName[name].dataset.picked = on ? "1" : "0";
      });
      render();
      updateTray();
      pushUrl();
    }

    if (clearPicks) {
      clearPicks.addEventListener("click", function () { picked = []; syncPicks(); });
    }

    /* The whole view — search, filters, sort, picks — is already in the address bar;
     * this button only makes that fact visible and puts it on the clipboard. */
    if (copyLink) {
      copyLink.addEventListener("click", function () {
        writeUrl();                       // flush the debounce so the URL is current
        function done(ok) {
          copyLink.textContent = ok ? "copied" : "copy failed — use the address bar";
          setTimeout(function () { copyLink.textContent = "copy link"; }, 2000);
        }
        function fallback() {             // non-HTTPS and other clipboard-API refusals
          var scratch = el("textarea");
          scratch.value = location.href;
          scratch.setAttribute("readonly", "");
          scratch.style.position = "fixed";
          scratch.style.opacity = "0";
          document.body.appendChild(scratch);
          scratch.select();
          var ok = false;
          try { ok = document.execCommand("copy"); } catch (err) { ok = false; }
          document.body.removeChild(scratch);
          done(ok);
        }
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(location.href).then(
            function () { done(true); }, fallback);
        } else {
          fallback();
        }
      });
    }

    /* ---- the tray -------------------------------------------------------
     * Ticking a checkbox sixty rows down must change something the reader can see,
     * and the panel it fills is at the top of the page. A small fixed tray holds the
     * picks while the panel is off-screen, and gets out of the way once it is not.
     * Created here, so with scripting off it does not exist.
     */

    var tray = el("div", "bank-tray");
    tray.hidden = true;
    var trayChips = el("span", "bank-tray__chips");
    var trayStatus = el("span", "bank-tray__status");
    var trayView = el("a", "bank-tray__view", "view ↑");
    trayView.href = "#bank-compare";
    var trayClear = el("button", "bank-reset", "clear");
    trayClear.type = "button";
    trayClear.addEventListener("click", function () { picked = []; syncPicks(); });
    tray.appendChild(el("span", "bank-tray__label", "side by side"));
    tray.appendChild(trayChips);
    tray.appendChild(trayStatus);
    tray.appendChild(trayView);
    tray.appendChild(trayClear);
    root.appendChild(tray);

    var panelOnScreen = false;
    if (panel && typeof IntersectionObserver !== "undefined") {
      new IntersectionObserver(function (entries) {
        panelOnScreen = entries[0].isIntersecting;
        updateTray();
      }).observe(panel);
    }

    function updateTray() {
      var ready = picked.length >= MIN_PICK;
      tray.hidden = !picked.length || (ready && panelOnScreen);
      trayChips.textContent = "";
      picked.forEach(function (name) {
        var m = data.methods[name];
        var chip = el("span", "bank-tray__chip");
        chip.appendChild(el("span", null, m ? m.title : name));
        var x = el("button", "bank-tray__x", "×");
        x.type = "button";
        x.setAttribute("aria-label", "remove " + name + " from the comparison");
        x.addEventListener("click", function () {
          picked = picked.filter(function (n) { return n !== name; });
          syncPicks();
        });
        chip.appendChild(x);
        trayChips.appendChild(chip);
      });
      trayStatus.textContent = !picked.length ? ""
        : picked.length < MIN_PICK ? "tick one more to compare"
        : picked.length + " of " + MAX_PICK;
      trayView.hidden = !ready;
    }

    /* ---- the side-by-side table ----------------------------------------- */

    function cellFor(prop, value, ref, max) {
      var td = el("td");
      td.dataset.kind = prop.kind;
      if (prop.kind === "flag") {
        td.appendChild(el("span", "bank-cmp__flag", value ? "yes" : "no"));
        td.dataset.flag = value ? "yes" : "no";
      } else if (prop.kind === "code") {
        td.appendChild(el("code", null, String(value)));
      } else if (prop.kind === "list") {
        if (!value.length) {
          td.appendChild(el("span", "bank-cmp__none", "none"));
        } else {
          value.forEach(function (item, i) {
            if (i) td.appendChild(document.createTextNode(" "));
            td.appendChild(el("code", null, item));
          });
        }
      } else if (prop.kind === "num") {
        // The bar is linear in the quantity itself, and the longest bar in a row is
        // the largest *selected* value -- columns are equal width (table-layout:
        // fixed), so equal percentages are equal lengths. The number is still the
        // declaration; the bar only ranks it.
        if (max > 0) {
          var track = el("span", "bank-cmp__track");
          track.setAttribute("aria-hidden", "true");
          var fill = el("span", "bank-cmp__fill");
          fill.style.width = Math.max(100 * value / max, 0.5).toFixed(2) + "%";
          track.appendChild(fill);
          td.appendChild(track);
        }
        if (ref) {
          td.appendChild(el("span", "bank-cmp__ratio", g3(value / ref) + "×"));
        }
        td.appendChild(el("span", "bank-cmp__num", num(value)));
      } else {
        td.textContent = String(value);
      }
      return td;
    }

    function headCell(name) {
      var m = data.methods[name];
      var th = el("th");
      th.scope = "col";
      // the category hue, as a rule over the column -- the same hue this category
      // carries in every monograph figure; never behind text
      if (m.category) th.dataset.cat = m.category;
      var link = el("a", "bank-cmp__name", m.title);
      link.href = name + "/";
      th.appendChild(link);
      th.appendChild(el("span", "bank-cmp__key", name));
      var drop = el("button", "bank-cmp__drop", "×");
      drop.type = "button";
      drop.title = "remove " + name + " from the comparison";
      drop.setAttribute("aria-label", "remove " + name + " from the comparison");
      drop.addEventListener("click", function () {
        picked = picked.filter(function (n) { return n !== name; });
        syncPicks();
      });
      th.appendChild(drop);
      th.appendChild(el("p", "bank-cmp__summary", m.summary));
      return th;
    }

    function render() {
      if (!panel) return;
      panel.hidden = picked.length < MIN_PICK;
      panelTable.textContent = "";
      if (panelHint) {
        panelHint.textContent = picked.length
          ? picked.length + " of " + MAX_PICK + " selected"
          : "";
      }
      // the prompt above the table, which is where a reader who has ticked one box
      // is looking -- the panel it would otherwise appear in is still hidden
      if (pickHint) {
        pickHint.textContent =
          picked.length === 0 ? "Tick 2–4 boxes to compare those methods column by column." :
          picked.length === 1 ? "One ticked — tick one more to put them side by side." :
          picked.length + " ticked, side by side below." +
            (picked.length === MAX_PICK ? " Four is the most that fits in columns." : "");
      }
      if (picked.length < MIN_PICK) return;

      var mats = picked.map(function (n) { return data.methods[n]; });
      var refCost = data.methods.softmax || {};

      var thead = el("thead");
      var hrow = el("tr");
      hrow.appendChild(el("th", "bank-cmp__corner", "property"));
      picked.forEach(function (n) { hrow.appendChild(headCell(n)); });
      thead.appendChild(hrow);
      panelTable.appendChild(thead);

      var tbodyC = el("tbody");
      var nSame = 0, nDiff = 0;
      var undeclared = [];
      var hasCost = false;

      GROUPS.forEach(function (group) {
        var body = [];
        group.props.forEach(function (prop) {
          // Only show a property every selected method declares. A half-filled row
          // invites the reader to read a blank as a "no", and a blank is not a
          // declaration.
          var have = mats.every(function (m) {
            return Object.prototype.hasOwnProperty.call(m, prop.key);
          });
          if (!have) {
            if (mats.some(function (m) {
              return Object.prototype.hasOwnProperty.call(m, prop.key);
            })) undeclared.push(prop.label);
            return;
          }
          var vals = mats.map(function (m) { return m[prop.key]; });
          var keys = vals.map(canon);
          var agree = keys.every(function (k) { return k === keys[0]; });

          var tr = el("tr");
          tr.dataset.same = agree ? "1" : "0";
          if (agree) { nSame += 1; } else { nDiff += 1; }
          var th = el("th", null, prop.label);
          th.scope = "row";
          tr.appendChild(th);

          // the odd one out, when there is one: a value held by fewer methods than
          // the most common value in the row. A 2-2 split has no odd one out and is
          // left unmarked rather than given an arbitrary winner.
          var freq = {};
          keys.forEach(function (k) { freq[k] = (freq[k] || 0) + 1; });
          var top = Math.max.apply(null, keys.map(function (k) { return freq[k]; }));

          var max = group.cost ? Math.max.apply(null, vals.map(Number)) : 0;
          vals.forEach(function (value, i) {
            var td = cellFor(prop, value, group.cost ? refCost[prop.key] : 0, max);
            // the bar takes the method's category hue, resolved by [data-cat];
            // numeric rows get no odd-one-out mark -- the bars already rank them,
            // and "odd" is a categorical idea, not a quantitative one
            if (group.cost) td.dataset.cat = mats[i].category;
            if (!agree && prop.kind !== "num" && freq[keys[i]] < top) td.dataset.odd = "1";
            tr.appendChild(td);
          });
          body.push(tr);
        });
        if (!body.length) return;
        if (group.cost) hasCost = true;
        var head = el("tr", "bank-cmp__group");
        var hcell = el("th");
        hcell.appendChild(el("span", "bank-cmp__gt", group.title));
        var sub = group.cost
          ? "analytic, per (batch, head), at " + shapeText() +
            (data.bytes_per_el ? ", bytes_per_el = " + data.bytes_per_el : "") +
            " — × against softmax"
          : group.sub;
        if (sub) hcell.appendChild(el("span", "bank-cmp__gs", sub));
        hcell.colSpan = picked.length + 1;
        hcell.scope = "colgroup";
        head.appendChild(hcell);
        head.dataset.same = body.every(function (r) { return r.dataset.same === "1"; })
          ? "1" : "0";
        tbodyC.appendChild(head);
        body.forEach(function (r) { tbodyC.appendChild(r); });
      });

      // The fold lives where the folded rows are: a diff-style expander row at the
      // foot of the table, like an elided hunk. Nothing differs => the agreed rows
      // are the whole answer, so they are forced visible and there is no expander.
      var forced = nDiff === 0;
      if (nSame && !forced) {
        var fold = el("tr", "bank-cmp__fold");
        var foldCell = el("td");
        foldCell.colSpan = picked.length + 1;
        var foldBtn = el("button", "bank-cmp__foldbtn");
        // the label is sticky inside the scroll container, so it stays readable
        // when four columns force a horizontal scroll
        foldBtn.appendChild(el("span", "bank-cmp__foldtxt", showSame
          ? "hide the " + nSame + " row" + (nSame === 1 ? "" : "s") + " all " +
            picked.length + " agree on"
          : "⋯ " + nSame + " row" + (nSame === 1 ? "" : "s") + " all " +
            picked.length + " agree on — show " + (nSame === 1 ? "it" : "them")));
        foldBtn.type = "button";
        foldBtn.setAttribute("aria-expanded", String(showSame));
        foldBtn.addEventListener("click", function () {
          showSame = !showSame;
          render();
          var next = panelTable.querySelector(".bank-cmp__foldbtn");
          if (next) next.focus();
        });
        foldCell.appendChild(foldBtn);
        fold.appendChild(foldCell);
        tbodyC.appendChild(fold);
      }
      panelTable.appendChild(tbodyC);
      panelTable.dataset.showSame = (showSame || forced) ? "1" : "0";
      panelTable.dataset.cols = picked.length;

      if (panelNote) {
        var bits = [];
        if (forced) {
          bits.push("All " + picked.length +
                    " declare the same thing on every row compared.");
        } else {
          bits.push(nDiff + " of " + (nDiff + nSame) + " rows differ.");
          bits.push("A marked cell is the odd one out — a value fewer of the " +
                    "selected methods declare than the most common one; an even " +
                    "split is left unmarked.");
        }
        if (hasCost) {
          bits.push("Cost bars are linear in each quantity, longest = largest " +
                    "selected; the × is against softmax at the same shape.");
        }
        if (undeclared.length) {
          bits.push("Not shown, because not every selected method declares it: " +
                    undeclared.join(", ") + ".");
        }
        panelNote.textContent = bits.join(" ");
      }
    }

    function shapeText() {
      var s = data.shape || {};
      return Object.keys(s).map(function (k) { return k + " = " + s[k]; }).join(", ");
    }

    /* ---- the URL -------------------------------------------------------- */

    function pushUrl() {
      if (restoring) return;
      if (urlTimer) clearTimeout(urlTimer);
      urlTimer = setTimeout(writeUrl, 180);
    }

    function writeUrl() {
      urlTimer = null;
      // a stale timer from a page we have navigated away from must not rewrite the
      // URL of the page we are now on
      if (document.getElementById("bank") !== root) return;
      var params = new URLSearchParams();
      var q = (input.value || "").trim();
      if (q) params.set("q", q);
      var by = active();
      FIELDS.forEach(function (field) {
        if (by[field]) params.set(field, by[field].join(","));
      });
      if (sortField) params.set("sort", (sortDesc ? "-" : "") + sortField);
      if (picked.length) params.set("compare", picked.join(","));
      var qs = params.toString().replace(/%2C/g, ",");
      var url = location.pathname + (qs ? "?" + qs : "") + location.hash;
      if (url !== location.pathname + location.search + location.hash) {
        history.replaceState(history.state, "", url);
      }
    }

    function restore() {
      restoring = true;
      var params = new URLSearchParams(location.search);
      input.value = params.get("q") || "";
      FIELDS.forEach(function (field) {
        var want = (params.get(field) || "").split(",").filter(Boolean);
        chips.forEach(function (chip) {
          if (chip.dataset.filter !== field) return;
          chip.setAttribute("aria-pressed",
            String(want.indexOf(chip.dataset.value) !== -1));
        });
      });
      picked = (params.get("compare") || "").split(",").filter(function (n) {
        return Object.prototype.hasOwnProperty.call(boxes, n);
      }).slice(0, MAX_PICK);

      var sort = params.get("sort") || "";
      var desc = sort.charAt(0) === "-";
      var field = desc ? sort.slice(1) : sort;
      var known = headers.some(function (h) { return h.dataset.sort === field; });

      apply();
      syncPicks();
      if (known) applySort(field, desc);
      restoring = false;
      writeUrl();                                   // normalise what was pasted
    }

    restore();
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
