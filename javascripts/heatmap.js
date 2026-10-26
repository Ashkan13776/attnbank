/* The interactive attention heat map.
 *
 * Owned by one aspect of the site redesign; see lib/SITE_DESIGN.md.
 *
 * What this draws
 * ---------------
 * Every method in the bank satisfies invariant W, so every one of them can hand you the
 * measure it realises over context positions. docs/gen_heatmaps.py computes that measure
 * for each (method, configuration) over one real GPT-2 head, quantises it to eight bits
 * and ships it as one greyscale PNG *atlas* per method -- every frame of a configuration
 * ladder side by side in a single image. This file decodes an atlas once, and from then
 * on colour-maps frames onto a canvas.
 *
 * Why colour-map here instead of at build time
 * --------------------------------------------
 * Two reasons, and they are the same reason twice. The sequential ramp is oriented so
 * that no mass is the page ground and full mass is ink, which means it is a different
 * ramp in each theme; and the diverging ramp's zero must *be* the page ground, which is
 * also a different colour in each theme. Pre-coloured images would need two copies of
 * every map. One greyscale payload plus two ramps in the manifest serves both themes and
 * follows the theme toggle live.
 *
 * Nothing here invents a number. The only arithmetic applied to a cell is the inverse of
 * the generator's codec and a lookup into a ramp; no interpolation, no smoothing. Canvas
 * smoothing is explicitly disabled so an upscaled cell stays a square of one value.
 */
(function () {
  "use strict";

  var MANIFEST = null;          // parsed once per session, reused across navigations
  var MANIFEST_URL = null;
  var ATLAS = {};               // name -> {w, h, data} decoded pixel planes

  // ---------------------------------------------------------------- small helpers

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }

  /* Superscript digits, so a decade tick reads 10^-4 and not "1e-4". */
  function sup(n) {
    var map = { "-": "\u207b", 0: "\u2070", 1: "\u00b9", 2: "\u00b2", 3: "\u00b3",
                4: "\u2074", 5: "\u2075", 6: "\u2076", 7: "\u2077", 8: "\u2078",
                9: "\u2079" };
    return String(n).split("").map(function (ch) { return map[ch] || ch; }).join("");
  }

  function fmt(x, nd) {
    if (x === 0) return "0";
    var a = Math.abs(x);
    if (a < 1e-3 || a >= 1e5) return x.toExponential(1).replace("e", "×10^");
    return x.toFixed(nd === undefined ? 3 : nd);
  }

  /* Material stamps its palette on <body>, and on <html> for the server-rendered
   * first paint. Read whichever carries it; treat anything but "slate" as light. */
  function theme() {
    var a = document.body.getAttribute("data-md-color-scheme") ||
            document.documentElement.getAttribute("data-md-color-scheme");
    return a === "slate" ? "dark" : "light";
  }

  function reducedMotion() {
    return window.matchMedia &&
           window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  // ---------------------------------------------------------------- colour ramps
  //
  // The manifest ships the *stops*, not the resampled ramp, and this is the same
  // piecewise-linear resample gen_heatmaps.py runs, so the canvas and the static
  // no-JavaScript PNGs are the same colours computed from the same numbers.

  function resample(stops, n) {
    var out = new Uint8Array(n * 3), i, t, lo, f, a, b, c;
    for (i = 0; i < n; i++) {
      t = (i / (n - 1)) * (stops.length - 1);
      lo = Math.min(Math.floor(t), stops.length - 2);
      f = t - lo;
      a = stops[lo];
      b = stops[lo + 1];
      for (c = 0; c < 3; c++) out[i * 3 + c] = Math.round(a[c] + (b[c] - a[c]) * f);
    }
    return out;
  }

  /* The sequential ramp's first two stops cover BLEND of the scale and the rest share
   * what is left, so the two segments are resampled separately -- exactly as the
   * generator does it. BLEND is 0.08; the split index is round(0.08 * 255). */
  function sequentialRamp(stops) {
    var split = Math.round(0.08 * 255) + 1;
    var head = resample(stops.slice(0, 2), split);
    var tail = resample(stops.slice(1), 256 - split + 1);
    var out = new Uint8Array(256 * 3);
    out.set(head, 0);
    out.set(tail.subarray(3), split * 3);
    return out;
  }

  /* 256 entries with index 128 exactly the page ground, so zero on the signed map is
   * the page and not a colour. */
  function divergingRamp(neg, pos) {
    var n = resample(neg, 129), p = resample(pos, 128);
    var out = new Uint8Array(256 * 3), i, c;
    for (i = 0; i <= 128; i++)
      for (c = 0; c < 3; c++) out[(128 - i) * 3 + c] = n[i * 3 + c];
    for (i = 0; i < 128; i++)
      for (c = 0; c < 3; c++) out[(128 + i) * 3 + c] = p[i * 3 + c];
    return out;
  }

  function ramps() {
    var r = MANIFEST.ramps[theme()];
    return { seq: sequentialRamp(r.seq), div: divergingRamp(r.neg, r.pos) };
  }

  // ---------------------------------------------------------------- codec inverse

  /* The generator stores u = 255 * (log10(pi) + DECADES) / DECADES, with u = 0 reserved
   * for a structural zero. This is its inverse, and the only place a stored byte becomes
   * a number again. */
  function dequantPi(u, decades) {
    if (u === 0) return 0;
    return Math.pow(10, (u / 255) * decades - decades);
  }

  /* Display position in [0, 1] for a stored byte under the chosen display scale. The
   * stored byte is already the logarithmic position, so "log" is the identity and costs
   * nothing; the other two dequantise first. */
  function display(u, decades, scale) {
    if (u === 0) return 0;
    if (scale === "log") return u / 255;
    var p = dequantPi(u, decades);
    return scale === "sqrt" ? Math.sqrt(p) : p;
  }

  // ---------------------------------------------------------------- asset loading

  function loadManifest(url) {
    if (MANIFEST && MANIFEST_URL === url) return Promise.resolve(MANIFEST);
    return fetch(url, { credentials: "same-origin" })
      .then(function (r) {
        if (!r.ok) throw new Error("manifest " + r.status);
        return r.json();
      })
      .then(function (j) { MANIFEST = j; MANIFEST_URL = url; return j; });
  }

  /* Decode one atlas to a plane of bytes. The PNG is greyscale, so only the red channel
   * is read; the alpha channel is opaque throughout and nothing is premultiplied, which
   * is what makes the round trip through a canvas exact. */
  function loadAtlas(src, key) {
    if (ATLAS[key]) return Promise.resolve(ATLAS[key]);
    return new Promise(function (resolve, reject) {
      var img = new Image();
      img.decoding = "async";
      img.onload = function () {
        var c = document.createElement("canvas");
        c.width = img.naturalWidth;
        c.height = img.naturalHeight;
        var g = c.getContext("2d", { willReadFrequently: true });
        g.imageSmoothingEnabled = false;
        g.drawImage(img, 0, 0);
        var px = g.getImageData(0, 0, c.width, c.height).data;
        var plane = new Uint8Array(c.width * c.height), i;
        for (i = 0; i < plane.length; i++) plane[i] = px[i * 4];
        ATLAS[key] = { w: c.width, h: c.height, data: plane };
        resolve(ATLAS[key]);
      };
      img.onerror = function () { reject(new Error("image " + src)); };
      img.src = src;
    });
  }

  function b64bytes(s) {
    var bin = atob(s), out = new Uint8Array(bin.length), i;
    for (i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  // ---------------------------------------------------------------- the widget

  function init() {
    var root = document.getElementById("ab-heatmap");
    if (!root || root.dataset.ready === "1") return;
    root.dataset.ready = "1";
    // n, the ramps and the ladders all come from the manifest, so nothing is built
    // until it is in. If it never arrives the static triptych already in the page
    // stays exactly where it is, which is why it is in the page and not injected.
    loadManifest(root.dataset.manifest).then(function () {
      build(root);
    }).catch(function (err) {
      root.dataset.ready = "";
      if (window.console) console.warn("heatmap: " + err.message);
    });
  }

  function build(root) {
    var app = root.querySelector(".ab-hm-app");
    var statics = root.querySelector(".ab-hm-static");
    var base = root.dataset.src;
    var alive = true;                       // cleared when this DOM is navigated away

    var select = root.querySelector("#ab-hm-method");
    var range = root.querySelector("#ab-hm-config");
    var knobLabel = root.querySelector("#ab-hm-knoblabel");
    var valueOut = root.querySelector("#ab-hm-value");
    var summary = root.querySelector("#ab-hm-summary");
    var maps = root.querySelector("#ab-hm-maps");
    var profileBox = root.querySelector("#ab-hm-profile");
    var statsBox = root.querySelector("#ab-hm-stats");
    var chips = [].slice.call(root.querySelectorAll(".ab-hm-cat"));

    // `locked` is PySvelte's soft/hard lock: hover previews a row, a click fixes it so
    // the pointer can leave the map without taking the numbers with it, and clicking
    // the fixed row again releases it. Without it the readout is pointer-only, which
    // is both a WCAG 1.4.13 problem and simply annoying to use.
    var state = { method: null, frame: 0, row: -1, locked: false, scale: "log" };
    var panels = {};                         // role -> {canvas, ctx, caption, legend}
    var current = null;                      // {spec, frame, hat, ref, diff, tvRows}

    // -- the scale toggle is built here, not in the page, because with scripting off
    //    it would be a control that does nothing, and this site does not ship those.
    var scaleBox = el("div", "ab-hm-scaleswitch");
    scaleBox.setAttribute("role", "group");
    scaleBox.setAttribute("aria-label", "Display scale");
    scaleBox.appendChild(el("span", "ab-hm-swlabel", "Scale"));
    var scaleRow = el("div", "ab-hm-scalerow");
    scaleBox.appendChild(scaleRow);
    [["log", "log"], ["sqrt", "√"], ["linear", "linear"]].forEach(function (p) {
      var b = el("button", "ab-hm-sw", p[1]);
      b.type = "button";
      b.dataset.scale = p[0];
      b.setAttribute("aria-pressed", String(p[0] === state.scale));
      b.addEventListener("click", function () {
        state.scale = p[0];
        [].forEach.call(scaleBox.querySelectorAll(".ab-hm-sw"), function (o) {
          o.setAttribute("aria-pressed", String(o.dataset.scale === state.scale));
        });
        paint();
      });
      scaleRow.appendChild(b);
    });
    root.querySelector(".ab-hm-controls").appendChild(scaleBox);

    var live = el("p", "ab-hm-live");
    live.setAttribute("aria-live", "polite");
    live.setAttribute("role", "status");
    root.appendChild(live);

    // ---- panel construction ------------------------------------------------
    function makePanel(role, title, note) {
      var fig = el("figure", "ab-hm-panel");
      fig.dataset.role = role;
      var head = el("figcaption", "ab-hm-cap");
      head.appendChild(el("b", "ab-hm-captitle", title));
      head.appendChild(el("span", "ab-hm-capnote", note || ""));
      fig.appendChild(head);

      var row = el("div", "ab-hm-plotrow");
      var wrap = el("div", "ab-hm-plot");
      var canvas = el("canvas", "ab-hm-canvas");
      canvas.width = MANIFEST.n;
      canvas.height = MANIFEST.n;
      canvas.setAttribute("role", "img");
      wrap.appendChild(canvas);
      var line = el("div", "ab-hm-rowline");
      wrap.appendChild(line);
      row.appendChild(wrap);
      fig.appendChild(row);

      var legend = el("div", "ab-hm-legend");
      fig.appendChild(legend);

      var ctx = canvas.getContext("2d");
      ctx.imageSmoothingEnabled = false;
      panels[role] = { fig: fig, canvas: canvas, ctx: ctx, line: line,
                       note: head.querySelector(".ab-hm-capnote"), legend: legend };
      return fig;
    }

    var grid = el("div", "ab-hm-grid");
    grid.appendChild(makePanel("hat", "π̂", ""));
    grid.appendChild(makePanel("ref", "π", "dense softmax"));

    grid.appendChild(makePanel("diff", "π̂ − π", ""));
    var tvStrip = el("canvas", "ab-hm-tvstrip");
    tvStrip.width = 1;
    tvStrip.height = MANIFEST.n;
    tvStrip.setAttribute("role", "img");
    tvStrip.setAttribute("aria-label",
      "Total variation against dense softmax for each query row, top to bottom");
    tvStrip.title = "total variation per query row";
    panels.diff.fig.querySelector(".ab-hm-plotrow").appendChild(tvStrip);
    maps.appendChild(grid);

    // The plot area is one focusable control per panel: a reader can tab to a map and
    // walk query rows with the arrow keys, which is the same affordance the pointer
    // gets and the reason the hover readout is not pointer-only.
    Object.keys(panels).forEach(function (role) {
      var p = panels[role];
      var plot = p.fig.querySelector(".ab-hm-plot");
      plot.tabIndex = 0;
      plot.setAttribute("aria-describedby", "ab-hm-rowread");

      plot.addEventListener("mousemove", function (ev) {
        if (state.locked) return;
        var r = p.canvas.getBoundingClientRect();
        setRow(Math.floor((ev.clientY - r.top) / r.height * MANIFEST.n));
      });
      plot.addEventListener("mouseleave", function () {
        if (!state.locked) setRow(-1);
      });
      plot.addEventListener("click", function (ev) {
        var r = p.canvas.getBoundingClientRect();
        var i = Math.floor((ev.clientY - r.top) / r.height * MANIFEST.n);
        if (state.locked && i === state.row) { state.locked = false; }
        else { state.locked = true; setRow(i, true); }
        markLock();
      });
      plot.addEventListener("focus", function () {
        if (state.row < 0) setRow(Math.floor(MANIFEST.n / 2));
      });
      plot.addEventListener("keydown", function (ev) {
        var step = ev.shiftKey ? 10 : 1, r = state.row < 0 ? 0 : state.row;
        if (ev.key === "ArrowDown") { setRow(Math.min(MANIFEST.n - 1, r + step)); }
        else if (ev.key === "ArrowUp") { setRow(Math.max(0, r - step)); }
        else if (ev.key === "Home") { setRow(0); }
        else if (ev.key === "End") { setRow(MANIFEST.n - 1); }
        else if (ev.key === "Escape") {
          state.locked = false; markLock(); setRow(-1); return;
        } else if (ev.key === "Enter" || ev.key === " ") {
          state.locked = !state.locked; markLock(); ev.preventDefault(); return;
        }
        else { return; }
        ev.preventDefault();
      });
    });

    var rowRead = el("div", "ab-hm-rowread");
    rowRead.id = "ab-hm-rowread";
    var profCanvas = el("canvas", "ab-hm-profile-canvas");
    var topBox = el("div", "ab-hm-top");
    profileBox.appendChild(rowRead);
    profileBox.appendChild(profCanvas);
    profileBox.appendChild(topBox);

    // ---- data plumbing -----------------------------------------------------
    function spec(name) {
      var i;
      for (i = 0; i < MANIFEST.methods.length; i++)
        if (MANIFEST.methods[i].name === name) return MANIFEST.methods[i];
      return null;
    }

    function selectMethod(name, frame) {
      var s = spec(name);
      if (!s) return;
      state.method = name;
      state.frame = Math.max(0, Math.min(s.frames.length - 1,
        frame === undefined ? 0 : frame));
      select.value = name;
      root.setAttribute("data-cat", s.category);

      range.max = String(s.frames.length - 1);
      range.value = String(state.frame);
      range.disabled = s.frames.length < 2;
      knobLabel.textContent = s.knob ? s.knob : "no configuration knob";
      summary.textContent = s.summary;

      var want = [
        loadAtlas(base + "m-" + name + ".png", "m-" + name),
        loadAtlas(base + "ref-" + s.mask + ".png", "ref-" + s.mask)
      ];
      if (!s.zero_delta) want.push(loadAtlas(base + "d-" + name + ".png", "d-" + name));
      maps.setAttribute("aria-busy", "true");
      Promise.all(want).then(function (got) {
        if (!alive || state.method !== name) return;
        current = { spec: s, hat: got[0], ref: got[1], diff: got[2] || null };
        maps.removeAttribute("aria-busy");
        paint();
      }).catch(function (err) {
        maps.removeAttribute("aria-busy");
        fail(err);
      });
      writeUrl();
    }

    function fail(err) {
      // A failed asset must not leave an empty frame where a figure was: fall back to
      // the static triptych, which is real and already in the page.
      app.hidden = true;
      statics.hidden = false;
      live.textContent = "The interactive maps could not be loaded (" +
        (err && err.message ? err.message : "unknown error") +
        "); the static figure above is the same data.";
    }

    // ---- painting ----------------------------------------------------------
    function blitFrame(panel, atlas, frame, ramp, kind, decades, scale) {
      var n = MANIFEST.n;
      var img = panel.ctx.createImageData(n, n);
      var out = img.data, x, y, u, t, idx, off = frame * n;
      for (y = 0; y < n; y++) {
        for (x = 0; x < n; x++) {
          u = atlas.data[y * atlas.w + off + x];
          if (kind === "seq") {
            t = Math.round(display(u, decades, scale) * 255);
            if (t > 255) t = 255;
          } else {
            t = u;                                  // already a symmetric linear code
          }
          idx = (y * n + x) * 4;
          out[idx] = ramp[t * 3];
          out[idx + 1] = ramp[t * 3 + 1];
          out[idx + 2] = ramp[t * 3 + 2];
          out[idx + 3] = 255;
        }
      }
      panel.ctx.putImageData(img, 0, 0);
    }

    function blitFlat(panel, ramp) {           // a difference that is numerically zero
      var n = MANIFEST.n;
      var img = panel.ctx.createImageData(n, n);
      var out = img.data, i;
      for (i = 0; i < n * n; i++) {
        out[i * 4] = ramp[128 * 3];
        out[i * 4 + 1] = ramp[128 * 3 + 1];
        out[i * 4 + 2] = ramp[128 * 3 + 2];
        out[i * 4 + 3] = 255;
      }
      panel.ctx.putImageData(img, 0, 0);
    }

    function seqLegend(box, ramp, scale, decades) {
      box.textContent = "";
      var bar = el("div", "ab-hm-bar");
      var stops = [], i, t;
      for (i = 0; i <= 32; i++) {
        t = Math.round((i / 32) * 255);
        stops.push("rgb(" + ramp[t * 3] + "," + ramp[t * 3 + 1] + "," +
                   ramp[t * 3 + 2] + ") " + ((i / 32) * 100).toFixed(1) + "%");
      }
      bar.style.background = "linear-gradient(to right," + stops.join(",") + ")";
      box.appendChild(bar);

      var ticks = el("div", "ab-hm-ticks"), vals, pos;
      if (scale === "log") {
        // every other decade: at the width a panel actually gets, five labels over
        // four decades overlap and the first two become one smear.
        vals = [];
        for (i = decades; i >= 0; i -= 2) vals.push(Math.pow(10, -i));
        pos = vals.map(function (v) { return (Math.log10(v) + decades) / decades; });
      } else if (scale === "sqrt") {
        vals = [0, 0.01, 0.1, 0.25, 0.5, 1];
        pos = vals.map(function (v) { return Math.sqrt(v); });
      } else {
        vals = [0, 0.25, 0.5, 0.75, 1];
        pos = vals.slice();
      }
      vals.forEach(function (v, j) {
        var t2 = el("span", "ab-hm-tick",
          v === 0 ? "0" : (v >= 0.01 ? String(v) : "10" + sup(Math.round(Math.log10(v)))));
        t2.style.left = (pos[j] * 100).toFixed(2) + "%";
        ticks.appendChild(t2);
      });
      box.appendChild(ticks);
      box.appendChild(el("span", "ab-hm-unit",
        "mass per cell, " + (scale === "sqrt" ? "square-root" : scale) + " scale"));
    }

    /* Each arm is logarithmic over MANIFEST.diff_decades below dmax, so the ticks are
     * dmax, dmax/10, ... on each side with a true zero in the middle. Labelled with the
     * real differences, never with the code. */
    function divLegend(box, ramp, dmax) {
      box.textContent = "";
      var dd = MANIFEST.diff_decades;
      var bar = el("div", "ab-hm-bar");
      var stops = [], i, t;
      for (i = 0; i <= 48; i++) {
        t = Math.round((i / 48) * 255);
        stops.push("rgb(" + ramp[t * 3] + "," + ramp[t * 3 + 1] + "," +
                   ramp[t * 3 + 2] + ") " + ((i / 48) * 100).toFixed(1) + "%");
      }
      bar.style.background = "linear-gradient(to right," + stops.join(",") + ")";
      box.appendChild(bar);

      var ticks = el("div", "ab-hm-ticks");
      [[-dmax, 0], [0, 50], [dmax, 100]].forEach(function (p) {
        var t2 = el("span", "ab-hm-tick",
          p[0] === 0 ? "0" : (p[0] > 0 ? "+" : "−") + fmt(Math.abs(p[0]), 2));
        t2.style.left = p[1] + "%";
        ticks.appendChild(t2);
      });
      box.appendChild(ticks);
      box.appendChild(el("span", "ab-hm-unit",
        "removed ← 0 → added, log over " + dd + " decades each side"));
    }

    function paintTvStrip(tv, ramp) {
      var g = tvStrip.getContext("2d");
      var img = g.createImageData(1, MANIFEST.n), i, t;
      for (i = 0; i < MANIFEST.n; i++) {
        t = 128 + Math.round((tv[i] / 255) * 127);
        img.data[i * 4] = ramp[t * 3];
        img.data[i * 4 + 1] = ramp[t * 3 + 1];
        img.data[i * 4 + 2] = ramp[t * 3 + 2];
        img.data[i * 4 + 3] = 255;
      }
      g.putImageData(img, 0, 0);
    }

    function paint() {
      if (!current) return;
      var s = current.spec, f = s.frames[state.frame], R = ramps(), n = MANIFEST.n;
      var dec = MANIFEST.decades;

      blitFrame(panels.hat, current.hat, state.frame, R.seq, "seq", dec, state.scale);
      blitFrame(panels.ref, current.ref, 0, R.seq, "seq", dec, state.scale);
      if (current.diff) {
        blitFrame(panels.diff, current.diff, state.frame, R.div, "div", dec, state.scale);
      } else {
        blitFlat(panels.diff, R.div);
      }

      var cfg = s.knob ? s.knob + " = " + f.label : "no knob";
      panels.hat.note.textContent = s.name + ", " + cfg;
      panels.ref.note.textContent = s.mask === "causal" ? "causal" : "no mask";
      panels.diff.note.textContent = s.zero_delta
        ? "zero to " + fmt(s.dmax) + ", the floating-point floor"
        : "±" + fmt(s.dmax, 3) + " full scale";

      panels.hat.canvas.setAttribute("aria-label",
        "Attention map realised by " + s.name + " at " + cfg + ", " + n + " by " + n +
        " cells, queries down and keys across. Mean total variation from dense " +
        "softmax " + fmt(f.tv) + ".");
      panels.ref.canvas.setAttribute("aria-label",
        "Dense softmax attention on the same head, " + n + " by " + n + " cells.");
      panels.diff.canvas.setAttribute("aria-label", s.zero_delta
        ? "The signed difference is zero everywhere to within " + fmt(s.dmax) + "."
        : "Signed difference, " + s.name + " minus dense softmax, full scale plus or " +
          "minus " + fmt(s.dmax, 3) + ".");

      seqLegend(panels.hat.legend, R.seq, state.scale, dec);
      seqLegend(panels.ref.legend, R.seq, state.scale, dec);
      divLegend(panels.diff.legend, R.div, s.dmax);

      paintTvStrip(b64bytes(f.tv_rows), R.div);

      valueOut.textContent = s.knob ? f.label : "—";
      paintStats(s, f);
      paintProfile();
    }

    function paintStats(s, f) {
      statsBox.textContent = "";
      var head = el("thead"), hr = el("tr");
      var body = el("tbody"), br = el("tr");
      var rows = [
        ["TV(π, π̂)", fmt(f.tv), "mean over queries of ½‖π − π̂‖₁"],
        ["‖y − ŷ‖ / ‖y‖", fmt(f.rel), "relative output error"],
        ["distributional", fmt(f.dist), "‖(π − π̂)v‖, the term this map draws"],
        ["value", fmt(f.value), "‖π̂(v − v̂)‖, the term it cannot"],
        ["eff. support", fmt(f.supp, 1), "exp(entropy) of π̂, per query"],
        ["cells drawn", fmt(f.nnz, 1), "per query, above the 1e-" + MANIFEST.decades + " floor"]
      ];
      rows.forEach(function (r) {
        var th = el("th", null, r[0]);
        th.scope = "col";
        th.title = r[2];
        hr.appendChild(th);
        br.appendChild(el("td", null, r[1]));
      });
      head.appendChild(hr);
      body.appendChild(br);
      statsBox.appendChild(head);
      statsBox.appendChild(body);
      statsBox.setAttribute("aria-label",
        "Measurements for " + s.name + " at this configuration");
    }

    // ---- the row profile ---------------------------------------------------
    function markLock() {
      Object.keys(panels).forEach(function (role) {
        panels[role].line.classList.toggle("is-locked", state.locked);
      });
    }

    function setRow(i, force) {
      if (i === state.row && !force) return;
      state.row = (i >= 0 && i < MANIFEST.n) ? i : -1;
      var pct = state.row < 0 ? 0 : ((state.row + 0.5) / MANIFEST.n) * 100;
      Object.keys(panels).forEach(function (role) {
        panels[role].line.style.display = state.row < 0 ? "none" : "block";
        panels[role].line.style.top = pct + "%";
      });
      paintProfile();
    }

    function paintProfile() {
      if (!current) return;
      var n = MANIFEST.n, dec = MANIFEST.decades, s = current.spec;
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      var cssW = profCanvas.clientWidth || 600, cssH = 96;
      profCanvas.width = Math.round(cssW * dpr);
      profCanvas.height = Math.round(cssH * dpr);
      profCanvas.style.height = cssH + "px";
      var g = profCanvas.getContext("2d");
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, cssW, cssH);

      var css = getComputedStyle(root);
      var inkMuted = css.getPropertyValue("--ab-ink-muted").trim() || "#5d6673";
      var ruleSoft = css.getPropertyValue("--ab-rule-soft").trim() || "#e9edf2";
      var hue = css.getPropertyValue("--hl").trim() || inkMuted;

      if (state.row < 0) {
        topBox.textContent = "";
        rowRead.textContent =
          "Point at a map, or focus one and use the arrow keys, to read a query row. " +
          "Click, or press Enter, to lock it.";
        g.strokeStyle = ruleSoft;
        g.beginPath();
        g.moveTo(0, cssH - 0.5);
        g.lineTo(cssW, cssH - 0.5);
        g.stroke();
        return;
      }

      var row = state.row, off = state.frame * n, x, u, h, w = cssW / n;
      var f = s.frames[state.frame];
      var tv = b64bytes(f.tv_rows)[row] / 255;

      g.strokeStyle = ruleSoft;
      g.lineWidth = 1;
      [0.25, 0.5, 0.75].forEach(function (t) {
        g.beginPath();
        g.moveTo(0, Math.round(cssH * (1 - t)) - 0.5);
        g.lineTo(cssW, Math.round(cssH * (1 - t)) - 0.5);
        g.stroke();
      });

      // dense softmax first, as a filled silhouette; the method's measure over it.
      g.fillStyle = ruleSoft;
      for (x = 0; x < n; x++) {
        u = current.ref.data[row * current.ref.w + x];
        h = display(u, dec, state.scale) * cssH;
        if (h > 0) g.fillRect(x * w, cssH - h, Math.max(w - 0.4, 0.6), h);
      }
      g.fillStyle = hue;
      for (x = 0; x < n; x++) {
        u = current.hat.data[row * current.hat.w + off + x];
        h = display(u, dec, state.scale) * cssH;
        if (h > 0) g.fillRect(x * w + w * 0.25, cssH - h, Math.max(w * 0.5, 0.6), h);
      }
      g.strokeStyle = inkMuted;
      g.beginPath();
      g.moveTo(0, cssH - 0.5);
      g.lineTo(cssW, cssH - 0.5);
      g.stroke();

      var cells = 0;
      for (x = 0; x < n; x++)
        if (current.hat.data[row * current.hat.w + off + x] > 0) cells++;
      rowRead.textContent = "query " + row + " — TV " + tv.toFixed(2) + " — " +
        cells + " of " + (s.mask === "causal" ? row + 1 : n) +
        " visible keys carry mass above the floor" +
        (state.locked ? " — locked, click again or press Escape to release"
                      : " — click to lock");
      live.textContent = rowRead.textContent + ". " + topText(row, off);
      paintTop(row, off);
    }

    /* The numbers behind the picture.
     *
     * A dense matrix is exactly where matching a colour back to a legend fails --
     * simultaneous contrast makes a mid-tone cell read lighter or darker depending on
     * its neighbours -- so the top of each row is also printed as text. It is the one
     * part of this figure that works with no colour vision at all, and it is what a
     * screen reader is given.
     */
    function topRanked(plane, w, row, off, n, count) {
      var items = [], x;
      for (x = 0; x < n; x++) {
        var u = plane[row * w + off + x];
        if (u > 0) items.push([x, dequantPi(u, MANIFEST.decades)]);
      }
      items.sort(function (a, b) { return b[1] - a[1]; });
      return items.slice(0, count);
    }

    function topText(row, off) {
      var n = MANIFEST.n;
      var a = topRanked(current.hat.data, current.hat.w, row, off, n, 5);
      var b = topRanked(current.ref.data, current.ref.w, row, 0, n, 5);
      var say = function (xs) {
        return xs.map(function (p) {
          return "key " + p[0] + " " + p[1].toFixed(3);
        }).join(", ") || "nothing above the floor";
      };
      return "Largest cells, method: " + say(a) + ". Dense softmax: " + say(b) + ".";
    }

    function paintTop(row, off) {
      var n = MANIFEST.n;
      var a = topRanked(current.hat.data, current.hat.w, row, off, n, 5);
      var b = topRanked(current.ref.data, current.ref.w, row, 0, n, 5);
      topBox.textContent = "";
      var mk = function (title, xs) {
        var line = el("div");
        line.appendChild(el("span", null, title + " "));
        if (!xs.length) { line.appendChild(el("b", null, "nothing above the floor")); }
        xs.forEach(function (pp, j) {
          if (j) line.appendChild(el("span", null, "   "));
          line.appendChild(el("b", null, "k" + pp[0] + " " + pp[1].toFixed(3)));
        });
        return line;
      };
      topBox.appendChild(mk("π̂ ", a));
      topBox.appendChild(mk("π  ", b));
    }

    // ---- controls ----------------------------------------------------------
    select.addEventListener("change", function () { selectMethod(select.value, 0); });
    range.addEventListener("input", function () {
      state.frame = parseInt(range.value, 10) || 0;
      paint();
      writeUrl();
    });

    chips.forEach(function (chip) {
      chip.addEventListener("click", function () {
        var cat = chip.dataset.value;
        var on = chip.getAttribute("aria-pressed") !== "true";
        chips.forEach(function (o) {
          o.setAttribute("aria-pressed", String(o === chip && on));
        });
        filterOptions(on ? cat : null);
      });
    });

    function filterOptions(cat) {
      var groups = [].slice.call(select.querySelectorAll("optgroup"));
      groups.forEach(function (gr) {
        gr.hidden = !!(cat && gr.label !== cat);
        gr.disabled = gr.hidden;
      });
      if (cat && spec(state.method) && spec(state.method).category !== cat) {
        var first = select.querySelector('optgroup[label="' + cat + '"] option');
        if (first) selectMethod(first.value, 0);
      }
    }

    // ---- URL state ---------------------------------------------------------
    function writeUrl() {
      if (!window.history || !history.replaceState) return;
      var p = new URLSearchParams(location.search);
      p.set("m", state.method);
      if (state.frame) p.set("c", String(state.frame)); else p.delete("c");
      history.replaceState(null, "", location.pathname + "?" + p.toString());
    }

    // ---- theme follows the toggle -----------------------------------------
    var obs = new MutationObserver(function () { if (alive && current) paint(); });
    obs.observe(document.body, { attributes: true,
                                 attributeFilter: ["data-md-color-scheme"] });

    var onResize = null;
    if (!reducedMotion()) {
      var t = null;
      onResize = function () {
        clearTimeout(t);
        t = setTimeout(function () { if (alive) paintProfile(); }, 120);
      };
    } else {
      onResize = function () { if (alive) paintProfile(); };
    }
    window.addEventListener("resize", onResize);

    // Instant navigation replaces the body; stop painting into the old one.
    function teardown() {
      alive = false;
      obs.disconnect();
      window.removeEventListener("resize", onResize);
    }
    if (typeof document$ !== "undefined" && document$.subscribe) {
      var sub = document$.subscribe(function (doc) {
        if (doc !== document || !document.body.contains(root)) {
          teardown();
          sub.unsubscribe();
        }
      });
    }

    // ---- go ----------------------------------------------------------------
    var q = new URLSearchParams(location.search);
    var want = q.get("m") && spec(q.get("m")) ? q.get("m") : MANIFEST.default.method;
    var startFrame = parseInt(q.get("c"), 10);
    if (isNaN(startFrame)) {
      startFrame = want === MANIFEST.default.method ? MANIFEST.default.frame : 0;
    }
    statics.hidden = true;
    app.hidden = false;
    selectMethod(want, startFrame);
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
