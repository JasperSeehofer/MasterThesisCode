/*
 * book.js — small, dependency-free widget library for the discovery book.
 *
 * FROZEN for chapter agents: request new capabilities via
 * book/design/WIDGET_REQUESTS.md; only the integrator edits this file.
 *
 * Provides:
 *   Book.theme              — light/dark toggle, respects prefers-color-scheme
 *   Book.renderMath(root)   — KaTeX auto-render over a subtree (or document)
 *   Book.loadJSON(url)      — fetch + cache a data/*.json file
 *   Book.gridSlider(opts)   — a slider bound to a precomputed data grid
 *   Book.predictReveal(el)  — "predict, then reveal" row (localStorage-persisted)
 *   Book.buildNav()         — top nav from window.BOOK_CHAPTERS (js/manifest.js)
 *   Book.themedPlot(...)    — Plotly plot that re-layouts on theme change
 *   Book.isDark()           — current effective theme
 *   Book.logsumexp / combineLogRows / normalizePosterior / trapz / argmaxIdx
 *                           — log-space posterior helpers (native representation)
 *   Book.biasRail(spec)     — the per-chapter Bias Ledger Rail (BW1, minimal)
 *
 * No build step: this file is loaded directly via <script src="js/book.js">
 * after js/manifest.js, vendor/plotly/plotly.min.js and vendor/katex/katex.min.js
 * + vendor/katex/auto-render.min.js.
 */
(function (global) {
  "use strict";

  const Book = {};

  // ------------------------------------------------------------------
  // Theme toggle
  // ------------------------------------------------------------------
  Book.theme = {
    STORAGE_KEY: "book-theme",

    current() {
      return document.documentElement.getAttribute("data-theme");
    },

    apply(mode) {
      if (mode === "light" || mode === "dark") {
        document.documentElement.setAttribute("data-theme", mode);
      } else {
        document.documentElement.removeAttribute("data-theme");
      }
    },

    toggle() {
      const prefersDark = global.matchMedia &&
        global.matchMedia("(prefers-color-scheme: dark)").matches;
      const cur = Book.theme.current() || (prefersDark ? "dark" : "light");
      const next = cur === "dark" ? "light" : "dark";
      Book.theme.apply(next);
      try {
        localStorage.setItem(Book.theme.STORAGE_KEY, next);
      } catch (e) {
        /* localStorage unavailable (e.g. file:// in some browsers) — ignore */
      }
      Book.theme._updateButton();
    },

    _updateButton() {
      const btn = document.querySelector("[data-theme-toggle]");
      if (!btn) return;
      const prefersDark = global.matchMedia &&
        global.matchMedia("(prefers-color-scheme: dark)").matches;
      const cur = Book.theme.current() || (prefersDark ? "dark" : "light");
      btn.textContent = cur === "dark" ? "☀ Light" : "☽ Dark";
    },

    init() {
      try {
        const saved = localStorage.getItem(Book.theme.STORAGE_KEY);
        if (saved) Book.theme.apply(saved);
      } catch (e) {
        /* ignore */
      }
      const btn = document.querySelector("[data-theme-toggle]");
      if (btn) btn.addEventListener("click", Book.theme.toggle);
      Book.theme._updateButton();
    },
  };

  // ------------------------------------------------------------------
  // Math rendering (KaTeX auto-render over $...$ and $$...$$ / \[...\])
  // ------------------------------------------------------------------
  Book.renderMath = function (root) {
    const target = root || document.body;
    if (!global.renderMathInElement) {
      console.warn("book.js: KaTeX auto-render not loaded; math left as raw text.");
      return;
    }
    global.renderMathInElement(target, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
    });
  };

  // ------------------------------------------------------------------
  // Tiny JSON cache (data/*.json is small — whole-file fetch is fine)
  // ------------------------------------------------------------------
  const _jsonCache = new Map();
  Book.loadJSON = function (url) {
    if (_jsonCache.has(url)) return _jsonCache.get(url);
    const p = fetch(url).then((r) => {
      if (!r.ok) throw new Error(`book.js: failed to load ${url} (${r.status})`);
      return r.json();
    });
    _jsonCache.set(url, p);
    return p;
  };

  // ------------------------------------------------------------------
  // Linear interpolation helper for closed-form JS math widgets
  // ------------------------------------------------------------------
  Book.lerp = (a, b, t) => a + (b - a) * t;

  /** Nearest-grid-point lookup: given a sorted array of numeric keys and a
   * target value, return the closest key (as a string, matching how the
   * generators key their `posteriors` objects). Used by gridSlider so a
   * continuous-looking slider can drive a discrete precomputed grid. */
  Book.nearestKey = function (sortedNumericKeys, target) {
    let best = sortedNumericKeys[0];
    let bestDist = Math.abs(best - target);
    for (const k of sortedNumericKeys) {
      const d = Math.abs(k - target);
      if (d < bestDist) {
        best = k;
        bestDist = d;
      }
    }
    return best;
  };

  // ------------------------------------------------------------------
  // Slider bound to a precomputed grid, redrawing one Plotly trace
  // ------------------------------------------------------------------
  /**
   * opts:
   *   sliderEl      - <input type="range"> element (its min/max/step must
   *                    already be set by the caller to index into `steps`)
   *   readoutEl     - element whose textContent is updated per step
   *   steps         - array of step values (e.g. subset sizes N)
   *   formatReadout - (stepValue) => string
   *   onStep        - (stepValue, stepIndex) => void  -- draw/update logic
   */
  Book.gridSlider = function (opts) {
    const { sliderEl, readoutEl, steps, formatReadout, onStep } = opts;
    sliderEl.min = "0";
    sliderEl.max = String(steps.length - 1);
    sliderEl.step = "1";

    function render(idx) {
      const val = steps[idx];
      if (readoutEl) readoutEl.innerHTML = formatReadout ? formatReadout(val) : String(val);
      onStep(val, idx);
    }

    sliderEl.addEventListener("input", () => render(Number(sliderEl.value)));
    return { render, steps };
  };

  // ------------------------------------------------------------------
  // Predict-then-reveal button row
  // ------------------------------------------------------------------
  /**
   * Wires a container with:
   *   <button data-predict="A">...</button> (one or more)
   *   <div class="reveal" data-reveal-for="...">...</div>
   * Clicking a predict button marks it pressed, reveals the answer block,
   * and fires an optional onPredict(choice) callback (e.g. to also draw
   * the real curve on top of the reader's guess).
   *
   * The reveal block stays hidden until a prediction is recorded (the
   * pedagogy's "locked, not suggested" rule). If the container carries a
   * data-predict-id attribute, the reader's choice is persisted to
   * localStorage under "book-predict:<id>" and restored on reload, so a
   * chapter can re-surface earlier predictions (e.g. Ch 11 re-surfacing
   * the Ch 3 guess).
   */
  Book.predictReveal = function (container, onPredict) {
    const buttons = container.querySelectorAll("[data-predict]");
    const reveal = container.querySelector(".reveal");
    const pid = container.getAttribute("data-predict-id");
    const storageKey = pid ? "book-predict:" + pid : null;

    function select(btn, fire) {
      buttons.forEach((b) => b.setAttribute("aria-pressed", "false"));
      btn.setAttribute("aria-pressed", "true");
      if (reveal) reveal.classList.add("shown");
      if (storageKey) {
        try {
          localStorage.setItem(storageKey, btn.getAttribute("data-predict"));
        } catch (e) { /* ignore */ }
      }
      if (fire && onPredict) onPredict(btn.getAttribute("data-predict"));
    }

    buttons.forEach((btn) => {
      btn.setAttribute("aria-pressed", "false");
      btn.addEventListener("click", () => select(btn, true));
    });

    if (storageKey) {
      try {
        const saved = localStorage.getItem(storageKey);
        if (saved !== null) {
          const btn = container.querySelector(`[data-predict="${saved}"]`);
          if (btn) select(btn, false);
        }
      } catch (e) { /* ignore */ }
    }
  };

  /** Read back a persisted prediction made anywhere else in the book. */
  Book.getPrediction = function (predictId) {
    try {
      return localStorage.getItem("book-predict:" + predictId);
    } catch (e) {
      return null;
    }
  };

  // ------------------------------------------------------------------
  // Log-space posterior helpers (the book's native representation)
  // ------------------------------------------------------------------
  /** Stable log(sum(exp(a))) over a numeric array. */
  Book.logsumexp = function (arr) {
    let m = -Infinity;
    for (const v of arr) if (v > m) m = v;
    if (!isFinite(m)) return -Infinity;
    let s = 0;
    for (const v of arr) s += Math.exp(v - m);
    return m + Math.log(s);
  };

  /**
   * Combine per-event log-likelihood rows (array of arrays, each length
   * n_h) into an UN-normalized posterior on the h-grid: Sigma_i log L_i,
   * max-subtracted, exponentiated. Mirrors combine_log_space() in
   * master_thesis_code/bayesian_inference/posterior_combination.py —
   * cite it, do not re-derive.
   */
  Book.combineLogRows = function (rows) {
    if (!rows.length) return [];
    const nH = rows[0].length;
    const sum = new Array(nH).fill(0);
    for (const row of rows) for (let j = 0; j < nH; j++) sum[j] += row[j];
    let m = -Infinity;
    for (const v of sum) if (v > m) m = v;
    return sum.map((v) => Math.exp(v - m));
  };

  /** Trapezoid integral of y over (possibly NON-uniform) grid x.
   *  Safe on the book's seamed h-grid (0.01 / 0.005 / 0.01 spacing). */
  Book.trapz = function (y, x) {
    let s = 0;
    for (let i = 1; i < x.length; i++) s += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]);
    return s;
  };

  /** Normalize a density on grid x to unit trapezoid integral (in place-safe copy). */
  Book.normalizePosterior = function (y, x) {
    const n = Book.trapz(y, x);
    return n > 0 ? y.map((v) => v / n) : y.slice();
  };

  /** Index of the maximum of an array (the grid MAP). */
  Book.argmaxIdx = function (y) {
    let best = 0;
    for (let i = 1; i < y.length; i++) if (y[i] > y[best]) best = i;
    return best;
  };

  // ------------------------------------------------------------------
  // Theme-reactive Plotly wrapper
  // ------------------------------------------------------------------
  /** Current effective theme, honouring the data-theme override. */
  Book.isDark = function () {
    const attr = document.documentElement.getAttribute("data-theme");
    if (attr) return attr === "dark";
    return !!(global.matchMedia && global.matchMedia("(prefers-color-scheme: dark)").matches);
  };

  /**
   * Plotly.newPlot with automatic re-layout when the theme toggles.
   *   divId    - target element id
   *   traces   - initial trace array
   *   layoutFn - () => layout object (called again on every theme flip; use
   *              Book.isDark() inside it for colors)
   *   config   - optional Plotly config (default: no modebar, responsive)
   * Returns { update(traces) } which re-renders traces with a fresh layout.
   */
  Book.themedPlot = function (divId, traces, layoutFn, config) {
    const cfg = config || { displayModeBar: false, responsive: true };
    Plotly.newPlot(divId, traces, layoutFn(), cfg);
    const observer = new MutationObserver(() => {
      Plotly.relayout(divId, layoutFn());
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    return {
      update(newTraces) {
        Plotly.react(divId, newTraces, layoutFn(), cfg);
      },
    };
  };

  // ------------------------------------------------------------------
  // Bias Ledger Rail (BW1, minimal implementation)
  // ------------------------------------------------------------------
  /**
   * Renders the persistent per-chapter bias rail into #bias-rail (a fixed
   * side element; falls back to prepending into main.book-content on
   * narrow screens via CSS).
   *
   * spec = {
   *   entries: [{ label: "no D(h)", bias: -0.178, note: "ledger #9",
   *               active: true|false }, ...],
   *   title: "Estimator bias so far"   (optional)
   * }
   * `bias` is the measured bias in h (truth at 0); use null for
   * "not defined yet". Call again (e.g. from a sandbox toggle) to update —
   * the rail re-renders in place.
   */
  Book.biasRail = function (spec) {
    let host = document.getElementById("bias-rail");
    if (!host) {
      host = document.createElement("aside");
      host.id = "bias-rail";
      host.setAttribute("aria-label", "Bias ledger");
      document.body.appendChild(host);
    }
    const MIN = -0.18, MAX = 0.08;
    const pct = (b) => Math.max(0, Math.min(100, ((b - MIN) / (MAX - MIN)) * 100));
    let html = `<div class="bias-rail-title">${spec.title || "Bias ledger (in h, truth = 0)"}</div>`;
    for (const e of spec.entries) {
      const activeCls = e.active ? " active" : "";
      if (e.bias === null || e.bias === undefined) {
        html += `<div class="bias-rail-row${activeCls}"><span class="bias-rail-label">${e.label}</span><span class="bias-rail-na">not defined yet</span></div>`;
      } else {
        const sign = e.bias > 0 ? "+" : "";
        html += `<div class="bias-rail-row${activeCls}" title="${e.note || ""}">` +
          `<span class="bias-rail-label">${e.label}</span>` +
          `<span class="bias-rail-track"><span class="bias-rail-zero"></span>` +
          `<span class="bias-rail-marker" style="left:${pct(e.bias)}%"></span></span>` +
          `<span class="bias-rail-value">${sign}${e.bias.toFixed(3)}</span></div>`;
      }
    }
    host.innerHTML = html;
  };

  // ------------------------------------------------------------------
  // Nav built from the chapter manifest (js/manifest.js)
  // ------------------------------------------------------------------
  /**
   * If window.BOOK_CHAPTERS exists and the page carries
   * <nav class="book-nav" data-nav></nav>, populate it: "live" chapters as
   * links, "planned" ones greyed out. Pages with a hand-written nav (no
   * data-nav attribute) are left untouched.
   */
  Book.buildNav = function () {
    const nav = document.querySelector(".book-nav[data-nav]");
    if (!nav || !global.BOOK_CHAPTERS) return;
    nav.innerHTML = "";
    for (const ch of global.BOOK_CHAPTERS) {
      if (ch.status === "live") {
        const a = document.createElement("a");
        a.href = ch.file;
        a.textContent = ch.short;
        a.title = ch.title;
        nav.appendChild(a);
      } else {
        const s = document.createElement("span");
        s.className = "nav-planned";
        s.textContent = ch.short;
        s.title = ch.title + " (planned)";
        nav.appendChild(s);
      }
    }
  };

  // ------------------------------------------------------------------
  // Chapter nav: mark the current page's link active (data-nav-current
  // is set per-page via a data attribute on <body> or the link itself)
  // ------------------------------------------------------------------
  Book.markCurrentNav = function () {
    const here = global.location.pathname.split("/").pop() || "index.html";
    document.querySelectorAll(".book-nav a").forEach((a) => {
      const href = a.getAttribute("href");
      if (href === here) a.setAttribute("aria-current", "page");
    });
  };

  document.addEventListener("DOMContentLoaded", () => {
    Book.theme.init();
    Book.buildNav();
    Book.markCurrentNav();
    Book.renderMath(document.body);
  });

  global.Book = Book;
})(window);
