/*
 * book.js — small, dependency-free widget library for the discovery book.
 *
 * FROZEN for chapter agents: request new capabilities via
 * book/design/WIDGET_REQUESTS.md; only the integrator edits this file.
 *
 * Provides:
 *   Book.theme              — light/dark toggle, respects prefers-color-scheme
 *   Book.renderMath(root)   — KaTeX auto-render over a subtree (or document)
 *   Book.loadJSON(url)      — fetch + cache a data/*.json file; on failure
 *                             surfaces the page's own <noscript> fallbacks
 *                             instead of leaving blank boxes (§D-2)
 *   Book.widget(t,url,fn)   — explicit load-then-render wrapper (same fallback)
 *   Book.canon              — the canonical shared strings from manifest.js
 *                             (D1 dossier row + erratum, D3 cell-B pip and
 *                             job-ID split rule) — one definition, §D-6
 *   Book.gridSlider(opts)   — a slider bound to a precomputed data grid
 *   Book.predictReveal(el)  — "predict, then reveal" row (localStorage-persisted)
 *   Book.chrome()           — topbar controls group + ☰ mobile menu toggle
 *   Book.buildNav()         — chapter PICKER from window.BOOK_CHAPTERS
 *                             (js/manifest.js)
 *   Book.buildPager()       — previous/next chapter buttons at the page foot
 *   Book.themedPlot(...)    — Plotly plot that re-layouts on theme change
 *   Book.isDark()           — current effective theme
 *   Book.logsumexp / combineLogRows / normalizePosterior / trapz / argmaxIdx
 *                           — log-space posterior helpers (native representation)
 *   Book.biasRail(spec)     — the per-chapter Bias Ledger Rail (BW1); spec.pips
 *                             renders amber/grey "live, unquantified" annotations
 *                             (WIDGET_REQUESTS R-ch07-1 / R-ch11-1, integrator)
 *   Book.interp1(xs,ys,x)   — linear interpolation on a tabulated monotone grid
 *                             (R-ch07-2)
 *   Book.axis(text)         — Plotly-3.x-safe axis title object (R-ch11-3); in
 *                             addition Plotly.newPlot/react/relayout are wrapped
 *                             so string-form axis titles keep rendering
 *   Book.predictValue(opts) — numeric predict-then-reveal marker (R-ch04-1)
 *   Book.predictReveal(el, cb, {gates}) — cross-widget reveal locks (R-ch05-1)
 *   Book.passport           — the Symbol Passport (BW2): hover/tap any
 *                             `.term[data-term]`, pin to a personal glossary;
 *                             chapter-gated (SYMBOLS.firstChapter + .gloss)
 *                             so shared chrome cannot break the rung-guard
 *   Book.ledger             — "Has this been tried?" (BW3): per-page search over
 *                             data/museum_ledger.json + verdict hints for
 *                             sandbox states tagged data-hypothesis="<row#>";
 *                             tagged CONTROLS volunteer "⚖ #N — verdict" chips
 *                             in-widget on first use (§D-5; opt-out
 *                             data-hypothesis-verdict="inline")
 *   Book.lazyPlot(el, fn)   — build a widget when it nears the viewport
 *                             (ch03's IntersectionObserver recipe, §D-9)
 *   Book.chapter()          — current chapter number (99 = ungated pages)
 *   Book.persona            — global "Reading as: Curious / Methodology /
 *                             All details" switch (pre-expands strata; a step
 *                             down re-collapses what IT opened, never what
 *                             the reader opened)
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
  // Canonical shared strings (js/manifest.js -> window.BOOK_CANON)
  // ------------------------------------------------------------------
  /* REVISION_WORKLIST §D-6: the D1 dossier row + erratum line, the D3
   * cell-B rail pip and job-ID split rule have exactly ONE definition,
   * in js/manifest.js. Read them here rather than re-typing them. */
  Book.canon = global.BOOK_CANON || {};

  // ------------------------------------------------------------------
  // Tiny JSON cache (data/*.json is small — whole-file fetch is fine)
  // ------------------------------------------------------------------
  const _jsonCache = new Map();
  Book.loadJSON = function (url) {
    if (_jsonCache.has(url)) return _jsonCache.get(url);
    const p = fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`book.js: failed to load ${url} (${r.status})`);
        return r.json();
      })
      .catch((err) => {
        // UX MAJOR-1 (REVISION_WORKLIST §D-2): a failed fetch must never
        // leave a silent blank box. Surface the page's own static
        // fallbacks, then re-throw so no widget renders bogus data.
        Book.dataFailure(url, err);
        const e = err instanceof Error ? err : new Error(String(err));
        e.bookHandled = true;
        throw e;
      });
    _jsonCache.set(url, p);
    return p;
  };

  // ------------------------------------------------------------------
  // Data-failure surface (UX MAJOR-1 / REVISION_WORKLIST §D item 2)
  // ------------------------------------------------------------------
  /* Every chapter page loads its widgets with
   *   Book.loadJSON("data/chNN_x.json").then(render)
   * and (as shipped) no .catch(). Under `file://` in Chromium, a stale
   * deploy, or any 404, the reader used to get a fixed-height empty box
   * next to prose that assumes a chart is there — no error, no pointer,
   * nothing. Since the failure handler cannot live in 13 chapter files
   * (they are the chapter agents'), it lives here and works WITHOUT any
   * page edit: the first failure schedules one sweep that finds every
   * still-empty widget target and replaces it with that widget's own
   * <noscript> copy — which every chapter already wrote, and which is
   * exactly the right text.
   *
   * Newly written widgets should prefer the explicit wrapper
   * Book.widget(target, url, render) below; the sweep is the safety net
   * for everything already shipped. */

  /* Selectors for "a container a widget fills in with JS". `.readout`
   * spans are deliberately NOT here: they ship with static label text
   * ("$d_L$ = — Mpc"), so they are never "empty" and would veto the
   * emptiness test below. */
  const _FILL_TARGETS = ".widget-plot, .num-table, .table-scroll, [data-fill]";
  let _sweepTimer = null;
  let _finalSweepArmed = false;
  const _failedURLs = new Set();

  function _isEmptyTarget(el) {
    // rendered content = a Plotly graph, an inline figure, or table rows
    if (el.querySelector(".plotly, svg, canvas, tr, img")) return false;
    return el.textContent.replace(/[\s—–-]/g, "") === "";
  }

  /** Replace every still-empty widget container with the page's own
   *  <noscript> fallback (or a one-line pointer if it has none). */
  Book.degradeWidgets = function () {
    const urls = Array.from(_failedURLs).join(", ");
    const inserted = [];
    let touched = 0;
    document.querySelectorAll(".widget").forEach((w) => {
      if (w.classList.contains("is-data-failed")) return;
      const targets = Array.from(w.querySelectorAll(_FILL_TARGETS));
      if (!targets.length) return;
      // Only degrade a widget whose targets are ALL still empty — a widget
      // whose own data loaded fine must never be annotated.
      if (!targets.every(_isEmptyTarget)) return;
      w.classList.add("is-data-failed");
      const box = document.createElement("div");
      box.className = "data-fallback";
      box.setAttribute("role", "status");
      const ns = w.querySelector("noscript");
      // With scripting enabled a <noscript>'s content is parsed as raw
      // text, so .textContent is its markup — re-render it here.
      const fallback = ns ? ns.textContent : "";
      box.innerHTML =
        `<p class="data-fallback-head">Interactive data failed to load` +
        (urls ? ` (<code>${urls}</code>)` : "") +
        `. The static version of this figure follows.</p>` +
        (fallback ||
          `<p>Open this book over <code>http://</code> (e.g. ` +
          `<code>python3 -m http.server</code> in <code>book/site/</code>) — ` +
          `browsers block <code>fetch()</code> on <code>file://</code>. ` +
          `The surrounding text carries every number this widget shows.</p>`);
      const anchor = w.querySelector(".widget-controls") || targets[0];
      anchor.parentNode.insertBefore(box, anchor);
      inserted.push(box);
      touched += 1;
    });

    // Second pass: JS-filled tables that live in the prose with no .widget
    // wrapper and no <noscript> of their own (ch01's Omega_m table, ch10's
    // kernel and N-scaling tables). They would otherwise be the one blank
    // gap left on the page.
    document.querySelectorAll(_FILL_TARGETS).forEach((el) => {
      if (el.closest(".widget") || el.closest(".data-fallback")) return;
      if (el.dataset && el.dataset.bookFallback) return;
      if (!_isEmptyTarget(el)) return;
      const host = el.closest(".table-scroll") || el;
      if (host.previousElementSibling &&
          host.previousElementSibling.classList.contains("data-fallback")) return;
      const box = document.createElement("div");
      box.className = "data-fallback";
      box.setAttribute("role", "status");
      box.innerHTML =
        `<p class="data-fallback-head">A table belongs here, and its data ` +
        (urls ? `(<code>${urls}</code>) ` : "") +
        `failed to load. Open the book over <code>http://</code> — browsers ` +
        `block <code>fetch()</code> on <code>file://</code>. The paragraphs ` +
        `around this gap state the numbers it would show.</p>`;
      host.parentNode.insertBefore(box, host);
      if (el.dataset) el.dataset.bookFallback = "1";
      inserted.push(box);
      touched += 1;
    });

    // Render math only inside what was just inserted (the promoted <noscript>
    // copy is full of $...$); never re-walk the whole, already-rendered page.
    inserted.forEach((b) => Book.renderMath(b));
    return touched;
  };

  Book.dataFailure = function (url, err) {
    if (!_failedURLs.has(url)) {
      _failedURLs.add(url);
      console.warn(`book.js: ${url} could not be loaded — falling back to the ` +
        `page's static copy for every widget that needed it.`, err);
    }
    // Debounced, and re-armable: a fetch that fails late (a slow 404) must
    // still degrade its widget. degradeWidgets() is idempotent.
    const schedule = () => {
      if (_sweepTimer) clearTimeout(_sweepTimer);
      _sweepTimer = setTimeout(() => { _sweepTimer = null; Book.degradeWidgets(); }, 300);
    };
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", schedule, { once: true });
    } else {
      schedule();
    }
    // ...plus one final sweep after everything else on the page has had its
    // turn, so a widget whose render raced the first sweep is still caught.
    if (!_finalSweepArmed) {
      _finalSweepArmed = true;
      const fin = () => setTimeout(() => Book.degradeWidgets(), 1200);
      if (document.readyState === "complete") fin();
      else global.addEventListener("load", fin, { once: true });
    }
  };

  /* One clear warning instead of N unhandled-rejection traces: the
   * failure has already been surfaced on the page by the time this fires. */
  global.addEventListener("unhandledrejection", (ev) => {
    if (ev.reason && ev.reason.bookHandled) ev.preventDefault();
  });

  /**
   * Explicit data-widget wrapper (the going-forward API):
   *   Book.widget("#ch07-plot", "data/ch07_c7.json", (d, el) => {...})
   * Renders on success; on failure the shared fallback above takes over.
   * Returns the promise so callers can chain.
   */
  Book.widget = function (target, url, render) {
    const el = typeof target === "string"
      ? (document.querySelector(target) || document.getElementById(target))
      : target;
    return Book.loadJSON(url).then((d) => {
      if (el) render(d, el);
      return d;
    }).catch(() => { /* surfaced by Book.dataFailure */ });
  };

  // ------------------------------------------------------------------
  // Linear interpolation helper for closed-form JS math widgets
  // ------------------------------------------------------------------
  Book.lerp = (a, b, t) => a + (b - a) * t;

  /**
   * Linear interpolation of tabulated (xs, ys) at x (scalar or array).
   * xs must be sorted ascending; values are clamped at the ends.
   * (WIDGET_REQUESTS R-ch07-2 — pairs with Book.trapz for live integrals
   * over Python-tabulated functions.)
   */
  Book.interp1 = function (xs, ys, x) {
    function one(v) {
      if (v <= xs[0]) return ys[0];
      if (v >= xs[xs.length - 1]) return ys[ys.length - 1];
      let lo = 0, hi = xs.length - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (xs[mid] <= v) lo = mid; else hi = mid;
      }
      const t = (v - xs[lo]) / (xs[hi] - xs[lo]);
      return ys[lo] + (ys[hi] - ys[lo]) * t;
    }
    return Array.isArray(x) ? x.map(one) : one(x);
  };

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
   * pedagogy's "locked, not suggested" rule). If the container — or any
   * element inside it (the template puts data-predict-id on the
   * .predict-row; WIDGET_REQUESTS R-ch04-1) — carries a data-predict-id
   * attribute, the reader's choice is persisted to localStorage under
   * "book-predict:<id>" and restored on reload, so a chapter can
   * re-surface earlier predictions (e.g. Ch 11 re-surfacing the Ch 3
   * guess).
   *
   * opts.gates (WIDGET_REQUESTS R-ch05-1): elements or selectors of OTHER
   * widgets that stay inert (class `is-predict-locked`) until this
   * prediction is made. The class is only ever added by JS, so a no-JS
   * reader is never locked out.
   */
  Book.predictReveal = function (container, onPredict, opts) {
    const buttons = container.querySelectorAll("[data-predict]");
    const reveal = container.querySelector(".reveal");
    const idHost = container.hasAttribute("data-predict-id")
      ? container
      : container.querySelector("[data-predict-id]");
    const pid = idHost ? idHost.getAttribute("data-predict-id") : null;
    const storageKey = pid ? "book-predict:" + pid : null;
    // Uniform grading (REVISION_WORKLIST §D item 8, mara MAJOR-3): an
    // optional data-predict-correct="<choice>" on the container (or any
    // descendant, matching the id-resolution rule) names the graded
    // option. On reveal the correct button is marked .predict-correct
    // and a wrong chosen button .predict-missed — the reveal's prose
    // still owns the verdict wording (ch02 is the first customer).
    const gradeHost = container.hasAttribute("data-predict-correct")
      ? container
      : container.querySelector("[data-predict-correct]");
    const correct = gradeHost ? gradeHost.getAttribute("data-predict-correct") : null;
    const gates = ((opts && opts.gates) || [])
      .map((g) => (typeof g === "string" ? document.querySelector(g) : g))
      .filter(Boolean);

    function setGates(locked) {
      gates.forEach((g) => g.classList.toggle("is-predict-locked", locked));
    }

    function select(btn, fire) {
      buttons.forEach((b) => b.setAttribute("aria-pressed", "false"));
      btn.setAttribute("aria-pressed", "true");
      if (reveal) reveal.classList.add("shown");
      if (correct) {
        buttons.forEach((b) => {
          const isCorrect = b.getAttribute("data-predict") === correct;
          b.classList.toggle("predict-correct", isCorrect);
          b.classList.toggle("predict-missed", b === btn && !isCorrect);
        });
      }
      setGates(false);
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

    let restored = false;
    if (storageKey) {
      try {
        const saved = localStorage.getItem(storageKey);
        if (saved !== null) {
          const btn = container.querySelector(`[data-predict="${saved}"]`);
          if (btn) {
            select(btn, false);
            restored = true;
          }
        }
      } catch (e) { /* ignore */ }
    }
    if (!restored) setGates(true);
  };

  /**
   * Numeric predict-then-reveal (WIDGET_REQUESTS R-ch04-1): a continuous
   * "where will it land?" marker instead of a button choice.
   *
   *   opts = { slider, button, id, onLock, format }
   *     slider  - <input type="range"> element or selector
   *     button  - the single lock/commit button (element or selector)
   *     id      - persistence key -> localStorage "book-predict:<id>"
   *     onLock  - (value, wasRestored) => void
   *     format  - optional (value) => string for a .readout inside the row
   *
   * On lock: persists the number, reveals the nearest `.reveal` (sibling of
   * the button's .predict-row, or inside the closest .widget), and fires
   * onLock(value, false). On load with a stored value: restores the slider
   * position and fires onLock(value, true).
   */
  Book.predictValue = function (opts) {
    const slider = typeof opts.slider === "string" ? document.querySelector(opts.slider) : opts.slider;
    const button = typeof opts.button === "string" ? document.querySelector(opts.button) : opts.button;
    if (!slider || !button) return;
    const storageKey = "book-predict:" + opts.id;
    const scope = button.closest(".widget") || document;
    const reveal = scope.querySelector(".reveal");

    function lock(value, wasRestored) {
      button.setAttribute("aria-pressed", "true");
      if (reveal) reveal.classList.add("shown");
      if (!wasRestored) {
        try { localStorage.setItem(storageKey, String(value)); } catch (e) { /* ignore */ }
      }
      if (opts.onLock) opts.onLock(value, !!wasRestored);
    }

    button.addEventListener("click", () => lock(Number(slider.value), false));
    try {
      const saved = localStorage.getItem(storageKey);
      if (saved !== null && saved !== "" && isFinite(Number(saved))) {
        slider.value = saved;
        if (opts.format) {
          const ro = scope.querySelector(".readout strong") || scope.querySelector(".readout");
          if (ro) ro.textContent = opts.format(Number(saved));
        }
        lock(Number(saved), true);
      }
    } catch (e) { /* ignore */ }
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
  /* ONE theme observer for every themed plot (REVISION_WORKLIST §D item 9,
   * ux MINOR): the per-plot MutationObservers are consolidated — pages with
   * 8 Plotly instances used to run 8 observers watching the same
   * attribute flip. Registrations survive re-renders (update() re-reads
   * layoutFn); a plot whose container has left the DOM is dropped. */
  const _themedPlots = [];
  let _themeObserver = null;

  function _ensureThemeObserver() {
    if (_themeObserver) return;
    _themeObserver = new MutationObserver(() => {
      for (let i = _themedPlots.length - 1; i >= 0; i--) {
        const p = _themedPlots[i];
        if (!document.getElementById(p.divId)) {
          _themedPlots.splice(i, 1);
          continue;
        }
        try {
          Plotly.relayout(p.divId, p.layoutFn());
        } catch (e) {
          console.warn(`book.js: theme relayout failed for #${p.divId}`, e);
        }
      }
    });
    _themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
  }

  Book.themedPlot = function (divId, traces, layoutFn, config) {
    const cfg = config || { displayModeBar: false, responsive: true };
    Plotly.newPlot(divId, traces, layoutFn(), cfg);
    _ensureThemeObserver();
    _themedPlots.push({ divId: divId, layoutFn: layoutFn });
    return {
      update(newTraces) {
        Plotly.react(divId, newTraces, layoutFn(), cfg);
      },
    };
  };

  // ------------------------------------------------------------------
  // Lazy widget construction (REVISION_WORKLIST §D item 9, ux MAJOR-3)
  // ------------------------------------------------------------------
  /**
   * Book.lazyPlot(target, build) — run `build()` once, when `target`
   * (element or selector) approaches the viewport (±300px), instead of at
   * page load. This is the ch03 IntersectionObserver recipe, centralized;
   * without IntersectionObserver (or with a missing target) it degrades
   * to the eager path. ch02/ch08/ch09 currently ship the identical
   * page-local copy (zero-coupling rule, same precedent as interp1);
   * new chapters should call this instead.
   */
  Book.lazyPlot = function (target, build) {
    const el = typeof target === "string"
      ? (document.querySelector(target) || document.getElementById(target))
      : target;
    if (!el || !("IntersectionObserver" in global)) {
      build();
      return;
    }
    const io = new IntersectionObserver((entries, obs) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        obs.disconnect();
        build();
      });
    }, { rootMargin: "300px 0px 300px 0px" });
    io.observe(el);
  };

  // ------------------------------------------------------------------
  // Plotly 3.x title-form shim (WIDGET_REQUESTS R-ch11-3)
  // ------------------------------------------------------------------
  /** Plotly-3.x-safe axis/layout title: Book.axis("h") -> { text: "h" }. */
  Book.axis = function (text) {
    return { text: text };
  };

  /* Plotly 3.x (vendored: 3.4.0) dropped the string shorthand for
   * `layout.title` and `layout.<axis>.title` — a bare string silently renders
   * NO title at all.  Several chapters were written with the string form, so
   * every layout that passes through Plotly is normalized here: any string
   * `title` property in the layout tree is wrapped into `{ text: ... }`.
   * Trace objects are untouched. */
  function _fixTitleForms(layout, depth) {
    if (!layout || typeof layout !== "object" || Array.isArray(layout)) return layout;
    const d = depth || 0;
    if (d > 6) return layout;
    for (const key of Object.keys(layout)) {
      const v = layout[key];
      if ((key === "title" || key.slice(-6) === ".title") && typeof v === "string") {
        layout[key] = { text: v };
      } else if (v && typeof v === "object" && !Array.isArray(v)) {
        _fixTitleForms(v, d + 1);
      }
    }
    return layout;
  }
  Book._fixTitleForms = _fixTitleForms; // exposed for testing

  if (global.Plotly) {
    const _newPlot = global.Plotly.newPlot.bind(global.Plotly);
    const _react = global.Plotly.react.bind(global.Plotly);
    const _relayout = global.Plotly.relayout.bind(global.Plotly);
    global.Plotly.newPlot = function (div, data, layout, config) {
      return _newPlot(div, data, _fixTitleForms(layout), config);
    };
    global.Plotly.react = function (div, data, layout, config) {
      return _react(div, data, _fixTitleForms(layout), config);
    };
    global.Plotly.relayout = function (div, update, value) {
      if (value === undefined && update && typeof update === "object") {
        return _relayout(div, _fixTitleForms(update));
      }
      return _relayout(div, update, value);
    };
  }

  // ------------------------------------------------------------------
  // Current chapter number (shared: passport gating + cumulative rail)
  // ------------------------------------------------------------------
  /** Which chapter is the reader on? ch07-redshift.html -> 7. Pages with
   *  no chapter number (index, museum, template) return 99, i.e. no
   *  gating — the museum is the annex that may say everything. A page
   *  can override with <body data-chapter="7"> if it is ever renamed. */
  Book.chapter = function () {
    const attr = document.body && document.body.getAttribute("data-chapter");
    if (attr !== null && attr !== undefined && attr !== "") {
      const n = parseInt(attr, 10);
      if (isFinite(n)) return n;
    }
    const file = (global.location.pathname.split("/").pop() || "");
    const m = /^ch(\d+)/.exec(file);
    return m ? parseInt(m[1], 10) : 99;
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
   *   title: "Estimator bias so far",  (optional)
   *   pips: [{ label: "C7 — inflates at σ_z/z > 0.256", tone: "amber"|"grey",
   *            note: "live, FINDING" }, ...]   (optional)
   * }
   * `bias` is the measured bias in h (truth at 0); use null for
   * "not defined yet". `pips` (WIDGET_REQUESTS R-ch07-1 / R-ch11-1) render as
   * coloured dots under the entries — the "live, has no bias number, and that
   * is the point" channel: amber = measured-but-unresolved defect, grey =
   * confounded / in-flight. Call again (e.g. from a sandbox toggle) to
   * update — the rail re-renders in place.
   *
   * CUMULATIVE HISTORY (REVISION_WORKLIST §D item 4, ped M7): the page's
   * entries are merged with window.BOOK_BIAS_ROWS (js/manifest.js) so the
   * rail never loses rows moving forward. Book rows with
   * from_chapter <= Book.chapter() render even when the page did not
   * declare them; a page row whose label matches a book row's `match`
   * list REPLACES that book row (the chapter's own wording, note and
   * arming pattern win). Page rows with no book counterpart (e.g. a
   * chapter's live sandbox row) are appended after the history.
   */
  function _mergeBiasRows(pageEntries) {
    const n = Book.chapter();
    const rows = (global.BOOK_BIAS_ROWS || []).filter(
      (r) => typeof r.from_chapter === "number" && r.from_chapter <= n
    );
    if (!rows.length) return pageEntries;
    const consumed = new Set();
    const merged = [];
    for (const r of rows) {
      let hit = -1;
      for (let i = 0; i < pageEntries.length; i++) {
        if (consumed.has(i)) continue;
        const label = String(pageEntries[i].label || "").toLowerCase();
        if ((r.match || []).some((m) => label.indexOf(m) >= 0)) { hit = i; break; }
      }
      if (hit >= 0) {
        consumed.add(hit);
        merged.push(pageEntries[hit]);
      } else {
        merged.push({ label: r.label, bias: r.bias, note: r.note });
      }
    }
    pageEntries.forEach((e, i) => { if (!consumed.has(i)) merged.push(e); });
    return merged;
  }
  Book._mergeBiasRows = _mergeBiasRows; // exposed for testing

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
    for (const e of _mergeBiasRows(spec.entries || [])) {
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
    if (spec.pips && spec.pips.length) {
      html += `<div class="bias-rail-pipsep">live, unquantified</div>`;
      for (const p of spec.pips) {
        const tone = p.tone === "grey" ? " grey" : " amber";
        html += `<div class="bias-rail-pip${tone}" title="${p.note || ""}">` +
          `<span class="bias-rail-pip-dot"></span><span>${p.label}</span></div>`;
      }
    }
    host.innerHTML = html;
  };

  // ------------------------------------------------------------------
  // Symbol Passport (BW2, WIDGET_REQUESTS R-INT-1)
  // ------------------------------------------------------------------
  /* The binding notation table of BOOK_DESIGN.md §3.1, transcribed verbatim
   * (meaning / units / defining source). Chapters mark occurrences with
   * <span class="term" data-term="KEY">…</span>; the passport attaches a
   * hover/tap card and a pin-to-glossary control to every one.
   *
   * CHAPTER GATING (REVISION_WORKLIST §D item 1; mara MAJOR-2 / ped M4).
   * Shared chrome must not defeat the rung-guard that binds every chapter
   * agent: hovering w_G in Ch 5 used to hand the reader "β_G/D —
   * ESTIMAND-DEPENDENT" and "C9 is a live FINDING", i.e. Ch 9's reveal,
   * four chapters early. An entry may therefore carry:
   *
   *   firstChapter: <n>   the first chapter from which the FULL card is
   *                       safe — i.e. one past any chapter whose own
   *                       reveal the card would pre-empt. (eps is 8, not
   *                       7: both `data-term="eps"` tags sit in Ch 7's
   *                       deck and §1, while §6 is where the chapter
   *                       *arrives* at 0.256 — gating at 7 would be a
   *                       no-op and ped M4's complaint would stand.)
   *   gloss:  "<rung-safe one-liner>"   shown INSTEAD of `meaning` before
   *                                      that chapter (and `note` is
   *                                      suppressed entirely)
   *
   * Both are required together — gating without a gloss would show an
   * empty card, so `firstChapter` alone is ignored. Symbol, units and the
   * defining code site stay unconditional: that is the passport's actual
   * job for Mara and Tomas, and none of it spoils anything. */
  Book.SYMBOLS = {
    h:      { sym: "$h$", meaning: "H₀ / 100 km s⁻¹ Mpc⁻¹; mock truth h_true = 0.73", units: "—", src: "constants.py H" },
    H0:     { sym: "$H_0$", meaning: "Hubble constant", units: "km s⁻¹ Mpc⁻¹", src: "constants.py" },
    dL:     { sym: "$d_L$", meaning: "luminosity distance", units: "Mpc (Gpc for pools)", src: "physical_relations.py:132 dist" },
    z:      { sym: "$z$", meaning: "true redshift", units: "—", src: "dark_siren_likelihood.md §2.4" },
    zg:     { sym: "$z_g$", meaning: "catalogue (observed) redshift of galaxy g", units: "—", src: "handler.py; K1" },
    sigz:   { sym: "$\\sigma_z$", meaning: "host-z kernel width (total: measurement ⊕ peculiar velocity)", units: "—", src: "hostz_pv_photoz_kernel.md" },
    eps:    { sym: "$\\sigma_z/z$", meaning: "fractional z width — the C7 variable; rail threshold quoted at 0.256", units: "—", src: "C7_README", note: "C7 is a live FINDING — see Ch 7 §6 / Ch 11",
              firstChapter: 8, gloss: "how wide the host-redshift smear is compared with the redshift itself — the fractional width of the kernel" },
    Om:     { sym: "$\\Omega_m$", meaning: "matter density; fiducial 0.2726 (Barausse M1, a design choice)", units: "—", src: "constants.py; G11" },
    Ez:     { sym: "$E(z)$", meaning: "√(Ω_m(1+z)³ + Ω_Λ)", units: "—", src: "dark_siren_likelihood.md §2.4" },
    dVc:    { sym: "$dV_c/dz$", meaning: "comoving volume element per unit z (per sr where noted)", units: "Mpc³", src: "physical_relations.py:571" },
    wpop:   { sym: "$w_{\\rm pop}$", meaning: "(dV_c/dz)/(1+z) — the volume/rate prior in z", units: "Mpc³", src: "G2b §1" },
    Zg:     { sym: "$Z_g$", meaning: "per-galaxy kernel normalization (∝ h⁻³, exact)", units: "Mpc³", src: "G2b §3" },
    M:      { sym: "$M$", meaning: "MBH (source-frame) mass", units: "M☉", src: "datamodels/parameter_space.py" },
    mu:     { sym: "$\\mu$", meaning: "compact-object mass", units: "M☉", src: "parameter_space.py" },
    Mz:     { sym: "$M_z$", meaning: "redshifted (detector-frame) mass M(1+z)", units: "M☉", src: "dark_siren_likelihood.md §7" },
    Mg:     { sym: "$M_g$", meaning: "catalogue BH-mass proxy of galaxy g (d_L-derived — hidden h-dependence, RATIFY-M7)", units: "M☉", src: "mass_marginal_2d_kernel.md" },
    slnM:   { sym: "$\\sigma_{\\ln M}$", meaning: "mass-proxy scatter (≈0.58 kernel-side / ≈1.28 catalogue-side — state which)", units: "—", src: "mass_marginal_2d_kernel.md; RV15" },
    snr:    { sym: "SNR / $\\rho$", meaning: "matched-filter signal-to-noise; threshold 20", units: "—", src: "parameter_estimation.py:488; constants.py" },
    Gam:    { sym: "$\\Gamma_{ab}$", meaning: "Fisher matrix ⟨∂_a h | ∂_b h⟩", units: "mixed", src: "parameter_estimation.py:399" },
    Sig:    { sym: "$\\Sigma$", meaning: "Cramér–Rao covariance Γ⁻¹", units: "mixed", src: "parameter_estimation.py:430",
              firstChapter: 6, gloss: "the measurement's covariance: how big the error ellipsoid is and which way it tilts" },
    u:      { sym: "$u$", meaning: "fractional distance d_L/d̂_L (mean 1)", units: "—", src: "E4; bayesian_statistics.py:1856" },
    phth:   { sym: "$\\phi, \\theta$", meaning: "sky coordinates (frame-stamped)", units: "rad", src: "LISA_configuration.py" },
    wg:     { sym: "$w_g$", meaning: "rate weight R_eff(M_g)/(1+z_g)", units: "yr⁻¹-ish (relative)", src: "bayesian_statistics.py:879; G2c D1" },
    Reff:   { sym: "$R_{\\rm eff}(M)$", meaning: "per-MBH EMRI rate", units: "yr⁻¹", src: "Babak 2017 arXiv:1703.09722" },
    Ng:     { sym: "$N_g$", meaning: "per-galaxy numerator integral", units: "—", src: "G2c §2" },
    Dg:     { sym: "$D_g$", meaning: "per-galaxy selection integral", units: "—", src: "G2c §2" },
    Lcat:   { sym: "$\\mathcal{L}^{\\rm cat}$", meaning: "catalogue leg Σ w_g N_g / Σ w_g D_g (ratio of sums)", units: "—", src: "G2c §2, §4; ledger #26" },
    Lcomp:  { sym: "$\\mathcal{L}^{\\rm comp}$", meaning: "completion leg B^num/β_Ḡ (diagnostic identity D4)", units: "—", src: "G2c D4" },
    Bnum:   { sym: "$B^{\\rm num}$", meaning: "completion numerator (1/4π sky marginal)", units: "Mpc³ sr⁻¹", src: "G2a; bayesian_statistics.py:3210-3238" },
    betaG:  { sym: "$\\beta_G$", meaning: "catalogued-side selection integral", units: "Mpc³ sr⁻¹", src: "G2c; bayesian_statistics.py (β_G = D − β_Ḡ)" },
    betaGbar: { sym: "$\\beta_{\\bar G}$", meaning: "dark-side selection integral", units: "Mpc³ sr⁻¹", src: "bayesian_statistics.py:1170" },
    Dh:     { sym: "$D(h)$", meaning: "full-volume selection normalization β_G + β_Ḡ", units: "Mpc³ sr⁻¹", src: "bayesian_statistics.py:1013; G2c §6 C2 (cite “denominator of (A14)”)" },
    wG:     { sym: "$w_G$", meaning: "mixture weight β_G/D — ESTIMAND-DEPENDENT; always name the mode", units: "—", src: "bayesian_statistics.py:3309-3311; C9", note: "C9 is a live FINDING — see Ch 9 §6 / Ch 11",
              firstChapter: 9, gloss: "mixture weight — the probability the host is catalogued; how it is computed, and what it is normalized to, is Ch 9" },
    pdet:   { sym: "$p_{\\rm det}$", meaning: "detection probability (horizon-survival estimator)", units: "—", src: "simulation_detection_probability.py" },
    fz:     { sym: "$f(z,\\Omega)$", meaning: "catalogue completeness fraction", units: "—", src: "G2c D2; pixel_completeness.py" },
    pi:     { sym: "$p_i(h)$", meaning: "per-event likelihood (the master equation)", units: "—", src: "bayesian_statistics.py:3006-3009, 1042-1048" },
    Cscale: { sym: "$C$", meaning: "the arbitrary mass-coordinate rescale of the C8 walk", units: "—", src: "README_C8.md", note: "C8 is a live FINDING — see Ch 8 §6 / Ch 11",
              firstChapter: 8, gloss: "a free rescaling of the mass coordinate — a change of units the answer should not notice" },
    sigh:   { sym: "$\\sigma_h$", meaning: "posterior width in h", units: "—", src: "readouts" },

    /* --- the two-estimands normalization block (REVISION_WORKLIST §D-7,
     * tomas M5): Ch 9 §4 introduced these inside two display equations
     * with no passport entry at all; Σ_glob also leaks into Ch 7 §6 and
     * Ch 11's C8 dimension count. Units are the derivation's own
     * (DERIVATION_GENERATOR_CONSISTENT_NORM.md §4 units table), not
     * inferred here. None of these is gated: they carry no later
     * chapter's verdict. */
    nw:     { sym: "$n_w$", meaning: "rate-weight density that converts the discrete catalogue sum Σ w_g N_g into the same measure as the model integrals — mode-dependent: n̄_w = Σ_glob/β_G (absolute_marginal) vs n̂_w = W_cat/V_f (generator_marginal)", units: "yr⁻¹ sr Mpc⁻³", src: "DERIVATION_GENERATOR_CONSISTENT_NORM.md §2.3 (4); bayesian_statistics.py" },
    Sglob:  { sym: "$\\Sigma_{\\rm glob}$", meaning: "catalogue-channel detection expectation Σ_{g: z_g<z_max} w_g P_det(d_L(z_g;h)) — the normalization packet's spelling of G2c's Σ_global", units: "yr⁻¹", src: "bayesian_statistics.py:998-1195 precompute_global_catalog_selection; DERIVATION_GENERATOR_CONSISTENT_NORM.md §2.1" },
    Wcat:   { sym: "$W_{\\rm cat}$", meaning: "total draw-eligible rate weight of the pruned catalogue, Σ_{g: z_g<1.5} w_g — an h-independent scalar (the generator's own draw normalizer)", units: "yr⁻¹", src: "DERIVATION_GENERATOR_CONSISTENT_NORM.md §2.2; :243" },
    Vf:     { sym: "$V_f(h)$", meaning: "completeness-weighted population volume ∫₀^{1.5} f̄(z,h)(dV_c/dz)/(1+z) dz; exactly ∝ h⁻³, V_f(0.73) = 2.3237×10⁸", units: "Mpc³ sr⁻¹", src: "DERIVATION_GENERATOR_CONSISTENT_NORM.md §2.2, §3.2" },
    Fincat: { sym: "$F$", meaning: "pre-detection in-catalogue fraction V_f/V_tot — the generator's Bernoulli channel split, derived not posited; 0.0175370 and exactly h-independent", units: "—", src: "DERIVATION_GENERATOR_CONSISTENT_NORM.md §2.1 (2), §3.2; ledger #81" },
    phcat:  { sym: "$\\phi_{\\rm cat}$", meaning: "selected number density — what a flux-limited catalogue row is drawn from: f(z,Ω)·(dV_c/dz)/(1+z). Distinct from the prior over where an EMRI host is (that is w_pop)", units: "Mpc³ sr⁻¹ per unit z", src: "G2c D2 (f) × G2b §1 (w_pop); Ch 7 §6 / Ch 11 §4" },
  };

  Book.passport = {
    GLOSSARY_KEY: "book-glossary",
    /* Whether the glossary panel is open — persisted, so paging to the next
     * chapter keeps the reader's pinned symbols on screen instead of making
     * them re-open it every page (reader feedback 2026-08-04). */
    GLOSSARY_OPEN_KEY: "book-glossary-open",
    _pop: null,

    _pinned() {
      try {
        return JSON.parse(localStorage.getItem(Book.passport.GLOSSARY_KEY) || "[]");
      } catch (e) { return []; }
    },
    _setPinned(list) {
      try {
        localStorage.setItem(Book.passport.GLOSSARY_KEY, JSON.stringify(list));
      } catch (e) { /* ignore */ }
    },

    /** The reader's chapter number — delegates to the shared Book.chapter()
     *  (one definition; the cumulative bias rail uses the same one). */
    _chapter() {
      return Book.chapter();
    },

    /** The card's visible fields for the current page (chapter-gated). */
    _view(key) {
      const s = Book.SYMBOLS[key];
      if (!s) return null;
      const gated = !!(s.gloss && typeof s.firstChapter === "number" &&
        Book.passport._chapter() < s.firstChapter);
      return {
        sym: s.sym,
        meaning: gated ? s.gloss : s.meaning,
        units: s.units,
        // the code site stays (it is the passport's job for Tomas), but a
        // trailing claim reference — "…py:3309-3311; C9" — is a verdict
        // pointer, not a source, and is dropped with the rest of the card.
        src: gated ? s.src.replace(/[;,]\s*C\d+\b/g, "").trim() : s.src,
        note: gated ? "" : (s.note || ""),
        gatedFrom: gated ? s.firstChapter : null,
      };
    },

    _card(key) {
      const v = Book.passport._view(key);
      if (!v) return null;
      const pinned = Book.passport._pinned().indexOf(key) >= 0;
      const note = v.note ? `<div class="passport-note">${v.note}</div>` : "";
      // D4 spoiler discipline: name the chapter, never its number or verdict.
      const later = v.gatedFrom !== null
        ? `<div class="passport-later">full card from Ch ${v.gatedFrom} on</div>`
        : "";
      return (
        `<div class="passport-sym">${v.sym}</div>` +
        `<div class="passport-meaning">${v.meaning}</div>` +
        `<div class="passport-row"><span>units</span> ${v.units}</div>` +
        `<div class="passport-row"><span>defined by</span> <code>${v.src}</code></div>` +
        note + later +
        `<button type="button" class="passport-pin" data-key="${key}" aria-pressed="${pinned}">` +
        (pinned ? "★ pinned — unpin" : "☆ pin to my glossary") + `</button>`
      );
    },

    _show(termEl, key) {
      let pop = Book.passport._pop;
      if (!pop) {
        pop = document.createElement("div");
        pop.className = "passport-pop";
        pop.setAttribute("role", "tooltip");
        document.body.appendChild(pop);
        Book.passport._pop = pop;
      }
      const html = Book.passport._card(key);
      if (!html) return;
      pop.innerHTML = html;
      pop.style.display = "block";
      const r = termEl.getBoundingClientRect();
      const popW = Math.min(300, window.innerWidth - 24);
      let left = window.scrollX + r.left;
      if (left + popW > window.scrollX + window.innerWidth - 12) {
        left = window.scrollX + window.innerWidth - popW - 12;
      }
      pop.style.left = left + "px";
      pop.style.top = window.scrollY + r.bottom + 6 + "px";
      pop.style.width = popW + "px";
      Book.renderMath(pop);
      const pin = pop.querySelector(".passport-pin");
      if (pin) {
        pin.addEventListener("click", (ev) => {
          ev.stopPropagation();
          const k = pin.getAttribute("data-key");
          let list = Book.passport._pinned();
          if (list.indexOf(k) >= 0) list = list.filter((x) => x !== k);
          else list.push(k);
          Book.passport._setPinned(list);
          Book.passport._show(termEl, key); // re-render pin state
          Book.passport._renderGlossary();
        });
      }
    },

    _hide() {
      if (Book.passport._pop) Book.passport._pop.style.display = "none";
    },

    /** Unpin a symbol from anywhere (glossary row ✕, or the card's pin). */
    unpin(key) {
      Book.passport._setPinned(
        Book.passport._pinned().filter((x) => x !== key)
      );
      Book.passport._renderGlossary();
    },

    _renderGlossary() {
      const panel = document.getElementById("passport-glossary");
      if (!panel) return;
      const list = Book.passport._pinned();
      if (!list.length) {
        panel.innerHTML =
          `<div class="passport-glossary-title">My glossary</div>` +
          `<p class="passport-glossary-empty">Nothing pinned yet — tap any ` +
          `<span class="term-example">dotted symbol</span> in the text and pin it.</p>`;
      } else {
        // Removal lives in the panel itself (reader feedback 2026-08-04): a
        // reader who over-pinned had to hunt the original term in the prose
        // to get the card back and unpin from there.
        let html = `<div class="passport-glossary-title">My glossary` +
          `<button type="button" class="passport-glossary-clear" ` +
          `title="Unpin every symbol">clear all</button></div>`;
        for (const k of list) {
          // same chapter gating as the hover card — a symbol pinned in Ch 9
          // must not re-spoil Ch 9 when the reader pages back to Ch 5.
          const v = Book.passport._view(k);
          if (!v) continue;
          html += `<div class="passport-glossary-item">` +
            `<button type="button" class="passport-unpin" data-unpin="${k}" ` +
            `title="Remove from my glossary" aria-label="Remove ${k} from my glossary">&times;</button>` +
            `<span class="passport-glossary-body"><strong>${v.sym}</strong> ` +
            `${v.meaning} <em>[${v.units}]</em> <code>${v.src}</code></span></div>`;
        }
        panel.innerHTML = html;
      }
      Book.renderMath(panel);
    },

    init() {
      const terms = document.querySelectorAll(".term[data-term]");
      if (!terms.length) return;
      terms.forEach((el) => {
        const key = el.getAttribute("data-term");
        if (!Book.SYMBOLS[key]) return;
        el.classList.add("term-live");
        el.setAttribute("tabindex", "0");
        el.setAttribute("aria-label", "Symbol passport: " + key);
        el.addEventListener("mouseenter", () => Book.passport._show(el, key));
        el.addEventListener("focus", () => Book.passport._show(el, key));
        el.addEventListener("click", (ev) => {
          ev.stopPropagation();
          Book.passport._show(el, key);
        });
        el.addEventListener("mouseleave", (ev) => {
          // keep the card open if the pointer moved onto it (to reach the pin)
          const to = ev.relatedTarget;
          if (to && Book.passport._pop && Book.passport._pop.contains(to)) return;
          setTimeout(() => {
            const pop = Book.passport._pop;
            if (pop && !pop.matches(":hover")) Book.passport._hide();
          }, 250);
        });
      });
      document.addEventListener("click", (ev) => {
        const pop = Book.passport._pop;
        if (pop && !pop.contains(ev.target)) Book.passport._hide();
      });
      document.addEventListener("keydown", (ev) => {
        if (ev.key === "Escape") Book.passport._hide();
      });
      if (Book.passport._pop) {
        Book.passport._pop.addEventListener("mouseleave", () => Book.passport._hide());
      }
      // glossary panel + topbar toggle
      const topbar = document.querySelector(".book-topbar");
      if (topbar && !document.getElementById("passport-glossary")) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "theme-toggle passport-glossary-toggle";
        btn.textContent = "★ Glossary";
        btn.setAttribute("aria-expanded", "false");
        const panel = document.createElement("aside");
        panel.id = "passport-glossary";
        panel.setAttribute("aria-label", "Pinned symbol glossary");
        panel.style.display = "none";
        document.body.appendChild(panel);

        const setOpen = (open, persist) => {
          panel.style.display = open ? "block" : "none";
          btn.setAttribute("aria-expanded", String(open));
          if (open) Book.passport._renderGlossary();
          if (persist) {
            try {
              localStorage.setItem(Book.passport.GLOSSARY_OPEN_KEY, open ? "1" : "0");
            } catch (e) { /* ignore */ }
          }
        };
        btn.addEventListener("click", () => {
          setOpen(panel.style.display === "none", true);
        });

        // Removal controls, delegated (the panel re-renders on every change).
        panel.addEventListener("click", (ev) => {
          const un = ev.target.closest("[data-unpin]");
          if (un) {
            Book.passport.unpin(un.getAttribute("data-unpin"));
            return;
          }
          if (ev.target.closest(".passport-glossary-clear")) {
            Book.passport._setPinned([]);
            Book.passport._renderGlossary();
          }
        });

        const controls = topbar.querySelector(".book-controls");
        const themeBtn = topbar.querySelector("[data-theme-toggle]");
        if (controls) controls.insertBefore(btn, themeBtn || null);
        else topbar.insertBefore(btn, themeBtn);

        // Restore the panel's open state across chapters.
        let wasOpen = false;
        try {
          wasOpen = localStorage.getItem(Book.passport.GLOSSARY_OPEN_KEY) === "1";
        } catch (e) { /* ignore */ }
        if (wasOpen) setOpen(true, false);
      }
    },
  };

  // ------------------------------------------------------------------
  // "Has this been tried?" — ledger search (BW3, WIDGET_REQUESTS R-INT-2)
  // ------------------------------------------------------------------
  /* A per-page search box over data/museum_ledger.json (the 98-row digest of
   * BIAS_HISTORY_LEDGER.md, owned by the museum agent), injected next to the
   * chapter's sandboxes, plus verdict hints for sandbox states tagged
   * data-hypothesis="<ledger row #>". The museum's own full-table instrument
   * (#mus-search) takes precedence on that page. */
  Book.ledger = {
    _data: null,

    load() {
      if (Book.ledger._data) return Book.ledger._data;
      Book.ledger._data = Book.loadJSON("data/museum_ledger.json").then((d) => d.rows || d);
      return Book.ledger._data;
    },

    _fmtRow(r) {
      const dnr = r.do_not_retry ? ` <span class="ledger-dnr">do-not-re-try</span>` : "";
      // book_note: a book-added disambiguation (NOT ledger text) — e.g. row
      // #88's "Cell B" naming collision with the 2×2 cell B (worklist MJ-3).
      // Rendering it here keeps a chapter-page search from misleading.
      const note = r.book_note
        ? ` <span class="ledger-date">${r.book_note}</span>` : "";
      return (
        `<div class="ledger-result"><strong>#${r.id}</strong> ` +
        `<span class="ledger-date">${r.date || r.era || ""}</span> — ` +
        `${r.hypothesis} → <em>${r.verdict}</em>${dnr} ` +
        `<span class="prov-chip">${r.documented || ""}</span>${note}</div>`
      );
    },

    search(rows, q) {
      const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
      if (!terms.length) return [];
      return rows.filter((r) => {
        const hay = `#${r.id} ${r.id} ${r.hypothesis} ${r.test} ${r.verdict} ${r.documented} ${r.date}`.toLowerCase();
        return terms.every((t) => hay.indexOf(t) >= 0);
      }).slice(0, 12);
    },

    /* ---- scoped inline verdict chips (REVISION_WORKLIST §D item 5,
     * ped B4 steps 2–3 as adjudicated in §B-4) ----------------------
     * When a data-hypothesis-tagged CONTROL becomes active (a tagged
     * button is clicked / a tagged input is moved), the widget gains a
     * one-line chip per tagged row: ⚖ #N — <verdict>. Opt-outs:
     *   - data-hypothesis-verdict="inline" on the control or the widget
     *     ("this widget already hard-codes its verdict — no second
     *     report", the integrator's original double-report concern);
     *   - any other non-empty data-hypothesis-verdict value is used as
     *     the chip's verdict text verbatim (ch07 supplies page-scoped
     *     wording this way).
     * Tags on non-control elements INSIDE a widget (e.g. ch05's own
     * hidden verdict box) only seed the panel — they never chip, so a
     * page-local reveal is never doubled. Tags on the .widget element
     * itself chip on the first interaction with any control inside it.
     * Known limit (BUILD_REPORT gap #3): controls tagged dynamically
     * after init (ch03's #26) are armed on a best-effort re-scan when
     * the chip container is first needed — a tag set after the reader's
     * first interaction may only surface in the search panel. */
    _chipContainer(widget) {
      let box = widget.querySelector(".ledger-inline-chips");
      if (!box) {
        box = document.createElement("div");
        box.className = "ledger-inline-chips";
        box.setAttribute("role", "status");
        box.setAttribute("aria-live", "polite");
        const controls = widget.querySelector(".widget-controls");
        if (controls && controls.parentNode) {
          controls.parentNode.insertBefore(box, controls.nextSibling);
        } else {
          widget.appendChild(box);
        }
      }
      return box;
    },

    _inlineReveal(widget, ids, overrideText) {
      const box = Book.ledger._chipContainer(widget);
      const todo = ids.filter(
        (id) => !box.querySelector(`[data-ledger-chip="${id}"]`)
      );
      if (!todo.length) return;
      Book.ledger.load().then((rows) => {
        const byId = {};
        rows.forEach((r) => { byId[String(r.id)] = r; });
        todo.forEach((id, i) => {
          if (box.querySelector(`[data-ledger-chip="${id}"]`)) return;
          const r = byId[String(id)];
          if (!r) return;
          const chip = document.createElement("div");
          chip.className = "ledger-inline-chip";
          chip.setAttribute("data-ledger-chip", id);
          const head = document.createElement("strong");
          head.textContent = `⚖ #${r.id}`;
          chip.appendChild(head);
          chip.appendChild(document.createTextNode(" — "));
          const verdict = document.createElement("span");
          verdict.textContent = (i === 0 && overrideText) ? overrideText : r.verdict;
          chip.appendChild(verdict);
          if (r.do_not_retry) {
            const dnr = document.createElement("span");
            dnr.className = "ledger-dnr";
            dnr.textContent = "do-not-re-try";
            chip.appendChild(document.createTextNode(" "));
            chip.appendChild(dnr);
          }
          chip.title = `${r.hypothesis} (${r.documented || "ledger"})`;
          box.appendChild(chip);
        });
      }).catch(() => { /* ledger data unavailable — search panel says so */ });
    },

    _armInlineChips() {
      document.querySelectorAll("[data-hypothesis]").forEach((el) => {
        const ids = [];
        for (const attr of ["data-hypothesis", "data-hypothesis-2"]) {
          const v = el.getAttribute(attr);
          if (v && v !== "none" && ids.indexOf(v) < 0) ids.push(v);
        }
        if (!ids.length) return;
        const own = el.getAttribute("data-hypothesis-verdict");
        if (own === "inline") return; // hard-coded verdict: no second report
        const widget = el.closest(".widget");
        if (!widget) return;
        if (widget !== el &&
            widget.getAttribute("data-hypothesis-verdict") === "inline") return;
        const override = own && own !== "inline" ? own : null;
        const fire = () => Book.ledger._inlineReveal(widget, ids, override);
        const isControl = /^(BUTTON|INPUT|SELECT)$/.test(el.tagName);
        if (isControl) {
          el.addEventListener(el.tagName === "BUTTON" ? "click" : "input", fire);
        } else if (widget === el) {
          // widget-level tag: first interaction with any control inside
          widget.querySelectorAll("button, input, select").forEach((c) => {
            c.addEventListener(c.tagName === "BUTTON" ? "click" : "input", fire);
          });
        }
        // non-control, non-widget tags (page-local verdict boxes) only
        // seed the search panel — deliberately no chip.
      });
    },

    init() {
      if (document.getElementById("mus-search")) return; // museum has the full instrument
      const main = document.querySelector("main.book-content");
      if (!main || !document.querySelector(".widget")) return;

      // 1. the search box, placed before the provenance panel / footer
      const details = document.createElement("details");
      details.className = "ledger-ask";
      details.id = "ledger-ask";
      details.innerHTML =
        `<summary>⚖ Has this been tried? — search the 98-row defect ledger</summary>` +
        `<p class="ledger-ask-hint">Every hypothesis this project tried, with its verdict. ` +
        `Before proposing a fix in any sandbox, check whether it is already dead. ` +
        `Full table in <a href="museum.html">the Defect Museum</a>.</p>` +
        `<input type="search" class="ledger-ask-input" placeholder="e.g. truncate, photo-z, mass window, #61…" ` +
        `aria-label="Search the defect ledger" />` +
        `<div class="ledger-ask-results" aria-live="polite"></div>`;
      const anchor = main.querySelector(".provenance-panel") || main.querySelector(".book-footer");
      if (anchor) main.insertBefore(details, anchor);
      else main.appendChild(details);
      const input = details.querySelector(".ledger-ask-input");
      const out = details.querySelector(".ledger-ask-results");
      let rows = null;
      details.addEventListener("toggle", () => {
        if (details.open && rows === null) {
          Book.ledger.load().then((r) => { rows = r; }).catch(() => {
            out.innerHTML = `<p class="ledger-ask-hint">Ledger data unavailable (data/museum_ledger.json).</p>`;
          });
        }
      });
      input.addEventListener("input", () => {
        if (!rows) return;
        const hits = Book.ledger.search(rows, input.value);
        out.innerHTML = hits.length
          ? hits.map(Book.ledger._fmtRow).join("")
          : (input.value.trim() ? `<p class="ledger-ask-hint">No ledger row matches.</p>` : "");
      });

      // 2. arm the scoped inline verdict chips (§D item 5): a tagged
      // control volunteers "⚖ #N — verdict" inside its own widget the
      // moment it becomes active. Widgets that hard-code their verdict
      // opt out with data-hypothesis-verdict="inline", so nothing
      // double-reports and no predict-lock is pre-empted.
      Book.ledger._armInlineChips();

      // 3. seed the panel with the hypotheses this page's sandboxes can
      // reach (their data-hypothesis="<row#>" tags).
      const tagIds = [];
      document.querySelectorAll("[data-hypothesis]").forEach((el) => {
        for (const attr of ["data-hypothesis", "data-hypothesis-2"]) {
          const v = el.getAttribute(attr);
          if (v && v !== "none" && tagIds.indexOf(v) < 0) tagIds.push(v);
        }
      });
      if (tagIds.length) {
        const seed = document.createElement("p");
        seed.className = "ledger-ask-hint ledger-ask-tags";
        seed.innerHTML = "Dead hypotheses reachable from this page's sandboxes: " +
          tagIds.map((i) => `<button type="button" class="ledger-tag" data-row="${i}">#${i}</button>`).join(" ");
        details.insertBefore(seed, input);
        seed.querySelectorAll(".ledger-tag").forEach((b) => {
          b.addEventListener("click", () => {
            details.open = true;
            input.value = "#" + b.getAttribute("data-row") + " ";
            const fire = () => input.dispatchEvent(new Event("input"));
            if (rows) fire();
            else Book.ledger.load().then((r) => { rows = r; fire(); });
          });
        });
      }
    },
  };

  // ------------------------------------------------------------------
  // Persona switch (WIDGET_REQUESTS R-INT-3)
  // ------------------------------------------------------------------
  /* "Reading as: Curious / Methodology / All details" — progressive
   * disclosure by depth, per BOOK_PEDAGOGY.md §1.2. The three modes are the
   * book's three personas (internally still keyed mara / tomas / examiner,
   * so stored preferences and the body.persona-* CSS hooks are unchanged);
   * the visible labels name the DEPTH the reader wants rather than a person.
   * It never touches self-check answers (rubric D: answers stay hidden until
   * asked). Persisted across pages.
   *
   * Reversibility (reader feedback 2026-08-04): stepping DOWN a level now
   * re-collapses the strata this switch opened — but only those. A fold the
   * reader opened (or kept open) by hand is theirs: the toggle listener in
   * init() clears our ownership flag, so "never hides the reader's own
   * content" (§D item 9 / ped m3) still holds. */
  Book.persona = {
    STORAGE_KEY: "book-persona",
    MODES: ["mara", "tomas", "examiner"],
    LABELS: { mara: "Curious", tomas: "Methodology", examiner: "All details" },
    TITLES: {
      mara: "Curious — the main thread only; nothing pre-expanded",
      tomas: "Methodology — the 'For the GW reader' strata (code sites, derivations) open by default",
      examiner: "All details — methodology strata plus every numbers view, provenance emphasized",
    },
    _applying: false,

    current() {
      try {
        const v = localStorage.getItem(Book.persona.STORAGE_KEY);
        return Book.persona.MODES.indexOf(v) >= 0 ? v : "mara";
      } catch (e) { return "mara"; }
    },

    /** The strata each mode pre-expands (in depth order). */
    _managed() {
      return {
        gw: Array.from(document.querySelectorAll("details.gw-reader")),
        num: Array.from(document.querySelectorAll("details.num-view")),
      };
    },

    apply(mode) {
      document.body.classList.remove("persona-mara", "persona-tomas", "persona-examiner");
      document.body.classList.add("persona-" + mode);
      const { gw, num } = Book.persona._managed();
      const wanted = new Set();
      if (mode === "tomas") gw.forEach((d) => wanted.add(d));
      if (mode === "examiner") { gw.forEach((d) => wanted.add(d)); num.forEach((d) => wanted.add(d)); }

      // Symmetric switching: open what this mode wants, and re-collapse what
      // a PREVIOUS mode opened and this one does not — never a fold the
      // reader opened by hand (init() clears personaOpened on user toggles).
      Book.persona._applying = true;
      gw.concat(num).forEach((d) => {
        if (wanted.has(d)) {
          if (!d.open) {
            d.open = true;
            d.dataset.personaOpened = "1";
          }
        } else if (d.dataset.personaOpened === "1") {
          d.open = false;
          delete d.dataset.personaOpened;
        }
      });
      Book.persona._applying = false;

      document.querySelectorAll(".persona-switch button").forEach((b) => {
        b.setAttribute("aria-pressed", String(b.getAttribute("data-persona") === mode));
      });
    },

    set(mode) {
      try { localStorage.setItem(Book.persona.STORAGE_KEY, mode); } catch (e) { /* ignore */ }
      Book.persona.apply(mode);
    },

    init() {
      const topbar = document.querySelector(".book-topbar");
      if (!topbar || topbar.querySelector(".persona-switch")) return;
      const wrap = document.createElement("div");
      wrap.className = "persona-switch";
      wrap.setAttribute("role", "group");
      wrap.setAttribute("aria-label", "Reading persona");
      let html = `<span class="persona-label">Reading as</span>`;
      for (const m of Book.persona.MODES) {
        html += `<button type="button" data-persona="${m}" title="${Book.persona.TITLES[m]}">` +
          `${Book.persona.LABELS[m]}</button>`;
      }
      wrap.innerHTML = html;
      // order in the group: chapter picker | persona | glossary | theme
      const controls = topbar.querySelector(".book-controls");
      const themeBtn = topbar.querySelector("[data-theme-toggle]");
      (controls || topbar).insertBefore(wrap, themeBtn || null);
      wrap.querySelectorAll("button").forEach((b) => {
        b.addEventListener("click", () => Book.persona.set(b.getAttribute("data-persona")));
      });

      // A fold the reader touches becomes the reader's: clear our ownership
      // flag so a later mode change never closes it under them.
      const { gw, num } = Book.persona._managed();
      gw.concat(num).forEach((d) => {
        d.addEventListener("toggle", () => {
          if (!Book.persona._applying) delete d.dataset.personaOpened;
        });
      });

      Book.persona.apply(Book.persona.current());
    },
  };

  // ------------------------------------------------------------------
  // Topbar chrome: controls row + mobile menu (reader feedback 2026-08-04)
  // ------------------------------------------------------------------
  /* The topbar used to be brand + 14 chapter links + N buttons on one flex
   * row: on a phone that wrapped into a five-line wall before the chapter
   * even started. Chrome is assembled HERE rather than in the 14 chapter
   * files (which are frozen): the existing <nav class="book-nav"> and the
   * theme button are moved into a .book-controls group, and a ☰ button —
   * shown by CSS only on narrow viewports — collapses that group. Persona
   * and glossary buttons insert themselves into the same group. */
  Book.chrome = function () {
    const topbar = document.querySelector(".book-topbar");
    if (!topbar || topbar.querySelector(".book-controls")) return;

    const controls = document.createElement("div");
    controls.className = "book-controls";
    controls.id = "book-controls";

    const nav = topbar.querySelector(".book-nav");
    const themeBtn = topbar.querySelector("[data-theme-toggle]");
    if (nav) controls.appendChild(nav);
    if (themeBtn) controls.appendChild(themeBtn);
    topbar.appendChild(controls);

    const menu = document.createElement("button");
    menu.type = "button";
    menu.className = "topbar-menu";
    menu.setAttribute("data-topbar-menu", "");
    menu.setAttribute("aria-expanded", "false");
    menu.setAttribute("aria-controls", "book-controls");
    menu.setAttribute("aria-label", "Menu");
    menu.textContent = "☰";
    menu.addEventListener("click", () => {
      const open = controls.classList.toggle("is-open");
      menu.setAttribute("aria-expanded", String(open));
    });
    topbar.insertBefore(menu, controls);
  };

  // ------------------------------------------------------------------
  // Nav built from the chapter manifest (js/manifest.js)
  // ------------------------------------------------------------------
  /**
   * If window.BOOK_CHAPTERS exists and the page carries
   * <nav class="book-nav" data-nav></nav>, populate it with a chapter
   * PICKER (a native <select>): the flat 14-link list was unreadable on a
   * phone and hard to scan on a laptop, and it gave no sense of order.
   * Sequential movement is the pager at the foot of the page
   * (Book.buildPager). "planned" chapters render as disabled options.
   * Pages with a hand-written nav (no data-nav attribute) are untouched.
   */
  Book.buildNav = function () {
    const nav = document.querySelector(".book-nav[data-nav]");
    if (!nav || !global.BOOK_CHAPTERS) return;
    const here = Book.currentFile();
    nav.innerHTML = "";

    const sel = document.createElement("select");
    sel.className = "book-nav-select";
    sel.setAttribute("aria-label", "Jump to a chapter");
    for (const ch of global.BOOK_CHAPTERS) {
      const opt = document.createElement("option");
      opt.value = ch.file;
      opt.textContent = Book.chapterLabel(ch) +
        (ch.status === "live" ? "" : "  (planned)");
      if (ch.status !== "live") opt.disabled = true;
      if (ch.file === here) opt.selected = true;
      sel.appendChild(opt);
    }
    if (!global.BOOK_CHAPTERS.some((c) => c.file === here)) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "Jump to…";
      opt.selected = true;
      sel.insertBefore(opt, sel.firstChild);
    }
    sel.addEventListener("change", () => {
      if (sel.value && sel.value !== here) global.location.href = sel.value;
    });
    nav.appendChild(sel);
  };

  /** The current page's file name ("" — a directory URL — means index). */
  Book.currentFile = function () {
    return global.location.pathname.split("/").pop() || "index.html";
  };

  /** "Ch. 7 — A Redshift Is Not a Number" (no duplication for Contents). */
  Book.chapterLabel = function (ch) {
    return ch.short === ch.title ? ch.title : ch.short + " — " + ch.title;
  };

  // ------------------------------------------------------------------
  // Foot-of-page pager (reader feedback 2026-08-04)
  // ------------------------------------------------------------------
  /**
   * Previous / next chapter buttons appended to main.book-content (before
   * the .book-footer note if there is one). Order and titles come from
   * window.BOOK_CHAPTERS, so a chapter never hardcodes its neighbours —
   * same contract as the nav. Only "live" chapters take part.
   */
  Book.buildPager = function () {
    const main = document.querySelector("main.book-content");
    if (!main || !global.BOOK_CHAPTERS) return;
    if (main.querySelector(".book-pager")) return;
    const live = global.BOOK_CHAPTERS.filter((c) => c.status === "live");
    const here = Book.currentFile();
    const i = live.findIndex((c) => c.file === here);
    if (i < 0) return; // not a book page (template, stray file) — no pager

    const cell = (ch, dir) => {
      if (!ch) return `<span class="book-pager-slot"></span>`;
      const arrow = dir === "prev" ? "← Previous" : "Next →";
      return `<a class="book-pager-slot book-pager-${dir}" href="${ch.file}" ` +
        `rel="${dir}"><span class="book-pager-dir">${arrow}</span>` +
        `<span class="book-pager-title">${Book.chapterLabel(ch)}</span></a>`;
    };

    const pager = document.createElement("nav");
    pager.className = "book-pager";
    pager.setAttribute("aria-label", "Chapter navigation");
    pager.innerHTML =
      cell(live[i - 1], "prev") +
      (here === "index.html"
        ? `<span class="book-pager-slot"></span>`
        : `<a class="book-pager-slot book-pager-toc" href="index.html">` +
          `<span class="book-pager-dir">Contents</span>` +
          `<span class="book-pager-title">All chapters</span></a>`) +
      cell(live[i + 1], "next");

    const footer = main.querySelector(".book-footer");
    if (footer) main.insertBefore(pager, footer);
    else main.appendChild(pager);
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
    // chrome() first: persona + glossary insert themselves into the
    // .book-controls group it creates.
    try { Book.chrome(); } catch (e) { console.warn("book.js: chrome init failed", e); }
    Book.buildNav();
    Book.markCurrentNav();
    try { Book.buildPager(); } catch (e) { console.warn("book.js: pager init failed", e); }
    Book.renderMath(document.body);
    // Phase-4 instruments — each guarded so a failure can never take down
    // theme/nav/math on a page.
    try { Book.persona.init(); } catch (e) { console.warn("book.js: persona init failed", e); }
    try { Book.passport.init(); } catch (e) { console.warn("book.js: passport init failed", e); }
    try { Book.ledger.init(); } catch (e) { console.warn("book.js: ledger init failed", e); }
  });

  global.Book = Book;
})(window);
