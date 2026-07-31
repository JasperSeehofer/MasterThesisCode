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
 *                             `.term[data-term]`, pin to a personal glossary
 *   Book.ledger             — "Has this been tried?" (BW3): per-page search over
 *                             data/museum_ledger.json + verdict hints for
 *                             sandbox states tagged data-hypothesis="<row#>"
 *   Book.persona            — global "Reading as: Mara / Tomas / Examiner"
 *                             switch (pre-expands strata; never hides content)
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
   * hover/tap card and a pin-to-glossary control to every one. */
  Book.SYMBOLS = {
    h:      { sym: "$h$", meaning: "H₀ / 100 km s⁻¹ Mpc⁻¹; mock truth h_true = 0.73", units: "—", src: "constants.py H" },
    H0:     { sym: "$H_0$", meaning: "Hubble constant", units: "km s⁻¹ Mpc⁻¹", src: "constants.py" },
    dL:     { sym: "$d_L$", meaning: "luminosity distance", units: "Mpc (Gpc for pools)", src: "physical_relations.py:132 dist" },
    z:      { sym: "$z$", meaning: "true redshift", units: "—", src: "dark_siren_likelihood.md §2.4" },
    zg:     { sym: "$z_g$", meaning: "catalogue (observed) redshift of galaxy g", units: "—", src: "handler.py; K1" },
    sigz:   { sym: "$\\sigma_z$", meaning: "host-z kernel width (total: measurement ⊕ peculiar velocity)", units: "—", src: "hostz_pv_photoz_kernel.md" },
    eps:    { sym: "$\\sigma_z/z$", meaning: "fractional z width — the C7 variable; rail threshold quoted at 0.256", units: "—", src: "C7_README", note: "C7 is a live FINDING — see Ch 7 §6 / Ch 11" },
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
    Sig:    { sym: "$\\Sigma$", meaning: "Cramér–Rao covariance Γ⁻¹", units: "mixed", src: "parameter_estimation.py:430" },
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
    wG:     { sym: "$w_G$", meaning: "mixture weight β_G/D — ESTIMAND-DEPENDENT; always name the mode", units: "—", src: "bayesian_statistics.py:3309-3311; C9", note: "C9 is a live FINDING — see Ch 9 §6 / Ch 11" },
    pdet:   { sym: "$p_{\\rm det}$", meaning: "detection probability (horizon-survival estimator)", units: "—", src: "simulation_detection_probability.py" },
    fz:     { sym: "$f(z,\\Omega)$", meaning: "catalogue completeness fraction", units: "—", src: "G2c D2; pixel_completeness.py" },
    pi:     { sym: "$p_i(h)$", meaning: "per-event likelihood (the master equation)", units: "—", src: "bayesian_statistics.py:3006-3009, 1042-1048" },
    Cscale: { sym: "$C$", meaning: "the arbitrary mass-coordinate rescale of the C8 walk", units: "—", src: "README_C8.md", note: "C8 is a live FINDING — see Ch 8 §6 / Ch 11" },
    sigh:   { sym: "$\\sigma_h$", meaning: "posterior width in h", units: "—", src: "readouts" },
  };

  Book.passport = {
    GLOSSARY_KEY: "book-glossary",
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

    _card(key) {
      const s = Book.SYMBOLS[key];
      if (!s) return null;
      const pinned = Book.passport._pinned().indexOf(key) >= 0;
      const note = s.note ? `<div class="passport-note">${s.note}</div>` : "";
      return (
        `<div class="passport-sym">${s.sym}</div>` +
        `<div class="passport-meaning">${s.meaning}</div>` +
        `<div class="passport-row"><span>units</span> ${s.units}</div>` +
        `<div class="passport-row"><span>defined by</span> <code>${s.src}</code></div>` +
        note +
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
        let html = `<div class="passport-glossary-title">My glossary</div>`;
        for (const k of list) {
          const s = Book.SYMBOLS[k];
          if (!s) continue;
          html += `<div class="passport-glossary-item"><strong>${s.sym}</strong> ` +
            `${s.meaning} <em>[${s.units}]</em> <code>${s.src}</code></div>`;
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
        btn.addEventListener("click", () => {
          const open = panel.style.display !== "none";
          panel.style.display = open ? "none" : "block";
          btn.setAttribute("aria-expanded", String(!open));
          if (!open) Book.passport._renderGlossary();
        });
        const themeBtn = topbar.querySelector("[data-theme-toggle]");
        topbar.insertBefore(btn, themeBtn);
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
      return (
        `<div class="ledger-result"><strong>#${r.id}</strong> ` +
        `<span class="ledger-date">${r.date || r.era || ""}</span> — ` +
        `${r.hypothesis} → <em>${r.verdict}</em>${dnr} ` +
        `<span class="prov-chip">${r.documented || ""}</span></div>`
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

      // 2. seed the panel with the hypotheses this page's sandboxes can
      // reach (their data-hypothesis="<row#>" tags). The IN-WIDGET verdict
      // reveals themselves stay page-local by design: the museum meta-rule
      // required every chapter to hard-code its own reveal, so injecting a
      // second one here would double-reveal (and pre-empt predict-locks).
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
  /* "Reading as: Mara / Tomas / Examiner" — progressive disclosure by reader,
   * per BOOK_PEDAGOGY.md §1.2. It only PRE-EXPANDS strata, never hides
   * content, and never touches self-check answers (rubric D: answers stay
   * hidden until asked). Persisted across pages. */
  Book.persona = {
    STORAGE_KEY: "book-persona",
    MODES: ["mara", "tomas", "examiner"],
    LABELS: { mara: "Mara", tomas: "Tomas", examiner: "Examiner" },
    TITLES: {
      mara: "Curious physicist: nothing pre-expanded",
      tomas: "GW reader: 'For the GW reader' strata open by default",
      examiner: "Examiner: GW strata + numbers views open, provenance emphasized",
    },

    current() {
      try {
        const v = localStorage.getItem(Book.persona.STORAGE_KEY);
        return Book.persona.MODES.indexOf(v) >= 0 ? v : "mara";
      } catch (e) { return "mara"; }
    },

    apply(mode) {
      document.body.classList.remove("persona-mara", "persona-tomas", "persona-examiner");
      document.body.classList.add("persona-" + mode);
      const gw = document.querySelectorAll("details.gw-reader");
      const num = document.querySelectorAll("details.num-view");
      if (mode === "mara") {
        gw.forEach((d) => { d.open = false; });
        num.forEach((d) => { d.open = false; });
      } else if (mode === "tomas") {
        gw.forEach((d) => { d.open = true; });
        num.forEach((d) => { d.open = false; });
      } else {
        gw.forEach((d) => { d.open = true; });
        num.forEach((d) => { d.open = true; });
      }
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
      const themeBtn = topbar.querySelector("[data-theme-toggle]");
      topbar.insertBefore(wrap, themeBtn);
      wrap.querySelectorAll("button").forEach((b) => {
        b.addEventListener("click", () => Book.persona.set(b.getAttribute("data-persona")));
      });
      Book.persona.apply(Book.persona.current());
    },
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
    // Phase-4 instruments — each guarded so a failure can never take down
    // theme/nav/math on a page.
    try { Book.persona.init(); } catch (e) { console.warn("book.js: persona init failed", e); }
    try { Book.passport.init(); } catch (e) { console.warn("book.js: passport init failed", e); }
    try { Book.ledger.init(); } catch (e) { console.warn("book.js: ledger init failed", e); }
  });

  global.Book = Book;
})(window);
