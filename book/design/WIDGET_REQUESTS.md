# WIDGET_REQUESTS.md — shared-capability request queue

**Rule.** `book/site/js/book.js`, `book/site/js/manifest.js`, `book/site/css/book.css`,
`book/site/_template.html`, `book/site/index.html`, `book/generators/make_all.py`, and
`.github/workflows/ci.yml` are **FROZEN** for chapter agents. If your chapter needs a
capability those files do not provide, **append a request block here** and implement a
**page-local workaround** in your own chapter file in the meantime (an inline `<script>`
in your `chNN-*.html` is fine — it is your file). The integrator triages this queue,
implements accepted requests in the shared files, and replaces workarounds.

Do NOT block your chapter on a request. Do NOT edit shared files "just this once".

## Request format (append below the line)

```
### R-<chapter>-<n>: <one-line capability name>
- Requested by: ch<NN> agent, <date>
- Need: <what the widget must do that book.js cannot>
- Current workaround: <inline in chNN-*.html | none — degraded to static>
- Proposed API: <sketch, optional>
- Status: OPEN            <- integrator sets: ACCEPTED / IMPLEMENTED / REJECTED (reason)>
```

---

## Pre-approved backlog (integrator phase 4 — do not re-request)

### R-INT-1: Symbol Passport (BW2, full version)
- Hover/tap any `<span class="term" data-term="w_G">` for definition, units, code site,
  ratifying derivation, status badge; click pins to a personal glossary.
  Chapter agents SHOULD already mark symbols with `class="term" data-term="<key>"`
  (keys = the notation table in BOOK_DESIGN.md §4.1) so the passport can attach later.
  Until then the markup is inert — no workaround needed.
- Status: IMPLEMENTED (integrator, 2026-07-31) — `Book.passport` in book.js: hover/tap card on every `.term[data-term]` (definition, units, defining source, caveat note from the BOOK_DESIGN §3.1 table), pin-to-glossary persisted in localStorage, ★ Glossary panel injected into the topbar. Chapter markup was already in place (127 tagged terms, all keys valid).

### R-INT-2: "Has this been tried?" ledger search (BW3)
- A search box over the 98-row `BIAS_HISTORY_LEDGER.md` available inside sandboxes;
  volunteers the verdict when a sandbox configuration matches a historical hypothesis.
  Needs `data/museum_ledger.json` (owned by the museum agent). Chapter sandboxes SHOULD
  tag their toggle states with `data-hypothesis="<ledger row #>"` where a state matches
  a known dead hypothesis, and hard-code the reveal of that verdict locally (museum
  meta-rule: an interactive that lets the reader "try" a dead hypothesis must reveal the
  measured verdict, not leave it open).
- Status: IMPLEMENTED (integrator, 2026-07-31) — `Book.ledger` in book.js: a per-page collapsible search box over `data/museum_ledger.json` injected before the provenance panel on every chapter page with widgets (museum's own #mus-search takes precedence), plus verdict hints: sandbox controls tagged `data-hypothesis="<row#>"` volunteer the ledger row's verdict on activation; widgets that hard-code their own reveal via `data-hypothesis-verdict` are left alone.

### R-INT-3: Persona switch (Reading as Mara / Tomas / Examiner)
- Global control pre-expanding `details.gw-reader` and provenance panels. Pure chrome;
  chapters need only use the standard `gw-reader` / `provenance-panel` classes.
- Status: IMPLEMENTED (integrator, 2026-07-31) — `Book.persona` in book.js: 'Reading as' segmented control injected into every topbar; Tomas pre-opens `details.gw-reader`, Examiner additionally pre-opens `details.num-view` and emphasizes provenance (body class + CSS). Never touches self-check answers; persisted across pages.

---

<!-- Chapter-agent requests go below this line. -->

### R-ch07-1: bias-rail "pips" — amber/grey annotations with no bias number
- Requested by: ch07 agent, 2026-07-31
- Need: `BOOK_DESIGN.md` §1 asks several chapters to add rail entries that are
  *not* a bias value — Ch 5's "two branches disagree (0.86 / 0.64) — unresolved",
  Ch 7's "C7: inflates at σ_z/z > 0.256 (live)", Ch 9's "w_G ≠ realized rate
  (z = −11.86) — C9", Ch 11's three amber pips + one grey pip. `Book.biasRail`
  currently renders `bias: null` as the literal string "not defined yet", which is
  the wrong semantics for an *unresolved-but-measured* annotation, and it offers no
  amber/grey severity channel.
- Current workaround: inline in `ch07-redshift.html` — after each `Book.biasRail()`
  call the page appends its own `.ch07-pip` nodes into `#bias-rail` with inline
  styles (biasRail replaces `innerHTML`, so the append must follow every call).
- Proposed API: `Book.biasRail({ entries: [...], pips: [{ label, tone: "amber"|"grey",
  note }] })` — rendered under the entries, no track, no numeric column; `tone` maps
  to the existing `--planck` / `--prior` CSS variables so no new palette is needed.
- Status: IMPLEMENTED (integrator, 2026-07-31) — `Book.biasRail({ entries, pips: [{label, tone: "amber"|"grey", note}] })` renders a 'live, unquantified' section; shared CSS `.bias-rail-pip`. Page-local workarounds in ch05/ch07/ch09/ch11 replaced with the shared API. Note: amber standardized to #D55E00 (the rail's active-marker orange; ch07's draft used --planck pink).

### R-ch07-2: `Book.interp1` — linear interpolation onto a tabulated grid
- Requested by: ch07 agent, 2026-07-31
- Need: chapters that ship a tabulated function from the Python side (here
  `w_pop(z)` and `f(z) = d_L(z; h=1)` from `physical_relations`) and let the reader
  drive a live integral over it all need the same monotone-grid linear
  interpolation. `Book.lerp` interpolates between two scalars only.
- Current workaround: inline `interp1(xs, ys, x)` in `ch07-redshift.html`.
- Proposed API: `Book.interp1(xs, ys, x)` (scalar or array `x`, binary search,
  clamped at the ends) — pairs naturally with the existing `Book.trapz`.
- Status: IMPLEMENTED (integrator, 2026-07-31) — `Book.interp1(xs, ys, x)` (scalar or array, binary search, clamped). ch07's inline copy left in place deliberately (works, zero coupling beats deduplication per BOOK_DESIGN §2); new chapters should use the shared helper.

### R-ch04-1: numeric predict-marker (predict-then-reveal with a value, not a choice)
- Requested by: ch04 agent, 2026-07-31
- Need: `BOOK_DESIGN.md` §1 asks several chapters for a *marker* prediction rather than a
  multiple choice — Ch 4's I4.1 ("the reader must first drag a 'where will the MAP go?'
  marker"), Ch 3's I3.1 place-your-marker host guess, Ch 10's I10.1. `Book.predictReveal`
  only records `data-predict` string choices from buttons, so a continuous guess has to be
  smuggled into that attribute, and the localStorage restore path
  (`querySelector('[data-predict="<saved>"]')`) then depends on the page having already
  re-written the attribute before `predictReveal` is called.
- Current workaround: inline in `ch04-loud-half.html` — a `<input type="range">` whose
  `input` handler rewrites the single lock button's `data-predict` attribute, plus an
  explicit `Book.getPrediction()` read *before* `Book.predictReveal()` to restore the
  slider and pre-set the attribute so the restore path matches. Also note: the frozen
  `_template.html` puts `data-predict-id` on the `.predict-row` but passes
  `.closest(".widget")` to `predictReveal`, which reads the attribute off the *container* —
  so persistence silently no-ops unless the id is also on the `.widget`. This page sets it
  on both.
- Proposed API: `Book.predictValue({ slider, button, id, onLock })` — persists a number
  under `book-predict:<id>`, restores the slider on load, fires `onLock(value)` on both the
  click and the restore, and reveals the sibling `.reveal`. Plus: have `predictReveal` look
  for `data-predict-id` on the container **or** on the `.predict-row` inside it.
- Status: IMPLEMENTED (integrator, 2026-07-31) — `Book.predictValue({slider, button, id, onLock, format})` added, and `Book.predictReveal` now resolves `data-predict-id` on the container OR any descendant (fixing the template's silent-no-op persistence quirk for every page). ch04's inline wiring left in place (works; benefits from the id-resolution fix automatically).

### R-ch07-3: `<!DOCTYPE html>` + `<meta charset>` are missing from every shipped page
- Requested by: ch07 agent, 2026-07-31
- Need: **not a widget — a live rendering bug in frozen files.** `_template.html`,
  `index.html` and `ch00-demo.html` all begin with `<title>`, with no doctype. Browsers
  therefore render every book page in **quirks mode**, and KaTeX hard-refuses to run in
  quirks mode. Verified headless (Chromium 141, served over HTTP):
  `ch00-demo.html` emits 27 × `ParseError: KaTeX parse error: KaTeX doesn't work in
  quirks mode` and every equation on the page stays as literal `$...$` source. Adding
  `<!DOCTYPE html>` as the first line takes that count to 0. Separately, no page declares
  a charset, so any server that does not append `; charset=utf-8` renders σ, ⟨z⟩, ←, ‑
  as mojibake.
- Current workaround: `ch07-redshift.html` carries its own `<!DOCTYPE html>` +
  `<meta charset="utf-8" />` as its first two lines (with an explanatory comment pointing
  here). Every other page is currently affected.
- Proposed fix (integrator, frozen files): prepend those two lines to `_template.html`,
  `index.html`, and the legacy demo — 2 lines each, no other change. Worth doing before
  wave 2 copies the template.
- Status: IMPLEMENTED (integrator, 2026-07-31) — doctype + charset prepended to `_template.html` and the rewritten `index.html`, and added to the four chapter pages that lacked them (ch01, ch04, ch08, ch09). The legacy demo page was retired. All 14 shipped pages now open with `<!DOCTYPE html>` + `<meta charset="utf-8">`.

### R-ch11-1: bias rail rows without a bias number ("amber pips" for live open defects)
- Requested by: ch11 agent, 2026-07-31
- Need: `BOOK_DESIGN.md` §1 requires several chapters to put *unquantified* entries on the
  rail — Ch 5's "two branches disagree (0.86 / 0.64) — unresolved", Ch 7's
  "C7: inflates at σ_z/z > 0.256 (live)", Ch 9's "w_G ≠ realized rate (z = −11.86) — C9",
  and Ch 11's full honest state (three amber pips C7/C8/C9 + one grey pip "C6: attribution
  confounded — cell B in flight"). `Book.biasRail` renders exactly two row shapes:
  a numeric marker on the [−0.18, +0.08] track, or the literal text "not defined yet".
  There is no way to say *"this is live, it has no bias number, and that is the point"* —
  which is the whole honesty contract of the closing chapter.
- Current workaround: inline in `ch11-honest-state.html` — `renderRail()` calls
  `Book.biasRail(...)` first, then appends its own `.ch11-pip` rows into `#bias-rail`
  (the host element is public and `biasRail` replaces its `innerHTML`, so the pips are
  re-appended on every re-render). Page-local CSS only; no shared class is restyled.
- Proposed API: allow `bias: null` **with** an optional `state: "amber" | "grey" | "open"`
  and `pip: true`, rendering a coloured dot + label + note instead of the track, e.g.
  `{ label: "C7 — kernel omits selection", bias: null, pip: "amber", note: "live, FINDING" }`.
  Keep the current `bias: null` → "not defined yet" behaviour when `pip` is absent.
- Status: IMPLEMENTED (integrator, 2026-07-31) — same capability as R-ch07-1 (implemented once, as `spec.pips` with `tone`); ch11's page-local append workaround replaced.

### R-ch11-2: a published registry of canonical `data-predict-id`s (cross-chapter recall)
- Requested by: ch11 agent, 2026-07-31
- Need: `BOOK_DESIGN.md` §1 Ch 11 requires the closing chapter to re-surface **the reader's
  Chapter 3 host-guess marker** via `Book.getPrediction`. `getPrediction(id)` exists and
  works, but the *id* is invented by whichever chapter wrote it, and Ch 3 is built in a
  later wave — so Ch 11 cannot know the string. Any mismatch fails silently (returns
  `null`), which is the worst failure mode for a payoff beat: it degrades to nothing with
  no error anywhere.
- Current workaround: inline in `ch11-honest-state.html` — the recall block probes a small
  list of plausible ids (`ch03-host-guess`, `ch03-host`, `ch03-marker`, `ch03-guess`,
  plus the already-shipped `ch04-map-guess`) and renders an honest "you have not made this
  prediction yet — it is recorded when you take Chapter 3's guess" state when all are
  `null`. The block never fabricates a remembered guess.
- Proposed API: a short table in `BOOK_DESIGN.md` or `manifest.js` — e.g.
  `window.BOOK_PREDICT_IDS = { ch03HostGuess: "ch03-host-guess", ch04MapGuess: "ch04-map-guess", ... }`
  — so recall beats resolve by name rather than by guessing, and so the integrator's
  cross-chapter link check can verify that every id a chapter *reads* is written somewhere.
  Ch 3 / Ch 10 agents: `ch03-host-guess` is the string Ch 11 probes first.
- Status: IMPLEMENTED (integrator, 2026-07-31) — `window.BOOK_PREDICT_IDS` in js/manifest.js, populated from the shipped chapters' actual ids (19 entries). Ch 3 writes `ch03-host-guess`, exactly the string Ch 11 probes first, so the payoff beat resolves.

### R-ch05-1: predict-gating a *different* widget (cross-widget reveal lock)
- Requested by: ch05 agent, 2026-07-31
- Need: the pedagogy's "predict-then-reveal is locked, not suggested" (Part 4.2 rule 2) is
  strongest when the *interactive itself* stays inert until the reader commits — not just a
  `.reveal` paragraph inside the same block. Ch 5 does this twice (I5.3 gated on the "how big
  is peak-vs-integrate?" prediction, I5.2 gated on the "do the two classes agree?"
  prediction); Ch 4's I4.1 and Ch 10's I10.1 want the same shape. `Book.predictReveal` only
  toggles the `.reveal` sibling inside the container it is given.
- Current workaround: inline in `ch05-unseen-galaxy.html` — a page-local `.ch05-locked`
  class (`opacity: .45; pointer-events: none`) on the target widget, removed by the
  `onPredict` callback, plus an explicit `Book.getPrediction(id)` check on load so a
  returning reader is not re-locked. Also needs a `<noscript><style>` block to unlock
  everything and force `.reveal { display: block }`, since with JS off the lock can never be
  released and the reader would otherwise lose both the widget and the answer.
- Proposed API: `Book.predictReveal(container, onPredict, { gates: [el|selector, ...] })` —
  adds/removes a shared `is-predict-locked` class on each gate, restores from localStorage,
  and ships the no-JS unlock in `book.css` (`@media` / `noscript` equivalent) so every
  chapter gets the same accessible fallback instead of hand-rolling one.
- Note: this chapter also uses the already-open **R-ch07-1** (bias-rail pips) and
  **R-ch07-3** (`<!DOCTYPE html>` + `<meta charset>`) workarounds verbatim; both are
  re-confirmed as needed here, not re-requested.
- Status: IMPLEMENTED (integrator, 2026-07-31) — `Book.predictReveal(container, onPredict, { gates: [...] })` adds/removes a shared `.is-predict-locked` class (added only ever by JS, so no-JS readers are never locked out — book.css). ch05's page-local `.ch05-locked` mechanism left in place (works, including its noscript fallback); new chapters should use `gates`.

### R-ch11-3: Plotly 3.x dropped the string form of `axis.title` — every chapter is losing its axis labels
- Requested by: ch11 agent, 2026-07-31
- Need: **not a widget — a silent rendering regression across chapters.** The vendored build
  is **Plotly 3.4.0** (`Plotly.version`), and Plotly 3.x removed support for
  `layout.xaxis.title = "some string"`. It is not an error and nothing is logged: the axis
  title is simply **not drawn**. Verified headless — with `xaxis: { title: "h = H₀ / 100" }`
  the plot contains **0** `.xtitle`/`.ytitle` nodes; with
  `xaxis: { title: { text: "h = H₀ / 100" } }` it contains 2. `ch04-loud-half.html` and any
  chapter that copies its `baseLayout()` helper (which uses the string form) currently
  ship plots with **no axis labels at all**, which is an accessibility and comprehension
  problem, not a cosmetic one.
- Current workaround: `ch11-honest-state.html` uses the object form throughout, with a
  comment pointing here; margins widened to `l: 68, b: 52` to fit the restored titles.
- Proposed fix (integrator): a one-line grep across `book/site/ch*.html` for
  `title: <string>` inside an axis object, and either fix each chapter's `baseLayout()` or
  add a shared `Book.axis(text)` helper to `book.js` so the form is set in one place.
- Status: IMPLEMENTED (integrator, 2026-07-31) — central fix in book.js: `Plotly.newPlot/react/relayout` are wrapped to normalize any string `title` in the layout tree to `{ text: ... }` (object-form layouts pass through untouched), plus a `Book.axis(text)` helper. This repairs the string-form call sites still present in ch06/ch07/ch09 and all baseLayout() helpers without editing every call site.

### R-ch06-1: narrow-viewport horizontal overflow is a shared-CSS condition, not a per-chapter one
- Requested by: ch06 agent, 2026-07-31
- Need: **not a widget — a layout bug affecting every page.** Measured headless
  (Chromium 141, served over HTTP, `Emulation.setDeviceMetricsOverride` at 390×900):
  `document.documentElement.scrollWidth / clientWidth` reads **696 / 390** on ch06 before
  the fix below, **620 / 390** on `ch04-loud-half.html`, **515 / 390** on
  `ch07-redshift.html`. The page body scrolls sideways on a phone. Three causes, all in
  shared CSS or shared markup conventions: (a) `.prov-chip { white-space: nowrap }` with
  citation chips like `Cutler & Flanagan 1994, arXiv:gr-qc/9402014 Eq. 2.4` (387 px wide
  on its own); (b) long artifact paths inside `<code>` in `.provenance-panel` (675 px);
  (c) KaTeX `$$…$$` display blocks, which have no `overflow-x` and so widen their parent
  (the 3×3 covariance matrix is 481 px).
- Current workaround: page-local CSS in `ch06-black-box.html` — `.book-content
  .katex-display { overflow-x: auto }` unconditionally, plus a `@media (max-width: 720px)`
  block relaxing `white-space` on `.prov-chip` and `overflow-wrap` on `code` /
  `.provenance-panel li`. Verified: 390 / 390 at both 390 px and 360 px, unchanged at
  768 px and 1440 px. The only element still wider than the viewport is `.table-scroll`,
  which is the intended scroll container.
- Proposed fix (integrator, `css/book.css`, ~6 lines, benefits every chapter):
  `.katex-display { overflow-x: auto; overflow-y: hidden; }` unconditionally, and inside a
  `@media (max-width: 720px)` block `.prov-chip { white-space: normal }` +
  `code, .provenance-panel li { overflow-wrap: anywhere }`.
- Status: IMPLEMENTED (integrator, 2026-07-31) — `.katex-display { overflow-x: auto }` was already in book.css; the `@media (max-width: 720px)` rules (`.prov-chip { white-space: normal }`, `code, .provenance-panel li { overflow-wrap: anywhere }`) added. ch06's page-local copy is now redundant but harmless and was left in place.
