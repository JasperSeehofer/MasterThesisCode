# BUILD_REPORT.md — integrator pass, 2026-07-31

Integration report for the interactive dark-siren foundations book
(`book/` in the `MasterThesisCode-book` worktree, branch `book/foundations-interactive`).
All 13 chapter/museum agents delivered; this pass implemented the shared capabilities,
wired the front door and nav, ran the full QA sweep, and consolidated the flag files.
**Nothing numeric was reconciled by the integrator** — every dispute the chapters raised
is carried forward in §5, unresolved, exactly as the flag files state it.

---

## 1. Chapter inventory

### Pages (`book/site/`)

| page | bytes | status |
|---|---:|---|
| `index.html` (front door, rewritten this pass) | 10,930 | live |
| `ch00-two-numbers.html` | 59,060 | live |
| `ch01-ruler.html` | 71,679 | live |
| `ch02-bayes.html` | 91,514 | live |
| `ch03-which-galaxy.html` | 88,156 | live |
| `ch04-loud-half.html` | 68,645 | live |
| `ch05-unseen-galaxy.html` | 102,366 | live |
| `ch06-black-box.html` | 104,599 | live |
| `ch07-redshift.html` | 109,096 | live |
| `ch08-mass-channel.html` | 118,183 | live |
| `ch09-universe-factory.html` | 101,892 | live |
| `ch10-calibration.html` | 92,833 | live |
| `ch11-honest-state.html` | 113,292 | live |
| `museum.html` | 112,390 | live |
| `_template.html` | 8,205 | reference (doctype/charset added) |
| `ch00-demo.html` | — | **RETIRED** (deleted, with `gen_ch00_demo.py` + `data/ch00_demo.json`) |

Shared: `js/book.js` 44,568 · `js/manifest.js` 3,612 · `css/book.css` 22,171 ·
`vendor/` 6,226,747 (Plotly 3.4.0 + KaTeX 0.16.11, unchanged).

### Data (`book/site/data/`) — 36 files, **1,362,744 bytes total (~1.30 MiB)**

Largest: `ch03_skyball.json` 287,637 · `ch06_fisher.json` 216,694 · `ch02_stacker.json`
108,761. **Every file is under the 500 KB per-file budget** (max 288 KB). No degrade-to-static
fallback had to be exercised for size.

### Generators (`book/generators/`) — full `make_all.py` run, 2026-07-31, exit 0

| generator | runtime | notes |
|---|---:|---|
| gen_ch00 | 0.01 s | closed-form; two-number tension + arbitration toy |
| gen_ch01 | 0.37 s | G7 row-6 hard gate PASS (all six redshifts) |
| gen_ch02 | 0.32 s | 46% metric pinned to `score_realistic.py` curvature |
| gen_ch03 | 12.61 s | 20.8 M-row catalogue census; host-frame gate 2.96e-10 |
| gen_ch04 | 1.63 s | readout gate PASS (MAP 0.740 / mean 0.7321, to the digit) |
| gen_ch05 | 1.33 s | identity re-verified over 65,108 event×h cells |
| gen_ch06 | 7.32 s | real CRB ellipses + dt² horizon histograms |
| gen_ch07 | 0.77 s | G1–G4 gates PASS (worst 7.2e-16 on the tilt identity) |
| gen_ch08 | 1.24 s | 43 numeric gates + 8 structural guards, all PASS |
| gen_ch09 | 0.82 s | all C9/G1/derail gates PASS on first run |
| gen_ch10 | 61.07 s | 16 pp-harness cells × 3 truths re-run, bit-equal to archives |
| gen_ch11 | 0.38 s | 191 checks, all_pass=True |
| gen_museum | 15.81 s | ledger 98 rows; FINDING.md tables reproduced to 4 dp |
| **total** | **~104 s** | 13/13 green |

**Integrator fix to `make_all.py`:** generators now run in **subprocesses** instead of one
shared interpreter. The shared-process driver broke `gen_ch03`: earlier generators bound
`sys.modules["master_thesis_code"]` to this worktree's older package, while gen_ch03 needs
the sibling checkout's newer `handler._mass_redshift_prune_mask`. Isolation restores the
documented "independently re-runnable" contract. (First run failed exactly this way;
post-fix run is 13/13 green.)

---

## 2. Shared-capability work (WIDGET_REQUESTS ledger)

All three pre-approved instruments plus **nine** wave-filed requests were implemented in the
shared files. Statuses are recorded in `book/design/WIDGET_REQUESTS.md`; summary:

| request | disposition |
|---|---|
| **R-INT-1 Symbol Passport (BW2)** | IMPLEMENTED — `Book.passport`: hover/tap card on every `.term[data-term]` (definition, units, defining source, caveat notes for live findings) from the BOOK_DESIGN §3.1 table transcribed into `Book.SYMBOLS`; pin-to-glossary (localStorage) + ★ Glossary topbar panel. 127 tagged terms across the chapters; **all 37 distinct keys used resolve** against the table (verified programmatically). |
| **R-INT-2 "Has this been tried?" (BW3)** | IMPLEMENTED — `Book.ledger`: collapsible search box over `data/museum_ledger.json` (98 rows) injected before the provenance panel on every chapter page with widgets; museum's own full-table `#mus-search` takes precedence. Sandboxes' `data-hypothesis="<row#>"` tags seed the panel with "dead hypotheses reachable from this page" chips. Per-widget auto-reveal was deliberately **not** injected: every chapter already hard-codes its verdict reveal per the pre-approved instruction, and a second reveal would double-report and pre-empt predict-locks. |
| **R-INT-3 Persona switch** | IMPLEMENTED — `Book.persona`: "Reading as Mara / Tomas / Examiner" segmented control injected into every topbar. Tomas pre-opens `details.gw-reader`; Examiner additionally pre-opens `details.num-view` and emphasizes provenance chrome. Never touches self-check answers; persisted across pages. |
| R-ch07-1 / R-ch11-1 (rail pips) | IMPLEMENTED — `Book.biasRail({entries, pips:[{label, tone:"amber"\|"grey", note}]})` + shared `.bias-rail-pip` CSS. The four page-local workarounds (ch05, ch07, ch09, ch11) were **replaced** with the shared API. Amber standardized to `#D55E00`. |
| R-ch07-2 (`Book.interp1`) | IMPLEMENTED (binary search, clamped, scalar/array). ch07's inline copy left (works; spec §2: zero coupling beats deduplication). |
| R-ch04-1 (numeric predict-marker) | IMPLEMENTED — `Book.predictValue({slider, button, id, onLock, format})`, plus the `predictReveal` id-resolution fix: `data-predict-id` is now found on the container **or** any descendant, repairing the template's silent-no-op persistence for every page that copied the template pattern (ch07-q1/q2, ch09-q-bench/q-identity now persist; ch07 already anticipated this in its restore handling). |
| R-ch07-3 (doctype/charset) | IMPLEMENTED — `<!DOCTYPE html>` + `<meta charset="utf-8">` in `_template.html`, the rewritten `index.html`, and added to the four pages that lacked them (ch01, ch04, ch08, ch09). **All 14 shipped pages + template now open in standards mode** (KaTeX refuses quirks mode). |
| R-ch11-2 (predict-id registry) | IMPLEMENTED — `window.BOOK_PREDICT_IDS` in `manifest.js`, 19 entries from the shipped chapters' actual ids. Ch 3 writes `ch03-host-guess`, exactly the string Ch 11 probes first: the payoff beat resolves. |
| R-ch11-3 (Plotly 3.x title regression) | IMPLEMENTED **centrally**: `Plotly.newPlot/react/relayout` are wrapped in book.js to normalize any string `title` in the layout tree to `{text:…}` (object form passes through; flattened `"xaxis.title"` keys handled; direct `Plotly.relayout` calls in ch03/ch06 covered). Plus `Book.axis(text)`. The string-form call sites remaining in ch06/ch07/ch09 and every chapter's `baseLayout()` now render their axis titles again without editing each call site. |
| R-ch06-1 (mobile overflow) | IMPLEMENTED — `.katex-display{overflow-x:auto}` was already shared; the `@media (max-width:720px)` prov-chip/code rules added to book.css. ch06's page-local copy left (redundant, harmless). |
| R-ch05-1 (cross-widget gates) | IMPLEMENTED — `predictReveal(..., {gates})` + shared `.is-predict-locked` (JS-only class: no-JS readers can never be locked out). ch05's local mechanism left in place (works, incl. its noscript unlock). |

**Rejected/deferred:** none — every filed request was judged worth centralizing.

---

## 3. Integrator edits to chapter files (complete log)

Chapter files are chapter-agent property; the integrator touched them only under the QA
mandate (notation/link/typo-level + replacing adopted workarounds):

1. `ch01-ruler.html`, `ch04-loud-half.html`, `ch08-mass-channel.html`,
   `ch09-universe-factory.html` — prepended `<!DOCTYPE html>` + `<meta charset="utf-8">`
   (R-ch07-3), with a pointer comment. No content changed.
2. `ch05-unseen-galaxy.html`, `ch07-redshift.html`, `ch09-universe-factory.html`,
   `ch11-honest-state.html` — replaced the page-local bias-rail pip append workaround with
   `Book.biasRail({..., pips})`; removed the page-local `#bias-rail .chNN-pip` CSS (comment
   left in place). Pip labels/notes carried **verbatim** — no wording changed.
3. No other chapter edits. Notation drift against the §3.1 table: **none found** (all 37
   `data-term` keys valid; no banned sentence asserted anywhere — all grep hits are
   quotation-in-trap usages, which the pedagogy requires).

Frozen-by-spec but stale: `book/README.md` still names the retired `ch00-demo` files.
README is on the integrator's frozen list, so it was left; one-line fix for the orchestrator.

---

## 4. QA sweep — every check and its result

| check | result |
|---|---|
| `make_all.py` end-to-end | **PASS** (13/13, exit 0, ~104 s; after the subprocess-isolation fix — first run failed on gen_ch03, see §1) |
| `node --check` on `js/book.js`, `js/manifest.js` | **PASS** (chapters carry no separate .js files; inline scripts are exercised only in a browser — see gaps) |
| Runtime smoke of new helpers (node + DOM stubs) | **PASS** — `interp1` (interior/clamp/array), title normalization (string→`{text}`, object preserved, `layout.title`/`legend.title`), `Book.axis`, ledger search (match, `#id` match, empty query) |
| Doctype + charset on every page | **PASS** — 14/14 pages + `_template.html` |
| Only relative refs; no CDN/absolute/external | **PASS** — zero `http(s)://`, `file://`, root-absolute or `/home/` references in any shipped HTML/CSS/JS (one comment mention of `file://` in book.js) |
| Internal links resolve | **PASS** — all `href="*.html"` targets exist; index links all 14 pages |
| Museum anchor integrity | **PASS** — all 14 FIXED exhibit anchors present in `museum.html`; all 12 incoming `museum.html#ex-*` links from chapters target existing ids; the five interlude-carrying chapters link their mandated exhibits (ch04→pdet-anchor, ch05→starvation, ch07→volume-trunc, ch08→mass-trunc, ch10→h0-independent); ch09 links its two annex exhibits (absolute-marginal, wg-bookkeeping) |
| Cross-chapter prediction recall | **PASS** — every id read via `Book.getPrediction` is written by some page; `ch03-host-guess` written by ch03 exactly as ch11 probes; registry published in manifest.js |
| Noscript/static fallbacks | **PASS** — every chapter page ≥3 `<noscript>` blocks (range 3–8) |
| Question blocks well-formed, answers hidden | **PASS** — selfcheck count = hidden-answer count on every page; zero `<details class="answer" open>`; zero hardcoded `.reveal.shown` |
| KaTeX delimiter convention | **PASS** — `$…$`/`$$…$$` only; zero `\(`/`\[` usages |
| Script order per template | **PASS** — manifest.js → plotly → katex → auto-render → book.js on all 14 pages |
| Template conformance (topbar/nav/rail/provenance/dossier) | **PASS** — all pages carry topbar, `data-nav`, `Book.biasRail`, provenance panel, self-checks; dossier on ch01–ch11+museum (ch00 has none by design — it opens in Ch 1) |
| Data budget | **PASS** — 36 files, 1.30 MiB total, max 288 KB < 500 KB |
| Notation vs Symbol Passport | **PASS** — 127 term tags, 37 distinct keys, 0 unknown keys |
| Banned-sentence scan | **PASS** — hits only inside traps/adjudicator blocks that quote-and-dismantle |
| Light/dark CSS audit (static) | **PASS at grep level** — all literal colors in chapter styles are theme-safe Okabe-Ito accents or `white-space`; all surfaces use `var(--…)` tokens; new shared chrome styled for both themes. **Not verified in a browser** (gap #1) |
| h-grid seam rule | Not re-derived by the integrator; every chapter's flags file states its own gate (e.g. ch05: "nothing takes a numerical derivative across the seams") — trusted as delivered, reviewer spot-check recommended |

---

## 5. Consolidated flags (from `book/design/flags/*.md` — NOT resolved here)

The 13 flag files total ~1,570 lines. Full texts remain authoritative; this is the
integrator's index, grouped by blast radius. **Per the build contract, none of these were
reconciled** — every chapter shows both values where a dispute exists.

### 5.1 Cross-chapter numeric disputes (need a spec-owner decision)

1. **EMRI-889's "σ_dL/dL = 8.0×10⁻⁵" — the book-wide one.** Six chapters independently
   measured the CRB row: σ_dL = 7.984×10⁻⁵ **Gpc** (absolute), σ_dL/d_L = **8.98×10⁻⁴**.
   The spec's figure appears to be the absolute Gpc value carried under a fractional label
   (×11.25 slip). Three independent corroborations (1/ρ scaling, IDEALIZATION_LEDGER's own
   0.09–0.11%, dimensional identity). Downstream casualties: Q1.2's "0.008%" answer and
   Q6.5's "6000×" (measured ≈550×). Chapters print both values everywhere; the decision is
   above any single agent. *(ch01 F1, ch02 F-ch02-1, ch03 F-ch03-1, ch06 F-ch06-1/-2,
   ch07 FLAG-1, ch09 F-ch09-6.)*
2. **C7 rail threshold 0.256 vs 0.2644.** Three artifacts state 0.256; solving those same
   artifacts' corrected law for the 0.86 edge gives 0.2644, and the delivered per-host
   measurement brackets 0.264. Origin of 0.256 could not be reconstructed. The page prints
   the artifact value as the threshold (the book may not resolve what the project has not)
   and draws the live crossing where the law puts it. *(ch07 FLAG-2.)*
3. **C5 leverage gloss.** "dh*/dε 1500–2400× idealized" vs recomputed per-run
   141.8–2457.8× (median 197×) from the adjudicator's own output file; the internal
   5×/1000× cross-check favors ~200×. Also the "0.12–0.51 σ_h" gloss (measured
   0.085–1.124). The 0.025 Poisson headline itself reproduces to 1e-6 and is what the
   chapter quotes. *(ch11 F-ch11-1/-2; touches Ch 5's Q5.4 too.)*
4. **C11 band upper endpoints.** +0.0097/+0.0181 not found in the two_branch archives the
   window names (recomputed +0.0078/+0.0157, bit-equal re-runs); plausible source = pooling
   with the catalogue-mode harness family. Exoneration verdict unaffected (if anything
   stronger); "6–16×" as stated matches neither quoted band. *(ch10 F-ch10-1.)*
5. **The 4π "5000×".** The measured pipeline inflation (~5000×, ledger #46) vs the ratified
   analytic factor (1.6×10³ at 2°, 1.8×10⁵ at median localization); the fix-site code
   comment pins "5000× at σ_sky≈2°", which the derivation contradicts; a second concurrent
   change (conditional→marginal width) plausibly owns the gap, but no artifact decomposes
   it. *(ch05 F-ch05-1.)*
6. **C9's ×2.19 vs ×2.446** — inherits the already-registered sources-map §7.7 open dispute
   on the `generator_marginal` w_G curve; both point sets plotted. *(ch09 F-ch09-1/-2.)*

### 5.2 Museum red flags (candidate new ledger entries — for the author)

7. **⚠⚠ F-museum-1: the flagship exhibit's mechanism attribution does not reproduce.**
   "`fixed_quad(n=50)` aliases the GW peak to 0.0000" is a **scalar-collapse artifact of
   the diagnostic script** (`dist()` is scalar-only; under `fixed_quad` the GW factor
   becomes a constant at the window's lower limit — reproducing the published table digit
   for digit, flat in n). Production uses `dist_vectorized` and is unaffected; n=50
   vectorized is within 1.7–2.3% of exact; genuine aliasing exists at lower n (factors
   3.5–6.3× both directions, n≤50). The FALSIFIED verdict on `volume_trunc` stands
   (production A/B gate); mechanism (2) also reproduces. **Two production code comments
   propagate the wrong attribution** (`bayesian_statistics.py:384`, `:3670`). The exhibit
   ships both evaluation modes and adjudicates nothing.
8. **F-museum-2:** #49a's "catalog_only tracks truth EXACTLY" is a grid-step overstatement
   in 2 of 5 re-run cells (0.67→0.66, 0.70→0.69); the headline (production MAP = 0.86 for
   every truth) reproduces exactly.

### 5.3 Resolved-by-arithmetic / definition-pinning (no action needed, recorded)

9. Injection pool 200,807 "rows" = 200,100 data rows + 707 headers (`wc -l` vs `len(df)`);
   confirmed independently three times. Quote data rows. *(ch04 F-ch04-1, ch06 F-ch06-7,
   ch09 F-ch09-5 — the two design docs disagree with each other here.)*
10. "3 golden events carry 46%" reproduces **only** under `score_realistic.py`'s 3-point
    curvature metric (other natural metrics: 41.9–52.5%); denominator ambiguity 46.4%
    (in-cat) vs 47.0% (signed total). Metric now pinned; Ch 2/10/11 all import it.
    *(ch02 F-ch02-2, ch10 F-ch10-2.)*
11. 1588 events (delivered posteriors) vs 1590 stored CRB rows — both correct. *(ch01 F2.)*
12. `dl_max(0.73)=9.164987 Gpc` is the p_det grid ceiling (1.1× the 8.33181 max horizon),
    not the pool's max distance column (10.686). *(ch04 F-ch04-2.)*
13. Ledger #26's 0.010 is the **joint** effect of two bundled changes (numerator-p_det
    removal + ratio-of-sums); no artifact splits them; per-event rearrangement sizes
    measured fresh instead. *(ch03 F-ch03-4/-9.)*
14. C10's 39.1% counts the sign of (1−w_G)·L_comp (prefactor included); L_comp alone gives
    27.7% — understated, not contradicted. *(ch08 F-ch08-6.)* The 0.0354→0.0061 estimator
    identified to 4 s.f. *(F-ch08-7.)*

### 5.4 Spec/pedagogy wording vs measurement (chapters amended the page, logged)

15. "Tens of thousands of galaxies in the ball" is the top-decile tail: median ball 1616,
    median candidates after z-window 12; EMRI-889's own ball holds **3**; 552/1590 events
    have zero candidates (truth-catalogue reconstruction — *not* the run's drop count,
    which is zero). *(ch03 F-ch03-2/-10, ch05 F-ch05-2: the 493 count is
    zero-catalogue-leg, not zero-host; 2 of 493 have in_catalog=True, undiagnosed.)*
16. Trap 6.A's "largest exactly for the loud events": flat in SNR (Spearman −0.019);
    the in-cat vs dark split (0.058 vs 0.041) is what the run supports. *(ch06 F-ch06-3.)*
17. Ch 8 card's "M_z at 10⁻⁴": CRB table median is 8.8×10⁻⁸; 10⁻⁴ is plausibly a test
    tolerance transcription. *(ch06 F-ch06-5.)*
18. 2D pull **range** +3.4…+4.5 does not reproduce (recomputed +2.474…+4.735; mean +4.04
    and 10/10 do). *(ch08 F-ch08-1.)* I8.2's four spec MAPs are the constant-C sweep, not
    the literal unit dial — both dials shipped, correctly labelled; Ch 11 must not call the
    constant-C walk a "unit walk". *(F-ch08-2.)* 606's suppression measured 73.8× vs the
    pedagogy's "80×". *(F-ch08-4.)*
19. Prologue: 73.04±1.04 used (published pair) vs the docs' 73.0; 4.89σ two-number
    combination printed beside the paper's 5.0σ. *(ch00 F-ch00-1/-2.)*
20. I10.1's "run 200 universes" → the archived ensembles are n=120; button reads the count
    from data. *(ch10 F-ch10-3.)* I5.2/Q5.4's "~0.73" → measured 0.740 printed as distinct
    from truth 0.73. *(ch05 F-ch05-3.)* The "84%" fair-framing rule applied in ch03/ch05
    per the binding amendment. *(F-ch03-11, F-ch05-4.)*

### 5.5 Provenance/venue clarifications that protect other chapters

21. Ch 3's reconstruction venue is the TRUTH catalogue; the run's own L_cat used observed
    realizations absent from this checkout — 889's rebuilt leg peaks 0.77 vs the run's
    0.75; candidate explanation named, not verified (Ch 9 territory). *(ch03 F-ch03-3,
    F-ch03-5: the chapter teaches the local ratio-of-sums while the campaign runs the
    global-denominator `absolute_marginal` branch — stated on-page with a Ch 9 chip.)*
22. `c9_darkdraw_results.json` carries three KS statistics; only the `production_pool`
    block (D=0.0863) is current. *(ch09 F-ch09-4.)* G1's −17.2% is a difference of
    normalized shapes, not a ratio (×2.48). *(F-ch09-7.)* The 4-dp w_G log-line rule
    verified in kind (measured max rel. dev. 3.95e-4 vs the quoted 4.8e-4 bound).
    *(F-ch09-3.)* Pool has two writer eras; 6,000 early rows lack p0/t_plunge columns —
    excluded, never imputed; only stratum 'a' carries the population measure.
    *(F-ch09-5.)* ±3σ z-window is hardcoded (`sigma_multiplier` is dead) — do not quote
    "±2σ". *(ch06 F-ch06-4.)* σ_lnM 1.28 (catalogue-side) vs 0.58 (kernel floor) are
    different objects; chapters state which. *(ch08 F-ch08-5.)*
23. **Line-anchor drift:** several spec chips predate the current tree
    (`handler.py:519→558`, `:592→623`, `:605→634`; `bayesian_statistics.py:1052→2459`,
    `:3309-3311→:3388-3392`, `:4014→:4097`, `:4363-4370→:4442-4459`, `:3362→:3445`).
    Chapters kept the spec's anchors (per §3.2 they are re-grep anchors) and recorded
    current positions. Content matches at every site — drift, not conflict.
    *(ch05 F-ch05-5, ch06 F-ch06-6, ch08 F-ch08-3.)*
24. **Build portability:** three source families are untracked working-tree-only artifacts
    of the main checkout — `real_r*/diagnostics/event_likelihoods.csv`, the 707-file
    injection pool, and `results/commission_20260701/**`. Generators use a tracked-first /
    sibling-checkout-fallback / keep-committed-output pattern and print a NOTICE rather
    than failing. A fresh clone **without** the sibling checkout regenerates most data and
    keeps committed JSON for the rest. Either commit those artifacts or keep the pattern.
    *(ch04 F-ch04-5, ch07 note, museum F-museum-3.)*
25. Plunge-window sidebar placement followed BOOK_DESIGN over sources-map §8 (precedence
    rule). *(ch01 F3.)* Ch 2's realistic-venue information *shares* are ill-conditioned and
    not quotable — signed sums only, per the artifact's own words. *(ch02 F-ch02-3.)*

---

## 6. CI integration (restated from BOOK_TECH_DESIGN §2.4)

One step in this worktree's `.github/workflows/ci.yml` `pages` job, after "Generate
interactive figures", before "Upload Pages artifact" — **already applied in this worktree**
(never in the main worktree's copy):

```diff
+      - name: Build discovery book
+        run: |
+          uv run python book/generators/make_all.py
+          mkdir -p _site/book
+          cp -r book/site/. _site/book/
+        continue-on-error: true
```

No new job, no new dependencies (`pages` already runs `uv sync --extra cpu --extra dev`);
`continue-on-error` matches the interactive-figures step's philosophy. Deploys to
`https://jasperseehofer.github.io/MasterThesisCode/book/`. **CI caveat from §5.5 item 24:**
in CI the sibling-checkout fallback does not exist, so generators depending on untracked
artifacts will keep the committed JSON (by design) — the deploy still succeeds and serves
current data.

---

## 7. Known gaps (ranked)

1. **No in-browser verification of the integrator's changes.** The chapter agents ran
   their own headless Chromium checks during their waves, but everything added in *this*
   pass — persona switch, Symbol Passport popover/glossary, ledger search panel, shared
   rail pips on four converted pages, the Plotly title wrapper, the rewritten index —
   has been verified only by static analysis, node syntax checks, and DOM-stub smoke
   tests. **A full-site serve + click-through is the phase-4 reviewers' first job**
   (`cd book/site && python3 -m http.server 8000`).
2. **Persistence side-effects of the predictReveal id fix.** Pages where the id sat only on
   `.predict-row` (ch07-q1/q2, ch09-q-bench/q-identity) now restore choices across
   reloads for the first time. ch07 anticipated this; ch09's reveal-resize callback does
   not fire on restore (plots may need one window-resize to settle — cosmetic, unverified).
3. **Ledger tag chips race dynamically-tagged controls.** ch03 sets `data-hypothesis="26"`
   inside a data-load promise; if that resolves after `Book.ledger.init()`'s scan, the #26
   chip is absent from ch03's panel seed (search still finds it by hand). Best-effort by
   design.
4. **Q11.6 carries a model answer** ("run the control" — the discipline answer). The spec
   wants the book's last question to end without an answer key *where the project has
   none*; the ch11 agent judged the discipline answer to be pedagogy-conformant. Phase-4
   should check this against BOOK_PEDAGOGY Part 3 §Ch 11 verbatim text.
5. **`book/README.md` is stale** (names the retired demo files) but frozen to the
   integrator; one-line orchestrator fix.
6. **Museum has no per-chapter backlinks** (nav is the only return path from an exhibit to
   the chapter that sent you). Acceptable; a "referenced by Ch N" line per exhibit would be
   a nice phase-5 touch.
7. **`__pycache__` from the pre-fix make_all runs** lingers under `book/generators/`
   (untracked build churn; harmless).

---

## 8. Reviewer's guide — what to attack hardest

**For the student reviewer (Mara test):**
- Ch 5 §2 and Ch 9 §3–§4 are the rung-violation danger zones (w_G mechanics and estimands
  compressed hard). Can you follow both without opening a linked doc?
- The Symbol Passport is new chrome: hover `w_G` in Ch 5 vs Ch 9 — does the
  "estimand-dependent" caveat land *before* Ch 9 confuses you, or does it spoil Ch 9's
  reveal?
- Predict-locks: try to see I5.2's reveal without committing a guess. Try it with JS off —
  you should get the static fallback and the answer, never a dead lock.

**For the expert reviewer (Examiner test, rubric B):**
- §5.1 items 1–3 are the highest-value audits: re-do the σ_dL/dL arithmetic from
  `prepared_cramer_rao_bounds.csv` row 889 yourself; re-solve the C7 law for the 0.86
  edge; re-divide `c5_leverage_results.json`. Each chapter claims to show both values —
  verify no page quietly prefers one.
- **F-museum-1** is the sharpest thing in the build: a falsified mechanism attribution
  inside a correct falsification, reproduced live in exhibit M1's two-state switch. Check
  the museum really does *not* adjudicate it, and decide whether it becomes a ledger row
  and whether `bayesian_statistics.py:384`/`:3670` comments get corrected (physics-change
  protocol applies to the main repo — out of book scope).
- The C7/C8/C9 + C6 presentation: Ch 7 §6, Ch 8 §6, Ch 9 §6, Ch 11 §4–§5 must match
  `CLAIM_2D_BIAS_20260730.md` as amended + `ADJUDICATION_20260730.md` — statuses verbatim,
  both halves of every amended pair, cell B named as decider, no narrative resolution.
- Spot-check five provenance chips per chapter against the artifacts (the §3.4 rubric
  minimum); the line-drift table in §5.5 item 23 tells you where re-grepping is expected.
- The bias rail's numbers at each chapter: −0.178/0.000 are Phase-32/venue-scoped values;
  the r1 realization reads 0.740 — check every page separates "bias" from "one
  realization's MAP" (ch04 F-ch04-4 is the template).

**For both:** the book's core honesty claim is that **nothing interactive lets a dead
hypothesis look alive** (museum meta-rule). Try to defeat it: drive I7.1 past the C7
threshold, rebuild #61 in I5.1's dial, re-try `volume_trunc` in the museum's quadrature
dial — every dead end must volunteer its verdict and its ledger row.

---

*Integrator, 2026-07-31. No git operations were performed; the orchestrator commits.*

---

# Revision pass 2026-07-31 (post-review; spec = `book/design/REVISION_WORKLIST.md`)

Executed in three waves per the worklist's §E fan-out: wave 0 (ch03 census regeneration +
integrator pass 1: §D items 1–3, 6, 7, 12), wave 1 (thirteen chapter/museum agents), wave 2
(this pass: index.html, §D items 4, 5, 8–11, close-out). Every flag file gained a dated,
append-only REVISION section; nothing above those lines was edited. The four §D-12 content
gates now run inside `make_all.py` and are **all green** (0 violations).

## 9.1 What changed per unit

**Book-wide decisions applied (worklist §A):**
- **D1 (σ_dL units slip, author mandate):** spec value is now σ_dL/d_L = **8.98×10⁻⁴**
  (absolute 7.98×10⁻⁵ Gpc). One canonical dossier row + erratum line, defined once in
  `js/manifest.js` `BOOK_CANON.sigmaDL`, applied on every dossier card ch01–ch11 + museum.
  The old value survives only inside erratum notes (D1 gate PASS site-wide). Downstream
  casualties fixed: Q1.2 rewritten as the erratum lesson, Q6.5's "6000×" → measured **≈550×**,
  `ch04_denominator.json` key renamed `sigma_dL_Gpc` + corrected `sigma_dL_over_dL = 8.98e-4`.
- **D2 (ball-search multiplier):** ch03 regenerated at the production **n_σ = 1.5**
  (`bayesian_statistics.py:2838`; `gen_ch03.py` `SIGMA_MULTIPLIER = 1.5` with the
  signature-default-trap warning on the page). Regenerated census: EMRI-889's ball = **2**
  galaxies at radius **0.757′**; zero-candidate count **607/1590** (was 552 at 2σ) — consumed
  by ch02 (BLOCKER-1 rewrite), ch04 §5, ch05 §4; ch03/ch06 print identical 889 ball facts.
- **D3 (cell B landed):** the pre-registration → control → scored-readout arc is on
  ch07/ch08/ch09/ch10/ch11/museum/index. Pre-registration blocks kept verbatim (registered
  jobs 6101146/6101147); results cite the resubmission 6103219/6103220 with the one-sentence
  plumbing note. **Honest scoring everywhere: 2 of 3 exact + 1D MAP 0.7450 one grid step
  above the registered 0.70–0.74 band (mean 0.7320 inside; band written in MAPs)** — matches
  the readout's own same-day ERRATUM; the readout's original "confirmed on every
  pre-registered read" sentence is explicitly not copied (ch11 says why). 2×2 printed in MAPs
  throughout, means in a footnote, "72% is 2D-only" warning attached. The four canonical
  cell-B rail pips (ch07/ch09/ch10/ch11) are byte-identical, defined once in
  `BOOK_CANON.cellB`.
- **D4 (spoiler discipline):** forward references name phenomenon + chapter only; ch08/ch10
  decks de-spoiled (+0.077 appears only at/after its reveal); traps relocated to where the
  misconception forms; index journey de-spoiled (below).
- **D5 (both-values policy):** every page quoting one half of a still-open pair now carries
  the other + flag pointer (ch05 Q5.4 leverage, ch08's five σ_Mz sites, ch11's opening 2D
  pull row +2.47…+4.74 with the F-ch08-1 footnote).

**Per unit (headline items; full detail in each flag file's REVISION section):**
- **ch00** — budget trim 2.33×→1.70× (σ_tot algebra into a num-view); GWTC-3 dark-siren row
  (68⁺⁸₋₆, arXiv:2111.03604) added to the third-methods figure with provenance chip;
  Trap 0.A de-spoiled; "most generous case" names its branch; Q0.3 re-aimed (decision logged).
- **ch01** — F1 → resolved-erratum block (the 1/ρ check is now the proof); Q1.2 = the erratum
  lesson; σ_Mz both-values note (10⁻⁴ claim / 8.8×10⁻⁸ measured, F-ch06-5); dossier mass row
  relabelled M_z; Q1.3 folded into §2; standard-siren scope sentence; Q1.4 chain via Ch 2.
- **ch02** — "tens of thousands" ×3 → the measured distribution (median ball 1,616 was the 2σ
  figure; regenerated numbers consumed, Q2.5 graded "the answer is a distribution");
  central predict graded ((d) — a handful; wrong options each answered); "62%" → **52%** for
  r1's pair with the ensemble figure labelled; β(h)^N box moved out of the main column
  (rung repair, received by ch04); Trap 2.A/2.B de-spoiled; percentages name denominators;
  Q2.3 re-aimed; 7 Plotly instances lazy-initialized.
- **ch03** — D2 regeneration (above) + corrections appended to F-ch03-2/-10/-12; RATIFIED box
  states n_σ = 1.5 + the signature-default trap; σ_z-derived numbers carry the §7.19(d)
  parent-staleness caveat (new F-ch03-13); Q3.4 internally consistent; new F-ch03-14
  (search-disc vs localization ellipse).
- **ch04** — D1 dossier + JSON key rename; p_det marginality stated in §2 ("marginal" /
  intrinsics, ⏭ Ch 9); Q4.3/Q4.4 answers derivable from ch00–ch04 only; guess-marker
  desync fixed (slider locks); §5 zero-candidate figure = 607 (regenerated).
- **ch05** — w_G's *type* (one number per h; selection- and volume-weighted line-of-sight
  average) now in the narrator flow before "First: 12%", with per-sample labels
  (76/1588 = 4.8% one seed vs 164/3135 two seeds); C10 attribution corrected (dark
  ΣΔln L_comp = −22.72; 27.7% L_comp-alone vs 39.1% with prefactor); Q5.4 carries both
  leverage halves; I5.1 gains the narrated κ midrange (V51 third state, 0.86 plateau
  explained — new measurement F-ch05-8); §3 de-spoiled ("factor of 5000" out of the heading).
- **ch06** — D1 (§4 erratum form; Q6.5 ≈550× with the units-lesson dagger); §3 gains the
  measured 14×14 condition-number distribution (median 2.6×10⁹ / p95 1.4×10¹⁰ / max
  3.9×10¹², recomputed by gen_ch06, with the float64 price tag) + Babak-2017 plausibility
  clause on §4.1's σ_Mz ("not tested here"); §4.1/§5 bookkeeping folded (budget 2.37×→
  reduced; residual over-budget logged as F-ch06-10); traps relocated; ball cross-ref to
  ch03's census closes the loop.
- **ch07** — §6's "It has not landed." → the dated landed-block (90.7% vs 89.2% vs 5.3%;
  argmax 0.860 as registered; combined rail 69.7%→57.9%) + the G2b scope constraint;
  provenance OPEN→dated FINDING; Trap 7.B "…and did, 2026-07-31"; the honest staleness
  nuance (98.7% ≠ 90.7%, different statistics, "resolves confirming, somewhat weaker");
  `ch07_c7.json` gained `hosts.resolved_by_cellB` + landed decider; noscript updated;
  φ_cat defined at first use; Q7.1 → transfer form; traps relocated; σ_Mz both values.
- **ch08** — the five lone σ_Mz ≈ 10⁻⁴ sites → both-values treatment (RATIFIED display
  equation now prints claim | measured, "Two values, both printed, neither preferred", new
  F-ch08-10); deck de-spoiled ("Watch what it does instead"; +0.077 only inside the reveal);
  cell-B §4/§5 block (+18.00 nats unscattered = estimator-borne; C4 partition
  configuration-scoped with the reviewer-computed marker, raised as F-ch08-9 ⚑ for the
  author); Q8.1 rewritten on §2's correct mechanism as transfer; spectral-siren
  disambiguation sidebar; 8 Plotly instances lazy-initialized; rail 2D row arms at the
  cold-open reveal (D4).
- **ch09** — re-litigation guard + verdict: C9 **live, cell-B gate released 2026-07-31**,
  fix = joint C9+C8 mass-consistent mixture; the bit-identical w_G payoff stated as the
  book's cleanest pre-registration hit (max|Δ| = 0.0 across 41 grid points; 0.1625175 /
  0.1215039 / 0.1038732 — now a hard generator gate, F-ch09-8); §4 symbols picture-first +
  passport-tagged (nw/Sglob/Wcat/Vf/Fincat); the two "residuals" renamed (shape −17.2% vs
  1D bias +1.667%, F-ch09-10); `global`-mode deprecation venue-scoped with gwcosmo named
  (F-ch09-11); Q9.1 hardened; dossier beat leads with 889's own w_G.
- **ch10** — §5's forward promise → landed ("submitted, and landed 2026-07-31…"); Q10.5
  original answer intact above its dated postscript (0.7300→0.7900, 72%; "before the
  control: still nothing"); deck de-spoiled; Laghi et al. 2021 (~1%) anchors the §3
  scenario table; I10.2 num-view added; job IDs split per D3.
- **ch11** — §5 retitled "The confound, and the control that resolved it": pre-registration
  verbatim + dated readout block, 2×2 in MAPs, B−A/C−B/C−A rows, the 1D-share>100% note,
  explicit scoring (2✓, 1D ✗ by one grid step), badge RESOLVED 2026-07-31; closing block:
  item 1 leaves the no-answer list via a dated resolution card, renumbered to four;
  `ch11_board.json` regenerated (C6 amended heading verbatim, C9 gate-released, C7
  adjudication appended, n_live = 4 on all three surfaces); opening-table 2D pull row both
  halves; §4's C7 block = the two-priors framing (φ_cat), adjudicating nothing; recall beat
  maps slugs to human labels, verdict lands after; "3130" → 3135; §4 scoreboard collapsed
  to one-line statuses + folds; Q11.6 kept + dated postscript; meta/subtitle "…the control
  that landed the night the book was built".
- **museum** — parser fixes: separator class `[,/·;]` recovers #41/#43/#44/#52 →
  **do-not-re-try union = 30** (counts updated in census caption, noscript, M.4's answer;
  new DNR gate); row #68 7-cell round-trip restored (`[AMBIG] see #69` back; ROW hard gate);
  the two cell-B statements date-scoped (gate released, C9 live, w_G off the exonerated
  list); ledger #88 gains the book-added "Cell B ≠ the 2×2 cell B" annotation (`book_note`,
  rendered in the museum browser and now also in the shared search results); §7's BW3 claim
  → the delivered instrument (updated again this pass after the inline chips shipped);
  "twenty-one were fixed" → the ledger's own "insufficient alone" qualifier; M1 flag box
  0.2–2.3%; static fallback 10⁻⁷⁹–10⁻⁸⁶; per-exhibit backlinks ("referenced by Ch N §… ·
  ← back") on every chapter-referenced exhibit; exhibit browsing gets a path via mus-folds.

**index.html (this pass):**
- **[P0] expB MJ-7:** the honest-state callout now reads "three live, measured
  inconsistencies … **plus a fourth (C5) …, and an attribution (C6) that was confounded
  until the control was run on 2026-07-31**" — consistent with ch11's board count of four
  (C5, C7, C8, C9); the contract callout says "the four live inconsistencies".
- **[P1] ped B1/m7, tomas M10:** "The journey" now maps by *question* (failure-framed, no
  reveal numbers un-collapsed); the discovery statements moved into a
  `<details>` ("Spoiler: what each chapter discovers"); Ch 4's and Ch 8's blurbs rewritten
  as failures ("Watch what it does instead" / "…and the answer moves the wrong way"); the
  ch11 blurb's "control still running" → "the control that landed the night the book was
  built"; the dossier teaser no longer leaks ch05's 76-events reveal.
- **[P1] ped B4 step 1:** the BW3 promise now describes the shipped instrument (search box +
  seeded dead hypotheses + the new inline verdict chips).

**Shared files (this pass — §D items 4, 5, 8–11; WIDGET_REQUESTS R-INT-10…14):**
- **§D-4 cumulative bias rail:** `window.BOOK_BIAS_ROWS` (manifest.js, five rows with
  `from_chapter` + `match`) merged by `Book.biasRail`; the rail never loses rows moving
  forward; ch11 and the museum now show the full five-row history; page-declared rows win
  (ch08's reveal-armed 2D row preserved; `from_chapter` doubles as the D4 spoiler boundary).
  New shared `Book.chapter()`. Verified per-page by node smoke test (ch02/04/07/08/11/museum
  orderings all correct).
- **§D-5 BW3 inline chips:** `Book.ledger` renders `⚖ #N — verdict` (+ do-not-re-try badge)
  inside a widget when a `data-hypothesis`-tagged control becomes active;
  `data-hypothesis-verdict="inline"` opts out (ch09 I9.2 uses it); any other value is the
  chip's verbatim verdict text (ch07's two widgets). Non-control tags inside widgets (ch05's
  own verdict box) seed the panel only — no double report. Tag inventory now: ch03 #26
  (dynamic), ch04 #9 (the denominator switch — the worklist's named AC), ch06 #51,
  ch07 #42/#70, ch08 #71/#72/#89, ch09 #61 (opt-out), ch10 **#49a (added this pass on
  I10.1)**, ch11 #61/#64.
- **§D-8 predict grading:** `data-predict-correct` support in `predictReveal`
  (`.predict-correct` ✓ ring / `.predict-missed` strike); ch02's already-authored attribute
  is now live.
- **§D-9:** `Book.lazyPlot` shipped (ch03 recipe centralized; ch02/ch08/ch09 keep their
  identical page-local copies per the zero-coupling precedent); `themedPlot`'s per-plot
  MutationObservers consolidated to one; the two `@media print` blocks merged; persona
  switch now only ever opens strata (never force-closes); `_template.html` placeholder-JSON
  guard comment (the one granted template exception); the three persona nudges placed
  (ch06 Fisher, ch08 mass measure, ch11 adjudication).
- **§D-10 anchor-drift tooltips:** 16 `title="current lines/tree …"` tooltips added
  (ch02 ×4, ch04, ch05, ch07, ch08 ×7, ch10, museum ×2), positions verified against the
  tree this pass (`handler.py :519→558 / :592→623 / :605→632-634`;
  `bayesian_statistics.py :3309-3311→3388-3392 / :3362→3445 / :4014→4097 /
  :4363-4370→4442-4459`; `IDEALIZED_BASELINE_READOUT.md :36-39→50-52 / :42-47→54-60 /
  :47-48→59-60 / :50-52→64-66`; `CLAIM_2D_BIAS_20260730.md :587-588→602`). ch01/ch03/ch06
  had already re-grepped theirs in wave 1.
- **§D-11 README:** rewritten — retired `ch00-demo` references dropped, current layout,
  qa_gates documented, vendor-scoped external-ref grep policy for future CI recorded.
- Pass-1 leftovers closed: ch06's and ch10's page-local rail-pip appends converted to
  `Book.biasRail({pips})` and their page-local pip CSS removed (§D-6 / tomas-m2; ch05, ch07,
  ch09, ch11, ch08 were already converted).

## 9.2 QA results (close-out)

| check | result |
|---|---|
| `make_all.py` end-to-end (subprocess mode) | **PASS — 13/13 green, exit 0, ~103 s** (gen_ch00 0.01 · ch01 0.37 · ch02 0.32 · ch03 12.2 · ch04 1.63 · ch05 1.32 · ch06 7.29 · ch07 0.75 · ch08 1.31 · ch09 0.86 · ch10 60.8 · ch11 0.39 · museum 15.7) |
| §D-12 content gates (D1 / ROW / DNR / TNS) | **PASS, 0 violations.** One NOTE: `_template.html:112` still carries the retired dossier row — frozen-list; author call (see 9.4). |
| Canonical-string advisory (cell-B pip wording) | **PASS** — all four pips byte-identical to `BOOK_CANON.cellB.pipLabel` |
| `node --check` book.js / manifest.js | PASS |
| Node smoke: `_mergeBiasRows` per-page orderings | PASS (6 venues incl. ch11 five-row history, ch08 arming preserved) |
| Doctype + charset, all 14 pages | PASS |
| Relative refs only (first-party files; vendor-scoped per README policy) | PASS |
| Internal links + index links all 14 pages | PASS |
| Museum anchors both ways (`ex-*`) | PASS |
| Notation vs Symbol Passport (`data-term` keys) | PASS — 0 unknown keys |
| Self-checks: answers hidden, no pre-shown reveals | PASS |
| KaTeX delimiter convention (`$…$` only) | PASS |
| Script order per template | PASS |
| Data budget | PASS — 1.32 MiB total, max `ch03_skyball.json` 281 KB < 500 KB |
| Cross-chapter predictions read⊆written | PASS |
| noscript ≥3 per chapter page | PASS |
| Retired-string sweep ("not landed"/"in flight"/"still running", "tens of thousands" as assertion, "6000×", lone "1500–2400×", "0.008%", old σ_dL outside errata) | PASS — survivors are quotation-in-flag/erratum/predict-option contexts only |
| Light/dark on new chrome (chips, grading, nudges) | token-based + explicit dark overrides; grep-level PASS, **not browser-verified** (gap #1) |

**Wave-1 acceptance cross-check (2 criteria per unit, spot-verified in the shipped files):**
ch00 (GWTC-3 row + chip ✓ · Trap 0.A clean of #49a ✓) · ch01 (Q1.2 erratum-lesson, no live
0.008% ✓ · σ_Mz both values + F-ch06-5 ✓) · ch02 (no census contradiction, distribution
answer ✓ · 52%-with-ensemble-figure ✓) · ch03 (0.757′/2-galaxy ball identical on ch03+ch06 ✓
· `:2838` pin + 1.5σ in gen_ch03 ✓) · ch04 (JSON key rename ✓ · marginal/intrinsic + locked
slider ✓) · ch05 (w_G type-before-value ahead of "First: 12%" with sample labels ✓ ·
−22.72/27.7%/39.1% ✓) · ch06 (≈550× with retired-6000× note ✓ · 14×14 κ distribution ✓) ·
ch07 (zero stale-tense, 90.7/89.2/5.3 + G2b constraint ✓ · `resolved_by_cellB` in JSON +
noscript ✓) · ch08 (no lone 10⁻⁴, RATIFIED box carries both + flag ✓ · deck de-spoiled,
+0.077 only inside the reveal ✓) · ch09 (no live "gated on cell B" ✓ · bit-identical w_G
payoff numbers ✓) · ch10 (Q10.5 answer + dated postscript ✓ · Laghi anchor ✓) · ch11 (MAPs
table + "✗ — by one grid step" ✓ · n_live = 4 on JSON/widget/noscript ✓) · museum (DNR = 30
everywhere ✓ · #88 book-note disambiguation ✓). **No material failures.** Trivial misses
fixed by this pass: ch06/ch10 pip conversion, ch10's missing #49a tag, `book_note` invisible
to the shared search (now rendered).

## 9.3 Flag addendum (new/changed flags this round — full text in `design/flags/*.md`)

Resolved by mandate (D1): ch01 F1 · ch02 F-ch02-1 · ch03 F-ch03-1 · ch04 F-ch04-6 · ch05
F-ch05-6 · ch06 F-ch06-1/-2 · ch07 FLAG-1 · ch09 F-ch09-12 · ch10 R-ch10-5 · museum D1
entry. Corrections recorded: ch03 F-ch03-2/-10/-12 (census re-measured at 1.5σ; 607/1590);
ch05 F-ch05-7 (C10 scoping). New flags: ch02 F-ch02-4 (quoted-census process note) · ch03
F-ch03-13 (z_error parent staleness), F-ch03-14 (disc vs ellipse) · ch05 F-ch05-8 (κ-dial
non-monotonicity, measured) · ch06 F-ch06-9 (14×14 conditioning + Babak check), F-ch06-10
(budget not reachable by relocation alone) · ch07 "C7 decider landed" entry · ch08
**F-ch08-9 ⚑ (reviewer-computed cell-B C4 partition — for the author)**, F-ch08-10 (σ_Mz
both-values pair) · ch09 F-ch09-8 (w_G equality = hard gate), F-ch09-9 (gate released),
F-ch09-10 (two "residuals" renamed), F-ch09-11 (global-mode venue scoping) · ch11 F-ch11-4
(readout self-scoring vs 2-of-3), F-ch11-5 (2D pull range both halves) · museum F-museum-5
(parser defects — fixed + gated), F-museum-6 (BW3 claim — corrected, then upgraded when the
chips shipped).

## 9.4 Remaining known gaps (ranked)

1. **No in-browser visual check — still #1** (inherited from the build's gap #1, now larger):
   everything this pass added (cumulative rail merge, inline chips, predict grading, persona
   open-only behavior, single theme observer, the rewritten index) is verified by static
   analysis, node syntax checks, and DOM-stub smoke tests only. A full serve + click-through
   (`cd book/site && python3 -m http.server 8000`) is the next reviewer's first job —
   priority pages: ch02 (graded reveal), ch04 (#9 chip on the denominator switch), ch10
   (#49a chip on I10.1), ch11 (five-row rail + chips on the λ dial), index (details spoiler).
2. `_template.html:112` still carries the retired σ_dL dossier row (frozen list; gate NOTE).
   It is the file agents copy dossier markup from — granting the frozen-list exception and
   replacing it with `BOOK_CANON.sigmaDL.dossierRowHTML` is a one-line author call.
3. Inline chips are best-effort for dynamically-tagged controls (ch03's #26 — tag set inside
   a data-load promise; the search panel still finds it).
4. The `qa-allow` escape hatch is used sparingly but is only as honest as its comments —
   periodic audit recommended.
5. Museum per-exhibit backlinks cover the chapter-referenced exhibits (9); exhibits nothing
   links to have none (defensible; noted).
6. `__pycache__` build churn under `book/generators/` (harmless, untracked).

## 9.5 Still-open disputes (nothing below was resolved by this pass)

**Both-values items (worklist D5 — any page quoting one half must carry the other + flag):**
1. C7 rail threshold **0.256 vs 0.2644** (§5.1-2; ch07 FLAG-2, ch10 R-ch10-7).
2. C5 leverage **"1500–2400×" vs recomputed 141.8–2457.8× (median 197×)**; also the
   "0.12–0.51 σ_h" gloss vs measured 0.085–1.124 (§5.1-3; ch11 F-ch11-1/-2, ch05 Q5.4).
3. C11 band upper endpoints **+0.0097/+0.0181 vs recomputed +0.0078/+0.0157** (§5.1-4;
   ch10 F-ch10-1).
4. The 4π **"5000×"** vs the ratified analytic factor (§5.1-5; ch05 F-ch05-1).
5. C9's **×2.19 vs ×2.446** (§5.1-6; ch09 F-ch09-1/-2, sources-map §7.7).
6. σ_Mz/M_z **≈10⁻⁴ (claim file) vs 8.8×10⁻⁸ (measured)** (tomas B3 / §B-8; ch06 F-ch06-5,
   ch08 F-ch08-10 — carried at every site, no book-side correction).
7. ch08's 2D pull **range** +3.4…+4.5 vs recomputed +2.474…+4.735 (mean +4.04 and 10/10
   reproduce; F-ch08-1 — carried with the footnote in ch11's opening table).

**Back to the AUTHOR (worklist §F, carried verbatim):**

1. **F-museum-1's production code comments** — `bayesian_statistics.py:384` and
   `:3670` carry the falsified scalar-collapse mechanism attribution. Main-repo
   change, `/physics-change` protocol. Also: whether F-museum-1 and F-museum-2 become
   BIAS_HISTORY_LEDGER rows (BUILD_REPORT §5.2 already asks).
2. **`CLAIM_2D_BIAS_20260730.md:172`'s σ_Mz ≈ 1e-4** — measured 8.8×10⁻⁸ (F-ch06-5;
   plausibly a test-tolerance transcription). The book carries both values (D5);
   amending the claim file is the author's.
3. **`CELLB_READOUT_20260731.md`'s "confirmed on every pre-registered read"** — the 1D
   MAP (0.7450) missed the registered 0.70–0.74 band by one grid step (band written in
   MAPs; the mean is inside). The book scores it honestly (D3); annotating the project
   artifact is the author's. (expB BL-4.) *[Update, same day: the readout now carries an
   ERRATUM scoring itself "2 of 3 exact, 1D marginal" — the book and the readout agree;
   the original sentence stands in the artifact's history above its erratum.]*
4. **Expert B's reviewer-computed unscattered C4 partition** (219 survivors / 80.7%,
   688 zeroed / 19.2%; dark zero-leg fraction 0.855; ×8.39 de-weighting) — used in
   ch08 with a "recomputed for the book" marker; whether it is promoted into an
   adjudicated artifact / claim-file amendment is the author's.
5. **`BIAS_HISTORY_LEDGER.md:88`'s unescaped pipes** (row #68) — one-character
   upstream fix; the book's parser now tolerates it either way. Also the §3
   "Cell B"/A′ naming collision with the 2×2 — a one-line upstream annotation would
   help future readers (the book disambiguates on its side regardless).
6. **C7 rail threshold 0.256 vs 0.2644** — origin still unreconstructed
   (BUILD_REPORT §5.1-2); no book action beyond the existing flags; project-side
   archaeology is the author's if wanted.
7. **σ_dL upstream** — D1 corrects the book; the spec cards/design docs and any
   project artifact carrying 8.0×10⁻⁵ as a fraction remain uncorrected historical
   record unless the author amends them.
8. **The two `/physics-change` gates cell B unblocked** (joint C9+C8 mass-consistent
   mixture; C7 kernel fix superseding G2b, not the exonerated p_det-in-numerator
   form) — the book now states these as the fix surface; running them is the thesis
   work, not the book's.

*(Plus, from this pass: the `_template.html` dossier-row frozen-list exception, 9.4 item 2.)*

---

*Integrator pass 2, 2026-07-31. No git operations were performed; the orchestrator commits.*
