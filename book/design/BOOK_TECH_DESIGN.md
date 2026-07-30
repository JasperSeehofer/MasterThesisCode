# Discovery Book — Technical Design

Status: **skeleton built and verified** (2026-07-31). Branch: `book/foundations-interactive`
in the dedicated worktree `/home/jasper/Repositories/MasterThesisCode-book`. Never write to
`/home/jasper/Repositories/MasterThesisCode` (main worktree) from this project.

---

## 1. Inspection — how the current site is composed and deployed

**Read-only investigation of `/home/jasper/Repositories/MasterThesisCode` (identical content
at this commit in the book worktree, since `book/foundations-interactive` shares full history
with `main`).**

### 1.1 `.github/workflows/ci.yml` — the only workflow file

Four jobs, in dependency order:

1. **`check`** — `uv sync --extra cpu --extra dev`, ruff/mypy/pytest (fast suite), uploads a
   `coverage-xml` artifact.
2. **`integration`** (needs `check`) — runs `pytest -m slow --save-test-plots`, uploads
   `test-artifacts/` as the `integration-test-plots` artifact.
3. **`docs`** — `make -C docs html SPHINXOPTS="-W"` (Sphinx, autodoc+napoleon+mathjax+copybutton),
   uploads `docs/build/html/` as the `docs-html` artifact.
4. **`pages`** (needs `[integration, docs]`, only on push to `main`) — downloads both artifacts
   into `_site/`, then a "Generate interactive figures" step:
   ```yaml
   mkdir -p _site/interactive
   cp interactive/*.html _site/interactive/
   if [ -d cluster_results ]; then
     uv run python -m master_thesis_code cluster_results --generate_interactive _site/interactive/ || true
   fi
   ```
   (`continue-on-error: true` — never blocks the deploy), then
   `actions/upload-pages-artifact@v5` + `actions/deploy-pages@v4`.

So the deployed site (`https://jasperseehofer.github.io/MasterThesisCode/`) is a flat merge of
three trees under `_site/`: Sphinx docs at the root, `test-plots/` (integration test artifacts,
best-effort), and `interactive/` (a **static, pre-committed** set of 10 HTML files copied
verbatim — `cluster_results`-driven regeneration is best-effort and normally a no-op since that
directory doesn't exist in CI).

### 1.2 `interactive/` — the existing Plotly export convention

`master_thesis_code/plotting/interactive.py` provides 5 factory functions
(`interactive_combined_posterior`, `interactive_h0_tension_explorer`, `interactive_sky_map`,
`interactive_fisher_ellipses`, `interactive_h0_convergence`, `interactive_m_z_improvement`) that
each return a `go.Figure`. The docstring's stated convention is
`fig.write_html(path, include_plotlyjs="cdn")`, but the **committed HTML files actually embed**
`<script src="https://cdn.plot.ly/plotly-3.4.0.min.js" integrity="sha256-..." crossorigin="anonymous">`
— i.e. `include_plotlyjs="cdn"` was used literally: a versioned, integrity-pinned CDN `<script>`
tag, not `include_plotlyjs="directory"` (there is no `plotly.min.js` sibling file next to any
`interactive/*.html`, confirmed by directory listing — only the 10 `.html` files + none else).
`docs_src/interactive/` holds a partial, slightly older subset of the same files (5 of the 10),
apparently an intermediate staging copy, not referenced by the CI workflow at all.

**Consequence for the book:** the existing interactive figures are **not offline-safe** — they
require network access to `cdn.plot.ly` to render. The book's requirement ("must work offline /
`file://`") is a stricter, new bar; the book cannot reuse the existing HTML files' loading
convention and must vendor Plotly itself (see §2.1).

Palette/typography sources inspected for consistency:
- `master_thesis_code/plotting/_colors.py` — Okabe-Ito 7-color cycle (Wong 2011) plus an
  "Observatory + Atlas" `METHOD` token dict (`dark`, `combined`, `spectral`, `bright`),
  `PLANCK_BAND`/`SHOES_BAND`/`PRIOR` tokens, and `SEQUENTIAL_CMAP`/`DIVERGING_CMAP` (batlow/vik
  via optional `cmcrameri`, falling back to `cividis`/`RdBu`).
- `master_thesis_code/plotting/emri_thesis.mplstyle` — 8pt base font (REVTeX-column-sized),
  frameless legends, inward ticks, no top/right spines, `pdf.fonttype/ps.fonttype = 42`.
- `master_thesis_code/plotting/_style.py` — `apply_style()` sets Agg backend + loads the
  mplstyle sheet; irrelevant to the book (no matplotlib in the browser) but the color/typography
  *tokens* are the thing to mirror for a "one system" feel.

### 1.3 `docs/` Sphinx build

`docs/source/conf.py`: autodoc + autosummary (auto-generated stubs) + napoleon (Google-style
docstrings, despite `CLAUDE.md` mandating NumPy-style for new code — pre-existing drift, not the
book's concern) + `sphinx.ext.mathjax` + `sphinx_copybutton`. Deploys to the Pages root. The book
is **additive** — a new `book/` subtree, not touching `docs/`.

### 1.4 No existing interactive/notebook conventions beyond the above

`notebooks/` exists (not inspected in depth — out of scope; nothing under it feeds Pages).
No `book/`, no static-site generator, no JS bundler anywhere in the repo. This confirms the
"no build framework" instruction is not fighting an existing convention — there isn't one yet.

---

## 2. Architecture decision

### 2.1 Static site, vendored libraries, no build framework — DECIDED

**Options considered:**

| Option | Verdict |
|---|---|
| Plain HTML/CSS/JS, vendored libs, `book/site/` deployable as-is | **Chosen** |
| Static-site generator (Eleventy/Hugo/mdBook) | Rejected: adds a Node/Go toolchain + build step for a ~10-chapter book; the project has zero JS tooling today and the instructions explicitly ask to avoid one "unless clearly justified" — it isn't, at this scale |
| SPA framework (React/Vue + bundler) | Rejected: interactive widgets here are simple (sliders, KaTeX, Plotly traces) — a framework's reactivity model buys nothing a hand-rolled `book.js` (~200 lines) doesn't already give, and it breaks the `file://`/no-build simplicity |
| Jupyter Book / MyST | Rejected: excellent for narrative+code notebooks, but the desired widget interactions (predict-then-reveal, custom slider-to-Plotly wiring, hidden-answer `<details>`) are bespoke UX, not a notebook-execution story; would fight the tool more than use it |

**Decision:** `book/site/` is the deployable artifact — plain HTML files, one shared
`css/book.css` and `js/book.js`, vendored third-party libraries under `book/site/vendor/`.
Every page is openable standalone (modulo the `fetch()`/`file://` CORS caveat in §5) and the
whole tree copies verbatim into GitHub Pages' `_site/book/`.

**Vendored libraries:**
- **Plotly** — copied byte-for-byte from
  `.venv/lib/python3.13/site-packages/plotly/package_data/plotly.min.js` (plotly **6.6.0**, the
  exact version pinned in `pyproject.toml`/`uv.lock`). This is *better* than a fresh CDN/npm
  download: the JS build is guaranteed to match the Python `plotly` version the repo's own
  generators use, so a figure spec built server-side and one built client-side agree on schema.
  4.7 MB, `book/site/vendor/plotly/plotly.min.js`.
- **KaTeX 0.16.11** (not MathJax) — chosen over MathJax for offline weight and rendering model:
  KaTeX is synchronous/instant (no reflow-on-load flash), ships a small, fully self-contained
  JS+CSS+webfont bundle (~1.5 MB total incl. all font weights) ideal for a `file://`/no-CDN
  constraint, and its `auto-render` extension gives the same `$...$`/`$$...$$` delimiter UX
  authors expect from Sphinx/MathJax with zero runtime dependency resolution. MathJax's async,
  multi-pass typesetting is a better fit when equation *density* or accessibility (MathML output)
  dominates; this book's math load is modest (motivating equations per chapter, not a full
  derivation appendix), so KaTeX's speed and vendoring simplicity win. Fetched via `npm install
  katex@0.16.11 --no-save` into a scratch directory, then `dist/{katex.min.js,katex.min.css,
  fonts/,contrib/auto-render.min.js}` copied into `book/site/vendor/katex/` (MIT license, noted
  in `vendor/katex/VENDORED.txt`).

Both vendor directories carry a `VENDORED.txt` provenance note (what, which version, why, whose
build). No other runtime dependency.

### 2.2 Data pipeline pattern — DECIDED

`book/generators/*.py` are plain Python modules with a `main() -> None` entry point, run with
the **main repo's synced venv** (`/home/jasper/Repositories/MasterThesisCode/.venv/bin/python`,
or this worktree's own `.venv` once `uv sync` is run here) so `import master_thesis_code` and
heavy deps (`numpy`, `few`, ...) resolve. `book/generators/make_all.py` is the single driver —
imports and runs every registered generator in order; **idempotent and re-runnable** (verified:
running `gen_ch00_demo.py` twice produces byte-identical JSON, since the RNG seed and source
data are fixed).

Key portability decision made *during* this build (not in the original brief, but required for
correctness): **the repo root is resolved relative to the generator script's own path**
(`Path(__file__).resolve().parents[2]`), never hardcoded to `/home/jasper/...`. This was
verified necessary and safe because `results/campaign51_20260728/` — the "bulk data" the brief
describes as living in the main repo — is actually **git-tracked** (`git ls-files` confirms 817
tracked files under that path) and therefore identical and present in *any* checkout of this
branch, including a fresh GitHub Actions runner. Only the Python *interpreter* (the venv with
compiled/heavy deps) is checkout-specific; the *data* is not. Re-running the demo generator with
the fixed path produced identical output to the original hardcoded-path version (1588/1588
events, MAP trajectory unchanged) — confirming the fix is behavior-preserving.

Generators emit compact JSON (the ch00 demo: 41 h-values × 11 subset sizes = 16 KB) to
`book/site/data/*.json`. These files are committed (reproducible from `make_all.py`, but also
just data — no reason to regenerate on every Pages build when nothing changed; the CI step
regenerates them anyway as a freshness check, see §2.4).

Closed-form math (Gaussians, the volume element $dV_c/dz$, simple algebra) is computed directly
in JS in the browser where the brief calls for it — no generator round-trip needed for curves a
reader can derive by typing the formula into `book.js`. The ch00 demo needed neither: its whole
point is a *real*, non-closed-form quantity (a product of 1588 measured likelihood curves), so
it is generator-computed by construction.

### 2.3 Widget pattern — DECIDED

`book/site/js/book.js` (~200 lines, no dependencies beyond the vendored Plotly/KaTeX globals it
calls into): `Book.theme` (light/dark toggle, `prefers-color-scheme` default +
`localStorage`-persisted override, mirrors the Artifact-runtime convention of a `data-theme`
attribute on `<html>`), `Book.renderMath()` (KaTeX `auto-render` wrapper), `Book.loadJSON()`
(tiny fetch+cache), `Book.gridSlider()` (binds an `<input type="range">` to an arbitrary array of
precomputed "steps", redrawing via a caller-supplied callback — used here to redraw a Plotly
trace via `Plotly.react`), `Book.predictReveal()` (wires a button row + hidden `.reveal` block
for the predict-then-reveal pattern), `Book.markCurrentNav()` (nav highlighting). All chapter
pages get theme+math+nav for free by including the one script tag.

### 2.4 CI integration — DECIDED, minimal diff

One new step added to the existing `pages` job in **this worktree's**
`.github/workflows/ci.yml` (never touch the main worktree's copy), placed right after the
existing "Generate interactive figures" step and before "Upload Pages artifact":

```diff
       - name: Generate interactive figures
         run: |
           mkdir -p _site/interactive
           cp interactive/*.html _site/interactive/
           if [ -d cluster_results ]; then
             uv run python -m master_thesis_code cluster_results --generate_interactive _site/interactive/ || true
           fi
         continue-on-error: true

+      - name: Build discovery book
+        run: |
+          uv run python book/generators/make_all.py
+          mkdir -p _site/book
+          cp -r book/site/. _site/book/
+        continue-on-error: true
+
       - name: Upload Pages artifact
```

No new job, no new dependency-install step needed (the `pages` job already runs
`uv sync --extra cpu --extra dev`, which is sufficient since `master_thesis_code` and its heavy
deps are exactly what `book/generators/*.py` import). `continue-on-error: true` matches the
existing interactive-figures step's philosophy: a book-generator bug must never block the
Sphinx-docs deploy. The book lands at `https://jasperseehofer.github.io/MasterThesisCode/book/`.
**This diff has been applied in this worktree's `.github/workflows/ci.yml`** (verified present)
— it does not exist in `/home/jasper/Repositories/MasterThesisCode`.

### 2.5 Layout/style — DECIDED

`book/site/css/book.css` reuses the exact hex values from `_colors.py`'s `METHOD`/`PLANCK_BAND`/
`SHOES_BAND`/`PRIOR` tokens as CSS custom properties, so book prose/callouts and any embedded
Plotly figure read as one palette. Typography: system-font stack, `72ch` measure
(`main.book-content { width: min(72ch, 100% - 2.5rem) }`), mobile breakpoint at 640px, a
`@media print` rule that hides interactive chrome (topbar, sliders, predict buttons) for a clean
print/PDF export. Light/dark: `prefers-color-scheme` is the default signal; a `data-theme`
attribute on `<html>` (set by the toggle button, persisted via `localStorage`) overrides it in
either direction, following the same contract used elsewhere for Artifact theming.

---

## 3. Skeleton built (this session)

```
book/README.md                                  — quick-start + chapter-authoring guide
book/design/BOOK_TECH_DESIGN.md                 — this file
book/generators/make_all.py                     — driver (imports + runs each generator)
book/generators/gen_ch00_demo.py                — real generator (see §2.2)
book/site/index.html                            — landing page / table of contents
book/site/ch00-demo.html                        — working chapter: Bayes' rule + live widget
book/site/css/book.css                          — shared stylesheet (palette, layout, widgets)
book/site/js/book.js                            — shared widget library
book/site/data/ch00_demo.json                   — generator output (16 KB, committed)
book/site/vendor/plotly/plotly.min.js           — vendored, version-matched to pyproject.toml
book/site/vendor/plotly/VENDORED.txt
book/site/vendor/katex/{katex.min.js,katex.min.css,auto-render.min.js,fonts/*,VENDORED.txt}
.github/workflows/ci.yml                        — +8-line diff (§2.4), this worktree only
```

`ch00-demo.html` is the proof-of-chain page requested in the brief: it states Bayes' rule and
the log-space combination rule (both cited, `combine_log_space()` in
`posterior_combination.py`, not re-derived), renders both via KaTeX, and drives one Plotly trace
with a slider bound to 11 precomputed "stack the first $N$ (of 1588) real events" combined
posteriors — generated from one real delivered posterior-JSON set
(`results/campaign51_20260728/realistic_20260729/seed61000/real_r1/posteriors/`), reusing the
project's own `load_posterior_jsons` / `build_likelihood_array` / `apply_strategy` (physics-floor)
/ `combine_log_space` functions verbatim. A predict-then-reveal block and two self-check
questions (hidden-answer `<details>`) close the chapter, per the brief's required patterns.

---

## 4. Verification performed

1. **Data generation** — `uv run`-equivalent invocation (`/home/jasper/Repositories/
   MasterThesisCode/.venv/bin/python book/generators/make_all.py`) ran cleanly, wrote
   `book/site/data/ch00_demo.json` (1588/1588 complete events, 11 subset sizes). Sanity-checked
   the MAP trajectory numerically: $N{=}1$ MAP lands at a grid edge (0.62), small-$N$ MAPs jump
   around (0.86 at $N{=}5,10$ — real sampling noise, not a bug, and exactly the point of the
   chapter's predict-then-reveal block), and $N{=}1588$ (full sample) MAP is 0.74 — consistent
   with this campaign's known result (`CLAIM_2D_BIAS_20260730.md`). Re-ran after the
   hardcoded-path → repo-relative-path fix (§2.2): byte-for-byte identical output, confirming the
   portability fix didn't change behavior.
2. **Path hygiene** — `grep -rn "http://\|https://\|cdn\."` over `book/site/{index.html,
   ch00-demo.html,css/,js/}` returns nothing (no external network reference anywhere); a second
   grep for any leaked `/home/jasper` absolute path in the same tree also returns nothing.
3. **Serving smoke test** — `python3 -m http.server` from `book/site/`, then `curl` against
   `index.html`, `ch00-demo.html`, `data/ch00_demo.json`, `vendor/plotly/plotly.min.js`,
   `vendor/katex/{katex.min.js,katex.min.css,auto-render.min.js}`, and one representative
   `vendor/katex/fonts/*.woff2` — **all returned HTTP 200** with purely relative paths, i.e. the
   exact layout GitHub Pages will serve at `/MasterThesisCode/book/...`.
4. Browser-rendered correctness (does the slider *look* right, does KaTeX actually typeset)
   could not be tested in this text-only environment — flagged as a risk below, not silently
   assumed.

---

## 5. Risks / follow-ups

- **`file://` + `fetch()` CORS (real limitation, not fully "offline" in the strictest sense).**
  Chromium-family browsers refuse `fetch()` of local files opened via plain `file://` due to a
  same-origin-policy quirk (`file://a` and `file://b` are cross-origin). Firefox does not have
  this restriction for local files. Static prose/math pages (no `data/*.json` fetch) work
  identically either way; **any data-driven widget page requires either a local HTTP server or
  Firefox** for a true double-click `file://` open. This is documented in `book/README.md` and is
  exactly why the brief's own verification step (4c) asks for a `python -m http.server` smoke
  note rather than a raw `file://` open — treated here as the intended, accepted interpretation
  of "offline" (no external network dependency; a trivial local server is fine), not silently
  glossed over.
- **No browser-rendered visual verification yet.** The KaTeX delimiters, Plotly trace styling,
  and dark-mode contrast have been reasoned through and unit-smoke-tested (files exist, parse,
  serve) but not eyeballed in an actual browser in this session. First priority for the next
  session working on this book: open `ch00-demo.html` in a real browser (via the http.server
  recipe) and visually confirm math rendering, slider responsiveness, and theme-toggle contrast
  in both light and dark.
- **`plotly.min.js` is 4.7 MB.** Fine for a single-page load (one-time cache), but if the book
  grows to many chapter pages each embedding a full Plotly figure, consider Plotly's `plotly-
  basic`/`plotly-cartesian` slimmer partial bundles (this repo's installed wheel only ships the
  full bundle; a slim bundle would need a separate npm-sourced vendor step, currently not done —
  flagged, not implemented, since chapter count is still 1).
  Note: this repo's `interactive/` figures ARE full `plotly.min.js` size already; consistent.
- **`docs_src/interactive/`** — an apparently-stale partial duplicate of `interactive/` (5 of 10
  files, not referenced by CI) was noticed during inspection but is out of scope for this task;
  flagged for a future cleanup pass, not touched here.
- **Napoleon vs. NumPy-style docstring drift** in `docs/source/conf.py` (`napoleon_google_
  docstring = True` while `CLAUDE.md` mandates NumPy-style for new code) — pre-existing,
  unrelated to the book, noted for completeness only.
