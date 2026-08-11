# The EMRI Dark Siren Discovery Book

An interactive, step-by-step journey from Bayes' rule to the full EMRI dark-siren
H₀ inference pipeline, published alongside the thesis on GitHub Pages.

See [`design/BOOK_TECH_DESIGN.md`](design/BOOK_TECH_DESIGN.md) for the full architecture
write-up (rationale, trade-offs, CI diff). This file is the short "how do I run it" version.

## Layout

```
book/
  README.md                 <- you are here
  BUILD_REPORT.md            <- integration + QA report (incl. the revision pass)
  design/
    BOOK_TECH_DESIGN.md      <- architecture decision record
    BOOK_DESIGN.md           <- authoritative build spec (chapters, ownership contract)
    REVISION_WORKLIST.md     <- the post-review revision spec
    flags/                   <- per-chapter flag files (append-only historical record)
  generators/                <- Python data generators (read the main package + results/, read-only)
    make_all.py               <- driver: auto-discovers and runs every gen_ch*.py / gen_museum*.py
    qa_gates.py               <- content build gates (run by make_all.py; runnable standalone)
    gen_ch00.py … gen_ch11.py, gen_museum.py
  site/                      <- the deployable artifact (this is what ships to GH Pages)
    index.html
    ch00-two-numbers.html … ch11-honest-state.html, museum.html
    _template.html             <- reference-only chapter skeleton (not linked, not shipped as a page)
    css/book.css
    js/book.js, js/manifest.js
    data/*.json                <- generator output, committed (reproducible, but also just data)
    vendor/plotly/plotly.min.js <- copied from the repo's pinned plotly wheel (see VENDORED.txt)
    vendor/katex/               <- KaTeX 0.16.11 + fonts + auto-render (see VENDORED.txt)
```

## Regenerating the data

```bash
# from the repo root, using a venv with darksiren_emri installed
/home/jasper/Repositories/MasterThesisCode/.venv/bin/python book/generators/make_all.py
# or, once this worktree has its own `.venv` (uv sync --extra cpu --extra dev):
uv run python book/generators/make_all.py
```

Generators are deterministic (fixed seeds) and read-only against `darksiren_emri/` and
`results/` — never edit the package from here. `make_all.py` runs each generator in its own
subprocess and then executes the `qa_gates.py` content gates against the built site; a gate
hit fails the build loudly.

## Viewing locally

```bash
cd book/site
python3 -m http.server 8000
# open http://127.0.0.1:8000/
```

A plain double-click (`file://.../index.html`) works for static prose/math, but the
data-driven widgets use `fetch()` to load `data/*.json`, which Chromium-family browsers
block under `file://` (CORS). Use the local server above, or Firefox, for full-fidelity
local testing — under `file://` the widgets degrade to their static `<noscript>` copies
rather than failing silently. GitHub Pages serves over HTTPS, so production is unaffected.

## Adding a new chapter (build-phase rules)

The authoritative build spec is **`design/BOOK_DESIGN.md`** — chapter titles, content,
interactives, sources, and the file-ownership contract all live there. Mechanics:

1. Write your generator(s) as `book/generators/gen_chNN*.py` — `make_all.py`
   **auto-discovers** `gen_ch*.py` / `gen_museum*.py`; never edit `make_all.py`.
   Generators read the main package + `results/` read-only and write only
   `book/site/data/chNN_*.json`.
2. Copy `book/site/_template.html` to `book/site/chNN-<slug>.html` (exact filenames in
   `js/manifest.js`) and fill the slots. The nav is built from `js/manifest.js`
   automatically — do not hardcode chapter links anywhere.
3. **Frozen files** (integrator-only): `css/book.css`, `js/book.js`, `js/manifest.js`,
   `_template.html`, `index.html`, `generators/make_all.py`, `generators/qa_gates.py`,
   `vendor/`, `.github/workflows/ci.yml`. Missing shared capability? Append to
   `design/WIDGET_REQUESTS.md` and use a page-local inline workaround in your own file.

## CI note — external-reference audits must scope out `vendor/`

The "relative refs only / zero external URLs" audit applies to first-party files
(`site/*.html`, `site/css/`, `site/js/book.js`, `site/js/manifest.js`, `site/data/`).
The vendored `plotly.min.js` bundles dead default URLs for its geo/mapbox subsystem
(openstreetmap, mapbox, cartocdn, cdn.plot.ly); no shipped trace type ever reaches them
(verified — no `scattermapbox`/`scattergeo`/`choropleth` anywhere in the book). A future
automated CSP/external-ref check must therefore grep first-party files only, or it will
false-positive on `vendor/` forever. (ux review, 2026-07-31; REVISION_WORKLIST §D-11.)
