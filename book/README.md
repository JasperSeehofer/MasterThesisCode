# The EMRI Dark Siren Discovery Book

An interactive, step-by-step journey from Bayes' rule to the full EMRI dark-siren
H₀ inference pipeline, published alongside the thesis on GitHub Pages.

See [`design/BOOK_TECH_DESIGN.md`](design/BOOK_TECH_DESIGN.md) for the full architecture
write-up (rationale, trade-offs, CI diff). This file is the short "how do I run it" version.

## Layout

```
book/
  README.md                 <- you are here
  design/
    BOOK_TECH_DESIGN.md      <- architecture decision record
  generators/                <- Python data generators (read the main package + results/, read-only)
    make_all.py               <- driver: runs every generator in order
    gen_ch00_demo.py           <- ch00 demo: real posteriors -> data/ch00_demo.json
  site/                      <- the deployable artifact (this is what ships to GH Pages)
    index.html
    ch00-demo.html
    css/book.css
    js/book.js
    data/*.json                <- generator output, committed (reproducible, but also just data)
    vendor/plotly/plotly.min.js <- copied from the repo's pinned plotly wheel (see VENDORED.txt)
    vendor/katex/               <- KaTeX 0.16.11 + fonts + auto-render (see VENDORED.txt)
```

## Regenerating the data

```bash
# from the repo root, using a venv with master_thesis_code installed
/home/jasper/Repositories/MasterThesisCode/.venv/bin/python book/generators/make_all.py
# or, once this worktree has its own `.venv` (uv sync --extra cpu --extra dev):
uv run python book/generators/make_all.py
```

Generators are deterministic (fixed seeds) and read-only against `master_thesis_code/` and
`results/` — never edit the package from here.

## Viewing locally

```bash
cd book/site
python3 -m http.server 8000
# open http://127.0.0.1:8000/
```

A plain double-click (`file://.../index.html`) works for static prose/math, but the
data-driven widgets use `fetch()` to load `data/*.json`, which Chromium-family browsers
block under `file://` (CORS). Use the local server above, or Firefox, for full-fidelity
local testing. GitHub Pages serves over HTTPS, so production is unaffected.

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
   `_template.html`, `index.html`, `generators/make_all.py`, `vendor/`,
   `.github/workflows/ci.yml`. Missing shared capability? Append to
   `design/WIDGET_REQUESTS.md` and use a page-local inline workaround in your own file.
