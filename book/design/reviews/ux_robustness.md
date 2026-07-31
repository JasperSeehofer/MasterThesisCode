# UX / Accessibility / Robustness Review

Reviewer: mechanical audit pass (serve + curl + node + static analysis), 2026-07-31.
Scope: `book/site/` as shipped (14 pages + `_template.html`), `js/book.js`, `js/manifest.js`,
`css/book.css`, `vendor/`. Read-only; no git operations performed.

Method: served the site (`python3 -m http.server 8123`), curled every page and every
`data/*.json`/asset reference, ran `node --check` on `js/book.js`, `js/manifest.js`, and all
14 inline `<script>` blocks extracted per page, parsed every page with `html.parser` for
tag-balance, grepped for external references, computed WCAG contrast ratios for the
theme-dependent badge colors, and traced `localStorage`/fetch/predict-lock code paths in
`book.js`. Findings below are ranked; every item that follows a BUILD_REPORT §5/§7 flag says
so explicitly, otherwise it's new.

---

## [MAJOR] Every data-driven widget has zero `.catch()` — a failed fetch fails silently, with no on-page signal, on 12 of 13 chapter pages

`Book.loadJSON()` (`js/book.js:123-131`) rejects its promise on both network failure and
non-2xx status. Grepping every chapter page for `.catch(` on the `Book.loadJSON(...)` chain:

```
ch01–ch11, museum.html: 0 matches each
ch00-two-numbers.html:  1 match — logs `console.error(...)`, nothing rendered to the page
```

So the *only* page with a `.catch()` still degrades to a silent console log, and every other
page has no handler at all — an unhandled promise rejection. The widget containers are plain
empty divs (e.g. `<div id="ch01-strain-plot" class="widget-plot" style="min-height:240px">
</div>`, `ch01-ruler.html:235`): on a fetch failure the reader sees a blank box of fixed
height next to prose that assumes a chart is there, with **no** error text, no "reload",
nothing. This is a distinct failure mode from "no JS" (which the `<noscript>` blocks handle
well, see PRAISE below) — this is "JS ran, fetch broke," and it is completely unhandled.

This is not a hypothetical: the README itself documents that `fetch()` is blocked under
`file://` in Chromium (`book/README.md:49-51`), and the CI note in BUILD_REPORT §5.5 item 24
/ §6 describes generators that fall back to committed JSON precisely because some source
data isn't always present — i.e., there are real, already-acknowledged paths by which a
`data/*.json` could legitimately be stale, missing, or fail to load in production. A reader
who double-clicks `index.html` (the single most natural thing to do with a downloaded book)
gets 30+ blank boxes and no explanation, in a book whose stated core ethic is that "nothing
interactive lets a dead hypothesis look alive" — a widget that fails this way isn't lying,
but it is uninformatively silent, which cuts against the same design goal.

**Fix:** give `Book.loadJSON` (or a `Book.widget(id, url, render)` wrapper) a shared
`.catch()` that swaps the container's content for exactly the adjacent `<noscript>` text (or
a one-line "Data failed to load — see the static fallback above/below" pointing at it), so
the existing, already-well-written fallback copy actually reaches the reader when it's
needed instead of only satisfying `<noscript>`.

---

## [MAJOR] Badge/chip accent colors have no `prefers-color-scheme` fallback — first-time OS-dark-mode visitors get sub-AA contrast text

`css/book.css` has exactly one `@media (prefers-color-scheme: dark)` block (line 41), and it
only covers the base tokens (`--accent`, `--ink`, `--paper`, `--border`, `--muted`,
`--code-bg`). Everything else that changes for dark mode is gated *only* behind the
manually-toggled `:root[data-theme="dark"]` attribute (lines 326, 347–351, 569, 699):

```
:root[data-theme="dark"] .badge.ratified   { color: #35d0a5; }
:root[data-theme="dark"] .badge.candidate  { color: #E69F00; }
:root[data-theme="dark"] .badge.refuted,
:root[data-theme="dark"] .badge.exonerated { color: #ff8b47; }
:root[data-theme="dark"] .badge.confounded { color: #e096bf; }
```

`Book.theme.init()` (`js/book.js:86-96`) only sets the `data-theme` attribute if
`localStorage` already has a saved preference. A brand-new visitor whose OS is set to dark
mode gets the *dark* base tokens (via the media query) but the *light-tuned* badge/chip
colors (via the CSS cascade default), because the attribute-gated overrides never fire until
they click the toggle once. Measured contrast of the light-tuned colors against the dark
paper (`#14161a`, sRGB relative luminance, WCAG formula):

| element | light-tuned (used) | contrast vs `#14161a` | dark-tuned (only reachable after toggling) | contrast |
|---|---|---:|---|---:|
| `.badge.ratified` | `#007a5c` | **3.40:1** | `#35d0a5` | 9.24:1 |
| `.badge.candidate` | `#9c6d00` | **3.97:1** | `#E69F00` | 8.04:1 |
| `.badge.refuted`/`.exonerated` | `#a44900` | **3.05:1** | `#ff8b47` | 7.79:1 |
| `.badge.confounded` | `#a24e7f` | **3.39:1** | `#e096bf` | 8.01:1 |

All four fall well below WCAG AA's 4.5:1 threshold for the small (`0.66rem`) badge text used
for `RATIFIED`/`CANDIDATE`/`REFUTED`/`EXONERATED`/`CONFOUNDED` — the exact status vocabulary
the book uses everywhere to tell the reader how much to trust a claim. This is the one part
of the theme system that's supposed to be read carefully, and it's the part that silently
degrades for anyone whose first visit happens to be in dark mode.

**Fix:** wrap the same badge/chip/note overrides in a
`@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { ... } }` block
mirroring the base-token pattern already used at line 41-49, so the fallback and the
explicit toggle produce identical output.

---

## [MAJOR] ch02, ch08, ch09 eagerly render 7–8 Plotly instances on page load — no lazy init, unlike ch03's own precedent in the same codebase

Counting `Book.themedPlot(` call sites (each one a `Plotly.newPlot`) per page:

```
ch08-mass-channel.html:     8
ch09-universe-factory.html: 8
ch02-bayes.html:            7
ch07-redshift.html:         5
ch05-unseen-galaxy.html:    6
museum.html:                6
(all others: ≤5)
```

None of these three pages defer plot construction — no `IntersectionObserver`, no
toggle-on-open gating (`grep addEventListener("toggle"` returns nothing on any of them). All
8 plots on ch08/ch09 and all 7 on ch02 construct on initial load. `ch03-which-galaxy.html` is
the *only* page in the book that uses `IntersectionObserver` (2 hits) — precedent already
exists in this codebase for exactly this problem (ch03 also happens to ship the single
largest data file, 287 KB `ch03_skyball.json`, so the same team clearly identified the cost
of an expensive widget and fixed it there but not on ch02/ch08/ch09).

**Fix:** apply the same `IntersectionObserver` pattern ch03 already uses to gate
`Book.themedPlot` calls on ch02/ch08/ch09 behind visibility, or at minimum behind their
enclosing `<details>` opening where the widget sits inside a collapsed disclosure.

---

## [MINOR] Two separate `@media print` blocks in `css/book.css` (lines 498, 776)

`@media print { #bias-rail { display: none !important; } }` (498) and a second full print
block at 776 that hides the topbar/persona/passport/predict chrome and forces
`color:#000; background:#fff`. Not a bug — both apply correctly — but they should be merged
into one block for maintainability; a future editor extending "hide on print" is likely to
add a third block instead of finding the other two.

## [MINOR] `Book.themedPlot` registers one `MutationObserver` per plot, never disconnected

`js/book.js:407-419`: every call to `Book.themedPlot` creates a new
`MutationObserver(...).observe(document.documentElement, {attributes:true, ...})` watching
the same `data-theme` attribute, and never calls `.disconnect()`. On ch08/ch09 (8 plots) that
is 8 independent observers all firing `Plotly.relayout` on every theme toggle — harmless
functionally (each relayout is idempotent) but it's 8x the necessary work per toggle and 8x
the retained-object count for the lifetime of the page. Not worth a redesign, but a shared
single observer that iterates registered plot ids would be strictly better and is a small
change against the existing `Book.themedPlot` return-value pattern.

## [MINOR] `_template.html` is unreachable and its own `Book.loadJSON("data/chNN_widget1.json")` path is a genuine 404 — correctly, since it's a placeholder, but worth a guard comment

`_template.html` is not linked from `index.html`, `manifest.js`, or any chapter (confirmed by
grep) — it is reference-only, exactly as BUILD_REPORT §1 states, so no reader can hit the
404. No fix needed, but the file could use one line noting the JSON path is a placeholder to
avoid a future contributor treating the console fetch error (if they open the template
directly) as a real bug — very low priority.

## [MINOR] Vendored `plotly.min.js` contains dead references to external map tile hosts

`vendor/plotly/plotly.min.js` bundles default URLs for `a.tile.openstreetmap.org`,
`api.mapbox.com`, `cartocdn.com`, `cdn.plot.ly/un/`, etc. (Plotly's built-in geo/mapbox
subsystem). Confirmed these paths are **never** reached: no `scattermapbox`/`scattergeo`/`choropleth`
trace types appear anywhere in the shipped HTML or `book.js`. This is standard, unavoidable
Plotly-bundle content, not a live external reference — the "zero external refs" audit item
passes in practice, but a strict `grep -r "https://"` over `vendor/` will flag these strings,
so future automated CSP/external-ref checks should scope the grep to first-party files
(`*.html`, `css/`, `js/book.js`, `js/manifest.js`) rather than `vendor/` verbatim, or they'll
false-positive forever.

---

## PRAISE — what works, keep it

- **No-JS story is genuinely strong, not a token gesture.** Sampled noscript fallbacks across
  10 of 13 chapters (ch00–ch02, ch03, ch05, ch07, ch08–ch11, museum): every single one states
  the actual measured numeric answer in prose (e.g. ch07's noscript literally states "Of the
  76 in-catalogue hosts... only 1 has an indicative σ_z/z at or below the 0.256 rail
  threshold"), not a placeholder. Zero pages had a "please enable JavaScript" non-answer.
  This is exactly what a book about honesty under failure should do, and it's applied
  uniformly rather than on a couple of showcase widgets.
- **Predict-locks structurally cannot lock out a no-JS reader.** `.is-predict-locked`
  (`css/book.css:711`) is only ever added via `classList.toggle` inside `js/book.js:242` —
  never present as static markup — so with JS disabled the class is never applied and gated
  reveals are simply never gated. Verified by grep, not just by reading the design doc's
  claim: the mechanism matches the claim.
- **Doctype/charset/quirks-mode discipline is complete.** All 14 shipped pages plus
  `_template.html` open with `<!DOCTYPE html>` and `<meta charset="utf-8">` inside the first
  ~200 bytes; parsed all 14 with Python's `html.parser` for tag-balance and found zero
  mismatched/unclosed tags and zero orphaned closing tags across the whole site.
- **KaTeX delimiter discipline is completely clean.** Zero `\(`/`\[` usages anywhere in
  shipped HTML (only `$...$`/`$$...$$`); every page's dollar-sign count is even, i.e. no
  orphan delimiter that would break rendering or leak raw LaTeX to the reader.
- **Zero real external references.** Full grep across every shipped HTML/CSS/JS file for
  `http(s)://` found no live network dependency; the only hits are inert unused code paths
  inside the vendored Plotly bundle (see MINOR above) and the KaTeX SVG namespace URIs
  (`http://www.w3.org/2000/svg`), which are XML namespace strings, not network requests.
- **Keyboard accessibility on the Symbol Passport is properly wired, not an afterthought.**
  `.term[data-term]` elements get `tabindex="0"`, `aria-label`, a `focus` listener alongside
  `mouseenter`/`click`, `Escape`-to-dismiss on `document`, and click-outside-to-close — this
  is a small popover component built to the accessibility bar a lot of production sites miss.
- **Every internal link and museum cross-reference resolves.** Verified programmatically:
  all `href`/`src`/`fetch` targets across all 14 pages resolve to files that exist (0
  broken); all 14 museum exhibit `id="ex-*"` anchors exist and every chapter's
  `museum.html#ex-*` link targets one of them; all 19 `BOOK_PREDICT_IDS` registry entries
  match an id actually written by its source chapter, with 0 unresolved.
- **`localStorage` key namespacing is clean.** Four independent key families —
  `book-theme`, `book-persona`, `book-glossary`, `book-predict:<id>` — with no collisions
  found; the predict-id space is further namespaced per-chapter (`ch03-host-guess` etc.) and
  cross-checked against the manifest registry, so the deliberate ch03→ch11 recall works by
  construction rather than by convention.
- **`file://` limitation is disclosed, not discovered the hard way.** `book/README.md:49-51`
  states plainly that `fetch()`-driven widgets don't work under Chromium's `file://` CORS
  restriction and tells the reader to use the local server or Firefox — this is exactly the
  kind of honesty the book asks of its own physics content, applied to its own delivery
  mechanism. (Undercut somewhat by the missing `.catch()` above: the doc promises a
  degrade-gracefully story that the code doesn't quite deliver for that specific case.)
- **`node --check` is clean everywhere.** Both shared JS files and all 14 pages' inline
  `<script>` blocks (extracted and checked individually) parse with zero syntax errors.
- **Data budget is real, not aspirational.** Directly measured: 36 files, max 288 KB
  (`ch03_skyball.json`), total 1.30 MiB — matches BUILD_REPORT's figures exactly.

---

## Notes on the mechanical checklist (for completeness)

- **HTTP smoke:** all 14 pages return 200; every `data/*.json` referenced via
  `Book.loadJSON(...)` (33 distinct paths across 13 pages) returns 200 (the one 404,
  `data/chNN_widget1.json`, is `_template.html`'s placeholder and unreachable from any live
  page — not a real broken link).
- **Page weight:** per-chapter total (own HTML + own data files, shared `js/book.js` +
  `js/manifest.js`, and the vendor bundle counted once) ranges ~5.14 MiB (ch00) to ~5.52 MiB
  (ch03), and **~4.85 MiB of that is `plotly.min.js` alone**, paid once per session since it's
  the same URL on every page (browser HTTP cache) but a real first-page cost, especially over
  a slow link or the `file://` path where no cache-control semantics apply the same way. This
  is an accepted, deliberate trade-off for the site's genuine zero-CDN/offline-safe goal, not
  a defect — flagging it here only so the size is visible and measured rather than assumed.
- **Mobile rules:** `@media (max-width: 1180px/900px/720px/640px)` all present; `pre`,
  `.table-scroll`, and `.katex-display` all carry `overflow-x: auto`; spot-checked every
  chapter's `<table>` count against `.table-scroll`-wrapped count — the one unwrapped table
  per chapter in every case is the per-chapter `.dossier` accumulator table (2-column
  label/value, `white-space:nowrap` only on the label column), which doesn't carry the same
  overflow risk as the wide numeric tables that do get wrapped.
