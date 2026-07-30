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
- Status: ACCEPTED (integrator, phase 4)

### R-INT-2: "Has this been tried?" ledger search (BW3)
- A search box over the 98-row `BIAS_HISTORY_LEDGER.md` available inside sandboxes;
  volunteers the verdict when a sandbox configuration matches a historical hypothesis.
  Needs `data/museum_ledger.json` (owned by the museum agent). Chapter sandboxes SHOULD
  tag their toggle states with `data-hypothesis="<ledger row #>"` where a state matches
  a known dead hypothesis, and hard-code the reveal of that verdict locally (museum
  meta-rule: an interactive that lets the reader "try" a dead hypothesis must reveal the
  measured verdict, not leave it open).
- Status: ACCEPTED (integrator, phase 4)

### R-INT-3: Persona switch (Reading as Mara / Tomas / Examiner)
- Global control pre-expanding `details.gw-reader` and provenance panels. Pure chrome;
  chapters need only use the standard `gw-reader` / `provenance-panel` classes.
- Status: ACCEPTED (integrator, phase 4)

---

<!-- Chapter-agent requests go below this line. -->
