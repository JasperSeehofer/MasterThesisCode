# museum_FLAGS.md — the Defect Museum annex

Raised by the museum agent, 2026-07-31, per `BOOK_DESIGN.md` §4.1: *"if a generator's
recomputation disagrees with a spec number, **stop and flag; do not silently reconcile in
either direction**."*

Both items below are presented **on the page in both forms**, side by side, with the
disagreement named. Nothing was adjusted, dropped, or averaged.

---

## F-museum-1 — ⚠⚠ The flagship exhibit's stated mechanism does not reproduce: `volume_trunc`'s "`fixed_quad(n=50)` aliases the GW peak to 0.0000" is a **scalar-collapse artifact of the diagnostic script**, not a property of `fixed_quad(n=50)`

**This is the sharpest flag in the annex. It does not overturn any project verdict, and the
museum does not adjudicate it — but it must not be shipped silently.**

### What the spec and the artifacts say

- `BOOK_SOURCES_MAP.md` §4 exhibit 1, `BOOK_PEDAGOGY.md` B5 (Ch-7 interlude) and Part 4 M1,
  and `BOOK_DESIGN.md` §1 (Ch 7 interlude + the museum card) all state the flagship AHA as:
  *"at n = 50 the integral reads **0.0000** where the exact value is 0.24–0.65 — the peak
  falls between nodes"*, and *"two independent causes at once"*.
- The primary artifact is `results/volume_trunc_ab_20260712/FINDING.md:1-58`
  ("Mechanism (two compounding effects; `quadrature_diagnostic.py`)"), whose table reads:

  | h | numerator, GW window (n=50) | numerator, host window (n=50) | numerator, host window (exact quad) |
  |---|---|---|---|
  | 0.60 | 0.0003 | **0.0000** | 0.2417 |
  | 0.73 | 0.0005 | **0.0000** | 0.4314 |
  | 0.86 | 0.0007 | **0.0000** | 0.6537 |

  and concludes "**Quadrature aliasing (dominant).** … the sparse Gauss-Legendre nodes
  straddle the narrow GW peak and miss it (n=50 → 0.0 vs exact 0.24–0.65)".
- The same attribution is carried into production source comments:
  `bayesian_statistics.py:384` and `:3670` ("fixed_quad n=50 aliases the narrow GW peak over
  the wide host window").

### What `gen_museum.py` measures

Running `results/volume_trunc_ab_20260712/quadrature_diagnostic.py` unchanged reproduces its
published table **exactly** (verified, this session). Re-computing the *same integrand* with
the GW leg written the way **production** writes it does not:

| h | host-window `fixed_quad(n=50)`, GW leg via `dist()` (the diagnostic's form) | host-window `fixed_quad(n=50)`, GW leg via `dist_vectorized()` (production's form) | exact |
|---|---|---|---|
| 0.60 | 0.0000 | **0.2376** | 0.2417 |
| 0.73 | 0.0000 | **0.4412** | 0.4314 |
| 0.86 | 0.0000 | **0.6524** | 0.6537 |

n = 50 is accurate to **1.7–2.3%**, not to 100%.

### The mechanism of the disagreement (measured, not inferred)

`master_thesis_code/physical_relations.py:132 dist()` is **scalar-only**: given an array it
returns a 0-dimensional array holding the value at the array's *first* element.
`scipy.integrate.fixed_quad` passes the whole node array in one call, so inside the
diagnostic the GW factor becomes a **constant** — the likelihood at the window's lower
limit — and the node count then cannot matter.

That hypothesis is quantitative and it reproduces the diagnostic's **entire** table:

- host window, lower limit z = 0 ⇒ `d_L = 0` ⇒ distance fraction 0 ⇒
  `exp(−0.5·(1/0.05)²) ≈ 1.4×10⁻⁸⁷` ⇒ prints as `0.0000` at every h and **every n**
  (measured ladder, h = 0.73: 6.5e−79 at n = 10 … 1.1e−86 at n = 600 — flat in n);
- GW window, lower limit `z(d_L − 4σ_dL)` ⇒ fraction 0.8 ⇒ `exp(−8)` ⇒ predicted column
  **0.000265 / 0.000468 / 0.000701** at h = 0.60/0.73/0.86, i.e. **0.0003 / 0.0005 / 0.0007**
  to the printed 4 dp — the published GW-window column, digit for digit
  (`museum_quadrature.json.gates`, all `match_4dp: true`).

`master_thesis_code/bayesian_inference/bayesian_statistics.py:3806`
(`numerator_integrant_without_bh_mass`) and `:3826` (`denominator_integrant_without_bh_mass`)
both use `dist_vectorized`, so **the production path does not have this issue**, and the
diagnostic's zeros were never production numbers.

### What is and is not affected

- **NOT affected — the verdict.** `volume_trunc` is FALSIFIED on the seed600 494-event A/B
  gate (`gate_result.json`: 1D mean 0.7450 → 0.8000, 2D 0.7681 → 0.8000), which was run with
  production code. The museum presents that verdict unchanged.
- **NOT affected — mechanism (2).** "Even the exact host-window numerator tilts high" is
  reproduced here exactly (0.2417 → 0.4314 → 0.6537, monotone in h), and it is by itself a
  sufficient explanation of the collapse onto h = 0.80.
- **Affected — mechanism (1) and the "two independent causes" framing.** The evidence
  offered for the aliasing cause is the `0.0000` column, and that column is explained by the
  scalar collapse.
- **Still true — aliasing is real, at a different order.** The vectorized ladder at h = 0.73
  reads 1.504 (n = 10), 0.147 (15), 0.068 (20), 0.458 (25), 0.604 (30), 0.382 (40), 0.441
  (50), 0.4314 (75, converged) against an exact 0.4314 — erratic by factors 3.5× high and
  6.3× low, in both directions, exactly the "which nodes happen to catch the peak" behaviour
  the FINDING describes. The lesson ("quadrature is physics") is intact; only the node count
  at which it bites has moved.

### Disposition

- `museum.html` exhibit `#ex-volume-trunc` and interactive **M1** carry a two-state
  evaluation switch — *scalar `dist()` (2026-07-12 diagnostic)* vs *`dist_vectorized()`
  (production)* — so the reader sees both columns and the flat-in-n signature themselves.
  The recorded FINDING.md table is drawn as a labelled `rec` overlay and is never replaced.
- The page states in the narrator's voice that the book does **not** adjudicate this, links
  this flag, and leaves the project's verdict on `volume_trunc` standing.
- `gen_museum.py` enforces both reproductions as hard gates: it raises rather than writing a
  file if the exact column, or the scalar-collapse prediction of the GW-window column, ever
  stops matching FINDING.md to 4 dp.
- **For the author / integrator:** this is a candidate ledger entry in its own right (a
  falsified *mechanism attribution* inside a correct falsification), and it touches two
  production source comments (`bayesian_statistics.py:384`, `:3670`) that propagate the
  attribution. It is raised here, not resolved.

---

## F-museum-2 — commission injection scan (#49a): "tracks the truth exactly" vs a re-run that is one grid step low in 2 of 5 cells

- **Recorded:** ledger row **#49a** and `synthesis/WF2_DIGEST.md:26-30` /
  `synthesis/DRAFT_REPORT.md:24-27` — *"catalog_only MAP tracks truth **EXACTLY**
  (0.63→0.63 … 0.77→0.77); PRODUCTION MAP = 0.86 for EVERY injected truth"*.
- **Re-run by `gen_museum.py`** (importing `results/commission_20260701/injection_scan.py`
  and calling its own functions with its own seeds — `default_rng(2024)` for the catalogue,
  `default_rng(int(h·1000))` per injection):

  | injected truth | 0.63 | 0.67 | 0.70 | 0.73 | 0.77 |
  |---|---|---|---|---|---|
  | `catalog_only` MAP | 0.630 | **0.660** | **0.690** | 0.730 | 0.770 |
  | production MAP | 0.860 | 0.860 | 0.860 | 0.860 | 0.860 |

- **Assessment.** The headline — the production estimator's MAP is *independent of the
  injected truth* — reproduces exactly, and it is what the exhibit is about. The
  `catalog_only` control lands one 0.01 grid step below truth at two of the five injections,
  so "tracks the truth exactly" is a **grid-step-level** overstatement, not a different
  result. The digest's own parenthetical (`0.63→0.63 … 0.77→0.77`) quotes only the two
  endpoints, both of which are exact.
- **Disposition:** the page prints the **re-run** table with all five cells, quotes the
  digest's wording next to it, and says which two cells differ and by how much. No number
  was adjusted. The exhibit's claim is stated as *"the production MAP does not move with the
  truth; the control does"* — which both the record and the re-run support.
- **Venue note carried on the page:** this scan is the commission's own synthetic harness
  (20,000-galaxy catalogue, moderate completeness `f(z) = exp(−(z/0.3)²)`, an `erfc`
  detection horizon at 3.0 Gpc) — **not** the production catalogue, and the museum says so.

---

## F-museum-3 — build portability: two of the museum's sources are untracked

- `results/commission_20260701/**` (the WF digests, `DRAFT_REPORT.md`, `injection_scan.py`)
  is **not git-tracked** — it exists only in the working tree of the main checkout, so it is
  absent from this worktree and from a fresh CI clone. The ledger, the claim file,
  `results/volume_trunc_ab_20260712/`, `results/mass_trunc_ab_20260713/`,
  `docs/gates/G6_starvation_postmortem.md` and `docs/H0_BIAS_RESOLUTION.md` **are** tracked
  and present.
- **Disposition:** `gen_museum.py` resolves every artifact from this repo root first, then
  from a sibling `MasterThesisCode` checkout; if `injection_scan.py` is missing it prints a
  NOTICE and leaves the already-committed `museum_h0_independent.json` untouched rather than
  failing the build or writing a degraded file (the same pattern as `gen_ch04.py`, see
  `ch04_FLAGS.md` F-ch04-5).
- **For the integrator:** identical to F-ch04-5. Either those artifacts get committed, or
  every generator keeps the tracked-first / sibling-fallback / keep-committed-output pattern.

---

## F-museum-4 — no disagreement found (recorded for the audit trail)

Checked and reproduced without discrepancy: the ledger's own row count (**98** rows, ids
1–94 plus 49a/49b/49c/49d — matching the spec's "98 hypotheses"); the ledger §2 DO-NOT-RE-TRY
union (17 items, back-referencing 26 distinct ledger rows) and the claim file's 15-entry
Exonerated list, both parsed from the artifacts rather than transcribed; the `volume_trunc`
A/B posteriors (`gate_result.json`); the FINDING.md exact-quadrature column (see F-museum-1).
