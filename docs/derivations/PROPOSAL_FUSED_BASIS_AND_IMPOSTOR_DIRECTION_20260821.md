# JOINT DECISION PROPOSAL — production basis (off→fused), the fleet fused null, and the impostor-leg direction

**Date:** 2026-08-21 (late) · **Status:** PROPOSED — all items await author ruling
**Trigger:** row #157 item 3 deferred the production-basis fork "jointly with the impostor-leg
question, after the fused confirmation seed reports, via a reviewable physics-change proposal";
row #158 items 3–4 opened the fleet-null [DO] and advanced the landscape chain to its third link.
This is that proposal. Template: `PROPOSAL_2D_SELECTION_FUSION_20260817.md`.

## 0. Where this sits

O6 (row #158) confirmed the mechanism in the production code path: the `off` cell's matched-channel
violation is the omission of the S̄_φ survival factor from the completion numerator, and the
in-tree `fused` cell is numerically identical to the mechanism's prediction (+1.94e-6, storage-
dominated). Three coupled questions were left with the author, plus two carried items this
proposal folds in because their resolution depends on the same rulings:

- **[D1]** Is the fleet-level "fused nulls the matched channel" claim closed, and how?
- **[D2]** Do runs-of-record switch basis off→fused (physics-change-gated)?
- **[D3]** What is the direction on the impostor catalogue leg — the dominant production channel?
- **[D4]** (carried, runbook 23) systematics row 16 re-grade.
- **[D5]** (conditional on D2) landscape/T1 resubmission — the gated sweep is itself 13 *fused*
  cells, so it presupposes D2's answer.

## 1. The items

### [D1] The fleet-level fused null — close by transfer, by fleet, or leave open

**Evidence in hand.** The A20/O4 restored-arm fleet: S̄₁₅ = **+0.0076 ± 0.0184** (0.41σ from
zero), per-seed shift +0.1249 sd 0.0046 (`A20_REVIEW_O4_20260821.md`). O6 proved the transfer:
the production `fused` dispatch equals the harness convention at machine precision for seed
910101 (delta +1.94e-6, 100% storage; per-event B_num agreement 3.5e-15), and the identity is
*structural* — one dispatch (`bayesian_statistics.py:5157-5170`), the same for every seed; only
the data varies. The harness convention gap (aligned vs production numerator, with S̄_φ) measured
1.75e-5 on seed 910101 — negligible against the fleet SEM 0.0184.

**A gap this fixes either way:** the restored-arm 15-per-seed vector is currently banked only as
prose fleet statistics — the computing scripts lived in the reviewer's session scratchpad. Any
option below starts by banking a machine-readable per-seed vector from committed code.

| option | what runs | cost | what it claims |
|---|---|---|---|
| **A (recommended)** | Extend the committed `o6_reference_derivation.py` to all 15 F seeds (zero-`evaluate()`, banked CSVs, ~20–30 min local) → banks the per-seed r_prod vector and the fleet statistic from committed code; **+ 2-seed end-to-end spot-check** under `fused` (2 × ~30 min local, D6-style gates) to verify the O6 transfer on seeds it was not proven on | ~1.5 h local, zero cluster | fleet fused null = the banked reference fleet **by measured transfer**; registered as such (A21 text, A17 bands from the O6-measured storage floor 1.94e-6 × per-seed variation) |
| B | Full 15-seed end-to-end fused fleet | ~8 CPU-h (cluster array or ~8 h local sequential) | the direct measurement; numerically expected to differ from A by ≤ ~1e-3/seed |
| C | Leave OPEN | 0 | row #158 item 3 stays open; label and O6 stand regardless |

Power note (both A and B): the statistic's seed-scatter is the same (sd ≈ 0.071 → SEM 0.0184 at
n=15); B buys no power over A, only directness. The fused cell's *posterior* information collapse
(span 1.53 vs 7.59 nats) affects GATE-V-style checks, not this score's scatter.

### [D2] Production-basis fork: `off` → `fused` for runs-of-record

**The physics-change gate package (all six items):**

1. **Old formula** — completion numerator `B_num(h) = ∫ (1−f_k)·p_gw·dVc/(1+z) dz` over the
   per-event window (`bayesian_statistics.py:5171-5179`, the `off` else-branch) while the
   normalizer β̄_Ḡ_φ = ∫ (1−f̄)·S̄_φ·p_pop dz carries S̄_φ (`:2058-2066`) — the measured
   numerator/normalizer mismatch (rows #155/#157: IMPLEMENTATION-CONVENTION DEFECT).
2. **New formula** — the same numerator × S̄_φ(z;h) via `completion_numerator_integrand_sel_1d`
   (`:4992-5007`), i.e. `selection_in_completion_numerator="fused"` — both legs read one
   `precompute_phi_marginal_survival` table (verified object-identity, A20/O6 review).
3. **Reference** — rows #117–#118's fusion derivation (`PROPOSAL_2D_SELECTION_FUSION_20260817.md`
   [P2]); selection-consistency per Gray et al. (2020) arXiv:1908.06050 (numerator and
   normalizer must carry the same selection factor for the ratio to be a probability).
4. **Dimensional analysis** — S̄_φ is dimensionless (a survival probability ∈ [0,1]); B_num's
   units are unchanged.
5. **Limiting case** — S̄_φ ≡ 1 (no selection) recovers the `off` numerator exactly; and the
   generator identity: the accepted per-pixel z-density is ∝ (1−f_k)·p_pop·S̄_φ, exactly the
   fused integrand (A20/O4 review, FATAL-2 table).
6. **Regression evidence** — O6 GATE D6: the plumbing leaves the `off` path bit-exact
   (9200/9200 rows); O6 F6 record = the fused regression anchor; 1709-test suite green.

**Materiality is already measured (row #119, banked):** off→fused moved MAP and width by
**zero** in every channel of record (1D railed 0.600→0.600; 2D 0.780→0.780, 0.800→0.800;
σ shifts ≤ 0.002), with `run_20260817_fusion_counterfactual/` as the standing bridge artifact.
The estimator's own log labels `off` under `absolute_marginal` *"not a production posterior."*

| option | meaning | consequences |
|---|---|---|
| **A (recommended)** | Future runs-of-record (landscape included) run `fused` (i.e. the CLI default `auto` under `absolute_marginal`); `PRODUCTION_FLAGS` pin updated; **past runs stand** on `off` with the ratified defect label + row #119 bridge — no re-runs (row #119's own ruling) | correctness aligned with the in-tree production default; D5's landscape needs no special-casing; commit is `[PHYSICS]` with ledger row |
| B | Stay on `off`; record the omission as a tracked systematic (G7 row) | every future run inherits a known, mechanism-identified defect; landscape's fused cells contradict the pinned basis |
| C | Defer again | blocks D5 (landscape is fused-cell); leaves runbook 28 §1 open |

**Carried caveat rides with any option:** fused does NOT cure the H₀ rail (O6 F6 full channel
mean_h = 0.618, r_low; rail is impostor-leg/photo-z territory — [D3]).

### [D3] The impostor-leg direction — the dominant production channel

**What is measured.** The impostor catalogue leg carries **−0.079 of B-SEL's −0.108 (73%)**
(row #149, 12/12 seeds): in-cone catalogue galaxies that are not the true host — the host is
never in the catalogue at our depth — sit at low z and drag h down. This is *venue physics*
(shallow catalogue + photo-z), not an implementation defect: our 4.79% in-catalogue rate sits
below Gray 2020's own validated completeness floor of 25% (Stage-L), and the rail's root cause
is GLADE photo-z error σ_z ≈ 0.035 ≫ GW precision (bridge-the-closure investigation).

**Mitigation routes already closed by measurement:** spec-z subset (F4 — spec-z hosts dominate
0/40 events; refuted); σ_z/σ_M precision rescue (F5 — with-BH-mass channel needs σ_M ≲ 1–2%,
far below the ~0.55 dex R&V15 scatter; no GLADE rescue); ensemble/hierarchical coherence
unproven at our σ_z/z (photo-z literature comparison). The catalogue-leg *convention* fork
([P3], per-host selection weighting) is deferred to the Gray-convention paper task (row #110,
reaffirmed row #119 item 1).

| option | meaning |
|---|---|
| **A (recommended)** | **Accept as venue property and make it the paper's finding**: the three-channel decomposition (impostor −0.079 ⊕ tilt +0.055 ⊕ matched −0.085, the last now mechanism-identified and fixed in-tree) is literature-novel (Stage-L: no published work decomposes a mixture posterior per sector); the rail is attributed to photo-z starvation with F4/F5 as the closing evidence. [P3] stays with the Gray paper task (row #110). No new estimator work on this channel. |
| B | Pull the [P3] Gray-convention catalogue-leg fork forward now (out of row #110) and measure whether the convention choice moves the impostor drag | ~a research cycle; row #119 M-4 bounds the mixture skew small (median +0.02–0.03 catalogue share) |
| C | Open a new mitigation measurement (author to specify — the known routes above are refuted) |

### [D4] Systematics row 16 re-grade (carried [RULE], runbook 23)

Row 16's clause "affects rates/shape, not estimator calibration" is contradicted by measurement
(row #138). Proposed: re-grade to a **measured, calibration-affecting systematic** with row #138
as the evidence line. One-line G7 edit + ledger note.

### [D5] Landscape/T1 resubmission ([DO], conditional on D2 = A)

The cancelled 13-fused-cell catalogue-quality landscape sweep resubmits with
`--time=48:00:00` and the RUN_DIR export fix (runbook 22 §48); blind T2 predictions
(`MECHANISM_SIGMA_M_SIGMA_Z_DERIVATION.md` §5) stand for the current estimator. Un-gates as the
chain's fourth link once D2 fixes the basis. Costing: 18 cells × cluster; preflight + A21
registration before submission.

## 2. Decision table

| # | tag | decision | recommendation | cost |
|---|---|---|---|---|
| D1 | [DO]+[RULE] | fleet fused null | **A** — transfer + banked 15-seed reference + 2-seed spot-check | ~1.5 h local |
| D2 | [RULE] | production basis off→fused | **A** — fused for future runs; past runs stand (row #119) | one `[PHYSICS]` commit + gate ledger row |
| D3 | [RULE] | impostor-leg direction | **A** — venue property; paper centers the decomposition; [P3] stays with row #110 | none now |
| D4 | [RULE] | G7 row 16 re-grade | adopt (row #138 evidence) | one-line edit |
| D5 | [DO] | landscape resubmission | approve conditional on D2=A | cluster, 18 cells, 48 h walltime |

Per the binding default, each row is ruled separately; "approved"/"ratified" on this document
covers exactly these five rows as recommended, and any option-B/C selection replaces the
corresponding row.
