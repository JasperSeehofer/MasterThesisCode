# NEGLECT TRIGGER REGISTER — Eq. (31) cross-term, stage-5 conditional closure

**Date:** 2026-08-07 (stage-5 closure).
**Author ruling (2026-08-07, Jasper Seehofer):** NEGLECT-WITH-NUMBER is **accepted in all four
venue × channel cells**, **CONDITIONED** on this register — the author's words: conclude "with
clear triggers when this needs be re-evaluated… for this assumption to break, at least X,Y,Z has
to become apparent/realized". This document is those X, Y, Z, derived quantitatively from the
measured record.
**Measured record of reference (all committed, this directory):** `readout_20260806.json`;
`readout_adjudication_20260807.json` (independent recompute, verdict **CONFIRMED**, T values
bit-identical); `CROSSTERM_READOUT_20260806.md` incl. its appended CORRECTION section. Verdicts:
**NEGLECT-WITH-NUMBER** in joint_r1/1d, joint_r1/2d, iiib/1d, iiib/2d, with
T = 1.468160e-05 / 5.648753e-05 / 1.024615e-05 / 5.379542e-06 class-summed mixture-composed chord
nats vs the LOCKED band X = 2.78 / Y = 7.96 (X/T factors **1.89e+05 / 4.92e+04 / 2.71e+05 /
5.17e+05**; minimum margin **4.92e+04×** at joint_r1/2d).
**This file is append-only from here on:** any trigger firing, any re-evaluation result, and any
retirement of a trigger is appended as a dated section; no text above an append line is edited.

---

## 1. What the 4.9e+04× minimum margin actually rests on — quantitative anatomy

The scored per-pair object is the **mixture-composed** correction (prereg §5 identity):

```
Δ̃_ij(h) = log1p( w_G,i · w_G,j · L_cat,i · L_cat,j · expm1(Δ_ij(h)) / (combined_i · combined_j) )
```

i.e. Δ̃ ≈ F_ij · (e^Δ − 1) with the **composition factor**
`F_ij = w_G,i·w_G,j·L_cat,i·L_cat,j / (combined_i·combined_j) = r_i · r_j`, where
`r_e = w_G·L_cat,e / combined_e` is the **per-event catalogue-leg share** of event *e*'s combined
(mixture) likelihood. The class statistic factorizes exactly at readout level as

```
T (composed) = F_eff × T_raw ,   F_eff ≡ T / T_raw  (the class-chord-effective composition factor)
```

Measured decomposition, per cell (from `readout_adjudication_20260807.json`; F_med/F_max are the
emitted per-pair composition factors at h = 0.73):

| cell | T (scored) | T_raw (diagnostic) | F_eff = T/T_raw | X/T (margin) | F_eff needed to reach X (raw fixed) | required F_eff growth | F_med / F_max (h=0.73) | w_G(0.73) | median r_e ≈ √F_med |
|---|---|---|---|---|---|---|---|---|---|
| joint_r1/1d | 1.468160e-05 | 181.208 | 8.10e-08 | 1.89e+05 | 1.53e-02 | ×1.89e+05 | 1.02e-07 / 9.46e-02 | 0.070802 | ~3.2e-04 |
| joint_r1/2d | 5.648753e-05 | 5.799 | 9.74e-06 | **4.92e+04** | 4.79e-01 | ×4.92e+04 | 1.43e-07 / 1.29e-01 | 0.070802 | ~3.8e-04 |
| iiib/1d | 1.024615e-05 | 77.514 | 1.32e-07 | 2.71e+05 | 3.59e-02 | ×2.71e+05 | 1.49e-07 / 7.16e-02 | 0.061967 | ~3.9e-04 |
| iiib/2d | 5.379542e-06 | 0.0960 | 5.61e-05 | 5.17e+05 | 2.90e+01 (**>1: unreachable**) | — | 2.54e-06 / 3.74e-02 | 0.061967 | ~1.6e-03 |

Three structural facts fall out of this table, and they are what the margin rests on:

1. **In 3 of 4 cells the raw catalogue-leg range EXCEEDS X** (181.2 / 5.80 / 77.5 nats vs
   X = 2.78). In those cells the NEGLECT verdict rests **entirely on the mixture composition**
   annihilating the catalogue leg — not on the catalogue-leg physics being small. The raw 2D
   chords are moreover **positive (low-h)**: +2.507 nats (joint_r1) / +0.0116 nats (iiib) — the
   H-2 coherence is physically present in the catalogue leg; the composition suppresses it by
   ~5 orders (median per-pair F ~1.5e-07–2.5e-06) and the h-dependence of w_G (falling
   0.0957 → 0.0556 across the grid, ×1.72) **reverses the class-level chord sign**. Anything
   that changes the composition therefore attacks the verdict directly, and the sign of the
   effect can flip along with the magnitude.
2. **The one exception is iiib/2d:** its raw range (0.0960 nats) is itself **29× below X**, so no
   composition change alone can ever fire that cell — it additionally needs the raw catalogue-leg
   cross-term to grow.
3. **The composition factor is a product of two per-event catalogue-leg shares** (F = r_i·r_j),
   with median r ≈ 3e-04–1.6e-03 today and w_G ≈ 0.062–0.071 at h = 0.73. Because F ∝ w_G², the
   **w_G lever is bounded**: even the saturation w_G → 1 gives only ×199 (joint_r1) / ×260 (iiib)
   growth in F — **w_G alone cannot close any margin** (min margin 4.92e+04). Closing a margin
   through the composition requires the *whole* catalogue-leg share r = w_G·L_cat/combined to
   approach O(1) per event, i.e. the catalogue leg coming to **dominate the mixture** — a
   qualitative regime change (deep/complete catalogue), not a tuning drift.

## 2. Re-evaluation instrument ladder (cheapest first)

| level | instrument | what it does | cost |
|---|---|---|---|
| **L1 — census re-run** | `make_target_pairs.py` | zero-compute sharing-pair census from the frozen ball emits: pair counts, per-pair n_shared, in/outside-C-4 split. No quadrature, no likelihood. | minutes |
| **L2 — composition arithmetic** | readout-side recompute (pattern of `readout_adjudicate.py`) | re-derive per-event r = w_G·L_cat/combined from the post-change per-event emits and **recompose Δ̃ from the EXISTING raw Δ rows** in `outputs/run_*.json` (the prereg §5 identity is composable from row fields alone; rr1 N8 verified to 3.9e-16). Valid whenever the raw catalogue-leg Δ is untouched (completion-leg / weight changes). | minutes–hours, CPU |
| **L3 — full instrument re-run** | `crossterm_instrument.py` (frozen hash or its re-certified successor) over a fresh census | re-measures the raw Δ itself. Needed whenever L_cat's own kernel/structure, the balls' contents, the catalogue, or the σ_z regime change. | ~0.5–3 h/channel; 1d channels chunked (30 GB host OOM record) |

## 3. The triggers

Summary table (details per trigger below). **A trigger "fires" when its threshold is met or when
the change it names lands in production, whichever is first.**

| ID | WHAT to watch | THRESHOLD (order of magnitude) | cheapest re-evaluation |
|---|---|---|---|
| (a) | w_G and the per-event catalogue-leg share r (mixture composition) | w_G(0.73) ≥ ~0.3, or median per-pair F ≥ ~1e-05, or max F ≥ ~0.5 | L2 |
| (b) | any /physics-change to L_cat's structure or the mixture weights — **named: open queue item 3, the S̄_φ-inside-1D N-2 adoption** | fires on adoption, no threshold | L2 (L3 only if L_cat's own kernel changes) |
| (c) | ball geometry / localisation model — **named: issue #53 (3σ vs 2σ z-window)** | any radius/window ENLARGEMENT or localisation-model change fires; 2σ shrink does NOT | L1 (escalate if census grows ≥10×) |
| (d) | catalogue replacement or σ_z regime | median σ_z/(1+z) rising ≥ ~3× above today's 0.033, or catalogue swap; σ_z SHRINK does NOT fire | L1 + L3 |
| (e) | evaluated event count N_ev (pairs ∝ N_ev²) | N_ev ≳ 3e+04 (~20× today's 1588) alarms; nominal margin closure at N_ev ≈ 3.5e+05 | L1 + L2, L3 near ≥1e+05 |
| (f) | the mixture/composition scoring convention itself (prereg flag (c)) | fires on any de-ratification or change of the composed-scoring convention | none — re-adjudication from existing numbers |

### (a) Mixture composition factor growth — the load-bearing ~1e-07

**What:** the annihilation documented in §1: median per-pair F ~1.0e-07–2.5e-06, driven by
w_G ≈ 0.062–0.071 and per-event r = w_G·L_cat/combined ≈ 3e-04–1.6e-03. The minimum margin
(4.92e+04×, joint_r1/2d) is **numerically identical** to the required F_eff growth, so the margin
IS the composition suppression.
**Watch:** w_G per venue at h = 0.73 (emitted per run; degenerate single value); the
median/max per-pair composition factor at h = 0.73 (emitted row fields, summarized in
`readout_adjudication_20260807.json` → `composition_factor_h073`).
**Threshold:** re-evaluate when **w_G(0.73) ≥ ~0.3** (≈4–5× today — e.g. a much deeper/more
complete catalogue or a completeness-correction change; note w_G alone tops out at ×199–260 in F
even at w_G = 1, §1 fact 3), **or** median per-pair F ≥ ~1e-05 (≈×70–100 today — leaves ~3
orders of the worst margin), **or** max per-pair F ≥ ~0.5 (individual pairs saturating; today
0.129 max). Any completeness-correction change that re-weights the mixture (w_G's definition,
β_G/D bookkeeping) fires this trigger regardless of the numeric sentinels.
**Instrument:** **L2** — re-run the composition arithmetic over the existing raw rows. L3 is not
needed unless (b)/(d) also fire.

### (b) Catalogue-leg FORMULA changes — any /physics-change touching L_cat's structure

**What:** the raw Δ rows encode today's L_cat structure; the composition encodes today's
combined-mixture structure. A formula change to either invalidates the corresponding half.
**Named standing trigger:** **open queue item 3 — the S̄_φ(z;h)-inside-1D N-2 adoption**
(RUNBOOK_NEXT_SESSION_8.md §1 item 3, author rulings R-0..R-5, still OPEN as of this closure;
ledger #93: real, positive, bounded correction, NOT the rail owner). If the author adopts
S̄_φ inside the 1D completion numerator, the per-event `combined` (and w_G's leg weighting)
changes → every r_e changes → this register's §1 table is stale in the 1D cells.
**What "re-check" means:** **re-run the composition arithmetic (L2), not necessarily the full
instrument.** The N-2 adoption touches the completion leg, which enters `combined_e` (the
denominator of r) but not the catalogue-leg Δ itself — the existing raw rows stay valid and
Δ̃ can be recomposed from them plus the post-adoption per-event emits. Only if a change touches
**L_cat's own kernel/structure** (host-z kernel family, ball-side weighting, the catalogue-leg
integrand) do the raw Δ rows die with it → **L3**.
**Threshold:** none — fires on adoption/landing of any such change.

### (c) Ball geometry changes — issue #53 and localisation

**What:** the target set is the M-4 truly-sharing census over the production balls
(`get_redshift_outer_bounds`, which **hardcodes 3σ and silently ignores the
`sigma_multiplier=2.0` call-site argument — issue #53, OPEN**; prereg flag (b): acknowledged,
physics adjudication deferred to its own /physics-change; the instrument mirrors the hardcoded
3σ production behavior).
**Direction, stated explicitly:** a **2σ adoption shrinks the balls and REDUCES sharing —
favorable to NEGLECT** (fewer sharing pairs, smaller n_shared, smaller W). If #53 is resolved as
2σ, the verdict stands a fortiori; an L1 census refresh to record the smaller target set is
optional book-keeping, not a re-adjudication.
**Fires on:** any radius/window **ENLARGEMENT** (>3σ, wider σ_z floor, additive padding) or any
**localisation-model change** (sky-ball construction, CRB→ball mapping) — these grow or reshuffle
the sharing census.
**Instrument:** **L1 — census re-run (`make_target_pairs.py`, cheap, zero-compute)**. Escalate to
L2/L3 if the census changes materially (order-of-magnitude sentinel: total sharing-pair count or
Σ n_shared grows ≥ ~10× vs today's 349/104 (joint_r1) and 280/21 (iiib) pairs).

### (d) Catalogue replacement or σ_z regime change

**Direction, derived:** the composed cross-term vanishes in the perfect-z limit — the
certification record measured **O(σ_z²) convergence of Δ to 0 as σ_z → 0** (cert §3, tracked to
≤3.3e-13 nats against an independent reference). **Smaller σ_z shrinks the cross-term toward the
perfect-z limit — favorable to NEGLECT; no re-run needed.** A **WIDER kernel** (σ_z regime up) or
a **qualitatively different n(z)** (catalogue replacement, e.g. GLADE+ → another catalogue, or a
changed parse-time σ_z floor — today's regime: median σ_z/(1+z) = 0.033, M-1) changes both the
raw Δ and the sharing census in ways this record cannot bound.
**Threshold:** median σ_z/(1+z) rising ≥ ~3× above 0.033, or any catalogue swap / n(z)-model
replacement.
**Instrument:** **L1 + L3** (fresh census, full instrument re-measurement — the raw leg itself is
invalidated).

### (e) Event-population scale — where the margin nominally closes

**What:** sharing pairs grow ~N_ev² (fixed geometry/sky density) and the class sum grows with the
sharing-pair count. Today: **N_ev = 1588 evaluated events/venue**, 385 overlap-involved, pair
counts 349/104/280/21; per-pair mean composed chord c̄ = T/n_pairs = 4.2e-08 / 5.4e-07 /
3.7e-08 / 2.6e-07 nats.
**Nominal closure N_ev, assuming today's per-pair scale and today's LOCKED band** (T scales
×(N/1588)², X fixed): N_close = 1588·√(X/T) =
**≈3.5e+05 (joint_r1/2d, binding)** / 6.9e+05 (joint_r1/1d) / 8.3e+05 (iiib/1d) / 1.1e+06
(iiib/2d). If the band is instead re-derived with the population (X ∝ √N per its §7.2
construction), closure moves to (X/T)^{2/3}: binding ≈2.1e+06. Either way this is **2+ orders
beyond any planned campaign** (~1.6e+03 events).
**Threshold:** alarm at **N_ev ≳ 3e+04** (~20× today; the quadratic pair term has then eaten
~2.6 of the 4.7 orders of the worst margin) → L1 + L2 with the actual census (the N² scaling is
nominal — the true growth law is what L1 measures); **full L3** re-measurement before relying on
the verdict anywhere near N_ev ≈ 1e+05.

### (f) Change to the mixture/composition convention itself — prereg flag (c)

**What:** flag (c), RATIFIED 2026-08-06, is the convention that **the band applies to the
mixture-composed value** — the object the posterior actually consumes. Every margin in §1 is
downstream of that convention.
**Fires on:** any de-ratification or replacement of flag (c); any change to how the posterior
consumes the catalogue leg (e.g. un-mixed/raw-scored catalogue-leg contributions, a different
composition identity, per-leg posteriors).
**Why it is the sharpest trigger:** under a raw-scored convention **the existing numbers already
break the NEGLECT in 3 of 4 cells** — raw ranges 181.2 / 5.80 / 77.5 nats ≥ X = 2.78 (and
joint_r1/2d raw sits inside [X, Y)); only iiib/2d (0.0960) stays below X.
**Instrument:** none needed for the first read — **immediate re-adjudication from the existing
measured record** (the raw diagnostics were reported-never-scored precisely so that this
re-scoring costs nothing); then whatever instrument the new convention requires.

## 4. Standing favorable directions (recorded so good news is not re-run)

- **Issue #53 resolved as 2σ** → balls shrink, sharing shrinks → NEGLECT a fortiori (trigger (c)).
- **σ_z improvements** (spec-z influx, smaller photo-z errors) → O(σ_z²) shrink toward the
  perfect-z limit → NEGLECT a fortiori (trigger (d)).
- **Fewer events / pruned catalogue** → fewer sharing pairs → NEGLECT a fortiori (trigger (e)).

None of these require any re-run for the verdict to remain valid.

## 5. Related unowned object (cross-reference, not a trigger)

**M-2's matched 2D overlap residual is REAL and now UNOWNED** (+0.02070 joint_r1 / +0.02225 iiib
nats/event, cluster-robust p 0.0042/0.0050, low-h-preferring, ~8 nats class-scale if coherent):
this measurement **excludes the Eq. (31) cross-term as its mechanism** (composed 2D chord has the
opposite sign at 5–6 orders below Y). Its owner-hunt is ledger §4 open thread 16. If that hunt
ever re-implicates likelihood-factorization structure or the mixture composition, that is
trigger-(b)/(f) territory and re-opens this register by construction.

## 6. Conditional closure — the ruling, rendered precisely

**The NEGLECT-WITH-NUMBER closure of the Eq. (31) cross-term thread stands unless and until at
least one named trigger (a)–(f) fires.** For this assumption to break, at least one of the
following has to become apparent/realized: **(a)** the mixture composition factor grows by the
stated orders (w_G ≳ 0.3, median F ≳ 1e-05, or max F ≳ 0.5); **(b)** a /physics-change lands on
L_cat's structure or the mixture weights — including the open S̄_φ-inside-1D N-2 adoption
(queue item 3); **(c)** the ball z-window/radius is enlarged or the localisation model changes;
**(d)** the σ_z regime widens ≥ ~3× or the catalogue is replaced; **(e)** the evaluated event
population approaches ~3e+04; or **(f)** the mixture-composed scoring convention (prereg
flag (c)) is itself changed. On any firing, the named cheapest instrument (L1 census / L2
composition arithmetic / L3 full instrument) is run first, and its result is appended — dated —
to this register and to the bias-history ledger. Absent a firing, the four T values
(1.468160e-05 / 5.648753e-05 / 1.024615e-05 / 5.379542e-06 nats) are the permanent
NEGLECT-WITH-NUMBER record and the cross-term is neglected **with that number, not with the word
"small"**.
