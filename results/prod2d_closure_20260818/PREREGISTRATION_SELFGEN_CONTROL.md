# PRE-REGISTRATION — C-SG: a positive control that CAN fail (v2)

**Status:** registered, pre-run. **v1 returned NOT-READY** from its own adversarial pre-check, with
11 required amendments; all 11 are applied below, plus both optional suggestions. The orchestrator
independently re-derived the five decisive findings before accepting them (§0). Append-only below
the freeze line.

**Provenance gate.** Ledger row #144 §6 registered this as the settling measurement. Both blocks are
cleared: the scoring defect (rows #145/#146) and the `g_frac` generator question (row #147,
CONTROL-SAFE). **Verifier coverage is not inherited** (row #144 item 8): this registration carries
its own pre-check, and v2 supersedes v1 in full.

---

## 0. What the pre-check overturned — verified, not accepted on report

Five findings changed the design. Each was re-derived independently before adoption:

1. **The measurement kernel: v1 had the direction BACKWARDS.** The estimator's default `ratio`
   kernel satisfies, to machine precision (verified over random `d̂`, `σ_dL`, `d_L`; ratio = 1 to
   1e-12):
   `N(d_L(z;h)/d̂ ; 1, σ_dL/d̂) ≡ d̂ · N(d̂ ; d_L(z;h), σ_dL)`
   — an h-independent constant times the **fixed-σ_dL linear Gaussian**. So B-SEL's *linear* draw
   was the matched one, and v1's proposed ratio draw would have injected the `d_L(z;h) ∝ 1/h`
   factor — the campaign's own predicted `E[score] ≈ −1/h` defect — into the generator. **C-SG
   generates linearly.**
2. **Selection was applied TWICE.** `S̄_φ(z;h) ≡ ∫ φ(log₁₀M) S_4D(d_L(z;h), M(1+z)) dlog₁₀M`
   (`bayesian_statistics.py:1932-1975`) **is** the marginal detection probability, dimensionless in
   [0,1]. v1 drew `z ∝ w_pop(1−f̄)S̄_φ` — already conditioned on detection — and then accepted again
   with `p_det`. **Fatal; redesigned in §2.**
3. **BAND C's INTERNAL-DEFECT branch was unreachable.** A constant estimator bias `b` displaces all
   three arms equally, so v1's accuracy-form GATE S failed for exactly `|b| ≥ 0.005` — precisely
   when BAND C would have said INTERNAL-DEFECT. Verified by enumeration. **GATE S is now a
   regression (§6).**
4. **Two of four arms had targets a vacuous posterior hits exactly.** A flat log-posterior returns
   `mean_h` = **0.7300000000** on `H_GRID_41` (= C-SG-F's `h_gen`) under *both* weight conventions —
   the B-F1 mechanism survived the row #146 correction untouched — and **0.6800000000** on
   `H_GRID_FULL` under trapezoid (= C-SG-δ−'s `h_gen`). **GATE V and a pinned scoring grid (§6).**
5. **"Dark-only universe" was false.** `in_catalog`/`host_galaxy_index` are bookkeeping; the
   ball-tree runs unconditionally (`bayesian_statistics.py:4443`). Measured on `bsel_seed900101` at
   h = 0.73: **128/174 events (73.6%)** have `L_cat_no_bh > 0`. Impostor share of the per-event
   numerator: median 6e-4, 90th 0.057, 99th 0.647, max 0.821, mean 0.034, with 2 events above 0.5.

**Also corrected:** the numbers of record are row #146's **−0.1083** (B-SEL), not −0.1120; and row
#144's ≥0.073 residual bound was derived against −0.112 and is **OPEN**, not settled.

## 1. What is being tested

Row #140 measured B-SEL, now **−0.1083** on the corrected combine. What blocks banking it as an
estimator defect is that **the harness has never had a control that could fail**.

> **The single reading, stated once (v1 carried two contradictory ones):**
> **ESTIMATOR-SELF-CONSISTENT ⇒ B-SEL's −0.1083 is owned by generator–model mismatch, and row
> #140's residual is REFUTED as an estimator defect. INTERNAL-DEFECT ⇒ row #140 is promoted from
> PROVISIONAL to a banked defect claim.**

## 2. The generative model (design B — the only self-consistent one)

At generating constant `h_gen`, per event:

| # | stage | C-SG v2 | why |
|---|---|---|---|
| 1 | **(z, Ω) jointly** | `∝ w_pop(z)·(1 − f_k(Ω; z; h_gen))` | the estimator's numerator uses `f_k` at the **event's own pixel** (`:4902, 4963`), not the sky-marginal `f̄` the generator used; drawing z and Ω independently is a first-order per-event mismatch |
| 2 | **mass** | `log₁₀M ~ φ`, the **same** `_phi_dark_mass_log10_grid` the estimator contracts | makes the mass self-consistent AND supplies the `M` column the ball-tree uses (§5 item 1) |
| 3 | **selection** | accept with `S_4D(d_L(z;h_gen), M(1+z))` — **once, and only here** | `S̄_φ` is the φ-marginal of exactly this object, so drawing with `S̄_φ` *and* accepting would double-count (§0 item 2) |
| 4 | **measurement σ** | `σ_dL` fixed per event, drawn independently of z (§3) | the estimator conditions on σ_frac and never models its distribution — verified: everything σ-dependent (`_log_norm_3d`, `:4044`) is h-independent and cancels |
| 5 | **observation** | **linear**: `d̂ = d_L(z;h_gen) + σ_dL·ε`, `ε ~ N(0,1)` | §0 item 1 |

No donor-row host mismatch, no SNR-weighted borrowing of the quantity that sets selection, no
quality filter.

## 3. σ_frac — the arms

Production anchor, measured over the pinned CRB reference (n = 1590): quantiles 5/25/50/75/95 =
0.0112/0.0266/0.0373/0.0454/0.0531, mean 0.0355, and **corr(σ_frac, d_L) = 0.656**.

- **C-SG-F (decisive):** `σ_dL := 0.0373 · d_L(z;h_gen)` — **stated explicitly**, because
  `0.0373·d̂` would make the recorded "measurement error" fluctuate with the noise draw, which no
  Fisher σ does. These are different arms and v1 did not say which.
- **C-SG-E (robustness):** `σ_frac` i.i.d. empirical, independent of z. Its z-independence is a
  **large** departure from production (corr = 0.656) — deliberate, since this is a control on the
  estimator's mathematics, not a fidelity test.

## 4. Arms, seeds, cost

| arm | `h_gen` | σ | seeds | purpose |
|---|---|---|---|---|
| C-SG-F | 0.73 | fixed | 15 | decisive |
| C-SG-E | 0.73 | empirical | 15 | σ-robustness |
| C-SG-δ− | **0.68** | fixed | 8 | sensitivity |
| C-SG-δ+ | **0.78** | fixed | 8 | sensitivity |

46 seeds. **Cost corrected:** A-3's own anchor (18 seeds ≈ 20 CPU-h) scales to **≈ 51 CPU-h**, and
≈ 69 CPU-h at `--cpus-per-task=2` × 45 min wall — not v1's "≈ 35". Workspace expires **2026-09-23**.
Seeds start at 910101. Scorer committed before the data; scored with the corrected combine (#146).

## 5. What C-SG does NOT control — the 128-column inventory

`write_mirror_crb_csv` (`:1709-1720`) is a bare dump of `draw_realization`'s frame, which is a donor
row with **five columns overwritten** (`:1278-1282`). The pinned CRB CSV has **128 columns**, so
~123 are inherited. Load-bearing ones, named as claims rather than infrastructure:

1. **`M`, `M_error`** — drive the ball-tree candidate set (`:4443-4453`). **Design B fixes this**:
   the drawn `M` is written to the CSV.
2. **Sky covariance block** — sets the localization cone and feeds `_excluded_mask`.
3. **`d_L`–φ/θ cross-covariances** — kept at donor values while `σ_dL` is replaced; the direct cause
   of the non-PD attrition in GATE Q.
4. **`SNR`** — gates `:3673` and is physically incoherent with C-SG's own selection.

**The impostor leg is NOT removed** (§0 item 5): C-SG's true hosts are never in the catalogue, but
the estimator finds impostors in ~74% of cones. **C-SG supplies a 0% in-catalogue share against the
estimator's assumed `f̄(z)`** — a generator–model mismatch, not merely a scope limitation. Its size
is measured pre-run by **O2** below.

## 6. Statistics, gates and bands (A15)

### Primary statistic — the per-event score (adopted from pre-check O1)

The self-consistency claim **is** `E[∂_h ln p_i] = 0` at `h_gen`. Computed by finite difference over
the existing h-grid from the banked `combined_no_bh`, it uses **n = 200 × 15 = 3000 events** rather
than 15 seed-means, never rails, has **no grid-midpoint coincidence**, and is the same quantity row
#137 reports at 37σ. `mean_h` becomes **reported-only secondary**.

### Pre-flight gates, all zero-compute, scored before any seed is analysed

> **GATE H (h_gen threading).** For each `h_gen`, assert it reaches `draw_selected_population_redshifts`
> (`:1213`), `dist_vectorized(host_z, h=h_gen)` (`:1242`), `build_bsel_selection_objects(h_true=h_gen)`
> (`:894`, `lru_cache`d), and `compute_seed_statistics(h_true=h_gen)` (`:2049`, default `H_TRUE`).
> Report `S̄_φ`'s `z_max(h_gen)` (`:1975-1979`: the domain **shrinks** at 0.68, **grows** at 0.78,
> asymmetrically) and the fraction of drawn events beyond the injection pool's calibrated depth,
> since `allow_low_pdet_coverage=True` (`:1731`) silences production's own STOP.
>
> **GATE Q (attrition).** Report per arm the events removed by each production cut: `SNR < 20`
> (`:3673`), `σ_dL/d̂ ≥ 0.10` (`:5397`), and the **Fisher non-PD exclusion** (`:4012-4051`, skipped
> silently at `:4418`) — which v1 never named and which C-SG *introduces*: forcing σ_dL while
> keeping donor cross-covariances is projected at **1.07% (F) / 2.70% (E)** non-PD. **Any arm above
> 1% attrition from that cut is redesigned, not run** (rescale the cross-covariances with σ_dL).
>
> **GATE D (premise).** `run_d1_premise_check` (`:2442`) on the C-SG generator per `h_gen`: max CDF
> gap of surviving events vs the model density. Band = `D_crit(5%)` **at the actual n**
> (n=200 ⇒ 0.0960), never the retired 0.05.
>
> **GATE V (anti-vacuity, per seed).** A flat log-posterior returns exactly 0.7300000000 on
> `H_GRID_41` (= C-SG-F's `h_gen`) and 0.6800000000 on `H_GRID_FULL` trapezoid (= δ−'s). Every seed
> must satisfy `max(log_posterior) − min(log_posterior) ≥ 5 nats` on `H_GRID_41` and
> `σ_h ≤ 0.5·σ_prior`; failures are reported and excluded. **All scoring is on `H_GRID_41` via
> `compute_seed_statistics`; scoring on the banked 46-node grid is FORBIDDEN.**

### GATE S — sensitivity, as a regression (replaces v1's accuracy form)

> Fit `mean_h(seed) = b + s·h_gen` over all 31 F+δ seeds (`h_gen ∈ {0.68, 0.73, 0.78}`), residual
> variance `σ̂_seed²` from the pilot.
> **CONTROL-VALID** if `|ŝ − 1| ≤ 3·SE(ŝ)`. **CONTROL-INERT** if the CI on `ŝ` contains 0 ⇒ the arm
> carries no h-information — STOP.
> The intercept offset `b̂ + (ŝ−1)·0.73` **is** the bias estimate, read **alongside** the gate, never
> blocked by it.
> *Rationale, registered: a constant bias displaces all three arms equally, so an accuracy-form gate
> returns "biased" under exactly the hypothesis BAND C exists to test (§0 item 3).*

### Bands — no threshold is registered until the pilot measures σ̂_seed

**v1's `σ_seed = 0.0058` is DELETED.** It came from B-SEL, which is floor-saturated at the 0.600
grid edge (coverage 0/0/0, railed 12/12); saturation truncates the per-seed distribution and
**deflates** scatter, inflating claimed power. Measured across the banked fleet
(corrected trapezoid weights, `H_GRID_41`, sentinel-affected seeds dropped):

| arm | railed | n | sd(mean_h) | median σ_h | ratio |
|---|---|---|---|---|---|
| b0 | no | 14 | 0.0230 | 0.0214 | 1.07 |
| eden2 | no | 4 | 0.0185 | 0.0245 | 0.76 |
| bsig005 | no | 14 | 0.0102 | 0.0169 | 0.61 |
| eden05 | no | 9 | 0.0084 | 0.0170 | 0.50 |
| bsel | **yes** | 12 | 0.0060 | 0.0220 | **0.27** |
| bden | **yes** | 15 | 0.0024 | 0.0158 | **0.15** |

Unrailed 0.50–1.07 vs railed 0.15–0.27. C-SG is designed not to rail at `σ_h ≈ 0.016–0.022`, so its
expected `σ_seed` is **0.009–0.022** — 1.5–3.8× v1's value. At those values v1's hard 0.005 band
false-fails **11–38% at N=15**, **24–52% at N=8**, and **GATE S (both δ arms) 22–77%** — D-1's
failure reproduced exactly.

> **MANDATORY 4-seed C-SG-F PILOT.** Its only outputs are `σ̂_seed` and per-seed `σ_h`. **N and every
> band are set from it, and the false-fail table is published, before the remaining seeds run.**

**BAND C** (on the score at `h_gen`, with `mean_h` secondary), thresholds set post-pilot:
ESTIMATOR-SELF-CONSISTENT / INTERNAL-DEFECT / MIXED as in §1, with the pilot's SE.

**BAND R** — either drive F and E from **independent RNG sub-streams** so their (z, Ω) draws are
bit-identical and score the **paired** difference, or declare them independent and use
`√2·σ̂_seed/√15`. v1 did neither: inserting a σ draw desynchronizes the shared
`default_rng(seed)` stream (`:1176, 1182, 1199-1226, 1246, 1272`), so "paired seeds" were not paired.
Publish the false-fail rate either way.

## 7. A10 — invariants and structural blindness (v1's list was materially incomplete)

**Shared with the estimator by construction, therefore INVISIBLE to C-SG** — all six, not one:
`w_pop`, `f_k`/`f̄`, `S̄_φ`/`S_4D`, `P_det`, `dist()`/cosmology, and the z-domain
`[1e-6, HOST_DRAW_Z_MAX]`. **This includes the M1-vs-comoving population misspecification from which
row #138 predicted 87% of production's dark-class score.** Also invisible: any **h-independent**
misnormalization, which cancels in the normalized posterior.

**Consequence, registered now:** an ESTIMATOR-SELF-CONSISTENT verdict is **conditional on all six by
name**, and auditing `S̄_φ` (never independently audited) is the designated next step under that
branch — not a later discovery.

## 8. A14 — this registration's falsifier

If C-SG returns ESTIMATOR-SELF-CONSISTENT, the attribution "B-SEL's −0.1083 is generator–model
mismatch" stays **provisional** until: re-run **one** C-SG-F seed with B-SEL's donor-row + quality-filter
machinery reinstated, nothing else changed. Bias reappearing at the −0.11 scale confirms the
attribution and localises the mismatch to those stages; otherwise it is refuted. ≈ 45 min.

## 9. Run this BEFORE spending 51–69 CPU-h (pre-check O2, adopted)

The 12 banked B-SEL diagnostics carry `alpha_G_phi`, `L_cat_no_bh`, `B_num`, `D_tilde_phi` per event
per h. **Recompute the posterior with `L_cat_no_bh ≡ 0`** — the pure-completion arm — at zero
compute. If the impostor leg carries part of the −0.1083, C-SG's design must change before it runs;
if not, §5's mismatch is quantified rather than asserted. **This is the next action.**

## 10. What this does NOT do

Does not re-open the #29/#55 zero-host exoneration (row #147); does not re-open B-DEN's
MEASURE-NOT-IT (that was the *estimator*-side measure; §0 item 1 is the *generator* side and now
resolves to the linear draw); does not touch production; does not resolve the `B_num = 0` cause left
open by row #147 item 6. No production code changes.

---

*(FREEZE LINE — after this file is committed, no edits above this line; append VERDICT blocks below.)*

---

## PRE-CHECK O2 — BAND REGISTRATION (appended pre-data, 2026-08-21)

Scorer: `decompose_impostor_leg.py`, committed before `delta_bias` was ever computed. Decision
statistic: `Δ_bias = mean₁₂(mean_h_pure) − mean₁₂(mean_h_full)` under the row #146 corrected
combine on `H_GRID_41`, where `pure` sets `L_cat_no_bh ≡ 0` by exact subtraction
(`combined − (α_G_φ/r_Malm)·L_cat/D̃_φ`).

**A15 statement:** this is a deterministic paired recomputation on fixed banked data — the paired
difference's sampling variance is exactly zero, so *no statistical band is applied* (the A-7
counter-example recorded in A15's evidence is exactly the mistake being avoided). The bands are
**materiality thresholds referenced to the downstream decision** (does C-SG's design change?):

| band | condition | consequence |
|---|---|---|
| IMPOSTOR-SUBSTANTIAL | \|Δ_bias\| ≥ 0.0110 (10% of 0.1083) | rows #137/#140 "pure completion carries it" language revisited AND C-SG design change |
| IMPOSTOR-MATERIAL | \|Δ_bias\| ≥ 0.0023 (C-SG's best 15-seed SE, 0.009/√15) | C-SG design change before it runs |
| IMPOSTOR-IMMATERIAL | \|Δ_bias\| < 0.0023 | §5's mismatch quantified below C-SG resolution; C-SG proceeds unchanged |

Validity: GATE I (identity ≤1e-9 rel, all cells), GATE F (full-arm fleet bias reproduces −0.1083
to ≤5e-5; in-scorer moments ≡ `compute_seed_statistics` to ≤1e-12), GATE P (§0 item 5 impostor-share
quantiles reproduced on seed900101@0.73). Gates fail ⇒ Δ_bias may not be read. Design-time
sightings disclosed in the scorer docstring. REPORTED-ONLY: pure-arm map/σ_h/r_low/c68, per-event
score-at-truth decomposition, physics-floor exclusion counts, the 3 unbanked bsel CSV dirs
(900113–900115) which carry no banked JSON and are not scored.

### O2 GATE AMENDMENT 1 (2026-08-21, pre-read of Δ_bias; both failures diagnosed to the cell)

First run: GATE F **PASS** (fleet bias −0.10830227, dev 2.3e-6 from record; moments ≡
`compute_seed_statistics` to 0.0), GATES I and P **FAIL**. Per registration Δ_bias was not read as
a verdict. Diagnoses, each verified to the mechanism:

1. **GATE I** — max relative identity error is uniformly 5.0–5.5e-7 on all 12 seeds. Cause:
   `bayesian_statistics.py:4365` writes `alpha_G_phi`, `r_Malm`, `D_tilde_phi` (the `_seven_sf`
   tuple) at **7 significant figures**; 7-sf quantization has max rel error 4.9e-7 (measured), three
   quantized columns compound to the observed level. The identity **holds at the storage precision
   of the banked columns** — a convention error would be O(1). Tolerance re-set to **2e-6** (3×
   per-column bound). Propagation to the decision statistic: worst-case per-event
   δln(pure) ≤ 1.5e-6·(cat/pure) ≤ 2e-5 (max share 0.923), summed effect on Δ_bias ≲ 1e-5 —
   **4600× below the 0.0023 MATERIAL band**; the bands are unaffected.
2. **GATE P** — the §0 item 5 quantiles were reproduced **exactly** (all seven, to the quoted
   digit: med 6.02e-4, p90 0.0568, p99 0.647, max 0.821, mean 0.0335, 2 events >0.5) under the
   registration-time verifier's convention: `α_G_φ·L_cat/(α_G_φ·L_cat + B_num)` — **omitting the
   1/r_Malm factor** — with quantiles over the **128 active events only**. The assembly-true share
   (β_G_φ = α_G_φ/r_Malm, verified by GATES I+F against the banked `combined` column) is *larger*:
   active-events med 1.57e-3, p90 0.136, p99 0.822, **max 0.923, 5 events >0.5**. GATE P is re-set
   to assert the verifier's numbers under the verifier's convention (provenance), and the
   β-convention shares become the descriptive numbers of record ("verifier output is evidence, not
   authority" — the prereg's §0 item 5 understated the impostor share).

Scorer updated accordingly and re-run; no band or decision statistic was changed.

---

## PRE-CHECK O2 — VERDICT (2026-08-21, all gates PASS, independently recomputed)

Gates: I PASS (max rel 5.5e-7 ≤ 2e-6), F PASS (fleet bias −0.10830227, dev 2.3e-6; moments ≡
`compute_seed_statistics` exactly), P PASS (all 8 targets). Independent recompute by a separately
implemented agent script (forbidden from reading the scorer) reproduced every per-seed mean and the
fleet numbers to 10 decimals.

> **Δ_bias = +0.0791883246 ⇒ IMPOSTOR-SUBSTANTIAL.** The impostor catalogue leg carries **73.1%**
> of B-SEL's −0.1083 (pure-completion arm: −0.0291). Positive in **12/12 seeds** (range +0.030 to
> +0.164). The pure arm un-rails (map 0.600–0.850, r_low 2/12 vs 12/12), c68 recovers in 5/12, and
> the mean per-event score at truth moves −0.28 → −0.06. Consequences per the registered band:
> **(a)** rows #137/#140's "pure completion carries it" attribution must be revisited; **(b) C-SG's
> design must change before it runs** (→ pre-check O3 and the v3 scoring-channel design below).
> Assembly-true impostor share (β convention, of record, seed900101@0.73, active events): med
> 1.57e-3, p90 0.136, p99 0.822, max 0.923, 5 events >0.5 — larger than §0 item 5's α-convention
> figures.

## PRE-CHECK O3 — BAND REGISTRATION (appended pre-data, 2026-08-21)

Motivated by O2's structure, registered before any O3 number was computed. The mixture
normalization splits as `D̃_φ = α_G_φ + β_Ḡ_φ` (`bayesian_statistics.py:2427`). B-SEL draws events
**conditioned on dark-detected**, so the model-matched conditional likelihood for its draw is the
dark-sector conditional `L_matched = B_num/β_Ḡ_φ = B_num/(D̃_φ − α_G_φ)`; O2's pure channel
`B_num/D̃_φ` differs by the event-independent tilt `ln(D̃/β_Ḡ)(h) = −ln(1−w̃_G(h))`, amplified ×n
in the seed posterior — the registered candidate owner of O2's −0.0291 residual. If the completion
leg is internally self-consistent and B-SEL's draw matches the model's dark-detected density,
`E[∂_h ln L_matched] = 0` at truth.

Scorer: `decompose_matched_channel.py`, committed pre-data. Statistic:
`bias_matched = mean₁₂(mean_h_matched) − 0.73` (row #146 combine, `H_GRID_41`). Same A15
deterministic-read statement as O2; same materiality scale:

| band | condition | meaning |
|---|---|---|
| MATCHED-CONSISTENT | \|bias\| < 0.0023 | completion leg self-consistent at C-SG resolution on B-SEL data; C-SG v3 confirms with a clean generator |
| MATCHED-SMALL | 0.0023 ≤ \|bias\| < 0.0110 | residual at C-SG resolution; a-priori attributable to B-SEL's known generator-side caveats (f_k-vs-f̄ pixel mismatch, donor rows, σ draw); C-SG v3 adjudicates |
| MATCHED-INCONSISTENT | \|bias\| ≥ 0.0110 | completion leg itself carries a substantial defect; rows #137/#140 partially reinstated in the matched channel |

Gates: T (α_G_φ, D̃_φ h-only across events to ≤2e-6; β_Ḡ > 0 everywhere), F2 (full-channel fleet
reproduces −0.1083 ≤ 5e-5). REPORTED-ONLY: analytic tilt slope and ×n amplification per seed;
matched-channel per-seed stats; b0 catalogue-sector corroboration (`L_cat/r_Malm`, 25 seeds) —
EXPLORATORY, convention not independently verified, carries no verdict.

## PRE-CHECK O3 — VERDICT (2026-08-21, gates T + F2 PASS)

> **bias_matched = −0.0846, per-seed sd 0.0329, SEM 0.0095 ⇒ MATCHED-INCONSISTENT.** The tilt
> `ln(D̃/β_Ḡ)` is measured at −0.133/h per event (≈ −24 nats/h per seed) and owns the pure−matched
> gap. Three-channel decomposition of −0.1083: **matched −0.0846 ⊕ tilt ≈+0.055 ⊕ impostor −0.079**.
> EXPLORATORY: b0's catalogue-sector conditional is **+0.0402** — opposite sign. Ledger row #150.

## C-SG v3 — DESIGN CHANGE (2026-08-21, mandated by O2's fired band; append-only)

**The generator is UNCHANGED from v2 design B (§2–§4).** What changes is the scoring, per O2+O3:

1. **PRIMARY channel = the MATCHED channel** `L_matched = B_num/β_Ḡ_φ` — the model-matched
   conditional for C-SG's dark-detected draw. The primary statistic (per-event score at `h_gen`,
   §6) and **BAND C** now apply to this channel. Rationale: the full mixture provably carries a
   −0.079-scale impostor-leg mismatch (O2) that would swamp any internal-defect signal BAND C
   exists to detect; the matched channel is the completion leg's own conditional self-consistency
   test, and it is the channel in which B-SEL measures −0.0846 (O3).
2. **SECONDARY, reported-only:** the full-mixture posterior (its offset from the matched channel
   measures C-SG's impostor + composition mismatch, to compare against B-SEL's structure) and the
   pure channel `B_num/D̃_φ` (tilt bookkeeping).
3. **The reading, restated for the matched channel:** ESTIMATOR-SELF-CONSISTENT (matched-channel
   score ≈ 0) ⇒ B-SEL's −0.0846 is owned by its residual generator-side caveats (pixel f̄-vs-f_k,
   donor rows, σ borrowing, quality filter) — and each becomes individually testable per §8's
   falsifier. INTERNAL-DEFECT (matched-channel score reproduces the −0.08 scale) ⇒ the completion
   leg's misnormalization is banked as an estimator defect with the production-facing consequence
   that follows. MIXED per the pilot's bands. **The branch comparison is a fresh [RULE] for the
   author in either case.**
4. **Diagnostics obligation:** every C-SG seed banks the same per-event diagnostics CSV columns
   (`alpha_G_phi`, `D_tilde_phi`, `B_num`, `L_cat_no_bh`, …) so all three channels are recomputable
   at zero compute, exactly as O2/O3 were.
5. Pilot mandate, GATE V/H/Q/D, GATE S regression, and the post-pilot band-setting discipline are
   unchanged — with GATE S and BAND C evaluated on the matched channel. GATE V's vacuity targets
   apply to the matched-channel posterior.

## PILOT BAND-FORMULA REGISTRATION (2026-08-21, appended pre-pilot-data)

Implementation landed (commit `7ab5f001`; GO from both adversarial lenses; gates H/Q/D pass all
arms/h_gen; GATE Q measured the unrescaled draw at 43% non-PD — the cross-covariance rescale is
active and takes it to 0.0%). Pilot = 4 C-SG-F seeds 910101–910104, job **6415588**.

Registered BEFORE any pilot JSON is read (`csg_pilot_bands.py`, committed with this block):
the band **formulas** are fixed now; the pilot's σ̂ fills in the numbers, which then freeze.

- References (computed pre-pilot from banked B-SEL, zero compute): **S_REF = −0.1932 ± 0.0264**
  (matched-channel per-event score at truth, 12 seeds, per-seed sd 0.0914) and **B_REF = −0.0846**.
- Primary (score, matched channel): SELF-CONSISTENT `|S̄₁₅| ≤ 3·σ̂_score/√15`;
  INTERNAL-DEFECT `S̄₁₅ ≤ S_REF/2 = −0.0966` and outside the self-consistent band; else MIXED.
  Confirmatory tri-band on `bias` with B_REF/2 = −0.0423. Disagreement ⇒ MIXED.
- False-fail table published with the 3-dof sd caveat (95% CI multipliers 0.57×/3.73×).
- GATE S: CONTROL-VALID `|ŝ−1| ≤ 3·SE(ŝ)`; CONTROL-INERT if CI(ŝ) ∋ 0 ⇒ STOP.
- BAND R: F vs E **declared independent** (v2: σ-draw desynchronizes the shared RNG stream;
  pairing would need stream surgery); threshold `3·√2·σ̂_seed/√15`.
- N-adequacy (A15): fleet launches only if the half-reference effect is ≥5σ detectable at N=15;
  otherwise STOP and return to the author (N is not silently changed).
- GATE V roll-up: ≥2 of 4 pilot seeds failing ⇒ STOP.

## PILOT GATE V AMENDMENT (2026-08-21) — the registered STOP fired, was diagnosed, and the gate is re-derived on independent reference data

**The STOP fired as registered:** pilot job 6415588 completed 4/4 (40–54 min/seed, on-anchor), and
the pre-committed band-setter returned `fleet_may_launch=false` — 3/4 seeds failed GATE V (spans
2.15/4.10/4.76 nats < 5; σ_h 0.048–0.060 > 0.5·σ_prior). **No fleet was launched on a fired STOP.**

**Diagnosis, on data independent of the pilot:** v2 §6 wrote GATE V's numbers (5 nats, 0.5·σ_prior)
against the FULL-channel posterior; the v3 design change ported them to the matched channel without
re-deriving operating characteristics — an A15-class omission by the orchestrator, recorded in the
retrospective ledger. Applied to the 12 banked **B-SEL** matched posteriors (the known-informative
reference carrying O3's −0.0846), the v2 thresholds false-fail **5/12 (42%)** (spans 2.01–14.45,
σ_h/σ_prior 0.26–0.81). A gate false-failing 42% of known-informative reference data carries no
verdict — A15's own logic, applied symmetrically.

**Amended thresholds, derived from the vacuity signature itself** (§0 item 4: the failure mode is a
FLAT log-posterior — span ≡ 0 nats, σ_h/σ_prior ≡ 1.0, mean at the weight-convention flat value):
`span ≥ 1 nat` AND `σ_h ≤ 0.9·σ_prior`; the flat-mean coincidence is REPORTED (a matched posterior
genuinely centered at 0.73 is the self-consistent expectation and can never fail on mean alone).
Operating characteristics published: reference false-fail **0/12 (B-SEL) + 0/4 (pilot) = 0/16**;
the B-F1 flat mode (span 0.0, ratio 1.0) fails BOTH prongs — the gate remains can-fail. The v2
verdicts stay recorded in every banked JSON (`v2_span_pass`/`v2_sigma_pass`) so the fired STOP stays
reproducible. σ_prior convention ((b−a)/√12 of the grid) remains flagged for author review.

**No band, statistic, or reference in the BAND-FORMULA registration is touched by this amendment.**

---

## C-SG v3 — FLEET VERDICT (2026-08-21, 46/46 seeds, frozen bands, pre-committed scorer)

Jobs 6415588 (pilot) + 6420343 (fleet), 46/46 COMPLETED, 0 missing, **0 amended-GATE-V failures**.
Scorer `csg_fleet_readout.py` (committed pre-fleet-data); decisive numbers independently
re-derived by the orchestrator from raw diagnostics (seed-level bit-match).

> **BAND C = INTERNAL-DEFECT, on both registered statistics.**
> S̄₁₅ = **−0.1173** (band edge −0.0966; SELF-CONSISTENT would need |S̄| ≤ 0.0373) and
> bias₁₅ = **−0.0665** (edge −0.0423; SELF-CONSISTENT ≤ 0.0209). Per §1/v3: this is the branch
> under which row #140 is *promoted* from PROVISIONAL to a banked estimator-defect claim —
> **recorded here as the registered outcome; the promotion itself is a fresh author [RULE].**

Cross-arm structure (matched channel):

| arm | h_gen | n | bias | per-event score at h_gen | sd |
|---|---|---|---|---|---|
| csgf | 0.73 | 15 | −0.0665 | −0.1173 | 0.0392 |
| csge | 0.73 | 15 | −0.0667 | −0.1184 | 0.0399 |
| csgdm | 0.68 | 8 | −0.0863 | −0.1332 | 0.0154 |
| csgdp | 0.78 | 8 | −0.0495 | −0.1131 | 0.0493 |

- **The score violation is h_gen-independent** (−0.113 … −0.133 across all four arms, both σ
  modes) — the signature of a systematic completion-leg misnormalization, not a generator artifact.
- **The FULL channel reproduces the campaign's headline bias in every arm** (−0.1090 / −0.1081 /
  −0.1099 / −0.1044 vs B-SEL's −0.1083) with a fully clean generator — the mixture-composition +
  impostor structure of rows #149/#150 transfers quantitatively.
- **BAND R: CONSISTENT** (F-vs-E gap 0.0002 ≪ 0.0296) — σ-mode invariant.
- **GATE S: CONTROL-INERT-STOP fired by the registered letter** (ŝ = 0.368, SE = 0.186; 3·SE
  brackets 0). Qualification recorded with it: the three arm means are strictly ordered in h_gen
  (0.6437 → 0.6635 → 0.6805), ŝ sits ~2σ from 0 and **3.4σ below 1** — an *attenuated*, not
  absent, h-response; the sub-unit slope is itself a diagnostic (the matched posterior
  under-responds to the generating h, and grid-edge truncation at h_gen=0.68 works *against* this
  reading). The PRIMARY score statistic involves no slope and fires identically in all arms.
  **The meaning of this gate outcome is put to the author as a [RULE]** — the registered INERT
  sentence ("the arm carries no h-information") is contested by the ordered means.
- Conditionality (§7/A10): this verdict is conditional on the six shared invariants by name
  (`w_pop`, `f_k`/`f̄`, `S̄_φ`/`S_4D`, `P_det`, cosmology, z-domain). **Auditing `S̄_φ` — never
  independently audited — is the designated next step**, now doubly so: the matched channel's
  normalization is exactly the `S̄_φ`-built dark-sector integral.
- Cost disclosure: 46 tasks × 40–55 min wall; submitted at the house fleet convention
  `--cpus-per-task=16` for a single-process job (gotcha-7 over-reservation, ≈550 reserved
  core-h vs ≈35 consumed) — flagged as an A6 perf-audit item, matching the banked
  correspondence-fleet convention rather than the prereg's 2-cpu cost line.
- Artifacts: 46 banked JSONs + 46 per-event diagnostics CSVs (89 MB) under
  `csg_pilot_20260821/` with `MANIFEST.sha256`; all three channels recomputable at zero compute.

---

## CORRECTION & REVIEW ADDENDUM (2026-08-21, post-verdict; author-requested adversarial review)

An independent adversarial review (`ADVERSARIAL_REVIEW_CSG_20260821.md`, banked verbatim) attacked
the night's chain. **Every decisive finding was re-derived by the orchestrator before this addendum
was written** (row #152). Corrections to the FLEET VERDICT block above — the block itself is left
unedited per append-only discipline; where they conflict, THIS addendum governs:

1. **FATAL-1 CONFIRMED — the δ-arm "bias" numbers above are wrong.** `csg_fleet_readout.py`
   subtracted the global 0.73 for every arm. Corrected (mean_h − h_gen): matched **−0.0363 (δ−)
   / −0.0665 (F) / −0.0995 (δ+)**; full −0.0599 / −0.1090 / −0.1544; pure +0.0371 / +0.0115 /
   −0.0219. The scorer is fixed (superseded values retained as `bias_vs_073_SUPERSEDED`), and the
   matched-channel bias is in fact strongly h_gen-dependent — only the SCORE is
   approximately h_gen-flat (−0.133/−0.117/−0.113).
2. **FATAL-2 CONFIRMED — the "full channel reproduces −0.108 in every arm" claim is WITHDRAWN.**
   Full-channel `map_h` = 0.600 (the grid floor) in **46/46 seeds**; what is constant is the railed
   posterior's location (~0.62), and "−0.108" is `0.62 − 0.73`. The full channel is railed and
   uninformative (slope in h_gen ≈ 0.055); the "reconstructed from first principles" sentence in
   the readout is withdrawn. The full channel stays REPORTED-ONLY as v3 designated.
3. **MAJOR-1 CONFIRMED — the verdict's margin, restated on realized scatter.** Realized F-arm
   score sd = 0.0751 (1.56× the pilot's σ̂), SEM = 0.0194. **S̄₁₅ = −0.1173 ± 0.0194: non-zero at
   6.05σ (bankable), but past the INTERNAL-DEFECT edge by only 1.07σ** (reviewer's seed bootstrap:
   P ≈ 0.13 of landing MIXED). The INTERNAL-DEFECT *label* is therefore **PROVISIONAL**; the
   bankable statement is: *the implemented `B_num` and `β_Ḡ_φ` are not a matched
   numerator/normalizer pair — their h-derivatives differ by ~10% (−1.222 vs −1.105)*. The frozen
   bands are NOT retuned (anti-tuning); the realized-scatter fields are added to the output.
4. **MAJOR-2 — the N-adequacy gate re-evaluated on realized scatter gives 4.98σ (< the registered
   5)**; queued with the author's A17 ruling (extension: gates re-check operating characteristics
   at readout on realized scatter).
5. **GATE S is VOID-CANDIDATE, superseding the "attenuated" qualification** (MAJOR-3): the
   reviewer's truncation simulation reproduces ŝ ≈ 0.37 from a plain truncated Gaussian with the
   observed widths (my "truncation works against this" sentence was built on the FATAL-1-corrupted
   δ-arm numbers and is withdrawn); the registered rule's branches also overlap on `H_GRID_FULL`
   (both VALID and INERT satisfiable), so the gate is not a partition. Author [RULE] #2 becomes
   "rule GATE S void", not "INERT vs attenuated".
6. **BAND R re-characterized** (MAJOR-8): the σ draw happens AFTER the accept loop, so F and E
   share bit-identical (z, Ω, M) draws — they are PAIRED, not independent (measured corr 0.9975);
   the registered independence rationale was wrong in the implementation's favour. Correct paired
   3σ band = 0.0022; observed gap 0.0002 — the σ-mode null passes an informative band after all.
7. **The discriminating next test is re-targeted** (MAJOR-5/6): the verdict is a ~10% residual
   between two large derivatives computed by DIFFERENT quadratures on DIFFERENT h-dependent
   z-domains (`β_Ḡ_φ`: 1500-node trapezoid to `min(z_max(h),1.55)`; `B_num`: 50-node
   Gauss–Legendre over per-event ±4σ with endpoint-clamped interpolation; generator: fixed
   [1e-6, 1.5]). The identity `∫B_num dx = β_Ḡ_φ` was assumed, never verified, and the z-domain
   is a SEVENTH un-listed invariant that is *not* shared. **Proposed pre-check O4 (the fired
   branch's falsifier, retrofitting MAJOR-9):** re-evaluate `B_num` and `β_Ḡ_φ` on a common
   domain/quadrature (±10σ window, aligned caps, no clamp) and re-read the score. If −0.117 moves
   materially ⇒ numerical-pairing artifact (the §1 reading flips to generator–model/implementation
   mismatch); if it survives ⇒ the defect label hardens. Cheap; awaits the author's [RULE].
8. Also recorded: GATE V's amended prongs have near-zero discriminating power on real posteriors
   (46/46 PASS is not evidence of quality — it is a flat-null detector only); `S̄_φ` audit
   (previous designation) is DE-PRIORITIZED because `S̄_φ` appears in both legs and largely
   cancels; gates H/Q/D raw outputs are now banked (`csg_gate_hqd_outputs.json`).
9. **What the review confirmed clean** (recorded for balance): O2's mechanics (identity to 5.5e-7,
   no leak, r_Malm placement verified), S_REF/B_REF reproduction, the pilot-STOP chronology, the
   f_k/f̄ pixel-marginal pairing (the reviewer's own initial suspicion, refuted by their check),
   physics-floor inertness (0 zero cells in 58 seeds), score-statistic grid-invariance, and zero
   analysis-stage attrition (the cross-covariance rescale worked).
