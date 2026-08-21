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

---

## PRE-CHECK O4 — REGISTRATION (2026-08-21, author-authorized, ledger row #153 item 4; pre-data)

**Question (the fired branch's falsifier, A19):** is the banked matched-channel score
(−0.1173 ± 0.0194 realized) owned by the **domain-and-quadrature pairing** of the two
implementations — `B_num` (per-event, 50-node Gauss–Legendre over the h-dependent window
`[z(d̂−4σ;h), min(z(d̂+4σ;h),1.55)]`, `S̄_φ` via endpoint-clamped `np.interp`) vs `β_Ḡ_φ`
(1500-node trapezoid over `[1e-6, min(z_max(h),1.55)]`) — or does it survive alignment (a deeper
math defect in the estimator)?

**Design (deterministic recomputation, local, 15 F seeds):** regenerate each F seed's event set
via `draw_csg_realization` (deterministic given seed). Arms:

| arm | B_num settings | β_Ḡ settings |
|---|---|---|
| **P (replica)** | production code path, unmodified | production values (column-derived) |
| **A (aligned, primary)** | full common domain `[1e-6, min(z_max(h), 1.55)]`, 1500-node trapezoid on β_Ḡ's own z-grid, **no clamp** — outside `[1e-6, z_max(h)]` the selection factor is exactly 0 (registered choice, matching β_Ḡ's domain) | same rule, same grid, same domain |
| A1–A3 (factorial, REPORTED-ONLY) | one alignment component at a time: window→full-domain; GL-50→trapezoid-1500; clamp→zero-extension | production |

**Statistic:** `S̄₁₅(O4-A)` = fleet-mean per-event matched score at h_gen = 0.73 under arm A,
central difference at 0.725/0.735, same combine conventions.

**Validity gates (can-fail, scored before the statistic is read):**
- **GATE R4 (replica bit-exactness):** arm P's per-event `B_num(h)` reproduces the banked
  diagnostics column bit-exactly (calls the production functions; any mismatch ⇒ the regenerated
  event set or the harness is wrong ⇒ STOP). This simultaneously proves regeneration determinism.
- **GATE T4:** arm A's β_Ḡ at production settings reproduces the column-derived
  `D̃_φ − α_G_φ` to the 7-sf storage precision (2e-6).

**Bands (materiality; deterministic recomputation — A15 no-statistical-null statement as in O2/O3;
thresholds referenced to the frozen SELF-CONSISTENT edge 0.0373 and the realized SEM 0.0194):**

| band | condition | meaning |
|---|---|---|
| PAIRING-OWNS-IT | \|S̄(O4-A)\| ≤ 0.0373 | the violation is the production pairing's numerics — production still runs the mismatched pair, so the finding becomes an implementation-pairing defect (production-facing), not an estimator-math defect; the INTERNAL-DEFECT label is re-worded accordingly |
| DEFECT-HARDENED | \|S̄(O4-A) − (−0.1173)\| ≤ 3·0.0194 | alignment changes nothing within realized resolution ⇒ the mismatch is upstream of quadrature/domain choices; the label hardens |
| PAIRING-PARTIAL | else | report the owned fraction `1 − S̄(O4-A)/(−0.1173)`; both readings carry |

**A19 falsifiers for O4's own branches:** PAIRING-OWNS-IT is falsified if a further-refined
production replica (higher-order production-side quadrature at unchanged domain) moves the banked
score by an amount comparable to the alignment effect (would show the "alignment" gain is generic
refinement, not pairing); DEFECT-HARDENED is falsified by exhibiting any residual
convention difference between the two legs after arm A (checklist audit, zero compute);
PAIRING-PARTIAL carries both.

**A10 note:** O4 deliberately VARIES the seventh invariant (domain/quadrature pairing) that
rows #149–#151 held fixed; the six shared physics invariants remain fixed and named.

*(O4 scorer committed before it runs; VERDICT appended below when it reports.)*

## PRE-CHECK O4 — VERDICT (2026-08-21; as-run band VOID-BY-DEVIATION; mechanism IDENTIFIED)

**Execution:** 15/15 F seeds (4 local survivors + 11 cluster, job 6441957, after a local OOM taught
the ~9 GB/seed lesson); GATE R4 bit-exact 15/15 across BOTH venues; GATE T4 15/15. Merged by
`o4_merge_shards.py` via the committed scorer's own reduction. As-run: S̄(A) = −0.117321 ≈
production −0.117318; factorial: window +0.0006, quadrature/clamp ~1e-6 ⇒ bands fired
**DEFECT-HARDENED**.

**A20 review (first application; `A20_REVIEW_O4_20260821.md`, banked verbatim) — the band is
UNEARNED, and the registered A19 falsifier for DEFECT-HARDENED FIRES:**

1. **VOID-BY-DEVIATION:** the executed arm A dropped the REGISTERED S̄_φ zero-extension ("outside
   [1e-6, z_max(h)] the selection factor is exactly 0") on a post-registration "corrected premise"
   (production's `off` cell never queries S̄_φ) — disclosed in a docstring, **bands never
   re-derived**. The registered arm, restored: **S̄₁₅ = +0.0076 ± 0.0184 (0.41σ from zero) ⇒
   PAIRING-OWNS-IT**. Per-seed shift +0.1249 ± 0.0012 = 106.5% of the banked score.
   **Orchestrator re-derivation:** 3-seed locus rerun reproduces the as-run shard scores exactly
   and the restored shifts (+0.120…+0.127).
2. **The mechanism, named:** the completion numerator under the pinned runs-of-record basis
   (`PRODUCTION_FLAGS: selection_in_completion_numerator="off"`, verified) **omits the S̄_φ
   survival factor its own normalizer β̄_Ḡ_φ carries and the generator applies at accept time** —
   the legacy pre-#118 cell, labelled *"not a production posterior"* by the estimator's own log
   line in every C-SG shard; the `fused` cell is the in-tree fix (rows #117–#118).
3. **O5 (free cross-check, banked B-SELF = mirror generator + fused cell, 11 seeds):** matched
   score **−0.0637 ± 0.0188**, bias −0.0364 — the omission owns ~⅔ of the mirror arm's violation;
   the 3.4σ residual is provisionally attributed to B-SELF's known generator caveats, consistent
   with the clean-generator restored-arm null.
4. **Registration failures recorded against the orchestrator (A15-class):** the domain/quadrature
   axis had ~6000× too little leverage to reach any band but DEFECT-HARDENED (computable in one
   line pre-data — an axis-leverage calculation now joins the A17 checklist); A1's GL-50
   full-domain arm is numerically invalid (<1 node/σ); R4's two-venue bit-exactness is one
   determinism check, powerless against a shared regeneration error; 3/15 R4 rows compared cached,
   not regenerated, artifacts.
5. **What is BANKED (restated, typed MEASURED):** the 6.05σ non-zero matched-channel score is a
   real, reproducible numerator/normalizer mismatch **of the `off` cell, mechanism identified
   (missing S̄_φ)** — not a deeper estimator-math defect. The INTERNAL-DEFECT label resolves to
   **IMPLEMENTATION-CONVENTION DEFECT (off-cell S̄_φ omission)**, pending the author's ratification.
6. **Registered next measurement (author [DO]):** ONE C-SG-F seed end-to-end under `fused` (both
   legs, not a numerator patch); expected matched score ≈ 0. Then the production-facing fork: the
   runs-of-record basis is the off cell — whether production switches basis is a physics-change
   [RULE] with the full gate protocol.

## CONFIRMATION RUN O6 — REGISTRATION (2026-08-21, author-approved: ledger row #157 item 2; pre-data)

**Authorization:** row #157 item 2, author's selection verbatim: "Approved — register & run". Row
#157 item 1 ratified the defect label (IMPLEMENTATION-CONVENTION DEFECT, off-cell S̄_φ omission);
this run is its registered confirmation. A21 governs this text: any premise correction discovered
during implementation STOPS the run; this registration is amended and bands re-derived BEFORE any
`evaluate()` executes.

**Question:** does the REAL `fused` cell — `selection_in_completion_numerator="fused"`, executed
end-to-end inside `BayesianStatistics.evaluate()` (both legs: the numerator via
`completion_numerator_integrand_sel_1d`, `bayesian_statistics.py:4992-5007`, and the normalizer
`β̄_Ḡ_φ` via `precompute_phi_selection_integrals`, `:2019-2066`, both reading the SAME
`precompute_phi_marginal_survival` table) — null the matched-channel violation for a C-SG-F seed,
as the identified mechanism predicts? This is an end-to-end code-path run, NOT a numerator patch:
O4's restored arm simulated S̄_φ restoration in a standalone harness; O6 runs the in-tree cell.

**Seed (registered choice):** **910101** — the first registered F seed (registry start, prereg
§4), pilot-batch member, locally regenerable. Chosen by registry order, not by score.

**Arms (exact text; the executed configuration may not deviate from this table — A21):**

| arm | what runs | purpose |
|---|---|---|
| **D6 (off replica, gate)** | `run_csg_arm_seed(work_root=FRESH_DIR, "csgf", 910101, ...)` with the new `selection_in_completion_numerator` passthrough parameter present but set to `None` (→ pinned production `"off"`), in a **fresh work root** so the idempotent-skip cannot substitute cached artifacts (the O4/A20 3/15-cached finding) | proves the plumbing edit is inert on the off path AND re-proves regeneration determinism for this seed |
| **F6 (fused, primary)** | same call with `selection_in_completion_numerator="fused"`, fresh work root | the measurement |

The only code change permitted for O6 is the plumbing passthrough: `run_csg_arm_seed`
(`selfgen_control.py:1367-1478`) gains an optional `selection_in_completion_numerator: str | None
= None` forwarded verbatim to `run_mirror_seed_inprocess` (`correspondence_1d.py:1723-1733`, which
already accepts it and defaults to the pinned `"off"`). `None` preserves byte-identical current
behavior. No physics-trigger file is edited.

**Primary statistic:** `S(F6)` = the per-event matched-channel score for seed 910101 at
h_gen = 0.73, central difference at 0.725/0.735, computed by the SAME committed
`csg_channel_scores`/`score_at_h_gen` path every C-SG shard used, read from arm F6's record.

**Registered reference (A18; derived pre-data by instrument, value appended below before any
`evaluate()` runs):** `r_prod(910101)` = a harness replica of the fused-cell numerator at
PRODUCTION settings — per-event window `[z(d̂−4σ;h), min(z(d̂+4σ;h), 1.5)]`, GL-50 quadrature,
S̄_φ applied via endpoint-clamped `np.interp` on the `precompute_phi_marginal_survival` grid (a
literal convention copy of `completion_numerator_integrand_sel_1d`), divided by the SAME
`β̄_Ḡ_φ`, scored by the same `score_at_h_gen`. REPORTED-ONLY companion: `r_A(910101)` = O4's
aligned arm A with S̄_φ restored (full-domain trapezoid) — the orchestrator's independent
re-derivation of the A20 review's restored-arm value for this seed (review fleet numbers:
restored S̄₁₅ = +0.007604 ± 0.018361; per-seed shift +0.124925, sd 0.004625). The instrument
(`o6_reference_derivation.py`) is zero-`evaluate()` (banked-CSV inputs) — costing line: < 5 min
wall, < 2 GB RSS, local.

**Bands (applied to `S(F6)`; both reachable, see axis-leverage):**

| band | condition | meaning |
|---|---|---|
| **MECHANISM-CONFIRMED** | \|S(F6) − r_prod(910101)\| ≤ **1e-4** | the real fused cell behaves exactly as the identified mechanism predicts; the off-cell S̄_φ omission account is complete at production numerics |
| **REPLICA-BROKEN** | else | the in-tree fused cell and the harness replica of its own convention disagree ⇒ the mechanism account (or the replica) is incomplete — STOP; zero-compute factorial audit of the discrepancy BEFORE any interpretation; no label change may be argued from this outcome |

δ = 1e-4 derivation: the O4/A20 harness reproduced the banked `B_num` column to 3.4e-15 relative
and GATE R4's registered fallback for multiprocessing float-order is 1e-12 relative; propagated to
the log-derivative score both are < 1e-6. 1e-4 is ~100× above that noise floor and ~1250× below
the axis effect (+0.1249). SECONDARY, REPORTED-ONLY (explicitly NOT a band): \|S(F6)\| vs the
frozen SELF-CONSISTENT edge 0.037339 — at single-seed level this is a realization statement, not
a fleet null (per-seed realization scatter σ̂ ≈ 0.075; seed 910101's own restored expectation is
≈ −0.029, i.e. a clean-mechanism seed may sit nonzero); the fleet-level "score ≈ 0" claim is NOT
adjudicated by one seed and is not registered here.

**Axis-leverage statement (A17):** the registered axis (off→fused numerator S̄_φ convention) has
MEASURED per-seed leverage +0.1249 ± 0.0046 (A20 review, orchestrator-re-derived on a 3-seed
locus: shifts +0.120…+0.127). Band half-widths 1e-4 (primary) ⇒ leverage/width ≈ 1250×; the
banked off-cell score for this seed (−0.154062) sits 1540 half-widths from r_prod's predicted
locus (≈ −0.029): both bands are reachable and the primary band CAN fail. The identity statistic's
noise floor (≤1e-6, above) is 100× below the half-width.

**Validity gates (scored before the statistic is read; any failure ⇒ VOID, not a band):**

- **GATE D6 (off replica):** arm D6's `B_num` diagnostics column vs the banked
  `csgf_seed910101/event_likelihoods.csv`: bit-exact, or ≤ 1e-12 relative under the registered
  multiprocessing-float-order fallback. Wall time must exceed 60 s (anti-idempotent-skip check —
  a 0.3 ms "regeneration" is the cached-artifact signature).
- **GATE T6 (normalizer invariance):** arm F6's column-derived `D̃_φ − α_G_φ` at 0.725/0.735
  equals the banked off-run values to 2e-6 relative — the cell switch must not move the
  normalizer leg (it is built unconditionally under `absolute_marginal`).
- **GATE L6 (cell identity, zero compute):** arm F6's log does NOT contain the off-cell
  "not a production posterior" counterfactual line; its run metadata records
  `selection_in_completion_numerator="fused"`; arm D6's log DOES contain the off-cell line.
- **GATE V6 (anti-vacuity):** arm F6's `B_num` column differs from arm D6's on > 99% of rows
  (the numerator must actually have changed; guards a silent fall-through to the off dispatch).

**A19 falsifiers:**

- MECHANISM-CONFIRMED is falsified by (i) any gate failure (⇒ VOID); (ii) a zero-compute code
  audit exhibiting a convention difference between `o6_reference_derivation.py`'s replica and
  `completion_numerator_integrand_sel_1d` (replica-circularity: identity can hold while replica
  and cell share an error — the audit is the check, and it is REQUIRED before the verdict is
  banked); (iii) the fused numerator's S̄_φ table failing code-identity with the table
  `precompute_phi_selection_integrals` integrates (would void the "both legs, same S̄_φ" claim).
- REPLICA-BROKEN is falsified by showing the discrepancy is owned by multiprocessing summation
  order (re-run arm F6 single-process; if the identity then holds to 1e-12, the primary band
  re-fires under the registered fallback).

**Costing line (A6/A17):** 2 × `evaluate()` (arms D6, F6): anchor 0.478 CPU-h/seed, ~29–45 min
wall each single-process (n=200), peak RSS ≈ 9 GB each. **Venue: LOCAL dev box (30 GB),
sequential, 1-wide** (the O4 OOM was 12-wide; 1-wide fits with >3× headroom); cluster fallback =
the `o4_fleet.sbatch` pattern if the box is contended. Instruments < 5 min. Total ≈ 1.5–2 h wall.

**A10 note:** all six shared physics invariants remain fixed; O6 varies exactly one axis — the
numerator's completion cell (`off`→`fused`) — through the production dispatch itself.

**A18 note:** every O6 output JSON carries machine-readable `reference` fields naming what each
statistic subtracts: the primary subtracts `r_prod(910101)` (provenance: this registration +
`o6_reference_derivation_output.json`); the reported-only secondary subtracts 0 with the frozen
edge 0.037339 named.

*(O6 scorer + instrument committed before any arm runs; reference values appended below pre-data;
VERDICT appended when it reports.)*

## O6 — REFERENCE-VALUE REGISTRATION (2026-08-21, appended pre-data; no `evaluate()` has run)

Instrument `o6_reference_derivation.py` (zero-`evaluate()`, banked-CSV inputs; wall < 3 min,
RSS ≪ 2 GB, local — within the registered costing line):

- **r_prod(910101) = −0.02669443370359812** (the registered primary reference; A18 provenance:
  `o6_reference_derivation_output.json`).
- r_A(910101) = −0.02667690999014469 (REPORTED-ONLY companion). Consistency: shift from the
  banked off-cell score −0.154063 is **+0.12737** — inside the A20 review's independently-derived
  per-seed shift range (+0.120…+0.127, fleet +0.124925 sd 0.004625) — the orchestrator's own
  re-derivation corroborating the review's restored-arm value for this seed; r_prod − r_A =
  1.75e-5, consistent with O4's measured window/quadrature leverages.
- **Bands now numeric:** MECHANISM-CONFIRMED iff S(F6) ∈ [−0.026794, −0.026594]; else
  REPLICA-BROKEN; any gate failure ⇒ VOID.

**Registered falsifier audits (A19 items ii/iii) — executed pre-data, both PASS:**
(ii) the harness replica (`o6_reference_derivation.py:103-132`) is a convention-faithful copy of
the in-tree fused dispatch (`bayesian_statistics.py:4992-5007` + `:5120-5179`): same window incl.
1e-6 floor / 1.5 cap / degenerate→0, `fixed_quad` n=50 (`_HOST_QUAD_N` env-clean: no `MTC_*` set),
endpoint-clamped `np.interp`; (iii) `_phi_survival_table` is passed as the SAME OBJECT to
`precompute_phi_selection_integrals` (`:3813-3821`) — numerator and normalizer read one S̄_φ table.

**Disclosed instrumentation details (band-inert, A21 note):** GATE L6's log capture attaches to
the ROOT logger, not a `darksiren_emri`-named logger — `bayesian_statistics.py`'s `_LOGGER` is
the root logger, so a named-logger handler would never see the gate lines; the gate's registered
content conditions are unchanged. The instrument's per-event B_num-ratio diagnostic compares
against the banked off-cell column (diagnostic-only; feeds no band). `run_csg_arm_seed`'s
passthrough parameter landed in-tree before the scripts (same session, registered scope).

*(D6/F6 execute next; VERDICT appended when the committed scorer reports.)*

## CONFIRMATION RUN O6 — VERDICT (2026-08-21; MECHANISM-CONFIRMED, banked WITH the A20 amendments)

**Execution (registered venue, local sequential 1-wide):** arm D6 wall 1852 s, arm F6 wall 1811 s,
peak within the ~9 GB/seed anchor; no `MTC_*` overrides; HEAD = `50476453` = the `git_commit` in
both arm records. **All four gates PASS and each was fail-able:** D6 bit-exact 9200/9200 rows
(genuine regeneration, wall-time floor honored — the O4 cached-artifact hole stayed closed);
L6 both directions (off line present in d6.log, fused line present / off line absent in f6.log);
T6 normalizer identical; V6 numerator changed on 100% of rows.

**Primary [MEASURED]:** S(F6) = −0.02669249255841575; registered reference
r_prod(910101) = −0.02669443370359812; **delta = +1.9411e-6, 50× inside the ±1e-4 band ⇒
MECHANISM-CONFIRMED** (registered leverage check realized: off→fused moved this seed +0.127368 =
1274 band half-widths). REPORTED-ONLY: r_A = −0.0266769; |S(F6)| = 0.0267 vs frozen edge 0.0373.

**A20 review (second application; `A20_REVIEW_O6_20260821.md`, banked verbatim):**
**BANK-WITH-AMENDMENTS, zero FATAL.** The reviewer re-derived S(F6), r_prod, r_A and the delta
from scratch (exact agreement), diagnosed the entire +1.94e-6 residual as the 7-sf CSV storage of
the normalizer columns (per-event B_num agreement 3.5e-15; orchestrator verified the storage
arithmetic independently), and confirmed falsifier (iii) with its own chain. The four amendments
below are ADOPTED into this verdict as its binding reading:

1. **Scope:** MECHANISM-CONFIRMED = the in-tree fused numerator IS the pre-registered harness
   replica of its own convention (machine precision) — the **harness→production transfer** of the
   O4/A20 restored-arm result. "The off-cell S̄_φ omission account is complete" is withdrawn and
   replaced by "…complete **as an implementation account of the off→fused difference, for this
   seed, at production numerics**".
2. **The approved "matched score ≈ 0" question is NOT yet answered** — the single-seed null was
   demoted REPORTED-ONLY pre-data for lack of power (σ̂_seed ≈ 0.075 vs edge 0.037). |S(F6)| =
   0.0267 (0.43σ on its own SEM) is consistent with, not evidence for, the null. The fleet-level
   "fused nulls the matched channel" claim is **OPEN** and returns to the author as a fresh
   [DO]/[RULE]; note the fused cell's per-seed power collapse (span 1.53 vs 7.59 nats) for its
   costing.
3. **Residual blind spot (disclosed):** `precompute_phi_marginal_survival` itself is common-mode
   to replica and cell — untested by every O6 gate.
4. **A17 fold-in:** identity-band noise floors scored off `event_likelihoods.csv` must be derived
   from the CSV's 7-sf stored precision (measured 1.94e-6 here), not the estimator's internal
   precision; the pre-data "+0.120…+0.127 range" consistency sentence corrected (+0.127368 is
   0.0004 outside the quoted range, within 1 sd of the fleet mean).

**Carried caveat:** F6's full channel remains mean_h = 0.618, r_low — the fused cell does not
cure the H₀ rail; runs-of-record remain on the `off` cell (row #157 item 3 stands).

**Amendment credits:** A20 +1 (second application: narrowed an over-broad verdict sentence before
banking), A21 +1 (first fully-clean registration–execution identity under its own rule), A17
evidence extended (CSV-storage noise-floor term), A18 exercised (all O6 outputs carry `reference`
fields).

## O7 — FLEET TRANSFER CLOSE: REGISTRATION (2026-08-22, author-approved: row #159 item 1 "A: Transfer + spot-check"; pre-data)

**Question:** does the fleet-level claim "the `fused` cell nulls the matched-channel score"
close by measured transfer — the banked per-seed reference fleet (committed code) plus O6's
proven harness↔production identity, stress-tested on the two most extreme seeds?

**Arms (exact text; A21 governs — any premise correction STOPS and re-registers):**

| arm | what runs | cost |
|---|---|---|
| **R7 (reference fleet, instrument)** | `o7_reference_fleet.py`: the committed O6 reference computation (`o6_reference_derivation.compute_reference`, unmodified) looped over ALL 15 F seeds (910101–910115), banking the per-seed r_prod and r_A vectors + fleet mean/SEM from committed code — repairing the review-scratchpad provenance gap | zero-`evaluate()`; ~20–30 min local, ≲4 GB |
| **S7 (spot-checks, 2 seeds)** | seeds **910105** and **910113** end-to-end under `fused` via the committed O6 driver pattern (fresh work roots, `run_csg_arm_seed(..., "fused")`). Selection criterion (registered, pre-data): the two EXTREMES of the banked off-cell score range — 910105 the most negative (−0.2648), 910113 the sign-flip outlier (+0.0828) — maximal transfer stress, chosen by banked data only | 2 × ~30 min local sequential, ~9 GB each |

Spot-check gates per seed: **L7** = GATE L6's log-content check (fused line present, off-cell
counterfactual line absent); **V7** = GATE V6's anti-vacuity (fused `B_num` differs from the
banked off CSV on > 99% of rows). The off-replica gate (O6 D6) is NOT re-run — registered
justification: regeneration determinism is proven 15/15 bit-exact across two venues (O4 R4) and
plumbing-inertness proven bit-exact (O6 D6); re-proving per seed buys nothing the bands read.

**Bands (per spot-check seed):** **TRANSFER-HOLDS** iff |S_fused(seed) − r_prod(seed)| ≤ **1e-4**
(floor per A20/O6 amendment 4: the identity noise floor is the 7-sf CSV storage of the
normalizer columns, measured 1.94e-6 on 910101; 1e-4 keeps the ~50× margin). Else
**TRANSFER-BROKEN**; any gate failure ⇒ VOID for that seed.

**The fleet claim (registered wording):** IF both spot-checks fire TRANSFER-HOLDS, the banked
fused-fleet statement becomes: *"the fused-cell fleet matched score at h_gen = 0.73 is the R7
reference fleet, by measured transfer: S̄₁₅(fused) = r̄_prod(15) ± SEM(15), expected ≈ +0.008 ±
0.018 (0.4σ from zero) — the fused cell nulls the matched-channel violation at fleet level, with
per-seed transfer error bounded by the band 1e-4."* IF either fires TRANSFER-BROKEN, the claim
does NOT close; the discrepancy is audited zero-compute before any interpretation and the full
15-seed end-to-end fleet (proposal D1 option B) returns to the author.

**Axis-leverage (A17):** the off→fused axis moves each seed by +0.1249 ± 0.0046 (measured, 15
seeds) = ~1250 band half-widths; the two spot-check seeds sit at the extremes of the off-score
range (span 0.35 = 3500 half-widths), so TRANSFER-BROKEN is reachable if the identity is
seed-dependent in any way the band resolves. Realized-scatter re-check: SEM(15) is recomputed
from R7's banked vector, not carried from prose.

**A18:** every output carries `reference` fields (per-seed statistics subtract r_prod(seed) with
provenance; fleet statistics name R7's banked vector).

**A19 falsifiers:** TRANSFER-HOLDS is falsified by (i) any L7/V7 failure; (ii) a zero-compute
audit exhibiting a convention delta between `compute_reference` and the fused dispatch larger
than the band (the O6 audits (ii)/(iii) already cover this and PASSED — carried, not re-run).
TRANSFER-BROKEN is falsified by a single-process re-run restoring the identity to ≤ 1e-12
(multiprocessing float-order, the registered O6 fallback).

*(R7 + S7 committed before execution; R7 values appended pre-S7-data; VERDICT appended when the
scorer reports.)*
