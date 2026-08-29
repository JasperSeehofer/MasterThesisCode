# PREREGISTRATION — [WIN] k=3 log-window counterfactual (charter node B5.2, stage 2)

**launched under rows #222/#223 — charter node B5.2**

Author-ratified design: row #221 item 2 (F-ii) — "REDESIGN: log-symmetric, k=3,
ε=2Φ(−3)=0.27%, adopt only after a registered counterfactual." This document is that
registration. Per docket 1 §2 B5 ("5.2 is warranted... but the object has changed and the
registration must say so before any CPU-h") and §4.2 item 3. No CPU-h has been spent to
produce this document; it is a pre-code-execution registration (Stage 2 of
`docs/RESEARCH_CYCLE.md`), authored before the C3 sbatch per amendment F3 (shared instruments
register all predictions first).

**Standing-rule-5 exoneration re-check (both layers, mechanism-grepped):** carried forward
from `B5_2_PULL_READ_20260829.md` §"Exoneration check" — neither
`EXONERATION_REGISTER_20260827.md` nor `BIAS_HISTORY_LEDGER.md` §2 (~line 127) name
mass-window *geometry* / true-host retention under any exonerated tag; `WGEO`/`WGEOM` names a
different, already-adjudicated object (hard-truncation-as-support-truncation, HB, +0.0015
ceiling — engaged below as a comparison band, not re-litigated as a bias claim). No collision.

---

## 1. The run

- **Venue:** iiib only (true reduced catalogue, md5 `c52c13b5…`; CRB md5 `9a1f2a14…`;
  `EVAL_SEED=777000`). One venue: `T_mat` was derived as the max over both 2D venues and is
  conservative on joint_r1 (`MEASUREMENT_HEAD_READOUT_20260827.md:268-285`); joint_r1's
  HEAD-config cost is ≥2.2× iiib's (§4.1 of the docket) and is excluded from wave 2 for that
  reason, matching B7.2's precedent.
- **Code state:** wave-2 HEAD, after B6.1's [ALIGN] `[PHYSICS]` commit (must land before any
  h-dependent read, per L8) and B5.1's `[PHYSICS]` commit (the `mass_filter_geometry`/
  `mass_filter_k` flags, byte-identical default). A22 dirty-state stamp = clean at run start.
- **Arms:**
  - **Baseline B** — `mass_filter_geometry="linear"`, `mass_filter_k=1.5` (current production
    default, unchanged). Source: the banked HEAD readout, REUSED at zero compute on every H4
    node only if the same-commit baseline task (shared task **C0**, serving B3.2/B5.2/B7.2 per
    L5) reproduces the banked per-event `L_cat_with_bh`/`combined_with_bh` columns to ≤1e-12
    relative (the row-#201 PROD-A0 ingredient gate, historically passing at ≤8.5e-15);
    otherwise baseline is re-run at this arm's own nodes.
  - **Arm T** — `mass_filter_geometry="log"`, `mass_filter_k=3.0` (the ratified F-ii design).
    `mass_filter_sigma` held at its production value `"symmetric"` in BOTH arms (an invariant,
    §7 below) — per B5.1's disclosed resolution (`B5_1_WIN_RECORD.md` §3), under
    `"symmetric"` the candidate-side multiplier IS `mass_filter_k` under either geometry, so
    varying `mass_filter_k`/`mass_filter_geometry` together is the correct and only way to
    engage the ratified design; `mass_filter_sigma` itself is not varied by this counterfactual.
  - This is arm T vs. baseline B, run through the **shared baseline gate task C0** (L5) — not
    a three-arm design.
- **H4 grid:** `{0.660, 0.665, 0.670, 0.730}` — same stencil logic as B7.2 (docket §2 B5, §2
  B7): 0.730 is the production read (candidate growth, retention, class migration, zero-rate
  census, R6 bit-identity — all at zero *additional* compute once arm T's h=0.730 task
  completes); `{0.660, 0.665, 0.670}` bracket the HEAD 2D MAP (0.665) / mean_h (0.663347),
  giving Δℓ(h) = Σᵢ ln[Lᵢᵀ/Lᵢᴮ]'s slope and curvature at the peak — the first-order predicted
  shift Δmean_h,pred ≈ Δℓ′(0.665)/I_HEAD, **I_HEAD = 2965** (σ_h = 0.018366, 2D iiib
  production Fisher information at MAP, same anchor as B7.2). Baseline B is re-scored on the
  identical 4 nodes at zero compute (deterministic recomputation over banked columns, subject
  to the C0 bit-identity gate above) — only arm T requires new cluster compute.

---

## 2. Design matrix

| | `mass_filter_geometry` | `mass_filter_k` | `mass_filter_sigma` | h-nodes | source |
|---|---|---|---|---|---|
| **Baseline B** | `"linear"` | 1.5 | `"symmetric"` | H4 (4) | banked HEAD readout / C0 zero-compute re-score |
| **Arm T** | `"log"` | 3.0 | `"symmetric"` | H4 (4) | new cluster compute, ledger row C3 |

All other flags are held at the production invariants list (§7).

---

## 3. Primary reading — ΔMAP / Δmean_h vs. HB's +0.0015, two-sided

**Definition, sign convention stated explicitly:**
`ΔMAP ≡ MAP(arm T) − MAP(baseline B)`; `Δmean_h ≡ mean_h(arm T) − mean_h(baseline B)`
(H4-stencil-predicted form: `Δmean_h,pred ≈ Δℓ′(0.665)/I_HEAD`). A **positive** ΔMAP/Δmean_h
moves the estimate upward, the same direction HB's own quoted effect moves it
("removing the window moves the MAP up by ~+0.0015" — window REMOVAL is a different
counterfactual than a geometry change, but shares the sign axis: both relax/reshape the same
mass-eligibility constraint) and the same direction that is corrective at HEAD (HEAD 2D
offset −0.066653 iiib, so upward is toward truth 0.73).

**Comparison object.** HB's bound (`CLAIM_WGEO_20260827.md` §4.1/D-WGEO-1:
ΔMAP ≈ +0.0015, from −0.317 nats × 4.9e-3 h/nat) is a *different* counterfactual — full mass
window **removal** (k→∞), not a geometry swap at fixed finite k — and is scoped as
"support truncation." The physics-change doc's §5 limiting-case derivation shows both
geometries converge to that SAME k→∞ object identically, so HB's +0.0015 is the correct
**reference scale** for "how big an effect can this class of mass-window manipulation
plausibly produce," but it is NOT automatically an upper bound on the log-k=3 geometry swap —
the physics-change doc's own §7 second caveat states this explicitly ("the 17-point true-host
loss rate is a qualitatively different quantity from a candidate-count change and could carry
its own, uncharacterized H₀ effect… the net sign is UNDETERMINED by the zero-compute read").
This registration is what resolves that.

**Verdict map (two-sided, A8):**

| band | condition | reading |
|---|---|---|
| **IMMATERIAL-CONSISTENT-WITH-HB** | \|ΔMAP\| ≤ 0.003 (2× HB's own +0.0015 bound — generous enough to absorb stencil/measurement noise while staying clearly the same order of magnitude as HB's already-exonerated support-truncation effect, not confusable with the HEAD readout's own materiality floor) | the geometry swap behaves like a small reshaping of the same already-bounded mass-window effect HB measured; no new H₀ mechanism |
| **INTERMEDIATE** | 0.003 < \|ΔMAP\| < 0.008 | between the HB scale and the registered HEAD materiality floor; REPORTED, not adjudicated as either — an explicit non-verdict, per the honesty discipline of this ledger |
| **MATERIAL** | \|ΔMAP\| ≥ T_mat = 0.008 (`MEASUREMENT_HEAD_READOUT_20260827.md:268-285`, ratified row #213) | the geometry swap is a genuine new H₀-moving mechanism, distinct in size from HB's bounded object |

**MATERIAL-UP vs. MATERIAL-DOWN.** Per the correctness-over-bias-removal ruling
([[author-values-correctness-over-bias-removal]]) and its precedent application in
`PROPOSAL_2D_TWIN_ADOPTION_20260829.md` §6.1: a MATERIAL-DOWN read (ΔMAP ≤ −0.008, moving
AWAY from truth) does not by itself refute adoption of the ratified design, but it opens a
**mandatory stage-0** on the sign before any production default flip — the true-host
retention loss (§5) is a documented, principled mechanism that could plausibly produce either
sign, so a MATERIAL-DOWN read is not dismissed as noise.

**Secondary edge, restated:** IMMATERIAL/MATERIAL at `T_mat=0.008` is carried as the
secondary bright-line (identical to every other wave-2 arm's edge, for cross-branch
comparability), nested inside the three-way primary map above.

---

## 4. Zero-compute secondaries through the production flags at h = 0.730

These close R4 falsifier item 2 of `PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md` (the
"reimplemented `b5_window_count.py` vs. production flags" divergence check) as a side effect
of the h=0.730 arm-T task, at zero additional compute:

1. **Per-event candidate growth** (arm T vs. baseline B, through the now-existing production
   `get_possible_hosts_from_ball_tree` flags, not the `b5_window_count.py` replica):
   predicted **median 0.949, p95 1.498, max 10.0** {value, source:
   `b5_window_count.json:growth_factor_iii_vs_i.{median,p95,max}`, date: 2026-08-29} — a
   **prediction** for what the production call will show; the replica-vs-production match is
   itself gate **R1** below.
2. **True-host retention on iiib** — predicted **0.789 ± 0.009** {value, source:
   `b5_window_count_arm_jackknife.json:summary_across_arms.iii_log_k3.0.retention_fraction_across_arms.{mean,SE}`
   ≈ 0.7898 ± 0.0093, date: 2026-08-29} carried forward as a **two-sided prediction on a
   different fleet**: `b5_window_count.json`/the arm-jackknife are measured on the
   `p3_2d_fleet_20260825` mirror (`bc_9001XX_work`, 2,261 events), NOT on the real iiib
   production catalogue this arm actually runs against — the transfer is an assumption, not a
   measurement, and is exactly what R1 (below) tests. Falsifier: §6 item 2.
3. **Class migration C-A/C-B → C-C.** Per L3 (docket §3): events whose true host leaves the
   window under arm T become dark-class (C-C) by construction — reported per-event class
   table, cross-tabulated against baseline B's class assignment for the same event set.
4. **R6-style 1D/2D channel split — bit-identical columns.** The mass window only enters the
   with-BH (2D) host-candidate resolution; the no-BH (1D) channel does not consult
   `BH_MASS`/`BH_MASS_ERROR` at all. Registered bit-identical columns between arms:
   `L_cat_no_bh`, `combined_no_bh` (the full 1D channel). Any divergence ⇒ **R6 gate failure**
   (§6).

---

## 5. The pull-read's consequence for the ε rationale — decision and justification

`B5_2_PULL_READ_20260829.md` closes the docket's L9/§7-second-caveat question with a specific,
negative result: **no `σ_lnM` redefinition rescues the ε=2Φ(−k) retention rationale** at this
fleet's median CV≈1.02. Two candidate definitions were tested — `σ_def1 = CV`
(`= BH_MASS_ERROR/BH_MASS`, "the presented σ," confirmed by L9 to be exactly the code's own
R&V15 ln-space budget, `handler.py:1446-1459`) and `σ_def2 = ln(1+CV)` — and neither reaches
anywhere near the Gaussian 99.7% target (78.8% / 74.9% pooled respectively); `σ_def2` is
**strictly worse** on both axes that matter (further from the Gaussian target AND further
from the independently-measured production/replica retention number). The mechanism is a
**model-shape mismatch**: the mirror's generator (`correspondence_1d.py:1736-1739`) draws the
true source-frame mass as a **linear**-Gaussian around the host's catalogue mass (truncated at
`M>0`), not a log-normal — no choice of log-space σ definition can make a symmetric log-window
bound a linear-Gaussian-truncated population the way it bounds a log-normal one.

**Decision: the presented σ (`σ_lnM = BH_MASS_ERROR/BH_MASS`, i.e. the as-coded, as-ratified
F-ii design) is registered as PRIMARY. No CORRECTED-σ arm is registered.**

**Justification:**
1. There is no "corrected" definition to promote — both candidates were tested and the
   presented one is the *better* of the two, not the worse one. Registering `σ_def2` as an
   alternative primary would mean deliberately choosing the option the pull-read already shows
   is further from both the Gaussian target and the code's own cross-validated behavior.
2. The presented σ independently cross-validates against `b5_window_count.json`'s own
   full-interval-overlap retention census to within 0.1–0.2 percentage points at every k
   tested (`B5_2_PULL_READ_20260829.md` §3, cross-check table) — two different code paths
   agreeing this tightly is strong evidence the presented σ is being read and applied
   correctly, not evidence it is the wrong quantity.
3. L9 (§1 of the pull-read) independently confirms `BH_MASS_ERROR/BH_MASS` **is** the ln-space
   σ the R&V15 error budget was built to produce, from first principles in the code
   (`handler.py:44,1446-1459`) — it is not an ad hoc ratio chosen for convenience.
4. The actual defect (linear-vs-log generative shape) is **not addressable by a σ-definition
   change at all** — it is a separate, out-of-scope physics question about the mirror's
   injection model (`correspondence_1d.py`'s `catalogue_selected_2d` draw law), not about the
   window formula under test here. Registering a second σ-definition arm would not close that
   gap and would spend compute on a comparison the pull-read has already adjudicated.

**Consequence for the ε=2Φ(−k)=0.27% rationale (row #221 F-ii's stated design justification):
RETIRED as a fleet-level retention statement, not retracted as a single-candidate quantity.**
`eps_log(k)=2Φ(−k)` remains correctly derived and G4-verified for the object it actually
measures — a fixed-σ_lnM synthetic log-normal population's own tail mass (physics-change doc
§6 item 3) — but it does not predict, and must not be cited as predicting, this fleet's
true-host retention rate at k=3. **78.9% (0.789±0.009, mirror-measured) is the number that
carries into this registration's §4 item 2 and into any adoption argument, not 99.73%.** This
correction is carried forward verbatim from the pull-read's own verdict line and is treated as
settled for this registration; it is not re-derived here.

---

## 6. Gates

- **R1 (falsifier / consistency gate, closes physics-change-doc R4 item 2):** the production
  flags (`mass_filter_geometry="log"`, `mass_filter_k=3.0`), called directly through
  `get_possible_hosts_from_ball_tree` on the **iiib production fleet** at h=0.730, must
  reproduce both (a) the per-event candidate-growth distribution and (b) the true-host
  retention fraction to within **±2 percentage points** of the mirror-fleet numbers in §4
  items 1–2, treating the mirror→production transfer as the thing under test. **Outside ±2pp:
  the mirror-to-production transfer assumption (not necessarily the window design) is
  implicated** — the ΔMAP read proceeds regardless (it is measured directly on iiib, not
  inferred from the mirror), but §4's retention PREDICTION is falsified and must be flagged
  before any adoption argument cites it.
- **R2 (A13 engagement gate):** fraction of events with a non-empty baseline-B with-BH
  candidate set whose with-BH candidate mask (or `L_cat_with_bh`) differs between arm T and
  baseline B at h=0.730 must be **≥ 0.90** (set below B7.2's 0.95 because §7's own measured
  per-event growth distribution has a wash/no-change tail the 2D-TWIN's `S_4D<1` mechanism
  does not — median growth ratio 0.949 means many events see only a small change, not a large
  one; 0.90 is chosen as comfortably above the ~5% "empty under both" + partial-wash floor
  implied by §7's `n_events_zero_under_both=24`/2261≈1.1% and the median-near-1 growth ratio).
  Below 0.90 ⇒ the switch is not reaching the production dispatch path ⇒ **STOP**.
- **R6 (instrument-defect gate):** `L_cat_no_bh`/`combined_no_bh` bit-identical between arms
  at every H4 node (production-scale test, mirroring B7.2's `test_1d_channel_unaffected_*`
  pattern). Violation ⇒ **INSTRUMENT-DEFECT** — the mass window has leaked into the 1D
  channel, which is a code defect, not an H₀ finding.
- **Validity condition on the stencil (R5):** `|Δℓ''| ≪ I_HEAD = 2965` at the three-node
  {0.660, 0.665, 0.670} local fit. Violation ⇒ the H4 read is **AMBIGUOUS**, escalating to a
  conditional G27 grid (27 nodes, 0.010 step) exactly as B7.2's precedent — not a silent pass.
  Numerically: a T_mat-scale shift (0.008) corresponds to `Δℓ' = 23.7` nats per unit h; the
  IMMATERIAL-CONSISTENT-WITH-HB edge (0.003) corresponds to `Δℓ' ≈ 8.9` nats per unit h; HB's
  own +0.0015 corresponds to `Δℓ' ≈ 4.4` nats per unit h.

Reported (not gated): **R3** ΔT — the with-BH score-at-truth tilt shift, arm T vs. baseline B,
at h=0.730 (row-#201 form); **R4** Δw̄₂ — mean with-BH catalogue mixture-weight shift (sign
not predicted in advance: §5's retention loss removes some correct hosts from the with-BH
support, which could either lower the catalogue term's weight via lost support or raise its
per-included-event confidence via a tighter window — the net is exactly what R4 measures, not
assumed).

---

## 7. A10 — invariants held fixed across both arms, and blindness sentence

**Invariants (last derivation-audit date), carried unchanged from B7.2's registered list plus
this branch's own:**
`normalization_mode=absolute_marginal` · `host_z_kernel=volume_deconv` ·
`selection_in_completion_numerator=fused` (rows #117–#118) ·
`catalogue_numerator_survival=phi` (row #197) · `catalogue_global_selection=phi`
(rows #172–#178) · **`mass_filter_sigma=symmetric`** (row #202, `cf4f8a2a` — held fixed in
BOTH arms; only `mass_filter_geometry`/`mass_filter_k` vary, per §1) ·
`completion_b_scale=derived` · `eddington_m=on` · `sigma4d_mass_kernel=point` (Instrument J,
NEVER derivation-audited as a design choice — disclosed) · `catalogue_mass_overlap=production`
· `pdet_wbh_z_resolved=False` · `H_GRID_41`/subsets (H4) · CRB md5 `9a1f2a14…` and catalogue
md5 `c52c13b5…` pins · `EVAL_SEED=777000` · **`k_sky=1.5`** (L3 invariant: A2/cone-widening
NOT triggered in wave 2, so B5.2's measured candidate-count and retention factors are
attributable to the mass geometry alone, not a confounded sky-cone change) · the B6.1 θ-hook
s-placement commit (must precede this arm's banking, per L8).

**Blindness sentence.** By construction, this design cannot detect: (a) a defect in the
shared with-BH host-matching/`S_4D`-adjacent machinery common to both arms (a common-mode bug
would cancel in the arm-vs-baseline difference, not surface as a bad ΔMAP); (b) whether a
different, shape-matched window (e.g. an asymmetric or truncated-linear-consistent window,
rather than a symmetric log window) would perform better — the design compares exactly the two
registered geometries, not the full space of possible windows; (c) anything in the 1D channel
(bit-identical by construction, R6); (d) which of the mirror generator's linear-Gaussian draw
law or the window's log-normal assumption is the "more correct" physical model for the
host-mass injection process — that is a separate, unresolved modeling question about
`correspondence_1d.py`'s generative law, orthogonal to whether the *window*, given either
generative assumption, is internally consistent; this registration measures ΔH₀ under the
ratified design, not which generative model is right.

---

## 8. A14 — falsifiers

1. **Retention-transfer falsifier (§4 item 2, §6 R1):** the mirror-fleet retention prediction
   0.789 ± 0.009 is FALSIFIED if the production iiib census (via the now-existing flags, R1)
   falls outside **[0.762, 0.816]** (±3 SE, using the arm-jackknife's own SE=0.0093 as the
   generalization uncertainty per R5/R8 of the physics-change doc) — outside that band, the
   mirror-to-production transfer itself (not the window design) needs revisiting before any
   ΔMAP number from this registration is cited as representative of the ratified design in
   general, though the ΔMAP read itself remains valid for the iiib venue as measured.
2. **Attribution falsifier:** any ΔMAP/Δmean_h attributed in this registration's readout to
   "mass-window geometry" is FALSIFIED if R6 (1D bit-identity) fails, or if R1's ±2pp
   candidate-growth/retention cross-check fails outside its band without a documented,
   independent explanation — either failure means the measured H₀ shift's cause is not
   isolated to the intended mechanism.
3. **Stencil-validity falsifier:** the H4-derived Δmean_h,pred is FALSIFIED as a usable
   first-order estimate if `|Δℓ''|` at the three-node local fit is not `≪ I_HEAD=2965`; this
   does not falsify the underlying ΔMAP/Δmean_h measurement itself (which the G27 escalation
   or the unconditional wave-3 full HEAD readout — F2 — resolves unconditionally), only the
   *H4-only* first-order shortcut.

---

## 9. A15 — operating characteristics at N = 1588 (iiib production events)

All reads are **paired deterministic recomputations** on the identical iiib event set across
both arms (same seeds, same catalogue, same CRB, only the two mass-filter flags differ) ⇒
**sampling variance is exactly zero** for the ΔMAP/Δmean_h/ΔT/Δw̄₂ point differences —
the operative uncertainty is not a significance test but a **materiality band** (§3), exactly
the A15 pattern already established for B7.2 and B3.2. False-fail rate under the
reproducibility floor: **0**, bounded by the PROD-A0 ingredient-gate floor `≤ 8.5e-15`
(row #201, historically measured). Detectable effect: any `|Δmean_h| ≥ T_mat = 0.008` is
detected with certainty on the full `H_GRID_41` posterior (delivered for free by the wave-3
shared blind HEAD readout, F2); on the sparser H4 grid alone, the limitation is the
first-order predictor's own model error, bounded by the stencil-validity condition (§8 item 3)
and made harmless by that same unconditional wave-3 full read — a stencil misclassification
changes which band the interim read reports, never the eventual adoption verdict, since F2
guarantees a full-grid re-check regardless of the H4 outcome.

The true-host-retention prediction (§4 item 2, §8 item 1) is the one quantity in this
registration that IS a genuine cross-fleet generalization (mirror → production), not a paired
recomputation on one fixed set — its uncertainty is the arm-jackknife SE (0.0093, 24 arms),
explicitly disclosed in the source document (`PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md`
R5) as a **lower bound** on true fleet-to-fleet variability (all 24 mirror arms share one
pruned catalogue and selection pipeline; a genuinely independent catalogue realization could
show more spread).

---

## 10. The 5.2 adoption rule

Per the charter (docket §2 B5, row #221 F-ii "adopt only after a registered counterfactual"),
adoption of `mass_filter_geometry="log"`, `mass_filter_k=3.0` as the new production default is
gated on **all three** of the following, evaluated together, not on ΔMAP/cost alone:

1. **H₀ delta immaterial (or argued benign if not).** ΔMAP/Δmean_h lands in
   IMMATERIAL-CONSISTENT-WITH-HB or, if MATERIAL-UP, is accepted per the
   correctness-over-bias-removal ruling (a corrective, truth-ward shift is not a reason to
   withhold adoption); a MATERIAL-DOWN read does not itself block adoption but triggers the
   mandatory stage-0 on sign (§3) before any default flip — adoption is deferred, not refused,
   pending that stage-0.
2. **Candidate growth stays inside the registered compute ceiling.** The measured/predicted
   candidate-count growth (§4 item 1: median 0.949, p95 1.498, max 10.0) must keep the
   production per-arm cost inside the charter's ~50–130 CPU-h envelope for this class of
   change (C3's own registered band, §11) — sized off the **p95**, not the mean, per B5.1's
   own explicit recommendation (`B5_1_WIN_RECORD.md` §8), since a fixed-array-size SLURM
   submission is bounded by its slowest task, not its average one.
3. **The true-host retention loss is argued as physically right, or the design returns.**
   The measured ~17–21 point true-host retention drop (95.7%→78.9% on the mirror; iiib
   production TBD, R1) is **not**, by itself, evidence against adoption — the physics-change
   doc's §5/§7 argument is that a bounded log window, by design, excludes catalogue mass
   estimates too discrepant from the GW mass to be plausible under the stated R&V15 scatter
   budget, and that this is the *intended*, principled behavior of closing linear's
   negative-lower-edge loophole (§7 first ⚠, "heavy-cut reintroduction"). **Adoption requires
   this argument to be made explicitly, in this registration's own readout, comparing the
   ΔMAP evidence against it** — not merely observed and left unaddressed. If the readout
   cannot sustain that argument (e.g., the retention loss turns out to correlate with a
   physically implausible exclusion pattern, not merely "too-discrepant" catalogue masses),
   **the design returns to depth-3** for a shape-matched redesign (e.g., an asymmetric or
   linear-consistent window informed by §5 of the pull-read) rather than being adopted on the
   ΔMAP/cost numbers alone.

All three conditions are ANDed: satisfying (1) and (2) without an explicit (3) argument is
**not** sufficient for adoption under this rule.

---

## 11. Cost (F4) and archive-scheduled field

**Cost anchor:** production iiib per-h-point 14.93–22.9 CPU-h (`cluster/LAUNCHING_JOBS.md:47`
instructed band; `MEASUREMENT_HEAD_READOUT_20260827.md` §9/§F fresher bracket), scaled by the
measured candidate factor 0.73–1.5 (§7 of the physics-change doc: aggregate ratio 0.726,
p95 per-event ratio 1.498). **4 H4 nodes × 14.93–22.9 CPU-h × 0.73–1.5 ⇒ 44–137 CPU-h**
(compute-ledger row **C3**: `results/campaign51_20260728/realistic_20260729/fanout1_20260829/COMPUTE_LEDGER.md:45`,
"log k=3 counterfactual (B5.2), H4 grid | 44–137") plus a share of the shared baseline gate
task **C0** (15–23 CPU-h, serving B3.2/B5.2/B7.2 jointly, L5) — not counted again against this
arm's own ceiling. SLURM per-task `--time` sized off the measured **p95** growth factor
(1.498×), not the mean, per B5.1's own recommendation; 16 events (the zero-to-nonzero class,
§7 of the physics-change doc) flagged for a first small smoke before the full H4 submission,
consistent with `cpu_il`, 16 cpus/task, backfill-friendly array shape used by the other wave-2
arms (docket §4.3).

**Archive-scheduled field:** workspace expires **2026-09-23**; this arm's out-root is
**MUST-ARCHIVE tier** (Option A rsync, in flight per row #224/#233). Per compute-ledger
convention (F4), this arm's "archive-scheduled?" cell in `COMPUTE_LEDGER.md` row C3 must read
**"yes"** — confirmed by the orchestrator in the launch summary — **before** `sbatch`; it does
not launch otherwise.

---

## Stamp

Launched under rows #222/#223 — charter node B5.2. Registration authored 2026-08-29, before
any C3 cluster compute. Inputs: `PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md` (§7, R1–R9),
`b5_window_count.json`, `b5_window_count_arm_jackknife.json`, `B5_1_WIN_RECORD.md`,
`B5_2_PULL_READ_20260829.md`, `CLAIM_WGEO_20260827.md` §4.1/D-WGEO-1,
`MEASUREMENT_HEAD_READOUT_20260827.md:268-285`, `SYNTHESIS_DOCKET_1_20260829.md` §2 B5/§3
(L3, L5, L8, L9)/§4.2 item 3, `PROPOSAL_2D_TWIN_ADOPTION_20260829.md` §6.2 (stencil/cost
pattern precedent), `COMPUTE_LEDGER.md` row C3, `docs/RESEARCH_CYCLE.md` A8/A10/A11/A13/A14/A15.
This document does not itself launch `sbatch`; it is the registration that must exist before
the orchestrator does.

---

## 12. Appended note (2026-08-29 — wave-2 GAP-CLOSURE archive/notes worker, launched under rows
#222/#223 — charter node: NODE archive+minor-notes, GAP 7)

Closes `WAVE2_REGISTRATION_CHECK_20260829.md` §1.4 / §5 item 7 (four minor gaps). Standing rule 1
(append-only) applies — nothing above this section is altered.

1. **H4 adjudicates on Δmean_h,pred, not ΔMAP.** This registration's H4 grid (§4 item 1, §11) is
   measurable at production scale, this wave, only as `Δmean_h,pred` (the stencil quantity, per
   the materiality map of §10/§11); `ΔMAP` is NOT delivered by this wave's H4 read — it is
   delivered separately by the wave-3 G41 read (the pattern used in
   `PROPOSAL_2D_TWIN_ADOPTION_20260829.md` §6.2 for its own H4/materiality distinction). State
   explicitly: **the wave-2 adjudication under this registration is on `Δmean_h,pred`**; any
   `ΔMAP` claim citing this document is out of scope until the wave-3 G41 read exists.
   {source: `WAVE2_REGISTRATION_CHECK_20260829.md:153`; 2026-08-29}
2. **Reconcile R1 ±2 pp vs §8 ±3 SE.** Two bands exist for retention: R1's registered ±2 pp
   ([0.769, 0.809]) and the §8 falsifier's derived ±3 SE band ([0.762, 0.816]) around the mirror
   retention 0.78903 (jackknife SE 0.0093, §9/§4 item 1). A production retention value in
   [0.762, 0.769) ∪ (0.809, 0.816] would fail the tighter R1 band but not be "falsified" under the
   §8 statistical band. **Decision (this note): the derived ±3 SE band ([0.762, 0.816]) is the
   band used for BOTH R1 and the §8 falsifier** — it is the statistically-grounded interval (a
   jackknife SE on the actually-measured mirror quantity), while R1's ±2 pp is an a priori
   round-number assertion with no derivation given in §2/§4. R1's ±2 pp wording stands as
   originally asserted in this document (append-only; not edited) but is **superseded for
   adjudication purposes** by the ±3 SE band from this note forward. {source:
   `WAVE2_REGISTRATION_CHECK_20260829.md:155`, `b5_window_count_arm_jackknife.json:summary_across_arms`
   (SD 0.04553, n=24 arms ⇒ SE 0.0093); 2026-08-29}
3. **Class definition: h = 0.730 vs all-41-node.** §4 item 3 ("class migration C-A/C-B → C-C")
   is defined, in this registration, **at h = 0.730 only** — the single zero-compute node this
   arm's registration reasons from — NOT over all 41 h-nodes (B3.1's definition,
   `B3_1_POP_RECORD.md:75-82`). The two definitions differ for any event with
   `L_cat_no_bh > 0` at some h-nodes but `== 0` at others. **Both counts, where cheap (i.e. where
   the all-41-node CSVs are already banked and require no new `evaluate()` call), should be
   reported side-by-side in the C3 readout** — this note does not itself compute either count, it
   only registers that the distinction must be surfaced rather than silently resolved to one
   definition. {source: `WAVE2_REGISTRATION_CHECK_20260829.md:163`, `B3_1_POP_RECORD.md:75-82`;
   2026-08-29}
4. **Verifier-scope line.** This registration, as amended by this note (items 1–3 above) and by
   the launch-stamp placeholder below, routes to the end-of-fan-out verifier alongside the adoption
   gate and wave-3 readout named in §10. {source: `WAVE2_REGISTRATION_CHECK_20260829.md:162`;
   2026-08-29}

**Launch-stamp placeholder (A22).** Wave-2 commit: `<hash at launch>` (does not exist yet — the
working tree was dirty at registration-check time, `WAVE2_REGISTRATION_CHECK_20260829.md` §0/§3
item 1; to be filled by whichever node performs the wave-2 commit). Baseline commit:
`d04d9dc9bfe39e6c5a72e768a26f2dcc38355bf5` (the banked HEAD readout, `run_metadata_21.json`,
2026-08-27T19:40:20).

Stamped: launched under rows #222/#223 — charter node NODE archive+minor-notes (GAP 7), 2026-08-29.
