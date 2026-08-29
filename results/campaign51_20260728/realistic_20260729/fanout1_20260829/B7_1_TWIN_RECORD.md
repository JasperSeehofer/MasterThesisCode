# B7.1 RECORD — the with-BH catalogue-leg twin (`catalogue_numerator_survival_2d="mz_sel"`, centre `eff`)

**Launched under rows #222/#223 — charter node B7.1.** `[FABLE-B7.1 2026-08-29]`

**Subject document:** `results/campaign51_20260728/realistic_20260729/fanout1_20260829/PROPOSAL_2D_TWIN_ADOPTION_20260829.md`
(568 lines; status PROPOSAL, no code changed by it; branch `fix/p32d-classg-venue-repair` @ `a794404c`).

**Panel state:** clean after 0 rounds. Two independent reports (builder/smoke-test report +
verifier report), both `"refuted": false`, `"severity": "minor"`, `"must_fix": []`.

---

## 1. Decided centering + argument

**Decision (proposal §2, unaltered by this record):** `catalogue_numerator_survival_2d_center`
resolves to **`eff`** under `"auto"` — the numerator's own centre (`μ_gal,eff = M_eff,g(1+z)/M_z,det`,
the same mass the `mz_g` overlap prefactor already uses), not `raw` (`M_g`, uncorrected).

**Argument, as verified:**
1. `σ_gal → 0` limit: both centres coincide with the point query `Σ^4D` already makes
   (`bayesian_statistics.py:2965-2983`).
2. `σ_cond → 0` limit (the actual production operating point: `σ_cond` p50 = 8.8e-8,
   `bayesian_statistics.py:2314-2317`, dated 2026-08-17, vs `σ_gal,frac` O(0.3–3)): both centres
   converge to `S_4D(d_L, μ_cond M_z,det)`. Centering is numerically inert to ~1e-14 in `x` at
   production precision — a definitional choice, not a discriminable one, and disclosed as such
   in the A10 blindness sentence (item (e)).
3. Structural symmetry: `eff` makes the catalogue leg "the numerator's own `(z,x)` integrand times
   `S_4D`" — the same construction as the fused completion leg (`g_sel,prod`) and the 1D twin's
   own-integrand-times-`S̄_φ` form. `raw` has no 1D analogue.
4. Reproduces banked evidence: the A20 review (F2) ruled `eff` for the Gaussian branch before
   execution; the 33-seed repair fleet that produced the CONFIRMED result ran with centre `eff`
   (`p3_2d_fleet.py:27`, `p3_2d_companion.py:46`).
5. `raw` would break the eventwise bound `W̃₂ = L^twin/L^coded ≤ 1` the identity's sign proof
   consumes (hybrid-centre `E[S]` is still ≤1 in magnitude, but the object is no longer "the coded
   integrand with `S` inserted").

Both panel reports independently confirmed the code-line correspondence for this section
(`:6782-6786`, `:7471-7475` for centre wiring; `:6763`/`:7457` for the always-`eff` prefactor) and
raised no objection to the centering choice itself. The verifier report additionally confirmed the
A11 provenance spot-check on the decisive `σ_cond` = 8.8e-8 number, matched against the committed
blob at the cited line.

## 2. The core formula change (item under decision, restated for the record)

`mz_sel,g(z;h) = mz_g(z;h) · E_{x~N(μ*,σ*²)}[S_4D(d_L(z;h), x·M_z,det,i)]`, survival entering the
innermost `(z,x)` quadrature rather than only the completion (`B_num,wbh`) leg. Both reports
verified this against the committed code (`a794404c`) symbol-for-symbol: the product-Gaussian
completing-the-square identity, the Gauss–Hermite substitution in `_mz_sel_2d_expectation`
(order 24, `:444`), and the `α_G_φ`/`D̃_φ` combination formulas at `:2484-2489`/`:5729`. The
`"auto"` resolution value does not exist in code today (only `"off"`/`"mz_sel"` are accepted,
`ValueError` otherwise, `:3666-3669`/`:6274-6277`/`:7257-7260`) — the proposal is explicit that
`"auto"` is a future gate item (§8 G-3), not a claim about present code state. This distinction
was checked and confirmed correct by the builder-report reviewer.

Not independently re-derived by either report: the full S-homogeneity degree bookkeeping in §1.5
tracing `β_Ḡ_φ`'s own dependence on `S_4D` through code not inspected — flagged by the builder
report as an open item, but one the proposal itself treats as a falsifiable, zero-compute
regression test (R3, homogeneity under `S_4D → c·S_4D`) rather than an asserted fact, so an error
there is caught pre-adoption rather than smuggled in.

## 3. Arm design (wave-2 counterfactual, charter node B7.2) + CPU-h

**Venue:** iiib only (true reduced catalogue, md5 `c52c13b5…`, 1588 events, `EVAL_SEED=777000`).
**Grid — recommended H4:** `{0.660, 0.665, 0.670, 0.730}` (0.730 = production read node;
`{0.660,0.665,0.670}` bracket the HEAD 2D MAP 0.665). G27 (0.010-step, 27 nodes) is the
conditional-escalation grid if H4 lands AMBIGUOUS; G41 (full `H_GRID_41`) is explicitly NOT
proposed for B7.2 (deferred to the wave-3 shared blind HEAD readout).

**Gates/reads:** R1 (zero-free-parameter eventwise inequality gate, `ln L^T ≤ ln L^B`, instrument
defect on violation), R2 (A13 engagement fraction ≥0.95), R6 (1D channel bit-identical between
arms — instrument-defect gate), R3–R5 reported with bands. Verdict map two-sided: T_mat = 0.008,
IMMATERIAL ≤ 0.004, AMBIGUOUS 0.004–0.008 (escalates to G27), MATERIAL beyond ±0.008.

**Cost, as instructed (A11-flagged, band not point):** H4 recommended design = **74.7–101.4
CPU-h** (twin arm 59.7–81.1 + one baseline gate task), ceiling with assumed 1.3× overhead ≈ 132
CPU-h. G27 escalation, if triggered: 418.0–567.6 CPU-h (≈738 at ceiling). The charter's
"~50–130 CPU-h" B7.2 envelope holds for H4 at nominal cost but is exceeded (4–8×) by any full-grid
design at the instructed anchor. Both panel reports treated this cost accounting as consistent and
raised no objection.

## 4. Falsifier (A14, restated)

Registered falsifier of the attribution ("the missing per-candidate with-BH survival is a
structural omission whose closure is `mz_sel`/`eff`"):

- **(i) Homogeneity, zero-compute.** Under `S_4D → c·S_4D`, twin `combined_wbh` invariant to
  ≤1e-10 relative; coded is not. Violation ⇒ §1.5 degree bookkeeping wrong ⇒ proposal RETURNS.
- **(ii) Identity residual attribution (S̄_φ double-weight, rung 1).** Registered v2.9 conditional
  prediction LHS2(bt) = 0.00740040 ± 0.00024951 (±3σ two-sided) AND G4 arm-coherence ratio inside
  [0.8613, 0.8675]. Inside both ⇒ ladder complete to rung 3, residual stays venue-side
  (provisional). LHS2 outside ⇒ calibration status drops `supported → derivation-only`. G4
  outside ⇒ adoption RETURNS to gate as REFUTED-AS-CALIBRATED.
- **(iii) Production behaviour (wave-2 arm).** R1/R6 are instrument-defect falsifiers; the H₀
  read is two-sided (MATERIAL-UP/DOWN/IMMATERIAL); per the correctness-over-bias-removal ruling,
  MATERIAL-DOWN does not itself refute adoption but opens a mandatory stage-0 review before flip.

## 5. Open caveats (carried forward, not resolved by this record)

- The 2D bounded identity (C₂\*) has **not closed** for 2D — the calibration rung the 1D adoption
  had is absent here. Epistemic status of the twin's calibration is `supported`, capped, pending
  falsifier (ii).
- The residual ladder implies a ×2.25–2.35 discrepancy (RHS₂/LHS₂) still unattributed between
  venue and estimator; common-mode across arms (X_bt/X_bc = 0.834) but not closed.
- Blindness sentence (A10, §7 of proposal) stands unaltered: this design cannot detect (a) a
  defect in the shared `S_4D` table itself, (b) the `Σ^4D` point-vs-kernel mismatch (Instrument
  J's axis), (c) anything in the 1D channel, (d) which side (venue/estimator) carries the residual,
  (e) the centering choice (numerically inert, not discriminable at production precision).
- `"auto"` resolution value is proposed, not yet implemented — a future gate item (§8 G-3), and
  any code change implementing it is itself subject to `/physics-change` (this file is `bayesian_statistics.py`,
  a physics-trigger file; production changes are covered by row #223 but the gate presentation
  must still be authored before code, per CLAUDE.md).
- Neither panel report re-derived the full S-homogeneity degree bookkeeping in §1.5 independently;
  this is deferred to falsifier (i), a zero-compute regression test, rather than asserted.
- H4 grid's Δmean_h,pred is a first-order stencil estimate subject to a registered validity
  condition (|Δℓ''| ≪ I_HEAD); violation triggers escalation to G27, not a silent pass.

## 6. Provenance table (A11: {value, source, date})

| value | source file:line | date |
|---|---|---|
| σ_cond p50 = 8.8e-8 | `bayesian_statistics.py:2314-2317` | 2026-08-17 |
| GH order 24 (`_MASS_TRUNC_GH_ORDER`) | `bayesian_statistics.py:444` | quoted in proposal §1.3, 2026-08-29 |
| T_mat = 0.008 | `MEASUREMENT_HEAD_READOUT_20260827.md:268-285`, ratified row #213 | 2026-08-27/28 |
| H4 cost band 74.7–101.4 CPU-h | proposal §6.2 table, derived from `MEASUREMENT_HEAD_READOUT` §9 and §F | 2026-08-29 |
| row #223 authorization text | `BIAS_HISTORY_LEDGER.md:3020` (append-only diff, 2 insertions) | 2026-08-29 (verifier-confirmed) |
| exoneration items 6, 17 | `BIAS_HISTORY_LEDGER.md:127-155` | verifier-quoted 2026-08-29 |

## 7. Panel state and stamp

Panel: clean after 0 rounds, both reports non-refuting, minor severity, no must-fix items. No
disagreement between builder-report and verifier-report on the centering decision, the core
formula, the falsifier design, or the cost accounting. The one substantive gap either report
flagged (full §1.5 degree-bookkeeping re-derivation) is explicitly deferred to a zero-compute
regression test already in the proposal's own plan, not a defect requiring return.

**Stamp:** launched under rows #222/#223 — charter node B7.1.
