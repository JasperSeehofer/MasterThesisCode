# PRE-REGISTRATION — [P3-WBHZERO] measure-first counterfactual: the mass-filter σ-window (asymmetric vs symmetric)

**Date:** 2026-08-25 · **Thread:** `[P3-WBHZERO]` (author grant row #198: the measure-first fix
chain — flag → mirror-venue measurement → production counterfactual read → 6-item package;
adoption returns to the author as a fresh [RULE]). Stage 0 = `CLAIM_P3_WBHZERO_20260825.md`
(Gate-B verified DEFECT candidate-confirmed, row #196; every stage-0 number [AGENT]-tagged
until the pre-execution review re-derives the decisive ones). **Orchestrator-autonomous;
[ORCH-*] tags bind. Append-only after commit; A21 governs.** The C-A governance stack
(PA-CA-1/8/10/11 conventions, out-root guard, A22 resolved-flag stamps) is INHERITED
wholesale; only WBHZERO-specific items are registered here.

## 1. Hypothesis and what is being measured

**Mechanism (Gate-B verified):** `handler.py` `get_possible_hosts_from_ball_tree` applies
`sigma_multiplier` (1.5) to the GW mass uncertainty but not to the galaxy's own
`BH_MASS_ERROR` — an asymmetric ±1.5σ-vs-±1σ eligibility window that empties non-empty
z-passed candidate balls (iiib: 688/1588 = 43.3% of h=0.73 production rows, 688/688 exact
attribution; a symmetric ±1.5σ window retains ≥1 candidate in 689/689). Σ^4D/`B_num_wbh`
carry NO matching cut — an unmodeled one-sided numerator selection for the with-BH channel.

**Measurement targets (the row-#196 unmeasured quantities):** the MAGNITUDE and direction of
the readout-level effect of `mass_filter_sigma = "symmetric"` vs `"asymmetric"` — (i) the
structural retention (per-event with-BH candidate availability), (ii) the with-BH mixture
weight, (iii) the venue's registered per-seed score statistic — plus the production
counterfactual read at stage 2. **Directional prediction (registered, from stage 0):**
symmetric moves weight toward the with-BH catalogue channel (away from completion/no-BH).
Magnitude carries NO prediction — that is the measurement.

**This measurement quantifies exclusion relief only.** The numerator-model mismatch (no
matching cut in Σ^4D/`B_num_wbh`) is present in BOTH arms and is NOT resolved by either flag
value — disclosed in §5 and owned by the eventual 6-item package, not by this prereg.

## 2. Arms, venue, and instruments

| arm | what | runs |
|---|---|---|
| **WZ-A (asymmetric)** | the b0i 1D venue fleet at the production default `mass_filter_sigma="asymmetric"` | 12 tasks |
| **WZ-S (symmetric)** | the SAME fleet, same seeds/draws, `mass_filter_sigma="symmetric"` | same 12 tasks (both arms per task, one draw each — the pairing rule) |
| **WZ-P (structural control)** | the zero-compute Gate-B counterfactual recount re-run from the PRESERVED scripts (`gate_b_20260730/wbhzero_gate_b_scripts/counterfactual_symmetric.py`, `prod_reconstruct.py`) as the per-event exact predicted comparand for WZ-S retention | zero-compute |

**Venue:** the b0i 1D mirror venue (`p3_b0_identity_test.py` driver family →
`run_mirror_seed_inprocess`), seeds **900101–900112**, single-h read `h_values=(0.73,)`,
`h_bounds=(0.50, 0.86)` — the banked-fleet configuration, so WZ-A doubles as a
venue-drift/bit-identity control against the banked bc artifacts (§3 GATE BIT-A).

**Instruments (committed before running):** (i) `mass_filter_sigma ∈
{"asymmetric","symmetric"}` — ALREADY LANDED (commit `9c948ea0`; byte-identical default,
single read/validate site in `handler.py`, threaded `evaluate()` → `p_D` call site; 4
regression tests). (ii) `run_mirror_seed_inprocess` gains the pass-through parameter
`mass_filter_sigma: str = "asymmetric"` threaded to its `evaluate()` call — same inert-plumbing
pattern as the four existing flags; the known `run_arm_seed` flag-drop gap is bypassed by the
driver calling `run_mirror_seed_inprocess` directly (the committed-driver precedent, PA-2D-2).
(iii) the driver wrapper for the paired two-arm run, banking per-event
`n_cand_nomass`/`n_pass_mass_filter` per arm alongside the venue's standard artifacts.

## 3. Gates

The C-A stack verbatim (out-root guard; A22 resolved-flag stamps — the five resolved flags of
PA-2D-1 F7 PLUS `mass_filter_sigma` stamped per arm) **plus**:

- **GATE BIT-A:** the WZ-A arm's banked-comparable artifacts must be bit-identical (or
  documented-noise-identical, the 9.1e-15 CSV round-trip class) to the banked bc fleet for the
  same seeds — the venue-scale regression that the flag default changed nothing. Failure ⇒
  VOID (instrument defect, not a finding).
- **GATE CF-X:** WZ-S per-event structural retention must match the WZ-P zero-compute
  prediction EXACTLY per event (candidate sets are deterministic given event + pinned
  catalogue). Any mismatch ⇒ VOID + forensic (either the flag or the Gate-B reconstruction is
  wrong — both cannot stand).
- **Catalogue pin:** the pinned reduced-catalogue checksum verified per task (the 2026-08-20
  dataset-pinning rule); STOP on mismatch.

## 4. Statistics, bands, and verdict map

**Registered statistic:** the paired per-seed difference **Δ_s = readout(WZ-S) −
readout(WZ-A)** on the venue's registered per-seed score statistic, with the with-BH mixture
weight w̄ and the zero-with-BH-live-no-BH rate reported alongside; pooled Δ̄ ± SEM over the 12
pairs. **Band:** frozen at the σ freeze from the realized paired SEM (anchor: 3·SEM_Δ; the
pre-execution review ratifies or replaces the anchor; may only tighten post-data).

**Verdict map:** **EXCLUSION-MATERIAL** (|Δ̄| > band, direction as predicted) /
**EXCLUSION-MATERIAL-CONTRA** (|Δ̄| > band, direction OPPOSITE — returns to the author with
frozen numbers, no interpretation banked) / **EXCLUSION-IMMATERIAL** (|Δ̄| ≤ band with the
POWER clause: if the band exceeds a review-fixed materiality scale, the verdict is
UNDERPOWERED, not immaterial) / **VENUE-MISSPEC / CONTROL-FAIL** per the C-A map. Verdicts
[ORCH-banked, provisional] pending the author's stage-5 ruling; **no adoption decision is
implied by any verdict** (row #198 binding-default).

**Stage 2 — the production counterfactual read (registered, costing deferred):** after the
mirror verdict banks, a single-h (h=0.73) production `evaluate()` counterfactual read under
`mass_filter_sigma="symmetric"` against the banked iiib configuration, plus the joint_r1
question IF the cluster-side r1 observed-catalogue artifact is retrieved (else disclosed as
open). Its costing line is A21-fixed AFTER the mirror measurement (the mirror Δ scale decides
whether a multi-h production read is warranted) — not authorized by this prereg alone.

## 5. A10 — invariants and blindness

**Invariants:** the adopted production physics (rows #197 twin, #178, Σ^φ slots) at defaults
in BOTH arms · h = 0.73 read, h_bounds (0.50, 0.86) · the pinned catalogue · the C-A LHS/RHS
machinery as committed. **Blindness:** (i) the numerator-model mismatch is common to both
arms (this measurement cannot see it — §1); (ii) the redshift filter's shared asymmetric
convention is OUT OF SCOPE (row #198 grant covers the mass filter only) — any coupling is
disclosed, not measured; (iii) h-dependence is measured only at 0.73 in the mirror (stage 2
owns the h-question); (iv) venue-conditional as ever (the b0i venue's known idealizations
carry); (v) the symmetric arm changes candidate-set SIZES, so per-event runtime and any
size-coupled numerics (ball-tree ordering, per-candidate loops) differ by construction —
disclosed, gate-mitigated by CF-X exactness.

## 6. Falsifiers (A19)

EXCLUSION-MATERIAL falsified by: GATE CF-X failure on re-audit; the WZ-P prediction and the
realized WZ-S retention diverging on ANY event; the paired Δ sign flipping under seed
partition (6+6 split-half check, report-only unless it crosses zero at the band scale);
GATE BIT-A failure (voids the arm, not the claim). The stage-0 production percentages
(43.3%/30.7%) are NOT comparands for the mirror venue — venue rates are expected to differ
(the b0i fleet's banked rate was 5.0%); treating them as targets is registered as a
misreading.

## 7. Costing (A6/A17; cluster-first per row #185) — [ORCH-COST]

12 tasks × 2 arms, single-h `evaluate()` per arm. The banked b0i per-arm precedent (~30 min
class) puts the total at ~12 CPU-h ⇒ **cluster job array** (12 tasks, both arms per task;
queue-wait banked per row #185; preflight `VERDICT: READY ✓` required). Local fallback only
on chronic fair-share blockade per the row-#185 reversion test, disclosed. WZ-P: zero-compute
local. Stage 2: costing deferred to its own A21 line (§4). No other `evaluate()` calls.

*(Committed before instruments (ii)/(iii) exist; the pre-execution adversarial review —
including re-derivation of the decisive stage-0 numbers (688/688, 689/689, 5/7, 127/129), the
band anchor, the BIT-A comparability set, and the materiality scale for the POWER clause —
precedes any instrument run; A20 review before banking.)*
