# Campaign Readout Report — Production-Configuration Calibration Ladder

**Coverage/P–P calibration campaign · #66/#67 production-calibration front (row #120) · 26
invocations / 120 realizations each / n_events 250–1600 · read out 2026-08-18.**

**The question:** *Is the production estimator, in the exact configuration that runs in production
(no measurement scatter, const-σ at truth, fused selection), calibrated — including the mass
channel — at a deep, multi-candidate venue?*

**Verdict strip:** H-P fired **FAIL** (registered band: cov68 < 0.50) at the harness's deep venue ·
the instrument audit attributes the failure to a venue-regime property of the *asymmetric* [P2]
insertion, not to a production defect, and production's own counterfactual regime is the opposite
one · **presented, not adjudicated.**

---

## 2. The goal

**The prior finding.** The selection fusion ([P2] S̄_φ in the 1D completion numerator, [P1] g_sel
in the 2D mass integral) landed in production at `2b10b8b8` with direct counterfactual evidence
that it moves production posteriors by zero MAP steps. But *calibration* (bias vs truth, coverage)
was never measured for this configuration: the July harness result #67 said the selection-inside
half calibrates **only when paired** with the model-σ noise convention, and the Q-0 audit
established production carries the **const-σ half instead** (no scatter; σ frozen at truth). The
fusion has been carried ever since with "#66/#67 calibration" as its named disappointment path.

**The objection this campaign answers.** "The landed production configuration
(selection-inside + const-σ-at-truth) is the configuration class the July harness measured as
breaking calibrated controls — nobody has ever run a truth-injection coverage test on it, with a
mass channel, at production event counts."

**Design in one line:** vary {noise convention × selection cell × venue × n_events} on a
mass-bearing multi-candidate synthetic harness whose completion/selection legs mirror production's
forms; hold seeds fixed within each (venue, n) so every comparison is paired; all bands locked at
pre-registration (`fe72d52b`), amendments verifier-gated before execution.

## 3. The design — arms and control

| rung | cells | axes ON | headline (1D, h=0.72, n=1600 unless noted) |
|---|---|---|---|
| Block A grid | 12 | 3 noise × 2 selection × 2 venues, n=250 | production+fused V-deep: −0.034 |
| Block B n-ladder | 8 | {production, const} × {off, fused}, n∈{800,1600} | production+fused: −0.032 · production+off: −0.0017 |
| S-1 decomposition | 2 | `1d` / `2d` cells, V-deep n=250 | `1d`: −0.033 (full shift) · `2d`: −0.0011 (inert) |
| Block N1 replication | 4 | mass OFF, continuum, July venue+seed | const +0.0023 / corrected +0.0004 (July record ✓) |

**The control that licenses the page:** Block N1 — after an execution erratum was fixed (below) —
reproduces the 2026-07-11 record on every cell, and the `off` cells at V-deep are nearly
calibrated (−0.0013…−0.0017 at n=1600). The apparatus can measure a zero.

**What would have broken the effect:** any of N-1/N-2/N-5 firing without an audit-established
cause; the fused−off deltas degenerating; the shift failing the n-ladder's asymptotic signature.

## 4. The result

Per-trial MAP distributions (quartiles from the raw `maps` arrays, h_true = 0.72, n=1600):

```
truth 0.720                     q10    q25    med    q75    q90
production + off              0.7120 0.7160 0.7180 0.7220 0.7240   ← brackets truth
production + fused            0.6820 0.6840 0.6880 0.6920 0.6960   ← entire distribution below truth
const      + fused            0.6820 0.6840 0.6900 0.6940 0.6980   ← same disease
model      + fused (n=250)    0.6600 0.6720 0.6840 0.6930 0.7000   ← same disease, wider
N1 replication (corrected)    0.7120 0.7160 0.7200 0.7240 0.7280   ← textbook calibration
```

**Headline cards (decision cell = V-deep, production+fused, n=1600, 1D):**
- **Coverage:** observed cov68 = **0.000** vs locked PASS band [0.594, 0.766].
- **Effect size:** MAP bias **−0.032** in h (locked PASS |bias| ≤ 0.0010); at h_true=0.62 the
  distribution rails at the grid floor (99% rail fraction).
- **Relative to claimed uncertainty:** the shift is ≈ 7× the posterior's own σ ≈ 0.0046 —
  confidently wrong, not noisy.

**Read this before anything else — the shape of the failure.** This is the *confidently-wrong*
disease, and it is switched on by exactly one lever: the paired `off` twins are nearly calibrated
everywhere, and the S-1 cells localize the entire shift to the **[P2] 1D completion-numerator
insertion** ([P1] is inert, matching the production counterfactual's channel structure). The
audit then reproduced the shift analytically: ΔlogL(fused−off) has slope −1453 nats/h at the off
MAP, and slope·σ² = −0.0309 predicts the measured −0.030 to 3%. The driver is S̄_φ falling ~5×
across V-deep's completion window — a **z-reweighting** of the completion leg. Production's
completion volume sits in the opposite regime (S̄ nearly flat), where the same formula gives a
small *positive* tilt (+0.0155 nats/h/event measured in the counterfactual) and sub-grid-step MAP
motion. Same physics, opposite regime — the harness venue turned the lever's amplitude up 68×.

## 5. The mechanism check (H-N)

Registered signature of a real asymptotic bias (#67): floor flat in n, cov68 collapsing.

| n | production+fused bias (h=0.72) | cov68 |
|---|---|---|
| 250 | −0.0340 ± 0.0014 | 0.050 |
| 800 | −0.0305 ± 0.0007 | 0.000 |
| 1600 | −0.0320 ± 0.0005 | 0.000 |

The signature **fires**: the shift is N-coherent, consistent with the audit's first-order-tilt
mechanism (slope ∝ n, σ² ∝ 1/n ⇒ shift constant in n). Within the harness venue this is a real
asymptotic bias of the asymmetric configuration, not a small-n fluctuation.

**The audit's decisive side-experiment (exploratory, outside the registered grid):** inserting
S̄_φ **symmetrically** (both numerator legs) restores near-calibration (+0.008), and
catalogue-leg-only breaks it the other way (+0.041). The displacement is owned by the
**asymmetry** of the insertion, not by the insertion itself.

## 6. The scorecard

Registered reads vs locked bands (bands fixed at `fe72d52b`, unchanged after readout):

| read | locked band (PASS) | observed | outcome |
|---|---|---|---|
| **H-P** production+fused V-deep n=1600 | \|bias\| ≤ 0.0010 AND cov68 ∈ [0.594,0.766] | bias −0.032, cov68 0.000 | **FAIL** (cov < 0.50 leg) |
| H-B (const−production)+fused paired, n=1600 | precondition: const+fused floor ≥ +0.0015 sign-coherent | const+fused = −0.030 (precondition unmet) | **UNDETERMINED-BY-DESIGN** (unscored, as registered) |
| H-N n-ladder | bias→0 with nominal cov at all n | flat −0.030, cov collapsing | **FAIL branch** = asymptotic-bias signature fires |
| N-1 replication vs July | within 2·SE | after erratum fix: all cells within band | **PASS** |
| N-2 model+fused V-deep | \|bias\| ≤ 0.0010 | −0.0379 | **FAIL** → registered STOP fired; audit resolved (§8) |
| N-3 V-ctrl paired vs off | delta ≈ 0, no sign flip | delta ≡ 0 (bit-degenerate) | formal PASS, **VOID** (§9 flag 1) |
| N-4 2D coverage fused vs off | no degradation > 2·SE | fused−off 2D delta −0.0006 | **PASS** (but see §9 flag 3) |
| N-5 engagement | deltas non-degenerate; V-ctrl #66-direction | V-deep: all non-degenerate ✓ · V-ctrl: degenerate | **FAIL at V-ctrl** → STOP fired; audit resolved as structural void |

Footnote: full per-cell table in `readout_prodcal_output.json` (scorer `--registered`, 18-pair
manifest, no missing pairs). The off cells at n=1600 sit at −0.0013…−0.0017, marginally outside
the ±0.0010 H-P bias leg — a band-anchoring caveat (§9 flag 4), not a fired read.

## 7. Vocabulary

- **production noise cell** — no measurement scatter, σ frozen at the truth's Fisher value: the
  configuration production actually runs (Q-0). Decision-cell value: the −0.032 row above.
- **fused / off** — production's selection factor inserted into the completion numerator(s) /
  absent (pre-fusion form). The off twin: −0.0017.
- **S̄_φ(z;h)** — the mass-marginal detection survival; the object the fusion inserts. Falls
  0.31→0 across V-deep's completion window (the amplitude of the lever); nearly flat in
  production's regime.
- **cov68** — fraction of 120 universes whose true h lies in the 68% HPD. Decision cell: 0.000.
- **rail fraction** — realizations whose MAP sits on the grid edge. 0.99 at h_true=0.62 in the
  decision configuration; 0 in the calibrated controls (its absence there is information).
- **first-order tilt** — MAP shift ≈ (dΔlogL/dh)·σ²; the mechanism that predicts −0.031 vs
  measured −0.030.
- **asymmetric insertion** — [P2] applied to the completion leg only (production's landed form);
  the audit's variant table shows symmetric insertion calibrates (+0.008) where asymmetric
  fails (−0.029).

## 8. Why the numbers stand

**Validity.** All 26 registered invocations completed (cluster job 6355028, 14.1 CPU-h of the
18 CPU-h ceiling; DEVIATION-1 environment). Seeds per the registered paired scheme; scorer run in
its `--registered` invocation of record; 18/18 manifest pairs present; N-5 non-degeneracy holds
everywhere the completion leg exists. Block N1, after the erratum fix, reproduces the July record
on the July seed — instrument continuity across six weeks of harness development.

**Independent recompute / audit.** The registered STOPs (N-2, N-5, N-1-first-pass) triggered a
read-only instrument audit that: traced the fused insertion against production's code
(same asymmetry, same objects — the harness mirrors production's form [LOCAL]); reproduced the
−0.030 analytically (slope·σ², 3% agreement); verified the α_M=0 identity to 5.6e-16, quadrature
convergence to ≤0.0002, and no interpolation clamping; and corrected one misattribution (§9
flag 3). Verdict: **instrument faithful; no code defect found.** The N-1 first-pass failure was
independently traced to a driver flag omission (mixture_mode) whose magnitude matched the July
two_branch record *exactly* (+0.0235), and vanished on the faithful rerun.

## 9. What the adjudicator flagged anyway

1. **[compliance, ratification bundle] V-ctrl was structurally void.** The D-1 amendment set
   z_support = 1.5 > Z_MAX_POP = 0.95, emptying the completion window for every event
   (completion_fraction ≡ 0.0000): the fused lever had nothing to act on. N-3's formal PASS and
   N-5's V-ctrl FAIL both carry **zero information**. Does not change the fired branch (H-P fired
   at V-deep). Changes: the control's role must be re-registered (decision 3).
2. **[compliance, ratification bundle] Block N1 execution erratum.** The driver omitted
   `mixture_mode="exact"`; first-pass N1 cells ran two_branch and were quarantined to
   `cells_unfaithful_n1/`; the faithful rerun passes. Does not change the branch; enters the
   record as an execution deviation with its diagnosis.
3. **[correction] The 2D channel's +0.01 bias is NOT a fusion effect** — it is present
   identically in the `off` cells and is venue noise physics (photo-z error × completion share
   ~+0.006, galaxy mass-observation error ~+0.005; both →0 as σ→0). Does not change the branch.
   Changes: it opens one instrument-design question — the 2D catalogue-leg overlap carries no
   φ-prior weight where the completion leg integrates against φ; a cheap φ-slope cell decides
   whether that is an instrument defect or Malmquist physics (decision 4).
4. **[interpretive] H-P's bias leg (±0.0010) is marginally failed by the calibrated `off` null
   itself** (−0.0013…−0.0017 at n=1600) — the band was anchored on the mass-free July record and
   transfers imperfectly to the mass-bearing venue. Affects band design for any re-registration,
   not this branch (H-P fired on the coverage leg, which is unambiguous).
5. **[interpretive] N-2's band anchored a different venue.** The July "model-σ + inside pairs
   calibrate" result was mass-free and shallow-window; at V-deep the fused shift (−0.03)
   dominates every noise convention, so N-2 as designed could never pass there. The STOP it fired
   was resolved by audit, not by band satisfaction.

## 10. The decisions (author-gated; per the binding default, each returns on its own)

| # | tag | decision |
|---|---|---|
| 1 | **[RULE]** | **Ratify the readout bundle:** the registered H-P/H-N FAIL at V-deep with the audit's mechanism attribution (venue-regime property of the asymmetric [P2] insertion; instrument faithful); the two compliance deviations (§9 flags 1–2); the 2D misattribution correction (flag 3). Consequence: the campaign is banked as a MEASUREMENT of where the asymmetric insertion is unsafe — not as evidence of production bias. |
| 2 | **[RULE]** | **Production-calibration status:** with V-ctrl void, the production-*regime* calibration question (flat-S̄ completion window) remains OPEN — this campaign answered the deep-regime question only. Options: (a) hold status OPEN pending decision 3's cell; (b) accept the counterfactual + audit reconciliation as sufficient and close stage-4 leg 1 for the production regime now. |
| 3 | **[DO]** | **AMENDMENT-2: re-register the control as a flat-S̄ completion venue** (non-empty completion window, S̄ ≳ 0.9 across it — the audit's specified production-analog), ~1 CPU-h, verifier one-item pre-check per the standing discipline. Audit prediction: small positive tilt, ≈0 MAP motion. This is the designated one-more-measurement for H-P's FAIL branch and the cheapest closure of decision 2(a). |
| 4 | **[DO]** | **φ-slope decider cell** for the 2D catalogue-leg overlap question (§9 flag 3): bias2d tracks φ-slope ⇒ instrument defect (fix + retest); doesn't ⇒ Malmquist-type venue physics (documented). ~minutes. |
| 5 | **[RULE]** | **The symmetric-insertion finding enters the record as a new claim** (stage-0 intake for a future front): production's [P2] is a one-leg insertion of a two-leg mixture; the harness shows the asymmetry — not the insertion — breaks calibration where S̄ varies. **No production change is proposed**; in production's flat-S̄ regime the counterfactual measured the asymmetry's effect as sub-grid-step. The claim's Refute-by: the decision-3 cell failing to calibrate under the symmetric variant. |

## 11. Provenance footer

Cluster job 6355028 (`cpu` partition, 1 node, 26 workers, wall 3:42:26, 14.1 CPU-h) at repo
commit `39e016d2` · pre-registration `fe72d52b` (+ AMENDMENT-1 `dada62f3`, §7 fill-in `7d0b3f5f`,
DEVIATION-1 `39e016d2`) · instrument = same freezing chain · N1 faithful rerun local
(`n1_rerun.log`, driver fix in `run_ladder.py`) · scorer output `readout_prodcal_output.json` ·
audit: read-only, this session, agent-derived with orchestrator spot-verification of the
band-relevant numbers. **Branch presented, not adjudicated; bands locked at pre-registration and
unchanged after readout.**
