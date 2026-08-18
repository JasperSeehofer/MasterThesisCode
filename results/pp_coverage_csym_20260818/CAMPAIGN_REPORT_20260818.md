# Campaign Readout Report — G-1/G-2 Grounding Measurements ([C-SYM]/[P3] Front)

**Coverage/P–P harness campaign · [C-SYM]/[P3] correct-form + Gray-paper front (rows
#121/#122/#124) · 22 scored cells + 2 N-A cells / 120 realizations each / n_events 250 · read
out 2026-08-18 · AUTONOMOUS-CYCLE SESSION (author directive of 2026-08-18, quoted in both
preregs; every branch call below is presented, not adjudicated).**

**The questions:** *(G-1)* Does adding the survival factor to the catalogue leg — the [P3]
fork, [C-SYM]'s correct form — change anything in production's regime, and does the symmetric
form calibrate where the asymmetric form is confidently wrong? *(G-2)* Does the
selection-insertion cost actually vanish as σ_z → 0, as the paper's caveat wording assumed?

**Verdict strip:** H-G1 **PASS** (catalogue-leg fusion immaterial in the production-analog
regime, ≤ +0.0002 in h — the ~170 CPU-h production counterfactual is NOT warranted) · H-CAT
**PASS** (+0.0428, the audit's ad-hoc mechanism trio now registered) · H-SYM **MIXED →
resolved to interpretation A by SEP-Z** (the symmetric form's residual is photo-z venue
physics; it is calibrated-in-bias at spec-z) · H-Zcat **PASS with a material caveat**
(catalogue-leg cost collapses ~11× but retains +0.0038 at σ_z = 0.002, above the 0.0018
yardstick) · H-Zcomp **FAIL branch fires** (the completion-leg cost collapses ~70× — the
registered persistence prediction is refuted; the σ_z-conditionality generalizes) · all
validity nulls PASS · **presented, not adjudicated.**

---

## 1. The goal

Row #121 directed: collect the paper-grounding measurements once the pipeline settled. The
prodcal ladder (G-4) closed rows #120–#124; G-3 (Gray A.10 source check) was already banked.
This campaign is G-1 + G-2 — the two measurements that decide how the paper presents the
retained catalogue-leg convention ([P3], decision item 2) and the σ_z → 0 wording (decision
item 3) — run at harness scale first, explicitly to decide whether any production-scale run
is warranted.

The [C-SYM] claim (row #122 item 5) also gets its first registered test: the prodcal audit's
exploratory trio (symmetric +0.008 / completion-only −0.029 / catalogue-only +0.041) was
produced by an uncommitted ad-hoc patch — recon showed those numbers were not reproducible
from any committed artifact. This campaign built the `cat1d`/`symmetric` instrument knobs
(+13 structural-null tests) and re-measured everything under registration.

## 2. The design — and what the anti-void mandate caught

Two pre-registrations, verifier-gated across **six pre-check parts** (Parts I–VI, 4+1+2
BLOCKING + 11 NON-BLOCKING amendments, every one applied verbatim before the relevant run).
The author's no-void-arms mandate was load-bearing three times, each catch BEFORE any scored
realization:

1. **Verifier Part II:** my draft H-SYM band was unsatisfiable on the 2D channel even by the
   calibrated reference (the V-deep 2D venue bias) — the exact V-ctrl class; all reads scoped
   to the 1D channel.
2. **Registered preflight:** the cat1d probe railed 100% at h_true = 0.84 (the +0.04-class
   displacement pushed the top truth off the [0.60, 0.86] grid) — fixed by the wide grid
   [0.56, 0.92]; the rail relaxed to 0.25 and a rail-validity gate (>0.10 ⇒
   UNDETERMINED-BY-RAIL, precedence registered) entered both preregs.
3. **N-A false alarm resolved:** a suspected cross-machine bit-identity break was diagnosed
   as an R=4-vs-R=120 comparison artifact (child seeds are drawn per truth×realization from
   one master stream); at full R the harness is **bit-exact cluster-vs-local** (maxabsdiff
   0.0, both venues) — a strong instrument-determinism result banked in passing.

**Venues (frozen from prodcal, no tuning discretion):** V-deep (strong-S̄-gradient amplitude
venue), V-prod (flat-S̄, production-like completion 0.384), V-flat (flat-S̄, completion
0.855; paired reads only — absolute legs venue-confounded per VERDICT-2/3). Seeds
deliberately reuse the prodcal masters (paired twins + N-A) plus fresh 20280311/20280399 for
G-2. All paired reads share generative streams; every cross-cell comparison reported as the
per-realization paired distribution ([A2]).

## 3. The results

### G-1 — the catalogue-leg / symmetric-form measurements (1D channel)

| read | cells | measured (h = 0.62 / 0.72 / 0.84) | band outcome |
|---|---|---|---|
| **H-G1** symmetric − fused @ V-prod | paired, seed 20271218 | +0.00023 ± 0.00009 / +0.00003 / +0.00000 | **PASS** (≤ 0.0010 everywhere) |
| **H-CAT** cat1d − off @ V-deep | paired, seed 20270818 | +0.0435 / **+0.0428 ± 0.0011** / +0.0409 | **PASS** (mechanism trio confirmed) |
| P1 symmetric − fused @ V-deep | paired | +0.0380 / +0.0413 / +0.0397 | (the asymmetry's direct cost, registered) |
| **H-SYM** symmetric absolute @ V-deep | bias / cov68 | +0.0058/+0.0073/+0.0080 · cov68 0.475/0.617/0.575 | **MIXED** (bias leg PASS, cov leg misses at one truth by 0.6σ) |
| **SEP-Z** symmetric − off @ σ_z = 0.002 | paired, seed 20280311 | +0.0033 / **+0.0029 ± 0.0002** / +0.0025 | **interpretation A fires** (≤ ½·P2 edge) |
| V-flat symmetric − fused | paired, seed 20271118 | **0 bit-exact** (n_nonzero = 0, all truths) | PASS(inert at grid resolution) |

**Read this first — the shape of the answer.** The landed production form and the correct
[C-SYM] form are **measurably indistinguishable in production's regime** (≤ +0.0002 at
V-prod; bit-identical at V-flat). At the amplitude venue the story is fully coherent: the
asymmetric insertion is confidently wrong (−0.034), the catalogue-only insertion displaces
+0.043 the other way, the symmetric form cancels to +0.007, and SEP-Z shows that residual is
photo-z venue physics — at spec-z precision the symmetric estimator's bias is
−0.0003…−0.0016, indistinguishable-or-better than its calibrated off twin. **The asymmetry
owns the displacement; the symmetric form is the correct one; production loses nothing by
carrying the asymmetric approximation in its regime.**

### G-2 — the σ_z → 0 limit (V-deep, paired 1D deltas, h = 0.72 column)

| rung σ_z | C_cat = cat1d − off | C_comp = 1d − off |
|---|---|---|
| 0.035 | **+0.0424 ± 0.0013** | **−0.0300 ± 0.0007** |
| 0.010 | +0.0057 ± 0.0002 | −0.0027 ± 0.0002 |
| 0.002 | **+0.0038 ± 0.0001** | **−0.0004 ± 0.0001** |

- **H-Zcomp FAIL branch fires (informative):** the completion-leg cost collapses ~70× —
  my registered structural counter-prediction (GW-width-driven persistence) was refuted by
  the instrument. Both legs' insertion costs are photo-z-driven; the V-deep
  asymmetric-insertion boundary (rows #120–#124) is itself σ_z-conditional — flagged to the
  [C-SYM] claim card as new stage-0 evidence.
- **H-Zcat PASS with a material caveat:** monotone ~11× collapse, inside the registered
  band — but the σ_z = 0.002 residual (+0.0038 ± 0.0001, ~40σ nonzero, ~9% of baseline)
  **exceeds the 0.0018 materiality yardstick**. The "no-cost limit" is approximate, not
  exact; a σ_z-independent floor component (e.g. S̄_φ variation across the ball's intrinsic
  z-extent) is the natural reading — measured, not mechanism-attributed.

## 4. Why the numbers stand

All cells at the freeze commit `4dd822ad` (single frozen instrument; preregs + extension +
scorers in one commit); registered scorer invocations only; 0 missing manifest entries; no
rail-flagged truth in any scored cell (wide grid); N-A byte-identity PASS at full R both
venues; N-b continuity ties the G-2 off cell to the prodcal record (≤ 0.78 combined-SE);
instrument continuity trio: ad-hoc audit +0.041 → registered +0.0428 (H-CAT), prodcal S-1
−0.033 → G-2 −0.0300 (C_comp), VERDICT-2 V-flat fused−off +0.00127/+0.00073/+0.00003 →
bit-identical wide-grid replication. Quality gate: 1546 tests green at freeze; 13 new
structural-null tests pin the knob extension. Budgets: G-1 ≈ 2.75 of 3 CPU-h; G-2 ≈ 0.9 + 
0.2 pretune of 6 CPU-h (κ = 1 — the G2-6 likely-κ-STOP did not realize).

## 5. What the adjudicator should still know (flags, none verdict-changing)

1. **[interpretive]** H-SYM's cov68 miss (0.475 vs ≥ 0.50 at h = 0.62) is a 0.6σ edge miss
   whose interpretation is now measured (SEP-Z: photo-z venue physics) — but the MIXED is the
   registered outcome of record; the resolution is a proposed [RULE], yours to ratify.
2. **[drafting, disclosed in the VERDICT]** The §4 branch's designated V-flat separating cell
   was registered without noting the VERDICT-2 absolute-leg confound (neither drafter nor
   verifier caught it in Parts I–IV); AMENDMENT B repaired it to paired-only and both
   separating reads ran.
3. **[interpretive]** The V-flat sym−off read is formally MIXED at h = 0.62 (+0.00120 vs the
   0.0010 edge) but is bit-identical to fused−off — the already-certified row-#124 lever;
   the symmetric ADDITION contributes exactly zero there.
4. **[instrument note]** The spec-z venue's coverage profile is distorted for BOTH symmetric
   and calibrated-off cells (over-wide 68% intervals; cov50 low at h=0.62) — a σ_z = 0.002
   venue property, descriptive, possibly relevant if a future front runs spec-z coverage
   claims.
5. **[scope]** G-1 bounds the 1D-leg [P3] materiality; the 2D-leg per-candidate fusion
   remains unbuilt/unmeasured (registered validity limit; 2D channel measured inert in S-1 /
   M-1 and exactly degenerate in every pair here).

## 6. The decisions (author-gated; per the binding default, each returns on its own)

| # | tag | decision |
|---|---|---|
| 1 | **[RULE]** | **Ratify the G-1 readout bundle:** H-G1 PASS (catalogue-leg fusion immaterial in regime, ≤ +0.0002), H-CAT PASS, H-SYM MIXED with the SEP-Z interpretation-A resolution (residual = photo-z venue physics; symmetric form calibrated-in-bias at spec-z), V-flat bit-inertness; the two disclosed drafting flags (§5 items 2–3). Consequence: **the ~170 CPU-h production catalogue-leg counterfactual is NOT warranted** (registered PASS meaning), and [C-SYM]'s mechanism story is confirmed at harness fidelity. |
| 2 | **[RULE]** | **Ratify the G-2 readout bundle:** H-Zcat PASS-with-material-residual (+0.0038 at σ_z = 0.002), H-Zcomp FAIL = both legs' costs photo-z-driven (my registered persistence prediction refuted); the σ_z-conditionality annotation onto the [C-SYM] claim card. |
| 3 | **[RULE]** | **[P3] presentation pick (proposal §2.4 / §5 item 2) — now with measurements in hand:** option **(a) documented-convention** is the measured-bound pick (retained catalogue convention, immaterial at ≤ +0.0002 in regime, symmetric form as future work), option **(b) resolved-in-paper** is equally available and now carries the FULL measured ladder (the derivation presented as correct for both legs + the harness showing symmetric calibrates at the amplitude venue). The record's orchestrator-derived note: (b) is no longer "lacking its grounding measurement" — that was G-1's purpose. Your pick. |
| 4 | **[RULE]** | **σ_z → 0 wording (decision item 3):** proposed wording of record — per-leg, with measured numbers: completion-term cost vanishes to sub-materiality at spec-z (−0.0004); catalogue-leg cost collapses ~11× but retains a material +0.0038 floor at σ_z = 0.002; extrapolation below stated as trend. Supersedes both the "caveated-statement" and blanket "no-cost" options. |
| 5 | **[DO]** | **Paper work already granted (row #121 item 4) now unblocked:** TO-MAKE figures + `discussion.tex:235` rewrite, now carrying these final numbers (+ the G-2 ladder as a candidate figure). Sequenced on your items 3–4 picks. |
| 6 | **[RULE]** | **Seeds (not opened):** the σ_z = 0.002 coverage-profile distortion (§5 item 4); the catalogue-leg σ_z-independent floor component (mechanism unattributed); both logged, neither opened. |

## 7. Provenance footer

Freeze commit `4dd822ad` (instrument + preregs + scorers + tests) · preregs
`PREREGISTRATION_G1_CATLEG_SYMMETRY.md` / `PREREGISTRATION_G2_SPECZ_LIMIT.md` (verifier
`VERIFIER_PRECHECK_G1G2.md` Parts I–VI, all amendments applied verbatim) · scorer outputs
`readout_g1_output.json` / `readout_g2_output.json` + the AMENDMENT-B paired reads (scorer
`paired_read`, values in the G-1 AMENDMENT-B VERDICT) · cells in `cells/`, probes in
`preflight/`, logs `amendment_b.log` + task logs · N-A referent
`cells/referent_preext_vdeep_250_production_off.json` (pre-extension HEAD `6504c8b9`
worktree) · all execution local dev machine · **autonomous-cycle session: [DO]-class steps
executed under the author's 2026-08-18 directive (quoted verbatim in both preregs);
orchestrator-derived reads flagged throughout; bands locked at registration and unchanged
after readout; branches presented, not adjudicated.**
