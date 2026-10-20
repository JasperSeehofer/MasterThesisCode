# T1.4 — Richardson half-step falsifier readout (PA-HIER-33 item (ii)) — 2026-08-30

**Independent reader of tree-2 node T1.4** (the run itself was executed by the orchestrator/runner-8,
registered in `PREREGISTRATION_HIER_HTHETA_20260826.md` PA-HIER-33 item (ii); inputs also
`T1_3_ES_NULL_DET_VALIDITY_20260830.md` sections 2-5). Foreground only, <=600 s per command, no git,
no ssh, append-only; every number below is `{value, source, date}`; adjudicated only against the
three bands PA-HIER-33 registered before this run. Did not touch `b8_cal_harness*` or its work roots
(runner-9/B8.2 S3 concurrently running). Full numbers: `t1_4_readout.json` (this directory).

## 1. What this arm was for

PA-HIER-32(d)'s null correction for the s-axis score (`Es_null_det`) is the exact finite-step bias
of a SINGLE isolated host's kernel — but the registered statistic is a sum over hundreds of
candidate galaxies, whose likelihood is much flatter in `ln s`. `T1_3_ES_NULL_DET_VALIDITY` derived
a much smaller "correct" null (`+0.0013 +/- 0.0008`, the Bartlett-identity estimate from the three
banked P1 s-nodes) and proposed replacing PA-HIER-32(d)'s subtraction with it (PA-HIER-33). That
derivation carries a disclosed weakness: the 3-node estimator underestimates the true finite-step
bias by ~19% on a test family. This arm removes that weakness by adding two MORE s-nodes at HALF the
step (`s = 2^(+/-1/4)`) and forming the Richardson secant `S_R = (4*S_half - S_full)/3`, which has NO
`O(Delta^2)` bias term for any smooth per-event likelihood — a cleaner instrument than the 3-node
estimate, and fresh, unseen data.

## 2. What was measured (re-derived independently from the raw CSVs, not read off the driver's cache)

Re-derivation method: for each of the 461 pooled events (4 seeds x ~106-130 events), computed
`S_full = [ln L(s+) - ln L(s-)]/ln2`, `S_half = [ln L(s+half) - ln L(s-half)]/ln2 x 2` (i.e. the
secant over the half-step pair), and the Richardson combination `S_R = (4 S_half - S_full)/3`,
directly from `event_likelihoods.csv`'s `combined_no_bh` column at each of the 5 nodes per seed —
without calling `hier_s0_driver.py`. Result reproduces the driver's own `s0a_score_output.json` to
the last meaningful digit (float noise only, ~13th significant digit):

| statistic (no-BH, primary channel) | mean | SEM (per-event) | Z | n |
|---|---|---|---|---|
| `score_lns` (= `S_full`, unchanged from P1) | +0.00396 | 0.01289 | +0.31 | 461 |
| `score_lns_R` (Richardson secant, this arm) | **+0.00640** | 0.01361 | **+0.470** | 461 |
| paired shift `score_lns_R - score_lns` | **+0.002435** | **0.001404** (per-event) / 0.001724 (seed-clustered, the binding one per PA-HIER-5 leg a) | — | 461 |

Per-seed `score_lns_R`: 900101 +0.0040 (n=106), 900102 **-0.0355** (n=120), 900103 +0.0518 (n=105),
900104 +0.0104 (n=130). Seed 900102 is the only seed with a negative pooled mean (on both the
Richardson statistic and the raw `score_lns`) — flagged, not adjudicated: its per-seed SEM (~0.028)
does not distinguish this from sampling scatter at n=4 seeds, and it is the same class of
opposite-sign stratification the derivation's own section 2.5 item 4 disclosed (c-quartile and
`z_g` strata of opposite sign inside the P1 pooled mean).

With-BH channel (companion, REPORTED-ONLY, no registered band): `score_lns_R = +0.0384 +/- 0.0179`
(Z +2.145, inside 3); paired shift `+0.00758 +/- 0.00207`.

Cost: measured wall 7976.4 s (log `START` 19:45:48 -> `END` 21:58:47) x 14 cores = **31.0 CPU-h**
(2.22 h wall) — about 1.5x the prereg's own ~20 CPU-h / 1.5 h estimate.

## 3. Adjudication against the three registered bands (3 sigma of the paired SEM, applied literally)

PA-HIER-33 registered three mutually exclusive point predictions for the paired shift and a rule:
decide at 3 sigma of the measured paired SEM. Using the task-specified per-event SEM (0.0014037):

| null | predicted paired shift | measured - predicted | sigma | verdict |
|---|---|---|---|---|
| PA-HIER-32(d) (unweighted, current rule of record) | -0.046 | +0.04843 | **~34.5 sigma** | **EXCLUDED** |
| c-weighted (gate-doc 5.6 proposal) | -0.025 (range -0.023 to -0.027) | +0.02743 | **~19.5 sigma** | **EXCLUDED** |
| Bartlett null (PA-HIER-33, this amendment) | -0.0013 +/- 0.0008 | +0.00373 | **~2.66 sigma** | **NOT excluded** (2.66 < 3) |

The Bartlett-null margin is inside the registered decision rule but not by a wide margin — 2.66
sigma sits close to the 3 sigma line, and the amendment's own disclosed caveats bear directly on it:
(i) the 3-node Bartlett estimate is disclosed as ~19% low on a test family, so the true predicted
shift could be as large (in magnitude) as ~-0.0017, which would move the residual to +0.00403 and
the sigma to ~2.87 — still under 3, but closer; (ii) using the seed-clustered SEM (0.001724, the
PA-HIER-5-leg-(a) BINDING choice when it exceeds the per-event SEM, which it does here) gives 2.17
sigma, i.e. the more defensible SEM choice makes the null read MORE consistent, not less. Neither
adjustment crosses the line. Applying the registered rule literally: **the Bartlett null is NOT
excluded.**

## 4. Verdict of record

- The fresh-data falsifier **REFUTES** the PA-HIER-32(d) null (~34.5 sigma) and the c-weighted null
  (~19.5 sigma) as candidate null expectations for `score_lns`.
- The **Bartlett-scale null (~0, PA-HIER-33) is the only one of the three that survives** the
  registered 3-sigma rule.
- Under PA-HIER-33's proposed rule, the Delta^2-free Richardson secant itself is null at truth
  (Z = +0.470, no-BH primary channel) — **the s-axis score at truth is CONSISTENT WITH ZERO** under
  that rule.
- **This does not change the verdict of record.** The P1 B0-A' (s) STOP stands under the current
  rule of record (PA-HIER-32(d), unweighted) exactly as PA-HIER-33's own reading rule specified in
  advance: this arm adjudicates *which null is correct*, not the s-axis verdict itself. Re-reading
  P1 under whichever null this arm favors requires the author's ratification of PA-HIER-33.
- With-BH channel: REPORTED-ONLY, Z = +2.145 (inside 3), not adjudicated (no registered band).

## 5. What returns to the author

PA-HIER-33's ratification question returns to the author with a fresh, decisive, unseen-data result
behind it: the amendment's central claim (the many-candidate likelihood's true finite-step null is
approximately zero, not the single-host or c-weighted values) held up against a Delta^2-free
instrument built specifically to be immune to the disclosed weakness of the 3-node estimate that
originally produced it. **S0-B remains unlaunched.**

*Stamp: independent reader, 2026-08-30. Foreground only, no git, no ssh, no code, no compute (zero
`evaluate()` calls); read-only on `hier_s0_zwin_run/**`; did not touch `b8_cal_harness*` or its work
roots. Launched under rows #255/#268 — tree 2 node T1.4 (Richardson falsifier, independent reader).*
