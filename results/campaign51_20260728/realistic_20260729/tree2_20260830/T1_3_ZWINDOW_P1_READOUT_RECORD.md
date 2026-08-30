# T1.3-zwin P1 — independent-reader readout (2026-08-30)

**Role:** independent reader of runner-7's run of record. Did not build the instrument, did not
run the registered measurement, did not touch source code or the concurrently-running B8.2 files.
Foreground only, no git, no ssh, append-only. Launched under rows #255/#268 — tree 2 node T1.3-zwin
(P1 readout).

**Registration:** `PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md` sections 5.6/9 (P1 arm, F1 falsifier),
`PREREGISTRATION_HIER_HTHETA_20260826.md` PA-HIER-32(d) (corrected `score_s`), the S0-A registered
map §4.1/§4.5. Run of record: `hier_s0_zwin_run/` (4 seeds × {truth, s_plus, s_minus},
`theta_zwindow=on`, `z_window_k=4.0`, `theta_phi_divisor=on`, `sky_cone_k=1.5`, `theta_sites=2.2`,
smear off, h=0.73).

## 0. One-paragraph summary

Every number the driver itself reported (`s0a_score.md`) reproduces **to the last digit** from an
independent re-implementation reading only the raw `event_likelihoods.csv` + `es_null_det.csv`
files — the arithmetic is not in question. The **closed-form `Es_null_det_i`** itself is also
independently confirmed: a from-scratch reimplementation of PA-HIER-32(d)'s formula (not calling
the driver's own function) matches the cached values to machine precision (8 hosts, seed 900101,
max diff 8.3e-17). Where this reader diverges from a purely literal reading is a **convention
finding, not an arithmetic one**: the registered falsifier F1 names the **c-weighted** `Es_null_det`
convention as primary "fixed before unblinding P1", but the driver's own `compute_scores()` never
applies that weighting — a gap the gate document's own Implementation record already discloses and
explicitly says a reader "must" account for. Re-deriving the c-weighted statistic independently
(cross-validated against the forensic's own `c_i` reference numbers) **passes** the same band
comfortably, landing close to the registered predicted point. The two conventions disagree by more
than the margin to the band edge.

## 1. The literal read (driver's own output, unweighted convention)

| statistic | mean | SEM | Z | n |
|---|---|---|---|---|
| score_s_raw | +0.003887 | 0.012639 | +0.3075 | 461 |
| score_lns | +0.003965 | 0.012894 | +0.3075 | 461 |
| **score_s (corrected, driver's own/unweighted)** | **−0.042371** | **0.012752** | **−3.3228** | **461** |

**Registered band:** `|Z_s| ≤ 3.0`. **Literal verdict: FAILS**, by 0.323 in Z (10.8% over the
threshold) — B0-A′ (s) persists. A15 false-fail rate at this band, N=461: 0.27% two-sided
(PA-HIER-32's own restatement), so this is not a marginal-noise miss at the 5% level; it clears
the band edge by more than 10 SEM-fractions of slack the null distribution allows.

With-BH channel (same read): raw/lns Z = +1.848, corrected (unweighted) Z = −0.935 — both inside
the band regardless of convention (the with-BH channel was never the registered decisive read;
reported for completeness).

Per-seed corrected (unweighted) score_s: seed 900101 Z=−1.647 (n=106), 900102 Z=−2.961 (n=120),
900103 Z=−0.104 (n=105), 900104 Z=−2.046 (n=130) — three of four seeds individually negative and
one (900102) individually within 0.04 of the pooled band edge; no single seed is an outlier driving
the pooled fail.

## 2. E12's own prediction, checked both ways

Section 5.6 registered **two** predictions for P1, in two different statistics:

- **Raw/ln-s form** (E12's own reference quote): predicted Z → −0.5 ± 1.0, band [0, +2.5].
  **Measured: Z = +0.3075. MET** — comfortably inside, and inside the E12 reference point's own
  ±1 uncertainty. This is the number that moved the farthest: the pre-zwindow divisor-only run
  (T1.2 recert, row #266) measured raw Z_s = **−5.971**; this run's raw/lns Z_s = **+0.308**. The
  z-window fix removed essentially all of the truncation defect the *raw* statistic sees.
- **Corrected, c-weighted form** (section 5.6's registered PRIMARY for P1, point −0.026±0.012,
  Z≈−2.1, band [−0.031,+0.005]): **not computed by the driver at all** (see §3). The driver's own
  reported "corrected" number uses a different (unweighted) convention that section 5.6 never
  registered as primary.

## 3. The convention gap (the decisive finding of this readout)

`PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md`'s F1 falsifier text reads: *"P1 must return |Z_s| ≤ 3
in PA-HIER-32's debiased statistic (**c-weighted convention, fixed before unblinding**...)"*. The
same document's own "Implementation record" section (builder, distinct agent) discloses, verbatim:

> "The raw-vs-c-weighted `Es_null_det` convention Revision note 2 downgraded to PROPOSED pending an
> author/orchestrator [RULE] is UNCHANGED by this implementation: this driver's own `compute_scores`
> has never applied a catalogue-share weighting `c_i` to any statistic (raw pooling only, as
> before) ... **Whoever reads P1's output against F1 must still apply Revision note 2 item 3's
> process constraint (report both conventions, do not declare F1 CONFIRMED/REFUTED on the driver's
> `score_s` alone) — this implementation record does not, and cannot, discharge that constraint.**"

So the `−3.32` figure that fails the band is, by the gate document's own words, not the number F1
names. This reader computed the c-weighted statistic directly from the same raw CSVs:

  score_s(c-weighted) = score_lns − c_i × Es_null_det_i,  c_i = 1 − B_num / (combined_no_bh × D̃_φ)

(the forensic's own `c_nb` definition, `f4_mechanism.py`). Cross-check: this run's `c_i` has
mean 0.6161 / median 0.6592, matching the forensic's independently-quoted reference (mean 0.616,
median 0.651) to within sampling noise — the definition is applied consistently.

| statistic | mean | SEM | Z | n |
|---|---|---|---|---|
| score_s (unweighted, driver's own) | −0.042371 | 0.012752 | **−3.3228** | 461 |
| **score_s (c-weighted, this reader, DERIVED)** | **−0.023052** | **0.012906** | **−1.7861** | 461 |
| registered predicted point (c-weighted) | −0.026 | 0.012 | −2.1 | — |

The c-weighted read **passes** the registered band by a wide margin and sits close to the
registered predicted point/band (`[−0.031, +0.005]`, mine at −0.023 is inside it). The unweighted
Es_null_det mean this run (0.0463) matches E13's *unweighted* per-unit-s figure (+0.0455) almost
exactly, while the c-weighted mean is what the forensic quoted as +0.0265 — confirming the driver
subtracted the larger (unweighted) offset from an already-near-null raw mean, manufacturing most
of the "corrected" fail. This is exactly the mechanism the task's own framing anticipated ("the
question is whether E13's null-offset derivation still applies once the window is θ-consistent")
plus a second, more mundane contributor this reader is adding to the record: **the wrong
convention of that same offset was applied**, on the gate document's own terms.

This reader does **not** declare F1 CONFIRMED or REFUTED — per the gate document's own stated
constraint, that call needs both conventions in front of the author/orchestrator, and the
c-weighted number above is a *derived cross-check by an independent reader*, not a re-run of any
registered instrument.

## 4. A14/F1 falsifier text, applied literally

*"|Z_s| > 3 REFUTES the attribution of the s-axis residual to the candidate-window truncation ...
[STOP returns as] INSTRUMENT-DEFECT (s) UNRESOLVED with the next candidates named here: (n1) the
S̄_φ table's own σ_z dependence ... (n2) the V2 mixture-weight covariance."*

Applied literally to the driver's own (unweighted) `−3.32`: **F1's literal trigger condition is
met** — the STOP reasserts, and per A14's own text, the routing would move to (n1)/(n2), not stay
with the truncation attribution. Applied to the c-weighted number this reader derived: the trigger
condition is **not** met, and the s-axis truncation attribution (E12) stands. Sharper F1 checks
(hw_sig-q1 class move ≥ +0.15, n_cand median growth ×2.2–3.6) were **not independently verified by
this reader** — they need the candidate-level dump joined to f7 quartiles, which is outside this
run's own CSVs and this reader's time budget; disclosed as unverified, not as failing.

## 5. Class split

- **True dark class (host_galaxy_index = −1):** 0 of 461 pooled events lack an `es_null_det` cache
  row this run — i.e. by that (the only available) proxy, dark n = 0, matched n = 461. This
  differs from the T1.2 recert's disclosed dark n=5 / matched n=456 on an equal n=461 total for
  (nominally) the same underlying venue draws. Not chased down further — flagged, not adjudicated.
- **Zero-`L_cat_no_bh`-at-truth events** (a different, z-window-related null): exactly 2 of 461
  (seed 900103 event 25, seed 900104 event 51), both with `score_lns` exactly 0.0. Direction is
  consistent with section 5.6's own prediction that the z_out class (~8 previously) is "recovered
  ... within 2" under the k=4 window.

## 6. Gates

- **GATE ENG:** s_plus/s_minus mean fraction of events moved = 0.99570, PASS (≥ far above any
  plausible bar).
- **GATE PARITY:** `pass_exact=False` on every seed, consistent in *kind* with the already-RATIFIED
  E19 disclosure (a 401→4001 generator z_true-grid comparand delta, not an estimator defect). This
  reader flags that the observed max `ln_L` diffs (max rel 3.9%–44.7% across seeds, worst on
  seed 900101) are numerically much larger than the previously-quoted 5.718e-4 headline figure —
  that figure described the injection-level z_true/d_L delta, not the downstream `ln_L` deltas
  reported here, and a steep `ln_L` sensitivity to a tiny z_true shift for edge-case (low
  candidate-count) events is a plausible, consistent-with-ratified amplification path. Not
  re-adjudicated by this reader; disclosed for the record.

## 7. Cost

Measured: wall 8248.47 s × 14 cores = **32.08 CPU-h** (wall 2.291 h) — against the gate document's
own §6 P1 anchor (~9,000 s / 2.5 h wall, ~35 CPU-h nominal): within ~8% of the wall-time anchor,
consistent.

## 8. Verdict of record

**Literal (per the registered band, on the driver's own unweighted output):** B0-A′ (s) persists —
`|Z_s| = 3.3228` — **INSTRUMENT-DEFECT (s), REPORTED-ONLY** (PA-HIER-28 item 9 cap). Raw/lns-form
observation (REPORTED): Z = +0.3075, inside band, matching E12's own prediction.

**Disclosed alongside, per the gate document's own process constraint (not this reader's ruling):**
the registered PRIMARY (c-weighted) statistic, independently derived from the same raw data, is
Z = −1.786 and **passes** the same band, close to the registered predicted point. F1 cannot be
honestly read as CONFIRMED-REFUTED on the driver's own number alone; the convention gap is the
decisive open item.

**Licenses:** nothing beyond rows #255/#268 — tree-2 charter. No S0-B, no further Stage-P/F beyond
what is already registered. Per the task's own routing: an Es_null_det-validity derivation (does
the c-weighted convention correctly capture the combined-channel secant expectation) is dispatched
before any re-run; S0-B stays unlaunched.

Full numeric payload: `t1_3_p1_readout.json` (same directory).
