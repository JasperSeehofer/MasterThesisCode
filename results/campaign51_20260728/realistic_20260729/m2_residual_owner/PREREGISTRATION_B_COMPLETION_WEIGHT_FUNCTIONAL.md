# Pre-registration — Instrument B: the completion-weight functional read

Registered 2026-08-08, **BEFORE** the run. Research Cycle stage 2
(`docs/RESEARCH_CYCLE.md`); stage 0–1 parents (same directory):
`CLAIM_M2_RESIDUAL_OWNER_20260807.md` (committed `e253e0c1`) and
`STAGE1_READOUT_20260807.md` §5(B) + LOOPHOLE READS A1/A2 (committed
`c188f460`). Author authorization of record: **"go for B please"** (Jasper
Seehofer, 2026-08-08), selecting the readout's §5(B) named instrument —
*completion-weight functional read ("chord-vs-density-inside-the-weights")*.

**REGISTERED — committed before first execution (W-PRE-18). Append-only
discipline is in force from this commit.** Every band below is fixed at this
commit from EXISTING verified numbers only (each cited with file + field) and
may not be adjusted after any readout. Main HEAD at drafting: `c188f460`; this
file's commit is its immediate child and is the registration pin.

**Governing value ruling (binding, author, 2026-08-05,
`../gate_b_20260730/BIAS_HISTORY_LEDGER.md` §5):** scientific correctness +
novel insight over bias-removal. **A confirmed density-coupling is a
first-class result, not a nuisance** — branch (ii) below is a success ending,
not a failure mode. Corollary (author, 2026-08-05): *measure, never refute by
convenience.*

---

## 1. The question, sharp

Stage 1 closed with a dissolution-candidate verdict in modified-H-c form
carrying two qualifications:

- **(q1)** H-e (chance) closed only conditionally — literal win279-graph
  reading PASS at both venues (p ≈ 0.0020–0.0035), the alternative
  1620-sky-pair reading fails on power (p 0.062–0.070)
  (`adjudicate_a1_a2_results.json:A1`, `STAGE1_READOUT_20260807.md` A1 block).
- **(q2)** The smooth d_L-functional response explains only ~2/3 of the
  matched residual — full-chain reproduction ratios **0.6662 (iiib) / 0.6525
  (joint_r1)**, outside the pre-stated [0.70, 1.30] band
  (`a2_results.json:full_chain_reproduction_per_venue.<venue>.predicted_full_chain.all_events_fit.ratio_pred_over_obs`)
  — with a **+0.0083 ± 0.0029 (≈2.9 cluster-σ) carrier-level excess at FIXED
  (d_L, σ_dL)**
  (`a2_results.json:carrier_level_reproduction.fit_residual_matched_diff`),
  corroborated by the placebo null (−0.0051 ± 0.0050,
  `adjudicate_a1_a2_results.json:A2.placebo_control_vs_control`) and by D-1's
  independent d_L-matched re-match leftover (+0.0060–0.0072,
  `adjudication_results.json:d1.sensitivity_rematch_dL`).

Stage 1 also established that **matching cannot attribute** the residual
between the d_L-geometry family and the ball-density family inside the C-4
stratum — the covariates are collinear there (matching on ball count balances
log10_dL SMD 0.64 → 0.004 uninstructed and vice versa;
`adjudication_results.json:d2.<venue>.rungs`, verifier discrepancy govern-point
`STAGE1_READOUT_20260807.md` §4.1).

**The question this instrument answers:** over the FULL 1588-event population
— where the two families are expected to decorrelate far better than inside
the stratum — decompose the per-event completion-leg chord's dependence
JOINTLY on

- the **d_L family**: x1 = log10 d_L, x2 = log10(σ_dL/d_L) (A2's exact
  predictors, `a2_results.json:predictors`), and
- the **density family**: log10 ball w_pop totals and ball counts, 1D and 2D
  (D-2's exact covariates, `d2_results.json:covariate_definitions`),

and use the joint model to **attribute the M-2 matched residual** — and
specifically the +0.0083 fixed-d_L excess — between the families:

> **(a)** does the fixed-d_L excess dissolve when density terms enter (the
> residual is fully owned by the joint covariate account — complete
> dissolution), or **(b)** does the density family carry a stable,
> sign-coherent, significant share at fixed d_L (a genuine density-coupling of
> the completion-leg functional — a novel finding)?

Secondary attribution target: the weight channel **T_wG** (LMDI
composition-weight term, −24 %/−30 % offset of the residual,
`d1_results.json`), whose per-event variation is entirely the catalogue-leg
share S_A — the one object that *sees* the ball contents by construction.

---

## 2. The instrument

| item | value |
|---|---|
| Script (new file, this directory) | `b1_completion_weight_functional.py` → `b1_results.json` |
| Compute class | FREE — existing CSVs + frozeng ball emits; **no likelihood evaluations, no cluster jobs** |
| Chord objects (venue-identical, bitwise-asserted) | `c_pure = ln L_comp(0.60) − ln L_comp(0.73)`; `c_gfrac = ln(g_frac·L_comp)(0.60) − same(0.73)` — exactly A2's objects (`a2_results.json:objects`) |
| Weight object (per venue) | `S_A` = LMDI catalogue-leg share `L(A60,A73)/L(F60,F73)` per `d1_component_decomposition.py:26` (log-mean machinery `:122`); outcome `y3 = log10(S_A)` on the `S_A > 0` subset |
| d_L design **D** | OLS total-degree-3 bivariate polynomial in (x1, x2), 10 columns — byte-compatible with `a2_completion_functional.py:158` (`poly_design`), all-events fit primary, controls-only fit reported (A2's two variants) |
| Density design **P** | total-degree-3 bivariate polynomial in **(z1, z3) = (log10_n_ball_2d, log10_W_pop_2d)**, 10 columns — the 2D density pair, built verbatim by `d2_confounding_check.py`'s ball-covariate builder (`:383–412`; `w_g = R_eff_per_mbh(M_g)/(1+z_g)`, M-4 deref against the staged pruned catalogues) |
| Joint design **J** | column union of D and P (single intercept), 19 columns, fitted on all 1588 events |
| Radius overlay (DS-5) | append `[r, r², r³]`, `r = log10_radius_chord` (M-2 original covariate), to BOTH D and J → designs D+r, J+r |
| Matched-pair machinery | the exact M-2 pairs: 385 overlap events, deterministic 1-NN (log10_radius_chord, SNR), 234 control clusters; `matched_read` + `cluster_se` verbatim from `a2_completion_functional.py:142,377`; `signflip_p`/`cluster_signflip_p` verbatim from `d1_component_decomposition.py:132,142` |
| RNG policy | all decision-bearing point estimates and SEs are RNG-free; p-values: N_PERM = 20000, fresh seeds **signflip 20260808, cluster 20260808** (anchors below are seed-free) |
| Inputs (md5-asserted before any read) | `results/run_20260804_postfix/{iiib,joint_r1}/diagnostics/event_likelihoods.csv` (md5 `ee9c997b7f41b18a34049e7e0ff1a20f` / `c895f2e4a5b4fd127e347a941d6b6263`); `results/run_20260804_postfix/joint_r1/diagnostics/prepared_cramer_rao_bounds.csv` (md5 `9a1f2a14384a9281c97ca3be312ddaab`, 1590 rows); `results/run_20260804_frozeng/<venue>/posteriors_with_bh_mass/h_0_73.json` (md5 `34c50e91028b6a6458a2b145db545705` iiib / `6c5aff4896459105a8ac047f1a48ca8c` joint_r1); staged pruned catalogues per `d2_results.json:provenance.catalogues` |
| Inherited assumption (flagged, not re-proven) | `event_idx == CRB row index` (M-2 flag, carried by D-1/D-2/A2 unchanged) |

The 1D density members (z2 = log10_n_ball_1d, z4 = log10_W_pop_1d) and the
4-variable total-degree-2 density design are **reported robustness reads R-1/
R-2** (non-decision-bearing, §6).

---

## 3. The decision statistics, exact

All fits OLS (`numpy.linalg.lstsq`), all 1588 events unless stated; the run
agent has zero freedom below.

**DS-1 — family-decorrelation gate (runs FIRST; may abort attribution).**
For each density covariate z ∈ {z1, z2, z3, z4}: cross-family R² = R² of OLS
of z on design D over the 1588 events. Also report: pairwise Pearson and
Spearman correlations across families; VIFs of all columns of J; condition
number of the column-standardized J.
*Abort criterion (locked):* attribution proceeds **iff at least one of the two
2D density covariates (z1, z3) has cross-family R² ≤ 0.80** (equivalent to
VIF ≤ 5, the standard moderate-collinearity line — a convention, flagged as
such, since no repo number can supply it: the full-population cross-R² has
never been measured) **and** the standardized-J condition number is < 1e8.
If the gate fails, DS-2..DS-5 are not interpreted; branch (iii) fires with the
"undecidable at population level" wording (§5).

**DS-2 — the fixed-d_L excess re-estimated with density terms.**
For each chord object c ∈ {c_pure, c_gfrac}:
`E_D` = matched mean paired diff of (c − ĉ_D) over the 385 pairs (ĉ_D =
all-events D-fit prediction) — must bitwise-reproduce the A2 anchors
(V-2, §4). `E_J` = same with ĉ_J (all-events J-fit).
*Collapse criterion (locked):* `|E_J| < 1 × cluster_se(E_J)` **and** cluster
sign-flip p(E_J) ≥ 0.0455, for **both** chord objects.

**DS-3 — the density-attributed component of the excess (exact additive
attribution).** `A_ρ` = matched mean paired diff of (ĉ_J − ĉ_D) over the same
385 pairs. Identity check (machine precision): `E_D = A_ρ + E_J` per pair sum.
Report `A_ρ`, its cluster-robust SE, sign-flip p and cluster sign-flip p, and
the share `A_ρ / E_D`.
*Significance criterion (locked):* cluster sign-flip p(A_ρ) < 0.0455 and
sign(A_ρ) = sign(E_D) (> 0), for both chord objects.

**DS-4 — stratum-composition reproduction ratio recomputed with the joint
model.** Per venue: predicted total 2D = observed T_legA + predicted T_legB,
where predicted T_legB composes ĉ_gfrac through the exact A2 chain (S_B and
`dln(1−wt)` from the venue weight constants;
`a2_results.json:full_chain_reproduction_per_venue.<venue>.weight_constants`
must reproduce: wt_073 = 0.0619668411108587 / 0.0708022510819941, dln(1−wt) =
−0.022898659328417684 / −0.027145606456715793; observed T_legA =
−0.004711922212657903 / −0.005129927211131639).
`ρ_J` = predicted/observed total 2D with the J-fit.
*In-band criterion (locked):* `ρ_J ∈ [0.70, 1.30]` at **both** venues
(all-events fit; controls-only reported) — the band is A2's own pre-stated
reproduction band, reused verbatim
(`a2_results.json:pre_stated_criteria.reproduction`).

**DS-5 — radius-overlay stability (decision-bearing for branch (ii) only).**
Recompute DS-3 with designs D+r and J+r: `A_ρ^{+r}` = matched diff of
(ĉ_{J+r} − ĉ_{D+r}).
*Stability criterion (locked):* sign(A_ρ^{+r}) = sign(A_ρ) and cluster
sign-flip p(A_ρ^{+r}) < 0.0455 for both chord objects. Additionally, sign
coherence: the partial associations of each chord object with z1 and with z3
(both residualized on D) carry the same sign as each other, for both chord
objects, and that sign is unchanged in all 5 folds of a 5-fold CV refit
(A2's CV machinery) and under the controls-only fit.

**DS-6 — the weight channel (secondary, attribution context; never a branch
criterion).** Per venue: count of `S_A = 0` events (structural zeros: empty
with-BH ball); on the complement, fit y3 = log10(S_A) on D, P, J; report the
three R², the semi-partials ΔR²(P|D) and ΔR²(D|P), and partial-association
signs. This locates whether any density share is carried by the weight-side
object that *definitionally* sees the ball — feeding the §5(ii) escalation
tracing, not the branch decision.

Semi-partials `ΔR²(P|D) = R²(J) − R²(D)` and `ΔR²(D|P) = R²(J) − R²(P)` are
reported for every fitted object (context, never a criterion).

---

## 4. Validity criteria (all hard asserts, run before any decision read)

- **V-1 — bitwise M-2 anchor reproduction:** matched 2D combined residual
  +0.022252643015992925 (iiib) / +0.020697491999731973 (joint_r1)
  (`../crossterm_instrument/m2_results.json`, reproduced bitwise by D-1/D-2/A2).
- **V-2 — bitwise A2 anchor reproduction:** primary D-fit R² 0.8832406614871592
  (c_pure) / 0.8747947939465979 (c_gfrac)
  (`a2_results.json:functional_characterization.<obj>.poly_fits_all_events.degree_3.r2`);
  fixed-d_L excess E_D = +0.008340732036016641 (c_pure) /
  +0.008352697414993901 (c_gfrac) with cluster SEs 0.0029160903955559583 /
  0.0029146831062962105
  (`a2_results.json:carrier_level_reproduction.fit_residual_matched_diff`);
  d_L-only chain ratios 0.6662301458609439 / 0.652514656137263 (all-events).
- **V-3 — venue identity asserts on completion-leg columns:** `L_comp`,
  `B_num`, `B_num_wbh`, `g_frac` bitwise identical across venues at h = 0.60
  and h = 0.73 (`a2_results.json:venue_identity_check` — all True at A2 time;
  re-asserted here). Chord objects are computed once; S_A/T_wG objects are
  per-venue and must never be assumed identical.
- **V-4 — density-covariate fidelity anchor:** with the rebuilt ball
  covariates, the D-2 m2 rung (matching on log10_radius_chord, SNR,
  log10_n_ball_2d) must bitwise-reproduce the recorded effects
  +0.003949491314625633 (iiib) / +0.003845526436421696 (joint_r1)
  (`d2_results.json:trajectory.<venue>.effects_2d.m2`, verifier-confirmed
  bitwise in `adjudication_results.json:comparison_vs_reported.d2`).
- **V-5 — census + determinism:** census asserts 1620 sky pairs / 279 window
  pairs / 385 overlap events; 65108 rows per event_likelihoods CSV (= 1588
  events × 41 h; events 1203, 1356 dropped); all input md5s as tabled in §2;
  every decision-bearing number RNG-free; p-values only from the locked
  seeds/N_PERM; two consecutive full runs of the script must emit
  byte-identical JSON apart from a timestamp field.

Any V-failure **voids the run before any branch is read**: fix the instrument
or report the failure; no decision statistic is quoted from a run with a
failed validity assert.

---

## 5. Pre-registered readings and branches — locked BLIND

**Blindness fact:** as of this commit, **no joint (d_L + density) model of any
completion-leg object exists anywhere in this repository.** A2 fitted the d_L
family only (`a2_results.json` contains no density term); D-1/D-2 matched,
never fitted; the full-population family cross-R² of DS-1 has never been
computed. Every band below is locked from already-verified numbers (cited) or
from named conventions (flagged), before any joint-model number exists.

**Collinearity caveat, pre-stated:** the families decorrelate only partially
even population-wide — that expectation is structural (ball volume grows with
d_L and radius), and the within-stratum evidence is that they are strongly
collinear there. DS-1 is the pre-stated diagnostic (family cross-R², VIF,
condition number) and its §3 abort criterion is the registered response if
decorrelation is insufficient to attribute. Context (existing): among
controls, the chord's strongest single predictor is log10_dL (Spearman +0.8414
/ +0.8344) with density opposite-signed (−0.50..−0.65)
(`adjudication_results.json:d2.<venue>.spearman_cov_vs_2d_chord_controls`) —
opposite signs are why joint attribution is possible at all where correlation
permits.

### Branch (i) — DISSOLVES (complete dissolution)

**Fires iff** DS-1 gate passes, DS-2 collapse criterion holds (both chord
objects), **and** DS-3 in-band holds (`ρ_J ∈ [0.70, 1.30]`, both venues).

⇒ The joint covariate account owns the full matched residual: the ~1/3
fixed-d_L excess dissolves into density terms and the stratum-composition
chain closes in-band. **Chronicle as complete dissolution** of the M-2
residual (modified H-c, now fully grounded: smooth completion-leg functional
of (d_L, σ_dL) + density composition of the stratum), closing qualification
(q2). The thread routes to stage 5/6 (author ruling → chronicle + ledger row);
no follow-up instrument.

### Branch (ii) — DENSITY-COUPLING (novel finding)

**Fires iff** DS-1 gate passes **and** DS-3 significance holds (A_ρ > 0,
cluster p < 0.0455, both chord objects) **and** DS-5 stability holds (radius
overlay + sign coherence + CV/controls-only stability).

⇒ The density family carries a stable, sign-coherent, significant share of
the completion-leg chord **at fixed (d_L, σ_dL)** — a genuine
density-coupling, per the author value ruling a **first-class novel result**.

*Precedence:* branches (i) and (ii) can co-fire (dissolution INTO a
significant density term). If both fire, the verdict is recorded as
**"complete dissolution WITH confirmed density-coupling"** — (ii)'s
escalation applies and the chronicle carries both statements.

*Escalation (named, and it is NOT `/physics-change`):* a new stage-0 intake in
this directory, working name **`CLAIM_COMPLETION_DENSITY_COUPLING`** — the
follow-up thread that traces the coupling to an object. No production formula
is implicated by branch (ii) itself. `/physics-change` would be triggered
**only if** the follow-up traces the coupling into a production object,
specifically: (a) the completion-leg numerator construction
`B_num`/`B_num_wbh`/`L_comp` in `bayesian_inference/bayesian_statistics.py` —
which is exonerated as a *defective integral* (#80/#87) and may be reopened
only with the new tracing evidence, on the derivation question of whether a
catalogue-independent completion integral may lawfully co-vary with ball
density at fixed d_L; (b) the composition-weight objects (`w_G`/`w̃_G`,
`alpha_G_phi`, `r_Malm`, `D_tilde_phi`) — if DS-6 localizes the share to the
weight channel, the follow-up intake **must** run the
`../crossterm_instrument/NEGLECT_TRIGGER_REGISTER.md` §5 trigger-(b)/(f)
assessment (mixture-composition re-implication) before anything else; (c) the
ball construction in `galaxy_catalogue/handler.py`
(`get_possible_hosts_from_ball_tree`). Until such a trace exists, this branch
authorizes **measurement and chronicle only**.

### Branch (iii) — MIXED / UNDETERMINED (first-class, non-forcing)

**Fires** on anything else, including: DS-1 abort (⇒ verdict wording: "family
attribution undecidable at population level too — the collinearity map is the
finding"; the ~1/3 excess stays qualified exactly as stage 1 left it); DS-2
collapse without DS-3 in-band (carrier-level dissolution, chain still short —
report the gap between carrier and chain as the open object); DS-3
significance failing DS-5 (share is geometry-absorbable — radius, not
density-specific; report as such, no novel-finding claim); a venue split on
ρ_J; sign-incoherence between z1 and z3; identity-check failure in DS-3
(voids the attribution arithmetic — instrument bug until proven otherwise).

*Gap handling (explicit):* ALL DS values are reported for both venues and both
chord objects regardless of branch; no branch is forced; opposite-direction
results (e.g. E_J **grows** under J) are findings and are reported as such.
The only further free reads permitted inside this registration are R-1/R-2
(§6) — anything beyond them requires a new pre-registration. A branch-(iii)
outcome returns to the author with the split laid out; the default route is
chronicle-with-qualifications (stage 1's verdict candidate stands, with this
instrument's numbers appended).

**Anti-tuning:** the locked constants of this registration — cross-R² 0.80 /
condition 1e8 (DS-1), 1 × cluster-SE + α = 0.0455 (DS-2), α = 0.0455 + sign
(DS-3), [0.70, 1.30] (DS-4/DS-3-band), the DS-5 stability battery, seeds
20260808/20260808, N_PERM 20000 — are fixed at this commit and may not be
adjusted after any readout. α = 0.0455 is the house 2σ-equivalent used by
M-2/D-2/A1; [0.70, 1.30] is A2's pre-stated band; the 1σ collapse bar is the
authorized instrument sketch's own criterion; the collinearity thresholds are
named conventions (VIF-5 equivalence; numerical-rank hygiene) because no
measured number exists to supply them — all flagged, none post-hoc.

---

## 6. Reported robustness reads (non-decision-bearing)

- **R-1 — 4-variable density design:** total-degree-2 polynomial in
  (z1, z2, z3, z4) (15 columns) replacing P; DS-2/DS-3 statistics recomputed
  and reported. Colors the chronicle wording only.
- **R-2 — 1D-member overlay:** P extended with [z2, z4, z2², z4²]; same
  reporting. (The 1D channel carries no matched residual — C2 — so these are
  attribution context, not outcome claims.)
- **R-3 — density-design degree ladder:** degrees 1/2/4 for P, R² ladder
  reported (underfitting/saturation check mirroring A2's ladder).

---

## 7. Scope guard

- **FREE compute only.** Existing CSVs + frozeng ball emits + staged pruned
  catalogues; no likelihood evaluations, no cluster jobs, no re-simulation.
- **No production formula is touched.** This file authorizes a measurement,
  never a formula change; any fix routes through `/physics-change` with its
  5-item package, author-gated, only via the §5(ii) escalation path.
- **No edits to existing files.** New files only, in this directory
  (`b1_completion_weight_functional.py`, `b1_results.json`); the claim file,
  stage-1 readout, ledger, register, and book are untouched by the run.
- **Exoneration discipline honored:** #80/#87 (`L_comp`/`B_num` defective
  integral), #61 (`w_G` bookkeeping), R-A (`g_frac` h-slope), row 96
  (Eq. (31) cross-term NEGLECT) are not re-litigated; the only reopening path
  is the named-evidence escalation of §5(ii).
- **Adversarial verification:** per house practice the readout is followed by
  an independent re-implementation pass (fresh seeds) before any branch is
  presented to the author; where the verifier files a discrepancy, verifier
  numbers win.
- Model/effort policy: mechanical extraction cheap; interpretation and the
  branch presentation at full effort; **the branch call is presented to the
  author, never self-adjudicated.**

---

Verdict to be appended below by the session that reads out the run — after
this file is committed, no edits above this line.
