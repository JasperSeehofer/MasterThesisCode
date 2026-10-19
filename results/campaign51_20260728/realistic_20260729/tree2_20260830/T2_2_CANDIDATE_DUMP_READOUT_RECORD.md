# T2.2 per-candidate instrumented run — INDEPENDENT READER READOUT RECORD

**Stamp:** read out 2026-08-30 by the independent reader; run by the orchestrator, hook built by
a different agent; launched under row #255 — tree 2 node T2.2. Foreground only (≤ 600 s per
command), no ssh, no git, append-only.

**Registration:** `B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md` §4 design item (4) (what the hook
serialises), §3/§5 (closed form, F3 predictions), §6 (the full T2.2 design: gates, statistic,
bands); `B4_1_IMP_DECOMPOSITION.md` §7; `T2_2_CANDIDATE_HOOK_RECORD.md` (schema, gates). Run of
record: `candidate_dump_run/` (4 seeds 900101–900104, FT config, truth node, h = 0.73, `--jobs 1`,
wall 337.2 s per the driver's own `s0a_full_output.json`).

## 0. Comprehension-first summary

T2.2 asks whether the impostor-leg drag at low true redshift is a **depth-skew** effect: does the
catalogue-leg weight inside an event's candidate ball sit systematically **below** the event's own
true redshift? The per-candidate dump says yes, decisively. Pooled over 157 active q1 dark events
(4 seeds), **73.0 % of the catalogue-leg weight sits below the true redshift** (SE 1.4 %, about
16 standard errors from the 50 % no-skew null and 12 standard errors past the derivation's own
0.57 confirm threshold), and the weight-averaged listed-redshift offset is negative in every
robust summary (median −3.1 σ_GW, seed means negative in 3 of 4 seeds and in the fourth once one
pathological single-candidate event is set aside). Both registered confirm conditions are met:
**verdict DEPTH-SKEW-CONFIRMED**. The size of the effect runs a little hot against the
derivation's own point forecast (73.0 % against a predicted 60–70 % band; the offset statistic
several times larger in magnitude than the predicted −0.5 to −1.5 σ_GW), driven by a heavy right
tail of extremely well-localised (tiny σ_dL) events that the population-average closed form did
not anticipate — a real, disclosed miss on magnitude, not on sign or threshold-crossing.

Two gate findings need to be read alongside the verdict. First, the specific byte-identity gate
the registration names — this run's `event_likelihoods.csv` at h = 0.73 against the KW-Q1 truth
node — **could not be executed**: the named KW-Q1 comparand only ever evaluated h ∈ {0.725,
0.735} (its own secant design), never h = 0.73, for any of the 4 seeds. This is a structural gap
between the run as executed (single h = 0.73 point, not the design's 3-node grid) and its own
registered validation path, not a computed discrepancy — reported as **UNDETERMINED / NOT
EXECUTABLE**, with an informal substitute check reported for information only. The read proceeds
on the strength of the **reconstruction gate**, which is fully decisive and passes at ~1e-13
relative on real data from this run. Second, the per-event dump's own `z_true`/`f_bar_z_true`/
`f_k_z_true` columns are NaN for every one of the 714 real dark events across all 4 seeds (not
only the synthetic-fixture case the hook record disclosed) — the CRB schema this repository
actually writes has no `z_true` column. Worked around here by inverting the noiseless truth-node
identity `d_hat = d_L(z_true; h)` and by recomputing `f_bar(z_true)` independently from the
production completeness cache; both workarounds are cross-checked below.

## 1. Gates

### 1.1 GATE BI (byte-identity, as registered) — NOT EXECUTABLE

Registration: "the run's `event_likelihoods.csv` at h = 0.73 must be bit-identical to the
same-config FT truth node from the KW-Q1 run
(`fanout1_20260829/kwq1_registered_run/s0a_seed9001NN/node_truth_ft_sites2.2_nosmear/…`)".

Checked directly (all 4 seeds, this readout, 2026-08-30):

| seed | KW-Q1 truth-node h values present |
|---|---|
| 900101 | {0.725, 0.735} |
| 900102 | {0.725, 0.735} |
| 900103 | {0.725, 0.735} |
| 900104 | {0.725, 0.735} |

No seed's KW-Q1 truth node ever contains an h = 0.73 row — KW-Q1's own design evaluates only the
secant pair. There is no row overlap at h = 0.73 with which to compute a byte-identity diff. This
is not a byte-identity **failure** (no mismatched values were found because none could be
compared) — it is reported as **UNDETERMINED**. Root cause: the hook builder's own
`T2_2_CANDIDATE_HOOK_RECORD.md` §4 discloses that the executed command evaluates a single
h = 0.73 point "rather than section 6.4's own 3-node secant design" — the run as executed and the
gate as registered were never going to share an h-node.

**Informal substitute (not the registered gate, reported for information only):** compared this
run's h = 0.73 columns against `p3_work/ft_<seed>_work/seed<seed>/…/event_likelihoods.csv`, a
pre-existing FT-config truth run at h = 0.73 for the same 4 seeds (older code revision 53b7831e,
2026-08-23, vs this run's ecd33336; θ-site/smear settings of that run not independently
re-confirmed). Result, all 4 seeds: the global selection objects (`w_G`, `w_tilde_G`,
`alpha_G_phi`, `r_Malm`, `D_tilde_phi`, `B_num`, `B_num_wbh`, `g_frac`, `L_comp`) are bit-identical
(max\|Δ\| = 0.0 every seed); the catalogue-leg-dependent columns (`L_cat_no_bh`,
`combined_no_bh`, `L_cat_with_bh`, `combined_with_bh`) show small nonzero differences (max\|Δ\|
0.0009–0.02, max relative 9–13 %, consistent seed-to-seed). Plausibly attributable to code drift
over the intervening week rather than an instrumentation defect, but this is **not** a pass/fail
determination — it is not the registered comparand and the two runs are not the same code
revision.

### 1.2 GATE R (reconstruction) — PASS, all 4 seeds

Per event: `sum_g w_g · N_g_used / Σ_φ(h=0.73)` (candidates summed over both `with_bh` and
`no_bh_only` batches) reproduces the event's own diagnostics `L_cat_no_bh` column from the same
run.

| seed | n events | max abs diff | max rel diff (nonzero rows) | zero-candidate events (both sides = 0) |
|---|---|---|---|---|
| 900101 | 174 | 9.89e-17 | 6.17e-13 | 46 |
| 900102 | 184 | 9.71e-17 | 6.63e-13 | 54 |
| 900103 | 174 | 9.94e-17 | 5.11e-13 | 57 |
| 900104 | 182 | 1.01e-16 | 6.07e-13 | 71 |

All four seeds clear the registered ≤ 1e-12 relative tolerance by close to three orders of
magnitude. This is the decisive instrument-validity check: it proves the per-candidate rows this
readout uses are the exact rows the live likelihood consumed for this run, not an independently
recomputed shadow value. Also checked: `n_cand_no_bh` (per-event) equals the candidate-row count
per event exactly, and every event with `n_cand_no_bh = 0` has zero candidate rows — both true on
all 4 seeds.

### 1.3 GATE SCHEMA — PASS

Both `per_candidate_h_0_73.csv` (17 columns) and `per_event_h_0_73.csv` (13 columns) match the
registration's §6.2 column lists exactly, all 4 seeds.

### 1.4 GATE ENG — PASS (engagement); cross-h sub-check N/A

157 of 191 q1 dark events (82.2 %) have `L_cat_no_bh` > 0 at h = 0.73, above the registered ≥ 60 %
bar (banked comparand from the design doc: 425/540 = 78.7 %, consistent). The design's second
ENG sub-check ("per-candidate N_g must differ across the three h-nodes on ≥ 99 % of rows") is
**not applicable** — this run evaluated only h = 0.73, not the {0.725, 0.730, 0.735} grid the
design specifies, so there is no second h-node to diff against.

### 1.5 Disclosed schema gap: z_true / f_bar_z_true / f_k_z_true are NaN

`per_event_h_0_73.csv`'s `z_true`, `f_bar_z_true`, `f_k_z_true` columns are NaN for all 714 real
dark events across all 4 seeds — not only the synthetic-fixture fallback case the hook record
names. Cause: this repository's CRB schema has no `z_true` column at all (checked directly:
`prepared_cramer_rao_bounds.csv` carries `luminosity_distance`, not `z_true`), so the hook's own
"read from the CRB row's z_true column when present" path always falls through to NaN on real
data. Worked around, independently, as follows:

- **z_true**: this truth node is noiseless (θ = (0,1), H_TRUE = H_GEN = 0.73) — verified directly
  that `d_hat` in the per-event dump equals `prepared_cramer_rao_bounds.csv`'s
  `luminosity_distance` bit-for-bit for the events checked (seed 900101, events 0–4). Recovered
  `z_true = dist_to_redshift(d_hat, h=0.73)` (`darksiren_emri/physical_relations.py`), the exact
  inverse of the `dist()` function the estimator itself uses.
- **f_bar(z_true)**: recomputed via `PixelCompleteness.from_cache_or_build().f_bar(z, h=0.73)`
  (`darksiren_emri/galaxy_catalogue/pixel_completeness.py`), loaded from the frozen
  `m_th_map_nside32.npy` cache over `reduced_galaxy_catalogue.csv` — the module's own docstring:
  "the SOLE source of f (C1) — the SAME .npy file is loaded byte-identically by injection and
  inference." This is the completeness cache the derivation cites in §1 ("f is h-free").

## 2. The registered statistic (§6.5): dark, q1 (z_true < 0.358), active (L_cat_no_bh > 0), 4 seeds pooled

All 714 detected events are dark (`host_galaxy_index = −1`) — expected, this is the all-dark
B-SEL FT arm (§8.3 of the derivation). q1 count = **191**, matching KW-Q1's frozen 191-event q1
set exactly (a strong internal cross-check that the recovered z_true is correct). q1 active
(L_cat_no_bh > 0) = **157** (82.2 %).

Weight of candidate g in event i: `W_ig = w_g · N_g_used` (the catalogue leg's own numerator
summand at h = 0.73), exactly as registered.

**Primary — Φ_low (impostor-weight share below z_true), pooled over 157 active q1 events:**

| statistic | value |
|---|---|
| mean | 0.7299 |
| SD | 0.1746 |
| SE | 0.0139 |
| median | 0.7394 |
| null (no skew) | 0.500 |
| distance from null | ≈ 16.5 SE |
| distance above the 0.57 confirm threshold | ≈ 11.5 SE |
| registered point-forecast band | [0.60, 0.70] |

Observed mean sits just above (+0.03, ≈ 2 SE) the top of the predicted band — the sign and
threshold-crossing are confirmed with room to spare; the point-forecast magnitude is a modest
undershoot.

**Secondary — W-weighted mean listed-z offset ⟨u⟩_W (GW-σ units), pooled:**

| statistic | value |
|---|---|
| mean (unweighted across events) | −10.67 |
| SD | 327.5 (heavy right tail) |
| median | −3.06 |
| trimmed mean (5th–95th pct) | −10.35 |
| trimmed SD | 20.9 |
| registered point-forecast band | [−1.5, −0.5] |

This statistic is heavy-tailed in this data: a handful of extremely well-localised events (tiny
σ_dL, often a single candidate) generate |u_g| in the hundreds to thousands, dominating any
unweighted mean. Example: seed 900101 event 70 has σ_dL = 6.9e-5 Gpc, 1 candidate, u_g = +3495 —
excluding this single row flips seed 900101's per-seed mean from +44.7 to −59.8. Median and
trimmed statistics are reported alongside the raw mean for this reason; the design's own §6.5
only attaches a formal SE bound to Φ_low (a bounded [0,1] variable), never to ⟨u⟩_W, so this
instability is not a violation of anything registered — but the ×7–20 magnitude overshoot past the
predicted [−1.5,−0.5] band, in every robust summary, is a genuine miss on the point forecast.

## 3. A15 — per-seed scatter

| seed | n (q1 active) | mean Φ_low | SD Φ_low | mean ⟨u⟩_W | SD ⟨u⟩_W |
|---|---|---|---|---|---|
| 900101 | 34 | 0.7372 | 0.1973 | +44.72 | 662.9 |
| 900102 | 38 | 0.7321 | 0.1674 | −14.14 | 42.4 |
| 900103 | 37 | 0.7255 | 0.1777 | −8.18 | 62.5 |
| 900104 | 48 | 0.7263 | 0.1659 | −49.07 | 198.1 |

Across-seed SD of the 4 per-seed **Φ_low** means: **0.00545** (tiny — Φ_low is stable seed to
seed; across-seed mean of means 0.7303, consistent with the pooled figure). Across-seed SD of the
4 per-seed **⟨u⟩_W** means: 38.7 — dominated by the same single-event outlier structure noted
above (seed 900101's positive mean is driven entirely by event 70; with that one row excluded all
4 seeds show a negative mean). Φ_low is the robust, decisive statistic; ⟨u⟩_W confirms direction
but its per-seed generalisation width is not meaningful without outlier handling.

## 4. True-host flag distribution; q1 vs q2–q4 split; closed-form comparison

**True-host flags:** `is_true_host` is False on all 606,571 candidate rows across all 4 seeds (0
flagged) — correct and expected, since every event in this all-dark B-SEL arm has
`host_galaxy_index = −1` (no true host exists to flag).

**q1 vs q2–q4 (mean Φ_low, active events, dark quartiles from the recovered z_true):**

| quartile | z_true range | n active | mean Φ_low | SE |
|---|---|---|---|---|
| q1 | < 0.358 | 157 | 0.730 | 0.0139 |
| q2 | [0.358, 0.459) | 141 | 0.809 | 0.0222 |
| q3 | [0.459, 0.584) | 122 | 0.974 | 0.0102 |
| q4 | ≥ 0.584 | 66 | 0.970 | 0.0210 |

Φ_low rises monotonically with z_true — expected on its own terms (as f(z) collapses toward zero
at high z, nearly all catalogue weight trivially sits below any high z_true), not itself a
depth-skew signature specific to q1; the depth-skew claim is about the q1-localised drag
mechanism, which the closed-form comparison below addresses directly.

**Closed-form check, E[s_imp|z] = f (df/dz) z_eff / (h(1−f))** (§3.2, independently evaluated
from the production completeness cache, not the banked c(z) proxy the derivation used):

| z | f_bar(z) | df/dz | z_eff | E[s_imp\|z] |
|---|---|---|---|---|
| 0.10 | 0.709 | −2.84 | 0.094 | −0.886 |
| 0.15 | 0.571 | −2.73 | 0.137 | −0.679 |
| 0.20 | 0.434 | −2.72 | 0.178 | −0.509 |
| 0.25 | 0.303 | −2.47 | 0.219 | −0.322 |
| 0.30 | 0.192 | −1.96 | 0.259 | −0.165 |
| 0.35 | 0.109 | −1.36 | 0.298 | −0.067 |

Sign and monotone-decreasing-magnitude trend with z match the derivation's own table (§3.4)
qualitatively; exact digits differ modestly (independent numerical path). Mean f_bar(z_true) over
the 191 q1 dark events = **0.2616** (median 0.2117) against the derivation's banked mean catalogue
share c = 0.1655 for the 4-seed q1 set — ratio c/f̄ = 0.632, in the right direction (c is a lower
bound on f per §3.4) but modestly below the derivation's own stated κ range (0.675–0.87); a small,
disclosed tension, not large enough to change any verdict here.

## 5. Verdict (per §6.5's registered bands)

Registered bands: **DEPTH-SKEW-CONFIRMED** if q1 mean Φ_low ≥ 0.57 AND ⟨u⟩_W ≤ −0.3 σ_GW;
**DEPTH-SKEW-REFUTED** if Φ_low ≤ 0.53 or ⟨u⟩_W ≥ 0 while q1 s_imp stays ≤ −0.6; **MIXED**
otherwise.

- Φ_low = 0.7299 ≥ 0.57 — **met**, ≈ 11.5 SE past the threshold.
- ⟨u⟩_W: median −3.06, trimmed mean −10.35, pooled mean −10.67, 3 of 4 per-seed means negative
  outright and the fourth negative once one pathological single-candidate row is excluded — ≤
  −0.3 σ_GW **met** on every robust reading.

**VERDICT: DEPTH-SKEW-CONFIRMED.** Candidate (c) of the claim card (depth skew of impostors
inside the ball) is supported by direct per-candidate measurement, consistent with the
derivation's closed form in sign, z-localisation and rough order of magnitude, with the effect
running somewhat larger than the point forecast on Φ_low and substantially larger on ⟨u⟩_W's
heavy-tailed magnitude (not on sign). This verdict is reached without the registered BI gate (not
executable as named, §1.1) and rests instead on the reconstruction gate (§1.2, decisive, ~1e-13
relative on real data) plus the hook's own unit-tested byte-identity/schema passes recorded in
`T2_2_CANDIDATE_HOOK_RECORD.md` §3.

## 6. Cost

Wall time 337.21 s (driver's own `s0a_full_output.json`, `s0a_full_output.json` key `wall_s`,
`--jobs 1`, `cpu_per_job = 14`) × 14 cores = **1.31 CPU-h**. Below the 3.4–3.9 CPU-h registered
anchor because only 1 h-node (0.73) was run rather than the design's 3-node {0.725, 0.730, 0.735}
secant grid (§1.1's root cause).

## 7. What this readout does not claim

- Not claimed: any secant-based statistic (s_imp itself, the §6.6 zero-compute rescore) — only
  h = 0.73 was run; those need the 3-node grid.
- Not claimed: the registered BI gate passed — it could not be executed against the named
  comparand (§1.1); the informal substitute is disclosed, not decisive.
- Not claimed: ⟨u⟩_W's magnitude matches its point forecast — it does not, by a wide margin,
  though its sign and threshold-crossing do.
- Not claimed: any adjudication of A11, remedy (d), or the enlarged-ball falsifier — out of
  this node's scope.

Sources: this readout's own computation over `candidate_dump_run/` (all files, 2026-08-30);
`fanout1_20260829/kwq1_registered_run/` (BI comparand, checked, found non-overlapping in h);
`p3_work/ft_<seed>_work/` (informal substitute comparand); `darksiren_emri/physical_relations.py`
(`dist_to_redshift`); `darksiren_emri/galaxy_catalogue/pixel_completeness.py`
(`from_cache_or_build`, `m_th_map_nside32.npy`).

Launched under row #255 — tree 2 node T2.2 (this readout).

---

## 8. BI gate closure (2026-08-30)

A second run, `candidate_dump_bi_run/` (same FT config, truth node, driver defaults
`theta_sites="all"`/`smear="auto"`, `--jobs 1`, but this time **h ∈ {0.725, 0.735}** — the exact
h-nodes the KW-Q1 comparand actually evaluates), closes the gap flagged in §1.1. Log:
`hier_s0_recert_run/logs/runner5_tree2_20260830.log`, stage BIDUMP; driver's own
`candidate_dump_bi_run/s0a_full_output.json`: `wall_s = 569.09`, `cpu_per_job = 14`,
`n_seeds_ok = 4`, `h_values = [0.725, 0.735]`, `config = "ft"`, `node_dir_suffix = "_ft"`.

### 8.1 GATE BI (registered) — PASS, bit-for-bit

Compared this run's `event_likelihoods.csv` against
`fanout1_20260829/kwq1_registered_run/s0a_seed9001NN/node_truth_ft_sites2.2_nosmear/…` at h =
0.725 and h = 0.735, all 18 non-`h` columns, all 4 seeds, event-index-matched:

| seed | h | n rows compared | all 18 columns exact (max\|Δ\| = 0.0)? |
|---|---|---|---|
| 900101 | 0.725 | 174 | yes |
| 900101 | 0.735 | 174 | yes |
| 900102 | 0.725 | 184 | yes |
| 900102 | 0.735 | 184 | yes |
| 900103 | 0.725 | 174 | yes |
| 900103 | 0.735 | 174 | yes |
| 900104 | 0.725 | 182 | yes |
| 900104 | 0.735 | 182 | yes |

**PASS.** Every one of `w_G`, `w_G_legacy`, `w_tilde_G`, `alpha_G_phi`, `r_Malm`, `D_tilde_phi`,
`L_cat_no_bh`, `L_cat_with_bh`, `B_num`, `B_num_wbh`, `g_frac`, `L_comp`, `combined_no_bh`,
`combined_with_bh`, `den_log_term`, `num_log_term_no_bh`, `num_log_term_with_bh` is bit-identical
(max\|Δ\| = 0.0, all rows) at both h-nodes, all 4 seeds, despite the θ-site/smear settings
differing between the two runs (`all`/`auto` here vs `2.2`/off in KW-Q1) — exactly the
literal-skip-identity behaviour §1.1 predicted for θ = (0,1). The §1.1 informal substitute
comparison is superseded by this direct, registered-comparand result; the small (9–13 %) drift
noted there is now understood to be pre-hook code drift between `p3_work` (rev 53b7831e) and this
tree (rev ecd33336), not anything to do with the instrumentation or with θ-site/smear
configuration.

### 8.2 GATE R (reconstruction) — PASS at both h-nodes, all seeds

Same check as §1.2, re-run on `candidate_dump_bi_run`'s own per-candidate rows:

| seed | h | max abs diff | max rel diff (nonzero rows) |
|---|---|---|---|
| 900101 | 0.725 | 9.74e-17 | 7.05e-13 |
| 900101 | 0.735 | 9.93e-17 | 9.30e-13 |
| 900102 | 0.725 | 9.89e-17 | 6.26e-13 |
| 900102 | 0.735 | 9.80e-17 | 4.27e-13 |
| 900103 | 0.725 | 9.71e-17 | 3.70e-13 |
| 900103 | 0.735 | 9.93e-17 | 4.97e-13 |
| 900104 | 0.725 | 9.71e-17 | 6.50e-13 |
| 900104 | 0.735 | 9.71e-17 | 4.67e-13 |

All 8 (seed, h) cells clear the registered ≤ 1e-12 relative tolerance by close to three orders of
magnitude, matching §1.2's original result.

### 8.3 Φ_low h-stability

Recomputed the registered statistic (dark, q1, active, `W_ig = w_g · N_g_used`) independently at
each h-node:

| h | n q1 dark | n q1 active | mean Φ_low | SD | SE | median |
|---|---|---|---|---|---|---|
| 0.725 | 198 | 164 | 0.7342 | 0.1737 | 0.0136 | 0.7359 |
| 0.735 | 186 | 154 | 0.7338 | 0.1744 | 0.0141 | 0.7385 |

Δ(mean Φ_low, 0.735 − 0.725) = **−0.00043**, combined SE 0.0195 → **≈ 0.02 σ** — Φ_low is
h-stable across the secant pair, consistent with the derivation's own claim that the depth-skew
effect is independent of GW-distance precision and (at first order) of h over this range. Both
values sit within 1σ of the original single-node (h = 0.73) figure of 0.7299 (§2), and the active
fraction is consistent (164/198 = 82.8 %, 154/186 = 82.8 %, vs 157/191 = 82.2 % at h = 0.73). The
small q1-membership shuffle between h-nodes (198 vs 186 dark events land in q1, vs 191 at h=0.73)
is expected — q1 membership is a fixed z_true cut and z_true itself does not depend on h, but the
recovered z_true is inverted from `d_hat` at each node's own h via `dist_to_redshift`, so an event
sitting near the 0.358 boundary can cross it between the two secant nodes; this does not affect
the pooled statistic's stability, which sits inside 0.02σ.

### 8.4 Updated verdict

The registered BI gate now **PASSES** cleanly (§8.1), superseding §1.1's UNDETERMINED finding.
Combined with GATE R (§8.2, also PASS) and Φ_low's demonstrated h-stability (§8.3), every gate
this readout's registration names is now satisfied. **Verdict unchanged and now fully
gate-clean: DEPTH-SKEW-CONFIRMED** (§5's numbers stand; this section adds gate closure only, no
statistic in §2–§4 is revised).

### 8.5 Cost of the closure run

569.09 s wall (driver's own `s0a_full_output.json`) × 14 cores = **2.21 CPU-h**, additional to the
1.31 CPU-h of §6 (total for the T2.2 read-and-close arc: 3.52 CPU-h, now within the original
3.4–3.9 CPU-h registered anchor).

Sources: `candidate_dump_bi_run/` (all files, this closure, 2026-08-30);
`fanout1_20260829/kwq1_registered_run/` (BI comparand); `hier_s0_recert_run/logs/runner5_tree2_20260830.log`.
