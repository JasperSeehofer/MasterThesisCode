# rd-parity-decomp — READ RECORD (Branch J, execution item 2)

Date: 2026-09-03. VERDICT-FREE per row #325. Chair rules; this file reports numbers, three-valued
outcomes, and a labelled RECOMMENDATION only.

Question (q-parity-growth): is the row #273 T1.3-zwin GATE PARITY residual exactly the z-window's
registered added-candidate term (`PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md` lines 401–403, R8), or
an uncaptured mechanism?

## 0. Inputs — existence contract (three-valued)

| input | status | path |
|---|---|---|
| zwin-truth (T1.3-zwin) per-event CSV, 4 seeds | PRESENT | `tree2_20260830/hier_s0_zwin_run/s0a_seed{900101,900102,900103,900104}/node_truth_sites2.2_nosmear_divisor_zwin_zk4/simulations/diagnostics/event_likelihoods.csv` |
| T1.2-truth (pre-window) per-event CSV, 4 seeds | PRESENT | `tree2_20260830/hier_s0_recert_run/s0a_seed{...}/node_truth_sites2.2_nosmear_divisor/simulations/diagnostics/event_likelihoods.csv` |
| banked comparand bc, 4 seeds | PRESENT | `p3_b0_work/bc_{900101,...}_work/seed{...}/simulations/diagnostics/event_likelihoods.csv` (all h; filtered to h=0.73 here) |
| `gate_parity` scoring function (methodology reference) | PRESENT | `fanout1_20260829/hier_s0_driver.py:1994` (`gate_parity`), `:963` (`read_event_ln_l`: `ln_L = log(combined_*)`) |
| T2.2 candidate dump, correct venue (`sites2.2_nosmear_divisor[_zwin_zk4]`), per-candidate breakdown | **ABSENT** — see §3 | only `candidate_dump_run/s0a_seed{...}/node_truth_ft/...` exists, config=`ft` (not `sites2.2_nosmear_divisor`); no `results/.../candidate_dump/` or per-candidate CSV exists under either truth-node directory used here |

All four CSV triples exist locally, load cleanly, and merge on `event_idx` with **zero** left-only or
right-only rows at any seed (zwin and T1.2 cover the identical event set at every seed — no truncation
mismatch). The correct-venue per-candidate dump that the proposal names as the primary source for "the
added-candidate set" does **not exist** — this is disclosed and handled per §3, not substituted.

## 1. Methodology

Reproduced `hier_s0_driver.py`'s own `read_event_ln_l`/`gate_parity` transform exactly:
`ln_L_no_bh = log(combined_no_bh)`, `ln_L_with_bh = log(combined_with_bh)` (NaN where `combined_* <= 0`,
none occurred). `Δln L = ln_L(zwin-truth) − ln_L(T1.2-truth)`, per event, both channels, all 4 seeds.
Full script: `parity_decomp.py` in this node's exec directory.

**Re-derivation check (self-check before classification):** recomputed the T1.2-truth-vs-bc and
zwin-truth-vs-bc `max_rel_diff`/`max_abs_diff` (no_bh) with the identical driver methodology, from the
raw CSVs, independent of the banked `s0a_score_output.json`. Result: **exact match** to row #273 / the
proposal's Branch J table for all 4 seeds, both the zwin numbers and the "pre-window baseline" numbers:

| seed | recomputed T1.2-vs-bc max_rel (no_bh) — E19 floor | recomputed zwin-vs-bc max_rel (no_bh) | proposal-cited zwin (no_bh) |
|---|---|---|---|
| 900101 | 4.8809e-05 | 0.44682 | 0.447 |
| 900102 | 4.2653e-05 | 0.03882 | 0.0388 |
| 900103 | 3.1893e-05 | 0.21601 | 0.216 |
| 900104 | 2.0344e-05 | 0.04348 | 0.0435 |

(with_bh zwin-vs-bc: 0.3448 / 0.0919 / 0.1884 / 0.0743 — matches 0.345/0.0919/0.188/0.0743 in the
proposal table.) The banked `s0a_score_output.json` numbers are confirmed independently reproducible
from the raw per-event CSVs, not an artifact of the scoring JSON. E19 floor = max over seeds of the
recomputed T1.2-vs-bc no_bh `max_rel_diff` = **4.8809e-05** (900101).

Candidate-added classification: an event is **no-added-candidate** iff `L_cat_no_bh` AND
`L_cat_with_bh` are bit-identical (`diff == 0.0`, not just within-floor) between the zwin-truth and
T1.2-truth rows; otherwise **candidate-added**. (`L_cat_*` is the raw catalogue-likelihood sum the gate
doc's lines 401–403 describe as differing "ONLY through the added candidates" — this is the direct,
zero-compute operationalization of that claim.)

Two supporting structural checks, both zero-compute and both bearing directly on "uncaptured mechanism
vs added-candidate term only":
- **Monotonicity**: for every candidate-added event, is `L_cat_no_bh`/`L_cat_with_bh` strictly ≥ the
  T1.2 value (window widening k=1→4 can only ADD rows to a positive-term sum, never remove or reweight
  existing ones — a decrease would falsify the "added candidates only" mechanism outright)?
- **Aux-column identity**: are the non-candidate intermediate columns (`w_G`, `w_G_legacy`,
  `w_tilde_G`, `alpha_G_phi`, `r_Malm`, `D_tilde_phi`, `g_frac`, `L_comp`, `B_num`, `B_num_wbh`,
  `den_log_term`) bit-identical zwin-truth vs T1.2-truth for every event? If yes, literally nothing in
  the per-event likelihood pipeline changed except the candidate-derived terms — a direct, per-event
  test of "ONLY through the added candidates."

## 2. Results

| seed | n events | n no-added | n added | Δln L no_bh, no-added bucket (max\|Δ\|) | Δln L no_bh, added bucket (range) | # added events where L_cat DECREASED (no_bh / with_bh) | aux columns identical (all events)? |
|---|---|---|---|---|---|---|---|
| 900101 | 106 | 3 | 103 | **0.0** (exact) | 6.05e-09 .. 5.2324 | 0 / 0 | yes (0 events with any aux-col drift) |
| 900102 | 120 | 5 | 115 | **0.0** (exact) | 2.49e-06 .. 0.2257 | 0 / 0 | yes |
| 900103 | 105 | 5 | 100 | **0.0** (exact) | 6.51e-07 .. 1.3556 | 0 / 0 | yes |
| 900104 | 130 | 6 | 124 | **0.0** (exact) | 2.00e-10 .. 0.3295 | 0 / 0 | yes |
| **total** | **461** | **19 (4.1%)** | **442 (95.9%)** | **0.0 across all 19** | — | **0 / 0 across all 442** | **yes, all 461** |

The `max\|Δ|` for the no-added bucket is exactly `0.0` in every seed — not merely "within the E19
floor" but bit-identical, tighter than the 4.88e-05 floor by construction (0 floor units). Spot-checked
raw values directly (not just the diff) for 2 events per seed, e.g. seed 900101 event 11:
`combined_no_bh` zwin=`0.0244465939576578` t12=`0.0244465939576578`, `equal=True` — confirmed bit-exact,
not a coincidental rounding match.

For the added-candidate bucket: `L_cat_no_bh` and `L_cat_with_bh` moved in the **positive direction in
every single one of the 442 candidate-added event-instances across all 4 seeds** (0 decreases, both
channels) — exactly the structural signature a window-widening (rows-only-added-to-a-positive-sum)
mechanism must produce, and inconsistent with a defect that would perturb existing candidate weights
(which could push the sum either direction). All 11 auxiliary (non-candidate) columns are bit-identical
between zwin-truth and T1.2-truth for all 461 events in both buckets — nothing in the pipeline changed
except the candidate-derived terms (`L_cat_no_bh`, `L_cat_with_bh`, and their downstream
`combined_*`/`num_log_term_*`).

Max |Δln L_no_bh| among candidate-added events: **5.2324 nats** (seed 900101), i.e. **≈1.07e5 floor
units** relative to the 4.88e-05-rel E19 floor (in the sense that this event alone accounts for a
`max_rel_diff` that, propagated through `gate_parity`, reproduces the row #273 0.447 residual for that
seed — consistent with a small number of high-leverage events, not a diffuse population-wide drift:
19/461 events carry zero delta, and within the 442-event added bucket the range spans 9 orders of
magnitude, i.e. most added candidates contribute a negligible term and a minority dominate the
max-diff statistic).

Disclosed sub-question (not part of q-parity-growth's kill criterion, reported only): the **with_bh**
channel already carries a **5.9%–12.0%** `max_rel_diff` at the **pre-window** T1.2-vs-bc comparison
(recomputed: 900101 5.94%, 900102 8.92%, 900103 11.99%, 900104 3.43%) — this pre-dates
`theta_zwindow` entirely and is not covered by the 5.718e-4 (no_bh) E19 wording. It is orthogonal to
this node's kill criterion (scoped to the no_bh primary channel) and is flagged for
`d-photoz-leverage`/wave-3 attention, not adjudicated here.

## 3. The one gap: exact per-candidate sum equality

The kill criterion's stronger clause — "[Δ] equals the added-candidate sum where candidates were
added" — asks for numerical equality against an independently-computed added-candidate contribution,
not just the two necessary-condition structural checks above. That requires a per-candidate breakdown
(`catalog_index`, `z_g`, per-candidate log-term) at the **correct venue**
(`sites2.2_nosmear_divisor` / `..._zwin_zk4`, `theta_sites=2.2`, smear=`nosmear`, divisor on).

The only local per-candidate dump is `candidate_dump_run` (`node_truth_ft`, config=`ft`,
`theta_sites=all`, smear=`auto`) — a different venue. Verified this mismatch directly rather than
assuming it from the tag: `candidate_dump_run`'s **own** `gate_parity` vs the same bc comparand gives
`max_rel_diff` (no_bh) of **1.00 / 3.60 / 0.69 / 1.40** across the 4 seeds — order-unity, nothing like
the target venue's 0.02–0.45 — so the `ft` dump is evaluating a materially different likelihood
altogether and cannot stand in for "the added-candidate set" of the venue under test. Per the node's own
edge condition ("used ONLY if its venue tag is verified to match... else reconstructed"), it is
excluded. No candidate-level data exists under either truth-node directory for the correct venue
(`selection_tables_h_0_73.json` there carries only 9 scalar config fields, no per-candidate rows) and
reconstructing it would require an `evaluate()` call — explicitly out of scope for this zero-compute
read (that is what the conditional `m-parity-401grid` node, A-J2, already pre-authorizes if this lands
on (ii)).

**This one sub-check is therefore NOT-EVALUABLE from local zero-compute data as literal numerical
equality.** The two structural necessary conditions that a true "added-candidate-only, monotone,
additive" mechanism must satisfy (monotonicity: 0/442 violations; aux-column invariance: 0/461
violations) are both satisfied with no exceptions, which is the strongest evidence obtainable at this
node's cost budget; it stops short of a literal sum-reconciliation.

## 4. Counts summary (for the record)

- Events with NO added candidate: 19 / 461 (4.1%), all 4 seeds, Δln L = 0.0 exactly (both channels) — 0
  floor units, 0 anomalies.
- Events WITH added candidates: 442 / 461 (95.9%), Δln L strictly non-negative in both channels for
  100% of them (0 decreases), aux columns bit-identical for 100% of them.
- Max unexplained |Δ| in the no-added bucket: **0 floor units** (exact).
- Max |Δ| in the added bucket: 5.2324 nats (no_bh, seed 900101) — this is a *reported* magnitude, not an
  "unexplained" one under the classification test actually run (monotonicity + aux-invariance both
  pass); whether it also equals the literal added-candidate sum is the §3 gap, not evaluated.

## 5. RECOMMENDATION (labelled — not a ruling)

**(i) EXPLAINED-BY-DESIGN**, on the following basis: every event with no added candidate shows exactly
zero `Δln L` (both channels, all 4 seeds, bit-exact — tighter than the E19 floor by construction); every
event with an added candidate shows a `Δln L` that is monotonically non-negative in both channels with
zero exceptions across 442 event-instances; and all 11 non-candidate intermediate columns are
bit-identical for all 461 events. These three independent, zero-compute checks are exactly the
observable signature the gate doc's lines 401–403 claim predicts and are jointly inconsistent with an
uncaptured mechanism reaching any other term in the likelihood pipeline. The row #273 "consistent in
kind with E19" wording is corroborated as needing correction: E19 (max no_bh rel ≈ 4.88e-5) is the
right floor for the *zero-candidate* subpopulation and is reproduced exactly; it is the wrong mechanism
label for the 44.7%-class residual, which sits entirely in the 95.9% of events that gained catalogue
candidates under `z_window_k=4`.

Recommend flagging, not blocking, one open item: the literal per-candidate numerical
sum-reconciliation (§3) was not possible from data on disk and would need either (a) `m-parity-401grid`
run at the correct venue (already conditionally pre-authorized, A-J2, cap 15 CPU-h — but that node's own
trigger condition is landing on (ii), which this record does not), or (b) a fresh, correctly-tagged
`evaluate()`-based per-candidate dump at `sites2.2_nosmear_divisor`/`zwin_zk4`, k=1 vs k=4, which is a
compute node outside this read's scope. This is disclosed as a residual verification gap, not treated as
grounds for (ii); the two structural checks that ARE zero-compute-reachable were both run to completion
with zero exceptions.

The disclosed with_bh pre-window residual (5.9%–12.0%, §2) is flagged for `d-photoz-leverage` attention
as a separate, unadjudicated item — it is not part of q-parity-growth's kill criterion and this record
takes no position on it.

## 6. Sources

- `results/campaign51_20260728/realistic_20260729/tree2_20260830/hier_s0_zwin_run/s0a_seed{900101..900104}/node_truth_sites2.2_nosmear_divisor_zwin_zk4/simulations/diagnostics/event_likelihoods.csv`
- `results/campaign51_20260728/realistic_20260729/tree2_20260830/hier_s0_recert_run/s0a_seed{900101..900104}/node_truth_sites2.2_nosmear_divisor/simulations/diagnostics/event_likelihoods.csv`
- `results/campaign51_20260728/realistic_20260729/p3_b0_work/bc_{900101..900104}_work/seed{900101..900104}/simulations/diagnostics/event_likelihoods.csv`
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py` (`gate_parity` L1994, `read_event_ln_l` L963, `PARITY_FALLBACK_RTOL` L293)
- `results/campaign51_20260728/realistic_20260729/tree2_20260830/hier_s0_zwin_run/s0a_score_output.json`, `tree2_20260830/hier_s0_recert_run/s0a_score_output.json` (banked scoring JSONs, cross-checked not relied on)
- `results/campaign51_20260728/realistic_20260729/tree2_20260830/candidate_dump_run/s0a_seed{900101..900104}/node_truth_ft/simulations/diagnostics/event_likelihoods.csv` and `s0a_full_output.json` (venue-mismatch evidence, §3)
- `results/campaign51_20260728/realistic_20260729/tree2_20260830/PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md` (lines 401–403, R8 — cited, not re-quoted in full here)
- Analysis script: `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/rd-parity-decomp/parity_decomp.py`

## ERRATUM (ratified 2026-09-04, R11)

"Consistent in kind with E19" -- E19 is the floor for the zero-candidate subpopulation
(Δln L = 0 exactly on 19/461 events); the 44.7 %-class residual sits entirely in the 442 events
that gained candidates under z_window_k = 4 (EXPLAINED-BY-DESIGN, row #342).

(Note, chair 2026-09-04: the "ERRATUM (ratified 2026-09-04, R11)" section appended above belongs to the row #273 record `tree2_20260830/T1_3_ZWINDOW_P1_READOUT_RECORD.md`, where it has also been appended; it is kept here as a cross-reference.)
