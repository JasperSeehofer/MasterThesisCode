# READ_RECORD_m1 — sealed-mock stage (m1): HEAD re-score of the UNSEALED 0.67 pool, iiib venue

Chair read, 2026-09-04 ~12:30 CEST (local clock; the ledger rows #367–#386 carry mis-labelled
afternoon CEST stamps — see the chronicler filing W-CTX-03). Job of record **6794421** (array
0–40, `cpu_il`, 16 cpus/task), submitted under the batch-2 grant (row #373, content-guard
correction). Registration: `D_SEALED_REGISTER_DOSSIER.md` §2 (statistic, cells, gates) — nothing
below departs from it; every number is chair re-derived from the retrieved CSV with the T0 scorer
(script `t0_read.py`, validated this session against the head-rebaseline iiib anchors: it
reproduces mean 0.665854 / σ 0.018475 (2D) and 0.666987 / 0.017526 (1D) exactly).

**Status of the cell call: NONE MADE BY THE CHAIR.** The registered cell table gives two
answers on this posterior (§3); the disposition is a fresh [RULE] for the author (docket R25).

## 1. Inputs and retrieval (ops record: `../batch2_cluster_ops/OPS_RECORD_m1_retrieval.md`)

| item | value |
|---|---|
| run dir (cluster) | `$WS/graph1_sealed_m1_iiib_20260904` |
| read input | `simulations/diagnostics/event_likelihoods.csv`, 55,063 rows = 41 h × **1343** events, md5 `ed3d5479c1dad0a27b8060106836206f` |
| local copy | `m1_iiib_retrieved/` (85 files, `MANIFEST.md5` list-md5 `f88a444ffb179add23954349b59dbb1c`, remote/local md5 verified) |
| not retrieved | `posteriors_with_bh_mass/` (41 × 86 MB = 3.3 GB) — not a read input; left on the cluster |
| cost | Σ CPUTimeRAW = 205,152 s = **57.63 core-h** (reservation basis); per-task wall 4.80 / 5.20 / 5.88 min (min/median/max). Cap 75 → within cap; dossier §1 item 6 condition (≤ 60) **satisfied** — joint_r1 sibling NOT submitted (launch is a fresh [DO], docket R26) |

## 2. Gates at read time (dossier §2 "(m1) subset")

| gate | result | evidence |
|---|---|---|
| G-1 pins | GREEN **against the CORRECTED pin** (registered text was RED: dossier §2 registered `injection_pool_depth15_50k`, count 500; the 0.67 run's `injections` resolves to `injection_pool_mix200k_20260728`, 707 files; the chair overrode this as a factual error in the registration — PIN CORRECTION section — before submission. Put to the author as **R28**). Cross-pool check (chair, this session, one read-only ssh line): the 0.73 re-baseline run `run_20260729_seed61000` and the 0.67 run `run_20260729_seed64000_h0p67` both link the SAME pool, per-file manifest list-md5 `75f4030d5d3b0405fd948049bef5767e` on both — the anchor comparison is same-pool | task logs: CRB67 md5 `8e9253fef42f574c569a04a3e19299ab` (= §5 slot), rows 1345 (= §5 slot; g-population disclosure), catalogue md5 `c52c13b5cab61f6b3f04bbe202550969`, pool content list-md5 `75f4030d5d3b0405fd948049bef5767e` (707 files; PIN CORRECTION) — all 41 tasks printed the pin-OK lines |
| G-2 commit | GREEN (ancestor form) | `GIT_COMMIT_AT_RUN.txt` = `d9e50179`; tasks 0–4 (h = 0.600–0.640, the low-h tail carrying the rail mass) ran at `8f933e7b`, tasks 5–40 at `d9e50179` (HEAD moved mid-array, 09:40→10:2x). Both are descendants of `081b1f28` and both are `darksiren_emri/` diff-quiet against it (chair re-checked locally: `git diff --quiet 081b1f28 <sha> -- darksiren_emri/` passes for both, and HEAD is diff-quiet vs `d9e50179`). Caveat (verifier): the sbatch freeze compares COMMITS; the cluster tree was dirty (665/702 porcelain lines) so an uncommitted edit under `darksiren_emri/` on the cluster would pass undetected — not checkable locally; disclosed |
| g-population | DISCLOSED | CRB 1345 rows (1343 + 2 empty events, `n_events_empty=2`); the scorer sees exactly 1343 events |
| resolved-flags equality (13 tokens vs `REGISTERED_RESOLVED_FLAGS`) | GREEN 13/13 | 11 tokens read from `run_metadata_{0..40}.json::cli_args`, identical on all 41 files; `catalogue_numerator_survival` = "phi" from the `[PHYSICS] catalogue_numerator_survival="phi" ACTIVE` line present in all 41 completed-task `.err` logs, 0 `COUNTERFACTUAL` warnings; `mass_filter_sigma` = "symmetric" is the constructor default at the frozen code (never CLI-exposed, no override in `cli_args`). `catalogue_leg_1d_mass_aware` left at `auto` → logs say ACTIVE (post-flip default), as registered |
| g-censoring | **RED → BOUND** (per the dossier: "a red here is a BOUND, not NO-READ") | MAP interior (0.630) in both channels, but edge-node mass at h = 0.600 is 4.39e-2 (2D) / 2.59e-2 (1D) ≫ 1e-3. Upper edge 1e-17 / 1e-16 |
| g-znorm | NOT EVALUABLE (as the parent panel) | no `global_denom_*` columns in this output type — disclosed, not skipped |
| g-precision | full-precision columns present | — |
| physics floor | no-op | 0 all-zero events in either channel; min nonzero L = 1.57e-6 (2D) / 1.46e-5 (1D) |

NO-READ triggers: none fired.

## 3. The registered statistic (T0 gradient-weighted scorer, `H_GRID_41`, uniform prior)

| channel | n_h | n_events | MAP | mean_h | σ_h | mass(h ≤ 0.63) | mass at node 0.600 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2D `combined_with_bh` (primary) | 41 | 1343 | **0.630** | **0.627674** | 0.014006 | 0.713 | 0.0439 |
| 1D `combined_no_bh` (replicate) | 41 | 1343 | **0.630** | **0.631084** | 0.013939 | 0.619 | 0.0259 |

Per-node posterior mass, 2D: 0.600 : 0.044 · 0.610 : 0.138 · 0.620 : 0.253 · 0.630 : 0.277 ·
0.640 : 0.186 · 0.650 : 0.059 · 0.655 : 0.022 · 0.660 : 0.011 (1D: 0.026 · 0.095 · 0.211 · 0.287 ·
0.233 · 0.085 · 0.033 · 0.017). Mass above 0.66 is < 1 % in both channels.

Distances in units of the measured σ_h (registered 3σ band):

| channel | \|mean − anchor\| / σ (anchor 0.6659 / 0.6670) | \|MAP − 0.665\| / σ | \|mean − 0.67\| / σ (reported) | \|mean − 0.666\| / σ (reported) |
|---|---:|---:|---:|---:|
| 2D | **2.73** | **2.50** | 3.02 | 2.74 |
| 1D | **2.58** | **2.51** | 2.79 | 2.50 |

Anchors are the row #302 iiib re-baseline at truth 0.73 (2D 0.665854 / 1D 0.666987), as
registered. Truth of this pool is 0.67 (public; the (m1) node makes no anti-tuning claim).

## 4. Reading against the registered cell table — TWO CELLS FIRE

- **TUNED** criterion: `|mean_h − 0.6659| ≤ 3σ_h AND |MAP − 0.665| ≤ 3σ_h`, 1D agreeing.
  2D: 2.73 ≤ 3 and 2.50 ≤ 3 → **satisfied**; 1D: 2.58 and 2.51 → satisfied. Literal TUNED.
- **NOT-TUNED** criterion, first clause: `|mean_h − 0.6659| > 3σ_h with mean_h < 0.6659` → 2.73
  is NOT > 3 → **not satisfied**. Second clause ("INCLUDING a railed posterior (MAP at 0.600 or
  edge-node mass > 1e-3), booked NOT-TUNED-AT-RAIL and read AS A BOUND"): edge-node mass 0.044 >
  1e-3 → **satisfied** → literal NOT-TUNED-AT-RAIL.
- The NOT-TUNED row's head phrase "posterior mass concentrated at/below 0.63" is ALSO satisfied on its face
  (71 % / 62 % of mass at h ≤ 0.63, MAP = 0.630) while its parenthetical formalisation (> 3σ) fails —
  the table is inconsistent with itself on this posterior in two places (verifier finding).
- **INTERMEDIATE** ("neither") does not apply — both fire; the table did not anticipate a
  posterior that sits 2.5–2.7 σ below the anchor while leaking 4 % onto the 0.600 floor.

What is not in dispute (numbers only): the posterior is centred at 0.628–0.631, i.e. **0.0382 (2D) / 0.0359 (1D)
below the channel anchors and 0.0423 (2D) / 0.0389 (1D) below the 0.67 truth**, with 71 % (2D) / 62 % (1D) of its
mass at or below 0.63, and its σ_h is narrower than the re-baseline's at N = 1588 (2D 0.014006/0.018475 = 0.758, −24 %; 1D 0.013939/0.017526 = 0.795, −20 %) (the dossier expected ≈ 9 % wider at N = 1343). Distance to the TRANSFER prediction
of draft §7 (0.606–0.611): 2D mean is 1.2–1.6 σ above it; the 0.600 floor is 2.0 σ below the
mean. The chair does not adjudicate which cell binds; both readings and their consequences are
put to the author as **R25** (docket §9).

## 5. Verification

Independent top-tier verifier (decision 7, "unseal read tier"): see §6 (appended after the
verifier returns). The verifier is asked to refute the numbers, the scorer-convention match and
the cell-table reading — not to make the call.

## 6. Verifier appendix (top-tier decisive verifier, independent scorer written from scratch, 2026-09-04)

Scorer re-validated on the head-rebaseline iiib CSV (exact anchors). **Every §3 number reproduces**
(mean/σ to < 1e-6, MAP and per-node masses exact, all eight σ-ratios to 2 dp). (a) VERIFIED ·
(b) VERIFIED · (c) VERIFIED as a literal reading, plus the second inconsistency now in §4 ·
(d) 11/13 tokens VERIFIED from `cli_args` on all 41 files (also h_value = H_GRID_41[TID], seed =
777000 + TID); the 2 internal tokens CANNOT-VERIFY locally (chair's evidence is the cluster log
line and the code default; logs not retrieved) · (e) commits VERIFIED (exit 0 on both diffs and
the merge-base checks; split is tasks 0–4 / 5–40); working-tree caveat added to §2 · (f) VERIFIED,
wording corrected (−24 % 2D, −20 % 1D) · (g) VERIFIED: 0 zero / negative / NaN entries.
Defects raised and applied above: G-1 pool pin was a chair override of the registered text (→ R28);
cross-pool anchor question (chair resolved: same pool, same manifest); "0.038/0.040" made
channel-explicit; "1.2–1.5σ" → "1.2–1.6σ"; the commit split named. No cell call made by the verifier.
