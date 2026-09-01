# rd-runner11 — read node (verdict-free) — 2026-09-02

**Authorization:** ledger row #290 ("rows 3-11 [DO] APPROVED — branch heads A-I trigger their
first items ... PA-HIER-33 scorer + iiib driver build"), decisions-table row 6 of
`RESEARCH_GRAPH_1_PROPOSAL_20260901.md` §1.4 (Branch D): `rd-runner11 -> b-pahier33-scorer`,
DO, Approved. This is the `rd-runner11` read node — verdict-free, three-valued existence
contract (present / absent / unreachable are distinct outcomes). Effort: low.

Read-first precondition per runbook 40 §3 item 1: "Read runner-11's output (8-cell b-node pair,
T1.3 config) — this is the S0-B precondition."

## What was read

`results/campaign51_20260728/realistic_20260729/tree2_20260830/hier_s0_zwin_bnodes_run/`:
- `s0a_score_output.json`, `s0a_score.md` (both present, from `--score-only`, 2026-08-31 15:38)
- `logs/runner11_bnodes.log`, `logs/runner11_stage.txt`
- `s0a_seed900101/` .. `s0a_seed900104/` (4 seed directories, each with `es_null_det.csv` and
  `node_b_plus_sites2.2_nosmear_divisor_zwin_zk4/`, `node_b_minus_sites2.2_nosmear_divisor_zwin_zk4/`
  simulation output — `cramer_rao_bounds.csv`, `event_likelihoods.csv`, posteriors, diagnostics)

## Config fields, verbatim (from `s0a_score_output.json`)

```
"arm": "S0-A"
"seeds_requested": [900101, 900102, 900103, 900104]
"nodes_requested": ["b_plus", "b_minus"]
"n_present_by_node": {"b_plus": 4, "b_minus": 4}
"seeds_present_by_node": {"b_plus": [900101..900104], "b_minus": [900101..900104]}
"n_missing_csv": 0
"missing_csv_paths": []
"theta_sites": "2.2"
"smear": "off"
"config": "b0i"
"h_values": [0.73]
"score_h": 0.73
"theta_phi_divisor": "on"
"sky_cone_k": 1.5
"catalogue_leg_1d_mass_aware": "off"
"theta_zwindow": "on"
"z_window_k": 4.0
"registration": ".../PREREGISTRATION_HIER_HTHETA_20260826.md"
```

This matches the task's expectation exactly (b0i config, divisor on, zwin on,
`catalogue_leg_1d_mass_aware="off"`).

## The 8 cells' scores, verbatim

`ln_L_no_bh` (primary channel), `score_b`:
- mean = -0.8623345057895397
- sem  = 0.47694418757541013
- Z    = -1.8080407063419661
- n_pooled = 461

`ln_L_with_bh` (secondary channel), `score_b`:
- mean = 0.3166507176559612
- sem  = 0.4097264347381919
- Z    = 0.772834483716667
- n_pooled = 461

**Comparison against row #287** (ledger, `BIAS_HISTORY_LEDGER.md:3213`): row #287 quotes
"score_b(no-BH) = -0.862 +/- 0.477, Z_b = -1.808, n = 461; score_b(with-BH) = +0.317 +/- 0.410,
Z = +0.773." **No discrepancy** — these are the same numbers to the quoted precision (row #287's
figures are the rounded form of the JSON's full-precision values above). No adjudication is made
here per the read-node's verdict-free scope; this is a record of agreement, not a ruling.

`score_s`/`score_lns`/`score_s_raw`/`score_lns_R` all report `n_pooled=0`/NaN for both channels
— this is a b-only run (`nodes_requested: ["b_plus", "b_minus"]`), the s-axis was never
requested. `score_s_available: false`, `score_lns_R_available: false`.

`gate_eng`: all four node keys (`b_plus`, `b_minus`, `s_plus`, `s_minus`) report
`eng_available: false`, `per_seed_fraction_moved: []`, `mean_fraction_moved: NaN`, `pass: false`.
This is the row #287 `gate_eng` fix in effect (degrades gracefully on a b-only node dict with no
`"truth"` key, instead of raising `KeyError: 'truth'`) — see below.

`note` field, verbatim: "only the b-axis is ready on disk (b_ready=True, s_ready=False) -- the
OTHER axis's score in payload['scores'] is unavailable (n_pooled=0/NaN), by design, NOT an
error."

## Three-valued existence contract applied

- **PRESENT**: `s0a_score_output.json`, `s0a_score.md`, both logs, and all 4 seed directories
  with both b-nodes' full simulation output (CRB, event_likelihoods, posteriors, diagnostics)
  exist on disk and are readable. The b-axis (8 cells: 4 seeds x {b_plus, b_minus}) is fully
  PRESENT.
- **ABSENT**: the s-axis (s_plus/s_minus nodes) was never requested by this run
  (`nodes_requested: ["b_plus", "b_minus"]`) — no directories, no CSVs, no log lines reference
  s_plus/s_minus at all (`find ... -iname "*s_plus*" -o -iname "*s_minus*"` returns nothing).
  This is a deliberate ABSENCE (never launched), not a failure.
- **UNREACHABLE — explicitly ruled out**: checked `logs/runner11_bnodes.log` directly for any
  I/O, SSH, or network failure signature (`grep -ic "error\|exception\|traceback\|failed"` = 2
  hits, both accounted for below). The log shows the venue/evaluate() pipeline running end-to-end
  cleanly for all 8 cells (per-event diagnostics written, `n_events=130`, `evaluate_s`/`wall_s`
  timings present for the final b_minus/seed900104 cell), THEN a single Python traceback at the
  very end:
  ```
  KeyError: 'truth'
    File ".../hier_s0_driver.py", line 1449, in gate_eng
      truth_by_seed = {r.seed: r.ln_l.set_index("event_idx")[channel] for r in all_nodes["truth"]}
  === 2026-08-31T15:28:53+02:00 END run rc=1
  ```
  This is a **driver code defect** (unconditional `all_nodes["truth"]` index on a b-only node
  dict), not an I/O/SSH/unreachable failure — the 8 cells themselves computed cleanly and are on
  disk; only the driver's own post-hoc `gate_eng` scoring crashed after them. This matches row
  #287's account exactly ("computed all 8 cells cleanly, then crashed in post-hoc scoring").
  `logs/runner11_stage.txt` = `FAIL-run`, consistent with `rc=1`.
- **Disposition**: this crash was already fixed (row #287, subagent-built, orchestrator-reviewed)
  — `gate_eng` now degrades with `eng_available=False` on a missing truth node, with regression
  test `test_gate_eng_handles_a_b_only_node_dict_with_no_truth_node`
  (`darksiren_emri_test/bayesian_inference/test_theta_zwindow.py`) — and the subsequent
  `--score-only` zero-compute rescore (this read's `s0a_score_output.json`) reproduced the 8
  banked cells' scores cleanly, matching the numbers quoted above. **The three-valued read of
  record here is: b-axis PRESENT (uncontaminated), s-axis ABSENT (never requested), no
  UNREACHABLE/I-O failure anywhere in the chain** — the one failure that did occur (the
  `gate_eng` KeyError) is a code defect on already-computed data, already fixed and verified, not
  an existence-contract event.

## Fields the PA-HIER-33 scorer will need

Cross-checked against `s0a_score_output.json`'s schema and the `es_null_det.csv` /
`event_likelihoods.csv` files under each seed directory:

- Per-event `ln_L_no_bh` / `ln_L_with_bh` columns at the **truth** node (`l_i(0)`) — ABSENT from
  this run (b-only; no truth node was requested). PA-HIER-33's `Es_null^{(arm)}` estimator needs
  the truth node's per-event log-likelihood in addition to `s_plus`/`s_minus` (see the scorer
  build record for the closed-form rule) — this run cannot feed it. A future S0-B/S0-A s-axis run
  (the registered node list `{truth, s_plus, s_minus}`, per PA-HIER-33's own falsifier design)
  is what will supply this.
- Per-event `ln_L_no_bh` / `ln_L_with_bh` at `s_plus`/`s_minus` — needed for `score_lns`'s own
  numerator (already computed elsewhere in the driver); ABSENT from this b-only run for the same
  reason.
- `es_null_det.csv` (present per seed, node-independent) — this is PA-HIER-32(d)'s SUPERSEDED
  per-host closed form, not an input to PA-HIER-33's pooled Bartlett-identity estimator (which
  needs only the three s-nodes' raw ln-likelihoods, no per-host cache).
- `(seed, event_idx)` join keys — present and consistent across every CSV read here; this is the
  same join key the scorer's truth/s_plus/s_minus merge will use.

No verdict is rendered by this read. Everything above is a record of what exists, what does not,
and what crashed vs. what did not — per the read-node's verdict-free scope.

*Stamp: rd-runner11, 2026-09-02. Read-only; no git, no code changes, no compute; append-only.*
