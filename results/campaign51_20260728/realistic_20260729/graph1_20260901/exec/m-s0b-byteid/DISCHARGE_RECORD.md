# m-s0b g-byte-id precondition — SAME-MACHINE DISCHARGE (chair-performed comparison)

Date: 2026-09-02. Performed by the chair directly (decisive-number verification is chair work).
Authorization: row #290 decisions row 6 precondition; follows the RED cross-machine comparison
(row #318, COMPARISON_RECORD.md) whose chair diagnosis this run tests.

## Run
The identical cell (S0-A, seed=900103, node=b_minus, config verbatim: b0i, sites 2.2, smear off,
divisor on, sky 1.5, zwin on zk4, catalogue_leg_1d_mass_aware=off, h=0.73) re-executed on the
SAME machine that produced the banked reference (the local dev machine; runner-11 was a local
runner), at HEAD c83e391d-era working tree (contains both intervening [PHYSICS] commits a26959b4
and 2b657255, argued inert in LAUNCH_RECORD.md). Out-root: local_discharge_run/ (gitignored data).
Launcher script: scratchpad launch_s0b_local.sh (orchestrator-as-runner). Clean completion.

## Comparison (chair-run, python md5 walk over the banked cell tree)
- Banked: tree2_20260830/hier_s0_zwin_bnodes_run/s0a_seed900103/node_b_minus_sites2.2_nosmear_divisor_zwin_zk4
- Fresh:  local_discharge_run/s0a_seed900103/node_b_minus_sites2.2_nosmear_divisor_zwin_zk4
- **714 / 715 files md5-IDENTICAL** (symlinks excluded).
- The single differing file: simulations/fisher_quality_diagnostic.pdf — identical size (19022),
  exactly 7 differing bytes, all inside `/CreationDate (D:20260831143718 -> D:20260902145627)` —
  a render timestamp, zero data content. (Same artifact class as the cluster comparison's PDF diff.)

## Verdict
**GREEN — the g-byte-id precondition intent is DISCHARGED**: at same-machine reproduction the
b-pahier33-scorer build (and both intervening [PHYSICS] commits) leave the non-S0-B default path
byte-identical (>1.8e6 values via the md5-exact posteriors_with_bh_mass alone). The row #318 RED
stands on the record as evaluated; its cause is now PROVEN cross-machine FP non-associativity,
not a build defect. The cross-machine tolerance semantics question remains open for the author
(row #318). m-s0b-production's g-byte-id precondition is met; g-score-null evaluates at run time.
