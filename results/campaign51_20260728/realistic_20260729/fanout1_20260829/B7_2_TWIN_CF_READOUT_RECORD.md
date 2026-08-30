# B7.2 (PROD-CF-2D) independent-reader readout

**[SUPERSEDED — see §6 below for the full RESULT RECORD, appended 2026-08-29 after the diagnostics-CSV-only readout was executed on the coordinator's instruction. The BLOCKED-ON-RETRIEVAL stub immediately below (§1–§5) is kept verbatim, append-only, as the record of what was known/missing at the outage; it is no longer the current status.]**

**Status: BLOCKED ON RETRIEVAL (2026-08-29, ~21:25).** Launched under rows #222/#223 — charter
node B7.2, independent reader. This is a stub, not the result record: cluster SSH went down for
non-interactive auth mid-retrieval (the `ControlMaster` session expired around 21:15; a
non-interactive key attempt now fails `Permission denied (publickey,keyboard-interactive)` even
with `BatchMode=yes` — re-authentication needs an interactive OTP that only the author can supply).
Retries were stopped on the orchestrator's instruction rather than continuing to hammer a
non-interactive login that cannot succeed. Nothing below is a registered reading; the readout
script is written and dry-run-verified against real production data (C3's arm, structure/logic
check only — no C3 record touched, no C3 gate values asserted here), and will produce the actual
record the moment retrieval finishes and it is re-run against the completed
`wave2_20260829/c4/` tree.

## 1. What is already confirmed (does not require further cluster access)

| item | value | source | date |
|---|---|---|---|
| Wave-2 commit at run | `ff2306213e9e65abbd474f66348bc05a6f3e6547` | `cat GIT_COMMIT_AT_RUN.txt` over ssh (pre-outage) + local `git log` on `ff230621` | 2026-08-29 |
| Job 6739000 (task 0, h=0.730, STEP-2 smoke) | COMPLETED, ExitCode 0:0, Elapsed 00:06:25, node uc2n609 | `sacct -j 6739000,6739001 -X -n -P` (pre-outage) | 2026-08-29 |
| Job 6739001 task 1 | COMPLETED, ExitCode 0:0, Elapsed 00:06:38, node uc2n853 | same | 2026-08-29 |
| Job 6739001 task 2 | COMPLETED, ExitCode 0:0, Elapsed 00:06:17, node uc2n848 | same | 2026-08-29 |
| Job 6739001 task 3 | COMPLETED, ExitCode 0:0, Elapsed 00:06:10, node uc2n810 | same | 2026-08-29 |
| All 4 tasks COMPLETED | yes, 4/4 | derived from the row above | 2026-08-29 |
| STEP-2 overhead pin | measured task-0 (h=0.730) Elapsed 00:06:25 (385 s) vs C0's 00:06:28 (388 s, `off` arm, same h, same venue) ⇒ **overhead factor ≈ 0.99×** (i.e. ≈ 1.0×, NOT slower) | `sacct` row above; `REGISTRATION_C0_BASELINE_GATE_20260829.md` §13 RESULT RECORD | 2026-08-29 |
| Walltime resubmit rule (Appendix note 14 item 2) | NOT triggered — all 4 tasks finished in ≤ 00:06:38, far under `--time=03:00:00` | same sacct row | 2026-08-29 |
| Measured cost | Σ Elapsed = 385+398+377+370 = 1530 s = 0.425 h × 16 cpus/task ≈ **6.8 CPU-h actual** for the 4-task arm, vs the registered **59.7–81.1 CPU-h (arm only) / 74.7–101.4 CPU-h (+ baseline gate task, not needed — C0 PASSED zero-compute) / ceiling 105 / 132 CPU-h** estimate (proposal §6.2/§13.3) — roughly a **9–15× overestimate**, consistent with C0's own finding of a 9–13× overestimate from the same stale (3355-event) anchor (`REGISTRATION_C0_BASELINE_GATE_20260829.md` §13 "F4 costing correction") | derived from the sacct row above | 2026-08-29 |
| Baseline reuse | C0 gate PASSED bit-identical (row #246; `REGISTRATION_C0_BASELINE_GATE_20260829.md` §13) ⇒ the banked HEAD readout `headreadout_20260827/iiib/` (commit `d04d9dc9`) is the zero-compute baseline for C4 — no separate baseline task needed | same | 2026-08-29 |

## 2. What is retrieved vs missing (local tree state at the outage)

`results/campaign51_20260728/realistic_20260729/wave2_20260829/c4/` currently holds:

| artifact | status | note |
|---|---|---|
| `simulations/diagnostics/event_likelihoods.csv` | **COMPLETE** (1,795,584 bytes, 6353 lines = 1 header + 4×1588 rows, all 4 H4 nodes present: 0.660/0.665/0.670/0.730) | this is the ONLY input the R1/R2/R6 gates and the registered Δℓ′ stencil need — **the readout script can already run its gate/stencil computation on real C4 data**, but per the orchestrator's instruction this stub does not assert that as the record |
| `simulations/posteriors/h_0_{66,665,67,73}.json` | **COMPLETE**, all 4 files (~48.8 KB each) | redundant with the CSV per-event values; not additionally load-bearing for the registered reads |
| `simulations/posteriors_with_bh_mass/h_0_66.json`, `h_0_665.json` | present (130,590,466 / 130,602,224 bytes) | |
| `simulations/posteriors_with_bh_mass/h_0_67.json`, `h_0_73.json` | **MISSING** — rsync dropped mid-transfer of this subpath when the control session expired | not required by the registered R1/R2/R6/stencil reads (those consume `event_likelihoods.csv` per §3 of `REGISTRATION_C0_BASELINE_GATE_20260829.md`, "C4 columns consumed: `L_cat_with_bh`, `combined_with_bh`") |
| `run_metadata_*.json` (7, 8, 9, 21) | **MISSING locally** (contents not yet pulled; commit/CLI already independently confirmed via direct `cat` over ssh pre-outage, see §1) | |
| `logs/` (SLURM out/err, provenance JSON) | **MISSING locally** | needed for the dataset-pin STOP-gate check (CRB/catalogue md5) and any per-task diagnostic detail, not for the gates themselves |
| `GIT_COMMIT_AT_RUN.txt` | **MISSING locally** (value already confirmed via ssh `cat`, §1) | |

## 3. Exact commands to resume (once SSH is authenticated again)

**Retrieval** (brace expansion silently fails over this cluster's remote shell — one `rsync` per
subpath, not the combined `{a,b,c}` form):

```bash
WS=/pfs/work9/workspace/scratch/st_ac147838-emri
DEST=results/campaign51_20260728/realistic_20260729/wave2_20260829/c4
mkdir -p "$DEST/simulations" "$DEST/logs"
rsync -az bwunicluster:"$WS/run_20260829_wave2_c4_iiib/simulations/diagnostics" "$DEST/simulations/"
rsync -az bwunicluster:"$WS/run_20260829_wave2_c4_iiib/simulations/posteriors" "$DEST/simulations/"
rsync -az bwunicluster:"$WS/run_20260829_wave2_c4_iiib/simulations/posteriors_with_bh_mass" "$DEST/simulations/"
rsync -az bwunicluster:"$WS/run_20260829_wave2_c4_iiib/run_metadata_"*.json "$DEST/"
rsync -az bwunicluster:"$WS/run_20260829_wave2_c4_iiib/logs" "$DEST/"
rsync -az bwunicluster:"$WS/run_20260829_wave2_c4_iiib/GIT_COMMIT_AT_RUN.txt" "$DEST/"
```

(The first two `rsync` calls are idempotent re-runs over already-complete data — harmless, kept
for a single clean resume script rather than a hand-picked subset.)

**Verification** (re-confirm nothing changed during the outage):

```bash
ssh bwunicluster 'sacct -j 6739000,6739001 -X -n -P --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList'
diff <(ssh bwunicluster 'cat '"$WS"'/run_20260829_wave2_c4_iiib/GIT_COMMIT_AT_RUN.txt') \
     results/campaign51_20260728/realistic_20260729/wave2_20260829/c4/GIT_COMMIT_AT_RUN.txt
```

**Readout** (the script, written and dry-run-verified 2026-08-29 against `wave2_20260829/c3/`'s
real production data as a structure/logic check only — no C3 gate values are asserted anywhere in
this record):

```bash
uv run python results/campaign51_20260728/realistic_20260729/fanout1_20260829/b7_2_readout.py \
  --arm-dir results/campaign51_20260728/realistic_20260729/wave2_20260829/c4 \
  --baseline-dir results/campaign51_20260728/realistic_20260729/headreadout_20260827/iiib \
  --out-json results/campaign51_20260728/realistic_20260729/fanout1_20260829/b7_2_readout.json
```

The script (`fanout1_20260829/b7_2_readout.py`) implements, per §6.2/§13.3 of
`PROPOSAL_2D_TWIN_ADOPTION_20260829.md`:
- **R1** (eventwise `ln L_cat,wbh^T ≤ ln L_cat,wbh^B`, every event × H4 node) — INSTRUMENT-DEFECT on
  any violation (tiny 1e-12 fp slack only, not a band relaxation).
- **R2** (A13 engagement at h=0.730: fraction of baseline-active events with `|Δ ln L_cat,wbh| >
  1e-6` ≥ 0.95) — STOP below threshold.
- **R6** (1D channel — `L_cat_no_bh`, `combined_no_bh` — bit-identical between arms at every H4
  node) — operationalized at `max_abs ≤ 1e-12`, matching the project's own C0-gate definition of
  "bit-identical" (a C3 dry run showed ~1e-16 float noise between independently-run arms on shared
  columns; an exact `atol=0.0` would false-flag that noise, so 1e-12 is used as the noise floor,
  not slack for a real defect — disclosed in the script's `R6_ATOL` comment).
- The registered stencil `Δℓ′(0.665)` (central difference over {0.660,0.665,0.670} of
  `Σ_events ln[combined_with_bh^T/combined_with_bh^B]`), `Δℓ″` validity check (interpreted as
  `|Δℓ″| < 0.1·I_HEAD`, disclosed), `Δmean_h,pred = Δℓ′/I_HEAD` with `I_HEAD = 2965`, and the
  two-sided verdict classification at `T_mat = 0.008` / `T_mat/2 = 0.004`.
- A secondary, REPORTED-ONLY 4-node MAP/mean cross-check (explicitly caveated in its own output as
  not a valid full-grid posterior read).

## 4. Dry-run verification (C3, structure/logic check only — no C3 record)

Run 2026-08-29 against `wave2_20260829/c3/` purely to confirm the script parses the real A22
production layout (`simulations/diagnostics/event_likelihoods.csv` with the 19-column
post-`d40fe5c8` header, `simulations/posteriors{,_with_bh_mass}/h_0_*.json`) and executes without
error against the banked baseline's flatter layout (`headreadout_20260827/iiib/event_likelihoods.csv`,
16-column pre-`d40fe5c8` header) — column lookups are by name, not position, so the header-width
mismatch is handled transparently. The script ran to completion and produced well-formed gate/
stencil/verdict output. **These numeric values are C3's (log-k3 counterfactual, a different arm
with a different registered inequality direction) and are NOT reported here as any B7.2/C4
reading** — per the task boundary, C3 belongs to another reader and this session does not assert
anything about its gates or verdict.

## 5. Next step

Resume this reader (same context) once cluster SSH authentication is restored. The remaining
retrieval is small (2 posteriors_with_bh_mass files, 4 run_metadata JSONs, logs/, one text file);
the diagnostics CSV that the registered gates and stencil actually consume is already complete
and locally verified (§2). On resume: run §3's verification + readout commands, then produce the
full RESULT RECORD (comprehension-first paragraph + tables), append the ⟨SUBMIT⟩/RESULT RECORD to
`PROPOSAL_2D_TWIN_ADOPTION_20260829.md`, and write `b7_2_readout.json` (the script's `--out-json`
already produces this in the registered schema).

Stamped: launched under rows #222/#223 — charter node B7.2, independent reader, BLOCKED ON
RETRIEVAL, 2026-08-29 ~21:25.

---

## 6. RESULT RECORD (appended 2026-08-29, ~21:35) — PROVISIONAL on provenance extras

**PROVISIONAL — provenance extras (`run_metadata_*.json`, `logs/`, the 2 missing
`posteriors_with_bh_mass` JSONs for h=0.67/0.73, a local `GIT_COMMIT_AT_RUN.txt` copy) pending
retrieval; commit `ff230621` confirmed pre-outage via `ssh cat`; the numeric readout below is
complete on the diagnostics CSV.** No ssh was attempted in producing this section, per the
coordinator's instruction.

### 6.1 Comprehension-first summary

All three registered gates PASS on the complete diagnostics CSV (all 4 H4 nodes, 1588 events
each, commit `ff230621`, all 4 SLURM tasks COMPLETED): the twin arm never produces a higher
with-BH catalogue likelihood than the baseline at any of the 6352 checked (event, h) pairs (R1),
the switch demonstrably reaches production dispatch on every one of the 982 events with a
non-empty candidate set (R2, engagement = 1.0, well above the 0.95 floor), and the untouched 1D
channel is exactly bit-identical between arms at every node (R6, `max_abs = 0.0`). The registered
stencil reading — the first-order predicted mean-h shift `Δmean_h,pred = Δℓ′(0.665)/I_HEAD` —
comes out to **+0.0025**, an order of magnitude below the materiality half-band (`T_mat/2 = 0.004`)
and well clear of the AMBIGUOUS zone, so the read classifies cleanly as **IMMATERIAL-PREDICTED**
with no conditional G27 escalation triggered. A secondary CSV-derived 4-node MAP/mean cross-check
and a per-event sign census both corroborate the same small, uniformly-signed effect: at every H4
node, zero events show the twin's likelihood tilting upward relative to baseline, matching the
proposal's own "expected direction" argument that `S_4D<1` should lower (never raise) the with-BH
catalogue-leg likelihood. Net: on production behaviour, this specific flip looks safe and small at
these four points, but the story is not yet closed — the attribution of *why* stays PROVISIONAL
until falsifier (ii) runs, and this read is explicitly not the flip's H₀ verdict (that comes from
the wave-3 blind HEAD readout).

### 6.2 Gates (registered order)

| gate | verdict | detail |
|---|---|---|
| **R1** — eventwise `ln L_cat,wbh^T ≤ ln L_cat,wbh^B`, every (event,h), equality only on empty candidate sets | **PASS** | 6352 rows checked (4 H4 nodes × 1588 events); 0 violations; 2424 rows with an empty candidate set on both arms (equality by construction, not a discriminating check) |
| **R2** — A13 engagement at h=0.730, ≥0.95 fraction of active events with `\|Δ ln L_cat,wbh\|>1e-6` | **PASS** | 982/982 active events (baseline `L_cat_with_bh>0`) engaged; **engagement fraction = 1.0** |
| **R6** — 1D channel (`L_cat_no_bh`, `combined_no_bh`) bit-identical, every H4 node | **PASS** | `max_abs = 0.0` exactly at all 4 nodes (script's operational floor is ≤1e-12, matching C0's own "bit-identical" band; the observed value is exact zero, not just inside the floor) |

Source: `b7_2_readout.json` (`gates.R1`, `gates.R2`, `gates.R6`), produced by
`fanout1_20260829/b7_2_readout.py --arm-dir wave2_20260829/c4 --baseline-dir headreadout_20260827/iiib`,
run 2026-08-29 on the retrieved `simulations/diagnostics/event_likelihoods.csv`.

### 6.3 Primary registered reading — the Δℓ′(0.665)/I_HEAD stencil

| h | Δℓ(h) = Σ_events ln[combined_with_bh^T/combined_with_bh^B] | n events (0 dropped as non-positive) |
|---|---:|---:|
| 0.660 | −3.030674 | 1588 |
| 0.665 | −2.993148 | 1588 |
| 0.670 | −2.956381 | 1588 |

- Central-difference slope: **Δℓ′(0.665) = +7.429355 nats/unit h** (step 0.005)
- Curvature: Δℓ″(0.665) = −30.311364; validity condition `\|Δℓ″\| ≪ I_HEAD` (I_HEAD=2965) —
  operationalized here as `< 0.1·I_HEAD = 296.5` (disclosed interpretation of the registered "≪",
  not itself a registered numeric threshold) — **HOLDS** (30.3 ≪ 296.5).
- **Δmean_h,pred = Δℓ′(0.665) / I_HEAD = +0.0025057**

**Verdict per the registered map (two-sided, T_mat=0.008, IMMATERIAL-PREDICTED at ≤T_mat/2=0.004):**
`\|+0.0025057\| ≤ 0.004` ⇒ **IMMATERIAL-PREDICTED.** Validity condition holds and the value sits
well clear of the 0.004–0.008 AMBIGUOUS band, so no conditional G27 escalation (proposal §6.2 item
2 / §11 item 2) is triggered by this reading.

Source: `b7_2_readout.json:stencil`.

### 6.4 Secondary readings (REPORTED-ONLY)

- **Direct 4-node MAP/mean cross-check**, computed from the diagnostics CSV's own
  `combined_with_bh` column summed in log over events at each H4 node (explicitly **not** a valid
  full 41-node posterior — a REPORTED-ONLY sanity check against the primary stencil, per the
  registered design):

  | | MAP | mean |
  |---|---:|---:|
  | arm (twin) | 0.665 | 0.665212 |
  | baseline | 0.665 | 0.665020 |
  | Δ | 0.0 | **+0.000192** |

  Same sign, same order of immateriality as the primary stencil reading (§6.3). Per the
  coordinator's instruction: this is stated as **the CSV-derived total** for h=0.67 and h=0.73 —
  the independent `posteriors_with_bh_mass` JSON-object cross-check at those two nodes has not yet
  run (the files were not retrieved before the SSH outage, §2 above); it is added when retrieval
  completes. (Weak prior that it will reconcile: C0's RESULT RECORD found the JSON posterior
  objects are per-event dicts keyed by event index, byte-identical in content to the corresponding
  CSV rows for the reproduction-gate case — i.e., structurally redundant with the CSV, not an
  independent computation path — though that equivalence has not been directly re-verified for
  this arm's own JSON files.)

- **Per-event sign distribution of Δln combined_with_bh** (active events only, i.e. baseline
  `L_cat_with_bh > 0`):

  | h | n active | positive | negative | ≈0 (\|Δ\|≤1e-9) | mean Δln | median Δln |
  |---|---:|---:|---:|---:|---:|---:|
  | 0.660 | 982 | 0 | 936 | 46 | −0.003086 | −0.000027 |
  | 0.665 | 982 | 0 | 935 | 47 | −0.003048 | −0.000025 |
  | 0.670 | 982 | 0 | 931 | 51 | −0.003011 | −0.000023 |
  | 0.730 | 982 | 0 | 872 | 110 | −0.002618 | −0.000007 |

  **Zero events tilt positive at any H4 node** — no counterexample to the proposal's own
  "expected direction" note (§6.2 last paragraph: `S_4D<1` should lower, never raise, the with-BH
  catalogue-leg likelihood and its mixture share). This is corroborating evidence, not itself a
  registered gate or band.

- **Not computed this pass** (out of the scope the coordinator specified — gates, primary
  numbers, verdict, caveats): R3 (ΔT score-at-truth tilt, row-#201 form), R4 (Δw̄₂ mixture-weight
  shift), the ×2.5 residual check (requires the separate unrun Option A′ fleet re-run, falsifier
  (ii), row #220), and class-resolved deltas. None of these affect the gate verdicts or the
  primary/secondary readings above.

### 6.5 Verdict of record

**IMMATERIAL-PREDICTED** (primary stencil reading, §6.3; corroborated by the secondary 4-node
cross-check and the sign census, §6.4). All three gates PASS. No conditional escalation to G27.

### 6.6 Caveats

1. **Attribution provisional on falsifier (ii).** Falsifier (i) PASSED (§13.1 of the proposal);
   falsifier (ii) (Option A′ + 24–33-task fleet re-run, 208–286 CPU-h) has not run this wave
   (row #220) — the mechanistic attribution of this IMMATERIAL-PREDICTED result to the twin's
   `S_4D`-homogeneity property is PROVISIONAL until it returns.
2. **Not the adoption's H₀ verdict.** Per F2 (proposal §9), the actual H₀ effect of this flag flip
   is read once, unconditionally, in the wave-3 blind HEAD readout with its own per-change arm on
   both venues at `H_GRID_41`; this B7.2 read is the measure-first production read informing the
   B7.3 `/physics-change` gate, not that verdict.
3. **Provenance extras pending retrieval** (see the PROVISIONAL label, top of §6): 2
   `posteriors_with_bh_mass` JSONs, `run_metadata_*.json`, `logs/`, a local `GIT_COMMIT_AT_RUN.txt`
   copy. None of the gate or reading numbers above depend on these — all are diagnostics-CSV-only.
4. **Disclosed operationalizations, not silent band changes:** the registered "≪" in the validity
   condition is read as `<0.1·I_HEAD` here; the registered "bit-identical" in R6 is operationalized
   at `≤1e-12` (though the observed value in this run is exact 0.0). Both are stated with numeric
   thresholds so they can be checked or tightened by a later reader.
5. **No ssh attempted** in producing this record, per the coordinator's explicit instruction.
6. R3/R4/the ×2.5 residual/class-resolved deltas are not computed (§6.4) — their absence does not
   change the gate verdicts or the materiality classification, which rest entirely on R1/R2/R6 and
   the registered Δℓ′/I_HEAD stencil.

Source data: `results/campaign51_20260728/realistic_20260729/wave2_20260829/c4/simulations/diagnostics/event_likelihoods.csv`
vs `results/campaign51_20260728/realistic_20260729/headreadout_20260827/iiib/event_likelihoods.csv`;
script `fanout1_20260829/b7_2_readout.py`; full machine-readable output
`fanout1_20260829/b7_2_readout.json`. Mirrored (compact form) into
`PROPOSAL_2D_TWIN_ADOPTION_20260829.md` §15.

Stamped: read out 2026-08-29 by the independent reader; launched under rows #222/#223 — charter
node B7.2.

---

## 7. A14 housekeeping append (2026-08-30, row #255 -- provenance extras retrieved)

Provenance extras retrieved 2026-08-30 (run_metadata_*.json, logs/, 4/4
posteriors_with_bh_mass JSONs, GIT_COMMIT_AT_RUN.txt = ff230621 -- verify) under the local
files at results/campaign51_20260728/realistic_20260729/wave2_20260829/c4/. Confirmed on disk:
GIT_COMMIT_AT_RUN.txt contains ff2306213e9e65abbd474f66348bc05a6f3e6547 (matches the
pre-outage ssh cat confirmation of commit ff230621 cited in section 6's header); logs/
holds 4 provenance JSONs + 8 SLURM task out/err files (tasks 0/1/2/3, jobs 6739000, 6739027,
6739028, 6739001); posteriors_with_bh_mass/ holds all 4 H4-node JSONs
(h_0_665.json, h_0_66.json, h_0_67.json, h_0_73.json) -- the "2 missing" of caveat 3
(h=0.67/0.73) are present, so retrieval is 4/4, complete. run_metadata_7.json,
run_metadata_8.json, run_metadata_9.json, run_metadata_21.json are present (one per task).

**PROVISIONAL label dropped:** the "PROVISIONAL on provenance extras" label at the top of
section 6, and caveat 3 in section 6.6, are resolved -- all four provenance items are now
retrieved and verified on disk. This does NOT lift caveat 1 ("attribution provisional on
falsifier (ii)"): the mechanistic attribution of the IMMATERIAL-PREDICTED result stays
PROVISIONAL until falsifier (ii) (Option A', 208-286 CPU-h, row #220) runs; only the provenance
retrieval provisionality is closed by this append. No gate or reading number in sections 6.1-6.5
changes -- they were always diagnostics-CSV-only and complete. {source:
results/campaign51_20260728/realistic_20260729/wave2_20260829/c4/GIT_COMMIT_AT_RUN.txt;
results/campaign51_20260728/realistic_20260729/wave2_20260829/c4/logs/;
results/campaign51_20260728/realistic_20260729/wave2_20260829/c4/posteriors_with_bh_mass/;
results/campaign51_20260728/realistic_20260729/wave2_20260829/c4/run_metadata_{7,8,9,21}.json;
verified 2026-08-30}

Launched under row #255 -- tree 2 node A14.
