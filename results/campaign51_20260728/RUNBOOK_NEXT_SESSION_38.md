# Runbook 38 — after fan-out 1's end-verifier pass (supersedes runbook 37)

**Read first.** Fan-out 1 (runbook 37) ran its full charter through wave-3 build and the
registered end-of-fan-out verifier, part 1. **Nothing in the next tree runs on the old grant
(A17, below) — row #222's STANDING grant lapses at the verifier pass, by its own text (ledger
row #254).** This session opens by presenting the verifier's docket to the author, not by
launching anything. Author-facing docket artifact:
`claude.ai/code/artifact/eeb5c7c3-54a5-414c-b05e-c8dfa842bfce`.

## 0. State at hand-off (2026-08-30)

- **Verdict counts (items 1–19):** confirmed 18 · refuted 0 · undetermined 1 (item 19, cluster
  `sacct` primitives unverifiable — SSH down, arithmetic itself is right) · deferred 1 (item 20,
  the wave-3 blind readout); no headline number was reversed. Commits of record: adoption
  `d4765539` (B7.3, `mz_sel`/`eff` twin, ledger row #253); wave-3 sbatch set `60f9996e`; HEAD
  `85dae577` (one commit later than the verifier brief's `b87ad2e6`, no verdict affected).
- **Wave 3 is built but NOT submitted** (`cluster/submit_wave3.sh` defaults `DRY_RUN=1`; three
  scripts: `wave3_c0prime_off_gate.sbatch`, `wave3_headreadout_{iiib,joint_r1}.sbatch`; 82+2
  tasks, 159.8–290.1 CPU-h + ≈6 CPU-h C0′ gate; F2-blind, neither readout script passes the 2D
  flag). Blocked because **cluster SSH is down** (`Permission denied` under `BatchMode=yes` since
  ≈21:15 on 2026-08-29, ControlMaster expired) — `ssh bwunicluster true` restores it (only the
  author can re-authenticate; `ControlPersist` is 8 h, OTP-gated). Also blocked on SSH: C4's
  provenance extras (`posteriors_with_bh_mass` h=0.67/0.73, `run_metadata_*.json`, `logs/`,
  `GIT_COMMIT_AT_RUN.txt` — gate/stencil numbers themselves are diagnostics-CSV-complete already).
- **Local run archive:** the registered Stage-0/KW-Q1 run dirs (41 gitignored raw files/logs, no
  git copy, no cluster copy — verifier F4/item 18) are tarred at
  `results/_archive/local_runs/fanout1_stage0_kwq1_runs_20260830.tgz` (gitignored); A13 (below,
  open) decides git-force-add / fold into the Option-A archive / both.
  `results/_archive/archive_run_wave2.sh`'s "wave 3" `ITEMS` block covers the four new out-roots
  — **run it AFTER retrieval**, once SSH returns; not yet run. Workspace `emri` expires
  **2026-09-23, 0 extensions** (24 days at wave-3 build time).

## 1. Entry points

- Verifier report (primary input): `fanout1_20260829/END_VERIFIER_REPORT_PART1_20260830.md`
  (§1 verdict table, §4 the 17 author items + 10 path decisions, §5 item-20 deferral, §6
  plain-language summary).
- Wave-2 chair docket: `fanout1_20260829/SYNTHESIS_DOCKET_2_20260829.md` (§2 tree state, §6 the
  same [RULE]s pre-adjudication, §7 next-tree ranking by yield/cost).
- Compute ledger: `fanout1_20260829/COMPUTE_LEDGER.md` (F4; anchor correction [A11]; the
  unbanked ≈8.6 CPU-h of crashed runner attempts, still un-lined). Wave-3 build note:
  `cluster/WAVE3_SUBMISSION_NOTE_20260830.md` (§1 tasks/`--time`, §1a C0′ off-gate logic, §2
  exact `sbatch` lines + 9-item checklist, §3 retrieval, §5 dataset registration).
- Driver limits: `B1_2_DRIVER_EXTENSION_NOTE.md` §8/§8.1 — `--jobs>1` is dead in
  `hier_s0_driver.py` (daemonic-process limit; always launch with `--jobs 1`); `kwq1_score.py`
  needs `--theta-sites 2.2 --smear off` or it silently scores 0 rows (disclosure D16).
- Registrations behind author items: `PREREGISTRATION_HIER_HTHETA_20260826.md` (PA-HIER-27..31),
  `PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md` (A4/A14 falsifier),
  `PREREGISTRATION_MKER_WGEOM_20260828.md` (folded into A1),
  `B1_1_S0A_DEFECT_FORENSIC_20260829.md` (Σ^φ(θ) divisor fix, A3).

## 2. The tree (state at verifier pass; branches own their own next node)

| branch | state | next node (needs a ruling) |
|---|---|---|
| **B1 [HIER]** | STOPPED at 1.1 — LOCALISED to a θ-free no-BH divisor Σ^φ (hook-placement gap, not arithmetic; twin agrees to 9.2e-13) | Σ^φ(θ) fix + sky-cone flag (A3) → S0-A re-cert → S0-B (A6); rank 1 next tree |
| **B2 [CMEM]** | PARKED at depth 1 (R2c p=0.0358, ≈68% power); A2 not triggered | A7 (register the pooled note?) / A8 (bank-and-park vs follow-up) |
| **B3 [POP]** | CLOSED — PREMISE-REFUTED (provenance, 0 CPU-h); mock's dark-host prior IS the estimator's own law | A5 (G7 row 16 re-grade) |
| **B4 [IMP]** | routed to 4.3 (INERT at 4.2 — kernel width, R=+0.085); no merge into B1 | mixture-weight h-slope + per-candidate instrumented run (A10 scoping word); rank 2 |
| **B5 [WIN]** | INTERMEDIATE, returned with numbers (+0.0035, sign up); retention-transfer FALSIFIED (66/76 unchanged; collapse is 621/1588 dark-class) | A1 (design ruling, folds WGEOM F-ii); no adoption, no k=3 in wave 3 |
| **B6 [ALIGN]** | CLOSED (`1f003da6`, C0-certified bit-identical) | A9 (raw-z vs z̃ judgment call) |
| **B7 [2D-TWIN]** | ADOPTED (`d4765539`, ledger row #253) — structural-consistency, PROVISIONAL | A4 (ratify after wave-3 readout + `off` arm) |
| **B8 [CAL]** | 8.2 DESIGNED, S1–S5 NOT BUILT; F (width factor) unmeasured | rank 3 next tree, 130–475 CPU-h local |

**17 author items (none pre-approved — every input post-dates rows #221–223):**

| # | tag | one-line question |
|---|---|---|
| A1 | [RULE] | mass-window design: adopt log k=3 / keep linear k=1.5 / commission a k-scan first? (folds WGEOM §9 F-ii) |
| A2 | [RULE]×4 | PA-HIER-31: CoR-P vs CoR-M smear-form authority (a,b); ratify the 5.7e-4 parity diagnosis (c); secant/z-bin amendments (d) |
| A3 | [DO]+[RULE] | authorize the Σ^φ(θ) divisor-fix + sky-cone-flag gate presentation as next tree's node 1 |
| A4 | [RULE] | ratify `mz_sel`/`eff` as production default after the wave-3 readout + `off` arm, or revert to `off` |
| A5 | [RULE] | re-grade G7 row 16 (population prior); retire rows #137/#138 as citations |
| A6 | [RULE] | S0-B: launch only after the divisor fix, or launch now REPORTED-ONLY with the post-hoc subtraction disclosed |
| A7 | [RULE] | CMEM pooled-observation note (two fleets, deficit-direction): leave unregistered or open a registration |
| A8 | [RULE] | R2c: bank-and-park, or a ≥90%-power follow-up (≈30 mirror seeds × 2 arms, ≈15 CPU-h) |
| A9 | [RULE] | B6.1 judgment call: confirm the raw-z (not z̃) reading as authoritative |
| A10 | [RULE] | does B4.3's per-candidate hook need the full `/physics-change` gate or a lighter instrumentation guard? |
| A11 | [RULE] | row #167 D̃_φ sub-convention: COMPLETED-MATERIAL (+0.0344) or COMPLETED-SMALL (−0.0028)? |
| A12 | [RULE] | implement `catalogue_numerator_survival_2d_center="auto"`, and does it need its own gate? |
| A13 | [DO] | archive the local-only registered-run artefacts: git-force-add / archive-only / both? |
| A14 | [DO] | zero-compute housekeeping batch (placeholder row-#, log-text reconciliation, citation fixes F1/F2/F6/F8, GitHub-rejection note, the ≈8.6 CPU-h unbanked line, re-word A22 "tracked tree clean") — approve as a batch? |
| A15 | [DO] | Stage P costing stays moot until A3/A6 resolve — confirm? |
| A16 | [DO] | S0-R stays FALLBACK/DISARMED, not scheduled for a future session — confirm? |
| A17 | [STANDING] | issue a new standing grant for the next tree (§7 ranking) — what scope, what lapse condition? Nothing below runs without it. |

**10 orchestrator path decisions (P1–P10), open to veto, not approval requests:** P1 B1 stops at
1.1, C1 unsubmitted · P2 B2 parked, A2 untriggered · P3 B3 closed premise-refuted, C2 struck ·
P4 B4→4.3, no B1 merge · P5 B5→C3 as registered, no k=3 in wave 3 · P6 B6 closed at depth 1 ·
P7 B7→C4→7.3 adoption opened+executed, batched into wave 3 · P8 B8 harness designed only · P9
wave-2 = C0+C3+C4 first (C1 held), wave-3 = C0′ + two 41-task blind arrays · P10 docket-1
deviations (B1 three corrections, C2 struck, 13 tasks not 16) — argued with numbers, confirmed
by the verifier (item 17).

## 3. Resume recipe (one line)

Present the verifier report + §2's tables to the author → on their words: run the wave-3
sequence (pull → preflight → `DRY_RUN=0 submit_wave3.sh` → retrieve → C0′ gate → blind readout vs
`T_mat=0.008` → verifier part 2 → rule A4) → execute A13/A14 → open the next tree under A17's
grant, ranked per docket-2 §7 (1. Σ^φ(θ) fix + S0-A re-cert + S0-B; 2. B4.3; 3. B8.2; 4. B7 (ii)).

## 4. Standing rules carried (do not re-learn)

Verifier output is evidence, not authority · subagents never run the registered measurement they
built · never end a turn to wait on an untracked process · per-poll SSH, Monitor for watchers ·
every submission stamps its authorization · exoneration grep is for the MECHANISM, not the tag ·
banked ✓VER rows can be internally inconsistent — check a row's own counts before escalating ·
row #223: production changes inside the tree are covered too, every gate still goes to the end
verifier.

**New gotchas from fan-out 1:** workflow tier-lint mis-parses parentheses inside prompt strings
— route such calls through `S()`/`T()` helpers · no backticks inside workflow template strings ·
`pkill` patterns can self-match the caller's own command line — build the pattern at runtime ·
`multiprocessing.Pool` workers are daemonic — a driver cannot nest `evaluate()`'s inner pool
inside an outer seed-level pool (`--jobs>1` dead in `hier_s0_driver.py`, use `--jobs 1`) · GitHub
rejects blobs >100 MB — `posteriors_with_bh_mass/` is gitignored, no ledger row named the
rejection until A14 closes it · SSH `ControlPersist` is 8 h and OTP-gated · the 1588-event iiib
venue's per-h anchor is **≈5–7 min** measured (16 cpus), not `LAUNCHING_JOBS.md`'s 56–76 min
(a 3355-event anchor) — re-cost from the measured range on this venue.

## 5. Rulings of record carried into this session (closed — read before presenting the docket)

- **Row #221:** Arc Follow-ups approved as recommended (the six items feeding B1–B8). **Row #222
  [STANDING]:** the fan-out charter grant — continue through every node on orchestrator
  judgement, one docket per wave, an independent verifier pass at the end as the author's check.
  **Lapses at the verifier pass, by its own text — does NOT cover anything below this line.**
- **Row #223:** production changes inside fan-out 1's tree are covered by the same grant; every
  physics gate still presents before code and still goes to the end verifier.
- **No ruling exists yet** on A1–A17 or P1–P10 (item 20 deferred) — this runbook's job is to put
  the report and §2's tables in front of the author before §3's resume recipe executes.

## Cluster storage incident (2026-08-30 ~09:00 CEST, orchestrator)

bwUniCluster Lustre `/pfs/data6` (home + `/opt/bwhpc` software stack) reports `ost 5 is inactive: rc = -100` (dmesg). Consequences measured: ~12 % of tracked repo files, 10/70 package `.py` files, 43/300 sampled venv `.so` files and 15/127 core stdlib files of the module Python (`encodings/utf_8.py`, `libpython3.13.so`) are unreadable; `git fetch` fails on unreadable objects; the module Python cannot initialise. The workspace `/pfs/work9` is HEALTHY (wave-2 outputs intact; C4 provenance extras retrieved; catalogue md5 c52c13b5… and CRB md5 9a1f2a14… verified by full read). Prepared for recovery: a shallow clone at `4159fc28` in `$WS/darksiren-emri-wave3` with the catalogue copied in (md5 verified); its `uv sync` (workspace-local `UV_CACHE_DIR`/XDG dirs) is blocked only by the module Python. Wave-3 submission WAITS for the OST. No compute lost.

