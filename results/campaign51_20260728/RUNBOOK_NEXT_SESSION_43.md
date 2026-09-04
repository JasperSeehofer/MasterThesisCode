# Runbook 43 — after the overnight wave-3 session of 2026-09-03 (supersedes runbook 42)

**Read first.** Runbook 42's §7 device transfer is MOOT (author stayed on `thinkpad`). The 2026-09-03
evening/night session (Fable 5.1 chair) executed Research Graph 1 wave 3 to its decide layer and the
addendum's zero-compute set. Rows #334–#347. Nothing is running. Nothing was submitted to the cluster.

## 1. FIRST ACTIONS
1. Put the morning docket in front of the author: `graph1_20260901/MORNING_DOCKET_20260904.md`
   (artifact https://claude.ai/code/artifact/3cd5efee-f4d8-4cbf-977e-788698662305, dark-first, same URL as
   the wave-3 docket). Thirteen [RULE]s R1–R13, each with the chair's flagged provisional reading. The
   docket-2.2 STANDING grant of 2026-09-03 LAPSED at the author's next message — do not launch anything
   on it.
2. Check `exec/END_VERIFICATION_20260903.md` for the completion-residual APPENDIX (requested at ~23:30;
   may have landed after the last ledger row). If it lists a discrepancy that changes a booking, append
   an erratum to `exec/r-completion-residual/CHAIR_REDERIVATION_20260903.md` and a ledger row BEFORE
   the author reads R12.
3. SSH keepalive loop (scratchpad `ssh_keepalive.sh`, pid in the session) — kill it or let it die with the
   box; the ControlMaster socket needs author OTP re-auth after 8 h idle anyway.

## 2. State of record (rows #334–#347)
- Reads DONE, verdict-free, chair-rederived, end-verified: rd-s3-readout (DEFECT-SIGNATURE at N=200,
  catalogue-hosted class, rails 15–24 %), m-s0b-production (LEVER-LIVE/MIXED/POWERED; truth node ≠
  production comparand on 562/1588 events — OPEN), m-cone-loss (IMMATERIAL on the leave-out number,
  φ 7.8 %), m-completion-residual (INTERMEDIATE (b), ρ = 0.26: T_prod −0.197 ± 0.019, T_harn −0.051 ± 0.007).
- Addendum (J–N) proposal written (`GRAPH1_ADDENDUM_PROPOSAL_20260903.md`), NOT ratified; its
  zero-compute reads ran (J.1 explained-by-design; J.2 confounded — no matched comparand; K offset carried
  by 3–6 % of events; L.1 timeout rate 12σ mass-dependent) and two registrations drafted (r-sealed-mock,
  r-b0-finite-moment + B-R control DISCRIMINATING).
- Process incident row #340 (gate reviewers unblinded G/H statistics — chair prompt); mitigation path
  applied; author's R7 decides accept vs discard. Memory: gate-reviewers-must-not-compute-registered-statistic.
- Errata of record: row #345 (nine end-verification discrepancies; cone re-booked; S0-B T-ID unsatisfied;
  F label at n≈180; T-pin post-flip).

## 3. Then-next queue (after the author's replies)
1. R4 comparand re-evaluation (iiib, theta_sites 2.2, θ=(0,1), h=0.73, single task, ≈0.1 CPU-h; local
   `--evaluate` or one cluster task behind preflight) → re-diff vs the S0-B truth node → unblocks R3/R5,
   d-photoz-leverage, and then d-residual-attribution (its manifest is otherwise full).
2. R1 follow-up register node: catalogue-hosted-class localisation (S3) × the 3–6 % subset (K jackknife
   ranking in `exec/rd-2d-bootstrap-jackknife/`) — one top-tier prereg author; this is the Graph 2 seed.
3. R9: fetch `run_20260729_seed61000` `*.log` from the cluster (a READ, not evacuation) → rd-timeout-bin-seed61000.
4. R8 build node: relative-threshold dark-class criterion (g-byte-id on physics; re-derive B3 anchors).
5. GATE-ACC reporting-only relaunch on the cluster (login-node run died: libpython; use `source cluster/modules.sh`).
6. Backlog (author words): 12a archive backup · 12b evacuation before 2026-09-23 · 12c · 12e merge → main · 12f/12g.

## 4. Gotchas added this session
- A gate reviewer told to "re-derive the SE from the data" computes the registered statistic → unblinding.
  Gates check computability/formula only; use holdouts or synthetic tables.
- The exact-zero dark-class criterion flips labels on float noise (157 events between two runs with
  physics identical to 1.6e-7 on those events) — never anchor a population count on `== 0`.
- Two disposition rows can fire at once; apply the registration's own resolution rule before booking.
- `floor(200)` in the S3 harness is really floor(median n_scored ≈ 180); label F accordingly.
- The 2026-08-29 ladder cell-T checkpoints are post-flip (stamped after 5e7fda16) — not a flip-invariance reference.
- The S0-B driver's gate_parity is hardcoded to the b0i banked path (NO_BANKED_CSV for iiib is structural).
- The chair mis-stated the clock once (wrote 00:30 for 21:00) — check `date` before writing a timestamp.
- Ledger clerk convention worked well: one long-lived sonnet clerk, rows #334–#347, quote-verbatim.

## 5. Tiering used (for the record)
Top-tier: chair + wave-3 prereg author + addendum prereg author + end-verifier (≤3 concurrent). Sonnet:
~60 agents (readers, panels, builders, forensics, clerk). Cluster: 0 CPU-h. Local: ≈0.5 CPU-h.

## 6. AMENDMENT after batch 2 (2026-09-04 ~00:20 CEST; rows #349–#359)
**FIRST ACTION of the morning: the author logs in once (`ssh bwunicluster`, password + OTP) to
restore the ControlMaster.** The socket was lost at 23:03 (row #357; root cause + enforced fix in
row #359: `.claude/hooks/ssh-guard.py` + `cluster/agent_ssh.sh` — every agent cluster call goes
through the wrapper now; one ops agent per batch).
Then, through ONE ops agent using `cluster/agent_ssh.sh`:
1. `poll 6790794,6790859,6790465` → retrieve S0-C (`$WS/graph1_s0c_hgrid_20260904/h_0p665|h_0p780`,
   exclude injection-pool symlinks), sealed m1 iiib (`$WS/graph1_sealed_m1_iiib_20260904`; then the
   joint_r1 switch only if iiib ≤ 60 core-h), GATE-ACC (`$WS/p3_2d_fleet_aprime_20260902/gates_{bt,bc}.json`).
2. Submit R4b (`cluster/graph1_r4b_comparand_sites22_2doff.sbatch`, prepared, ~6.5 min) → diff vs
   the S0-B truth node (byte-identical ⇒ S0-B was measured on the pre-[P3-2D] counterfactual).
3. Fetch the pool build logs for r-timeout-selection Q1 (NOT-covered [DO] A in its §8).
Batch-2 state of record: r-offset-subset built (blind table + influence vector byte-id 30/30; scorer
in fix round 3 — check exec/r-offset-subset/READ_RECORD.md exists); r-timeout-selection drafted
(rows #358; p0 axis withdrawn — D1 bound; per-draw timeout 0.92 %); R8 build done, migration HELD
(R14); S0-C addendum + sealed-m1 dossier ratified-by-chair under the batch-2 grant.
Morning docket additions: R14 (class definition), the R4/R4b provenance question for S0-B, R9 downgraded.

## 7. Session close (2026-09-04 ~03:40 CEST; last row #366)
Batch 2 closed. Open in the morning, in this order: (1) author OTP login → `cluster/agent_ssh.sh poll
6790794,6790859,6790465` → retrieve S0-C, sealed m1, GATE-ACC → submit R4b; (2) the docket's R1–R18
(R18 = launch r-timeout-selection Q2: exec/r-timeout-selection/, gate rev2 RED only on F6, closed by
the chair erratum; rerun the gate→build→read chain, zero compute); (3) Q1 after the pool build-log
fetch. Everything else is recorded; nothing is running locally. Keepalive loop: kill it
(`pgrep -f ssh_keepalive.sh`) — it only logs FAIL until the socket is restored.

## 8. Morning batch of 2026-09-04 (rows #367–#385) — state at ~16:40 CEST
Author disposed R1–R18 as tagged ("all approved, granted, ratified"). Executed: ratification writebacks
(#368); r-highz-completion registered → 4 gate rounds → READ: TERM-OWNS(B), harness reproduces
(ESTIMATOR-INTERNAL candidate) → RULE R24 (#369, #378, #381, #384, #385); S0-C read (#371; Δh_θ
saturates; conditional on R21); S0-B driver difference found = H_BOUNDS (0.50, 0.86) vs production
0.60 (#372) → --h-bounds flag (#375) → truth-node test job 6794615 (QUEUED at close; #382) → if
byte-identical to R4b, submit cluster/graph1_s0b_production_window.sbatch (5 nodes) and re-read;
timeout Q2 READ: POPULATION-MISMATCH-MATERIAL −0.011 (#374, R19); Q1 PARKED after 3 design rounds
(#383, R20); sealed m1 resubmitted 6794421 with a content guard (#373; tasks completing); GATE-ACC
6790465 still PENDING; R8 build → RULE R14 (#350); pre-commit hook installed (#376).
Next session: (1) poll 6794615 / 6794421 / 6790465 via cluster/agent_ssh.sh; retrieve; the R4b diff;
the m1 binary read (dossier exec/r-sealed-mock/); (2) docket R19–R24 to the author; (3) end-verification
appendix "MORNING BATCH" (requested; check it landed); (4) b-highz-bnum-factor after R24.

(Session-close note, 16:55 CEST: SSH keepalive loop still running locally (scratchpad ssh_keepalive.sh, 4-min touch); jobs 6794615 (queued), 6794421 (m1, ~24/41 done), 6790465 (pending) unretrieved; nothing else running. Author rulings R19–R24 pending, non-blocking.)
