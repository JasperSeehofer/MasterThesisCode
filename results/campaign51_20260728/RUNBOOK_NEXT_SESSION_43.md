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
