# Runbook — next session(s) (written 2026-08-14, supersedes RUNBOOK_NEXT_SESSION_9)

RUNBOOK-9's Track A is DONE and thread 17's second half is DONE. The venue-transfer campaign was
read out (**TRANSFER-CONFIRMED**, author-ratified 2026-08-13, ledger row #99, commit `d45fbf15`),
the author opened the `/physics-change` gate verbatim (*"I want to open the physics change"*), and
an overnight mechanism-isolation study ran to fill the gate's missing item: **which estimator term
produces the +1 × σ_z displacement of H₀?**

**READ THIS FIRST, BEFORE ANYTHING ELSE IN THIS DIRECTORY:**
`results/mechanism_study_20260813/CAMPAIGN_REPORT_20260814.md` (commit `66141e08`) — the A7 campaign
report for the whole arc, with the dose-surface figure. It is written to be read cold, it is
`presented, not adjudicated`, and everything in §1 below is drawn from its §10 decision table.

Everything from the overnight session is **committed and pushed**. Nothing is running on the cluster.

---

## 0. State of the physics (post rows #99–#101)

- **The defect is confirmed in production realism** (row #99). Decision cell T-c(0.730), N = 400,
  1D: MAP bias **+0.037237 ± 0.000230**, HPD coverage **0.000/0.000/0.000** (0 of 400), PIT–KS
  **1.000** saturated, `post_sd` median 0.004376 ⇒ displaced by **8.5×** its own claimed width,
  rails 0.000, R_dose 0.891. The ladder found **no killing axis**. T-0 (σ_z = 0) puts 200/200 seeds
  exactly on truth — the apparatus is unbiased.
- **Reproduced independently by the null arm** at N = 100: MN0X **+0.037250 ± 0.000494**,
  |Δ| = 0.000013 against the campaign, coverage 0/100, PIT–KS 1.000, bias/post_sd 8.49.
- **Four of six candidate mechanisms are closed analytically at L0**, before any instrument run:
  **M3** (truncation window) short by 6.2e4 with the *wrong dose trend*; **M4** (α σ_z-blind) — the
  "missing" term is identically 1, and deleting α outright still leaves +0.0165 keyed to σ_z;
  **M1** (missing w_pop volume prior) has the **wrong SIGN** (biases H₀ low); **M5-as-stated** fails
  on attribution (76 % of the bias survives an *unscattered* population); **M2** by the clean T-0
  anchor. Notes: `M1_*.md`, `M3_*.md`, `M4_*.md`, `M5_*.md`.
- **THE PARITY ARGUMENT — the constraint that outlives all of them.** Gaussian convolution is
  exp(σ²∂²/2), an expansion in **even powers of σ only**, so every "we convolved wrong" story is
  O(σ_z²) and predicts R_dose ∝ σ_z — a **3.5×** change across the B1→B2 dose lever. **Measured:
  0.92** (fitted exponent 0.93). **A surviving mechanism cannot be a symmetric smoothing of any
  kind.** It needs genuine first-order structure at scale σ_z — a support edge, the argmax
  operation, or a host/impostor asymmetry inside the ball window.
- **The shape is gate × amplifier, not a product.** Split-dose arms: MEI (impostors only)
  **+0.000000**, MEH (host only) **+0.004000**, MN0 (both) **+0.034667** — strongly NON-ADDITIVE
  (residual **18.40σ** vs MN0, **45.67σ** vs MN0X); the split recovers only **11.5 %** of the null.
  The 16-cell 2-D scan measures the same residual on fresh seeds: D(1,1) = +0.033667 at **23.4σ**.
  The entire **f_host = 0 row is EXACTLY zero at every impostor dose** (60/60 seeds, degenerate
  posterior, sd exactly 0) — an absolute gate; the impostor sea carries ~85–88 % but **only once the
  gate is open**. **Both registered shapes are REFUTED**: H-INT at **+10.33σ above its own point
  prediction** (registered SE), H-THRESH at **17.96σ / 50.18σ**.
- **Adversarial verification: both readouts CONFIRMED**, independent reimplementation from raw
  `ln_post`, **max deviation exactly 0.0 across 425 seeds**; the parent readout likewise 0.0.
- Paper #47 hold reason unchanged.

### The three things that matter most — do not read past these

1. **THE STUDY DID NOT ANSWER ITS OWN TITLE QUESTION.** Both the parent arms and all 16 scan cells
   vary a **generator-side dose**. Not one arm and not one cell ablates an estimator formula. What
   is established is an **input condition** (host-redshift exactness) and a **shape** (gate ×
   amplifier) — **not a term**. **M2′ — missing measure/Jacobian *inside* the z-integral, acting at
   the integrand peak — is the register's only unrun candidate**, and it is the destination DS-M5's
   own refutation clause names. **The `/physics-change` new-formula slot is EMPTY.**
2. **TWO INDEPENDENTLY DRAFTED PRE-REGISTRATION TREES EACH FIRED A BRANCH WHOSE MEANING CLAUSE HAS
   NO REFERENT.** The scan's DS-D3 is a **one-sided threshold with no upper edge**, so it returns
   SHAPE-INTERACTION for any large value *including values that refute the hypothesis it names*.
   The parent's branch 2 (SINGLE-OWNER) is satisfied by **MEI — an arm registered as
   zero-estimator-change, generator-side**, so *"that term is the identified mechanism"* has nothing
   to point at. A third face of the same root: the **asserted ±0.002 V-M1 window**, never derived,
   ~21 % false-fail under an exact null. Proposed remedy: **amendment A8 — NOT ADOPTED, PENDING
   AUTHOR APPROVAL** (`docs/RESEARCH_CYCLE.md` row A8 + `docs/gates/BRANCH_REFERENT_FAULT_20260814.md`,
   commit `cd9c610e`). This is a methodological finding in its own right.
3. **ABORT (d) IS THE STUDY'S CLOSEST CALL.** The L0 toy predicted +0.0247 for MEI where the
   instrument measured **exactly 0.000000** — a **100 % magnitude disagreement**. The registered
   wording fires on a disagreement **in sign**, and zero has no sign, so on the literal reading it
   does **not** fire. **If it is deemed to fire, the study STOPs and every L0 closure (M3, M4,
   M5→M5′, W1) reopens** — all four rest on that same toy.

---

## 1. Author ratification bundle (open — this is the session's first business)

Build the review against `CAMPAIGN_REPORT_20260814.md` §10 (tags per CLAUDE.md approval scope:
**[DO]** authorizes work · **[RULE]** rules on evidence already present · **[STANDING]**
pre-authorizes a class). **Present; do not self-adjudicate.**

1. **[RULE] The parent's branch-2 (SINGLE-OWNER) ruling — PRESENTED, NOT RULED.** Two readings on
   the table: **(a)** branch 2 fired, consequence clause recorded as *inexecutable*, no term named;
   **(b)** a generator-side arm was never eligible for DS-M1, so the count is 0 and **branch 4
   (NO-OWNER)** is the reading — which routes to the registered NO-OWNER handling (*"the register is
   exhausted, not the question"*) and a mandatory Stage-L literature sweep before any further arm.
   Either way **no term is named and no repair is proposed.**
2. **[RULE] Whether abort (d) is deemed to fire** (see §0 item 3). This is the only item that can
   void §3–§6 of the parent readout.
3. **[STANDING] Amendment A8** — branch-referent check and two-sidedness check proposed
   **BLOCKING**; band-derivation disclosure proposed **NON-BLOCKING** (A6's spirit). The author must
   set its scope and when it lapses.
4. **[DO] Whether arm A-M2′ gets built** — ≈15–25 seeds ≈ 15–25 CPU-h at the realized
   0.969 CPU-h/seed anchor, inside the unspent L1 budget (3 of ≤ 5 arms used). **The only arm in
   this thread that would alter an estimator term.**
5. **[RULE] Whether the three recorded design faults stay recorded or are amended prospectively**
   (±0.002 window, DS-D3 one-sidedness, branch-2 mismatch). Retroactive re-scoring is barred by the
   anti-tuning clauses in every case.
6. **[RULE] The open disclosures** — accept as disclosed, or order closures: **D-A1-3** (MEH/MEI ran
   at the pre-refactor instrument commit `e83ed0b9`; the `"host"`/`"impostors"` paths have no stored
   cross-commit determinism check — closing it means one cross-commit determinism run); the
   **V-D5 header over-claim** (header says PASS, body is strictly NOT-EVALUABLE); the
   **convention-fragile f_h = 0.5 dip** (−2.93σ MARGINAL at ddof = 1 vs **−3.034σ RESOLVED** at
   ddof = 0 — no account may lean on it either way); the **§4.6/§4.7 contradiction**; the
   **`dose_scales` naming deviation** (single tuple vs the registered two `VenueConfig` fields;
   semantics identical). **D-A1-2 is CLOSED** by the V-M5 artifact.
7. **[RULE] Whether V-M5 and A1-PASS are ratified as verdicts of record.** **This is the only
   dependency that can overturn the whole readout** — declining A1-PASS fires branch 1
   (STUDY-CONFOUNDED) on the N = 15 arm and voids every mechanism measurement in the thread.
8. **The `/physics-change` gate itself.** Intake dossier is **complete on the old side**:
   `PHYSICS_CHANGE_INTAKE_DOSSIER.md` (commit `ee5815f9`) — §1 the old formula written exactly with
   per-symbol provenance, §3 sixteen constraints **C1–C16** any candidate must satisfy mechanically,
   §4 what the package still lacks. **The new-formula slot is empty and author-gated by CLAUDE.md.**
9. **Carried forward, still open from RUNBOOK-9 §1:** items (i)–(vi) of the 08-11 continuation
   record; N-2 adoption; ledger renumber item 6 (recommend §1b fork); DS-5 F5 matched-population.

---

## 2. What the next measurement is, if the author says go

**A-M2′ is the whole queue.** Registered in `PREREGISTRATION_MECHANISM_ISOLATION.md` §1 (candidate
register) as *missing measure/Jacobian inside the z-integral, acting at the integrand peak*, cited at
`darksiren_emri/validation/venue_transfer.py` around the marginalisation quadrature. Ground rules
inherited from the thread and **not to be renegotiated at readout**:

1. It ablates an **estimator term** — so V-M2/AR-1..AR-3 generator invariance is the wrong check for
   it; the pre-dose realisation must stay bit-identical and only the estimator path differs.
2. **Register it under A8's proposed discipline even if A8 is not yet adopted** — name, per branch,
   the arm that can satisfy it and what that arm ablates; two-side any rule that names a point
   prediction; state each band's derivation and its false-fail rate at that arm's own N.
3. L1 = 15–25 seeds. Budget headroom: 2 of 5 L1 arms unspent; no L2 arm has been run or requested.
   Reserved-and-unconsumed seed blocks: **+46000…+46399** (A-M5b, withdrawn at registration) and
   **+47000…+47399** (O2). Never post-hoc.
4. The parity argument (§0) is a **live pre-data constraint on the candidate**: if the proposed
   M2′ correction is a symmetric smoothing, it is already refuted before the arm runs.
5. If branch 4 (NO-OWNER) is the ruling on §1 item 1, the registered handling requires a **Stage-L
   literature sweep before any further arm** — that comes first, not A-M2′.

---

## 3. Filler-task menu (while the ratification bundle is with the author)

3.1 **Book ch14 — NOT DONE.** There is no `book/generators/gen_ch14.py` and no `book(ch14):` commit
    anywhere in the history. It was RUNBOOK-9 §3.2 and it did not land overnight. The arc it should
    cover is now larger and better documented than when it was queued: calibration gate v2 →
    venue transfer (row #99) → mechanism isolation (rows #100/#101) → the branch-referent fault.
    `CAMPAIGN_REPORT_20260814.md` is a ready-made source; ch12/ch13 are the structural template.
3.2 **Delete or archive `oldrepo/`** — a 150 MB untracked full copy of the pre-rename checkout is
    sitting in the repo root (from the 2026-08-13 local directory rename). It is untracked and
    gitignored-by-omission, but it doubles every repo-wide `grep`/`find` and it already polluted
    searches during this session. Confirm nothing unique lives in it, then remove it.
3.3 **Rebrand checklist §1 is now factually stale** — `docs/REBRAND_MIGRATION_CHECKLIST.md` §1 still
    reads *"Not done. The local checkout stays at /home/jasper/Repositories/MasterThesisCode"*, but
    the rename **has happened** and memory **did** carry over (see §4). Update §1 to CLOSED with the
    verification evidence, and re-check §2's file-by-file list against the cluster rename that
    landed in `4d0d827b` / `54c719a6`.
3.4 **Paper #47 verdict-independent sections** — methods (pre-registration discipline, gate design),
    the dissolved-threads narrative (rows #96–#97), and now a genuinely publishable methodological
    result: the branch-referent fault, two independent instances in one thread.
3.5 **GitHub**: 11 issues now carry the "Paper Submission" milestone; #52/#51/#44/#42/#41/#40/#39
    are the open physics/paper-blocker set. #4 (wCDM guard PR state) is still unreviewed.
3.6 **Working-tree hygiene**: `book/generators/*` (14 files), `book/design/*`, two `.claude/skills/`
    files and `darksiren_emri_test/bayesian_inference/test_posterior_combination.py` have been
    sitting **staged-modified** across several sessions. Either land them or reset them.
3.7 W1 (rate-weights) / O2 (volume_deconv) reserved arms — untouched, seeds reserved, never post-hoc.

---

## 4. Gotchas (new or re-confirmed since RUNBOOK-9)

- **Use `.venv/bin/python`, NOT `uv run`.** After the repo rename the venv's console-script shebangs
  are STALE. Run `.venv/bin/python -m pytest -m "not gpu and not slow"` and `.venv/bin/python -m mypy`.
- **Retrieval: `rsync -az --exclude='*.md'`, always.** A plain `rsync` from the cluster can silently
  **revert append-only registered documents** (preregs, amendments, ARMS.md) to their cluster-side
  state. The exclude is deliberate and is registered discipline, not a convenience.
- **`git pull` on the cluster aborts on untracked outputs already committed from the dev box**, and
  git **TRUNCATES** the "would be overwritten" list — so the move-aside must **loop** (67 files, then
  20, on the rename migration). **md5-verify byte-identity BEFORE moving anything.**
- **`sbatch --test-only` ignores backfill.** Twice now it predicted a start ~4 days out for jobs that
  started in ~30 minutes. Registered as **non-predictive** for this job shape; keep logging
  probe-vs-actual pairs (EXP-61 discipline).
- **Unit economics have moved: ~0.969 CPU-h/seed measured**, ≈3.9× faster than the stale 3.79 CPU-h
  anchor (plausibly the ratified Route 1 adaptive Gauss–Hermite contraction reaching the validation
  stack). The whole mechanism programme was **≈258 CPU-h**. Re-cost anything quoted against 3.79.
- **Cluster repo is `~/darksiren-emri`** (renamed 2026-08-13, venv rebuilt, preflight READY —
  `54c719a6`). Workspace expiry **2026-09-23** is inherited from RUNBOOK-9 and not re-verified this
  session — run `ws_list` before assuming it.
- **Rebrand checklist §1 — the open question is now ANSWERED, in the affirmative.** A fresh session
  in the renamed local directory **does** retrieve prior project memory: the memory index and all
  entity files are live under `~/.claude/projects/-home-jasper-Repositories-darksiren-emri/memory/`.
  The checklist text has not caught up (see §3.3).
- **Arms sit at two instrument commits** (`e83ed0b9` for MN0/MEH/MEI, `3aedbe55` for MN0X + scan,
  which refactored the dose from a boolean mask to continuous scales). Every DS-M1/DS-M5 comparison
  is within-commit; A1-DET showed the refactor bit-inert on the `"all"` path (0.0 deviation, 15 seeds
  × 44 fields, 15/15 shared seeds, bit-identical). Label any cross-commit use.
- **Do not trust an `aggregate` block** in a results JSON — every readout in this thread recomputes
  from the raw per-seed `ln_post` and compares against `aggregate` as a check, never reads it.
- Carried from RUNBOOK-9 and still true: instruments write JSON only at the end (timeouts leave
  NOTHING); dev_cpu_il QOS is MaxSubmit 4 / MaxRunning 1 / 30-min cap; `scontrol` walltime extension
  is denied for users; silence ≠ death for background agents.

---

## 5. Provenance quick map (thread 17, both halves)

| artifact | path | commit |
|---|---|---|
| Venue-transfer pre-registration | `results/venue_transfer_20260811/PREREGISTRATION_VENUE_TRANSFER.md` | `e77eecad` (instrument `2ece8801`) |
| Venue-transfer readout (TRANSFER-CONFIRMED) | `results/venue_transfer_20260811/VENUE_TRANSFER_READOUT.md/.json` | `d45fbf15` |
| Mechanism-isolation pre-registration (parent) | `results/mechanism_study_20260813/PREREGISTRATION_MECHANISM_ISOLATION.md` | `891109e1`, REGISTERED at `73141160` |
| Amendment A1 (V-M1 null at N = 100) | `results/mechanism_study_20260813/AMENDMENT_A1_VM1_NULL_AT_N100.md` | `73141160` (author-ratified 2026-08-13) |
| 2-D dose-scan pre-registration | `results/mechanism_study_20260813/PREREGISTRATION_2D_DOSE_SCAN.md` | `73141160` |
| Instruments | `darksiren_emri/validation/venue_transfer.py` + `ARMS.md` | `067ecc19`/`1c5459bc`/`e83ed0b9` (arms) · `3aedbe55` (MN0X + scan) |
| Raw arm + cell data (470 seeds) | `results/mechanism_study_20260813/*_h0p730_results_seeds*.json` | `9fd0386b` (arms) · `5b0bd17a` (MN0X + scan) |
| Readout — parent (branch 2 SINGLE-OWNER, DS-M5 refutes M5′) | `MECHANISM_ISOLATION_READOUT.md/.json` | `f0817dfe` |
| Readout — A1 (A1-PASS, \|Δ\| = 0.000013) | `A1_READOUT.md/.json` | `94c0480a` |
| Readout — 2-D scan (branch 2, meaning barred) | `SCAN_READOUT.md`, `score_2d_scan.py/_output.json` | `94c0480a` |
| Readout — adversarial adjudication | `adjudicate_mechanism_study.py/_output.json` | `f0817dfe` |
| V-M5 values golden (rtol ≤ 1e-12, closes D-A1-2) | `VM5_GOLDEN_20260814.md/.json`, `verify_vm5_golden.py` | run at HEAD `94c0480a`, artifact `38465df8` |
| `/physics-change` intake dossier (C1–C16, empty new-formula slot) | `results/mechanism_study_20260813/PHYSICS_CHANGE_INTAKE_DOSSIER.md` | `ee5815f9` |
| A7 campaign report + dose-surface figure | `results/mechanism_study_20260813/CAMPAIGN_REPORT_20260814.md`, `fig_dose_surface_20260814.png/.pdf/.json`, `plot_dose_surface.py` | `66141e08` (+ gitignore allowlist `53787fd5`) |
| Ledger row #99 (TRANSFER-CONFIRMED, ratified) | `.../gate_b_20260730/BIAS_HISTORY_LEDGER.md` §Row #99 | `2be19dc7` |
| Ledger rows #100 + #101 (A1-PASS · scan branch 2) | same file, §Row #100 / §Row #101 | `94c0480a` |
| Ledger row #100 addendum (D-A1-2 CLOSED) | same file | `1e4bf7ca` |
| Ledger row #100 addendum 2 (parent readout FILED, presented not ruled) | same file | `4f79f6e4` |
| **Amendment A8 (PROPOSED, NOT ADOPTED)** | `docs/RESEARCH_CYCLE.md` row A8 + `docs/gates/BRANCH_REFERENT_FAULT_20260814.md` | `cd9c610e` |
| Approval-scope convention ([DO]/[RULE]/[STANDING]) | `CLAUDE.md` | `804b4c5d` |
| Cluster rename executed (repo → `~/darksiren-emri`) | `docs/REBRAND_MIGRATION_CHECKLIST.md` §2, `cluster/*` | `4d0d827b`, `e83ed0b9`, `54c719a6` |

Also landed overnight, not in the thread proper: GitHub hygiene (#54 and #55 closed with
ancestor-verified evidence; the "Paper Submission" milestone applied to 7 previously-unmilestoned
physics/paper-blocker issues, now 11 in the milestone); `docs/LITERATURE_WARNINGS.md` verified
already conformant and left unchanged.

---

## 6. Resume recipe (fresh session, cold start)

1. `git log --oneline -12` — expect `53787fd5` at HEAD (or a descendant). `git status` should show
   only the long-standing `book/`-and-tests staged set plus untracked `results/` and `oldrepo/`.
2. Read `results/mechanism_study_20260813/CAMPAIGN_REPORT_20260814.md` end to end. It is the
   handoff. Its §10 table is the ratification bundle; its §9 is the disclosure list.
3. Nothing is running on the cluster. Do **not** re-arm monitors. If you want the state anyway:
   `ssh bwunicluster 'squeue -u $USER'` (expect empty) and `ssh bwunicluster 'bash -s' < cluster/preflight.sh`.
4. Put §1 in front of the author **as a reviewable artifact**, not a chat summary (CLAUDE.md
   "Proposing decisions"). Items 1, 2 and 7 are the load-bearing ones; item 7 gates the rest.
5. Record the author's words **verbatim** in the ledger, with any itemization explicitly marked
   orchestrator-derived — the attribution-precise form row #99 uses.
6. Only after item 4 in §1 is granted: pre-register A-M2′ under §2's ground rules. Do not open the
   `/physics-change` new-formula slot without the author; the gate is author-gated on its face.
7. Filler while waiting: §3 — ch14 first (it is the only queued deliverable that is purely
   verdict-independent), then `oldrepo/` cleanup and the rebrand-checklist correction.

**Standing constraint for any session touching this thread:** registered documents under
`results/mechanism_study_20260813/` and `results/venue_transfer_20260811/`, the ledger, and anything
under `paper/` are append-only or read-only. Bands were locked at pre-registration and are not to be
moved after a readout — the three known-defective rules stay recorded until an amendment fixes them
**prospectively**.
