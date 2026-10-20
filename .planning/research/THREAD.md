# Thread state — EMRI → H₀ dark-siren inference

*Layer-2 thread-state contract (vault: `wiki/analyses/research-routine-packs-spec.md`). Read by
the vault's `/cartographer` bridge. Keep current: same-day update after every experiment run
(vault: `wiki/meta/research-operating-manual.md`, "per numerical experiment / run"). Vault mirror
of the hypothesis rows: `wiki/meta/hypothesis-ledger.md`.*

> **Created 2026-09-21 by mission M65** (first research project wired into the manyfold vault).
> Vault entity page: `wiki/entities/master-thesis-code.md` — note the vault slug is
> `master-thesis-code`, deliberately decoupled from this repo's directory name
> (see `wiki/meta/darksiren-emri-rename-plan.md`). Experiment/hypothesis prefix is **MTC**, the
> prefix already in use in the vault ledger — not a new one coined here.
>
> **Continuity warning.** `STATE.md` (repo root, "Last updated: 2026-08-12") is **stale** and must
> not be read as the thread's frontier: `main` HEAD is 2026-08-26, while the live m1 work sits on
> the unmerged branch `fix/p32d-classg-venue-repair` (HEAD `d201d9e4`, 2026-09-04). The whole
> `graph1_20260901/` tree — 77 commits — **does not exist on `main` at all**. STATE.md still says
> T-1 is "still unrun", which remains true, but it predates the entire m1 exercise.
>
> **Update 2026-10-19 (M65 Round 2).** The m1 docket tree this file cites
> (`graph1_20260901/exec/r-sealed-mock/`, the morning docket, the addendum, the m1 ops records)
> now travels with this file by path allowlist, together with the m1 likelihood CSV and the 41
> posterior JSONs; the author ruled (2026-10-19) that `main` is first fast-forwarded to
> `fix/p32d-classg-venue-repair` (tip `387cf33f`) and this branch lands on top.

## Hard core

**[PROPOSED — awaiting author ratification as a [RULE]; not ratified by M65.]**

The closure of the dark-siren H₀ estimator at the injected truth is a consequence of
**generator-consistent normalisation** (n̂_w = W_cat/V_f + D_gen, plus z-resolved survival
S(d_L|z)) — that is, of the physics of the selection function — and **not** of any
truth-referencing constant, anchor, or tuning introduced by the estimator.

*Source of the formulation:* `H-MTC-genmarg` in `wiki/meta/hypothesis-ledger.md`, promoted to
thread level as kill criterion (1). The `project-init` skill deliberately leaves the hard core as
a fill-in stub because it is the author's to set; the statement above is a **mirror of the
author's own already-written ledger row**, offered for ratification, not a new claim invented by
the mission.

## Live hypotheses (mirror of vault hypothesis-ledger)

| id | statement | discriminating experiment | status |
|---|---|---|---|
| H-MTC-genmarg | The deep-venue closure at truth comes from generator-consistent normalisation plus z-resolved survival S(d_L\|z), not from any truth-referencing constant | **T-1** blind alternative-truth mock: seal h_inj, re-run the production stack, check the MAP tracks the sealed truth | **live** (opened 2026-07-26) — T-1 **unrun**, see Active experiment |
| H-MTC-realdata | The precision the δ-kernel host-z numerator buys survives the move from mock to real data once the photo-z kernel replaces point/point pairing | re-derive the photo-z kernel, then re-measure σ_h against the mock-internal baseline on the same venues | **live** (opened 2026-07-28) — unstarted |
| H-MTC-norm | The H₀ rail is a curable normalisation artifact, not information starvation | run the production estimator under `{global, local_ratio, volume_deconv}` | merged 2026-07-02 (named winner superseded by H-MTC-genmarg) |
| H-MTC-massch | The 2-D mass channel tightens H₀ by ~2× (4%→1.8%) | forecast with realistic host-mass scatter | pruned 2026-07-01 — information-starved |
| H-MTC-vtransfer | The ball-venue calibration DEFECT is a venue caricature, not a property of the production estimator | production-matched venue-transfer campaign | **REFUTED** 2026-08-13 by its own decider (TRANSFER-CONFIRMED, author-ratified); vault status change to `pruned` is **proposed, not applied** — a prune is a kill and the human confirms kills |

## Kill criteria

Thread-level, **written 2026-08-11 by the author** (vault: `wiki/meta/hypothesis-ledger.md`,
[[proposal-queue|PROP-20260811-01]]). Quoted, not paraphrased:

1. **Anti-tuning** — if T-1 (blind sealed-h mock) closes anywhere other than the sealed `h_inj`,
   the `generator_marginal` + `--pdet_z_resolved` adoption **reverts** and the closure claim is
   **withdrawn, not patched**.
2. **Precision** — if the real-data photo-z kernel degrades σ_h to the host-known limit (×3.3–×6.8)
   or worse, every precision claim is **restated as mock-internal** and the manuscript's headline σ
   is withdrawn.
3. **Time-bound** — if no blind-mock verdict exists by **2026-12-15**, the thread ships as a
   **methods** paper with the closure reported as *unblinded*, rather than as an H₀ measurement.
   *(Date borrowed from the workspace copy-off deadline already on the books, not derived from the
   physics — amendable.)*
   *(Date history: written 2026-08-11 as **2026-09-23**; extended 2026-09-21; re-dated to
   **2026-12-15** by author ruling 2026-10-19 — see Status of (3). The rest of the text is the
   author's, unchanged.)*

**Status of (3) as of 2026-09-21:** extended by author ruling on card `overnight-report-2026-09-21`
(*"Explicit extension: run T-1 + adjudicate m1 firs[t]"*), conditioned on running T-1 **and**
adjudicating the m1 docket first. M65 has adjudicated m1 (see Current claims); **T-1 could not be
run** (see Active experiment), so the extension's stated precondition cannot be met as written.
Returned to the author as a `go-kill` review card by M65.

**Status of (3) as of 2026-10-19 — dated rulings (author, via Answer Wire):**

- **2026-10-18** (`ask-20261018-w174804a1ed` Q2): **Continue.** The retrospective recommended a
  formal pause; the author chose to continue. The open kill-gate card `m65-r1-darksiren-manyfold`
  is superseded by this ruling.
- **2026-10-19** (`ask-20261018-w223027afac`):
  **R1** pipeline-first — `M38-cluster-session-unlock` builds first, then agents run T-1 on the
  cluster; **R2** the T-1 deadline is extended to **2026-12-15** (second extension; no third
  without a new ruling); **R3** this thread's branch may land on `main` by the author's tap;
  **R4** `/cartographer`, `/commission` and `/project-init` stay live.
- **2026-10-19** (`ask-20261018-w224256df6b`): **R25** — m1 banked **INTERMEDIATE** (parent
  wording); **R27** — draft the (m2) erratum with the rail clause winning a both-fire, ratified
  before any draw; pool option P2 first, P1 fallback.
- **2026-10-19** (`ask-20261018-w231446a3b9`): **R28 ratified** as a registration erratum — the
  pool is the 707-file `injection_pool_mix200k_20260728`, list-md5
  `75f4030d5d3b0405fd948049bef5767e` (= md5 of the tracked
  `graph1_20260901/exec/r-timeout-selection/POOL_MANIFEST.md5`; also pinned in
  `cluster/datasets.yaml`), contents identical to the 0.73 baseline pool. **R24 — no code change
  before T-1**: the freeze pins the `darksiren_emri/` tree `71acb98b0fd8a0f9d3aad9595eb9fc084cae0b6d`
  (identical at the freeze `081b1f28`, at m1 and at `387cf33f`); R24 (the B_num term) is a
  disclosed limitation, taken up after the blind result. Hazard: `main` before the fast-forward
  holds an older tree (`91446066…`), so (m2) must never be cut from pre-fast-forward `main`.

**The 2026-12-15 gate fires as written unless the author rules otherwise.** If no blind-mock
verdict exists by then, (3) applies: methods paper, closure reported as unblinded.

## Current claims

| claim | status | evidence |
|---|---|---|
| The m1 sealed-mock's registered cell disposition | **UNRESOLVED-AS-REGISTERED** — the pre-registration does not cover a posterior on which TUNED and NOT-TUNED-AT-RAIL both hold; no tie-break invented | M65 adjudication, `.planning/missions/M65-darksiren-emri-into-manyfold/REPORT-R1.md`; raw record `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-sealed-mock/READ_RECORD_m1.md` (branch `fix/p32d-classg-venue-repair`) |
| m1 posterior is centred at 0.628–0.631, i.e. 0.0423 (2D) / 0.0389 (1D) **below** the 0.67 truth, with 71 %/62 % of mass at or below 0.63 | verified twice (chair + M65 independent verifier); cell-free, stands regardless of the R25 ruling | `READ_RECORD_m1.md` §4; M65 `VERIFY-R1.md` §1 |
| m1 makes **no anti-tuning claim** and does not discharge T-1 | registered, explicit | `D_SEALED_REGISTER_DOSSIER.md` §2 (*"the truth is public (no anti-tuning claim can bank from (m1))"*), §1.1 (*"explicitly NOT the T-1 verdict"*) |
| Anti-tuning (kill criterion 1) | **untested** — T-1 unrun | — |
| Precision on real data (kill criterion 2) | **untested** — photo-z re-derivation unstarted | — |
| Every decisive m1 number is prose-only | **defect** — the underlying posteriors and the scorer are not in version control | M65 `VERIFY-R1.md` §3 T1 |

## Active experiment

**T-1 (blind sealed-h mock) — BLOCKED BY TWO INDEPENDENT GATES.** Not running, not scheduled, and
*not* substituted for.

In this campaign's framing T-1 is the **sealed (m2) run**. It has never been launched: the only
sealed-mock exec directory on the branch is `r-sealed-mock` (m1); no `(m2)` directory exists.

**Gate 1 — no cluster credential (infrastructure). As of 2026-10-19 this gate is
`M38-cluster-session-unlock`, ruled first in the arc (R1 pipeline-first) and due about
2026-11-20**; T-1 runs on the cluster after it lands. *(R1 text, 2026-09-21:)* T-1 requires a fresh GPU EMRI simulation
campaign at a newly sealed `h_inj` on bwUniCluster 3.0 (SLURM, `gres=gpu:1`, CUDA 12 —
`cluster/simulate.sbatch`), then a CPU eval array
(`results/redteam_20260726/PHYSICS_METHODOLOGY_REVIEW.md` §4: *"Cost: one simulation campaign +
one eval array"*). The brain host holds no SSH key and no university VPN; the mission built to
solve that, `M38-cluster-session-unlock`, is itself `stage: idea` and unbuilt. The redteam review
rules the CPU-only "cheap partial" (rescaling an existing CSV's d_L) **"NOT a substitute"** — it
*"confirms the §1 mechanism; it does not test the selection model."*

**Gate 2 — the author's own launch bar (governance).** Docket **R27** states: *"no (m2) launch
until ratified"*, pending an erratum that adds a both-fire precedence rule and a stated treatment
for σ_measured < σ_anchor. **This gate is independent of the credential**: even with cluster access
tonight, T-1 could not legitimately launch. Ratifying R27 is an author action that unblocks T-1's
governance side regardless of when the credential lands.
*(2026-10-19: unchanged and still the author's. The direction of R27 is ruled — rail clause wins
— but the erratum is not yet drafted or ratified; the governance package is mission M162.)*
*(2026-10-20: the M162 erratum is ratified on card `m162-r1-t1-governance` — this gate is cleared;
see § T-1 package hashes. Gate 1, M38, remains.)*

**Do not book m1 as T-1.** m1 is a sealed *mock* at an alternative truth, sharing T-1's design but
not its blinding: its truth (h = 0.67) is public and legible in the run directory name
(`run_20260729_seed64000_h0p67`). The register says so in its own words, repeatedly.

*No `EXP-MTC-*` row is pre-registered in `wiki/meta/agentic-experiments-research.md` yet; the next
experiment to pre-register is T-1 itself, once the credential lands.*

## Claim history

`CLAIMS.jsonl` (commission-compatible) is created by the first `/commission --research` run on
this thread; until then the table above is the claim record. Note the repo also maintains a
long-running append-only record at
`results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`
(row #387 is the m1 read), which predates this file and is not superseded by it.

## Last verification

**2026-09-21 — M65 independent verification pass (`VERIFY-R1.md`).** Fresh-context adversarial
verifier, dispatched before any adjudication draft existed. Verdict: the four m1 numbers
(2.73σ / 2.58σ / 4.39 % / 2.59 %) **all reproduce exactly** against the record; the
seal-before-result ordering is **evidenced by git** (threshold block byte-identical across all
four dossier commits, hash `c908790d…`, frozen ~11 h before the first array task); but the
"both cells fire" reading is **CONTESTED, not a fact**, and the underlying posterior data is
**absent from version control**.

## T-1 package hashes (mission M162, 2026-10-19; RATIFIED on card `m162-r1-t1-governance`, relayed 2026-10-20)

The T-1 governance package lives under
`results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-sealed-mock/` and
`scripts/sealed_mock/` on branch `mission/M162-t1-governance-package`. It was cut from the work
branch at `387cf33f`.
- Status: **ratified** by the author on review card `m162-r1-t1-governance` (all six questions;
  rulings verbatim in `.planning/missions/M162-t1-governance-package/publish-round-r1/RATIFICATION.md`).
  The rule blocks below are now append-only: a change is a new dated erratum, never an edit.
- **No draw, no pool, no simulation and no cluster command has happened.**
- How to recompute: run `python scripts/sealed_mock/rule_block_hashes.py <file>`. Each value is
  the sha256 of the exact text between the block markers. After ratification, any edit inside a
  block makes its hash stop matching, and the draw script refuses a params block whose hash
  differs from the value below.

**Rule blocks of `ERRATUM_m2_R27_R28.md`:**

| block | binds | sha256 |
|---|---|---|
| E1-pool-identity | R28: pool of record mix200k_20260728, 707 files, list-md5 `75f4030d…` | `120dc00242710bacc651018df05195356e3826c3af113b82d228e60089241f68` |
| E2-precedence | R27(i): gates ⇒ rail wins (banked bound) ⇒ single band cell, else INTERMEDIATE | `5f6688ed0046d6c87bb7358cbfdef05a96c31600d967a981dfd5790adb04cdc9` |
| E3-sigma-treatment | R27(ii): ρ = σ_h/σ_exp; outside [0.8, 1.25] re-assign at σ_exp; a change ⇒ INTERMEDIATE (σ-sensitive) | `fe8822b9341fffc3611ce9504fab139712fb3301f0f8380f10bb1769ada28f6c` |
| E4-code-freeze | R24: tree `71acb98b0fd8a0f9d3aad9595eb9fc084cae0b6d`, HEAD == H_code strictly, dirty count 0 | `a974f3352abe956239f34f819bf0039b258b56331d50d6edc27e99607722b4c9` |
| E5-pool-option | P2 first (paired, P2-0 ≥ 95 %, 1e-3 gate), P1 fallback ≤ 250 GPU-h | `5b65b897eb0afd2544396756de558ccdf498202df0c8fd39986eec99367dcbd3` |
| E6-read-rule-gates | 55-node grid; whitelist; squeue/sacct only; two gate layers | `20d2f4b91814be242dc297e7c0c454a042ffc591426eca2ba95e630736169813` |
| E7-dated-default | 2026-12-15, Europe/Berlin; what counts as a verdict; no third extension | `75c8e307e64e8ba0e01cc2a3617346cac6d50d60e5332e0c130d7f78a944faf1` |
| E8-disposition | (m2) consequence rows, each a fresh RULE | `944a8ec2a0296598aedaaaf69c76bc2640c7e7503b49d822d23b12fc46c69283` |
| T1-SEAL-PARAMS | the block `draw_sealed_h.py` reads (`--params-sha256`) | `0df280258bc1523f1c5c91cb1fd11745e595843e438dd3175293908c23200715` |

**Companion files** (file sha256 and git blob at authoring; authoritative once committed):

| file | sha256 | git blob |
|---|---|---|
| `ERRATUM_m2_R27_R28.md` | `46ef3846086b7b0ca9c0b3ac761aa73ab8d9e057f88aa2b293768e2da87edb1b` | — |
| `POOL_OPTION_MEMO_m2.md` | `85e34bb1597dcbee354d02d3f5927fa2a1120f8fbdfc230450e959d8877887ca` | `78b97ecc` |
| `UNSEAL_CEREMONY_m2.md` | `012f13adb38af67df72f8b1f87ccb9a82cf88e948a3727eb917ec8a4a825e579` | `6b455a1e` |
| `PREREG_EXP-MTC-T1.md` | `e4458bb55ef5a377f9eaabdd62ed521d2d708216640dc9be7e84856c9fc3fbfe` | `9d26a44a` |
| `scripts/sealed_mock/draw_sealed_h.py` | `23729a186340678ccc31485656505be48493fbe78a95d585e919197c9c5eba50` | `bd74dd54` |
| `scripts/sealed_mock/rule_block_hashes.py` | `83fe78fea77690872145a3167769e1d1948419ef8241ce33173ad4575d6c0b32` | `55469936` |

**Unchanged pins that the package relies on:**
- the m1 dossier threshold block md5 `c908790d…` (M65 `VERIFY-R1.md` §4). M162 appended a pointer
  at the end of the dossier; the threshold block is untouched.
- `darksiren_emri/` tree `71acb98b…`, with dirty count 0 on this branch.

**Effect on "Active experiment" above:** Gate 2 (the R27 launch bar) is **cleared** by the
ratification of card `m162-r1-t1-governance`. Gate 1 (the cluster credential, M38) is unchanged. T-1 is due **2026-12-15**; that is
the dated default, and there is no third extension without a new ruling (the 2026-09-23 date in
kill criterion 3 is superseded by `ask-20261018-w223027afac` R2). m1 is banked **INTERMEDIATE**
(R25, `ask-20261018-w224256df6b`), which supersedes the "UNRESOLVED-AS-REGISTERED" claim row above.

**Ruling on kill criterion 1 (card `m162-r1-t1-governance` Q4):** a (m2) TRANSFER result, a
posterior closing at h_inj − 0.064, **passes** anti-tuning and **does not trigger** kill criterion 1.
The criterion's text above is the author's and is unchanged; this ruling fixes how the erratum's
E-8 TRANSFER row maps onto it. **Q5:** the read-only cluster reads R-1/R-2
(`POOL_OPTION_MEMO_m2.md` §5) are approved inside an M38 window; they are not yet run. R-3 (the P2 pairing seeds) is not named in
the ruling and returns to the author.
