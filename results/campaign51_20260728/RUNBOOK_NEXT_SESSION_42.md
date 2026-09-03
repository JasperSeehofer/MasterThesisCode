# Runbook 42 — graph-1 mid-batch handoff for the Fable 5.1 session (supersedes runbook 41)

**Read first.** The 2026-09-01→03 session executed Research Graph 1 waves 1–2 end-to-end
(rows #291–#330). The session restarts NOW (model upgrade) with TWO measurements in flight and
watchers that DIED with the old session — §1 is the very first thing to do.

## 1. FIRST ACTIONS — re-arm the watchers (nothing else before this)

1. **m-s3 cell S: CLOSED at n_U = 67 — do NOT resume (author ruling, row #333).** Invocation 1
   ended `wall_limited` (67/100 universes, wall 87 016 s); no local process survives. The
   `n_U_min = 60` floor was crossed, so the cell is read-valid. The author ruled "Read out at
   n_U=67 / never invoke cell S again" — invocations 2 and 3 of the frozen rule (r-b82-s4 §3,
   row #301 item 2) are FORFEIT BY RULING, not exhausted. **There is no watcher to re-arm.**
   → go straight to aggregation + `rd-s3-readout` (§4 item 1), which must report n_U = 67, not
   the design target of 100, and must disclose `stopped_reason: wall_limited`. Cell T is
   COMPLETE (25/25, row #326) — do not touch its work.
2. **S0-B production: COMPLETE (rows #330–#332).** Job 6779532, 5/5 COMPLETED; out-root
   `graph1_20260901/exec/m-s0b-production/s0b_run_20260902` on the cluster (925M). Chair-verified:
   production Σφ/Σ_4D loaded, registered config verbatim, single-h design → the ~7.5-min/node cost
   is EXPECTED (the §7.2 anchor priced a 41-node grid; misapplied in item 10 — disclosed).
   **FIRST SCIENCE ACTION for this session: the S0-B read** — retrieve (rsync -aL + md5, exclude
   injection-pool symlinks), then the registered reads: g-score-null |Z|≤3 (rows #225/#251/#287),
   score_b_re secant per PA-HIER-31(d) (denominator 0.066), score_s per the standing convention,
   B0-B disposition per §2.1(e) (LEVER-DEAD-AT-N iff |Z_b|≤3 AND |Z_lns|≤3; materiality
   |b̂|<0.0165; power σ_b<0.0661) — mechanical dispositions only; **d-photoz-leverage returns to
   the author with the numbers**. Reader = fresh sonnet agent; chair re-derives decisive numbers. Re-arm a sacct watcher. On completion: retrieval + the PA-HIER-33 /
   score_b_re reads (g-score-null |Z|≤3 gate per rows #225/#251/#287) → d-photoz-leverage
   dossier to the author. Cost anchor: 74.7–101.4 CPU-h across 5 tasks (~15–23 CPU-h each).
3. SSH is authenticated as of handoff; `ControlPersist` expires ~8h (OTP, author-only re-auth).

## 2. State of record (rows #291–#330, all committed + pushed to origin, HEAD f82f77a8+)

- **Docket items 1–11 all ruled** (author rows #301/#304/#308/#314; chair-under-grant row #325 —
  the author granted "you can make decisions but flag them", every chair decision is veto-able).
- **SETTLED ×2:** F-ii window (not adopted, rows #302/#312/#314) · q-a4-provisional (falsifier
  INSIDE-BOTH row #322, A4 final row #325).
- **[PHYSICS] landed ×2 this session:** a26959b4 (h_grid_admissibility_max decoupling; byte-id
  GREEN row #311; G-EXT 55-node grid complete row #313) · 2b657255 (Option A′ class-G
  S̄_φ de-double-weight; fleet ran on it).
- **m-jr1 Z-CONFIRMED** (zero-compute registered read, row #305; claim promotion at d-calibration).
- **Interactive docket artifact:** https://claude.ai/code/artifact/4319714c-7ff7-4166-a86d-329338c01df1
  (dark-first; update via url param from the new session). Research-interests essay: artifact
  538dd463 (non-record).

## 3. OPEN AUTHOR WORDS (docket item 12, rows #328): 12a backup destination for the 159GB
sole-copy ~/emri-archive (TOP PRIORITY) · 12b cluster evacuation before Sep 23 (~250GB,
0 extensions) · 12c disk culls · ~~12d DATA_INVENTORY commit~~ [CLOSED row #333] · 12e merge→main (112 ahead) ·
12f safe builds (renderer/readout/utilities now; EvaluationConfig after merge) · 12g docs sync.
Health scan: `graph1_20260901/health_scan/` (synthesis + 4 lenses + archive-fix record).

## 4. Then-next queue (graph order)

1. Cell S is CLOSED at n_U = 67 (row #333) → aggregation NOW (`--score-only --population 200` per cell +
   `--score-only-ratio-t-s`, from repo root) → **rd-s3-readout** (three-valued existence;
   the two design-gate caveats routed there: general-n exact-KS computed externally; the
   binom_bands normal-approx-vs-"exact" label, row #303) → F value → **r-completion-residual +
   r-cone-loss authoring** (ONE top-tier prereg author, k-wave2 cap) → d-completion/d-cone-register
   [RULEs] → the measures → **d-calibration** dossier (needs: rd-s3-readout green, rebaseline
   banked-as-comparand ask, m-jr1 disposition) → d-residual-attribution → the three paper rulings.
2. S0-B reads → **d-photoz-leverage** [RULE].
3. GATE-ACC addendum (reporting-only) may still be computing on the login node — check
   `exec/v-falsifier-ii-classG/` notes.

## 5. Standing gotchas (new this session, on top of runbook 41 §4's)

- Comparand FLAG-STATE pinning (rows #298/#299: a flag-mismatched comparand manufactures
  mechanistic-looking phantom deltas) · byte-id gates are SAME-MACHINE by definition (row #325
  decision; cross-machine FP 1e-16→1e-9 is not a defect, rows #318/#319) · `&` backgrounds a whole
  `&&`-chain · `source X | tail` subshell eats env vars · double-quoted ssh remote strings expand
  $vars LOCALLY (row #317) · run-dir symlinks to shared injection pools must be excluded from
  `rsync -aL` (row #311) · untracked-scp'd-then-committed sbatch files collide with cluster pulls
  (move aside, row #330) · the b8 harness must run from REPO ROOT · local OOM with 3 heavy
  processes on the 30G box (row #321) — stagger heavy local runs.
- Ledger clerk convention: one long-lived clerk agent wrote rows #291–#330 (quote-verbatim,
  chair-derived marked); re-create one in the new session.

## 6. Operating mode (unchanged, rows #290/#325)

Orchestrator = chair: delegates writes/runs (sonnet for mechanical, top-tier only for
derivation/prereg/decisive verification, ≤4 identities per batch), reviews, adjudicates, commits.
Gate-panel law: no science read without green-or-waived stamps. Every event = a ledger row.
The author's autonomy grant (row #325) stands: decide-and-flag, veto reverts.

## 7. DEVICE TRANSFER — what is `thinkpad`-only (audited 2026-09-03)

**Verdict: switching devices is possible for every open thread, but ONLY after the transfer
manifest below is executed. Nothing here is a blocker; three items are silently lossy if
skipped.** The audit was run because "continue on another device" was proposed; the numbers
are as-measured on 2026-09-03, not carried over from the 2026-09-02 storage register.

### 7.1 Already portable (in git, pushed to `origin/fix/p32d-classg-venue-repair`)

HEAD `82599c91`, **0 unpushed commits**. All code, all runbooks 1–42, `BIAS_HISTORY_LEDGER.md`
rows #1–#332, every registration/prereg/readout record, and 5 289 tracked files under
`results/`. A fresh clone + `uv sync --extra cpu --extra dev` reconstructs the whole *decision*
record. **The scientific state of record is device-independent — only data and in-flight
process state are not.**

### 7.2 Uncommitted work — STRANDED unless committed before the switch

| Path | Size | What it is |
|---|---|---|
| `DATA_INVENTORY.md` (modified) | +145 lines | The entire **Local Storage Register** — device-tag convention, the single-filesystem finding, the 2026-09-02 161 GB dedup record, the three VERIFIED-ABSENT datasets, the off-device storage assessment. This is docket item **12d (open author word)**. |
| `docs/CLAUDE_SCIENCE_BRIEF.md` | 243 lines | External-collaborator briefing (prepared 2026-08-29) |
| `docs/CLAUDE_SCIENCE_ABSTRACT.md` | 45 lines | Companion abstract |
| `results/.../ca_rhs_work/ca_rhs_{acceptance,fidelity}_output.json`, `ca_rhs_fidelity_rerun.json` | small | CA-RHS outputs (the surrounding `*_work/` dirs are 5.3 GB of scratch — do NOT commit those) |
| `selection_tables_h_0_{725,73,735}.json` (repo root) | small | stray selection tables — classify or delete |
| `scripts/bridge_closure/outputs/f4_specz_decomposition.json` | small | F4 output |

**Note the recursion:** `_run_status_S.json`'s provenance stamp records 1 114 dirty paths at the
last harness invocation. Every measurement stamped on this box carries "dirty tree" provenance
until this is cleaned up.

### 7.3 Local-only DATA required to continue (transfer or re-fetch)

| Item | Size | Second copy? | Note |
|---|---|---|---|
| `darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv` | **1.68 GB** | `bwuni` (copy of record) | **Hard requirement** for the b8 harness and every local evaluation. Gitignored. Pinned `REDUCED_CATALOGUE_MD5 = c52c13b5cab61f6b3f04bbe202550969` (`validation/correspondence_1d.py:313`) — **verify the md5 on the new device before any run** (dataset-pinning rule; a stale local catalogue already caused one silent-corruption incident). |
| **Cell-S/T resume state** — `tree2_20260830/b8_cal_harness_work_s4_postflip/universe_seed*_{S,T}.json` (92 files) + `_run_status_{S,T}.json` + `_gridsplit_check_verified.json` | **13 MB** | none | See §7.4 — the good news of this audit. |
| `realistic_20260729/seed61000/` (staged pool + raw CRB) | 57 MB | `bwuni` (expiring) | md5-manifest-verified |
| `darksiren_emri/galaxy_catalogue/GLADE+.txt` | 6.0 GB | GLADE+ upstream | only needed to *rebuild* the reduced catalogue — re-download instead of transferring |
| `results/campaign51_20260728/realistic_20260729/` gitignored blocks (`wave*/posteriors_with_bh_mass/`, retrieved run dirs, harness scratch) | ~200 GB | `bwuni`, **expires 2026-09-23, 0 extensions** | Not needed for the queued threads (S0-B reads retrieve fresh from the cluster). Needed only to re-open a *banked* posterior locally. Do not bulk-transfer. |
| `~/emri-archive/` | **159 GB, SOLE COPY** | **NONE** | Docket item **12a**. ⚠️ **Switching devices does not move this and does not reduce its risk** — it stays on `thinkpad`, still unbacked, still one NVMe failure from total loss. A device switch is not an answer to 12a. |

### 7.4 The in-flight measurement IS portable (13 MB, not 46 GB)

`m-s3` cell S: **invocation 1 has ENDED** — `_run_status_S.json` reports
`stopped_reason: wall_limited`, `n_done_this_invocation: 67`, wall 87 016 s. The `n_U_min = 60`
floor is CROSSED, so the cell is **read-valid right now**; invocation 2 (of the ≤3 allowed by the
frozen rule r-b82-s4 §3, row #301 item 2) is optional, not blocking. No local process is running
(PID 2428302 is gone) — nothing is lost by powering down `thinkpad`.

The harness resumes by skipping seeds whose checkpoint exists (`checkpoint_path()` →
`universe_seed{seed}_{cell}.json`, harness l.1127). Those 92 JSONs total **13 MB**. The 46 GB
work root is per-universe scratch (`seed901058_S/` 1.7 GB, etc.) plus regenerable
`draw_weight_cache/` (477 MB) and `precompute_cache/`. **So the resume state moves in seconds.**

✅ **RULED (author, row #333): read out cell S at n_U = 67 and never invoke it again.** The
alternative — resuming universes 68–100 on the new box — would have made the population
machine-heterogeneous (not a defect: byte-id gates are same-machine by definition, row #325;
cross-machine FP drift 1e-16→1e-9 is not a defect, rows #318/#319 — but a provenance fact
attaching to the readout). Invocations 2–3 are forfeit by ruling. **Consequence for the
transfer: the 13 MB of checkpoints still travel** (they ARE the measurement), but only as
read-input for aggregation, never as resume state.

### 7.5 Access + environment on the new device

- `~/.ssh/config` `Host bwunicluster` block (HostName `uc3.scc.kit.edu`, User `st_ac147838`,
  ControlMaster/ControlPersist 8h) must be copied. **Login needs password + OTP and is
  author-only** — no session can bootstrap cluster access unaided. The `thinkpad` ControlMaster
  socket does not travel.
- `uv sync --extra cpu --extra dev` (needs GSL; `.venv/` is not transferable).
- Add the new machine to the **Device Registry** in `DATA_INVENTORY.md` §Local Storage Register
  with its own tag — the convention is that an untagged path is a rumour. `ext-1` is still
  *(not yet acquired)*; a second laptop is a second *device* but is only *redundancy* for what is
  actually copied onto it.
- Disk on the new box: ≥2 GB for the catalogue is trivial; ≥600 GB if the campaign tree is to
  follow. `thinkpad` is at 87 % (117 GB free of 931 GB) as of 2026-09-03.
