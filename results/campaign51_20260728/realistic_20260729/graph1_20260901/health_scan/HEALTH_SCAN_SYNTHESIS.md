# Overnight Project Health Scan — Chair Synthesis (2026-09-03)

Commissioned under the author's 2026-09-03 directive (row #327); four sonnet lenses
(LENS1_CODE_HEALTH / LENS2_REPO_DOCS / LENS3_INFRA_DATA / LENS4_PERF_FEATURES, this directory),
chair synthesis. **PROPOSALS ONLY** except the one flagged safe-hygiene action noted in §2.
Every ask is tagged per the approval-scope convention.

## 1. The picture in one paragraph

The science machinery is in excellent shape — gate discipline, provenance stamps and dataset pins
are consistently applied, dead code is nearly absent, and the reputed performance bottleneck
(`scalar_product_of_functions`) turns out to be already optimized. The risks are all *around* the
science: a 159 GB sole-copy dataset with zero backup, a cluster workspace expiring in 20 days with
~250 GB unarchived and no destination chosen, a local disk at 85%, a 112-commit branch that has
outgrown `main`, and a 9,332-line god-module whose 74-flag `evaluate()` taxes every new branch of
work. None of it is broken today; several of it becomes unrecoverable if left until October.

## 2. Action taken under the row #325 grant (flagged)

- **`archive_run_wave2.sh` three-valued-existence fix** — owed since row #288, confirmed still
  unlanded by lens 3 (a textbook false-negative on the 2026-09-01 log). Builder dispatched:
  PRESENT/ABSENT/UNREACHABLE, loud abort on UNREACHABLE, session-start reachability probe,
  extracted as a reusable helper, self-testable without the cluster. Ops-script only, no physics.

## 3. Decision docket — item 12 (sub-items each take one word)

| # | ask | tag | chair recommendation |
|---|---|---|---|
| 12a | **Backup the 159 GB `~/emri-archive/` evidence locker** — the sole copy of the seed600 dataset; needs a destination (external drive / institutional storage / bwDataArchive). The *decision* is the destination; the copy itself is mechanical once named. | [RULE→DO] | Decide the destination TODAY; this outranks everything else in this docket. |
| 12b | **Cluster evacuation before 2026-09-23** — ~250 GB post-campaign51 fleet data UNKNOWN/unarchived, 0 workspace extensions left (verified by a failed `ws_extend` probe). Grant a triage-and-archive campaign (agent-run, using the fixed archive tooling, against a named destination from 12a). | [DO] | Approve; run it this week, not in week 3. |
| 12c | **Local-disk cull** — 85% used, 136 GB free; two ~15 GB cull candidates already named in the uncommitted DATA_INVENTORY.md diff, plus the graph1 `retrieved/` 38 GB mirror once cluster archiving is confirmed. results/ is gitignored → deletions are unrecoverable → each cull needs your word. | [DO, itemized] | Approve the two named candidates now; the 38 GB mirror only after 12b completes. |
| 12d | **Commit the DATA_INVENTORY.md diff** — lens 2 read it as complete and well-formed (it documents the storage situation). | [DO] | Approve. |
| 12e | **Merge `fix/p32d-classg-venue-repair` → `main`** — 112 commits ahead, 0 behind, six production [PHYSICS] flips; `main` no longer describes production. | [RULE] | Merge after S0-B lands and the wave closes, as a reviewed PR so CI runs once over the whole delta. |
| 12f | **Safe-build grants** (no physics gate; each is S/M effort): (i) campaign-driver/sbatch renderer replacing 16+ hand-copied scripts; (ii) auto-filled campaign readout report from the existing template (mechanical rendering only, per the template's own rule 2); (iii) `EvaluationConfig` dataclass refactor of the 74-flag `evaluate()` signature — mechanical, byte-identity-gated; (iv) the remote-existence + same-machine byte-id gate utilities promoted into `cluster/`. | [DO, itemized] | Approve (i), (ii), (iv) for this batch's idle time; schedule (iii) as its own reviewed change AFTER the merge (12e), since it touches a physics-trigger file's structure (behavior-preserving, byte-id-gated, but big). |
| 12g | **Docs sync** — CLAUDE.md stale entries (bug #6 guard already landed; the removed re-export note; the `scalar_product` "bottleneck" framing now wrong; Known-Bugs-vs-GitHub split made explicit), TODO/README/CHANGELOG 3 weeks behind, the two likely-duplicate issues (#25/#57). | [DO] | Approve; safe-hygiene, one agent pass + chair review. |
| 12h | **NOT proposed**: splitting `bayesian_statistics.py` (9,332 lines, 1,621-line `evaluate()`). The seams are real (mass-aware leg / mixture assembly / workers) but mid-campaign surgery on the production likelihood is exactly the wrong moment. Defer to the post-paper refactor window; 12f(iii) buys most of the ergonomics at a fraction of the risk. | [INFO] | — |

## 4. Corrections to the record (safe, folded into 12g)

- CLAUDE.md bug #6: the wCDM `NotImplementedError` guard HAS landed — entry should be struck through.
- CLAUDE.md architecture note on `cosmological_model.py`'s backward-compat re-export: code already removed.
- CLAUDE.md's `scalar_product_of_functions` "computational bottleneck" note: the function already
  does per-length PSD caching + batched multi-channel FFT; the framing misleads future optimizers.
- The PHYSICS-GATE-LEDGER `pre-commit` placeholder convention defeats naive hash-audits (lens 2
  finding 3): today's rows are fine, but the ledger header should document that audits must match
  on the *row content*, not the hash column alone — or rows should always be SHA-updated (the
  convention this session already followed).

## 5. What was checked and found healthy

Provenance/pin discipline (consistent across all current runs) · runbook cadence · dead code
(≈none) · unused imports (0) · Pipeline-A remnants (0) · gate-ledger coverage of all six
[PHYSICS] commits (complete, once the placeholder convention is understood) · `scalar_product`
performance · memory-dir hygiene.
