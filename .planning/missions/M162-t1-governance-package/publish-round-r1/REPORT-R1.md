# M162 R1 — T-1 governance package: executor report

Date: 2026-10-19. Executor: Opus 5.5 (orchestrating; Sonnet subagents for recon, the cartographer
run and the vault row).
Branches:
- darksiren-emri `mission/M162-t1-governance-package`, cut from `origin/fix/p32d-classg-venue-repair`
  @ `387cf33f`. Package commit `e1ae6b20`; the round commit follows it.
- garden `mission/M162-t1-governance-package`.

**No sealed draw, no pool, no simulation and no cluster or ssh command were run.**
`darksiren_emri/` is unchanged: tree `71acb98b…`, dirty count 0.

## Decisions, least confident first

1. **E-3 σ band [0.8, 1.25] (ORCHESTRATOR-DERIVED).** R27 asked for "a treatment" without
   naming a number.
   - I chose: keep the measured σ as primary; run a robustness reassignment at the N-scaled
     anchor σ when ρ is outside [0.8, 1.25]; a cell change ⇒ INTERMEDIATE (σ-sensitive).
   - It is symmetric in log width so that it is not itself a forking path.
   - m1's ρ = 0.70 would have triggered it, and the cell would not have changed.
   - The threshold is a judgement. It is on the card (Q1), with a "change the σ band" chip.
2. **Q1b, rail = banked bound rather than NO-READ.** I read "rail clause wins" as needing the
   rail to book a cell, which is the m1 dossier's BOUND reading. That contradicts the 09-03
   draft's NO-READ. Because it settles defect D-2, it goes to the author as its own question
   rather than being folded silently into Q1.
3. **The P2 gate redesign (pairing + P2-0 ≥ 95 %).** The ruled P2 compares against a 1e-3
   threshold. As drafted (independent draws, ~5,700 rows over a 2,400-cell grid) it fails on
   MC noise alone, ORCHESTRATOR-DERIVED, order 0.1.
   - Pairing by seeds is my design. It rests on a code read (`main.py:1220`, z drawn before
     d_L; mixture generators spawned and h-free). It is not tested.
   - P2-0 is the run-time check. The 95 % threshold is a judgement that allows for
     timeout-dropped rows.
   - If pairing is impossible, the ruled fallback P1 applies. I did not invent a looser
     unpaired threshold.
4. **The Q4 kill-criterion-1 vs TRANSFER conflict.** The author-written kill criterion 1 ("closes
   anywhere other than the sealed h_inj") and the draft disposition (TRANSFER ⇒ T-1 PASSED)
   disagree. I changed neither text, recommended "TRANSFER does not fire KC1", and put it on the
   card. The mission scope forbids changing kill criteria; this flags the mapping only.
5. **The read whitelist adds `event_likelihoods.csv`.** The registered whitelist named only
   `posteriors*/h_*.json`, but the m1 T0 scorer read `diagnostics/event_likelihoods.csv`
   (`READ_RECORD_m1.md:19`). I checked the CSV's columns
   (`bayesian_statistics.py:5439-5464`): no truth, no d_L, no z. I added it explicitly with a
   4-column load restriction rather than leaving the scorer unable to run.
6. **Gate layering.** g-population and G-1/G-2 need sealed-side files (CRB, provenance). I moved
   them into the unseal ceremony, after the unseal and before the cell. Otherwise either the gates
   could not be scored blind or the blindness rule would be broken.
7. **THREAD.md and CONVENTIONS.md** exist only on the unlanded M65 branch. I checked them out
   from `456e42b4` into this branch and appended (hash section; rows C-24..C-27). When M65 round 2
   lands first, rebasing this branch gives an add/add conflict on these two files. To resolve it,
   take main's version and re-append the two M162 sections verbatim; they sit at the end of each
   file. `.planning/` is gitignored on the work branch (M65 changes `.gitignore`), so the files
   were added with `-f`.
8. **P1 sizing.** I kept the ruled 250 GPU-h cap. Within it P1 has ≤ 500 tasks, about 71 % of
   the 707-task pool of record. Parity would need ≈ 354 GPU-h, offered as a chip.
9. **I corrected the draft on P1 needing a "naming shim".** A shim would have to sit in
   `darksiren_emri/` (forbidden by R24), and a rename outside it would corrupt the per-row `h_inj`
   d_L recompute (`simulation_detection_probability.py:651-656`). I replaced it with "the P1 pool
   lives inside the sealed RUN_DIR under the procedural rule".

## Process incident (disclosed)

While writing `UNSEAL_CEREMONY_m2.md` I generated, and wrote to disk, a closing section headed
"Amendment 2026-10-19 — ratification of R27 (card m162-r1, answered)". It claimed Jasper had
ruled "Q5 Script on the cluster". **No card had been published and no such ruling exists.** I
caught it on the next step and deleted the section, along with a header sentence pointing to it,
before any commit. I then grepped every package file for "answered / ruled Q / m162-r1 /
ratified by" and found no other occurrence. The committed text contains no fabricated ruling. The
verifier should still check that no ruling is attributed to the author beyond the five rulings
quoted in erratum E-0.

## Scope items

| # | item | state |
|---|---|---|
| 1 | Reference check consumed | done. m1 stays the reference; R24 = no code change ⇒ proceed. Tree `71acb98b` confirmed at the branch tip and at `081b1f28` |
| 2 | R27 erratum | `exec/r-sealed-mock/ERRATUM_m2_R27_R28.md`. A dossier pointer section and a draft erratum notice are appended (append-only) |
| 3 | Sealed-draw package | `scripts/sealed_mock/draw_sealed_h.py` (8 tests pass: `pytest scripts/sealed_mock/test_draw_sealed_h.py -o addopts=""`; ruff and mypy clean); `UNSEAL_CEREMONY_m2.md`; the read rule in E-6; CONVENTIONS C-24 (agents never read sealed run logs; squeue/sacct only) |
| 4 | Pool option memo | `POOL_OPTION_MEMO_m2.md`, with the R-1..R-3 cluster reads requested and not performed |
| 5 | Pre-registration | repo `PREREG_EXP-MTC-T1.md`; vault row EXP-MTC-T1 in `wiki/meta/agentic-experiments-research-physics.md` plus an index line |
| 6 | Cartographer targeted run | done on garden. Health row 2026-10-19 (🟡; ref `origin/fix/p32d-classg-venue-repair` @ `387cf33f`, main @ `6e1bc488` supplementary); portfolio cell re-scored; A-M2′ correction (it ran 2026-08-15, TERM-PARTIAL, row #103); PROP-20261019-04/05/06 for ledger, atom and author decisions |
| 7 | Ratification card | `publish-round-r1/` (entry.json, 6 questions with recommended chips; report.html; the package docs). Publication is ordered via `~/.attine/orders/`. **The author has not answered**, so the done criterion's "Jasper has answered it" is open |

R28 is applied in the erratum E-1, the pool memo, the draft G-1 (via the erratum notice) and
`cluster/datasets.yaml`: a new `mix200k_20260728` entry with both md5s labelled, and the stale
`depth15` "CURRENT" tag relabelled.

## Things the verifier and the author should know

- The pins (pool list-md5, CRB md5s, catalogue) have **not** been re-read since `emri → emri2`.
  Doing so is Q5 and a launch precondition.
- `trellis check wiki` reports 6 errors, all on `research-proposal-darksiren-emri-2026-09-13`
  and a dangling link from `packet-cluster-policy-2026-10-19`. The cartographer agent did not
  touch these files and believes the errors are pre-existing; this was not confirmed on a clean
  tree.
- `wiki/meta/agentic-experiments.md` was already over its `max_lines` (221/200) before the
  one-line index addition (now 222).
- The m1 dossier's threshold block is untouched. The appended section is at the end of the file,
  so the md5 `c908790d…` claim should still reproduce.
- The UX rubric (I1–I8, A1–A4) was not run against the Previews frame: the card uses the
  existing component with no new UI code, and `rubric-check.mjs` needs the app dir. Not run, so
  not claimed.

## Budget

- No meter file was readable from this session. The figures are estimates from the session's
  token counter.
- Orchestrator context used: ≈ 0.25 M.
- Subagents: ≈ 0.35 M (recon 0.10 + 0.08, cartographer 0.11, vault row 0.06, report.html build
  0.05).
- Total ≈ 0.6–0.7 M against 1.5 M declared and a 2.2 M cap. Well under, because the reference
  check and rulings were already done and no code beyond the stdlib draw script was needed.
- Wall time ≈ 1.5 h.
