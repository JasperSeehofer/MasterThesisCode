# Handoff — Paper A peer review restart after GPD update (2026-07-02 evening)

## Why this handoff
The `/gpd:peer-review` run was started on an **outdated GPD install**; the user updated GPD
mid-session and chose to restart the review from scratch in a fresh session. The partial
review-prep artifacts were reverted (see §3). Everything else from today is committed/pushed.

## 1. Do next (fresh session)
1. **Restart the cluster monitor first** — the overnight background poll died with the old
   session. Watch combine job `5698618` (`ssh bwunicluster 'squeue -j 5698617,5698618 ...'`);
   on COMPLETED, retrieve + verify per
   `.planning/HANDOFF-DERAIL-CLUSTER-CONFIRM-20260702.md` §"When the combine COMPLETES",
   then fill Paper A's single `[PENDING]` slot (realdata section) and regenerate the PDF.
2. **Re-run `/gpd:peer-review` with the UPDATED GPD** on the Paper A manuscript. The updated
   command supports "an explicit external artifact" target — point it at `paper_a/`
   (NOT `paper/`, which is Paper B's base). 21-pp compiled PDF: `paper_a/main.pdf`
   (rebuild recipe in `paper_a/Makefile` header — TeX only on the cluster login node;
   scratch build dir `bwunicluster:paper_a_build/` exists and works).
3. After review: `/gpd:respond-to-referees` / revision loop as routed.

## 2. Paper A state (branch `paper/paper-a-draft`, all pushed, HEAD `7914c87`)
- `efe4b54` full first draft (10 sections + 5 appendices + abstract + 4 figures with
  standalone scripts + data artifact). One intentional `% [RESULT PENDING: cluster jobs
  5698617/5698618 ...]` in `sections/realdata.tex` (user-accepted).
- `bcb1cb5` bibliographer (all 14 MISSING keys resolved+verified INSPIRE/ADS/DOI, zero
  hallucinated → `CITATION-AUDIT.md`) + notation audit (108 mechanical fixes,
  `NOTATION-AUDIT.md`) + consistency fixes (unified cite keys 22→14, budget-table
  Eddington-in-z row −0.024 absolute not −2.4%, fig:pp caption from verified data).
- `ddeb58d` notation harmonization: **C1 resolved** — full-sky `w_pop=(dV_c/dz)/(1+z)`
  paper-wide, per-steradian written explicitly as `w_pop/(4π)`; verified equation-identical
  vs G2a, **no hidden prefactor errors**; C2 `C→K`, C3 `s(z)→q(z)`, C4 `g(z)→\tilde w(z)`;
  M3–M7 done. 300 refs / 88 labels resolve. NOTATION-AUDIT.md §5 has the change log.
- `7914c87` vendored `mnras.cls`/`mnras.bst` (CTAN) + `Makefile` + `.gitignore`.
  First compile: **0 errors, 0 undefined refs**, 21 pp; bibtex warnings only
  (empty journal on `Babak:2023lro`, `Hogg:1999ad` — arXiv-only entries).
- Known polish items: abstract 258 words (MNRAS cap 250), author 2 + acknowledgments TBD,
  9 orphan bib entries seeded from Paper B (prune before submission).

## 3. Reverted peer-review prep (regenerate under the NEW gpd schemas)
Deleted (were uncommitted): `paper_a/BIBLIOGRAPHY-AUDIT.json`, `paper_a/reproducibility-manifest.json`.
Regeneration notes (10 min of work, all inputs still present):
- **BIBLIOGRAPHY-AUDIT.json**: was generated schema-valid via
  `gpd.mcp.paper.bibliography.BibliographyAudit/CitationAuditRecord` pydantic models from
  `references.bib` + cited-key scan. Facts: 39 entries, 30 cited (all resolve, none missing),
  14 session-verified (INSPIRE/ADS/DOI), 16 seeded-and-cited (verified in Paper-B audit),
  9 uncited orphans (flag: prune). Check whether the NEW gpd has a CLI/agent that emits this
  natively before hand-rolling.
- **reproducibility-manifest.json**: template `templates/paper/reproducibility-manifest.md`;
  validator `gpd validate reproducibility-manifest <path> --strict`. Last run scored
  checksum 81.8%, seeds 100%, `ready_for_review: false` with 6 issues — fixes known:
  exact versions numpy 2.4.3 / scipy 1.17.1 / matplotlib 3.10.8 / pandas 3.0.1;
  sha256 for `master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv.zhelio_20260702`
  (1.7 GB, the variant the committed baselines used) and for the seed600 prepared CRB csv
  (in `~/data-backups/seed600_local_derail_20260702/crux_ws/` or on the cluster RUN_DIR).
- **Old-gpd strict preflight blockers to expect** (may differ in the new version):
  (a) it resolved `paper/main.tex` (Paper B!) as "the" manuscript — must target paper_a
  explicitly; (b) legacy `.gpd/` state debt from the April v2.2 era (state.json/STATE.md
  drift, missing CONVENTIONS.md, old-schema phase frontmatter in phases 14–18) — this is a
  DIFFERENT milestone's bookkeeping; today's gate evidence lives in `.planning/gate/` +
  `docs/derivations/` + `results/commission_20260701/redteam/`, not in `.gpd/phases/`.

## 4. Rest of today's state (unrelated to the review, all done)
- **PR #18 MERGED** to main `8737acc` (docs-CI docstring fix `e283354` + G4b CHANGELOG).
- **PR #21 OPEN** (`campaign/phase2-prep`): campaign runbook `180c7c3` +
  `[PHYSICS]` d_L pre-screen fix `b7e3edd` (fixes #19; population-derived bound
  `1.05 × d_L(z_max; h)`, 5 regression tests, 723 passed). Merge after CI.
- **Issue #19** open only for `PRESCREEN_DL_MARGIN` re-measurement on post-dt² injections.
- **Issue #20** open — `HOST_DRAW_Z_MAX=0.5` population-depth DECISION (campaign-submit
  blocker; needs user + /physics-change).
- **Cluster jobs 5698617/5698618** still PENDING (cpu,cpu_il, in-place widen, no cancel).
  Root cause: cluster at full capacity (MOTD) + PrivateData hides other users' jobs — all
  "empty queue" readings were illusory. Correct posture: wait; short jobs backfill.
- Runbook: `.planning/CAMPAIGN-PREP-PHASE2.md`. Do NOT pull main / stage z_cmb catalogue on
  the cluster until the seed600 jobs finish (baseline consistency).
- Pending user offer: `/wiki-debrief` (4 lessons: stale-calibration constants class,
  in-place partition widening, pipe-masked pre-commit failure, PrivateData queue illusion).
