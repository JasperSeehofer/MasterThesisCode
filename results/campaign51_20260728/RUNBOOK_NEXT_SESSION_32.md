# Runbook — next session (REVISED 2026-08-25 ~16:55 at the deliberate context reset, supersedes the morning revision; rows now #174 → #197; everything committed AND PUSHED, CI green)

**Read first:** ledger rows **#186 → #197** (the arc: TWIN-CALIBRATED → author ratification →
**the twin ADOPTED in production** ([PHYSICS] `bac48696`, gate-ledger rows 2026-08-25) → the
**[P3-WBHZERO] Gate-B-verified DEFECT candidate** (rows #194/#196 — the with-BH mass filter's
undocumented σ-asymmetry, 43.3% of production iiib rows, 688/688 exact attribution, no
normalization counterpart)). Session board:
https://claude.ai/code/artifact/ed640faf-33d7-42da-bcf2-4e2c09e59347 · Ch 10½ LIVE on Pages.

## 0. In-flight state at the reset

- **The [P3-2D] fleet + RHS₂ are HELD** (rows #191/#194) pending the author's [P3-WBHZERO]
  ruling — deliberately: the 2D twin should be calibrated against whichever eligibility model
  the author chooses. The pilot artifacts, the M2-LINK re-attribution note (zeros = filter
  exclusions), and the PA-2D-1/2-amended prereg are all committed.
- **The Σ̃^4D companion pass is RUNNING DETACHED** (PID on this box; started ~14:05; its slow
  scipy-quad spot-check was at ~70/100 at the reset) — it writes
  `results/campaign51_20260728/realistic_20260729/ca_rhs_work2d/p3_2d_companion.json`
  autonomously. Next session: collect + bank it (spot-check table included) — note Σ̃^4D is
  draw-law-side and likely filter-ruling-independent, but VERIFY that before freezing C₂\*.
- No cluster jobs; SSH master live (watchdog dies with this session — re-arm on demand).

## 1. OPEN AUTHOR DECISIONS

1. **[RULE — the gate for everything 2D] the [P3-WBHZERO] disposition** (row #196): ratify the
   filter asymmetry retroactively as a design choice, OR authorize the measure-first fix chain
   (counterfactual flag `mass_filter_sigma ∈ {asymmetric, symmetric}` → mirror-venue
   measurement → production counterfactual read → the 6-item package). On the ruling:
   un-HOLD [P3-2D] (A21-amend its prereg for the chosen eligibility model + the M2-LINK
   re-attribution), then run its fleet/RHS₂ per the registered costing.
2. **[DO, granted+sequenced] [HIER]** (rows #192/#193/#195/#197): the (h,θ)-grid prereg drafts
   after [P3-2D]; the [86] mapping is banked (`STAGE_L_HIER_V86_READING_20260825.md` — our
   instrument IS the un-built generalization).
3. Carried: [P3-HGRID] claim card; joint_r1 attribution (needs the cluster-side r1
   observed-catalogue artifact); MFG-a verbatim check before paper quotation; F0-SEL cheap
   follow-up; the AMEND-2 stale log-substring gates (fix on next instrument use);
   bias-state artifact refresh (rows #166–#197); workspace `emri` expires 2026-09-23.

## 2. Standing rules & session-earned ops (delta over runbook 31)

- **Row #185 [STANDING, author]:** cluster-first ≥2 CPU-h; queue-waits banked; chronic-blockade
  reversion test. Row #187/#188 author rulings verbatim in the ledger.
- **PA-2D-2 lesson (reusable):** GH quadrature borrowed from a narrow-σ regime silently biases
  wide-σ marginals over piecewise-linear grids — the exact per-cell erf-moment rule is the fix;
  every S_4D consumer should state its σ regime.
- **Agent-prompt hard opener (10+ parking incidents):** the no-parking rule + no-resubmit rule
  + __main__-guard/CWD rule go verbatim at the TOP of every spawned agent prompt (this worked;
  omission did not).
- CI is GREEN since `26795160` (machine-of-record artifacts get skipif guards; exact-pin tests
  carry rel_tol 1e-12 for cross-arch portability).

## 3. Housekeeping

- Prune candidates: runbook 30 §3 set + `ca_rhs_work/score_chunk*_work` contaminated dirs
  (PA-CA-11 evidence — keep chunks 0–4 quarantined copies until the author has read row #186).
- The bias-state artifact refresh (rows #166–#190) still pending; the session board covers the
  narrative meanwhile.
- Chronicler: file at close (rich: three contaminated-number catches, the erf-rule lesson,
  parking-rule efficacy, the CI-debt lesson).

## 4. Resume recipe (one line)

§0's in-flight arc to its verdict → author rules items 1–2 → the production adoption commit
([PHYSICS], the row-#178 pattern) → then either the hierarchical thread (item 3, if granted)
or the paper-facing consolidation.
