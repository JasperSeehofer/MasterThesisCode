# Runbook — next session (written 2026-08-21 ~22:15, supersedes RUNBOOK_NEXT_SESSION_26)

**Read first:** ledger rows **#152 → #156**, then the prereg's last three appended blocks
(`PREREGISTRATION_SELFGEN_CONTROL.md`: CORRECTION & REVIEW ADDENDUM → O4 REGISTRATION → O4
VERDICT) and the two banked reviews (`ADVERSARIAL_REVIEW_CSG_20260821.md`,
`A20_REVIEW_O4_20260821.md`). Runbook 26 covers the overnight campaign and is background —
**its §1 decision queue is superseded** by what follows. Do not redo anything.

## 0. Where the campaign stands — the answer, in one paragraph (typed)

B-SEL's −0.1083 is fully decomposed with mechanisms [MEASURED]: **impostor catalogue leg −0.079**
(shallow-catalogue chance alignments; O2, row #149) ⊕ **dark-fraction tilt +0.055** (O3, row #150)
⊕ **matched-channel mismatch −0.085**, whose mechanism is now identified [MEASURED, row #155]:
**the off-cell completion numerator omits the S̄_φ survival factor its own normalizer carries**
(`PRODUCTION_FLAGS: selection_in_completion_numerator="off"` — the legacy pre-#118 cell, labelled
"not a production posterior" by the estimator's own log; `fused` is the in-tree fix). Restoring
the registered arm nulls the C-SG score (+0.0076 ± 0.0184, 0.41σ); O5 (banked B-SELF) corroborates
(fused-cell mirror arm: −0.064 ± 0.019, residual provisionally generator-caveat-side). The 6.05σ
non-zero score stands as a real numerator/normalizer mismatch OF THE OFF CELL — "deeper estimator
math" is withdrawn. Caveat [BANKED]: fixing the off cell does NOT cure the H₀ rail — the full
posterior is impostor-leg and rail dominated (row #119's "not material" is consistent with this).

## 1. OPEN AUTHOR DECISIONS

**Scope note (row #156):** the author's "all ratified" (2026-08-21 evening) covered retrospective
entry 4's AMENDMENTS (A21 + three fold-ins — adopted). Row #155's campaign decisions were a
separate earlier list and are **OPEN — confirm or rule first thing**:

1. **[RULE] Label disposition:** rows #140/#151 → "IMPLEMENTATION-CONVENTION DEFECT (off-cell
   S̄_φ omission), mechanism identified and quantified".
2. **[DO] One C-SG-F seed end-to-end under `fused`** (both legs, not a numerator patch; expect
   matched score ≈ 0; ~45 min cluster; A21 applies — register the arm text exactly).
3. **[RULE] Production-basis fork:** all runs-of-record stand on the off cell; off→fused is
   physics-change-gated (`bayesian_statistics.py` trigger file, full 6-item gate package), and
   should be decided jointly with the impostor-leg question (the dominant production channel).
4. Carried: landscape/T1 un-gate (chain now at its third link: mechanism → fused confirmation →
   fix fork → landscape); systematics row 16 re-grade; workspace `emri` expires **2026-09-23**.

## 2. Standing rules now in force (all author-ratified TODAY — apply from the first action)

- **A17** gate/band portability + realized-scatter re-check + **axis-leverage statement** (a band
  unreachable by the registered axis is void on its face).
- **A18** every scorer prints, per arm, the reference each bias statistic subtracts.
- **A19** every fireable branch carries a registered falsifier.
- **A20** clean-context adversarial verification before any BANK/PROMOTE/WITHDRAW ruling;
  verifiers get artifacts + the REGISTRATION TEXT (never the implementer's summary); report
  sentences typed MEASURED/BANKED/PROVISIONAL/NARRATIVE-HYPOTHESIS. Model choice unregulated;
  default pairing Fable-orchestrates/Opus-critical-thinker; on Fable rate-limit, Opus orchestrates
  with mandatory Fable revisit.
- **A21** registration–execution identity: a "corrected premise" mid-implementation STOPS the run
  → registration amendment + re-derived bands BEFORE execution.
- Instrument registrations carry a **costing line** (wall/seed, peak RSS/seed — evaluate() ≈ 9 GB —
  venue decision). Pre-checks that recompute production quantities are instruments.
- The model-tendency question is settled for now (row #154, probe: role+context dominates; n=1
  caveat, Opus-as-builder direction untested — a P5 instrument).

## 3. Housekeeping owed

- **Overview artifact** (claude.ai/code/artifact/134076ad-…) is two cycles stale (pre-O4) —
  refresh after the §1 rulings so it carries the final story.
- **Campaign readout report** similarly ends at row #151 — either append an O4/row-#155 section
  or supersede it in the next report.
- `/chronicler` for the afternoon/evening session (the morning session was filed; O4's lessons —
  A21's birth, the OOM costing gotcha, verifier-anchoring — are unfiled).
- SSH keepalive loop may still be running from 2026-08-21 (~10-min pings) — kill or keep as
  preferred: `pkill -f "ssh_keepalive"` pattern in scratchpad.
- 130 O4 shard JSONs + work dirs: `o4_shards/` is committed; `o4_pairing_test_work/` is large and
  UNCOMMITTED — decide keep/prune (regeneration is deterministic).

## 4. Resume recipe (one line)

Confirm/rule §1 items 1–3 → register the fused end-to-end seed per A21 → run it (cluster,
costing line first) → A20 review → then the production fork + landscape chain.
