# Runbook — next session (written 2026-08-23 ~01:45, supersedes RUNBOOK_NEXT_SESSION_29; the session context was cleared at the author's request immediately after this commit)

**Read first:** ledger rows **#166 → #173**; the [P3-IMP] chain
(`realistic_20260729/PREREGISTRATION_P3_TWIN_20260822.md` — now carrying amendments 4–21 and
three arm verdicts) with `PROPOSAL_P3_MIXTURE_WEIGHT_20260822.md` (Appendix A refuted /
**Appendix B ratified**); `PROPOSAL_SIGMA_PHI_DIVISOR_20260822.md` (**PENDING §7**);
`CLAIM_P3_RPHI_20260822.md` (ratified); **`docs/PRIMER_BIAS_CHANNELS_20260822.md`** — the
canonical vocabulary (BINDING: "contribution" for the bias split, "channel" for readouts;
define new terms on first use).

## 0. Where the campaign stands (one paragraph, typed)

The **matched-channel contribution is RESOLVED** (fused; O6/O7/O8 — both BAND-C legs closed on
the replica, CI fences carried). The **catalogue-leg twin** is the derivation-coherent
candidate (Appendix B, ratified after my Appendix A was refuted — the R-rescale was
B_scale-class): on its coherent (fused) basis it fires **TWIN-FUSED-MATERIAL:
Δ̄ = +0.029068 ± 0.005088 (12/12, 5.7σ; un-truncated +0.0634 — censoring makes the verdict
conservative)**, ~26–30% of the venue headline (FC fused-basis baseline −0.1135 floor-clamped /
−0.2136 un-truncated). ALL of this is leverage on an all-impostor venue — **correctness rests
entirely on the b0 catalogued-host identity test, which is now the sole gate between the
candidate and a production proposal.** Separately: **[P3-RPHI] ratified** (the Σ³ᴰ/Σ^φ slot is
derivation-wrong regardless of its small venue effect −0.0043) and the **Σ^φ divisor fix
proposal is PENDING** (§7; measure-first recommended). The impostor drag persists ~80% under
every measured variant; the rail is photo-z venue physics throughout.

## 1. OPEN AUTHOR DECISIONS

1. **[RULE] `PROPOSAL_SIGMA_PHI_DIVISOR_20260822.md` §7** (items 1–3; recommended:
   measure-first — production r_φ read + counterfactual flag, ≲1 CPU-h).
2. **[RULE] Ratify the TWIN-FUSED-MATERIAL verdict** as amended (row #173) — quotation rules
   binding (registered-grid vs un-truncated; never as an effect size unqualified).
3. **[DO] The b0 identity test** — design + costing, then run: three arms
   (coded / twin / R-rescaled) on the FUSED basis, on the corrected Σ^φ slot if §7 grants;
   25 banked b0 seeds exist as the coded baseline; candidate arms need fresh runs (costing
   line before launch). **This is the correctness adjudicator for the entire catalogue-leg
   question.**
4. **[RULE] Book addendum** "The Anatomy of the Bias" (row #170).
5. Carried: landscape/T1 (gated on 3); MFG-a verbatim check before paper use; comparand-CSV
   checksum pinning (recurring); workspace `emri` expires 2026-09-23.

## 2. Standing rules & ops (the session's earned state)

- **A22 as amended (row #173):** stamp git commit + dirty BEFORE the evaluate call; record the
  completion cell in every meta; no HEAD moves during registered runs (five landed mid-FC —
  non-material, verified, owned).
- **A17 additions this session:** engagement-gate denominators explicit; censored-statistic
  quotation rules; bank full reduction outputs; comparands must be banked artifacts
  (fail-closed on unexecuted gates); evidence channels verified observable.
- **A20 record:** eight clean-context reviews in two days, zero FATAL, every verdict amended —
  keep the discipline; my own instruments produced ~8 owned defects, all caught pre-banking.
- **Ops:** background commands get reaped — run measurements via detached setsid + sentinel
  file + Monitor; heredoc-stdin python cannot host forkserver pools; assert every scripted
  string-replace.

## 3. Housekeeping

- Prune candidates (~12 GB, regeneration deterministic): `p3_work/*_work` (fc/ft/phi/kflat/p),
  `o6_work/*_work`, `o7_work/*_work`, `o4_pairing_test_work/`.
- Bias-state artifact (https://claude.ai/code/artifact/a17b3f2d-9027-49ee-87be-d977221484f7)
  is current through row #165 + partial #167; refresh with rows #166–#173 next session.
- Chronicler filed at this session's close.

## 4. Resume recipe (one line)

Rule §1 items 1–2 → register + cost the b0 identity test (item 3, the centerpiece) → run it →
the production catalogue-leg proposal (or its refutation) follows from its verdict.
