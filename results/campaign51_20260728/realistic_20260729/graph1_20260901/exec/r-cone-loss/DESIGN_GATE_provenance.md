# r-cone-loss — DESIGN-GATE PROVENANCE CHECK (lens: PROVENANCE)

Node: `r-cone-loss` (Research Graph 1, Branch H, wave 3). This is a **provenance-only** lens
on `REGISTRATION_DRAFT.md` — every quoted number traced to the record/file/CSV it cites, opened
directly, byte- or value-compared. No pipeline run, no source-code execution of
`darksiren_emri/`, no cluster access. Two lightweight local re-derivations were done (host-index
counting on a CSV already on disk; a mean/SEM recombination over 67 already-banked JSON
checkpoints) — arithmetic on retrieved data, not a simulation or estimator run.

Companion inputs read: `../r-completion-residual/INFORMATION_FORECAST.md` (forecast, cited by
the task as style precedent for numbers appearing on "the board" around this draft), and
`../r-b82-s4/DESIGN_GATE_RECORD.md` (design-gate style precedent, 6-check format followed here).

## Verdict up front

**RED.** One decisive number in the draft's own §0 table is cited to a ledger row that does not
contain it (the actual source is a different row, one day later). A second decisive number is
cited to a node whose own output artifacts do not contain it (the number is real and I
independently reproduced it from raw data, but not from the record the draft names). Both are
must-fix before this draft can be called SOUND and launched.

## Check 1 — §0 provenance table, row-by-row

| draft claim | cited source | opened? | verbatim match? |
|---|---|---|---|
| "2261 events (bc arm) · 380 outside · fraction 0.1681"; anchor chord 1.6746585172e-03 / radius 1.4956979546e-03 | `CLAIM_P3_MKER_20260826.md:928-947` | yes | **exact**, including the anchor floats, the "16.8%" headline, and the 13.4%/32.5% chair envelope |
| "380/2336 (0.1627)" | `fanout1_20260829/B2_1_CMEM_A1_RECORD.md` | yes | **exact** — file states `combined: 380/2336 (0.16267)` line 29, and explicitly flags the 380-coincidence itself (line 31-32), which the draft also flags |
| "[CMEM] verdict of record (row #220): C-STRUCTURAL-ONLY; the outside-cone truth-likelihood deficit R2c NOT-DISTINGUISHED (p = 0.0358 vs α = 0.01; power ≈ 68%)" | `BIAS_HISTORY_LEDGER.md` **row #220** | yes, full row opened | **FAILS.** Row #220 (2026-08-28, the author's "ratified" ruling) contains ONLY "(2) [CMEM] C-STRUCTURAL-ONLY is RATIFIED as the verdict of record (row #219 binds...)". It explicitly lists the R2c question as **NOT yet answered**: "NOT covered ...: the higher-power R2c follow-up vs bank-and-park (**one word still required**)." Neither `0.0358` nor `68` nor `R2c` nor `NOT-DISTINGUISHED` appears anywhere in row #220's text (grepped the full row body). The p=0.0358/power≈68% result is stated, verbatim, one day later in **row #226** (2026-08-29): "primary equal-weight p = 0.0358 ≥ α = 0.01 ... pre-registered power at the original −16% effect ≈ 68%." **The draft cites the wrong row for this half of the sentence.** |
| production pool: `seed61000/prepared_cramer_rao_bounds.csv`, 76 in-catalogue events, 10/76=13.2% outside, chord median 1.24e-3/max 5.11e-2, cone radius median 2.70e-3; P6 log "1D 66/76 hosts recovered/in-cat events seen (86.84%)" at `.../darksiren_emri_20260902_000633_h_0_73.log:8622` | CSV + log, both on disk | yes | **CSV**: independently recounted `host_galaxy_index >= 0` → exactly **76** rows (of 1590 total). **Log line 8622**: read directly, verbatim match — `P6 host-recovery (h=0.7300): 1D 66/76 hosts recovered/in-cat events seen (86.84211%), 2D 66/76 ...`. 76−66=10 agrees with the claimed 10/76=13.2% by construction. The chord/radius distribution figures (median 1.24e-3, max 5.11e-2, cone-radius median 2.70e-3) are **not present in any flat CSV** — `event_likelihoods.csv`'s 19 columns carry no chord/radius/cone field, so this sub-claim is not independently checkable without running the sky-cone geometry code (out of scope per the task's "never touch darksiren_emri/ source" instruction). Not contradicted by anything found; flagged as an accepted gap, not a defect — consistent with the draft's own §4 blindness disclosure that this fraction was author-computed as a stage-0 fact before the draft was written. |

## Check 2 — the estimator block (§2), row #302

| draft claim | cited source | verbatim? |
|---|---|---|
| `I_1D = 1/0.017526² = 3256`; `I_2D = 1/0.018475² = 2930` | row #302 | Row #302 states, verbatim: "iiib 1D (`combined_no_bh`) **0.665 / 0.666987 / 0.017526**" (map/mean/σ) and "iiib 2D (`combined_with_bh`) map_h **0.665**, mean_h **0.665854**, σ_h **0.018475**". Independently recomputed both: `1/0.017526² = 3255.6 → 3256` ✓; `1/0.018475² = 2929.75 → 2930` ✓. |
| `mean_h − 0.73 = −0.0630 (1D) / −0.0641 (2D)` | row #302 | `0.666987 − 0.73 = −0.063013 → −0.0630` ✓; `0.665854 − 0.73 = −0.064146 → −0.0641` ✓. **Note:** the draft never quotes `0.666987`/`0.665854` literally — only the derived offset — but the raw values ARE in row #302 verbatim, so the derivation is fully traceable, just one arithmetic step removed from the printed row. Not a defect. |
| CRB md5 `9a1f2a14384a9281c97ca3be312ddaab` | pinned in §2/G-1 | `md5sum seed61000/prepared_cramer_rao_bounds.csv` → **exact match**. |
| catalogue md5 `c52c13b5cab61f6b3f04bbe202550969` | pinned in §2/G-1 | `md5sum darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv` and the retrieved run's `cwd/.../reduced_galaxy_catalogue.csv` → **both exact match**. |
| `GIT_COMMIT_AT_RUN.txt = 1ec9514d` | G-1 | file reads `1ec9514dd1808c48b18c0792dce558e5bba0f116` → prefix **matches**. |
| "1590 rows, 1588 scored, event_idx gaps {1203, 1356}" | §2 | Independently recomputed from the retrieved run's own `simulations/diagnostics/event_likelihoods.csv` (65,109 lines incl. header = 65,108 data rows, matching the §6 cost estimate's "65,108 rows" too): 1588 distinct `event_idx` values present; missing set from {0..1589} = **exactly {1203, 1356}**. `prepared_cramer_rao_bounds.csv` independently confirmed at **1590** data rows. **Exact match on every count.** |
| `combined_no_bh` / `combined_with_bh` columns | §2, the score definition | header of `retrieved/.../event_likelihoods.csv` and `headreadout_20260827/iiib/event_likelihoods.csv` both list `combined_no_bh` (col 15), `combined_with_bh` (col 16) — **present, exact names**. |

## Check 3 — the harness replicate (§2, last paragraph)

| draft claim | cited source | verbatim / reproduced? |
|---|---|---|
| "67 post-flip S3 cell-S universes" | `b8_cal_harness_work_s4_postflip/seed9010NN_S/...` | `ls -d seed*_S` → **exactly 67** directories (901000–901066); `seed*_T` → exactly 25, matching the companion cell-T population. |
| "843 catalogue-hosted events" | same | summed `universe.n_catalogue_hosted` across all 67 checkpoint JSONs → **exactly 843**. |
| "+0.587 ± 0.064 per event, 67 universes ... rd-s3-readout" | cited as **rd-s3-readout** | **The number is real and independently reproducible — the citation is wrong.** I re-derived it directly from the 67 raw `universe_seed9010NN_S.json` checkpoints (`score_at_truth.no_bh.catalogue_hosted`, per-universe `mean`), taking the across-universe mean and its SEM (n=67): **mean = 0.58749, SEM = 0.06380** — matches "+0.587 ± 0.064" to the stated precision. But I grepped `rd-s3-readout`'s own three output artifacts — `exec/m-s3-postflip-coverage/READOUT_RECORD.md`, `.../CHAIR_REDERIVATION_20260903.md`, and ledger **row #335** (the "rd-s3-readout DONE" row) — for `0.587`: **zero hits in all three.** What those documents DO contain, and what the draft's own line correctly cites separately, is `Z 9.76` (score-zero Z, catalogue_hosted class, cell S no_bh) — that figure IS verbatim in all three (`READOUT_RECORD.md:73/181`, `CHAIR_REDERIVATION_20260903.md:14`, row #335). The `+0.587 ± 0.064` magnitude instead traces to the **companion forecast document**, `../r-completion-residual/INFORMATION_FORECAST.md:19` ("harness catalogue-hosted full score | +0.587 ± 0.064 (843 events) | ... the S3 DEFECT-SIGNATURE locus"), which itself cites the raw checkpoints, not an rd-s3-readout deliverable. **Must-fix:** re-cite this figure to the checkpoints directly (or to `INFORMATION_FORECAST.md`), not to "rd-s3-readout" — rd-s3-readout never computed or published it. |

## Check 4 — numbers on the assigned check-list that do NOT appear in this draft at all

The task's check-list included `F 11.44` and `σ floor 0.00518915`. Neither string, nor a value
rounding to either, appears anywhere in `REGISTRATION_DRAFT.md` (grepped the full file). Tracing
them anyway: **`F 11.44`** is real — it is cell T with-BH's `F = SD/floor` in
`exec/m-s3-postflip-coverage/READOUT_RECORD.md:83` ("F = SD/floor (harness-quoted) | 11.33 |
**11.44**"), a Branch-A rd-s3-readout number, and it also appears in the companion
`INFORMATION_FORECAST.md:22` ("F band in h (context) ... F 11.44 (DEFECT-context)"). **`σ floor
0.00518915`** likewise is real, from `../r-b82-s4/DESIGN_GATE_RECORD.md:40` (a *different node's*
design-gate record, the style-precedent file, not a citation the r-cone-loss draft makes).
**Neither is a defect in the r-cone-loss draft** — the draft simply never asserts these two
numbers, so there is nothing in it to fail. Flagged so the check-list's own provenance is clear:
these two entries describe sibling-node output, not this draft's content, and must not be
reported as "confirmed in the draft" by a future pass that doesn't re-open the actual file.
`mean_h 0.666987` is the one check-list number that at first looks like the same situation but
isn't: it is not quoted literally in the draft either, but it IS the literal value inside row
#302, which the draft DOES cite for its derived `−0.0630` — see Check 2. `380/2261 = 16.8%` is a
correct rounding of the draft's own `0.1681`, not a separate unsourced claim.

## Check 5 — internal consistency of what was found

- G-2's own anchor figures (draft §5, lines 111-114) restate the R-MKER-6 anchor at higher
  precision (`1.674660e-03 ± 5e-10` / `1.4956979545757095e-03 ± 1e-15`) than the §0 table's
  `1.6746585172e-03` / `1.4956979546e-03` — consistent rounding, not a conflict.
- The production P6 log's `66/76` and the independently-recounted CRB `76` in-catalogue rows are
  two independent data sources agreeing exactly with each other and with the draft's `10/76`
  arithmetic — strong convergent provenance for the one number (13.2%) I could not verify by
  direct geometry recomputation.
- No other numeric claim inspected (I_1D/I_2D, offsets, md5 pins, commit pin, row/column counts,
  checkpoint counts, event totals) showed any discrepancy against its cited source.

## Overall

**RED** — not because the underlying science is wrong (every re-derivable number checked out,
several to exact bit/value precision), but because the PROVENANCE lens's own bar — "a number
whose source you cannot open is RED" — is failed twice in ways a reader would not catch without
opening row #220 and rd-s3-readout's actual artifacts directly:

1. **Must-fix:** §0's row #220 citation. The p=0.0358/power≈68% clause must cite **row #226**
   (or both #220 and #226, correctly split by clause) — row #220 does not contain these numbers
   and explicitly defers the underlying question to a later, unspecified resolution.
2. **Must-fix:** §2's "+0.587 ± 0.064 per event ... rd-s3-readout" citation. Re-point to the raw
   `b8_cal_harness_work_s4_postflip` checkpoints (`score_at_truth.no_bh.catalogue_hosted`) or to
   `INFORMATION_FORECAST.md:19` — rd-s3-readout's own artifacts (READOUT_RECORD.md,
   CHAIR_REDERIVATION_20260903.md, ledger row #335) do not carry this figure, only `Z=9.76` for
   the same class.

Neither error changes the disposition math in §3-§4 of the draft (both numbers are otherwise
correct), and neither touches the launch block's zero-fresh-choices guarantee — but per the
standing rule ("verifier output is evidence not authority — re-derive, do not trust prose") and
this lens's explicit bar, a citation to a source that does not contain the quoted number is a
provenance failure regardless of whether the number itself survives re-derivation elsewhere.
