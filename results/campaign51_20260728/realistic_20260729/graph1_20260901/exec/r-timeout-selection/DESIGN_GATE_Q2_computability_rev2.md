# DESIGN_GATE_Q2_computability_rev2 — r-timeout-selection, `q-timeout-population-mismatch` ONLY, REVISION 2 (post-`CHAIR ERRATUM`)

Reviewer: FRESH computability-only pass, independent of the prior `DESIGN_GATE_Q2_computability_rev2.md`
(the file this document replaces at the same path). Per the launch instruction, `DESIGN_GATE_Q2_computability_rev1.md`'s
F5 and AMBER items were read only to confirm each is closed. The prior rev-2 gate's own F6 finding and
the `CHAIR ERRATUM (append-only, 2026-09-04 ~03:35 CEST; closes gate rev2 F6; ...)` section it produced
in `REGISTRATION_DRAFT.md` were both read, since the erratum is now part of REVISION 2's own text and is
the object under test here — every numeric claim in both was independently re-derived from the pinned §1
inputs, not trusted. `MECHANISM_NOTE.md` was read for the code trace it supplies. `INFORMATION_FORECAST.md`
was **not opened** (FORBIDDEN, honored). Q1 is out of scope; no Q1 row below, and the one Q1-scoped
observation this pass turned up (§5) is filed as non-blocking exactly because it is Q1-scoped.

**Blindness line.** No registered Q2 aggregate (S2.1–S2.4, `w_b`, `w_e`, `Δmean_h^{Q2}`, `σ'_h/σ_h`, any
Spearman/permutation/Fisher statistic) was computed over the registered population by this review. What
was run: `md5sum`/`wc -l`/`md5sum -c` (manifest checksum verification) on every pinned §1 object; direct
pandas/numpy loads for header, dtype, and row-count checks; independent `np.digitize(..., right=False)`
binnings of the pinned CRB-CSV `M` column, the pool a-stratum `M` column, and the pinned log-derived
SNR-stage/CRB-stage timeout `M` values against the pinned bin edges (the same class of small,
non-registered reproduction the rev-1 and prior rev-2 gates themselves used); one `grep -c` count of the
D1-gate ("in dervative") log lines and of the ZeroDivisionError log lines. Never S2.1–S2.4 themselves,
never a scorer run (real or `--dry-run`), never a synthetic-table run.

**Verdict: GREEN.** F5 and AMBER (rev-1) and F6 (prior rev-2) are all independently re-verified closed —
REVISION 2's own text closes F5/AMBER exactly as it claims, and the `CHAIR ERRATUM` closes F6 exactly as
it claims, both reproduced to the bit below. This fresh pass over the *entire* Q2 surface (every
statistic, gate, disposition row, and reported-only output) found no new defect that would make a Q2 read
wrong or unregistered. One informational, non-blocking, out-of-scope observation is filed for the chair
in §5 (a label imprecision in `MECHANISM_NOTE.md`'s own context table that touches only a Q1-scoped
gate, `g-closure(i)`, and no Q2 statistic).

## 1. F5 + AMBER closure check (rev-1 findings, as instructed — confirm only, independently re-verified fresh)

| # | rev-1 finding | REVISION 2 text | Independently re-verified (this pass, on disk) | Status |
|---|---|---|---|---|
| F5 | `g-byteid`'s `n_kept` target `0/9/1279/304/0` (from unpinned `rate_table_M.csv`) does not reproduce from §1's own pins + `MECHANISM_NOTE.md`'s kept-population definition | Re-pins `rate_table_M.csv`, re-derives `n_kept` from the pinned CRB CSV + pinned edges restricted to the 1588 scored events → `[0, 9, 1276, 303, 0]` | `md5sum prepared_cramer_rao_bounds.csv` = `9a1f2a14384a9281c97ca3be312ddaab` (exact); `md5sum design_gate_bin_edges.json` = `e24b07fe3948559b02d8dd4dbe8df8b3` (exact); `md5sum rate_table_M.csv` = `b0d6284c06eb2f185158819d47123de5` (exact). Loaded the CRB CSV (1590 rows), took `event_idx` = `influence_iiib.csv`'s index set (1588 rows, md5 `d20a01734cc825625f14ba7ec82c67ae`, exact); the missing pair from `{0..1589}` is exactly `{1203, 1356}` (independently confirmed, not assumed). `np.digitize(M, edges[1:-1], right=False)` on the 1588-row subset → **`[0, 9, 1276, 303, 0]`**, bit-identical to the target; on all 1590 rows → **`[0, 9, 1278, 303, 0]`**, bit-identical to REVISION 2's parenthetical. | **CLOSED** |
| AMBER | §10's "pool a-rows SNR≥20: 82.7% in M-bin 2 vs. kept 80.4%" conflated bin-2-alone with bins-2+3-combined | Restates on one bin set: pool-det bin 2 alone 58.1%, bins 2+3 82.7%; kept bin 2 = 80.4%, bins 2+3 = 99.4% | Loaded all 707 pool files directly (`POOL_MANIFEST.md5` = `75f4030d5d3b0405fd948049bef5767e`, exact; `md5sum -c` on all 707 rows → 100% OK) → 200,100 rows, `stratum` a/b/c = 99,014/50,947/50,139 (exact). `a`-stratum `SNR>=20` → **7,548** rows (exact), binned on the pinned M edges → **`[76, 1217, 4387, 1852, 16]`**. `4387/7548 = 58.12%`; `(4387+1852)/7548 = 82.66%` ≈ 82.7%; kept (from the F5-verified `[0,9,1276,303,0]`): `1276/1588 = 80.35%` ≈ 80.4%; `(1276+303)/1588 = 99.43%` ≈ 99.4%. All four figures reproduce to the quoted precision. | **CLOSED** |

## 2. `CHAIR ERRATUM` closure check (prior rev-2's F6 finding — independently re-verified fresh, not trusted)

**The gap F6 found.** REGISTRATION_DRAFT §3 told Phase A to parse "the 822 timeout dicts" into
`timeouts_seed61000.csv` and match it to `rate_table_M.csv`'s `n_timeout` column ("EXACTLY... Any miss =
INSTRUMENT"), while that pinned column is actually the 820-row SNR-stage-only subset — a silent
population mismatch that (a) trips a false-positive `INSTRUMENT` halt for a literal-822 builder, and (b)
left S2.3's REPORTED-ONLY `share_to(b)` line ("of the 822 timeouts") ambiguous between 820 and 822.

**Direct reproduction, independent of the erratum's own numbers.** Re-parsed the pinned log manifest
(`ebf09fc4ab66b55e4eb592731ee46ae6`; `md5sum -c` on all 2,194 listed files → 100% OK; exactly 100
`simulate_6088772_*.err` files) directly with two separate regexes, one per catch-site log message
(`main.py:763-771` "Waveform/SNR computation timed out" vs. `main.py:812-818` "Cramér-Rao bound
computation timed out"):

```
SNR-stage timeouts, binned:  n = 820,  [206, 302, 216,  81, 15]   (sum 820)
CRB-stage timeouts, binned:  n =   2,  M=576074.3016354897 -> bin 2, M=1950892.8981102726 -> bin 3
All 822 (SNR+CRB), binned:   n = 822,  [206, 302, 217,  82, 15]   (sum 822)
```

The 820-only figure is bit-identical to `rate_table_M.csv`'s pinned `n_timeout` column
(`[206, 302, 216, 81, 15]`, confirmed by direct read in §1 above); the two CRB-stage `M` values are
bit-identical to what REVISION 2/the erratum state. This independently confirms the underlying gap F6
described was real and correctly characterized: a literal "822" build would produce `[206, 302, 217, 82,
15]`, which mismatches the pinned target in bins 2 and 3.

**Does the `CHAIR ERRATUM` close it?** Its text: *"Phase A parses ALL 822 records but bins them by
stage: the SNR-stage 820 form the g-byteid target; the 2 CRB-stage records ... are listed separately as
reported-only and never enter n_timeout or the S2.3 decomposition line, which is restated as 'share_to(b)
of the 820 SNR-stage timeouts (+2 CRB-stage, reported)'."* Checked against both of F6's own listed fix
options: this is fix option 2 verbatim in substance (register the 820-row population for both the
byte-check and `share_to(b)`, correct the "822" wording). It resolves **both** halves of F6:
(i) `g-byteid`'s `n_timeout` target is now unambiguous — a mechanical builder bins `timeouts_seed61000.csv`
by its own `stage` column and checks only the `stage=="snr"` (820) subset against
`[206,302,216,81,15]`, which this pass reproduces bit-identically; no path trips a false-positive
`INSTRUMENT`; (ii) `share_to(b)`'s population is now stated explicitly (820 SNR-stage, primary; 2
CRB-stage, reported alongside) rather than left to infer. Nothing in S2.4 (the timeout scatter, which
already and correctly used all 822 per `MECHANISM_NOTE.md` — reconfirmed unaffected) or in the S2.3
PRIMARY formula (which draws no input from the timeout counts at all — reconfirmed by direct read of the
REVISION 1 formula text: `share_pool,det(b)` from the pool a-stratum, `share_kept(b)` from the CRB CSV,
neither from `timeouts_seed61000.csv`) changes.

**Status: F6 CLOSED**, independently, on the erratum's own terms and against this pass's own
from-scratch reproduction of the 820/822 split.

## 3. Fresh enumeration — every Q2 statistic, gate, disposition row, and reported-only output

| # | item | inputs on disk (pin verified) | zero fresh choices? | verdict |
|---|---|---|---|---|
| S2.1 | info map (n/median/IQR of `σ_lnDL`, `Ω`, SNR, `generation_time` per bin) + Spearman(log10 M, ln `σ_lnDL`), 10k-perm | CRB CSV header confirmed by direct read: `luminosity_distance`, `delta_luminosity_distance_delta_luminosity_distance` (col 42), `delta_qS_delta_qS`/`delta_phiS_delta_qS`/`delta_phiS_delta_phiS` (cols 50/58/59), `SNR` (col 123), `generation_time` (col 124), `T`/`dt` (cols 120/121) — all present by name; bin edges md5-verified | YES — closed-form over named columns | GREEN |
| S2.2 | Spearman(log10 M, `d_e`) + REPORTED-ONLY `\|d_e\|`, 10k-perm seed `20260904`; top-k Fisher/Holm, k=82/94/72 | `influence_iiib.csv` (md5 `d20a01734cc825625f14ba7ec82c67ae`, 1588 rows, verified) and `influence_joint_r1.csv` (md5 `38f3f1813a3d460093763dd89019ca8a`, 1588 rows, verified) both carry `event_idx, influence_2D, influence_1D, rank` | YES — `d_e = sign(0.73 − mean_h)·(−influence_2D_e)`; anchor `mean_h = 0.6658540600 < 0.73` fixes the sign to `+1` unambiguously (`d_e = −influence_2D_e`); F1/F4 (rev-0/rev-1) reconfirmed closed | GREEN |
| **S2.3 PRIMARY** | `w_b`, `w_e`, `Δmean_h^{Q2}`, `σ'_h/σ_h`, same-size null | pool a-stratum (99,014 rows, independently reproduced exact), CRB CSV, both `event_likelihoods.csv` (md5 `8e6a2c18d…`/`745954a0f…`, 65,108 rows = 41×1588 each, both verified) all present; draws no input from the timeout logs | **YES** — F5 closed: `n_kept = [0,9,1276,303,0]` re-derived bit-identical two ways in this pass; support set `{2,3}` (1276+303=1579 of 1588) and the renormalisation rule are fully mechanical | **GREEN** |
| S2.3 decomposition | REPORTED-ONLY `share_to(b)`, now "of the 820 SNR-stage timeouts (+2 CRB-stage, reported)"; D1-gate share (disclosed unmeasurable) | pinned log manifest, `stage` column now explicit per the erratum | **YES, now** — F6 closed (§2): population for `share_to(b)` and for `g-byteid` are the same, stated, 820-row set | **GREEN (F6 closed)** |
| S2.4 | REPORTED-ONLY: timeout `(log10 M, p0, mu/M)` scatter vs. kept | 822 timeout dicts independently re-parsed (820 SNR-stage + 2 CRB-stage); `mu` constant `10` in every sampled record (spot-checked across files) | YES — overlay/scatter, no aggregate, no band; unaffected by the erratum (uses all 822 as before) | GREEN |
| Q2-S2.2 disposition | 3-valued, fresh RULE (`p_perm<0.01` + top-k Fisher/Holm-p<0.05 → MATERIAL; `p_perm≥0.10` → IMMATERIAL; else INTERMEDIATE) | n/a | tags present, fresh-RULE line present (§5 of REGISTRATION_DRAFT: "Fresh RULE on each of the four rows; none pre-decided") | GREEN |
| Q2-S2.3 disposition | 3-valued, fresh RULE, width band `[0.80,1.25]` | n/a | band/tags present; feeds only from the GREEN PRIMARY formula and the now-unambiguous `g-byteid` gate | GREEN |
| `max_revisions` | header: "max_revisions 2" | — | REVISION 1 and REVISION 2 are the two counted revisions (header, `REGISTRATION_DRAFT.md` line 6); F1–F4 closed inside REVISION 1's own text, F5/AMBER closed inside REVISION 2's own text, F6 closed by an appended `CHAIR ERRATUM` — none of the three closures opened a numbered "REVISION 3", consistent with the document's own precedent (F1–F4 were likewise closed without a new revision). Budget not exceeded on its face. | GREEN on form |
| Blindness line | §10, disclosed partial pre-read, AMBER-corrected in REVISION 2 | — | present, internally consistent post-AMBER-fix (§1 above); scope claim ("did NOT compute `w_b`...") accurate and unaffected by F6/erratum | GREEN |

## 4. Gates as they bind Q2

- **G-1 pins:** GREEN — every §1 Q2 input (CRB CSV, bin edges, both `event_likelihoods.csv`, both influence
  CSVs, the pool manifest, the log manifest, and `rate_table_M.csv` per REVISION 2's F5 fix) verified on
  disk to its stated md5 in this pass, exactly. No new unpinned input surfaced.
- **g-byteid:** GREEN, both halves — `n_kept` (F5, reconfirmed `[0,9,1276,303,0]` bit-identical) and
  `n_timeout` (F6, reconfirmed: the erratum's stage-scoping rule makes `[206,302,216,81,15]` the
  unambiguous target for the `stage=="snr"` 820-row subset, bit-identical to `rate_table_M.csv`).
- **g-closure(ii)** (`Σ_e w_e = 1588`): GREEN, unaffected — holds by construction under the global
  renormalisation; no timeout weight enters `w_e`.
- **g-population:** GREEN — every Q2-relevant tally independently re-verified: Σ Y = 89,456 (exact,
  re-summed from the pinned `.err` logs' own "X / Y evaluations successful" lines), 822 = 820 + 2 timeout
  dicts (exact, re-parsed by catch-site message, not by count alone), 4,071 D1-gate lines (exact,
  `grep -c "in dervative"`), pool 200,100 / 99,014 a-rows / 6,000 p0-NaN rows / code_rev split
  194,100+6,000 (all exact), 65,108 = 41×1588 rows in both `event_likelihoods.csv` (exact), kept p0 range
  `[10.0025, 15.987]` (exact against the CRB CSV).
- **g-precision, g-scope:** unaffected by REVISION 2 or the erratum; no input relevant to either changed —
  not re-litigated here.

## 5. Informational, non-blocking, out-of-scope observation (filed for the chair — does not gate Q2)

While independently re-deriving the log-manifest tallies for §2/§4, this pass found that
`MECHANISM_NOTE.md` §3's own table row **"ZeroDivisionError (SNR stage) | 3,488"** does not match a
direct count of that catch site's own log message (`main.py:757-760`, "Caught ZeroDivisionError during
trajectory integration"): that message occurs **3,449** times, not 3,488. Separately, the CRB-stage
catch-all (`main.py:819-825`, "Caught ZeroDivisionError during CRB computation") occurs **39** times —
which is exactly `MECHANISM_NOTE.md`'s own next row, "CRB other (ZeroDiv/Runtime/Value) | 39". `3,449 +
39 = 3,488` exactly: the "3,488" figure is the **combined SNR+CRB stage** ZeroDivisionError total, mislabeled
in the table as SNR-stage-only. This is confined entirely to `MECHANISM_NOTE.md`'s descriptive table (not
a `REGISTRATION_DRAFT.md` §1 pin, not an S2.x formula input) and to `g-closure(i)` (`REGISTRATION_DRAFT
§6`: `89,456 − 3,488 − 85,584 = 384`, and `MECHANISM_NOTE §3(a)`'s own "≈ Y" reconciliation), which is a
**Q1-scoped** gate item — `g-closure(i)`'s `85,584 = 84,762 + 822` term feeds Q1's S1.2 completed-draw
scale factor, not any Q2 statistic. The downstream arithmetic in both places already and consistently
uses the correct **combined** total (3,488), so nothing computationally breaks and no Q2 number moves;
this is a row-label imprecision only. Filed for the chair's awareness (a one-word table-header fix in
`MECHANISM_NOTE.md` §3, e.g. "ZeroDivisionError (SNR 3,449 + CRB 39, combined)"), explicitly **not**
gating this Q2 review and **not** a finding under this launch's own scope fence ("Q1 is NOT in scope").

## 6. Bottom line for the chair

F5, AMBER, and F6 are all genuinely closed, independently re-verified from scratch in this pass rather
than trusted from the prior gate documents. Every Q2 statistic (S2.1–S2.4), both disposition rows, the
`max_revisions` accounting, and the blindness line are computable exactly as registered, with **zero
fresh choices**, against every input verified on disk to its pinned md5/row-count in this pass. The
`n_kept` anchor `[0, 9, 1276, 303, 0]` was re-derived independently from `REGISTRATION_DRAFT.md`'s own
pinned CRB CSV + pinned bin edges, restricted to the 1588 scored events (`event_idx {0..1589} − {1203,
1356}`), and reproduces bit-identically. **Q2 is launchable as registered** (REVISION 2 + the `CHAIR
ERRATUM`). The one item filed in §5 is informational and Q1-scoped; it requires no action before launch.
