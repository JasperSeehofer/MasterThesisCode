# DESIGN_GATE_Q2_computability_rev2 — r-timeout-selection, `q-timeout-population-mismatch` ONLY, REVISION 2

Reviewer: fresh computability-only pass on REVISION 2 (2026-09-04, the "F5 + AMBER" append-only section of
`REGISTRATION_DRAFT.md`). Per the launch instruction, `DESIGN_GATE_Q2_computability_rev1.md`'s F5 and AMBER
items were read only to confirm each is closed — every other check below is independent, on-disk
verification, not inherited from rev-1 or rev-0. `MECHANISM_NOTE.md` was read for the code trace it supplies.
`INFORMATION_FORECAST.md` was **not opened** (FORBIDDEN, honored). Q1 is out of scope; no Q1 row below.
**Blindness line:** no registered Q2 aggregate (S2.1–S2.4, `w_b`, `w_e`, `Δmean_h^{Q2}`, `σ'_h/σ_h`, any
Spearman/permutation/Fisher statistic) was computed over the registered population by this review. What was
run: `md5sum`/`wc -l` on every pinned §1 object; direct pandas/numpy loads for header and row-count checks;
independent `np.histogram` **and** `np.digitize(right=False)` binnings of the pinned CRB-CSV `M` column and
the pinned log-derived timeout-`M` values against the pinned bin edges (the same class of small, non-registered
reproduction the rev-1 gate itself used to find F5) — never S2.1–S2.4 themselves, never a scorer run (real or
`--dry-run`), never a synthetic-table run.

**Verdict: RED.** F5 is genuinely closed and the AMBER item is genuinely closed — REVISION 2 fixes both
exactly as it claims, independently reproduced below to the bit. But this fresh pass over the *rest* of the
document surfaced **F6**, a decisive, previously-uncaught defect one register down from F5: the document
names two different timeout populations ("the 822 timeout dicts" vs. `rate_table_M.csv`'s 820-row
`n_timeout` column) under one CSV name and one gate check, with no rule telling a mechanical Phase A builder
which population feeds which artifact. It sits on the PRIMARY statistic's own REPORTED-ONLY decomposition
line and on the shared `g-byteid` gate that a false-positive miss would use to halt the whole node
("Any miss = INSTRUMENT"). The revision budget (`max_revisions 2`, header line 6) is now spent — see §5.

## 1. F5 + AMBER closure check (as instructed — confirm only, independently re-verified)

| # | rev-1 finding | REVISION 2 text | Independently re-verified | Status |
|---|---|---|---|---|
| F5 | `g-byteid`'s `n_kept` target `0/9/1279/304/0` (from unpinned `rate_table_M.csv`) does not reproduce from §1's own pins + `MECHANISM_NOTE.md`'s kept-population definition; two CRB-stage timeouts were folded into "kept" | Re-pins `rate_table_M.csv` (md5 given), re-derives `n_kept` from the pinned CRB CSV + pinned edges restricted to the 1588 scored events → new target `[0, 9, 1276, 303, 0]`; states `share_kept(b)` uses this same 1588-event histogram; retains `rate_table_M.csv`'s `n_kept` column as explicitly NOT a target | `md5sum` on `prepared_cramer_rao_bounds.csv` = `9a1f2a14384a9281c97ca3be312ddaab` (exact match); on `design_gate_bin_edges.json` = `e24b07fe3948559b02d8dd4dbe8df8b3` (exact match); on `rate_table_M.csv` = `b0d6284c06eb2f185158819d47123de5` (exact match). Independently reproduced the 1588-scored histogram TWICE — `np.histogram(M, edges)` and `np.digitize(M, edges[1:-1], right=False)` (the exact convention `rd-timeout-bin-seed61000/analyze.py` uses) — both give **`[0, 9, 1276, 303, 0]`**, sum 1588, bit-identical to the new target; the all-1590-row figure independently reproduces as **`[0, 9, 1278, 303, 0]`**, matching REVISION 2's parenthetical exactly. Grepped the pinned log manifest directly for the 2 CRB-stage timeout params: `M=576074.3016354897` → bin 2, `M=1950892.8981102726` → bin 3 — exactly the "+1/+1 in bins 2–3" REVISION 2 attributes the discrepancy to. `event_idx` set of `influence_iiib.csv` (1588 rows, md5-verified) independently confirms the excluded pair is precisely `{1203, 1356}`. | **CLOSED** |
| AMBER | §10's "pool a-rows SNR≥20: 82.7% in M-bin 2 vs. kept 80.4%" conflated bin-2-alone with bins-2+3-combined across the two populations | Restates on one bin set: pool-det bin 2 alone = 58.1%, pool-det bins 2+3 = 82.7%; kept bin 2 = 1276/1588 = 80.4%, kept bins 2+3 = 1579/1588 = 99.4% | Reproduced the full pool a-stratum directly (707 files, 200,100 rows, 99,014 `stratum=="a"` rows — exact match to §1's pin counts), filtered `SNR>=20` → **7,548** rows (exact match), binned on the pinned M edges → **`[76, 1217, 4387, 1852, 16]`**, sum 7,548 (exact match to the implied source table). `4387/7548 = 58.09%`; `(4387+1852)/7548 = 82.66%` ≈ 82.7%; `1276/1588 = 80.35%` ≈ 80.4%; `(1276+303)/1588 = 99.43%` ≈ 99.4%. All four percentages reproduce to the quoted precision. | **CLOSED** |

F5 and AMBER are each closed exactly as claimed. Neither the PRIMARY `w_b`/`Δmean_h^{Q2}` formula (unaffected
by AMBER, a §10 disclosure sentence) nor the supported-bin set (`{2, 3}`, robust under all three population
interpretations per rev-1's own materiality note) needed to change, and none did.

## 2. NEW FINDING — F6 (RED): "the 822 timeout dicts" and `rate_table_M.csv`'s `n_timeout` column are two different populations, and REVISION 2 pins the second without reconciling the first — a live gap in `g-byteid` and in S2.3's own REPORTED-ONLY decomposition

**The claim under test.** `REGISTRATION_DRAFT.md` §3 instructs Phase A: *"parse the **822 timeout dicts** +
skip tallies → `timeouts_seed61000.csv` (re-derived, **must match** the read's `rate_table_M.csv` `n_timeout`
column **EXACTLY** — `g-byteid`)."* Independently of F5/F6, "822" is used consistently everywhere else in the
document (and in `MECHANISM_NOTE.md`) to mean **both** timeout stages combined: `MECHANISM_NOTE.md` §3's own
table gives "SNR-stage timed out (params logged) | 820" **+** "CRB-stage timed out (params logged) | 2" = 822
(§0's row also reads "820 + 2"); §2's Q1 formula reads `N_to(b)` directly "from the **822** logged param
dicts"; §8 item C and `MECHANISM_NOTE.md` §5 both call the rescue-run target "the **822** logged parameter
sets." There is no passage anywhere that calls 822 anything but the union of both stages.

**Direct reproduction.** Re-parsing the pinned log manifest (`ebf09fc4ab66b55e4eb592731ee46ae6`, verified)
directly, independent of any script in the repo:
```
SNR-stage timeouts (820), binned on the pinned M edges:            [206, 302, 216,  81, 15]   sum 820
CRB-stage timeouts (2; M=576074.30 -> bin 2, M=1950892.90 -> bin 3): adds  +1 to bin 2, +1 to bin 3
All 822 timeout dicts, binned the same way:                        [206, 302, 217,  82, 15]   sum 822
```
`rate_table_M.csv`'s pinned `n_timeout` column (md5-verified `b0d6284c06eb2f185158819d47123de5`) is
`[206, 302, 216, 81, 15]` — the **820-row, SNR-stage-only** figure, not the 822-row union. This is internally
forced by the source script's own logic (independently re-derivable, and consistent with rev-1's F5 finding
about the *complementary* column): the script's `n_kept` column folds the 2 CRB-stage timeouts into "kept"
(F5's finding), so by construction its `n_timeout` column must exclude them — the two columns partition the
same 822 records differently than `MECHANISM_NOTE.md`'s "kept vs. timeout" split does. REVISION 2 reaffirms
this exact 820-based figure as "**still** the phase-A byte-id target for timeouts," without ever stating that
it is a proper *subset* of "the 822 timeout dicts" §3 tells Phase A to parse into the very CSV that gate
checks. **A mechanical Phase A builder who does what §3 literally says — parse the 822 timeout dicts into
`timeouts_seed61000.csv`, then bin that CSV's `M` column by bin to check `g-byteid` — gets `[206, 302, 217, 82,
15]`, which does NOT match `[206, 302, 216, 81, 15]` in bins 2 and 3, and trips `"Any miss = INSTRUMENT"`.**
This is a false-positive INSTRUMENT halt of exactly the same shape F5 was rated RED for (a builder following
the document's own pinned inputs and its own population definitions cannot reproduce a byte-check target that
silently depends on an unstated population restriction) — except this time on the *timeout* side rather than
the *kept* side, and it survives REVISION 2 because REVISION 2's fix was scoped to F5 (the `n_kept` half)
only.

**Why this also touches Q2 directly, not just the shared gate.** S2.3's own registered REPORTED-ONLY
decomposition line reads: *"`share_to(b)` of the 822 timeouts."* If a builder honors "822" literally here
(as every other use of "822" in the document requires), `share_to(b)` is computed from `[206, 302, 217, 82,
15]`/822. If instead a builder — to keep `g-byteid` green — restricts `timeouts_seed61000.csv` to the
820-row SNR-stage-only subset, `share_to(b)` silently becomes an 820-based figure under a label that says
"822," and the 2 CRB-stage timeouts (which, per F5, have no `event_likelihood` row and can never receive a
real `w_e` — the exact population S2.3's own formula already excludes for the *kept* side) are dropped from a
REPORTED number without that choice being written down anywhere. Either resolution is a genuine, undisclosed
fresh choice on a registered Q2 output — the same category of defect the launch instructions' own cited
lesson names directly: *"a missing registered input is a hard INSTRUMENT-DEFECT."* Here the input isn't
missing, but its population boundary is unstated where two different, both-plausible readings give different
numbers.

**Why RED, not AMBER.** Two independent grounds meet the charter's own bar ("RED only for a defect that would
make the read wrong or unregistered"): (1) `share_to(b)` is a registered Q2 statistic (S2.3's decomposition
line) whose defining population is genuinely ambiguous between two on-disk-verified alternatives that differ
by construction — an unregistered fresh choice on its face, independent of any gate; (2) the shared `g-byteid`
gate, read literally, produces a false-positive `INSTRUMENT` stop that (per §3's own "Adjudication... returns
to the author" framing and the Q1-S1.4 "no disposition" precedent for an `INSTRUMENT` verdict) would halt
Phase A's build entirely — and Phase A's `timeouts_seed61000.csv`/`pcomplete_by_bin.csv` products are named in
§3 as feeding the arm generally, so an `INSTRUMENT` trip here is not contained to Q1.

**Materiality note (does not change the verdict, mirrors rev-1's own F5 note).** The PRIMARY statistic itself
— `w_b = share_pool,det(b)/share_kept(b)`, `w_e`, `Δmean_h^{Q2}`, `σ'_h/σ_h` — takes no input from the timeout
dicts at all (REVISION 1/2's formula draws `share_pool,det(b)` from the pool a-stratum and `share_kept(b)`
from the CRB CSV only); F6 cannot move `Δmean_h^{Q2}`. The numeric spread between the two `share_to(b)`
readings is one event in bin 2 and one in bin 3 out of 822 (≈0.1 pp per bin) — immaterial to any threshold.
The defect is on registration completeness and build-gate robustness, exactly the register F5 sat in, not on
the disposition-driving number.

**Fix (either resolves it, same shape as F5's fix, one sentence, no threshold/band touched):**
1. State explicitly that `g-byteid`'s `n_timeout` check is scoped to `timeouts_seed61000.csv[stage == "snr"]`
   (820 rows) — matching what the pinned `rate_table_M.csv` column already, provably, is — and that the 2
   CRB-stage records are retained in `timeouts_seed61000.csv` with a `stage` column for S2.4's scatter and for
   `share_to(b)`, which is computed over **all 822** (state so, one sentence); or
2. Register that `share_to(b)` (and the byte-check) both use the **820-row SNR-stage-only** population, and
   correct S2.3's "of the 822 timeouts" to "of the 820 SNR-stage timeouts (CRB-stage timeouts excluded, no
   `event_likelihood` row, disclosed)" — consistent with how F5 already excludes the same 2 records from the
   *kept* side.
Either fix is a single sentence in an append-only section, identical in cost to how F5 and AMBER were closed.

## 3. Fresh enumeration — every Q2 statistic, gate, disposition row, reported-only output

| # | item | on disk? | zero fresh choices? | verdict |
|---|---|---|---|---|
| S2.1 | info map (n/median/IQR of `σ_lnDL`, `Ω`, SNR, `generation_time` per bin) + Spearman(log10 M, ln `σ_lnDL`), 10k-perm | YES — confirmed by direct header read: `delta_luminosity_distance_delta_luminosity_distance`, `luminosity_distance`, `delta_qS_delta_qS`/`delta_phiS_delta_qS`/`delta_phiS_delta_phiS`, `SNR`, `generation_time`, `T`, `dt` all present by name | YES — closed-form over named columns, bin edges md5-verified (`e24b07fe…`) | GREEN |
| S2.2 | Spearman(log10 M, `d_e`) + REPORTED-ONLY `\|d_e\|`, 10k-perm seed `20260904`; top-k Fisher/Holm, k=82/94/72 | YES — `influence_iiib.csv` (md5 `d20a01734c…`, 1588 rows) and `influence_joint_r1.csv` (md5 `38f3f1813a…`, 1588 rows) both verified on disk, both carry `rank` | YES — `d_e = sign(0.73 − mean_h)·(−influence_2D_e)`, mean_h(anchor)=0.6658540600 < 0.73 so sign is fixed and unambiguous; F1/F4 closed per rev-1, reconfirmed here | GREEN |
| **S2.3 formula** | **PRIMARY** — `w_b`, `w_e`, `Δmean_h^{Q2}`, `σ'_h/σ_h`, same-size null | YES — pool a-stratum (99,014 rows, independently reproduced exact), CRB CSV, both `event_likelihoods.csv` (md5-verified) all present | **YES, now** — F5 closed: `n_kept=[0,9,1276,303,0]` independently re-derived bit-identical two ways; support set `{2,3}` and the 1588-histogram convention for `share_kept(b)` are both explicit; no timeout input in this formula | **GREEN (F5 closed)** |
| S2.3 decomposition | REPORTED-ONLY `share_to(b)` of "the 822 timeouts"; D1-gate share (already disclosed unmeasurable) | YES — pinned log manifest | **NO** — F6: "822" (all-stage) vs. the pinned `n_timeout` column's actual 820-row (SNR-stage-only) population are silently different populations under one instruction | **RED (F6)** |
| S2.4 | REPORTED-ONLY: timeout `(log10 M, p0, mu/M)` scatter vs. kept | YES — 822 timeout dicts independently re-parsed (820 SNR-stage + 2 CRB-stage, matching `MECHANISM_NOTE.md` exactly) | YES — overlay/scatter, no aggregate, no band; a ±2-point scatter difference is immaterial to a plot with no threshold | GREEN (F6 does not gate this — noted for completeness only) |
| Q2-S2.2 disposition | 3-valued, fresh RULE | n/a | tags present, fresh-RULE line present | GREEN |
| Q2-S2.3 disposition | 3-valued, fresh RULE, width band [0.80,1.25] | n/a | band/tags present; feeds only from the now-GREEN PRIMARY formula, not from `share_to(b)` | GREEN on form; **no longer inherits F5** — but the node's `g-byteid` gate as a whole is still RED (below), and an `INSTRUMENT` trip there would prevent this disposition from ever being computed |
| `max_revisions` | header: "max_revisions 2" | — | REVISION 2 is the second of two — **budget now spent**; a fresh RED at this point cannot be closed by a REVISION 3 under the document's own stated cap (see §5) | GREEN on form, flagged for the chair on substance |
| Blindness line | §10, disclosed partial pre-read, AMBER-corrected in REVISION 2 | — | present, internally consistent post-AMBER-fix (verified §1 above); scope claim ("did NOT compute `w_b`...") accurate and unaffected by F6 | GREEN |

## 4. Gates as they bind Q2

- **G-1 pins:** GREEN — every §1 Q2 input (CRB CSV, bin edges, both `event_likelihoods.csv`, both influence
  CSVs, the pool manifest, the log manifest, and now `rate_table_M.csv` per REVISION 2) verified on disk to
  its stated md5, exactly. No new unpinned input surfaced in this pass.
- **g-byteid:** `n_kept` half **GREEN** (F5 closed, reconfirmed above). `n_timeout` half **RED (F6)** — the
  pinned numeric target is correct and reproducible, but the *build instruction* that is supposed to produce
  the CSV checked against it names a different (822-row) population, an unresolved fresh choice that a literal
  reading resolves to a false-positive `INSTRUMENT` stop.
- **g-closure(ii)** (`Σ_e w_e = 1588`): GREEN, unaffected — holds by construction under the global
  renormalisation regardless of F6 (no timeout weight enters `w_e`).
- **g-population:** GREEN — every Q2-relevant tally independently re-verified in this pass: Σ Y = 89,456
  (exact, re-summed from the pinned `.err` logs' own "`X / Y evaluations successful`" lines), 822 = 820 + 2
  timeout dicts (exact, re-parsed), 4,071 D1-gate lines (exact, `grep -c "in dervative"`), pool 200,100 /
  99,014 a-rows / 6,000 p0-NaN rows (all exact), 65,108 = 41×1588 rows in both `event_likelihoods.csv` (exact,
  row-count and md5 both verified), kept p0 range `[10.0025, 15.987]` and kept M range `[1.33e5, 1.63e6]`
  (both exact against the CRB CSV).
- **g-precision, g-scope:** unaffected by REVISION 2; not re-litigated (no input relevant to either changed).

## 5. Bottom line for the chair

F5 and AMBER are genuinely closed — do not reopen them. `g-population`, `g-closure(ii)`, `g-precision`,
`g-scope`, and every G-1 pin are GREEN. S2.1, S2.2 (with F1/F4 already closed), S2.4, both disposition tables'
form, and the S2.3 PRIMARY formula itself are all computable with zero fresh choices, verified directly on
disk in this pass.

Do not launch S2.3's decomposition line or trust `g-byteid`'s `n_timeout` half as written: **F6** is a live,
previously-uncaught population ambiguity ("the 822 timeout dicts" vs. the pinned 820-row `n_timeout` figure)
that a literal Phase A build resolves to a false-positive `INSTRUMENT` stop, and that leaves an undisclosed
fresh choice on the registered `share_to(b)` REPORTED-ONLY output. The PRIMARY `Δmean_h^{Q2}` number is
unaffected and will not move once F6 is fixed (§2 materiality note). The fix is one sentence, the same shape
and cost as F5's and AMBER's own fixes (§2 lists both options) — but **the header's own `max_revisions 2` is
now spent by this REVISION 2**, so a further full "REVISION 3" is not available under the document's own
stated budget. Recommend the chair close F6 the same way `rate_table_M.csv`'s pin and the
`rd-timeout-bin-seed61000/READ_RECORD.md` erratum were both closed — an append-only chair note/erratum
attached to the existing REVISION 2 section rather than a new numbered revision — since neither fix option
touches a threshold, a band, or the PRIMARY formula.
