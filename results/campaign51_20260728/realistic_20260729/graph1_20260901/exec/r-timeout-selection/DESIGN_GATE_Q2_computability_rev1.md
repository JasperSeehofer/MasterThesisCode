# DESIGN_GATE_Q2_computability_rev1 — r-timeout-selection, `q-timeout-population-mismatch` ONLY, REVISION 1

Reviewer: fresh computability-only pass on REVISION 1 (2026-09-04, the "F1–F4" append-only section of
`REGISTRATION_DRAFT.md`). Per the launch instruction, F1–F4 from the rev-0 gate
(`DESIGN_GATE_Q2_computability.md`) were read only to confirm each is closed — every other check below is
independent, on-disk verification, not inherited from rev-0. Scope: Q2 exclusively (S2.1–S2.4, the
Q2-S2.2/Q2-S2.3 disposition rows, the g-* gates as they bind Q2, `max_revisions`, the blindness line). Q1 is
out of scope. `INFORMATION_FORECAST.md` was not opened (FORBIDDEN). No registered aggregate was computed over
the registered population; the pinned CRB CSV, pool files, bin-edges JSON, timeout logs, and influence CSVs
were read directly for headers/counts/md5 and for a small number of independent group-by/binning
reproductions of already-pinned per-bin counts — never a registered Q2 statistic (S2.1–S2.4) itself.

**Verdict: RED.** F1–F4 are each individually closed exactly as worded. But an independent on-disk
reproduction of the per-bin `n_kept` figures that REVISION 1's own F2/F3 fix is built on top of
(`0/9/1279/304/0`, sourced from `rate_table_M.csv`) does **not** reproduce from `REGISTRATION_DRAFT.md`'s own
pinned inputs and its own definition of "kept" — a new, decisive finding (F5) on the PRIMARY statistic (S2.3)
that neither the rev-0 gate nor REVISION 1 caught. Full detail below.

## 1. F1–F4 closure check (as instructed — confirm only)

| # | rev-0 finding | REVISION 1 text | Independently re-verified | Status |
|---|---|---|---|---|
| F1 | `influence_joint_r1.csv` registered (k=72 replicate) with no §1 pin | Adds the pin in the REVISION 1 section (append-only convention, §0 line 9: not edited into §1's table itself, consistent with how F2/F3 are also resolved as append-only text): md5 `38f3f1813a3d460093763dd89019ca8a`, 1588 rows | `md5sum` on disk: **`38f3f1813a3d460093763dd89019ca8a`** — exact match. `wc -l` = 1589 lines = 1588 data rows — exact match. Rev-0's own quoted md5 (`…8a4`, 34 chars) is confirmed **not** a valid md5 (32 hex chars); REVISION 1's correction of the "stray trailing character" is itself correct. | **CLOSED** |
| F2 | Formula's `n_kept(b) ≥ 10` threshold (bin 1 = 9, fails) contradicted the prose ("bins 0 and 4" / "bins 1–3") describing the blind spot | States one number: supported = bins {2,3} only; corrects both prose passages to "bins 2–3 (1583 of 1588 events); bins 0, 1, 4 are unsupported" | Arithmetic self-consistent: on the cited counts `0/9/1279/304/0`, only bins 2 (1279) and 3 (304) clear ≥10; 9 < 10 fails bin 1 as stated. (Whether `0/9/1279/304/0` is itself the *right* set of counts is a separate question — see F5.) | **CLOSED as worded** |
| F3 | No weight rule for bin 1's 9 kept events (support threshold fails but bin is non-empty) | `w_e = 1` for all events in unsupported bins (0, 4: empty; 1: the 9 events), then one global renormalisation `w_e ← w_e·1588/Σw_e` over all 1588 events | Formula is now fully mechanical: assign (`w_b(e)` for bins 2–3, `1` otherwise) → renormalise once. `g-closure(ii)` (`Σw_e = 1588`) holds by construction under *any* pre-renormalisation values (multiplying by `1588/Σ` always yields sum 1588), so this resolution is robust regardless of F5. Zero fresh choices remain in the weight-assignment rule itself. | **CLOSED** |
| F4 | S2.2 registers `ρ_S(log10 M, d_e)` and `ρ_S(log10 M, \|d_e\|)`; disposition row names only the former | One sentence: `\|d_e\|` is REPORTED-ONLY, does not gate the disposition; disposition driven solely by `ρ_S(log10 M, d_e)` + top-k Fisher/Holm | Explicit, unambiguous, matches the §5 disposition row's own statistic column. | **CLOSED** |

F1–F4 are each closed exactly as the launch instruction frames the check. The remainder of this gate is the
fresh pass over every Q2 item, which surfaced a defect none of the four addressed.

## 2. NEW FINDING — F5 (RED): the `n_kept` ground truth F2/F3's fix is built on is not the "kept" population REVISION 1 (or `MECHANISM_NOTE.md`) defines, and does not reproduce from §1's pins

**The claim under test.** REVISION 1 grounds its bin-support determination in a specific pinned figure:
> "on the pinned counts `0/9/1279/304/0` the supported set is bins {2, 3} only"

and the `g-byteid` gate (`REGISTRATION_DRAFT.md` §6) requires this exact figure to be reproduced by the
builder: *"its `n_kept` = 0/9/1279/304/0 ... Any miss = INSTRUMENT."* The source of `0/9/1279/304/0` is
`rate_table_M.csv`'s `n_kept` column (`exec/rd-timeout-bin-seed61000/rate_table_M.csv`) — **a file with no
row, and no md5, anywhere in `REGISTRATION_DRAFT.md` §1** (the same unpinned-registered-input pattern F1
found in `influence_joint_r1.csv`, now on a second file that a live gate — `g-byteid` — treats as ground
truth).

**Direct reproduction from §1's own pinned inputs.** `REGISTRATION_DRAFT.md` defines "kept" precisely once,
in the document it imports as its code trace: `MECHANISM_NOTE.md` §80 — *"The kept event set is `SNR >= 20 ∧
completed(SNR stage) ∧ completed(CRB stage) ∧ D1 p0-window`"* — and again at §107–108, where the 1590-row CRB
population is treated as "kept" and both timeout stages (820 + 2 = 822) are treated as "timeout," never
"kept." Binning the pinned `prepared_cramer_rao_bounds.csv` (md5 `9a1f2a14384a9281c97ca3be312ddaab`,
verified) against the pinned `M_edges` in `design_gate_bin_edges.json` (md5 `e24b07fe3948559b02d8dd4dbe8df8b3`,
verified), using the exact `np.digitize(values, edges[1:-1], right=False)` rule `rd-timeout-bin-seed61000
/analyze.py` itself uses for every other axis table, gives:

```
all 1590 CRB rows ("kept" per MECHANISM_NOTE.md §107-108):        n_kept = [0, 9, 1278, 303, 0]   sum 1590
1588 scored rows (excl. the 2 physics-floor drops {1203,1356},
   the population Σ_e w_e / event_likelihoods.csv actually ranges over): n_kept = [0, 9, 1276, 303, 0]  sum 1588
```

**Neither equals the pinned target `[0, 9, 1279, 304, 0]` (sum 1592).** The +1/+1 discrepancy in bins 2–3
(vs. the 1590-row figure) is traced exactly: `rd-timeout-bin-seed61000/analyze.py:206` builds
`snr_not_to = pd.concat([kept, timeouts[timeouts.stage == "crb"]])` and bins *that* concatenation (line 219)
to produce the column `rate_table_M.csv` calls `n_kept` — silently folding the **2 CRB-stage-timeout**
parameter dicts (`M = 576074.30` → bin 2, `M = 1950892.90` → bin 3; independently re-parsed from the pinned
log manifest, `stage == "crb"` count = 2, matching `MECHANISM_NOTE.md`'s own "CRB-stage `timed out`" row of 2)
into the *kept* side rather than the *timeout* side. This is a legitimate, self-consistent choice for that
script's own question ("rate of loss given SNR-pass"), but it is a **different population** from the one
`REGISTRATION_DRAFT.md` calls "kept" everywhere else — including in the very same REVISION 1 paragraph, which
renormalises `w_e` "over ALL 1588 events." (`n_timeout` is unaffected: SNR-stage-only timeouts, independently
re-binned from the pinned logs, reproduce `[206, 302, 216, 81, 15]` exactly — that half of `g-byteid` is
genuinely GREEN.)

**Why this is RED, not AMBER.** Per the launch instruction's own lesson, "a missing registered input is a
hard INSTRUMENT-DEFECT" — `rate_table_M.csv` is exactly that: an unpinned file whose own numbers cannot be
reproduced by a builder following only `REGISTRATION_DRAFT.md`'s pinned §1 inputs and its own
`MECHANISM_NOTE.md`-cited kept-population definition. A Phase A builder doing the natural thing — binning the
pinned CRB CSV against the pinned edges — gets `[0,9,1278,303,0]` or `[0,9,1276,303,0]`, neither of which
equals the required `[0,9,1279,304,0]`, and trips `g-byteid`'s "Any miss = INSTRUMENT" on a false positive (the
mismatch is a population-definition quirk in an unrelated, unpinned script, not an instrument fault). If
instead a builder copies `rate_table_M.csv`'s numbers uncritically (bypassing the byte-check's intent), the 2
CRB-stage-timeout events — which have no row in `event_likelihoods.csv` / the influence CSVs and can never
receive a real `w_e` — silently inflate the denominator of `share_kept(b)` for bins 2–3, contaminating the
PRIMARY statistic's input with an unregistered population choice. Either path is a computability failure on
S2.3, exactly the category this gate exists to catch. This is independent of, and additional to, F2/F3 — it
sits *underneath* their fix, in the numbers REVISION 1 trusted rather than the threshold logic REVISION 1
wrote.

**Materiality note (does not downgrade the verdict).** The size of the ambiguity is small:
`share_kept(2)` over bins {2,3} is `1279/1583 = 0.80796` (pinned/hybrid), `1276/1579 = 0.80811` (scored-1588,
the population `Σ_e w_e` actually spans), or `1278/1581 = 0.80835` (all-1590) — a spread of ≈0.0005, far
below anything that could move `Δmean_h^{Q2}` near `T_mat = 0.008`. The supported-bin **set** is also robust:
bins {2,3} clear `≥10` and bins {0,1,4} fail it under all three interpretations, so F2/F3's qualitative fix
stands. The verdict is RED on **reproducibility/registration grounds** per this review's own charter ("RED
only for a defect that would make the read wrong or unregistered") — not because the eventual `Δmean_h^{Q2}`
is expected to move materially.

**Fix (either resolves it):**
1. Pin `rate_table_M.csv` by md5 in §1, **and** correct the `g-byteid` target and REVISION 1's "pinned
   counts" line to the CRB-CSV-reproducible figures — `[0,9,1278,303,0]` (all-kept) or `[0,9,1276,303,0]`
   (scored-1588, the population that matches `Σ_e w_e = 1588` and `share_kept(b)`'s actual denominator) — and
   state which of the two is intended for `share_kept(b)`; or
2. Explicitly register that `share_kept(b)`'s denominator is deliberately the "reached-SNR-gate" population
   (kept ∪ CRB-stage timeouts, 1592) rather than the scored/inference population (1588), with one sentence
   explaining why a population that includes 2 events with no `event_likelihood` row and no possible `w_e`
   is the intended denominator for a ratio that reweights only the 1588 — and pin `rate_table_M.csv` either
   way, since a live gate (`g-byteid`) depends on it.

## 3. Minor secondary observation (AMBER, non-blocking — does not touch the registered formula)

§10's disclosed pre-read states "the pool a-rows with SNR ≥ 20 (n = 7,548) have 82.7% in M-bin 2 vs the kept
80.4%." Independent reproduction: pool a-stratum SNR≥20 rows = 7,548 (exact match, `results/campaign51_20260728
/realistic_20260729/gate_b_20260730/injection_pool_mix200k_20260728/`, 707 files, 200,100 rows, 99,014
a-rows — all reproduced directly), binned on the same pinned M edges: `[76, 1217, 4387, 1852, 16]`. Bin 2
alone is `4387/7548 = 58.1%`, **not** 82.7%; bins **2+3 combined** are `6239/7548 = 82.66%` — matching the
quoted 82.7% almost exactly, while the quoted "kept 80.4%" independently reproduces as bin 2 **alone**
(`1276/1588 = 80.4%`, scored population). The two halves of that one sentence appear to describe different
bin sets (kept: bin 2 alone; pool-det: bins 2+3 combined). This sits in §10 (the leak-inventory / disclosed
pre-read), not in S2.3's formula — REVISION 1's own formula text ("`share_pool,det(b)` and `share_kept(b)`
are both computed over bins 2-3 only... before forming `w_b`") is unambiguous and self-consistent
independent of this sentence, so it does not block computability of the registered statistic. Flagged for the
chair to correct in §10 for the record, not gating.

## 4. Fresh enumeration — every Q2 statistic, gate, disposition row, reported-only output

| # | item | on disk? | zero fresh choices? | verdict |
|---|---|---|---|---|
| S2.1 | info map (n/median/IQR of σ_lnDL, Ω, SNR, generation_time per bin) + Spearman(log10 M, ln σ_lnDL), 10k-perm | YES — CRB CSV columns confirmed by direct header read: `delta_luminosity_distance_delta_luminosity_distance` (col 42), `delta_qS_delta_qS`/`delta_phiS_delta_qS`/`delta_phiS_delta_phiS` (cols 50/58/59, the qS/φS block), `SNR`, `generation_time` (cols 123/124), all present exactly as named | YES — closed-form over named columns, bin edges md5-verified | GREEN |
| S2.2 | Spearman(log10 M, d_e) + (log10 M, \|d_e\|), 10k-perm seed 20260904; top-k Fisher/Holm, k=82/94/72 | YES — `influence_iiib.csv` (md5 verified) and `influence_joint_r1.csv` (md5 verified, F1 closed) both present, 1588 rows each | YES — `d_e` formula cited/confirmed against `r-offset-subset` §2; F4 closes the disposition-vs-reported-only ambiguity | GREEN (F1, F4 closed) |
| **S2.3** | **PRIMARY** — `w_b`, `w_e`, `Δmean_h^{Q2}`, `σ'_h/σ_h`, same-size null, `share_to(b)` REPORTED-ONLY | YES — pool a-stratum (99,014 rows, reproduced), CRB CSV, `event_likelihoods.csv` ×2 (md5 verified) all present | **NO** — F2/F3's own fix rests on an `n_kept` figure (`rate_table_M.csv`, unpinned) that does not reproduce from §1's own pinned CRB CSV + bin edges under `MECHANISM_NOTE.md`'s own kept-population definition (F5) | **RED (F5)** |
| S2.4 | REPORTED-ONLY: timeout `(log10 M, p0, mu/M)` scatter vs kept | YES — 822 timeout dicts parseable, `mu` constant 10, confirmed via independent re-parse of the pinned log manifest | YES — overlay/scatter, no aggregate, no band | GREEN |
| Q2-S2.2 disposition | 3-valued, fresh RULE | n/a | tags present, fresh-RULE line present | GREEN |
| Q2-S2.3 disposition | 3-valued, fresh RULE, width band [0.80,1.25] | n/a | tags + band present on form | GREEN on form; **inherits RED from S2.3 (F5)** — a disposition is not trustworthy while its feeding statistic has an unresolved, unpinned population ambiguity |
| `max_revisions` | header: "max_revisions 2" | — | present, REVISION 1 is revision 1 of 2 | GREEN |
| Blindness line | §10, disclosed partial pre-read | — | present; the specific numbers in the pre-read sentence do not internally reconcile (§3 above) but the disclosure's scope claim ("did NOT compute w_b...") is accurate and unaffected | GREEN, with the §3 note filed for the chair |

## 5. Gates as they bind Q2

- **G-1 pins:** GREEN for the F1 fix (`influence_joint_r1.csv` now pinned and verified) and for every other
  §1 Q2 input. **RED**: `rate_table_M.csv` is a second registered-but-unpinned input (F5) that a live gate
  (`g-byteid`) treats as ground truth.
- **g-byteid:** `n_timeout` half GREEN (`[206,302,216,81,15]` independently reproduced exactly from the
  pinned logs). `n_kept` half **RED** (F5) — the pinned target is not reproducible from §1's own inputs under
  the document's own kept-population definition.
- **g-closure(ii)** (`Σ_e w_e = 1588`): GREEN — REVISION 1's global renormalisation (§1 F2/F3 table) makes
  this hold by construction regardless of F5's outcome; F5 affects the *distribution* of `w_e` across bins
  2–3, not the sum.
- **g-population:** GREEN for the Q2-relevant tallies independently reproduced in this pass: 822 timeouts
  (820 snr-stage + 2 crb-stage, both re-parsed directly from the pinned log manifest), 1588×41 scored matrix.
- **g-precision, g-scope:** unaffected by REVISION 1; not re-litigated here (rev-0's GREEN calls stand — no
  input relevant to these two gates changed).

## 6. Bottom line for the builder

F1–F4 are genuinely closed — do not reopen them. Do not launch S2.3 yet: resolve F5 first (pin
`rate_table_M.csv`, and either correct the `g-byteid`/REVISION-1 `n_kept` target to a CRB-CSV-reproducible
figure or explicitly register the "reached-SNR-gate" population as `share_kept(b)`'s intended denominator).
The expected numeric impact on `Δmean_h^{Q2}` is small (§2, materiality note) and the supported-bin set
(bins {2,3}) will not change — this is a registration-completeness defect, not a redesign. Also correct the
§10 pool-det/kept bin-2 sentence (§3) before it is read as a computed cross-check by a later agent. Everything
else in Q2 — S2.1, S2.2's core formula (with F1/F4 closed), S2.4, both disposition tables' form,
`max_revisions`, and the blindness line's scope claim — is computable exactly as registered, with zero fresh
choices, against inputs verified on disk in this pass.
