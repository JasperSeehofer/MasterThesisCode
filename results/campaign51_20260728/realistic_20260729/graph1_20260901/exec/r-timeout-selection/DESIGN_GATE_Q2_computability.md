# DESIGN_GATE_Q2_computability — r-timeout-selection, `q-timeout-population-mismatch` ONLY

Reviewer: fresh computability-only pass (no prior read of this node's registered statistics; formula
correctness/physics judged out of scope by design). Scope: Q2 exclusively (S2.1–S2.4, the Q2-S2.2 and
Q2-S2.3 disposition rows, the g-* gates as they bind Q2, max_revisions, blindness line). Q1 is explicitly
out of scope per the launch instruction and is not reviewed here; `INFORMATION_FORECAST.md` was not opened
(FORBIDDEN). Source reviewed: `REGISTRATION_DRAFT.md`, `MECHANISM_NOTE.md`, `POOL_MANIFEST.md5`, and every
pinned object on disk (md5/sha256/row-count verified directly, not taken on faith).

**Verdict: RED.** Two of the four Q2 items are not computable as specified without the builder making a
fresh, unregistered choice — one on the PRIMARY statistic (S2.3). One registered replicate input has no pin.
Full detail below; every ambiguity is quoted verbatim from the draft next to the on-disk fact that contradicts
or under-specifies it.

## 1. Enumeration and per-item computability

| # | Q2 item | On disk? | Formula fully specified (zero fresh choices)? | Verdict |
|---|---|---|---|---|
| S2.1 | info map: n/median/IQR of σ_lnDL, Ω, SNR, generation_time per M (and p0/e0) bin; Spearman(log10 M, ln σ_lnDL), 10k-perm p | YES — `seed61000/prepared_cramer_rao_bounds.csv` (md5 verified, see §2) has `M, p0, e0, SNR, generation_time, luminosity_distance, delta_luminosity_distance_delta_luminosity_distance, delta_qS_delta_qS(50), delta_phiS_delta_phiS(59), delta_phiS_delta_qS(58)`; bin edges JSON verified | YES — σ_lnDL and Ω formulas are closed-form over named columns; bin edges pinned and md5-verified | GREEN |
| S2.2 | Spearman(log10 M, d_e) and (log10 M, \|d_e\|), 10k-perm, seed 20260904; top-k M-bin composition vs bulk, Fisher exact/bin, Holm over 5; replicates k=94 (1D), k=72 (joint_r1 2D) | PARTIAL — iiib source (`influence_iiib.csv`) pinned and verified; **`influence_joint_r1.csv`, needed for the k=72 replicate, is NOT in §1's pin table** (exists on disk, unpinned — see §2 finding F1) | MOSTLY — d_e formula, k values, and "Bulk = remaining events" are all inherited by explicit citation ("identical to r-offset-subset §2") and independently confirmed against that section's text; one soft gap: the disposition row names only `ρ_S(log10 M, d_e)` as the M-STRUCTURED/M-FLAT driver, leaving `\|d_e\|`'s registered correlation undesignated as disposition-bearing or purely reported (see F4) | AMBER (F1, F4) |
| S2.3 | **PRIMARY** — `w_b`, per-event `w_e`, Δmean_h^Q2 + σ'_h/σ_h (iiib 2D primary; 1D/joint_r1 replicates), same-size null (1000 draws, seed 20260904) → Δ_null SD; share_to(b) of 822 timeouts (REPORTED-ONLY) | YES — pool a-stratum (99,014 rows, confirmed) for `share_pool,det`; CRB CSV for `share_kept`; 822 timeout dicts confirmed parseable (all carry M, p0; `mu` is a constant 10) for the decomposition; `event_likelihoods.csv` (iiib + joint_r1, both md5-verified) and `tier0_bootstrap_jackknife.py`'s `_moments`/`_physics_floor_apply` confirmed present for the reweighted-posterior assembly | **NO** — the bin-support rule stated in the formula (`n_kept(b) ≥ 10`) is **inconsistent with the two prose descriptions of the same blind spot elsewhere in the same document**, and **no weight is defined for the events this inconsistency strands** (see F2, F3) | **RED (F2, F3)** |
| S2.4 | REPORTED-ONLY: timeouts' (log10 M, p0, mu/M) scatter vs kept | YES — `_parameters_to_dict()` confirmed to log M, p0, mu (constant 10) for all 822 timeout records; kept-set M/p0/mu from the CRB CSV | YES — a scatter/overlay, no aggregate statistic, no band | GREEN |
| Q2-S2.2 disposition | 3-valued, fresh RULE | n/a | tags present (`M-STRUCTURED`/`M-FLAT`/else); fresh-RULE line at bottom of §5 covers it | GREEN, modulo F4 |
| Q2-S2.3 disposition | 3-valued, fresh RULE | n/a | tags present (`POPULATION-MISMATCH-MATERIAL`/`IMMATERIAL`/`INTERMEDIATE`); width band `[0.80, 1.25]` present and listed for ratification (§9 item 1) | GREEN on form; **inherits RED from S2.3's own formula gap (F2/F3) — a disposition cannot be trusted if the statistic feeding it has an unresolved fresh choice** |
| max_revisions | header: "max_revisions 2" | — | present | GREEN |
| Blindness line | §10, with a disclosed partial pre-read of one S2.3 input (item ii) | — | present, and the disclosed pre-read is scoped correctly (one bin of `share_pool,det`, not `w_b`/Δmean_h itself) | GREEN |

## 2. Findings (defects the builder must resolve before launch)

**F1 — `influence_joint_r1.csv` is a registered Q2 input with no pin.** §1's pin table has exactly one
"per-event influence (frozen T0)" row, for `exec/r-offset-subset/influence_iiib.csv` (md5
`d20a01734cc825625f14ba7ec82c67ae`). S2.2 registers "Replicates: 1D (k = 94), joint_r1 2D (k = 72)" — the
joint_r1 replicate requires `influence_2D` from `influence_joint_r1.csv`, which is a different file with no
row in §1 at all. It exists on disk (confirmed: `event_idx,influence_2D,influence_1D,rank`, 1588 rows,
md5 `38f3f1813a3d460093763dd89019ca8a4`) but is invisible to G-1 ("every md5/count in §1; STOP on
mismatch") — there is nothing to STOP against. Per the repo's own dataset-pinning convention and the
lesson this launch instruction quotes ("a missing registered input is a hard INSTRUMENT-DEFECT, never a
silent skip"), an input a registered statistic depends on but that carries no pin is functionally the same
gap: the builder cannot tell, after the fact, whether the file used was the one the registration meant.
**Fix:** add a §1 row pinning `influence_joint_r1.csv` by md5 before launch.

**F2 — S2.3's own bin-support threshold contradicts its own bin-support prose, twice.** The formula reads:
> "`w_b = share_pool,det(b) / share_kept(b)` over bins with `n_kept(b) ≥ 10`"

`g-byteid` pins the per-bin kept counts explicitly: `n_kept = 0/9/1279/304/0` (bins 0–4). Bin 1's `n_kept =
9 < 10` — it fails the formula's own threshold exactly as much as bins 0 and 4 do. Yet the surrounding prose
describes the blind spot as touching only bins 0 and 4:
> S2.3 NOTE: "bins 0 and 4 (0 kept) cannot be re-created — the counterfactual is a bound over the supported
> range (structural blindness, §6)."
> §6 structural blindness (2): "Q2 cannot re-create events in bins with zero kept support — MATERIAL there
> can only be bounded from bins 1–3."

Both of these say or imply bin 1 is *supported* (either by omission from "bins 0 and 4," or by explicit
inclusion in "bins 1–3"). Only bins 2 and 3 (`n_kept` 1279, 304) actually clear `n_kept(b) ≥ 10`. This is not
a stylistic nit: it changes the read's own self-description of its structural blindness from "one contiguous
low-M and one high-M bin excluded, three bins carry the counterfactual" to "three of five bins excluded,
only two carry it" — a materially different claim about how much of the M-axis the PRIMARY statistic actually
speaks to. **Fix:** the registration must pick one number (2 or 3 supported bins) and correct whichever of
the three passages (formula, S2.3 NOTE, §6 item 2) is wrong; a design gate cannot pass with the formula and
its own prose disagreeing on the support set.

**F3 — no weight rule for events in a bin that has kept events but fails the support threshold.** Bin 1 has 9
kept events (per `n_kept = 0/9/1279/304/0`) but, per F2, does not clear `n_kept(b) ≥ 10`, so `w_b` is
undefined there by the formula as written. Nothing in §2 (Definitions) or §4 (S2.3) says what `w_e` those 9
events receive in the reweighted sum. This is not a cosmetic gap: `g-closure` (ii) requires "Q2 `Σ_e w_e =
1588` to 1e-9" — i.e. the renormalization is over *all* 1588 kept events, not just the supported-bin subset —
so the 9 bin-1 events must get *some* weight for the sum to close, and the formula supplies none. A builder
facing this has at least three live options with different numeric consequences (drop the 9 events and
renormalize over 1579; assign them `w_e = 1`; assign them the nearest supported bin's `w_b`) — exactly the
kind of fresh, unregistered choice this review exists to catch, and it sits on the PRIMARY statistic.
**Fix:** the registration must state explicitly what `w_e` is for events in a `n_kept(b) < 10` bin that is
not literally empty (i.e., bin 1 specifically), or must widen the support threshold / redefine bins so no
such bin exists.

**F4 (minor, AMBER not RED) — S2.2 registers two correlations but the disposition row names one.** S2.2's
body registers both `ρ_S(log10 M, d_e)` and `ρ_S(log10 M, |d_e|)`; the Q2-S2.2 disposition row's statistic
column reads only `ρ_S(log10 M, d_e)`. This is very likely intentional (`|d_e|` reported alongside, not
disposition-bearing — consistent with S2.1's single-correlation pattern), but the draft never says so in
words, and a builder should not have to infer it. **Fix:** one clause in S2.2 or the disposition row stating
`|d_e|` is REPORTED-ONLY and does not gate the disposition.

## 3. What checked out clean (verified directly, not assumed)

- `prepared_cramer_rao_bounds.csv`: md5 `9a1f2a14384a9281c97ca3be312ddaab` matches pin; 1590 data rows
  matches pin; all columns S2.1/S2.3 need are present by name (`M, p0, e0, SNR, generation_time,
  luminosity_distance, delta_luminosity_distance_delta_luminosity_distance`, and the qS/phiS covariance
  triple for `Ω`).
- Pool of record: 707 files, 200,100 total rows, 99,014 `stratum == "a"` rows — all three counts reproduced
  exactly by direct read, matching the §1 pin. Header carries `M, SNR, stratum, code_rev` as S2.3 needs.
- `design_gate_bin_edges.json`: md5 `e24b07fe3948559b02d8dd4dbe8df8b3` matches pin; M/p0/e0 edges (6 values
  = 5 bins each) present and internally consistent with the pinned kept p0 range.
- `influence_iiib.csv`: md5 `d20a01734cc825625f14ba7ec82c67ae` matches pin, 1588 rows; header
  `event_idx,influence_2D,influence_1D,rank` — confirms Q2's own d_e formula (computed from `influence_2D` +
  sign, not read from a `d_e` column) is the only workable approach, since no `d_e` column exists on disk
  despite r-offset-subset's own chair notes describing one — Q2's draft already does this correctly.
- `event_likelihoods.csv` (iiib): md5 `8e6a2c18dc5838dd1d52641589243672` matches, 65,108 data rows = 41×1588
  matches; `combined_no_bh`, `combined_with_bh`, `den_log_term` columns present exactly as the PRIMARY
  reweighted-posterior formula needs.
- `event_likelihoods.csv` (joint_r1): md5 `745954a0fdee5f10878fb5e622a06144` matches.
- `cluster_logs_fetch_20260904_MANIFEST.md5`: md5 `ebf09fc4ab66b55e4eb592731ee46ae6` matches; 100/100
  `.err` logs present; direct `grep -c "timed out"` across all 100 sums to exactly **822**, matching the
  draft's 820+2 tally; every sampled timeout line carries a full parameter dict (`M, mu, a, p0, e0, x0, ...`)
  via `_parameters_to_dict()` (confirmed against `datamodels/parameter_space.py:275-291`), so S2.4's
  (log10 M, p0, mu/M) scatter is directly computable with `mu` a fixed constant (10) across all records.
- `M` is confirmed detector-frame `M_z` on both the pool and the CRB CSV by a direct code comment
  (`simulation_detection_probability.py:492-497`: "the injection CSV 'M' column already stores the
  DETECTOR-FRAME mass M_z ..., consistent with the event CRBs"), so binning both tables on the same `M`
  column against the same pinned edges is not a frame-mismatch risk.
- `tier0_bootstrap_jackknife.py`: present, `_moments`, `_physics_floor_apply`, `np.gradient(h_grid)` all
  confirmed by direct read; `_moments(logpost, h_grid, weights)`'s `weights` argument is the **h-grid**
  weight (unrelated to Q2's per-event `w_e`), so the reweighted-posterior formula in §2 Definitions
  (`Σ_e w_e·ln L_e(h)`, applied before the `_moments` call) is a well-defined, zero-fresh-choice composition
  of the existing function with an event-level pre-multiply — confirmed computable exactly as written,
  independent of the F2/F3 gap in *which* `w_e` values are supplied.
- `max_revisions 2` and the blindness/leak-inventory line (§10) are both present, and §10's one disclosed
  S2.3 pre-read (item ii, one bin of `share_pool,det`) is scoped narrowly enough that it does not touch
  `w_b`, `Δmean_h^Q2`, or any disposition-bearing quantity.

## 4. Gates as they bind Q2 (not re-litigating Q1's g-formula/g-hardware, out of scope)

- **G-1 pins:** GREEN for every Q2 input except `influence_joint_r1.csv` (F1).
- **g-byteid:** the `n_kept` values it pins (0/9/1279/304/0) are exactly what exposes F2 — the gate's own
  numbers contradict the surrounding prose. GREEN as a gate (it states the true counts correctly); RED is in
  the draft's *use* of those counts elsewhere.
- **g-population:** GREEN for the Q2-relevant tallies (822 timeouts, 1588×41 scored matrix) — independently
  reproduced.
- **g-closure (ii)** ("Q2 `Σ_e w_e = 1588` to 1e-9"): not satisfiable as stated until F3 is resolved — this
  is the gate that F3 breaks.
- **g-precision:** GREEN — `combined_with_bh`/`combined_no_bh` (full precision) columns confirmed present in
  both event_likelihoods.csv files; no reliance on the 7-s.f. columns needed for Q2.
- **g-scope:** GREEN — nothing in S2.1–S2.4 computes a disposition-bearing statistic on p0 bins outside
  S2.1/S2.4's REPORTED-ONLY carve-out.

## 5. Bottom line for the builder

Do not launch Q2 with S2.3 as written. Resolve F2 (state which bins are actually supported — 2 or 3) and F3
(state the weight rule for any non-empty bin that fails the support threshold) as a single coherent fix,
since they are the same underlying gap read twice. Pin `influence_joint_r1.csv` (F1) before touching it.
Clarify F4 in one sentence. Everything else in Q2 — S2.1, S2.2's core formula, S2.4, both disposition tables'
form, `max_revisions`, and the blindness line — is computable exactly as registered, with zero fresh choices,
against inputs verified on disk in this pass.
