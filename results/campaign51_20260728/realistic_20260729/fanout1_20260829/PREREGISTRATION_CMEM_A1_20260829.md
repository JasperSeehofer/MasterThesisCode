# PRE-REGISTRATION — [CMEM] Node B2.1 A1: higher-power R2c re-read (bc AND bt arms)

`launched under rows #222/#223 — charter node B2.1` · Fan-out Charter wave 1, branch B2
[CMEM], depth-1 node A1 (`RUNBOOK_NEXT_SESSION_37.md` §2 row "B2 [CMEM]", column 1:
"A1: bc+bt arms, paired within-seed ln-ratio, 10 000 perms, p < 0.01 (free)"). Authorized
by row #221 item 3 (ratification of the [CMEM] higher-power R2c follow-up as the charter's
B2.1) and by the standing [STANDING] grant of row #222 (continue through every consecutive
node of every branch on orchestrator judgement) plus row #223 (production changes inside
the tree are covered too). This is a **fresh registration** (stage 2 of the research
cycle), not an amendment to `PREREGISTRATION_CMEM_READS_20260828.md`, which stays
append-only and unedited.

**Class: structural/composition measurement. ZERO H₀-space reads. Verdict capped
REPORTED-ONLY**, per the original CMEM class designation (row #216 item 4; mirrors the
[HIER] item-9 affordability cap). This node measures the SAME registered probe (R2c: the
combined_no_bh truth-likelihood deficit between catalogued hosts recovered outside vs
inside the recovered localization cone) with (a) a different, doubled-arm fleet, (b) a
paired-mean statistic in place of the original's pooled-median statistic, at the SAME
nominal significance band.

## 0. What changed relative to the ratified CMEM read, and why this is a fresh
registration, not a rerun

The charter text for B2.1 said: "bc AND bt arms ... N_out ≈ 760 outside-cone events over
the banked seeds under `.../p3_b0_work/{bc,bt}_9001NN_work/...`". Per this node's own
instruction to "verify the seed span and the row basis the previous instrument used"
before registering, two load-bearing corrections were made **before** any comparison was
computed:

1. **Different fleet, not a superset.** `p3_b0_work` (venue `b0i`, meta.json
   `"venue": "b0i"`, the b0-identity fleet used by `PREREGISTRATION_B0_IDENTITY_20260823.md`
   / `PREREGISTRATION_P3_WBHZERO_MEASURE_20260825.md`) is a **different simulation run**
   from `p3_2d_fleet_20260825` (venue `b0i2d`) that the original `cmem_reads.py` used —
   confirmed by diffing `bc_900101_meta.json` in both trees: `n_events` 106 (`b0i`) vs 200
   (`b0i2d`), disjoint `work_root`s, disjoint on-cluster provenance
   (`p3_b0_identity_fleet_20260823` vs a 24-seed `p3_2d_fleet_20260825` tree with seeds
   900101-900124). **The two fleets are not nested and their census numbers are not
   expected to agree.** Accordingly, the charter's instruction to "reuse the census
   reproduction gate: 380/2261 on bc must reproduce" is **not applicable as literally
   written** — that number (380/2261, fraction 0.1681) is `p3_2d_fleet_20260825`'s bc
   census, not `p3_b0_work`'s. §5 registers the corrected gate structure.
2. **Seed span is 10 usable seeds, not 12.** `p3_b0_work/{bc,bt}_900111_work` and
   `..._900112_work` have `diagnostics/event_likelihoods.csv` but **no**
   `prepared_cramer_rao_bounds.csv` (confirmed by directory listing: only `diagnostics/`
   is present under those two seeds' `simulations/`, for both arms). This is independently
   corroborated by `PREREGISTRATION_P3_WBHZERO_MEASURE_20260825.md` line ~160: "the
   preserved counterfactual JSON lacks seeds 900111/900112 — their banked runs lack CRB
   artifacts." The flag-recomputation instrument needs `qS`, `phiS`, the sky-Fisher block,
   `host_galaxy_index`, and `in_catalog` — all CRB-only columns — so seeds 900111/900112
   **cannot** be included. **Usable span: seeds 900101-900110, both arms → 20
   (arm, seed) strata**, not the nominal 12.

**Consequence for N_out:** the actual combined outside-cone count (both arms, 10 usable
seeds each) is **N_out = 380** (190 `bc` + 190 `bt`), **not the charter's estimated
≈ 760** — almost exactly half. The most likely source of the ≈ 760 estimate is applying
the ORIGINAL read's per-seed outside rate (380/24 ≈ 15.8/seed) to a wrongly-assumed
24-seed span for `p3_b0_work` (15.8 × 2 arms × 24 ≈ 758), when the actual usable span is
10 seeds. This is disclosed, not silently corrected: **this node is a genuinely smaller
re-read than the charter anticipated**, and its power characteristics (§6) reflect the
true N, not the charter's N.

`bc` and `bt` give **identical** per-seed outside/inside counts for every usable seed
(confirmed at dry-run, §7) — expected, since both arms share event/host realizations and
differ only in the completion-cell likelihood construction downstream of the geometry.
This means `bt` does not add independent cone-classification information; it adds a
second, independently-computed set of `combined_no_bh` values under the same partition,
which is exactly the intended replication for the R2c probe (a truth-likelihood
comparison), not for R2a/R2b (composition, out of scope for this node — see §1).

## 1. Frozen inputs

| input | pin |
|---|---|
| estimator source | `bayesian_statistics.py` + `galaxy_catalogue/handler.py` at HEAD (`a794404c`); same code path as the original CMEM read (`get_possible_hosts_from_ball_tree`, `bayesian_statistics.py:4787`, `sigma_multiplier=1.5`) |
| fleet | `results/campaign51_20260728/realistic_20260729/p3_b0_work/{bc,bt}_9001NN_work/`, venue `b0i`, `catalogue_numerator_survival="off"`, `catalogue_global_selection="phi"`, `mass_filter_sigma` at this fleet's banked (retired asymmetric) resolved-flag set — see caveat below |
| seed span | **900101-900110 (10 seeds), both arms → 20 strata.** Seeds 900111/900112 EXCLUDED (no CRB artifacts; §0). This deviates from the charter's nominal 900101-900112. |
| row basis | posterior-joined subset: CRB rows with `in_catalog == True` AND an `h == 0.73` row present in `diagnostics/event_likelihoods.csv` — identical convention to `cmem_reads.py`'s `if d is None: continue`, confirmed by code inspection |
| catalogue pin | `reduced_galaxy_catalogue.csv` md5 `c52c13b5cab61f6b3f04bbe202550969` (`darksiren_emri/validation/correspondence_1d.py:311`), **verified by direct local `md5sum` this session** (1.68 GB / 20 834 171 rows loaded into `GalaxyCatalogueHandler`) |
| cone convention | chord-length-on-unit-sphere BallTree metric, radius = `1.5 · sqrt(λ_max(J Σ Jᵀ))` on the sky-Fisher block, replicated line-for-line from `handler.py:558-633` (`K = 1.5` matches the production call site's `sigma_multiplier=1.5`) |
| h | 0.73 only (single-h read; same structural blindness as the original CMEM read — §6) |
| statistic column | `combined_no_bh` (per-event, `diagnostics/event_likelihoods.csv`) |

**Disclosed caveat (mass-filter convention):** `mker_r1_census_notes.md` §1 records
`p3_b0_work` as run under the **retired asymmetric** mass-filter window (pre-row
#198-202's `mass_filter_sigma="symmetric"` adoption of 2026-08-25). The sky-cone geometry
this instrument recomputes (chord/radius on the Fisher sky block) is **unaffected** by
`mass_filter_sigma` — that flag only gates the MASS-window candidate pool, a disjoint code
path from `get_possible_hosts_from_ball_tree`'s radius computation. `combined_no_bh`
itself, however, is downstream of whichever candidate pool the ORIGINAL evaluation run
used, so this node's R2c read is at the **retired, non-current-production** mass-filter
default. This is an A10 invariant, not a defect: the read still isolates the sky-cone
outside/inside partition's effect on the truth-likelihood, which is what R2c is about; it
means this result cannot be pooled numerically with a symmetric-window read without
disclosing the flag difference.

**Host draw mode:** confirmed `host_draw_mode == "catalogue_selected"` for 100% of rows in
both fleets (`p3_b0_work` and `p3_2d_fleet_20260825`) — not a differentiator between them.

## 2. Scope: R2c only

This node is scoped to the **R2c** probe only (the charter text: "the higher-power R2c
read"). R2a (catalogue-share `c_share`) and R2b (catalogue-collapse rate) are NOT
registered here and are not computed by the instrument's gated path — only a positivity
sanity check on `combined_no_bh` (C-G2, §5) is run, which is a data-quality gate, not a
composition read. (An exploratory, undisclosed-until-now collapse-rate check WAS computed
during registration prep, purely as a data-plausibility sanity check while designing the
gates — disclosed in full in §8; it is a distinct quantity from R2c and was not used to
tune the statistic, the pairing, or the band.)

## 3. The statistic — pairing definition (registered, justified)

**Statistic:** within-stratum paired mean of `ln(combined_no_bh)`, stratum = one
`(arm, seed)` pair (20 strata: `bc`×10 seeds + `bt`×10 seeds).

For stratum *s* with outside set O_s and inside set I_s:

```
d_s = mean_{e ∈ O_s}[ln combined_no_bh(e)] − mean_{e ∈ I_s}[ln combined_no_bh(e)]
```

**Pooling across strata — PRIMARY: equal weight per stratum.**

```
T = mean_s(d_s)     (unweighted mean over the 20 strata)
```

**Justification for equal-weight as primary:** the registered design calls the statistic
"within-seed PAIRED" — the natural unit of pairing is the stratum (one `(arm, seed)`), not
the event. Equal weighting treats each of the 20 strata as one paired observation,
consistent with a paired-sample framing and robust to the substantial per-seed imbalance
in this fleet (`n_out` ranges 12-29/stratum, `n_in` ranges 89-117/stratum — a 2.4×/1.3×
spread) — a single large-`n_out` stratum (e.g. seed 900108, `n_out=29`) does not dominate
the grand statistic under equal weighting the way it would under event-count weighting.
This also matches the permutation design (§4): labels are permuted independently within
each stratum, so the reference distribution is constructed the same way the strata are
pooled.

**SECONDARY (registered, reported alongside, not verdict-bearing): event-count-weighted
pooling**, weight ∝ `n_out,s`:

```
T_w = Σ_s(n_out,s · d_s) / Σ_s(n_out,s)
```

Both statistics and both permutation p-values are computed and reported by the instrument
(§7's `run_statistic`); **the equal-weight statistic is the one the band (§4) is scored
against.** The event-weighted number is a disclosed sensitivity check.

## 4. Permutations and band

- 10 000 within-stratum label permutations (each of the 20 `(arm, seed)` strata's
  `outside` labels permuted independently; `n_out,s`/`n_in,s` preserved exactly per
  stratum by construction).
- RNG: `numpy.random.default_rng(20260829)`, disclosed (the pre-registration fixes the
  count, not that the seed determines any comparison — same convention as
  `cmem_reads.py`).
- Two-sided: `p = fraction of permuted |T_perm| ≥ |T_obs|`.
- **Band, re-frozen at p < 0.01** (charter instruction; same nominal α as the original CMEM
  read's R2c band). **DISPLACED** if p < 0.01, else **NOT-DISTINGUISHED**.
- **Direction registered:** deficit outside — i.e. a defect-consistent result is
  `T < 0` (mean ln truth-likelihood lower outside the cone than inside), matching the
  original CMEM read's registered direction and its S-SHARP structural finding (Read 1,
  unchanged — this node does not re-run Read 1).

## 5. Gates

Reframed from the original CMEM read's C-G1/C-G2 because the fleet differs (§0); see
`cmem_a1.py` docstring for the implementation-level statement.

- **C-G1a — catalogue pin:** `reduced_galaxy_catalogue.csv` md5 == `c52c13b5cab61f6b3f04bbe202550969`. FAIL ⇒ INSTRUMENT-DEFECT, stop.
- **C-G1b — anchor (NEW, this fleet has no seed 900121):** `bc`/seed 900101/event_idx 0,
  chord = `0.0116656941007181`, radius = `0.0359121946154451` (full-float, computed once
  at registration and frozen; tolerances 5e-10 / 1e-15 matching `cmem_reads.py`'s
  convention). FAIL ⇒ INSTRUMENT-DEFECT, stop.
- **C-G1c — bc/bt cross-arm consistency:** per-seed `(n_out, n_in)` identical between `bc`
  and `bt` for all 10 usable seeds. This is the internal-consistency check that substitutes
  for cross-fleet reproduction (§0 item 1): if it fails, the flag recomputation is not
  deterministic in the shared event/host geometry, which would be a defect regardless of
  which fleet's census "should" be reproduced. FAIL ⇒ INSTRUMENT-DEFECT, stop.
- **C-G1d — seed-span disclosure:** not pass/fail; records `usable_seeds` and
  `missing_crb_seeds` in the gate JSON so a reader can verify §0's claim without re-deriving
  it.
- **C-G2 — positivity sanity:** `combined_no_bh > 0` for 100% of joined rows (required for
  the `ln` transform in §3; a violation here is a hard STOP, not a graded tolerance, unlike
  the original C-G2's `c_share ∈ [0,1]` bound, because a non-positive value makes the
  statistic itself undefined rather than merely imprecise). FAIL ⇒ INSTRUMENT-DEFECT, stop.
- **Census (both arms), reported alongside the gates, NOT itself a pass/fail gate against
  the original 380/2261 number (§0):** this run's own `bc` and `bt` censuses are the NEW
  numbers registered here, to be reproduced bit-for-bit by the independent runner.

## 6. A10 — invariants and blindness sentence

**Invariants (frozen, disclosed above):** estimator source at HEAD `a794404c`; fleet
`p3_b0_work` venue `b0i`; seed span 900101-900110 both arms; row basis = posterior-joined
subset; catalogue pin md5 `c52c13b5cab61f6b3f04bbe202550969` (verified this session);
K = 1.5 cone convention; h = 0.73 only; mass-filter convention = this fleet's own banked
(retired asymmetric) resolved flags, disclosed as non-current-production (§1).

**Blindness sentence:** this is a single-h (h = 0.73) read of a truth-likelihood
comparison; it cannot measure or bound any H₀-direction effect of the composition
mechanism it characterizes, by construction — exactly the same structural blindness as the
original CMEM read (§3 there). A DISPLACED verdict here licenses only a proposal for a
dedicated H₀-space measurement (charter A2, gated on A1 DISPLACED), never a direct H₀
claim.

## 7. A14 — falsifier

Unchanged from the original CMEM read (`PREREGISTRATION_CMEM_READS_20260828.md` §4):
any attribution of a truth-likelihood deficit to the "catalogued host outside the
candidate ball" mechanism is falsified if a future registered arm that re-routes the
dropped weight (e.g. adding the out-of-cone in-catalogue term to `B_num`) fails to move
the outside-cone class's truth-likelihood by the deficit measured here. Registered now;
unrun ⇒ this node's attribution, like the original's, stays provisional even on a
DISPLACED verdict.

## 8. A15 — operating characteristics at the actual N (N_out = 380, N_total = 2336)

All numbers below are computed **without computing the outside/inside split of R2c** —
i.e. without evaluating `d_s` or `T` on the real labels — per the node's own instruction.
Two auxiliary quantities were computed that DO touch the real outside/inside partition
(disclosed, not hidden): the per-seed `n_out`/`n_in` counts (§0, §5 — structural counts,
not an outcome comparison) and, during registration prep, an R2b-style collapse-rate check
(`L_cat_no_bh == 0`) split by outside/inside on this fleet, run as a data-plausibility
sanity check while designing C-G2 (outside 3.16%, inside 0.51% — same direction as the
original read's R2b, weaker ratio: 6.2× here vs 54× on the original fleet). **This
collapse-rate check is R2b, not R2c; R2b is out of this node's registered scope (§2); it
was not used to tune the statistic, the pairing, the permutation design, or the band, and
the actual R2c comparison (`combined_no_bh` split by outside/inside) was never computed by
the builder.** It is disclosed here in full per the standing "list every caveat" mandate,
and the orchestrator/chair should weigh whether it constitutes an inappropriate peek before
treating this registration as untouched.

- **Permutation null:** exact under the null of within-stratum label exchangeability, by
  construction of the design (each stratum's `n_out,s` is preserved under permutation, and
  under H0 the labels are exchangeable within a stratum by the registered model).
  **Caveat on "exact":** 10 000 is a Monte Carlo approximation to the full permutation
  distribution (each stratum has `C(n_s, n_out,s)` distinct label assignments; the full
  product across 20 strata is astronomically larger than 10 000), so the reported p-value
  carries Monte Carlo resolution 1/10 000, not exhaustive-enumeration exactness. This
  qualifies but does not contradict the original prereg's "exact by construction" language,
  which refers to the null MODEL's validity, not the enumeration's completeness.
- **False-fail rate:** 0.01 by construction of the two-sided permutation test at the frozen
  α (type-I error controlled at the nominal level under H0, modulo the Monte Carlo
  resolution above).
- **Detectable deficit at 80% power**, computed from the `bc` CSVs' pooled (unsplit) scatter
  of `ln(combined_no_bh)`: N = 1168 (`bc`, 10 seeds, posterior-joined subset), pooled
  SD = **1.00359** (source: `cmem_a1.py`-equivalent census build, this session; no
  outside/inside split used for this number). Per-stratum `(n_out,s, n_in,s)` from §5's
  gate output feed a standard per-stratum SE `σ·sqrt(1/n_out,s + 1/n_in,s)`, combined under
  each of the two registered pooling schemes:

  | pooling | SE (ln units) | MDE at 80% power, α = 0.01 two-sided (ln units) | MDE as a ratio (exp) |
  |---|---|---|---|
  | **equal-weight (primary, §3)** | 0.05802 | **0.19830** | outside/inside ≈ 0.820 (≈ 18% deficit) |
  | event-count-weighted (secondary) | 0.05668 | 0.19369 | ≈ 0.824 (≈ 17.6% deficit) |
  | naive fully-pooled (optimistic bound, ignores stratum structure — NOT a registered pooling scheme, reported for context only) | 0.05626 | 0.19227 | ≈ 0.825 (≈ 17.5% deficit) |

  (z_{0.005} = 2.5758, z_{0.80} = 0.84162; MDE = (z_{α/2} + z_β)·SE.)

- **Power at the previously-observed effect magnitude:** the original CMEM read's ratified
  R2c result was a median ratio outside/inside of 0.838 (a −16 % deficit, ln = −0.1767),
  reported NOT-DISTINGUISHED at p = 0.0152 on a different statistic (pooled medians) and a
  different fleet. Plugging that magnitude into this node's primary (equal-weight) SE:
  power ≈ **0.68** (68 %) — i.e., **even this doubled-arm, paired-mean design is not
  guaranteed (< 80%) to detect a true effect of the same size the original read reported
  as near-band.** (Event-weighted / naive-pooled give 0.71 / 0.70 — same qualitative
  conclusion.) This is disclosed as a genuine limitation of the "higher-power" framing in
  the charter: the fleet is smaller than the charter assumed (§0), so the power gain over
  the original read is real (a paired mean statistic and log transform are more efficient
  than a pooled-median test) but partial, not decisive.

## 9. Verdict map (unchanged from the original CMEM read's §4, restricted to R2c)

- **R2c DISPLACED** (equal-weight p < 0.01, deficit direction) ⇒ node **B2.1 warrants
  charter node A2** (the cone-widening H₀ counterfactual), per `RUNBOOK_NEXT_SESSION_37.md`
  §2 ("A2 ... only if A1 DISPLACED").
- **R2c NOT-DISTINGUISHED** ⇒ park with the bound reported here (§8); C-STRUCTURAL-ONLY
  stands as the verdict of record for the broader [CMEM] thread (ratified row #220), and
  this node's bound is appended to that record, not substituted for it.
- Any result opposite the registered direction (deficit outside) or S-OTHER-like anomaly in
  the gate/census layer is reported verbatim, per the original read's C-MIXED handling.

## 10. Authorization and independence

**Authorization stamp:** launched under rows #222/#223 — charter node B2.1.
**Builder/runner split (standing rule 2):** this file and `cmem_a1.py` were authored and
smoke-tested (`--dry-run` only) by the builder agent. All gates in §5 PASS at dry-run
(§ Dry-run gate results below). **The registered statistic (§3-§4) was NOT run by the
builder.** A different agent must execute `cmem_a1.py` (without `--dry-run`) to produce the
verdict-bearing R2c comparison.

---

## Dry-run gate results (builder smoke-test, this session)

```
C-G1a catalogue pin:            PASS  (md5 c52c13b5cab61f6b3f04bbe202550969, verified locally)
C-G1b anchor (bc/900101/idx 0): PASS  (chord 0.0116656941007181, radius 0.0359121946154451)
C-G1c bc/bt cross-consistency:  PASS  (10/10 seeds, n_out and n_in identical both arms)
C-G1d seed-span disclosure:     usable = [900101..900110], missing-CRB = [900111, 900112]
C-G2 positivity:                PASS  (0/2336 non-positive combined_no_bh)
Overall gates:                  PASS

Census (registered, new numbers — NOT a reproduction of the original 380/2261):
  bc:       n_outside=190, n_total=1168, fraction=0.16267
  bt:       n_outside=190, n_total=1168, fraction=0.16267
  combined: n_outside=380, n_total=2336, fraction=0.16267
```

Full gate JSON: `cmem_a1_work/cmem_a1_gates.json` (this directory).

---

## SUBMIT + RESULT RECORD (independent runner, this session, 2026-08-29)

`launched under rows #222/#223 — charter node B2.1`

**Runner independence:** this section was produced by a different agent from the one that
authored `cmem_a1.py` and this pre-registration (standing rule 2). The runner did not
modify the instrument (no crash, no fix needed); `cmem_a1.py` sha1 at run time
(`75751f3c71375cec0c4f67d5957a1b5158e1c2b6`) is identical to the file the builder produced.

**Provenance for every number below:** {value, source, date 2026-08-29}. Source file for
all registered-run numbers is
`results/campaign51_20260728/realistic_20260729/fanout1_20260829/cmem_a1_work/cmem_a1_result.json`
(written by `cmem_a1.py` at run time, this session) unless stated otherwise. Full combined
JSON also saved to
`results/campaign51_20260728/realistic_20260729/fanout1_20260829/cmem_a1_result.json`.

### Command and run parameters

```
uv run python results/campaign51_20260728/realistic_20260729/fanout1_20260829/cmem_a1.py
```
(no `--dry-run`; runner-only per the verifier-independence contract in the file's own
docstring). Wall time **59.6 s** (`time`, this session — well inside the 1800 s budget).
`N_PERM = 10000`, `PERM_SEED = 20260829` — {both, `cmem_a1.py:80-81`, hardcoded and
unmodified, 2026-08-29}.

### Gates (evaluated first, per instructions)

All re-derived independently by the runner's own execution (not copied from the builder's
dry-run JSON), and they match the builder's dry-run numbers exactly:

| gate | result | value {source, date} |
|---|---|---|
| C-G1a catalogue pin | PASS | md5 `c52c13b5cab61f6b3f04bbe202550969` {`cmem_a1_work/cmem_a1_gates.json:c_g1a_catalogue_pin`, 2026-08-29} |
| C-G1b anchor (bc/900101/idx0) | PASS | chord `0.01166569410071811`, radius `0.035912194615445196`, both within registered tolerance of the frozen anchor {`cmem_a1_work/cmem_a1_gates.json:c_g1b_anchor`, 2026-08-29} |
| C-G1c bc/bt cross-consistency | PASS | 10/10 usable seeds, `(n_out,n_in)` identical both arms {`cmem_a1_work/cmem_a1_gates.json:c_g1c_bc_bt_cross_consistency`, 2026-08-29} |
| C-G1d seed-span disclosure | recorded (not pass/fail) | usable = [900101..900110], missing-CRB = [900111, 900112] {`cmem_a1_work/cmem_a1_gates.json:c_g1d_seed_span`, 2026-08-29} |
| C-G2 positivity | PASS | 0/2336 non-positive `combined_no_bh` {`cmem_a1_work/cmem_a1_gates.json:c_g2_positivity`, 2026-08-29} |
| **Overall gates** | **PASS** | — |

**Census reproduction — IMPORTANT DEVIATION FROM THE ORCHESTRATOR'S ORIGINAL INSTRUCTION,
disclosed per the caveat mandate.** The runner's launch instruction stated the census
reproduction gate as "bc = 380/2261 exact". That number belongs to the ORIGINAL CMEM read's
fleet (`p3_2d_fleet_20260825`, venue `b0i2d`), not to this node's registered fleet
(`p3_b0_work`, venue `b0i`) — this is exactly the fleet-substitution the pre-registration's
own §0/§5 disclosed and reframed BEFORE this run, and is not a new finding of this
session. The runner evaluated the gate structure actually registered in §5 (C-G1a/b/c/d +
C-G2), not the orchestrator's paraphrase. This session's own **bc census is 190/1168
(fraction 0.16267)**, {`cmem_a1_work/cmem_a1_gates.json:census.bc`, 2026-08-29} — it does
**not**, and is not expected to, reproduce 380/2261 (0.1681). **Coincidence flagged for the
record:** the COMBINED (bc+bt) outside count is 380 — numerically identical to the
original read's bc-only numerator — but over a different, larger denominator (2336 vs
2261) and a different split (190+190 across two arms, not 380 in one arm). This is a
coincidence, not a reproduction; treating it as one would be an [AMBIG]-class error and is
flagged so no future reader makes it.

`bt` census (recorded, per instructions): **190/1168 (fraction 0.16267)**, identical to
`bc`'s, consistent with C-G1c and with §0's expectation that both arms share
event/host-geometry realizations {`cmem_a1_work/cmem_a1_gates.json:census.bt`, 2026-08-29}.

**Identity sanity:** the anchor gate (C-G1b) is the registered identity check for this
fleet (§5, replacing the original read's now-inapplicable seed-900121 anchor); it PASSED
to the registered tolerance (5e-10 chord / 1e-15 radius), confirming the chord/radius
recomputation is deterministic and matches the value frozen at registration.

### Registered statistic (§3-§4), pooled bc+bt, 20 strata — computed BEFORE the per-arm
breakdown below, per instructions

| quantity | value {source, date} |
|---|---|
| n_strata | 20 {`cmem_a1_result.json:n_strata`, 2026-08-29} |
| **Primary (equal-weight) T** | **−0.12311421153794763** {`cmem_a1_result.json:primary_equal_weight.statistic`, 2026-08-29} |
| Primary permutation p (two-sided, 10 000 perms) | **0.0358** {`cmem_a1_result.json:primary_equal_weight.perm_p`, 2026-08-29} |
| Primary displaced? (p < 0.01) | **False** {`cmem_a1_result.json:primary_equal_weight.displaced`, 2026-08-29} |
| Primary direction | deficit-consistent (T < 0) {`cmem_a1_result.json:primary_equal_weight.direction_deficit_outside`, 2026-08-29} |
| Secondary (event-weighted) T_w | −0.10828010490112266 {`cmem_a1_result.json:secondary_event_weighted.statistic`, 2026-08-29} |
| Secondary permutation p | 0.0522 {`cmem_a1_result.json:secondary_event_weighted.perm_p`, 2026-08-29} |
| Secondary displaced? | False {`cmem_a1_result.json:secondary_event_weighted.displaced`, 2026-08-29} |

### Verdict (per §9's registered map)

**R2c NOT-DISTINGUISHED.** Primary equal-weight p = 0.0358 ≥ α = 0.01. The point
estimate is in the pre-registered deficit direction (T = −0.1231 ln, outside/inside ratio
≈ exp(−0.1231) ≈ 0.884, an ≈ 11.6% deficit) and is qualitatively consistent with the
original CMEM read's R2c finding (−16% deficit, also NOT-DISTINGUISHED, p = 0.0152, a
different fleet/statistic), but does **not** cross the frozen band even at this node's
(partially) higher power. Per §9: **park with the bound reported here; C-STRUCTURAL-ONLY
(row #220) stands as the [CMEM] thread's verdict of record**, and this bound is appended
to, not substituted for, that record. Charter node A2 (cone-widening H₀ counterfactual) is
**NOT** warranted by this result under the registered map.

Consistent with the pre-registration's own §8 power analysis: at the originally-observed
effect magnitude (ln ≈ −0.1767, a −16% deficit) this design's power was calculated in
advance at only ≈ 68% (equal-weight), i.e. a NOT-DISTINGUISHED outcome was disclosed in
advance as a plausible, non-decisive result even if the true effect matches the original
read almost exactly — which the observed T = −0.1231 (a somewhat smaller magnitude than
−0.1767) is consistent with, within that pre-registered power ceiling.

### Per-arm breakdown (bc alone, bt alone, pooled) — computed and reported AFTER the
pooled registered read above, per instructions

Not part of the registered verdict (§3 registers only the pooled equal-weight statistic as
verdict-bearing); reported as a disclosed sensitivity/decomposition check, using the
identical stratum-diff definition and permutation scheme as `cmem_a1.py`, restricted to
each arm's own 10 strata. Computed by a runner-authored, non-registered helper script
(`/tmp/.../cmem_a1_breakdown.py`, this session; imports `build_census`/`stratum_diff` from
`cmem_a1.py` unmodified — does not alter the registered instrument). Output saved to
`cmem_a1_work/cmem_a1_breakdown.json`.

| arm | n_strata | equal-weight T | perm p | displaced? | direction |
|---|---|---|---|---|---|
| bc alone | 10 | −0.13148134972146686 | 0.1117 | False | deficit-consistent |
| bt alone | 10 | −0.11474707335442842 | 0.1533 | False | deficit-consistent |
| **pooled (registered)** | 20 | **−0.12311421153794763** | **0.0358** | False | deficit-consistent |

{all values, `cmem_a1_work/cmem_a1_breakdown.json:per_arm_and_pooled`, 2026-08-29}.

**Reading:** both arms individually point the same direction and are individually further
from the band than the pooled read (p ≈ 0.11-0.15 vs pooled 0.0358) — expected, since
pooling both arms doubles the strata count and roughly halves the standard error under the
equal-weight design (§8's power table), without either arm containing independent
cone-classification information (§0: `bc`/`bt` share identical `n_out`/`n_in` per seed by
construction, confirmed again by C-G1c this run). The pooled number is not an artifact of
one arm dominating; the two arms' point estimates (−0.131 and −0.115) are close to each
other and to the pooled value (−0.123).

### Covariate check: z_true medians, outside vs inside (A10 diagnostic, not a gate)

| stratum group | median z_true (outside) | median z_true (inside) | n_out | n_in |
|---|---|---|---|---|
| bc | 0.18959113213702064 | 0.1828732663738775 | 190 | 978 |
| bt | 0.18959113213702064 | 0.1828732663738775 | 190 | 978 |
| pooled | 0.18959113213702064 | 0.1828732663738775 | 380 | 1956 |

{all values, `cmem_a1_work/cmem_a1_breakdown.json:covariate_check_z_true`, 2026-08-29}.
Outside-cone events sit at a very slightly higher median redshift (+0.0067, ≈ 3.7%
relative), consistent with the mechanism's own geometric picture (worse localization,
hence a larger cone-exclusion probability, correlates weakly with distance) but far too
small a shift to be a confound explaining an ≈ 11.6% ln-scale deficit in
`combined_no_bh` on its own; not independently tested against the statistic (no
z-stratified re-run was in scope for this node).

### Exoneration-register check (standing rule 5)

Grepped both `EXONERATION_REGISTER_20260827.md` (case-insensitive, terms: cone, truth
likelihood, `combined_no_bh`, R2c, ball tree, candidate ball, outside/inside) and
`gate_b_20260730/BIAS_HISTORY_LEDGER.md` §2 "DO NOT RE-TRY" (all 17 numbered items, both
layers) before running. **No hit on this node's mechanism** (the outside-cone catalogued
host truth-likelihood deficit, R2c). The [CMEM] thread itself is ratified
C-STRUCTURAL-ONLY (row #220), not exonerated/refuted, so this A1 follow-up is a legitimate
fresh registration, not a re-litigation of a closed item.

### Caveats (full list, nothing omitted)

1. This node is class **REPORTED-ONLY / structural**, single-h (h = 0.73); it makes and
   licenses NO H₀-space claim (§6 blindness sentence, unchanged by this result).
2. Fleet is `p3_b0_work` under the **retired asymmetric** mass-filter convention, not the
   current-production symmetric window (§1 disclosed caveat) — this result cannot be
   pooled numerically with a symmetric-window read.
3. Power at the originally-observed effect size was pre-registered at only ≈ 68%
   (equal-weight); a NOT-DISTINGUISHED outcome here does not rule out an effect near that
   magnitude, it fails to confirm one at the frozen α = 0.01.
4. The "380" combined outside-count numerically coincides with the unrelated original
   read's bc-numerator (380/2261) — flagged above as a coincidence, not a reproduction, to
   forestall a future mis-citation.
5. 10 000 permutations is a Monte Carlo approximation to the full within-stratum
   permutation distribution (§8); reported p-values carry resolution 1/10 000.
6. The A14 falsifier (a dedicated re-routing arm) is registered but unrun; this result, on
   its own, cannot attribute the observed direction to the candidate-ball mechanism with
   certainty even if it had crossed the band.
7. Per-arm breakdown and covariate check are non-registered, disclosed sensitivity/context
   checks (§3, §9 register only the pooled equal-weight number as verdict-bearing) —
   consistent with, not overriding, the registered verdict above.
8. The exploratory R2b-style collapse-rate peek disclosed in the pre-registration's §8 was
   not used to tune anything in this run; flagged again here for completeness per the
   "list every caveat" mandate, not because it affects this result.
