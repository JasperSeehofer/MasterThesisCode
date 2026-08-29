# B2.1 [CMEM] A1 — RESULT RECORD

`launched under rows #222/#223 — charter node B2.1`

**Role:** independent runner (standing rule 2). Builder authored/dry-ran
`PREREGISTRATION_CMEM_A1_20260829.md` and `cmem_a1.py`; this session executed the
registered statistic (no `--dry-run`) and is a different agent from the builder.

**Instrument:** `cmem_a1.py`, sha1 `75751f3c71375cec0c4f67d5957a1b5158e1c2b6` — identical
to the builder's file; no modification was needed (no crash).

**Run:** `uv run python cmem_a1.py` (no flags). Wall time 59.6 s (budget 1800 s).
`N_PERM = 10000`, `PERM_SEED = 20260829` (frozen in the instrument, unchanged).

## Gates (evaluated first) — all PASS

| gate | result |
|---|---|
| C-G1a catalogue pin (md5 `c52c13b5cab61f6b3f04bbe202550969`) | PASS |
| C-G1b anchor (bc/900101/idx0, chord/radius to tolerance) | PASS |
| C-G1c bc/bt cross-arm consistency (10/10 seeds) | PASS |
| C-G1d seed-span disclosure | usable 900101-900110, missing-CRB 900111/900112 |
| C-G2 positivity (`combined_no_bh > 0`) | PASS (0/2336 non-positive) |

**Census (this fleet, NOT a reproduction of the original read's 380/2261 — different
fleet, see prereg §0/§1):**
- bc: 190/1168 outside (0.16267)
- bt: 190/1168 outside (0.16267) — identical to bc, expected (C-G1c)
- combined: 380/2336 (0.16267)

**Coincidence flagged:** the combined outside count (380) numerically matches the
unrelated original read's bc-numerator (380/2261) — different denominator, different
split, NOT a reproduction. Noted so it is never mis-cited as one.

**Identity sanity:** C-G1b anchor is this fleet's registered identity check; passed to
tolerance (chord 5e-10, radius 1e-15).

## Registered statistic (pooled bc+bt, 20 strata) — verdict-bearing

| quantity | value |
|---|---|
| Primary (equal-weight) T | **−0.12311421153794763** |
| Primary permutation p (10 000 perms, two-sided) | **0.0358** |
| Displaced (p < 0.01)? | **No** |
| Direction | deficit-consistent (T < 0) |
| Secondary (event-weighted) T_w | −0.10828010490112266 |
| Secondary permutation p | 0.0522 |

## Verdict

**R2c NOT-DISTINGUISHED.** p = 0.0358 ≥ α = 0.01. Direction matches the pre-registered
deficit hypothesis and is qualitatively consistent with the original CMEM read's R2c
(−16% deficit, also NOT-DISTINGUISHED, p = 0.0152, different fleet/statistic), but this
higher-N-strata, paired-mean design still does not cross the frozen band. Per the
registered verdict map (prereg §9): **park with this bound; C-STRUCTURAL-ONLY (row #220)
remains the [CMEM] thread's verdict of record.** Charter node A2 is **NOT** triggered.

This null is not a surprise on its own terms: the pre-registration's own §8 power
analysis disclosed, before this run, that even this doubled-arm design has only ≈ 68%
power to detect an effect as large as the original read's point estimate at α = 0.01 —
so a NOT-DISTINGUISHED outcome was a live, pre-registered possibility even under a true
effect close to what was previously reported.

## Per-arm breakdown (reported after the pooled read, not verdict-bearing)

| arm | n_strata | equal-weight T | perm p | direction |
|---|---|---|---|---|
| bc alone | 10 | −0.13148 | 0.1117 | deficit-consistent |
| bt alone | 10 | −0.11475 | 0.1533 | deficit-consistent |
| pooled (registered) | 20 | −0.12311 | **0.0358** | deficit-consistent |

Both arms individually point the same direction with similar magnitude to each other and
to the pooled estimate; the pooled result is not driven by one arm. Neither arm alone
reaches the band; pooling ~halves the SE (consistent with §8's design), moving p from
~0.11-0.15 (either arm) to 0.036 (pooled) without changing sign or crossing α = 0.01.

## Covariate check: z_true medians (outside vs inside)

| group | median z (outside) | median z (inside) | n_out | n_in |
|---|---|---|---|---|
| bc | 0.189591 | 0.182873 | 190 | 978 |
| bt | 0.189591 | 0.182873 | 190 | 978 |
| pooled | 0.189591 | 0.182873 | 380 | 1956 |

Small (+0.0067, ≈3.7% relative) higher median z outside the cone — directionally
plausible (worse localization slightly correlates with distance) but far too small to
explain an ≈11.6% ln-scale deficit; not stratified against the statistic in this node.

## Exoneration check (standing rule 5)

Grepped `EXONERATION_REGISTER_20260827.md` and `BIAS_HISTORY_LEDGER.md` §2 "DO NOT
RE-TRY" (both layers, mechanism terms not just tags) before running: **no hit** on this
node's mechanism (outside-cone catalogued-host truth-likelihood deficit / R2c). The
[CMEM] thread is ratified C-STRUCTURAL-ONLY (row #220), not exonerated — this A1 follow-up
is a legitimate fresh registration.

## Caveats

1. REPORTED-ONLY / structural class, single-h (0.73); zero H₀-space claim licensed.
2. Fleet under the retired asymmetric mass-filter convention — not numerically poolable
   with a symmetric-window read.
3. Power at the previously-reported effect size was pre-registered at only ≈68%; this
   null does not rule out an effect of that magnitude, only fails to confirm one here.
4. The "380" combined count coincidentally matches the unrelated original bc-numerator —
   not a reproduction (different denominator/split).
5. 10 000 permutations = Monte Carlo approximation (resolution 1/10 000), not exhaustive
   enumeration.
6. A14 falsifier (re-routing arm) registered but unrun; attribution stays provisional
   regardless of this outcome.
7. Per-arm breakdown and covariate check are disclosed, non-registered sensitivity/context
   checks — they do not override the pooled registered verdict.
8. The pre-registration's disclosed exploratory R2b-style collapse-rate peek (§8) was not
   used to tune this run; re-flagged here per the "list every caveat" mandate.

## Provenance

- Prereg + full RESULT RECORD:
  `results/campaign51_20260728/realistic_20260729/fanout1_20260829/PREREGISTRATION_CMEM_A1_20260829.md`
- Instrument: `results/campaign51_20260728/realistic_20260729/fanout1_20260829/cmem_a1.py`
  (sha1 `75751f3c71375cec0c4f67d5957a1b5158e1c2b6`)
- Gate JSON: `cmem_a1_work/cmem_a1_gates.json`
- Registered-statistic JSON (as written by the instrument):
  `cmem_a1_work/cmem_a1_result.json`
- Per-arm/covariate breakdown JSON (runner-authored, non-registered helper):
  `cmem_a1_work/cmem_a1_breakdown.json`
- Combined record JSON (all of the above merged, single file):
  `results/campaign51_20260728/realistic_20260729/fanout1_20260829/cmem_a1_result.json`
