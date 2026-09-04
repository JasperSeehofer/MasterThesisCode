# S0-C READER — READOUT RECORD (verdict-free)

Registration: `REGISTRATION_ADDENDUM_PA-HIER-34.md` + `DESIGN_GATE_computability.md`.
Companion data file: `readout_data.json` (all numbers below, machine-readable).
Reader is read-only except this file, `readout_data.json`; no pipeline run, no cluster
command, no edit under `darksiren_emri/`. **All dispositions below are booked
CONDITIONAL-ON-R4 per addendum §5.4 — see "Conditional clause" below; the condition is
NOT lifted.**

## Existence contract (three-valued, as required)

| input | state | evidence |
|---|---|---|
| job 6779532 (h=0.73, 5 nodes) | present, reused, not rerun | `retrieved/s0b_run_20260902/.../event_likelihoods.csv`, 1588 rows/node |
| job 6790794 (h=0.665, h=0.78, 10 tasks) | present, retrieved 2026-09-04, md5 124/124 MATCH | `retrieved/s0c_hgrid_20260904/{h_0p665,h_0p780}/`; sacct 10/10 COMPLETED |
| driver/package unchanged since 081b1f28 | present, confirmed | `git diff --stat 081b1f28 06a12422 -- darksiren_emri/ hier_s0_driver.py` empty; ancestor check OK |
| R4b comparand job 6794207 | present, retrieved, md5 11/11 MATCH | `OPS_RECORD_morning.md` item 5 |
| R4b vs S0-B truth diff | present — **NOT byte-identical** | quoted below |
| N_common across all 15 cells | 1588 (expected 1588), 0 NaN drops | computed here |

## Conditional clause (row #345 D3 / docket R4-R5), quoted

`OPS_RECORD_morning.md` item 5: *"**Verdict: NOT byte-identical.** ... the catalogue-leg
columns (`L_cat_no_bh`, `L_cat_with_bh`, `num_log_term_{no_bh,with_bh}`, and the downstream
`combined_{no_bh,with_bh}`) diverge on the majority of rows (594-1083 of 1588 rows over the
1e-9 threshold, up to relative 1.0 on `L_cat_*`) — consistent with R4b's driver-pinned
deviation ... actually changing the catalogue numerator computation relative to S0-B
truth."* Non-catalogue legs (`D_tilde_phi`, `alpha_G_phi`, `den_log_term`, `B_num*`) matched
to float noise only. Per addendum §5.4: since R4 does **not** reproduce the S0-B truth node
to GATE T-ID precision, **the condition does not lift** — every disposition below "stand[s]
as measurements on the S0-B instrument only and the h-bound does NOT enter the split," by the
addendum's own rule, not a reader judgment.

## g-precision: h=0.73 reproduction (STOP gate — not triggered)

Reproduced from raw `event_likelihoods.csv`, never the driver cache:

| statistic | mean | SEM | Z | target (rows #336/#345) | match |
|---|---|---|---|---|---|
| no_bh score_b_re | −0.68219 | 0.12934 | −5.2744 | −0.6822 / 0.1293 / −5.274 | yes |
| no_bh score_lns | −0.03266 | 0.00454 | −7.1880 | −0.0327 / 0.0045 / −7.188 | yes |
| with_bh score_b_re | −0.74123 | 0.11948 | −6.2036 | −0.7412 / 0.1195 / −6.204 | yes |
| with_bh score_lns | −0.03682 | 0.00502 | −7.3326 | −0.0368 / 0.0050 / −7.333 | yes |

All four reproduce to stated precision. STOP not triggered.

## Gates (scored before any §5 band read; per addendum §6)

| gate | h=0.665 | h=0.73 | h=0.78 | verdict |
|---|---|---|---|---|
| g-score-null (no_bh) \|Z_b_re\|/\|Z_lns\| | 5.91 / 7.93 | 5.27 / 7.19 | 4.87 / 7.50 | REPORTED, all >3 (production = measurement not control, per addendum) |
| C-C identity (n=449 dark events, max\|Δ\|) | 0.0 | 0.0 | 0.0 | GREEN all h |
| GATE ENG (per off-truth node, frac moved ≥1e-6 rel) | 0.56–0.62 | 0.51–0.57 | 0.46–0.53 | GREEN, all ≥10% at all h/nodes |
| g-znorm (selection_tables md5, 5 nodes) | identical | identical | identical | GREEN all h |
| provenance (commit, driver diff-quiet) | 06a12422, ancestor OK, diff-quiet | (job 6779532, 081b1f28) | 06a12422, ancestor OK, diff-quiet | GREEN |
| g-population | N=1588, 0 NaN | 1588, 0 | 1588, 0 | GREEN |

**No INSTRUMENT-DEFECT triggered on any axis/channel/h.**

## Derivative (3-point non-uniform Lagrange stencil at h₀=0.73, Δ₋=0.065, Δ₊=0.050)

Coefficients re-derived independently (matches addendum §4.3 term-for-term, see
`DESIGN_GATE_computability.md` §3 — carried, not re-derived a third time here).

| channel | axis | D̄ | SEM | Z_D | linearity ok? | §5.1 |
|---|---|---|---|---|---|---|
| no_bh (primary) | b_re | 2.7034 | 0.3655 | **7.396** | yes | **RESOLVED** |
| no_bh (primary) | lns | 0.00667 | 0.00922 | **0.723** | no (n/a, unresolved) | **NOT-RESOLVED** |
| with_bh | b_re | 2.9817 | 0.3660 | **8.148** | yes | RESOLVED |
| with_bh | lns | 0.03466 | 0.00992 | **3.494** | **no** | RESOLVED, but linearity fails |

## h-displacement Δh_θ = −S̄(0.73)/D̄

| channel/axis | S̄(0.73) | Δh_θ | SE (delta method) | SE (bootstrap B=2000, seed 20260904) | §5.2 |
|---|---|---|---|---|---|
| no_bh b_re | −0.6822 | **0.2523** | 0.0293 | 0.0304 | \|Δh\|−3SE ≈ 0.16–0.17 ≥ T_mat=0.008 → **MATERIAL-IN-h** |
| no_bh lns | −0.0327 | n/a | — | — | NOT-RESOLVED → report-only one-sided bound \|Δh\|≥0.951 |
| with_bh b_re | −0.7412 | 0.2486 | 0.0268 | 0.0275 | \|Δh\|−3SE ≥ T_mat → MATERIAL-IN-h |
| with_bh lns | −0.0368 | 1.0623 | 0.3173 | 0.4432 | RESOLVED but §4.3 linearity check fails → **INDETERMINATE** (falsifier, addendum §8) |

Bootstrap 3σ bands: no_bh b_re [0.163, 0.346]; with_bh b_re [0.177, 0.334]; with_bh lns
[0.499, 4.174] (wide — consistent with the linearity failure). Fraction of the −0.0641 iiib
mean offset (row #302): no_bh b_re −3.94×, with_bh b_re −3.88× (saturation — Δh_θ exceeds
the whole offset — disclosed per §5.2, not band-bearing).

## Dispositions (exactly §5.1/§5.2, all CONDITIONAL-ON-R4, condition not lifted)

| channel/axis | §5.1 | §5.2 | CONDITIONAL-ON-R4 status |
|---|---|---|---|
| no_bh b_re (primary) | RESOLVED | MATERIAL-IN-h | stands as instrument-only measurement; no h-bound asserted into the split |
| no_bh lns (primary) | NOT-RESOLVED | n/a, report-only bound | stands as instrument-only measurement |
| with_bh b_re | RESOLVED | MATERIAL-IN-h | stands as instrument-only measurement |
| with_bh lns | RESOLVED | INDETERMINATE | stands as instrument-only measurement |

No INSTRUMENT-DEFECT disposition reached on any row.

## Facts for the decider

1. All 5 g-precision h=0.73 targets reproduce exactly; all 6 gate families are GREEN at all
   3 h; no gate failure anywhere in the 15-cell grid.
2. Primary channel: the b_re derivative is RESOLVED and linear (Z_D=7.40); the lns derivative
   is NOT-RESOLVED (Z_D=0.72). The two axes disagree on resolvability.
3. Where resolved and linear, Δh_θ ≈ 0.25 (both channels, b_re) — several times larger than
   the T_mat=0.008 materiality threshold and larger in magnitude than the −0.0641 iiib mean
   offset itself (saturation, disclosed).
4. with_bh lns is RESOLVED by the Z_D≥3 criterion alone but fails the addendum's own §4.3
   linearity check, which by the §8 falsifier rule demotes it to INDETERMINATE — a case the
   addendum anticipated and pre-registered a rule for.
5. R4b (job 6794207) shows the S0-B truth node's catalogue leg is NOT byte-identical to the
   production comparand (594–1083/1588 rows over threshold, up to relative 1.0 on `L_cat_*`);
   per the addendum's own §5.4 rule, this keeps every disposition above CONDITIONAL and
   withholds it from d-residual-attribution item 3 as written — not a reader interpretation.
6. This readout computes and reports every §5.1/§5.2 disposition per addendum §5.4's
   permission ("computing is free"); it does not rule on open item §12 items 1–4, which
   remain for the chair/author.

*No recommendation. Verdict-free per role.*

## ERRATUM D15 (end-verification, 2026-09-04): the R4 condition cites "562/1588" (the c0prime_off comparand count); the R4b diff reports 594–1083 differing rows — different comparands, both true; the CONDITIONAL-ON-R4 clause should cite the R4b figure.
