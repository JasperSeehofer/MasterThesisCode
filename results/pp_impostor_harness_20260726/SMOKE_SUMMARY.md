# Catalogue / impostor-ball harness — coverage SMOKE (2026-07-26)

**Status: SMOKE, NOT A GATE.** 200 realizations per cell. Reproduce with
`.venv/bin/python results/pp_impostor_harness_20260726/run_smoke.py`; raw output in
`smoke_{lcat,absolute,generator_marginal}.json` and `smoke_console.log`.
Derivation: `DERIVATION_HARNESS_ANALOG.md`.

## Configuration

| knob | value |
|---|---|
| `catalogue_mode` | True |
| `n_realizations` × `n_events` | 200 × 120 |
| `n_galaxies` / `sky_frac` | 200 000 / 2·10⁻⁴ |
| `z_support` (completeness edge) | 0.30 |
| `sigma_z` / `sigma_dl_frac` | 0.035 / 0.05 |
| `d50_gpc` / `w_pdet_gpc` | 1.85 / 0.30 (commission venue) |
| kernel / seed | volume / 20260726 |
| injected truths | 0.62, 0.72, 0.84 |

The three estimators are run on **identical universes** (same seed → same frozen catalogue,
same host draws, same caps): the diagnostics columns `comp`, `ball`, `imp`, `hib` are
bit-identical across modes, which is the check that the comparison is paired.

## Result

| mode | h_true | cov50 | cov68 | cov90 | rail | MAP | bias | sd |
|---|---|---|---|---|---|---|---|---|
| `lcat` | 0.62 | 0.375 | 0.535 | 0.760 | 0.055 | 0.6375 | **+0.0175** | 0.0260 |
| `absolute` | 0.62 | 0.395 | 0.610 | 0.795 | 0.025 | 0.6339 | **+0.0139** | 0.0190 |
| `generator_marginal` | 0.62 | 0.420 | 0.625 | 0.810 | 0.035 | 0.6326 | **+0.0126** | 0.0195 |
| `lcat` | 0.72 | 0.260 | 0.415 | 0.555 | 0.010 | 0.7549 | **+0.0349** | 0.0341 |
| `absolute` | 0.72 | 0.295 | 0.465 | 0.690 | 0.000 | 0.7432 | **+0.0232** | 0.0286 |
| `generator_marginal` | 0.72 | 0.300 | 0.465 | 0.680 | 0.000 | 0.7432 | **+0.0232** | 0.0291 |
| `lcat` | 0.84 | 0.120 | 0.300 | 0.600 | 0.775 | 0.8540 | **+0.0140** | 0.0144 |
| `absolute` | 0.84 | 0.195 | 0.350 | 0.665 | 0.640 | 0.8501 | **+0.0101** | 0.0180 |
| `generator_marginal` | 0.84 | 0.200 | 0.350 | 0.665 | 0.645 | 0.8501 | **+0.0101** | 0.0181 |

Universe diagnostics (identical across modes): mean ball size 2.55–2.88 candidates,
**impostor fraction 0.73–0.83**, host-in-ball fraction 0.79 (h=0.62) → 0.44 (h=0.84),
completion fraction 0.21 → 0.56. The impostor-only-ball configuration that V1 targets is
therefore well populated at every truth.

## Readings (all provisional)

1. **The mechanism is now exercised.** Balls carry 2.5–2.9 candidates of which ~78 % are
   impostors, and 21–56 % of events have an uncatalogued host — the configuration the
   pre-existing one-candidate harness could not construct at all. This is the point of the
   extension.

2. **The absolute-mass forms beat the legacy self-normalized `lcat` in every cell.** MAP bias
   drops by 0.012 (h=0.72), 0.004 (h=0.62 → 0.0175→0.0126) and 0.004 (h=0.84); 68 % coverage
   rises by +0.09, +0.05, +0.05; the h=0.84 rail fraction falls 0.775 → 0.64. Power: the
   unpaired binomial SE on a coverage estimate at n=200 is ±0.033 and the MAP-bias SE is
   `sd/√200 ≈ 0.002`; the comparison is paired on identical universes, so the true SE on the
   *difference* is smaller than that. The h=0.72 bias improvement (0.0117, ≥5× the unpaired
   SE) is the most solid of the three. **This is the first harness evidence for V1's core
   impostor-suppression claim** (`DERIVATION_ESTIMATOR_REDESIGN.md` §3.4(b)), which commit
   `7c513dd`'s campaign could not obtain.

3. **`absolute` and `generator_marginal` are indistinguishable here** (identical to 4 decimal
   places at h=0.72 and h=0.84; 0.0013 apart at h=0.62, well inside the noise). Predicted in
   advance by `DERIVATION_HARNESS_ANALOG.md` §5.5: the harness catalogue is drawn from exactly
   the density the estimator models, so production's Option-A identity `Σ_glob = n̂_w β_G`
   nearly holds (measured `n̄_w/n̂_w ∈ [0.92, 0.99]`) and FIX-3 has almost nothing to correct.
   **The harness cannot adjudicate FIX-3 against V1** — that requires the real GLADE+
   catalogue, i.e. the production gates.

4. **A residual HIGH bias survives in all three modes** (+0.010…+0.023, coverage below
   nominal at every level, 64–78 % HIGH rail at h=0.84). V1 attenuates the misassociation
   channel; it does not cure this venue.

## `z_support` sweep: the residual is the completion term, not the impostors

`run_zsupport_sweep.py` (100 realizations, `generator_marginal`, h_true = 0.72,
raw: `zsupport_sweep.json`). Sweeping the completeness edge moves the completion fraction from
0.91 to exactly 0 while the balls get **larger and more impostor-dominated**:

| `z_support` | completion frac | mean ball | impostor frac | cov68 | MAP | bias |
|---|---|---|---|---|---|---|
| 0.15 | 0.906 | 0.39 | 0.759 | 0.41 | 0.7688 | **+0.0488** |
| 0.30 | 0.396 | 2.69 | 0.776 | 0.45 | 0.7408 | **+0.0208** |
| 0.60 | 0.000 | 14.31 | 0.930 | 0.70 | 0.7194 | **−0.0006** |
| 0.95 | 0.000 | 41.06 | 0.976 | 0.63 | 0.7224 | **+0.0024** |

The bias is **monotone in the completion fraction and consistent with zero the moment the
completion fraction reaches zero** — at 14 candidates per ball with 93 % impostors
(`bias = −0.0006`, `sd/√100 = 0.002`), and still at 41 candidates with 97.6 % impostors
(`+0.0024`, ≈1σ). Coverage recovers to 0.63–0.70 against a nominal 0.68 in the same limit.

Two conclusions:

* **The absolute-mass estimator handles impostor-dominated candidate balls without bias.**
  This is a direct, positive validation of the derivation's absolute-scale/tiling identity
  (`DERIVATION_HARNESS_ANALOG.md` Eq. 5, §5.1) in exactly the regime it was built for. Note
  the `n̂_w · sky_frac` normalization has no free parameter: had it been wrong by a constant,
  the h=0.60/0.95 rows could not have come out unbiased.
* **The residual bias is entirely the completion (`B_num`) channel.** It is not the impostor
  channel (it vanishes when the impostor load is maximal) and not the normalization channel
  (`absolute` ≡ `generator_marginal` to 4 decimals). This independently reproduces, in the
  impostor-bearing universe, the prime suspect already named in
  `results/pp_coverage_absolute_20260726/SUMMARY.md` for the single-candidate universe.
  Secondary contributor to check: the 1.5 % above-edge kernel leak of
  `DERIVATION_HARNESS_ANALOG.md` §5.1 (production-faithful but not exact), which also scales
  with the amount of catalogue mass near the edge.

## What this does NOT say

* Nothing about FIX-2 (z-resolved survival): vacuous in this harness by construction
  (`DERIVATION_HARNESS_ANALOG.md` §4).
* Nothing that substitutes for the production-code gates
  (`DERIVATION_GENERATOR_CONSISTENT_NORM.md` §6 gates 2–3, seed600/seed1000).
* No calibration certificate: 200 realizations is a smoke. A gate-grade run needs ≥1000
  realizations per cell and a pre-registered acceptance band.

## Suggested next step

Repeat the `z_support` sweep at n_realizations ≥ 500 across all three modes and all three
truths (to turn the smoke conclusion into a gate-grade statement), then take the `B_num`
completion term itself as the next derivation target — the estimator composition and the
selection normalization are now both exonerated in this harness, and the completion channel
is the only one left carrying the bias.
