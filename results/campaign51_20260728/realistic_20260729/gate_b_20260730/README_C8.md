# C8 — mass-coordinate reparametrization dependence (Gate B item 4, 2026-07-30)

Target: claim **C8** in `../CLAIM_2D_BIAS_20260730.md`.
Verdict: **PARTIAL** — headline CONFIRMED and reproduced exactly; the claim's
*cause attribution* and its *literal test statement* are both wrong and are
corrected below. Everything here is `[LOCAL, VERIFIED]`; nothing was run on the
cluster and `master_thesis_code/` was not modified.

## Files

| file | what it is |
|---|---|
| `c8_reparam.py` | the numerics: exact reconstruction of the delivered 2D posterior from the diagnostics CSV, the constant-C sweep, alternative-measure evaluations, the sensitivity slope, and the 1D bitwise-invariance check |
| `c8_reparam_results.json` | its machine-readable output |
| `c8_canonical_measure.py` | part (3): is there a canonical mass measure? Computes the dark-population mass likelihood `g_i(z)` from the code's own `mbh_mass_function × R_eff_per_mbh` prior and re-runs the mixture with it. **Indicative estimate, not a ratified physics change.** |
| `c8_canonical_measure_results.json` | its output |

Run both with `.venv/bin/python <file>` from the repo root.

## Exactness gate

`c8_reparam.py` reconstructs `real_r1`'s delivered `combined_posterior_2d.json`
from `diagnostics/event_likelihoods.csv` via
`Σ_i ln[w_G L_cat_with_bh + (1−w_G) L_comp]` with
**max |Δ ln P| = 3.6e-12 nats** over all 41 h. The C-sweep is therefore a faithful
re-computation of the pipeline's own posterior, not a proxy.
The mixture identity itself holds to 9.2e-13 (2D) / 3.9e-13 (1D) relative.

## Structural result (code trace)

`combined_with_bh = (β_G·L_cat_with_bh + B_num)/D(h)` — `bayesian_statistics.py:3306-3311`,
the *same* `β_G`, `B_num`, `D_h` objects the 1D line uses one line earlier.

Mass-density factor count per event (the exponent `s` in `L_cat → C^s L_cat`):

* **2D catalogue leg — exactly one.** `numerator_with_bh_mass` carries `mz_integral`
  (`:4363-4370`), a Gaussian **density in the mass-fraction coordinate**
  `x = M_z/M_z,det,i`; the `mass_trunc` family carries the same thing via an explicit
  push-forward `p_a = p_M · det_M/(1+z)` (`:_mass_trunc_mz_integral`). Its
  normalising partner is mass-**dimensionless** in every mode:
  `Σ_glob(with_bh) = Σ_g w_g p_det` (`:1786`), the local `D_g` = `∫ inner_m·prior_bh dz`
  with `inner_m = ∫p_det N(M)dM` a normalised probability average, and
  `n_hat_w = W_cat/V_f`. ⇒ `s = −1` in **all** normalization modes.
* **Completion leg — none.** `B_num`'s integrand is
  `(1−f)·N(d_L_frac)·sinθ/4π·dVc/(1+z)` (`:3210-3238`) — no mass factor.
* **`D(h)`, `β_G`, `β_Ḡ` — none, and correctly so.** `D(h)` (`:1056-1145`) uses the
  mass-marginalised 3D `p_det`; it is a selection *probability×volume*, not a density
  in the data. It is dimensionless in mass in **both** channels, which is what a
  Mandel–Farr–Gair denominator should be.
* **1D — zero factors.** `handler.py:593` builds the candidate list with a redshift
  filter only (`:595-605` adds the mass filter for 2D); `numerator_without_bh_mass` and
  `denominator_without_bh_mass` contain no mass object at all.

⇒ `β_G·L_cat_2D` (a 4D density) is added to `B_num` (a 3D density). **The
dimensional mismatch is in the numerator, not in `D(h)`.** Mass-marginalising
`D`/`β_G`/`β_Ḡ` alone cannot restore invariance — they carry no mass dimension to
begin with.

## Numerics — the C-walk reproduced

`L_cat_2D → L_cat_2D/C`, everything else fixed (seed61000 real_r1, 1588 events):

| C | claim C8 | this work (parabola) | Δ |
|---|---|---|---|
| 1 | 0.8133 | **0.81329** | 1e-5 |
| 0.3 | 0.7821 | **0.78107** | 1.0e-3 |
| 0.1 | 0.7438 | **0.74440** | 6e-4 |
| ≤0.01 | 0.600 (rail) | **0.60000** (rail) | 0 |

Extra points: C=3 → 0.84096; C≥10 → 0.86000 (rail, = the pure-completion limit).
Sensitivity in the unrailed band C∈[0.05, 3]: **d(MAP)/d(ln C) = +0.031 in h**, i.e.
**3.1 km/s/Mpc per e-fold of the mass unit**.

1D: the summed log-likelihood is **bitwise identical** across the entire sweep
(MAP 0.74050 throughout) — invariance is exact, not approximate.

## Corrections to the claim as written

1. **The naive reading of the test is FALSE.** A *consistent* change of the mass
   unit — catalogue `M_g`, `σ_M,g`, detection `M_z`, `σ_Mz`, the `p_det` M-grid, and
   the mass argument of `R_eff_per_mbh`/`mbh_mass_function` (hence also
   `eddington_shifted_host_mass` and the `_MASS_TRUNC_M_MIN/MAX` bounds) — leaves the
   code **exactly invariant**: mass enters the 2D numerator only through
   the ratios `M_g(1+z)/M_z,det` and `σ_M,g(1+z)/M_z,det`, the 4D covariance row is
   `σ_M²/M²` (`:2408-2435`) with `means_4d = [φ, θ, 1, 1]` (`:2475`), and the
   candidate mass window (`handler.py:595-604`) compares two mass-proportional
   sides. An input-rescaling A/B would report *zero* movement. The real dependence
   is on the **measure** the mixture is taken in — the code has already hard-wired
   one, `dM_z/M_z,det,i`, and the walk is what happens when you pick a different one.
2. **The cause is mis-attributed.** Not "a 4D numerator against a 3D selection
   denominator" — the denominator is fine. It is a 4D numerator leg added to a
   3D numerator leg.
3. **Minor:** the 1D channel is not entirely mass-blind — the rate weight
   `w_g = R_eff_per_mbh(M_g)/(1+z_g)` (`:2996-2999`) uses the host mass. It is a rate,
   not a density, so it is measure-invariant and the 1D C-invariance still holds
   exactly; but "the 1D channel never sees the mass coordinate" is too strong.

## Part (3) — arbitrary, or a fixed physical scale?

The code's implicit mass unit is **the event's own measured detector-frame mass**,
which spans 1.33e5 … 1.63e6 M⊙ across this catalogue (factor 12). It carries no
population meaning, and it is per-event: replacing it by a single constant of the
same geometric mean already moves the MAP by 0.0056. So it is **not** a fixed
physical scale — but the dependence is **not arbitrary either**, because a canonical
choice exists and is computable.

The invariance-restoring object is the completion leg's missing mass-data
likelihood for a dark host at redshift z,
`g_i(z) = ∫φ(M) N(M_z,obs,i; M(1+z), σ_Mz,i) dM ≈ φ(M_z,obs,i/(1+z))/(1+z)`
(σ_Mz/M_z ≈ 1e-4), with φ the population's own source-frame prior
`mbh_mass_function × R_eff_per_mbh` normalised over `[M_SOURCE_FRAME_MIN, MAX]`.
With both legs carrying the same mass density, any measure change multiplies
`p_i` by the same event-wise, h-independent constant ⇒ **MAP exactly invariant**.

Measured (indicative — a scale estimate that evaluates g at `z_i(h)` rather than
re-running B_num's quadrature):

* `g_frac` (in the code's fraction coordinate) median **0.135**, 10–90% [0.108, 0.166].
  The code implicitly uses 1.0 ⇒ the completion leg is **over-weighted ≈ 7.4×**.
* h-frozen `g(0.73)` (the **pure measure** correction): 2D MAP 0.8133 → **0.7558**
  (−0.058; consistent with the constant-C sweep at C≈0.135).
* Full `g(h)`, which additionally injects the dark population's genuine
  mass–redshift information (`Σ_i ln g_i` tilts **+19.0 nats** over 0.73→0.81):
  2D MAP → **0.84917**.

That last number independently reproduces the already-exonerated **HA** correction
(reported 0.8492) to 3e-5, from a completely different starting point. **The
exoneration stands** — HA is not the bias owner, it moves the MAP the wrong way.
What is new is *why*: HA's net wrong sign is the sum of a −0.058 well-posedness
correction and a larger +0.093 mass-function/redshift term. The latter is strongly
model-dependent (Babak M1 mass function) and is the part that deserves scrutiny
before any fix is written.
