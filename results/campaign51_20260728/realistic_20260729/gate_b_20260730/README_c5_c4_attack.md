# Gate B — adversarial attack on C5 and C4-as-interpretation (2026-07-30)

(`README.md` in this directory belongs to the concurrent Gate C item-2 agent; this
file covers only the `attack_c4_*` / `attack_c5_*` scripts and their `c4_*`/`c5_*`
JSON outputs.)

Target: `../CLAIM_2D_BIAS_20260730.md`, claims **C5** and **C4-as-mechanism**
(`../../RUNBOOK_NEXT_SESSION_6.md` §4 items 1–2). Attacker's brief: try to BREAK them.
All work LOCAL and read-only w.r.t. `master_thesis_code/`; no cluster access, no jobs.
Run from the repo root with `.venv/bin/python`.

## Scripts (read in this order)

| script | what it does | output |
|---|---|---|
| `attack_c4_decomposition.py` | Exact algebraic decomposition of the dark class's +15.83-nat 2D−1D channel difference. Writes each per-event likelihood as `ln p = ln C + ln(1+R)` with `C = (1−w_G)L_comp` (channel-common) and `R = w_G L_cat / C`, so the completion leg **cancels exactly** from the channel difference. Partitions the dark class into 2D-dead / both-dead / survivors. Verifies the mixture identity to 9e-13 and the decomposition closure to 6e-13. | `c4_decomposition_results.json` |
| `attack_c5_rail.py` | C5(a): top-K parabola vertex per railed in-cat event, K = 3,5,7,9. C5(b)(i): leave-one-out / leave-top-k / random-half jackknife on the in-cat class-summed argmax. (b)(ii): per-event Δln p between peak and h = 0.73 vs the uniform-argmax null (1/41). (b)(iii): Poisson class reweighting → combined MAP shift. (c): leg split of the summed profiles. All 10 realistic runs + both idealized baselines. | `c5_rail_results.json` |
| `attack_c5_extrap_validation.py` | Credibility test for (a): applies the *same* top-K parabola extrapolator at standoffs of 8/13/19 grid steps below events whose peak is known, and measures the recovery error. Plus a non-uniform-grid-safe concavity check on the top 7 (uniformly spaced) grid points, and per-event implied Gaussian widths. | `c5_extrap_validation_results.json` |
| `attack_c5_leverage.py` | Pure standalone leg profiles (`Σ ln[w_G L_cat]` vs `Σ ln[(1−w_G)L_comp]`) plus the actual mixture weight `f = w_G L_cat / p`; and the crossing leverage `dh*/dε = −S'_in(h*)/S''_tot(h*)`. | `c5_leverage_results.json` |
| `attack_c5_class_weight.py` | Exact nonlinear leverage: combined MAP with the in-cat class-summed log-likelihood rescaled by λ ∈ {0, 0.5, 1, 1.5, 2}; plus the 2D-channel pure-leg split. | `c5_class_weight_results.json` |

## Gotcha found — worth propagating

**The h grid is NON-UNIFORM**: spacing 0.01 on [0.60, 0.65] and [0.80, 0.86], 0.005 on
[0.65, 0.80]. A plain `np.diff(y, 2)` curvature test over the top 10 points is therefore
meaningless; only the top 7 points (0.80…0.86) are uniformly spaced.
`attack_c5_extrap_validation.py` asserts this. All parabola fits use `np.polyfit(h, y, 2)`
with the true abscissae and are unaffected. (The claim file's Gate-A3 2nd-difference over
0.84/0.85/0.86 is on the uniform part and is fine.)

## Verdicts

- **C5(a) — "the 0.86 pile-up is a prior-bound artifact" is REFUTED; C5's rail SURVIVES.**
  Railed in-cat events extrapolate to finite peaks at h_eff ≈ 0.93–1.05 (median, stable
  over K = 3…9), i.e. 0.07–0.19 *beyond* the grid edge; 24–70% extrapolate past 0.99.
  The extrapolator is validated on known-peak events at the same standoffs: median error
  −0.006…+0.006, IQR 0.014–0.026. Profiles are smoothly concave at the top
  (86–96% of railed events have all 5 second differences negative on 0.80…0.86,
  |d²| ≈ 3–7e-4 ≫ roundoff).
- **C5(b) — "not a centred measurement" is CONFIRMED at class level but the per-event
  framing in the claim is misleading.** Per event the rail is cosmetic
  (median Δln p peak−0.73 = 0.07–0.13 nats; implied σ_h ≈ 0.24–0.31; displacement
  0.30–0.47σ; 0–1.3% of events exceed 1σ). But it is not noise either: the uniform-argmax
  null predicts 2.4% at the edge, observed 54–67%. Summed as a class the in-cat profile is
  displaced **+3.4σ…+6.1σ** above truth (8/10 runs; 2 runs runaway). Dark-only argmax 0.640,
  in-cat-only argmax 0.860, combined 0.700–0.742 → literally a crossing.
- **C5 interpretation — AMENDED.** The in-cat rail is NOT the identified hosts' catalogue
  information preferring 0.86. Their catalogue leg *alone* peaks at 0.760 (1D) / 0.790 (2D)
  and rises only +0.80 / −0.50 nats over 0.73→0.86. The class's +3.92-nat rise is ~84%
  contributed by the ~9%-weight channel-common completion admixture (+33.09 nats).
- **C4-as-mechanism — REFUTED as stated.** The 487 dark events with an identically-zero 2D
  catalogue leg — C4's flagship evidence — carry **+0.24 of the +15.83 nats (1.5%)**.
  98.5% is carried by the 534 *survivors*. The completion term contributes **exactly 0** to
  the channel difference (algebraic cancellation). Amended mechanism: the mass kernel
  *de-weights* the surviving catalogue leg (mean mixture weight f: 0.0354 → 0.0061), so the
  dark class's mixture tilt collapses from −24.46 to −0.63 nats and its argmax moves
  0.640 → 0.785, close to the dark completion leg's own preference of 0.810.
