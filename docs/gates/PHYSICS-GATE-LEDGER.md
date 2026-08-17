# Physics-change gate ledger

Append-only record of every `/physics-change` hard-gate run. Its purpose is to make gate
compliance **evidence** rather than **inference**: a `[PHYSICS]` commit with no ledger row is a
gate that cannot be shown to have run.

**This ledger starts 2026-07-30.** `[PHYSICS]` commits before that date have no rows and must
not be back-filled — their gate compliance is genuinely unrecorded, and inventing rows would
destroy the property the ledger exists for.

## Row format (stable — do not reorder columns)

```
| YYYY-MM-DD | <commit-ref> | <step> | <verdict> | <target> | <note> |
```

| Field | Values |
|---|---|
| `YYYY-MM-DD` | date the step completed |
| `<commit-ref>` | short SHA once committed, or `pre-commit` if the commit does not exist yet |
| `<step>` | `presented` (the 5-item gate was put to the user) · `implemented` (code written after approval) · `verified` (post-implementation checks reported) |
| `<verdict>` | `APPROVED` · `REJECTED` · `PASS` · `FAIL` · `WAIVED` (with a reason in `<note>`) |
| `<target>` | `file.py:line` or `file.py` — the physics file changed |
| `<note>` | one clause: what changed, or why waived |

Greppable: every ledger row starts with `| 20`.

```bash
grep '^| 20' docs/gates/PHYSICS-GATE-LEDGER.md          # all rows
grep '^| 20' docs/gates/PHYSICS-GATE-LEDGER.md | grep FAIL
git log --oneline --grep='^\[PHYSICS\]'                 # cross-check against commits
```

A complete gate run leaves three rows (`presented` → `implemented` → `verified`) sharing a
target; a run that stopped at `REJECTED` leaves one. `pre-commit` rows should be updated to the
real short SHA when the commit lands — the trailing `<note>` is free text, the first five fields
are not.

## Ledger

| Date | Commit | Step | Verdict | Target | Note |
|---|---|---|---|---|---|
<!-- APPEND NEW ROWS BELOW THIS LINE — newest last -->
| 2026-08-04 | pre-commit | presented | APPROVED | bayesian_statistics.py:4226-4293 | Fix A C7-core: host-z volume_deconv kernel gains f_k(z) selection weight (GATE_PACKAGE_FINAL.md §1.2); author approved with honest framing (rail persists, 1D moves down) |
| 2026-08-04 | pre-commit | presented | APPROVED | bayesian_statistics.py:3296-3331 | Fix B C8 half: 2D completion leg mass density g_i (measure-invariance PROVEN); C9 half NOT presented — blocked on gates ii-b/ii-c, author asked for measurement rationale |
| 2026-08-04 | pre-commit | presented | APPROVED | bayesian_statistics.py (selection stack + mixture) | Fix B path-A joint C9+C8 (FIXB_PATHA_PACKAGE.md §3): S̄_φ replaces fitted S_3D in all three slots, D^φ, g-inside, w̃_G=α_G^φ/D̃^φ. Author decisions: D1=both (S_and now, retire stale p0 bounds next campaign; p0-window onto 2D-bias suspect list), D2=delivered-convention pins primary with MANDATORY promotion to truth once truth Σ4D(h) measured at 41h, D3=point form. Gate (ii) demoted to monitored consistency number; ship-on-correctness |
| 2026-08-04 | pre-commit | implemented | PASS | bayesian_statistics.py:4351 | C7-core: host-z volume_deconv kernel carries the catalogued-host intensity f_k(g)(z)·w_pop(z) in numerator + Z_g/D_g (batched :4351/:4437/:4656, scalar twin :3865); ZoA all-zero-window falls back to the pre-C7 kernel, warn once, no elementwise clamp; ratified in GATE_PACKAGE_FINAL.md §1 (author-approved 2026-08-04) |
| 2026-08-04 | pre-commit | verified | PASS | bayesian_statistics.py:4351 | Sign: γ_f = dln f/dln z ≤ 0 ⇒ the kernel's z-weight and h_eff can only move DOWN, bounded by f ≡ 1 (measured Δ/e² = −0.400 ± 5% over e = 0.04/0.02/0.01, matching the stub's analytic γ_f(0.1) = −0.4). Units: f dimensionless ∈ (0,1] ⇒ ρ_g still a unit-mass density in z; N_g, D_g, Z_g, Σ_glob, w_G unit-unchanged. Limits: f ≡ 1 byte-identical to HEAD (all 3 modes × both channels); σ_z → 0 exact and f-independent (gap monotone, 1.2e-2 → 2.0e-3 over σ_z = 6e-3 → 1.5e-3, residual is the known n=50 numerator-window aliasing floor); kernel h-invariance exact (f_k h-independent to <1e-10 via the m_star cancellation, w_pop·f_k ratio z-separable to <1e-10); w_G = β_G/D bit-identical (pure quadrature) and the #51 point-kernel numerator bit-identical. Pins moved: PIN_VD/PIN_VT/PIN_CLAMP (see the test file's re-pin block); PIN_LR unchanged. Batched/scalar parity rtol 1e-9 with completeness threaded through child_process_init. |

| 2026-08-04 | pre-commit | implemented | PASS | bayesian_statistics.py (selection stack + mixture), dark_siren_injection.py | S̄_φ = ∫φ S_4D dlog10M on the production pooled-2D with-BH object (pdet_wbh_z_resolved=False); new φ-convention tables β_G^φ/β_Ḡ^φ/Σᶲ/D^φ consumed by absolute_marginal ONLY (legacy tables + generator_marginal byte-identical, gate iii-a); mixture w̃_G=α_G^φ/D̃^φ, α_G^φ=β_G^φ·r_Malm; (N8) B_num_wbh with g_i inside the quadrature, 1D B_num unmultiplied; φ exported once from dark_siren_injection (never re-typed); point-mass evaluation (D3); instrumentation (w_G RENAMED to w_G_legacy, 7 s.f. diagnostics, T9 Σ4D band shares, g_i support-exit); monitored gate (ii) under S_and (ρ=0.7305) |
| 2026-08-04 | pre-commit | verified | PASS | bayesian_statistics.py (selection stack + mixture) | Dimensions: S̄_φ dimensionless, β^φ/D^φ in [p_pop dz], r_Malm/r_φ dimensionless, g_i the ONLY x_M density (2D completion only) — no leg gains/loses a mass measure. Signs: S̄_φ ∈ [0,1] monotone in d_L; α_G^φ, D̃^φ > 0; w̃_G ∈ (0,1). Anchors reproduced on the pool/catalogues of record: β_G^φ=1.533228e8, β_Ḡ^φ=8.884038e8, D^φ=1.041727e9, Σᶲ_del=9.5623703e8, r_Malm_del=0.4415122, w̃_G=0.0708023 — all ≤4e-8 rel. Gates: (i) 2D exactly homogeneous in x_M at 1e-14 and h-independent (dMAP/dlnC=0); (iii-a) #51 1D digest + generator_marginal unmoved; (iv) 1D bitwise invariant; T8 r_φ≡1 exact; L4 s=0 tilt vanishes; L5 σ_Mz→0 finite=point form; falsifier (c) window-collapse limit. pytest -m "not gpu and not slow" 1192 passed / 15 skipped, ruff+mypy clean. Gate (ii) NOT evidence (monitored: −0.48 under S_and) |
| 2026-08-12 | 87c6670b | presented | PENDING AUTHOR RATIFICATION | bayesian_statistics.py:dark_mass_density_per_mass | phi(M) two-segment affine swap, 5-item package in PERF_ROADMAP.md §5 + test file docstring |
| 2026-08-12 | 87c6670b | implemented | PASS | bayesian_statistics.py:dark_mass_density_per_mass | ref comments + regression pins + seam test + tripwire test |
| 2026-08-12 | 87c6670b | verified | PASS | bayesian_statistics.py:dark_mass_density_per_mass | sign/units unchanged (same phi, same normalisation); limits: pins at band edges + kink; adversarial verify CONFIRMED 1.8e-15; counterfactual smoke tolerance registered (rel 1e-8, 2D channel) |
| 2026-08-12 | dfedf19c | presented | DIRECTION APPROVED 2026-08-12 / PACKAGE PENDING AUTHOR RATIFICATION | bayesian_statistics.py:completion_mass_factor_g | Route 1 adaptive GH order, package in ROUTE1_GATE_PACKAGE.md |
| 2026-08-12 | dfedf19c | implemented | PASS | bayesian_statistics.py:completion_mass_factor_g | adaptive path + 7 tests incl. fast-order pin + gate doc |
| 2026-08-12 | dfedf19c | verified | PASS | bayesian_statistics.py:completion_mass_factor_g | xhigh adversarial CONFIRMED, bound 2.5e-37, in-support max 9.6e-16, off-support zero-vs-dust divergence registered, smoke rel 1.26e-14, 9.28x cumulative |
| 2026-08-12 | 7c58f31e | ratified | RATIFIED | bayesian_statistics.py:dark_mass_density_per_mass | phi(M) two-segment affine swap 87c6670b — author ratified 2026-08-12 ("all approved"), incl. rel-1e-8 2D tolerance class registered as a divergence |
| 2026-08-12 | 7c58f31e | ratified | RATIFIED | bayesian_statistics.py:completion_mass_factor_g | Route 1 adaptive Hermite dfedf19c — author ratified 2026-08-12 ("all approved"), incl. off-support zero-vs-dust divergence registered |
| 2026-08-17 | 44aa239e | presented | APPROVED | bayesian_statistics.py:4344,:4295 | selection fusion [P1]+[P2]: 5-item package = GATE_PRESENTATION_SELECTION_FUSION_20260817.md (verifier GO-W-AMENDMENTS 44aa239e); approval = ledger rows #117 item 1 + #118 G1 (adaptive+guard) / G2 (ratio+track) / G3 (deferral confirmed) |
| 2026-08-17 | pre-commit | implemented | PASS | bayesian_statistics.py (completion legs), arguments.py, main.py | [P1] completion_mass_factor_g_sel (new callable; g_i untouched for external callers) + [P2] S_bar_phi default-on via 'fused' cell ('auto' default resolves per normalization mode); G1 adaptive+S-var guard (_G_SEL_S_VAR_TOL); MINOR-1/2/6 folded; ref comments (MFG 2019 Eqs. 5-7) + amended [P5]-1 suite in test_selection_fusion.py; pre-change pins recorded at 4ab5da0e |
| 2026-08-17 | pre-commit | verified | PASS | bayesian_statistics.py (completion legs) | Sign/bounds: S_4D in [0,1] => 0 <= g_sel <= g_i measured TRUE; monotone suppression only. Units: S dimensionless => g_sel stays a 1/x_M density, same measure as mz_integral (gate (i) addability preserved). Limits: S==1 recovers g_i BIT-exact on both quadrature paths (test); S_bar_phi==1 recovers old B_num (pins); constant S=c => exact c-scaling (closed form, rtol 5e-15); beyond-horizon S->0 zeros classified. G1 recorded bound: adaptive-vs-pinned max rel 6.65e-16 (smooth S, production-regime rows). Byte-identity: off/1d cells reproduce pre-change pins EXACTLY; generator_marginal auto->off; 1506 passed / 15 skipped, ruff+format+mypy clean |
