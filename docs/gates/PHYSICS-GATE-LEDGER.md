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
| 2026-08-04 | pre-commit | presented | APPROVED | bayesian_statistics.py (selection stack + mixture) | Fix B path-A joint C9+C8 (FIXB_PATHA_PACKAGE.md §3): S̄_φ replaces fitted S_3D in all three slots, D^φ, g-inside, w̃_G=α_G^φ/D̃^φ. Author decisions: D1=both (S_and now, retire stale p0 bounds next campaign; p0-window onto 2D-bias suspect list), D2=delivered-convention pins primary with MANDATORY promotion to truth once truth Σ4D(h) measured at 41h, D3=point form. Gate (ii) demoted to monitored consistency number; ship-on-correctness |
| 2026-08-04 | pre-commit | implemented | PASS | bayesian_statistics.py (selection stack + mixture), dark_siren_injection.py | S̄_φ = ∫φ S_4D dlog10M on the production pooled-2D with-BH object (pdet_wbh_z_resolved=False); new φ-convention tables β_G^φ/β_Ḡ^φ/Σᶲ/D^φ consumed by absolute_marginal ONLY (legacy tables + generator_marginal byte-identical, gate iii-a); mixture w̃_G=α_G^φ/D̃^φ, α_G^φ=β_G^φ·r_Malm; (N8) B_num_wbh with g_i inside the quadrature, 1D B_num unmultiplied; φ exported once from dark_siren_injection (never re-typed); point-mass evaluation (D3); instrumentation (w_G RENAMED to w_G_legacy, 7 s.f. diagnostics, T9 Σ4D band shares, g_i support-exit); monitored gate (ii) under S_and (ρ=0.7305) |
| 2026-08-04 | pre-commit | verified | PASS | bayesian_statistics.py (selection stack + mixture) | Dimensions: S̄_φ dimensionless, β^φ/D^φ in [p_pop dz], r_Malm/r_φ dimensionless, g_i the ONLY x_M density (2D completion only) — no leg gains/loses a mass measure. Signs: S̄_φ ∈ [0,1] monotone in d_L; α_G^φ, D̃^φ > 0; w̃_G ∈ (0,1). Anchors reproduced on the pool/catalogues of record: β_G^φ=1.533228e8, β_Ḡ^φ=8.884038e8, D^φ=1.041727e9, Σᶲ_del=9.5623703e8, r_Malm_del=0.4415122, w̃_G=0.0708023 — all ≤4e-8 rel. Gates: (i) 2D exactly homogeneous in x_M at 1e-14 and h-independent (dMAP/dlnC=0); (iii-a) #51 1D digest + generator_marginal unmoved; (iv) 1D bitwise invariant; T8 r_φ≡1 exact; L4 s=0 tilt vanishes; L5 σ_Mz→0 finite=point form; falsifier (c) window-collapse limit. pytest -m "not gpu and not slow" 1192 passed / 15 skipped, ruff+mypy clean. Gate (ii) NOT evidence (monitored: −0.48 under S_and) |
