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

