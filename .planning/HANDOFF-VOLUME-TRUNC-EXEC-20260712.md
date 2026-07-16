# Next-session kickoff — execute Part 1 `volume_trunc` (production host-z kernel fix)

Paste the block below as the first message of the next session.

---

Execute **Part 1 of the production host-z kernel fix** on branch
`physics/zero-host-completion-fallback` (all committed + pushed, tip after `bb9edf2`).
The `/physics-change` presentation gate for Part 1's formula is **ALREADY PASSED** (user
approved 2026-07-12: formula + staged approach + mode name `volume_trunc` + `volume_deconv`
stays golden). **Do NOT re-derive or re-present the formula** — implement it.

**Read first (current state, ~5 min):**
- `.planning/PRODUCTION-KERNEL-FIX-SCOPING-20260712.md` §1 (old formula), §2 Part 1 (new
  formula), §5–§6 (dimensional analysis + the 6 regression gates), and **§7b (the code-level
  implementation spec — the load-bearing section)**.
- `.planning/DECISIONS-20260712.md` (user calls D1–D5 + PROD=implement-all, staged).
- The existing golden guard: `master_thesis_code_test/bayesian_inference/test_bayesian_statistics_host_z_kernel.py`
  (already pins `volume_deconv` — these MUST stay unchanged).

**What Part 1 is (and is NOT):** `volume_trunc` = z≥0-floor truncation + **unified numerator
support** (integrate `N_g` over the per-host galaxy window `[z_lo, z_hi]`, not today's shared
event-level GW window, with the same `Z_g`). It is **SHALLOW-only** and a **no-op on the deep
venue by construction** (z_lo = z_g−4σ > 0 there) — it does NOT fix the deep L-7 leak (that is a
separate `z_support`-edge truncation = a later part). The substantive change is the numerator
window, NOT the z-floor (production already floors Z_g/D_g at 1e-6≈0).

**Implement (keep `volume_deconv` BYTE-IDENTICAL — branch on the mode):**
1. Add `"volume_trunc"` to the valid-modes set at `bayesian_statistics.py:999`.
2. Scalar `single_host_likelihood` (`:2170–2510`): add a `_use_volume_trunc` gate;
   `den_lo = max(z_g − 4σ_eff, 0.0)`; **integrate the numerator over `[z_lo, z_hi]`** (the
   per-host galaxy window) with the shared `Z_g`; optional `z_hi = min(z_g+4σ_eff, z_max)` (defer
   if z_max isn't a worker global — it rarely binds in the shallow regime).
3. Batched `single_host_likelihood_batch` (`:2512–2810`): the numerator window becomes per-host
   `[den_lo, den_hi]`, so `y_num`/`d_L_num`/`luminosity_distance_fraction`/`gw_3d` become `(n,50)`
   (the shared-node optimization is lost for the numerator; the denominator path is already
   per-host). Keep the `volume_deconv` branch unchanged.
4. Keep `single_host_likelihood ≡ single_host_likelihood_batch` (`test_kernel_batch_equivalence`).

**Test (physics-change regression requirement):**
- Existing `volume_deconv` pins UNCHANGED (golden guard — if they move, you broke the default).
- Add `volume_trunc` pins + a σ_z→0 limiting-case test (→ spec-z limit) in
  `test_bayesian_statistics_host_z_kernel.py`; reuse the h-independence check.
- `uv run pytest -m "not gpu and not slow"` green before the empirical run.

**DECISIVE EMPIRICAL GATE (must run — genuine uncertainty, cannot derive):** seed600 494-event
A/B, `volume_trunc` vs `volume_deconv`. Reuse the N-5 harness: driver `scripts/eddington_m_impact.py`
(pass `normalization_mode` through — or a direct `--evaluate`), composed data_dir = crux_ws CRBs
(`~/data-backups/seed600_local_derail_20260702/crux_ws/simulations/{prepared_,}cramer_rao_bounds.csv`)
+ the REAL pool `~/data-backups/seed600_local_derail_20260702/simulations/injections` (the crux_ws
`injections` symlink is dead → /tmp). **`allow_shallow_pool=True` needed in BOTH `evaluate()` AND
`combine_posteriors()`.** ~10 min. **Success = shallow 1D mean 0.745 → toward 0.73, no pathology.**
If it does NOT move, the numerator-window is NOT the +0.013 lever — that is a real finding; report
it (don't force it). Deep-venue no-regression holds by construction locally; the campaign is the
cross-seed adjudicator.

**Post-checklist:** `[PHYSICS]` commit prefix; reference comment above changed lines
(Gray 2020 A.10 + G2b §1.4); sign/dimensional consistency; then `/pre-commit-docs`.

**Model discipline:** default Sonnet for implementation; do NOT launch a Workflow (single-threaded);
lean. This is a delicate hot-path change — verify `volume_deconv` pins stay green at every step.

**After Part 1 verifies:** Parts 2 (deep `z_support`-edge membership truncation, completion-coupled)
and 3 (soft photo-z membership + [L7] distance-error coupling) are the sub-dominant follow-ups —
each needs its own derivation + `/physics-change` presentation (user-gated) before code.

**Everything else:** all other LOCAL bias work is exhausted (N-4 closed, N-5 done); D1–D5 recorded
(`.planning/DECISIONS-20260712.md`); cluster items (D2 combined deploy, EXP-40, campaign) on cluster
return per `.planning/HANDOFF-LOCAL-NO-CLUSTER-20260710.md` L-F.

---
