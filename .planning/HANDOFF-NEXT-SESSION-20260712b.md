# Next-session kickoff prompt (2026-07-12b → next)

Paste the block below as the first message of the next session.

---

Continue the H₀ bias work on branch `physics/zero-host-completion-fallback` (all committed +
pushed?; tip `7a3f318`). Start by reading `.planning/STATE.md` (Quick Tasks table, top rows) and
`.planning/BIAS-INVESTIGATION-20260710.md` ledger items **[L7]**–**[L9]**. **The entire LOCAL,
harness-only bias investigation is now EXHAUSTED** — what remains is cluster-gated, user-decision-
gated, or the user-gated production `/physics-change`.

**What closed this session (2026-07-12):**
- **N-4 shallow attribution CLOSED (`d966156`)** — seed600's low-z hosts are 89.7% photometric,
  σ_z ≈ 0.0344, **σ_z/z ≈ 0.65 (O(1))** at z_med 0.046; the likelihood kernel width IS this
  catalogue σ_z and the z≥0 clamp is active for z_g<4σ_z (`bayesian_statistics.py:2243`,`:2234-2239`).
  ⇒ the shallow +0.0132 IS the σ_z/z-at-low-z truncated-volume-kernel Eddington effect.
- **N-5 2D subsample check DONE (`7a3f318`)** — the 494-event 2D subsample is well-behaved under
  current code (edge_mass 0.216→0.003, mean 0.790→0.768); +0.0135 above full-venue 0.7546 is a
  subsample-selection offset, not a defect. Venue +0.025 2D residual stays campaign-gated (D4).
  Bonus: post-D_g-fix Eddington-in-M Δ2D = −0.0022 (was −0.020) ⇒ `bayesian_statistics.py:2400-2401`
  comment/value STALE (flagged, not edited).
- **Production-fix SCOPING done (`45398f4`, `.planning/PRODUCTION-KERNEL-FIX-SCOPING-20260712.md`)** —
  the full `/physics-change` presentation gate for the z≥0-truncation-aware / photo-z-marginalized
  volume host-z kernel that BOTH regimes ([L7] deep leak, [L8] shallow σ_z/z) converge on. USER-GATED.

**The bias story is mechanistically CLOSED at the harness level.** Deep = membership kernel leak
(exact truncation removes) + noise-model floor (≤σ_boot). Shallow = σ_z/z Eddington, attributed.

**Model / cost discipline (all session):** default to Sonnet; do NOT launch a Workflow (remaining
probes are single-threaded); pre-register CALIBRATED/BIASED predictions in a RUNBOOK before any run;
assert on continuous tilt diagnostics or a fine h-grid, not the coarse MAP grid.

**USER-GATED — do NOT start autonomously (need explicit approval):**
- **The production kernel `/physics-change`** — read `.planning/PRODUCTION-KERNEL-FIX-SCOPING-20260712.md`
  first; then `/gpd:derive-equation` (truncated-normal × volume prior + soft photo-z membership,
  co-designed with the [L7] distance-error model — do NOT add p_det-inside alone) → `/physics-change`
  presentation of the ONE chosen formula → user approval → implement behind a NEW `normalization_mode`
  (keep `volume_deconv` bit-identical golden) → re-verify the 6 binding regression gates. Trigger file
  `bayesian_inference/bayesian_statistics.py`.
- **D1** (fix in production vs Paper-B robustness bound), **D2** (PR merge order #22→#31→#32),
  **D3** (Paper A venue caveat), **D4** (2D +0.025 → campaign), **D5** (time allocation).

**Optional cheap doc follow-up (low priority):** refresh the stale `bayesian_statistics.py:2400-2401`
comment (Eddington-in-M "−0.020" → post-D_g-fix "−0.0022"). It is a physics-trigger file but the change
is comment-only (no computed value); still, surface it before editing.

**Cluster (verify return): runbook unchanged** (`.planning/HANDOFF-LOCAL-NO-CLUSTER-20260710.md` L-F):
security hygiene → preflight READY → rsync depth15 pool → h=0.705 re-run → deploy merged branch per D2
→ EXP-40 (watch: interior-but-biased-HIGH; the post-#29 mixture carries BOTH the leak and the floor
same-signed HIGH per [L7]) → only then seeds 2000–6000 = the §4b definitive verdict.

---
