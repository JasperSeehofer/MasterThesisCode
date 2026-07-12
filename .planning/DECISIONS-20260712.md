# User decisions — 2026-07-12 (bias investigation + production fix)

Recorded from the user's answers this session. Supersedes the "open decisions" in
`.planning/HANDOFF-LOCAL-NO-CLUSTER-20260710.md` §Decisions and the scoping doc §7.

| ID | Decision | User's call | Consequence |
|---|---|---|---|
| **D1** | Population-depth endgame framing (#30): depth-1.5+fallback (statistical-siren) vs truncate z≈0.5–1.0 (catalogue-driven) | **EVIDENCE-DRIVEN — do not choose now.** Let the EXP-40 posterior + information-content (width with/without deep events) measurement decide (a) vs (b). | Framing deferred to cluster evidence. Independent of the kernel fix (which corrects both regimes regardless). No local action. |
| **D2** | Merge/deployment order of the stacked branches | **ALL TOGETHER, one deployment** — #22 → #31 → #32 stacked, WITH the #27 cluster (CLU-*) fixes in the SAME cluster deployment window. | Single combined merge + cluster deploy on cluster return (per D2). |
| **D3** | Paper A: caveat vs re-derivation for the 0.745 claim + zero-host disclosure | **PAPER ON HOLD** until we are happy with the pipeline + results; THEN upgrade the paper. | No Paper A revision now. `realdata.tex` "3343 events" correction + venue caveat deferred to the post-satisfaction upgrade. |
| **D4** | The 2D residual (venue +0.025 after the D_g fix) | **DEFER** — accept as campaign-gated; no further local bias investigation until new cluster data. | 2D +0.025 parked. N-5 already confirmed no subsample/grid pathology; nothing local remains. |
| **D5** | Time allocation while cluster down | **See above** (moot) — all local tracks complete; remaining is cluster-gated + the production fix. | — |
| **PROD** | The user-gated production host-z kernel `/physics-change` | **IMPLEMENT ALL** — build the full corrected kernel (truncated-normal × volume prior + soft photo-z membership + distance-error coupling), not a partial. | ACTIVE WORK. Keep `volume_deconv` as the golden baseline; implement behind a NEW `normalization_mode`. Must pass the 6 regression gates (scoping §6). See execution plan below. |

## Production fix — execution sequence (physics hard gate)

`bayesian_statistics.py` is a physics-trigger file → the `/physics-change` protocol governs.
"Implement all" authorizes the work; the derivation + presentation gate still runs first so the
concrete formula is on record before code lands.

1. **Derive** the single concrete kernel formula: truncated-normal N₊(z; z_g, σ_z) over [0, z_max]
   × volume prior w_pop(z), with soft photo-z-marginalized membership, co-designed with the
   latent-threshold distance-error model (model-σ + p_det-inside per [L7] — NOT p_det alone).
2. **Present** (physics-change gate): old formula, new formula, reference, dimensional analysis,
   limiting cases (scoping §5/§6). Natural checkpoint — user can course-correct before code.
3. **Implement** behind a new `normalization_mode` (e.g. `volume_trunc_soft`); `volume_deconv`
   stays bit-identical golden default.
4. **Verify** the 6 binding regression gates (scoping §6): σ_z→0 → bare; deep venue reproduces
   commission-d2 (−0.002); shallow venue removes +0.030; deep-incompleteness no leak;
   noise-model coupling no floor re-open; h-independence of the prior shape. Re-run the
   pp_coverage harness on the deep + shallow venues.
5. Campaign (cluster) is the final cross-seed adjudicator (D1 evidence also lands here).

Scoping reference: `.planning/PRODUCTION-KERNEL-FIX-SCOPING-20260712.md`.
