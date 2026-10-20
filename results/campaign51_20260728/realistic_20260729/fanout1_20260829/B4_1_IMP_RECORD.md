# B4.1 [IMP] part 2 — node record (stage-0 intake + stage-1 information forecast)

*launched under rows #222/#223 — charter node B4.1 [IMP] part 2*
*Append-only. Top-tier (inherit) agent, per row #224 tiering. Zero `evaluate()` calls. Nothing here is
a registered measurement; every number is a stage-1 forecast input with provenance, no band.*

## 1. Verdict of this node (one paragraph)

**Stage 0 exit reached; stage 1 forecast complete; the 4.2 read is NAMED (not merged into B1).**
The claim is *not exonerated* (17-row mechanism-grepped table, both layers; the one entry naming the
mechanism — #87's harness impostor null — is venue-scoped and was overtaken by the ledger's own O2
measurement). The remainder is **not diffuse**: on the fused+twin (production-leg) basis it is
+0.1227 ± 0.0077 of fleet displacement (80.8 % of the coded-leg drag; removing it un-rails 12/12 → 0/12),
and **87–92 % of its per-event score at truth is carried by the lowest-z quartile of events**
(z_true < 0.36), with catalogue share the strongest single correlate (r ≈ −0.77) and SNR carrying nothing
(η² 0.009). A first-order split puts ~63 % on the global mixture-weight h-slope and ~37 % on the per-event
catalogue-vs-completion slope. The "all-dark composition artefact" alternative is closed at zero
compute (composite score −0.146 at the model's own w̃_G = 0.062; the catalogue class does not
compensate). On production HEAD (`iiib`, ASSUMPTION-JOIN) removing the dark-class catalogue leg alone
moves the 1D posterior from the 0.60 rail to 0.713 ± 0.028 (covers truth) — the dark-class impostor leg
is a NECESSARY cause of the production 1D rail; sufficiency is not established (the fused completion leg
pulls +0.11 high on its own; B3's object). Mechanism is **UNDETERMINED** among kernel width (B1's
`s`-axis), the catalogue-leg mixture-weight h-slope (C9 live), and catalogue-depth skew inside the ball.
**B4.2 = "KW-Q1"**: the kernel-width discriminator on the frozen low-z quartile, riding B1.1's θ-driver
(F3 shared instrument, predictions registered in the claim card §1.3), 8.4 CPU-h (≤ 14 with the
diagnostic variant), bands KERNEL-WIDTH-OWNS |R| ≥ 0.5 ⇒ merge into B1 / INERT |R| ≤ 0.2 ⇒ derivation
path / MIXED. Merge into B1 is therefore CONDITIONAL, declared now per charter 4.3.

## 2. What was done (in order)

1. Read: `docs/RESEARCH_CYCLE.md` stages 0–1 (`:27-107`) + amendment ledger (`:496-`); the template
   `CLAIM_COMPLETION_MEMBERSHIP_20260828.md`; runbook 37 §0/§2/§5; ledger rows #137–#140, #146,
   #149–#159, #161–#165, #173, #177, #195–#197, #213, #221–#224; `CLAIM_P3_IMPOSTOR_CONVENTION_20260822.md`
   (135 lines, incl. its stage-1 append); `B4_1_IMP_DECOMPOSITION.md` (part 1, `[AGENT]`);
   `PREREGISTRATION_P3_TWIN_20260822.md` §1/§8 + fused verdict; `PREREGISTRATION_HIER_HTHETA_20260826.md`
   §1.2/§1.6/§4.1/§7.1–7.2; `PREREGISTRATION_CMEM_READS_20260828.md` + result record; the O2 scorer
   `results/prod2d_closure_20260818/decompose_impostor_leg.py` and its output JSON.
2. Exoneration check, both layers, mechanism-grepped (rule 5): `EXONERATION_REGISTER_20260827.md` read in
   full (930 lines) and ledger §2/§3/§4 (`:127-243`), greps for `kernel width | sigma_z/σ_z/photo-z |
   impostor | candidate/in-ball | cone | z-prior/w_pop | Malmquist/magnitude-limit/depth | starv`;
   17 hits tabulated verbatim with file:line and a binds/does-not-bind disposition (claim card §0.2).
3. Stage-1 free reads (A1/A12), three scripts written in this directory and run once each:
   - `b4_imp_stage1_forecast.py` → `b4_imp_stage1_forecast.json`, `b4_imp_stage1_events.csv` (2152
     events × 3 bases): O2 construction on `bsel` (off/coded), `fc` (fused/coded), `ft` (fused/twin);
     per-event impostor-leg score at truth; covariate localisation (z_true, SNR, catalogue share,
     candidate count).
   - `b4_imp_stage1_split.py` → `b4_imp_stage1_split.json`: first-order split of the impostor-leg score
     into the global mixture-weight slope and the per-event relative slope.
   - `b4_imp_stage1_production_o2.py` → `b4_imp_stage1_production_o2.json`: the O2 construction on the
     production HEAD readout (`headreadout_20260827/{iiib,joint_r1,off_iiib}`), all-legs and dark-only.
   - inline (not saved as a script; reproduced from the printed table in the claim card C4): b0i `bc`/`bt`
     per-event full-mixture score at truth for the composition composite.
4. Wrote `CLAIM_IMPOSTOR_DRAG_20260829.md` (claim card: §0.1–0.4 intake, C1–C6, §1.0–1.4 forecast and the
   named 4.2 read, NOT-claimed list, exonerated list, errors list) and this record.

## 3. Decisive numbers (A11: value · source · date)

| name | value | source | date |
|---|---|---|---|
| O2 reproduction (off/coded) | Δ_41 = +0.07918832458493737 vs record +0.07918832458493741 (dev 4e-17) | `b4_imp_stage1_forecast.json` `o2_reproduction`; `decompose_impostor_leg_output.json` | 2026-08-29 / 2026-08-21 |
| FC full / pure / Δ (registered grid) | −0.11351 / +0.03830 / **+0.15181 ± 0.01071** (SD 0.0371; 12/12; rail 12/12 → 0/12) | `b4_imp_stage1_forecast.json` `arms.fc.fleet`; banked FC −0.113508 `PREREGISTRATION_P3_TWIN_20260822.md:567` | 2026-08-29 (FC run 2026-08-23, `53b7831e`) |
| FT full / pure / Δ (registered grid) | −0.08444 / +0.03830 / **+0.12274 ± 0.00774** (SD 0.0268; 12/12; rail 12/12 → 0/12) | same, `arms.ft.fleet`; banked FT −0.084440 (`:567`) | same |
| remainder fraction on the fused basis | 0.12274 / 0.15181 = **0.808** (row #162's off-basis 0.806) | derived from the two rows above; ledger `:2321-2322` | 2026-08-29 |
| un-truncated Δ (H_GRID_FULL) | bsel +0.1498 ± 0.0144; fc +0.2499 ± 0.0113; ft +0.1865 ± 0.0073 | `b4_imp_stage1_forecast.json` `delta_FULL_*` | 2026-08-29 |
| impostor-leg score at truth, fleet mean | bsel −0.2167 ± 0.0158; fc −0.3282 ± 0.0218; ft −0.2178 ± 0.0158 (seed SEM) | `arms.*.fleet.score_imp_mean/_seed_sem` | 2026-08-29 |
| z_true q1 share of Σ s_imp | **ft 91.7 % / fc 86.2 %** (q1: z < 0.358, mean −0.798 ± 0.042 / −1.132 ± 0.048); q3+q4 < 2 % | `covariates.{ft,fc}.z_true` | 2026-08-29 |
| η² by covariate (ft / fc) | z_true 0.326 / 0.405 (active-only 0.448 / 0.539); share 0.384 / 0.525 (r −0.77 / −0.78); log n_cand 0.102 / 0.144; SNR — / 0.009 | `covariates.*` | 2026-08-29 |
| first-order split (ft) | global `c·s_β` −0.1366 ± 0.0068 (62.7 %); per-event `c·(s_L − s_B)` −0.0812 ± 0.0112 (37.3 %); s_β = −3.2891/h, s_D = −1.2373/h; mean s_L (active) −27.08 ± 1.06/h | `b4_imp_stage1_split.json` `ft.fleet` | 2026-08-29 |
| b0i catalogue-class full-mixture score at truth | bt −0.1238 ± 0.0527; bc −0.4128 ± 0.0525 (12 seeds, 1377 events); completion leg +2.292 ± 0.018; catalogue leg −2.415 ± 0.062 | claim card C4 table (inline read of `p3_b0_work/{bc,bt}_*` CSVs) | 2026-08-29 (arms run 2026-08-24) |
| model class weight | w̃_G(0.73) = 0.0620, event-independent, both venues | `w_tilde_G` column at h = 0.73 | 2026-08-29 |
| composite full-mixture score at model composition | 0.062·(−0.124) + 0.938·(−0.147) = **−0.146** (twin basis); −0.268 (coded) | derived | 2026-08-29 |
| production HEAD `iiib`: full / pure-all / pure-dark-only | mean 0.6077 (MAP 0.60, floor mass 0.446) / 0.8396 (MAP 0.86) / **0.7134 (MAP 0.70, σ 0.0277, c68 TRUE)** | `b4_imp_stage1_production_o2.json` `iiib`; HEAD readout row #213 (`d04d9dc9`) | 2026-08-29 (run 2026-08-27) |
| production HEAD `joint_r1`: full / pure-all | 0.6143 (MAP 0.60) / 0.8459 (MAP 0.86); dark-only NOT computable (no matching CRB: `seed62000` 1545 rows < 1588) | same, `joint_r1` | same |
| production `off_iiib`: full / pure-all / dark-only | 0.6032 / 0.6891 (MAP 0.67) / 0.6321 (MAP 0.63) | same, `off_iiib` | same |
| production impostor-leg score (iiib) | −0.265 ± 0.051 pooled; dark −0.193; in-cat −1.707; dark share 69 %; nearest-d_L quartile carries 93.8 % | same | same |
| joined in-catalogue fraction (consistency check of the ASSUMPTION-JOIN) | 0.04786 = 76/1588 (matches the known production value) | same | same |
| KW-Q1 cost | 4 seeds × 3 s-nodes × 2 h-nodes × (0.2843 + …) = **8.4 CPU-h**; +5.6 for the site-2.2 diagnostic | `PREREGISTRATION_HIER_HTHETA_20260826.md:584-585` anchors | 2026-08-26 |
| F5 1-D forecast at production σ_z | σ_eff/H₀ 26.1–29.8 % (σ_z 0.015–0.05, N = 400; saturation band) | `docs/SIGMA_Z_SIGMA_M_FORECAST.md:215-236` | — |

## 4. Exoneration disposition (summary; full table in the claim card §0.2)

NOT exonerated. Binding-on-remedies entries carried into every future B4 arm: `w_pop` form/tuning
(C7/G2b, [WPOP-TUNING]); hard clamp on observed z; numerator-only / unpaired insertions
([NUMERATOR-ONLY-CLEAN], [PDET-NUM-ALONE]); depth truncation as a fix; the E1 subset-conditioning trap.
The #87 harness null (*"Impostor channel … EXONERATED as residual carriers"*, register `:446-462`; ledger
§3 `:195`) is the only entry naming the mechanism; disposed as venue-scoped (ledger `:157-159`) and
overtaken by O2 (row #149, ratified row #153). The cone/in-ball object is B2's (C-STRUCTURAL-ONLY,
rows #219/#220; A1 upgrade granted row #221) — B4 runs no in-ball split.

## 5. Information forecast — the per-read table (from the claim card §1.2)

(i) impostor-weight switch — NOT decisive (remedy class; bounded [0, +0.123]; D̃ sub-convention is an
author [RULE], row #167). (ii) **kernel-width switch (HIER s) — DECISIVE for the merge question; named
as B4.2 KW-Q1**, 8.4 CPU-h, riding B1.1's driver. (iii) true-host-only — for the dark class it IS O2
(already measured here on three bases); for the catalogue class an oracle arm (11.8 CPU-h + new
instrument) is not decisive for this claim — deferred. (iv) in-ball split — B2's object; not run.
(v) ensemble-level free reads (composite score −0.146; production dark-only 0.713) — recommended as
KW-Q1's registered zero-compute secondaries, runner ≠ this builder.

## 6. Hand-off to the orchestrator / B4.2

- **Dependencies for KW-Q1:** B1.1's θ-driver (S0-A) must exist and pass T-ID/PARITY at s = 1 on a
  HEAD-basis FT re-evaluation (θ engages the smeared kernel path, `bayesian_statistics.py:2799-2806`);
  GATE ENG must be scored on the CATALOGUE leg. Schedule KW-Q1 after B1.1 reports; the prediction
  (bands, statistic, frozen q1 set) is registered in the claim card §1.3 BEFORE B1.1's result (F3).
- **Rule 2:** KW-Q1's runner ≠ this agent ≠ B1.1's driver author. The zero-compute secondaries
  (validated-join production dark-only read; HEAD-basis re-anchor of C1) likewise need a fresh runner.
- **Path map for B4.3:** OWNS ⇒ merge into B1 with the q1 localisation + absorption prediction; INERT ⇒
  derivation proposal for the catalogue-leg mixture weight's h-slope (`s_β = −3.29/h`; C9 / Gate C
  item 1 live object) + the per-candidate instrumented run (part 1 §7, ≈ 3.4 CPU-h, one h-value) to
  measure the q1 impostor z-offsets directly; MIXED ⇒ both, reported.
- **Cross-branch notes:** B3 [POP] — the fused completion leg is +0.038 high on the mirror and +0.11 high
  on production (pure-all arm at the 0.86 edge); the 1D posterior is a balance of the two legs.
  B2 [CMEM] — part 1's 18.4 % not-recovered (1D) class is presumably B2's 16.8 % outside-cone class;
  left to B2. B7 [2D-TWIN] — nothing here touches the with-BH leg.

## 7. Caveats (complete)

1. Every `[LOCAL]` number is a forecast input with no registered band; none may be quoted as a verdict.
2. FC/FT (and `bsel`) predate the Σ^φ adoption (`e35ea018`, row #179): the "production basis" numbers
   are the 2026-08-23 fused/twin basis; the HEAD leg differs by a global 1/r_φ(h) factor (1.1287 at
   0.73; row #171 measured the slot's venue headline effect at −0.0043). Re-anchor before band-setting.
3. The production dark/in-catalogue split is an ASSUMPTION-JOIN on CSV row order (event_idx == CRB row);
   the joined in-catalogue fraction reproduces 76/1588 exactly, a consistency check only.
4. The composition composite (C4) combines two different fleets (b0i catalogue class with the Σ^φ slot;
   FT dark class without) with the model's global class prior w̃_G; it closes the composition-artefact
   alternative by magnitude (−0.146 vs a needed ≈ 0), not as a registered measurement.
5. All grid statistics are boundary-censored (map_h at the 0.60 floor 12/12 on every full arm; the
   production pure-all arm at the 0.86 upper edge); un-truncated values are quoted alongside.
6. `bsel` has no logs on disk, so its candidate counts are inherited from FT's realisations (same seeds;
   `L_cat_no_bh` bit-identical to 1.6e-13 at h = 0.73 — verified).
7. Per-candidate covariates (impostor z, σ_z, mass, c_k) remain structurally absent (part 1); the
   localisation is by EVENT covariates only.
8. The s-sensitivity of the catalogue leg is genuinely unknown — the KW-Q1 bands are materiality
   fractions of the q1 score, not derived from a predicted effect size (disclosed per A15's spirit).
9. The Stage-L R0 sweep was satisfied by reference to the existing LITERATURE_WARNINGS rows; no new
   time-boxed search was run.
10. This node did not open, edit or run anything in a physics-trigger file; no git operations.

## 8. Compute

Zero `evaluate()`. Four single-core pandas passes over banked CSVs (36 mirror arms × 46–41 h-nodes, three
production venues × 41 h-nodes, 24 b0i arms at 3 h-nodes): ≈ 6 min wall total ⇒ **≈ 0.1 CPU-h**,
local. Appended to `COMPUTE_LEDGER.md`.

## 9. Files

- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/CLAIM_IMPOSTOR_DRAG_20260829.md`
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/B4_1_IMP_RECORD.md` (this file)
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/b4_imp_stage1_forecast.py` / `.json` / `b4_imp_stage1_events.csv`
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/b4_imp_stage1_split.py` / `.json`
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/b4_imp_stage1_production_o2.py` / `.json`
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/COMPUTE_LEDGER.md` (row appended)
