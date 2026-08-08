# Pre-registration — calibration-gate extension: P–P/coverage leg + multi-candidate host balls + σ–d_L joint texture

Registered 2026-08-08, **BEFORE** the build and **BEFORE** any run. Research
Cycle stage 2 (`docs/RESEARCH_CYCLE.md`) registration **for the stage-4
calibration gate itself** — the instrument this file registers is the missing
input of `docs/RESEARCH_CYCLE.md` §Stage 4 leg 1 ("SBC / P–P coverage of the
FULL two-channel estimator on truth-known synthetic universes at the
production venue"), which the stage doc says "do[es] not exist yet" and which
building **is** this mission.

Mandate of record, `results/campaign51_20260728/RUNBOOK_NEXT_SESSION_8.md`
§1 item 7 (verbatim):

> **Calibration-gate next step:** extend `closed_loop_gfrac.py` toward the A3
> criteria still open (realistic host-observation model / multi-candidate
> balls; σ–d_L joint texture; the P–P leg of §9's CONFIRM that was not
> evaluated) — then the stage-4 gate can adjudicate keep-digging vs
> report-bound for the 1D rail itself.

Parents: `results/closed_loop_gfrac_20260805/PREREGISTRATION.md` (the
registered closed-loop instrument, verdict MIXED, code `77b524af`);
`docs/RESEARCH_CYCLE.md` §Stage 4 amendment A3; RUNBOOK-8 §1 item 8 (paper
#47 ON HOLD until the P–P leg exists). Author value ruling of record: the
gate exists for **correctness + insight, not bias-removal**; a REPORT-BOUND
verdict (the 1D rail is a measured property of the estimator class under
photo-z starvation) is a fully legitimate outcome.

**REGISTERED — committed before the build. Append-only discipline is in force
from this commit.** Every band below was fixed at this commit and may not be
adjusted after any readout. The new module's code commit is appended to §11
when it exists (the closed-loop prereg's §7 pattern); nothing above §11 may
be edited after this commit.

---

## 0. Binding constraints of record

- `master_thesis_code/validation/closed_loop_gfrac.py` is **NOT modified**
  (it is a registered instrument with a logged verdict; its prereg's
  append-only discipline covers its code identity `77b524af`).
- `master_thesis_code/validation/pp_coverage.py` is **NOT modified** (its
  production-independence is its stated scientific value; the closed-loop
  prereg §2 records this explicitly).
- No production physics file is touched — any estimator fix that this gate
  motivates routes through `/physics-change` (5-item package → author
  approval → ledger rows), never through this registration.
- New code lives in exactly two new files:
  `master_thesis_code/validation/calibration_gate.py` (the extension module,
  a `pp_coverage.py`-adjacent NEW file) and
  `master_thesis_code_test/validation/test_calibration_gate.py`.
  All outputs, the readout script, and this registration live under
  `results/calibration_gate_20260808/`.
- Local CPU only (the closed-loop instrument's venue); no cluster jobs.
- **No production posterior is produced.** Every posterior emitted is a
  synthetic-universe diagnostic, quotable only against its own truth.

---

## 1. Questions of record

**Q1 (gate construction, A3 completion).** Does a genuinely 2-channel
(A3-i: `g` recomputed per h), production-N (A3-ii: N_det = 1500),
multi-candidate (A3-iii: host balls) closed-loop harness with a P–P/HPD rank
statistic — the harness Stage 4 leg 1 requires and no existing asset
provides — show the full estimator **calibrated** on truth-known synthetic
universes?

**Q2 (the 1D-rail adjudication, the mission's object).** In a truth-known
loop with a realistic host-observation model (multi-candidate balls, photo-z
scatter at the GLADE value σ_z = 0.035), does the 1D channel **reproduce the
rail signature** (production: h = 0.600 railed low, both venues; closed-loop
info-starved configuration: railed low 200/200) — and does the rail
**disappear** when the starvation is removed (σ_z → 0; anchor: the
`--numerator-pdet on` diagnostic un-railed 1D 200/200 → 0/50)? Reproduction
under starvation + disappearance without it supports **REPORT-BOUND**;
a calibrated, un-railed 1D channel at σ_z = 0.035 in-loop means the
production rail has an additional owner and supports **KEEP-DIGGING**.

**Q3 (declared-deviation closure).** Does restoring the σ–d_L joint texture
(production `corr(ln σ_dL, ln d_L) = 0.82`; the closed-loop prereg §3
declared its independent-draw simplification) move the 2D displacement
Δ2 = +0.0113 ± 0.0043 of the registered run?

## 2. Why this instrument is necessary (concrete provenance)

- Stage 4 leg 1 requires P–P/HPD coverage of the full 2-channel estimator;
  the §9 second CONFIRM clause ("and the P–P curve is inside the 90 % band")
  was explicitly **NOT EVALUATED** by the registered closed-loop run — no
  rank statistic is computed anywhere in `closed_loop_gfrac.py` (its own
  appendix records this as a declared deviation).
- A3 status: (i) and (ii) are satisfied by `closed_loop_gfrac.py`;
  **(iii) is satisfied nowhere in 2-channel form** — `pp_coverage.py` has
  genuine multi-candidate/impostor-ball machinery but is 1D-only ("No mass
  dimension, no BH-mass channel"), and `closed_loop_gfrac.py` is single-host
  by design ("a one-candidate-per-event harness structurally cannot exercise
  the impostor-ball mechanism, and no claim about that mechanism may be
  drawn from this run").
- A3 verbatim (`docs/RESEARCH_CYCLE.md` §Stage 4): "the extension is not
  accepted, and its coverage verdict does not count, unless all three hold:
  (i) genuinely 2-channel, with the completion-leg mass factor g recomputed
  per h — never frozen across the h grid, never elided; (ii) run at
  production N — the mechanism is N-coherent (per-event sub-threshold;
  event-summed ḡ(h) Δln ≈ 0.048 grid-wide) and is invisible at small N;
  (iii) multi-candidate host balls — a one-candidate-per-event harness
  structurally cannot exercise the mechanism (cf. `BIAS_HISTORY_LEDGER.md`
  §1 row 86)."
- The production 1D rail (h = 0.600, both venues) has standing account
  photo-z information starvation ([[h0-railing-rootcause-photoz]]); the
  closed-loop instrument's 1D rail (200/200) is **structurally uninformative
  about it** (f = 0 means no host redshift of any kind reaches the 1D
  channel — its own prereg says so). Only a harness with noisy,
  confusable host redshifts can put the starvation account on trial
  in a truth-known loop. That harness does not exist. This file registers it.
- Paper #47 is ON HOLD on exactly this gap (RUNBOOK-8 §1 item 8: "not yet
  the trusted-run gatekeeper RUNBOOK-7 §4 requires (P–P leg missing)").

## 3. THE INSTRUMENT

**Architecture: a thin extension module that reuses both existing
instruments without modifying either.** New module
`master_thesis_code/validation/calibration_gate.py`:

| capability | source | how used |
|---|---|---|
| universe generator (production φ with `kappa_cap` kink, `w_pop`, `S_4D` Bernoulli selection, CRB-bootstrap noise), estimator quadrature (per-h `g_i` verbatim, shared `α(h)`), canonical 41-h grid, worker-pool sweep | `closed_loop_gfrac.py` **imported as a library** (`build_context`, `ClosedLoopConfig`, `draw_universe`, `log_channel_posteriors`, `posterior_readout`, `_g_at_nodes`, `_w_pop`, `CANONICAL_H_GRID`) | called, never re-implemented; A3(i)+(ii) inherited from the registered instrument |
| HPD credible-region test | `pp_coverage._hpd_contains` **ported** (≈16 dependency-free lines) as `calibration_gate.hpd_contains` | the port is certified by a unit test asserting exact boolean agreement with `pp_coverage._hpd_contains` on 1000 random synthetic posteriors (import of `pp_coverage` in the TEST only — the runtime module stays independent of it) |
| impostor-ball design pattern | `pp_coverage.SyntheticCatalogue` / `_build_catalogue` (design borrowed, code not imported — it is 1D-only and production-independent) | the ball generative model of §4.2 |
| σ–d_L joint texture | new function `load_sigma_triples_dl_binned` in the new module | §4.3 |

**What the instrument deliberately does NOT do** (stated here so the verdict
is never over-read):

- It does not use GLADE+, real n(z), the completeness map, or the sky
  dimension: the host ball is a **redshift-window Poisson caricature** of the
  production localisation cone (§4.2). It adjudicates the **estimator-class
  mechanism** (multi-candidate photo-z confusion in a truth-known loop), not
  the exact production `L_cat` object.
- Ball members supply **redshift information only**; the 2D channel keeps the
  completion-leg mass structure (`g_i` per h). Production's in-catalogue
  host-mass kernel (R&V15 stellar→BH, `L_cat_with_bh`) is **not** modeled.
- The host-z kernel is production-style **bare** (point Gaussian
  `N(z; z_obs, σ_z)`); the `volume_deconv` kernel of the production config of
  record is an **optional arm** (cell O1, §5) — if it is not built, the
  kernel-form sensitivity is **NOT-EVALUABLE** and is flagged as such (§9).
- The catalogue-leg completeness mixture (β_G/β_Ḡ, `f_k`) is not modeled;
  `f_incl = 1` (true host always in the ball). Host incompleteness is out of
  scope v1.
- It does not evaluate production-side "no unmodeled selection" (the stage-5
  third condition) — in-loop that condition holds **by construction**
  (generator and estimator share `S_4D`); the production-side condition is
  carried by Stage 4 leg 2's standing result plus the open f_k–pool-coupling
  intake thread (§9).

## 4. Generative model and estimator

**Truth:** flat ΛCDM, fiducial `OMEGA_M`, `h_true` per cell (§5).
Per universe (one seed), steps 1–4 are **exactly** the registered closed-loop
generator (`draw_universe`, called as a library): `z ~ w_pop(z; h_true)`,
`M ~ φ(M)` (kink included), Bernoulli detection by `S_4D(d_L, M_z)` until
`N_det = 1500`, fractional 2×2 `(d_L, M_z)` Gaussian noise with
CRB-bootstrapped `(σ_dL/d_L, σ_Mz/M_z, ρ)` triples.

### 4.1 P–P/coverage readout (new, per seed, per channel)

From each channel's 41-point unnormalised `ln P(h)` (trapezoid-normalised on
the grid):

- **PIT** `q = ∫_{0.600}^{h_true} P(h) dh` — under calibration
  `q ~ Uniform(0,1)` across seeds.
- **HPD containment** booleans at 50/68/90 % via the certified
  `hpd_contains` port.
- **posterior sd** from grid moments (for DS-5).
- **edge mass** `E` = posterior mass in the first plus last grid interval
  (`[0.600, 0.610] ∪ [0.850, 0.860]`), for the edge-contamination guard (§8).

### 4.2 Multi-candidate host balls (A3-iii; ball cells only)

Generative side, per event i (after step 4 above):

1. The event's localisation window at the truth,
   `W_i = [z(d_L^obs(1−4σ_dL); h_true), z(d_L^obs(1+4σ_dL); h_true)]`
   (the same ±4σ convention the estimator uses).
2. Impostor count `n_i ~ Poisson(λ_ball)`, `λ_ball = 4` (recon budget; the
   `pp_coverage` catalogue-mode campaigns' K ≈ 3–5 range). Impostor redshifts
   i.i.d. `z ~ w_pop(z; h_true)` restricted to `W_i`. *Justification:* for a
   Poisson galaxy field with intensity ∝ `w_pop`, conditioning on the host at
   `z_true` leaves the remaining field in `W_i` Poisson with unchanged
   intensity (Slivnyak–Mecke), so i.i.d. `w_pop|W_i` draws with a Poisson
   count are the exact field statement — the ball is a cut of the same field
   the host lives in, not noise injected around the host.
3. The ball is the true host plus the impostors, **order shuffled** (the
   estimator never learns which member is the host); every member gets an
   observed redshift `z_obs,k = z_k + σ_z ε_k`, `ε_k ~ N(0,1)`, with flat
   σ_z per cell (the `pp_coverage` commission convention; the (1+z) scaling
   is a declared simplification). `f_incl = 1.0` (host always present).

Estimator side (mirrors the production catalogue-leg **structure**: bare
kernel × distance likelihood, equal candidate prior, no selection factor in
the numerator — the shipped `numerator_pdet = off` convention), for each h on
the canonical grid:

```
L_i^{1D}(h) = (1/K_i) Σ_k ∫ dz  N(z; z_obs,k, σ_z) · N(d_L(z;h)/d_L^obs_i; 1, σ_dL,i)
L_i^{2D}(h) = (1/K_i) Σ_k ∫ dz  N(z; z_obs,k, σ_z) · N(d_L(z;h)/d_L^obs_i; 1, σ_dL,i) · g_i(z;h)
ln P(h)     = Σ_i ln L_i(h) − N_det · ln α(h)
```

with `g_i` the production `completion_mass_factor_g` recomputed at every h
(A3-i preserved), `α(h)` the shared production
`precompute_phi_marginal_survival` normalisation (unchanged), and the
per-candidate quadrature 50-node Gauss–Legendre on
`[max(z_lo(h), z_obs,k − 5σ_z), min(z_hi(h), z_obs,k + 5σ_z)]` where
`[z_lo, z_hi]` is the production ±4σ window capped at `z_max(h)`. At
σ_z = 0 the kernel is a point evaluation at `z_obs,k` (zero contribution if
outside the window). Registered honestly: the generative host is
detected-weighted (`w_pop · S̄`-drawn) while the bare kernel carries no
`w_pop` and no selection factor — that mismatch **is the production kernel
form**, and measuring its in-loop calibration is the point, not a bug.

### 4.3 σ–d_L joint texture (all new cells)

`load_sigma_triples_dl_binned`: the production CRB CSV
(`results/run_20260804_postfix/iiib/diagnostics/prepared_cramer_rao_bounds.csv`,
1590 rows, `luminosity_distance` column present — verified at registration)
is split into **deciles of d_L**; each synthetic event's triple is drawn
uniformly from the rows of the decile matching the event's own true `d_L`
empirical quantile (nearest bin outside the CSV's range). This is
rank-matching: it reproduces the monotone `(σ, d_L)` trend, not the exact
copula — declared. Config knob `sigma_texture ∈ {independent, dl_binned}`;
`independent` is bit-compatible with the registered closed-loop behaviour.
Validity check V4 (§10) requires the detected synthetic set to reproduce
`corr(ln σ_dL, ln d_L) = 0.82 ± 0.10`.

## 5. Cell matrix, seed plan, N floor, runtime budget

**N floor (binomial usefulness at 0.90/0.68).** To resolve a coverage defect
of 0.05 at the 90 % level at ≳2σ needs
`σ_bin = sqrt(0.9·0.1/N) ≤ 0.025 ⇒ N ≥ 144`; to keep the 2σ band at
±≤0.03 needs `N ≥ 400`. **Registered: 400 seeds per truth per cell**
(floor: 300, the pre-declared runtime fallback — bands for both are locked
in §7; no other N may be substituted).

**Truths.** `T = {0.690, 0.730, 0.770}` — all on the canonical grid,
symmetric, and 0.09 from each grid edge (the registered 2D MAP spread is
sd = 0.0608; more distal truths would flood the §8 edge guard, which is why
`pp_coverage`'s default `[0.62, 0.72, 0.84]` near-edge convention is NOT
copied for coverage cells — near-edge behaviour is instead carried by the
rail statistic DS-4/DS-6, which the edge guard exempts).

| cell | config | truths × seeds | purpose |
|---|---|---|---|
| **R0** | retro-read of the committed registered run (`closed_loop_results.json` `per_seed[].ln_post_{1d,2d}`, 200 seeds, single-host, `independent` texture, h_true = 0.73) | 0.73 × 200 (existing, zero compute) | free anchor + readout-layer certification on committed data; **anchor-only, carries no gate weight** (its parent verdict is MIXED, Δ2 = +0.0113) |
| **A** | single-host, f = 0, `dl_binned` texture | T × 400 | **2D decision cell** for DS-1/2/3 (A3 i+ii + texture); 1D expected starved-railed (anchor 200/200) — reported, exempt from gate reads; A(0.73) vs R0 is the Q3 texture contrast |
| **B0** | ball (λ_ball = 4, f_incl = 1, bare kernel), **σ_z = 0**, `dl_binned` | 0.73 × 400 | unstarved ball venue: impostor confusion at perfect per-candidate z; DS-6 low-anchor cell |
| **B1** | ball, **σ_z = 0.010** | 0.73 × 400 | dose-response midpoint |
| **B2** | ball, **σ_z = 0.035** (GLADE photo-z of record) | T × 400 | **decision cell for Q2** (the 1D-rail adjudication) and the 2-channel+A3-iii P–P read |
| **O1** *(optional)* | as B2(0.73) with `volume`-style kernel | 0.73 × 400 | kernel-form sensitivity; run only if built; pre-banded identically to B2 — registered now so it can never be post hoc |
| **V1** | ball code path, λ_ball = 0, σ_z = 0 | 0.73 × 50 | plumbing control: must reproduce the `--f-cat 1.0` precedent (MAP = 0.730 exactly, both channels, every seed) through the NEW code path |

**Seed plan** (disjoint blocks, base **20260808**): A: +0…+399 / +1000…+1399
/ +2000…+2399 (truths 0.69/0.73/0.77); B0: +3000…+3399; B1: +4000…+4399;
B2: +5000…+5399 / +6000…+6399 / +7000…+7399; O1: +8000…+8399;
V1: +9000…+9049. Fixed per-cell; a seed appears in exactly one cell.

**Runtime budget** (from measured baselines: 27.3 s CPU-equivalent/seed
single-host at N_det = 1500 × 41 h; ball multiplier budgeted ×3–4 per recon):
A ≈ 33 k CPU-s; B0+B1 ≈ 66–87 k; B2 ≈ 98–131 k; V1 negligible; total
≈ **4.0–5.2 h wall on 14 local workers** (O1 +0.6–0.9 h) — inside the
8–10 h overnight window. **Smoke first** (10 seeds/cell, the closed-loop
`--smoke` pattern); abort criteria in §10.

**Estimator config mirrored from production** (unchanged from the registered
closed-loop instrument): canonical 41-point h grid `0.600…0.860`;
`N_det = 1500`; `numerator_pdet = off` (the shipped-estimator convention);
`snr_threshold = 20`; 50-node Gauss–Legendre, 64-node Gauss–Hermite;
injection pool `mix200k_20260728` defining `S_4D`; CRB CSV as §4.3.

## 6. Per-seed outputs (fixed before the build)

The `run_seed`-style JSON record (seed, n_events, n_proposed, z_median,
M_source_median, frac_below_kink, both channels' grid/refined MAP, posterior
mean, rail flags, `sum_dlog_gfrac_dh`, full 41-point `ln_post_1d/2d`) **plus**
new fields: `pit_1d`, `pit_2d`; `hpd50_1d … hpd90_2d` (six booleans);
`post_sd_1d`, `post_sd_2d`; `edge_mass_1d`, `edge_mass_2d`; ball statistics
(`K_mean`, `n_impostors_total`, `sigma_z`, `f_incl`); `sigma_texture`;
and the cell id. Aggregates per cell: everything in §7, plus the closed-loop
quantile battery.

## 7. DECISION STATISTICS for the stage-4 gate (exact, bands locked blind)

All bands below are **analytic nulls or quotes of committed numbers** — no
number is tuned on data this instrument will produce. Both channels are
always reported together. "Decision cells" = A (2D only) and B2 (both
channels); B0/B1 are dose/control cells; R0 is anchor-only.

**DS-1 — HPD coverage.** `C_β` = fraction of seeds with h_true inside the
β-HPD region, β ∈ {0.50, 0.68, 0.90}. Binomial null `σ = sqrt(β(1−β)/N)`:

| N | β=0.50 (1σ) | β=0.68 (1σ) | β=0.90 (1σ) | 2σ bands (0.50 / 0.68 / 0.90) |
|---|---|---|---|---|
| 400 | 0.0250 | 0.0233 | 0.0150 | [0.450, 0.550] / [0.633, 0.727] / [0.870, 0.930] |
| 300 (fallback) | 0.0289 | 0.0269 | 0.0173 | [0.442, 0.558] / [0.626, 0.734] / [0.865, 0.935] |
| 200 (R0) | 0.0354 | 0.0330 | 0.0212 | [0.429, 0.571] / [0.614, 0.746] / [0.858, 0.942] |

PASS = all three β inside their 2σ band; FAIL = any β outside its 3σ band
(3σ at N=400: [0.425, 0.575] / [0.610, 0.750] / [0.855, 0.945]);
MARGINAL between.

**DS-2 — P–P/KS.** One-sample KS distance `D_N = sup|ECDF(q) − q|` of the
PIT values against Uniform(0,1), per cell per channel. Asymptotic critical
values `c(α)/√N`, `c(0.05) = 1.358`, `c(0.01) = 1.628`:

| N | D at 95 % | D at 99 % |
|---|---|---|
| 400 | 0.0679 | 0.0814 |
| 300 | 0.0784 | 0.0940 |
| 200 | 0.0960 | 0.1151 |

PASS = `D ≤ D_95`; FAIL = `D > D_99`; MARGINAL between. This is the §9
second CONFIRM clause ("the P–P curve is inside the 90 % band") made
operational — with the band tightened to the standard 95/99 KS convention
and fixed here. Pooling PIT across truths is a secondary read only (edge
effects differ per truth); the per-truth statistic is primary.

**DS-3 — MAP bias.** `b = ⟨grid-argmax MAP⟩ − h_true ± sd/√N` per cell per
channel (grid-argmax primary; parabolic and posterior-mean reported
alongside, never substituted — the closed-loop §5/§6 ordering). Bands reuse
the closed-loop §6 frozen edges verbatim: in-band `|b| ≤ 0.010`;
defect-scale `|b| ≥ 0.030`; intermediate = MIXED-scale. Anchor:
Δ2 = +0.0113 ± 0.0043 (registered run, 2D, independent texture).

**DS-4 — rail statistic.** `R_low`, `R_high` = fraction of seeds with
grid-argmax at the low/high grid edge, per cell per channel. Anchors
(committed): starved single-host 1D `R_low = 1.000` (200/200);
`--numerator-pdet on` diagnostic 1D `R_low = 0.000` (0/50); registered-run
2D `R_low/R_high = 0.005/0.035`. Production signature being mirrored:
**railed LOW at h = 0.600, both venues** — a high-edge rail is NOT
reproduction and is reported as its own finding.

**DS-5 — width vs F5 forecast (Stage 4 leg 3, coarse screen).** Per-seed
posterior sd; cell median `σ_med`. Compared against the F5 closure engine
(`scripts/bridge_closure/sigma_z_sigma_M_forecast.py`, stage-1 procedure:
read σ_eff/H₀ at the venue's (σ_z, σ_M), rescale from the N = 400 baseline
by `sqrt(400/1500)`, floor caveat ≥ ≈1.4 %, `--out` redirected per the
RUNBOOK-8 gotcha). Screen band, fixed blind: `W = σ_med/σ_F5 ∈ [0.5, 2.0]`.
Declared honestly: F5 is single-host and its metric is RMSE-based σ_eff, so
this is a **factor-2 consistency screen, not the leg-3 fine read**; the fine
read requires a matched-population F5 run and is **NOT-EVALUABLE as
registered** (§9 item 3) — it is appended to §11 if run, never silently
folded into a branch.

**DS-6 — rail-reproduction contrast (the Q2 statistic).** Using DS-4's
`R_low` of the **1D channel**:

- **RAIL-REPRODUCED**: `R_low(B2) ≥ 0.90` at **all three truths** AND
  `R_low(B0) ≤ 0.05`.
- **RAIL-NOT-REPRODUCED**: `R_low(B2) ≤ 0.05` at all three truths AND
  B2-1D passes DS-1 and DS-2.
- otherwise → MIXED: report the full dose–response `R_low(σ_z)` over
  {0, 0.010, 0.035} and the truth-dependence; do not force. In particular
  `R_low(B0) > 0.05` — railing under impostor confusion at *perfect*
  per-candidate z — would implicate the ball/N-2 estimator structure rather
  than photo-z, and is a first-class named finding.

Thresholds 0.90/0.05 are fixed here, blind: the committed anchors are 1.000
and 0.000; at N = 400 the binomial 2σ width at these rates is ≤ 0.03, so
the two conditions cannot both fire and neither can fire by fluctuation
from the other's anchor. Direction is part of the statistic (railed LOW).

**DS-7 — in-loop generator-closure identity (Stage 4 leg 2, in-loop form).**
Because generator and estimator share `S_4D` **by construction**, the
absolute-count audit reduces in-loop to an accounting identity: per cell,
`|N_det/(⟨n_drawn⟩ · p̄) − 1| ≤ 0.05`, with `p̄` the mean `S_4D` acceptance
over a fresh 10⁶-proposal MC at the cell's truth. A violation is an
instrument defect (V-class), not a physics finding. The **production** leg-2
result stands separately (D̃/D = 0.926; p0-window CONFIRMED dominant
×1.342; `GATE_PACKAGE_FINAL.md` §2.6 / `FIXB_PATHA_PACKAGE.md` §0–§1) and
is **not re-run here** — its venue-agnosticism is PENDING-AUTHOR-CONFIRMATION
(§9 item 2).

## 8. Edge-contamination guard (applies to DS-1/DS-2 reads only)

A seed is edge-loaded (per channel) if `edge_mass > 0.01`. A cell×channel is
**EDGE-CONTAMINATED** if > 10 % of its seeds are edge-loaded; its DS-1/DS-2
values are then reported but carry **no gate weight** (the truth was placed
too close to the grid boundary for HPD/PIT to be meaningful). DS-4/DS-6 are
**exempt** — rails are their subject matter, not their contamination. The
guard exists so a boundary-truncation artefact can never masquerade as a
coverage verdict in either direction.

## 9. NOT-EVALUABLE registry (doc-required inputs this instrument cannot build)

Flagged honestly per the mission contract, **never approximated silently**:

1. **Stage-5 third stop-digging condition, production side** ("no unmodeled
   selection between generator and estimator"): in-loop it holds by
   construction (DS-7); for production it is carried by leg 2's standing
   result **plus the open f_k–pool-coupling intake thread** (RUNBOOK-8 §1
   item 5, discovered by the D1 N2 root-cause). Any REPORT-BOUND presented
   to the author is **explicitly conditional** on those two items and the
   verdict text must say so.
2. **Leg 2 re-run for the synthetic venue**: not needed in-loop (DS-7);
   the standing production result's transfer is PENDING-AUTHOR-CONFIRMATION.
3. **Leg 3 fine read**: DS-5 is a factor-2 screen; the matched-population F5
   run is a registered follow-up, appended if executed.
4. **Production in-catalogue host-mass kernel** (R&V15 `L_cat_with_bh`):
   balls carry redshift only; NOT-EVALUABLE here.
5. **GLADE n(z) / completeness map / sky-cone geometry / host
   incompleteness (`f_incl < 1`)**: the ball is a z-window Poisson
   caricature; NOT-EVALUABLE here.
6. **`volume_deconv` kernel form** (production `HOST_Z_KERNEL` of record):
   evaluable only if optional cell O1 is built; otherwise NOT-EVALUABLE.

## 10. Validity: determinism, controls, provenance, abort criteria

- **V1 — plumbing control** (cell V1): ball path at λ = 0, σ_z = 0 must give
  MAP = 0.730 exactly, both channels, all 50 seeds (the committed
  `--f-cat 1.0` precedent through the NEW code path). Any failure ⇒ STOP.
- **V2 — HPD port certification**: `calibration_gate.hpd_contains` must agree
  boolean-exactly with `pp_coverage._hpd_contains` on 1000 random synthetic
  posteriors (unit test, runs in CI before any cell).
- **V3 — determinism**: re-running any seed with the same config must
  reproduce its record bit-identically (spot-checked on 3 seeds per cell in
  smoke). The instrument is seed-pinned end to end
  (`np.random.default_rng(seed)`, no wall-clock state).
- **V4 — texture certification**: the `dl_binned` detected set must show
  `corr(ln σ_dL, ln d_L) ∈ 0.82 ± 0.10` and marginal σ quantiles matching
  the CSV's within bootstrap noise. Failure ⇒ the texture cells are void;
  `independent`-texture cells are unaffected.
- **V5 — R0 reproduction**: before R0's HPD/PIT read is quoted, the readout
  layer must reproduce the committed aggregate MAP statistics from
  `per_seed` to ≤ 1e-12 relative.
- **Config provenance**: every run JSON embeds `git_commit` (with dirty
  flag), the full config dump, the seed list, wall time, and worker count —
  the `closed_loop_gfrac` convention. Runs that would execute on a dirty
  tree STOP instead.
- **Abort criteria**: (a) smoke extrapolation > 12 h wall ⇒ drop to the
  registered 300-seed fallback; if still > 12 h ⇒ STOP, report, author
  call. (b) non-finite `ln_post` in > 1 % of any cell's seeds ⇒ STOP
  (instrument defect). (c) any V-failure ⇒ STOP. No band may be adjusted
  after any readout.
- **GATE-NOT-TRUSTWORTHY trigger set** = {V1…V5 failure, DS-7 violation,
  abort (b)} ∪ {both decision cells EDGE-CONTAMINATED in the channel being
  read}. B0/B1/B2 *measurement* anomalies are findings, never trustworthiness
  escapes — the controls that certify the instrument (V1–V5) are separate
  from the cells that measure the estimator, by design.

## Branches (the stage-4/5 adjudication; presented to the author, never self-adjudicated)

Stage-5 decision table of record (`docs/RESEARCH_CYCLE.md`, verbatim):

> | verdict | action |
> |---|---|
> | **CALIBRATED + narrow** | measure — report the H₀ measurement |
> | **CALIBRATED + wide (≈ forecast)** | **stop digging, report the bound** |
> | **DEFECT** — ≥3σ coherent class displacement, or coverage failure | fix via `/physics-change` |
> | **UNDETERMINED** | identify and run the *one* measurement that decides; return to stage 2 |

and the stop rule: "'Stop digging' requires all of: coverage pass **and**
width on the F5 forecast **and** no unmodeled selection between generator
and estimator."

- **GATE-NOT-TRUSTWORTHY** — any §10 trigger fires. The instrument's own
  verdict is void; report which control failed and why; no stage-4 leg-1
  claim of any kind may be made. (This branch exists so an instrument bug
  can never be laundered into either a PASS or a DEFECT.)
- **KEEP-DIGGING** — gate trustworthy AND (a) DS-6 = RAIL-NOT-REPRODUCED
  (a calibrated 1D channel survives σ_z = 0.035 in-loop ⇒ the production
  rail has an additional owner — consistent with pre-reg row 5's "the
  production rail has a bigger owner than N-2"), OR (b) DEFECT-class:
  DS-1 FAIL or DS-2 FAIL in a non-exempt decision cell×channel that is not
  the registered starvation signature ⇒ stage-5 DEFECT row, fix routes
  through `/physics-change`, author-gated. In case (b) the "one measurement
  that decides" (stage-5 UNDETERMINED row) must be named in the readout.
- **REPORT-BOUND** — gate trustworthy AND A-2D and B2-2D PASS DS-1+DS-2 AND
  DS-6 = RAIL-REPRODUCED AND DS-5 in-band wherever evaluable ⇒ the 1D rail
  is a **measured property of the estimator class under photo-z starvation**
  in a truth-known loop (stage-5 "CALIBRATED + wide (≈ forecast): stop
  digging, report the bound"), **conditional on** §9 items 1–3, stated
  verbatim in the verdict. Per the author value ruling this is a fully
  legitimate outcome, not a consolation prize.
- **MIXED — first-class, non-forcing.** Anything else: partial dose
  response; truth-dependent railing; a channel split (e.g. 2D passes while
  B0 1D miscalibrates — the N-2/kernel-form structure surfacing in the ball
  venue); DS-1 and DS-2 disagreeing; O1 (if run) contradicting B2. Report
  the split directly with all DS values; do not force a branch. A B0-1D
  anomaly at σ_z = 0 is specifically pre-named here as *the impostor-ball
  analog of the N-2 finding* and must be reported under that name.

**Anti-tuning:** every threshold above (400/300 seeds; 2σ/3σ binomial
bands; KS 95/99; 0.010/0.030; 0.90/0.05; [0.5, 2.0]; 0.01/10 % edge guard;
±0.10 texture corr; 0.05 DS-7) is fixed at this commit, derived analytically
or quoted from committed artifacts, and may not be adjusted after any
readout. The git object of this file is the evidence of what was registered;
any later change is visible as a diff and is by construction an amendment,
not a registration.

**Model/effort policy for the readout:** mechanical extraction and
fingerprints at low effort; interpretation and any adversarial pass at high
effort; **the branch call is presented to the author, never
self-adjudicated.**

---

Verdict to be appended below by the session that reads out the run — after
this file is committed, no edits above this line.

---

## 11. Appendix log (append-only)

*(code commit of the new module, smoke results, deviations discovered during
the build/run, and the readout are appended here with dated headings; the
original text above stays.)*
