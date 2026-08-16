# L6-DER2 §4 step 2 — adversarial verifier addendum (derivation + fused-g_sel premeasure)

**Date:** 2026-08-16 · **Scope:** verification of
`L6_DER2_CORRECT_FORM_2D_20260816.md` (commit `09c02c06`) and
`L6_DER2_GSEL_PREMEASURE_20260816.md` + `l6_der2_gsel_premeasure.py` +
`L6_DER2_GSEL_PREMEASURE_output.json` (commit `fbc60b3a`), taken together, against
the ratified channel-B context (row #114). **This addendum verifies; it does not
adjudicate — the arm decision returns to the author.** Append-only; nothing outside
this file was changed.

---

## Overall verdict

**GO-with-amendments** for proceeding to the A-FULL-2D registered-arm proposal
(a recommendation to the author, not an adjudication). The factorization error is
real, the implementation is faithful (and now independently validated on the
with-S path, which the script's own gates did not exercise), the registration
timeline is clean, and the −11.7 ± 1.0 residual is demonstrably NOT a numerical
artifact and carries the seed-level fingerprint of the known residual class — but
the headline claim must be stated per amendments V1/V2/V4 below, not as
"cancellation confirmed".

---

## A. Derivation soundness — CONFIRMED, with two amendments (V2, V4)

**The factorization error is real.** Checked against the generator's actual 2D
data model (`closed_loop_gfrac.py`, `draw_universe`, lines ~440–486): a detected
event is drawn from `w_pop(z)·φ(M)` with Bernoulli detection at probability
`S_4D(d_L(z;h_true), M(1+z))`, then correlated fractional noise
(`frac_m = σ_M(ρe1 + √(1−ρ²)e2)`) produces `(d_L_obs, M_z_obs)`. The selected
joint prior of a pinned event is therefore `w_pop(z)·φ(M)·p_det(M,z)/α`, and the
per-candidate marginal over the unobserved mass runs ONCE:
`∫dM φ·p_det·N_d(d_obs|d_L)·N_{M|d}(M_z_obs|M(1+z), d_obs)`. The coded/A-FULL 2D
weight instead carries `S̄_φ(z) = ∫dM φ·p_det` (as node weight / inside α) TIMES
`g = ∫dM φ·N` — two `∫dM`, a genuine double marginalization of φ, exactly as
§2 claims.

**Denominator/α claim — CONFIRMED.** `α(h) = ∫ w_pop·S̄_φ dz =
∫∫ w_pop·φ·p_det dz dM` (`closed_loop_gfrac.py` ~376–384) is precisely the
selected-prior normalization the fused numerator needs; no change required. Note
additionally that `log_alpha[k]` is subtracted identically from ln1 and ln2, so α
cancels in every excess (T2−T1) — the premeasure could not have detected a wrong
α either way; the claim rests on the derivation check above, which passes.

**No-d-side-change claim — CONFIRMED for this venue.** Channel A measured null
(row #113/114), and independently: the venue's CRB mass fractions are tiny
(measured on seed 20310808: `σ_Mz` 5/50/95-pct = 2.0e-8 / 8.9e-8 / 3.5e-7;
`|ρ|` median 0.015), so `proj = ρσ_M/σ_d ~ 1e-6` and the entire d-conditioning
machinery (`μ_cond`, `σ_cond`) is numerically inert here regardless of its exact
convention.

**V2 (amendment) — the D2-analogue measure prefactor is NOT in the measured
candidate, contradicting §2's "for free" clause.** §2 states the
`1/(σ_M·M(1+z))` M_z_obs-density prefactor "rides along inside the single ∫dM
and should be included in the same candidate"; §3's `g_sel` formula, and the
implementation (verbatim `μ_cond/σ_cond/x_M` conventions), omit it. In THIS
venue the omission is immaterial — with `σ_cond ~ 1e-7`, ratio-vs-density
corrections deviate from a per-event constant (tilt-neutral) by ≲1e-6 relative —
but the internal inconsistency should be recorded, and the prefactor question
must be re-decided explicitly for production, where σ_M is not tiny.

**V4 (clarification, not an error) — the sharp-likelihood limit.** With
`σ_cond ~ 1e-7`, the Hermite window is ~1e-7 wide in x_M and S is locally
constant across it, so in this venue
`g_sel(z,f;h) ≈ g_coded(z,f) · S(μ_cond·M_z_obs; z, h)`: the fusion is
effectively "evaluate the survival at the event's own observed detector-frame
mass instead of φ-averaged". This is the sharp-likelihood limit of §2's
mechanism (the effective prior at the likelihood point is φ·p_det), it explains
why the collapse is clean, and it sharpens the production-transfer question
(item F).

## B. Implementation fidelity — CONFIRMED

`g_sel_mass_factor` read line-by-line against the §3 formula and the reference
code:

- `scale = M_z_obs/(1+z)`, `μ_cond = 1 + proj(f−1)`,
  `x_M = μ_cond + √2·σ_cond·t_j`, `φ_x = φ(x_M·scale)·scale`,
  `(integrand @ w)/√π` — verbatim `completion_mass_factor_g._contract_group`
  (`bayesian_statistics.py` ~2118–2126). ✓
- S query: `detection_probability_with_bh_mass_interpolated(d_L_abs, x_M·M_z_obs,
  0, 0, h=h, **_wbh_z_kwargs(...))` — argument-for-argument the
  `precompute_phi_marginal_survival` convention (`~1925–1940`); the frames agree
  (`M_z = x_M·M_z_obs = M_source(1+z)`); `d_L_abs` is the same node value the
  outer integral computed, threaded before the `/d_obs` ratio. ✓
- 2D weight: `kern·p_gw_full·w_pop·loo_w·g_sel` — S̄_φ removed from the 2D node
  weight ONLY; c1 keeps A-FULL's full weight; LOO weights kept in both branches;
  point branch (σ_z = 0) handled with the same drop of `s_phi_p` and the
  `d_pt` absolute distance threaded. ✓ The `afull` mirror lines match
  `venue_transfer._channel_terms_at_h`'s `ESTIMATOR_VARIANT_A_FULL` branch
  (~1656–1671, ~1683–1697) exactly (floored `w_pop` argument, unfloored
  `np.interp` for s_phi — same in both).

**The four gates, assessed:**

- *base-vs-stored* (gate 1): genuine — exercises the full mirror path against the
  stored MN0X vectors. ✓
- *afull-vs-direct* (gate 2): genuine — bit-exact against the untouched
  `venue_transfer` code path; the mid-run restructure (reusing pool afull values)
  does not weaken what is compared. 2 seeds × both k is thin coverage, but
  bit-exactness leaves no room for a partial match. ✓
- *S≡1 refactor* (gate 3): valid for what it tests, but **it does not exercise
  the S-query path** (`force_S_one=True` skips the query entirely). The run
  path's with-S branch was validated only by convention-reading — a gap. **Closed
  by this verifier:** an independent dense-trapezoid reference (μ±12σ, 16001
  points, S query included, written from scratch) agrees with
  `g_sel_mass_factor` at n=64 to **max rel diff 8.0e-16** over 150 kernel-branch
  rows × both k (seed 20310808).
- *c1 bit-identity* (gate 4): **by construction, not a measurement**
  (`ln1["gsel"] = ln1["afull"]` is a shared value; the assert compares it to
  itself) — but unlike the A2 issue on the switch script, the premeasure script
  openly documents this ("documents the by-construction sharing rather than
  'discovering' it"). The 1D-untouched fact rests on code construction (no
  separate gsel c1 accumulation exists), which is sound. Recorded, no amendment
  needed.

**Independent reproduction of the aggregates:** recomputed all means/SEs from
`per_seed_rows` — excess_base +131.508 ± 0.058, excess_afull +135.787 ± 0.078,
excess_gsel **−11.740 ± 1.038** (range −18.95…−4.81, 15/15 seeds negative),
dT2 −147.527 ± 0.962; per-row identity `excess_gsel = T2_gsel − T1_afull` holds
to 0.0. All match the JSON.

## C. The −11.7 ± 1.0 residual — assessed: structural, same class as the 1D residual; NOT a numerical artifact; "cancellation confirmed" must be amended (V1)

Attacked every numerical channel that could fake the residual; all are dead:

| candidate cause | measured bound | verdict |
|---|---:|---|
| n=64 Hermite truncation vs dense reference (with S), k22−k20 signed kern-weighted rel-err | ~1.6e-16/row → tilt-fake ≲ **3e-12 nats/h** (982 events) | dead |
| adaptive n=8 vs pinned n=64 in the afull column's coded g | max rel **1.1e-15** | dead |
| S̄_φ table (n_z=1500/n_M=600 trapezoid, enters T1_afull but not T2_gsel) vs refined n_z=12000/n_M=4800 tables | event-weighted tilt **−0.026 nats/h** (crude population-mean −0.245; worst-case single-z 8.8e-3 in log S sits in the near-horizon tail where no events are) | dead |
| pool code-revision contamination | see item E | dead |

What the residual IS consistent with:

1. **Per-seed correlation r = 0.847** between the fused-form residual
   (excess_gsel per seed) and the c2 switch's own unattributed residual
   (`T2_sb − T1_base` per seed, −7.489 ± 0.065, sd 0.253) — computed by this
   verifier from both committed JSONs, 15 matched seeds. The two instruments are
   different (rigid linearized switch vs a genuinely different estimator), yet
   they share seed-level structure: the residual is realization-coupled, i.e.
   the pool-vs-model class, not an h-grid or convention artifact.
2. **Scale:** −11.7 is 8.4% of the coded channel's 139 (a 91.4% collapse of the
   A-FULL excess +135.8), the same order as the 1D repair's own leftover
   (T1_afull = +30.6 mean) and the same few-percent quality band as the ratified
   channel-B ownership (~94–106%). A wrong-term signature would leave O(100)
   nats/h or a seed-rigid offset; instead the residual is seed-scattered
   (sd 4.0) and class-correlated.

**V1 (amendment of record):** the claim to carry forward is NOT "cancellation
confirmed" but: *the fused form cancels channel B to −11.7 ± 1.0 nats/h — a
decisively nonzero, all-15-seeds-negative overcorrection of 8.4% of the channel,
whose seed-level structure is strongly correlated (r = 0.85) with the known
base-instrument residual class and which is not attributable to any numerical
convention of the premeasure (all bounded ≤ 0.03 nats/h).* Whether the
systematic negative sign is purely the pool-vs-model class or contains a small
genuine missing term (e.g. the V2 prefactor at production-relevant σ_M, or a
d-side pairing beyond A-FULL) is undetermined and does not need to be resolved
before a registered arm — the arm measures it.

## D. Registration integrity — CONFIRMED

- Derivation committed `09c02c06` at 2026-08-16 10:56:27 +0200; the premeasure
  script's mtime is 12:15, the output JSON 13:56, committed `fbc60b3a` at
  13:57:44 — the prediction predates the measurement by ~3 h. `git diff
  09c02c06..HEAD` on the derivation file is empty (no post-hoc edit).
- The script's quoted prediction matches §2/§4 verbatim in substance (collapse
  to ~few-nat level; 1D bit-untouched) and honestly flags that no numeric value
  was pre-registered for this measurement.
- No tuning knobs point at the target: the conventions (non-adaptive n=64,
  isotropic S query, `_wbh_z_kwargs` pass-through, seeds = the same first 15
  MN0X seeds as the switch study, k=20/22) are all inherited or justified, and
  this verifier measured that the only genuinely free one (quadrature order) is
  inert to 1e-15. The honest disclosure that the result landed NEGATIVE of the
  predicted "at/near zero" — rather than rounding it into "confirmed" — is
  itself evidence against tuning.

## E. Pool provenance — CONFIRMED uncontaminated

The injection pool spans revisions `a9f29e82` → `f6449051`. The complete diff
between them is two commits: `acaa0afe` (adds two provenance columns
`t_plunge_yr`/`p0` to `_INJECTION_COLUMNS` in `master_thesis_code/main.py` — 7
inserted lines, output-schema only) and `f6449051` (results files only). No SNR
computation, waveform, threshold, or detection-semantics change. The premeasure
is not contaminated; earlier-revision rows merely lack the two optional columns.

## F. Production-transfer scope — one paragraph, as posed

The fused-form conclusion transfers to the production completion leg **as a
question of the same factorization structure, not as a result**. In
`absolute_marginal` (the only mode that calls g; `bayesian_statistics.py`
~3179), the 2D completion numerator is `∫(1−f_k)·p_gw·dVc/(1+z)·g_i dz`
(~4334–4363) with `g_i = completion_mass_factor_g` a bare φ-contraction — no
`p_det(M)` inside its `∫dM` — while ALL selection M-dependence lives in the
global denominator legs (`S̄_φ`-built `β^φ`/`α_G^φ`/`D̃^φ`) and, in the
instrumentation-only `selection_in_completion_numerator='1d'` cell, as a
separable `S̄_φ(z)` z-weight (~4295–4310): the same φ-double-marginalization
geometry, expressed as "p_det omitted from g's ∫dM + counted once globally"
rather than the venue's per-node `S̄_φ(z)×g` product. The reopened
`/physics-change` proposal should therefore be scoped to: *whether, under a
latent-thresholded detection model, `completion_mass_factor_g`'s `∫dM` must fuse
the with-BH survival `S(d_L(z;h), x_M·M_z,det,i)` (a `g_sel`-form object built
from the already-tabulated `S̄_φ` integrand), and what paired change (if any)
the `D̃^φ` denominator convention then requires* — noting the material
geometry differences (no host kernels/LOO structure, `(1−f)dVc` weighting,
per-event h-moving windows ~4368–4384, global-denominator rather than per-event
α, and the catalogue leg's separate `mz_integral` object), the V2 prefactor
question (σ_M is NOT tiny in production), and A3's standing ruling that venue
magnitudes do not transfer.

---

## Open questions (for the author, none blocking the arm proposal)

1. Is the systematic negative sign of the fused-form residual (15/15 seeds)
   purely the pool-vs-model class, or a small genuine overcorrection term? A
   targeted decomposition (e.g. fused form with the generator's own pool
   replayed as the prior) could separate these, if ever worth the cost.
2. Should the D2-analogue prefactor (V2) be added to the A-FULL-2D candidate
   before the registered arm, given it is immaterial in this venue but the
   derivation text says "included in the same candidate"? Either way the
   derivation §2/§3 mismatch should be resolved in text.
3. The sharp-likelihood limit (V4) makes the venue's g_sel ≈ g·S(at observed
   mass); the production σ_M regime will exercise the genuinely non-separable
   integrand — the production derivation should not inherit venue intuition
   about which limit dominates.

**Verifier verdict line: GO-with-amendments (V1, V2, V4 above; V1 and V2 are
the amendments of record, V4 is a clarification, B's gate-3 gap is closed by
independent reproduction, C's residual is structural-same-class, D and E are
clean).**

*Verifier reproductions run 2026-08-16: dense-reference g_sel check (150 rows ×
2 k, max rel 8.0e-16); adaptive-vs-pinned bound (1.1e-15); S̄_φ refined-table
tilt (−0.026 nats/h event-weighted); aggregate recomputation from per_seed_rows
(exact match); per-seed residual correlation r = 0.847; commit-timeline and
pool-revision diffs as cited. Scripts in the session scratchpad; not committed.*
