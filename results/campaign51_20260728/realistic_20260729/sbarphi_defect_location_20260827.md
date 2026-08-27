# Locating the class-G S̄_φ de-double-weight fix (grant, defect, code)

Prepared 2026-08-27 for the [P3-2D] thread resume. Scope: **locate**, not implement — no
production code touched, no `/physics-change` gate run. This document is the presentation
package for that gate, to be executed by the author or an author-directed session.

---

## 1. Where the fix was granted

**There is no verbatim author quote in the record that grants this specific fix.** That is
itself a finding, given the repo's own attribution-precise convention ("quote the author's
verbatim words; mark any itemisation as orchestrator-derived", CLAUDE.md §Proposing
decisions).

What the record actually contains:

- `results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md:2990`
  (row #209, orchestrator narration, no author quote): *"State of the thread: C₂\* CORRECT;
  class-G S̄_φ double-weight REAL (13.5–16%, fix granted); the dominant ~×2.5 residual is
  UNATTRIBUTED on either side ... Returns to the author with options."*
- `BIAS_HISTORY_LEDGER.md:2992` (row #210, same phrasing): *"S̄_φ double-weight real
  (13.5–16%, fix granted, not yet run)"*.
- `BIAS_HISTORY_LEDGER.md:2994` (row #211, tagged **"Author ruling:"** — not **"Author ruling
  (verbatim):"**, unlike rows #206/#208 which do carry a quoted string): *"the measured S̄_φ
  double-weight (fix granted, UNRUN — first action on thread resume)"*.
- `results/campaign51_20260728/realistic_20260729/STUCK_P3_2D_SYMPTOM_CARD_20260826.md:22`:
  *"A real, measured selection double-weighting exists in the empirical venue's draw law
  (13.5–16% tilt) — sign-correct but ~7× too small; the fix is authorized but unrun."*
- `results/campaign51_20260728/RUNBOOK_NEXT_SESSION_34.md:16-19`: *"First action on thread
  resume: the GRANTED-but-UNRUN class-G S̄_φ de-double-weight fix + fleet re-run (~2–4 CPU-h)
  — measured 13.5–16%, ... necessary regardless of the big residual."*
- `results/campaign51_20260728/RUNBOOK_NEXT_SESSION_35.md:73`: same "GRANTED-but-UNRUN"
  phrasing, no quote.

**What the record does contain, adjacent and dated the same day**, is a verbatim grant for a
*different, related* item — the alternative counterfactual construction (PA-2D-10), which is
NOT the S̄_φ fix:

- `results/campaign51_20260728/realistic_20260729/PREREGISTRATION_P3_2D_20260825.md:336`:
  *"PA-2D-10 (2026-08-26; **author-granted** alternative counterfactual construction —
  resolving row #209's operationalization caveat)."*

Reconstructing the chain: row #208 (`BIAS_HISTORY_LEDGER.md:2988`) has a genuine verbatim
quote — *"Author ruling (verbatim): 'approved' — the F4 C₂\* re-derivation pass is
authorized"* — but that authorizes the **blind re-derivation + adversarial review** (which
*discovered* the double-weight as a byproduct, in
`p3_2d_forensic_20260826/C2_star_review.md`), not a fix to it. The fix itself is first
proposed in that same review's Task 3, option (b1) (`C2_star_review.md:167-176`, quoted in
§2 below), under the recommendation "(c) now, (b) next, (a) never" — i.e. the reviewer's own
recommendation was to defer (b1), not run it immediately. Between row #209 ("returns to the
author with options") and row #210 (PA-2D-10 executed), an author decision evidently
occurred, but only the PA-2D-10 half of it is recorded with the "author-granted" tag; the
S̄_φ fix's authorization is asserted by the orchestrator at every mention from row #209
onward with no corresponding quote.

**Conclusion for the author:** the fix is repeatedly described as "granted" across four
files, consistently and without contradiction, so it is very likely a real verbal grant this
session — but the ledger does not preserve the words, unlike its own practice elsewhere in
the same row range. Recommend treating "granted" as reliable (the pattern is too consistent
and too specific — "granted, not yet run", "authorized but unrun" — to be a hallucinated
placeholder) but flagging the missing quote as a ledger hygiene gap, not re-litigating the
grant itself.

---

## 2. The defect, in the record's own words

**Discovery** — `p3_2d_forensic_20260826/C2_star_review.md:29-34` (Task 1, item 7 of the
blind-derivation review, "the flag is REAL"):

> "**The flagged class-G venue drift** — verified in code: `_draw_2d_accepted_latents`
> (`correspondence_1d.py` :1605–1760) draws z_true ∝ k̄_g·S̄_φ (`_draw_kernel_survival_redshifts`
> :1440–1502, 1D law unchanged) AND layers Bernoulli(S_4D) on top, while the mz_sel numerator
> carries no per-candidate S̄_φ. The S̃_φ,g host weight cancels against the z-conditional's
> normalizer, so the accepted-event law is exactly (model class-G law) × S̄_φ(z_ev),
> renormalized. The flag is REAL. ✓"

**Measurement** — `C2_star_review.md:107-110` (Task 2 conclusion):

> "the flagged double-S̄_φ class-G venue drift is REAL and now MEASURED — a 13.5% (identity) /
> 16.0% (BR) deficit, incidentally matching the A20 F9 line's 'review-computed venue-drift
> reference' −13.6% — but it explains only ~0.865 of the observed ~0.345. It does **not**
> reproduce the observed ratios..."

**Mechanism, algebraically** — `venue_drift_adjudication.py:1-13` (the reviewer's own
scratch-file header, matching the review verbatim):

> "venue accepted-event law = model class-G law x S_bar_phi(z_ev), renormalized by
> Sigma~^{phi4D}/Sigma~^{4D} = <S_bar_phi>_{model,1}."

**The code's own docstring discloses the defect it implements**, `correspondence_1d.py:1624-1628`
(`_draw_2d_accepted_latents`, describing its own rejection-sampling construction):

> "accepting the WHOLE triple with probability `Bernoulli(S_4D(d_L(z_true;h),
> M_true*(1+z_true)))` reproduces exactly the target joint law **up to the (unchanged)
> z-marginal's own existing survival weighting** — 'on top of the existing F-0 machinery'..."

**Independent confirmation via elimination** — `results/.../gate_b_20260730/BIAS_HISTORY_LEDGER.md:2992`
(row #210): *"S̄_φ double-weight real (13.5–16%, fix granted, not yet run)"* — banked after
the completion-mass axis was exonerated in two independent constructions (rows #209/#210),
i.e. the double-weight is not itself sufficient to explain the ~×2.5 residual, but is real
and separately actionable.

### What is double-weighted, and why it is wrong

The generator (`darksiren_emri/validation/correspondence_1d.py`, the **validation/harness**
module that builds the b0i2d "catalogue_selected_2d" mirror-universe venue — **not** the
production estimator) applies survival probability to a class-G candidate **twice**:

1. **First application** — the per-host `z_true` is drawn (one draw per host, inverse-CDF)
   from a density that already includes the phi-marginal survival `S̄_φ(z)` as a factor:
   `correspondence_1d.py:1497` (`_draw_kernel_survival_redshifts`):
   ```python
   density_i = kernel_i * w_pop_eff_i * s_i   # s_i = S_bar_phi(z) interpolated, line 1496
   ```
   `S̄_φ(z)` is defined (`bayesian_statistics.py:1947`, `precompute_phi_marginal_survival`) as
   the **mass-marginal** of the 2D survival function — i.e. `S̄_φ(z) = ∫ S_4D(z,M) φ(M) dM`,
   already an average over the host mass distribution.

2. **Second application** — `_draw_2d_accepted_latents` then draws a latent mass `M` for that
   same host/z_true (`correspondence_1d.py:1698-1709`) and applies a **second**, independent
   survival gate — Bernoulli acceptance on the full 2D survival `S_4D(z_true, M_z_true)` at
   the specific drawn mass:
   ```python
   s4d_batch = detection_probability.detection_probability_with_bh_mass_interpolated(
       d_l_true_batch, m_z_true_batch, host_phiS_batch, host_qS_batch, h=h)   # :1712-1717
   accept_mask = u_batch < s4d_batch                                          # :1719
   ```

Because `S̄_φ(z)` (step 1) is already the mass-marginal of `S_4D(z,M)` (step 2), applying both
means the realized accepted-event population is proportional to `S̄_φ(z) × S_4D(z,M)` — the
z-marginal survival information is counted once in the density used to *draw* z_true, and
again in the density used to *accept/reject* the (z_true, M) pair. A correctly-implemented
generator should apply survival **once**: either weight the z-draw by the plain
population/kernel density and let the mass-conditional Bernoulli(S_4D) carry all of the
survival information, or keep the S̄_φ-weighted z-draw and drop the Bernoulli step in favor of
an S̃-reweighting. The implemented code does both, which over-thins the accepted sample at
high z (where `S̄_φ` falls fastest) relative to the model-side law the estimator's C₂*
constant assumes it is scoring against — this is the "double-weight" / "double-S̄_φ" defect.

Confirming the production estimator does **not** have this defect (i.e. it is harness-only,
`C2_star_review.md:19-22`, item 3, "**No S̄_φ in the with-BH numerator**"): the with-BH ("2D",
`_cat_surv_2d_on`) numerator inserts `S_4D` once, inside the mass quadrature
(`bayesian_statistics.py:6620-6646`, `_mz_sel_2d_expectation`), with no separate `S̄_φ(z)`
factor; only the *without*-BH twin numerator (`_cat_surv_on`) applies `S̄_φ(z)` once, correctly
(`bayesian_statistics.py:6362-6368`). The fix is confined to the validation harness's
generative law, not to production physics — `C2_star_review.md:170-174` labels it
"harness-only (`correspondence_1d.py`)".

---

## 3. Every site where the term enters the code

### Harness (`darksiren_emri/validation/correspondence_1d.py`) — where the defect lives

| Site | Role | Expression |
|---|---|---|
| `:1440-1499` `_draw_kernel_survival_redshifts` | Per-host z_true draw density — **1st survival application** | `:1496-1497` `s_i = np.interp(z_i_grid, z_grid, s_phi); density_i = kernel_i * w_pop_eff_i * s_i` |
| `:1605-1756` `_draw_2d_accepted_latents` | Calls the z-draw, then latent-mass draw, then Bernoulli(S_4D) — **2nd survival application** | z-draw call `:1687-1696`; mass draw `:1698-1709`; S_4D eval `:1712-1717`; accept `:1719` `accept_mask = u_batch < s4d_batch` |
| `:1624-1628` (docstring) | Self-disclosure of the compounding | "reproduces exactly the target joint law up to the (unchanged) z-marginal's own existing survival weighting" |
| `:1251-1341` `kernel_smeared_survival` / `_precompute_class_g_draw_objects` | Builds `S̃_φ,g` (host-level kernel-smeared survival) used for the *host draw weight* `w_g·S̃_φ,g` (a different, upstream use of S̄_φ — the *host selection* weight, not the z/M draw; not itself double-counted per `C2_star_rederivation.md:153-159`, but shares the same `phi_survival_table` input) | `:1265` `s_tilde_phi = kernel_smeared_survival(...)` |

### Production estimator (`darksiren_emri/bayesian_inference/bayesian_statistics.py`) — confirmed NOT double-applying (contrast/control)

| Site | Role | Expression |
|---|---|---|
| `:1947` `precompute_phi_marginal_survival` | Defines `S̄_φ(z;h)`, the mass-marginal survival table both harness and estimator read | (definition site) |
| `:6362-6368` | Without-BH twin numerator — `S̄_φ(z)` applied **once**, correctly | `if _cat_surv_on: _num = _num * np.interp(z, _z_s, _s_phi)` |
| `:6620-6646` `_mz_sel_2d_expectation` call site | With-BH ("2D") numerator — `S_4D` applied **once**, inside the mass quadrature, no separate `S̄_φ` factor | `if _cat_surv_2d_on: mz_integral = mz_integral * _mz_sel_2d_expectation(...)` |
| `:2455-2459` `path_a_mixture_objects` | `α_G_φ = β_G_φ·Σ^4D/Σ^φ` — the normalizer pairing C₂\* depends on (unaffected by the harness bug; listed for completeness per the task's "denominator/normalization" clause) | `n_hat_w_phi = sigma_phi/beta_G_phi; alpha_G_phi = sigma_4d/n_hat_w_phi` |

No other `precompute_phi_marginal_survival` / `S_bar_phi` / `phi_survival_table` consumer in
`darksiren_emri/` applies the table twice to the same candidate; the above two harness sites
are the only defect location. (Full `grep -rn "S_bar_phi\|phi_survival_table\|S̄_φ"
darksiren_emri/` cross-checked; no other production or validation module double-applies it.)

---

## 4. Current form vs. proposed corrected form

**Current (implemented) form** — accepted-event law realized by
`_draw_2d_accepted_latents`, per the reviewer's own re-derivation
(`C2_star_rederivation.md:153-159`, `venue_drift_adjudication.py:5-7`):

```
p_venue(z, M) ∝ [ k̄_g(z) · w_pop(z) · S̄_φ(z) ]   ×   S_4D(z, M)
                  \_________ z-draw density _________/   \_ Bernoulli accept _/
```

i.e. `p_venue(z,M) ∝ k̄_g(z)·w_pop(z)·S̄_φ(z)·S_4D(z,M)` — survival counted twice (once via
its z-marginal `S̄_φ(z)`, once via the full `S_4D(z,M)`).

**Proposed corrected form** — the record specifies **two mathematically-equivalent options**
and explicitly does not choose between them. `C2_star_review.md:167-170` (Task 3, option
(b1), quoted in full):

> "(b1) Class-G: remove the double survival weight in the 2D branch (z-draw from k̄_g·w_pop
> without the S̄_φ factor when the Bernoulli(S_4D) layer is active, **or** keep the z-draw and
> drop the Bernoulli in favor of an S̃-reweighting — **one of the two, not both**); harness-only
> (`correspondence_1d.py`), fleet re-run ≈ 24 seeds × 2 arms × ~2 min evaluate ≈ 2–4 CPU-h."

Written out, the two candidate corrected forms are:

- **Option A** (drop S̄_φ from the z-draw, keep the Bernoulli exact-mass gate):
  `p_venue(z,M) ∝ k̄_g(z)·w_pop(z)·S_4D(z,M)` — implement by dropping the `s_i` factor at
  `correspondence_1d.py:1497` when called from the 2D path (or passing a flat `s_phi≡1` table
  into `_draw_kernel_survival_redshifts` for that call site), leaving `accept_mask = u_batch
  < s4d_batch` (`:1719`) unchanged.
- **Option B** (keep the S̄_φ-weighted z-draw, replace the Bernoulli mass-gate with an
  S̃-reweighting): `p_venue(z,M) ∝ k̄_g(z)·w_pop(z)·S̄_φ(z) · [S_4D(z,M)/S̄_φ(z)]` — i.e. weight
  each accepted event by `S_4D(z_true,M_z_true)/S̄_φ(z_true)` rather than gating on a fresh
  Bernoulli draw of `S_4D` outright; requires reworking the acceptance step at `:1712-1719`
  into an importance weight rather than a rejection filter (a larger code-shape change than
  Option A).

**The record does not fully specify the corrected form** — it names two structurally
different fixes and defers the choice ("one of the two, not both") without a stated
preference or derivation showing the two are exactly equivalent for every downstream
consumer of the harness's accepted-event output (e.g. `s4d_at_truth`,
`n_drawn_total`/`n_rounds` diagnostics, and the rejection-sampling convergence guard at
`:1733-1738`, all of which are Bernoulli-shaped and would need rework under Option B but not
Option A). Per this task's own framing, that is an under-specified grant: picking between A
and B, and re-deriving that the chosen option reproduces the mixture's own class-G predictive
exactly (the object `Σ̃^4D`/`C₂*` was derived against), is real derivation work requiring the
`/physics-change`-style presentation-before-implementation discipline this repo enforces for
harness generative laws that feed physics identities — even though the file itself sits
outside the `physics-change` trigger list (`CLAUDE.md` lists only production modules;
`darksiren_emri/validation/correspondence_1d.py` is not among them, which is itself worth the
author's attention, since this harness law is exactly what the C₂\* physics identity is
checked against).

---

## 5. Pinning the 13.5–16% figure

**What is measured:** the ratio `R_pred(ω) = E_b[ω]·E_b[1/S_φ] / E_b[ω/S_φ]`
(`venue_drift_adjudication.py:9-11`, docstring) — the predicted attenuation of the
venue-accepted sample relative to the model-side expectation, evaluated for two per-event
weight functions `ω`: the primary identity weight `ω_identity = (1-w₂)·1_F0` and the B2-R
("BR") transform weight `ω_BR = (1-w₂)/(1+(r₂-1)w₂)·1_F0`.

**Result** (`C2_star_review.md:95-96`, table row):

> "LHS₂/RHS₂ (identity, B-T/twin) | **0.86473 ± 0.00511** (pooled 0.86288) | ...
> LHS₂,BR/RHS₂,BR | **0.84024 ± 0.00626** (pooled 0.83769) | ..."

→ deficit = `1 − R_pred`: **13.5%** for the identity weight (`1 − 0.86473 = 0.13527`), **16.0%**
for the BR weight (`1 − 0.84024 = 0.15976`), matching the stated "13.5% (identity) / 16.0%
(BR) deficit" (`C2_star_review.md:107`) and independently cross-checked against the earlier
"A20 F9 line's 'review-computed venue-drift reference' −13.6%" (same paragraph).

**On what venue:** the b0i2d "catalogue_selected_2d" mirror-universe fleet — 24 seeds × 2
arms (twin/coded), `results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825/`
(`C2_star_review.md:6-7`, inputs list: *"the 24-seed bt/bc fleet frames
(`p3_2d_fleet_20260825/`)"*), the same fleet that produced the frozen `PA-2D-9` numbers
(`BIAS_HISTORY_LEDGER.md:2986-2988`, rows #207-208).

**By which artifact:** `results/campaign51_20260728/realistic_20260729/p3_2d_forensic_20260826/venue_drift_adjudication.py`,
consuming the fleet CSVs and producing `venue_drift_per_seed.csv` +
`venue_drift_adjudication.json` (`C2_star_review.md:117-120`, "Numbers appendix": *"R_pred
per-seed vectors + diagnostics: `venue_drift_per_seed.csv`; summary:
`venue_drift_adjudication.json`."*). The same appendix records the replication check:
*"my recomputed LHS₂ and LHS₂,BR match the frozen values to <1e-8"* — i.e. the 13.5–16%
figure is a re-derivation-confirmed measurement against the already-frozen PA-2D-9 numbers,
not a fresh/uncontrolled run.

**Caveat on sufficiency, stated in the same source:** the measured 13.5–16% deficit, while
real and now measured, "explains only ~0.865 of the observed ~0.345" ratio
(`C2_star_review.md:108`) — i.e. it accounts for a small fraction of the ~×2.5–2.9 residual
that triggered the [P3-2D] STUCK/PARK. Running the fix is necessary (per runbook 34's "necessary
regardless of the big residual") but is not expected to close the identity by itself.

---

## Summary for the author

1. **Grant:** consistently asserted ("fix granted" / "authorized but unrun" / "GRANTED-but-UNRUN")
   across the ledger (rows #209-#211), the STUCK card, and runbooks 34-35 — but **no verbatim
   author quote exists** for this specific item, unlike the ledger's normal practice. Worth a
   one-line author confirmation before spending the ~2-4 CPU-h, purely for ledger hygiene.
2. **Defect:** confirmed real, confirmed harness-only (`darksiren_emri/validation/correspondence_1d.py`,
   NOT the production estimator), confirmed by the code's own docstring. `_draw_2d_accepted_latents`
   draws `z_true` from a density already weighted by `S̄_φ(z)` (`:1497`), then applies a second,
   independent survival gate `Bernoulli(S_4D(z,M))` (`:1719`) on top.
3. **Corrected form is NOT fully specified.** The record offers two non-equivalent-in-code-shape
   options (A: drop `S̄_φ` from the z-draw; B: drop the Bernoulli gate for an S̃-reweighting) and
   defers the choice. Implementing this fix is therefore not a mechanical one-line change — it
   needs a short derivation confirming the chosen option's accepted-event law matches the
   mixture's own class-G predictive that `Σ̃^4D`/`C₂*` was derived against, before code changes,
   per this repo's physics-change discipline (even though the trigger-file list does not
   currently name this harness module).
4. **13.5-16% is real but not sufficient.** It is a confirmed re-derivation of a measurement
   already implicit in the frozen PA-2D-9 fleet numbers, not a new run; it explains only a
   fraction of the dominant ~×2.5 residual still UNATTRIBUTED per row #210/#211.
