# [AGENT] RHS2 completion-side per-draw inflation confirmation (C2_star_review.md Task 3(b))

**Instrument**: `rhs_inflation_confirmation.py` (this directory). No production code edited;
no commits. Data: `rhs_chunks/{task0_chunk0,task5_chunk1,task20_chunk2}_twin/` (rsynced from
`$(ws_find emri)/p3_2d_rhs2_20260826/task{0,5,20}/rhs2_chunk{0,1,2}_twin_work/simulations/`).
Full per-chunk numbers: `rhs_inflation_confirmation.json` (also
`rhs_inflation_checkpoint.json`, the same content, written incrementally per chunk).

## What was tested

`C2_star_review.md` Task 1's completeness criticism + Task 2's elimination flags the
completion-class venue (`host_mode="population_selected"`) as the prime suspect for the
unattributed residual factor X_id ≈ 2.506 / X_BR ≈ 2.297: it assigns each drawn event's
`M̂_z` from the SNR-weighted donor Fisher row's **own** mass — unlinked to the drawn
`z_true` — while the estimator's own predictive `ḡ₂ = B₂/β̄_Ḡ_φ` requires
`M̂|z ~ g_sel(z,·)/S̄_φ(z)`, i.e. a z-linked mass law (the same host-conditional law the
class-G leg already got, `_draw_2d_accepted_latents`, `correspondence_1d.py:1605-1739`).

**(a) LINKAGE TEST.** For each of 3 chunks, `z_true` was recovered for the (no-host,
unlogged) `population_selected` draw via the exact byte-identical rng replay already
registered for this purpose, `ca_rhs_scorer._replay_completion_host_z` (F10(c)), and
correlated against `ln(M)` (the donor's own, unlinked mass — the value production actually
scores). Reference: the class-G leg's own `M_z_true`/`z_true` columns, banked directly in
`p3_2d_fleet_20260825/bt_900101_work` (`host_mode="catalogue_selected_2d"`, mechanically
z-linked, `M_z_true = M_true·(1+z_true)`).

**(b) INFLATION TEST.** Per draw, a LINKED-mass counterfactual was built using the *exact*
class-G mass-law kernel (`correspondence_1d.py:1698-1709`: host ~ `catalogue_selected_host_draw_weights`
(`w_g·S̃_φ,g`), `M_true ~` Eddington-shifted host mass, `M_z_linked = M_true·(1+z_true)`)
evaluated at each completion draw's own recovered `z_true` (no re-draw of `z`, no S₄D
rejection — only the `M` column is swapped). Both the unmodified (donor-M) replay and the
linked-M counterfactual were then re-scored through the **identical** production wholesale
call (`ca_rhs_scorer._score_events_2d` → `run_mirror_seed_inprocess`, `catalogue_numerator_survival_2d="mz_sel"`,
`center="eff"` — the twin arm's own registered flags, `task0_rhs2_output.json`), so only the
`M` column differs between the two scored dataframes.

## Linkage-test numbers

| | r(ln M, ln(1+z)) |
|---|---|
| completion draws (donor M, unlinked) — task0/chunk0 | +0.0118 |
| completion draws (donor M, unlinked) — task5/chunk1 | +0.0170 |
| completion draws (donor M, unlinked) — task20/chunk2 | +0.0936 |
| **class-G reference** (`M_z_true` vs `z_true`, bt_900101, n=200) | **+0.1243** |
| linked-M construction check (`M_z_linked` vs `z_true`) | +0.065, +0.185, −0.035 |

The completion-class donor mass shows weak-to-no dependence on `z_true` (r ≈ 0.01–0.09), the
class-G reference itself is *also* weak (r ≈ 0.12) — `ln(1+z)`'s dynamic range (z_true
median 0.115, 1–99% range 0.014–0.271) is small relative to the ~7-decade spread of `ln M`,
so this correlation metric is a weak diagnostic in either direction and does not by itself
discriminate the mechanism. It is directionally consistent with *some* extra decorrelation on
the completion side but is not decisive on its own — the inflation test below is.

## Inflation-test numbers (the decisive measurement)

| chunk | n accepted (donor / linked) | w2 mean, donor-M (unlinked) | w2 mean, linked-M | X = w2_donor / w2_linked |
|---|---|---|---|---|
| task0/chunk0 (seed 980001) | 177 / 176 | 0.005899 | 0.308328 | 0.0191 |
| task5/chunk1 (seed 980502) | 183 / 181 | 0.017430 | 0.264589 | 0.0659 |
| task20/chunk2 (seed 982003) | 188 / 187 | 0.020191 | 0.353132 | 0.0572 |
| **pooled (mean ± SEM, n=3 chunks)** | | | | **0.0474 ± 0.0144** |

(As a consistency check: the re-scored donor-M w2 mean matches the already-banked cluster
`event_likelihoods.csv` w2 mean to machine precision in all 3 chunks — 0.005899/0.005899,
0.017430/0.017430, 0.020191/0.020191 — confirming the local replay pipeline reproduces the
cluster run exactly and the swap isolates the `M` column only.)

**Predicted**: X_id = 2.506 (from `C2_star_review.md` Task 2's elimination — the factor by
which the completion-side RHS is hypothesized to be *inflated* relative to a correctly-linked
mass law, i.e. donor-M w2 should exceed linked-M w2 by ≈2.5×).

**Measured**: X = 0.0474 ± 0.0144 (pooled over 3 chunks) — the donor-M (as-implemented) w2 is
**~21× smaller**, not ~2.5× larger, than the linked-mass counterfactual. Both direction and
magnitude are wrong: z-score vs X_id = **−171σ**.

## Verdict: **REFUTED**

The "unlinked donor mass inflates w2" mechanism, in the specific form tested here (linked-M
built from the class-G host-conditional Eddington-shifted mass law at the completion draw's
own recovered z_true, scored through the identical production pipeline), does not reproduce
the predicted inflation — it produces a large *deflation* in the opposite direction. This
falsifies the "prime suspect" hypothesis as a mechanism for the X_id ≈ 2.506 / X_BR ≈ 2.297
residual, at least under this operationalization of "linked M̂_z."

**Caveat (disclosed, not resolved)**: the class-G host-conditional mass law draws from
catalogue-galaxy central-BH masses (`pool.M`/`pool.M_error`, median ≈1.17e6 M_sun in the
class-G reference, 1–99% range 1–9.65e6 M_sun) at a random host independent of the donor
Fisher row, whereas the donor's own mass (median ≈5.8e5 M_sun in these chunks) is tied to a
specific SNR-weighted EMRI Fisher row. The 21×-in-the-wrong-direction result is large enough
that a scale mismatch between "host BH mass drawn from the pool" and "the mass the with-BH
scorer expects at this donor row's own `(d_L, M)` Fisher block" is a plausible confound *of
this specific counterfactual construction*, not necessarily evidence that donor-mass linkage
is irrelevant to the venue-drift residual altogether. The test as specified in the review
(reuse the committed class-G mass-law kernel verbatim, substituting only `z_true`) was
followed literally; a different construction (e.g., linking only via a `z`-conditional
rescaling of the donor's own mass rather than redrawing from an independent host) was not
attempted here and would need fresh registration before being run.

## Files

- `rhs_inflation_confirmation.py` — the instrument (this directory)
- `rhs_inflation_confirmation.json` — full numeric output
- `rhs_inflation_checkpoint.json` — per-chunk checkpoint (identical content)
- `rhs_chunks/{task0_chunk0,task5_chunk1,task20_chunk2}_twin/{event_likelihoods,prepared_cramer_rao_bounds}.csv`
  — the retrieved per-draw artifacts

---

## Round 2 — PA-2D-10 (2026-08-26; alternative counterfactual construction) [AGENT]

**Registration**: `PREREGISTRATION_P3_2D_20260825.md`, final amendment block, PA-2D-10
(author-granted, resolving round 1's own disclosed caveat above). Round 1's linked-M
counterfactual drew an INDEPENDENT catalogue host (class-G mass-law kernel) — a plausible
scale-mismatch confound. PA-2D-10 replaces the independent draw with a construction that
changes ONLY the redshifting and holds the donor's own mass scale fixed:

    M_hat_z,linked = M_donor,source * (1 + z_true,replayed)

**Construction detail (column identification):** `prepared_cramer_rao_bounds.csv` "M" is
DETECTOR-frame `M_hat_z = M_source*(1+z)` (`darksiren_emri/datamodels/parameter_space.py:261-268`,
confirmed, not source-frame). The CRB CSV carries no source-frame mass column, so
`M_donor,source` was recovered as `M_donor,M_hat_z / (1 + z_donor)`
(`physical_relations.redshifted_mass_inverse`), with `z_donor` recovered by inverting
`dist(z, h=H_TRUE)` against the donor's own `luminosity_distance` column — the SAME `(z, h)`
pair `parameter_space.py:268/273` used to set both `M` and `luminosity_distance` at generation
time (`physical_relations.dist_to_redshift`). The donor row was then re-redshifted onto the
completion draw's own recovered `z_true` (F10(c) byte-identical rng replay, IDENTICAL to
round 1) via `physical_relations.redshifted_mass`. No independent host/mass draw at all.
Same 3 chunks, same seeds, same production wholesale scoring call
(`_score_events_2d`/`run_mirror_seed_inprocess`, `catalogue_numerator_survival_2d="mz_sel"`,
`center="eff"`), only the `M` column differs — round 1's replay/scoring path reused exactly.

**Instrument**: `rhs_inflation_alt_construction.py` (this directory). Run per-chunk in
foreground (task0_chunk0_twin, task5_chunk1_twin, task20_chunk2_twin ≈ 160-235s each), then
the summary pass. Output: `rhs_inflation_alt.json`.

**Per-chunk results:**

| chunk | seed | w2_donor | w2_linked | X_alt | median M̂_z donor | median M̂_z linked |
|---|---|---|---|---|---|---|
| task0_chunk0_twin | 980001 | 0.005899 | 0.005904 | 0.9992 | 5.812e5 | 5.816e5 |
| task5_chunk1_twin | 980501 | 0.017430 | 0.017436 | 0.9996 | 5.616e5 | 5.587e5 |
| task20_chunk2_twin | 982001 | 0.020191 | 0.020188 | 1.0001 | 5.891e5 | 5.879e5 |

**Pooled**: X_alt = **0.9997 ± 0.0003** (mean ± 3-chunk SEM). z-score vs X_id (2.506):
≈ −5644σ (SE-tight; the measured value sits essentially exactly at 1, nowhere near 2.5).

**M̂_z sanity check**: pooled median M̂_z donor = 5.812e5 M_sun; pooled median M̂_z linked =
5.816e5 M_sun (ratio 0.9993). The round-1 scale mismatch (host-pool BH mass, median ≈1.17e6
M_sun, vs donor Fisher-row mass, median ≈5.8e5 M_sun) is GONE by construction — donor and
linked masses now differ only by the small `(1+z_donor)` vs `(1+z_true,replayed)` ratio, as
intended.

## Verdict (round 2): **REFUTED**

X_alt = 0.9997 ± 0.0003 is indistinguishable from 1 (essentially no inflation or deflation)
and nowhere near the predicted X_id ≈ 2.506 / X_BR ≈ 2.297 residual class, in EITHER
direction. With the donor-mass-scale confound removed, re-redshifting the donor's own mass
onto the completion draw's replayed z_true changes w2 by <0.1% — the completion-side M̂_z
redshift-unlinking is NOT the mechanism for the ×2.5 residual. Per PA-2D-10's pre-registered
disposition, the residual attribution moves off the completion-mass axis entirely; candidate
next steps (class-G side's own draw-law contraction vs Σ̃^4D, or the identity's
acceptance-measure assumption) are registered in PA-2D-10 for the author to select from — the
STUCK response activates.

## Files (round 2)

- `rhs_inflation_alt_construction.py` — the instrument (this directory)
- `rhs_inflation_alt.json` — full numeric output
- `rhs_inflation_alt_checkpoint.json` — per-chunk checkpoint (identical content)
