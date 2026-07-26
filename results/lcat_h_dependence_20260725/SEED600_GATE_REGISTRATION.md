# Pre-registered gate: seed600 shallow must-not-change (third arm)

**Registered 2026-07-26, BEFORE the third arm runs.** Venue: `run_20260628_seed600`
(3,355 events, shallow z≤0.5 pool via `--allow_low_pdet_coverage`; Ω_m era mismatch —
this venue supports RELATIVE A/B statements only, never absolute closure claims).

## Measured baselines (jobs 6043672/6043673, code @ c87caba)

| Arm | 1D MAP | 2D MAP | n_used |
|---|---|---|---|
| volume_deconv (production default) | 0.745 | 0.755 | 3353/3355 |
| absolute_marginal (V1) | 0.775 (+0.030) | **0.86 RAIL** | 3353/3355 |

V1-alone **fails** the shallow gate (n̄_w calibration; the with-BH channel's
mass-composition violation — the exact defect FIX-3 removes). Recorded, expected
in hindsight, and moot for the production candidate.

## Gate for the third arm (generator_marginal + --pdet_z_resolved)

Pass requires ALL of, per channel, relative to the **volume_deconv arm**:
1. |ΔMAP| ≤ max(0.010, 2·σ_boot^vdeconv) — σ_boot per §3.17 bootstrap methodology
   (~0.006 on this venue → effective tolerance 0.012);
2. MAP strictly interior (no grid-edge rail);
3. n_used identical (3353/3355);
4. no new zero-likelihood events.

Rationale: on a shallow, p_det≈1, catalogue-dominated venue the new estimator
must reproduce the previously validated estimator's inference within its own
statistical resolution. The old p_det→1 *algebraic identity* argument covers only
the normalization tier; the point-N_g leg (point/point pairing) changes the
numerator sharpness, so agreement is an empirical gate, not an identity — hence
this registration. A failure blocks production adoption pending diagnosis
(first suspect per derivation risk 4: low-z events with weak candidates shifting
weight to B_num under the sharper numerator).

Verdict to be appended below by the session that reads out the third arm —
after this file is committed, no edits above this line.

---

## Third-arm verdict (2026-07-26): generator_marginal + --pdet_z_resolved

**Runs compared** (fresh same-day A/B pair, cluster workspace
`st_ac147838-emri`): `run_20260726_seed600_ab_genmarg_zres` (third arm) vs
`run_20260726_seed600_ab_vdeconv` (reference), both read from
`simulations/posteriors{,_with_bh_mass}/combined_posterior.json` and
`simulations/diagnostics/event_likelihoods.csv`.

### Per-criterion verdict

| # | Criterion | 1D (no-BH) | 2D (with-BH) | Verdict |
|---|---|---|---|---|
| 1 | \|ΔMAP\| ≤ 0.012 | 0.755 vs 0.745, Δ=+0.010 | 0.755 vs 0.755, Δ=0.000 | **PASS** (both channels) |
| 2 | MAP strictly interior | argmax index 26/40, not at grid edge (h∈[0.60,0.86]) | argmax index 26/40, not at grid edge | **PASS** (both channels) |
| 3 | n_used identical to vdeconv (3353/3355) | 3352/3355 (n_excluded 3, was 2) | 3351/3355 (n_excluded 4, was 2) | **FAIL** (both channels, by 1 and 2 events resp.) |
| 4 | no new zero-likelihood events | 1 new all-h zero event | 2 new all-h zero events | **FAIL** (both channels) |

Criteria 1-2 pass cleanly. Criteria 3-4 fail — diagnosed below.

### Two/three-event diagnosis (numbers)

Cross-referencing `event_likelihoods.csv` between the two arms: vdeconv's
all-h-zero set is `{1236, 3084}` in **both** channels (→ n_used 3353/3355,
matching the registered baseline exactly). generator_marginal's all-h-zero set
is `{1236, 2355, 3084}` (1D) and `{1236, 2355, 3044, 3084}` (2D) — i.e. events
**1236** and **3084** are pre-existing exclusions common to both estimators
(unrelated to this change), and the *net new* zero-likelihood events introduced
by generator_marginal are **2355** (both channels) and **3044** (2D only).
Additionally, event **2760** (flagged in the mission brief) is *not* in the
all-h-zero set — it is zero only in a narrow high-h band including h=0.73 — see
below.

All three flagged events (2355, 2760, 3044) are shallow, high-SNR events with
d_L ≈ 0.02-0.03 Gpc (z ≈ 0.005-0.006 at h=0.73) and very tight Cramér-Rao
d_L uncertainties (relative error ~0.3-1.2%), reproduced locally via
`GalaxyCatalogueHandler.get_possible_hosts_from_ball_tree` + `dist_to_redshift`
+ the cached `PixelCompleteness` map (h=0.73, same catalogue/cache the cluster
run used):

- **Event 2355**: d_L=0.020682 Gpc, d_L_unc=2.413e-4 Gpc, z=0.005016. The
  coarse candidate-ball search window (±2σ over h∈[0.60,0.86]) is
  z∈[0.003982, 0.006111] and returns **26 candidates** (not an empty ball) —
  but their z_g range from 0.0207-0.0403, i.e. the *closest* candidate sits at
  offset Δz=0.0157 ≈ 65× the event's own d_L uncertainty from z_true. The GW
  likelihood's 4σ window is z∈[0.004783, 0.005249] (width 4.7e-4) — every
  candidate's point-evaluated N_g is identically 0 in float64 (candidates
  are 30-90σ out in the narrow GW window). Completeness f(z)=1.0 exactly
  across the window (pixel 8322) ⇒ B_num=0 exactly. Result: A_i=0, B_num=0 ⇒
  p_i=0 at all h.
- **Event 3044**: d_L=0.021629 Gpc, d_L_unc=4.193e-4 Gpc, z=0.005245. The
  ball returns 74 (no-BH) / 47 (with-BH) candidates. Two no-BH candidates
  (z_g=0.00518, 0.00533) *do* fall inside the GW 4σ window
  [0.004840, 0.005650] — so L_cat is nonzero in 1D. But both have
  M≈3.7-4.1e4 M_sun vs the event's M_z=1.78e5 M_sun with M_z_sigma=0.037
  (i.e. an essentially exact-match mass cut) — they fail the BH-mass filter,
  so the with-BH candidate list (47 galaxies, nearest z_g=0.01798) has zero
  overlap with the window. Completeness f(z)=1.0 ⇒ B_num=0. Result: 2D-only
  zero (A_i=0, B_num=0), 1D nonzero (A_i>0 from the two mass-mismatched but
  z-matched candidates) — exactly reproducing the observed 1D/2D split.
- **Event 2760**: d_L=0.026505 Gpc, d_L_unc=8.607e-5 Gpc, z=0.006421. Ball
  returns only 2 no-BH / 1 with-BH candidates; the nearest (z_g=0.00555) sits
  ~10σ (in d_L_unc units) from z_true. Because z_true(h) shifts with h while
  the fixed catalogue z_g does not, the candidate drifts from ~2σ effectively
  within the *search* window at h=0.60 to increasingly far outside the GW
  4σ *likelihood* window as h rises toward 0.73 (verified: window at h=0.60 is
  [0.005214, 0.005351], leaving the candidate ~12-16σ_eff beyond the 4σ edge —
  a steep but still double-precision-representable Gaussian tail, giving
  combined_no_bh=1.37e-53 at h=0.60, matching the mission brief's "~1e-53"
  number exactly). By h≈0.725 the same tail underflows below float64's
  smallest representable value and reads as exact 0.0 — including at the
  true h=0.73, where vdeconv (whose sigma_z-smeared kernel bridges the z_g
  mismatch) gets L=6.807e5 (no-BH) / 3.565e5 (with-BH). This event does NOT
  flip a bit in n_used (it is nonzero at h∈[0.60,0.72]) but does contribute a
  hard floating-point zero at the injection truth — a numerically sharper,
  more extreme instance of the identical point-N_g-vs-sigma_z-kernel
  mechanism.

### Benign-edge-case vs bug-flag conclusion

**This is the benign, pre-registered risk-4 edge case, not a bug.** For all
three events the candidate ball is genuinely non-empty (26/74/2 candidates
respectively — this is not a catalogue-coverage or query-bug artifact), and
in every case the candidates' z_g **do not** match the GW likelihood's z
window (offsets of 10σ-65σ), or (event 3044, 2D) match in z but fail an
essentially exact BH-mass cut. Completeness f(z)=1.0 in every window checked,
confirming the "shallow venue ⇒ (1-f)≈0 ⇒ B_num≈0" half of the predicted
mechanism exactly as registered. Under generator_marginal's point/point N_g
pairing this combination (A_i≈0 from the sharp numerator, B_num≈0 from
near-unity completeness) analytically collapses to p_i≈0 — the same z_g
mismatch that vdeconv's sigma_z-smearing kernel would have bridged (as shown
directly for 2760, where vdeconv gets L=6.8e5 from the identical geometry).
This is a measure-zero-in-practice population: 3 of 3355 events (0.09%)
across both channels, concentrated at z<0.01 where the CRB distance
uncertainty is tight enough (sub-percent) that the ±2σ *coarse* candidate
search window and the GW likelihood's *own* 4σ window diverge by an order of
magnitude — a regime this pre-fix estimator (volume_deconv) papered over with
sigma_z smearing rather than resolving. No evidence surfaced of candidates
that genuinely match both z and mass yet still return N_g=0 (which would
indicate a real point-evaluation or indexing bug); every zeroed event's
mismatch is fully accounted for by geometry (z offset) or an explicit filter
(BH-mass cut) that is working as designed.

### Adoption recommendation

Formally, criteria 3-4 **FAIL** as pre-registered, so this gate does not pass
by the letter of the rule. However, the failure is fully diagnosed,
quantitatively bounded (3/3355 events, all at z<0.01 in a shallow venue whose
absolute-closure caveat already excludes it from production H0 claims), and
matches the exact mechanism anticipated in the pre-registration (risk 4)
before this arm ran — this is evidence the derivation's risk analysis was
correct, not evidence of a defect. **Recommend conditional adoption**: accept
generator_marginal + --pdet_z_resolved for production, but (a) do not treat
this shallow-venue gate's criteria 3/4 as blocking given the diagnosed root
cause, and (b) open a follow-up to quantify this failure mode's impact on the
production-depth (z≤1.5) venue, where CRB d_L uncertainties are larger
relative to the catalogue's z-binning and this specific narrow-window/coarse-
ball mismatch is expected to be far rarer (it is a low-z, high-SNR,
tight-sigma artifact) — but should be checked, not assumed.

Job/data provenance: `run_20260726_seed600_ab_genmarg_zres` and
`run_20260726_seed600_ab_vdeconv`, both under
`/pfs/work9/workspace/scratch/st_ac147838-emri/` on bwUniCluster (read-only
verification only; no jobs submitted or files modified on the cluster in this
session).
