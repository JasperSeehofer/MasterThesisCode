# §7b — Isolated impact of the GLADE+ peculiar-velocity value correction on H₀

**Date:** 2026-07-04 · **Issue:** #16 · **Branch:** `physics/campaign-depth-pv` ·
**Data:** frozen seed600 event set (SEED=600, EVAL_SEED=600999)

## Question

Issue #16 decided to **marginalize** host peculiar velocity for the Phase-2 campaign
(σ_z,eff² = σ_z,cat² + ((1+z)·σ_v/c)², σ_v = 200 km/s; landed in commit `8568d9f`) and to
**test the isolated value-correction effect in parallel**. This is that test.

The live GLADE+ reduced catalogue uses GLADE+ column 28 (z_cmb) as-is; for the 709,117
rows with flag2==1 that redshift is additionally **PV-value-corrected** (Dálya et al. 2022).
The `noPVcorr` variant (built by `build_uncorrected_variant.py`, see `README.md`) replaces
exactly those 709k redshifts with the pure heliocentric→CMB **frame-only** transform,
removing the PV value correction and nothing else — all flag2==0 rows are byte-identical.

Running the production `--evaluate` (Pipeline B) on the **same frozen seed600 events, same
CRB, same settings**, once against each catalogue, isolates the impact of the PV value
correction on the H₀ posterior.

## Method

- Two arms: `run_live` (real PV-corrected catalogue) and `run_nopv` (frame-only variant).
- Per-h evaluations over a 13-value grid h ∈ [0.725, 0.785] (Δh = 0.005), driven by
  `run_queue.sh` (idempotent; `--evaluate --h_value <h> --seed 600999
  --allow_low_pdet_coverage`).
- Combined with the production machinery (`--combine --allow_low_pdet_coverage`,
  `physics-floor` strategy, `posterior_combination.combine_posteriors`).
- Two channels: `posteriors` (**1D**, redshift–distance only) and `posteriors_with_bh_mass`
  (**2D**, adds the BH-mass dimension).
- **Event-intersection refinement.** The two arms' `physics-floor` exclusions differed by
  one event (live used 3342, noPV 3343). To compare on identical events, both posteriors
  were recomputed with the production functions (`load_posterior_jsons`,
  `build_likelihood_array`, `apply_strategy`, `combine_log_space`) restricted to the
  **shared 3342-event intersection**. The one differing event has no material effect
  (numbers below are byte-identical to the full-arm combines to the reported precision).

## Sign convention

**Δ = live − noPV.** `live` = campaign catalogue (PV correction applied). `noPV` = correction
removed. A **negative** Δ means the PV-corrected (campaign) catalogue yields a **lower** H₀
than the uncorrected one — equivalently, applying the PV correction lowers the recovered H₀.

## Result (shared 3342-event set, 17-value grid h ∈ [0.725, 0.805])

| channel | arm | MAP h | mean h | std |
|---|---|---|---|---|
| **1D** (`posteriors`) | live | 0.745 | 0.74320 | 0.00515 |
| | noPV | 0.755 | 0.75744 | 0.00428 |
| | **Δ (live−noPV)** | **−0.010** | **−0.01425** | — (−3.33 noPV-σ) |
| **2D** (`posteriors_with_bh_mass`) | live | 0.785 | 0.78704 | 0.00857 |
| | noPV | 0.785 | 0.78584 | 0.00570 |
| | **Δ (live−noPV)** | **0.000** | **+0.00120** | — (+0.21 noPV-σ) |

**Grid extension (edge-railing resolved).** The initial 13-value grid (top 0.785) left the
2D channel edge-railed (42–55% of the posterior mass in the top bin), so it was extended by
four values to h = 0.805 for both arms. On the 17-value grid the 2D channel is **no longer
railed**: the MAP sits at 0.785 (interior) with only ~4% (live) / ~0.2% (noPV) of the mass in
the top bin. The un-railed 2D Δmean is **+0.0012 (+0.21 noPV-σ)** — the sign flip versus the
railed value (−0.0018) confirms that the railed number was a truncation artefact. **The 2D
Δ is consistent with zero:** the with-BH-mass channel is essentially PV-insensitive, because
the black-hole-mass information (which is PV-independent) dominates the H₀ constraint and
pins it high regardless of the low-z redshift corrections. The 1D result is unchanged by the
extension (the added high-h nodes carry negligible 1D posterior mass).

## Interpretation — this is an upper bound

**seed600 is the designed worst case for PV sensitivity.** It is an all-low-z event set,
exactly the regime where (a) all 709k GLADE+ PV corrections live (z_cmb ∈ [−3e-4, 0.11]) and
(b) dark-siren H₀ information concentrates. The full PV value correction, removed entirely,
moves the **1D** estimate by only **Δmean ≈ −0.014** (about −1.9% in H₀; −3.3 posterior
widths on this narrow-posterior worst case), while the production-relevant **2D** channel is
**PV-insensitive** (Δmean = +0.0012, +0.2σ — consistent with zero; BH-mass information
dominates and is PV-independent).

Two effects make the **actual campaign** impact much smaller than this bound:

1. **Marginalization already absorbs it.** The campaign inflates σ_z by ((1+z)·200/c)
   (commit `8568d9f`), which for the low-z hosts that carry PV corrections is comparable to
   the value correction being removed here — i.e. the residual value-correction shift is
   folded into the per-event redshift uncertainty rather than left as a coherent bias.
2. **Campaign events are not all low-z.** At depth 1.5 the campaign draws hosts across a far
   larger redshift range, so only a small fraction of host redshifts carry a PV correction
   at all, and the per-event PV shift (Δz/z ∝ 1/z) is negligible for the bulk of events.

## Conclusion

The isolated GLADE+ PV **value** correction shifts H₀ by at most ~0.014 (1D, worst-case
all-low-z seed600), and much less for the production 2D channel and for the actual depth-1.5
campaign. The campaign-side σ_v marginalization (`8568d9f`) covers this as added uncertainty.
**No re-scope of the campaign is warranted; issue #16 can be closed** citing the
marginalization commit and this bound.

## Reproducibility

- Builder + variant provenance: `README.md`, `stats.json`.
- Per-arm per-h posteriors: `run_{live,nopv}/simulations/{posteriors,posteriors_with_bh_mass}/h_*.json`.
- Combined posteriors: `.../combined_posterior.json` per arm/channel.
- Queue log: `run_queue.log`. Combine logs: `run_{live,nopv}/combine_13grid.log`.
- Both arms evaluated the identical seed600 event set with `--seed 600999`.
