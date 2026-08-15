# M7-L0 toy results — A/B frozen-edge harness, boundary-layer flux measured

**Date:** 2026-08-15 · **Authorized:** ledger row #103 item 3 ([DO]) · **Status: PRESENTED, NOT
ADJUDICATED.** Numbers only; the read below applies the rule table registered in
`M7_L0_DERIVATION_20260815.md` §3 mechanically. The author adjudicates.

**Script:** `toys/m7_ab_toy.py` · **Output:** `M7_L0_toy_output.json` · `ruff check` / `mypy` clean.

## 1. Setup as run

- N_EV = 500, K = 400 candidates/event (1 exact-membership host + 399 impostors), GL-50 quadrature,
  8 seeds (101–108), h_true = 0.73, dh = 0.005 (central difference for the slope; also used for the
  arm-A curvature second difference).
- sigma_d: bootstrapped fractional σ_dL/d_L from the production CRB CSV (`closed_loop_gfrac.
  load_sigma_triples`, the same loader m3_toy/m5_toy use).
- sigma_z: the **GLADE-empirical mix**, not a flat value — reused verbatim from
  `darksiren_emri.validation.venue_transfer.load_pruned_z_sigma` +
  `venue_transfer.build_sigma_sampler` (the production VT-D3 z-decile sampler, built against the
  same pruned catalogue the venue uses: `cluster_parent_reduced_galaxy_catalogue.csv`, 20,834,171
  rows after the m4 prune recipe). Per-candidate σ_z is z-decile matched exactly as production draws
  it (`draw_member_sigma_z`'s algorithm, reused not reimplemented).
- Population: per event, K candidates drawn on **true** z, uniform-in-comoving-volume inside the
  ball window built at **h_true** from d_obs (membership fixed, h-independent, matching the
  production fixed-K ball). z_obs = z_true + σ_z·ε for scattered candidates.
- Arm A: window edges move with h (as coded). Arm B: edges frozen at their h_true values for every h
  (same population, same draws — only the edge motion differs).
- Dose f_i ∈ {0.25, 0.5, 1.0} scales σ_z for **impostors only**; the host is never dosed.
- Host variants: **scattered** (full-dose null-arm venue — every candidate incl. the host is
  z-scattered; primary/registered) at all three doses, and **exact** (host z_obs = z_true, no noise)
  at full dose only, per the note's "if cheap" allowance.
- S_need = Δ/σ_post², Δ = **+0.0192** (the A-M2′ residual, `STAGE2_READOUT.md`), σ_post computed from
  the **toy's own** arm-A joint-posterior curvature (Laplace, second difference of the stacked lnL_A
  at h_true), not imported from production. Implied MAP shift = slope_982 · σ_post²; the M3
  background (+6e-7 in h, `M3_truncation_window.md`, CLOSED) is subtracted to isolate the M7-only
  contribution (A−B measures M3+M7 combined; the note states the M3 share is known ≤6e-7).

## 2. Numbers (mean ± seed-scatter std over 8 seeds)

| config | slope_982 (per unit h) | σ_post (toy) | S_need | implied shift, M3+M7 (A−B) | **implied shift, M7 only** |
|---|---|---|---|---|---|
| scattered, f=0.25 | −18.2 ± 21.4 | 1.941e-3 | 5292 | −6.08e-5 ± 6.77e-5 | **−6.14e-5 ± 6.77e-5** |
| scattered, f=0.50 | −20.9 ± 26.1 | 2.227e-3 | 3903 | −9.81e-5 ± 1.15e-4 | **−9.87e-5 ± 1.15e-4** |
| scattered, f=1.00 (**registered**) | −20.3 ± 26.3 | 2.478e-3 | 3136 | −1.204e-4 ± 1.490e-4 | **−1.210e-4 ± 1.490e-4** |
| exact host, f=1.00 (secondary) | −19.7 ± 25.3 | 2.477e-3 | 3142 | −1.172e-4 ± 1.436e-4 | −1.178e-4 ± 1.436e-4 |

Per-seed values (scattered, f=1.00) span both signs: seeds 101–104, 107 give a negative slope
(−72 … −11 per unit h stacked, implied shift −1.2e-4 … −4.0e-4), seeds 105, 106, 108 give a small
positive slope (+1.7 … +3.1, implied shift +1.7e-5 … +1.9e-5). The seed-scatter std exceeds the
mean's magnitude at every dose — the measured effect does not reach a stable sign across 8 seeds at
this toy scale; the reported mean is the best point estimate, not a resolved sign.

## 3. Realized boundary-layer fractions (scattered arm, mean ± std over 8 seeds)

| dose f_i | within 1σ_z of an edge | fully outside the h_true window |
|---|---|---|
| 0.25 | 10.36% ± 0.65% | 4.32% ± 0.34% |
| 0.50 | 19.27% ± 0.89% | 8.24% ± 0.50% |
| 1.00 | 33.92% ± 1.02% | 15.29% ± 0.69% |

The boundary-layer population fraction grows with dose as expected from §2 of the derivation note
(saturating toward the predicted O(0.2–0.5) range at full dose) — the mechanism's premise about
population geometry is realized in the toy. The host-exact variant's fractions (33.82% / 15.24% at
full dose) are statistically indistinguishable from the scattered variant's, consistent with the
note's claim that the host is structurally exempt from the effect.

## 4. Mechanical read (applying §3's rule table as registered)

**Registered configuration:** scattered host, full dose (f_i = 1.0).
**Implied MAP shift, M7 only: −1.210e-4 ± 1.490e-4 in h.**

| rule (from §3) | condition | this result |
|---|---|---|
| M7-CLOSED | shift ∈ [−1e-3, +1e-3] | **−1.210e-4 is inside the band** |
| M7-LIVE | shift > +1e-3 | not met |
| M7-REFUTED-ON-SIGN | shift < −1e-3 | not met (−1.210e-4 > −1e-3) |

Applying the rule mechanically: **M7-CLOSED.** The magnitude condition is decisive — the measured
shift sits roughly 8× inside the closure band, comfortably clear of both the +1e-3 (LIVE) and −1e-3
(REFUTED-ON-SIGN) thresholds, and the closure band carries the ≥30× margin over the note's §2
analytic lower bracket (6e-5…6e-4) that the false-read note states makes a CLOSED read robust to
O(1) toy-scale error.

**Sign note (consistency information, not part of the CLOSED/LIVE/REFUTED-ON-SIGN gate):** §2
pre-registered a positive sign (h pushed up) as the mechanism's structural prediction; the toy's
full-dose point estimate is negative, though its seed-scatter std is larger than the mean and 3/8
seeds are positive — the sign is not resolved at this toy scale. Per the note's own gate, sign
outside ±1e-3 would trigger REFUTED-ON-SIGN; since |shift| never approaches 1e-3 at any dose, this
gate is not engaged and the read stays CLOSED at every dose in the sweep.

**Dose trend (consistency check only, no closure weight per §3):** |implied shift| grows
monotonically with dose (6.14e-5 → 9.87e-5 → 1.210e-4 at f = 0.25, 0.50, 1.00), sub-linearly in f
(roughly ×2 over the ×4 dose range) rather than tracking the boundary-layer fraction's more than
×3 growth over the same range — a mild, not dose-proportional, trend.

## 5. Relation to the other open accounts

Per the note's §4: this reads CLOSED, so — mechanically, per the note's own stated fallback — the
dose-dependent tilt residual's account falls back to the σ_z²-class compounding terms (M1's retained
negative quadratic among them) and any remaining structure the M6R decomposition surfaces. No
identity between the M6-residual and M7 is asserted; none is supported by this result.

*No repair proposed; no instrument arm requested, per the note's closing line.*
