# rd-s3-readout — chair re-derivation + booking (2026-09-03, Fable 5.1 chair)

Panel law: the reader's record (READOUT_RECORD.md) is evidence, not authority. Every decisive
number below was recomputed by the chair from the checkpoints / logs; the booking is CHAIR-DERIVED
under the row #325 grant and returns to the author inside d-calibration (docket 2.3).

## 1. Re-derived numbers (all MATCH the reader)
- Exact two-sided 5 % Kolmogorov critical values, `scipy.stats.kstwo.isf(0.05, n)`:
  n=67 → 0.1632 · n=25 → 0.2640 · (n=100 → 0.1340, the harness's stale informational line).
- F = SD/floor: S 11.439 / 11.380 · T 11.332 / 11.440 (no_bh / with_bh). Sanity band [1,25]: inside.
- PIT-KS D: S 0.3217 / 0.3340 · T 0.306 / 0.3459 — all four > critical → OUTSIDE (KS is the PRIMARY
  clause, registration §2.1).
- mean(MAP) − h_true: S +0.04187 (Z 5.89) / +0.05022 (Z 6.48); T +0.0388 (Z 2.82) / +0.0488 (Z 3.58).
- score-zero |Z|≤3 by class, cell S: catalogue_hosted 9.76 / 7.15 OUTSIDE · dark 1.26 / 1.76 INSIDE ·
  all 4.93 / 4.26 OUTSIDE. The failing statistic LOCALIZES to the catalogue-hosted class (design A12).
- Count audit: all 5 z-bins |Z| < 1.2, both cells — the harness universe instrument is healthy.
- g-population: 0 mixed rows (67 S seeds 901000–901066; 25 T seeds 902000–902024; N=200 everywhere).

## 2. Two gates the reader could not close — closed here
- **§2.3 with-BH byte-identity pin: GREEN.** Chair compared `posterior.with_bh.{ln_post,sd,map_h,pit,
  hpd50,hpd68,hpd90,hpd95}` for every shared seed against `b8_cal_harness_work_ladder/`:
  S 63/63 byte-identical, T 20/20 byte-identical, max_rel_dev 0.0, 0 scalar diffs (same machine).
  Consequence (registration §2.3): F_with_bh and T/S with_bh on the overlap are identities, not
  measurements; the deferred g-byte-id criterion of row #291 is DISCHARGED by this pin (routed to
  d-s4-review's line item as the registration proposed — flagged for the author).
- **§6 g-censoring rail fraction (grid 0.60–0.86, 41 nodes):** S no_bh 10/67 = 14.9 %, S with_bh
  14/67 = 20.9 %, T no_bh 5/25 = 20 %, T with_bh 6/25 = 24 % — all > 10 % → by the registered rule
  every coverage/centering number above is a **BOUND, not a measurement**. Rail side recorded in
  the script output appended below.

## 3. Booking (chair-derived, per registration §5 — returns as fresh RULE at d-calibration)
- Cell S, both channels: **DEFECT-SIGNATURE at N=200** ("any clause out of band"): KS primary OUTSIDE,
  centering Z 5.9/6.5 OUTSIDE, score-zero OUTSIDE in the catalogue-hosted class only. Not INTERMEDIATE:
  KS fails in both channels, so the "channels split" branch does not apply. No INSTRUMENT-DEFECT
  (byte-pin green, g-population green, count audit green).
- Cell T: width-only by design; no coverage claim (standing non-claim). T/S no_bh 0.9934 REPORTED-ONLY.
- q-postflip-calibration kill criterion ("unusable in both channels at registered bands after
  revision 2") is NOT reached: this is revision 1. A revision-2 re-registration is NOT covered by
  tonight's grant (docket 2.2 NOT-covered cell) → returns to the author.
- What d-calibration must weigh (facts): the harness universe centres the MAP +0.042 ABOVE truth in
  the flipped no-BH leg while production sits −0.063 BELOW (row #286); the dark class is
  score-clean; the catalogue-hosted class carries the whole defect signature; 15–24 % rail mass.

## 4. Consequence for Branch G (r-completion-residual)
F is delivered (11.44 no_bh at N=200, i.e. σ_h,harness 0.0594 vs floor 0.00519) and is a
DEFECT-SIGNATURE-context F, not a COVERAGE-USABLE F. The registration author must treat F as a
disclosed context number and design the arm so its discrimination does not lean on coverage
validity (the arm's own g-closure carries it). Chair decision under the grant: authoring proceeds
tonight; launch stays behind d-completion-register and docket 2.2.

## Appendix — rail side (chair script output, verbatim)
S no_bh lo-rail 0 hi-rail 10 median MAP 0.765 mean MAP 0.7719
S with_bh lo-rail 0 hi-rail 14 median MAP 0.78 mean MAP 0.7802
T no_bh lo-rail 1 hi-rail 4 median MAP 0.76 mean MAP 0.7688
T with_bh lo-rail 0 hi-rail 6 median MAP 0.785 mean MAP 0.7788
All rail mass sits at the UPPER grid edge h = 0.86 (one T no_bh universe at 0.60). The harness
universe's truth is h = 0.73; the mean MAP of 0.77–0.78 is a +0.04–0.05 offset with 15–24 % of
posteriors pinned at the ceiling — the coverage numbers are therefore upper-rail-censored bounds.

## ERRATA (end-verification, 2026-09-03 ~22:10)
- D1: the harness's centering Z uses std(MAP)/√n; registration §2.1 defines SEM = σ̄_post/√n_U → Z =
  6.00/7.27 (S), 3.48/4.34 (T). Outcome unchanged (OUTSIDE), number corrected.
- D2: "floor(200) = 0.00518915" is in fact floor(180) (median n_scored per universe); floor(200) =
  0.004923 → F = 12.06/12.00 (S). F is "at the realised median n_scored ≈ 180", not "at N=200". Label
  corrected; the [1,25] sanity band is unaffected.
- D7: the "pre-flip" ladder cell T reference is POST-flip (stamped 6c43f8f9 after 5e7fda16); the
  20/20 T byte-identity is a same-code rerun, not a flip-invariance check. The S pin (63/63 vs the
  pre-flip ladder S) stands. The chair's claim that the pin "DISCHARGES the deferred row #291
  g-byte-id criterion" is withdrawn to the registration's own wording: PROPOSED, routed to d-s4-review.
- D8: no checkpoint carries a `catalogue_leg_1d_mass_aware` token, and the S stamps span 13 commits —
  g-population is green on seed-block/N purity but only half-checked on the registration's
  population-identity token; disclosed for d-calibration.
