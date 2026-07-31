# Cell B readout — the 2×2 decisive test (2026-07-31)

Pre-registration: `PREREGISTRATION_2x2_cellB.md` (written before submission).
Run: `$WS/run_20260729_seed61000/estimatorB_2x2/`, evaluate 6103219 + combine
6103220 (resubmission of 6101146/6101147 — those failed on a missing raw-CRB
symlink in the run-dir setup, a pure plumbing error fixed without touching the
test design; pre-registration unchanged). Config verified from
`run_metadata_0.json`: `absolute_marginal` + `volume_deconv` +
`host_mass_kernel auto`, `observed_catalogue: null` (the unscattered parent),
code at `7fd60bb` — the same commit as cells A and C, same CRB, same injection
pool. Readout by `cellb_readout.py` (this directory); artifacts pulled to
`seed61000/estimatorB_2x2/`.

## The 2×2, filled

|             | point / generator_marginal | volume_deconv / absolute_marginal |
|-------------|----------------------------|-----------------------------------|
| unscattered | A = #51: 1D 0.7299, 2D 0.7300 | **B: 1D 0.7450, 2D 0.7900** |
| scattered   | forbidden by guard         | C = #53 r1: 1D 0.7400, 2D 0.8133  |

| effect | 1D | 2D |
|---|---|---|
| **B − A = estimator** | **+0.0151** | **+0.0600** |
| **C − B = scatter**   | −0.0050 | +0.0233 |
| C − A = total (r1)    | +0.0101 | +0.0833 |

2D: B's posterior is interior (edge/peak 1.2e-2), σ_h 0.019, mean 0.7962.
1D: MAP 0.7450, mean 0.7320, σ_h 0.026.

## Verdict: pre-registered outcome 1 — **THE ESTIMATOR OWNS IT**

The joint prediction registered in `ADJUDICATION_20260730.md` §3 (B ≈ C: 2D in
0.78–0.82, in-cat class argmax 0.86, 1D an interior crossing) is **confirmed on
every pre-registered read**:

1. **2D**: B = 0.7900 with exact host redshifts — 72% of the total r1 2D
   displacement (+0.060 of +0.083) is the estimator configuration alone. The
   realized scatter adds +0.023 (28%) — incidentally the same scale as the old
   seed600-era "2D +0.025 residual".
2. **The in-cat rail needs no scatter.** Per-event 1D argmax at the 0.86 edge:
   **69.7% in B** (53/76) vs 57.9% in C vs 5.3% under the #51 estimator. The
   catalogue-leg-only rail (the adjudication's independent C7 adjudicator):
   **90.7% in B vs 89.2% in C** — statistically identical. The rail is the
   `volume_deconv` kernel integrating the catalogue's `z_error` *column*
   (present in the unscattered parent, and here the TRUE cluster parent — the
   staleness-free confirmation the C7 verdict was waiting for). Realized
   scatter is not the cause; if anything it slightly *damps* the combined-leg
   rail (69.7% → 57.9%).
3. **Class structure transfers.** B's class-summed argmaxes: 1D in-cat 0.860 /
   dark 0.640 (the two runaways), 2D dark 0.800 (the de-weighting collapse) —
   identical structure to every #53 run. The per-class channel difference:
   dark **+18.00** nats in B vs +15.83 in C-r1 (in-cat −1.80 vs +2.97; recall
   the in-cat component is realization-noisy, sign-flipping in r3). The dark
   mass-de-weighting channel difference is **estimator-borne, not
   scatter-borne** — confirming what the sig0_control's estimand-confounded
   hint suggested.
4. **w_G(h) is bit-identical to the #53 curve** (0.1625175 / 0.1215039 /
   0.1038732 at h = 0.60/0.73/0.81) — the pre-registered pure-quadrature
   prediction holds exactly, so C9's transfer arithmetic applies to B verbatim.

## Consequences (per the pre-registered outcome-1 clause)

- **The realistic host-observation model (the #53 scatter layer) is largely
  exonerated for the headline biases.** R1–R9 realism needs no revisiting for
  the bias question; its measured incremental effect is +0.023 in the 2D MAP
  and −0.005 in 1D.
- **The fix surface is exactly the estimator configuration**, as the
  adjudication's three convictions said: C7 (the host-z kernel's selection-free
  population weight — now confirmed against the true parent widths), C9 (the
  mass-blind w_G, present and bit-identical in B), and C8 (the completion leg's
  missing mass density). The joint mass-consistent-mixture derivation
  (ADJUDICATION §5 item 6, `/physics-change`, author-gated) plus the C7 kernel
  fix (§5 item 5, must supersede G2b) are the whole fix program.
- **C6 is resolved**: attribution is no longer confounded. The claim file's C6
  gets a dated resolution block pointing here.
- Cell B's diagnostics CSV also provides the first **unscattered** C4
  partition — the dark channel difference (+18.0 nats) exists without any
  realized noise, closing the "is the dark de-weighting scatter-induced?"
  question definitively: it is not.

## Next steps (concluded, in order)

1. **Author decisions now unblocked** (the two `/physics-change` gates):
   (a) the joint C9+C8 mass-consistent mixture derivation — cell B removes its
   last external gate; the remaining input is the author's leg-adjudication,
   for which the evidence now reads: three measured convictions on the
   completion/prefactor/kernel side, zero on the catalogue side, and the
   realism layer exonerated. (b) the C7 kernel fix derivation — mechanism +
   staleness-free magnitude both confirmed; the derivation must explicitly
   supersede G2b and must not be the historically-exonerated "p_det inside the
   numerator alone" form.
2. **Campaign policy**: do not relaunch multi-seed production on the current
   estimator; truth-seed GPU sims (scoping memo, wave 1 incl. finishing
   seed 63000) are estimator-independent and can run in parallel with the
   derivation work, evaluates deferred until the fixed estimator exists.
3. **Paper #47 stays on hold** until a post-fix trusted run exists (unchanged).
