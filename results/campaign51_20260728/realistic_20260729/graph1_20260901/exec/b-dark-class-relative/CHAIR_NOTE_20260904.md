# b-dark-class-relative — chair note (2026-09-04 ~00:40)

Build done (BUILD_RECORD.md). Chair decision (flagged): the MIGRATION of the 24 `== 0` call sites is
HELD. The build shows the question is definitional, not numerical:

| file (h = 0.73) | exact-zero dark/hosted | relative (<1e-6) dark/hosted | differ |
|---|---|---|---|
| 2026-08-27 head readout (pre-flip) | 606 / 982 | 727 / 861 | 121 |
| S0-B truth node (mass_aware off) | 449 / 1139 | 723 / 865 | 274 |
| 2026-09-02 re-baseline iiib (post-flip auto) | 606 / 982 | 1241 / 347 | 635 |
| 2026-09-02 re-baseline joint_r1 | 493 / 1095 | 967 / 621 | 474 |

- The relative label stabilizes the S0-B-vs-08-27 flip (Δ = 4, down from 157) — the R8 purpose.
- But on the post-flip production re-baseline, 635 events counted as "catalogue-hosted" by the exact
  criterion carry a catalogue-leg weight below 1e-6 of the combined likelihood. Every A12
  score-by-class read (incl. row #335's "the S3 defect localizes to the catalogue-hosted class") is
  conditioned on the exact-zero label; under a materiality label the "hosted" class is 347 events.
- Threshold 1e-6 is a margin call (max moved-event ratio 9.75e-7; the ratio distribution is
  continuous, no natural gap) — not a physical boundary.

Returns to the author as [RULE] R14: what "catalogue-hosted" MEANS for class-conditioned reads
(exact-zero support vs a materiality threshold vs a per-event catalogue-leg fraction as a continuous
covariate). The r-offset-subset registration (batch 2) is instructed to carry BOTH labels and the
continuous fraction as covariates, so the question gets a measurement rather than a convention.

## RULING R14 (ratified 2026-09-04)

f_cat = L_cat_no_bh/combined_no_bh is registered as a continuous covariate in every new
class-conditioned read; the exact-zero label is kept for backward comparability with its fragility
disclosed; the 24-site migration is NOT performed.
