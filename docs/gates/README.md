# Gate ledgers

Durable home of the scientific-soundness **gate ledgers** (G1–G11 and the Phase-1 sign-off).
These are the paper-facing records of what each gate checked, what it found, and how it was
closed — cited from `CLAUDE.md`, `constants.py`, `DATA_INVENTORY.md`, `docs/derivations/`, and
`paper_a/`.

## Why this directory exists

The ledgers originally lived in `.planning/gate/`. The 2026-07-16 hygiene split (`1fe428a`)
untracked the whole `.planning/` tree, which is also `.gitignore`d — so the ledgers were
untracked, then lost from the working tree entirely, while tracked files kept citing them. They
were recovered from `1fe428a^` and moved here on 2026-07-30.

**Rule: a gate ledger cited from a tracked file must live under `docs/gates/`, tracked.**
`.planning/` is local churn and cannot hold anything another file points at.

Older references of the form `.planning/gate/<name>` (in scripts' `--output_json` defaults and
in historical `results/`, `paper_a/` notes) refer to the same artifacts now at
`docs/gates/<name>`. Scripts that write gate JSON should have their output paths repointed the
next time they are touched.

## Contents

| File | Gate | What it records |
|---|---|---|
| `GATE_SIGNOFF.md` | — | Phase-1 scientific-soundness gate sign-off |
| `G1_beta_g_check.md` / `.json` | G1 | β_G discrete GLADE-sum verification |
| `G3_ablation_cube.json` | G3 | Ablation-cube results (volume kernel as primary de-railer) |
| `G5a_gwcosmo_inspection.md` | G5 | External-code inspection: gwcosmo |
| `G5b_chimera_icarogw_inspection.md` | G5 | External-code inspection: CHIMERA / icarogw |
| `G6_starvation_postmortem.md` | G6 | Estimator-starvation post-mortem |
| `G7_systematics_budget.md` | G7 | **Systematic-error budget** — 16-row paper-ready inventory |
| `G7row9_eddington_m_impact.json` (+ `_postDgfix`) | G7 | Row-9 Eddington-in-M impact runs |
| `G7row9_N5_postDgfix_SUMMARY.md` | G7 | Row-9 N=5 post-D_g-fix summary |
| `G8_inner_product_finding.md` | G8 | Missing dt² DFT normalization in the inner product |
| `G9_timeout_scan.md` | G9 | Waveform-timeout selection scan |
| `mass_trunc_ab.json` · `volume_trunc_ab.json` | — | A/B results for the mass/volume truncation kernels |

## Related

- `PHYSICS-GATE-LEDGER.md` — the per-change compliance ledger for the `/physics-change` hard
  gate (started 2026-07-30). Different purpose: gate ledgers above record *findings*; the
  physics-gate ledger records *that the protocol was run*.
