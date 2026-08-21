# Campaign Readout Report — C-SG v3 (the self-generated positive control)

**Date:** 2026-08-21 (overnight autonomous session) · **Jobs:** 6415588 (pilot) + 6420343 (fleet),
46/46 COMPLETED · **Registration:** `PREREGISTRATION_SELFGEN_CONTROL.md` v2 + v3 design change +
appended band/gate blocks (all pre-data) · **Ledger:** rows #149, #150, #151 ·
**Scorers:** all committed before their data existed.
**Binding rule:** this report presents the fired branches; it does not adjudicate. Every verdict
marked [RULE] awaits the author.

## 1. The goal — what was actually being asked

One question, carried since row #140: **is B-SEL's −0.11 an internal defect of the estimator, or
an artifact of how B-SEL generated its events?** The harness had never had a control that could
fail (the old B-F1 "control" was a flat posterior whose "0.7300" was the grid midpoint). C-SG is
that control: generate events from *the estimator's own model*, cleanly, and ask whether the
estimator recovers its own truth.

## 2. What changed before any CPU was spent (the two free reads)

Two zero-compute decompositions of the 12 banked B-SEL seeds reshaped the design:

- **O2 (row #149):** the per-event likelihood splits exactly as
  `combined = (β_G·L_cat + B_num)/D̃`. Zeroing the catalogue leg (`L_cat ≡ 0`) moves the fleet
  from −0.1083 to −0.0291 — **the impostor catalogue leg carries 73%** of the headline bias
  (12/12 seeds, independently recomputed to 10 decimals). In-cone *wrong* galaxies — the true
  hosts are never in the catalogue — sit at low z and drag h down.
- **O3 (row #150):** the mixture normalization splits as `D̃ = α (catalogue) + β_Ḡ (dark)`. For
  events *known* to be dark, the matched conditional is `B_num/β_Ḡ` — and it is biased
  **−0.0846 ± 0.0095**. O2's mild −0.029 was an accidental cancellation: matched −0.085 ⊕
  mixture-tilt +0.055 ⊕ impostor −0.079 = full −0.108. The catalogue-sector conditional (b0,
  exploratory) tilts the *opposite* way (+0.040).

Consequence (registered trigger): C-SG's verdict statistic moved to the **matched channel** — the
full mixture would have swamped any internal-defect signal with the now-quantified impostor
mismatch.

## 3. The design — arms and the control's own controls

Generator (design B, unchanged from v2): per event, draw (z, Ω) **jointly** ∝ `w_pop(z)·(1−f_k)`
at the event's own pixel; mass from the estimator's own φ; select **once** with `S_4D`; observe
**linearly** (`d̂ = d_L + σ·ε`). This removes every generator-side caveat B-SEL carried (sky-
marginal f̄, donor-row borrowing of z-linked quantities, σ from SNR-weighted donors, quality
filter). Arms: **F** (σ = 0.0373·d_L, 15 seeds), **E** (σ empirical i.i.d., 15), **δ−** (h_gen =
0.68, 8), **δ+** (0.78, 8). Implementation: 2-lens adversarial verification GO; gates H (h_gen
threading), Q (43% Fisher-non-PD before the mandated cross-covariance rescale, 0.0% after), D
(draw matches the model density at all three h_gen) all passed pre-run.

**The pilot did its job twice.** The registered 4-seed pilot STOPped the fleet: GATE V (ported
unchanged from the v2 full-channel numbers) false-failed 3/4 pilot seeds — and, decisively, 5/12
*banked B-SEL* matched posteriors, which are known-informative. The gate was re-derived against
that independent reference (flat-null signature: span ≥ 1 nat, σ_h ≤ 0.9·σ_prior; reference
false-fail 0/16; the historical flat mode still fails both prongs), the bands were frozen from the
pilot's σ̂ by the pre-registered formulas, and only then did the fleet launch. (Retrospective
ledger entry 1 records the porting slip as the orchestrator's own A15-class miss.)

## 4. The result

| arm | h_gen | n | matched bias | matched score at h_gen | full-channel bias | pure bias |
|---|---|---|---|---|---|---|
| F  | 0.73 | 15 | **−0.0665** | **−0.1173** | −0.1090 | +0.0115 |
| E  | 0.73 | 15 | −0.0667 | −0.1184 | −0.1081 | +0.0107 |
| δ− | 0.68 | 8  | −0.0863 | −0.1332 | −0.1099 | −0.0129 |
| δ+ | 0.78 | 8  | −0.0495 | −0.1131 | −0.1044 | +0.0281 |

**BAND C = INTERNAL-DEFECT on both frozen statistics** (score −0.1173 vs edge −0.0966; bias
−0.0665 vs edge −0.0423). Three structural facts carry the reading:

1. **The score violation is h_gen-independent** (−0.113 … −0.133 everywhere, both σ modes,
   F-vs-E gap 0.0002). A generator artifact would move with the generator; this does not.
2. **The full channel lands on the campaign's headline number in every arm** (−0.104 … −0.110 vs
   B-SEL's −0.1083): the O2/O3 three-channel structure transfers quantitatively to the clean
   generator. The "−0.11" the campaign has chased since row #137 is now *reconstructed from
   first principles*: a completion-leg conditional violation, a dark-fraction normalization tilt,
   and an impostor drag.
3. **The pure channel is near zero in all arms** — nobody looking only at `B_num/D̃` would see the
   defect; the cancellation O3 found is generic, not a B-SEL fluke.

## 5. The gate that fired anyway

**GATE S returned CONTROL-INERT-STOP by its registered letter** (ŝ = 0.368 ± 0.186; 3·SE brackets
0). The qualification, recorded beside it: the arm means are strictly *ordered* in h_gen
(0.6437 → 0.6635 → 0.6805), ŝ is ~2σ from 0 and 3.4σ *below 1* — the matched posterior responds
to the generating h, but at a third of the unit slope. Grid-edge truncation at h_gen = 0.68 works
*against* explaining this away. The primary score statistic contains no slope and is unaffected.
The sub-unit slope is itself an unexplained diagnostic worth a registered follow-up.

## 6. Scorecard vs locked bands

| check | locked band | outcome |
|---|---|---|
| BAND C (score, primary) | SC ≤ 0.0373 / DEFECT ≤ −0.0966 | **INTERNAL-DEFECT** (−0.1173) |
| BAND C (bias, confirmatory) | SC ≤ 0.0209 / DEFECT ≤ −0.0423 | **INTERNAL-DEFECT** (−0.0665) |
| GATE V (amended, all 46) | span ≥ 1 nat, σ_h ≤ 0.9σ_prior | 46/46 PASS |
| BAND R (σ-mode) | gap ≤ 0.0296 | CONSISTENT (0.0002) |
| GATE S (slope) | VALID if \|ŝ−1\| ≤ 3SE; INERT if CI ∋ 0 | **CONTROL-INERT-STOP** (qualified, §5) |
| N-adequacy (A15) | half-reference ≥ 5σ | PASS (7.8σ) |

## 7. Vocabulary

- **Matched channel** — `B_num/β_Ḡ`: the estimator's completion numerator over its own
  dark-sector normalization; the correct conditional likelihood for an event known to be dark.
- **Impostor leg** — catalogue-sector weight collected by in-cone galaxies that are not the true
  host (the true host is not in the catalogue at all).
- **Score at h_gen** — per-event `∂_h ln L` at the generating h; zero in expectation for a
  self-consistent estimator fed its own model's data.
- **INTERNAL-DEFECT** — the registered branch name; it asserts the violation survives a clean
  self-generated draw, i.e. it lives in the estimator's own mathematics, conditional on §7's six
  shared invariants (`w_pop`, `f_k/f̄`, `S̄_φ/S_4D`, `P_det`, cosmology, z-domain).

## 8. Why the numbers stand

Scorers committed before their data (O2, O3, band-setter, fleet readout); bands frozen from the
pilot by pre-registered formulas with a published false-fail table; the O2 headline independently
recomputed by a firewalled agent; fleet numbers re-derived by the orchestrator from raw
diagnostics (seed-level bit-match); 46 JSONs + 46 per-event CSVs banked with a SHA-256 manifest so
every channel is recomputable at zero compute; the superseded GATE V verdicts remain in the banked
JSONs so the fired STOP is reproducible.

## 9. Flags the report carries anyway

- GATE S's INERT letter vs its ordered-means reality (§5) — needs an author reading.
- σ_prior's numeric convention in GATE V was invented at implementation time (flagged in every
  gate_v dict); the amendment's thresholds derive from reference data, not from that convention.
- The 16-cpu single-process over-reservation (house convention; ≈550 reserved vs ≈35 consumed
  core-h) — A6 perf-audit item.
- Stage-L (2026-08-21): our 4.79% in-catalogue venue is *below Gray 2020's own validated
  completeness floor* (25%), and no published work decomposes a mixture posterior per sector —
  the O2/O3/C-SG decomposition chain appears to be literature-novel.

## 10. The decisions (all [RULE], author)

1. **Ratify BAND C = INTERNAL-DEFECT** → row #140 promoted to a banked estimator-defect claim
   (conditional on the six named invariants).
2. **Rule on GATE S**: does CONTROL-INERT void the mean_h confirmatory band (score primary stands
   either way), and is the sub-unit slope registered as its own follow-up thread?
3. **Re-grade rows #137/#140/#144** per O2/O3 (row #149 items 3–4): the "pure completion carries
   it" language and the ≥0.073 internal-residual bound are both contradicted by measurement.
4. **Open the fix fork** (carried decision #2): designated first step = independent audit of
   `S̄_φ` (`bayesian_statistics.py:1932-1975`) — the never-audited object that builds the exact
   normalization the matched channel divides by.
5. **A17 (proposed)**: gates moved across channels/statistics re-derive operating characteristics
   against reference data in the same commit.
6. Landscape/T1 un-gate (carried decision #3) — now that C-SG has returned, the chain
   C-SG → B-SEL verdict → fix fork → landscape is at its second link.

## 11. Provenance footer

Commits this campaign (all pushed): 2b9cf0c6 (A16 instruments) · 9d91ecf8 (O2 scorer pre-data) ·
[O2 verdict + O3 registration] · [O3 verdict + rows #149/#150 + v3 design] · a80ce4b2 (Stage-L) ·
7ab5f001 (C-SG v3 implementation) · e5bd5bf0 (pilot sbatch) · dae957d6 (band formulas + fleet
sbatch) · 3b43732a (GATE V amendment + frozen bands) · 3d385152 (fleet readout scorer) · this
commit (fleet verdict). Cluster: bwUniCluster jobs 6415588, 6420343; workspace `emri` (expires
2026-09-23). Data: `csg_pilot_20260821/` with `MANIFEST.sha256` (92 files).
