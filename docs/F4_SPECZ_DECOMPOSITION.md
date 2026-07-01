# F4 — spec-z vs photo-z host decomposition of the stacked H₀ posterior

**Figure:** `docs/figures/f4_specz_decomposition.pdf` (PNG gitignored)
**Data:** `scripts/bridge_closure/outputs/f4_specz_decomposition.json` (literal 1-D channel, 3361 events),
`scripts/bridge_closure/outputs/f4_specz_decomposition_conv.json` (σ_z-aware sky channel, 40-event demo)
**Renderer:** `scripts/bridge_closure/f4_plot.py`
**Companions:** `docs/SIGMA_Z_SIGMA_M_FORECAST.md`, `docs/BIAS_RESOLUTION_ATTEMPTS_REPORT.md`, `docs/H0_BIAS_RESOLUTION.md`

> **Headline.** The intended "money figure" hypothesis was tested on the real seed-600
> detections + real GLADE+ and **refuted**: there is *no* spectroscopic-host subset that
> carries the informative shape of the stacked H₀ posterior. Spectroscopic hosts are
> 0.56% of GLADE+ and are outnumbered ~180:1, so even when a spec-z host sits in a LISA
> localisation cone it carries ≤ 8.7% (median ~0%) of the rate-weighted in-catalogue
> likelihood — never the majority. The figure is therefore the **inverse** proof of
> photo-z information starvation: spec-z presence does not rescue any single-event
> posterior, and the stack rails regardless.

---

## 1. What the figure was supposed to show (the hypothesis)

The cleanest imaginable visual proof of photo-z information starvation would be:

> The informative shape of the stacked dark-siren H₀ posterior is carried **entirely** by
> events whose sky-localisation/redshift box contains a **spectroscopic** host
> (GLADE flag = 3, σ_z ≈ 0.0017), while events whose box holds only **photometric** hosts
> (flag = 1, σ_z ≈ 0.035) give flat/railing single-event posteriors.

If true, panel B would show the *all-events* and *spec-z-subset* stacks coinciding and
peaking at truth, while the *photo-z-only* stack stays flat. This document records that
this is **not** what the data show, and why.

## 2. Method

### 2.1 Flag retention (schema change, Phase 1)

GLADE+ encodes the redshift provenance per galaxy (Dálya et al. 2022, arXiv:2110.06184):
flag 1 = photometric z, flag 2 = luminosity-distance-only (excluded), flag 3 =
spectroscopic z. The reduced catalogue previously dropped this column; Phase 1 retains it
as `InternalCatalogColumns.REDSHIFT_FLAG` so every candidate host can be classified as
spec-z or photo-z. Catalogue composition after the finite-`z`/`M` mask:

| class | flag | count | fraction |
|---|---|---:|---:|
| photometric | 1 | 9,009,634 | 99.44% |
| spectroscopic | 3 | 50,383 | **0.56%** |
| total | — | 9,060,017 | 100% |

### 2.2 Two channels, two classifiers

The decomposition was run two ways to close off every escape route. Both reuse the real
seed-600 detections (SNR ≥ 20, rel-dist-err < 0.10 cuts), the real selection precomputes
(`precompute_completion_denominator` / `precompute_missing_completion_denominator` /
`precompute_global_catalog_selection`), and the real pixelated completeness. Nothing in the
H₀ computation is modified — this is an additive, read-only analysis.

**(a) Literal 1-D channel** — `f4_specz_decomposition.py`, all 3361 events.
Single-event posteriors from `_bridge_lib.event_log_likelihood` (the partition-norm
likelihood used by `run_bridge(catalog='real', events='real')`), computed on the pipeline
grid `h ∈ [0.60, 0.87]` step 0.01. An event is **`specz_hosted`** if *any* catalogue galaxy
in its H₀-independent ±5σ d_L candidate window (the same window that feeds `L_cat`) is
spectroscopic, else **`photoz_only`**. A verified fast path reproduces
`event_log_likelihood` bit-identically (max |Δ log-post| = 1.6 × 10⁻¹³).

**(b) σ_z-aware sky channel** — `f4_specz_decomposition_conv.py`, 40-event demo (conv is
~8.7 s/event; the subset already settles the question). Single-event posteriors from
`_bridge_sky.event_loglik_sky(mode="conv")`, which prunes candidates to the LISA Fisher
cone and **convolves each host by its redshift PDF** `norm(z; z_g, σ_z)` — the mechanism
where photo-z vs spec-z actually differs. An event is **`specz_dominated`** if
spectroscopic hosts carry ≥ 50% of the σ_z-broadened, rate-weighted candidate contribution
near the GW distance, else **`photoz_dominated`**; spec-z *presence* in the cone is recorded
separately.

## 3. Result

### 3.1 The literal classifier is degenerate

The H₀-independent ±5σ d_L box holds 15k–2.7M galaxies (median 742k) and therefore
**always** contains thousands of spectroscopic hosts. Every one of the 3361 events is
tagged `specz_hosted`; the `photoz_only` subset is **empty**.

| quantity | all | spec-z | photo-z |
|---|---:|---:|---:|
| event count | 3361 | 3361 | **0** |
| peaked fraction | 0.401 | 0.401 | — |
| stacked MAP (h) | 0.822 | 0.822 | — |

Second structural blocker: `event_log_likelihood` treats each catalogue redshift as
*exact* (`norm.pdf(dist(z,h), d_meas, σ_dL)`), i.e. it is **σ_z-blind** — the redshift flag
never enters the likelihood, so per-event peaked/railed cannot depend on host-z provenance.

### 3.2 The σ_z-aware channel makes presence non-degenerate — and still refutes it

Cone pruning makes spec-z presence a real split, but domination never occurs:

| quantity | value |
|---|---:|
| events | 40 |
| spec-z **present** in cone | **24 / 40** |
| spec-z **dominated** (≥ 50% weight) | **0 / 40** |
| spec-z weight fraction | median ~0%, mean 0.7%, **max 8.7%** |
| single-event peaked (conv) | **0 / 40** (all rail) |
| stacked MAP: all / spec-z-present / photo-z-only (h) | **0.87 / 0.87 / 0.87** |

Truth is `h = 0.73`. All three stacks rail to the upper grid edge together.

### 3.3 What each panel shows

- **(A)** the 40 single-event posteriors, coloured by spec-z presence (blue = present,
  grey = photo-z only). Every posterior is a monotone ramp to a grid edge; the blue and
  grey curves are interleaved — **spec-z presence does not produce a peak at truth**.
- **(B)** the stacked posterior split three ways (all / spec-z-present cone / photo-z-only
  cone). All three coincide in railing to `h = 0.87` — the spec-z-present subset does **not**
  recover the informative shape.
- **(C)** the smoking gun: the per-event spectroscopic likelihood-weight fraction, sorted.
  The maximum over all 40 events is 8.7%, far below the 50% domination threshold; most events
  are at ~0%. Spectroscopic hosts simply never carry the in-catalogue term.

## 4. Conclusion

The premise that a spec-z-hosted subset carries the informative shape **does not hold**, and
the reason is structural rather than statistical: spectroscopic hosts are 0.56% of GLADE+ and
are outnumbered ≈ 180:1 by photometric hosts, so even when present in a localisation cone they
never carry the majority of the rate-weighted in-catalogue likelihood. The photo-z information
starvation is therefore **total** — photometric hosts dominate every event's in-catalogue
term, the σ_z ≈ 0.035 convolution (≈ 14× the GW distance precision) washes out the sharp GW
distance information, and the stack rails to the grid edge regardless of spec-z presence.

**Implication for the paper.** F4 cannot be the "spec-z carries it" figure. The honest money
figure is the inverse rendered here: spec-z hosts are *present* in 60% of cones yet contribute
≤ 8.7% (median ~0%) of the likelihood weight, so the stack rails regardless — the cleanest
visual statement that photo-z information starvation is complete at the GLADE+ spec-z fraction.
This is consistent with the σ_z/σ_M feasibility forecast (`docs/SIGMA_Z_SIGMA_M_FORECAST.md`),
which shows the informative regime requires host-redshift precision far beyond what the GLADE+
spectroscopic sub-sample provides at our detected-event redshifts, and with the railing
diagnosis in `docs/BIAS_RESOLUTION_ATTEMPTS_REPORT.md` / `docs/H0_BIAS_RESOLUTION.md`.

## 5. Reproduce

```bash
# decomposition data (Phase 2; already produced)
uv run python scripts/bridge_closure/f4_specz_decomposition.py            # literal, 3361 events
uv run python scripts/bridge_closure/f4_specz_decomposition_conv.py 40    # conv, 40-event demo

# figure (Phase 3)
uv run python scripts/bridge_closure/f4_plot.py
# -> docs/figures/f4_specz_decomposition.pdf (+ .png, gitignored)
```
