# Bridge-the-Closure Investigation — H₀ Railing Root Cause

**Result (root cause): the H₀ posterior rails to the upper grid edge because the
GLADE host-galaxy redshift error σ_z ≈ 0.035 is ~10–18× larger than the GW
redshift precision (σ_z^GW ≈ 0.037·z ≈ 0.002 at z≈0.05). The in-catalogue
likelihood marginalises this photo-z (convolves each host's redshift PDF), which
washes out the sharp GW distance information, so the catalogue redshift-density
gradient — not the GW measurement — drives H₀, biasing it to the edge. The
closure test recovered H₀ only because it assumed spectroscopic hosts
(σ_z = 8×10⁻⁴).** This is the closure↔pipeline gap (railing-handoff suspect #6),
now confirmed and reproduced.

**Method.** Start from the closure (recovers 0.73) and add ONE real-pipeline
ingredient at a time ("rungs"), re-running the same partition-norm MAP recovery,
until the railing appears. Harness: `scripts/bridge_closure/_bridge_lib.py` (+
`_bridge_sky.py`). Every rung uses the SAME real production functions the closure
used (`precompute_completion_denominator`, `precompute_missing_completion_denominator`,
`precompute_global_catalog_selection`), so only the ingredient under test changes.

---

## Dataset
- **Railing run:** seed-600, self-consistent dark-event CRBs (`/tmp/seed600_local/
  simulations/`). 3375 detections → **3361** after the inference cuts (SNR≥20,
  σ_dL/d_L<0.10). True H₀/100 = 0.73. Cluster MAP = **0.86** (grid edge, +0.13).
- **Catalogue:** `reduced_galaxy_catalogue.csv`, 2,197,704 galaxies, median z 0.082.

---

## Reframing (established up front, with code evidence)
1. **Classic Malmquist is impossible** — SNR≥20 is cut on the OPTIMAL (true) SNR
   (`parameter_estimation.py:455`, `main.py:539`); measured d_L is written
   downstream of selection. P0: measured-vs-true scatter is unbiased (mean +0.0007).
2. **99.2 % of detections are in-catalogue** (nearby; median z≈0.046) → the bias
   is in the in-catalogue term `β_G·L_cat/D(h)`, not the completion/dark channel.

---

## Rung ladder (each row = one ingredient swapped synthetic→real)

| Rung | Ingredient | MAP | bias | rails? | verdict |
|------|-----------|-----|------|--------|---------|
| **R0** | synthetic baseline (median/seeds) | 0.735 | +0.005 | no | harness reproduces the closure ✓ |
| **A** | real σ_dL distribution, N=3361 | 0.729 | −0.001 | no | **measurement side does NOT rail** |
| **B** | real GLADE n(z) shape (no sky, 1-D) | 0.734 | +0.004 | no | **density shape alone does NOT rail** |
| **C-real** | real catalogue + sky + 3-D MVN (full Fisher cov) | 0.725 | −0.005 | no | faithful in-cat channel recovers |
| **C-iso** | same, but galaxy sky positions SHUFFLED | 0.855 | +0.125 | **yes** | broken host↔sky matching rails |
| **D** | C-real + real pixelated f_k + B_num | 0.735 | +0.005 | no | completion does NOT rail |
| **E** | C-real + real survival p_det | 0.725 | −0.005 | no | selection p_det does NOT rail |
| **F** | C-real + real p_det + real f_k + B_num (fully faithful) | 0.735 | +0.005 | no | even fully faithful recovers (delta-z) |
| radius sweep | tight 1.5σ candidate ball (drops 15.6% of hosts) | 0.73 | ~0 | no | host-dropping alone does NOT rail |
| **G (delta-z)** | F + 1.5σ radius, exact host z | 0.725 | −0.005 | no | — |
| **G (photo-z, σ_z×1.0)** | F + 1.5σ radius + **host-z convolution at real σ_z=0.035** | **0.857** | **+0.127** | **yes** | **REPRODUCES the real pipeline (0.86)** |

### What it is NOT (eliminated)
Measurement σ²/distance scatter; high-N amplification; the σ-distribution; the
catalogue n(z) density shape; the sky dimension / 3-D MVN / Fisher correlations;
the candidate-selection radius/frame (the real ball-tree returns the true host for
**99 %** of events, median |z_cand−z_true|=0.000); the completion term B_num; the
real survival p_det; the pixelated completeness f_k. **Each of these recovers 0.73
when added alone.** The ONLY ingredients that rail are (i) actively-wrong
host↔sky matching (C-iso) and (ii) the **host-redshift photo-z convolution** (G).

### The decisive ingredient — host-redshift photo-z (Rung G)
The real `single_host_likelihood` convolves each candidate's GW contribution with
`norm(z_g, σ_z_g)`. Sweeping σ_z (scaling the catalogue z-error), fully faithful:

| σ_z | MAP | bias |
|-----|-----|------|
| 0 (delta-z) | 0.725 | −0.005 |
| ≈0.002 (×0.05, spec-z) | 0.600 | −0.130 |
| ≈0.009 (×0.25) | 0.870 | +0.140 |
| ≈0.018 (×0.5) | 0.870 | +0.140 |
| **≈0.035 (×1.0, REAL GLADE)** | **0.857** | **+0.127** |

The bias is a sensitive function of σ_z (a density-gradient effect, sign included),
and at the **real** σ_z it rails to +0.13 — matching the pipeline. (`outputs/rungG_photoz.pdf`.)

### σ_z ≈ 0.035 = including PHOTOMETRIC redshifts (GLADE flag misuse)
The column **indices are all correct** (verified vs Dálya+ 2022, arXiv:2110.06184:
0-based col 27=z_helio, 30=pec-vel z-error, 31=z measurement error, 34=measurement
flag). But the parse (`handler.py:284`) keeps `flag ∈ {1,3}` with the comment
*"1, 3 are measured redshifts"* — **wrong**: GLADE's measurement flag is
**1 = photometric, 3 = spectroscopic, 2 = luminosity distance**. σ_z split by flag:

| flag | meaning | σ_z median | σ_z 90th |
|------|---------|-----------|----------|
| 1 | photometric | **0.0346** | 0.0482 |
| 3 | spectroscopic | **0.0017** | 0.0036 |

So the host catalogue is dominated by **photometric** redshifts (the large-σ_z
hosts that rail). This is a data-usage finding, not a parse/index bug.

---

## Mechanism (and the simplification it implicates)
With σ_z ≫ σ_z^GW, the in-catalogue numerator `N_g = ∫ p_GW(d_L(z,h)) norm(z;z_g,σ_z) dz`
is dominated by the broad host-z PDF, so the candidate sum tracks the catalogue
redshift-density `n(z)` over the σ_z window rather than the sharp GW distance.
Crucially the **selection denominator** (`precompute_global_catalog_selection`,
`D(h)`) uses the *narrow* galaxy-redshift-PDF limit (`D_g ≈ p_det(z_g)`, NO
convolution — a deliberately documented partition-norm simplification). The
**numerator marginalises the photo-z but the denominator does not**; this
asymmetry is negligible for spec-z (closure) but biases H₀ strongly for GLADE
photo-z. ⚑ This implicates the documented narrow-window approximation — flagged
for a decision, not silently changed.

---

## Fix options (bridge-tested where noted)
1. **Spectroscopic hosts only** (`flag==3`): **FAILS as a pure inference-side
   filter** — bridge-tested, σ_z<0.005 catalogue (32,849 gals) still rails to
   0.870, because events were injected from the photo-z-dominated catalogue so
   the true (photo-z) hosts are removed → host mismatch (sim↔inference
   inconsistency). A self-consistent spec-z pipeline needs RE-INJECTION from
   spec-z hosts (loses most events). NB delta-z on the full catalogue recovers
   (0.725): if the photo-z hosts *had* exact z, it works.
2. **Photo-z-consistent selection normalisation** (the chosen direction): the bias
   is that the in-cat numerator marginalises the photo-z (→ smoothed catalogue
   density n_smooth) but the selection denominator uses the un-smoothed n(z). The
   density-gradient bias is intrinsic to catalogue dark sirens with photo-z +
   incompleteness and is NOT removed by naively convolving the global denominator
   (p_det varies slowly, so that convolution is ≈ a no-op). Requires a proper
   re-derivation of the partition-norm in-cat likelihood with photo-z — a physics
   task (GPD), prototype in the bridge first.
3. **Sim-mock forecast arm** (project's hybrid framing): the forecast/methodology
   arm uses simulated spec-z hosts → recovers; the railing is a GLADE-real-data
   issue. Strongest near-term path for the paper's forecast result.
4. **Report the limitation**: catalogue dark sirens with GLADE photo-z at z≈0.05
   are density-gradient-dominated; quote a caveated posterior.

---

## Reproducibility
```bash
uv run python scripts/bridge_closure/rung_A_measurement.py    # P0 + sigma ladder
uv run python scripts/bridge_closure/rung_B_catalog.py        # real GLADE n(z)
uv run python scripts/bridge_closure/rung_C_sky.py 1500       # sky + MVN ablations (incl. C-iso)
uv run python scripts/bridge_closure/rung_D_completion_pdet.py 1500   # B_num + real p_det
uv run python scripts/bridge_closure/rung_E_radius.py 2500    # candidate-radius sweep
uv run python scripts/bridge_closure/rung_F_combined.py 2500  # radius + completion
uv run python scripts/bridge_closure/rung_G_photoz.py 1200    # photo-z sigma_z sweep (ROOT CAUSE)
```
Outputs (JSON + paper PDFs) in `scripts/bridge_closure/outputs/`. All randomness
seeded; real CRBs/catalogue from `/tmp/seed600_local/` and
`darksiren_emri/galaxy_catalogue/`.

### Figures
- `P0_event_characterization.pdf` — measurement is unbiased; nearby/in-catalogue.
- `rungA_sigma_ladder.pdf` — measurement side does not rail.
- `rungB_nz_comparison.pdf`, `rungB_posteriors.pdf` — n(z) shape does not rail.
- `rungC_posteriors.pdf` — sky/MVN recover; only sky-shuffle (C-iso) rails.
- `rungD_posteriors.pdf` — completion + real p_det recover.
- `rungE_radius.pdf`, `rungF_combined.pdf` — candidate radius does not rail.
- `rungG_photoz.pdf` — **photo-z σ_z reproduces the +0.13 rail; delta-z recovers.**
