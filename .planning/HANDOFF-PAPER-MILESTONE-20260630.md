# HANDOFF — Paper milestone: dark-siren H₀ feasibility (story + σ_z/σ_M heatmap)

**Created 2026-06-30. Start a FRESH session for this milestone (clean context).**
Suggested kickoff: `/gpd:new-milestone` (physics paper) or `/gsd:new-milestone`, then plan the tracks below.
This is the master handoff; the σ_z/σ_M heatmap implementation specifics live in the companion
`.planning/HANDOFF-SIGMAZ-SIGMAM-FORECAST-20260630.md` (read both).

---

## 0. Read order for the next session
1. This file (the plan).
2. `docs/BIAS_RESOLUTION_ATTEMPTS_REPORT.md` + `docs/PIPELINE_BUGS_REPORT.md` (what we found & tried).
3. `.planning/derivation-photoz-incatalog/INCREMENT3-DSM-VERDICT.md` (the decisive info-starvation result).
4. `docs/H0_BIAS_RESOLUTION.md` §3.18 (the photo-z chapter) + `docs/PIPELINE_FLOWCHART.md`.
5. `.planning/HANDOFF-SIGMAZ-SIGMAM-FORECAST-20260630.md` (heatmap method).
6. Memory: `photoz-comparison-premise-refuted`, `h0-railing-rootcause-photoz`, `inference-consistency-audit`.

## 1. The paper's story (the argument)
A honest, falsifiable arc:
1. **Goal:** unbiased H₀ from LISA EMRI dark sirens using the *entire* GLADE+ catalogue.
2. **Result with the current catalogue:** the in-catalogue **photometric** redshifts (σ_z≈0.035 ≈ 17×
   the GW precision, σ_z/z≈0.7 at z≈0.05) make the inference **largely uninformative** — the posterior
   does not peak; it rails/flattens. Demonstrated, not asserted: every normalization (standard,
   numerator cleans, local consistent-denominator, global same-kernel `D_sm`) was tested; `D_sm`
   removes the gradient bias but recovers no peak → **information-starvation**.
3. **Then open it up:** make σ_z (redshift error) and σ_M (host-mass/BH-mass error) free parameters and
   map **H₀ precision vs (σ_z, σ_M)** — the feasibility **heatmap**. Read off *what measurement
   precision makes LISA dark-siren cosmology useful, and where it is futile* ("know where to stop").
   Hypothesis to confirm: the **with-BH-mass (2-D) channel converges faster** (tolerates larger σ_z).
4. **(Stretch, high-value) The spec-z demonstration:** with the current catalogue, show that the final
   posterior's shape is carried *entirely by the spectroscopic-host events* — events whose
   sky-localization box contains a spec-z galaxy give informative single-event posteriors; events with
   only photo-z hosts give flat/railing ones. See §3.3 — this is the cleanest visual proof of point 2.

## 2. Paper planning (use GPD)
- Kick off with the GPD paper workflow (`/gpd:write-paper` after a planning pass, or `/gpd:plan-phase`
  for the structure). Engage `gpd-bibliographer` for the .bib (Gray 2020, Hitchhiker 2212.08694,
  GWcosmo 2308.02281, Cross-Parkin 2502.17747, Echoes/CHIMERA 2509.18243, McConnell & Ma 2013, etc.).
- **Two versions:** a **short/letter** (headline: current catalogue → uninformative; the feasibility
  heatmap) and a **long** (full methodology, the bias-source catalog, the bridge investigation, the
  derivations, the mass-relation assessment). Plan shared figures.
- **Proposed figure list (decide/refine):**
  - F1 — pipeline overview (from `docs/PIPELINE_FLOWCHART.md`).
  - F2 — bridge rung ladder: photo-z is the decisive ingredient (`outputs/rungG_photoz.pdf`).
  - F3 — candidate landscape: truth stranded between rails; `D_sm` de-biases-no-peak
    (`docs/figures/bias_resolution_summary.png`).
  - F4 — **spec-z demonstration** (§3.3): per-event posteriors coloured by host redshift type; the few
    informative (spec-z) vs many flat (photo-z). **The money figure.**
  - F5 — **σ_z/σ_M precision heatmap**, 1-D and 2-D panels + a target-precision contour. **The forecast.**
  - F6 — example single-event posteriors (one good spec-z-hosted, one flat photo-z).
  - (long only) the M_BH–M_* relation + scatter; the bias-source catalog table.

## 3. Implementation track

### 3.1 Final "current-state" run (entire catalogue, real errors) — WITH the correctness fixes
Produce the honest current-catalogue result. Before the production run, land (all are real, all feed
this number):
- **Frame fix #15 / PR #17** (z_helio→z_cmb) — needs the reduced catalogue **regenerated** from
  `GLADE+.txt` and a **re-simulation** (host z changes → d_L/SNR/CRB change).
- **dV_c-once interpretation** (`CATALOG-INTERPRETATION.md`): the in-cat numerator is missing one dV_c
  vs the dark branch (`bayesian_statistics.py` :1623/:1646 vs B_num :1462-1480). Apply `p_red = N·p_bg/Z_g`.
- **Num/denom photo-z smearing symmetry** (smear P_det in the selection denominator with the same kernel).
  NB: these correctness fixes do NOT cure the info-starvation (proven), but the *final reported result*
  must be correct, not just rail-free-by-accident. `[PHYSICS]` changes → use `/gpd:` + the protocol.

### 3.2 Flexible σ_z / σ_M parameterization (the heatmap engine)
- Make σ_z and σ_M injectable as free parameters (orders of magnitude) in BOTH simulation and inference,
  self-consistently (draw true z/M; report noisy; source at true; infer with matching kernel).
- **Design decision to settle (you flagged it):** keep Gaussian (normal) measurement errors, OR fix the
  reported values exactly at chosen σ for a clean setup? Recommendation to evaluate: keep Gaussian for
  realism (it's what the math assumes), but offer a "delta/fixed" mode for clean asymptotics — the
  bridge already shows the Gaussian case; a fixed mode isolates the prior/normalization cleanly.
- Engine + heatmap recipe: see `HANDOFF-SIGMAZ-SIGMAM-FORECAST-20260630.md`. Build the with-BH-mass
  (σ_M axis) closure (the bridge `rung_I` is currently `with_bh_mass=False`); report posterior WIDTH
  (not MAP), multi-seed; widen the H₀ grid so "flat" is distinguishable from "peaked".

### 3.3 Spec-z-tag per-event decomposition (the demonstration in §1.4)
- **Precondition:** the reduced catalogue currently **drops the redshift flag column** (`handler.py`
  ~:310-315). Retain it (flag 1=photometric, 3=spectroscopic; flag 2 still excluded) so each candidate
  host carries its measurement type.
- Then, on the final current-state run, decompose: for each event, is there a spec-z host in the
  sky-localization box? Plot single-event posteriors coloured by "has spec-z host" vs "photo-z only".
  Expected: the informative (peaked) single-event posteriors are the spec-z ones; the photo-z-only
  events are flat/railing. Confirm the final stacked posterior's shape is driven by the spec-z subset.
- This is BOTH an analysis and figure F4, and it directly proves the paper's central claim.

## 4. Research track (GPD) — stellar-mass → BH-mass relation (NEVER verified)
- **Current code** (`handler.py`): `M_BH = exp(α + β·ln(M*/10))`, with `α = 7.45·ln10`, `β = 1.05`,
  fit errors `d_α = 0.08·ln10`, `d_β = 0.11` (`:30-33`); function `_empiric_stellar_mass_to_BH_mass_relation`
  (`:1033`), applied in `_map_stellar_masses_to_BH_masses` (`:801`); M* is in 10¹⁰ M_⊙ (so `/10` → 10¹¹ M_⊙).
  i.e. `log₁₀ M_BH = 7.45 + 1.05·log₁₀(M*/10¹¹ M_⊙)`. There is also an inverse `_empiric_MBH_to_M_stellar_relation`.
- **Tasks (use `/gpd:` — research + derivation + verification):**
  1. Identify and CITE the source relation (the 7.45/1.05 form is McConnell & Ma 2013-style M_BH–M_*;
     confirm exact paper/table, and whether it is M_BH–M_bulge vs M_BH–M_*total — GLADE M* is *total*).
  2. **Intrinsic scatter:** `d_α`,`d_β` are fit-PARAMETER uncertainties, NOT the relation's intrinsic
     scatter (~0.3–0.5 dex). The code's σ_M is therefore likely a large UNDER-estimate of the true
     M_BH uncertainty — this matters for the with-BH-mass channel AND for calibrating the σ_M heatmap
     axis to reality. Re-derive σ_M including intrinsic scatter; verify dimensional/units handling.
  3. **Low-mass extrapolation:** the relation is calibrated on massive galaxies/BHs; EMRI-host MBHs are
     ~10⁵–10⁷ M_⊙ — assess validity/flattening/scatter growth at low mass (cf. Reines & Volonteri 2015).
  4. Verify the code (and the inverse) correctly implement the chosen relation; propose a better one if warranted.

## 5. Open blind spots / remaining investigations (TRIAGE before/within the paper)
1. **σ_M railing analog — UNTESTED.** The same prior/normalization-domination that rails σ_z could
   affect the host-MASS channel (the 4-D / with-BH-mass path) since σ_M is also large (see §4.2). The
   original photo-z handoff flagged this (§7) and it was never investigated. Check it on the bridge.
2. **inference-consistency-audit (paper-blocker, memory `inference-consistency-audit`):** a documented
   p_sample ≠ p_comp prior mismatch + sky-population blind spot across the two pipelines, "awaiting
   reconciliation-direction decision." VERIFY whether it's been resolved; it must be before the paper.
3. **Mass-relation scatter (§4.2)** feeds 1 and the heatmap σ_M axis — resolve early.
4. **2-D (with-BH-mass) channel history:** it had its own fixes (H3 etc., `H0_BIAS_RESOLUTION.md` §3.15);
   confirm it is correct in the current photo-z regime and with the true (large) σ_M.
5. **Pixelated HEALPix completeness (Change 5, recent commits `aa8054a`/derivation-change5-*):** confirm
   it is integrated and validated for the final current-state run.
6. Frame fix needs catalogue regen + re-sim (§3.1); flag column must be retained (§3.3).

## 6. The σ_z/σ_M heatmap → companion handoff
Full method, pitfalls (MAP-vs-width, multi-seed, N_obs/n_gal scaling), and first steps:
`.planning/HANDOFF-SIGMAZ-SIGMAM-FORECAST-20260630.md`.

## 7. Side items (not blocking)
- **PR #17** (`[PHYSICS]` frame fix z_helio→z_cmb to `main`) — **DEFERRED**, not merged: GitHub was
  unreachable from the environment at wrap-up (TLS timeouts on api + git host; intermittent — creation
  worked earlier). The branch `fix/cmb-frame-redshift` is pushed; just merge the PR via web or
  `gh pr merge 17 --squash --delete-branch` when connectivity returns (keeps CI in the loop; no local
  force-merge needed). Issues **#15** (frame), **#16** (host PV treatment) are filed on milestone "Paper Submission".
- The reduced catalogue + GLADE+.txt are gitignored (local data); regenerate after the frame fix.

## 8. Artifact index (produced this session, all committed on `physics/photoz-joint-normalisation`)
- `.planning/derivation-photoz-incatalog/`: COMPARISON, GAP-ANALYSIS, CATALOG-INTERPRETATION,
  FRAME-SYSTEMATIC, DERIVATION-HIERARCHICAL, INCREMENT3-DSM-VERDICT (+ earlier NORMALISATION-FIX etc.).
- `docs/PIPELINE_BUGS_REPORT.md`, `docs/BIAS_RESOLUTION_ATTEMPTS_REPORT.md`, `docs/PIPELINE_FLOWCHART.md`,
  `docs/H0_BIAS_RESOLUTION.md` §3.18, `docs/figures/bias_resolution_summary.png`
  (regen: `scripts/bridge_closure/plot_bias_resolution_summary.py`).
- Bridge prototype: `scripts/bridge_closure/_rungI_verify_B.py` (`hierarchical_shared_latent` = `D_sm`).
- Commits: 0bd1f73, bd66f5b, c42f558, 415500b, 5ef8c6e, a8cbab0, 145bf7f (+ frame fix 7021f6f on its branch).
