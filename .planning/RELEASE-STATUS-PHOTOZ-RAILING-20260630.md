# Release status — H₀ railing root-caused to host redshift error (photo-z)

**Release marker for the seed-600 H₀ "railing" investigation.** This documents where
the dark-siren H₀ pipeline stands: the railing is understood, root-caused, reproduced,
and the fix space is rigorously mapped. No production physics is changed in this release
(only a corrected code comment); the actual H₀ measurement path and the open fix are
stated below.

Full technical record: `scripts/bridge_closure/BRIDGE-FINDINGS.md` (+ `rung_A..I` scripts
and `outputs/` figures), `.planning/derivation-photoz-incatalog/` (DERIVATION,
PHYSICS-CHANGE-PROTOCOL, NORMALISATION-FIX), memory `h0-railing-rootcause-photoz`.

---

## 1. The symptom
The first fully self-consistent seed-600 production run does NOT recover the injected
H₀ = 0.73; its posterior climbs **monotonically (no interior peak)** to the upper grid
edge (0.86, bias +0.13).

## 2. Root cause (confirmed + reproduced)
The in-catalogue likelihood is **host-redshift-error-dominated**. The GLADE host catalogue
is **~62% photometric** (flag-1; the parse keeps `flag∈{1,3}` and the comment wrongly
called both "measured" — flag 1 is *photometric*, only flag 3 is *spectroscopic*). The
photometric redshift error is **σ_z ≈ 0.035**, which at z≈0.05 is **~10–18× the GW
redshift precision** (σ_z^GW ≈ 0.037·z ≈ 0.002). When σ_z ≫ σ_z^GW the GW distance no
longer localizes the host, and the inference becomes **prior/selection-dominated**: the
rising galaxy redshift prior n(z)∝dV_c/dz drives the posterior up. This is exactly a
*monotonic, non-peaked* posterior — not the wider-but-unbiased result a correctly-
normalized inference would give. Reproduced in the bridge: exact-z → recovers; σ_z=0.035
→ rails.

## 3. What is RULED OUT (each recovers when added alone)
Classic Malmquist (selection is on the *true* SNR); measurement σ²/distance scatter at
production N; the catalogue n(z) density *shape*; the sky dimension / 3-D MVN / Fisher
correlations; candidate selection / ball-tree / coordinate frame (returns the true host
for 99% of events); the completion term B_num; the survival p_det; the pixelated
completeness f_k; the candidate search radius; D(h) magnitude (a confirmed red herring).
The σ_z→0 (spectroscopic) limit recovers — **the method is unbiased in the data-dominated
regime.**

## 4. Why the obvious fixes DON'T work (rigorous)
- **Spec-z-only inference filter** → still rails (events were injected from photo-z hosts;
  filtering one side breaks sim↔inference consistency).
- **Comoving-volume kernel regularization / global de-count** → pass the σ_z→0 gate but
  do NOT de-rail; they *sign-flip* the bias (rail up to 0.87 instead of down to 0.60).
  Truth sits strictly *between* the two rails.
- **Local "consistent denominator"** → rails at *all* σ_z (breaks the Option-A global
  scale-freedom — disqualified).
- **PROVEN OBSTRUCTION:** `p_det ≈ 1` across the whole in-catalogue redshift range (hosts
  far inside the GW horizon) ⇒ **no local selection gradient**. A selection normalization
  can only cancel a density gradient where the selection *varies*; here it doesn't, so the
  global scalar denominator cannot track the local catalogue-density gradient at z*(h). No
  per-event, numerator-only normalization can fix it.

## 5. Potential fixes (open)
- **Robust / ready now — data-dominated regime:** use spectroscopic-quality host redshifts
  (σ_z ≲ 0.002). The forecast/methodology arm on a **simulated catalogue with spectroscopic
  hosts recovers H₀ cleanly** (the project's hybrid framing). This is the path to an actual
  H₀ result today.
- **Deep fix (research) — make photometric catalogues usable:** a *joint* numerator+
  denominator **global same-kernel ratio** ∫p_GW·p_cat / ∫p_det·p_cat sharing one
  population density to the horizon, **plus photo-z-consistent re-injection**, with the
  de-railing coming from **ensemble coherence** across events. The literature (Echoes,
  arXiv:2509.18243 / 2502.17747) shows consistently-normalized photometric catalogues
  *are* unbiased, so a correct form exists — it is outside the numerator-only space we
  searched. Substantial; an active 2025 research frontier.

## 6. Related open item — σ_M (mass error) is also large
The with-BH-mass (4-D) channel uses the catalogue/Fisher **mass** uncertainty σ_M, which
is likewise large in the current setup. The same prior/normalization-domination mechanism
could affect that channel when σ_M is large. **Not yet investigated** — a candidate for the
σ_z/σ_M-realization study (below).

## 7. LISA-era outlook (why this may be far less severe by launch)
LISA flies ~2035. By then DESI / 4MOST / Euclid-spectroscopy / SDSS-V and GLADE-successor
catalogues will provide vastly more **spectroscopic** redshifts and deeper completeness, so
the photometric-σ_z limitation that drives the railing is expected to be substantially
mitigated — pushing real low-z dark sirens toward the **data-dominated (unbiased) regime**.
This is a hopeful framing for the forecast: the method is sound where the data dominates,
and LISA-era catalogues move us there.

## 8. Suggested next directions (separate sessions / branches)
- **`study/sigma-z-m-realizations`** — sweep hypothetical σ_z and σ_M (current vs improved
  vs spectroscopic) to map where the pipeline transitions from data-dominated (unbiased) to
  normalization-dominated (railing); quantify the LISA-era outlook. (Bridge `rung_I` is the
  ready harness.)
- **`physics/photoz-joint-normalisation`** — attempt the deep joint num+denom same-kernel
  ratio + consistent re-injection + ensemble coherence (the genuine GLADE-photo-z fix).

Tag this commit as the release; branch off it for each path.
