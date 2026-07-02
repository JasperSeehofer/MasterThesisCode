# Paper A Consistency Audit

Date: 2026-07-02
Scope: `main.tex` + `sections/*.tex` + `references.bib` + `figures/`
Method: exhaustive cross-check of (1) `\ref`/`\eqref` targets vs emitted `\label`s, (2) `\cite*` keys vs `references.bib` and the `MISSING:` convention, (3) repeated numerical values, (4) `% [RESULT PENDING ...]` markers, (5) `\includegraphics` targets vs `figures/`.

---

## 1. Cross-references (`\ref` / `\eqref` / `\citealt` targets)

**PASS — no dangling references.** Every one of the 300+ `\ref`/`\eqref` usages resolves to a defined label (section labels emitted by `main.tex`, plus all equation/figure/table labels defined in the section bodies).

Defined-but-never-referenced labels (harmless, listed for completeness):
`eq:app:sigmaglob`, `sec:app:gray:beyond`, `sec:app:gray:exact`, `sec:app:gray:local`, `sec:coverage:full`, `sec:coverage:harness`, `sec:coverage:results`, `sec:est:default`, `sec:est:eddm`, `sec:est:eddz`, `sec:est:localratio`, `sec:framework:completion`, `sec:framework:consistency`, `sec:framework:decomposition`, `sec:framework:incat`, `sec:framework:notation`, `sec:intro`, `sec:pitfall:kernel`.

Note: `eq:app:sigmaglob` (appendix_beta_g.tex:16) defines the same object as `eq:pitfall:sigmaglobal` (pitfall.tex:67); only the latter is ever referenced. Consider referencing or dropping the appendix label.

## 2. Citations

**PASS with findings.** All 16 non-`MISSING:` cited keys exist in `references.bib`:
`Abbott:2017xzu, Babak:2017tow, Babak:2023lro, Chen:2017rfc, Dalya:2021ewn, Gair:2022zsa, Gray:2019ksv, Hogg:1999ad, Laghi:2021pqk, LISA:2024hlh, Mandel:2018mve, Planck:2018vyg, Riess:2021jrx, Schutz:1986gp, Vallisneri:2007ev, Wilson:1927`.
All 22 unresolved keys follow the `MISSING:` convention with inline `% MISSING CITATION:` annotations.

### Finding 2a (MAJOR): duplicate `MISSING:` keys for the same work
When the bibliographer resolves these, each variant would become a separate bib entry. Unify to one key per work:

| Work | Keys in use | Locations |
|---|---|---|
| Eddington (1913), MNRAS 73, 359 | `MISSING:Eddington1913-bias`; `-correcting-statistics`; `-noise-bias-correction`; `-statistical-bias` (4 variants) | coverage.tex:2, appendix_volume_deconv.tex:88; pitfall.tex:13; appendix_eddington_m.tex:38; estimators.tex:55 |
| Gray et al. (2022), pixelated completeness, arXiv:2111.04629 | `MISSING:Gray2022-pixelated`; `MISSING:Gray2022-pixelated-completeness` (2 variants) | codes.tex:5,23; introduction.tex:36, framework.tex:212, appendix_gray_mapping.tex:25,156 |
| Gray et al. (2023), LOS z-prior, arXiv:2308.02281 | `MISSING:Gray2023-los-zprior`; `MISSING:Gray2023-gwcosmo-los-zprior` (2 variants; plus see 2b) | codes.tex:5,17,23; introduction.tex:36, estimators.tex:42, appendix_gray_mapping.tex:146 |
| Turski et al. (2023), arXiv:2302.12037 | `MISSING:Turski2023-photoz`; `-photoz-uncertainties`; `-photometric-redshift-dark-sirens` (3 variants) | codes.tex:25,44; introduction.tex:70; postmortem.tex:1,40 |

### Finding 2b (MAJOR): mislabeled MISSING annotation — wrong-target risk
`appendix_sky_marginal.tex:111` cites `MISSING:Gray2023-pixelated` with the annotation *"Gray et al. 2023, 'A pixelated approach to galaxy catalogue incompleteness...', arXiv:2308.02281"*. That title belongs to the **2022** pixelated paper (arXiv:2111.04629, MNRAS 512, 1127); arXiv:2308.02281 is the 2023 joint-inference/LOS paper. The sentence context ("equal-area sky pixels implement the sinθ measure exactly") points to the 2022 pixelated paper. Fix key + annotation (fold into `MISSING:Gray2022-pixelated`).

### Bib housekeeping (minor)
9 entries in `references.bib` are never cited: `Chua:2020stf, Cornish:2017oic, DES:2024tys, Farr:2019rap, Hinshaw:2012aka, Katz:2021yft, Katz:2022yqe, Reines:2015moa, Tiwari:2017ndi`.

## 3. Numerical-value consistency

### PASS — the headline numbers are quoted identically everywhere
- **Rail at 0.86** (production pre-fix, edge mass 1.0): abstract.tex:16, introduction.tex:98, pitfall.tex:100, realdata.tex:29/42/47, postmortem.tex:22, conclusions.tex:27. Consistent.
- **Flip to 0.60** (1/(4π)-only, edge mass 1.0): abstract.tex:20, introduction.tex:101, pitfall.tex:102, realdata.tex:30/42/48, postmortem.tex:23, conclusions.tex:27. Consistent.
- **De-rail to 0.73** (local_ratio MAP 0.73 / mean 0.730; volume_deconv MAP 0.73 / mean 0.740; catonly mean 0.737): abstract.tex:25, introduction.tex:107, pitfall.tex:104, realdata.tex:32–34/51–53, postmortem.tex:27, conclusions.tex:29. Consistent.
- **+0.010 offset** between cured estimators (0.740 vs 0.730): realdata.tex:70 matches Table `tab:derail` means. Consistent.
- **+0.03 residual** of volume_global (MAP 0.76, mean 0.755, edge 2.3e−2): pitfall.tex:105, realdata.tex:31/58–59/65, postmortem.tex:10/26. Consistent.
- **−17 / −17.2 per cent** β_G tilt: −17.2 in pitfall.tex:73/88 and appendix_beta_g.tex:76; rounded −17 in estimators.tex:24, appendix_gray_mapping.tex:102, budget.tex:17, pitfall.tex:101, appendix_beta_g.tex:107. Consistent (rounding); but see Finding 3b.
- **Eddington-in-M**: 1-D mean shift −5×10⁻⁵; 2-D mean 0.790 → 0.770 (−0.020); edge mass 0.216 → 0.023 — identical in estimators.tex:81, budget.tex:18/39, appendix_eddington_m.tex:85–91. Consistent.
- **σ_z-scan biases** −0.0016/−0.0064/−0.023/−0.046 at σ_z = 0.005/0.015/0.035/0.050: identical in pitfall.tex:34, estimators.tex:64, coverage.tex:57, appendix_volume_deconv table `tab:app:eddz`. Consistent.
- **Coverage percentages**: table `tab:pp-clean` (volume rows 0.53–0.61 / 0.66–0.73 / 0.88) matches the text ranges in coverage.tex:35; bare 0–3 per cent matches pitfall.tex:37 and the abstract; 250-realization run 0.004/0.020/0.048 vs 0.548/0.720/0.900 (coverage.tex:61) appears once; full-machinery 0.00/0.00/0.02 → 0.40/0.54/0.82 (coverage.tex:75) appears once. Consistent.
- **Sky factors**: 1.6×10³ at 2° (pitfall.tex:54, appendix_sky_marginal.tex:132) vs "order 1640" (realdata.tex:48) — consistent; 1.8×10⁵ at 0.2 deg² and the 10³–10⁵ range consistent everywhere; sinθ residual median 1.15 / mean π/2 consistent (pitfall.tex:58, budget.tex:16, appendix_sky_marginal.tex:97).
- **Injected truth h = 0.73 vs synthetic-suite h_true = 0.72**: clearly and explicitly distinguished (appendix_volume_deconv.tex:117–118 flags the difference).
- **Postmortem grid [0.60, 0.87], rail at 0.87**: a deliberately different configuration (full 3355-event sample, 0.01-spaced grid); the tab:partialfix caption states this explicitly. Not a contradiction.

### Finding 3a (MAJOR): budget vs coverage — "−2.4 per cent" vs "−0.024 (≈ 3.3 per cent)"
`budget.tex:14` states "−2.4 per cent MAP bias in h at σ_z = 0.035", while `coverage.tex:34` states the same defect as "a fixed MAP bias of −0.024 in h (about 3.3 per cent low in H0)". The budget-table caption itself declares "Magnitudes on H0 and on h are interchangeable percentages", under which rule −0.024 absolute in h is −3.3 per cent, not −2.4 per cent. The "−2.4" reads as the absolute value ×100 mislabeled as a percentage.
**Fix**: budget.tex:14 → "−0.024 absolute MAP bias in h (≈ −3.3 per cent on H0) at σ_z = 0.035".

### Finding 3b (minor): ±8.7 per cent endpoints vs quoted shape values
`pitfall.tex:73` and `appendix_beta_g.tex:77` quote the h³-corrected residual as "+8.7 per cent (h=0.60) to −8.7 per cent (h=0.86)", but the shape endpoints quoted in the same sentences (1.085 → 0.913; source JSON 1.0854 → 0.9135) imply **+8.5 / −8.7 per cent**, and ±8.7 would sum to 17.4, not the quoted −17.2 end-to-end. Round consistently (e.g. "+8.5 to −8.7" or "≈ ±8.6").

### Finding 3c (minor): −0.023 vs −0.024 at σ_z = 0.035, h_true = 0.72
Within `coverage.tex`, the table (`tab:pp-clean`, line 49: −0.024) and the σ_z-scaling text (line 57: −0.023) quote slightly different bare-kernel biases for the nominally identical configuration (they come from different runs: RESULT 1 vs RESULT 2 of the calibration note). `conclusions.tex:22` picks −0.024, `estimators.tex:64/86` and `pitfall.tex:34` pick −0.023. Not wrong, but a referee will ask; add half a sentence ("−0.023 to −0.024 across independent runs") or harmonize.

### Finding 3d (minor): 3375 vs 3355 events in the full-scale confirmation
`realdata.tex:5–6` correctly defines 3375 detected → 3355 after the σ_dL/dL < 0.1 quality cut, and lines 9/54 + `postmortem.tex:4` consistently use 3355 for analysis. But `realdata.tex:73` and `conclusions.tex:74` describe the pending cluster confirmation as running on "all 3375 seed600 events". If the production pipeline applies the same quality cut, this should read 3355 (`realdata.tex:47`'s "archived production run on the full 3375-event sample" may likewise need checking against crux_realdata.md). Verify against jobs 5698617/5698618 when the PENDING result lands.

## 4. `% [RESULT PENDING ...]` markers (complete list)

| File:line | Marker | What is owed |
|---|---|---|
| `main.tex:26` | `RESULT PENDING --- Abstract is written last...` (in-body placeholder, not a comment) | See Finding 5a — abstract is in fact drafted. |
| `sections/realdata.tex:73` | `% [RESULT PENDING: cluster jobs 5698617/5698618 (38-task eval array + dependent combine) provide the full-grid volume_deconv combined posterior MAP; expected peaked ~0.73]` (with in-text `$\text{[PENDING]}$`) | Full-grid confirmation MAP; also echoed in prose at `conclusions.tex:73–77`. |
| `sections/coverage.tex:67` | `% [RESULT PENDING: fig:pp caption facts]` | Numbers for the `fig:pp` caption (realization count, coverage values — likely the 0.004/0.020/0.048 vs 0.548/0.720/0.900 pair from coverage.tex:61). |

Related placeholders (not RESULT PENDING but blocking submission): `main.tex:13–16` author/affiliation `TBD`; `main.tex:52` acknowledgements `TBD`.

### Finding 5a (MAJOR): drafted abstract is not wired into main.tex
`sections/abstract.tex` exists, is fully drafted ("Written LAST" header satisfied), and is internally consistent with all section numbers (0.73 / 0.86 / 0.60 / σ_z/z ~ 0.7). But `main.tex:25–27` still contains the `RESULT PENDING` placeholder and never `\input{sections/abstract}`. Either the builder is expected to inject it (then the placeholder is stale), or `main.tex` should read `\begin{abstract}\input{sections/abstract}\end{abstract}`.

## 5. Figure files

**PASS.** All four `\includegraphics` targets exist in `figures/`:

| Reference | File | Exists |
|---|---|---|
| pitfall.tex:81 | `figures/fig_beta_g.pdf` | yes |
| realdata.tex:41 | `figures/fig_derail_matrix.pdf` | yes |
| realdata.tex:64 | `figures/fig_ablation.pdf` | yes |
| coverage.tex:65 | `figures/fig_pp_coverage.pdf` | yes |

Each figure also has its generator under `figures/scripts/` and `fig_pp_coverage` has data under `figures/data/`.

---

## Summary verdict

Structurally clean: zero dangling `\ref`s, zero unconventioned missing cites, all figures present, and the headline numbers (0.86 rail / 0.60 flip / 0.73 recovery, +0.010, −0.020, −17.2 per cent, coverage collapse) are quoted identically across all sections. Four issues need action before submission: (1) unify 11 duplicate `MISSING:` keys down to 4 works; (2) fix the mislabeled `MISSING:Gray2023-pixelated` citation in appendix_sky_marginal.tex:111; (3) reconcile budget.tex:14's "−2.4 per cent" with coverage.tex:34's "−0.024 (≈3.3 per cent)"; (4) wire `sections/abstract.tex` into `main.tex` (or delete the stale placeholder). Two PENDING result slots (full-grid confirmation MAP; fig:pp caption facts) and the author/acknowledgement TBDs remain open by design.
