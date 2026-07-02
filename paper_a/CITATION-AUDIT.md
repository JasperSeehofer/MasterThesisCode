# Citation Audit — MISSING-Marker Resolution

**Date:** 2026-07-02
**Scope:** Resolution of the 14 `\citep{MISSING:...}` placeholder keys in `paper_a/sections/*.tex`, verification of each reference against authoritative databases, and addition of verified BibTeX entries to `paper_a/references.bib`.
**Branch:** `paper/paper-a-draft` (no commit made; orchestrator commits).

## Summary

| Metric | Count |
|---|---|
| MISSING markers resolved | 14 / 14 |
| Reused existing `.bib` keys | 0 (none of the 14 works pre-existed in `references.bib`) |
| New entries added | 14 |
| Unverifiable / left as MISSING | 0 |
| Hallucinated annotations detected | 0 |
| Total `.bib` entries after update | 39 |

Every `\citep`/`\citet`/`\citealt` occurrence of a `MISSING:` key was replaced and every resolved `% MISSING CITATION:` comment (including the `%% CITATIONS NEEDED` block in `postmortem.tex`) was deleted. No `MISSING` string remains in `paper_a/`. All cite keys in the section files resolve to `.bib` entries.

## Resolution table

| MISSING key | Final key | Work (verified metadata) | Verification source |
|---|---|---|---|
| `MISSING:Barausse2012-mbh-population` | `Barausse:2012fy` | Barausse, MNRAS 423 (2012) 2533–2557, arXiv:1201.5888 | INSPIRE API (texkey match) |
| `MISSING:Bishop2006-PRML` | `Bishop2006` | Bishop, *Pattern Recognition and Machine Learning*, Springer 2006, ISBN 978-0-387-31073-2 (`@book`) | Springer catalogue / Microsoft Research page |
| `MISSING:Borghi2024-chimera` | `Borghi:2023opd` | Borghi et al., ApJ 964 (2024) 191, arXiv:2312.05302 | INSPIRE API |
| `MISSING:CookGelmanRubin2006-posterior-quantile-validation` | `Cook2006` | Cook, Gelman & Rubin, J. Comput. Graph. Stat. 15(3) (2006) 675–692, DOI 10.1198/106186006X136976 | Taylor & Francis DOI + Columbia PDF |
| `MISSING:DelPozzo2012-statistical-dark-siren` | `DelPozzo:2011vcw` | Del Pozzo, Phys. Rev. D 86 (2012) 043011, arXiv:1108.1317 | INSPIRE API |
| `MISSING:DiValentino2021-hubble-tension-review` | `DiValentino:2021izs` | Di Valentino et al., Class. Quant. Grav. 38 (2021) 153001, arXiv:2103.01183 | INSPIRE API |
| `MISSING:Eddington1913` | `Eddington1913` | Eddington, MNRAS 73(5) (1913) 359–360, DOI 10.1093/mnras/73.5.359 | ADS bibcode 1913MNRAS..73..359E + OUP |
| `MISSING:Finke2021-darksirensstat` | `Finke:2021aom` | Finke et al., JCAP 08 (2021) 026, arXiv:2101.12660 | INSPIRE API |
| `MISSING:Gray2022-pixelated` | `Gray:2021sew` | Gray, Messenger & Veitch, MNRAS 512(1) (2022) 1127–1140, arXiv:2111.04629 | INSPIRE API |
| `MISSING:Gray2023-los-zprior` | `Gray:2023wgj` | Gray et al., JCAP 12 (2023) 023, arXiv:2308.02281 | INSPIRE API |
| `MISSING:LVK2021-gwtc3-cosmology` | `LIGOScientific:2021aug` | Abbott et al. (LVK), ApJ 949 (2023) 76, arXiv:2111.03604 | INSPIRE API |
| `MISSING:Malmquist1922-stellar-statistics` | `Malmquist1922` | Malmquist, Medd. Lunds Astron. Obs. Ser. I 100 (1922) 1–52 | ADS bibcode 1922MeLuF.100....1M |
| `MISSING:Mastrogiovanni2023-icarogw` | `Mastrogiovanni:2023zbw` | Mastrogiovanni et al., A&A 682 (2024) A167, arXiv:2305.17973 | INSPIRE API |
| `MISSING:Turski2023-photoz` | `Turski:2023lxq` | Turski, Bilicki, Dálya, Gray & Ghosh, MNRAS 526(4) (2023) 6224–6233, arXiv:2302.12037 | INSPIRE API |

Key convention: INSPIRE texkeys where the work is indexed on INSPIRE; `AuthorYYYY` for the four non-INSPIRE works (Bishop, Cook, Eddington, Malmquist), matching the pre-existing `Wilson:1927`-adjacent style.

## Published-version status of arXiv-flagged entries

All ten arXiv-based works are published; every new entry carries full journal metadata (journal, volume, pages, year, DOI) alongside the eprint field. Two annotations carried the arXiv year rather than the publication year (see mismatches below); both were updated to the published metadata.

## Annotation vs. actual paper — mismatches found (all minor; none wrong-paper)

1. **`LVK2021-gwtc3-cosmology`** — annotation says "2021" (arXiv posting year); the paper was published as ApJ 949 (2023) 76. Entry uses published 2023 metadata. Title verified as "Constraints on the Cosmic Expansion History from GWTC-3"; author list is LIGO Scientific–Virgo–KAGRA (collaboration field set).
2. **`Mastrogiovanni2023-icarogw`** — annotation says "2023, A&A 682, A167"; A&A 682 is a 2024 volume. Entry uses year 2024 with the annotated volume/page confirmed correct.
3. **`Eddington1913`** — annotation title reads "…a known error of observation" (the ADS short form). The journal (OUP) title is "On a Formula for Correcting Statistics for the Effects of a known **Probable** Error of Observation". Entry uses the full journal title; a `.bib` comment records the discrepancy. Same paper (MNRAS 73, 359 confirmed).
4. **`Turski2023-photoz`** — the annotation in `postmortem.tex` used a shortened title ("Impact of modelling uncertainties on the dark standard siren measurement…"); the actual title is "Impact of modelling **galaxy redshift** uncertainties on the **gravitational-wave** dark standard siren measurement of the Hubble constant" (as correctly annotated in `introduction.tex`/`codes.tex`). Same arXiv ID; same paper.
5. **`DelPozzo2012-statistical-dark-siren`** — annotation year 2012 is the publication year (PRD 86, 043011); the arXiv posting (1108.1317) is 2011, hence texkey `DelPozzo:2011vcw`. Consistent; no error.

## Notes and flags

- **Erratum on Cook, Gelman & Rubin (2006):** a published Correction exists — Gelman, J. Comput. Graph. Stat. 26(4) (2017) 940, DOI 10.1080/10618600.2017.1377082. The correction concerns the distributional claim for their aggregate test statistic; the property cited in `coverage.tex` (posterior quantiles of the truth are Uniform(0,1) under calibration — the basis of simulation-based calibration) is unaffected. A `note` field with the erratum was added to the entry. If a referee objects, the modern replacement citation is Talts et al. 2018 (arXiv:1804.06788, simulation-based calibration).
- **No retractions or withdrawals** were found for any of the 14 works (publisher pages / INSPIRE records show none).
- **Content check (abstract-level):** the Turski et al. abstract confirms the specific claims cited in `codes.tex` (width-inflation factors 2 and 5 experiment on real data; posterior pushed toward the empty-catalogue case) and `introduction.tex` (information loss quantifiable, bias not, on real data). The Borghi et al. (CHIMERA) abstract confirms the photometric-vs-spectroscopic forecast claims cited in `codes.tex`. Verification of in-text equation-level claims (e.g. Bishop Eqs. 2.81–2.82, CHIMERA Eq. 12, Gray 2023 sec. 2.1.4) is beyond abstract-level access and is taken from the project's own G5 inspection notes (`.planning/gate/G5a/G5b`).
- **Orphan entries** in `references.bib` (seeded from Paper B; e.g. `Katz:2021yft`, `Chua:2020stf`, `Hinshaw:2012aka`, …) were left untouched per instruction.
- **Pre-existing entry flag (not in scope, worth a later pass):** `Babak:2023lro` (arXiv:2108.01167, "LISA Sensitivity and SNR Calculations", year 2021) carries a texkey suffix inconsistent with its content year; INSPIRE's texkey for that record is `Babak:2021mhe`. Key not changed here since it is cited under the existing name elsewhere.

## Files modified

- `paper_a/references.bib` (14 entries appended, each with a verification comment)
- `paper_a/sections/introduction.tex`
- `paper_a/sections/framework.tex`
- `paper_a/sections/pitfall.tex`
- `paper_a/sections/estimators.tex`
- `paper_a/sections/coverage.tex`
- `paper_a/sections/codes.tex`
- `paper_a/sections/budget.tex`
- `paper_a/sections/postmortem.tex`
- `paper_a/sections/appendix_eddington_m.tex`
- `paper_a/sections/appendix_gray_mapping.tex`
- `paper_a/sections/appendix_sky_marginal.tex`
- `paper_a/sections/appendix_volume_deconv.tex`
- `paper_a/CITATION-AUDIT.md` (this report)
