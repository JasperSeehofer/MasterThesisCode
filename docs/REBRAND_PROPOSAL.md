# Rebrand Proposal — "MasterThesisCode" → a self-descriptive, book-forward name

**Status:** PROPOSAL ONLY. No renaming, no GitHub action, no commit has been made as part of
this document. This is a decision artifact for the author to review, edit, and rule on.
**Date:** 2026-08-12. **Scope:** name candidates + collateral drafts + the operational rename
plan reproduced from `results/campaign51_20260728/RUNBOOK_NEXT_SESSION_9.md` §3.1, expanded
with fresh occurrence counts.

---

## 1. Context

Current identity:
- **GitHub repo:** `JasperSeehofer/MasterThesisCode`
- **Python package (import name):** `master_thesis_code` (`pyproject.toml` project name:
  `master-thesis-code`)
- **README title:** "EMRI Bayesian H₀ Inference"
- **Book title:** *A Dark Siren Discovery Book* (`book/design/BOOK_PEDAGOGY.md`), site
  `<title>` tags read "… | EMRI Dark Siren Book" (`book/site/*.html`), served at
  `jasperseehofer.github.io/MasterThesisCode/` (docs) and a book path under the same Pages site.

Author's rename intent (2026-08-12, paraphrased into criteria):

| # | Criterion | Weight | Note |
|---|---|---|---|
| 1 | Discoverability / self-description — someone searching "dark siren cosmology" finds this and immediately sees there's a book | **High** | Primary driver |
| 2 | Book visibility — the name should signal "instructive book alongside the code" | **High** | Distinguishes from every peer repo below, none of which has one |
| 3 | Researcher appeal — LISA / dark-siren / EMRI researchers recognize it as a serious tool | **High** | Must not read as a toy |
| 4 | Future-proofing — after the two papers, the project continues as a general GW-research/LISA-prep engine | **Minor** | Mild generality preferred, don't over-weight |
| — | Non-competitive tone toward peer codes | (constraint, not scored) | Proud, not disparaging |

---

## 2. Landscape survey — comparable public repositories

Sources: live WebSearch survey (2026-08-12) plus this repo's own prior code-level inspections
(`docs/gates/G5a_gwcosmo_inspection.md`, `docs/gates/G5b_chimera_icarogw_inspection.md` —
line-cited comparisons against gwcosmo and CHIMERA/icarogw/DarkSirensStat already exist here
from the 2026-07-02 commission and are the most rigorous comparison material available; this
section summarizes them plus fills in codes not previously inspected).

| Code | Scope | Docs / pedagogy level | What it doesn't cover |
|---|---|---|---|
| **gwcosmo** (`git.ligo.org/lscsoft/gwcosmo`) | The LVK dark-siren pipeline (Gray et al. 2020/2022/2023): pixelated / line-of-sight galaxy-catalogue $H_0$ inference for compact-binary GW events, production code behind published LVK cosmology results | API-reference style docs; no narrative/tutorial book; assumes the reader already knows the Bayesian dark-siren framework | Ground-based CBC only — no EMRI/LISA waveform or Fisher-matrix layer; no instructive walkthrough of *why* the estimator is built the way it is |
| **icarogw** (`github.com/simone-mastrogiovanni/icarogw`) | General-purpose hierarchical population-inference package (spectral + dark sirens), noisy/heterogeneous/incomplete observations, published in A&A | Package-reference docs + example notebooks | Same audience as gwcosmo — population-inference researchers already fluent in the method; no LISA/EMRI-specific layer, no beginner arc |
| **DarkSirensStat** (`github.com/CosmoStatGW/DarkSirensStat`) | The original dark-siren + galaxy-catalogue $H_0$/modified-propagation statistical method (predecessor to CHIMERA) | Method-paper-adjacent, minimal standalone docs | Superseded in active development by CHIMERA; ground-based only |
| **MGCosmoPop** (`github.com/CosmoStatGW/MGCosmoPop`) | Hierarchical Bayesian inference for $H_0$, modified GW propagation, and BBH population models | Same lineage/level as DarkSirensStat | Population-model + modified-gravity focus, not an end-to-end simulation→inference pipeline; no LISA/EMRI |
| **CHIMERA** (`github.com/CosmoStatGW/CHIMERA`, `chimera-gw` on PyPI) | Combined Hierarchical Inference Model for EM+GW analysis — bright/dark/spectral siren methods, JAX/GPU-accelerated, actively developed, readthedocs-hosted docs | Fullest docs of the group (readthedocs, install/quickstart, API); still reference-oriented, not a teaching narrative | Ground-based CBC only; no EMRI/LISA waveform generation, no Fisher/CRB layer, no galaxy-catalogue photo-z systematics campaign of the depth this project ran (see G5a/G5b) |
| **GWPopulation** (JOSS-published, `arxiv:2409.14143`) | Hardware-agnostic hierarchical population-inference *toolkit* (building block, not a turnkey $H_0$ pipeline) | Package docs, JOSS paper | Not siren-specific; no catalogue completeness handling, no EMRI |
| **StableEMRIFisher** (`github.com/perturber/StableEMRIFisher`) | EMRI-specific: stable finite-difference Fisher matrices for `few`-based waveforms, CPU/GPU | README + code | Fisher-matrix computation only — no galaxy catalogue, no $H_0$ posterior, no cosmology layer, no book |

**What this project uniquely offers** (grounded in the repo itself, not marketing):

1. **The only surveyed code that is EMRI/LISA-specific *and* runs the full dark-siren chain** —
   GPU EMRI waveform simulation (`few`) → 5-point-stencil Fisher/Cramér-Rao bounds → GLADE+
   completeness-corrected $H_0$ posterior (Gray et al. 2020 formalism) → validated combination.
   gwcosmo/icarogw/CHIMERA/DarkSirensStat/MGCosmoPop are ground-based-CBC pipelines;
   StableEMRIFisher stops at the Fisher matrix and never reaches cosmology.
2. **A calibrated validation instrument other repos don't ship**: the synthetic-universe
   P–P/coverage harness (`validation/pp_coverage.py`), a pre-registration discipline
   (`.claude/skills/physics-change/`, the `[PHYSICS]` commit convention, the physics gate
   ledger) — i.e. the project treats "does the estimator actually recover the truth" as a
   first-class, repeatedly-run question, not a one-time MDC check. G5a documents that even
   gwcosmo's own published validation record is $\sigma_z=0$-only for its foundational paper.
3. **A from-first-principles literature audit of the exact failure mode this project itself
   hit** (the photo-z bare-vs-posterior numerator, the $1/4\pi$ completion prefactor) — cross-
   checked line-by-line against gwcosmo/CHIMERA/icarogw source, not just their papers
   (`docs/gates/G5a_*`, `G5b_*`). No peer repo publishes this kind of adversarial self-audit.
4. **The instructive book** — *A Dark Siren Discovery Book*, an interactive, chapter-by-chapter
   build-and-break narrative (prologue + 10 chapters + honest closing + a "Defect Museum" of
   98 historical wrong turns) that teaches the *reasoning*, including the mistakes, not just
   the final estimator. None of the six peer codes surveyed has anything like it.
5. **AI-assisted-but-author-owned development discipline** documented as a first-class,
   citable methods point (relevant to the papers' methods sections regardless of the physics
   verdict, per RUNBOOK-9 §3.4).

**Tone check:** none of the above is a claim that peer codes are worse at their job — gwcosmo
and CHIMERA are the production/state-of-the-art tools for ground-based dark sirens and this
project's own validation work leans on their published methods. The differentiation is *scope*
(EMRI/LISA vs. ground-based) and *presentation* (a teaching book + a public self-audit), not
quality.

---

## 3. Name candidates

All candidates avoid the bare word "Siren" as a standalone package/repo name: `siren` and
`siren-pytorch`/`siren-torch` (Sitzmann et al., "Implicit Neural Representations with Periodic
Activation Functions") are an entrenched, unrelated ML namespace collision on PyPI/GitHub.
Compound names below were spot-checked against PyPI/GitHub/general web search on 2026-08-12;
none of the compounds returned a collision.

| # | Name | Package (import name) | Tagline | Collision check |
|---|---|---|---|---|
| A | **Siren Primer** | `siren_primer` (dist: `siren-primer`) | "Learn dark-siren cosmology by building the estimator that measures H₀ from LISA EMRIs." | No PyPI/GitHub hit for "siren primer"/"sirenprimer" combined; "primer" alone is a generic, heavily-used word but the compound is free |
| B | **darksiren-emri** | `darksiren_emri` (or short alias `dse`) | "End-to-end EMRI dark-siren H₀ inference — with the book that teaches it." | No PyPI/GitHub hit for "darksiren" as a package name; "EMRI-Search" exists (unrelated tool) but no name overlap |
| C | **SirenForge** | `sirenforge` | "A gravitational-wave dark-siren inference engine, forged on EMRIs, built to extend." | No PyPI/GitHub hit for "sirenforge"/"siren-forge" |
| D | **Sirenarium** | `sirenarium` | "An observatory and classroom for dark-siren cosmology — starting with LISA EMRIs." | No PyPI/GitHub hit for "sirenarium" |
| E | **H0 Siren Lab** | `h0_siren_lab` | "A lab notebook, an inference engine, and a book — for measuring H₀ with LISA EMRI dark sirens." | No exact-name hit; "H0" prefix is common in astro code names generally (e.g. `h0py` does not exist as of this check) but no direct collision found |
| F | **EchoSiren** | `echosiren` | "The gravitational-wave echo of a galaxy you can't quite see — dark-siren H₀ from LISA EMRIs." | No PyPI/GitHub hit for "echosiren" |

At least one candidate per requested axis: **B** is the EMRI/siren-specific literal name,
**A** is the book-forward brand, **C** is the general-engine/future-proof name, **F** is the
playful-but-searchable name (D is a second playful option).

### 3.1 Decision table

Scores 1 (poor) – 5 (excellent) against the author's weighted criteria (§1). Weighted total =
`3×discoverability + 3×book-visibility + 3×researcher-appeal + 1×generality` (max 70).

| Name | Discoverability (×3) | Book visibility (×3) | Researcher appeal (×3) | Generality (×1) | Weighted total |
|---|---|---|---|---|---|
| A. Siren Primer | 4 | 5 | 3 | 4 | **40** |
| B. darksiren-emri | 5 | 2 | 5 | 2 | **38** |
| E. H0 Siren Lab | 4 | 3 | 4 | 3 | **36** |
| D. Sirenarium | 3 | 4 | 3 | 3 | **33** |
| C. SirenForge | 3 | 2 | 4 | 5 | **32** |
| F. EchoSiren | 3 | 2 | 3 | 4 | **28** |

Reasoning notes:
- **A (Siren Primer)** wins on the two criteria the author called *high*-weight together
  (discoverability is strong because "siren" + the cosmology context surfaces the topic, and
  "primer" is an unambiguous book/teaching-material signal); "primer" also mildly serves
  criterion 4 (a primer is a foundation other things build on).
- **B (darksiren-emri)** is the most literally self-descriptive for a researcher's search query
  ("dark siren" + "EMRI" are both in the name) and would rank very well for exact-topic search,
  but says nothing about the book, and is the least future-proof (name is locked to EMRIs even
  after the project generalizes).
- No candidate should be read as a runaway winner — A and B are 2 points apart on a 70-point
  scale. §6 asks the author to rule on this explicitly rather than defaulting to the top score.

---

## 4. Rebrand collateral draft (top candidate: A — Siren Primer)

**Marked adaptable:** every string below keyed to the name is isolated so swapping in
candidate B/C/D/E/F is a find-replace, not a rewrite. Package identifier shown as
`siren_primer` throughout — replace with the chosen candidate's import name.

### 4.1 README hero paragraph (draft)

> # Siren Primer
>
> **Dark-siren cosmology, end to end — and the book that teaches it.**
>
> Siren Primer is a dark-siren inference pipeline that measures the Hubble constant H₀ from
> Extreme Mass Ratio Inspiral (EMRI) gravitational-wave events detected by the LISA space
> observatory, using Bayesian analysis with the GLADE+ galaxy catalogue and a completeness
> correction (Gray et al. 2020). It ships as working, GPU-capable research code — and as
> *[A Dark Siren Discovery Book](https://jasperseehofer.github.io/siren-primer/book/)*, an
> interactive, build-and-break narrative that walks a reader from "why does H₀ disagree with
> itself" to a working estimator, including the wrong turns the project itself took along the
> way. If you're new to dark sirens, start with the book. If you know the field, the pipeline
> below is production code with a pre-registration discipline and a public validation record.
>
> > **Development note.** This code is AI-*assisted* and human-*verified*. The author owns all
> > scientific decisions; every change to physics is gated by a documented verification
> > protocol (dimensional analysis, limiting-case checks, literature references, regression
> > tests). See the `physics-change` protocol in [`CLAUDE.md`](CLAUDE.md).

### 4.2 "How this relates to other codes" comparison table (proud, non-competitive tone)

> ## How this relates to other codes
>
> The dark-siren / GW-population-inference community has excellent public tools already —
> this project builds on their published methods and, in `docs/gates/`, on a line-by-line
> comparison against their source. It doesn't replace them; it covers different ground.
>
> | Code | What it's great at | Where Siren Primer sits alongside it |
> |---|---|---|
> | [gwcosmo](https://git.ligo.org/lscsoft/gwcosmo) | The LVK production dark-siren pipeline for ground-based compact binaries — the reference implementation of the Gray et al. galaxy-catalogue method | Siren Primer follows the same completeness-correction formalism, applied to LISA EMRIs instead of ground-based CBCs |
> | [CHIMERA](https://github.com/CosmoStatGW/CHIMERA) | The most actively developed, best-documented, JAX/GPU-accelerated bright/dark/spectral-siren code, with the fullest reference docs in the field | Siren Primer adds an EMRI waveform → Fisher-matrix layer upstream of the H₀ inference, and a narrative teaching book alongside the reference material |
> | [icarogw](https://github.com/simone-mastrogiovanni/icarogw), [GWPopulation](https://github.com/ColmTalbot/gwpopulation) | General-purpose hierarchical population-inference toolkits — the right choice if you need flexible population models beyond a single H₀ estimator | Siren Primer is narrower and more opinionated: one estimator, EMRI-specific, run to a published validation standard |
> | [DarkSirensStat](https://github.com/CosmoStatGW/DarkSirensStat), [MGCosmoPop](https://github.com/CosmoStatGW/MGCosmoPop) | The methodological lineage CHIMERA grew from — modified-propagation and population-model extensions | Same formalism family; Siren Primer's contribution is the LISA/EMRI branch, not a competing ground-based method |
> | [StableEMRIFisher](https://github.com/perturber/StableEMRIFisher) | Focused, well-validated EMRI Fisher-matrix computation | Siren Primer computes EMRI Fisher/CRB too, then carries the result all the way to a galaxy-catalogue H₀ posterior |
>
> If your project needs ground-based dark sirens today, gwcosmo or CHIMERA are the mature
> choice. If you're learning the field or working on LISA/EMRI dark sirens specifically, that's
> what Siren Primer is for.

### 4.3 GitHub topics / keywords list (draft)

```
dark-sirens, gravitational-waves, cosmology, hubble-constant, lisa, emri,
extreme-mass-ratio-inspiral, bayesian-inference, galaxy-catalog, glade,
fisher-matrix, gpu, cupy, astrophysics, standard-sirens, gw-cosmology
```

Rationale: mirrors the vocabulary a learner or researcher would actually type ("dark sirens",
"hubble constant", "LISA", "EMRI"), includes the concrete tools (GLADE, Fisher matrix, GPU) a
researcher scans a topics list for, and avoids topics already crowded by unrelated projects
(e.g. bare "siren").

---

## 5. Operational rename plan (reproduced + expanded from RUNBOOK-9 §3.1)

Source: `results/campaign51_20260728/RUNBOOK_NEXT_SESSION_9.md` §3.1 (author-named, phased,
explicitly "NOT a single yolo rename"). Reproduced verbatim below as (a)-(d), with occurrence
counts and one addition found during this survey.

**(a) GitHub repo rename** — `JasperSeehofer/MasterThesisCode` → new name. GitHub auto-redirects
the old URL, but: update the local `git remote` on every clone (dev box + cluster), update any
hardcoded `github.com/JasperSeehofer/MasterThesisCode` URLs (badges, Pages links, book site
links), and note that **git remote redirects do not follow forever** if the old name is later
reused by someone else — treat the redirect as a grace period, not a permanent alias.

**(b) Python package rename** `master_thesis_code` → new import name. Large mechanical refactor:
imports across source + tests, `pyproject.toml` (`name`, any `[tool.*]` path references),
sbatch/cluster scripts, CI workflows (`.github/workflows/*.yml`), docs (Sphinx `conf.py`,
autodoc module paths), book links that reference module paths. **Do on a branch with full-suite
green; coordinate with cluster venv rebuild** (the cluster venv is built against the current
package name — a stale venv after rename will import-error opaquely).

**(c) Operational traps:**
- **Local directory path is keyed into Claude memory/session state**
  (`~/.claude/projects/-home-jasper-Repositories-MasterThesisCode`) **and the garden registry
  Path column** — renaming the local directory silently orphans memory + briefings. Plan an
  explicit migration step (do not rename the local dir until the memory/registry migration is
  scripted and verified).
- **Cluster ONE-repo rule**: the cluster is required to hold exactly one repo copy — rename
  there in the *same window* as the GitHub rename, not staggered, or the cluster ends up
  pointing at a now-redirected/stale URL.
- **Workspace paths and `DATA_INVENTORY` references** — anything that hardcodes the old repo
  or package name as a path component (workspace symlinks, inventory manifests) needs an audit
  pass, not just a global find-replace (some of these are on the cluster filesystem, not in
  git).
- **Book/Pages URLs** — `jasperseehofer.github.io/MasterThesisCode/` (docs) and the book's own
  path under Pages both change; any external links (e.g. from the papers, once submitted) would
  break unless a redirect page is left at the old Pages path.
- **Addition found during this survey (not in RUNBOOK-9):** the `pyproject.toml` **PyPI
  distribution name** `master-thesis-code` — if this project is ever published to PyPI (not
  currently the case, no `pypi.org/project/master-thesis-code` found), the dist name would need
  reserving under the new name separately from the GitHub/import-name rename, and PyPI project
  names cannot be reused/redirected the way GitHub repos can. Low priority now, but worth
  reserving the chosen new dist name early and cheaply (an empty placeholder release) once the
  name is picked, to prevent squatting.

**(d) Needs the author's name choice first.** No further mechanical work should start until §6
is resolved.

### 5.1 Occurrence counts (measured 2026-08-12, repo-tracked files only, `.git/` excluded)

| String | Total files | source (`master_thesis_code/`) | tests (`master_thesis_code_test/`) | `cluster/` | `book/` | `docs/` | `.github/` |
|---|---|---|---|---|---|---|---|
| `master_thesis_code` (lowercase/import form) | 509 hits across the tree | 46 files | 102 files | 19 files | 35 files | 70 files | 4 files |
| `MasterThesisCode` (repo-name/CamelCase form) | 219 hits across the tree | 0 files | 1 file | 21 files | 25 files | 1 file | 0 files |

Read: the lowercase import-name form is the dominant occurrence and is concentrated in source +
tests (mechanical import rename, step (b)); the CamelCase repo-name form is concentrated in
`cluster/` and `book/` (URLs, remote references, badge links) — exactly the areas RUNBOOK-9 (c)
flags as trap-prone, and consistent with it being a *repo identity* string rather than a code
symbol. `docs_src/` had zero hits for both (Sphinx source likely references the module
programmatically, not by literal string — worth a quick manual check before assuming it's
untouched).

---

## 6. Open decisions for the author

1. **Name choice.** §3.1's decision table is advisory (A/B are 2 points apart, not decisive).
   The author's call, weighing especially: is book-visibility (favoring A) or literal
   EMRI/siren searchability (favoring B) the more important lead signal for the *researcher*
   audience specifically, vs. the *learner* audience.
2. **Timing — before or after paper submissions?** Renaming before submission means the papers
   can cite the final name/URL from the start (cleaner citation, no redirect dependency for
   reviewers). Renaming after means zero risk of a mid-review-cycle broken link and zero
   distraction during the physics-verdict work RUNBOOK-9 prioritizes (§3.1 is explicitly filed
   under "filler-task menu", i.e. lower priority than the venue-transfer verdict track).
3. **One window or staged?** RUNBOOK-9 (a)+(c) already require the GitHub rename and the
   cluster-repo rename to happen in the *same* window (the ONE-repo rule). Open question is
   whether the **Python package rename (b)** should happen in that same window or be staged
   later: doing it together is one mechanical PR to review instead of two, but couples a
   large, low-risk-but-high-diff refactor to the higher-stakes GitHub/cluster identity change.
   A staged option (GitHub+cluster rename first, package import name second, once full-suite
   green on a dedicated branch) reduces blast radius per step at the cost of a longer window
   where the repo name and package name disagree.
