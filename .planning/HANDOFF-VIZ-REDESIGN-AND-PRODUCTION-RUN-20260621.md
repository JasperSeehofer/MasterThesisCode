# HANDOFF — Visualization redesign (new thrust) + production-run in flight

**Date:** 2026-06-21  **Author:** long session with Jasper (continuation of the mass-fix/D(h) work)
**Two independent threads:** (A) a NEW visualization-redesign request (the focus of the next session)
and (B) the in-flight fresh production run (autonomous on the cluster; just needs monitoring + the
gated fan-out). Read both; A is the work, B is the watch.

---

## THREAD A — Visualization redesign (the next session's main work)

### What the user asked for (verbatim intent)
> "I am unhappy about the plots/figures the pipeline creates — paper, non-paper, and interactive. The
> information is mostly right but the **design is old and classic**. I want a **more modern approach to
> scientific visualization**, inspired by the GW/cosmology research field AND general scientific-viz best
> practices. Take a look at the current state, do **in-depth research**, then **propose changes and
> prepare a few different designs I can choose from**. Be creative — think about *what* information should
> be shown, *how*, and in *which formats*."

So the deliverable is a **proposal document with multiple cohesive design directions to choose from**,
per-figure redesign recommendations, and creative new-figure ideas — NOT (yet) an implementation. Let the
user pick a direction first, then implement in a follow-up.

### Current state (scouted — start here, then go deeper)
- **Plotting package:** `master_thesis_code/plotting/` — factory-function architecture (data in →
  `(fig, ax)` out) per CLAUDE.md. Topic modules:
  - `bayesian_plots.py` (H0 posteriors), `evaluation_plots.py`, `model_plots.py`, `catalog_plots.py`,
    `fisher_plots.py`, `convergence_plots.py` (+ `convergence_analysis.py`), `simulation_plots.py`,
    `sky_plots.py`, `single_event_detail.py`, `dashboard_plots.py`, `physical_relations_plots.py`.
  - `paper_figures.py` (30 KB — the publication figures), `interactive.py` (62 KB — the interactive/HTML
    figures, likely Plotly; published to GH Pages under `interactive/`).
  - Infrastructure: `_style.py` (Agg backend + loads the mplstyle via `apply_style()`), `_helpers.py`
    (`save_figure`, `get_figure`), `_colors.py`, `_labels.py`, `_data.py`, `_metrics.py`.
- **Design language today** = `master_thesis_code/plotting/emri_thesis.mplstyle`:
  - REVTeX-sized (6.4×4.0 default; fonts 7–9 pt for 3.375"/7.0" columns), `savefig.dpi 300`,
    `pdf.fonttype 42`, `text.usetex False`.
  - Spines top/right off, inward ticks, frameless legend, `constrained_layout`.
  - **Color cycle: Okabe–Ito (Wong 2011)** — already colorblind-safe. Default cmap `viridis`.
  - Verdict: it's a *competent classic* matplotlib publication style. The user wants a step beyond
    "competent classic" → modern, distinctive, field-aware, better information design.
- **Outputs:** `simulations/figures/` (paper/thesis PDFs), `simulations/interactive/` (HTML), GH Pages
  deploy (`interactive/`; CI `pages` job on main). Paper `.tex` lives in the paper sources (find:
  `results.tex`, `method.tex` — grep the repo / a `paper/` or `docs/` dir; the registry notes paper
  self-contradictions e.g. SNR>15 vs ≥20).
- **What the figures show today** (enumerate precisely in the workflow): H0 posterior (1D + 2D channels,
  combined + per-event), D(h)/selection function, p_det grids, Fisher/CRB quality, convergence
  (m_z_improvement: MAP vs N events), catalog/sky distributions, single-event detail, the EMRI population
  model. The interactive set mirrors several of these as Plotly HTML.

### How to run the redesign (recommended workflow — ultracode is on)
Scout inline to finalize the figure list, then run a **research + design Workflow**:
1. **Inventory (parallel readers):** enumerate every figure (paper / non-paper / interactive), what each
   encodes, its current chart type + design, and the data behind it. One agent per cluster:
   paper_figures, interactive, the posterior/bayesian plots, the diagnostic/convergence/fisher plots.
2. **Research (parallel, multi-modal — use WebSearch/WebFetch):**
   - **Field conventions:** how GW/dark-siren/cosmology work presents results — gwcosmo, ICAROGW,
     LVK GWTC catalog papers, LISA papers, corner/`ChainConsumer`/`arviz` posterior styles, H0-tension
     plots (the "whisker/forest" comparison plots), PP-plots/coverage. What's the *current* visual
     vocabulary of the field.
   - **General best practices:** perceptually-uniform + colorblind-safe colormaps (viridis/cividis/crameri
     scientific colour maps — Crameri 2020), direct labeling vs legends, small multiples, uncertainty
     visualization (gradient/fan/HDI bands, hypothetical-outcome plots), redundant encoding,
     accessibility (WCAG contrast, CB-safe), typography for figures, data-ink ratio, annotation-as-design.
   - **Modern interactive viz:** beyond static Plotly — linked views, brushing, animated MCMC/HOPs,
     observable-style explorables, what reads well on a GH-Pages site; libraries (Plotly, Bokeh, Altair/
     Vega-Lite, D3, mpl + mplcursors). Trade-offs for a physics audience + a thesis/paper context.
3. **Synthesize (one agent):** produce `docs/VIZ_REDESIGN_PROPOSAL.md` with:
   - **N cohesive "design directions"** (e.g. 3 distinct visual languages — give each a name, a palette,
     typography, layout system, and an ASCII/figure mockup of the flagship H0-posterior figure in each).
   - **Per-figure redesign table:** current → proposed chart type/encoding + why.
   - **Creative new figures / reframings:** what *isn't* shown that should be (e.g. an interactive
     dark-siren "event → host-galaxy candidates → posterior contribution" explorable; a coverage/PP-plot;
     an H0-in-context forest plot vs Planck/SH0ES; a selection-function explainer; an EMRI-population
     "where do the constraints come from" view).
   - A recommended default + an implementation sketch (mplstyle v2 + a paper/interactive theme).
4. **Present to the user with `AskUserQuestion`** (use the `preview` field with ASCII mockups) so they pick
   a direction before any implementation.

### Constraints / preferences to honor
- Keep the factory-function architecture + `apply_style()` ([[feedback_plotting_style]]). A redesign =
  a new mplstyle (v2) + revised factories + possibly a theme switch, not a rewrite of the data plumbing.
- Paper figures must stay REVTeX-publishable (vector PDF, fonts ≥ the journal min, CB-safe, grayscale-safe).
- Interactive figures deploy to GH Pages — keep them self-contained HTML.
- No throwaway analysis scripts ([[feedback_no_adhoc_scripts]]); changes go through the package + its
  `--generate_figures` path / existing tooling.
- This is software/design work → GSD, not GPD (no physics). [[feedback_viz_milestone_gsd]] (the prior viz
  milestone v2.1 used GSD). Consider `/gsd:new-milestone` or `/gsd:plan-phase` for the implementation
  after the user picks a direction.

---

## THREAD B — Production run in flight (monitor + gated fan-out)

### State right now
- **Both physics fixes merged to `main`** (`af6014d`): mass-redshift-convention (`0099ce2`) + L_cat Gray
  A.9/A.10 (`816f904`). `/check` 569 green. ALL prior data RETIRED (DATA_INVENTORY banner; local working
  dirs archived to `simulations/_RETIRED_20260620_pre_massfix_lcat/`).
- **Injection pool DONE + validated:** `injection_20260620-213449_seed43000/simulations/injections` on the
  cluster — 560 files, **504,000 events**, 7 h-nodes, M_z confirmed (max M 1.40e6 > 1e6 source cap). Reused
  by every seed + closure truth (do NOT regenerate). NOTE: survival p_det is h-invariant → future
  injection campaigns should use single-h (`submit_injection.sh` default already changed to 0.73,
  `e00fd7d`); see [[project_injection_todo]].
- **seed500 validation chain (autonomous, SSH-independent):** jobs `5094525`(sim, gpu_h100_il,gpu_a100_il,
  gpu_h100) → `5094526`(merge) → `5094527`(eval, 83-pt superdense 0.73 grid, injection-dep cleared) →
  `5094528`(combine). **Still PENDING — GPU-queue-bound; est start drifted 12:52 → 23:40 and climbing**
  (H100 + both `_il` partitions all contended). Run dir: `run_20260620_seed500_phase50`. It completes on
  its own whenever a GPU slot frees; result → `run_20260620_seed500_phase50/simulations/posteriors{,_with_bh_mass}/combined_posterior.json`.

### Resume recipe (next session)
1. Cluster access: same machine reuses the SSH master socket (`ssh -O check bwunicluster`). If it's dead,
   the user must re-auth (2FA) — the key alone is refused (verified). The SLURM jobs are unaffected by SSH.
2. Check seed500: `ssh bwunicluster 'squeue -u $USER; ls run_20260620_seed500_phase50/simulations/posteriors/combined_posterior.json'`.
   If present, read the MAP (argmax of h_values/posterior) for both variants.
3. **Validation gate:** seed500 1D + 2D MAP should sit sane near 0.73 → confirms the merged code + M_z fix
   produce correct end-to-end results. (NOT a bias verdict — just sanity.)
4. **If sane → fan out:** seeds 600/700/800 @ 0.73 + **closure 0.67 & 0.77**. For 0.73 seeds reuse
   `INJECTION_SOURCE=<pool> submit_resimulate_phase50.sh --seed <S>` (redirect the sim partition to
   `gpu_h100_il,gpu_a100_il,gpu_h100` via `scontrol update` to dodge the gpu_h100 jam; and add the
   injection-array dependency to the eval if injections weren't reused/complete — here they ARE complete,
   so no dep needed). For closure truths: `simulate.sbatch` now threads `H_VALUE` (`61cdd02`) — pass the
   truth; AND build a **closure eval grid** re-centred on 0.67/0.77 (the superdense grid is hardcoded 0.73).
5. **Still to build (Thread B tooling):** (a) the closure eval-grid sbatch; (b) a **PP-plot/coverage +
   per-channel multi-seed MAP + CATONLY-vs-FULL aggregation harness** (absent from 600+ tests) — needed to
   turn the multi-seed + closure data into the bias verdict + the coverage figure referees want.

### The science question this run answers
Residual ~+0.005–0.010 H0 high-bias: is it a systematic or single-seed scatter? Multi-seed @0.73 settles
mean+σ; closure (CATONLY-vs-FULL at h≠0.73) settles whether the completion/D(h) term *tracks truth* (correct
selection physics) or is a fixed offset (EXP-26). Full context: [[project_residual_bias_decomposition]],
`docs/H0_BIAS_RESOLUTION.md`. The D(h) investigation already PROVED the stale "−N·log D(h)" framing dead
(D(h) enters only via per-event L_comp; completion is common-mode, can't drive the 1D-vs-2D split; D(h) is
Gray-faithful, no mandatory fix; f_i h-invariant).

---

## POINTERS
- **Memory:** [[project_residual_bias_decomposition]] (D(h) verdict + campaign-cancel note),
  [[project_injection_todo]] (single-h / h-invariance, DONE), [[project_canonical_data_seed400]] (now
  convention-stale), [[feedback_plotting_style]], [[feedback_no_adhoc_scripts]], [[feedback_viz_milestone_gsd]].
- **Commits this session:** `583f872`/`af6014d` (merges), `935c709` (retire data), `341ca62` ([PHYSICS]
  injection M_z fix), `61cdd02` (simulate H_VALUE thread), `e00fd7d` (single-h injection default).
- **Vault debriefs filed:** sessions 4 + 5 (W-PRE-12 injection-CSV bug; W-CONF-14 wrong-7×-GPU claim;
  W-TOOL-10 watcher false-positive; SCV patterns; EXP-26/27).
- **Docs:** `docs/H0_BIAS_RESOLUTION.md`, `DATA_INVENTORY.md` (RETIRED banner + injection note).
- **Working tree:** `.planning/debug/*` mods are a different session's — leave unstaged. Untracked
  `results/` are local eval outputs.
