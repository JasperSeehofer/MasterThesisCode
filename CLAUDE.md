# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## AI-assisted development — authorship & scientific ownership

This repository is developed with AI coding assistance (Claude Code) under an explicit discipline:
**the author (Jasper Seehofer) owns every scientific decision.** AI assistance is confined to
implementation and is gated by the project's `physics-change` protocol — no formula, physical
constant, waveform parameter, or model choice is changed without a documented derivation,
dimensional analysis, limiting-case check, literature reference, and a regression test (see
`.claude/skills/physics-change/` and `.claude/rules/physics-validation.md`). The physics, the
interpretation, and the "why does this result appear" analysis are the author's; the tooling
documented below encodes the guard-rails that keep the AI-assisted parts verifiable and honest.

## Orchestration: model & effort tiering (author mandate, 2026-08-07)

**A session running on Fable (top-tier model) is ALWAYS an orchestration session.** Fable holds
the scientific farsight and structure and uses it to orchestrate the research steps; it delegates
everything not on the high-orchestration / scientifically-complex end to subagents (use them
heavily) and workflows. Applies to single `Agent` spawns exactly as to `Workflow` launches —
do **not** let every agent inherit the top-tier session model by default. Before every launch,
assign per agent from this routing table:

- **Model** — mechanical/formulaic stages (record appending, file generation, commits, schema
  recon, running existing scripts on new inputs) → `sonnet` (or `haiku` for pure lookups);
  derivation, adversarial verification, pre-registration authoring, physics interpretation →
  inherit (top tier). When unsure between two tiers for a mechanical task, pick the cheaper one.
- **Effort** — `low/medium` for mechanical stages, `high` for standard analysis, `xhigh` ONLY for
  adversarial verifiers, novel derivations, and band/prereg authoring. Uniform-xhigh workflows are
  a drift smell: justify each xhigh in the launch summary or lower it.
- **Top-tier hard cap (author mandate, 2026-08-14)** — at most **~3 top-tier (inherit) agents per
  workflow**: the synthesis chair plus at most 1–2 decisive verifiers. Every fanned-out stage
  (audit lenses, reproducers, judge panels, per-item verifiers) runs `sonnet` regardless of its
  adversarial label — panel redundancy substitutes for model tier. Compute each phase's fan-out
  (items × panel size) BEFORE launch and state it in the launch summary; any phase whose fan-out
  depends on an earlier phase's output must carry an explicit cap in the script. Third-party
  workflow engines (e.g. /commission) must be re-tiered to this table before launch.

State the chosen tiering (one line) when proposing a workflow so the author can veto overkill.

**Cluster access discipline (2026-09-03, ledger row #357):** all agent cluster traffic goes
through `cluster/agent_ssh.sh` (3-slot semaphore, local sleeps, backoff on a refused mux
session). Closing the ControlMaster (`ssh -O exit|stop`), deleting `~/.ssh/cm-*`, `pkill ssh`,
`sleep ≥ 60` inside a remote command, and parallel ssh fan-out from one command are BLOCKED by
`.claude/hooks/ssh-guard.py`. A refused mux session means the server's session cap is full, not
a dead master; the socket is OTP-authenticated and only the author can restore it. One
cluster-ops agent per batch holds the cluster; readers and builders never ssh.

**Subagent waiting (2026-08-20):** subagents must never end a turn to "wait for a completion
notification" on a process the harness does not track — every wait is a blocking foreground
command. Evidence: five parking incidents across three agents in one session.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### System prerequisites (must be installed before uv sync)

- **CUDA 12 toolkit** — required on the GPU cluster for the `gpu` extras
- **GSL** (GNU Scientific Library) — required by `fastemriwaveforms` at build time
  - Arch/Manjaro: `sudo pacman -S gsl`
  - Ubuntu/Debian: `sudo apt install libgsl-dev`
  - On the cluster: check with `module load gsl` or ask the sysadmin

### Set up the environment

```bash
# Dev machine (no GPU) — installs CPU waveform variant
uv sync --extra cpu --extra dev

# Cluster (GPU, CUDA 12) — installs GPU waveform variant
uv sync --extra gpu
```

`uv sync` creates `.venv/` in the project root and installs exactly what is in `uv.lock`.
The lock file is committed to git, so every machine gets the same versions.

### Running code

```bash
# Run the package
uv run python -m darksiren_emri <working_dir> --simulation_steps N

# Run tests (dev machine) — also prints coverage summary
uv run pytest -m "not gpu and not slow"

# Run benchmarks
uv run pytest -m "slow" --benchmark-only

# Run mypy
uv run mypy darksiren_emri/ darksiren_emri_test/
```

Note: `fastemriwaveforms` installs as the `few` Python package — `import few`, not `import fastemriwaveforms`.

### Reproducible simulation runs

Pass `--seed <int>` to fix the random state; when omitted, a random seed is chosen, logged, and recorded — with `git_commit`, `timestamp`, and all CLI args — in `run_metadata.json` in the working directory, so any result ties back to the exact code and parameters that produced it. Cluster per-task seeding (`BASE_SEED + SLURM_ARRAY_TASK_ID`) is owned by the `/cluster` skill.

**Dataset pinning (2026-08-20):** any multi-GB input not in version control (galaxy
catalogues, cluster outputs) carries a checksum pin at each consumer, STOP-gated on mismatch —
machine-to-machine copies of "the same" file are not the same file. Evidence: a stale local
galaxy catalogue silently fed every local analysis until a fidelity gate caught it; the
cluster copy of record differed.

## Dev Workflow

### Linting and formatting (run manually or triggered automatically on commit)

```bash
uv run ruff check --fix darksiren_emri/   # lint and auto-fix
uv run ruff format darksiren_emri/        # format
uv run mypy darksiren_emri/               # type check
```

Pre-commit hooks run ruff, ruff-format, and mypy automatically on every `git commit` (whole-tree;
there is **no** pytest hook — run `uv run pytest -m "not gpu and not slow"` manually before committing).
To run all hooks on all files manually:
```bash
uv run pre-commit run --all-files
```

Alternatively, activate the virtual environment once for a session:

```bash
source .venv/bin/activate
python -m darksiren_emri ...  # works without uv run prefix
```

### Adding a new dependency

```bash
uv add <package>                    # add to core deps
uv add --optional gpu <package>     # add to gpu group
uv add --optional dev <package>     # add to dev group
```

This updates both `pyproject.toml` and `uv.lock`. Commit both files.
Never manually edit the dependencies list in `pyproject.toml`.

## Running the Code

```bash
# EMRI simulation (generates SNR + Cramer-Rao bounds)
python -m darksiren_emri <working_dir> --simulation_steps N [--simulation_index I] [--log_level DEBUG]

# Bayesian inference (evaluate Hubble constant posterior)
python -m darksiren_emri <working_dir> --evaluate [--h_value 0.73]

# SNR analysis only
python -m darksiren_emri <working_dir> --snr_analysis
```

Every `--evaluate` run emits a per-event both-channel diagnostics CSV — check the run dir before building provenance caveats.

## Cluster Deployment

All bwUniCluster 3.0 (KIT) operations — submit/monitor/retrieve, SLURM/CLI flags, preflight, and dataset provenance — are owned by the `cluster` skill, the single source of operational truth. **Use the `/cluster` skill (`.claude/skills/cluster/SKILL.md`)** instead of duplicating flags and recipes here; see also `cluster/README.md`.

## Architecture

The codebase has two distinct pipelines:

### 1. EMRI Simulation Pipeline
`main.py:data_simulation()` drives a loop over `simulation_steps`:
1. `Model1CrossCheck` (cosmological model) samples EMRI events from a distribution
2. `GalaxyCatalogueHandler` resolves each event to a host galaxy from a galaxy catalog
3. `ParameterSpace.randomize_parameters()` + `set_host_galaxy_parameters()` set up the 14-parameter EMRI
4. `ParameterEstimation.compute_signal_to_noise_ratio()` computes SNR using a LISA waveform
5. If SNR >= threshold: `compute_Cramer_Rao_bounds()` computes the Fisher matrix and saves to CSV

### 2. Bayesian Inference Pipeline
`main.py:evaluate()` -> `BayesianStatistics.evaluate()`:
- Loads saved Cramer-Rao bounds from CSV
- Uses `BayesianStatistics` (in `bayesian_inference/bayesian_statistics.py`) to compute the posterior over H0
- `GalaxyCatalogueHandler` (`galaxy_catalogue/handler.py`) resolves candidate hosts from the GLADE+ reduced catalogue; `SimulationDetectionProbability` supplies p_det

### Key Module Responsibilities

- **`parameter_estimation/parameter_estimation.py`** — waveform generation via `few`, Fisher matrix computation (5-point stencil derivatives, default since Phase 10), SNR and Cramer-Rao bounds. The `scalar_product_of_functions` inner product is the computational bottleneck (PSD loop).
- **`LISA_configuration.py`** — LISA antenna patterns (F+, Fx), PSD, SSB<->detector frame transformations
- **`datamodels/parameter_space.py`** — 14-parameter EMRI space with randomization and bounds
- **`bayesian_inference/bayesian_statistics.py`** — Pipeline B (production, the only H0 pipeline): `BayesianStatistics`, `single_host_likelihood`, multiprocessing workers, helper functions. Invoked by `--evaluate`. (Pipeline A — the old `bayesian_inference.py`/`bayesian_inference_mwe.py` dev cross-check — was removed in commit `c1571a2`, 2026-05-01.)
- **`bayesian_inference/simulation_detection_probability.py`** — `SimulationDetectionProbability`: survival-estimator detection probability built from the injection pool, with `RegularGridInterpolator` look-ups. Used by Pipeline B. (Replaced the removed KDE-based `detection_probability.py`.)
- **`bayesian_inference/posterior_combination.py`** — combines per-h-value per-event posterior JSONs into the joint H0 posterior (`--combine`); zero-handling strategies and the canonical Σ log L reference implementation.
- **`cosmological_model.py`** — `Model1CrossCheck` wraps the EMRI event rate model; `LamCDMScenario`, `DarkEnergyScenario` parameter spaces. Backward-compat re-export of `BayesianStatistics`.
- **`galaxy_catalogue/handler.py`** — interfaces with the GLADE galaxy catalog (BallTree-based lookups)
- **`validation/pp_coverage.py`** — synthetic-universe P–P/coverage calibration harness (G4b): flat-ΛCDM tables, Malmquist selection, single-host dark-siren H₀ estimator with switchable host-z kernel ('bare' vs calibrated 'volume'). Run per-seed during campaigns.
- **`constants.py`** — all physical constants and simulation configuration. Key: `H=0.73`, `SNR_THRESHOLD=20`
- **`plotting/`** — all visualization code lives here. Factory functions (`data in, (fig, ax) out`) in topic modules (`bayesian_plots.py`, `evaluation_plots.py`, `model_plots.py`, `catalog_plots.py`, etc.). `_style.py` sets Agg backend + loads `emri_thesis.mplstyle`. `_helpers.py` provides `save_figure()` and `get_figure()`.
- **`callbacks.py`** — `SimulationCallback` Protocol for decoupling the simulation loop from visualization; `PlottingCallback` in `plotting/simulation_plots.py` collects data and produces plots in `on_simulation_end`

### Known Bugs to Be Aware Of

#### Code health
~~1. **`LISA_configuration.py` unconditional `import cupy`**~~ [FIXED, commit `4894648`]: the cupy import is guarded with `try/except ImportError` + `_CUPY_AVAILABLE` in `LISA_configuration.py`, `parameter_estimation.py`, `memory_management.py`, and `decorators.py`. All source modules are CPU-importable.

#### Physics / mathematics (Physics Change Protocol required)
~~4. **`parameter_estimation.py:336` Fisher matrix uses O(e) forward difference** [HIGH]~~ [FIXED Phase 10]: `use_five_point_stencil=True` is now default. Ref: Vallisneri (2008) arXiv:gr-qc/0703086.
~~5. **`LISA_configuration.py` galactic confusion noise absent from PSD** [MEDIUM]~~ [FIXED Phase 9]: `_confusion_noise()` added to `LisaTdiConfiguration`. Ref: Babak et al. (2023) arXiv:2303.15929 Eq. (17).
6. **`physical_relations.py` wCDM params w0, wa silently ignored** [MEDIUM] — GitHub #4: `dist()` accepts them but passes to a hardcoded-ΛCDM hypergeometric function. The review PR (2026-07-04) adds a `NotImplementedError` guard so non-default `w_0`/`w_a` raise instead of silently returning ΛCDM.
~~7. **`bayesian_inference/bayesian_inference.py` hardcoded 10% distance error**~~ [MOOT — Pipeline A removed in `c1571a2`]. Production Pipeline B uses per-source Cramér-Rao bounds from the CSV. GitHub #5 closed.
~~8. **`constants.py` WMAP-era cosmology**~~ [RESOLVED as design choice — G11]: fiducial `OMEGA_M=0.2726`, H0=70.4 km/s/Mpc deliberately match the Barausse (2012) M1 EMRI-population cosmology (arXiv:1201.5888) for a self-consistent mock universe; the Planck-2018 mismatch is a tracked systematic in `docs/gates/G7_systematics_budget.md` (row 6), not a bug. GitHub #6 closed.
~~9. **`datamodels/galaxy.py:66` galaxy redshift uncertainty non-standard scaling**~~ [MOOT — file deleted]: `datamodels/galaxy.py` (Pipeline-A synthetic catalog, GitHub #7) was removed in commit `90bd40ee` (2026-07-04) as dead code; production uses `galaxy_catalogue/handler.py`.

---

## Skill-Driven Workflows

Custom skills in `.claude/skills/` encode repeatable, multi-step workflows. Claude must
use them at the appropriate trigger points — not as optional suggestions, but as mandatory
workflow gates.

### Trigger rules

| Trigger condition | Skill | Behavior |
|---|---|---|
| About to edit a physics file (see list below) with a formula or constant change | `/physics-change` | **Hard gate.** Must invoke before writing any code. Do not skip. |
| After modifying array/GPU computation code | `/gpu-audit` | Suggest running on changed files. |
| Before any `git commit` | `/check` | Run full quality gate (ruff + mypy + pytest). |
| Before any `git commit` (after `/check` passes) | `/pre-commit-docs` | Verify CHANGELOG, TODO, CLAUDE.md, README are consistent with staged changes. |
| User asks "what should I work on?" or "what bugs remain?" | `/known-bugs` | Show current bug status with priorities. |
| User wants to run the simulation or evaluation pipeline | `/run-pipeline` | Use instead of ad-hoc bash commands. |
| About to submit, monitor, or retrieve **anything on bwUniCluster** | `/cluster` | **Consult first.** Run `ssh bwunicluster 'bash -s' < cluster/preflight.sh` and require `VERDICT: READY ✓` before submitting. |
| User opens a new investigation, mechanism hunt, or claim assessment | `/research-cycle` | **Consult first.** Start at stage 0; do not reinvent a runbook. |

### Physics-change trigger files

Any edit to these files that modifies a computed value (not just refactoring/types/comments)
**requires** `/physics-change`:

- `physical_relations.py`
- `constants.py`
- `LISA_configuration.py`
- `parameter_estimation/parameter_estimation.py`
- `bayesian_inference/bayesian_statistics.py`
- `bayesian_inference/simulation_detection_probability.py`
- `cosmological_model.py`

---

## Python Conventions

Dataclass mutable-default handling (`field(default_factory=...)`) and mandatory type annotations (modern `list[...]`/`X | None` syntax, `npt.NDArray[np.float64]`, `Callable` typing, mypy config) — live in **`.claude/rules/python-conventions.md`**, auto-loaded when editing `darksiren_emri/**/*.py`.

## Proposing decisions

Decision-gating proposals go in a **reviewable artifact** — a book chapter, a docs page, or a
standalone explainer — with the decision table inline, not in a chat summary. The approval happens
against something that persists and can be re-read; a summary in the transcript cannot be revisited,
diffed, or cited later. Applies to research-cycle proposals, pre-registered measurements, and
standing-decision changes. (Codified 2026-08-11 from two consistent author signals, 2026-08-05 and
2026-08-07.)

### Approval scope — tag every item in a decision list

Every item put to the author carries a tag, so a one-word reply is unambiguous:

- **[DO]** — authorize work. "Approved" grants it.
- **[RULE]** — a scientific ruling on evidence already in front of the author. "Approved"/"ratified"
  grants it and it binds the record.
- **[STANDING]** — pre-authorize a *class* of future decisions. Granted only when the author says so
  explicitly; the proposal must state the scope and when it lapses.

**Binding default: an approval never propagates to a decision whose inputs did not exist when it was
given.** A branch call, verdict or band comparison that has not been computed yet is never covered by
a blanket "all approved" — it returns to the author as a fresh [RULE]. Keep "approved" for [DO] and
"ratified" for [RULE] so even an untagged reply is readable.

This is the input side of the attribution-precise recording convention already used in
`BIAS_HISTORY_LEDGER.md` (quote the author's verbatim words; mark any itemisation as
orchestrator-derived). That convention makes the *record* honest; this one makes the *ask* honest.
(Codified 2026-08-14 from the author's clarification that "all approved" meant the listed items, not
recursively.)

## Testing Strategy

Tests must be fully runnable on a CPU-only development machine. Code written with the `xp` pattern is testable on CPU without mocking.

```bash
pytest -m "not gpu"                # dev machine (CPU only)
pytest                             # cluster (GPU available)
pytest -m "not gpu and not slow"   # fast subset only
```

- **GPU marker:** Any test requiring a real CUDA GPU must use `@pytest.mark.gpu`. Tests for math/physical functions (`dist`, `power_spectral_density`, etc.) must NOT require GPU.
- **xp fixture:** The `xp` fixture in `conftest.py` parametrizes tests over `numpy` (always) and `cupy` (when available). Use `use_gpu=(xp.__name__ == "cupy")` to thread the flag.
- **Guarding cupy imports:** Test files importing modules that transitively depend on `LISA_configuration` must guard the import with `try/except` and use `pytest.mark.skipif` or `pytest.importorskip("cupy")`.

### Test priority order

1. **Physical correctness** — functions with known analytical limits: `dist(z=0) == 0.0`, `power_spectral_density(f) > 0`, `gw_detection_probability` in `[0, 1]`, `scalar_product(h, h) > 0`
2. **Bounds** — `ParameterSpace` randomized values stay within declared limits; `_parameters_to_dict` returns the correct 14 keys
3. **Regression** — before changing any formula, add a test asserting the old numerical result so the change is verifiable

---

## Math/Physics Validation Workflow

The physics-change protocol — what counts as a physics change, the before-writing presentation gate (old formula, new formula, reference, dimensional analysis, limiting case), the post-implementation checks, and the `[PHYSICS]` commit convention — lives in **`.claude/rules/physics-validation.md`**, auto-loaded when editing physics-trigger files. It is the detail behind the `/physics-change` hard gate (see the Skill-Driven Workflows table above). See [[scientific-computing-validation]] for the promoted cross-project form.

Every gate run appends a row to **`docs/gates/PHYSICS-GATE-LEDGER.md`** (date · commit ref · step · verdict · target), so compliance is evidence rather than inference: a `[PHYSICS]` commit with no ledger row is a gate that cannot be shown to have run. The ledger starts 2026-07-30 and is never back-filled; `/check` reads it as a pre-commit evidence check.

<!-- GSD:project-start source:PROJECT.md -->
## Project

**EMRI Dark Siren H₀ Inference**

A dark siren inference pipeline for measuring the Hubble constant H₀ from LISA Extreme Mass Ratio Inspiral (EMRI) detections. Two pipelines: (1) GPU-accelerated EMRI simulation that computes SNR and Cramer-Rao bounds on bwUniCluster 3.0, and (2) CPU-based Bayesian inference that evaluates the H₀ posterior using the GLADE+ galaxy catalog with completeness correction.

**Core Value:** Measure H₀ from simulated EMRI dark siren events with galaxy catalog completeness correction (Gray et al. 2020), producing publication-ready results.

### Constraints

- **GPU:** CUDA 12 required for `cupy-cuda12x` and `fastemriwaveforms-cuda12x` — must use GPU partition on cluster
- **GSL:** Build-time requirement for `fastemriwaveforms` — must be available via module or container
- **uv:** Primary package manager; must be installable on login nodes (may need local install to `~/.local/bin`)
- **Workspace:** bwHPC workspaces expire (default 30 days, extendable) — final results must be copied to persistent storage
- **Network:** Compute nodes may have restricted outbound access — all dependency installation must happen on login nodes
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

See `pyproject.toml` for complete dependency list and tool configuration.
Key: Python 3.13, NumPy/SciPy/Pandas/Matplotlib, CuPy (GPU), fastemriwaveforms (EMRI waveforms), astropy (constants).
Dev tools: ruff (lint/format), mypy (types), pytest (tests), pre-commit (hooks).
CI: check (lint+type+test on source+tests), integration (slow tests), docs (Sphinx), pages (docs+plots deploy on main).
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

See Typing Conventions, Dataclass Conventions, and HPC/GPU sections above for detailed rules.

- **Files:** `snake_case.py` for source, `test_<module>.py` for new tests. Exception: `LISA_configuration.py` (physics convention).
- **Functions:** `snake_case` — `dist_to_redshift()`, `compute_fisher_information_matrix()`
- **Classes:** `PascalCase` — `ParameterSpace`, `BayesianInference`. Exceptions: `PascalCase` + `Error` suffix.
- **Constants:** `SCREAMING_SNAKE_CASE` — `SNR_THRESHOLD`, `OMEGA_M`, `SPEED_OF_LIGHT_KM_S`
- **Physics symbols** preserved in names: `M`, `H`, `d_L`, `S_OMS()`, `delta_dist`. Ruff N802/N803/N806/N815/N816 ignored.
- **Docstrings:** NumPy-style (`Args:` / `Returns:` / `References:`) for new code.
- **Errors:** `ArgumentsError`, `ParameterEstimationError`, `TimeoutError`, `ParameterOutOfBoundsError`, `WaveformGenerationError`
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture (GSD)

See Architecture section above for pipeline descriptions and module responsibilities.
<!-- GSD:architecture-end -->

## Workflow & GitHub Integration

Workflow entry points and the GitHub issue/label/milestone sync contract live in
`.claude/rules/gsd-workflow.md` (relocated 2026-08-12 to keep this file within its byte budget).

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
