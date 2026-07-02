# HANDOFF — Independent verification of the H₀ railing + flagged-TODO ledger (2026-07-01)

**Two purposes:**
1. **Part A** — a durable ledger of every flagged TODO / next step from the current session.
2. **Part B** — commission a **genuinely independent, skeptical re-investigation** of the project
   state that does **NOT** inherit this project's stored conclusions, and that tries hard to
   *falsify* them. The user is (rightly) skeptical that "in-catalogue photometric dark sirens are
   information-starved → the posterior irreducibly rails to the H₀ grid edge" is the *whole* story,
   because the pipeline gave sensible (peaked, near-truth) results in the past. We want to raise
   *both* our beliefs by an in-depth rediscovery — confirming the conclusion only if an independent
   effort genuinely fails to break it.

**Branch:** `physics/photoz-joint-normalisation`. **HEAD:** `b828ab0`. Commits this session:
`555f018` [PHYSICS] mass-relation error budget · `b10433f`/`586562e` F5 forecast + assessment ·
`b82a2d4` [PHYSICS] sky-aware selection · `5e4c312` planning (audit + sky protocol) ·
`479afdd` catalog flag retention · `b828ab0` F4 spec-z decomposition (premise refuted).
Gate green throughout (ruff + mypy + 691 dev tests + slow closures).

---

## PART A — Flagged TODOs / next-steps ledger

### A1. Deferred / flagged physics items (from this session's fixes)
1. **Log-normal host-mass model** (deferred, `[PHYSICS]`). The R&V-2015 relation scatter is ~0.24–0.55
   dex; at that magnitude the current **linear-Gaussian** host-mass error (`handler.py` `BH_mass_error`
   + the with-BH-mass likelihood `bayesian_statistics.py:1773-1783`) leaks ~5–30% probability to
   M<0 and has the wrong skew. Refactor to a log-normal (Gaussian-in-log M) treatment across
   `handler.py` **and** the 2-D likelihood; re-validate the closure. Low practical payoff (the 2-D
   channel is uninformative for H₀ at realistic σ_M — F5), so hygiene, not a result-changer.
2. **With-BH-mass 4D sky branch left ISOTROPIC + flagged** (`bayesian_statistics.py`
   `precompute_global_catalog_selection`, with_bh_mass branch). The sky-aware selection fix
   (`b82a2d4`) made the 3-D channel sky-aware but kept the 4-D (sky×M_z) branch isotropic because a
   per-band 4-D survival is statistics-starved (~41 inj/pixel at NSIDE=32). Needs either a full
   sky×M_z injection campaign (GPU) or a better estimator. Affects only the with-BH-mass posterior.
3. **4-yr PSD vs 5-yr signal-integration mismatch** (pre-existing). `LISA_configuration.py:67`
   `t_obs_years = 4.0` (confusion-noise foreground subtraction) vs the signal SNR integrated over
   `T = 5 yr` (`parameter_estimation.py:88`). Both >1 yr so the sky azimuthal-symmetry argument holds,
   but the two observation times should be reconciled. Flag, decide the correct T, unify.
4. **B_num event-pixel sky delta-collapse** (2nd-order). The completion numerator evaluates `f_k` at
   the *measured* detection direction, not the true-host pixel; only matters if GW localization
   straddles pixels of sharply different `f_k` (3.36 deg² cells). Audit R-item; low severity.
5. **§3.1 correctness fixes for a final current-state number** (NOT yet implemented; note the
   investigation showed these do **not** de-rail): (a) **dV_c-once** — the in-cat numerator carries
   net-0 dV_c while the dark branch `B_num` carries net-1 (`CATALOG-INTERPRETATION.md`); apply
   `p_red = N·p_bg/Z_g`. (b) **num/denom photo-z smearing symmetry**. (c) **frame fix #15**
   (z_helio→z_cmb) — needs a catalogue regen + re-sim to take effect. These make the *reported number*
   correct even though they don't cure the railing.

### A2. Reproducibility / correctness items (pre-existing, from the audits)
6. **Unseeded emcee proposals** (`cosmological_model.py:270-274`) — `--seed` does not make sampling
   fully reproducible. Statistical-correctness item.
7. **H bookkeeping**: `constants.py:26` `TRUE_HUBBLE_CONSTANT = 0.7`, but the injection/story use
   **h = 0.73**. Verify the *actually injected* H₀ and that grid/truth/fiducial are consistent
   everywhere (a 0.70↔0.73 split could masquerade as bias — though it cannot alone explain a rail to
   0.86). **The independent investigation MUST pin this down.**
8. **CLAUDE.md Known Bugs 6–9** (open): wCDM `w0,wa` silently ignored in `dist()`; hardcoded 10%
   distance error in the dev-cross-check `bayesian_inference.py`; outdated cosmology (Ω_m=0.25);
   galaxy z-error scaling `0.013(1+z)^3`.

### A3. Ops / housekeeping
9. **PR #17** (frame fix z_helio→z_cmb) → merge to `main` when GitHub reachable; issues **#15** (frame),
   **#16** (host peculiar-velocity value correction).
10. **File GitHub issues** for the now-fixed mass-relation bugs (555f018) + the sky blind spot
    (b82a2d4) as resolved, when GitHub reachable.
11. **Uncommitted-but-intended**: the 4.7 MB literal F4 JSON (`outputs/f4_specz_decomposition.json`)
    is regenerable and left out of git. Pre-existing clutter (`.planning/debug/*`, `results/`,
    `simulations` symlink) untouched per prior handoffs.

### A4. Paper-milestone tracks still open
12. **A peaked-posterior demonstration arm** — the spec-z forecast / sim-mock (σ_z ≲ 0.002 recovers
    h≈0.725). *This is gated on Part B's verdict* (see below).
13. **Paper drafting (GPD)** — F1–F6 material now in hand (pipeline flowchart, bias-resolution report,
    F5 forecast, mass-relation assessment, sky-selection derivation, F4-inverse). Short letter + long.

---

## PART B — Independent re-investigation (the main ask)

> **Mission.** From the **code and data alone**, independently determine whether the current pipeline
> is correct (physically, statistically, and as software) and whether a **peaked, non-railing H₀
> posterior is achievable**. Form your own hypotheses; do **not** adopt this project's stored
> conclusions. Try hard to *break* the "irreducible railing" claim before accepting it.

### B0. Why this exists (the skepticism to honour — do not pre-answer these)
- The pipeline produced **sensible (peaked, near-truth) posteriors in the past**. Three live
  possibilities, none privileged:
  - **H-past-bug-good:** past sensible results were an *artifact* of a bug that pulled the posterior
    toward the injected truth (e.g., a normalization/prior/grid that happened to center near 0.73, or
    a leak of the true H₀). The railing is the *now-revealed* true behavior.
  - **H-new-bug:** a bug was *introduced* during this project's many changes; the railing is that bug,
    not physics. (This project made ~15 physics commits — ample surface for a regression.)
  - **H-physics (our claim):** the railing is real photo-z information starvation and is irreducible
    with GLADE.
  - **H-artifact:** the railing is a grid/prior/window/normalization artifact, not a data statement.
- We may have **rabbit-holed** into the photo-z explanation and confirmation-biased our tests.

### B1. Independence guardrails (critical — enforce these)
- The **investigator agents must receive ONLY**: the codebase, the data locations, and a **neutral
  problem statement** (below). They must **NOT** be given this project's `.planning/` reports, the
  `memory/` conclusions, `docs/H0_BIAS_RESOLUTION.md`, the bias-resolution report, or any of our
  verdicts. Withhold conclusions until the final comparison phase.
- Practical mechanism: run the investigation as a **workflow** whose agents get only the neutral brief.
  The orchestrator (which has our context) must **not** relay our conclusions into the agent prompts.
  If a fresh human session is used instead, it auto-loads `MEMORY.md` — so treat every memory/report
  claim as an **untested claim to audit**, never as a premise.
- Our prior results **may be requested** (see the Evidence Locker, B6), but **only** with the rule:
  *audit the test's implementation for faithfulness/bugs BEFORE trusting its output.* A result whose
  harness is not independently verified is inadmissible.
- **Blind the injected truth.** Where feasible, run recovery tests without telling the investigators
  the injected H₀ value (hide 0.73), so they cannot unconsciously tune toward it. Reveal only at
  scoring.

### B2. Neutral problem statement (give this to the investigators, verbatim)
> "This repository implements a Bayesian pipeline to infer the Hubble constant H₀ from simulated LISA
> EMRI 'dark siren' events cross-matched against the GLADE+ galaxy catalogue. There are two stages:
> (1) an EMRI simulation producing per-event distance/mass measurements and their uncertainties;
> (2) a Bayesian inference producing a posterior over H₀. Audit the pipeline for physical, statistical,
> and software correctness. Determine whether it can produce a **peaked, well-behaved H₀ posterior**
> that recovers the injected value, and if not, diagnose *why* (find the mechanism from first
> principles). Report bugs, questionable assumptions, and the regimes (if any) where it works."

### B3. Investigation plan (phases)
1. **Fresh-eyes correctness audit** (parallel, independent lenses: physics, statistics, software).
   Read the sim + inference end-to-end; assess correctness from first principles; produce an
   independent bug/concern list — *without* our bug list. Cross-check each other.
2. **Independent minimal reproduction.** Implement a clean, first-principles dark-siren H₀ estimator
   (do NOT reuse our `bayesian_statistics`/`_bridge_lib`) and run injection–recovery on: (a) a spec-z
   mock (σ_z→0) — must recover; (b) a photo-z mock (σ_z≈0.035) — does it rail?; (c) a real-data
   subsample. If the *independent* implementation also rails on photo-z, that is strong evidence for
   the physics claim; if it does not, we have a bug in our pipeline.
3. **Temporal / regression archaeology (targets H-past-bug-good & H-new-bug).** Use git history +
   past run artifacts to find the **last commit/run where the posterior was sensible**, then diff the
   pipeline logic to HEAD and **classify the sensible→railing transition** as: a correctness FIX
   revealing true behavior, a NEW bug, or a data/config change. If feasible, **re-run a past sensible
   configuration on current code** and vice-versa (old code on current data) to localize it.
   Specifically ask: *what made the old results peaked* — spec-z hosts? exact-z? a narrower/centered
   grid? a prior? a bug pulling to truth?
4. **Hypothesis tournament.** Enumerate ALL plausible rail causes (photo-z starvation; normalization
   bug; grid/prior/window artifact; selection-function bug; d_L→z inversion; a project-introduced
   regression; a truth/fiducial bookkeeping mismatch 0.70↔0.73; catalog cross-match bug). Design ONE
   discriminating test per hypothesis. Do **not** privilege the photo-z hypothesis.
5. **Constructive de-rail attempt (red team).** Try *by any correct means* to produce a peaked,
   non-railing posterior on the real data (wider grid; a defensible normalization; a bug fix; a
   sub-selection). Genuinely try to win. If you succeed → we missed something. If you fail after real
   effort → the irreducibility claim is strengthened.
6. **Critical re-assessment of OUR key experiments** (Evidence Locker, on request) — audit each
   harness for faithfulness before accepting its result.
7. **Synthesis / independent report.** Answer the four correctness axes + the peaked-posterior
   question; state whether our conclusions HOLD / are WRONG / are INCOMPLETE, with the decisive
   evidence.

### B4. The decisive tests (gold-standard + improvements — includes things we may have skipped)
- **D1 — End-to-end full-pipeline injection–recovery on a spec-z mock (GOLD STANDARD).** Run the
  *actual production* simulation + inference on a synthetic universe with a KNOWN H₀ and spec-z hosts
  (σ_z→0). If the real pipeline recovers H₀ with a peaked posterior → the machinery is correct and the
  issue is the photo-z data; if it rails even here → there is a pipeline bug independent of photo-z.
  *(We validated the bridge/closure and reduced closures, but a clean end-to-end recovery in the
  FULL production pipeline on spec-z is the single most decisive test — verify whether it has truly
  been done, and if not, do it.)*
- **D2 — Statistical calibration (P–P / coverage).** Over many synthetic realizations, check the
  injected H₀ falls in the X% credible interval X% of the time. This tests *statistical* correctness
  beyond a single point estimate — a standard test we likely have NOT run. A mis-calibrated posterior
  (even if it "peaks") signals a normalization/likelihood error.
- **D3 — Single-event vs stack decomposition.** Do *individual* events give sensible (wide but
  interior) posteriors, with railing appearing only in the *combination*? If so, the fault is in the
  stacking/normalization, not per-event information — a different (fixable) diagnosis than "each event
  is starved." (F4 looked at this but on the railed pipeline and via our harness — redo independently.)
- **D4 — Grid & prior stress test.** Widen the H₀ grid well beyond [0.60,0.86]; check the H₀ prior is
  truly flat; confirm the "rail" isn't a grid-edge/truncation or an implicit prior. Does the MAP move
  with the grid? (A real peak shouldn't.)
- **D5 — Rail-direction sign check.** Our own data show the rail *flips* (production up to 0.86;
  a reduced closure down to 0.60) under normalization changes. A genuine data constraint cannot flip
  sign under a normalization choice — independently verify this flip and its interpretation (it argues
  for prior/normalization domination, i.e. information starvation, but must be reproduced cleanly).
- **D6 — Truth/fiducial bookkeeping.** Independently determine the *actually injected* H₀ and confirm
  every stage (grid, selection, distance relation, `TRUE_HUBBLE_CONSTANT=0.70` vs h=0.73) is
  consistent. Rule out a bookkeeping-induced apparent bias.
- **D7 — External-code cross-check (stretch).** If installable, run an independent published dark-siren
  code (e.g. `gwcosmo`, CHIMERA) on the same events+catalogue. Agreement/disagreement is a strong
  external arbiter. (Heavy; optional but high-value.)
- **D8 — d_L→z inversion & σ² bias.** Independently reproduce the generic distance-inversion
  (σ²) bias magnitude and confirm it is small (not the rail).

### B5. Deliverable — the independent report
`.planning/INDEPENDENT-VERIFICATION-REPORT-YYYYMMDD.md`, answering:
1. Is the pipeline **physically** correct? (population priors, selection, distance–redshift, sky.)
2. Is it **statistically** correct? (likelihood normalization, calibration/coverage, combination.)
3. Is it **software** correct? (units, indexing, frames, reproducibility, no truth leakage.)
4. **Can a peaked, non-railing posterior be achieved** — with what data/config, and if not, the
   *independently derived* mechanism.
5. **Verdict on our conclusions:** HOLD / WRONG / INCOMPLETE, with the decisive test(s). Explicitly
   state which of H-past-bug-good / H-new-bug / H-physics / H-artifact is supported.
6. A ranked list of any bugs found (severity, file:line, fix), independent of our bug list.

### B6. Evidence Locker (our prior results — provide ON REQUEST, each with an audit gate)
Provide only if the investigators ask, and always with: *"audit this harness for faithfulness before
trusting it."* For each, the specific thing to check:
- **Seed-600 production run** (rails to 0.86): `results/`/workspace CRBs + posteriors. *Check:* is the
  injected H₀ what we think; is the grid wide enough; is the combination correct.
- **Bridge closure ladder** (`scripts/bridge_closure/`, rungs A–I): the σ_z sweep (rung_G) claims
  delta-z recovers 0.725 and σ_z=0.035 rails. *Check:* is `event_log_likelihood` faithful to the
  production likelihood; is the closure truly self-consistent; does the σ_z knob isolate only photo-z;
  is the candidate window/grid an artifact.
- **rung_I self-consistent closure** (rails at large σ_z): *Check:* is it genuinely normalized; does
  the sign-flip under normalization reproduce.
- **F5 σ_z/σ_M forecast** (`scripts/bridge_closure/sigma_z_sigma_M_forecast.py`): *Check:* the metric
  (RMSE-to-truth), the grid, the with-BH-mass mass term, the ≲1% floor claims.
- **The closure unit tests** (`test_partition_norm_closure.py`, `test_change5_pixel_closure.py`):
  *Check (important):* do they validate the REAL pipeline, or a simplified/spec-z stand-in? The prior
  note is that they use near-exact hosts — i.e. they may not exercise the photo-z regime at all. This
  is a prime target for critical assessment.
- **The p_sample/p_comp audit + sky-selection derivation** (`.planning/PSAMPLE-PCOMP-AUDIT-*`,
  `derivation-sky-selection/`): *Check:* the ≲1% sky-effect measurement and the reconciliation claims.

### B7. How to run it (suggested)
- **A fresh session** (clean context) with this handoff, running the investigation as a sequence of
  **workflows** whose agent prompts carry only the neutral brief (B2) + code access — NOT our
  conclusions. One workflow per phase (audit → minimal-repro → temporal → hypothesis-tournament →
  de-rail attempt → evidence-audit → synthesis), reviewing between phases.
- Use **diverse agent lenses** (physics / statistics / software / red-team) and require **majority
  independent agreement** before a conclusion is accepted.
- Keep a **decision log** of every hypothesis tested and its discriminating result.

---

## PART C — Improvements & things possibly missed (fold into B)
- **The gold-standard test (D1) may not have been run end-to-end in the REAL pipeline** — we leaned on
  the bridge + reduced closures. Prioritize D1.
- **No calibration/coverage test (D2) has been done** — arguably the strongest single check of
  statistical correctness; add it.
- **"Why were past results sensible" is under-investigated** — the temporal archaeology (D3/B3.3)
  directly targets the user's central doubt and we never did it systematically.
- **Blinding (B1)** — we never blinded the injected H₀; do it to kill confirmation bias.
- **External-code cross-check (D7)** — the ultimate independent arbiter; attempt it.
- **The closure tests may not exercise photo-z** — if they use exact/near-exact hosts, they cannot
  confirm OR deny the photo-z claim; the investigators must establish this.
- **Rail-direction sign-flip (D5)** is a powerful, under-exploited diagnostic — make it central.
- **Truth/fiducial bookkeeping (0.70 vs 0.73)** — verify before anything else; cheap and decisive if
  wrong.
- **F4 depends on Part B:** the spec-z-vs-photo-z decomposition is only meaningful once a sensible
  (peaked) posterior exists. Do NOT re-run F4 until Part B establishes whether/how a peak is
  achievable. (This session's F4 ran on the railed posterior and is therefore only suggestive.)

---

## PART D — Current state anchor (for whoever runs Part B)
- Code + data present locally: real GLADE (`master_thesis_code/galaxy_catalogue/GLADE+.txt` +
  `reduced_galaxy_catalogue.csv`, now 8-col with the retained redshift flag); seed-600 CRBs at
  `/tmp/seed600_local/simulations/`; injection pool referenced by the bridge.
- Local eval: `uv run python -m master_thesis_code <dir> --evaluate --h_value H --num_workers N`.
- Gate: `uv run ruff check …`, `uv run mypy master_thesis_code/`, `uv run pytest -m "not gpu and not
  slow"` (+ `-m slow` for closures).
- The current pipeline includes: M1 analytic rate, direct rate-weighted catalog draw, dark-event
  injection, Task-A partition-norm single Gray ratio, Change-5 pixelated completeness, the mass-relation
  error fix, and the sky-aware selection. All committed on `physics/photoz-joint-normalisation`.
- The **honest current expectation** (to be tested, not assumed by Part B): a full production re-run
  still rails, because the fixes this session are each ≲1% while the rail is +13–18%, and the driver
  (host photo-z σ_z≈0.035 ≫ GW precision) is untouched. **Part B exists precisely to challenge this.**

---

## PART E — Sequencing & how to prompt the next session

### E1. Serialize; freeze the code during the commission
Do **Part B first, alone**; do **not** run Part A pipeline fixes in parallel. Reasons: (1) the
commission must audit + reproduce against a **frozen codebase** — parallel edits make it a moving
target and muddy the temporal/regression archaeology (the key tool for the "was it a bug?" doubt);
(2) the verdict is **upstream** — if it finds a de-railing bug the priorities invert, so Part A done
now may be wasted/misframed; (3) bias hygiene — an untouched codebase keeps the audit independent.
- **Safe to parallelize (optional, low value):** pure ops that don't touch the audited branch —
  merge PR #17 to `main`, file GitHub issues. NOT the pipeline fixes; NOT the paper *results*.
- **Better use of spare capacity:** deepen the commission (add D7 external-code cross-check, more
  seeds for D2 calibration) rather than splitting focus.
- **Run it in an isolated worktree off HEAD:** `git worktree add ../verify-worktree b828ab0` — frozen
  snapshot, clean main branch, report lands in `.planning/`.

### E2. Independence is enforced at the AGENT level (memory auto-loads into the orchestrator)
A fresh session auto-loads `MEMORY.md` (our conclusions), so the orchestrator can't be fully blind —
but workflow/Task **subagents get fresh context and see only the prompt passed to them**. So the
orchestrator must pass investigator agents **only the §B2 neutral brief + code access**, never our
conclusions, and withhold comparison until synthesis.

### E3. Suggested prompt for the next session (lean; does not restate our conclusion)
> "You are running an independent verification of a LISA EMRI dark-siren H₀ pipeline. Read
> `.planning/HANDOFF-INDEPENDENT-VERIFICATION-20260701.md` and execute **Part B** (on a clean
> checkout of HEAD — no worktree needed on a fresh clone; use a worktree only if sharing the repo
> with other work). **Independence is the point:** spawn every investigator agent (via workflows)
> with **only the §B2 neutral brief + code access** — never the `.planning/`/`docs/` conclusions or
> our verdicts; treat every prior result of ours as an **untested claim to audit before use** (§B6),
> and **blind the injected H₀** to the agents where feasible (§B1). Have the agents work from first
> principles and genuinely **try to produce a peaked, non-railing posterior / find a de-railing
> bug** — i.e. falsify the (withheld) 'irreducible railing' claim. **Prioritize the
> temporal/regression archaeology — *why were past results sensible and what changed?* (§B3.3 /
> D3)** — that is the central doubt. Run the hypothesis tournament wide and use **≥2 independent
> from-scratch reproductions** for consensus (§E4). Start with **synthetic-mock tests** (repo-only);
> real-catalogue reproduction needs the copied local data and should be throttled (§E4/E5). Only in
> the final **synthesis** compare the agents' findings to our stored conclusions; deliver the report
> per §B5 (HOLD/WRONG/INCOMPLETE + decisive tests)."

**Fresh-machine bonus:** `MEMORY.md` lives in `~/.claude/`, not the repo, so on a *new* machine it is
absent and the orchestrator is naturally uncontaminated by our stored conclusions; only the in-repo
`.planning/`/`docs/` conclusions remain (guardrail handles them). **Max-blindness alternative (first
pass):** give *only* the §B2 neutral brief as the prompt, let the session form hypotheses cold, then
hand it this handoff's Evidence Locker for the deeper phases — cleanest independence, at the cost of
re-deriving some structure.

### E4. Parallelism within the commission + compute/device guidance
**Parallelize the thinking, throttle the running.**
- **More parallelism (also improves rigor via consensus):** fan the **hypothesis tournament** wide
  (one agent per candidate cause: photo-z; normalization bug; grid/prior artifact; selection bug;
  project-introduced regression; 0.70↔0.73 bookkeeping; cross-match bug; d_L→z inversion — ~8–12
  agents, one discriminating test each); run **2–3 *independent* minimal reproductions** (different
  agents build from-scratch estimators, blind to each other — all railing on photo-z is far stronger
  than one); **more audit lenses** (numerics/units, frames/coordinates, GPU/array-semantics on top of
  physics/stats/software); parallelize the **evidence-locker audits** (one agent per prior experiment)
  and the three minimal-repro cases (spec-z/photo-z/real) once the estimator exists.
- **Throttle the compute-heavy runs:** each real-catalogue eval loads the ~22.6M-row catalogue
  (~3–5 GB RAM). Fanning many at once will OOM. Cap real-catalogue runs to ~2–4 concurrent (or
  serialize); keep the many *synthetic-mock* runs (light) fully parallel.

### E5. Data portability (running the commission on another non-cluster machine)
- **Portable via git (nothing to copy):** all code/tests/docs + the committed
  `master_thesis_code/galaxy_catalogue/m_th_map_nside32.npy` (sole `f` source).
- **The decisive/synthetic-mock work needs ONLY the git repo + `uv sync`** — fully portable.
- **Local-only (NOT in git) — copy only if reproducing/auditing the REAL results:**
  - `/tmp/seed600_local/simulations/` (**~991 MB**: seed-600 CRBs + the 560-CSV / 504k injection
    pool) — **NOT regenerable** (GPU-sim output). **Must copy.** Then repoint the repo-root
    `simulations` symlink to wherever it lands.
  - `GLADE+.txt` (6.0 GB) — copy or re-download (`elysium.elte.hu/~dalyag/GLADE+.txt`).
  - `reduced_galaxy_catalogue.csv` (1.6 GB) — do NOT copy; auto-rebuilds from GLADE+.txt (~2 min).
- **Cluster-only:** a fresh GPU EMRI simulation.

**Device:** THIS dev box (CPU-only) suffices for the DECISIVE parts — the railing lives in the CPU
**inference**, so the audit, from-scratch reproductions, injection–recovery, and calibration all run
on **synthetic mocks / synthetic CRBs** (inject known d_L/M + Gaussian errors; skip the waveform):
pure CPU + numpy, moderate RAM, fully parallel. **Cluster needed ONLY for:** (1) a *true* full-pipeline
end-to-end run — the EMRI **simulation** is GPU (`few`); the inference-side D1 (synthetic CRBs → real
inference) is the important discriminator and is CPU-feasible; (2) fast heavy parallel evals over the
real 22.6M-row catalogue. A higher-RAM/more-core machine is a nice-to-have for real-catalogue
throughput, **not required** (throttle instead); it cannot substitute for the GPU sim.
