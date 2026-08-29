# PRE-REGISTRATION — [HIER] hierarchical photo-z self-calibration: the (h, θ)-grid experiment

**Date:** 2026-08-26 · **Thread:** `[HIER]` · **Stage:** research-cycle stage 2
(pre-registration), entered from stage 0/1 assets `PROPOSAL_HIER_SELFCAL_20260825.md`
(RATIFIED, author grant row #195 verbatim: *"all approved, the new finding is huge, lets see
what the verification agent returns with."*), `STAGE_L_HIER_20260825.md` (row #193) and
`STAGE_L_HIER_V86_READING_20260825.md` (**stage-L reading obligation DISCHARGED**, row #195
item 2).
**Orchestrator-autonomous drafting; `[OPUS-ORCH 2026-08-26]` tags bind. Append-only after
commit; amendments are `PA-HIER-<n>` below the divider and nothing above the divider may be
edited.** A21 governs. The C-A governance stack (PA-CA-1/7/8/10/11 conventions, out-root guard,
resolved-flag A22 stamps, `[ORCH-*]` tag discipline) is INHERITED wholesale from
`PREREGISTRATION_P3_2D_20260825.md`; only `[HIER]`-specific items are registered here.

**Provenance-gating (rule 4) — what upstream gate made this test necessary.** Row #137
(2026-08-20) stated the dark-class score-at-truth as a defect (**−0.635 ± 0.017**, iiib, 37σ;
class scope: 605/1588 events) and localized it as z-structured. Row #192 (2026-08-25) opened
`[HIER]` with the author's own scope fence, restated verbatim in §1.4. Row #193 banked the
reportable absence. Rows #198–#211 parked `[P3-2D]` at UNATTRIBUTED-bounded, which is what
frees the cluster for this thread under D5.

---

## 1. Hypothesis and identity

### 1.1 The registered hypothesis

**[HYPOTHESIS]** Part of the N-coherent ensemble bias is *error-model mis-specification*: the
per-event Gaussian photo-z kernel evaluated at the catalogue-quoted `σ_z` mis-states the true
error law (bias curve and/or scatter scale), and a hierarchical layer that infers a
low-dimensional error-model **θ** JOINTLY with h across the ensemble would (a) absorb the
z-structured tilt into θ and (b) yield calibrated (coverage-passing) posteriors, at the price
of honest width.

**Refute by (rule 3):** on the mirror venue, where truth-θ is known by construction, the joint
(h, θ) posterior evaluated at truth-θ still rails / coverage still fails **and** the recovery
control (Arm R, §2.2) fails to recover a *deliberately injected, known* θ-misspecification —
then the coherence lever is dead in our regime and the thread closes with a documented bound.

### 1.2 The registered parameterization (identity of θ)

θ = (b, s), applied to the per-host redshift kernel only:

```
bias slope      :  z_centre  ->  z_centre + b·(1 + z_centre)
scatter scale   :  σ_z,raw   ->  s · σ_z,raw          (RAW catalogue column, BEFORE the
                                                       peculiar-velocity quadrature sum)
```

Registered arithmetic form, pinned to remove the recon's §5/finding-9 ambiguity — `s` scales
the *catalogue's quoted* error (the quantity whose misstatement is the hypothesis), never the
peculiar-velocity term (a separate physical contribution the hypothesis says nothing about):

```
sigma_z_pv       = (1 + z_centre) · SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S
host_z_error_eff = sqrt( (s · host_z_error)**2 + sigma_z_pv**2 )
```

This is currently *unobservable* — `SIGMA_V_PEC_KM_S = 0.0` (`darksiren_emri/constants.py:95`,
verified 2026-08-26) makes the two candidate placements numerically identical — and is pinned
here precisely because it stops being a no-op the moment that constant is ever set.
**Truth-θ ≡ (b, s) = (0, 1).**

### 1.3 Field position, and what may NOT be claimed

The [86] reading obligation is **DISCHARGED** (`STAGE_L_HIER_V86_READING_20260825.md`). Its
binding finding for this prereg: *"This paper contains no photo-z error model, no
galaxy-catalogue completeness correction, and no per-galaxy redshift-error kernel of any kind —
the entire σ_z machinery our (h, θ)-grid experiment is built around is absent from [86]"*
(§4, that file). **Consequence, registered:** only the *joint-inference pattern* is borrowed;
the (h, θ) construction, the truth-θ = (0,1) mirror-venue design and the registered reads are
**original to this thread and may NOT be presented, in any output of this thread, as
literature-validated.** Hanselman+ 2024 §IV.5 names the direction (quote-verified, row #193);
nobody has built it; no small-N validity statement exists anywhere in the surveyed literature.

### 1.4 Scope fence (author's own words, row #192, restated verbatim as required)

> "interpretation-layer coherence — shared photo-z error-model hyperparameters + shared latent
> z of overlapping candidates — NOT the LISA global-fit data-stream problem; events stay
> physically and measurement-independent."

This prereg registers **only** the shared-hyperparameter half (θ). The shared-latent-z half of
row #192's fence is **out of scope here** and needs its own registration.

### 1.5 STANDING scope guard (proposal §5 item 3, restated as binding text)

**No production kernel change issues from this thread without its own `/physics-change` gate.**
Every θ hook is an *instrumentation* flag with a byte-identical default (GATE T-ID, §3.1).
A CALIBRATED+narrow verdict (§4.5) authorizes a `/physics-change` *proposal*, never an edit.

### 1.6 Rule-1 exoneration delimitation (both layers checked; source: `hier_provenance_stamps_20260826.md` §1)

Ledger §2 ("DO NOT RE-TRY", `:127–169`) and §3 (`:172–207`) plus all 12 local `CLAIM_*.md`
files were swept. **Verdict: PASSED, with one conditional dependency**, registered here so the
condition is visible rather than inherited silently:

- **The one adjacent standing ruling is C7 / G2b** (ledger `:197`): `w_pop = (dV_c/dz)/(1+z)`
  is CONFIRMED as *"the unique weight consistent with the project's own rate model and with
  every selection integral"*, exactly h-independent
  (`docs/derivations/G2b_host_z_volume_prior.md:413-436`).
- **Delimitation:** G2b's ratified claim is about the **functional FORM** of `w_pop` — unique
  *given* any σ_z — and does not depend on σ_z's value. θ's (b, s) acts on the **separate
  measurement-error kernel** convolved *against* `w_pop`, not on `w_pop` itself. On that reading
  the grid does not re-open G2b/C7.
- **The delimitation is CONDITIONAL on GATE D3 (§3.2).** If the pre-launch check finds that θ's
  `s` is the same knob as `w_pop`'s own σ_z-dependence rather than a distinct measurement-error
  axis, the grid *would* be sweeping the quantity G2b ratified, and the delimitation must be
  re-derived and re-reviewed before any node runs. This is an exoneration-side reason for D3's
  blocking status, independent of the code-hygiene reason.
- **Adjacent prior art, named so it is not mistaken for re-litigation:** ledger #68/#62 — the
  deconvolution **over-corrects at σ_z/z ~ O(1)**. That is a ratified finding about the
  *current, fixed* kernel; θ instead treats the kernel's own parameters as unknowns. Different
  question. The registered `s` grid (§2.3) deliberately straddles that regime boundary.
- **Vocabulary separation, registered:** ledger §2 item 13 (`:151`) OVERTURNED "information
  starvation" *as the explanation for the observed H₀ rail* (#41/#52: *"a property of
  prior-INCONSISTENT estimators, not of the data"*). F5 (§7, §4.4) is a **forecast of achievable
  precision under a correctly-specified estimator** — an orthogonal, forward-looking question.
  **F5's "information-starved" language may NOT be cited anywhere in this thread as if it
  supported the overturned #41/#52 claim.**

### 1.7 Registered design decisions `[OPUS-ORCH 2026-08-26]`

Each row is registered **as issued**; where this prereg refines a decision the refinement is in
its own column and is flagged for the review chain. The review chain may challenge any row but
must do so explicitly, by tag.

| tag | as issued | as registered here |
|---|---|---|
| **D1** `[OPUS-ORCH 2026-08-26]` | Two-stage pilot-then-fleet, NOT a single 25-node commitment. **Stage P** = coarse 3×3 θ-grid × 4 mirror seeds. **Stage F** = full 5×5 θ-grid × 12 mirror seeds, launched ONLY if the pilot's registered gate passes. Refinement beyond 5×5 is OUT of scope and needs its own registration. | Adopted verbatim, **plus a Stage 0** (D7's read, §2.1) that precedes Stage P and can terminate the thread before any grid runs. Stage-F `n_h` is deliberately NOT pinned at registration; it is set by GATE P→F (§3.6). |
| **D2** `[OPUS-ORCH 2026-08-26]` | θ's (b, s) transform applies ONLY to the DEFAULT production host-z kernel variant (the configuration-of-record). The variants at `bayesian_statistics.py:6249-6277` — `"global"` / `"local_ratio"` / `"volume_deconv"` — are OUT of scope and MUST be named as a structural-blindness item under A10. | Adopted, **with the fourth clause the recon showed D2 requires** (§2.4): "the default production host-z kernel variant" is ambiguous and, read literally, selects the one mode in which `s` is structurally inert. The configuration-of-record is pinned in §2.4 as **CoR-M** (mirror venue, Stages 0/P/F) and **CoR-P** (production venue, the deferred S0-B read). Out-of-scope variants are §5.2 item (i). **AUTHORED RULING — flagged for the review chain.** |
| **D3** `[OPUS-ORCH 2026-08-26]` | Whether θ's `s` generalizes or collides with the existing `smear_sigma_z` flag is a PRE-LAUNCH BLOCKING instrument check (the PA-2D-6 stale-flag pattern). Must be resolved in code before any array runs. | Adopted verbatim. Recon verdict **GENERALIZATION**, conditional on three implementation clauses (a)(b)(c) registered as **GATE D3**, §3.2. Blocking status unchanged; §1.6 adds an exoneration-side reason for it. |
| **D4** `[OPUS-ORCH 2026-08-26]` | 1-D SLURM array over a FLATTENED (θ-node, seed) index with an explicit decode in the driver, reusing the `BASE_SEED + SLURM_ARRAY_TASK_ID` template of `cluster/p3_2d_fleet.sbatch`. No new 2-D indexing scheme. | Adopted verbatim; decode and the collision-free-by-construction argument in §7.3, including why the deliberate seed-repetition across θ-nodes is **not** a recurrence of PA-2D-8/F8. |
| **D5** `[OPUS-ORCH 2026-08-26]` | Prereg authoring proceeds in parallel with `[P3-MKER]` stage-1; the CLUSTER ARRAY launches only after `[P3-MKER]` stage-1 is banked — one thread on the cluster at a time. | Adopted verbatim and registered as **GATE SEQ** (§3.7): no `sbatch` for any [HIER] stage until `[P3-MKER]` stage-1 is banked with a ledger row. |
| **D6** `[OPUS-ORCH 2026-08-26]` | The proposal's "50–100 CPU-h" line is STALE by rule [A11] and must be re-derived from the measured per-h-value timing anchors, not copied. | Adopted; re-derivation in §7, sourced to `hier_costing_20260826.md`. The stale line is **superseded and may not be quoted**. Root cause of the staleness, registered: it never priced the h-sweep that the proposal's own §2 requires. |
| **D7** `[OPUS-ORCH 2026-08-26]` (rule 6 / [A12] rule 13) | The score-at-truth-θ test (`E[∂_θ ln L] = 0` at truth for model-consistent data) is a STAGE-0 zero-compute-or-cheap read that runs BEFORE the pilot. If the score at truth-θ is already consistent with zero, the error-model lever is dead on its own terms and the expensive grid is not warranted — register that early-exit branch explicitly with its band. | Adopted, **split by venue** (§2.1): on the *self-consistent mirror venue* the θ-score at truth is **zero by construction**, so that read is a CONTROL (an expected null, rule 4) and cannot carry the early-exit; the early-exit is carried by **S0-R** (the recovery control, mirror venue with an injected known misspecification) and, if the author grants its costing, **S0-B** (production venue). Bands in §4.1. **AUTHORED REFINEMENT — the single loudest flag in this document; see §8 of the return note.** |

---

## 2. Arms, venue, and instruments

### 2.1 Stage 0 — the D7 measurement-before-gate reads (run FIRST, can end the thread)

| arm | venue / construction | what it measures | registered expectation |
|---|---|---|---|
| **S0-A** (control) | mirror venue, generator kernel = estimator kernel, truth-θ = (0,1) by construction | θ-score at truth-θ, 4 seeds × the 5-node θ-cross × `n_h = 1` (h = 0.73) | **EXPECTED NULL.** `\|Z_b\| ≤ 3.0` and `\|Z_s\| ≤ 3.0`. A fail is an INSTRUMENT DEFECT, not a discovery (§4.1, §6). |
| **S0-R** (recovery control / positive control) | mirror venue realized at `realize_observed_catalogue(sigma_scale = 1.5)` (`galaxy_catalogue/observed_realization.py:167-184`) — the catalogue's quoted `z_error` column is copied unchanged while the realized scatter is 1.5×, i.e. a KNOWN misspecification with truth-θ = (0, 1.5) | (i) θ-score at (0,1) — must be non-zero with the registered sign; (ii) the grid's recovered ŝ | **REGISTERED PREDICTION:** `Z_s` at θ=(0,1) is negative and `\|Z_s\| ≥ 5`; the Stage-P/F grid peaks at ŝ within band of 1.5 (§4.1). |
| **S0-C** (costing probe) | mirror venue, 1 cell at `n_h = 41` | the MEASURED marginal per-h cost (the §7 anchor prices every h-point at the full single-h wall time — an upper bound) | reported, feeds GATE P→F's re-costing leg (§3.6). Not band-bearing. |
| **S0-B** (decisive production read — **DEFERRED, author costing gate**) | production venue at CoR-P, the banked realistic scattered catalogue | θ-score at truth-θ where the measured tilt actually lives | **NOT AUTHORIZED at registration.** ~75–101 CPU-h (§7.2). Runs only on an explicit author costing grant, and only after S0-A/S0-R land. |

**Why S0-A cannot carry the early-exit (the D7 refinement).** On the mirror venue the estimator's
kernel *is* the generating kernel at θ = (0,1); `E[∂_θ ln L] = 0` at truth is then a theorem
about the construction, not a measurement about the world. A null there refutes nothing about
the error-model lever — it only certifies the instrument. The lever question ("does the data
pull on θ at all, and enough to matter?") is answerable only where the kernel may be wrong:
S0-R (a known, injected wrongness) or S0-B (the real venue). Registering S0-A as the early-exit
read would have manufactured a guaranteed "lever is dead" verdict.

**Disclosed defect in the S0-R instrument, registered rather than patched silently:**
`realize_observed_catalogue`'s `sigma_scale` scales **both** `z_obs` and `ln M_BH,obs`
(`observed_realization.py:178-183`) — it is not a pure z-knob. S0-R therefore injects a joint
z+mass misspecification. Registered handling: S0-R is scored in the **no-BH channel only**
unless a z-only scale knob is added first; if the with-BH channel is scored, the mass-side
misspecification must be reported alongside and no θ attribution banks from it.

### 2.2 Stages P and F — the grid arms

| arm | what | runs |
|---|---|---|
| **P-GRID** | per-event `L(h, θ)` cubes on the 3×3 θ-grid × 4 mirror seeds at `n_h = 41` (`H_GRID_41`) | 36 cluster-array tasks, `--array=0-35` |
| **F-GRID** | the 5×5 θ-grid × 12 mirror seeds; `n_h` set by GATE P→F | 300 tasks as 5 sub-arrays of 60 (§7.3) — **NOT AUTHORIZED at registration** |
| **R-GRID** (control) | S0-R's injected-misspecification venue re-scored on the same grid the primary arm uses | folded into P-GRID as a second venue label, same tasks' machinery, separate out-root |

Every arm writes its own out-root under the PA-CA-11 out-root guard; no arm writes into
another's directory.

### 2.3 The registered θ grid (derived, with anchors shown)

```
b (bias slope, Δz = b·(1+z))
  Stage F : { -0.04, -0.02,  0.00, +0.02, +0.04 }
  Stage P : { -0.04,         0.00,        +0.04 }
s (scatter scale, σ_z -> s·σ_z), log-uniform in ×√2 steps
  Stage F : {  0.50, 0.7071,  1.00, 1.4142, 2.00 }
  Stage P : {  0.50,          1.00,         2.00 }
Truth-θ = (0.00, 1.00) is an exact grid node in BOTH stages (Stage F flat index 12,
Stage P flat index 4, row-major theta_idx = b_idx·N_S + s_idx).
Stage-0 θ-cross = { (0,1), (±0.02, 1), (0, 1/√2), (0, √2) } — all five are Stage-F grid nodes,
so nothing computed at Stage 0 is thrown away.
```

**Anchor for b's half-width 0.04.** The catalogue's own quoted photo-z scatter,
`σ_z = 0.035` (`darksiren_emri/validation/pp_coverage.py:630`, `PPCoverageConfig.sigma_z`), at
the campaign's measured mean host redshift `z ≈ 0.485` (ledger `:1355`, row #137 item 4):
`b_max = 2·0.035 / (1 + 0.485) = 0.0471`, rounded **down** to 0.04 (conservative, and gives a
clean 0.02 step). So the grid spans ±2 catalogue-σ_z of bias at the median z.

**Anchor for s's half-width, factor 2.** `[ORCH-ANCHOR, convention]` A factor-2 misstatement of
a quoted photometric scatter is the registered plausible-range convention; log-spacing makes
truth-θ an exact node and makes the grid symmetric in the natural (multiplicative) parameter.
Second, independent property, disclosed as a deliberate feature: at the campaign's measured
`σ_z/z` band of **0.25–0.6** (see §4.4 — this is a band, not a point, by [A11]), the `s = 2`
node lands at `σ_z/z ≈ 0.5–1.2`, straddling the boundary of the ledger's ratified
"deconvolution over-corrects at σ_z/z ~ O(1)" regime (#68/#62). The grid is *meant* to cross
that boundary; §1.6 registers the adjacency so a reviewer does not read it as re-litigation.

**Freeze rule:** both grids may only **tighten** post-data (a sub-grid refinement inside the
registered range is a reportable read); **widening either range requires a `PA-HIER` amendment
before the widened nodes are scored**, and any node outside the registered range is
REPORTED-ONLY and never band-bearing.

### 2.4 Instruments and their dispatch paths (committed before running; all five)

The recon (`hier_instrument_recon_20260826.md`) enumerated **five** independent sites that
compute a host-z kernel width, not the two (scalar/batch) the design assumed. Every one that is
in scope must receive its own θ hook; every one out of scope must be named in §5.2.

| # | site | file:line | in scope? | θ hook required |
|---|---|---|---|---|
| 2.1 | `single_host_likelihood` — **SCALAR** width | `bayesian_statistics.py:6223-6224` | YES | `s` (§1.2 form) |
| | scalar denominator z-clamp | `bayesian_statistics.py:6242-6244` | YES | `b` (window re-centres) + `s` |
| | scalar quadrature numerator centre | `bayesian_statistics.py:6247` (`norm(loc=host_z, scale=host_z_error_eff)`) | YES | `b` |
| | scalar point-kernel numerator, no-BH | `bayesian_statistics.py:6396-6416` (`_z_point = host_z`, `:6401`) | YES | `b` only (no σ_z term exists on this branch) |
| | scalar point-kernel numerator, with-BH | `bayesian_statistics.py:6652-6692` (`:6661`) | YES | `b` only |
| 2.2 | `single_host_likelihood_batch` — **BATCH** width (**this is what production dispatches**; the scalar function has no other caller in `darksiren_emri/`) | `bayesian_statistics.py:6878-6879` | YES | `s` |
| | batch denominator z-clamp | `bayesian_statistics.py:6899-6901` | YES | `b` + `s` |
| | batch kernel fork mirror | `bayesian_statistics.py:6903` | YES | assertion only |
| 2.3 | `precompute_global_catalog_selection` → `_smeared_global_pdet_expectation` — the **GLOBAL selection denominator's own independent copy** of the width formula (`sigma_eff`, `:1672`; quadrature centre `zc`, `:1679`) | `bayesian_statistics.py:2657-2882`, `:1619-1720`; call sites `:4010`, `:4018`, `:4062` | YES (under CoR-P) | `s` **and** `b` — GATE D3 clause (c): **no `b` analog exists here at all today** |
| 2.4 | `host_z_error_eff()` — the **MIRROR-VENUE harness's own copy**, used by every C-A/P3-2D/P3-TWIN/[HIER] fleet task; a hook added only inside `bayesian_statistics.py` does not reach it | `validation/correspondence_1d.py:1167-1199`, called `:1323`, `:1485` | **YES — this is where Stages 0/P/F actually execute** | `s` + `b`, independently maintained |
| 2.5 | `pp_coverage.py` (G4b) — a **fully independent reimplementation** with zero import of `bayesian_statistics`, its own `sigma_z = 0.035` (`:630`) and `kernel: Literal["bare","volume"]` (`:641`) | `validation/pp_coverage.py:504-856` | **NO — OUT OF SCOPE** (§5.2 item iv) | none |
| 2.6 | CLI surface | `arguments.py:699` (`--host_z_kernel`), `:736` (`--normalization_mode`), `:773-786` (`--smear_global_selection`) | YES | `--theta_b` / `--theta_s` registered here with the GATE D3 incompatibility guard |

**Registered departure from the proposal, flagged.** Proposal §2 registers read (ii) as
"coverage/P–P over seeds at the h-marginal", which reads as an appeal to `pp_coverage.py`. This
prereg executes read (ii) instead from the **mirror fleet's own per-seed h-posteriors**
(`correspondence_1d.compute_seed_statistics`, `compute_full_log_posterior_vector`, `:3788-3789`)
— same venue as Stages P/F, no fifth reimplementation, no venue mismatch inside one verdict.
`pp_coverage.py` gaining its own θ parameterization is named as future work and as a structural
blindness (§5.2 iv), not silently assumed.

**D2's fourth clause — the configuration of record, pinned.** `resolve_host_z_kernel`
(`bayesian_statistics.py:166-213`) returns `"point"` under the literal class/CLI default
(`normalization_mode="generator_marginal"`, `host_z_kernel="auto"`;
`arguments.py:699-702,745-748`), and on the point branch the numerator carries **no σ_z term at
all** (`:6396-6416`, `:6652-6692`, governing comment `:6191-6205`) while the one quantity that
does carry σ_z — the per-host `D_g` — is computed and then **discarded** by the
`generator_marginal` assembly (`:5069-5091`, comment `:6267-6269`: *"the assembly never divides
by it"*). Under that default **`s` is structurally inert and `b` alone is live.** Therefore:

- **CoR-M (Stages 0, P, F — the venue that actually runs):** the `correspondence_1d` mirror
  harness's own quadrature kernel and window (`:1167-1199`), at its production-tracking
  defaults, with θ hooked into that copy. `s` is live here natively.
- **CoR-P (the deferred S0-B production read):** `normalization_mode = "absolute_marginal"`,
  `host_z_kernel = "volume_deconv"` — the pair `main.py:130-137` logs as the recommended eval
  flags for an observed/scattered catalogue, and the **only legal pair** on a scattered
  catalogue (`validate_scatter_guards`, `bayesian_statistics.py:3864-3878`, raises `ValueError`
  for the point kernel) — with `smear_global_selection` forced True per GATE D3(a).
- **`generator_marginal` is REFUSED for any `s ≠ 1`** (GATE D3(b)), mirroring the existing
  unconditional incompatibility guard at `bayesian_statistics.py:3849-3858`.

### 2.5 A22 resolved-flag stamps — ELEVEN

The six inherited from PA-2D-4/PA-2D-6 (`catalogue_numerator_survival="phi"`,
`selection_in_completion_numerator="fused"`, `mass_filter_sigma="symmetric"`, plus the three
already in the C-A set) **plus five new**: `host_z_kernel`, `normalization_mode`,
`smear_global_selection`, `theta_b`, `theta_s`. Every task JSON stamps all eleven at their
**runtime-resolved** values (not their nominal defaults — the PA-2D-6 stale-pin lesson), and
GATE STAMP (§3.8) fails the task if any stamp is absent.

---

## 3. Gates

Ordered. A21 STOP semantics throughout: a failed gate halts, banks nothing, and returns to the
author with frozen numbers. Gates 3.1–3.5 and 3.7–3.8 are **pre-launch**; nothing is submitted
until all of them pass.

### 3.1 GATE T-ID — truth-θ byte-identity (the [A13] dispatch-path gate, decisive)

At θ = (0, 1) every in-scope dispatch path must reproduce the pre-θ banked per-event `ln L`
**bit-for-bit**: `max |Δ ln L| = 0.0 exactly (0 ULP)`, over all events, on each of sites 2.1,
2.2, 2.3 and 2.4 **independently**. Any non-zero difference is a STOP.

**Registered implementation requirement (not a preference):** byte-identity is guaranteed by a
**literal early-return/skip at (b, s) == (0.0, 1.0)**, never by a "reduces mathematically to the
identity" argument. Rationale (recon §5 item 1–2): `host_z_error * 1.0` is exact in IEEE 754,
but a reordered `np.sqrt`/`**2` chain (e.g. `sqrt(host_z_error**2 * s**2 + pv**2)` vs
`sqrt((host_z_error*s)**2 + pv**2)`) is not guaranteed to reproduce the original expression's
rounding. A regression test asserting bit-identity against the pre-θ banked CSV ships with the
hook.

### 3.2 GATE D3 — the `smear_sigma_z` reconciliation (D1's blocking pre-launch check)

Verdict from the recon: `smear_sigma_z` (`bayesian_statistics.py:2664`; CLI
`--smear_global_selection`, `arguments.py:773-786`) is a Boolean "`s = 1`, engaged-or-not"
special case of θ's `s`, confined to the global selection denominator ⇒ **GENERALIZATION**.
The gate passes only if the implementation satisfies all three clauses:

- **(a)** `s ≠ 1` **forces** the `smear_sigma_z=True` branch itself, rather than depending on a
  separately-set `--smear_global_selection` (which defaults to **False**, `arguments.py:781`).
  Without this, `s` is live on the numerator and silently discarded on the denominator — a
  mixed-arm result **by default, not by misconfiguration**.
- **(b)** any `s ≠ 1` under `normalization_mode="generator_marginal"` **raises**, mirroring
  `bayesian_statistics.py:3849-3858`.
- **(c)** `b` is threaded into `_smeared_global_pdet_expectation`'s quadrature centre
  (`zc`, `:1679`, where **no `b` analog exists today**) — **or** it is explicitly documented in
  a `PA-HIER` amendment that `b` is deliberately not applied to the global denominator, as a
  design choice with its stated consequence, never as an inherited oversight.

**GATE D3 also discharges §1.6's condition:** the check must state, in its banked output,
whether θ's `s` touches `w_pop`'s own σ_z-dependence. If it does, the G2b delimitation is void
and this prereg returns to the author before any node runs.

### 3.3 GATE PARITY — the NEVER-audited load-bearing invariant, audited this cycle ([A10])

`correspondence_1d.py:1173` claims *"byte-identical functional form to production's per-host
sigma (`bayesian_statistics.py:5908-5909`)"* — but the current production sites are
`:6223-6224` / `:6878-6879`. **The comment's own anchor is stale, so the parity claim has not
been re-verified since at least one renumbering**, and every mirror-venue byte-identity claim
(including GATE T-ID at site 2.4) rests on it. Registered check, zero-compute: re-diff the two
expressions; then evaluate both on 10³ random `(host_z, host_z_error)` draws and require
`max |Δ| = 0.0` exactly. Fix the stale line cite in the same pass. A parity failure is a STOP
and a finding in its own right.

### 3.4 GATE ENG — engagement ([A13], rule 14)

At each Stage-P corner θ-node, **≥10% of scored events move by ≥1e-6 relative** in per-event
`ln L` versus truth-θ, measured **independently on each in-scope dispatch path**. Plus: every
task asserts its own runtime `(theta_b, theta_s)` and echoes them into its output JSON (the A22
stamp, §2.5). A null from any arm is uninterpretable until GATE ENG passes for that arm.

### 3.5 GATE TABLE-FRESH — one `BayesianStatistics` per θ node

The global precompute tables (`_D_h_table`, `_beta_G_table`, `_beta_Gbar_table`,
`_global_cat_denom_no_bh`, `_global_cat_denom_with_bh`, `_V_f_table`;
`bayesian_statistics.py:3987-4360`) are `dict[h -> value]`, built **once per construction /
`evaluate()` entry and keyed by h only, never by θ**. Registered invariant, stated in the driver
rather than assumed: **exactly one `BayesianStatistics` construction per θ node**; an instance
is never swept across a θ axis in-process. The D4 one-task-per-(θ,seed) decode makes this the
natural case, but the driver asserts it and stamps the construction's θ into the task JSON.

### 3.6 GATE P→F — the pilot's launch gate for Stage F (three legs, all must pass)

1. **Identifiability leg:** the pooled ensemble `Δ ln L` between truth-θ and the Stage-P grid
   corner, median over the 4 seeds, is `≥ 3.00` nats (§4.2). `≤ 1.15` nats ⇒ UNIDENTIFIABLE
   and Stage F is **not launched** (its own verdict, §4.5).
2. **Recovery leg:** S0-R/R-GRID recovers the injected `s_gen = 1.5` within band (§4.1). A
   recovery failure means the instrument cannot detect a *known* misspecification at this N, so
   a Stage-F null would be uninterpretable ⇒ no launch.
3. **Re-costing leg ([A11]/D6):** Stage F's `n_h` and CPU-h are re-derived from S0-C's
   **measured** marginal per-h cost, not from §7's conservative anchor, and Stage F's h-grid is
   the smallest sub-grid of `H_GRID_41` that reproduces the pilot's per-seed `mean_h` and
   `σ_h` to within `ε_h = 0.005` (§4.3). Stage F launches only under a **fresh costing line
   granted by the author** (the PA-2D-8/F15 precedent) — this prereg does not authorize it.

### 3.7 GATE SEQ — D5 sequencing

No `sbatch` for any [HIER] stage until `[P3-MKER]` stage-1 is banked with its ledger row. The
`/cluster` preflight must return `VERDICT: READY ✓`, and the same preflight queries the live
`MaxArraySize` / `MaxSubmitJobs` for `cpu_il` (**NOT FOUND** anywhere in the repo — §7.3).

### 3.8 GATE STAMP / GATE SEEDS / GATE SCATTER-PAIR

- **STAMP:** all eleven A22 flags present at runtime-resolved values in every task JSON, else
  the task fails.
- **SEEDS:** a full disjoint-seed-range audit before commit. `BASE_SEED_HIER = 940001` is a
  **proposal** checked against the ranges one grep sweep found (900101–900124 P3-2D;
  960001/970001 `ca_rhs_scorer.py:296-297`; 980001–~983104 RHS2; ≥990001 F10C) — it is not a
  substitute for the audit.
- **SCATTER-PAIR:** the *resolved* `(normalization_mode, host_z_kernel)` pair is recorded per
  task and must be **identical across every stage**. `validate_scatter_guards` is
  catalogue-conditional, so the same literal CLI flags can resolve differently against a
  scattered vs. unscattered catalogue (recon §5 item 4). A difference is a STOP unless
  registered by amendment with its reason.

---

## 4. Bands and verdict map

Every band is numeric, has its anchor's derivation shown, and carries the freeze rule: **a band
may only TIGHTEN post-data; widening requires a `PA-HIER` amendment recorded before the widened
comparison is made.** Bands stated in standardized (σ) units are unit-free by construction and
do not require an a-priori scale estimate.

### 4.1 D7 — the score-at-truth-θ bands (the registered early-exit)

Statistic: the **grid-step secant score**, per component, pooled over the arm's events and
seeds, with its SEM from the per-event scatter:

```
score_b = [ lnL(b=+0.02, s=1) - lnL(b=-0.02, s=1) ] / 0.04
score_s = [ lnL(b=0, s=√2)   - lnL(b=0, s=1/√2) ] / (√2 - 1/√2)     (denominator 0.70711)
Z_x     = mean(score_x) / SEM(score_x)
```

| # | branch | band | disposition |
|---|---|---|---|
| **B0-A** | S0-A control (mirror, truth-θ by construction) | `\|Z_b\| ≤ 3.0` **and** `\|Z_s\| ≤ 3.0` | expected null ⇒ instrument certified, proceed |
| **B0-A′** | S0-A control fails | `\|Z_b\| > 3.0` or `\|Z_s\| > 3.0` | **INSTRUMENT-DEFECT** (§4.5) — a non-zero score on a self-consistent venue is a bug in the hook, the venue, or GATE PARITY. STOP. |
| **B0-R** | S0-R recovery (injected `s_gen = 1.5`) | `Z_s < 0` **and** `\|Z_s\| ≥ 5.0`; and the grid's `ŝ` satisfies `\|ŝ - 1.5\| ≤ 0.35` (one Stage-F `s` grid step at that point: `1.5·(√2-1)/√2 ≈ 0.44`, tightened to 0.35) | **LEVER-LIVE.** Proceed to Stage P. |
| **B0-R′** | S0-R null | `\|Z_s\| ≤ 3.0` while GATE ENG passes | **LEVER-DEAD-AT-N** (§4.5): the instrument cannot see a factor-1.5 error-model misstatement at this N ⇒ the expensive grid is not warranted. **This is D7's registered early exit.** |
| **B0-P** | power leg (makes any null interpretable) | the implied 1σ_θ from the corner curvature must be **smaller than the grid half-width** (`σ_b < 0.04`, `σ_ln s < ln 2`) | if violated: **UNPOWERED** ⇒ A21 STOP, no early-exit claim, return to the author with frozen numbers |
| **B0-M** | materiality | if `\|Z\| > 3.0` but the implied best-fit displacement is `< 0.5` grid step (`\|b̂\| < 0.01` and `\|ln ŝ\| < 0.5·ln√2`) | **MIXED** (§4.5): lever live but immaterial at N |

**Anchor derivation for the 3.0 threshold:** this repo's own registered significance threshold
for a coherent class displacement — `.claude/skills/research-cycle/SKILL.md`, stage-5 decision
table: *"DEFECT (≥3σ coherent class displacement, or coverage failure)"*. It is not chosen for
this prereg. The 5.0 threshold on the *positive control* is deliberately stricter than the
detection threshold so that a recovery pass cannot be a marginal fluctuation.
**Anchor derivation for B0-B (the deferred production read):** identical bands, applied to S0-B.

### 4.2 Identifiability band (registered read iv — the answer the literature lacks)

Statistic: pooled ensemble `Δ ln L` between truth-θ and the registered grid corner, per seed,
median over seeds.

| band | value | anchor |
|---|---|---|
| **IDENTIFIABLE** | `Δ ln L ≥ 3.00` nats | Wilks, 2 parameters, 95% joint contour: `χ²₂(0.95)/2 = 2.9957` — the grid half-width exceeds the 95% contour, i.e. θ is constrained inside the registered range |
| **UNIDENTIFIABLE** | `Δ ln L ≤ 1.15` nats | `χ²₂(0.683)/2 = 1.1479` — the whole grid half-width sits inside the 68.3% joint contour, i.e. the ensemble does not constrain θ at this N over this range |
| **MIXED** | `1.15 < Δ ln L < 3.00` | by construction the first-class Mixed branch on this axis |

Secondary, expected-null read: the **(h, θ) degeneracy structure** — the correlation
`ρ(h, b)` and `ρ(h, ln s)` from the pooled grid. **Expected null:** `ρ(h, b) ≈ 0` is *not*
expected; a strong `h–b` ridge is the physically anticipated structure (a bias slope and a
distance-scale both move the inferred z–d_L mapping). The genuine expected null is
`ρ(h, ln s) ≈ 0` at truth-θ under S0-A. Both are REPORTED with their pooled SEM.

### 4.3 h-bias, width and the honest-trade band (registered reads i and iii)

```
t   = |<h> - 0.73| / σ_h            (the calibration statistic; a 68% interval covers truth iff t ≲ 1)
k   = σ_h(θ-marginalized) / σ_h(θ = truth-θ)      (the width ratio)
k*  = sqrt(1 + t₀²)                 (the RMSE-neutral trade point, t₀ frozen at the Stage-P landing)
ε_h = 0.005                         (the h-resolution floor)
```

**`ε_h` anchor, derived and fresh:** the registered h-grid's own step in the peak region —
`H_GRID_41` uses 0.005 across 0.655–0.785 and 0.010 in the wings
(`correspondence_1d.py:351-356`; `cluster/evaluate.sbatch:50-51`). A displacement smaller than
one peak-region grid step is not resolvable by the instrument that measures it. `ε_h` may only
tighten post-data.

**`k*` derivation (shown, not asserted):** if a bias `b₀` is fully absorbed and the width grows
from `σ` to `kσ`, RMSE goes from `sqrt(b₀² + σ²)` to `kσ`; these are equal at
`k = sqrt(1 + (b₀/σ)²) = sqrt(1 + t₀²)`. Below `k*` the trade improves RMSE; above it the
posterior has bought calibration at a net information loss. **`t₀` is frozen from the realized
Stage-P landing at truth-θ and recorded before any Stage-F comparison is made.**

| band | condition | reads as |
|---|---|---|
| **ABSORBED** | `t` falls to `≤ 1.0`, the fall is `≥ 3σ_t`, and `\|Δ⟨h⟩\| ≥ ε_h` | the tilt reabsorbs into θ |
| **HONEST-WIDTH-ONLY** | `t ≤ 1.0` reached with `\|Δ⟨h⟩\| < ε_h` and `k > 1` | calibration bought purely by widening |
| **FAVOURABLE TRADE** | additionally `k < k*` | RMSE improves |
| **UNFAVOURABLE TRADE** | `k ≥ k*` | calibration at net information loss — report the bound |
| **NO-ABSORPTION** | `t` unchanged within `3σ_t` and `\|Δ⟨h⟩\| < ε_h` | the lever does nothing at the h-marginal |

### 4.4 Coverage / P–P band (registered read ii — the stage-4 currency)

Per-seed: the rank of `h_true = 0.73` in that seed's h-marginal posterior CDF (mirror fleet,
`correspondence_1d.compute_seed_statistics` / `compute_full_log_posterior_vector`, `:3788-3789`).

| stage | statistic | band | anchor |
|---|---|---|---|
| **Stage P (4 seeds)** | 68.3% coverage count | exact central 95% acceptance region is `k ∈ [1, 4]` of 4 — **vacuous** | ⇒ **REPORTED-ONLY at Stage P, never band-bearing** (the PA-2D-1 F13 discipline) |
| **Stage F (12 seeds)** | 68.3% coverage count `k` | **CALIBRATED if `5 ≤ k ≤ 11`; MISCALIBRATED if `k ≤ 4` or `k = 12`** | exact central 95% binomial acceptance region for `n = 12, p = 0.6827` |
| **Stage F (12 seeds)** | one-sample KS of the 12 ranks vs. Uniform(0,1) | **PASS if `D ≤ 0.3754`** (α = 0.05); `D ≤ 0.4490` at α = 0.01 | exact `kstwo.ppf(0.95, 12)` / `(0.99, 12)` |

**Registered asymmetry, stated up front so it cannot be over-read post-data:** at 12 seeds a
coverage **FAIL is decisive**; a coverage **PASS is weak** (the acceptance region spans
0.417–0.917 in coverage fraction). A Stage-F coverage pass therefore supports CALIBRATED only
in conjunction with §4.3's bias/width bands, never alone.

### 4.5 Verdict map, mapped onto the research-cycle stage-5 decision table

| verdict | trigger | stage-5 action | disposition registered here |
|---|---|---|---|
| **LEVER-DEAD-AT-N** | B0-R′ (with B0-P satisfied) | *UNDETERMINED → name the one decisive measurement* | the one measurement is **S0-B** (the production-venue θ-score, §2.1) under an author costing grant. If the author declines it, the thread closes with a documented bound: "at mirror-venue N, a factor-1.5 error-model misstatement is undetectable by the (b,s) span." |
| **ERROR-MODEL-SHARE** | ABSORBED + CALIBRATED (§4.3, §4.4) + IDENTIFIABLE | *CALIBRATED + narrow → measure* | authorizes a `/physics-change` **proposal** for the production kernel (never an edit — §1.5), with its own 6-item gate package |
| **HONEST-WIDTH-ONLY** | HONEST-WIDTH-ONLY + CALIBRATED, any `k` | *CALIBRATED + wide → stop digging, report a bound* | banks as a calibration result: the thesis's posterior-honesty chapter + the `k` vs `k*` trade number |
| **UNIDENTIFIABLE** | §4.2 `Δ ln L ≤ 1.15` | *UNDETERMINED → name the one decisive measurement* | the one measurement is the **N-scaling read**: repeat Stage P at 2× the per-seed event count on 2 seeds and test whether `Δ ln L` scales `∝ N` (it must, for a correctly-normalized ensemble likelihood). Also banks the small-N validity statement the field lacks (row #193). |
| **MIXED / PARTIAL-ABSORPTION** | any of: B0-M; §4.2 MIXED; ABSORBED with MISCALIBRATED coverage; ABSORBED with `k ≥ k*` | *UNDETERMINED* | **first-class branch, with a disposition, not a fallback.** Bank the partial absorption fraction `1 − t/t₀` with its SEM as the thread's quantitative result; the one decisive next measurement is the **z-resolved θ-score** (score by z-bin at truth-θ), which discriminates "θ absorbs the low-z part only" from "θ absorbs a uniform fraction" — free, on the same banked cubes. |
| **INSTRUMENT-DEFECT** | B0-A′; GATE T-ID, PARITY, ENG, D3, TABLE-FRESH, STAMP or SCATTER-PAIR failure | *DEFECT → route to `/physics-change`* | if the defect is in a physics-trigger file (`bayesian_statistics.py` is one), the fix routes through `/physics-change` with its own gate package and ledger row; if it is harness-only (`correspondence_1d.py`), it is an `instrumentation` fix, disclosed, with GATE T-ID re-run |
| **CONTROL-FAIL** | GATE ENG passes but B0-R fails while B0-A passes | *DEFECT* | the venue can inject a misspecification the instrument cannot see: the comparison frame is defective; no [HIER] statement banks until diagnosed (the PA-2D-9 precedent) |

All verdicts are `[ORCH-banked, provisional]` pending the author's stage-5 ruling.

---

## 5. A10 — invariants and structural blindness

### 5.1 INVARIANTS — held fixed across every arm (one line each, with last-audited date)

| # | invariant | last audited |
|---|---|---|
| 1 | `h_true = 0.73` (`constants.H`, `correspondence_1d.py:359` `H_TRUE`) | 2026-08-25 (P3-2D fleet) |
| 2 | h grid = `H_GRID_41` verbatim, `h_bounds = (0.50, 0.86)` pin | 2026-08-25 (PA-CA-10 / P3-2D §5) |
| 3 | `mass_filter_sigma = "symmetric"` (production default, `[PHYSICS] cf4f8a2a`, row #202) | 2026-08-25 (PA-2D-4 item 1, PA-2D-6) |
| 4 | `catalogue_numerator_survival = "phi"` (production default since row #197) | 2026-08-26 (PA-2D-6, PA-2D-8 F9) |
| 5 | `selection_in_completion_numerator = "fused"` | 2026-08-25 (PA-2D-1 F7) |
| 6 | mirror-venue draw law + host/candidate linkage (the M2-LINK object) | 2026-08-26 (PA-2D-7 parts i/ii PASS on every seed/arm) |
| 7 | `SIGMA_V_PEC_KM_S = 0.0` (`constants.py:95`) — load-bearing for §1.2's `s` placement | **2026-08-26 (this cycle, recon §1.1)** |
| 8 | mirror↔production `host_z_error_eff` parity (`correspondence_1d.py:1167-1188`) | **NEVER** — audited this cycle by **GATE PARITY** (§3.3); until it passes, every mirror-venue conclusion is explicitly conditional on it, by name |
| 9 | `eddington_m` at its production default (mass-mean centering, PA-2D-1 F2) | 2026-08-25 (PA-2D-1 F2) |
| 10 | `host_mass_kernel` at its production default | **NEVER in this thread** — conclusions are conditional on it, by name; a `trunc_lognormal` resolution combined with a point-resolving `host_z_kernel` hits a pre-existing prior-consistency guard (`bayesian_statistics.py:6175-6178`) and must be smoke-checked before the pilot |
| 11 | `OMEGA_M = 0.2726` / the Barausse M1 mock cosmology (CLAUDE.md G11 design choice) | **NEVER in this thread** — conclusions are conditional on it; a θ–Ω_m degeneracy is invisible here (§5.2 viii) |
| 12 | the with-BH mass kernel's R&V15 intrinsic-scatter state | **OPEN** — `[P3-MKER]` (opened 2026-08-26) claims production omits it entirely; every with-BH read in this thread is conditional on that thread's outcome, by name |
| 13 | one `BayesianStatistics` construction per θ node (global tables keyed by h only) | **2026-08-26 (this cycle, GATE TABLE-FRESH §3.5)** |

### 5.2 STRUCTURAL BLINDNESS — defect classes this design cannot detect by construction

1. **(D2's kernel-variant restriction.)** θ is wired only to the configuration of record
   (CoR-M / CoR-P); a defect that lives *only* in `"global"`, `"local_ratio"`, bare
   `"volume_deconv"`, `"volume_global"`, `"volume_trunc"` or `"mass_trunc"`
   (`bayesian_statistics.py:6249-6277`) is invisible to every arm here.
2. **(The span of θ.)** θ = (b, s) spans exactly *linear-in-(1+z) bias* and *uniform
   multiplicative scale*. Any misspecification outside that 2-dimensional span — an outlier
   fraction, a heavy tail, a z-**dependent** scatter, a skew, a catastrophic-failure mode — is
   invisible **by construction**, and a null θ-score is a statement about the span, never about
   the error model. This is the sharpest blindness in the design, and it lands exactly where
   the Stage-L sweep found the literature lives (the minority-outlier regime, row #193).
3. **(Mirror-venue self-consistency — the D1-class defect.)** On the mirror the generator kernel
   *is* the estimator kernel at truth-θ, so any misspecification **shared** by generator and
   estimator cancels and is undetectable — precisely the class the research-cycle's stop/continue
   rule records that SBC/coverage cannot catch. S0-R mitigates this for the `s` axis only, and
   only for the one injected value.
4. **(No independent-reimplementation cross-check.)** `pp_coverage.py` (site 2.5) is deliberately
   NOT θ-parameterized (§2.4), so a defect **common** to `bayesian_statistics.py` and
   `correspondence_1d.py` — which already share a formula by intent, and whose parity claim is
   itself un-reverified (invariant 8) — cannot be caught by a third, independent implementation.
5. **(Common-mode instruments.)** All four registered reads ride the same dispatch paths, the
   same `S̄_φ`/survival tables, and the same `p_det` interpolant; a defect in any of those is
   common-mode across every arm and cancels out of every within-thread comparison.
6. **(Venue and N conditionality.)** Everything here is measured at the mirror venue's
   ~200 events/seed. Nothing transfers to production `N = 1588` or to the realistic scattered
   catalogue without S0-B, which is deferred and unfunded at registration.
7. **(Single-parameter cosmology.)** Only h is varied; a θ–Ω_m or θ–w degeneracy is invisible.
8. **(S0-R's joint z+mass injection.)** `sigma_scale` moves the mass column too
   (`observed_realization.py:178-183`), so the positive control cannot separate a z-kernel
   recovery from a mass-kernel one in the with-BH channel — hence the no-BH-only registration
   in §2.1.

---

## 6. Falsifiers (A19 / [A14] rule 15) — one per named verdict category

Registered **before** any attribution banks; an unrun falsifier leaves its attribution
explicitly provisional.

| verdict | its registered falsifier | band |
|---|---|---|
| **LEVER-DEAD-AT-N** | S0-B (production venue, CoR-P) returns `\|Z_b\| > 3.0` or `\|Z_s\| > 3.0`. A live production score with a dead mirror score refutes "dead" and re-attributes it to the mirror venue's own self-consistency (§5.2 item 3), not to the lever. | the §4.1 bands, unchanged |
| **ERROR-MODEL-SHARE** | (a) a **leave-one-z-bin-out** re-fit: if the absorbed tilt is carried by a single z-bin, the "ensemble self-calibration" attribution is refuted and the effect is one bin's misspecification (free, on the banked cubes); (b) B0-R fails on a **second** injected value `s_gen = 0.67` — recovery at 1.5 but not at 0.67 refutes a genuine scatter-scale lever. | (a) absorbed fraction changes by `> 3σ` on removal of any one bin; (b) the B0-R band applied at `ŝ` vs 0.67 |
| **HONEST-WIDTH-ONLY** | the width increase is reproduced by a **θ-blind prior-widening control**: inflate `σ_h`'s prior to match `k` with θ held at truth. If coverage improves identically, the widening is not θ-specific and the "honest trade" attribution collapses to "any widening would have done". | coverage counts agree within the §4.4 acceptance region |
| **UNIDENTIFIABLE** | the N-scaling read (§4.5): `Δ ln L` must grow `∝ N`. If it does **not** scale with N, the flatness is a normalization defect, not an information limit, and UNIDENTIFIABLE is refuted in favour of INSTRUMENT-DEFECT. | `\|Δ lnL(2N)/Δ lnL(N) − 2\| ≤ 3σ` |
| **MIXED / PARTIAL-ABSORPTION** | the z-resolved θ-score (§4.5): a partial absorption attributed to "θ captures part of the error model" is refuted if the residual score is z-**flat** (a flat residual is a normalization offset, not an unmodeled z-structure). | residual score slope in z consistent with 0 at `3σ` |
| **INSTRUMENT-DEFECT** | GATE T-ID and GATE PARITY both re-run after the fix, plus GATE ENG re-scored. If bit-identity is restored and the anomalous score persists, the "defect" attribution is refuted and the finding is physical. | `max \|Δ ln L\| = 0.0` exactly; `\|Z\|` unchanged within `1σ` |
| **CONTROL-FAIL** | a single global factor test (the PA-2D-9 pattern): if one multiplicative constant brings both the primary and the control readings into band simultaneously, the failure is a pairing/normalization defect and not an instrument blindness. | both `\|Δ\| < ε` under one common factor |

---

## 7. Costing (A6/A17; cluster-first per row #185) — `[ORCH-COST]`

**D6 disposition:** the proposal's *"25 θ-nodes × the 12-seed mirror fleet ≈ ~50–100 CPU-h"*
(`PROPOSAL_HIER_SELFCAL_20260825.md:51`) is **STALE and superseded; it may not be quoted.**
Root cause, registered: it priced one likelihood evaluation per (θ,seed) cell and never priced
the h-sweep that the same document's §2 requires. Full re-derivation:
`hier_costing_20260826.md` (2026-08-26).

### 7.1 Measured anchors ([A11]: {value, source, date})

| anchor | value | source | date |
|---|---|---|---|
| Mirror-venue `evaluate()`, single arm = **single h**, 200 events | 64.996 / 62.944 s @ `--cpus-per-task=16` | `cluster/p3_2d_rhs2.sbatch:15-16`, from `p3_2d_work/b{c,t}_900101_meta.json` | 2026-08-25/26 |
| corroborating | "~64 s/arm (single-h) uncontended; 2 arms/task ≈ 128 s" | `cluster/p3_2d_fleet.sbatch:31-33` | 2026-08-25 |
| per-task fixed overhead (host-pool/catalogue build) | ~10–30 s (the **upper** bound, 30 s, is used below) | `cluster/p3_2d_rhs2.sbatch:38-39` | 2026-08-26 |
| production `evaluate()`, per h-value, 3355 events | 56–76 min @ 16 cpus ⇒ 14.93–20.27 CPU-h/h-point | `cluster/LAUNCHING_JOBS.md:47` | 2026-07-03 |
| cross-task contention >2 tasks/node on `cpu_il` | ~1.7× slower/task | `.claude/skills/cluster/SKILL.md` gotcha 6 | — |
| largest arrays actually run in-repo (precedent, not a limit) | 49 tasks (`cluster/venue_transfer.sbatch:19`); 80 tasks (`cluster/LAUNCHING_JOBS.md:99-101`) | — | — |
| workspace `emri` expiry, zero extensions | **2026-09-23** (28 days from today) | `HANDOFF_20260730.md:179` | 2026-07-30 |
| queue wait, any numeric value | **NOT FOUND** | grep of all cluster docs + preregs | — |
| site `MaxArraySize` / `MaxSubmitJobs` for regular `cpu_il` | **NOT FOUND** (only `dev_cpu_il` QOS is documented) | grep of `preflight.sh`, `SKILL.md`, `LAUNCHING_JOBS.md`, `README.md`, `cluster.env` | — |

```
cost_per_h_point_per_cell = 63.97 s × 16 cpus / 3600 = 0.2843 CPU-h        [mirror venue]
CPU-h(stage, n_h) = cells × 0.2843·n_h + cells × 0.1333
```

### 7.2 Registered budgets and the authorization ceiling

| stage | cells | `n_h` | CPU-h | authorization |
|---|---|---|---|---|
| **S0-A** | 4 seeds × 5 θ-nodes = 20 | 1 | **8.35** | AUTHORIZED, ceiling 12 CPU-h |
| **S0-R** | 4 seeds × 5 θ-nodes = 20 | 1 | **8.35** | AUTHORIZED, ceiling 12 CPU-h |
| **S0-C** | 1 | 41 | **11.79** | AUTHORIZED, ceiling 15 CPU-h |
| **Stage 0 total** | — | — | **≈ 28.5** | **ceiling 35 CPU-h** |
| **Stage P** | 3×3 × 4 seeds = 36 | 41 (`H_GRID_41` verbatim) | **424.4** | AUTHORIZED **only on an author costing grant**, ceiling 450 CPU-h. Flagged: this is ~5.8× the largest fresh costing line yet granted in this campaign (PA-2D-8: 72.8 CPU-h). |
| **Stage F** | 5×5 × 12 seeds = 300 | set by GATE P→F | 807.6 (`n_h=9`) · 1319.4 (`n_h=15`) · 3537.0 (`n_h=41`) | **NOT AUTHORIZED at registration.** Requires a fresh costing line at GATE P→F, re-derived from S0-C's measured marginal (§3.6 leg 3). |
| **S0-B** (deferred) | 5 θ-nodes × 1 h, production venue | 1 | **74.7–101.4** | **NOT AUTHORIZED at registration.** |

Per-task walltime `= n_h × 63.97 s + 30 s` (94 s at `n_h=1`; 2653 s ≈ 44 min at `n_h=41`;
×1.7 contended ⇒ ≤ 75 min) — `--time=02:00:00` covers every `n_h ≤ 41` even fully contended.

**Workspace fit:** no single task risks the 2026-09-23 deadline. Total-CPU-h fit at `n_h ≤ 15`
reads comfortable by the PA-2D-8 turnaround precedent; **fit at `n_h = 41` for Stage F is NOT
certified** — queue-wait and achievable concurrency are NOT FOUND anywhere in this repo, so
GATE P→F's re-costing leg must include a live queue-depth read, not an a-priori claim.

### 7.3 Array design (D4)

```
theta_idx = SLURM_ARRAY_TASK_ID // N_SEED          # bijection on [0, N_THETA·N_SEED)
seed_idx  = SLURM_ARRAY_TASK_ID %  N_SEED
(theta_b, theta_s) = THETA_GRID_FLAT[theta_idx]    # row-major, §2.3
SEED = BASE_SEED_HIER + seed_idx                   # deliberately REPEATED across θ-nodes

Stage P: N_THETA=9,  N_SEED=4,  36 tasks   --array=0-35
Stage F: N_THETA=25, N_SEED=12, 300 tasks  -> 5 sub-arrays of 60 (--array=0-59 + THETA_ROW offset)
#SBATCH --partition=cpu_il --cpus-per-task=16 --time=02:00:00 --ntasks=1
```

**Why this is collision-free by construction, and is NOT a recurrence of PA-2D-8/F8.** The
(θ_idx, seed_idx) map is a bijection by integer div/mod, so no two tasks decode to one cell.
`SEED` repeating across θ-nodes is **required by the design** — every θ hypothesis must score
the *same* realized mirror-universe draw for the θ-comparison to mean anything. F8
(`PREREGISTRATION_P3_2D_20260825.md:284-286`, fix in `cluster/p3_2d_rhs2.sbatch:76-80`) was a
different failure mode: RHS₂ ran multiple stochastic draw-chunks *inside* one task and adjacent
tasks' chunk-seed ranges overlapped, double-counting draws into one accumulator. **[HIER] has no
within-task multi-chunk draw loop** — one task = one deterministic realization at its seed, then
a θ-fixed h-sweep on top — so there is no F8-shaped collision class. *If a future amendment adds
within-task chunking, the ×100 stride and the >100-chunk STOP must be re-applied.*

Stage F's 300-task single submission is **not** attempted: the site limit is NOT FOUND, and the
5×60 chunking keeps every submission inside the 49/80-task in-repo precedent band. The live
limit is queried during the mandatory `/cluster` preflight (GATE SEQ).

### 7.4 Quantities carried as BANDS, never as point numbers ([A11]; source: `hier_provenance_stamps_20260826.md` §3)

1. **F5's headline percentages** (`σ_M ≲ 1–2%`, `~50×`, `~1–3×`) — **STALE**: measured at
   `N_events = 400` (the document's own §4 caveat 4 requires re-quoting at the adopted N), under
   a configuration with **no candidate eligibility window at all**, ~2 months before the
   symmetric window landed (`cf4f8a2a`), and against an idealized σ_M handling that `[P3-MKER]`
   currently claims production does not implement. **Only the qualitative frontier relation
   (`σ_M·(1+z) ≲ σ_z`) and "GLADE rails regardless of σ_M at realistic scatter" may be cited**,
   as the best-case correctly-specified-kernel ceiling — misspecification only makes the
   achievable width worse, so the ceiling's *direction* is safe.
2. **"median σ_z/z ≈ 49%"** — **NOT FOUND as a median anywhere.** Carried as the range
   **0.25–0.6** spanning both sourced readings (`CLAIM_2D_BIAS_20260730.md:426` local-sample
   0.25–0.49; `:462-463` production-implied 0.35–0.6). 0.49 is a range upper bound.
3. **"N ≈ 40–200 events"** — **UNSOURCED**; appears only in the symptom card itself and
   conflicts with this campaign's recorded `n_events = 1588`
   (`CLAIM_PRODUCTION_CALIBRATION_HARNESS_20260817.md:101,113`). Not quoted anywhere in this
   prereg; the venue N used here is the **mirror fleet's ~200 events/seed**, stated as such.
4. **`p_det ≈ 1`** — sourced only to a June-era regime descriptor (ledger row #40), not to this
   campaign. Carried as a **working assumption**, to be spot-checked against this campaign's own
   `p_det` column; no band rides on it.
5. **the z-binned tilt (≈0 below z≈0.4, −1.08 by z≈0.9)** — usable **only with its true scope
   stated**: a dark/completion-class statistic (605/1588 iiib; 491/1588 joint_r1), not a
   whole-sample one, and not independently re-verified after `cf4f8a2a` (low risk by row #137
   item 1's own mass-channel-independence finding, but disclosed).
6. **The one FRESH number** quotable as-is, with its class scope stated: the aggregate dark-class
   score-at-truth **−0.635 ± 0.017** (iiib, 37σ) — explicitly re-verified post-sentinel-fix
   (ledger `:1647-1648`, `:1687-1690`).

---

*(Committed before the instruments exist. Pre-execution adversarial review — including D2's
configuration-of-record ruling, D7's venue split, GATE D3's three clauses, GATE PARITY, the θ
grid anchors, the ε_h/k\*/Z bands, and the Stage-P costing grant — precedes any instrument run;
A20 review before banking. Nothing above this line may be edited after commit.)*

---

## APPEND-ONLY AMENDMENT LOG

*(Amendments are numbered `PA-HIER-1`, `PA-HIER-2`, … Each records its date, whether any
instrument had run at the time of writing, what changed, and why. Nothing above the divider is
ever edited; a correction is an amendment.)*

<!-- PA-HIER-1: -->

## PRE-EXECUTION ADVERSARIAL REVIEW — PA-HIER-1 … PA-HIER-18
**(2026-08-26, pre-commit, NO instrument has run, zero compute spent. Independent re-derivation of
every registered object from source. Verdict: LAUNCH-BLOCKED. Blockers are PA-HIER-1, -2, -3, -4,
-6, -7. Nothing above the divider is edited; each block names the section it supersedes.)**

---

### PA-HIER-1 — BLOCKER. The venue's generator law (`host_mode`) is never registered, and on the mode the prereg's own prose describes, **truth-θ ≠ (0, 1)**.
**Date:** 2026-08-26 · **Instrument run:** none · **Severity:** BLOCKER · **Status:** AUTHOR-RULING-NEEDED
**Supersedes:** §1.1 (Refute-by), §1.2 (last line), §2.1 (S0-A row + the "Why S0-A cannot carry the
early-exit" paragraph), §4.1 (B0-A / B0-A′), §5.2 item 3.

**Found.** The word `host_mode` does not appear anywhere in this prereg (`grep -n host_mode` on the
file: no hits). `MirrorUniverseGenerator.draw_realization` offers five mutually exclusive generator
laws, and the truth value of θ differs between them:

- `host_mode="catalogue"` (the **default argument**, B-0, "D-B item d"):
  `correspondence_1d.py:1779-1783` — *"host photo-z: B-0 (`sigma_z_scale == 1.0`) uses the
  catalogue's OWN z_obs/z_error columns AS-IS (D-B item d: "z_true := the catalogue z_obs treated as
  exact ... for the mirror universe's truth" ... so B-0 needs no extra re-scattering pass)"*, and
  `:2141` `true_d_L = dist_vectorized(host_z, h=H_TRUE)` with `host_z = pool.z[host_idx]` (`:2008`).
  **The generating law for z_true given the drawn host is a delta at z_g.** The estimator
  nevertheless integrates `N(z; z_g, σ_z_eff)` (`bayesian_statistics.py:6247`, `:6878-6879`). The
  model is therefore misspecified in the `s` direction at s = 1 **by construction**, and truth-θ on
  the `s` axis is s → 0 — which is exactly what the harness's own G-1 "exact z" variant realizes by
  flooring the error column to `EXACT_Z_ERROR_FLOOR = 1e-6` (`correspondence_1d.py:362-371`). This
  is not a new reading: the repo's own A20 review already recorded it —
  `correspondence_1d.py:1130-1133`, *"Finding 2 refuted the stock "catalogue" mode as the
  b0-identity venue: it ... sets z_true := the listed z (no photo-z scatter), so E[p_gen/q_G] != 1
  even for a correct arrangement — the identity test's B-T PASS branch was structurally
  unreachable."*
- `host_mode="catalogue_selected"` / `"catalogue_selected_2d"` (the b0i / b0i2d arms, PA-2/PA-11):
  `:1141-1153` and `:2071-2088` — z_true **is** drawn per event from
  `k_g(z)·S̄_φ(z;H_TRUE)/S̃_φ,g` on the host's own ±4σ window, with `k_g` the **estimator's own
  numerator kernel** (`:1250-1264`). Here truth-θ = (0, 1) genuinely holds, up to the window
  truncation and the S̄_φ/f_k factors matching the estimator's resolved flags.

**Consequence.** §2.1's registered expectation for S0-A ("EXPECTED NULL, |Z| ≤ 3.0") and §4.1's
B0-A′ disposition ("a non-zero score on a self-consistent venue is a bug in the hook, the venue, or
GATE PARITY. STOP.") are correct **only** under `catalogue_selected*`. Under the default
`"catalogue"` law S0-A is expected to return a large **negative** `Z_s` (the estimator's z-kernel is
grossly over-dispersed relative to a delta truth; ∫p_GW·p_g dz is maximised as p_g narrows, so
∂lnL/∂s < 0 at s = 1), and the verdict map would route a correct measurement of the venue's own
declared-truth convention into an INSTRUMENT-DEFECT A21 STOP — burning the 28.5 CPU-h Stage-0 grant
to manufacture a false defect.

**Correction (required before commit).** Register `host_mode` explicitly as a §5.1 invariant, with
its file:line, for every arm (S0-A, S0-R, P-GRID, F-GRID, R-GRID). If the answer is
`catalogue_selected*`, §5.2 item 3's blindness statement must also be corrected: the generator law
is `k_g·S̄_φ/S̃`, **not** `k_g`, so the shared-misspecification cancellation is only exact if the
estimator's resolved `catalogue_numerator_survival="phi"` slot carries the same `S̄_φ(z)` inside the
same z-integral — that identity is an unaudited load-bearing invariant and must be added to §5.1
with `NEVER` and audited this cycle, or every verdict declared conditional on it by name.

---

### PA-HIER-2 — BLOCKER. §2.4 site 2.4 and the CoR-M pin identify the **generator**, not the estimator; hooking θ there would make θ move the data.
**Date:** 2026-08-26 · **Instrument run:** none · **Severity:** BLOCKER · **Status:** FIXED-BY-AMENDMENT (re-pin below), AUTHOR-RULING-NEEDED on D2
**Supersedes:** §2.4 row 2.4 (including the bolded *"YES — this is where Stages 0/P/F actually
execute"*), §2.4's D2-fourth-clause paragraph and the **CoR-M** definition, and D1/D2's row in §1.7.

**Found (three independent re-derivations, all from source).**

1. `correspondence_1d.host_z_error_eff` has **exactly two callers in the entire repo**:
   `grep -n "host_z_error_eff(" darksiren_emri/validation/correspondence_1d.py` →
   `1323:` (inside `kernel_smeared_survival`, which computes `S̃_φ,g` — the **host-draw weight**) and
   `1485:` (inside `_draw_kernel_survival_redshifts` — the **z_true draw**). **Both are
   generator-side.** It is not "the mirror-venue harness's own copy" of the estimator kernel; it is
   the venue's own copy, used to *make the universe*.
2. The estimator at Stages 0/P/F is production's own code: `run_mirror_seed_inprocess` calls
   `bs.evaluate(...)` at `correspondence_1d.py:2844`. Dispatch therefore runs through §2.4 sites
   2.1/2.2/2.3, not 2.4.
3. That `evaluate()` call passes `normalization_mode=PRODUCTION_FLAGS["--normalization_mode"]` and
   `host_z_kernel=PRODUCTION_FLAGS["--host_z_kernel"]` (`:2851-2852`), and
   `PRODUCTION_FLAGS` (`:328-337`) is
   `{"--normalization_mode": "absolute_marginal", "--host_z_kernel": "volume_deconv", ...}`.

**Consequence A — the D2 fourth clause rests on a false premise.** §2.4 argues that "the default
production host-z kernel variant", read literally, resolves to `"point"` under
`normalization_mode="generator_marginal"` and therefore makes `s` structurally inert, and it
invents the CoR-M/CoR-P split to escape that. **The mirror venue already runs
`absolute_marginal` + `volume_deconv`** — the very pair the prereg pins as CoR-P. `s` is already
live natively in the production **batch** kernel (site 2.2) at the mirror venue. The CoR split is
unnecessary, and CoR-M as written pins the wrong object.

**Consequence B — the registered hook would destroy the experiment.** If θ is threaded into
`host_z_error_eff` as §2.4 requires ("`s` + `b`, independently maintained"), then every θ node draws
a **different mirror universe**: the host-draw weights (`S̃_φ,g`, `:1323`) and each event's `z_true`
(`:1485`) both move with θ. `L(h, θ)` is then not a likelihood surface over one fixed dataset, the
score-at-truth identity `E[∂_θ ln L] = 0` does not apply at all, and §4.2's Wilks anchor is void.
GATE T-ID would still pass (the literal early-return at (0,1)) and GATE ENG would still pass (events
move) — the exact silently-mixed-arm failure mode [A13] exists to prevent, arriving through the
registered design rather than through an oversight.

**Correction, registered.** (i) **CoR-M is re-pinned to production's own batch kernel** — site 2.2,
`bayesian_statistics.py:6878-6879` and `:6899-6901`, at `absolute_marginal`/`volume_deconv`
(`correspondence_1d.py:328-337`), i.e. **CoR-M ≡ CoR-P in the estimator flags**; the venue differs,
the configuration of record does not. (ii) **No θ hook may be added to
`correspondence_1d.host_z_error_eff`, `kernel_smeared_survival`, or
`_draw_kernel_survival_redshifts`.** A registered GATE is added: **GATE GEN-FROZEN** — the venue's
realized per-seed event table (`write_mirror_crb_csv` output, including `z_true`) must be
**byte-identical across every θ node at fixed seed**; a single differing byte is an A21 STOP.
(iii) §1.7's D2 row and §2.4's D2 paragraph are superseded; D2's "OUT of scope" list of variants
(`"global"`/`"local_ratio"`/…) survives unchanged as §5.2 item 1.

---

### PA-HIER-3 — BLOCKER. `realize_observed_catalogue(sigma_scale=1.5)` injects **no** z-kernel misspecification. S0-R is a null instrument, so D7's registered early exit fires for a reason unrelated to the lever.
**Date:** 2026-08-26 · **Instrument run:** none · **Severity:** BLOCKER · **Status:** AUTHOR-RULING-NEEDED
**Supersedes:** §2.1 S0-R row and its "Disclosed defect" paragraph, §2.2 R-GRID, §3.6 leg 2,
§4.1 B0-R and B0-R′, §4.5 LEVER-DEAD-AT-N, §6 row 2(b), §7.2 S0-R line.

**Found.** `realize_observed_catalogue` rewrites the redshift **column**:
`z_obs = z_g + sigma_scale · z_error_g · N(0,1)`, clipped at a 1e-5 floor, and — per its own
docstring (`observed_realization.py:176-186`) — *"`z_error`, flags, sky positions and B magnitudes
are copied as their original strings"*. The realized catalogue is then loaded as an ordinary
catalogue (`host_pool_for_sigma_scale`, `:1889-1891`), the host pool is extracted **from it**, and
the mirror places each event at that same row's z (`:2141`, or `:2088`/`:2122` in the
`catalogue_selected*` modes, whose `z_true` is drawn from the **realized** row's own
`k_g(z; z_obs, z_error)`).

**Therefore the relation "the estimator's kernel is centred on the same catalogue value the
generator used, with the same quoted width" is preserved exactly at every `sigma_scale`.**
`sigma_scale` is a *catalogue-perturbation* knob (it changes which galaxies sit at which z), **not**
an estimator/generator width-mismatch knob. There is no `s_gen = 1.5`: truth-θ after the call is
still (0, 1) in `catalogue_selected*`, and still (0, s→0) in `"catalogue"`. §2.1's characterisation
— *"the catalogue's quoted `z_error` column is copied unchanged while the realized scatter is 1.5×,
i.e. a KNOWN misspecification with truth-θ = (0, 1.5)"* — is refuted at source: the quoted column is
copied unchanged **and it is still the width of the law that generated the pool's own z values**,
because the realized z *is* the catalogue z.

**Two further, independent problems with S0-R at dose 1.5.**
(a) `correspondence_1d.py:376-401` records the z-floor clip artefact: *"how many depends on
sigma_z_scale (354 rows at 0.05, 4188 at 0.25, per the job-6383719 sidecar logs — monotone
increasing but not linearly)"*, together with a whole harness bug-fix (`HOST_DRAW_WEIGHT_Z_FLOOR`)
that exists because at **dose 0.25** the clipped rows *"swamped the entire 200-event
weighted-without-replacement draw"*. Dose **1.5 is six times the largest dose the harness has ever
been exercised at**, and the artefact class is monotone increasing.
(b) The realized catalogue carries `sidecar sigma_scale > 0` ⇒ `GalaxyCatalogueHandler.scattered =
True` ⇒ `validate_scatter_guards` / `resolve_host_z_kernel` (`bayesian_statistics.py:3864-3878`,
`:166-213`) apply the one-directional scatter guard to S0-R and **not** to S0-A/Stage P/F (pinned,
unscattered baseline). GATE SCATTER-PAIR (§3.8) demands the resolved pair be *"identical across
every stage"*; the prereg never pins that pair for CoR-M, so SCATTER-PAIR is presently unverifiable
and possibly violated by S0-R's own construction.

**Consequence.** GATE P→F leg 2 and B0-R can never be satisfied by S0-R as constructed, and B0-R′
("|Z_s| ≤ 3.0 with ENG passing ⇒ **LEVER-DEAD-AT-N**", D7's registered early exit) would fire on an
arm in which nothing was injected. That is the strongest possible form of the PA-2D-9 CONTROL-FAIL
class: the control does not exercise the axis it certifies.

**Correction.** A real `s`-axis positive control requires a **z-only, estimator-side** knob:
realize the catalogue at the baseline, then multiply the **quoted** `z_error` column by `1/1.5`
(or scatter `z_obs` by `1.5·z_error` while writing the *unscaled* column into a second file used
only by the estimator). Either is **new code before launch** and needs its own registration. Until
such a control exists, **no LEVER-DEAD-AT-N verdict may bank**, and D7's early exit is unarmed.

---

### PA-HIER-4 — BLOCKER. `score_s` is a mis-formed statistic: a secant that is not centred on truth-θ.
**Date:** 2026-08-26 · **Instrument run:** none · **Severity:** BLOCKER · **Status:** FIXED (registered form replaced below)
**Supersedes:** the `score_s` line of §4.1's code block.

**Found.** Registered: `score_s = [lnL(s=√2) − lnL(s=1/√2)] / (√2 − 1/√2)`, denominator 0.70711.
The two nodes are symmetric in **ln s**, not in **s**: `1 − 1/√2 = 0.292893` below, `√2 − 1 =
0.414214` above. A difference quotient estimates the derivative at the interval's **arithmetic**
midpoint, `(√2 + 1/√2)/2 = 1.0606602`, not at s = 1. Expanding f(s) = lnL(s) about s = 1:

```
[f(√2) − f(1/√2)] / (√2 − 1/√2) = f'(1) + 0.060660·f''(1) + 0.022674·f'''(1) + …
```
(coefficient re-derived and checked numerically: `((√2−1)² − (1−1/√2)²) / 2 / (√2 − 1/√2) =
0.0606601717798213`).

At truth-θ, `E[f'(1)] = 0` but `E[f''(1)] = −I_ss < 0`, so **the registered statistic has a
non-zero expectation at truth**: `E[score_s] ≈ −0.06066·I_ss` per event. Since
`Z_s = mean/SEM` with `SEM = sd/√n`, the spurious `|Z_s|` grows as `√n` — at Stage 0's
n ≈ 4 × 200 = 800 event-instances, `|Z_s| ≈ 0.06066·√800·√I_ss ≈ 1.7·√I_ss`, which exceeds the
B0-A band (3.0) for any per-event Fisher information `I_ss ≳ 3` in s-units. **The S0-A control is
therefore liable to fail from the statistic's own form, independently of PA-HIER-1.** This is
precisely the PA-2D-8 F3 `κ̂₂` class of defect.

**Correction, registered.** The whole document already treats `ln s` as the natural parameter
(§2.3 "log-uniform in ×√2 steps"; §4.1 B0-M `|ln ŝ| < 0.5·ln√2`; §4.1 B0-P `σ_ln s < ln 2`).
Reparameterize the score to match:

```
score_lns = [ lnL(b=0, ln s=+ln√2) − lnL(b=0, ln s=−ln√2) ] / (2·ln√2)      (denominator ln 2 = 0.6931472)
Z_lns     = mean(score_lns) / SEM(score_lns)
```
Nodes are unchanged (s = √2 and 1/√2 are the same grid nodes), so **no re-costing is required**;
only the denominator and the centring change. This form is exactly centred on truth (ln s = 0) and
its leading error is the odd `(ln√2)²/6 · g'''` term, with **no `f''` contribution**. Every §4.1
band that names `Z_s` (B0-A, B0-A′, B0-R, B0-R′) is hereby restated in terms of `Z_lns`.
`score_b` is **not** affected — its nodes ±0.02 are symmetric about b = 0 and its denominator 0.04
is correct.

---

### PA-HIER-5 — HIGH. `Z`'s variance is estimated from the wrong sampling unit, and the statistic has no tail guard.
**Date:** 2026-08-26 · **Instrument run:** none · **Severity:** HIGH · **Status:** AUTHOR-RULING-NEEDED
**Supersedes:** §4.1's `Z_x = mean(score_x)/SEM(score_x)` line and the "pooled over the arm's events
and seeds, with its SEM from the per-event scatter" clause.

1. **Wrong sampling unit.** θ is a *shared* hyperparameter; the object the hypothesis is about is the
   ensemble score `Σ_i ∂_θ ln L_i`, whose sampling distribution is over **mirror universes**, not
   over events. A per-event SEM with n = N_events × N_seeds treats events as i.i.d., but within one
   seed they share the realized catalogue, the same global selection denominator (site 2.3), the
   same `p_det` interpolant and the same completeness tables (§5.2 item 5 names this common mode and
   then the statistic ignores it). Any positive within-seed correlation ρ inflates |Z| by
   `√(1 + (N_ev−1)ρ)` — with N_ev ≈ 200, ρ = 0.01 already inflates |Z| by 1.7×. **The direction of
   the error is toward manufacturing B0-A′ (INSTRUMENT-DEFECT) and toward passing B0-R.**
2. **No tail guard.** `mean/SEM` on a heavy-tailed summand is the exact vacuity class the b0-identity
   thread hit and PA-2D-1 F17 answered with a Hill-estimator check. §4.1 registers no such check.
   A concrete generator of such tails exists here — see PA-HIER-17.

**Correction.** Register both legs: (a) a **seed-clustered** SEM (cluster-robust, clusters = seeds)
reported alongside the per-event SEM, with the *larger* of the two used for every band decision, and
the design-effect ratio banked; and (b) a tail diagnostic per component — Hill α on |score|, plus a
5%-trimmed-mean re-score — with the registered rule that if the untrimmed and trimmed `Z` disagree
in band, no `Z`-based verdict banks. Note that at 4 seeds a clustered SEM has 3 d.o.f., so the
B0-R `|Z| ≥ 5` band is not reliably estimable at Stage 0; this is a second reason to defer any
B0-R/B0-R′ decision (see PA-HIER-8).

---

### PA-HIER-6 — BLOCKER. No θ prior / marginalization measure is registered anywhere, yet three verdict families depend on it.
**Date:** 2026-08-26 · **Instrument run:** none · **Severity:** BLOCKER · **Status:** AUTHOR-RULING-NEEDED
**Supersedes:** §4.3's definition of `k`, §4.4's per-seed rank statistic, §4.5 rows
ERROR-MODEL-SHARE / HONEST-WIDTH-ONLY / MIXED.

**Found.** `grep -n "prior"` on this file returns six hits, none of which defines a prior on θ.
`σ_h(θ-marginalized)` (§4.3), the per-seed h-marginal posterior whose CDF supplies the §4.4 rank,
and the §4.5 honest-trade verdicts are all functions of a θ prior that does not exist in the
registered text. Marginalizing a 3- or 5-node grid implicitly asserts a discrete uniform prior —
uniform in `b`, uniform in `ln s` — **whose support is the grid**. The width inflation `k`, and
hence FAVOURABLE vs UNFAVOURABLE TRADE and the entire coverage result, are therefore set by the
prereg's own choice of grid extent (§2.3), not by the data. Widening the b half-width from 0.04 to
the 0.163–0.392 that the prereg's own §7.4 item 2 band implies (see PA-HIER-9) would change `k`, and
so the verdict, without a single new likelihood evaluation.

**Correction.** Register, before launch: (i) the θ prior explicitly — measure, support and whether
it is the discrete grid measure or a quadrature weight over the continuum; (ii) the h prior/support
used for the marginal (see PA-HIER-14); (iii) a **prior-sensitivity leg** — recompute `k`, `t` and
the §4.4 rank on the Stage-P sub-grid (b ∈ {−0.02, 0, +0.02}, s ∈ {1/√2, 1, √2}) and require the
verdict to be invariant; a verdict that flips under the sub-grid is REPORTED-ONLY. This is free on
the banked cubes. Note that §6's HONEST-WIDTH-ONLY falsifier ("a θ-blind prior-widening control")
already presupposes a prior comparison and is not executable until (i) exists.

---

### PA-HIER-7 — BLOCKER. The identifiability statistic does not say what happens to h; GATE P→F leg 1 and 424.4 CPU-h ride on the answer.
**Date:** 2026-08-26 · **Instrument run:** none · **Severity:** BLOCKER · **Status:** AUTHOR-RULING-NEEDED
**Supersedes:** §4.2's statistic line and its three band rows; §3.6 leg 1.

**Found.** §4.2 registers *"pooled ensemble `Δ ln L` between truth-θ and the registered grid corner,
per seed, median over seeds"* — with no statement of whether h is held at 0.73, **profiled**
(`max_h`), or **marginalized** (`∫dh`). The three give materially different numbers, and the
difference is exactly the quantity the thread exists to measure: §4.2's own secondary read
anticipates *"a strong `h–b` ridge"*, and along a ridge the profiled ΔlnL is much smaller than the
fixed-h ΔlnL. A fixed-h ΔlnL can pass the 3.00-nat IDENTIFIABLE band while the profiled ΔlnL sits
below 1.15 (UNIDENTIFIABLE) on the same cubes.

Second: the anchor. `χ²₂(0.95)/2 = 2.9957` and `χ²₂(0.6827)/2 = 1.1479` (both re-derived exactly:
`−2 ln(1−p)/2` for 2 d.o.f.) are Wilks anchors for the **profile** likelihood ratio between the MLE
and a point, with **2** free parameters. If h is held fixed, the correct reference has 2 d.o.f. only
if h is genuinely known; if h is profiled, the correct reference is still χ²₂ **but the statistic
must be `max_h lnL(h,θ̂) − max_h lnL(h, corner)`**, and `lnL(truth-θ) ≈ lnL(θ̂)` must itself be
checked (on a 3×3 grid it need not hold). Registering the anchor without registering the statistic's
h-treatment makes the band unfalsifiable as posed.

**Correction.** Pin the statistic as `Δ ln L = max_h Σ_i ln L_i(h, truth-θ) − max_h Σ_i ln L_i(h,
corner)` (h profiled over `H_GRID_41`), report the fixed-h and h-marginalized variants alongside as
REPORTED-ONLY, and add a registered pre-condition to the IDENTIFIABLE branch: `lnL(truth-θ) ≥
lnL(θ)` for every other Stage-P node, else the "truth ≈ MLE" premise of the Wilks anchor fails and
the read is REPORTED-ONLY.

---

### PA-HIER-8 — HIGH. B0-R is circular (its recovery clause needs the stage it gates), and B0-P references an object Stage 0 does not compute.
**Date:** 2026-08-26 · **Instrument run:** none · **Severity:** HIGH · **Status:** FIXED (split registered below)
**Supersedes:** §4.1 rows B0-R and B0-P; §3.6 leg 2.

1. **Circularity.** B0-R's disposition is *"LEVER-LIVE. Proceed to Stage P"*, but its second clause
   is *"the grid's `ŝ` satisfies |ŝ − 1.5| ≤ 0.35"*, and §2.1 itself scopes that clause to *"the
   **Stage-P/F** grid"*. A conjunction that gates entry to Stage P cannot contain a Stage-P
   measurement. **Registered split:** **B0-R(i)** = the Stage-0 score clause alone (the gate on
   proceeding to Stage P); **B0-R(ii)** = the recovery clause, evaluated at Stage P, feeding GATE
   P→F leg 2 only. §3.6 leg 2 is restated to cite B0-R(ii).
2. **`ŝ` is never defined.** "the grid's recovered `ŝ`" — argmax over grid nodes, parabolic vertex in
   `ln s`, or posterior mean under the (unregistered, PA-HIER-6) θ prior? These differ. Under the
   plainest reading (grid argmax) the clause is **unsatisfiable at Stage P by construction**: the
   Stage-P s-grid is {0.50, 1.00, 2.00} and `min_node |node − 1.5| = 0.50 > 0.35`. **Registered:**
   `ŝ ≡ exp(vertex of the parabola through the three `ln s` nodes' pooled `lnL`)`, with the clause
   restated in the log parameter as `|ln ŝ − ln 1.5| ≤ 0.2352` (= ln(1.85/1.5), the log-image of the
   asserted 0.35 upper leg; the linear ±0.35 was asymmetric on a multiplicative parameter — 1.5's
   grid steps are −0.4393 / +0.6213, both re-derived).
3. **B0-P names a "corner curvature" Stage 0 does not have.** The Stage-0 θ-cross is
   `{(0,1), (±0.02,1), (0,1/√2), (0,√2)}` (§2.3) — a cross, with no corner. **Registered:** B0-P's
   implied 1σ_θ is computed from the **cross's own** three-point curvature per axis
   (`σ_b² = −1/∂²_b lnL`, `σ_ln s² = −1/∂²_ln s lnL`), and if either second difference is ≥ 0
   (non-concave) the arm is **UNPOWERED** by definition.
4. **B0-P is weaker than it reads.** `σ_b < 0.04` = "the half-width is at least 1σ" ⇒ corner
   `Δ lnL > 0.5` nats — far below §4.2's own 3.00-nat identifiability bar. As registered, B0-P
   cannot make a null "interpretable" in the sense §4.1 claims. **Registered tightening** (bands may
   only tighten): B0-P passes only at `σ_b ≤ 0.04/2.449` and `σ_ln s ≤ ln2/2.449` — i.e. the grid
   half-width is ≥ the 95% joint contour, matching §4.2's anchor rather than contradicting it.

---

### PA-HIER-9 — HIGH. The b-grid anchor misattributes a synthetic-harness config default as "the catalogue's own quoted photo-z scatter", and contradicts this prereg's own §7.4 band by 3.5–8×.
**Date:** 2026-08-26 · **Instrument run:** none · **Severity:** HIGH · **Status:** AUTHOR-RULING-NEEDED
**Supersedes:** §2.3's "Anchor for b's half-width 0.04" paragraph.

**Found.** §2.3 anchors `b_max = 2·0.035/(1+0.485) = 0.0471` on *"The catalogue's own quoted photo-z
scatter, `σ_z = 0.035` (`pp_coverage.py:630`, `PPCoverageConfig.sigma_z`)"*. Quote-verified at
source: `pp_coverage.py:630` is `sigma_z: float = 0.035` inside `PPCoverageConfig`, documented at
`:510` as *"Host photo-z scatter (**commission value** 0.035)"* — a **synthetic-universe config
default** in the very module §2.4 declares *"a **fully independent reimplementation** … **NO — OUT
OF SCOPE**"*. It is not a GLADE statistic and not this campaign's. (`z ≈ 0.485` **is** verified:
ledger `:1355`, *"mean z 0.485"*. The arithmetic `2·0.035/1.485 = 0.04714` is also correct.)

**Internal contradiction.** §7.4 item 2 registers this campaign's σ_z/z as the band **0.25–0.6** —
and §2.3's *own* s-anchor paragraph, four lines below, uses that band. At z = 0.485 that band means
σ_z ≈ 0.121–0.291, i.e. **3.5×–8.3× the 0.035 used one paragraph earlier**. Applying §2.3's own
formula to §7.4's own band gives `b_max = 0.163 … 0.392`. The registered ±0.04 grid is therefore
**4–10× narrower** than the prereg's own registered scatter band, and the claim *"the grid spans ±2
catalogue-σ_z of bias at the median z"* is false under this document's own §7.4. This is a STALE/
out-of-configuration quantity leaking into a band anchor — the exact [A11] failure §7.4 exists to
prevent, committed inside the same file.

**Correction.** Either (a) re-anchor `b` on a **measured** quantity from this campaign's own
catalogue (the median `host_z_error/(1+host_z)` over the mirror host pool — free, zero-compute, one
pandas read of `REDUCED_CATALOGUE_PATH`), and re-derive the grid; or (b) keep ±0.04 and restate its
anchor honestly as `[ORCH-ANCHOR, convention]` — a *local* score/curvature probe, not a
"±2 catalogue-σ_z span" — in which case §4.2's IDENTIFIABLE reasoning ("the grid half-width exceeds
the 95% contour, i.e. θ is constrained inside the registered range") loses its physical
interpretation and must be restated as a statement about the registered range only. **Option (a) is
free and is recommended before launch.** Note the coupling to PA-HIER-6: the b half-width also sets
the width inflation `k`, so this is not a cosmetic choice.

---

### PA-HIER-10 — HIGH. GATE D3(a)'s *conditional* forcing puts a branch discontinuity exactly at truth-θ, and it is live at the mirror venue (not only at CoR-P).
**Date:** 2026-08-26 · **Instrument run:** none · **Severity:** HIGH · **Status:** FIXED (unconditional pin registered below)
**Supersedes:** §3.2 clause (a); §2.4's site-2.3 scope entry *"YES (under CoR-P)"*; §2.4's CoR-P
bullet.

**Found.** `--smear_global_selection` is `action="store_true"`, default **False**
(`arguments.py:773-786`, quote-verified). D3(a) registers *"`s ≠ 1` **forces** the
`smear_sigma_z=True` branch"*. Because the mirror venue runs `absolute_marginal` (PA-HIER-2), it
**does** consume `Σ_glob`, and `run_mirror_seed_inprocess` does **not** pass
`smear_global_selection` — so it resolves False at the truth node. Under D3(a) as written, the truth
node (0,1) is evaluated with a **point-evaluated** global selection denominator while every s ≠ 1
node is evaluated with a **kernel-smeared** one. That is a step discontinuity in `lnL` located
precisely at truth-θ:

- it enters `score_lns` directly (both nodes s = √2, 1/√2 are smeared; truth is not) — the secant
  then measures the branch switch, not the derivative;
- it enters §4.2's `Δ ln L` (truth-θ vs a corner with s ≠ 1) — the identifiability number becomes
  "θ effect ⊕ smear-branch offset", an un-decomposable mixed arm;
- and §2.4 simultaneously scopes site 2.3 *out* of the mirror ("YES (under CoR-P)"), so the
  discontinuity would enter unhooked and unaudited.

Note also that §2.4's CoR-P bullet pins `smear_global_selection` **forced True** unconditionally,
which directly contradicts §3.2(a)'s conditional forcing. The prereg carries both readings.

**Correction, registered.** `smear_sigma_z` is **pinned True for the entire arm, at every θ node
including truth-θ**, at both CoR-M and CoR-P — so the s-axis is homogeneous and the branch is an
invariant, not a function of θ. Consequences that follow and are registered here: (i) site 2.3 is
**IN SCOPE at the mirror venue**, not only under CoR-P, and needs its `s` and `b` hooks there;
(ii) GATE T-ID's comparand must be re-banked at `smear_sigma_z=True` (the pre-θ banked CSVs were
produced at False — bit-identity against them would otherwise fail for a reason unrelated to θ);
(iii) `smear_global_selection: True` joins the A22 stamp set, making it **twelve** stamps, not
eleven (§2.5 superseded); (iv) D3(b) (`generator_marginal` refusal) is retained but is moot at every
registered venue, since neither CoR-M nor CoR-P uses that mode.

---

### PA-HIER-11 — MEDIUM. The dispatch-path enumeration is incomplete: there are **six** σ_z sites, not five.
**Date:** 2026-08-26 · **Instrument run:** none · **Severity:** MEDIUM · **Status:** FIXED (site added)
**Supersedes:** §2.4's header sentence *"enumerated **five** independent sites"* and its table.

**Found.** `grep -rn "SIGMA_V_PEC_KM_S" darksiren_emri/` returns a host-z width computation the
table does not carry: **`bayesian_statistics.py:7518-7520`**, inside
`single_host_likelihood_integration_testing` —
`_sigma_z_pv = (1.0 + possible_host.z) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S`;
`_z_error_eff = sqrt(possible_host.z_error**2 + _sigma_z_pv**2)`;
`galaxy_redshift_normal_distribution = norm(loc=possible_host.z, scale=_z_error_eff)`, whose own
comment states it exists *"so the integration-testing twin stays a faithful cross-check of the
production path"*. It is exercised by `/integration-test-eval` and referenced from
`darksiren_emri_test/bayesian_inference/test_simulation_detection_probability.py:1098`.

**Consequence.** It is not a production dispatch path (production `evaluate()` never calls it), so
it does not create a mixed arm — but it is the repo's designated *fidelity twin* of sites 2.1/2.2,
and a θ hook that skips it silently breaks that cross-check. It also partially contradicts §5.2 item
4's claim that no independent cross-check exists.

**Registered as site 2.7** — **NOT** θ-hooked, but carrying a **registered obligation**: the θ hook's
regression test asserts that the integration-testing twin still reproduces sites 2.1/2.2 at
θ = (0,1); if the hook's refactor changes the shared expression's shape, the twin is updated in the
same commit. Add to §5.2 item 4 that the twin exists and is deliberately not θ-parameterized.

---

### PA-HIER-12 — MEDIUM. `k*` is derived only for the fully-absorbed limit, but the ABSORBED band admits `t` up to 1.0 — the registered `k*` overstates the favourable region by up to √2.
**Date:** 2026-08-26 · **Instrument run:** none · **Severity:** MEDIUM · **Status:** FIXED (corrected form registered)
**Supersedes:** §4.3's `k*` definition and its derivation paragraph; the FAVOURABLE / UNFAVOURABLE
TRADE rows.

**Found.** The registered derivation is correct **as far as it goes**: with the bias *fully*
absorbed, `sqrt(b₀² + σ²) = kσ ⇒ k* = sqrt(1 + t₀²)`, re-derived and confirmed. But ABSORBED is
declared at `t ≤ 1.0`, i.e. the residual bias `b₁ = t·k·σ` need not be zero. Redoing it with a
residual:

```
sqrt(b₁² + k²σ²) = sqrt(b₀² + σ²) ,   b₁ = t·k·σ ,  t₀ = b₀/σ
⇒ k²(1 + t²) = 1 + t₀²   ⇒   k* = sqrt( (1 + t₀²) / (1 + t²) )
```
The registered `k* = sqrt(1 + t₀²)` is the `t → 0` special case. At the band's own boundary `t = 1`
the true neutral point is `sqrt((1+t₀²)/2)` — **a factor √2 = 1.414 smaller**. A design landing at
`sqrt((1+t₀²)/2) ≤ k < sqrt(1+t₀²)` would be banked as **FAVOURABLE TRADE** while its RMSE actually
got worse.

**Correction.** `k* ≡ sqrt((1 + t₀²)/(1 + t²))` with `t` the realized post-marginalization
calibration statistic and `t₀` frozen at the Stage-P landing (freeze rule unchanged). Both `t₀` and
`t` are recorded before the comparison.

---

### PA-HIER-13 — MEDIUM. The MIXED branch's headline quantity `1 − t/t₀` is not an absorption fraction.
**Date:** 2026-08-26 · **Instrument run:** none · **Severity:** MEDIUM · **Status:** FIXED (corrected form registered)
**Supersedes:** §4.5's MIXED / PARTIAL-ABSORPTION disposition cell.

**Found.** `t = |⟨h⟩ − 0.73| / σ_h`. Marginalizing over θ moves **both** the numerator and the
denominator: `k > 1` alone makes `t` fall with `⟨h⟩` completely unmoved. So `1 − t/t₀` measures
*calibration improvement*, of which pure widening is a sufficient cause — it is not "the partial
absorption fraction", and §4.3 already has a separate band (HONEST-WIDTH-ONLY) for exactly the case
`1 − t/t₀ > 0` with zero absorption. Banking `1 − t/t₀` as "the thread's quantitative result" would
report widening as absorption.

**Correction.** The MIXED branch banks **two** numbers, both with SEM: the **absorption fraction**
`A ≡ 1 − |⟨h⟩_marg − 0.73| / |⟨h⟩_truth-θ − 0.73|` (zero if the mean does not move — the honest
statistic), **and** the width ratio `k` from §4.3. `1 − t/t₀` may be reported as the derived
calibration-improvement statistic, explicitly labelled as such, and is never the attribution
quantity. The registered next measurement (the z-resolved θ-score) is unchanged. Note this branch
otherwise passes the first-class-Mixed test: it has a real trigger set, a disposition, a banked
quantity and a named next measurement — §4.2's MIXED row likewise. **§4.5's MIXED branch is not
decorative**; only its statistic is wrong.

---

### PA-HIER-14 — MEDIUM. The coverage/P–P read's h support is unpinned, and the two cited functions disagree on it at their only call site.
**Date:** 2026-08-26 · **Instrument run:** none · **Severity:** MEDIUM · **Status:** AUTHOR-RULING-NEEDED
**Supersedes:** §4.4's statistic line and §2.4's "Registered departure from the proposal" paragraph.

**Found.** §4.4 cites `correspondence_1d.compute_seed_statistics` / `compute_full_log_posterior_vector`
at `:3788-3789`. Those two lines are a **call site**, not the definitions (`def` at `:3043` and
`:3611`), and at that call site they are invoked on **different h grids**:
`compute_seed_statistics(diag_csv, seed, h_grid=H_GRID_41)` (`:3788`) versus
`compute_full_log_posterior_vector(diag_csv, h_grid=H_GRID_FULL)` (`:3789`). `H_GRID_FULL` adds the
low wing `{0.50…0.58}` which `correspondence_1d.py:348-350` marks *"REPORTED-ONLY … never
band-bearing"* and which this prereg's own §5.1 invariant 2 excludes (*"h grid = `H_GRID_41`
verbatim"*). The rank of `h_true` in a posterior CDF is a function of the support; two supports give
two ranks and two KS statistics.

Three further unregistered properties of the rank statistic: (i) with 41 nodes the rank is
**discrete**, while the KS band (`kstwo.ppf(0.95, 12) = 0.375430`, `(0.99, 12) = 0.449045` — both
re-derived exactly, matching §4.4's 0.3754/0.4490) assumes a continuous null; (ii) the grid is
**truncated** at [0.60, 0.86] while §5.1 invariant 2 also pins `h_bounds = (0.50, 0.86)` — a
railing posterior piles ranks at 0 or 1 and Uniform(0,1) is not the correct null under truncation;
(iii) at 4 seeds the exact central-95% acceptance region is `k ∈ [1,4]` — **re-derived and
confirmed**, but note `k = 0` **is** excluded, so §4.4's "vacuous" is a conservative
over-statement; REPORTED-ONLY remains the right call. The n = 12 region `5 ≤ k ≤ 11` is re-derived
and **confirmed exactly**.

**Correction.** Pin the support to `H_GRID_41` for **both** functions and for the rank/KS null;
register the rank tie-breaking rule (randomized vs. mid-rank) since it changes the KS statistic at
41 nodes; and add the truncation caveat to §4.4's "Registered asymmetry" paragraph.

---

### PA-HIER-15 — MEDIUM. The D4 decode is collision-free (verified), but the seed design has two unregistered properties and the Stage-F offset is underspecified.
**Date:** 2026-08-26 · **Instrument run:** none · **Severity:** MEDIUM · **Status:** FIXED (registrations added)
**Supersedes:** §7.3's array-design block and its "collision-free by construction" paragraph;
§3.8 GATE SEEDS.

**Decode, explicitly constructed and verified — NO defect.** With `theta_idx = TID // N_SEED`,
`seed_idx = TID % N_SEED`:

| TID | Stage P (N_SEED=4) → (θ_idx, seed_idx) | SEED |
|---|---|---|
| 0 | (0, 0) | 940001 |
| 3 | (0, 3) | 940004 |
| 4 | (1, 0) | 940001 |
| 17 | (4, 1) | 940002 |
| 35 | (8, 3) | 940004 |

Integer div/mod on `[0, N_THETA·N_SEED)` is a bijection onto `{0..N_THETA−1} × {0..N_SEED−1}`; no
two task ids decode to one (θ, seed) cell, at either stage. The prereg's F8 non-recurrence argument
is also **confirmed correct**: F8 (`PREREGISTRATION_P3_2D_20260825.md` PA-2D-8 item 1) was a
*within-task chunk-seed* overlap fixed by a ×100 stride (`cluster/p3_2d_rhs2.sbatch:76-81`,
`SEED=$((BASE_SEED + 100 * TID))`), and [HIER] has no within-task chunk loop. `BASE_SEED_HIER =
940001` is clean against every seed literal greppable in `cluster/` and
`results/campaign51_20260728/realistic_20260729/*.py` (found: 900101–900125, 960001, 970001, 980001,
983104, 990001, 999999 — nothing in 94xxxx). **GATE SEEDS is retained** since a grep is not an audit.

**Three registrations added.**
1. **Stage P's seed set is a strict subset of Stage F's** (940001–940004 ⊂ 940001–940012), and GATE
   P→F conditions the launch on those same four realizations. The Stage-F coverage read is therefore
   **not fully out-of-sample**: 4 of its 12 seeds passed a data-dependent stopping rule. Registered
   as §5.2 blindness item 9, and the Stage-F coverage count is additionally reported on the
   **8 fresh seeds only** (940005–940012) alongside the 12-seed number, with the 12-seed band
   band-bearing and the 8-seed number REPORTED-ONLY.
2. **Stage F's `THETA_ROW offset` is not defined.** Registered: sub-array `r ∈ {0..4}` runs
   `--array=0-59` and computes `global_id = 60·r + SLURM_ARRAY_TASK_ID`, then applies the *same*
   div/mod decode to `global_id`. `r` is passed by `--export=THETA_ROW=r`, echoed into the task JSON,
   and a driver assertion requires `0 ≤ global_id < 300` and that the (θ, seed) cell's out-root does
   not already exist.
3. **S0-R's `realization_seed` equals its draw seed** (`host_pool_for_sigma_scale(..., seed, ...)`
   at `correspondence_1d.py:1889` and `draw_realization(seed)` share the integer). Registered as a
   deliberate pin, not an accident; if S0-R is rebuilt per PA-HIER-3 the two streams are separated
   and the separation is registered.

---

### PA-HIER-16 — MEDIUM. GATE T-ID cannot detect a missing hook, and GATE ENG as written cannot isolate site 2.3.
**Date:** 2026-08-26 · **Instrument run:** none · **Severity:** MEDIUM · **Status:** FIXED (gate forms strengthened)
**Supersedes:** §3.1's scope sentence and §3.4.

**Found.** §3.1 mandates *"a **literal early-return/skip at (b, s) == (0.0, 1.0)**"* — which is the
right call for byte-identity (the IEEE-754 reordering argument is correct: `host_z_error * 1.0` is
exact but `sqrt(x**2 * s**2 + pv**2)` need not round like `sqrt((x*s)**2 + pv**2)`), **but it makes
T-ID vacuous as evidence that a hook exists at all**: a path with **no** hook passes T-ID
identically. T-ID certifies the default, never the wiring; only ENG can certify the wiring, and §4.5
routes a T-ID failure to INSTRUMENT-DEFECT while a *missing* hook produces no failure anywhere.

§3.4's ENG form — *"≥10% of scored events move by ≥1e-6 relative … measured independently on each
in-scope dispatch path"* — is ill-posed for site 2.3: the global selection denominator is a per-h
**scalar** shared by every event, so perturbing it moves **100%** of events by the same relative
amount. ENG on site 2.3 therefore passes trivially and would also pass if the *numerator* hooks
(2.1/2.2) were missing entirely.

**Correction, registered.** ENG becomes a **path-isolated toggle matrix**: for each in-scope site
`k`, run θ ≠ truth with the hook active **only at site k** and all others forced to their θ = (0,1)
values, and require (i) ≥10% of events move ≥1e-6 relative, **and** (ii) for the global-denominator
site specifically, that the movement is **not** a single common multiplicative factor across all
events (a rank-1 offset is the signature of "only the denominator moved"). Additionally: a **hook
inventory assertion** — the driver asserts at runtime that each registered site's θ-aware code
object was imported and its counter incremented at least once per task, stamped into the task JSON
next to the A22 flags. Both are zero marginal cost.

---

### PA-HIER-17 — MEDIUM. The `s < 1` nodes shrink the estimator's own ±4σ window below the generating width, producing heavy negative-`lnL` tails at exactly the nodes `score_lns` consumes.
**Date:** 2026-08-26 · **Instrument run:** none · **Severity:** MEDIUM · **Status:** OPEN
**Supersedes:** nothing; adds a gate and a §5.2 item.

**Found.** Every host-z kernel in this pipeline is evaluated on a hard `±4σ_eff` window:
`bayesian_statistics.py:6242-6244` / `:6899-6901` (denominator z-clamp,
`integration_limit_sigma_multiplier = 4.0` at `:6874`), `:1676-1677`
(`lo = max(zc − 4·se, 1e-6)`), and `correspondence_1d._host_kernel_window`
(`_B0I_KERNEL_SIGMA_MULTIPLIER = 4.0`, `:1157`). Under `s < 1` the **estimator's** window narrows
while the realized `z_true` stays where the generator put it. Fraction of a Gaussian truth outside
the estimator's window:

| node | effective window | mass outside |
|---|---|---|
| s = 1/√2 (a `score_lns` node) | ±2.83σ | **0.468 %** |
| s = 0.50 (Stage-P/F grid corner) | ±2.00σ | **4.55 %** |

Those events land in the far tail of the numerator quadrature, contributing large negative
per-event `lnL`. Combined with PA-HIER-5's untailored `mean/SEM`, a handful of such events can
dominate both `mean(score_lns)` and `SEM(score_lns)` — the heavy-tail band-vacuity class that a
control caught in the b0-identity thread.

**Registered.** (i) **GATE WINDOW**, zero-compute-on-banked-data: at every θ node with `s < 1`,
report the count and fraction of events whose realized `z_true` falls outside the estimator's own
`±4·s·σ_eff` window; if that fraction exceeds 1% at any band-bearing node, the node is REPORTED-ONLY
and its `lnL` may not enter `score_lns` or §4.2's `Δ ln L`. (ii) §5.2 gains item 10: *the design
cannot distinguish "the data disprefer a narrower kernel" from "the estimator's fixed ±4σ window
truncated the truth" — the two have the same sign and the same node dependence.* (iii) A registered
alternative if GATE WINDOW fails: scale the window multiplier with `1/s` so the window covers a
fixed number of **generating** σ, and re-run GATE T-ID (this is an instrumentation change, and it
changes the `s` axis's meaning, so it needs its own amendment before use).

---

### PA-HIER-18 — LOW (bundle). Six smaller defects, corrected in place.
**Date:** 2026-08-26 · **Instrument run:** none · **Severity:** LOW · **Status:** FIXED unless noted

1. **§2.3's "nothing computed at Stage 0 is thrown away" is false.** Stage 0 runs `n_h = 1` (§2.1,
   §7.2) while Stages P/F run `n_h = 41`; the five Stage-0 cells are not reusable cubes, only a
   single h-slice each. The five nodes *are* Stage-F grid nodes (verified against §2.3's lists), so
   the claim should read "no Stage-0 **node** is off-grid". Costing is unaffected (§7.2 already
   prices Stage P at the full 36 × 41).
2. **§4.1's "Anchor derivation for B0-B" names a branch that is never defined.** No row `B0-B`
   exists in §4.1's table. Registered: `B0-B ≡ B0-A/B0-A′'s bands applied to S0-B`, still NOT
   AUTHORIZED.
3. **§2.5's "eleven" stamps cannot be checked**: three of the six inherited stamps are referenced
   only as *"the three already in the C-A set"* and are never named. Registered obligation: the
   driver enumerates all stamps by literal key in the task JSON, and GATE STAMP compares against a
   committed key list, not a count. Per PA-HIER-10 the set is now **twelve**.
4. **§4.5/§6's N-scaling falsifier is uncosted and its band is undefined.** "repeat Stage P at 2×
   the per-seed event count on 2 seeds" at `n_h = 41` costs, on §7.1's own formula,
   `2 seeds × 9 θ × (0.2843·41·2 + 0.1333) ≈ 420 CPU-h` — comparable to Stage P itself, and it
   appears nowhere in §7.2. The band `|Δ lnL(2N)/Δ lnL(N) − 2| ≤ 3σ` never says what σ is (the two
   estimates are correlated if the N-sample is nested in the 2N-sample). **Status: OPEN** —
   register a costing line and a variance definition, or downgrade the falsifier to "unrun ⇒ the
   UNIDENTIFIABLE attribution stays explicitly provisional" per [A14].
5. **§5.1 invariant 10 carries a pre-launch action inside an invariants table.** *"a
   `trunc_lognormal` resolution combined with a point-resolving `host_z_kernel` hits a pre-existing
   prior-consistency guard (`bayesian_statistics.py:6175-6178`) and must be smoke-checked before the
   pilot"* — verified at source (`resolve_host_mass_kernel` … *"the point-z × trunc-mass combination
   raises (prior-consistency guard)"*, `:6172-6178`). A must-do-before-launch check belongs in §3 as
   a gate. Registered as **GATE MASS-KERNEL**, pre-launch, blocking.
6. **§3.7/§7.1's `MaxArraySize`/`MaxSubmitJobs` NOT FOUND is correctly disclosed** and correctly
   routed to the live `/cluster` preflight; the 5×60 chunking is inside the 49/80-task in-repo
   precedent band. **No defect** — recorded so the review's silence is not read as an omission.
   Likewise **§7's costing arithmetic was independently re-derived and is correct**: 63.97·16/3600 =
   0.28431; 30·16/3600 = 0.13333; S0-A/S0-R 8.35; S0-C 11.79; Stage 0 28.49; Stage P 424.44; Stage F
   807.6 / 1319.4 / 3537.0; per-task walltime 2652.8 s (×1.7 = 75 min < 2 h). All match.

---

### REVIEW SUMMARY — items attacked and found SOUND (recorded so silence is not read as endorsement)

- **§4.5's MIXED branch is genuinely first-class** (trigger set, stage-5 mapping, a banked quantity,
  a named free next measurement); only its statistic is wrong (PA-HIER-13). §4.2's MIXED row
  likewise. §4.5 also carries CONTROL-FAIL as a separate branch, honouring PA-2D-9.
- **The D4 decode is collision-free** (PA-HIER-15, explicit construction) and the F8 non-recurrence
  argument holds.
- **Every band anchor that claims exactness was recomputed and matches**: `χ²₂(0.95)/2 = 2.99573`,
  `χ²₂(0.6827)/2 = 1.14791`, `kstwo.ppf(0.95,12) = 0.375430`, `kstwo.ppf(0.99,12) = 0.449045`,
  binomial central-95% `n=4 → [1,4]`, `n=12 → [5,11]`, `2·0.035/1.485 = 0.047138`,
  `1.5·(1−1/√2) = 0.43934`, `0.5·ln√2 = 0.17329`. The `k*` **algebra** is right for its stated case
  (PA-HIER-12 corrects only its scope). `ε_h = 0.005` is verified against `H_GRID_41`
  (`correspondence_1d.py:351-356`: step 0.005 across 0.65–0.79) and `h = 0.73` is an exact node.
- **§7.4's STALE-quantity discipline is honoured in §7** — no STALE point number from that list is
  quoted as a band anchor anywhere in §§2–6. The **one leak is the reverse direction**: a *fresh-
  looking* number (`σ_z = 0.035`) that is out-of-configuration (PA-HIER-9). The one FRESH number
  (−0.635 ± 0.017) is verified in the ledger (`:1647-1648`, `:1687-1690`) and `z ≈ 0.485` at
  `:1355`.
- **Scope (rule 1).** Nothing here re-opens a standing exoneration. §1.6's G2b/C7 delimitation is
  sound as reasoning and correctly made **conditional on GATE D3**; the #68/#62 adjacency is
  declared rather than re-litigated; the F5-vs-#41/#52 vocabulary fence is correct and binding.
  §1.5's no-production-kernel-change guard holds: every registered hook is instrumentation with a
  byte-identical default, so no `/physics-change` gate is triggered by the *edits* themselves.
  **One caveat registered:** GATE D3(a)'s forcing (as amended in PA-HIER-10, now an unconditional
  arm pin) changes how a shipped production flag resolves as a function of an instrument parameter.
  It must be implemented so the forcing is reachable **only** from the θ-instrumented entry points
  and can never alter a production default; a regression test asserting the production default
  remains `False` ships with the hook.
- **§1.3's [86] non-citability clause, §1.4's verbatim scope fence, §5.1's `NEVER`-audited
  disclosures and GATE PARITY (a genuinely load-bearing, genuinely never-audited invariant, audited
  this cycle at zero compute) all meet the [A10] bar.** GATE PARITY is additionally *vindicated* by
  this review: the stale anchor it flags (`correspondence_1d.py:1173` cites
  `bayesian_statistics.py:5908-5909`, while the live sites are `:6223-6224`/`:6878-6879`) is
  confirmed stale at source — and PA-HIER-2 shows the drift is worse than a renumbering, since the
  two objects are no longer even on the same side of the generator/estimator line.

---

### LAUNCH VERDICT — **LAUNCH-BLOCKED**

No `sbatch` for any [HIER] stage, and no Stage-0 CPU-h, until PA-HIER-1, -2, -3, -4, -6 and -7 are
resolved by the author. PA-HIER-1/-2/-3 are venue/instrument identity failures that would make every
banked number uninterpretable; PA-HIER-4 is a mis-formed registered statistic of the PA-2D-8 F3
class; PA-HIER-6/-7 are undefined registered objects that three verdict families depend on. The
remaining blocks are corrections registered in place and do not by themselves block.

**Cheapest path back to LAUNCH-READY (all zero-compute):** (1) register `host_mode` and audit the
generator/estimator kernel identity for the chosen mode (PA-HIER-1); (2) re-pin CoR-M to site 2.2
and add GATE GEN-FROZEN (PA-HIER-2); (3) either build a z-only estimator-side positive control or
disarm D7's early exit and re-scope Stage 0 to S0-A + S0-C only (PA-HIER-3); (4) adopt `score_lns`
(PA-HIER-4); (5) register the θ prior + the prior-sensitivity leg (PA-HIER-6); (6) pin the
identifiability statistic's h-treatment (PA-HIER-7); (7) re-anchor `b` on a measured catalogue
statistic (PA-HIER-9). Items (4)–(7) change no node and no costing line.

---

## ZERO-COMPUTE BLOCKER RESOLUTION PASS — PA-HIER-19 … PA-HIER-26
**(2026-08-27, NO instrument has run, zero compute spent. Independent source re-verification of the
six LAUNCH-BLOCKED items, run as two adversarial reads — blocker A: generator law / truth-θ / hook
side; blocker B: control / prior / identifiability / wiring. Purpose: turn each blocker from
"unknown" into a decision the author can answer in one line, or resolve it outright where it is a
matter of fact rather than judgement. This pass rules on nothing. Nothing above the divider is
edited; each block names what it supersedes.)**

---

### PA-HIER-19 — PA-HIER-1 RESOLVED as fact-finding. The five generator laws are enumerated and truth-θ is derived per mode: exactly **two** modes admit truth-θ = (0, 1), and the prereg's implicit default is not one of them.
**Date:** 2026-08-27 · **Instrument run:** none · **Severity:** BLOCKER (fact half RESOLVED) · **Status:** OPEN-FOR-AUTHOR (one [RULE])
**Supersedes:** PA-HIER-1's *"Correction (required before commit)"* paragraph (discharged by the
table below); §5.1 (adds the missing invariant); §2.1's S0-A/S0-R arm rows insofar as they assume
the default law.

**Found.** `draw_realization`'s `host_mode` is a five-valued `Literal`, default `"catalogue"`
(`correspondence_1d.py:1897-1903`), assigned per arm by `ARM_HOST_MODE` (`:452-473`). The truth
value of θ = (b, s) was derived from each branch's own generating law at source:

| `host_mode` | arms (`:452-473`) | generating law for `z_true` | truth-θ = (b, s) |
|---|---|---|---|
| `catalogue` **(default)** | b0, bsig005, bsig025, eden05, eden2, bf1 | delta at the catalogue's own listed z — `host_z = pool.z[host_idx]`, `true_d_L = dist_vectorized(host_z, h=H_TRUE)` (`:2003-2012`, `:2141`); no separate z_true draw | b **undefined** (zero width), **s → 0** — confirms PA-HIER-1 |
| `population` | bout | `draw_population_redshifts(rng, n, h=H_TRUE)` (`:2013-2020`); no catalogue `z_g`/`z_error` involved | **axis inapplicable** |
| `population_selected` | bsel, bself, bden | `draw_selected_population_redshifts(rng, n, completeness, phi_survival_table, h=H_TRUE)` (`:2021-2052`); same | **axis inapplicable** |
| `catalogue_selected` | **b0i** | per-event draw from `k_g(z)·w_pop·f_k·S̄_φ(z; H_TRUE)` on the host's own ±4σ window (`:2053-2088`; kernel `:1440-1499`) | **(0, 1)** |
| `catalogue_selected_2d` | b0i2d | byte-for-byte the same z-draw law (`:2090-2131`; `_B0i2DLatents` docstring `:1572-1575`, *"UNCHANGED from the 1D `catalogue_selected` mode"*), plus an orthogonal latent-mass extension | **(0, 1)** |

**One finding stronger than PA-HIER-1.** The two `population*` modes are not merely "truth-θ is
something else" — those hosts carry **no `(z_g, z_error)` pair at all** (`host_index_col = -1`,
`in_catalog = False`), so the θ = (b, s) photo-z kernel is **not a term in their generative law**.
Those arms carry **zero [HIER] information content on θ**, in any direction, at any node. PA-HIER-1
contrasted only `catalogue` against `catalogue_selected*` and did not reach this.

**Correction, registered.** (i) `host_mode` is added to §5.1 as an invariant, with its file:line,
pinned per arm. (ii) The [HIER] venue is registered as **`host_mode="catalogue_selected"` (arm
b0i)**, with `"catalogue_selected_2d"` (b0i2d) as its z-axis-identical sibling should the 2D mass
extension ever be wanted. This is a **one-line `host_mode` change** from the prereg's implicit
default (arm b0), **not a redesign**. (iii) No other mode is admissible: `"catalogue"` gives
s → 0 and both `population*` modes make the axis inapplicable. This is a structural fact about the
harness, not a detail — there is exactly one available venue for this experiment.

**What remains for the author.** Only the [RULE] itself: ratify the venue switch b0 → b0i on the
evidence in this document. The fact-finding is complete.

---

### PA-HIER-20 — PA-HIER-1's demanded generator/estimator kernel-identity audit, executed. **Five legs certified at source; two legs NOT certified**, one of them the PA-2D-2/-3 borrowed-quadrature failure shape one axis over.
**Date:** 2026-08-27 · **Instrument run:** none · **Severity:** BLOCKER (audit half RESOLVED) · **Status:** RESOLVED (5 legs) / NEEDS-CODE (2 legs) + one author line
**Supersedes:** PA-HIER-1's closing sentence (*"that identity is an unaudited load-bearing invariant
and must be added to §5.1 with `NEVER` and audited this cycle"*) — the audit is below; §5.2 item 3's
blindness statement.

**Certified at source (b0i / `catalogue_selected`).**

1. **Gaussian loc/scale match exactly at (b, s) = (0, 1).** Generator: `norm.pdf(z_i_grid,
   loc=host_z[i], scale=z_error_eff[i])` — loc **unshifted**, scale **unscaled**
   (`correspondence_1d.py:1490`, inside `_draw_kernel_survival_redshifts`). Estimator:
   `galaxy_redshift_normal_distribution = norm(loc=host_z, scale=host_z_error_eff)`
   (`bayesian_statistics.py:6247`). Same two arguments, same law.
2. **`w_pop · f_k` is live estimator-side, not a generator-only artefact.** It is the estimator's own
   `volume_deconv` host-z prior (`bayesian_statistics.py:6335-6339`), live because
   `PRODUCTION_FLAGS["--host_z_kernel"] = "volume_deconv"` (`correspondence_1d.py:328-337`).
3. **The `S̄_φ` factor is live estimator-side.** `catalogue_numerator_survival` defaults to `"auto"`
   and resolves `"phi" if normalization_mode == "absolute_marginal" else "off"`
   (`bayesian_statistics.py:3535-3541`, quote-verified), and the mirror venue runs
   `absolute_marginal` (`correspondence_1d.py:328-337`). So the generator's extra
   `w_pop·f_k·S̄_φ` factors are **the estimator's own terms**, not a mismatch.
4. **Both sides call the same imported production functions** — `comoving_volume_element`,
   `_completeness_at_host_nodes`, `_host_pixels`, `precompute_phi_marginal_survival`
   (`correspondence_1d.py:227-262`) — not parallel reimplementations.
5. **The `S̃_φ` normalization quadrature matches production's, and this is explicitly NOT a
   PA-2D-2/-3-class mismatch.** Generator: `_B0I_KERNEL_QUAD_N = 50`, *"mirrors `_HOST_QUAD_N`'s
   default"* (`correspondence_1d.py:1156-1157`). Estimator: `_HOST_QUAD_N = 50`, `FIXED_QUAD_N =
   _HOST_QUAD_N` (`bayesian_statistics.py:409`, `:6139`). Same rule (Gauss–Legendre), same node
   count. Recorded so the review's silence is not read as an omission.

Consequently the generator's *"draw z_true given the fixed observed z_g"* conditional equals, by
Gaussian symmetry, the forward-model conditional a true (z_true ~ prior, z_g = z_true + N(0, σ))
process would produce, at (b, s) = (0, 1) and only there. **Registered caveat on what this buys:**
truth-θ = (0, 1) at b0i is a *self-consistency* identity of the venue's own construction, so a null
at S0-A certifies **wiring and arithmetic**, not physics. §4.1's B0-A′ disposition (a non-zero score
here is a bug, STOP) is correct precisely because of that, and is now on a certified footing.

**Ancillary fact, registered.** `SIGMA_V_PEC_KM_S = 0.0` (`constants.py:95`), so
`host_z_error_eff` is presently the **identity** on the quoted `z_error` on both sides
(`correspondence_1d.py:1186-1188`; `bayesian_statistics.py:6223-6224`, `:6878-6879`). Parity is
exact and `s` scales the quoted column directly, with no peculiar-velocity floor damping it. If that
constant is ever set non-zero, `s` stops being a pure scale on the catalogue column and this
amendment must be revisited.

**NOT certified — two legs, both requiring compute or new code.**

(a) **Value-identity of the two independently constructed `phi_survival_table` objects** — the
    generator's (via `build_bsel_selection_objects`) versus `evaluate()`'s own internal build. Both
    are deterministic given identical inputs, but **no runtime equality assertion exists** anywhere
    in the code read. This is an assumed, unasserted invariant sitting directly under leg 3.
(b) **The 401-node uniform inverse-CDF draw grid** (`_B0I_ZTRUE_GRID_N = 401`,
    `correspondence_1d.py:1164`, consumed at `:1490-1498` via `np.linspace` + `_inverse_cdf_draw`)
    is a **different numerical operation** from the GL-50 normalization certified in leg 5, and it
    is **un-audited in the wide-window / near-horizon regime**. This is exactly the
    PA-2D-2/PA-2D-3 borrowed-quadrature failure shape, one axis over. It needs a
    convergence / brute-force spot-check (401 vs 4001 nodes, and against a direct rejection sample)
    **before any b0i-mode number is banked**. That check costs compute and is out of this pass's
    zero-compute scope.

**Registered as §5.1 invariants with `NEVER`-audited status** for (a) and (b), pending the author's
line on whether they are pre-launch gate items or disclosed residual risk.

---

### PA-HIER-21 — The θ instrumentation **does not exist anywhere in the repository**. Six estimator-side hook sites are pinned here; the generator-side site is excluded; and because the hook lands inside a physics-trigger file, the §1.5 scope question is re-opened as an author decision.
**Date:** 2026-08-27 · **Instrument run:** none · **Severity:** BLOCKER · **Status:** NEEDS-CODE + OPEN-FOR-AUTHOR ([DO] + a /physics-change scope ruling)
**Supersedes:** §2.4's site table (as already amended by PA-HIER-2/-10/-11) — re-stated in full
below; the LAUNCH VERDICT's *"Cheapest path back to LAUNCH-READY (all zero-compute)"* framing;
`correspondence_1d.py:1173`'s docstring citation.

**Found.** `grep -n "theta_b\|theta_s\|--theta" darksiren_emri/arguments.py` returns **zero hits**.
Nothing in the repository implements θ — no CLI flag, no parameter, no hook, no default. Every
[HIER] stage, **including the cheapest one (S0-A)**, requires new code before a single CPU-h can be
spent. The LAUNCH VERDICT's seven-item path is correctly described as zero-compute, but it is not a
path to a runnable experiment: it leaves the instrument unbuilt. That is registered here so the
distinction between *"the registration is clean"* and *"the experiment can run"* is never elided.

**The six σ_z sites, re-stated with their θ dispositions.**

| site | file:line | role | θ hook |
|---|---|---|---|
| **2.1** | `bayesian_statistics.py:6223-6224` (width); kernel object `:6247`; prior `:6335-6339` | scalar `single_host_likelihood` | **YES** |
| **2.2** | `bayesian_statistics.py:6878-6879` (width); window `:6899-6901` | `single_host_likelihood_batch` — production's **actual** dispatch path | **YES** |
| **2.3** | `bayesian_statistics.py:1669-1672` (`sigma_eff`); window `:1675-1679`; `precompute_global_catalog_selection` `:2657-2882`, call `:2872` | global selection denominator — a per-h **scalar** shared by every event | **YES** (per PA-HIER-10) |
| **2.7** | `bayesian_statistics.py:7518-7519` | integration-testing fidelity twin | **NO** — regression obligation only (PA-HIER-11) |
| **2.4** | `correspondence_1d.py:1167-1188`, callers `:1323`, `:1485` | **GENERATOR-side** | **FORBIDDEN** — GATE GEN-FROZEN (PA-HIER-2) |
| — | `validation/pp_coverage.py` | independent reimplementation | OUT OF SCOPE (§2.4) |

**The reparametrization, registered.** θ = (b, s) enters as a new parameter threaded into
`BayesianStatistics.evaluate()`, applying `host_z → host_z + b·(1 + host_z)` and
`host_z_error_eff → s · host_z_error_eff` at each in-scope estimator site, with a **literal
early-return/skip at (b, s) == (0.0, 1.0)** per GATE T-ID (§3.1, as read by PA-HIER-16). It is
**not** implemented at `correspondence_1d.host_z_error_eff` — that would make θ move the data
(PA-HIER-2), and GATE GEN-FROZEN forbids it.

**Scope question, re-opened and handed to the author.** `bayesian_statistics.py` is on CLAUDE.md's
`/physics-change` trigger list. The REVIEW SUMMARY's judgement — byte-identical default at (0, 1) ⇒
no computed production value changes ⇒ no gate — is defensible, and this pass does not dispute it.
But it is a **ruling the author owns, not an assumption a reviewer may bank**, for two reasons: the
edit is to a trigger file by the letter of the rule, and PA-HIER-10's now-unconditional
`smear_sigma_z = True` arm pin changes how a **shipped production flag resolves** as a function of
an instrument parameter. Registered as an explicit decision, with the REVIEW SUMMARY's own
mitigation retained: the forcing must be reachable only from θ-instrumented entry points, and a
regression test asserting the production default stays `False` ships in the same commit.

**Trivial fix, bundled.** `correspondence_1d.py:1173` cites *"`bayesian_statistics.py:5908-5909`"*
as the byte-identical production form. Confirmed **stale** at source (that range is unrelated code).
Correct targets: `:6223-6224` (scalar) and `:6878-6879` (batch). Non-scientific; authorize with the
hook commit. This is the drift GATE PARITY flagged, now with its replacement text.

---

### PA-HIER-22 — PA-HIER-3 upgraded from AUTHOR-RULING-NEEDED to **NEEDS-CODE**, and the obvious control construction is shown to carry **its own confound**. No configuration anywhere in this repository injects a z-kernel misspecification.
**Date:** 2026-08-27 · **Instrument run:** none · **Severity:** BLOCKER · **Status:** NEEDS-CODE (residual choice OPEN-FOR-AUTHOR)
**Supersedes:** PA-HIER-3's *"Correction"* paragraph (its two proposed constructions are shown
incomplete below); PA-HIER-15 registration 3's contingency.

**Found, re-verified at the correct path.** The control lives in
`darksiren_emri/galaxy_catalogue/observed_realization.py`, not under `validation/`.

1. **`z_error` is round-tripped verbatim, deliberately.** `_realize_and_write`'s write block
   (`:454-462`) rewrites only the z, M\*, and M\*-error columns; the z-error column is never
   touched. The docstring states the property that makes S0-R a null instrument, in the harness's
   own words (`:185-187`): *"the z width law is scale-free in z, so the stored column IS the width
   the kernel consumes and `sigma_kernel == sigma_realized` identically."*
2. **One catalogue handler serves both sides.** `host_pool_for_sigma_scale`
   (`correspondence_1d.py:1850-1891`) returns a single `GalaxyCatalogueHandler`, and its own
   docstring (`:1868-1871`) says it is *"the SAME object the host pool was extracted from, for
   direct reuse as `BayesianStatistics.evaluate`'s `galaxy_catalog` argument"* — confirmed at
   `run_mirror_seed_inprocess:2844`. **The generating width and the estimator's quoted width are the
   identical number at every `sigma_scale`.** `sigma_scale` perturbs *which galaxy sits at which z*,
   never a width mismatch. PA-HIER-3 is confirmed in full.
3. **No knob exists.** Grep over `arguments.py`, `correspondence_1d.py` and `observed_realization.py`
   found no `z_error`-only scale parameter. `--smear_global_selection` (`arguments.py:773-786`) only
   toggles whether site 2.3 integrates a kernel at all; it has no bearing on data generation.

**Two new findings that change the shape of the fix.**

(a) **Trap, flagged.** Feeding the estimator the *unscattered parent* catalogue does **not** inject
    an s-axis mismatch. It reproduces PA-HIER-1's `s → 0` confound instead, and additionally moves
    the assumed **centre**, not only the width. It is not a fallback and must not be used as one.

(b) **The obvious construction is not equivalent to the θ hook — it perturbs the candidate list.**
    Because `SIGMA_V_PEC_KM_S = 0.0` (`constants.py:95`), writing `z_error_est = z_error_gen /
    s_gen` into an estimator-facing copy would reproduce the s-scaling **exactly** at sites
    2.1/2.2/2.3 (PA-HIER-20's ancillary fact). But the quoted `z_error` column is **also a
    candidate-list input**: `handler.py:250` prunes the parent catalogue on `redshift −
    redshift_error ≤ z_max`, and `handler.py:636-644` builds each event's candidate-host list on
    `z_min ≤ z + z_error` and `z_max ≥ z − z_error`. A column-rewrite control therefore changes
    **which galaxies are candidates at all**, which the estimator-side θ hook does not — the
    production comment at `bayesian_statistics.py:6218-6222` records that the candidate window
    *intentionally* keeps the bare catalogue `z_error`. **The control and the lever would not be
    measuring the same perturbation.** That is a CONTROL-FAIL of the PA-2D-9 class built into the
    control's own construction, and PA-HIER-3's proposed *"multiply the quoted `z_error` column by
    1/1.5"* fix walks straight into it.

**What a genuine s-axis positive control now requires** (all new code, none of it a parameter tweak):
(i) a second, **estimator-facing** catalogue that byte-copies the generator-facing realization and
rewrites only the quoted width column; (ii) new driver plumbing to carry **two decoupled catalogue
handlers** through one mirror-seed run — today there is exactly one, shared; and (iii) an explicit
decision on the candidate-list confound — either freeze the candidate list from the
**generator-facing** column (new code inside `handler.py`'s search path, i.e. a third instrument) or
disclose the confound and accept that the control perturbs candidacy as well as width, in which case
its verdict is REPORTED-ONLY by construction.

**Unchanged and reinforced:** PA-HIER-3's ruling that **no LEVER-DEAD-AT-N verdict may bank** and
that D7's early exit is **unarmed** stands. The residual author choice is build-vs-fallback, where
the fallback is: disarm D7's early exit and re-scope Stage 0 to **S0-A + S0-C only**.

---

### PA-HIER-23 — GATE ENG's toggle matrix given a decisive, registered numeric form. The site-2.3 vacuity is closed by testing the shift's **uniformity and independent recomputation**, not an event-count fraction.
**Date:** 2026-08-27 · **Instrument run:** none · **Severity:** MEDIUM · **Status:** RESOLVED as registration; NEEDS-CODE to implement
**Supersedes:** PA-HIER-16's *"Correction, registered"* paragraph (sharpened, not reversed) and §3.4.

**Registered form.** A **path-isolated one-at-a-time (OAT) toggle matrix**, not a full 2³ factorial:
baseline (θ = (0, 1), all hooks disabled — today's T-ID), then one run per in-scope site
(2.1-only, 2.2-only, 2.3-only) at a non-zero θ with that site's hook active and the **others forced
to their θ = (0, 1) evaluation**, plus one all-sites run. Site 2.4 is excluded by construction
(GATE GEN-FROZEN forbids a hook there).

**Measurement.** Extend the existing per-event diagnostics CSV to decompose each event's `ln L` into
its **numerator-log-term** and **denominator-log-term** separately. Reading only the aggregate
`ln L` is exactly what cannot catch a missing site while the others are live.

- **Sites 2.1 / 2.2 (per-host numerator).** PRESENT iff (i) ≥ 10 % of events move ≥ 1e-6 relative in
  the **numerator-log-term**, **and** (ii) the per-event magnitude tracks that host's own
  `(z, z_error)` — spot-check 2–3 hosts against a hand-computed closed form to machine precision.
  **MISSING = zero numerator-log-term movement in that site's own single-site run**, decisive
  regardless of what the all-sites run shows.
- **Site 2.3 (global denominator).** Its correct positive signature is a **single shift applied
  identically to every event**, not heterogeneity. PRESENT requires (i) the shift is bit-identical
  in magnitude across all events; (ii) it is bit-identical to baseline (i.e. absent) in the 2.1-only
  and 2.2-only runs, confirming isolation held; and (iii) it matches an **independent
  recomputation** of `_smeared_global_pdet_expectation` (`bayesian_statistics.py:1619-1720`)
  evaluated directly at the same (h, θ) outside the pipeline. MISSING = zero denominator-log-term
  movement in the 2.3-only run. This is what breaks the *"100 % of events moved, trivially"*
  vacuity: the test is on the shift's **uniformity and independent match**, never on a raw
  event-count fraction.

**Complementary, not a substitute.** PA-HIER-16's hook-inventory assertion (the driver asserts each
site's θ-aware code object was imported and its counter incremented, stamped in the task JSON) proves
the path was **entered**; it does not prove the arithmetic **took effect** — a compute-then-discard
bug passes it. It is retained as a cheap corroborant; the toggle-matrix numeric test is the decisive
evidence.

**Cost.** The toggle matrix is 4 extra single-h runs at Stage-0 scale (well inside §7.2's Stage-0
line); the diagnostics columns and the site-toggle switch are instrumentation-only and ship in the
same commit as PA-HIER-21's hook.

---

### PA-HIER-24 — PA-HIER-6 (θ prior): three candidate measures enumerated and costed. **Option B is free**, is consistent with the document's own log-uniform commitment, and one author line closes the blocker.
**Date:** 2026-08-27 · **Instrument run:** none · **Severity:** BLOCKER · **Status:** OPEN-FOR-AUTHOR (one line)
**Supersedes:** nothing — this discharges PA-HIER-6's correction item (i) as fact-finding and adds a
second sensitivity leg.

**The three options, with their consequences.**

- **(A) Discrete uniform on the registered grid nodes, equal weight.** The implicit reading of the
  present text. `k` is then set almost entirely by the grid's own half-width rather than by data —
  and PA-HIER-9's own measured b-anchor correction (4–10× wider than the registered ±0.04) would
  change `k` and could flip FAVOURABLE ↔ UNFAVOURABLE TRADE with **zero new likelihood
  evaluations**. Also carries an edge-weighting bias at the grid boundary.
- **(B) Uniform in `b`, uniform in `ln s`, over an explicitly STATED continuous support, realized by
  proper quadrature weights (Simpson/trapezoid) on the SAME already-computed nodes.** Zero marginal
  compute; removes A's edge-weighting bias for free. The *"support = grid extent"* limitation
  persists, because nothing exists outside the registered nodes — that is a fact about the design,
  not about the prior, and must be disclosed either way.
- **(C) A stated weakly-informative continuum prior** (e.g. Gaussian, scale tied to a measured
  catalogue statistic) **with its own quadrature nodes.** Only trustworthy with **more** likelihood
  evaluations than the registered grid provides — a real costing consequence against an already
  807–3537 CPU-h Stage F — and it collapses back to B if forced onto the existing 5 nodes.

**Recommendation (not a ruling): B.** It discharges PA-HIER-6 at zero marginal compute, matches this
document's already-registered log-uniform-in-`s` convention (§2.3's ×√2 grid spacing; PA-HIER-4's
`score_lns`; §4.1's B0-M and B0-P bands, all stated in `ln s`), and removes A's discretization bias
for free.

**Two independent sensitivity legs, both free, answering different questions.**
(i) **Support-width sensitivity** — PA-HIER-6's own registered leg: recompute `k`, `t` and the §4.4
rank on the 3-node Stage-P sub-grid; a verdict that flips under narrower support is REPORTED-ONLY.
(ii) **Weighting-scheme sensitivity (new)** — recompute under A vs B on the **same** full node set; a
verdict that flips under weighting alone is numerically fragile and **must be reported, not
dropped**. The two legs are not redundant: (i) tests the support, (ii) tests the measure on a fixed
support.

**Sub-choices the author owns even though the document's own conventions nearly force them:** node
weighting (equal-count vs quadrature); whether the stated support is a **hard truncation** or a
**merely-affordable window** — this decides whether a passing verdict may be CALIBRATED or must stay
REPORTED-ONLY; and the coupled h-prior/support choice (PA-HIER-14's `H_GRID_41`-vs-`H_GRID_FULL`
disagreement, shared with PA-HIER-25).

**Coupling, restated:** PA-HIER-9's b-anchor decision **is** option B's stated support. The two must
be answered together, and both sensitivity legs re-run afterwards, before any width-inflation
verdict is called final.

---

### PA-HIER-25 — PA-HIER-7 (identifiability) RESOLVED as fact-finding. **PROFILED is the only statistic with a valid χ²₂ correspondence**; PA-HIER-7's pin is confirmed correct and the reason is now on the record.
**Date:** 2026-08-27 · **Instrument run:** none · **Severity:** BLOCKER (fact half RESOLVED) · **Status:** RESOLVED; one residual author line (h support)
**Supersedes:** nothing — this supplies the derivation PA-HIER-7's correction asserted without.

**Derived.** The registered anchors `χ²₂(0.95)/2 = 2.9957` and `χ²₂(0.6827)/2 = 1.1479` are **Wilks
profile-likelihood-ratio** anchors, valid for `k = 2` **tested** parameters (b, s) with **any**
number of nuisance parameters (here h) profiled out — the nuisance count never enters the degrees of
freedom. PA-HIER-7's pin (`Δ ln L = max_h Σᵢ ln Lᵢ(h, truth-θ) − max_h Σᵢ ln Lᵢ(h, corner)`) is
therefore **correct**, and the two alternatives fail for stateable reasons:

- **FIXED-h** (h held at 0.73 on both sides) has **no valid χ²₂ correspondence** when a genuine h–b
  ridge exists: it crosses the ridge's full transverse curvature without letting h compensate. It can
  clear the 3.00-nat IDENTIFIABLE bar while the operationally relevant profiled number sits below
  1.15 (UNIDENTIFIABLE) **on the same cubes**. It **systematically overstates identifiability** — and
  the direction of that error is toward authorizing Stage F's 424.4 CPU-h.
- **MARGINALIZED** (∫dh with a prior) has **no general Wilks-type asymptotic guarantee at all**.
  Bayesian evidence ratios and profile-likelihood ratios coincide only under regularity + flat-prior
  conditions that a 41-node, potentially boundary-railing h-posterior cannot be assumed to satisfy —
  this repository's own documented H₀-railing pathology is exactly that non-regular behaviour.
  Applying the χ²₂ bands to it is **unfalsifiable as posed**. It stays **REPORTED-ONLY regardless of
  its numeric value, never band-bearing.**

**Why the three differ along a ridge** (registered so the reported spread is interpretable, not
alarming): PROFILED re-maximizes h at each θ node — it climbs back up the ridge — leaving only the
ridge-**transverse** residual; FIXED-h skips that re-optimization and so measures a strictly larger,
inflated quantity; MARGINALIZED integrates rather than maximizes, generically diluting the contrast
further and making the number prior-dependent.

**Registered to report alongside the pinned PROFILED number:** the fixed-h and marginalized variants
(REPORTED-ONLY); `ĥ(θ)` at **every** grid node — the ridge itself, a free byproduct of profiling;
`ρ(h, b)` and `ρ(h, ln s)` (§4.2's own anticipated secondary read); and PA-HIER-7's registered
precondition `lnL(truth-θ) ≥ lnL(θ)` at every other Stage-P node — if it fails, the "truth ≈ MLE"
premise the Wilks anchor requires does not hold on a 3×3 / 5×5 grid and the read downgrades to
REPORTED-ONLY.

**Residual for the author:** only the h support for the profile — `H_GRID_41` (§5.1 invariant 2)
versus `H_GRID_FULL` (PA-HIER-14's disagreement at `correspondence_1d.py:3788-3789`). One line,
shared with PA-HIER-24's sub-choice (iii).

---

### PA-HIER-26 — LOW. PA-HIER-17's window-truncation table is computed for a **Gaussian** truth and is exactly right for **neither** registered mode; and its constant citation is off by four lines.
**Date:** 2026-08-27 · **Instrument run:** none · **Severity:** LOW · **Status:** FIXED (corrected numbers registered)
**Supersedes:** PA-HIER-17's table and its `_B0I_KERNEL_SIGMA_MULTIPLIER` line citation.

**Found.** Under `host_mode="catalogue"` the truth is a **delta at z_g**, which is the estimator
window's own centre — the fraction outside is identically **0 at every s**, so GATE WINDOW is
**vacuous** there (the s-misspecification is total by construction instead, PA-HIER-1). Under the
recommended `catalogue_selected` venue the truth is drawn **on the estimator's own ±4σ window**
(`_host_kernel_window`, `_B0I_KERNEL_SIGMA_MULTIPLIER = 4.0` at `correspondence_1d.py:1161` — **not
`:1157`**, which is the `roots_legendre` line; draw at `:1490-1498`), so the truth is a
**±4σ-truncated**, `w_pop·f_k·S̄_φ`-tilted Gaussian, not a Gaussian.

| node | estimator window | PA-HIER-17 (untruncated Gaussian) | corrected, ±4σ-truncated, pre-tilt |
|---|---|---|---|
| s = 1/√2 (a `score_lns` node) | ±2.828σ | 0.4678 % | **0.4615 %** |
| s = 0.50 (Stage-P/F corner) | ±2.000σ | 4.5500 % | **4.5440 %** |

PA-HIER-17's two figures are **re-verified as exactly correct for their stated Gaussian
assumption**; the corrections are second-order. The `w_pop·f_k·S̄_φ` tilt is not analytic and must be
measured — which GATE WINDOW already does empirically.

**Conclusion unchanged and reinforced.** s = 0.50 still exceeds GATE WINDOW's 1 % bar by ≈ 4.5×, so
the Stage-P/F `s = 0.50` corner nodes are **expected to arrive REPORTED-ONLY**, and §4.2's `Δ ln L`
must be defined on a corner that survives GATE WINDOW or the identifiability read has no
band-bearing corner at all. **Registered:** GATE WINDOW's measured counts are the operative numbers;
PA-HIER-17's table is a Gaussian-truth approximation and is **not** a band anchor.

---

## LAUNCH GATE, RE-STATED `[OPUS-ORCH 2026-08-27]`

*(Supersedes the LAUNCH VERDICT block above — its verdict is unchanged, its "cheapest path"
inventory is not. Zero compute spent; no instrument has run.)*

### (i) Blockers now RESOLVED — and how

| was | now | how |
|---|---|---|
| **PA-HIER-1** (generator law unregistered; truth-θ ≠ (0,1)) | **RESOLVED as fact** (PA-HIER-19) — one author [RULE] remains | Five `host_mode` laws enumerated at source and truth-θ derived per mode. Exactly two admit (0, 1): `catalogue_selected` (b0i) and `catalogue_selected_2d` (b0i2d). `catalogue` gives s → 0; both `population*` modes make the axis **inapplicable** (their hosts carry no `(z_g, z_error)` pair at all). Fix is a **one-line `host_mode` change**. |
| **PA-HIER-1's kernel-identity audit** (an unaudited load-bearing invariant) | **RESOLVED, 5 legs of 7** (PA-HIER-20) | Gaussian loc/scale identical at (0,1); `w_pop·f_k` and `S̄_φ` are the **estimator's own** live terms under PRODUCTION_FLAGS; both sides call the same imported production functions; the `S̃_φ` quadrature is GL-50 on both sides — explicitly **not** a PA-2D-2/-3 mismatch. Two legs remain uncertified (below). |
| **PA-HIER-3** (S0-R is a null instrument) | **CONFIRMED and upgraded to NEEDS-CODE** (PA-HIER-22) | The harness's own docstring states the property: `sigma_kernel == sigma_realized` identically. One shared handler serves generator and estimator. No knob exists. **Additionally:** PA-HIER-3's own proposed fix carries a candidate-list confound. |
| **PA-HIER-7** (identifiability statistic's h-treatment) | **RESOLVED** (PA-HIER-25) — one residual line on h support | PROFILED is the only variant with a valid χ²₂ correspondence (nuisance count never enters the d.o.f.); FIXED-h systematically **overstates** identifiability toward authorizing Stage F; MARGINALIZED has no Wilks guarantee and is REPORTED-ONLY regardless of value. |
| **PA-HIER-16** (GATE ENG cannot isolate site 2.3) | **RESOLVED as registration** (PA-HIER-23) | Path-isolated OAT matrix on a per-term `ln L` decomposition; site 2.3 judged on **uniformity + independent recomputation**, not an event-count fraction. |
| **PA-HIER-17** (window truncation) | **numbers corrected** (PA-HIER-26) | Re-derived on the actual truncated truth law; citation `:1157` → `:1161`. Conclusion unchanged: the s = 0.50 corner is expected REPORTED-ONLY. |

**Already FIXED by the first review and untouched here:** PA-HIER-4 (`score_lns`), PA-HIER-2
(CoR-M re-pin + GATE GEN-FROZEN), PA-HIER-8, -10, -11, -12, -13, -15, -18.

### (ii) Genuinely OPEN-FOR-AUTHOR — each answerable in one word

1. **[RULE] Venue.** Ratify the switch `host_mode="catalogue"` (arm b0) → `host_mode="catalogue_selected"` (arm b0i) as the [HIER] venue? *(No other mode gives truth-θ = (0,1).)* — **ratified / not**
2. **[DO] θ hook.** Authorize implementing θ = (b, s) as a new parameter threaded into `BayesianStatistics.evaluate()` at sites 2.1 / 2.2 / 2.3, **not** at `correspondence_1d.host_z_error_eff`? — **approved / not**
3. **[RULE] Physics-change scope.** Does the θ hook — an edit to `bayesian_statistics.py` (a trigger file) with a byte-identical default at (0,1), plus PA-HIER-10's unconditional `smear_sigma_z=True` arm pin — require the `/physics-change` protocol, or is §1.5's instrumentation guard sufficient? — **gate / no gate**
4. **[RULE] Certification bar.** Are PA-HIER-20's two uncertified legs — the `phi_survival_table` value-identity assertion, and the 401-node inverse-CDF grid convergence spot-check — **pre-launch gate items**, or **disclosed residual risk**? — **gate / disclose**
5. **[DO] Control.** Build the two-catalogue s-axis control (new code, own registration, plus a decision on the candidate-list confound), or take the free fallback — disarm D7's early exit and re-scope Stage 0 to **S0-A + S0-C only**? — **build / fallback**
6. **[RULE] θ prior.** Adopt option B (uniform in `b`, uniform in `ln s`, quadrature-weighted on the registered nodes, support pinned to the registered half-widths), or state a different prior explicitly? — **B / other**
7. **[RULE] b-grid anchor** *(PA-HIER-9, load-bearing for #6 — it **is** option B's support)*. Re-anchor `b`'s half-width on a measured catalogue statistic (free, one pandas read), or keep ±0.04 and restate its anchor as an arbitrary local-probe convention? — **re-anchor / keep**
8. **[RULE] h support.** Pin the profile's h grid (and the §4.4 rank/KS null) to `H_GRID_41` for both cited functions, per §5.1 invariant 2? — **H_GRID_41 / other**
9. **[RULE] Support semantics** *(§4.4/§4.5 currency)*. Is option B's stated support a **hard truncation** (a passing verdict may be CALIBRATED) or a **merely-affordable window** (REPORTED-ONLY)? — **hard / affordable**

Items 1, 4, 6, 7, 8, 9 are [RULE]s on evidence now fully in front of the author. Items 2 and 5 are
[DO]s. Item 3 is a scope ruling this pass deliberately declines to make on the author's behalf.

### (iii) NEEDS-CODE — the instruments that do not exist

| # | Instrument | Why it is needed | Scope note |
|---|---|---|---|
| C1 | **The θ hook itself** — a `(b, s)` parameter threaded into `evaluate()`, reparametrizing `host_z → host_z + b(1+host_z)` and `host_z_error_eff → s·host_z_error_eff` at sites 2.1 / 2.2 / 2.3, with a literal early-return at (0,1) | `grep -n "theta_b\|theta_s\|--theta" darksiren_emri/arguments.py` → **zero hits**. **Nothing** in the repo implements θ. Without C1, **no [HIER] stage can run at all**, including S0-A. | Lands **inside `bayesian_statistics.py`**, a `/physics-change` trigger file → re-opens the scope question (author item 3). Ships with: the production-default `False` regression test, the PA-HIER-11 twin-parity assertion, and the `:1173` docstring fix. |
| C2 | **Per-term `ln L` diagnostics + a per-site toggle switch** (PA-HIER-23) | The GATE ENG toggle matrix is undecidable on aggregate `ln L`; the site-2.3 test needs a separable denominator term. | Instrumentation-only; same commit as C1. Adjacent to the same trigger file. |
| C3 | **The two-catalogue s-axis positive control** (PA-HIER-22) — an estimator-facing catalogue with a rewritten quoted-width column, **plus** driver plumbing for two decoupled handlers in one mirror-seed run, **plus** a resolution of the candidate-list confound | S0-R injects **nothing**; today's harness cannot produce an s-mismatch by any parameter setting. Without C3, D7's early exit stays unarmed and **no LEVER-DEAD-AT-N verdict may bank**. | Option (iii) of the fix — freezing the candidate list from the generator-facing column — would touch **`galaxy_catalogue/handler.py`'s search path**, i.e. a third new instrument adjacent to production selection code. Needs its own pre-registration. |
| C4 | **The 401-node inverse-CDF convergence spot-check** (PA-HIER-20 leg b) | The b0i z-draw grid is a *different* numerical operation from the certified GL-50 normalization, un-audited in the wide-window / near-horizon regime — the PA-2D-2/-3 borrowed-quadrature failure shape one axis over. | Small compute (401 vs 4001 nodes, plus a rejection-sampling cross-check), zero production risk. Gate-vs-disclose is author item 4. |
| C5 | **The `phi_survival_table` value-identity assertion** (PA-HIER-20 leg a) | Two independently constructed tables are assumed equal at runtime with no assertion anywhere. | One assertion; trivial. Gate-vs-disclose is author item 4. |

**Standing note on scope:** C1, C2 and C3(iii) all place new code in or immediately adjacent to
files on CLAUDE.md's physics-change trigger list. The REVIEW SUMMARY's byte-identical-default
argument is retained and is defensible, but it is the **author's ruling**, and it must be made once,
explicitly, before the first line of C1 is written — not inferred after the fact from a passing
GATE T-ID.

### (iv) LAUNCH VERDICT — **LAUNCH-BLOCKED (unchanged), and the reason has changed shape**

No `sbatch` for any [HIER] stage and no Stage-0 CPU-h.

The first review left [HIER] blocked on **six unresolved questions**. This pass answers the
factual half of five of them and, in doing so, replaces the diagnosis. The honest statement of
where the thread stands:

1. **A venue with truth-θ = (0, 1) does exist** — `host_mode="catalogue_selected"` (arm b0i) — and
   the generator/estimator kernel identity on it is now **audited and certified on five of seven
   legs at source**. The self-consistency arm S0-A is therefore a real, well-posed measurement after
   a one-line venue change. That is the good news, and it is genuine: PA-HIER-1's worst reading —
   that no mode admits truth-θ = (0,1) — is **refuted**.

2. **But no existing configuration anywhere in this repository injects a z-kernel misspecification.**
   At every `sigma_scale`, the generating width and the estimator's quoted width are **the identical
   number**, by the harness's own deliberate design, and one catalogue handler serves both sides. The
   `s` axis has **no positive control** today, and the obvious way to build one perturbs the
   candidate list as well as the width, so it would not even be measuring the same perturbation as
   the lever it certifies. **The `s` axis cannot currently be shown to be alive at all.**

3. **And the instrument itself does not exist.** θ has **zero occurrences** in the codebase. Every
   [HIER] number — S0-A included — requires new code inside a physics-trigger file before a single
   CPU-h can be spent. The first review's *"cheapest path back to LAUNCH-READY (all zero-compute)"*
   is accurate about registration and silent about this; the two are not the same thing, and this
   amendment exists partly to stop that elision from carrying forward.

**The unwelcome finding, stated plainly.** The (h, θ) experiment **as designed is not runnable on
the current mirror venue without new generator-side and estimator-side code.** The venue problem is
a one-line fix; the instrument problem (C1/C2) is a bounded, well-specified build; but the **control
problem (C3) is a new instrument with a confound of its own**, and until it exists the thread can
demonstrate that θ is *wired* (GATE ENG) and that the venue is *self-consistent* (S0-A) without ever
demonstrating that the `s` axis carries information about a real misspecification. A thread that can
only measure its own self-consistency is not yet testing the hypothesis it was opened to test.

**What that implies, without softening.** The author's item-5 choice is not a convenience call. Taking
the **fallback** (S0-A + S0-C only, D7 disarmed) yields a defensible but **strictly weaker** thread:
it can certify wiring, measure identifiability, and report the h–θ ridge, but it can **never** bank a
LEVER-DEAD-AT-N verdict, and any null it produces is confounded with "the axis was never shown to be
live". Taking **build** costs a new instrument, its own pre-registration, and a candidate-list
decision — and only then does the registered question become answerable as posed. There is no third
option in which the existing harness answers it.

**Path to LAUNCH-READY.** Author items 1–9 (all one-liners, all zero-compute) → C5 + C4 (small) →
C1 + C2 as one commit under whatever scope item 3 rules → GATE T-ID, GATE ENG (PA-HIER-23 form),
GATE PARITY, GATE MASS-KERNEL → S0-A. C3 gates only the S0-R / D7 / LEVER-DEAD-AT-N branch and can
proceed in parallel or be dropped per item 5. **No Stage-P grant is re-opened by this pass**;
§7.2's ceilings stand unchanged.

---

**PA-HIER-27 (2026-08-28; AUTHOR RULING RECORD on §(ii)'s nine one-liners; append-only;
`[FABLE-ORCH]`)**

Author reply (verbatim, to the Runbook 36 Docket artifact which carried the nine one-liners as
a summarized card): **"all ratified also the thirteen earlier ones"**. Per the approval-scope
convention, a blanket ratification grants yes/no items and items with a clearly proposed
option, but cannot pick a side of a two-option fork that stated no proposal. Orchestrator-derived
itemization — **the fork assignments below are flagged for author veto**:

- **Item 1 (venue) RATIFIED** — `host_mode="catalogue"` → `"catalogue_selected"` (arm b0i) is
  the [HIER] venue. S0-A unblocks.
- **Item 2 (θ hook) APPROVED** — θ = (b, s) threaded into `BayesianStatistics.evaluate()` at
  sites 2.1/2.2/2.3. **Implementation HELD until item 3 resolves** (the hook edits a
  physics-trigger file; absent the scope ruling the conservative default is that the
  `/physics-change` hard gate applies).
- **Item 3 (physics-change scope) — NOT RESOLVED** (gate / no gate; the prereg itself declined
  to propose). One word required.
- **Item 4 (certification bar) — NOT RESOLVED** (gate / disclose; no proposed side). One word
  required.
- **Item 5 (control) — NOT RESOLVED** (build / fallback; no proposed side, and the two options
  differ by an entire instrument registration). One word required.
- **Item 6 (θ prior) RATIFIED as option B** (the stated proposal: uniform b, uniform ln s,
  quadrature-weighted, support pinned to registered half-widths).
- **Item 7 (b-grid anchor) RATIFIED as RE-ANCHOR** (the proposed measured-statistic option,
  "free, one pandas read") — orchestrator-interpreted; veto if "keep" was intended.
- **Item 8 (h support) RATIFIED as H_GRID_41** (the option stated "per §5.1 invariant 2").
- **Item 9 (support semantics) — NOT RESOLVED** (hard / affordable; no proposed side). One word
  required.

**Launch state after this record: still LAUNCH-BLOCKED** — items 3/4/5/9 remain open, and
C1–C3 remain unbuilt. What is now unblocked: S0-A (venue fixed), the item-7 pandas re-anchor
read, and drafting the θ-hook design against option B + H_GRID_41 (no trigger-file edit until
item 3).

---

**PA-HIER-28 (2026-08-28; the four fork items RESOLVED — author verbatim: "exactly as
recommended by you", against the Six Forks brief; `[FABLE-ORCH]`)**

- **Item 3 = GATE.** The θ-hook edit to `bayesian_statistics.py` takes the FULL
  `/physics-change` protocol (presentation gate before code, byte-identity regression test at
  θ = (0,1), ledger row). §1.5's instrumentation guard is superseded for this edit. Item 2's
  approval is hereby UN-HELD: implementation may proceed *through* the gate.
- **Item 4 = GATE.** The two uncertified legs — the `phi_survival_table` value-identity
  assertion and the 401-node inverse-CDF grid convergence spot-check — are PRE-LAUNCH GATE
  items. Certification execution ordered 2026-08-28.
- **Item 5 = FALLBACK.** D7's early exit is DISARMED; Stage 0 re-scopes to **S0-A + S0-C
  only**. This is sequencing, not waiver: any stage-F launch still requires a positive control
  under the standing LAUNCH GATE, and a "build" proposal returns as its own registration if
  Stage 0 keeps the thread alive.
- **Item 9 = AFFORDABLE.** Option B's support is a merely-affordable window: **all [HIER]
  verdicts are capped REPORTED-ONLY.** Upgrade to hard-truncation/CALIBRATED requires a
  registered justification AND a positive control (coherence rule from the brief:
  fallback + hard is forbidden).
- Veto window closed without veto: item 7 stands as RE-ANCHOR; R-MKER-3 stands in R2 form.

**Launch state:** still LAUNCH-BLOCKED pending C1–C3 (θ hook now via /physics-change; control
deferred per item 5) and the item-4 certifications. Unblocked and ordered now: the item-7
b-grid re-anchor read, the item-4 certifications, S0-A.

---

**PA-HIER-29 (2026-08-28; item-7 RE-ANCHOR EXECUTED — the measured b half-width;
`[FABLE-ORCH]`)**

PA-HIER-9 option (a) executed against the pinned catalogue (md5 `c52c13b5…` verified before
reading; 22,641,048 rows, chunked pandas; agent-run, decisive statistic independently
spot-checked by the orchestrator on a 4M-row slice: median 0.033054 vs full 0.033038 ✓).

- **Measured statistic:** median `REDSHIFT_MEASUREMENT_ERROR/(1+REDSHIFT)` = **0.033038**
  (mean 0.032018; 6,284 NaN rows excluded).
- **Re-anchored half-width: b_max ≈ 0.0661** (2×median, PA-HIER-9's stated convention) —
  ~1.65× the superseded ±0.04.
- **Spec ambiguity DISCLOSED, resolved by the spec's own words:** PA-HIER-9 names both "the
  mirror host pool" and "free, zero-compute, one pandas read" — the pruned-pool reading is NOT
  zero-compute (the on-disk `STELLAR_MASS` is raw stellar mass in 10¹⁰ M_sun units; reaching
  BH-mass thresholds requires `_empiric_stellar_mass_to_BH_mass_relation`). Executed readings:
  (A) all valid rows and (B) z-only prune (z−z_err ≤ 1.5) — identical to 4 decimal places
  (median 0.033038 both), so the ambiguity is low-consequence. The full-pool reading (C) is
  registered as NOT-EXECUTABLE-AS-SPECIFIED.
- Robustness: driven by REDSHIFT_FLAG=1 photometric rows (98.8%, median 0.03304); the flag-3
  minority (1.2%, median 0.00239) cannot move the median — consistent with F4.
- **NOT the 0.163–0.392 figure** from PA-HIER-9's internal-contradiction paragraph (that band
  used the campaign's injected σ_z/z, not the catalogue's on-disk column). The b grid for
  option B (item 6) re-derives from **±0.0661** when the θ instrument is built.

---

**PA-HIER-30 (2026-08-28; item-4 CERTIFICATIONS EXECUTED — both legs CERTIFIED; `[FABLE-ORCH]`)**

- **Leg (a) `phi_survival_table` value-identity: PASS.** Generator-path
  (`build_bsel_selection_objects` → `precompute_phi_marginal_survival`,
  `correspondence_1d.py:983/:1049`) and estimator-path (as `evaluate()` builds it,
  `bayesian_statistics.py:3931-3949/:4050-4054`) tables built independently at h = 0.73 on the
  REAL injection pool + completeness cache with all production kwargs matched: `z_grid` and
  `s_phi` (shape (1500,)) **bit-identical** (max abs diff 0.0). Scope: certifies the
  construction identity; the C5 runtime guard remains unbuilt (disclosed).
- **Leg (b) 401-node inverse-CDF convergence: PASS with a disclosed residual.** 401 vs 4001 vs
  40001 nodes + independent rejection-sampling cross-check on real catalogue hosts:
  non-straddling hosts converge ≤0.1%; worst-case mean bias ≈ 2e-4 in z (~0.2% of σ_eff)
  everywhere. **Residual: hosts whose ±4σ window straddles the S̄_φ table's z_max = 1.5 edge
  show ~13–15% relative std inflation at 401 nodes** (rejection sampling confirms the finer
  grids, not the 401 grid, are correct). Affected class: a handful of the 386/20.8M rows with
  z > 1.0 — order 1e-5 of the catalogue.
- **CHAIR GATE (the prereg named no tolerance): CERTIFIED.** Rationale: the mean bias is three
  orders below any registered read's resolution; the width residual is confined to an ~1e-5
  host fraction, and every [HIER] verdict is capped REPORTED-ONLY (item 9 = AFFORDABLE). The
  residual is REGISTERED, not waived: if the θ instrument build touches
  `correspondence_1d.py` anyway (it will, via /physics-change per item 3), raising
  `_B0I_ZTRUE_GRID_N` 401 → 4001 is a free hardening to include in that same gated change.

---

### PA-HIER-31 (2026-08-29; S0-B registration addendum after Stage-0 wave-1; orchestrator decisions of record; `[FABLE-ORCH]`)

**Launched under rows #222/#223 — charter node B1.2.** Source: `WAVE2_REGISTRATION_CHECK_20260829.md`
§2 skeleton (chair, wave-2 PREP) + `SYNTHESIS_DOCKET_1_20260829.md` §2 B1 P2 items (a)–(e) +
orchestrator path decisions of record, 2026-08-29 ("B1 → S0-B (C1) proceeds AFTER PA-HIER-31 +
θ CLI plumbing (P6) + S0-A completion (P0), in the CoR-P-faithful form `theta_sites="2.2"` +
`smear_global_selection=False`"). Nothing above this divider is edited.

**(a) b-node re-derivation and the two-arm pairing rule.**
S0-B's four θ-nodes use the re-derived half-width **±0.033** from `b_max = 0.0661` (PA-HIER-29:
*"Re-anchored half-width: b_max ≈ 0.0661 (2×median, PA-HIER-9's stated convention) — ~1.65× the
superseded ±0.04,"* itself from *"Measured statistic: median `REDSHIFT_MEASUREMENT_ERROR/(1+REDSHIFT)`
= 0.033038"* — a 5-node grid over ±0.0661 gives a half-step of 0.033). The S0-A remainder (the
b_minus, s_minus, s_plus nodes not yet run at check time) keeps the **as-built ±0.02** with a
disclosed "as-built" label — docket P2(a): *"the chair recommends the re-derived node for S0-B
and a disclosed 'as-built' label for the S0-A remainder (paired within arm, so mixing is not
allowed)."* The two arms (±0.02 as-built vs ±0.033 re-derived) are never combined into one
secant, one Z, or one materiality read; each is reported against its own registered grid.

**(b) CoR-P-faithful θ form for every no-BH read.**
`theta_sites = "2.2"` (the batched per-host host-z kernel, the production dispatch path) and
`smear_global_selection = False` for S0-B and for the S0-A remainder (P0) alike. **Site 2.3 is
OUT OF SCOPE**, reason F-A (`WAVE2_REGISTRATION_CHECK_20260829.md` §0): *"the ternary at
`bayesian_statistics.py:5187-5191` does pick the θ-inert `_global_cat_selection_phi` for
`global_denom_no_bh`, **but** the path-(A) assembly `path_a_mixture_objects` (`:2440-2500`)
takes `sigma_4d = _global_cat_denom_with_bh[h]` (`:4160-4171` under `smear_sigma_z=smear_global_selection`),
and the no-BH per-event likelihood is `(β_G^φ·L_cat + B_num^φ)/D̃^φ` (`:5770`). So under
`"all"`+smeared, site 2.3 reaches the no-BH channel through Σ^4D → r_Malm → α_G^φ → D̃^φ ... it is
absent from CoR-P (`smear_global_selection=False`, `headreadout_20260827/iiib/run_metadata_21.json:cli_args`)."*
Measured consequence (F-A, seed 900101, 9 shared events, b = +0.02, h = 0.73): `L_cat_no_bh`
bit-identical between the two forms (max_rel 0.0) but `combined_no_bh` diverges (max_rel
**7.45e-3**), driven by `alpha_G_phi` (5.8688310e7 → 5.1635200e7, −12.0%), `D_tilde_phi`
(9.470921e8 → 9.40039e8, −0.745%), `w_G` (0.06196684 → 0.05492879). Therefore: **the already-run
S0-A `"all"`+smeared b_plus node is DIAGNOSTIC / REPORTED-ONLY / non-CoR-P**, and P0 re-runs the
truth node's remaining grid points (and re-scores the truth/b_plus comparison) in the
2.2/unsmeared form for comparability. Registered instrument identity check: for every C-C event
(`L_cat_no_bh == 0` at h = 0.73), `combined_no_bh` must be bit-identical across all five θ-nodes
regardless of form (θ has no referent there); any deviation is INSTRUMENT-DEFECT.
Optional, non-blocking: **P1′** (one (0,1)-smeared node, ≈0.33 CPU-h) to attribute the −12.0%
α_G^φ shift to θ itself vs. the smear switch — informational only.

**(c) `score_s` → `score_lns` relabel.**
The driver's linear secant (`hier_s0_driver.py:242-245`) computes what this thread calls
`score_s`; the object registered by this prereg (§4.1) as `score_s` is the **ln-s form**, now
relabelled **`score_lns`** to remove the ambiguity (docket P2(c): *"relabel `score_s` (linear
secant) vs the registered `score_lns` (Z identical; magnitudes not comparable to ln-s bands)"*).
For S0-B: `score_lns,i = [lnL_i(0,√2) − lnL_i(0,1/√2)]/ln 2`. `Z_lns` is numerically identical
under either labelling (Z is scale-invariant); **magnitudes are not** — any point number quoted
in ln-s units is not directly comparable to a linear-secant magnitude quoted elsewhere in the
thread (e.g. the mirror KW-Q1 card's `s_imp` figures, §3 item 9 of the wave-2 check, are on a
different statistic entirely and are not affected by this relabel).

**(d) S0-B design — nodes, statistics, reads.**
Four θ-nodes at h = 0.730 on venue **iiib** (production, CoR-P per §1 below), plus the truth
node (θ = (0,1)) which doubles as **C0**, the shared baseline gate task:

```
truth        (0, 1)              = C0
b_plus_re    (+0.033, 1)
b_minus_re   (−0.033, 1)
s_plus       (0, √2)
s_minus      (0, 1/√2)
```

Per-event statistic (both channels computed, primary = no-BH; `ln` of `combined_*` per
`hier_s0_driver.py:242-245`):

```
score_b,i   = [ lnL_i(+0.033,1) − lnL_i(−0.033,1) ] / 0.066
score_lns,i = [ lnL_i(0,√2)   − lnL_i(0,1/√2) ] / ln 2
Z_x = mean(score_x) / SEM(score_x)
```

Read **pooled** (N = 1588); **by class** — C-A: `in_catalog = True` & `L_cat_no_bh > 0` (n ≤ 76,
`b3_pop_prediction.json:venues.iiib.n_matched`, B3.2 §F item 3); C-B: `in_catalog = False` &
`L_cat_no_bh > 0`; C-C: `L_cat_no_bh == 0` (n = 606) — class definitions per
`B3_1_POP_RECORD.md` ("Class split" paragraph, its Method section, quoted verbatim: *"'dark' =
`L_cat_no_bh == 0` at every one of the 41 h nodes (class C-C, `PREREG_COMPLETION_CLASS_DECOMPOSITION.md`);
'matched' = the complement (≥1 node with `L_cat_no_bh > 0`, i.e. C-A ∪ C-B combined — this
conflates true in-catalogue hosts with impostor-only catalogue support; row #141 found C-A alone
pulls the *opposite* sign, so the 'matched' number here is a coarser read than that finding and
is reported only for context, not compared against a per-class prediction)"*); C-A ∪ C-B = 982
(0.6184 × 1588, same source) — class is defined **at h = 0.73, this arm's single node**,
disclosed as differing from B3.1's all-41-node definition (wave-2 check §1.4 "class-definition
note"). By **z-bin**, B3.1's registered edges {0.075, 0.392, 0.559, 0.659, 0.753, 1.018}
(`b3_pop_prediction.json:registered_bin_edges`), using CRB `z_true = dist_to_redshift(d_L, 0.73)`.

**Predictions registered before the run (F3):**
- **B1's own** [HIER]: no point prediction — the hypothesis is the LIVE/DEAD fork itself. Sign
  expectation, REPORTED-ONLY: `score_lns > 0` pooled — the likelihood is expected to prefer a
  wider kernel if the quoted photo-z errors understate realised scatter, landing in exactly the
  blindness class this design's own §5.2 item 2 names: *"Any misspecification outside that
  2-dimensional span — an outlier fraction, a heavy tail, a z-**dependent** scatter, a skew, a
  catastrophic-failure mode — is invisible **by construction**... and it lands exactly where the
  Stage-L sweep found the literature lives (the minority-outlier regime, row #193)."* `score_b`:
  **no sign registered.** Both computed from the §4.1 `Z_x = mean(score_x)/SEM(score_x)`
  machinery, applied here to the production venue as the deferred **B0-B** read named in §4.1's
  anchor note (*"Anchor derivation for B0-B (the deferred production read): identical bands,
  applied to S0-B"*).
- **B4's (L2, `CLAIM_IMPOSTOR_DRAG_20260829.md` §1.3/§1.4):** the KW-Q1 read's own falsifier
  (§1.3) is the localisation test this arm's z-bin 1 must reproduce: *"the KW-Q1 diagnostics CSVs
  also give S over q2–q4; if q1's share of Σ s_imp at s = 1 on the HEAD basis falls below 50%,
  C2's localisation is withdrawn regardless of R."* Transposed to S0-B: on C-A ∪ C-B, the share
  of `Σ|score_lns,i|` carried by z-bin 1 (0.075–0.392, ≈ the mirror q1 edge z_true < 0.358) is
  predicted **≥ 0.50** (mirror analogue: q1 carries 91.7% (ft) / 86.2% (fc) of the impostor-leg
  h-score, §1.3; q1 mean `s_imp` = −0.798 ± 0.041 (ft), `b4_imp_stage1_forecast.json:covariates.ft.z_true`).
  **< 0.50 ⇒ C2's localisation does not transfer to production and is WITHDRAWN there**
  regardless of Z. If a mechanism read is wanted: KW-Q1 **OWNS** (`§1.3`: *"|R| ≥ 0.5 ... ⇒ the
  remainder is a kernel-width-class object ⇒ B4 MERGES INTO B1 (charter 4.3)"*) predicts
  `|Z_lns| > 3` on C-A ∪ C-B concentrated in bin 1; KW-Q1 **INERT** (*"|R| ≤ 0.2 ⇒ not a width
  object"*) predicts `|Z_lns| ≤ 3` in bin 1 — REPORTED alongside, not a band (KW-Q1's `R` is an
  h-score response to `s`, not `∂lnL/∂s`, §1.4).
- **B3's (L1, reduced by §F):** (i) C-C class θ-score ≡ 0 by identity (the population prior has
  no θ referent); (ii) the C0 truth node reproduces B3.1's dark-class h-score profile
  (+0.081/−0.332/−0.562/−0.701/−0.855; bins 2–5 −0.612, n = 484,
  `b3_pop_prediction.json:venues.iiib.bins`) to ≤ 1e-6 — the baseline pin, doubling as C0's F3
  secondary prediction (wave-2 check §1.1); (iii) no population-term prediction on the θ-score
  exists — the data carry no such term after C2 was struck (§F of
  `PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md`).

**(e) Bands (A8, two-sided, referents = the four θ-nodes vs C0).**
**B0-B** ≡ `|Z_b| ≤ 3` **and** `|Z_lns| ≤ 3` pooled (two-sided) ⇒ **LEVER-DEAD-AT-N
(production)**; either `> 3` ⇒ **LEVER-LIVE** — then B0-M (materiality): MIXED if `|b̂| < 0.0165`
(half the 0.033 step) or `|ln ŝ| < 0.5·ln√2 = 0.173`; B0-P (power): `σ_b < 0.0661` and
`σ_ln s < ln 2`, else UNPOWERED (no DEAD claim). Curvature leg: quadratic fit through the three
b-nodes (truth, b_plus_re, b_minus_re) → `b̂ = −S′/S″`, `σ_b = 1/√(−S″)`; likewise in ln s. Per-
class and per-z-bin: same `|Z| ≤ 3` bands, REPORTED (not gating); the C-C identity check (item d
above) is a pass/fail instrument check, not a band. **All verdicts carry the REPORTED-ONLY cap**
(PA-HIER-28 item 9: *"Option B's support is a merely-affordable window: all [HIER] verdicts are
capped REPORTED-ONLY. Upgrade to hard-truncation/CALIBRATED requires a registered justification
AND a positive control (coherence rule from the brief: fallback + hard is forbidden)."*). Fork
mapping (docket §2 B1): DEAD ⇒ 1.3b (park, redirect); LIVE ⇒ 1.3a (Stage P re-costed under L6).

**A15 at N = 1588.** Null: `Z ~ N(0,1)` ⇒ `|Z| ≤ 3` false-fail **0.27%** two-sided; 80% power at
`mean = 3.84·SEM`. Per-event score_b SD proxy, independently re-derived this node from the banked
seed-900101 node CSVs (`hier_s0_registered_run/s0a_seed900101/{node_truth,node_b_plus}/simulations/diagnostics/event_likelihoods.csv`,
merged on `event_idx`, restricted to active rows `L_cat_no_bh>0` at truth or b_plus):
half-secant `score_b,i = [lnL_i(combined_no_bh, b=+0.02) − lnL_i(combined_no_bh, b=0)]/0.02` —
**one-sided** here (`node_b_minus` at this pin has `cramer_rao_bounds.csv` only, no
`event_likelihoods.csv` — S0-A remainder not yet run at check time) — gives, "all"/smeared form,
n = 105 active of 106: **mean −5.591, SD 16.879, SEM 1.647**; and (P1 smoke,
`hier_s0_work/b1_2_smoke/p1_2p2_off/s0a_seed900101/{node_truth_sites2.2_nosmear,node_b_plus_sites2.2_nosmear}/…`,
merged on `event_idx`) "2.2"/unsmeared form, n = 9 (all active): **mean −5.147, SD 13.972** — both
figures match the wave-2 chair's independently-quoted values (mean −5.59/−5.15, SD 16.88/13.97)
to reported precision. This is a **mirror→production transfer, disclosed** (production balls
carry ~10³× more candidates than the mirror). Scaling the two SDs to N_active = 982 (C-A ∪ C-B):
`SEM ≈ SD/√982` = 0.446–0.539 per unit b ⇒ **detectable |mean score| at 3σ (80% power, 3.84·SEM)
≈ 1.71–2.07 per unit b** (≈ 0.11–0.14 nats per event across the full ±0.033 secant). C-A alone
(n ≈ 76): `SEM ≈ SD/√76` = 1.602–1.936 ⇒ **≈ 6.15–7.44 per unit b** (weak, REPORTED only).
**s-component SD: UNMEASURED** — S0-A's s_plus/s_minus nodes were not run at check time; P0 must
fill it before any S0-B sbatch, else the s-bands are registered using the b-proxy SD with an
explicit flag that the transfer is doubly indirect. Controls capable of failing: C0 (T-ID); the
C-C identity check; GATE ENG on `L_cat_no_bh`.

**(f) GATE PARITY residual 5.718e-4 — disposition.**
Docket P2(e): *"accept as below-band with the batch-order hypothesis recorded, or diagnose (one
re-run of the banked bc CSV at the current commit decides it)."* **The batch-order hypothesis is
REFUTED** — chair finding F-B (`WAVE2_REGISTRATION_CHECK_20260829.md` §0): *"The 9-event P1 smoke
truth node and the 106-event registered truth node are **bit-identical on all 17 numeric
columns** over the 9 shared events ... Summation order does not depend on N here, so the
5.718e-4 residual of the driver vs the **banked** bc CSV ... is not a batch-size effect; the live
hypotheses are a code/config delta between that CSV's commit and HEAD, or a process/thread-count
effect in the banked run ... the docket P2(e)'s 'one re-run of the banked bc CSV at the current
commit' remains the deciding read."* **Disposition (this amendment): diagnose, not accept.** P0
includes one re-run of the banked `bc_900101_work` CSV at the wave-2 commit; if the residual
persists at the same magnitude the live hypothesis narrows to a process/thread-count effect
(both smoke and registered nodes ran at a 14-core pin); if it changes, a code/config delta
between commits is confirmed. Bearing on C0: this is exactly the kind of same-N, same-commit
reproduction gate C0 is built to catch (A15 control-capable-of-failing, satisfied by this
evidence, per F-B).

**(g) A10 invariants + blindness.**
Carries forward prereg §5.1 items 1–5, 7, 9–13 verbatim (items 6 and 8 are mirror-venue-specific
and do not bind a production arm) with the CoR-P list of §5.1 item 1 stamped concretely for this
arm: commit = **the wave-2 commit hash, placeholder pending §0 GAP 1's commit of the dirty tree**
(`WAVE2_REGISTRATION_CHECK_20260829.md` §5 item 1); CLI flags verbatim from
`headreadout_20260827/iiib/run_metadata_21.json:cli_args` (`absolute_marginal` / `volume_deconv`
/ `fused` / `phi` / `smear_global_selection=False` / `pdet_wbh_z_resolved=False` /
`eddington_m=on` / `sigma4d_mass_kernel=point` / `catalogue_numerator_survival_2d=off`) plus
explicit `--mass_filter_geometry linear --mass_filter_k 1.5`; reduced-catalogue md5
`c52c13b5…` pinned before reading (A11 dataset pin, CLAUDE.md 2026-08-20 rule); **H_GRID node
0.730 only** (`evaluate.sbatch --array=21`); driver identity **pinned at the wave-2 commit's
blob** for `hier_s0_driver.py` (superseding the stale `5313c319…` cite in L2 — current sha1
`9f831b9f7d6b8fed820d547bbe8cd64ff00873e3`, 567+/42− vs `dd63fe0c`, per
`WAVE2_REGISTRATION_CHECK_20260829.md` §5 item 10). New invariant this amendment adds:
`smear_global_selection = False` (2026-08-29, item (b) above); item 8's mirror↔production parity
becomes moot for S0-B itself (it *is* production) but the 5.718e-4 undiagnosed residual (item f)
stays a live open item, named by number, not waived.
**Blindness**, carried and extended: (a) anything acting only through a smeared global selection
(site 2.3, out of scope, item (b)); (b) the production venue has no truth-θ — a non-zero score
cannot by itself separate a photo-z kernel misspecification from any other misnormalisation
sharing the catalogue leg (the B4 impostor object; hence the registered L2 profile prediction,
item (d)); (c) θ's 2-D span, prereg §5.2 item 2 (quoted in item (d) above: *"θ = (b, s) spans
exactly linear-in-(1+z) bias and uniform multiplicative scale. Any misspecification outside that
2-dimensional span ... is invisible by construction"*); (d) single h (0.730 only); (e) the
with-BH channel is secondary and inherits invariant 12 (the open `[P3-MKER]` state).

**(h) A14 falsifiers.**
Carried from prereg §6, **LEVER-DEAD-AT-N** row, quoted verbatim: *"S0-B (production venue,
CoR-P) returns `|Z_b| > 3.0` or `|Z_s| > 3.0`. A live production score with a dead mirror score
refutes 'dead' and re-attributes it to the mirror venue's own self-consistency (§5.2 item 3), not
to the lever."* — §5.2 item 3 itself, quoted: *"On the mirror the generator kernel **is** the
estimator kernel at truth-θ, so any misspecification **shared** by generator and estimator
cancels and is undetectable — precisely the class the research-cycle's stop/continue rule
records that SBC/coverage cannot catch."* Additional falsifiers specific to this amendment: LIVE
attributed to "the host-z kernel" is FALSIFIED if the C-C identity check (item d) fails
(instrument defect) or if the `score_lns` z-profile is flat within 3σ **and** the L2 q1-share
prediction (item d, B4's) fails (then it is a normalisation object, the B4.3 class, not a kernel
object). DEAD is provisional until B0-P passes **and** the S0-A remainder (P0) certifies the
instrument in the same 2.2/unsmeared form.

**(i) Cost (F4) and archive.**
**4 θ-nodes × 14.93–22.9 CPU-h = 60–92 CPU-h**, unsmeared form (`COMPUTE_LEDGER.md` row C1); **the
81–113 CPU-h smeared band is STRUCK** — it was priced on the now-refuted P1 equivalence and
described a non-CoR-P form (item (b) above; wave-2 check §3 item 6). P0 (S0-A remainder, same
2.2/unsmeared form) ≈ **5 CPU-h / 40 min wall** at 5 parallel nodes (docket §2 P0, re-scoped).
S0-C ceiling **≤ 15 CPU-h** (registered, marginal cost still unmeasured). Archive: **MUST-ARCHIVE
(Option A)**; out-root field `run_20260829_wave2_c1_iiib`; `COMPUTE_LEDGER.md` row C1 "archive-
scheduled: yes" required before sbatch (F4 gate). Deadline 2026-09-23 (workspace expiry, 0
extensions).

**(j) P6 precondition — θ on the production CLI.**
Blocking, non-physics: expose `--theta_b`, `--theta_s`, `--theta_sites` in `arguments.py` +
`main.py` → `BayesianStatistics.evaluate()` (defaults `0.0`, `1.0`, `"all"` byte-identical to
today; `run_metadata_*.json` records the passed values, which **is** this arm's A22 stamp) — a
plumbing commit only, the B5.1 pattern, non-physics files, ledger note — **or** a production
in-process driver, in which case the driver's own emitted JSON is the A22 stamp instead. GATE
T-ID at production scale **is C0** (θ = (0,1) reproduces the banked `d04d9dc9` columns ≤ 1e-12
relative, all 17 numeric columns). GATE ENG, scored on `L_cat_no_bh`: ≥ 99% of C-A ∪ C-B events
must move ≥ 1e-6 relative at each off-truth node (mirror precedent: 105/105 moved, §0 of the
wave-2 check). GATE TABLE-FRESH: one `BayesianStatistics` construction per node — the four
separate sbatch tasks structurally guarantee this (invariant 13).

**REPORTED-ONLY cap, carried.** Per PA-HIER-28 item 9 (quoted above, item (e)): every verdict
produced by this amendment — B0-B, B0-M, B0-P, per-class, per-bin, and the KW-Q1 coupling read —
is REPORTED-ONLY. No CALIBRATED or hard-truncation claim may be made from this arm without its
own registered justification and a positive control.

**Ordering (A22).** Wave-2 commit hash + dirty-state clean at run START (gap-list item 1);
`1f003da6` (B6.1, s-placement) precedes ✓ (L8, already landed); **P0** (S0-A remainder, 2.2/
unsmeared form) and the **GATE PARITY disposition** (item (f), with F-B's corrected hypothesis)
must both be recorded **before** S0-B banks; execution-completeness (A8(d)): no class/bin branch
is adjudicated until all four θ-nodes exist.

**Verifier scope.** This amendment; finding F-A (site-2.3 non-inertness through α_G^φ/D̃^φ);
finding F-B (batch-order REFUTED, code-delta hypothesis remains open); finding F-C (θ not on the
production CLI, P6 required); the b-node re-derivation (item a); the S0-A smeared-node
reclassification to REPORTED-ONLY/non-CoR-P (item b).

*Authorization: launched under rows #222/#223 — charter node B1.2. Append-only; nothing above
the PA-HIER-31 divider is edited. No git operations; no source edits; `hier_s0_driver.py` and
`kwq1_score.py` untouched (owned by another agent). Chair: inherit-tier subagent, scoped
package, 2026-08-29.*

---

### PA-HIER-31 REVISION NOTE 1 (2026-08-29; refuter-panel `must_fix` response; append-only; `[FABLE-ORCH]`)

**Launched under rows #222/#223 — charter node B1.2.** A refuter panel REFUTED the
`PA-HIER-31` block above with five `must_fix` items. Nothing above this divider (including
`PA-HIER-31` itself) is edited — corrections below are append-only supersession notices,
the same pattern `PA-HIER-10` already used against earlier §-text. Each item is checked
against the shipped code / cited sources before disposition; two are confirmed-and-fixed
outright, one is confirmed-and-partially-fixed with an open item returned for a fresh
author `[RULE]`, two are confirmed-and-fixed citation corrections. All five `must_fix`
findings are independently re-verified true (evidence below); none is disputed.

**R1 (must_fix 1). Supersedes: `PA-HIER-10`'s unconditional `smear_sigma_z=True` pin, CoR-P
clause only; `§2.4`'s CoR-P bullet ("...with `smear_global_selection` forced True per GATE
D3(a)").**
CONFIRMED, material, not cosmetic — re-verified: `PA-HIER-10`'s correction (line ~1047-1048
above) reads *"`smear_sigma_z` is pinned True for the entire arm, at every θ node including
truth-θ, at both CoR-M and CoR-P"* — an unconditional, HIGH-severity FIXED status covering
CoR-P by name. `PA-HIER-31(b)`/(g) sets `smear_global_selection=False` for the whole S0-B
arm and P0, calling it "the CoR-P-faithful form," with **zero** occurrences of "Supersedes,"
"PA-HIER-10," or "GATE D3" anywhere in the `PA-HIER-31` block (grepped, confirmed). The flag
is non-inert on the exact quantity S0-B's scores are built from: F-A (quoted in `PA-HIER-31(b)`
itself, `WAVE2_REGISTRATION_CHECK_20260829.md` §0) measures `alpha_G_phi` −12.0% and
`combined_no_bh` max_rel 7.45e-3 from this one flag flip, same seed/node.

Reconciliation: `PA-HIER-31(b)` is a genuine, *targeted* narrowing of `PA-HIER-10` restricted
to **CoR-P (the production venue) only** — `PA-HIER-10`'s pin is otherwise unchanged and still
binds **CoR-M (the mirror)**, where every prior/future mirror-venue read keeps
`smear_sigma_z=True` at every node including truth-θ. Authority for the CoR-P-only carve-out
is the 2026-08-29 orchestrator path decision of record, quoted verbatim in `PA-HIER-31`'s own
header ("in the CoR-P-faithful form `theta_sites=\"2.2\"` + `smear_global_selection=False` ...
reason: the chair's F-A finding"). That decision is real and on the record, but it is an
orchestration-level path call made in response to a newly-measured confound, not a re-derivation
that adjudicates `PA-HIER-10` vs. the new F-A evidence — and CLAUDE.md's approval-scope
convention is explicit that *"a branch call, verdict or band comparison that has not been
computed yet is never covered by a blanket 'all approved' ... it returns to the author as a
fresh [RULE]."* F-A's measurement did not exist when `PA-HIER-10` was ratified FIXED, so this
qualifies.

**Disposition, both must_fix options addressed, not just one:**
- **Option (i), adopted for the wave-2 execution window:** the F-A-measured −12.0%/7.45e-3
  shift is treated as an accepted, quantified confound — folded into item (g)'s blindness list
  as its own named sub-item (not merely "site 2.3, out of scope"): *"(a′) the CoR-P
  `smear_global_selection=False` choice itself is a narrowing of `PA-HIER-10`'s unconditional
  pin, carrying a measured −12.0% `alpha_G_phi` / 7.45e-3 `combined_no_bh` confound relative to
  the `PA-HIER-10`-pinned alternative (F-A) — disclosed, not resolved."* Item (e)'s REPORTED-ONLY
  cap already covers every verdict unconditionally, so no band escapes this without a positive
  control regardless.
- **Option (ii)'s resolving instrument already exists in the registered design:** `P1′`
  (item (b), "one (0,1)-smeared node, ≈0.33 CPU-h, informational only, non-blocking") is
  **upgraded here from purely informational to the designated resolving measurement** for this
  specific contradiction. Recommended (non-blocking per its original registration; does not gate
  S0-B execution) before any S0-B verdict is cited as more than
  REPORTED-ONLY-pending-adjudication.
- **What remains genuinely open, returned to the author:** *which value is authoritative for
  CoR-P going forward* is a scientific ruling narrowing a previously-FIXED HIGH-severity
  finding, not a mechanical documentation fix — this worker has no standing to make it (CLAUDE.md:
  "the author owns every scientific decision"; the row #222/#223 grant is a STANDING grant
  for wave-2 *execution*, not a `[RULE]` on this specific `PA-HIER-10`-vs-`PA-HIER-31(b)`
  contradiction, which did not exist as a named contradiction until this refuter pass).
  **Registered here as an OPEN CONTRADICTION pending a fresh author `[RULE]`** adjudicating
  `PA-HIER-10` vs. `PA-HIER-31(b)` for CoR-P. Until that `[RULE]` lands: P0/S0-B execution
  proceeds under the existing row #222/#223 STANDING grant (this note does not block running
  them — REPORTED-ONLY already caps every claim they can produce), but no `|Z_b|`/`|Z_lns|`
  verdict from this arm may be banked or cited as more than
  REPORTED-ONLY-pending-`[RULE]` until adjudicated.

**R2 (must_fix 2). Citation fix, item (c) — "the driver's linear secant."**
CONFIRMED. Verified against the pinned commit: `git show dd63fe0c:.../hier_s0_driver.py` shows
`:242-245` is inside `read_event_ln_l()`, computing
`sub[out] = np.where(vals > 0.0, np.log(vals), np.nan)` for `combined_no_bh`/`combined_with_bh`
— the `ln(combined_*)` transform, correctly cited (only) by item (d) for that purpose. The
actual secant arithmetic — `score_b = (b_join["b_plus"] - b_join["b_minus"]) / 0.04`,
`score_s = (s_join["s_plus"] - s_join["s_minus"]) / (sqrt2 - 1/sqrt2)`, matching this prereg's
own §4.1 formula verbatim — lives in `compute_scores()`, **`hier_s0_driver.py:394-449`**, same
commit. **Item (c)'s file:line pointer is corrected**: "the driver's linear secant"
→ `compute_scores()`, `hier_s0_driver.py:394-449`, commit `dd63fe0c`. Item (c)'s substantive
argument (Z is scale-invariant under the `score_s`/`score_lns` relabel; magnitudes are not) is
unaffected — only the pointer was wrong. Per the must_fix's instruction (the `PA-HIER-29`
catalogue-md5 pinning pattern), this citation is **commit-pinned to `dd63fe0c`** rather than
"current" — the working file is under known concurrent edit
(`WAVE2_REGISTRATION_CHECK_20260829.md` §5 item 10: +567/−42 lines, sha1
`9f831b9f7d6b8fed820d547bbe8cd64ff00873e3` vs `dd63fe0c`) and this node does not touch
`hier_s0_driver.py` (owned by another agent), so re-verifying the line numbers against the
eventual wave-2 commit is deferred to whoever banks that commit, not done here.

**R3 (must_fix 3). GATE D3(a) prose vs. shipped code — reconciled in favour of the code;
no source edit (physics-trigger file, out of this node's scope regardless).**
CONFIRMED. Verified: `precompute_global_catalog_selection`
(`bayesian_statistics.py:2799-2805`) —
```
if (theta_b != 0.0 or theta_s != 1.0) and not smear_sigma_z:
    raise ValueError(
        "theta (site 2.3) requires smear_sigma_z=True — the registered "
        "site is the smeared host-z kernel; got "
        f"(theta_b, theta_s) = ({theta_b}, {theta_s}) with smear_sigma_z=False"
    )
```
— a guard that **requires** the caller to pass `smear_global_selection=True` (raises
`ValueError` otherwise), not a **force**/auto-engage. `bayesian_statistics.py` is a
physics-trigger file (CLAUDE.md) and editing it is outside this node's authorized scope in any
case, so — following the same append-only supersession pattern `PA-HIER-10` used on earlier
§-text — **this note supersedes `§3.2` GATE D3(a)'s wording** ("`s ≠ 1` forces the
`smear_sigma_z=True` branch itself") **and `§2.4`'s CoR-P bullet's parallel wording**
("with `smear_global_selection` forced True per GATE D3(a)") with:

> `s ≠ 1` (equivalently, any non-identity θ reaching site 2.3) **REQUIRES**
> `smear_global_selection=True` — the call **raises `ValueError` if unmet**
> (`bayesian_statistics.py:2799-2805`), rather than auto-forcing the flag. The gate's
> underlying safety intent (no silent θ-on-numerator/θ-inert-on-denominator mixed arm) is
> satisfied by the raise exactly as it would be by a force; only the literal mechanism differs.

Confirmed moot for the registered S0-B/P0 runs: `theta_sites="2.2"` zeroes `(theta_b, theta_s)`
before they reach site 2.3's precompute call (`bayesian_statistics.py:4147-4148`), so this
guard's branch is never exercised in the registered configuration (matches the panel's own
finding 3, independently re-confirmed here). It binds only a future `theta_sites="all"` run,
where the caller must now explicitly pass `smear_global_selection=True` or receive the raise.

**R4 (must_fix 4). Item (e) — undisclosed step-width extrapolation on the b-band power
projection.**
CONFIRMED; no basis exists in the record to assert curvature is negligible over
`[-0.033, +0.033]` — the curvature leg (item (e)'s own quadratic fit through the three b-nodes)
is explicitly registered as *unmeasured until all θ-nodes exist*, so the disclosure is added
rather than a negligibility claim asserted without evidence. Appended, symmetric with the
already-disclosed s-band flag:

> **Disclosure (b-band power projection, symmetric with the s-band flag above).** The b-band
> SD proxy (A15) is a ONE-SIDED half-secant over a 0.02 step (`node_truth` vs `node_b_plus`
> only). It is carried, via population-size scaling alone (mirror n≈106 → production
> N_active=982), into a power projection for S0-B's TWO-SIDED, wider (±0.033, span 0.066)
> secant. If `lnL(b)` has curvature over this range, a wider two-point secant's variance need
> not scale from the narrower one-sided proxy's variance by population size alone — this
> transfer is doubly indirect (step-width extrapolation stacked on population-size
> extrapolation), on top of the already-flagged mirror→production transfer itself. No claim of
> negligible curvature is made; the registered curvature leg (item (e)) is the instrument that
> settles this once C0/b_plus_re/b_minus_re all exist.

**R5 (must_fix 5). Citation fix, item (j) — "105/105 moved."**
CONFIRMED. Re-read `WAVE2_REGISTRATION_CHECK_20260829.md`: §0 (the three chair findings F-A/F-B/
F-C) states the mean per-event Δln`combined_no_bh` decomposition (−0.1118 = −0.1193 kernel +
0.0075 global) and the bit-identity checks, but **not** the 105/105 count. That count appears in
**§6** ("Numbers with provenance (A11)"), row: *"registered b_plus vs truth (106 events): active
105/105 moved on `L_cat_no_bh`; mean Δln`combined_no_bh` −0.1118 = ..."*. (The same mis-citation
— "mirror 105/105 moved, §0" — is independently present in the wave-2 check's own §4 row 4, so
this is an inherited citation error, not one newly introduced by `PA-HIER-31`.) Item (j)'s
sentence is corrected: "mirror precedent: 105/105 moved, §0 of the wave-2 check" **→ "mirror
precedent: 105/105 moved, §6 of the wave-2 check."**

**Findings not requiring action (independently re-checked, no dispute).** The panel's
remaining findings all PASS on re-verification and require no further note here: `PA-HIER-31`
is genuinely append-only (`git diff HEAD` on this file is a pure addition after the prior
divider); the authorization stamp and orchestrator-quote wording match the row #222/#223 grant
and the 2026-08-29 path decision verbatim; the A22 commit-hash placeholder in item (g) is
explicitly named as a placeholder, not silently omitted; the arithmetic underlying items (a)
and (e) (b_max, node half-steps, materiality thresholds, SEM scaling, the 3.84·SEM detectable-
effect band) independently reproduces to the panel's own re-derivation; gap item 3 (this node)
is genuinely closed by items (a)/(b)/(i) even with R1's open contradiction noted, and gap item 4
(P6, θ CLI plumbing) is correctly left open, consistent with the uncommitted
`arguments.py`/`main.py` plumbing already present on this working tree.

*Authorization: launched under rows #222/#223 — charter node B1.2. Append-only; nothing above
this divider (including `PA-HIER-31` itself) is edited. No git operations; no source edits;
`hier_s0_driver.py` and `kwq1_score.py` untouched (owned by another agent);
`bayesian_statistics.py` untouched (physics-trigger file, out of scope; R3's correction is a
prose-only supersession of the registered gate text, not a code change). One item (R1) returned
to the author as a fresh `[RULE]` per CLAUDE.md's approval-scope convention; execution is not
blocked by it. Worker: sonnet-tier subagent, wave-2 GAP-CLOSURE workflow, 2026-08-29.*

---

### PA-HIER-31 REVISION NOTE 2 (2026-08-29; second refuter-panel `must_fix` response; append-only; `[FABLE-ORCH]`)

**Launched under rows #222/#223 — charter node B1.2.** A second refuter panel reviewed
`PA-HIER-31` together with `REVISION NOTE 1` and found `REVISION NOTE 1`'s own **R1** disposition
materially wrong on *scope* (not on its governance handling, which the panel separately confirmed
sound — see the "findings not requiring action" note below). Four `must_fix` items; three are
CONFIRMED and fixed as sub-notes below; the fourth (explicitly labelled "optional" by the panel)
is registered as a recommended, non-blocking follow-up **not executed by this node**, with the
reason stated. Nothing above this divider (including `PA-HIER-31` and `REVISION NOTE 1`) is
edited — corrections below are append-only supersession notices, the same pattern `PA-HIER-10`
and `REVISION NOTE 1` already used.

**R1′ (must_fix 1). Sub-note to `REVISION NOTE 1`'s R1 reconciliation — CoR-M is NOT untouched.**
CONFIRMED, material. `REVISION NOTE 1`'s R1 reconciliation paragraph (`:2231-2234`) states
verbatim: *"`PA-HIER-31(b)` is a genuine, targeted narrowing of `PA-HIER-10` restricted to CoR-P
(the production venue) only — `PA-HIER-10`'s pin is otherwise unchanged and still binds CoR-M
(the mirror), where every prior/future mirror-venue read keeps `smear_sigma_z=True` at every node
including truth-θ."* This is contradicted by the very item it reconciles. `PA-HIER-31(b)` itself
(`:1972`) reads: *"`smear_global_selection = False` for S0-B and for the S0-A remainder (P0)
alike."* **S0-A is not CoR-P.** By its own §2.1 definition (`:145`): *"**S0-A** (control) | mirror
venue, generator kernel = estimator kernel, truth-θ = (0,1) by construction."* — i.e. CoR-M. The
P0/S0-A-remainder scope is restated three further times in `PA-HIER-31(b)`/(e)/(i)/(A22) —
`:2090` ("S0-A remainder not yet run at check time"), `:2159` ("the S0-A remainder (P0)
certifies the instrument"), `:2165` ("P0 (S0-A remainder, same 2.2/unsmeared form)"), `:2190`
("**P0** (S0-A remainder, 2.2/unsmeared form) ... must both be recorded before S0-B banks") — not
a single slip.
**Correction, superseding `REVISION NOTE 1`'s R1 reconciliation paragraph (`:2231-2234`):**
`PA-HIER-31(b)`/(i) narrows `PA-HIER-10`'s unconditional `smear_sigma_z=True`-at-every-node pin
for **two** arms, not one — CoR-P (S0-B) **and** the S0-A remainder run under P0, which is a
CoR-M/mirror-venue arm by its own registered construction. `PA-HIER-10`'s pin is therefore
narrowed at **both** venues by this amendment, not "otherwise unchanged" at CoR-M. Every
downstream statement in `REVISION NOTE 1` that assumed a CoR-P-only narrowing (the "Option (i)"
blindness sub-item text (a′), and the "for CoR-P" framing of the open contradiction) was scoped
too narrowly; R2′ below broadens it.

**R2′ (must_fix 2). The CoR-M instance is registered as its own OPEN CONTRADICTION.**
CONFIRMED. `REVISION NOTE 1` registered exactly one open contradiction — *"`PA-HIER-10` vs
`PA-HIER-31(b)` for CoR-P"* — and returned it to the author as a fresh `[RULE]`. R1′ above shows
the same document pair also conflicts at CoR-M, and this is not free of consequence: `PA-HIER-20`
(line 1427, quote-verified against `correspondence_1d.py:328-337`: *"the mirror venue runs
`absolute_marginal`"*) independently found the mirror venue also runs
`normalization_mode="absolute_marginal"` — the same `normalization_mode` that gates the
path-(A) `phi` assembly (`catalogue_numerator_survival` resolving `"phi"` under
`absolute_marginal`, `bayesian_statistics.py:3535-3541`, PA-HIER-20 leg 3) which F-A found
non-inert for site 2.3 at CoR-P (`alpha_G_phi` −12.0%, `combined_no_bh` max_rel 7.45e-3, same
seed/node, `WAVE2_REGISTRATION_CHECK_20260829.md` §0). So the mechanism F-A measured only at
CoR-P is **mechanistically live at CoR-M too**, and its magnitude there is currently UNMEASURED —
a CoR-P-only F-A measurement cannot by itself license loosening the CoR-M pin, exactly the
situation GATE D3 clause (c) requires be *"explicitly documented ... as a design choice with its
stated consequence, never as an inherited oversight"* (`:305-307`), which has not happened for
CoR-M. **Registered here as a second, separate OPEN CONTRADICTION** (`PA-HIER-10` vs
`PA-HIER-31(b)`/(i) for CoR-M / S0-A), pending its own fresh author `[RULE]` — kept distinct from
the CoR-P item rather than folded into it, since the evidence bases differ (CoR-P has a direct
F-A measurement of the shift; CoR-M has only the PA-HIER-20 mechanism match, not a direct
measurement of its magnitude there — that gap is what R4′ below would close). This does not block
P0/S0-A-remainder execution under the row #222/#223 STANDING grant: every verdict is already
REPORTED-ONLY-pending-`[RULE]` per `REVISION NOTE 1`'s disposition, now extended to cover the
CoR-M arm as well.

**R3′ (must_fix 3). Item (h)'s falsifier text downgraded pending the CoR-M `[RULE]`.**
CONFIRMED; no basis exists to let the certification clause stand unqualified. Item (h) (`:2159`)
reads: *"DEAD is provisional until B0-P passes **and** the S0-A remainder (P0) certifies the
instrument in the same 2.2/unsmeared form."* As written this presents the P0 certification as
settled instrumentation without qualification. Given R1′/R2′, P0 now runs in a form that departs
from `PA-HIER-10`'s still-nominally-binding CoR-M pin, with no CoR-M-specific measurement of the
departure's consequence (only the PA-HIER-20 mechanism match, not a magnitude). **Appended
qualifier, superseding item (h)'s cited sentence:**

> The S0-A remainder (P0) "certifies the instrument" only with respect to wiring/arithmetic at
> `theta_sites="2.2"` (site 2.3 excluded, C-C identity check, GATE T-ID/ENG). It does **not**
> certify site 2.3's behaviour under `PA-HIER-10`'s originally-pinned, unconditional
> `smear_sigma_z=True`-at-every-node CoR-M form, because P0 does not run that form. Any DEAD
> verdict resting on "the instrument is certified" is REPORTED-ONLY-pending-`[RULE]` with respect
> to site 2.3 at CoR-M, exactly as `REVISION NOTE 1` already caps the CoR-P arm, until the R2′
> open contradiction is adjudicated.

**R4′ (must_fix 4, optional). Symmetric P1′ at CoR-M — registered as a recommended, non-blocking
follow-up; NOT executed by this node.**
The panel is right that a mirror-venue P1′ (one (0,1)-smeared node at CoR-M, mirroring the
already-registered CoR-P P1′ in `PA-HIER-31(b)`, ≈0.33 CPU-h / ≈20 min wall) would convert R2′
from a documentation gap into a measured, disclosed quantity for both venues symmetrically —
exactly how F-A was already handled for CoR-P. **Registered as the designated resolving
measurement for R2′**, promoted from purely-hypothetical to recommended-before-any-CoR-M-verdict,
mirroring the upgrade `REVISION NOTE 1` already applied to the CoR-P P1′ ("Option (ii)").
**Not executed by this node**, for two stated reasons: (i) this node's charter scope is the
`PA-HIER-31` **registration text** (gap-list item 3), not execution of new measurements —
`hier_s0_driver.py` is explicitly owned by another agent and this node is barred from editing it,
and driving a new run through its existing CLI is an execution action, not a text-registration
one; (ii) the panel's own ≈20-minute wall-clock estimate exceeds this workflow's per-command
foreground timeout (≤ 600 s) and this workflow may not run anything in the background or park
waiting on it, so it cannot be executed as a single compliant foreground command from this node.
**Recommended action, handed to whichever node executes P0/S0-B next:** run the mirror-venue P1′
before banking any CoR-M verdict that leans on "the instrument is certified" language, and record
its `alpha_G_phi`/`combined_no_bh` shift alongside F-A's CoR-P figures in the same table (item
(b)'s F-A table), so R2′ converts from an open contradiction into a disclosed, measured quantity
for both venues.

**Findings not requiring action (independently re-checked, no dispute).** The remaining items in
the panel's finding list — R2/R5's citation fixes (`compute_scores()` at `hier_s0_driver.py:394-449`;
the "105/105 moved" cite belonging to §6, not §0, of the wave-2 check), GATE D3(a)'s code-vs-prose
reconciliation (the raise-guard at `bayesian_statistics.py:2799-2806`, moot for the registered
`theta_sites="2.2"` configuration), the A15 SEM/detectable-effect arithmetic, the F4 cost/archive
figures (`COMPUTE_LEDGER.md` GAP-6 closure table, archive-scheduled=yes), the A22 placeholder
handling, the authorization-stamp format, the append-only-diff check, and the code citations
against `bayesian_statistics.py`/`correspondence_1d.py` at `dd63fe0c` — all independently
re-verify accurate on this pass and require no further correction here. In particular, the
panel's own assessment that R1's *governance* handling was sound (disclosing the confound
quantitatively, distinguishing an orchestrator execution-path call from a fresh scientific
`[RULE]`, returning the unsettled question to the author rather than resolving it unilaterally)
is unaffected by R1′/R2′ above: what was wrong was the **scope** of what got returned to the
author (CoR-P only, when CoR-M needed its own return too), not the returning-to-the-author
mechanism itself — R2′ now applies that same mechanism symmetrically to CoR-M.

*Authorization: launched under rows #222/#223 — charter node B1.2. Append-only; nothing above
this divider (including `PA-HIER-31` and `REVISION NOTE 1`) is edited. No git operations; no
source edits; `hier_s0_driver.py` and `kwq1_score.py` untouched (owned by another agent, and not
run by this node either, per R4′); `bayesian_statistics.py` untouched (physics-trigger file, out
of scope). Two items returned to the author: R2′ (a second, CoR-M-scoped fresh `[RULE]`,
additional to `REVISION NOTE 1`'s CoR-P-scoped item) and, by extension, R4′'s recommended-but-
unexecuted measurement (handed to the next executing node, not gated on author action);
execution of P0/S0-B is not blocked by either. Worker: sonnet-tier subagent, wave-2 GAP-CLOSURE
workflow, 2026-08-29.*

---

**P1 full-N result (orchestrator as runner) — appended 2026-08-29.** Registered CoR-P `b_plus`
node, seed900101, `theta_sites="2.2"`, unsmeared, at full N (single-`--jobs` path, run before the
separate `--jobs 2` P0 crash below). Diffs, smeared "all" form vs. unsmeared "2.2" form:

- `L_cat_no_bh`: max_abs **0.0** (bit-identical).
- `combined_no_bh`: max_abs **4.378e-4**, max_rel **7.447e-3**.
- `D_tilde_phi`: max_rel **7.503e-3**.
- `alpha_G_phi`: max_rel **0.1366** (13.66%).

Source: `fanout1_20260829/hier_s0_registered_run/logs/runner_wave2pre_20260829.log:762-765`.

**P0 crash disclosure.** The subsequent local S0-A completion run (`--jobs 2`, 4 seeds × all
nodes) crashed: `hier_s0_driver.py:970` `run_arm()` → `hier_s0_driver.py:647` `compute_scores()`
→ `pd.concat([...])` raised `ValueError: No objects to concatenate` — the per-seed node results
for `b_plus` are not collected across parallel workers when `--jobs>1` (only 1 of 4 requested
seeds present for `b_plus`, 0 for `truth`/`b_minus`/`s_plus`/`s_minus`). This is a driver defect
in the `--jobs>1` node loop, not a statistic: no `L_cat_no_bh`/`combined_no_bh`/`D_tilde_phi`/
`alpha_G_phi` value above is affected, since the P1 full-N result was produced on the
unaffected single-`--jobs` path before this run. `hier_s0_driver.py` is not touched by this
node (owned by another agent per standing scope); re-run pending the fix.

Launched under rows #222/#223 — [FABLE-ORCH], 2026-08-29.
