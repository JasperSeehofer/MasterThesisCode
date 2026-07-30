# Runbook — next session (written 2026-07-30)

Supersedes `RUNBOOK_NEXT_SESSION_5.md` (its §1–§2 items are DONE: campaign #53
complete, P1–P6 scored, 0.67 closure landed). Its §5 threads 3–7 remain open and
are re-listed in §8 below.

**Read in this order:**
1. `realistic_20260729/CLAIM_2D_BIAS_20260730.md` — the claim to be attacked. **Start here.**
2. `realistic_20260729/HANDOFF_20260730.md` — full state of the campaign and the
   investigation. (Its §4 is stale: HB is now refuted; see the claim's exoneration list.)
3. `realistic_20260729/REALISTIC_READOUT.md` — the P1–P6 scorecard. **Its §6 recommendation
   "the 1D channel is the defensible one" is WRONG** and is left in place deliberately so
   the next session can correct it with evidence rather than inherit a silent edit.

---

## 1. PRIME DIRECTIVE

**Refute before you build.** The previous session killed six candidate mechanisms,
two of which were its own confident leads (the Option-A drift; the "#51 is a
non-control" framing). Its remaining claim is written up as a *claim*, not a
result, with per-item provenance tags.

Do **not** start from "the mechanism is impostor rejection → completion fallback."
Start from "is that true, and is it even attributable to the scatter?"

Three gates, in order. Do not skip forward.

- **Gate A — provenance.** The headline class-split number (C3) came from an
  artifact that no longer exists. Regenerate it or the claim stays unfalsifiable.
- **Gate B — refutation.** Attack C3–C8 adversarially. Anything that survives is
  promoted from CLAIM to FINDING and written back into the claim file.
- **Gate C — alternatives.** Before accepting *any* surviving mechanism, run the
  alternative-cause sweep in §5. The previous session's failure mode was
  committing to a plausible mechanism early and burning ~1M tokens on it.

Only then tackle the fix.

---

## 2. MODEL & EFFORT POLICY (mandatory — the previous session overspent)

The previous session spent **~4.0M subagent tokens** across four workflows, with
roughly 0.8M lost to API failures and ~1M on a mechanism that was refuted. Every
`agent()` call below **must** carry an explicit `model` and `effort`. Never inherit.

| task shape | model | effort | rationale |
|---|---|---|---|
| Mechanical extraction: grep logs, checksum, pull columns, recompute a documented formula on delivered files | `haiku` | `low` | verifiable output, no judgement; wrong answers are obvious |
| Bounded code tracing with a named file/function target ("what does `_z_prior_pdf_at` weight by?") | `sonnet` | `medium` | comprehension, not invention |
| Independent re-derivation of a physics quantity; adversarial refutation of a claim | `opus` | `high` | must not be fooled by a plausible story; this is where the previous session's wins came from |
| Final adjudication across conflicting agents; the physics decision | `fable` | `xhigh` | hardest reasoning, smallest number of calls |
| Literature check against Gray et al. 2020 (arXiv:1908.06050) equations | `sonnet` | `medium` | lookup + compare |

Hard rules:
- **Cap the refutation workflow at 6 agents.** It was 7 before and 3 of those were wasted.
- **Never spawn an `opus`/`fable` agent for something a `haiku` agent can verify.**
  Regenerating a CSV is not a reasoning task.
- **Give every agent the claim file path**, not a paraphrase of the claim. Paraphrase
  drift caused two of last session's dead ends.
- Pass the exoneration list verbatim. Re-litigating an exonerated suspect is the
  single most expensive failure mode in this project.

---

## 3. GATE A — provenance repair (do first, cheap)

**Blocker: the cluster was refusing SSH at handoff** (`Connection refused`,
bwunicluster.scc.kit.edu:22). Check first: `timeout 30 ssh -o BatchMode=yes bwunicluster 'echo ok'`.

### A1. One file decides whether Gate C's big test is even needed [`haiku`/`low`]
Read `$WS/run_20260729_seed61000/sig0_control/run_metadata_0.json` and report
`normalization_mode` + `host_z_kernel`.
- If it records **`generator_marginal` + point** → confirms C6: no estimator control
  exists, and §6's 2×2 test must be run.
- If it records **`absolute_marginal` + `volume_deconv`** and still matched #51
  byte-for-byte (md5 `1e81ba22`/`733c8d32`) → **C6 collapses**, the estimator switch
  is proven inert, and the whole attribution simplifies. This would be the cheapest
  significant result available.

### A2. Regenerate the 2D per-event data [`haiku`/`low`]
Only 4 h-points are needed to test C3/C4: pull
`$WS/run_20260729_seed61000/real_r1/simulations/posteriors_with_bh_mass/h_0_{725,73,735,81}.json`
(~80 MB each — pull, do not copy the 3.2 GB tree). Recompute the class split and
compare against C3's +2.97 / +15.83.
**Constraint that makes this a real test:** C2 fixes the *sum* at +18.80 [LOCAL,
verified], so only the partition is at risk.

### A3. Extend to other realizations [`haiku`/`low`]
Re-run the C1/C5 measurements (both are pure local recomputes, scripts implied in
the claim file) across all 10 runs and both seeds. Confirms nothing is r1-specific.
Seed62000 r1 already showed the same signature (in-cat argmax 0.8600, dark 0.6400).

---

## 4. GATE B — refutation workflow (≤6 agents)

Targets, in priority order. Each gets one `opus`/`high` attacker.

1. **C3+C4 (the mechanism).** Does the dark class really own 84%? Is the completion
   leg genuinely up-tilted in this venue, or is that assumed? Requires A2.
2. **C5 (the in-cat rail).** Is the 0.86 concentration real, or an artifact of the
   prior's upper bound? **Decisive sub-test:** widen the grid above 0.86 and see
   whether the peaks keep moving (real runaway) or stop (edge artifact). This is
   cheap and it is the highest-value single check in the whole set — C5 is the claim
   that most damages the 1D headline.
3. **C7 (the host-z kernel).** Compute the kernel's *actual* induced host-z shift
   numerically for the 76 hosts at their real σ_z, instead of via the mode formula.
   Beware: the local `z_error` column is stale vs the cluster parent (#40b PV width).
4. **C8 (reparametrization dependence).** Re-run the mass-coordinate C-scaling.
   Confirm 1D invariance is exact.

Adjudicate with one `fable`/`xhigh` agent. It must be told explicitly that
**"the claim is refuted" and "undetermined" are acceptable, valued outputs** — the
previous session's synthesis agents were told this and it materially improved them.

---

## 5. GATE C — alternative causes (before accepting anything)

Ranked, from the previous session's synthesis. Do not skip this because a
mechanism already looks good.

1. **The two mixture legs disagree by ~25 nats about the same dark population.**
   Untested by anyone. Is `β_G(h) = D − β_Ḡ` (taken from the completeness model `f`)
   consistent with `Σ_glob = Σ_g w_g P_det` (the actual catalogue sum)? If `f(z,h)`
   and the realized catalogue disagree, `(1−w_G)^N` with N=1588 amplifies it into
   hundreds of nats. **Readable from the per-h logs; costs nothing** [`haiku`/`low`
   to extract, `opus`/`high` to judge].
2. **The completion leg's absolute calibration at comp_frac ≈ 0.07.** Prior work
   (L-A) measured this estimator biased **high** by +0.7–5.4% at comp_frac 0.22–0.85
   in a truth-known harness. This venue is ~3× deeper than anything tested. Extend
   `validation/pp_coverage.py` down to comp_frac ≈ 0.05 — CPU, 10 seeds, truth known
   [`sonnet`/`medium` to wire, `opus`/`high` to interpret].
3. **The 2D selection denominator is never mass-marginalised** (`D(h)`, `:1056-1145`,
   is 3D and channel-common). This is HA's other half and the only reason its patch
   failed. Testing it *is* testing HA properly.
4. **`w_G(0.73) = 0.0697` derived vs empirical in-catalogue rate 0.0479** — a 45%
   discrepancy in the base of an exponent that supplies +394 nats/unit-h. Nobody has
   looked [`haiku`/`low` to measure].
5. **HB's residual:** the window's h-*flat* 68% scale suppression, which HB
   explicitly did not close. Rides on item 3.

---

## 6. THE DECISIVE TEST (after Gates A–C)

**Run the unscattered #51 catalogue through the #53 estimator.** One CPU evaluate
job, seed61000, 41 h-points, existing CRB, existing catalogue, **no code change** —
the scatter guards no-op on an unscattered catalogue (`bayesian_statistics.py:310`,
`if not catalogue_scattered: return`).

```
--evaluate --normalization_mode absolute_marginal --host_z_kernel volume_deconv \
           --host_mass_kernel auto      # on run_seed61000's unscattered catalogue + CRB
```

The missing cell of the 2×2:

| | point / generator_marginal | volume_deconv / absolute_marginal |
|---|---|---|
| **unscattered** | A = #51: 1D 0.7299, 2D 0.7300 | **B = the test** |
| **scattered** | forbidden by guard | C = #53: 1D 0.732, 2D 0.813 |

**B − A = estimator effect. C − B = scatter effect.** Read `map_h`, `map_h_2d`, and
the per-class summed profiles.

**Pre-register before running** (the campaign's own discipline):
- *Estimator owns it*: B's 2D MAP ≈ 0.78–0.82 and B's in-cat class argmax ≈ 0.86
  **even with exact host redshifts** ⇒ the realistic host-observation model is
  largely exonerated and the target is the host-z kernel's population weight (C7).
- *Scatter owns it*: B ≈ A (2D 0.730, in-cat argmax 0.730) ⇒ the estimator switch is
  inert and the mass-window/completion imbalance under scatter is the whole story.
- *Mixed*: read the split directly off B, in nats per class.

**Skip this only if A1 collapses C6.**

Cost ranking of everything on the table — nothing else should run first:
this test (1 CPU job, no patch) < HB's `MASS_WINDOW_MODE` A/B (1 CPU job + patch)
< HA's paired numerator+denominator fix (physics change + 2 jobs) < any GPU re-sim.

---

## 7. FIXES — routing and gates

Per `CLAUDE.md`, physics work routes **GSD → GPD**, and any formula/constant change
in a trigger file is a **hard `/physics-change` gate** (derivation, dimensional
analysis, limiting case, reference, regression test) before any code is written.

| item | nature | route |
|---|---|---|
| Per-class per-h `Σ ln p_i` log split by `host_galaxy_index >= 0`, both channels | instrumentation | plain GSD. **6 lines; would have caught everything in this investigation on day one** |
| Emit per-event `L_cat`/`B_num`/`w_G`/`D` to the diagnostics CSV in both channels on **every** run | instrumentation | plain GSD. The r1 extract was the only reason this was possible **and it is already gone** |
| Log `w_G(h)` at 7 s.f. not 4 (`:2335`) | instrumentation | plain GSD |
| P6 host-recovery counter | instrumentation | plain GSD. **Must translate index spaces** — `host_galaxy_index` is a positional label of the *pruned* frame and the prune runs on observed columns (20,834,171 parent vs 19,874,547 realized rows); a naive counter reports garbage. Diff was at `/tmp/.../scratchpad/p6/P6_instrumentation.diff` — **/tmp is volatile, regenerate** |
| Fix `docs/derivations/realistic_host_observation_model.md:645` (falsely claims "host-miss rate logged (P6)") | docs | plain GSD |
| HA: mass-marginalise `D`/`β_G`/`β_Ḡ` for the 2D channel, paired with the numerator | **formula** | `/physics-change`. Acceptance gate: ~~**2D MAP invariant under M → kM**~~ **[2026-07-30 adjudication] struck — vacuous as written: a *consistent* unit change M → kM of all inputs is exactly invariant, so the gate tests nothing. Gate restated as measure-invariance: 2D MAP invariant under `L_cat,2D → L_cat,2D/C` for arbitrary C (equivalently: both numerator legs must carry the same mass-density dimension, so C cancels event-wise). The 1D bitwise-invariance anchor already holds. See `realistic_20260729/gate_b_20260730/ADJUDICATION_20260730.md` C8.** (1D already satisfies it exactly — use as regression anchor). Expect the MAP to get *worse*; that is acceptable, the current stability is illusory |
| C7: add `p_det` + catalogue selection to the host-z numerator population weight (`:4201-4207`) | **formula** | `/physics-change`. Limiting case: **σ_z → 0 must reproduce the point kernel exactly** (it does today), and the induced shift must vanish ∝ (σ_z/z)² — so the fix is a pure large-σ_z correction and cannot disturb #51 |

---

## 8. Still open from runbook 5

3. **(d2) derivation** — selection-side M scatter/truncation. Note HB (its mass
   *support-truncation* form) is now REFUTED; the thread's remaining form is the
   mixture-balance one (§5 item 3).
4. **B_num residual-bias model.**
5. **#39 blind alternative-truth mock** — arguably wait for a trusted #53 universe.
6. **#23 completion-term realism** [paper-blocker]. Now much more urgent: the
   realistic headline is substantially a *completion-term* measurement (P3 missed:
   dark curvature +0.047–0.049 vs in-catalogue 0…0.09).
7. **Paper (#47) ON HOLD.** Every pre-`49251f3` number is suspect below ~3 mHz
   (issue #52). The headline H₀ must come from a trusted #53 run — **which does not
   yet exist**: 2D is biased +4σ and 1D is a crossing of railed runaways (C5).

---

## 9. Environment gotchas (all cost time last session)

- **SSH ControlMaster dies every few hours and then HANGS.** Clear with
  `ssh -O exit bwunicluster; rm ~/.ssh/cm-*`, re-auth with 2FA. **Always wrap ssh in
  `timeout 120`** — an unwrapped hang wedges a monitor or an agent silently.
- **Cluster was refusing connections at handoff.** Not the ControlMaster issue.
- **Workflow cache-resume is same-session only.** A new session cannot replay
  completed agents; salvage results by reading the run's `journal.jsonl` and feeding
  them into the new script as text.
- **Local `reduced_galaxy_catalogue.csv` is NOT the realization parent** (local
  sha256 `623527929d…` vs sidecar `parent_csv_sha256 7af3f4f4a2…`). Differs in
  exactly one column, `z_error` (#40b PV width). Use the cluster parent for
  width-sensitive work.
- **Posterior-directory trap:** seed61000's canonical dir is `posteriors_fixed`
  (plain `posteriors/` is the stale PRE-`ec09ed0` backup); seed62000's canonical dir
  IS `posteriors/`. Reading "posteriors/" uniformly mixes eras.
- **Workspace expires 2026-09-23.** No extensions left.

---

## 10. Author decisions still open

- **[RATIFY-R7] extra GPU truth seeds.** Measured: realization-level scatter
  sd ≈ 0.006, truth-seed difference 0.023, per-run σ_h ≈ 0.020 ⇒ **more truth seeds
  buy a stable headline; more realizations per seed do not.** Note the deferral's own
  premise is falsified: it expected Poisson-dominance in the spectroscopic-host count,
  but **all 164 in-catalogue hosts across both seeds are photometric, zero spec.**
- Whether to fix HA on well-posedness grounds knowing it worsens the MAP.
- Whether to fill the 0.67 closure row into `IDEALIZED_BASELINE_READOUT.md` from the
  parabola estimate (σ_h = 4.42e-4, peak 0.670053, +0.12σ) or run the zoom first.
- **Nothing in `realistic_20260729/` is committed** (60 MB of pulled posteriors plus
  the scoring script, readout, claim, and handoffs).
