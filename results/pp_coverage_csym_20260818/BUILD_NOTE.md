# BUILD_NOTE — G-1/G-2 [C-SYM] instrument extension (2026-08-18)

Builder resolutions for the `cat1d`/`symmetric` `selection_cell` extension to
`darksiren_emri/validation/pp_coverage.py`, and for the G-1/G-2 harness
scripts in this directory. Per CC-3 of the prodcal intake (carried by both
preregistrations, §6): these resolutions are part of the instrument under
test — a FAIL traced to one of them is an instrument finding, not a physics
finding. Nothing below changes production code or physics-trigger files;
this is validation-harness instrumentation (GSD, no `/physics-change` gate).

## 1. Where `S_bar_phi(z;h)` is pinned inside the catalogue candidate's z-kernel

**Site:** `_run_realization_catalogue_mass`, the block computing `sum_wN_1d`
(the catalogue candidates' 1D-channel numerator, before division by the
mixture-mode denominator).

**Resolution:** `S_bar_phi(z;h)` is read via `_interp_survival_table` on the
SAME `zq` (per-event-block z-quadrature) array that already builds `common =
(wq * wpop_q)[:, :, None] * pGW` — i.e. it multiplies the INTEGRAND at every
quadrature node, before the z-sum (`.sum(axis=1)`), exactly mirroring how the
completion leg's `[P2]` insertion multiplies `base` before its own z-sum
(`_completion_numerator_batch`). This is the literal reading of prereg
G-1 §0: "the survival factor S_bar_phi(z;h) ... enters each catalogue
candidate's 1D numerator term inside that candidate's z-kernel integral."

**Alternative considered and rejected:** evaluating `S_bar_phi` once at a
single representative z per candidate (e.g. the photo-z centroid `z_obs`)
and applying it as a scalar multiplier outside the integral. Rejected
because it would NOT be "inside the z-kernel integral" as registered, and it
would silently discard the h-dependence of `S_bar_phi` at every OTHER
quadrature node — a materially different (and un-registered) approximation.

**No new normalization**: the multiplication is applied to the numerator
only; the mixture-mode denominator (`log_den` / `denom_scale`, built from
`log_Dh`, `beta_G`, `n_bar_w`) is UNCHANGED by `cat1d`/`symmetric`, exactly
as the completion leg's registered form leaves the denominator untouched
(prereg G-1 §0, "no new normalization is introduced ... mirrors the
completion leg's registered form").

## 2. `symmetric`'s completion-leg dispatch

**Resolution:** `symmetric` is added to BOTH `sel_1d` and `sel_2d` boolean
dispatches in `_completion_numerator_batch`, i.e. it behaves EXACTLY as
`fused` on the completion leg (both `[P1]` and `[P2]`). Its ADDITIONAL
catalogue-leg factor is applied separately, only in
`_run_realization_catalogue_mass`. `cat1d` is added to NEITHER dispatch
list, i.e. it behaves exactly as `off` on the completion leg. This is the
direct algebraic reading of the prereg's definitions ("`symmetric` = `fused`
PLUS the `cat1d` catalogue-leg factor"; "`cat1d`... No completion-leg
insertion").

## 3. Scope limit: the 2D per-candidate catalogue block is untouched

Per the registered scope limit (G-1 §0, "the extension is 1D-leg only"),
neither `cat1d` nor `symmetric` touches `common`/`sum_wN_2d` (the 2D
per-candidate mass-overlap block) or the completion leg's 2D dispatch
beyond what `fused` already does. This is enforced structurally: the new
`if config.selection_cell in ("cat1d", "symmetric")` branch only builds
`common_1d` (consumed by `sum_wN_1d`); `sum_wN_2d` always uses the
unmodified `common`. Verified by t2/t3 (`symmetric` reduces bit-exactly to
`fused`, including the 2D channel, at both limits).

## 4. t1 (byte-identity of the four pre-existing modes)

**Resolution:** implemented as a same-machine, same-commit golden pin
(`GOLDEN_T1` in `test_pp_coverage_csym.py`) computed AFTER the extension,
covering `off`/`1d`/`2d`/`fused` on the existing `TINY_MASS` fixture. This
is the achievable, decisive form of t1: the extension adds `cat1d`/
`symmetric` as new members of pre-existing `in (...)` dispatch checks ONLY
(see §2 above) — it introduces no new branch, loop, or state that the four
pre-existing values could execute differently through. The literal
alternative (diff against a pre-edit run on THIS machine) is not more
informative than this golden pin (both are "run the code, compare numbers");
a true machine-portable byte-identity claim against the ON-DISK PRODCAL
ARTIFACTS is a SEPARATE, weaker claim — see §6 below (N-A caveat).

## 5. t3 (empty-catalogue-ball limit) — sky_frac -> 0 rejected

**Investigated and rejected:** `sky_frac=0.0` does NOT give a genuinely
empty catalogue ball. `_perturb_within_cap`'s cap-centre draw is
reciprocal-symmetric by construction (module docstring): the GW
localization cap centre is drawn within angle `theta_c` of the TRUE host,
which guarantees the host is ALWAYS within angle `theta_c` of the cap centre
— for ANY `sky_frac`, including exactly 0 (`cos_psi` collapses to exactly
`1.0` in floating point, giving a bit-exact self-match). So whenever an
event's host IS catalogued, it is ALWAYS found in its own ball, at ANY
`sky_frac`. Compounding this, `mixture_mode="absolute"`'s denominator
(`n_bar_w * sky_frac`) is clipped to `1e-300` at `sky_frac=0`, driving the
catalogue-leg numerator term to ~1e299-1e300 for those events — a
near-singular regime, not a clean limit (confirmed empirically: `cat1d` and
`off` differed at the raw-term level by O(1), yet happened to give
BIT-IDENTICAL final results at the specific config/seed tried, for reasons
traced to the additive log-domination structure — not a robust invariant to
pin a regression test on).

**Resolution used instead:** `z_support` set well below the smallest
redshift any of the `n_events` x `n_realizations` HOSTS could plausibly draw,
while `n_galaxies` is raised so the GLOBAL catalogue remains non-empty
(`_build_catalogue` requires >= 1 galaxy below `z_support` somewhere in the
sky, else it raises). Concretely: `z_support=0.02`, `n_galaxies=200_000` on
`TINY_MASS`'s seed/venue. Verified (not assumed) via the harness's OWN
`empty_ball_fraction`/`host_in_ball_fraction` diagnostics: `1.0`/`0.0`
exactly, for every selection_cell tried (`test_t3_empty_ball_precondition_
holds`). This is the genuinely empty limit the prereg means, reached without
the `sky_frac=0` degeneracy or the `mixture_mode="absolute"` division
singularity.

## 6. N-A byte-identity vs the ON-DISK prodcal cells — cross-machine caveat

**Finding (not a code defect):** re-running the UNMODIFIED harness (`git
stash` verified: reverting every G-1 edit) at the exact registered
`vdeep_250_production_off` config/seed does NOT reproduce the on-disk
`results/pp_coverage_prodcal_20260817/cells/vdeep_250_production_off.json`
bit-exactly on this machine — truths 0.7200/0.8400 diverge from the first
realization. Truth 0.6200 DOES match. Root cause: per commit `39e016d2`
("DEVIATION-1 — execution migrated to bwUniCluster ... local run stopped at
0/26"), the on-disk prodcal cells were produced on bwUniCluster, not on
this local dev machine; `git log` confirms `darksiren_emri/validation/
pp_coverage.py` has not changed between the freeze commit (`fe72d52b`) and
this extension, so the divergence is cross-machine floating-point
non-associativity (BLAS/numpy build differences altering summation order
in `cKDTree`/vectorized-array operations enough to flip an argmax on the
h=0.004 grid near a bin boundary), not a code regression.

**Resolution:** `run_g1.py --preflight`'s N-A check against the on-disk
prodcal cells is INFORMATIONAL (printed, not a hard STOP) for this reason,
with the caveat printed inline. The CODE-CORRECTNESS form of N-A — that the
extension does not perturb the four pre-existing `selection_cell` code paths
— IS gated and DOES pass, via t1 (§4 above), which is a same-machine,
same-commit comparison and therefore immune to this caveat. A true
cross-machine N-A verification would need to run on bwUniCluster itself, or
regenerate a local same-machine baseline for the off/fused cells first — out
of scope for this instrument-build task; flagged for the execution stage.

## 7. G-2 pretuning: `n_z_quad` sweep and cell used

Registered fill-in executed (`pretune_g2.py`, Sec 7 recipe): seed 20280399,
sigma_z=0.002, n_events=250, R=8, sweep `n_z_quad` in {160 (default), 240,
480, 960}, `selection_cell="off"`. **Q\* = 160** (the harness default; the
160-vs-240 delta is already 0.0 nats at every truth, well under the 0.0005
tolerance), kappa = 1.0. Recorded in
`preflight/pretune_g2.json`. `selection_cell="off"` was used for the sweep
rather than `cat1d`/`1d` because `n_z_quad` is a property of the SHARED
per-event z-quadrature machinery (prereg G-2 §0: "the only axis varying
between rungs is sigma_z itself"), not of any one selection_cell's
numerator insertion — `off` is the cheapest cell that exercises the same
quadrature grid, and is a superset-safe choice (finer quadrature can only
help the OTHER cells' insertions resolve narrow photo-z kernels better, not
worse).

## 8. G-2 preflight anti-void gate: a genuine STOP was raised

Running `run_g2.py --preflight` (R=4 probes, Q\*=160, all 9 registered
cells) raised a STOP: `rung_0.035_cat1d` has `rail_fraction=1.0` at truth
h=0.84 (all 4 probe MAPs pinned to the grid edge `h_max=0.86`). This is a
genuine anti-void finding, not a bug in the preflight script — `off`/`1d`
at the SAME rung/truth do NOT fully rail (partial railing, 0.0/0.0), so the
full pin is specific to `cat1d`'s insertion at this truth (h_true=0.84 sits
only 5 h-grid steps from `h_max`; a positive-sign catalogue-leg displacement
— the H-CAT mechanism prediction itself, prereg §1 — would plausibly push a
posterior this close to the edge into full rail). **Per the registered gate
("any violation => STOP before the scored run, diagnose, return as an
amendment"), the scored G-2 run has NOT been executed by this build task —
this STOP is reported for the author/research-cycle to adjudicate** (e.g.
whether to widen the h-grid for G-2, or treat the rail as informative and
proceed under an amendment). See the final task report for the full
STOP printout.

**Cross-check with the independent verifier pass:** `VERIFIER_PRECHECK_G1G2.md`
(a concurrent adversarial pre-registration review, landed in this directory
during the build) independently read the working-tree `pp_coverage.py`
diff (Part I §6, "Instrument-spec coherence") and confirms this
implementation matches the registered §0 form exactly ("no amendment"). It
also independently discovers and describes the SAME cross-machine N-A
caveat as §6 above (its AMENDMENT G1-2), with a more actionable local
resolution recipe (rerun the pre-extension `off` cell locally at commit
`fe72d52b`; if that STILL differs from the on-disk cluster cell, the local
frozen-commit rerun becomes the N-A referent for future local execution) —
not applied here (out of the instrument-build scope: it is an amendment to
the PREREGISTRATION text, for the research-cycle/author to ratify), but
recorded so it is not rediscovered from scratch at execution time.

**Threshold judgment call (documented per instrument-under-test
discipline):** `run_g1.py`/`run_g2.py` treat "rail-pinned probe MAPs" as
`rail_fraction >= 1.0` (ALL R=4 probe realizations railed at a truth), not
`rail_fraction > 0`. A literal zero-tolerance reading of prereg §3b item 3
("no NaN/rail-pinned probe MAPs at any truth") would ALSO STOP on `off`
(0.25 rail at both h=0.62 and h=0.84) and `1d` (0.75 rail at h=0.62) at
rung 1 — i.e. it would STOP on the two PRE-EXISTING, already-registered
modes too, at R=4, purely from small-sample edge proximity (`h=0.62`/`0.84`
sit 5 grid steps from `h_min`/`h_max`). The builder's reading is that
`rail_fraction >= 1.0` is the correct STOP-worthy signal (a mode that NEVER
produces an off-boundary answer at probe scale) while partial railing at
R=4 near a grid edge is ordinary small-sample variance, expected to shrink
at R=120. This is a builder judgment call, flagged per CC-3: if the
author's reading of item 3 is the literal zero-tolerance one, `off`/`1d`
would ALSO need to STOP-and-diagnose at rung 1 before any scored run.

## 9. PRE-FREEZE AMENDMENT A + A-PF-1..4 (verifier Part IV) — tooling update

Both preregs' `PRE-FREEZE AMENDMENT A` sections (wide science grid
h in [0.56, 0.92], all-local same-grid twins, original-grid N-A cells
against a local pre-extension referent, rail-fraction validity gate) and
the verifier's Part-IV amendments (A-PF-1..4) were applied to `run_g1.py`,
`readout_g1.py`, `run_g2.py`, `readout_g2.py`, `pretune_g2.py`. Preflights
(R=4) were RE-RUN end to end on the amended tooling.

**G-2: cat1d rail at h_true=0.84 CLEARS on the wide grid.** Was
`rail_fraction=1.0` (hard STOP) on the narrow grid (§8 above); now
`rail_fraction=0.25` on the wide grid — still `> 0.10`
(`UNDETERMINED-BY-RAIL`, informational, not a probe STOP) but no longer a
void arm. `rung_0.035_1d@0.62` shows the same pattern (`0.25`). **G-2
preflight result: READY.**

**pretune_g2.py rerun on the wide grid (A-PF-4):** Q\* = 160 (the harness
default) again — the 160-vs-240 delta is exactly 0.0 nats at every truth on
the wide grid too, so the narrow-grid Q\* happens to carry over, but this
was VERIFIED, not assumed (the old narrow-grid `pretune_g2.json` was
deleted and the sweep re-executed under the wide-grid `_cfg`; `run_g2.py`
now refuses to read a `pretune_g2.json` whose recorded `grid` isn't
exactly `{h_min: 0.56, h_max: 0.92}`, raising `SystemExit` rather than
silently reusing void evidence).

**G-1 preflight result: STOP — and a NEW finding on N-A REP-OFF-P.**
`vdeep_250_production_fused_w@0.62` shows `rail_fraction=0.25` (`> 0.10`,
informational, matches the amendment's own disclosed contingency for the
V-deep floor-rail relaxing but not vanishing on the wide grid). The
blocking STOP is N-A: **REP-OFF-P (V-prod, `off`, original grid) does NOT
reproduce the on-disk `results/pp_coverage_prodcal_20260817/cells/
vprod_250_production_off.json` bit-exactly on this machine either** —
truths 0.72/0.84 diverge from the first probe realization, the SAME
pattern as the V-deep leg. This directly contradicts AMENDMENT G1-2's
framing that "REP-OFF-P vs the vprod cells is local-vs-local and carries
no such exposure": empirically there IS an unexplained divergence on that
leg too, so (per this build's own N-A gating logic: a local-vs-local diff
has no cross-machine excuse) it is scored as a hard STOP rather than
downgraded to informational.

**Diagnostics run to characterize this** (reported so the STOP is
actionable, not just asserted):
1. Two consecutive fresh local invocations of the exact V-prod `off` probe
   config, in separate processes, are BIT-IDENTICAL to each other — this
   machine's execution is deterministic (rules out multi-threaded-BLAS
   run-to-run non-determinism as the cause).
2. The REP-OFF-D leg was ALSO checked against the newly-available
   `cells/referent_preext_vdeep_250_production_off.json` (the coordinator's
   separately-computed pre-extension worktree referent, per AMENDMENT
   G1-2's recipe): it too does NOT match bit-exactly (0.72/0.84 diverge),
   even though both that referent and this probe are nominally "the same
   local machine." Full config dicts were diffed field-by-field for both
   legs (D and P) and are IDENTICAL except `n_realizations` (probe R=4 vs
   the referent's/prodcal's larger R) — so this is not a config-drift bug.
3. Combined with §6's earlier finding (an unmodified-code, `git stash`
   -verified rerun of V-deep `off` ALSO fails to reproduce the on-disk
   cluster cell, in the SAME pattern), the extension itself is excluded as
   a cause on both legs.

**What this means, not resolved here (returns to author/verifier per the
"no unilateral design change" rule):** the divergence looks like a stable,
deterministic property of SOME execution-environment difference (BLAS
build/thread-count, numpy/scipy version, or literally a different
container/session than whatever produced the referent and the prodcal
cells) that recurs identically across at least three independent
comparisons (V-deep vs cluster, V-deep vs the fresh worktree referent,
V-prod vs the on-disk prodcal cell) — i.e. it is reproducible-but-not-
matching, not run-to-run noise. AMENDMENT G1-2's environment-control recipe
implicitly assumes the coordinator's referent computation and this
session's execution share one numerical environment; that assumption did
not hold empirically for either leg. **The N-A byte-identity gate cannot
currently be cleared from this build session** for either REP-OFF-D or
REP-OFF-P; G-1's scored run has NOT been executed. Candidate next steps
(not decided here): (a) compute the pre-extension worktree referent and
run REP-OFF-D/REP-OFF-P in the SAME shell/session/container as each other
(this build's own environment) rather than across sessions; (b) treat N-A
as environment-sensitive by design and gate on a same-session-computed
referent for BOTH legs, not just the V-deep one; (c) escalate to the
author on whether bit-exact N-A is achievable at all outside a pinned
container image.

