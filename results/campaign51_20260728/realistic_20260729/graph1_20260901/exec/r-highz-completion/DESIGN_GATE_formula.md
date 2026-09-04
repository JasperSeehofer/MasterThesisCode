# DESIGN_GATE_formula.md — r-highz-completion, FRESH integration + formula review

Reviewer: fresh integration/formula reviewer, no prior context. `INFORMATION_FORECAST.md` was
**not opened** (forbidden). Method: (1) read `REGISTRATION_DRAFT.md` + `MECHANISM_NOTE.md` fully
and built my own enumeration of every statistic/gate/disposition-row/CLI-band named there, before
opening `highz_decomp_reads.py`; (2) located the implementing code for each item, independent of
`BUILD_RECORD.md`'s own checklist (read afterward, for cross-check only — several gaps below were
never claimed as done by that checklist either); (3) ran `--dry-run` on the real, pinned inputs
myself; (4) ran `--synth` myself and independently re-derived the closure identity, the B-term
freeze `Δ_B`, and the non-additivity residual `r` by hand, outside the script; (5) constructed a
standalone numeric test of one statistical formula (harness `SE_F^harn`) using the script's own
`harness_pool_score` function on synthetic per-universe arrays. **No registered aggregate was
computed over `K_dark`/`R`/`K`/`P_dark` or any harness universe's real `event_likelihoods.csv`** —
every number below is either from the synthetic fixture, a hand-constructed correlated toy input,
the `--dry-run`/`--synth` console output, or a source-line citation.

## Verdict: **RED**

Three confirmed, code-level formula/gate defects would let the real-mode read either produce a
disposition the draft's own band table forbids, or silently skip byte-id anchors the draft
registers as gates — i.e. the read could complete with exit 0 while carrying a wrong or
unregistered number. A fourth item is a genuine statistical-formula ambiguity that measurably
changes a decisive statistic (`Z_harn`) in a constructed example. None of the four was flagged in
`DESIGN_GATE_computability.md` (out of scope there — the script didn't exist yet) or in
`BUILD_RECORD.md`'s own checklist (which never claims to have wired up G-2(i)/(ii)/(iii), and
doesn't mention `--nonadditivity-max` being unused).

---

## 1. What I verified independently and confirm CORRECT

**1.1 `--dry-run` on the real, pinned Sec.8 launch block** (run myself, verbatim CLI):
exit code 0; all four file pins OK; all 8 population pins OK (`iiib P_dark=606`,
`K=159`, `K_dark=144`, `R=231`; `jr1 P_dark=493`, `K=159`, `K_dark=111`, `R=191` — **matches the
task's required 606/144/231 exactly**); harness manifest sha256 matched
(`6a06063dd5…adb1c0a2`, 67 universes); G-3d 13/13 tokens, 67/67; harness pooled
`Σn_scored=12,060` matched anchor; G-1 5-row real-slice closure residual `2.665e-15` (band `1e-9`);
`[SYNTH OK]`; no `--out` file written (confirmed by `ls` failing afterward). Reproduces
`BUILD_RECORD.md` §4's transcript byte-for-byte.

**1.2 Missing-input path is a hard INSTRUMENT-DEFECT.** Ran with `--logl-iiib /nonexistent/path.csv`
(dry-run): script prints `INSTRUMENT-DEFECT: missing registered input file(s):
['/nonexistent/path.csv']` and exits **1** (`InstrumentDefect(SystemExit)`, raised by `preflight()`,
line ~266, before any pin/hash is even attempted). Confirmed — matches the node-lesson requirement
("a missing registered input is a hard INSTRUMENT-DEFECT").

**1.3 T0 convention is imported, not re-implemented.** `_import_t0_module()` (line ~63) uses
`importlib.util.spec_from_file_location` + `exec_module` to load `build_influence_vector.py` by
path at runtime, then pulls `_load_matrix`, `_physics_floor_apply`, `_moments` off the live module
object (lines 78-80); these three names are genuinely defined in `build_influence_vector.py` (I
grepped it directly) with the content REGISTRATION_DRAFT.md §1 claims (gradient weights, physics
floor, log-sum). This is a real dynamic import, exercised successfully by both `--dry-run` and
`--synth` above (both call `_load_matrix`/`_moments` transitively). Confirmed.

**1.4 SYNTH fixture, hand-verified outside the script** (`make_synth_fixture`, 6 events × 5 nodes,
seed 0, `T_g`/`T_D` exactly flat, `T_B` carrying all the tilt for events 4,5). I reproduced the
fixture construction independently in a fresh Python snippet (not importing the script's
functions) and re-derived, by hand, the centered profiles, the reference profile (median over
`R = {0,1,2}`), the single-term freeze `Δ_B`, the all-terms freeze `Δ_F`, and the closure residual:

```
delta_B = -0.0004994591393654435
delta_g =  0.0
delta_F = -0.0004994591393654435   (= delta_B, since T_g is exactly flat -> zero contribution)
r (non-additivity) = delta_F - (delta_B + delta_g) = 0.0   (exact, float64)
s_B = delta_B/delta_F = 1.0 ;  s_g = 0.0
```
This matches the script's own asserted values (`s_B==1.0` to 1e-9, `s_g==0` to 1e-6, `r==0` to
1e-9) exactly, and independently confirms the closure identity `Λ_F = Λ_full − Σ t̂ + n·t̄` and the
`Δ_t = mean_h(Λ_t) − mean_h(Λ_full)` formula are implemented as REGISTRATION_DRAFT.md §2.3
specifies, for the one-term-owns case. The G-1 closure gate's pass/INSTRUMENT-DEFECT-raise paths
in `run_synth_check` (hand-built 3-row table, one row's `L_cat_no_bh` perturbed to `1e-30`) were
read and are correct: `gate_g1_closure` raises `InstrumentDefect` on the perturbed row (G-1a,
bit-exact-zero check) and passes on the clean one.

**1.5 Formula mapping, term profiles and closure (Sec.2.1, MECHANISM_NOTE §2-4).**
`T_B = ln(B_num)`, `T_g = ln(B_num_wbh) - ln(B_num)`, `T_D = -den_log_term` (`compute_term_profiles`
line ~531, `compute_T_D` line ~549) match (I-2D)/(I-1D) exactly; `gate_g1_closure` (line ~484)
implements G-1(a)-(e) as specified, using `float_precision="round_trip"` (Finding C, resolved) and
correctly reproduces `DESIGN_GATE_computability.md`'s own 2.665e-15 residual on the real 5-row
slice (confirmed in the dry-run transcript, §1.1 above). `T_D`'s `Δ_D = 0.0` is hardcoded as an
identity (`TermFreezeResult.delta_D`, not measured) — correct per Sec.2.3/G-1c.

**1.6 Bands correctly threaded — five of six.** `production_ownership_disposition`'s `share_own`/
`share_diffuse` and `harness_outcome_disposition`'s `z_gate`/`rho_hi`/`rho_lo`/`se_unpowered` are
all genuine function *parameters*, populated from `args.*` at the call sites in `run_production_family`
/`main`, and are exercised correctly by `run_synth_check`'s six harness-disposition assertions and
four production-disposition assertions (all read and hand-checked against REGISTRATION_DRAFT.md
§5's two tables — correct for every row **except** the two gaps in Finding C below).

---

## 2. Confirmed defects

### Finding A (RED, blocking) — `--nonadditivity-max` is a dead CLI flag

`build_argparser` (line ~252) adds `--nonadditivity-max` (default `0.6`) — the fix for
`DESIGN_GATE_computability.md` Finding B. But `production_ownership_disposition()` (the function
that actually applies the band) takes no such parameter; its body hardcodes the literal `0.6`:

```
if top_share >= share_own and r_over_abs_delta_F <= 0.6:
    return f"TERM-OWNS({top_name})"
```

`args.nonadditivity_max` is written **only** to `out["bands"]["nonadditivity_max"]` (line ~1513) —
metadata, never passed to the disposition call in `run_production_family` (grepped every use of
`nonadditivity` in the file: argparse definition, the `TermFreezeResult.r_nonadditivity` field
name — an unrelated field — and the metadata write; nothing else). Compare with the other five
bands (`share_own`, `share_diffuse`, `rho_hi`, `rho_lo`, `z_gate`, `se_unpowered`), which **are**
genuine parameters of their disposition functions (§1.6 above) — this is the one exception. A
launch that deliberately overrides `--nonadditivity-max` (e.g. to sensitivity-test the R-c band
before ratification) would silently have zero effect on the booked disposition while the run's own
metadata reports the overridden value as if it had been applied — a false record, not just an
unused flag. Fix: add `nonadditivity_max: float` to `production_ownership_disposition`'s signature,
replace the literal `0.6`, and pass `args.nonadditivity_max` at the call site.

### Finding B (RED, blocking) — G-2(i)/(ii) byte-id anchors are defined but never checked

`G2_MEAN_H_FULL` (the four full-sample `mean_h` values to 1e-9, G-2(i)) and `G2_DELTA_K_IIIB_2D` /
`G2_DELTA_K_TOL` (the `+0.086106` K-leave-out anchor, G-2(ii)) are declared as module constants
(lines ~133-143) but grepping the whole file for `G2_MEAN_H_FULL`, `G2_MEAN_H_TOL`,
`G2_DELTA_K_IIIB_2D`, `G2_DELTA_K_TOL` shows **zero other occurrences** — no `InstrumentDefect`,
no assertion, no print, compares the script's own computed `mean_h_full` (`ProductionFamilyResult.
mean_h_full`) or `delta_K_dark_leaveout` against either anchor at any point, dry-run or real. This
is not merely undocumented: `BUILD_RECORD.md`'s own §3 checklist table lists **only** G-2(iv) among
the four G-2 sub-items (i, ii, iii, iv) — (i)/(ii)/(iii) were never claimed as implemented, and are
not. Since these are the anchors that would catch a wrong population→CSV join, a wrong `h_true`
index, or a T0-import drift (the entire point of "byte-id anchor," per Sec.6 G-2), their absence
means a real-mode run could reproduce a `mean_h_full` that silently disagrees with the value
`build_influence_vector.py`/`BUILD_RECORD_B2.md` already established, or a `Δ_K,dark` leave-out
that disagrees with the independently-banked `Δ_K = +0.086106`, and the script would not notice.
Fix: after computing `mean_h_full` per venue/channel and `delta_K_dark_leaveout` (iiib 2D only),
assert against `G2_MEAN_H_FULL[(venue, channel_label)]` (tol `G2_MEAN_H_TOL`) and
`G2_DELTA_K_IIIB_2D` (tol `G2_DELTA_K_TOL`) respectively, raising `InstrumentDefect` on mismatch.

### Finding C (RED, blocking) — G-2(iii) "0 physics-floor exclusions" is silently discarded

`_load_matrix()` (the imported T0 function) returns a 4-tuple `(h_grid, event_idx, logL,
n_excluded)` — `n_excluded` is the count of events the physics floor excluded (an all-zero row).
Every one of the three call sites in `highz_decomp_reads.py` (`run_production_family` line ~1061,
`run_K_hosted_leaveout` line ~1153, `run_harness_universe` line ~1229) slices `[0:3]`, discarding
`n_excluded` entirely — it is never read, printed, or compared to the registered anchor "0". Two
consequences: (a) G-2(iii) is unimplemented, exactly like Finding B; (b) more concretely, if any
physics-floor exclusion *did* occur, `_load_matrix` silently **drops that event's row** from
`event_idx`/`logL` before returning — `run_production_family`'s `idx_pos = {int(e): i for i, e in
enumerate(event_idx_full)}` lookup for a `K_dark`/`K_hosted` member that got excluded would then
raise an uncaught `KeyError`, not a clean `InstrumentDefect` — a crash rather than the registered
gate's fail-safe STOP. Fix: capture `n_excluded` at all three call sites and raise
`InstrumentDefect` if nonzero (or explicitly assert `== 0` against the G-2(iii) anchor).

### Finding D (AMBER→ requires resolution, not merely cosmetic) — `production_ownership_disposition` omits two named INTERMEDIATE carve-outs

REGISTRATION_DRAFT.md §5's production table gives TERM-OWNS(t) a single literal test (`s_t≥0.5`,
largest, `|r|/|Δ_F|≤0.6`), then separately *names* three example patterns that fall into
INTERMEDIATE's "otherwise" bucket: `0.2 ≤ max s_t < 0.5`; **"both ≥ 0.5 with r < 0"**; and
**"sign-opposed terms with |s| > 1 each."** The code (`production_ownership_disposition`, line
~753) implements only the literal TERM-OWNS numeric test and the DIFFUSE-IN-TERMS test, falling
through to INTERMEDIATE by exclusion — it never separately checks whether the *second*-largest
share is also ≥0.5, or whether the two shares are sign-opposed with `|s|>1` each. Both of the
draft's named exclusions are constructible cases that literally satisfy the code's TERM-OWNS
condition. Hand-verified with two constructed share-pairs, run through the *actual*
`production_ownership_disposition` function on the built script (not a copy):

- `s_B=0.55, s_g=0.52` (both ≥ 0.5; `r/Δ_F = 1-(0.55+0.52) = -0.07`, so `r<0` and `|r|/|Δ_F|=0.07≤0.6`)
  → code returns `TERM-OWNS(B)`. Draft names this pattern INTERMEDIATE.
- `s_B=3.0, s_g=-2.0` (sign-opposed, `|s_B|=3>1` and `|s_g|=2>1`; `r/Δ_F = 1-(3-2)=0`, `r_over=0≤0.6`)
  → code returns `TERM-OWNS(B)`. Draft names this pattern INTERMEDIATE.

Since this is a 2-term system (2D: B, g only) this is not an exotic edge case — it is exactly the
regime a real anti-correlated completeness/mass-factor pull could land in. Because the harness
outcome table (§5's second table) is *conditioned* on which term production names as owning
(`s_t^harn_owning`, wired in `main()` via `prod_owning_term = ordered[0][0] if
prod_primary.disposition.startswith("TERM-OWNS") else None`), a wrong TERM-OWNS booking here would
also misroute the harness-side disposition and the §9 fresh-RULE routing (item 1's MANDATORY
follow-up fires only on TERM-OWNS+ESTIMATOR-INTERNAL). This needs either a code fix (check the
second-largest share explicitly) or an explicit author ruling that the code's simpler literal test
is what was actually meant (in which case the draft's parenthetical examples should be struck, not
just left inconsistent with the implementation).

### Finding E (AMBER, informational but decisive-statistic-adjacent) — harness `SE_F^harn` combines per-term jackknife SEs by quadrature, not a joint jackknife of the sum

§4.3 registers "the POOLED `S_t^harn, S_F^harn`... with delete-one-universe jackknife SE." In
`main()`, `harness_pool_score` (the jackknife implementation) is called **separately per term**
(`for term in ("B","g")`), and the two resulting SEs are combined as
`se_F_harn = sqrt(sum(v**2 for v in harness_pooled_SE.values()))` (line ~1411) — i.e. `S_F_harn`'s
SE is computed as if `S_B^harn` and `S_g^harn` were independent, even though `S_F_harn = S_B_harn +
S_g_harn` is an exact sum whose own delete-one-universe jackknife (recomputing `S_B_u + S_g_u` per
held-out universe directly) is the more literal reading of "delete-one-universe jackknife SE" of
`S_F^harn`. I constructed a standalone test calling the script's own `harness_pool_score` on 20
synthetic universes with a shared per-universe nuisance term correlating `B_u` and `g_u` (a
plausible real structure — both terms are computed from the same held-out universe's event set):
quadrature-combined `SE_F = 0.03023` vs. direct-joint-jackknife `SE_F = 0.02695` — a **12%**
difference, propagating directly into `Z_harn = S_F^harn/SE` (`27.3` vs `30.6` in the toy). Since
`Z_harn` is the first-line gate in `harness_outcome_disposition` (UNPOWERED-CONTROL / FLOOR-
CONSISTENT / the `|Z_harn|>3` branch), a 12%-scale SE difference is not academic near any of the
`z_gate=3.0`/`se_unpowered=0.1` boundaries. Recommend: jackknife `S_F_u = S_B_u + S_g_u` directly
(sum the per-event stencil-slope arrays for B and g before pooling) rather than combining two
separately-jackknifed SEs in quadrature — cheap, since both terms already share the same per-
universe event sets (`K_dark,u`/`R_u`).

### Finding F (informational, non-blocking) — two "reported-only" cross-checks from §2.3/§2.4 are never materialized

§2.3's concordance ratios (`Δ_F/Δ_K,dark`, `Δ_K,dark/Δ_K` with anchor `+0.086106`) and §2.4's
stencil-consistency check (`Δ_t ≈ |K_dark|·S_t/I_HEAD`) are both explicitly "reported-only (not a
gate)." Grepped the file: neither is computed. Low severity — `Δ_F`, `Δ_K,dark`, and the
`G2_DELTA_K_IIIB_2D` anchor constant are all already present in the script/output, so a reader can
compute the ratios by hand from `--out`'s JSON without a script change; the stencil-consistency
check needs `I_HEAD` (undefined in the draft/script — presumably the H_GRID_41 node spacing or
Fisher information scale, not otherwise named), so it cannot even be added without a fresh
definition from the author. Recommend leaving both as reader-side post-processing rather than
holding up the launch, but note the gap so the reader doesn't assume `--out` already contains them.

### Finding G (informational, low-confidence, no action forced) — `stencil_slope` does not assert exact node presence

The g-precision gate (Sec.6) requires "the stencil nodes 0.725/0.730/0.735 must be present in
every table." `stencil_slope()` (line ~677) locates the *nearest* grid node via
`np.argmin(np.abs(h_grid - lo))` rather than asserting an exact match — silently tolerant of a
missing/shifted node rather than raising. `DESIGN_GATE_computability.md` confirmed exact presence
for both production tables' full 41-node grid; I did not re-check all 67 harness CSVs (would
require opening real per-universe likelihood tables beyond header/count, outside this review's
remit) and the harness universes are produced by the same estimator config so are very likely
identical. Flagging only so the design record doesn't claim this gate is enforced in code — it
currently relies on the input tables being well-formed rather than the script verifying it.

---

## 3. Checklist mapping (my own enumeration, cross-checked against `BUILD_RECORD.md` §3)

| draft item | code | status |
|---|---|---|
| Population construction P_dark/K/K_dark/K_hosted/R (§1) | `construct_populations` | ✓ correct, dry-run-confirmed |
| Population sha256 pins, G-3(a)/(b)/(c) | `verify_population_pins`, `verify_g3a_set_identity`, `main` K-equality check | ✓ correct (G-3c satisfied implicitly by the partition check) |
| Harness discovery + manifest + G-3(d) | `discover_harness_universes`, `harness_manifest_hash`, `verify_g3d_resolved_flags` | ✓ correct, dry-run-confirmed; Finding-D-of-computability-gate's own-motion fix (`n_scored` via CSV `event_idx.nunique()`) verified present |
| (I-2D)/(I-1D) term identity, T_B/T_g/T_D (§2.1) | `compute_term_profiles`, `compute_T_D` | ✓ correct |
| G-1 closure (a)-(e) | `gate_g1_closure` | ✓ correct, dry-run-confirmed on real 5-row slice |
| G-2(i) mean_h anchor | *(none)* | ✗ **Finding B — unimplemented** |
| G-2(ii) Δ_K anchor | *(none)* | ✗ **Finding B — unimplemented** |
| G-2(iii) 0 exclusions | *(none — `n_excluded` discarded)* | ✗ **Finding C — unimplemented** |
| G-2(iv) harness pooled sizes | `main` `pooled_sizes`/`HARNESS_POOLED_ANCHORS` | ✓ correct, dry-run-confirmed |
| G-2(v) k=1588 endpoint / G-2(vi) SYNTH | `--dry-run` counts print, `run_synth_check` | ✓ correct (k=1588 visible in `[counts]` line; SYNTH hand-verified §1.4) |
| Term-freeze Λ_t/Δ_t/Δ_F/r/s_t (§2.3) | `term_freeze_lambda`, `delta_t`, `run_term_freeze` | ✓ correct, hand-verified on SYNTH |
| Null draws + CI99 (§2.5) | `null_draw_ci99` | ✓ correct (pool = P_dark\K via `setdiff1d`, same `t̄`, 0.5/99.5 percentiles) |
| Score excess S_t/S_F + Welch SE (§2.4) | `stencil_slope`, `welch_se`, `score_excess` | ✓ correct |
| Harness pooled S_t/S_F + jackknife SE (§4.3) | `harness_pool_score`, `main` combination | ⚠ **Finding E — SE combination is quadrature-of-parts, not joint jackknife of the sum** |
| Production disposition (§5 table 1) | `production_ownership_disposition` | ⚠ **Finding D — two named INTERMEDIATE cases unimplemented** |
| Harness disposition (§5 table 2) | `harness_outcome_disposition` | ✓ correct against all 6 rows (hand-checked, SYNTH-exercised) |
| `--nonadditivity-max` band (computability Finding B fix) | argparse only | ✗ **Finding A — never reaches the disposition function** |
| Concordance ratios / stencil-consistency (§2.3/§2.4, reported-only) | *(none)* | ⚠ **Finding F — not materialized, low severity** |
| K_hosted leave-out (§4.4) | `run_K_hosted_leaveout` | ✓ correct |
| Missing-input INSTRUMENT-DEFECT | `preflight` | ✓ confirmed live (§1.2) |
| T0 import (not re-implementation) | `_import_t0_module` | ✓ confirmed live (§1.3) |
| Launch CLI == Sec.8 token-for-token | `build_argparser` | ✓ (all 23 Sec.8 flags present + `--nonadditivity-max`/`--out`/`--dry-run`/`--synth` as documented additions) |

## 4. What this means for launch

Findings A/B/C are code-level omissions of registered gates/bands, not merely stylistic — a
real-mode run today would exit 0 and produce `--out` without ever checking two of the six G-2
byte-id anchors and without the sixth disposition band actually doing anything. Finding D is a
genuine gap between the draft's own band table and the code for a 2-term system where the missed
cases are not exotic. None of these require reopening the registration's populations, pins, or
science — they are `highz_decomp_reads.py`-only fixes, cheap relative to the node's 0.05 CPU-h
budget, and should go back to the builder (or a second sonnet/medium build pass) before the
disjoint reader runs Sec.8 in real mode. Findings E-G are lower severity / informational and can be
resolved by author ruling (E, F) or left as noted residual risk (G) without blocking re-submission,
but E should at minimum be disclosed to the reader alongside the real-mode `Z_harn` if the code is
not changed.
