# v-falsifier-ii-classG — RECOST RECORD (zero-compute, banked evidence only)

Research Graph 1, Branch E, node `v-falsifier-ii-classG` / checkpoint `k-falsifier-ii-fleet`.
Written 2026-09-02 in response to `LAUNCH_RECORD.md` (this directory), which SKIPPED the launch
because the registered anchor (208–286 CPU-h) exceeds the row #290 hard cap (60 CPU-h) and the
graph proposal's "40–60 CPU-h" figure has no located derivation. This record derives a SOURCED
number from banked evidence. **No code was edited, no commit made, no cluster access, no new
compute.**

---

## 1. Empirical CPU-h/task anchor

**8.6667 CPU-h/task**, from TWO independent completed SLURM arrays running the exact instrument
(`p3_2d_fleet.py --stage fleet`, both arms `bc`+`bt` per task, `--cpus-per-task=16`,
partition `cpu_il`):

| job | tasks | seeds | result | source |
|---|---:|---|---|---|
| 6723958 | 24 | 900101–900124 | 24/24 COMPLETED, **~32.5 min/task** | `P3_2D_REPAIR_READOUT_20260828.md:43` |
| 6730213 (PA-2DR-15 extension) | 9 | 900125–900133 | 9/9 COMPLETED, **~32.5 min/task** (same figure, independently reported) | `P3_2D_REPAIR_READOUT_20260828.md:148` |

32.5 min × 16 cpus / 60 = **8.6667 CPU-h/task**, replicated identically across a 24-task batch
and an independent 9-task batch (33 tasks total, both arms, 66 arm-seed pairs) — this is not a
single-sample estimate. No finer per-task granularity (individual task elapsed times / sacct
records) is present in the local repo; the cluster workspace that held the raw provenance/sacct
files was not located under `results/campaign51_20260728/realistic_20260729/` (search for
provenance_*.json under the repair out-root and for the job IDs elsewhere in-repo returned
nothing beyond the two averaged figures above). **The empirical distribution available is
therefore two independent batch-mean samples at the same value (~32.5 min/task), not a
per-task histogram — disclosed as a limitation, not upgraded to more than it is.**

This is exactly the anchor the proposal itself cites: `PROPOSAL_2D_TWIN_ADOPTION_20260829.md:274`
— "~8.67 CPU-h/task ... from the readout's ~32.5 min/task at 16 cpus." The registered anchor and
the empirical anchor derived here from the primary readout are **the same number**; the proposal
did not need re-sourcing, only re-confirming.

### 1.1 Is the anchor stale?

**No — checked directly, not assumed.** `git log d04d9dc9..HEAD` (d04d9dc9 = the commit the
repair-readout jobs ran at) touches neither `p3_2d_fleet.py` nor the 2D-branch draw-law lines of
`darksiren_emri/validation/correspondence_1d.py` that `stage_fleet` exercises (`:2107`,
`:1687-1696`, per `PHYSICS_CHANGE_SBARPHI_20260827.md` §9.1's "files that would change under
Option A′" list). Eight `[PHYSICS]` commits landed on `correspondence_1d.py` since d04d9dc9
(5e7fda16, 7e1ed96f, 62f7d61e, 6c6f2a63, d4765539, 901653a1, d40fe5c8, ece1bd1b) but every one is
a **byte-identical-default instrument flag** on the 1D channel or on unrelated plumbing — none
touch the 2D `catalogue_selected_2d` draw path. **Cross-scale references, explicitly NOT
comparable, stated for context only:** wave-3 HEAD-readout model ~6.5 min/task (a different
computation — production-scale `--evaluate` over 1588 events at fixed h, not a 200-event mirror-
venue fleet generation) and T5 Arm S ~5 CPU-h per 4-task k-set (also a different harness/venue).
Neither is substitutable for the class-G fleet anchor.

**Conclusion: 8.6667 CPU-h/task is CURRENT, not stale**, and is the number to price the falsifier
against.

Disclosed but immaterial to cost: `PREREGISTRATION_P3_2D_REPAIR_20260827.md:675/1014` states the
repair "changes RNG consumption" so the fresh fleet, though reusing the same seed labels,
performs genuinely new draws — i.e. the measured wall-time already reflects a real generation
pass, not a cache hit, and the same will be true of an Option A′ (rung-1) re-run.

---

## 2. What the falsifier (ii) design itself requires

Quoted verbatim, `PROPOSAL_2D_TWIN_ADOPTION_20260829.md` §6.1(ii):

> "(ii) Identity residual is venue-side (the S̄_φ double-weight as the registered falsifier). On
> the class-G venue with rung 1 repaired in the Option A′ form (harness-only gate; fleet re-run
> ~8.67 CPU-h/task × 24–33 tasks ≈ 208–286 CPU-h ...), the registered v2.9 conditional prediction
> must land: LHS2(bt) = 0.00740040 ± 0.00024951, band ±3σ_comb (two-sided), AND the G4
> arm-coherence ratio must stay inside its registered interval [0.8613, 0.8675]. ... (Null: paired
> deterministic fleet re-score; the band's false-fail rate under the exact null is set by the
> frozen planning SEMs, **already realized below planning at 33 seeds, §7 of the readout**.)"

The design's own minimum-power statement is not abstract — it was already measured on the
adjacent (rung-2/3) repair fleet, `P3_2D_REPAIR_READOUT_20260828.md` §3/§7:

- At **24 seeds**: P2 (`LHS2_D1only`) realized SEM was **16.7% above** the frozen planning SEM;
  P3 (paired ratio) realized SEM was **4.4% above** planning. Disposition: **UNDERPOWERED**
  (chair verdict §4, author-ratified 2026-08-28, "all ratified").
- At **33 seeds** (the +9 seed PA-2DR-15 extension): both SEMs fell **below** planning for every
  read (P1/P2/P3/P4 all "below ✓"). Disposition: **CONFIRMED** (§7, author-ratified).
- §7 explicitly states this was "the single pre-committed extension, decided post-data ...
  no further extension may run" — i.e. 33 seeds is the design's own demonstrated floor for
  meeting its planning-SEM bands, not an arbitrary choice.

Falsifier (ii) reuses this exact "frozen planning SEMs" machinery for its LHS2/G4 bands. **24
seeds is a configuration the design's own adjacent measurement already showed to be
UNDERPOWERED on the same statistic family (LHS2/paired-ratio SEMs); 33 seeds is the smallest
configuration shown to meet the design's planning bands.** This is the most direct, sourced
answer available to "does the design specify a minimum fleet size" — it does not state one in
the abstract, but it has already empirically located one for the sibling measurement that
shares its band machinery.

Caveat, stated plainly: the 24-vs-33 power result was measured on the **rung-2/3 repair fleet**
(commit `d04d9dc9`, pre-Option-A′), not on a rung-1-repaired fleet — falsifier (ii) has never
itself been run. There is no banked evidence that the SEM behavior transfers exactly, only that
it is the closest and only same-instrument, same-venue, same-band-machinery precedent available.

---

## 3. Author option table (all totals at the 8.6667 CPU-h/task empirical anchor)

| option | config | CPU-h (empirical anchor) | power status | source |
|---|---|---:|---|---|
| **(a) Full registered fleet, 24 tasks (design minimum config, no power margin)** | 24 seeds, both arms/task | **208.0 CPU-h** | design's own adjacent measurement: **UNDERPOWERED** on P2 (+16.7% SEM) and P3 (+4.4% SEM) at 24 seeds | §1 anchor × 24; power precedent §2 |
| **(a′) Full registered fleet, 33 tasks (design's own demonstrated power floor)** | 33 seeds, both arms/task | **286.0 CPU-h** | precedent: CONFIRMED-level power (all SEMs below planning) | §1 anchor × 33; §2 |
| **(b) Largest subset fitting the 60 CPU-h hard cap** | 6 seeds | **52.0 CPU-h** (7 seeds = 60.67, over cap) | **UNSOURCED as adequate** — no banked measurement exists at n=6; the only two banked power points (24, 33) are both ≥24 seeds and 24 was already shown insufficient, so a 6-seed fleet is expected to be MORE underpowered, not less (SEM ∝ 1/√n: 6 seeds ≈ 2× the 24-seed SEM ≈ 33% above planning on P2, extrapolated, NOT measured) | §1 anchor; extrapolation flagged as such, not measured |
| **(c) Cheaper design-sanctioned equivalent (score-only / banked-cell reuse)** | — | **NONE FOUND** | the design's own text (`PREREGISTRATION_P3_2D_REPAIR_20260827.md:675/1014`, restated for Option A′ scope in `PHYSICS_CHANGE_SBARPHI_20260827.md` §9.1) states a draw-law repair "changes RNG consumption" so the fleet "must not be reused" — `stage_gates`/`stage_lhs2d` are cheap post-processing (`p3_2d_fleet.py` docstring: "this whole stage is post-processing of an already-written diagnostics CSV") but require a freshly generated `stage_fleet` CSV, which is the expensive step being priced. Some quantities ARE zero-compute reuse (e.g. RHS2, `PREREGISTRATION_P3_2D_REPAIR_20260827.md:71`) but RHS2 is not among falsifier (ii)'s target statistics (LHS2, G4) | as cited |

**Every row's total exceeds the row #290 hard cap of 60 CPU-h except (b), and (b) has no power
justification — it is a cap-fitting exercise, not a design-sanctioned configuration.** The
"40–60 CPU-h" figure carried in the graph proposal (`RESEARCH_GRAPH_1_PROPOSAL_20260901.md`
§1.5) and in `tree2_20260830/TREE2_CHARTER_20260830.md` (T4 row) /
`TREE2_SYNTHESIS_DOCKET_20260830.md` (items 5, 4) remains **UNSOURCED**: all three occurrences
are the identical unattributed phrase "approx 40-60 CPU-h cluster (chair recost from 208-286)"
with no task count, no per-task anchor, and no narrowed-scope rationale given anywhere in the
accessible record — re-confirmed by direct grep of both tree-2 documents in this session,
independently of the prior launcher's search.

---

## 4. What this hands the author

1. The 8.6667 CPU-h/task anchor is **not stale and not improvable** from banked evidence — it is
   twice-replicated, current-commit, same-harness. Any smaller CPU-h/task number would need a
   NEW measurement (compute), not a recost.
2. The design's own band machinery has already shown **24 seeds insufficient** for the same
   statistic family (LHS2/paired-ratio SEMs) and **33 seeds sufficient**, on the sibling
   rung-2/3 fleet. There is no banked basis for a fleet smaller than 24 meeting the falsifier's
   bands; extrapolating downward from the measured SEM-vs-n behavior makes 6 seeds (the
   cap-fitting size) look WORSE-powered than the already-rejected 24-seed case, not better.
3. **No configuration meeting the design's own demonstrated power floor fits under 60 CPU-h at
   the sourced anchor** (286.0 CPU-h at 33 seeds vs the 60 CPU-h cap — a 4.77× overage; even the
   underpowered 24-seed floor is 208.0 CPU-h, 3.5× the cap).
4. The three occurrences of "40-60 CPU-h chair recost" carry no arithmetic anywhere in the
   accessible record and should not be used to authorize a launch; if the author wants to price
   this against a $60 CPU-h cap, the two live options are: raise the cap to accommodate the
   design's own 208–286 CPU-h registered cost ([RULE], since row #290 decision row 7 caps it at
   60 hard), or authorize a smaller, explicitly UNDERPOWERED fleet with the power loss stated
   up front (not the silent 24-seed UNDERPOWERED repeat, and not an unmeasured 6-seed guess).

---

*Builder/runner independence: this record ran no registered measurement and submitted no job;
every number above is either quoted verbatim from a banked, author-ratified document or an
arithmetic evaluation of quoted numbers, each cited to file:line.*
