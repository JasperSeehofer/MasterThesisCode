# [P3-MKER] zero-compute read (i): fleet-wide kernel-pull vs window-pull census

Session: 2026-08-26. Zero-compute (reads + one <5s python script over already-banked
CSV/JSON only; no pipeline execution, no catalogue load, no cluster jobs).

## 0. Task recap

Quantify, from banked artifacts only, how often a window-passed candidate (k=1.5
eligibility window, symmetric per `mass_filter_sigma="symmetric"`) is nonetheless crushed
by the narrow with-BH-mass kernel — the row-#205 exhibit (seed 900121 event 20,
`L_cat_with_bh = 1.39e-85`, n_sym=2, one candidate at ~19σ_kernel/-176.6 nats despite
sitting inside the ±1.5σ_g window).

## 1. Recon — what's banked

Per `results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825/`
(24 seeds × 2 arms = 48 `{bc,bt}_9001{01..24}_work/` dirs, symmetric window,
production default per PROPOSAL_MASS_FILTER_SYMMETRIC_20260825.md):

- **`bc_*_meta.json` / `bt_*_meta.json`** — run metadata; `diagnostics_csv` key points at
  `seed*/simulations/diagnostics/event_likelihoods.csv`.
- **`*_work/seed*/simulations/diagnostics/event_likelihoods.csv`** — one row per
  *detected* event (not all 200 injected; ~44-89 rows/seed), columns incl.
  `L_cat_no_bh`, `L_cat_with_bh` (the raw in-catalogue numerator SUMS over all
  window-passed candidates, confirmed at `bayesian_statistics.py:5645`/`5644` —
  `self._diagnostic_rows.append({..., "L_cat_no_bh": L_cat_without_bh_mass,
  "L_cat_with_bh": L_cat_with_bh_mass, ...})`). **Event-level aggregate, NOT
  per-candidate.**
- **`*_work/seed*/simulations/prepared_cramer_rao_bounds.csv`** — per-event `M`
  (mu_cond) and `delta_M_delta_M` (sigma_cond² via `Detection.M_uncertainty =
  sqrt(delta_M_delta_M)`, `datamodels/detection.py:143`). Banked, per event.
- **`*_work/seed*/simulations/posteriors_with_bh_mass/h_0_73.json`** — has
  `galaxy_likelihoods[event_idx] = [(catalog_index, [6-tuple]), ...]`, one entry per
  WINDOW-PASSED candidate. Confirmed via `bayesian_statistics.py:4927-4935`
  (`galaxy_likelihoods = list(zip([galaxy.catalog_index for galaxy in
  possible_host_galaxies_with_bh_mass], results_with_bh_mass))`). The 6-tuple order
  (confirmed at `bayesian_statistics.py:6736-6742`, `single_host_likelihood`'s
  `return [...]` when `evaluate_with_bh_mass=True`):
  `[numerator_without_bh_mass, denominator_without_bh_mass,
  numerator_with_bh_mass, denominator_with_bh_mass,
  quad_weight_outside_grid_numerator, quad_weight_outside_grid_denominator]`.
  **This IS per-candidate** (catalog_index + integral outputs) — the count of
  entries for an event equals its n_sym. Verified against the row-#205 exhibit:
  `galaxy_likelihoods["20"]` in `bc_900121_work/.../h_0_73.json` has exactly 2 entries
  (catalog_index 6791158, 6791138) matching the claim's "n_sym = 2". Candidate
  6791138's `numerator_with_bh_mass` = 6.037718947254922e-79 →
  `ln = -180.1` → `sqrt(-2*ln) = 18.98` ≈ the claim's "~19σ_kernel" — reproduces
  the cited exhibit number from the banked JSON alone (no catalogue needed for this
  single check).
- **`m2link_iii_reattribution_check.json`** — per-seed lists `monsters` (events with
  `ln L_cat_with_bh − ln L_cat_no_bh < −50` per A20_REVIEW F11) vs `predicted_sym_zero`
  (events the symmetric-window model predicts n_sym=0 for). 9 seeds have monsters;
  8/9 have `exact_match: true` (monsters == predicted_sym_zero, i.e. ALL monster events
  in that seed are the window-EXCLUSION class, not window-passed); seed 900121 alone has
  `exact_match: false` (`monsters:[20,115]`, `predicted_sym_zero:[115]` — event 20 is the
  RESIDUAL, the one genuine window-passed-but-kernel-crushed instance).
- **`gate_b_20260730/wbhzero_gate_b_scripts/counterfactual_out.json` +
  `counterfactual_symmetric.py`** — a DIFFERENT, EARLIER fleet
  (`p3_b0_work`, asymmetric window, seeds 900101-900112) and ONLY the
  `L_cat_with_bh==0` exact-zero events. Gives per-candidate `pulls` (a combined
  window-boundary distance in units of raw `BH_MASS_ERROR`, NOT the clean
  `|mu_gal-mu_cond|/sigma_gal` the task asks for, and NOT a kernel pull at all —
  see script lines 83-98: `pulls` folds in `SIG*sigma_cond` and the `(1+z)` frame
  conversion already). Does not cover the current fleet or non-zero events, so it
  cannot answer this task's question, but the *script itself* is a template for what
  a proper regeneration would look like (see §4).

## 2. What per-candidate mu_gal / sigma_gal would require

`catalog_index` (e.g. 6791158, 6791138) is a row position in
`GalaxyCatalogueHandler.reduced_galaxy_catalog` — the PRUNED-and-reset-index frame
built at runtime from `darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv`
(1,681,954,844 bytes / 22,641,048 rows on this machine — confirmed via `ls -la`/`wc -l`).
Recovering `BH_MASS`/`BH_MASS_ERROR` at that index needs the SAME prune (mass/z filter +
`reset_index`) the evaluation run applied — not a flat row lookup — so it requires
instantiating `GalaxyCatalogueHandler` and replaying its `__init__` prune. That is a real
compute step (load + filter a 1.68 GB / 22.6M-row file), not a "small one-liner," and the
file is exactly the un-pinned multi-GB-input case CLAUDE.md's dataset-pinning rule flags
(no checksum was verified this session). **Not attempted — out of zero-compute scope.**

Separately, `numerator_with_bh_mass` (the per-candidate with-BH integral value banked in
`galaxy_likelihoods`) is NOT a clean, isolated mass-kernel factor: it's a redshift
quadrature integral (`fixed_quad` over the host-z window) of an integrand that also
carries the host-z kernel, `R_eff_per_mbh(host_M)`, and the detection-probability factor.
Converting its raw magnitude to a "kernel pull in mass-sigma units" the way the task asks
(a clean `|mu_gal-mu_cond|/sigma_kernel`) would require decomposing that integral —
a derivation task, not a data read, even though the *number itself* is banked.

## 3. What IS reconstructible — two results banked/computed this session

### 3a. Exact, banked: the extreme-tier (k≳10) window-passed-crush census

Cross-referencing `m2link_iii_reattribution_check.json` against
`A20_REVIEW_P3_2D_DESIGN_20260825.md` F11/F17 (already-banked, not recomputed here):
over the full 24-seed / 48-arm `p3_2d_fleet_20260825`, the `ln L_cat_with_bh − ln
L_cat_no_bh < −50` nats ("monster") scan flags 18 arm-instances in 9 seeds. Of those,
**17/18 are the window-EXCLUSION class** (`predicted_sym_zero` reproduces `monsters`
exactly — n_sym=0, i.e. NOT window-passed, out of scope for this census) and **exactly
1 unique (seed, event) pair — seed 900121, event 20 — is the genuine window-passed
kernel-crush class**, present identically in both bc/bt arms (2 of the 18 arm-instances).
No other instance of the row-#205 pattern exists in the current fleet at this severity.

### 3b. This-session proxy: event-level Δln census (all 48 `event_likelihoods.csv`, h=0.73)

Script (run once, <5s, reads only the 48 already-banked per-seed/arm CSVs):

```python
import csv, glob, math, re

files = sorted(glob.glob(
    "results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825/"
    "b[ct]_9001??_work/seed*/simulations/diagnostics/event_likelihoods.csv"))

n_events_total = n_with_bh_zero = n_both_positive = 0
dln_list, extreme = [], []
for f in files:
    tag = f.split("/p3_2d_fleet_20260825/")[1].split("/")[0]
    for row in csv.DictReader(open(f)):
        if abs(float(row["h"]) - 0.73) > 1e-9:
            continue
        n_events_total += 1
        L_no, L_wbh = float(row["L_cat_no_bh"]), float(row["L_cat_with_bh"])
        if L_wbh == 0.0:
            n_with_bh_zero += 1
            continue
        if L_no <= 0.0:
            continue
        n_both_positive += 1
        dln = math.log(L_no) - math.log(L_wbh)
        dln_list.append(dln)
        extreme.append((dln, tag, row["event_idx"], L_no, L_wbh))
dln_list.sort()
extreme.sort(reverse=True)
```

`Δln = ln(L_cat_no_bh) − ln(L_cat_with_bh)`, restricted to events where
`L_cat_with_bh > 0` (excludes the 80 window-exclusion-zero rows — a DISTINCT class per
GATE M2-Z/M2-LINK, not this census's target). `k_equiv = sqrt(2·Δln)` — the Gaussian-nats
inversion the CLAIM card itself uses for the row-#205 number (its `-176.6 nats` ↔
`~19σ_kernel` pairing).

**IMPORTANT CAVEAT (why this is a proxy, not the requested census):**
`L_cat_with_bh` is a SUM over all window-passed candidates for that event, so `Δln`
reflects at best the BEST-matched (least-crushed) candidate's compression — a
**lower bound** on any individual candidate's crush, not a per-candidate value. An event
with many candidates could have most of them severely kernel-crushed while one
well-matched candidate keeps the event-level Δln small. It also does not isolate the
mass-kernel factor from the shared host-z-kernel/R_eff/detection-probability factors
(same caveat as §2). Results:

| stat | value |
|---|---|
| n rows (h=0.73, both channels > 0) | 4442 / 4522 |
| n window-exclusion-zero rows (excluded) | 80 |
| median Δln (nats) | 1.235 (k_equiv ≈ 1.57) |
| p90 Δln | 15.09 (k_equiv ≈ 5.49) |
| p99 Δln | 16.57 (k_equiv ≈ 5.76) |
| max Δln | 176.59 (k_equiv ≈ 18.79) |
| frac. Δln > 4.5 nats (k_equiv > 3) | 18.03% (801/4442) |
| frac. Δln > 12.5 nats (k_equiv > 5) | 16.75% (744/4442) |
| frac. Δln > 50 nats (k_equiv > 10) | 0.045% (2/4442) |

Top of the distribution:

```
bt_900121_work event=20 dln=176.59 nats (k_equiv=18.79) L_no=6.838e-09 L_wbh=1.392e-85
bc_900121_work event=20 dln=176.56 nats (k_equiv=18.79) L_no=6.838e-09 L_wbh=1.431e-85
bt_900116_work event=118 dln=19.95 nats (k_equiv=6.32)
bc_900116_work event=118 dln=19.95 nats (k_equiv=6.32)
bt_900113_work event=94  dln=18.52 nats (k_equiv=6.09)
bt_900117_work event=193 dln=18.30 nats (k_equiv=6.05)
bt_900118_work event=88  dln=18.30 nats (k_equiv=6.05)
bt_900103_work event=19  dln=18.08 nats (k_equiv=6.01)
bt_900109_work event=178 dln=17.96 nats (k_equiv=5.99)
bt_900120_work event=109 dln=17.47 nats (k_equiv=5.91)
```

**Corroboration, not conflation:** the max of this independent event-level scan is
900121:20 at `176.59/176.56` nats — matching the CLAIM card's own `−176.6 nats` figure
and confirming it as the single most extreme instance fleet-wide by this metric too (both
arms). This is a useful cross-check but the 0.045% (k_equiv>10) figure here is NOT the
same count as the already-banked "18 arm-instances/9 seeds" −50-nats "monster" scan in
§3a — the two use different definitions (this scan requires `L_cat_with_bh>0`
strictly and is event-level Δln; the banked monster scan's exact inclusion rule for
`L_cat_with_bh==0` "accepted events" was not re-derived here) — reported separately,
not merged.

**Second caveat on 3b's mid-distribution numbers (18.0% at k_equiv>3):** because Δln is a
lower bound dominated by the best candidate, this 18% almost certainly UNDER-counts the
true fraction of individual window-passed CANDIDATES with kernel-pull > 3 — multi-candidate
events could carry many more crushed candidates hidden under one good match. No fleet-wide
correction for this is possible without the per-candidate reconstruction of §2.

## 4. What would be needed for the exact requested census

A new script (same recipe as the banked `counterfactual_symmetric.py`, extended and
re-scoped) that:
1. Instantiates `GalaxyCatalogueHandler` once (loads + prunes the 1.68 GB catalogue —
   pin its checksum per CLAUDE.md's dataset-pinning rule first).
2. For every seed/arm/event, replays
   `get_possible_hosts_from_ball_tree(sigma_multiplier=1.5,
   mass_filter_sigma="symmetric")` to get the window-passed candidate set with
   `BH_MASS`/`BH_MASS_ERROR` per candidate (mu_gal, sigma_gal).
3. Reads `M`/`sqrt(delta_M_delta_M)` per event from the banked
   `prepared_cramer_rao_bounds.csv` (mu_cond, sigma_cond — already available, no compute
   needed for this half).
4. Computes `window_pull = |mu_gal − mu_cond·(1+z)| / sigma_gal` and
   `kernel_pull = |mu_gal − mu_cond·(1+z)| / sigma_cond` (or the decomposed mass-only
   factor pulled out of `numerator_with_bh_mass`, cross-checked against this formula) for
   every window-passed candidate, fleet-wide.
5. Tabulates the requested median/p90/p99/max and fraction-exceeding-{3,5,10} directly.

This is genuine (if modest) compute — a catalogue load + ball-tree query per event — and
is explicitly out of scope for this zero-compute pass.
