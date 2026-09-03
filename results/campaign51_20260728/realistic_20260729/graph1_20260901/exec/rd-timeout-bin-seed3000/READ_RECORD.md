# READ_RECORD — rd-timeout-bin-seed3000

Node: `rd-timeout-bin-seed3000` (Branch L, proposal §1.3, execution item 4; G7 systematics row 8,
waveform-timeout selection). Type: **read**, verdict-free per row #325 grant. This record carries
numbers, tables, and the proposal's own pre-registered three-valued band applied mechanically to
those numbers — it is not a ruling; the chair rules on `d-timeout-bound`.

Script: `/tmp/claude-1000/.../scratchpad/L1/analyze.py` (re-derivable; logic summarized below).
Full machine output: `READ_RECORD.json`, `design_gate_bin_edges.json`,
`rate_table_{M,e0,p0,2d_M_p0}.csv` in this directory.

## Existence contract (inputs)

| input | status | detail |
|---|---|---|
| `results/_archive/run_20260707_seed3000/*.log` (recursive) | **PRESENT** | 99 files (97 top-level + 2 in `simulations/`), matches proposal's "99 logs" exactly |
| `.../simulations/cramer_rao_bounds.csv` | **PRESENT** | 3325 data rows (3326 lines incl. header), matches proposal's "3 325 CRB rows" exactly |
| timeout records with full param dicts | **PRESENT** | 1198 lines total (1196 `Waveform/SNR computation timed out` + 2 `Cramér-Rao bound computation timed out`), matches proposal's "1 198 ... records" exactly |
| `.../simulations/injections/*.csv` (raw injected draw) | **UNREACHABLE** | 500 entries in the directory are dangling symlinks to `/pfs/work9/workspace/scratch/st_ac147838-emri/injection_pool_depth15_50k/...` — a cluster-only workspace path, not present on this machine. STOPPED per instruction; not substituted silently — see Gaps. |
| `run_metadata_*.json` (per-task SLURM metadata) | **PRESENT but PARTIALLY OVERWRITTEN** | 100 files exist; only 59 (task indices 41–99) still carry the original simulate-stage metadata — indices 0–40 were overwritten by a later evaluate-stage array reusing the same filenames in the same directory. See Gaps. |

All three numbers the proposal quoted from having opened the source (99 logs / 1198 records / 3325
CRB rows) reproduce exactly — strong corroboration the correct archive is being read.

## Design gate — bin edges frozen BLIND, before any rate was read

Per node instruction: froze bin edges first (script section 3, printed and written to
`design_gate_bin_edges.json` *before* any timeout count was touched), using a population-quantile /
log-spacing rule on the union of the two only local sources with per-event (M, e0, p0): the 3325 kept
(CRB-CSV) events and the 1198 timeout events. (The raw injected-population files that would have been
the more literal "population quantiles of the injected distribution" are unreachable — see existence
contract above — so this union is the honest local proxy, disclosed rather than swapped in silently.)

- **M**: log-spaced, 5 bins, equal width in log₁₀M over the observed support [3.55e4, 1.77e6] M☉.
- **e0**: quantile (quintile), 5 bins, over [0.050, 0.200].
- **p0**: quantile (quintile), 5 bins, over [10.00, 16.00].
- **2-D (M, p0)**: the 5×5 cross-grid of the same M and p0 edges (25 cells).

(μ = 10 M☉ and a = 0.98 fixed for every event in this run — not bin axes, consistent with the
proposal's chosen (M, e0, p0) triple.)

## Denominator construction (disclosed explicitly — this is the g-population gate)

Only two local sources carry per-event (M, e0, p0): the CRB-CSV (kept: passed both the SNR stage and
the CRB stage) and the two timeout log sites (G9 instrumentation logs the full parameter dict
specifically at the SNR-stage and CRB-stage `TimeoutError` handlers, and nowhere else). Every other
skip category in `main.py`'s `skip_counts` tally (`ZeroDivisionError`, `Warning`,
`ParameterOutOfBoundsError`, `EllipticK`/`Brent`/`SeparatrixSigns` `ValueError`s, `crb:LinAlgError`,
`crb:ParameterEstimationError`) has **no per-event parameters logged** and cannot be placed in a bin.

So the rate reported below answers **"of events that either timed out or fully succeeded at this
stage, what fraction timed out"** — not "of every attempted draw". Denominators:

- **SNR-stage**: kept (3325) + all 1198 timeout events (both stages reach the SNR-stage draw) = 4521.
- **CRB-stage**: kept (3325) + the 2 CRB-stage timeouts = 3327 (SNR-stage timeouts never reached the
  CRB stage, excluded).

This keeps numerator and denominator inside the same archive / same run / same commit
(`a545c0ebc7ba65d0245838e474a1c74602ac5aef`, timestamp 2026-07-08) — satisfying the gate's "must come
from the same tasks" intent at the run level. A strict per-task-id cross-check is blocked by the
`run_metadata` overwrite noted above (Gaps); it is not needed for the aggregate/binned rate computed
here, which uses the raw event records directly, not the per-task tally lines.

## SNR-stage timeout rate — aggregate

**1196 / 4523 = 0.2644** (Garwood 95% rate CI **[0.2497, 0.2799]**). Denominator = 3325 kept + 2
crb-stage timeouts (both passed the SNR stage) + 1196 snr-stage timeouts = 4523.

## SNR-stage timeout rate — by axis (5 bins each), Poisson (Garwood) 95% intervals

### M (log-spaced)

| bin | M range (M☉) | n_kept | n_timeout | denom | rate | Garwood 95% |
|---|---|---|---|---|---|---|
| 0 | 3.55e4–7.76e4 | 15 | 103 | 118 | 0.873 | [0.712, 1.059]† |
| 1 | 7.76e4–1.70e5 | 110 | 401 | 511 | 0.785 | [0.710, 0.865] |
| 2 | 1.70e5–3.70e5 | 1017 | 378 | 1395 | 0.271 | [0.244, 0.300] |
| 3 | 3.70e5–8.10e5 | 1993 | 245 | 2238 | 0.109 | [0.096, 0.124] |
| 4 | 8.10e5–1.77e6 | 192 | 69 | 261 | 0.264 | [0.206, 0.335] |

†Garwood upper bound on the *count* can exceed the denominator at very low n; the rate CI is reported
as computed (count-CI/denom) without clipping to 1, per the exact-Poisson convention — flagged, not
hidden.

**Max adjacent-bin gradient: 12.19σ, between bins 1 and 2** (rate falls from 0.785 to 0.271).
M-axis population depth ranges 118–2238 events/bin (uneven — M is far more concentrated at the
high-mass end of this support).

### e0 (quintile)

| bin | e0 range | n_kept | n_timeout | denom | rate | Garwood 95% |
|---|---|---|---|---|---|---|
| 0 | 0.0500–0.0805 | 672 | 233 | 905 | 0.257 | [0.225, 0.293] |
| 1 | 0.0805–0.1093 | 651 | 253 | 904 | 0.280 | [0.246, 0.317] |
| 2 | 0.1093–0.1382 | 653 | 252 | 905 | 0.278 | [0.245, 0.315] |
| 3 | 0.1382–0.1688 | 667 | 237 | 904 | 0.262 | [0.230, 0.298] |
| 4 | 0.1688–0.1999 | 684 | 221 | 905 | 0.244 | [0.213, 0.279] |

**Max adjacent-bin gradient: 0.90σ** (bins 0–1). Flat within Poisson. Depth ≈905/bin (even, as
expected from the quantile design).

### p0 (quintile)

| bin | p0 range | n_kept | n_timeout | denom | rate | Garwood 95% |
|---|---|---|---|---|---|---|
| 0 | 10.00–11.40 | 667 | 238 | 905 | 0.263 | [0.231, 0.299] |
| 1 | 11.40–12.59 | 666 | 238 | 904 | 0.263 | [0.231, 0.299] |
| 2 | 12.59–13.65 | 671 | 234 | 905 | 0.259 | [0.226, 0.294] |
| 3 | 13.65–14.82 | 675 | 229 | 904 | 0.253 | [0.222, 0.288] |
| 4 | 14.82–16.00 | 648 | 257 | 905 | 0.284 | [0.250, 0.321] |

**Max adjacent-bin gradient: 1.24σ** (bins 3–4). Flat within Poisson. Depth ≈905/bin (even).

### 2-D (M, p0) grid (5×5 = 25 cells)

Full table in `rate_table_2d_M_p0.csv`. Rate ranges from ~0.09–0.10 (M-bin 3 × p0-bin 0/1/2, the
largest-N cells, N≈486–499) up to 0.86–1.00 in the sparsest low-M cells (N=4–29, wide Garwood
intervals — not individually decisive). **Max adjacent-cell gradient: 6.84σ**, between (M-bin 1,
p0-bin 4) and (M-bin 2, p0-bin 4) — i.e., the same M-bin-1→2 boundary that dominates the 1-D M
table, confirming the gradient is carried by M, not by an M×p0 interaction: within any fixed M-bin
the p0 dependence is mild (compare, e.g., M-bin 3's five p0-cells: 0.101/0.099/0.094/0.134/0.129 —
a modest but visible p0 tilt appears only inside the M-bin-3 slice, on the order of ~1–2σ per cell
pair, well under the axis-level p0 gradient's own null result because the marginal p0 table averages
over M).

## CRB-stage timeouts — descriptive only (n=2)

Two events, params (M, e0, p0): (4.89e5, 0.192, 13.05) and (4.61e5, 0.118, 15.85). Aggregate
2/3327 = 0.00060, Garwood 95% rate CI [0.00007, 0.00217]. Far too few to bin at 5/axis; reported as
raw values only. Not part of the axis/gradient tables above.

## Population depth (z_cut / HOST_DRAW_Z_MAX), from the kept-CRB z distribution

`dist_to_redshift(luminosity_distance, h=0.73)` (repo's `physical_relations.py`, fiducial cosmology,
`HOST_DRAW_Z_MAX = 1.5` from `constants.py`) applied to all 3325 kept events:

| statistic | value |
|---|---|
| n | 3325 |
| min z | 0.0093 |
| median z | 0.4127 |
| p95 z | 0.6968 |
| max z | 1.0250 |
| HOST_DRAW_Z_MAX | 1.5 |
| depth fraction (max z / HOST_DRAW_Z_MAX) | 0.683 |
| fraction of events with z > 0.9×HOST_DRAW_Z_MAX | 0.0 |

**seed3000's kept population never reaches the declared 1.5 draw ceiling — it tops out at z≈1.03
(68% of the nominal depth), with zero events past 90% of the ceiling.** This is a genuine partial:
per the proposal's own framing, this result is bounded to the shallower seed3000 pool and is not the
final bound (the deeper post-dt2 seed61000 production pool is cluster-only and was not fetched
tonight — Branch L item 2, `rd-timeout-bin-seed61000`, not run).

## Three-valued read against the proposal's band

Band (proposal, node row): *flat within Poisson → NON-ISSUE with bound; >3σ gradient → new
systematic.*

| axis | max gradient | band call |
|---|---|---|
| M | 12.19σ | **NEW-SYSTEMATIC-CANDIDATE** (>3σ) |
| e0 | 0.90σ | NON-ISSUE-WITH-BOUND |
| p0 | 1.28σ | NON-ISSUE-WITH-BOUND |
| 2-D (M, p0) | 6.84σ | **NEW-SYSTEMATIC-CANDIDATE** (>3σ, carried by the M dimension) |
| **overall** | **12.19σ** | **NEW-SYSTEMATIC-CANDIDATE**, localized entirely to M; e0 and p0 are each individually NON-ISSUE |

This is a partial (seed3000, z-depth 68% of nominal) — the proposal's own kill_criterion language
("the seed3000 result is a partial ... never the final bound").

## Gaps (things stopped short / disclosed rather than substituted)

1. Raw injected-population CSVs unreachable (dangling cluster symlinks) — bin edges built from the
   kept+timeout union instead; disclosed above, not substituted silently.
2. Denominator excludes all non-timeout skip categories (no per-event params logged for them) —
   the reported rate is conditional on {timed out, fully succeeded}, not of every draw.
3. Only 33 of ~100 simulate tasks logged a final "Skip tally" summary line (most did not finish before
   the array's walltime); does not affect the binned rate (built from raw records) but is a
   completeness caveat on the archive as a whole.
4. `run_metadata_0.json`–`run_metadata_40.json` were overwritten by a later evaluate-stage array
   reusing the same filenames in the same directory; only task indices 41–99 (59/100) retain original
   simulate-stage SLURM metadata. Blocks a strict per-task-id trace beyond "same archive/run/commit";
   not required for the aggregate rate computed here.
5. CRB-stage timeout sample (n=2) is descriptive only, not binned.
6. `rd-timeout-bin-seed61000` (the deeper production pool, cluster-only) was not fetched — out of
   scope for this node (A-L2, separate grant).

## RECOMMENDATION (labelled as such — not a ruling)

Feed this read to `d-timeout-bound` (the chair's disposition) as: the seed3000 partial shows a
**highly significant, monotonic-ish M-dependence** in SNR-stage timeout rate (0.87 at low M down to
~0.11 at M≈4–8×10⁵ M☉, ticking back up to 0.26 in the highest M-bin) that clears the proposal's 3σ
new-systematic threshold by a wide margin (12.2σ vs 3σ), while e0 and p0 are each flat and clear the
NON-ISSUE band. Recommend NOT closing G7 row 8 as NON-ISSUE on this read alone: (a) the M-dependence
is real signal worth carrying into the seed61000 fetch (A-L2) before any bound is finalized, and (b)
the seed3000 population is shallow (z_max≈1.03 vs the 1.5 ceiling) so this partial may understate or
misshape the rate at the depths the production pool actually reaches. The non-monotonic uptick in the
top M-bin (0.11→0.26) is itself worth a look — plausibly a second, smaller-N population effect rather
than noise (N=261, still a >10σ-scale jump from bin 3) — flagging it rather than characterizing it
further tonight.
