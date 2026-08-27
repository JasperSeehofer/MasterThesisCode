# MEASUREMENT REGISTRATION — production H₀ readout at HEAD (iiib / joint_r1)

`[OPUS-ORCH 2026-08-27]`

**Class:** STATE MEASUREMENT (not a hypothesis test). No stage-2 pre-registration apparatus;
interpretation is nevertheless fixed here, **before** the data exists, because the failure mode
this campaign spends most of its time guarding against is reading a mechanism out of a surprising
number after the fact.

**Status:** REGISTERED PRE-DATA · NOT SUBMITTED · not committed (working tree).
**Author decisions required before submission:** §10.

---

## 0. Why this exists

An adjudicated synthesis (2026-08-27) established that **no full-campaign H₀ readout has been taken
under the production physics currently in the tree.** The headline `+0.077` (campaign #53) is stale
and has since flipped sign. Ledger row #132's offsets — the most recent production baseline of
record — predate three landed `[PHYSICS]` estimator changes and were themselves taken on a
deliberately non-default completion cell. Two of the three changes act directly on the with-BH
catalogue leg to which the 2D offset is attributed, and one of them (`cf4f8a2a`) removed a defect
that was zeroing **43.32 %** of that leg's h = 0.73 numerator rows in iiib, with its **H₀-space
magnitude never measured** (ledger row #201).

The venue-level moves that *have* been measured point in **opposite directions** on different mirror
venues (row #171: −0.004309 ± 0.000736, 0/12 positive; row #173: +0.029068 ± 0.005088, 12/12
positive), and the third change has no H₀-space measurement at all. They cannot be composed into a
forecast. **The bias must be measured, not predicted.**

---

## 1. What is measured

One `--evaluate` sweep of the **41-node production h grid** per venue, both channels, scored against
injected truth **h_true = 0.73** (`constants.H`; `correspondence_1d.py:358` `H_TRUE`).

### 1.1 Venues (identical event set, different read-time catalogue)

| | iiib | joint_r1 |
|---|---|---|
| CRB / event set | `run_20260729_seed61000/simulations/prepared_cramer_rao_bounds.csv` | **same file** (symlink) |
| CRB md5 | `9a1f2a14384a9281c97ca3be312ddaab` (verified cluster + local, this session) | identical |
| Catalogue | baseline TRUE reduced GLADE+ (`--observed_catalogue` unset) | `observed_catalogue_seed900001.csv` (photo-z + mass scattered, σ_scale 1.0) |
| Catalogue pin | md5 `c52c13b5cab61f6b3f04bbe202550969` = `REDUCED_CATALOGUE_MD5` (`correspondence_1d.py:311`); verified on cluster this session | sidecar `observed_csv_sha256` `e8f7ab31…` (`.meta.json`, read this session) |
| Events scored | 1588 | 1588 |

Both venues share one h_true = 0.73 mirror-universe simulation. **The only difference is the
catalogue fed to `--evaluate`.** This was re-verified against the ledger this session (§2.2), not
assumed: comparing against a different venue would silently invalidate the whole measurement.

### 1.2 Channels

- **2D / with-BH** — `combined_with_bh` (host mass channel; the channel row #132's headline offsets
  refer to).
- **1D / without-BH** — `combined_no_bh` (the railed channel).

### 1.3 Statistics (T0 convention — frozen here, verbatim from the row #132 scorer)

Reference implementation: `results/prod2d_closure_20260818/bscale_counterfactual_exploratory.py:23-30`
("uniform prior, gradient-trapezoid weights over the h grid (P7-2a), sum log likelihood over
events (canonical raw Σ log L)"), reading `simulations/diagnostics/event_likelihoods.csv`.

For grid nodes `h_k` (k = 0…40) and per-event channel likelihood `L_i(h_k)`:

```
ln P(h_k) = Σ_i ln L_i(h_k)                          # uniform prior, raw Σ log L
w_k       = np.gradient(h)_k                          # gradient-trapezoid weights (P7-2a)
p_k       ∝ exp(ln P(h_k) − max_k ln P) · w_k ,  Σ p_k = 1
mean_h    = Σ p_k h_k
sigma_h   = sqrt( Σ p_k h_k² − mean_h² )
MAP       = h_argmax(ln P)                            # DISCRETE grid argmax (row #132's convention)
offset    = mean_h − 0.73
pull      = (0.73 − mean_h) / sigma_h
C_q       = 1 iff h_true ∈ HPD_q(p),  q ∈ {0.68, 0.90}
```

**Deliverables per venue × per channel:** `MAP`, `mean_h`, `sigma_h`, `offset`, `pull`, `C68`, `C90`.

**REPORTED-ONLY companions** (never band-bearing): `continuous_map` from
`posterior_combination.compute_canonical_combined_posterior` (parabolic sub-grid refinement,
`posterior_combination.py:764-800`), and the combine-path `combined_posterior.json` as an
independent cross-check of the CSV scorer.

### 1.4 The h grid

`H_GRID_41` (`correspondence_1d.py:351-356`) — 41 nodes on [0.600, 0.860]; 0.005 step across the
peak 0.655–0.785, 0.010 step in the wings. Verified textually identical to
`cluster/evaluate.sbatch:56`'s `H_VALUES` array, so any `evaluate.sbatch` array job sweeps exactly
the support row #132 swept. No grid-mismatch is possible unless `H_VALUES_OVERRIDE` is exported
(`evaluate.sbatch:58-69`, unset by default). **`H_VALUES_OVERRIDE` must remain unset.**

---

## 2. The comparand — ledger row #132, stated exactly

### 2.1 Row #132 of record

> **Ledger row #132** — 2026-08-19 — `gate_b_20260730/BIAS_HISTORY_LEDGER.md:1246`
> Source prereg: `results/prod2d_closure_20260818/PREREG_POSTFIX_BASELINE.md` (VERDICT, appended
> 2026-08-19). SLURM jobs **6372475 / 6372476**, 82/82 COMPLETED.
> Commit **`e65d263c406a461570cf07132301105d51642b47`** (both `run_metadata_0.json`).
> Run dirs of record: `$WS/run_20260819_postfix_baseline_{iiib,joint_r1}` (3.4 G / 3.3 G, 41/41
> posteriors present — verified live this session).

### 2.2 Independently re-derived this session (mandate: re-derive the decisive parts)

Recomputed from the banked per-event CSVs
`results/prod2d_closure_20260818/postfix_baseline/{iiib,joint_r1}/event_likelihoods.csv`
(65 109 lines = 1 header + 1588 × 41, both venues) with an independent implementation of §1.3:

| venue | channel | n_used | mean_h | offset | sigma_h | MAP | pull | C68 | C90 |
|---|---|---:|---:|---:|---:|---:|---:|:--:|:--:|
| iiib | **2D** | 1588 | 0.6771 | **−0.0529** | 0.0239 | 0.675 | +2.218 | 0 | 0 |
| iiib | 1D | 1588 | 0.6010 | −0.1290 | 0.0033 | 0.600 | +39.17 | 0 | 0 |
| joint_r1 | **2D** | 1588 | 0.6788 | **−0.0512** | 0.0225 | 0.675 | +2.281 | 0 | 0 |
| joint_r1 | 1D | 1588 | 0.6020 | −0.1280 | 0.0046 | 0.600 | +27.67 | 0 | 0 |

Every ledger-quoted figure reproduced to 4 decimals (mean_h 0.6771/0.6788, offsets −0.0529/−0.0512,
σ_h(2D) 0.0239/0.0225, MAP 0.675/0.675, 1D 0.6010/0.6020). **Venue identification confirmed:
iiib ↦ −0.0529, joint_r1 ↦ −0.0512** (`PREREG_POSTFIX_BASELINE.md:39-40`, explicit venue labels).
Zero non-positive likelihood cells in either banked CSV (no floor/sentinel contamination).

**New to this registration** (not quoted in row #132, derived here): the 1D widths
σ_h = 0.0033 / 0.0046, the pulls, and coverage — **C68 = C90 = 0 in all four cells.**

### 2.3 The row #132 configuration, verbatim

From `run_metadata_0.json`, both venues (differences between the two are only
`observed_catalogue` and `working_directory`):

```
normalization_mode              = absolute_marginal      # explicit
host_z_kernel                   = volume_deconv          # explicit
selection_in_completion_numerator = off                  # explicit  ← see §3.2
completion_b_scale              = derived
catalogue_mass_overlap          = production
host_mass_kernel                = auto
pdet_z_resolved = True   pdet_wbh_z_resolved = False
pdet_dl_bins = 60   pdet_mass_bins = 40   pdet_estimator = local_linear
seed = 777000 + task_id (EVAL_SEED 777000)
```

---

## 3. What landed between row #132 and HEAD

Local HEAD `bbfdd2e0`, branch `fix/p32d-classg-venue-repair`.
`git status --short -- darksiren_emri/` **clean**.
`git rev-parse HEAD:darksiren_emri` = `7bfff25dbdb95383304b2cef576edde17d957242`.
Only source difference vs `main` is `validation/correspondence_1d.py` (+51/−6, harness/scorer) —
**the production `--evaluate` path is byte-identical on `main` and HEAD.**

Seven `[PHYSICS]` commits landed on `darksiren_emri/` since `e65d263c` (all verified ancestors of
HEAD via `git merge-base --is-ancestor`).

### 3.1 Three change what a bare production `--evaluate` computes

| # | commit | date | change | reachable from CLI? |
|---|---|---|---|---|
| 1 | `e35ea018` | 2026-08-24 | **Σ^φ adopted as the no-BH catalogue divisor.** `catalogue_global_selection` `"auto"` → `"phi"` under `absolute_marginal` (`bayesian_statistics.py:3605-3609`, verified). Row #178/#179. | yes — `--catalogue_global_selection` |
| 2 | `bac48696` | 2026-08-25 | **Catalogue-leg twin adopted.** `catalogue_numerator_survival` default `"off"` → `"auto"` → `"phi"` under `absolute_marginal` (`bayesian_statistics.py:3421`). Rows #195/#197. | **NO CLI FLAG** — not present in `arguments.py`, not wired in `main.py`. Baked in. |
| 3 | `cf4f8a2a` | 2026-08-25 | **Symmetric mass-filter window adopted.** `mass_filter_sigma` default `"asymmetric"` → `"symmetric"` at 5 declaration sites (`handler.py:570`; `bayesian_statistics.py:3248/3311/3460`; `correspondence_1d.py:2762`). Rows #196–#202. | **NO CLI FLAG** — not present in `arguments.py`. Baked in. |

### 3.2 CORRECTION to the task brief — `266d7290` is *not* a fourth estimator change

The brief lists `266d7290` (2026-08-22, "PRODUCTION_FLAGS completion-cell pin off→fused") as one of
four commits that change what a bare production `--evaluate` at HEAD computes. **It does not.**
Verified this session:

- `git show --stat 266d7290` touches only `validation/correspondence_1d.py`,
  `validation/selfgen_control.py`, a test, and the gate ledger. **It never touches
  `bayesian_statistics.py` or `arguments.py`.**
- At `e65d263c` — row #132's own commit — `selection_in_completion_numerator="auto"` **already**
  resolved to `"fused"` under `absolute_marginal`
  (`git show e65d263c:darksiren_emri/bayesian_inference/bayesian_statistics.py`, lines 3142-3145),
  and the same code already logged
  `COUNTERFACTUAL: selection_in_completion_numerator='off' under absolute_marginal — the legacy
  pre-#118 estimator … Not a production posterior.`

So the off/fused difference between row #132 and this readout is a **configuration delta, not a code
delta**: row #132 explicitly pinned `off` as its then-basis-of-record; this readout takes the
current production basis `fused`, ratified as the basis for all future runs-of-record by the
author's D2 ruling (**ledger row #159**, 2026-08-22) and pinned in `PRODUCTION_FLAGS`
(`correspondence_1d.py:328-337`).

**The composition to be reported is therefore: 3 estimator-code changes ⊕ 1 ratified configuration
change (off → fused).** Any readout narrative that says "four landed physics commits" is wrong and
must be corrected before it enters the ledger.

### 3.3 Three landed `[PHYSICS]` commits that are default-inert

`606af0e0` (`--completion_event_measure`, default `"ratio"` unchanged), `6d9e21a1`
(`--eddington_m` / `--sigma4d_mass_kernel` battery instruments, defaults unchanged), `24921db3`
(`validation/correspondence_1d.py` scorer sentinel/trapezoid fix — validation module only, does not
touch `bayesian_statistics.py`). None of these are exercised by this measurement.

### 3.4 Documentation gap (awareness only, not a runtime bug)

`PRODUCTION_FLAGS` (`correspondence_1d.py:328-337`) lists only `absolute_marginal`,
`volume_deconv`, `fused`, `production`, `derived`, and the p_det settings. It does **not** enumerate
`catalogue_global_selection = phi`, `catalogue_numerator_survival = phi`, or
`mass_filter_sigma = symmetric`, although its header comments (lines 313-327) narrate the Σ^φ
adoption. A reader consulting the dict literal alone would miss two of the three adoptions.

---

## 4. What a change would, and would not, license

### 4.1 LICENSED

- **THAT the production bias moved, and by how much**, on each venue and each channel, against a
  comparand whose configuration and commit are both pinned (§2.3, §3).
- **Whether the 1D rail survives** the three adoptions.
- **Whether coverage changed** from row #132's C68 = C90 = 0.
- A statement of production's **current** H₀ bias for reporting purposes, with the §6 anchor caveat
  attached.

### 4.2 NOT LICENSED — registered structural blindness

**This design cannot attribute the move to any individual change.** Registered now, pre-data, so it
cannot be quietly dropped at readout time:

1. **The three code changes are not separately identified.** All three land together in one arm.
   There is no term-by-term decomposition in this design and none may be inferred from the total.
2. **The banked per-change magnitudes do not transfer.** Row #171's −0.004309 and row #173's
   +0.029068 were measured on the b0i/mirror venue under the twin's own coherent basis, not on
   iiib/joint_r1 production. Row #201's mass-filter read (ΔT = +0.800030, Δw̄ = +0.000449, zero-rate
   43.32 % → 0.00 %) is a per-event log-weight/weight-share read at a **single h = 0.73**, not an
   H₀-space quantity. **None of the three has an H₀-space production measurement.** Quoting any of
   them as "the contribution" to this readout's move is forbidden.
3. **Directions conflict.** Two of the measured mirror-venue moves have opposite sign. Their sum is
   not the total, and their absence of a sum is not evidence of cancellation.
4. **Sign agreement is not attribution.** If the readout moves in the direction of, say, the twin's
   mirror-venue sign, that is a coincidence of one bit and licenses nothing.
5. **Interaction terms are unbounded.** The three changes all touch the catalogue leg and are not
   linearly separable a priori.

**Any per-change attribution requires its own arm.** The only decomposition this design *can*
support is the 2-way split described in §8.5, if the optional `off` arm is authorised.

### 4.3 Also not licensed

- No claim that the estimator is calibrated (§6).
- No claim that any of the three changes is "correct" or "vindicated" by a bias reduction. Author
  standing ruling: **correctness over bias-removal** (2026-08-05; `CLAUDE.md`). A bias that shrinks
  is not evidence a change was right, and a bias that grows is not evidence it was wrong.
- No re-grade of any systematics-budget row.

---

## 5. Registered directional reads and thresholds

**No point prediction is registered.** The moves cannot be composed (§4.2), so a point prediction
would be invented, not derived. What is registered is the band structure and the numeric thresholds.

### 5.1 Threshold derivation

Row #132 **carries no uncertainty on its offset**: it is a single realization of a single mirror
universe, with no repeat-realization standard error. Its σ_h is a *posterior width*, not an offset
error bar. Two scales are therefore available, and the **larger** is taken (conservative — it makes
"materially changed" harder to declare):

**(a) Grid-resolution scale.** The discrete MAP cannot move by less than one node.
- 2D channel: row #132's MAP = 0.675 sits in the dense peak, node spacing **0.005**.
- 1D channel: row #132's MAP = 0.600 sits in the low wing, node spacing **0.010**.

**(b) Claimed-width scale, σ_h ⁄ 3.** A systematic combined in quadrature at σ_stat/3 inflates the
quoted total by `sqrt(1 + (1/3)²) = 1.05409`, i.e. **5.41 %** — below the level at which any
statement made with the quoted width would change. Hence a shift smaller than σ_h/3 is not
decision-relevant.
- 2D: 0.0239/3 = **0.007967** (iiib), 0.0225/3 = **0.0075** (joint_r1) → take the max, round up.
- 1D: 0.0033/3 = **0.0011** (iiib), 0.0046/3 = **0.00153** (joint_r1) → both far below the wing step.

**Registered thresholds:**

| | 2D channel | 1D channel |
|---|---|---|
| `T_res` (MAP resolution floor) = grid node spacing | **0.005** | **0.010** |
| `T_mat` (material offset change) = max(node spacing, σ_h/3) | **0.008** | **0.010** |
| consistency check `T_mat > T_res`? | 0.008 > 0.005 ✓ | 0.010 = 0.010 (grid-limited) |
| `T_mat` as a fraction of \|offset₁₃₂\| | 15.1 % (iiib) / 15.6 % (joint_r1) | 7.8 % / 7.8 % |

A single `T_mat` = **0.008** is used for both 2D venues (the larger of 0.007967 and 0.0075),
conservative on joint_r1.

**Registered limitation, 1D channel:** σ_h(1D) = 0.0033 / 0.0046 is **smaller than the 0.010 wing
node spacing**. The 1D posterior is a near-delta piled against the h = 0.600 grid boundary and is
**not resolved by the production grid**. All 1D σ_h values, and any 1D σ_h comparison, are
**REPORTED-ONLY** — quadrature artefacts of a boundary-truncated near-delta, not widths. This is a
pre-existing condition of row #132, not something this run introduces. The 1D read of record is the
**rail statistic**, not the width.

### 5.2 Registered bands — 2D channel (the channel of record)

Let `Δ = offset_new − offset_132` (signed) and evaluate per venue.

| band | condition | meaning |
|---|---|---|
| **RESOLVED** | `\|offset_new\| < 0.008` | The production 2D bias is consistent with zero at the materiality scale. Strongest possible outcome. |
| **MATERIALLY REDUCED** | `Δ ≥ +0.008` **and** `\|offset_new\| < \|offset_132\|` **and** not RESOLVED | Bias moved toward truth by ≥15 % of its own size. |
| **UNCHANGED** | `\|Δ\| < 0.008` | Not distinguishable from row #132 at the materiality scale. The three adoptions are H₀-immaterial *in composition* on this venue. |
| **MATERIALLY GROWN** | `\|offset_new\| > \|offset_132\| + 0.008` | Bias moved away from truth. |
| **SIGN-FLIPPED** | `offset_new > 0` **and** `offset_new ≥ +0.008` | The estimator now over-estimates h. A distinct qualitative state, not merely a large reduction. |
| **MOVED-BUT-UNCLASSIFIED** | any `\|Δ\| ≥ 0.008` not matching the above | Report as-is; do not force a band. |

Bands are evaluated **independently per venue**. If iiib and joint_r1 land in different bands, that
is itself the result (a catalogue-realization-dependent response) and must be reported as such —
**no averaging, no "the venues agree" summary.**

Companion reads, REPORTED-ONLY: `ΔMAP` (band-bearing only at `|ΔMAP| ≥ 0.005`), Δσ_h, Δpull.

### 5.3 Registered bands — 1D channel (the rail)

Rail statistic: the already-registered `R_LOW_THRESHOLD = 0.605` (`correspondence_1d.py:359`,
"DS-6 rail statistic (prereg S-RAIL)"). Not invented here.

| band | condition | row #132 |
|---|---|---|
| **RAIL PERSISTS** | `mean_h(1D) ≤ 0.605` **or** `MAP(1D) == 0.600` | ✅ fires (0.6010 / 0.6020; MAP 0.600 both) |
| **RAIL BROKEN** | `mean_h(1D) > 0.605` **and** `MAP(1D) > 0.600` | — |
| **RAIL LOOSENED** | `mean_h(1D) > 0.605` **but** `MAP(1D) == 0.600` | intermediate; report, do not call it broken |

`Δoffset(1D)` is band-bearing at `|Δ| ≥ 0.010` (§5.1). Note the 1D channel is **boundary-truncated**
at h = 0.600: if the true 1D posterior mode lies below 0.600, `mean_h(1D)` and `offset(1D)` are
censored and their magnitudes are **lower bounds on the true 1D bias**. Registered as a bound, not a
value.

### 5.4 Coverage — registered as binary, with the caveat inline

`C68`, `C90` ∈ {0, 1} per venue × channel: does h_true = 0.73 fall inside the HPD interval of the
combined posterior on this **one** realization?

**This is a hit/miss indicator, not a calibrated coverage fraction.** A coverage *fraction* requires
many independent seeds/injections and is what `validation/pp_coverage.py` exists to produce. Any
readout, figure caption, or ledger row that reports these numbers as "coverage" without the N = 1
qualifier misrepresents the statistical confidence. **The word "coverage" alone is forbidden in the
readout; use "single-draw coverage indicator (N = 1)".**

The informative continuous companion is the **pull** `(0.73 − mean_h)/σ_h`. Row #132: 2D +2.218
(iiib) / +2.281 (joint_r1); 1D +39.17 / +27.67.

---

## 6. Anchor status

**What this measurement is anchored to.** Injected-truth recovery in a self-generated mock: the
seed61000 events were drawn at h_true = 0.73, so truth is known by construction, and offsets against
it are meaningful.

**Does that clear the missing-anchor cap? NO.**

The campaign's one genuinely anchored result is **ledger row #99** (venue-transfer, 2026-08-13),
and what made it anchored was its **null-dose control**: `T-0 anchor (σ_z = 0): all 200 seeds argmax
exactly on truth, rails 0 — the apparatus is unbiased` (`BIAS_HISTORY_LEDGER.md:389`). The control
demonstrated that the measuring apparatus returns truth when the effect under study is switched off.

**This readout has no such control, and none is available at this scope.** Specifically:

- There is **no null-dose arm** here. The three adopted changes are baked-in defaults; two
  (`catalogue_numerator_survival`, `mass_filter_sigma`) have **no CLI flag at all** (§3.1), so a
  "changes off" cell cannot even be requested from `--evaluate` at HEAD without a code change.
- The nearest existing controls are **harness-scope, not production-scope**: row #136's G-1 `f ≡ 1`
  control and row #139's B-F1 (`0.7300`, truth to 4 dp, coverage 1/1/1) both recovered truth exactly
  — but in the mirror harness, which row #138 item 2 established is **structurally blind** to the
  data-vs-model population mismatch that dominates production's dark class (production is
  ~95 % out-of-catalogue; the mirror is 100 % in-catalogue, row #136).
- The optional `off` arm (§8.5) is a **configuration** control, not a null-dose control. It isolates
  one axis; it does not demonstrate the apparatus is unbiased.

**What the cap implies for how this result may be used.** The readout is admissible as a *state
measurement of production's offset against its own injected truth*, and deltas against row #132 are
legitimate (same venues, same events, same grid, same scorer). It is **not** admissible as evidence
that the estimator is calibrated, that any adopted change is correct, or that a reduced bias means a
resolved bias. It carries the same UNATTRIBUTED-bounded status the campaign already lives under
(row #211). If the author wants an anchored production result, that requires a separate,
production-scope null-dose control — a new instrument, not this run.

---

## 7. Invariants and non-invariants relative to row #132

### 7.1 HELD FIXED (verified this session)

| invariant | evidence |
|---|---|
| CRB / event set (1590 rows → 1588 scored, 2 filtered) | md5 `9a1f2a14…` identical cluster + local |
| Events per venue = 1588 | banked CSVs 65 109 lines = 1 + 1588×41, both venues |
| True reduced catalogue (iiib input) | md5 `c52c13b5…` = `REDUCED_CATALOGUE_MD5` pin, verified on cluster |
| Observed realization (joint_r1 input) | `observed_catalogue_seed900001.csv`, 2 526 653 003 B, sidecar `observed_csv_sha256 e8f7ab31…` |
| h grid | `H_GRID_41` ≡ `evaluate.sbatch:56` `H_VALUES`; `H_VALUES_OVERRIDE` unset |
| Seeding | `EVAL_SEED = 777000`, per-task seed = 777000 + task_id (row #132 identical) |
| `normalization_mode` | `absolute_marginal` (explicit, both runs) |
| `host_z_kernel` | `volume_deconv` (explicit, both runs) |
| `completion_b_scale` | `derived` |
| `catalogue_mass_overlap` | `production` |
| p_det settings | `dl_bins 60`, `mass_bins 40`, `local_linear`, `z_resolved True`, `wbh_z_resolved False` |
| Scorer / statistic convention | §1.3, byte-for-byte the row #132 T0 convention |
| Truth | h_true = 0.73 |

### 7.2 NOT HELD FIXED — this is the measurement

| axis | row #132 | this readout |
|---|---|---|
| `catalogue_global_selection` (no-BH divisor) | Σ³ᴰ (`s3d`; pre-`e35ea018`) | **Σ^φ (`phi`)** |
| `catalogue_numerator_survival` (no-BH catalogue-leg twin) | `off` (pre-`bac48696`) | **`phi`** (baked in) |
| `mass_filter_sigma` (mass-filter window) | `asymmetric` (pre-`cf4f8a2a`) | **`symmetric`** (baked in) |
| `selection_in_completion_numerator` (completion cell) | `off` — explicitly pinned basis of record | **`fused`** — ratified basis, row #159 D2 |
| commit | `e65d263c` (2026-08-19) | HEAD source tree `7bfff25d` |

### 7.3 Changed but asserted inert (must be recorded, not assumed)

The cluster checkout is at commit `d04d9dc9` on branch `fix/p32d-classg-venue-repair`, **not** local
HEAD `bbfdd2e0`. Verified this session: `git rev-parse d04d9dc9:darksiren_emri` ==
`git rev-parse HEAD:darksiren_emri` == `7bfff25dbdb95383304b2cef576edde17d957242` — the *source
tree* is byte-identical; the commits differ only in docs/results. `git rev-parse d04d9dc9:cluster`
== `HEAD:cluster` likewise. The provenance stamp of record for this run is therefore
**`d04d9dc9` (tree `7bfff25d`)**, and the readout must state that HEAD `bbfdd2e0` carries the
identical `darksiren_emri/` tree.

---

## 8. Submission plan

**This registration does not submit.** The orchestrator submits.
Cluster preflight (`CLAUDE.md` hard gate) must return **`VERDICT: READY ✓`** immediately before
STEP 2. Run everything below from `~/darksiren-emri` on the cluster.

Verified live this session: `WS = /pfs/work9/workspace/scratch/st_ac147838-emri`; workspace expires
**2026-09-23**, **0 extensions remaining** (26 d 22 h left); filesystem 62 % used, 1.7 P free; user
queue **empty** (job 6723958 completed 24/24 at 19:20 today); `run_20260827_*` out-roots **do not
exist** (no idempotency collision).

### STEP 0 — dataset pin (mandatory, `CLAUDE.md` pinning rule)

```bash
WS=$(ws_find emri)
md5sum "$WS/run_20260729_seed61000/simulations/prepared_cramer_rao_bounds.csv"
#   expect 9a1f2a14384a9281c97ca3be312ddaab                        — STOP on mismatch
sha256sum "$WS/realizations_20260729/observed_catalogue_seed900001.csv"
#   expect e8f7ab310ea70ddfdd3b81970dc99ad943808e6b6c128777bb085db01b4f6751  — STOP on mismatch
md5sum "$HOME/darksiren-emri/darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv"
#   expect c52c13b5cab61f6b3f04bbe202550969                        — STOP on mismatch
git -C ~/darksiren-emri rev-parse HEAD:darksiren_emri
#   expect 7bfff25dbdb95383304b2cef576edde17d957242                — STOP on mismatch
```

### STEP 1 — build the two out-roots

```bash
WS=$(ws_find emri)
for V in iiib joint_r1; do
  RD="$WS/run_20260827_headreadout_$V"
  mkdir -p "$RD/simulations" "$RD/logs"
  ln -sfn "$WS/run_20260729_seed61000/simulations/cramer_rao_bounds.csv"          "$RD/simulations/cramer_rao_bounds.csv"
  ln -sfn "$WS/run_20260729_seed61000/simulations/prepared_cramer_rao_bounds.csv" "$RD/simulations/prepared_cramer_rao_bounds.csv"
  ln -sfn "$WS/run_20260729_seed61000/simulations/injections"                     "$RD/simulations/injections"
done
```

This is exactly the symlink pattern `run_20260819_postfix_baseline_*` used (verified live on the
cluster this session). `cluster/submit_pipeline.sh` was **not** used to create those dirs and is not
used here — the symlinks are made by hand.

### STEP 2 — SMOKE (mandatory; 2 tasks) ⚠️

`--array=21` is **h = 0.730**, the injected truth node — and, from the row #132 anchor, the
**slowest task of all 82** (`6372475_21`, 1306 s). It is therefore the correct walltime probe *and*
the most decision-relevant single node.

```bash
WS=$(ws_find emri); PR=$HOME/darksiren-emri

# --- iiib smoke ---
export PROJECT_ROOT="$PR" EVAL_SEED=777000
export NORMALIZATION_MODE=absolute_marginal HOST_Z_KERNEL=volume_deconv
export EXTRA_EVAL_ARGS="--selection_in_completion_numerator fused --catalogue_global_selection phi"
export RUN_DIR="$WS/run_20260827_headreadout_iiib"; unset OBSERVED_CATALOGUE
sbatch --parsable --array=21 --time=03:00:00 --job-name=hr-smoke-iiib \
  --output="$RUN_DIR/logs/evaluate_%A_%a.out" --error="$RUN_DIR/logs/evaluate_%A_%a.err" \
  --export=ALL cluster/evaluate.sbatch

# --- joint_r1 smoke ---
export RUN_DIR="$WS/run_20260827_headreadout_joint_r1"
export OBSERVED_CATALOGUE="$WS/realizations_20260729/observed_catalogue_seed900001.csv"
sbatch --parsable --array=21 --time=03:00:00 --job-name=hr-smoke-joint-r1 \
  --output="$RUN_DIR/logs/evaluate_%A_%a.out" --error="$RUN_DIR/logs/evaluate_%A_%a.err" \
  --export=ALL cluster/evaluate.sbatch
```

**Smoke acceptance gate — all four must pass before STEP 3:**

1. Both tasks `COMPLETED` (`sacct -j <id> -X`).
2. `run_metadata_21.json` in each run dir shows `normalization_mode=absolute_marginal`,
   `host_z_kernel=volume_deconv`, `selection_in_completion_numerator=fused`,
   `catalogue_global_selection=phi`, `completion_b_scale=derived`,
   `catalogue_mass_overlap=production`, `pdet_*` as §2.3, and `observed_catalogue` = null (iiib) /
   the seed900001 path (joint_r1). **Any deviation ⇒ STOP.**
3. Job log contains both `[PHYSICS] selection fusion ACTIVE` and
   `[PHYSICS] catalogue_global_selection="phi" ACTIVE`, and contains **no** line starting
   `COUNTERFACTUAL:`.
4. Size the full array: `--time = ceil(1.7 × max(elapsed_smoke))`, floor 01:00:00, cap 04:00:00.

### STEP 3 — full evaluate (2 × 41 tasks; `--array=21` will self-skip)

Identical env exports to STEP 2; only `--array`, `--time` and `--job-name` change.

```bash
# iiib
export RUN_DIR="$WS/run_20260827_headreadout_iiib"; unset OBSERVED_CATALOGUE
sbatch --parsable --array=0-40 --time=<from smoke> --job-name=hr-iiib \
  --output="$RUN_DIR/logs/evaluate_%A_%a.out" --error="$RUN_DIR/logs/evaluate_%A_%a.err" \
  --export=ALL cluster/evaluate.sbatch

# joint_r1
export RUN_DIR="$WS/run_20260827_headreadout_joint_r1"
export OBSERVED_CATALOGUE="$WS/realizations_20260729/observed_catalogue_seed900001.csv"
sbatch --parsable --array=0-40 --time=<from smoke> --job-name=hr-joint-r1 \
  --output="$RUN_DIR/logs/evaluate_%A_%a.out" --error="$RUN_DIR/logs/evaluate_%A_%a.err" \
  --export=ALL cluster/evaluate.sbatch
```

Partition: `cpu,cpu_il` (from `evaluate.sbatch:36`, unchanged). `--cpus-per-task=16` unchanged —
matched to the internal worker pool (`os.sched_getaffinity(0) − 2` = 14 workers), so no idle-core
violation. The `--time` CLI flag overrides the script's `#SBATCH --time=06:00:00` placeholder; this
is a flag override, **not** a file edit.

`--array=21` is re-submitted deliberately: `evaluate.sbatch:104-113`'s idempotency guard sees the
smoke's `h_0_73.json` and skips it in ~0 s. That is correct **only because the env is byte-identical
between STEP 2 and STEP 3** — if any flag is changed between the two steps, the smoked node must be
deleted and re-run.

### STEP 4 — combine (cross-check; 2 tasks, cheap)

```bash
export RUN_DIR="$WS/run_20260827_headreadout_iiib"; unset OBSERVED_CATALOGUE
sbatch --dependency=afterok:<EVAL_IIIB> --output="$RUN_DIR/logs/combine_%j.out" \
  --error="$RUN_DIR/logs/combine_%j.err" --export=ALL cluster/combine.sbatch
export RUN_DIR="$WS/run_20260827_headreadout_joint_r1"
export OBSERVED_CATALOGUE="$WS/realizations_20260729/observed_catalogue_seed900001.csv"
sbatch --dependency=afterok:<EVAL_JOINT> --output="$RUN_DIR/logs/combine_%j.out" \
  --error="$RUN_DIR/logs/combine_%j.err" --export=ALL cluster/combine.sbatch
```

`combine.sbatch`'s own `--time=01:30:00` (~4.5× its ~20 min anchor) is left unchanged.

### 8.5 OPTIONAL companion arm — the `off` cell (author decision, §10 item 3)

Adding two more `--array=0-40` submissions with
`EXTRA_EVAL_ARGS="--selection_in_completion_numerator off --catalogue_global_selection phi"` into
out-roots `run_20260827_headreadout_off_{iiib,joint_r1}` converts the comparand from a 4-term
composition into a clean **2-way split**:

```
off@HEAD  −  off@e65d263c(row #132)   =   the THREE estimator-code changes, isolated
fused@HEAD −  off@HEAD                =   the off→fused configuration delta, isolated
```

It does **not** separate the three code changes from each other (§4.2 item 1 stands regardless).
Prior bound: row #119 M-3 measured "no MAP motion" for off vs fused on these venues — but at the
**pre-B_scale-fix** estimator (2026-08-17), so it does not transfer.

Cost: exactly doubles the evaluate stage. Recommended (§10).

### 8.6 Provenance-stamp confirmation

`cluster/write_provenance.sh` is **not** needed here: it exists for bespoke drivers that bypass the
package entry point (`cluster/SKILL.md` gotcha 12). `evaluate.sbatch` calls
`python -m darksiren_emri`, which writes `run_metadata_<task_id>.json` natively via
`main.py:_write_run_metadata` — 41 stamps per venue, each carrying `git_commit`, `timestamp`, and
the full `cli_args`. **Stamp of record: `run_metadata_21.json`, checked at the STEP 2 gate.**
After STEP 3, verify all 41 `run_metadata_*.json` per venue agree on every physics key.

### 8.7 Post-run integrity gates (run before any scoring)

The diagnostics CSV is written **append-mode** from 41 concurrent array tasks
(`bayesian_statistics.py:4527-4530`, `_write_diagnostic_csv` at :4572, "append mode, header on
first write"), to `$RUN_DIR/simulations/diagnostics/event_likelihoods.csv`. Row #132's 82-task run
produced a clean file (65 108 rows, 0 duplicates, per the banked counterfactual readout's
`sanity_dedupe`), so the race is empirically survivable — but it must be checked, not assumed.

Per venue, before scoring:

1. `event_likelihoods.csv` has exactly **65 108** data rows (1588 × 41).
2. Zero duplicate `(event_idx, h)` pairs.
3. Exactly 41 distinct `h`, matching `H_GRID_41` element-wise.
4. Exactly 1588 distinct `event_idx`, and the set equals row #132's set for that venue.
5. Zero non-positive values in `combined_no_bh` and `combined_with_bh` (row #132 had zero; a
   non-zero count is a floor/sentinel signal and a STOP — cf. the 2026-08-20 sentinel defect).
6. `posteriors/h_*.json` and `posteriors_with_bh_mass/h_*.json` = 41 files each.
7. Scorer cross-check: `mean_h`/`MAP` from the §1.3 CSV scorer agree with
   `compute_canonical_combined_posterior` on `posteriors/` to ≤ 1e-6 in `discrete_map` and the
   posterior shape. Disagreement ⇒ STOP, do not report.

---

## 9. Cost

| stage | tasks | cpus/task | CPU-h |
|---|---:|---:|---:|
| smoke (STEP 2) | 2 | 16 | 0.6 – 1.8 |
| evaluate, both venues (STEP 3) | 80 new | 16 | **102 (lower bound) – 260** |
| combine (STEP 4) | 2 | 4 | 2.7 |
| **total, primary** | 84 | | **~105 – 265 CPU-h** |
| optional `off` arm (§8.5) | +82 (+2 combine) | | +105 – 265 |

**The 104 CPU-h anchor is a LOWER BOUND, and the costing recon treated it as a point estimate.**
Re-derived from `sacct` this session over jobs 6372475/6372476 (82 COMPLETED tasks): sum 23 452 s,
mean 286 s, min 199 s, max 1306 s ⇒ 23 452 × 16/3600 = **104.2 CPU-h**. But that anchor ran
`selection_in_completion_numerator=off` at `e65d263c`, which **skips** the fused survival cell
entirely. The HEAD configuration does strictly more work:

- the fused cell computes S̄_φ in the 1D completion numerator **and** S₄ᴰ inside the 2D mass
  quadrature (rows #117-#118);
- the twin multiplies the without-BH catalogue numerator integrand per host by S̄_φ (row #197);
- the symmetric mass filter **retains more hosts** — 43.32 % of iiib's h = 0.73 catalogue-leg rows
  go from zero to non-zero (row #201) — so quadratures that previously short-circuited now run.

No banked anchor bounds this slowdown. **The STEP 2 smoke resolves it for ~1 CPU-h and is why it is
mandatory.** Wall-clock is queue-dependent (the anchor day achieved ~16 nodes / high concurrency;
the queue is currently empty for this user) and should be read from `sacct`, not predicted.

---

## 10. Decisions required before submission

| # | tag | item |
|---|---|---|
| 1 | **[RULE]** | Ratify the §3.2 correction: the composition is **3 estimator-code changes ⊕ 1 ratified configuration change (off → fused)**, not four code changes. `266d7290` is a validation-module pin. |
| 2 | **[RULE]** | Ratify §4.2 as binding structural blindness: **no per-change attribution may be read out of this run**, and rows #171 / #173 / #201 magnitudes may not be quoted as contributions to it. |
| 3 | **[DO]** | Authorise (or decline) the §8.5 `off` companion arm. Recommended: it splits the comparand 2 ways at exactly 2× a modest cost, and no new inputs are needed. Declining is defensible; the readout must then state the 4-term composition plainly. |
| 4 | **[RULE]** | Ratify §5's thresholds (`T_mat` = 0.008 2D / 0.010 1D; `T_res` = 0.005 / 0.010) and the §5.4 N = 1 coverage language restriction. |
| 5 | **[RULE]** | Ratify §6: this readout does **not** clear the missing-anchor cap and may not be used as calibration evidence. |
| 6 | **[DO]** | Archive `$WS/realizations_20260729/observed_catalogue_seed900001.csv` (2.5 GB, sole copy) and `run_20260819_postfix_baseline_{iiib,joint_r1}` (6.7 GB cluster vs 36 MB local — figures/summary only; graded **MUST-ARCHIVE** in `cluster/WORKSPACE_ARCHIVAL_TRIAGE_20260827.md`, "iiib/joint_r1 pairs" row) before **2026-09-23**. The `realizations_20260729` parent (14 GB) is graded UNKNOWN/not-yet-archived in the same table. Independent of this run, but the same workspace. |

**Per `CLAUDE.md`'s approval-scope rule, the readout's verdict is not covered by any approval given
here.** These items authorise the *measurement*; the band call and any comparison against row #132
return to the author as a fresh **[RULE]** once the numbers exist.

---

## 11. Risks

| # | risk | status |
|---|---|---|
| R1 | **Walltime under-sizing.** The 1306 s anchor max is a `sel=off` figure; HEAD does strictly more work (§9). `evaluate.sbatch` writes `h_*.json` only at task end — a walltime kill loses the whole task, and regular users cannot extend. | **Mitigated by the mandatory STEP 2 smoke.** Do not skip it. |
| R2 | **Silent wrong estimator.** `arguments.py:748` defaults `normalization_mode` to `generator_marginal`; production is `absolute_marginal`. Omitting `NORMALIZATION_MODE` runs a different estimator **with no error**, and would also flip `host_z_kernel` auto→`point` and `catalogue_global_selection` auto→`s3d`. | Mitigated by the explicit export + STEP 2 gate item 2. |
| R3 | **joint_r1 silently becomes iiib.** Omitting `OBSERVED_CATALOGUE` runs joint_r1 against the true catalogue with no error. | Mitigated by STEP 2 gate item 2 (`observed_catalogue` in `run_metadata_21.json`). |
| R4 | **Idempotency false success.** Any out-root already holding `posteriors/h_*.json` makes tasks print "Skipping" and exit 0 — a green `sacct` with no new output. | Fresh out-roots `run_20260827_headreadout_*` verified absent on the cluster this session. Do **not** reuse any of the ~44 existing iiib/joint_r1-family dirs. |
| R5 | **Append-mode CSV race** across 41 concurrent tasks (§8.7). | Empirically survived at row #132; §8.7 gates 1-3 make it checked, not assumed. |
| R6 | **Comparand mis-framing.** Row #132's cell is labelled by the code itself `COUNTERFACTUAL … Not a production posterior`. Reporting the delta without §3.2's framing invites a reader to attribute a configuration change to physics. | §10 item 1. |
| R7 | **Single-realization coverage.** N = 1 ⇒ hit/miss only (§5.4). | §10 item 4. |
| R8 | **Cluster checkout ≠ local HEAD** (`d04d9dc9` on `fix/p32d-classg-venue-repair`). Source trees verified byte-identical (§7.3), but the run stamps `d04d9dc9`. | Record `d04d9dc9` + tree `7bfff25d` as the provenance of record. Do not pull/rebase the cluster checkout between STEP 2 and STEP 4 — that would break the smoke↔full-array equivalence R4/STEP 3 relies on. |
| R9 | **Catalogue schema drift** (cluster `/cluster` gotcha 2) was checked by md5 only (matches the pin, 8 columns confirmed on the header row); the full preflight was not run in this pass. | `VERDICT: READY ✓` required before STEP 2. |
| R10 | **Workspace expiry 2026-09-23, 0 extensions.** 26 d 22 h remaining, verified via `ws_list`. Two new 3.4 GB run dirs are created. Free space is not a constraint (1.7 P). | §10 item 6. |
| R11 | **No `/physics-change` gate applies** — this registration edits no physics-trigger file. But the readout is the input to a future **[RULE]**, and must not be auto-adjudicated by an agent against row #132. | §10 closing note. |

---

*Registered pre-data. Every claim above cites a file:line, a ledger row, or the output of a command
run in this session. Nothing is banked until STEP 3 completes and §8.7's gates pass.*

---

# POST-DATA READOUT [FABLE-ORCH 2026-08-28] — APPEND-ONLY; nothing above this line edited

**Everything below was computed AFTER the data existed. The pre-data registration above is
untouched (verified: this section is a pure append).**

## A. Submission provenance — A RECORD GAP, flagged for the author

The registration's status line reads "NOT SUBMITTED" and §10 lists six author decisions
"required before submission". **The run was nevertheless submitted and completed on
2026-08-27**: smoke jobs 6724169 (iiib) / 6724170 (joint_r1) at ~19:40 (array 21, h=0.730),
full arrays 6725283 / 6725284 at ~20:45, all tasks COMPLETED. No record of an author §10 ruling
exists in this document, the ledger, or the runbook, and no submission trace survives in shell
history. **Either a later session obtained approval that was never recorded, or the submission
jumped the gate.** The author is asked to rule on the record (item A-1 below). The data itself
is protocol-clean: every §8 gate that can be checked post-hoc passes (§B), the configuration
stamps match §2.3/§7.2 exactly, and the §8.5 optional `off` arm was NOT run (so §10 item 3
resolves to "declined-by-omission" and the 4-term composition framing of §3.2 is mandatory).

## B. STEP 2 / §8.7 gates — ALL PASS, both venues

- `run_metadata_21.json`, both venues: `absolute_marginal` / `volume_deconv` / `fused` / `phi` /
  `derived` / `production`; `observed_catalogue` null (iiib) vs seed900001 path (joint_r1);
  commit `d04d9dc9` (tree `7bfff25d` = local HEAD's darksiren_emri/, per §7.3). ✓
- Smoke logs: `[PHYSICS] selection fusion ACTIVE` + `[PHYSICS] catalogue_global_selection="phi"
  ACTIVE` present; **zero** `COUNTERFACTUAL:` lines in any of the 82 task logs. ✓
- The full array's task 21 idempotency-skipped in 11 s under a byte-identical env, as designed. ✓
- §8.7 per venue: 65 108 data rows exactly; 0 duplicate `(event_idx, h)` pairs; 41 distinct h
  matching `H_GRID_41` (range [0.600, 0.860]); 1588 events, set-identical to row #132's; zero
  non-positive cells in either channel; 41+41 posterior JSONs; scorer↔combine cross-check:
  `discrete_map` agrees exactly (iiib 0.665/0.600, joint_r1 0.660/0.600). ✓

Scoring executed by the orchestrator (not a subagent) with a fresh implementation of §1.3;
the comparand table §2.2 was independently reproduced to ≤5e-5 by a separate verifier before
this readout was taken.

## C. The measurement

### C.1 2D / with-BH channel (channel of record)

| venue | mean_h | offset | σ_h | MAP | pull | C68/C90 (N=1 single-draw indicator) |
|---|---:|---:|---:|---:|---:|:--:|
| iiib | 0.663347 | **−0.066653** | 0.018366 | 0.665 | +3.629 | 0 / 0 |
| joint_r1 | 0.663013 | **−0.066987** | 0.018637 | 0.660 | +3.594 | 0 / 0 |

Deltas vs row #132 (per venue, no averaging):

| venue | Δoffset | ΔMAP | Δσ_h | Δpull | **registered band (§5.2)** |
|---|---:|---:|---:|---:|---|
| iiib | **−0.013720** | −0.010 | −0.005501 | +1.411 | **MATERIALLY GROWN** (\|−0.0667\| > 0.0529 + 0.008) |
| joint_r1 | **−0.015774** | −0.015 | −0.003816 | +1.313 | **MATERIALLY GROWN** (\|−0.0670\| > 0.0512 + 0.008) |

Both ΔMAP are band-bearing (≥0.005). The continuous_map companions (REPORTED-ONLY): 0.662923
(iiib), 0.661631 (joint_r1).

### C.2 1D / no-BH channel (the rail)

| venue | mean_h | offset | σ_h (REPORTED-ONLY) | MAP | pull | rail read |
|---|---:|---:|---:|---:|---:|---|
| iiib | 0.605309 | −0.124691 | 0.007871 | 0.600 | +15.84 | MAP railed; mean_h crossed 0.605 by **+0.0003** |
| joint_r1 | 0.611683 | −0.118317 | 0.011747 | 0.600 | +10.07 | MAP railed; mean_h crossed 0.605 by +0.0067 |

Δoffset(1D): +0.004277 (iiib), +0.009722 (joint_r1) — **neither is band-bearing** (< 0.010,
joint_r1 marginally). **Registered-map wording gap, flagged rather than forced:** §5.3's
RAIL PERSISTS (`mean_h ≤ 0.605 OR MAP == 0.600`) and RAIL LOOSENED (`mean_h > 0.605 but
MAP == 0.600`) BOTH fire on both venues — the map's conditions overlap. Reading the LOOSENED row
as the more specific subcase: **RAIL LOOSENED on both venues; not broken** (MAP still 0.600, the
boundary-truncation caveat of §5.3 stands: 1D offsets remain lower bounds on the true 1D bias).
The author is asked to ratify this reading (item A-3).

### C.3 What this does and does not say (registered blindness §4.2, restated post-data)

The production 2D bias **grew** by ~0.014–0.016 in \|offset\| under the composition
[3 estimator-code changes ⊕ off→fused configuration change]. **No per-change attribution is
licensed**; rows #171/#173/#201 magnitudes may not be quoted as contributions; sign agreement
with any mirror-venue move licenses nothing; interaction terms are unbounded. Per §4.3 and the
author's standing correctness-over-bias-removal ruling, a grown bias is NOT evidence any adopted
change was wrong. The missing-anchor cap (§6) stands — this is a state measurement, not
calibration evidence. The `off` companion arm (§8.5, not run) remains the only registered route
to a 2-way split; it stays available at ~105–265 CPU-h.

## D. Fresh author decisions (band calls return as [RULE] per §10's closing note)

- **[RULE] A-1** — rule on the §A submission-record gap (retroactively ratify the submission +
  §10 items 1–5 against the completed data, or direct remediation).
- **[RULE] A-2** — ratify the §C.1 band calls: **MATERIALLY GROWN on both venues** (the offset
  moved away from truth by 0.0137/0.0158, ~26–31% of its own size).
- **[RULE] A-3** — ratify the §C.2 rail reading (RAIL LOOSENED both venues; the §5.3
  overlapping-conditions wording gap resolved in favour of the more specific row).
- **[DO] A-4** — authorize (or decline) the §8.5 `off` companion arm now that the primary shows
  MATERIALLY GROWN; it is the only registered instrument that can split code-changes from the
  off→fused configuration delta.
- **[DO] A-5** — §10 item 6 (archival of the 2.5 GB observed catalogue + postfix_baseline pair)
  remains open; workspace expires 2026-09-23 with 0 extensions.

*Post-data section computed and appended 2026-08-28. Scorer, gate outputs, and retrieved data:
`headreadout_20260827/{iiib,joint_r1}/` (local copy of the cluster out-roots' diagnostics +
posteriors).*
