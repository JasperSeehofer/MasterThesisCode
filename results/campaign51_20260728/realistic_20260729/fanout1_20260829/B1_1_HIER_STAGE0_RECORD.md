# B1.1 [HIER] Stage-0 — INDEPENDENT READER VERDICT RECORD

**Launched under rows #222/#223 — charter node B1.1.** Role: independent reader/verifier, a
different agent from the P0 runner (`runner3_wave2pre_20260829.log`) and from the driver's
builder (`B1_1_HIER_BUILD_NOTE.md` / `B1_2_DRIVER_EXTENSION_NOTE.md`). This record re-derives
every pooled number from the raw per-event CSVs independently (own script, not the driver's
functions) and states the verdict the registered text requires — it does not open a new
measurement and it does not soften, upgrade, or rescue the outcome.

## Comprehension-first summary

Stage 0's S0-A control asks a narrow, mechanical question: on a **mirror universe built so that
the estimator's kernel is exactly the generating law at θ = (0, 1)** (`host_mode=
"catalogue_selected"`, PA-HIER-19), does moving θ off (0, 1) produce a **zero** average pull on
the log-likelihood? If the instrument is honest, the answer must be yes — any nonzero pull is
mechanically a bug (in the hook, the venue's construction, or an un-audited invariant), not a
discovery, because generator and estimator share the same kernel by construction at truth. This
record re-derives that score by hand, from scratch, off the 4-seed × 5-node grid that finished
today (P0, the `theta_sites="2.2"` / `smear_global_selection=False` "CoR-P-faithful" form
registered in `PA-HIER-31(b)`) and gets **the same answer as the driver, to full float
precision**: the registered primary channel (`ln_L_no_bh`) pulls **7.1σ** on the scale-secant
score and **3.7σ** on the bias-secant score — both far past the registered 3σ line. Per the
prereg's own verdict map this is **B0-A′ — INSTRUMENT-DEFECT — STOP**: the measurement halts,
banks no physics, and returns to the author with frozen numbers. This localizes the defect
unusually tightly, because this run's flags (`theta_sites="2.2"`, `smear off`) mean θ is
mechanically zeroed before it ever reaches the global-selection site (site 2.3) — the *only*
live channel here is the per-host numerator kernel (sites 2.1/2.2), so the 7σ pull says
something is wrong there, or in the un-audited mirror↔production kernel-identity invariant it
depends on (§5.1 invariant 8), not in the global-selection machinery the earlier (now
superseded) smeared partial run had implicated. The with-BH channel, secondary and
non-registered, is quiet by comparison (0.4σ / 2.0σ) — the defect, whatever it is, is
concentrated in the no-BH likelihood path.

**What this licenses:** nothing beyond itself. No Stage-P/F launch, no S0-B launch, no C1 build.
**What it does not license:** treating "the instrument is certified" as settled — a companion
open contradiction (`PA-HIER-31` REVISION NOTE 2, R2′) already flags that this run's `smear off`
form is itself an unadjudicated narrowing of the originally-pinned CoR-M invariant, pending a
fresh author `[RULE]`. KW-Q1 (`kwq1_registered_run/`, currently writing, untouched by this node)
rides the same driver and inherits the same REPORTED-ONLY cap and the same open-contradiction
disclosure.

## 1. What ran (facts, not re-derived)

| item | value | source |
|---|---|---|
| Driver | `hier_s0_driver.py`, sha1 `3aad2da63bc48bc193f8b4fa5df9ca41be56e418` | this session, `sha1sum` |
| Form | `theta_sites="2.2"`, `smear_global_selection=False` (the CoR-P-faithful form, `PA-HIER-31(b)`) | `runner3_wave2pre.sh` invocation |
| Command of record | `--arm S0-A --seeds 900101,900102,900103,900104 --nodes truth,b_plus,b_minus,s_plus,s_minus --theta-sites 2.2 --smear off --out-root hier_s0_registered_run --jobs 1 --total-cpu-budget 14` | `hier_s0_registered_run/logs/runner3_wave2pre_20260829.log:1` |
| Run window | 2026-08-29 21:59:44 → 22:49:05 CEST, `rc=0`, wall 2959.6 s (≈49.3 min) at 14 cpu | `s0a_full_output.json:wall_s`; log lines 1, 17485 |
| h | 0.730 only | `s0a_full_output.json:h_values` |
| Venue | `bc`/`b0i` (`host_mode="catalogue_selected"`), `catalogue_numerator_survival="off"`, `catalogue_global_selection="phi"` | driver module docstring / constants |
| Out-root | `results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run/` (node dirs `node_<name>_sites2.2_nosmear`) | filesystem |

**Run history, this driver, this out-root (append-only chain):**
1. **runner-1** (`logs/runner_wave2pre_20260829.log`): P1 equivalence check (seed 900101,
   `b_plus`, sites2.2/nosmear, `rc=0`) → then a **P0 attempt** (`--jobs 2`, all nodes) that
   **crashed `rc=1`** — `pd.concat` `ValueError: No objects to concatenate` inside
   `compute_scores()` (the per-seed node results for `b_plus` were not collected across
   `--jobs>1` workers; disclosed in the prereg's own "P0 crash disclosure" tail entry) — then
   S0-C (`rc=0`, banked as `s0c_full_output.json`).
2. **runner-2** (`logs/runner2_wave2pre_20260829.log`): a second P0 attempt (`--jobs 2`, fixed
   driver) **crashed differently**: `AssertionError: daemonic processes are not allowed to have
   children` — the per-seed worker pool (`--jobs 2`) itself tries to open a nested
   `multiprocessing.Pool` inside `evaluate()`'s per-event loop, which Python forbids for a
   process already running inside a pool worker.
3. **runner-3** (`logs/runner3_wave2pre_20260829.log`, **the run of record scored below**):
   same command with **`--jobs 1`** (serializes the seed loop, avoiding the nested-pool
   restriction) — completed clean, `rc=0`, 2026-08-29 21:59:44–22:49:05. The log's own echoed
   label line says "jobs2" (a copy-pasted stage-label string left over from the runner-2
   script), but the actual invocation flag, read from `runner3_wave2pre.sh` and confirmed by
   `s0a_full_output.json:"jobs": 1`, is `--jobs 1` — a cosmetic label bug, not a parameter
   error, disclosed here rather than silently corrected.

## 2. Independent re-derivation

Re-implemented the score, GATE ENG, and GATE PARITY computations from scratch (own script, not
`hier_s0_driver.compute_scores`/`gate_eng`/`gate_parity`), reading directly from
`s0a_seed<seed>/node_<name>_sites2.2_nosmear/simulations/diagnostics/event_likelihoods.csv`,
filtering `h == 0.73`, de-duplicating on `event_idx` (keep last — see the duplicate-row note
below), taking `ln(combined_no_bh)` / `ln(combined_with_bh)` where the raw value is `> 0`
(`NaN` otherwise), joining `b_plus`/`b_minus` and `s_plus`/`s_minus` per `(seed, event_idx)`,
and pooling over all seeds and events.

```
score_b,i = [lnL_i(b=+0.02, s=1) - lnL_i(b=-0.02, s=1)] / 0.04
score_s,i = [lnL_i(b=0, s=√2)   - lnL_i(b=0, s=1/√2)]   / (√2 - 1/√2)     (denominator 0.707107)
Z_x       = mean(score_x) / SEM(score_x),  SEM = sample-std(ddof=1)/√n, pooled over events+seeds
```

**Result: bit-identical to the driver's own `s0a_score_output.json` / `s0a_full_output.json` in
every reported field** (mean, SEM, Z, n_pooled, GATE ENG fractions, GATE PARITY diffs). No
correction to the driver's arithmetic is needed.

### 2.1 Pooled scores (all 4 seeds, N = 461 event-instances pooled)

| channel | statistic | mean | SEM | **Z** | n_pooled |
|---|---|---:|---:|---:|---:|
| `ln_L_no_bh` (**registered primary**) | score_b | −1.61646 | 0.43968 | **−3.6764** | 461 |
| `ln_L_no_bh` (**registered primary**) | score_s | −0.08625 | 0.012185 | **−7.0786** | 461 |
| `ln_L_with_bh` (secondary/diagnostic) | score_b | +0.13830 | 0.36465 | +0.3793 | 461 |
| `ln_L_with_bh` (secondary/diagnostic) | score_s | −0.02920 | 0.014409 | −2.0268 | 461 |

Source of my re-derivation: `/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/70977a05-4e21-4daa-91f0-d0330808c1ea/scratchpad/rederive.py`, run 2026-08-29, against the CSVs listed in §1. Cross-check: `hier_s0_registered_run/s0a_score_output.json:scores` (identical to the last reported digit).

### 2.2 Per-seed breakdown (registered primary channel, `ln_L_no_bh`)

| seed | n events | score_b mean | score_b SEM | score_b Z | score_s mean | score_s SEM | score_s Z |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 900101 | 106 | −3.0379 | 1.1380 | −2.6694 | −0.10523 | 0.026680 | −3.9442 |
| 900102 | 120 | −2.6751 | 0.9056 | −2.9539 | −0.11887 | 0.027732 | −4.2864 |
| 900103 | 105 | −0.1493 | 0.6026 | −0.2477 | −0.05216 | 0.019785 | −2.6364 |
| 900104 | 130 | −0.6653 | 0.7803 | −0.8525 | −0.06821 | 0.021809 | −3.1275 |

Every one of the 4 individual seeds already shows `score_s` at or past the 3σ line on its own
(2.64σ–4.29σ), all with the **same sign**; `score_b` is more variable seed-to-seed (2 of 4 seeds
individually exceed 3σ) but pools to 3.68σ. This is not one outlier seed driving the pooled
number — the `score_s` pull is coherent across all 4 seeds independently.

### 2.3 By class (registered primary channel, "dark" ≡ `L_cat_no_bh == 0`, `B3_1_POP_RECORD.md` convention)

| class | n_pooled | score_b mean/SEM/Z | score_s mean/SEM/Z |
|---|---:|---|---|
| dark (`L_cat_no_bh == 0`) | 5 | 0.0 / 0.0 / undefined (zero variance) | 0.0 / 0.0 / undefined (zero variance) |
| matched (`L_cat_no_bh > 0`) | 456 | −1.63419 / 0.44444 / **−3.6770** | −0.08720 / 0.012311 / **−7.0828** |

The 5 pooled dark-class events return **exactly zero** score on both axes — `combined_no_bh` is
bit-identical across all five θ-nodes for these events, exactly the instrument-identity check
`PA-HIER-31(d)` registers for the production arm ("for every C-C event ... `combined_no_bh` must
be bit-identical across all five θ-nodes ... any deviation is INSTRUMENT-DEFECT"): **that check
PASSES here.** The entire 7σ/3.7σ pull lives in the 456 matched-class events — consistent with
this venue's `catalogue_numerator_survival="off"` construction, where matched events carry the
per-host kernel term (sites 2.1/2.2) that dark events, by definition, do not. This venue
(mirror `b0i`, `host_mode="catalogue_selected"`) is **not** the production `iiib` venue and its
class fractions (456/461 matched vs. `PA-HIER-31(d)`'s production 982/1588) are not comparable —
reported for context, not as a cross-venue check.

### 2.4 GATE ENG (independently recomputed, `ln_L_no_bh`, ≥1e-6 relative move vs. truth)

| node | per-seed fraction moved (900101, 900102, 900103, 900104) | mean | **pass (≥0.10)** |
|---|---|---:|---|
| b_plus | 0.9906, 1.0000, 0.9714, 0.9923 | 0.98858 | **PASS** |
| b_minus | 0.9906, 1.0000, 0.9714, 0.9923 | 0.98858 | **PASS** |
| s_plus | 0.9906, 1.0000, 0.9714, 0.9923 | 0.98858 | **PASS** |
| s_minus | 0.9906, 1.0000, 0.9714, 0.9923 | 0.98858 | **PASS** |

θ engages essentially every scored event (≈98.9% mean, vs. the 10% registered floor) at every
off-truth node — GATE ENG passes decisively and uniformly. Because `theta_sites="2.2"`, this
movement is attributable to the per-host numerator kernel alone (site 2.3 is mechanically
un-instrumented in this run — θ is zeroed before it reaches that precompute call, per
`PA-HIER-31` REVISION NOTE 1, R3).

### 2.5 GATE PARITY — this driver's own truth-vs-banked-CSV check (distinct from the registered §3.3 GATE PARITY)

**Disambiguation, stated explicitly because both objects share a name in this thread's prose.**
The prereg's **registered** GATE PARITY (§3.3) is a zero-compute analytic diff of two functional
forms (`correspondence_1d.py:1173` vs. the live production sites), evaluated on 10³ random draws
requiring exact agreement — it was already run and **vindicated** during the pre-execution
review (line 1335: *"GATE PARITY is additionally vindicated by this review"*), independent of
any run reported here. The driver's own `gate_parity()` function is a **different, informal,
per-run check**: does θ = (0,1) on this run reproduce an **older banked CSV**
(`p3_b0_work/bc_<seed>_work/...`) bit-for-bit? Its docstring says so explicitly ("distinct from
prereg §3.3's ... claim, NOT re-litigated here"). Both are named "GATE PARITY" in this thread's
prose; only §3.3 is one of the eight gates §4.5 names as INSTRUMENT-DEFECT triggers.

Re-derived per seed, per column, against `p3_b0_work/bc_<seed>_work/seed<seed>/simulations/diagnostics/event_likelihoods.csv`:

| seed | n | `ln_L_no_bh` max abs diff | `ln_L_no_bh` max rel diff | `ln_L_with_bh` max abs diff | `ln_L_with_bh` max rel diff | exact? |
|---:|---:|---:|---:|---:|---:|---|
| 900101 | 106 | 5.716e-4 | 4.881e-5 | 0.5420 | 0.05945 | **NO** |
| 900102 | 120 | 2.380e-4 | 4.265e-5 | 0.7735 | 0.08924 | **NO** |
| 900103 | 105 | 2.186e-4 | 3.189e-5 | 0.8841 | 0.11992 | **NO** |
| 900104 | 130 | 1.153e-4 | 2.034e-5 | 0.3114 | 0.03432 | **NO** |

No seed reproduces the banked CSV bit-for-bit. The no-BH residual (max abs ≤ 5.7e-4 nats, max
rel ≤ 4.9e-5) is small and, per `PA-HIER-31(f)`'s F-B finding, **not** a batch-order artifact
(the 9-event smoke and the full registered truth node are bit-identical over their shared
events, so summation order is ruled out) — the live hypotheses remain a code/config delta
between the banked CSV's commit and HEAD, or a process/thread-count effect, undiagnosed as of
this record. **Materiality:** this residual (order 1e-4–1e-5 relative) is roughly 2–4 orders of
magnitude below the per-event score scatter that drives the Z-values above (score_s SEM alone is
0.012, i.e. order 1e-2) — it cannot, by itself, manufacture the 7σ pull in §2.1. It remains an
open, disclosed item bearing on §5.1 invariant 8 (mirror↔production `host_z_error_eff` parity),
not a candidate explanation for the B0-A′ finding. The with-BH residual is much larger
(3–12% relative) — expected, since `combined_with_bh` involves the small-value mass channel this
driver's own docstring already flags as noisier.

### 2.6 A minor, harmless engineering artifact found while reading the raw CSVs (disclosed, not a numerical issue)

`node_b_plus_sites2.2_nosmear/…/event_likelihoods.csv` for seed 900101 contains **212 rows for
106 events** — every `event_idx` appears exactly twice at `h = 0.73`. Checked bit-for-bit:
**the two rows per event are identical in every numeric column** (max abs diff across all
duplicated pairs = 0.0). This is consistent with a restarted/re-invoked `evaluate()` call
appending to an existing CSV rather than truncating it (plausible given the runner-1/-2/-3
retry history in §1) — harmless here because the driver's own `read_event_ln_l()` (and my
independent re-derivation) both de-duplicate on `event_idx` with `keep="last"`, and the kept
value is identical to the dropped one. Flagged because a *future* re-run under different code
that produced genuinely different duplicate values would silently pick "last" without warning —
worth a defensive assertion in the driver if it is touched again, not fixed here (out of this
node's scope).

## 3. Gate → band → verdict chain, read from the registered text

1. **GATE ENG (§3.4): PASS.** ≥10% moved required; measured 98.9% mean at every off-truth node
   (§2.4). "A null from any arm is uninterpretable until GATE ENG passes for that arm" — it
   passes, so the score below is interpretable.
2. **GATE PARITY (§3.3, the registered gate): already PASS/vindicated** at zero-compute review
   (line 1335), not re-litigated by this run. The driver's own separate, informal per-run
   byte-identity check against an older banked CSV (§2.5) fails at a magnitude (≤5.7e-4 nats)
   disclosed as immaterial to the Z-scale statistic and not diagnosed to a root cause — an open
   item, not a §4.5 gate trigger in its own right.
3. **Band B0-A / B0-A′ (§4.1, applied to the registered primary channel `ln_L_no_bh`):**
   `|Z_b| ≤ 3.0` **and** `|Z_s| ≤ 3.0` required for the expected-null band B0-A. Measured
   `|Z_b| = 3.676 > 3.0` **and** `|Z_s| = 7.079 > 3.0` — **either** alone is sufficient to route
   to B0-A′; both do. → **Band = B0-A′.**
4. **Verdict (§4.5, B0-A′ row, quoted verbatim):** *"a non-zero score on a self-consistent venue
   is a bug in the hook, the venue, or GATE PARITY. STOP."* → **INSTRUMENT-DEFECT.**
5. **Routing (§4.5, INSTRUMENT-DEFECT row):** *"if the defect is in a physics-trigger file
   (`bayesian_statistics.py` is one), the fix routes through `/physics-change` with its own gate
   package and ledger row; if it is harness-only (`correspondence_1d.py`), it is an
   `instrumentation` fix, disclosed, with GATE T-ID re-run."* This record does not adjudicate
   which — that diagnosis is out of an independent-reader node's scope — but narrows the search
   space: since `theta_sites="2.2"` and `smear off` mechanically exclude site 2.3, the defect (if
   real and not itself an artifact of the un-audited invariant below) lives in sites 2.1/2.2 (the
   per-host numerator kernel) or in §5.1 invariant 8 (the never-fully-audited mirror↔production
   `host_z_error_eff`/kernel-identity match that `PA-HIER-1`'s correction flagged as
   load-bearing for exactly this venue).
6. **REPORTED-ONLY cap (`PA-HIER-28` item 9, carried without exception):** this verdict, like
   every [HIER] verdict this thread can produce under the current author grant, is
   **REPORTED-ONLY** — it cannot be banked as CALIBRATED or hard-truncation, and it does not
   need to be for a STOP disposition to hold: STOP does not require calibration, only the band
   comparison above.

**Verdict, in one line: Band B0-A′ → INSTRUMENT-DEFECT → STOP (prereg §4.5), REPORTED-ONLY
(PA-HIER-28 item 9).** "Refuted" language does not apply here — this is not a claim being
refuted, it is the registered control failing its own self-consistency check, which is itself a
valued, banked outcome (an honest STOP, not a rescued PASS).

## 4. Caveats — what this record found, and what it explicitly does not settle

- **The open CoR-M contradiction (`PA-HIER-31` REVISION NOTE 2, R1′/R2′) is live and unresolved
  for exactly the run scored here.** `PA-HIER-10` pinned `smear_sigma_z=True` at *every* node,
  *including truth*, for the mirror venue (CoR-M) unconditionally. This P0 run uses
  `smear off` for the whole grid (the `PA-HIER-31(b)` "CoR-P-faithful form", extended to the
  "S0-A remainder" by the same amendment). REVISION NOTE 2 registers this as a **second, distinct
  open contradiction** (separate from the CoR-P one `REVISION NOTE 1` already returned to the
  author), pending its own fresh `[RULE]`. Per R3′'s qualifier, quoted here rather than
  paraphrased: *"The S0-A remainder (P0) 'certifies the instrument' only with respect to
  wiring/arithmetic at `theta_sites='2.2'` ... It does **not** certify site 2.3's behaviour under
  `PA-HIER-10`'s originally-pinned, unconditional `smear_sigma_z=True`-at-every-node CoR-M
  form."* This record's INSTRUMENT-DEFECT finding is scoped identically: it is a defect finding
  about sites 2.1/2.2 under this run's flags, not a full "all sites, smeared" S0-A finding. It
  does **not** retroactively certify or refute the earlier (superseded) `"all"`/smeared partial
  run reported in `B1_1_HIER_RECORD.md` — that run is REPORTED-ONLY/non-CoR-P by
  `PA-HIER-31(b)`'s own disposition and is not re-litigated here.
- **This finding does NOT license a Stage-P or Stage-F launch, an S0-B launch, or a C1/C3
  build.** GATE SEQ (§3.7) and the standing LAUNCH-BLOCKED state (last cleared item: the item-4
  certifications, `PA-HIER-30`) are unaffected by a STOP at Stage 0 — a STOP is a terminus for
  *this* arm, not a green light for the next one. The registered next action on an
  INSTRUMENT-DEFECT verdict is diagnosis (§4.5's routing, §3 above), which is itself gated by
  `/physics-change` if it touches `bayesian_statistics.py`, per CLAUDE.md's hard gate — this
  record does not perform that diagnosis, propose a fix, or edit any code.
- **S0-R was not run this session** (disclosed, out of scope for this node — S0-R is
  FALLBACK/DISARMED per `PA-HIER-28` item 5 in any case and was never going to be verdict-bearing
  regardless of its outcome).
- **KW-Q1** (`kwq1_registered_run/`) is a separate, currently-running local process on the same
  driver, explicitly excluded from this node's scope by the task instructions ("do not touch
  it"). It inherits both the REPORTED-ONLY cap and the open CoR-M contradiction disclosed above,
  since it uses the same `theta_sites="2.2"`/`smear off` flags (`runner3_wave2pre.sh`) — this is
  disclosed here for whoever reads KW-Q1's own output next, not adjudicated.
- **This is not a re-litigation.** Checked `EXONERATION_REGISTER_20260827.md` and
  `gate_b_20260730/BIAS_HISTORY_LEDGER.md` §2 "DO NOT RE-TRY" for the θ-hook /
  host-z-kernel-misspecification / `smear_global_selection` mechanism: no match. This measurement
  and its STOP verdict are new information about an open thread, not a retry of a closed one.
- **"INSTRUMENT-DEFECT" is the registered, valued outcome here — it is not being softened,
  argued around, or treated as a disappointing result to explain away.** Per the standing rule
  (CLAUDE.md, this node's own instructions): refuted/undetermined verdicts are as valuable as
  confirmations, and no verdict is rescued.

## 5. Compute ledger contribution

P0 (this run): 2959.6 s wall at 14 cpu ≈ **11.51 CPU-h** (`14 × 2959.6 / 3600`), `--jobs 1`,
run-of-record (runner-3). S0-C (seed 900101, 41-h grid, 12 cpu): 3125.1 s wall ≈ **10.42 CPU-h**
(`12 × 3125.1 / 3600`); measured mean marginal per-h cost **24.37 s** (41 h-nodes, 2681.0 s of
per-h sweep after a one-time 1704.3 s table-build/first-h cost); full result in
`hier_s0_registered_run/s0c_full_output.json`. Both figures are measured wall-time × allocated
cpu, not the §7.1 anchor's assumed proportional-speedup model (already flagged as invalid for
theta-engaged/smeared cells in `B1_1_HIER_RECORD.md` §1 item 5 — moot for this record's own
S0-A run since `smear off`, but still the right caution for anyone re-costing S0-C against a
smeared cell).
