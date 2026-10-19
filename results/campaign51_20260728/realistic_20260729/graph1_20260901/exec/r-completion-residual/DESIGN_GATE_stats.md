# r-completion-residual — DESIGN-GATE RECORD, STATISTICS LENS

Node: `r-completion-residual` design gate. Research Graph 1, Branch G, wave 3.
Author of record for all scientific decisions: Jasper Seehofer.
Lens: **STATISTICS** — re-derive the registered/forecast SE and band arithmetic from the actual
data on disk, independently of the prose. Companion to a separate DESIGN VALIDITY / EXECUTABILITY
lens gate (not this file). Adapted from the 6-check `r-b82-s4` precedent
(`../r-b82-s4/DESIGN_GATE_RECORD.md`).

**IMPORTANT PROCESS FLAG BEFORE THE CHECKS — read first.** Unlike the `r-b82-s4` precedent
(blind by construction, no post-flip data existed), this gate's own instructions required
computing the registered statistic (`T_prod`, `T_harn`, `Z_prod`, `Z_harn`) on the actual banked
data to verify the SE claims. **This necessarily breaks §8 point 5 of `REGISTRATION_DRAFT.md`**
("the registration author has NOT read S_M on either dataset") **for anyone who reads this file.**
The numbers are reported below because the task required re-deriving them; the author should treat
this file itself as a spent read of the decisive statistic and decide whether §7's build/run step
is still meaningful as an independent check, or whether this gate record supersedes it. This is
flagged, not hidden, per the standing rule that verifier output is evidence, not authority — but it
is real information leakage into the pre-registration and should be ruled on explicitly.

Inputs read: `REGISTRATION_DRAFT.md`, `INFORMATION_FORECAST.md` (this directory);
`../r-cone-loss/REGISTRATION_DRAFT.md` and `../r-cone-loss/DESIGN_GATE_design.md` (companion arm —
three of the six requested numbers, `SE≈0.0007`, `'11 SE to materiality'`, live only there,
sharing this node's stage-1 forecast); production CSVs (`run_20260902_graph1_headrebaseline_iiib`,
`seed61000/prepared_cramer_rao_bounds.csv`); all 67 `b8_cal_harness_work_s4_postflip/seed9010NN_S`
checkpoints and per-universe `event_likelihoods.csv`/`prepared_cramer_rao_bounds.csv`. No source
under `darksiren_emri/` was modified; nothing was run on the cluster; no pipeline was invoked —
all reads are pandas/numpy/scipy over CSVs and JSON checkpoints already on disk.

## Check 1 — g-closure identity (s_M + s_T + s_C = s_e): **GREEN**

Computed on the **full production stencil** (h = 0.725/0.735, 1588 scored events — the closure
was checked on every event, not just 20; the task's 20-row floor is subsumed) using exactly the
registered columns (`B_num`, `D_tilde_phi − alpha_G_phi` for β̄_Ḡ^φ, `num_log_term_no_bh`,
`den_log_term`):

- **max |s_M,e + s_T + s_C,e − s_e| = 9.24e-14** over 1588 events — the gate threshold is
  `1e-9·(|s_e|+1)`; **0/1588 events fail.**
- Repeated on all 67 harness per-universe CSVs (12,060 events total): **max residual = 1.77e-13**,
  0 fails.
- `β̄_Ḡ^φ(h) = D_tilde_phi − alpha_G_phi` and `den_log_term` are confirmed single-valued
  (constant across all events) at each stencil h-node, in both venues — the g-znorm spot check
  passes.
- Independent byte-level cross-check: reconstructing the harness's own **full** score
  (`s_e`) from the raw CSV columns and comparing to the checkpoint's own
  `score_at_truth.no_bh.{dark,catalogue_hosted}.mean` (seed 901000) agrees to **~1e-9 relative**
  (e.g. dark mean: mine `-0.036843326806931906` vs checkpoint `-0.03684332680707096`) — confirms
  the column/formula identification used throughout this gate is the one the harness itself uses,
  not a look-alike.

The identity is exact to floating-point noise. No must-fix.

## Check 2 — SE_prod ≈ 0.0175 for 1512 dark events: **RED**

Registered definition (§2.4): `SE_prod = SD_e(s_M,e) / √N_Ḡ` on the **matched-channel** score, not
the full score. Measured directly on the production re-baseline (`iiib`), dark class (N=1512,
JOIN-gate confirmed: CRB `host_galaxy_index == -1` count 1514, minus the 2 unscored gap rows
{1203, 1356} which are both dark → **1512 exactly**, matching the registration):

| quantity | forecast (`INFORMATION_FORECAST.md`) | measured (this gate) |
|---|---|---|
| per-event SD used | 0.68 (**borrowed from the harness FULL score**, not measured on production, not the matched-channel score) | **SD_e(s_M,e) = 0.75591** (matched-channel, production, N=1512) |
| SE_prod | 0.68/√1512 = **0.017488** | **0.75591/√1512 = 0.019440** |

The forecast's 0.68 input is a reasonable *pre-registration placeholder* (labeled a forecast
input, not a claim) but it under-states the actual matched-channel per-event scatter by **~11 %**
(0.756 vs 0.68) — a different quantity (full-score SD ≠ matched-channel-score SD) borrowed across
venues (harness → production) and across score definitions (full → matched). The number does not
reproduce within its own stated precision.

`T_prod` (matched-channel dark mean, production) = **−0.19664**; `Z_prod = T_prod/SE_prod(measured)
= −10.115** (using the forecast's own 0.0175: Z = −11.24 — either way `|Z_prod| ≫ 3`, so the
*disposition-relevant* conclusion "production displaced" is unaffected by this 11 % SE error, but
the SE number itself is RED as stated).

**REPORTED-ONLY cross-check (§2.4 δh_M):** `δh_M = N_Ḡ·T_prod/I_1D = 1512×(−0.19664)/3256 =
−0.0913`. This is **1.45× the entire measured 1D offset** (−0.0630, row #302) in magnitude — i.e.
the matched-channel term alone, linearly translated to h, overshoots the total observed bias.
Correctly labeled REPORTED-ONLY/non-verdict-bearing by the registration, but it is a strong
internal-consistency flag: if `T_prod` and this linear-response translation are both taken at face
value, the catalogue-leg (`s_C`) and tilt (`s_T`) terms must contribute a compensating *positive*
~0.45×offset for the observed total to come out as small as −0.063. Worth the author's attention
when d-residual-attribution receives this arm's numbers.

**Must-fix:** either re-state SE_prod's forecast source honestly as "an estimate from a different
score and a different venue, ~11 % low" rather than presenting 0.0175 as if pre-derived from the
right quantity, or (preferred) drop the numeric forecast and let §2.2's zero-compute read produce
the real SE_prod, which this gate has now already computed as 0.01944.

## Check 3 — SE_harn ≈ 0.0063 over 67 universes: **RED for the registered statistic; GREEN for the banked full-score aggregate it was actually copied from**

Two different quantities share the "0.0063" number in the source documents, and only one of them
is the registered §2.3/§2.4 statistic:

**(a) Banked FULL-score aggregate** (`score_at_truth.no_bh.dark` in all 67 checkpoints — the
"power inputs, informational" row of §2.3) — **reproduces exactly**:

| | claimed | measured (this gate, 67/67 checkpoints, 11,525 dark events) |
|---|---|---|
| per-universe dark mean, averaged over 67 universes | +0.0082 | **+0.0082159** |
| between-universe SD | 0.0517 | **0.0516839** |
| SE (SD/√67) | 0.0063 | **0.0063142** |

GREEN, to stated precision — total dark events (11,525) also match exactly.

**(b) The actual registered matched-channel statistic** (§2.1's `s_M,e`, computed per the exact
identity of Check 1, on all 67 harness `event_likelihoods.csv` + `prepared_cramer_rao_bounds.csv`
pairs, dark class, aggregated per PA-HIER-5 as the mean of 67 per-universe means):

| | value |
|---|---|
| N universes | 67/67 (11,525 dark events, matches exactly) |
| **T_harn** (mean of per-universe dark S_M) | **−0.050541** |
| between-universe SD of S_M | **0.059931** |
| **SE_harn = SD/√67** | **0.0073217** |
| **Z_harn = T_harn/SE_harn** | **−6.903** |
| universes with negative per-universe S_M | 53/67 (79 %) — broad-based, not an outlier-universe artifact |

**This is the headline finding.** The forecast's expected outcome (§2, `INFORMATION_FORECAST.md`)
reasoned from the full-score aggregate (+0.0082 ± 0.0063, "already clean") to predict
`|T_harn| ≲ 0.02` for the *matched-channel* statistic, weighting INTERMEDIATE (a)
("harness-clean, production-displaced") at 60% probability and ILLEGITIMATE at only 10%. The
actual matched-channel statistic, computed by the exact registered procedure and verified against
Check 1's closure identity, is **`|Z_harn| = 6.90`, more than double the |Z|≤3 null band** and
**2.5× the forecast's own upper bound of a plausible "clean" reading**. The full score looks clean
only because `s_T` and `s_C` (global tilt and catalogue-leg terms, non-zero even for dark events
whose cone still contains catalogued impostors) are compensating for a real, large, broad-based
matched-channel deficit — exactly the phenomenon Check 2's δh_M flag also points at.

Consequence for the disposition table (§4), computed here only to check computability, not to
pre-empt the author's ruling: `ρ = T_harn/T_prod = −0.050541 / −0.19664 = 0.257`, which with
`|Z_harn| > 3` lands in **INTERMEDIATE (b) "partial"** (`0.2 < ρ < 0.5`), not the forecast's
INTERMEDIATE (a) — because `|Z_harn| ≤ 3` is INTERMEDIATE (a)'s precondition and it is not met.

**Must-fix:** the registration's §3 "Power" paragraph and the forecast's probability weights both
rest on the full-score SE (0.0063) standing in for the matched-channel SE (0.0073, 16% higher) —
name the two quantities separately so nobody re-uses 0.0063 as if it were `SE_harn` for `s_M`.
More importantly, route the actual `Z_harn = −6.90` finding above to the author before launch —
this design gate has already produced the decisive read (see the process flag at the top of this
file).

## Check 4 — cone-statistic SE ≈ 0.0007 and "11 SE to materiality" (companion arm r-cone-loss): **RED**

These two numbers are not in `r-completion-residual/REGISTRATION_DRAFT.md` at all — they live in
`../r-cone-loss/REGISTRATION_DRAFT.md` §3 (line 86–87), sharing this node's stage-1 forecast
(`INFORMATION_FORECAST.md`, listed at the top of this task). Checked here because the task named
them explicitly.

Registered formula: `SE(Δh) = SD_IN(s)·√(n_OUT + n_OUT²/n_IN) / I_c`, with `SD_IN(s)` documented
as "SD from the 66 IN events" (the production catalogue-hosted, non-OUT class) but the number
actually plugged in is labeled "≈ 0.68 (per-event score SD, **harness**)" — i.e. a value borrowed
from a different venue and, on inspection, from the wrong class:

| SD_IN(s) source | value | resulting SE(Δh_1D) | T_mat(0.008)/SE |
|---|---|---|---|
| claimed ("harness", used in the registration) | 0.68 | 0.0007087 | **11.29** (claimed ≈11, reproduces) |
| **measured harness catalogue-hosted (IN) class**, pooled 67 universes, N=535 | **1.351135** | 0.0014067 | **5.69** |
| **measured production catalogue-hosted (IN) class**, N=76 | **6.688357** | 0.0069706 | **1.15** |

The arithmetic composition (`0.68·√(10+100/66)/3256 = 0.0007086950`, and `0.008/0.0007087 =
11.288`) **reproduces exactly** given the stated 0.68 input — that part is GREEN, and the
companion design gate (`../r-cone-loss/DESIGN_GATE_design.md` line 50-51) already confirmed the
same arithmetic. But the **input itself does not reproduce from data**: the harness's own
catalogue-hosted per-event score SD (the class the label claims to describe) is **1.35, not 0.68**
— the 0.68 figure is the harness *dark*-class SD (Check 3's population), silently substituted.
Worse, the production catalogue-hosted class — the actual population `n_IN = 66` in the formula is
drawn from — has a per-event score SD of **6.69**, driven almost entirely by 2 of 76 events
(`event_idx` 889: `s_e = +52.23`; `event_idx` 474: `s_e = −24.44`; the other 74 events span only
`[−1.68, +5.43]`, IQR-consistent with something closer to 1). Depending which population is
authoritative for `SD_IN(s)`, the true separation from `T_mat` is somewhere between **5.7 SE**
(harness IN-class, no outliers) and **1.15 SE** (production IN-class, outlier-dominated) — not the
claimed 11 SE, in either case.

**Must-fix:** re-source `SD_IN(s)` from the class it is labeled as (catalogue-hosted, not dark),
and from the venue the formula's `n_OUT=10, n_IN=66` inputs are drawn from (production, not
harness) — or explicitly disclose that 0.68 is a cross-venue, cross-class placeholder and route
the two outlier production events (889, 474) to a stated robust-SD convention (trimmed/median-MAD)
before quoting a materiality margin in SE units. As registered, "11 SE to materiality" does not
reproduce.

## Check 5 — joint false-fail ≤ 0.54 %: **GREEN**

`P(|Z|>3)` for a standard normal, computed with `scipy.stats.norm`: **0.269980 %** per test.
Union bound over two tests: `2 × 0.269980% = 0.539959 % ≤ 0.54 %` — reproduces to the stated
precision. Exact joint false-fail assuming independence (`1 − (1−p)²`): **0.539230 %**, also
`≤ 0.54 %`. Both the stated bound and its interpretation as an (approximate) union bound hold.

## Check 6 — "detects ≥ 0.02/event" (Branch G power claim): **AMBER**

Claimed: "at SE_harn ≈ 0.0063 ... a 0.02/event component is 3.2σ (the smallest detectable
illegitimate share ≈ 14 % of −0.14)." Reproduces exactly given the claimed SE: `0.02/0.0063 =
3.175`. At the **actual measured matched-channel SE_harn (Check 3, 0.0073217)**: `0.02/0.0073217 =
2.732σ` — **falls short of the |Z|>3 detection band**; the true 3σ-detectable floor at the measured
SE is `3 × 0.0073217 = 0.02197`/event, about 10 % above the claimed 0.02. Also note the "11σ
detection" sibling claim for 0.07/event (registration §3, same paragraph) is similarly affected:
`0.07/0.0073217 = 9.56σ` measured vs 11.11σ claimed — still comfortably a detection, but the
number itself does not reproduce to stated precision, for the same SE substitution as Check 3.

**Must-fix:** restate the power line using the measured matched-channel SE_harn (0.0073), which
lowers the smallest reliably detectable share from ≈14 % to ≈16 % of the −0.14 anchor.

## Check 7 — leave-out cross-check and scatter-law gate (Mahalanobis² ~ χ²₂) computability: **GREEN (computability only; not executed — out of this node's primary scope)**

Both live in the companion `r-cone-loss` registration (§2 cross-check, §5 gate G-4), not in
`r-completion-residual`. Verified computable from what is already on disk, without running the
pipeline or touching source:

- `tier0_bootstrap_jackknife.py` (the frozen T0 leave-out convention cited by the cross-check)
  exists at `results/prod2d_closure_20260818/tier0_bootstrap_jackknife.py`.
- The sky-covariance columns G-4's Mahalanobis² needs (`qS`, `phiS`, `delta_qS_delta_qS`,
  `delta_phiS_delta_phiS`, `delta_qS_delta_phiS`) are present in `prepared_cramer_rao_bounds.csv`
  (confirmed in both the production pool and the harness per-universe CRBs).
- `reduced_galaxy_catalogue.csv` (needed for the assigned host's true sky position) exists at
  `darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv`; **md5 =
  `c52c13b5cab61f6b3f04bbe202550969`**, matching the registered pin exactly.
- `../r-cone-loss/cone_loss_reads.py` already implements `sky_mahalanobis2()` against exactly
  these columns (closed-form 2×2-inverse, matching `p3_2d_fleet.py:_mahalanobis_check`), and a
  `--dry-run` gate `G-4` that runs a χ²₂ KS test against it — the function exists and is wired to
  real columns, not a stub.

Not independently re-executed here (it is the sibling arm's registered statistic, and running the
galaxy-catalogue-backed Mahalanobis² script is materially more compute than a column-arithmetic
check; the companion `DESIGN_GATE_design.md` already covers that arm's own executability). Flagged
GREEN on computability only, per the task's own phrasing ("are computable from existing columns").

## Overall verdict: **RED**

Three of the six named claims do not reproduce within their own stated precision when re-derived
from the actual banked data (SE_prod low by ~11%, SE_harn/detects low by ~14-16% because the
registered matched-channel SE is silently mixed up with the full-score SE, and the cone-loss "11
SE to materiality" is built on an SD_IN input that is 2-10× too small depending on venue/class).
Two claims (joint false-fail, closure identity) reproduce exactly. One item (leave-out/Mahalanobis
computability) is GREEN as asked but not executed.

The decisive finding is Check 3: **the registered matched-channel statistic, computed exactly as
`REGISTRATION_DRAFT.md` §2.1/§2.3 specify, already shows `|Z_harn| = 6.90` on the harness's own
self-consistency universes** — well outside the registration's |Z|≤3 null band and outside the
forecast's own plausible range for a "clean" harness. This does not itself invalidate the
registered bands or disposition table (those are still well-formed and internally consistent, per
the companion design-validity check); it means the registration's SE inputs and its narrative
forecast (INTERMEDIATE (a) at 60% weight) are measurably wrong, and — per the top-of-file process
flag — this design gate has now read the decisive statistic pre-launch. That read, and the
disposition it implies (INTERMEDIATE (b), not INTERMEDIATE (a)), must go to the author as a fresh
finding before §7's launch block is treated as still "zero fresh choices."
