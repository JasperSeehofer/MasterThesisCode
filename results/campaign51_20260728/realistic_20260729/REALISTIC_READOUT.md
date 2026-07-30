# Campaign #53 — realistic host-observation run, readout (2026-07-29)

Scoring of the 2 truth seeds × 5 observation realizations submitted 2026-07-29
(jobs 6092512–6092531; `absolute_marginal` × `volume_deconv`; observed
catalogues `realizations_20260729/observed_catalogue_seed90000{1..5}.csv`)
against the predictions registered **before** evaluation in
`docs/derivations/realistic_host_observation_model.md` §8.

All numbers produced by `score_realistic.py` (this directory) from the cluster
copies in `seed{61000,62000}/real_r{1..5}/`; canonical originals on `$WS`.
Every job completed: 14 job groups, 41/41 h-points each, zero failed tasks.

`h_true = 0.73`. The h-grid is 0.005-spaced over [0.655, 0.79], so MAPs carry
±0.0025 quantization — small against the measured σ_h ≈ 0.02, but not against
the realization-to-realization scatter (see §3).

## 1. Per-run results (1D channel, `posteriors`)

| seed | real | MAP h | mean h | σ_H0 [km/s/Mpc] | pull | 68% interval |
|---|---|---|---|---|---|---|
| 61000 | r1 | 0.740 | 0.7321 | 1.98 | +0.51 | [0.713, 0.751] |
| 61000 | r2 | 0.725 | 0.7256 | 1.79 | −0.28 | [0.708, 0.744] |
| 61000 | r3 | 0.730 | 0.7267 | 2.01 | 0.00 | [0.707, 0.747] |
| 61000 | r4 | 0.725 | 0.7271 | 1.54 | −0.33 | [0.712, 0.743] |
| 61000 | r5 | 0.740 | 0.7351 | 2.24 | +0.45 | [0.712, 0.758] |
| 62000 | r1 | 0.715 | 0.7201 | 2.24 | −0.67 | [0.699, 0.743] |
| 62000 | r2 | 0.700 | 0.6989 | 2.61 | −1.15 | [0.672, 0.725] |
| 62000 | r3 | 0.710 | 0.7082 | 2.41 | −0.83 | [0.684, 0.733] |
| 62000 | r4 | 0.710 | 0.7110 | 2.06 | −0.97 | [0.691, 0.731] |
| 62000 | r5 | 0.710 | 0.7043 | 2.64 | −0.76 | [0.677, 0.731] |

Per-seed means: **0.7320** (61000), **0.7090** (62000). Pooled mean 0.7205.

## 2. Pre-registered scorecard

| | prediction | measured | verdict |
|---|---|---|---|
| **P1** | σ_H0 = 1.3–1.7 expected; per-seed range [0.5, 4.0]; **< 0.3 falsifies** | 1.54–2.64 (seed means 1.91 / 2.39) | **not falsified, expectation missed high** — 1/10 runs inside the central band, all 10 inside the admissible range, none anywhere near the 0.3 leak threshold |
| **P2** | pulls N(0,1)-consistent; \|pull\| > 2 in ≥ 6 of 10 falsifies | max \|pull\| = 1.15; mean −0.40, sd 0.58; 0/10 exceed 2 | **PASS** |
| **P3** | in-catalogue retains ≈ 100% of curvature; dark contribution in [−5%, +5%] | dark = +35% to +140% of the signed total | **PREDICTION MISSED** (see §4) |
| **P4** | 3 golden events each lose ≥ 95% of their curvature | retained 0.045% (seed 61000), 0.0013% (seed 62000) | **PASS**, by ~3 orders of magnitude |
| **P5** | σ→0 bit-identity — hard pass/fail | control md5 `1e81ba22` ≡ seed-61000 baseline md5 `1e81ba22` | **PASS** |
| **P6** | host-loss 10–30% photo, ≲5% spec; ~0% falsifies the plumbing | — | **NOT SCORABLE** (see §5) |

P1's forecast band was optimistic: the seed means (1.91, 2.39) sit 12% and 41%
above the top of the expected band, and 9/10 individual runs are wider than
predicted, none narrower. Since the falsification direction was *too narrow*
(a leak of the unscattered premise), missing high is the safe direction — the
noise realization is doing at least as much damage as budgeted, not less.

## 3. What the spread actually decomposes into — bears on [RATIFY-R7]

Pairing by realization index (the same five observed catalogues are applied to
both truth seeds) separates the two variance components:

| | r1 | r2 | r3 | r4 | r5 | mean | sd |
|---|---|---|---|---|---|---|---|
| paired Δ (62000 − 61000) | −0.025 | −0.025 | −0.020 | −0.015 | −0.030 | **−0.023** | 0.006 |

- **Realization-level scatter** (within a seed): sd = 0.0076 (61000), 0.0055
  (62000) in h — i.e. σ_H0-equivalent 0.55–0.76 km/s/Mpc.
- **Truth-seed–level difference**: −0.023 in h, ≈ 2.3 km/s/Mpc.
- **Per-run posterior width**: σ_h ≈ 0.019–0.026.

So the observation realization contributes ~3–4× less scatter than the choice
of truth universe, and the seed-to-seed difference is ≈ 1σ of a single run's
own posterior width — exactly what an honest σ predicts for two independent
universes. **The −0.023 offset is therefore consistent with statistical noise;
it is not evidence of a per-seed bias.** With only two universes it cannot be
anything more definite than that.

The consequence for the deferred decision is directional and, unusually, clean:
**more truth seeds buy a stable headline; more realizations per seed do not.**
Five realizations already pin the realization-level term to sd ≈ 0.006, while
the universe-level term — the one that sets the headline's uncertainty — rests
on a sample of two. Note this inverts the assumption behind the deferral
(§ RATIFY-R7 expected the forecast to be Poisson-dominated in the *spectroscopic
host count*, a realization-level effect); the measured realization spread is
the *small* component. The decision remains the author's.

Note also that pull sd = 0.58 (< 1) must **not** be read as over-conservative
σ: the ten runs share two truth universes, so the pulls are strongly
correlated within a seed and 10 correlated draws cannot estimate a pull
dispersion. Two effective degrees of freedom, not ten.

## 4. P3 — where the information now comes from

Signed 3-point curvature at h ∈ {0.725, 0.73, 0.735}, decomposed by
`host_galaxy_index ≥ 0`:

| seed | in-catalogue | dark | signed total | implied σ_h |
|---|---|---|---|---|
| 61000 r1–r5 | +0.092 … −0.014 | **+0.049 ± 0.000** | 0.035–0.141 | 0.013–0.027 |
| 62000 r1–r5 | +0.031 … −0.004 | **+0.047 ± 0.001** | 0.043–0.077 | 0.018–0.024 |

Two things are worth separating here.

**The prediction is genuinely missed.** Under the idealized stack the dark
events carried −1% of the curvature; they now carry more than the in-catalogue
events do, and in two runs the in-catalogue contribution goes *negative*. The
golden in-catalogue carriers lost ~3 orders of magnitude of information (P4)
while the completion term did not, so the balance inverted. This is a real regime change,
not a bookkeeping artifact, and it means the realistic headline is substantially
a *completion-term* measurement. Given that issue #23 (completion-term realism)
is an open paper-blocker, that is the single most consequential number in this
readout.

**But the percentages are ill-conditioned and should not be quoted.** The
signed total (mean 0.076) is only 62% of the absolute curvature mass (mean
0.123), so shares are cancellation-dominated — that is why "dark share" reaches
140% and one run's golden share goes to −159%. Quote the signed sums
(in-catalogue ≈ 0…0.09, dark ≈ 0.048), never the ratios.

The dark contribution is near-constant across all ten runs (+0.047…+0.049),
which is the expected signature: the completion term depends on the *catalogue's*
completeness, not on which particular hosts got scattered where. The
realization-dependent piece lives entirely in the in-catalogue column, and that
column is what drives the run-to-run MAP scatter of §3.

Cross-check: σ_h from the curvature (0.013–0.027) agrees with the posterior
moment σ_h (0.019–0.026) run by run, so the small totals are not a numerical
pathology — the posterior really is that wide now.

## 5. P6 — not scorable, and that is a gap

The prediction requires the ball-tree host-miss rate, and **no such counter is
written by the evaluate path** — the logs (`ev_*.out`) contain no candidate/miss
accounting, and the per-h JSONs are `{event_index: [likelihood]}` only. The
prediction's own falsification clause ("a ~0% measured miss rate ... falsifies
the observed-catalogue plumbing") therefore cannot be exercised on the delivered
artifacts.

This does not invalidate the run — P5's bit-identity gate independently shows
the realization plumbing is inert at σ→0, and P4's 3-orders-of-magnitude
demotion shows the scatter is reaching the likelihood. But P6 was adopted as
part of the acceptance set, and it is currently unmeasured. Closing it needs a
counter in `get_possible_hosts` and a re-run (cheap: CPU-only, and the pool is
reusable), or an explicit author decision to retire it.

## 6. ⚠️ Unregistered finding — the 2D channel is biased high in every run

Not part of P1–P6, and the more serious result of the two.

| | 1D (`posteriors`) | 2D (`posteriors_with_bh_mass`) |
|---|---|---|
| MAP range | 0.700–0.740 | **0.780–0.820** |
| pull vs truth | −1.15 … +0.51 | **+3.4 … +4.5 (mean +4.04)** |
| runs with \|pull\| > 2 | 0/10 | **10/10** |

σ_H0 in the 2D channel (1.4–2.5) is comparable to the 1D channel, so the
offset is a genuine ~+0.07–0.09 shift in h, not a width artifact. Maximum
edge/peak posterior ratio is 0.17, so the posteriors are not railing against
the grid boundary — the peak is interior and simply in the wrong place.

The 1D and 2D channels differ only by the host BH-mass dimension, and the
realistic model is the first run in which the host BH mass is *realized* as
noise (0.24 dex) rather than used as a width. A mismatch between the realized
scatter and the mass kernel would produce exactly this: a channel-specific,
realization-independent shift. This is plausibly the same defect as the open
`(d2)` thread (selection-side M scatter/truncation, the standing owner of the
≈ +23 ln 2D residual) — now showing up at +4σ instead of as a log-likelihood
residual.

~~**Recommendation: do not quote a 2D headline from this run.** The 1D channel is
the defensible one.~~ Diagnosing the mass-kernel/scatter pairing should precede
any use of the with-BH-mass posteriors.

**[AMENDED 2026-07-30, Gate B/C adjudication]** Do not quote a headline from
**EITHER** channel of this run. The 2D channel is biased **+4σ** (see the CLAIM
file's C3 / C4-amended); the 1D channel's apparent unbiasedness is a **crossing
of two opposing displaced class profiles** (in-cat argmax **0.86**, dark **0.64**
in 10/10 runs), with per-event in-catalogue information destroyed (C5, FINDING),
and the crossing point is contingent on the **mis-calibrated mixture weight**
(C9: w_G model **0.1215** vs realized **0.0523**, binomial **z = −11.86**).
Diagnosis and the jointly-derived mixture fix precede any headline.
`gate_b_20260730/ADJUDICATION_20260730.md` is the record.

## 7. Status and what is not in here

- The **0.67 closure seed** (`run_20260729_seed64000_h0p67`, jobs 6090909–6090912)
  was still running at the time of writing; its row in
  `IDEALIZED_BASELINE_READOUT.md` remains empty. Note its GPU simulate array
  ends each task by 30-min wall-clock TIMEOUT on the `*_short` partitions,
  yielding ~10 of 40 requested steps per task (rows are flushed, so the data is
  lossy rather than lost) — worth confirming the event count is sufficient
  before scoring the closure.
- **No zoom grid was run** on the realistic posteriors. σ here comes from the
  0.005-spaced production grid, which resolves σ_h ≈ 0.02 comfortably; the
  MAP quantization (±0.0025) is however about half the realization-level
  scatter, so §3's realization sd is coarse.
- P1–P6 were scored as written. Where a prediction proved ill-posed against the
  delivered data (P3's ratios, P6's missing counter), that is recorded above
  rather than reinterpreted into a pass.
