# Mechanism-Indexed Exoneration Register — 2026-08-27

**Purpose.** A single grep target for a stage-0 rule-1 exoneration check. Every entry is indexed by
the PHYSICAL/STATISTICAL MECHANISM in plain words a fresh investigator would naturally use, not by
the tag or thread name the filing session happened to pick. **A future check must search this file
for the MECHANISM it is proposing, not for its own thread's vocabulary.**

**Why this file exists.** On 2026-08-27, a stage-0 rule-1 check on thread [WGEO] reported "PASSED"
on a mechanism — "hard mass window as support truncation" (tag **HB**) — that had already been
measured and refuted four weeks earlier. HB's entry sat **two lines below** the entries the checking
agent had quoted (`CLAIM_2D_BIAS_20260730.md:732-734`, immediately after the entries ending
`:727`). It was missed because the agent grepped for its own thread's objects ("candidate-window
membership", "mass-kernel family") and stopped, rather than reading every entry in the list. See the
**[WGEO]/HB near-miss** in §6 for the full anatomy of that failure — it is preserved here as a
worked example, not just a warning.

**How to use this register.**
1. State your candidate mechanism in your own words.
2. `grep -i` this file for every synonym you can think of, including words your mechanism does NOT
   use but a filing session might have (a hard cut, a floor, a filter, a window, a truncation, a
   clamp, an eligibility test — these five terms *can denote the same object*).
3. Read every matched entry in full, including its "WHAT IT DOES NOT COVER" field — a matched
   keyword does not mean your claim collides; the delimitation field is where you check that.
4. If your mechanism engages an entry's covered claim without new evidence that meets the entry's
   own falsifier, **do not open the thread** — cite the entry and stop (as [WGEO] correctly did for
   HB, one investigation late).
5. If you bank a new exoneration, add it here with the same six fields used throughout, and re-file
   any thread whose delimitation field this entry would now change.

**Scope of the sweep and what was NOT re-verified by execution.** This register is a records
compilation. Every BOUND quoted below is copied from its source, not recomputed; verification (§5)
checked only that the cited anchor still exists in the file at the location referenced, not that
the historical number itself is correct. Sources swept are listed in §0.

---

## 0. Sources swept

- `gate_b_20260730/BIAS_HISTORY_LEDGER.md` — §1 chronological table (entries #1–#98, read in full),
  §2 consolidated exoneration list (read in full), §3–§5, and every row #100–#211 (read in full via
  section-header scan + full-text keyword sweep for `exonerat|refuted|REFUTED|ruled out|bounded
  at|too small|sign-invert|closed`).
- `CLAIM_2D_BIAS_20260730.md` — read in full (858 lines), including the `## Exonerated` list
  (`:721–734`) through its two appendices (`:736–757`) and the `## Errors made this session`
  section (`:758–859`).
- `HANDOFF_20260730.md` — read in full (193 lines).
- `gate_b_20260730/ADJUDICATION_20260730.md` — per-claim adjudication section, cross-report
  reconciliation, and claim-file edit list (§1, §2, §6) read; C1–C11 verdicts cross-checked against
  their landed form in `CLAIM_2D_BIAS_20260730.md`.
- `CLAIM_WGEO_20260827.md` — read in full (553 lines): the near-miss case itself.
- Every other `CLAIM_*.md` in this directory (`CLAIM_P3_MKER_20260826.md`,
  `CLAIM_P3_IMPOSTOR_CONVENTION_20260822.md`, `CLAIM_B0_FINITE_MOMENT_20260824.md`,
  `CLAIM_P3_2D_20260825.md`, `CLAIM_PRODUCTION_CALIBRATION_HARNESS_20260817.md`,
  `CLAIM_SYMMETRIC_SELECTION_INSERTION_20260818.md`, `CLAIM_P3_RPHI_20260822.md`,
  `CLAIM_F0_SEL_20260825.md`, `CLAIM_P3_WBHZERO_20260825.md`,
  `CLAIM_D1_P0WINDOW_20260805.md`) — grepped for exoneration-list content and their own rule-1
  delimitation sections read.
- `m2_residual_owner/CLAIM_M2_RESIDUAL_OWNER_20260807.md` — full two-layer exoneration check (§2)
  and local `## Exonerated` list (`:477–487`) read.
- `gate_b_20260730/README.md`, `README_C8.md`, `README_gateC_1_4_wG.md`, `C7_README.md`,
  `README_c5_c4_attack.md`, `c3c4_allruns_summary.md` — grepped and spot-read for exoneration
  content not already carried by the ledger/claim file.
- Repo-wide `grep -rn` (this directory tree) for `EXONERAT` (case-insensitive) to catch any
  standing exoneration outside the primary two documents; every hit traced to its source.
- `wgeo_s0_coupling_20260827.md`, `hier_provenance_stamps_20260826.md`,
  `STUCK_P3_2D_SYMPTOM_CARD_20260826.md` — read for rule-1-check methodology and cross-references.

**Not exhaustively read line-by-line:** the `seed*/real_r*/posteriors/*.md` run artifacts (these are
raw run outputs, not claim/exoneration documents — grepped only, no exoneration content found beyond
the already-covered HC "physics-floor" object).

---

## 1. PRIMARY 2D-BIAS EXONERATIONS — `CLAIM_2D_BIAS_20260730.md` list (binding, layer 1)

These are the entries at `CLAIM_2D_BIAS_20260730.md:721–734` (list) plus its two 2026-07-30
appendices (`:736–757`). This is **layer 1** of the two-layer rule-1 check; layer 2 is §2 below.
**The two layers are a UNION — checking only one is the exact failure mode this register exists to
prevent.**

### [JACOBIAN] — catalogue distance-redshift Jacobian omission

- **MECHANISM (plain terms):** a missing `dd_L/dz` Jacobian factor in the galaxy-catalogue redshift
  integral. Search also: "Jacobian", "distance-redshift derivative", "missing volume element in the
  catalogue integral".
- **BOUND:** REFUTED as a cause — the factor is ∝1/h and *decreasing*, which biases h **low**, wrong
  direction for the observed high-h 2D tilt; separately, the GW term is a likelihood and carries no
  such Jacobian by construction.
- **DATE/SOURCE:** 2026-06-19. Anchor: `"Missing dd_L/dz Jacobian in the catalog z-integral"` and
  `"REFUTED — ∝1/h is decreasing → biases h LOW"` — `BIAS_HISTORY_LEDGER.md` row 28 (§1 table).
  Carried into `CLAIM_2D_BIAS_20260730.md`'s list as `"catalogue dd_L/dz Jacobian"`.
- **WHAT IT DOES NOT COVER:** any Jacobian-shaped object outside the catalogue z-integral (e.g. a
  volume element inside a *different* kernel) is a distinct object and not covered.

### [FISHER-FRAME] — ecliptic vs equatorial frame mismatch in the Fisher matrix

- **MECHANISM:** a coordinate-frame (ecliptic/equatorial) mismatch between the CRB/Fisher
  construction and the GLADE host frame. Search also: "frame mismatch", "coordinate rotation",
  "ecliptic", "equatorial".
- **BOUND:** EXONERATED / non-cause — rotating would double-rotate (the frames already agree at the
  point tested); confirmed by two adversarial refuters on the construction chain.
- **DATE/SOURCE:** 2026-06-19 (seed400). Anchor: `"Fisher ecliptic/equatorial frame mismatch
  (seed400)"` / `"EXONERATED / non-cause; rotating would double-rotate"` —
  `BIAS_HISTORY_LEDGER.md` row 27.
- **NOTE — do not confuse with a DIFFERENT, EARLIER, LANDED Fisher-frame fix:** row 12 (Ph 36 +
  43-H2, "Equatorial CRB vs ecliptic GLADE frame … fixed — PRIMARY mover per A5") is a real,
  already-landed correction from an earlier phase. Row 27 is a *later, separate* re-check of the
  *post-fix* frame consistency, which came back clean. Citing row 12's number ("host recovery
  31→38/60") against a *current* frame-mismatch hypothesis would be citing the wrong entry.
- **WHAT IT DOES NOT COVER:** any frame question in a component other than the Fisher/CRB
  construction (e.g. a host catalogue's own internal sky-coordinate convention) is untested here.

### [PDET-CHOICE] — detection-probability estimator choice

- **MECHANISM:** which `p_det` estimator (KDE vs simulation-based survival estimator vs other) is
  used. Search also: "detection probability estimator", "p_det method".
- **BOUND:** exonerated as a 2D-bias driver (bundled item in the claim file's list; see also row 33's
  landed fix and row 82's later confirmation that the estimator choice, once fixed, is not itself
  the residual).
- **DATE/SOURCE:** carried in `CLAIM_2D_BIAS_20260730.md:723` as `"p_det estimator choice"`.
- **WHAT IT DOES NOT COVER:** `p_det` *placement* (inside vs outside the numerator) is a SEPARATE
  entry — see [PDET-IO] and [PDET-NUM-ALONE] below. Do not conflate estimator choice with
  numerator/denominator placement; they were tested and exonerated separately.

### [PDET-IO] — p_det inside vs outside the likelihood numerator

- **MECHANISM:** whether the detection-probability factor sits inside or outside the per-event
  numerator term. Search also: "p_det inside", "p_det outside", "selection factor placement".
- **BOUND:** exonerated as a 2D-bias driver in this general form; see [PDET-NUM-ALONE] for the
  specific "add p_det inside the numerator alone" variant, which is a distinct, more specific claim
  that FAILS its own calibrated controls.
- **DATE/SOURCE:** `CLAIM_2D_BIAS_20260730.md:723`, `"p_det inside/outside"`.
- **WHAT IT DOES NOT COVER:** the specific "numerator-only" family of fixes (row 37/38/70) — see
  [NUMERATOR-ONLY-CLEAN].

### [H-PRIOR] — sensitivity to the H0 prior

- **MECHANISM:** the choice/width of the prior on h. Search also: "prior sensitivity", "h-prior".
- **BOUND:** exonerated as a 2D-bias driver.
- **DATE/SOURCE:** `CLAIM_2D_BIAS_20260730.md:723`, `"h-prior sensitivity"`.
- **WHAT IT DOES NOT COVER:** nothing further specified in the source; treat as a coarse bundled
  exoneration, not a quantified bound.

### [VOLUME-TRUNC] — unifying the host-z numerator window (`volume_trunc`)

- **MECHANISM:** truncating/clipping the host-redshift window in the numerator to "fix" a rail.
  Search also: "volume truncation", "z-window unification", "host-z clip".
- **BOUND:** FALSIFIED as a fix — moved the bias the WRONG way by ~4×: 1D mean 0.745→0.800, 2D
  0.768→0.800. Two causes identified: `fixed_quad(n=50)` aliases the GW peak (0.0000 vs exact
  0.24–0.65), and the exact host-window numerator itself tilts high.
- **DATE/SOURCE:** 2026-07-12 (`c4a1c7d`), pre-registered seed600 494-event A/B. Anchor:
  `"volume_trunc … fixes the shallow bias"` / `"FALSIFIED: 1D mean 0.745 → 0.800"` —
  `BIAS_HISTORY_LEDGER.md` row 70.
- **WHAT IT DOES NOT COVER:** **VENUE-SCOPED.** Measured on the *same seed600 494-event shallow
  subsample* as `mass_trunc` (#72). A shared venue idiosyncrasy would fool both — do not cite as
  universal (`"Standing scoping rule"`, ledger §2, anchor: `"negative conclusions are venue-scoped"`).

### [Z-LEG] — the host-redshift leg as channel-common

- **MECHANISM:** the idea that the redshift-side machinery (shared between 1D and 2D) is the source
  of the 2D-only excess. Search also: "z leg", "redshift channel", "channel-common z term".
- **BOUND:** exonerated — by construction the z leg is bit-identical between channels (only `L_cat`
  differs), so it cannot produce a *2D-specific* residual.
- **DATE/SOURCE:** `CLAIM_2D_BIAS_20260730.md:724`, `"the z leg (channel-common)"`; corroborated by
  `HANDOFF_20260730.md:29-30`, `"w_G and L_comp are channel-common and bit-identical; only L_cat
  differs."`
- **WHAT IT DOES NOT COVER:** a z-leg object that DOES differ between channels (none has been found)
  would not be covered by this exoneration.

### [LNM-DRAW] — the ln-mass draw itself

- **MECHANISM:** the log-mass sampling/draw step for host masses. Search also: "ln-M draw", "mass
  draw", "log-mass sampling bias".
- **BOUND:** exonerated — mean |Δln M| ≤ 0.0009 dex, negligible.
- **DATE/SOURCE:** `CLAIM_2D_BIAS_20260730.md:724-725`, `"the ln-M draw itself (mean |Δln M| ≤
  0.0009 dex)"`.
- **WHAT IT DOES NOT COVER:** downstream consumers of the draw (the mass *kernel*, the mass
  *window*) are separate, separately-tested objects — see [MASS-KERNEL-FAMILY] and **HB**.

### [REALIZATION] — realization plumbing / noise-injection pipeline

- **MECHANISM:** the machinery that adds observational scatter (realizes noisy observed values from
  true ones). Search also: "realization plumbing", "noise injection", "scatter pipeline".
- **BOUND:** exonerated — byte-identical between channels at σ→0.
- **DATE/SOURCE:** `CLAIM_2D_BIAS_20260730.md:725`, `"realization plumbing (σ→0 byte-identical in
  both channels)"`.
- **WHAT IT DOES NOT COVER:** nothing further specified.

### [WINDOW-MEMBERSHIP] — candidate-window membership (coarse on/off)

- **MECHANISM:** whether a candidate host is admitted to the search list at all (binary
  membership), as distinct from the shape of any window. Search also: "candidate membership",
  "eligibility list", "search-window inclusion".
- **BOUND:** exact removal of realization-added 2D candidates moves MAP 0.81→0.82 — **wrong sign**
  for a mechanism that should push toward 0.73.
- **DATE/SOURCE:** `CLAIM_2D_BIAS_20260730.md:726-727`, `"candidate-window membership (exact removal
  moves MAP 0.81→0.82, wrong sign)"`; also `HANDOFF_20260730.md:63-64`.
- **WHAT IT DOES NOT COVER — IMPORTANT DELIMITATION, explicitly worked out by [WGEO] (2026-08-27):**
  this is "a coarse binary lever (window OFF vs ON), not the window's internal geometry"
  (`CLAIM_WGEO_20260827.md:342-344`). It does **not** cover: (a) the *shape* of any window
  (linear vs log — see HB and [WGEO]'s own delimitation below), (b) the with-BH channel's
  ASYMMETRIC filter defect (GW ±1.5σ vs galaxy ±1σ — see [WBHZERO-ASYMMETRY] in §4, which is a
  **confirmed defect, not an exoneration** and must not be read as "window questions are closed").

### [MASS-KERNEL-FAMILY] — the functional family of the mass-marginalization kernel

- **MECHANISM:** which probability-density family (Gaussian vs truncated-lognormal, etc.) is used to
  marginalize over host mass uncertainty inside the likelihood, applied to candidates that already
  passed eligibility. Search also: "mass kernel family", "mass PDF choice", "truncated lognormal
  kernel", "`mass_trunc`", "Gaussian vs lognormal mass marginalization".
- **BOUND:** bounded at **+0.002** — exonerated as the 2D *driver*, twice, independently:
  (i) pipeline A/B: Δ2D **+0.0029, wrong sign**, Δ1D exactly 0.0000 (`mass_trunc_ab_20260713`,
  seed600 494-event A/B) — reason: the same prior enters the selection denominator `D_g`, so
  `N_g/D_g` cancels the numerator shift ("the selection denominator is not a spectator").
  (ii) the ratified-kernel 4-cell A/B: moves 2D only **−1.8…−2.3 ln** of a **+25.6…+29.1 ln**
  excess, MAP unmoved at 0.80 (`mass_ab_20260727`, "NECESSARY, NOT SUFFICIENT").
- **DATE/SOURCE:** 2026-07-13 (`70bee1f`) and 2026-07-27 (`e9bec6d`). Anchors:
  `"EXONERATED: Δ2D mean +0.0029, wrong sign"` (`BIAS_HISTORY_LEDGER.md` row 72) and
  `"NECESSARY, NOT SUFFICIENT … 2D MAP unmoved at 0.80"` (row 89). Also
  `CLAIM_2D_BIAS_20260730.md:727`, `"mass-kernel family (bounded +0.002)"`.
- **WHAT IT DOES NOT COVER — explicitly, per the ledger's own note:** "do not re-derive it a third
  time" (`BIAS_HISTORY_LEDGER.md:139`). It does NOT cover: (a) the mass **eligibility window** — a
  hard cut applied *before* the kernel runs on the *search* (this is **HB**, a different object,
  different stage, different file region); (b) the **uncertainty budget** the kernel is built on
  (R&V15 intrinsic scatter, ~0.55 dex, omitted from the propagated error) — this is the correctness
  question opened as [P3-MKER] and explicitly delimited FROM this exoneration
  (`CLAIM_P3_MKER_20260826.md:56-61`, `"Ledger §2 item 1 exonerates the mass-kernel FAMILY as the
  2D-bias driver … against that exoneration's bound before banking"`); (c) the window's *geometry*
  (linear vs log shape) — this is [WGEO], likewise explicitly delimited
  (`CLAIM_WGEO_20260827.md:345-348`).

### [OPTIONA-DRIFT] — Option-A `β_G/Σ_glob` calibration drift

- **MECHANISM:** an apparent drift between the model's completeness normalization `β_G` and the
  discrete catalogue sum `Σ_glob`, read as a possible bias source. Search also: "Option-A drift",
  "β_G/Σ_glob", "global normalization drift", "h-cubed volume factor".
- **BOUND:** NOT a defect — it is exactly the expected `h⁻³` volume Jacobian: (0.73/0.81)³−1 =
  **−26.80%**, matching the measured −25.6%/−26.8%. After removing the expected Jacobian, a
  residual survives but is **1D-only, +1.667% (= +0.017 in h)**; the 2D leg passes at |δ| < 0.4%
  over ±18% in h, which *removes catalogue-selection calibration from the 2D suspect list entirely*.
- **DATE/SOURCE:** `CLAIM_2D_BIAS_20260730.md:727-729`, `"Option-A calibration drift β_G/Σ_glob (=
  the exact h⁻³ volume Jacobian, (0.73/0.81)³−1 = −26.80%; residual is 1D-only, +0.017 in h)"`;
  also `HANDOFF_20260730.md:68-72`.
- **WHAT IT DOES NOT COVER:** the surviving **+0.017 in h, 1D-only** residual remains genuinely
  open (`BIAS_HISTORY_LEDGER.md §4` item 5, `"1D residual-after-Jacobian +1.667%"`) — do not cite
  this entry as closing the 1D channel.

### HA — completion term not mass-marginalised ("D(h) not mass-marginalised")

- **MECHANISM:** the completion-leg denominator/normalization (`D`, `β_G`, `β_Ḡ`) is missing a
  mass-marginalization dimension that the catalogue leg has (a "4D numerator leg added to a 3D
  numerator leg" dimensional mismatch). Search also: "completion term dimensionality", "D(h) not
  mass-marginalised", "dimensional mismatch", "mass-marginalization asymmetry between legs".
- **BOUND:** **Real defect, CONFIRMED, but WRONG SIGN.** The full code-consistent correction moves
  the MAP from r1 0.8133→0.8492 and r2 0.7820→0.8527 — mean **+0.053 where −0.077 was required**.
  Decomposed (2026-07-30 adjudication): **−0.058 measure + +0.093 population tilt**, net +0.036
  (still wrong sign). Endpoint (0.8492) independently reproduced to **3e-5** from a different
  starting point via C8 — the exoneration is "upheld and decomposed", not merely repeated.
- **DATE/SOURCE:** original: `HANDOFF_20260730.md:73-77`, `"HA — completion term not
  mass-marginalised … moves r1 0.8133 → 0.8492 … Real defect, wrong sign"`. Decomposition:
  `CLAIM_2D_BIAS_20260730.md:736-738`, `"HA — upheld and decomposed (−0.058 measure + +0.093
  population tilt; endpoint independently reproduced to 3e-5 …)"`.
- **WHAT IT DOES NOT COVER:** the *premise* that "D(h) not mass-marginalised" is itself the right
  framing was ALSO separately refuted (§C8, `CLAIM_2D_BIAS_20260730.md:736-740`): `D`, `β_G`, `β_Ḡ`
  ARE correctly mass-dimensionless; the imbalance is **numerator-internal**, not a denominator
  defect. Any future claim using the literal words "D(h) not mass-marginalised" should be read as
  a formulation already refuted, distinct from HA's (upheld, wrong-sign) numeric correction.

### HC — mixture-floor / zero-handling

- **MECHANISM:** the idea that a numerical floor/zero-handling strategy for combining per-event
  likelihoods (to avoid a single zero collapsing the joint posterior) introduces or masks a bias.
  Search also: "mixture floor", "zero-handling", "physics-floor", "log-space combination floor",
  "underflow guard".
- **BOUND:** REFUTED structurally — exactly **0.000 in h**. The `physics-floor` guard never fires:
  65,108/65,108 cells nonzero, 0 excluded events in all 16 combined posteriors. The apparent
  `n_events_empty = 2` asymmetry was a miscount of two non-integer JSON keys, not real exclusions.
- **DATE/SOURCE:** `HANDOFF_20260730.md:78-81`, `"HC — mixture-floor / zero-handling. REFUTED
  structurally: physics-floor never fires (65,108/65,108 cells nonzero …)"`; carried into
  `CLAIM_2D_BIAS_20260730.md:731-732`.
- **WHAT IT DOES NOT COVER:** this is a statement about **this project's own production combination
  code**, on the runs measured. A DIFFERENT harness's zero-handling (e.g. the mirror harness's
  `-1e300` log-space sentinel, row #145, 2026-08-20) is a **different artifact in a different
  codepath** and is NOT covered by HC — do not treat HC as blanket clearance for all zero-handling
  code in the repository. (The mirror sentinel is its own, separately-diagnosed defect; see
  `BIAS_HISTORY_LEDGER.md` row/§ for row #145 if that thread needs re-checking.)

### HB — hard mass window as support truncation ★ the near-miss mechanism

- **MECHANISM (plain terms, deliberately verbose because this is the entry that was missed):** a
  **hard, one-sided cut on candidate-host mass**, applied where the mass information should instead
  enter as a soft (continuous) kernel weight. Because the GW-side mass is near-exact but the
  catalogue-side mass error is huge (median σ_lnM ≈ 1.28 ≈ 0.56 dex), the window's *upper* leg is
  almost always vacuous (mass minus its error is negative), so only a **hard lower floor** does any
  work — an asymmetric, one-sided truncation. Because the window's bounds are built from z_min/z_max,
  which depend on h, the truncation is **h-dependent by construction**. Search also (use ANY of
  these; the filing session's exact words will not match a future proposer's): "mass window",
  "hard cut", "support truncation", "eligibility filter", "one-sided cut", "negative lower edge",
  "truncation bias", "hard clamp on mass", "mass floor", "candidate-search mass filter", "sigma
  clipping on host mass", "h-dependent selection cut".
- **BOUND:** **REFUTED. Tilt = −0.317 nats over h 0.73→0.81 = 0.063% of the 504.8-nat target,
  ~40–50× too small at its ceiling, and SIGN-INVERTED** (removing the window moves the MAP *up* by
  ~+0.0015, i.e. the opposite direction from what a bias-driver correction would need). Strengthened
  in adjudication: "its hard-zeros are worth 1.5% of the target" — still far too small.
  Independently corroborated in TWO NEW venues by [WGEO] (2026-08-27, chair-re-derived, whole
  pruned catalogue, N=20,834,171): the same 193-low-side-vs-1-high-side asymmetry reproduces as
  29:1 (cone-exact fleet) and 12.93:1 (single GW interval, whole catalogue) — both far too small to
  own the target, both wrong-signed in the same qualitative way as the original 2026-07-30
  measurement.
- **DATE/SOURCE:** 2026-07-30. Anchor (the primary, binding form): `"HB hard mass window as
  support truncation (tilt −0.317 nats = 0.063% of the target, sign-inverted, 40–50× too small)"` —
  `CLAIM_2D_BIAS_20260730.md:732-734`. Fuller rationale and the 193:1 census:
  `"the truncation's h-tilt is −0.317 nats over 0.73→0.81 = 0.063% of the 504.8-nat target, ~50×
  too small at its ceiling, and sign-inverted"`, `"pooled 193 low-side rejections vs 1 high-side"` —
  `HANDOFF_20260730.md:85-88, 107`. Strengthening: `"HB (its hard-zeros are worth 1.5% of the
  target)"` — `gate_b_20260730/ADJUDICATION_20260730.md` (quoted verbatim at
  `CLAIM_2D_BIAS_20260730.md:742-744`). §2 of the ledger carries it by name:
  `BIAS_HISTORY_LEDGER.md:135`, `"… HA as owner · HC zero-handling · HB."`
- **WHAT IT DOES NOT COVER — and why the near-miss happened anyway.** HB's OWN measurement is
  scoped to: (a) the **estimator/candidate-search-side** hard mass window in `handler.py`
  (production call site `bayesian_statistics.py:4691`, mask logic `handler.py` — line numbers have
  since drifted, see §7 stale-citation note); (b) applied as **window removal / presence-vs-absence**,
  not as a shape change. It does **not**, on its own text, explicitly rule out: (i) a *different
  shape* for the same window (linear vs log-symmetric) — this looked, to a search on
  "candidate-window membership" or "mass-kernel family", like an open question; **it is not** — see
  [WGEO]'s decisive re-derivation below, which shows the shape question is ALSO governed by HB
  because HB's own rationale ("hard cut where mass information should be a soft kernel weight",
  "negative lower edge", "193 low-side vs 1 high-side", "h-dependent by construction") is **[WGEO]'s
  H1 almost verbatim** (`CLAIM_WGEO_20260827.md:299-309`). A shape variant of a mechanism HB already
  bounds is bounded by HB's ceiling unless the new claim brings a construction HB's measurement did
  not exercise. (ii) It does not cover the with-BH-channel eligibility ASYMMETRY defect
  ([WBHZERO-ASYMMETRY], §4) — that is a confirmed defect in a *different* filter comparison
  (GW ±1.5σ vs galaxy ±1σ), not a re-opening of HB.

---

## 2. LEDGER §2 SUPPLEMENTAL EXONERATIONS — layer 2 (⚠ items, `BIAS_HISTORY_LEDGER.md:137-155`)

**These 17 items are explicitly flagged in the ledger as "absent from the [layer-1] claim file …
and therefore the live re-litigation risk."** They are numbered items 1–17 in the source; the same
numbers are kept here for traceability, cross-referenced to their §1 chronological-table row.

### 1. [MASS-KERNEL-FAMILY] (duplicate pointer)
Same object as the §1 entry above. Ledger's own note: `"the same verdict measured differently — do
not re-derive it a third time"` (`BIAS_HISTORY_LEDGER.md:139`).

### 2. [GRAY-MIX-AMPLIFY] — full Gray mixture as the compensation channel

- **MECHANISM:** using the FULL Gray et al. mixture `(β_G·L_cat + B_num)/D` (rather than the
  project's two-branch construction) as a fix for the deep-venue rail. Search also: "full Gray
  mixture", "Gray-formula compensation", "canonical mixture restoration".
- **BOUND:** **AMPLIFIES** the bias — worst case **+0.123** vs the two-branch construction's +0.032;
  12/12 configurations fail; the host branch flips from a counterweight (−26…−182) to a co-tilting
  term (+47…+166).
- **DATE/SOURCE:** 2026-07-11 (N-1). Anchor: `"Full Gray mixture … restores deep calibration …
  AMPLIFIES: worst +0.123 …12/12 fail"` — `BIAS_HISTORY_LEDGER.md` row 60.
- **WHAT IT DOES NOT COVER:** nothing further specified; this is a coarse construction-level test.

### 3. [WG-BOOKKEEPING-FIX] — `w_G = β_G/D` bookkeeping / membership-conditioned inverse, AS A FIX

- **MECHANISM:** re-deriving `w_G` as a membership-conditioned inverse to fix the deep-venue rail.
  Search also: "w_G bookkeeping", "membership-conditioned inverse", "weight relocation fix".
- **BOUND:** REFUTED as a fix — bias +0.005…+0.044, 12/12 fail; the tilt merely **relocates** to the
  host branch (+94…+455), it does not disappear.
- **DATE/SOURCE:** 2026-07-11 (N-2b). Anchor: `"The defect is w_G = β_G/D bookkeeping … REFUTED —
  +0.005…+0.044, 12/12 fail; tilt merely relocates"` — `BIAS_HISTORY_LEDGER.md` row 61.
- **WHAT IT DOES NOT COVER — explicit, load-bearing note in the record itself:** `"this is NOT the
  same as Gate C item 1, which asks whether β_G and Σ_glob are mutually consistent"`
  (`BIAS_HISTORY_LEDGER.md:141`). The FIX is refuted; the underlying **calibration-level question**
  (C9, `w_G` mis-calibrated by 2.35× at truth) is separately **LIVE**, gated on cell B — see
  `CLAIM_2D_BIAS_20260730.md:741`, `"w_G is deliberately NOT added to this list: C9 is live"`.
  **Do not read "w_G bookkeeping refuted" as "w_G calibration is fine."**

### 4. [HARD-CLAMP-OBSERVED-Z] — hard support truncation on observed z as the production fix

- **MECHANISM:** proposing a hard membership cut evaluated on the *observed* (noisy) redshift as
  the production selection mechanism. Search also: "hard clamp", "observed-z membership",
  "truncation on observed redshift".
- **BOUND:** REFUTED for production — sign-flipping bias −0.021…+0.015, coverage 0.18–0.46; needs a
  **soft, photo-z-marginalized** membership instead.
- **DATE/SOURCE:** 2026-07-11 (N-2d). Anchor: `"A hard support truncation is the production fix …
  REFUTED for production: sign-flipping biases −0.021…+0.015"` — `BIAS_HISTORY_LEDGER.md` row 63.
- **WHAT IT DOES NOT COVER:** this is about the **fix construction** (hard cut on observed z), not
  about whether a truncation mechanism exists at all (see HB for the mass-window case; this item is
  the redshift-membership analogue and a DIFFERENT variable).

### 5. [WPOP-TUNING] — tuning the population-prior weight `w_pop`

- **MECHANISM:** adjusting the population-rate prior weighting to absorb the residual. Search also:
  "population prior tuning", "w_pop misspecification", "rate-prior tilt".
- **BOUND:** NEGLIGIBLE — ≤ +0.0004 at a 10% deliberate misspecification. "Escape hatch closed."
- **DATE/SOURCE:** 2026-07-11 (N-3). Anchor: `"Tuning w_pop … NEGLIGIBLE / escape hatch CLOSED: ≤
  +0.0004"` — `BIAS_HISTORY_LEDGER.md` row 64.
- **WHAT IT DOES NOT COVER:** nothing further specified.

### 6. [PDET-NUM-ALONE] — adding `p_det` inside the numerator ALONE

- **MECHANISM:** adding a detection-probability factor inside the per-event numerator without a
  matched denominator change. Search also: "p_det inside numerator alone", "unpaired p_det factor",
  "single-sided p_det correction".
- **BOUND:** REFUTED — deep cells unchanged (Δ≤+0.0006) and it actively **breaks** the calibrated
  controls (−0.003 → +0.003…+0.006). Ledger's own words: **"Do not cargo-cult it."** Only the
  JOINT model-σ + p_det-inside pair works (a separate, matched construction, #67).
- **DATE/SOURCE:** 2026-07-11 (27m). Anchor: `"The floor is the p_det-inside-numerator factor …
  REFUTED — deep cells unchanged … and it flips the calibrated controls"` —
  `BIAS_HISTORY_LEDGER.md` row 66.
- **WHAT IT DOES NOT COVER:** the PAIRED construction (model-σ change + p_det-inside together, #67)
  is a DIFFERENT, NOT-refuted object — do not cite this entry against a paired proposal. This
  delimitation was correctly applied by `CLAIM_P3_IMPOSTOR_CONVENTION_20260822.md:44-49`,
  `"this intake does NOT re-open the refuted 'p_det alone' move; any stage-2 arm that degenerates to
  an unpaired numerator p_det is void by this exoneration"` — a worked example of doing this check
  correctly.

### 7. [DEPTH-TRUNC] — depth truncation (`--max_redshift` cut) as the deep-rail fix

- **MECHANISM:** cutting the catalogue/event depth at some `z_max` to cure a rail. Search also:
  "depth truncation", "max_redshift cut", "redshift depth cap".
- **BOUND:** empirically dead — rails at every depth tested (0.2 / 0.3 / 0.5).
- **DATE/SOURCE:** 2026-07-25. Anchor: `"Depth truncation cures the deep rail … empirically dead —
  rails at all three depths"` — `BIAS_HISTORY_LEDGER.md` row 56.
- **WHAT IT DOES NOT COVER:** nothing further specified; this is a coarse parameter sweep, not a
  mechanism-level exoneration of "depth" as a concept.

### 8. [ZERO-HOST-FALLBACK] — the #29 zero-host fallback as the rail cause

- **MECHANISM:** events with no catalogue host silently dropped (or falling back to a
  completion-only estimator) as the source of a deep rail. Search also: "zero-host fallback",
  "dropped events", "no-host bookkeeping bug", "fallback estimator rail".
- **BOUND:** the underlying bug is REAL (58% of events dropped on the deep venue) but fallback
  events are **h-inert** — the rail PERSISTS after the fix (0.6000 both channels); host events move
  −4265 over the grid, fallback events move only −59.
- **DATE/SOURCE:** fix 2026-07-10 (`8db6c6e`, #29/#54); re-test 2026-07-19/25 (EXP-40, #55). Anchor:
  `"The #29 fix clears the deep rail … REFUTED — rail PERSISTS at 0.6000 both channels"` —
  `BIAS_HISTORY_LEDGER.md` row 55. Strengthened 2026-08-20 (row #147): `"the §2 item 8 exoneration
  is CONFIRMED, strengthened"` (machine-precision confirmation, `BIAS_HISTORY_LEDGER.md:1777`).
- **WHAT IT DOES NOT COVER:** the fallback-estimator's OWN calibration (whether it is unbiased when
  it IS used) is a separate, later-measured object — see [LCOMP-BNUM-DEFECT] item 10 and #57
  (`pp_coverage_deepvenue`, biased HIGH +0.7…+5.4% at high completion fraction) — that finding is
  NOT an exoneration and is not superseded by this entry.

### 9. [OMEGA-M-ERA] — Ω_m cosmology-era mismatch

- **MECHANISM:** using a WMAP-era Ω_m (0.2726) rather than Planck-era, as a bias source. Search
  also: "Omega_m mismatch", "cosmology era", "WMAP vs Planck fiducial".
- **BOUND:** as a bias explanation for the seed600 shallow residual: NEGLIGIBLE and wrong-signed
  (Δh̄ = −0.00059, era-corrected residual is LARGER, +0.0138). As a "bug": NOT a bug — it is a
  deliberate DESIGN CHOICE matching the Barausse (2012) M1 EMRI population model.
- **DATE/SOURCE:** 2026-07-10 (L-C, `BIAS_HISTORY_LEDGER.md` row 59) and 2026-07-02 (G11,
  `bdf5339`, row 53). Anchors: `"seed600 Ω_m era mismatch … NEGLIGIBLE and wrong-signed"` (row 59);
  `"WMAP-era Ω_m is a bug … design choice: Ω_m=0.2726 matches Barausse-2012 M1"` (row 53).
- **WHAT IT DOES NOT COVER:** the Planck-era comparison case is QUOTED (+1.5–3%), not exonerated —
  it remains a tracked systematic (per CLAUDE.md's own note on `docs/gates/G7_systematics_budget.md`
  row 6), not a closed question about which cosmology is "correct" for real-data use.

### 10. [LCOMP-BNUM-DEFECT] — `L_comp`/`B_num` as a defective integral

- **MECHANISM:** the claim that the completion-leg numerator integral (`L_comp`/`B_num`) is itself
  mis-formulated or numerically wrong. Search also: "completion term defect", "B_num defective
  integral", "completion numerator bug".
- **BOUND:** exonerated TWICE, by different methods: (i) self-consistency Monte Carlo (#80,
  2026-07-26): the fallback-only apparent low bias (0.6118±0.0176) was a **subset-conditioning
  artifact** — `B/β_Ḡ = 0.7366±0.0155` closes at truth on a membership-clean subset. (ii) the
  impostor-ball full-power harness (#87, 2026-07-27): impostor channel AND normalization channel
  both exonerated as residual carriers; **"the residual is entirely B_num"** — i.e. B_num carries
  the residual but is NOT SHOWN to be a defect itself.
- **DATE/SOURCE:** Anchors: `"The completion term B_num is defective … EXONERATED — subset-
  conditioning artifact"` (`BIAS_HISTORY_LEDGER.md` row 80); `"Impostor channel and normalization
  channel both EXONERATED as residual carriers; the residual is entirely B_num"` (row 87).
- **WHAT IT DOES NOT COVER — this is a subtle, load-bearing delimitation, quoted directly from the
  ledger's own item 10:** `"B_num is the residual carrier but not a shown defect"`
  (`BIAS_HISTORY_LEDGER.md:148`). A claim that B_num CARRIES the residual is NOT contradicted by
  this exoneration; only a claim that B_num's *integral formulation* is internally wrong is. This
  distinction is exactly the kind a future rule-1 check could blur — check which claim you are
  making before citing this entry.

### 11. [VOLUME-DECONV-H-DEP] — `volume_deconv` kernel h-dependence

- **MECHANISM:** the possibility that the volume-deconvolution kernel used for the broadened host-z
  numerator carries an unaccounted h-dependence. Search also: "volume_deconv h-dependence", "kernel
  h-sensitivity".
- **BOUND:** EXONERATED — exactly h-invariant to **1e-15** (`Z_g ∝ h⁻³` factors out cleanly).
- **DATE/SOURCE:** 2026-07-25 (D2). Anchor: `"volume_deconv kernel carries h-dependence … EXONERATED
  — exactly h-invariant (Z_g ∝ h⁻³ to 1e-15)"` — `BIAS_HISTORY_LEDGER.md` row 75.
- **WHAT IT DOES NOT COVER:** nothing further specified.

### 12. [PDET-ANCHOR] — p_det anchor / first-bin asymptote escalation

- **MECHANISM:** the near-d_L=0 anchor value of the detection-probability curve as a bias source.
  Search also: "p_det anchor", "first-bin asymptote", "d_L=0 escalation".
- **BOUND:** REFUTED — wrong layer; raising the anchor by +12% moved the MAP by **zero grid steps**.
- **DATE/SOURCE:** 2026-05-02/03 (Ph 45). Anchor: `"p_det first-bin asymptote / d_L=0 anchor is the
  bias … REFUTED — anchor is the wrong layer: MAP unchanged … ZERO grid steps"` —
  `BIAS_HISTORY_LEDGER.md` row 17.
- **WHAT IT DOES NOT COVER:** nothing further specified.

### 13. [INFO-STARVATION] — "information starvation" as the explanation for the H0 rail

- **MECHANISM:** the claim that in-catalogue photo-z dark sirens are fundamentally information-poor,
  hence the rail. Search also: "information starvation", "starved posterior", "information-poor
  regime".
- **BOUND:** **OVERTURNED.** Originally concluded 2026-06-30, then reversed 2026-07-02 (G6):
  starvation is "a property of prior-INCONSISTENT estimators, not of the data" — consistency
  ("counted exactly once") is the actual cure. **Ledger's own instruction: "Do NOT resurrect it as
  an explanation."**
- **DATE/SOURCE:** original: `BIAS_HISTORY_LEDGER.md` row 41 (`"VERDICT: in-catalogue photo-z dark
  sirens are information-starved … later OVERTURNED — see #52"`). Overturn: row 52,
  `"STARVATION OVERTURNED: 'a property of prior-INCONSISTENT estimators, not of the data'"`.
- **WHAT IT DOES NOT COVER:** nothing further specified — this is a full reversal, not a scoped
  bound. Checked and correctly respected by later threads: `m2_residual_owner/
  CLAIM_M2_RESIDUAL_OWNER_20260807.md:156` (H-a/H-d hypotheses explicitly do NOT resurrect
  starvation) and `PREREGISTRATION_HIER_HTHETA_20260826.md:115`.

### 14. [SPECZ-RESCUE] — spectroscopic-redshift subset as an information rescue

- **MECHANISM:** the hope that a spec-z (rather than photo-z) subset of hosts carries the
  informative posterior shape. Search also: "spec-z rescue", "spectroscopic subset", "spec-z money
  figure".
- **BOUND:** REFUTED — spec-z hosts are 0.56% of GLADE+, contribute ≤8.7% (median ~0%) of the
  rate-weighted in-catalogue likelihood; the inference-side `flag==3` cut still rails at 0.870.
- **DATE/SOURCE:** 2026-06-30. Anchor: `"Spec-z host subsets carry the informative posterior shape
  … REFUTED — spec-z = 0.56% of GLADE+, ≤8.7% … of rate-weighted in-cat likelihood"` —
  `BIAS_HISTORY_LEDGER.md` row 42. Companion document: `docs/F4_SPECZ_DECOMPOSITION.md`.
- **WHAT IT DOES NOT COVER:** nothing further specified.

### 15. [PV-FRAME] — heliocentric/CMB frame and peculiar-velocity corrections

- **MECHANISM:** the choice of rest frame (heliocentric vs CMB-dipole) for host redshifts, and the
  host peculiar-velocity correction. Search also: "heliocentric frame", "CMB dipole", "peculiar
  velocity correction", "PV frame".
- **BOUND:** net effect **+0.15%** in h (frame), and 2D **PV-insensitive at +0.0012** (peculiar
  velocity); both far too small to matter, ~120× smaller than the rail and orthogonal to it.
- **DATE/SOURCE:** 2026-07-04/07021f6f. Anchors: `"Heliocentric vs CMB-frame host z … fixed: net
  +0.15%"` (`BIAS_HISTORY_LEDGER.md` row 43); `"Host peculiar-velocity value correction … 2D
  PV-insensitive +0.0012; #16 CLOSED"` (row 44).
- **WHAT IT DOES NOT COVER:** nothing further specified.

### 16. [CODE-HYGIENE-BUNDLE] — inert code items (bundled)

- **MECHANISM:** a bundle of five separately-checked, all-inert code items: `galaxy.py`'s (1+z)³
  σ_z scaling; `TRUE_HUBBLE_CONSTANT=0.7` dead-code inconsistency; the CRB row-424 apparent
  bimodality; `p_det` grid resolution (30 vs 60 bins); `allow_singular=True`; numerical posterior
  underflow. Search also: "row-424 seam", "p_det grid resolution", "allow_singular", "underflow",
  "TRUE_HUBBLE_CONSTANT dead code".
- **BOUND:** all inert / non-causes. Row-424 is a seam (seed200⊕seed300, emcee under-mixing), impact
  on H0 NONE. `TRUE_HUBBLE_CONSTANT` is dead code, not on the production path. Grid resolution is
  "not a bias source" (re-confirmed twice, Audit A8 G8a). `allow_singular` — fixed-and-landed,
  irrelevant to bias. Underflow — fixed (log-space + 4 strategies), "not the cause".
- **DATE/SOURCE:** rows #25 (row-424), #30 (galaxy.py, Ω_m — note Ω_m is separately listed as item
  9 above), #10 (grid resolution), #18 (`TRUE_HUBBLE_CONSTANT`), #7 (underflow) —
  `BIAS_HISTORY_LEDGER.md:154` bundles them: `"galaxy.py (1+z)³ σ_z, TRUE_HUBBLE_CONSTANT, CRB
  row-424 seam, p_det grid resolution, allow_singular, underflow — all inert/non-causes"`.
- **WHAT IT DOES NOT COVER:** `datamodels/galaxy.py` itself was later DELETED as dead code
  (2026-07-04, commit `90bd40ee`, per `CLAUDE.md`'s Known Bugs §9) — so a claim invoking that file
  by path is now moot for a different reason (the file doesn't exist), not re-covered by this entry.

### 17. [NUMERATOR-ONLY-CLEAN] — numerator-only normalization cleans, and the local same-kernel denominator

- **MECHANISM:** "cleaning" only the numerator side of the mixture (leaving the denominator alone)
  as a de-rail strategy, in two variants (Angle A/C and Angle B), plus a third variant using a
  locally-matched (same-kernel) denominator. Search also: "numerator-only fix", "local same-kernel
  denominator", "de-rail via numerator cleaning".
- **BOUND:** REFUTED — both numerator-only variants rail the estimator UP to **0.870** at σ_z=0.035
  (#37, "DISQUALIFIED"). The local same-kernel denominator variant fails identically at BOTH σ_z
  values ("GATE FAIL — 0.870 at both", #38). A related global photo-z-smeared denominator `D_sm`
  de-biases the SPREAD but produces no peak at truth (std 0.11→0.097, never converges to 0.73, #39).
- **DATE/SOURCE:** 2026-06-30. Anchors: `"Numerator-only normalization cleans … cure the rail …
  DISQUALIFIED — both rail UP to 0.870"` (row 37); `"Local same-kernel consistent denominator cures
  it … GATE FAIL — 0.870 at both σ_z"` (row 38); `"Global photo-z-smeared selection D_sm cures it …
  DE-BIAS, NO PEAK"` (row 39).
- **WHAT IT DOES NOT COVER:** nothing further specified; "numerator-only" as a general STRATEGY is
  what's exonerated, referenced explicitly by the C7 adjudication as the reason a proposed
  numerator-only host-z fix is pre-empted (`BIAS_HISTORY_LEDGER.md:197`).

---

## 3. AMBIGUOUS / PROVENANCE-FLAGGED ITEMS FROM §2 (carry forward, do not treat as clean)

The ledger itself flags these — reproduced here so a future check inherits the caveat, not a false
sense of resolution:

- **`rung_I` harness sign-flip risk.** The commission's evidence-locker audit graded 0/5 of the
  project's own railing harnesses faithful, 0 trustworthy, and classified `rung_I` (the SOURCE of
  the [NUMERATOR-ONLY-CLEAN] rows #37–#39 above) as an ARTIFACT with **inverted labels**
  (`STANDARD → 0.60`, `CONSISTENT_DENOM → 0.87` in the commission's independent re-run, vs
  production's own directions). Anchor: `"the commission's evidence-locker audit graded 0/5 …
  faithful … classified rung_I … as an ARTIFACT with sign flips"` — `BIAS_HISTORY_LEDGER.md:162-166`.
  **Treat [NUMERATOR-ONLY-CLEAN]'s rail directions as contested, not settled**, if re-litigating.
- **Commission report path.** `results/commission_20260701/synthesis/REPORT.md` does NOT exist; the
  actual file is `DRAFT_REPORT.md`, and its own header discloses §7 was still being appended.
  Anchor: `BIAS_HISTORY_LEDGER.md:167-168`.

---

## 4. CONFIRMED DEFECTS — adjacent to exonerations but NOT exonerations (do not misfile)

These are mechanisms that were investigated and CONFIRMED as real, live bias/correctness issues.
They are listed here **specifically because their vocabulary overlaps with exonerated mechanisms**
(mass filter, window, eligibility) and a future rule-1 check could wrongly treat "this area was
looked at" as "this area is cleared."

### [WBHZERO-ASYMMETRY] — with-BH mass-filter asymmetry (GW ±1.5σ vs galaxy ±1σ)

- **MECHANISM:** the with-BH-mass candidate eligibility filter applies a DIFFERENT sigma-multiplier
  to the GW-side uncertainty (±1.5σ) than to the galaxy-side uncertainty (±1σ), an asymmetry
  recorded nowhere as a deliberate design choice. Search also: "mass filter asymmetry", "sigma
  multiplier mismatch", "GW-side vs galaxy-side window width".
- **STATUS:** **CONFIRMED, candidate-confirmed defect — NOT exonerated.** Production iiib: 688/1588
  = 43.3% of h=0.73 rows attributed to this filter emptying an otherwise-nonzero candidate ball
  (688/688 exact); `Σ^4D`/`B_num_wbh` carry NO matching cut (unmodeled one-sided numerator
  selection). Direction: toward completion/no-BH; magnitude and h-dependence UNMEASURED at
  confirmation time.
- **DATE/SOURCE:** 2026-08-25. Anchor: `"Gate-B VERIFIED [P3-WBHZERO]: DEFECT (candidate-confirmed)
  … the handler.py:634-642 mass-filter asymmetry (GW ±1.5σ vs galaxy ±1σ) is real and recorded
  nowhere as a design choice"` — `BIAS_HISTORY_LEDGER.md` row #196.
- **SUBSEQUENT ACTION (fixed):** measure-first fix chain authorized (row #198), then the SYMMETRIC
  mass-filter window was **adopted as production physics** ([PHYSICS] commit `cf4f8a2a`, row #202,
  2026-08-25) — `"THE SYMMETRIC MASS-FILTER WINDOW IS PRODUCTION PHYSICS"`.
- **WHY IT IS HERE, NOT IN §1/§2:** this is the opposite of an exoneration — it is a found-and-fixed
  defect. It sits directly adjacent to HB in the search space (both are "hard filter on candidate
  mass") and a search for "mass filter" or "eligibility window" will hit BOTH this entry and HB.
  **They are different objects**: HB is about the window's PRESENCE/removal (refuted as a bias
  owner, whole window); this is about a specific ASYMMETRY in how the window's two sides are scaled
  (confirmed real, fixed). Do not let a hit on this entry make you think "windows are a live
  suspect, ignore HB's exoneration" — nor let a hit on HB make you think "this asymmetry defect
  must be wrong too." They were correctly kept separate throughout the record.
- **DELIMITATION FROM HB, stated explicitly in the record:** the symmetric-window fix (rows
  #198–#202) is scoped to **"which side's uncertainty gets the multiplier"**
  (`handler.py:648-661`, `"Scope: the MASS filter only"`), which is orthogonal to HB's and
  [WGEO]'s **shape** question (linear vs log geometry) — `CLAIM_WGEO_20260827.md:318-335` works this
  delimitation out in full and confirms "nothing in this card disturbs the rows #198–#202 ruling."

---

## 5. VERIFICATION — evidence located at cited anchors

Every anchor quoted in §1–§4 was checked by direct read of the source file this session (not by
grep alone) at the location the citing document itself asserts. Status:

| entry | anchor located? | note |
|---|---|---|
| All of §1 (`CLAIM_2D_BIAS_20260730.md:721-757`) | **YES** | read in full; current line numbers 721-734 for the list, 736-757 for appendices (see §7 stale-citation note — the ledger's OWN pointer to this section is stale, not this register's) |
| All of §2 (`BIAS_HISTORY_LEDGER.md:127-170`) | **YES** | read in full |
| All of §1 table rows cited (#1-#98) | **YES** | read in full, `BIAS_HISTORY_LEDGER.md:13-126` |
| `HANDOFF_20260730.md` (HA/HC/HB rationale) | **YES** | read in full, 193 lines |
| `ADJUDICATION_20260730.md` (HB 1.5% strengthening) | **YES** | located at the cross-report reconciliation section; exact wording corroborated by its verbatim quote inside `CLAIM_2D_BIAS_20260730.md:742-744`, which is the citation of record |
| §4 WBHZERO rows #196, #198, #202 | **YES** | read in full |
| §3 M2/f_k/cross-term items | **YES** | `m2_residual_owner/CLAIM_M2_RESIDUAL_OWNER_20260807.md:477-487` and ledger row #96 read in full |
| [WGEO] card itself | **YES** | read in full, 553 lines |

**Nothing swept in this pass could not be located.** No entry is flagged "evidence not found."

---

## 6. NEAR-MISS SWEEP

**The known instance — [WGEO] vs HB (2026-08-27), reconstructed for the record:**

- A "coupling read" (one of three parallel stage-0 reads on the [WGEO] thread) ran a rule-1 check
  and reported it **PASSED**. Its own artifact, `wgeo_s0_coupling_20260827.md`, shows the check
  explicitly named and cleared two adjacent entries — "candidate-window membership" (§1 item 2 in
  this register) and "mass-kernel family" (§1 item 1) — both of which sit on the SAME lines of the
  same list as HB (`CLAIM_2D_BIAS_20260730.md:726-727` vs HB at `:732-734`). It did not reach HB,
  two entries later in the same unbroken list.
- The synthesis chair caught it on a full re-read, and the resulting card explicitly states the
  mechanism of the miss: **"The coupling read's rule-1 check passed only because it checked
  'candidate-window membership' and 'mass-kernel family' and stopped two lines short of HB"**
  (`CLAIM_WGEO_20260827.md:310-311`).
- Root cause as diagnosed by the chair: object-matching against the *proposing thread's own
  vocabulary* ("window shape", "geometry") rather than exhaustively reading every list entry — which
  is exactly the failure this register is built to make structurally harder (§1's HB entry now
  carries ten-plus synonyms precisely so a shape-worded proposal matches on the first grep).

**Sweep for OTHER instances (requested explicitly by the task):** every other thread in this
directory that opens near an existing exoneration was checked for whether it performed and RESPECTED
a genuine two-layer check, or (like the coupling read) stopped short. Findings:

- **`CLAIM_D1_P0WINDOW_20260805.md`** (a `p0`-orbital-parameter window, adjacent in vocabulary to
  HB's mass window) — explicitly tabulates HB as the "nearest exonerated item" and gives a
  file/stage/variable-level delimitation (`:76-79`) BEFORE opening. **Correctly delimited, not a
  near-miss.**
- **`CLAIM_P3_MKER_20260826.md`** (mass-kernel uncertainty budget, adjacent to
  [MASS-KERNEL-FAMILY]) — explicitly checks the exoneration's bound (+0.002) and scopes itself to
  "correctness-class, NOT bias-driver-class" before banking any H0-effect statement (`:54-61`).
  **Correctly delimited.**
- **`CLAIM_P3_IMPOSTOR_CONVENTION_20260822.md`** (adjacent to [PDET-NUM-ALONE] and
  [LCOMP-BNUM-DEFECT]) — explicitly names both items 6 and 10 by number and states why neither
  covers the new paired-convention question (`:38-52`). **Correctly delimited.**
- **`CLAIM_PRODUCTION_CALIBRATION_HARNESS_20260817.md`** (adjacent to [PDET-NUM-ALONE], #66/#67) —
  ratifies in its own decision table that it "does not re-open #66/#67" (`:169`). **Correctly
  delimited.**
- **`m2_residual_owner/CLAIM_M2_RESIDUAL_OWNER_20260807.md`** (adjacent to
  [LCOMP-BNUM-DEFECT], [WG-BOOKKEEPING-FIX], [MASS-KERNEL-FAMILY], [INFO-STARVATION], row #96
  cross-term) — runs a full table naming each adjacent exoneration and stating, per hypothesis, why
  it is a different estimand (`:150-159`). **Correctly delimited.**
- **`PREREGISTRATION_HIER_HTHETA_20260826.md`** (adjacent to C7/G2b and [INFO-STARVATION]) —
  dedicated §1.6 rule-1 section, names both, delimits by construction. **Correctly delimited.**

**Conclusion of the near-miss sweep: [WGEO]/HB is the ONE instance found in this sweep where a
rule-1 check was run, reported PASSED, and was subsequently shown (by a later, more careful read
within the SAME thread) to have missed a covering entry.** Every other thread checked in this
directory performed a two-layer check that named its adjacent exonerations explicitly and gave a
substantive (not just asserted) delimitation. This is a genuinely positive finding about the
record's general discipline — the failure mode is real but appears contained to the one instance
that prompted this register, not systemic across the sampled threads. (Caveat: this sweep did not
open every `PREREGISTRATION_*.md` and satellite `*_measure_*.md` file in the directory in full; it
targeted files whose own headers or ledger rows advertise a rule-1 check. A future, more exhaustive
pass could still surface others.)

---

## 7. STALE CROSS-REFERENCES FOUND

1. **`BIAS_HISTORY_LEDGER.md:130`** cites the layer-1 exoneration list as
   `"CLAIM_2D_BIAS_20260730.md:191-204"`. **The list is no longer at that location.** It currently
   begins at **line 721** (`## Exonerated — do NOT re-open without new evidence`) and its core
   sentence runs `:722-734`, with HB specifically at `:732-734`. This drift was already caught and
   flagged by [WGEO] itself (`CLAIM_WGEO_20260827.md:365-367`, §4.5, `"Citation drift, for the
   record"`) — carried forward here so it does not need re-discovering a third time.
2. **`HANDOFF_20260730.md:88-90`** cites `"handler.py:592"` (1D candidate-list construction, no mass
   filter) vs `"handler.py:605"` (2D mass filter applied). **[WGEO] (2026-08-27) found these have
   drifted** to `"handler.py:646"` vs `"handler.py:663"` respectively
   (`CLAIM_WGEO_20260827.md:170-179`, explicit: `"those line numbers have since drifted to :646 vs
   :663"`). The PHYSICAL claim (1D never sees the mass window) is unaffected; only the pointer moved.
3. **`m2_residual_owner/...` "A1/A2" labeling collision.** The string "A2" is heavily overloaded in
   the ledger across unrelated objects: (a) a battery-arm label in the row-#94 D1 pool-reweight
   three-arm test (`A0/A1/A2`); (b) an R² completion-weight functional in the M-2 residual-owner
   thread (row #97, `"A2: R² 0.88, chain ratios 0.666/0.653"`); (c) the MFG-A2 two-detection-model
   naming convention (rows near line 1234-1300); (d) the label attached to the NOW-VOID exoneration
   in row #90 (`"Fix A + Fix B path-A shrink … (A2 exoneration)"`, see §8 below). **A bare grep for
   "A2" in this ledger will return at least four unrelated objects.** Any future citation of "A2"
   without surrounding context is not resolvable from the string alone — quote the surrounding
   sentence, not the bare label.
4. **`RUNBOOK_NEXT_SESSION.md:113`** is flagged by the ledger's own §3 table as calling a FAIL
   verdict (seed600 shallow gate, criteria 3-4 failed) "MAP PASS" without the word FAIL — an
   internal inconsistency, not a line-number drift, but recorded here because a future reader citing
   that runbook line would inherit a wrong verdict label. Anchor:
   `"[AMBIG] — RUNBOOK_NEXT_SESSION.md:113 calls it 'MAP PASS' without the word FAIL"` —
   `BIAS_HISTORY_LEDGER.md` row 84.

**Not checked for drift (out of this sweep's scope):** the hundreds of other `file:line` citations
scattered through the ~3,000-line ledger and the satellite `README_*.md`/`*_READOUT.md` files. Only
citations that were directly load-bearing for an exoneration entry in this register were checked.

---

## 8. RETRACTED / VOID EXONERATIONS — do not cite as standing

**This category exists because the failure this register prevents cuts both ways: treating a DEAD
exoneration as ALIVE is exactly as dangerous as missing a live one.**

### [A2-VOID] — "Fix A + Fix B shrink the dark catalogue-leg channel difference"

- **MECHANISM (as originally believed, now void):** the expectation that landing the project's
  post-fix corrections ("Fix A" and "Fix B" — not further identified by name within this register's
  swept sources beyond the row-#90 label) would shrink or eliminate the dark-class catalogue-leg
  1D-vs-2D channel difference, removing it as a candidate bias-owner. This was tracked in the record
  under the shorthand "A2 exoneration."
- **STATUS: VOID.** Directly re-measured post-fix (both venues, paired cross-venue check, leg
  ablation): the channel difference **GREW**, from −504.8 to **−604.8 nats** (bit-identical N=534,
  +19.8%/event mean), and was found to be composition-dominated (81% from 316 scatter-resurrected
  events tilting 3.01× steeper than the robust stratum).
- **DATE/SOURCE:** 2026-08-04/05, gate (vii). Anchor: `"Fix A + Fix B path-A shrink / cannot move
  the dark catalogue-leg channel difference (A2 exoneration) … REFUTED — GREW −504.8→−604.8 …"`,
  outcome column: `"A2 exoneration void; D1 NOT demoted"` — `BIAS_HISTORY_LEDGER.md` row 90.
- **WHAT THIS MEANS FOR A FUTURE CHECK:** if any earlier document (outside this sweep's scope) still
  asserts the "A2" shrink-exoneration as standing, **it is superseded by row #90 and must not be
  cited**. This register lists it here specifically so a future grep for "catalogue leg exonerated"
  or "channel difference shrinks" surfaces the VOID status rather than an outdated positive claim.
  The companion ruling — "D1 NOT demoted" — means the D1 (`p0`-window) thread's standing was
  explicitly NOT weakened by this reversal; do not infer the opposite.

---

## 9. MAINTENANCE

- **Binding set = §1 ∪ §2 ∪ §3's caveats ∪ §4's confirmed-defect delimitations ∪ §8's void list.**
  A rule-1 check that reads only §1 (the layer-1 claim-file list) and skips §2 (the ledger's own
  supplemental list) reproduces exactly the class of failure that motivated this register — the
  union, not either list alone, is binding.
- **When adding a new entry:** use the same six fields (TAG, MECHANISM IN PLAIN TERMS with
  synonyms, BOUND, DATE/SOURCE by content anchor, WHAT IT DOES NOT COVER) and re-scan every existing
  entry's "WHAT IT DOES NOT COVER" field for whether the new entry changes any of them.
- **When citing this register:** cite by quoted content anchor (the entry's TAG heading, e.g.
  `## HB — hard mass window as support truncation`), not by line number — this file will drift too.

---

## ADVERSARIAL CHECK [OPUS-ORCH 2026-08-27]

Independent re-check of this register by the orchestrator, under the standing rule *"verifier output
is evidence, not authority"*. Every decisive number below was re-read from its source this session.
**Verdict: the register's CONTENT is sound; its GREP SURFACE and its NEAR-MISS CONCLUSION are not.**

### A. Usability test — PARTIAL PASS (6 probes, no false positives, one systematic defect)

Each probe used only words a fresh investigator would choose, before reading this file.

| # | probe mechanism (proposer's own words) | intended entry | landed? |
|---|---|---|---|
| 1 | "a linear-symmetric **mass cut** on a **log-normal** error model → asymmetric, **z-dependent** selection" ([WGEO]) | **HB** | **yes, but only on generic nouns** — `mass window` ✓ `one-sided` ✓ `asymmetric` ✓ `truncat` ✓; **`mass cut` 0, `z-dependent` 0, `log-normal` 0, `selection asymmetry` 0, `error model` 0** |
| 2 | "p_det is fit with a smoothed **KDE**; the smoothing biases H₀" | [PDET-CHOICE] | yes — `KDE` ✓ `detection probability` ✓; `kernel density` 0, `smoothing` 0, `detection efficiency` 0 |
| 3 | "host **peculiar velocities** are not removed from catalogue redshifts" | [PV-FRAME] | yes — `peculiar` ✓ `CMB` ✓ `heliocentric` ✓; **`peculiar velocity` (the exact phrase) 0** |
| 4 | "events with **no catalogue host** are dropped, biasing the ensemble" | [ZERO-HOST-FALLBACK] | yes — `dropped` ✓ `fallback` ✓ `no catalogue host` ✓; `zero host` 0, `hostless` 0 |
| 5 | "the completeness normalisation **β_G** disagrees with the discrete catalogue sum" | [OPTIONA-DRIFT] | yes — `completeness` ✓ `catalogue sum` ✓ `calibration drift` ✓; **`beta_G` 0, `Sigma_glob` 0** |
| 6 | control: "the LISA **antenna pattern** uses the wrong **arm length**" (nothing here covers it) | none | correct negative — `antenna` 0, `arm length` 0, `LISA` 0, `waveform` 0 |

**No probe landed on the WRONG entry, and no probe produced a false clearance.** Probe 1 — the one
that matters — reaches HB on the first generic noun. **The register works if its own §"How to use"
step 2 is followed** (grep the generic nouns: window · filter · cut · floor · truncation · clamp ·
eligibility). It does **not** survive a single grep of the proposer's own phrasing — which is
precisely what the coupling read did.

**FAILURE F1 — the hard wrap defeats the synonym lists.** This file is wrapped at ~100 columns, so
synonyms deliberately placed here to be greppable straddle a line break and are invisible to a
line-based `grep`. Confirmed 0-hit despite being present verbatim: **`sigma clipping`** (`:298-299`
— arguably the single most natural phrase for [WGEO]'s hypothesis), `peculiar velocity` (`:528-529`),
`log-mass draw` (`:172-173`), `kernel h-sensitivity` (`:479-480`), `B_num defective integral`
(`:458-459`), `truncated lognormal kernel` (`:209-210`), `spec-z money figure` (`:516-517`),
`sigma multiplier` (`:605-606`, in §4's WBHZERO entry). 31 "Search also" lists exist; most wrap.
**This is the register's core promise failing mechanically.** → `[DO]` **fix: put each "Search
also" list on one unwrapped line** (or one synonym per line). ~15 minutes, no content change.

**FAILURE F2 — Unicode-only tokens.** `β_G`, `Σ_glob`, `σ_z` appear only in Unicode; an agent typing
a grep in a terminal writes `beta_G` / `Sigma_glob` / `sigma_z` → **0 hits each**. (`Omega_m` is the
one entry that carries an ASCII alias, and it works.) → `[DO]` add ASCII aliases to every
Greek-lettered term.

### B. Bound spot-checks — 9/9 FAITHFUL, no wrong bounds

Re-read at source: [JACOBIAN] (ledger row 28, `:44`) · [FISHER-FRAME] (row 27, `:43`) ·
[VOLUME-TRUNC] (row 70, `:90`) · [MASS-KERNEL-FAMILY] (rows 72 `:92` / 89 `:109`) ·
[GRAY-MIX-AMPLIFY] (row 60, `:80`) · [HARD-CLAMP-OBSERVED-Z] (row 63, `:83`) ·
[ZERO-HOST-FALLBACK] (row 55, `:75`) · [A2-VOID] (row 90, `:110`) · [WBHZERO-ASYMMETRY] (row #196,
`:2915`); plus HA / HC / HB / [WINDOW-MEMBERSHIP] against `HANDOFF_20260730.md:63-64, 73-81, 85-88`
and `CLAIM_2D_BIAS_20260730.md:721-744`. **Every quoted number, sign and verbatim phrase is
faithful to its source**, including details the claim-file list does not carry (e.g.
[WINDOW-MEMBERSHIP]'s "realization-added 2D candidates", correctly sourced from `HANDOFF:63-64`).
One omission, minor: ledger row 72's own Open/Next column reads **"2D +0.025 open"**, which
[MASS-KERNEL-FAMILY]'s "WHAT IT DOES NOT COVER" does not carry.

### C. Near-miss sweep — **the register's conclusion does NOT survive** ✗

§6 concludes "[WGEO]/HB is the **ONE** instance found" and "the failure mode … appears contained to
the one instance, not systemic." **Both halves fail an independent check**, because §6 sampled only
threads that *advertise* a rule-1 check — survivorship bias, admitted as a method caveat but not
carried into the conclusion.

1. **A SECOND failed rule-1 check, same failure mode, one week earlier.**
   `CLAIM_P3_WBHZERO_20260825.md:34-36` states: *"**Exoneration check:** no prior exoneration covers
   the candidate mass filter (searched the claim files' Exonerated lists + ledger §2 — the with-BH
   channel's candidate ELIGIBILITY has never been an arm in any campaign)."* This is **false on its
   face**: **HB** *is* an exoneration of the candidate mass filter, and **[WINDOW-MEMBERSHIP]** is
   literally about candidate eligibility — both sit in the list the card says it searched
   (`CLAIM_2D_BIAS_20260730.md:726-727, :732-734`). The card's own Gate-B pass later *amended* the
   exoneration check (`:57-59`) toward `CODE_INVENTORY.md §7` — and **still never named HB**.
   *Materiality: benign.* WBHZERO is genuinely a different object (σ-multiplier asymmetry, not
   window presence), so no scientific harm followed. But the **check itself failed identically** to
   the coupling read: it searched its own thread's vocabulary and reported a clean pass.
   ⇒ §4's line *"They were correctly kept separate throughout the record"* **overstates**: they were
   separate objects by luck of the physics, not by any check that ran.

2. **Three post-2026-07-30 claim cards carry NO rule-1 / exoneration check at all** (not a wrong one
   — none): `CLAIM_F0_SEL_20260825.md` (225 lines, reaches a drafted verdict),
   `CLAIM_B0_FINITE_MOMENT_20260824.md` (437 lines; its §3.4 is a *non-reproposal* check against its
   own thread's alternatives, which is a different discipline), `CLAIM_P3_2D_20260825.md` (475
   lines). `CLAIM_F0_SEL` is the notable one: its object — *"UNMODELED — one-sided, structurally so
   … no counterpart anywhere in `p_det`/Σ"* (`:19-21`) — sits in the same family as
   [WBHZERO-ASYMMETRY]'s "unmodeled ONE-SIDED numerator selection" and adjacent to [DEPTH-TRUNC] and
   [HARD-CLAMP-OBSERVED-Z], and it never checks an exoneration. (Its verdict is *small* — 0.13% /
   0.59% of the pool — so materiality is again low; the missing check is the finding.)

3. **Confirmed sound:** the register's positive readings of `CLAIM_P3_MKER_20260826.md` (heading
   `## 4. Delimitation against the standing exonerations (hard-rule-1 check, PASSED with scope)` at
   `:54`) and `CLAIM_D1_P0WINDOW_20260805.md:79` (*"Different file, different stage, different
   variable"*) are accurate and substantive.

**Corrected §6 conclusion:** of the threads that *ran* a rule-1 check, **2 of 8 failed** (WGEO
coupling read; WBHZERO intake), and **3 further cards ran none**. The failure mode is **not**
contained to one instance.

### D. HB reconciliation — ADJUDICATED: **different quantities, settled from source. Not an inconsistency.**

| | **0.063 %** | **1.5 %** |
|---|---|---|
| numerator | **−0.317 nats** — h-tilt of the *counterfactual removal* of the mass window, h 0.73→0.81 | **+0.24 nats** — *static share* of the observed budget carried by the 487 events whose 2D catalogue term is identically zero at every h |
| denominator | **504.8 nats** — dark-class `Σ ln(L_cat,2D/L_cat,1D)`, **catalogue leg only**, 534 events, −3165.7 → −3670.5 (`HANDOFF_20260730.md:55`, `:86`) | **15.83 nats** — dark-class share of the **full per-event mixture** channel difference (C3 split +2.97 / +15.83 / +18.80) (`ADJUDICATION_20260730.md:9`, `:11`, `:61`) |
| record's own framing | — | *"HB (its hard-zeros are worth 1.5% of the target — **corroborates its self-refutation**)"* (`ADJUDICATION_20260730.md:275`) |

Different numerators, different denominators, different accounting objects — and the two denominators
differ by ~32× for a coherent physical reason: the catalogue leg enters the mixture with weight
0.0354 → 0.0061 (`ADJUDICATION_20260730.md:65-66`), so a catalogue-leg-only tilt is far larger than
the mixture-level tilt it produces. `ADJUDICATION:66` closes the budget exactly
(`+15.83 = 0 + 19.10 − 3.27`). **Nothing needs re-measuring.**

**ROOT CAUSE, and a defect in THIS register:** the informal label **"the target"** is used in the
record for **both** denominators — `HANDOFF:86` *"0.063% of the 504.8-nat target"* vs
`ADJUDICATION:275` *"1.5% of the target"* (= 15.83). **This register reproduces the collision
verbatim at `:304` and `:311` and never disambiguates.** → `[DO]` amend HB's BOUND field to name
each denominator inline.

**The +0.010 vs +0.0015 "factor ≈ 7":** an artifact of `CLAIM_WGEO_20260827.md §4.4` (`:356-361`),
which calls both figures *"the window-removal counterfactual"*. They are two different perturbations
of the same code object: **+0.010** = removal of *realization-added 2D candidates*
(`HANDOFF:63-64`); **+0.0015** = removal of *the window* (`HANDOFF:87-88`). §4.4 also **mis-cites**
the membership bullet as `:729-731` (those lines are HA; the actual range is **`:726-727`**) and so
contradicts WGEO's own §4.3, which cites it correctly. **This register does not carry the
conflation** — its [WINDOW-MEMBERSHIP] and HB entries keep them separate. ✓

**ONE MAGNITUDE REMAINS UNSOURCED — flagged, not smoothed.** *"~40–50× too small at its ceiling"* is
asserted at `HANDOFF:87` and copied into `CLAIM_2D_BIAS:734`, `CLAIM_D1_P0WINDOW:79`, `CLAIM_WGEO`
and **this register (`:302`, `:311`, `:314`, `:330`)**. **No document in the tree derives it**, and
it matches neither 0.063 % (→ ~1600×) nor 1.5 % (→ ~67×). *Orchestrator's reconstruction* **[INFER,
not in the record]**: it is the **MAP-space** form of the same counterfactual — the bias to be
cancelled is **+0.077 in h** (`HANDOFF:25`), a fix must move the MAP by **−0.077** (`HANDOFF:76`),
HB's removal delivers **+0.0015**, and `0.077 / 0.0015 = 51×`. If that is right, HB's bound mixes
two accounting spaces (nats-ratio 0.063 %, MAP-ratio ~50×) under one sentence — a third instance of
the same label-collision hazard. **The verdict is robust in both spaces** (too small by 1.5–3 orders
of magnitude, sign-inverted), so nothing is at risk; but *"~40–50×"* should not be quoted as a
banked number until the author confirms the space. → `[RULE]` for the author.

### E. New stale citations (add to §7)

5. **Ledger row numbers 90, 91, 92, 93 and 94 are each used TWICE** for unrelated objects — an
   08-04/05 block (`BIAS_HISTORY_LEDGER.md:110-114`) and an 07-27/29 block (`:116-120`). The table
   holds **103 numbered rows but only reaches #98**. §8's own citation *"row 90"* for [A2-VOID] is
   therefore ambiguous between the gate-(vii) A2-void row (`:110`) and the (d1) z×M_z NULL (`:116`).
   §7 item 3 caught the overloaded **label** "A2" but not that the **row number** on that very row is
   duplicated. Cite row 90 as *"row 90 (08-04/05 gate (vii))"*.
6. **Off-by-one, on-theme:** §7 item 1 (and `CLAIM_WGEO_20260827.md:365`) cite the stale pointer as
   `BIAS_HISTORY_LEDGER.md:130`. It is at **`:129`**.

### F. What is unchanged by this check

The exoneration **content** is sound. No entry was found with a wrong bound, a misattributed
mechanism, or a fabricated anchor; the §5 "nothing could not be located" claim holds. **HB stands as
refuted**, and nothing here reopens it or any other entry.
