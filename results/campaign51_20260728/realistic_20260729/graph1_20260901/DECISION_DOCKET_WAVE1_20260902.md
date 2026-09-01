# Research Graph 1 — Wave-1 End Docket

Date: 2026-09-02. Author of record for all scientific decisions: Jasper Seehofer.
Status: DECISION DOCKET — a reviewable decision artifact per `CLAUDE.md` "Proposing decisions."
Compiled mechanically from the wave-1 execution records; zero fresh numbers computed here. Sources
are quoted verbatim and cited by path; anything not quoted from a source is marked
ORCHESTRATOR-DERIVED. Tags and grant vocabulary follow the approval-scope convention (`CLAUDE.md`
"Approval scope — tag every item in a decision list"): **[DO]** → "Approved"; **[RULE]** →
"Ratified"; **[STANDING]** → "Granted". Per the binding default, an approval never propagates to a
disposition whose inputs did not exist when it was given — every item's NOT-covered cell says what
still returns later regardless of how this docket is answered.

Committed by the chair; the author rules on it via the decisions table (one-word replies).

---

## Decisions table (one-word replies suffice)

| # | item | tag | ask | explicitly NOT covered (returns as fresh RULE) |
|---|---|---|---|---|
| 1 | d-rphi-retire | RULE | Ratified — retire `c-rphi-mismatch` from the open-branches board | any future re-opening of the pre-flip `r_φ` mismatch as a live claim; the historical `COUNTERFACTUAL` branch's own status |
| 2 | d-s4-review (whole item) | RULE | Ratified — the r-b82-s4 re-frozen bands + stop rule, as amended by 2a–2f | the actual S4/S5 production-N launch (stays behind B8_2 design §8 regardless of this ruling); any in-band/out-of-band verdict once S3 re-runs |
| 2a | — null-referenced no-BH bands + F sanity range [1,25] | RULE | a word | the read itself once S3 re-runs |
| 2b | — stop rule (n_U_min 60/16, 86400s×≤3, INCOMPLETE-RUN path, resume-to-complete) | RULE | a word | — |
| 2c | — seed re-use election (paired pre/post + with-BH byte-pin; discharges row #291's deferred g-byte-id criterion; block 901100+ stays reserved) | RULE | a word | the byte-pin result itself |
| 2d | — population-tag amendment (`_population_tag` resolved-flag token, follow-on build item if granted) | RULE | a word | the follow-on build, if granted |
| 2e | — comparand interim rule (§4: PROD-A0 vs locally recomputed post-flip HEAD, retro-flag on mismatch once banked) | RULE | a word | the retro-flag event itself |
| 2f | — confirm ratifying this document at d-s4-review IS the row #290 row-2 "band re-freeze and stop rule" ruling, edits = revision 1 | RULE | a word | — |
| 3 | d-jr1-band (whole item) | RULE | Ratified — the joint_r1 registered band + grid scope, as amended by 3a–3d | any actual MAP/mean read once m-joint-r1-mass-aware runs; promotion of `c-auto-default-venue-general` |
| 3a | — band width: ratify/amend PROPOSED `map_h ∈ [0.64,0.70]` AND `mean_h ∈ [0.64,0.70]` (MAP-AND-mean) | RULE | a word | the actual verdict once the run lands |
| 3b | — grid election: H_GRID_41, conditionally extended to G-EXT iff b-hprior-fix byte-identity green; plus §7 item(2)'s interim-comparand question | RULE | a word | resolves automatically to H_GRID_41-only if item 4 below is not granted or lands red |
| 3c | — secondary-read bands (§4 items 1–2) as verdict-free diagnostics only | RULE | a word | — |
| 3d | — whether an out-of-band-LOW read (<1.021) alone escalates to a registered mechanism arm (candidate-set composition, caveat C4) or stays evidence | RULE | a word | the escalation itself, if granted |
| 4 | h-prior decoupling (fresh, row #293) | RULE | Choose (a), (b), or (c) | the decoupling design's own detail once chosen — that returns as its own physics-change gate item |
| 4a | — chosen mechanism | RULE | (a) decouple [chair-recommended] / (b) drive host window off eval grid / (c) park Branch I | the chosen design itself (fresh `/physics-change` gate) |
| 4b | — rerun cost overrun (23.8 CPU-h vs ≤20 ORCHESTRATOR-DERIVED cap) | RULE | a word — needed only if Branch I proceeds (i.e., not if (c) is chosen) | — |
| 5 | Status annex | — | no decision asked | — |

---

## 1. d-rphi-retire — retire `c-rphi-mismatch` from the open-branches board

**[RULE]** — grant word: **Ratified**.

Per row #290 (`results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`),
`d-rphi-retire` is item 12's "board retirement itself," explicitly listed as NOT covered by the
batch-1 charter ratification and returning "with the note." That note is now in hand.

**Requires-manifest** (`RESEARCH_GRAPH_1_PROPOSAL_20260901.md` §1.11, `d-rphi-retire` implicit via
row 12's g-znorm gate): rd-rphi-note done, g-znorm green. Both satisfied.

**Evidence, quoted verbatim** (row #292,
`results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`; source
`graph1_20260901/exec/rd-rphi-note/RECORD.md`):

> "rd-rphi-note (closure) COMPLETE — g-znorm GREEN, first standing panel evaluation on the flipped
> 1D catalogue leg; `d-rphi-retire` UNBLOCKED."

> Measured deviation on the production divisor identity: **0.0 (exact)** — `bayesian_statistics.py:6125-6126`,
> `global_denom_no_bh` is a literal Python reassignment `= global_denom_with_bh` under
> `catalogue_leg_1d_mass_aware == "on"`, so "the deviation is not merely small, it is not computed
> at all — numerator and divisor share one float value by construction."

> The local numeric check ... re-derived the raw floats: **`Z_on = 1.0`, `|Z_on - 1| = 0.000e+00`**,
> against the discriminating control **`Z_off = 1.0169076423251329`, `|Z_off - 1| = 0.016908`** —
> confirming the "off" leg genuinely fails the identity so the "on" pass is not degenerate.

> ... closing c-rphi-mismatch (the pre-flip `Σ^φ` vs `Σ_4D`/`Σ³ᴰ` mismatch, `r_φ ≈ 0.886`/`0.9119`)
> "by construction, not by improved agreement" ... the pre-flip "off" branch, now logged
> `COUNTERFACTUAL` at `bayesian_statistics.py:4431-4437`, still carries the historical mismatch.

> Chair independently verified the `bayesian_statistics.py:6125-6131` identity reassignment
> against the live file. No code edited, no commit, no cluster job. `d-rphi-retire` returns to the
> author WITH this note ... this record does not itself retire c-rphi-mismatch.

**NOT covered by a Ratified reply:** any future re-opening of the pre-flip `r_φ` mismatch as a
live claim on a different code path; the historical `COUNTERFACTUAL` branch's own disposition
(it is not deleted, only inert under the flipped default).

---

## 2. d-s4-review — ratify r-b82-s4's re-frozen bands + stop rule

**[RULE]** — grant word: **Ratified** (whole item), plus one word each for sub-items 2a–2f.

**Requires-manifest** (`RESEARCH_GRAPH_1_PROPOSAL_20260901.md` §1.11): "r-b82-s4 design-gate
record done." Satisfied — draft complete, "design-validity only and blind to results" per row #297.

Source: row #297 (`BIAS_HISTORY_LEDGER.md`; `graph1_20260901/exec/r-b82-s4/REGISTRATION_DRAFT.md`).
Binding premise, quoted: "the pre-flip pilot numbers (row #288/#291) are stated to be 'motivation
and instrument anchors only, never calibration' of the post-flip no-BH channel ... so every no-BH
band in the draft is **null-referenced**, not pilot-referenced; the sole legitimate pre-flip
carry-over is the untouched with-BH channel (§2.3)."

### Proposed bands / stop rule, compact

- **Cell S (§2.1):** PIT-KS D ≤ exact 5% Kolmogorov critical value at realized n_U (≤0.134 at
  n_U=100, PRIMARY); HPD coverage 50/68/90/95 within exact Binomial(n_U, level) 2σ bands;
  mean(MAP)−h_true and score-zero-at-truth both |Z| ≤ 3; F_no_bh = SD/floor(200) REPORTED, no
  verdict band, sanity flag F outside [1, 25] → anomalous read, STOP, fresh RULE.
- **Cell T (§2.2):** width-only by design — "No coverage/PIT/verdict claim from cell T ever," read
  via row #291's `score_ratio_t_over_s()`; no-BH T/S REPORTED-ONLY.
- **With-BH byte-identity pin (§2.3):** re-using pilot seed blocks 901000-901099 (S) / 902000-902024
  (T) in a fresh work root makes every with-BH per-universe checkpoint a structural byte-identity
  check, "Volume of pairs ... ≥ 6.8e5 shared values ≥ the 1e5-pair g-byte-id criterion (infra 2.5)
  that row #291 explicitly deferred" — PROPOSED to discharge that deferred criterion if green.
- **Stop rule (§3):** COMPLETE = `stopped_reason="exhausted_n_universes"` or cumulative checkpoints
  ≥ registered n_U (100 S / 25 T), resume-to-complete allowed, ≤3 invocations × 86400 s/cell;
  **WALL-LIMITED-VALID** at realized n_U ≥ n_U_min = **60 (S) / 16 (T)**; below that,
  **INCOMPLETE-RUN** — no read, fresh RULE, "the word 'starved' is never used in any harness
  verdict"; sidecar absent on a fresh post-flip cell → INSTRUMENT-DEFECT.
- **Comparand rule (§4):** m-s3 cell launches "do not block on the rebaseline being banked ...
  running against a locally recomputed post-flip HEAD as an interim disclosed substitute" when the
  m-head-rebaseline CSV is not yet banked, retro-flag on mismatch once it is.

### §8 open questions, quoted verbatim (sub-items 2a–2f)

> "(1) ratify the null-referenced no-BH bands and the F sanity flag range [1,25];" — **2a**

> "(2) ratify the stop rule (n_U_min 60/16, 86400 s × ≤3 invocations, INCOMPLETE-RUN path,
> resume-to-complete semantics);" — **2b**

> "(3) ratify the seed re-use election — paired pre/post read + with-BH byte-pin, including
> whether its green discharges row #291's deferred g-byte-id criterion — and confirm block 901100+
> stays reserved for the falsifier;" — **2c**

> "(4) ratify the population-tag amendment (extend `_population_tag` with the resolved-flag token,
> a follow-on build item if granted);" — **2d**

> "(5) ratify the comparand interim rule (§4);" — **2e**

> "(6) confirm that ratifying this document at d-s4-review IS the row #290-row-2 'band re-freeze
> and stop rule' ruling, with any edits recorded as revision 1." — **2f**

**NOT covered by these Ratified replies:** the actual S4/S5 production-N cell launch — stays behind
"B8_2 design section 8 regardless" per §1.11; any in-band/out-of-band coverage verdict, which only
exists once S3 actually re-runs.

---

## 3. d-jr1-band — ratify the joint_r1 registered band + grid scope

**[RULE]** — grant word: **Ratified** (whole item), plus one word each for sub-items 3a–3d.

**Requires-manifest** (`RESEARCH_GRAPH_1_PROPOSAL_20260901.md` §1.11): "dv-jr1-transform done;
r-jr1-massaware draft done." Both satisfied per row #296.

Source: row #296 (`BIAS_HISTORY_LEDGER.md`; `graph1_20260901/exec/dv-jr1-transform/DERIVATION.md`,
`graph1_20260901/exec/r-jr1-massaware/REGISTRATION_DRAFT.md`). r-jr1-massaware is "stamped
'EVERYTHING BAND-SHAPED HERE IS PROPOSED, NOT FROZEN.'"

### Derived transform, quoted

> "the derived joint_r1 T2.2b-equivalent transform under the log-normal realized-forward mass law
> gives realized-median ≈1.031, quoted from the h-stability table — **1.0316 / 1.0314 / 1.0312 at
> h = 0.725 / 0.730 / 0.735, spread 4e-4** ('h-stable at the same order as the iiib transform,'
> T2.2b-parity vs iiib's own spread 6e-4) — against iiib's delta-law comparand **1.039** (row #282);
> 95% MC predictive band **[1.021, 1.036]**."

### Proposed band and grid, quoted

> "PROPOSED band **map_h ∈ [0.64, 0.70] AND mean_h ∈ [0.64, 0.70]** (Z-CONFIRMED iff both in-band,
> A18 map-AND-mean rule form), REFUTED iff map_h ≤ 0.605 with the C-C pin intact, neither band →
> INTERMEDIATE → 'returns to the author as a fresh [RULE],' max_revisions 2 (ORCHESTRATOR-DERIVED,
> charter-ratified)."

> "Grid election: **H_GRID_41, extended to the 55-node G-EXT grid CONDITIONAL on b-hprior-fix's
> byte-identity gate going green** — coupling stated explicitly here: b-hprior-fix is itself
> BLOCKED-ON-FRESH-RULE per row #293, so 'if b-hprior-fix is red or not yet green at launch: run
> H_GRID_41 alone; the matched-class sub-read is then reported as a censored bound, disclosed, and
> does NOT block the primary MAP/mean verdict.'"

### §8 open questions, quoted verbatim (sub-items 3a–3d)

> "1. Ratify or amend the PROPOSED band [0.64, 0.70] (MAP-AND-mean)..." — **3a** (band width)

> "2. Ratify the conditional grid election (§3), including the interim-comparand question in §7
> item (2)." — **3b** (grid election + interim comparand)

> "3. Ratify the secondary-read bands (§4 items 1-2) as verdict-free diagnostics." — **3c**
> (secondary diagnostic bands)

> "4. Rule whether an out-of-band-LOW transform read (< 1.021) alone should escalate to a
> registered mechanism arm (candidate-set composition, caveat C4) or remain evidence." — **3d**
> (C4 escalation question)

**NOT covered by these Ratified replies:** the actual MAP/mean verdict once m-joint-r1-mass-aware
runs and lands in-band or not; promotion of `c-auto-default-venue-general`, explicitly listed as
NOT covered at branch-head grant (row #290 decisions-table row 5).

---

## 4. h-prior decoupling — fresh [RULE] from row #293

**[RULE]** — grant word: choose one of (a)/(b)/(c) for 4a, plus a word for 4b if applicable.

This item did **not** exist at row #290 and is not covered by that ratification. It is a fresh
chair adjudication surfaced during b-hprior-fix execution.

**Why the drafted fix cannot land, quoted** (row #293,
`results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`; source
`graph1_20260901/exec/b-hprior-fix/RECORD.md` plus chair-derived reading):

> "b-hprior-fix (Branch I) RECON COMPLETE — BLOCKED-ON-FRESH-RULE, no edit made; chair adjudication
> finds the record's flagged second-order risk DISPOSITIVE."

> The record's own verdict (§4): "Trigger-file-required. The fix cannot be made in run
> configuration outside `cosmological_model.py`: `BayesianStatistics.__init__` constructs
> `LamCDMScenario()` unconditionally with no override path, and the 0.86 ceiling is a hardcoded
> dataclass-field literal at `cosmological_model.py:388`."

> The record flags, but does not resolve, a second-order risk at §2.5: `h.upper_limit` is also read
> at line ~5716 feeding a `z_max`/`redshift_upper_limit` clamp, "since a wider `h_max` could in
> principle shift a `min(z_max, redshift_upper_limit)` clamp even for an in-bound h evaluation."

**ORCHESTRATOR-DERIVED (chair adjudication, quoted, not in the record):**

> "independent reading of `bayesian_statistics.py:5716` confirms this risk is dispositive, not
> merely theoretical — the call site is `get_redshift_outer_bounds(z_max = dist_to_redshift(d_L+3σ,
> h_max))`, monotone increasing in `h_max`, and the code's own line-1255 comment establishes the
> ~1.5 clamp never bites for `h_max ≤ 0.86` (`z_max(h≤0.86) ≤ ~1.33`). Raising `upper_limit`
> 0.86→1.00 therefore widens every detection's candidate-host window for IN-BOUND evaluations, and
> the ratified g-byte-id gate (0 mismatches required below 0.86, per the record's own §2.6 plan)
> would go red. The drafted one-line edit (`cosmological_model.py:388`, `upper_limit=0.86 → 1.00`)
> therefore CANNOT land as drafted."

### Options, presented neutrally (4a)

- **(a) Decouple** — build a new admissibility mechanism for the G-EXT h-grid, kept structurally
  separate from the host-window `z_max` bound that `h.upper_limit` currently also drives.
  Requires a `/physics-change` gate on the chosen design (any edit to `cosmological_model.py` is a
  physics-trigger file per `CLAUDE.md`). **ORCHESTRATOR-DERIVED recommendation: (a).** Preserves
  the byte-identity gate below 0.86 with zero risk to the certified region and confines the new
  surface to the extension itself.
- **(b) Drive the host window off the actually-evaluated h-grid** — widen the physics surface so
  `z_max` tracks the true per-evaluation h rather than the static `upper_limit` ceiling. Larger
  change footprint; touches the same trigger file with a broader blast radius than (a).
- **(c) Park Branch I** — G-EXT stays unusable this batch. Consequence, quoted from row #296
  (coupling stated explicitly there): "if b-hprior-fix is red or not yet green at launch: run
  H_GRID_41 alone; the matched-class sub-read is then reported as a censored bound, disclosed, and
  does NOT block the primary MAP/mean verdict" — i.e. r-jr1-massaware's grid election (item 3b
  above) falls back to H_GRID_41 automatically.

### 4b — rerun cost overrun, needs a word only if Branch I proceeds ((a) or (b) chosen)

> "'14 × 1.7 = 23.8 CPU-h, which is ~5 CPU-h over the stated ≤20 CPU-h bound' for the scoped rerun
> of tasks 41-54 against `cluster/a18_ma1d_headreadout_iiib.sbatch --array=41-54`."
> (row #293, quoting `graph1_20260901/exec/b-hprior-fix/RECORD.md` §2.5.)

The ≤20 CPU-h figure is the record's own stated bound, carried forward; whether it is exceeded and
by how much is arithmetic on record numbers, not a fresh measurement.

**NOT covered:** the chosen decoupling design's own detail — that is a fresh `/physics-change` gate
item once (a) or (b) is chosen, per the trigger-file rule in `CLAUDE.md`.

---

## 5. Status annex — no decision asked

Wave-1 execution summary, one line per row (`BIAS_HISTORY_LEDGER.md`):

- **Row #291** — b-s4-harness-repair COMPLETE: three S4 defects repaired in `b8_cal_harness.py`
  (seed-population separation, cell-T invocation, stop-reason sidecar); clean cell-S re-score
  `F_no_bh=7.450, F_with_bh=11.38` (n=63); cell-T clean `F_no_bh=11.27` (n=20); T/S ratios
  `no_bh: 1.517`, `with_bh: 0.9984`; 9/9 tests, ruff/mypy clean, chair-reproduced; not committed at
  that point.
- **Row #292** — rd-rphi-note COMPLETE: g-znorm GREEN (`Z_on=1.0`, exact), `d-rphi-retire`
  unblocked; no code edited.
- **Row #293** — b-hprior-fix RECON COMPLETE, BLOCKED-ON-FRESH-RULE per chair adjudication (host-
  window coupling at `bayesian_statistics.py:5716` dispositive); no edit, no cluster submission.
- **Row #294** — m-head-rebaseline + m-t5-armS LAUNCHED: cluster-checkout-behind blocker (`38cc0f58`
  missing row #286's flip commit) found and repaired via `git pull --ff-only`; Lustre OST5
  confirmed cleared; four SLURM jobs submitted (6764460–6764463), all dataset checksum pins
  STOP-gated and verified, fresh out-roots confirmed absent pre-submission.
- **Row #295** — Branch D wave-1 sequence COMPLETE: rd-runner11 read confirms b-axis matches row
  #287 to full precision (`Z=-1.808`), s-axis absent-by-design (not requested); b-pahier33-scorer
  build adds the PA-HIER-33 null estimator + the driver's `build_iiib_venue()`; full suite
  `2016 passed, 15 skipped, 30 deselected, 0 failed`, ruff/mypy clean, chair-reproduced a 32/32
  subset; N≥1e5 byte-identity check explicitly deferred as the residual precondition for
  m-s0b-production; both changes uncommitted at that point.
- **Row #296** — Branch C wave-1 authoring COMPLETE: dv-jr1-transform derivation (realized-median
  1.031, spread 4e-4, band [1.021,1.036]) + r-jr1-massaware PROPOSED registration draft; no
  trigger-file touch, no code edit, no cluster job; chair structural review done, full adversarial
  re-derivation reserved for the wave-3 end-verifier.
- **Row #297** — r-b82-s4 REGISTRATION DRAFT COMPLETE (Branch A wave-1 node set now fully
  executed): PROPOSED throughout, design-validity only, blind to results; ratification reserved to
  d-s4-review (item 2 above); chair structural review done.

**Cluster jobs 6764460–6764463** (per the two LAUNCH_RECORD.md files under
`graph1_20260901/exec/m-head-rebaseline/` and `graph1_20260901/exec/m-t5-armS/`), state as of those
launch records:

| Job | Purpose | Array | State as of launch records |
|---|---|---|---|
| 6764460 | C0-prime gate (h=0.730, both venues) | 0-1 | C0-prime **completed**; the g-c0-baseline evaluation against `wave3_20260830/{iiib,joint_r1}` task-21 outputs is the orchestrator's own readout step, not computed by the job — **IN FLIGHT / pending, not fabricated here** |
| 6764461 | Blind HEAD, iiib, full H_GRID_41 | 0-40 | submitted; chair monitors, does not poll — no completion state asserted here |
| 6764462 | Blind HEAD, joint_r1, full H_GRID_41 | 0-40 | submitted; chair monitors, does not poll — no completion state asserted here |
| 6764463 | T5 Arm S k-scan (4 k-values × 4 H4 nodes) | 0-15 | submitted; chair monitors, does not poll — no completion state asserted here |

Both launches were STOP-gated on dataset checksum pins (CRB md5 `9a1f2a14384a9281c97ca3be312ddaab`,
catalogue md5 `c52c13b5cab61f6b3f04bbe202550969`; joint_r1 additionally sha256
`e8f7ab310ea70ddfdd3b81970dc99ad943808e6b6c128777bb085db01b4f6751`), with fresh out-roots verified
absent before submission in both records.

**Commit:** `97b2062a` — "graph1 wave 1: rows #291-#296 — S4 harness repair, PA-HIER-33 scorer +
iiib venue path, g-znorm GREEN closure, jr1 transform 1.031 + registration draft, h-prior
BLOCKED-ON-RULE, cluster launches 6764460-63." Note: this commit's own message covers rows
#291–#296; row #297 (r-b82-s4 draft) postdates it and is not yet reflected in a commit as of this
docket's compilation.
