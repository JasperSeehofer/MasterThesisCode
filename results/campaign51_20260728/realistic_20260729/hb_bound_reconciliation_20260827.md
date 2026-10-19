# HB bound reconciliation — +0.010 vs +0.0015 — 2026-08-27

**Scope:** record-only. No `darksiren_emri/` file touched, no measurement re-run, no
bound re-derived. This is a records read that determines *what the existing text
says*, not a new verdict on HB.

**Answer up front:** the two numbers are **not the same quantity**. `+0.010`
(`CLAIM_2D_BIAS_20260730.md:726–727`) is the exoneration of **"candidate-window
membership"**, a bulleted item in the exoneration list that is textually and
mechanistically distinct from **HB**, which sits two bullets later at
`CLAIM_2D_BIAS_20260730.md:732–734`. HB's own quoted bound is `+0.0015`
(`HANDOFF_20260730.md:87–88`), and it is internally consistent everywhere it is
cited. The apparent "factor ≈7 inconsistency" flagged in
`CLAIM_WGEO_20260827.md` §4.4 is itself a citation error — it treats two
different bulleted exonerations as if they were one repeated measurement. This
is the *same* failure mode the task brief describes for the stage-0 rule-1 check:
matching on adjacency/similar wording ("window removal") rather than on the
physical object being tested.

---

## 1. The two quoted values, in full context

### 1a. `+0.010` — NOT HB. This is "candidate-window membership."

`CLAIM_2D_BIAS_20260730.md:721–734`, the "Exonerated — do NOT re-open without new
evidence" list, verbatim (line numbers from `nl -ba`):

```
726  both channels) · candidate-window **membership** (exact removal moves MAP
727  0.81→0.82, wrong sign) · mass-kernel **family** (bounded +0.002) · **Option-A
...
732  nonzero, 0 excluded events in all 16 combined posteriors) · **HB** hard mass window
733  as support truncation (tilt −0.317 nats = 0.063% of the target, sign-inverted,
734  40–50× too small).
```

These are **two separate `·`-delimited bullets** in the same list: "candidate-
window **membership**" (:726–727) and "**HB** hard mass window as support
truncation" (:732–734), six bullets apart, with "mass-kernel family," "Option-A
calibration drift," "HA as the bias owner," and "HC mixture-floor/zero-handling"
in between. Nothing in the passage equates them.

The origin document, `HANDOFF_20260730.md:63–64`, states what "candidate-window
membership" actually tested:

```
63  candidate-window **membership** — exact removal of realization-added 2D
64  candidates moves MAP 0.81 → 0.82, wrong sign.
```

I.e. the counterfactual removed **specific candidates that entered the search
pool at the realization step** ("realization-added"), not the mass-eligibility
cut itself. `CLAIM_D1_P0WINDOW_20260805.md:80` later characterizes it precisely:
> "Same estimator-side object as HB, tested by removal. Says nothing about a
> filter upstream of the CSV the estimator reads."

I.e. even the record's own disambiguation treats "membership" and "HB" as **the
same code-level object probed by two different perturbations** (partial
candidate-set removal vs. full window/support-truncation removal), not as the
same test.

### 1b. `+0.0015` — HB's own number, sign-inverted MAP shift under full window removal

`HANDOFF_20260730.md:85–96` (the "⚠ UPDATE" block that supersedes the older HB
write-up further down the same file), verbatim:

```
85  > **HB was subsequently REFUTED** (self-refuted by its own investigator): the
86  > truncation's h-tilt is −0.317 nats over 0.73→0.81 = **0.063%** of the 504.8-nat
87  > target, ~50× too small at its ceiling, and **sign-inverted** (removing the window
88  > moves the MAP *up* by ~+0.0015). Two framing assumptions in this section were also
89  > wrong: ...
```

The stated quantity: removing the **entire hard mass-eligibility window**
(`handler.py:594-603`, per `HANDOFF_20260730.md:102`) — turning the eligibility
cut off altogether, not deleting specific realization-added candidates — moves
the 2D MAP up by ≈ +0.0015 in raw `h`, against a target correction that would
need to move it **down** toward 0.73. Same sentence also gives the nats-domain
figure: −0.317 nats over the same h-range, stated as 0.063% of a **−504.8-nat**
"target" (defined at `HANDOFF_20260730.md:55`: `Σ ln(L_cat,2D/L_cat,1D)` for the
534-event dark-class catalogue-leg survivors, 0.73→0.81 — a candidate source of
the dark-class tilt that C4 later re-attributes; see `HANDOFF_20260730.md:109`
and `CLAIM_2D_BIAS_20260730.md` C4).

**No other document quotes a different magnitude for this specific
measurement.** `ADJUDICATION_20260730.md:275–276` and
`CLAIM_2D_BIAS_20260730.md:742–744` both *strengthen* HB with a second, distinct
figure (§3 below) rather than re-quoting a shift value, and both are consistent
with `+0.0015`/`−0.317`/`0.063%`/`~50×`.

---

## 2. Determination — (b) different quantities, later conflated

This is case **(b)**: two different counterfactual constructions on a shared
code object, subsequently misread as one measurement.

- **Different perturbation.** `+0.010` deletes a *specific subset* of candidates
  (those added to the search pool at the realization step). `+0.0015` disables
  the *entire hard eligibility cut*, changing which candidates the likelihood
  ever sees at all, for every realization. A window-removal superset does not
  have to produce a larger MAP shift than a targeted subset-removal — the two
  probe different mechanisms (candidate-set membership vs. the window's role as
  a hard-vs-soft support boundary) and there is no textual claim anywhere in the
  record that they should agree.
- **Different label, different location, never equated.** The claim file lists
  them as separate `·`-delimited items; the ledger (`BIAS_HISTORY_LEDGER.md:135`)
  lists "HB" and (implicitly, via the claim-file pointer) membership as separate
  entries in the same enumeration; `CLAIM_D1_P0WINDOW_20260805.md:79–80`
  explicitly treats them as two distinct table rows with two distinct
  descriptions.
- **Where the conflation actually happened.** `CLAIM_WGEO_20260827.md` §4.3
  (lines 342–344) correctly keeps "candidate-window membership" separate from
  HB ("A coarse binary lever... Not overturned here"). Its own §4.4 (lines
  356–361), two sections later, then calls **both** numbers "the window-removal
  counterfactual... quoted at two different magnitudes," citing
  `CLAIM_2D_BIAS_20260730.md:729-731` for `+0.010` (itself an off-by-a-few-lines
  mis-citation — the membership bullet is at :726–727, not :729–731) — this is
  the passage that manufactured the "factor ≈7 inconsistency," and it is the
  passage this reconciliation was commissioned to resolve
  (`CLAIM_WGEO_20260827.md:468, 492, 536` — "D-WGEO-1").

**Conclusion:** there is no arithmetic error in either 2026-07-30 source. The
"inconsistency" is a 2026-08-27 citation/labeling error in `CLAIM_WGEO_20260827.md`
§4.4, which cites the membership bullet's line range and value while describing
it in prose as if it were HB's own counterfactual.

---

## 3. The 0.063% vs 1.5% figures — different quantities, not a second inconsistency

Both are genuine, both are about HB's mass window, and both are correctly
attributed to HB — but they answer different questions with different
denominators:

- **0.063%** = `−0.317 nats / −504.8 nats`. This is HB's own **counterfactual**:
  what changes, over 0.73→0.81, if the window is **removed**. Denominator is the
  dark catalogue-leg survivors' `Σ ln(L_cat,2D/L_cat,1D)` tilt (`HANDOFF_20260730.md:55`,
  "534 ev... Δ = −504.8 nats" — the quantity HB was proposed as a candidate
  source for).
- **1.5%** = `+0.24 nats / +15.83 nats`. This is a **static partition** of the
  *observed* dark-class channel difference (2D − 1D posterior nats, same h
  range) between the 487 events the window zeroes out **at every h**
  ("hard zeros," +0.24 nats = 1.5%) and the 534 surviving, de-weighted events
  (+15.60 nats = 98.5%) — `CLAIM_2D_BIAS_20260730.md` C4 (:87 area, "flagship
  evidence... 487 events 2D-zeroed at every h... carries +0.24 nats = 1.5%...
  98.5% (+15.60) is carried by the 534 survivors"), reconfirmed at
  `ADJUDICATION_20260730.md:60-63` and `:439`. Denominator is the **total observed
  dark-class channel difference (+15.83 nats)**, a completely different
  accounting object from the −504.8-nat catalogue-leg-only figure above.

`ADJUDICATION_20260730.md:274–276` states this correctly as *corroboration*, not
as a competing point estimate: "HB (its hard-zeros are worth 1.5% of the
target — corroborates its self-refutation)." Nothing here is described anywhere
as a replacement for the 0.063% figure, and no document treats 1.5% as HB's
"real" bound. The ~24× gap between 0.063% and 1.5% is explained entirely by the
different numerators (−0.317 vs +0.24 nats) and different denominators (−504.8
vs +15.83 nats); it requires no further reconciliation.

One loose thread worth flagging for the author, not resolved here: both
"targets" (−504.8 and +15.83 nats) are informally called "the target" in
different places, which is the terminology habit that makes 0.063% and 1.5%
*look* like they should be directly comparable. They are not commensurable
without first fixing which accounting object "the target" means.

---

## 4. Underlying measurement — artifacts status

Neither the `+0.010` (membership) nor the `+0.0015` (HB) counterfactual has a
locally reproducible artifact in this repository.

- `HANDOFF_20260730.md:123–127` states the HB investigation ran inside an
  **external** Claude Code workflow: `~/.claude/projects/-home-jasper-Repositories-
  MasterThesisCode/9e5f1e9d-3971-4ed8-a067-4aa8532b0fa3/workflows/scripts/
  dark-completion-attack-wf_f5e50977-072.js`, with results in that run's
  `journal.jsonl`, explicitly noting "Workflow cache-resume is same-session
  only — a new session must relaunch." I checked: that path does not exist
  (`ls` and `find` both report "No such file or directory" on
  2026-08-27) — the project directory itself (`-home-jasper-Repositories-
  MasterThesisCode`) is gone, consistent with the repo having since been
  renamed/relocated to `darksiren-emri`. The four completed agents in that same
  run (HA, HC, membership, mass-kernel-family) share the same journal, so the
  membership number's backing artifact is equally unrecoverable.
- No script under `results/campaign51_20260728/realistic_20260729/` or its
  `gate_b_20260730/` subdirectory reproduces either number: `grep` for
  `mass_filter_mask`, `sigma_multiplier` (the decisive-test variables named at
  `HANDOFF_20260730.md:117-119`), or a `0.81→0.82`/`+0.0015`-style MAP-shift
  computation across `gate_b_20260730/*.py` (the C7–C9 kernel/measure scripts)
  and the root-level `attack_c1_c5*.py` / `attack_c3_c4*.py` files returns
  nothing relevant — those scripts compute the nats budgets (C1–C4), not the
  window-removal MAP shift.
- **Both figures are therefore text-only** as far as this repository is
  concerned: quotable from the record, but not independently re-verifiable
  without re-running the (lost) external workflow — which is out of this task's
  scope and would itself constitute reopening HB.
- One tangential caveat, not a defect in either number: `CLAIM_2D_BIAS_20260730.md`'s
  own "Errors made this session" §1 flags that its `1 nat/unit-h ≈ 4.5e-4 in h`
  conversion understated window-integrated nats-to-h shifts by ~12×. This does
  **not** touch the 0.063% figure (a nats/nats ratio, not a nats→h conversion)
  and does not touch the `+0.0015` figure (stated as a directly observed MAP
  shift, not derived through that conversion formula) — flagged only so the
  author does not need to re-check it.

---

## 5. Downstream claims that lean on HB's bound

| Document | Line(s) | How it depends on HB |
|---|---|---|
| `CLAIM_WGEO_20260827.md` | §4.1 (`:286-299`), §6 item 1 (`:415-418`), and its verdict §5 (`:371`) | HB is named as the single **decisive** collision that kills `[WGEO]`-H1 at stage 0 ("HB's banked rationale is [WGEO]-H1 almost verbatim"); §6 explicitly states "the only banked H₀ bound on this object is HB's" (quotes `−0.317 nats = 0.063%`, `~50×`, sign-inverted). |
| `CLAIM_WGEO_20260827.md` | R-WGEO-2 (`:532-534`), D-WGEO-1 (`:536`) | The pending `[RULE]`/`[DO]` items — not yet author-ratified — bind any future window-geometry reopening to "new evidence that engages HB's −0.317-nat/0.063%/sign-inverted measurement directly," and explicitly task **this** reconciliation ("D-WGEO-1... resolve +0.010 vs +0.0015") as the prerequisite for HB becoming "quotable as a point value." |
| `CLAIM_WGEO_20260827.md` | §4.4 (`:356-361`) | Contains the conflation this report resolves — currently reads as an open, unresolved "×7 discrepancy," which is now shown to be a citation error, not a physics discrepancy. This section needs a correction (author's call whether to edit `CLAIM_WGEO_20260827.md` — out of scope for this read, since editing existing claim cards was excluded from this task). |
| `wgeo_s0_coupling_20260827.md` | read (3), `:1-40+` | The original stage-0 rule-1 check that quoted the membership/mass-kernel-family bullets (`:726-727`) and stopped there — the exact miss described in this task's "WHY THIS EXISTS." Its "PASSED" conclusion (superseded by `CLAIM_WGEO_20260827.md` §4.1, which does reach HB) is the artifact that motivated this whole reconciliation. |
| `gate_b_20260730/BIAS_HISTORY_LEDGER.md` | `:135`, `:206` | Line 135 carries HB by name in the binding "do not re-try" list. Line 206 ("Gate C alternatives," item 4) states "the h-*tilted* component is the refuted HB," i.e. treats HB's own counterfactual sign/magnitude as settling that sub-question; it separately flags an **unmeasured** "h-flat 68% window suppression" component as still open — not covered by HB's bound at all. |
| `gate_b_20260730/ADJUDICATION_20260730.md` | `:60-63`, `:274-276`, `:439` | Source of the 1.5%/98.5% figure that "strengthens" (does not replace) HB; §3 above traces this. |
| `CLAIM_D1_P0WINDOW_20260805.md` | `:79-80` | Uses HB (and membership) as a disambiguation anchor to argue D1 is a different object ("different file, different stage, different variable"); does not lean on HB's exact magnitude, only on HB being a settled, named exoneration. |
| `CLAIM_2D_BIAS_20260730.md` | `:721-744` (its own binding exoneration list) | The master claim card's own account of the 2D bias mechanism rests on this exoneration list being solid; a HB inconsistency, had it been real, would have reopened the list's own closing note ("This list is not the whole exoneration set... the ledger's §2... is therefore the live re-litigation risk," `:754-756`). |

If the author does not ratify a fix to `CLAIM_WGEO_20260827.md` §4.4, the
practical risk is narrow: nothing downstream currently *uses* the wrong
magnitude for a decision (R-WGEO-2/D-WGEO-1 are both still `PENDING`), but a
future reader skimming §4.4 in isolation could conclude HB's bound is
unquotable, which is not correct per §§1-2 above.

---

## 6. Bottom line

**Reconciled: yes, as two different quantities — not an error in either
2026-07-30 source.**

- **HB's own defensible bound**, quotable as a point value:
  **`−0.317 nats over h = 0.73→0.81, ≈ +0.0015 in raw MAP-h, sign-inverted
  (wrong direction to explain the +0.077 2D bias), ~50× too small at its
  ceiling`** — `HANDOFF_20260730.md:85-88`, corroborated (not re-measured) by
  the 1.5%/98.5% hard-zero partition at `ADJUDICATION_20260730.md:274-276`.
- **`+0.010`** is a real, separately-exonerated number, but it belongs to
  **"candidate-window membership"**, a different bulleted exoneration
  (`CLAIM_2D_BIAS_20260730.md:726-727`, `HANDOFF_20260730.md:63-64`) — not to
  HB. It should not be cited as, or averaged against, HB's bound.
- The one place in the record that *does* treat them as the same
  ("the window-removal counterfactual... quoted at two different magnitudes")
  is `CLAIM_WGEO_20260827.md` §4.4, which is a citation/labeling error, not a
  measurement discrepancy — its own §4.3, two paragraphs earlier, keeps the two
  items correctly separate.
- Caveat: neither number's raw computation is locally reproducible — both trace
  to an external workflow run whose session directory no longer exists on this
  machine (§4). The reconciliation above is therefore a **textual** resolution
  (which label the record itself attaches to which number), not an independent
  numerical re-verification, per the task's explicit prohibition on re-deriving
  the bound.
