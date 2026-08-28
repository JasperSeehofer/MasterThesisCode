# PRE-REGISTRATION — [P3-MKER] window GEOMETRY: what truncation does the linear-symmetric mass-eligibility window actually realize? (stage 2, correctness-class)

`[FABLE-ORCH fork 2026-08-28]` · authorized by author grant **D-MKER-2** (blanket ratification
2026-08-28, ledger row #214: "all ratified also the thirteen earlier ones") — **PRE-REGISTRATION
ONLY; nothing here launches** (§9).

**Class: CORRECTNESS measurement, not a bias hunt.** The [P3-MKER] thread (ledger row #206) is
correctness-class by its opening delimitation; exoneration **HB** governs every window-as-BIAS
claim and its reconciled bound (ΔMAP ≈ +0.0015, wrong-signed, 40–50× too small at ceiling —
`CLAIM_WGEO_20260827.md`, "D-WGEO-1 RESULT", ratified R-WGEO-2) is engaged in §1.3 and §7. This
document registers **zero H₀-space reads** (§7 item 1).

**Status: REGISTERED · NOT LAUNCHED.** Placeholders marked `⟨SUBMIT⟩` are filled at launch per
the PA-2DR-13 pattern, including the authorization stamp (runbook 36 §2 standing rule).

---

## 1. Question, motivation, and delimitation

### 1.1 The question

The production mass-eligibility filter (`handler.py`, `mass_filter_mask` — the two-condition
overlap test between the GW interval `[(M_z − kσ_Mz)/(1+z_max), (M_z + kσ_Mz)/(1+z_min)]` and
the candidate interval `[M − k·σ_M, M + k·σ_M]`, `k = 1.5` via the single call site
`bayesian_statistics.py:4691`, `"symmetric"` mode per rows #198–#202) applies a **linear-symmetric
±kσ interval** to a mass error that the code's own error model makes **log-normal**: R&V15 is a
log₁₀-linear relation with dex scatter, so `BH_MASS_ERROR/BH_MASS = CV` is an ln-space width
(`CLAIM_P3_MKER_20260826.md` §R2.7(ii)).

**Registered question: what effective truncation ε of the log-normal mass law does each geometry
(linear-symmetric vs log-symmetric, both at k = 1.5, budget unchanged) actually realize — in
closed form, on the catalogue's CV census, and on the frozen fleet's candidate sets?** This is
the concrete target the [P3-MKER] card's part (b) needs: F-ii wants the window to be an
ε-derived truncation bound on the corrected kernel, and no document yet states what ε the
current window corresponds to.

### 1.2 What is already banked (this design extends, it does not re-measure)

Verified at source during authoring:

- CV census over the N = 20 834 171 pruned catalogue: min 0.5930 · p10 0.7846 · median 0.8614 ·
  p90 1.2137; **negative-lower-edge fraction 0.996112** (threshold CV ≥ 1/k = 2/3)
  (`CLAIM_WGEO_20260827.md` §3.3, ✓CHAIR ✓VER).
- Cone-exact fleet census (4 800 event rows): n_lin/n_all = 0.9490, n_log/n_all = 0.4210,
  **n_log/n_lin = 0.4437**; linear failures 29 : 1 too-light : too-heavy (§3.9, ✓VER; chair
  variant on one GW interval: 0.3025, 12.93 : 1).
- Exhibit chain (seed 900121 event 20): GW floor **1 237 046.5023702232**; `6791151`
  (BH_MASS 223 872.11385683485, σ_ln 1.3032395587986776) excluded in linear, **t_ln = 1.3117 σ**
  in ln-space, log-window upper edge **1 581 192.05 ≥ GW floor → readmitted**; true host
  `6791134` outside the sky cone regardless (chord ×1.1196) (`CLAIM_P3_MKER_20260826.md`
  §R2.3–R2.7).
- Edge factors at the exhibit: linear `1 + 1.5σ_ln` = **2.954859338198016** vs log
  `e^{1.5σ_ln}` = **7.062925470001435** (both re-derived to full precision during authoring).

### 1.3 Delimitation against HB (rule-1 engagement, per ratified R-WGEO-2)

HB measured **window removal** in H₀ space and bound it at ΔMAP ≈ +0.0015, wrong-signed. This
design measures **what truncation the window realizes** — a model-semantics question, not an
H₀ question — and registers no H₀ read. Consequences:

- No verdict of this measurement may be phrased as, or converted into, a bias claim. Any future
  H₀-space follow-up (e.g. a geometry-switch counterfactual through `--evaluate`) requires its
  own registration and must engage HB's +0.0015 bound and the mass-kernel-family +0.002 bound
  (row #206 delimitation) **before banking**.
- The banked −14.5 % eligible-set mean-redshift moment is an eligibility-set statement, and the
  exoneration-list guard ("confusing an eligibility-set moment with a posterior effect")
  applies verbatim: P5 re-derives the number; nothing interprets it as a posterior effect.

### 1.4 What this measurement decides

It produces the **ε-semantics table** (P2) any F-ii window redesign must either match or
explicitly reject, and the **discordance census** (P3) quantifying exactly which candidate-set
changes a geometry switch would cause on the frozen fleet. The physics ruling — which geometry
is *correct* for the production filter — is the author's, taken on this evidence (§9).

---

## 2. Instruments and inputs (all frozen, all local)

| input | pin |
|---|---|
| pruned reduced catalogue | md5 `c52c13b5cab61f6b3f04bbe202550969` (`REDUCED_CATALOGUE_MD5`, `correspondence_1d.py`) — STOP on mismatch (dataset-pinning rule) |
| fleet candidate/CRB artifacts | the banked `p3_2d_fleet_20260825/` per-seed CSVs used by the [WGEO] census (the SAME 4 800 event rows; G1 reproduces the banked census before any new read) |
| exhibit chain | `CLAIM_P3_MKER_20260826.md` §R2.2–R2.6 banked values (verbatim comparands) |
| window code | `handler.py` `mass_filter_mask` (anchor text, not line numbers), `"symmetric"` mode, k = 1.5 — **rows #198–#202's adoption is NOT disturbed; no source file is edited by this measurement** |
| instrument script | `⟨SUBMIT⟩` new standalone census script under `realistic_20260729/` (reads catalogue + fleet CSVs; imports nothing from the production estimator beyond the R&V15 constants), SHA frozen at launch |

Convention statement (required by constraint): the measured window is the **adopted
`"symmetric"` variant** (`_bh_mass_error_multiplier = k = 1.5` on the candidate side). The
retired `"asymmetric"` variant is out of scope.

---

## 3. Registered reads and their predictions (BEFORE any run)

All reads are **deterministic on frozen inputs** — there is no seed scatter anywhere in this
design, hence **no σ bands and no UNDERPOWERED branch**; every read carries an exactness
tolerance instead (§4). Central predictions below are derived in closed form during authoring
(python, `Φ` = standard normal CDF; light-side cut `Φ(ln(1−kCV)/CV)` for kCV < 1, else 0;
heavy-side cut `1 − Φ(ln(1+kCV)/CV)`).

**P1 · Closed-form geometry function.** Reproduce `A(x) = ln(1−x²)/ln[(1+x)/(1−x)]` spot values
(−0.050084 at x = 0.1 … −0.635421 at x = 0.95) and the negative-edge threshold CV = 2/3
(`CLAIM_WGEO_20260827.md` §3.2). Pure arithmetic re-derivation; anchors the sign conventions.

**P2 · The ε-semantics table (the deliverable).** Effective truncated log-normal mass per
geometry at k = 1.5:

| CV (banked census quantile) | ε_lin light | ε_lin heavy | **ε_lin total** | ε_log total |
|---|---|---|---|---|
| min 0.5930 | 0.000102 | 0.141627 | **0.141729** | 0.133614 |
| p10 0.7846 | 0 (edge ≤ 0) | 0.160730 | **0.160730** | 0.133614 |
| median 0.8614 | 0 | 0.167791 | **0.167791** | 0.133614 |
| p75 0.9401 | 0 | 0.174704 | **0.174704** | 0.133614 |
| p90 1.2137 | 0 | 0.196454 | **0.196454** | 0.133614 |
| exhibit 1.3032 | 0 | 0.202887 | **0.202887** | 0.133614 |

ε_log = 2Φ(−1.5) = **0.133614**, CV-independent and two-sided by construction. Registered
qualitative content (the correctness finding this table constitutes if it verifies): the linear
window's realized ε is **one-sided (entirely heavy-side for 99.61 % of the catalogue),
CV-dependent (0.142 → 0.203 across the census), and nowhere equal to a symmetric truncation**;
the small-CV regime where linear ≈ log is never reached (banked min CV = 0.5930). The full
registered read is this table recomputed **exactly** by the frozen instrument, plus the
catalogue-weighted mean ε_lin over the true CV distribution (predicted ≈ 0.17, registered as
REPORTED-ONLY since it depends on the full census, not the quantile interpolation).

**P3 · Discordance census (fleet, cone-exact, same rows as [WGEO]).** Candidate-row eligibility
under linear vs log at fixed k and budget:

- **P3a** reproduce the banked totals first (this doubles as gate G1): n_lin/n_all = 0.9490,
  n_log/n_all = 0.4210, n_log/n_lin = 0.4437.
- **P3b** `lin∩¬log` fraction — registered bound: **≥ 0.5280** of all rows (= 0.9490 − 0.4210,
  equality iff log ⊂ lin on these rows), predicted heavy-side-dominated (the log window
  re-introduces the heavy cut; banked 29 : 1 and 46.38 %-above-1e7 census).
- **P3c** `log∩¬lin` fraction — registered prediction: **non-empty**, light-side readmissions
  of the `6791151` class (candidates whose linear upper edge `M(1+kCV)` falls below the GW floor
  that `M·e^{kCV}` clears). Its size is NOT predicted (no banked anchor); REPORTED with its
  composition (CV distribution, redshift moments, share of no-BH likelihood weight the readmitted
  rows carry — the interloper-weight question R2.7 raised).

**P4 · Exhibit regression.** The frozen chain re-run under the log geometry must give exactly:
`6791151` and `6791153` READMITTED (upper edge 1 581 192.0549825 ≥ 1 237 046.5023702232),
`6791138`/`6791158` unchanged PASS, true host `6791134` unchanged OUTSIDE the cone (the sky
cone is untouched by the mass geometry). Under the linear geometry the banked pass-set
`{6791138, 6791158}` must reproduce member-for-member (= gate G3).

**P5 · Chair re-derivation of the −14.5 % moment** (✓VER-only, flagged in
`CLAIM_WGEO_20260827.md` §6.2 as requiring chair re-derivation before quantitative use).
Registered read: recompute the eligible-set mean-redshift shift census (median, mean, p5,
max |·|, z_true trend) on the frozen fleet. Band: sign match and |median_rel − (−0.145)| ≤ 0.01
absolute in the relative shift. Whatever the outcome, its INTERPRETATION stays barred from
posterior-effect language (§1.3).

---

## 4. Tolerances, and the freeze rule's role

Deterministic design → exactness tolerances, registered here and not adjustable post-data:

| read | tolerance |
|---|---|
| P1, P2 closed forms | ≤ 1e-9 relative vs the instrument's recomputation (pure float arithmetic) |
| P3a (= G1) | exact match to 4 decimal places of the banked ratios; row counts exact |
| P3b bound | arithmetic identity (≥ 0.5280 up to the P3a rounding); composition directional |
| P4 | exact set membership, both geometries |
| P5 | sign + |Δmedian_rel| ≤ 0.01 |

The house freeze rule applies degenerately: nothing may be loosened post-data; a read that
misses its tolerance is not re-toleranced, it fails (§5). There is no stochastic component to
grow a band on, and no UNDERPOWERED disposition exists in this design (stated per the
runbook 36 §2 partition checklist).

## 5. Verdict map — conditions PARTITION (checked)

Let **G** = "all gates §6 pass" and evaluate reads in the fixed order P1, P2, P3, P4, P5
against §4. Exactly one verdict fires:

1. **INSTRUMENT-DEFECT** — ¬G. (Evaluated first; no other verdict may fire.)
2. **CONFIRMED (geometry-mismatch measured)** — G ∧ all five reads within tolerance. Content:
   the ε-semantics table and discordance census are banked as the measured truncation semantics
   of the production window; the linear window realizes a one-sided, CV-dependent ε, never a
   symmetric truncation. Epistemic cap: `supported` (§8 item 1 — no external anchor for the
   error-model premise itself).
3. **REFUTED-IN-PART** — G ∧ at least one read outside tolerance. The failing read(s) are named;
   passing reads still bank individually (they are independent deterministic facts); the
   composite ε-semantics claim does NOT bank. (Disjoint from 2 by "all vs at least one"; from 1
   by G.)

No other outcome is reachable: {¬G} ∪ {G ∧ all} ∪ {G ∧ ¬all} is exhaustive and pairwise
disjoint. There is deliberately **no MIXED branch and no verdict about which geometry is
"correct"** — that ruling is the author's (§9), taken on the banked table.

## 6. Gates

- **G1 · banked-census reproduction** (= P3a): the instrument must reproduce the [WGEO]
  cone-exact totals on the frozen fleet inputs before any new number is read. FAIL ⇒
  INSTRUMENT-DEFECT.
- **G2 · catalogue pin**: md5 of the pruned catalogue equals `c52c13b5…` at run start (STOP on
  mismatch, dataset-pinning rule); fleet CSV row counts equal the banked 4 800.
- **G3 · exhibit reproduction** (= P4 linear leg): pass-set `{6791138, 6791158}` and
  exclusion-set `{6791151, 6791153}` member-for-member, GW floor to full float precision.
  FAIL ⇒ INSTRUMENT-DEFECT.
- **G4 · closed-form anchor**: the instrument's ε formulas reproduce the §3 P2 table
  (authoring-time derivation) to 1e-9 — guards against a sign/convention slip between the
  authoring arithmetic and the instrument.

## 7. Falsifiers (one per verdict, registered now)

1. **For CONFIRMED**: if the instrument finds the linear window's realized ε symmetric, or
   CV-independent, or two-sided for the bulk of the catalogue, the mismatch claim is false —
   REFUTED-IN-PART fires on P2 and the F-ii framing loses its target.
2. **For REFUTED-IN-PART**: any tolerance miss traceable to a genuine property of the frozen
   inputs (not an instrument bug — that is G's job) falsifies the corresponding banked anchor
   and re-opens the [WGEO] census with the discrepancy as evidence.
3. **For INSTRUMENT-DEFECT**: G1/G3 failing on inputs whose pins pass would mean the banked
   census/exhibit themselves are not reproducible — escalate to the author before anything
   else; that outcome impeaches banked records, not this design.

## 8. Structural blindness (what this design cannot see)

1. **It cannot validate the log-normal error model itself.** Both geometries are evaluated
   *under* the R&V15 log-normal premise; if the true mass-error law is neither, both ε columns
   are model-conditional. No external anchor exists (missing-anchor cap: verdict ≤ `supported`).
2. **No H₀-space consequence is visible** (registered out, §1.3). The discordance census bounds
   candidate-set changes only.
3. **Catalogue-realization specificity**: the CV census is GLADE-quantization-driven below
   z ≈ 0.2 (banked r-quantization finding); the ε table transfers to another catalogue only
   through its own CV census.
4. **k = 1.5 fixed.** No k-sweep is registered; the ε table is a function of the production
   k only.
5. **The true-host/sky-cone finding is out of scope** (R-MKER-6, pending author one-word).
6. **The interloper-weight read (P3c composition) has no banked comparand** — it is REPORTED,
   first-of-its-kind, and must not be band-graded post-hoc.

## 9. Costing (A6/A17) and decisions required before any run

**Cost: local CPU only, zero cluster.** The [WGEO] census of the same scope ran locally; the
new instrument re-runs it plus closed forms — estimate **≤ 3 CPU-h local**, fan-out 1 process
(pandas over 20.8M rows × the 4 800-row fleet join). Cluster-first does not apply (row #185
threshold not reached).

| # | tag | decision |
|---|---|---|
| 1 | **[DO]** | Launch this measurement as registered (fills `⟨SUBMIT⟩`: instrument SHA, date, **authorization stamp** "launched under author grant of ⟨date⟩" per runbook 36 §2). |
| 2 | **[RULE]** | On CONFIRMED: rule on the F-ii consequence — whether the production window is redesigned as an ε-derived truncation (which ε, which space), or the linear geometry is kept and documented as a deliberate one-sided design choice. **Not covered by any approval given here**; returns with the banked table. |
| 3 | **[RULE]** | Whether P3c's interloper-weight composition, once seen, opens a follow-up thread (it is the readmission-of-`6791151`-class question) — fresh decision post-data by the approval-scope rule. |

*Registered 2026-08-28, pre-launch. Every number cited above was re-verified at its source file
or re-derived in closed form during authoring; the P2 table is the authoring-time derivation the
instrument must independently reproduce (G4).*

---

## ⟨SUBMIT⟩ + RESULT RECORD [2026-08-28; FABLE-ORCH]

**Launched under author grant of 2026-08-28** (ledger row #216 item 3, "all approved" against
the Runbook 37 Docket — authorization stamp per runbook 36 §2). Instrument:
`wgeom_instrument.py` (sonnet-built against this registration; measurement RUN BY THE
ORCHESTRATOR, not the building agent, per the verifier-independence rule). Local CPU,
single run, `--mode full`; outputs `wgeom_work/wgeom_result.{json,md}`.

**VERDICT (per §6, evaluated first): INSTRUMENT-DEFECT — ¬G via G1 (P3a) and P5.**
G2 (catalogue pin md5 PASS), G3 (P4 exhibit: ALL SEVEN comparands exact — GW floor/ceiling,
cone set, linear pass/fail sets, log-readmission, true-host-outside-cone), G4 (P2 authoring
table reproduction) all PASS. P3b's bound passes (0.5808 ≥ 0.5280). No non-gate read is
banked under this verdict; the P2 ε-table and P3c stand as instrument output only.

**CHAIR FORENSIC (appended before escalation, zero-compute):** the instrument's census totals
are **bit-identical to the banked §3.8 fleet read** of `CLAIM_WGEO_20260827.md`
(2 154 066 passed + 95 165 excluded = n_all 2 249 231; ratio 0.9577) — the instrument did
reproduce A banked census exactly. The G1 comparands registered here (0.9490/0.4210/0.4437)
came from §3.9's "cone-exact, whole fleet, 4 800 event rows" row, whose own failure counts
(112 416 623 too-light + 3 868 708 too-heavy = 116 285 331) are arithmetically incompatible
with a ~2.25M-row cone-exact basis (they imply n_all ≈ 2.3e9): **the §3.9 ✓VER row is
internally inconsistent with its stated scope.** P5's banked −0.145 anchor is suspect for the
same scope reason (measured −0.0986, sign matches).

**Per §7 clause 3 (registered, binding): G1 failing on inputs whose pins pass impeaches the
banked record, not this design — ESCALATED TO THE AUTHOR.** Items for ruling:
- **[RULE] W-1**: accept the chair forensic reading — §3.9's census row (and P5's −0.145) are
  scope-mislabelled/internally inconsistent banked numbers; the §3.8 fleet census (0.9577) is
  the reproducible comparand. On acceptance: correct CLAIM_WGEO §3.9 by appended note (never
  edit), re-anchor G1/P5 to the §3.8-scope values, and re-run this instrument's verdict map
  against the corrected anchors (zero marginal compute — the JSON already holds every read).
- **[RULE] W-2**: whether §3.9's directional claim (the 29:1 too-light:too-heavy split and
  the log-window heavy-end-cut mechanism) survives — its SIGN is corroborated by the chair's
  independent single-interval read and by HB's 2026-07-30 census, but its magnitudes carry
  the same scope inconsistency.

No further run, no band evaluation, no F-ii consequence (decision 2 does not fire under
INSTRUMENT-DEFECT). Instrument sha1 `17dbccbac7eb` (the JSON records git_commit + timestamp;
the script file is committed alongside this record).
