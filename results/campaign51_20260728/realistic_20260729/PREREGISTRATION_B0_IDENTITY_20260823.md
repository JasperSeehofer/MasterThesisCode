# PRE-REGISTRATION — [P3-IMP] the b0 catalogued-host identity test (stage 2; the correctness adjudicator)

**Date:** 2026-08-23 · **Thread:** `[P3-IMP]` (row #169 grant: "run fused re-measure + b0 test";
row #173: the b0 test is "the sole gate between the candidate and a production proposal").
**Orchestrator-autonomous session (author directive 2026-08-23): every delegated ruling in this
file is tagged [ORCH-RULE]/[ORCH-DESIGN] for the author's trace and correction.**
**Append-only after commit; A21 governs** (premise corrections STOP execution, amend first).
Vocabulary per `docs/PRIMER_BIAS_CHANNELS_20260822.md` (binding).

## 0. Governing rulings inherited (with provenance)

- Appendix B ratified (row #169): the twin is the candidate of record; the identity test runs
  THREE arms (coded / twin / R-rescaled) on the FUSED basis.
- TWIN-FUSED-MATERIAL ratified as amended **[ORCH-RULE 4, this session]** (row #173 evidence;
  amendment-20 quotation rules binding).
- Σ^φ divisor proposal §7 **[ORCH-RULE 1–3, this session]**: measure-first adopted; the
  verification plan (i)–(iii) approved; **the b0 arms run on the corrected Σ^φ slot via the
  counterfactual flag `catalogue_global_selection="phi"`** (production default `"s3d"`
  untouched — no production adoption granted).

## 1. The question and the registered identity (derivation-grounded)

`PROPOSAL_P3_MIXTURE_WEIGHT_20260822.md` §1 (ratified chain): *"under the generator, the
posterior's expected catalogue-class responsibility must equal the generator's realized
catalogue-hosted fraction."* **[ORCH-DESIGN, A21-class correction at registration]:** in the b0
venue (`ARM_HOST_MODE["b0"]="catalogue"` — every true host is a catalogue row; realized
fraction ≡ 1.0 by construction, i.e. the venue CONDITIONS on the catalogue class rather than
drawing it at model frequency) the ensemble-mean form is unsatisfiable as written (w < 1 a.s.
for any finite-completeness model). The exact identity for class-conditional draws is the
**odds form**: for data d ~ p(d | G, h_true) under a correctly arranged class mixture
p(d|h) = P_G·p(d|G) + P_Ḡ·p(d|Ḡ),

    E_{d~p(d|G)}[ (1 − w(d)) / w(d) ] · P_G / P_Ḡ  =  1,      w(d) = P_G·p(d|G) / p(d)

(proof: (1−w)/w = P_Ḡ p(d|Ḡ)/(P_G p(d|G)); the p(d|G) measure cancels the denominator and
∫p(d|Ḡ) = 1). **The identity holds for the correctly arranged mixture and is violated when the
catalogue leg's numerator/normalizer pairing is broken — the discrimination the all-impostor
B-SEL venue structurally cannot provide.**

**Mapping to code objects (no-BH channel, `absolute_marginal` + Path A):** per event and node,
`combined_no_bh = (β_G_φ·L_cat_no_bh + B_num)/D̃_φ`; the class-G posterior responsibility is
`w_e(h) = β_G_φ·L_cat_no_bh / (β_G_φ·L_cat_no_bh + B_num)`, reconstructed from banked columns
as `w_e = 1 − B_num/(combined_no_bh·D̃_φ)`. The prior odds `P_G/P_Ḡ` = the arrangement's own
selected class masses at h: `M_G/M_Ḡ` with `M_G = α_G_φ`, `M_Ḡ = D̃_φ − α_G_φ = β̄_Ḡ_φ`
(columns `alpha_G_phi`, `D_tilde_phi`). **The exact identification of (w, M_G, M_Ḡ) per arm is
a registered verification item for the pre-execution adversarial review — if the reviewer
derives a different odds constant for any arm, this section is amended (A21) before any arm
runs.** All reads at the truth node h = H_TRUE = 0.73 (an H-grid node).

## 2. Arms (all on the fused completion basis; Σ^φ slot per [ORCH-RULE 3])

| arm | flags (beyond PRODUCTION_FLAGS + fused) | runs |
|---|---|---|
| **B-C (coded)** | `catalogue_numerator_survival="off"`, `catalogue_global_selection="phi"` | 12 fresh `evaluate()` |
| **B-T (twin)** | `catalogue_numerator_survival="phi"`, `catalogue_global_selection="phi"` | 12 fresh `evaluate()` |
| **B-R (R-rescaled)** | zero-`evaluate()` rescore of B-T's diagnostics by `R(h) = β_G(h)/β_G_φ(h)` (the committed `p3_completed_rescore.py` construction; the refuted-Appendix-A arrangement, run as the registered CONTROL — expected to FAIL the identity) | 0 |

Seeds: **900101–900112** (first 12 of `ARM_SEEDS["b0"]`, registry order). Venue: b0 exactly as
registered in `correspondence_1d.py` (`ARM_SPECS["b0"]=(1.0,1.0)`, catalogue host mode, real
completeness, `completion_event_measure="ratio"`), EXCEPT `selection_in_completion_numerator=
"fused"` (the ratified basis) and the Σ^φ slot flag. H grid: `H_GRID_FULL` (un-truncated;
amendment-20 lesson — the registered-grid censoring is not re-imported here).
Driver: `p3_b0_identity_test.py` (committed before any arm runs), the arm-parametrized
generalization of the committed `p3_twin_test.py`.

The 25 banked b0 seeds (off basis, commit `198724e2`) are the ZERO-COMPUTE coded-arm
**LEV instrument** (§5) and the replica-gate anchor — never the fused-basis comparand.

## 3. Gates (scored before any statistic is read; failure ⇒ VOID unless amended per A21)

- **GATE R-B0 (venue replica):** the driver re-runs seed 900101 under the BANKED configuration
  (`off` basis, `s3d` slot, survival `off`) and reproduces the banked
  `b0_seed900101/.../event_likelihoods.csv` `L_cat_no_bh`, `B_num`, `combined_no_bh` columns to
  ≤1e-12 relative (multiprocessing fallback), wall > 60 s. Proves venue fidelity across the 11
  intervening commits; a mismatch is an A21 STOP (diagnose, never proceed).
- **GATE E-B0 (A13 engagement + dispatch, denominators explicit per row-#173/A17):**
  (a) Σ^φ slot: `L_cat_no_bh` level ratio B-C/replica-config ≠ 1 on 100% of live-catalogue-leg
  rows (denominator: rows with `L_cat_no_bh > 0` in both), consistent with a single h-dependent
  factor 1/r_φ(h); (b) twin cell: B-T vs B-C `L_cat_no_bh` differ on ≥99% of live rows;
  (c) runtime assertions confirm both flags reach BOTH scalar and batch dispatch paths.
- **GATE L-B0:** counterfactual log lines (`fused` line in all arms; Σ^φ line in all arms; twin
  line in B-T only); flags recorded in each run meta.
- **GATE W-B0 (identity computability):** reconstructed `w_e(0.73) ∈ (0,1]` on all live rows,
  and the reconstruction closes: `|combined·D̃_φ − (β_G_φ·L_cat + B_num)|/… ≤ 1e-9` rel
  (using the run's own β_G_φ; a violation means the responsibility read is not the mixture's).
- **GATE N-B0 (A18):** every readout prints the statistic, the reference it subtracts, and the
  band constants, machine-readable.
- **A22 (as amended row #173):** git commit + dirty flag stamped and WRITTEN before each
  `evaluate()`; completion cell + both flag values in every meta; no HEAD moves during
  registered runs.

## 4. Statistics and bands

**Primary (per arm a):** per-seed identity score `I_s(a) = mean_e[(1−w_e)/w_e] · M_G/M_Ḡ − 1`
at h = 0.73 (live rows; denominator count banked per seed), fleet mean `Ī(a)` over 12 seeds
with SEM across seeds. Robustness twin (registered, verdict-participating per §6 falsifier i):
the 5%-trimmed version `Ī_trim(a)`.

**Band FORMULAS (frozen now; NUMBERS frozen from the LEV read + pilot realized scatter,
appended below pre-fleet — the twin-prereg precedent):**

- **IDENTITY-PASS(a):** `|Ī(a)| ≤ max(3·SEM(a), ε_I)` AND `Ī_trim` agrees in band.
- **IDENTITY-FAIL(a):** `|Ī(a)| > max(3·SEM(a), ε_I)` with sign reported, AND `Ī_trim` agrees.
- **UNDETERMINED(a):** `Ī` and `Ī_trim` disagree on the band (heavy-tail dominance) — no
  verdict for that arm; reported as such.

`ε_I` (the identity-resolution floor) is frozen pre-fleet from the pilot's realized per-seed
scatter and the LEV read's coded-arm displacement scale; it may only TIGHTEN after data exist.

**Verdict mapping (registered now, two-sided, no commentary):**
- B-T PASS ∧ B-C FAIL ⇒ **TWIN-IDENTITY-CONFIRMED** — the production catalogue-leg proposal
  proceeds to its own 6-item physics-change gate (author-gated; nothing adopts this session).
- B-C PASS ∧ B-T FAIL ⇒ **TWIN-IDENTITY-REFUTED** — the candidate falls; TWIN-FUSED-MATERIAL
  stands as leverage-only; the thread returns to stage 0 with the refutation banked.
- Both PASS ⇒ **UNDISCRIMINATING** (venue insufficiently sensitive; report `Ī ± SEM` both arms,
  no correctness claim either way).
- Both FAIL ⇒ **VENUE-MISSPEC** — a defect COMMON to both arrangements dominates (see §7
  blindness); no arm-level correctness claim; the common-mode becomes the next stage-0 claim.
- B-R participates only as the control: B-R PASS would falsify the identity's discriminating
  power (the refuted arrangement scoring calibrated ⇒ UNDISCRIMINATING regardless of B-C/B-T).

**Secondary (registered, reported with the verdict):**
1. Per-event distribution of `(1−w)/w · M_G/M_Ḡ` per arm ([A2] paired read — cancellation
   cannot hide); banked as percentiles + the full per-seed vectors.
2. Identity profile across the H grid (shape diagnostic: the correct arm's profile should cross
   its zero nearest h_true; reported-only).
3. Paired `Δmean_h(B-T − B-C)` on the 12 b0 seeds — the leverage read connecting to
   TWIN-FUSED-MATERIAL (this venue has real hosts; expected same sign, magnitude unregistered).
4. Rail read: floor-node mass B-C vs B-T (expected materially lower than B-SEL's 27–31%;
   surprise ⇒ registered as a finding).
5. Score-at-truth (A12) `∂_h ln L` at 0.73, per arm, class-resolved.
6. The banked-b0 off-basis coded LEV read (§5) quoted alongside, labeled cross-basis.

## 5. LEV — the zero-compute leverage instrument (A1; runs BEFORE the pilot, after this commit)

On the 25 banked b0 CSVs (off basis, `s3d` slot, coded cell): compute `Ī_banked(coded)` per §4
with the banked columns' own (α_G_φ, D̃_φ) and the run's β_G_φ recovered via the W-B0 closure.
Purpose: (i) order-of-magnitude of the coded arm's identity displacement (the LEV threshold:
predicted |Ī| must be ≥5× the band resolution, else STOP and re-design per the O4 lesson);
(ii) the heavy-tail read that calibrates the trim convention; (iii) the cross-basis comparator
for secondary 6. NOT a fused-basis statistic; never the comparand.

## 6. Falsifiers (A19)

Any arm verdict is falsified by: (i) trim/untrim band disagreement (auto-UNDETERMINED, built
into §4); (ii) GATE E-B0 failure on re-audit (engagement vacuity); (iii) a zero-compute audit
showing the reconstructed `w` is not the mixture's own responsibility (W-B0 closure violation);
(iv) GATE R-B0 failure (venue infidelity). TWIN-IDENTITY-CONFIRMED is additionally falsified by
B-R passing (control failure ⇒ no discriminating power). The banked verdict carries provisional
status until the author's stage-5 ruling — **[ORCH-RULE] verdicts here are orchestrator-banked
under the autonomous-session directive and flagged for the author's review.**

## 7. A10 — invariants and structural blindness

**Invariants (held fixed in every arm):** the b0 venue registries (`ARM_SPECS/ARM_HOST_MODE/
ARM_UNITY_COMPLETENESS/ARM_EVENT_MEASURE`, audited this session via recon) · the pinned
catalogue + injection pool (checksum-pinned at the consumer; 2026-08-20 rule) · H_GRID_FULL ·
with-BH catalogue numerator = coded · D̃_φ selected (Appendix A's surviving half — NEVER
re-opened here) · `PRODUCTION_FLAGS` otherwise verbatim · the S̄_φ table construction.

**Structural blindness:** (i) an error inside `precompute_phi_marginal_survival` is common-mode
to ALL arms and both flags (the disclosed three-instrument blind spot, now four); (ii) the
prior-odds constant M_G/M_Ḡ is derived from the same mixture bookkeeping under test — a defect
that mis-states BOTH the responsibility and the odds by compensating amounts is invisible
(partially mitigated by the B-R control and the h-profile secondary); (iii) the b0 venue is the
synthetic mirror, not production — venue-conditional throughout; (iv) catalogue-mode conditioning
means the completion leg's own numerator correctness is exercised only through the odds
denominator, not through dark-host draws.

## 8. Costing line (A6/A17) — [ORCH-COST]

LEV: zero-compute, <15 min. R-B0 replica + 2-arm pilot (seed 900101): 3 × `evaluate()` ≈
2–3 CPU-h. Fleet: 2 arms × 12 seeds = 24 × `evaluate()` ≈ 12–20 CPU-h (banked b0 anchor:
0.478–0.9 CPU-h/seed) — local 2-wide detached ≈ 6–10 h wall, ~18 GB peak (630 GB disk free,
verified). B-R rescore < 5 min. Total ≤ ~23 CPU-h, disclosed against the row-#169 grant
pattern (the granted b0 costing line was "to be lined before launch" — this is that line).
Cluster fallback (preflight-gated) only if the local box is contended.

*(Committed before the driver and flag exist; pre-execution adversarial review (A20-style,
design-stage) runs before the first arm; LEV values and frozen band numbers appended below
pre-fleet; VERDICT appended when the committed scorer reports; A20 review before banking.)*

---

## PRE-EXECUTION DESIGN-REVIEW AMENDMENTS PA-1…PA-10 (2026-08-23, pre-commit, NO arm has run; review banked verbatim in `A20_REVIEW_B0_DESIGN_20260823.md`; A21 STOP taken on Findings 2+4 and discharged by this block)

**PA-1 (Finding 1 — §1 proof disclosures).** §1's proof parenthesis is replaced by: *"(proof:
(1−w)/w = M_Ḡ q_Ḡ/(M_G q_G); under d ~ q_G the q_G measure cancels and ∫q_Ḡ = 1 — valid only
where supp(q_Ḡ) ⊆ supp(q_G) on generator-reachable data, and note the test constrains one
moment, E_{q_Ḡ}[p_gen/q_G] = 1, not pointwise correctness)."*

**PA-2 (Finding 2 — venue premise REFUTED; [ORCH-RULE 5]: primary fix adopted).** The stock b0
generator does not realize the mixture's class-G law (1/d_L² proxy draw with no R_eff(M) mass
weighting; no S̄_φ acceptance; z_true := listed z). **All identity arms run in a NEW host mode
`catalogue_selected`** (committed with the driver, harness-only): host g drawn ∝ w_g·S̃_φ,g
with w_g the estimator's own `_rate_weight` and S̃_φ,g = ∫k_g(z)·S̄_φ(z;H_TRUE)dz the
kernel-smeared survival; z_true drawn per event from k_g(z)·S̄_φ(z)/S̃_φ,g; d_L and sky noise
conventions unchanged. Justification per the review: rate-weighted hosting is the injection
side's own convention, S̄_φ thinning is the venue's own detection model (the B-SEL draw
convention), and z_true|z_listed ~ kernel is the catalogue's declared error model. The venue is
named **b0i** (b0-identity); the 25 banked b0 seeds are a DIFFERENT generator — replica-gate
anchor and cross-venue LEV only. The importance-weight fallback is NOT adopted (it leaves an
unquantified O(1)-class venue term against exactly the B-T PASS branch this test exists for).

**PA-3 (Finding 3 — ratio computation).** The per-event ratio is computed as
`(1−w_e)/w_e = B_num/(β_G_φ·L_cat_no_bh)` directly (β_G_φ = alpha_G_phi/r_Malm from banked
columns), never through 1−w.

**PA-4 (Finding 4 — odds constant REFUTED and replaced).** §1's M_G/M_Ḡ = α_G_φ/(D̃_φ−α_G_φ)
is WRONG (α_G_φ is the with-BH class weight; and per-arm self-consistent masses would make the
B-R control pass vacuously). The registered constant is the SINGLE derivation-fixed
**C\* = β_G_φ(0.73)·ρ(0.73)/β̄_Ḡ_φ(0.73), identical for all three arms**, with
ρ = Σ̃^φ/Σ^φ (kernel-smeared vs point-evaluated selection mass; ρ ≡ 1 under the PA-2 aligned
generator, computed and banked anyway as the zero-variance mass companion). Registered per-arm
predictions: **B-T: I = 0 (the PASS target) · B-C: I+1 ≈ ⟨S̄_φ⟩ (order-of-magnitude, fails
low) · B-R: I+1 = 1/R(0.73) EXACTLY — the control must fail AT its predicted value** (a B-R
value away from 1/R(0.73) falsifies the scorer, not the arrangement). §7 blindness gains:
*"M_Ḡ = β̄_Ḡ_φ is asserted from the O6/O7 confirmations, not re-derived here; an M_Ḡ error is
common-mode multiplicative across all arms — it cannot reorder them but can push all out of
band."*

**PA-5 (Finding 5 — verdict map).** Both-PASS is upgraded: under C\*, B-C carries a predicted
O(1) displacement, so both-PASS falsifies the mass derivation itself and returns to stage 0
(not mere insensitivity). Both-FAIL ⇒ VENUE-MISSPEC is sound only now that PA-2/PA-4 are in
force. The driver asserts `H_TRUE in H_GRID_FULL` (float equality) before any arm.

**PA-6 (Finding 6 — heavy tails; the trim twin REFUTED).** The 5%-trim verdict twin is
replaced: (a) primary unchanged in form (untrimmed per-seed mean, fleet SEM at seed level) with
**dead-row accounting registered**: n_dead(a) = rows with L_cat_no_bh = 0 banked per arm; any
dead row under a catalogue-hosted generator is a support violation reported as such, never
dropped; |Δ dead-row rate| > 1/200 between compared arms VOIDS the comparison. (b) The
robustness twin is **PSIS** (generalized-Pareto fit to the top tail of (1−w)/w; report k̂):
UNDETERMINED(a) iff k̂ > 0.7 AND raw/PSIS disagree in band. (c) Per-seed median and
reciprocal-form alternatives are refuted (banked in the review). (d) The mass-half companion
(Σ_w, Σ^φ, Σ̃^φ, ρ, C\*) is banked at machine precision as a registered secondary.

**PA-7 (Finding 7 — gates).** E-B0(c) is replaced: *"runtime assertions confirm both flags
reached every dispatch path the run exercised; the driver additionally invokes the scalar path
once on one event as a smoke check, outside the registered statistics."* Denominators defined
once: LIVE(a) = rows with L_cat_no_bh > 0 in arm a (the §4 primary's denominator);
PAIRED-LIVE = intersection (E-B0(b), secondary 3). W-B0 names its β source (alpha_G_phi/r_Malm,
cross-checked against the run's selection-table JSON per h). E-B0(a) exists for seed 900101
only. §5 LEV purpose (i) reworded: the banked read is the coded arm's TOTAL displacement
(arrangement ⊕ venue-premise terms ⊕ the Σ³ᴰ-slot constant, inseparable on the banked basis —
never quoted as arrangement-only). **ε_I is frozen from the pilot's realized scatter ONLY.**

**PA-8 (Finding 8).** §7 gains the one-line note: without-replacement host draws negatively
correlate events within a seed; negligible at n≈200 from a ≳10⁵ pool; absorbed by seed-level SEM.

**PA-9 (driver finding — W-B0 tolerance; [ORCH-RULE 6]).** The registered 1e-9 closure
tolerance is unsatisfiable on CSV-scored columns: `alpha_G_phi`/`r_Malm`/`D_tilde_phi` are
stored at 7 significant figures, flooring the residual at ~1.25e-7 (measured, uniform across
all 25 banked seeds). W-B0 tolerance is amended to **≤1e-6 relative** with the storage-precision
floor disclosed; the full-precision cross-check runs against the selection-table JSON.

**PA-10 (Σ^φ plan item (i) disposition; [ORCH-RULE 7]).** The faithful production-object
r_φ(h) measurement requires the cluster-resident `simulations/injections/` pool (recon-verified
absent locally). It is DEFERRED to a cluster task (queued in the next runbook); the Σ^φ
production-adoption gate stays open pending it, per measure-first. The b0i arms carry the
corrected slot via the committed counterfactual flag regardless (venue-internal consistency,
not a production claim).

**LEV (per §5, run zero-compute on the 25 banked b0 CSVs, PRE-pilot, banked in
`p3_b0_work/p3_b0_lev_output.json`):** untrimmed Ī_banked(coded) = 2.19e4 ± 2.19e4 — one
near-zero-w event on seed 900112 dominates (the PA-6 pathology realized in data, pre-fleet);
trimmed per-seed I_s ∈ [−0.99, −0.64], Ī_trim ≈ −0.968. Read per PA-7: TOTAL displacement,
inseparable; order-of-magnitude O(1) ≫ any plausible band resolution ⇒ the ≥5× LEV threshold
passes; the heavy-tail regime confirms PSIS as the registered twin.

*(This block, the banked review, the Σ^φ flag implementation, and the driver are committed
together BEFORE any gate/pilot/fleet arm runs. Band NUMBERS (ε_I) still freeze pre-fleet from
the pilot.)*

## IMPLEMENTATION-REVIEW AMENDMENTS PA-11…PA-14 (2026-08-23, pre-commit, NO arm has run; verification banked verbatim in `A20_REVIEW_B0_IMPL_20260823.md`, verdict BLOCKED with exact fixes — all folded below before commit)

**PA-11 (kernel alignment — FATAL 1 fix registered).** The `catalogue_selected` draw, S̃_φ,g,
and the Σ̃^φ companion use the ESTIMATOR'S OWN numerator kernel under the run flags:
`k_g(z) ∝ N(z; z_g, σ_eff)·w_pop(z)·f_k(z at the host pixel; ZoA fallback)` renormalized on the
±4σ / 1e-6-floored window (the Z_g convention), at h = H_TRUE — NOT the bare Gaussian (measured
misalignment of the bare form: z_true mis-centered median +0.32σ, S̃ off ~3% — first-order
against exactly the B-T PASS target; unwaivable).

**PA-12 (ρ — the PA-4 exactness claim corrected).** The estimator's Σ^φ leaf point-evaluates
S̄_φ at listed z while Σ̃^φ smears ⇒ ρ ≢ 1 structurally (second order in σ_z via S̄_φ
curvature; bare-kernel measurement 0.9964, expected ~0.97 under PA-11). C\* uses the ACTUAL
banked ρ (machine precision, PA-6d); no exactness assert — a sanity window ρ ∈ (0.9, 1.1) only.

**PA-13 (gates).** (a) E-B0(a) is re-registered on a SAME-VENUE pair: one additional b0i
seed-900101 run with `catalogue_global_selection="s3d"` (all else = B-C); the gate requires the
`L_cat_no_bh` ratio to be one h-dependent constant on 100% of PAIRED-LIVE rows (costing +≈1
CPU-h, folded into §8). The cross-venue replica comparison is DEMOTED to reported-only.
(b) W-B0 (closure ∧ selection-JSON cross-checks, both arms) is wired into the verdict-gating
set. (c) B-R scoring threads the rescale into the scored ratio itself (denom = r_h·β_G_φ·L_cat,
closure against the patched combined) so I(B-R)+1 = (I(B-T)+1)/R by construction of the scorer,
and the control-at-predicted-value tolerance is REGISTERED at 0.05. (d) The PA-2 rate-weight
parity gate (≤1e-12 vs the estimator's own `_rate_weight` leaf) is invoked by the driver itself
before any b0i draw. (e) Per-arm k̂ = max over seeds (disclosed convention).

**PA-14 (LEV re-bank — the PA-10 block's numbers superseded).** The earlier LEV figures
predate the PA-3 direct-ratio scorer. Corrected banked read (25 banked b0 seeds, coded,
cross-basis, TOTAL displacement, inseparable): untrimmed Ī = 7.88e41 ± 7.88e41 (one
near-zero-denominator event; k̂ ≫ 1 infinite-mean regime, PSIS correctly refuses to tame it:
5.29e10), trimmed (reported-only) −0.916, dead rows 48/1690 banked as support violations,
closure floor 1.249e-7 (= the PA-9 storage floor). The ≥5× LEV threshold conclusion stands
(O(1) trimmed displacement ≫ any plausible band resolution); the untrimmed banked mean is
quotable ONLY with its infinite-mean caveat.

**PA-14 CORRECTION (2026-08-23, pre-commit; re-verification finding):** the PA-14 prose quoted
the pre-PA-11 (bare-kernel) LEV values. The banked artifact of record
(`p3_b0_work/p3_b0_lev_output.json`, byte-reproduced by the re-verifier from a fresh run) is
post-PA-11: Ī = 7.814e41 ± 7.814e41, PSIS 5.2405e10, k̂_max = 11.085, trimmed (reported-only)
−0.9166, dead rows 48/1690, closure floor 1.249e-7, ρ = 0.9877707323280376. No gate, band, or
the ≥5× LEV conclusion changes. Registration is COMMIT-READY per the banked re-verification.

**PA-15 (2026-08-23 ~14:55; ops incident + instrument correction, NO evaluate() had run in the
pilot).** The first pilot launch (10:56) OOM-SIGKILLed within ~1 min: `kernel_smeared_survival`
materialized five (20.8M × 50) float64 intermediates (~40+ GB) building the b0i draw weights
over the full host pool — the verifier's probes ran at n≤3000 and could not see it; the
driver's own `mass_companion` had the same lesson (chunked at 20k). Fix: internal 100k-row
chunking (pure memory-shape transform; probed byte-identical, max abs diff 0.0 at 1.5× the
chunk size; 8 relevant tests green). The death was silent (no traceback) and BOTH watcher
generations missed it — a `grep|head` exit-status bug, then a pgrep self-match — detected only
on the author's ETA query ~4 h later. Watcher protocol corrected: the launcher writes the
driver PID to a file; the monitor polls `kill -0` on that PID (no pattern matching). Registered
statistics unaffected (no arm data existed). [OWNED ×3: the unchunked full-pool pass, and two
defective watchers.]

## BAND FREEZE (2026-08-23 ~19:20, post-pilot, PRE-FLEET; formulas as registered in §4/PA-6)

Pilot (seed 900101, both arms fresh; banked `p3_b0_work/pilot_identity_freeze_inputs.json`):
C\* = 0.170472 (mass companion: ρ = 0.987771); n_live = 105/106 both arms, dead-rate
IDENTICAL (1/106, same event ⇒ no VOID trigger); W-B0 closure 1.246e-7 ≤ 1e-6 both arms.
Realized identity reads: **B-C I_s = −0.5887** (PSIS −0.6037, k̂ = 0.851, event-level SEM of
the mean 0.1315) · **B-T I_s = −0.1526** (PSIS −0.1797, k̂ = 0.963, event-level SEM 0.3199).
Direction per the PA-4 predictions (B-C fails low; B-T nearest 0); one-seed reads, NO verdict
content (banked for the freeze only).

**ε_I FROZEN = 0.10** — the projected fleet resolution of the NOISIER arm (0.3199/√12 ≈ 0.092,
rounded up; the B-C projection is 0.038). Bands as registered: PASS(a) iff |Ī(a)| ≤
max(3·SEM(a), 0.10) with PSIS agreement; k̂ > 0.7 AND raw/PSIS band disagreement ⇒
UNDETERMINED(a). B-R must land at (Ī(B-T)+1)/R − 1 within 0.05 (PA-13(c)). Anchors may only
TIGHTEN post-data (the §4 rule). Pilot Δmean_h(B-T − B-C) = +0.0455 (same sign as the B-SEL
twin; leverage secondary 3's prior). Fleet launches: 2 arms × 12 seeds, seed 900101 reused
idempotently (22 fresh evaluates remain).

**PA-16 (2026-08-23 ~21:20; execution-venue split, author-directed).** Author (verbatim): "You
should be using the cluster for this. If you switch it would still be faster, wouldn't it?"
The fleet moves to the registered §8 cluster-fallback path mid-fleet: seeds 900101–900104 are
banked from the LOCAL leg (both arms, complete pairs); seeds 900105–900112 run as a cluster
job array (both arms per seed on the same machine — the pairing constraint: the primary is a
paired per-seed statistic, so no seed mixes venues across arms; partial local work roots for
unbanked seeds deleted before handoff). Same code state (commit of record stamped per A22 in
every meta; the cluster checkout is synced to it and verified before submission). The
machine-split is banked per seed in the metas and disclosed to the A20 review; cross-machine
float non-identity is bounded by the same ≤1e-12-class fallback reasoning as the
multiprocessing gates — it affects both arms of a pair equally by construction.

---

## VERDICT (2026-08-24; all gates PASS; review banked verbatim in `A20_REVIEW_B0_VERDICT_20260824.md`; A21-B0-A/B/C adopted)

**[MEASURED, as adjudicated]: UNDISCRIMINATING** — the §4 B-R control clause fires: the
deliberately-wrong R-rescaled arrangement scores IDENTITY-PASS under the same bands as B-C/B-T,
falsifying the bands' discriminating power in the realized k̂ ≈ 1–2.7 heavy-tail regime
(one legitimate near-zero-responsibility event — seed 900108 idx 2, w ≈ 2.3e-5, listed-z pull
−0.79σ — owns the fleet means). The driver's printed "MASS-DERIVATION-FALSIFIED" is VOID as a
verdict quote (A21-B0-A: the verdict map dropped the control clause's "regardless of B-C/B-T"
scope). The scorer itself is VALIDATED (A21-B0-B: the constructive identity holds to 1.4e-14;
the coded control-FAIL compared against the superseded 1/R−1 target). Banked: all gates; the
E-B0(a) venue-internal 1/r_φ(h) confirmation (ratio 1.128688 at h = 0.73, CV ~3e-15); the mass
companion (C\* = 0.170472, ρ = 0.987771, ⟨S̄_φ⟩_w = 0.785133); per-seed vectors; secondary 3
Δmean_h(B-T − B-C) = +0.0566 (12/12 positive). REPORTED-ONLY (unregistered conditioning,
heavy-tail-biased low by an unquantified amount, never an arm verdict): clean-11-seed
B-C = −0.665 ± 0.042 (its predicted sign/direction; ×2.34 off the ⟨S̄_φ⟩−1 point prediction),
B-T = −0.350 ± 0.096, **B-T closer to calibrated odds in 11/11 clean seeds** (sign test
p ≈ 1e-3). Returns to the author/stage 0: the mass derivation (unresolved), the twin
(neither confirmed nor refuted; direction encouraging, sub-verdict), a finite-moment identity
statistic redesign, and the M_Ḡ common-mode question. A21-B0-C binds any rerun of this family.
**[ORCH-banked, provisional — awaiting the author's stage-5 ruling.]**
