# PROPOSAL — Cross-model epistemics protocol ("model tendency" concern)

**Status: PROPOSED, awaiting author ruling.** Raised by the author 2026-08-21, verbatim: *"I worked
for ~2 weeks only with fable as an orchestrator and only on Wednesday/Thursday I used Opus. This is
exactly when the narrativ changed drastically [...] Now we switched overnight to fable again and
the answer is 'we clearly identified all components of the bias' and the Opus verifier sounds very
different again. I am worried these are model tendencies and I want to discuss what the best
operating rule should be."*

## 1. Orchestrator's assessment of the evidence (honest, against interest)

Three factors are **confounded** in the observed narrative oscillation, and the record cannot
currently separate them:

1. **Model identity** (Fable vs Opus) — the author's hypothesis.
2. **Role and brief** — Opus was deployed *both times* as an adversarial verifier/retrospective
   with an explicit falsification brief; Fable ran *both times* as the builder/orchestrator whose
   job is synthesis and forward motion. Builders overclaim and reviewers overturn **in any
   pairing**; we have never run the reverse assignment.
3. **Context freshness** — the Opus passes also had *clean context windows*. A builder carrying
   200k+ tokens of its own narrative is invested in that narrative's coherence; a fresh reader is
   not. This is a known failure axis independent of model.

What the record DOES support: **the numbers survived every transition; the confidence packaging
did not.** The sentinel defect (Opus-era) was real and stands. The O2/O3 decomposition (Fable-era)
was real and was explicitly *cleared* by the Opus review. The two FATALs were in the Fable-written
*presentation layer* (a bias-reference bug and a rail-coincidence narrative), not in the banked
statistics. So the failure mode to defend against is precisely: **the builder's narrative layer
sets the confidence dial, and the builder's model+context+role biases that dial** — whether the
model term is large is an open empirical question (§3).

## 2. Proposed operating rules (decision table)

| # | rule | tag | cost |
|---|---|---|---|
| P1 | **Role–model rotation:** the model that builds a claim never verifies it. Every ledger row that BANKS, PROMOTES, or WITHDRAWS a claim requires an adversarial pass by a *different* model with a *clean context* before the author rules. (Codifies what caught rows #151's FATALs.) | [STANDING] | ~1 review/campaign |
| P2 | **Blind verification:** the verifier receives the registered artifacts, scorers, and data — *not* the readout narrative — mirroring Stage L's symptom-card independence. (The 2026-08-21 reviewer read the readout prose; anchoring risk was real even though it overturned anyway.) | [STANDING] | none |
| P3 | **Typed narrative layer:** every interpretive sentence in a readout/report carries a type tag — MEASURED (number + provenance), BANKED (author-ratified), PROVISIONAL, or NARRATIVE-HYPOTHESIS — and a report may not contain claims absent from its pre-committed scorer's output. (FATAL-2's "reconstructed from first principles" was an untyped narrative addition; it would have been impossible to write as MEASURED.) | [STANDING] | writing discipline |
| P4 | **Calibration ledger:** extend the retrospective ledger with a per-era table: claims banked under each orchestration configuration (model, role, context age) vs claims later withdrawn/downgraded — so "model tendency" becomes a tracked rate, not an impression. | [DO] | minutes/cycle |
| P5 | **Symmetric-brief probes (periodic):** on selected checkpoints, run the *identical* adversarial brief on both models against the same artifacts and diff the finding sets — the direct measurement of the model term, isolating it from role and context. First instance: §3. | [DO] | ~1 probe/major verdict |

**Recommendation:** adopt P1–P3 as standing rules (they are cheap and each is evidenced by a
specific failure in this repo's record); P4–P5 as [DO] instruments. If adopted, P1–P3 become
amendment **A20** in `docs/RESEARCH_CYCLE.md`.

## 3. First symmetric probe (EXP-class, running 2026-08-21)

A fresh-context **Fable** subagent has been dispatched with the brief given to the Opus reviewer,
**byte-identical up to two documented deviations**: (i) the repo path points to a detached git
worktree pinned at `f59a6f48` — the exact pre-correction snapshot the Opus reviewer attacked — so
the probe cannot see the review, the corrections, or rows #152–#153; (ii) a one-line venv
bootstrap note (the worktree has no `.venv`). Readout:

- If fresh-Fable independently finds FATAL-1/FATAL-2/MAJOR-1-class findings → the oscillation is
  dominated by **role + context freshness**, not model identity; P1's "different model" clause can
  relax to "different context + adversarial brief", though cross-model remains cheap insurance.
- If fresh-Fable misses them or produces a systematically more confirmatory read → direct evidence
  for the **model term**; P1's cross-model clause is load-bearing and should be strict.
- Either way the finding-set diff is appended here verbatim.

*Caveat registered up front: n = 1, single direction (Fable-as-verifier); the complementary probe
(Opus-as-builder) is a future instance under P5.*

## 2b. Author direction (2026-08-21, pre-probe; verbatim)

> "lets wait for the fable verifier experiment and if it shows that it is only about critically
> verifying rather then the model this should be the rule instead of complicating about model
> choices. The hirarchy should be clear anyway Fable orchestration and Opus could be the critical
> thinker if it turns out to be good at this. this naturally pairs Fable with Opus also. And if
> Fable rate limit is exceeded we can switch to Opus and have to revisit the journey with fable
> once it is available again"

Orchestrator reading (flagged as derived): the probe adjudicates between two rule shapes —
**(i)** if fresh-context same-model verification reproduces the decisive findings ⇒ adopt the
*simple* rule: **mandatory clean-context adversarial verification at every BANK/PROMOTE/WITHDRAW**
(P1 without the cross-model clause; P2/P3 unchanged), with the standing pairing Fable-orchestrates /
Opus-as-critical-thinker kept as the natural default rather than a requirement; **(ii)** if the
same-model probe misses them or reads confirmatorily ⇒ the cross-model clause is load-bearing and
P1 stays strict. Standing operational rule either way: Fable orchestrates; on Fable rate-limit,
Opus may orchestrate, and the Fable-orchestrated review of that stretch is REQUIRED once Fable is
available again ("revisit the journey").

## 3b. Probe result (appended 2026-08-21, same day)

**The finding sets are essentially ISOMORPHIC on every decisive item.** Fresh-context Fable
(banked verbatim: `results/prod2d_closure_20260818/SYMMETRIC_PROBE_FABLE_REVIEW_20260821.md`),
blind to the Opus review and to all corrections, independently found:

| decisive finding | Opus review | Fable probe | agreement |
|---|---|---|---|
| δ-arm bias reference bug (FATAL-1) | ✔ | ✔ Finding 1 — same corrected values (−0.0599/−0.1544 full; −0.0363/−0.0995 matched) | exact |
| full-channel rail coincidence, "reconstructed" withdrawn (FATAL-2) | ✔ | ✔ Finding 1 — map=0.600, r_low 100%, same conclusion | exact |
| realized sd 0.0751 → 6.0σ from zero, 1.07σ past edge (MAJOR-1) | ✔ | ✔ Finding 6 — identical numbers | exact |
| N-adequacy realized 4.98σ < registered 5 (MAJOR-2) | ✔ | ✔ Finding 6 — identical | exact |
| GATE S slope = truncation/shrinkage artifact, ŝ→0.616 on extended grid (MAJOR-3) | ✔ | ✔ Finding 3 — same model, same 0.616 | exact |
| mean_h statistics grid-truncation-dominated, extended-grid shifts (MAJOR-4) | ✔ | ✔ Finding 2 — same computed shifts (0.5828/0.6444) | exact |
| ∫B_num = β_Ḡ identity assumed, z-cap parity untested (MAJOR-5) | ✔ | ✔ Finding 5 — same attack | exact |
| BAND R actually paired (corr ≈0.998) (MAJOR-8) | ✔ | ✔ Finding 7 — same measurement | exact |
| GATE V amended prongs near-vacuous (MINOR-1) | ✔ | ✔ Finding 4 | exact |
| S_REF/2 edge conventional (MINOR-3) | ✔ | ✔ Finding 6 | exact |

Each reviewer additionally contributed 1–3 *secondary* items the other did not (Opus: the
−1.222/−1.105 derivative decomposition, quadrature specifics, heteroscedastic SE; Fable: cross-arm
common-seed correlations 0.87–0.95 → paired GATE S SE = 0.130, O2's 73% carries ±11 rel-% seed
scatter, the GATE V counterfactual-freedom note, the noiseless-sky/sampler un-listed invariants —
the last corroborating Opus's N-5). The two disagreed on **nothing decisive**; even the
next-step recommendations converge (both demand the ∫B_num = β_Ḡ / z-cap test — Opus as "O4",
Fable as the content of the audit).

**Conclusion (n = 1, direction stated in §3):** the oscillation is dominated by **role + clean
context + adversarial brief**, not by model identity. Per the author's pre-registered direction
(§2b), the simple rule applies.

## 4. ADOPTED per the author's pre-commitment — becomes amendment A20

The author's §2b direction ("if it shows that it is only about critically verifying rather then
the model **this should be the rule** instead of complicating about model choices") is a
conditional ruling whose condition the probe satisfied. Adopted as **A20**:

> **A20 — mandatory clean-context adversarial verification.** Every ledger row that BANKS,
> PROMOTES, or WITHDRAWS a claim receives, before the author rules, an adversarial review by an
> agent with a **clean context** and an explicit falsification brief; the builder's own context
> never verifies its own claim. Model choice is NOT regulated — the standing default pairing is
> Fable-orchestrates / Opus-as-critical-thinker; on Fable rate-limit, Opus may orchestrate and a
> Fable-orchestrated revisit of that stretch is REQUIRED once available. Verifiers receive
> artifacts, scorers, and data — not the readout narrative (P2). Interpretive sentences in
> reports are typed MEASURED / BANKED / PROVISIONAL / NARRATIVE-HYPOTHESIS, and a report may not
> contain claims absent from its pre-committed scorer output (P3).

P4 (calibration ledger) and P5 (periodic symmetric probes, including the untested Opus-as-builder
direction) remain [DO] instruments, unadopted-as-rules.
