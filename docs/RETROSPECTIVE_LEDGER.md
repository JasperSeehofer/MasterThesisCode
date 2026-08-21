# Retrospective Ledger

**Established 2026-08-21 by author mandate** (verbatim, from the 2026-08-21 overnight-autonomy
instruction): *"Once a research cylce ends or something fails I want you to put this in a
retrospective ledger or a comparable ledger that tracks and suggests amendements."*

**Contract.** One entry per (a) research-cycle end — any stage-5/6 close, whether the verdict
confirmed, refuted, or MIXED — and (b) any **failure**: a gate that fired, a control that proved
vacuous, a registration returned NOT-READY, a withdrawn claim, a wasted-compute incident, an
operational loss (expired artifacts, dead workspace). **Append-only, never back-filled, no silent
edits** — the same discipline as `BIAS_HISTORY_LEDGER.md` and the amendment ledger in
`docs/RESEARCH_CYCLE.md`.

Each entry carries five fields:

1. **What ran / what failed** — one paragraph, with the ledger rows and artifacts it closes over.
2. **What worked** — which rules, amendments, and instruments earned their keep, by name.
   Every amendment named here gets **+1** in `docs/AMENDMENT_IMPACT_TRACKER.md` with a dated
   evidence line (same commit).
3. **What failed or dragged** — the honest list, including orchestrator errors, not only code.
4. **Suggested amendments** — each tagged `PROPOSED` (needs author ruling) or, when covered by an
   explicit author standing grant, `ADOPTED-UNDER-STANDING` with the grant quoted. Suggestions
   route to the amendment ledger in `docs/RESEARCH_CYCLE.md`; this ledger is where they are *born*,
   that one is where they *bind*.
5. **Disposition** — where the thread went next (runbook §, follow-on cycle, or CLOSED).

Precursor: `docs/RETROSPECTIVE_D1_20260820.md` (the D-1 post-mortem, written before this ledger
existed) is the form this ledger generalises; it is referenced, not rewritten.

---

*(entries below, newest last)*

## Entry 1 — 2026-08-21 — C-SG pilot STOP: GATE V fired on 3/4 seeds (gate miscalibrated by channel porting)

1. **What ran / what failed.** The mandatory 4-seed C-SG-F pilot (job 6415588) completed 4/4
   on-anchor, and the pre-committed band-setter returned `fleet_may_launch=false`: 3/4 seeds
   failed GATE V (spans 2.15–4.76 nats < 5; σ_h > 0.5·σ_prior). No fleet was launched on the
   fired STOP. Diagnosis: v2 §6's GATE V numbers were written for the FULL-channel posterior;
   the orchestrator's v3 design change ported them to the matched channel **without re-deriving
   their operating characteristics** — the exact omission A15 exists to prevent, committed by the
   same orchestrator that had credited A15 twice earlier the same night. Independent reference
   check: the v2 thresholds false-fail **5/12 banked B-SEL matched posteriors** (known-informative
   data). Amended thresholds (span ≥ 1 nat, σ_h ≤ 0.9·σ_prior — the flat-null vacuity signature)
   published with reference false-fail 0/16 and a still-can-fail demonstration (B-F1 flat mode
   fails both prongs). Prereg block "PILOT GATE V AMENDMENT"; v2 verdicts remain recorded in every
   banked JSON.
2. **What worked.** A15's pilot-first mandate + the pre-committed STOP wiring (the gate fired
   BEFORE 42 seeds ran — cost of the failure: 4 pilot seeds that remain usable). The
   reference-data method (amend gates against independent known-informative data, never against
   the data that fired them). → A15 **+1** in the impact tracker.
3. **What failed or dragged.** Orchestrator porting slip (above). Also: GATE V's σ_prior
   convention was never numerically specified in the prereg and had to be invented at
   implementation time — a registration under-specification.
4. **Suggested amendments.** PROPOSED (needs author ruling): *every gate that is moved to a new
   statistic, channel, or venue re-derives its operating characteristics against reference data
   in the same commit — porting numeric thresholds across channels without a stated false-fail
   rate is registered as an A15 violation.* (Candidate A17; not adopted under the standing grant
   because it binds future physics-adjacent judgment calls — author review requested.)
5. **Disposition.** Bands frozen (`csg_pilot_bands_output.json`), fleet launched; thread
   continues in the C-SG readout.

## Entry 2 — 2026-08-21 — C-SG cycle END: BAND C = INTERNAL-DEFECT (46/46 seeds); the −0.11 reconstructed from first principles in one overnight session

1. **What ran.** The full registered chain in one autonomous night: pre-check O2 (impostor leg
   carries 73% of −0.1083) → pre-check O3 (matched channel −0.0846; the pure channel's mildness is
   cancellation) → v3 design change (matched channel primary) → implementation (7-agent workflow,
   2-lens adversarial GO) → pilot (STOP fired, diagnosed, gate re-derived on reference data) →
   frozen bands → 42-seed fleet → **BAND C = INTERNAL-DEFECT on both statistics**, full channel
   reproducing −0.108 in every arm. Rows #149–#151; readout report
   `CAMPAIGN_READOUT_REPORT_CSG_20260821.md`. Plus Stage-L: the venue is below Gray 2020's own
   validated completeness floor and the per-sector decomposition is literature-novel.
2. **What worked.** A1 (two free reads reshaped the design before any CPU; +2 credited),
   A12 (score-zero frame defined the matched channel; +1), A15 (twice: the O2/O3 band design and
   the pilot-first STOP that caught the ported gate; +2 total), A5 (Stage-L found the
   validated-floor gap; +1), A10 (the invariance declaration forced the S̄_φ designation as the
   next step; +1 this entry). Prereg-first discipline held at every step: every scorer existed
   before its data. Independent recomputation held: O2 by a firewalled agent, fleet numbers by
   the orchestrator from raw diagnostics.
3. **What failed or dragged.** (a) The GATE V porting slip (entry 1). (b) GATE S fired
   CONTROL-INERT by its letter against ordered arm means — the INERT band was registered without
   asking what an *attenuated* (rather than absent) slope would mean; a two-sided/three-outcome
   slope rule would have been cleaner (A8's two-sidedness lesson, recurring in a new guise).
   (c) The 16-cpu single-process over-reservation followed the house convention unexamined
   (gotcha 7); ≈550 reserved vs ≈35 consumed core-h.
4. **Suggested amendments.** A17 (entry 1, PROPOSED) covers (a). For (b): PROPOSED — *slope-type
   validity gates register all three outcomes (absent / attenuated / unit-consistent) with the
   attenuated band's meaning stated at registration time*; could fold into A8 rather than a new
   number. For (c): route to the standing A6 audit (no new amendment).
5. **Disposition.** Six author [RULE]s queued (readout report §10); next technical step under
   either branch = independent `S̄_φ` audit; runbook 26 is the entry point.

## Entry 3 — 2026-08-21 — Author-requested adversarial review found 2 FATALs in the orchestrator's readout layer; overclaims withdrawn same-morning

1. **What failed.** After a discipline-heavy night (4 pre-committed scorers, frozen bands,
   independent recomputes), the *readout/presentation layer* still shipped two fatal errors:
   (a) every arm's bias measured against the global 0.73 instead of its own h_gen — corrupting
   the δ-arm numbers, the "every arm reproduces −0.108" claim, AND the orchestrator's own
   GATE S "attenuated" qualification built on them; (b) a railed posterior's location (map_h =
   0.600 in 46/46) narrated as a "first-principles reconstruction" of the −0.11. Both entered
   ledger row #151 and the readout report before being caught — by an Opus review the author
   requested, not by the night's own checks. Also confirmed: realized scatter 1.56× the pilot's
   σ̂ (the defect-edge margin is 1.07σ, not categorical); BAND R's registered independence
   rationale was wrong (arms provably paired, corr 0.9975).
2. **What worked.** The review-then-re-derive loop: every finding was reproduced by the
   orchestrator before any correction was applied, and the corrections landed append-only with
   the superseded values retained. The score-zero primary statistic survived everything (6.05σ,
   grid-invariant) — A12's design choice is what kept the campaign's core finding alive through
   the review. The review also *cleared* O2/O3's mechanics explicitly.
3. **What failed structurally.** Independent recomputation was applied to *inputs* (O2 headline,
   fleet means) but not to the *readout semantics* (which reference does "bias" use per arm?);
   the falsifier asymmetry (A14 falsifier registered only for the branch that did NOT fire) went
   unnoticed until the review named it; verification effort concentrated pre-run, thinned
   post-run.
4. **Suggested amendments.** PROPOSED A17-extension: gates and bands re-state their operating
   characteristics on REALIZED scatter at readout (the N-adequacy gate would have failed at 4.98σ
   vs its registered 5). PROPOSED (new): every readout scorer declares, per arm, the reference
   value each "bias"/"error" statistic subtracts, as a printed field — a wrong implicit reference
   is a silent FATAL. PROPOSED (new): A14 tightened — a pre-registration must arm a falsifier for
   EVERY branch, not only the branch the designer expects.
5. **Disposition.** Corrections in prereg CORRECTION & REVIEW ADDENDUM + ledger row #152; the
   INTERNAL-DEFECT label downgraded to PROVISIONAL (non-zero score at 6σ stands); pre-check O4
   (common-domain/quadrature pairing test) proposed as the discriminating next step, superseding
   the S̄_φ designation; review banked verbatim.

## Entry 4 — 2026-08-21 — O4 cycle: an unearned verdict caught by its own falsifier and the first A20 review; mechanism identified; three registration/execution failures and one ops failure logged

1. **What ran / what failed.** Pre-check O4 (domain/quadrature pairing test, 15 F seeds) fired
   DEFECT-HARDENED as-run — and was overturned the same evening: the first A20 clean-context
   review found the executed arm A had dropped the REGISTERED S̄_φ zero-extension on a
   post-registration "corrected premise" (bands never re-derived ⇒ VOID-BY-DEVIATION), and the
   registered A19 falsifier fired — the S̄_φ leg difference owns 106.5% of the score; the restored
   registered arm nulls it (+0.0076 ± 0.0184). Mechanism identified: the off-cell completion
   numerator omits the survival factor its normalizer carries (legacy pre-#118 cell; `fused` is
   the in-tree fix). O5 (free B-SELF read) corroborated. Execution itself also failed twice
   before succeeding: a serial run at ~13 min/seed was killed for pace, then a 12-wide local
   parallel run was OOM-slaughtered (evaluate ≈ 9 GB/seed vs a 30 GB box) and recovered via a
   cluster array + 2-wide local hedge. Ledger row #155.
2. **What worked.** A19 (the falsifier the author mandated for EVERY branch fired on its first
   outing and overturned an unearned band — +1 credited); A20 (first application: the
   artifacts-only clean-context review caught the void verdict BEFORE banking and did the
   mechanism identification — +1 credited); the free-read reflex (O5 on banked B-SELF confirmed
   the mechanism at zero compute); GATE R4's bit-exactness held across two venues; the two-track
   recovery (cluster + hedged local) converged without losing a seed; the orchestrator re-derived
   every decisive review number before the record was written (3-seed locus reproduction).
3. **What failed or dragged.**
   (a) **Post-registration arm deviation without band re-derivation** — the implementer's
   "corrected premise" was technically right about production's code path and still invalidated
   the band table; it was disclosed in a docstring, which is not a registration amendment.
   (b) **The verify-phase agent accepted the deviation** — it checked "alignment spec implemented"
   against the docstring's own framing rather than the registration text; spec-compliance
   verification anchored on the implementer's narrative is exactly the P2 failure mode, one level
   down.
   (c) **The registration lacked an axis-leverage calculation** (orchestrator, A15-class): the
   domain/quadrature axis could move the statistic ~1e-5 against a 0.0582 band half-width —
   DEFECT-HARDENED was the only reachable outcome before any seed ran, computable in one line at
   registration time.
   (d) **Instrument costing missed per-seed wall time AND peak RSS** — the serial grind, the OOM,
   and the cluster recovery each cost ~1–2 h of wall clock that a two-line costing table at
   registration (time/seed, RSS/seed, venue decision) would have avoided.
   (e) Minor: factorial arm A1 numerically invalid (<1 GL node/σ); 3/15 replica-gate rows
   compared cached artifacts while claiming regeneration; harness background watchers were killed
   repeatedly until the compute was detached via setsid (keepalive pattern).
4. **Suggested amendments.**
   PROPOSED (**A21 candidate — the load-bearing one**): *no executed arm may deviate from its
   registration text — any "corrected premise" discovered during implementation STOPS the run and
   returns to the registration for an amendment with re-derived bands BEFORE execution; a
   docstring disclosure is not an amendment.* Evidence: this cycle, where the deviation selected
   the band.
   PROPOSED (fold into A17): every registration states each varied axis's maximum leverage on the
   decision statistic; a band unreachable by the registered axis is void on its face (the
   symmetric counterpart of A15's "a band inside the null is void").
   PROPOSED (fold into A6/A17): instrument registrations carry a costing line — wall time/seed,
   peak RSS/seed, and the venue decision (local vs cluster) — before launch.
   PROPOSED (verification practice, P2 extension): spec-compliance verifiers receive the
   REGISTRATION text as their reference, never the implementer's summary/docstrings as framing.
5. **Disposition.** Mechanism banked as MEASURED (row #155); author queue: label ratification,
   the one fused end-to-end confirmation seed, the production-basis fork. The campaign's central
   question — what the 6σ score violation IS — is answered pending those rulings.
