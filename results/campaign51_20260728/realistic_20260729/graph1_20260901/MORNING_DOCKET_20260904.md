# Morning docket — 2026-09-04 (Research Graph 1 wave 3 + addendum; overnight session of 2026-09-03)

Chair: Fable 5.1 orchestration session. Author of record: Jasper Seehofer. Every chair decision below
was made under the row #325 grant ("decide but flag") and is veto-able; every scientific ruling is a
[RULE] that returns here. Ledger rows #334–#347 (all pushed). End-verification of every decisive
number: exec/END_VERIFICATION_20260903.md (row #345). Approval tags per CLAUDE.md.

## 0. One-paragraph state
Wave 3 executed to its decide layer. The post-flip coverage re-run (S3) reads DEFECT-SIGNATURE at
N=200 in both channels with the defect localized to the catalogue-hosted class and 15–24 % of
posteriors pinned at the upper grid rail. S0-B on the production venue is LEVER-LIVE on both θ axes
(Z_b −5.3, Z_ln s −7.2; the data prefer ~3× narrower photo-z errors) — but the run's truth node
differs from the production comparand on 562/1588 events, so "θ-pull on the production venue" is
unverified until a matched comparand exists. Cone loss does NOT own the bias floor (0.4 %, powered;
the 10 outside-cone events pull the estimate toward truth). The 2D offset is carried by ~3–6 % of
events, not diffusely. A new systematic candidate: waveform-timeout rate falls from 87 % to 11 %
across the injected-mass range (12σ), i.e. the injection pool is selected against low-M EMRIs.
Branch G (completion residual): see §3. Five addendum branches drafted; two registrations written.

## 1. Findings of the night (facts; sources in the ledger rows)
| # | finding | row |
|---|---|---|
| F1 | S3 post-flip: PIT-KS D 0.32/0.33 (S) vs exact crit 0.163 → OUTSIDE both channels; centering +0.042/+0.050 above truth (Z 6.0/7.3 per registration SEM); score-zero fails ONLY in the catalogue-hosted class (Z 9.8/7.2; dark 1.3/1.8 inside); byte-pin S 63/63 green; rails 15–24 % all at h=0.86 → bounds | #335, #345 |
| F2 | S0-B production: LEVER-LIVE / MIXED (b small: b̂ −0.011; s material: ln ŝ −1.17, extrapolated) / POWERED; REPORTED-ONLY | #336 |
| F3 | S0-B truth node ≠ production comparand: combined_no_bh differs on 562/1588 events (max_rel 0.73, all positive); T-ID unsatisfied; cause OPEN; no iiib/θ-sites-2.2 comparand exists | #345, #342 |
| F4 | Dark-class criterion `L_cat_no_bh == 0` is float-fragile (157 labels flipped on 1e-110…1e-8 values; physics identical on those events) | #337 |
| F5 | Cone loss IMMATERIAL: Δh −0.00027 ± 0.00088 (φ 0.4 %, M 9); leave-out −0.0049 (φ 7.8 %) booked with non-linearity flag; OUT events pull toward truth | #344, #345 |
| F6 | 2D offset carried by ~3–6 % of events (minimal subset), bootstrap SE/σ_h ≈ 0.9; NOT diffuse | #342 |
| F7 | Timeout selection: SNR-stage timeout rate by M bin 0.87 / 0.78 / 0.27 / 0.11 / 0.26 (12.2σ gradient); e0, p0 flat; seed3000 only (68 % depth) | #342 |
| F8 | GATE PARITY residual (row #273) is EXPLAINED-BY-DESIGN: zero-candidate events Δ = 0 exactly; added-candidate events Δ ≥ 0, no exceptions | #342 |
| F9 | The charter's "−0.14/event" completion residual is a STALE [INFER] on a pre-flip class split; the 17 % cone figure is sound but mirror-scoped | #338 |
| F10 | B-R control (power form) on the b0 identity is DISCRIMINATING on the banked basis (Z −16.6); the finite-moment redesign was already executed as C-A | #342 |
| F11 | Process incident: design-gate reviewers unblinded both G/H primary statistics (chair prompt); mitigated, disclosed | #340 |

## 2. [RULE]s asked (each with the chair's flagged provisional reading; "Ratified" accepts it)
| # | ruling | chair's provisional reading (veto-able) | inputs |
|---|---|---|---|
| R1 | d-calibration (partial): S3 booking DEFECT-SIGNATURE at N=200, both channels; coverage numbers are BOUNDS (rail); q-postflip-calibration NOT killed (revision 1 of 2) | Ratify the booking; do NOT launch S5; route the catalogue-hosted-class localization (F1) + the 3–6 % subset (F6) into ONE follow-up register node (Graph 2 seed) | #335/#345 |
| R2 | S3 revision-2 re-registration: spawn or park? | PARK until R1's follow-up says what to change; a re-run at N=200 without a mechanism is compute without information | #335 |
| R3 | d-photoz-leverage item 1: charter clause (g-score-null red STOPs) vs registration (LEVER-LIVE foreseen) | Option A: amend the panel clause ("g-score-null runs on CONTROL venues; on production the truth-node score is the read") | dossier |
| R4 | d-photoz-leverage item 5 (new): accept a 1-task matched comparand re-evaluation (iiib, θ-sites 2.2, θ=(0,1), h=0.73, ≈0.1 CPU-h) BEFORE any θ-pull interpretation | Approve; until then c-theta-pull stays CONJECTURED with a LIVE registered read attached | #345 |
| R5 | d-photoz-leverage items 2–3: the h-bound for the residual split (S0-C 3-h follow-on ≈ 6 CPU-h) and promotion of c-theta-pull | Defer both behind R4 | dossier |
| R6 | d-cone-register (retro) + m-cone-loss disposition: IMMATERIAL-FLOOR-SHARE on the leave-out number with flag; q-cone-loss SETTLED-refuted ("cones own the floor" is false) | Ratify; c-residual-floor-consistent loses its geometric candidate | #344/#345 |
| R7 | Blindness incident (F11): accept the mitigation path (thresholds frozen pre-leak, disjoint readers) or discard both G/H arms and re-register with fresh reviewers | Accept the mitigation; the reads are REPORTED-ONLY either way | #340 |
| R8 | Dark-class criterion (F4): adopt a relative threshold (build node, g-byte-id on physics, re-derive class counts) | Approve as a build node | #337 |
| R9 | d-timeout-bound (F7): open G7 row 8 as a NEW SYSTEMATIC (mass-dependent timeout selection) and authorize the seed61000 log fetch (A-L2, cluster read, not evacuation) | Approve; this is a paper-relevant selection effect | #342 |
| R10 | Addendum charter A-0 (branches J–N) + the per-branch DOs of its §3 (measures only; the zero-compute reads are DONE) | Ratify A-0; approve A-J2 (401-grid, ≤15 CPU-h) only if R3/R4 need it; A-M1 prior P-A = U[0.79,0.82] as registered; A-N: close on C-A (N1 parity stamp optional) | addendum |
| R11 | Row #273 wording correction (E19 is the zero-candidate floor, not the mechanism of the 44.7 % residual) | Ratify the erratum | #342 |

## 3. Branch G (completion residual) — READ DONE (row #347)
Four build/gate rounds (each gate fresh, computability-only; no revision consumed), then one real-mode
read by a disjoint agent: production dark-class matched-channel score T_prod = −0.197 ± 0.019
(Z −10.1, 1512 events); harness (self-consistent universe, 67 × ~172 dark events) T_harn = −0.051 ±
0.007 (Z −6.9); ρ = T_harn/T_prod = 0.26 ± 0.05 → **INTERMEDIATE (b) partial**: about a quarter of
the production dark-class residual is reproduced where the estimator is consistent by construction;
three quarters is production-only. Closure exact (1e-14); all gates green incl. resolved-flags
equality 67/67. delta_h_M (reported-only) −0.091 vs the −0.063 rail → the linear map over-predicts.
| R12 | d-completion-register (retro) + m-completion-residual disposition INTERMEDIATE (b) | Ratify the booking; the attribution (what the 74 % production-only part IS — venue physics per S0-B's LIVE θ-pull? catalogue-leg inconsistency? the 3–6 % event subset?) is d-residual-attribution's question and needs R4's comparand first | #347 |
| R13 | Optional replication cell R (seed block 903000–903029, ≈ 22–43 CPU-h) for the harness leg | Not needed: Z_harn = −6.9 is decisive on 67 universes; spend the compute on R4/R9 instead | #347 |

## 4. Chair decisions taken tonight (veto list)
D1 executed the addendum §6.1 zero-compute set (4 reads + 2 drafts) under row #325 (#339) ·
D2 ratified d-cone-register under docket 2.2 conditional on a computability gate and let the
disjoint read run (#341/#344) · D3 booked S3 DEFECT-SIGNATURE, S0-B LEVER-LIVE, cone IMMATERIAL
(all chair-derived, all return here) · D4 accepted the blindness mitigation path (#340) · D5 accepted
all nine end-verification discrepancies as errata (#345).

## 5. Backlog (author words, untouched tonight by your instruction)
12a archive backup destination (159 GB sole copy) · 12b cluster evacuation before 2026-09-23 ·
12c disk (87 %) · 12e merge → main · 12f/12g safe builds, docs sync · GATE-ACC reporting-only stage
relaunch (dead on the login node, libpython) · seed61000 timeout-log fetch (R9).

## 6. Compute spent tonight
Cluster: 0 CPU-h (no job submitted). Local: ≈ 0.5 CPU-h (aggregations, reads, bootstrap).
Top-tier identities: chair, wave-3 prereg author, addendum prereg author, end-verifier (4 across two
workflows; ≤3 concurrent). Sonnet agents: ~55.

## 7. Batch 2 additions (rows #349–#362; SSH lost at 23:03, row #357/#359)
| # | ruling | chair's provisional reading | inputs |
|---|---|---|---|
| R14 | What "catalogue-hosted" MEANS for class-conditioned reads (exact-zero support vs materiality threshold vs continuous f_cat): the relative label re-splits the post-flip re-baseline 1241/347 | Register f_cat as a continuous covariate everywhere; keep exact-zero for backward comparability with a disclosed fragility | #350, #362 |
| R15 | m-offset-subset: INTERMEDIATE by the literal table; primary family SUBSET-IDENTIFIED — the 82 offset-carrying events are high-z (AUC 0.87), few-candidate, low-f_cat, low-SNR; top-z decile leave-out +0.086 | Ratify SUBSET-IDENTIFIED for the primary with the 1D data-contract gap disclosed; open Graph 2 branch "high-z incomplete-catalogue balance" as the mechanism register node | #362 |
| R16 | S0-B provenance: the S0-B driver's iiib venue differs from production in the catalogue leg (θ-sites has zero effect; suspect the 2D-survival pin); R4b job prepared, not submitted | Submit R4b first thing; if byte-identical, the θ-pull read is on the pre-[P3-2D] counterfactual and d-photoz-leverage needs a re-run on production (≈2 CPU-h) | #355 |
| R9′ | Timeout selection downgraded to one axis (M), 0.92 % of draws; r-timeout-selection Q1/Q2 registered; p0 axis = the D1 bound | Approve Q2 (zero-compute) after R15; Q1 after the pool build-log fetch | #358 |
| R17 | Sealed-mock m1 (job 6790859) and S0-C (6790794) ran unattended; reads in the morning | Retrieve via cluster/agent_ssh.sh; the m1 binary read and the S0-C derivative return as dossiers | #354, #357 |
Chair decisions added to the veto list: D6 batch-2 execution under the renewed standing (#349); D7 S0-C cap on the allocation basis (#352); D8 m1 pool pin corrected to mix200k (#357); D9 the offset-subset blindness note (#360); D10 the ssh-guard hook + wrapper (#359, code change under the author's "ensure it does not happen again").
