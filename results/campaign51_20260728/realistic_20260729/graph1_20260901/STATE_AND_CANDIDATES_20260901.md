# State collector — 2026-09-01

Reader-node record (no verdict, no adjudication, no source edits). Sources: BIAS_HISTORY_LEDGER.md
rows #278-#288 (results/campaign51_20260728/realistic_20260729/gate_b_20260730/), RUNBOOK_NEXT_SESSION_40.md
(results/campaign51_20260728/), artifact "Two Trees and the Residual Bias" (a8824799) sections 06-10,
B8_2_S3_PILOT_READOUT_RECORD.md (results/campaign51_20260728/realistic_20260729/tree2_20260830/).

## 1. State table

| Headline number | Value | Source row / doc |
|---|---|---|
| Flip executed | [PHYSICS] commit 5e7fda16 -- catalogue_leg_1d_mass_aware default changed to "auto" (engages "on" iff numerator+global-selection resolve "phi" and theta-divisor is "off") | Row #286 |
| 1D MAP (post-flip, arm c, iiib) | 0.665 | Row #286 |
| 1D mean_h (post-flip, arm c, iiib) | 0.66699 | Row #286 |
| Residual 1D rail | -0.063 (mean 0.667 vs truth 0.730) -- now OWNED as the mass-blind/mass-aware mismatch, named B8 [CAL]'s next centerpiece | Row #286; runbook 40 section 0 and section 3 item 6 |
| 2D offset, iiib (pre-row-282/286 headline figure) | -0.0667 vs truth; sigma_h = 0.0184 (bias approx 3.6x its own width) | Artifact a8824799 section 00 KPI strip |
| Information floor, sigma_h | 0.001747058397810697 (no_bh channel) / 0.001746970592930231 (with_bh channel), at N_ref = 1588 | b8_information_floor.json, quoted in B8_2_S3_PILOT_READOUT_RECORD.md section 3.5; also artifact section 07 ("0.00175") |
| F candidates, pre-flip pilot (cell S, N=200, mixed-N contaminated aggregate) | no_bh F = 7.426 (approx 7.4); with_bh F = 11.35 (approx 11.4) | Row #288; pilot record section 3.1 |
| Coverage numbers (cell S pilot, n_universes=66, mixed-N) | no_bh HPD 50/68/90/95 = 0.015 / 0.015 / 0.061 / 0.121 (all far out of band); with_bh HPD 50/68/90/95 = 0.364 / 0.470 / 0.803 / 0.894 (out of band); PIT-KS D = 0.8045 (no_bh) / 0.3313 (with_bh) | Row #288; pilot record section 3.1 |
| [HIER] certified both axes | b-axis measured null-consistent at the full T1.3 configuration (score_b no-BH Z=-1.808, with-BH Z=+0.773, both inside |Z|<=3); the a-axis was already certified. Instrument now certified on both axes at T1.3 by direct measurement, not assumption | Row #287 |
| A14 corrected deltas | +0.002507 (iiib, 2D) / +0.004114 (joint_r1, 2D), both <= T_mat = 0.008 -> PASS; 1D exact-zero both venues at all 41 nodes | Row #284 (item-20 end-verifier Part 2, correcting row #283's unit-grid-weight read) |
| Transform (T2.2b derived) | S_4D / S-bar_phi median = 1.039 (66 hosts, h-stable) | Row #282 |
| Depth-skew | 73.0% +/- 1.4% of catalogue-leg weight sits below the true redshift for a dark event; 16 standard errors from no-skew, stable across seeds/h-nodes | Artifact section 05, Chain B item 1 (fanout-1 branch B4/[IMP] handed to tree 2) |
| Cone loss | approx 17% -- localisation cones that structurally cannot contain the true host; leading candidate for the absolute bias floor that consistency fixes will not touch | Artifact section 09 (open-branches board) |
| Completion-leg residual | approx -0.14 per event (dark class); largest unexplained item on the board, named B8's object | Artifact section 09; runbook 40 section 0 ("B8 [CAL]'s next centerpiece") |

Supporting numbers used above but not separately itemised in the task list: floor mass collapsed 0.617 -> 1.8e-4 on the A18 arm c verdict (row #286); suite 2006 passed / 6+1 skipped, the one nominal T8 failure adjudicated not-flip-related (row #286); the 14 G-EXT extension-node tasks failed on the h-prior upper bound, disclosed as verdict-irrelevant at tail 5e-13 (row #286); pure-input fork RESOLVED, +157.92 nats binds, +123.11 was an O2 7-s.f. storage artifact from catastrophic cancellation (row #282).

## 2. Candidate list for the next batch

Each candidate: what it settles / cost band / inputs / natural gate-instrument / shared-decision flag.

1. S3 post-flip coverage re-run + S4 registration repair
   - Settles: whether the B8.2 coverage/F numbers are usable at all post-flip, and delivers a clean (non-contaminated, both-cell) coverage readout.
   - Cost band: moderate. The pre-flip pilot at N=200 took approx 12h wall for 63/100 (cell S) and approx 4h for 20/25 (cell T), both wall-limited, not completion-limited -- a rerun at the same N is the same order of cost, cheaper if parallelised across cluster workers.
   - Inputs: the executed flip (row #286); the S4 registration fixes named in row #288 (a)-(c) -- separate the N-ladder timing seeds from the N=200 pilot seeds, add the missing cell-T aggregation, and decide the stop-rule under wall-limited runs.
   - Gate/instrument: B8.2 harness design note (B8_2_HARNESS_DESIGN_20260829.md) section 8's S4 registration review, required before any S5 production-N launch.
   - Shared decision: yes -- feeds the paper's "F/coverage-validated result" deliverable and is the concrete measurement behind row #288 item (d)'s fresh RULE ask (whether S3 re-runs post-flip at all, since the pre-flip no-BH channel numbers cannot calibrate a post-flip stop rule).

2. S0-B production run
   - Settles: whether the photo-z error-model theta-pull is real venue physics (interpretable now that the instrument reads null on both axes) rather than an instrument artefact -- this is the leading remaining lever on the "irreducible venue physics" side of the residual.
   - Cost band: cheap-to-moderate cluster item, comparable to the 8-cell b-node precursor (runner-11) that just closed.
   - Inputs: PA-HIER-33 scorer implementation (ratified row #278/#280, not yet built) must land FIRST; the driver's missing iiib venue path must be fixed. Runner-11's output (8-cell b-node pair) is the read-first precondition per runbook 40 section 3 item 1.
   - Gate/instrument: cluster queue, behind PA-HIER-33 scorer + driver fix; certified [HIER] instrument at T1.3 (row #287) is the qualifying condition that unblocked it.
   - Shared decision: yes -- feeds the "irreducible venue physics" bound in the artifact's three-way residual split (section 10 item 3) and any paper claim about photo-z-model leverage on H0.

3. Falsifier (ii) -- class-G fleet
   - Settles: the PROVISIONAL attribution cap on B7.1/B7.2 (whether the A4 mz_sel/eff structural-consistency ratification can drop its PROVISIONAL flag).
   - Cost band: approx 40-60 CPU-h (runbook 40 section 2), the next cluster rung alongside S0-B and the T5 k-scan.
   - Inputs: independent of the flip; can run in parallel with S0-B and T5 once cluster queue slots free up.
   - Gate/instrument: the A4 [RULE] (rows #278(4)/#280/#284(3)) -- explicitly "returns with numbers, not auto-ratified" pending this falsifier.
   - Shared decision: yes -- directly gates final A4 ratification, one of the three still-open T5/A4 scope words.

4. Dark-class completion-leg residual program (approx -0.14/event)
   - Settles: the largest unexplained item on the board -- how much of the residual gap is illegitimate estimator inconsistency vs legitimate (floor-consistent) noise, once F is known.
   - Cost band: unscoped / open-ended. No registered arm exists yet -- this is a derivation-plus-measurement program, not a single run.
   - Inputs: needs the B8.2 F measurement (candidate 1) to land first, so the "how many legitimate sigma" question is answerable; also benefits from the post-flip HEAD re-baseline (candidate 11) as its reference point.
   - Gate/instrument: none registered yet -- needs a fresh pre-registration (research-cycle stage 2/3) before any decisive run; named explicitly as "B8 [CAL]'s next centerpiece" (row #286) and "the dark-class completion-leg object" (runbook 40 section 3 item 6).
   - Shared decision: yes -- central to the three-way residual split (artifact section 10) and the paper's honest-bound framing.

5. Cone-loss quantification (approx 17%)
   - Settles: how much of the absolute bias floor is structurally untouchable by any consistency fix (localisation cones that cannot contain the true host at all).
   - Cost band: unscoped -- "no registered arm yet" (artifact section 09); likely a catalogue-geometry analysis rather than a cluster-heavy run, but not yet designed or costed.
   - Inputs: independent of the flip and of B8.2; needs its own pre-registration against the localisation-cone / catalogue-completeness geometry.
   - Gate/instrument: none yet.
   - Shared decision: yes -- feeds the same "irreducible venue physics" floor as S0-B and depth-skew, needed for the paper's honest bound on how far any fix can get.

6. T5 k-scan, Arm S and Arm R
   - Settles: whether a log-symmetric (exact-by-construction) mass window materially outperforms the current linear k=1.5 window, per-venue (delta law on iiib vs log-normal on joint_r1).
   - Cost band: Arm S (iiib, log-geometry k in {2.0, 2.5, 3.5} plus optional k=infinity anchor) approx 15-20 CPU-h; Arm R (joint_r1, decisive k=3) approx 11-15 CPU-h plus a C0-prime ingredient gate.
   - Inputs: cluster queue slot; Arm R additionally needs a C0-prime-equivalent gate check for its own configuration (the generic C0-prime gate already passed in wave 3, job 6746274, was for the 2D-twin check, not this arm).
   - Gate/instrument: the F-ii [RULE] (row #278(2), restated row #284(4)) which already ratified the design and the 78.9% retention-figure retirement; Arm R itself is "RATIFIED-AS-RECOMMENDED, launch when cluster allows" (row #284(4a)).
   - Shared decision: yes -- resolves one of the three remaining open T5 scope words and the mass-law-window adoption decision.

7. r_phi divisor question
   - Settles: whether the catalogue leg's mass-blind divisor (r_phi approx 0.886) is a live discrepancy or is now structurally resolved.
   - Cost band: cheap -- a confirmation/closure check, not a new measurement.
   - Inputs: the executed A18 flip (row #286) and the T2.2b-derived transform (1.039, row #282). The artifact itself flagged this as "Likely mooted if the mass-aware flip lands" (section 09) -- the flip has now landed, so under the "auto"/"on" mass-aware branch the catalogue leg's numerator and divisor are matched-content by construction (Z = 1 identically per the Chain B derivation), which is precisely the condition that made r_phi's mismatch possible in the mass-blind branch.
   - Gate/instrument: none needed beyond a written confirmation note tying it into the A18 closure record.
   - Shared decision: low-weight but should be formally retired on the open-branches board so it does not linger as a stale live item; does not block any of the above.

8. The h-prior upper bound (blocked the G-EXT nodes)
   - Settles: whether the extended h-grid (G-EXT amendment, above 0.86) is trustworthy at its outer nodes for any future measurement that needs them.
   - Cost band: cheap -- a config/code fix plus a rerun of the 14 failed extension-node tasks (a fraction of the original approx 94 CPU-h grid).
   - Inputs: independent of other candidates; relevant wherever the extended grid is load-bearing.
   - Gate/instrument: none formal; disclosed as "verdict-irrelevant at tail 5e-13" for the A18 verdict already reached (row #286), so this is maintenance, not a blocking gate for anything already decided.
   - Shared decision: only if a future run needs the full extended grid (e.g. a post-flip HEAD re-baseline or T5 arm that probes high h) -- otherwise deferred.

9. Paper deliverables (1D-vs-2D verdict, F/coverage-validated result, negative mass-channel information result)
   - Settles: the terminal synthesis -- what actually goes in the paper.
   - Cost band: low compute, but a substantial authoring/synthesis task; per CLAUDE.md's "Proposing decisions" rule, needs a reviewable artifact with the decision table inline, not a chat summary.
   - Inputs: draws on nearly everything else on this list -- the A18 flip and its residual (done), B8.2's F/coverage result (candidate 1, pending), falsifier (ii) for A4 (candidate 3, pending), the completion-leg and cone-loss programs (candidates 4-5, pending) for the honest-bound framing, and the already-derived negative mass-channel information result (sigma_M sweep, artifact section 08 table).
   - Gate/instrument: the physics-change protocol where any formula/claim changes; the approval-scope convention (RULE tags) for any scientific ruling folded into the paper text.
   - Shared decision: yes -- this is the terminal shared decision that nearly every other candidate rolls up into.

10. joint_r1 mass-aware arm (venue transfer of the flip)
    - Settles: whether the "auto" default's resolution logic (engage "on" iff numerator+global-selection resolve "phi" and theta-divisor "off") also produces a valid, in-band 1D result on joint_r1, not just iiib -- the A18 production arm (job 6747032) ran on the banked iiib CLI only (row #285).
    - Cost band: comparable order to the A18 iiib arm, approx 90-100 CPU-h for an equivalent grid, though it may be smaller if reusing the existing 41-node grid without the G-EXT extension.
    - Inputs: the executed flip (row #286); F-ii's finding that the production mass law is venue-dependent (delta law on iiib vs log-normal realized-forward law on joint_r1, row #270) means joint_r1's own T2.2b-equivalent transform may differ from 1.039 and would need its own derivation before a band can be registered.
    - Gate/instrument: none registered yet -- needs a fresh readout rule analogous to A18's (registered band, map-AND-mean criterion), since the iiib band does not automatically transfer to a differently-scattered venue.
    - Shared decision: yes -- validates (or bounds) the flip's generality across venues before the "auto" default can be called settled project-wide, and is scheduling-adjacent to the T5 Arm R work (both touch joint_r1 and its per-venue mass law).

11. Post-flip HEAD re-baseline
    - Settles: establishes a new reference "banked" HEAD readout under the post-flip production default, since the current banked comparand (2026-08-27, certified bit-identical by the C0-prime gate in row #281) predates the flip and was only ever used to validate the 2D-twin adoption (A14), not the 1D mass-aware change.
    - Cost band: cheap, modeled on the wave-3 blind HEAD arrays (84 tasks total across both venues, approx 6.5 min/task -- single-digit CPU-h).
    - Inputs: the executed flip (row #286) is the only precondition.
    - Gate/instrument: none registered yet -- would follow the same C0-prime-then-blind-HEAD pattern used for wave 3 (rows #279/#281/#283).
    - Shared decision: yes -- becomes the new comparand baseline that every subsequent delta-read (S3 post-flip, falsifier (ii), the joint_r1 mass-aware arm, T5 arms) should measure against; foundational rather than decisive on its own.
