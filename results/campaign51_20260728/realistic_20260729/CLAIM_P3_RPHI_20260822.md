# CLAIM INTAKE + STAGE-2 REGISTRATION — [P3-RPHI]: the no-BH catalogue leg's Σ³ᴰ/Σ^φ slot mismatch

**Opened:** 2026-08-22 (row #168 item 4; verifier's §6 finding, now re-measured [LOCAL]).
Research-cycle stages 0–2 in one card (the claim is a measured object; the exoneration layers
carry no entry on the global-selection slot pairing — checked; R0: the bscale memo and Path-A
package are the in-repo references, MFG-a UNCHECKED carried).

## Stage 0/1 — the claim, now [LOCAL]

The no-BH catalogue term divides the local sum by **Σ³ᴰ** (`_global_cat_denom_no_bh`, S_3D-based,
`:3826/:4823`) while its class weight **β_G_φ** is φ-marginal-4D. The measure-conversion pairing
(the ratified bscale memo's §2 logic, applied by the Appendix-A verifier) wants **Σ^φ** in that
slot. Measured on the venue objects (`rphi_measure.py`, committed leaves, zero-evaluate):

| h | Σ^φ | Σ³ᴰ | r_φ |
|---|---|---|---|
| 0.600 | 9.0199e8 | 1.05758e9 | 0.852883 |
| 0.665 | 9.4485e8 | 1.08521e9 | 0.870664 |
| 0.730 | 9.8087e8 | 1.10709e9 | **0.885984** |
| 0.795 | 1.01126e9 | 1.12457e9 | 0.899241 |
| 0.860 | 1.03699e9 | 1.13857e9 | 0.910782 |

**d ln r_φ/dh (chord) = +0.2526** — an un-derived, h-SLOPED ~0.89 multiplicative factor on the
dominant (catalogue) channel: the B_scale defect class (rows #130–#131), in production's coded
arrangement TODAY. Contested against Path A's "all three slots" intent: `Σ^φ` IS built
(`:3878`) but feeds the weight chain (n̂_w^φ), not the leg divisor — whether that asymmetry was
a ruled Path-A decision or an oversight is an open provenance question for the author
(refute-by, part 2). **Refute by:** (1) the 1/r_φ(h) rescore below (does correcting the slot
move the headline?); (2) locating a Path-A decision record ruling the Σ³ᴰ divisor deliberate.

## Stage 2 — RESCORE REGISTRATION (pre-data; A21/A22; the day's amendment rules in force)

**Construction (zero-evaluate):** `cat_term_corrected(e,h) = cat_term_off(e,h) · [1/r_φ(h)]`
with r_φ(h) computed at ALL H_GRID_41 nodes by the SAME committed-leaf instrument (extended
h-list); mixture reassembled by the verified identity; scored by `compute_seed_statistics`
(trapezoid); baseline = banked trapezoid (gated on the headline anchor −0.108302, the
amendment-8-discharged form, disclosed). Per-node r_φ vector + both Σ vectors BANKED (A17(d)).

**Gates:** T-R (the Σ³ᴰ rebuild matches the run-era... no banked Σ³ᴰ column exists — instead:
the REBUILD-CONSISTENCY gate: β̄_Ḡ_φ from the same build matches banked D̃−α ≤2e-6, the T-C
form); I-S mixture identity ≤2e-6 on the banked set (implemented, fail-closed per A17(f));
S-R sanity: r_φ ∈ (0.8, 1.0) monotone. A22 stamp incl. instrument tree.

**Primary:** paired Δ̄_rφ(12) vs the banked baseline; per-seed vector + sd + SEM banked.
**Bands (frozen now):** RPHI-MATERIAL iff |Δ̄| > 0.02 · RPHI-SMALL iff ≤ 0.01 AND SEM ≤ 0.004 ·
REPORT-BOUND otherwise. Two-sided, no commentary. **Axis leverage:** the factor's level (~0.89,
i.e. 1/r_φ ≈ 1.13 up-weighting of the drag-carrying leg) and slope (+0.25/unit h) sit between
the twin's (level 0.35-ish per event) and unity — K-flat's measured +0.039 for a 0.27-level
factor bounds the level term well above band resolution; both bands reachable.
**Costing:** r_φ 41-node build ≈ 2–3 min + rescore < 5 min; local; zero-evaluate.
**A10:** off-basis conditional (same as all P3 rescores — disclosed); the S_3D-vs-φ-marginal
tower question itself (WHY r_φ ≠ 1) is upstream physics not adjudicated here; all-impostor
venue scope warning rides.

*(Instrument `p3_rphi_rescore.py` committed before it runs; VERDICT + A20 review before
banking.)*

## VERDICT + A20 AMENDMENTS (2026-08-22; review banked verbatim in `A20_REVIEW_P3_RPHI_20260822.md`, BANK-WITH-AMENDMENTS, zero FATAL; primary reproduced to −1.5e-16; **the band ruling returns to the author as a fresh [RULE] per the binding default** — nothing below is banked as ruled)

**Measured:** Δ̄(12) = **−0.004309 ± 0.000736** (0/12 positive) ⇒ RPHI-SMALL as a band
assignment, PENDING the author's ruling.

**AMENDMENT 1 — provenance corrected [OWNED].** The stage-0 instrument was never committed
(scratchpad-only) — attribution void; the table VERIFIED by the reviewer's independent leaf
rebuild (all five rows exact). Repaired: the instrument is now committed as
`p3_rphi_measure.py`; future citations cite the committed leaves.

**AMENDMENT 2 — framing strengthened.** Path A's "all three slots" are β_G^φ/β̄_Ḡ^φ/Σ^φ; the
no-BH L_cat divisor is a FOURTH object Path A never rules on. The defect: the code's own
"r_phi == 1 by construction/identically" assertions (`:1751-1753`, `:2422-2423`) are
unqualified and FALSE for this leg (0.886), while the with-BH channel in the same assembly
block pairs correctly. No committed decision rules Σ³ᴰ deliberate.

**AMENDMENT 3 — derivation ratified, alternative refuted.** 1/r_φ = Σ³ᴰ/Σ^φ is algebraically
EXACT (n̂_w^φ ≡ Σ^φ/β_G^φ is a defined code object; Σ^φ is the unique divisor making
β_G^φ·L_cat = A_ball/n̂_w^φ); the β-ratio alternative injects a second rate density (β objects
not even commensurate: ≈62, slope −0.945). The patch is minimal AND complete.

**AMENDMENT 4 — attribution corrected.** ≈56% level (−0.002411 ± 0.000466) ⊕ ≈44% slope
(−0.002112 ± 0.000349) — "h-sloped factor" over-attributed; both components 0/12 positive.

**AMENDMENT 5 — scope conditional (binding).** r_φ is venue/catalogue-specific: 0.8860 here vs
the 0.9119 the code quotes for the production object — NOT comparable, neither supersedes.
RPHI-SMALL bounds THIS venue only; it licenses no production-headline statement and is NOT an
exoneration — the slot is derivation-wrong regardless of |Δ̄|. The anti-conservative direction
(the defect makes the reported bias look +0.0043 smaller) is venue-conditional.

**AMENDMENT 6 — gate reporting corrected.** GATE S-R cannot fail (its band was frozen in the
card that already contained the table — the measured cost of the stage-0/2 fusion) and must
not count as an independent pass; I-S/T-C sit at the 7-digit print-rounding floor (resolving
power ~1e-6); the Σ³ᴰ divisor-of-record remains rebuild-assumed (second-order exposure).

**AMENDMENT 7 — registration hygiene.** Stage 1 retro-labelled as discharged by the
axis-leverage paragraph (A21(a)); recurring MINORs carried (anchor literal; unregistered
GATE_BS_TOL; unpinned comparand CSVs; A22 dirty-scope). Commit-before-run clean; no A21
registration/execution deviation found.
