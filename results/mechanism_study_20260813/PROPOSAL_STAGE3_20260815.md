# PROPOSAL — stage 3: locating T_res and testing the score-balance account

**Date:** 2026-08-15 · **Status: PROPOSAL — nothing below is registered, no seed is reserved, no
arm is authorized.** Presented per CLAUDE.md "Proposing decisions" as the reviewable artifact for
the next [DO]. · **Stands on:** ledger rows #103–#104 (M2′ measured PARTIAL on-prediction; T_res
unlocated; M7 closed; M1-as-residual refuted) · Stage-L sweep `STAGE_L_SWEEP_20260815.md`
(`da5f1364`) · verifier-corrected L0 pair (`0a3d940e`).

## 1. The two open questions, stated as falsifiable hypotheses

**H-REN — the unrenormalized truncated kernel owns T_res.** The kernel `N(z; z_obs, σ_k)` is
integrated over `[max(z_lo(h), z_obs−5σ_k), min(z_hi(h), z_obs+5σ_k)]` and **never divided by the
retained mass** `Φ((b−z_obs)/σ_k) − Φ((a−z_obs)/σ_k)` (M3 note §1, confirmed at code level; the
literature states the renormalization requirement explicitly — Stage-L Q4, arXiv:2302.12037).
Why this is not M3 redux: M3 ablated the **window** (what is integrated) and its effect is
p_gw-edge-suppressed (e⁻⁸); renormalization changes the **per-candidate weight** by the retained
kernel fraction — an **O(1)-varying, h-dependent factor at kernel scale** (5σ_k ≈ 0.21 ≫ window
half-width ≈ 0.08, so most candidates are heavily clipped), whose h-derivative
`d/dh ln[Φ(·)−Φ(·)]` carries **no p_gw suppression**. It is σ_k-dependent by construction —
matching T_res's dose dependence — and vanishes identically at σ_k = 0 (point branch), passing
constraint (a). **No arm or toy in this thread has ever ablated it.**

**H-SB — the residual displacement is the misspecification (score-balance) mechanism.** After
J-restoration the measured tilt at truth (1492 ± 31) is ≈ the α tilt (+1393.6), which is correct
physics (M4); in a well-specified model the score at truth is mean-zero, so an uncancelled tilt
indicates the *event-term* likelihood is misspecified and the MAP sits at the pseudo-true (KL-argmin)
value with sandwich-form (not inverse-information) spread — classical grounding White (1982),
Kleijn & van der Vaart (2012) (Stage-L Q3). H-SB predicts quantitatively: displacement ≈
c·T·σ²_post with c ≈ the measured 0.749 ± 0.046, and the ~8.5× overconfidence is the
information-vs-sandwich mismatch — testable on stored posteriors with no new simulation.

H-REN and H-SB are complementary, not rivals: H-REN locates the remaining *misspecified term*;
H-SB explains *how* any residual misspecification displaces a confident posterior. If H-REN closes
T_res, H-SB becomes the framework statement of the whole defect; if H-REN fails, H-SB's diagnostic
tells us where the score imbalance lives.

## 2. Proposed work, in ladder order (all L0 first; CPU-minutes; committed data + toys only)

| item | what | cost | kills/settles |
|---|---|---|---|
| **L0-REN-A** | analytic: derive sign and σ_k-scaling of the renormalization tilt `Σ d/dh ln[retained mass]` on the pinned population; check against the parity constraint and T_res's measured decay (+699/+149/−62) | hours, analytic | H-REN's shape; a wrong sign or wrong dose trend kills it before any toy |
| **L0-REN-B** | A/B toy (committed harness pattern): arm A as coded, arm B with per-candidate renormalization; stacked slope + implied MAP shift + f_i sweep; pre-stated reads in the registration | CPU-minutes | H-REN magnitude vs the registered 1e-3 band |
| **L0-SB** | sandwich diagnostic on stored posteriors (2,400 campaign + 470 mechanism + 40 stage-2): per-seed score variance vs curvature at truth; predicted vs measured overconfidence factor; test bias ≈ c·T·σ²_post out-of-sample (cells not used to fit c) | CPU-minutes | H-SB quantitatively; also independently re-derives the 0.75 as sandwich structure or refutes it |
| **L0-LIT** | full-text reads of the two `UNCHECKED` rows the sweep names (Gray 2020 §2; Gray 2023 §2.1.4) — confirm published pipelines' kernel normalization and measure conventions before we claim novelty | hours | the novelty claims in any eventual paper text |
| **A-REN (conditional L1)** | instrument arm, renormalization restored, fresh registration under A8-v2 with a paired A-NULL-style control; **only if L0-REN survives both its kill checks and the author grants the [DO]** | ~25 CPU-h | H-REN on the instrument |
| **A-JREN (conditional L1)** | combination arm J + renormalization — the first candidate *full-repair* measurement; only meaningful if A-REN lands PARTIAL-or-better | ~25 CPU-h | whether the located terms jointly restore calibration (coverage), which no single term does |

**Pre-decided, so it is cheap now:** if L0-REN kills H-REN and L0-SB confirms H-SB, the honest
conclusion is that the residual is *distributed* misspecification (no further single-term arm), and
the thread's product becomes the measured decomposition + the sandwich account + the repair
candidate for the located terms — routed to the `/physics-change` gate with the author.

## 3. Decision table

| # | decision | tag |
|---|---|---|
| 1 | Run the four L0 items (L0-REN-A/B, L0-SB, L0-LIT) — committed data, toys, and reading only; no instrument time, no registration yet | **[DO]** |
| 2 | Pre-authorize drafting (NOT registering) the A-REN registration in parallel with L0, so a surviving H-REN loses no calendar time | **[DO]** |
| 3 | Whether A-REN (and conditionally A-JREN) may be *registered and run* — returns to the author as a fresh [DO] with the L0 evidence and the full A8-v2 registration attached; seeds from an unreserved block; ≤ 2 L1 arms | **[RULE+DO], deferred by design** |
| 4 | Whether the `/physics-change` gate should receive the J-term dossier update now (the measured, on-prediction partial) or wait for the full-repair candidate — the gate is author-gated on its face either way | **[RULE], author's timing call** |

**Tiering for the [DO] items (stated per mandate):** L0-REN-A derivation — orchestrator; L0-REN-B
toy + L0-SB diagnostic — one sonnet/high agent each; L0-LIT full-text reads — one sonnet/medium
agent; one inherit/xhigh adversarial verifier over the combined L0 results before anything returns
to the author (single top-tier agent; ≤3 cap respected). No workflow needed.

*Append-only from its commit. No repair is proposed here; the `/physics-change` new-formula slot
remains empty and author-gated.*
