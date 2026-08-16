# L6-DER — the 2D excess derived: mass-factor node tracking (channel B)

**Date:** 2026-08-16 · **Authorized:** ledger rows #111 item 3 + #112 (option A) · **Status:
PRESENTED, NOT ADJUDICATED.** Derivation-first per `L6_2D_GI_PLAN_20260816.md` §2 item 3: the
prediction below was computed BEFORE any switch measurement; the c2-mirror switch confirmation is
the registered next step.

## 1. The mechanism

Of g's two h-channels (plan §1), the owner is **channel B — node tracking**: the 2D integrand
c2 = ∫ kern·p_gw·g dz concentrates at the GW peak z*(h); as h rises, z* moves up (dz*/dh =
D/(hD′) > 0) and g is re-evaluated there. g's explicit z-dependence at fixed d_L_frac runs
through scale = M_z_obs/(1+z) against the φ mass-function: moving to higher z lowers the implied
source-frame mass, and the population sits where φ rises toward lower masses — so
**d ln g/dz > 0 (population mean +0.306)** and the responsibility-weighted g grows with h:

    T_B = Σ_e (d ln g/dz)|_{z_true, f=1} · (dz*/dh)_e

**Channel A is null at the peak:** d ln g/df at f = 1 measures the μ_cond-shift sensitivity of the
Hermite integral, and it evaluates to ~0 (population median +0.0000) — the conditional mean sits
at the ratio-symmetric point.

## 2. The numbers (pinned population, production `completion_mass_factor_g`, n_hermite=64)

| quantity | value |
|---|---:|
| d ln g/df at f=1 (median / mean) | +0.0000 / −0.0000 |
| d ln g/dz at fixed f (median / mean) | +0.2981 / +0.3064 |
| **T_A (channel A, population sum)** | **+0.0 nats/h** |
| **T_B (channel B, population sum)** | **+139.0 nats/h** |
| measured 2D−1D excess (coded / AM2P / AJREN / A-FULL) | +129.0 / +131.5 / +128.7 / +135.7 |

The parameter-free channel-B prediction lands inside the measured range (+139.0 vs +129…+136,
≤8% of every arm) — and the mechanism explains the excess's **variant-independence** exactly:
node tracking is inherited from the shared integrand support by every 1D variant, so no 1D repair
can touch it.

## 3. What it means (posed, not adjudicated)

Channel B is the mass-channel analogue of the 1D mass-growth term: a tracking term whose
correct-form cancellation partner must come from the M-side population/selection structure. The
venue's generator pins M_row (no φ-draw) exactly as it pins z_true — so whether the correct 2D
form cancels T_B via an M-selection pairing (the S̄_φ-in-α structure, conditioned on the 2D
data), or whether the φ-prior's z-drift is genuinely spurious for pinned events, is the
correct-form derivation still owed. Production relevance is direct: `completion_mass_factor_g`
is production code, and this term biases h UP by T_B/Ā ≈ +0.0020 per the displacement law
(measured 2D−1D bias gap: +0.0066 — the remaining structure beyond T_B's share belongs to the
width/coverage channel, unpriced here).

## 4. Registered next steps (plan §2, unchanged)

1. c2 mirror (bit-exact vs stored `ln_post_2d`) + freeze-switches S-A/S-B/S-AB → confirm channel
   ownership on the instrument (prediction: S-B removes ≈ +139; S-A removes ≈ 0).
2. The correct-form 2D derivation (the cancellation-partner question above) → an A-FULL-2D
   candidate if a repair falls out.
3. xhigh verifier before any of it returns to the author as a claim.

Script: `scratchpad/quick_gi_channels.py` logic to be promoted into the committed L6 scorer with
the mirror. Host-dominated approximation disclosed: derivatives at z_true, f=1, M_z at truth —
the mirror switches supersede these approximations.

---

## Addendum (2026-08-16) — §4 item 1 EXECUTED: switch confirmation, on-prediction to the nat

Row #113. Script `l6_c2_switch_decomposition.py`, output `L6_C2_SWITCH_output.json` (15 MN0X
seeds, mirror validated **bit-exactly** on BOTH channels: ln1 and ln2 max-abs-diff 0.0; c1
bit-identical across all switches):

| quantity | measured | registered prediction (committed `718128d1`, pre-run) |
|---|---:|---:|
| T2 − T1 (excess, base) | +131.5 ± 0.1 | — |
| **ΔT2(S-B)** (node tracking frozen) | **−139.0 ± 0.0** | **−139** |
| ΔT2(S-A) (μ_cond frozen) | −0.0 ± 0.0 | ≈ 0 |
| ΔT2(S-AB) | −139.0 ± 0.0 | (no interaction) |

**The 2D excess is fully owned by channel B**, deterministic across seeds at this precision;
post-switch residual excess ≈ −7.5 nats/h. The §3 correct-form question (the M-side
cancellation partner) is now the only open item before an A-FULL-2D candidate.

---

## Addendum 2 (2026-08-16) — xhigh verifier amendments (GO conditional on these; they supersede above where they conflict)

The verifier independently reproduced the +139.0 prediction from scratch (+139.01), confirmed the
timeline (prediction in-tree 35 min before the switch script existed; parameter-free though not
target-blind — the excess band was already committed), confirmed all aggregates, and confirmed
the switch isolation at the g-argument level. Four amendments of record:

**A1 — "fully owned" is withdrawn.** The post-switch residual excess is **−7.489 ± 0.065** —
decisively nonzero (all 15 seeds negative), 5.7% of the effect. Amended claim: **channel B owns
the excess to within ~6%**; the residual's origin is undetermined — most plausibly rigid-shift
over-subtraction of kern-anchored responsibility (the S-B counter-shift subtracts the linearized
tracking term for ALL responsibility, including any that does not actually move with h), possibly
a small third channel.

**A2 — what each agreement demonstrates.** ΔT2(S-B) equals the responsibility-weighted linearized
B-functional by construction, whatever the base mechanism; so "on-prediction to the nat" is the
DERIVATION's validation (the z_true/f=1 approximations are structural — verifier fragility test:
±0.03 z-shifts move the prediction only ±2%). The OWNERSHIP evidence is |ΔT2(S-B)| = 139.0 vs
base excess 131.5 — agreement to 5.7%. Also: the script's c1-bit-identity "assertion" is
tautological (compares ln1 to itself); the true 1D-untouched fact rests on code construction plus
the genuine bit-exact mirror-vs-stored validation on both channels.

**A3 — the production clause is corrected (supersedes §3's "direct").** g's only production call
site is the **completion (dark) leg** integrand (`bayesian_statistics.py:4344`), **mode-gated to
`normalization_mode="absolute_marginal"`** (`:3178`) — the realistic-campaign configuration, NOT
the `generator_marginal` default (which never calls g). The completion leg has h-moving windows
(`:4368-4384`), so **the channel's existence transfers to the campaign path**; its geometry
(no host kernels; (1−f)·dVc weighting) differs from the venue ball, and the catalogue leg's 2D
factor is a different function (`mz_integral`, `:5239`, without the φ-slope structure). **The
venue magnitudes (+131.5 nats/h; +0.0020 displacement share) do NOT transfer** — production's
expression of the channel needs its own derivation on the completion leg, folded into the §3
correct-form question.

**A4 — the "width channel" attribution is withdrawn** (it was speculative and unreconciled with
the committed displacement-law-neutrality of the excess); the unexplained remainder of the 2D−1D
bias gap is recorded as open, unattributed.

**Verifier verdict: GO as amended** — "the core scientific claim — the 2D−1D excess is dominantly
produced by h-moving evaluation of `completion_mass_factor_g`'s z-argument against the φ slope,
channel A is null at f=1, and the derivation's +139.0 is structural — stands."

---

## Addendum 3 (2026-08-16) — RATIFIED

Ledger row #114: the L6 findings as amended by addendum 2 are ratified. Author's verbatim
ruling: "ratified".
