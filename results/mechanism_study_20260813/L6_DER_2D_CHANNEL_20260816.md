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
