# M4 — "the selection normalisation α(h) is σ_z-blind"

Mechanism study for the confirmed dark-siren estimator defect (venue-transfer campaign,
`results/venue_transfer_20260811/VENUE_TRANSFER_READOUT.md`; ledger rows 98–99 in
`results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`).

**Analysis only.** No production module was edited, no `/physics-change` gate was opened,
nothing was committed. All code references are reads.

---

## VERDICT

> **M4 REFUTED.** The term M4 says is missing from α(h) is **identically equal to 1**, at
> every σ_z, because the photo-z kernel is normalised over its own data variable and
> catalogue membership is decided on *true* redshifts. σ_z-blindness of α(h) is **correct**,
> not a defect.
>
> **Sign:** none — the "missing" correction to ln α is exactly 0.
> **Scaling:** σ_z⁰ × 0. Not linear, not quadratic; identically zero.
>
> The strongest admissible reformulations of M4 are also dead:
> - a σ_z-**aware** α is mathematically identical to the current α (exactness argument, §2);
> - a σ_z-**blind** defect in α (i.e. α wrong for a reason unrelated to σ_z) predicts
>   bias ∝ σ_z^1.45 from the measured posterior-width scaling, against the measured
>   σ_z^0.93 — a factor-1.9 miss on the 3.5× dose lever, at ~50σ (§4b);
> - **direct test, run here:** removing α(h) *entirely* from the 2 400 already-stored
>   posteriors leaves a σ_z-keyed positive bias of **+0.0165 at σ_z = 0.035** (from +0.0353),
>   still ~linear in dose. The σ_z keying survives the total deletion of α (§5).
>
> Relocated finding (not M4): the numerator *does* omit a normalisation — the **per-candidate
> evidence** `Z_k(h) = ∫ N(ζ_k; z, σ_k) w_pop(z;h) dz` together with the population prior
> `w_pop(z;h)` inside the z-quadrature (§3). It is O(σ_z²), its h-dependence cancels exactly
> (h⁻³ prefactor), and its sign would make the observed bias **worse**. Also refuted as owner.

---

## 1. What α(h) integrates, and what it omits

### 1.1 The estimator as implemented

`darksiren_emri/validation/venue_transfer.py::_channel_terms_at_h` (lines 1099–1180), the
vector-σ twin of the certified gate path `validation/calibration_gate.py:800–884`:

- per-candidate kernel × GW factor, Gauss–Legendre in z
  (`venue_transfer.py:1132–1141`):
  `integ = N(z; ζ_k, σ_k) · N(d_L(z,h)/d_obs_i; 1, σ_d,i)`, integrated over
  `[a,b] = [max(z_lo_p, ζ_k − 5σ_k), min(z_hi_p, ζ_k + 5σ_k)]` (lines 1127–1128), where
  `z_lo_p/z_hi_p` is the **h-dependent** ±4σ_d GW window (lines 1109–1115);
- σ_k = 0 branches to point evaluation, no quadrature (lines 1148–1165) — this is the code
  path that produces constraint (a);
- event likelihood `L_i = Σ_k c_k / K_i` (lines 1167–1168);
- `ln P(h) = Σ_i ln L_i(h) − N · log_alpha[k]` (lines 1176–1177), N **fixed**.

The numerator contains **no** `w_pop`, **no** `S̄_φ`, and **no** per-candidate normaliser.

### 1.2 α(h)

`darksiren_emri/validation/closed_loop_gfrac.py:374–383`:

```
s_phi_table = precompute_phi_marginal_survival(h_grid, detection)     # line 375
for i, h in enumerate(h_grid):                                        # line 378
    z_grid, s_phi = s_phi_table[h]                                    # line 379
    alpha = float(np.trapezoid(_w_pop(z_grid, h) * s_phi, z_grid))    # line 380
    log_alpha[i] = math.log(alpha)                                    # line 383
```

with

- `_w_pop(z,h) = (dV_c/dz)/(1+z)` — `closed_loop_gfrac.py:285–299`, calling
  `physical_relations.comoving_volume_element` (`physical_relations.py:571–601`,
  `dV_c/dz dΩ = d_com²c/H(z)`);
- `S̄_φ(z;h) = ∫ φ(log₁₀M) S_4D(d_L(z;h), M(1+z)) dlog₁₀M` —
  `bayesian_inference/bayesian_statistics.py:1859–1941` (contraction at line 1937), i.e. the
  production φ-marginal survival of the injection-pool detection object;
- the z-range is `[1e-6, z_max(h)]` with `z_max(h) = z(d_L^max(h))` (lines 1911–1915).

**α(h) integrates:** the comoving population measure × the GW detection survival, over redshift,
per h. Nothing else.

**α(h) omits:** σ_z (any), ζ_k (any), per-event ball geometry, K_i, the number/identity of
candidates, and the photo-z kernel entirely. It is one scalar per h, shared by all 982 events,
both channels, and every seed.

**Measured shape** (built here from the campaign's own config,
`injection_pool_mix200k_20260728` + the pinned CRB CSV):
`d ln α / d ln h = −1.0358` over the 41-point grid, with a power-law residual rms of
0.0022 nats — α is, to 0.2%, `α(h) ∝ h^(−1.036)`. (The pure comoving-volume limit is h⁻³; the
survival factor eats two of the three powers.) So the α term contributes
`−N ln α = +1.036 N ln h`, a monotone pull toward **high** h of slope 1 393 nats per unit h at
N = 982.

---

## 2. Does the asymmetry displace the MAP? Sign and scaling.

### 2.1 The hierarchical statement

Write the generative model exactly as the harness implements it. Latent per event:
(z, M). Data: the GW observable `d_obs` (with its σ_d) and the catalogue readings
{ζ_k} of the K_i ball members. Detection is a Bernoulli draw on the **GW observables only**
— `closed_loop_gfrac.py:446–452`, `keep = rng.random(batch) < S_4D(d_L, M_z)`; the estimator's
mirror of that statement is exactly `S̄_φ` (`bayesian_statistics.py:1937`).

The selection-normalised likelihood for one detected event is

```
                  ∫dz dM  w_pop(z;h) φ(M) p(d_obs | z,M,h) 1_det · p({ζ} | z)
p(D | h, det) = ───────────────────────────────────────────────────────────────
                  ∫dz dM  w_pop(z;h) φ(M) [∫dd_obs p(d_obs|z,M,h) 1_det] [∫dζ p(ζ|z)]
```

The denominator is α(h) — **provided the two bracketed factors evaluate as they do here**:

- `∫dd_obs p(d_obs|z,M,h) 1_det = S_4D(d_L(z;h), M(1+z))`, contracted over φ ⇒ `S̄_φ`; ✔
- **`∫dζ p(ζ|z) = 1`.** ✔

### 2.2 The missing term has value 1

The code's kernel is `norm.pdf(z_nodes, loc=ζ_k, scale=σ_k)` (`venue_transfer.py:1139`), i.e.
`N(z; ζ_k, σ_k) = N(ζ_k; z, σ_k)` by symmetry of the Gaussian — a properly normalised density
**in the datum ζ_k**. Therefore

```
α_σ(h) ≡ ∫dz w_pop(z;h) S̄_φ(z;h) · ∫dζ N(ζ; z, σ)  =  ∫dz w_pop S̄_φ · 1  =  α(h)
```

**exactly, for every σ and every per-candidate σ_k.** The term M4 nominates as missing equals 1;
Δ ln α = 0; the induced MAP displacement is 0 at every dose. There is no sign and no scaling to
report because the object does not exist.

The only escape would be selection that depends on the *noisy* redshift — then `∫dζ` would run
over an acceptance region and would no longer be 1. It does not: ball membership is decided on
**true** z inside the GW distance window on the truth ladder
(`calibration_gate.py:677–702` and `venue_transfer.py:845–868`), and the σ_z scatter is applied
**afterwards**, to the already-fixed member list (`calibration_gate.py:705–708`;
`venue_transfer.py:1393,1396`). K_i is pinned and σ_z-independent
(`venue_transfer.py:852`, `draw_ball_pinned`). Nothing about who is in the catalogue, or
whether an event is detected, depends on ζ.

### 2.3 The even-order theorem (why *any* smearing-mismatch story is quadratic)

Convolution with a symmetric kernel of variance σ² is the heat semigroup:

```
(K_σ * f)(ζ) = exp( (σ²/2) ∂²_ζ ) f(ζ) = f + (σ²/2) f'' + (σ⁴/8) f'''' + …
```

Every term is an **even** power of σ. Consequently *any* candidate mechanism whose entire
σ_z-dependence enters through the kernel — the numerator smearing, a hypothetical denominator
smearing, or the mismatch between them — contributes

```
bias(σ_z) = c₂ σ_z² + c₄ σ_z⁴ + …      ⇒      R_dose = bias/σ_z = c₂ σ_z + O(σ_z³)
```

i.e. R_dose **proportional to the dose**. The 3.5× dose lever therefore predicts
R_dose(0.035)/R_dose(0.010) = 3.5. Measured (calgate v2 B-cells, N = 400 each, read here from
`results/calibration_gate_v2_20260810/B*_h0p730_results.json`):

| cell | σ_z | median MAP (refined) | bias | R_dose | post_sd |
|---|---|---|---|---|---|
| B0 | 0 | 0.73004 | +0.00004 | n/a | 0.00000 |
| B1 | 0.010 | 0.74103 | **+0.01103** | **1.103** | 0.00120 |
| B2 | 0.035 | 0.76542 | **+0.03542** | **1.012** | 0.00298 |

Measured ratio **0.92**, predicted **3.5** — a factor 3.8 miss, against per-cell SEs of
~2×10⁻⁴. Fitted exponent: bias ∝ σ_z^**0.93**. The venue-transfer cells extend the lever to
the heterogeneous GLADE mix (σ̄ = 0.0418, R_dose 0.891) — R_dose *falls* with dose, i.e. the
response is very slightly **sub**-linear, the opposite direction from any even-order term.

---

## 3. The self-consistent hierarchical form, and which term is actually missing

Each catalogue entry k is a real galaxy with unknown true redshift z, prior = the population
inside the event's window, observed as ζ_k with error σ_k. Its posterior is
`p(z | ζ_k, h) = N(ζ_k; z, σ_k) w_pop(z;h) 1_{W_i(h)}(z) / Z_k(h)`. The correct catalogue
likelihood is therefore

```
             1        ⌠           N(ζ_k; z, σ_k) · w_pop(z;h) · 1_{W_i(h)}(z)
L_i(h) =  ─────  Σ_k  ⎮ dz  p_gw(d_obs,i | z, h) · ──────────────────────────────
            K_i       ⌡                                      Z_k(h)

             Z_k(h) = ∫ dz  N(ζ_k; z, σ_k) · w_pop(z;h) · 1_{W_i(h)}(z)

ln P(h) = Σ_i ln L_i(h) − N ln α(h),      α(h) = ∫ w_pop(z;h) S̄_φ(z;h) dz   [unchanged]
```

Comparing with `venue_transfer.py:1139–1141`, the code has

```
integ = N(z; ζ_k, σ_k) · p_gw       ⟵  missing: · w_pop(z;h) / Z_k(h)
```

**The missing term is `w_pop(z;h)/Z_k(h)` inside the per-candidate quadrature — not a term in
α(h).** M4 identifies the right *family* (an unmatched normalisation created by the smearing)
but the wrong *object*: the smearing's normaliser is per-candidate, and it lives in the
numerator.

Three properties of that missing term, each of which also disqualifies it as the defect's owner:

1. **h-cancellation.** At fixed Ω_m, `dV_c/dz dΩ = (c/H₀)³ · [∫dz/E]²/E`, so
   `w_pop(z;h) = h⁻³ G(z)` with G h-independent. The h⁻³ prefactor cancels **exactly** between
   `w_pop(z;h)` and `Z_k(h)`. The missing pair carries no explicit h-dependence at all; it acts
   only by re-weighting candidates and re-shaping the z-integrand.
2. **Quadratic scaling.** `Z_k(h) = h⁻³ (exp(σ_k²∂²/2) G)(ζ_k)`, so the correction to each
   candidate term is `1 + (σ_k²/2)(G''/G)(ζ_k) + O(σ_k⁴)` — even-order, per §2.3. Fails
   constraint (b) by the same factor 3.8.
3. **Wrong sign.** To O(σ²) the correct per-candidate posterior is Eddington-shifted upward by
   `σ_k² d ln G/dz > 0` (G rises steeply with z over 0.1 ≲ z ≲ 0.8). Candidates effectively at
   *higher* z, matched against a fixed `d_obs = G(z)/h`, require a **larger** h. Installing the
   missing term would push the MAP **further high** — enlarging the observed +bias, not curing
   it.

---

## 4. Falsification against the three hard constraints

**(a) Must vanish identically at σ_z = 0.** M4 passes — but *vacuously*, and the constraint
turns out not to be discriminating for α-family mechanisms at all. Two reasons, both worth
recording:

- M4's correction is 1 at every σ_z, so of course it is 1 at σ_z = 0. A mechanism that
  "vanishes at σ_z = 0" because it vanishes everywhere is not thereby supported.
- The T-0/B0 anchors cannot see *any* smooth per-h term. Measured here: the median local
  curvature of ln P at the peak is **1.73 × 10⁸ per h²** at T-0 versus **7.35 × 10⁴** at T-b.
  An additive per-h function A(h) shifts the MAP by ≈ A′/C; the 2 645 nats/h of slope needed to
  cure the dosed bias would move the T-0 MAP by 1.5 × 10⁻⁵. **T-0 is blind to α-family
  perturbations**, so its perfection is not evidence that α is right. (Established here; this
  corrects an easy misreading of constraint (a).)

**(b) Must be linear in σ_z.** M4 **fails**, on both admissible branches:

- *σ_z-aware α branch:* forbidden outright — the correction is exactly 1 (§2.2). Even granting
  it, any σ-dependence of a normalised symmetric kernel entering a marginalisation is an even
  analytic function of σ (§2.3): admissible responses are σ⁰, σ², σ⁴. Linear is inadmissible.
- *σ_z-blind α-defect branch* (α simply wrong, for reasons unrelated to σ_z, with the σ_z-keying
  arising because the posterior gets softer as the dose rises): this predicts
  `bias = A′/C = A′ · post_sd²`. The measured widths give `post_sd ∝ σ_z^0.726`
  (0.00120 → 0.00298 over a 3.5× dose; and independently in the venue cells
  post_sd/σ_z = 0.1054 at σ̄ = 0.035 vs 0.1048 at σ̄ = 0.0418), hence a predicted
  `bias ∝ σ_z^1.45`, i.e. a ratio of 6.2 across the B1→B2 lever. Measured ratio **3.21**
  (0.01103 → 0.03542, SEs ~2×10⁻⁴). To reconcile, A′ would have to scale as **σ_z^(−0.52)** —
  a negative half-power of σ_z, inadmissible for any analytic even function of the kernel width.

**(c) Not misspecification.** Correctly not invoked. Generator and estimator share each
candidate's σ_k by construction (`venue_transfer.py:1393,1396` draw the scatter with the same
`sigma_pairs` the estimator receives at `_channel_terms_at_h(…, sig_z=…)`), and ζ is zero-mean
about the true z. M4 does not rely on misspecification, so (c) neither supports nor damages it.
Note however that (c) is what forces the exactness argument of §2.2: because the kernel is the
true, correctly normalised sampling density of ζ, `∫dζ p(ζ|z) = 1` holds *exactly*, not
approximately.

**Truncation sub-case (the strongest reformulation of M4).** The numerator integral is clipped
to the h-dependent GW window and to ±5σ_k (`venue_transfer.py:1127–1128`), and α does not
compensate for the clipped mass — a genuine unmatched normalisation. It is bounded and tiny:
the ±5σ_k clip loses ≤ 5.7 × 10⁻⁷ of the kernel, and the ±4σ_d window clip removes only the GW
tail beyond 4σ, ≤ 6.3 × 10⁻⁵ of the *integrand* mass. Both are exponentially small in the
window width and carry no σ_z-linear component. Not the mechanism.

---

## 5. Direct experiment: α deleted from the stored posteriors (RUN)

Because α enters `ln P(h)` as a single additive per-h function shared by every seed and cell,
**the whole α family is testable at zero simulation cost** by post-processing the already-stored
41-point `ln_post` vectors. Executed here: `log_alpha` was rebuilt from the campaign's own
config (`closed_loop_gfrac.build_context`, injection pool + CRB CSV as pinned in the chunk
JSONs), and every stored posterior was re-argmaxed with `+N·log_alpha` added back, i.e. with the
selection normalisation **entirely removed**.

| cell | σ_z | MAP as-run | MAP, α deleted | Δ | bias as-run | bias, α deleted |
|---|---|---|---|---|---|---|
| venue T-0 | 0 | 0.7300 | 0.7300 | +0.0000 | +0.0000 | +0.0000 |
| calgate B0 | 0 | 0.7300 | 0.7300 | +0.0000 | +0.0000 | +0.0000 |
| calgate B1 | 0.010 | 0.7407 | 0.7356 | −0.0051 | +0.0107 | **+0.0056** |
| calgate B2 | 0.035 | 0.7653 | 0.7465 | −0.0187 | +0.0353 | **+0.0165** |
| venue T-b | 0.035 | 0.7659 | 0.7471 | −0.0187 | +0.0359 | **+0.0171** |
| venue T-c | 0.0418 | 0.7672 | 0.7444 | −0.0229 | +0.0372 | **+0.0144** |

(Grid argmax on the 41-point grid, spacing 0.0065; means over 200–400 seeds per row.)

**Reading.** Deleting α does not delete the defect. A positive, σ_z-keyed displacement of
+0.0056 → +0.0165 survives (dose exponent 0.86, still ≈ linear), against per-seed MAP spreads
of ~0.005 and per-cell SEs of ~2 × 10⁻⁴. The σ_z **keying** — the thing that has to be
explained — lives in `Σ_i ln L_i(h)`, not in α. What α does contribute is a σ_z-blind
high-h pull (`−N ln α = +1.036 N ln h`) that *amplifies* an already-displaced numerator; that
is an amplification term, not the mechanism, and it is bit-identical between T-0 (zero bias)
and T-b (full bias).

**Supporting toy** (`scratchpad/m4_toy.py`, standalone, 8 seeds × 150 events × K = 8, the code's
numerator structure and ball construction reproduced, no production import): with the
normalisation removed the σ_z response is +0.0006 / +0.0039 / +0.0180 at σ_z = 0.011 / 0.035 /
0.070 — **superlinear, ≈ σ_z²**, exactly the even-order signature of §2.3, and *not* the
observed dose-linear +1×σ_z. The toy is not a reproduction of the venue (no selection, flat σ_d,
K = 8) and is quoted only as an illustration that kernel-smoothing families produce quadratic
dose response.

### Had M4 survived — what would have been the minimal change and the cheapest test

For the record, since the question was posed:

- **Minimal code-level change:** at `closed_loop_gfrac.py:379–380`, replace `s_phi` by its
  σ_z-convolution, `s̃_φ(z) = ∫ N(z'; z, σ̄) S̄_φ(z';h) dz'`, before the `np.trapezoid`. Two
  lines, read-only elsewhere. *(This is exactly the change that §2.2 proves to be a no-op up to
  O(σ̄²) curvature of S̄_φ — which is why it is worth stating: it makes the null falsifiable.)*
- **Cheapest decisive experiment:** none needed at simulation cost — the α-family is an additive
  per-h function, so the post-processing above (§5) is decisive and costs seconds. If a
  simulation confirmation were still wanted, 15 seeds of the T-b cell (`--n-seeds 15`,
  `n_events_cap` on) at ~3.8 CPU-h/seed ≈ 57 CPU-h suffices: per-seed bias +0.037 against a
  per-seed MAP spread of ~0.005 is a ~7σ single-seed signal.

---

## 6. Literature framing

Only statements verifiable in this repo are cited.

- `docs/LITERATURE_WARNINGS.md:84` (row "Gray et al. 2020, arXiv:1908.06050 — our partition-norm
  template"): its photo-z handling is *"an **unexercised** equation — validated at σ_z = 0
  ('ignore these crucial redshift uncertainties altogether'); under flat p_det the same-kernel
  denominator degenerates to a constant N"*. This is precisely M4's territory, and it says the
  opposite of M4: under flat p_det the kernel-aware denominator **degenerates**, i.e. it carries
  no σ_z-linear information to recover.
- `docs/LITERATURE_WARNINGS.md:85` (Gray et al. 2023, arXiv:2308.02281, GWcosmo §2.1.4): the
  comoving-volume LOS-prior h-dependence *"**cancels** between numerator and denominator"* —
  status `UNCHECKED` as a register row, but consistent with the exact h⁻³ cancellation derived
  independently in §3 item 1 here.
- `docs/BIAS_RESOLUTION_ATTEMPTS_REPORT.md:186–194`, the **p_det ≈ 1 degeneracy**, "confirmed by
  all four working extractions": *"when p_det is flat over the in-catalogue support,
  `∫ p_det p_cat → ∫ p_cat = const`, so the per-event/global-scalar denominator is H0-blind"*,
  and *"de-railing protection cannot come from the prior normalisation at all"*. The same
  document records that *"the empirical two-rail bracket (0.60 / 0.87, truth between)
  independently shows the defect is **not** in the numerator kernel"* — note that this earlier
  statement concerns the production rail, a different phenomenon from the venue-transfer
  σ_z-dosed displacement studied here; §5 above locates the σ_z **keying** in the numerator sum,
  which is not in conflict with it but should not be conflated.
- `docs/LITERATURE_WARNINGS.md:47` (H-a, Gair et al. 2023 arXiv:2212.08694 §2.3 after Eq. 30):
  the reduction to the per-event form requires *perfect* galaxy redshifts; status
  **VIOLATED — every venue**. This is the standing literature warrant that the defect lives in
  the per-event catalogue factorisation (the ζ-dependent latent structure), not in the scalar
  selection normalisation — consistent with the M4 refutation.

No citation here is drawn from outside the repository.

---

## 7. What this closes, and what it leaves open

**Closed.** M4 as posed, plus the entire α(h) family — both the "α should know σ_z" branch
(exactly zero) and the "α is wrong and σ_z only softens the posterior" branch (wrong exponent by
1.9×, requires σ_z^(−0.52)). Also closed: the numerator-truncation reformulation (bounded at
10⁻⁴, exponentially small), and the per-candidate `w_pop/Z_k` omission as *owner* (quadratic,
h-cancelling, and sign-wrong — though it is a genuine specification defect worth its own note).

**Open, and sharpened by this study.** The σ_z keying is dose-**linear** with a per-event
signature of ≈ +1 kernel width, it survives deletion of α, and the posterior width itself scales
as σ_z^0.73. A surviving mechanism must be linear in σ_z, which by §2.3 means it cannot act
through a symmetric smoothing alone: it needs a first-order structure at scale σ_z — a support
edge, an extremum/argmax operation, or an asymmetry between the host's kernel and the impostor
background's kernel inside the finite GW window. The candidate-density edge geometry
(ball window half-width in z vs σ_z, which are comparable at these doses) is the natural next
place to look; that is a different mechanism and belongs in its own study.

---

*Prepared 2026-08-13. Read-only study; no module, gate, ledger, or commit was touched.
Reproduction scripts used for §2.3/§4/§5 are in the session scratchpad
(`m4_toy.py`, `alpha_probe.py`) and import only `closed_loop_gfrac` for the α table.*
