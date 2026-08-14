# M5 — "equal-weight 1/K candidate prior over a smeared population"

Derivation note + toy-model adjudication. Analysis only; no production code was modified,
no physics gate opened, nothing committed.

**Target defect (established, not re-litigated):** the ball estimator returns
bias = +1 x sigma_z (h units), 0/1200 HPD coverage, width ~8.5x smaller than the
displacement. Readout: `results/venue_transfer_20260811/VENUE_TRANSFER_READOUT.md`.

---

## VERDICT

**M5 REFUTED as stated — but a modified form (M5', below) is the carrier.**

- The **prior half of M5 is exonerated.** The equal-weight `1/K` candidate prior is *not* the
  defect: replacing it with per-galaxy rate weights (the W1 arm), with oracle weights evaluated
  at the candidates' *true* redshifts, with the population prior `w_pop(z)` inside the
  marginalisation integral, or with a window-renormalised prior, changes the bias by
  **+1% to +30% — i.e. never attenuates, and three of the five repairs make it *worse***. Proven both
  algebraically (exact weight-invariance, §2) and numerically (§4, rows H/I/P/Q/R).
- The **"scattered out of the window" half of M5 is refuted quantitatively.** Removing the window
  truncation from the estimator entirely changes the bias by **-1%** (row B); the estimator's
  window is *h-independent in the natural variable* (§1), so it cannot carry an h-displacement.
  And with **zero population scatter** (candidates at their exact true z, estimator kernel still
  on) **76% of the bias survives** (row C) — the smeared *population* is not necessary.
- **M5' (survives, all three constraints):** the carrier is the **over-broad effective candidate
  measure the estimator's own sigma_z kernel builds**, times the marginalisation Jacobian
  `tau = D/D'`, evaluated against a *box-supported* true candidate density. The population
  scatter contributes ~24-32% of it (row D, noisy), the estimator's kernel ~76%. Sign **positive**, scaling
  **linear in sigma_z**, vanishing identically at sigma_z = 0 (different code branch).
  It is **not repairable by any reweighting of candidates.**

---

## 0. Code read (line numbers, `darksiren_emri/validation/venue_transfer.py` unless noted)

| element | where | what it does |
|---|---|---|
| ball placement | `draw_ball_pinned` 818-878 | window `d_obs(1 +- 4 sigma_d)` -> `z_lo/z_hi` on the **h_true** ladder (838-847); `n_imp = K_i - 1` pinned (852); impostor z by CDF inversion of `w_pop \| W_i` (849-860). `z_obs` returned = **TRUE** member z |
| sigma draw | `build_sigma_sampler` 651-675, `draw_member_sigma_z` 678-706 | rank-based z-decile pools from the pruned GLADE frame; decile chosen on the member's **true** z (1395 passes `ball.z_obs` while it is still the true z) |
| smearing | `_draw_seed_realization` 1393 (flat), 1396 (GLADE) | `z_obs <- z + sigma_k * N(0,1)`, zero-mean, per-candidate sigma shared with the estimator (constraint (c) holds) |
| estimator | `_channel_terms_at_h` 1061-1188 | per-candidate kernel 1139, GL-50 clip `[max(z_lo(h), z_obs-5s), min(z_hi(h), z_obs+5s)]` 1127-1128, h-dependent window 1109-1116, point path at sigma=0 1157-1165 |
| **1/K prior** | 1167-1168 | `L1 = bincount(ev, c1)/K` — flat weight, no `w_pop`, no selection factor in the numerator |
| penalty | 1174-1175 | dead event -> finite `-745` |
| measure note | `calibration_gate.py:662` | impostors are *sampled from* `w_pop\|W_i` (Slivnyak-Mecke) — **the equal weight already carries the rate measure** |

---

## 1. Exact reduction of the estimator (the algebra everything else uses)

Because `dist(z,h) = D(z)/h` exactly (h is a pure prefactor, `physical_relations.dist_vectorized`),
substitute `u = D(z)/(h d_obs)` in one candidate's term:

```
l_k(h) = INT dz N(z; z_k, sigma_k) N(D(z)/(h d_obs); 1, sigma_d)
       = INT du N(z(u); z_k, sigma_k) N(u; 1, sigma_d) * J,     J = h d_obs / D'(z(u))
```

Two exact consequences:

**(1a) The window is h-independent in u.** `z_lo(h) = Zof(d_obs(1-4 sigma_d) h)` maps to
`u = 1 - 4 sigma_d` for *every* h. The estimator's truncation is therefore the fixed interval
`|u - 1| < 4 sigma_d` — it removes the +-4 sigma tails of the GW factor and **cannot generate an
h-displacement**. (Prediction: deleting the truncation must not move the bias. Confirmed: row B,
-1.2%.) The window-edge / "leak" reading of M5 is dead on arrival.

**(1b) The per-candidate term collapses to a Gaussian times a Jacobian.** With
`zeta(h) = D^-1(h d_obs)` (the redshift the GW distance implies at trial h),
`tau = D/D'` at zeta, `z(u) ~= zeta + (u-1) tau`, and `J = D(zeta)/D'(zeta) = tau`:

```
l_k(h) ~= tau(zeta) * N( zeta(h) ; z_k , Sigma_k ),     Sigma_k^2 = sigma_k^2 + s^2,  s = sigma_d * tau
L_i(h)  = tau(zeta) * f_hat_i(zeta),  f_hat_i = (1/K) SUM_k N( . ; z_obs,k , Sigma_k )   (KDE, bandwidth Sigma)
ln P(h) = SUM_i [ ln tau(zeta_i(h)) + ln f_hat_i(zeta_i(h)) ] - N ln alpha(h)
```

At the registered scales this is not a perturbative regime: `sigma_d ~ 0.037` (median of the pinned
CRB rows), typical `z ~ 0.5`, so `s ~ 0.022`, window half-width `4s ~ 0.088`, dose `sigma_z = 0.035`.
The dose is **1.6x the GW width and 0.4x the window half-width**.

**Expectation of the estimator's density vs the truth.** `E f_hat = n_i (*) N(0,sigma) (*) N(0,Sigma)`
= `n_i (*) N(0, sqrt(2 sigma^2 + s^2))`, whereas the correct marginal likelihood is
`tau * (n_i (*) N(0, s))(zeta)`. The estimator therefore evaluates a density that is
**over-broad by exactly 2 sigma^2 of excess variance** — one sigma^2 from the population scatter
(M5's half) and one sigma^2 from the estimator's own kernel re-smearing the already-scattered
positions. This double-count is the object; **the kernel half dominates** (§4 row C).

---

## 2. The common-mode question (D11) — the answer is YES, and it is exact

Let every candidate of event i be displaced by a common `delta`: `z_k -> z_k + delta`. For **any**
h-independent weights `w_k` (equal, rate, oracle — anything):

```
L_i^w(h) = SUM_k w_k l_k / SUM_k w_k = tau(zeta) * SUM_k w_k N(zeta; z_k + delta, Sigma) / SUM w
         = tau(zeta) * f_hat_w(zeta - delta)
```

The shift sits *inside every kernel*, so it factors out of the convex combination **identically**
— not to leading order. Hence:

- the stationary point of `ln f_hat_w` moves by exactly `delta` regardless of `K` and of `w`;
- in h: `Delta h = h * delta / tau`, since `dzeta/dh = tau/h`. **K-independent, weight-independent.**
- `1/K` is an h-independent constant and drops out of `argmax_h` entirely.

**So D11's reading is correct: equal-weight dilution can never attenuate a displacement shared by
host and impostors, and neither can any reweighting.** The T-a -> T-b observation (K ~ 5 -> ~1216,
bias +0.0349 -> +0.0359) is exactly what this predicts.

One refinement the toy adds (§4 rows E-G): the empirical bias is *not* literally a fixed `delta` —
it is a **weight-invariant functional of the smeared pedestal**, which is K-invariant only in the
large-K limit. It *grows* with K (K=2: +0.0138; K=5: +0.0252; K=20: +0.0314; K=100: +0.0341) and
**saturates**, because the only K-sensitive ingredient is the *sharp host's* share `1/K` acting as
a pin (§4 rows on split dosing). Production K (median 84, nonempty mean 1216) is deep in
saturation — which is why T-a and T-b agree to 0.001. Adding candidates cannot attenuate;
*removing* them attenuates only by giving an (unavailable) exactly-known host more weight.

---

## 3. Sign and scaling of the residual — linear, positive, and why

**Sign (positive).** Both smooth factors multiplying the smeared pedestal rise with zeta, and
zeta rises with h:
`d ln tau/dz > 0` (tau = D/D' ~ z at low z) and `d ln n/dz > 0` (n ∝ dV_c/dz/(1+z)).
Removing the Jacobian `tau` by hand (row J) removes **32%** of the bias and leaves the sign
unchanged; the remainder is the density's own rise plus the numerator/`alpha(h)` balance.

**Scaling (linear, not quadratic).** The naive expansion gives a *quadratic* answer:
for a locally Gaussian peak of variance `V`, a constant drift `g = d ln tau/dz` displaces the mode by
`Delta z = V * g` with `V ~ 2 sigma^2 + s^2` -> `Delta h/h ~ sigma^2 (D'^2/D^2 - D''/D) ~ 3e-3` at the
registered dose, i.e. `Delta h ~ 0.0025` — **14x too small and quadratic**. That term exists and is
excluded by the data.

The true scaling comes from the fact that the un-smeared candidate density is **not a peak but a
box**: `n_i` is `w_pop` *restricted to the window* `|u-1| < 4 sigma_d`, i.e. a flat-topped support of
half-width `4s ~ 0.088` with hard edges. `ln` of a box has **zero curvature in the interior and all
of its curvature at the edges**. Against a monotone drift, the MAP does not sit at a curvature-
balanced interior point — it sits on the **upper soft flank**, whose position is
`~ 4s + kappa * sqrt(2 sigma^2 + s^2)`. Subtracting the sigma = 0 anchor (where the flank is at
`4s + kappa*s` *and* the exact host pins h anyway) leaves

```
Delta zeta  ~=  kappa * ( sqrt(2 sigma^2 + s^2) - s )   ->   kappa * sqrt(2) * sigma   for sigma >> s
Delta h     ~=  (h/tau) * Delta zeta            =>   R_dose = Delta h / sigma ~= kappa*sqrt(2)*h/tau = O(1)
```

**linear in sigma_z, positive, and O(1) in units of the dose** — which is the observed
`R_dose ~ 0.88-1.07`. The toy reproduces it over a **14x dose range** (§4): R_dose = 1.16, 0.85,
0.75, 0.72, 0.76, 0.88 for sigma_z = 0.005 ... 0.07. A sigma^2 mechanism would have swung R_dose by
14x across that ladder; it swings by 1.5x. **Quadratic is excluded; linear confirmed.**

---

## 4. Toy model — construction, validation, and the ablation table

A ~120-line standalone toy (listing in the appendix) mirroring the estimator exactly:
real fiducial `D(z)` spline; `d_L = D(z)/h`; `w_pop = (dV_c/dz)/(1+z)`; hard `d_L < 6 Gpc` horizon
with `alpha(h) = INT_0^{z_max(h)} w_pop dz`; per-event `sigma_d` resampled from the **pinned CRB CSV**
(`results/run_20260804_postfix/iiib/diagnostics/prepared_cramer_rao_bounds.csv`, 1590 rows,
median `sigma_d` = 0.0373); `+-4 sigma_d` window; `K-1` impostors from `w_pop|W`; GL-50; `+-5 sigma`
clip; `1/K`; `-745`; `- N ln alpha(h)`. N = 400 events, 5-6 seeds, h grid step 0.002 + parabolic
refinement. Per-seed scatter of the bias is small, so 5 seeds give SE ~ 0.001.

**Validation (this is what licenses the toy):** at sigma_z = 0 it returns bias
**+0.00093 +- 0.00042** — it reproduces the T-0 anchor. At the registered dose it returns
**R_dose = 0.72-0.95** against the instrument's **0.88-1.07**, with the same sign and the same
dose-linearity. The toy is a faithful bench for mechanism discrimination.

### 4.1 Dose ladder (K = 5, N = 400, 6 seeds)

| sigma_z | bias | R_dose |
|---|---|---|
| 0 | +0.00093 +- 0.00042 | — (anchor) |
| 0.005 | +0.00579 +- 0.00099 | 1.157 |
| 0.011 | +0.00934 +- 0.00109 | 0.849 |
| 0.020 | +0.01502 +- 0.00115 | 0.751 |
| 0.035 | +0.02519 +- 0.00101 | 0.720 |
| 0.050 | +0.03820 +- 0.00121 | 0.764 |
| 0.070 | +0.06140 +- 0.00182 | 0.877 |

### 4.2 Ablations at sigma_z = 0.035 (N = 400, 5 seeds)

| # | variant | bias | vs baseline | reading |
|---|---|---|---|---|
| A | **baseline (production form)** | +0.02518 +- 0.00121 | — | reference |
| B | window truncation deleted | +0.02489 +- 0.00219 | -1% | **edge/leak term is null** (predicted by §1a) |
| C | estimator kernel on, **population NOT scattered** | +0.01919 +- 0.00060 | **76%** | the smeared *population* is not necessary |
| D | population scattered, estimator uses **point** kernel | +0.00812 +- 0.00694 | 32% | M5's own half — minor and noisy |
| E | K = 2 | +0.01384 +- 0.00101 | 55% | sharp-host pin still partly effective |
| F | K = 20 | +0.03138 +- 0.00042 | 125% | pin diluted |
| G | K = 100 | +0.03408 +- 0.00040 | 135% | **saturated** (production regime) |
| H | **W1 weights** `v(z_obs)/(1+z_obs)` | +0.02565 +- 0.00108 | +2% | **no attenuation** |
| I | oracle weights at **true** z | +0.02538 +- 0.00114 | +1% | **no attenuation even with oracle z** |
| P | `w_pop(z)` prior **inside** the integral | +0.03217 +- 0.00125 | +28% | **worse** |
| Q | flat prior renormalised on the window | +0.03074 +- 0.00103 | +22% | worse |
| R | P + Q | +0.03282 +- 0.00104 | +30% | worse |
| J | Jacobian `tau` divided out | +0.01718 +- 0.00117 | -32% | tau carries ~1/3 of the drift |
| K | P + Q + J | +0.02064 +- 0.00110 | -18% | no combination repairs it |

### 4.3 Split dosing — which candidates carry the bias

Per-candidate sigma vector (already supported by the production estimator, 1139): dose only the
host, or only the impostors.

| K | both dosed | host dosed only | impostors dosed only |
|---|---|---|---|
| 5 | +0.02452 +- 0.00189 | +0.00457 +- 0.00197 | +0.00118 +- 0.00026 |
| 50 | +0.03340 +- 0.00030 | +0.00617 +- 0.00105 | **+0.02468 +- 0.00035** |

This is the decisive structural result:

- **The bias is carried by the dosed impostor pedestal** (K=50: 74% of the full effect with an
  exactly-known host).
- **A sharp host is a pin whose strength is its weight `1/K`.** At K=5 the sharp host suppresses
  the impostor-only bias to +0.001; at K=50 it cannot. Production K (median 84, mean 1216) is far
  past the point where the pin matters — consistent with T-a/T-b agreeing to 0.001.
- **Dosing the host alone is nearly harmless** (+0.005) because an *un-smeared* pedestal carries no
  tau tilt (point evaluations have no Jacobian). The effect is super-additive: it needs the
  pedestal smeared **and** the pin gone.

---

## 5. The W1 question — CLOSED, on paper

**Would per-galaxy rate weights `R_eff(M_g)/(1+z_g)` break the symmetry and attenuate the bias?
NO.** Three independent arguments, all pointing the same way:

1. **Algebraic (exact).** §2: a displacement shared by the candidate kernels factors out of any
   h-independent weighting identically. Weights cannot attenuate what they commute with.
2. **Numerical (matched to the production form).** Rate-shaped weights change the bias by **+2%**
   (row H); weights evaluated at the candidates' **true** redshifts — an oracle W1 that no real
   analysis could build — change it by **+1%** (row I). The stronger repair (population prior
   *inside* the marginalisation, row P) makes it **28% worse**.
3. **Structural — W1 would have double-counted.** The ball's impostors are *drawn from*
   `w_pop | W_i` (`calibration_gate.py:662`, Slivnyak-Mecke). The equal `1/K` weight over a
   `w_pop`-distributed sample **already is** the rate measure — that is precisely the registered
   bracketing argument (VT-D2). Applying `R_eff/(1+z)` on top would apply the measure twice, which
   is exactly what row P measures (+28%).

Caveat to the third point, stated so the closure is honest: in a *real* GLADE ball the candidates
are **not** drawn from `w_pop`, so a rate weight there is not a double count — it is a genuine
correction to the host-identity prior. But arguments 1 and 2 do not depend on that: the weights are
h-independent, and the displacement is weight-invariant. **W1 would have measured a null.** The
dropped arm loses nothing; the question is answered.

---

## 6. Self-falsification against the three hard constraints

| constraint | M5 as stated | M5' (kernel-smeared pedestal x tau, against a box support) |
|---|---|---|
| **(a) vanishes identically at sigma_z = 0** | passes | **passes** — toy anchor +0.00093 +- 0.00042 (cf. T-0 exact); and the sigma = 0 code path is a *different branch* (point evaluation, 1157-1165: no kernel, no Jacobian, no smearing), so the term is structurally absent, not merely small |
| **(b) linear in sigma_z** | passes | **passes** — R_dose 1.16 -> 0.88 across a 14x dose ladder (§4.1); a sigma^2 term is present but is 14x too small at the registered dose and is excluded by both instrument and toy |
| **(c) not misspecification** | passes | **passes** — every baseline/ablation row except C and D is matched-model (generator and estimator share sigma_k). C and D are deliberately mismatched *diagnostics*, labelled as such |

**Where M5 as stated fails** (these are not constraint failures, they are causal-attribution failures):

- **Row C kills the central claim.** With the population *not* scattered at all — no
  `n (*) N(0,sigma)`, no candidates scattered out of the window, nothing for a "smeared population"
  to mean — **76% of the bias remains**. M5 names an ingredient that is neither necessary nor
  dominant.
- **Row B kills the edge sub-claim.** "Candidates scattered OUT of the true window are still
  counted": deleting the window from the estimator changes the bias by -1%. §1a explains why —
  the window is h-independent in `u`.
- **Rows H/I/P/Q/R kill the prior sub-claim.** The "equal-weight `1/K` prior" is the named
  mechanism; no prior repair attenuates, three make it worse. A mechanism whose repair does not
  repair is not the mechanism.

**What survives, restated as M5':** the estimator's per-candidate sigma_z kernel builds an effective
candidate measure over-broad by `2 sigma^2` (half from the population scatter, half from re-smearing),
attaches the marginalisation Jacobian `tau = D/D'`, and evaluates it against a candidate density
whose true support is a **hard box** (the +-4 sigma_d window). The box's flat log-interior means the
MAP is set by the monotone drift on the smeared upper flank, at linear order in the smearing width.
The host's exact redshift is the only thing that pins this at sigma_z = 0, and its pinning power is
`1/K` — negligible at production multiplicity.

---

## 7. Cheapest decisive experiments

The toy in the appendix (`~4 CPU-minutes per configuration`, 5 seeds, N = 400) **already is** the
cheap experiment and has already discriminated the hypotheses. To transfer the discrimination to
the registered instrument, per-seed bias is ~7 sigma, so **N = 15 seeds per cell** suffices
(SE ~ 0.0006 against a 0.035 effect).

**E1 — split-dose cell (highest value, zero estimator change).** The estimator already consumes a
per-candidate `sig_z` vector (1139, `_channel_terms_at_h`). Only the *generator-side* sigma
assignment in `_draw_seed_realization` (1393/1396) needs a variant that dosises host-only or
impostors-only. Two cells x 15 seeds at production K.
*Prediction (from §4.3):* impostors-only ~ full bias (>= 0.030); host-only ~ +0.006. If instead
host-only reproduces the bias at production K, M5' is wrong and the mechanism is host-side.

**E2 — kernel-only cell (misspecified diagnostic, 15 seeds).** Generator applies no scatter;
estimator still runs with sigma_k = 0.035. *Prediction:* ~76% of the dosed bias. This isolates the
estimator kernel from the population scatter in the real venue. Must be reported as a deliberately
mismatched diagnostic, never as a calibration statement.

**E3 — dose ladder at fixed K (15 seeds x 3 doses: 0.005 / 0.020 / 0.070).** Extends the registered
0.011/0.035/GLADE ladder to 14x and pins linear-vs-quadratic in the instrument itself, where the
toy already predicts R_dose ~ 1.2 / 0.75 / 0.88.

**Not worth running:** W1 (answered null here, §5), any window/clip variation (row B), any prior
renormalisation (rows P/Q/R).

**Repair direction implied (not proposed for implementation here):** nothing that reweights or
re-priors candidates will work. The defect is that a *marginalised* candidate term
(`tau x smeared density`) is scored against a normalisation `alpha(h)` built for the *un-marginalised*
measure, on a box-supported candidate set. A repair has to make numerator and `alpha(h)` the same
functional of h at finite sigma_z — i.e. a normalisation derivation, which is an author-gated
physics change and explicitly out of scope for this note.

---

## Appendix — toy listing (as run)

Scratch location this session:
`.../scratchpad/m5_toy.py` + `run1..run5.py`. Core (elided for the ablation switches):

```python
# d_L(z,h) = D(z)/h exactly; D from physical_relations.dist_vectorized(h=1) on a 40k spline
# v(z) = comoving_volume_element(z,h=1)/(1+z);  alpha(h) = Vcdf(Zof(h*DMAX)) / h^3
# sigma_d resampled from the pinned CRB CSV; SIGMA_WINDOW=4, KERN_WINDOW=5, GL-50

def draw(seed, n_ev, K, sigma_z):
    rng = np.random.default_rng(seed)
    z_true = Vinv(rng.random(n_ev) * Vcdf(Zof(H_TRUE*DMAX)))          # w_pop, horizon-cut
    sig_d  = SIG_D_POOL[rng.integers(0, SIG_D_POOL.size, n_ev)]
    d_obs  = (D(z_true)/H_TRUE) * (1 + sig_d*rng.standard_normal(n_ev))
    z_lo, z_hi = Zof(d_obs*(1-4*sig_d)*H_TRUE), Zof(d_obs*(1+4*sig_d)*H_TRUE)
    ev = np.repeat(np.arange(n_ev), K-1)                               # pinned multiplicity
    u  = Vcdf(z_lo)[ev] + (Vcdf(z_hi)-Vcdf(z_lo))[ev]*rng.random(ev.size)
    z_cand = np.concatenate([z_true, Vinv(u)])                         # host + impostors ~ w_pop|W
    z_obs  = z_cand + sigma_z*rng.standard_normal(z_cand.size)         # zero-mean, matched model
    ...

def lnpost(R, h_grid, sz, truncate=True, weights=None, point_kernel=False):
    for j, h in enumerate(h_grid):
        z_lo = Zof(d_obs_p*(1-4*sig_p)*h); z_hi = Zof(d_obs_p*(1+4*sig_p)*h)
        a = np.maximum(z_lo, z_obs-5*sz); b = np.minimum(z_hi, z_obs+5*sz)   # `truncate=False` drops z_lo/z_hi
        zn   = mid[:,None] + half[:,None]*gl_x[None,:]
        frac = (D(zn)/h)/d_obs_p[:,None]
        integ = norm.pdf(zn, z_obs[:,None], sz) * norm.pdf(frac, 1.0, sig_p[:,None])
        c = np.where(b>a, half*(integ @ gl_w), 0.0)
        L = np.bincount(ev, weights=wk*c, minlength=n_ev) / np.bincount(ev, weights=wk, minlength=n_ev)
        out[j] = np.sum(np.where(L>0, np.log(L), -745.0)) - n_ev*log_alpha(h)
```

**Toy caveats (disclosed):** 1D channel only (no `g_i` completion / BH-mass channel); flat
per-candidate sigma_z rather than the GLADE decile sampler; K uniform per cell rather than the real
K distribution (median 84, max 245364); simplified hard-horizon `alpha(h)`; N = 400 events vs 982.
It reproduces the anchor (sigma = 0), the sign, the magnitude (R_dose 0.72-0.95 vs 0.88-1.07), and
the dose-linearity — sufficient for mechanism discrimination, not a substitute for a registered cell.

---

## Addendum (2026-08-14) — toy unfaithful at production K (ledger row #102)

The commission re-executed this note's toy (recovered from session scratchpad; registered K=50 impostor-only value +0.02468 reproduced) unchanged at K=84 and K=1216: the impostor-only prediction **grows** to +0.0279 and **+0.0341**, against the instrument's exactly 0.000000 (MEI, 15/15 seeds). The chair independently re-drove it (+0.0317±0.0007 at K=50, +0.0339±0.0006 at K=1216, protocol variant kerneling the exact host). The K-saturation account is therefore **inverted at production K** and abort (d) was ruled met in substance. Consequence: every sub-closure of this note that consumes toy output (rows B/C/D/H/I/P/Q/R) and the W1 toy leg are **NOT ESTABLISHED** pending re-derivation on a faithful instrument-side ablation. The refutation of M5′-as-registered by DS-M5 on the instrument is unaffected (it is a raw measurement, not a toy result). Toy source now committed under `toys/`.
