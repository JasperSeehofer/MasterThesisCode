# `/physics-change` INTAKE DOSSIER — the σ_z displacement of the ball dark-siren estimator

**Prepared 2026-08-14.** Scope: assemble, in one place, the two things the `/physics-change` gate
can be given today — **the old formula, exactly** and **the constraint set any candidate new
formula must satisfy** — and state plainly which slots of the gate package are still empty.

> ## THIS DOCUMENT PROPOSES NO REPAIR.
> No corrected formula is named here, no candidate mechanism is adopted, and no term is nominated
> as the owner of the defect. The 2-D dose scan's registered readout bars proposing a repair
> (`PREREGISTRATION_2D_DOSE_SCAN.md` §6 item 7, §7 branch 5), and both registered shapes (H-INT,
> H-THRESH) are **refuted** (ledger row #101). What follows is an inventory of what is established
> and how narrow the remaining space is. **The "new formula" slot of the gate package is empty and
> stays empty until the author fills it.**

**Status of this file.** New, additive, analysis-only. It edits no registered document, no
pre-registration, no ledger, no paper, and no production module. It is not a readout, not a verdict,
and not an author ruling.

**Provenance of everything quoted below** (nothing here is new measurement):

| source | what it supplies |
|---|---|
| `results/venue_transfer_20260811/` (prereg `e77eecad`, readout `d45fbf15`) | the campaign: 1,400 seeds, decision cell T-c(0.730), the ladder |
| `BIAS_HISTORY_LEDGER.md` rows **#99 / #100 / #101** | the three author-ratified verdicts of thread 17 |
| `PREREGISTRATION_MECHANISM_ISOLATION.md` §7 | the closure register (M1/M3/M4/M5/M5′/W1, the parity argument) |
| `M1_…md`, `M3_…md`, `M4_…md`, `M5_…md` | the four closure notes, with their toys and arithmetic |
| `AMENDMENT_A1_VM1_NULL_AT_N100.md` + `A1_READOUT.md` | the null-arm anchor at N = 100 (A1-PASS) |
| `PREREGISTRATION_2D_DOSE_SCAN.md` + `SCAN_READOUT.md` + `score_2d_scan_output.json` | the 16-cell surface, 325 seeds |
| `VM5_GOLDEN_20260814.md` | the V-M5 values-golden artifact (closes D-A1-2) |
| instrument commits `3aedbe55` (code), `5b0bd17a` (data), `73141160` (preregs), `94c0480a` (V-M5 run) | code/data identity |

---

## 1. THE OLD FORMULA, EXACTLY

This is the one slot of the five-item gate package (`.claude/rules/physics-validation.md`;
`.claude/skills/physics-change/SKILL.md`) that can be filled today. It is filled completely.

### 1.1 Where it lives

The object under investigation is the **certified mirror**, not the production `BayesianStatistics`
path (`PREREGISTRATION_2D_DOSE_SCAN.md` §6 item 4, carried from the venue prereg §9 item 1 —
registered NOT-EVALUABLE):

| element | file:line |
|---|---|
| per-h estimator body | `darksiren_emri/validation/venue_transfer.py:1182-1301` (`_channel_terms_at_h`) |
| per-h driver / grid loop | `venue_transfer.py:1101-1158` (`log_channel_posteriors_ball_sigma_vector`) |
| bit-identical h-grain twin | `venue_transfer.py:1319-1376` (`…_hgrain`) |
| α(h) construction | `darksiren_emri/validation/closed_loop_gfrac.py:374-383` |
| population measure `w_pop` | `closed_loop_gfrac.py:285-299` |
| 2-D completion factor `g_i` | `venue_transfer.py:1030-1098` (`_g_ball_capped`) → `bayesian_inference/bayesian_statistics.py:2012` (`completion_mass_factor_g`) |
| dose application (generator side) | `venue_transfer.py:1457-1508` (`_apply_dose_mask`) |

### 1.2 The log-posterior as implemented

For each h on the grid, for channel c ∈ {1D, 2D}:

```
ln P_c(h)  =  Σ_{i=1..N}  ln L_i^{(c)}(h)   −   N · ln α(h)                    [1297-1298]

L_i^{(c)}(h)  =  (1 / K_i) · Σ_{k ∈ ball(i)}  c_{c,ik}(h)                      [1288-1289]
```

with, per candidate k of event i,

```
                ⌠ b_ik(h)
c_{1,ik}(h)  =  ⎮        dz  N(z ; ζ_k , σ_k) · N( d_L(z;h)/d^obs_i ; 1 , σ_dL,i )       [1260, 1259, 1262]
                ⌡ a_ik(h)

c_{2,ik}(h)  =  same integrand × g_i(z;h)                                                [1263-1266]
```

evaluated by a **50-node Gauss–Legendre** rule affine-mapped onto `[a_ik(h), b_ik(h)]`
(`half·(integ @ w_gl)`, nodes from `roots_legendre(n_quad)` at `closed_loop_gfrac.py:388`,
`n_quad = _HOST_QUAD_N = 50`, `bayesian_statistics.py:392`).

**σ_k = 0 takes a different code branch** — point evaluation, no quadrature, no kernel, no Jacobian
(`venue_transfer.py:1269-1286`):

```
c_{1,ik}(h)  =  N( d_L(ζ_k;h)/d^obs_i ; 1 , σ_dL,i ) · 1[ z_lo,i(h) ≤ ζ_k ≤ z_hi,i(h) ]
c_{2,ik}(h)  =  same × g_i(ζ_k;h)
```

This branch is why constraint **C11** below is structurally satisfied by every σ_z-keyed candidate
and is therefore a weak discriminator.

### 1.3 Every symbol, with provenance

| symbol | meaning | provenance |
|---|---|---|
| `h` | dimensionless Hubble parameter; `d_L(z,h) = D(z)/h` **exactly** (h is a pure prefactor) | `physical_relations.dist_vectorized`; used at `venue_transfer.py:1255` |
| `h`-grid | canonical **41 points**, non-uniform: 0.01 spacing on [0.60, 0.65] and [0.80, 0.86], **0.005 on the core [0.65, 0.80]** | `closed_loop_gfrac.CANONICAL_H_GRID` (`:186`); read back from every cell JSON `config.h_grid` |
| `N` | number of events, **fixed** at 982 in the venue (pinned, nonempty-ball) | `venue_transfer.py:1216`, `1297-1298`; pin ΣK = 1,193,703 |
| `K_i` | candidate multiplicity of event i, pinned to the real frozeng per-galaxy emit (median 6, max 245,364, ΣK 1,193,703) | `draw_ball_pinned`; V-D1/V-D3 pins |
| `ζ_k` (code `z_obs`) | the candidate's **observed** (σ_z-scattered) redshift | `venue_transfer.py:1220`; scattering at `:1506-1508` |
| `σ_k` (code `sig_z`/`sigma_pairs`) | that candidate's own photo-z σ_z, drawn z-decile-matched from the pruned GLADE frame | `draw_member_sigma_z`; realized mean 0.041813 at full dose |
| `d^obs_i`, `σ_dL,i` | the event's GW luminosity distance and its fractional Cramér–Rao error (median σ_dL/d_L = 0.0373) | `universe.d_L_obs`, `universe.sigma_dL`; CRB CSV md5 `9a1f2a14384a9281c97ca3be312ddaab` |
| `g_i(z;h)` | 2-D completion-leg mass density — Gauss–Hermite contraction of φ against the conditional Gaussian of the (d_L, M_z) 2×2 block | `bayesian_statistics.py:2012-2047`; adaptive GH (Route 1, ledger 2026-08-12) |
| `α(h)` | the shared selection normalisation, one scalar per h | `closed_loop_gfrac.py:374-383` |
| `w_pop(z;h)` | `(dV_c/dz)/(1+z)`, per steradian | `closed_loop_gfrac.py:285-299` |
| `S̄_φ(z;h)` | φ-marginal detection survival of the injection pool | `bayesian_statistics.py:1859-1941` |

### 1.4 The integration limits, exactly

```
z_hi,i(h) = interp( d^obs_i · (1 + 4·σ_dL,i) , d_L_nodes[h] , z_tab[h] )      [1231]
z_lo,i(h) = interp( d^obs_i · (1 − 4·σ_dL,i) , d_L_nodes[h] , z_tab[h] )      [1232]
z_lo ← max(z_lo, 1e-6) ;  z_hi ← min(z_hi, z_tab[-1] = z_max(h))              [1233-1234]

a_ik(h) = max( z_lo,i(h) , ζ_k − 5·σ_k )                                       [1248]
b_ik(h) = min( z_hi,i(h) , ζ_k + 5·σ_k )                                       [1249]
valid   = b > a ;  contribution is exactly 0 where not valid                   [1250, 1267-1268]
```

- `_SIGMA_WINDOW = 4.0` — `closed_loop_gfrac.py:155`
- `_IMPOSTOR_KERNEL_WINDOW = 5.0` — `calibration_gate.py:215`
- The `z_of_dl_tables[k]` ladder is built **per h**, so both outer edges move with h
  (`closed_loop_gfrac.py:344-351`).
- **The kernel is NOT renormalised over `[a,b]`.** There is no division by
  `Φ((b−ζ_k)/σ_k) − Φ((a−ζ_k)/σ_k)` anywhere between lines 1248 and 1266 — confirmed in
  `M3_truncation_window.md` §1. The retained kernel mass is therefore a function of h.
- **Exact algebraic consequence** (`M5_smeared_candidate_prior.md` §1a): under the substitution
  `u = D(z)/(h·d^obs)` the outer window maps to the *fixed* interval `|u − 1| < 4σ_dL` **for every
  h**, so the truncation is h-independent in that variable.

### 1.5 The α(h) normalisation

```python
# closed_loop_gfrac.py:374-383
s_phi_table = precompute_phi_marginal_survival(h_grid, detection)
for i, h in enumerate(h_grid):
    z_grid, s_phi = s_phi_table[h]
    alpha = float(np.trapezoid(_w_pop(z_grid, h) * s_phi, z_grid))   # over [1e-6, z_max(h)]
    log_alpha[i] = math.log(alpha)
```

- **α(h) integrates:** the comoving population measure × GW detection survival, over z, per h.
- **α(h) omits:** σ_z, ζ_k, ball geometry, K_i, candidate identity, and the photo-z kernel entirely.
  It is one scalar per h shared by all 982 events, both channels, and every seed.
- **Measured shape** (`M4_alpha_sigma_blindness.md` §1): `d ln α / d ln h = −1.0358`, power-law to
  0.2 % (residual rms 0.0022 nats). So `−N ln α = +1.036·N·ln h` — a monotone pull toward **high** h
  of ≈ 1,393 nats per unit h at N = 982.
- **This blindness is CORRECT, not a defect** (M4 verdict): the σ_z-aware form
  `α_σ(h) = ∫dz w_pop S̄_φ ∫dζ N(ζ;z,σ)` has inner integral **identically 1**, because the code's
  kernel is a properly normalised density in the datum and ball membership is decided on **true** z
  (`calibration_gate.py:677-702`, `venue_transfer.py:845-868`) with σ_z applied afterwards.

### 1.6 The −745 convention

```python
# venue_transfer.py:1293-1298 ; cg._LN_ZERO_EVENT = -745.0  (calibration_gate.py:222)
ok1   = (L1 > 0.0) & np.isfinite(L1)
lnL1  = np.where(ok1, np.log(np.where(ok1, L1, 1.0)), cg._LN_ZERO_EVENT)
ln1_k = float(np.sum(lnL1)) - float(n) * gctx.cl_ctx.log_alpha[k]
```

`−745` is ln of the smallest positive double: a **finite, JSON-safe stand-in for −∞**. An event
whose likelihood vanishes at some h excludes that h by a finite penalty rather than a NaN. `N` in
the α term is **fixed** (the gate's "divergence-10" convention verbatim) and does *not* drop dead
events. In the entire scan and every arm: **0 non-finite `ln_post` values** across 325 + 130 seeds,
and zero rails — so the −745 branch never fired in the measurements quoted here.

### 1.7 The 1D / 2D channel distinction

- **1D** (`ln1`, the registered headline channel): observable set `cov_obs = cov_4d[:3,:3]`; the
  M-integral collapses; the completion numerator is **unmultiplied** — inserting `g` there would be
  a double count (`bayesian_statistics.py:2042-2046`, gate (iv)).
- **2D** (`ln2`): the identical integrand multiplied by `g_i(z;h)` **inside** the z-quadrature (the
  z-dependence of `μ_cond` and of the `1/(1+z)` mass lift is not separable).
- Both channels subtract the **same** `N ln α(h)`.
- `ln_gfrac` = `Σ_i ln(L2/L1)` over both-finite events; its grid slope at truth is the reported
  `sum_dlog_gfrac_dh` (`venue_transfer.py:1161-1179`).
- **Empirically the two channels never split** on this defect: 2D tracks 1D in all 16 scan cells and
  all four arms, running **+0.000333 … +0.002333 above** 1D (gap growing with total dose, always
  below one core h-grid step), with **identical classifications at every registered decision point**
  (`SCAN_READOUT.md` §2.2).

### 1.8 The dose knob (an instrument switch, not part of the production formula)

```python
# venue_transfer.py:1493-1508
scale        = np.where(host_mask, s_host, s_imp)
scaled_sigma = sigma_pairs * scale
return (scaled_sigma, z_obs + scaled_sigma * noise)
```

Both the scatter and the kernel width are scaled together — **matched model by construction**, which
is what makes every measurement below a *handling* defect and not a misspecification artefact
(parent constraint (c)). `(1,1)/(1,0)/(0,1)/(0,0)` reduce exactly to the registered arms and to the
σ = 0 anchor.

---

## 2. THE ESTABLISHED PHENOMENOLOGY

### 2.1 The decision cell (ledger row #99, TRANSFER-CONFIRMED, author-ratified 2026-08-13)

Venue-transfer campaign, prereg `e77eecad`, instrument `2ece8801`, readout `d45fbf15`, 49 chunks /
1,400 seeds. Cell T-c(0.730), N = 400, **1D**:

| statistic | value | registered band |
|---|---|---|
| MAP bias | **+0.037237 ± 0.000230** | DEFECT edge 0.030 |
| HPD 50/68/90 coverage | **0.000 / 0.000 / 0.000** | 0.870–0.930 at 90 % |
| PIT–KS D | **1.000** (saturated) | PASS ≤ 0.0679 |
| `post_sd` median | 0.004376 ⇒ displaced by **8.5×** its own claimed width | — |
| σ̄_pairs | 0.041775 ⇒ **R_dose = 0.891** | [0.75, 1.25] |
| rails | 0.000 / 0.000 | ≤ 0.02 |
| 2D alongside | bias +0.039713, R_dose 0.951 | — |
| T-0 anchor (σ_z = 0) | **all 200 seeds argmax exactly on truth**, rails 0 | — |

Adversarial adjudication: **CONFIRMED**, every scored statistic reproduced from the raw 41-point
`ln_post` vectors by independent implementations to ≤ 5.33e-15.

### 2.2 The displacement is +1 × σ_z, and linear in dose

| dose | cell | bias | R_dose = bias/σ̄_z |
|---|---|---|---|
| σ_z = 0.010 | calgate v2 B1 | +0.01103 | **1.103** |
| σ_z = 0.011 | v2 B1 (venue framing) | — | **1.0688** |
| σ_z = 0.035 | calgate v2 B2 | +0.03542 / +0.0353 | **1.012 / 1.0075** |
| σ_z = 0.035 | venue T-b | +0.0359 | — |
| GLADE mix σ̄ ≈ 0.0418 | venue T-c | +0.037237 | **0.877–0.913** (0.891 at the decision cell) |

Fitted exponent **bias ∝ σ_z^0.93**; `post_sd ∝ σ_z^0.726`
(`M4_alpha_sigma_blindness.md` §2.3, §4b). The R_dose *drift* 1.069 → 1.008 → 0.877–0.913 is fitted
by `bias(σ) = aσ + bσ²` with **a ≈ +1.15, b ≈ −5.29** (`M1_missing_volume_prior.md` §4b) — a
dominant positive **linear** driver plus a negative quadratic one. **a ≈ +1.15 is the quantity every
surviving candidate must supply.**

### 2.3 The venue-transfer ladder — no killing axis

`v2 +0.0353 → T-a +0.0349 → T-b +0.0359 → T-c +0.0372` (ledger row #99, DS-VT5). Real events, real
multiplicity and real heterogeneous GLADE σ_z each left the collapse intact. T-a → T-b moves K from
Poisson λ = 4 (≈ 5) to the real distribution (ΣK = 1,193,703, nonempty mean ≈ 1,216) and changes the
bias by **0.001**.

### 2.4 The null anchor at N = 100 (ledger row #100, A1-PASS)

`MN0X`, 100 seeds, `dose_target="all"`: 1D mean bias **+0.037250 ± 0.000494** against the campaign
reference +0.037237 — **|Δ| = 0.000013**, 153.8× inside the unchanged ±0.002 window, 0.024σ. 2D
+0.039750 vs +0.039713. `bias/post_sd` 8.49 / 9.02, coverage 0/0/0, PIT–KS 1.000, rails 0 — the null
arm reproduces the campaign's **entire signature**, not merely its mean. A1-DET: 15/15 shared seeds,
44 fields, **max relative deviation 0.0**, MAPs exactly equal, across the `e83ed0b9 → 3aedbe55`
refactor.

**Recorded, not repaired:** the N = 15 `MN0` arm FAILED V-M1 as written (|Δ| = 0.002570 against
±0.002, a 1.63σ deviation on a window tighter than the statistic it gates). That failure stands on
the record; the ±0.002 window was **not** widened.

### 2.5 The 2-D dose scan surface (ledger row #101, BRANCH 2 fired, meaning barred)

16 cells, 325 seeds, `f ∈ {0, 0.25, 0.5, 1.0}` of each candidate's own GLADE σ_z, host axis × impostor axis.
**1D bias surface** (`SCAN_READOUT.md` §2.1):

| f_h \ f_i | 0.0 | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|
| **0.0** | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| **0.25** | +0.004667 | +0.012667 | +0.012000 | +0.014000 |
| **0.5** | +0.005333 | +0.019000 | +0.016000 | **+0.023650** (N = 100) |
| **1.0** | +0.006000 | +0.022000 | +0.023333 | +0.039667 |

Registered scoring: DS-D2 **NON-ADDITIVE** at S33 (D = +0.033667, **23.4σ**) and at all nine interior
cells (≥ 10σ); DS-D3 **SHAPE-INTERACTION** at S23 (+28.2 realized SE above the boundary); DS-D4
**PIN-BINARY**; DS-D5 **SUPER-LINEAR** at S31 (+10.9 SE); DS-D6 S33 in band (0.9487). Validity: 0 of
4 SCAN-CONFOUNDED members fired; 16/16 dosing checks in tolerance; zero rails; zero non-finite.

**Both registered shapes are refuted by the scan's own registered statistics:**
- **H-INT** (strictly bilinear `D = I·f_h·f_i`): b(S23) sits **+10.33σ** above H-INT's own point
  prediction 0.017333 **using the registered SE** (+14.64σ realized); bilinear residuals positive at
  all nine evaluable cells and > 3σ at S22 (+3.76), S31 (+7.64), S23 (+5.47).
- **H-THRESH**: refuted at **17.96σ** (S13) and **50.18σ** (S23); row f_h = 0.25 — less than half the
  registered threshold f* = 0.5262 — already carries +0.014000.

**The positive structural finding: GATE × AMPLIFIER.**
- **Host = absolute gate.** The entire f_host = 0 row is **exactly +0.000000** at every impostor dose
  including full dose: 60/60 seeds, per-seed sd exactly 0, every posterior on a **single grid point**,
  `post_sd` identically 0. Not attenuated — annihilated.
- **Impostor sea = graded amplifier.** Removing it leaves **+0.0047 … +0.0060**, ≈ 15 % of the effect,
  and essentially flat in the host dose (only the first quarter-step is RESOLVED, 14.0σ).

---

## 3. THE CONSTRAINT SET — the core deliverable

**How to use this table.** A future candidate mechanism (or repair) is checked **mechanically**
against every row. A candidate that fails any row is refuted by an already-committed measurement and
needs no new run. A candidate that passes every row is *not thereby correct* — it is merely still
admissible, and then owes the gate its derivation, dimensional analysis, limiting cases and
regression test (§4).

Strength column: **EXACT** = an algebraic identity or an exactly-zero measurement, no statistical
tolerance involved; **nσ** = the registered separation; **BOUND** = an analytic ceiling.

| # | The candidate MUST … | Establishing measurement | Strength | Source |
|---|---|---|---|---|
| **C1** | **vanish EXACTLY when the host redshift is exact, at every impostor dose** — an exact zero with a degenerate single-grid-point posterior, not a small residual | S00/S01/S02/S03: bias +0.000000, per-seed sd 0.000000, 1 distinct MAP, `post_sd` 0.000000, **60/60 seeds**, impostor dose spanning 0 → full GLADE (σ̄ = 0.0417). Replicates MEI at fresh seeds (exact equality). | **EXACT** | `SCAN_READOUT.md` §1.7, §4.4; ledger #101 |
| **C2** | **be LINEAR in σ_z** — hence **cannot be a symmetric smoothing of any kind** | Parity argument: Gaussian convolution is `exp(σ²∂²/2)`, an expansion in **even** powers of σ, so every kernel-mismatch story is O(σ²) and predicts `R_dose ∝ σ_z`, i.e. a **3.5×** change across the B1→B2 lever. **Measured ratio 0.92.** Fitted exponent 0.93. | **~50σ** (per-cell SE ≈ 2e-4 on a factor-3.8 miss) | `M4…md` §2.3, §4b; parent §7 |
| **C2′** | (sharpened) be **more** than first-order: the f_i response is **steeper than linear at small dose and shallower at large dose** | f_h = 1 row slope change: m2 − m1 = **−9.29σ**, m3 − m2 = **+5.03σ**; second difference −7.07σ; DS-D5 SUPER-LINEAR at S31 (+10.9 SE, ≥ 8σ self-anchored) | **9.3σ / 5.0σ** | `SCAN_READOUT.md` §4.2, §3.4 |
| **C3** | **survive TOTAL DELETION of α(h)**, still producing **≈ +0.0165 at σ_z = 0.035** | Re-argmax of 2,400 stored posteriors with `log_alpha` added back: bias +0.0353 → **+0.0165** (σ_z = 0.035) and +0.0107 → **+0.0056** (σ_z = 0.010), still ≈ linear (dose exponent 0.86). α is a σ_z-**blind** amplifier (`−N ln α = +1.036 N ln h`), not the key. | direct, on stored data; per-cell SE ≈ 2e-4 | `M4…md` §5; parent §7 |
| **C4** | **NOT be reachable by reweighting the candidates** — in any h-independent scheme | Toy at σ_z = 0.035 vs baseline +0.02518: W1 rate weights **+2 %**, oracle weights at true z **+1 %**, `w_pop` inside the integral **+28 %**, window renormalisation **+22 %**, both **+30 %**. **None attenuated.** Plus the exact algebraic reason: a displacement shared by all candidate kernels factors out of *any* h-independent convex combination, so the stationary point moves by exactly δ for any K and any weights. Plus the structural double-count: ball impostors are *sampled from* `w_pop\|W` (`calibration_gate.py:662`), so `1/K` **already carries the rate measure**. | **EXACT** (algebraic) + numerical | `M5…md` §2, §4.2, §5; 2D prereg §5.1 |
| **C4′** | in particular, **not be a weight change**, because no weight change can turn a nonzero bias into an **exact** zero with a degenerate posterior (C1) | conjunction of C1 and C4 | **EXACT** | `SCAN_READOUT.md` §8.2 item 2 |
| **C5** | **act at the INTEGRAND PEAK, not in the wings of `p_gw`** | The truncation edge is pinned at ±4σ in the **GW-likelihood** variable, so everything discarded lives under `e^{-8}` of the peak: fractional perturbation capped at `2Φ̄(4) = 6.3e-5` **however much kernel mass is clipped**. A/B toy (12σ_d vs 4σ_d window): mean per-event \|Δ ln L\| = 3.8e-5, max 6.3e-5 (hits the analytic ceiling exactly) ⇒ implied MAP shift **+6.0e-7**, short by **6.2e4**. Corollary: **`_SIGMA_WINDOW` is an inert knob.** The ±5σ_k kernel clip loses ≤ 5.7e-7 of the kernel. | **BOUND** (analytic ceiling, measured tight) | `M3…md` §2, §4, §5 |
| **C5′** | not depend on the outer window generating an h-displacement | Under `u = D(z)/(h·d^obs)` the window is the **fixed** interval \|u−1\| < 4σ_dL for every h; deleting the truncation entirely changes the toy bias by **−1 %** | **EXACT** (algebraic) + −1 % measured | `M5…md` §1a, row B |
| **C6** | **NOT be the missing volume prior alone** — M1 has the **WRONG SIGN** | Bayes-correct `E[z_true\|z_obs] = z_obs + σ_z²·λ`, `λ = d ln p_pop/dz`. On the venue's actual 982-event population (median z = 0.494, IQR [0.35, 0.63]) **λ ≈ 2.3–2.9, positive throughout** ⇒ M1 predicts H₀ biased **LOW** by 0.02–0.04 — same order, opposite sign. Also quadratic (measured log-log slopes 1.26 → 1.51 → 1.95). Retained **only** as a compounding negative quadratic `b ≈ −5.29` against a linear driver `a ≈ +1.15`. | sign is categorical; every committed `bare` run is negative | `M1…md` §2–5; parent §7 |
| **C7** | **NOT be strictly bilinear** in the two doses | H-INT refuted: b(S23) is **+10.33σ** above its own point prediction **using the REGISTERED SE** (+14.64σ realized); bilinear residuals positive at all nine evaluable cells, > 3σ at S22 (+3.76σ), S31 (+7.64σ), S23 (+5.47σ); ≥ 7σ at S31 under **both** anchorings | **10.33σ** (registered SE) | `SCAN_READOUT.md` §3.2, §4.1; ledger #101 |
| **C8** | **NOT be threshold-shaped in the host dose** | H-THRESH refuted at **17.96σ** (S13) and **50.18σ** (S23). Row f_h = 0.25 ramps hard at less than half the registered threshold f* = 0.5262; there is no step and no dead zone below f*. | **18σ / 50σ** | `SCAN_READOUT.md` §4.1; ledger #101 |
| **C9** | **be NON-SEPARABLE** — the *shape* in the impostor dose must change with the host dose (not `f(f_h)·g(f_i)`) | Row-normalised interaction residual `D(f_h,f_i)/D(f_h,1)`: 0.857 / 0.786 at f_h = 0.25 vs **0.475 / 0.515** at f_h = 1.0. 86 % of the full interaction is delivered by a quarter impostor dose at f_h = 0.25, only 48 % at f_h = 1.0. | descriptive; underlying cells 10–34σ NON-ADDITIVE | `SCAN_READOUT.md` §4.6, §3.1 |
| **C10** | **leave the f_imp = 0 column SMALL and POSITIVE** — hence **must NOT by itself explain the `pp_coverage` sign flip** | Column measured **+0.0047 / +0.0053 / +0.0060** at f_h = 0.25 / 0.5 / 1.0, never negative. `pp_coverage`'s structurally identical bare kernel (`pp_coverage.py:868` vs the bare kernel now at `venue_transfer.py:1260`, cited as `:1136` in the registered notes at their commit) biases H₀ **LOW** by **−0.02 … −0.046** across σ_z = 0.005–0.05 with 0–3 % coverage. The pre-named carrier of that sign flip remains **M1's negative quadratic term**, not the interaction. | column: 14.0σ first step, then UNRESOLVED; flip carries **NO branch weight** | `SCAN_READOUT.md` §4.5, §8.2 item 3; 2D prereg §5.4 |
| **C11** | **vanish identically at σ_z = 0** — necessary but **WEAK** | T-0: all 200 seeds exactly on truth; S00 exact zero. **But** T-0's log-posterior peak curvature is **1.73e8** against **7.35e4** at T-b, so the ~2,645 nats/h needed to cure the dosed bias would displace T-0 by only **1.5e-5** — invisible. T-0 is evidence the *apparatus* is sound, not that any per-h term is right. | EXACT but **non-discriminating** for smooth per-h terms | parent §0(a) caveat; `M4…md` §4(a) |
| **C12** | **supply the amplitude budget**: the linear coefficient **a ≈ +1.15**, i.e. +0.037237 ± 0.000230 at σ̄_z = 0.0418, with `post_sd` 0.004376 (**8.5× narrower than the displacement**), coverage 0/0/0, PIT–KS D = 1.000, zero rails | decision cell T-c(0.730), N = 400, plus MN0X N = 100 (8.49 / 9.02) | ~160σ (campaign); the fit is order-of-magnitude, not precision | ledger #99, #100; `M1…md` §4b |
| **C12′** | supply a **slope of ≈ 1.94e3 per unit h** across the stack (≈ 1.98 per event, i.e. each event's `L_i` swinging ~8 % over the relevant ±0.04 in h) | `S_need = Δ/σ_post² = 0.037237/0.004376²` | arithmetic identity given the measured width | `M3…md` §3 |
| **C13** | **NOT rely on misspecification** — generator and estimator share each candidate's σ_k by construction | `_apply_dose_mask` scales the scatter **and** the kernel width together (`venue_transfer.py:1506-1508`); undosed members get σ_k = 0 *and* an unperturbed redshift, so the estimator point-evaluates them | **EXACT** (by construction) | parent §5 constraint (c); `ARMS.md` |
| **C14** | **be essentially K-insensitive at production multiplicity**, and survive every realism axis | Ladder: v2 +0.0353 → T-a +0.0349 (Poisson K ≈ 5) → T-b +0.0359 → T-c +0.0372 — **no killing axis**. The toy's K-saturation account (+0.0247 impostors-only at K = 50) is **falsified at production K** along the whole impostor axis: MEI and the entire f_h = 0 row are exactly zero. | ladder ~160σ; MEI/row EXACT | ledger #99; 2D prereg §0(ii); `SCAN_READOUT.md` §8.1 |
| **C15** | **act identically in both channels** — no 1D/2D split | 2D tracks 1D in all 16 cells and all 4 arms, same sign, **identical classification at every registered decision point**, offset +0.000333 … +0.002333 (below one core grid step) | no registered split threshold; qualitative but uniform | `SCAN_READOUT.md` §2.2, D-3 |
| **C16** | **live in the ESTIMATOR, not the generator** | V-M2 / AD-1..AD-3: at fixed seed, `K_sum`, `event_idx`, the **pre-dose** `z_obs`, the σ vector and the standard-normal scatter vector are bit-identical across all arms and all 16 cells; only the dose fraction differs. `_channel_terms_at_h`, `log_channel_posteriors_ball_sigma_vector` and `_g_ball_capped` are **byte-identical** across all 16 cells. | **EXACT** (unit-tested; but see §6 open item 5) | 2D prereg §2.2; parent §5 V-M2 |

**Sixteen constraints (C1–C16, with C2′, C4′, C5′, C12′ as sharpened sub-rows of C2, C4, C5, C12).**

### 3.1 The shape of the surviving space, stated without naming a candidate

Reading C1, C2 and C5 together: the surviving object must be **first-order in σ_z** (C2 kills every
symmetric-smoothing story by parity), must act **where the integrand is O(1)** (C5 kills every
wing/edge story), and must be **switched off completely by one exact redshift out of ~1,216
candidates** (C1). The parity argument's own statement of what that leaves is: *"genuine first-order
structure at scale σ_z — a support edge, the argmax operation itself (which is not a smooth
functional of the posterior), or a host/impostor asymmetry inside the finite ball window."* That
sentence is a **list of shapes, not a formula**, and this dossier does not narrow it further.

The most recent structural fact — the **gate × amplifier** asymmetry — is likewise a *shape* claim
(§5 below): the host acts as a binary switch and the impostor sea as a graded, saturating gain, and
the two do not factor.

**What the constraint set does NOT contain.** No row of the table names a term. `M5′` — the
over-broad effective candidate measure × Jacobian τ against a box support — is the one object that
ever reproduced the defect in a validated toy, and it is **refuted as registered**: its decisive
split-dose prediction (impostors-only ≥ 0.030, host-only ≤ 0.012) is **inverted on its decisive
half** at production K, and its `Δζ ∝ σ` flank scaling is refuted along the f_host = 1 row at
+10.9 SE. Its *structure* was never re-derived with the host pin included. Doing that re-derivation
is author-gated work; this document does not do it and does not assume it would succeed.

---

## 4. WHAT THE GATE PACKAGE STILL LACKS

The `/physics-change` protocol requires five items before any code is written, plus item 6 (each
source equation's stated validity conditions, per venue) and a regression test asserting the old
value. Current state:

| gate item | state | note |
|---|---|---|
| **1. Old formula** (exact expression + file:line) | ✅ **COMPLETE** | §1 of this document |
| **2. New formula** (proposed replacement) | ❌ **EMPTY** | No candidate is admissible-and-derived. M1/M3/M4/M5/M5′/M2/W1 are closed or refuted; the two registered surface shapes are refuted. |
| **3. Reference** (arXiv/DOI + equation, or a step-by-step derivation) | ❌ **EMPTY** | Nothing to cite, because there is nothing to derive *to*. The literature register carries a standing warning that the reduction to the per-event form requires **perfect** galaxy redshifts (Gair et al. 2023, arXiv:2212.08694 §2.3 after Eq. 30 — `docs/LITERATURE_WARNINGS.md:47`, status **VIOLATED — every venue**), and that Gray et al. 2020's photo-z handling is an **unexercised** equation validated only at σ_z = 0 (`LITERATURE_WARNINGS.md:84`). **There is no published equation that is known to hold in this regime.** |
| **4. Dimensional analysis** | ❌ **EMPTY** | Requires item 2. |
| **5. Limiting case** (an analytically known limit) | ⚠️ **PARTIALLY PRE-SPECIFIED** | The limits any replacement must hit are already fixed by C1 (exact zero at host-exact), C11 (exact zero at σ_z = 0) and C12 (the amplitude budget). These are *acceptance criteria*, not a check of a formula that exists. |
| **6. Validity conditions of each source equation, per venue** | ❌ **EMPTY** | Requires item 2. The venue-side conditions that must be re-checked are already enumerated as the NOT-EVALUABLE registry (§5 below). |
| **Regression test asserting the OLD value** | ⚠️ **AVAILABLE, NOT WRITTEN** | The materials exist: the V-M5 values golden at rtol ≤ 1e-12 with exact MAPs (`VM5_GOLDEN_20260814.md`), MN0X's N = 100 null, and the 16 committed cell JSONs. None is yet packaged as a `pytest` regression pinning the pre-change numbers. |
| **Ledger row** (`docs/gates/PHYSICS-GATE-LEDGER.md`) | ❌ not appended | The gate has not been presented; no row is due yet. A row is due the moment a five-item package is put to the author. |

**Stated plainly: the new-formula slot is empty.** Filling it is a derivation — an author-gated
physics decision about how a marginalised catalogue term and its selection normalisation must be
made the same functional of h at finite σ_z, on a candidate set whose support is a hard window and
whose host is exact-in-the-limit. That is not overnight work, it is not a subagent task, and it is
explicitly **not** what this dossier does. Per `CLAUDE.md`, the scientific decision is the author's.

---

## 5. REGISTERED LIMITS ON INTERPRETATION

These are binding on any use of the material above, and they are quoted rather than paraphrased
where they bind hardest.

1. **"Gate × amplifier" is a SHAPE claim, not a functional form.** The registered resolution floor
   gives **≈ 5.175 distinguishable levels** across the full dynamic range (0.034667 / 0.0066990),
   and `PREREGISTRATION_2D_DOSE_SCAN.md` §6 item 8 bars *"functional forms finer than ~5 levels of
   contrast."* A 4×4 grid with four levels per axis cannot separate a sigmoid from a saturating
   exponential from a two-component sum. `SCAN_READOUT.md` records explicitly that fitting one was
   tempting and that it is barred. **No parametric fit to the surface may be quoted.**
2. **No repair may be proposed from the current branch.** §6 item 7 of the 2-D prereg
   (*"Any repair. This scan proposes none and adopts no candidate"*), §7 branch 5 (*"no repair may
   be proposed"*), the parent's NO-OWNER handling (*"No repair may be proposed from a NO-OWNER
   read"*), and the ratified ledger row #101 all say the same thing. This dossier obeys all four.
3. **The DS-D3 defect is recorded, unadjusted.** DS-D3 is a **one-sided threshold with no upper
   edge**, so SHAPE-INTERACTION fires for any sufficiently large value — *including values that
   refute the hypothesis it names*. It was **not** adjusted (§4.7 anti-tuning); it is logged as a
   design fault of that pre-registration for a future amendment. Branch 2's **meaning clause is
   barred from being quoted.**
4. **The ±0.002 V-M1 window is likewise recorded, unadjusted.** It was asserted rather than derived,
   was tighter than the statistic it gated (~21 % false-fail rate under an exact null), and was
   settled **on data** at N = 100 rather than widened. The N = 15 `MN0` V-M1 status remains
   **FAILED** on the record; A1-PASS is a *new measurement*, not a re-scoring.
5. **Both registered shapes are wrong — that is the finding.** H-INT and H-THRESH are refuted at
   +10.33σ and 17.96/50.18σ respectively. Neither may be quoted as an account.
6. **Not established, and barred from any account** (`SCAN_READOUT.md` §4.7): the f_host = 1 flat
   middle interval (1.17σ, UNRESOLVED); the f_host = 0.5 dip at f_imp = 0.5 (2.93σ MARGINAL, 0.71σ
   in the row below, 2.65σ pooled); any functional form; anything about K-scaling, about transfer to
   production `BayesianStatistics`, or about reweighting.
7. **The NOT-EVALUABLE registry is carried in full** and constrains what any downstream package may
   claim: (1) bilinearity below f_h·f_i ≈ 0.163 — S11/S12/S21 excluded *even though they carry the
   largest anti-bilinear residuals*, which runs **against** the readout's own conclusion;
   (2) paired variance reduction forfeited (√2 inflation on every cell-to-cell comparison);
   (3) K-dependence; (4) transfer to production `BayesianStatistics` — **any estimator fix routes
   `/physics-change`**; (5) f_incl < 1 / empty-ball events / completeness / window-interior n(z) /
   sky-cone geometry — the read is conditional on host-in-ball over the 982 nonempty-ball events;
   (6) the `pp_coverage` sign flip (analogue, not replication: K = 1 vs K̄ ≈ 1,216, and α differs);
   (7) any repair; (8) functional forms finer than ~5 levels.
8. **Attribution discipline.** Every itemisation in rows #99/#100/#101 is *orchestrator-derived*,
   not author dictation; the author's rulings are the quoted verbatim words. The branch-5 ruling of
   2026-08-14 was **SUPERSEDED** after adversarial verification corrected the framing, and the
   author then ruled verbatim **"a"** — branch 2 fired, DS-D3 defect logged, meaning barred. The
   `[DO]/[RULE]/[STANDING]` approval-scope convention in `CLAUDE.md` (commit `804b4c5d`) exists
   because of that correction loop.

---

## 6. OPEN ITEMS

| # | item | status |
|---|---|---|
| 1 | **D-A1-2 — V-M5 not re-executed as the registered artifact** | **CLOSED 2026-08-14.** `VM5_GOLDEN_20260814.md` (+ `.json`, `verify_vm5_golden.py`) re-executes the registered condition (rtol ≤ 1e-12 on every shared field **and** exactly equal MAPs) at commit `94c0480a68606d0f19f7f56feb62817a917a1b90`, on registered seeds 20286808–20286810, against committed `B2_h0p730_results.json`: **PASS**, overall max relative deviation **1.6135e-14** (`pit_2d`), two orders of magnitude inside the ceiling; all four MAP fields exactly equal; the entire 1D channel bit-identical; all deviation confined to the 2D channel + `M_source_median` at 1–2 ULP, consistent with the certified Route 1 adaptive Gauss–Hermite change. Filing the closure into `A1_READOUT.md` / the ledger was left to the orchestrator and **has not yet been done** — that clerical step is still outstanding. |
| 2 | **D-A1-3 — V-M2/AR-3 cross-arm generator invariance not scored** | **LIVE.** A1-DET certifies the `dose_target="all"` path across `e83ed0b9 → 3aedbe55`. The `"host"`/`"impostors"` paths were refactored by the same commit and **MEH/MEI ran at `e83ed0b9`, before the refactor**. Any comparison of MN0X against MEH/MEI crosses an instrument commit. AR-1/AR-3 unit tests pass at HEAD; that is evidence, not the registered artifact. |
| 3 | **V-D5 scope over-claim** | **LIVE (disclosed, not cleaned up).** `SCAN_READOUT.md` §1.4 labels V-D5 **PASS** in its header, but for the 16 scan cells V-D5 is strictly **NOT-EVALUABLE** — they run on fresh disjoint seeds with no committed golden. The body discloses the scope (D-5); the header over-claims. A non-evaluable check is not a failed one, so branch 1 is unaffected. |
| 4 | **The f_h = 0.5 dip call is CONVENTION-FRAGILE** | **LIVE.** −2.93σ (MARGINAL, "not established") under the inherited **ddof = 1** convention, but **−3.034σ (RESOLVED)** under ddof = 0 — undisclosed in the readout. It is the single "not established" call sitting inside rounding distance of its own boundary. Recorded; **no account may lean on the dip either way.** |
| 5 | **§4.7 / §4.6 internal contradiction** | **LIVE.** `SCAN_READOUT.md` §4.7 "Supported" item 3 asserts the fast-then-slow behaviour at > 3σ, while §4.6's own caveat notes the f_host = 0.25 row difference is **1.36σ**. The supported-list entry overstates what §4.6 licenses for that row. |
| 6 | **`dose_scales` naming deviation** | **LIVE (naming only).** Prereg §2.1 fixes two `VenueConfig` fields `dose_frac_host` / `dose_frac_imp`; the implementation uses a single `dose_scales` tuple (`venue_transfer.py:316, 441, 1463`). Semantics and float operand order identical, corners reduce exactly (AD-1), so this is a naming deviation, not a behavioural one. |
| 7 | **V-D2 / AD-1..AD-3 not re-executed by the readout** | **LIVE.** They are unit-test obligations discharged before the cells ran; the readout verifies their observable consequences (the 982 host-mask pin, the §5.3 dosing table) but does not re-run the tests. |
| 8 | **The H-INT-distance SE choice** | **DISCLOSED.** The readout's "+14.6σ above H-INT's prediction" uses the **realized** SE; the **registered** SE gives **+10.33σ**. This dossier quotes the registered SE in C7. |
| 9 | **Both corner cross-checks land high, same sign** (D-2) | **LIVE, low impact.** S33 +2.42σ above MN0, S30 +2.64σ above MEH; each passes its own registered per-cell 3σ tolerance. If the +51000 seed block carries a small positive offset, the surface *levels* shift by ~0.002–0.005 while every **difference-based** shape statistic is unaffected. It affects only DS-D6's R_dose values and DS-D5's registered-line comparison. |
| 10 | **§6 item 1 escape (S11, S12, S21 at N = 100)** | **NOT REQUESTED, author-order only.** Realized SE_D in the low corner came in at 0.00074–0.00080 against the 0.0016672 the exclusion was written against, and those three cells carry the largest anti-bilinear residuals (+8.2σ, +4.8σ, +12.3σ). |
| 11 | **Grid-spacing inconsistency between registered notes** | **MINOR, recorded.** `M4_alpha_sigma_blindness.md` §5 describes the 41-point grid as "spacing 0.0065" (a mean); the grid read back from every cell JSON is non-uniform — 0.01 in the wings, **0.005 on the core [0.65, 0.80]**, which is the spacing the A1 quantisation argument and the 2-D prereg use. No verdict depends on it. |
| 12 | **The one production-side physics bug still open in `CLAUDE.md`** | unrelated to this thread but carried: `physical_relations.py` w0/wa silently ignored (GitHub #4). Named here only so the intake is not read as the sole open physics item. |

---

## 7. ONE-LINE SUMMARY

The old formula is fully specified (§1); the phenomenology is established at ~160σ with an exact
host gate and a graded impostor amplifier (§2); **sixteen constraints** now bound any candidate
(§3); and the **new-formula slot of the `/physics-change` package is empty** and remains
author-gated work (§4). **No repair is proposed anywhere in this document.**
