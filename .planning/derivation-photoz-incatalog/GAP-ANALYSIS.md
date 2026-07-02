# GAP-ANALYSIS: what our partition-norm pipeline is missing vs the working photometric methods

Status: **pre-derivation.** This file (1) tabulates OURS vs THEIRS across the five scope dimensions,
(2) ranks the specific divergences that plausibly cause our railing, each mapped to one of the four
solution hints, (3) resolves the `p_det~1` degeneracy question with evidence from the verified
extractions, (4) gives the regime verdict, and (5) states the single most likely missing piece and a
recommended next step for discussion. **No derivation, no code** — the handoff says STOP here.

---

## 1. OURS-vs-THEIRS table (five dimensions)

| Dimension | Gray 2020 (1908.06050) | Hitchhiker (2212.08694) | GWcosmo (2308.02281) | Cross-Parkin (2502.17747) | Echoes/CHIMERA (2509.18243) | **OURS** |
|---|---|---|---|---|---|---|
| **1. per-galaxy z term** | prior slot; `p(z_i)` general; spec only | **likelihood** `L_red` (Eq.17) → posterior `p_red = L·p_bg/Z` (Eq.16) | posterior `G(z−ẑ_k)` (Eq.2.9); **no extra dV_c on hosts** (fn.10) | **likelihood** (Eq.10) × `p_bg=dV_c/dz` (Eq.11) → posterior (Eq.9) | likelihood × `p_bkg` → per-gal posterior (Eq.8) | **BARE `N(z;z_g,σ)`; NO `p_bg`, NO `1/Z_g`** (Deviation 1) |
| **2. same kernel in num=denom?** | YES per window (Eq.A.10), but **degenerates** at flat `p_det` | YES in Eq.3 (approx); exact = hierarchical Eq.31/33 | NUANCED: prior H0-dep **cancels** num+denom (Sec.2.1.4) | **YES, purest Hint-1** (Eq.3); de-rails via **varying `p_det`** | **NO** — `p_cat` num-only, smooth `p_pop` in `ξ` (Option-A) | **NO — three different densities** (bare-N / point-`p_det(z_g)` / smooth `p_bg`) |
| **3. normalisation level** | per-event, frozen scalar | **per-event=approx; hierarchical=exact (shared latent z)** | **hierarchical** `−N_det` (Eq.2.2) | per-event (Eq.2) | hierarchical global `ξ` (Eq.1) | **per-event + single GLOBAL SCALAR `D(h)`** (no coherence) |
| **4. photo-z injection self-consistent?** | NO (delta only) | **YES** (Sec.3.1, Hint-3 form) | NO controlled photo-z mock | **YES** (Eq.1, Hint-3 form) | YES but **spectroscopic only** | **NO — inject at exact z, infer with σ=0.035 kernel** (Hint-3 violation) |
| **5. is our regime (z~0.05, σ_z/z~0.7, p_det~1) inside validated range?** | **NO** (σ_z=0 only) | **NO** (tested ≤3%; sparse-axis) | **NO** (varying p_det, incomplete) | **NO** (z∈[0.15,0.7], σ_z/z≤0.07, varying p_det, photo-z sub-dominant) | **NO** (spec only, varying P_det) | **— this IS our regime; it RAILS** (+0.13) |

---

## 2. Ranked divergences plausibly causing the rail (each mapped to a hint)

The empirical anchor (NORMALISATION-FIX.md): the truth (0.73) sits **strictly between two rails** —
the bare-Gaussian standard rails **DOWN to 0.60**, every cleaned numerator-only variant rails **UP to
0.87**. No numerator-only fix lands on truth. This is the decisive constraint: the missing piece is
**not** a numerator-kernel choice.

### Rank 1 — Per-event frozen GLOBAL SCALAR denominator instead of a hierarchical shared-latent normalisation → **Hint 2 (ensemble/hierarchical coherence)**
- **Divergence:** OURS divides each event by one event-independent scalar `D(h)` (`:701-707`). The
  exact object (Hitchhiker Eq.31/33) marginalises the host's TRUE redshift as a SHARED latent
  variable ONCE across all events, with `p_det(H0,{z_g})` INSIDE the z-integral as a function of the
  true redshifts.
- **Why it rails:** Hitchhiker proves the per-event Eq.3 is only an APPROXIMATION; it biases H0 HIGH
  for imperfect (photo-z) redshifts (Sec.3.3, Fig.8 — VERIFIED CONFIRMED). The `{z_g}`-dependence of
  the denominator "breaks the separability" — collapsing it to a frozen scalar is exactly our
  approximation. Our cleaned-numerator rail UP to 0.87 is the same-direction signature as Hitchhiker's
  per-event photo-z HIGH bias.
- **Literature support:** Hitchhiker Eq.31/33 (HIGH, with the downgrade that σ_z/z~0.7 itself is not
  tested); GWcosmo `−N_det` ensemble form (Eq.2.2, HIGH). **Confidence: HIGH that this is the
  structural class of the fix; MEDIUM that hierarchy alone suffices at σ_z/z~0.7.**

### Rank 2 — Object-for-object sim↔inference photo-z inconsistency → **Hint 3 (photo-z-consistent re-injection)**
- **Divergence:** OURS injects EMRIs at the host's EXACT catalogue z (delta-sharp truth, true host
  99%, `_bridge_lib.py:351-353`) but inference convolves a `σ_z=0.035` kernel around it — smearing a
  delta. The consistent setup (draw true z; report `z_g=z_true+N(0,σ_z)`; source at true z; convolve
  around `z_g`) is NOT implemented.
- **Why it rails:** the bridge Rung G shows the photo-z convolution over a delta-sharp truth is the
  decisive ingredient that reproduces +0.13 (BRIDGE-FINDINGS.md). A pure spec-z filter FAILS because
  it removes the injected photo-z hosts. **This is a prerequisite to even TEST any normalisation
  cleanly** — until injection and inference agree object-for-object, no normalisation can be unbiased
  at fixed catalogue.
- **Literature support:** Hitchhiker Sec.3.1 and Cross-Parkin Eq.1 both implement exactly this
  consistent loop (both VERIFIED CONFIRMED). **Confidence: HIGH that this is a real defect and a
  testing prerequisite; it may be necessary-but-not-sufficient on its own.**

### Rank 3 — Bare Gaussian missing `p_bg/Z_g` and conflation of catalogue density with the prior → **Hint 4 (don't double-count dV_c)**
- **Divergence:** OURS numerator convolves the BARE `N(z;z_g,σ)` with NO `p_bg`, NO `1/Z_g`
  (`:1623,:1646`), while `D(h)`/`B_num` carry the smooth `p_bg ∝ dV_c/dz`. The correct per-galaxy
  object is the regularised posterior `p_red = norm·p_bg/Z_g`.
- **Why it (partly) rails:** the doubly-smeared dV_c in the standard numerator behaves like an
  effective dV_c double-count → Hitchhiker's "Inconsistency 1" predicts a LOW H0 bias, matching our
  standard rail DOWN to 0.60. Removing it (cleaning to `p_bg`) **removes the low rail** but overshoots
  to the HIGH rail (0.87) — because the residual per-event structure (Rank 1) then dominates. So Hint
  4 is **necessary to stop the low rail but insufficient alone** (NORMALISATION-FIX.md proves cleaning
  the numerator does not de-rail).
- **Literature support:** Hitchhiker Inconsistency 1 / Fig.11 (physics CONFIRMED; label unverified);
  GWcosmo footnote 10 (CONFIRMED, but CONDITIONAL — "no dV_c on hosts" holds only if catalogue
  redshifts are POSTERIORS; if GLADE photo-z are likelihoods, even GWcosmo applies a comoving-volume
  prior); Cross-Parkin Eqs.9–12 (CONFIRMED). **Confidence: HIGH that this is a genuine defect; HIGH
  that it is NOT the de-railing lever by itself.**

### Rank 4 (DEMOTED) — Numerator/denominator NOT a global same-kernel ratio → **Hint 1 (joint same-kernel ratio)**
- **Divergence:** OURS uses three different redshift densities (Section 2, COMPARISON.md). The
  literature same-kernel form (Cross-Parkin Eq.3, Gray2020 Eq.A.10) puts the identical `p_cat` in num
  and denom.
- **Why it does NOT de-rail us:** see Section 3 below — under `p_det~1` the global same-kernel ratio
  PROVABLY degenerates. Adopting it is correctness-improving but **cannot by itself break the rail**.
  Demoted from "candidate fix" to "necessary consistency property that is insufficient."
- **Literature support:** ALL FOUR working extractions confirm the degeneracy (HIGH). **Confidence:
  HIGH that Hint 1 alone is insufficient in our regime.**

---

## 3. The `p_det~1` degeneracy — does the global same-kernel ratio degenerate? (decisive)

**YES — confirmed independently by all four working-method extractions, and consistent with the
project's own negative result.** When `p_det(z,H0) ≈ 1` across the in-catalogue support:

- **Gray 2020 (Eq. A.10):** denominator `Σ_i ∫ p_det(z_i) p(z_i) dz_i → Σ_i ∫ p(z_i) dz_i = N` (a
  constant; residual H0-dependence only via the Schechter luminosity weight, NOT the local gradient).
  VERIFIED CONFIRMED.
- **Hitchhiker (Eq. 3):** `∫ p_det p_CBC → ∫ p_CBC = W` (const); the paper's own remedy for imperfect
  redshifts is the hierarchical Eq.31/33, not the per-event ratio. VERIFIED CONFIRMED (flagged as our
  inference, mathematically sound).
- **GWcosmo (Sec. 2.1.4):** stronger still — the comoving-volume prior's H0-dependence "drops out
  when normalised" and "cancels" between numerator and denominator, so the denominator **never** tracks
  the local gradient in ANY regime, and in flat `p_det` it is H0-independent overall. VERIFIED
  CONFIRMED.
- **Cross-Parkin (Eq. 3):** `∫ p_det(z,H0) p_cat(z) dz → ∫ p_cat(z) dz` (H0-independent); their
  unbiasedness rides entirely on the threshold `z~0.29` sitting INSIDE `[0.15,0.7]` so `p_det` varies.
  VERIFIED CONFIRMED (paper silent on flat-`p_det`; degeneracy is our derivation).

**So what do the working methods actually rely on to break the degeneracy?**

1. **Selection VARIATION** (Cross-Parkin, Echoes, Gray2020, GWcosmo, in their validated regimes): a
   `p_det` that sweeps `1 → 0` across the catalogue does the de-railing work — it cuts the rising
   `dV_c/dz`. **We do not have this** (`p_det~1`, hosts at z~0.05 far inside the horizon).
2. **The ENSEMBLE/HIERARCHY** (Hitchhiker Eq.31/33; GWcosmo `−N_det`): when per-event selection is
   uninformative, de-railing comes from marginalising a SHARED latent host redshift ONCE across all
   events, with selection inside the integral. The collective product of GW likelihoods over the
   shared z-integral localises `z*(H0)` even though no single event's denominator is informative.
   **This is Hint 2, realised via Hint 3 (shared-latent photo-z likelihood).**
3. **Catalogue-as-likelihood with dV_c counted once** (Hint 4): a necessary correctness condition in
   all of them, but NOT the de-railing lever.

**Verdict on the degeneracy:** Hint 1 (global same-kernel ratio) is **necessary-but-insufficient** in
our regime. With flat `p_det` it degenerates per-event; the de-railing must come from selection
variation (which we lack) or from ensemble/hierarchical coherence (Hint 2) — leaving Hint 2 as the
only available lever for us.

---

## 4. Regime verdict

**The GLADE-at-z~0.05 regime is inside NO literature method's validated range.** Every working method
requires at least one of: varying selection, higher z, or much smaller `σ_z/z`:

- **Gray 2020:** spectroscopic only (`σ_z=0`); photo-z never exercised.
- **Hitchhiker:** photo-z tested only to `σ_z/z = 3%` (Eq.3 already biased there); the hierarchical
  fix validated to ~tens-% but **not** to `σ_z/z~0.7`; failure demonstrated in the SPARSE one-galaxy
  axis, not our many-galaxy axis.
- **GWcosmo:** varying `p_det`, incomplete catalogue, no controlled photo-z mock.
- **Cross-Parkin:** `z∈[0.15,0.7]`, `σ_z/z ≤ 0.07`, varying `p_det`, **and** `σ_z(photo) ≲ σ_z^GW`
  (GW still localises). We have `σ_z(photo) ≈ 17× σ_z^GW`, `σ_z/z~0.7`, flat `p_det`, `z~0.05`.
- **Echoes/CHIMERA:** spectroscopic only; photometric explicitly deferred to future work.

The single deepest discriminator: in every validated demonstration the **GW distance still localises
the host** (either via varying `p_det` or via `σ_z ≲ σ_z^GW`). In our regime the photo-z dominates the
GW by ~17×, so per-event host localisation is genuinely lost — pushing us into either (a) the
ensemble/hierarchical regime or (b) an information-starvation conclusion that the in-catalogue
per-event method cannot de-rail here.

---

## 5. Single most likely missing piece

> **The per-event frozen global-scalar normalisation itself.** Our pipeline approximates the exact
> hierarchical likelihood by collapsing the host-redshift selection into one event-independent scalar
> `D(h)`. The exact object (Hitchhiker Eq.31/33) marginalises the host's TRUE redshift as a SHARED
> latent variable ONCE across all events, with the detection probability evaluated INSIDE the z-integral
> as a function of the true redshifts. That hierarchical/ensemble coupling (Hint 2), realised through a
> photo-z-consistent shared-latent likelihood (Hint 3) and with dV_c counted exactly once (Hint 4), is
> the lever that survives when `p_det~1`. **Hint 1 (global same-kernel ratio) is necessary for
> consistency but provably insufficient in our flat-`p_det` regime.**

Supporting logic: the empirical two-rail bracket (0.60 low / 0.87 high, truth between) shows the defect
is not in the numerator kernel; the degeneracy proof (Section 3) shows it is not curable by any
denominator change that keeps the per-event/global-scalar structure; the only literature object
demonstrated to handle imperfect redshifts with non-varying selection is the hierarchical shared-latent
likelihood.

**Necessary companions (not the single piece, but required for it to work or be tested):**
- **Hint 4** stops the low rail (remove the effective dV_c double-count; use `p_red = norm·p_bg/Z_g`),
  but overshoots to the high rail alone — it is a correctness fix, not the de-railing lever.
- **Hint 3** is a testing prerequisite: until injection and inference agree object-for-object, no
  normalisation can be cleanly validated at fixed catalogue.

---

## 6. Recommended next step for discussion (NOT a derivation)

Bring to the discussion three coupled decisions, in this order:

1. **Commit to the hierarchical re-derivation as the target structure** (Hitchhiker Eq.31/33 adapted
   to the partition-norm in/out-of-catalogue split), accepting that this is a denominator/structure
   change — outside the frozen-global-scalar search space that the negative result has exhausted.
2. **Decide the catalogue-redshift interpretation** (the GWcosmo footnote-10 fork): are GLADE photo-z
   entries POSTERIORS (then no extra dV_c on hosts) or LIKELIHOODS (then a comoving-volume prior is
   required)? This decision fixes whether Hint 4 is "remove dV_c" or "add the regularised posterior,"
   and it changes which rail-direction the fix must cancel.
3. **Scope the photo-z-consistent re-injection** (Hint 3) as the first bridge rung, since it is the
   prerequisite for any clean test and is cheap relative to the full hierarchical marginalisation.

Open question to settle before any code: **does ensemble coherence actually de-rail at `σ_z/z~0.7`,
or is the in-catalogue method information-starved in this corner** (in which case the project's
honest output is the spec-z forecast arm plus a caveated GLADE limitation)? No literature method has
demonstrated de-railing at our `σ_z/z`, so this must be established by a bridge prototype of the
hierarchical form, not assumed.
