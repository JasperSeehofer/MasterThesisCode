# ch02_FLAGS.md — Chapter 2 ("Bayes, Once and For All")

Raised by the ch02 agent, 2026-07-31, per `BOOK_DESIGN.md` §4.1: *"if a generator's
recomputation disagrees with a spec number, **stop and flag; do not silently reconcile in
either direction**."*

Neither item blocks the chapter. Both are presented on the page in **both** forms.

---

## F-ch02-1 — EMRI-889's "σ_dL/dL = 8.0×10⁻⁵" is the **absolute** σ_dL in Gpc, not a fraction — BLOCKING FOR OTHER CHAPTERS

- **Spec value (three places, all identical):**
  - `BOOK_DESIGN.md` §1 Ch 1 running example: "σ_dL/dL = 8.0×10⁻⁵";
  - `BOOK_DESIGN.md` §1 Ch 6 dossier: "σ_dL/dL = 8.0×10⁻⁵, correlated with sky";
  - `BOOK_PEDAGOGY.md` Part 2 beat B4: "fractional distance precision **σ_dL/dL = 8.0×10⁻⁵**";
  - `BOOK_PEDAGOGY.md` Part 3 **Q1.2** builds an answer on it: "σ_H0/H0 ≈ σ_dL/dL ≈ **0.008%**".
  - It is already rendered on a shipped page: `book/site/ch04-loud-half.html` dossier row.
- **Measured by `gen_ch02.py`** from
  `results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv`,
  row 889:
  - `delta_luminosity_distance_delta_luminosity_distance` = 6.3748×10⁻⁹ — a **covariance**
    entry, i.e. a variance, in the parameter's own units
    (`parameter_estimation.py:430-480`: the columns are the entries of Σ = Γ⁻¹).
  - ⇒ **σ_dL = 7.9843×10⁻⁵ Gpc = 0.0798 Mpc**.
  - `luminosity_distance` = 0.0888792 **Gpc** (the FEW/`few` convention; 88.9 Mpc).
  - ⇒ **σ_dL/dL = 8.983×10⁻⁴ ≈ 9.0×10⁻⁴ (0.090%)**, *not* 8.0×10⁻⁵.
- **Three independent confirmations that the fraction is ~9×10⁻⁴:**
  1. **The repo says so itself.** `results/campaign51_20260728/idealization_audit/IDEALIZATION_LEDGER.md:31`:
     "The **3 loudest** (SNR 995–1425, z ≈ 0.016–0.021, **σ_dL/dL = 0.09–0.11%**) carry
     **46%**…". `gen_ch02.py` reproduces exactly 0.090% / 0.101% / 0.107% for events
     889 / 1536 / 118.
  2. **Order of magnitude.** For a matched-filter amplitude parameter σ_dL/dL ≈ 1/ρ. Over
     all 1590 rows the measured median of (σ_dL/dL)·SNR is **1.040** (IQR 1.006–1.097).
     With ρ = 1424.7 that gives 7.3×10⁻⁴; 8.0×10⁻⁵ would require ρ ≈ 1.3×10⁴.
  3. **Dimensional.** 8.0×10⁻⁵ is dimensionally the Gpc number: 7.9843×10⁻⁵ Gpc to
     2 s.f. is 8.0×10⁻⁵. The two quantities differ by exactly the factor d_L/1 Gpc.
- **Disposition in this chapter:** the Ch 2 dossier prints **both** — "σ_dL = 7.98×10⁻⁵ Gpc
  = 0.0798 Mpc ⇒ σ_dL/dL = 9.0×10⁻⁴ (0.090%)" — names the design docs' `8.0×10⁻⁵` as the
  absolute-σ reading, shows the arithmetic that relates them, and links this flag. Neither
  value is dropped and neither is asserted to supersede the other by fiat.
- **For the integrator (action needed, not by me):** Ch 1, Ch 4 and Ch 6 all carry the
  spec's fractional reading, and `BOOK_PEDAGOGY.md` Q1.2's *answer* ("0.008%") is used
  verbatim by Ch 1. That answer's arithmetic changes by 11× if the fraction is 9.0×10⁻⁴.
  This is a cross-chapter consistency item and is above a single chapter agent's authority.

## F-ch02-2 — "3 golden events carry 46%" reproduces **only** under the project's own 3-point curvature metric

- **Spec value:** `BOOK_DESIGN.md` §1 Ch 2 ("3 carry 46%"), `BOOK_PEDAGOGY.md` Q2.2, both
  citing `IDEALIZED_BASELINE_READOUT.md:42-47`.
- **Reproduced exactly (0.46996 → 46%)** using the metric declared in
  `results/campaign51_20260728/realistic_20260729/score_realistic.py:14-21` and reused
  verbatim by `gen_ch02.py`:
  `curv_k = ln(L_k(0.73)/L_k(0.725)) + ln(L_k(0.73)/L_k(0.735))`, evaluated on the
  **canonical idealized directory** `run_seed61000/posteriors_fixed` (§4.2 rule 1). Total
  238.332, implied σ_h = dh/√Σcurv = 3.239×10⁻⁴. In-catalogue share 101.3%, dark −1.3%
  (readout: "100%" and "~1%"). Golden set = {1536, 889, 118}, SNR 1068 / 1425 / 995
  (readout: "SNR 995–1425"). ✓
- **It does NOT reproduce under other natural metrics**, and the chapter says which metric
  it is using every time it prints the number:
  - quadratic fit of ln L over the **zoom** grid → top-3 share **52.5%**;
  - CRB-only Fisher weights 1/(σ_dL/dL)² over the 76 in-catalogue events → **41.9%**.
- **Disposition:** no contradiction with the spec — the spec number is right and the metric
  is now pinned. Logged so that no later chapter recomputes "46%" a different way and
  reports a conflict that is really a metric change. **Ch 10 and Ch 11 use the same
  statistic and should import this definition, not invent one.**

## F-ch02-3 — the realistic-venue information shares are **not quotable** (carried, not a disagreement)

- `REALISTIC_READOUT.md:110-113` (the artifact's own words): *"the percentages are
  ill-conditioned and should not be quoted … that is why 'dark share' reaches 140% and one
  run's golden share goes to −159%. Quote the signed sums, never the ratios."*
- `gen_ch02.py` therefore emits the realistic-r1 **signed** curvature sums and the absolute
  curvature mass, sets `quotable_ratios: false` in `ch02_information.json`, and the page's
  Event Stacker refuses to display a golden/in-catalogue *share* in the realistic venue,
  saying why. This is a live constraint on the chapter, not a numerical disagreement.
