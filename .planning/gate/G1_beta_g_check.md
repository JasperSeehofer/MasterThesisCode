# G1 — β_G discrete GLADE-sum verification (P0) — 2026-07-02

## Verdict: FAIL for the 'global' mode (real, quantified); local modes structurally immune

The commission flagged (verification report §7 / scratch/d2 RESULT 3): the Option-A cancellation
between the discrete catalogue sum Σ_global(h) = Σ_g w_g P_det(d_L(z_g,h)) and the continuous
β_G(h) = D(h) − β_Ḡ(h) "is delicate … should be checked by summing the real GLADE catalogue
directly". Done: `scripts/verify_beta_g_discrete_sum.py` (seed600 injection pool, n_sky_bands=6,
SNR_THRESHOLD=20, 14 h-values in [0.60, 0.86]; results `.planning/gate/G1_beta_g_check.json`).

## Findings

1. **Raw ratio Σ_global/β_G grows ×2.48 across the grid** (raw tilt +93%). The dominant piece is
   the *expected* catalogue-density scaling: the catalogue is a fixed set of galaxies, so its
   implied comoving density is n_gal(h) ∝ 1/V_c ∝ h³ ((0.86/0.60)³ = 2.94). This factor is common
   to the discrete numerator and denominator of L_cat and cancels in the likelihood ratio — it is
   NOT by itself a bug, but it falsifies the code-comment claim that n_gal is "an overall constant"
   (it is h-dependent; only its *ratio* cancellation is exact).
2. **After the h³ correction a smooth monotonic residual remains: −17.2% end-to-end**
   (+8.7% at h=0.60 → −8.7% at h=0.86, normalized at 0.72). This is a REAL h-dependent mismatch
   between the discrete f-weighted catalogue content and the continuous completeness-model
   integral. Candidate origins: (a) modeled f(z,Ω,h) vs the catalogue's true incompleteness as
   z_max(h) sweeps outward; (b) real n(z) inhomogeneity (LSS + depth-dependent density) vs the
   constant-comoving-density assumption; (c) the mass-integrated-rate-constancy surrogate.
3. **Impact:** in 'global' mode the residual multiplies the in-catalogue channel β_G·L_cat once per
   event — a coherent per-event tilt of this size is enormous after N≈500 events, corroborating the
   rail mechanism. In 'local_ratio'/'volume_deconv', Σ_global is never used (L_cat is the local
   self-normalized ratio; N_ball-common factors cancel), and w_G = β_G/D is continuous/continuous —
   the check does not constrain those modes negatively.

## Gate consequences

- 'global' is confirmed unusable for results (already deprecated + warned in 235b783).
- Paper A gains a quantitative panel: raw ×2.5 (n_gal(h) ∝ h³, cancels) vs residual −17%
  (does not cancel) — the anatomy of why global-denominator dark-siren normalizations are fragile.
- Follow-up (optional, Paper A polish): decompose the −17% residual into (a)/(b)/(c) by re-running
  with f≡1 and with a z-restricted subsample.

Reproduce: `uv run python scripts/verify_beta_g_discrete_sum.py --injections_dir <seed600 pool>`.
