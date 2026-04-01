# Enhanced Sampling Design Report

## 1. Executive Summary

**Result: Neyman-optimal stratified allocation achieves 7-42x variance reduction in P_det boundary bins, far exceeding the >2x contract target.**

The two-stage sampling design uses Phase 18 uniform injections (~22k-26k events per h-value) as pilot data and prescribes a targeted campaign of ~12k-18k additional injections per h-value, with 70% of the targeted budget concentrated on boundary bins (0.05 < P_det < 0.95). The defensive mixture proposal (alpha=0.3) guarantees full support and bounded importance weights (max w <= 3.33).

Key findings:
- Weighted-average VRF across boundary bins ranges from **11.8x** (h=0.90) to **24.9x** (h=0.60)
- CI half-widths in boundary bins improve by **3.4-4.6x** compared to uniform sampling with the same total budget
- All 7 h-values pass the VRF > 2.0 contract target
- h=0.90 benefits least (VRF_min = 6.9) due to fewer pilot events (17k vs 22-26k)

**Recommendation:** Implement the two-stage design for the next injection campaign. Even the worst-case h-value (0.90) achieves VRF = 6.9 in its weakest boundary bin, substantially exceeding the 2x threshold.

## 2. IS Estimator

The self-normalized importance sampling estimator (Plan 19-01) computes per-bin detection probability as:

$$
\hat{P}_{\text{det}}(B) = \frac{\sum_{i \in B} w_i \cdot \mathbb{1}[\text{SNR}_i \geq 15]}{\sum_{i \in B} w_i}
$$

where $w_i = p(\theta_i) / q(\theta_i)$ is the importance weight.

For pilot samples drawn from $p_{\text{uniform}}$: $w_i = 1$ (no reweighting needed).
For targeted samples drawn from $q$: $w_i = p(\theta_i) / q(\theta_i) \leq 1/\alpha = 3.33$.

**Implementation:** `analysis/importance_sampling.py:weighted_histogram_estimator()`

Reference: Tiwari (2018), arXiv:1712.00482, Eq. 5-8.

## 3. Neyman Allocation

### Method

The Neyman-optimal allocation assigns targeted injections proportional to the Bernoulli standard deviation:

$$
N_k = \left\lfloor N_{\text{targeted}} \cdot \frac{\sigma_k}{\sum_k \sigma_k} \right\rfloor, \quad \sigma_k = \sqrt{\hat{P}_k (1 - \hat{P}_k)}
$$

with constraints:
- $N_k \geq 5$ for all bins with pilot data (minimum allocation floor)
- $\sum_k N_k = N_{\text{targeted}}$ exactly (integer conservation via remainder distribution)
- Bins with $P_{\text{det}} = 0$ or $P_{\text{det}} = 1$: $\sigma_k = 0$, receive only the minimum

Reference: Cochran (1977), Sampling Techniques, Ch. 5; Owen (2013), Ch. 8.

### Per-h Allocation Summary

| h    | N_pilot | N_targeted | Boundary bins | Mean alloc/bd bin | Mean alloc/non-bd | Ratio  |
|------|---------|------------|---------------|-------------------|-------------------|--------|
| 0.60 | 22500   | 15749      | 4             | 2844.5            | 32.1              | 88.5x  |
| 0.65 | 26000   | 18200      | 5             | 3459.8            | 6.7               | 518.4x |
| 0.70 | 23500   | 16450      | 3             | 3525.0            | 42.9              | 82.2x  |
| 0.73 | 25500   | 17850      | 5             | 3435.0            | 5.0               | 687.0x |
| 0.80 | 25000   | 17500      | 5             | 3326.4            | 6.4               | 517.4x |
| 0.85 | 25500   | 17850      | 5             | 3353.2            | 8.0               | 417.6x |
| 0.90 | 17000   | 11900      | 5             | 1971.6            | 16.6              | 118.8x |

The boundary concentration ratio far exceeds the 5x target in all cases. This is expected: only 3-5 of 140-150 non-empty bins are boundary bins (~2-3%), so concentrating 70% of the budget on them yields ratios > 80x.

### Boundary Bin Details

| h    | Bin(i,j)  | P_det | sigma | n_pilot | N_k  | VRF  | CI_unif | CI_strat |
|------|-----------|-------|-------|---------|------|------|---------|----------|
| 0.60 | (0,3)     | 0.056 | 0.229 | 126     | 2072 | 10.3 | 0.0307  | 0.0096   |
| 0.60 | (0,5)     | 0.050 | 0.219 | 119     | 1979 | 10.4 | 0.0302  | 0.0094   |
| 0.60 | (0,6)     | 0.202 | 0.402 | 94      | 3633 | 23.3 | 0.0623  | 0.0129   |
| 0.60 | (0,8)     | 0.212 | 0.408 | 52      | 3694 | 42.4 | 0.0851  | 0.0131   |
| 0.65 | (0,5)     | 0.076 | 0.266 | 144     | 2791 | 12.0 | 0.0333  | 0.0096   |
| 0.65 | (0,6)     | 0.121 | 0.326 | 141     | 3422 | 14.9 | 0.0412  | 0.0107   |
| 0.65 | (0,7)     | 0.214 | 0.410 | 112     | 4312 | 23.2 | 0.0583  | 0.0121   |
| 0.65 | (0,8)     | 0.168 | 0.374 | 107     | 3931 | 22.2 | 0.0544  | 0.0115   |
| 0.65 | (0,9)     | 0.080 | 0.271 | 88      | 2843 | 19.6 | 0.0434  | 0.0098   |
| 0.70 | (0,5)     | 0.126 | 0.332 | 151     | 3181 | 13.0 | 0.0406  | 0.0113   |
| 0.70 | (0,7)     | 0.159 | 0.366 | 88      | 3508 | 24.0 | 0.0586  | 0.0120   |
| 0.70 | (0,8)     | 0.207 | 0.405 | 116     | 3886 | 20.3 | 0.0565  | 0.0126   |
| 0.73 | (0,5)     | 0.152 | 0.359 | 99      | 3526 | 21.5 | 0.0542  | 0.0117   |
| 0.73 | (0,6)     | 0.081 | 0.273 | 136     | 2009 | 9.3  | 0.0351  | 0.0115   |
| 0.73 | (0,7)     | 0.282 | 0.450 | 103     | 4423 | 25.8 | 0.0666  | 0.0131   |
| 0.73 | (0,8)     | 0.295 | 0.456 | 112     | 4484 | 24.1 | 0.0648  | 0.0132   |
| 0.73 | (0,9)     | 0.084 | 0.278 | 83      | 2733 | 20.0 | 0.0459  | 0.0103   |
| 0.80 | (0,5)     | 0.090 | 0.286 | 89      | 2564 | 17.5 | 0.0456  | 0.0109   |
| 0.80 | (0,6)     | 0.149 | 0.356 | 121     | 3190 | 16.1 | 0.0486  | 0.0121   |
| 0.80 | (0,7)     | 0.281 | 0.450 | 96      | 4031 | 25.3 | 0.0690  | 0.0137   |
| 0.80 | (0,8)     | 0.250 | 0.433 | 132     | 3882 | 17.9 | 0.0567  | 0.0134   |
| 0.80 | (0,9)     | 0.125 | 0.331 | 80      | 2965 | 22.4 | 0.0556  | 0.0117   |
| 0.85 | (0,5)     | 0.077 | 0.266 | 130     | 2330 | 11.1 | 0.0351  | 0.0105   |
| 0.85 | (0,6)     | 0.221 | 0.415 | 140     | 3631 | 15.8 | 0.0528  | 0.0133   |
| 0.85 | (0,7)     | 0.377 | 0.485 | 151     | 4239 | 17.1 | 0.0593  | 0.0143   |
| 0.85 | (0,8)     | 0.226 | 0.418 | 93      | 3656 | 23.7 | 0.0652  | 0.0134   |
| 0.85 | (0,9)     | 0.127 | 0.333 | 134     | 2910 | 13.4 | 0.0432  | 0.0118   |
| 0.90 | (0,5)     | 0.074 | 0.262 | 121     | 1298 | 6.9  | 0.0359  | 0.0137   |
| 0.90 | (0,6)     | 0.234 | 0.424 | 128     | 2096 | 10.2 | 0.0563  | 0.0176   |
| 0.90 | (0,7)     | 0.410 | 0.492 | 122     | 2434 | 12.3 | 0.0669  | 0.0191   |
| 0.90 | (0,8)     | 0.301 | 0.459 | 103     | 2270 | 13.6 | 0.0679  | 0.0185   |
| 0.90 | (0,9)     | 0.149 | 0.356 | 74      | 1760 | 14.6 | 0.0622  | 0.0163   |

All boundary bins are in row 0 (smallest d_L bin, d_L < 0.1 Gpc), consistent with Phase 18 finding that detections are confined to d_L < 1 Gpc and M > 2e5 Msun.

## 4. Variance Reduction

### VRF Definition

The variance reduction factor compares the per-bin variance under stratified vs uniform allocation for the same total budget:

$$
\text{VRF}_k = \frac{\text{Var}_{\text{uniform},k}}{\text{Var}_{\text{stratified},k}} = \frac{n_{\text{pilot},k} + N_k}{n_{\text{pilot},k} + N_{\text{new}} \cdot f_k}
$$

where $f_k = n_{\text{pilot},k} / N_{\text{pilot}}$ is the pilot fraction per bin. The Bernoulli variance $P_k(1-P_k)$ cancels in the ratio.

The weighted-average VRF across boundary bins uses sigma-weighting (higher-variance bins have more influence):

$$
\overline{\text{VRF}} = \frac{\sum_{k \in \text{boundary}} \sigma_k \cdot \text{VRF}_k}{\sum_{k \in \text{boundary}} \sigma_k}
$$

### VRF Results

| h    | VRF_mean (boundary) | VRF_min | Mean CI_uniform | Mean CI_stratified | CI improvement |
|------|---------------------|---------|-----------------|---------------------|----------------|
| 0.60 | 24.9                | 10.3    | 0.0521          | 0.0112              | 4.6x           |
| 0.65 | 18.9                | 12.0    | 0.0461          | 0.0107              | 4.3x           |
| 0.70 | 19.3                | 13.0    | 0.0519          | 0.0119              | 4.4x           |
| 0.73 | 21.2                | 9.3     | 0.0533          | 0.0120              | 4.5x           |
| 0.80 | 20.1                | 16.1    | 0.0551          | 0.0124              | 4.5x           |
| 0.85 | 16.8                | 11.1    | 0.0511          | 0.0127              | 4.0x           |
| 0.90 | 11.8                | 6.9     | 0.0578          | 0.0170              | 3.4x           |

**All h-values exceed VRF > 2.0.** The VRF is large because boundary bins are rare (3-5 out of 140-150 non-empty bins) and the Neyman allocation concentrates targeted injections there.

Note: CI improvement is approximately sqrt(VRF) since CI half-width scales as 1/sqrt(n).

## 5. Two-Stage Design Specification

### Stage 1: Pilot (existing)

- **Data:** Phase 18 uniform injection campaign
- **Events:** ~22k-26k per h-value (17k for h=0.90)
- **Distribution:** Current EMRI prior p(z, M) = emri_distribution(M, z) from Model1CrossCheck
- **Grid:** 15x10 (d_L x M), matching Phase 18 recommendation
- **Purpose:** Estimate per-bin P_det and sigma_k; identify boundary bins

### Stage 2: Targeted (future)

- **Budget:** N_targeted = 0.7 * N_pilot per h-value (~12k-18k events)
- **Allocation:** Neyman-optimal, proportional to sigma_k = sqrt(P_hat * (1-P_hat))
- **Minimum:** 5 injections per non-empty bin (prevents empty strata)
- **Distribution:** Defensive mixture proposal q = 0.3 * p_uniform + 0.7 * g_targeted

### Proposal Distribution

$$
q(\theta) = \alpha \cdot p_{\text{uniform}}(\theta) + (1 - \alpha) \cdot g_{\text{targeted}}(\theta)
$$

where:
- $\alpha = 0.3$ (defensive mixture fraction, Hesterberg 1995)
- $p_{\text{uniform}}(\theta) = \text{emri\_distribution}(M, z)$ (current injection prior)
- $g_{\text{targeted}}(\theta)$ is piecewise constant in (d_L, M) bins, proportional to $N_k / A_k$ where $A_k$ is the bin area

### Weight Formula

- **Pilot samples** (drawn from $p_{\text{uniform}}$): $w_i = 1$
- **Targeted samples** (drawn from $q$): $w_i = p_{\text{uniform}}(\theta_i) / q(\theta_i)$

The combined estimator pools pilot and targeted samples:

$$
\hat{P}_{\text{det}}(B) = \frac{\sum_{i \in B}^{\text{pilot}} 1 \cdot I_i + \sum_{i \in B}^{\text{targeted}} w_i \cdot I_i}{\sum_{i \in B}^{\text{pilot}} 1 + \sum_{i \in B}^{\text{targeted}} w_i}
$$

This is the self-normalized IS estimator (Tiwari 2018, Eq. 5-8), already implemented in `analysis/importance_sampling.py:weighted_histogram_estimator()`.

### Effective Sample Size

Per-bin $N_{\text{eff}}$ from Kish (1965):

$$
N_{\text{eff}}(B) = \frac{\left(\sum_{i \in B} w_i\right)^2}{\sum_{i \in B} w_i^2}
$$

Farr criterion: $N_{\text{eff}} > 4 \cdot N_{\text{det}}$ per bin (Farr 2019, arXiv:1904.10879).

## 6. Full Support Guarantee

**Theorem.** For $\alpha > 0$, the defensive mixture $q(\theta) = \alpha \cdot p(\theta) + (1-\alpha) \cdot g(\theta)$ satisfies $q(\theta) > 0$ wherever $p(\theta) > 0$.

**Proof.** Since $g(\theta) \geq 0$ by construction (it is a probability density), we have:

$$
q(\theta) = \alpha \cdot p(\theta) + (1-\alpha) \cdot g(\theta) \geq \alpha \cdot p(\theta)
$$

For $\alpha = 0.3 > 0$ and $p(\theta) > 0$ over the injection domain (emri_distribution > 0 for $z \in (0, 0.5]$ and $M \in [10^4, 10^7] M_\odot$), we conclude $q(\theta) \geq 0.3 \cdot p(\theta) > 0$.

**Corollary.** The importance weight $w_i = p(\theta_i) / q(\theta_i) \leq p(\theta_i) / (\alpha \cdot p(\theta_i)) = 1/\alpha = 3.33$ is bounded above. No infinite weights are possible.

This guarantees that the IS estimator has finite variance for any sample from $q$ (Hesterberg 1995).

## 7. Implementation Checklist

To implement this design in the injection campaign code:

- [ ] **Load pilot data:** Read Phase 18 CSVs and build 15x10 P_det grid per h-value
- [ ] **Compute allocation:** Call `sampling_design.neyman_allocation()` to get per-bin N_k
- [ ] **Map bins to (z, M):** Convert bin centers from (d_L, M) to (z, M) using `dist_to_redshift()` with the target h-value
- [ ] **Sample from q:** For each targeted injection:
  1. With probability alpha=0.3: sample from p_uniform (existing EMRI prior)
  2. With probability 0.7: choose bin k with probability proportional to N_k, then sample uniformly within that bin
- [ ] **Compute weights:** For each targeted sample, compute $w_i = p_{\text{uniform}}(\theta_i) / q(\theta_i)$ using `sampling_design.defensive_mixture_weight()`
- [ ] **Store weights:** Add a `weight` column to injection CSVs (w=1.0 for pilot, w=w_i for targeted)
- [ ] **Build weighted grid:** Pass weights to `SimulationDetectionProbability.__init__()` via the weights parameter (Plan 19-01)
- [ ] **Check N_eff:** Verify Farr criterion per-bin using `importance_sampling.farr_criterion_check()`

## 8. Fallback

**If VRF < 2.0 for some h-values:** Grid coarsening from 30x20 to 15x10 already provides ~3.2x CI improvement (Phase 18 result). This is an alternative path that requires no new injections.

**Current status:** All h-values achieve VRF >> 2.0, so the fallback is not needed. However, h=0.90 has the weakest performance (VRF_min = 6.9) due to fewer pilot events (17k vs 22-26k). If the next injection campaign achieves fewer events for h=0.90, the VRF will decrease but is unlikely to fall below 2.0 unless pilot data is less than ~5000 events.

**Weakest assumption:** Pilot sigma estimates from Phase 18 may be noisy for bins with $n_{\text{total}} < 30$. In the 15x10 grid, all boundary bins have $n_{\text{total}} \geq 52$, so this is not a concern. The sigma accuracy is approximately $1 / \sqrt{2n} \approx 7-10\%$ for the observed counts, sufficient for Neyman allocation (the allocation is robust to sigma misspecification up to ~30%, Owen 2013 Ch. 8).

---

_Generated by `analysis/sampling_design.py` using Phase 18 injection data._
_Phase: 19-enhanced-sampling-design, Plan: 02_
_Date: 2026-04-01_
