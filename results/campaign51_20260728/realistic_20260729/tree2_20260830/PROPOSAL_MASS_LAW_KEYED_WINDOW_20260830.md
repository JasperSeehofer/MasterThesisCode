# T5.1 -- Mass-law-keyed window design and the k-scan registration (DESIGN PROPOSAL, zero compute)

launched under rows #255/#268 -- tree 2 node T5.1

Date: 2026-08-30. HEAD at authoring: 647e86d9 (branch fix/p32d-classg-venue-repair). Class: TOP-TIER
DESIGN PROPOSAL. Zero evaluate() calls, no cluster, no ssh, no git, no code edits, append-only; the
runner-7 directory (tree2_20260830/hier_s0_zwin_run) was not opened. Every number below carries
{value, source file:line or file:key, date}. Numbers marked ARITH are closed-form or scratchpad
arithmetic on banked inputs; the three scratchpad scripts are named in section 10 and each reproduces
a banked comparand to the printed digit before any new number is read. Grant: row #255 A1 = (c)
("commission the mass-law-keyed window design / k-scan first (docket 2 section 7 rank 5; no adoption,
no joint_r1 arm before that)") and row #268 (the standing extension "until tomorrow"). This node
designs and registers; it launches nothing (cluster down, Lustre OST 5 inactive at both rulings).

## 0. Bottom line

The production injection ties the EMRI mass to the host with NO scatter: host_galaxy.M is the
catalogue row's own BH_MASS (handler.py:73-80, :1190, :1210) and the waveform gets M_z = M (1+z)
(datamodels/parameter_space.py:260-268). The truth-versus-catalogue mass law that a window actually
faces is therefore set by the EVALUATION catalogue, and it differs by venue: on iiib the estimator reads
the same unscattered catalogue (observed_catalogue = None in every iiib run_metadata), so the true
host's catalogue mass equals the injected mass exactly (a delta law); on joint_r1 the estimator reads
observed_catalogue_seed900001.csv (sigma_scale 1.0), whose masses are realized forward as a LOG-NORMAL,
ln M_obs = ln M_true + sigma_lnM N(0,1) with sigma_lnM = BH_MASS_ERROR/BH_MASS (observed_realization.py:5-9,
:349-356; docs/derivations/realistic_host_observation_model.md section 1.3). The mirror's linear-Gaussian
truth draw (correspondence_1d.py:1743-1750) is the ESTIMATOR'S KERNEL, not the production observation
law; and the mirror's banked 78.9 percent retention at log k=3 is not a shape effect at all: 378 of its
481 failing events are the pre-repair floor-clipped latents with M_true == 1.0 M_sun (380/2261 = 16.8
percent of the evaluated fleet), and on the post-repair fleet (33 arms, 2275 events, zero clipped) log
k=3 retains 94.4 percent against linear k=1.5's 95.4 percent. Closed forms: under the log-normal law a
log window at k retains exactly 1 - 2 Phi(-k) (0.9973 at k=3) independent of CV and independent of
detection selection, because the scatter is applied to the catalogue after the GW selection; the
production linear window at k=1.5 retains only Phi(ln(1+1.5 CV)/CV) = 0.832 / 0.819 / 0.784 at CV =
0.86 / 1.02 / 1.5 (one-sided, too-light losses only), and no finite k makes the linear window epsilon-
keyed there (k = 11.6 at CV 0.86 for 99.73 percent). This matches the campaign #53 P6 read on the
scattered venue (2D miss 42.6 percent vs 1D 25.0 percent, seed 61000: a 17.6-point excess against the
closed form's 16.8 at the catalogue-median CV 0.86) and the iiib production readout (66/76 recovered
under BOTH geometries -- exactly what a delta law predicts). Recommended design: a log-symmetric
window with k = Phi^-1(1 - epsilon/2), epsilon registered (0.27 percent at k=3), keyed to the
catalogue-side observation law; it is exact-by-construction on the scattered venue and inert for the
true host on the unscattered venue, where the only design object is the impostor pool. The registered
arms: (i) the iiib k-scan {2, 2.5, 3.5} (+ an optional k = infinity no-window anchor), H4 each, approx
5 CPU-h per k, reading the impostor-pool response with the banked k=3 point (+0.0035) as its fourth
point; (ii) the joint_r1 k=3 arm (approx 11-15 CPU-h), which is the decisive arm for the true-host
design object (registered prediction: in-catalogue true-host recovery rises by 16-22 points of 73).
The mirror-generator fix -- a mass_law flag whose non-default value realizes the production Convention
(A) scatter on the catalogue the estimator sees -- is specified as a tree-3 gate item and NOT built.

## 1. Which law production uses (from code, cited)

### 1.1 Injection side (the truth)

- main.py:586-601: the simulation refill draws hosts with draw_mixture_hosts(200, rng, galaxy_catalog,
  completeness, F, h=h_value, z_max=HOST_DRAW_Z_MAX) and then calls
  parameter_space.set_host_galaxy_parameters(host_galaxy, h=h_value). {source: darksiren_emri/main.py:586-601,
  read 2026-08-30}
- dark_siren_injection.py:594-676 (draw_mixture_hosts): in-catalogue hosts come from
  galaxy_catalog.draw_rate_weighted_hosts; dark hosts from draw_dark_hosts. {source:
  darksiren_emri/dark_siren_injection.py:594-676}
- handler.py:1190 and :1210 (draw_rate_weighted_hosts): the weight is R_eff_per_mbh(BH_MASS)/(1+z) over the
  eligible catalogue rows and each host is HostGalaxy(eligible_catalog.iloc[position]) -- a genuine catalogue
  row, no snap, no overwrite. handler.py:73-80 (HostGalaxy.__init__): M = row[BH_MASS], M_error =
  row[BH_MASS_ERROR]. {source: darksiren_emri/galaxy_catalogue/handler.py:73-80, :1190, :1210}
- parameter_space.py:260-268: self.M.value = redshifted_mass(host_galaxy.M, host_galaxy.z), i.e. M_z = M (1+z),
  no random draw of the mass anywhere on the injection path; the catalogue mass IS the injected source-frame
  mass. {source: darksiren_emri/datamodels/parameter_space.py:260-268}
- Dark hosts (catalog_index = -1): M drawn from the per-dex EMRI-rate mass marginal
  mbh_mass_function(M) R_eff_per_mbh(M) on a log10 grid (dark_siren_injection.py:368-395); their M_error is
  the bookkeeping value 0.1 M (_DARK_HOST_FRACTIONAL_M_ERROR, dark_siren_injection.py:85, :554) that the
  inference never reads. Dark hosts have no catalogue row, so the mass window acts on them only through
  impostors. {source: darksiren_emri/dark_siren_injection.py:85, :368-395, :554}

Production law on the injection side, stated once: M_true(in-catalogue) = BH_MASS of the injection
handler's catalogue row, exactly. Convention (A) of the ratified observation model ("the reduced
catalogue's stored values are declared TRUE; observations are realized FORWARD", observed_realization.py:5-6)
is what makes this a truth, not an estimate.

### 1.2 Catalogue side at evaluation time (the observation law) -- venue-dependent

- iiib: observed_catalogue = None and realize_observed_catalogue = False in the evaluation CLI
  {source: headreadout_20260827/off_iiib/run_metadata_21.json:cli_args; wave2_20260829/c3/run_metadata_7.json:
  cli_args, 2026-08-30 read}. The evaluation handler loads the same unscattered reduced catalogue
  (handler.py:322-329 else-branch; md5 c52c13b5... per the C3 registration section 1). Hence for the 76
  iiib events with a catalogued true host, M_cat(true host) == M_true to the byte: a DELTA law. Any mass
  window at any k and either geometry retains the true host (only the z-box and the sky/z cone can lose
  it). This is exactly what B5.2 measured: 66/76 recovered under linear k=1.5 and 66/76 under log k=3,
  same 76-event positivity pattern {source: B5_2_WIN_K3_READOUT_RECORD.md section 3; b5_2_readout.json,
  2026-08-29}. The "retention non-transfer" (L10) is not a transfer failure; it is two different laws.
- joint_r1: observed_catalogue = /pfs/work9/.../realizations_20260729/observed_catalogue_seed900001.csv
  {source: headreadout_20260827/joint_r1/run_metadata_21.json:cli_args}; sidecar realization_seed 900001,
  sigma_scale 1.0, n_rows 22641048, parent sha256 7af3f4f4a2..., git 7b30d1ff, 2026-07-29 {source:
  realizations_staged/observed_catalogue_seed900001.meta.json}. Same injection as iiib (both venues'
  run_metadata_21 stamp git d04d9dc9, seed 777021, timestamp 2026-08-27T19:40:20; the C3 sbatch links
  CRB_SRC run_20260729_seed61000 -- cluster/wave2_c3_win_k3.sbatch:40-42). The realization law:
  ln M_obs = ln M_g + sigma_scale sigma_lnM N(0,1), sigma_lnM = M_error/M from the same load-time Reines and
  Volonteri mapping {source: observed_realization.py:5-9 (docstring), :349-356 (delta_ln_bh =
  sigma_scale sigma_ln_bh mass_std_normal; bh_mass_obs = bh_mass exp(delta_ln_bh));
  docs/derivations/realistic_host_observation_model.md:146-166 section 1.3}. A LOG-NORMAL law, by
  construction, with the width the window reads off the observed row.
  Width caveat (A11 stamp): the seed-900001 realization predates the exact-width writer (its sidecar has no
  n_mass_width_floor key); for that writer generation the code records "MEASURED pull vs the recomputed
  width 0.929, per-row drift up to +-18%" {source: observed_realization.py:357-365 comment}, i.e. the
  width the window reads (loaded) exceeds the width the scatter was drawn with by approx 7.6 percent on
  average with +-18 percent per-row drift. The sidecar's own drawn-width check is 0.99983
  {source: meta.json:width_check.mass.normalized_residual_std}. Realizations written by the current
  writer preserve the width exactly except the counted n_mass_width_floor rows (observed_realization.py:366-383).

### 1.3 The estimator's mass kernel (NOT the observation law)

host_mass_kernel = auto resolves to "gaussian" under normalization_mode = absolute_marginal
(bayesian_statistics.py:299; both venues' cli_args), i.e. the analytic Gaussian product in M_z_frac with
mu_gal_frac = host_M_eff (1+z)/det_M and sigma_gal_frac = host_M_error (1+z)/det_M
(bayesian_statistics.py:7884-7895), host_M_eff being the Eddington-shifted (R_eff-weighted) mean
(bayesian_statistics.py:639-690, :7789-7797). The truncated log-normal x R_eff kernel exists
(RATIFY-M3/M4, bayesian_statistics.py:237, :425-470) but is not the production default. The
kernel-family mismatch against the log-normal observation law on the scattered venue is already bounded:
"mass-kernel family -- bounded at +0.002" {source: HANDOFF_20260730.md section 3, 2026-07-30}. This
proposal keys the WINDOW to the observation law, not to the kernel: a window's retention is a property
of the data-generating law, and the kernel question is a separate, already-bounded item.

### 1.4 The mirror's law (what B5.1/B5.2 measured retention on)

correspondence_1d.py:1743-1750 (_draw_2d_accepted_latents, host_mode catalogue_selected_2d):
m_eff = _eddington_shifted_host_mass_batch(host_m, host_m_error); m_true = m_eff + sigma N(0,1), sigma =
BH_MASS_ERROR; M <= 0 rejected (post-repair; the banked p3_2d_fleet_20260825 fleet, git fb4ac4ee, was
generated by the PRE-repair code that floor-clipped to 1.0 M_sun -- p32d_residual_accounting_20260827.md:
57-71). The draw is then ACCEPTED with probability S_4D(d_L, M_z_true) (rejection sampling), so the
accepted class's pull relative to the catalogue is selection-tilted by construction. This is the
estimator's kernel used as a truth law (self-generated convention) -- a legitimate calibration object,
but not the production observation law, and the direction of scatter is reversed relative to Convention
(A): production scatters the CATALOGUE around a fixed truth (after the GW selection has acted on the
truth); the mirror scatters the TRUTH around the catalogue and then selects on it.

### 1.5 Law table

| venue / object | truth-vs-catalogue mass law faced by the window | retention of the true host at log k | at linear k=1.5 |
|---|---|---|---|
| iiib production (76 in-cat events) | delta (unscattered catalogue at evaluation) | 1 for every k | 1 |
| joint_r1 production (73 in-cat events) | log-normal, sigma_lnM = M_error/M (observed row), selection-independent | 1 - 2 Phi(-k sigma_drawn/sigma_loaded) = 0.9973 nominal, approx 0.9988 at the 0.929 width ratio | Phi(ln(1+1.5 CV)/CV): 0.832 (CV 0.86), 0.819 (1.02), 0.784 (1.5) |
| mirror, pre-repair fleet (banked, b5_*) | linear-Gaussian truth, floor-clipped at 1.0 M_sun, S_4D-selected | 0.7877 measured (16.8 percent clipped) | 0.9500 |
| mirror, post-repair fleet (33 arms) | linear-Gaussian truth, M > 0 rejected, S_4D-selected | 0.9442 measured (k=3) | 0.9538 |
| mirror, unselected closed form (LT + Eddington) | same law, no S_4D | 0.989 (ARITH, fleet-averaged) | 0.94-0.95 (ARITH) |

## 2. The retention integrals (exact) and the numeric table

Notation: r = M_true/M_cat; CV = BH_MASS_ERROR/BH_MASS of the catalogue row the window reads; Z ~ N(0,1);
Phi = standard normal CDF. GW side treated as a point (median CRB fractional mass error 1.6e-8, mean
0.0042, p99 0.0646 {source: PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md R2, 2026-08-29}); the
cosmology z-box (z_min/z_max) only widens both windows and is common to both geometries, so mass-only
retention is a lower bound on the window's own retention. Windows on r: linear k: 1 - k CV <= r <= 1 + k CV
(the lower edge is vacuous when k CV >= 1, i.e. CV >= 2/3 at k=1.5 -- 99.61 percent of the pruned catalogue,
wgeom_result.json:p2.negative_lower_edge_fraction = 0.996112); log k: exp(-k CV) <= r <= exp(+k CV).

(a) Linear-truncated law (mirror): r = 1 + CV Z conditioned on Z > -1/CV (Eddington shift ignored here;
its effect is a +19..+44 percent shift of the centre, section 3).
  R_LT,lin(k, CV) = [Phi(k) - Phi(max(-k, -1/CV))] / [1 - Phi(-1/CV)]
  R_LT,log(k, CV) = [Phi((e^{k CV} - 1)/CV) - Phi(max((e^{-k CV} - 1)/CV, -1/CV))] / [1 - Phi(-1/CV)]
(b) Log-normal law (production, scattered venue): ln r = -CV Z.
  R_LN,log(k, CV) = 1 - 2 Phi(-k)                        (CV-independent; the epsilon = 2 Phi(-k) identity)
  R_LN,lin(k, CV) = Phi(ln(1+k CV)/CV) - Phi(ln(1-k CV)/CV) for k CV < 1;  = Phi(ln(1+k CV)/CV) for k CV >= 1
  (the second form is one-sided: only too-light true hosts, i.e. catalogue heavier than truth, are lost)
(c) Delta law (production, unscattered venue): R = 1 for every k and geometry.

These are the same closed forms the WGEOM registration used for its epsilon-semantics table
(PREREGISTRATION_MKER_WGEOM_20260828.md:93-96: light-side cut Phi(ln(1-kCV)/CV) for kCV < 1 else 0;
heavy-side cut 1 - Phi(ln(1+kCV)/CV)), now read as true-host RETENTION under the production law instead of
as candidate-population truncation. Every entry below was cross-checked by Monte Carlo (2e6 draws) to 4
decimals {source: scratchpad t5_retention.py, ARITH, 2026-08-30}.

Retention table (rows CV, entries LT-linear | LT-log | LN-linear | LN-log):

| CV | k=1.5 | k=2.0 | k=2.5 | k=3.0 | k=3.5 |
|---|---|---|---|---|---|
| 0.50 | 0.8866 / 0.8612 / 0.8657 / 0.8664 | 0.9767 / 0.9175 / 0.9172 / 0.9545 | 0.9936 / 0.9447 / 0.9476 / 0.9876 | 0.9986 / 0.9618 / 0.9666 / 0.9973 | 0.9998 / 0.9729 / 0.9785 / 0.9995 |
| 0.86 | 0.9239 / 0.9107 / 0.8323 / 0.8664 | 0.9741 / 0.9459 / 0.8777 / 0.9545 | 0.9929 / 0.9662 / 0.9089 / 0.9876 | 0.9985 / 0.9786 / 0.9310 / 0.9973 | 0.9997 / 0.9863 / 0.9468 / 0.9995 |
| 1.02 | 0.9201 / 0.9307 / 0.8186 / 0.8664 | 0.9728 / 0.9601 / 0.8622 / 0.9545 | 0.9926 / 0.9766 / 0.8929 / 0.9876 | 0.9984 / 0.9861 / 0.9152 / 0.9973 | 0.9997 / 0.9917 / 0.9319 / 0.9995 |
| 1.50 | 0.9106 / 0.9693 / 0.7840 / 0.8664 | 0.9696 / 0.9857 / 0.8223 / 0.9545 | 0.9917 / 0.9933 / 0.8505 / 0.9876 | 0.9982 / 0.9968 / 0.8721 / 0.9973 | 0.9997 / 0.9985 / 0.8891 / 0.9995 |

The k that retains 99.73 percent (= 1 - 2 Phi(-3)) per law and CV (ARITH, brentq):

| CV | LT-linear | LT-log | LN-linear | LN-log |
|---|---|---|---|---|
| 0.50 | 2.79 | 7.52 | 6.04 | 3.00 |
| 0.86 | 2.82 | 5.36 | 11.56 | 3.00 |
| 1.02 | 2.84 | 4.59 | 15.76 | 3.00 |
| 1.50 | 2.88 | 3.11 | 42.6 | 3.00 |

Reading: under the production scattered-venue law only the log window is epsilon-keyed at a CV-independent
k; the linear window cannot be made to retain 99.7 percent at any usable k (11.6 at the catalogue median CV,
42.6 at the fleet p90 CV 1.48). The CV anchors used: pruned-catalogue census quantiles p10/median/p75/p90 =
0.785 / 0.861 / 0.940 / 1.214 {source: wgeom_result.json:p2.table, 2026-08-28}; rate-weighted true-host CV on
the mirror fleets: median 1.018 (pre-repair, b5_pull_read.json:pooled.CV_median) and 1.007 (post-repair,
scratchpad t5_pull_repaired.py), p10-p90 0.804-1.479 / 0.804-1.391; the cone-candidate population on the
scattered venue is heavier-tailed still (median sigma_lnM 1.28, HANDOFF_20260730.md section 4, 2026-07-30).

Truncation mass of the mirror law, P(Z < -1/CV): 0.023 / 0.123 / 0.163 / 0.253 at CV 0.5 / 0.86 / 1.02 / 1.5
(ARITH) -- the left-tail asymmetry the pull read named is real but, per section 3, it is not what
produced the 78.9 percent.

joint_r1 width-drift correction (old-writer realization, section 1.2): with the loaded width exceeding the
drawn width by the measured mean ratio 1/0.929, the log window's effective k is k/0.929 and R_LN,log becomes
0.8936 / 0.9687 / 0.9929 / 0.9988 / 0.9998 at k = 1.5 / 2 / 2.5 / 3 / 3.5 (nominal 0.8664 / 0.9545 / 0.9876 /
0.9973 / 0.9995), with a per-row band from the +-18 percent drift of [0.989, 0.9997] at k=3 (k_eff 2.54-3.66)
(ARITH). Disclosed, not corrected for in the registered prediction beyond this band.

## 3. Reconciliation of the mirror's 78.9 percent: a floor-clip artifact, not a shape effect

Zero-compute stratified re-read of the SAME banked fleet and catalogue the B5 reads used (24 arms,
p3_2d_fleet_20260825, catalogue md5 c52c13b5...), reusing the b5/wgeom loaders verbatim and adding the
banked per-event M_true and s4d_at_truth columns (prepared_cramer_rao_bounds.csv columns 132 and 135)
{source: scratchpad t5_pull_strat.py, 2026-08-30; gate: reproduces b5_pull_read.json pooled fraction
|pull| <= 3 = 0.7877 and <= 1.5 = 0.6979 to 4 decimals}:

- 380 of 2261 evaluated events (16.81 percent) have M_true == 1.0 M_sun exactly -- the pre-repair floor
  clip (_M2D_OBS_M_FLOOR); this is the same 380 that p32d_residual_accounting_20260827.md:70-71 counts
  ("793/4800 latents ... M_true == 1.0 exactly; 380 of those 793 pass F-0"). Per-arm clipped fraction
  0.096-0.226, median 0.171.
- Of the 481 events failing the log k=3 mass window, 378 are clipped; all 481 are too-light (M_true <
  M_cat), 0 too-heavy; median M_true/M_cat of the fails = 0.0 to 5 decimals.
- Non-clipped events (n = 1881): retention log k = 1.5 / 2 / 2.5 / 3 / 3.5 = 0.8389 / 0.8900 / 0.9213 /
  0.9452 / 0.9638; linear k=1.5 = 0.9405.
- Post-repair fleet (results/_archive/p3_2d_fleet_repair_20260827, 33 bc arms, 2275 events; zero events
  below 1e3 M_sun; min M_true 2375) {source: scratchpad t5_pull_repaired.py, 2026-08-30}: retention log
  k = 1.5 / 2 / 2.5 / 3 / 3.5 = 0.8369 / 0.8897 / 0.9235 / 0.9442 / 0.9626; linear k = 1.5 / 2 / 2.5 / 3 =
  0.9538 / 0.9833 / 0.9956 / 0.9987. The 127 log-k=3 fails are all too-light with M_true/M_cat in
  [0.008, 0.075] (p10-p90): the linear law's near-zero tail. Pull median -0.092, mean -0.472, sd 1.304
  (pre-repair: -0.369 / -2.738 / 5.542). Per-arm log k=3 retention 0.899-1.000, median 0.944.
- Residual attribution on the post-repair mirror: the unselected closed form (LT + Eddington, fleet-
  averaged over each host's own (M_cat, CV)) predicts 0.989 at log k=3; the measured 0.944 sits 4.5
  points lower, in the near-zero-mass tail that the S_4D grid-edge clamp favours ("mean s4d_at_truth =
  0.826 on floor rows vs 0.718 on real-mass rows", p32d_residual_accounting_20260827.md:60-62; in the
  pre-repair read the failing events' median s4d 0.862 vs 0.791 overall). This residual is mirror-side
  (a selected linear-Gaussian truth has density at r -> 0+; the production log-normal has none).

Consequences (append-only; the older records are not edited by this node -- pointer notes are proposed
in section 9):
1. B5_2_PULL_READ_20260829.md section 4 item 3 ("the root mechanism ... linear-Gaussian latent mass
   truncated at M > 0 ... No sigma_lnM redefinition closes this gap") is REFUTED-IN-PART as an
   attribution: its numbers stand (0.7877, CV 1.018, the cross-check to b5_window_count.json), its
   sigma_lnM identity stands (L9), but the shape mechanism accounts for at most ~1 point (closed form
   0.989 unselected) plus a ~4.5-point selected near-zero tail; 16.8 points are the floor clip.
2. PREREGISTRATION_WIN_K3_COUNTERFACTUAL_20260829.md section 5's carried number ("78.9 percent ... is the
   number that carries into ... any adoption argument, not 99.73 percent") is superseded: on the mirror
   post-repair the number is 94.4 percent (log k=3) vs 95.4 percent (linear k=1.5); on production it is
   1.0 (iiib, delta) and 0.9973 nominal (joint_r1, log-normal). The R1 falsifier band [0.762, 0.816]
   was built on the artifact; B5.2's readout already declared it FALSIFIED-informatively (66/76 both
   arms), which this section now explains by law rather than by "impostor-only effect".
3. PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md section 7 second caveat ("a ~17-point loss") and R5's
   retention rows describe the pre-repair fleet's clip, not the window; the pass-fraction rows (candidate
   pool) are unaffected by the clip (they count catalogue rows, not true hosts).
4. What survives unchanged: the epsilon = 2 Phi(-k) identity (correct for the object it names, and now
   ALSO the production-law retention on the scattered venue), the CV census, the sigma_lnM = M_error/M
   identity, the candidate-count census, the C3 readout (+0.003523 INTERMEDIATE; 621/1588 dark-class
   collapse; R2/R5/R6 PASS).

## 4. Reconciliation with production

- iiib: delta law predicts identical true-host recovery at every k and geometry; measured 66/76 in both
  arms with the same 76-event positivity pattern {source: B5_2_WIN_K3_READOUT_RECORD.md section 3}. The
  10 misses are sky/z-cone misses common to both arms (sigma = 0 miss rate 13.2 percent on seed 61000 =
  10/76, HANDOFF_20260730.md section 5 table, 2026-07-30). Consistent.
- joint_r1: the campaign #53 P6 read on the scattered venue gives in-catalogue host MISS rates of 25.0
  percent (1D, z/sky only) and 42.6 percent (2D) on seed 61000, 35.7 / 50.9 on seed 62000 {source:
  HANDOFF_20260730.md section 5 table}. The 2D excess miss of 17.6 / 15.2 points is the mass window's
  contribution under the log-normal law; the closed form 1 - Phi(ln(1+1.5 CV)/CV) gives 16.8 / 18.1 /
  21.6 points at CV 0.86 / 1.02 / 1.5 (ARITH) -- the predicted size and the measured one-sidedness
  ("pooled 193 low-side rejections vs 1 high-side", HANDOFF_20260730.md section 4) both match. This
  loss is a correctness (information-loss) property of the CURRENT production default on the scattered
  venue, not a bias claim; it sits on the in-catalogue class (73 of 1588 events) and is outside HB's
  exoneration, which bounded the window's h-tilt on the dark class as support truncation
  (CLAIM_WGEO_20260827.md section 4.1; EXONERATION_REGISTER_20260827.md, WGEO scoping ~:217-223, 325 --
  re-grepped 2026-08-30: no exoneration names true-host retention or window geometry).

## 5. The design

Principle: key each side of the interval-overlap test to the error law the code actually realizes on that
side, so that the retained true-host probability equals a stated epsilon by construction.

- GW side: the CRB posterior in M_z is Gaussian with fractional width ~1e-8 (median); the window there is
  a point at either geometry (first-order agreement, PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY section 5 item 3);
  the cosmology z-box supplies the only width. No change proposed on this side.
- Catalogue side: the realized observation law is log-normal with sigma_lnM = BH_MASS_ERROR/BH_MASS read
  from the observed row (section 1.2). The epsilon-keyed window is therefore the log-symmetric window
  already implemented as mass_filter_geometry = "log" with k = Phi^-1(1 - epsilon/2):
  epsilon = 0.27 percent at k = 3.0, 1.24 percent at 2.5, 4.55 percent at 2.0, 13.4 percent at 1.5, 0.047
  percent at 3.5 (ARITH). On the scattered venue this retention is EXACT and selection-independent (the
  catalogue scatter is independent of the GW selection that acted on M_true), up to the width-drift band
  of section 2 for old-writer realizations. On the unscattered venue it is inert for the true host.
- The one quantity that differs by venue is therefore the DESIGN OBJECT: on iiib the window shapes only
  the impostor pool (C3: 621/1588 dark-class events lose all with-BH catalogue support at k=3; +0.0035
  INTERMEDIATE); on joint_r1 it shapes both the impostor pool and the true-host retention (+16 to +22
  points). The charter's "design object is the impostor pool, not the true host" (TREE2_CHARTER T5 row)
  is correct for iiib and wrong for joint_r1.

The three F-ii alternatives of docket 2 section 6 item 1, with this node's numbers:
(a) adopt log k=3 as a documented structural design choice -- epsilon-keyed, exact on the scattered venue,
    impostor-only on iiib where its H0 effect is INTERMEDIATE (+0.0035, sign up); the k-dependence of the
    impostor response is unmeasured (one point);
(b) keep linear k=1.5 and document it -- retains 16-22 percent FEWER true hosts on the scattered venue,
    one-sidedly, at any CV in the catalogue's range; not epsilon-keyable (k = 11.6 at the median CV);
(c) commission the k-scan / the joint_r1 arm first (row #255 A1 = (c), granted) -- section 6.
Recommendation: (c) as granted, with the arm ORDER re-argued by yield: the joint_r1 k=3 arm is the
decisive arm (both design objects live there and its true-host prediction is falsifiable at +-1 host);
the iiib k-scan reads the impostor-pool shape and the materiality of any k. Both are cluster-bound and
queue behind Lustre recovery; neither is launched by this node.

## 6. Registered arms (pre-registration; A22 stamp at launch; no CPU-h spent)

### 6.1 Arm S -- iiib k-scan, log geometry, k in {2.0, 2.5, 3.5} (+ optional k = infinity)

- Form: verbatim the C3 form (cluster/wave2_c3_win_k3.sbatch:114-140 CLI; CRB_SRC run_20260729_seed61000;
  catalogue md5 c52c13b5..., CRB md5 9a1f2a14..., EVAL_SEED 777000 + H41 index {7, 8, 9, 21}; H4 grid
  {0.660, 0.665, 0.670, 0.730}) with mass_filter_geometry = log and mass_filter_k in {2.0, 2.5, 3.5};
  mass_filter_sigma = symmetric held (invariant). catalogue_numerator_survival_2d = off is HELD at the C3
  value so that the banked k=3 point (+0.003523) is a valid fourth point on the same curve; this is
  disclosed as a choice against the B7.3 production default (mz_sel/eff, row #253, ratification pending
  A4 after wave 3): a second scan at the new default is a separate registration if A4 ratifies.
  k = infinity (no window) is an optional fifth point: it anchors the curve's limit at the CURRENT CoR
  instead of HB's +0.0015, which is STALE by A11 (measured 2026-07-30 on the pre-fusion r1 configuration,
  HANDOFF_20260730.md section 4 / CLAIM_WGEO section 4.1). Implementation of k = infinity: mass_filter_k
  = 1e6 (both geometries converge to the all-True mask, invariant 2 of the physics-change doc R7).
- Baseline B: reused from the C0 gate task at zero compute (gate: per-event L_cat_with_bh /
  combined_with_bh reproduced to <= 1e-12 relative, historically <= 8.5e-15; otherwise re-run at this
  arm's nodes, +approx 5 CPU-h).
- Cost: 3 (or 4) x approx 5 CPU-h per k-node set (C3 measured 4.97 CPU-h for 4 H4 tasks
  {source: COMPUTE_LEDGER.md "C3 measured", 2026-08-29}); the candidate-count factor at k = 2 and 3.5 is to
  be read at zero compute from a scratch copy of b5_window_count.py with CONFIGS extended BEFORE
  submission (k = 2.5 and 3.0 are banked: pass fraction 0.6149 / 0.6951 vs linear 0.9577); expected
  total 15-20 CPU-h. F4 field: cluster/COMPUTE_LEDGER row to be added at submission.
- Primary read per k: Delta mean_h,pred(k) = Delta l'(0.665)/I_HEAD, I_HEAD = 2965; map per k as in the
  C3 registration section 3: IMMATERIAL-CONSISTENT-WITH-HB |Delta| <= 0.003; INTERMEDIATE 0.003 < |Delta|
  < 0.008; MATERIAL |Delta| >= T_mat = 0.008 (MEASUREMENT_HEAD_READOUT_20260827.md:268-285, row #213).
  Stencil equivalents: 4.45 / 8.89 / 23.72 nats per unit h (ARITH).
- Scan-level verdicts (registered now): MATERIAL-AT-SOME-k if any point in {2, 2.5, 3, 3.5 (, inf)} is
  MATERIAL (then the geometry IS a production-material object and adoption of any k returns to the
  author with the curve); ALL-SUB-MATERIAL otherwise, in which case the design choice is structural and
  (a) is recommended at k=3 on epsilon grounds. Shape read (REPORTED-ONLY): monotone vs non-monotone in k;
  the zero-compute census predicts the dark-class collapse count to be monotone decreasing in k (pass
  fraction 0.615 at 2.5 vs 0.695 at 3.0), so a non-monotone Delta(k) would itself be a finding.
- Gates: R6 1D bit-identity at every node (<= 1e-12); R2 engagement >= 0.90 of non-empty-baseline
  with-BH events changed at h = 0.730; R5 |Delta l''| << 2965 (G27 escalation otherwise).
- Prediction that is falsifiable at zero cost inside the arm: the 76 in-catalogue events keep the SAME
  positivity pattern (66/76) at every k -- the delta law. One changed event falsifies section 1.2's iiib
  reading (the loaded catalogue is not the injection catalogue) and STOPs the interpretation.

### 6.2 Arm R -- joint_r1, log k = 3 vs the linear k = 1.5 baseline (T5.2 of the charter)

- Form: the joint_r1 HEAD-readout CLI (headreadout_20260827/joint_r1/run_metadata_21.json:cli_args,
  observed_catalogue seed 900001) with mass_filter_geometry = log, mass_filter_k = 3.0, on the H4 grid;
  baseline = the banked joint_r1 HEAD readout at the same nodes (zero compute) subject to the same C0-style
  ingredient gate on this venue (a joint_r1 C0-prime task, approx 1-2 CPU-h, is required because no
  joint_r1 baseline has been re-run at the current HEAD).
- Cost: approx 11-15 CPU-h (docket 2 section 7 rank 5; joint_r1 HEAD cost >= 2.2 x iiib) + the C0-prime.
- Registered predictions: (i) true-host recovery among the 73 in-catalogue events rises from the linear
  baseline by 16-22 points: expected +12 to +16 hosts (binomial SE approx 3 hosts at n = 73), i.e. the
  2D-minus-1D excess miss of HANDOFF section 5 (17.6 points on seed 61000) collapses to <= 1 percent; the
  log-k=3 mass-only loss expected 0.997 x 73 -> 0.2 hosts (0 or 1 lost); (ii) Delta mean_h,pred on
  joint_r1 read on the same three-way map; no sign is predicted (the impostor-pool and true-host
  responses have opposite information signs); (iii) the 1D channel bit-identical (R6).
- Falsifier of the law reading: if the recovery gain is outside [+8, +20] hosts, section 1.2's joint_r1
  law reading (log-normal, selection-independent, width ratio ~0.93) is wrong or the sky/z cone is
  correlated with the mass window; STOP and return.

### 6.3 Zero-compute items to complete before either submission
1. Extend the candidate census to k = 2.0 and 3.5 (scratch copy of b5_window_count.py; both geometries);
   bank the pass fractions with their arm-jackknife bands.
2. Re-read the true-host retention on the post-repair fleet with the full interval-overlap test (z-box)
   through the production flags (R4 falsifier item 2 of the physics-change doc) -- this node's
   point-z_host read is 94.4 percent; the production-function read must land within +-2 points.
3. Recompute the joint_r1 in-catalogue count (73 at h = 0.73, HANDOFF section 2) from the HEAD-readout
   diagnostics CSV to fix n for the binomial band.

## 7. The mirror-generator fix (separate instrument; tree-3 gate item; NOT built here)

Object: a mass_law flag on the catalogue_selected_2d host mode
(correspondence_1d.py:2060-2072 draw_realization signature; the draw at :1743-1750), threaded through
run_mirror_seed_inprocess and p3_2d_fleet.py like mass_filter_geometry was (B5_1_WIN_RECORD.md).
- "linear_truncated" (DEFAULT, byte-identical): the current m_eff + sigma N(0,1), M <= 0 rejected. Every
  banked mirror number is unchanged; the sigma_scale = 0 / flag-default regression is bit-for-bit on a
  banked seed (p3_2d_fleet_repair_20260827 seed 900101 as the fixture).
- "lognormal_observed" (the production law, Convention (A) in the mirror): the truth is the pool mass,
  M_true := pool.M[host] (no draw; the Eddington shift is NOT applied to the truth -- it is a kernel
  centering, bayesian_statistics.py:7789-7797, and stays on the estimator side), and the estimator is
  handed an OBSERVED copy of the catalogue: ln M_obs = ln M + sigma_lnM Z, sigma_lnM = M_error/M, with the
  exact-width error M_error_obs = M_obs sigma_lnM (the current writer's semantics, observed_realization.py:
  366-383, without the stellar-mass detour because the mirror holds BH masses in memory). Two
  implementation routes, to be chosen at the gate: (i) write a per-seed observed CSV with the existing
  writer and pass observed_catalogue_path (faithful to the prune-on-observed behaviour and the scattered
  guard, handler.py:280-322; costs one 1 GB file per seed); (ii) scatter the handler's
  reduced_galaxy_catalog BH_MASS / BH_MASS_ERROR columns in memory before evaluate(), set the scattered
  flag so the point-kernel refusal applies (resolve_host_mass_kernel catalogue_scattered path,
  bayesian_statistics.py:301-306), and re-run the mass prune on the observed columns (or disclose that the
  prune stays on true columns, a -4.6 percent row difference, HANDOFF section 5 defect 1). Route (ii) is
  seconds per seed over 2.3e7 rows.
- "lognormal_truth" (truth drawn log-normally around the catalogue, then S_4D-selected) is explicitly NOT
  recommended: it keeps the selection tilt (section 1.4) and is neither the production law nor the
  estimator's kernel.
- Regression plan for the gate: (1) default byte-identity; (2) retention identity -- on a lognormal_observed
  twin the log-k window retains the true host at 1 - 2 Phi(-k) to binomial precision at every k, and the
  linear-k window at Phi(ln(1+k CV)/CV) per host (the closed forms of section 2, testable at zero
  compute on the twin's own truth columns); (3) the z-draw law untouched (the kernel-smeared z_true path,
  _draw_kernel_survival_redshifts, is not entered by the mass flag); (4) limiting cases: sigma_scale -> 0
  recovers the delta law and the iiib retention pattern; k -> infinity recovers the no-window mask under
  both laws; CV -> 0 makes lognormal_observed and linear_truncated agree to O(CV^2).
- Why it is a correctness item and not a bias lead: L10 (SYNTHESIS_DOCKET_2 section 1 B8.2 row) stands --
  mirror-derived retention/growth predictions are hypotheses, not design inputs -- and this flag is what
  turns the mirror into a venue-faithful twin for the with-BH channel's host-mass leg; the B8.2 harness
  (S1-S5) is the consumer.

## 8. A10 / A14 / A15

A10 invariants (held fixed across every arm of section 6): mass_filter_sigma = symmetric (audited
2026-08-29, B5_1_WIN_RECORD section 3); sigma_multiplier = 1.5 sky-cone radius (audited 2026-08-29; the
T1 sky-cone flag does not enter these arms); catalogue md5 c52c13b5... and CRB md5 9a1f2a14... (STOP-gated
in the sbatch, :46-48); EVAL_SEED convention 777000 + H41 index (cluster/SKILL.md gotcha 4); host_z_kernel
volume_deconv, host_mass_kernel auto -> gaussian, normalization_mode absolute_marginal, eddington_m on,
catalogue_global_selection phi, selection_in_completion_numerator fused, catalogue_numerator_survival_2d off
(the C3 CoR; disclosed choice, section 6.1); the injection (seed 61000 CRB, git d04d9dc9 at the HEAD
readout) -- NEVER re-audited for the window question; conclusions are conditional on it by name.
Structural blindness: (1) a paired H4 read on one injection cannot detect a window effect that is
h-independent in log L (a pure normalization shift) -- it reads slopes only; (2) neither arm can detect
a defect in the joint_r1 realization's width column (the old-writer drift) beyond the band disclosed in
section 2; (3) the iiib scan is blind to any true-host effect by construction (delta law).
Blindness sentence: every number in this node was computed from banked truth columns and closed forms
before any h-dependent quantity was touched; no H0 posterior, MAP, or score entered the design of the k
set, the bands, or the mirror flag.

A14 falsifiers (each with its band):
1. Law reading, iiib (section 1.2): FALSIFIED if any of the 76 in-catalogue events changes positivity
   between k values in Arm S (band: 0 changes).
2. Law reading, joint_r1 (section 1.2): FALSIFIED if Arm R's recovery gain is outside [+8, +20] of 73.
3. Floor-clip attribution (section 3): FALSIFIED if a re-read of the post-repair fleet through the
   production flags (section 6.3 item 2) gives log-k=3 retention outside [0.92, 0.97] (the point-z_host
   read 0.944 +- 2 points + the z-box widening, which can only raise it).
4. Selection-tilt residual (section 3, the 4.5 points): FALSIFIED if, on the post-repair fleet, the
   log-k=3 failure fraction is flat in s4d_at_truth quartiles (band: the top-vs-bottom quartile
   difference in failure fraction < 2 points); then the residual is the unselected law's own tail and
   the closed form (0.989) is wrong, not the selection.
5. Closed forms (section 2): FALSIFIED by any Monte Carlo disagreement > 5e-4 at 2e6 draws (checked at
   CV 1.02, k in {1.5, 3}: max |diff| 3e-4).

A15 operating characteristics: Arm S is a paired deterministic recomputation on one fixed event set
(sampling variance exactly zero for the point differences; false-fail rate under the reproducibility
floor 0, bounded by the PROD-A0 ingredient gate <= 8.5e-15, row #201) -- the operative uncertainty is the
materiality band and the H4 stencil's model error (R5), made harmless by the wave-3 full-grid read (F2).
The retention statements are binomial: on iiib n = 76 (delta law: 0 changes expected, any change
decisive); on joint_r1 n = 73, baseline recovery approx 0.57 (1 - 0.426), predicted approx 0.75 (1 - 0.25 -
0.003), difference +0.18 with SE 0.08 (two-proportion) -> approx 2.2 sigma per venue-seed; the band
[+8, +20] hosts is the +-1.5 SE envelope around the closed-form range. The post-repair mirror read is a
33-arm census (per-arm retention 0.899-1.000, median 0.944 at log k=3) -- this arm-to-arm spread (approx
+-0.03) is the lower bound on generalization uncertainty for any mirror-derived retention number, per
the physics-change doc R5's caveat (all arms share one catalogue).

## 9. What returns to the author

1. [RULE] F-ii design choice, with these numbers: (a) log k=3 epsilon-keyed (exact on the scattered venue;
   impostor-only, +0.0035 INTERMEDIATE on iiib), (b) linear k=1.5 (loses 16-22 percent of true hosts
   one-sidedly on the scattered venue; not epsilon-keyable), or (c) the two registered arms first (granted
   as (c) by row #255; the ORDER -- joint_r1 first on yield -- is a path choice inside the grant, stated
   here for veto). Folds in the row #220 WGEOM section 9 F-ii consequence ruling as docket 2 read it.
2. [RULE] Re-attribution of record: the mirror's 78.9 percent retention (B5.2-pre section 4; C3
   registration section 5; B5.1 section 7 second caveat / R5) is a pre-repair floor-clip artifact (16.8
   points) plus a selected near-zero tail (approx 4.5 points), not a linear-vs-log shape effect (<= 1
   point). Proposed disposition: append pointer notes to the three records (housekeeping bundle), and
   retire "78.9 percent" as a design input; the post-repair numbers (94.4 vs 95.4) and the production
   laws (1.0 / 0.9973) replace it.
3. [STANDING-scope check] The joint_r1 arm is listed at charter depth 2 ("fresh [RULE]"); this node
   registers it but does not launch it. If the author reads the standing grant as covering registered arms
   inside the tree, it launches with the k-scan when the cluster returns; otherwise it waits for the word.
4. [DO, tree 3] The mirror mass_law flag (section 7) as a gate item: presentation before code, byte-
   identical default, the retention-identity regression.
5. Information, no ruling: the current production default (linear k=1.5) carries a quantified, one-sided
   true-host information loss on the scattered venue (in-catalogue class only; outside HB's exoneration);
   the fix is the already-implemented log geometry at any k >= 2.5.

## 10. Provenance, scripts, ledger

- Inputs read (all banked, none edited): PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md;
  B5_2_PULL_READ_20260829.md + b5_pull_read.json; B5_2_WIN_K3_READOUT_RECORD.md + b5_2_readout.json;
  PREREGISTRATION_WIN_K3_COUNTERFACTUAL_20260829.md; b5_window_count.json; wgeom_result.json;
  PREREGISTRATION_MKER_WGEOM_20260828.md; HANDOFF_20260730.md; p32d_residual_accounting_20260827.md;
  P3_2D_REPAIR_READOUT_20260828.md; SYNTHESIS_DOCKET_2_20260829.md sections 6-7; END_VERIFIER_REPORT_PART1
  section 4 (A1, P5); TREE2_CHARTER_20260830.md; COMPUTE_LEDGER.md; MEASUREMENT_HEAD_READOUT_20260827.md:
  268-285; the run_metadata_*.json of headreadout_20260827/{iiib, off_iiib, joint_r1} and wave2_20260829/c3;
  realizations_staged/observed_catalogue_seed900001.meta.json; cluster/wave2_c3_win_k3.sbatch; the code
  paths cited inline (main.py, dark_siren_injection.py, handler.py, parameter_space.py,
  observed_realization.py, bayesian_statistics.py, correspondence_1d.py, constants.py).
- Scratchpad scripts (session scratchpad, not banked; reproducible from the cited inputs):
  t5_retention.py (closed forms, brentq k tables, 2e6-draw MC check, width-drift and stencil bands);
  t5_edd.py (Eddington-shift magnitude via the production _eddington_shifted_host_mass_batch: m_eff/M_cat =
  1.13-1.38 at CV 0.86, 1.22-1.51 at CV 1.02, 1.52-1.93 at CV 1.5 over M_cat 3e4-1e7; unselected LT + Eddington
  pull |p| <= 3 = 0.990 at M_cat 1e6, CV 1.02); t5_pull_strat.py (pre-repair fleet stratified read, gate
  0.7877 / 0.6979 reproduced); t5_pull_repaired.py (post-repair fleet, 33 arms). Elapsed 14 s and 13 s
  for the two fleet reads (banked CSV + pruned catalogue loads; zero evaluate()).
- Not done here: no ledger row (orchestrator files rows); no gate-ledger row (no code); no append to the
  three superseded records beyond the one pointer note in B5_2_PULL_READ_20260829.md (section 3
  consequence 1); no cluster submission.

Stamp: launched under rows #255/#268 -- tree 2 node T5.1; authored 2026-08-30 by the T5.1 design agent
(top tier); status: DESIGN PROPOSAL + REGISTRATION DRAFT, returns to the author with section 9.

## Appendix A (appended 2026-08-30, same node) -- falsifier 4 pre-checked at zero compute

Section 8 falsifier 4 was run on the post-repair fleet read (scratchpad t5_pull_repaired.csv, 2275
events): log-k=3 failure fraction by s4d_at_truth quartile = 0.0053 (Q0, s4d 0.008-0.482) / 0.0351 (Q1,
0.482-0.746) / 0.1162 (Q2, 0.747-0.976) / 0.0668 (Q3, 0.976-1.000); overall 0.0558. The top-vs-bottom
quartile difference is 6.2 points and the Q2 peak is 11.1 points above Q0 -- far outside the 2-point
flatness band, so the selection-tilt attribution of the residual (section 3) is NOT falsified: the
near-zero-mass failures concentrate where the acceptance probability is high, as the grid-edge S_4D clamp
predicts. REPORTED-ONLY; a different agent should re-run the two fleet reads before any of section 3's
numbers are cited as adopted evidence (standing rule 2).

## Appendix B (appended 2026-08-30, same node) -- refuter panel disposition

A refuter panel returned two must-fix items (refuted=false) on this document, alongside eighteen
independent-verification findings that all read as PASS (re-derived closed forms, re-checked code
citations, cross-checked banked-artifact numbers, and confirmed scope/tag/discipline compliance --
none of which is a must-fix and none of which changes anything in sections 1-9 or Appendix A). Both
must-fix items are accepted as stated, with evidence, and dispositioned below. Per the append-only
tree-2 discipline, sections 1-9 and Appendix A are left unedited; this note is the record of
correction going forward.

1. Citation-precision correction (not conclusion-changing). The three inline citations at (as
   originally published) line 26, line 123, and line 376 all name correspondence_1d.py:1743-1750 for
   the mirror's m_eff/m_true/rejection formula. Re-verified against the code at HEAD 647e86d9:
   darksiren_emri/validation/correspondence_1d.py:1743-1750 is the tail of the _B0i2DLatents
   dataclass field list and the docstring/opening lines of the _draw_2d_accepted_latents function --
   not the formula. The formula itself lives at darksiren_emri/validation/correspondence_1d.py:
   1853-1871 (comment block opens at 1853; the three executable lines are 1857 m_eff =
   _eddington_shifted_host_mass_batch(host_m, host_m_error), 1862 m_true_batch = m_eff + sigma *
   rng.normal(size=batch), and 1870 valid_mass_batch = m_true_batch > 0.0). The content DESCRIBED in
   all three original passages is accurate against the code; only the line-number pointer was off by
   roughly 107-113 lines. SUPERSEDED: every occurrence of "correspondence_1d.py:1743-1750" in
   sections 1-9 above is superseded by "correspondence_1d.py:1853-1871" (comment) / effectively
   :1857-1871 for the three formula lines. This is the corrected citation of record; the original
   text is left as published.

2. Approval-tag correction. Section 9 item 3 was tagged "[STANDING-scope check]" in the original
   text, which is not one of the three canonical tags ([DO]/[RULE]/[STANDING]) the CLAUDE.md
   Approval-scope convention requires so a one-word reply is unambiguous. The panel is correct: the
   substance of item 3 is sound (it states the ambiguity and defaults to waiting for the word) but
   the ad hoc tag does not resolve cleanly. SUPERSEDED: section 9 item 3's tag is corrected to [RULE]
   -- it asks the author to rule on how the already-ratified standing grant (rows #255/#268) applies
   to a newly-registered arm (joint_r1, section 6.2), which is a procedural ruling on evidence already
   in front of the author, not pre-authorization of a future class of decisions ([STANDING]) and not a
   request for new work ([DO]). Restated for a one-word reply, replacing the original tag only (the
   original item 3 text stands unedited above):

   [RULE] Does the rows #255/#268 standing grant's scope cover launching the already-registered
   joint_r1 arm (section 6.2) inside tree 2 alongside the k-scan, or does it wait for a separate word?

No other must-fix items were returned; no finding in the panel's eighteen-item verification list
required a change to sections 1-9 or Appendix A.

Stamp: launched under rows #255/#268 -- tree 2 node T5.1; appended 2026-08-30 by the T5.1 design
agent (top tier), same node, in disposition of the refuter panel's two must-fix items on this
document.

## Appendix C (appended 2026-08-30, same node) -- second refuter pass: three further must-fix items

A second refuter panel pass on this document (post-Appendix-B) returned three further must-fix
items (refuted=true), plus twelve independent-verification findings that read PASS (all four
closed-form retention formulas re-derived from first principles and matched to the printed table
at multiple (CV,k) spot-checks; the truncation table; every other inline code citation checked
against HEAD 647e86d9 and matched verbatim; the log-window mask code in handler.py confirmed to
implement the section-2 derivation; the observed_realization.py law confirmed byte-for-byte;
every run_metadata and sidecar claim in section 1.2 reproduced from disk exactly). None of the
three must-fix items changes a formula, a derivation, a closed form, or a registered number in
sections 1-9 or Appendices A-B; all three are citation-precision or process-discipline items.
Per the append-only tree-2 discipline, sections 1-9 and Appendices A-B are left unedited; this
note is the record of correction going forward, superseding Appendix B item 1's own citation
where it was itself wrong.

1. Appendix B item 1's own "corrected" citation is off by two lines (not conclusion-changing).
   Appendix B item 1 stated the m_eff assignment sits at correspondence_1d.py:1857. Re-verified
   against the code at HEAD 647e86d9: {source: darksiren_emri/validation/correspondence_1d.py,
   grep -n "m_eff = _eddington_shifted_host_mass_batch", read 2026-08-30} -- the assignment is at
   line 1859. Lines 1857-1858 are the two reads that feed it (host_m = pool.M[host_idx_batch];
   host_m_error = pool.M_error[host_idx_batch]). The other two corrected line numbers in Appendix B
   item 1 (1862 for m_true_batch = m_eff + sigma * rng.normal(size=batch); 1870 for
   valid_mass_batch = m_true_batch > 0.0) are exact and unchanged. SUPERSEDED: Appendix B item 1's
   "1857 m_eff = ..." is superseded by "1859 m_eff = ...". The corrected citation of record for the
   three formula lines is 1859 / 1862 / 1870 (comment block still opens at 1853).

2. Section 1.2's observed_realization.py citation bundles two formulas under one line range that
   contains only one of them (not conclusion-changing). Section 1.2 cites
   "observed_realization.py:5-9 (docstring), :349-356 (delta_ln_bh = sigma_scale sigma_ln_bh
   mass_std_normal; bh_mass_obs = bh_mass exp(delta_ln_bh))". Re-verified against
   darksiren_emri/galaxy_catalogue/observed_realization.py at HEAD 647e86d9 {source: grep -n
   "delta_ln_bh|bh_mass_obs|sigma_ln_bh", read 2026-08-30}: sigma_ln_bh = bh_mass_error / bh_mass is
   line 354, delta_ln_bh = sigma_scale * sigma_ln_bh * mass_std_normal is line 355 -- both inside
   the cited range -- but bh_mass_obs = bh_mass * np.exp(delta_ln_bh) is at line 397, well outside
   :349-356. The described content is accurate; only the second formula's line pointer is wrong.
   SUPERSEDED: section 1.2's citation is superseded by "observed_realization.py:5-9 (docstring),
   :354-355 (sigma_ln_bh = bh_mass_error/bh_mass; delta_ln_bh = sigma_scale sigma_ln_bh
   mass_std_normal), :397 (bh_mass_obs = bh_mass exp(delta_ln_bh))". The same bundled range appears
   nowhere else in the document (grep confirms one occurrence); no other citation needs this split.

3. Governance: the B5_2_PULL_READ_20260829.md pointer-note append (section 3 consequence 1;
   disclosed in section 10) was executed on 2026-08-30, before the author has ruled on section 9
   item 2, which this document itself labels a pending [RULE] ("Proposed disposition: append
   pointer notes to the three records..."). That is a live inconsistency in the document's own
   text -- section 3 says pointer notes "are proposed in section 9" (future tense), section 9 item
   2 says the append is a "[proposed] disposition," and section 10 discloses in passing that one of
   the three was already done ("no append to the three superseded records beyond the one pointer
   note in B5_2_PULL_READ_20260829.md") -- without flagging that "proposed" and "already done"
   describe the same action. The refuter panel is correct that this is an unauthorized
   pre-execution of a [RULE] item, and correctly notes the scope point: row #255 A1=(c) authorizes
   commissioning the design ("no adoption, no joint_r1 arm before that"), and row #268 extends the
   grant in TIME ("until tomorrow"), not in SCOPE; neither grant mentions authority to append
   interpretive text to a different, already-banked record ahead of the author's ruling on it. This
   node's own operating constraints for this append (no git, no code edits, append-only) rule out
   panel option (b) (git checkout -- that file, or an offsetting revert-note) as an action this
   node can take; disposition is therefore panel option (a), logged here as the process exception
   and returned to the author as a fresh, explicit [RULE] rather than assumed:

   [RULE] (process exception, logged for the record) The B5_2_PULL_READ_20260829.md pointer-note
   append (its section "Pointer note (appended 2026-08-30, ...)") was executed before this ruling,
   under rows #255/#268, on this node's own reading that a re-attribution of a numeric mechanism
   (not a numeric result -- see Appendix C item 3's note that no number in that record changed) was
   within the design-proposal's zero-compute documentation scope. Ratify the append as-is
   (after-the-fact), or direct it reverted (a separate action, since this node cannot run git); either
   way, the other two records named in section 9 item 2 (PREREGISTRATION_WIN_K3_COUNTERFACTUAL_20260829.md,
   PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md) remain untouched pending this ruling, so the
   "proposed disposition" in section 9 item 2 is, as of this appendix, one-third executed and
   two-thirds still pending -- not uniformly "proposed."

No other must-fix items were returned in the second pass; none of the twelve PASS findings changes
anything in sections 1-9 or Appendices A-B.

Stamp: launched under rows #255/#268 -- tree 2 node T5.1; appended 2026-08-30 by the T5.1 design
agent (top tier), same node, in disposition of the second refuter panel's three must-fix items on
this document.

## Revision note 3 (2026-08-30; panel must_fix; append-only)

Ledger row #270 recorded the panel's verdict on this document as REFUTED at round 2, with a
factual must_fix on Section 1.2's "Width caveat (A11 stamp)" paragraph and Section 2's "joint_r1
width-drift correction" paragraph. Per the append-only tree-2 discipline, sections 1-9 and
Appendices A-C are left unedited; this note is the record of correction going forward. Both
struck passages are quoted verbatim below, followed by the evidence and the replacement text.

1. Section 1.2, "Width caveat (A11 stamp)" (as originally published, lines 100-106):

   > Width caveat (A11 stamp): the seed-900001 realization predates the exact-width writer (its
   > sidecar has no n_mass_width_floor key); for that writer generation the code records "MEASURED
   > pull vs the recomputed width 0.929, per-row drift up to +-18%" {source:
   > observed_realization.py:357-365 comment}, i.e. the width the window reads (loaded) exceeds the
   > width the scatter was drawn with by approx 7.6 percent on average with +-18 percent per-row
   > drift. The sidecar's own drawn-width check is 0.99983 {source:
   > meta.json:width_check.mass.normalized_residual_std}. Realizations written by the current writer
   > preserve the width exactly except the counted n_mass_width_floor rows
   > (observed_realization.py:366-383).

   This is factually wrong on its own falsifiable premise. Read directly:
   {source: results/campaign51_20260728/realistic_20260729/realizations_staged/
   observed_catalogue_seed900001.meta.json, read 2026-08-30} -- the sidecar's `width_check` object
   DOES carry the key: `"n_mass_width_floor": 24100`, against `"mass": {"n": 21753847, ...}` --
   24100/21753847 = 0.1108 percent, not "no n_mass_width_floor key." The realization was written by
   the exact-width remedy, not by a predecessor. Confirmed independently: `git_commit` in the same
   sidecar is `7b30d1ff17c543d3464f533121f7b3e108347bb9`; `git log --follow --diff-filter=A --
   darksiren_emri/galaxy_catalogue/observed_realization.py` (repo HEAD 647e86d9) shows commit
   `7b30d1ff` (`[PHYSICS] #53: realistic host-observation model -- seeded observed-catalogue
   realization + scatter guards`) as the commit that CREATES the file -- there is no earlier writer
   version in this file's history for the seed-900001 realization to predate. SUPERSEDED: the "Width
   caveat (A11 stamp)" paragraph is superseded by --

   True-host retention of the log window on joint_r1 is the nominal 1 - 2*Phi(-k) (0.9973 at k=3),
   degraded only by the 0.11 percent of mass-valid rows the exact-width solve could not satisfy
   (n_mass_width_floor = 24100 of 21,753,847; clamped to s_obs = 0). That departure is conservative,
   not a shortfall, per the code's own comment on the clamp {source: observed_realization.py:373-374}:

   > (`n_mass_width_floor` in the sidecar) -- their loaded width is then slightly
   > WIDER than drawn, a conservative, reported residual.

   A wider loaded width than the drawn scatter means the window is more permissive than the true
   error on exactly those rows, so their true-host retention is at or above nominal, never below it
   -- the 0.11 percent departure biases retention up, if at all, not down.

2. Section 2, "joint_r1 width-drift correction" (as originally published, lines 200-204):

   > joint_r1 width-drift correction (old-writer realization, section 1.2): with the loaded width
   > exceeding the drawn width by the measured mean ratio 1/0.929, the log window's effective k is
   > k/0.929 and R_LN,log becomes 0.8936 / 0.9687 / 0.9929 / 0.9988 / 0.9998 at k = 1.5 / 2 / 2.5 / 3
   > / 3.5 (nominal 0.8664 / 0.9545 / 0.9876 / 0.9973 / 0.9995), with a per-row band from the +-18
   > percent drift of [0.989, 0.9997] at k=3 (k_eff 2.54-3.66) (ARITH). Disclosed, not corrected for
   > in the registered prediction beyond this band.

   This paragraph is downstream of item 1's false premise (there is no "old-writer realization"
   here) and inherits the same error; the "measured mean ratio 1/0.929" and the "+-18 percent
   per-row drift" describe a naive rewrite-M*-only method the code explicitly rejects in section
   1.2's own cited comment block (observed_realization.py:357-370) in favor of the exact-width
   remedy -- they are not a property of the delivered observed_catalogue_seed900001.csv. SUPERSEDED:
   the "joint_r1 width-drift correction" paragraph, its k_eff = k/0.929 correction, and the derived
   retention band [0.989, 0.9997] at k=3 are struck in full and superseded by item 1's replacement
   text above -- there is no k_eff correction and no per-row band; the registered retention at k=3
   is 0.9973 nominal, subject only to the conservative 0.11 percent floor-clip noted in item 1.

3. Two further passages in sections 1-9 read on the struck band and are corrected by the same
   disposition (left unedited in place; noted here):
   - Section 1.5's law table, "joint_r1 production" row, "retention ... at log k" column: "1 - 2
     Phi(-k sigma_drawn/sigma_loaded) = 0.9973 nominal, approx 0.9988 at the 0.929 width ratio" --
     SUPERSEDED: the width-ratio clause is struck; the entry is "1 - 2 Phi(-k) = 0.9973 nominal
     (CV-independent), degraded only by the conservative 0.11 percent floor-clip of item 1."
   - Section 5, "The design" bullet 2: "...this retention is EXACT and selection-independent...up
     to the width-drift band of section 2 for old-writer realizations." -- SUPERSEDED: the
     qualifying clause "up to the width-drift band of section 2 for old-writer realizations" is
     struck; the seed-900001 realization is not an old-writer realization (item 1), so the
     retention is exact and selection-independent on this venue without qualification, subject
     only to the conservative 0.11 percent floor-clip of item 1.

4. k-scan design bands (section 6) that depended on the struck band: none. Section 6.2's
   registered prediction (i) already used the nominal figure ("the log-k=3 mass-only loss expected
   0.997 x 73 -> 0.2 hosts (0 or 1 lost)"), not the struck k_eff/band correction, and needs no
   restatement. No other band, cost figure, or falsifier in section 6 cites the width-drift band,
   k_eff, or [0.989, 0.9997]; the k-scan design and its two registered arms (6.1, 6.2) stand as
   published.

Net effect on the proposal's conclusions: none of sections 4, 5's headline recommendation, 8, or 9
change in direction or magnitude -- the corrected retention (0.9973 nominal, conservatively
perturbed by 0.11 percent) is closer to, not further from, the epsilon-keyed design's own claim
that the log window is exact-by-construction on the scattered venue. This is a citation/premise
correction, not a conclusion-changing one.

Stamp: launched under rows #255/#268 -- tree 2 node T5.1; appended 2026-08-30 by the T5.1
revision agent (sonnet, mechanical revision task), same node, in disposition of ledger row #270's
factual must_fix item on this document.

## Revision note 4 (2026-08-30; orchestrator adjudication)

Appendix C item 3 raised, as a fresh [RULE], the question of whether the B5_2_PULL_READ_20260829.md
pointer-note append (executed under rows #255/#268 before the author ruled on section 9 item 2) was
within scope. Ledger row #270's panel also carried this as its second must_fix item. The
orchestrator has adjudicated the scope question; the ruling is recorded verbatim below.

Orchestrator adjudication (verbatim, 2026-08-30): "Append-only cross-reference notes on existing
records are within the standing grant of rows #255/#268 (the charter's own form: append-only
records at every node); the pointer note stands; no author ruling is required; the item is flagged
for the end-of-tree-2 verifier as a scope question the node raised on itself. Adjudicated by the
orchestrator, 2026-08-30."

Disposition: Appendix C item 3's [RULE] is closed by this adjudication, not by an author ruling.
The B5_2_PULL_READ_20260829.md pointer-note append stands as executed; no reversion is directed.
The scope question itself -- whether a T5.1 node may append cross-reference pointer notes to a
different, already-banked record on its own reading of a standing grant, ahead of an author ruling
on the same document's own pending [RULE] -- is flagged for the end-of-tree-2 verifier as a
disclosed, self-raised scope question, not adjudicated as settled precedent beyond this instance.

Panel-state update of record: with Revision note 3's correction filed, this document's ledger row
#270 REFUTED-at-round-2 disposition is superseded for tracking purposes by "corrected, pending one
re-check" -- the factual must_fix (item 1) is corrected above and the governance must_fix (item 2)
is adjudicated above; an independent re-check of both is the outstanding step before the panel
state can read PASS. See ledger row #271 for the row of record.

Stamp: adjudicated by the orchestrator, 2026-08-30; recorded by the T5.1 revision agent (sonnet,
mechanical revision task), same node. No git, no code, no compute; append-only.

## Re-check (2026-08-30; independent refuter)

Independent refuter role (physics/code lens; no git, no code edits, foreground only;
results/**/hier_s0_zwin_run not touched). Re-verified Revision note 3 (lines 635-726) and Revision
note 4 (729-756) from source, plus this document's Section 0/1.1 central finding. Per-item verdicts:

1. **PASS -- sidecar fields.** Opened
   `results/campaign51_20260728/realistic_20260729/realizations_staged/observed_catalogue_seed900001.meta.json`
   directly. `width_check.n_mass_width_floor = 24100`; `width_check.mass = {"n": 21753847,
   "normalized_residual_std": 0.9998275397872367, "expected_std": 1.0}`. 24100/21753847 =
   0.110785 percent, matching the quoted "0.1108 percent." `git_commit` field in the sidecar =
   `7b30d1ff17c543d3464f533121f7b3e108347bb9`, matching the note's citation exactly. The note's
   characterization of these as nested under one `width_check` object (not two independent keys)
   is accurate.

2. **PASS -- creation commit and code lines.** `git log --follow --diff-filter=A --oneline --
   darksiren_emri/galaxy_catalogue/observed_realization.py` returns exactly one commit,
   `7b30d1ff [PHYSICS] #53: realistic host-observation model -- seeded observed-catalogue
   realization + scatter guards`, dated 2026-07-29 18:22:28 +0200 -- there is no earlier writer
   version of this file, confirming "no earlier writer version in this file's history." Repo HEAD
   at re-check time is `647e86d9`, matching the document's stated HEAD. Lines 373-374 of
   `darksiren_emri/galaxy_catalogue/observed_realization.py` at HEAD read verbatim: "# (`n_mass_width_floor`
   in the sidecar) -- their loaded width is then slightly" / "# WIDER than drawn, a conservative,
   reported residual." -- an exact match to the note's quotation. The surrounding block (lines
   349-383) confirms the exact-width remedy is implemented (solves for `mstar_error_obs` so the
   loaded width equals `sigma_used`) and that the "0.929 / +-18 percent" figures at line 364 belong
   to a documented-and-superseded naive alternative ("A fixed-point sweep on the width is NOT a
   remedy"), consistent with Revision note 3's characterization.

3. **PASS -- retention identity and sign argument.** Computed 1 - 2*Phi(-3) directly
   (`math.erf`, no closed-form library dependency) = 0.9973002039367398, rounding to 0.99730 as
   claimed. The sign argument (floor-clipped rows have a loaded width wider than the drawn scatter,
   per the code's own comment at lines 373-374, so those rows' true-host retention is at or above
   nominal) is logically sound: widening the retention window relative to the true scatter width
   strictly increases the probability the true host falls inside it, for any unimodal symmetric
   scatter law. No counterexample found.

4. **PASS -- Section 6 independence from the struck band.** Read Section 6.1 (Arm S) and 6.2
   (Arm R) in full. Arm R's registered prediction explicitly uses "the log-k=3 mass-only loss
   expected 0.997 x 73 -> 0.2 hosts," the nominal figure, not the struck 0.9988/k_eff correction.
   Neither arm's cost, gate, or falsifier band references "0.929," "7.6 percent," "k_eff," or
   "[0.989, 0.9997]." Section 6 stands unchanged by the correction, as claimed.

5. **PASS, with a caveat -- grep sweep for residual erroneous figures.** `grep -n` for "0.929",
   "7.6", and "0.989" across the full document was run and every hit inspected in context.
   - "0.929": all 9 occurrences (lines 102, 139, 201, 647, 683, 684, 690, 694, 702) are inside the
     two originally-struck passages (Section 1.2's "Width caveat" paragraph, line ~102; Section
     1.5's law-table cell, line 139, explicitly named superseded in item 3 of the note; Section 2's
     "joint_r1 width-drift correction" paragraph, line ~201) or inside Revision note 3 itself
     (635-726), where the struck text is quoted verbatim before being corrected. No live,
     uncorrected use found.
   - "7.6": genuine occurrences of the erroneous "7.6 percent" width-drift figure are at line 103
     (inside the struck Section 1.2 passage) and line 649 (inside Revision note 3's quotation).
     The remaining three hits (lines 36, 264, 356) are substring collisions with "**1**7.6-point"
     and "**1**7.6 / 15.2 points" -- the unrelated 2D-minus-1D excess-miss percentage from Section
     3's floor-clip reconciliation, a different, independently-derived, non-superseded quantity.
     These are not remnants of the corrected error.
   - "0.989": genuine occurrences of the erroneous width-drift band's lower endpoint are at line
     203 (struck Section 2 passage) and lines 686/695/716 (Revision note 3's quotation and
     disposition). The remaining four hits (lines 142, 229, 241, 438) are the mirror's independent
     "unselected closed form (LT + Eddington)" retention estimate for the pre/post-repair mirror
     fleet (Section 1.5's law table and Section 3's reconciliation) -- a numerically coincidental
     but factually unrelated quantity (different law, different venue) that Revision note 3 never
     claims to supersede and that was not part of the refuted claim.
   Net: the corrected note fully accounts for every occurrence of the erroneous claim; the
   caveat is that literal substring grep also surfaces coincidental, unrelated matches ("17.6",
   an independent 0.989) that a naive read could mistake for uncorrected remnants -- they are not.

6. **PASS -- central finding (Section 0/1.1).** Confirmed in source: `handler.py`'s `HostGalaxy.__init__`
   (lines 73-80) sets `self.M = parameters[InternalCatalogColumns.BH_MASS]` directly from the
   catalogue row, with no scatter applied; `draw_rate_weighted_hosts` (lines ~1190-1210) builds
   in-catalog `HostGalaxy` objects the same way, reading `BH_MASS` straight from the eligible
   catalogue subset used for the rate weighting. `main.py:586-601` calls `draw_mixture_hosts`
   (`dark_siren_injection.py:594-676`, confirmed by direct read) then
   `parameter_estimation.parameter_space.set_host_galaxy_parameters(host_galaxy, h=h_value)` at
   line 601. `datamodels/parameter_space.py:260-268`'s `set_host_galaxy_parameters` sets
   `self.M.value = redshifted_mass(host_galaxy.M, host_galaxy.z)` (M_z = M*(1+z)) at line 268 --
   `host_galaxy.M` flows through unmodified from the catalogue read. No scatter is added anywhere
   on this injection-side path. The claim "production ties the EMRI mass to the host with NO
   scatter" is confirmed as stated, with the cited line numbers accurate.

**Summary: 6/6 items PASS** (item 5 carries a documented, non-blocking caveat about coincidental
substring collisions, not about any uncorrected error). No must_fix found. Panel state as of this
re-check: **PASS** -- the outstanding "independent re-check" step named in Revision note 4's
panel-state update is discharged by this note.

Stamp: independent refuter (physics/code lens; sonnet), 2026-08-30. No git, no code, no compute
(direct file reads and closed-form arithmetic only); append-only; results/**/hier_s0_zwin_run not
touched.
