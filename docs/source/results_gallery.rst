Results Gallery
===============

Publication figures from the production dark-siren H\ :sub:`0` campaign
``run_20260620_seed500_phase50`` — a single homogeneous catalogue of **1385**
simulated EMRI detections evaluated on an **83-point** super-dense H\ :sub:`0`
grid (injected truth ``h = 0.73``).

Combined-posterior MAP estimates:

* **Without** M\ :sub:`z` (1D): ``h = 0.737`` (+1.0 %)
* **With** M\ :sub:`z` (2D): ``h = 0.732`` (+0.3 %)

All figures are produced by the package itself via
``python -m master_thesis_code <simulations_dir> --generate_figures <simulations_dir>``
(the PDFs shipped on the cluster are converted to PNG for web display).

H\ :sub:`0` inference
---------------------

.. figure:: figures/fig01_h0_posterior_combined.png
   :width: 90%
   :alt: Combined H0 posterior

   **fig01 — Combined H₀ posterior.** Catalogue-combined posterior on H₀ for both
   mass conventions, with the injected truth marked.

.. figure:: figures/fig02_event_posteriors.png
   :width: 90%
   :alt: Per-event posteriors

   **fig02 — Per-event posteriors.** Individual single-event H₀ likelihoods with
   the combined posterior overlaid.

.. figure:: figures/fig08_h0_convergence.png
   :width: 90%
   :alt: H0 convergence

   **fig08 — H₀ convergence.** MAP H₀ and its credible interval as a function of
   the number of stacked detections.

Detection catalogue
-------------------

.. figure:: figures/fig03_snr_distribution.png
   :width: 90%
   :alt: SNR distribution

   **fig03 — SNR distribution.** Distribution of signal-to-noise ratios across the
   detected catalogue.

.. figure:: figures/fig04_detection_yield.png
   :width: 90%
   :alt: Detection yield

   **fig04 — Detection yield.** Detected EMRIs as a function of source parameters.

.. figure:: figures/fig09_detection_efficiency.png
   :width: 90%
   :alt: Detection efficiency

   **fig09 — Detection efficiency.** The selection function / detection efficiency
   used in the completeness correction.

.. figure:: figures/fig05_sky_localization.png
   :width: 90%
   :alt: Sky localization

   **fig05 — Sky localization.** Sky-localization areas for the detected events.

.. figure:: figures/fig11_distance_redshift.png
   :width: 90%
   :alt: Distance-redshift relation

   **fig11 — Distance–redshift.** Luminosity distance versus redshift for the
   catalogue under the fiducial cosmology.

Parameter estimation
--------------------

.. figure:: figures/fig06_fisher_ellipses.png
   :width: 90%
   :alt: Fisher ellipses

   **fig06 — Fisher ellipses.** Fisher-matrix uncertainty ellipses for
   representative events.

.. figure:: figures/fig07_corner_plot.png
   :width: 90%
   :alt: Corner plot

   **fig07 — Corner plot.** Parameter covariances for a representative event.

.. figure:: figures/fig12_uncertainty_violins.png
   :width: 90%
   :alt: Uncertainty violins

   **fig12 — Uncertainty violins.** Distributions of per-parameter Cramér–Rao
   uncertainties across the catalogue.

.. figure:: figures/fig14_crb_coverage.png
   :width: 90%
   :alt: CRB coverage

   **fig14 — CRB coverage.** Cramér–Rao bound coverage across the detected events.

Instrument & signals
-------------------

.. figure:: figures/fig10_lisa_psd.png
   :width: 90%
   :alt: LISA PSD

   **fig10 — LISA PSD.** The LISA noise power spectral density / sensitivity curve
   used in the analysis.

.. figure:: figures/fig13_characteristic_strain.png
   :width: 90%
   :alt: Characteristic strain

   **fig13 — Characteristic strain.** Characteristic strain of representative
   signals against the LISA sensitivity.

Campaign summary
---------------

.. figure:: figures/fig15_campaign_dashboard.png
   :width: 90%
   :alt: Campaign dashboard

   **fig15 — Campaign dashboard.** One-page summary dashboard of the full
   production campaign.
