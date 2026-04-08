Known Limitations & Scientific References
==========================================

This page documents model assumptions, known limitations, verified components,
and scientific references for the EMRI dark siren H₀ inference pipeline.

For the H₀ posterior bias investigation timeline, see
`docs/H0_BIAS_RESOLUTION.md <../../H0_BIAS_RESOLUTION.md>`_.


Model Assumptions
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Assumption
     - Where used
     - Notes
   * - Flat ΛCDM (:math:`w_0=-1`, :math:`w_a=0`)
     - All distance integrals
     - ``physical_relations.py``; wCDM infrastructure exists but is untested
   * - Gaussian measurement noise on :math:`d_L`
     - GW likelihood
     - Width hardcoded to 10% of :math:`d_L` in Pipeline A (see Limitation 5); Pipeline B uses Fisher covariance
   * - SNR threshold as detection proxy
     - ``parameter_estimation.py``, ``bayesian_inference.py``
     - Threshold = 20; detailed waveform-parameter dependence not captured
   * - Uniform prior on :math:`H_0`
     - ``bayesian_inference.py``
     - No prior declared; implicitly flat over :math:`[H_\mathrm{min}, H_\mathrm{max}]`
   * - Synthetic galaxy catalog
     - Pipeline A
     - Galaxies drawn uniformly in log-mass and from comoving volume; GLADE used in Pipeline B
   * - LISA mission duration 5 years
     - Waveform generation
     - Feeds directly into TDI response and sky-averaged sensitivity


Known Limitations
-----------------

Items are ordered by severity. Each references the specific source location and carries a
status tag: **bug** (incorrect formula or logic), **design choice** (deliberate simplification),
or **pending fix** (acknowledged issue not yet addressed).


Limitation 1 — Comoving volume formula ``[FIXED]``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**File:** ``master_thesis_code/datamodels/galaxy.py``, ``master_thesis_code/physical_relations.py``

The function computes the comoving volume *element* :math:`dV_c/dz`, not total volume :math:`V_c`.
The exponent 2 and :math:`4\pi` prefactor were correct for the element, but the formula was
missing the :math:`1/E(z)` factor. Fix applied: :math:`cv\_grid = 4\pi \cdot (c/H_0)^3 \cdot I(z)^2 / E(z)`
(Hogg 1999, Eq. 27). All methods renamed from ``comoving_volume`` to
``comoving_volume_element`` for clarity. Standalone ``comoving_volume_element()`` in
``physical_relations.py`` verified against astropy to 0.07% accuracy. Regression test added.


Limitation 2 — Fisher matrix derivative accuracy ``[FIXED]``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**File:** ``master_thesis_code/parameter_estimation/parameter_estimation.py``

The Fisher matrix previously used an :math:`O(\varepsilon)` forward difference via
``finite_difference_derivative()``. This was replaced with the :math:`O(\varepsilon^4)`
five-point stencil (``five_point_stencil_derivative()``) as the default in GPD Phase 10.
Controlled by the ``use_five_point_stencil`` constructor parameter (default ``True``).
The forward difference remains available as a fallback but is no longer used in production.

References: Vallisneri (2008), arXiv:gr-qc/0703086; Cutler & Flanagan (1994), PRD 49, 2658.


Limitation 3 — Galactic confusion noise ``[FIXED]``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**File:** ``master_thesis_code/LISA_configuration.py``

Galactic confusion noise is now included in the LISA PSD via ``_confusion_noise()`` in
``LisaTdiConfiguration``, implementing Babak et al. (2023) arXiv:2303.15929 Eq. (17) with
observation-time-dependent knee frequency. Controlled by ``include_confusion_noise``
parameter (default ``True``). The constants from ``constants.py:77–83`` are now used.


Limitation 4 — wCDM parameters silently ignored ``[bug · MEDIUM]``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**File:** ``master_thesis_code/physical_relations.py:72``

``dist()`` accepts ``w_0`` and ``w_a`` but passes them only to ``lambda_cdm_analytic_distance()``,
which ignores them — the hypergeometric formula is exact only for flat ΛCDM (:math:`w_0=-1`,
:math:`w_a=0`). No warning is raised. Any caller supplying non-fiducial dark-energy parameters
silently receives ΛCDM distances. The correct general formula would require numerical
integration via ``hubble_function()``, which already implements the full CPL parameterisation.

Reference: Hogg (1999), arXiv:astro-ph/9905116, Eq. (14–16).


Limitation 5 — GW likelihood distance uncertainty hardcoded at 10% ``[design choice · MEDIUM]``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**File:** ``master_thesis_code/bayesian_inference/bayesian_inference.py``

**Applies to Pipeline A only.** Pipeline B (``BayesianStatistics`` in
``bayesian_statistics.py``) uses the full Fisher-matrix covariance.

The Pipeline A GW likelihood uses :math:`\sigma_{d_L} = 0.1\,d_L` for every event (via
``FRACTIONAL_LUMINOSITY_ERROR``). The simulation already computes the actual per-source
Cramér–Rao bound on :math:`d_L` (stored as ``delta_luminosity_distance_delta_luminosity_distance``
in the CSV output). Using a source-by-source uncertainty from the Fisher matrix would make
nearby, well-localised events contribute more sharply to the :math:`H_0` posterior, as they
physically should.


Limitation 6 — Two Bayesian pipelines with inconsistent formulations ``[design choice · IMPORTANT]``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Files:** ``master_thesis_code/bayesian_inference/bayesian_inference.py`` (Pipeline A);
``master_thesis_code/bayesian_inference/bayesian_statistics.py`` (Pipeline B)

Pipeline A (synthetic catalog) marginalises over a continuous redshift grid with a scalar
Gaussian likelihood on :math:`d_L` and a simplified selection correction.
Pipeline B (GLADE catalog) constructs a full multivariate Gaussian likelihood over
:math:`(\varphi, \theta, d_L/d_L^\mathrm{pred})` using the actual Fisher-matrix covariance and a
simulation-based detection-probability estimate.
The two formulations are not mathematically equivalent and would yield different posteriors
on identical data. **Pipeline B is the science-grade implementation; Pipeline A is a
development-only cross-check.**


Limitation 7 — Outdated fiducial cosmological parameters ``[design choice · LOW]``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**File:** ``master_thesis_code/constants.py:29–30``

.. list-table::
   :header-rows: 1

   * - Parameter
     - Code
     - Planck 2018
   * - :math:`\Omega_m`
     - 0.25
     - 0.3153 ± 0.0073
   * - :math:`\Omega_\Lambda`
     - 0.75
     - 0.6847 ± 0.0073
   * - :math:`h` (simulation)
     - 0.73
     - 0.6736 ± 0.0054

The WMAP-era values used here differ from the current Planck 2018 best fit by ~2σ in
:math:`\Omega_m` and ~1σ in :math:`h`. For a simulation intended to represent realistic LISA science
the fiducial point should be updated.

Reference: Planck Collaboration (2018), arXiv:1807.06209, Table 2.


Limitation 8 — Galaxy redshift uncertainty has non-standard scaling ``[design choice · LOW]``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**File:** ``master_thesis_code/datamodels/galaxy.py:64``

.. code-block:: python

   redshift_uncertainty = min(0.013 * (1 + redshift) ** 3, 0.015)

The :math:`(1+z)^3` scaling grows rapidly and hits the cap of 0.015 at :math:`z \approx 0.14`, so all
galaxies above that redshift are assigned identical uncertainty. Standard photometric
redshift errors scale as :math:`\sigma_z \approx 0.05(1+z)`; spectroscopic errors as
:math:`\sigma_z \approx 0.001(1+z)`. No reference for the cubic form is provided.


What Is Mathematically Correct
------------------------------

The following components have been verified against their cited references:

- **Luminosity distance hypergeometric integral** (``physical_relations.py``): the form
  :math:`\,{}_2F_1(1/3,1/2;4/3;-\Omega_m(1+z)^3/\Omega_\Lambda)` is the correct analytic
  solution for flat ΛCDM. ✓
- **LISA instrumental PSD (A/E channels)** (``LISA_configuration.py``): matches Babak et al.
  (2023), arXiv:2303.15929, Eqs. (8)–(11), excluding galactic confusion noise. ✓
- **Noise-weighted inner product** (``parameter_estimation.py``): the factor-of-4 prefactor
  and one-sided PSD convention are correct; FFT normalisation is handled correctly via
  ``trapz`` over the frequency axis. ✓
- **Five-point stencil formula** (``five_point_stencil_derivative`` in ``parameter_estimation.py``):
  the coefficients :math:`(-1, 8, -8, 1)/12\varepsilon` are the correct :math:`O(\varepsilon^4)`
  centred finite difference. Now used by default in the Fisher matrix computation. ✓
- **Bayesian selection-effects correction** (Pipeline A): the ratio
  numerator/denominator where the denominator integrates
  :math:`p_\mathrm{det}(z,H_0)\,p(z|\mathrm{cat})` correctly implements the Loredo–Mandel
  selection-bias correction for the marginalised likelihood. ✓
- **Redshifted mass conversion**: :math:`M_z = M(1+z)` and its inverse are correctly implemented
  in ``physical_relations.py``. ✓


Bibliography
------------

.. [Hogg1999] Hogg, D. W. (1999). *Distance measures in cosmology*. arXiv:astro-ph/9905116.

.. [Babak2023] Babak, S. et al. (2023). *LISA sensitivity and SNR calculations*. arXiv:2303.15929.

.. [CF1994] Cutler, C. & Flanagan, É. E. (1994). Gravitational waves from merging compact binaries:
   How accurately can one extract the binary's parameters from the inspiral waveform?
   *Phys. Rev. D* **49**, 2658.

.. [Vallisneri2008] Vallisneri, M. (2008). Use and abuse of the Fisher information matrix in the assessment
   of gravitational-wave parameter-estimation prospects. *Phys. Rev. D* **77**, 042001.
   arXiv:gr-qc/0703086.

.. [Chen2018] Chen, H.-Y., Fishbach, M. & Holz, D. E. (2018). A two percent Hubble constant measurement
   from standard sirens within five years. *Nature* **562**, 545–547. arXiv:1709.08079.

.. [Gray2020] Gray, R. et al. (2020). Cosmological inference using gravitational wave standard sirens:
   A mock data challenge. *Phys. Rev. D* **101**, 122001. arXiv:1908.06050.

.. [Planck2018] Planck Collaboration (2020). Planck 2018 results VI: Cosmological parameters.
   *Astron. Astrophys.* **641**, A6. arXiv:1807.06209.
