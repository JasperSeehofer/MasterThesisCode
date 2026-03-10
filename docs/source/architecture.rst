Architecture
============

The codebase has two distinct pipelines: the **EMRI Simulation Pipeline** and the
**Bayesian Inference Pipeline**.

EMRI Simulation Pipeline
------------------------

``main.py:data_simulation()`` drives a loop over ``simulation_steps``:

1. ``Model1CrossCheck`` (cosmological model) samples EMRI events from a distribution.
2. ``GalaxyCatalogueHandler`` resolves each event to a host galaxy from the GLADE catalog.
3. ``ParameterSpace.randomize_parameters()`` + ``set_host_galaxy_parameters()`` set up the
   14-parameter EMRI.
4. ``ParameterEstimation.compute_signal_to_noise_ratio()`` computes SNR using a LISA waveform.
5. If SNR ≥ threshold: ``compute_Cramer_Rao_bounds()`` computes the Fisher matrix and saves
   to CSV.

Bayesian Inference Pipeline
----------------------------

``main.py:evaluate()`` → ``BayesianStatistics.evaluate()``:

* Loads saved Cramér-Rao bounds from CSV.
* Uses ``BayesianInference`` (in ``bayesian_inference/bayesian_inference.py``) to compute the
  posterior over H₀.
* ``GalaxyCatalog`` models the galaxy distribution and mass distribution using
  normal/truncnorm distributions.

Key Data Flow
-------------

.. code-block:: text

    ParameterSpace (14 params)
        │
        ▼
    WaveformGenerator (few / ResponseWrapper)
        │
        ▼
    ParameterEstimation
        ├── compute_signal_to_noise_ratio()   →  SNR
        └── compute_Cramer_Rao_bounds()        →  Fisher matrix  →  CSV
                                                                       │
                                                      ┌────────────────┘
                                                      ▼
                                              Detection (from CSV)
                                                      │
                                                      ▼
                                              BayesianInference
                                                      │
                                                      ▼
                                              posterior p(H₀ | {dᵢ})

Module Responsibilities
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Responsibility
   * - ``parameter_estimation/parameter_estimation.py``
     - Waveform generation via ``few``, Fisher matrix (5-point stencil), SNR and
       Cramér-Rao bounds. The ``scalar_product_of_functions`` inner product is the
       computational bottleneck (PSD loop).
   * - ``LISA_configuration.py``
     - LISA antenna patterns (F+, F×), PSD, SSB↔detector frame transformations.
   * - ``datamodels/parameter_space.py``
     - 14-parameter EMRI space with randomization and bounds.
   * - ``bayesian_inference/bayesian_inference.py``
     - ``BayesianInference`` class: likelihood, posterior, detection probability,
       and ``dist_array`` helper.
   * - ``datamodels/galaxy.py``
     - ``Galaxy`` and ``GalaxyCatalog`` dataclasses; comoving volume, redshift/mass
       distributions.
   * - ``datamodels/emri_detection.py``
     - ``EMRIDetection`` dataclass constructed from a host galaxy.
   * - ``datamodels/detection.py``
     - ``Detection`` dataclass parsed from Cramér-Rao CSV output.
   * - ``cosmological_model.py``
     - ``Model1CrossCheck`` wraps the EMRI event rate model; ``BayesianStatistics``
       orchestrates the H₀ evaluation.
   * - ``galaxy_catalogue/handler.py``
     - Interfaces with the GLADE galaxy catalog (BallTree-based lookups).
   * - ``physical_relations.py``
     - Canonical distance functions: ``dist()``, ``dist_vectorized()``,
       ``hubble_function()``, ``redshifted_mass()``.
   * - ``constants.py``
     - All physical constants and simulation configuration.
       Key: ``H=0.73``, ``SNR_THRESHOLD=20``.

GPU / CPU Portability
---------------------

All computation functions resolve the array module via the ``_get_xp`` pattern, so
the same code runs on both NumPy (CPU) and CuPy (GPU) without branching.
See :doc:`quickstart` for how to install the GPU extras.
