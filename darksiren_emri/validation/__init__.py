"""Independent validation instruments for the dark-siren inference pipeline.

This subpackage hosts self-contained, pure numpy/scipy calibration harnesses
that deliberately do NOT import the production inference code
(``bayesian_inference``), so they can serve as independent cross-checks.
"""

from darksiren_emri.validation.pp_coverage import PPCoverageConfig, run_coverage

__all__ = ["PPCoverageConfig", "run_coverage"]
