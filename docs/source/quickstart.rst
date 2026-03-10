Quickstart
==========

Prerequisites
-------------

Install `uv <https://docs.astral.sh/uv/>`_ for dependency management::

    curl -LsSf https://astral.sh/uv/install.sh | sh

System packages required before ``uv sync``:

* **GSL** (GNU Scientific Library) — required by ``fastemriwaveforms`` at build time

  * Arch/Manjaro: ``sudo pacman -S gsl``
  * Ubuntu/Debian: ``sudo apt install libgsl-dev``

* **CUDA 12 toolkit** — required on the GPU cluster only

Installation
------------

Dev machine (CPU only)::

    uv sync --extra cpu --extra dev

GPU cluster (CUDA 12)::

    uv sync --extra gpu

Usage
-----

EMRI simulation (generates SNR + Cramér-Rao bounds)::

    uv run python -m master_thesis_code <working_dir> --simulation_steps N [--simulation_index I] [--log_level DEBUG]

Bayesian inference (evaluate Hubble constant posterior)::

    uv run python -m master_thesis_code <working_dir> --evaluate [--h_value 0.73]

SNR analysis only::

    uv run python -m master_thesis_code <working_dir> --snr_analysis

Running Tests
-------------

::

    # Dev machine (CPU only) — default
    uv run pytest -m "not gpu"

    # Cluster (GPU available) — runs everything
    uv run pytest

    # Fast subset only
    uv run pytest -m "not gpu and not slow"

Building the Documentation
--------------------------

::

    uv run make -C docs html

Then open ``docs/build/html/index.html`` in a browser.

Development Workflow
--------------------

Linting and formatting::

    uv run ruff check --fix master_thesis_code/   # lint and auto-fix
    uv run ruff format master_thesis_code/        # format
    uv run mypy master_thesis_code/               # type check

Pre-commit hooks run ruff and mypy automatically on every ``git commit``.
To run all hooks manually::

    uv run pre-commit run --all-files
