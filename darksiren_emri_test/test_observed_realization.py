"""Tests for the seeded observed-catalogue realization (campaign #53).

[PHYSICS] Realistic host-observation model, RATIFIED 2026-07-29
(docs/derivations/realistic_host_observation_model.md). The gates pinned here:

- [RATIFY-R6] / P5: ``sigma_scale = 0`` writes a BYTE-IDENTICAL copy of the
  parent catalogue (sha256 equality) — the limiting case tying campaign #53
  to the validated campaign-#51 baseline.
- [RATIFY-R2] counted-once: the realized scatter's per-flag standard
  deviation equals the STORED ``z_error`` column (and ``M_error/M`` in log
  space), to Monte-Carlo tolerance — sigma_realized == sigma_kernel by
  construction, not by a re-derived component split.
- §6.1 reproducibility: the realization is a pure function of
  (parent bytes, seed, sigma_scale).
- §2.4: the z >= 1e-5 floor is a CLIP (point mass), not a redraw.
- §9 guards 4 and 5: missing sidecar => legacy/unscattered; refusal to
  overwrite a realization produced from a different parent.
- Loader default path (no override) is unchanged.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from darksiren_emri.constants import GALAXY_CATALOG_REDSHIFT_LOWER_LIMIT
from darksiren_emri.galaxy_catalogue import handler as handler_module
from darksiren_emri.galaxy_catalogue.handler import (
    CatalogueColumns,
    GalaxyCatalogueHandler,
    _empiric_stellar_mass_to_BH_mass_relation,
    _reduced_catalog_column_names,
)
from darksiren_emri.galaxy_catalogue.observed_realization import (
    _sha256_of_file,
    load_realization_sidecar,
    observed_catalogue_filename,
    realize_observed_catalogue,
    sidecar_path_for,
)

_COLUMNS = _reduced_catalog_column_names()
_Z = CatalogueColumns.REDSHIFT.name
_Z_ERR = CatalogueColumns.REDSHIFT_MEASUREMENT_ERROR.name
_MSTAR = CatalogueColumns.STELLAR_MASS.name
_MSTAR_ERR = CatalogueColumns.STELLAR_MASS_ABSOULTE_ERROR.name
_FLAG = CatalogueColumns.REDSHIFT_FLAG.name


def _write_catalogue(
    path: Path,
    z: list[float],
    z_error: list[float],
    flag: list[int],
    mstar: list[float] | None = None,
    mstar_error: list[float] | None = None,
) -> None:
    """Write a headerless reduced-catalogue CSV in the production schema."""
    n = len(z)
    frame = pd.DataFrame(
        {
            CatalogueColumns.RIGHT_ASCENSION.name: np.linspace(10.0, 350.0, n),
            CatalogueColumns.DECLINATION.name: np.linspace(-40.0, 40.0, n),
            CatalogueColumns.APPARENT_B_MAG.name: np.full(n, 15.5),
            _Z: z,
            _Z_ERR: z_error,
            _MSTAR: mstar if mstar is not None else np.full(n, 1.0),
            _MSTAR_ERR: mstar_error if mstar_error is not None else np.full(n, 0.1),
            _FLAG: flag,
        }
    )[_COLUMNS]
    frame.to_csv(path, header=False, index=False)


def _synthetic_parent(tmp_path: Path, n_per_flag: int = 4000) -> Path:
    """A large two-flag synthetic catalogue for the width-identity gate."""
    rng = np.random.default_rng(20260729)
    z_photo = rng.uniform(0.20, 0.60, n_per_flag)
    z_spec = rng.uniform(0.20, 0.60, n_per_flag)
    # Photometric rows: wide sigma (~0.036 at the venue). Spectroscopic:
    # narrow (~0.0024, incl. the folded PV term). Deliberately z-shifted away
    # from the 1e-5 floor so no clipping contaminates the width check.
    err_photo = np.full(n_per_flag, 0.036)
    err_spec = np.full(n_per_flag, 0.0024)
    path = tmp_path / "parent.csv"
    _write_catalogue(
        path,
        z=list(z_photo) + list(z_spec),
        z_error=list(err_photo) + list(err_spec),
        flag=[1] * n_per_flag + [3] * n_per_flag,
        mstar=list(rng.uniform(0.5, 5.0, 2 * n_per_flag)),
        mstar_error=list(rng.uniform(0.05, 0.5, 2 * n_per_flag)),
    )
    return path


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, names=_COLUMNS)


# --------------------------------------------------------------------------
# [RATIFY-R6] / P5: sigma -> 0 byte-identity regression gate
# --------------------------------------------------------------------------


def test_sigma_scale_zero_is_byte_identical_copy(tmp_path: Path) -> None:
    """P5 hard gate: sigma_scale = 0 => sha256(child) == sha256(parent)."""
    parent = tmp_path / "parent.csv"
    # Values chosen to break under any float round-trip (17-digit reprs,
    # trailing zeros, exponent form).
    parent.write_text(
        "10.0,20.0,15.5,0.0500000000000000003,0.036,1.00000000000000009,1.0e-01,1\n"
        "30.0,-10.0,14.0,0.030,0.0024,2.500,2.5e-01,3\n"
    )
    child = tmp_path / observed_catalogue_filename(7)
    sidecar = realize_observed_catalogue(
        str(parent), str(child), realization_seed=7, sigma_scale=0.0
    )

    assert child.read_bytes() == parent.read_bytes()
    assert _sha256_of_file(str(child)) == _sha256_of_file(str(parent))
    assert sidecar["parent_csv_sha256"] == sidecar["observed_csv_sha256"]
    assert sidecar["sigma_scale"] == 0.0
    assert sidecar["n_rows"] == 2


def test_sigma_scale_zero_sidecar_marks_catalogue_unscattered(tmp_path: Path) -> None:
    """sigma_scale = 0 realizations stay BASELINE-legal (one-directional guard)."""
    parent = tmp_path / "parent.csv"
    _write_catalogue(parent, z=[0.05, 0.30], z_error=[0.036, 0.0024], flag=[1, 3])
    child = tmp_path / observed_catalogue_filename(1)
    realize_observed_catalogue(str(parent), str(child), realization_seed=1, sigma_scale=0.0)

    sidecar = load_realization_sidecar(str(child))
    assert sidecar is not None
    assert sidecar["sigma_scale"] == 0.0


# --------------------------------------------------------------------------
# [RATIFY-R2] counted-once: realized width == stored width, per flag
# --------------------------------------------------------------------------


def test_realized_z_scatter_matches_stored_z_error_per_flag(tmp_path: Path) -> None:
    """The realized scatter's per-flag std equals the STORED z_error column.

    [RATIFY-R2]: one total Gaussian drawn from the same column every
    inference kernel consumes. Monte-Carlo tolerance for n = 4000 per flag is
    ~1/sqrt(2n) ~ 1.1% on the std; 5% is a comfortable envelope.
    """
    parent = _synthetic_parent(tmp_path)
    child = tmp_path / observed_catalogue_filename(11)
    realize_observed_catalogue(str(parent), str(child), realization_seed=11, sigma_scale=1.0)

    truth = _read(parent)
    observed = _read(child)
    assert (observed[_Z_ERR] == truth[_Z_ERR]).all(), "stored widths must be untouched"
    assert (observed[_FLAG] == truth[_FLAG]).all()

    residual = (observed[_Z] - truth[_Z]) / truth[_Z_ERR]
    for flag_value in (1, 3):
        mask = truth[_FLAG] == flag_value
        assert float(residual[mask].std()) == pytest.approx(1.0, rel=0.05)
        assert float(residual[mask].mean()) == pytest.approx(0.0, abs=0.05)


def test_realized_mass_scatter_is_lognormal_with_stored_width(tmp_path: Path) -> None:
    """ln M_BH,obs - ln M_BH = N(0, M_error/M): the §2.3 mass counterpart.

    The realization writes the STELLAR mass column, so the check maps both the
    true and observed stellar masses through the SAME load-time Reines &
    Volonteri relation the handler applies and compares in ln M_BH.
    """
    parent = _synthetic_parent(tmp_path)
    child = tmp_path / observed_catalogue_filename(12)
    realize_observed_catalogue(str(parent), str(child), realization_seed=12, sigma_scale=1.0)

    truth = _read(parent)
    observed = _read(child)
    # [RATIFY-R2] The stellar-ERROR column is REWRITTEN (it was not, until the
    # 2026-07-29 width fix): the BH-mass width law is not scale-free in M*, so
    # leaving the error column alone made the width the inference recomputes
    # differ from the width the scatter was drawn with (measured pull 0.929,
    # per-row drift up to +-18%). The realization re-solves the propagated term
    # so the two coincide.
    assert not (observed[_MSTAR_ERR] == truth[_MSTAR_ERR]).all(), (
        "stellar-error column must be re-solved, not copied"
    )

    m_true, m_err_true = _empiric_stellar_mass_to_BH_mass_relation(
        truth[_MSTAR].to_numpy(), truth[_MSTAR_ERR].to_numpy()
    )
    m_obs, m_err_obs = _empiric_stellar_mass_to_BH_mass_relation(
        observed[_MSTAR].to_numpy(), observed[_MSTAR_ERR].to_numpy()
    )
    sigma_ln = np.asarray(m_err_true) / np.asarray(m_true)
    # THE gate: the width the inference recomputes from the observed row equals
    # the width the displacement was drawn with, per row (up to the documented
    # width-floor rows where no solution exists and the loaded width is wider).
    sigma_ln_obs = np.asarray(m_err_obs) / np.asarray(m_obs)
    solved = sigma_ln_obs <= sigma_ln * (1.0 + 1e-9)
    np.testing.assert_allclose(sigma_ln_obs[solved], sigma_ln[solved], rtol=1e-12)

    residual = (np.log(np.asarray(m_obs)) - np.log(np.asarray(m_true))) / sigma_ln_obs

    assert float(np.std(residual)) == pytest.approx(1.0, rel=0.05)
    assert float(np.mean(residual)) == pytest.approx(0.0, abs=0.05)
    # Lognormal-ness: the ln-space residual is symmetric (|skew| small) while
    # the LINEAR mass ratio is right-skewed by construction.
    assert abs(float(pd.Series(residual).skew())) < 0.15
    linear_ratio = np.asarray(m_obs) / np.asarray(m_true)
    assert float(pd.Series(linear_ratio).skew()) > 0.1
    assert (linear_ratio > 0).all()


def test_sidecar_width_check_statistics_are_recorded_per_flag(tmp_path: Path) -> None:
    """§6.1: the sidecar records the realized-vs-stored width check per flag."""
    parent = _synthetic_parent(tmp_path)
    child = tmp_path / observed_catalogue_filename(13)
    sidecar = realize_observed_catalogue(
        str(parent), str(child), realization_seed=13, sigma_scale=1.0
    )
    width_check = sidecar["width_check"]
    assert isinstance(width_check, dict)
    per_flag = width_check["z_per_flag"]
    assert isinstance(per_flag, dict)
    assert set(per_flag) == {"1", "3"}
    for stats in per_flag.values():
        assert stats["normalized_residual_std"] == pytest.approx(1.0, rel=0.05)
        assert stats["expected_std"] == 1.0
    assert width_check["mass"]["normalized_residual_std"] == pytest.approx(1.0, rel=0.05)


def test_sigma_scale_scales_the_realized_width(tmp_path: Path) -> None:
    """sigma_scale multiplies the realized width (and only that)."""
    parent = _synthetic_parent(tmp_path)
    child = tmp_path / observed_catalogue_filename(14)
    realize_observed_catalogue(str(parent), str(child), realization_seed=14, sigma_scale=0.5)
    truth = _read(parent)
    observed = _read(child)
    residual = (observed[_Z] - truth[_Z]) / truth[_Z_ERR]
    assert float(residual.std()) == pytest.approx(0.5, rel=0.05)


# --------------------------------------------------------------------------
# §6.1 seeded reproducibility
# --------------------------------------------------------------------------


def test_same_seed_reproduces_identical_file_and_hash(tmp_path: Path) -> None:
    parent = tmp_path / "parent.csv"
    _write_catalogue(
        parent,
        z=[0.05, 0.30, 0.80],
        z_error=[0.036, 0.0024, 0.05],
        flag=[1, 3, 1],
    )
    first = tmp_path / "a" / observed_catalogue_filename(99)
    second = tmp_path / "b" / observed_catalogue_filename(99)
    first.parent.mkdir()
    second.parent.mkdir()
    sc_a = realize_observed_catalogue(str(parent), str(first), realization_seed=99)
    sc_b = realize_observed_catalogue(str(parent), str(second), realization_seed=99)

    assert first.read_bytes() == second.read_bytes()
    assert sc_a["observed_csv_sha256"] == sc_b["observed_csv_sha256"]


def test_different_seed_gives_a_different_realization(tmp_path: Path) -> None:
    parent = tmp_path / "parent.csv"
    _write_catalogue(
        parent,
        z=[0.05, 0.30, 0.80],
        z_error=[0.036, 0.0024, 0.05],
        flag=[1, 3, 1],
    )
    first = tmp_path / "a" / observed_catalogue_filename(99)
    second = tmp_path / "b" / observed_catalogue_filename(100)
    first.parent.mkdir()
    second.parent.mkdir()
    sc_a = realize_observed_catalogue(str(parent), str(first), realization_seed=99)
    sc_b = realize_observed_catalogue(str(parent), str(second), realization_seed=100)

    assert first.read_bytes() != second.read_bytes()
    assert sc_a["observed_csv_sha256"] != sc_b["observed_csv_sha256"]


def test_realization_preserves_schema_and_column_order(tmp_path: Path) -> None:
    parent = tmp_path / "parent.csv"
    _write_catalogue(parent, z=[0.05, 0.30], z_error=[0.036, 0.0024], flag=[1, 3])
    child = tmp_path / observed_catalogue_filename(5)
    realize_observed_catalogue(str(parent), str(child), realization_seed=5)

    lines = child.read_text().strip().splitlines()
    assert all(len(line.split(",")) == len(_COLUMNS) for line in lines)
    assert all(line.split(",")[-1] in {"1", "3"} for line in lines)
    truth = _read(parent)
    observed = _read(child)
    # Sky / photometry columns round-trip unchanged.
    for column in (
        CatalogueColumns.RIGHT_ASCENSION.name,
        CatalogueColumns.DECLINATION.name,
        CatalogueColumns.APPARENT_B_MAG.name,
    ):
        assert (observed[column] == truth[column]).all()


# --------------------------------------------------------------------------
# §2.4 z-floor clip behaviour
# --------------------------------------------------------------------------


def test_low_z_rows_are_clipped_at_the_floor_not_redrawn(tmp_path: Path) -> None:
    """§2.4: z_obs is CLIPPED at 1e-5 (point mass), never redrawn.

    A z = 0.001, sigma = 0.05 population puts ~49% of draws below zero; the
    clipped rows must sit EXACTLY at the floor (a redraw would scatter them),
    and the count must be logged in the sidecar.
    """
    n = 3000
    parent = tmp_path / "parent.csv"
    _write_catalogue(parent, z=[0.001] * n, z_error=[0.05] * n, flag=[1] * n)
    child = tmp_path / observed_catalogue_filename(3)
    sidecar = realize_observed_catalogue(str(parent), str(child), realization_seed=3)

    observed = _read(child)
    assert (observed[_Z] >= GALAXY_CATALOG_REDSHIFT_LOWER_LIMIT).all()
    at_floor = np.isclose(observed[_Z], GALAXY_CATALOG_REDSHIFT_LOWER_LIMIT, rtol=0, atol=0)
    assert int(at_floor.sum()) == sidecar["n_z_floor_clipped"]
    # ~P(N(0.001, 0.05) < 1e-5) ~ 0.492 — a redraw or truncation would give 0.
    assert 0.35 * n < int(at_floor.sum()) < 0.65 * n


def test_no_clipping_when_rows_are_far_from_the_floor(tmp_path: Path) -> None:
    parent = _synthetic_parent(tmp_path, n_per_flag=500)
    child = tmp_path / observed_catalogue_filename(4)
    sidecar = realize_observed_catalogue(str(parent), str(child), realization_seed=4)
    assert sidecar["n_z_floor_clipped"] == 0


# --------------------------------------------------------------------------
# §9 guards 4 and 5 (provenance)
# --------------------------------------------------------------------------


def test_refuses_to_overwrite_realization_from_a_different_parent(tmp_path: Path) -> None:
    """§9 guard 5."""
    parent_a = tmp_path / "parent_a.csv"
    parent_b = tmp_path / "parent_b.csv"
    _write_catalogue(parent_a, z=[0.05, 0.30], z_error=[0.036, 0.0024], flag=[1, 3])
    _write_catalogue(parent_b, z=[0.06, 0.31], z_error=[0.036, 0.0024], flag=[1, 3])
    child = tmp_path / observed_catalogue_filename(8)
    realize_observed_catalogue(str(parent_a), str(child), realization_seed=8)

    with pytest.raises(ValueError, match="refusing to overwrite"):
        realize_observed_catalogue(str(parent_b), str(child), realization_seed=8)
    # Same parent is allowed (idempotent re-run).
    realize_observed_catalogue(str(parent_a), str(child), realization_seed=8)


def test_refuses_to_overwrite_a_file_without_a_sidecar(tmp_path: Path) -> None:
    parent = tmp_path / "parent.csv"
    _write_catalogue(parent, z=[0.05], z_error=[0.036], flag=[1])
    child = tmp_path / observed_catalogue_filename(9)
    child.write_text("pre-existing, unknown provenance\n")
    with pytest.raises(ValueError, match="without a sidecar"):
        realize_observed_catalogue(str(parent), str(child), realization_seed=9)


def test_missing_sidecar_reads_as_none(tmp_path: Path) -> None:
    """§9 guard 4: no sidecar => legacy/unscattered."""
    catalogue = tmp_path / "legacy.csv"
    _write_catalogue(catalogue, z=[0.05], z_error=[0.036], flag=[1])
    assert load_realization_sidecar(str(catalogue)) is None


def test_sidecar_hash_mismatch_raises(tmp_path: Path) -> None:
    parent = tmp_path / "parent.csv"
    _write_catalogue(parent, z=[0.05, 0.30], z_error=[0.036, 0.0024], flag=[1, 3])
    child = tmp_path / observed_catalogue_filename(10)
    realize_observed_catalogue(str(parent), str(child), realization_seed=10)
    child.write_text(child.read_text() + "10.0,20.0,15.5,0.5,0.01,1.0,0.1,1\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_realization_sidecar(str(child))


def test_sidecar_records_full_provenance(tmp_path: Path) -> None:
    parent = tmp_path / "parent.csv"
    _write_catalogue(parent, z=[0.05, 0.30], z_error=[0.036, 0.0024], flag=[1, 3])
    child = tmp_path / observed_catalogue_filename(21)
    realize_observed_catalogue(str(parent), str(child), realization_seed=21, sigma_scale=1.0)

    with open(sidecar_path_for(str(child))) as handle:
        sidecar = json.load(handle)
    for key in (
        "realization_seed",
        "sigma_scale",
        "parent_csv_sha256",
        "observed_csv_sha256",
        "git_commit",
        "timestamp",
        "n_rows",
        "n_z_floor_clipped",
        "n_mass_window_crossings",
        "width_check",
    ):
        assert key in sidecar, key
    assert sidecar["realization_seed"] == 21
    assert sidecar["parent_csv_sha256"] == _sha256_of_file(str(parent))


def test_negative_sigma_scale_and_self_overwrite_raise(tmp_path: Path) -> None:
    parent = tmp_path / "parent.csv"
    _write_catalogue(parent, z=[0.05], z_error=[0.036], flag=[1])
    with pytest.raises(ValueError, match="sigma_scale must be >= 0"):
        realize_observed_catalogue(
            str(parent), str(tmp_path / "out.csv"), realization_seed=1, sigma_scale=-1.0
        )
    with pytest.raises(ValueError, match="must never overwrite the TRUE catalogue"):
        realize_observed_catalogue(str(parent), str(parent), realization_seed=1)


# --------------------------------------------------------------------------
# Loader plumbing
# --------------------------------------------------------------------------


def test_loader_default_path_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No override => the baseline reduced catalogue, byte-identical behaviour."""
    baseline = tmp_path / "reduced.csv"
    _write_catalogue(baseline, z=[0.05, 0.30], z_error=[0.036, 0.0024], flag=[1, 3])
    monkeypatch.setattr(handler_module, "REDUCED_CATALOGUE_FILE_PATH", str(baseline))

    handler = object.__new__(GalaxyCatalogueHandler)  # bypass __init__ (repo idiom)
    default_frame = handler.read_reduced_galaxy_catalog()
    explicit_frame = handler.read_reduced_galaxy_catalog(path=str(baseline))
    pd.testing.assert_frame_equal(default_frame, explicit_frame)
    assert handler.scattered is False


def test_loader_reads_observed_catalogue_and_exposes_scattered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The observed override loads the realized rows and sets `scattered`."""
    baseline = tmp_path / "reduced.csv"
    _write_catalogue(baseline, z=[0.05, 0.30], z_error=[0.036, 0.0024], flag=[1, 3])
    monkeypatch.setattr(handler_module, "REDUCED_CATALOGUE_FILE_PATH", str(baseline))
    observed = tmp_path / observed_catalogue_filename(31)
    realize_observed_catalogue(str(baseline), str(observed), realization_seed=31)

    handler = object.__new__(GalaxyCatalogueHandler)
    frame = handler.read_reduced_galaxy_catalog(path=str(observed))
    assert len(frame) == 2
    # The realized z's differ from the truth (sigma_scale = 1).
    assert not np.allclose(frame[_Z].to_numpy(), _read(baseline)[_Z].to_numpy())

    sidecar = load_realization_sidecar(str(observed))
    assert sidecar is not None
    assert sidecar["sigma_scale"] == 1.0


def test_mass_window_crossings_are_counted(tmp_path: Path) -> None:
    """§5.3: rows whose pruning-window membership changes are logged."""
    # Stellar masses straddling the M_SOURCE_FRAME_MIN = 1e4 Msun BH edge.
    n = 2000
    rng = np.random.default_rng(5)
    parent = tmp_path / "parent.csv"
    _write_catalogue(
        parent,
        z=list(rng.uniform(0.2, 0.4, n)),
        z_error=[0.01] * n,
        flag=[1] * n,
        mstar=list(np.full(n, 0.02)),
        mstar_error=list(np.full(n, 0.002)),
    )
    child = tmp_path / observed_catalogue_filename(41)
    sidecar = realize_observed_catalogue(str(parent), str(child), realization_seed=41)
    assert isinstance(sidecar["n_mass_window_crossings"], int)
    assert 0 <= sidecar["n_mass_window_crossings"] <= n
    assert not math.isnan(float(sidecar["n_mass_window_crossings"]))
