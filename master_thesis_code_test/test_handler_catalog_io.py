"""End-to-end test of the GLADE+ reduced-catalogue writer/reader contract.

Covers the previously-untested ``parse_to_reduced_catalog`` →
``read_reduced_galaxy_catalog`` round trip (review finding TQ-01, 2026-07-04). The
8-headerless-column positional contract is load-bearing: every downstream reader
(inference, ``pixel_completeness.build_m_th_map``) depends on the column order and the
folded redshift error, and a column-order drift has bitten this project before (the
``.stale6col_mar28`` / ``.zhelio_20260702`` variants at the repo root).

The full GLADE+ parse pipeline is exercised on a tiny synthetic raw file:
flag filtering ({1, 3} survive), the peculiar-velocity error NaN→0.0015 floor + quadrature
fold into the redshift error, the trailing integer redshift flag, the column reorder, and
the PHI_S/THETA_S rename on read.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from master_thesis_code.galaxy_catalogue import handler as handler_module
from master_thesis_code.galaxy_catalogue.handler import (
    CatalogueColumns,
    GalaxyCatalogueHandler,
    InternalCatalogColumns,
    _reduced_catalog_column_names,
)

# Raw GLADE+ rows are space-separated with the physics columns at fixed 0-based
# indices (CatalogueColumns.*.value). Build 37-token rows and set only the used cells.
_N_RAW_COLUMNS = 37
_USED = {
    CatalogueColumns.RIGHT_ASCENSION.value: "ra",
    CatalogueColumns.DECLINATION.value: "dec",
    CatalogueColumns.APPARENT_B_MAG.value: "bmag",
    CatalogueColumns.REDSHIFT.value: "z",
    CatalogueColumns.REDSHIFT_PECULIAR_VELOCITY_ERROR.value: "pv_err",
    CatalogueColumns.REDSHIFT_MEASUREMENT_ERROR.value: "z_err",
    CatalogueColumns.REDSHIFT_FLAG.value: "flag",
    CatalogueColumns.STELLAR_MASS.value: "mstar",
    CatalogueColumns.STELLAR_MASS_ABSOULTE_ERROR.value: "mstar_err",
}


def _raw_row(**cells: str) -> str:
    tokens = ["0"] * _N_RAW_COLUMNS
    for col_value, key in _USED.items():
        tokens[col_value] = cells[key]
    return " ".join(tokens)


# flag 1 (photo) and 3 (spec) survive; 0 (none) and 2 (distance-only) are dropped.
# Row "spec_nan" uses "null" for the PV error → pandas NaN → 0.0015 floor.
_ROWS = [
    dict(
        ra="10.0",
        dec="20.0",
        bmag="15.0",
        z="0.05",
        pv_err="0.001",
        z_err="0.02",
        flag="1",
        mstar="1.0",
        mstar_err="0.1",
    ),  # survives (photo)
    dict(
        ra="30.0",
        dec="-10.0",
        bmag="14.0",
        z="0.03",
        pv_err="null",
        z_err="0.001",
        flag="3",
        mstar="2.0",
        mstar_err="0.2",
    ),  # survives (spec, NaN pv)
    dict(
        ra="99.0",
        dec="0.0",
        bmag="18.0",
        z="0.10",
        pv_err="0.001",
        z_err="0.05",
        flag="0",
        mstar="0.3",
        mstar_err="0.03",
    ),  # dropped (flag 0)
    dict(
        ra="98.0",
        dec="1.0",
        bmag="17.0",
        z="0.20",
        pv_err="0.001",
        z_err="0.05",
        flag="2",
        mstar="0.4",
        mstar_err="0.04",
    ),  # dropped (flag 2)
    dict(
        ra="40.0",
        dec="5.0",
        bmag="16.0",
        z="0.08",
        pv_err="0.0005",
        z_err="0.0015",
        flag="3",
        mstar="0.5",
        mstar_err="0.05",
    ),  # survives (spec)
]


@pytest.fixture
def reduced_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[GalaxyCatalogueHandler, Path]:
    """Point the module-global reduced-catalogue path at a fresh tmp file and parse."""
    raw = tmp_path / "GLADE_raw.txt"
    raw.write_text("\n".join(_raw_row(**row) for row in _ROWS) + "\n")

    reduced = tmp_path / "reduced.csv"
    monkeypatch.setattr(handler_module, "REDUCED_CATALOGUE_FILE_PATH", str(reduced))

    handler = object.__new__(GalaxyCatalogueHandler)  # bypass __init__ (repo idiom)
    handler.parse_to_reduced_catalog(galaxy_catalogue_file_path=str(raw))
    return handler, reduced


def test_only_flag_1_and_3_rows_survive(
    reduced_csv: tuple[GalaxyCatalogueHandler, Path],
) -> None:
    _handler, reduced = reduced_csv
    df = pd.read_csv(reduced, names=_reduced_catalog_column_names())
    assert len(df) == 3
    assert set(df[CatalogueColumns.REDSHIFT_FLAG.name]) == {1, 3}


def test_on_disk_column_order_and_field_count(
    reduced_csv: tuple[GalaxyCatalogueHandler, Path],
) -> None:
    _handler, reduced = reduced_csv
    lines = reduced.read_text().strip().splitlines()
    # 8 headerless comma-separated columns, flag trailing as an integer "1"/"3".
    assert all(len(line.split(",")) == 8 for line in lines)
    assert _reduced_catalog_column_names()[-1] == CatalogueColumns.REDSHIFT_FLAG.name
    assert all(line.split(",")[-1] in {"1", "3"} for line in lines)


def test_redshift_error_folds_pv_in_quadrature_with_nan_floor(
    reduced_csv: tuple[GalaxyCatalogueHandler, Path],
) -> None:
    _handler, reduced = reduced_csv
    df = pd.read_csv(reduced, names=_reduced_catalog_column_names())
    by_z = {
        round(z, 4): err
        for z, err in zip(
            df[CatalogueColumns.REDSHIFT.name],
            df[CatalogueColumns.REDSHIFT_MEASUREMENT_ERROR.name],
        )
    }
    # photo row: sqrt(0.02^2 + 0.001^2)
    assert by_z[0.05] == pytest.approx(math.sqrt(0.02**2 + 0.001**2))
    # spec-NaN row: pv NaN → 0.0015 floor, sqrt(0.001^2 + 0.0015^2)
    assert by_z[0.03] == pytest.approx(math.sqrt(0.001**2 + 0.0015**2))
    # spec row: sqrt(0.0015^2 + 0.0005^2)
    assert by_z[0.08] == pytest.approx(math.sqrt(0.0015**2 + 0.0005**2))


def test_read_renames_sky_columns_to_frame_neutral_symbols(
    reduced_csv: tuple[GalaxyCatalogueHandler, Path],
) -> None:
    handler, _reduced = reduced_csv
    df = handler.read_reduced_galaxy_catalog()
    assert InternalCatalogColumns.PHI_S in df.columns
    assert InternalCatalogColumns.THETA_S in df.columns
    assert CatalogueColumns.RIGHT_ASCENSION.name not in df.columns
    assert CatalogueColumns.DECLINATION.name not in df.columns


def test_round_trip_values_are_preserved(
    reduced_csv: tuple[GalaxyCatalogueHandler, Path],
) -> None:
    handler, _reduced = reduced_csv
    df = (
        handler.read_reduced_galaxy_catalog()
        .sort_values(CatalogueColumns.REDSHIFT.name)
        .reset_index(drop=True)
    )
    # spec row at z=0.03 keeps its RA/Dec (→ PHI_S/THETA_S), B-mag and BH-mass columns.
    spec = df[np.isclose(df[CatalogueColumns.REDSHIFT.name], 0.03)].iloc[0]
    assert spec[InternalCatalogColumns.PHI_S] == pytest.approx(30.0)
    assert spec[InternalCatalogColumns.THETA_S] == pytest.approx(-10.0)
    assert spec[CatalogueColumns.STELLAR_MASS.name] == pytest.approx(2.0)
