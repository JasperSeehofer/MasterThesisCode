"""D2-01 provenance fix: per-cell preregistration stamping.

``venue_transfer.py`` runs three disjoint cell families under three
disjoint preregistrations (venue-transfer §5, mechanism-isolation, and the
2-D dose scan). Before this fix ``PREREG_PATH`` (the venue-transfer
document) was unconditionally stamped into every result JSON's
``"preregistration"`` field, mis-attributing the 20 mechanism-study result
JSONs (commission finding D2-01). This test locks the per-family mapping
implemented by :func:`preregistration_path_for_cell`.
"""

from darksiren_emri.validation.venue_transfer import (
    CELL_SPECS,
    MECH_CELL_SPECS,
    MECH_PREREG_PATH,
    PREREG_PATH,
    SCAN_CELL_SPECS,
    SCAN_PREREG_PATH,
    preregistration_path_for_cell,
)


def test_venue_transfer_cells_stamp_the_venue_transfer_prereg() -> None:
    """T0/Ta/Tb/Tc (and any unknown cell id) keep the original default path."""
    for name in CELL_SPECS:
        assert preregistration_path_for_cell(name) == PREREG_PATH
    # Unknown cell ids (e.g. "custom") fall back to the same default.
    assert preregistration_path_for_cell("custom") == PREREG_PATH


def test_mechanism_isolation_cells_stamp_the_mechanism_prereg() -> None:
    """MN0/MEH/MEI/MN0X stamp the mechanism-isolation preregistration."""
    for name in MECH_CELL_SPECS:
        assert preregistration_path_for_cell(name) == MECH_PREREG_PATH
    assert (
        MECH_PREREG_PATH
        == "results/mechanism_study_20260813/PREREGISTRATION_MECHANISM_ISOLATION.md"
    )


def test_scan_cells_stamp_the_2d_dose_scan_prereg() -> None:
    """The 16 S{h}{i} scan cells stamp the 2-D dose-scan preregistration."""
    assert len(SCAN_CELL_SPECS) == 16
    for name in SCAN_CELL_SPECS:
        assert preregistration_path_for_cell(name) == SCAN_PREREG_PATH
    assert SCAN_PREREG_PATH == "results/mechanism_study_20260813/PREREGISTRATION_2D_DOSE_SCAN.md"


def test_cell_families_are_disjoint_so_the_mapping_is_unambiguous() -> None:
    """No cell id belongs to more than one registry (else the lookup order matters)."""
    assert set(CELL_SPECS) & set(MECH_CELL_SPECS) == set()
    assert set(CELL_SPECS) & set(SCAN_CELL_SPECS) == set()
    assert set(MECH_CELL_SPECS) & set(SCAN_CELL_SPECS) == set()
