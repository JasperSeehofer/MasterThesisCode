"""Fast, pool-free tests for the Option-B mirror-universe generator (D-B).

These tests exercise :class:`~darksiren_emri.validation.correspondence_1d.MirrorUniverseGenerator`
in isolation from the (expensive) real ``GalaxyCatalogueHandler``/CRB-CSV
build: a tiny synthetic donor CRB CSV and a hand-built
:class:`~darksiren_emri.validation.correspondence_1d.HostPool` stand in, so
the whole suite runs in well under a second and needs no pinned production
inputs. G-0/G-1/G-2 themselves (which DO need the pinned inputs and are
expensive) are exercised by the harness's own CLI (``--stage g0|g1|g2``),
not by this test module.
"""

import dataclasses
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from darksiren_emri.emri_rate import R_eff_per_mbh
from darksiren_emri.validation import correspondence_1d as c1d

from .conftest import requires_artifact

_N_DONOR_ROWS = 12
_N_HOST_POOL = 20


def _make_donor_csv(tmp_path_factory: pytest.TempPathFactory, n_rows: int = _N_DONOR_ROWS) -> str:
    rng = np.random.default_rng(1234)
    df = pd.DataFrame(
        {
            "SNR": rng.uniform(20.0, 80.0, n_rows),
            "luminosity_distance": rng.uniform(1.0, 3.0, n_rows),
            "phiS": rng.uniform(0.0, 2 * np.pi, n_rows),
            "qS": rng.uniform(0.1, np.pi - 0.1, n_rows),
            "delta_luminosity_distance_delta_luminosity_distance": rng.uniform(0.001, 0.05, n_rows),
            "delta_phiS_delta_phiS": rng.uniform(1e-4, 1e-2, n_rows),
            "delta_qS_delta_qS": rng.uniform(1e-4, 1e-2, n_rows),
            "delta_phiS_delta_qS": np.zeros(n_rows),
            "host_galaxy_index": -1,
            "in_catalog": False,
            "_coord_frame": "ecliptic_BarycentricTrue_J2000",
            "_cov_frame": "ecliptic_BarycentricTrue_J2000",
        }
    )
    path = tmp_path_factory.mktemp("mirror_donor") / "donor_crb.csv"
    df.to_csv(path, index=False)
    return str(path)


def _make_host_pool(n_pool: int = _N_HOST_POOL) -> c1d.HostPool:
    rng = np.random.default_rng(5678)
    return c1d.HostPool(
        phiS=rng.uniform(0.0, 2 * np.pi, n_pool),
        qS=rng.uniform(0.1, np.pi - 0.1, n_pool),
        z=rng.uniform(0.01, 0.3, n_pool),
        z_error=rng.uniform(0.001, 0.04, n_pool),
        n=n_pool,
    )


@pytest.fixture
def generator(tmp_path_factory: pytest.TempPathFactory) -> c1d.MirrorUniverseGenerator:
    donor_csv = _make_donor_csv(tmp_path_factory)
    cfg = c1d.CorrespondenceConfig(n_events=5, crb_reference_csv=donor_csv)
    return c1d.MirrorUniverseGenerator(cfg)


def test_draw_realization_shape_and_columns(generator: c1d.MirrorUniverseGenerator) -> None:
    pool = _make_host_pool()
    events = generator.draw_realization(seed=1, host_pool=pool)
    assert len(events) == generator.config.n_events
    # Same column set as the donor CSV -- no columns added or dropped.
    assert set(events.columns) == set(pd.read_csv(generator.config.crb_reference_csv).columns)


def test_draw_realization_is_deterministic(generator: c1d.MirrorUniverseGenerator) -> None:
    pool = _make_host_pool()
    a = generator.draw_realization(seed=42, host_pool=pool)
    b = generator.draw_realization(seed=42, host_pool=pool)
    pd.testing.assert_frame_equal(a, b)


def test_draw_realization_seed_sensitivity(generator: c1d.MirrorUniverseGenerator) -> None:
    pool = _make_host_pool()
    a = generator.draw_realization(seed=1, host_pool=pool)
    b = generator.draw_realization(seed=2, host_pool=pool)
    assert not a["luminosity_distance"].equals(b["luminosity_distance"])


def test_host_z_convention_true_d_l_matches_host_redshift(
    generator: c1d.MirrorUniverseGenerator,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """D-B item (c)/(d): true d_L is dist(host_z, h_true); observed d_L scatters about it.

    The catalogue's stored z IS the mirror truth (D-B item d); the observed
    d_L must be centered on ``dist(host_z, H_TRUE)``, not on the donor row's
    own (discarded) luminosity_distance value.
    """
    n = 4000
    pool = _make_host_pool(n_pool=n)
    donor_csv = _make_donor_csv(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    events = gen.draw_realization(seed=7, host_pool=pool)

    host_idx = events["host_galaxy_index"].to_numpy()
    true_d_l = c1d.dist_vectorized(pool.z[host_idx], h=c1d.H_TRUE)
    residual = events["luminosity_distance"].to_numpy() - true_d_l
    # The residual must be centered near zero at this sample size (noise std
    # per-row is O(0.001-0.05) Gpc; over 4000 draws the mean residual should
    # be << the smallest per-row sigma).
    assert abs(float(np.mean(residual))) < 0.01

    # host_galaxy_index/in_catalog are stamped as the drawn host (D-B item a).
    assert (events["host_galaxy_index"].to_numpy() == host_idx).all()
    assert bool(events["in_catalog"].all())
    assert host_idx.min() >= 0
    assert host_idx.max() < pool.n


def test_sky_recentered_at_host(tmp_path_factory: pytest.TempPathFactory) -> None:
    """The observed sky position is drawn about the HOST's (phiS, qS), not the donor row's."""
    n = 2000
    pool = _make_host_pool(n_pool=n)
    donor_csv = _make_donor_csv(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    events = gen.draw_realization(seed=3, host_pool=pool)
    host_idx = events["host_galaxy_index"].to_numpy()
    d_qS = events["qS"].to_numpy() - pool.qS[host_idx]
    assert abs(float(np.mean(d_qS))) < 0.02


def test_host_draw_weights_normalized_and_positive(generator: c1d.MirrorUniverseGenerator) -> None:
    pool = _make_host_pool()
    w = c1d.MirrorUniverseGenerator._host_draw_weights(pool)
    assert w.shape == (pool.n,)
    assert np.all(w > 0.0)
    assert abs(float(w.sum()) - 1.0) < 1e-10


def test_exact_z_catalogue_floors_redshift_error(tmp_path: Path) -> None:
    from darksiren_emri.galaxy_catalogue.handler import _reduced_catalog_column_names

    cols = _reduced_catalog_column_names()
    rng = np.random.default_rng(99)
    n = 6
    row = {c: rng.uniform(0.01, 1.0, n) for c in cols}
    row["REDSHIFT_FLAG"] = np.full(n, 3)
    parent = tmp_path / "parent_catalogue.csv"
    pd.DataFrame(row)[cols].to_csv(parent, header=False, index=False)

    out = tmp_path / "exact_z.csv"
    c1d.build_exact_z_catalogue(str(out), catalogue_path=str(parent))
    written = pd.read_csv(out, names=cols)
    idx = cols.index("REDSHIFT_MEASUREMENT_ERROR")
    assert written.iloc[:, idx].to_numpy() == pytest.approx(c1d.EXACT_Z_ERROR_FLOOR)
    # Redshift itself (truth, D-B item d) is untouched.
    z_idx = cols.index("REDSHIFT")
    parent_df = pd.read_csv(parent, names=cols)
    assert written.iloc[:, z_idx].to_numpy() == pytest.approx(parent_df.iloc[:, z_idx].to_numpy())


def test_run_mirror_seed_inprocess_accepts_mass_filter_sigma_default_asymmetric() -> None:
    """[P3-WBHZERO] instrument (ii) (PREREGISTRATION_P3_WBHZERO_MEASURE_20260825.md
    §2(ii); row #198): the pass-through parameter exists with the
    production-inert default, following the same inert-plumbing pattern as
    the neighboring ``catalogue_*`` flags on this function."""
    sig = inspect.signature(c1d.run_mirror_seed_inprocess)
    assert "mass_filter_sigma" in sig.parameters
    assert sig.parameters["mass_filter_sigma"].default == "asymmetric"


# ── Fleet arm-runner CLI tests (task spec item 4: fast, no pool) ────────────


def test_arm_specs_registered_mapping() -> None:
    """The task spec's exact arm -> (sigma_z_scale, area_scale) mapping."""
    assert c1d.ARM_SPECS == {
        "b0": (1.0, 1.0),
        "bsig005": (0.05, 1.0),
        "bsig025": (0.25, 1.0),
        "eden05": (1.0, 0.5),
        "eden2": (1.0, 2.0),
        "bout": (1.0, 1.0),
        "bf1": (1.0, 1.0),
        "bsel": (1.0, 1.0),
        "bself": (1.0, 1.0),
        "bden": (1.0, 1.0),
        "b0i": (1.0, 1.0),
        "b0i2d": (1.0, 1.0),
    }


def test_arm_host_mode_and_completeness_registered_mapping() -> None:
    """AMENDMENT A-2/A-3/A-4/A-5: bout/bsel/bself/bden are the population-draw arms; bf1 the only f=1 control."""
    assert c1d.ARM_HOST_MODE == {
        "b0": "catalogue",
        "bsig005": "catalogue",
        "bsig025": "catalogue",
        "eden05": "catalogue",
        "eden2": "catalogue",
        "bout": "population",
        "bf1": "catalogue",
        "bsel": "population_selected",
        "bself": "population_selected",
        "bden": "population_selected",
        "b0i": "catalogue_selected",
        "b0i2d": "catalogue_selected_2d",
    }
    assert c1d.ARM_UNITY_COMPLETENESS == {
        "b0": False,
        "bsig005": False,
        "bsig025": False,
        "eden05": False,
        "eden2": False,
        "bout": False,
        "bf1": True,
        "bsel": False,
        "bself": False,
        "bden": False,
        "b0i": False,
        "b0i2d": False,
    }
    # Every ARM_SPECS key has an entry in both registries (no silent fallback
    # to the default for a registered arm).
    assert set(c1d.ARM_HOST_MODE) == set(c1d.ARM_SPECS)
    assert set(c1d.ARM_UNITY_COMPLETENESS) == set(c1d.ARM_SPECS)
    # AMENDMENT A-4: bself is otherwise IDENTICAL to bsel (host mode,
    # completeness override, specs) -- only ARM_SELECTION_CELL differs.
    assert c1d.ARM_HOST_MODE["bself"] == c1d.ARM_HOST_MODE["bsel"]
    assert c1d.ARM_UNITY_COMPLETENESS["bself"] == c1d.ARM_UNITY_COMPLETENESS["bsel"]
    assert c1d.ARM_SPECS["bself"] == c1d.ARM_SPECS["bsel"]
    # AMENDMENT A-5: bden is otherwise IDENTICAL to bsel (host mode,
    # completeness override, specs) -- only ARM_EVENT_MEASURE differs.
    assert c1d.ARM_HOST_MODE["bden"] == c1d.ARM_HOST_MODE["bsel"]
    assert c1d.ARM_UNITY_COMPLETENESS["bden"] == c1d.ARM_UNITY_COMPLETENESS["bsel"]
    assert c1d.ARM_SPECS["bden"] == c1d.ARM_SPECS["bsel"]
    assert c1d.ARM_SELECTION_CELL["bden"] == c1d.ARM_SELECTION_CELL["bsel"]


def test_arm_selection_cell_registered_mapping() -> None:
    """AMENDMENT A-4: every pre-A-4 arm defaults to "off"; only bself is "fused".

    This is the byte-identical-behaviour guarantee for every existing arm --
    a regression here means an existing arm's `selection_in_completion_numerator`
    silently changed.
    """
    assert c1d.ARM_SELECTION_CELL == {
        "b0": "off",
        "bsig005": "off",
        "bsig025": "off",
        "eden05": "off",
        "eden2": "off",
        "bout": "off",
        "bf1": "off",
        "bsel": "off",
        "bself": "fused",
        "bden": "off",
        # PA-2 (b0i): "fused" -- the identity test scores the production
        # runs-of-record cell, not the pre-A-4 "off" basis.
        "b0i": "fused",
        # [P3-2D] (b0i2d): same convention as b0i.
        "b0i2d": "fused",
    }
    assert set(c1d.ARM_SELECTION_CELL) == set(c1d.ARM_SPECS)
    non_bself = {
        k: v for k, v in c1d.ARM_SELECTION_CELL.items() if k not in ("bself", "b0i", "b0i2d")
    }
    assert set(non_bself.values()) == {"off"}
    assert c1d.ARM_SELECTION_CELL["bself"] == "fused"
    assert c1d.ARM_SELECTION_CELL["b0i"] == "fused"
    assert c1d.ARM_SELECTION_CELL["b0i2d"] == "fused"
    # D2 ruling (ledger row #159, 2026-08-22): the pin is "fused" for all
    # FUTURE runs-of-record; every banked pre-A-4 arm's historical "off" basis
    # lives in ARM_SELECTION_CELL above (regeneration passes it explicitly),
    # so this flip cannot silently change any banked arm's cell.
    assert c1d.PRODUCTION_FLAGS["--selection_in_completion_numerator"] == "fused"


def test_arm_event_measure_registered_mapping() -> None:
    """AMENDMENT A-5: every pre-A-5 arm defaults to "ratio"; only bden is "data".

    This is the byte-identical-behaviour guarantee for every existing arm --
    a regression here means an existing arm's `completion_event_measure`
    silently changed.
    """
    assert c1d.ARM_EVENT_MEASURE == {
        "b0": "ratio",
        "bsig005": "ratio",
        "bsig025": "ratio",
        "eden05": "ratio",
        "eden2": "ratio",
        "bout": "ratio",
        "bf1": "ratio",
        "bsel": "ratio",
        "bself": "ratio",
        "bden": "data",
        "b0i": "ratio",
        "b0i2d": "ratio",
    }
    assert set(c1d.ARM_EVENT_MEASURE) == set(c1d.ARM_SPECS)
    non_bden = {k: v for k, v in c1d.ARM_EVENT_MEASURE.items() if k != "bden"}
    assert set(non_bden.values()) == {"ratio"}
    assert c1d.ARM_EVENT_MEASURE["bden"] == "data"


def test_arm_seeds_registered_paired_discipline() -> None:
    """Paired-seed discipline (prereg §1 D-C + AMENDMENT A-2/A-3): counts + shared anchor."""
    assert len(c1d.ARM_SEEDS["b0"]) == 25
    assert len(c1d.ARM_SEEDS["bsig005"]) == 25
    assert len(c1d.ARM_SEEDS["bsig025"]) == 10
    assert len(c1d.ARM_SEEDS["eden05"]) == 10
    assert len(c1d.ARM_SEEDS["eden2"]) == 10
    assert len(c1d.ARM_SEEDS["bout"]) == 15
    assert len(c1d.ARM_SEEDS["bf1"]) == 2
    assert len(c1d.ARM_SEEDS["bsel"]) == 15
    assert len(c1d.ARM_SEEDS["bself"]) == 15
    assert len(c1d.ARM_SEEDS["bden"]) == 15
    assert len(c1d.ARM_SEEDS["b0i"]) == 25
    assert len(c1d.ARM_SEEDS["b0i2d"]) == 24  # PA-2D-1/F14 power decision
    total = sum(len(v) for v in c1d.ARM_SEEDS.values())
    # 25 + 25 + 10 + 10 + 10 + 15 + 2 + 15 + 15 + 15 = 142, the fleet task-list
    # arithmetic (80 pre-A-2 tasks + 17 AMENDMENT A-2 tasks + 15 AMENDMENT
    # A-3 tasks + 15 AMENDMENT A-4 tasks + 15 AMENDMENT A-5 tasks); PA-2's
    # b0i (25 seeds) and [P3-2D]'s b0i2d (12 seeds) are identity-test-only and
    # not part of that fleet count.
    assert total == 142 + 25 + 24  # b0i2d at 24 (PA-2D-1/F14)
    for arm, seeds in c1d.ARM_SEEDS.items():
        assert seeds[0] == 900101, arm
        assert list(seeds) == sorted(seeds), arm
    # b0/bsig005 share the FULL seed range (paired by construction).
    assert c1d.ARM_SEEDS["b0"] == c1d.ARM_SEEDS["bsig005"]
    # bsig025/eden05/eden2 share the SAME N=10 seed range (paired by
    # construction), and it is a prefix of the N=25 arms' range.
    assert c1d.ARM_SEEDS["bsig025"] == c1d.ARM_SEEDS["eden05"] == c1d.ARM_SEEDS["eden2"]
    assert c1d.ARM_SEEDS["bsig025"] == c1d.ARM_SEEDS["b0"][:10]
    # bout (N=15), bf1 (N=2), bsel (N=15) and bself (N=15) are prefixes of the
    # N=25 range too (same paired-seed discipline extended to the AMENDMENT
    # A-2/A-3/A-4 arms). bout/bsel/bself share the IDENTICAL seed range (same
    # universe construction seed at each index -- the true isolation test's
    # paired discipline against B-OUT, extended by A-4's bisection step
    # against B-SEL).
    assert c1d.ARM_SEEDS["bout"] == c1d.ARM_SEEDS["b0"][:15]
    assert c1d.ARM_SEEDS["bf1"] == c1d.ARM_SEEDS["b0"][:2]
    assert c1d.ARM_SEEDS["bsel"] == c1d.ARM_SEEDS["b0"][:15]
    assert c1d.ARM_SEEDS["bsel"] == c1d.ARM_SEEDS["bout"]
    assert c1d.ARM_SEEDS["bself"] == c1d.ARM_SEEDS["bsel"]
    assert c1d.ARM_SEEDS["bden"] == c1d.ARM_SEEDS["bsel"]


@requires_artifact(c1d.CRB_CSV_PATH)
def test_run_arm_seed_threads_selection_cell_to_evaluate_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AMENDMENT A-4 plumbing proof: ARM_SELECTION_CELL[arm] actually reaches
    the ``selection_in_completion_numerator`` kwarg :func:`run_arm_seed`
    passes to :func:`run_mirror_seed_inprocess` -- for EVERY registered arm,
    not just bself. This is the load-bearing check that bself does not
    silently fall back to "off" (which would produce a false
    CONVENTION-NOT-IT verdict) and that no pre-A-4 arm's behaviour changed.

    Stubs everything upstream of the (arm -> kwarg) wiring itself
    (``MirrorUniverseGenerator.host_pool_for_sigma_scale``/``draw_realization``,
    ``build_bsel_selection_objects``, the pinned-catalogue md5 check) so the
    test needs no real GLADE catalogue/injection pool and runs in well under
    a second; :func:`run_mirror_seed_inprocess` is stubbed at its call site
    (module-global lookup, so the monkeypatch is visible to
    :func:`run_arm_seed`) purely to CAPTURE the kwarg it was given -- the
    function's own body (verified by direct source inspection, not
    re-exercised here) forwards that same value, completely unconditionally
    and with no intervening default, straight into
    ``BayesianStatistics.evaluate(selection_in_completion_numerator=...)``.

    One dependency is NOT stubbed: ``run_arm_seed`` builds a real
    ``MirrorUniverseGenerator(cfg)``, whose ``__init__`` eagerly
    ``pd.read_csv``s the pinned production CRB CSV
    (``c1d.CRB_CSV_PATH``) -- a machine-of-record ``results/`` artifact not
    committed to VCS. Guarded with ``@requires_artifact`` (skips on CI/any
    checkout without it, runs+asserts on the data-bearing machine).
    """
    captured: dict[str, str] = {}

    def _fake_host_pool_for_sigma_scale(
        self: c1d.MirrorUniverseGenerator, work_root: Path, seed: int, sigma_z_scale: float
    ) -> tuple[c1d.HostPool, str | None, object]:
        pool = c1d.HostPool(
            phiS=np.array([0.1]),
            qS=np.array([1.0]),
            z=np.array([0.1]),
            z_error=np.array([0.01]),
            n=1,
        )
        return pool, None, object()

    def _fake_draw_realization(
        self: c1d.MirrorUniverseGenerator, seed: int, **kwargs: object
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "SNR": [30.0],
                "luminosity_distance": [1.0],
                "phiS": [0.1],
                "qS": [1.0],
                "host_galaxy_index": [-1],
                "in_catalog": [False],
            }
        )

    def _fake_run_mirror_seed_inprocess(
        work_root: Path,
        events: pd.DataFrame,
        seed: int,
        galaxy_catalog: object,
        h_values: tuple[float, ...] = c1d.H_GRID_41,
        completeness_override: bool = False,
        injection_dir: str = c1d.INJECTION_POOL_DIR,
        allow_low_pdet_coverage: bool = True,
        selection_in_completion_numerator: str = "off",
        completion_event_measure: str = "ratio",
    ) -> tuple[Path, float]:
        captured["selection_in_completion_numerator"] = selection_in_completion_numerator
        captured["completion_event_measure"] = completion_event_measure
        rows = [{"event_idx": 0, "h": h, "combined_no_bh": 1.0 + h} for h in h_values]
        work_root.mkdir(parents=True, exist_ok=True)
        diag_csv = work_root / "diag.csv"
        pd.DataFrame(rows).to_csv(diag_csv, index=False)
        return diag_csv, 0.01

    monkeypatch.setattr(
        c1d.MirrorUniverseGenerator, "host_pool_for_sigma_scale", _fake_host_pool_for_sigma_scale
    )
    monkeypatch.setattr(c1d.MirrorUniverseGenerator, "draw_realization", _fake_draw_realization)
    monkeypatch.setattr(c1d, "build_bsel_selection_objects", lambda: (None, None))
    # [P3-2D] (b0i2d): same stub convention as build_bsel_selection_objects
    # above -- this test exercises the arm -> kwarg wiring only, never the
    # real SimulationDetectionProbability/injection-pool construction.
    monkeypatch.setattr(c1d, "build_b0i_2d_selection_objects", lambda: (None, None, None))
    # PA-2 (b0i): the runtime rate-weight parity gate needs the REAL pinned
    # catalogue -- stubbed here too (same convention as build_bsel_selection_objects
    # above), since this test is exercising the arm -> kwarg wiring only.
    monkeypatch.setattr(c1d, "_verify_rate_weight_parity", lambda *args, **kwargs: 0)
    monkeypatch.setattr(c1d, "run_mirror_seed_inprocess", _fake_run_mirror_seed_inprocess)
    monkeypatch.setattr(c1d, "check_reduced_catalogue_pin", lambda: True)

    for arm, expected in c1d.ARM_SELECTION_CELL.items():
        captured.clear()
        out_dir = tmp_path / f"out_{arm}"
        record_path = c1d.run_arm_seed(tmp_path / f"work_{arm}", arm, 900101, out_dir)
        assert captured["selection_in_completion_numerator"] == expected, arm
        record = json.loads(record_path.read_text())
        assert record["selection_cell"] == expected, arm
        assert record["arm"] == arm

    # The one arm the amendment actually changes.
    assert c1d.ARM_SELECTION_CELL["bself"] == "fused"
    # Every other pre-PA-2 arm is provably unchanged (still "off"); b0i/b0i2d
    # are registered "fused" too, disclosed separately above.
    assert all(
        v == "off" for k, v in c1d.ARM_SELECTION_CELL.items() if k not in ("bself", "b0i", "b0i2d")
    )


@requires_artifact(c1d.CRB_CSV_PATH)
def test_run_arm_seed_threads_event_measure_to_evaluate_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AMENDMENT A-5 plumbing proof: ARM_EVENT_MEASURE[arm] actually reaches
    the ``completion_event_measure`` kwarg :func:`run_arm_seed` passes to
    :func:`run_mirror_seed_inprocess` -- for EVERY registered arm, not just
    bden. This is the load-bearing check that bden does not silently fall
    back to "ratio" (which would produce a false MEASURE-NOT-IT verdict) and
    that no pre-A-5 arm's behaviour changed.

    Same stub topology as
    ``test_run_arm_seed_threads_selection_cell_to_evaluate_call`` -- see that
    test's docstring for what is stubbed and why.
    """
    captured: dict[str, str] = {}

    def _fake_host_pool_for_sigma_scale(
        self: c1d.MirrorUniverseGenerator, work_root: Path, seed: int, sigma_z_scale: float
    ) -> tuple[c1d.HostPool, str | None, object]:
        pool = c1d.HostPool(
            phiS=np.array([0.1]),
            qS=np.array([1.0]),
            z=np.array([0.1]),
            z_error=np.array([0.01]),
            n=1,
        )
        return pool, None, object()

    def _fake_draw_realization(
        self: c1d.MirrorUniverseGenerator, seed: int, **kwargs: object
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "SNR": [30.0],
                "luminosity_distance": [1.0],
                "phiS": [0.1],
                "qS": [1.0],
                "host_galaxy_index": [-1],
                "in_catalog": [False],
            }
        )

    def _fake_run_mirror_seed_inprocess(
        work_root: Path,
        events: pd.DataFrame,
        seed: int,
        galaxy_catalog: object,
        h_values: tuple[float, ...] = c1d.H_GRID_41,
        completeness_override: bool = False,
        injection_dir: str = c1d.INJECTION_POOL_DIR,
        allow_low_pdet_coverage: bool = True,
        selection_in_completion_numerator: str = "off",
        completion_event_measure: str = "ratio",
    ) -> tuple[Path, float]:
        captured["completion_event_measure"] = completion_event_measure
        rows = [{"event_idx": 0, "h": h, "combined_no_bh": 1.0 + h} for h in h_values]
        work_root.mkdir(parents=True, exist_ok=True)
        diag_csv = work_root / "diag.csv"
        pd.DataFrame(rows).to_csv(diag_csv, index=False)
        return diag_csv, 0.01

    monkeypatch.setattr(
        c1d.MirrorUniverseGenerator, "host_pool_for_sigma_scale", _fake_host_pool_for_sigma_scale
    )
    monkeypatch.setattr(c1d.MirrorUniverseGenerator, "draw_realization", _fake_draw_realization)
    monkeypatch.setattr(c1d, "build_bsel_selection_objects", lambda: (None, None))
    # [P3-2D] (b0i2d): same stub convention as build_bsel_selection_objects
    # above -- this test exercises the arm -> kwarg wiring only, never the
    # real SimulationDetectionProbability/injection-pool construction.
    monkeypatch.setattr(c1d, "build_b0i_2d_selection_objects", lambda: (None, None, None))
    # PA-2 (b0i): the runtime rate-weight parity gate needs the REAL pinned
    # catalogue -- stubbed here too (same convention as build_bsel_selection_objects
    # above), since this test is exercising the arm -> kwarg wiring only.
    monkeypatch.setattr(c1d, "_verify_rate_weight_parity", lambda *args, **kwargs: 0)
    monkeypatch.setattr(c1d, "run_mirror_seed_inprocess", _fake_run_mirror_seed_inprocess)
    monkeypatch.setattr(c1d, "check_reduced_catalogue_pin", lambda: True)

    for arm, expected in c1d.ARM_EVENT_MEASURE.items():
        captured.clear()
        out_dir = tmp_path / f"out_{arm}"
        record_path = c1d.run_arm_seed(tmp_path / f"work_{arm}", arm, 900101, out_dir)
        assert captured["completion_event_measure"] == expected, arm
        record = json.loads(record_path.read_text())
        assert record["event_measure"] == expected, arm
        assert record["arm"] == arm

    # The one arm the amendment actually changes.
    assert c1d.ARM_EVENT_MEASURE["bden"] == "data"
    # Every other arm is provably unchanged (still "ratio").
    assert all(v == "ratio" for k, v in c1d.ARM_EVENT_MEASURE.items() if k != "bden")


def test_run_arm_seed_unknown_arm_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        c1d.run_arm_seed(tmp_path / "work", "not_a_real_arm", 900101, tmp_path / "out")


def test_run_arm_seed_idempotent_skips_existing_output(tmp_path: Path) -> None:
    """The idempotency guard fires BEFORE any catalogue/evaluate() work.

    Pre-seeds the output JSON with a sentinel and checks it is returned
    byte-for-byte unchanged -- proves the skip path never touches the (here
    absent/invalid) pinned production inputs, so this test needs no pool.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_path = out_dir / "b0_seed900101.json"
    out_path.write_text('{"sentinel": true}')

    result = c1d.run_arm_seed(tmp_path / "work", "b0", 900101, out_dir)
    assert result == out_path
    assert json.loads(out_path.read_text()) == {"sentinel": True}


def test_compute_full_log_posterior_vector_shape_and_values(tmp_path: Path) -> None:
    """Aggregation matches :func:`compute_seed_statistics`'s Sigma-log convention."""
    grid = (0.60, 0.61, 0.62)
    rows = []
    for event_idx in range(3):
        for h in grid:
            rows.append({"event_idx": event_idx, "h": h, "combined_no_bh": 1.0 + h})
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "diag.csv"
    df.to_csv(csv_path, index=False)

    h_grid, log_posterior = c1d.compute_full_log_posterior_vector(csv_path, h_grid=grid)
    assert h_grid == list(grid)
    assert len(log_posterior) == len(grid)
    expected = [3.0 * np.log(1.0 + h) for h in grid]
    assert log_posterior == pytest.approx(expected)


def test_area_scale_scales_sky_covariance(tmp_path_factory: pytest.TempPathFactory) -> None:
    """E-DEN mechanism: area_scale multiplies the WHOLE (phi,theta) cov sub-block.

    Drawing with area_scale=s about a fixed host+seed must reproduce the
    area_scale=1 draw's offset scaled by sqrt(s) (Cholesky of s*cov = sqrt(s)
    * Cholesky of cov, for the SAME underlying standard-normal draw) -- the
    registered "sub-block scaled by area_scale before the recenter draw"
    mechanism (task spec item 1).
    """
    n = 50
    pool = _make_host_pool(n_pool=n)
    donor_csv = _make_donor_csv(tmp_path_factory, n_rows=n)

    cfg1 = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv, area_scale=1.0)
    gen1 = c1d.MirrorUniverseGenerator(cfg1)
    events1 = gen1.draw_realization(seed=11, host_pool=pool)

    scale = 4.0  # perfect square -> exact sqrt(scale) factor, no float noise
    cfg2 = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv, area_scale=scale)
    gen2 = c1d.MirrorUniverseGenerator(cfg2)
    events2 = gen2.draw_realization(seed=11, host_pool=pool)

    host_idx = events1["host_galaxy_index"].to_numpy()
    assert (events2["host_galaxy_index"].to_numpy() == host_idx).all()
    # qS is clipped to [0, pi] post-draw (draw_realization); the scale=4x
    # draw can push a FEW events past that boundary where the exact
    # sqrt(scale) relation no longer holds post-clip -- exclude qS draws
    # sitting exactly on the clip boundary from the comparison (phiS wraps
    # mod 2*pi instead, so it never clips).
    unclipped = (events2["qS"].to_numpy() > 0.0) & (events2["qS"].to_numpy() < np.pi)
    assert unclipped.sum() >= n - 3  # clipping should be rare at these variances

    d_phi_1 = (events1["phiS"].to_numpy() - pool.phiS[host_idx])[unclipped]
    d_phi_2 = (events2["phiS"].to_numpy() - pool.phiS[host_idx])[unclipped]
    np.testing.assert_allclose(d_phi_2, d_phi_1 * np.sqrt(scale), atol=1e-10)
    d_theta_1 = (events1["qS"].to_numpy() - pool.qS[host_idx])[unclipped]
    d_theta_2 = (events2["qS"].to_numpy() - pool.qS[host_idx])[unclipped]
    np.testing.assert_allclose(d_theta_2, d_theta_1 * np.sqrt(scale), atol=1e-10)


# ── AMENDMENT A-2: population-model host draw (B-OUT) tests ─────────────────


def test_population_z_weights_matches_w_pop_eff_form() -> None:
    """w_pop(z) = dV_c/dz(z, h) / (1+z), the bare _w_pop_eff functional form."""
    z = np.array([0.05, 0.3, 0.8, 1.4])
    w = c1d.population_z_weights(z, h=0.73)
    expected = c1d.comoving_volume_element(z, h=0.73) / (1.0 + z)
    np.testing.assert_allclose(w, expected)
    assert np.all(w > 0.0)


def test_draw_population_redshifts_is_deterministic() -> None:
    rng_a = np.random.default_rng(2026)
    rng_b = np.random.default_rng(2026)
    z_a = c1d.draw_population_redshifts(rng_a, 500)
    z_b = c1d.draw_population_redshifts(rng_b, 500)
    np.testing.assert_array_equal(z_a, z_b)


def test_draw_population_redshifts_within_domain() -> None:
    rng = np.random.default_rng(7)
    z = c1d.draw_population_redshifts(rng, 2000)
    assert z.min() >= c1d.POPULATION_Z_MIN
    assert z.max() <= c1d.POPULATION_Z_MAX


def test_draw_population_redshifts_matches_w_pop_quantiles() -> None:
    """Coarse quantile check: drawn z's median matches the w_pop CDF's median.

    A full KS test against a fine reference CDF built independently from
    :func:`population_z_weights` (same functional form the harness draws
    from, so this is a self-consistency/implementation check on the
    inverse-CDF machinery, not an independent physics check).
    """
    rng = np.random.default_rng(11)
    n = 20000
    z = c1d.draw_population_redshifts(rng, n)

    z_ref = np.linspace(c1d.POPULATION_Z_MIN, c1d.POPULATION_Z_MAX, 20001)
    w_ref = c1d.population_z_weights(z_ref, h=c1d.H_TRUE)
    seg = 0.5 * (w_ref[1:] + w_ref[:-1]) * np.diff(z_ref)
    cdf_ref = np.concatenate(([0.0], np.cumsum(seg)))
    cdf_ref /= cdf_ref[-1]

    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        expected = float(np.interp(q, cdf_ref, z_ref))
        drawn = float(np.quantile(z, q))
        # w_pop(z) ~ z^2 near the origin and turns over near z~1 -- the
        # quantile spacing is O(0.01-0.1) over this domain; a coarse
        # tolerance (Monte Carlo noise at n=20000 plus grid discretization)
        # is the registered check, not exact agreement.
        assert abs(drawn - expected) < 0.05, (q, drawn, expected)


def test_draw_isotropic_sky_domain_and_moments() -> None:
    rng = np.random.default_rng(3)
    n = 20000
    phi, q = c1d.draw_isotropic_sky(rng, n)
    assert phi.min() >= 0.0
    assert phi.max() <= 2.0 * np.pi
    assert q.min() >= 0.0
    assert q.max() <= np.pi
    # Isotropic: cos(qS) uniform on [-1, 1] -> mean(cos(qS)) ~ 0.
    assert abs(float(np.mean(np.cos(q)))) < 0.02
    # phiS uniform on [0, 2pi] -> mean ~ pi.
    assert abs(float(np.mean(phi)) - np.pi) < 0.1


def test_draw_isotropic_sky_is_deterministic() -> None:
    rng_a = np.random.default_rng(99)
    rng_b = np.random.default_rng(99)
    phi_a, q_a = c1d.draw_isotropic_sky(rng_a, 100)
    phi_b, q_b = c1d.draw_isotropic_sky(rng_b, 100)
    np.testing.assert_array_equal(phi_a, phi_b)
    np.testing.assert_array_equal(q_a, q_b)


def test_bout_draw_realization_never_injects_host_into_candidates(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """B-OUT: host_galaxy_index=-1 / in_catalog=False for EVERY event.

    This is the exact production "dark"/completion-leg bookkeeping
    convention (bayesian_statistics.py:4485) -- host_galaxy_index=-1 means
    the drawn host is, by construction, never a candidate-set member
    (candidates come only from the real GLADE BallTree search, keyed on
    position/redshift, never on this column).
    """
    n = 300
    donor_csv = _make_donor_csv(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    events = gen.draw_realization(seed=42, host_mode="population")

    assert (events["host_galaxy_index"].to_numpy() == -1).all()
    assert not events["in_catalog"].any()
    host_in_catalogue_fraction = float((events["host_galaxy_index"].to_numpy() >= 0).mean())
    assert host_in_catalogue_fraction == 0.0


def test_bout_draw_realization_host_pool_argument_is_ignored(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """host_pool is accepted but unused in the population branch (no crash, no leakage)."""
    n = 50
    donor_csv = _make_donor_csv(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    pool = _make_host_pool(n_pool=n)
    events = gen.draw_realization(seed=1, host_pool=pool, host_mode="population")
    assert (events["host_galaxy_index"].to_numpy() == -1).all()


def test_bout_draw_realization_is_deterministic(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    n = 40
    donor_csv = _make_donor_csv(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    a = gen.draw_realization(seed=5, host_mode="population")
    b = gen.draw_realization(seed=5, host_mode="population")
    pd.testing.assert_frame_equal(a, b)


def test_draw_realization_unknown_host_mode_raises(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    n = 10
    donor_csv = _make_donor_csv(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    with pytest.raises(ValueError):
        gen.draw_realization(seed=1, host_mode="not_a_real_mode")  # type: ignore[arg-type]


# ── AMENDMENT A-3: estimator-self-consistent host draw (B-SEL) tests ────────
#
# These use SYNTHETIC completeness/survival test doubles (never the real
# GLADE from_cache_or_build()/precompute_phi_marginal_survival() objects,
# which need the pinned completeness cache + injection pool and are NOT
# fast) so the sampler MACHINERY (determinism, domain, the direction of the
# selection effect, weight positivity) is exercised in well under a second,
# pool-free. build_bsel_selection_objects() itself (which DOES need the
# pinned inputs) is exercised only by the fleet CLI (--stage arm --arm
# bsel), not by this test module -- same convention as G-0/G-1/G-2.


class _FakeIncompleteness:
    """Test double satisfying ``CompletenessModel``: ``f_bar(z) = clip(z, 0, 1)``
    -- completeness DECREASES with z (i.e. ``1 - f_bar`` INCREASES with z),
    mimicking the real catalogue's less-complete-at-high-z behaviour without
    reading the pinned completeness cache.
    """

    def f_bar(self, z: float | np.ndarray, h: float = c1d.H_TRUE) -> float | np.ndarray:
        return np.clip(np.asarray(z, dtype=np.float64), 0.0, 1.0)

    def f_k(self, z: float | np.ndarray, k: int, h: float = c1d.H_TRUE) -> float | np.ndarray:
        return self.f_bar(z, h)

    def ang2pix(self, phi: float, theta: float) -> int:
        return 0

    def get_completeness_at_redshift(
        self, z: float | np.ndarray, h: float = c1d.H_TRUE
    ) -> float | np.ndarray:
        return self.f_bar(z, h)


def _fake_phi_survival_table(
    h: float = c1d.H_TRUE, z_max: float = 1.0
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Test double survival table: ``S_bar_phi(z) = exp(-3z)`` -- declines
    with z, mimicking real detection survival's falloff, without building a
    ``SimulationDetectionProbability``/injection-pool grid.
    """
    z_grid = np.linspace(0.0, z_max, 500)
    s_phi = np.exp(-3.0 * z_grid)
    return {h: (z_grid, s_phi)}


def test_selected_population_z_weights_finite_nonneg() -> None:
    completeness = _FakeIncompleteness()
    table = _fake_phi_survival_table()
    z = np.linspace(c1d.POPULATION_Z_MIN, c1d.POPULATION_Z_MAX, 1000)
    w = c1d.selected_population_z_weights(z, completeness, table, h=c1d.H_TRUE)
    assert w.shape == z.shape
    assert np.all(np.isfinite(w))
    assert np.all(w >= 0.0)
    assert w.sum() > 0.0
    # Beyond the survival table's domain (z > z_max=1.0 here), S_bar_phi is
    # read as 0 (production's "undetectable beyond the table" convention),
    # so the weight is exactly 0 there.
    assert np.all(w[z > 1.0] == 0.0)


def test_selected_population_z_weights_missing_h_raises() -> None:
    completeness = _FakeIncompleteness()
    table = _fake_phi_survival_table()
    with pytest.raises(KeyError):
        c1d.selected_population_z_weights(np.array([0.1]), completeness, table, h=0.9)


def test_draw_selected_population_redshifts_is_deterministic() -> None:
    completeness = _FakeIncompleteness()
    table = _fake_phi_survival_table()
    rng_a = np.random.default_rng(2026)
    rng_b = np.random.default_rng(2026)
    z_a = c1d.draw_selected_population_redshifts(rng_a, 500, completeness, table)
    z_b = c1d.draw_selected_population_redshifts(rng_b, 500, completeness, table)
    np.testing.assert_array_equal(z_a, z_b)


def test_draw_selected_population_redshifts_within_domain() -> None:
    completeness = _FakeIncompleteness()
    table = _fake_phi_survival_table()
    rng = np.random.default_rng(7)
    z = c1d.draw_selected_population_redshifts(rng, 2000, completeness, table)
    assert z.min() >= c1d.POPULATION_Z_MIN
    assert z.max() <= c1d.POPULATION_Z_MAX


def test_selected_population_draw_median_below_bare_population_draw() -> None:
    """The registered expectation (AMENDMENT A-3): selection suppresses high z.

    ``1 - f_bar`` increases with z (less complete at high z) but
    ``S_bar_phi`` decreases with z (harder to detect at high z) more
    steeply over this test double's domain -- net effect: the weighted
    (population x (1-completeness) x survival) draw's median sits BELOW the
    bare population draw's median, exactly the "selection suppresses high
    z" signature the amendment's isolation-test logic depends on.
    """
    completeness = _FakeIncompleteness()
    table = _fake_phi_survival_table()
    n = 20000
    rng_weighted = np.random.default_rng(123)
    z_weighted = c1d.draw_selected_population_redshifts(rng_weighted, n, completeness, table)
    rng_unweighted = np.random.default_rng(123)
    z_unweighted = c1d.draw_population_redshifts(rng_unweighted, n)
    assert float(np.median(z_weighted)) < float(np.median(z_unweighted))


def test_inverse_cdf_draw_helper_deterministic_and_within_domain() -> None:
    """The shared inverse-CDF machinery (:func:`_inverse_cdf_draw`) underlying
    both :func:`draw_population_redshifts` (B-OUT) and
    :func:`draw_selected_population_redshifts` (B-SEL): deterministic given
    the same rng state, and every draw stays within the supplied grid's
    domain (implies the internal CDF normalization is well-formed)."""
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    z_grid = np.linspace(0.0, 2.0, 200)
    w = np.exp(-z_grid)  # arbitrary positive, non-normalized weight
    z_a = c1d._inverse_cdf_draw(rng_a, 1000, z_grid, w)
    z_b = c1d._inverse_cdf_draw(rng_b, 1000, z_grid, w)
    np.testing.assert_array_equal(z_a, z_b)
    assert z_a.min() >= z_grid[0]
    assert z_a.max() <= z_grid[-1]


def test_inverse_cdf_draw_helper_raises_on_nonpositive_weights() -> None:
    rng = np.random.default_rng(1)
    z_grid = np.linspace(0.0, 1.0, 50)
    with pytest.raises(ValueError):
        c1d._inverse_cdf_draw(rng, 10, z_grid, np.zeros_like(z_grid))


def test_bsel_draw_realization_requires_selection_objects(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """host_mode='population_selected' without completeness/phi_survival_table raises."""
    n = 10
    donor_csv = _make_donor_csv(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    with pytest.raises(ValueError):
        gen.draw_realization(seed=1, host_mode="population_selected")


def test_bsel_draw_realization_never_injects_host_into_candidates(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """B-SEL, like B-OUT: host_galaxy_index=-1 / in_catalog=False for EVERY event."""
    n = 300
    donor_csv = _make_donor_csv(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    completeness = _FakeIncompleteness()
    table = _fake_phi_survival_table()
    events = gen.draw_realization(
        seed=42,
        host_mode="population_selected",
        completeness=completeness,
        phi_survival_table=table,
    )
    assert (events["host_galaxy_index"].to_numpy() == -1).all()
    assert not events["in_catalog"].any()


def test_bsel_draw_realization_is_deterministic(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    n = 40
    donor_csv = _make_donor_csv(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    completeness = _FakeIncompleteness()
    table = _fake_phi_survival_table()
    a = gen.draw_realization(
        seed=5, host_mode="population_selected", completeness=completeness, phi_survival_table=table
    )
    b = gen.draw_realization(
        seed=5, host_mode="population_selected", completeness=completeness, phi_survival_table=table
    )
    pd.testing.assert_frame_equal(a, b)


def test_bsel_draw_realization_records_host_z_quantiles_diagnostic(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """last_diagnostics carries both quantile sets, weighted strictly below
    unweighted at the median (the registered selection-suppresses-high-z
    signature, task spec item 2)."""
    n = 2000
    donor_csv = _make_donor_csv(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    completeness = _FakeIncompleteness()
    table = _fake_phi_survival_table()
    assert gen.last_diagnostics == {}
    gen.draw_realization(
        seed=17,
        host_mode="population_selected",
        completeness=completeness,
        phi_survival_table=table,
    )
    diag = gen.last_diagnostics
    assert diag["quantile_levels"] == [0.05, 0.25, 0.5, 0.75, 0.95]
    weighted = diag["host_z_quantiles_weighted"]
    unweighted = diag["host_z_quantiles_unweighted_population"]
    assert len(weighted) == 5
    assert len(unweighted) == 5
    median_idx = diag["quantile_levels"].index(0.5)
    assert weighted[median_idx] < unweighted[median_idx]
    # A "catalogue"-mode draw does not touch last_diagnostics (reset to {}
    # at the top of every draw_realization call).
    pool = _make_host_pool(n_pool=n)
    gen.draw_realization(seed=17, host_pool=pool, host_mode="catalogue")
    assert gen.last_diagnostics == {}


# ── AMENDMENT A-6 D-1 diagnostic: pool-free (same convention as the bsel
# draw-realization tests above -- monkeypatch build_bsel_selection_objects
# with a fake completeness/survival table instead of touching the pinned
# injection pool/completeness cache) ─────────────────────────────────────


def test_max_cdf_gap_zero_for_sample_drawn_from_the_model_density() -> None:
    """A large sample drawn (inverse-CDF) FROM the model density itself has
    a max CDF gap that shrinks with n (KS statistic sanity check)."""
    completeness = _FakeIncompleteness()
    table = _fake_phi_survival_table(z_max=1.0)
    z_grid = np.linspace(0.0, 1.0, 2000)
    density = c1d.selected_population_z_weights(z_grid, completeness, table, h=c1d.H_TRUE)
    rng = np.random.default_rng(11)
    sample = c1d._inverse_cdf_draw(rng, 20000, z_grid, density)
    gap = c1d._max_cdf_gap(sample, z_grid, density)
    assert 0.0 <= gap < 0.02


def test_max_cdf_gap_large_for_a_shifted_sample() -> None:
    """A sample concentrated away from the model density's support scores a
    large gap -- sanity check that the statistic actually detects mismatch."""
    completeness = _FakeIncompleteness()
    table = _fake_phi_survival_table(z_max=1.0)
    z_grid = np.linspace(0.0, 1.0, 2000)
    density = c1d.selected_population_z_weights(z_grid, completeness, table, h=c1d.H_TRUE)
    shifted_sample = np.full(500, 0.95)
    gap = c1d._max_cdf_gap(shifted_sample, z_grid, density)
    assert gap > 0.5


def test_max_cdf_gap_empty_sample_is_nan() -> None:
    z_grid = np.linspace(0.0, 1.0, 10)
    density = np.ones_like(z_grid)
    assert np.isnan(c1d._max_cdf_gap(np.array([]), z_grid, density))


def test_run_d1_premise_check_pool_free(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The D-1 diagnostic end to end, without the pinned injection pool.

    Monkeypatches :func:`~darksiren_emri.validation.correspondence_1d.build_bsel_selection_objects`
    with the same fake completeness/survival-table doubles the bsel draw-
    realization tests use above -- runs in well under a second.
    """
    n = 200
    donor_csv = _make_donor_csv(tmp_path_factory, n_rows=n)
    completeness = _FakeIncompleteness()
    table = _fake_phi_survival_table(z_max=1.0)
    monkeypatch.setattr(c1d, "build_bsel_selection_objects", lambda: (completeness, table))

    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    result = c1d.run_d1_premise_check(seed=900101, n_model_grid=500, config=cfg)

    assert result.arm == "bsel"
    assert result.seed == 900101
    assert result.n_drawn == n
    assert 0 <= result.n_surviving <= n
    assert result.survival_fraction == pytest.approx(result.n_surviving / n)
    assert result.verdict in ("MIRROR-MATCHED", "MIRROR-MISMATCHED")
    assert (result.verdict == "MIRROR-MATCHED") == (
        result.max_cdf_gap_surviving_vs_model <= c1d.D1_CDF_GAP_BAND
    )
    assert result.drawn_vs_model_anomaly == (
        result.max_cdf_gap_drawn_vs_model > c1d.D1_CDF_GAP_BAND
    )
    for key in ("drawn", "surviving", "model"):
        assert len(result.z_quantiles[key]) == 5
    assert result.quantile_levels == [0.05, 0.25, 0.5, 0.75, 0.95]
    assert result.elapsed_s >= 0.0


def test_run_d1_premise_check_is_deterministic(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    n = 100
    donor_csv = _make_donor_csv(tmp_path_factory, n_rows=n)
    completeness = _FakeIncompleteness()
    table = _fake_phi_survival_table(z_max=1.0)
    monkeypatch.setattr(c1d, "build_bsel_selection_objects", lambda: (completeness, table))
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)

    a = c1d.run_d1_premise_check(seed=42, n_model_grid=300, config=cfg)
    b = c1d.run_d1_premise_check(seed=42, n_model_grid=300, config=cfg)

    assert a.n_surviving == b.n_surviving
    assert a.max_cdf_gap_surviving_vs_model == pytest.approx(b.max_cdf_gap_surviving_vs_model)
    assert a.z_quantiles == b.z_quantiles


# ── PA-2 (prereg PREREGISTRATION_B0_IDENTITY_20260823.md; A20 review
# A20_REVIEW_B0_DESIGN_20260823.md Finding 2) -- the b0i "catalogue_selected"
# host mode. Same pool-free convention as the bsel tests above: synthetic
# completeness/survival test doubles, never the real GLADE
# from_cache_or_build()/precompute_phi_marginal_survival() objects.


def _make_host_pool_with_mass(n_pool: int = _N_HOST_POOL) -> c1d.HostPool:
    """The SAME pool :func:`_make_host_pool` builds (identical rng draws for
    phiS/qS/z/z_error), plus a source-frame BH mass column in
    ``[1e5, 1e6]`` M_sun (well inside ``[M_SOURCE_FRAME_MIN, M_SOURCE_FRAME_MAX]``
    = ``[1e4, 1e7]``, so :func:`~darksiren_emri.emri_rate.R_eff_per_mbh` is
    well-defined for every row)."""
    rng = np.random.default_rng(5678)
    return c1d.HostPool(
        phiS=rng.uniform(0.0, 2 * np.pi, n_pool),
        qS=rng.uniform(0.1, np.pi - 0.1, n_pool),
        z=rng.uniform(0.01, 0.3, n_pool),
        z_error=rng.uniform(0.001, 0.04, n_pool),
        n=n_pool,
        M=rng.uniform(1.0e5, 1.0e6, n_pool),
    )


def test_catalogue_selected_draw_realization_requires_phi_survival_table(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """(a) Mode guard: host_mode='catalogue_selected' without phi_survival_table raises."""
    n = 10
    donor_csv = _make_donor_csv(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    pool = _make_host_pool_with_mass()
    with pytest.raises(ValueError):
        gen.draw_realization(seed=1, host_pool=pool, host_mode="catalogue_selected")


def test_catalogue_selected_host_draw_weights_requires_mass() -> None:
    """(a) Mode guard: a HostPool without M (the default) raises for the PA-2 weighting."""
    pool = _make_host_pool()  # no M column
    table = _fake_phi_survival_table()
    completeness = _FakeIncompleteness()
    with pytest.raises(ValueError):
        c1d.catalogue_selected_host_draw_weights(pool, table, completeness)


def test_catalogue_selected_draw_realization_is_deterministic(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """(b) Determinism under a fixed seed (host draw + z_true draw + row/noise draws)."""
    n = 8
    donor_csv = _make_donor_csv(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    pool = _make_host_pool_with_mass()
    table = _fake_phi_survival_table(z_max=1.0)
    completeness = _FakeIncompleteness()
    a = gen.draw_realization(
        seed=5,
        host_pool=pool,
        host_mode="catalogue_selected",
        completeness=completeness,
        phi_survival_table=table,
    )
    b = gen.draw_realization(
        seed=5,
        host_pool=pool,
        host_mode="catalogue_selected",
        completeness=completeness,
        phi_survival_table=table,
    )
    pd.testing.assert_frame_equal(a, b)


def test_catalogue_selected_draw_realization_seed_sensitivity(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """(b) Different seeds must not reproduce the same z_true draw."""
    n = 8
    donor_csv = _make_donor_csv(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    pool = _make_host_pool_with_mass()
    table = _fake_phi_survival_table(z_max=1.0)
    completeness = _FakeIncompleteness()
    a = gen.draw_realization(
        seed=5,
        host_pool=pool,
        host_mode="catalogue_selected",
        completeness=completeness,
        phi_survival_table=table,
    )
    b = gen.draw_realization(
        seed=6,
        host_pool=pool,
        host_mode="catalogue_selected",
        completeness=completeness,
        phi_survival_table=table,
    )
    assert not a["z_true"].equals(b["z_true"])


def test_catalogue_selected_draw_realization_records_pa2_columns(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """(item 4) host draw mode / drawn z_true / host row index / S̃_φ,g are recorded."""
    n = 8
    donor_csv = _make_donor_csv(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    pool = _make_host_pool_with_mass()
    table = _fake_phi_survival_table(z_max=1.0)
    completeness = _FakeIncompleteness()
    events = gen.draw_realization(
        seed=9,
        host_pool=pool,
        host_mode="catalogue_selected",
        completeness=completeness,
        phi_survival_table=table,
    )
    assert (events["host_draw_mode"] == "catalogue_selected").all()
    assert (events["host_galaxy_index"].to_numpy() >= 0).all()
    assert bool(events["in_catalog"].all())
    _, _w_g, s_tilde = c1d.catalogue_selected_host_draw_weights(pool, table, completeness)
    host_idx = events["host_galaxy_index"].to_numpy()
    np.testing.assert_allclose(
        events["s_tilde_phi_host"].to_numpy(), s_tilde[host_idx], rtol=0.0, atol=1e-15
    )
    # z_true differs from the listed catalogue z (the mode's whole point --
    # Finding 2(iii)'s refuted "z_true := listed z" convention).
    assert not np.allclose(events["z_true"].to_numpy(), pool.z[host_idx])
    # true_d_L (item c, unchanged convention) is dist(z_true, H_TRUE) -- NOT
    # dist(listed z, H_TRUE) -- so the observed luminosity_distance must sit
    # close to dist(z_true), within a generous Gpc-scale sanity bound (the
    # donor rows' own noise sigma is at most O(0.05) per the synthetic donor
    # CSV's construction, so any O(1) mismatch would mean d_L was built from
    # the wrong redshift).
    true_d_l_from_z_true = c1d.dist_vectorized(events["z_true"].to_numpy(), h=c1d.H_TRUE)
    obs_d_l = events["luminosity_distance"].to_numpy()
    assert np.all(np.abs(obs_d_l - true_d_l_from_z_true) < 1.0)


def _independent_w_pop_f_k(
    z: float, phi: float, theta: float, completeness: "_FakeIncompleteness", h: float
) -> float:
    """Independent (non-vectorized, ``scipy.integrate.quad``-friendly)
    recomputation of ``w_pop(z) * f_k(z)`` -- calls
    :func:`c1d.comoving_volume_element`/``completeness.f_k`` directly, never
    the module's own :func:`c1d._kernel_w_pop_eff`/``_completeness_at_host_nodes``
    batch helpers, so this is a genuine independent check of the PA-11 kernel
    factors, not a self-consistency echo."""
    w_pop = float(c1d.comoving_volume_element(z, h=h)) / (1.0 + z)
    pixel = completeness.ang2pix(phi, theta)
    f_k = float(completeness.f_k(z, pixel, h))
    return w_pop * f_k


def test_catalogue_selected_host_draw_weights_matches_independent_computation() -> None:
    """(c) Draw weights match w_g*S̃_φ,g computed independently.

    PA-11 (A20_REVIEW_B0_IMPL_20260823.md Finding 1 FATAL fix): the module's
    ``S̃_φ,g`` now uses the estimator's volume_deconv+C7 kernel
    ``N(z;z_g,sigma)*w_pop(z)*f_k(z)``, window-renormalized (Z_g) -- not a
    bare Gaussian. This check recomputes that SAME integral independently
    per host via ``scipy.integrate.quad`` (never the module's own
    Gauss-Legendre quadrature or its ``_kernel_w_pop_eff``/
    ``_completeness_at_host_nodes`` helpers), including the ``w_pop(z)*f_k(z)``
    factors -- a genuine recomputation, not an echo of the implementation.
    """
    pool = _make_host_pool_with_mass(n_pool=8)  # small n: per-host quad is O(host)
    table = _fake_phi_survival_table(z_max=2.0)  # S_bar_phi(z) = exp(-3z), non-trivial
    completeness = _FakeIncompleteness()

    normalized, w_g, s_tilde = c1d.catalogue_selected_host_draw_weights(pool, table, completeness)

    # Independent w_g: the production leaf, called directly (not re-derived).
    assert pool.M is not None
    w_g_ref = R_eff_per_mbh(pool.M) / (1.0 + pool.z)
    np.testing.assert_allclose(w_g, w_g_ref, rtol=1e-14)

    # Independent S̃_φ,g: per-host scipy.integrate.quad over the SAME
    # +/-4sigma/1e-6-floored window, including w_pop(z)*f_k(z) and the Z_g
    # window renormalization -- computed with the table's own S_bar_phi
    # (data input, not reimplemented logic).
    z_grid, s_phi_grid = table[c1d.H_TRUE]
    z_error_eff = pool.z_error.copy()  # SIGMA_V_PEC_KM_S == 0.0, so eff == raw here
    lower = np.clip(pool.z - 4.0 * z_error_eff, 1.0e-6, None)
    upper = pool.z + 4.0 * z_error_eff
    s_tilde_ref = np.empty(pool.n)
    for i in range(pool.n):
        z_g, sigma = float(pool.z[i]), float(z_error_eff[i])
        phi_i, theta_i = float(pool.phiS[i]), float(pool.qS[i])

        def _numerator_integrand(
            z: float,
            z_g: float = z_g,
            sigma: float = sigma,
            phi_i: float = phi_i,
            theta_i: float = theta_i,
        ) -> float:
            s_bar = float(np.interp(z, z_grid, s_phi_grid))
            return float(
                norm.pdf(z, loc=z_g, scale=sigma)
                * _independent_w_pop_f_k(z, phi_i, theta_i, completeness, c1d.H_TRUE)
                * s_bar
            )

        def _norm_integrand(
            z: float,
            z_g: float = z_g,
            sigma: float = sigma,
            phi_i: float = phi_i,
            theta_i: float = theta_i,
        ) -> float:
            return float(
                norm.pdf(z, loc=z_g, scale=sigma)
                * _independent_w_pop_f_k(z, phi_i, theta_i, completeness, c1d.H_TRUE)
            )

        numerator, _ = quad(_numerator_integrand, lower[i], upper[i], limit=200, epsabs=1e-14)
        z_g_norm, _ = quad(_norm_integrand, lower[i], upper[i], limit=200, epsabs=1e-14)
        s_tilde_ref[i] = numerator / z_g_norm

    np.testing.assert_allclose(s_tilde, s_tilde_ref, rtol=1e-5, atol=1e-12)

    unnormalized_ref = w_g_ref * s_tilde_ref
    normalized_ref = unnormalized_ref / unnormalized_ref.sum()
    max_rel = float(np.max(np.abs(normalized - normalized_ref) / np.abs(normalized_ref)))
    assert max_rel <= 1e-5, max_rel


def test_kernel_smeared_survival_missing_h_raises() -> None:
    pool = _make_host_pool_with_mass()
    table = _fake_phi_survival_table()
    completeness = _FakeIncompleteness()
    with pytest.raises(KeyError):
        c1d.kernel_smeared_survival(
            pool.z, pool.z_error, table, completeness, pool.phiS, pool.qS, h=0.9
        )


def test_draw_kernel_survival_redshifts_matches_model_density() -> None:
    """(d) z_true samples from k_g(z)*S_bar_phi(z) (CDF-gap / moment check).

    All events are hosted at the SAME single (z, z_error) so a large sample
    can be checked against ONE model density via the shared
    :func:`c1d._max_cdf_gap` diagnostic (the same tool the D-1 premise check
    uses for :func:`~darksiren_emri.validation.correspondence_1d.selected_population_z_weights`).
    """
    table = _fake_phi_survival_table(z_max=1.0)  # S_bar_phi(z) = exp(-3z)
    completeness = _FakeIncompleteness()
    z0, z_err0 = 0.2, 0.03
    phi0, theta0 = 1.1, 1.4
    n = 20000
    rng = np.random.default_rng(11)
    host_z = np.full(n, z0)
    host_z_error = np.full(n, z_err0)
    host_phiS = np.full(n, phi0)
    host_qS = np.full(n, theta0)
    sample = c1d._draw_kernel_survival_redshifts(
        rng, host_z, host_z_error, table, completeness, host_phiS, host_qS, h=c1d.H_TRUE
    )

    lower, upper = c1d._host_kernel_window(np.array([z0]), np.array([z_err0]))
    z_grid = np.linspace(float(lower[0]), float(upper[0]), 4000)
    kernel = norm.pdf(z_grid, loc=z0, scale=z_err0)
    z_table, s_table = table[c1d.H_TRUE]
    s_vals = np.interp(z_grid, z_table, s_table)
    # Independent recomputation of the PA-11 w_pop(z)*f_k(z) factor (never
    # the module's own _kernel_w_pop_eff/_completeness_at_host_nodes helpers).
    pixel0 = completeness.ang2pix(phi0, theta0)
    w_pop_f_k = np.array(
        [
            float(c1d.comoving_volume_element(float(z), h=c1d.H_TRUE))
            / (1.0 + float(z))
            * float(completeness.f_k(float(z), pixel0, c1d.H_TRUE))
            for z in z_grid
        ]
    )
    density = kernel * w_pop_f_k * s_vals

    gap = c1d._max_cdf_gap(sample, z_grid, density)
    assert 0.0 <= gap < 0.02, gap

    # Moment check: the drawn mean should sit close to the model density's mean.
    seg = 0.5 * (density[1:] + density[:-1]) * np.diff(z_grid)
    model_mean = float(np.sum(0.5 * (z_grid[1:] + z_grid[:-1]) * seg) / seg.sum())
    assert abs(float(np.mean(sample)) - model_mean) < 0.002


def test_draw_kernel_survival_redshifts_is_deterministic() -> None:
    table = _fake_phi_survival_table(z_max=1.0)
    completeness = _FakeIncompleteness()
    host_z = np.array([0.1, 0.2, 0.05])
    host_z_error = np.array([0.01, 0.02, 0.005])
    host_phiS = np.array([0.5, 1.5, 2.5])
    host_qS = np.array([0.5, 1.0, 1.5])
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    a = c1d._draw_kernel_survival_redshifts(
        rng_a, host_z, host_z_error, table, completeness, host_phiS, host_qS, h=c1d.H_TRUE
    )
    b = c1d._draw_kernel_survival_redshifts(
        rng_b, host_z, host_z_error, table, completeness, host_phiS, host_qS, h=c1d.H_TRUE
    )
    np.testing.assert_array_equal(a, b)
    assert a.shape == host_z.shape


def test_catalogue_mode_does_not_enter_catalogue_selected_code_path(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(e) Regression guard: host_mode='catalogue' must never touch the new
    PA-2 machinery. Monkeypatches each new entry point to raise; a
    'catalogue' draw must complete unaffected (and gain none of the new
    columns)."""

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("catalogue_selected code path entered by host_mode='catalogue'")

    monkeypatch.setattr(c1d, "catalogue_selected_host_draw_weights", _boom)
    monkeypatch.setattr(c1d, "kernel_smeared_survival", _boom)
    monkeypatch.setattr(c1d, "_draw_kernel_survival_redshifts", _boom)

    n = 5
    donor_csv = _make_donor_csv(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    pool = _make_host_pool_with_mass()
    events = gen.draw_realization(seed=1, host_pool=pool, host_mode="catalogue")
    assert "host_draw_mode" not in events.columns
    assert "z_true" not in events.columns
    assert "s_tilde_phi_host" not in events.columns


def test_catalogue_mode_byte_unchanged_regression(tmp_path: Path) -> None:
    """(e) Fixed-seed regression pin: host_mode='catalogue' output is
    UNCHANGED by the PA-2 addition (values captured from the pre-existing
    "catalogue" branch, which this change does not touch)."""
    rng = np.random.default_rng(5678)
    n_pool = 20
    pool = c1d.HostPool(
        phiS=rng.uniform(0.0, 2 * np.pi, n_pool),
        qS=rng.uniform(0.1, np.pi - 0.1, n_pool),
        z=rng.uniform(0.01, 0.3, n_pool),
        z_error=rng.uniform(0.001, 0.04, n_pool),
        n=n_pool,
    )
    rng2 = np.random.default_rng(1234)
    n_rows = 12
    df = pd.DataFrame(
        {
            "SNR": rng2.uniform(20.0, 80.0, n_rows),
            "luminosity_distance": rng2.uniform(1.0, 3.0, n_rows),
            "phiS": rng2.uniform(0.0, 2 * np.pi, n_rows),
            "qS": rng2.uniform(0.1, np.pi - 0.1, n_rows),
            "delta_luminosity_distance_delta_luminosity_distance": rng2.uniform(
                0.001, 0.05, n_rows
            ),
            "delta_phiS_delta_phiS": rng2.uniform(1e-4, 1e-2, n_rows),
            "delta_qS_delta_qS": rng2.uniform(1e-4, 1e-2, n_rows),
            "delta_phiS_delta_qS": np.zeros(n_rows),
            "host_galaxy_index": -1,
            "in_catalog": False,
            "_coord_frame": "ecliptic_BarycentricTrue_J2000",
            "_cov_frame": "ecliptic_BarycentricTrue_J2000",
        }
    )
    donor_csv = tmp_path / "donor_crb_regr.csv"
    df.to_csv(donor_csv, index=False)

    cfg = c1d.CorrespondenceConfig(n_events=5, crb_reference_csv=str(donor_csv))
    gen = c1d.MirrorUniverseGenerator(cfg)
    events = gen.draw_realization(seed=555, host_pool=pool, host_mode="catalogue")

    expected_luminosity_distance = [
        0.18625746623057443,
        0.06871630539154697,
        0.32309516540067595,
        0.20943400385463906,
        0.8147148069849749,
    ]
    expected_phiS = [
        3.3292432950382462,
        4.748267935276316,
        5.831028711539124,
        4.228419555411897,
        4.006419403472291,
    ]
    expected_qS = [
        1.3044173246251798,
        1.9313731301490729,
        1.0923656799287413,
        0.43830070826936485,
        2.6738108442307853,
    ]
    expected_host_idx = [3, 16, 0, 2, 19]

    np.testing.assert_allclose(
        events["luminosity_distance"].to_numpy(), expected_luminosity_distance, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(events["phiS"].to_numpy(), expected_phiS, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(events["qS"].to_numpy(), expected_qS, rtol=0.0, atol=0.0)
    assert events["host_galaxy_index"].to_numpy().tolist() == expected_host_idx
    assert "host_draw_mode" not in events.columns


# ── [P3-2D] (prereg PREREGISTRATION_P3_2D_20260825.md) -- the b0i-2D venue
# mass-law extension ("catalogue_selected_2d" host mode). Same pool-free,
# synthetic-test-double convention as the PA-2 (b0i) tests above.


class _FakeS4D:
    """Test double for ``SimulationDetectionProbability.detection_probability_with_bh_mass_interpolated``:
    a HIGH constant survival by default (so the rejection-sampling loop
    converges in ~1-2 rounds during tests), or a caller-supplied constant
    (e.g. ``0.0`` to exercise the GATE-ACC-style non-convergence STOP).
    """

    def __init__(self, value: float = 0.9) -> None:
        self.value = value
        self.calls: list[tuple[int, float]] = []  # (batch size, h) per call, for assertions

    def detection_probability_with_bh_mass_interpolated(
        self,
        d_L: np.ndarray,
        M_z: np.ndarray,
        phi: np.ndarray,
        theta: np.ndarray,
        *,
        h: float,
        z: np.ndarray | None = None,
    ) -> np.ndarray:
        d_l_arr = np.atleast_1d(np.asarray(d_L, dtype=np.float64))
        self.calls.append((int(d_l_arr.size), float(h)))
        return np.full_like(d_l_arr, self.value, dtype=np.float64)


def _make_host_pool_with_mass_and_error(n_pool: int = _N_HOST_POOL) -> c1d.HostPool:
    """The SAME pool :func:`_make_host_pool_with_mass` builds, PLUS a source-frame
    BH mass 1-sigma uncertainty column (10% of mass -- well inside the
    Eddington-shift quadrature's guard, ``_eddington_shifted_host_mass_batch``)."""
    pool = _make_host_pool_with_mass(n_pool)
    rng = np.random.default_rng(5679)
    assert pool.M is not None
    m_error = pool.M * rng.uniform(0.05, 0.2, n_pool)
    return dataclasses.replace(pool, M_error=m_error)


def _make_donor_csv_2d(
    tmp_path_factory: pytest.TempPathFactory, n_rows: int = _N_DONOR_ROWS
) -> str:
    """The SAME donor CSV :func:`_make_donor_csv` builds, PLUS the (M,
    delta_M_delta_M, delta_luminosity_distance_delta_M) columns the
    ``"catalogue_selected_2d"`` joint (d_hat, M_hat_z) draw needs."""
    donor_csv = _make_donor_csv(tmp_path_factory, n_rows=n_rows)
    df = pd.read_csv(donor_csv)
    # Independent RNG draws (a fresh generator) so this fixture's mass columns
    # do not perturb the base fixture's own stream/values.
    rng = np.random.default_rng(4321)
    df["M"] = rng.uniform(1.0e5, 1.0e6, n_rows)
    df["delta_M_delta_M"] = rng.uniform(1.0e6, 1.0e8, n_rows)  # (M_sun)^2-scale variance
    # Small, physically-plausible d_L-M correlation (|corr| << 1 -- Cholesky-safe).
    sigma_dl = np.sqrt(df["delta_luminosity_distance_delta_luminosity_distance"].to_numpy())
    sigma_m = np.sqrt(df["delta_M_delta_M"].to_numpy())
    corr = rng.uniform(-0.1, 0.1, n_rows)
    df["delta_luminosity_distance_delta_M"] = corr * sigma_dl * sigma_m
    df.to_csv(donor_csv, index=False)
    return donor_csv


def test_catalogue_selected_2d_draw_realization_requires_phi_survival_table(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Mode guard: host_mode='catalogue_selected_2d' without phi_survival_table raises."""
    n = 6
    donor_csv = _make_donor_csv_2d(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    pool = _make_host_pool_with_mass_and_error()
    with pytest.raises(ValueError, match="phi_survival_table"):
        gen.draw_realization(
            seed=1,
            host_pool=pool,
            host_mode="catalogue_selected_2d",
            completeness=_FakeIncompleteness(),
            detection_probability=_FakeS4D(),  # type: ignore[arg-type]
        )


def test_catalogue_selected_2d_draw_realization_requires_detection_probability(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Mode guard: host_mode='catalogue_selected_2d' without detection_probability raises."""
    n = 6
    donor_csv = _make_donor_csv_2d(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    pool = _make_host_pool_with_mass_and_error()
    table = _fake_phi_survival_table(z_max=1.0)
    with pytest.raises(ValueError, match="detection_probability"):
        gen.draw_realization(
            seed=1,
            host_pool=pool,
            host_mode="catalogue_selected_2d",
            completeness=_FakeIncompleteness(),
            phi_survival_table=table,
        )


def test_catalogue_selected_2d_host_pool_requires_mass_error() -> None:
    """Mode guard: a HostPool with M but no M_error raises for the [P3-2D] latent draw."""
    pool = _make_host_pool_with_mass()  # M set, M_error left None
    table = _fake_phi_survival_table()
    completeness = _FakeIncompleteness()
    host_w, _w_g, s_tilde = c1d.catalogue_selected_host_draw_weights(pool, table, completeness)
    with pytest.raises(ValueError, match="M_error"):
        c1d._draw_2d_accepted_latents(
            np.random.default_rng(1),
            pool,
            host_w,
            s_tilde,
            table,
            completeness,
            _FakeS4D(),  # type: ignore[arg-type]
            5,
        )


def test_catalogue_selected_2d_draw_realization_is_deterministic(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Determinism under a fixed seed (host/z/mass/joint-observation/Bernoulli draws)."""
    n = 8
    donor_csv = _make_donor_csv_2d(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    pool = _make_host_pool_with_mass_and_error()
    table = _fake_phi_survival_table(z_max=1.0)
    completeness = _FakeIncompleteness()
    detection_probability = _FakeS4D(value=0.9)
    a = gen.draw_realization(
        seed=5,
        host_pool=pool,
        host_mode="catalogue_selected_2d",
        completeness=completeness,
        phi_survival_table=table,
        detection_probability=detection_probability,  # type: ignore[arg-type]
    )
    b = gen.draw_realization(
        seed=5,
        host_pool=pool,
        host_mode="catalogue_selected_2d",
        completeness=completeness,
        phi_survival_table=table,
        detection_probability=detection_probability,  # type: ignore[arg-type]
    )
    pd.testing.assert_frame_equal(a, b)


def test_catalogue_selected_2d_draw_realization_seed_sensitivity(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Different seeds must not reproduce the same M_true/z_true draw."""
    n = 8
    donor_csv = _make_donor_csv_2d(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    pool = _make_host_pool_with_mass_and_error()
    table = _fake_phi_survival_table(z_max=1.0)
    completeness = _FakeIncompleteness()
    a = gen.draw_realization(
        seed=5,
        host_pool=pool,
        host_mode="catalogue_selected_2d",
        completeness=completeness,
        phi_survival_table=table,
        detection_probability=_FakeS4D(value=0.9),  # type: ignore[arg-type]
    )
    b = gen.draw_realization(
        seed=6,
        host_pool=pool,
        host_mode="catalogue_selected_2d",
        completeness=completeness,
        phi_survival_table=table,
        detection_probability=_FakeS4D(value=0.9),  # type: ignore[arg-type]
    )
    assert not a["z_true"].equals(b["z_true"])
    assert not a["M_true"].equals(b["M_true"])


def test_catalogue_selected_2d_draw_realization_records_columns(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Task spec item 4: M_true, M_z_true, M_z_obs, s4d_at_truth, link_id, host_draw_mode,
    z_true, s_tilde_phi_host are all recorded, and the "monster event" class is
    structurally absent (M_z_obs sits near its OWN event's M_z_true, not an
    unrelated donor-row value)."""
    n = 10
    donor_csv = _make_donor_csv_2d(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    pool = _make_host_pool_with_mass_and_error()
    table = _fake_phi_survival_table(z_max=1.0)
    completeness = _FakeIncompleteness()
    detection_probability = _FakeS4D(value=0.9)
    events = gen.draw_realization(
        seed=9,
        host_pool=pool,
        host_mode="catalogue_selected_2d",
        completeness=completeness,
        phi_survival_table=table,
        detection_probability=detection_probability,  # type: ignore[arg-type]
    )

    assert (events["host_draw_mode"] == "catalogue_selected_2d").all()
    assert (events["host_galaxy_index"].to_numpy() >= 0).all()
    assert bool(events["in_catalog"].all())
    for col in (
        "M_true",
        "M_z_true",
        "M_z_obs",
        "s4d_at_truth",
        "link_id",
        "z_true",
        "s_tilde_phi_host",
    ):
        assert col in events.columns

    assert np.all(events["M_true"].to_numpy() > 0.0)
    assert np.all(events["M"].to_numpy() == events["M_z_obs"].to_numpy())
    assert np.all((events["s4d_at_truth"].to_numpy() >= 0.0) & (events["s4d_at_truth"] <= 1.0))
    # GATE M2-LINK forensics (structural): link_id indexes the donor CRB pool
    # this event's (d_hat, M_hat_z) covariance was drawn from -- in range and
    # distinct from a "the mass is just whatever the donor row happened to
    # carry" configuration: M_z_obs must sit within a generous many-sigma
    # window of ITS OWN event's M_z_true (never an unrelated donor value --
    # the donor CSV's mass column spans [1e5, 1e6], an entirely different
    # scale from a typical M_z_true here, so a monster mismatch would be
    # obvious as an outlier far outside this window).
    assert np.all(events["link_id"].to_numpy() >= 0)
    assert np.all(events["link_id"].to_numpy() < n)
    diff = np.abs(events["M_z_obs"].to_numpy() - events["M_z_true"].to_numpy())
    sigma_m = np.sqrt(pd.read_csv(donor_csv)["delta_M_delta_M"].to_numpy())
    assert np.all(diff < 15.0 * sigma_m)  # generous many-sigma bound, not a tight physics check


def test_catalogue_selected_2d_rejection_loop_stops_on_zero_survival(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """GATE-ACC-style closed-loop STOP: a venue where S_4D is identically 0 can never
    accept -- the rejection loop must raise RuntimeError, not spin/hang or silently
    under-fill."""
    n = 4
    donor_csv = _make_donor_csv_2d(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    pool = _make_host_pool_with_mass_and_error()
    table = _fake_phi_survival_table(z_max=1.0)
    completeness = _FakeIncompleteness()
    with pytest.raises(RuntimeError, match="did not converge"):
        gen.draw_realization(
            seed=1,
            host_pool=pool,
            host_mode="catalogue_selected_2d",
            completeness=completeness,
            phi_survival_table=table,
            detection_probability=_FakeS4D(value=0.0),  # type: ignore[arg-type]
        )


def test_catalogue_selected_mode_does_not_enter_catalogue_selected_2d_code_path(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression guard: host_mode='catalogue_selected' (the 1D mode) must never touch the
    new [P3-2D] machinery. Monkeypatches the new entry point to raise; a
    'catalogue_selected' draw must complete unaffected (and gain none of the new columns)."""

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "catalogue_selected_2d code path entered by host_mode='catalogue_selected'"
        )

    monkeypatch.setattr(c1d, "_draw_2d_accepted_latents", _boom)

    n = 6
    donor_csv = _make_donor_csv(tmp_path_factory, n_rows=n)
    cfg = c1d.CorrespondenceConfig(n_events=n, crb_reference_csv=donor_csv)
    gen = c1d.MirrorUniverseGenerator(cfg)
    pool = _make_host_pool_with_mass()
    table = _fake_phi_survival_table(z_max=1.0)
    completeness = _FakeIncompleteness()
    events = gen.draw_realization(
        seed=1,
        host_pool=pool,
        host_mode="catalogue_selected",
        completeness=completeness,
        phi_survival_table=table,
    )
    assert (events["host_draw_mode"] == "catalogue_selected").all()
    assert "M_true" not in events.columns
    assert "s4d_at_truth" not in events.columns
    assert "link_id" not in events.columns
