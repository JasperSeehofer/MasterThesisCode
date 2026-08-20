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

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from darksiren_emri.validation import correspondence_1d as c1d

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
    }
    assert set(c1d.ARM_SELECTION_CELL) == set(c1d.ARM_SPECS)
    non_bself = {k: v for k, v in c1d.ARM_SELECTION_CELL.items() if k != "bself"}
    assert set(non_bself.values()) == {"off"}
    assert c1d.ARM_SELECTION_CELL["bself"] == "fused"
    # Matches the runs-of-record basis every pre-A-4 arm is registered against.
    assert c1d.PRODUCTION_FLAGS["--selection_in_completion_numerator"] == "off"


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
    total = sum(len(v) for v in c1d.ARM_SEEDS.values())
    # 25 + 25 + 10 + 10 + 10 + 15 + 2 + 15 + 15 + 15 = 142, the fleet task-list
    # arithmetic (80 pre-A-2 tasks + 17 AMENDMENT A-2 tasks + 15 AMENDMENT
    # A-3 tasks + 15 AMENDMENT A-4 tasks + 15 AMENDMENT A-5 tasks).
    assert total == 142
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
    # Every other arm is provably unchanged (still "off").
    assert all(v == "off" for k, v in c1d.ARM_SELECTION_CELL.items() if k != "bself")


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
