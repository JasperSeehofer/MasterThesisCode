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
