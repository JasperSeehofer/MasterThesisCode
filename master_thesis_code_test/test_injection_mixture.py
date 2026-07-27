"""Tests for the stratified 3-component injection sampling measure (issue #51).

Covers main.injection_campaign's --injection_mixture path with a stubbed
ParameterEstimation (no waveforms): stratum assignment proportions, CSV
schema (per-row ``stratum``), the stratum-'c' source-band rejection /
flat-(u, m) support, the degenerate (1, 0, 0) mixture's bit-identity to the
flag-off run (spawned-rng isolation), and the CLI flag default.

Spec: results/lcat_h_dependence_20260725/campaign_sizing_20260728/
SIZING_ANALYSIS.md §6 (mix3_50_25_25) and §4 (stratified campaign, option 1).
"""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from master_thesis_code.arguments import Arguments, _parse_arguments
from master_thesis_code.constants import (
    HOST_DRAW_Z_MAX,
    M_SOURCE_FRAME_MAX,
    M_SOURCE_FRAME_MIN,
)
from master_thesis_code.cosmological_model import Model1CrossCheck
from master_thesis_code.datamodels.parameter_space import ParameterSpace
from master_thesis_code.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    InternalCatalogColumns,
    ParameterSample,
)
from master_thesis_code.main import injection_campaign


class _StubParameterEstimation:
    """Drop-in for ParameterEstimation: real ParameterSpace, constant SNR."""

    def __init__(
        self, waveform_generation_type: object, parameter_space: ParameterSpace, use_gpu: bool
    ) -> None:
        self.parameter_space = parameter_space

    def compute_signal_to_noise_ratio(self) -> float:
        return 25.0


class _StubCosmologicalModel:
    """Duck-typed Model1CrossCheck: real ParameterSpace + synthetic emcee batches.

    Draws come from an INTERNAL generator (seeded at construction) so two stub
    instances with the same seed produce identical batches regardless of the
    campaign's parent rng — mirroring the fact that the real model consumes
    the parent rng identically in flag-off and (1, 0, 0)-mixture runs.
    """

    def __init__(self, seed: int = 123) -> None:
        self.parameter_space = ParameterSpace()
        self.max_redshift = HOST_DRAW_Z_MAX
        self._rng = np.random.default_rng(seed)

    def sample_emri_events(self, n: int) -> list[ParameterSample]:
        z = self._rng.uniform(0.01, HOST_DRAW_Z_MAX, n)
        log_m = self._rng.uniform(np.log10(M_SOURCE_FRAME_MIN), np.log10(M_SOURCE_FRAME_MAX), n)
        return [
            ParameterSample(M=float(10.0 ** log_m[i]), a=0.9, redshift=float(z[i]))
            for i in range(n)
        ]


class _StubGalaxyCatalog:
    """Duck-typed GalaxyCatalogueHandler exposing only the pruned frame."""

    def __init__(self, n: int = 500, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.reduced_galaxy_catalog = pd.DataFrame(
            {
                InternalCatalogColumns.BH_MASS: 10.0 ** rng.uniform(4.5, 6.8, n),
                InternalCatalogColumns.REDSHIFT: rng.uniform(0.005, 0.3, n),
            }
        )


def _run_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    steps: int,
    seed: int,
    injection_mixture: bool,
    stratum_probs: tuple[float, float, float] = (0.50, 0.25, 0.25),
    with_catalog: bool = True,
    subdir: str = "run",
) -> pd.DataFrame:
    """Run injection_campaign with stubs; return the written CSV as a DataFrame."""
    out_dir = tmp_path / subdir
    out_dir.mkdir(exist_ok=True)
    csv_template = str(out_dir / "injection_h_{h_label}_task_{index}.csv")
    monkeypatch.setattr("master_thesis_code.constants.INJECTION_CSV_PATH", csv_template)
    monkeypatch.setattr(
        "master_thesis_code.parameter_estimation.parameter_estimation.ParameterEstimation",
        _StubParameterEstimation,
    )
    model = cast(Model1CrossCheck, _StubCosmologicalModel(seed=123))
    catalog = cast(GalaxyCatalogueHandler, _StubGalaxyCatalog()) if with_catalog else None
    injection_campaign(
        simulation_steps=steps,
        cosmological_model=model,
        h_value=0.73,
        simulation_index=0,
        rng=np.random.default_rng(seed),
        use_gpu=False,
        galaxy_catalog=catalog,
        injection_mixture=injection_mixture,
        stratum_probs=stratum_probs,
    )
    return pd.read_csv(csv_template.format(h_label="0p73", index=0))


class TestMixtureOff:
    """Default path: pure stratum-a, column present and all-'a'."""

    def test_default_off_all_rows_stratum_a(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        df = _run_campaign(
            tmp_path, monkeypatch, steps=40, seed=1000, injection_mixture=False, with_catalog=False
        )
        assert "stratum" in df.columns
        assert (df["stratum"] == "a").all()
        assert len(df) == 40


class TestMixtureProportionsAndSchema:
    """Realized proportions ~= 50/25/25; per-stratum row semantics."""

    def test_proportions_and_row_semantics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        steps = 400
        df = _run_campaign(tmp_path, monkeypatch, steps=steps, seed=2000, injection_mixture=True)
        assert len(df) == steps
        counts = df["stratum"].value_counts()
        assert set(counts.index) <= {"a", "b", "c"}
        # Binomial sd at p=0.5 is ~0.025 for n=400; +-0.08 is a >3-sigma band.
        assert abs(counts.get("a", 0) / steps - 0.50) < 0.08
        assert abs(counts.get("b", 0) / steps - 0.25) < 0.08
        assert abs(counts.get("c", 0) / steps - 0.25) < 0.08

        # EVERY row: CSV "M" is detector-frame M_z; the implied source mass
        # M_z/(1+z) must lie in the single-source band (no double lift for
        # stratum 'c', standard lift for 'a'/'b').
        m_source = df["M"].to_numpy() / (1.0 + df["z"].to_numpy())
        assert np.all(m_source >= M_SOURCE_FRAME_MIN * (1.0 - 1e-12))
        assert np.all(m_source <= M_SOURCE_FRAME_MAX * (1.0 + 1e-12))
        assert np.all(df["z"].to_numpy() <= HOST_DRAW_Z_MAX)

        # Stratum-'b' rows must be catalogue rows: (z, M_source) pairs drawn
        # from the stub catalogue's own values.
        catalog = _StubGalaxyCatalog()
        cat_m = np.sort(catalog.reduced_galaxy_catalog[InternalCatalogColumns.BH_MASS].to_numpy())
        b_rows = df[df["stratum"] == "b"]
        if len(b_rows) > 0:
            m_src_b = b_rows["M"].to_numpy() / (1.0 + b_rows["z"].to_numpy())
            pos = np.searchsorted(cat_m, m_src_b)
            pos = np.clip(pos, 1, len(cat_m) - 1)
            nearest = np.minimum(np.abs(cat_m[pos] - m_src_b), np.abs(cat_m[pos - 1] - m_src_b))
            assert np.all(nearest / m_src_b < 1e-9)


class TestStratumCFlatSupport:
    """Stratum-'c': flat-(u, m) draw with source-band rejection."""

    def test_source_band_and_reachable_wedge(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        steps = 400
        df = _run_campaign(
            tmp_path,
            monkeypatch,
            steps=steps,
            seed=3000,
            injection_mixture=True,
            stratum_probs=(0.0, 0.0, 1.0),
            with_catalog=False,
        )
        assert (df["stratum"] == "c").all()
        z = df["z"].to_numpy()
        u = np.log1p(z)
        m = np.log10(df["M"].to_numpy())
        m_source = df["M"].to_numpy() / (1.0 + z)

        # Box support in (u, m).
        assert np.all(u >= 0.0)
        assert np.all(u <= np.log1p(HOST_DRAW_Z_MAX) + 1e-12)
        assert np.all(m >= np.log10(M_SOURCE_FRAME_MIN) - 1e-12)
        assert np.all(m <= np.log10(M_SOURCE_FRAME_MAX * (1.0 + HOST_DRAW_Z_MAX)) + 1e-12)

        # NO generated row violates the source band; the unreachable wedge
        # m > log10(M_max) + log10(1+z) yields no rows (rejection).
        assert np.all(m_source >= M_SOURCE_FRAME_MIN * (1.0 - 1e-12))
        assert np.all(m_source <= M_SOURCE_FRAME_MAX * (1.0 + 1e-12))
        assert np.all(m <= np.log10(M_SOURCE_FRAME_MAX) + np.log10(1.0 + z) + 1e-9)

        # Coarse flatness on the reachable region: split the u range in half;
        # both halves must be well-populated (the flat measure puts ~half the
        # reachable box area on each side; a population-weighted measure like
        # 'a' would not look like this at high m).
        lo = int(np.count_nonzero(u < 0.5 * np.log1p(HOST_DRAW_Z_MAX)))
        hi = steps - lo
        assert lo > steps // 5
        assert hi > steps // 5
        # High-m half-box (m > 5.7 = midpoint of [4, 7.4]) is populated —
        # flat in m, unlike the steeply falling population mass function.
        m_mid = 0.5 * (
            np.log10(M_SOURCE_FRAME_MIN) + np.log10(M_SOURCE_FRAME_MAX * (1.0 + HOST_DRAW_Z_MAX))
        )
        assert int(np.count_nonzero(m > m_mid)) > steps // 5


class TestDegenerateMixtureBitIdentity:
    """(1, 0, 0) mixture == flag-off run for the same parent rng stream."""

    def test_p100_matches_flag_off(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        df_off = _run_campaign(
            tmp_path,
            monkeypatch,
            steps=60,
            seed=4242,
            injection_mixture=False,
            with_catalog=False,
            subdir="off",
        )
        df_p100 = _run_campaign(
            tmp_path,
            monkeypatch,
            steps=60,
            seed=4242,
            injection_mixture=True,
            stratum_probs=(1.0, 0.0, 0.0),
            with_catalog=False,
            subdir="p100",
        )
        # Stratum labels/draws come from SPAWNED child generators, so the
        # parent stream — and hence the whole stratum-'a' path — is untouched.
        for col in ("z", "M", "phiS", "qS", "SNR", "luminosity_distance"):
            assert np.array_equal(df_off[col].to_numpy(), df_p100[col].to_numpy()), col
        assert (df_p100["stratum"] == "a").all()


class TestMixtureGuards:
    """Input validation of the mixture path."""

    def test_missing_catalog_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValueError, match="galaxy_catalog"):
            _run_campaign(
                tmp_path,
                monkeypatch,
                steps=10,
                seed=1,
                injection_mixture=True,
                with_catalog=False,
            )

    def test_missing_rng_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        csv_template = str(tmp_path / "injection_h_{h_label}_task_{index}.csv")
        monkeypatch.setattr("master_thesis_code.constants.INJECTION_CSV_PATH", csv_template)
        monkeypatch.setattr(
            "master_thesis_code.parameter_estimation.parameter_estimation.ParameterEstimation",
            _StubParameterEstimation,
        )
        with pytest.raises(ValueError, match="rng"):
            injection_campaign(
                simulation_steps=10,
                cosmological_model=cast(Model1CrossCheck, _StubCosmologicalModel()),
                h_value=0.73,
                simulation_index=0,
                rng=None,
                use_gpu=False,
                galaxy_catalog=cast(GalaxyCatalogueHandler, _StubGalaxyCatalog()),
                injection_mixture=True,
            )


class TestRunMetadataStratumCounts:
    """Realized stratum counts are appended to an existing run_metadata JSON."""

    def test_counts_written(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import json

        meta_path = tmp_path / "run_metadata.json"
        meta_path.write_text(json.dumps({"random_seed": 5}))
        out_dir = tmp_path / "run"
        out_dir.mkdir()
        csv_template = str(out_dir / "injection_h_{h_label}_task_{index}.csv")
        monkeypatch.setattr("master_thesis_code.constants.INJECTION_CSV_PATH", csv_template)
        monkeypatch.setattr(
            "master_thesis_code.parameter_estimation.parameter_estimation.ParameterEstimation",
            _StubParameterEstimation,
        )
        injection_campaign(
            simulation_steps=30,
            cosmological_model=cast(Model1CrossCheck, _StubCosmologicalModel()),
            h_value=0.73,
            simulation_index=0,
            rng=np.random.default_rng(9),
            use_gpu=False,
            galaxy_catalog=cast(GalaxyCatalogueHandler, _StubGalaxyCatalog()),
            injection_mixture=True,
            run_metadata_path=str(meta_path),
        )
        metadata = json.loads(meta_path.read_text())
        assert metadata["injection_mixture"] is True
        counts = metadata["injection_stratum_counts"]
        assert sum(counts.values()) == 30
        assert set(counts.keys()) == {"a", "b", "c"}


class TestInjectionMixtureCliFlag:
    """--injection_mixture flag: default OFF, opt-in, captured in cli_args."""

    def test_default_is_off(self) -> None:
        args = Arguments(_parse_arguments(["."]))
        assert args.injection_mixture is False

    def test_flag_turns_on(self) -> None:
        args = Arguments(_parse_arguments([".", "--injection_mixture"]))
        assert args.injection_mixture is True

    def test_flag_in_cli_args_dict(self) -> None:
        args = Arguments(_parse_arguments([".", "--injection_mixture"]))
        assert args.to_dict()["injection_mixture"] is True
