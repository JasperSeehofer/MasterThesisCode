# Phase 35: Coordinate Bug Characterization — Pattern Map

**Mapped:** 2026-04-21
**Files analyzed:** 6 new / 0 modified
**Analogs found:** 6 / 6

---

## File Classification

| New File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `master_thesis_code_test/test_coordinate_roundtrip.py` | test (RED xfail, BallTree round-trip + astropy ground truth) | request-response (synthetic input → assertion) | `master_thesis_code_test/test_glade_completeness.py` + `master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py` | role-match (xfail is new convention for this repo) |
| `master_thesis_code_test/fixtures/coordinate.py` | test fixture module (in-memory catalog builder + astropy wrapper) | transform (params → DataFrame/tuple) | `master_thesis_code_test/fixtures/evaluation/generate_fixtures.py` | role-match (same directory family, same DataFrame-builder pattern) |
| `.planning/audit_coordinate_bug.md` | audit artifact (markdown) | file I/O (template fill) | `.planning/debug/comparison_current.md` + `master_thesis_code/bayesian_inference/evaluation_report.py:578-609` | exact (same `.planning/` markdown convention; same summary-table shape) |
| `.planning/audit_coordinate_bug.json` | audit artifact (structured sidecar JSON) | file I/O (dict → json.dumps) | `.planning/debug/baseline.json` + `evaluation_report.py:70-91` (`BaselineSnapshot.to_json`) | exact (same `git_commit` + `created_at` pattern) |
| `.planning/audit_coordinate_bug_histogram.png` | figure artifact | transform (1-D array → figure → PNG) | `master_thesis_code/plotting/bayesian_plots.py:plot_detection_redshift_distribution` | exact (histogram factory pattern) |
| Baseline audit generator (e.g. `master_thesis_code_test/audit_coordinate_bug.py` or `scripts/audit_coordinate_bug.py`) | standalone CLI module (reads CSV, writes 3 artifacts) | batch (CSV in → MD + JSON + PNG out) | `scripts/merge_cramer_rao_bounds.py` + `master_thesis_code/bayesian_inference/evaluation_report.py:generate_diagnostic_summary` | role-match (argparse entry-point + JSON+MD sidecar pattern) |

---

## Pattern Assignments

### `master_thesis_code_test/test_coordinate_roundtrip.py` (test — BallTree roundtrip + astropy ground truth)

**Analog primary:** `master_thesis_code_test/test_glade_completeness.py` (class-based, parametrized, docstring contracts)
**Analog secondary:** `master_thesis_code_test/physical_relations_test.py` (plain `def test_*` style, tight assertions)
**xfail convention:** NEW to this test suite (no existing `@pytest.mark.xfail` in the active tree; CHANGELOG notes prior xfails were removed when their bugs were fixed). Precedent-from-history is documented; use the exact reason-string shape from D-01.

**Imports pattern** — mirror `physical_relations_test.py` (lines 1-16) for plain test files; use `test_glade_completeness.py` (lines 1-24) when the test contains a module docstring and class groupings:

```python
# From master_thesis_code_test/test_glade_completeness.py lines 1-24
"""Tests for GLADE+ completeness function and comoving volume element.

Covers:
- Completeness fraction bounds (contract: claim-fz-range, test-fz-bounds)
- Dalya et al. (2022) reference values (contract: claim-fz-correct, test-fz-dalya)
...
References
----------
Dalya et al. (2022), arXiv:2110.06184, Section 3.
"""

import numpy as np
import pytest

from master_thesis_code.constants import GPC_TO_MPC, SPEED_OF_LIGHT_KM_S
from master_thesis_code.galaxy_catalogue.glade_completeness import GladeCatalogCompleteness
from master_thesis_code.physical_relations import comoving_volume_element, dist
```

**Test-class + contract-docstring pattern** (`test_glade_completeness.py` lines 31-55):

```python
class TestCompletenessFractionBounds:
    """Contract: claim-fz-range, test-fz-bounds."""

    def test_completeness_fraction_bounds(self) -> None:
        """f(z) in [0, 1] for z in np.linspace(0, 0.25, 200) at h=0.73."""
        gc = GladeCatalogCompleteness()
        z_arr = np.linspace(0.0, 0.25, 200)
        frac = gc.get_completeness_at_redshift(z_arr, h=0.73)
        assert isinstance(frac, np.ndarray)
        assert np.all(frac >= 0.0), f"Found negative values: {frac[frac < 0]}"
        assert np.all(frac <= 1.0), f"Found values > 1: {frac[frac > 1]}"


class TestCompletenessFractionAtZero:
    """f(z=0) must be 1.0 for any h."""

    def test_completeness_fraction_at_zero_default_h(self) -> None:
        gc = GladeCatalogCompleteness()
        assert gc.get_completeness_at_redshift(0.0) == 1.0

    @pytest.mark.parametrize("h", [0.6, 0.73, 0.86])
    def test_completeness_fraction_at_zero_various_h(self, h: float) -> None:
        gc = GladeCatalogCompleteness()
        assert gc.get_completeness_at_redshift(0.0, h=h) == 1.0
```

**Seed-pinned RNG pattern** (`master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py` lines 18-50):

```python
def _create_synthetic_injection_csv(
    directory: str,
    h_value: float,
    n_rows: int = 200,
    seed: int = 42,
) -> None:
    """Create a synthetic injection CSV with known properties."""
    rng = np.random.default_rng(seed)
    # ...
    z = rng.uniform(0.01, 1.0, size=n_rows)
    qS = rng.uniform(0, np.pi, size=n_rows)  # noqa: N806
```

D-07 locks `np.random.default_rng(seed=42)` for the N=100 random test — this matches the project pattern verbatim.

**xfail marker convention (NEW)** — the only existing precedent is in `CHANGELOG.md` mentioning historical xfails that have since been removed. D-01 locks the shape; use this exact form per the lock:

```python
# New convention — see .planning/phases/35-coordinate-bug-characterization/35-CONTEXT.md D-01
@pytest.mark.xfail(
    strict=True,
    reason="Phase 36 fixes coordinate frame bug — see .planning/milestones/v2.2-...",
)
def test_ball_tree_recovers_ecliptic_equator_galaxy() -> None:
    """BallTree retrieves a synthetic Dec=0° ecliptic-equator galaxy for a query at the same position."""
    # ... assertions describing the CORRECT post-fix behavior
```

**Tolerance/assertion style** (`physical_relations_test.py` lines 87-92):

```python
@pytest.mark.parametrize("z", [0.5, 1.0, 2.0])
def test_dist_round_trip(z: float) -> None:
    """dist_to_redshift(dist(z)) should recover z within 1e-5."""
    d = dist(z)
    z_recovered = dist_to_redshift(d)
    assert abs(z_recovered - z) < 1e-5
```

For the astropy ground-truth test (D-04c, tolerance <0.001°), use the same `abs(...) < tol` idiom.

**Recovery-rate assertion (D-06, ≥99%)** — pattern `assert recovered / total >= 0.99` with an f-string failure message, matching `test_glade_completeness.py` line 40:

```python
assert np.all(frac >= 0.0), f"Found negative values: {frac[frac < 0]}"
# adapted:
assert recovered_count >= 99, f"Recovered {recovered_count}/100 ecliptic-equator galaxies (≥99% required)"
```

---

### `master_thesis_code_test/fixtures/coordinate.py` (fixtures module)

**Analog:** `master_thesis_code_test/fixtures/evaluation/generate_fixtures.py` (the one existing sibling inside `fixtures/`).

**Module docstring + public-helper pattern** (`generate_fixtures.py` lines 1-10):

```python
"""Generate synthetic CSV fixtures for evaluation pipeline integration tests.

Produces three CSV files that mimic the output of the EMRI simulation pipeline:
- synthetic_cramer_rao_bounds.csv (5 detected events)
...

Run as a script to regenerate:
    uv run python master_thesis_code_test/fixtures/evaluation/generate_fixtures.py
"""
```

Apply the same shape for `coordinate.py`: a module docstring explaining what the fixtures produce (synthetic GalaxyCatalogueHandler-compatible DataFrame, astropy ground truth, BallTree builder helper) and which tests consume them (Phase 35 RED tests, Phase 36 post-fix regression).

**Column-list + builder-function pattern** (`generate_fixtures.py` lines 19-47 + `_make_detected_row` lines 60-112):

```python
# 14 EMRI parameter names in CSV order
PARAM_NAMES = [
    "M", "mu", "a", "p0", "e0", "x0",
    "luminosity_distance",
    "qS", "phiS",
    "qK", "phiK",
    "Phi_phi0", "Phi_theta0", "Phi_r0",
]
# ...
def _make_detected_row(spec: dict) -> dict:
    """Build a single detected-event CSV row from a specification dict."""
    # ... returns dict with exact schema
```

For the synthetic galaxy-catalog builder, the **exact column schema** is dictated by `setup_galaxy_catalog_balltree` at `handler.py:281-293`, which reads two columns from `self.reduced_galaxy_catalog`:

```python
# From master_thesis_code/galaxy_catalogue/handler.py lines 281-293
def setup_galaxy_catalog_balltree(self) -> None:
    # expects the reduced galaxy catalog to be setup already
    ra = self.reduced_galaxy_catalog[InternalCatalogColumns.PHI_S].values
    dec = self.reduced_galaxy_catalog[InternalCatalogColumns.THETA_S].values

    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    data = np.vstack((x, y, z)).T

    self.catalog_ball_tree = BallTree(data, metric="euclidean")
    self.reduced_galaxy_catalog = self.reduced_galaxy_catalog.reset_index()
```

`InternalCatalogColumns` string values from `handler.py:122-128`:

```python
class InternalCatalogColumns:
    PHI_S = "RIGHT_ASCENSION"
    THETA_S = "DECLINATION"
    REDSHIFT = "REDSHIFT"
    REDSHIFT_ERROR = "REDSHIFT_MEASUREMENT_ERROR"
    BH_MASS = "STELLAR_MASS"
    BH_MASS_ERROR = "STELLAR_MASS_ABSOULTE_ERROR"
```

**Minimum column set required by the BallTree path:** `"RIGHT_ASCENSION"`, `"DECLINATION"`. For tests that exercise `get_possible_hosts_from_ball_tree` (lines 295-354), add `"REDSHIFT"`, `"REDSHIFT_MEASUREMENT_ERROR"`, `"STELLAR_MASS"`, `"STELLAR_MASS_ABSOULTE_ERROR"`. For fixtures that mirror the ingestion stage, build the DataFrame in **degrees** so the test calls `_map_angles_to_spherical_coordinates` as a stage under test (per `handler.py:486-492`).

**Caveat — the synthetic builder must choose a stage:** `_map_angles_to_spherical_coordinates` (handler.py:486-492) is the buggy function. The fixture builder should expose a knob (e.g. `already_rotated: bool = False`) so one test can inject a pre-rotated (radians-in-ecliptic) catalog to isolate the BallTree bug, and another test can inject degrees-in-equatorial to exercise the missing rotation. Keep this explicit — do NOT silently pick one.

**Seed-pinned RNG + `default_rng` pattern** (`generate_fixtures.py` lines 115-139):

```python
def _make_undetected_row(rng: np.random.Generator, index: int) -> dict:
    """Build a single undetected-event CSV row with SNR < 20."""
    z = rng.uniform(0.01, 1.0)
    # ...
    return {
        # ...
        "qS": np.arccos(rng.uniform(-1.0, 1.0)),
        "phiS": rng.uniform(0.0, 2 * np.pi),
        # ...
    }


# at call-site (line 165):
rng = np.random.default_rng(seed=42)
```

For `synthetic_catalog_builder(n, sky_band)`, accept `rng: np.random.Generator` or `seed: int = 42` and draw positions in the requested band.

**astropy wrapper — NEW USAGE.** No existing file uses `astropy.coordinates.SkyCoord` or `BarycentricTrueEcliptic`. Only `astropy.constants` and `astropy.units` appear in the repo (per CONTEXT §code_context). This is new first-use.

Canonical example from CONTEXT.md `<specifics>` (already user-approved):

```python
# From .planning/phases/35-coordinate-bug-characterization/35-CONTEXT.md §specifics
from astropy.coordinates import SkyCoord, BarycentricTrueEcliptic
import astropy.units as u
import numpy as np

def equatorial_to_ecliptic_astropy(ra_deg, dec_deg):
    coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
    ecl = coord.transform_to(BarycentricTrueEcliptic(equinox='J2000'))
    lambda_rad = ecl.lon.to(u.rad).value
    beta_rad = ecl.lat.to(u.rad).value
    theta_polar_rad = np.pi/2 - beta_rad
    return lambda_rad, beta_rad, theta_polar_rad
```

Match this verbatim. D-12-ish (Claude's discretion) lets you return a namedtuple, dict, or tuple — a `NamedTuple` keeps unpacking clean in tests. Add complete type annotations per project typing conventions.

**BallTree builder helper (`build_balltree`)** — delegate to the real `setup_galaxy_catalog_balltree` rather than reimplementing the (buggy) Cartesian embedding. Cleanest shape:

```python
def build_balltree(df: pd.DataFrame) -> BallTree:
    """Build a BallTree from an in-memory catalog DataFrame.

    Delegates to master_thesis_code.galaxy_catalogue.handler.setup_galaxy_catalog_balltree
    by constructing a minimally-viable GalaxyCatalogueHandler-like shim.
    """
```

Since `setup_galaxy_catalog_balltree` is a *method* on `GalaxyCatalogueHandler` (not a free function), the fixture must either (a) call the method via an instance that has `reduced_galaxy_catalog` set (skip `__init__`), or (b) reproduce the two-line Cartesian embedding. Option (a) — `object.__new__(GalaxyCatalogueHandler)` + direct attribute assignment — has project precedent:

```python
# From master_thesis_code_test/bayesian_inference/test_catalog_only_diagnostic.py lines 37-73
@pytest.fixture()
def mock_bayesian_stats(self) -> "BayesianStatistics":
    """Create a minimally mocked BayesianStatistics for testing p_Di behavior."""
    from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics

    # Create instance without __init__ (avoids CSV loading)
    instance = object.__new__(BayesianStatistics)
    instance.h = 0.73
    instance.catalog_only = True
    # ... assign attributes directly
    return instance
```

This is the repo-idiomatic way to side-step an `__init__` that does heavy I/O. Use the same pattern.

---

### `.planning/audit_coordinate_bug.md` + `.planning/audit_coordinate_bug.json` (audit artifacts)

**Analogs:**
- `.planning/debug/comparison_current.md` (project convention for `.planning/` markdown sidecars; table + sections)
- `.planning/debug/baseline.json` (project convention for structured sidecars with `git_commit` + `created_at`)
- `master_thesis_code/bayesian_inference/evaluation_report.py:578-609` (live code that writes the JSON+MD pair)

**Markdown shape** — exact structure from `.planning/debug/comparison_current.md` lines 1-45:

```markdown
# H0 Posterior Comparison: baseline vs current

Generated: 2026-04-09T19:01:30.248192+00:00Z

## Summary Table

| Metric | Baseline | Current | Delta |
|--------|----------|---------|-------|
| MAP h | 0.7300 | 0.7100 | -0.0200 |
| CI lower | 0.7100 | 0.6982 | -0.0118 |
...

## Verdict

Bias IMPROVED by 2.7 percentage points.
```

Adapt for Phase 35 — the markdown must contain (per D-11):

1. Header with timestamp + source CSV path.
2. Band-count table: columns `Band | Count | Fraction (observed) | Fraction (isotropic) | Deviation`, rows ±5°, ±10°, ±15°.
3. Embedded histogram via relative path: `![|qS − π/2| distribution](audit_coordinate_bug_histogram.png)`.
4. JSON link: literal-block line `` `.planning/audit_coordinate_bug.json` ``.
5. "Verdict" or "Summary" closing paragraph (mirror `.planning/debug/comparison_current.md:43-45`).

**JSON shape** — locked fields from D-11: `event_count`, `band_counts`, `band_fractions`, `expected_fraction_5deg`, `csv_source_path`, `git_commit`, `timestamp`.

Follow the `.planning/debug/baseline.json` field-naming conventions (snake_case, ISO 8601 with `+00:00Z` suffix, full 40-char git SHA):

```json
{
  "map_h": 0.73,
  "ci_lower": 0.7101197732498211,
  "ci_upper": 0.7498802267501788,
  "ci_width": 0.03976045350035773,
  "bias_percent": 0.0,
  "n_events": 5,
  "created_at": "2026-04-09T19:01:30.241281+00:00Z",
  "git_commit": "1af4dc78f47b8986fa6190cb4f626a7a88c2e30c"
}
```

**Code precedent for dict → MD + JSON dual-write** — `evaluation_report.py:570-609` is the cleanest live analog and should be the mental template for the generator:

```python
# From master_thesis_code/bayesian_inference/evaluation_report.py lines 570-609
# Write JSON summary
output_dir.mkdir(parents=True, exist_ok=True)
json_path = output_dir / f"diagnostic_summary_{label}.json"
json_path.write_text(json.dumps(summary, indent=2))

# Write markdown summary
md_path = output_dir / f"diagnostic_summary_{label}.md"
md_lines = [
    f"# Diagnostic Summary: {label}",
    "",
    f"Source: `{diagnostic_csv_path}`",
    f"Events: {summary['n_events']}, H-values: {summary['n_h_values']}, "
    f"Rows: {summary['n_rows']}",
    "",
    "## Catalog Completeness (f_i)",
    "",
    f"- Mean f_i: {summary['mean_f_i']}",
    # ...
]
md_path.write_text("\n".join(md_lines))
```

Same literal shape: build a dict → dump to JSON with `indent=2` → build a list of markdown lines → `"\n".join(...) + "\n"` → write.

**git_commit + timestamp helpers** — use the project-canonical implementations verbatim (do not re-implement):

```python
# From master_thesis_code/bayesian_inference/evaluation_report.py lines 23-32
def _get_git_commit_safe() -> str:
    """Return current git commit hash, or 'unknown' if not available."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
```

```python
# From master_thesis_code/bayesian_inference/evaluation_report.py lines 65-67
created_at: str = field(
    default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat() + "Z"
)
```

**Recommendation:** `from master_thesis_code.bayesian_inference.evaluation_report import _get_git_commit_safe` — do NOT copy-paste.

---

### `.planning/audit_coordinate_bug_histogram.png` (figure artifact)

**Analog:** `master_thesis_code/plotting/bayesian_plots.py:plot_detection_redshift_distribution` (lines 329-344) — the closest existing 1-D histogram factory.

**Factory function shape** (verbatim, lines 329-344):

```python
# From master_thesis_code/plotting/bayesian_plots.py lines 329-344
def plot_detection_redshift_distribution(
    redshifts: npt.NDArray[np.float64],
    *,
    bins: int = 30,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of detection redshifts."""
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    ax.hist(redshifts, bins=bins, edgecolor=EDGE, alpha=0.7)
    ax.set_xlabel(LABELS["z"])
    ax.set_ylabel("Count")
    return fig, ax
```

Rules derived from this:
1. Takes `npt.NDArray[np.float64]` data in, returns `(fig, ax)` out.
2. Accepts optional `ax: Axes | None = None` for composition.
3. Calls `get_figure(preset="single")` for auto-sized figures with the project style.
4. Never calls `plt.show()` or `plt.savefig()` — callers do that via `save_figure()`.
5. Imports `EDGE` from `_colors`; imports `get_figure` and `_fig_from_ax` from `_helpers`.

**Where to put this factory** — two options, both acceptable:
- (a) Inline inside the audit generator module (since it's a one-off figure — mirrors `plot_snr_distribution` pattern but private to the audit).
- (b) Add as a public factory in `master_thesis_code/plotting/catalog_plots.py` or a new `coordinate_plots.py` — only if you anticipate reuse.

**User instruction from CONTEXT.md line 119**: "call `apply_style()` in the histogram-generation code per project convention" AND line 121: "do NOT override the session `apply_style()` fixture; trust it." So: inside the **standalone CLI module**, call `apply_style()` once at entry (since the module runs outside pytest). Inside any **new test** that happens to draw plots, do NOT call `apply_style()` — the session fixture at `conftest.py:45-53` handles it.

**save_figure() invocation pattern** (`_helpers.py:169-197`):

```python
def save_figure(
    fig: Figure,
    path: str,
    *,
    formats: Sequence[str] = ("pdf",),
    dpi: int = 300,
    close: bool = True,
) -> None:
    """Save *fig* to *path*, creating parent directories as needed.
    ...
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for fmt in formats:
        fig.savefig(f"{path}.{fmt}", dpi=dpi)
    if close:
        plt.close(fig)
```

CAVEAT — default `formats` is `("pdf",)`. The audit artifact is specified as `.png`. Pass `formats=("png",)` explicitly:

```python
save_figure(fig, ".planning/audit_coordinate_bug_histogram", formats=("png",))
```

`save_figure` appends the extension from `formats`, so pass the path **without** extension.

**Bin count** — Claude's discretion per D-11. Pattern from `bayesian_plots.py:332`: `bins: int = 30` is the project default for histogram factories. For 42 events, 10–15 bins is saner (CONTEXT.md specifics). Use `bins=12` or pass via CLI flag.

---

### Baseline audit generator (standalone module — Claude's discretion path)

**Analogs (two path choices, both with precedent):**

**Path A — Script in `scripts/`:** `scripts/merge_cramer_rao_bounds.py` (lines 106-118) is the canonical `scripts/` template:

```python
# From scripts/merge_cramer_rao_bounds.py lines 106-118
def main(argv: list[str] | None = None) -> None:
    """Entry point for the merge script.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).
    """
    args = parse_args(argv)
    workdir = Path(args.workdir)
    merge_cramer_rao_bounds(workdir, args.delete_sources)


if __name__ == "__main__":
    main()
```

And the `parse_args` shape (lines 19-42):

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the merge script."""
    parser = argparse.ArgumentParser(
        description="Merge per-index Cramer-Rao bounds CSVs.",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=".",
        help="Working directory; paths resolved relative to this",
    )
    parser.add_argument(
        "--delete-sources",
        action="store_true",
        help="Delete per-index source CSVs after successful merge",
    )
    return parser.parse_args(argv)
```

**Path B — Module in `master_thesis_code/` invoked via `python -m`:** `master_thesis_code/__main__.py` (4 lines total) is the pattern for `python -m master_thesis_code`:

```python
# From master_thesis_code/__main__.py (entire file)
from master_thesis_code import main

if __name__ == "__main__":
    main.main()
```

A Phase-35 variant would place a similar stub at `master_thesis_code_test/audit_coordinate_bug/__main__.py` OR put everything in `master_thesis_code_test/audit_coordinate_bug.py` with `if __name__ == "__main__": main()` at the bottom and invoke via `python -m master_thesis_code_test.audit_coordinate_bug`.

**Recommendation — based on CONTEXT.md §specifics line 152:**
> Baseline audit invocation pattern: keep the generator standalone (a module or script), NOT a pytest fixture — artifact must be reproducible outside the test suite for Phase 40 re-runs.

Either **(A) `scripts/audit_coordinate_bug.py`** or **(B) `master_thesis_code/galaxy_catalogue/audit_coordinate.py` + `python -m ...`** works. `scripts/` has 11 existing precedents (merge, prepare, compare, estimate, quick_*, remove_*); it is the shortest and most idiomatic path. Prefer **(A)** unless the planner has reason to route via the main package.

**CSV → summary-dict → JSON + MD + PNG dual-write shape** — direct analog `evaluation_report.py:468-612` (`generate_diagnostic_summary`):

```python
# From master_thesis_code/bayesian_inference/evaluation_report.py lines 468-574
def generate_diagnostic_summary(
    diagnostic_csv_path: Path,
    output_dir: Path,
    label: str = "diagnostic",
) -> dict[str, object]:
    """Analyze per-event diagnostic CSV and generate explanatory summary.
    ...
    Args:
        diagnostic_csv_path: Path to event_likelihoods.csv from evaluate().
        output_dir: Directory to write the summary report.
    """
    df = pd.read_csv(diagnostic_csv_path)

    # --- Compute statistics ---
    # ...
    summary: dict[str, object] = {
        "n_rows": len(df),
        "n_events": int(df["event_idx"].nunique()),
        # ...
    }

    # Write JSON summary
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"diagnostic_summary_{label}.json"
    json_path.write_text(json.dumps(summary, indent=2))
```

This is the closest in-repo template for the audit generator's main function. Mirror it: `audit_coordinate_bug(csv_path: Path, output_dir: Path) -> dict[str, object]`.

**Input CSV** per CONTEXT.md line 97: `simulations/cramer_rao_bounds.csv`, column of interest `qS`. Pattern for reading a Cramer-Rao bounds CSV is already used at `scripts/merge_cramer_rao_bounds.py:81`: `pd.read_csv(output_path)`.

---

## Shared Patterns

### Module docstring

**Source:** `master_thesis_code/plotting/bayesian_plots.py:1-8`, `scripts/merge_cramer_rao_bounds.py:1-5`, `master_thesis_code_test/fixtures/evaluation/generate_fixtures.py:1-10`.
**Apply to:** All new files (test file, fixture module, audit generator).

Every source file opens with a triple-quoted module docstring summarizing purpose + how to run. Test file should also mention which bugs the RED tests pin.

### Seed-pinned RNG

**Source:** `master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py:25`, `master_thesis_code_test/fixtures/evaluation/generate_fixtures.py:165`, 13 other call-sites.
**Apply to:** Fixture `synthetic_catalog_builder(n, sky_band, seed=42)` + the N=100 random-gate test.

```python
rng = np.random.default_rng(seed)
# pass rng through to any helper that needs random draws
```

Do NOT use bare `np.random.*` calls (TODO.md REPRO-1 identifies 21 legacy sites; do not add more).

### Type annotations

**Source:** `pyproject.toml:disallow_untyped_defs = true`; enforced by pre-commit mypy.
**Apply to:** Every public and private function in every new file.

Follow CLAUDE.md Typing Conventions verbatim — `list[...]`, `dict[...]`, `X | None`, `npt.NDArray[np.float64]`, no `from __future__ import annotations`. Notable examples in the analogs: `_helpers.py:140-167` (full annotations including `Literal`), `bayesian_plots.py:329-344` (Figure/Axes return typing).

### Pytest conftest session fixtures

**Source:** `master_thesis_code_test/conftest.py:45-53`.
**Apply to:** Any new test file in `master_thesis_code_test/`.

Session-level `apply_style()` fixture runs automatically. Do NOT call `apply_style()` inside new test functions. Do NOT override. If a test generates a plot (none of Phase 35's RED tests should), the session fixture handles backend + style.

### JSON + Markdown dual-write for audit artifacts

**Source:** `master_thesis_code/bayesian_inference/evaluation_report.py:570-609` (live code), `.planning/debug/baseline.json` + `.planning/debug/comparison_current.md` (historical artifacts).
**Apply to:** Baseline audit generator.

Template:
1. Build `summary: dict[str, object]` with flat, serializable fields.
2. `output_dir.mkdir(parents=True, exist_ok=True)`.
3. `json_path.write_text(json.dumps(summary, indent=2))`.
4. Build `md_lines: list[str]`, `md_path.write_text("\n".join(md_lines) + "\n")`.
5. Include `git_commit` + ISO-8601 `created_at` / `timestamp` fields.

### NEVER-DO list (project conventions)

- NEVER import `matplotlib.pyplot` at module level of a factory function (`sky_plots.py:1-21` shows the OO pattern; only `_helpers.py` imports pyplot and it is the shared utility).
- NEVER use `from __future__ import annotations`.
- NEVER call `apply_style()` inside a test — the session fixture handles it.
- NEVER import `cupy` unconditionally at module top level (irrelevant for Phase 35 — no GPU code — but noted for completeness).

---

## Shared Patterns — Analog File Index

| File | Purpose | Key line ranges |
|---|---|---|
| `master_thesis_code_test/conftest.py` | Session `apply_style()` fixture, `xp` parametrize | 45-53, 56-72 |
| `master_thesis_code_test/physical_relations_test.py` | Flat `def test_*` style, parametrize | 18-21, 87-92 |
| `master_thesis_code_test/test_glade_completeness.py` | Class-based + contract-docstring style | 31-55, 79-100 |
| `master_thesis_code_test/fixtures/evaluation/generate_fixtures.py` | Existing sibling fixture module | 1-10, 19-47, 60-112, 142-178 |
| `master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py` | Seed-pinned RNG synthetic CSV builder | 18-50, 53-77, 80-92 |
| `master_thesis_code_test/bayesian_inference/test_catalog_only_diagnostic.py` | `object.__new__(...)` pattern for bypassing heavy `__init__` | 37-73 |
| `master_thesis_code/galaxy_catalogue/handler.py` | DataFrame column schema + buggy BallTree embedding | 122-128, 281-293, 295-315, 486-492 |
| `master_thesis_code/plotting/_style.py` | `apply_style()` definition | 13-47 |
| `master_thesis_code/plotting/_helpers.py` | `get_figure()`, `save_figure()` | 140-167, 169-197 |
| `master_thesis_code/plotting/bayesian_plots.py` | Histogram factory pattern | 329-344 |
| `master_thesis_code/bayesian_inference/evaluation_report.py` | `_get_git_commit_safe()`, `created_at`, `to_json`, `generate_diagnostic_summary` dual-write | 23-32, 65-91, 468-612 |
| `scripts/merge_cramer_rao_bounds.py` | `scripts/` CLI entry-point template | 19-42, 106-118 |
| `scripts/prepare_detections.py` | More elaborate CLI with JSON metadata sidecar + `--seed` + `--force` | 33-62, 65-142 |
| `master_thesis_code/__main__.py` | `python -m master_thesis_code` stub | 1-4 |
| `.planning/debug/baseline.json` | JSON sidecar field-naming convention | entire file |
| `.planning/debug/comparison_current.md` | Markdown summary-table convention | 1-45 |

---

## No Analog Found

| File / Concern | Reason | Planner action |
|---|---|---|
| `@pytest.mark.xfail(strict=True)` in the active test suite | CHANGELOG notes historical xfails were removed once bugs were fixed. No live examples to copy. | Use the exact marker shape locked in D-01 verbatim; no analog needed. This is the first current xfail in the repo. |
| `astropy.coordinates.SkyCoord` / `BarycentricTrueEcliptic` usage | Only `astropy.constants` and `astropy.units` appear in the existing codebase (per CONTEXT §code_context, confirmed by grep). | Use the user-approved example from CONTEXT.md §specifics lines 145-151 verbatim. |
| `.planning/` `.png` artifact | `.planning/debug/` holds JSON + MD; no PNG precedent there. | No structural analog. Use `save_figure(..., formats=("png",))` with the `.planning/` relative path; let the markdown embed the PNG inline. |
| Dedicated directory `master_thesis_code_test/fixtures/coordinate/` (vs single file) | Existing `fixtures/` has one subdirectory (`evaluation/` with CSVs + generator) and an empty `__init__.py`. Single-file `coordinate.py` is fine; subdirectory is fine if fixtures grow. | D-08 locks single file `fixtures/coordinate.py`. Follow the lock. |

---

## Metadata

**Analog search scope:** `master_thesis_code/`, `master_thesis_code_test/`, `scripts/`, `.planning/debug/`, `pyproject.toml`, `CLAUDE.md`.
**Files scanned for analogs:** ~35.
**Grep queries run:** `xfail`, `SkyCoord|BarycentricTrueEcliptic|astropy.coordinates`, `default_rng\(seed`, `git_commit|git rev-parse`, `\.hist\(`, `def apply_style`, `markers|strict_markers`.
**Pattern extraction date:** 2026-04-21.
