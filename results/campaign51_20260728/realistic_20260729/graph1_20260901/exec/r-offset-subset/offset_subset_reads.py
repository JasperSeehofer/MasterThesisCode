"""Phase C (the reader) — r-offset-subset registered read.

REGISTRATION_DRAFT.md §3 ("Phase C, the reader"), §4 (registered statistics), §5
(disposition bands). Joins the blind covariate table (phase A,
``offset_subset_table.py``) with the influence vectors (phase B,
``offset_subset_influence.py``) on ``event_idx``, and for every registered covariate
in the closed family C1-C11 (§2) computes:

    1. separation (AUC via Mann-Whitney U for continuous covariates, odds ratio via
       Fisher exact for binary covariates), Holm-corrected over the family;
    2. materiality (leave-out re-marginalisation of the enriched stratum under the
       frozen T0 convention, vs a 1000-draw random-same-size null), for the primary
       family (iiib 2D) only, per §4.3 (materiality is not re-run per replicate --
       only separation is a replicate consistency check);
    3. the 2-of-3 replicate-consistency rule (§4.3);
    4. the disposition (§5): SUBSET-IDENTIFIED / DIFFUSE-IN-COVARIATES /
       INTERMEDIATE / INSTRUMENT-NO-READ, plus the mandatory R14 class-label line
       naming which of (a) C2, (b) C3, (c) C3c separates.

This script NEVER computes a registered aggregate itself outside of a run the author
has authorised (real-mode CLI use is gated the same way whether invoked here or on
the cluster) -- the build record for this script (BUILD_RECORD_B3.md) exercises it
ONLY on a synthetic <=10-row table, never on the registered population.

Blindness (§3, §6 G-4): this script is the ONLY one of the three phase agents
permitted to open both the covariate table and the influence vectors -- and it must
refuse to do so unless the covariate table's sha256 (recomputed here) equals the one
supplied via ``--table-sha256`` (the value phase A committed to BUILD_RECORD.md
*before* phase B's first run).

Materiality data contract (documented here because the registered launch block
(REGISTRATION_DRAFT.md §8) does not pass phase C a separate path to the raw
``event_likelihoods.csv``): for the PRIMARY family only (iiib / combined_with_bh),
``influence_vectors.csv`` (phase B) must carry, per event_idx, the physics-floored
per-event log-likelihood at every H_GRID_41 node, as self-describing columns named
``logL_h<value>`` with ``<value>`` the grid h formatted to 6 decimals (e.g.
``logL_h0.730000``). This lets phase C reconstruct the full-sample and any
stratum-removed log-posterior (``logpost(h) = sum_e logL_e(h)``) and its T0 moments
(gradient-trapezoid weights, no re-floor needed) using ONLY the two registered input
files -- no third path, no re-reading the pinned CSVs. Absence of these columns is an
INSTRUMENT-DEFECT for materiality only; separation statistics do not need them.

CLI: exactly the r-offset-subset launch block (REGISTRATION_DRAFT.md §8), plus
optional ``--k-<family>`` sanity-check flags (defaulted to the registered banked k,
§2) that verify ``in_S`` column cardinality without ever re-deriving S.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr

FloatArray = npt.NDArray[np.float64]

# ---------------------------------------------------------------------------
# Registered constants (REGISTRATION_DRAFT.md §2, §5, §8)
# ---------------------------------------------------------------------------

FAMILIES: tuple[str, ...] = ("iiib_2d", "iiib_1d", "jr1_2d", "jr1_1d")
PRIMARY_FAMILY = "iiib_2d"
REPLICATE_FAMILIES: tuple[str, ...] = ("iiib_1d", "jr1_2d", "jr1_1d")
REGISTERED_K: dict[str, int] = {"iiib_2d": 82, "iiib_1d": 94, "jr1_2d": 72, "jr1_1d": 46}

#: covariate id -> (type, "family" i.e. counted toward Holm m, conditional-testable)
COVARIATE_TYPE: dict[str, Literal["binary", "continuous"]] = {
    "C1": "binary",
    "C2": "binary",
    "C3": "binary",
    "C3c": "continuous",
    "C4": "continuous",
    "C5": "continuous",
    "C6": "continuous",
    "C7": "continuous",
    "C8": "binary",
    "C10": "continuous",
    "C10b": "binary",
    "C11": "continuous",
}
#: registered Holm family (§2); C9 is an alias of C1 (no test), C11 is reported-only.
HOLM_FAMILY: tuple[str, ...] = ("C1", "C2", "C3", "C3c", "C4", "C5", "C6", "C7", "C8", "C10", "C10b")
CLASS_LABELS: dict[str, str] = {"C2": "(a) hosted_exact", "C3": "(b) hosted_rel", "C3c": "(c) log10_f_cat"}
REPORTED_ONLY: tuple[str, ...] = ("C11",)

C10B_MIN_N = 10
LOGL_COL_PREFIX = "logL_h"

# ---------------------------------------------------------------------------
# PIN CORRECTIONS (REGISTRATION_DRAFT.md, 2026-09-04 ~00:40 CEST) + DESIGN_GATE_
# formula_rev3.md finding 4.2: "column schema of record = the BUILT files'
# headers". The built phase-A/B outputs are split ONE PER VENUE (iiib,
# joint_r1) rather than the draft's single combined table/influence file, and
# use suffixed covariate ids and a generic influence_2D/influence_1D pair with
# NO `_in_S` flag. This block is the explicit, asserted mapping from the
# registered bare ids (C1..C11) and family names to those real, built columns
# -- confirmed via `head -1` on the four committed CSVs and BUILD_RECORD_B1.md/
# BUILD_RECORD_B2.md's own column definitions.
# ---------------------------------------------------------------------------

#: registered covariate id -> real column name in covariate_table_{iiib,joint_r1}.csv
#: (confirmed via `head -1` on both files, 2026-09; identical schema in both venues).
COVARIATE_COLUMN_MAP: dict[str, str] = {
    "C1": "C1_in_catalog",
    "C2": "C2_hosted_exact",
    "C3": "C3_hosted_rel",
    "C3c": "C3c_log10_f_cat",
    "C4": "C4_z_gw",
    "C5": "C5_log10_sky_area",
    "C6": "C6_mass_window_retention",
    "C7": "C7_log10_n_cand_1d",
    "C8": "C8_cone_outside",
    "C10": "C10_log10_M",
    "C10b": "C10b_low_M_timeout_bins12",
    "C11": "C11_log10_snr",
}
assert set(COVARIATE_COLUMN_MAP) == set(HOLM_FAMILY) | set(REPORTED_ONLY), (
    "COVARIATE_COLUMN_MAP must cover exactly the registered covariate family "
    "(HOLM_FAMILY + REPORTED_ONLY) -- a mismatch here would silently exempt a "
    "registered covariate from the schema pre-flight."
)

#: influence_{iiib,joint_r1}.csv's real, built columns (confirmed via `head -1`):
#: event_idx, influence_2D, influence_1D, rank -- NO `_in_S`/`_d_e` suffixed
#: columns exist. Per BUILD_RECORD_B2.md "Output files": influence_2D/
#: influence_1D ARE the directional statistic d_e (positive = removing the
#: event moves mean_h toward truth) despite the column name -- NOT the raw
#: `influence = mean_h(full) - mean_h(full-e)` the name suggests.
REQUIRED_INFLUENCE_COLUMNS: tuple[str, ...] = ("event_idx", "influence_2D", "influence_1D", "rank")

#: registered family -> the real influence-CSV column that carries its d_e.
#: iiib_2d/jr1_2d both read influence_2D and iiib_1d/jr1_1d both read
#: influence_1D because venue is selected by WHICH per-venue file is loaded
#: (§8 launch block invokes this script once per venue), never by column name.
FAMILY_D_E_SOURCE_COL: dict[str, str] = {
    "iiib_2d": "influence_2D",
    "iiib_1d": "influence_1D",
    "jr1_2d": "influence_2D",
    "jr1_1d": "influence_1D",
}

#: venue (as named in the built filenames) -> the two families native to it.
#: A single invocation loads exactly one venue's table + influence file
#: (PIN CORRECTIONS item 1), so only that venue's two families are computable
#: from the data actually on disk.
VENUE_FAMILIES: dict[str, tuple[str, str]] = {
    "iiib": ("iiib_2d", "iiib_1d"),
    "jr1": ("jr1_2d", "jr1_1d"),
}

#: g-censoring (§6): "any MAP at 0.60/0.86 => that Delta is a BOUND, rail fraction reported."
#: The draft mandates disclosure of the null-draw rail fraction but does not itself state a
#: numeric threshold at which that gate counts as "red" for the INSTRUMENT / NO-READ
#: disposition (table row 151: "any Section 6 gate red"). Below this fraction the null
#: distribution used for the outside-null materiality test still discriminates; at or above
#: it, a majority of null draws hit an h_grid boundary, so the 0.5/99.5 percentile CI is
#: itself degenerate and cannot certify MATERIAL/not-MATERIAL. FLAGGED IN BUILD_RECORD_B3.md
#: as an orchestrator-derived default (the draft is silent on the numeric cut).
CENSORING_NULL_RAIL_RED_FRACTION = 0.5


# ---------------------------------------------------------------------------
# Small typed containers
# ---------------------------------------------------------------------------


@dataclass
class SeparationResult:
    covariate: str
    kind: Literal["binary", "continuous"]
    n_s: int
    n_b: int
    n_nan: int
    effect_name: str
    effect: float
    p_raw: float
    p_holm: float | None
    holm_significant: bool | None
    band_pass: bool
    verdict: Literal["SEPARATES", "WEAK", "NULL", "NOT-TESTED", "REPORTED-ONLY"]


@dataclass
class MaterialityResult:
    covariate: str
    stratum_rule: str
    n_stratum: int
    delta_strat: float
    delta_s_oracle: float
    captured_fraction: float
    null_percentile: float
    null_ci99: tuple[float, float]
    material: bool
    map_rail_full: bool
    map_rail_stratum: bool
    null_rail_fraction: float
    censoring_gate_red: bool
    n_missing: int


class InstrumentDefectError(Exception):
    """Hard pre-flight failure (PIN CORRECTIONS item 1 / task item 2): a registered
    covariate or influence column the schema-of-record requires is absent. Raised
    BEFORE any covariate is touched or any disposition computed -- never a silent
    skip. Caught once, at the top of `main()`, and written into the output JSON's
    `disposition` (real mode) or printed with a non-zero exit (dry-run).
    """

    def __init__(self, message: str, detail: dict[str, Any]):
        super().__init__(message)
        self.message = message
        self.detail = detail


@dataclass
class CovariateReport:
    separation: dict[str, SeparationResult] = field(default_factory=dict)
    materiality: dict[str, MaterialityResult] = field(default_factory=dict)
    replicate_consistent: dict[str, bool] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Gate: sha256 / blindness
# ---------------------------------------------------------------------------


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_table_hash(table_path: Path, expected_sha256: str) -> str:
    """G-4: refuse to run unless the recomputed hash matches the committed one."""
    actual = sha256_of_file(table_path)
    if actual != expected_sha256:
        raise SystemExit(
            "G-4 BLINDNESS-HASH-MISMATCH: covariate table sha256 does not match "
            f"--table-sha256.\n  expected: {expected_sha256}\n  actual:   {actual}\n"
            "Refusing to run (INSTRUMENT / NO-READ)."
        )
    return actual


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def detect_venue(table_path: Path, influence_path: Path) -> str:
    """Which of the two BUILT per-venue files (iiib / joint_r1) this invocation loads.

    §8's launch block CLI takes a single ``--table``/``--influence`` pair; the real
    built outputs are one covariate table + one influence file PER VENUE (PIN
    CORRECTIONS item 1), so a single invocation covers exactly one venue's two
    families. Raises INSTRUMENT-DEFECT if the venue cannot be determined
    unambiguously from both filenames, or if they disagree -- never guesses.
    """

    def _venue_of(p: Path) -> str | None:
        name = p.name.lower()
        if "joint_r1" in name or "jr1" in name:
            return "jr1"
        if "iiib" in name:
            return "iiib"
        return None

    t_venue = _venue_of(table_path)
    i_venue = _venue_of(influence_path)
    if t_venue is None or i_venue is None or t_venue != i_venue:
        raise InstrumentDefectError(
            "cannot determine a single venue (iiib / joint_r1) from "
            f"--table={table_path.name} and --influence={influence_path.name}; this "
            "script processes exactly one venue's covariate table + influence file "
            "per invocation (PIN CORRECTIONS item 1).",
            {"table_path": str(table_path), "influence_path": str(influence_path)},
        )
    return t_venue


def check_covariate_schema(df: pd.DataFrame) -> None:
    """Hard pre-flight (task item 2 / PIN CORRECTIONS item 1): every registered
    covariate's real (suffixed) column must be present in the loaded table BEFORE
    any covariate is touched. A missing column is INSTRUMENT-DEFECT, never a
    silent `if cov not in table.columns: continue` skip (DESIGN_GATE_formula_
    rev3.md finding 4.2 -- that skip path let a total schema mismatch bank a
    false DIFFUSE-IN-COVARIATES undetected).
    """
    missing = [(bare, real) for bare, real in COVARIATE_COLUMN_MAP.items() if real not in df.columns]
    if missing:
        raise InstrumentDefectError(
            "covariate table missing required column(s) for registered covariate(s): "
            + ", ".join(f"{bare} -> {real}" for bare, real in missing),
            {"missing_covariate_columns": [{"covariate": b, "expected_column": r} for b, r in missing]},
        )


def check_influence_base_schema(df: pd.DataFrame) -> None:
    """Hard pre-flight: the influence CSV's real, built columns must all be present."""
    missing = [c for c in REQUIRED_INFLUENCE_COLUMNS if c not in df.columns]
    if missing:
        raise InstrumentDefectError(
            f"influence vectors missing required column(s): {missing}",
            {"missing_influence_columns": missing},
        )


def load_table(path: Path) -> pd.DataFrame:
    """Load the real, built covariate table and map it onto the registered bare
    C1..C11 ids (PIN CORRECTIONS item 1: schema of record = the built headers).
    """
    df = pd.read_csv(path)
    if "event_idx" not in df.columns:
        raise InstrumentDefectError("covariate table missing required 'event_idx' column", {"missing_column": "event_idx"})
    check_covariate_schema(df)
    df = df.rename(columns={real: bare for bare, real in COVARIATE_COLUMN_MAP.items()})
    return df.set_index("event_idx", drop=False)


def load_influence(path: Path, venue: str, family_k: dict[str, int]) -> tuple[pd.DataFrame, FloatArray, FloatArray]:
    """Return (influence dataframe indexed by event_idx, h_grid, logL matrix [n_events, n_h]).

    The real, built influence CSV carries `event_idx, influence_2D, influence_1D,
    rank` only -- no `_d_e`/`_in_S` columns (PIN CORRECTIONS item 1). This loader
    is the schema adapter: for `venue`'s two native families it (a) aliases the
    generic influence_2D/influence_1D column (already the directional d_e per
    BUILD_RECORD_B2.md) to the `{family}_d_e` name the rest of this module
    expects, and (b) derives `{family}_in_S` from the top-k rank over that
    column -- S is the BANKED k (§2), never re-derived; `family_k` supplies it
    (registered default or the `--k-<family>` CLI override) so the CARDINALITY
    is by construction k, and `verify_k` remains a live byte-id sanity check on
    ties/rank stability rather than a vacuous read of a pre-baked flag.

    The logL matrix/h_grid are extracted from the self-describing ``logL_h<value>``
    columns (primary family only, per the module docstring's data contract) and are
    empty arrays if those columns are absent (materiality then reports NOT-TESTED)
    -- true today for the real built files, which carry no logL_h* columns at all.
    """
    df = pd.read_csv(path)
    if "event_idx" not in df.columns:
        raise InstrumentDefectError("influence vectors missing required 'event_idx' column", {"missing_column": "event_idx"})
    check_influence_base_schema(df)
    df = df.set_index("event_idx", drop=False)

    for family in VENUE_FAMILIES[venue]:
        src_col = FAMILY_D_E_SOURCE_COL[family]
        df[family_d_e_col(family)] = df[src_col]
        k = family_k[family]
        ranked = df[src_col].rank(method="first", ascending=False)
        df[family_in_s_col(family)] = (ranked <= k).to_numpy()

    logl_cols = sorted(c for c in df.columns if c.startswith(LOGL_COL_PREFIX))
    if not logl_cols:
        return df, np.array([], dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
    h_grid = np.array([float(c[len(LOGL_COL_PREFIX) :]) for c in logl_cols], dtype=np.float64)
    order = np.argsort(h_grid)
    h_grid = h_grid[order]
    logl_cols = [logl_cols[i] for i in order]
    logl_matrix = df[logl_cols].to_numpy(dtype=np.float64)
    return df, h_grid, logl_matrix


def family_in_s_col(family: str) -> str:
    return f"{family}_in_S"


def family_d_e_col(family: str) -> str:
    return f"{family}_d_e"


def verify_k(infl: pd.DataFrame, family: str, k: int) -> None:
    col = family_in_s_col(family)
    if col not in infl.columns:
        raise SystemExit(f"influence vectors missing required column '{col}'")
    observed = int(infl[col].astype(bool).sum())
    if observed != k:
        raise SystemExit(
            f"INSTRUMENT-DEFECT: family {family} in_S cardinality {observed} != "
            f"registered k={k}. S is a banked constant (§2), never re-derived."
        )


def check_join_completeness(table: pd.DataFrame, infl: pd.DataFrame) -> dict[str, Any]:
    """g-population (§6): "every table row joined (0 unmatched)".

    `separation_for_covariate`'s `pandas.Index.intersection()` calls silently drop any
    `event_idx` present on one side only, with no error and no count
    (DESIGN_GATE_formula_rev2.md §C) -- this check runs the join completeness test
    explicitly, in both directions, and its result is disclosed in the output JSON
    (`gates.g_population`) and wired into INSTRUMENT / NO-READ in `build_report()` when
    non-empty, rather than silently shrinking n_s/n_b for affected covariates.
    """
    table_idx = set(table.index.tolist())
    infl_idx = set(infl.index.tolist())
    unmatched_table = sorted(table_idx - infl_idx)
    unmatched_infl = sorted(infl_idx - table_idx)
    return {
        "n_table_rows": len(table_idx),
        "n_influence_rows": len(infl_idx),
        "n_unmatched_table_only": len(unmatched_table),
        "n_unmatched_influence_only": len(unmatched_infl),
        "unmatched_table_only_event_idx": unmatched_table,
        "unmatched_influence_only_event_idx": unmatched_infl,
        "join_complete": len(unmatched_table) == 0 and len(unmatched_infl) == 0,
    }


# ---------------------------------------------------------------------------
# Separation statistics (§4.1)
# ---------------------------------------------------------------------------


def _continuous_auc(x_s: FloatArray, x_b: FloatArray) -> tuple[float, float]:
    """AUC = U / (n_S * n_B) via Mann-Whitney U; two-sided p."""
    if x_s.size == 0 or x_b.size == 0:
        return float("nan"), float("nan")
    stat = mannwhitneyu(x_s, x_b, alternative="two-sided", method="auto")
    auc = float(stat.statistic) / (x_s.size * x_b.size)
    return auc, float(stat.pvalue)


def _binary_or(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Haldane-corrected odds ratio odds(TRUE|S)/odds(TRUE|B); two-sided Fisher p.

    a = TRUE in S, b = FALSE in S, c = TRUE in B, d = FALSE in B.
    """
    odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    or_haldane = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    return float(or_haldane), float(p_value)


def separation_for_covariate(
    covariate: str,
    table: pd.DataFrame,
    s_index: pd.Index,
    b_index: pd.Index,
    auc_band: float,
    or_band: float,
    restrict_index: pd.Index | None = None,
) -> SeparationResult:
    kind = COVARIATE_TYPE[covariate]
    col = table[covariate]
    if restrict_index is not None:
        s_index = s_index.intersection(restrict_index)
        b_index = b_index.intersection(restrict_index)

    s_vals = col.loc[col.index.intersection(s_index)]
    b_vals = col.loc[col.index.intersection(b_index)]
    n_nan = int(s_vals.isna().sum() + b_vals.isna().sum())
    s_vals = s_vals.dropna()
    b_vals = b_vals.dropna()

    if kind == "continuous":
        x_s = s_vals.to_numpy(dtype=np.float64)
        x_b = b_vals.to_numpy(dtype=np.float64)
        auc, p_raw = _continuous_auc(x_s, x_b)
        band_pass = bool(abs(auc - 0.5) >= auc_band) if not np.isnan(auc) else False
        return SeparationResult(
            covariate=covariate,
            kind=kind,
            n_s=x_s.size,
            n_b=x_b.size,
            n_nan=n_nan,
            effect_name="AUC",
            effect=auc,
            p_raw=p_raw,
            p_holm=None,
            holm_significant=None,
            band_pass=band_pass,
            verdict="NOT-TESTED" if (x_s.size == 0 or x_b.size == 0) else "NULL",
        )

    s_bool = s_vals.astype(bool)
    b_bool = b_vals.astype(bool)
    a = int(s_bool.sum())
    b = int((~s_bool).sum())
    c = int(b_bool.sum())
    d = int((~b_bool).sum())
    if min(a + b, c + d) == 0:
        return SeparationResult(
            covariate=covariate,
            kind=kind,
            n_s=a + b,
            n_b=c + d,
            n_nan=n_nan,
            effect_name="OR",
            effect=float("nan"),
            p_raw=float("nan"),
            p_holm=None,
            holm_significant=None,
            band_pass=False,
            verdict="NOT-TESTED",
        )
    or_val, p_raw = _binary_or(a, b, c, d)
    band_pass = bool(or_val < (1.0 / or_band) or or_val > or_band)
    return SeparationResult(
        covariate=covariate,
        kind=kind,
        n_s=a + b,
        n_b=c + d,
        n_nan=n_nan,
        effect_name="OR",
        effect=or_val,
        p_raw=p_raw,
        p_holm=None,
        holm_significant=None,
        band_pass=band_pass,
        verdict="NULL",
    )


def holm_correct(results: dict[str, SeparationResult], alpha: float, auc_band: float, or_band: float) -> None:
    """Holm step-down over the tested (non NOT-TESTED) members of results, in place."""
    tested = [cov for cov, r in results.items() if r.verdict != "NOT-TESTED"]
    m = len(tested)
    if m == 0:
        return
    order = sorted(tested, key=lambda cov: results[cov].p_raw)
    running_max = 0.0
    for i, cov in enumerate(order):
        r = results[cov]
        adj = (m - i) * r.p_raw
        running_max = max(running_max, adj)
        p_holm = min(1.0, running_max)
        r.p_holm = p_holm
        r.holm_significant = bool(p_holm < alpha)
        if r.holm_significant and r.band_pass:
            r.verdict = "SEPARATES"
        elif r.holm_significant and not r.band_pass:
            # REGISTRATION_DRAFT.md SS4.1: "a covariate SEPARATES iff Holm-adjusted p < 0.05
            # AND effect outside the practical-null band ... a significant-but-small effect
            # is reported as WEAK". "significant" is defined one clause earlier as the SAME
            # Holm-adjusted test used for SEPARATES -- so WEAK keys on r.holm_significant,
            # never on the raw p (DESIGN_GATE_formula_rev2.md SSD: the raw-p branch let a
            # raw-significant/Holm-non-significant covariate through as WEAK when the
            # registered definition says NULL).
            r.verdict = "WEAK"
        else:
            r.verdict = "NULL"


# ---------------------------------------------------------------------------
# T0 convention moments (materiality, §4.2) -- replicated per
# results/prod2d_closure_20260818/tier0_bootstrap_jackknife.py:_moments,
# gradient-trapezoid weights, no re-floor (phase B floors on load).
# ---------------------------------------------------------------------------


def t0_moments(logpost: FloatArray, h_grid: FloatArray, weights: FloatArray) -> tuple[float, float, bool]:
    lp = logpost - logpost.max()
    post = np.exp(lp)
    norm = float((post * weights).sum())
    post_n = post / norm
    mean_h = float((post_n * h_grid * weights).sum())
    map_h = float(h_grid[int(np.argmax(logpost))])
    rail = bool(map_h == h_grid[0] or map_h == h_grid[-1])
    return mean_h, map_h, rail


def materiality_for_covariate(
    covariate: str,
    sep: SeparationResult,
    table: pd.DataFrame,
    event_order: FloatArray,
    logl_matrix: FloatArray,
    h_grid: FloatArray,
    s_index: pd.Index,
    decile: float,
    t_mat: float,
    null_draws: int,
    null_seed: int,
) -> MaterialityResult | None:
    """§4.2. Returns None if this covariate has no logL data (primary-family-only gate)."""
    if logl_matrix.size == 0 or h_grid.size == 0:
        return None

    weights = np.gradient(h_grid)
    full_logpost = logl_matrix.sum(axis=0)
    mean_h_full, map_h_full, rail_full = t0_moments(full_logpost, h_grid, weights)

    n_total = event_order.size
    col = table[covariate].reindex(event_order)
    n_missing = 0

    if sep.kind == "binary":
        # Finding A (DESIGN_GATE_formula.md): the enriched level is the level over-
        # represented in S relative to the bulk -- i.e. the direction already certified
        # by the registered separation statistic (sep.effect, the OR), never a raw
        # majority recomputed from S alone. Symmetric with the continuous branch, which
        # uses sep.effect >= 0.5 (the AUC) to pick top-vs-bottom decile.
        enriched_level = bool(sep.effect >= 1.0)
        stratum_mask = (col.astype(bool) == enriched_level).to_numpy()
        rule_desc = f"binary level == {enriched_level} (enriched via OR={sep.effect:.6g} >= 1.0)"
    else:
        # Finding C: NaN is excluded from the decile-tail stratum (and, by construction
        # of `valid_n` below, from the bulk denominator used to size that tail) for this
        # covariate -- the draft (REGISTRATION_DRAFT.md) is silent on decile-stratum NaN
        # handling specifically (it only mandates n_NaN disclosure for the *separation*
        # statistic, §6 g-population). Orchestrator-derived default, flagged in
        # BUILD_RECORD_B3.md: "NaN excluded from both S and bulk for that covariate and
        # disclosed as n_missing." `rank(na_option="keep")` leaves NaN entries as NaN
        # (never coerced into the top or bottom rank, unlike na_option="bottom" which
        # Finding C showed sorts NaN to the *top* under an ascending rank).
        values = col.to_numpy(dtype=np.float64)
        nan_mask = np.isnan(values)
        n_missing = int(nan_mask.sum())
        valid_n = n_total - n_missing
        auc_above_half = sep.effect >= 0.5
        n_tail = max(1, round(valid_n * decile)) if valid_n > 0 else 0
        ranked = col.rank(method="first", na_option="keep")
        if auc_above_half:
            stratum_mask = ((ranked > (valid_n - n_tail)) & ~nan_mask).to_numpy()
        else:
            stratum_mask = ((ranked <= n_tail) & ~nan_mask).to_numpy()
        rule_desc = (
            f"{'top' if auc_above_half else 'bottom'} decile ({n_tail}/{valid_n}, "
            f"NaN excluded from stratum and bulk, n_missing={n_missing})"
        )

    n_stratum = int(stratum_mask.sum())
    stratum_logpost = full_logpost - logl_matrix[stratum_mask].sum(axis=0)
    mean_h_strat, map_h_strat, rail_strat = t0_moments(stratum_logpost, h_grid, weights)
    delta_strat = mean_h_strat - mean_h_full

    s_mask = np.isin(event_order, s_index.to_numpy())
    s_logpost = full_logpost - logl_matrix[s_mask].sum(axis=0)
    mean_h_s, _, _ = t0_moments(s_logpost, h_grid, weights)
    delta_s_oracle = mean_h_s - mean_h_full
    captured_fraction = float(delta_strat / delta_s_oracle) if delta_s_oracle != 0 else float("nan")

    # Finding D (g-censoring, §6): "MAP position for the full sample, every stratum
    # leave-out and every null draw; any MAP at 0.60/0.86 => that Delta is a BOUND, rail
    # fraction reported." The null-draw MAP was previously discarded (`_`); it is now
    # tracked so `null_rail_fraction` can be disclosed and, per
    # CENSORING_NULL_RAIL_RED_FRACTION above, wired into the INSTRUMENT / NO-READ
    # disposition when the null distribution itself is degenerate.
    rng = np.random.default_rng(null_seed)
    null_deltas = np.empty(null_draws, dtype=np.float64)
    idx_all = np.arange(n_total)
    null_rail_count = 0
    for i in range(null_draws):
        draw = rng.choice(idx_all, size=n_stratum, replace=False)
        draw_logpost = full_logpost - logl_matrix[draw].sum(axis=0)
        mean_h_draw, _map_h_draw, rail_draw = t0_moments(draw_logpost, h_grid, weights)
        null_deltas[i] = mean_h_draw - mean_h_full
        if rail_draw:
            null_rail_count += 1

    null_rail_fraction = float(null_rail_count / null_draws) if null_draws else float("nan")
    censoring_gate_red = bool(null_rail_fraction >= CENSORING_NULL_RAIL_RED_FRACTION)

    null_percentile = float((null_deltas < delta_strat).mean() * 100.0)
    ci_lo = float(np.percentile(null_deltas, 0.5))
    ci_hi = float(np.percentile(null_deltas, 99.5))
    outside_null = bool(delta_strat < ci_lo or delta_strat > ci_hi)
    material = bool(delta_strat >= t_mat and outside_null)

    return MaterialityResult(
        covariate=covariate,
        stratum_rule=rule_desc,
        n_stratum=n_stratum,
        delta_strat=delta_strat,
        delta_s_oracle=delta_s_oracle,
        captured_fraction=captured_fraction,
        null_percentile=null_percentile,
        null_ci99=(ci_lo, ci_hi),
        material=material,
        map_rail_full=rail_full,
        map_rail_stratum=rail_strat,
        null_rail_fraction=null_rail_fraction,
        censoring_gate_red=censoring_gate_red,
        n_missing=n_missing,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_family_separation(
    family: str,
    table: pd.DataFrame,
    infl: pd.DataFrame,
    alpha: float,
    auc_band: float,
    or_band: float,
) -> dict[str, SeparationResult]:
    in_s_col = family_in_s_col(family)
    s_mask = infl[in_s_col].astype(bool)
    s_index = infl.index[s_mask]
    b_index = infl.index[~s_mask]

    low_m_n = int(table["C10b"].astype(bool).sum()) if "C10b" in table.columns else 0
    c10b_testable = low_m_n >= C10B_MIN_N

    results: dict[str, SeparationResult] = {}
    for cov in HOLM_FAMILY:
        if cov not in table.columns:
            continue
        if cov == "C10b" and not c10b_testable:
            results[cov] = SeparationResult(
                covariate=cov,
                kind=COVARIATE_TYPE[cov],
                n_s=0,
                n_b=0,
                n_nan=0,
                effect_name="OR",
                effect=float("nan"),
                p_raw=float("nan"),
                p_holm=None,
                holm_significant=None,
                band_pass=False,
                verdict="NOT-TESTED",
            )
            continue
        restrict = table.index[table["C1"].astype(bool)] if cov == "C8" else None
        results[cov] = separation_for_covariate(cov, table, s_index, b_index, auc_band, or_band, restrict)

    holm_correct(results, alpha, auc_band, or_band)

    for cov in REPORTED_ONLY:
        if cov in table.columns:
            r = separation_for_covariate(cov, table, s_index, b_index, auc_band, or_band)
            r.verdict = "REPORTED-ONLY"
            results[cov] = r
    return results


def replicate_direction(r: SeparationResult) -> int:
    if r.kind == "continuous":
        return 1 if r.effect >= 0.5 else -1
    return 1 if r.effect >= 1.0 else -1


def disposition_for(
    primary: dict[str, SeparationResult],
    materiality: dict[str, MaterialityResult],
    replicate_sep: dict[str, dict[str, SeparationResult]],
    replicate_families: tuple[str, ...] = REPLICATE_FAMILIES,
) -> tuple[str, list[str]]:
    """§5 disposition for `primary` treated as the primary family.

    `replicate_families` defaults to the registered REPLICATE_FAMILIES (the three
    non-primary families relative to the true primary, iiib_2d) but is a parameter so
    the same logic can be re-run with iiib_1d substituted as primary (§5 INTERMEDIATE
    trigger: "primary 2D and 1D iiib families disagree in disposition", wired in
    build_report()) without hardcoding which family is "primary" inside this function.
    """
    separators = [cov for cov, r in primary.items() if cov in HOLM_FAMILY and r.verdict == "SEPARATES"]
    identified: list[str] = []
    intermediate: list[str] = []
    for cov in separators:
        mat = materiality.get(cov)
        if mat is None or not mat.material:
            intermediate.append(cov)
            continue
        n_consistent = 0
        for fam in replicate_families:
            fam_r = replicate_sep.get(fam, {}).get(cov)
            if fam_r is not None and fam_r.verdict == "SEPARATES":
                if replicate_direction(fam_r) == replicate_direction(primary[cov]):
                    n_consistent += 1
        if n_consistent >= 2:
            identified.append(cov)
        else:
            intermediate.append(cov)

    if identified:
        return "SUBSET-IDENTIFIED", identified
    if intermediate:
        return "INTERMEDIATE", intermediate
    if not separators:
        # Finding B (DESIGN_GATE_formula.md): REGISTRATION_DRAFT.md §5's INTERMEDIATE row
        # explicitly includes "C8 or C10b NOT-TESTED and no other covariate separates" --
        # that must NOT fall through to the (stronger, revision-consuming)
        # DIFFUSE-IN-COVARIATES claim writeback, which asserts every registered covariate
        # was actually tested against the band.
        not_tested_gate = [cov for cov in ("C8", "C10b") if cov in primary and primary[cov].verdict == "NOT-TESTED"]
        if not_tested_gate:
            return "INTERMEDIATE", []
        return "DIFFUSE-IN-COVARIATES", []
    return "INTERMEDIATE", []


def class_label_line(primary: dict[str, SeparationResult], materiality: dict[str, MaterialityResult]) -> dict[str, Any]:
    line: dict[str, Any] = {}
    separating: list[str] = []
    for cov, label in CLASS_LABELS.items():
        r = primary.get(cov)
        if r is None:
            line[cov] = {"label": label, "verdict": "NOT-TESTED"}
            continue
        entry: dict[str, Any] = {
            "label": label,
            "effect_name": r.effect_name,
            "effect": r.effect,
            "p_holm": r.p_holm,
            "verdict": r.verdict,
        }
        mat = materiality.get(cov)
        if r.verdict == "SEPARATES" and mat is not None:
            entry["delta_strat"] = mat.delta_strat
            entry["material"] = mat.material
        line[cov] = entry
        if r.verdict == "SEPARATES":
            separating.append(cov)

    if separating == ["C3c"]:
        reading = "catalogue-hosted is a continuous-weight notion; neither binary label indexes S"
    elif set(separating) == {"C3"}:
        reading = "the negligible-weight events are bulk; the materiality label is the right class"
    elif set(separating) == {"C2"}:
        reading = "S sits among the support-only hosted events -- a NEW lead on its own"
    elif not separating:
        reading = "class is not the axis"
    else:
        reading = f"multiple class labels separate: {separating}"
    return {"per_label": line, "separating": separating, "r14_reading": reading}


# ---------------------------------------------------------------------------
# Reported-only secondaries (§2, §4.1) -- no disposition role: none of these
# feed disposition_for() or class_label_line(); they are computed and
# disclosed in the output JSON only.
# ---------------------------------------------------------------------------


def spearman_secondaries(table: pd.DataFrame, infl: pd.DataFrame, family: str) -> dict[str, dict[str, Any]]:
    """§4.1 secondary: Spearman rho between d_e and each continuous covariate, all events."""
    d_e_col = family_d_e_col(family)
    out: dict[str, dict[str, Any]] = {}
    if d_e_col not in infl.columns:
        return out
    d_e = infl[d_e_col]
    for cov, kind in COVARIATE_TYPE.items():
        if kind != "continuous" or cov not in table.columns:
            continue
        merged = pd.concat([table[cov], d_e], axis=1, join="inner").dropna()
        if merged.shape[0] < 3:
            out[cov] = {"rho": None, "p": None, "n": int(merged.shape[0])}
            continue
        rho, p_value = spearmanr(merged[cov].to_numpy(), merged[d_e_col].to_numpy())
        out[cov] = {"rho": float(rho), "p": float(p_value), "n": int(merged.shape[0])}
    return out


def class_composition_counts(table: pd.DataFrame, s_index: pd.Index) -> dict[str, dict[str, int]]:
    """§4.1 secondary: raw C1/C2/C3 class composition of S, as counts."""
    out: dict[str, dict[str, int]] = {}
    for cov in ("C1", "C2", "C3"):
        if cov not in table.columns:
            continue
        vals = table[cov].reindex(s_index)
        n_nan = int(vals.isna().sum())
        bool_vals = vals.dropna().astype(bool)
        out[cov] = {
            "n_true": int(bool_vals.sum()),
            "n_false": int((~bool_vals).sum()),
            "n_nan": n_nan,
        }
    return out


def truth_disagreement_tables(table: pd.DataFrame) -> dict[str, dict[str, int]]:
    """§2 secondary: C1 (truth) vs C2/C3 disagreement, a 2x2 table per label."""
    out: dict[str, dict[str, int]] = {}
    if "C1" not in table.columns:
        return out
    c1 = table["C1"]
    for cov in ("C2", "C3"):
        if cov not in table.columns:
            continue
        both = pd.concat([c1.rename("C1"), table[cov].rename(cov)], axis=1, join="inner").dropna()
        b1 = both["C1"].astype(bool)
        b2 = both[cov].astype(bool)
        out[cov] = {
            "C1_true_and_cov_true": int((b1 & b2).sum()),
            "C1_true_and_cov_false": int((b1 & ~b2).sum()),
            "C1_false_and_cov_true": int((~b1 & b2).sum()),
            "C1_false_and_cov_false": int((~b1 & ~b2).sum()),
        }
    return out


def build_report(
    table: pd.DataFrame,
    infl: pd.DataFrame,
    h_grid: FloatArray,
    logl_matrix: FloatArray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    event_order = infl.index.to_numpy(dtype=np.float64)

    join_info = check_join_completeness(table, infl)

    per_family_sep: dict[str, dict[str, SeparationResult]] = {}
    for family in FAMILIES:
        per_family_sep[family] = run_family_separation(family, table, infl, args.alpha, args.auc_band, args.or_band)

    primary_sep = per_family_sep[PRIMARY_FAMILY]
    primary_s_index = infl.index[infl[family_in_s_col(PRIMARY_FAMILY)].astype(bool)]

    materiality: dict[str, MaterialityResult] = {}
    for cov, r in primary_sep.items():
        if cov not in HOLM_FAMILY or r.verdict != "SEPARATES":
            continue
        mat = materiality_for_covariate(
            cov,
            r,
            table,
            event_order,
            logl_matrix,
            h_grid,
            primary_s_index,
            args.decile,
            args.t_mat,
            args.null_draws,
            args.null_seed,
        )
        if mat is not None:
            materiality[cov] = mat

    replicate_sep = {fam: per_family_sep[fam] for fam in REPLICATE_FAMILIES}
    disposition, named_covariates = disposition_for(primary_sep, materiality, replicate_sep)
    primary_disposition_raw = disposition

    # SS5 INTERMEDIATE trigger: "primary 2D and 1D iiib families disagree in disposition".
    # Re-run disposition_for() with iiib_1d substituted as primary, against ITS OWN
    # separation results and the complementary replicate set {iiib_2d, jr1_2d, jr1_1d}.
    # iiib_1d has no materiality of its own: the data contract (module docstring) only
    # requires logL_h* columns for the PRIMARY family (iiib_2d), so
    # `iiib_1d_materiality` is always empty here -- this is a genuine, disclosed
    # limitation (DESIGN_GATE_formula_rev2.md SSB note), not a bug: iiib_1d's disposition
    # can therefore only ever read DIFFUSE-IN-COVARIATES or INTERMEDIATE, never
    # SUBSET-IDENTIFIED, but that is still a well-defined "whole disposition" to compare
    # against the primary's for the purpose of this trigger.
    iiib_1d_replicate_families = tuple(fam for fam in FAMILIES if fam != "iiib_1d")
    iiib_1d_replicate_sep = {fam: per_family_sep[fam] for fam in iiib_1d_replicate_families}
    iiib_1d_materiality: dict[str, MaterialityResult] = {}
    iiib_1d_disposition, iiib_1d_named = disposition_for(
        per_family_sep["iiib_1d"],
        iiib_1d_materiality,
        iiib_1d_replicate_sep,
        replicate_families=iiib_1d_replicate_families,
    )
    families_agree = disposition == iiib_1d_disposition
    if not families_agree:
        disposition = "INTERMEDIATE"

    logl_missing = logl_matrix.size == 0
    if any(cov in materiality or (per_family_sep[PRIMARY_FAMILY][cov].verdict == "SEPARATES") for cov in HOLM_FAMILY) and logl_missing:
        instrument_note = "materiality NOT computable: influence_vectors.csv carries no logL_h* columns for the primary family"
    else:
        instrument_note = None

    # g-population (SS6): "every table row joined (0 unmatched)" -- route a join
    # mismatch into INSTRUMENT / NO-READ instead of letting it silently shrink n_s/n_b
    # for whichever covariates lose rows (DESIGN_GATE_formula_rev2.md SSC).
    if not join_info["join_complete"]:
        join_note = (
            "g-population RED: table/influence join incomplete -- "
            f"{join_info['n_unmatched_table_only']} table row(s) with no influence match, "
            f"{join_info['n_unmatched_influence_only']} influence row(s) with no table "
            "match (0 unmatched required, SS6 g-population)"
        )
        instrument_note = join_note if instrument_note is None else f"{instrument_note}; {join_note}"

    # Finding D: wire the g-censoring null-rail gate (and, generically, any other
    # per-covariate materiality gate flagged red) into the INSTRUMENT / NO-READ
    # disposition -- a red must actually override SUBSET-IDENTIFIED / DIFFUSE /
    # INTERMEDIATE, not just sit unused in the materiality dict.
    censoring_red_covariates = [cov for cov, m in materiality.items() if m.censoring_gate_red]
    if censoring_red_covariates:
        censoring_note = (
            "g-censoring RED: null-draw MAP rail fraction >= "
            f"{CENSORING_NULL_RAIL_RED_FRACTION} for {censoring_red_covariates} "
            "-- the null distribution used for the outside-null materiality test is "
            "degenerate at the h_grid boundary and cannot certify MATERIAL/not-MATERIAL"
        )
        instrument_note = censoring_note if instrument_note is None else f"{instrument_note}; {censoring_note}"

    r14 = class_label_line(primary_sep, materiality)

    def sep_to_dict(r: SeparationResult) -> dict[str, Any]:
        return {
            "kind": r.kind,
            "n_s": r.n_s,
            "n_b": r.n_b,
            "n_nan": r.n_nan,
            "effect_name": r.effect_name,
            "effect": r.effect,
            "p_raw": r.p_raw,
            "p_holm": r.p_holm,
            "holm_significant": r.holm_significant,
            "band_pass": r.band_pass,
            "verdict": r.verdict,
        }

    def mat_to_dict(m: MaterialityResult) -> dict[str, Any]:
        return {
            "stratum_rule": m.stratum_rule,
            "n_stratum": m.n_stratum,
            "delta_strat": m.delta_strat,
            "delta_s_oracle": m.delta_s_oracle,
            "captured_fraction": m.captured_fraction,
            "null_percentile": m.null_percentile,
            "null_ci99": list(m.null_ci99),
            "material": m.material,
            "map_rail_full": m.map_rail_full,
            "map_rail_stratum": m.map_rail_stratum,
            "null_rail_fraction": m.null_rail_fraction,
            "censoring_gate_red": m.censoring_gate_red,
            "n_missing": m.n_missing,
        }

    report: dict[str, Any] = {
        "meta": {
            "table_path": str(args.table),
            "table_sha256": args.table_sha256,
            "influence_path": str(args.influence),
            "alpha": args.alpha,
            "auc_band": args.auc_band,
            "or_band": args.or_band,
            "t_mat": args.t_mat,
            "decile": args.decile,
            "null_draws": args.null_draws,
            "null_seed": args.null_seed,
            "n_events": len(table),
            "primary_family": PRIMARY_FAMILY,
            "logl_columns_present": not logl_missing,
        },
        "family_k": {fam: int(infl[family_in_s_col(fam)].astype(bool).sum()) for fam in FAMILIES},
        "separation": {fam: {cov: sep_to_dict(r) for cov, r in results.items()} for fam, results in per_family_sep.items()},
        "materiality": {cov: mat_to_dict(m) for cov, m in materiality.items()},
        "r14_class_label_line": r14,
        "gates": {
            "g_population": join_info,
        },
        "iiib_1d_disposition_check": {
            "iiib_1d_disposition": iiib_1d_disposition,
            "iiib_1d_named_covariates": iiib_1d_named,
            "primary_disposition_before_this_trigger": primary_disposition_raw,
            "agrees_with_primary": families_agree,
            "note": (
                "iiib_1d has no logL_h* columns under the current data contract "
                "(primary-family-only, per module docstring); its materiality is always "
                "empty, so its own disposition can only read DIFFUSE-IN-COVARIATES or "
                "INTERMEDIATE, never SUBSET-IDENTIFIED."
            ),
        },
        "secondaries": {
            "spearman_d_e_vs_continuous": spearman_secondaries(table, infl, PRIMARY_FAMILY),
            "class_composition_S": class_composition_counts(table, primary_s_index),
            "truth_disagreement_2x2": truth_disagreement_tables(table),
        },
        "disposition": {
            "value": disposition,
            "named_covariates": named_covariates,
            "instrument_note": instrument_note,
        },
    }
    if instrument_note is not None:
        report["disposition"]["value"] = "INSTRUMENT / NO-READ"
        report["disposition"]["named_covariates"] = []
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="r-offset-subset phase C: registered read")
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--table-sha256", type=str, required=True)
    p.add_argument("--influence", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--auc-band", type=float, default=0.20)
    p.add_argument("--or-band", type=float, default=3.0)
    p.add_argument("--t-mat", type=float, default=0.008)
    p.add_argument("--decile", type=float, default=0.10)
    p.add_argument("--null-draws", type=int, default=1000)
    p.add_argument("--null-seed", type=int, default=20260904)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    for fam, k_default in REGISTERED_K.items():
        p.add_argument(f"--k-{fam.replace('_', '-')}", type=int, default=k_default, dest=f"k_{fam}")
    return p.parse_args(argv)


def _write_instrument_defect(exc: InstrumentDefectError, args: argparse.Namespace) -> int:
    """Item 2: an INSTRUMENT-DEFECT is written to the JSON (real mode) and always
    exits non-zero -- never a silent skip. `--dry-run` has no `--out` contract to
    honour (it never writes a file, per its own docstring/CLI contract), so it
    prints and exits non-zero without touching `args.out`.
    """
    print(f"INSTRUMENT-DEFECT: {exc.message}")
    if not args.dry_run:
        report = {
            "meta": {"table_path": str(args.table), "influence_path": str(args.influence)},
            "disposition": {
                "value": "INSTRUMENT-DEFECT",
                "named_covariates": [],
                "instrument_note": exc.message,
                "detail": exc.detail,
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote {args.out}: disposition = INSTRUMENT-DEFECT")
    return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    family_k: dict[str, int] = {fam: getattr(args, f"k_{fam}") for fam in FAMILIES}

    try:
        check_table_hash(args.table, args.table_sha256)
        venue = detect_venue(args.table, args.influence)
        active_families = VENUE_FAMILIES[venue]
        table = load_table(args.table)
        infl, h_grid, logl_matrix = load_influence(args.influence, venue, family_k)

        # Item 2, belt-and-suspenders: with `check_covariate_schema` already gating
        # `load_table`, `table.columns` covers every registered covariate by
        # construction -- this recomputes the same check as a regression guard, so
        # a future edit that weakens the schema pre-flight still cannot reach
        # `build_report()`/`disposition_for()` with a partially-populated table
        # (DESIGN_GATE_formula_rev3.md finding 4.2's silent-DIFFUSE-IN-COVARIATES
        # failure mode).
        missing_covariates = [c for c in list(HOLM_FAMILY) + list(REPORTED_ONLY) if c not in table.columns]
        if missing_covariates:
            raise InstrumentDefectError(
                f"covariate table missing registered covariate(s) after schema mapping: {missing_covariates}",
                {"missing_covariates": missing_covariates},
            )
        for fam in active_families:
            verify_k(infl, fam, family_k[fam])
    except InstrumentDefectError as exc:
        return _write_instrument_defect(exc, args)

    join_info = check_join_completeness(table, infl)

    if args.dry_run:
        print(f"venue: {venue}")
        print(f"table: {args.table} ({len(table)} rows), sha256 OK")
        print(f"influence: {args.influence} ({len(infl)} rows)")
        print(
            f"join: {join_info['n_table_rows']} table rows / {join_info['n_influence_rows']} "
            f"influence rows joined on event_idx; unmatched table-only="
            f"{join_info['n_unmatched_table_only']}, unmatched influence-only="
            f"{join_info['n_unmatched_influence_only']}; join_complete={join_info['join_complete']}"
        )
        print(f"logL columns present: {logl_matrix.size > 0} (h_grid n={h_grid.size})")
        for fam in active_families:
            k = int(infl[family_in_s_col(fam)].astype(bool).sum())
            print(f"  family {fam}: k={k}")
        print("dry-run OK")
        return 0

    report = build_report(table, infl, h_grid, logl_matrix, args)
    report["meta"]["missing_covariates"] = []
    report["meta"]["venue"] = venue
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {args.out}: disposition = {report['disposition']['value']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
