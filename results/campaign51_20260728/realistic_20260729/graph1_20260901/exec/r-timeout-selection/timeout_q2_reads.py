"""r-timeout-selection Q2 scorer — `q-timeout-population-mismatch` ONLY.

REGISTRATION_DRAFT.md §3 (Phase C, "Q2 reads (pure pandas)"), §4 (S2.1-S2.4),
§5 (Q2-S2.2 / Q2-S2.3 disposition rows), REVISION 1 + REVISION 2 + the CHAIR
ERRATUM (all append-only sections of the same file). Built per
`DESIGN_GATE_Q2_computability_rev2.md` (GREEN, F5/AMBER/F6 all closed).

Q1 (`q-timeout-selection-pdet`) is OUT OF SCOPE — this script never opens the
pool build's own timeout tally, never touches `p_det`/`SimulationDetectionProbability`,
and computes no S1.x statistic. The p0 axis is OUT OF SCOPE by construction
(D1 record, ratified bound) — p0 appears only as a REPORTED-ONLY covariate
(S2.1 bins, S2.4 scatter).

Every §1 input is md5-pinned on the CLI; a mismatch, missing file, or missing
required column is a hard INSTRUMENT-DEFECT, raised BEFORE any statistic is
touched, written to `--out` (real mode) or printed non-zero (`--dry-run`) —
never a silent skip.

Statistics (REGISTRATION_DRAFT.md §4, Q2 only):
  S2.1 — per-M-bin (+ REPORTED-ONLY p0/e0 quintile bins) info map: n, median,
         IQR of sigma_lnDL, sky area Omega, SNR, generation_time; Spearman
         rho_S(log10 M, ln sigma_lnDL) with a 10,000-permutation p.
  S2.2 — Spearman rho_S(log10 M, d_e) [gates] and rho_S(log10 M, |d_e|)
         [REPORTED-ONLY], 10,000-perm p (seed 20260904); top-k M-bin
         composition vs bulk, Fisher exact per bin + Holm over 5 bins, for
         k = 82 (iiib 2D, primary), k = 94 (iiib 1D, replicate), k = 72
         (joint_r1 2D, replicate).
  S2.3 — PRIMARY composition counterfactual: w_b = share_pool,det(b) /
         share_kept(b) over the SUPPORTED bins {2, 3} only (REVISION 1 F2/F3,
         REVISION 2 F5), unit weight elsewhere, ONE global renormalisation to
         Sigma_e w_e = 1588 (g-closure ii); re-weighted T0 posterior (mean_h',
         sigma_h') under the frozen T0 convention (imported from
         `build_influence_vector.py`, cited not re-derived), Delta_mean_h^Q2,
         sigma'_h/sigma_h (iiib 2D primary; 1D + joint_r1 2D replicates);
         same-size null (1000 draws, w_e permuted over events, seed 20260904).
         Decomposition REPORTED-ONLY: share_to(b) of the 820 SNR-stage
         timeouts (+2 CRB-stage, reported per the CHAIR ERRATUM), and the
         D1-gate share (disclosed NOT-EVALUABLE per bin).
  S2.4 — REPORTED-ONLY: timeouts' (log10 M, p0, mu/M) summary vs the kept set.

Disposition (REGISTRATION_DRAFT.md §5, fresh RULE, not pre-decided by this
script — it reports the three-valued tag per the registered bands only):
  Q2-S2.2 (iiib 2D / k=82 only; 1D/jr1 reported as replicates):
      MATERIAL   p_perm < 0.01 AND top-k Fisher Holm-p < 0.05 in >=1 bin -> M-STRUCTURED
      IMMATERIAL p_perm >= 0.10 -> M-FLAT
      else INTERMEDIATE
  Q2-S2.3 (iiib 2D primary):
      MATERIAL   |Delta_mean_h^Q2| >= T_mat OR sigma'_h/sigma_h not in [0.80,1.25] -> POPULATION-MISMATCH-MATERIAL
      IMMATERIAL |Delta| <= T_null AND ratio in [0.95,1.05] -> POPULATION-MISMATCH-IMMATERIAL
      else INTERMEDIATE

Both rows carry the mandatory line: "p0 axis not evaluated (D1 record,
ratified bound; read's p0 row corrected per MECHANISM_NOTE.md §3)".

Blindness: this script computes every registered Q2 statistic over the
REGISTERED 1588-event population ONLY when invoked in real mode by the
author/orchestrator outside this build. The build record for this script
(BUILD_RECORD_Q2.md) exercises it ONLY on --dry-run (real inputs, no
aggregate) and a synthetic <=10-row fixture — never real mode.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import fisher_exact, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "r-offset-subset"))
import build_influence_vector as biv  # type: ignore[import-not-found]  # noqa: E402  (frozen T0 convention, cited not re-derived; resolved via the sys.path.insert above, not statically visible to mypy)

FloatArray = npt.NDArray[np.float64]

# ---------------------------------------------------------------------------
# Registered constants (REGISTRATION_DRAFT.md §1, §2, §5, REVISION 1/2, CHAIR ERRATUM)
# ---------------------------------------------------------------------------

TRUTH = 0.73
T_MAT = 0.008
NULL_DRAWS_DEFAULT = 1000
NULL_SEED_DEFAULT = 20260904
PERM_DRAWS_DEFAULT = 10000
PERM_SEED_DEFAULT = 20260904
N_SCORED = 1588
SCORED_EXCLUDED_EVENT_IDX = (1203, 1356)

# g-byteid anchors (REVISION 2, F5 closure).
N_KEPT_ANCHOR: tuple[int, ...] = (0, 9, 1276, 303, 0)
N_TIMEOUT_SNR_STAGE_ANCHOR: tuple[int, ...] = (
    206,
    302,
    216,
    81,
    15,
)  # rate_table_M.csv n_timeout, REPORTED-ONLY source
N_TIMEOUT_SNR_STAGE_TOTAL = 820
N_TIMEOUT_CRB_STAGE_TOTAL = (
    2  # CHAIR ERRATUM: reported alongside, never enters n_timeout / S2.3 decomposition
)
N_TIMEOUT_ALL_TOTAL = 822

# g-population anchors (REGISTRATION_DRAFT.md §6).
G_POPULATION_N_TASKS = 100
G_POPULATION_SUM_Y = 89_456
G_POPULATION_D1_GATE_LINES = 4_071
G_POOL_TOTAL_ROWS = 200_100
G_POOL_A_ROWS = 99_014
G_POOL_MANIFEST_FILES = 707

# Supported-bin rule (REVISION 1 F2/F3, REVISION 2 F5): n_kept(b) >= 10.
SUPPORTED_BINS: tuple[int, ...] = (2, 3)
N_BINS = 5

# S2.2 registered top-k families (REGISTRATION_DRAFT.md §4 S2.2).
FAMILY_K: dict[str, int] = {"iiib_2d": 82, "iiib_1d": 94, "jr1_2d": 72}
FAMILY_INFLUENCE_COL: dict[str, str] = {
    "iiib_2d": "influence_2D",
    "iiib_1d": "influence_1D",
    "jr1_2d": "influence_2D",
}
PRIMARY_FAMILY = "iiib_2d"
REPLICATE_FAMILIES: tuple[str, ...] = ("iiib_1d", "jr1_2d")

# Frozen T0 anchors (REGISTRATION_DRAFT.md §1 "Anchors" line, row #302/#342 JSON).
ANCHOR_MEAN_H: dict[str, float] = {
    "iiib_2d": 0.6658540600,
    "iiib_1d": 0.6669870586,
    "jr1_2d": 0.6671274830,
}
ANCHOR_SIGMA_H_IIIB_2D = 0.018474739  # only the iiib-2D sigma_h is pinned in §1

FAMILY_VENUE: dict[str, str] = {"iiib_2d": "iiib", "iiib_1d": "iiib", "jr1_2d": "jr1"}
FAMILY_CHANNEL: dict[str, str] = {
    "iiib_2d": "combined_with_bh",
    "iiib_1d": "combined_no_bh",
    "jr1_2d": "combined_with_bh",
}

MANDATORY_P0_LINE = (
    "p0 axis not evaluated (D1 record, ratified bound; read's p0 row corrected "
    "per MECHANISM_NOTE.md §3)"
)


class InstrumentDefectError(Exception):
    """Hard pre-flight failure — raised BEFORE any registered statistic is
    touched. Caught once at the top of `main()`; written into the output
    JSON's `disposition` (real mode) or printed with a non-zero exit
    (`--dry-run`). Never a silent skip.
    """

    def __init__(self, message: str, detail: dict[str, Any]):
        super().__init__(message)
        self.message = message
        self.detail = detail


# ---------------------------------------------------------------------------
# Pin verification (G-1)
# ---------------------------------------------------------------------------


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_pin(path: Path, expected_md5: str, label: str) -> None:
    if not path.exists():
        raise InstrumentDefectError(
            f"G-1 pin: {label} not found at {path}", {"path": str(path), "label": label}
        )
    actual = _md5(path)
    if actual != expected_md5:
        raise InstrumentDefectError(
            f"G-1 pin: {label} md5 mismatch — expected {expected_md5}, got {actual}",
            {"path": str(path), "label": label, "expected_md5": expected_md5, "actual_md5": actual},
        )


def _check_manifest(
    manifest_path: Path, expected_manifest_md5: str, data_dir: Path, label: str
) -> int:
    """G-1: verify the manifest file's own md5, then md5sum -c every listed
    (md5, filename) pair against `data_dir`. Returns the number of files
    verified. Any mismatch or missing member file is INSTRUMENT-DEFECT.
    """
    _check_pin(manifest_path, expected_manifest_md5, f"{label} manifest")
    bad: list[str] = []
    n = 0
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        expected_md5, fname = parts
        fpath = data_dir / fname
        if not fpath.exists():
            bad.append(f"{fname}: MISSING")
            continue
        actual = _md5(fpath)
        if actual != expected_md5:
            bad.append(f"{fname}: expected {expected_md5} got {actual}")
        n += 1
    if bad:
        raise InstrumentDefectError(
            f"G-1 pin: {label} member-file md5 mismatch(es): {len(bad)}",
            {"label": label, "bad": bad[:20]},
        )
    return n


# ---------------------------------------------------------------------------
# §1 loaders
# ---------------------------------------------------------------------------


def load_crb_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.reset_index(drop=True).copy()
    df["event_idx"] = df.index
    required = {
        "M",
        "p0",
        "e0",
        "SNR",
        "luminosity_distance",
        "delta_luminosity_distance_delta_luminosity_distance",
        "delta_qS_delta_qS",
        "delta_phiS_delta_qS",
        "delta_phiS_delta_phiS",
        "qS",
        "generation_time",
    }
    missing = required - set(df.columns)
    if missing:
        raise InstrumentDefectError(
            f"CRB CSV missing required column(s): {sorted(missing)}", {"missing": sorted(missing)}
        )
    return df


def scored_crb(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["event_idx"].isin(SCORED_EXCLUDED_EVENT_IDX)].copy()


def load_bin_edges(path: Path) -> FloatArray:
    obj = json.loads(path.read_text())
    if "seed61000_M_edges" not in obj:
        raise InstrumentDefectError(
            "bin edges JSON missing 'seed61000_M_edges'", {"path": str(path)}
        )
    edges = np.asarray(obj["seed61000_M_edges"], dtype=np.float64)
    if edges.size != N_BINS + 1:
        raise InstrumentDefectError(
            f"bin edges JSON 'seed61000_M_edges' has {edges.size} entries, expected {N_BINS + 1}",
            {"path": str(path), "n_edges": int(edges.size)},
        )
    return edges


def digitize_M(M: FloatArray, edges: FloatArray) -> npt.NDArray[np.intp]:
    return np.digitize(M, edges[1:-1], right=False)


def load_pool(pool_dir: Path, manifest_path: Path, manifest_md5: str) -> pd.DataFrame:
    n_files = _check_manifest(manifest_path, manifest_md5, pool_dir, "pool")
    if n_files != G_POOL_MANIFEST_FILES:
        raise InstrumentDefectError(
            f"pool manifest lists {n_files} files, expected {G_POOL_MANIFEST_FILES}",
            {"n_files": n_files, "expected": G_POOL_MANIFEST_FILES},
        )
    files = sorted(glob.glob(str(pool_dir / "injection_h_0p73_task_*.csv")))
    if len(files) != G_POOL_MANIFEST_FILES:
        raise InstrumentDefectError(
            f"pool dir has {len(files)} injection_h_0p73_task_*.csv files, manifest lists {G_POOL_MANIFEST_FILES}",
            {"n_files_on_disk": len(files)},
        )
    dfs = [pd.read_csv(f) for f in files]
    pool = pd.concat(dfs, ignore_index=True)
    required = {"M", "SNR", "stratum"}
    missing = required - set(pool.columns)
    if missing:
        raise InstrumentDefectError(
            f"pool CSVs missing required column(s): {sorted(missing)}", {"missing": sorted(missing)}
        )
    if len(pool) != G_POOL_TOTAL_ROWS:
        raise InstrumentDefectError(
            f"pool has {len(pool)} rows, g-population anchor is {G_POOL_TOTAL_ROWS}",
            {"n_rows": len(pool), "expected": G_POOL_TOTAL_ROWS},
        )
    n_a = int((pool["stratum"] == "a").sum())
    if n_a != G_POOL_A_ROWS:
        raise InstrumentDefectError(
            f"pool a-stratum has {n_a} rows, g-population anchor is {G_POOL_A_ROWS}",
            {"n_a": n_a, "expected": G_POOL_A_ROWS},
        )
    return pool


_SNR_STAGE_MSG = "Waveform/SNR computation timed out"
_CRB_STAGE_MSG = "bound computation timed out"  # matches both 'Cramér-Rao' encodings
_M_RE = re.compile(r"'M':\s*(?:np\.float64\()?([\-0-9.eE]+)\)?")
_Y_RE = re.compile(r"(\d+) / (\d+) evaluations successful")


def parse_logs(log_dir: Path, manifest_path: Path, manifest_md5: str) -> dict[str, Any]:
    """`log_dir` is the manifest's OWN root (`cluster_logs_fetch_20260904/`,
    the directory the manifest's `./logs/...` entries are relative to) — the
    `.err` files parsed below live one level down, in `log_dir/logs/`.
    """
    n_files = _check_manifest(manifest_path, manifest_md5, log_dir, "log")
    err_files = sorted(glob.glob(str(log_dir / "logs" / "simulate_6088772_*.err")))
    if len(err_files) != G_POPULATION_N_TASKS:
        raise InstrumentDefectError(
            f"{len(err_files)} simulate_6088772_*.err files found, g-population anchor is {G_POPULATION_N_TASKS}",
            {"n_files": len(err_files), "expected": G_POPULATION_N_TASKS},
        )
    del (
        n_files
    )  # manifest covers the whole log tree, not just these .err files; count checked above

    snr_M: list[float] = []
    crb_M: list[float] = []
    d1_gate_lines = 0
    sum_Y = 0
    for f in err_files:
        last_Y: int | None = None
        text = Path(f).read_text(errors="replace")
        for line in text.splitlines():
            if _SNR_STAGE_MSG in line:
                m = _M_RE.search(line)
                if m:
                    snr_M.append(float(m.group(1)))
            elif _CRB_STAGE_MSG in line:
                m = _M_RE.search(line)
                if m:
                    crb_M.append(float(m.group(1)))
            elif "in dervative" in line:
                d1_gate_lines += 1
            ym = _Y_RE.search(line)
            if ym:
                last_Y = int(ym.group(2))
        if last_Y is not None:
            sum_Y += last_Y

    if len(snr_M) != N_TIMEOUT_SNR_STAGE_TOTAL:
        raise InstrumentDefectError(
            f"parsed {len(snr_M)} SNR-stage timeout records, anchor is {N_TIMEOUT_SNR_STAGE_TOTAL}",
            {"n_parsed": len(snr_M), "expected": N_TIMEOUT_SNR_STAGE_TOTAL},
        )
    if len(crb_M) != N_TIMEOUT_CRB_STAGE_TOTAL:
        raise InstrumentDefectError(
            f"parsed {len(crb_M)} CRB-stage timeout records, anchor is {N_TIMEOUT_CRB_STAGE_TOTAL}",
            {"n_parsed": len(crb_M), "expected": N_TIMEOUT_CRB_STAGE_TOTAL},
        )
    if sum_Y != G_POPULATION_SUM_Y:
        raise InstrumentDefectError(
            f"Sigma last-Y across tasks = {sum_Y}, g-population anchor is {G_POPULATION_SUM_Y}",
            {"sum_Y": sum_Y, "expected": G_POPULATION_SUM_Y},
        )
    if d1_gate_lines != G_POPULATION_D1_GATE_LINES:
        raise InstrumentDefectError(
            f"'in dervative' D1-gate line count = {d1_gate_lines}, g-population anchor is {G_POPULATION_D1_GATE_LINES}",
            {"d1_gate_lines": d1_gate_lines, "expected": G_POPULATION_D1_GATE_LINES},
        )
    return {
        "snr_stage_timeout_M": np.asarray(snr_M, dtype=np.float64),
        "crb_stage_timeout_M": np.asarray(crb_M, dtype=np.float64),
        "sum_Y": sum_Y,
        "d1_gate_lines": d1_gate_lines,
    }


def load_influence(path: Path, expected_md5: str, label: str) -> pd.DataFrame:
    _check_pin(path, expected_md5, label)
    df = pd.read_csv(path)
    required = {"event_idx", "influence_2D", "influence_1D", "rank"}
    missing = required - set(df.columns)
    if missing:
        raise InstrumentDefectError(
            f"{label} missing required column(s): {sorted(missing)}", {"missing": sorted(missing)}
        )
    if len(df) != N_SCORED:
        raise InstrumentDefectError(
            f"{label} has {len(df)} rows, expected {N_SCORED}",
            {"n_rows": len(df), "expected": N_SCORED},
        )
    return df.set_index("event_idx", drop=False).sort_index()


# ---------------------------------------------------------------------------
# g-byteid gate (REVISION 2 F5 closure)
# ---------------------------------------------------------------------------


def compute_n_kept(crb_scored: pd.DataFrame, edges: FloatArray) -> npt.NDArray[np.int64]:
    bins = digitize_M(crb_scored["M"].to_numpy(dtype=np.float64), edges)
    return np.bincount(bins, minlength=N_BINS)


def compute_n_timeout_snr_stage(
    snr_stage_M: FloatArray, edges: FloatArray
) -> npt.NDArray[np.int64]:
    bins = digitize_M(snr_stage_M, edges)
    return np.bincount(bins, minlength=N_BINS)


def gbyteid_gate(n_kept: npt.NDArray[np.int64], n_timeout_snr: npt.NDArray[np.int64]) -> None:
    if tuple(int(x) for x in n_kept) != N_KEPT_ANCHOR:
        raise InstrumentDefectError(
            f"g-byteid: n_kept = {list(n_kept)}, anchor is {list(N_KEPT_ANCHOR)}",
            {"n_kept": [int(x) for x in n_kept], "anchor": list(N_KEPT_ANCHOR)},
        )
    if tuple(int(x) for x in n_timeout_snr) != N_TIMEOUT_SNR_STAGE_ANCHOR:
        raise InstrumentDefectError(
            f"g-byteid: n_timeout (SNR-stage) = {list(n_timeout_snr)}, anchor is {list(N_TIMEOUT_SNR_STAGE_ANCHOR)}",
            {
                "n_timeout": [int(x) for x in n_timeout_snr],
                "anchor": list(N_TIMEOUT_SNR_STAGE_ANCHOR),
            },
        )


def gclosure_i_gate(sum_Y: int, d1_gate_lines: int) -> int:
    """g-closure(i): the residual = Sigma_Y - ZeroDiv(combined) - (N_sim_completed
    + n_timeout_all). REGISTRATION_DRAFT §6 pins the residual formula as
    Q1-scoped (feeds S1.2's scale factor, not any Q2 statistic); this Q2
    scorer computes it ONLY as a disclosed cross-check, never a Q2 gate.
    """
    # 78,841 (SNR-fail) + 5,921 (SNR-pass) = 84,762; + 822 timeouts = 85,584 (g-closure i, by construction).
    n_sim_completed = 78_841 + 5_921
    denom = n_sim_completed + N_TIMEOUT_ALL_TOTAL
    zerodiv_combined = (
        3_488  # MECHANISM_NOTE.md §3 combined SNR(3,449)+CRB(39) total; Q1-scoped, disclosed only
    )
    residual = sum_Y - zerodiv_combined - denom
    return residual


# ---------------------------------------------------------------------------
# S2.1 — information map (REPORTED-ONLY)
# ---------------------------------------------------------------------------


def _sky_area(row: pd.Series) -> float:
    c_qsqs = row["delta_qS_delta_qS"]
    c_phisphis = row["delta_phiS_delta_phiS"]
    c_qsphis = row["delta_phiS_delta_qS"]
    det = c_qsqs * c_phisphis - c_qsphis**2
    return float(2 * np.pi * abs(np.sin(row["qS"])) * np.sqrt(max(det, 0.0)))


def s2_1_information_map(
    crb_scored: pd.DataFrame, edges: FloatArray, perm_draws: int, perm_seed: int
) -> dict[str, Any]:
    df = crb_scored.copy()
    df["sigma_lnDL"] = (
        np.sqrt(df["delta_luminosity_distance_delta_luminosity_distance"])
        / df["luminosity_distance"]
    )
    df["Omega_sr"] = df.apply(_sky_area, axis=1)
    df["M_bin"] = digitize_M(df["M"].to_numpy(dtype=np.float64), edges)
    df["p0_bin"] = pd.qcut(df["p0"], 5, labels=False, duplicates="drop")
    df["e0_bin"] = pd.qcut(df["e0"], 5, labels=False, duplicates="drop")

    def _stats(g: pd.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {"n": int(len(g))}
        for col in ("sigma_lnDL", "Omega_sr", "SNR", "generation_time"):
            vals = g[col].to_numpy(dtype=np.float64)
            out[f"{col}_median"] = float(np.median(vals))
            out[f"{col}_iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        return out

    by_M = {int(b): _stats(g) for b, g in df.groupby("M_bin")}
    by_p0 = {int(b): _stats(g) for b, g in df.groupby("p0_bin")}
    by_e0 = {int(b): _stats(g) for b, g in df.groupby("e0_bin")}

    log10M = np.log10(df["M"].to_numpy(dtype=np.float64))
    ln_sigma = np.log(df["sigma_lnDL"].to_numpy(dtype=np.float64))
    rho, _p_asymp = spearmanr(log10M, ln_sigma)
    rng = np.random.default_rng(perm_seed)
    perm_rhos = np.empty(perm_draws)
    for i in range(perm_draws):
        shuffled = rng.permutation(ln_sigma)
        perm_rhos[i], _ = spearmanr(log10M, shuffled)
    p_perm = float((np.sum(np.abs(perm_rhos) >= abs(rho)) + 1) / (perm_draws + 1))

    return {
        "by_M_bin": by_M,
        "by_p0_bin_reported_only": by_p0,
        "by_e0_bin_reported_only": by_e0,
        "spearman_log10M_ln_sigma_lnDL": {
            "rho": float(rho),
            "p_perm": p_perm,
            "n_perm": perm_draws,
            "seed": perm_seed,
        },
    }


# ---------------------------------------------------------------------------
# S2.2 — influence vs M
# ---------------------------------------------------------------------------


def _holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((m - rank) * pvals[idx], 1.0)
        running_max = max(running_max, val)
        adj[idx] = running_max
    return [float(x) for x in adj]


def s2_2_influence_vs_M(
    crb_scored: pd.DataFrame,
    edges: FloatArray,
    infl_iiib: pd.DataFrame,
    infl_jr1: pd.DataFrame,
    perm_draws: int,
    perm_seed: int,
) -> dict[str, Any]:
    M_by_event = crb_scored.set_index("event_idx")["M"]
    result: dict[str, Any] = {}
    for family, k in FAMILY_K.items():
        infl = infl_iiib if FAMILY_VENUE[family] == "iiib" else infl_jr1
        d_e_col = FAMILY_INFLUENCE_COL[family]
        joined = infl.join(M_by_event, how="inner")
        if len(joined) != N_SCORED:
            raise InstrumentDefectError(
                f"S2.2 {family}: joined influence/M table has {len(joined)} rows, expected {N_SCORED}",
                {"family": family, "n_joined": len(joined)},
            )
        d_e = joined[d_e_col].to_numpy(dtype=np.float64)
        M = joined["M"].to_numpy(dtype=np.float64)
        log10M = np.log10(M)
        M_bin = digitize_M(M, edges)

        rho, _ = spearmanr(log10M, d_e)
        rho_abs, _ = spearmanr(log10M, np.abs(d_e))
        rng = np.random.default_rng(perm_seed)
        perm_rhos = np.empty(perm_draws)
        for i in range(perm_draws):
            shuffled = rng.permutation(d_e)
            perm_rhos[i], _ = spearmanr(log10M, shuffled)
        p_perm = float((np.sum(np.abs(perm_rhos) >= abs(rho)) + 1) / (perm_draws + 1))

        order_top_k = np.argsort(-d_e)[:k]
        in_top_k = np.zeros(len(joined), dtype=bool)
        in_top_k[order_top_k] = True

        fisher_p: list[float] = []
        fisher_detail: list[dict[str, Any]] = []
        for b in range(N_BINS):
            in_bin = M_bin == b
            a_ = int(np.sum(in_top_k & in_bin))
            b_ = int(np.sum(in_top_k & ~in_bin))
            c_ = int(np.sum(~in_top_k & in_bin))
            d_ = int(np.sum(~in_top_k & ~in_bin))
            _odds, p = fisher_exact([[a_, b_], [c_, d_]])
            fisher_p.append(float(p))
            fisher_detail.append(
                {
                    "bin": b,
                    "top_k_in_bin": a_,
                    "top_k_not_in_bin": b_,
                    "bulk_in_bin": c_,
                    "bulk_not_in_bin": d_,
                    "p": float(p),
                }
            )
        holm_p = _holm(fisher_p)
        for entry, hp in zip(fisher_detail, holm_p, strict=True):
            entry["holm_p"] = hp

        result[family] = {
            "k": k,
            "n": len(joined),
            "rho_log10M_d_e": float(rho),
            "p_perm_d_e": p_perm,
            "rho_log10M_abs_d_e_reported_only": float(rho_abs),
            "n_perm": perm_draws,
            "seed": perm_seed,
            "fisher_per_bin": fisher_detail,
            "min_holm_p": float(min(holm_p)),
            "any_bin_holm_p_lt_0p05": bool(min(holm_p) < 0.05),
        }
    return result


# ---------------------------------------------------------------------------
# S2.3 — composition counterfactual (PRIMARY)
# ---------------------------------------------------------------------------


def s2_3_weights(crb_scored: pd.DataFrame, pool: pd.DataFrame, edges: FloatArray) -> dict[str, Any]:
    M_bin_kept = digitize_M(crb_scored["M"].to_numpy(dtype=np.float64), edges)
    n_kept = np.bincount(M_bin_kept, minlength=N_BINS)

    a = pool[pool["stratum"] == "a"]
    det = a[a["SNR"] >= 20]
    M_bin_det = digitize_M(det["M"].to_numpy(dtype=np.float64), edges)
    n_pool_det = np.bincount(M_bin_det, minlength=N_BINS)

    share_kept_support = {b: n_kept[b] / n_kept[list(SUPPORTED_BINS)].sum() for b in SUPPORTED_BINS}
    share_pool_det_support = {
        b: n_pool_det[b] / n_pool_det[list(SUPPORTED_BINS)].sum() for b in SUPPORTED_BINS
    }
    w_b = {b: share_pool_det_support[b] / share_kept_support[b] for b in SUPPORTED_BINS}

    events = crb_scored[["event_idx"]].copy()
    events["M_bin"] = M_bin_kept
    events["w_e_raw"] = events["M_bin"].map(lambda b: w_b.get(b, 1.0))
    total = events["w_e_raw"].sum()
    # g-closure(ii): renormalise to the ACTUAL number of scored events passed
    # in (== N_SCORED == 1588 in real mode; a synthetic fixture may pass a
    # smaller table, and the closure sum must still equal len(events), not
    # the hard-coded registered constant).
    events["w_e"] = events["w_e_raw"] * (len(events) / total)

    return {
        "n_kept_per_bin": [int(x) for x in n_kept],
        "n_pool_det_per_bin": [int(x) for x in n_pool_det],
        "share_kept_support": {str(k): float(v) for k, v in share_kept_support.items()},
        "share_pool_det_support": {str(k): float(v) for k, v in share_pool_det_support.items()},
        "w_b": {str(k): float(v) for k, v in w_b.items()},
        "supported_bins": list(SUPPORTED_BINS),
        "n_events_reweighted": int((events["M_bin"].isin(SUPPORTED_BINS)).sum()),
        "n_events_unit_weight": int((~events["M_bin"].isin(SUPPORTED_BINS)).sum()),
        "sum_w_e_pre_renorm": float(total),
        "sum_w_e_post_renorm": float(events["w_e"].sum()),
        "events": events.set_index("event_idx")["w_e"],
    }


def _load_logl(
    path: Path, expected_md5: str, venue_key: str, channel: str
) -> tuple[FloatArray, FloatArray, FloatArray]:
    actual = biv._md5(path)
    if actual != expected_md5:
        raise InstrumentDefectError(
            f"event_likelihoods ({venue_key}) md5 mismatch — expected {expected_md5}, got {actual}",
            {
                "path": str(path),
                "venue": venue_key,
                "expected_md5": expected_md5,
                "actual_md5": actual,
            },
        )
    h_grid, event_idx, logL, n_excluded = biv._load_matrix(path, channel)
    if n_excluded:
        raise InstrumentDefectError(
            f"event_likelihoods ({venue_key}/{channel}) physics-floor excluded {n_excluded} row(s); "
            "the frozen T0 anchor assumes 0 exclusions",
            {"venue": venue_key, "channel": channel, "n_excluded": n_excluded},
        )
    return h_grid, event_idx, logL


def _weighted_moments(logL: FloatArray, h_grid: FloatArray, w_e: FloatArray) -> tuple[float, float]:
    weights = np.gradient(h_grid)
    logpost = (w_e[:, None] * logL).sum(axis=0)
    mean_arr, sigma_arr, _map_arr = biv._moments(logpost[None, :], h_grid, weights)
    return float(mean_arr[0]), float(sigma_arr[0])


def s2_3_reweighted_posterior(
    family: str,
    logl_path: Path,
    logl_md5: str,
    w_e_by_event: pd.Series,
    null_draws: int,
    null_seed: int,
) -> dict[str, Any]:
    venue = FAMILY_VENUE[family]
    channel = FAMILY_CHANNEL[family]
    h_grid, event_idx, logL = _load_logl(logl_path, logl_md5, venue, channel)
    order = np.argsort(event_idx)
    event_idx = event_idx[order]
    logL = logL[order]
    if event_idx.size != N_SCORED or not np.array_equal(
        event_idx, np.sort(w_e_by_event.index.to_numpy())
    ):
        raise InstrumentDefectError(
            f"S2.3 {family}: event_likelihoods event_idx set does not match the {N_SCORED}-event w_e index",
            {"family": family, "n_logl_events": int(event_idx.size)},
        )
    w_e = w_e_by_event.reindex(event_idx).to_numpy(dtype=np.float64)
    if abs(w_e.sum() - N_SCORED) > 1e-6:
        raise InstrumentDefectError(
            f"g-closure(ii): Sigma w_e = {w_e.sum()} for family {family}, expected {N_SCORED}",
            {"family": family, "sum_w_e": float(w_e.sum())},
        )

    mean_h, sigma_h = _weighted_moments(logL, h_grid, w_e)
    anchor_mean = ANCHOR_MEAN_H[family]
    delta_mean_h = mean_h - anchor_mean

    rng = np.random.default_rng(null_seed)
    null_deltas = np.empty(null_draws)
    for i in range(null_draws):
        w_perm = rng.permutation(w_e)
        m, _s = _weighted_moments(logL, h_grid, w_perm)
        null_deltas[i] = m - anchor_mean
    sd_null = float(np.std(null_deltas))
    t_null = max(0.002, 2 * sd_null)

    out: dict[str, Any] = {
        "family": family,
        "mean_h_reweighted": mean_h,
        "sigma_h_reweighted": sigma_h,
        "anchor_mean_h": anchor_mean,
        "delta_mean_h": delta_mean_h,
        "null_draws": null_draws,
        "null_seed": null_seed,
        "sd_delta_null": sd_null,
        "t_null": t_null,
    }
    if family == PRIMARY_FAMILY:
        ratio = sigma_h / ANCHOR_SIGMA_H_IIIB_2D
        out["anchor_sigma_h"] = ANCHOR_SIGMA_H_IIIB_2D
        out["sigma_ratio"] = ratio
    return out


def s2_3_decomposition(logs: dict[str, Any], edges: FloatArray) -> dict[str, Any]:
    """REPORTED-ONLY: share_to(b) of the 820 SNR-stage timeouts (+2 CRB-stage,
    reported per the CHAIR ERRATUM); D1-gate share is NOT-EVALUABLE per bin
    (no params logged at the D1 catch site) and disclosed as such.
    """
    snr_bins = digitize_M(logs["snr_stage_timeout_M"], edges)
    n_snr = np.bincount(snr_bins, minlength=N_BINS)
    share_to = {int(b): float(n_snr[b] / N_TIMEOUT_SNR_STAGE_TOTAL) for b in range(N_BINS)}
    crb_bins = digitize_M(logs["crb_stage_timeout_M"], edges)
    return {
        "share_to_snr_stage_820": share_to,
        "crb_stage_timeouts_reported_only": {
            "n": N_TIMEOUT_CRB_STAGE_TOTAL,
            "bins": [int(x) for x in crb_bins],
        },
        "d1_gate_share_per_bin": "NOT-EVALUABLE (no params logged at the D1 catch site; disclosed)",
    }


# ---------------------------------------------------------------------------
# S2.4 — hypothesis line (REPORTED-ONLY)
# ---------------------------------------------------------------------------


def s2_4_scatter_summary(crb_scored: pd.DataFrame, logs: dict[str, Any]) -> dict[str, Any]:
    def _summ(M: FloatArray) -> dict[str, float]:
        log10M = np.log10(M)
        return {
            "n": int(M.size),
            "log10M_median": float(np.median(log10M)),
            "log10M_iqr": float(np.percentile(log10M, 75) - np.percentile(log10M, 25)),
        }

    all_timeout_M = np.concatenate([logs["snr_stage_timeout_M"], logs["crb_stage_timeout_M"]])
    return {
        "timeouts_all_822_reported_only": _summ(all_timeout_M),
        "kept_1588": {
            "n": int(len(crb_scored)),
            "log10M_median": float(np.median(np.log10(crb_scored["M"].to_numpy(dtype=np.float64)))),
            "p0_median": float(np.median(crb_scored["p0"].to_numpy(dtype=np.float64))),
        },
        "mu_over_M_note": "mu constant 10 in every sampled record (MECHANISM_NOTE.md §5, spot-checked); mu/M reported via M only",
    }


# ---------------------------------------------------------------------------
# Dispositions (§5) — fresh RULE, reported not pre-decided
# ---------------------------------------------------------------------------


def disposition_s2_2(primary: dict[str, Any]) -> dict[str, Any]:
    p_perm = primary["p_perm_d_e"]
    any_holm = primary["any_bin_holm_p_lt_0p05"]
    if p_perm < 0.01 and any_holm:
        value = "M-STRUCTURED"
    elif p_perm >= 0.10:
        value = "M-FLAT"
    else:
        value = "INTERMEDIATE"
    return {
        "value": value,
        "p_perm": p_perm,
        "any_bin_holm_p_lt_0p05": any_holm,
        "mandatory_note": MANDATORY_P0_LINE,
    }


def disposition_s2_3(primary: dict[str, Any]) -> dict[str, Any]:
    delta = abs(primary["delta_mean_h"])
    ratio = primary["sigma_ratio"]
    t_null = primary["t_null"]
    if delta >= T_MAT or not (0.80 <= ratio <= 1.25):
        value = "POPULATION-MISMATCH-MATERIAL"
    elif delta <= t_null and 0.95 <= ratio <= 1.05:
        value = "POPULATION-MISMATCH-IMMATERIAL"
    else:
        value = "POPULATION-MISMATCH-INTERMEDIATE"
    return {
        "value": value,
        "abs_delta_mean_h": delta,
        "t_mat": T_MAT,
        "t_null": t_null,
        "sigma_ratio": ratio,
        "mandatory_note": MANDATORY_P0_LINE,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crb-csv", type=Path, required=True)
    p.add_argument("--crb-csv-md5", type=str, required=True)
    p.add_argument("--bin-edges-json", type=Path, required=True)
    p.add_argument("--bin-edges-md5", type=str, required=True)
    p.add_argument("--rate-table-m-csv", type=Path, required=True)
    p.add_argument("--rate-table-m-md5", type=str, required=True)
    p.add_argument("--pool-dir", type=Path, required=True)
    p.add_argument("--pool-manifest", type=Path, required=True)
    p.add_argument("--pool-manifest-md5", type=str, required=True)
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--log-manifest", type=Path, required=True)
    p.add_argument("--log-manifest-md5", type=str, required=True)
    p.add_argument("--event-likelihoods-iiib", type=Path, required=True)
    p.add_argument("--event-likelihoods-iiib-md5", type=str, required=True)
    p.add_argument("--event-likelihoods-jr1", type=Path, required=True)
    p.add_argument("--event-likelihoods-jr1-md5", type=str, required=True)
    p.add_argument("--influence-iiib", type=Path, required=True)
    p.add_argument("--influence-iiib-md5", type=str, required=True)
    p.add_argument("--influence-jr1", type=Path, required=True)
    p.add_argument("--influence-jr1-md5", type=str, required=True)
    p.add_argument("--null-draws", type=int, default=NULL_DRAWS_DEFAULT)
    p.add_argument("--null-seed", type=int, default=NULL_SEED_DEFAULT)
    p.add_argument("--perm-draws", type=int, default=PERM_DRAWS_DEFAULT)
    p.add_argument("--perm-seed", type=int, default=PERM_SEED_DEFAULT)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _write_instrument_defect(exc: InstrumentDefectError, args: argparse.Namespace) -> int:
    print(f"INSTRUMENT-DEFECT: {exc.message}")
    if not args.dry_run:
        report = {
            "meta": {"crb_csv": str(args.crb_csv), "bin_edges_json": str(args.bin_edges_json)},
            "disposition": {
                "value": "INSTRUMENT-DEFECT",
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
    try:
        _check_pin(args.crb_csv, args.crb_csv_md5, "CRB CSV")
        _check_pin(args.bin_edges_json, args.bin_edges_md5, "bin edges JSON")
        _check_pin(
            args.rate_table_m_csv,
            args.rate_table_m_md5,
            "rate_table_M.csv (reported-only source of n_timeout anchor)",
        )

        crb = load_crb_csv(args.crb_csv)
        edges = load_bin_edges(args.bin_edges_json)
        crb_scored = scored_crb(crb)
        if len(crb_scored) != N_SCORED:
            raise InstrumentDefectError(
                f"scored CRB subset has {len(crb_scored)} rows, expected {N_SCORED}",
                {"n_scored": len(crb_scored), "expected": N_SCORED},
            )

        pool = load_pool(args.pool_dir, args.pool_manifest, args.pool_manifest_md5)
        logs = parse_logs(args.log_dir, args.log_manifest, args.log_manifest_md5)

        n_kept = compute_n_kept(crb_scored, edges)
        n_timeout_snr = compute_n_timeout_snr_stage(logs["snr_stage_timeout_M"], edges)
        gbyteid_gate(n_kept, n_timeout_snr)
        residual = gclosure_i_gate(logs["sum_Y"], logs["d1_gate_lines"])

        infl_iiib = load_influence(
            args.influence_iiib, args.influence_iiib_md5, "influence_iiib.csv"
        )
        infl_jr1 = load_influence(
            args.influence_jr1, args.influence_jr1_md5, "influence_joint_r1.csv"
        )
    except InstrumentDefectError as exc:
        return _write_instrument_defect(exc, args)

    if args.dry_run:
        print(f"CRB CSV: {args.crb_csv} ({len(crb)} rows; scored subset {len(crb_scored)}), md5 OK")
        print(f"bin edges: {args.bin_edges_json}, md5 OK ({edges.size} edges)")
        print(f"rate_table_M.csv: {args.rate_table_m_csv}, md5 OK (reported-only)")
        print(f"pool: {args.pool_dir} ({len(pool)} rows), manifest md5 OK")
        print(
            f"logs: {args.log_dir} (100/100 tasks), manifest md5 OK; Sigma Y = {logs['sum_Y']}; D1-gate lines = {logs['d1_gate_lines']}"
        )
        print(f"influence_iiib.csv: {args.influence_iiib} ({len(infl_iiib)} rows), md5 OK")
        print(f"influence_joint_r1.csv: {args.influence_jr1} ({len(infl_jr1)} rows), md5 OK")
        print(f"n_kept per bin (g-byteid): {list(n_kept)}")
        print(f"n_timeout SNR-stage per bin (g-byteid): {list(n_timeout_snr)}")
        print(f"CRB-stage timeouts (reported only): n={len(logs['crb_stage_timeout_M'])}")
        print(f"g-closure(i) residual (Q1-scoped, disclosed only): {residual}")
        print("dry-run OK (no aggregate computed)")
        return 0

    s2_1 = s2_1_information_map(crb_scored, edges, args.perm_draws, args.perm_seed)
    s2_2 = s2_2_influence_vs_M(
        crb_scored, edges, infl_iiib, infl_jr1, args.perm_draws, args.perm_seed
    )

    try:
        weights = s2_3_weights(crb_scored, pool, edges)
        sum_w = weights["sum_w_e_post_renorm"]
        if abs(sum_w - N_SCORED) > 1e-6:
            raise InstrumentDefectError(
                f"g-closure(ii): Sigma w_e = {sum_w} after renormalisation, expected {N_SCORED}",
                {"sum_w_e": sum_w},
            )
        w_e_by_event = weights.pop("events")

        s2_3_families: dict[str, Any] = {}
        s2_3_families[PRIMARY_FAMILY] = s2_3_reweighted_posterior(
            PRIMARY_FAMILY,
            args.event_likelihoods_iiib,
            args.event_likelihoods_iiib_md5,
            w_e_by_event,
            args.null_draws,
            args.null_seed,
        )
        s2_3_families["iiib_1d"] = s2_3_reweighted_posterior(
            "iiib_1d",
            args.event_likelihoods_iiib,
            args.event_likelihoods_iiib_md5,
            w_e_by_event,
            args.null_draws,
            args.null_seed,
        )
        s2_3_families["jr1_2d"] = s2_3_reweighted_posterior(
            "jr1_2d",
            args.event_likelihoods_jr1,
            args.event_likelihoods_jr1_md5,
            w_e_by_event,
            args.null_draws,
            args.null_seed,
        )
    except InstrumentDefectError as exc:
        return _write_instrument_defect(exc, args)

    s2_3_decomp = s2_3_decomposition(logs, edges)
    s2_4 = s2_4_scatter_summary(crb_scored, logs)

    disp_s2_2 = disposition_s2_2(s2_2[PRIMARY_FAMILY])
    disp_s2_3 = disposition_s2_3(s2_3_families[PRIMARY_FAMILY])

    report = {
        "meta": {
            "node": "r-timeout-selection",
            "question": "Q2 (q-timeout-population-mismatch) ONLY",
            "n_scored": N_SCORED,
            "n_kept_per_bin": [int(x) for x in n_kept],
            "n_timeout_snr_stage_per_bin": [int(x) for x in n_timeout_snr],
            "gclosure_i_residual_q1_scoped_disclosed_only": residual,
            "inputs": {
                "crb_csv": {"path": str(args.crb_csv), "md5": args.crb_csv_md5},
                "bin_edges_json": {"path": str(args.bin_edges_json), "md5": args.bin_edges_md5},
                "rate_table_m_csv_reported_only": {
                    "path": str(args.rate_table_m_csv),
                    "md5": args.rate_table_m_md5,
                },
                "pool_dir": str(args.pool_dir),
                "log_dir": str(args.log_dir),
                "event_likelihoods_iiib": {
                    "path": str(args.event_likelihoods_iiib),
                    "md5": args.event_likelihoods_iiib_md5,
                },
                "event_likelihoods_jr1": {
                    "path": str(args.event_likelihoods_jr1),
                    "md5": args.event_likelihoods_jr1_md5,
                },
                "influence_iiib": {
                    "path": str(args.influence_iiib),
                    "md5": args.influence_iiib_md5,
                },
                "influence_jr1": {"path": str(args.influence_jr1), "md5": args.influence_jr1_md5},
            },
        },
        "S2_1_information_map_reported_only": s2_1,
        "S2_2_influence_vs_M": s2_2,
        "S2_3_weights": weights,
        "S2_3_reweighted_posterior": s2_3_families,
        "S2_3_decomposition_reported_only": s2_3_decomp,
        "S2_4_reported_only": s2_4,
        "disposition_Q2_S2_2": disp_s2_2,
        "disposition_Q2_S2_3": disp_s2_3,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {args.out}: Q2-S2.2 = {disp_s2_2['value']}, Q2-S2.3 = {disp_s2_3['value']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
