"""highz_decomp_reads.py -- r-highz-completion, builder b-highz-decomp.

Implements REGISTRATION_DRAFT.md (this directory) exactly: the population
construction (Sec.1), the term-freeze counterfactual and score-excess
statistics (Sec.2/Sec.4), the harness control with between-universe
jackknife SE (Sec.4.3), the three-valued dispositions (Sec.5) and the gates
(Sec.6). See MECHANISM_NOTE.md for the code identity this script encodes
and DESIGN_GATE_computability.md for the independent computability review
this build responds to (three AMBER findings, all resolved below -- see
BUILD_RECORD.md Sec."Design-gate findings resolved").

T0 convention (gradient-trapezoid weights, physics-floor zero handling,
log-sum combination, uniform prior) is IMPORTED, never re-implemented, from
``exec/r-offset-subset/build_influence_vector.py`` (``_load_matrix``,
``_physics_floor_apply``, ``_moments``), itself citing
``results/prod2d_closure_20260818/tier0_bootstrap_jackknife.py`` as the
source of record (REGISTRATION_DRAFT.md Sec.1).

Blindness (memory ``gate-reviewers-must-not-compute-registered-statistic``):
this file is the builder's DELIVERABLE -- the registered aggregates
(Delta_F, Delta_t, shares, S_t/S_F, harness pooled values) are computed only
when the DISJOINT reader runs it in real mode (Sec.8 launch block). The
builder that wrote this file never invoked it outside ``--dry-run`` (gates +
byte-id anchors only, real inputs, no aggregate) and the SYNTHETIC fixture
(<=10 rows) -- see BUILD_RECORD.md.

CLI ARGPARSE MUST MATCH REGISTRATION_DRAFT.md SEC.8's LAUNCH BLOCK
TOKEN-FOR-TOKEN (design-gate item). Do not add or rename flags without
updating the draft.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Path setup / T0-convention import (not re-implementation -- Sec.1, Sec.3)
# --------------------------------------------------------------------------

REPO_ROOT = Path("/home/jasper/Repositories/darksiren-emri")
NODE_DIR = REPO_ROOT / (
    "results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-highz-completion"
)
OFFSET_SUBSET_DIR = REPO_ROOT / (
    "results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset"
)


def _import_t0_module() -> Any:
    """Import ``build_influence_vector.py`` by path (it is a script, not a package).

    Returns the module object; callers pull ``_load_matrix``,
    ``_physics_floor_apply`` and ``_moments`` off it -- the frozen T0
    convention, reused by import per REGISTRATION_DRAFT.md Sec.1/Sec.3.
    """
    path = OFFSET_SUBSET_DIR / "build_influence_vector.py"
    spec = importlib.util.spec_from_file_location("_t0_build_influence_vector", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"could not load T0 convention module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_T0 = _import_t0_module()
_load_matrix = _T0._load_matrix
_physics_floor_apply = _T0._physics_floor_apply
_moments = _T0._moments

sys.path.insert(0, str(REPO_ROOT))
from darksiren_emri.physical_relations import dist_to_redshift  # noqa: E402

# --------------------------------------------------------------------------
# Frozen pins (REGISTRATION_DRAFT.md Sec.1 / Sec.8) -- STOP on mismatch
# --------------------------------------------------------------------------

TRUTH = 0.73
DECILE = 0.10
STENCIL_NODES = (0.725, 0.730, 0.735)
STENCIL_STEP = 0.010
PRODUCTION_COMMIT = "1ec9514dd1808c48b18c0792dce558e5bba0f116"

LOGL_MD5 = {
    "iiib": "8e6a2c18dc5838dd1d52641589243672",
    "jr1": "745954a0fdee5f10878fb5e622a06144",
}
TABLE_SHA256 = {
    "iiib": "90c92026bb7fecff46e5a55e1e2c67a33b424e4b71611ee0d0854576b189f7b0",
    "jr1": "fc2eebe7fa66afbe2e35b0dd09c889be790511a6f5dabce3338969c849fcdf3a",
}
# Manifest construction resolves DESIGN_GATE_computability.md Sec.5 finding A:
# sorted "{seed} {md5}" lines of the event_likelihoods.csv md5 ONLY (the CRB
# file's md5 is not part of the manifest), joined by "\n", no trailing
# newline, sha256'd. Verified against the real 67-universe tree in the
# build -- see BUILD_RECORD.md.
HARNESS_MANIFEST_SHA256 = "6a06063dd56aae74ee1cc8bbc63f7da8207ff3e3fc705290a81a2675adb1c0a2"

# Population sha256 pins (comma-joined ascending event_idx list), Sec.1.
POPULATION_SHA256 = {
    ("iiib", "P_dark"): "5e7f0cf51f0d4f8a312414edd88a31594a5d07886316e7b559e85e831bd2b1e5",
    ("jr1", "P_dark"): "14ad8c17dfccb3d598e6014951595907bcde3f5fd4b9cbd00390395c50940258",
    ("iiib", "K"): "c8ce89931d7659a4c34e99f2c64b43a05f045c0504de16077c276375c7f9241f",
    ("jr1", "K"): "c8ce89931d7659a4c34e99f2c64b43a05f045c0504de16077c276375c7f9241f",
    ("iiib", "K_dark"): "50ae82c30142dc8ad7a2622fea56a29e9fce1b44ac48c5182b0b1be7e977d6ce",
    ("jr1", "K_dark"): "cb1def75e3f06f2f703e09d169c4ab2203f188c4a2484427177f807dc65d698b",
    ("iiib", "R"): "f7f494ce8e7d15a91d33b9a54cfc0e334a474929611496fc4a30a0565bbea6aa",
    ("jr1", "R"): "db7cbbb97a57f529d4ced1a14f02611e2fee8944befdb1f49f9c664bda4ee2a8",
}
POPULATION_N = {
    ("iiib", "P_dark"): 606,
    ("jr1", "P_dark"): 493,
    ("iiib", "K"): 159,
    ("jr1", "K"): 159,
    ("iiib", "K_dark"): 144,
    ("jr1", "K_dark"): 111,
    ("iiib", "K_hosted"): 15,
    ("jr1", "K_hosted"): 48,
    ("iiib", "R"): 231,
    ("jr1", "R"): 191,
}

# G-2(i) byte-id anchor: full-sample mean_h, 10 s.f. (build/BUILD_RECORD_B2.md).
G2_MEAN_H_FULL = {
    ("iiib", "2D"): 0.6658540600,
    ("iiib", "1D"): 0.6669869414,
    ("jr1", "2D"): 0.6671265168,
    ("jr1", "1D"): 0.6670323337,
}
G2_MEAN_H_TOL = 1e-9

# G-2(ii) byte-id anchor: leave-out of K in iiib 2D (END_VERIFICATION BATCH 2).
G2_DELTA_K_IIIB_2D = 0.086106
G2_DELTA_K_TOL = 1e-6

# G-2(iv) harness pooled byte-id anchors.
HARNESS_POOLED_ANCHORS = {
    "n_scored": 12060,
    "P_dark": 4826,
    "K": 1207,
    "K_dark": 1148,
}

CHANNELS = ("combined_no_bh", "combined_with_bh")
CHANNEL_LABEL = {"combined_no_bh": "1D", "combined_with_bh": "2D"}


def _separable_terms_for_channel(channel: str) -> tuple[str, ...]:
    """1D (`combined_no_bh`) has only T_B; 2D (`combined_with_bh`) has T_B, T_g."""
    return ("B", "g") if channel == "combined_with_bh" else ("B",)


# Columns whose full precision the closure gate (G-1(b)) and the term
# profiles require -- MECHANISM_NOTE.md Sec.4.
FULL_PRECISION_COLUMNS = (
    "B_num",
    "B_num_wbh",
    "L_cat_no_bh",
    "L_cat_with_bh",
    "combined_no_bh",
    "combined_with_bh",
    "den_log_term",
    "num_log_term_no_bh",
    "num_log_term_with_bh",
)
SEVEN_SF_COLUMNS = ("D_tilde_phi", "g_frac", "alpha_G_phi", "r_Malm", "w_tilde_G", "w_G_legacy")


class InstrumentDefect(SystemExit):
    """Raised (as SystemExit, nonzero) for a hard INSTRUMENT-DEFECT.

    A missing registered input, a pin mismatch, or a gate miss is a hard
    stop -- CLAUDE.md dataset-pinning rule / node lesson from
    ``exec/r-offset-subset/DESIGN_GATE_formula_rev3/4/5.md``: "a missing
    registered input is a hard INSTRUMENT-DEFECT."
    """

    def __init__(self, message: str) -> None:
        super().__init__(f"INSTRUMENT-DEFECT: {message}")


# --------------------------------------------------------------------------
# Hashing helpers
# --------------------------------------------------------------------------


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def population_hash(event_idx: np.ndarray) -> str:
    """sha256 of the comma-joined ASCENDING integer event_idx list (Sec.1)."""
    ordered = sorted(int(i) for i in event_idx)
    return sha256_str(",".join(str(i) for i in ordered))


def harness_manifest_hash(seed_md5_pairs: list[tuple[int, str]]) -> str:
    """sha256 of sorted "{seed} {md5}" lines (event_likelihoods.csv md5 only)."""
    lines = sorted(f"{seed} {md5}" for seed, md5 in seed_md5_pairs)
    return sha256_str("\n".join(lines))


# --------------------------------------------------------------------------
# CLI (Sec.8 launch block, token-for-token)
# --------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logl-iiib", type=Path, required=True)
    p.add_argument("--logl-md5-iiib", type=str, required=True)
    p.add_argument("--logl-jr1", type=Path, required=True)
    p.add_argument("--logl-md5-jr1", type=str, required=True)
    p.add_argument("--table-iiib", type=Path, required=True)
    p.add_argument("--table-sha256-iiib", type=str, required=True)
    p.add_argument("--table-jr1", type=Path, required=True)
    p.add_argument("--table-sha256-jr1", type=str, required=True)
    p.add_argument("--harness-root", type=Path, required=True)
    p.add_argument("--harness-population", type=int, required=True)
    p.add_argument("--harness-cell", type=str, required=True)
    p.add_argument("--harness-manifest-sha256", type=str, required=True)
    p.add_argument("--h-true", type=float, required=True)
    p.add_argument("--decile", type=float, required=True)
    p.add_argument("--stencil", type=float, nargs=3, required=True)
    p.add_argument("--null-draws", type=int, required=True)
    p.add_argument("--null-seed", type=int, required=True)
    p.add_argument("--share-own", type=float, required=True)
    p.add_argument("--share-diffuse", type=float, required=True)
    p.add_argument("--rho-hi", type=float, required=True)
    p.add_argument("--rho-lo", type=float, required=True)
    p.add_argument("--z-gate", type=float, required=True)
    p.add_argument("--se-unpowered", type=float, required=True)
    p.add_argument("--nonadditivity-max", type=float, default=0.6)  # DESIGN_GATE finding B
    p.add_argument(
        "--g1d-tol", type=float, default=1e-6
    )  # PIN CORRECTION 4: G-1d absolute band on |den_log_term - ln D_tilde_phi|
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--synth", type=Path, default=None, help="run the SYNTH fixture instead of real inputs"
    )
    return p


# --------------------------------------------------------------------------
# Pre-flight / pins (hard INSTRUMENT-DEFECT gate)
# --------------------------------------------------------------------------


def preflight(args: argparse.Namespace) -> None:
    """Every input file named in the launch block must exist on disk."""
    required = [args.logl_iiib, args.logl_jr1, args.table_iiib, args.table_jr1, args.harness_root]
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise InstrumentDefect(f"missing registered input file(s): {missing}")


def verify_file_pins(args: argparse.Namespace) -> None:
    checks = [
        ("logl-iiib md5", md5_file(args.logl_iiib), args.logl_md5_iiib),
        ("logl-jr1 md5", md5_file(args.logl_jr1), args.logl_md5_jr1),
        ("table-iiib sha256", sha256_file(args.table_iiib), args.table_sha256_iiib),
        ("table-jr1 sha256", sha256_file(args.table_jr1), args.table_sha256_jr1),
    ]
    for name, actual, expected in checks:
        if actual != expected:
            raise InstrumentDefect(f"{name} mismatch: expected {expected}, got {actual}")
        print(f"[pin OK] {name}: {actual}")


# --------------------------------------------------------------------------
# Population construction (Sec.1)
# --------------------------------------------------------------------------


@dataclass
class Populations:
    venue: str
    n_total: int
    P_dark: np.ndarray
    K: np.ndarray
    K_dark: np.ndarray
    K_hosted: np.ndarray
    R: np.ndarray


def load_covariate_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.set_index("event_idx")


def construct_populations(table: pd.DataFrame, venue: str, decile: float = DECILE) -> Populations:
    """Sec.1 population rule, reusing ``offset_subset_reads.py``'s rank convention.

    K is the top decile by ``C4_z_gw.rank(method="first")``; P_dark is the
    zero-candidate class (``C7_log10_n_cand_1d == 0.0``); R is the lower
    half by the same rank rule of ``P_dark \\ K``.
    """
    n_total = len(table)
    dark_mask = table["C7_log10_n_cand_1d"].to_numpy() == 0.0
    p_dark = table.index[dark_mask].to_numpy()

    n_tail = round(decile * n_total)
    ranked = table["C4_z_gw"].rank(method="first")
    k_set = table.index[ranked > (n_total - n_tail)].to_numpy()

    p_dark_set = pd.Index(p_dark)
    k_index = pd.Index(k_set)
    k_dark = p_dark_set.intersection(k_index).to_numpy()
    k_hosted = k_index.difference(p_dark_set).to_numpy()

    rest = p_dark_set.difference(k_index)
    n_rest = len(rest)
    ranked_rest = table.loc[rest, "C4_z_gw"].rank(method="first")
    r_set = rest[(ranked_rest <= (n_rest // 2)).to_numpy()].to_numpy()

    return Populations(
        venue=venue,
        n_total=n_total,
        P_dark=np.sort(p_dark.astype(np.int64)),
        K=np.sort(k_set.astype(np.int64)),
        K_dark=np.sort(k_dark.astype(np.int64)),
        K_hosted=np.sort(k_hosted.astype(np.int64)),
        R=np.sort(r_set.astype(np.int64)),
    )


def verify_population_pins(pops: Populations, pin_key: str) -> None:
    """g-byteid gate: population membership must reproduce the draft's pins exactly."""
    for name, arr in (
        ("P_dark", pops.P_dark),
        ("K", pops.K),
        ("K_dark", pops.K_dark),
        ("R", pops.R),
    ):
        expected_n = POPULATION_N[(pin_key, name)]
        if len(arr) != expected_n:
            raise InstrumentDefect(f"{pin_key} {name}: n={len(arr)} != pinned n={expected_n}")
        expected_hash = POPULATION_SHA256[(pin_key, name)]
        actual_hash = population_hash(arr)
        if actual_hash != expected_hash:
            raise InstrumentDefect(
                f"{pin_key} {name}: population sha256 mismatch -- expected {expected_hash}, got {actual_hash}"
            )
        print(f"[pop pin OK] {pin_key} {name}: n={expected_n} sha256={actual_hash}")
    expected_k_hosted_n = POPULATION_N[(pin_key, "K_hosted")]
    if len(pops.K_hosted) != expected_k_hosted_n:
        raise InstrumentDefect(
            f"{pin_key} K_hosted: n={len(pops.K_hosted)} != pinned n={expected_k_hosted_n}"
        )
    # G-3(a) set-identity check.
    if (
        not np.array_equal(np.union1d(pops.K_dark, pops.K_hosted), pops.K)
        or len(np.intersect1d(pops.K_dark, pops.K_hosted)) != 0
    ):
        raise InstrumentDefect(f"{pin_key}: K_dark/K_hosted do not partition K")


def verify_g3a_set_identity(table: pd.DataFrame) -> None:
    """G-3(a): C7==0 identical to C2==False and to C3c_censored, all rows."""
    c7_zero = table["C7_log10_n_cand_1d"].to_numpy() == 0.0
    c2_false = ~table["C2_hosted_exact"].to_numpy().astype(bool)
    c3c = table["C3c_censored"].to_numpy().astype(bool)
    if not (np.array_equal(c7_zero, c2_false) and np.array_equal(c7_zero, c3c)):
        raise InstrumentDefect("G-3a: C7==0 / C2==False / C3c_censored set-identity failed")


# --------------------------------------------------------------------------
# Harness discovery + population construction (Sec.1, per universe)
# --------------------------------------------------------------------------


@dataclass
class HarnessUniverse:
    seed: int
    diag_path: Path
    crb_path: Path
    logl_md5: str
    resolved_flags: dict[str, Any]
    n_scored: int


def discover_harness_universes(
    harness_root: Path, population: int, cell: str
) -> list[HarnessUniverse]:
    """Select universes by checkpoint ``n_draw_requested == population`` (Sec.1)."""
    universes: list[HarnessUniverse] = []
    for seed in range(901000, 901067):
        checkpoint = harness_root / f"universe_seed{seed}_{cell}.json"
        seed_dir = harness_root / f"seed{seed}_{cell}" / "simulations"
        diag_path = seed_dir / "diagnostics" / "event_likelihoods.csv"
        crb_path = seed_dir / "prepared_cramer_rao_bounds.csv"
        if not checkpoint.exists():
            continue
        with checkpoint.open() as f:
            ckpt = json.load(f)
        if ckpt.get("universe", {}).get("n_draw_requested") != population:
            continue
        if not diag_path.exists() or not crb_path.exists():
            raise InstrumentDefect(f"harness seed {seed}: checkpoint selected but files missing")
        resolved = ckpt.get("resolved_flags", {})
        # NOT ckpt["universe"]["n_scored"] (that field is n_draw_requested's
        # counterpart, ==200 for every universe here, not the number of
        # events that actually got a scored row) -- the number of ACTUALLY
        # scored events is the CSV's own unique event_idx count (173-192,
        # matching posterior.no_bh.n_events_scored in the checkpoint).
        n_scored = int(pd.read_csv(diag_path, usecols=["event_idx"])["event_idx"].nunique())
        universes.append(
            HarnessUniverse(
                seed=seed,
                diag_path=diag_path,
                crb_path=crb_path,
                logl_md5=md5_file(diag_path),
                resolved_flags=resolved,
                n_scored=n_scored,
            )
        )
    if len(universes) != 67:
        raise InstrumentDefect(
            f"expected 67 harness universes with n_draw_requested={population}, got {len(universes)}"
        )
    return universes


def verify_harness_manifest(universes: list[HarnessUniverse], expected_sha256: str) -> None:
    actual = harness_manifest_hash([(u.seed, u.logl_md5) for u in universes])
    if actual != expected_sha256:
        raise InstrumentDefect(
            f"harness manifest sha256 mismatch: expected {expected_sha256}, got {actual}"
        )
    print(f"[pin OK] harness manifest sha256: {actual} (67 universes)")


def verify_g3d_resolved_flags(universes: list[HarnessUniverse]) -> None:
    """G-3(d): the resolved-flags equality is re-asserted from the checkpoints (67/67).

    DESIGN_GATE_computability.md finding D: do NOT diff against production's
    raw ``cli_args`` (2 of the 13 tokens have no literal CLI key); instead
    assert internal agreement across all 67 checkpoints, which is what
    "13 tokens, 67/67" means here.
    """
    first = universes[0].resolved_flags
    if len(first) != 13:
        raise InstrumentDefect(f"harness resolved_flags: expected 13 tokens, got {len(first)}")
    for u in universes[1:]:
        if u.resolved_flags != first:
            raise InstrumentDefect(f"harness resolved_flags mismatch at seed {u.seed}")
    print("[gate OK] G-3d: 13 resolved_flags tokens identical, 67/67 universes")


# --------------------------------------------------------------------------
# Term profiles (Sec.2.1) + G-1 closure gate (Sec.6)
# --------------------------------------------------------------------------


def load_term_columns(csv_path: Path, event_filter: np.ndarray | None = None) -> pd.DataFrame:
    """Load the full-precision columns needed for the term identity, pivoted long."""
    usecols = ["event_idx", "h", *FULL_PRECISION_COLUMNS, *SEVEN_SF_COLUMNS]
    # float_precision="round_trip": DESIGN_GATE_computability.md finding C --
    # default pandas float parsing understates the disclosed 1e-15-level
    # closure residuals (inert for the 1e-9 gate, but the exact route).
    df = pd.read_csv(csv_path, usecols=usecols, float_precision="round_trip")
    if event_filter is not None:
        df = df[df["event_idx"].isin(event_filter)]
    return df


def gate_g1_closure(df: pd.DataFrame, label: str, g1d_tol: float = 1e-6) -> dict[str, float]:
    """G-1(a)-(e), on every row passed in. Raises InstrumentDefect on any miss.

    ``g1d_tol`` (default 1e-6, PIN CORRECTION 4) is the absolute band on
    ``|den_log_term - ln D_tilde_phi|``; the 7-s.f. storage precision of the
    ``D_tilde_phi``/``g_frac`` display columns cannot pass a tighter band by
    construction (REGISTRATION_DRAFT.md PIN CORRECTION 4).
    """
    n_rows = len(df)
    if n_rows == 0:
        raise InstrumentDefect(f"G-1 ({label}): empty frame")

    # (a) bit-exact zero.
    zero_no_bh = (df["L_cat_no_bh"].to_numpy() == 0.0).all()
    zero_with_bh = (df["L_cat_with_bh"].to_numpy() == 0.0).all()
    if not (zero_no_bh and zero_with_bh):
        raise InstrumentDefect(f"G-1a ({label}): L_cat_* not bit-exact zero on all rows")

    ln_B = np.log(df["B_num"].to_numpy())
    ln_Bw = np.log(df["B_num_wbh"].to_numpy())
    den = df["den_log_term"].to_numpy()
    ln_comb_no_bh = np.log(df["combined_no_bh"].to_numpy())
    ln_comb_with_bh = np.log(df["combined_with_bh"].to_numpy())

    # (b) 1e-9 identity band.
    resid_with_bh = np.abs(ln_comb_with_bh - (ln_Bw - den))
    resid_no_bh = np.abs(ln_comb_no_bh - (ln_B - den))
    max_resid = float(max(resid_with_bh.max(), resid_no_bh.max()))
    if max_resid > 1e-9:
        raise InstrumentDefect(f"G-1b ({label}): closure residual {max_resid:.3e} > 1e-9")

    # (d) 7-s.f. consistency.
    g_frac = df["g_frac"].to_numpy()
    d_tilde = df["D_tilde_phi"].to_numpy()
    rel_g = np.abs(g_frac - df["B_num_wbh"].to_numpy() / df["B_num"].to_numpy()) / g_frac
    if rel_g.max() > 5e-7:
        raise InstrumentDefect(f"G-1d ({label}): g_frac relative residual {rel_g.max():.3e} > 5e-7")
    resid_den = np.abs(den - np.log(d_tilde))
    if resid_den.max() > g1d_tol:
        raise InstrumentDefect(
            f"G-1d ({label}): |den_log_term - ln D_tilde_phi| {resid_den.max():.3e} > {g1d_tol:.3e}"
        )

    # (e) D_tilde_phi / den_log_term event-independence, per node.
    for h_val, grp in df.groupby("h"):
        if grp["D_tilde_phi"].nunique() != 1 or grp["den_log_term"].nunique() != 1:
            raise InstrumentDefect(
                f"G-1e ({label}): D_tilde_phi/den_log_term not single-valued at h={h_val}"
            )

    return {"max_closure_residual": max_resid, "max_g_frac_rel_residual": float(rel_g.max())}


def compute_term_profiles(
    csv_path: Path, event_filter: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (h_grid sorted, event_idx, T_B[e,h], T_g[e,h]) for the given events.

    T_D(h) is event-common and returned separately by the caller from the
    full-sample load (it does not depend on the subset).
    """
    df = load_term_columns(csv_path, event_filter)
    h_grid = np.sort(df["h"].unique())
    piv_B = df.pivot(index="event_idx", columns="h", values="B_num").reindex(columns=h_grid)
    piv_Bw = df.pivot(index="event_idx", columns="h", values="B_num_wbh").reindex(columns=h_grid)
    event_idx = piv_B.index.to_numpy()
    T_B = np.log(piv_B.to_numpy(dtype=np.float64))
    T_g = np.log(piv_Bw.to_numpy(dtype=np.float64)) - T_B
    return h_grid, event_idx, T_B, T_g


def compute_T_D(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """T_D(h) = -den_log_term, event-common (Sec.2.1); returns (h_grid, T_D)."""
    df = pd.read_csv(
        csv_path, usecols=["event_idx", "h", "den_log_term"], float_precision="round_trip"
    )
    h_grid = np.sort(df["h"].unique())
    per_node = df.groupby("h")["den_log_term"].nunique()
    if (per_node != 1).any():
        raise InstrumentDefect("T_D: den_log_term not single-valued at some node")
    den = df.groupby("h")["den_log_term"].first().reindex(h_grid).to_numpy()
    return h_grid, -den


# --------------------------------------------------------------------------
# Term-freeze counterfactual (Sec.2.3)
# --------------------------------------------------------------------------


def center_profile(t: np.ndarray, h_grid: np.ndarray, h_true: float) -> np.ndarray:
    """t_hat_e(h) = t_e(h) - t_e(h_true) (Sec.2.1)."""
    idx = int(np.argmin(np.abs(h_grid - h_true)))
    return np.asarray(t - t[:, idx : idx + 1])


def reference_profile(t_hat_R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """t_bar(h) = median_{e in R} t_hat_e(h); mean reported alongside (Sec.2.2)."""
    return np.median(t_hat_R, axis=0), np.mean(t_hat_R, axis=0)


def term_freeze_lambda(
    logpost_full: np.ndarray,
    t_hat_target: np.ndarray,
    t_bar: np.ndarray,
) -> np.ndarray:
    """Lambda_t(h) = Lambda_full(h) - sum_{e in K_dark} t_hat_e(h) + |K_dark| * t_bar(h)."""
    n_target = t_hat_target.shape[0]
    return np.asarray(logpost_full - t_hat_target.sum(axis=0) + n_target * t_bar)


def mean_h_of(logpost: np.ndarray, h_grid: np.ndarray, weights: np.ndarray) -> float:
    return float(_moments(logpost[None, :], h_grid, weights)[0][0])


def delta_t(
    logpost_full: np.ndarray, lambda_t: np.ndarray, h_grid: np.ndarray, weights: np.ndarray
) -> float:
    return mean_h_of(lambda_t, h_grid, weights) - mean_h_of(logpost_full, h_grid, weights)


@dataclass
class TermFreezeResult:
    delta_terms: dict[str, float]
    delta_F: float
    delta_D: float
    r_nonadditivity: float
    r_over_abs_delta_F: float
    shares: dict[str, float]


def run_term_freeze(
    logpost_full: np.ndarray,
    h_grid: np.ndarray,
    weights: np.ndarray,
    terms_target: dict[str, np.ndarray],  # term name -> t_hat[e,h] for K_dark
    terms_bar: dict[str, np.ndarray],  # term name -> t_bar(h)
    separable_terms: tuple[str, ...],
) -> TermFreezeResult:
    delta_terms: dict[str, float] = {}
    for name in separable_terms:
        lam = term_freeze_lambda(logpost_full, terms_target[name], terms_bar[name])
        delta_terms[name] = delta_t(logpost_full, lam, h_grid, weights)

    # All-terms freeze: replace every separable term's t_hat by t_bar simultaneously.
    n_target = next(iter(terms_target.values())).shape[0]
    lam_F = logpost_full.copy()
    for name in separable_terms:
        lam_F = lam_F - terms_target[name].sum(axis=0) + n_target * terms_bar[name]
    delta_F = delta_t(logpost_full, lam_F, h_grid, weights)

    r = delta_F - sum(delta_terms.values())
    r_over = abs(r) / abs(delta_F) if delta_F != 0 else float("inf")
    shares = {
        name: (delta_terms[name] / delta_F if delta_F != 0 else float("nan"))
        for name in separable_terms
    }

    return TermFreezeResult(
        delta_terms=delta_terms,
        delta_F=delta_F,
        delta_D=0.0,  # identity, Sec.2.3 / G-1c
        r_nonadditivity=r,
        r_over_abs_delta_F=r_over,
        shares=shares,
    )


def null_draw_ci99(
    logpost_full: np.ndarray,
    h_grid: np.ndarray,
    weights: np.ndarray,
    terms_pool: dict[str, np.ndarray],  # term -> t_hat[e,h] for P_dark \ K (the draw pool)
    terms_bar: dict[str, np.ndarray],
    separable_terms: tuple[str, ...],
    n_target: int,
    n_draws: int,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    """1000 draws of |K_dark| events from P_dark \\ K (Sec.2.5). Returns (lo, hi, draws)."""
    rng = np.random.default_rng(seed)
    n_pool = next(iter(terms_pool.values())).shape[0]
    draws = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        sel = rng.choice(n_pool, size=n_target, replace=False)
        lam_F = logpost_full.copy()
        for name in separable_terms:
            lam_F = lam_F - terms_pool[name][sel].sum(axis=0) + n_target * terms_bar[name]
        draws[i] = delta_t(logpost_full, lam_F, h_grid, weights)
    draws.sort()
    lo = float(np.percentile(draws, 0.5))
    hi = float(np.percentile(draws, 99.5))
    return lo, hi, draws


# --------------------------------------------------------------------------
# Score excess (Sec.2.4)
# --------------------------------------------------------------------------


def stencil_slope(
    t: np.ndarray, h_grid: np.ndarray, stencil: tuple[float, float, float]
) -> np.ndarray:
    """t'_e = [t_e(h_hi) - t_e(h_lo)] / (h_hi - h_lo), stencil = (lo, mid, hi)."""
    lo, _mid, hi = stencil
    i_lo = int(np.argmin(np.abs(h_grid - lo)))
    i_hi = int(np.argmin(np.abs(h_grid - hi)))
    step = h_grid[i_hi] - h_grid[i_lo]
    return np.asarray((t[:, i_hi] - t[:, i_lo]) / step)


def welch_se(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size))


@dataclass
class ScoreExcess:
    S: float
    se: float


def score_excess(t_prime_target: np.ndarray, t_prime_ref: np.ndarray) -> ScoreExcess:
    s = float(t_prime_target.mean() - t_prime_ref.mean())
    se = welch_se(t_prime_target, t_prime_ref)
    return ScoreExcess(S=s, se=se)


# --------------------------------------------------------------------------
# Harness pooling (Sec.4.3) with delete-one-universe jackknife SE
# --------------------------------------------------------------------------


@dataclass
class HarnessPoolResult:
    S_pooled: dict[str, float]
    se_jackknife: dict[str, float]
    Z_harn: dict[str, float]
    n_universes: int


def harness_pool_score(
    per_universe_target: dict[int, np.ndarray],  # seed -> t'_e array over K_dark,u for this term
    per_universe_ref: dict[int, np.ndarray],  # seed -> t'_e array over R_u for this term
) -> tuple[float, float]:
    """Event-weighted pooled S over all universes, and its delete-one-universe jackknife SE."""
    seeds = sorted(per_universe_target.keys())
    all_target = np.concatenate([per_universe_target[s] for s in seeds]) if seeds else np.array([])
    all_ref = np.concatenate([per_universe_ref[s] for s in seeds]) if seeds else np.array([])
    if all_target.size == 0 or all_ref.size == 0:
        return float("nan"), float("nan")
    s_pooled = float(all_target.mean() - all_ref.mean())

    n = len(seeds)
    if n < 2:
        return s_pooled, float("nan")
    jack_vals = np.empty(n, dtype=np.float64)
    for i, held_out in enumerate(seeds):
        t_parts = [per_universe_target[s] for s in seeds if s != held_out]
        r_parts = [per_universe_ref[s] for s in seeds if s != held_out]
        t_cat = np.concatenate(t_parts) if t_parts else np.array([])
        r_cat = np.concatenate(r_parts) if r_parts else np.array([])
        jack_vals[i] = t_cat.mean() - r_cat.mean() if t_cat.size and r_cat.size else np.nan
    jack_vals = jack_vals[~np.isnan(jack_vals)]
    m = len(jack_vals)
    if m < 2:
        return s_pooled, float("nan")
    jack_mean = jack_vals.mean()
    se = float(np.sqrt((m - 1) / m * np.sum((jack_vals - jack_mean) ** 2)))
    return s_pooled, se


# --------------------------------------------------------------------------
# Bands / three-valued dispositions (Sec.5)
# --------------------------------------------------------------------------


def production_ownership_disposition(
    delta_F: float,
    null_lo: float,
    null_hi: float,
    shares: dict[str, float],
    r_over_abs_delta_F: float,
    share_own: float,
    share_diffuse: float,
    nonadditivity_max: float = 0.6,
) -> str:
    if null_lo <= delta_F <= null_hi:
        return "Z-DIFFERENTIAL-NULL"
    ordered = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
    top_name, top_share = ordered[0]
    # Finding D: two named INTERMEDIATE carve-outs (REGISTRATION_DRAFT.md Sec.5)
    # take precedence over the literal TERM-OWNS test below, even when the
    # literal test's own arithmetic would otherwise pass:
    #   (i) both the top two shares are >= share_own, with r < 0
    #       (r's sign, in the r/Delta_F convention the draft's own worked
    #       example uses: r/Delta_F = 1 - sum(shares));
    #   (ii) two-or-more sign-opposed terms with |s_t| > 1 each.
    second_share = ordered[1][1] if len(ordered) > 1 else None
    r_over_delta_F_signed = 1.0 - sum(shares.values())
    both_ge_share_own_r_negative = (
        second_share is not None
        and top_share >= share_own
        and second_share >= share_own
        and r_over_delta_F_signed < 0
    )
    sign_opposed_gt1 = (
        len(shares) >= 2
        and all(abs(s) > 1 for s in shares.values())
        and min(shares.values()) < 0 < max(shares.values())
    )
    if both_ge_share_own_r_negative or sign_opposed_gt1:
        return "INTERMEDIATE"
    if top_share >= share_own and r_over_abs_delta_F <= nonadditivity_max:
        return f"TERM-OWNS({top_name})"
    if all(abs(s) < share_diffuse for s in shares.values()):
        return "DIFFUSE-IN-TERMS"
    return "INTERMEDIATE"


def _owning_term(disposition: str) -> str | None:
    """`"TERM-OWNS(B)"` -> `"B"`; anything else -> None."""
    prefix, suffix = "TERM-OWNS(", ")"
    if disposition.startswith(prefix) and disposition.endswith(suffix):
        return disposition[len(prefix) : -len(suffix)]
    return None


@dataclass
class ReplicateRuleResult:
    booked_disposition: str
    downgraded: bool
    reasons: list[str]


def apply_replicate_rule(
    families: dict[tuple[str, str], ProductionFamilyResult],
) -> ReplicateRuleResult:
    """Sec.5 Replicate rule (BUILD_RECORD.md FIX 3 / DESIGN_GATE_formula_rev2.md
    Finding J): a cross-family post-step, run AFTER all four production
    families (iiib/jr1 x 2D/1D) are computed, that can downgrade the primary
    (iiib, 2D) family's *booked* disposition to INTERMEDIATE. Each family's
    own raw ``disposition`` field (computed independently in
    ``run_production_family``) is left untouched; only the booked value
    returned here reflects the replicate check.

    Registered condition (REGISTRATION_DRAFT.md Sec.5): "TERM-OWNS(t) must
    hold with the same t in joint_r1 2D (the other 2D family); the 1D
    families must show Delta_B^1D of the same sign as Delta_B^2D ... A miss
    -> INTERMEDIATE."
    """
    iiib_2d = families[("iiib", "combined_with_bh")]
    jr1_2d = families[("jr1", "combined_with_bh")]
    iiib_1d = families[("iiib", "combined_no_bh")]
    jr1_1d = families[("jr1", "combined_no_bh")]

    reasons: list[str] = []

    t_iiib = _owning_term(iiib_2d.disposition)
    t_jr1 = _owning_term(jr1_2d.disposition)
    if t_iiib is None:
        # Vacuous: the primary family is not itself TERM-OWNS, so the
        # replicate rule (which is stated in terms of "the same t") has
        # nothing to check against. Booked = raw, no downgrade.
        return ReplicateRuleResult(
            booked_disposition=iiib_2d.disposition, downgraded=False, reasons=reasons
        )

    same_t_replicate = t_iiib == t_jr1
    if not same_t_replicate:
        reasons.append(f"joint_r1 2D owning term ({t_jr1}) != iiib 2D owning term ({t_iiib})")

    def _same_sign(a: float | None, b: float | None) -> bool:
        return a is not None and b is not None and np.sign(a) == np.sign(b)

    delta_B_iiib_2d = iiib_2d.term_freeze.delta_terms.get("B")
    delta_B_iiib_1d = iiib_1d.term_freeze.delta_terms.get("B")
    delta_B_jr1_2d = jr1_2d.term_freeze.delta_terms.get("B")
    delta_B_jr1_1d = jr1_1d.term_freeze.delta_terms.get("B")

    sign_ok_iiib = _same_sign(delta_B_iiib_2d, delta_B_iiib_1d)
    if not sign_ok_iiib:
        reasons.append(
            f"iiib 1D Delta_B ({delta_B_iiib_1d}) sign != iiib 2D Delta_B ({delta_B_iiib_2d}) sign"
        )
    sign_ok_jr1 = _same_sign(delta_B_jr1_2d, delta_B_jr1_1d)
    if not sign_ok_jr1:
        reasons.append(
            f"jr1 1D Delta_B ({delta_B_jr1_1d}) sign != jr1 2D Delta_B ({delta_B_jr1_2d}) sign"
        )

    replicate_ok = same_t_replicate and sign_ok_iiib and sign_ok_jr1
    if replicate_ok:
        return ReplicateRuleResult(
            booked_disposition=iiib_2d.disposition, downgraded=False, reasons=reasons
        )
    return ReplicateRuleResult(booked_disposition="INTERMEDIATE", downgraded=True, reasons=reasons)


def assert_g2iii_no_physics_floor_exclusion(n_excluded: int, label: str) -> None:
    """G-2(iii): the physics-floor exclusion count from ``_load_matrix`` must be 0.

    Called at every ``_load_matrix`` call site, BEFORE any ``event_idx``-keyed
    lookup runs on the (possibly row-dropped) returned arrays -- Finding C:
    a nonzero count means ``_load_matrix`` silently dropped an excluded
    event's row, which would otherwise surface as an uncaught ``KeyError``
    downstream rather than this clean INSTRUMENT-DEFECT.
    """
    if n_excluded != 0:
        raise InstrumentDefect(
            f"G-2(iii) ({label}): physics-floor excluded {n_excluded} event(s) "
            "(registered anchor is 0)"
        )


def assert_g2i_mean_h_anchor(mean_h_full: float, venue: str, channel_label: str) -> None:
    """G-2(i): full-sample mean_h byte-id anchor, 1e-9."""
    anchor = G2_MEAN_H_FULL[(venue, channel_label)]
    if abs(mean_h_full - anchor) > G2_MEAN_H_TOL:
        raise InstrumentDefect(
            f"G-2(i) ({venue}/{channel_label}): mean_h_full={mean_h_full:.10f} != "
            f"anchor {anchor:.10f} (tol {G2_MEAN_H_TOL})"
        )


def assert_g2ii_delta_k_anchor(delta_K_leaveout: float, venue: str) -> None:
    """G-2(ii): iiib 2D K-leave-out byte-id anchor only (END_VERIFICATION BATCH 2).

    BUILD_RECORD.md FIX 3 / DESIGN_GATE_formula_rev2.md Finding H: the
    registered anchor (+0.086106, REGISTRATION_DRAFT.md Sec.1/Sec.6) is the
    leave-out of the FULL top-z-decile set K (159 events in iiib), not the
    144-event K_dark subset -- callers MUST pass the K-masked leave-out here,
    never ``delta_K_dark_leaveout`` (that is a separate, reported-only
    Sec.2.3 concordance object, never anchor-gated).
    """
    if venue != "iiib":
        return
    if abs(delta_K_leaveout - G2_DELTA_K_IIIB_2D) > G2_DELTA_K_TOL:
        raise InstrumentDefect(
            f"G-2(ii) (iiib/2D): delta_K_leaveout={delta_K_leaveout:.6f} != "
            f"anchor {G2_DELTA_K_IIIB_2D:.6f} (tol {G2_DELTA_K_TOL})"
        )


def leaveout_delta_mean_h(
    logpost_full: np.ndarray,
    logL: np.ndarray,
    event_idx_full: np.ndarray,
    h_grid: np.ndarray,
    weights: np.ndarray,
    mean_h_full: float,
    events: np.ndarray,
) -> float:
    """Plain leave-out: mean_h(Lambda_full - sum_{e in `events`} logL_e) - mean_h_full.

    Shared by the K-anchor leave-out (G-2(ii), 159 events), the K_dark
    reported-only leave-out (Sec.2.3, 144 events) and the K_hosted
    reported-only leave-out (Sec.4.4, 15/48 events) -- same formula, three
    different event sets (BUILD_RECORD.md FIX 3).
    """
    idx_pos = {int(e): i for i, e in enumerate(event_idx_full)}
    mask = np.array([idx_pos[int(e)] for e in events])
    logpost_remove = logpost_full - logL[mask].sum(axis=0)
    return mean_h_of(logpost_remove, h_grid, weights) - mean_h_full


def harness_outcome_disposition(
    z_harn: float,
    rho_s: float,
    s_t_harn_owning: float | None,
    se_harn: float,
    prod_delta_F: float,
    z_gate: float,
    rho_hi: float,
    rho_lo: float,
    se_unpowered: float,
    prod_null_lo: float,
    prod_null_hi: float,
) -> str:
    if se_harn > se_unpowered:
        return "UNPOWERED-CONTROL"
    if abs(z_harn) <= z_gate and prod_null_lo <= prod_delta_F <= prod_null_hi:
        return "FLOOR-CONSISTENT"
    if abs(z_harn) <= z_gate:
        return "PRODUCTION-ONLY"
    if rho_s >= rho_hi and s_t_harn_owning is not None and s_t_harn_owning >= 0.5:
        return "ESTIMATOR-INTERNAL candidate"
    if rho_s <= rho_lo:
        return "PRODUCTION-ONLY"
    return "INTERMEDIATE"


# --------------------------------------------------------------------------
# g-censoring (rail disclosure, Sec.6)
# --------------------------------------------------------------------------


def is_railed(map_h_value: float, h_grid: np.ndarray) -> bool:
    return bool(np.isclose(map_h_value, h_grid.min()) or np.isclose(map_h_value, h_grid.max()))


def map_h_of(logpost: np.ndarray, h_grid: np.ndarray) -> float:
    return float(h_grid[int(np.argmax(logpost))])


# --------------------------------------------------------------------------
# SYNTH fixture (Sec.3) -- <=10 rows, exercises every disposition + gates
# --------------------------------------------------------------------------


def make_synth_fixture() -> dict[str, Any]:
    """6 events x 5 nodes, hand-verifiable: one term (T_B) carries ALL the tilt.

    Construction: T_g and T_D are identically zero-sloped (flat in h) for
    every event, so the whole freeze effect lives in T_B alone -> s_B = 1,
    s_g = 0, r = 0 to float precision (additivity is exact when only one
    term is non-flat and the freeze machinery is linear in the frozen
    term's contribution). Event 3 carries its own, separate tilt (a
    "K_hosted"-style event: in K, but not in K_dark) -- used by the
    Finding H K-vs-K_dark leave-out regression test (BUILD_RECORD.md FIX 3);
    it plays no part in the s_B/s_g/r assertions below (those only ever look
    at K_dark_idx={4,5} and R_idx={0,1,2}).
    """
    h_grid = np.array([0.720, 0.725, 0.730, 0.735, 0.740])
    n_events = 6
    rng = np.random.default_rng(0)
    # T_B: linear-in-h per event with an event-specific slope (the "tilt").
    base_B = rng.normal(0.0, 1.0, size=n_events)
    # last two events ("K_dark") carry the registered tilt; event 3 ("K_hosted")
    # carries its own, independent tilt -- present in K, absent from K_dark.
    slope_B = np.array([0.0, 0.0, 0.0, 3.0, 5.0, 5.0])
    T_B = base_B[:, None] + slope_B[:, None] * (h_grid[None, :] - 0.73)
    # T_g, T_D: flat (no h-dependence at all) -> zero contribution to any freeze.
    T_g = np.tile(rng.normal(0.0, 0.1, size=n_events)[:, None], (1, len(h_grid)))
    T_D = np.zeros(len(h_grid))

    logpost_full = T_B.sum(axis=0) + T_g.sum(axis=0) + T_D * n_events

    return {
        "h_grid": h_grid,
        "T_B": T_B,
        "T_g": T_g,
        "T_D": T_D,
        "logpost_full": logpost_full,
        "K_dark_idx": np.array([4, 5]),
        "K_hosted_idx": np.array([3]),
        "K_idx": np.array([3, 4, 5]),
        "R_idx": np.array([0, 1, 2]),
    }


def run_synth_check() -> None:
    fx = make_synth_fixture()
    h_grid = fx["h_grid"]
    weights = np.gradient(h_grid)
    K_dark_idx = fx["K_dark_idx"]
    R_idx = fx["R_idx"]

    T_B_hat = center_profile(fx["T_B"], h_grid, TRUTH)
    T_g_hat = center_profile(fx["T_g"], h_grid, TRUTH)

    t_bar_B, _ = reference_profile(T_B_hat[R_idx])
    t_bar_g, _ = reference_profile(T_g_hat[R_idx])

    result = run_term_freeze(
        logpost_full=fx["logpost_full"],
        h_grid=h_grid,
        weights=weights,
        terms_target={"B": T_B_hat[K_dark_idx], "g": T_g_hat[K_dark_idx]},
        terms_bar={"B": t_bar_B, "g": t_bar_g},
        separable_terms=("B", "g"),
    )

    assert abs(result.shares["B"] - 1.0) < 1e-9, f"SYNTH: s_B={result.shares['B']} != 1"
    assert abs(result.shares["g"]) < 1e-6, f"SYNTH: s_g={result.shares['g']} != 0"
    assert abs(result.r_nonadditivity) < 1e-9, f"SYNTH: r={result.r_nonadditivity} != 0"

    disp = production_ownership_disposition(
        delta_F=result.delta_F,
        null_lo=-1e-6,
        null_hi=1e-6,
        shares=result.shares,
        r_over_abs_delta_F=result.r_over_abs_delta_F,
        share_own=0.5,
        share_diffuse=0.2,
    )
    assert disp == "TERM-OWNS(B)", f"SYNTH: disposition={disp}"

    # Exercise DIFFUSE-IN-TERMS and Z-DIFFERENTIAL-NULL on hand-built shares.
    diffuse = production_ownership_disposition(
        delta_F=1.0,
        null_lo=-1e-6,
        null_hi=1e-6,
        shares={"B": 0.1, "g": 0.1},
        r_over_abs_delta_F=0.9,
        share_own=0.5,
        share_diffuse=0.2,
    )
    assert diffuse == "DIFFUSE-IN-TERMS", diffuse
    z_null = production_ownership_disposition(
        delta_F=0.0,
        null_lo=-1e-6,
        null_hi=1e-6,
        shares={"B": 0.9, "g": 0.05},
        r_over_abs_delta_F=0.1,
        share_own=0.5,
        share_diffuse=0.2,
    )
    assert z_null == "Z-DIFFERENTIAL-NULL", z_null
    intermediate = production_ownership_disposition(
        delta_F=1.0,
        null_lo=-1e-6,
        null_hi=1e-6,
        shares={"B": 0.3, "g": 0.1},
        r_over_abs_delta_F=0.1,
        share_own=0.5,
        share_diffuse=0.2,
    )
    assert intermediate == "INTERMEDIATE", intermediate

    # Harness dispositions, hand-built.
    assert (
        harness_outcome_disposition(4.0, 0.6, 0.6, 0.05, 1.0, 3.0, 0.5, 0.2, 0.1, -1e-6, 1e-6)
        == "ESTIMATOR-INTERNAL candidate"
    )
    assert (
        harness_outcome_disposition(1.0, 0.6, 0.6, 0.05, 0.0, 3.0, 0.5, 0.2, 0.1, -1e-6, 1e-6)
        == "FLOOR-CONSISTENT"
    )
    assert (
        harness_outcome_disposition(1.0, 0.6, 0.6, 0.05, 1.0, 3.0, 0.5, 0.2, 0.1, -1e-6, 1e-6)
        == "PRODUCTION-ONLY"
    )
    assert (
        harness_outcome_disposition(4.0, 0.1, 0.6, 0.05, 1.0, 3.0, 0.5, 0.2, 0.1, -1e-6, 1e-6)
        == "PRODUCTION-ONLY"
    )
    assert (
        harness_outcome_disposition(4.0, 0.3, 0.6, 0.05, 1.0, 3.0, 0.5, 0.2, 0.1, -1e-6, 1e-6)
        == "INTERMEDIATE"
    )
    assert (
        harness_outcome_disposition(4.0, 0.6, 0.6, 0.2, 1.0, 3.0, 0.5, 0.2, 0.1, -1e-6, 1e-6)
        == "UNPOWERED-CONTROL"
    )

    # G-1 closure gate on a hand-built synthetic table (INSTRUMENT-DEFECT path).
    synth_df = pd.DataFrame(
        {
            "event_idx": [0, 1, 2],
            "h": [0.73, 0.73, 0.73],
            "L_cat_no_bh": [0.0, 0.0, 0.0],
            "L_cat_with_bh": [0.0, 0.0, 0.0],
            "B_num": [2.0, 3.0, 4.0],
            "B_num_wbh": [1.0, 1.5, 2.0],
            "den_log_term": [0.1, 0.1, 0.1],
            "combined_no_bh": [
                np.exp(np.log(2.0) - 0.1),
                np.exp(np.log(3.0) - 0.1),
                np.exp(np.log(4.0) - 0.1),
            ],
            "combined_with_bh": [
                np.exp(np.log(1.0) - 0.1),
                np.exp(np.log(1.5) - 0.1),
                np.exp(np.log(2.0) - 0.1),
            ],
            "g_frac": [0.5, 0.5, 0.5],
            "D_tilde_phi": [np.exp(0.1)] * 3,
            "alpha_G_phi": [1.0, 1.0, 1.0],
            "r_Malm": [1.0, 1.0, 1.0],
            "w_tilde_G": [1.0, 1.0, 1.0],
            "w_G_legacy": [1.0, 1.0, 1.0],
        }
    )
    gate_g1_closure(synth_df, "SYNTH-pass")

    bad_df = synth_df.copy()
    bad_df.loc[0, "L_cat_no_bh"] = 1e-30  # not bit-exact zero
    try:
        gate_g1_closure(bad_df, "SYNTH-fail")
        raise AssertionError("SYNTH: G-1a should have raised InstrumentDefect")
    except InstrumentDefect:
        pass

    # PIN CORRECTION 4: --g1d-tol threading. A 5e-7 residual on
    # |den_log_term - ln D_tilde_phi| (the disclosed 7-s.f.-precision scale,
    # REGISTRATION_DRAFT.md PIN CORRECTION 4: real full-iiib max 4.407e-7)
    # must pass at the registered default (1e-6) and fail at the old 1e-8 band.
    g1d_df = synth_df.copy()
    g1d_df["D_tilde_phi"] = np.exp(0.1 - 5e-7)
    gate_g1_closure(g1d_df, "SYNTH-g1d-tol-default", g1d_tol=1e-6)
    try:
        gate_g1_closure(g1d_df, "SYNTH-g1d-tol-tight", g1d_tol=1e-8)
        raise AssertionError("SYNTH: G-1d should have raised InstrumentDefect at g1d_tol=1e-8")
    except InstrumentDefect:
        pass

    # ---- Finding A: --nonadditivity-max actually reaches the disposition ----
    a_shares = {"B": 0.9, "g": 0.05}
    disp_default = production_ownership_disposition(
        delta_F=1.0,
        null_lo=-1e-6,
        null_hi=1e-6,
        shares=a_shares,
        r_over_abs_delta_F=0.7,
        share_own=0.5,
        share_diffuse=0.2,
    )  # default nonadditivity_max=0.6
    assert disp_default == "INTERMEDIATE", f"Finding A: default-band case={disp_default}"
    disp_widened = production_ownership_disposition(
        delta_F=1.0,
        null_lo=-1e-6,
        null_hi=1e-6,
        shares=a_shares,
        r_over_abs_delta_F=0.7,
        share_own=0.5,
        share_diffuse=0.2,
        nonadditivity_max=0.8,
    )
    assert disp_widened == "TERM-OWNS(B)", f"Finding A: widened-band case={disp_widened}"

    # ---- Finding B: G-2(i)/(ii) anchor gates raise InstrumentDefect on miss ----
    assert_g2i_mean_h_anchor(G2_MEAN_H_FULL[("iiib", "2D")], "iiib", "2D")  # passes
    try:
        assert_g2i_mean_h_anchor(G2_MEAN_H_FULL[("iiib", "2D")] + 1e-3, "iiib", "2D")
        raise AssertionError("Finding B: G-2(i) should have raised InstrumentDefect")
    except InstrumentDefect:
        pass
    assert_g2ii_delta_k_anchor(G2_DELTA_K_IIIB_2D, "iiib")  # passes
    try:
        assert_g2ii_delta_k_anchor(G2_DELTA_K_IIIB_2D + 1.0, "iiib")
        raise AssertionError("Finding B: G-2(ii) should have raised InstrumentDefect")
    except InstrumentDefect:
        pass
    assert_g2ii_delta_k_anchor(-999.0, "jr1")  # no-op for jr1 (iiib-2D-only anchor)

    # ---- Finding C: G-2(iii) physics-floor exclusion count must be 0 ----
    assert_g2iii_no_physics_floor_exclusion(0, "SYNTH")  # passes
    try:
        assert_g2iii_no_physics_floor_exclusion(2, "SYNTH")
        raise AssertionError("Finding C: G-2(iii) should have raised InstrumentDefect")
    except InstrumentDefect:
        pass

    # ---- Finding D: the two missing INTERMEDIATE band-table carve-outs ----
    # Reviewer counter-example 1: both shares >= 0.5, r < 0 (r/Delta_F = 1-(0.55+0.52) = -0.07).
    both_ge_half = production_ownership_disposition(
        delta_F=1.0,
        null_lo=-1e-6,
        null_hi=1e-6,
        shares={"B": 0.55, "g": 0.52},
        r_over_abs_delta_F=0.07,
        share_own=0.5,
        share_diffuse=0.2,
    )
    assert both_ge_half == "INTERMEDIATE", f"Finding D (both>=0.5, r<0): {both_ge_half}"
    # Reviewer counter-example 2: sign-opposed terms with |s| > 1 each.
    sign_opposed = production_ownership_disposition(
        delta_F=1.0,
        null_lo=-1e-6,
        null_hi=1e-6,
        shares={"B": 3.0, "g": -2.0},
        r_over_abs_delta_F=0.0,
        share_own=0.5,
        share_diffuse=0.2,
    )
    assert sign_opposed == "INTERMEDIATE", f"Finding D (sign-opposed, |s|>1): {sign_opposed}"

    # ---- FIX 3 item (1) / Finding H: K (159-event) leave-out must differ from
    # K_dark (144-event) leave-out when K_hosted (K \ K_dark) carries non-flat
    # h-dependence -- exactly the case a wrong-population G-2(ii) anchor would
    # miss. Event 3 (K_hosted, slope=3.0) is present in K_idx but absent from
    # K_dark_idx; both leave-outs share the same `leaveout_delta_mean_h` code
    # path used by `run_production_family`. ----
    event_idx_full_synth = np.arange(6)
    logL_synth = fx["T_B"] + fx["T_g"]  # per-event log-likelihood row (T_D is event-common)
    mean_h_full_synth = mean_h_of(fx["logpost_full"], h_grid, weights)
    delta_K_synth = leaveout_delta_mean_h(
        fx["logpost_full"],
        logL_synth,
        event_idx_full_synth,
        h_grid,
        weights,
        mean_h_full_synth,
        fx["K_idx"],
    )
    delta_K_dark_synth = leaveout_delta_mean_h(
        fx["logpost_full"],
        logL_synth,
        event_idx_full_synth,
        h_grid,
        weights,
        mean_h_full_synth,
        fx["K_dark_idx"],
    )
    # The gap is well above G2_DELTA_K_TOL (1e-6) -- this is exactly the
    # magnitude of false-INSTRUMENT-DEFECT/wrong-silent-pass risk Finding H
    # warned about, not float noise.
    assert abs(delta_K_synth - delta_K_dark_synth) > 1e-5, (
        f"Finding H: K-leaveout ({delta_K_synth}) should differ from K_dark-leaveout "
        f"({delta_K_dark_synth}) when K_hosted carries non-flat h-dependence"
    )

    # ---- FIX 3 item (2) / Finding I: 1D channel selects only the "B" term ----
    assert _separable_terms_for_channel("combined_with_bh") == ("B", "g")
    assert _separable_terms_for_channel("combined_no_bh") == ("B",)

    # ---- FIX 3 item (3) / Finding J: the Sec.5 Replicate rule cross-family
    # downgrade -- built on hand-constructed ProductionFamilyResult stand-ins
    # (no CSV/aggregate involved). ----
    def _mk_family(disposition: str, delta_B: float, two_term: bool) -> ProductionFamilyResult:
        delta_terms = {"B": delta_B, "g": 0.0} if two_term else {"B": delta_B}
        shares = {"B": 1.0, "g": 0.0} if two_term else {"B": 1.0}
        tf_fake = TermFreezeResult(
            delta_terms=delta_terms,
            delta_F=delta_B,
            delta_D=0.0,
            r_nonadditivity=0.0,
            r_over_abs_delta_F=0.0,
            shares=shares,
        )
        return ProductionFamilyResult(
            venue="synth",
            channel_label="2D" if two_term else "1D",
            separable_terms=("B", "g") if two_term else ("B",),
            mean_h_full=0.0,
            term_freeze=tf_fake,
            null_lo=-1e-6,
            null_hi=1e-6,
            disposition=disposition,
            score_excess={},
            map_h_full=0.73,
            map_rail_full=False,
            map_rails_freeze={},
            max_g1_closure_residual=0.0,
            delta_K_leaveout=float("nan"),
            delta_K_dark_leaveout=float("nan"),
        )

    families_pass = {
        ("iiib", "combined_with_bh"): _mk_family("TERM-OWNS(B)", 1.0, True),
        ("jr1", "combined_with_bh"): _mk_family("TERM-OWNS(B)", 1.0, True),
        ("iiib", "combined_no_bh"): _mk_family("TERM-OWNS(B)", 0.5, False),
        ("jr1", "combined_no_bh"): _mk_family("TERM-OWNS(B)", 0.5, False),
    }
    rep_pass = apply_replicate_rule(families_pass)
    assert not rep_pass.downgraded and rep_pass.booked_disposition == "TERM-OWNS(B)", rep_pass

    # Miss: jr1/2D owns a different term than iiib/2D.
    families_miss_t = dict(families_pass)
    families_miss_t[("jr1", "combined_with_bh")] = _mk_family("TERM-OWNS(g)", 1.0, True)
    rep_miss_t = apply_replicate_rule(families_miss_t)
    assert rep_miss_t.downgraded and rep_miss_t.booked_disposition == "INTERMEDIATE", rep_miss_t
    assert rep_miss_t.reasons, "Finding J: a replicate miss must record a reason"

    # Miss: iiib 1D Delta_B has the opposite sign of iiib 2D Delta_B.
    families_miss_sign = dict(families_pass)
    families_miss_sign[("iiib", "combined_no_bh")] = _mk_family("TERM-OWNS(B)", -0.5, False)
    rep_miss_sign = apply_replicate_rule(families_miss_sign)
    assert rep_miss_sign.downgraded and rep_miss_sign.booked_disposition == "INTERMEDIATE", (
        rep_miss_sign
    )

    # Vacuous: iiib/2D is not itself TERM-OWNS -- nothing to replicate-check,
    # raw disposition passes through unchanged.
    families_vacuous = dict(families_pass)
    families_vacuous[("iiib", "combined_with_bh")] = _mk_family("DIFFUSE-IN-TERMS", 1.0, True)
    rep_vacuous = apply_replicate_rule(families_vacuous)
    assert not rep_vacuous.downgraded and rep_vacuous.booked_disposition == "DIFFUSE-IN-TERMS", (
        rep_vacuous
    )

    print(
        "[SYNTH OK] closure identity, disposition rows (production 4 + harness 6), G-1 pass/fail "
        "path, Findings A-D counter-examples, Finding H K-vs-K_dark leaveout, Finding I channel "
        "term selection, Finding J replicate-rule pass/miss"
    )


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def _five_row_slice_closure(
    csv_path: Path, table: pd.DataFrame, h_true: float, g1d_tol: float = 1e-6
) -> float:
    """5-row real-slice closure check (design-gate computability category)."""
    p_dark_events = table.index[table["C7_log10_n_cand_1d"].to_numpy() == 0.0].to_numpy()
    slice_events = np.sort(p_dark_events)[:5]
    df = load_term_columns(csv_path, slice_events)
    df = df[np.isclose(df["h"], h_true)]
    stats = gate_g1_closure(df, "5-row real slice", g1d_tol=g1d_tol)
    return stats["max_closure_residual"]


# --------------------------------------------------------------------------
# Real-mode production family (Sec.2/Sec.4.1/Sec.4.2) -- run by the reader
# --------------------------------------------------------------------------


@dataclass
class ProductionFamilyResult:
    venue: str
    channel_label: str
    separable_terms: tuple[str, ...]
    mean_h_full: float
    term_freeze: TermFreezeResult
    null_lo: float
    null_hi: float
    disposition: str
    score_excess: dict[str, ScoreExcess]  # term -> S_t
    map_h_full: float
    map_rail_full: bool
    map_rails_freeze: dict[str, bool]
    max_g1_closure_residual: float
    delta_K_leaveout: float
    delta_K_dark_leaveout: float


def _term_arrays_for_events(
    csv_path: Path, h_grid: np.ndarray, h_true: float, events: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """T_B_hat, T_g_hat (centered) for exactly `events`, aligned to h_grid order."""
    _hg, event_idx, T_B, T_g = compute_term_profiles(csv_path, events)
    order = np.argsort(event_idx)
    event_idx = event_idx[order]
    T_B = T_B[order]
    T_g = T_g[order]
    if not np.array_equal(np.sort(event_idx), np.sort(events)):
        raise InstrumentDefect("term profile event set does not match the requested population")
    T_B_hat = center_profile(T_B, h_grid, h_true)
    T_g_hat = center_profile(T_g, h_grid, h_true)
    return T_B_hat, T_g_hat


def run_production_family(
    venue: str,
    channel: str,
    csv_path: Path,
    pops: Populations,
    h_true: float,
    stencil: tuple[float, float, float],
    n_null_draws: int,
    null_seed: int,
    share_own: float,
    share_diffuse: float,
    nonadditivity_max: float,
    g1d_tol: float = 1e-6,
) -> ProductionFamilyResult:
    channel_label = CHANNEL_LABEL[channel]
    separable_terms: tuple[str, ...] = _separable_terms_for_channel(channel)

    # G-1 closure over EVERY P_dark row, this venue (Sec.6).
    df_p_dark = load_term_columns(csv_path, pops.P_dark)
    g1 = gate_g1_closure(df_p_dark, f"{venue}/{channel_label} P_dark full", g1d_tol=g1d_tol)

    h_grid, event_idx_full, logL, n_excluded = _load_matrix(csv_path, channel)
    # Finding C: check BEFORE any event_idx-keyed lookup below can hit a
    # missing key from a silently row-dropped exclusion.
    assert_g2iii_no_physics_floor_exclusion(n_excluded, f"{venue}/{channel_label}")
    weights = np.gradient(h_grid)
    logpost_full = logL.sum(axis=0)
    mean_h_full = mean_h_of(logpost_full, h_grid, weights)
    map_h_full = map_h_of(logpost_full, h_grid)

    assert_g2i_mean_h_anchor(mean_h_full, venue, channel_label)

    T_B_hat_Kdark, T_g_hat_Kdark = _term_arrays_for_events(csv_path, h_grid, h_true, pops.K_dark)
    T_B_hat_R, T_g_hat_R = _term_arrays_for_events(csv_path, h_grid, h_true, pops.R)
    T_B_hat_pool, T_g_hat_pool = _term_arrays_for_events(
        csv_path, h_grid, h_true, np.setdiff1d(pops.P_dark, pops.K)
    )

    t_bar_B, _ = reference_profile(T_B_hat_R)
    t_bar_g, _ = reference_profile(T_g_hat_R)

    terms_target = {"B": T_B_hat_Kdark, "g": T_g_hat_Kdark}
    terms_bar = {"B": t_bar_B, "g": t_bar_g}

    tf = run_term_freeze(
        logpost_full=logpost_full,
        h_grid=h_grid,
        weights=weights,
        terms_target=terms_target,
        terms_bar=terms_bar,
        separable_terms=separable_terms,
    )

    null_lo, null_hi, _draws = null_draw_ci99(
        logpost_full=logpost_full,
        h_grid=h_grid,
        weights=weights,
        terms_pool={"B": T_B_hat_pool, "g": T_g_hat_pool},
        terms_bar=terms_bar,
        separable_terms=separable_terms,
        n_target=len(pops.K_dark),
        n_draws=n_null_draws,
        seed=null_seed,
    )

    disposition = production_ownership_disposition(
        delta_F=tf.delta_F,
        null_lo=null_lo,
        null_hi=null_hi,
        shares=tf.shares,
        r_over_abs_delta_F=tf.r_over_abs_delta_F,
        share_own=share_own,
        share_diffuse=share_diffuse,
        nonadditivity_max=nonadditivity_max,
    )

    term_arrays: list[tuple[str, np.ndarray, np.ndarray]] = [("B", T_B_hat_Kdark, T_B_hat_R)]
    if "g" in separable_terms:
        term_arrays.append(("g", T_g_hat_Kdark, T_g_hat_R))
    score_excess_by_term: dict[str, ScoreExcess] = {}
    for name, t_hat_target, t_hat_ref in term_arrays:
        sp_target = stencil_slope(t_hat_target, h_grid, stencil)
        sp_ref = stencil_slope(t_hat_ref, h_grid, stencil)
        score_excess_by_term[name] = score_excess(sp_target, sp_ref)

    map_rails_freeze: dict[str, bool] = {}
    for name in separable_terms:
        lam = term_freeze_lambda(logpost_full, terms_target[name], terms_bar[name])
        map_rails_freeze[name] = is_railed(map_h_of(lam, h_grid), h_grid)

    # Finding H (DESIGN_GATE_formula_rev2.md Sec.4/Sec.5): the G-2(ii) anchor
    # is the leave-out of the FULL top-z-decile K (159 events) -- compute and
    # assert THAT. delta_K_dark_leaveout (144 events, K_dark subset) is kept
    # as the separate, reported-only Sec.2.3 concordance object -- never fed
    # into the byte-id anchor assertion.
    delta_K_leaveout = float("nan")
    delta_K_dark_leaveout = float("nan")
    if channel == "combined_with_bh":
        delta_K_leaveout = leaveout_delta_mean_h(
            logpost_full, logL, event_idx_full, h_grid, weights, mean_h_full, pops.K
        )
        assert_g2ii_delta_k_anchor(delta_K_leaveout, venue)
        delta_K_dark_leaveout = leaveout_delta_mean_h(
            logpost_full, logL, event_idx_full, h_grid, weights, mean_h_full, pops.K_dark
        )

    return ProductionFamilyResult(
        venue=venue,
        channel_label=channel_label,
        separable_terms=separable_terms,
        mean_h_full=mean_h_full,
        term_freeze=tf,
        null_lo=null_lo,
        null_hi=null_hi,
        disposition=disposition,
        score_excess=score_excess_by_term,
        map_h_full=map_h_full,
        map_rail_full=is_railed(map_h_full, h_grid),
        map_rails_freeze=map_rails_freeze,
        max_g1_closure_residual=g1["max_closure_residual"],
        delta_K_leaveout=delta_K_leaveout,
        delta_K_dark_leaveout=delta_K_dark_leaveout,
    )


def run_K_hosted_leaveout(csv_path: Path, channel: str, pops: Populations) -> float:
    """Sec.4.4: K_hosted (15/48) plain leave-out, reported-only."""
    if len(pops.K_hosted) == 0:
        return float("nan")
    h_grid, event_idx_full, logL, n_excluded = _load_matrix(csv_path, channel)
    assert_g2iii_no_physics_floor_exclusion(n_excluded, f"K_hosted leave-out, {channel}")
    weights = np.gradient(h_grid)
    logpost_full = logL.sum(axis=0)
    mean_full = mean_h_of(logpost_full, h_grid, weights)
    return leaveout_delta_mean_h(
        logpost_full, logL, event_idx_full, h_grid, weights, mean_full, pops.K_hosted
    )


# --------------------------------------------------------------------------
# Real-mode harness control (Sec.4.3) -- run by the reader
# --------------------------------------------------------------------------


@dataclass
class HarnessUniverseRead:
    seed: int
    n_scored: int
    P_dark_u: np.ndarray
    K_u: np.ndarray
    K_dark_u: np.ndarray
    R_u: np.ndarray
    delta_terms_u: dict[str, float]
    delta_F_u: float
    r_u: float
    rail_full_u: bool
    stencil_slopes_K_dark: dict[str, np.ndarray]
    stencil_slopes_R: dict[str, np.ndarray]


def _harness_z(u: HarnessUniverse, h_true: float) -> pd.Series:
    crb = pd.read_csv(u.crb_path, usecols=["luminosity_distance"])
    diag = pd.read_csv(u.diag_path, usecols=["event_idx"])
    event_idx = np.sort(diag["event_idx"].unique())
    d_L = crb["luminosity_distance"].to_numpy()[
        event_idx
    ]  # C4 recipe: event_idx as CRB row position
    z = np.array([dist_to_redshift(float(d), h=h_true) for d in d_L], dtype=np.float64)
    return pd.Series(z, index=event_idx)


def construct_harness_populations(
    u: HarnessUniverse, h_true: float, decile: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (event_idx (scored), P_dark_u, K_u, K_dark_u, R_u)."""
    diag = pd.read_csv(
        u.diag_path, usecols=["event_idx", "h", "L_cat_no_bh"], float_precision="round_trip"
    )
    at_truth = diag[np.isclose(diag["h"], h_true)].set_index("event_idx")
    event_idx = np.sort(at_truth.index.to_numpy())
    p_dark_mask = at_truth.loc[event_idx, "L_cat_no_bh"].to_numpy() == 0.0
    p_dark_u = event_idx[p_dark_mask]

    z = _harness_z(u, h_true).reindex(event_idx)
    n_scored = len(event_idx)
    n_tail = round(decile * n_scored)
    ranked = z.rank(method="first")
    k_u = event_idx[(ranked > (n_scored - n_tail)).to_numpy()]

    p_dark_set = pd.Index(p_dark_u)
    k_set = pd.Index(k_u)
    k_dark_u = p_dark_set.intersection(k_set).to_numpy()
    rest = p_dark_set.difference(k_set)
    n_rest = len(rest)
    ranked_rest = z.loc[rest].rank(method="first")
    r_u = rest[(ranked_rest <= (n_rest // 2)).to_numpy()].to_numpy()

    return event_idx, np.sort(p_dark_u), np.sort(k_u), np.sort(k_dark_u), np.sort(r_u)


def run_harness_universe(
    u: HarnessUniverse,
    h_true: float,
    decile: float,
    stencil: tuple[float, float, float],
    channel: str = "combined_with_bh",
) -> HarnessUniverseRead:
    """Sec.4.3 harness control, one universe, one channel.

    BUILD_RECORD.md FIX 3 / DESIGN_GATE_formula_rev2.md Finding I:
    REGISTRATION_DRAFT.md Sec.4.3 registers the harness control for BOTH
    channels; `channel` selects which combined column is read (population
    construction -- P_dark_u/K_u/K_dark_u/R_u -- is channel-independent, so
    it is computed once regardless of `channel`). 1D (`combined_no_bh`) has
    only the T_B separable term (`_separable_terms_for_channel`).
    """
    event_idx, p_dark_u, k_u, k_dark_u, r_u = construct_harness_populations(u, h_true, decile)
    separable_terms = _separable_terms_for_channel(channel)

    h_grid, event_idx_full, logL, n_excluded = _load_matrix(u.diag_path, channel)
    assert_g2iii_no_physics_floor_exclusion(
        n_excluded, f"harness seed {u.seed}/{CHANNEL_LABEL[channel]}"
    )
    weights = np.gradient(h_grid)
    logpost_full = logL.sum(axis=0)
    map_h_full = map_h_of(logpost_full, h_grid)

    delta_terms_u: dict[str, float] = {}
    stencil_slopes_K_dark: dict[str, np.ndarray] = {}
    stencil_slopes_R: dict[str, np.ndarray] = {}

    if len(k_dark_u) == 0 or len(r_u) == 0:
        return HarnessUniverseRead(
            seed=u.seed,
            n_scored=len(event_idx),
            P_dark_u=p_dark_u,
            K_u=k_u,
            K_dark_u=k_dark_u,
            R_u=r_u,
            delta_terms_u={},
            delta_F_u=float("nan"),
            r_u=float("nan"),
            rail_full_u=is_railed(map_h_full, h_grid),
            stencil_slopes_K_dark={},
            stencil_slopes_R={},
        )

    T_B_hat_Kdark, T_g_hat_Kdark = _term_arrays_for_events(u.diag_path, h_grid, h_true, k_dark_u)
    T_B_hat_R, T_g_hat_R = _term_arrays_for_events(u.diag_path, h_grid, h_true, r_u)
    t_bar_B, _ = reference_profile(T_B_hat_R)
    t_bar_g, _ = reference_profile(T_g_hat_R)

    terms_target_all = {"B": T_B_hat_Kdark, "g": T_g_hat_Kdark}
    terms_bar_all = {"B": t_bar_B, "g": t_bar_g}
    terms_ref_all = {"B": T_B_hat_R, "g": T_g_hat_R}

    tf = run_term_freeze(
        logpost_full=logpost_full,
        h_grid=h_grid,
        weights=weights,
        terms_target={name: terms_target_all[name] for name in separable_terms},
        terms_bar={name: terms_bar_all[name] for name in separable_terms},
        separable_terms=separable_terms,
    )
    delta_terms_u = tf.delta_terms

    for name in separable_terms:
        stencil_slopes_K_dark[name] = stencil_slope(terms_target_all[name], h_grid, stencil)
        stencil_slopes_R[name] = stencil_slope(terms_ref_all[name], h_grid, stencil)

    return HarnessUniverseRead(
        seed=u.seed,
        n_scored=len(event_idx),
        P_dark_u=p_dark_u,
        K_u=k_u,
        K_dark_u=k_dark_u,
        R_u=r_u,
        delta_terms_u=delta_terms_u,
        delta_F_u=tf.delta_F,
        r_u=tf.r_nonadditivity,
        rail_full_u=is_railed(map_h_full, h_grid),
        stencil_slopes_K_dark=stencil_slopes_K_dark,
        stencil_slopes_R=stencil_slopes_R,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    if args.synth is not None:
        run_synth_check()
        return

    preflight(args)
    verify_file_pins(args)

    table_iiib = load_covariate_table(args.table_iiib)
    table_jr1 = load_covariate_table(args.table_jr1)
    verify_g3a_set_identity(table_iiib)
    verify_g3a_set_identity(table_jr1)

    pops_iiib = construct_populations(table_iiib, "iiib", args.decile)
    pops_jr1 = construct_populations(table_jr1, "jr1", args.decile)
    verify_population_pins(pops_iiib, "iiib")
    verify_population_pins(pops_jr1, "jr1")
    if not np.array_equal(pops_iiib.K, pops_jr1.K):
        raise InstrumentDefect("G-3b: K is not the same event set in both venues")

    universes = discover_harness_universes(
        args.harness_root, args.harness_population, args.harness_cell
    )
    verify_harness_manifest(universes, args.harness_manifest_sha256)
    verify_g3d_resolved_flags(universes)

    pooled_n_scored = sum(u.n_scored for u in universes)
    print(
        f"[counts] iiib: n={pops_iiib.n_total} P_dark={len(pops_iiib.P_dark)} "
        f"K={len(pops_iiib.K)} K_dark={len(pops_iiib.K_dark)} K_hosted={len(pops_iiib.K_hosted)} R={len(pops_iiib.R)}"
    )
    print(
        f"[counts] jr1:  n={pops_jr1.n_total} P_dark={len(pops_jr1.P_dark)} "
        f"K={len(pops_jr1.K)} K_dark={len(pops_jr1.K_dark)} K_hosted={len(pops_jr1.K_hosted)} R={len(pops_jr1.R)}"
    )
    print(
        f"[counts] harness: universes=67 Sigma n_scored(CSV event_idx)={pooled_n_scored} "
        f"(anchor {HARNESS_POOLED_ANCHORS['n_scored']})"
    )

    max_resid = _five_row_slice_closure(
        args.logl_iiib, table_iiib, args.h_true, g1d_tol=args.g1d_tol
    )
    print(f"[gate G-1] 5-row real-slice max closure residual: {max_resid:.3e} (band 1e-9)")
    print(f"[gate G-1d] resolved --g1d-tol: {args.g1d_tol:.3e}")

    run_synth_check()

    if args.dry_run:
        print(
            "[dry-run] gates + byte-id anchors only, no --out written, no registered aggregate computed."
        )
        return

    # ---- real mode (Sec.4-Sec.6): DISJOINT reader only -- this builder never
    # invokes main() without --dry-run or --synth (see BUILD_RECORD.md). ----
    stencil = tuple(args.stencil)  # (0.725, 0.730, 0.735)

    families: dict[tuple[str, str], ProductionFamilyResult] = {}
    for venue, csv_path, table, pops in (
        ("iiib", args.logl_iiib, table_iiib, pops_iiib),
        ("jr1", args.logl_jr1, table_jr1, pops_jr1),
    ):
        for channel in CHANNELS:
            families[(venue, channel)] = run_production_family(
                venue=venue,
                channel=channel,
                csv_path=csv_path,
                pops=pops,
                h_true=args.h_true,
                stencil=stencil,
                n_null_draws=args.null_draws,
                null_seed=args.null_seed,
                share_own=args.share_own,
                share_diffuse=args.share_diffuse,
                nonadditivity_max=args.nonadditivity_max,
                g1d_tol=args.g1d_tol,
            )

    delta_K_hosted = {
        venue: run_K_hosted_leaveout(csv_path, "combined_with_bh", pops)
        for venue, csv_path, pops in (
            ("iiib", args.logl_iiib, pops_iiib),
            ("jr1", args.logl_jr1, pops_jr1),
        )
    }

    # ---- Finding J / Sec.5 Replicate rule: cross-family disposition downgrade,
    # applied post-hoc now that all four production families are computed. ----
    replicate_rule = apply_replicate_rule(families)

    # ---- harness control (Sec.4.3), BOTH channels (Finding I) ----
    # Population construction (P_dark_u/K_u/K_dark_u/R_u) is channel-independent,
    # so it is only re-derived (cheaply, from the CSV) per channel per universe;
    # disposition-machinery inputs (pooled_sizes, S_F_harn, Z_harn, rho_S,
    # harness_disposition) are unchanged and still come from the 2D channel only.
    harness_reads_by_channel: dict[str, list[HarnessUniverseRead]] = {
        channel: [
            run_harness_universe(u, args.h_true, args.decile, stencil, channel) for u in universes
        ]
        for channel in CHANNELS
    }
    harness_reads = harness_reads_by_channel["combined_with_bh"]

    pooled_sizes = {
        "n_scored": sum(h.n_scored for h in harness_reads),
        "P_dark": sum(len(h.P_dark_u) for h in harness_reads),
        "K": sum(len(h.K_u) for h in harness_reads),
        "K_dark": sum(len(h.K_dark_u) for h in harness_reads),
    }
    for key, anchor in HARNESS_POOLED_ANCHORS.items():
        if pooled_sizes[key] != anchor:
            raise InstrumentDefect(f"harness pooled {key}: {pooled_sizes[key]} != anchor {anchor}")

    prod_primary = families[("iiib", "combined_with_bh")]
    S_F_prod = sum(prod_primary.score_excess[t].S for t in prod_primary.separable_terms)

    harness_pooled_S: dict[str, float] = {}
    harness_pooled_SE: dict[str, float] = {}
    for term in ("B", "g"):
        per_u_target = {
            h.seed: h.stencil_slopes_K_dark[term]
            for h in harness_reads
            if term in h.stencil_slopes_K_dark and h.stencil_slopes_K_dark[term].size
        }
        per_u_ref = {
            h.seed: h.stencil_slopes_R[term]
            for h in harness_reads
            if term in h.stencil_slopes_R and h.stencil_slopes_R[term].size
        }
        s_pooled, se = harness_pool_score(per_u_target, per_u_ref)
        harness_pooled_S[term] = s_pooled
        harness_pooled_SE[term] = se

    # ---- Finding I: harness pooled S/SE for the 1D channel too, reported ----
    # in --out alongside the 2D (disposition-critical) values above. 1D has
    # only the "B" separable term.
    harness_reads_1d = harness_reads_by_channel["combined_no_bh"]
    harness_pooled_S_1d: dict[str, float] = {}
    harness_pooled_SE_1d: dict[str, float] = {}
    for term in _separable_terms_for_channel("combined_no_bh"):
        per_u_target_1d = {
            h.seed: h.stencil_slopes_K_dark[term]
            for h in harness_reads_1d
            if term in h.stencil_slopes_K_dark and h.stencil_slopes_K_dark[term].size
        }
        per_u_ref_1d = {
            h.seed: h.stencil_slopes_R[term]
            for h in harness_reads_1d
            if term in h.stencil_slopes_R and h.stencil_slopes_R[term].size
        }
        s_pooled_1d, se_1d = harness_pool_score(per_u_target_1d, per_u_ref_1d)
        harness_pooled_S_1d[term] = s_pooled_1d
        harness_pooled_SE_1d[term] = se_1d

    # Finding E: S_F_harn = S_B_harn + S_g_harn is an EXACT sum -- its own
    # delete-one-universe jackknife SE (summing the per-universe B/g stencil
    # arrays before jackknifing, since both terms share the same per-universe
    # K_dark_u/R_u event sets and array order) is the literal reading of
    # "delete-one-universe jackknife SE" for S_F^harn (Sec.4.3), not the
    # quadrature combination of two separately-jackknifed per-term SEs (which
    # assumes S_B_u/S_g_u are independent across universes -- they are not,
    # both drawn from the same held-out universe's event set).
    per_u_target_F = {
        h.seed: h.stencil_slopes_K_dark["B"] + h.stencil_slopes_K_dark["g"]
        for h in harness_reads
        if "B" in h.stencil_slopes_K_dark
        and "g" in h.stencil_slopes_K_dark
        and h.stencil_slopes_K_dark["B"].size
    }
    per_u_ref_F = {
        h.seed: h.stencil_slopes_R["B"] + h.stencil_slopes_R["g"]
        for h in harness_reads
        if "B" in h.stencil_slopes_R and "g" in h.stencil_slopes_R and h.stencil_slopes_R["B"].size
    }
    S_F_harn, se_F_harn = harness_pool_score(per_u_target_F, per_u_ref_F)
    Z_harn = S_F_harn / se_F_harn if se_F_harn > 0 else float("nan")
    rho_S = S_F_harn / S_F_prod if S_F_prod != 0 else float("nan")
    rho_S_terms = {
        t: (
            harness_pooled_S[t] / prod_primary.score_excess[t].S
            if prod_primary.score_excess[t].S != 0
            else float("nan")
        )
        for t in ("B", "g")
    }
    s_t_harn = {
        t: (harness_pooled_S[t] / S_F_harn if S_F_harn != 0 else float("nan")) for t in ("B", "g")
    }

    # Use the replicate-rule-BOOKED disposition (Finding J), not the raw
    # per-family value, to decide the owning term fed to the harness side --
    # a replicate miss means the "TERM-OWNS" claim itself is not trustworthy.
    ordered = sorted(prod_primary.term_freeze.shares.items(), key=lambda kv: kv[1], reverse=True)
    prod_owning_term = (
        ordered[0][0] if replicate_rule.booked_disposition.startswith("TERM-OWNS") else None
    )
    s_t_harn_owning = s_t_harn.get(prod_owning_term) if prod_owning_term else None

    harness_disposition = harness_outcome_disposition(
        z_harn=Z_harn,
        rho_s=rho_S,
        s_t_harn_owning=s_t_harn_owning,
        se_harn=se_F_harn,
        prod_delta_F=prod_primary.term_freeze.delta_F,
        z_gate=args.z_gate,
        rho_hi=args.rho_hi,
        rho_lo=args.rho_lo,
        se_unpowered=args.se_unpowered,
        prod_null_lo=prod_primary.null_lo,
        prod_null_hi=prod_primary.null_hi,
    )

    out: dict[str, Any] = {
        "populations": {
            "iiib": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in vars(pops_iiib).items()
            },
            "jr1": {
                k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in vars(pops_jr1).items()
            },
        },
        "production_families": {
            f"{venue}_{CHANNEL_LABEL[channel]}": {
                "mean_h_full": r.mean_h_full,
                "delta_terms": r.term_freeze.delta_terms,
                "delta_F": r.term_freeze.delta_F,
                "delta_D_identity": r.term_freeze.delta_D,
                "r_nonadditivity": r.term_freeze.r_nonadditivity,
                "r_over_abs_delta_F": r.term_freeze.r_over_abs_delta_F,
                "shares": r.term_freeze.shares,
                "null_ci99": [r.null_lo, r.null_hi],
                "disposition": r.disposition,
                "score_excess": {t: {"S": se.S, "se": se.se} for t, se in r.score_excess.items()},
                "map_h_full": r.map_h_full,
                "map_rail_full": r.map_rail_full,
                "map_rails_freeze": r.map_rails_freeze,
                "max_g1_closure_residual": r.max_g1_closure_residual,
                # Finding H: delta_K_leaveout (159-event K) is the byte-id
                # G-2(ii)-anchored statistic; delta_K_dark_leaveout (144-event
                # K_dark) is the separate Sec.2.3 reported-only object -- kept
                # distinct, never conflated.
                "delta_K_leaveout": r.delta_K_leaveout,
                "delta_K_dark_leaveout": r.delta_K_dark_leaveout,
                "concordance_K_dark_over_K_reported_only": (
                    r.delta_K_dark_leaveout / r.delta_K_leaveout
                    if r.delta_K_leaveout not in (0.0,) and not np.isnan(r.delta_K_leaveout)
                    else float("nan")
                ),
            }
            for (venue, channel), r in families.items()
        },
        "K_hosted_leaveout_reported_only": delta_K_hosted,
        "replicate_rule": {
            # Sec.5 / Finding J: cross-family disposition downgrade, applied
            # to the primary (iiib, 2D) family only. `booked_disposition` is
            # what a reader should trust for iiib/2D; every family's own
            # `disposition` field above (in "production_families") remains
            # the raw, un-downgraded per-family value.
            "family": "iiib_2D",
            "raw_disposition": families[("iiib", "combined_with_bh")].disposition,
            "booked_disposition": replicate_rule.booked_disposition,
            "downgraded": replicate_rule.downgraded,
            "reasons": replicate_rule.reasons,
        },
        "harness": {
            "n_universes": len(harness_reads),
            "pooled_sizes": pooled_sizes,
            "pooled_S": harness_pooled_S,
            "pooled_SE_jackknife": harness_pooled_SE,
            "S_F_harn": S_F_harn,
            "SE_F_harn": se_F_harn,
            "Z_harn": Z_harn,
            "rho_S": rho_S,
            "rho_S_terms": rho_S_terms,
            "s_t_harn": s_t_harn,
            "s_t_harn_defined": bool(abs(Z_harn) > 3) if not np.isnan(Z_harn) else False,
            "disposition": harness_disposition,
            "n_universes_railed": sum(1 for h in harness_reads if h.rail_full_u),
            "per_universe": [
                {
                    "seed": h.seed,
                    "n_scored": h.n_scored,
                    "n_P_dark_u": len(h.P_dark_u),
                    "n_K_u": len(h.K_u),
                    "n_K_dark_u": len(h.K_dark_u),
                    "n_R_u": len(h.R_u),
                    "delta_terms_u": h.delta_terms_u,
                    "delta_F_u": h.delta_F_u,
                    "r_u": h.r_u,
                    "rail_full_u": h.rail_full_u,
                }
                for h in harness_reads
            ],
            # Finding I: 1D-channel harness pooled control, reported only --
            # the disposition machinery above is unchanged (2D-only, as before).
            "channel_1D_pooled_S": harness_pooled_S_1d,
            "channel_1D_pooled_SE_jackknife": harness_pooled_SE_1d,
            "channel_1D_n_universes_railed": sum(1 for h in harness_reads_1d if h.rail_full_u),
        },
        "bands": {
            "share_own": args.share_own,
            "share_diffuse": args.share_diffuse,
            "rho_hi": args.rho_hi,
            "rho_lo": args.rho_lo,
            "z_gate": args.z_gate,
            "se_unpowered": args.se_unpowered,
            "nonadditivity_max": args.nonadditivity_max,
        },
        "run_metadata": {
            "g1d_tol": args.g1d_tol,
            "h_true": args.h_true,
            "decile": args.decile,
            "stencil": list(stencil),
            "null_draws": args.null_draws,
            "null_seed": args.null_seed,
            "production_commit": PRODUCTION_COMMIT,
        },
    }

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump(out, f, indent=2)
        print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
