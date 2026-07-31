"""Generator for the Defect Museum annex (``book/site/museum.html``).

Four data files, all deterministic, all read-only outside ``book/``:

``book/site/data/museum_ledger.json``
    The 98-row ``BIAS_HISTORY_LEDGER.md`` digest that powers the museum's
    ledger browser and (later) the book-wide "Has this been tried?" instrument
    (BW3, ``WIDGET_REQUESTS.md`` R-INT-2).  Parsed straight out of the ledger's
    own markdown table -- row id, era/date, hypothesis, decisive test, verdict
    (verbatim, markdown stripped), documented artifact, residual -- plus a
    derived verdict class, a venue tag extracted only from tokens that actually
    appear in the row, and the DO-NOT-RE-TRY mapping built from the ledger's
    own section 2 back-references.  Nothing is invented: every string is a
    substring of the ledger, and the derived fields are keyword scans over it.

``book/site/data/museum_quadrature.json``   (M1, the flagship dial)
    A genuine re-computation of ``results/volume_trunc_ab_20260712/
    quadrature_diagnostic.py``'s integrand with the project's own
    ``physical_relations``, swept over the Gauss-Legendre order ``n`` in TWO
    evaluation modes:

      * ``scalar_dist``   -- the GW leg written with ``dist()``, exactly as the
        2026-07-12 diagnostic wrote it;
      * ``vectorized``    -- the GW leg written with ``dist_vectorized()``,
        exactly as production writes it
        (``bayesian_statistics.py:3806`` ``numerator_integrant_without_bh_mass``).

    The two disagree, and the disagreement is the exhibit.  See
    ``book/design/flags/museum_FLAGS.md`` F-museum-1 -- this generator does NOT
    reconcile them; it emits both, together with the recorded FINDING.md table
    and the real seed600 A/B posteriors from ``gate_result.json``.

``book/site/data/museum_archaeology.json``  (M2, the timeline scrubber)
    The stored-posterior era timeline of ledger row #49b, as *recorded*
    measurements (each carries its own venue, event count and artifact line);
    the museum never plots them as one continuous series.

``book/site/data/museum_h0_independent.json``
    A re-run of the commission's own ``injection_scan.py`` (ledger #49a).  That
    script is untracked (it lives only in the main checkout's working tree), so
    the generator resolves it from this repo root, then from a sibling
    ``MasterThesisCode`` checkout, and if neither is present leaves the
    already-committed JSON alone and prints a NOTICE -- it never writes a
    silently degraded file.

Determinism: no RNG of this generator's own; ``injection_scan.py`` seeds every
draw it makes (``default_rng(2024)`` for the catalogue, ``default_rng(int(h*1000))``
per injection), so its output is reproducible.

Run as::

    /home/jasper/Repositories/MasterThesisCode/.venv/bin/python \\
        book/generators/gen_museum.py
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import fixed_quad, quad
from scipy.special import roots_legendre
from scipy.stats import norm

BOOK_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = BOOK_ROOT / "book" / "site" / "data"

LEDGER_REL = (
    "results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md"
)
CLAIM_REL = "results/campaign51_20260728/realistic_20260729/CLAIM_2D_BIAS_20260730.md"
GATE_RESULT_REL = "results/volume_trunc_ab_20260712/gate_result.json"
INJECTION_SCAN_REL = "results/commission_20260701/injection_scan.py"


# ----------------------------------------------------------------------
# artifact resolution: this worktree first, then a sibling main checkout
# ----------------------------------------------------------------------
def _roots() -> list[Path]:
    return [BOOK_ROOT, BOOK_ROOT.parent / "MasterThesisCode"]


def resolve(rel: str) -> Path | None:
    for root in _roots():
        p = root / rel
        if p.exists():
            return p
    return None


def must_resolve(rel: str) -> Path:
    p = resolve(rel)
    if p is None:
        raise FileNotFoundError(f"gen_museum: required artifact not found in any root: {rel}")
    return p


def write_json(name: str, payload: Any) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name
    path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=False))
    print(f"  wrote {path.relative_to(BOOK_ROOT)}  ({path.stat().st_size / 1024:.1f} KB)")


def r(x: float, sig: int = 6) -> float:
    """Round to `sig` significant digits (keeps the JSON small and honest)."""
    if x == 0 or not np.isfinite(x):
        return float(x)
    return float(f"%.{sig}g" % x)


# ======================================================================
# 1. the ledger digest
# ======================================================================
_MD_STRIP = [
    (re.compile(r"\*\*(.+?)\*\*", re.S), r"\1"),
    (re.compile(r"~~(.+?)~~", re.S), r"\1"),
    (re.compile(r"`([^`]+)`"), r"\1"),
    (re.compile(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])"), r"\1"),
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),
]

# Verdict classes, in priority order.  Each is a (class, label, [needles]) triple;
# the needles are matched case-sensitively against the VERDICT cell as written.
_VERDICT_RULES: list[tuple[str, str, list[str]]] = [
    ("overturned", "overturned / re-attributed", ["OVERTURNED", "REVERSED", "RE-ATTRIBUTED"]),
    (
        "refuted",
        "refuted / falsified",
        [
            "REFUTED",
            "FALSIFIED",
            "falsified",
            "DISQUALIFIED",
            "GATE FAIL",
            "FAIL for",
            "AMPLIFIES",
            "VIOLATED",
            "relocates",
            "empirically dead",
        ],
    ),
    ("exonerated", "exonerated", ["EXONERATED", "RULED OUT", "NOT A FACTOR", "NOT A PRODUCTION"]),
    (
        "non_cause",
        "real, but not the cause",
        [
            "not the cause",
            "NOT the bias cause",
            "not a bias source",
            "seam, not bug",
            "inert",
            "bias-neutral",
            "WRONG SIGN",
        ],
    ),
    ("confirmed", "confirmed", ["CONFIRMED", "VALIDATED CORRECT", "ROOT-CAUSED"]),
    ("fixed", "fixed and landed", ["fixed", "FIXED"]),
    ("open", "open / untested", ["STILL OPEN", "UNTESTED", "open"]),
    (
        "qualified",
        "qualified / ambiguous",
        [
            "QUALIFIED FAIL",
            "[AMBIG]",
            "PARTIAL",
            "NECESSARY, NOT SUFFICIENT",
            "NO material",
            "NULL",
            "NEGLIGIBLE",
            "design choice",
        ],
    ),
]

# Venue tokens: only tags that literally appear in the row are attached.
_VENUE_TOKENS = [
    ("seed61000/real_r1", ["real_r1", "seed61000"]),
    ("seed600 (494-event shallow subsample)", ["seed600", "494-ev", "494-event"]),
    ("seed1000 (deep venue)", ["seed1000"]),
    ("seed400", ["seed400"]),
    ("seed200/seed300 seam", ["seed200"]),
    ("seed900/2000/3000/90000 multiseed", ["seed900", "90000"]),
    ("pp_coverage synthetic harness", ["pp_coverage", "harness", "coverage test", "realizations"]),
    ("single-host toy", ["toy"]),
    ("commission injection scan", ["injection scan", "tournament"]),
    ("campaign #51 / #53", ["#51", "#53", "campaign51"]),
    ("cluster production run", ["cluster"]),
]


def _strip_md(cell: str) -> str:
    out = cell.strip()
    for pat, repl in _MD_STRIP:
        out = pat.sub(repl, out)
    return re.sub(r"\s+", " ", out).strip()


def _split_row(line: str) -> list[str]:
    """Split a markdown table row on unescaped pipes."""
    tmp = line.strip().strip("|")
    tmp = tmp.replace(r"\|", "\x00")
    return [c.replace("\x00", "|") for c in tmp.split("|")]


def _classify(verdict_raw: str) -> tuple[str, str]:
    for cls, label, needles in _VERDICT_RULES:
        for needle in needles:
            if needle in verdict_raw:
                return cls, label
    return "measured", "measured"


def _venues(row_text: str) -> list[str]:
    found = []
    for label, needles in _VENUE_TOKENS:
        if any(n in row_text for n in needles):
            found.append(label)
    return found


def _date_of(era: str) -> str | None:
    m = re.search(r"\b(\d{2})-(\d{2})\b", era)
    if not m:
        return None
    return f"2026-{m.group(1)}-{m.group(2)}"


def build_ledger() -> dict[str, Any]:
    path = must_resolve(LEDGER_REL)
    text = path.read_text()
    lines = text.splitlines()

    rows: list[dict[str, Any]] = []
    in_table = False
    for line in lines:
        if line.startswith("| # | Era"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            if set(line.replace("|", "").strip()) <= set("-: "):
                continue
            cells = _split_row(line)
            if len(cells) < 7:
                continue
            rid = _strip_md(cells[0])
            if not re.fullmatch(r"\d+[a-z]?", rid):
                continue
            era = _strip_md(cells[1])
            verdict_raw = cells[4]
            cls, label = _classify(verdict_raw)
            row_text = " ".join(cells)
            rows.append(
                {
                    "id": rid,
                    "n": int(re.match(r"\d+", rid).group(0)),
                    "era": era,
                    "date": _date_of(era),
                    "hypothesis": _strip_md(cells[2]),
                    "test": _strip_md(cells[3]),
                    "verdict": _strip_md(verdict_raw),
                    "verdict_class": cls,
                    "verdict_label": label,
                    "documented": _strip_md(cells[5]),
                    "residual": _strip_md(cells[6]),
                    "venues": _venues(row_text),
                }
            )

    # ---- DO-NOT-RE-TRY: section 2's own back-references -------------
    sec2 = text.split("## 2. DO NOT RE-TRY")[1].split("## 3.")[0]
    dnr_items: list[dict[str, Any]] = []
    for m in re.finditer(r"^(\d+)\.\s+⚠?\s*(.+?)(?=^\d+\.\s|\Z)", sec2, re.S | re.M):
        body = _strip_md(m.group(2))
        # Only parenthesised groups that consist *entirely* of ledger back-refs
        # are read as back-refs: "(#25, #30, #10)" yes, "(#30 option b)" no
        # (that one is a GitHub issue number, not a ledger row).
        refs: set[str] = set()
        for grp in re.findall(r"\(([^()]*)\)", m.group(2)):
            if re.fullmatch(r"#\d+[a-z]?(\s*,\s*#\d+[a-z]?)*", grp.strip()):
                refs.update(re.findall(r"#(\d+[a-z]?)", grp))
        ref_list = sorted(refs)
        dnr_items.append({"item": int(m.group(1)), "text": body, "ledger_rows": ref_list})
    dnr_rows = sorted({rid for it in dnr_items for rid in it["ledger_rows"]})
    for row in rows:
        row["do_not_retry"] = row["id"] in dnr_rows

    # ---- the claim file's own Exonerated list (verbatim names) ------
    claim_path = must_resolve(CLAIM_REL)
    claim = claim_path.read_text()
    block = claim.split("## Exonerated — do NOT re-open without new evidence")[1]
    block = block.split("**[2026-07-30 adjudication")[0]
    claim_exonerated = [
        _strip_md(x) for x in block.strip().split("·") if _strip_md(x)
    ]

    census: dict[str, int] = {}
    for row in rows:
        census[row["verdict_class"]] = census.get(row["verdict_class"], 0) + 1

    print(f"  ledger: {len(rows)} rows, census {census}")
    return {
        "source": LEDGER_REL,
        "compiled": "2026-07-30",
        "n_rows": len(rows),
        "census": census,
        "class_labels": {cls: label for cls, label, _ in _VERDICT_RULES}
        | {"measured": "measured — a number, not a verdict"},
        "rows": rows,
        "do_not_retry_items": dnr_items,
        "do_not_retry_rows": dnr_rows,
        "claim_file_exonerated": claim_exonerated,
    }


# ======================================================================
# 2. M1 — the quadrature dial (genuine re-computation)
# ======================================================================
Z_G = 0.05
SIGMA_Z = 0.033
SIGMA_DL_FRAC = 0.05
H_VALUES = [0.60, 0.70, 0.73, 0.80, 0.86]
N_LADDER = [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 400, 600]
N_NODES_SHIPPED = [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300]

# The FINDING.md table, verbatim (recorded measurement, 2026-07-12).
RECORDED_FINDING = {
    "h": [0.60, 0.70, 0.73, 0.80, 0.86],
    "gw_window_n50": [0.0003, 0.0004, 0.0005, 0.0006, 0.0007],
    "host_window_n50": [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    "host_window_exact": [0.2417, 0.3835, 0.4314, 0.5495, 0.6537],
    "source": "results/volume_trunc_ab_20260712/FINDING.md (quadrature_diagnostic.py)",
}


def build_quadrature() -> dict[str, Any]:
    from master_thesis_code.physical_relations import (  # noqa: PLC0415
        comoving_volume_element,
        dist,
        dist_to_redshift,
        dist_vectorized,
    )

    d_L_det = float(dist(Z_G, h=0.73))  # Gpc
    dl_unc = SIGMA_DL_FRAC * d_L_det
    lo = max(Z_G - 4.0 * SIGMA_Z, 0.0)
    hi = Z_G + 4.0 * SIGMA_Z
    gz = norm(Z_G, SIGMA_Z)

    def prior_unnorm(z: np.ndarray, h: float) -> np.ndarray:
        z = np.atleast_1d(np.asarray(z, dtype=np.float64))
        return gz.pdf(z) * np.asarray(comoving_volume_element(z, h=h)) / (1.0 + z)

    def gw_vec(z: np.ndarray, h: float) -> np.ndarray:
        """GW distance-fraction likelihood, array-correct (production form)."""
        z = np.atleast_1d(np.asarray(z, dtype=np.float64))
        f = np.asarray(dist_vectorized(z, h=h)) / d_L_det
        return np.exp(-0.5 * ((f - 1.0) / SIGMA_DL_FRAC) ** 2) / (
            np.sqrt(2 * np.pi) * SIGMA_DL_FRAC
        )

    def gw_scalar(z: np.ndarray, h: float) -> np.ndarray:
        """GW leg written with scalar-only ``dist()`` — the 2026-07-12 form.

        ``dist()`` collapses an array argument to a 0-d array holding the value
        at its FIRST element, so under ``fixed_quad`` (which passes the whole
        node array at once) the GW factor becomes a constant.
        """
        f = np.asarray(dist(z, h=h)) / d_L_det
        return np.exp(-0.5 * ((f - 1.0) / SIGMA_DL_FRAC) ** 2) / (
            np.sqrt(2 * np.pi) * SIGMA_DL_FRAC
        )

    out: dict[str, Any] = {
        "host": {
            "z_g": Z_G,
            "sigma_z": SIGMA_Z,
            "sigma_z_over_z": r(SIGMA_Z / Z_G, 4),
            "d_L_det_Gpc": r(d_L_det),
            "sigma_dl_frac": SIGMA_DL_FRAC,
            "host_window": [r(lo), r(hi)],
        },
        "h_values": H_VALUES,
        "n_ladder": N_LADDER,
        "recorded_finding": RECORDED_FINDING,
    }

    z_peak, exact, gw_window = [], [], []
    fq_vec: dict[str, list[float]] = {}
    fq_sca: dict[str, list[float]] = {}
    integrand: dict[str, dict[str, list[float]]] = {}
    predicted_gw_window_scalar = []

    for h in H_VALUES:
        z_norm = float(fixed_quad(lambda z, hh=h: prior_unnorm(z, hh), lo, hi, n=50)[0])

        def f_vec(z: np.ndarray, hh: float = h, zn: float = z_norm) -> np.ndarray:
            return gw_vec(z, hh) * prior_unnorm(z, hh) / zn

        def f_sca(z: np.ndarray, hh: float = h, zn: float = z_norm) -> np.ndarray:
            return gw_scalar(z, hh) * prior_unnorm(z, hh) / zn

        ex = float(quad(lambda z, hh=h, zn=z_norm: float(f_vec(np.array([z]), hh, zn)[0]),
                        lo, hi, limit=400)[0])
        exact.append(r(ex, 6))
        zp = float(dist_to_redshift(d_L_det, h=h))
        z_peak.append(r(zp, 6))

        key = f"{h:.2f}"
        fq_vec[key] = [r(float(fixed_quad(f_vec, lo, hi, n=n)[0]), 6) for n in N_LADDER]
        fq_sca[key] = [r(float(fixed_quad(f_sca, lo, hi, n=n)[0]), 6) for n in N_LADDER]

        # the event-level GW window (what the default modes integrate over)
        n_lo = float(dist_to_redshift(d_L_det - 4 * dl_unc, h=h))
        n_hi = float(dist_to_redshift(d_L_det + 4 * dl_unc, h=h))
        gw_window.append([r(n_lo, 6), r(n_hi, 6)])
        predicted_gw_window_scalar.append(
            r(float(fixed_quad(f_sca, n_lo, n_hi, n=50)[0]), 6)
        )

        # display integrand: coarse over the whole window + dense over the peak
        z_coarse = np.linspace(lo, hi, 250)
        z_dense = np.linspace(max(zp - 0.015, lo), min(zp + 0.015, hi), 450)
        zz = np.unique(np.concatenate([z_coarse, z_dense]))
        yy = f_vec(zz)
        integrand[key] = {
            "z": [r(float(v), 6) for v in zz],
            "y": [r(float(v), 5) for v in yy],
        }

    out["z_peak"] = z_peak
    out["exact"] = exact
    out["gw_window"] = gw_window
    out["fixed_quad_vectorized"] = fq_vec
    out["fixed_quad_scalar_dist"] = fq_sca
    out["predicted_gw_window_n50_scalar"] = predicted_gw_window_scalar
    out["integrand"] = integrand

    nodes: dict[str, list[float]] = {}
    for n in N_NODES_SHIPPED:
        x, _w = roots_legendre(n)
        nodes[str(n)] = [r(float(0.5 * (hi - lo) * xi + 0.5 * (hi + lo)), 6) for xi in x]
    out["nodes"] = nodes

    # ---- gates against the recorded FINDING.md table ----------------
    gates: dict[str, Any] = {}
    for i, h in enumerate([0.60, 0.73, 0.86]):
        j = H_VALUES.index(h)
        rec = RECORDED_FINDING["host_window_exact"][RECORDED_FINDING["h"].index(h)]
        gates[f"exact_h{h:.2f}"] = {
            "recomputed": exact[j],
            "recorded": rec,
            "match_4dp": abs(exact[j] - rec) < 5e-5,
        }
        rec_gw = RECORDED_FINDING["gw_window_n50"][RECORDED_FINDING["h"].index(h)]
        gates[f"gw_window_n50_scalar_h{h:.2f}"] = {
            "recomputed_scalar_dist": predicted_gw_window_scalar[j],
            "recorded": rec_gw,
            "match_4dp": abs(predicted_gw_window_scalar[j] - rec_gw) < 5e-5,
        }
    n50 = N_LADDER.index(50)
    gates["host_window_n50_scalar_is_zero_to_4dp"] = all(
        abs(fq_sca[f"{h:.2f}"][n50]) < 5e-5 for h in H_VALUES
    )
    gates["host_window_n50_vectorized_within_3pct_of_exact"] = all(
        abs(fq_vec[f"{h:.2f}"][n50] - exact[i]) / exact[i] < 0.03 for i, h in enumerate(H_VALUES)
    )
    out["gates"] = gates

    bad = [k for k, v in gates.items() if isinstance(v, dict) and not v["match_4dp"]]
    if bad:
        raise AssertionError(f"gen_museum: FINDING.md reproduction gate failed for {bad}")
    if not gates["host_window_n50_scalar_is_zero_to_4dp"]:
        raise AssertionError("gen_museum: scalar-dist n=50 no longer reads 0.0000")
    print(
        "  quadrature: FINDING.md exact column + GW-window column reproduced to 4 dp; "
        f"vectorized n=50 within 3% of exact = "
        f"{gates['host_window_n50_vectorized_within_3pct_of_exact']}"
    )

    # ---- the real seed600 A/B posteriors ----------------------------
    ab = json.loads(must_resolve(GATE_RESULT_REL).read_text())
    out["ab"] = ab
    return out


# ======================================================================
# 3. M2 — the archaeology timeline (recorded eras)
# ======================================================================
# Every entry is a RECORDED measurement with its own venue and artifact line.
# They are deliberately NOT one time series: the event sets, the code and the
# estimator all change between rows.  The museum says so on the page.
ARCHAEOLOGY = [
    {
        "date": "2026-04-09",
        "label": "v2.1 baseline — no selection machinery",
        "map_h": 0.735,
        "venue": "stored posterior, n = 417 events",
        "what": "A raw per-event distance-redshift product: no Gray D(h), no zero-fill p_det, "
                "no completeness. Interior, recovers the injected 0.73; edge mass ~1e-36.",
        "commit": None,
        "artifact": "WF1_DIGEST.md:9-15 / ledger #49b",
        "state": "interior",
    },
    {
        "date": "2026-04-24",
        "label": "v2.2 — selection switched ON, rails to 0.86",
        "map_h": 0.86,
        "venue": "stored posterior, same era",
        "what": "The Gray selection/completeness machinery is switched on (zero-fill p_det "
                "a70d1a2 + Gray D(h) 2853c32). The posterior rails to the upper grid edge.",
        "commit": "a70d1a2 + 2853c32",
        "artifact": "WF1_DIGEST.md:9-15 / ledger #49b",
        "state": "railed",
    },
    {
        "date": "2026-04-29",
        "label": "zero-fill p_det fixed — interior, +4.8% high",
        "map_h": None,
        "map_note": "interior but +4.8% high (no single MAP recorded in the digest)",
        "venue": "stored posteriors, same era",
        "what": "Fixing the p_det zero-fill de-rails it, but leaves a +4.8% high bias.",
        "commit": "3697bdd",
        "artifact": "WF1_DIGEST.md:9-15",
        "state": "interior",
    },
    {
        "date": "2026-05 → 2026-06",
        "label": "interior, persistently +1.5–4% high",
        "map_h": None,
        "map_note": "+1.5–4% high, seed-dependent; the h-grid is narrowed to [0.70, 0.80]",
        "venue": "several seeds",
        "what": "Two months of interior-but-high posteriors. The narrowed grid is itself part "
                "of the archaeology: a grid that cannot show a rail will not show you one.",
        "commit": None,
        "artifact": "WF1_DIGEST.md:9-15",
        "state": "interior",
    },
    {
        "date": "2026-06-26",
        "label": "restructure to the single Gray ratio",
        "map_h": None,
        "map_note": "structural change, no MAP recorded at this commit",
        "venue": "code",
        "what": "The per-event likelihood becomes one quotient, "
                "p_i = (β_G·L_cat + B_num)/D(h).",
        "commit": "f1232de",
        "artifact": "WF1_DIGEST.md:9-15",
        "state": "structural",
    },
    {
        "date": "2026-06-28",
        "label": "seed600 at HEAD rails to 0.86 again",
        "map_h": 0.86,
        "venue": "seed600, HEAD code (auditor's out-of-band check)",
        "what": "The auditor's own re-run: posterior argmax 0.86 while D(h)'s own argmax is "
                "0.60 — the rail is back, and it is normalization-shaped.",
        "commit": None,
        "artifact": "WF1_DIGEST.md:9-15",
        "state": "railed",
    },
    {
        "date": "2026-07-01",
        "label": "de-rail matrix, step 1: pre-4π",
        "map_h": 0.86,
        "venue": "real data, 494 events, 7-point h-grid",
        "what": "The de-rail matrix starts where the audit left it: railed HIGH.",
        "commit": None,
        "artifact": "ledger #49 / project_commission_derail.md:12-18",
        "state": "railed",
    },
    {
        "date": "2026-07-02",
        "label": "de-rail matrix, step 2: 4π sky marginal only",
        "map_h": 0.60,
        "venue": "real data, 494 events",
        "what": "Fixing the completion term's 4π sky marginal flips the rail from the top of "
                "the grid to the bottom — necessary, not sufficient.",
        "commit": "cb16142 + 4a259b7",
        "artifact": "ledger #49 / #46",
        "state": "railed",
    },
    {
        "date": "2026-07-02",
        "label": "de-rail matrix, step 3: local_ratio",
        "map_h": 0.73,
        "venue": "real data, 494 events",
        "what": "A locally-normalized ratio of sums puts 98% of the posterior mass on an "
                "interior peak at truth.",
        "commit": None,
        "artifact": "ledger #49",
        "state": "interior",
    },
    {
        "date": "2026-07-02",
        "label": "de-rail matrix, step 4: volume_deconv",
        "map_h": 0.73,
        "venue": "real data, 494 events",
        "what": "The ratified host-z volume kernel keeps it there.",
        "commit": "235b783",
        "artifact": "ledger #49 / #47",
        "state": "interior",
    },
    {
        "date": "2026-07-29",
        "label": "campaign #51 — idealized baseline",
        "map_h": 0.7299,
        "map_label": "1D 0.72990 (−0.24σ) · 2D 0.7300 (−0.36σ)",
        "venue": "seed61000, point kernel + generator_marginal, unscattered catalogue",
        "what": "1D 0.72990 (−0.24σ), 2D 0.7300 (−0.36σ) on a 1e-4 zoom grid. 100% of the "
                "information comes from 76 in-catalogue events; 3 golden events carry 46%.",
        "commit": None,
        "artifact": "ledger #93 / IDEALIZED_BASELINE_READOUT.md:25-47",
        "state": "interior",
    },
    {
        "date": "2026-07-29",
        "label": "campaign #53 — realistic run",
        "map_h": 0.7205,
        "map_label": "1D pooled 0.7205 (per-run range 0.700–0.740); 2D 0.780–0.820, mean bias +0.077",
        "venue": "seed61000/62000, absolute_marginal + volume_deconv, scattered catalogue, 10 runs",
        "what": "1D 0.700–0.740 (pooled 0.7205); 2D 0.780–0.820, mean bias +0.077, 10/10 runs "
                "pull > 2σ. The 2D pairing is a designated CANDIDATE, not ratified ground.",
        "commit": None,
        "artifact": "ledger #94 / REALISTIC_READOUT.md:19-32",
        "state": "current",
    },
]


def build_archaeology() -> dict[str, Any]:
    return {
        "truth_h": 0.73,
        "warning": (
            "These are separate stored measurements, not one time series: the event set, "
            "the code and the estimator all change between rows. Each row carries its own "
            "venue and artifact."
        ),
        "source": "BIAS_HISTORY_LEDGER.md #49b; commission WF1_DIGEST.md:9-15; ledger #49, #93, #94",
        "entries": ARCHAEOLOGY,
    }


# ======================================================================
# 4. #49a — the H0-independent estimator (re-run of the commission's script)
# ======================================================================
RECORDED_49A = {
    "verdict": "production MAP = 0.86 for EVERY injected truth 0.63→0.77, while catalog_only "
               "tracks truth exactly (0.63→0.63 … 0.77→0.77)",
    "source": "ledger #49a; synthesis/DRAFT_REPORT.md:24-27; WF2_DIGEST.md:26-30",
}


def build_h0_independent() -> dict[str, Any] | None:
    path = resolve(INJECTION_SCAN_REL)
    if path is None:
        print(
            "  NOTICE: results/commission_20260701/injection_scan.py not found in this "
            "worktree or a sibling checkout (it is untracked). Keeping the committed "
            "museum_h0_independent.json."
        )
        return None

    spec = importlib.util.spec_from_file_location("_commission_injection_scan", path)
    if spec is None or spec.loader is None:
        print("  NOTICE: could not load injection_scan.py; keeping committed JSON.")
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module body is import-only; main() is under __main__

    rng = np.random.default_rng(2024)
    z_g, M_g = mod.build_catalog(20000, 1.2, 0.70, rng)
    D_h, bGbar, Sg = mod.globals_tables(z_g, M_g, mod.H_GRID)

    truths = [0.63, 0.67, 0.70, 0.73, 0.77]
    cat_only, production = [], []
    for h_true in truths:
        mc, mp = mod.run_injection(
            h_true, z_g, M_g, D_h, bGbar, Sg, np.random.default_rng(int(h_true * 1000))
        )
        cat_only.append(r(float(mc), 4))
        production.append(r(float(mp), 4))
    print(f"  injection scan re-run: catalog_only {cat_only}  production {production}")
    return {
        "injected_truth": truths,
        "map_catalog_only": cat_only,
        "map_production": production,
        "h_grid": [r(float(x), 3) for x in mod.H_GRID],
        "venue": (
            "the commission's own synthetic harness: 20,000-galaxy catalogue, moderate "
            "completeness f(z) = exp(−(z/0.3)²), erfc detection horizon at 3.0 Gpc — "
            "NOT the production catalogue"
        ),
        "script": INJECTION_SCAN_REL,
        "recorded": RECORDED_49A,
        "note": (
            "Re-run here with the script's own seeds. The production column reproduces the "
            "recorded verdict exactly (0.86 at every injected truth). The catalog_only column "
            "reads 0.630 / 0.660 / 0.690 / 0.730 / 0.770 — two of the five sit one 0.01 grid "
            "step below the injected truth, where the digest's wording is 'tracks the truth "
            "exactly'. See book/design/flags/museum_FLAGS.md F-museum-2."
        ),
    }


# ======================================================================
def main() -> None:
    print("gen_museum: building the Defect Museum data")
    write_json("museum_ledger.json", build_ledger())
    write_json("museum_quadrature.json", build_quadrature())
    write_json("museum_archaeology.json", build_archaeology())
    h0i = build_h0_independent()
    if h0i is not None:
        write_json("museum_h0_independent.json", h0i)
    print("gen_museum: done")


if __name__ == "__main__":
    sys.exit(main())
