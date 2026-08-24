"""Generator for Chapter 10½ — "The Anatomy of the Bias" (interlude).

Design Amendment 1 to BOOK_DESIGN.md (2026-08-23, author-approved row #175).
This is an interlude, not a numbered chapter: it sits between Ch 10 and Ch 11
and reports the two-day August 2026 campaign that decomposed the B-SEL
fleet's headline bias, found and fixed the off-cell S̄_φ omission, and then
ran the catalogue-leg "twin" thread through to the b0 identity adjudicator.

Every number below is HAND-AUTHORED from quoted ledger rows, exactly the
precedent of ``gen_ch11.py``'s ``BOARD`` table for I11.2 (see BOOK_DESIGN.md
§2's file-ownership map and the I11.2 licence it names). There is no
production data to re-run here — the "measurement" already happened inside
the ledger's own registered, A20-reviewed process; this generator's job is
only to carry those banked numbers into the two interactives without
touching a digit, and to gate on the cited source files actually existing.

Outputs
-------
``book/site/data/ch10x_decomposition.json``  (I10X.1 "The Decomposition Bench")
    The three-way split of the B-SEL fleet's headline bias -0.1083 (rows
    #149-#150), each contribution's status as of the restored-arm mechanism
    finding (row #155) and the fused-cell confirmation (O6/O7/O8, rows
    #158/#161/#165), plus the restored-arm null itself. Vocabulary is the
    primer's (docs/PRIMER_BIAS_CHANNELS_20260822.md §3): "contribution" for
    the three-way split, never "channel".

``book/site/data/ch10x_timeline.json``        (I10X.2 "Verdict Archaeology")
    The twin thread's dated verdict trajectory, registration through the b0
    identity verdict: one entry per ledger row / pre-registration section
    from PREREGISTRATION_P3_TWIN_20260822.md and
    PREREGISTRATION_B0_IDENTITY_20260823.md's own section headers, each
    tagged with a kind in {registration, amendment, refutation, ratification,
    measurement, verdict}.

Determinism: no RNG, no data re-run. Read-only outside ``book/`` (this
generator only *checks that the cited source files exist* — it does not
parse or recompute from them; the numbers are transcribed by hand from the
ledger rows named in each entry's ``row`` field, and a human re-grep of that
row is the verification path, exactly as gen_ch11.py's BOARD table documents
of itself).

Run as::

    /home/jasper/Repositories/darksiren-emri/.venv/bin/python \\
        book/generators/gen_ch10x.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BOOK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BOOK_ROOT.parent
OUT_DIR = BOOK_ROOT / "site" / "data"

REAL_REL = "results/campaign51_20260728/realistic_20260729"
LEDGER_REL = f"{REAL_REL}/gate_b_20260730/BIAS_HISTORY_LEDGER.md"
PRIMER_REL = "docs/PRIMER_BIAS_CHANNELS_20260822.md"
P3_TWIN_REL = f"{REAL_REL}/PREREGISTRATION_P3_TWIN_20260822.md"
B0_REL = f"{REAL_REL}/PREREGISTRATION_B0_IDENTITY_20260823.md"

REQUIRED_SOURCES = (LEDGER_REL, PRIMER_REL, P3_TWIN_REL, B0_REL)


def need(rel: str) -> Path:
    """Confirm a cited source file exists; raise (don't silently drift) if not."""
    p = REPO_ROOT / rel
    if not p.exists():
        msg = f"gen_ch10x: cited source not found: {rel} (looked in {REPO_ROOT})"
        raise FileNotFoundError(msg)
    return p


def rnd(x: float, n: int = 6) -> float:
    return round(float(x), n)


# ==========================================================================
# I10X.1 — the three-way decomposition of B-SEL's headline -0.1083
# ==========================================================================
# Every value, row citation and quote below is transcribed VERBATIM from
# BIAS_HISTORY_LEDGER.md rows #149-#157 and docs/PRIMER_BIAS_CHANNELS §3's
# table (the primer *is* the canonical vocabulary this chapter uses:
# "contribution" for these three, "channel" reserved for §2's readouts).
HEADLINE_BIAS = -0.1083
HEADLINE_QUOTE = "B-SEL fleet's headline bias (mean_h − 0.73 = −0.108)"

CONTRIBUTIONS: list[dict[str, Any]] = [
    {
        "id": "impostor",
        "label": "impostor drag",
        "value": -0.079,
        "value_precise": -0.0792,
        "share_pct": 73,
        "what": ("catalogue candidates that are NOT the true host sit at low z and drag h down"),
        "status": "OPEN — the venue-physics front",
        "row": "#149",
        "quote": (
            "Setting L_cat_no_bh ≡ 0 (exact subtraction) moves the 12-seed "
            "fleet from −0.1083 to −0.0291. Positive in 12/12 seeds "
            "(+0.030…+0.164)."
        ),
    },
    {
        "id": "tilt",
        "label": "dark-fraction tilt",
        "value": 0.055,
        "value_precise": 0.055,
        "share_pct": None,
        "what": "a composition effect of the catalogue/dark mixture",
        "status": "measured, understood",
        "row": "#150",
        "quote": (
            "The tilt is measured at −0.133/h per event (ln D̃/β̄_G), "
            "≈ −24 nats/h per seed, and owns the pure−matched gap "
            "(+0.025…+0.085, width-dependent)."
        ),
    },
    {
        "id": "matched",
        "label": "matched-channel violation",
        "value": -0.085,
        "value_precise": -0.0846,
        "sem": 0.0095,
        "share_pct": None,
        "what": "the completion leg's own broken S̄_φ pairing",
        "status": "RESOLVED (the fused fix; O6/O7/O8, rows #158/#161/#165)",
        "row": "#150 / #155",
        "quote": "bias_matched = −0.0846, per-seed sd 0.0329, SEM 0.0095 ⇒ MATCHED-INCONSISTENT",
    },
]

RESTORED_ARM_NULL = {
    "value": 0.0076,
    "sem": 0.0184,
    "sigma": 0.41,
    "row": "#155",
    "quote": "S̄₁₅ = +0.0076 ± 0.0184 (0.41σ) ⇒ PAIRING-OWNS-IT",
    "what": (
        "Restoring the registered arm (the S̄_φ zero-extension the executed "
        "arm had dropped) nulls the matched-channel score — the mechanism "
        "measurement, not a fourth contribution."
    ),
}

MECHANISM = {
    "row": "#155",
    "cause": (
        "the off-cell completion numerator (B_num) omits the S̄_φ survival "
        "factor its normalizer (β̄_G_φ) carries — the legacy pre-#118 cell "
        "every run-of-record stood on"
    ),
    "confirmation_rows": [
        "#158 (O6, +1.94e-6, 50× inside band)",
        "#161 (O7)",
        "#165 (O8, +0.00589±0.01078)",
    ],
    "fix": (
        'selection_in_completion_numerator = "fused" (B_num carries S̄_φ) — '
        'the pin for future runs; historical runs-of-record stand on "off"'
    ),
}


def build_decomposition() -> dict[str, Any]:
    naive_sum = rnd(sum(c["value_precise"] for c in CONTRIBUTIONS), 4)
    return {
        "chapter": "ch10x",
        "widget": "I10X.1 The Decomposition Bench",
        "headline_bias": HEADLINE_BIAS,
        "headline_quote": HEADLINE_QUOTE,
        "vocabulary_rule": (
            '"contribution" for this three-way split (never "channel" — '
            '"channel" is reserved for the full/matched/pure readouts of '
            "§2). docs/PRIMER_BIAS_CHANNELS_20260822.md §3."
        ),
        "contributions": CONTRIBUTIONS,
        "naive_sum": naive_sum,
        "sum_note": (
            f"The three contributions are independently measured, "
            f"control-referenced quantities — not algebraic terms of one "
            f"identity — so they need not sum to the headline exactly. "
            f"Naive sum here: {naive_sum:+.4f} vs headline {HEADLINE_BIAS:+.4f} "
            f"(difference {rnd(naive_sum - HEADLINE_BIAS, 4):+.4f})."
        ),
        "restored_arm_null": RESTORED_ARM_NULL,
        "mechanism": MECHANISM,
        "source": {
            "primer": PRIMER_REL,
            "ledger": LEDGER_REL,
            "ledger_rows": "#149-#157",
            "method": (
                "hand-authored from quoted ledger rows, precedent = gen_ch11.py's "
                "I11.2 BOARD table; every value re-grep-able at the cited row"
            ),
        },
    }


# ==========================================================================
# I10X.2 — the twin thread's verdict timeline
# ==========================================================================
# One entry per ledger row (dated) or pre-registration section header
# (PREREGISTRATION_P3_TWIN_20260822.md / PREREGISTRATION_B0_IDENTITY_20260823.md).
# `kind` follows the design entry's taxonomy exactly:
#   registration | amendment | refutation | ratification | measurement | verdict
TIMELINE: list[dict[str, Any]] = [
    {
        "id": "t01",
        "date": "2026-08-21",
        "kind": "measurement",
        "row": "#151",
        "label": (
            "C-SG v3 executed (46/46): matched-channel violation non-zero at "
            "6.05σ — BAND C = INTERNAL-DEFECT (provisional)"
        ),
    },
    {
        "id": "t02",
        "date": "2026-08-21",
        "kind": "refutation",
        "row": "#152",
        "label": (
            "Author-requested adversarial review: 2 FATAL findings against "
            "row #151's own presentation; INTERNAL-DEFECT label downgraded to "
            "PROVISIONAL"
        ),
    },
    {
        "id": "t03",
        "date": "2026-08-21",
        "kind": "registration",
        "row": "#153",
        "label": "O4 registered: the fired branch's own falsifier (A19), retrofit",
    },
    {
        "id": "t04",
        "date": "2026-08-21",
        "kind": "refutation",
        "row": "#155",
        "label": (
            "O4 executed and OVERTURNED by its own A19 falsifier: the "
            "executed arm VOID-BY-DEVIATION; restoring the registered arm "
            "nulls the score (+0.0076±0.0184) — mechanism identified"
        ),
    },
    {
        "id": "t05",
        "date": "2026-08-21",
        "kind": "ratification",
        "row": "#157",
        "label": (
            "Author ruling: defect label RATIFIED (implementation-convention "
            "defect); fused end-to-end confirmation seed APPROVED"
        ),
    },
    {
        "id": "t06",
        "date": "2026-08-21",
        "kind": "measurement",
        "row": "#158",
        "label": "O6 executed clean under A21: MECHANISM-CONFIRMED, delta +1.94e-6 (50× inside band)",
    },
    {
        "id": "t07",
        "date": "2026-08-22",
        "kind": "registration",
        "row": "#161",
        "label": "[P3-IMP] opened stage 2: the catalogue-leg twin cell built",
    },
    {
        "id": "t08",
        "date": "2026-08-22",
        "kind": "measurement",
        "row": "#162",
        "label": (
            "[P3-IMP] measured: the twin recovers +0.0155±0.0037 of the "
            "headline (REPORT-BOUND, 12/12 positive, 4.2σ)"
        ),
    },
    {
        "id": "t09",
        "date": "2026-08-22",
        "kind": "measurement",
        "row": "#164",
        "label": (
            "SHAPE-NULL banked: ~94–98% of the twin's effect is per-event "
            "level suppression; residual h-tilt is null (+0.00057±0.00010)"
        ),
    },
    {
        "id": "t10",
        "date": "2026-08-22",
        "kind": "measurement",
        "row": "#165",
        "label": (
            "O8 banked: the fused-replica bias leg closes as a point estimate "
            "(+0.00589±0.01078); paired off→fused correction "
            "+0.0724±0.0051 (14.1σ)"
        ),
    },
    {
        "id": "t11",
        "date": "2026-08-22",
        "kind": "refutation",
        "row": "#168",
        "label": (
            'Derivation fight: Appendix A (the "completed pairing" R-rescale) '
            "REFUTED — an un-derived B_scale-class multiplier; the twin as "
            "measured is the derivation-coherent candidate; Appendix B PROPOSED"
        ),
    },
    {
        "id": "t12",
        "date": "2026-08-22",
        "kind": "ratification",
        "row": "#169",
        "label": (
            "Author ruling: Appendix B RATIFIED — the twin is the candidate "
            "of record, off-basis-conditional; fused-basis re-measurement + "
            "the b0 identity test granted"
        ),
    },
    {
        "id": "t13",
        "date": "2026-08-23",
        "kind": "measurement",
        "row": "#173",
        "label": (
            "TWIN-FUSED-MATERIAL banked: on its coherent basis the twin moves "
            "the venue headline +0.0291±0.0051 (12/12, 5.7σ; un-truncated "
            "+0.0634, censoring makes the verdict conservative)"
        ),
    },
    {
        "id": "t14",
        "date": "2026-08-23",
        "kind": "amendment",
        "row": "#173",
        "label": "Amendments 17–21 adopted (E-P3 denominator rule, A22 stamp-before-evaluate, quotation rules made binding)",
    },
    {
        "id": "t15",
        "date": "2026-08-23",
        "kind": "registration",
        "row": "#174 / PREREGISTRATION_B0_IDENTITY_20260823.md",
        "label": (
            "The b0 catalogued-host identity test REGISTERED through two "
            "adversarial review rounds; venue premise and odds constant both "
            "corrected pre-commit; NO arm has run"
        ),
    },
    {
        "id": "t16",
        "date": "2026-08-23",
        "kind": "amendment",
        "row": "PREREGISTRATION_B0_IDENTITY_20260823.md § PRE-EXECUTION DESIGN-REVIEW",
        "label": "Pre-execution design-review amendments PA-1…PA-10 (pre-commit; NO arm has run)",
    },
    {
        "id": "t17",
        "date": "2026-08-23",
        "kind": "amendment",
        "row": "PREREGISTRATION_B0_IDENTITY_20260823.md § IMPLEMENTATION-REVIEW",
        "label": "Implementation-review amendments PA-11…PA-14 (pre-commit; NO arm has run)",
    },
    {
        "id": "t18",
        "date": "2026-08-24",
        "kind": "verdict",
        "row": "#177",
        "label": (
            "b0 identity test EXECUTED AND ADJUDICATED: UNDISCRIMINATING — "
            "the registration's own B-R control caught heavy-tail band "
            "vacuity; twin neither confirmed nor refuted; the 11/11 directional "
            "read later RETIRED as deterministic ordering (row #180); the "
            "12/12 mean-h read REPORTED-ONLY"
        ),
    },
    {
        "id": "t19",
        "date": "2026-08-24",
        "kind": "ratification",
        "row": "#178",
        "label": (
            'Author ruling: "decisions approved" — UNDISCRIMINATING ratified '
            'as adjudicated; Ch 10½ ("The Anatomy of the Bias") builds'
        ),
    },
    {
        "id": "t20",
        "date": "2026-08-24",
        "kind": "refutation",
        "row": "#180",
        "label": (
            "Gate-B adjudication of the successor-statistic draft: F-0 (intake-filter "
            "conditioning, 41.8%, outside the blindness list) CONFIRMED; the 11/11 "
            "directional read RETIRED (deterministic ordering, sign test void); the "
            "successor statistic (bounded-transform family) becomes the next registration"
        ),
    },
]

KIND_ORDER = ("registration", "amendment", "refutation", "ratification", "measurement", "verdict")


def build_timeline() -> dict[str, Any]:
    kinds_present = sorted({e["kind"] for e in TIMELINE})
    unknown = [k for k in kinds_present if k not in KIND_ORDER]
    if unknown:
        msg = f"gen_ch10x: unexpected timeline kind(s) {unknown}, not in {KIND_ORDER}"
        raise ValueError(msg)
    dates = [e["date"] for e in TIMELINE]
    if dates != sorted(dates):
        msg = "gen_ch10x: TIMELINE is not date-sorted — fix the literal order"
        raise ValueError(msg)
    return {
        "chapter": "ch10x",
        "widget": "I10X.2 Verdict Archaeology",
        "kind_order": list(KIND_ORDER),
        "kind_counts": {k: sum(1 for e in TIMELINE if e["kind"] == k) for k in KIND_ORDER},
        "events": TIMELINE,
        "source": {
            "prereg_twin": P3_TWIN_REL,
            "prereg_b0": B0_REL,
            "ledger": LEDGER_REL,
            "ledger_rows": "#149-#178",
            "method": (
                "one entry per dated ledger row or pre-registration section "
                "header; hand-authored, re-grep-able at the cited row/section"
            ),
        },
    }


# ==========================================================================
def write_json(name: str, payload: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    kb = path.stat().st_size / 1024
    print(f"  wrote {path.relative_to(BOOK_ROOT.parent)}  ({kb:.1f} KB)")
    if kb > 500:
        raise SystemExit(f"gen_ch10x: {name} exceeds the 500 KB budget ({kb:.1f} KB)")


def main() -> None:
    for rel in REQUIRED_SOURCES:
        need(rel)

    decomposition = build_decomposition()
    timeline = build_timeline()

    write_json("ch10x_decomposition.json", decomposition)
    write_json("ch10x_timeline.json", timeline)

    print(
        f"  I10X.1: headline {decomposition['headline_bias']:+.4f}, "
        f"naive sum {decomposition['naive_sum']:+.4f}, "
        f"restored-arm null {RESTORED_ARM_NULL['value']:+.4f}"
        f"±{RESTORED_ARM_NULL['sem']}"
    )
    print(
        f"  I10X.2: {len(TIMELINE)} events, "
        f"{timeline['events'][0]['date']} → {timeline['events'][-1]['date']}, "
        f"kinds {timeline['kind_counts']}"
    )


if __name__ == "__main__":
    main()
