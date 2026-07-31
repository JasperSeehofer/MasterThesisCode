"""Build gates for the book (REVISION_WORKLIST.md §D item 12).

Four hard gates, run by ``make_all.py`` after every generator and
runnable on its own::

    /home/jasper/Repositories/MasterThesisCode/.venv/bin/python \\
        book/generators/qa_gates.py

They are content gates, not style gates: each one encodes a statement the
book makes about itself that a stale string can silently falsify.

  D1  the retired sigma_dL value (8.0e-5 carried as a *fraction*) may
      appear only inside an erratum note.  Spec value book-wide is
      sigma_dL/d_L = 8.98e-4 (absolute sigma_dL = 7.98e-5 Gpc).
      -- REVISION_WORKLIST §A-D1.
  ROW every parsed museum-ledger row keeps its seven source cells, and
      `documented` really holds a citation (row #68's unescaped pipes
      shifted its cells one to the left and put the word "tilt" in the
      citation column).  -- §C-museum expA-M4.
  DNR the do-not-re-try union is 30 rows once the §2 back-reference
      separator class is fixed (26 today).  -- §C-museum expA-M3.
  TNS no page still says cell B has "not landed" / is "in flight" /
      "still running".  It landed 2026-07-31.  -- §D-12(d).

Escape hatch.  A line may carry ``qa-allow: <gate-id>`` (inside an HTML or
JS comment) when the string is deliberate and historically necessary --
e.g. a pre-registration block quoted verbatim under D3's verbatim rule.
Gate ids are ``sigma-dl``, ``ledger-row``, ``dnr-count``, ``cellb-tense``.
Use it sparingly: it is a claim that the string is history, not state.

Ownership: integrator file.  Chapter agents do not edit it; they make
their pages pass it.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

BOOK_DIR = Path(__file__).resolve().parent.parent
SITE = BOOK_DIR / "site"
DATA = SITE / "data"

# _template.html is on the worklist's frozen list (§E "Frozen"), and it
# still carries the retired dossier row.  It is therefore reported as a
# NOTE, never as a failure: granting the frozen-list exception is the
# integrator-pass-2 / author call, not this gate's.
FROZEN_FILES = {"_template.html"}


@dataclass
class Violation:
    """One gate hit: where it is, and what is wrong with it."""

    gate: str
    path: Path
    line: int
    text: str

    def render(self) -> str:
        rel = self.path.relative_to(BOOK_DIR.parent)
        return f"  {rel}:{self.line}  {self.text}"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def scanned_pages() -> list[Path]:
    """Shipped HTML pages (frozen files excluded, reported separately)."""
    return sorted(p for p in SITE.glob("*.html") if p.name not in FROZEN_FILES)


def scanned_files() -> list[Path]:
    """Everything a reader can reach: pages, data, shared JS."""
    return scanned_pages() + sorted(DATA.glob("*.json")) + sorted((SITE / "js").glob("*.js"))


def line_of(text: str, offset: int) -> int:
    """1-indexed line number of a character offset."""
    return text.count("\n", 0, offset) + 1


def line_text(text: str, offset: int) -> str:
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    return text[start: end if end != -1 else len(text)]


def allowed(text: str, offset: int, gate_id: str) -> bool:
    """True if this line carries the gate's explicit escape hatch."""
    return f"qa-allow: {gate_id}" in line_text(text, offset)


def window(text: str, offset: int, span: int) -> str:
    return text[max(0, offset - span): offset + span]


# ----------------------------------------------------------------------
# D1 -- the retired sigma_dL value
# ----------------------------------------------------------------------
# Every encoding the book actually uses for "8.0 x 10^-5".  The exponent
# is pinned to -5 on purpose: ch06 legitimately prints 8.0e12 for a
# condition number, and that must not trip the gate.
SIGMA_DL_PATTERNS = [
    re.compile(r"8\.0\s*\\times\s*10\^\{?\s*-\s*5\s*\}?"),          # KaTeX
    re.compile(
        r"8\.0\s*(?:&times;|×)\s*10\s*"
        r"(?:<sup>\s*(?:&minus;|&#8722;|−|-)\s*5\s*</sup>"           # <sup>-5</sup>
        r"|⁻⁵|&#8315;&#8309;"                                        # superscript glyphs
        r"|\^\s*-?\s*5)"                                             # 10^-5
    ),
    re.compile(r"8\.0\s*[eE]-0?5(?![0-9])"),                        # 8.0e-5 / 8.0e-05
]


def gate_sigma_dl() -> list[Violation]:
    """D1: the old value is legal only inside an erratum note."""
    out: list[Violation] = []
    for path in scanned_files():
        text = path.read_text(encoding="utf-8")
        for pat in SIGMA_DL_PATTERNS:
            for m in pat.finditer(text):
                if allowed(text, m.start(), "sigma-dl"):
                    continue
                if "erratum" in window(text, m.start(), 400).lower():
                    continue
                out.append(
                    Violation(
                        "D1",
                        path,
                        line_of(text, m.start()),
                        f"live retired value {m.group(0)!r} with no erratum note "
                        f"in scope (spec value is now sigma_dL/d_L = 8.98e-4)",
                    )
                )
    return out


def frozen_notes() -> list[str]:
    """Frozen-file carriers of the retired value -- reported, not failed."""
    notes: list[str] = []
    for name in sorted(FROZEN_FILES):
        path = SITE / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for pat in SIGMA_DL_PATTERNS:
            for m in pat.finditer(text):
                if "erratum" in window(text, m.start(), 400).lower():
                    continue
                notes.append(
                    f"  {path.relative_to(BOOK_DIR.parent)}:{line_of(text, m.start())}  "
                    f"{m.group(0)!r} -- FROZEN file (worklist §E); needs an explicit "
                    f"frozen-list exception before it can be corrected"
                )
    return notes


# ----------------------------------------------------------------------
# museum ledger -- row shape and the do-not-re-try count
# ----------------------------------------------------------------------
LEDGER = DATA / "museum_ledger.json"

# the seven source cells of BIAS_HISTORY_LEDGER.md's table
LEDGER_CELLS = ("id", "era", "hypothesis", "test", "verdict", "documented", "residual")

# a citation looks like "H0R:1973", "foo.md:24-33", "G2c §2", "ledger #26"
CITATION = re.compile(r"(:\d|\.md|\.py|\.json|§|#\d)")

EXPECTED_DNR_ROWS = 30


def _ledger_rows() -> list[dict]:
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    return payload["rows"] if isinstance(payload, dict) else payload


def gate_ledger_rows() -> list[Violation]:
    """ROW: seven cells, all present, and `documented` is a citation."""
    if not LEDGER.exists():
        return [Violation("ROW", LEDGER, 0, "museum_ledger.json missing")]
    out: list[Violation] = []
    for row in _ledger_rows():
        rid = row.get("id", "?")
        missing = [c for c in LEDGER_CELLS if not str(row.get(c, "")).strip()]
        if missing:
            out.append(
                Violation("ROW", LEDGER, 0, f"row #{rid}: empty source cell(s) {missing}")
            )
        doc = str(row.get("documented", ""))
        if doc and not CITATION.search(doc):
            out.append(
                Violation(
                    "ROW",
                    LEDGER,
                    0,
                    f"row #{rid}: documented={doc!r} is not a citation -- the row's "
                    f"cells are shifted (unescaped '|' in the source table)",
                )
            )
    return out


def gate_dnr_count() -> list[Violation]:
    """DNR: the do-not-re-try union is 30 rows, in the data and in prose."""
    if not LEDGER.exists():
        return [Violation("DNR", LEDGER, 0, "museum_ledger.json missing")]
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) else payload
    out: list[Violation] = []
    flagged = sum(1 for r in rows if r.get("do_not_retry"))
    listed = len(payload.get("do_not_retry_rows", [])) if isinstance(payload, dict) else flagged
    if flagged != EXPECTED_DNR_ROWS:
        out.append(
            Violation(
                "DNR", LEDGER, 0,
                f"{flagged} rows flagged do_not_retry, expected {EXPECTED_DNR_ROWS} "
                f"(§2's back-reference separator class must accept ',', '/', '·', ';' "
                f"-- recovers #41/#43/#44/#52)",
            )
        )
    if listed != EXPECTED_DNR_ROWS:
        out.append(
            Violation("DNR", LEDGER, 0,
                      f"do_not_retry_rows has {listed} entries, expected {EXPECTED_DNR_ROWS}")
        )
    # ... and no page may still print the old row count next to the phrase
    stale = re.compile(r"\b26\b[^.]{0,80}do[- ]?not[- ]?re[- ]?try"
                       r"|do[- ]?not[- ]?re[- ]?try[^.]{0,80}\b26\b(?!\d)", re.I)
    for path in scanned_pages():
        text = path.read_text(encoding="utf-8")
        for m in stale.finditer(text):
            if allowed(text, m.start(), "dnr-count"):
                continue
            out.append(
                Violation("DNR", path, line_of(text, m.start()),
                          f"prose still prints 26 do-not-re-try rows: {m.group(0)[:70]!r}")
            )
    return out


# ----------------------------------------------------------------------
# cell-B tense
# ----------------------------------------------------------------------
STALE_TENSE = re.compile(r"not landed|in flight|still running|has not landed", re.I)
CELLB_MARKER = re.compile(r"cell[ -]?B|2×2|2x2|610114[67]|610321[90]", re.I)


def gate_cellb_tense() -> list[Violation]:
    """TNS: nothing static may let the landed control look unresolved."""
    out: list[Violation] = []
    for path in scanned_pages():
        text = path.read_text(encoding="utf-8")
        for m in STALE_TENSE.finditer(text):
            if allowed(text, m.start(), "cellb-tense"):
                continue
            if not CELLB_MARKER.search(window(text, m.start(), 500)):
                continue
            out.append(
                Violation("TNS", path, line_of(text, m.start()),
                          f"{m.group(0)!r} next to a cell-B reference -- cell B landed "
                          f"2026-07-31 (jobs 6103219/6103220)")
            )
    return out


# ----------------------------------------------------------------------
# advisory: the canonical cell-B rail pip (§D item 6)
# ----------------------------------------------------------------------
CANON_JS = SITE / "js" / "manifest.js"


def canon(key_path: str) -> str | None:
    """Read one BOOK_CANON string out of manifest.js (its one definition)."""
    if not CANON_JS.exists():
        return None
    text = CANON_JS.read_text(encoding="utf-8")
    key = key_path.rsplit(".", 1)[-1]
    m = re.search(rf'^\s*{re.escape(key)}:\s*"((?:[^"\\]|\\.)*)"', text, re.M)
    return m.group(1) if m else None


def advisory_pip_wording() -> list[str]:
    """Report-only: every rail pip naming cell B should be the canonical one.

    Placing the four pips is integrator pass 2's item; this only tells that
    pass whether the wording has drifted.
    """
    want = canon("cellB.pipLabel")
    notes: list[str] = []
    if not want:
        return ["  manifest.js: BOOK_CANON.cellB.pipLabel not found"]
    pip = re.compile(r'label:\s*"([^"]*cell[ -]?B[^"]*)"', re.I)
    for path in scanned_pages():
        text = path.read_text(encoding="utf-8")
        for m in pip.finditer(text):
            if m.group(1) != want:
                notes.append(
                    f"  {path.relative_to(BOOK_DIR.parent)}:{line_of(text, m.start())}  "
                    f"pip label {m.group(1)!r} != canonical"
                )
    return notes


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
GATES = (
    ("D1  retired sigma_dL value outside an erratum note", gate_sigma_dl),
    ("ROW museum ledger row shape (7 cells, real citation)", gate_ledger_rows),
    ("DNR do-not-re-try union == 30 rows", gate_dnr_count),
    ("TNS cell-B tense ('not landed' / 'in flight')", gate_cellb_tense),
)


def run() -> int:
    """Run every gate; print a report; return the number of violations."""
    print("=== book QA gates (REVISION_WORKLIST §D item 12) ===")
    total = 0
    for title, fn in GATES:
        hits = fn()
        total += len(hits)
        status = "PASS" if not hits else f"FAIL ({len(hits)})"
        print(f"[{status:9s}] {title}")
        for v in hits:
            print(v.render())
    notes = frozen_notes()
    if notes:
        print("[NOTE     ] retired sigma_dL value in FROZEN files (not a failure):")
        for n in notes:
            print(n)
    pips = advisory_pip_wording()
    if pips:
        print("[ADVISORY ] cell-B rail pip wording (integrator pass 2 places these):")
        for n in pips:
            print(n)
    print(f"=== {total} gate violation(s) ===")
    return total


def main() -> None:
    raise SystemExit(1 if run() else 0)


if __name__ == "__main__":
    main()
