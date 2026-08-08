/*
 * manifest.js — the single source of truth for the book's chapter list.
 *
 * OWNED BY THE INTEGRATOR. Chapter agents must NOT edit this file: your
 * chapter is registered here by the integrator when it lands. Pages include
 * this script BEFORE js/book.js; Book.buildNav() renders the top nav from it,
 * so no chapter page ever hardcodes links to other chapters.
 *
 * status: "live" (rendered as a link) | "planned" (rendered greyed, no link).
 * All 13 chapters + the museum annex went live 2026-07-31 (integrator pass);
 * the legacy ch00-demo page was retired in the same pass. Ch. 12 (the living
 * bias-resolution record) added 2026-08-05, hand-authored, no generator.
 * Ch. 13 (the stage-0 proposal + open-decisions board) added 2026-08-07,
 * hand-authored, no generator; retitled 2026-08-08 when the thread-16
 * owner-hunt closed as a ratified dissolution (ch13 §A.7).
 */
window.BOOK_CHAPTERS = [
  { file: "index.html",              short: "Contents",   title: "Contents",                                    status: "live" },
  { file: "ch00-two-numbers.html",   short: "Prologue",   title: "Two Numbers That Should Be One",              status: "live" },
  { file: "ch01-ruler.html",         short: "Ch. 1",      title: "A Ruler That Needs No Ladder",                status: "live" },
  { file: "ch02-bayes.html",         short: "Ch. 2",      title: "Bayes, Once and For All",                     status: "live" },
  { file: "ch03-which-galaxy.html",  short: "Ch. 3",      title: "Which Galaxy?",                               status: "live" },
  { file: "ch04-loud-half.html",     short: "Ch. 4",      title: "The Universe Only Shows You Its Loud Half",   status: "live" },
  { file: "ch05-unseen-galaxy.html", short: "Ch. 5",      title: "The Galaxy You Cannot See",                   status: "live" },
  { file: "ch06-black-box.html",     short: "Ch. 6",      title: "Opening the Black Box",                       status: "live" },
  { file: "ch07-redshift.html",      short: "Ch. 7",      title: "A Redshift Is Not a Number",                  status: "live" },
  { file: "ch08-mass-channel.html",  short: "Ch. 8",      title: "A Second Handle: the Mass Channel",           status: "live" },
  { file: "ch09-universe-factory.html", short: "Ch. 9",   title: "Building a Universe to Break Your Estimator", status: "live" },
  { file: "ch10-calibration.html",   short: "Ch. 10",     title: "Is It Calibrated?",                           status: "live" },
  { file: "ch11-honest-state.html",  short: "Ch. 11",     title: "The State of the Art, Honestly",              status: "live" },
  { file: "ch12-bias-resolution.html", short: "Ch. 12",   title: "The Bias Resolution, a Live Thread",          status: "live" },
  { file: "ch13-unowned-residual.html", short: "Ch. 13",  title: "The Unowned Residual: Measured and Closed",   status: "live" },
  { file: "museum.html",             short: "Museum",     title: "The Defect Museum",                           status: "live" },
];

/*
 * Canonical predict-id registry (WIDGET_REQUESTS R-ch11-2).
 *
 * Cross-chapter recall beats (Book.getPrediction) resolve ids by name from
 * this table instead of guessing strings; the integrator's link check
 * verifies every id read somewhere is written somewhere. Keys are the ids
 * actually written by the shipped chapters (2026-07-31 inventory).
 */
window.BOOK_PREDICT_IDS = {
  ch00SysBudget:   "ch00-q-sys",
  ch02Concentration: "ch02-concentration",
  ch03BallSize:    "ch03-ball-size",
  ch03HostGuess:   "ch03-host-guess",   // re-surfaced by Ch 11's payoff beat
  ch04MapGuess:    "ch04-map-guess",    // re-surfaced by Ch 11
  ch05ClassGuess:  "ch05-class-guess",
  ch05SkyGuess:    "ch05-sky-guess",
  ch06Dt2Guess:    "ch06-dt2-guess",
  ch07Q1:          "ch07-q1",
  ch07Q2:          "ch07-q2",
  ch08SecondHandle: "ch08-second-handle",
  ch08SieveShape:  "ch08-sieve-shape",
  ch09Bench:       "ch09-q-bench",
  ch09Identity:    "ch09-q-identity",
  ch10Calibrated:  "ch10-calibrated",
  ch10TopK:        "ch10-topk",
  ch11CellB:       "ch11-cellb-guess",
  ch11Leverage:    "ch11-leverage-guess",
  musQuadrature:   "mus-quadrature-guess",
};

/*
 * ===================================================================
 * BOOK_CANON — the book's canonical shared strings (ONE definition)
 * ===================================================================
 * REVISION_WORKLIST.md §D item 6, integrator pass 1, 2026-07-31.
 *
 * These strings are quoted VERBATIM by every page that carries them.
 * Chapter agents must copy from here (or read `Book.canon` at runtime);
 * they must not re-word them locally. `book/generators/qa_gates.py`
 * greps the built pages against this object, so a local re-wording is a
 * build failure, not a style difference.
 *
 * Why here and not in book.js: these are DATA (spec-fixed text), and
 * manifest.js is the file both the site and the QA gate parse. The gate
 * extracts the values by key with a regex — keep every value on ONE
 * line, in double quotes.
 */
window.BOOK_CANON = {
  /* --- D1: the sigma_dL units slip, resolved book-wide 2026-07-31 -----
   * Spec value is now the six-chapter MEASURED value. The old spec figure
   * 8.0e-5 was the ABSOLUTE sigma_dL in Gpc carried under a fractional
   * label (a x11.25 slip). Pages print the corrected value once, plus the
   * one-line erratum note; the flag files remain the historical record.
   */
  sigmaDL: {
    fractional: "8.98×10⁻⁴",
    absolute: "7.98×10⁻⁵ Gpc",
    // the dossier row, identical on every dossier card ch01-ch11 + museum
    dossierRow: "d_L  88.9 Mpc  ·  σ_dL/d_L = 8.98×10⁻⁴",
    // the same row as the book's dossier-table markup (entity-escaped)
    dossierRowHTML: "<tr><td>d_L</td><td>88.9 Mpc &middot; &sigma;_dL/d_L = 8.98&times;10&#8315;&#8308;</td></tr>",
    // the erratum line — a footnote or small .note, NOT a boxed OPEN dispute
    erratum: "Erratum: the spec card carried σ_dL/dL = 8.0×10⁻⁵ — that is the absolute σ_dL in Gpc under a fractional label. Corrected book-wide 2026-07-31; record: ch01 flag F1 / BUILD_REPORT §5.1 item 1.",
    erratumHTML: "<p class=\"note\">Erratum: the spec card carried &sigma;_dL/dL = 8.0&times;10&#8315;&#8309; &mdash; that is the absolute &sigma;_dL in Gpc under a fractional label. Corrected book-wide 2026-07-31; record: ch01 flag F1 / BUILD_REPORT &sect;5.1 item 1.</p>",
  },

  /* --- D3: the 2x2 cell B landed 2026-07-31 --------------------------- */
  cellB: {
    // the canonical rail pip — identical wording on ch07 / ch09 / ch10 / ch11
    pipLabel: "cell B (2026-07-31): estimator owns +0.060 of the 2D +0.083",
    pipNote: "CELLB_READOUT_20260731.md — evaluate 6103219 / combine 6103220; the 2D-only share is 72%",
    pipTone: "amber",
    // the job-ID split rule (D3): pre-registration keeps the registered IDs,
    // results cite the resubmission, with the one-sentence note.
    jobsPrereg: "6101146 / 6101147",
    jobsResult: "6103219 / 6103220",
    jobIdRule: "Where the pre-registration is quoted, the job IDs stay 6101146/6101147; where the result is reported, cite 6103219/6103220 and carry the resubmission note once.",
    resubmissionNote: "Jobs 6103219/6103220 are the resubmission of 6101146/6101147 after a pure-plumbing symlink failure in the run-dir setup; the test design and the pre-registration are unchanged, and the code is the same commit (7fd60bb) as cells A and C.",
    // the naming rule (MJ-3): the 2026-07-31 2x2 object is always "the 2x2 cell B"
    naming: "the 2×2 cell B",
  },
};

/*
 * ===================================================================
 * BOOK_BIAS_ROWS — the cumulative bias-rail history (§D item 4, ped M7)
 * ===================================================================
 * Integrator pass 2, 2026-07-31.  The bias rail is the book's continuity
 * spine, but each chapter used to declare only its own rows, so the rail
 * FORGOT rows moving forward (ch05 showed two, ch11 showed two again).
 * These are the book-wide rows; Book.biasRail merges them into every
 * page's own spec:
 *
 *   - a row renders on every page whose chapter number n satisfies
 *     from_chapter <= n (index/museum render all of them);
 *   - if the page already declares an equivalent row (any `match`
 *     substring found in the page row's label, case-insensitive), the
 *     page's row wins — its wording, note, `active` state and arming
 *     pattern (ch08 arms its 2D row only at the cold-open reveal) are
 *     the chapter agent's;
 *   - rows a page does not declare are rendered from here, inactive.
 *
 * `from_chapter` is a D4 (spoiler-discipline) boundary: it is the first
 * chapter where the row's value is no longer that chapter's own reveal
 * (the 2D +0.077 row is ch08's reveal, so it becomes unconditional only
 * from ch09; ch08 itself arms it page-locally at the reveal moment).
 * Values are venue-scoped exactly as the chapters state them.
 */
window.BOOK_BIAS_ROWS = [
  {
    from_chapter: 4,
    label: "cat-only, no D(h)",
    bias: -0.178,
    note: "Phase 32 / ledger #9 — MAP 0.60, the bottom of the prior",
    match: ["cat-only", "no d(h)", "no selection"],
  },
  {
    from_chapter: 4,
    label: "full-volume D(h)",
    bias: 0.0,
    note: "Phase 32 / ledger #9 — MAP 0.73. Venue-scoped: campaign #51/#53 r1 reads MAP 0.740.",
    match: ["full-volume"],
  },
  {
    from_chapter: 8,
    label: "1D, realistic host-z (volume_deconv)",
    bias: -0.002,
    note: "G2b §2.3 — the σ_z-independent residual floor the volume estimator retains (−0.0014…−0.0030 across the archive-gated control cells)",
    match: ["volume_deconv", "realistic host-z"],
  },
  {
    from_chapter: 9,
    label: "2D, mass channel",
    bias: 0.077,
    note: "campaign #53, 10 runs: mean pull +4.04, 10/10 beyond 2σ — RATIFY-M6 CANDIDATE ground",
    match: ["2d"],
  },
  {
    from_chapter: 11,
    label: "1D (contingent)",
    bias: 0.0,
    note: "Not a clean zero: the 1D headline is the crossing of two railed opposing runaways (in-cat 0.86 / dark 0.64). C5, FINDING.",
    match: ["contingent"],
  },
];
