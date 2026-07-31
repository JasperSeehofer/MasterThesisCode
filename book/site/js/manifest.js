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
 * the legacy ch00-demo page was retired in the same pass.
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
