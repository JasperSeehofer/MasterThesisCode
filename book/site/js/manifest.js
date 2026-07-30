/*
 * manifest.js — the single source of truth for the book's chapter list.
 *
 * OWNED BY THE INTEGRATOR. Chapter agents must NOT edit this file: your
 * chapter is registered here by the integrator when it lands. Pages include
 * this script BEFORE js/book.js; Book.buildNav() renders the top nav from it,
 * so no chapter page ever hardcodes links to other chapters.
 *
 * status: "live" (rendered as a link) | "planned" (rendered greyed, no link).
 */
window.BOOK_CHAPTERS = [
  { file: "index.html",              short: "Contents",   title: "Contents",                                    status: "live" },
  { file: "ch00-two-numbers.html",   short: "Prologue",   title: "Two Numbers That Should Be One",              status: "planned" },
  { file: "ch01-ruler.html",         short: "Ch. 1",      title: "A Ruler That Needs No Ladder",                status: "planned" },
  { file: "ch02-bayes.html",         short: "Ch. 2",      title: "Bayes, Once and For All",                     status: "planned" },
  { file: "ch03-which-galaxy.html",  short: "Ch. 3",      title: "Which Galaxy?",                               status: "planned" },
  { file: "ch04-loud-half.html",     short: "Ch. 4",      title: "The Universe Only Shows You Its Loud Half",   status: "planned" },
  { file: "ch05-unseen-galaxy.html", short: "Ch. 5",      title: "The Galaxy You Cannot See",                   status: "planned" },
  { file: "ch06-black-box.html",     short: "Ch. 6",      title: "Opening the Black Box",                       status: "planned" },
  { file: "ch07-redshift.html",      short: "Ch. 7",      title: "A Redshift Is Not a Number",                  status: "planned" },
  { file: "ch08-mass-channel.html",  short: "Ch. 8",      title: "A Second Handle: the Mass Channel",           status: "planned" },
  { file: "ch09-universe-factory.html", short: "Ch. 9",   title: "Building a Universe to Break Your Estimator", status: "planned" },
  { file: "ch10-calibration.html",   short: "Ch. 10",     title: "Is It Calibrated?",                           status: "planned" },
  { file: "ch11-honest-state.html",  short: "Ch. 11",     title: "The State of the Art, Honestly",              status: "planned" },
  { file: "museum.html",             short: "Museum",     title: "The Defect Museum",                           status: "planned" },
];
