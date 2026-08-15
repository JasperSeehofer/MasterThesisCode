# Review — Physics Expert A · ch00–ch06 + museum

**Reviewer:** Physics Expert A (source-fidelity audit)
**Scope:** `ch00-two-numbers.html`, `ch01-ruler.html`, `ch02-bayes.html`,
`ch03-which-galaxy.html`, `ch04-loud-half.html`, `ch05-unseen-galaxy.html`,
`ch06-black-box.html`, `museum.html`, and their generators/data.
**Method:** every equation, number and attribution checked against
`docs/derivations/*`, `CLAIM_2D_BIAS_20260730.md` (as amended),
`gate_b_20260730/{ADJUDICATION,BIAS_HISTORY_LEDGER}.md`, `IDEALIZED_BASELINE_READOUT.md`,
`REALISTIC_READOUT.md`, `docs/gates/G7_systematics_budget.md`, and the code. ≥3 numeric
gates per chapter re-derived independently with
`/home/jasper/Repositories/darksiren-emri/.venv/bin/python` against the raw artifacts
(`prepared_cramer_rao_bounds.csv`, `real_r1/diagnostics/event_likelihoods.csv`,
`posteriors_fixed/`, `zoom/posteriors/`, `mixture_leg_log_extract.txt`,
`injection_pool_mix200k_20260728/`, `m_th_map_nside32.npy`, `gate_result.json`).
12 museum ledger rows diffed cell-by-cell against `BIAS_HISTORY_LEDGER.md`.
**Read-only throughout. No git.**

Headline: the arithmetic in this half of the book is, with the exceptions below,
**exceptionally good** — I could not break ch01, ch02, ch04 or ch06 on any number I
re-derived. The failures that exist are (a) one wrong production constant in ch03 that
propagates through a whole census and contradicts ch06, (b) three pages that quietly ship
the disputed σ_dL label the rest of the book was built to catch, and (c) two transcription
defects in the museum's ledger digest that break the museum's own binding meta-rule.

---

## [BLOCKER]

### B1 · Ch 3 §1 "The ball search" (`ch03-which-galaxy.html:199`, `<h2>1. The candidate ball</h2>`) — the search-radius σ-multiplier is wrong, and the whole candidate census is built on it

**Observed.** The RATIFIED-badged equation box prints

```
r = n_σ √λ_max(Σ'),   n_σ = 2
```

cited to `handler.py:519`, and calls it "the production ball rule". The generator hard-codes
it: `book/generators/gen_ch03.py:160` → `SIGMA_MULTIPLIER = 2  # handler.get_possible_hosts_from_ball_tree default`,
and `ch03_candidates.json.meta.ball_rule` says `"handler.py:519 r = 2*sqrt(lambda_max(J Sigma J^T))"`.

**Expected.** `2` is only the *signature default* of `get_possible_hosts_from_ball_tree`.
The production evaluate path never uses it — `bayesian_statistics.py:2838` passes
`sigma_multiplier=1.5` explicitly (I read the call site; it sits directly under the
`get_redshift_outer_bounds(..., sigma_multiplier=2.0)` call, which is a *different*
multiplier for a *different* cut, and is very likely how the two got crossed).
Ch 6 has this right: `gen_ch06.py:165-166` → `# Production BallTree call (bayesian_statistics.py:2837): sigma_multiplier=1.5`,
`ch06_fisher.json.meta.sigma_multiplier = 1.5`, and ch06's prose says
*"Production passes n_σ = 1.5."*

**Consequence — measured.** The two chapters now disagree about the same event on the same
data. I recomputed the radius from row 889's sky block by hand:

| | n_σ = 2 (ch03) | n_σ = 1.5 (ch06, production) |
|---|---|---|
| 889 ball radius | 1.00902′ (0.016817°) | 0.757′ |
| solid angle | 8.885×10⁻⁴ deg² | 3.29×10⁻⁴ deg² |
| galaxies in ball | **3** | **2** |

Ch 3 §1 says *"Inside that ball sit three catalogue galaxies"*; ch 6's cold-open table says
*"889 … 3.29×10⁻⁴ deg² · 2 · 2"*. Both are labelled real, neither cross-references the other.

Everything downstream of ch03's radius is at the wrong scale (area is off by (4/3)² = 1.78):
the census (median 1616 in ball / 12 after window; 95th 4891; max 431,670; 79 empty balls;
**552 of 1590 zero-candidate**, hence "1038 of 1590"), the featured extreme (1121:
57,535 → 26,652), the concentration statistics (52.6% > ½, 35.3% > 0.9, n_eff median 2.34),
Q3.4's answer, the ch03 dossier, and `ch03_FLAGS.md` F-ch03-2 / F-ch03-10 / F-ch03-12.
Ch 3's 552 is also cited forward into Ch 4 §5 and Ch 5 §4 framing.

**Suggested fix.** Set `gen_ch03.SIGMA_MULTIPLIER = 1.5`, regenerate, fix the equation box,
the `ball_rule` string, and the three flag entries; then add one sentence to §1 pinning the
call site (`bayesian_statistics.py:2838`, not the signature default) so this cannot recur.
If the author *wants* the 2σ census as a deliberate "what does a wider ball hold" exercise,
that is defensible — but it must be labelled as a counterfactual, not as "the production
ball rule", and ch06's 889 row must be reconciled on the page.

### B2 · `ch04-loud-half.html:627`, `ch05-unseen-galaxy.html:863`, `museum.html:1316` — the disputed "σ_dL/dL = 8.0×10⁻⁵" ships uncorrected in three dossiers

**Observed.** Each of these three pages prints the running dossier row as

> `d_L | 88.9 Mpc (σ_dL/dL = 8.0×10⁻⁵)`

and — I grepped all three — **neither `7.98…` nor `8.98…` appears anywhere on those pages.**
The generator propagates the mislabel too: `ch04_denominator.json` carries
`event889.sigma_dL_over_dL = 7.98e-05`, i.e. a key literally named "over_dL" holding the
absolute value in Gpc.

**Expected.** BUILD_REPORT §5.1 item 1 states *"Chapters print both values everywhere"*, and
the reviewer's guide asks explicitly that no page quietly prefer one. `ch03_FLAGS.md`
F-ch03-1 even records that `ch04-loud-half.html` ships the bad string. Ch 1 §2, Ch 2's
dossier, Ch 3's dossier and Ch 6 §4 all do carry both — Ch 6 goes further and gives two
independent arbitration checks.

**Re-verified independently.** Row 889 of `seed61000/prepared_cramer_rao_bounds.csv`:
`sqrt(delta_luminosity_distance_delta_luminosity_distance)` = **7.984273×10⁻⁵**,
`luminosity_distance` = **0.088879221 Gpc** ⇒ σ_dL/d_L = **8.9833×10⁻⁴** = 1.280/ρ at
ρ = 1424.7236 (also the loudest of 1590 rows, `argmax(SNR) = 889` — confirmed).

**Why this is a blocker rather than a typo.** Ch 1 §2 and Ch 6 §4 each spend a full boxed
adjudicator block teaching the reader to catch exactly this unit slip, and Q1.2's answer
turns on it. Three pages then repeat it in the object the reader is told to track across the
whole book — including the **museum**, whose stated contract is that nothing in it lets a
dead reading look alive.

**Suggested fix.** One-line edit ×3 to the format Ch 6 already uses
(`σ_dL = 7.98×10⁻⁵ Gpc = 0.0798 Mpc → as a fraction, 8.98×10⁻⁴`), plus rename the JSON key
in `gen_ch04.py` (`sigma_dL_Gpc`, with `sigma_dL_over_dL` holding 8.98e-4).

---

## [MAJOR]

### M1 · Ch 5 §2, adjudicator block "What the flatten-the-slope number is, and is not" + provenance panel — C10's attribution is garbled in the one place it matters

**Observed (page text).**

> "The completion likelihood 𝓛^comp itself pulls down for dark events (only **39.1%** of them
> have a positive completion tilt; **ΣΔln 𝓛^comp = −3.11** over the same window)."

and in the provenance panel: *"not by 𝓛^comp (−3.11 nats; 39.1% positive)"*.

**Measured** (r1 diagnostics CSV, window h = 0.73 → 0.81, my own recomputation):

| quantity | all 1588 | dark (1512) | in-cat (76) |
|---|---|---|---|
| ΣΔ ln 𝓛^comp | **−3.11** | **−22.72** | +19.61 |
| fraction of dark with positive tilt, 𝓛^comp alone | — | **27.71%** | — |
| fraction of dark with positive tilt, **(1−w_G)·𝓛^comp** | — | **39.09%** | — |
| Σ Δ ln[(1−w_G)𝓛^comp], dark | — | **+7.33** | — |

So the parenthesis (a) attaches the **all-event** total −3.11 to a sentence about **dark**
events (the dark sum is −22.72, 7.3× larger), and (b) attributes to `L_comp` alone a
percentage that C10 computes **with the prefactor included**. `ch08_FLAGS.md` F-ch08-6
already establishes exactly this scoping ("C10's 39.1% counts the sign of (1−w_G)·L_comp;
L_comp alone gives 27.7%"), which my number reproduces to the digit — so ch05 is
inconsistent with ch08 as well as with C10.

**Why it matters.** C10 exists *solely* to fix this attribution ("Any sentence of the form
'the completion term pulls up' must name the (1−w_G) prefactor, not L_comp"). A block whose
job is to enforce that rule should not itself mis-scope both of its numbers.

**Suggested fix.** "…dark ΣΔln 𝓛^comp = **−22.72** (all-event total −3.11); only **27.7%** of
dark events have a positive `L_comp` tilt — **39.1%** once the (1−w_G) prefactor is folded
in, which is the form C10 quotes."

### M2 · Ch 5 Q5.4 (`<h2>Self-check</h2>`) quotes the §5.1-flagged "1500–2400×" with no counterpart, against ch11's explicit written instruction

**Observed.** Q5.4's answer: *"with dh\*/dε leverage **1500–2400×**"*. Grep of
`ch05-unseen-galaxy.html`: one hit for `1500–2400`, **zero** for `141.8`, `2457.8`, `197`,
or any recomputed range.

**Expected.** `ch11_FLAGS.md` F-ch11-1, final bullet, addressed to other chapters:
*"Ch 5's I5.2 plants this same object and Q5.4's answer quotes '1500–2400×'. If Ch 5 quotes
the ratio it should carry this flag too, or quote the 0.025 Poisson figure instead, which
reproduces exactly."* BUILD_REPORT §5.1 item 3 lists this as one of the cross-chapter
disputes where both values must appear; Ch 11 handles it correctly, Ch 5 does not.

**Suggested fix.** Either drop the ratio and keep the Poisson figure (0.025 vs 0.0000
idealized, which Q5.4 already quotes and which reproduces to 1e-6), or add "(adjudicated;
recomputed from the adjudicator's own `c5_leverage_results.json` as 142–2458×, median 197× —
`ch11_FLAGS.md` F-ch11-1)".

### M3 · `museum.html` §7 ledger browser + census caption — the DO-NOT-RE-TRY flag set is short by 4 rows, and the page contradicts itself on click

**Observed.** `gen_museum.py:286` only accepts a §2 back-reference when a parenthesised
group matches `#\d+[a-z]?(\s*,\s*#\d+[a-z]?)*` — i.e. **comma-separated only**. The ledger's
§2 writes item 13 as `(#41/#52)` and item 15 as `(#43/#44)`, with slashes. Those four rows
are silently dropped:

- `museum_ledger.json.do_not_retry_rows` = 26 rows.
- §2 actually back-references **30 ledger rows** (32 `#N` tokens minus `#16` and `#29`, which
  are GitHub issue numbers — the guard correctly rejects those, and that guard is good design).
- Missing: **#41, #52** (information starvation) and **#43, #44** (heliocentric / PV frame).

**Repro (no build needed).** On `museum.html`, tick *"do-not-re-try only"* (`#mus-filter-dnr`,
filter at `museum.html:1615` on `r.do_not_retry`) and search `starvation` → **0 rows** —
while **Exhibit 12** (`#ex-starvation`, `museum.html:1079`) on the same page is badged
*"DO NOT RE-TRY — ledger §2 item 13"*. Same for the PV rows.

**Blast radius beyond this page.** `js/book.js:748` (`Book.ledger`, the book-wide
"Has this been tried?" instrument) badges from the same `do_not_retry` field, so information
starvation is reachable from any chapter's ledger panel *without* its verdict badge. That is
a direct functional violation of the museum meta-rule as `museum.html` §1 states it
("Any interactive anywhere in this book that lets you try one of those configurations must
volunteer the measured verdict").

**Also wrong in three printed places** (all say 26): the census caption (`:189`), the
`<noscript>` fallback (`:1290`), and M.4's answer (`:1397`).

**Suggested fix.** Widen the separator class to `[,/·;]` (or accept any non-`#` punctuation
between refs) and re-check the count lands on 30; update the three printed figures.

### M4 · `museum.html` §7 / `data/museum_ledger.json` row **#68** — the verdict is truncated and the residual and citation are wrong

**Observed.** `BIAS_HISTORY_LEDGER.md:88` contains **unescaped** pipes inside a cell:
`(trimming top-|tilt| GROWS it)`. The three other cell-internal pipes in the table are
escaped `\|` (rows #49d, #82) and parse correctly — I verified both round-trip fine. Row #68
does not:

| field | shipped JSON | ledger source |
|---|---|---|
| `verdict` | `"P-A refuted, P-B confirmed at the time: … (trimming top-"` **(truncated)** | `… (trimming top-\|tilt\| GROWS it)` |
| `residual` | `"GROWS it)"` | **`[AMBIG] see #69`** |
| `documented` | `"tilt"` | `pp_coverage_shallowvenue_20260711/SUMMARY.md:12-76` |

**Why this row specifically matters.** The cell that was destroyed is the **`[AMBIG] see #69`**
marker — the flag recording that #68's attribution was *reopened* by the h1_zclamp
re-attribution. That is the same open thread Exhibit 1 leans on ("the open attribution of
that +0.013 has been **unowned since 2026-07-13** — ledger #69"). The searchable ledger
therefore drops the ambiguity flag from precisely the row it belongs to, in the annex whose
contract is verbatim transcription ("Parsed, not transcribed" — provenance panel).

**Suggested fix.** In `gen_museum.py`, split table rows by taking the first 3 and last 3
cells from the ends (the count is fixed at 7) instead of a naive `split("|")`, or unescape
`\|` **and** tolerate a bare `|`; add a hard gate that every parsed row has exactly 7 cells
and non-empty `documented`. (A one-character fix in the ledger source would also work but is
outside book scope.)

### M5 · Ch 2 §4 (`<h2>4. The lurch</h2>`, "Why this page shows no percentages in the realistic venue") — a printed percentage does not follow from the two numbers beside it

**Observed.**

> "the signed total (**0.0851** for r1) is only about **62%** of the absolute curvature mass
> (**0.1650**), so ratios blow up"

**Measured** (r1 posteriors, the chapter's own pinned 3-point curvature statistic): signed
total **0.08514**, absolute curvature mass **0.16498** ⇒ **51.6%**, not 62%.

**Diagnosis.** 62% is `REALISTIC_READOUT.md` §4's **ensemble-mean** figure (mean 0.076 /
mean 0.123 = 61.8%). The chapter attached an ensemble ratio to r1's specific pair. The
chapter's *own data file* gets this right — `ch02_information.json.realistic_r1.why_not`
quotes the readout verbatim as an ensemble statement.

**Why it matters more than its size.** This is the book that teaches "do the division
yourself" (Q1.2, and the whole σ_dL arc). A reader who does gets 52%.

**Suggested fix.** "…the signed total (0.0851 for r1) is only **52%** of the absolute
curvature mass (0.1650) — **62%** averaged over the ten runs, which is the figure the
readout quotes — so ratios blow up".

### M6 · Ch 3 §4 (`<h2>4. The mechanism: h sweeps the shell</h2>`, "What the wave knows" card + Q3.5) — the ±0.155 / 236× headline is width-sensitive work done on the column the sources map forbids for it

**Observed.** The chapter's punchline — *"The true host's own catalogue redshift error, on
its own, smears its vote over ±0.155 in h — **236 times** the width the gravitational wave
contributes"* — is computed from `reduced_galaxy_catalogue.csv`'s `z_err` for row 859360
(σ_z = 0.0044429). I reproduced both numbers exactly from that file (σ_h from σ_z: 0.1549 for
the host, 0.0089 for the spec-z impostor; σ_h from σ_dL: 6.553×10⁻⁴; ratio **236.4**) —
which is the point: they are exactly as good as that column.

**Expected.** `BOOK_SOURCES_MAP.md` §7.19(d): *"The local `reduced_galaxy_catalogue.csv` is
**NOT** the #53 realization parent (local sha256 `623527929d…` vs sidecar
`parent_csv_sha256 7af3f4f4a2…`); they differ in **exactly one column, `z_error`** (the
cluster copy carries the #40b counted-once PV width). **Use the cluster parent for
width-sensitive work.**"* The project says the same in its own voice at C7: *"the local
`z_error` column is stale vs the cluster parent (#40b PV width), so the σ_z/z inputs are
indicative."* I confirmed the sidecar hash at
`real_r1/posteriors/realization_provenance.json` (`parent_csv_sha256 = 7af3f4f4a2…`,
`n_rows = 22641048`, `n_z_floor_clipped = 108395`).

Ch 6 carries the caveat verbatim ("licensed for positions and redshifts but is not the
production candidate list"). Ch 3's GW-reader box discusses the *observed realization*
(F-ch03-3) but never the *parent's* `z_error` — a different and, here, more relevant gap.
`ch03_FLAGS.md` has no entry for it.

**Suggested fix.** Add §7.19(d)'s sentence to Ch 3's venue box, chip ±0.155 / 236× / ±0.0089
as parent-dependent, and open F-ch03-13. The qualitative claim (photo-z dominates by ~2.4
orders of magnitude) is not in danger under any plausible PV width; only the digits are.

---

## [MINOR]

**m1 · Line anchors into `IDEALIZED_BASELINE_READOUT.md` now resolve to unrelated
sentences.** That file gained the 0.67 closure row at 02:43 on build day, after most
generators ran, shifting everything below line 30 by ~13 lines. Unlike the drift table in
BUILD_REPORT §5.5 item 23 (which lists *shifted* anchors), these now land on the wrong text:

| cited as | cited by | actual current lines | what `:cited` now holds |
|---|---|---|---|
| `:42-47` "76 of 1588 / 3 carry 46%" | ch00, ch01 (twice), ch02, ch03 | **:54-60** | the two-seeds-negative-bias paragraph |
| `:47-48` "σ_H0/H0 ≈ 0.38%/√76" | ch06 (twice) | **:60** | "…the sign coincidence is worth re-testing" |
| `:50-52` "median σ_z/z = 49%" | ch06 (twice) | **:64-66** | the "~15× narrower / zoom hook" paragraph |
| `:36-39` "~15× narrower / zoom hook" | ch02 (twice) | **:50-52** | the h=0.67 GPU-timeout caveat |

(`CLAIM_2D_BIAS_20260730.md:587-588`, cited by ch04 for the 200,807 figure, has drifted
similarly.) Suggest a regeneration pass that re-greps anchors, or switch to
quote-plus-section chips for files still under active edit.

**m2 · Museum §"What this annex is" overclaims one class.** *"…twenty-one were real defects
that were fixed and **did not fix the symptom**"* is false for at least row **#9**
("fixed — dominant at the time: MAP 0.60 → 0.73, bias −17.8% → 0.0%") and row **#12**
("fixed — **PRIMARY mover** per A5"), and it contradicts the museum's own closing Trap
("the single largest documented movement in the whole history, MAP 0.60 → 0.73 … ledger #9").
Use the ledger's own qualifier: "twenty-one were real defects that were fixed and landed —
most of them, in the ledger's own words, *insufficient alone*."

**m3 · Museum M1 flag box, "n = 50 is accurate to 1.7–2.3%".** Measured relative errors of
the vectorized `fixed_quad(n=50)` against exact: **1.70%** (h=0.60), **2.26%** (0.73),
**0.21%** (0.86). The stated range excludes the best of the three. Write "0.2–2.3%".

**m4 · Museum M1 static fallback, "every order from n=10 to n=600 returns ≈10⁻⁸⁶".** The
shipped scalar ladder runs **6.53×10⁻⁷⁹** (n=10) → 1.11×10⁻⁸⁶ (n=600). The claim being made
(flat in n, prints as 0.0000) is right; the exponent is only right at the high-n end. Write
"returns 10⁻⁷⁹–10⁻⁸⁶ — 0.0000 at every printed precision, and essentially flat in n".

**m5 · Ch 2 §4 switches denominators mid-sentence.** *"the top 3 alone carry 46% of the
in-catalogue budget …, the top 10 carry 71%, the top 20 carry 87%, and 24 of 1588 reach
91%."* Measured: of the **in-cat** budget 46.4 / 70.3 / 85.8 / 89.7%; of the **signed total**
47.0 / 71.1 / 86.9 / 90.8%. The first figure uses the in-cat denominator (correctly, per the
readout) and the other three use the signed total. Both sets are in
`ch02_information.json`; say which each uses, or quote one set throughout.

**m6 · Ch 0 §2 — "the most generous case imaginable" is actually the less generous of the two
branches.** The prose parks the hypothetical method on one anchor and excludes the far one;
the arithmetic that follows uses σ_A = 1.04 (i.e. parked on Planck, excluding SH0ES), giving
σ_tot ≤ **1.566**. Parked on SH0ES the far anchor is σ_A = 0.5 and the cap is **1.812** —
that is the generous branch. Nothing downstream is wrong (I reproduced 4.89σ, the 2.50σ
ceiling at σ_sys = 2, and all three static-fallback rows 3.01 / 2.50 / 4.49 exactly with
σ_A = 1.04), so this is a wording fix: name the anchor the method parks on, or say
"the more demanding of the two placements".

---

## [PRAISE]

**P1 · Ch 6 is the best-verified chapter in my scope, and it is not close.** Every 1590-row
population statistic reproduced to the digit from the CRB table: σ_u·ρ p5/median/p95
0.970 / 1.0398 / 1.250 (spread 1.289); σ_φ·ρ 0.0995 / 0.1886 / 0.5415; σ_θ·ρ
0.112 / 0.3131 / 1.032; |r_θφ| median 0.2603, 42.96% > 0.3, max 0.9815; 889's r(θ,φ) =
+0.2125 and σ_Mz/M_z = 1.365×10⁻⁹; 3×3 condition number median 53.2 / max 2813; in-cat
median σ_dL/d_L = 5.301×10⁻³. The dt² counterfactual reproduces exactly on the pool: 7548
vs 45 detections (**167.7×**, page says 168×), 7.623% detected, median z 0.468 vs 0.048,
z₉₀ 0.725 vs 0.096, deepest 1.108 vs 0.124, max stratum-a horizon 8.3318 → 9.16499 Gpc.
And it is the only chapter that gives *two independent* arbitration checks on the σ_dL unit
question (σ_u·ρ ≈ 1.04 ⇒ 7.3×10⁻⁴ at ρ = 1425; the readout's own 0.38%/√76). Keep the
"a normalization that multiplies a threshold changes which data exists" framing verbatim —
it is the single best sentence in the book.

**P2 · The museum's flagship F-museum-1 is the sharpest piece of work in the build, and it
survives independent re-execution end to end.** I reproduced: exact host-window numerator
0.241698 / 0.431397 / 0.653726 (= FINDING.md's 0.2417 / 0.4314 / 0.6537, digit for digit);
the vectorized `n=50` values 0.237592 / 0.441153 / 0.652358; the erratic ladder at h = 0.73
(1.5036 at n=10 = **3.49× high**, 0.0682 at n=20 = **6.33× low**, converging only near
n ≈ 75); and — the clincher — the scalar-collapse *prediction* 0.000265 / 0.000468 /
0.000701 reproducing FINDING.md's published GW-window column **0.0003 / 0.0005 / 0.0007**
digit for digit. The exhibit ships both evaluation modes, states exactly what the finding
does and does not touch (verdict untouched; mechanism (2) sufficient on its own; production
unaffected), and adjudicates nothing. That is precisely the right posture and it should not
be softened.

**P3 · Ch 5's mixture arithmetic is exact and self-verifying.** Independently confirmed:
mixture identity `p = w_G L_cat + (1−w_G) L_comp` to max rel. err. **3.907×10⁻¹³** over all
65,108 cells; N·Δln(1−w_G) = **31.5543** vs C10's +31.55; 493 events with L_cat = 0 at every
h, 1095 nonzero at every h, **0 mixed**; class argmaxes 0.86 (mean 0.8220, +22.1 nats from the
low edge) and 0.64 (mean 0.6571) with combined 0.740 / 0.7321; κ-dial endpoints 0.600
(catalogue alone) and 0.755 / 0.7461 (completion alone); flatten-the-slope 0.740 → 0.630.
Every completeness value reproduces from the frozen `m_th_map_nside32.npy`: f̄ = 0.9381 /
0.9204 / 0.7088 / 0.4341 / 0.3207 / 0.1916 / 0.0096, crossings at z = 0.1759 and 0.3565,
**751** empty pixels of 12,288. The 606 dossier row is right too (d_L 1.1747 Gpc, ρ 43.12,
dark, z = 0.2428, L_comp 0.0149 → 0.0195 → 0.0238, L_cat flat at 0.088). The
"5000× vs 1.6×10³ / 1.8×10⁵" flag (F-ch05-1) is presented with both values and no
adjudication — exactly as required.

**P4 · Ch 1's Ω_m gate is a model of how to cite a systematics budget.** Solving
d_L(z; h′, 0.2726) = d_L(z; 0.73, 0.3153) reproduces G7 row 6 at all six redshifts
(0.1601 / 0.3198 / 0.9384 / 1.5010 / 2.5951 / 3.3141% vs the published
0.16 / 0.32 / 0.94 / 1.5 / 2.6 / 3.31), and the chapter is right that the systematic is
**+0.064%** at EMRI-889's z ≈ 0.021. Worth noting for the record: I had to *restore flatness*
(`Omega_de = 1 − Omega_m`) to reproduce it — `dist()` takes Ω_m and Ω_de independently and
does **not** enforce flatness, so a careless reproduction gets +2.16% at z = 0.02 and −13.9%
for the Ω_m sweep instead of −0.45%. The generator evidently got this right; the chapter's
"Ω_Λ = 1 − Ω_m (flat)" in the equation box is doing real work.

**P5 · Ch 2's information decomposition reproduces exactly under the pinned metric.** Total
curvature **238.3321**, implied σ_h **3.2388×10⁻⁴** (vs the zoom posterior's 3.000×10⁻⁴),
**466** negative-curvature events, participation ratio **12.114**, in-cat share **1.0126** /
dark **−0.0126**, 24 events to 90%, cumulative first crossing 100% at **N = 39** ("around the
40th"), 889 ranked **#2** with curvature 37.97, and the top-3 share 0.4641 (in-cat) /
0.4700 (signed total). The two-grid figure is right too: production grid MAP 0.730 with
edge/peak **1.886×10⁻¹³⁹**, zoom MAP 0.72990 mean 0.72993 σ 3.000×10⁻⁴; the 41-point grid's
spacing (0.01 / 0.005 / 0.01, seams at 0.65 and 0.80) is exactly as described. The ten-run
table matches `REALISTIC_READOUT.md` §1 to the digit including "nine of the ten 68%
intervals contain the truth" (I checked all ten: only 62000/r2's [0.672, 0.725] misses).

**P6 · Ledger transcription fidelity is real.** I diffed 12 randomly chosen rows —
#11, #12, #22, #27, #37, #38, #41, #49b, #60, #68, #71, #86 — cell by cell against
`BIAS_HISTORY_LEDGER.md`. **Eleven are verbatim**: era, hypothesis, test, the full VERDICT
string and the residual all match, with markdown stripped and nothing paraphrased,
softened, or re-worded. (The twelfth is #68 — M4.) The row count is exactly 98 (94 numbered
+ 49a–49d), the verdict census sums to 98, and the class labels are honest (in particular
"measured — a number, not a verdict" for #49b/#68 rather than forcing them into a verdict
bucket).

**P7 · Venue discipline is enforced, not decorated.** Ch 4's separation of the Phase-32 rail
pair (−0.178 → 0.000, ledger #9, a different era and venue) from r1's live 0.740 / 0.7321 is
the template BUILD_REPORT says it is, and the adjudicator block that says so ("that is not a
bias measurement: bias is a property of an estimator over realizations") is the right voice
in the right place. Ch 6 §5 does the same two-sided job on the frame story — #12 fixed
(0.860 → 0.730) *and* #27 exonerated ("rotating would double-rotate"), with the guard's own
error message quoted as the lesson. And Ch 3's insistence that 552 is a *reconstruction*
count and "never a drop count" is the kind of distinction that normally gets lost in
translation. (The 552 itself needs re-measuring per B1 — the discipline around it does not.)

**P8 · Ch 4's D(h) section is fully traceable to the run's own log.** I pulled
`mixture_leg_log_extract.txt` directly: D(0.60) = 1.881202×10⁹, D(0.73) = 1.520637×10⁹,
D(0.86) = 1.257878×10⁹ Mpc³ sr⁻¹ (ratio 1.4955), z_max 1.1334 at h = 0.60 and capped at
1.5000 at h = 0.86, dl_max 9.1650 Gpc on every grid point — all as printed. The derived
budget checks out too: ln D(0.60) − ln D(0.86) = 0.4025, ×1588 = **639** nats against the
data's own 18.81 / 20.75 nats of depth (page: 18.8 / 20.8) — "roughly thirty times the whole
signal" is exactly right, and it is the most persuasive number in the chapter. The
200,807-vs-200,100 arithmetic is also exactly right (707 files, 200,100 data rows, strata
a/b/c = 99,014 / 50,947 / 50,139), as is `dl_max = 1.1 × 8.33181 = 9.16499` and the pool's
max `luminosity_distance` = 10.686 Gpc that F-ch04-2 warns against confusing it with.

---

## Verification appendix — gates re-derived independently

| chapter | gates re-run | result |
|---|---|---|
| ch00 | gap 5.64, 8.4%, 4.89σ; ceiling 1.566; (i)/(ii)/(iii) 3.01/2.50/4.49σ; σ_sys=2 asymptote 2.50σ | all reproduce |
| ch01 | row-889 all 14 params + σ_dL; z(d_L) at h = 0.60/0.73/0.86; d_L ratio 0.69767; Ω_m sweep −0.45%/−21.0%; G7 row 6 ×6 | all reproduce |
| ch02 | curvature 238.3321, σ_h 3.2388e-4, 466 neg, PR 12.114, shares, N=24/39; both grids; 889's L(h) both venues; 10-run table | all reproduce |
| ch03 | 889 ball radius + solid angle; host-frame lift M(1+z) → 2.96e-10; h\* 0.7294 / 0.7675; σ_h 0.00066 / 0.0089 / 0.155; ratio 236.4 | reproduce **at n_σ = 2** — see B1 |
| ch04 | D(h) ×3 from the run log; 639 vs 18.8/20.8 nats; 889 horizon 6.331 Gpc / 71.2×; pool 200,100 + 707; strata; dl_max 9.16499 | all reproduce |
| ch05 | mixture identity 3.9e-13; 493/1095/0; class argmaxes + means; κ-dial ends; N·Δln(1−w_G)=31.554; f̄(z) ×7 + crossings + 751 pixels; 606 + 889 legs | all reproduce (M1 is a scoping error, not an arithmetic one) |
| ch06 | 5 scaling products ×(p5,med,p95); 3 correlation families; 3×3 cond; 889 σ_Mz; dt² counterfactual (7548/45/168×/z-stats) | all reproduce |
| museum | 98 rows; census sums to 98; 12 rows diffed verbatim; M1 exact + vectorized + scalar ladders + GW-window prediction; gate_result.json 1D/2D means and Δ | reproduce except M3, M4 |

*Physics Expert A, 2026-07-31.*
