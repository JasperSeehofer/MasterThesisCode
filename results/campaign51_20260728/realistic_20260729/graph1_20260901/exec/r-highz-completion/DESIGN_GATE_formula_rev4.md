# DESIGN GATE — formula review, rev 4 (FRESH reviewer, 2026-09-04)

Scope: FIX 4 only — PIN CORRECTION 4 (`--g1d-tol` CLI flag, G-1d band 1e-8 → 1e-6 absolute on
ln D̃φ). This is a gate-tolerance widening on a display-precision consistency check, not a
disposition-threshold or formula change. Builders/reviewers never compute a registered aggregate;
`--dry-run` and synthetic tables only, per task instruction.

## 1. `git diff` — confirm FIX 4 touched only what it claims

```
$ git diff --stat
 .../r-highz-completion/BUILD_RECORD.md        | 72 ++++++++++++++++++++++
 .../r-highz-completion/REGISTRATION_DRAFT.md  |  6 ++
 .../r-highz-completion/highz_decomp_reads.py  | 46 +++++++++++---
 3 files changed, 116 insertions(+), 8 deletions(-)
```

No other file under `exec/r-highz-completion/` (or anywhere else) is touched — `git diff --stat -- <dir>`
returns the identical three-file list. **No production-pipeline / cluster / `darksiren_emri/` files
appear in the diff** (checked directly; none of the forbidden paths are present).

Read the `highz_decomp_reads.py` diff hunk-by-hunk:

| hunk | change | in scope? |
|---|---|---|
| argparser | adds `--g1d-tol` (`type=float, default=1e-6`) | ✓ plumbing |
| `gate_g1_closure` signature + docstring | adds `g1d_tol: float = 1e-6` param, docstring explains the pin | ✓ plumbing |
| G-1d check body | `resid_den.max() > 1e-8` → `> g1d_tol`; error string interpolates the resolved bound instead of the literal `1e-8` | ✓ plumbing (the ONLY behavioral line changed) |
| `run_synth_check()` | new block: builds a synthetic table with a 5e-7 residual, asserts PASS at `g1d_tol=1e-6`, asserts `InstrumentDefect` at `g1d_tol=1e-8` | ✓ new SYNTH assertion, as instructed |
| `_five_row_slice_closure` | adds `g1d_tol: float = 1e-6` param, threads to its `gate_g1_closure` call | ✓ plumbing |
| `run_production_family` | adds `g1d_tol: float = 1e-6` param, threads to its `gate_g1_closure` call | ✓ plumbing |
| `main()` | threads `args.g1d_tol` into `_five_row_slice_closure` and the per-family loop; prints `[gate G-1d] resolved --g1d-tol: ...`; records `run_metadata["g1d_tol"] = args.g1d_tol` in `--out` | ✓ plumbing + disclosure |

Every other threshold in the file is untouched: `G2_MEAN_H_TOL = 1e-9`, `G2_DELTA_K_TOL = 1e-6`,
G-1b's `max_resid > 1e-9` (closure residual, unrelated pin), G-1d's own `g_frac` relative-residual
line (`rel_g.max() > 5e-7`, unrelated to this pin and byte-identical), G-2(i)'s `1e-9`, all
`null_lo=-1e-6, null_hi=1e-6` disposition-band constants in the SYNTH harness-outcome fixtures, and
`harness_outcome_disposition`'s own bands — grepped explicitly, all identical to `HEAD`. **Confirmed:
FIX 4 changed exactly the G-1d tolerance plumbing, the new SYNTH assertion, and docs — nothing else.**

## 2. `--g1d-tol` threading — every G-1d call site

`gate_g1_closure` is the only function that evaluates the G-1d check. It is called from exactly
four places in the file, all four now accept/forward `g1d_tol`:

1. `run_synth_check()` — twice, explicitly (`g1d_tol=1e-6` then `g1d_tol=1e-8`, the new assertion).
2. `_five_row_slice_closure()` — forwards its own `g1d_tol` param (default `1e-6`), called from
   `main()` as `_five_row_slice_closure(args.logl_iiib, table_iiib, args.h_true, g1d_tol=args.g1d_tol)`.
3. `run_production_family()` — forwards its own `g1d_tol` param (default `1e-6`), called from
   `main()`'s per-family loop as `g1d_tol=args.g1d_tol`.

No orphaned call site retains the hard-coded `1e-8`. The resolved value is recorded in `--out`'s
`run_metadata.g1d_tol` (confirmed present in the diff) and printed under `--dry-run`/real mode alike
(reproduced live below).

## 3. Hand-check: is PIN CORRECTION 4's tolerance derivation sound?

**Claim:** 7-significant-figure storage of `D_tilde_phi` (a display column, `bayesian_statistics.py:5467`)
bounds its relative rounding error at ~5e-7, which propagates through `ln` to an absolute error on
`ln D̃φ` of the same order (~5e-7), so a 1e-6 absolute band (2× that bound) is a sound, non-arbitrary
choice for a display-precision consistency gate — not a loosening that could mask a real formula defect.

**Worst-case rounding of a 7-s.f. mantissa.** A value `x` stored to 7 significant figures is
representable at a grid spacing of `10^(e-6)` where `e = floor(log10|x|)` (e.g. for `x ~ 0.1`, grid
spacing `10^-7`; for `x ~ 1.1`, grid spacing `10^-6`). The worst-case rounding error is half that
spacing:

  |δx| ≤ 0.5 · 10^(e-6)

Relative to `x` (with `10^e ≤ |x| < 10^(e+1)`), the worst-case *relative* error is bounded by

  |δx|/|x| ≤ 0.5 · 10^(e-6) / 10^e = 0.5 × 10^-6 = 5e-7

— independent of `e` (scale-invariant, as expected for significant-figure rounding). This matches
PIN CORRECTION 4's stated "relative storage precision ~5e-7" exactly.

**Propagation through `ln`.** `d(ln x)/dx = 1/x`, so to first order

  δ(ln x) ≈ δx / x = (relative error in x)

so the worst-case absolute perturbation to `ln D̃φ` from 7-s.f. storage rounding alone is bounded by
the same ~5e-7. The second-order term is `-(δx)²/(2x²) ~ -(5e-7)²/2 ≈ -1.25e-13`, six orders of
magnitude below the linear term — negligible, so the linear bound is not merely a first approximation
but effectively exact at this scale. **The derivation holds**: a 1e-8 band on `ln D̃φ` cannot pass by
construction against a 7-s.f. source column, and 1e-6 (= 2× the ~5e-7 theoretical worst case) is a
correctly-derived, conservative-but-not-loose choice — tight enough that a genuine formula defect
(which would show up as an O(1) or at least O(1e-3)-scale residual, not an O(1e-7) one) would still
be caught, while no longer failing on storage-precision alone.

**Cross-check against the disclosed real-data residuals.** PIN CORRECTION 4 / BUILD_RECORD.md FIX 4
report measured max residuals of 4.407370e-7 (iiib) and 4.102515e-7 (jr1) over the full `P_dark`
tables, both venues, both channels. Both are `< 5e-7` (consistent with the theoretical worst-case
bound derived above — the measured values sit *under* the worst case, as they should, since not
every row hits the half-ULP rounding extremum) and both comfortably clear `1e-8` (confirming the old
band was indeed unpassable by construction) while sitting well inside `1e-6` with headroom
(`1e-6 / 4.407e-7 ≈ 2.27×`, `1e-6 / 4.103e-7 ≈ 2.44×`). The numbers are internally consistent with
the derivation and with each other (iiib and jr1 residuals agree to within ~7%, as expected since
both venues discretize `D_tilde_phi` through the same 7-s.f. display formatting).

I independently re-derived this bound from the storage-precision argument alone (not by trusting the
disclosed 5e-7 figure), and it reproduces PIN CORRECTION 4's number. **The tolerance derivation is
sound.**

## 4. `--dry-run` on the real §8 launch block — reproduced live

Ran the exact §8 block (with the appended `--g1d-tol 1e-6` and `--dry-run`) myself, token-for-token
against the file's current argparser:

```
[pin OK] logl-iiib md5: 8e6a2c18dc5838dd1d52641589243672
[pin OK] logl-jr1 md5: 745954a0fdee5f10878fb5e622a06144
[pin OK] table-iiib sha256: 90c92026bb7fecff46e5a55e1e2c67a33b424e4b71611ee0d0854576b189f7b0
[pin OK] table-jr1 sha256: fc2eebe7fa66afbe2e35b0dd09c889be790511a6f5dabce3338969c849fcdf3a
[pop pin OK] iiib P_dark: n=606 ...  [pop pin OK] iiib K: n=159 ...  [pop pin OK] iiib K_dark: n=144 ...
[pop pin OK] iiib R: n=231 ...      [pop pin OK] jr1 P_dark: n=493 ... [pop pin OK] jr1 K: n=159 ...
[pop pin OK] jr1 K_dark: n=111 ...  [pop pin OK] jr1 R: n=191 ...
[pin OK] harness manifest sha256: 6a06063dd56aae74ee1cc8bbc63f7da8207ff3e3fc705290a81a2675adb1c0a2 (67 universes)
[gate OK] G-3d: 13 resolved_flags tokens identical, 67/67 universes
[counts] iiib: n=1588 P_dark=606 K=159 K_dark=144 K_hosted=15 R=231
[counts] jr1:  n=1588 P_dark=493 K=159 K_dark=111 K_hosted=48 R=191
[counts] harness: universes=67 Sigma n_scored(CSV event_idx)=12060 (anchor 12060)
[gate G-1] 5-row real-slice max closure residual: 2.665e-15 (band 1e-9)
[gate G-1d] resolved --g1d-tol: 1.000e-06
[SYNTH OK] closure identity, disposition rows (production 4 + harness 6), G-1 pass/fail path,
  Findings A-D counter-examples, Finding H K-vs-K_dark leaveout, Finding I channel term selection,
  Finding J replicate-rule pass/miss
[dry-run] gates + byte-id anchors only, no --out written, no registered aggregate computed.
EXIT: 0
```

Matches BUILD_RECORD.md's FIX 4 record exactly, including the new `[gate G-1d] resolved --g1d-tol:
1.000e-06` line. No `--out` file was produced (`--dry-run` short-circuits before the write). Ran
independently, not by trusting the builder's transcript.

`ruff check` and `mypy` on the changed file both pass clean (re-confirmed independently).

## 5. Verdict

**GREEN.** `git diff` confirms the change is scoped to exactly the G-1d tolerance plumbing (CLI flag,
signature threading through all four `gate_g1_closure` call sites, `--out` disclosure) plus the new
SYNTH pass/fail assertion plus docs (REGISTRATION_DRAFT.md PIN CORRECTION 4 addendum,
BUILD_RECORD.md FIX 4 entry) — no disposition threshold, no other gate band (G-1b's 1e-9, G-1d's own
`g_frac` 5e-7 relative check, G-2's 1e-9/1e-6 anchors, the harness-outcome null bands) is touched.
`--g1d-tol` is correctly threaded to every G-1d call site and its resolved value is recorded in
`run_metadata` and printed under `--dry-run`. The 1e-6 absolute band on `ln D̃φ` is a sound,
independently-rederived consequence of 7-significant-figure storage rounding (worst case ~5e-7,
negligible second-order correction), cross-checked against the disclosed real-data residuals
(4.407e-7 iiib / 4.103e-7 jr1), which sit under the theoretical worst case and comfortably inside the
new band with ~2.3-2.4× headroom while still clearing the old (unpassable-by-construction) 1e-8 band.
`--dry-run` on the real, pinned §8 launch block reproduces exit 0 with the exact counts/manifest/gate
lines claimed, run independently. The disjoint reader may run §8 in real mode.
