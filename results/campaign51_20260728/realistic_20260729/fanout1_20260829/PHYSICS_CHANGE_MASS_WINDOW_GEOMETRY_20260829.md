# [PHYSICS] Gate presentation — mass-window GEOMETRY instrument flag

**Charter node B5.1 [WIN], part (B).** Launched under rows #222/#223 (author standing grant,
all depths + production changes within the tree). Presentation only — **no code has been
written**; this returns to the author/end-verifier before any edit to
`darksiren_emri/galaxy_catalogue/handler.py` or `darksiren_emri/bayesian_inference/
bayesian_statistics.py` (both physics-trigger files, `/physics-change` hard-gated).

Approval cited: **row #223** (author, verbatim, ledger 2026-08-29): *"production changes
inside the tree are covered too. Physics gates: presentation before code + ledger rows as
always; the approval step cites row #223; every gate goes to the end verifier."* This
presentation is that step — it authorizes writing the code below, not adopting it; adoption
(a production default flip) is a separate, later gate per runbook 37 §5's "Assumption
pending author confirmation: production DEFAULT flips still return to the author."

Companion zero-compute read: `b5_window_count.json` (this directory), part (A) of this node.

---

## 1. Old formula, verbatim, with lines

**Flag declaration + docstring** (`darksiren_emri/galaxy_catalogue/handler.py:648-660` —
**line numbers SUPERSEDED, see Note R3**: at HEAD `a794404c` this block is actually at
`handler.py:598-609`; the quoted text itself is verbatim-correct and unaffected):

```python
        mass_filter_sigma: mass pre-filter window selector (ledger
        rows #198–#202; "symmetric" adopted as the production default
        per ``docs/derivations/PROPOSAL_MASS_FILTER_SYMMETRIC_
        20260825.md`` §7(a)). "symmetric" (default): ``BH_MASS_ERROR``
        is scaled by ``sigma_multiplier`` on both sides, matching the
        GW-side convention — the single-k interval-overlap window.
        "asymmetric" is the explicit COUNTERFACTUAL pinning the
        retired pre-flag path: galaxy error at its bare (×1) value —
        the undocumented ±1.5σ-vs-±1σ window (Gate-B DEFECT, row
        #196). This is the single read/validate site for the flag.
```

**The mask itself** (`handler.py:654-673`):

```python
        if mass_filter_sigma == "asymmetric":
            _bh_mass_error_multiplier: float = 1.0
        elif mass_filter_sigma == "symmetric":
            _bh_mass_error_multiplier = float(sigma_multiplier)
        else:
            raise ValueError(
                f"mass_filter_sigma must be 'asymmetric' or 'symmetric', got {mass_filter_sigma!r}"
            )

        mass_filter_mask = (
            (M_z - M_z_sigma * sigma_multiplier) / (1 + z_max)
            <= candidate_hosts_without_bh_mass[InternalCatalogColumns.BH_MASS]
            + candidate_hosts_without_bh_mass[InternalCatalogColumns.BH_MASS_ERROR]
            * _bh_mass_error_multiplier
        ) & (
            candidate_hosts_without_bh_mass[InternalCatalogColumns.BH_MASS]
            - candidate_hosts_without_bh_mass[InternalCatalogColumns.BH_MASS_ERROR]
            * _bh_mass_error_multiplier
            <= (M_z + M_z_sigma * sigma_multiplier) / (1 + z_min)
        )
```

i.e. a **linear** two-sided interval-overlap test between the GW source-frame interval
`(M_z ± k·σ_Mz)/(1+z_{max,min})` and the candidate's catalogue interval
`M ± k·σ_M` — `k = sigma_multiplier`, currently a **single, dual-purpose parameter**:
the SAME value (`1.5`) also sets the sky-cone search radius at the call site.

**Call site** (`darksiren_emri/bayesian_inference/bayesian_statistics.py:4791-4798`):

```python
            possible_hosts = galaxy_catalog.get_possible_hosts_from_ball_tree(
                ...
                sigma_multiplier=1.5,  # type: ignore[arg-type]
                mass_filter_sigma=self._mass_filter_sigma,
            )
```

`self._mass_filter_sigma` is set from the constructor / `evaluate()` (`bayesian_statistics.py:
3292-3298` class default, `:3361` bare-fallback, `:3724` `evaluate()`-validated single read
site) and is currently `"symmetric"` in production (rows #198–#202, `docs/derivations/
PROPOSAL_MASS_FILTER_SYMMETRIC_20260825.md`).

**Catalogue error model actually feeding `BH_MASS_ERROR`** (`handler.py:1368-1382`,
`_empiric_stellar_mass_to_BH_mass_relation`, Reines & Volonteri 2015 arXiv:1508.06274 §4.1):

```python
def _empiric_stellar_mass_to_BH_mass_relation(
    stellar_mass: float, stellar_mass_error: float
) -> tuple[float, float]:
    BH_mass = np.exp(alpha + beta * np.log(stellar_mass / 10))
    BH_mass_error = BH_mass * np.sqrt(
        sigma_int**2
        + d_alpha**2
        + (np.log(stellar_mass / 10) * d_beta) ** 2
        + (beta / stellar_mass * stellar_mass_error) ** 2
    )
    return (BH_mass, BH_mass_error)
```

with `sigma_int = 0.24*ln(10) ≈ 0.5527` (the "~0.55 dex intrinsic scatter", the DOMINANT term
per the memory note `mass-relation-reines-volonteri`). **`BH_MASS_ERROR` is already
constructed as `BH_MASS · σ_lnM`** — the ln-space error budget is computed first and only
then multiplied through by the point mass to produce an absolute (linear) error. This is the
key fact the new log-geometry formula below exploits: **`σ_lnM ≡ BH_MASS_ERROR / BH_MASS`
exactly, with no re-derivation**, because that ratio is what the R&V15 formula computed
before multiplying by `BH_mass`.

---

## 2. New formula

Two new independent flags on `get_possible_hosts_from_ball_tree`, threaded from
`BayesianStatistics.evaluate` through the constructor (mirroring the existing
`mass_filter_sigma` threading at `bayesian_statistics.py:3298/3361/3724`):

- `mass_filter_geometry: Literal["linear", "log"] = "linear"` — **byte-identical default**.
- `mass_filter_k: float = 1.5` — decouples the mass-window half-width from `sigma_multiplier`
  (which stays the SKY-cone radius multiplier only). Default value chosen to exactly match
  today's coupled `sigma_multiplier=1.5` behaviour when `mass_filter_geometry="linear"`.

`"linear"` (default): **byte-identical** to §1 — `mass_filter_k` replaces `sigma_multiplier`
in the two `M_z ± k·σ_Mz` / `M ± k·σ_M` expressions above; `sigma_multiplier` no longer feeds
the mass window at all, but at the default `mass_filter_k=1.5 == sigma_multiplier=1.5` call
site value the arithmetic is identical to the last float.

`"log"` (new): the same interval-overlap test, both sides re-expressed in ln-space —

```
GW side:        exp( ln(M_z) − k·σ_lnM,z )  ≤  candidate upper edge
                candidate lower edge  ≤  exp( ln(M_z) + k·σ_lnM,z )      [both /(1+z) as today]
candidate side: [ M · exp(−k·σ_lnM),  M · exp(+k·σ_lnM) ]
```

with:
- **`σ_lnM` (candidate/catalogue side)** `= BH_MASS_ERROR / BH_MASS` — read directly off the
  existing column ratio, no new derivation (§1 above establishes this is already the R&V15
  ln-space budget, not an approximation).
- **`σ_lnM,z` (GW side)** `≈ M_z_sigma / M_z` — the **small-error correspondence**: for a
  Gaussian-distributed CRB estimator with relative error `ε = σ_M/M ≪ 1`,
  `ln(M+δ) = ln M + δ/M − δ²/(2M²) + O(δ³)`, so `Var[ln M] ≈ (σ_M/M)² + O(ε⁴)` to leading
  order — the CRB's *linear* relative error IS the ln-space sigma to first order. (This is
  the same correspondence check as banked P1's spot values in
  `PREREGISTRATION_MKER_WGEOM_20260828.md` §3 — `geometry_function_A(x)→0` as `x→0`.)

Code shape (illustrative; not yet written):

```python
if mass_filter_geometry == "linear":
    gw_lo, gw_hi = (M_z - k*M_z_sigma)/(1+z_max), (M_z + k*M_z_sigma)/(1+z_min)
    cand_lo, cand_hi = M - k*M_err, M + k*M_err
elif mass_filter_geometry == "log":
    sigma_lnM_z = M_z_sigma / M_z
    gw_lo  = M_z * np.exp(-k*sigma_lnM_z) / (1+z_max)
    gw_hi  = M_z * np.exp(+k*sigma_lnM_z) / (1+z_min)
    sigma_lnM = M_err / M  # == BH_MASS_ERROR/BH_MASS, R&V15 ln-space budget verbatim
    cand_lo, cand_hi = M*np.exp(-k*sigma_lnM), M*np.exp(+k*sigma_lnM)
mass_filter_mask = (gw_lo <= cand_hi) & (cand_lo <= gw_hi)
```

## 3. Reference

- `results/campaign51_20260728/realistic_20260729/PREREGISTRATION_MKER_WGEOM_20260828.md` §9
  and the RE-ANCHORED EVALUATION (W-1/W-2 corrections, 2026-08-28) — the closed-form geometry
  function `A(x)`, the ε-semantics table, and the "CV = 2/3" negative-lower-edge threshold
  that motivates a log-symmetric alternative in the first place.
- Ledger rows #220 (WGEOM REFUTED-IN-PART of record + W-2 finding ratified) / #221 (F-ii =
  REDESIGN: log-symmetric, k=3, ε=2Φ(−3)=0.27%, adopt only after a registered counterfactual
  — this presentation is the pre-code half of that grant, NOT the counterfactual itself).
- Reines & Volonteri (2015), arXiv:1508.06274, §4.1 — the AGN M_BH–M\* relation and its
  intrinsic scatter `σ_int`, already the sole source of `BH_MASS_ERROR`.
- `results/campaign51_20260728/realistic_20260729/wgeom_work/wgeom_w2_split.json` — the
  corrected too-light/too-heavy split (linear 45.64:1 too-light; log 47.99:1 too-heavy at
  k=1.5) — the mechanism this flag is built to let the author probe directly at k=3.

## 4. Dimensional analysis

`M`, `M_z`, `BH_MASS`, `BH_MASS_ERROR`, `M_z_sigma` are all solar masses `[M_sun]`.
`σ_lnM = BH_MASS_ERROR/BH_MASS` and `σ_lnM,z = M_z_sigma/M_z` are **dimensionless** (a ratio
of two same-unit quantities), so `k·σ_lnM` is dimensionless and `exp(±k·σ_lnM)` is a pure
number — multiplying it back onto `M` (`[M_sun]`) returns a quantity in `[M_sun]`, matching
both the linear window's units and the quantity it is compared against (`cand_lo`, `cand_hi`
both `[M_sun]`, compared to `gw_lo`, `gw_hi` both `[M_sun]/(1+z)` → `[M_sun]`, since redshift
is dimensionless). No unit mismatch is introduced; `k` itself stays dimensionless in both
geometries (a number of sigma), matching `sigma_multiplier`'s existing role.

## 5. Limiting cases

1. **`k → ∞`: no filter, reproducing the HB exoneration.** Linear: `cand_lo → -∞`,
   `cand_hi → +∞`, mask → all-True (window removed). Log: `exp(-k·σ_lnM) → 0`,
   `exp(+k·σ_lnM) → ∞`, mask → all-True identically (positive-mass support, same limit). Both
   geometries converge to the **hard mass window as support truncation** object HB already
   measured and exonerated (`CLAIM_WGEO_20260827.md` §4.1: "HB hard mass window as support
   truncation (tilt −0.317 nats = 0.063% of the target, sign-inverted, ~50× too small)") — a
   new geometry cannot resurrect a bias HB already bounded away, and the k→∞ agreement between
   the two geometries is the guard against a geometry-introduced discontinuity at that limit.
2. **`σ → 0`: point overlap.** Both `σ_lnM→0` and `σ_M/M→0` collapse each window to
   `[M, M]`/`[M_z,M_z]` at leading order (linear exactly; log via
   `exp(±k·σ)→1±k·σ→1` as `σ→0`) — both geometries degenerate to the same point-overlap test
   `M/(1+z) == M_z/(1+z)` up to `O(σ)`, so there is no discontinuity at the zero-uncertainty
   edge either.
3. **Log and linear agree to first order for CV ≪ 1.** `exp(±kσ) = 1 ± kσ + O(σ²)`, so the
   log window's edges `M(1±kσ)` match the linear window's edges `M ± kσM = M(1±kσ)`
   EXACTLY at first order — the two geometries are the same window to `O(σ²)`, diverging only
   once `kσ` is not small. This is exactly `geometry_function_A(x)→0` as `x→0` in the banked
   P1 table (`PREREGISTRATION_MKER_WGEOM_20260828.md` §3, spot value `A(0.1)=-0.050084`) —
   the pre-existing closed-form asymmetry measure for this same correspondence.

## 6. Regression plan

1. **Default byte-identity on a banked cone.** `mass_filter_geometry="linear"`,
   `mass_filter_k=1.5` must reproduce the CURRENT production candidate set bit-for-bit on a
   frozen cone (reuse the `bc_9001XX_work` cone-exact fixture already used by
   `wgeom_instrument.py`/`wgeom_w2_split.py` — no new fixture needed). Precedent: the row
   #198-#202 `mass_filter_sigma` default-flip gate used exactly this pattern
   (`PHYSICS-GATE-LEDGER.md` 2026-08-25 "verified" row: "diff scope exactly the 5 defaults").
2. **Log-window unit test with hand-computed edges.** A handful of `(M, M_err, M_z,
   M_z_sigma, z_min, z_max)` tuples with `σ_lnM`/`σ_lnM,z` computed by hand (not from the
   production formula) at `k∈{1.5,2.5,3}`, asserting `cand_lo/cand_hi`/`gw_lo/gw_hi` to
   float precision — the same style as `wgeom_instrument.check_g4`'s hand-anchored P2 table.
3. **Epsilon test: `2Φ(−k)` numerically.** For a synthetic log-normal population of
   candidate masses at a FIXED `σ_lnM = s`, the fraction excluded by the log window at
   sigma-multiplier `k` must equal `2·Φ(−k)` to Monte-Carlo precision, CV/σ-INDEPENDENT by
   construction — reproducing `wgeom_instrument.eps_log(k) = 2*norm.cdf(-k)`, already banked
   and G4-verified for k=1.5 (`eps_log_total=0.133614` in six banked rows,
   `wgeom_work/wgeom_result.json` p2 table). At k=3 this predicts `2Φ(−3) = 0.0027` — the
   ratified ε, and the single-candidate self-referential number that the FLEET-LEVEL zero-compute
   read below shows is **not** the same thing as the fleet's *true-host* retention rate (§7).

## 7. Zero-compute candidate-count factor — headline numbers (part A, `b5_window_count.json`)

> **⚠ SUPERSEDED IN PART — see Revision Notes R0–R9 (appended 2026-08-29) at the end of this
> document.** A refuter panel caught a material bug in the script that produced every number
> below: `gw_window()` always used the LINEAR formula for the GW-side window, even under the
> "log" configs (ii)/(iii)/(iv), contradicting this section's own §2 spec. The bug has been
> fixed and the script re-run by a different agent (Note R1); the numbers below are **NOT
> edited in place** (append-only discipline) but Note R2 shows they are numerically unchanged
> at the precision reported here (full-precision deltas are all ≤1.3e-6 relative). Cite the
> Revision Notes / the regenerated `b5_window_count.json` and `b5_window_count_arm_jackknife.json`
> for per-number provenance, falsifiers, invariants, blindness statement, and operating-
> characteristics bands going forward — this table stands as originally written and is
> confirmed, not retracted.

Reproducible cone-exact fleet basis, `bc_9001XX_work` × 24 arms, `n_all = 2,249,231`
candidate rows over 2,261 events (SAME basis as `wgeom_result.json` P3 / `wgeom_w2_split.json`,
the CLAIM_WGEO §3.8/CORRECTION-NOTE-W-1 corrected numbers — NOT the stale §3.9 figures still
hardcoded in `wgeom_instrument.BANKED_P3A`).

| config | geometry | k | pass fraction (n_pass/n_all) |
|---|---|---|---|
| (i) | linear | 1.5 | **0.95768** (gate target 0.9577 — **PASSED**, 4dp) |
| (ii) | log-symmetric | 1.5 | 0.40613 |
| (iii) | log-symmetric | 3.0 | 0.69509 |
| (iv) | log-symmetric | 2.5 | 0.61489 |

Gate: (i) reproduces the corrected §3.8 `n_lin/n_all = 0.9577` at 4 decimal places
(full precision 0.957681 vs. the frozen instrument's own production-JSON figure 0.957690 —
the ~9e-6 residual is the ALREADY-DOCUMENTED 21-row floating-point boundary artifact,
`wgeom_result.json` `p3.lin_recompute_mismatches=21`, not a new discrepancy). **PASSED.**

**Per-event candidate growth, (iii) log k=3 vs (i) linear k=1.5** (2,221 events with ≥1
linear-passing candidate): mean **0.814**, median **0.949**, p95 **1.498**, max **10.0**
(16 events gain candidates that linear admitted zero of; 24 events are empty under both).
The **aggregate** ratio (0.695/0.958 = 0.726) is markedly below the **median per-event**
ratio (0.949) — a right-skew: most events see a small net LOSS or wash, a minority see up to
10× growth, and a few large-candidate-count events dominate the aggregate sum.

**⚠ This contradicts the runbook's stated performance framing.** Runbook 37 §5's B5
performance note reads: *"the mass window currently removes only ~4.2% of cone candidates
(n_lin/n_all=0.9577), so k=3/log cannot add more than that."* That statement implicitly
assumed the change is monotone-widening (more admission, bounded above by the ~4.2% currently
excluded). **The measurement shows the opposite direction at the aggregate level**: k=3/log
*removes* ~30 percentage points MORE than linear k=1.5 (69.5% pass vs. 95.8%), not less. The
mechanism (identified by hand-tracing several events, see `b5_window_count.py` cross-check):
linear's window at CV > 2/3 (≈99.6% of the catalogue, `wgeom_result.json` p2
`negative_lower_edge_fraction`) has a NEGATIVE lower edge, so its "is this candidate too
HEAVY" test (`cand_lo ≤ gw_hi`) is **vacuously always true** — linear currently admits
arbitrarily-too-heavy candidates for free whenever the fractional mass error exceeds 67%. Log
closes this loophole (it is bounded, `cand_lo = M·exp(−kσ) > 0` always), so switching
geometry is not a pure widening — it is a widening on one tail (as ε(k=3)=0.27% vs the
much larger, CV-dependent `eps_lin_heavy` at k=1.5 suggests) **and a tightening of a
previously-vacuous exclusion on the other tail**, and the tightening dominates the aggregate
count at this catalogue's CV distribution (median CV ≈ 0.86, per the banked P2 table). This
is the SAME mechanism already named in `wgeom_w2_split.json` ("heavy-cut-reintroduction"):
log reintroduces a heavy-side cut that linear structurally cannot apply.

**True-host retention** (mass-window-only test — the true host's own catalogue
`BH_MASS`±`BH_MASS_ERROR` vs. the event's own observed GW mass window at the same k;
independent of the sky+redshift cone, isolating the mass-geometry effect; 2,261 events with a
valid catalogued, unscattered true host):

| config | fraction of true hosts retained |
|---|---|
| (i) linear k=1.5 | **0.9567** |
| (ii) log k=1.5 | 0.7001 |
| (iii) log k=3.0 | **0.7890** |
| (iv) log k=2.5 | 0.7682 |

**⚠ Second caveat, and the more important one for the adoption decision.** Moving from the
CURRENT production default (linear, k=1.5) to the RATIFIED candidate design (log, k=3)
measures a drop in true-host retention from 95.7% to 78.9% — a ~17-point loss, in this
zero-compute census. Manually traced examples (in `b5_window_count.py`'s cross-check script,
not yet in the JSON) show the SAME mechanism as above running in reverse for the true host:
the host's own R&V15-mapped catalogue mass estimate is frequently very discrepant from the
GW's independently-measured `M_z` (by up to ~30× in the traced cases) purely from the R&V15
relation's `σ_int ≈ 0.55` dex intrinsic scatter — the DOMINANT term in `BH_MASS_ERROR`
(§1). Under linear k=1.5, the negative-lower-edge loophole silently RETAINS these
badly-discrepant true hosts (vacuous "too heavy" test); under log — by design — a genuinely
`kσ`-wide but bounded window, at k=3, still excludes hosts whose catalogue-mass estimate is
off by more than ~exp(3σ_lnM). **This is not evidence the log/k=3 design is wrong** — the
window is doing exactly the principled job it was built to do — but it means the "adopt only
after a registered counterfactual" grant (row #221 F-ii) should explicitly measure the
production ΔMAP against this specific host-loss mode, not only against HB's already-banked
+0.0015 ceiling, since a 17-point true-host loss rate is a qualitatively different quantity
from a candidate-COUNT change and could carry its own, uncharacterized H₀ effect (INFORMATION
loss on the correct-host tail vs. contamination reduction on the wrong-host tail — the net
sign is UNDETERMINED by this zero-compute read and is exactly what the node B5.2 registered
counterfactual should measure).

**Status of this read:** builder-run, not independently verified (standing rule 2 — a
different agent should re-run `b5_window_count.py` before any of §7's numbers are cited as
adopted evidence, though the config-(ii) log-k1.5 total is already an exact bit-for-bit
cross-check against the frozen `wgeom_instrument.py`'s own `n_log_over_n_all`, which is strong
internal evidence the recomputation logic itself is correct).

> **Update, 2026-08-29 (Note R1/R2 below):** the "not independently verified" status above is
> now PARTIALLY superseded — a different agent from the one that wrote this document and
> `b5_window_count.py` has fixed the GW-window bug and re-run the (corrected) script,
> confirming these numbers hold at reported precision. This is a **re-run + crash/logic-fix
> verification of the corrected instrument**, not a full independent physics-design
> verification of node B5.2's eventual registered counterfactual (that remains open and is
> node B5.2's job, not this presentation's).

---

## Ledger

A "presented" row has been appended to `docs/gates/PHYSICS-GATE-LEDGER.md` citing this
document. Per the file's convention, `implemented`/`verified` rows are for the actual
edit (not made here) and a later independent pass. A second ledger row records this revision
(see Note R9).

---

## Revision Notes (appended 2026-08-29, post-refuter-panel)

Launched under rows #222/#223 — charter node B5.1, fanout wave 1, 2026-08-29. **Append-only:**
nothing above this line is edited; superseded passages are marked inline with a pointer to the
relevant note below. All work in this section (the bug fix in `b5_window_count.py`, the re-run,
and the new `b5_window_count_arm_jackknife.py`/`.json`) was performed by a **DIFFERENT AGENT**
from the one who authored the original document text and `b5_window_count.py` — satisfying
standing rule (2) (verifier independence): the refuter panel identified a genuine logic defect
(not merely a crash), so per that rule this correction + re-run required a different agent, not
a builder smoke-test, and that requirement is what this section records as discharged.

### R1 — the bug and the fix (must_fix items 1, 3)

**Bug** (refuter panel finding, confirmed): `b5_window_count.py`'s `gw_window()` (then at module
scope, no `geometry` parameter, old lines 183-186) computed the GW-side window with the LINEAR
formula `(M_z - k*M_z_sigma)/(1+z_max)`, `(M_z + k*M_z_sigma)/(1+z_min)` for **every** config,
including the three "log" configs (ii)/(iii)/(iv) — contradicting both this document's §2 and
the script's own docstring ("k=3 on BOTH sides"). `pass_mask()` correctly applied the log/exp
transform to the **candidate** side only; the GW side never got it.

**Fix** (`b5_window_count.py`, current file): `gw_window(M_z, M_z_sigma, z_min, z_max, k,
geometry)` now takes an explicit `geometry` argument and branches: `"linear"` unchanged;
`"log"` computes `sigma_lnM_z = M_z_sigma / M_z` (raises `ValueError` for `M_z <= 0` — a guard,
not observed to trigger on this fleet, but disclosed per the panel's request; the *near-zero*-
sigma tail, median fractional error 1.6e-8, is real and unaffected by the guard since it guards
`M_z`, not `M_z_sigma`) and then `gw_lo = M_z*exp(-k*sigma_lnM_z)/(1+z_max)`, `gw_hi =
M_z*exp(+k*sigma_lnM_z)/(1+z_min)` — matching §2 and the script's own docstring exactly. The
per-event `gw_windows_by_k` cache (keyed on `k` alone) is also widened to `gw_windows_by_geom_k`
(keyed on `(geometry, k)`) as a forward-safety fix: with the old cache, two configs sharing a
`k` but different `geometry` would have silently reused one geometry's window for the other; no
config pair in the current `CONFIGS` list shares a `k` across geometries today, so this was not
materially wrong for §7's numbers, but it would have been the next bug had a fifth config been
added. **Smoke-tested by hand** (not by re-deriving the physics): a synthetic `(M_z=1e6,
M_z_sigma=4.13e5, z_min=z_max=0, k=3)` case — the fleet's own observed max fractional error,
per the panel's materiality check — gives `gw_window(..., "log")` = `(289673.75, 3452159.58)`,
matching `M_z*exp(∓k·σ_lnM,z)` computed independently by hand to float precision; the `M_z<=0`
guard raises as expected.

### R2 — corrected §7 numbers (must_fix items 2, 5)

Re-run by the same (different-from-builder) agent, same command
(`uv run python b5_window_count.py`), same catalogue pin (md5 `c52c13b5cab61f6b3f04bbe202550969`)
and fleet base, same git HEAD `a794404c2adcc2857fad8c2f7abc6c7ad08b4159`. The corrected
`b5_window_count.json` (this directory, regenerated 2026-08-29) contains:

| quantity | pre-fix (buggy) | post-fix (corrected) | Δ (absolute) |
|---|---|---|---|
| pass_fraction.i_linear_k1.5 | 0.9576806472967873 | 0.9576806472967873 | 0 (unaffected — linear never touched the bug) |
| pass_fraction.ii_log_k1.5 | 0.40613214027372024 | 0.40613214027372024 | 0 |
| pass_fraction.iii_log_k3.0 | 0.6950855647997026 | 0.6950868985888955 | +1.33e-6 |
| pass_fraction.iv_log_k2.5 | 0.6148892665982285 | 0.6148892665982285 | 0 |
| totals.iii_log_k3.0.n_pass (of 2,249,231) | 1563408 | 1563411 | +3 candidates |
| growth_factor_iii_vs_i.mean | 0.8139465474910743 | 0.8139467452282884 | +1.98e-7 |
| growth_factor_iii_vs_i.{median,p95,max} | unchanged | unchanged | 0 |
| true_host_retention.fraction_retained.* (all 4 configs) | unchanged | unchanged | 0 |

**Every number changes by at most 1.3e-6 in relative terms, and at the precision §7 reports
them (5 significant figures for pass fractions, 3 decimals for the growth factor, 4 decimals
for retention) every single §7 figure is IDENTICAL before and after the fix**
(`round(0.6950868985888955, 5) == round(0.6950855647997026, 5) == 0.69509`, etc. — checked for
all 11 headline numbers). Only config (iii)'s aggregate total moved, by 3 candidates out of
2,249,231 (a fractional shift of 1.3e-6); configs (ii) and (iv) at `k=1.5`/`k=2.5` show a
literal zero-bit change, and **zero** true-host retention counts moved at any config.

**This resolves must_fix item 5's "provisional magnitude" concern empirically, not by
assumption:** on THIS fleet's actual GW-side fractional-mass-error distribution (median 1.6e-8,
mean 0.0042, p95 0.0255, p99 0.0646, max 0.413 — the refuter panel's own materiality-check
numbers, independently reproduced here), the bug's effect on every §7 aggregate headline number
is negligible. This is a fleet-specific empirical finding, not a claim that a linear/log
GW-window mismatch is *always* negligible — a fleet with a heavier tail of GW-side fractional
mass error, or a config pairing a shared-tail-heavy event set against a different `k`, could see
a larger effect; the refuter's materiality concern was a reasonable a priori caution given the
observed CV tail, and it was correctly resolved by re-running rather than by argument.
`b5_window_count.json`'s numbers as of 2026-08-29 (post-fix) are the citable source going
forward; the pre-fix JSON is not retained as a separate artifact (append-only discipline is
served by this note's table, which records both values exactly).

### R3 — stale citation (must_fix item 4)

Confirmed by direct read of `darksiren_emri/galaxy_catalogue/handler.py` at HEAD `a794404c`:
the `mass_filter_sigma` docstring block is at **lines 598-609**, not 648-660 as originally
cited in §1 (the quoted text is verbatim-correct; only the line numbers drifted). The mask-code
citation "`handler.py:654-673`" in §1 is **correct as originally written** — verified against
the live file (`if mass_filter_sigma == "asymmetric":` at line 654 through the closing
`<= (M_z + M_z_sigma * sigma_multiplier) / (1 + z_min)` at line 672-673) — no change needed
there, matching the refuter panel's own independent check.

### R4 — A14 falsifiers (must_fix item 6)

1. **Falsifier for the "linear's negative-lower-edge admits arbitrarily-too-heavy candidates
   for free" / "log closes this loophole" mechanism (§7, first ⚠):** this claim is FALSIFIED
   if an analytic (or Monte-Carlo, at Monte-Carlo precision) derivation of
   `eps_lin_heavy(k=1.5)` — the fraction of the catalogue's candidate mass admitted by linear
   geometry's "too heavy" test purely because that test is vacuous at the catalogue's observed
   CV distribution (median CV ≈ 0.86, `wgeom_result.json` p2 table) — computes to a value that
   does **not** exceed `eps_log(k=3) = 2·Φ(−3) = 0.27%`. If `eps_lin_heavy(k=1.5) ≤ 0.27%`, the
   claimed asymmetry direction (log removes MORE than linear on the heavy tail) reverses or
   dissolves, and the "contradicts the runbook's performance framing" claim would need to be
   withdrawn.
2. **Falsifier for the true-host retention drop being a genuine geometry effect (§7, second
   ⚠):** this claim is FALSIFIED if re-deriving the same retention fractions by calling
   `darksiren_emri.galaxy_catalogue.handler.get_possible_hosts_from_ball_tree` directly (once
   the `mass_filter_geometry`/`mass_filter_k` flags exist in production code, node B5.2) on the
   same fleet yields retention fractions that differ from this instrument's replica by more
   than ±2 percentage points at any config — since `b5_window_count.py`/`pass_mask()`
   REIMPLEMENTS the mass-window test rather than calling the production function, a material
   divergence would mean the measured retention drop is an artifact of the replica's own logic,
   not a property of the log/linear design itself.

### R5 — A15 operating characteristics at N=2,261/2,249,231 (must_fix item 7)

This is a **full deterministic census** over the fixed `bc_9001XX_work` fleet's own rows: every
one of the 2,249,231 candidate rows and every one of the 2,261 events' truth labels is counted
exactly, not estimated from a subsample of a larger pool — so there is no within-fleet sampling
error on the point statistics themselves. The fleet, however, is **one draw** (24 independently
seeded arms, `bc_900101_work`…`bc_900124_work`, 84-108 events each) from the stochastic
EMRI/catalogue-scatter generation process; generalizing beyond this specific 24-arm fleet
carries arm-to-arm (seed-to-seed) fluctuation, which we here MEASURE empirically rather than
assume away, via a new append-only companion script, `b5_window_count_arm_jackknife.py`
(→ `b5_window_count_arm_jackknife.json`, produced 2026-08-29, same agent as this note):

| config | pass-fraction across 24 arms (mean ± std, SE-of-mean) | retention across 24 arms (mean ± std, SE-of-mean) |
|---|---|---|
| (i) linear k=1.5 | 0.9573 ± 0.0161 (SE 0.0033) | 0.9562 ± 0.0232 (SE 0.0047) |
| (ii) log k=1.5 | 0.4119 ± 0.0688 (SE 0.0140) | 0.7010 ± 0.0497 (SE 0.0101) |
| (iii) log k=3.0 | 0.6971 ± 0.1127 (SE 0.0230) | 0.7898 ± 0.0455 (SE 0.0093) |
| (iv) log k=2.5 | 0.6188 ± 0.1027 (SE 0.0210) | 0.7691 ± 0.0504 (SE 0.0103) |

(Source: `b5_window_count_arm_jackknife.json:summary_across_arms.<label>.{pass_fraction_across_arms,retention_fraction_across_arms}`, 2026-08-29.)

Comparing the §7 headline drops against this empirical spread: the aggregate pass-fraction drop
(i)→(iii), 0.9577 → 0.6951 (Δ = −0.263), is ~2.3× config-(iii)'s own arm-to-arm std (0.1127) and
~11× its SE-of-mean (0.0230); the true-host retention drop, 0.9567 → 0.7890 (Δ = −0.168), is
~3.7× config-(iii)'s arm std (0.0455) and ~18× its SE-of-mean (0.0093). Both headline drops
clear a "reliably distinguishable from arm-to-arm noise at this N" bar (conservatively, ≥3
arm-SEs) by a wide margin — this is the closest thing to a false-fail-rate / detectable-effect
statement available from the data in hand: at this N (24 arms), a genuine aggregate change of
roughly ≥0.05-0.07 (pass fraction) or ≥0.03 (retention) would already be distinguishable from
this fleet's own arm-to-arm fluctuation floor, and the observed changes are 3-10× that floor.

**Caveat on this bound:** the 24 arms share the SAME pruned galaxy catalogue and the SAME
selection pipeline — only the EMRI event draws (and their associated scatter) differ across
arms. A genuinely independent fleet (different catalogue realization, different detector-noise
seed at the population level) could show MORE spread than this arm-to-arm figure — so R5's
numbers are a **lower bound** on true fleet-to-fleet variability, not a formal sampling-theory
confidence interval. This is disclosed, not resolved; a fully independent-catalogue replicate
fleet was out of scope for this zero-compute revision.

### R6 — A10 blindness sentence (must_fix item 8)

**True-host retention is NOT a blind measurement.** It is computed by reading each event's own
known `host_galaxy_index`/`in_catalog` truth columns (the production truth-labeling convention,
`main.py:826-830`) directly — the whole point of the metric is to ask "does the mass window,
under each geometry/k, keep the galaxy we already know is the true host?", which by
construction uses the known answer. No analysis choice anywhere in this document or its
companion scripts (which `k` values to report as configs i-iv, which geometry to compare, how
the arm-jackknife groups events) was tuned by peeking at any resulting H0 posterior or MAP
shift — no H0 computation of any kind was performed in producing any number in this document,
`b5_window_count.json`, or `b5_window_count_arm_jackknife.json`; every number here is a
pre-H0, mass-window-membership-only or candidate-count-only statistic.

### R7 — A10 invariants list (must_fix item 9)

Collected explicitly (previously implicit inside §5/§6): this instrument, and any production
implementation of the `mass_filter_geometry`/`mass_filter_k` flags (node B5.2+), MUST preserve:

1. **Default byte-identity** — `mass_filter_geometry="linear"`, `mass_filter_k=sigma_multiplier`
   reproduces the current production mask bit-for-bit (§6 item 1; gate (i) in
   `b5_window_count.json` confirms the census-level aggregate to 4dp: 0.95768 vs target 0.9577).
2. **`k→∞` agreement** — both geometries converge identically to the all-True (no filter) mask
   as `k→∞` (§5 item 1); a new geometry must not resurrect or diverge from the already-exonerated
   HB hard-truncation object at that limit.
3. **`σ→0` point-overlap agreement** — both geometries degenerate to the same
   `M/(1+z) == M_z/(1+z)` point test as `σ→0` (§5 item 2); no discontinuity at the
   zero-uncertainty edge.
4. **First-order (`kσ≪1`) agreement** — log and linear windows agree to `O(σ²)` when the
   catalogue's CV is small (§5 item 3); divergence between the geometries is a genuinely
   higher-order effect, not a discretization artifact.

Regression-plan item 1 (§6) formally tests invariant 1. Invariants 2-4 currently rest on the
derivation in §5 plus this revision's hand-check (R1) at one CV value; they do **not** yet have
a dedicated automated regression test sweeping a range of `σ`/`k` values toward each limit —
flagged here as a residual gap in the regression plan (§6), to be closed before, not after, any
production adoption gate.

### R8 — A8 two-sided bands (must_fix item 10)

The §7 pass-fraction, growth-factor, and retention numbers are point statistics of a **full
census** (every row/event counted, nothing sampled) — there is no within-fleet sampling error
to attach as a classical band to a single-fleet point estimate. The uncertainty that IS
meaningful here is generalization beyond this specific 24-arm fleet, which R5 supplies
empirically. Restating R5's numbers as explicit bands on the headline §7 ratios:

- pass fraction (i) 0.9577 → **0.9573 ± 0.0161** (1 arm-SD, N=24 arms) / ± 0.0033 (SE-of-mean)
- pass fraction (iii) 0.69509 → **0.6971 ± 0.1127** / ± 0.0230
- pass fraction (ii) 0.40613 → **0.4119 ± 0.0688** / ± 0.0140
- pass fraction (iv) 0.61489 → **0.6188 ± 0.1027** / ± 0.0210
- retention (i) 0.9567 → **0.9562 ± 0.0232** / ± 0.0047
- retention (iii) 0.7890 → **0.7898 ± 0.0455** / ± 0.0093
- retention (ii) 0.7001 → **0.7010 ± 0.0497** / ± 0.0101
- retention (iv) 0.7682 → **0.7691 ± 0.0504** / ± 0.0103

Growth-factor percentiles (mean/median/p95/max) are NOT re-derived per-arm in this revision
(the arm-jackknife script reports pass-fraction and retention only); no band is supplied for
those four numbers — a gap disclosed here rather than papered over with an assumed band.

### R9 — A11 per-number provenance (must_fix item 11)

- Pass fractions (post-fix): `b5_window_count.json:pass_fraction.{i_linear_k1.5,ii_log_k1.5,
  iii_log_k3.0,iv_log_k2.5}`, regenerated 2026-08-29 (this revision).
- Growth factor: `b5_window_count.json:growth_factor_iii_vs_i.{mean,median,p95,max,
  n_events_with_nonzero_linear_candidates,n_events_zero_linear_candidates_gain_some_under_iii,
  n_events_zero_under_both}`, regenerated 2026-08-29.
- True-host retention: `b5_window_count.json:true_host_retention.fraction_retained.<label>`
  and `.n_retained.<label>`, regenerated 2026-08-29.
- Gate (i) reproduction: `b5_window_count.json:gate_i_reproduces_0.9577` (unchanged by this
  revision — target/source citation as originally written stands).
- Arm-to-arm std/SE (Notes R5, R8): `b5_window_count_arm_jackknife.json:
  summary_across_arms.<label>.{pass_fraction_across_arms,retention_fraction_across_arms}.
  {mean,std,min,max}`, produced 2026-08-29.
- Bug-fix hand-check (Note R1): computed inline, this session, 2026-08-29 (not banked as a
  JSON artifact — a one-off smoke test, reproducible from the values quoted in R1 with any
  Python/NumPy).
- A ledger row recording this revision has been appended to
  `docs/gates/PHYSICS-GATE-LEDGER.md` (2026-08-29, after the original "presented" row for this
  document — append-only, that row is unedited).

### Standing rule (5) re-check

The mechanism this document and its revisions probe (mass-window *geometry*, linear vs
log-symmetric) remains outside every exoneration in
`EXONERATION_REGISTER_20260827.md` (HB's "hard mass window as support truncation" is scoped to
the k→∞ truncation limit, not geometry — confirmed again by direct grep of that file's WGEO
scoping language, lines ~217-223, 325) and outside
`BIAS_HISTORY_LEDGER.md` §2's "DO NOT RE-TRY" list item 4 ("hard support truncation / hard
clamp in production" — refuted under observed-z membership, N-2d, a DIFFERENT claim from
window geometry). No re-litigation collision found on this re-check; consistent with the
refuter panel's own independent finding.

---

## Implementation record (appended 2026-08-29, charter node B5.1 part B)

Launched under rows #222/#223. A **different agent** from whoever authored the presentation
text above and the Revision Notes R0-R9 implemented the code (append-only: nothing above this
line is edited). Full diff summary, byte-identity evidence, the count-factor table, the exact
list of files to commit, and the wave-2 counterfactual arm shape are in the companion record
`B5_1_WIN_RECORD.md` (this directory) and in `docs/gates/PHYSICS-GATE-LEDGER.md`'s
2026-08-29 "implemented"/"verified" rows for this node. Summary:

- `mass_filter_geometry: Literal["linear", "log"] = "linear"` and `mass_filter_k: float = 1.5`
  added to `GalaxyCatalogueHandler.get_possible_hosts_from_ball_tree`, threaded through
  `BayesianStatistics` (class defaults, `__init__`, `evaluate()`, the single call site),
  `correspondence_1d.run_mirror_seed_inprocess`, `arguments.py` (`--mass_filter_geometry`,
  `--mass_filter_k`), and `main.py` (both the CLI `main()` path and the module-level
  `evaluate()` helper) — exactly the code shape sketched in §2, implemented literally where
  §2 was literal (the "linear"/"log" branch bodies) and resolved conservatively, and disclosed
  as such, where §2's illustrative snippet was silent: §2's code sketch shows no
  `mass_filter_sigma` interaction at all (it plugs a bare `M_err` into the log formula), but
  invariant 1 (default byte-identity, §5/R7) REQUIRES the existing `mass_filter_sigma`
  "symmetric"/"asymmetric" split to keep functioning under the new default pairing. Resolution
  implemented: the candidate-side multiplier that `mass_filter_sigma` selects is now
  `mass_filter_k` (was `sigma_multiplier`) for "symmetric" and `1.0` for "asymmetric",
  under EITHER geometry — the two flags are read independently, each at its own single site,
  matching this document's own §2 framing of them as "two new independent flags." This
  preserves default byte-identity exactly (`mass_filter_k` defaults to the same `1.5` literal
  the call site already hardcodes for `sigma_multiplier`) while giving "log" a well-defined
  `mass_filter_sigma` semantics instead of silently ignoring the flag.
- `sigma_multiplier` now feeds ONLY the sky-cone search radius (as intended by §2); the mass
  window reads `mass_filter_k` exclusively.
- Regression plan §6 items 1-3 implemented as automated tests
  (`darksiren_emri_test/test_mass_filter_geometry.py`, 19 cases) plus a signature-passthrough
  test on `run_mirror_seed_inprocess`. R7's disclosed "no dedicated automated regression test"
  gap is partially closed: invariants 2 (`k→∞`) and 3 (`σ→0`) now have dedicated tests for
  both geometries; invariant 4 (first-order agreement) was not additionally pinned (direct
  algebraic consequence of `exp(x)=1+x+O(x²)`, not separately tested).
- R4 falsifier item 2 (re-deriving the fleet-level true-host retention drop against the
  now-existing production flags, to check whether `b5_window_count.py`'s REIMPLEMENTED
  mass-window logic diverges from the real production function by more than ±2 points) is
  explicitly **NOT attempted in this row** — it requires a fleet-scale run and is charter node
  B5.2 / wave 2's job, not this implementation task's. No adoption decision is made or implied
  here; both new flags default to the byte-identical pre-flag values, matching every other
  instrumentation flag in this codebase's convention.
- No commit was made by this agent (orchestrator commits); the exact file list is in
  `B5_1_WIN_RECORD.md`.
