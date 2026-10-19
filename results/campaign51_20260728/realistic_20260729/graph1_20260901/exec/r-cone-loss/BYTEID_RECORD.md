# r-cone-loss — BYTE-ID VERIFICATION RECORD (independent verifier, b-cone-scorer)

Role: INDEPENDENT BYTE-ID VERIFIER (separate agent from the builder of
`cone_loss_reads.py`). Scope: ONLY the G-2 double-anchor reproduction named in
`REGISTRATION_DRAFT.md` §5 ("G-2 anchors (instrument byte-id)") — R-MKER-6
(`p3_2d_fleet_20260825/bc_900121_work/seed900121`, event_idx 20) and CMEM-A1
(`p3_b0_work/bc_900101_work/seed900101`, event_idx 0). The registered statistic
(Δh_cone/φ_cone/SE/Z) was **not run** — out of scope per task and per the draft's own
verifier-independence contract.

## Method

Did **not** trust the builder's `--dry-run` printout as the check. Wrote a standalone
script, `byteid_check.py` (this directory), that:

- imports only production code (`GalaxyCatalogueHandler`, `_polar_to_cartesian`) —
  not `cone_loss_reads.py` — so a bug shared between builder and verifier code paths
  cannot manufacture a false GREEN;
- reads the two anchor CRB files directly from disk at the named `event_idx` rows;
- recomputes chord via TWO independent embeddings: (a) the handler's own
  `_polar_to_cartesian` (production code every consumer shares) and (b) a
  hand-written `independent_embed()` re-derivation of the standard
  (θ = polar, φ = azimuth) → Cartesian convention, written from scratch with no
  code shared with the handler — as a convention cross-check, not just a re-run;
- recomputes radius via `k*sqrt(lambda_max(J Σ' Jᵀ))` from the CRB's own Fisher sky
  sub-block, independently coded (not imported from the builder's `cone_radius()`).

## Result (verbatim run output)

```
INDEPENDENT BYTE-ID CHECK
--- R-MKER-6 (event_idx, hidx=6791134) ---
  chord_handler_embed: 0.001674659860716462
  chord_independent_embed: 0.001674659860716462  (embed methods agree to 0.0)
  chord_expected: 0.00167466
  chord_dev: 1.3928353782832747e-10   (tol 5e-10)  chord_pass: True
  radius_found: 0.0014956979545757095
  radius_expected: 0.0014956979545757095
  radius_dev: 0.0                     (tol 1e-15)  radius_pass: True
--- CMEM-A1 (event_idx, hidx=10791058) ---
  chord_handler_embed: 0.01166569410071811
  chord_independent_embed: 0.01166569410071811  (embed methods agree to 0.0)
  chord_expected: 0.0116656941007181
  chord_dev: 1.0408340855860843e-17   (tol 5e-10)  chord_pass: True
  radius_found: 0.035912194615445196
  radius_expected: 0.0359121946154451
  radius_dev: 9.71445146547012e-17    (tol 1e-15)  radius_pass: True

n_pairs=4 max_abs_dev=1.392835e-10 verdict=GREEN
```

- `hidx` (the row's `host_galaxy_index`) resolved directly as a positional index into
  `GalaxyCatalogueHandler(...).reduced_galaxy_catalog.reset_index(drop=True)`
  (20,834,171 rows; both anchor indices, 6,791,134 and 10,791,058, are well within
  range) — matching the builder's own indexing convention. No translation via
  `resolve_host_recovery_position` was needed or applied (that method is for a
  different pruning-position context per its own docstring; the CRB's
  `host_galaxy_index` is set at simulation time directly as `host_galaxy.catalog_index`
  — `main.py:837` — into this same catalogue instance).
- The two independently-coded sky→Cartesian embeddings (handler's production
  `_polar_to_cartesian` vs. a hand-written from-scratch re-derivation) agree to
  machine epsilon (0.0 and ~1e-17) on both anchor points — no embedding-convention
  ambiguity found.

## Draft tolerance quoted

`REGISTRATION_DRAFT.md` §5: "reproduce R-MKER-6's anchor ... (chord 1.674660e-03 ±
5e-10, radius 1.4956979545757095e-03 ± 1e-15 — `cmem_reads.py:32,107-111`) AND
CMEM-A1's anchor ... (0.0116656941007181 / 0.0359121946154451, [same tolerances])".
Both pairs measured here are within these stated tolerances (chord within 5e-10;
radius within 1e-15, and radius deviation is 0.0 for one anchor).

## Verdict

**GREEN.** Both double-anchor pairs (chord, radius) for R-MKER-6 and CMEM-A1
reproduce to within the draft's own stated tolerances, via an independently-coded
path that does not import or re-run the builder's script. n_pairs = 4 (2 anchors ×
{chord, radius}). max_abs_dev = 1.392835e-10 (the R-MKER-6 chord vs. its 6-significant-
-digit registered value; well inside the 5e-10 tolerance quoted for that pair).

This record covers ONLY the G-2 byte-id gate. It says nothing about G-1/G-3/G-4 (see
`BUILD_RECORD.md` for the builder's own G-4 RED-envelope finding, which this
verification does not touch) and nothing about the registered statistic, which was
not run.
