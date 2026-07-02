# FRAME-AUDIT: EMRI Dark-Siren Sky-Angle (qS/phiS) Coordinate Frame

Definitive audit of the coordinate frame of every sky-angle (`qS`/`phiS`) and sky-covariance
quantity in the EMRI dark-siren H0 pipeline. Synthesises a per-subsystem frame audit with
adversarial verification of four load-bearing claims. All four claims were **CONFIRMED**.

Scope: `master_thesis_code/` (production package), `scripts/` (migration/merge/prepare),
`cluster/` (job chain). Cited as `file:line` throughout.

---

## 1. CANONICAL FRAME PIPELINE

**There is exactly ONE coordinate rotation in the entire pipeline, and it happens at catalog
ingestion.**

The GLADE+ reduced catalog on disk is **equatorial ICRS** (RA/Dec in degrees; raw columns 8/9,
`galaxy_catalogue/handler.py:139-140`). At `GalaxyCatalogueHandler.__init__`
(`handler.py:202-203`, COORD-03 / Phase 36, commit `b460297`, 2026-04-22) the catalog is rotated
**in place** to **ecliptic `BarycentricTrueEcliptic(equinox='J2000')`**:

- `_rotate_equatorial_to_ecliptic()` (`handler.py:777-814`): `SkyCoord(icrs).transform_to(BarycentricTrueEcliptic(J2000))`, writing `lon/lat` back into the SAME `PHI_S`/`THETA_S` columns.
- `_map_angles_to_spherical_coordinates()` (`handler.py:816-832`): deg→rad and latitude→polar angle, `phi = lon` in `[0,2pi)`, `theta = pi/2 - lat` in `[0,pi]`.

**After that single rotation, EVERYTHING downstream is ecliptic `BarycentricTrueEcliptic(J2000)`,
phi = ecliptic longitude (rad), theta = ecliptic colatitude/polar angle in `[0,pi]` (rad):**

| Stage | Quantity | File:line | Frame |
|---|---|---|---|
| Catalog in-memory | `PHI_S`/`THETA_S` columns | `handler.py:202-203` | ecliptic |
| 2-D / 4-D BallTrees | `_polar_to_cartesian(PHI_S,THETA_S)` | `handler.py:323-329, 462-481` | ecliptic |
| Host objects | `HostGalaxy.phiS/qS` | `handler.py:64-65` | ecliptic |
| Event injection | `parameter_space.qS/phiS = host.qS/phiS` | `parameter_space.py:205-206` | ecliptic |
| Waveform / Fisher | `ResponseWrapper(index_beta=7,index_lambda=8,is_ecliptic_latitude=False)` | `waveform_generator.py:64` | ecliptic (SSB) |
| CRB CSV positions | `qS`/`phiS` row values | `parameter_estimation.py:471-483` | ecliptic |
| CRB CSV covariance | `delta_qS_delta_qS`, `delta_phiS_delta_phiS`, `delta_phiS_delta_qS` | `parameter_estimation.py:432-440` | ecliptic (rad^2) |
| Inference event read | `Detection.phi/theta` | `detection.py:113-116` | ecliptic |
| Inference host query | `get_possible_hosts_from_ball_tree(det.phi,det.theta)` | `bayesian_statistics.py:1175-1186` | ecliptic |
| Completeness pixel | `ang2pix(detection.phi, detection.theta)` | `bayesian_statistics.py:1444` | ecliptic |
| m_th HEALPix map | `build_m_th_map` rotates raw-disk RA/Dec → ecliptic | `pixel_completeness.py:401-407` | ecliptic-keyed |
| Dark-host draw | `sample_sky_in_pixels` | `pixel_completeness.py:297-311`, `dark_siren_injection.py:373-426` | ecliptic |

**Why ecliptic is physically MANDATORY, not a convention choice** (VERIFY claim 1, confirmed):
`fastlisaresponse.ResponseWrapper` builds the GW propagation vector `k` and polarization basis from
`(lambda=phiS, beta=qS)` as ecliptic longitude/colatitude and dots them against the LISA spacecraft
positions from `ESAOrbits`, which live in the SSB ecliptic frame (`response.py:459-525`). A `k·r`
projection is only physically meaningful if sky direction and orbit share that frame. Hence the
waveform — the very thing the Fisher matrix differentiates — only accepts ecliptic angles, so the
stored `qS`/`phiS` and their Fisher covariance ARE ecliptic by construction. `LISA_configuration.py`
is frame-neutral (PSD only, sky-averaged patterns; no per-source angle input).

**Empirical corroboration:** 99.2% of seed-600 events sit `<0.001 deg` on a catalog galaxy *in the
ecliptic frame* — exactly what this code path predicts.

> **A rotation is needed EXACTLY ONCE — and ONLY for legacy pre-COORD-03 (pre-commit `b460297`)
> equatorial CRBs. A fresh post-COORD-03 run is ecliptic-native and must NEVER be rotated again.**
> Rotating a fresh run is a double-rotation: ~0.07-0.2 deg typical, up to ~6 deg off the true host,
> plus a spurious position-dependent Jacobian applied to an already-ecliptic covariance.

---

## 2. THE AMBIGUITY

**Ecliptic-native (fresh) and equatorial (legacy) CRB CSVs are byte-indistinguishable without a
frame marker — and the system never writes one where it would resolve the ambiguity.**

Three facts collide:

1. **The simulation writes NO marker.** `save_cramer_rao_bound` assembles the row as
   `_parameters_to_dict() | crb_dict | {T, dt, SNR, generation_time, host_galaxy_index, in_catalog, _simulation_index}` (`parameter_estimation.py:471-483`). There is no `_coord_frame` / `_cov_frame`
   key. The only disk writer, `flush_pending_results` `to_csv` (`parameter_estimation.py:513-515`),
   appends exactly those keys. So fresh CRBs are ecliptic in content but **UNMARKED**. Merge
   (`scripts/merge_cramer_rao_bounds.py:96-100`) only `pd.concat`s — no markers added.

2. **The consumer guard DEMANDS the marker.** `Detection.__init__` (`detection.py:96-108`) hard-raises
   `ValueError` unless both `_coord_frame` AND `_cov_frame == "ecliptic_BarycentricTrue_J2000"`. The
   same guard fires at `bayesian_statistics.py:667, 788, 1161` and at `scripts/prepare_detections.py:113`
   (the `emri-prepare` step of `cluster/merge.sbatch:66`). A fresh cluster run therefore **hard-fails
   at the prepare stage** — it does not silently mis-evaluate; it crashes.

3. **The only committed marker-writer ALSO rotates.** `scripts/migrate_crb_to_ecliptic.py` is the
   only non-test writer of the markers. Its sole frame discriminator is marker **presence/absence**
   (`migrate_crb_to_ecliptic.py:190-196`). Marker-absence is equated with "is equatorial," but BOTH
   pre-COORD-03 (truly equatorial, needs rotation) AND post-COORD-03 (ecliptic-native, must NOT
   rotate) CRBs are unmarked. State 3 (`:226-261`) then unconditionally runs `_icrs_to_ecliptic` on
   `qS`/`phiS` AND `R·Sigma·R^T` on the covariance, then stamps the markers. There is no
   `--no-rotate` / mark-only flag anywhere (grep `no_rotate`/`mark_only`/`skip_rotat` → empty).

**Why the guard cannot save us:** it is a LABEL check, not a rotation-state check. It passes
identically for (a) ecliptic-native data with markers added without rotation [correct — the
seed500 case], (b) genuine equatorial data correctly single-rotated by migrate [correct], and
(c) ecliptic-native data DOUBLE-ROTATED by migrate [WRONG, ~6 deg off, markers still say ecliptic].
The range asserts in migrate (`:233-238`) pass for single- and double-rotated angles alike
(ecliptic-of-ecliptic still lands in `[0,2pi)`/`[0,pi]`), giving false confidence.

**Consequence (the trap):** the guard's own error text (`detection.py:101-103`) instructs the
operator to "run migrate_crb_to_ecliptic.py before evaluation" — i.e. it actively steers a fresh
ecliptic-native run into the exact script that double-rotates it. The cluster chain
(`cluster/submit_pipeline.sh:92-123`, `merge.sbatch:57,66`) invokes no migration, so there is **no
committed correct path**: a fresh run can only reach `--evaluate` via (a) a manual, uncommitted
marker-add without rotation, or (b) the wrong rotating migration. (VERIFY claim 4, confirmed.) The
one run that ever evaluated, `results/run_20260620_seed500_phase50`, has markers, is 100%
on-galaxy, and has NO `.bak_equatorial` — i.e. markers were added by hand WITHOUT rotation, an
uncommitted ad-hoc fix that is not reproducible from committed code.

A consumer cannot tell ecliptic from equatorial because the misnomer column names (Section 3)
make "is this equatorial?" unanswerable by column inspection — the disambiguation is purely
positional (which code path read it).

---

## 3. MISNOMERS TO RENAME

| # | Misnomer | File:line | Why wrong | Proposed name | Severity |
|---|---|---|---|---|---|
| M1 | `InternalCatalogColumns.PHI_S = "RIGHT_ASCENSION"` | `handler.py:156` | Holds ecliptic longitude (rad) after COORD-03, not equatorial RA | `"ECL_LON"` (or `"PHI_ECLIPTIC"`); keep Python const `PHI_S` | high |
| M2 | `InternalCatalogColumns.THETA_S = "DECLINATION"` | `handler.py:157` | DOUBLE wrong: a declination is a latitude in `[-90,90]` deg, but the value is a polar angle in `[0,pi]` rad — quantity TYPE changed (lat→colatitude) AND frame (eq→ecl) | `"ECL_THETA_POLAR"` (or `"THETA_COLAT_ECLIPTIC"`) | high |
| M3 | `read_reduced_galaxy_catalog()` returns columns named `RIGHT_ASCENSION`/`DECLINATION` | `handler.py:311-315` | These ARE equatorial deg (pre-rotation on disk) — same names mean equatorial-on-disk vs ecliptic-in-memory with no marker | Add docstring "returns RAW equatorial ICRS degrees (pre-rotation)"; keep disk header but document | high |
| M4 | `_polar_angle_to_declination(polar_angle) = pi/2 - polar_angle` | `handler.py:1001-1002` | Returns a LATITUDE (ecliptic latitude on rotated catalog), not an equatorial declination; also DEAD CODE (no callers) | Delete; or rename `_polar_angle_to_ecliptic_latitude` | low |
| M5 | `.bak_equatorial` backups written by migrate | `migrate_crb_to_ecliptic.py:249-251` | On ecliptic-native input the backup is NOT equatorial — the name lies about the frame of its contents | `.bak_premigration` (frame-neutral); refuse to write on ecliptic-native data (see Fix B) | medium |
| M6 | migrate State-3 docstring "unmarked => equatorial => rotate" | `migrate_crb_to_ecliptic.py:190-196, 226-230` | Stale post-COORD-03: unmarked now also means ecliptic-native | Rewrite docstring: "unmarked is AMBIGUOUS; require explicit `--assume-equatorial` + content check" | high |
| M7 | `interactive_sky_map` / `sky_plots` render ecliptic on unmarked geographic mollweide | `interactive.py:576-660`, `sky_plots.py:78-80` | A viewer reads the graticule as equatorial RA/Dec; data is ecliptic (off by the ~23.4 deg obliquity) | Add title/label "Ecliptic (BarycentricTrueEcliptic J2000)"; amend docstring | low |

Note: the CRB CSV itself uses neutral column names `qS`/`phiS`, so the M1/M2 misnomer does NOT leak
into the CSV — the risk is internal to handler/catalog readers.

---

## 4. EQUATORIAL-ASSUMING CONSUMERS

**Within `master_thesis_code/` (the production package): NONE.** (VERIFY claim 2, confirmed.)

Every active consumer of `qS`/`phiS`, the catalog `PHI_S`/`THETA_S` columns, and the CRB
`phi`/`theta` treats them as ecliptic `BarycentricTrueEcliptic(J2000)` and is mutually consistent:
BallTrees (`handler.py:317-331, 462-481`), `Detection` (`detection.py:96-126`), GW Gaussian means
(`bayesian_statistics.py:869,885`), `ang2pix` (`bayesian_statistics.py:1444`), `single_host_likelihood`
(`bayesian_statistics.py:1269-1289`), the pixelated dark-host draw (`dark_siren_injection.py:373-426`),
and `parameter_space.set_host_galaxy_parameters` (`parameter_space.py:205-206`). A package-wide grep
for `SkyCoord`/`transform_to`/`BarycentricTrueEcliptic`/`galactic`/`ICRS` finds rotations ONLY at
catalog LOAD (`handler.py:777-814`) and m_th build from raw disk (`pixel_completeness.py:401-407`) —
no second rotation on already-ecliptic data. Frame-independent code (does not assume any frame):
`_polar_to_cartesian` (`handler.py:977-998`), `simulation_detection_probability.py:751-998` (sky
marginalized/unused), `LISA_configuration.py` (PSD + sky-averaged patterns), the isotropic dark-host
fallback (`dark_siren_injection.py:359-370`), `glade_completeness.ang2pix` (single all-sky pixel).
The dead `_get_closest_possible_host` path is commented out (`bayesian_statistics.py:1196-1208`).

**Outside `master_thesis_code/` — the ONLY equatorial-assuming consumer in the repo:**

- **`scripts/migrate_crb_to_ecliptic.py` State 3** (`:226-261`). It assumes BOTH the stored
  `qS`/`phiS` positions AND the Fisher covariance block are equatorial ICRS and rotates both
  (`_icrs_to_ecliptic` `:70-83` + `_rotate_covariance_block` `:127-174` with the Jacobian embedded
  at rows/cols 7-8). On post-COORD-03 ecliptic-native CRBs this is a double-rotation of positions AND
  covariance. (VERIFY claim 3, confirmed: the covariance corruption is NOT a relabel — the eq↔ecl
  Jacobian in `(theta,phi)` is position-dependent with `sin(theta)` factors, so it rotates and
  rescales the error ellipse by a position-dependent amount.)
- **Indirect consumer:** the guard's remediation text (`detection.py:101-103`) that points operators
  at the script above.

Latent (no production caller, but ecliptic-assuming without documentation — not equatorial bugs):
`get_possible_hosts` (`handler.py:686-733`), `find_closest_galaxy_to_coordinates`
(`handler.py:484-513`), `ParameterSample.phi_S/theta_S` (`handler.py:42-47`),
`get_random_hosts_in_mass_range` (`handler.py:857-879`).

---

## 5. CONCRETE FIX PLAN

Goal: make the frame **self-describing and unambiguous forever** — born marked at the source so the
guard passes natively, migration never fires, and double-rotation becomes impossible. Prioritised,
minimal-diff.

### (a) PRIORITY 1 — Stamp the markers at CRB-write time in the simulation [SOFTWARE]

The data is already ecliptic; we are only recording provenance, not changing any number.

- **File:line:** `parameter_estimation.py:474-482` — add to the metadata dict in `save_cramer_rao_bound`,
  alongside `T`/`dt`/`SNR`/`host_galaxy_index`/`in_catalog`:
  ```python
  "_coord_frame": "ecliptic_BarycentricTrue_J2000",
  "_cov_frame": "ecliptic_BarycentricTrue_J2000",
  ```
- Single canonical insertion point: every row passes through this union, so all flushed/appended
  rows carry the columns and stay aligned. Alternative (worse) point: inject columns at the
  `to_csv` in `flush_pending_results` (`parameter_estimation.py:513-515`).
- **Classification: SOFTWARE change.** No formula, constant, waveform parameter, or computed value
  changes — it only records the frame the data already has. (Not a `/physics-change` trigger: no
  numeric output differs.) Add a regression test asserting the two columns are present and equal to
  the canonical string on fresh output.
- **Effect:** fresh CSVs are "born as State 1" (fully marked) → `Detection` guard passes natively →
  migrate State 3 never triggers → double-rotation impossible.

### (b) PRIORITY 1 — Make migrate REFUSE already-ecliptic / marked data [SOFTWARE, defensive]

- **File:line:** `migrate_crb_to_ecliptic.py:190-196` (state discriminator) and `:226-261` (State 3).
- Make State 3 refuse by default. Require explicit opt-in `--assume-equatorial`, gated on a source
  `git_commit` predating `b460297`. Add a **content-based** equatorial test: cross-match stored
  `qS`/`phiS` against GLADE in BOTH frames (reuse the handler rotation). If events already sit on
  catalog galaxies in the ECLIPTIC frame (as 99.2% of seed-600 do), abort:
  `"input is already ecliptic; do NOT migrate"`. Only rotate if they match in the EQUATORIAL frame.
- Replace the insufficient range asserts (`:233-238`) with this on-galaxy cross-match.
- If already-marked → skip/park (idempotent), never re-rotate.
- **Classification: SOFTWARE change** (adds guards/refusal logic; rotation math unchanged). It
  touches a script that performs a physics transform, but the change only PREVENTS a wrong
  transform — no formula altered. Treat the rotation math itself as physics if ever modified.

### (c) PRIORITY 2 — Fix the guard's remediation message [SOFTWARE]

- **File:line:** `detection.py:101-103`. Stop pointing at the double-rotation trap. New text:
  `"markers missing — if this CSV was produced post-COORD-03 (commit b460297) it is already`
  `ecliptic: stamp markers WITHOUT rotating. Run a rotating migration ONLY for genuine`
  `pre-Phase-36 legacy equatorial CRBs."` After fix (a) this message becomes moot for fresh runs.
- Optional hardening: add a one-time on-galaxy sanity check at load (fraction of events `<0.001 deg`
  on a catalog galaxy in ecliptic) and fail loud if it drops — distinguishes single- from
  double-rotated data, which the label cannot.
- **Classification: SOFTWARE change** (string + optional assert; no computed value changes).

### (d) PRIORITY 3 — Renames / docs [SOFTWARE]

- M1/M2: rename `InternalCatalogColumns.PHI_S`/`THETA_S` string values to `ECL_LON`/`ECL_THETA_POLAR`
  with a one-time alias for the reduced-CSV header; keep the `PHI_S`/`THETA_S` Python constants
  (`handler.py:156-157`). Add a range assert `0 <= theta <= pi` after
  `_map_angles_to_spherical_coordinates` (`handler.py:816-832`).
- M3: docstring on `read_reduced_galaxy_catalog` (`handler.py:311-315`): "RAW equatorial ICRS degrees,
  pre-rotation."
- M4: delete unused `_polar_angle_to_declination` (`handler.py:1001-1002`).
- M5: rename `.bak_equatorial` → `.bak_premigration` (`migrate_crb_to_ecliptic.py:249-251`).
- M6: rewrite migrate State-3 docstring (`:190-196`).
- M7: frame label/title on `interactive_sky_map`/`sky_plots` (`interactive.py:576-660`, `sky_plots.py:78-80`).
- **Shared-rotation dedup (HIGH):** extract one helper
  `rotate_equatorial_to_ecliptic(ra_deg, dec_deg) -> (lon, lat)` and call it from BOTH
  `handler._rotate_equatorial_to_ecliptic` (`handler.py:800-801`) and
  `build_m_th_map` (`pixel_completeness.py:401-407`), so the map that BUILDS `f` and the catalog
  that QUERIES `f` can never desync. Add a regression test pinning a few `(RA,Dec)→pixel` through
  both paths.
- **Classification: SOFTWARE change** (renames, docstrings, dead-code deletion, refactor extracting
  an identical transform; asserts only fail-loud, they do not change values). Add the closure/
  regression tests as part of this item.

### (e) PRIORITY 1 (operational) — One-time correct backfill for the existing seed-600 run [SOFTWARE]

- The existing seed-600 CRB output is ecliptic-native and unmarked. Add the markers WITHOUT rotating:
  write `_coord_frame=_cov_frame="ecliptic_BarycentricTrue_J2000"` directly into the merged CSV
  (a stamp-only pass — e.g. `migrate --stamp-only` once fix (b) adds that flag, or a 3-line pandas
  script). Verify post-stamp it is 100% on-galaxy in ecliptic (matching
  `results/run_20260620_seed500_phase50`).
- Do NOT run the rotating migration on it. Do NOT create a `.bak_equatorial`.
- **Classification: SOFTWARE change** (provenance stamp on already-correct data; no coordinate value
  changes). This unblocks `--evaluate` for the seed-600 run from committed code.

### Transitional note for the cluster chain

After fix (a), `cluster/submit_pipeline.sh` / `merge.sbatch` work unchanged (fresh CSVs are born
marked). If a transitional bridge is needed before the simulation is patched, insert a
**STAMP-ONLY (no-rotation)** step in `merge.sbatch` BEFORE `emri-prepare` (`merge.sbatch:66`),
guarded to run only on unmarked-but-ecliptic data. **Never** insert the rotating migration into the
cluster chain.

---

### Verdict summary

| Claim | Verdict |
|---|---|
| Waveform/response REQUIRES ecliptic qS/phiS (ecliptic storage physically mandatory) | CONFIRMED |
| No consumer in `master_thesis_code/` assumes equatorial post-COORD-03 | CONFIRMED |
| Fisher covariance block is already ecliptic; migration double-rotates it too | CONFIRMED |
| No committed path lets a fresh run satisfy the guard without manual-add or wrong migration | CONFIRMED |

Root cause: the simulation emits ecliptic-native data without provenance markers (fix a), the guard
is a label check that cannot detect double-rotation (fix c), and the only committed marker-writer
conflates "unmarked" with "equatorial" and rotates unconditionally (fix b). The forward fix is
source-stamping; the migration refusal is the defensive backstop.
