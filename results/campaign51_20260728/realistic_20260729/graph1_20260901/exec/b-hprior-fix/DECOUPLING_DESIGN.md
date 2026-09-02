# b-hprior-fix — DECOUPLING DESIGN + /physics-change PRESENTATION (docket item 4, option (a))

Date: 2026-09-02. Node: b-hprior-fix follow-on (Research Graph 1, Branch I), standing
top-tier prereg/derivation author. Authorization: row #301 item 4 [RULE] option (a)
(verbatim scope: "the h-prior decoupling: a separate admissibility mechanism for the G-EXT
grid, the host-window bound left untouched; the design detail returns as its own
`/physics-change` gate (NOT covered by this row)") on the row #293 chair adjudication.
**No code edit is applied by this node. No commit. No cluster submission.** The diffs below
are DRAFT for the chair's `/physics-change` gate run and the author's ratification.
Predecessor record: `RECORD.md` in this directory (row #293); its §2 one-liner
(`upper_limit=0.86 → 1.00`) is SUPERSEDED by this design — the chair adjudication found it
cannot land (it would widen every detection's candidate-host window and break the ratified
g-byte-id gate below 0.86).

## 1. The two roles of `h.upper_limit`, and the complete consumer map (re-verified today)

`self.cosmological_model.h.upper_limit` (constructed at `cosmological_model.py:388`,
literal 0.86) is consumed at exactly three places (repo-wide grep, source + tests):

| site | role | behavior under a raised bound |
|---|---|---|
| `bayesian_statistics.py:4655-4659` (evaluate() entry guard) | **grid ADMISSIBILITY**: pure gate, `raise ValueError("Hubble constant out of bounds.")`; feeds no computation | must admit the ratified G-EXT wing (h ≤ 1.00, AMENDMENT G-EXT, row #284) |
| `bayesian_statistics.py:5712-5722` (`get_redshift_outer_bounds(... h_max=self.cosmological_model.h.upper_limit)`) | **HOST-WINDOW bound**: `z_max = dist_to_redshift(d_L + 3σ, h_max)` (`physical_relations.py:546-567`), monotone increasing in h_max; sets every detection's candidate-host z-window; the `min(z_max, redshift_upper_limit)` clamp at `:5722` never bites for h_max ≤ 0.86 (row #293 chair adjudication; the `:1247-1259` comment: z_max(h ≤ 0.86) ≤ ~1.33 < 1.5) | must continue to receive **EXACTLY 0.86** — any change breaks byte-identity below 0.86 for every in-bound evaluation |
| `validation/correspondence_1d.py:3398-3399` (mirror harness) | runtime widening `h.upper_limit = max(h.upper_limit, eff_hi)` to cover the mirror's h-list ([P3-HGRID], rows #182-#184); its own comment states the widened bound "feed[s] the per-event candidate-ball z-window" and is proven bit-exact against banked b0i CSVs | must keep working unchanged — it is deliberate window-widening, not admissibility-only, and is out of this design's scope |
| (`H_MAX: float = 86.0` at `constants.py:24`) | declared but **dead in source** — no consumer found (only `H_MIN` is consumed, `cosmological_model.py:231-234`); NOT the source of the 0.86 literal | untouched; noted so nobody mistakes it for the mechanism |

The mirror site is the decisive design constraint the one-liner missed twice over: the
window-coupling is not only real (row #293), it is *relied upon* by an existing,
bit-exact-tested harness path.

## 2. The chosen mechanism (minimal, two files, both physics-trigger)

**A separate, explicitly-named admissibility ceiling on the scenario object, consumed by
the guard only; the host-window call site is not touched at all.**

- New attribute `LamCDMScenario.h_grid_admissibility_max: float = 1.00`, declared
  **directly beside** the `h` parameter in `cosmological_model.py`, with a comment block
  naming both roles — the side-by-side placement in one file is the strongest guard
  against ever confusing the two bounds again.
- The guard at `bayesian_statistics.py:4655-4659` compares against
  `max(h.upper_limit, h_grid_admissibility_max)` (with a `getattr` fallback to
  `h.upper_limit` for duck-typed scenarios). The `max()` keeps the mirror's runtime
  widening of `h.upper_limit` effective through the guard unchanged, and makes
  `h_grid_admissibility_max = h.upper_limit` an exact no-op (limiting case, §6).
- `bayesian_statistics.py:5716` is **byte-for-byte unchanged**: `get_redshift_outer_bounds`
  continues to receive exactly `h.upper_limit` = 0.86.

**Where the value lives — weighed:**
(i) `constants.py` (new `H_EVAL_ADMISSIBILITY_MAX`): rejected — `constants.py` already
carries `H_MIN/H_MAX = 60.0/86.0` in a ×100 unit convention, `H_MAX` is already dead, and
a third h-ceiling in a second file and a second unit convention is exactly the
role-confusion this design exists to remove.
(ii) A new field on `CosmologicalParameter`: rejected — the dataclass is shared with
`Omega_m` (and `parameter_space.Parameter`); an admissibility field on every parameter is
surface area with one consumer.
(iii) **Chosen**: a plain scenario attribute in `cosmological_model.py`, next to the `h`
construction, value 1.00 = the top of the ratified G-EXT wing (row #284) and nothing more
— it admits precisely the already-authorized grid; any wider grid needs a fresh gate.
Both candidate files are physics-trigger files, so the gate burden is identical either
way; the choice is purely about legibility.

**Naming**: `h_grid_admissibility_max` — "grid admissibility" (which h-nodes evaluate()
will answer for) vs `h.upper_limit` (the physical prior-support bound that shapes the
candidate-host window). The comment at each of the two consumers cross-references the
other.

## 3. OLD code (exact, as in the working tree today)

`darksiren_emri/cosmological_model.py:386-393`:
```python
        self.h = CosmologicalParameter(
            symbol="h",
            upper_limit=0.86,
            lower_limit=0.6,
            unit="s*Mpc/km",
            randomize_by_distribution=uniform,
            fiducial_value=0.73,
        )
```
`darksiren_emri/bayesian_inference/bayesian_statistics.py:4656-4660`:
```python
        for _h_check in _h_list:
            if (_h_check < self.cosmological_model.h.lower_limit) or (
                _h_check > self.cosmological_model.h.upper_limit
            ):
                raise ValueError("Hubble constant out of bounds.")
```

## 4. NEW code (exact minimal diff — NOT APPLIED)

```diff
--- a/darksiren_emri/cosmological_model.py
+++ b/darksiren_emri/cosmological_model.py
@@ class LamCDMScenario:
     h: CosmologicalParameter
     Omega_m: CosmologicalParameter
+    # Grid-admissibility ceiling for evaluate()'s entry guard ONLY (G-EXT wing,
+    # AMENDMENT G-EXT row #284; decoupling ratified row #301 item 4(a)).  This is
+    # NOT the host-window bound: get_redshift_outer_bounds(h_max=...) reads
+    # h.upper_limit below, which stays 0.86 so every detection's candidate-host
+    # z-window is unchanged (byte-identity below 0.86 by construction; see
+    # graph1_20260901/exec/b-hprior-fix/DECOUPLING_DESIGN.md).
+    h_grid_admissibility_max: float
     w_0: float = -1.0
     w_a: float = 0.0

     def __init__(self) -> None:
         self.h = CosmologicalParameter(
             symbol="h",
+            # HOST-WINDOW / prior-support bound — deliberately NOT raised for the
+            # G-EXT wing (row #293: raising it widens every candidate-host z-window
+            # and breaks byte-identity below 0.86). The wing is admitted via
+            # h_grid_admissibility_max instead.
             upper_limit=0.86,
             lower_limit=0.6,
             unit="s*Mpc/km",
             randomize_by_distribution=uniform,
             fiducial_value=0.73,
         )
+        self.h_grid_admissibility_max = 1.00
```
```diff
--- a/darksiren_emri/bayesian_inference/bayesian_statistics.py
+++ b/darksiren_emri/bayesian_inference/bayesian_statistics.py
@@ def evaluate(...):
-        for _h_check in _h_list:
-            if (_h_check < self.cosmological_model.h.lower_limit) or (
-                _h_check > self.cosmological_model.h.upper_limit
-            ):
-                raise ValueError("Hubble constant out of bounds.")
+        # Admissibility guard ONLY (row #301 item 4(a) decoupling): the ceiling is
+        # max(host-window bound, grid-admissibility ceiling) so (i) the ratified
+        # G-EXT wing (h <= 1.00) is admissible, (ii) the mirror harness's runtime
+        # widening of h.upper_limit (correspondence_1d.py:3398-3399, [P3-HGRID])
+        # keeps working, and (iii) setting h_grid_admissibility_max ==
+        # h.upper_limit reproduces the old guard exactly. The host-window call
+        # site (get_redshift_outer_bounds(h_max=h.upper_limit), :5716) is
+        # deliberately NOT changed — see DECOUPLING_DESIGN.md.
+        _h_admissible_max = max(
+            self.cosmological_model.h.upper_limit,
+            getattr(
+                self.cosmological_model,
+                "h_grid_admissibility_max",
+                self.cosmological_model.h.upper_limit,
+            ),
+        )
+        for _h_check in _h_list:
+            if (_h_check < self.cosmological_model.h.lower_limit) or (
+                _h_check > _h_admissible_max
+            ):
+                raise ValueError("Hubble constant out of bounds.")
```
`[PHYSICS]` commit line (draft): `[PHYSICS] decouple h grid-admissibility (<=1.00, G-EXT
row #284) from the host-window bound (0.86 untouched) — rows #293/#301 item 4(a);
g-byte-id 0 mismatches below 0.86`.

## 5. Justification (role separation) — including the honest physics note

`h.upper_limit` was doing two jobs: (1) "which h may I ask about" (admissibility — no
physics content; 0.86 is where H_GRID_41 happened to stop) and (2) "how far in h do I
search for candidate hosts" (the ±3σ d_L → z window — physics-laden: it shapes L_cat for
EVERY event at EVERY h). The G-EXT ruling needs (1) raised and (2) frozen; the design
gives each its own name.

**Honest physics note (belongs in this presentation, per the ruling):** a wing-node
evaluation at h ∈ (0.86, 1.00] will use a candidate-host window computed for
h_max = 0.86, i.e. `z_max = dist_to_redshift(d_L + 3σ, 0.86)` — SMALLER than the z_max a
h_max = h window would give (dist_to_redshift is increasing in h at fixed d_L), so the
wing nodes' catalogue legs are potentially truncated on the high-z side: hosts in
(z_max(0.86), z_max(h)] are structurally excluded for those hypotheses. This is accepted
because (i) the wing is disclosed-irrelevant where it has ever been read — posterior tail
at h ≥ 0.85 is 5e-13 (row #286) — and its declared purpose is censoring-guard nodes
(matched-class rail, AMENDMENT G-EXT), not load-bearing posterior mass; (ii) freezing the
window is precisely what preserves the banked 41-node grid byte-for-byte and keeps ONE
window convention per run (the mirror comment at `correspondence_1d.py:3391-3396` documents
that per-run window consistency is required to reproduce L_cat bit-exact); (iii) the
alternative (windowing the wing at its own h) is the row #293 red path. **Any future
measurement that makes the wing load-bearing must revisit this truncation first** — that
is already the standing rule (row #290 row 11 NOT-covered cell: "any claim that the
extended grid is load-bearing for a given arm — decided at that arm's registration"); this
note gives that rule its physical content. Second-order note, same class: the per-h
selection loop (`bayesian_statistics.py:1247-1260`) computes z_max(h) per node and its
`min(z_max, z_max_cap=1.5)` clamp, a no-op for h ≤ 0.86 (z_max ≤ ~1.33), MAY bite at wing
nodes (z_max(1.00) can approach/exceed 1.5) — wing-only, disclosed, no effect on any
in-bound node.

## 6. Dimensional analysis and limiting cases

- **Dimensions**: h is dimensionless; both bounds are bounds on a dimensionless number; no
  formula, unit, or constant of physics is touched. The only computed change anywhere is
  the guard's comparison ceiling.
- **Limiting case 1 (the load-bearing one): byte-identity below 0.86 — by construction.**
  For any h-list wholly inside [0.6, 0.86]: the guard passes exactly as before (max(0.86,
  1.00) = 1.00 only relaxes; a pure gate feeds no computation), `:5716` is textually
  unchanged, and no other consumer of either bound exists (§1 map). Check plan carried
  over verbatim from `RECORD.md` §2.6: re-run the 41 banked H_GRID_41 nodes of job
  6747032 after the edit, diff posteriors + event_likelihoods byte-for-byte, **gate: 0
  mismatches** (including at least one multi-node batch evaluate() call, since the window
  is computed once per instance). Under this design that gate is expected green by
  construction; running it anyway is the evidence, not the hope.
- **Limiting case 2 (degenerate ceiling)**: `h_grid_admissibility_max = h.upper_limit`
  reproduces the old guard exactly (max(a, a) = a) — pinned in the tests.
- **Limiting case 3 (absent attribute)**: a scenario without the attribute (duck-typed,
  e.g. test doubles) falls back via `getattr` to `h.upper_limit` — old behavior exactly.

## 7. Regression tests (written with the edit; both bounds asserted)

In `darksiren_emri_test/` (new `test_h_bound_decoupling.py` or an addition beside the
existing `physical_relations_test.py:31` window test):
1. `LamCDMScenario().h.upper_limit == 0.86` — the host-window bound is FROZEN (this pin is
   the test that fails loudly if anyone re-couples by raising it).
2. `LamCDMScenario().h_grid_admissibility_max == 1.00`.
3. Guard admits the wing: an `evaluate()`-path bounds check (or the extracted guard logic)
   accepts h ∈ {0.87, 1.00} and still rejects 1.01 and 0.59.
4. Window unchanged: `get_redshift_outer_bounds(..., h_max=0.86)` output for a fixture
   detection equals its pre-change golden (and the `:5716` call site is exercised via a
   small canned-CRB evaluate() at h = 0.73 asserting byte-identical posterior output — the
   permanent form of the §6 check, per `RECORD.md` §2.7).
5. Degenerate ceiling: with `h_grid_admissibility_max` set equal to `h.upper_limit`,
   h = 0.87 is rejected (limiting case 2).
6. Mirror path: `correspondence_1d`-style widening of `h.upper_limit` past a value keeps
   admitting that value through the guard (the max() clause).

## 8. Rerun plan carry-over (NOT SUBMITTED; two blockers)

Unchanged from `RECORD.md` §3: scoped re-submission `sbatch --array=41-54
cluster/a18_ma1d_headreadout_iiib.sbatch` against the same RUN_DIR (banked 41 + fresh 14 =
55), seeds 777041-777054, expected **≈ 23.8 CPU-h** (14 × 1.7). **The 4b cap word is still
pending with the author** (row #301: the ≤ 20 CPU-h authorization vs the 23.8 estimate;
chair recommends raising the cap to 25). The rerun MUST NOT submit until BOTH: (i) this
design passes its `/physics-change` gate, lands, and the cluster checkout's HEAD carries
it (preflight match per `/cluster`); and (ii) 4b is granted. Post-run sanity read (not a
band): the 14 wing posteriors should confirm the disclosed-irrelevant framing (tail
≈ 5e-13), and the §5 truncation note is quoted next to them in the readout.

## 9. What this design does NOT do

Does not touch `get_redshift_outer_bounds` or its `h_max=0.86` signature default
(`physical_relations.py:549`); does not touch `constants.py` (`H_MAX = 86.0` stays as-is —
its deadness is noted for a housekeeping ticket, not acted on here); does not change the
mirror's widening semantics; does not make the wing load-bearing (§5); does not alter
`h.lower_limit` or the low wing. Gate scope: two physics-trigger files
(`cosmological_model.py`, `bayesian_statistics.py`), one `/physics-change` presentation —
this document — with a PHYSICS-GATE-LEDGER row appended when the gate actually runs.
