# m-s0b-production — launch node — STOP (node-set ambiguity, not launched)

Date: 2026-09-02. Launcher: m-s0b-production node, Research Graph 1 Branch D.

**Authorization chain, quoted.** Ledger row #290 decisions-table row 6: "m-s0b-production behind
g-byte-id and g-score-null green." Ledger row #319 (2026-09-02): "m-s0b-production's g-byte-id
precondition is met; g-score-null evaluates at run time; `m-s0b-production` launches under the
standing S0-B prereg (rows #278/#280) on the row #287-certified instrument, cap ≤20 CPU-h
ORCHESTRATOR-DERIVED (state candidate 2)." Discharge record:
`graph1_20260901/exec/m-s0b-byteid/DISCHARGE_RECORD.md` (714/715 files md5-identical, same-machine
reproduction — GREEN). **Both preconditions read as satisfied for launch.** This record does not
dispute that; the STOP below is a separate, newly-found gap in what the build actually shipped.

## What was checked

1. `PREREGISTRATION_HIER_HTHETA_20260826.md` — the standing S0-B design, §2.1/§4.1 and the
   `PA-HIER-31` addendum (2026-08-29, "S0-B registration addendum after Stage-0 wave-1"),
   append-only, never superseded on this point by any later `PA-HIER-3x` note.
2. Ledger rows #278/#280 (ratification), #287 (certified instrument config, verbatim: b0i,
   divisor on, zwin on zk4, sky 1.5, `catalogue_leg_1d_mass_aware=off`), #318/#319 (byte-id RED
   then discharge).
3. `exec/b-pahier33-scorer/RECORD.md` (the build that added the `iiib` production venue path +
   PA-HIER-33 scorer) and the driver it produced,
   `fanout1_20260829/hier_s0_driver.py`, read directly (not assumed from the record's prose).
4. `exec/m-s0b-byteid/LAUNCH_RECORD.md` + `cluster/graph1_m_s0b_byteid_precheck.sbatch` (the
   working driver-invocation pattern: single-quoted remote `sbatch` command, `--jobs 1`,
   `PROJECT_ROOT` left unset so the remote script's own `${PROJECT_ROOT:-$HOME/darksiren-emri}`
   fallback resolves — the GATE SEQ note and the double-quote lesson from its RESUBMIT section).

## The registered S0-B run shape (PA-HIER-31(d), quoted)

> Four θ-nodes at h = 0.730 on venue **iiib** (production, CoR-P), plus the truth node (θ=(0,1))
> which doubles as C0:
> ```
> truth        (0, 1)              = C0
> b_plus_re    (+0.033, 1)
> b_minus_re   (−0.033, 1)
> s_plus       (0, √2)
> s_minus      (0, 1/√2)
> ```

5 θ-nodes total, 1 h-value (0.73), single production catalogue (no seed multiplicity — CoR-P is
the one banked/observed scattered catalogue, not a mirror realization swept over seeds). Every
no-BH read runs CoR-P-faithful: `theta_sites="2.2"`, `smear_global_selection=False`
(`PA-HIER-31(b)`, re-ratified as authoritative by row #255, discharging `PA-HIER-31` REVISION
NOTE 2's R1′/R2′ open contradiction). `--jobs 1` per the byte-id precursor's own pattern.

**The b-node half-width is registered, explicitly and by name, as the RE-DERIVED ±0.033**
(`b_max=0.0661`, PA-HIER-29's 2×median convention), **not** the driver's as-built ±0.02. §2.1(a)
of `PA-HIER-31` is explicit that this is a deliberate, non-interchangeable choice: *"The two arms
(±0.02 as-built vs ±0.033 re-derived) are never combined into one secant, one Z, or one
materiality read; each is reported against its own registered grid."* The ±0.02 grid is reserved
for the mirror-venue S0-A remainder (P0) — never for S0-B.

## The gap found

`fanout1_20260829/hier_s0_driver.py`'s `THETA_NODES` dict (lines 179–186) is hardcoded:

```python
THETA_NODES: dict[str, tuple[float, float]] = {
    "truth": (0.0, 1.0),
    "b_plus": (0.02, 1.0),
    "b_minus": (-0.02, 1.0),
    "s_plus": (0.0, math.sqrt(2.0)),
    "s_minus": (0.0, 1.0 / math.sqrt(2.0)),
}
```

There is no `b_plus_re`/`b_minus_re` entry, and grepping the whole file for `0.033`, `0.0661`,
`b_half_width`, `b-half-width`, or any CLI flag that would let a caller override the b-node
spacing returns **nothing**. The only precedent for a caller-supplied grid addition is
`--s-half-step`, which registers exactly two hardcoded s-nodes (`s_plus_half`/`s_minus_half` at
`s = 2**(±1/4)`) into the module dict for that invocation — there is no analogous `--b-*` flag,
and `--s-half-step` does not touch the b-axis at all.

`b-pahier33-scorer/RECORD.md`'s own "Files changed" section lists exactly two files touched
(`hier_s0_driver.py` +385/-7, and a new test file) and its "What was built" section describes
only two additions: the PA-HIER-33 scorer and the `iiib` venue path (`build_iiib_venue`,
`CONFIG_CHOICES` extension, the `run_theta_node` `iiib` dispatch branch). **The b-node
re-derivation from `PA-HIER-31(a)`/(d) is not mentioned anywhere in that record — built, checked,
or deferred.** It is a silent gap, not a disclosed one.

Consequence: running the driver today at `--config iiib --nodes truth,b_plus,b_minus,s_plus,s_minus`
would score the **as-built ±0.02** b-secant, not the **registered ±0.033** b-secant — a different
statistic than PA-HIER-31(d)/(e) defines (different `score_b` denominator, different B0-B power
band: `σ_b < 0.0661` and the materiality threshold `|b̂| < 0.0165` are both stated in terms of the
0.033 step and do not transfer to a 0.02 step without re-derivation). This is exactly the class of
substitution the task brief instructs against: *"if the prereg's S0-B run shape is ambiguous on
any point — venue election, seed block, node set — STOP and report rather than choosing."*

## Why this is not resolved by row #287 / the byte-id discharge

Row #287 and the byte-id discharge certify the **b0i/S0-A default path** (mirror venue, as-built
±0.02 nodes) at current HEAD — they say nothing about whether the ±0.033 nodes were ever added,
because they were never asked to be. `g-byte-id`'s own scope (`b-pahier33-scorer/RECORD.md`,
"g-byte-id evidence" section) is explicit that it checks "all non-S0-B default paths" — the S0-B
node set itself was never exercised, tested, or byte-compared by any node in this chain.

## STOP

**No `sbatch` submitted. No job ID.** The registered S0-B run shape (5 θ-nodes: truth,
`b_plus_re` (+0.033), `b_minus_re` (−0.033), `s_plus`, `s_minus`) cannot be launched as registered
with the current driver — `THETA_NODES` has no ±0.033 b-node entries and no CLI mechanism to add
them. Options for the author, not chosen here:

1. **[DO]** authorize a small driver amendment (analogous to `--s-half-step`) adding
   `b_plus_re`/`b_minus_re` at ±0.033 as opt-in nodes, then re-run this launch node.
2. **[RULE]** rule that the as-built ±0.02 b-grid is acceptable for S0-B after all (a genuine
   scope change from `PA-HIER-31(a)`'s explicit "never combined" language — would need its own
   `PA-HIER` amendment, not a launcher's unilateral substitution).
3. Descope S0-B's first pass to the s-axis only (`s_plus`, `s_minus`, `truth`/C0 — 3 nodes, all
   of which the driver already supports byte-identically) and treat the b-axis re-derivation as a
   follow-on build item — a genuine scope narrowing, so also an author call, not a launcher call.

## Cost cap check (not reached — no run to cost)

Anchor requested: the local 1134.37 s/cell runner-11 8-cell precursor (`exec/m-s0b-byteid/
LAUNCH_RECORD.md`'s own table, cheapest cell 900103/b_minus). At 16 cpus/cell that is 0.2843
CPU-h/cell/node (matching the prereg's own §7.1 `cost_per_h_point_per_cell` derivation). For the
registered 5-node, 1-h, single-catalogue S0-B shape: `5 × 0.2843 ≈ 1.42` CPU-h at the mirror-venue
per-cell rate — but PA-HIER-31(d)'s own registered range for the *production* (iiib) venue is
materially higher (§7.2's own anchor: 14.93–22.9 CPU-h **per theta-node** on the full production
set, i.e. 74.7–101.4 CPU-h across 5 nodes, per §7 "S0-B (deferred)" row). That upper figure alone
exceeds the ≤20 CPU-h ORCHESTRATOR-DERIVED cap stated in this task's own authorization (row #319 /
`RESEARCH_GRAPH_1_PROPOSAL_20260901.md` line 422) by 3.7×–5×. **This is a second, independent
reason not to submit without author input**: the cap this task was given (≤20 CPU-h, "state
candidate 2," anchored to the runner-11 8-cell *mirror-venue* precursor) does not match the
registered *production*-venue cost anchor for the same node (74.7–101.4 CPU-h, §7.2). Which
anchor governs is itself a fresh question for the author, not decided here — flagged, not
resolved, alongside the node-set gap above.

## What was NOT done

- No `git pull` on the cluster clone was run (nothing to sync against, since nothing launches).
- No preflight was run (same reason).
- No out-root was created on the cluster.
- No `sbatch` script was written or submitted.
- No commit was made (per task instruction).

*Stamp: m-s0b-production, 2026-09-02. STOP — returned to the author/chair for the b-node
re-derivation gap and the cost-anchor discrepancy above.*

---

## LAUNCH 2 (row #325 grant)

Date: 2026-09-02. Launcher: m-s0b-production node (second attempt), Research Graph 1 Branch D.

**Authorization chain, quoted.** Row #290 decisions-table row 6 ("m-s0b-production behind
g-byte-id and g-score-null green"); row #319 (g-byte-id precondition discharged,
`m-s0b-byteid/DISCHARGE_RECORD.md`, 714/715 files md5-identical GREEN); rows #323-#324
(`DRIVER_BNODE_BUILD_RECORD.md` — `--b-half-width` flag lands the registered PA-HIER-31(d)
±0.033 `b_plus_re`/`b_minus_re` pair byte-identically by default, plus the `score_b_re`
follow-on implemented "gated on whether the prereg's own text defines its form" — verdict:
"the registered text DEFINES it, unambiguously"; 2041 tests passing, ruff/mypy clean); row
#325 ("chair-decided-under-grant item 10 option (A): cap 105 CPU-h, flagged for author veto").
**Both row #320 blockers are resolved** — this record does not re-litigate that; see below for
what remains.

### What was checked against LAUNCH 1's STOP

1. The node-set gap (LAUNCH 1's primary STOP): resolved by `DRIVER_BNODE_BUILD_RECORD.md` —
   `--b-half-width 0.033` registers `b_plus_re`/`b_minus_re` at the exact registered
   `(±0.033, 1)` pair; `score_b_re`'s form was independently confirmed against
   `PA-HIER-31(d)`'s quoted formula (`0.066` denominator, `PA-HIER-31(a)`'s pairing rule,
   `§2.1(e)`'s B0-B band) and implemented with 4 dedicated regression tests, including a
   never-folded-into-`score_b` structural guard.
2. The cost-anchor discrepancy: row #325 rules item 10 option (A) — the ≤20 CPU-h
   ORCHESTRATOR-DERIVED figure from row #319 is superseded; the governing cap is **105 CPU-h**,
   matching (with headroom) the registered §7.2/(i) production-venue anchor
   (5 nodes × 14.93–22.9 CPU-h ≈ 74.7–101.4 CPU-h). Taken here as the chair-ratified figure
   under the row #325 grant, not re-derived; row #325 itself flags it for author veto —
   unchanged by this launch node.

### The registered S0-B run, as configured for this launch

`--arm S0-A --config iiib --theta-sites 2.2 --smear off --seeds 900101 --nodes <one of
truth,b_plus_re,b_minus_re,s_plus,s_minus> --b-half-width 0.033 --jobs 1`, one SLURM array
task per node (`GATE TABLE-FRESH`, `PA-HIER-31(j)`: "one `BayesianStatistics` construction per
node — the four separate sbatch tasks structurally guarantee this").

- **`--arm S0-A`**: the driver's `--arm` choices are only `("S0-A", "S0-R", "S0-C")` — there is
  no `"S0-B"` arm. `--config iiib` is what selects the CoR-P production venue (`_build_venue`
  dispatch, `CONFIG_CHOICES = ("b0i", "ft", "iiib")`); `--config`'s own help text: "Applies to
  S0-A/S0-R only ... S0-B's precondition." This is not ambiguous — verified by reading the
  argparse definitions directly, not assumed.
- **`--seeds 900101`**: PA-HIER-31(d) is explicit that CoR-P is "the one banked/observed
  scattered catalogue, not a mirror realization swept over seeds." `build_iiib_venue`'s `seed`
  parameter is threaded through unused (loads the real catalogue, does not draw a realization).
  Passing `--seeds` explicitly is required — omitting it defaults to sweeping all 4
  `DEFAULT_BC_SEEDS` for S0-A (a 4x cost multiplication for a venue where the seed is
  structurally inert). `900101` = `DEFAULT_BC_SEEDS[0]`, the same single-seed convention the
  driver's own S0-C branch already uses for this reason.
- **T1.3 mirror instruments left at default (off).** Row #287's "divisor on, zwin on zk4, sky
  1.5" is a **mirror-venue (b0i)** T1.3 certification (`tree2_20260830/hier_s0_zwin_bnodes_run`
  runs, all `--config` unset/b0i). `PA-HIER-31(g)`'s own CLI list for the iiib/production venue
  (quoted from `headreadout_20260827/iiib/run_metadata_21.json:cli_args`) does **not** name
  `theta_phi_divisor`, `theta_zwindow`, or `z_window_k` at all. Following the task brief's
  instruction to follow the PREREG's S0-B registration verbatim rather than substitute: these
  stay at driver defaults (`off`/`off`/`1.0`), and `--sky-cone-k` stays at its default `1.5`
  (already byte-identical to the pre-flag literal, so no explicit flag is needed). This is a
  genuine cross-check finding, not assumed — grepped `PA-HIER-31` end to end for these four
  flag names; every hit is a mirror-venue (`tree2_20260830`) run, never the `(g)` production CLI
  list.
- **Node list / output naming**: `run_theta_node`'s `node_root = work_root /
  f"node_{node}{suffix}"` is generic on the node-name string (confirmed by reading, not
  assumed) — `b_plus_re`/`b_minus_re` produce `node_b_plus_re.../node_b_minus_re...`
  automatically, no additional wiring needed.

### Preflight / HEAD / submission

- Local branch `fix/p32d-classg-venue-repair` HEAD `9336364c`; `git fetch
  origin fix/p32d-classg-venue-repair` confirms **0 ahead / 0 behind** — origin is current, the
  driver build (rows #323-#324) and the row #325 docket addendum are both already committed
  (no uncommitted diff on `hier_s0_driver.py` or `darksiren_emri_test/`).
- **STOP: cluster SSH access unavailable in this session.** `ssh -o BatchMode=yes bwunicluster
  'hostname'` (three attempts, including a `-v` diagnostic pass) returns `Permission denied
  (publickey,keyboard-interactive)` for every key offered by the local agent
  (`gQnEF4Ks...`/`o+ij3+8o...`, both ED25519). This is an access/environment blocker in this
  particular session — not a scientific or spec ambiguity, and not something a launcher can
  resolve unilaterally (no key material to add, no interactive 2FA channel available
  non-interactively). **No preflight was run, no `git pull` was executed on the cluster clone,
  no out-root was created remotely, and no `sbatch` was submitted.**

### What WAS prepared and is ready to fire

`cluster/graph1_m_s0b_production.sbatch` (new file, uncommitted per task instruction) — a
5-element `--array=0-4` job, one array task per registered node
(`truth,b_plus_re,b_minus_re,s_plus,s_minus`), `--partition=cpu_il --cpus-per-task=16
--time=03:00:00` per task (sized against the §7.2 production-venue per-node anchor,
14.93–22.9 CPU-h at 16 cpus ⇒ ~56–86 min per task, with margin), pinned to the current HEAD
short-hash `9336364c`, sourcing `write_provenance.sh` for the required `run_metadata`-equivalent
stamp (gotcha 12), and adapted directly from `cluster/graph1_m_s0b_byteid_precheck.sbatch`'s
working invocation pattern (single-quoted remote submission, `--jobs 1`,
`PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"` fallback). Out-root:
`exec/m-s0b-production/s0b_run_20260902/`.

### Cost estimate (unchanged from the recompute above)

5 nodes × 14.93–22.9 CPU-h/theta-node ≈ **74.7–101.4 CPU-h**, within the row #325 cap of
**105 CPU-h** (headroom 3.6–30.3 CPU-h). Not re-derived from first principles here — taken as
the chair-ratified anchor under the row #325 autonomy grant, which itself flags the figure for
author veto (unchanged, not resolved, by this launch node).

### STOP

**No `sbatch` submitted. No job ID.** Every scientific/spec blocker from LAUNCH 1 is resolved
(node set, cost cap, venue election, CLI flags all verified against the registered text with
no substitutions). The sole remaining blocker is operational: this session's SSH client cannot
authenticate to `bwunicluster` (`st_ac147838@uc3.scc.kit.edu`, `Permission denied`). Next
attempt: re-run `cluster/graph1_m_s0b_production.sbatch` (already written, needs no further
edits beyond confirming HEAD still matches `9336364c` at submission time) from a session with
working cluster SSH access — preflight first (`ssh bwunicluster 'bash -s' <
cluster/preflight.sh`, require `VERDICT: READY ✓`), then `git pull --ff-only` on the remote
clone, then `sbatch cluster/graph1_m_s0b_production.sbatch`.

*Stamp: m-s0b-production LAUNCH 2, 2026-09-02. STOP — access blocker only (SSH auth to
bwunicluster unavailable this session); the registered run is fully specified and the sbatch
script is committed-ready at `cluster/graph1_m_s0b_production.sbatch`.*
