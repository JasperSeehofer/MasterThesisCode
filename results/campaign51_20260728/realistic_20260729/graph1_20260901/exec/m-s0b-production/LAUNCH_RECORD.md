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
