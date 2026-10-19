# T2.2b (arm (b)) RUNSHEET — read-only extraction, 2026-08-30

Produced in response to the docket's §5 item 0 pointer (`TREE2_SYNTHESIS_DOCKET_20260830.md`),
which names T2.2b as "arm (b), ~4-5 CPU-h local... the 17.1 sequencing gate for item 2" and lists
it as a LOCAL PREREQUISITE runnable during the cluster outage. This node did no code, no git, and
started no run (per its own instruction); everything below is read from disk and cross-checked
against the working tree. **Headline finding: as literally registered, arm (b) is NOT runnable
locally today — see §5.** No git, no code, no run started by this node.

---

## 1. What arm (b) measures, and its registered bands

Arm (b) = T2.2b: the T2.2 per-candidate diagnostic hook (`--candidate_dump_dir`, guarded,
byte-identical when omitted; `B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md` §6) run **on the
production `iiib` venue** (not the FT-mirror T2.2 already ran on) **at the 3 secant h-nodes
{0.725, 0.730, 0.735}, under `catalogue_leg_1d_mass_aware` "on" AND "off"** — 6 h-points total.
It is prediction (ii) of `PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md` §6.2, registered arm text
at §6.2 and §9 item 3:

> "**Arm.** T2.2b (derivation section 6.4): the hook on iiib at the 3 nodes with the flag 'on' and
> 'off' — 6 h-points, 5-7 min each (charter anchor), about 4-5 CPU-h local; no cluster needed."
> — §6.2

> "**Arm (b), prediction (ii):** T2.2b on iiib at the 3 secant nodes, 'on' and 'off': 6 h-points x
> 5-7 min wall ≈ **4-5 CPU-h local** (derivation section 6.4 anchor). No cluster." — §9 item 3

**Statistic** (§6.2): per-event secant score at truth over h=0.725/0.735 on the 1588 iiib events,
by class (dark = `host_galaxy_index == -1`; in-catalogue = 76 events), "on" vs the banked "off"
(`headreadout_20260827/iiib`, row #213); and the per-event ratio q_i = s_imp,i(on)/s_imp,i(off) on
active dark events. **"The class split must use the T2.2b per-event CSV's `host_galaxy_index` (a
VALIDATED join), not the CRB row-order assumption-join of C5."**

**Registered bands (§6.2, quoted verbatim, with the current supersession state from REVISION NOTE
2026-08-30b):**
- dark-class impostor score: −0.1926 → **−0.074**, band **[−0.097, −0.048]** (ARITH, current, NOT
  superseded — scales by the anchored rho = 0.383).
- in-catalogue: −1.707 → about −1.54, band [−1.7, −1.4] — **[SUPERSEDED 2026-08-30b —
  REPORTED-ONLY/UNSUBSTANTIATED]**: the −130→−117 nats input has "no shown derivation anywhere...
  only the qualitative statement 'the true host's own term does not scale by rho'" (§15.1). Zero
  true-host rows exist in the only per-candidate dump on disk today (the FT-mirror T2.2 dump, 4
  seeds, 606,571 rows, 0 True — structural, because the FT-mirror arm is dark-only by design).
- pooled: −0.265 → −0.144, band [−0.166, −0.120] — **[SUPERSEDED, inherits the in-catalogue
  number]**.
- median q_i on active q1 dark events: predicted 0.38, band [0.25, 0.5]; **q_i > 1 on more than 10
  percent of active dark events is REFUTING** (F-3, unaffected by the supersession for its
  dark-class half).

**This is exactly the gap arm (b) exists to close.** §15.1's own words: "What would fix this going
forward... (a) run T2.2b on production iiib... and read the true in-catalogue rows'
S_4D(z_true,M_true)/S_bar_phi(z_true) ratio directly — a genuine zero-compute read once that dump
exists."

---

## 2. The exact command(s) — AS REGISTERED (currently BLOCKED, see §5)

**Not a zero-compute read.** Unlike the section 6.6 rescore of the *existing* T2.2 (FT-mirror)
dump, T2.2b requires a fresh `evaluate()` pass over the production iiib venue with the hook
engaged — it does not run against any banked production CSV at zero cost (the banked
`headreadout_20260827/iiib` run was never instrumented with `--candidate_dump_dir`; its
per-candidate rows do not exist).

**No driver support exists for "iiib."** The standing driver used for T2.2
(`results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py`), which *did*
receive the `--catalogue-leg-1d-mass-aware` CLI-flag threading in the mass-aware build
(§20.1 item 4 of the gate doc), only builds **synthetic mirror venues**:
`CONFIG_CHOICES: tuple[str, ...] = ("b0i", "ft")` (driver line 137) — `build_bc_venue` /
`build_ft_venue` call `c1d.MirrorUniverseGenerator.draw_realization(...)`, an in-process synthetic
draw that needs no external data. **There is no `"iiib"` choice.** T2.2b must therefore invoke the
production pipeline **directly** via `python -m darksiren_emri`, replicating the `iiib` venue's own
CLI configuration (below), not via `hier_s0_driver.py --config iiib` (that flag value does not
exist and would raise `ValueError`).

The `iiib` venue's exact CLI configuration is read from the banked
`headreadout_20260827/iiib/run_metadata_21.json` (git commit `d04d9dc9`, tree `7bfff25d` —
byte-identical to local HEAD's `darksiren_emri/` per `MEASUREMENT_HEAD_READOUT_20260827.md` §7.3).
Two invocations are needed (flag "off" then "on"), each covering all 3 secant nodes in one process
via `--h_values` (comma-separated, supersedes `--h_value`):

```bash
# ---- shared setup ----
REPO=/home/jasper/Repositories/darksiren-emri
OUT=$REPO/results/campaign51_20260728/realistic_20260729/tree2_20260830/t2_2b_arm_b_run
mkdir -p "$OUT/off/simulations" "$OUT/on/simulations"

# Registered CRB/event-set input pin (STOP-gate, CLAUDE.md dataset-pinning rule +
# MEASUREMENT_HEAD_READOUT_20260827.md §8 STEP 0 form) — verify BEFORE linking:
md5sum "$REPO/results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv"
#   expect 9a1f2a14384a9281c97ca3be312ddaab   -- STOP on mismatch
md5sum "$REPO/darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv"
#   expect c52c13b5cab61f6b3f04bbe202550969   -- STOP on mismatch (REDUCED_CATALOGUE_MD5, A11 pin)

ln -sfn "$REPO/results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv" \
  "$OUT/off/simulations/prepared_cramer_rao_bounds.csv"
ln -sfn "$REPO/results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv" \
  "$OUT/on/simulations/prepared_cramer_rao_bounds.csv"
# *** simulations/injections/ must also exist here -- see §5: no local copy found. ***

# ---- off arm (baseline, candidate dump ON, mass-aware flag OFF) ----
cd "$OUT/off" && uv run python -m darksiren_emri "$OUT/off" \
  --evaluate --h_values 0.725,0.73,0.735 --seed 777021 --strategy physics-floor \
  --num_workers 6 \
  --pdet_dl_bins 60 --pdet_mass_bins 40 --pdet_estimator local_linear \
  --pdet_z_resolved --host_z_kernel volume_deconv --host_mass_kernel auto \
  --normalization_mode absolute_marginal \
  --selection_in_completion_numerator fused --catalogue_mass_overlap production \
  --catalogue_mass_error_scale 1.0 --completion_b_scale derived --eddington_m on \
  --sigma4d_mass_kernel point --completion_event_measure ratio \
  --catalogue_numerator_survival_2d off --catalogue_global_selection phi \
  --catalogue_leg_1d_mass_aware off \
  --candidate_dump_dir "$OUT/off/candidate_dump"

# ---- on arm (candidate dump ON, mass-aware flag ON) ----
cd "$OUT/on" && uv run python -m darksiren_emri "$OUT/on" \
  --evaluate --h_values 0.725,0.73,0.735 --seed 777021 --strategy physics-floor \
  --num_workers 6 \
  --pdet_dl_bins 60 --pdet_mass_bins 40 --pdet_estimator local_linear \
  --pdet_z_resolved --host_z_kernel volume_deconv --host_mass_kernel auto \
  --normalization_mode absolute_marginal \
  --selection_in_completion_numerator fused --catalogue_mass_overlap production \
  --catalogue_mass_error_scale 1.0 --completion_b_scale derived --eddington_m on \
  --sigma4d_mass_kernel point --completion_event_measure ratio \
  --catalogue_numerator_survival_2d off --catalogue_global_selection phi \
  --catalogue_leg_1d_mass_aware on \
  --candidate_dump_dir "$OUT/on/candidate_dump"
```

Notes on this reconstruction:
- `--catalogue_numerator_survival` (the 1D no-BH numerator survival choice) has **no CLI flag** —
  it resolves internally to `"phi"` under `normalization_mode=absolute_marginal` (the
  bac48696/row #197 "auto"→"phi" pattern); it is not settable and not present in the banked
  `run_metadata`'s `cli_args`, consistent with this.
- `theta_phi_divisor` (must be `"off"` for the mass-aware guard) is likewise absent from the
  banked `iiib` run_metadata (that run predates the T1 theta-fix work); its default is `"off"`,
  satisfying the guard without an explicit flag.
- `--num_workers 6` stands in for the task's "`--total-cpu-budget 6`" instruction: `main.py`'s
  `evaluate()` has no literal `--total-cpu-budget` flag (that is a `hier_s0_driver.py`-specific
  concept for splitting seeds across concurrent jobs); `--num_workers` is the actual CPU-budget
  knob on the direct pipeline, so this substitutes it 1:1 against the stated "runner-9 holds 8
  workers" headroom rule. There is no `--jobs` flag on `main.py` either (that too is
  `hier_s0_driver.py`-specific) — running the two invocations sequentially (as written above) is
  the `--jobs 1` analogue.
- Output layout matches `constants.py`'s relative-path convention
  (`PREPARED_CRAMER_RAO_BOUNDS_PATH = "simulations/prepared_cramer_rao_bounds.csv"`,
  `INJECTION_DATA_DIR = "simulations/injections"`, both resolved against the process's CWD, not
  the `working_directory` CLI arg) — hence the `cd "$OUT/{off,on}"` before each invocation, mirroring
  the cluster recipe's own `mkdir -p "$RD/simulations"` + symlink pattern
  (`MEASUREMENT_HEAD_READOUT_20260827.md` §8 STEP 1).
- Registered cost: **4-5 CPU-h local, no cluster** (§6.4, §9 item 3) — true only of the *compute*;
  see §5 for why the *run* cannot start today regardless.

---

## 3. Gates to evaluate first (before trusting any number out of this run)

In order, per `B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md` §6.3 and
`PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md` §6.2/§8:

0. **Dataset pins (STOP-gated, CLAUDE.md pinning rule):** CRB/event-set md5
   `9a1f2a14384a9281c97ca3be312ddaab` and reduced-catalogue md5 `c52c13b5cab61f6b3f04bbe202550969`
   — both **before** linking any file into the run directory (§2 above already folds this in).
1. **GATE BI (byte-identity):** the "off" arm's `event_likelihoods.csv` (all columns) and both
   posterior JSONs must be bit-identical to the un-hooked `headreadout_20260827/iiib` banked run at
   the same h-nodes — proves the hook changed nothing. (This is the production-data analogue of the
   FT-mirror GATE T-ID that T2.2's own readout already confirmed on the mirror; here it is the
   decisive check because `iiib` is the real comparand everything downstream cites.)
2. **GATE R (reconstruction):** per event and h, `sum_g w_g * N_g_used / Sigma_phi(h)` must
   reproduce the diagnostics column `L_cat_no_bh` to <= 1e-12 relative, and the serialised candidate
   count must equal the "possible hosts found" log line. **If R fails the run is
   INSTRUMENT-DEFECT and nothing downstream is read.**
3. **GATE SCHEMA:** both dump files exist, columns match §6.2 of the derivation exactly, one event
   row per event that reached `p_Di`, `z_g`/`N_g_used`/`D_g`/`h` finite on every candidate row.
4. **GATE ENG (A13):** the per-candidate `N_g` must differ across the three h-nodes on >= 99 percent
   of rows (the "on" vs "off" comparison is only informative once this holds).
5. Only after 0-4 PASS: read the primary statistic (§6.5's `Phi_low`/`<u>_W`/candidate-count reads
   do not apply here — that is T2.2's own FT-mirror statistic; T2.2b's own read is the §6.2
   per-class impostor score / q_i ratio and, decisively, the **derived in-catalogue transform**
   S_4D(z_true,M_true;h)/S_bar_phi(z_true;h) computed directly from the 76 true in-catalogue hosts'
   dumped rows (`is_true_host == True`) — this is the number §15.1/§17.1 need.

---

## 4. What the result feeds

- **§17.1's hard sequencing rule (`PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md` lines 817-835,
  quoted):** "Arm (c)... is **BLOCKED** — it MUST NOT be submitted, and if it were ever run its
  outcome MUST NOT be read against those bands — until arm (b)... has been executed AND has
  produced a derived, ARITH in-catalogue transform S_4D(z_true,M_true;h)/S_bar_phi(z_true;h) for the
  76 true in-catalogue production hosts, directly superseding section 15.1's
  REPORTED-ONLY/UNSUBSTANTIATED placeholder." Arm (b)'s output is a **hard precondition**, not a
  recommendation — "the queue behind the OST 5 recovery is a NECESSARY but not SUFFICIENT
  precondition; arm (b)'s derived transform is a second, independent gate."
- **Docket §5 item 2 (A18 PRODUCTION ARM):** "SUBMIT ONLY AFTER T2.2b's derived transform exists
  (17.1 hard STOP...)" — the same gate, restated at the docket level for the 41-node cluster
  submission.
- Once derived, the transform supersedes the REPORTED-ONLY marks throughout
  `PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md` §6.2 (in-catalogue/pooled predictions), §6.3 (MAP
  band [0.64, 0.72] and dark-only-full band [0.60, 0.67] — note §17.3 already independently
  reconfirmed [0.60, 0.67] does NOT depend on the disputed input, so only the MAP band and the F-3
  in-catalogue half remain gated), and F-3's in-catalogue falsifier band.

---

## 5. Arm (b) is NOT actually runnable locally today — what the docket should have said

The docket's item 0 lists T2.2b as a "LOCAL PREREQUISITE, runs before or during the outage" on the
strength of its **compute cost** (4-5 CPU-h, "no cluster needed" per the gate doc's own §6.2/§9
wording). That wording is about compute only, and conflates it with **data availability**, which is
a separate axis. Two independent, verified blockers exist:

**(a) No driver path exists.** `hier_s0_driver.py`'s `CONFIG_CHOICES = ("b0i", "ft")` (line 137) has
no `"iiib"` entry; `_build_venue` (line ~295-303) raises `ValueError(f"config must be one of
{CONFIG_CHOICES}, got {config!r}")` for anything else. The mass-aware CLI flag *was* threaded through
this driver's own functions (gate doc §20.1 item 4), but that only means the driver can toggle the
flag on its two *synthetic* venues — it has no code path to the real GLADE-catalogue production
pipeline at all. Running T2.2b therefore requires a **direct** `python -m darksiren_emri` invocation
(§2 above), not the T2.2 template command verbatim with `--config` swapped.

**(b) The `iiib` venue's injection pool — required to build `SimulationDetectionProbability`, and
through it every `S_4D`/`S_bar_phi` value the whole flag operates on — has no local copy.**
Verified this session:
- `INJECTION_DATA_DIR = "simulations/injections"` (`darksiren_emri/constants.py:152`) is a bare
  relative path, glob'd directly (`simulation_detection_probability.py:298-299`,
  `f"{injection_data_dir}/injection_h_*_task_*.csv"`) with **no fallback search path** — an absent
  or empty directory raises `FileNotFoundError` immediately (`simulation_detection_probability.py`
  line ~313).
- The `iiib` venue's own registered recipe (`MEASUREMENT_HEAD_READOUT_20260827.md` §8 STEP 1) builds
  this by symlinking `$WS/run_20260729_seed61000/simulations/injections` — a **cluster-only**
  workspace path (`$WS = /pfs/work9/workspace/scratch/st_ac147838-emri`) — into each run directory.
  Every locally-checked-in `*_iiib`/`*_joint_r1` directory in this repo
  (`results/run_20260817_fusion_counterfactual/{off,fused}_iiib/simulations/`,
  `results/_archive/run_20260805_d1_a1_iiib/simulations/`, etc.) reproduces this **as a broken
  symlink** to that same cluster path — none of them carry real injection files locally.
  `find`-ing the whole repo for `injection_h_0p73_task*.csv` real files turns up only two unrelated,
  older campaigns (`run_20260620_seed500_phase50/injections/`,
  `lcat_h_dependence_20260725/data/injections/`) — different seeds, not interchangeable with
  `run_20260729_seed61000` without violating the dataset-pinning rule. The project-root fallback
  location `main.py` mentions for a different code path (`<repo>/simulations/injections`) exists but
  is **empty**.
- By contrast, the *other* half of the `iiib` input — `prepared_cramer_rao_bounds.csv` — **is**
  present locally, byte-identical to the registered pin: `md5sum` of
  `results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv` =
  `9a1f2a14384a9281c97ca3be312ddaab`, matching `MEASUREMENT_HEAD_READOUT_20260827.md`:439's
  registered expectation exactly. This is a real, non-symlinked 4.2 MB file (banked 2026-08-03), so
  the CRB/event-set half of the dataset pin is satisfiable purely locally — only the injection pool
  is missing.
- `hier_s0_driver.py`'s own synthetic venues (`build_bc_venue`/`build_ft_venue`, driver lines
  201-270) sidestep this entirely: they call `c1d.MirrorUniverseGenerator.draw_realization(...)`, an
  in-process synthetic draw with its own banked-and-local selection objects
  (`c1d.build_bsel_selection_objects`) — this is *why* T2.2's FT-mirror run had "no dataset pin
  needed" (`T2_2_CANDIDATE_HOOK_RECORD.md`:161-163). That property does not transfer to `iiib`,
  which is a real-data venue with no synthetic analogue.

**Verdict: arm (b) is BLOCKED on the same class of dependency as arm (c) — cluster-resident data —
just with a far smaller compute footprint once the data is in hand.** It is not "runs during the
outage"; it is "cheap once the outage ends, or once someone `rsync`s
`$WS/run_20260729_seed61000/simulations/injections` down before then." No amount of local CPU or
`--jobs`/`--total-cpu-budget` tuning changes this — the process will exit with `FileNotFoundError`
before doing any compute.

**What item 0 should have said**, to be accurate: "T2.2b (arm (b), ~4-5 CPU-h once its input data is
available) — **requires** the `run_20260729_seed61000` injection pool (`$WS`-resident; not banked
locally); either stage it down during a live SSH window before the outage closes it off, or queue
T2.2b behind the same OST 5 recovery as arm (c), just first in the queue once the cluster returns
(its own dependency, not "before or during the outage" independent of it). It is the 17.1 sequencing
gate for item 2 regardless of when it runs — item 2 (A18) still cannot submit until T2.2b's derived
transform exists." The registered ~4-5 CPU-h figure is a compute-time estimate, not a
runnability claim, and the docket's phrasing collapsed the two.

---

## 6. Summary for the orchestrator

- **Runsheet path:** `/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729/tree2_20260830/T2_2B_ARM_B_RUNSHEET.md` (this file).
- **Command:** given in full in §2 above (two `python -m darksiren_emri` invocations, "off" then
  "on", `--h_values 0.725,0.73,0.735`, `--candidate_dump_dir`, `--catalogue_leg_1d_mass_aware
  {off,on}`, `--num_workers 6`, out-root
  `results/campaign51_20260728/realistic_20260729/tree2_20260830/t2_2b_arm_b_run`).
- **Do not run it yet.** `simulations/injections/` for the `iiib` venue is not present locally; the
  command as written will raise `FileNotFoundError` in `SimulationDetectionProbability.__init__`
  before any CPU-hour is spent. Fix requires either an SSH window to the cluster (currently down,
  none this session) or the injections pool being staged locally by some other means.
  `prepared_cramer_rao_bounds.csv` (the other required input) IS present and pin-verified locally —
  only the injection pool blocks this.
