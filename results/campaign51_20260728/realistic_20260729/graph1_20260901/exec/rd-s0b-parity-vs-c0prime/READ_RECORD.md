# rd-s0b-parity-vs-c0prime — READ RECORD (verdict-free)

Node: `rd-s0b-parity-vs-c0prime` (Branch J, §1.1 item J.2, execution item 1).
Settles: `q-s0b-parity`. Grant: row #325 ("continue autonomous, decide but flag") — this record
contains numbers and a three-valued stamp against the proposal's own literal kill_criterion, plus
a RECOMMENDATION. No ruling, no re-certification, no caveat is attached to any dossier here; that
is the chair's action per decision A-J1.

---

## 0. Existence contract (three-valued, every input)

| input | proposal's path | status | evidence |
|---|---|---|---|
| S0-B truth node CSV | `graph1_20260901/retrieved/s0b_run_20260902/s0a_seed900101/node_truth_iiib_sites2.2_nosmear/simulations/diagnostics/event_likelihoods.csv` | **PRESENT** | 1588 data rows, 19 cols; retrieval confirmed md5 MATCH at row #334 item (4) ("S0-B run 6779532 retrieved ... 57 files, md5 MATCH") |
| comparand named by the proposal | `graph1_20260901/retrieved/run_20260902_graph1_c0prime_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv` | **PRESENT** | 1588 data rows, 19 cols; local md5 `228f12b0f086942fcfc80fbafdc1388f` matches the retrieval manifest `exec/wave1-retrieval/manifests/run_20260902_graph1_c0prime_headrebaseline_iiib.md5` exactly |
| driver's own gate comparand (`p3_b0_work/bc_900101_work/.../event_likelihoods.csv`) | cluster-absolute path quoted in `s0a_full_output.json:gate_parity` | **PRESENT locally** (not on the cluster path the driver checked) — see §3, not used for the diff (wrong venue, see §3) | `p3_b0_work/bc_900101_work/seed900101/simulations/...` exists in this repo |
| genuinely mass-aware-matched "c0prime_off" comparand (row #299/#301's actual subject) | `wave3_20260830/c0prime_off_iiib/simulations/diagnostics/event_likelihoods.csv` | **PRESENT**, found by opening `exec/m-head-rebaseline/c0prime_eval/GATE_RECORD.md` (not named in the proposal's Branch J text) | 1588 data rows, 19 cols |

No input was absent; nothing here is substituted. §3 explains why two *different* local files both plausibly answer to "c0prime_off," and why neither is flag-complete against S0-B.

---

## 1. CLI-flag audit (opened directly, not taken from the proposal's prose)

| flag | S0-B (`s0a_full_output.json`) | proposal-named comparand (`run_20260902_graph1_c0prime_headrebaseline_iiib/run_metadata_21.json`) | `wave3_20260830/c0prime_off_iiib/run_metadata_21.json` |
|---|---|---|---|
| `config` | `iiib` | `iiib` (CoR-P production venue — confirmed identical venue construction path, `hier_s0_driver.py` line 164, 483-484) | `iiib` |
| `h_value` | 0.73 | 0.73 | 0.73 |
| `theta_sites` | **`2.2`** | **`all`** | **`all`** |
| `catalogue_leg_1d_mass_aware` | `off` | **`auto`** | **`off`** |
| `theta_zwindow` | `off` | `off` | `off` |
| `theta_phi_divisor` | `off` | `off` | `off` |
| `sky_cone_k` | 1.5 | 1.5 | 1.5 |

**The proposal's own factual premise is wrong for the file it names.** §1.1/J.2 and the
`q-s0b-parity` kill_criterion both assert the named comparand is "mass-aware off, i.e. the same
flag state as S0-B." Opening `run_20260902_graph1_c0prime_headrebaseline_iiib/run_metadata_21.json`
directly shows `catalogue_leg_1d_mass_aware: "auto"`, not `"off"`. The file that is actually
mass-aware-matched (`off` exactly) is a *different* local file, `wave3_20260830/c0prime_off_iiib`
— the one `exec/m-head-rebaseline/c0prime_eval/GATE_RECORD.md`'s RE-STAMP section and ledger row
#299/#301 actually diffed to establish the "with-BH columns flip-invariant, max_abs=0" pattern the
proposal cites. The proposal appears to have conflated two distinctly-named local artifacts that
both contain the string "c0prime."

**Neither candidate matches S0-B on `theta_sites`.** Both locally available "c0prime"-named files
are `theta_sites="all"`; S0-B is registered at `theta_sites="2.2"` (PA-HIER-31(b), the
"CoR-P-faithful" form — confirmed in `s0a_full_output.json` and independently in ledger row #332:
"config stamp verbatim: `iiib`, `sites 2.2`, ... per the PA-HIER-31(g) CLI list"). Per
`hier_s0_driver.py` (module docstring + lines ~145-152), `theta_sites`/`smear` are explicitly a
**caller responsibility, orthogonal to venue construction** — set once for S0-B's registered
measurement and never varied in any locally-retrieved "iiib" production run. A CLI-driven
`theta_sites=2.2` + `config=iiib` + `h=0.73` production run does not exist anywhere on this
machine (checked: grepped every local `run_metadata_*.json` for `theta_sites=="2.2"` — zero
matches; theta_sites=2.2 only appears in `hier_s0_driver.py`-produced node dirs, i.e. b0i-venue
mirror runs, never iiib).

---

## 2. Diff — S0-B truth node vs the proposal-named comparand (`c0prime_headrebaseline_iiib`)

Matched on `event_idx`: **1588/1588** (0 S0-B-only, 0 comparand-only). 19 shared columns, all compared.

| column | max_abs | max_rel | n_nonzero / 1588 |
|---|---:|---:|---:|
| h, w_G, w_G_legacy, w_tilde_G, alpha_G_phi, r_Malm, D_tilde_phi, g_frac, den_log_term | 0.0 | 0.0 | 0 |
| B_num | 1.415610e-07 | 1.775069e-14 | 43 |
| B_num_wbh | 5.587935e-08 | 1.784888e-14 | 39 |
| L_comp | 1.040834e-16 | 5.360013e-15 | 11 |
| L_cat_no_bh | 4.845906e+00 | inf* | 1139 |
| L_cat_with_bh | 1.757943e-02 | inf* | 1083 |
| combined_no_bh | 1.209198e-02 | 1.594488e+00 | 1119 |
| combined_with_bh | 1.089342e-03 | 1.650558e+00 | 1042 |
| num_log_term_no_bh | 9.533894e-01 | 7.265724e-02 | 1114 |
| num_log_term_with_bh | 9.747701e-01 | 7.304050e-02 | 1044 |

\* `max_rel` is `inf` for `L_cat_*` on events where the comparand value is exactly 0 and the S0-B
value is not (denominator 0); reported as `inf` rather than silently dropped.

**no_bh channel max_abs = 4.845906e+00; with_bh channel max_abs = 9.747701e-01.** Both far from 0.

---

## 3. Diff — S0-B truth node vs the mass-aware-matched comparand (`wave3_20260830/c0prime_off_iiib`)

Matched on `event_idx`: **1588/1588** (0/0 unmatched). Same 19 columns.

| column | max_abs | max_rel | n_nonzero / 1588 |
|---|---:|---:|---:|
| h, w_G, w_G_legacy, w_tilde_G, alpha_G_phi, r_Malm, D_tilde_phi, g_frac, den_log_term | 0.0 | 0.0 | 0 |
| B_num | 1.415610e-07 | 1.775069e-14 | 43 |
| B_num_wbh | 5.587935e-08 | 1.784888e-14 | 39 |
| L_comp | 1.040834e-16 | 5.360013e-15 | 11 |
| L_cat_no_bh | 1.300574e-02 | inf* | 1086 |
| L_cat_with_bh | 1.757943e-02 | inf* | 1083 |
| combined_no_bh | 2.105473e-03 | 7.337622e-01 | 1065 |
| combined_with_bh | 1.089342e-03 | 1.650558e+00 | 1042 |
| num_log_term_no_bh | 5.502937e-01 | 3.714371e-02 | 1061 |
| num_log_term_with_bh | 9.747701e-01 | 7.304050e-02 | 1044 |

**no_bh channel max_abs = 5.502937e-01; with_bh channel max_abs = 9.747701e-01.** Both far from 0.

Note: `L_cat_with_bh`, `combined_with_bh`, `num_log_term_with_bh`, and every "precision-floor"
column (`B_num`, `B_num_wbh`, `L_comp`) are **byte-identical between the two comparand files
themselves** — the two candidate comparands differ from each other only through
`catalogue_leg_1d_mass_aware` (`auto` vs `off`), which by the established row #299 pattern moves
only the no_bh channel. The with-BH channel numbers above (max_abs ≈ 0.975, nonzero on
1042-1044/1588 events either way) are therefore **not explained by the mass_aware axis at all** —
they are identical against both candidates, consistent with being driven by the one flag both
candidates share and S0-B does not: `theta_sites` (`all` vs `2.2`).

---

## 4. Why the driver's own `gate_parity` said `NO_BANKED_CSV` (opened, not inferred from the tag)

`hier_s0_driver.py:gate_parity()` (lines 1994–2049) is **hardcoded** to
`bc_work_root / f"bc_{seed}_work" / f"seed{seed}" / "simulations/diagnostics/event_likelihoods.csv"`
for every `--config` value — the function's own docstring: "Compares this driver's truth-node
`combined_no_bh`/`combined_with_bh` values against the banked `bc_<seed>_work/...` ... theta=(0,1)
reproduces the banked **bc** seed's h=0.73 row." This is the `bc`/`b0i` venue's own banked
comparand (`host_mode="catalogue_selected"`, `catalogue_numerator_survival="off"` — module
docstring lines 42-49), **not** the `iiib` venue S0-B actually ran (`host_mode` default,
`catalogue_numerator_survival` hardcoded `"phi"` for iiib, lines 148-149). The `p3_b0_work/
bc_900101_work/seed900101/simulations/diagnostics/event_likelihoods.csv` file the driver looked
for **does exist locally** (`p3_b0_work/bc_900101_work/seed900101/simulations/...`, confirmed
present) — the cluster run reported `NO_BANKED_CSV` because that path was never present on the
cluster workspace, not because the file is missing everywhere. But even if it had been reachable
on the cluster, diffing an `iiib`-venue truth node against a `bc`-venue banked CSV would compare
two structurally different venues (different `host_mode`, different
`catalogue_numerator_survival`) — this gate is scoped to `b0i` runs only; it was never going to
produce a meaningful parity number for an `iiib` node. This matches, and gives the mechanism
behind, the proposal's own observation that the gate "looked for a b0-venue path."

---

## 5. Three-valued stamp

Per the proposal's literal kill_criterion (§1.0 `q-s0b-parity`): *"with-BH columns... max_abs = 0
AND no-BH columns max_abs = 0 against c0prime_off -> GREEN; any nonzero -> STOP."*

**STOP** — against both candidate comparands, on both channels:

- vs proposal-named `c0prime_headrebaseline_iiib`: no_bh max_abs 4.845906, with_bh max_abs 0.974770 — nonzero.
- vs mass-aware-matched `wave3_20260830/c0prime_off_iiib`: no_bh max_abs 0.550294, with_bh max_abs 0.974770 — nonzero.

STOP is returned **exactly as the proposal's own binary criterion requires** — this is not a
NOT-EVALUABLE call on the diff itself (both inputs were PRESENT and the diff is well-defined and
reported above). What is flagged for the chair (§6) is that the STOP's *interpretation* — "the
S0-B run is not on the production comparand" — is not supported by the evidence in hand: the
observed nonzero deltas track a flag (`theta_sites`) that (a) both proposal candidates fail to
match S0-B on, (b) no local file matches S0-B on at `config=iiib`, and (c) the driver's own source
documents as a deliberate, orthogonal, caller-set choice for the registered [HIER] measurement,
not a venue-construction parameter this gate was ever built to check.

---

## 6. RECOMMENDATION (not a ruling)

1. The `q-s0b-parity` kill_criterion as written cannot be discharged GREEN or interpreted as a
   genuine STOP-on-S0-B-correctness with any comparand currently on this machine, because no
   `config=iiib`, `theta_sites=2.2`, `h=0.73` production baseline exists locally to diff against.
   Recommend the chair treat this specific sub-question as **NOT-EVALUABLE-pending-comparand**
   rather than folding it into d-photoz-leverage as a clean GREEN/STOP stamp.
2. If a `theta_sites=2.2` / `config=iiib` comparand is wanted, the cheapest local path is a
   zero-compute check against the *already-banked* `p3_b0_work/bc_900101_work` family is **not**
   it (wrong venue, §4) — a fresh comparand would need either (a) a driver run at `--config iiib
   --theta-sites 2.2` with a second, independent-of-S0-B invocation to diff against (compute, not
   zero-cost), or (b) an author ruling that `theta_sites` is out of scope for this parity gate by
   design, in which case the gate should be re-specified to diff only the columns/rows the
   `theta_sites` window leaves untouched (candidate host galaxies inside window 2.2 at both ends).
3. The precision-floor columns (`B_num`, `B_num_wbh`, `L_comp`) show tiny nonzero max_abs
   (1e-7–1e-16) on a handful of events (11–43/1588) against **both** candidates identically —
   worth a note for whoever revisits g-precision on this pair, but three orders of magnitude below
   anything that could explain the STOP above.
4. Recommend the proposal's Appendix B / §1.1 text be corrected: the comparand it names
   (`run_20260902_graph1_c0prime_headrebaseline_iiib`) is not "c0prime_off" and is not
   mass-aware-matched to S0-B; the actually mass-aware-matched file is
   `wave3_20260830/c0prime_off_iiib`. This does not change the STOP outcome (§3 above uses the
   corrected file and is still STOP) but the mislabeling should not propagate into the dossier.

---

## Sources (every number opened directly)

- `graph1_20260901/retrieved/s0b_run_20260902/s0a_full_output.json` (S0-B config, `gate_parity` status)
- `graph1_20260901/retrieved/s0b_run_20260902/s0a_seed900101/node_truth_iiib_sites2.2_nosmear/simulations/diagnostics/event_likelihoods.csv` (1588 rows)
- `graph1_20260901/retrieved/run_20260902_graph1_c0prime_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv` (1588 rows) + its `run_metadata_21.json`
- `graph1_20260901/exec/wave1-retrieval/manifests/run_20260902_graph1_c0prime_headrebaseline_iiib.md5` (md5 cross-check, matched)
- `wave3_20260830/c0prime_off_iiib/simulations/diagnostics/event_likelihoods.csv` (1588 rows) + its `run_metadata_21.json`
- `graph1_20260901/exec/m-head-rebaseline/c0prime_eval/GATE_RECORD.md` (identifies `c0prime_off` as the flag-matched comparand; RE-STAMP section, row #299/#301 basis)
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py` (`gate_parity()` lines 1994-2049; `CONFIG_CHOICES`/venue docstrings lines 42-49, 135-168, 475-484)
- `gate_b_20260730/BIAS_HISTORY_LEDGER.md` rows #299, #301, #325 item (3), #332, #334 item (4)
- `p3_b0_work/bc_900101_work/seed900101/simulations/` (existence check only, §4)

Verdict-free per row #325 grant. No caveat attached to any wave-3 dossier by this record; that
action is the chair's (decision A-J1).
