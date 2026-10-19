# B4.1 [IMP] part 1 — per-event impostor covariate table (zero-compute, banked artifacts)

*launched under rows #222/#223 — charter node B4.1 [IMP] part 1*
*Append-only. Builder role only (rule 2): this is the instrument + the best decomposition
buildable from banked columns, not the registered decisive measurement. Any future
"is the impostor drag attributable to covariate X" claim must be run by a DIFFERENT agent
against a pre-registered band, not read off this document.*

## 0. Verdict (one line)

**UNDETERMINED, with a hard structural finding**: the requested per-candidate impostor
covariates (impostor z, impostor σ_z, impostor mass, per-candidate catalogue share c_i, and a
per-event ln L_cat − ln L_true-host score) **do not exist in any banked artifact** — the mirror
harness never serializes per-candidate data, only per-event AGGREGATES over the whole candidate
ball. This is not a gap in this pass's search; it is independently corroborated by
`CLAIM_P3_IMPOSTOR_CONVENTION_20260822.md`'s own 2026-08-22 stage-1 inventory (quoted in §3
below), reached by a different agent on a different day. What CAN be built at zero compute is
reported in §4–§6, and it is informative on its own terms (the 1D vs 2D recovery-rate gap, and
the dominant covariate of the aggregate catalogue-leg magnitude) — but it is a proxy, not the
impostor-isolated decomposition the node asked for. §7 gives the exact instrumented re-run that
would close the gap and its cost (~3.4 CPU-h, one seed-fleet pass).

## 1. Scope and inputs

- Fleet: the 12-seed **b0 identity test** B-C / B-T arms (`bc_9001NN` / `bt_9001NN`,
  `NN=01..12`), `results/campaign51_20260728/realistic_20260729/p3_b0_work/`. `bc` =
  `catalogue_numerator_survival="off"` (coded convention), `bt` = `"phi"` (the [P3-IMP] twin
  convention) — confirmed byte-for-byte by diffing `bc_900101_meta.json` /
  `bt_900101_meta.json` (only that flag, the derived `mean_h`/`map_h`, and path fields differ).
- Venue: `b0i`, host mode `catalogue_selected` (row #173/#174) — **every event's true host is a
  catalogue member by construction** (`in_catalog` is `True` for all 2336/2336 rows where it is
  known; there is no "dark class" in this venue, unlike the general B-SEL fleet the original
  73%/14% figures come from — a scope caveat, not an error: this fleet is the one available for
  a zero-compute pass, and CLAIM_P3_IMPOSTOR_CONVENTION's own inventory names the same fleet's
  banked columns as the substrate).
- Zero `evaluate()` calls made. Everything below is read from files already on disk plus text
  parsing of the run logs.

## 2. What exists on disk (with exact provenance)

| Artifact | Grain | What it carries |
|---|---|---|
| `<arm>_work/seed<N>/simulations/prepared_cramer_rao_bounds.csv` | 1 row per original detection index (0..199) | `z_true`, `host_galaxy_index`, `in_catalog`, `host_draw_mode`, `s_tilde_phi_host`, `SNR`, sky/mass/distance + full Fisher block. **Missing on disk for seeds 900111/900112** (both arms) — large-CSV sync gap, consistent with the ledger's "push-reject = 4GB CSVs" note. |
| `<arm>_work/seed<N>/simulations/diagnostics/event_likelihoods.csv` | 1 row per (surviving event, h) | `L_cat_no_bh`, `L_cat_with_bh`, `B_num`, `B_num_wbh`, `g_frac`, `combined_no_bh`, `combined_with_bh`, `den_log_term`, `num_log_term_*` — **candidate-ball AGGREGATES** (Σ over every candidate host in the ball; true host included in the sum whenever recovered). Written at `bayesian_statistics.py:5763` (`self._diagnostic_rows.append(...)`, full field list at `bayesian_statistics.py:4690-4706`). |
| `<arm>_work/seed<N>/selection_tables_h_*.json` | 1 file per h, GLOBAL scalar | `beta_G_phi`, `beta_Gbar_phi`, `sigma_phi`, `sigma_4d`, `r_Malm` — fleet-wide selection integrals, not per-event. Written by `write_selection_table_json` (`bayesian_statistics.py:2590-2624`). |
| `<arm>.log` | free text, INFO level only (0 DEBUG lines in every log checked) | Per-event `"possible hosts found A/B..."` INFO line (`bayesian_statistics.py:4830`, no event index printed — order-only); one `"P6 host-recovery (h=...)"` INFO line **per h**, an ARM-LEVEL aggregate (`bayesian_statistics.py:4941-4949`) of how many in-catalog events had their true host actually returned by the ball search, split 1D (no-BH) vs 2D (with-BH mass filter). |

**What is not written anywhere** (confirmed by `grep -rn` across `darksiren_emri/` for
`candidate`, `impostor`, `in_ball`, per-host writers): the python `HostGalaxy` lists
(`candidate_hosts`, `candidate_hosts_with_bh_mass`) built per event at
`bayesian_statistics.py:4780-4830` are consumed by `p_Di` (per-host likelihood integrals,
`bayesian_statistics.py:6967` docstring) and then discarded. No per-candidate z, mass, weight,
or true-host-identity flag is ever serialized. The per-event in-ball recovery check itself
(`bayesian_statistics.py:4869-4888`) only increments scalar counters
(`_n_recovered_no_bh`, `_n_recovered_with_bh`); it is never attached to the diagnostic row.

## 3. Independent corroboration (rule 5 exoneration-check style cross-read)

`CLAIM_P3_IMPOSTOR_CONVENTION_20260822.md`, Stage-1 inventory (2026-08-22, a prior agent, prior
day), reached the identical structural conclusion from the other direction (trying to re-score
the FULL-F convention on banked columns, not trying to decompose the impostor drag):

> "The inventory refutes that premise: the banked B-SEL CSVs (`arm_event_likelihoods/bsel_seed*/
> …/event_likelihoods.csv`, 12 banked seeds) are per-(event, h) AGGREGATES — the per-host terms
> FULL-F needs (`w_pop`, `S̄_φ`, `1/imp_k` per candidate) are never stored."

Two independent passes (different fleets even — B-SEL vs the b0i b0-identity fleet used here),
same file format, same conclusion. This materially raises confidence that the missing-data
finding is a real property of the harness, not a search miss in either pass.

## 4. The covariate table that CAN be built (zero compute)

Script: `build_b4_imp_decomposition.py` (this directory). Output:
`b4_imp_decomposition.csv` (2794 rows = 24 arms × 105–131 events/arm; 20 columns) and
`b4_imp_recovery_by_arm.csv` (24 arm-level recovery rows).

**Alignment method and its verification** (this is the one non-trivial reconstruction step, so
it is spelled out): the `"possible hosts found A/B..."` log line is printed in exactly the same
iteration order as `event_likelihoods.csv` rows within one h-block, for every event except the
(at most a few per seed) zero-host fallback events, which are identifiable in the CSV by
`L_cat_no_bh == L_cat_with_bh == 0.0` exactly (`bayesian_statistics.py:4816-4823`, the
issue-#29 pure-completion fallback). The script:
1. counts the zero-host rows in the first h-block per arm,
2. derives the expected non-zero-host print count per h-block,
3. **requires** `total log-line count == expected_count × n_h_blocks` (a strong global
   consistency check — this failing would mean the block-level per-h count assumption is wrong)
   before accepting the alignment, else it records `NaN` for that arm and a `WARNING` string
   with the exact numbers,
4. verified independently that the ball composition is h-invariant in the first place
   (bc_900101: h=0.50 and h=0.52 blocks' 105 `(n_no_bh, n_with_bh)` tuples are bit-identical —
   structurally expected, since `run_mirror_seed_inprocess` widens `h.lower_limit`/
   `upper_limit` to the FULL h-grid before the ball search, per its own docstring Note,
   `correspondence_1d.py:2818-2831`).

**Result: all 24 arms aligned cleanly** (0 NaN in `candidate_count_no_bh`/`_with_bh` across all
2794 rows) — the consistency check in step 3 never failed. Two arms (900111/900112, both `bc`
and `bt`) are missing the truth columns (`z_true` etc.) only, per §2's sync-gap note; candidate
counts and diagnostics aggregates are present for them.

### Columns delivered per event

| Column | Meaning | Caveat |
|---|---|---|
| `z_true` | true host redshift | direct; NaN for 900111/900112 |
| `host_galaxy_index`, `in_catalog`, `host_draw_mode` | truth-record identity | direct; `in_catalog` is `True` for 100% of known rows in this venue (see §1) |
| `candidate_count_no_bh`, `candidate_count_with_bh` | ball size (1D / 2D-mass-filtered) | reconstructed, verified per §4 method |
| `L_cat_no_bh`, `L_cat_with_bh`, `combined_no_bh`, `combined_with_bh`, `B_num`, `B_num_wbh`, `g_frac` | per-event catalogue-leg aggregate at the h nearest 0.73 | **whole-ball sums, true host included when recovered — this is the ceiling on what "impostor decomposition" can mean at this data resolution** |
| `impostor_share_proxy_no_bh` | `1.0` when the no-BH catalogue leg is STRUCTURALLY 100% impostor (candidate ball empty, `n_no_bh==0`, 30/2794 rows); `NaN` otherwise (cannot be bounded further without per-candidate data) | a floor, not a full share; see §6 for the honest way to read the ensemble-level version of this number |

## 5. Aggregate finding: the 1D/2D recovery-rate gap (arm-level, pooled 12 seeds)

`b4_imp_recovery_by_arm.csv`, `bc`/`bt` values bit-identical per seed (expected: the candidate
ball search does not depend on `catalogue_numerator_survival`) —

| Channel | Mean recovery rate (± seed SD) | Pooled: NOT-recovered / in-catalog events |
|---|---|---|
| 1D (no-BH) | 81.6% ± 3.9% | 257 / 1397 (18.4%) |
| 2D (with-BH mass filter) | 30.4% ± 3.4% | 970 / 1397 (69.4%) |

Reading this correctly: for **69.4% of in-catalog events, pooled**, the true host was NOT among
the candidates the 2D (with-BH) search returned — meaning `L_cat_with_bh` for those events is,
by construction, **100% impostor-sourced**. That is a materially larger "structurally-certain
impostor" floor than the 1D channel's 18.4%. This is an arm-level (seed-pooled) statement only;
it does not identify WHICH events (§2's missing per-event flag), so it cannot be joined onto
`b4_imp_decomposition.csv`'s rows.

## 6. Covariate decomposition of the aggregate catalogue-leg magnitude (proxy, not isolated impostor share)

Since no column isolates the impostor-only contribution, the best available proxy is the
variance of `log10(combined_no_bh)` / `log10(combined_with_bh)` — the event's total catalogue-leg
contribution to the posterior — explained by quartile splits of each available covariate
(one-way η², i.e. between-quartile SS / total SS), computed **per seed**, then compared `bc` vs
`bt` for stability. Full per-seed table: `b4_imp_eta2_by_seed.csv`; script:
`decomp_analysis_eta2.py`.

| Channel | Covariate | mean η² (12 seeds) | bc vs bt agreement |
|---|---|---|---|
| no-BH (1D) | `z_true` (quartiles) | **0.457** (bc) / 0.445 (bt) | stable: per-seed values agree to <2% relative, same top quartile (lowest-z) in all 10 seeds with truth data |
| no-BH (1D) | `candidate_count_no_bh` (log, quartiles) | 0.071 (bc) / 0.070 (bt) | stable, but 6-8× smaller than `z_true` |
| with-BH (2D) | `z_true` (quartiles) | **0.549** (bc = bt, bit-identical) | maximal stability — see below |
| with-BH (2D) | `candidate_count_with_bh` (log, quartiles) | 0.021 (bc = bt, bit-identical) | maximal stability |

**Which covariate carries the largest share:** `z_true`, by a wide and stable margin, in both
channels and both conventions (~6-8× the candidate-count-quartile η² in every seed checked). This
is the expected direction physically (low-z candidates carry more weight through the d_L(z)⁻²-type
kernel and a denser local galaxy density) but is reported here as a **measured association on the
whole-ball aggregate**, not a proof about impostors specifically, per §4's ceiling.

**bc vs bt stability, structural note:** the `with-BH` channel's η² values are **bit-identical**
between `bc` and `bt` for every seed (confirmed in `b4_imp_eta2_by_seed.csv`) — because
`catalogue_numerator_survival` only touches the WITHOUT-BH numerator
(`bayesian_statistics.py:3627,3643,3658`: "the WITHOUT-BH catalogue numerator carries
per-candidate..."). The `no-BH` channel shows the twin's actual (small) effect: η² moves by
≤2.5% relative per seed, consistent with row #173's "the 1D twin recovered ~14%" being a modest
shift on top of a covariate structure ([`z_true`] dominance) that the twin does not change.

## 7. What would close the gap, and its cost

To get the covariates the node actually asked for (impostor z min/median, impostor σ_z, impostor
mass, per-candidate catalogue share c_i, and a true-host-subtracted score), `p_Di`
(`bayesian_statistics.py:6967`) needs an additive instrumentation hook that serializes, per event,
the `candidate_hosts`/`candidate_hosts_with_bh_mass` list (z, mass, per-host weight, and a
`is_true_host` bool) to disk — analogous to the existing `write_selection_table_json` pattern,
and NOT a change to any computed value (so arguably outside the physics-change trigger list,
though the author should rule on that explicitly before it is written, per CLAUDE.md's
physics-trigger-file gate covering `bayesian_statistics.py`).

Cost: since the candidate ball is h-invariant (§4), **one h-value is sufficient** per seed to
capture the full per-candidate list; the measured mirror-venue cost anchor is **0.2843 CPU-h per
single-h-value cell** (`PREREGISTRATION_HIER_HTHETA_20260826.md:584`, sourced to
`cluster/LAUNCHING_JOBS.md:47`). For the same 12-seed fleet, one convention (candidate
membership does not depend on `catalogue_numerator_survival` either) that is **≈3.4 CPU-h**
total — cheap, and a clean part-2 task for this node's registered-measurement half (rule 2:
a different agent from this builder must run it).

## 8. Caveats (complete list)

1. This venue (`b0i`, `catalogue_selected`) has no dark-class events, unlike the general B-SEL
   fleet the 73%/−0.079/row #149 figures were measured on — the covariate structure reported
   here (§6) is scoped to the b0-identity fleet, not directly the B-SEL headline decomposition.
   A cross-fleet transfer claim would need its own check, not assumed here.
2. `candidate_count_*` alignment relies on the log-line-order reconstruction in §4; it passed a
   strong global consistency check in all 24 arms, but it is still an inferred alignment, not a
   directly-indexed one — a future re-run producing a real per-event `event_idx` in the
   "possible hosts found" line would remove this dependency entirely (a one-line, non-physics
   logging change).
3. Two seeds (900111, 900112, both arms) lack the truth-record columns; all `z_true`-covariate
   statistics in §6 are computed on the 10 seeds that have them (`z_true` NaN rows dropped by
   `pd.qcut`, not imputed).
4. `impostor_share_proxy_no_bh` is a FLOOR (only 30/2794 rows resolve to 1.0; every in-catalog,
   non-empty-ball event is `NaN`, i.e. genuinely unknown from banked data) — it must not be read
   as "impostor share ≈ 1%"; the true rate is much higher per §5's aggregate 18.4%/69.4% figures,
   which are correct but not attributable to individual rows.
5. η² (§6) measures association with the AGGREGATE catalogue-leg value, which includes the true
   host's own contribution whenever recovered (81.6%/30.4% of the time, per §5) — it is not a
   decomposition of the impostor drag itself. Reported as the best zero-compute proxy available,
   explicitly not as a substitute for the −0.079/73% figure's own decomposition.
6. No `evaluate()` calls were made; this document and its CSVs are a builder's zero-compute
   instrument per rule 2 — any claim drawn from §5/§6 as a decisive measurement needs a
   pre-registered band and a different runner agent, not this document alone.

## 9. Files written

- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/build_b4_imp_decomposition.py` — extraction script (source of §4's table)
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/b4_imp_decomposition.csv` — 2794-row per-event covariate table
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/b4_imp_recovery_by_arm.csv` — 24-row arm-level recovery rates (§5)
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/decomp_analysis_eta2.py` — η² decomposition script (§6)
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/b4_imp_eta2_by_seed.csv` — per-seed η² table (§6)
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/COMPUTE_LEDGER.md` — appended B4.1 [IMP] part 1 measured-CPU-h row
