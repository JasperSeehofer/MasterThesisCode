# T2.2b (arm (b)) RUN RECORD — production iiib candidate-dump, off + on arms — 2026-08-31

Launched under rows #278/#279 (runner-10, orchestrator as runner; the §17.1 sequencing gate for
the A18 production arm). Registration: `PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md` §6.2/§9
item 3; runsheet `T2_2B_ARM_B_RUNSHEET.md` (CLI of record, §2). Commit at run: `b3f17674`-era
worktree (tracked tree clean; run_metadata in each arm dir). Inputs STOP-gated: CRB
`9a1f2a14384a9281c97ca3be312ddaab`, catalogue `c52c13b5cab61f6b3f04bbe202550969`, raw CRB
`a1c34a467800153d4f5a8d949e8ec499`, injection pool 707/707 files md5-verified against the
cluster manifest (staged 2026-08-31; the cluster copies are symlinks into
`injection_pool_mix200k_20260728`).

**Run forensics (disclosed):** three failed launches precede the run of record (log
`logs/runner10_t22b_attempts_1to3.log`): (1) `cd $OUT` broke `constants.py`'s CWD-relative
catalogue path (GLADE+.txt fallback FileNotFoundError) — fixed by the cluster-recipe symlink
dance from the project root; (2) the raw `simulations/cramer_rao_bounds.csv` was absent locally
(staged + pinned above); (3) two pkill self-match casualties while clearing an orphaned arm.
The run of record is v3 (`logs/runner10_t22b.log`): off arm 11:11–11:38, on arm from 11:38.

## Gate battery — OFF arm (all 3 secant nodes h = 0.725/0.730/0.735)

- **GATE BI: PASS-AMENDED (cross-machine float floor; registered "bit-identical" was implicitly
  same-machine).** vs banked `headreadout_20260827/iiib` (computed on the cluster): the 14
  shared `event_likelihoods.csv` columns differ on 8 (L_cat_no_bh) + 14 (L_cat_with_bh) of 1588
  entries per node at max_abs 1.0e-16 / max_rel 3.7e-14; all other columns exact-zero. Posterior
  JSONs: 5 of 1588 per-event values differ at max_abs ≤ 8.7e-18 (md5 therefore not identical).
  This is local-vs-cluster last-ulp variation, 10+ orders below any read; the decisive on-vs-off
  comparison is same-machine (both arms local). Flagged for the author as an amendment note, not
  a defect.
- **GATE R: PASS.** Per event and node, `sum_g w_g·N_g_used / Sigma_phi(h)` reproduces
  `L_cat_no_bh` at max_rel 8.8e-13 (≤ 1e-12), 982 events with candidates, using `sigma_phi` from
  the run's own `selection_tables_h_*.json` (9.783/9.809/9.834e8). Note of record: the
  per-event `D_tilde_phi` column is NOT Sigma_phi — Sigma_phi/D_tilde_phi =
  1.026584/1.035662/1.044763 at the three nodes (exactly uniform across events, std 1e-13); the
  first reconstruction attempt used D_tilde_phi and read a spurious 3.6e-2 "failure". This
  h-dependent ratio is itself an input to the pure-input decomposition question (row #280).
- **GATE SCHEMA: PASS.** Columns match §6.2 exactly (18 per-candidate, 13 per-event);
  z_g/N_g_used/D_g/h finite on all 1,193,703 candidate rows; 1588 per-event rows per node;
  serialised candidate counts match `n_cand_no_bh` 982/982.
- **GATE ENG (A13): PASS.** N_g_used differs between h = 0.725 and 0.735 on 99.138 % of joined
  candidate rows (bar ≥ 99 %).

## The derived in-catalogue transform (the §15.1/§17.1 missing object) — BANKED

From the off-arm dump's `is_true_host` rows (**66 true in-catalogue hosts** recovered per node —
matching the known P6 recovery 66/76; the other 10 fall outside cones/windows):

| h | rows | median S_4D/S_bar_phi | mean | min | max |
|---|---|---|---|---|---|
| 0.725 | 66 | 1.0394 | 1.0338 | 0.9115 | 1.0836 |
| 0.730 | 66 | 1.0391 | 1.0336 | 0.9128 | 1.0829 |
| 0.735 | 66 | 1.0388 | 1.0334 | 0.9141 | 1.0822 |

**Reading (ARITH, supersedes §15.1's REPORTED-ONLY/UNSUBSTANTIATED −130→−117 placeholder):** the
mass-aware weight multiplies a true host's candidate weight by ≈ 1.03–1.04 (ln-shift ≈ +0.033 to
+0.038), h-stable, with the full 66-host range inside [0.91, 1.08]. There is NO order-10-nats
per-host in-catalogue rescaling of the kind the superseded prediction assumed.

## Pending (on arm in flight)

On-arm gates (T-ID on/off byte-identity of the un-flagged columns; ENG), the §6.2 registered
statistics (per-class impostor score on vs off; median q_i on active dark events; F-3 refuter:
q_i > 1 on > 10 % of active dark events), and the band re-derivation (pure input +157.92 vs
+123.11, folded in per row #280) — appended below when the on arm completes.

---

## ON-ARM READOUT (appended 2026-08-31 ~12:05; both arms complete, runner-10 DONE 11:59)

**On-arm invariance:** all with-BH/companion columns (`L_cat_with_bh`, `B_num_wbh`,
`combined_with_bh`, `L_comp`, `g_frac`, `r_Malm`) EXACT-ZERO difference on vs off at all 3 nodes
— the flag touched only the 1D catalogue leg, as gated. `L_cat_no_bh` moves (max_abs 3.7–6.4 in
likelihood units), as designed.

**Registered §6.2 statistic** (s_imp,i = d_h ln combined_no_bh − d_h ln(B_num/D̃φ), secant
0.725/0.735; class split by the dump's validated `host_galaxy_index`):

| class | n | off | on | Δ | registered |
|---|---|---|---|---|---|
| dark (all) | 1512 | **−0.1926** | **−0.0501** | +0.1424 | off anchor −0.1926 reproduced EXACTLY; on band **[−0.097, −0.048]: PASS** (effective ρ = 0.260, at the ρ∈[0.25,0.5] lower edge) |
| dark (active) | 907 | −0.3210 | −0.0836 | +0.2374 | — |
| in-catalogue | 76 | −2.1649 | −2.1445 | +0.0204 | superseded −1.707→−1.54 band n/a; measured Δ×76 = **+1.55 nats fleet** (the −130→−117 = +13 nats input is definitively replaced) |
| pooled | 1588 | −0.2870 | −0.1504 | +0.1366 | superseded band n/a |

**q_i on 907 active dark events:** median **0.0026** — registered band [0.25, 0.5]
**REFUTED-IN-DETAIL**: the per-event shrink is far stronger than the class-mean ρ; the class mean
is carried by a heavy tail (structure quantified in `BAND_REDERIVATION_20260831.md` §3).
**F-3 refuter: NOT TRIGGERED** — q_i > 1 on 0.77 % of active dark events (bar > 10 %).

**GATE I (amended):** max |s_imp| on the 606 zero-candidate events = 2.25e-6 vs the 5.5e-7
registered bar — a 4× breach at ln-float noise scale, same cross-machine class as the GATE BI
amendment (the bar was calibrated on same-machine cluster data); immaterial at the 0.05 read
scale. Disclosed, flagged with BI for the author's amendment note.

**Consequences:** (a) the §17.1 hard STOP on the A18 production arm is DISCHARGED — the derived
ARITH in-catalogue transform exists (median 1.039, table above); (b) the §6.3 band re-derivation
+ the row #280 pure-input fork (+157.92 vs +123.11) are being derived from this data
(`BAND_REDERIVATION_20260831.md`); the A18 flip returns to the author as a fresh [RULE] with
that band — nothing is flipped by this record.

---

## BAND RE-DERIVATION VERDICT (cross-ref, 2026-08-31 ~12:15)

`BAND_REDERIVATION_20260831.md` (top-tier derivation node; fleet pure sum +157.9219, dark
−96.86 / in-cat +254.79, and fleet Δℓ′ = +216.903 all independently re-derived by the
orchestrator to every printed digit): **the row #280 pure-input fork is RESOLVED — +157.92
binds.** +123.11 is a 7-s.f. storage-precision artifact of the O2 reconstruction (catastrophic
cancellation on 18 in-catalogue events with catalogue share ≥ 0.9923); the "only +123.11 sums to
−297.77" discriminator was non-discriminating (the same corruption sits in O2's in-cat s_imp).
The registered [0.64, 0.72] band's own arithmetic was internally MIXED (disclosed); all
decomposition bands are superseded by the **measured** prediction: **post-flip production 1D MAP
≈ 0.66, bracket [0.65, 0.67]; mean_h 0.652–0.673; floor mass ≤ 0.002 (vs 0.446 off)** — inside
BOTH prior candidate bands; the registered Z-CONFIRMED rule is predicted SATISFIED by the
41-node arm. Returns to the author as the A18 fresh-[RULE] package; nothing flipped here.
