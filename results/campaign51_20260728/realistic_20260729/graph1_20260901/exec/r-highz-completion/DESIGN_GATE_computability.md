# DESIGN_GATE_computability.md — r-highz-completion

FRESH computability-only reviewer, sonnet/medium, 2026-09-04. Scope: `REGISTRATION_DRAFT.md` +
`MECHANISM_NOTE.md` only (`INFORMATION_FORECAST.md` not opened, per instruction). No registered
aggregate (`Δ_F`, `Δ_t`, share, `S_t`, `S_F`, any harness pooled ln-likelihood value) was computed
anywhere below. What follows is population membership/counts/hashes (explicitly authorized),
file/line pins, and ≤5-row real-slice precision checks — the same three categories the draft's own
§3 assigns to this gate.

**Verdict: GREEN.** No INSTRUMENT-DEFECT found. Every reproducible numeric claim in the draft
(population counts, sha256/md5 pins, the g-closure precision residuals, the resolved-flags token
count, the harness pooled counts, all quoted source line numbers) was independently re-derived
from real on-disk data using the rule the draft itself states, and matched exactly or within the
draft's own disclosed tolerance. Three AMBER items need resolution before/at build (§5 below); none
would make a resulting read wrong or silently unregistered, because the two underspecified items
are STOP-gated pins (mismatch halts the run, it does not produce a bad number).

---

## 1. Populations — re-derived from `covariate_table_{iiib,joint_r1}.csv` real headers, draft's stated rule

Rule used, verbatim from §1: `P_dark = {C7_log10_n_cand_1d == 0.0}`; `K` = top `round(0.1·n)` by
`C4_z_gw.rank(method="first")`; `K_dark = K ∩ P_dark`; `K_hosted = K \ P_dark`; `R` = lower half by
the same rank rule of `P_dark \ K`. Population sha256 = sha256 of the comma-joined **ascending
integer** `event_idx` list (this ordering choice was the one that reproduced the pin — string/lexical
ordering does not).

| population | iiib n (draft) | iiib n (reproduced) | iiib sha256 match | jr1 n (draft) | jr1 n (repro) | jr1 sha256 match |
|---|---|---|---|---|---|---|
| P_dark | 606 | 606 | ✓ exact | 493 | 493 | ✓ exact |
| K | 159 | 159 | ✓ exact | 159 | 159 | ✓ exact (same event set both venues, as claimed — z min 0.7360883322653011 identical both venues) |
| K_dark | 144 | 144 | ✓ exact | 111 | 111 | ✓ exact |
| K_hosted (reported-only) | 15 | 15 | n/a (no hash pinned) | 48 | 48 | n/a |
| R | 231 | 231 | ✓ exact | 191 | 191 | ✓ exact |

z-bounds cross-check: K z ≥ 0.7360883… (draft: "≥ 0.736") ✓; R (iiib) z ≤ 0.5406176… (draft: "≤
0.541") ✓; R (jr1) z ≤ 0.4832951… (draft: "≤ 0.483") ✓. P_dark median z (iiib) = 0.6203155… (draft
§10 disclosed 0.62) ✓. `P_dark ≡ C2_hosted_exact==False ≡ C3c_censored==True` set-identity (G-3a):
verified True on all 1588 rows, both directions.

Set-identity `P_dark` (covariate `C7==0`) vs. `event_likelihoods.csv` bit-exact-zero class (G-1a),
checked on the **full 65,108-row iiib table** (not just the 5-row slice — this is a population/schema
check, not an aggregate): `L_cat_no_bh==0.0` at **all 41 nodes** for exactly the 606 `P_dark` events,
0 partially-zero events (would indicate a non-bit-exact float that only *looks* like 0.0 at some
nodes) → **True, exact set equality**. Same for `L_cat_with_bh`. `D_tilde_phi` and `den_log_term`
have exactly 1 distinct value per h-node (checked all 41 nodes) — confirms event-independence (G-1e).

## 2. File pins — verified on disk

| pin | draft value | measured | match |
|---|---|---|---|
| iiib `event_likelihoods.csv` md5 | `8e6a2c18dc5838dd1d52641589243672` | same | ✓ |
| iiib row count | 65,108 rows + header | 65,109 lines total = 65,108 rows + header | ✓ (= 1588×41) |
| jr1 `event_likelihoods.csv` md5 | `745954a0fdee5f10878fb5e622a06144` | same | ✓ |
| jr1 row count | (implied 65,108) | 65,109 lines | ✓ |
| `covariate_table_iiib.csv` sha256 | `90c92026bb7f…189f7b0` | same (full 64-hex match) | ✓ |
| `covariate_table_joint_r1.csv` sha256 | `fc2eebe7fa66…849fcdf3a` | same | ✓ |
| production commit | `1ec9514dd1808c48b18c0792dce558e5bba0f116` | `git log` confirms this is a real, reachable commit ("row #290… Research Graph 1 RATIFIED") | ✓ |
| harness seed dirs 901000–901066 | 67 universes | `ls` → exactly 67 `seed*_S` dirs | ✓ |
| harness `n_draw_requested==200` filter | yields 67 | all 67 checkpoints carry `"n_draw_requested": 200`, none missing | ✓ |
| harness manifest sha256 | `6a06063dd5…adb1c0a2` | **not reproduced** — see §5 item A | AMBER |
| H_GRID_41 core, stencil nodes | 0.60–0.86, incl. 0.725/0.730/0.735 and h_true=0.73 | all present, 41 distinct nodes | ✓ |
| harness resolved-flags token count | 13 tokens | checkpoint `resolved_flags` dict has exactly 13 keys | ✓ |
| harness pooled Σ n_scored / Σ\|P_dark,u\| / Σ\|K_u\| / Σ\|K_dark,u\| | 12,060 / 4,826 / 1,207 / 1,148 | 12,060 / 4,826 / 1,207 / 1,148 (recomputed over all 67 universes using the draft's §1 harness recipe, incl. the `dist_to_redshift(luminosity_distance, h=0.73)` z_u call and the "zero at h=0.73 ≡ zero at all 41 nodes" equivalence, which held 67/67) | ✓ exact, all four |
| n_scored,u range | 173–192 | measured 173–192 | ✓ |

## 3. The (I-2D)/(I-1D) identity and g-closure — verified against real source + real 5-row slice

`path_a_completion_numerators` (`:2509` both commits, unchanged) confirmed: `mode="derived"`
(production's actual `completion_b_scale` resolved value) returns `B_num_phi=B_num,
B_num_wbh_phi=B_num_wbh` unmodified — the identity's premise is exact, not approximate, for the
resolved production config. `combined_without_bh_mass`/`combined_with_bh_mass` (`:6741-6746`)
confirmed to reduce to `B_num_phi/D̃`, `B_num_wbh_phi/D̃` whenever `L_cat_*==0`, **regardless of the
weight multiplying the zeroed term** — so the zero-candidate identity holds structurally, not
coincidentally.

5-row real slice (events 0–4, h=0.73, iiib) reproduced with `float_precision="round_trip"` (default
pandas float parsing loses ~3 ULP here and understates precision — see §5 item C):

| check | draft claim | reproduced (events 0–3, zero-candidate) |
|---|---|---|
| `\|ln combined_with_bh − (ln B_num_wbh − den_log_term)\|` | ≤ 2.7e-15 | max 2.665e-15 ✓ |
| `\|ln combined_no_bh − (ln B_num − den_log_term)\|` | ≤ 1.8e-15 | max 1.776e-15 ✓ |
| `\|den_log_term − ln D_tilde_phi(7 s.f.)\|` | 4.2e-9 | 4.22e-9, identical all 5 events ✓ |
| `\|g_frac − B_num_wbh/B_num\|/g_frac` | ≤ 1.4e-7 | max 1.35e-7 ✓ |
| `\|num_log_term_with_bh − ln B_num_wbh\|` | 0 | 0 (events 0,1,3), 1.78e-15 (event 2) — both "≈0" at float64 noise floor |

All well inside the actual G-1(b) gate band (1e-9), so this is a documentation-precision note, not a
gate risk (see §5 item C).

## 4. Line-number re-pin: HEAD → commit `1ec9514d` (`git show`)

`git diff 1ec9514d HEAD -- darksiren_emri/bayesian_inference/bayesian_statistics.py` is **one hunk**:
`@@ -4653,9 +4653,26 @@` (an 18-line admissibility-guard insertion, row #301/#308 decoupling — net
+17 lines). Every line MECHANISM_NOTE quotes is either before the hunk (unchanged) or after it
(HEAD − 17); both endpoints were content-diffed byte-for-byte, not just offset-arithmetic — all
matched exactly.

| content | HEAD line | commit `1ec9514d` line |
|---|---|---|
| `def p_Di(` | 5932 | **5915** |
| `elif _use_g_inside and self.h in …` | 6684 | **6667** |
| `combined_without_bh_mass = float(` … block | 6741–6746 | **6724–6729** |
| `_den_used = D_tilde_phi if …` | 6726 | **6709** |
| `beta_Gbar_phi = self._beta_Gbar_phi_table[self.h]` / `sigma_phi = …` | 6705–6706 | **6688–6689** |
| `"no catalog results found"` branch | 6064–6068 | **6047–6051** |
| `def _completion_numerators(h_eval…)` | 6521 | **6504** |
| `_sel_1d = _sel_cell in ("1d", "fused")` | 6390 | **6373** |
| `if _sel_1d:` (S̄_φ fused branch) | 6558 | **6541** |
| `return (1.0 - f_z) * p_gw * dVc / (1.0 + z)` | 6370 | **6353** |
| `return base * s_bar_phi` | 6408 | **6391** |
| `z_upper = dist_to_redshift(` … `z_lower = max(…)` | 6530–6540 | **6513–6523** |
| `p_gw` / `f_k` block | 6343, 6351–6358 | **6326, 6334–6341** |
| `completion_mass_factor_g_sel` region | 6462–6471 | **6445–6454** |
| `return base * g_i` | 6519 | **6502** |
| `g_frac_used = (B_num_wbh / B_num) …` | 6600 | **6583** |
| `L_comp = float(B_num / beta_Gbar) …` | 6767 | **6750** |
| `den_log_term` / `num_log_term_*` write block | 6803–6811 | **6786–6794** |
| `_seven_sf = (…)` and writer region | 5438–5467 | **5421–5450** |
| `def path_a_completion_numerators(` | 2509 | **2509** (unchanged, before the hunk) |
| `return B_num, B_num_wbh, 1.0` | 2552 | **2552** (unchanged) |
| `def path_a_mixture_objects(` | 2449 | **2449** (unchanged) |
| `n_hat_w_phi = …` / `alpha_G_phi = …` / `D_tilde_phi = …` | 2494, 2496, 2497 | **unchanged** |
| `def precompute_missing_completion_denominator(` | 1327 | **1327** (unchanged) |

**Every quoted line's content is byte-identical at both commits** — this is a pure line-shift from
one unrelated insertion, not a code change to any quoted region. MECHANISM_NOTE's HEAD numbers are
safe to re-pin mechanically by −17 for everything ≥ line 4679, unchanged below that.

## 5. Findings

**A. (AMBER) Harness-manifest sha256 construction is underspecified in prose, and I could not
reproduce it.** §1 gives only "`per-file md5 manifest sha256 …` (manifest = sorted \"seed md5\"
lines)". I tried four reasonable readings (bare `"{seed} {md5}"` sorted, `"seed{s}_S {md5}"`,
`"{seed} {filename} {md5}"`, and standard `md5sum`-style `"{md5}  {relpath}"` sorted by path) against
the real 67×2 files; none produced `6a06063dd5…adb1c0a2`. Unlike the population sha256s (fully
specified — "comma-joined ascending event_idx list" — and reproduced exactly), this pin's exact byte
format isn't nailed down in the draft, and there is **no existing script in the repo** that already
implements it (`offset_subset_reads.py`, the only sibling reads script, has no manifest logic to
reuse). *Consequence:* this is a STOP-gated pin (§1: "STOP on mismatch"), so a wrong reproduction
fails safe — the reader will halt rather than proceed on a silently-different harness file set — but
the builder needs the exact algorithm specified (or embedded as a helper the design gate can diff
against) before `--dry-run` can be expected to pass on the first attempt. Recommend: pin the exact
construction as a short code block (or a `harness_manifest.py` helper checked into `exec/`) rather
than prose.

**B. (AMBER) §5's non-additivity band (`\|r\|/\|Δ_F\| ≤ 0.6`) has no corresponding `--flag` in the
§8 launch block.** The other four bands each have an explicit CLI arg (`--share-own 0.5
--share-diffuse 0.2 --rho-hi 0.5 --rho-lo 0.2 --z-gate 3.0 --se-unpowered 0.1`); `0.6` appears only in
frozen prose (§5, used as a precondition inside both the TERM-OWNS row and DIFFUSE-IN-TERMS's
reachability note). The value is unambiguous, so a builder hardcoding `0.6` produces the *same*
registered read — this is an auditability/consistency gap, not a correctness risk. Recommend adding
e.g. `--nonadditivity-max 0.6` to §8 so all five bands are visible in the run's own argv/metadata,
matching the pattern the other four already set.

**C. (AMBER, informational) Default pandas CSV float parsing does not reproduce the disclosed
1e-15-level closure residuals.** With plain `pd.read_csv(...)`, the 5-row slice residuals came out
up to 4.44e-15 (vs. the draft's disclosed ≤2.7e-15) — re-running with `float_precision="round_trip"`
reproduced the draft's numbers essentially exactly (§3 table above). This is inert for gate G-1(b)
(actual band 1e-9, satisfied either way by ~5 orders of magnitude), but the builder should read the
full-precision columns (`B_num`, `B_num_wbh`, `combined_*`, `den_log_term`) with
`float_precision="round_trip"` (or an equivalent exact parser) if the closure diagnostic is meant to
reproduce the disclosed 1e-15-scale numbers rather than merely pass the 1e-9 gate.

**D. (informational, no action needed) The harness/production resolved-flags equality (G-3d) is
not a literal key-name match, and the design gate should not implement it as one.** 2 of the
harness checkpoint's 13 `resolved_flags` tokens (`catalogue_numerator_survival`, `mass_filter_sigma`)
have no literally-matching key in production's `run_metadata_0.json` (`cli_args` stores raw argparse
strings; not all resolved attributes are explicitly passed on the CLI). Tracing the mechanism:
`b8_cal_harness.py:1250` passes `resolved_flags_out=resolved` into
`darksiren_emri/validation/correspondence_1d.py:3292`, which reads the estimator's actual
runtime-resolved attributes post-construction — the same mechanism row #347 already used and
validated ("13 tokens, 67/67"). The builder must reuse *that* extraction path (or row #347's own
comparison script, if it exists separately) rather than a naive `cli_args` string diff, which would
spuriously fail on these 2 tokens. This is confirmed real, existing, production code — not a gap —
flagged here only so the builder doesn't reinvent a wrong comparison.

## 6. Structural checks (draft's own requirements)

- **max_revisions = 2**: present, §0 preamble line 7 ("max_revisions = 2 (pre-launch design
  revisions, counted by the design gate; post-disposition revisions are a separate counter starting
  at 0)"). This is design-gate revision #0 of that counter.
- **Blindness line**: present, §10, and consistent with what a computability check is allowed to
  touch — the five pre-read categories it discloses (prior-row anchors, a different row-#347
  statistic, population counts/z-ranges, the 5-row slice, `D_tilde_phi` single-valuedness) are
  exactly the categories this review also used; nothing beyond them was touched here either.
- **Three-valued + fresh RULE**: the *production* ownership table (§5, first table) is genuinely
  three-valued (TERM-OWNS(t) / DIFFUSE-IN-TERMS / INTERMEDIATE). The *harness* mapping (§5, second
  table) is five-valued (ESTIMATOR-INTERNAL / PRODUCTION-ONLY / FLOOR-CONSISTENT / INTERMEDIATE /
  UNPOWERED-CONTROL) — not three-valued, but the draft doesn't claim it is; both tables explicitly
  end "All rows return as fresh RULEs; nothing here is pre-approved" (line 161) and every
  row/outcome in both tables is tagged toward a fresh RULE (§5, §9 items 3–6). No pre-approval leak
  found.
- **R0 sweep citations** (Gray 2020 "Eq. 32", MFG 2019 Eqs. 5–7): both verified present in
  `docs/LITERATURE_WARNINGS.md` (Gray 2020 section; MFG row `MFG-a`). "Gray 2020 Eq. 32" specifically
  traces to `docs/derivations/G2c_gray_a9_a10_mapping.md:32,152`, which maps it to `(A17)` — the
  out-of-catalogue numerator integral — and its quoted formula matches `T_B`'s integrand in
  MECHANISM_NOTE §3 term-for-term (the `(1−f_k)`, `p_gw`, `dVc/(1+z)` factors, the event-pixel `f_k`
  attribution cited to GMV 2022 Eq. 5). No new literature row needed, as the draft claims.
- **T0 convention import**: `_load_matrix`, `_physics_floor_apply`, `_moments` all exist in
  `build_influence_vector.py` with the claimed content (`w = np.gradient(h_grid)`, log-sum,
  gradient-trapezoid weights, physics floor). `tier0_bootstrap_jackknife.py` (source of record)
  exists on disk.

## 7. What was intentionally NOT computed

No `Δ_t`, `Δ_F`, `r`, `s_t`, `S_t`, `S_F`, harness pooled ln-likelihood value, null-draw CI, or
z-quintile score was computed at any point in this review, over any of `K_dark`, `R`, `K`, `P_dark`,
or the harness universes' event_likelihoods tables. `--dry-run` mode of `highz_decomp_reads.py` was
not run because the script does not exist yet (correctly disclosed in the draft's own §11 item (b) —
this design gate ran before the builder's stage, on the draft alone plus the fixed, already-frozen
`covariate_table`/`event_likelihoods` inputs).
