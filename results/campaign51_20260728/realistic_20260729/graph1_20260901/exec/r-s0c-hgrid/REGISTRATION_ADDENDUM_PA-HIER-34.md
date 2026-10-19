# PA-HIER-34 — S0-C registration addendum: the S0-B θ-grid at three h-values (∂score/∂h)

Date: 2026-09-04 (batch 2). Node: r-s0c-hgrid (Research Graph 1 Branch D follow-on; row #349
item 3). Prereg author: top-tier subagent B (xhigh). Author of record for every scientific
decision: Jasper Seehofer.

**Status: PROPOSED — returns to the author as a fresh [RULE]. Append-only addendum to
`results/campaign51_20260728/realistic_20260729/PREREGISTRATION_HIER_HTHETA_20260826.md`;
that file is NOT edited (its last in-force block is PA-HIER-33, ratified rows #278/#280; no
PA-HIER-34 exists anywhere in the repo — grepped 2026-09-04).** This file is the PA-HIER-34
text of record until it is either appended to the prereg by a ledger row or withdrawn.
Convention: [DOC] read from a committed/banked file · [LOCAL] recomputed here · [INFER] derived.
Every cap marked ORCHESTRATOR-DERIVED carries its derivation. Zero fresh scientific choices:
every number below is quoted from a registered source or derived from one; where a choice was
unavoidable it is flagged `[FLAG]` for veto.

Authorization chain, quoted. Row #349 (batch-2 grant, author verbatim: *"you can easily do
another big batch of research and feel free to also use the cluster!"*; chair reading (3):
*"S0-C — S0-B θ-grid at h ∈ {0.665, 0.73, 0.78} (≈6 CPU-h) behind a PA-HIER addendum"*).
Dossier `exec/d-photoz-leverage/DOSSIER_20260903.md` item 2 (*"a follow-on registered arm
'S0-C' — the same 5 θ-nodes at 3 h-values (0.665 / 0.73 / 0.78) to read ∂(score)/∂h"*).
Morning docket R5 provisional reading: *"Defer both behind R4"* — carried here as the
CONDITIONAL clause of §5.4, not overridden.

---

## 0. Existence contract (three-valued)

| input | state | evidence |
|---|---|---|
| S0-B run of record, job 6779532, 5 nodes at h=0.73 | present (retrieved) | `graph1_20260901/retrieved/s0b_run_20260902/s0a_seed900101/node_{truth,b_plus_re,b_minus_re,s_plus,s_minus}_iiib_sites2.2_nosmear/` — `event_likelihoods.csv` (1588 rows × 19 cols, `h` column = {0.73} only [LOCAL]), `selection_tables_h_0_73.json`, `es_null_det.csv` |
| S0-B reads | present | `exec/m-s0b-production/READOUT_RECORD.md`, `CHAIR_REDERIVATION_20260903.md` (+ ERRATA D3/D5), rows #332/#336/#337/#345 |
| driver | present, UNCHANGED since the S0-B commit | `fanout1_20260829/hier_s0_driver.py`; `git diff --stat 081b1f28 HEAD -- hier_s0_driver.py darksiren_emri/` is EMPTY at local HEAD `08060e2a` [LOCAL] |
| `--h-nodes` flag on the driver (S0-A/S0-R arms) | present | `hier_s0_driver.py:2926-2933` ("Comma-separated h values fused into ONE evaluate() call per theta node"); `H_BOUNDS = (0.50, 0.86)` (`:94`) — both fresh h-values inside |
| node-dir naming | h is NOT in the node-dir suffix | `_node_dir_suffix` (`:1093 ff.`) → `node_<node>_iiib_sites2.2_nosmear`; a second h at the same out-root would OVERWRITE the h=0.73 cells ⇒ per-h out-roots (§3.3) |
| `--score-only` reads a node dir at a chosen h | present | `gather_node_results_from_disk` + `read_event_ln_l(diag_csv, score_h)` (`:963-1000`, `:2543-2630`) — selects rows by `h` column, `rtol=1e-9` |
| production comparand for the S0-B truth node (R4) | **absent locally; job 6790708 submitted 2026-09-04** | `exec/batch2_cluster_ops/OPS_RECORD.md` step 5 — result unread |
| cluster state | READY ✓ (preflight 2026-09-03), HEAD `40509193` after ops sync; local HEAD `08060e2a` (docs commits only since) | `OPS_RECORD.md` steps 1–2 |

---

## 1. The registered text this addendum extends (quoted verbatim, not paraphrased)

**§2.1, S0-B row** [DOC]: *"S0-B (decisive production read — DEFERRED, author costing gate) |
production venue at CoR-P, the banked realistic scattered catalogue | θ-score at truth-θ where
the measured tilt actually lives | NOT AUTHORIZED at registration. ~75–101 CPU-h (§7.2). Runs
only on an explicit author costing grant, and only after S0-A/S0-R land."* — and §4.1's anchor
note: *"Anchor derivation for B0-B (the deferred production read): identical bands, applied to
S0-B."* (The reads in `READOUT_RECORD.md` cite these bands as "§2.1(e)"; the band text itself
lives in PA-HIER-31(e) below — same object, two labels.)

**PA-HIER-31(d)** [DOC]: *"Four θ-nodes at h = 0.730 on venue **iiib** (production, CoR-P per §1
below), plus the truth node (θ = (0,1)) which doubles as **C0**, the shared baseline gate task:*
```
truth        (0, 1)              = C0
b_plus_re    (+0.033, 1)
b_minus_re   (−0.033, 1)
s_plus       (0, √2)
s_minus      (0, 1/√2)
```
*Per-event statistic (both channels computed, primary = no-BH; `ln` of `combined_*` per
`hier_s0_driver.py:242-245`):*
```
score_b,i   = [ lnL_i(+0.033,1) − lnL_i(−0.033,1) ] / 0.066
score_lns,i = [ lnL_i(0,√2)   − lnL_i(0,1/√2) ] / ln 2
Z_x = mean(score_x) / SEM(score_x)
```
*Read **pooled** (N = 1588); … class is defined **at h = 0.73, this arm's single node**"*.

**PA-HIER-31(e)** [DOC]: *"**B0-B** ≡ `|Z_b| ≤ 3` **and** `|Z_lns| ≤ 3` pooled (two-sided) ⇒
**LEVER-DEAD-AT-N (production)**; either `> 3` ⇒ **LEVER-LIVE** — then B0-M (materiality): MIXED
if `|b̂| < 0.0165` (half the 0.033 step) or `|ln ŝ| < 0.5·ln√2 = 0.173`; B0-P (power): `σ_b <
0.0661` and `σ_ln s < ln 2`, else UNPOWERED (no DEAD claim). Curvature leg: quadratic fit through
the three b-nodes (truth, b_plus_re, b_minus_re) → `b̂ = −S′/S″`, `σ_b = 1/√(−S″)`; likewise in
ln s. … **All verdicts carry the REPORTED-ONLY cap**"*.

**PA-HIER-31(g), blindness (d)** [DOC]: *"(d) single h (0.730 only)"* — the blindness this
addendum removes.

**PA-HIER-33 (rule)** [DOC]: *"For an arm with s-nodes at `ln s = 0, +/-Delta` (`Delta = ln
sqrt2`), define on each channel `Es_null^{(arm)} = (Delta^2/6) . [ -3 <l'_i l''_i> - <l'_i^3> ]`,
`l'_i = score_lns_i`, `l''_i = [l_i(+Delta) - 2 l_i(0) + l_i(-Delta)]/Delta^2`, `score_s_i =
score_lns_i - Es_null^{(arm)}` (a pooled scalar shift, the arm's own null; NOT a per-host table),
`Z_s = mean(score_s) / SEM`, `SEM = max(per-event SEM, seed-clustered SEM)` (PA-HIER-5 leg (a)),
with the bootstrap uncertainty of `Es_null^{(arm)}` added in quadrature to the SEM."* Ratified
rows #278/#280. Applied here per h (§4.2).

**The S0-B read this addendum follows** (rows #336/#345, chair-rederived [DOC]): no-BH
`score_b_re` mean −0.6822 ± 0.1293 (Z −5.274); `score_lns` −0.0327 ± 0.0045 (Z −7.188;
PA-HIER-33-corrected −7.101); curvature leg b̂ = −0.01137 (σ_b 0.00324), ln ŝ = −1.165 (σ 0.150).
Disposition **LEVER-LIVE · MIXED · POWERED · REPORTED-ONLY**, population-robust (row #337).
**OPEN (row #345 D3):** the S0-B truth node differs from the mass-aware-matched baseline
`c0prime_off_iiib` on 562/1588 events (`combined_no_bh` max_rel 0.734, +2.58 nats); PA-HIER-31(d)
"truth = C0" and GATE T-ID are NOT satisfied; cause OPEN; R4 comparand job 6790708 pending.

---

## 2. Object and question (stage 0/1)

**Object.** S0-B is LEVER-LIVE at a single h = 0.73. Dossier item 2 asks what h-bound the
θ-pull contributes to the three-way residual split (`d-residual-attribution`, "irreducible venue
physics"); *"A bound in h requires the mapping ∂h/∂ln s at the MAP, which S0-B does not measure
(single h)."* The missing quantity is the h-dependence of the θ-score, ∂score/∂h.

**Question q-s0c.** Does the θ-score vary with h at a resolvable rate on the production venue,
and, if it does, what h-displacement Δh_θ = −score(0.73)/(∂score/∂h) does the linearised score
null at? Refute by: `|Z(∂score/∂h)| < 3` on both axes (derivative unresolved ⇒ Δh_θ carried as
report-only, no h-bound enters the split).

**What a perfect analysis would say (stage 1, [INFER]).** If the θ-pull were a pure
photo-z-kernel object it would be nearly h-inert over 0.665–0.78 (the host-z kernel enters
through z_host, not h) ⇒ ∂score/∂h ≈ 0 and |Δh_θ| ≫ the grid ⇒ no h-bound. If instead the pull is
a catalogue-leg normalisation object coupled to the d_L(h) mapping (the B4/B4.3 class named in
PA-HIER-31(g) blindness (b)), it tracks h and Δh_θ lands at grid scale. The measurement
discriminates these two, which S0-B alone cannot (blindness (b)+(d)).

---

## 3. Design (stage 2)

### 3.1 Cells
The five registered θ-nodes (PA-HIER-31(d) verbatim, `--b-half-width 0.033`) × three h-values:

| h | source of the 5 cells | fresh compute |
|---|---|---|
| 0.665 | this addendum | 5 cells |
| **0.730** | **job 6779532 (S0-B of record), REUSED — see §3.2** | 0 |
| 0.780 | this addendum | 5 cells |

h-values are the dossier's own (item 2: "0.665 / 0.73 / 0.78"); 0.665 is the production MAP
node of row #302 (iiib 2D/1D map_h 0.665), 0.78 the mirror-side node at the same coarse-grid
distance class. Both are H_GRID_41 nodes and inside the driver's `H_BOUNDS = (0.50, 0.86)`.
The stencil is NOT symmetric (Δ− = 0.065, Δ+ = 0.050) — handled exactly in §4.3.

### 3.2 Reuse of the h = 0.73 set (byte-identity argument, [LOCAL])
A fresh h=0.73 cell would be launched as `--h-nodes 0.73`; the driver resolves that to
`h_values=(0.73,)`, which is bit-for-bit the default `(H_GEN,)` job 6779532 ran under
(`main()`: `h_values = tuple(float(x) …) if args.h_nodes else (H_GEN,)`). Every other flag is
identical to `cluster/graph1_m_s0b_production.sbatch` (§3.3). Code: `hier_s0_driver.py` and the
whole `darksiren_emri/` package are UNCHANGED between the S0-B commit `081b1f28` and local HEAD
`08060e2a` (`git diff --stat` empty). Same `--cpus-per-task=16` (worker count = affinity − 2 =
14, `evaluate.sbatch` header; kept identical because row #337 showed summation-order noise can
flip the `L_cat_no_bh == 0` label). Therefore the 6779532 cells ARE the h=0.73 cells of this
design and are not re-run. **Registered here: h ∈ {0.665, 0.78} only (10 fresh cells).**
Disclosure: the h=0.73 numbers are already SEEN (rows #336/#345); every statistic in §4 that
uses them alone is a re-read, and the derivative is the fresh, blind object.

### 3.3 Configuration (identical to job 6779532; PA-HIER-31(b)/(g), row #332 item 2)
`--arm S0-A --config iiib --theta-sites 2.2 --smear off --seeds 900101 --b-half-width 0.033
--jobs 1 --nodes <one node>` plus the ONLY new flag `--h-nodes <h>` (single value per task).
T1.2/T1.3 mirror instruments at driver defaults: `theta_phi_divisor off`, `theta_zwindow off`,
`z_window_k 1.0`, `sky_cone_k 1.5`, `catalogue_leg_1d_mass_aware off`. Venue iiib loads (never
draws) the pinned production CRB (`c1d.CRB_CSV_MD5 = 9a1f2a14…`) and reduced catalogue
(`c52c13b5…`), STOP-gated inside `build_iiib_venue` (row #295). One node per SLURM array task
(GATE TABLE-FRESH, PA-HIER-31(j)). Out-root per h (`$WS/graph1_s0c_hgrid_20260904/h_0p665`,
`…/h_0p780`) because the node-dir suffix carries no h (§0). HEAD pin: `git merge-base
--is-ancestor 081b1f28 HEAD` (row #331 pattern) PLUS a physics-freeze guard `git diff --quiet
081b1f28 HEAD -- darksiren_emri/ hier_s0_driver.py` (STOP on any drift — this is what makes
§3.2's reuse valid at run time, not only at authoring time) `[FLAG: guard added, no physics]`.

### 3.4 What is NOT changed
No production CLI flag, no physics-trigger file, no driver edit. `/physics-change` NOT triggered.
No seed multiplicity (PA-HIER-31(d): CoR-P is "the one banked/observed scattered catalogue").

---

## 4. Registered statistics (fixed before the run)

Per-event ln-likelihoods at each (θ-node n, h): `lnL_i(n; h) = ln combined_no_bh` (primary
channel) and `ln combined_with_bh` (secondary, REPORTED-ONLY, inherits invariant 12), NaN where
non-positive — exactly `read_event_ln_l` (`hier_s0_driver.py:963-1000`). Events are matched on
`event_idx` across all 15 cells; N_common (expected 1588) and the NaN-drop count are disclosed
per statistic.

### 4.1 Secants per h (PA-HIER-31(d), unchanged)
```
score_b_re,i(h) = [ lnL_i(+0.033,1; h) − lnL_i(−0.033,1; h) ] / 0.066
score_lns,i(h)  = [ lnL_i(0,√2; h)   − lnL_i(0,1/√2; h)  ] / ln 2
Z_x(h) = mean_i score_x,i(h) / SEM_i          (per-event SEM; single catalogue ⇒ no seed clustering)
```
Reported at each of the three h (h=0.73 = the rows #336 numbers, re-read).

### 4.2 PA-HIER-33 correction per h (ratified rule, applied as written)
`Es_null^{(h)} = (Δ²/6)·[−3⟨l'ᵢ l''ᵢ⟩ − ⟨l'ᵢ³⟩]`, Δ = ln√2, from that h's own three s-nodes;
`score_s,i(h) = score_lns,i(h) − Es_null^{(h)}`; bootstrap SD of `Es_null` (B = 2000, event
resampling, seed 20260904 `[FLAG: seed value]`) in quadrature. At h=0.73 this reproduces Z −7.101
(row #336) — a g-precision check of the read script, not a fresh number.

### 4.3 The registered derivative — 3-point non-uniform stencil at h₀ = 0.73
With h₋ = 0.665 (Δ₋ = 0.065), h₊ = 0.78 (Δ₊ = 0.050), for x ∈ {b_re, lns}, per event:
```
D_x,i = [ Δ₋² s_i(h₊) − Δ₊² s_i(h₋) + (Δ₊² − Δ₋²) s_i(h₀) ] / ( Δ₊ Δ₋ (Δ₊ + Δ₋) ),   s_i ≡ score_x,i
```
(the Lagrange 3-point first-derivative at the middle node; on a uniform grid this reduces to the
classic central difference `[s(h₊) − s(h₋)]/(2Δ)`). Pooled: `∂score_x/∂h ≡ mean_i D_x,i`, SEM
from the per-event scatter of `D_x,i` (the "per-event secants" of the brief), `Z_D,x =
mean(D_x)/SEM(D_x)`. Secondary, REPORTED-ONLY: the outer-node secant
`[s_i(h₊) − s_i(h₋)]/0.115` and the curvature `C_x,i = 2[Δ₋ s_i(h₊) − (Δ₊+Δ₋) s_i(h₀) + Δ₊ s_i(h₋)] /
(Δ₊ Δ₋ (Δ₊+Δ₋))` (linearity check for §4.4: the band is read only if `|½ C̄_x Δh_θ| ≤ |D̄_x|`,
else the mapping is flagged NON-LINEAR and Δh_θ demoted to report-only — the row #345 D4
convention). Units: nats per unit θ per unit h.

### 4.4 The h-displacement mapping (dossier item 2's object)
```
Δh_θ,x = − S̄_x(0.73) / D̄_x,     S̄_x(0.73) = mean_i score_x,i(0.73)
```
= the h at which the linearised θ-score `S(h) = S(0.73) + D·(h − 0.73)` nulls, minus 0.73 — the
h-equivalent of the θ-pull for the residual split. Band: delta method from the per-event pairs
`(score_x,i(0.73), D_x,i)`:
`Var(Δh) = Δh² · [ Var(S̄)/S̄² + Var(D̄)/D̄² − 2 Cov(S̄,D̄)/(S̄ D̄) ]`, with `Cov` from the paired
per-event sample; cross-checked by the same B = 2000 event bootstrap (percentile 0.135/99.865 %
= the 3σ band). **Report-only rule:** if `|Z_D,x| < 3` the derivative is unresolved, the ratio
is undefined at 3σ (the denominator's 3σ interval contains 0) ⇒ Δh_θ,x is carried as
`|Δh_θ,x| ≥ |S̄_x| / (|D̄_x| + 3·SEM_D)` (a one-sided lower bound) and NO h-bound enters the split.

### 4.5 Secondary reads (REPORTED, not band-bearing)
(i) The curvature leg per h: b̂(h), σ_b(h), ln ŝ(h), σ_ln s(h) (PA-HIER-31(e) formulas) and
their finite-difference slopes ∂b̂/∂h, ∂ln ŝ/∂h. (ii) Per-class D̄ on C-A∪C-B vs C-C at each h
(class = `L_cat_no_bh == 0` at truth AT THAT h; float-fragile per row #337, counts disclosed;
C-C must give D ≡ 0 by the C-C identity). (iii) Per-z-bin D̄ with B3.1's registered edges
{0.075, 0.392, 0.559, 0.659, 0.753, 1.018}, `z_true = dist_to_redshift(d_L, 0.73)` — the read
S0-B's reader disclosed as NOT COMPUTED; attempted here on the same footing. (iv) With-BH channel
for every quantity above.

---

## 5. Bands and dispositions (A8 two-sided; every disposition three-valued; all return as fresh [RULE])

### 5.1 The derivative (per axis x ∈ {b_re, lns}; primary channel no-BH)
| disposition | condition | consequence proposed |
|---|---|---|
| **RESOLVED** | `|Z_D,x| ≥ 3` and all §6 gates green | Δh_θ,x read under §5.2 |
| **NOT-RESOLVED** | `|Z_D,x| < 3` and all §6 gates green | Δh_θ,x report-only as the one-sided bound of §4.4; the θ-pull enters the residual split as "unquantified in h, LIVE in θ" (dossier item 2's own alternative) |
| **INSTRUMENT-DEFECT** | any §6 gate red | nothing banked; returns as fresh RULE naming the gate |
The 3.0 threshold is the repo's registered coherent-displacement threshold (§4.1 "Anchor
derivation for the 3.0 threshold"), not chosen here.

### 5.2 The h-displacement (evaluated only on RESOLVED; three-valued)
| disposition | condition | reading proposed |
|---|---|---|
| **IMMATERIAL-IN-h** | `|Δh_θ,x| + 3·SE < T_mat = 0.008` | the θ-pull cannot own a material part of the −0.064 offset (row #302: iiib mean − 0.73 = −0.0641) |
| **MATERIAL-IN-h** | `|Δh_θ,x| − 3·SE ≥ 0.008` | an h-bound of size Δh_θ,x enters d-residual-attribution item 3, REPORTED-ONLY, sign disclosed |
| **INDETERMINATE** | otherwise, or the §4.3 linearity check fails | banked as a band `[|Δh| − 3SE, |Δh| + 3SE]`, fresh RULE |
`T_mat = 0.008` is the standing T5 materiality threshold (row #302: *"no point reaches
|Δmean_h,pred| ≥ T_mat = 0.008"*; row #345 D4 used the same) — quoted, not chosen `[FLAG: the
only imported threshold]`. Also reported: the fraction `Δh_θ,x / (−0.0641)`, clipped to the
band, and whether `|Δh_θ,x|` exceeds the full offset (saturation, disclosed not band-bearing).

### 5.3 Per-h S0-B reads (re-application of PA-HIER-31(e) at h = 0.665 and 0.78)
B0-B / B0-M / B0-P evaluated at each fresh h with the SAME bands and thresholds (identical
bands "applied to S0-B" per §4.1's anchor note; no new band). REPORTED-ONLY cap carried
(PA-HIER-28 item 9). These are context for the derivative, not a second verdict on LIVE/DEAD:
the S0-B disposition of record stays the h=0.73 one (rows #336/#345).

### 5.4 CONDITIONAL clause (row #345 D3 / docket R4–R5), binding on every read above
The S0-B truth node ≠ the production comparand (562/1588 events differ from `c0prime_off_iiib`;
cause OPEN). The secants and D are INTERNAL to the θ-grid and unaffected; but the READING of
Δh_θ as "the θ-pull's h-equivalent on the production venue" assumes the truth node IS the
production likelihood. **Registered: every §5.1/§5.2 disposition is booked CONDITIONAL-ON-R4
until the R4 comparand (job 6790708) is read.** If R4 shows the θ-sites-2.2 / mass-aware-off
production evaluation reproduces the S0-B truth node to GATE T-ID precision (≤ 1e-12 relative,
PA-HIER-31(j)), the condition lifts; if not, the dispositions stand as measurements on the
S0-B instrument only and the h-bound does NOT enter the split (fresh RULE). Launching this arm
before R4 is read is a cost decision (the arm is cheap and the reads are blind either way), not
a scientific one — flagged, consistent with docket R5's "defer behind R4" applying to the
INTERPRETATION.

---

## 6. Gates (scored before any §5 band is read; red ⇒ INSTRUMENT-DEFECT)

| gate | registered form | basis |
|---|---|---|
| **g-score-null (per h)** | `|Z_b_re(h)| ≤ 3` and `|Z_lns(h)| ≤ 3` REPORTED per h; on production this is the measurement, not a control (PA-HIER-31(h); dossier item 1, unruled) — the panel consequence text (*"red score-null → STOP d-photoz-leverage, reopen the instrument question as a fresh RULE"*) is quoted, not applied; the instrument certification of record remains row #287 (mirror, both axes `|Z| ≤ 3` at T1.3) | Graph 1 panel line 240; row #287 |
| **g-znorm (per h)** | (i) the θ-inert global tables `{beta_G_phi, beta_Gbar_phi, sigma_phi, sigma_4d, r_Malm}` in `selection_tables_h_<h>.json` must be BIT-IDENTICAL across the five θ-nodes at each h (site 2.3 is out of scope, PA-HIER-31(b): under `theta_sites=2.2, smear off` the global selection has no θ referent). Verified at zero compute on the banked h=0.73 set: all five `selection_tables_h_0_73.json` md5 `e68ab957…` identical [LOCAL, 2026-09-04]. (ii) any explicit normalisation-residual field, if present: `abs dev ≤ 1e-6` green / `> 1e-3` anomalous (panel line 239); ABSENT otherwise (three-valued, as the S0-B reader found) | panel line 239; PA-HIER-31(b) |
| **C-C identity (per h)** | for every event with `L_cat_no_bh == 0` at truth at that h, `combined_no_bh` bit-identical across all five θ-nodes (`max|Δ| = 0`); across h the C-C set may change (class defined per h, disclosed) | PA-HIER-31(b)/(d); passed at h=0.73 with n=449 |
| **C-C across nodes = C-C across h?** | NOT a gate: `combined_no_bh` at a C-C event legitimately varies with h (the population/completion legs are h-dependent) — reported so nobody reads it as a defect | — |
| **GATE ENG (per h, per off-truth node)** | ≥ 10 % of events move ≥ 1e-6 relative vs truth (driver's `ENG_REL_THRESHOLD`/`ENG_EVENT_FRACTION`), computed by hand on the `_re` names as the S0-B reader did (the driver's own `gate_eng` looks for `b_plus`/`b_minus`) | prereg §3.4; row #336 |
| **g-precision** | every decisive number re-derived from the raw `event_likelihoods.csv` (never the driver's cached JSON); the driver's `--score-only` on each per-h out-root must reproduce the hand secants to the digit (the row #336 pattern) | panel line 246 |
| **pins** | `provenance_*.json` per task: commit contains `081b1f28`, `darksiren_emri/` + driver diff-quiet vs `081b1f28`; `build_iiib_venue`'s CRB/catalogue md5 gates | CLAUDE.md pinning rule |
| **g-population** | N_common = 1588 across all 15 cells; NaN drops per cell disclosed (0 expected) | panel line 244 |

---

## 7. Invariants and structural blindness (A10)

Invariants: prereg §5.1 items 1–5, 7, 9–13 carried verbatim (per PA-HIER-31(g)), plus the
PA-HIER-31 additions (`smear_global_selection = False`, `theta_sites = "2.2"`,
`catalogue_leg_1d_mass_aware = off`, `b_half_width = 0.033`), plus this addendum's: the
driver/package blob unchanged since `081b1f28` (audited 2026-09-04, enforced at run time §3.3);
`--cpus-per-task = 16` (worker count 14) identical to 6779532; `H_BOUNDS = (0.50, 0.86)`; the
h=0.73 cells reused, never recomputed.

Structural blindness (carried from PA-HIER-31(g) (a)–(c),(e), with (d) "single h" now REMOVED
and replaced by): (d′) three h-nodes only — any non-quadratic h-dependence of the score between
0.665 and 0.78 is invisible; the linearity check of §4.3 catches curvature at the node scale,
not oscillation inside it. (f) The OPEN truth-node ≠ comparand fact (row #345 D3): if the S0-B
instrument's h=0.73 truth node carries an unexplained +2.58-nat offset from production, the SAME
offset may be h-dependent — this design measures the instrument's own ∂score/∂h and cannot
separate "θ-pull physics varies with h" from "the instrument's deviation from production varies
with h" without R4 (§5.4). (g) Single catalogue, no seed scatter — the SEM is per-event only.

---

## 8. Falsifiers (A14)

- *"∂score/∂h is a catalogue-leg/d_L-mapping object (B4.3 class)"* is FALSIFIED if
  `NOT-RESOLVED` on both axes with `SEM_D` small enough that `3·SEM_D < |S̄(0.73)|/0.115` (i.e. the
  score would have to change by more than its own size across the grid to be resolved, and it
  does not).
- *"the θ-pull is h-inert (pure kernel object)"* is FALSIFIED by `RESOLVED` with
  `MATERIAL-IN-h` on either axis.
- The C-C identity failing at any fresh h FALSIFIES the instrument at that h (θ reaching a
  dark-class event) — INSTRUMENT-DEFECT, never a discovery.
- The §4.3 linearity check failing while `RESOLVED` falsifies the linear mapping of §4.4 (not
  the derivative); Δh_θ then demotes to INDETERMINATE by rule.

---

## 9. Cost (both bases, per row #336 item 6) and cap

Anchor [DOC]: job 6779532, 5 tasks, sacct Elapsed 7:21–7:36 (`READOUT_RECORD.md` §8), 16
cores/task; driver-internal `wall_s ≈ 338 s`/node (row #332).

| basis | per (h, node) cell | 10 fresh cells | 15 cells (design) |
|---|---|---|---|
| sbatch allocation (elapsed × 16 cores) | 2.0 CPU-h | **20.0 CPU-h** | 30 |
| row #332 basis (elapsed × cores actually used; row #332's "≈2 CPU-h" for 5 cells) | ≈ 0.4 CPU-h | **≈ 4 CPU-h** | ≈ 6 (= the dossier's "≈ 6 CPU-h") |

**Cap: 10 CPU-h ORCHESTRATOR-DERIVED** (brief; row #349 item 3 "≈6 CPU-h" is the row #332
basis). Derivation: 3 h × 5 nodes × 0.4 CPU-h = 6 CPU-h, ×1.7 headroom ≈ 10. **On the row #332
basis the fresh compute (≈4 CPU-h) is inside the cap; on the allocation basis (20 CPU-h) it is
2× OVER.** Not reconciled here — the two bases are both true (row #336 item 6); which one the cap
binds is put to the chair BEFORE submission (§12). The reservation is NOT reduced below 16
cores to fit the allocation basis: that would change the worker count and break §3.2's
byte-comparability with the h=0.73 set. Wall: 10 tasks × ≤ 8 min, `--time=00:45:00` (≈ 5×
margin, the head-rebaseline convention), array runs in one wave ≈ 10 min if all start together.
Workspace expiry 2026-09-23 (19 days at preflight): no risk.

---

## 10. Launch block (the ops agent submits; nothing here was run)

**sbatch:** `cluster/graph1_s0c_hgrid.sbatch` (written with this addendum; modelled on
`cluster/graph1_m_s0b_production.sbatch` — CLI verbatim + `--h-nodes` — and on
`cluster/graph1_headrebaseline_iiib.sbatch` for the `$WORKSPACE` out-root and pin pattern).
Array `0-9`: `TID → h = H_LIST[TID / 5]`, `node = NODES[TID % 5]`, `H_LIST=(0.665 0.780)`,
`NODES=(truth b_plus_re b_minus_re s_plus s_minus)`. Out-root
`$WS/graph1_s0c_hgrid_20260904/h_0p665` and `…/h_0p780` (`$WS =
/pfs/work9/workspace/scratch/st_ac147838-emri`).

Sequence for the ops agent (mechanical):
```
ssh bwunicluster 'bash -s' < cluster/preflight.sh        # require VERDICT: READY ✓
ssh bwunicluster 'cd ~/darksiren-emri && git pull --ff-only && git rev-parse --short=8 HEAD'
rsync -avz cluster/graph1_s0c_hgrid.sbatch bwunicluster:darksiren-emri/cluster/   # if not yet committed
ssh bwunicluster 'test ! -e $(ws_find emri)/graph1_s0c_hgrid_20260904 && echo OUT-ROOT-ABSENT'
ssh bwunicluster 'cd ~/darksiren-emri && source cluster/modules.sh && sbatch cluster/graph1_s0c_hgrid.sbatch'
```
Retrieval (row #311 lesson, `--exclude='**/simulations/injections'` — the node dirs symlink the
shared pool): `rsync -aL --exclude='**/simulations/injections' bwunicluster:$WS/graph1_s0c_hgrid_20260904/
results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/s0c_hgrid_20260904/` +
md5 manifest.

Zero-compute scoring per h (the driver's own registered entry point, reproduces the §4.1 secants
and the PA-HIER-33 correction):
```
python results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py \
  --arm S0-A --score-only --config iiib --theta-sites 2.2 --smear off --seeds 900101 \
  --b-half-width 0.033 --nodes truth,b_plus_re,b_minus_re,s_plus,s_minus \
  --h-nodes 0.665 --out-root <retrieved>/s0c_hgrid_20260904/h_0p665        # and 0.780
```
The derivative (§4.3), the mapping (§4.4) and the gates (§6) are computed by a read script the
read node writes under `exec/r-s0c-hgrid/` from the RAW `event_likelihoods.csv` of all 15 cells
(the 5 h=0.73 cells from `retrieved/s0b_run_20260902/`), never from cached JSON. The read node
is a fresh reader (sonnet) + chair re-derivation of every decisive number (memory rule:
verifier output is evidence, not authority). No reader runs the registered measurement itself.

---

## 11. Blindness and ordering (A22)

Bands (§5), thresholds, the stencil (§4.3), the mapping (§4.4) and the gates (§6) are fixed in
this file BEFORE any h ≠ 0.73 cell exists. The h=0.73 inputs are disclosed as seen. The fresh
cells are unread until the read node runs. No band may be edited after the sbatch is submitted;
any revision is a new PA-HIER number. Revision cap on this un-run registration: 1 (design), 0
after launch.

---

## 12. Open items for the chair/author (none decided here)

1. **[chair, before sbatch]** Which cost basis the 10 CPU-h cap binds (§9): allocation (20 CPU-h,
   over) vs row #332 (≈4 CPU-h, inside). If allocation: the arm needs a cap of 20 CPU-h or a
   ruling that the h=0.78 half runs first (5 cells, 10 CPU-h allocation) — the latter would
   destroy the 3-point stencil (§4.3) and is NOT recommended.
2. **[RULE]** §5.4 CONDITIONAL-ON-R4: whether the dispositions may be READ (not computed) before
   R4 lands — docket R5 provisional reading says defer the interpretation; computing is free.
3. **[RULE]** Dossier item 1 (charter clause vs PA-HIER-31(h)) still unruled; §6's g-score-null
   row applies the same chair reading (row #336 (4)), flagged.
4. **[RULE]** Ratify PA-HIER-34 as written (append to the prereg by ledger row) — the bands of
   §5.1/§5.2 are the fresh registration content; `T_mat = 0.008` and the bootstrap seed are the
   two imported/flagged constants.

*Stamp: prereg author B, 2026-09-04. Read-only except this file and `cluster/graph1_s0c_hgrid.sbatch`;
no pipeline run, no cluster command, no edit under `darksiren_emri/`, no edit to the prereg.*
