# Independent step-back gate review — 2026-08-19 — VERDICT: GATE-HOLD

**Provenance:** fresh-context top-tier agent, author-mandated (row #129 item 2: judge
rabbit-hole vs principled; re-derive the 2D likelihood from principles). **Orchestrator
verification:** the agent's B_scale counterfactual script
(`bscale_counterfactual_exploratory.py`, promoted from scratchpad) was independently re-run
this session — it reproduces the banked posteriors exactly (2D 0.7842/0.7967, 1D
0.6040/0.6074) and yields B_scale ∈ [0.6503, 0.6765], d ln B_scale/dh = +0.1635, frozen@0.73
→ 0.6648/0.6596, removed(≡1) → 0.6771/0.6788. Report below is the agent's, verbatim in
substance (formatting normalized).

## 1. Rabbit-hole judgment

Not a classic rabbit hole — the elimination instruments are exemplary (rows #91, #96, #114,
#116, #119, #128: pre-registered, prediction-first, reproducing) and the physics fixes of
record are derivation-backed. But the current front asks an ill-posed question, and one
un-derived object has sat inside the production formula since 2026-08-04, unnamed by any
candidate list. Specific lapses:

1. **Row #92's discrepancy recorded, not chased:** the closed-loop harness reproduced only
   +0.011 of the +0.077 production displacement; correct inference: production contains
   structure the harness does not model. It does — the harness has NO analog of the
   production `B_scale` factor (verified; the object exists only at
   `bayesian_statistics.py:4904-4906`).
2. **Row #123's "no live unexplained production H0 bias remains" was contradicted by
   #127-#128** — venue closures were read as production closures in the summary sentence
   while the A3 small print said otherwise.
3. **Row #111 item-2 authorization (transfer the venue-validated correct-form 1D estimator
   into production) silently LAPSED.** Production 1D still rails at 0.600-0.604; calgate-v2
   DS-6 (row #98) failed to reproduce the production 1D rail in the multi-candidate venue —
   "1D starves" is NOT yet established as pure information starvation in production.
4. **The elimination frame is structurally unable to converge:** the production 2D posterior
   at 0.784/0.797 is the balance point of several opposing tilts each of magnitude
   ~0.12-0.16 in h (frozen-g → 0.66/0.64; frozen-B_scale → 0.665/0.660; removed-B_scale →
   0.677/0.679; the shared base tilt rails 1D at 0.600 in every scenario). Hunting "the
   owner of +0.06" one component at a time cannot work; the frame must become "close the
   tilt ledger" — each component's status (correct physics / defect / underived) assigned by
   derivation, jointly.

## 2. First-principles re-derivation

The MFG two-class single-event form: p_i^2D = [ (1/n̂_w^φ)Σ_g w_g N_g^3D mz_g + ∫(1−f_k)
p_gw dVc/(1+z) g_sel dz ] / D̃^φ(h) — one detection model everywhere, NO further factor on
either leg. The implemented catalogue leg matches exactly (α_G^φ·L_cat reduces to
Σw_gN_g·mz/n̂_w^φ; mz_integral, x_M convention, g_i measure pairing all check out). The dark
leg integrand matches the boxed form — EXCEPT:

### The structural mismatch: B_scale (bayesian_statistics.py:4904-4906)

    B_scale = beta_Gbar_phi / beta_Gbar
    B_num_phi     = B_num     * B_scale
    B_num_wbh_phi = B_num_wbh * B_scale

`beta_Gbar` is built on the separately-fitted mass-blind S_3D (sky-banded 1−f_k);
`beta_Gbar_phi` on the φ-contracted S̄_φ (isotropic f̄). The fixb_pathA doc §2 justifies
the multiplication as a "convention transfer" on the claim that B_num carries the legacy
β_Ḡ's normalisation. **It does not**: B_num is the direct physical integral, already in β's
units, already addable to the catalogue leg, and post-fusion already in the φ detection-model
convention. Every consistent MFG assembly gives **B_scale ≡ 1**. The shipped ratio of two
different detection models inside one term is the MFG-A2 violation the path-A package was
written to eliminate — removed from the catalogue leg, re-installed in the completion leg.
Dimensional analysis cannot see it (dimensionless), like the Phase-14 /(1+z) bug.

**Measured:** B_scale 0.6503 → 0.6765 over h ∈ [0.60, 0.86] (d ln/dh ≈ +0.16), a coherent
high-h multiplier on the leg carrying ~93-95% of the mixture weight across 1588 events.
Banked-data counterfactual (reproduces production bit-for-bit): the factor's h-slope moves
the 2D mean by **+0.119/+0.137** — 2× the entire +0.054/+0.067 offset. Every prior
instrument held B_scale fixed across its arms (fused-vs-off, V1′, frozen-g), and the harness
never implemented it — which is how a completion-leg common-mode factor survived four
elimination campaigns. NOTE: removing B_scale alone lands 0.677/0.679 — ~0.05 BELOW truth —
re-exposing the base tilt; the correct form must be derived, then the residual re-budgeted.

### Other mandated checks

- (a) φ vs injected population: CLEAN (φ imported from the generator; g's h-slope is genuine
  information; L6 rulings defensible in-venue).
- (c) Numerator/denominator consistency: the documented D3/F10 point-vs-kernel Σᶲ/Σ⁴ᴰ
  inconsistency stands, common-mode via D̃^φ, h-slope never measured — second tier.
- (d) The 2D offset is real (z = 4.75/5.53, jackknife-robust), but its interpretation is
  entangled with the un-transferred 1D correct form; P7-1's gating of the "1D starves"
  headline must stay in force.
- (e) Eddington-in-M, fusion lever, catalogue-overlap eliminations: sound as recorded.

## Candidate ranking (by derivational motivation)

1. **B_scale** — no derivation exists; MFG says ≡1; slope +0.16/h; moves 2D mean by
   +0.12/+0.14; absent from every harness/instrument arm.
2. **Un-transferred correct-form 1D base** (lapsed row #111 authorization).
3. **Selection point-vs-kernel J_α** (D3/F10) — documented, adverse-direction, unmeasured.
4. **g_i geometry** — already substantially exonerated as correct physics; deforming it
   before 1-3 are resolved yields another confounded elimination.
5. Catalogue-leg mass overlap — done (V1′).

## 3. GATE verdict: GATE-HOLD (short hold, specific exit)

The g_i battery must NOT run first: any g_i read on top of live B_scale is confounded by a
+0.12-class tilt of unestablished status, and the decomposition frame must change to
closing the tilt ledger. Exit:

- **(i) B_scale derivation (days, not weeks):** half a page of MFG algebra — derive why the
  dark-class numerator must carry β_Ḡ^φ(h)/β_Ḡ(h) (claim: it cannot), or declare it a
  defect and route via /physics-change. The instrument effectively exists (zero CPU,
  banked-data read).
- **(ii) Re-present the lapsed row-#111 production transfer** as a fresh [DO] (or an
  explicit author [RULE] that production knowingly carries the venue-validated 1D
  correct-form terms as a documented systematic). Arc: one derivation session + one
  ~15 CPU-h validation run on the rows #109-#116 pattern.
- **(iii) Only then** the g_i battery and the (gated) landscape — measuring a
  derivation-complete estimator, per the binding value.

**Files of record:** bayesian_statistics.py:4904-4906, :4907-4914, :2022/:2155, :1964,
:1263, :5636; docs/derivations/fixb_pathA_phi_marginal_selection.md §2;
run_20260804_postfix/*/diagnostics/event_likelihoods.csv;
run_20260817_fusion_counterfactual/off_iiib/*.log; bscale_counterfactual_exploratory.py.
