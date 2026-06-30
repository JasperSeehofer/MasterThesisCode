# Bridge prototype spec — testing the photo-z in-catalogue fixes

All prototyping is in `scripts/bridge_closure/_bridge_sky.py` (no production edits). Validation
harness: `scripts/bridge_closure/rung_G_photoz.py` (writes `outputs/rungG_photoz.pdf`,
`outputs/rungG_results.json`). All edits live in the `mode == "conv"` branch, lines ~240-252.

**Global acceptance criterion (every candidate):** at `zerr_scale = 1.0` (sigma_z = 0.035) the MAP
must move from the current **0.857** to **~0.72-0.74 (non-railed)**, while the delta-z leg
(`zerr_scale = 0` / `mode="mvn"`) stays at **~0.725**. A fix that regresses delta-z is rejected
outright.

---

## Candidates ranked by likelihood of success (per the three verdicts)

| Rank | Candidate | Verdict votes | Predicted MAP @ sigma_z=0.035 |
|---|---|---|---|
| 1 | **EXP-1** psi-regularised posterior kernel (Change 1) | C: yes; B: uncertain; A: no-alone | ~0.72-0.74 **if** degraded-regime dominates; possibly 0.78-0.86 if delta-sharp truth dominates |
| 2 | **EXP-0** D(h)-power falsification (negative control) | B: firm negative | stays railed for any sane P; needs P~1000 |
| 3 | **EXP-2** re-injection with genuine photo-z scatter (sim<->inference consistency) | A: this is the real fix | ~0.73 (clean), the decisive disambiguator |
| 4 | **EXP-3** EXP-1 + convolved global selection (Change 2) | proven no-op | ~same as EXP-1 |

Run order: **EXP-0 first** (cheap, settles the D(h) red herring), then **EXP-1**, then **EXP-2**
(decides between the two honest possibilities), then EXP-3 only if needed.

---

## EXP-0 — D(h)-power falsification (negative control, run FIRST)

Confirms Angle B's impossibility bound numerically so nobody chases D(h).

In `run_sky_rung`, after `D_tab` is built (~line 325), wrap with a tunable power and sweep:
```python
P_TEST = 0.0   # sweep {0, 3, 10, 100, 1000}
D_h_eff = D_tab[h] * (h / 0.73) ** P_TEST
```
**Predicted outcome:** MAP stays railed (>= 0.85) for `P_TEST` in {0, 3, 10}; only `P_TEST ~ 1000`
(absurd) reaches 0.73. **Pass = confirms D(h) is not the lever.** Then remove the wrapper.

---

## EXP-1 — psi-regularised posterior kernel (PRIMARY)

Replace the bare kernel at `_bridge_sky.py:246-248`. Add at the top of the conv branch:
```python
from master_thesis_code.physical_relations import comoving_volume_element
```
After `zgrid` is built (~line 241):
```python
p_bg = np.asarray(comoving_volume_element(zgrid, h=h), float) / (1.0 + zgrid)   # (ngrid,)
```
Replace lines 246-249 with the regularised, per-host-normalised posterior:
```python
nm = np.exp(-0.5 * ((zgrid[None, :] - zg[:, None]) / szg[:, None]) ** 2) / (
    np.sqrt(2 * np.pi) * szg[:, None]
)                                            # norm(z; z_g, sigma_z), (ncand, ngrid)
nm = nm * p_bg[None, :]                       # multiply comoving-volume prior  -> norm * p_bg
Z_g = (nm @ (np.ones_like(zgrid))) * dzg      # per-host normalisation Z_g, (ncand,)
nm = nm / np.maximum(Z_g, 1e-300)[:, None]    # p_red(z|z_g), unit integral per host
N_dL = nm @ (gw * dzg)                         # unchanged downstream
```
Gate it behind a flag `regularise_photoz: bool` so the sweep can run both ways.

**Numerical guards (mandatory before trusting any result):**
1. Verify per-host area: `np.allclose((nm @ np.ones_like(zgrid)) * dzg, 1.0, atol=1e-3)`.
2. Confirm `conv` at `zerr_scale -> 0` reproduces `mode="mvn"` (0.725). If conv at tiny sigma_z lands
   ~0.600 instead, there is a conv-branch bug (sky-factorisation / grid `sz_res`,`ngrid` clip
   [200,800], lines 239-240) the density fix will NOT cure — fix that first.
3. Re-run with `ngrid` doubled to confirm grid convergence (rules out the x0.05 -> 0.600 undershoot
   being a grid artefact).

**Sweep** `zerr_scale in {0, 0.05, 0.25, 0.5, 1.0}` via `rung_G_photoz.py`.

**Predicted outcome (honest, contested):**
- delta-z (`zerr_scale=0`): stays ~0.725 (kernel -> delta). **Must hold.**
- `zerr_scale=1.0` (sigma_z=0.035): **de-rails downward from 0.857.** Lands ~0.72-0.74 if the
  degraded-catalogue / MFG limit dominates (Angle C). Risk: lands interior-but-biased (0.78-0.86) if
  the delta-sharp injected truth dominates (Angle A) — because the events have NO real photo-z
  scatter, so a 0.035-wide prior over-smears a sharp truth.
- Intermediate points may NOT fully flatten (the fix is a large-sigma_z correction; at sigma_z~0.002
  it is ~a no-op, so the x0.05 anomaly likely persists — that is expected, not a failure of EXP-1).

**Pass/fail:** PASS if `zerr_scale=1.0` reaches ~0.72-0.74 AND delta-z holds ~0.725. If it de-rails
but only to ~0.78-0.82, that is a *partial* result -> proceed to EXP-2 to determine whether the
residual is the sim<->inference inconsistency.

Also report whether recovery is **information-driven or prior-flattening**: compare the posterior
width to the prior width. A much-broadened posterior centred near the prior median (~0.735 on
[0.60, 0.87]) means recovery-by-prior (acceptable for de-railing, expected at sigma_z/z ~ 0.7).

---

## EXP-2 — re-injection with genuine photo-z scatter (the disambiguator)

This is the decisive experiment when EXP-1 only partially de-rails. It tests whether the residual
bias is the sim<->inference inconsistency (events injected at the EXACT catalogue z, not scattered).

In `_bridge_lib.py:real_events_from_crb` (line ~353), the host redshift is currently the GW-true
redshift `z_host = dist_to_redshift_vec(d_true, TRUE_H)`. Add a photo-z-consistent injection mode:
```python
# consistent-injection leg: give each host a genuine photo-z offset
z_obs = z_true + rng.normal(0.0, sigma_z_per_host)     # sigma_z_per_host = real GLADE zerr
# store z_obs as the catalogue/host label; keep z_true as the (hidden) GW truth
```
Then run EXP-1's regularised kernel against this re-injected set.

**Predicted outcome:** with genuine photo-z scatter, the regularised same-kernel scheme should
recover **~0.73 cleanly** at sigma_z=0.035 (this is the literature's actual demonstration regime —
arXiv:2502.17747, Echoes 2509.18243: photometric catalogues unbiased, variance-only).

**Interpretation:**
- EXP-1 recovers on re-injected set but only partially on as-injected set
  -> residual is the **sim<->inference inconsistency**, NOT a likelihood error. Production fix =
  regularised kernel + consistent host-z injection.
- EXP-1 recovers on both -> the kernel fix alone suffices; the as-injected delta-sharp truth is
  forgiving enough at sigma_z=0.035.

---

## EXP-3 — convolved global selection (consistency leg, expected NO-OP)

Replace the `gcat` global denominator (`_bridge_sky.py:329-330`) with a per-host convolved
`D_g^conv = INT P_det(d_L(z,h)) p_red(z|z_g) dz` on the same `zgrid`. **Predicted:** MAP essentially
unchanged from EXP-1 (P_det broad; Option A beta_G cancellation). Run only to confirm the no-op and
attribute recovery to EXP-1's numerator change. Do not port to production if confirmed no-op.

---

## Summary of what each experiment decides

- **EXP-0:** kills the D(h) red herring (firm negative).
- **EXP-1:** tests the unanimous physically-correct fix; de-rails downward; magnitude contested.
- **EXP-2:** disambiguates "kernel fix sufficient" vs "needs consistent injection" — the crux.
- **EXP-3:** confirms the selection-denominator asymmetry is a no-op.
