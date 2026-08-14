# mechanism_study_20260813/toys — provenance note

## m5_toy.py

Recovered 2026-08-14 from a session scratchpad by the commission. The original was never
committed to git during the 2026-08-13 mechanism-isolation session, so it existed only as a
temp-directory artifact (`/tmp/claude-1000/.../739b23e2-.../scratchpad/m5_toy.py`) until this
recovery. Byte-identity to the original 2026-08-13 file **cannot be proven** — there is no
committed hash or copy from that date to diff against.

What can be shown instead: the recovered file **reproduces the registered K=50 impostor-only
value**, `+0.02468`, when driven with the impostor-only MEI protocol (all impostors z-scattered,
host held exact). This numerical match is the evidence standard used here in place of byte
provenance.

Two independent re-executions on top of the recovered `m5_toy.py`:

- **Commission re-execution**: K=84 → `+0.0279`, K=1216 → `+0.0341`
- **Chair's independent driver** (`run_m5_mei_chair_20260814.py`, a *protocol variant* that
  kernels the exact host with the same Gaussian z-kernel used for impostors, rather than
  point-evaluating it): K=50 → `+0.0317 ± 0.0007`, K=1216 → `+0.0339 ± 0.0006`

The two drivers diverge at low K (0.0279 vs 0.0317 at comparable K) because they differ in how
the true host is scored, not because either toy is broken — this is the documented
point-evaluate-host vs. kernel-host protocol difference, not a reproduction failure.

**Status: the toy is ruled UNFAITHFUL at production K** (ledger row #102 — see the mechanism
study's decision ledger). It tracks the intended sigma_z-dosed MAP-displacement mechanism only
qualitatively; it does not reproduce the production bias magnitude at the K~1000+ scale actually
used in the venue-transfer runs. It is committed here **for auditability of the recovery and the
re-execution record, not for further closure use** — do not cite its numbers as a production
estimate.

## run_m5_mei_chair_20260814.py

The chair's independent MEI (impostor-only) driver against `m5_toy.py`, copied in alongside it
from the same 2026-08-14 recovery session. Its `sys.path.insert` line originally pointed at the
`/tmp` scratchpad path it was written against; it has been repointed to import `m5_toy` from this
same `toys/` directory (`os.path.dirname(os.path.abspath(__file__))`) so it runs standalone from
a checkout with no `/tmp` dependency.

## m3_toy.py

Recovery status: **found and copied**, unlike the initial commission report which flagged it as
possibly lost. It was located in the same 2026-08-13 session scratchpad as the original
`m5_toy.py`
(`/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/739b23e2-e917-49e1-9ab6-05c0f8c53025/scratchpad/m3_toy.py`),
alongside a companion `m3_dose.py` (not copied — not requested). Search also covered
`/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/44bc7188-.../scratchpad/` (found only
`m3_ab_toy.py` and `m3_shoulder_stress.py`, later/derivative toys, not the original) and
`/tmp/claude-1000/-home-jasper-Repositories-MasterThesisCode*` (directory exists but has no
scratchpad subdirectory / no `m3_toy*` match). Since the original `m3_toy.py` was recovered
directly (not reconstructed), the fallback note about M3's closure standing only on its analytic
core does not apply here — the committed file is the actual toy, not a stand-in.

## Files in this directory

- `m5_toy.py` — the M5 minimal faithful-mirror toy (sigma_z-dosed MAP displacement estimator)
- `run_m5_mei_chair_20260814.py` — chair's independent MEI driver against `m5_toy.py`
- `m3_toy.py` — the M3 toy (h-dependent truncation of the unrenormalised z-kernel isolation test)
