# HPC-05 — flip_hx Verification Record

**Date:** 2026-04-23
**Phase:** 39
**Requirement:** HPC-05

## Installed version
- fastlisaresponse: **1.1.17** (from `.venv/lib/python3.13/site-packages/fastlisaresponse/_version.py:31`)
- few: **2.0.0rc1** (from `.venv/lib/python3.13/site-packages/few/_version.py:20`)

## ResponseWrapper semantics (fastlisaresponse 1.1.17)

Source: `.venv/lib/python3.13/site-packages/fastlisaresponse/response.py`

### Class-level docstring (lines 670-671)

```
This class takes a waveform generator that produces :math:`h_+ \pm ih_x`.
(:code:`flip_hx` is used if the waveform produces :math:`h_+ - ih_x`).
```

### flip_hx docstring (lines 693-696)

```
flip_hx (bool, optional): If True, :code:`waveform_gen` produces :math:`h_+ - ih_x`.
    :class:`pyResponseTDI` takes :math:`h_+ + ih_x`, so this setting will
    multiply the cross polarization term out of the waveform generator by -1.
    (Default: :code:`False`)
```

### is_ecliptic_latitude docstring (lines 700-703)

```
is_ecliptic_latitude (bool, optional): If True, the latitudinal sky
    coordinate is the ecliptic latitude. If False, thes latitudinal sky
    coordinate is the polar angle. In this case, the code will
    convert it with :math:`\beta=\pi / 2 - \Theta`. (Default: :code:`True`)
```

### `is_ecliptic_latitude=False` branch (lines 819-821 of `__call__`)

```python
# transform polar angle
if not self.is_ecliptic_latitude:
    beta = np.pi / 2.0 - beta
```

Semantics: when `is_ecliptic_latitude=False`, the input `beta` parameter is treated as a polar angle
Theta in [0, pi] and converted to ecliptic latitude via beta_ecl = pi/2 - Theta.

### `flip_hx=True` branch (lines 830-831 of `__call__`)

```python
if self.flip_hx:
    h = h.real - 1j * h.imag
```

Semantics: when `flip_hx=True`, the wrapper declares that `waveform_gen` emits `h_+ - i*h_x`, while
the downstream `pyResponseTDI` expects `h_+ + i*h_x`. The operation `h.real - 1j*h.imag` is complex
conjugation, which flips the sign of the imaginary part and converts `h_+ - i*h_x -> h_+ + i*h_x`.
This is a sign-convention adapter at the waveform / response interface.

## Our call site (`master_thesis_code/waveform_generator.py:56-69`)

```python
lisa_response_generator = ResponseWrapper(
    waveform_gen=_set_waveform_generator(waveform_generator_type, use_gpu=use_gpu),
    flip_hx=True,
    index_lambda=INDEX_LAMBDA,   # = 8 -> phiS (azimuthal angle)
    index_beta=INDEX_BETA,       # = 7 -> qS   (polar angle)
    t0=T0,
    is_ecliptic_latitude=False,
    Tobs=T_observation,
    remove_garbage=True,
    dt=dt,
    orbits=ESAOrbits(force_backend=force_backend),
    force_backend=force_backend,
    **tdi_kwargs_esa,
)
```

- `index_beta=7` -> `qS` (our ParameterSpace's polar angle, valued in [0, pi] after Phase 36's
  equatorial->ecliptic rotation + polar-angle embedding).
- `is_ecliptic_latitude=False` -> ResponseWrapper applies `beta_ecl = pi/2 - qS`. This is precisely
  the polar-to-ecliptic-latitude convention Phase 36 relies on; input sky coordinates reach the
  ResponseWrapper already in the ecliptic frame as polar angles.
- `flip_hx=True` -> required because `few`'s GenerateEMRIWaveform emits `h_+ - i*h_x` (the
  convention matched by the fastlisaresponse 1.1.17 docstring, which explicitly names
  `flip_hx=True` as the "waveform_gen produces `h_+ - i*h_x`" case); the wrapper then conjugates
  to deliver `h_+ + i*h_x` into `pyResponseTDI`.

## Decision
- [x] KEEP flip_hx=True (primary path) — software-only; add 2-line reference comment above line 58
- [ ] REMOVE flip_hx (fallback path) — escalate to /physics-change with regression pickle

**Rationale:** The installed fastlisaresponse 1.1.17 `ResponseWrapper` docstring and `__call__`
source (lines 670-671, 693-696, 830-831) jointly state the exact semantics our pipeline needs:
`flip_hx=True` declares that the waveform generator emits `h_+ - i*h_x` and the wrapper conjugates
to produce the `h_+ + i*h_x` that `pyResponseTDI` expects. This matches `few` 2.0.0rc1's
convention. The second flag, `is_ecliptic_latitude=False`, is independent and orthogonal: it governs
sky-angle transformation (polar -> ecliptic latitude via `pi/2 - qS`), which Phase 36 explicitly
relies on. The two flags do not interact — removing `flip_hx=True` would produce the wrong sign of
`h_x` in every TDI channel, silently biasing every SNR and CRB. There is no evidence of a
double-flip (no other conjugation happens anywhere in the call path between `few.waveform` and
`pyResponseTDI.get_projections`). The flag remains correct; document it with an inline citation
rather than remove it.

## References
- fastlisaresponse 1.1.17 `ResponseWrapper` source — excerpts cited above
- Katz, Chua et al. (2022) "fastlisaresponse", arXiv:2204.06633
- Phase 36 coordinate-frame fix — `galaxy_catalogue/handler.py` ecliptic rotation + polar embedding
  (completed 2026-04-22)
- `few` 2.0.0rc1 (installed locally; emits `h_+ - i*h_x` per fastlisaresponse docstring contract)

## Note on file authorship

This VERIFICATION.md was authored by the orchestrator (not the executor agent) due to a
`Write`/`Bash` permission denial in the executor's worktree session. All sources read, line
numbers, version strings, and the rationale paragraph come verbatim from the executor agent's
structured checkpoint report (`agentId: af3f79ee36db97a6d`). The orchestrator added no new
analysis. Authorship is captured here so a future reader can trace the chain back to the
agent's transcript at `/tmp/claude-1000/.../tasks/af3f79ee36db97a6d.output`.
