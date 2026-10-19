"""[MKER] W-2 re-measurement: linear-failure side split at the reproducible fleet scope.

Executed under the author grant of 2026-08-28 ("please go ahead" on the W-1/W-2
escalation, ledger row #217 follow-up) and the prereg's §7 clause 2 census
re-open. Reuses the frozen wgeom_instrument (sha1 17dbccbac7eb) verbatim:
same pin gate, same fleet loader, same window recomputation. Classifies every
linear-failing candidate row as too-LIGHT (candidate window entirely below the
GW floor: lin_hi < gw_lo) or too-HEAVY (lin_lo > gw_hi) — the corrected-scope
counterpart of CLAIM_WGEO §3.9's scope-impeached "29:1" split. Also splits the
log-failure side for the heavy-cut-reintroduction claim.
"""

import json
from pathlib import Path

import numpy as np

from wgeom_instrument import (
    K_SIGMA,
    OUT_DIR,
    load_fleet,
    load_pruned_catalogue,
    verify_catalogue_pin,
    verify_fleet_row_counts,
)


def main() -> None:
    verify_catalogue_pin()
    verify_fleet_row_counts()
    cat = load_pruned_catalogue()
    events = load_fleet()

    n_all = n_lin_fail = lin_light = lin_heavy = lin_straddle_fail = 0
    n_log_fail = log_light = log_heavy = 0
    for ev in events:
        pos = ev.candidate_positions
        if pos.size == 0:
            continue
        m = cat.bh_mass[pos]
        me = cat.bh_mass_error[pos]
        gw_lo = (ev.M_z - K_SIGMA * ev.M_z_sigma) / (1.0 + ev.z_max)
        gw_hi = (ev.M_z + K_SIGMA * ev.M_z_sigma) / (1.0 + ev.z_min)
        lin_lo, lin_hi = m - K_SIGMA * me, m + K_SIGMA * me
        fail_lin = ~((gw_lo <= lin_hi) & (lin_lo <= gw_hi))
        log_lo, log_hi = m * np.exp(-K_SIGMA * (me / m)), m * np.exp(K_SIGMA * (me / m))
        fail_log = ~((gw_lo <= log_hi) & (log_lo <= gw_hi))
        n_all += int(pos.size)
        n_lin_fail += int(fail_lin.sum())
        lin_light += int((fail_lin & (lin_hi < gw_lo)).sum())
        lin_heavy += int((fail_lin & (lin_lo > gw_hi)).sum())
        n_log_fail += int(fail_log.sum())
        log_light += int((fail_log & (log_hi < gw_lo)).sum())
        log_heavy += int((fail_log & (log_lo > gw_hi)).sum())
    # A failing interval must lie entirely on one side; count any residual.
    lin_straddle_fail = n_lin_fail - lin_light - lin_heavy

    out = {
        "scope": "reproducible fleet basis (== wgeom_result.json P3 n_all)",
        "n_all": n_all,
        "n_lin_fail": n_lin_fail,
        "lin_too_light": lin_light,
        "lin_too_heavy": lin_heavy,
        "lin_side_residual": lin_straddle_fail,
        "lin_light_over_heavy": (lin_light / lin_heavy) if lin_heavy else None,
        "n_log_fail": n_log_fail,
        "log_too_light": log_light,
        "log_too_heavy": log_heavy,
        "log_heavy_over_light": (log_heavy / log_light) if log_light else None,
        "banked_impeached_comparand": "CLAIM_WGEO §3.9: 112,416,623 too-light vs 3,868,708 too-heavy (29:1) — scope-inconsistent",
    }
    Path(OUT_DIR).mkdir(exist_ok=True)
    with open(Path(OUT_DIR) / "wgeom_w2_split.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
