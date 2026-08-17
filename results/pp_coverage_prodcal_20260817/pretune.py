"""Registered pretuning procedure (prereg §7, D-7) — mechanical, non-verdict-bearing.

Fixed disjoint seed 20270999; fixed lexicographic candidate sweep
z_support in {0.25, 0.30, 0.35} x sky_frac in {1e-4, 2e-4, 4e-4}; first pair
whose R=8, n=250 pretuning cell lands host_in_ball_fraction in [0.60, 0.70]
AND completion_fraction in [0.30, 0.42] wins and is frozen into §7. Outputs
are archived under pretuning/ and never scored.
"""

from __future__ import annotations

import json
from pathlib import Path

from darksiren_emri.validation.pp_coverage import PPCoverageConfig, run_coverage

PRETUNE_SEED = 20270999
CANDIDATES = [(zs, sf) for zs in (0.25, 0.30, 0.35) for sf in (1e-4, 2e-4, 4e-4)]
HOST_BAND = (0.60, 0.70)
COMP_BAND = (0.30, 0.42)


def main() -> None:
    outdir = Path(__file__).parent / "pretuning"
    outdir.mkdir(exist_ok=True)
    for zs, sf in CANDIDATES:
        cfg = PPCoverageConfig(
            n_realizations=8,
            n_events=250,
            injected_truths=[0.72],
            seed=PRETUNE_SEED,
            kernel="volume",
            catalogue_mode=True,
            mixture_mode="absolute",
            z_support=zs,
            sky_frac=sf,
            n_galaxies=200_000,
            mass_channel=True,
            mass_horizon_index=0.25,
            selection_cell="fused",
            gw_measurement_scatter=False,
            h_step=0.004,
        )
        res = run_coverage(cfg)
        block = res["results"]["0.7200"]
        hib, comp = block["host_in_ball_fraction"], block["completion_fraction"]
        (outdir / f"pretune_zs{zs}_sf{sf}.json").write_text(json.dumps(res, indent=2))
        landed = HOST_BAND[0] <= hib <= HOST_BAND[1] and COMP_BAND[0] <= comp <= COMP_BAND[1]
        print(
            f"zs={zs} sf={sf}: host_in_ball={hib:.3f} completion={comp:.3f} -> "
            f"{'LANDED' if landed else 'no'}"
        )
        if landed:
            (outdir / "CHOSEN.json").write_text(
                json.dumps(
                    {
                        "z_support": zs,
                        "sky_frac": sf,
                        "seed": PRETUNE_SEED,
                        "host_in_ball_fraction": hib,
                        "completion_fraction": comp,
                    }
                )
            )
            print(f"FROZEN: z_support={zs} sky_frac={sf}")
            return
    print("NO CANDIDATE LANDED — return to author (registered sweep exhausted).")


if __name__ == "__main__":
    main()
