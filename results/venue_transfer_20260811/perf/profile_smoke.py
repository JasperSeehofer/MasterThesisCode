"""Ad-hoc profiling driver for the venue-transfer realistic hot path.

PERF WORK ONLY — not part of the certified instrument, not imported by it,
never touches a registered output path. Runs one capped real-K seed
(cell Tc, sigma_mode='glade', the "realistic venue" cell) under cProfile,
seed-grain, 1 worker (in-process, no pool), and dumps:
  - a pstats binary (`profile_smoke.pstats`)
  - a plain-text top-N cumulative table (`profile_smoke_top.txt`)

Usage (from repo root):
    uv run python results/venue_transfer_20260811/perf/profile_smoke.py [n_events_cap]

Does not write anything under results/venue_transfer_20260811/ except this
perf/ subdirectory.
"""

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

from master_thesis_code.validation import venue_transfer as vt

PERF_DIR = Path(__file__).resolve().parent


def main() -> None:
    n_cap = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    vcfg = vt.VenueConfig(
        cell="Tc", h_true=0.730, balls="real_k", sigma_mode="glade", n_events_cap=n_cap
    )

    t0 = time.time()
    vctx = vt.build_venue_context(vcfg)
    t1 = time.time()
    print(
        f"[context] build_venue_context: {t1 - t0:.2f}s "
        f"n_events={vctx.event_rows.size} sum_K={int(vctx.K.sum())} max_K={int(vctx.K.max())}",
        flush=True,
    )

    seed = vt.VT_BASE_SEED + 44000  # inside the registered Tc(0.730) block
    profiler = cProfile.Profile()
    profiler.enable()
    t2 = time.time()
    rec = vt.run_seed_venue(seed, vctx)
    t3 = time.time()
    profiler.disable()
    print(f"[seed] run_seed_venue (profiled, seed-grain, 1 worker): {t3 - t2:.2f}s", flush=True)
    print(f"[seed] map_1d={rec['map_1d']:.4f} map_2d={rec['map_2d']:.4f}", flush=True)

    stats_path = PERF_DIR / "profile_smoke.pstats"
    profiler.dump_stats(str(stats_path))

    buf = io.StringIO()
    ps = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    ps.print_stats(25)
    ps2 = pstats.Stats(profiler, stream=buf).sort_stats("tottime")
    ps2.print_stats(15)
    text = buf.getvalue()
    (PERF_DIR / "profile_smoke_top.txt").write_text(
        f"n_events_cap={n_cap} n_events={vctx.event_rows.size} "
        f"sum_K={int(vctx.K.sum())} max_K={int(vctx.K.max())}\n"
        f"context_build_s={t1 - t0:.3f} seed_wall_s={t3 - t2:.3f}\n\n" + text
    )
    print(f"wrote {stats_path} and {PERF_DIR / 'profile_smoke_top.txt'}", flush=True)


if __name__ == "__main__":
    main()
