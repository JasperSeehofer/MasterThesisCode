"""Cross-figure MAP consistency summary (Phase H).

Loads the canonical combined posterior for both variants under the given
data directory and prints a markdown table of the MAP and event count.
Used by ``make validate-figures``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from master_thesis_code.plotting._helpers import load_canonical_combined_posterior


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing posteriors/ and posteriors_with_bh_mass/",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cache and recompute the canonical posterior.",
    )
    args = parser.parse_args()

    print("| variant | n_events | discrete MAP | continuous MAP | strategy |")
    print("|---|---|---|---|---|")
    for variant in ("posteriors", "posteriors_with_bh_mass"):
        try:
            _, _, meta = load_canonical_combined_posterior(
                args.data_dir, variant, refresh=args.refresh
            )
        except FileNotFoundError as e:
            print(f"| {variant} | — | — | — | missing ({e}) |")
            continue
        print(
            f"| {variant} | {meta['n_events_used']} "
            f"| {meta['discrete_map']:.4f} | {meta['continuous_map']:.4f} "
            f"| {meta['strategy']} |"
        )


if __name__ == "__main__":
    main()
