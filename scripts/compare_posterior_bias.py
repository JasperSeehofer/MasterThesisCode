"""Compare biased vs diagnostic posterior runs across h-value grid.

Loads posterior JSON files from two evaluation runs (biased and diagnostic),
computes log-posteriors, and generates a markdown comparison report showing
whether the diagnostic fix shifts the posterior peak toward h_true = 0.73.
"""

import argparse
import json
import math
import sys
from pathlib import Path


def load_posteriors(run_dir: Path) -> list[dict[str, float]]:
    """Load all h-value posterior JSONs from a run directory.

    Args:
        run_dir: Path to the evaluation run directory (e.g., evaluation/run_v12_validation).

    Returns:
        List of dicts with keys: h, log_posterior, sum_likelihood.
        Sorted by h value ascending.
    """
    posteriors_dir = run_dir / "simulations" / "posteriors"
    if not posteriors_dir.exists():
        print(f"ERROR: posteriors directory not found: {posteriors_dir}", file=sys.stderr)
        sys.exit(1)

    results: list[dict[str, float]] = []
    for json_path in sorted(posteriors_dir.glob("h_*.json")):
        with open(json_path) as f:
            data = json.load(f)

        h_value = float(data["h"])
        detection_keys = [k for k in data if k != "h"]

        # Compute log-posterior = sum(log(likelihood_i)) for each detection
        log_posterior = 0.0
        sum_likelihood = 0.0
        for k in detection_keys:
            likelihood = data[k][0]
            sum_likelihood += likelihood
            if likelihood > 0:
                log_posterior += math.log(likelihood)
            else:
                log_posterior += -1e30  # effectively -inf but keep numeric

        results.append(
            {
                "h": h_value,
                "log_posterior": log_posterior,
                "sum_likelihood": sum_likelihood,
            }
        )

    results.sort(key=lambda r: r["h"])
    return results


def find_peak(results: list[dict[str, float]]) -> dict[str, float]:
    """Find the h-value where log-posterior is maximum."""
    return max(results, key=lambda r: r["log_posterior"])


def generate_report(
    biased: list[dict[str, float]],
    diagnostic: list[dict[str, float]],
    true_h: float = 0.73,
) -> str:
    """Generate a markdown comparison report.

    Args:
        biased: Posterior results from the biased run.
        diagnostic: Posterior results from the diagnostic run.
        true_h: The true Hubble constant value.

    Returns:
        Markdown string with comparison table, peaks, and verdict.
    """
    lines: list[str] = []
    lines.append("# Posterior Bias Comparison Report")
    lines.append("")
    lines.append("Comparison of Pipeline B evaluation results:")
    lines.append("- **Biased**: Original code (with /d_L factor and P_det enabled)")
    lines.append("- **Diagnostic**: Bias fix (removed /d_L factor, disabled P_det)")
    lines.append(f"- **True value**: h = {true_h}")
    lines.append("")

    # Build comparison table
    lines.append("## Log-Posterior Comparison")
    lines.append("")
    lines.append("| h_value | biased_log_post | diagnostic_log_post | delta |")
    lines.append("|---------|-----------------|---------------------|-------|")

    biased_by_h = {r["h"]: r for r in biased}
    diagnostic_by_h = {r["h"]: r for r in diagnostic}
    all_h = sorted(set(biased_by_h.keys()) | set(diagnostic_by_h.keys()))

    for h in all_h:
        b = biased_by_h.get(h)
        d = diagnostic_by_h.get(h)
        b_lp = f"{b['log_posterior']:.4f}" if b else "N/A"
        d_lp = f"{d['log_posterior']:.4f}" if d else "N/A"
        if b and d:
            delta = d["log_posterior"] - b["log_posterior"]
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "N/A"
        lines.append(f"| {h:.3f} | {b_lp} | {d_lp} | {delta_str} |")

    lines.append("")

    # Peaks
    biased_peak = find_peak(biased)
    diagnostic_peak = find_peak(diagnostic)

    lines.append("## Peak Analysis")
    lines.append("")
    lines.append(
        f"- **Biased peak**: h = {biased_peak['h']:.3f} "
        f"(log-posterior = {biased_peak['log_posterior']:.4f})"
    )
    lines.append(
        f"- **Diagnostic peak**: h = {diagnostic_peak['h']:.3f} "
        f"(log-posterior = {diagnostic_peak['log_posterior']:.4f})"
    )
    lines.append(f"- **True value**: h = {true_h}")
    lines.append(f"- **Biased offset**: {biased_peak['h'] - true_h:+.3f}")
    lines.append(f"- **Diagnostic offset**: {diagnostic_peak['h'] - true_h:+.3f}")
    lines.append("")

    # ASCII visualization
    lines.append("## Posterior Shape (normalized log-posterior)")
    lines.append("")
    lines.append("```")
    lines.append("h_value  | Biased                           | Diagnostic")
    lines.append("---------+----------------------------------+----------------------------------")

    # Normalize both to their own peak
    b_max = biased_peak["log_posterior"]
    d_max = diagnostic_peak["log_posterior"]
    bar_width = 30

    for h in all_h:
        b = biased_by_h.get(h)
        d = diagnostic_by_h.get(h)

        if b and b_max != 0:
            # Relative to peak (0 = peak, negative = lower)
            b_rel = b["log_posterior"] - b_max
            b_bar_len = max(0, int(bar_width * math.exp(b_rel)))
        else:
            b_bar_len = 0

        if d and d_max != 0:
            d_rel = d["log_posterior"] - d_max
            d_bar_len = max(0, int(bar_width * math.exp(d_rel)))
        else:
            d_bar_len = 0

        b_bar = "#" * b_bar_len
        d_bar = "#" * d_bar_len
        marker = " <-- true" if abs(h - true_h) < 0.001 else ""
        lines.append(f" {h:.3f}  | {b_bar:<{bar_width}s} | {d_bar:<{bar_width}s}{marker}")

    lines.append("```")
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    biased_err = abs(biased_peak["h"] - true_h)
    diag_err = abs(diagnostic_peak["h"] - true_h)

    if diag_err < biased_err:
        if diag_err < 0.02:
            verdict = (
                f"The diagnostic fix successfully shifts the posterior peak from "
                f"h={biased_peak['h']:.3f} to h={diagnostic_peak['h']:.3f}, "
                f"recovering the true value h={true_h}. "
                f"The /d_L factor and P_det were confirmed as the root cause of the bias."
            )
        else:
            verdict = (
                f"The diagnostic fix improves the posterior peak from "
                f"h={biased_peak['h']:.3f} to h={diagnostic_peak['h']:.3f} "
                f"(closer to true h={true_h}), but a residual offset of "
                f"{diag_err:.3f} remains. Additional bias sources may exist."
            )
    elif diag_err > biased_err:
        verdict = (
            f"WARNING: The diagnostic fix moved the posterior peak AWAY from the true value "
            f"(biased: h={biased_peak['h']:.3f}, diagnostic: h={diagnostic_peak['h']:.3f}, "
            f"true: h={true_h}). The /d_L and P_det changes do not explain the bias."
        )
    else:
        verdict = (
            f"The diagnostic fix did not change the posterior peak "
            f"(both at h={biased_peak['h']:.3f}). The bias source is elsewhere."
        )

    lines.append(verdict)
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare biased vs diagnostic posterior runs")
    parser.add_argument(
        "--biased",
        type=Path,
        required=True,
        help="Path to biased evaluation run directory",
    )
    parser.add_argument(
        "--diagnostic",
        type=Path,
        required=True,
        help="Path to diagnostic evaluation run directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write comparison report",
    )
    args = parser.parse_args()

    print(f"Loading biased posteriors from: {args.biased}")
    biased = load_posteriors(args.biased)
    print(f"  Found {len(biased)} h-values")

    print(f"Loading diagnostic posteriors from: {args.diagnostic}")
    diagnostic = load_posteriors(args.diagnostic)
    print(f"  Found {len(diagnostic)} h-values")

    report = generate_report(biased, diagnostic)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"\nReport written to: {args.output}")
    print("\n" + "=" * 70)
    print(report)


if __name__ == "__main__":
    main()
