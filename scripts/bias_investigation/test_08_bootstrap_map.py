"""Test 8 (Phase 45 Step 0): Bootstrap MAP distribution.

Resamples the 412 cached per-event likelihoods at
``results/phase44_posteriors/`` with replacement (B=1000) and recomputes the
joint MAP per resample. Reports 68% and 95% bootstrap intervals plus a
verdict ``statistical`` vs ``systematic`` for the +0.0350 residual relative
to truth h=0.73.

Reuses ``master_thesis_code.bayesian_inference.posterior_combination`` for
log-space combination so the formula matches the cluster-produced
``combined_posterior.json`` exactly (Gray et al. 2020, Eq. A.19).

Run from project root:
    uv run python scripts/bias_investigation/test_08_bootstrap_map.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from master_thesis_code.bayesian_inference.posterior_combination import (
    CombinationStrategy,
    apply_strategy,
    build_likelihood_array,
    combine_log_space,
    load_posterior_jsons,
)

POSTERIORS_DIR = PROJECT_ROOT / "results" / "phase44_posteriors"
COMBINED_PATH = POSTERIORS_DIR / "combined_posterior.json"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "outputs" / "phase45"

H_TRUTH = 0.73
N_BOOTSTRAP = 1000
RANDOM_SEED = 20260429
STRATEGY = CombinationStrategy.PHYSICS_FLOOR


def _full_sample_posterior(
    likelihoods: npt.NDArray[np.float64],
    log_D_h: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], int]:
    processed, excluded = apply_strategy(likelihoods, STRATEGY)
    n_used = processed.shape[0]
    posterior = combine_log_space(processed, log_D_h=log_D_h, n_events_used=n_used)
    return posterior, n_used


def _bootstrap_map(
    likelihoods_processed: npt.NDArray[np.float64],
    h_arr: npt.NDArray[np.float64],
    log_D_h: npt.NDArray[np.float64],
    n_iter: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    n_events = likelihoods_processed.shape[0]
    log_likes = np.log(likelihoods_processed)
    map_samples = np.empty(n_iter, dtype=np.float64)
    for b in range(n_iter):
        idx = rng.integers(0, n_events, size=n_events)
        joint_log = np.sum(log_likes[idx], axis=0) - n_events * log_D_h
        max_log = np.max(joint_log)
        post = np.exp(joint_log - max_log)
        post /= np.sum(post)
        map_samples[b] = h_arr[int(np.argmax(post))]
    return map_samples


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load cached per-event likelihoods.
    h_values, event_likelihoods = load_posterior_jsons(POSTERIORS_DIR)
    likelihoods, det_indices = build_likelihood_array(h_values, event_likelihoods)
    h_arr = np.asarray(h_values, dtype=np.float64)
    print(f"Loaded {likelihoods.shape[0]} events x {likelihoods.shape[1]} h-bins")

    # 2. Load D(h) from the cached combined_posterior.json.
    with open(COMBINED_PATH) as f:
        combined = json.load(f)
    cached_h = np.asarray(combined["h_values"], dtype=np.float64)
    cached_post = np.asarray(combined["posterior"], dtype=np.float64)
    cached_map_h = float(combined["map_h"])
    D_h = np.asarray(combined["D_h_per_h"], dtype=np.float64)
    log_D_h = np.log(D_h)
    if not np.allclose(cached_h, h_arr):
        raise RuntimeError(f"h-grid mismatch: cached={cached_h[:3]}... vs loaded={h_arr[:3]}...")

    # 3. Sanity check: full-sample MAP should reproduce cached MAP=0.7650.
    posterior_full, n_used_full = _full_sample_posterior(likelihoods, log_D_h)
    full_map_h = float(h_arr[int(np.argmax(posterior_full))])
    print(
        f"Full-sample sanity check: n_used={n_used_full}, "
        f"recomputed MAP={full_map_h:.4f}, cached MAP={cached_map_h:.4f}"
    )
    if not np.isclose(full_map_h, cached_map_h, atol=1e-6):
        raise RuntimeError(
            f"Full-sample MAP {full_map_h:.6f} does not match cached "
            f"{cached_map_h:.6f}. Combination formula drift; abort before "
            f"reporting bootstrap stats."
        )
    if not np.allclose(posterior_full, cached_post, rtol=1e-9, atol=1e-12):
        max_diff = float(np.max(np.abs(posterior_full - cached_post)))
        print(
            f"WARNING: Posterior reproduction has max abs diff {max_diff:.2e} "
            "(MAP matches; difference is normalization-level, acceptable)."
        )

    # 4. Bootstrap.
    processed_full, excluded_full = apply_strategy(likelihoods, STRATEGY)
    rng = np.random.default_rng(RANDOM_SEED)
    print(f"Running {N_BOOTSTRAP} bootstrap iterations on {processed_full.shape[0]} events...")
    map_samples = _bootstrap_map(processed_full, h_arr, log_D_h, N_BOOTSTRAP, rng)

    # 5. Summary statistics.
    median = float(np.median(map_samples))
    mean = float(np.mean(map_samples))
    std = float(np.std(map_samples, ddof=1))
    p2_5, p16, p50, p84, p97_5 = (
        float(np.percentile(map_samples, q)) for q in (2.5, 16.0, 50.0, 84.0, 97.5)
    )
    interval_68 = [p16, p84]
    interval_95 = [p2_5, p97_5]

    # Verdict: statistical iff 68% interval contains H_TRUTH.
    verdict = "statistical" if (interval_68[0] <= H_TRUTH <= interval_68[1]) else "systematic"

    # Coverage: fraction of bootstrap MAPs within ±half-width of truth (sanity flag only).
    half_68 = 0.5 * (interval_68[1] - interval_68[0])
    half_95 = 0.5 * (interval_95[1] - interval_95[0])
    frac_truth_68 = float(np.mean(np.abs(map_samples - H_TRUTH) <= half_68))
    frac_truth_95 = float(np.mean(np.abs(map_samples - H_TRUTH) <= half_95))

    summary = {
        "n_iterations": int(N_BOOTSTRAP),
        "n_events_used": int(processed_full.shape[0]),
        "n_events_excluded": int(excluded_full),
        "strategy": STRATEGY.value,
        "h_truth": H_TRUTH,
        "full_sample_map_h": full_map_h,
        "cached_map_h": cached_map_h,
        "median": median,
        "mean": mean,
        "std": std,
        "p2.5": p2_5,
        "p16": p16,
        "p50": p50,
        "p84": p84,
        "p97.5": p97_5,
        "interval_68": interval_68,
        "interval_95": interval_95,
        "frac_within_68_of_truth": frac_truth_68,
        "frac_within_95_of_truth": frac_truth_95,
        "verdict": verdict,
        "map_distribution": [float(x) for x in map_samples],
        "random_seed": RANDOM_SEED,
    }

    out_json = OUTPUT_DIR / "bootstrap_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_json}")

    # 6. Plot.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bin_edges = np.unique(np.concatenate([h_arr, [h_arr[-1] + (h_arr[-1] - h_arr[-2])]]))
    bin_edges = np.r_[h_arr - 0.5 * np.diff(np.r_[h_arr[0], h_arr]), h_arr[-1] + 0.005]
    ax.hist(map_samples, bins=bin_edges, edgecolor="black", alpha=0.7, label="bootstrap MAP")
    ax.axvline(H_TRUTH, color="green", lw=2.0, label=f"truth h={H_TRUTH}")
    ax.axvline(full_map_h, color="red", lw=2.0, label=f"full-sample MAP={full_map_h:.4f}")
    ax.axvspan(interval_68[0], interval_68[1], color="red", alpha=0.10, label="68% bootstrap")
    ax.axvspan(interval_95[0], interval_95[1], color="red", alpha=0.05, label="95% bootstrap")
    ax.set_xlabel("MAP h per bootstrap resample")
    ax.set_ylabel("count")
    ax.set_title(
        f"Phase 45 Step 0 — bootstrap MAP distribution (B={N_BOOTSTRAP}); verdict: {verdict}"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_png = OUTPUT_DIR / "bootstrap_distribution.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")

    # 7. Console summary.
    print("")
    print(f"=== Phase 45 Step 0 verdict: {verdict.upper()} ===")
    print(f"  full-sample MAP = {full_map_h:.4f}")
    print(f"  bootstrap median = {median:.4f}")
    print(f"  bootstrap std    = {std:.4f}")
    print(
        f"  68% interval = [{interval_68[0]:.4f}, {interval_68[1]:.4f}]  (truth in: {interval_68[0] <= H_TRUTH <= interval_68[1]})"
    )
    print(
        f"  95% interval = [{interval_95[0]:.4f}, {interval_95[1]:.4f}]  (truth in: {interval_95[0] <= H_TRUTH <= interval_95[1]})"
    )
    if verdict == "systematic":
        print("  → proceed to Step 1 (first-bin diagnostics).")
    else:
        print("  → STOP. Lock in 'no fix needed' in HANDOFF-phase45-diagnosis.md.")


if __name__ == "__main__":
    main()
