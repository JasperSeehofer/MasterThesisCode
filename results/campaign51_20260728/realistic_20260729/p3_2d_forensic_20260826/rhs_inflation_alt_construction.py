"""[AGENT] PA-2D-10 (2026-08-26; PREREGISTRATION_P3_2D_20260825.md, final amendment block,
verbatim): author-granted ALTERNATIVE counterfactual construction, resolving row #209's
operationalization caveat on round 1 (``rhs_inflation_confirmation.py``).

Round 1 built the linked-mass counterfactual by drawing an INDEPENDENT catalogue host (the
class-G mass-law kernel: host ~ catalogue-selected host_w, M_true ~ Eddington-shifted host
mass) evaluated at the completion draw's own recovered z_true. PA-2D-10 replaces that draw
with a construction that changes ONLY the redshifting and holds the donor's own mass scale
fixed:

    M_hat_z,linked = M_donor,source * (1 + z_true,replayed)

i.e. no independent host/mass draw at all -- just re-redshift the SAME donor row's mass onto
the completion draw's own replayed z_true, instead of the donor's OWN original z (whatever z
the donor Fisher row was originally simulated at, production-pool-side).

Column identification (prepared_cramer_rao_bounds.csv, verified against
darksiren_emri/datamodels/parameter_space.py:261-273, ``set_host_galaxy_parameters``):
  - "M" is DETECTOR-FRAME M_hat_z = M_source * (1 + z_true) -- FEW/CRB convention, NOT source
    frame ("M_z = M_source*(1+z) in the M slot ... the stored CRB 'M' column genuinely holds
    M_z", parameter_space.py:261-266). There is no source-frame mass column in the CRB CSV.
  - "luminosity_distance" is `dist(host_galaxy.z, h=h_inj)` at THE SAME z and THE SAME h used
    for the M lift (parameter_space.py:268/273) -- so the donor's own original z (z_donor) is
    recoverable by inverting the SAME cosmology relation: z_donor = dist_to_redshift(d_donor,
    h=c1d.H_TRUE) (production h_inj = H_TRUE = H_GEN = 0.73 throughout this campaign; c1d.H_TRUE
    is the same constant round 1 already used for the completion/class-G host-z draw law).
  - Recovered donor source mass: M_donor,source = M_donor,M_hat_z / (1 + z_donor)
    (``physical_relations.redshifted_mass_inverse``, the exact inverse of the M-lift above).

Then: M_hat_z,linked = M_donor,source * (1 + z_true,replayed) via
``physical_relations.redshifted_mass`` -- same production function used at generation time,
just re-evaluated at the completion draw's replayed z instead of the donor's own original z.

z_true,replayed: IDENTICAL to round 1 -- the F10(c) byte-identical rng replay
(``ca_rhs_scorer._replay_completion_host_z``), z_true and d_hat/donor-row assignment held
fixed; only "M" is swapped.

Re-scoring: IDENTICAL production wholesale call to round 1 (``_score_events_2d`` /
``run_mirror_seed_inprocess``, ``catalogue_numerator_survival_2d="mz_sel"``, ``center="eff"``
-- the twin arm's own registered flags, task0_rhs2_output.json), same 3 chunks, same seeds.

Readout (PA-2D-10, verbatim): X_alt = w2(donor)/w2(linked-rescaled), pooled +/- chunk scatter,
vs the x2.5 residual. CONFIRMED => the completion-side M_hat_z redshift-unlinking is the
mechanism; REFUTED => the residual attribution moves off the completion-mass axis entirely.

No production code edited. Writes rhs_inflation_alt.json to this directory.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/jasper/Repositories/darksiren-emri")
FORENSIC_DIR = REPO / "results/campaign51_20260728/realistic_20260729/p3_2d_forensic_20260826"
CHUNKS_DIR = FORENSIC_DIR / "rhs_chunks"
FLEET_DIR = REPO / "results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825"
CA_DIR = REPO / "results/campaign51_20260728/realistic_20260729"
WORK_ROOT = Path("/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/"
                  "abb9d681-b424-483f-92ff-341423c5a742/scratchpad/rhs_inflation_alt_work")

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CA_DIR))

import darksiren_emri.validation.correspondence_1d as c1d  # noqa: E402
from darksiren_emri.physical_relations import (  # noqa: E402
    dist_to_redshift,
    redshifted_mass,
    redshifted_mass_inverse,
)

import ca_rhs_scorer as scorer  # noqa: E402

H_GEN = scorer.H_GEN
CHUNK = 200

# (task_id, chunk_idx, local_dirname) -- SEED = 980001 + 100*task_id + chunk_idx
# (cluster/p3_2d_rhs2.sbatch:81 + ca_rhs_scorer.stage_rhs2's per-chunk `seed = base_seed +
# chunk_idx` loop) -- IDENTICAL to round 1.
SPECS = [
    (0, 0, "task0_chunk0_twin"),
    (5, 1, "task5_chunk1_twin"),
    (20, 2, "task20_chunk2_twin"),
]


def seed_for(task_id: int, chunk_idx: int) -> int:
    base_seed = 980001 + 100 * task_id
    return base_seed + chunk_idx


CKPT_PATH = FORENSIC_DIR / "rhs_inflation_alt_checkpoint.json"


def load_ckpt() -> dict:
    if CKPT_PATH.exists():
        return json.loads(CKPT_PATH.read_text())
    return {"per_chunk": {}}


def save_ckpt(ckpt: dict) -> None:
    CKPT_PATH.write_text(json.dumps(ckpt, indent=2))


def main(only_dirname: str | None = None) -> None:
    t0 = time.time()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    ckpt = load_ckpt()

    print("Loading galaxy catalogue handler + host pool + completeness/phi tables "
          "(paid once)...", flush=True)
    completeness_obj, phi_survival_table = scorer._completion_class_objects(H_GEN)
    handler, pool = scorer._load_handler_and_pool()
    print(f"  context build done ({time.time() - t0:.1f}s)", flush=True)

    per_chunk_rows = []

    for task_id, chunk_idx, dirname in SPECS:
        if only_dirname is not None and dirname != only_dirname:
            continue
        if dirname in ckpt["per_chunk"]:
            print(f"[{dirname}] already checkpointed, skipping", flush=True)
            per_chunk_rows.append(ckpt["per_chunk"][dirname])
            continue
        seed = seed_for(task_id, chunk_idx)
        chunk_dir = CHUNKS_DIR / dirname
        prep = pd.read_csv(chunk_dir / "prepared_cramer_rao_bounds.csv")
        n = len(prep)
        assert n == CHUNK, (dirname, n)

        # --- recover z_true (F10(c) byte-identical replay, IDENTICAL to round 1) ---
        cfg = c1d.CorrespondenceConfig(sigma_z_scale=1.0, area_scale=1.0, n_events=CHUNK)
        gen = c1d.MirrorUniverseGenerator(cfg)
        z_true = scorer._replay_completion_host_z(
            seed, n, gen, completeness_obj, phi_survival_table
        )

        m_donor = prep["M"].to_numpy(dtype=np.float64)  # DETECTOR-frame M_hat_z (donor's own)
        d_donor = prep["luminosity_distance"].to_numpy(dtype=np.float64)  # Gpc, dist(z_donor, H_TRUE)

        # --- recover the donor's OWN original z (z_donor) by inverting the SAME
        # dist(z, h=H_TRUE) relation used to set luminosity_distance at generation time
        # (parameter_space.py:268/273 -- M and luminosity_distance share the SAME z, SAME h) ---
        z_donor = np.array(
            [dist_to_redshift(float(d), h=c1d.H_TRUE) for d in d_donor], dtype=np.float64
        )

        # --- recover the donor's source-frame mass (exact inverse of the production
        # M-lift, physical_relations.redshifted_mass_inverse) ---
        m_donor_source = np.array(
            [redshifted_mass_inverse(float(mz), float(zd)) for mz, zd in zip(m_donor, z_donor)],
            dtype=np.float64,
        )

        ln_m = np.log(m_donor)
        ln_1pz = np.log1p(z_true)
        r_completion = float(np.corrcoef(ln_m, ln_1pz)[0, 1])

        # --- PA-2D-10 construction: re-redshift the DONOR'S OWN source mass onto the
        # completion draw's own replayed z_true (physical_relations.redshifted_mass,
        # the exact production M-lift function) -- no independent host/mass draw ---
        m_z_linked = np.array(
            [redshifted_mass(float(ms), float(zt)) for ms, zt in zip(m_donor_source, z_true)],
            dtype=np.float64,
        )

        r_linked_construction = float(np.corrcoef(np.log(m_z_linked), ln_1pz)[0, 1])

        events_linked = prep.copy()
        events_linked["M"] = m_z_linked

        # --- re-score BOTH (unmodified replay + linked swap) through the SAME
        # production wholesale call, only "M" differs -- IDENTICAL to round 1 ---
        gcat = handler
        work_donor = WORK_ROOT / f"{dirname}_donor_work"
        work_linked = WORK_ROOT / f"{dirname}_linked_work"
        (work_donor / "simulations").mkdir(parents=True, exist_ok=True)
        (work_linked / "simulations").mkdir(parents=True, exist_ok=True)

        print(f"[{dirname}] scoring donor-M (unmodified replay)...", flush=True)
        at_donor = scorer._score_events_2d(
            prep, work_donor, seed, gcat,
            catalogue_numerator_survival_2d="mz_sel",
            catalogue_numerator_survival_2d_center="eff",
        )
        w2_donor = scorer._w2_from_csv_columns(at_donor)

        print(f"[{dirname}] scoring linked-M (PA-2D-10 rescale) counterfactual...", flush=True)
        at_linked = scorer._score_events_2d(
            events_linked, work_linked, seed, gcat,
            catalogue_numerator_survival_2d="mz_sel",
            catalogue_numerator_survival_2d_center="eff",
        )
        w2_linked = scorer._w2_from_csv_columns(at_linked)

        # cross-check against the ALREADY-BANKED cluster diagnostics for this chunk
        diag_bank = pd.read_csv(chunk_dir / "event_likelihoods.csv")
        diag_bank = diag_bank[np.isclose(diag_bank["h"], H_GEN)]
        w2_bank = scorer._w2_from_csv_columns(diag_bank)

        w2_donor_mean = float(w2_donor.mean())
        w2_linked_mean = float(w2_linked.mean())
        w2_bank_mean = float(w2_bank.mean())
        inflation_ratio = w2_donor_mean / w2_linked_mean if w2_linked_mean != 0 else float("nan")

        rec = dict(
            task_id=task_id, chunk_idx=chunk_idx, dirname=dirname, seed=seed,
            n=n,
            n_accepted_donor_replay=int(len(w2_donor)),
            n_accepted_linked=int(len(w2_linked)),
            n_accepted_bank=int(len(w2_bank)),
            r_ln_M_vs_ln1pz_completion=r_completion,
            r_ln_Mzlinked_vs_ln1pz_construction=r_linked_construction,
            median_M_hat_z_donor=float(np.median(m_donor)),
            median_M_hat_z_linked=float(np.median(m_z_linked)),
            median_z_donor_recovered=float(np.median(z_donor)),
            median_z_true_replayed=float(np.median(z_true)),
            w2_donor_mean_replay=w2_donor_mean,
            w2_linked_mean=w2_linked_mean,
            w2_bank_mean_cluster=w2_bank_mean,
            inflation_ratio=inflation_ratio,
        )
        per_chunk_rows.append(rec)
        ckpt["per_chunk"][dirname] = rec
        save_ckpt(ckpt)
        print(f"[{dirname}] r(lnM,ln1+z)={r_completion:+.4f}  "
              f"w2_donor={w2_donor_mean:.6f} (bank {w2_bank_mean:.6f})  "
              f"w2_linked={w2_linked_mean:.6f}  X={inflation_ratio:.4f}  "
              f"median M_hat_z donor={np.median(m_donor):.3e} linked={np.median(m_z_linked):.3e}  "
              f"[{time.time()-t0:.0f}s elapsed]", flush=True)

    if only_dirname is not None:
        print(f"[{only_dirname}] done, checkpointed. Run again for the next chunk / "
              f"omit --only for the final summary.", flush=True)
        return

    if len(ckpt["per_chunk"]) < len(SPECS):
        print(f"Only {len(ckpt['per_chunk'])}/{len(SPECS)} chunks checkpointed -- "
              f"run remaining chunks before the final summary.", flush=True)
        return

    per_chunk_rows = [ckpt["per_chunk"][d] for _, _, d in SPECS]

    df = pd.DataFrame(per_chunk_rows)
    X_mean = float(df["inflation_ratio"].mean())
    X_se = float(df["inflation_ratio"].std(ddof=1) / np.sqrt(len(df))) if len(df) > 1 else float("nan")

    # x2.5 residual class -- SAME registered candidates round 1 compared against
    # (C2_star_review.md Task 3(b); PA-2D-10 says "vs the x2.5 residual", not a re-derivation).
    X_id_predicted = 2.506
    X_br_predicted = 2.297
    z_id = (X_mean - X_id_predicted) / X_se if X_se == X_se and X_se > 0 else float("nan")

    if X_se == X_se and abs(X_mean - X_id_predicted) <= 2.0 * X_se:
        verdict = "CONFIRMED"
    elif X_se == X_se and (abs(X_mean - X_id_predicted) <= 4.0 * X_se or (2.0 <= X_mean <= 3.0)):
        verdict = "CONFIRMED (order-of-magnitude / same regime; SE-tight test INCONCLUSIVE)"
    elif 1.3 <= X_mean <= 5.0:
        verdict = "INCONCLUSIVE (materially >1 but outside the registered x2-3 CONFIRMED band)"
    else:
        verdict = "REFUTED"

    median_shift_donor = float(df["median_M_hat_z_donor"].median())
    median_shift_linked = float(df["median_M_hat_z_linked"].median())

    summary = {
        "registration": "PREREGISTRATION_P3_2D_20260825.md PA-2D-10 (final amendment block)",
        "method": (
            "3 RHS2 chunks (task0/chunk0, task5/chunk1, task20/chunk2; twin arm), IDENTICAL "
            "round-1 replay/scoring path (_score_events_2d / run_mirror_seed_inprocess, "
            "catalogue_numerator_survival_2d='mz_sel', center='eff'). ALTERNATIVE "
            "counterfactual (PA-2D-10): M_hat_z,linked = M_donor,source * (1 + z_true,replayed) "
            "-- the donor row's OWN source-frame mass (recovered via "
            "M_donor,source = M_donor,M_hat_z / (1 + z_donor), with z_donor recovered by "
            "inverting dist(z, h=H_TRUE) against the donor's OWN 'luminosity_distance' column, "
            "the SAME z/h pair parameter_space.py used to set both M and luminosity_distance at "
            "generation time), re-redshifted at the completion draw's own recovered z_true "
            "(F10(c) byte-identical rng replay, ca_rhs_scorer._replay_completion_host_z). NO "
            "independent host/mass draw (unlike round 1's class-G mass-law kernel draw). Only "
            "the M column differs between the two scored dataframes."
        ),
        "per_chunk": per_chunk_rows,
        "X_id_predicted": X_id_predicted,
        "X_br_predicted": X_br_predicted,
        "X_alt_measured_mean": X_mean,
        "X_alt_measured_se": X_se,
        "z_score_vs_X_id": z_id,
        "median_M_hat_z_donor_across_chunks": median_shift_donor,
        "median_M_hat_z_linked_across_chunks": median_shift_linked,
        "median_M_hat_z_ratio_donor_over_linked": (
            median_shift_donor / median_shift_linked if median_shift_linked else float("nan")
        ),
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = FORENSIC_DIR / "rhs_inflation_alt.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_chunk"}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    _only = sys.argv[1] if len(sys.argv) > 1 else None
    main(_only)
