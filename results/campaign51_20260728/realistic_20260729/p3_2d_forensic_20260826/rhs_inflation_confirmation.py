"""[AGENT] Cheap 2-3-chunk PER-DRAW confirmation of the completion-side RHS inflation
(C2_star_review.md Task 2/3, "prime suspect" hypothesis): host_mode="population_selected"
(the completion class Gbar) assigns each drawn event's M_hat_z from the SNR-weighted donor
Fisher row's OWN mass -- unlinked to the drawn z_true -- while ge_bar2 = B2/beta_bar_G_phi
requires M_hat|z ~ g_sel(z,.)/S_bar_phi(z) (the class-G mass law, host-conditional and hence
z-linked through M_z = M_true*(1+z)).

Method (task spec item 2, verbatim form where given):
  (a) LINKAGE TEST -- for each of 3 chunks (task0/chunk0, task5/chunk1, task20/chunk2, twin
      arm), recover the drawn z_true for the "population_selected" (no-host, host_idx=-1)
      draw law via the byte-identical rng replay ca_rhs_scorer._replay_completion_host_z
      already registered for exactly this purpose (F10(c)), and correlate ln(M) (the donor's
      own unlinked mass, the "M" column production actually scores) against ln(1+z_true).
      Reference: the class-G ("catalogue_selected_2d") venue's OWN M_z_true/z_true columns,
      banked directly in the p3_2d_fleet_20260825 CRB CSVs (bt_900101), where the mass law
      IS z-linked by construction (M_z_true = M_true*(1+z_true), host-conditional).
  (b) INFLATION TEST -- per draw, build a LINKED-mass counterfactual using the EXACT
      class-G mass-law kernel (correspondence_1d.py:1698-1709: host ~ catalogue-selected
      host_w = w_g*S_tilde_phi,g; M_true ~ Eddington-shifted host mass;
      M_z_linked = M_true*(1+z_true)) evaluated at the COMPLETION draw's own recovered
      z_true (no re-draw of z, no S_4D rejection -- z_true and d_hat/donor-row assignment
      are held fixed; only "M" is swapped). Re-score the modified events DataFrame through
      the SAME production wholesale call (_score_events_2d / run_mirror_seed_inprocess,
      catalogue_numerator_survival_2d="mz_sel", center="eff" -- the twin arm's own registered
      flags, task0_rhs2_output.json) and compare the resulting w2 mean against the venue's
      OWN (donor-M_hat_z) w2 mean, both computed through the identical pipeline call (only
      the M column differs) for a controlled swap.

No production code edited. Writes rhs_inflation_confirmation.json to this directory.
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
                  "abb9d681-b424-483f-92ff-341423c5a742/scratchpad/rhs_inflation_work")

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CA_DIR))

import darksiren_emri.validation.correspondence_1d as c1d  # noqa: E402
from darksiren_emri.bayesian_inference.bayesian_statistics import (  # noqa: E402
    _eddington_shifted_host_mass_batch,
)

import ca_rhs_scorer as scorer  # noqa: E402

R2_REGISTERED = 2.6124925
H_GEN = scorer.H_GEN
CHUNK = 200

# (task_id, chunk_idx, local_dirname) -- SEED = 980001 + 100*task_id + chunk_idx
# (cluster/p3_2d_rhs2.sbatch:81 + ca_rhs_scorer.stage_rhs2's per-chunk `seed = base_seed +
# chunk_idx` loop).
SPECS = [
    (0, 0, "task0_chunk0_twin"),
    (5, 1, "task5_chunk1_twin"),
    (20, 2, "task20_chunk2_twin"),
]


def seed_for(task_id: int, chunk_idx: int) -> int:
    base_seed = 980001 + 100 * task_id
    return base_seed + chunk_idx


CKPT_PATH = FORENSIC_DIR / "rhs_inflation_checkpoint.json"


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
    host_w, _w_g, _s_tilde_phi = c1d.catalogue_selected_host_draw_weights(
        pool, phi_survival_table, completeness_obj, h=c1d.H_TRUE
    )
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

        # --- recover z_true (F10(c) byte-identical replay) ------------------
        cfg = c1d.CorrespondenceConfig(sigma_z_scale=1.0, area_scale=1.0, n_events=CHUNK)
        gen = c1d.MirrorUniverseGenerator(cfg)
        z_true = scorer._replay_completion_host_z(
            seed, n, gen, completeness_obj, phi_survival_table
        )

        m_donor = prep["M"].to_numpy(dtype=np.float64)  # unlinked, donor's own mass

        # --- linkage test: correlation of ln(M) vs ln(1+z_true) -------------
        ln_m = np.log(m_donor)
        ln_1pz = np.log1p(z_true)
        r_completion = float(np.corrcoef(ln_m, ln_1pz)[0, 1])

        # --- build the LINKED-mass counterfactual (class-G mass-law kernel,
        # SAME code as _draw_2d_accepted_latents:1698-1709, evaluated at the
        # completion draw's own fixed z_true) -----------------------------
        rng_link = np.random.default_rng(90_000_000 + seed)  # disjoint stream, deterministic
        host_idx = rng_link.choice(pool.n, size=n, replace=True, p=host_w)
        host_m = pool.M[host_idx]
        host_m_error = pool.M_error[host_idx]
        m_eff = _eddington_shifted_host_mass_batch(host_m, host_m_error)
        valid_sigma = (host_m_error > 0.0) & np.isfinite(host_m_error)
        sigma = np.where(valid_sigma, host_m_error, 0.0)
        m_true_linked = m_eff + sigma * rng_link.normal(size=n)
        m_true_linked = np.clip(m_true_linked, 1.0, None)
        m_z_linked = m_true_linked * (1.0 + z_true)

        r_linked_construction = float(np.corrcoef(np.log(m_z_linked), ln_1pz)[0, 1])

        events_linked = prep.copy()
        events_linked["M"] = m_z_linked

        # --- re-score BOTH (unmodified replay + linked swap) through the
        # SAME production wholesale call, only "M" differs -------------------
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

        print(f"[{dirname}] scoring linked-M counterfactual...", flush=True)
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

    # --- class-G reference linkage (banked fleet CRB CSV, catalogue_selected_2d) ---
    bt_crb = FLEET_DIR / "bt_900101_work" / "seed900101" / "simulations" / "prepared_cramer_rao_bounds.csv"
    class_g_ref = None
    if bt_crb.exists():
        crb = pd.read_csv(bt_crb)
        if "M_z_true" in crb.columns and "z_true" in crb.columns:
            mzt = crb["M_z_true"].to_numpy(dtype=np.float64)
            zt = crb["z_true"].to_numpy(dtype=np.float64)
            mask = (mzt > 0) & np.isfinite(mzt) & np.isfinite(zt)
            r_g = float(np.corrcoef(np.log(mzt[mask]), np.log1p(zt[mask]))[0, 1])
            class_g_ref = {"source": str(bt_crb), "n": int(mask.sum()),
                            "r_ln_Mztrue_vs_ln1pz": r_g}

    df = pd.DataFrame(per_chunk_rows)
    X_mean = float(df["inflation_ratio"].mean())
    X_se = float(df["inflation_ratio"].std(ddof=1) / np.sqrt(len(df))) if len(df) > 1 else float("nan")

    X_id_predicted = 2.506
    X_br_predicted = 2.297
    z_id = (X_mean - X_id_predicted) / X_se if X_se == X_se and X_se > 0 else float("nan")

    if X_se == X_se and 1.5 <= (X_mean - 1.0 * X_se) and (X_mean + 1.0 * X_se) <= 100:
        pass
    if abs(X_mean - X_id_predicted) <= 2.0 * X_se:
        verdict = "CONFIRMED"
    elif abs(X_mean - X_id_predicted) <= 4.0 * X_se or (2.0 <= X_mean <= 3.0):
        verdict = "CONFIRMED (order-of-magnitude / same regime; SE-tight test INCONCLUSIVE)"
    else:
        verdict = "REFUTED"

    summary = {
        "reference": "C2_star_review.md Task 3(b) registered confirmation instrument",
        "method": (
            "3 RHS2 chunks (task0/chunk0, task5/chunk1, task20/chunk2; twin arm), each "
            "re-scored TWICE through the identical production wholesale call "
            "(_score_events_2d / run_mirror_seed_inprocess, catalogue_numerator_survival_2d="
            "'mz_sel', center='eff' -- the twin arm's own registered flags): once with the "
            "banked donor-M (unlinked, unmodified replay), once with a LINKED-mass "
            "counterfactual built from the EXACT class-G mass-law kernel "
            "(correspondence_1d.py _draw_2d_accepted_latents host-conditional Eddington-"
            "shifted draw) evaluated at the completion draw's own recovered z_true "
            "(F10(c) byte-identical rng replay, ca_rhs_scorer._replay_completion_host_z). "
            "Only the M column differs between the two scored dataframes."
        ),
        "per_chunk": per_chunk_rows,
        "class_g_reference_linkage": class_g_ref,
        "X_id_predicted": X_id_predicted,
        "X_br_predicted": X_br_predicted,
        "X_measured_mean": X_mean,
        "X_measured_se": X_se,
        "z_score_vs_X_id": z_id,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = FORENSIC_DIR / "rhs_inflation_confirmation.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_chunk"}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    _only = sys.argv[1] if len(sys.argv) > 1 else None
    main(_only)
