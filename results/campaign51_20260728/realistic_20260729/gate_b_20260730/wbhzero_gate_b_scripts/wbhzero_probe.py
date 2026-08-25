import sys, os, json, logging, time
os.chdir("/home/jasper/Repositories/darksiren-emri")
sys.path.insert(0, "/home/jasper/Repositories/darksiren-emri")

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    import darksiren_emri.validation.correspondence_1d as c1d
    from pathlib import Path

    VENUE = "b0i2d"
    SEED = 900101
    H_GEN = 0.73
    H_BOUNDS = (0.50, 0.86)
    ARM = sys.argv[1] if len(sys.argv) > 1 else "bc"
    FLAG = "off" if ARM == "bc" else "mz_sel"

    out_root = Path(
        "/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/f76e9d1f-e875-48cc-888f-70b6e70d2905/scratchpad/wbhzero_work"
    )
    work_root = out_root / f"{ARM}_{SEED}_work"
    work_root.mkdir(parents=True, exist_ok=True)

    sigma_z_scale, area_scale = c1d.ARM_SPECS[VENUE]
    cfg = c1d.CorrespondenceConfig(sigma_z_scale=sigma_z_scale, area_scale=area_scale)
    gen = c1d.MirrorUniverseGenerator(cfg)
    host_pool, _observed_path, handler = gen.host_pool_for_sigma_scale(
        work_root / "catalogue", SEED, sigma_z_scale=sigma_z_scale
    )
    completeness_obj, phi_survival_table, detection_probability_obj = (
        c1d.build_b0i_2d_selection_objects(h_true=H_GEN)
    )
    events = gen.draw_realization(
        SEED,
        host_pool=host_pool,
        host_mode="catalogue_selected_2d",
        completeness=completeness_obj,
        phi_survival_table=phi_survival_table,
        detection_probability=detection_probability_obj,
    )

    t0 = time.time()
    diag_csv, elapsed = c1d.run_mirror_seed_inprocess(
        work_root / f"seed{SEED}",
        events,
        SEED,
        galaxy_catalog=handler,
        h_values=(H_GEN,),
        selection_in_completion_numerator="fused",
        catalogue_numerator_survival="off",
        catalogue_numerator_survival_2d=FLAG,
        catalogue_numerator_survival_2d_center="eff",
        catalogue_global_selection="phi",
        h_bounds=H_BOUNDS,
    )
    print("ELAPSED", time.time() - t0)
    print("DIAG_CSV", diag_csv)


if __name__ == "__main__":
    main()
