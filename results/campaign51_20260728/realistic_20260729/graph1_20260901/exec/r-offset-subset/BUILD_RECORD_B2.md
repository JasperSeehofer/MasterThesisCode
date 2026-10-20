# BUILD_RECORD_B2.md -- Phase B (b-offset-subset-scorer), influence vector

Builder B2 (influence). Script: `build_influence_vector.py`. Frozen T0 convention (gradient-trapezoid weights, physics-floor zero handling) reused verbatim from `results/prod2d_closure_20260818/tier0_bootstrap_jackknife.py` (`_moments`, `_physics_floor_apply`, `w = np.gradient(h_grid)`).

**Blindness:** this builder never opened `covariate_table_blind*.csv` and computed no registered aggregate (AUC/OR/p-value/Delta_strat) over the registered population -- per-event influence only.

## Input pins (verified)

- `iiib`: `results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv` md5 `8e6a2c18dc5838dd1d52641589243672` -- MATCH
- `joint_r1`: `results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_joint_r1/simulations/diagnostics/event_likelihoods.csv` md5 `745954a0fdee5f10878fb5e622a06144` -- MATCH

## Full-sample mean_h (10 s.f.), minimal-subset k, byte-id anchors (reported for the verifier -- NOT compared to the registered anchor values by this builder)

| venue | channel | mean_h_full (10 s.f.) | sigma_h_full | map_h_full | minimal_k (recomputed) | banked_k (Sec.2, not re-derived) | n_excluded | mean_h(all removed) |
|---|---|---|---|---|---|---|---|---|
| iiib | 1D | 0.6669869414 | 0.0175255065 | 0.6650 | 94 | 94 | 0 | 0.7300000000 |
| iiib | 2D | 0.6658540600 | 0.0184747390 | 0.6650 | 82 | 82 | 0 | 0.7300000000 |
| joint_r1 | 1D | 0.6670323337 | 0.0203458146 | 0.6650 | 46 | 46 | 0 | 0.7300000000 |
| joint_r1 | 2D | 0.6671265168 | 0.0189236404 | 0.6650 | 72 | 72 | 0 | 0.7300000000 |

## Top-10 influence events per venue/channel (byte-id anchors)

Two lists per venue/channel, both derived from the same influence array (no free choice between them): **(A) literal top-10 by |influence|**; **(B) top-10 by decreasing directional influence d_e** -- cross-checked to be what `rd_2d_bootstrap_jackknife_output.json`'s `top10_events_by_abs_influence` field actually contains (its name notwithstanding: populated there from `order[:10]`, a directional-influence sort, not an abs-value sort). List (B) is the one the G-2(iii) anchor will match; list (A) is reported because the build mandate names it literally.

### iiib / 1D

**(A) top-10 by |influence|**

| rank | event_idx | influence (mean_h(full) - mean_h(full-e)) | d_e (directional) |
|---|---|---|---|
| 1 | 889 | 1.471039271807806e-03 | -1.471039271807806e-03 |
| 2 | 298 | 1.226138198229032e-03 | -1.226138198229032e-03 |
| 3 | 1536 | 1.222996351655015e-03 | -1.222996351655015e-03 |
| 4 | 656 | 9.721126775904532e-04 | -9.721126775904532e-04 |
| 5 | 270 | 9.462979792230763e-04 | -9.462979792230763e-04 |
| 6 | 915 | 9.335016900677839e-04 | -9.335016900677839e-04 |
| 7 | 396 | 9.005196527683834e-04 | -9.005196527683834e-04 |
| 8 | 904 | 7.431746889098312e-04 | -7.431746889098312e-04 |
| 9 | 1036 | 6.956242072392316e-04 | -6.956242072392316e-04 |
| 10 | 1166 | 6.945769077253416e-04 | -6.945769077253416e-04 |

**(B) top-10 by decreasing directional influence d_e**

| rank | event_idx | influence (mean_h(full) - mean_h(full-e)) | d_e (directional) |
|---|---|---|---|
| 1 | 576 | -3.647497123231425e-04 | 3.647497123231425e-04 |
| 2 | 160 | -3.592730250494558e-04 | 3.592730250494558e-04 |
| 3 | 1176 | -3.566365870343313e-04 | 3.566365870343313e-04 |
| 4 | 1482 | -3.355660576783714e-04 | 3.355660576783714e-04 |
| 5 | 465 | -3.331825810699574e-04 | 3.331825810699574e-04 |
| 6 | 55 | -3.319048468566344e-04 | 3.319048468566344e-04 |
| 7 | 190 | -3.294546408012522e-04 | 3.294546408012522e-04 |
| 8 | 1153 | -3.268886383255287e-04 | 3.268886383255287e-04 |
| 9 | 373 | -3.223912970814480e-04 | 3.223912970814480e-04 |
| 10 | 724 | -3.221140321432170e-04 | 3.221140321432170e-04 |

### iiib / 2D

**(A) top-10 by |influence|**

| rank | event_idx | influence (mean_h(full) - mean_h(full-e)) | d_e (directional) |
|---|---|---|---|
| 1 | 889 | 1.621156995963768e-03 | -1.621156995963768e-03 |
| 2 | 298 | 1.373238357711748e-03 | -1.373238357711748e-03 |
| 3 | 1536 | 1.367587530148096e-03 | -1.367587530148096e-03 |
| 4 | 656 | 1.094942542714361e-03 | -1.094942542714361e-03 |
| 5 | 270 | 1.068134675332288e-03 | -1.068134675332288e-03 |
| 6 | 915 | 1.063835828335136e-03 | -1.063835828335136e-03 |
| 7 | 396 | 9.632946382422958e-04 | -9.632946382422958e-04 |
| 8 | 474 | 9.488577732841286e-04 | -9.488577732841286e-04 |
| 9 | 494 | 8.453681746709574e-04 | -8.453681746709574e-04 |
| 10 | 1166 | 7.913451278022121e-04 | -7.913451278022121e-04 |

**(B) top-10 by decreasing directional influence d_e**

| rank | event_idx | influence (mean_h(full) - mean_h(full-e)) | d_e (directional) |
|---|---|---|---|
| 1 | 576 | -6.014649540266870e-04 | 6.014649540266870e-04 |
| 2 | 94 | -5.607951773091147e-04 | 5.607951773091147e-04 |
| 3 | 46 | -5.318541648164921e-04 | 5.318541648164921e-04 |
| 4 | 172 | -4.726212182755152e-04 | 4.726212182755152e-04 |
| 5 | 201 | -4.011215089647635e-04 | 4.011215089647635e-04 |
| 6 | 160 | -4.000450566892244e-04 | 4.000450566892244e-04 |
| 7 | 1176 | -3.976545473484139e-04 | 3.976545473484139e-04 |
| 8 | 158 | -3.852745652646039e-04 | 3.852745652646039e-04 |
| 9 | 1482 | -3.728435575857114e-04 | 3.728435575857114e-04 |
| 10 | 55 | -3.678810835249235e-04 | 3.678810835249235e-04 |

### joint_r1 / 1D

**(A) top-10 by |influence|**

| rank | event_idx | influence (mean_h(full) - mean_h(full-e)) | d_e (directional) |
|---|---|---|---|
| 1 | 474 | -8.332257556926903e-03 | 8.332257556926903e-03 |
| 2 | 889 | 4.670877426327880e-03 | -4.670877426327880e-03 |
| 3 | 656 | 1.594815943630090e-03 | -1.594815943630090e-03 |
| 4 | 18 | 1.573668064463574e-03 | -1.573668064463574e-03 |
| 5 | 915 | 1.458011500875989e-03 | -1.458011500875989e-03 |
| 6 | 298 | 1.337954106030326e-03 | -1.337954106030326e-03 |
| 7 | 383 | 1.337132938002750e-03 | -1.337132938002750e-03 |
| 8 | 189 | 1.290022546504455e-03 | -1.290022546504455e-03 |
| 9 | 1342 | 1.265267534055980e-03 | -1.265267534055980e-03 |
| 10 | 270 | 1.264916605795863e-03 | -1.264916605795863e-03 |

**(B) top-10 by decreasing directional influence d_e**

| rank | event_idx | influence (mean_h(full) - mean_h(full-e)) | d_e (directional) |
|---|---|---|---|
| 1 | 474 | -8.332257556926903e-03 | 8.332257556926903e-03 |
| 2 | 1285 | -8.232482212362502e-04 | 8.232482212362502e-04 |
| 3 | 386 | -5.882113211724826e-04 | 5.882113211724826e-04 |
| 4 | 396 | -5.761758420526064e-04 | 5.761758420526064e-04 |
| 5 | 160 | -4.706574455175527e-04 | 4.706574455175527e-04 |
| 6 | 1176 | -4.669895171859340e-04 | 4.669895171859340e-04 |
| 7 | 1482 | -4.385842978814614e-04 | 4.385842978814614e-04 |
| 8 | 55 | -4.336341416980583e-04 | 4.336341416980583e-04 |
| 9 | 190 | -4.303212420779801e-04 | 4.303212420779801e-04 |
| 10 | 1153 | -4.268885121541111e-04 | 4.268885121541111e-04 |

### joint_r1 / 2D

**(A) top-10 by |influence|**

| rank | event_idx | influence (mean_h(full) - mean_h(full-e)) | d_e (directional) |
|---|---|---|---|
| 1 | 474 | -2.152702347465452e-03 | 2.152702347465452e-03 |
| 2 | 889 | 1.929909828017773e-03 | -1.929909828017773e-03 |
| 3 | 396 | -1.655352886296702e-03 | 1.655352886296702e-03 |
| 4 | 656 | 1.412327229313948e-03 | -1.412327229313948e-03 |
| 5 | 18 | 1.394683721036571e-03 | -1.394683721036571e-03 |
| 6 | 915 | 1.321554373890699e-03 | -1.321554373890699e-03 |
| 7 | 765 | 1.304986611972425e-03 | -1.304986611972425e-03 |
| 8 | 1285 | -1.193821092703828e-03 | 1.193821092703828e-03 |
| 9 | 298 | 1.167724280091154e-03 | -1.167724280091154e-03 |
| 10 | 270 | 1.125820721593174e-03 | -1.125820721593174e-03 |

**(B) top-10 by decreasing directional influence d_e**

| rank | event_idx | influence (mean_h(full) - mean_h(full-e)) | d_e (directional) |
|---|---|---|---|
| 1 | 474 | -2.152702347465452e-03 | 2.152702347465452e-03 |
| 2 | 396 | -1.655352886296702e-03 | 1.655352886296702e-03 |
| 3 | 1285 | -1.193821092703828e-03 | 1.193821092703828e-03 |
| 4 | 386 | -7.713835632900956e-04 | 7.713835632900956e-04 |
| 5 | 576 | -6.635273369571815e-04 | 6.635273369571815e-04 |
| 6 | 94 | -6.415067869585123e-04 | 6.415067869585123e-04 |
| 7 | 160 | -4.075247759147693e-04 | 4.075247759147693e-04 |
| 8 | 1176 | -4.050198584674147e-04 | 4.050198584674147e-04 |
| 9 | 1482 | -3.790139315419383e-04 | 3.790139315419383e-04 |
| 10 | 55 | -3.738134725950193e-04 | 3.738134725950193e-04 |

## Output files

- `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/influence_iiib.csv`: columns `event_idx, influence_2D, influence_1D, rank` -- `influence_2D`/`influence_1D` are the directional statistic d_e (Sec.2; positive = removing the event moves mean_h toward truth); `rank` is by decreasing `influence_2D` (the registered PRIMARY family).
- `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/influence_joint_r1.csv`: columns `event_idx, influence_2D, influence_1D, rank` -- `influence_2D`/`influence_1D` are the directional statistic d_e (Sec.2; positive = removing the event moves mean_h toward truth); `rank` is by decreasing `influence_2D` (the registered PRIMARY family).

## Notes

- `mean_h(all removed)` is the k=n_events endpoint of the drop-cumulative curve: a grid-symmetry check independent of the CSV data (flat weighted posterior over H_GRID_41), reported here as a cross-check, not a registered number.
- Per Sec.2, S (the high-influence subset) is defined by the BANKED k, not the `minimal_k_recomputed` column above; the recomputed value is offered purely as the G-2(ii) byte-id anchor for the verifier.
- This builder did not compute, and does not report, any of the registered separation or materiality statistics (AUC, OR, Holm p, Delta_strat) -- those are Phase C only, over the joined table.
