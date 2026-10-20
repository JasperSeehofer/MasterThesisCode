# [MKER] window-GEOMETRY instrument result

verdict: **INSTRUMENT-DEFECT**
gates: {'G1': False, 'G2': True, 'G3': True, 'G4': True, 'G_all': False}
failing reads: ['P3a', 'P5']

## P2 table

| label | CV | eps_lin_light | eps_lin_heavy | eps_lin_total | eps_log_total |
|---|---|---|---|---|---|
| min | 0.5930 | 0.000102 | 0.141629 | 0.141730 | 0.133614 |
| p10 | 0.7846 | 0.000000 | 0.160727 | 0.160727 | 0.133614 |
| median | 0.8614 | 0.000000 | 0.167792 | 0.167792 | 0.133614 |
| p75 | 0.9401 | 0.000000 | 0.174700 | 0.174700 | 0.133614 |
| p90 | 1.2137 | 0.000000 | 0.196457 | 0.196457 | 0.133614 |
| exhibit | 1.3032 | 0.000000 | 0.202887 | 0.202887 | 0.133614 |

catalogue-weighted mean eps_lin: 0.172176 (REPORTED-ONLY per prereg §3)

## P3 discordance census

n_all=2249231, n_lin=2154066, n_log=913485
n_lin/n_all=0.9577 (banked 0.949)
n_log/n_all=0.4061 (banked 0.421)
n_log/n_lin=0.4241 (banked 0.4437)
P3b lin∩¬log fraction: 0.5808 (bound ≥ 0.5280, passed=True)
P3c: 65877 readmitted rows (CV median 1.2324816719956675)

## P4 exhibit regression

arm=bt_900121 event_idx=20
checks: {'gw_floor_matches': True, 'gw_ceiling_matches': True, 'cone_matches': True, 'lin_pass_set_matches': True, 'lin_fail_set_matches': True, 'log_readmitted_matches': True, 'true_host_outside_cone': True}

## P5 eligible-set mean-redshift shift

{'n_events_used': 1795, 'median_shift_abs': -0.013331565953401209, 'mean_shift_abs': -0.020939293895290655, 'p5_shift_abs': -0.06552972135786088, 'max_abs_shift': 0.1308379549808429, 'median_shift_rel': -0.09855083868797498, 'banked_median_shift_rel': -0.145, 'sign_match': True, 'within_tolerance_abs_0p01': False, 'passed': False}
