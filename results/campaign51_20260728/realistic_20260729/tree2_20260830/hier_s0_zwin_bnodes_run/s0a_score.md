# S0-A pooled score (prereg §4.1) -- score-only, zero-compute read

Seeds requested: [900101, 900102, 900103, 900104]
Nodes requested: ['b_plus', 'b_minus']
Nodes present on disk (n seeds each): {'b_plus': 4, 'b_minus': 4}

## ln_L_no_bh
- score_b: mean=-0.8623345057895397 sem=0.47694418757541013 Z=-1.8080407063419661 n_pooled=461
- score_s: mean=nan sem=nan Z=nan n_pooled=0
- score_s_raw: mean=nan sem=nan Z=nan n_pooled=0
- score_lns: mean=nan sem=nan Z=nan n_pooled=0
- score_lns_R: mean=nan sem=nan Z=nan n_pooled=0
- score_b_available (b-axis nodes present): True
- score_s_available (Es_null_det cache found): False
- score_lns_R_available (both ±ln√2 and ±ln√2/2 s-node pairs present): False
- score_lns_R - score_lns (paired shift, T1.3-zwin PA-HIER-33 falsifier): mean=nan sem=nan n_pooled=0

## ln_L_with_bh
- score_b: mean=0.3166507176559612 sem=0.4097264347381919 Z=0.772834483716667 n_pooled=461
- score_s: mean=nan sem=nan Z=nan n_pooled=0
- score_s_raw: mean=nan sem=nan Z=nan n_pooled=0
- score_lns: mean=nan sem=nan Z=nan n_pooled=0
- score_lns_R: mean=nan sem=nan Z=nan n_pooled=0
- score_b_available (b-axis nodes present): True
- score_s_available (Es_null_det cache found): False
- score_lns_R_available (both ±ln√2 and ±ln√2/2 s-node pairs present): False
- score_lns_R - score_lns (paired shift, T1.3-zwin PA-HIER-33 falsifier): mean=nan sem=nan n_pooled=0

## GATE ENG (mean fraction of events moved >=1e-6 rel, per node)
- b_plus: mean_fraction_moved=nan pass=False
- b_minus: mean_fraction_moved=nan pass=False
- s_plus: mean_fraction_moved=nan pass=False
- s_minus: mean_fraction_moved=nan pass=False

**Note:** only the b-axis is ready on disk (b_ready=True, s_ready=False) -- the OTHER axis's score in payload['scores'] is unavailable (n_pooled=0/NaN), by design, NOT an error.
