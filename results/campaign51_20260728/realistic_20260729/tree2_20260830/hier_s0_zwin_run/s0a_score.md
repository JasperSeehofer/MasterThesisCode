# S0-A pooled score (prereg §4.1) -- score-only, zero-compute read

Seeds requested: [900101, 900102, 900103, 900104]
Nodes requested: ['truth', 's_plus', 's_minus']
Nodes present on disk (n seeds each): {'truth': 4, 's_plus': 4, 's_minus': 4}

## ln_L_no_bh
- score_b: mean=nan sem=nan Z=nan n_pooled=0
- score_s: mean=-0.04237092110586195 sem=0.012751511508574368 Z=-3.3228155797350696 n_pooled=461
- score_s_raw: mean=0.0038865388170488424 sem=0.012639030883900082 Z=0.3075029132177859 n_pooled=461
- score_lns: mean=0.0039648115580011465 sem=0.012893573971421563 Z=0.3075029132177858 n_pooled=461
- score_b_available (b-axis nodes present): False
- score_s_available (Es_null_det cache found): True

## ln_L_with_bh
- score_b: mean=nan sem=nan Z=nan n_pooled=0
- score_s: mean=-0.015472730788592258 sem=0.01654442712799539 Z=-0.9352231218940378 n_pooled=461
- score_s_raw: mean=0.030253708920119284 sem=0.016371271789410613 Z=1.847975484695588 n_pooled=461
- score_lns: mean=0.03086300187527085 sem=0.016700980143334977 Z=1.8479754846955883 n_pooled=461
- score_b_available (b-axis nodes present): False
- score_s_available (Es_null_det cache found): True

## Verdict: band="B0-A'"
INSTRUMENT-DEFECT -- STOP (prereg §4.5)

## GATE ENG (mean fraction of events moved >=1e-6 rel, per node)
- b_plus: mean_fraction_moved=nan pass=False
- b_minus: mean_fraction_moved=nan pass=False
- s_plus: mean_fraction_moved=0.9956959706959707 pass=True
- s_minus: mean_fraction_moved=0.9956959706959707 pass=True

## GATE PARITY (truth node vs banked bc CSV, per seed)
- seed 900101: COMPARED, pass_exact=False
- seed 900102: COMPARED, pass_exact=False
- seed 900103: COMPARED, pass_exact=False
- seed 900104: COMPARED, pass_exact=False

**Note:** only the s-axis is ready on disk (b_ready=False, s_ready=True) -- the OTHER axis's score in payload['scores'] is unavailable (n_pooled=0/NaN), by design, NOT an error.
