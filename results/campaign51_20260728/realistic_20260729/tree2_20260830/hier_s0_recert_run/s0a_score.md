# S0-A pooled score (prereg §4.1) -- score-only, zero-compute read

Seeds requested: [900101, 900102, 900103, 900104]
Nodes requested: ['truth', 'b_plus', 'b_minus', 's_plus', 's_minus']
Nodes present on disk (n seeds each): {'truth': 4, 'b_plus': 4, 'b_minus': 4, 's_plus': 4, 's_minus': 4}

## ln_L_no_bh
- score_b: mean=-0.28878240960372603 sem=0.42705252094828333 Z=-0.6762222336551854 n_pooled=461
- score_s: mean=-0.07195958393659582 sem=0.012051274307800423 Z=-5.971118248467597 n_pooled=461

## ln_L_with_bh
- score_b: mean=0.1382981807270816 sem=0.36465410051128905 Z=0.379258537153897 n_pooled=461
- score_s: mean=-0.02920333412346437 sem=0.014408551690508958 Z=-2.02680566032885 n_pooled=461

## Verdict: band="B0-A'"
INSTRUMENT-DEFECT -- STOP (prereg §4.5)

## GATE ENG (mean fraction of events moved >=1e-6 rel, per node)
- b_plus: mean_fraction_moved=0.9885755753680282 pass=True
- b_minus: mean_fraction_moved=0.9885755753680282 pass=True
- s_plus: mean_fraction_moved=0.9885755753680282 pass=True
- s_minus: mean_fraction_moved=0.9885755753680282 pass=True

## GATE PARITY (truth node vs banked bc CSV, per seed)
- seed 900101: COMPARED, pass_exact=False
- seed 900102: COMPARED, pass_exact=False
- seed 900103: COMPARED, pass_exact=False
- seed 900104: COMPARED, pass_exact=False
