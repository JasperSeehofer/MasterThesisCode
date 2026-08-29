import numpy as np, pandas as pd, json
from pathlib import Path
R = Path('/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729')
RUN = R/'fanout1_20260829/hier_s0_registered_run'
res={}
for seed in [900101,900102,900103,900104]:
    b=pd.read_csv(R/f'p3_b0_work/bc_{seed}_work/seed{seed}/simulations/diagnostics/event_likelihoods.csv')
    b=b[np.isclose(b.h,0.73,rtol=1e-9,atol=1e-12)].drop_duplicates('event_idx',keep='last').set_index('event_idx').sort_index()
    t=pd.read_csv(RUN/f's0a_seed{seed}/node_truth_sites2.2_nosmear/simulations/diagnostics/event_likelihoods.csv').drop_duplicates('event_idx',keep='last').set_index('event_idx').sort_index()
    assert (b.index==t.index).all(), (len(b),len(t))
    cols=[c for c in t.columns if c in b.columns]
    d={}
    for c in cols:
        x=t[c].to_numpy(); y=b[c].to_numpy()
        ad=np.abs(x-y); rd=ad/np.where(np.abs(y)>0,np.abs(y),1.0)
        d[c]={'max_abs':float(ad.max()),'max_rel':float(rd.max()),'n_diff':int((ad>0).sum())}
    # ln combined diffs
    lt=np.log(t.combined_no_bh); lb=np.log(b.combined_no_bh)
    dl=lt-lb
    # decompose: ln combined = ln(beta L_cat + B_num) - ln D~ ; check beta via ratio
    jt=json.load(open(RUN/f's0a_seed{seed}/node_truth_sites2.2_nosmear/selection_tables_h_0_73.json'))
    jb=json.load(open(R/f'p3_b0_work/bc_{seed}_work/seed{seed}/selection_tables_h_0_73.json'))
    res[seed]={'cols':d,'dln_no_bh':{'mean':float(dl.mean()),'min':float(dl.min()),'max':float(dl.max()),'sd':float(dl.std())},
               'dln_with_bh':{'mean':float((np.log(t.combined_with_bh)-np.log(b.combined_with_bh)).mean()),'max_abs':float(np.abs(np.log(t.combined_with_bh)-np.log(b.combined_with_bh)).max())},
               'sel_json_truth':jt,'sel_json_banked':jb,'sel_json_rel':{k:(jt[k]-jb[k])/jb[k] for k in jt}}
    # compute residual attributable to D_tilde and B_num
    dD=np.log(t.D_tilde_phi)-np.log(b.D_tilde_phi)
    res[seed]['dln_Dtilde']=float(dD.iloc[0]); res[seed]['dln_alpha']=float((np.log(t.alpha_G_phi)-np.log(b.alpha_G_phi)).iloc[0])
    res[seed]['dln_Bnum_max']=float(np.abs(np.log(t.B_num)-np.log(b.B_num)).max())
    res[seed]['dln_Lcat_nb_max']=float(np.abs(np.log(t.L_cat_no_bh.replace(0,np.nan))-np.log(b.L_cat_no_bh.replace(0,np.nan))).max())
    res[seed]['dln_Lcat_wb_max']=float(np.nanmax(np.abs(np.log(t.L_cat_with_bh.replace(0,np.nan))-np.log(b.L_cat_with_bh.replace(0,np.nan)))))
    res[seed]['dln_num_minus_Dtilde_check']=float(np.abs((dl - (np.log(t.B_num/t.D_tilde_phi*0+1)))).max()) if False else None
json.dump(res,open('/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/70977a05-4e21-4daa-91f0-d0330808c1ea/scratchpad/f2_out.json','w'),indent=1,default=float)
for seed,r in res.items():
    print('== seed',seed)
    for c,v in r['cols'].items():
        print(f"  {c:22s} max_abs={v['max_abs']:.3e} max_rel={v['max_rel']:.3e} n_diff={v['n_diff']}")
    print('  dln_no_bh',r['dln_no_bh'],'dln_with_bh',r['dln_with_bh'])
    print('  sel_json_rel',r['sel_json_rel'])
    print('  dln_Dtilde',r['dln_Dtilde'],'dln_alpha',r['dln_alpha'],'dln_Bnum_max',r['dln_Bnum_max'],'dln_Lcat_nb_max',r['dln_Lcat_nb_max'],'dln_Lcat_wb_max',r['dln_Lcat_wb_max'])
