import numpy as np, pandas as pd, json, math, pickle
from pathlib import Path
SP=Path('/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/70977a05-4e21-4daa-91f0-d0330808c1ea/scratchpad')
A=pd.read_csv(SP/'f7_events.csv',index_col=0)
parts=[pickle.load(open(SP/f,'rb')) for f in ('f8_part_0_230.pkl','f8_part_230_461.pkl')]
res={}
C=pd.read_csv(SP/'f9_alldraws_C.csv'); C=C[C.evaluated].copy()
# align C (all draws, evaluated) to A by (seed,event_idx)
key=lambda df: list(zip(df.seed.astype(int),df.event_idx.astype(int)))
Cmap=dict(zip(key(C),zip(C.C_b,C.C_s)))
A['C_b']=[Cmap[k][0] for k in key(A)]; A['C_s']=[Cmap[k][1] for k in key(A)]
def st(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x); return dict(n=int(n),mean=float(x.mean()),sem=float(x.std(ddof=1)/math.sqrt(n)),Z=float(x.mean()/(x.std(ddof=1)/math.sqrt(n))))
for mode in ('E0_prod','E1_wide'):
    rows=[]; idx=[]
    for p in parts:
        rows+=p['results'][mode]; idx+=p['index']
    T=pd.DataFrame(rows,index=idx).loc[A.index]
    for form in ('off','phi'):
        T[f'{form}_sb']=(T[f'{form}_b_plus']-T[f'{form}_b_minus'])/0.04; T[f'{form}_ss']=(T[f'{form}_s_plus']-T[f'{form}_s_minus'])/(math.sqrt(2)-1/math.sqrt(2))
    T.to_csv(SP/f'f8_{mode}.csv')
    m=(~A.dark).to_numpy(); M=T[m]; c=A.c_nb[m].to_numpy(); Cb=A.C_b[m].to_numpy(); Cs=A.C_s[m].to_numpy()
    r={'n_cand_median':float(T.n_cand.median()),'n_cand_max':int(T.n_cand.max()),'n_zero_cand':int((T.n_cand==0).sum()),
       'off_sb':st(M.off_sb),'off_ss':st(M.off_ss),'phi_sb':st(M.phi_sb),'phi_ss':st(M.phi_ss),
       'off_sb_x_c':st(M.off_sb*c),'off_ss_x_c':st(M.off_ss*c),'phi_sb_x_c':st(M.phi_sb*c),'phi_ss_x_c':st(M.phi_ss*c),
       'phi_minus_C_sb':st(M.phi_sb-Cb),'phi_minus_C_ss':st(M.phi_ss-Cs),'phi_minus_C_sb_x_c':st((M.phi_sb-Cb)*c),'phi_minus_C_ss_x_c':st((M.phi_ss-Cs)*c),
       'off_minus_C_sb':st(M.off_sb-Cb),'off_minus_C_ss':st(M.off_ss-Cs)}
    if mode=='E0_prod':
        r['check_vs_measured']={'sb_max_abs_diff':float((M.off_sb-A.sb_cat[m]).abs().max()),'ss_max_abs_diff':float((M.off_ss-A.ss_cat[m]).abs().max()),'lnL_truth_max_abs_diff_minus_lnSigma':float(((T.off_truth-A.ln_cat_truth)[m]-(T.off_truth-A.ln_cat_truth)[m].median()).abs().max())}
    zg_edges=[0.0,0.075,0.15,0.25,0.392,2.0]; zb=pd.cut(A.z_g[m],zg_edges,labels=False).to_numpy()
    r['by_zg']={}
    for b in sorted(set(zb[~np.isnan(zb)])):
        s=(zb==b)
        r['by_zg'][f'{zg_edges[int(b)]}-{zg_edges[int(b)+1]}']={'n':int(s.sum()),'off_sb':st(M.off_sb[s]),'phi_sb':st(M.phi_sb[s]),'phi_minus_C_sb':st(M.phi_sb[s]-Cb[s]),'off_ss':st(M.off_ss[s]),'phi_ss':st(M.phi_ss[s]),'phi_minus_C_ss':st(M.phi_ss[s]-Cs[s]),'C_b':st(Cb[s]),'C_s':st(Cs[s])}
    res[mode]=r
res['C_pooled_evaluated']={'C_b':st(A.C_b[~A.dark]),'C_s':st(A.C_s[~A.dark])}
json.dump(res,open(SP/'f8_out.json','w'),indent=1,default=float)
f=lambda v:(round(v['mean'],3),round(v['sem'],3),round(v['Z'],2))
for mode in ('E0_prod','E1_wide'):
    r=res[mode]; print('==',mode,'ncand med',r['n_cand_median'],'max',r['n_cand_max'],'zero',r['n_zero_cand'])
    if 'check_vs_measured' in r: print('  check',r['check_vs_measured'])
    for k in ('off_sb','phi_sb','phi_minus_C_sb','off_minus_C_sb','off_sb_x_c','phi_sb_x_c','phi_minus_C_sb_x_c','off_ss','phi_ss','phi_minus_C_ss','off_minus_C_ss','off_ss_x_c','phi_ss_x_c','phi_minus_C_ss_x_c'):
        print('  ',k,f(r[k]))
    for k,v in r['by_zg'].items():
        print('   zg',k,'n',v['n'],'off_sb',f(v['off_sb']),'phi_sb',f(v['phi_sb']),'phi-C',f(v['phi_minus_C_sb']),'C_b',round(v['C_b']['mean'],2),'| off_ss',f(v['off_ss']),'phi_ss',f(v['phi_ss']),'phi-C',f(v['phi_minus_C_ss']),'C_s',round(v['C_s']['mean'],3))
print('C pooled',res['C_pooled_evaluated'])
