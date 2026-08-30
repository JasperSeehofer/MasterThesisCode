import numpy as np, pandas as pd, math, json
R='results/campaign51_20260728/realistic_20260729'
ZW=f'{R}/tree2_20260830/hier_s0_zwin_run'
f7=pd.read_csv(f'{R}/fanout1_20260829/b1_1_forensic_work/f7_events.csv',index_col=0)
D=math.log(math.sqrt(2)); LN2=math.log(2)
rows=[]
for seed in (900101,900102,900103,900104):
    nd={}
    for node in ('truth','s_plus','s_minus'):
        p=f'{ZW}/s0a_seed{seed}/node_{node}_sites2.2_nosmear_divisor_zwin_zk4/simulations/diagnostics/event_likelihoods.csv'
        d=pd.read_csv(p); d=d[np.isclose(d.h,0.73)].drop_duplicates('event_idx',keep='last').set_index('event_idx')
        nd[node]=d
    es=pd.read_csv(f'{ZW}/s0a_seed{seed}/es_null_det.csv').set_index('event_idx')
    t=nd['truth']
    df=pd.DataFrame(index=t.index)
    for ch,col in (('nb','combined_no_bh'),('wb','combined_with_bh')):
        for node in ('truth','s_plus','s_minus'):
            v=nd[node][col].reindex(df.index).to_numpy(float)
            df[f'l_{ch}_{node}']=np.where(v>0,np.log(np.where(v>0,v,1)),np.nan)
    for node in ('truth','s_plus','s_minus'):
        v=nd[node]['L_cat_no_bh'].reindex(df.index).to_numpy(float)
        df[f'lcat_{node}']=np.where(v>0,np.log(np.where(v>0,v,1)),np.nan)
    df['B_num']=t.B_num; df['Dt']=t.D_tilde_phi; df['comb']=t.combined_no_bh; df['Lcat']=t.L_cat_no_bh
    df['c']=1.0-df.B_num/(df.comb*df.Dt)
    df['es']=es.es_null_det.reindex(df.index)
    df['seed']=seed
    rows.append(df.reset_index())
A=pd.concat(rows,ignore_index=True)
# join forensic per-event table (z_g, sigma_g, n_cand, pi_true, c_nb (T1.0), Es_null_det (f4))
F=f7[['seed','event_idx','z_g','sigma_g','n_cand','pi_true','c_nb','Es_null_det','dark','z_true','mu_k','sd_k']].copy()
A=A.merge(F,on=['seed','event_idx'],how='left')
def st(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x); m=x.mean(); s=x.std(ddof=1)/math.sqrt(n); return dict(n=n,mean=float(m),sem=float(s),Z=float(m/s))
out={}
for ch in ('nb','wb'):
    lp,lm,l0=A[f'l_{ch}_s_plus'],A[f'l_{ch}_s_minus'],A[f'l_{ch}_truth']
    A[f'slns_{ch}']=(lp-lm)/LN2
    A[f'l2_{ch}']=(lp-2*l0+lm)/D**2   # second derivative in ln s
    out[f'score_lns_{ch}']=st(A[f'slns_{ch}'])
    out[f'score_s_unw_{ch}']=st(A[f'slns_{ch}']-A.es.fillna(0))   # driver: dark rows es NaN -> driver? check below
    out[f'score_s_unw_{ch}_dropnan']=st(A[f'slns_{ch}']-A.es)
    out[f'score_s_cw_{ch}']=st(A[f'slns_{ch}']-A.c*A.es.fillna(0))
    out[f'es_unw']=st(A.es); out['c_x_es']=st(A.c*A.es.fillna(0))
    # Bartlett (third Bartlett identity) estimate of E[l'''] under H0: E[l''']=-3E[l'l'']-E[l'^3]
    l1=A[f'slns_{ch}']; l2=A[f'l2_{ch}']
    k3=-3*np.nanmean(l1*l2)-np.nanmean(l1**3)
    out[f'bartlett_{ch}']=dict(E_l1l2=float(np.nanmean(l1*l2)),E_l1cubed=float(np.nanmean(l1**3)),E_l3_hat=float(k3),
        null_bias_hat=float(D**2/6*k3), E_l2=float(np.nanmean(l2)), I_hat=float(-np.nanmean(l2)), var_l1=float(np.nanvar(l1,ddof=1)))
    # bootstrap SEM of null_bias_hat
    rng=np.random.default_rng(0); bs=[]
    l1v=l1.to_numpy(); l2v=l2.to_numpy(); ok=np.isfinite(l1v)&np.isfinite(l2v); l1v=l1v[ok]; l2v=l2v[ok]
    for _ in range(4000):
        i=rng.integers(0,len(l1v),len(l1v)); bs.append(D**2/6*(-3*np.mean(l1v[i]*l2v[i])-np.mean(l1v[i]**3)))
    out[f'bartlett_{ch}']['null_bias_boot_sd']=float(np.std(bs)); out[f'bartlett_{ch}']['null_bias_boot_q']=[float(np.quantile(bs,q)) for q in (0.025,0.5,0.975)]
# catalogue-leg-only secant (divisor-in, as the column carries it) and via combined*Dt-B_num
A['slns_cat']=(A.lcat_s_plus-A.lcat_s_minus)/LN2
out['score_lns_cat_only']=st(A.slns_cat)
out['score_lns_cat_only_minus_es']=st(A.slns_cat-A.es)
# structure: by pi_true, n_cand, c quartiles
m=A[A.dark==False].copy() if 'dark' in A else A
for key in ('pi_true','n_cand','c','z_g','sigma_g'):
    q=pd.qcut(m[key],4,labels=False,duplicates='drop'); out[f'by_{key}']={}
    for b in sorted(q.dropna().unique()):
        S=m[q==b]; out[f'by_{key}'][int(b)]=dict(range=(float(S[key].min()),float(S[key].max())),slns_nb=st(S.slns_nb),slns_cat=st(S.slns_cat),es=st(S.es),c_es=st(S.c*S.es),c=float(S.c.mean()))
# pi_true>0.5 subset (true-host dominated) -> single-host limit test
for thr in (0.5,0.8):
    S=m[m.pi_true>thr]; out[f'pi_true>{thr}']=dict(n=len(S),slns_nb=st(S.slns_nb),slns_cat=st(S.slns_cat),es=st(S.es),c=float(S.c.mean()))
S=m[m.pi_true<0.05]; out['pi_true<0.05']=dict(n=len(S),slns_nb=st(S.slns_nb),slns_cat=st(S.slns_cat),es=st(S.es),c=float(S.c.mean()))
out['es_cache_vs_f4']=dict(max_abs_diff=float(np.nanmax(np.abs(A.es-A.Es_null_det))),corr=float(A[['es','Es_null_det']].dropna().corr().iloc[0,1]),n=int(A[['es','Es_null_det']].dropna().shape[0]))
out['n_dark']=int((A.dark==True).sum()); out['n_es_nan']=int(A.es.isna().sum())
out['c_stats']=dict(mean=float(A.c.mean()),median=float(A.c.median()))
json.dump(out,open('/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/70977a05-4e21-4daa-91f0-d0330808c1ea/scratchpad/t13_esnull_out.json','w'),indent=1,default=float)
A.to_csv('/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/70977a05-4e21-4daa-91f0-d0330808c1ea/scratchpad/t13_esnull_events.csv')
print(json.dumps(out,indent=1,default=float))
