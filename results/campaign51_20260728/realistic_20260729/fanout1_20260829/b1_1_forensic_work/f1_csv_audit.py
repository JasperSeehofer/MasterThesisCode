import numpy as np, pandas as pd, json, math
from pathlib import Path
R = Path('/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729')
RUN = R/'fanout1_20260829/hier_s0_registered_run'
SEEDS=[900101,900102,900103,900104]
NODES=['truth','b_plus','b_minus','s_plus','s_minus']
H=0.73
def load(seed,node):
    p=RUN/f's0a_seed{seed}'/f'node_{node}_sites2.2_nosmear/simulations/diagnostics/event_likelihoods.csv'
    df=pd.read_csv(p); assert np.allclose(df.h,H); df=df.drop_duplicates(subset="event_idx", keep="last"); return df.set_index("event_idx").sort_index()
out={}
allrows=[]
for seed in SEEDS:
    d={n:load(seed,n) for n in NODES}
    t=d['truth']
    cols=[c for c in t.columns if c!='h']
    # column audit
    audit={}
    for n in NODES[1:]:
        x=d[n]; assert (x.index==t.index).all()
        audit[n]={c: float(np.max(np.abs(x[c].to_numpy()-t[c].to_numpy()))) for c in cols}
    # CRB join
    crb=pd.read_csv(RUN/f's0a_seed{seed}'/'node_truth_sites2.2_nosmear/simulations/cramer_rao_bounds.csv')
    ev=t.index.to_numpy()
    sub=crb.iloc[ev]
    df=pd.DataFrame(index=t.index)
    df['seed']=seed
    for c in ['z_true','host_galaxy_index','in_catalog','s_tilde_phi_host','luminosity_distance','SNR','delta_luminosity_distance_delta_luminosity_distance','phiS','qS']:
        df[c]=sub[c].to_numpy()
    for n in NODES:
        for ch,col in (('nb','combined_no_bh'),('wb','combined_with_bh'),('cat','L_cat_no_bh'),('catwb','L_cat_with_bh')):
            v=d[n][col].to_numpy(); df[f'ln_{ch}_{n}']=np.where(v>0,np.log(np.where(v>0,v,1)),np.nan)
    for c in ['L_cat_no_bh','L_cat_with_bh','B_num','B_num_wbh','alpha_G_phi','D_tilde_phi','w_G','r_Malm','g_frac','L_comp']:
        df[c]=t[c].to_numpy()
    allrows.append(df)
    out[seed]={'audit':audit,'n':len(t),'snr_min_included':float(sub.SNR.min()),'snr_max_excluded':float(crb.drop(index=ev).SNR.max()) if len(crb)>len(ev) else None,'n_crb':len(crb)}
A=pd.concat(allrows)
A['dark']=A['L_cat_no_bh']==0
for ch in ('nb','wb','cat','catwb'):
    A[f'sb_{ch}']=(A[f'ln_{ch}_b_plus']-A[f'ln_{ch}_b_minus'])/0.04
    A[f'ss_{ch}']=(A[f'ln_{ch}_s_plus']-A[f'ln_{ch}_s_minus'])/(math.sqrt(2)-1/math.sqrt(2))
    A[f'slns_{ch}']=(A[f'ln_{ch}_s_plus']-A[f'ln_{ch}_s_minus'])/math.log(2)
    A[f'cb_{ch}']=(A[f'ln_{ch}_b_plus']-2*A[f'ln_{ch}_truth']+A[f'ln_{ch}_b_minus'])/0.02**2
    A[f'clns_{ch}']=(A[f'ln_{ch}_s_plus']-2*A[f'ln_{ch}_truth']+A[f'ln_{ch}_s_minus'])/(0.5*math.log(2))**2
def stat(x):
    x=x[np.isfinite(x)]; n=len(x); m=x.mean(); sem=x.std(ddof=1)/np.sqrt(n) if n>1 else np.nan; return dict(n=n,mean=float(m),sem=float(sem),Z=float(m/sem) if sem>0 else np.nan)
res={}
for ch in ('nb','wb','cat','catwb'):
    res[ch]={'score_b':stat(A[f'sb_{ch}'].to_numpy()),'score_s':stat(A[f'ss_{ch}'].to_numpy()),'score_lns':stat(A[f'slns_{ch}'].to_numpy())}
    res[ch]['by_class']={cls:{'score_b':stat(A.loc[A.dark==dk,f'sb_{ch}'].to_numpy()),'score_s':stat(A.loc[A.dark==dk,f'ss_{ch}'].to_numpy())} for cls,dk in (('dark',True),('matched',False))}
    res[ch]['by_seed']={str(s):{'score_b':stat(A.loc[A.seed==s,f'sb_{ch}'].to_numpy()),'score_s':stat(A.loc[A.seed==s,f'ss_{ch}'].to_numpy())} for s in SEEDS}
    # curvature / implied theta-hat (pooled over matched)
    M=A[~A.dark]
    Sp=np.nansum(M[f'sb_{ch}']); Spp=np.nansum(M[f'cb_{ch}'])
    Lp=np.nansum(M[f'slns_{ch}']); Lpp=np.nansum(M[f'clns_{ch}'])
    res[ch]['curv']={'S1_b':float(Sp),'S2_b':float(Spp),'b_hat':float(-Sp/Spp) if Spp<0 else None,'sigma_b':float(1/np.sqrt(-Spp)) if Spp<0 else None,
                     'S1_lns':float(Lp),'S2_lns':float(Lpp),'lns_hat':float(-Lp/Lpp) if Lpp<0 else None,'sigma_lns':float(1/np.sqrt(-Lpp)) if Lpp<0 else None}
# z bins
edges=[0.0,0.075,0.392,0.559,0.659,0.753,1.018,2.0]
A['zbin']=pd.cut(A.z_true,edges,labels=False)
res['by_zbin_nb']={}
for b in sorted(A.zbin.dropna().unique()):
    m=(A.zbin==b)&(~A.dark)
    res['by_zbin_nb'][f'{edges[int(b)]}-{edges[int(b)+1]}']={'n':int(m.sum()),'score_b':stat(A.loc[m,'sb_nb'].to_numpy()),'score_s':stat(A.loc[m,'ss_nb'].to_numpy()),'score_b_cat':stat(A.loc[m,'sb_cat'].to_numpy()),'score_s_cat':stat(A.loc[m,'ss_cat'].to_numpy()),'mean_z':float(A.loc[m,'z_true'].mean())}
# catalogue share c_i
A['c_i']=1-np.exp(A.ln_nb_truth - np.log(A.B_num/A.D_tilde_phi)) if False else np.nan
res['per_seed_meta']={str(k):v for k,v in out.items()}
json.dump(res,open('/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/70977a05-4e21-4daa-91f0-d0330808c1ea/scratchpad/f1_out.json','w'),indent=1,default=float)
A.to_csv('/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/70977a05-4e21-4daa-91f0-d0330808c1ea/scratchpad/f1_events.csv')
# print summary
for ch in ('nb','wb','cat','catwb'):
    print(ch, {k:res[ch][k] for k in ('score_b','score_s')})
    print('  class', res[ch]['by_class'])
    print('  curv', res[ch]['curv'])
print('zbin', json.dumps(res['by_zbin_nb'],indent=0,default=float)[:3000])
print('meta', out)
# column audit summary: which columns move
for seed in SEEDS:
    for n in NODES[1:]:
        moved={c:v for c,v in out[seed]['audit'][n].items() if v>0}
        print(seed,n,'moved cols:',sorted(moved))
print('n dark', int(A.dark.sum()), 'n', len(A), 'in_catalog all', bool(A.in_catalog.all()))
print('dark sb_nb max abs', float(np.nanmax(np.abs(A.loc[A.dark,'sb_nb']))), 'dark ss_nb max abs', float(np.nanmax(np.abs(A.loc[A.dark,'ss_nb']))))
