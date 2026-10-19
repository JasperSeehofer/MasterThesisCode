import numpy as np, pandas as pd, math, json, glob
R='results/campaign51_20260728/realistic_20260729'; RC=f'{R}/tree2_20260830/hier_s0_recert_run'
D=math.log(math.sqrt(2)); LN2=math.log(2)
rows=[]
for seed in (900101,900102,900103,900104):
    nd={}
    for node in ('truth','s_plus','s_minus'):
        p=glob.glob(f'{RC}/s0a_seed{seed}/node_{node}_sites2.2_nosmear_divisor*/simulations/diagnostics/event_likelihoods.csv')
        assert len(p)==1,p
        d=pd.read_csv(p[0]); d=d[np.isclose(d.h,0.73)].drop_duplicates('event_idx',keep='last').set_index('event_idx'); nd[node]=d
    df=pd.DataFrame(index=nd['truth'].index)
    for ch,col in (('nb','combined_no_bh'),('wb','combined_with_bh')):
        for node in ('truth','s_plus','s_minus'):
            v=nd[node][col].reindex(df.index).to_numpy(float); df[f'l_{ch}_{node}']=np.where(v>0,np.log(np.where(v>0,v,1)),np.nan)
    df['seed']=seed; rows.append(df.reset_index())
A=pd.concat(rows,ignore_index=True); out={'run':'hier_s0_recert_run (T1.2, theta-blind window, divisor on)'}
for ch in ('nb','wb'):
    l1=(A[f'l_{ch}_s_plus']-A[f'l_{ch}_s_minus'])/LN2; l2=(A[f'l_{ch}_s_plus']-2*A[f'l_{ch}_truth']+A[f'l_{ch}_s_minus'])/D**2
    ok=np.isfinite(l1)&np.isfinite(l2); x=l1[ok].to_numpy(); y=l2[ok].to_numpy()
    k3=-3*np.mean(x*y)-np.mean(x**3)
    rng=np.random.default_rng(0); bs=[]
    for _ in range(4000):
        i=rng.integers(0,len(x),len(x)); bs.append(D**2/6*(-3*np.mean(x[i]*y[i])-np.mean(x[i]**3)))
    out[ch]=dict(n=int(ok.sum()),score_lns_mean=float(x.mean()),score_lns_sem=float(x.std(ddof=1)/math.sqrt(len(x))),I_hat=float(-y.mean()),E_l3_hat=float(k3),bartlett_bias=float(D**2/6*k3),boot_sd=float(np.std(bs)))
print(json.dumps(out,indent=1))
json.dump(out,open('/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/70977a05-4e21-4daa-91f0-d0330808c1ea/scratchpad/t13_esnull3_out.json','w'),indent=1)
