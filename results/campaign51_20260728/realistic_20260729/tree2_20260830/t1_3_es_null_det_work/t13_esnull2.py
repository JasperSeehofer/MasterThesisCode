import numpy as np, pandas as pd, math, json
SP='/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/70977a05-4e21-4daa-91f0-d0330808c1ea/scratchpad'
A=pd.read_csv(f'{SP}/t13_esnull_events.csv',index_col=0)
D=math.log(math.sqrt(2)); LN2=math.log(2)
R='results/campaign51_20260728/realistic_20260729'; ZW=f'{R}/tree2_20260830/hier_s0_zwin_run'
# rho(s) check: (comb*Dt - B_num)/L_cat per node
out={}
for node in ('truth','s_plus','s_minus'):
    rat=[]
    for seed in (900101,900102,900103,900104):
        d=pd.read_csv(f'{ZW}/s0a_seed{seed}/node_{node}_sites2.2_nosmear_divisor_zwin_zk4/simulations/diagnostics/event_likelihoods.csv')
        d=d[np.isclose(d.h,0.73)].drop_duplicates('event_idx',keep='last')
        ok=d.L_cat_no_bh>0
        rat.append(((d.combined_no_bh*d.D_tilde_phi-d.B_num)/d.L_cat_no_bh)[ok].to_numpy())
    rat=np.concatenate(rat); out[f'beta_over_rho_{node}']=dict(median=float(np.median(rat)),min=float(rat.min()),max=float(rat.max()),n=len(rat))
b0=out['beta_over_rho_truth']['median']
out['rho_s_plus']=b0/out['beta_over_rho_s_plus']['median']; out['rho_s_minus']=b0/out['beta_over_rho_s_minus']['median']
out['C_s_lns']=(math.log(out['rho_s_plus'])-math.log(out['rho_s_minus']))/LN2
def st(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x); m=x.mean(); s=x.std(ddof=1)/math.sqrt(n); return dict(n=n,mean=float(m),sem=float(s),Z=float(m/s))
def clustered(x,seed):
    x=np.asarray(x,float); seed=np.asarray(seed); ok=np.isfinite(x); x=x[ok]; seed=seed[ok]
    m=x.mean(); G=np.unique(seed); g=len(G)
    resid=np.array([np.sum(x[seed==s]-m) for s in G]); n=len(x)
    var=g/(g-1)*np.sum(resid**2)/n**2
    return dict(mean=float(m),sem_clustered=float(math.sqrt(var)),Z_clustered=float(m/math.sqrt(var)),g=int(g))
for ch in ('nb','wb'):
    l1=A[f'slns_{ch}']; l2=A[f'l2_{ch}']
    out[f'per_seed_{ch}']={}
    for seed,S in A.groupby('seed'):
        k3=-3*np.nanmean(S[f'slns_{ch}']*S[f'l2_{ch}'])-np.nanmean(S[f'slns_{ch}']**3)
        out[f'per_seed_{ch}'][int(seed)]=dict(score_lns=st(S[f'slns_{ch}']),score_s_cw=st(S[f'slns_{ch}']-S.c*S.es.fillna(0)),score_s_unw=st(S[f'slns_{ch}']-S.es.fillna(0)),bartlett_bias=float(D**2/6*k3),I_hat=float(-np.nanmean(S[f'l2_{ch}'])))
    out[f'clustered_{ch}']=dict(score_lns=clustered(l1,A.seed),score_s_cw=clustered(l1-A.c*A.es.fillna(0),A.seed),score_s_unw=clustered(l1-A.es.fillna(0),A.seed))
    # trimmed Bartlett (5% two-sided on |l1|)
    ok=np.isfinite(l1)&np.isfinite(l2); x=l1[ok].to_numpy(); y=l2[ok].to_numpy()
    q=np.quantile(np.abs(x),0.95); m=np.abs(x)<=q
    out[f'bartlett_trim5_{ch}']=float(D**2/6*(-3*np.mean(x[m]*y[m])-np.mean(x[m]**3)))
    # single-host-like subset
    for thr in (0.5,0.2):
        S=A[A.pi_true>thr]; x=S[f'slns_{ch}']; y=S[f'l2_{ch}']; ok=np.isfinite(x)&np.isfinite(y)
        k3=-3*np.mean(x[ok]*y[ok])-np.mean(x[ok]**3)
        out[f'bartlett_pi_true>{thr}_{ch}']=dict(n=int(ok.sum()),bias=float(D**2/6*k3),I_hat=float(-np.mean(y[ok])),c_es=float((S.c*S.es).mean()),es=float(S.es.mean()),slns=st(x[ok]))
# curvature-based "alternative" reading: ln s_hat = mean(l1)/mean(-l2) pooled (E13-style curvature)
for ch in ('nb','wb'):
    out[f'lns_hat_{ch}']=float(np.nanmean(A[f'slns_{ch}'])/(-np.nanmean(A[f'l2_{ch}'])))
# Gaussian single-host reference values
u=np.linspace(-8,8,200001); phi=np.exp(-u**2/2)/math.sqrt(2*math.pi)
sec=-1+(u**2/2)*math.sinh(LN2)/D  # secant of ln N(u;0,s) in ln s at Delta=ln sqrt2
out['gauss_full_line_secant_bias']=float(np.trapezoid(phi*sec,u))
m=np.abs(u)<=4/math.sqrt(2)
out['gauss_truncated_Wminus_bias']=float(np.trapezoid(phi[m]*sec[m],u[m])/np.trapezoid(phi[m],u[m]))
out['gauss_E_l3']=4.0; out['gauss_bias_O(D2)']=float(D**2/6*4)
out['gauss_KL_asym']=float(((math.log(1/math.sqrt(2))+1-0.5)-(math.log(math.sqrt(2))+0.25-0.5))/LN2)
json.dump(out,open(f'{SP}/t13_esnull2_out.json','w'),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))
