"""Generator-law sanity on ALL 800 drawn events (not just the 461 evaluated): pull vs the reconstructed law; exclusion analysis."""
import numpy as np, pandas as pd, json, math, sys
from pathlib import Path
from scipy.stats import norm
sys.path.insert(0,'/home/jasper/Repositories/darksiren-emri')
import darksiren_emri.validation.correspondence_1d as c1d
from darksiren_emri.bayesian_inference.bayesian_statistics import _host_pixels, _completeness_at_host_nodes
from darksiren_emri.physical_relations import comoving_volume_element
SP=Path('/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/70977a05-4e21-4daa-91f0-d0330808c1ea/scratchpad')
RUN=Path('/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run')
H=c1d.H_TRUE
handler=c1d._load_galaxy_catalog_handler(c1d.REDUCED_CATALOGUE_PATH); pool=c1d._host_pool_from_handler(handler)
completeness, phi_table = c1d.build_bsel_selection_objects(h_true=H); zg_grid,s_phi=phi_table[H]
ev=pd.read_csv(SP/'f1_events.csv')
rows=[]
for seed in [900101,900102,900103,900104]:
    crb=pd.read_csv(RUN/f's0a_seed{seed}/node_truth_sites2.2_nosmear/simulations/cramer_rao_bounds.csv')
    fq=pd.read_csv(RUN/f's0a_seed{seed}/node_truth_sites2.2_nosmear/simulations/fisher_quality.csv')
    evaluated=set(ev[ev.seed==seed].event_idx.astype(int))
    for i,r in crb.iterrows():
        rows.append(dict(seed=seed,event_idx=i,z_true=r.z_true,host=int(r.host_galaxy_index),dL=r.luminosity_distance,SNR=r.SNR,sig_dL=math.sqrt(r.delta_luminosity_distance_delta_luminosity_distance),evaluated=(i in evaluated)))
    if seed==900101: print('fisher_quality columns:',list(fq.columns)[:12], 'n',len(fq))
D=pd.DataFrame(rows)
idx=D.host.to_numpy(); z_g=pool.z[idx]; sig=pool.z_error[idx]; phiS=pool.phiS[idx]; qS=pool.qS[idx]
hp=_host_pixels(completeness,phiS,qS)
mk=np.empty(len(D)); mp=np.empty(len(D)); sk=np.empty(len(D)); sp=np.empty(len(D))
for i in range(len(D)):
    lo=max(z_g[i]-4*sig[i],1e-6); hi=z_g[i]+4*sig[i]; zz=np.linspace(lo,hi,4001)
    w=_completeness_at_host_nodes(completeness, zz[None,:], hp[i:i+1], H)[0]
    if not np.any(w>0): w=np.ones_like(zz)
    k=norm.pdf(zz,loc=z_g[i],scale=sig[i])*np.asarray(comoving_volume_element(zz,h=H))/(1+zz)*w; k/=np.trapezoid(k,zz)
    p=k*np.interp(zz,zg_grid,s_phi); p/=np.trapezoid(p,zz)
    mk[i]=np.trapezoid(zz*k,zz); sk[i]=math.sqrt(np.trapezoid((zz-mk[i])**2*k,zz)); mp[i]=np.trapezoid(zz*p,zz); sp[i]=math.sqrt(np.trapezoid((zz-mp[i])**2*p,zz))
D['z_g']=z_g; D['sigma_g']=sig; D['mu_k']=mk; D['mu_p']=mp; D['sd_k']=sk; D['sd_p']=sp
D['pull_p']=(D.z_true-D.mu_p)/D.sd_p; D['pull_k']=(D.z_true-D.mu_k)/D.sd_k; D['pull_g']=(D.z_true-D.z_g)/D.sigma_g
# also the KS-type check: CDF value of z_true under p per host (should be U(0,1))
def st(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x); return dict(n=int(n),mean=float(x.mean()),sem=float(x.std(ddof=1)/math.sqrt(n)),sd=float(x.std(ddof=1)))
out={'all':{'pull_p':st(D.pull_p),'pull_k':st(D.pull_k),'pull_g':st(D.pull_g)},
     'evaluated':{'pull_p':st(D.pull_p[D.evaluated]),'pull_k':st(D.pull_k[D.evaluated]),'z_true':st(D.z_true[D.evaluated]),'dL':st(D.dL[D.evaluated]),'SNR':st(D.SNR[D.evaluated])},
     'excluded':{'pull_p':st(D.pull_p[~D.evaluated]),'pull_k':st(D.pull_k[~D.evaluated]),'z_true':st(D.z_true[~D.evaluated]),'dL':st(D.dL[~D.evaluated]),'SNR':st(D.SNR[~D.evaluated])},
     'n_evaluated':int(D.evaluated.sum()),'n_total':len(D)}
# exclusion vs d_L: fraction evaluated by d_L quartile
q=pd.qcut(D.dL,4,labels=False)
out['eval_frac_by_dL_quartile']={str(i):float(D.evaluated[q==i].mean()) for i in range(4)}
out['eval_frac_by_ztrue_bin']={f'{a}-{b}':float(D.evaluated[(D.z_true>=a)&(D.z_true<b)].mean()) for a,b in [(0,0.075),(0.075,0.15),(0.15,0.25),(0.25,0.4),(0.4,2)]}
out['eval_frac_by_SNR_quartile']={str(i):float(D.evaluated[pd.qcut(D.SNR,4,labels=False)==i].mean()) for i in range(4)}
D.to_csv(SP/'f6_alldraws.csv',index=False)
json.dump(out,open(SP/'f6_out.json','w'),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))
