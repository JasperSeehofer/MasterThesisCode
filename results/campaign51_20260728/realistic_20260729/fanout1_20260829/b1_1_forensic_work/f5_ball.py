"""True-host candidate-ball inclusion (no-BH channel) reproduced from the estimator's own window rules, and score splits."""
import numpy as np, pandas as pd, json, math, sys
from pathlib import Path
sys.path.insert(0,'/home/jasper/Repositories/darksiren-emri')
import darksiren_emri.validation.correspondence_1d as c1d
from darksiren_emri.physical_relations import dist_to_redshift
from darksiren_emri.galaxy_catalogue.handler import _polar_to_cartesian
SP=Path('/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/70977a05-4e21-4daa-91f0-d0330808c1ea/scratchpad')
A=pd.read_csv(SP/'f4_events.csv',index_col=0)
RUN=Path('/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run')
handler=c1d._load_galaxy_catalog_handler(c1d.REDUCED_CATALOGUE_PATH); pool=c1d._host_pool_from_handler(handler)
# per-event Fisher sky block from the CRB rows
parts=[]
for seed in [900101,900102,900103,900104]:
    crb=pd.read_csv(RUN/f's0a_seed{seed}/node_truth_sites2.2_nosmear/simulations/cramer_rao_bounds.csv')
    sub=A[A.seed==seed]
    r=crb.iloc[sub.event_idx.to_numpy().astype(int)]
    d=pd.DataFrame(index=sub.index)
    d['phi_var']=r.delta_phiS_delta_phiS.to_numpy(); d['th_var']=r.delta_qS_delta_qS.to_numpy(); d['cov']=r.delta_phiS_delta_qS.to_numpy()
    parts.append(d)
F=pd.concat(parts); A=pd.concat([A,F],axis=1)
idx=A.host_galaxy_index.to_numpy().astype(int)
hphi=pool.phiS[idx]; hq=pool.qS[idx]
# sky cone: radius = 1.5*sqrt(lambda_max(J S J^T)), J=diag(|sin theta_obs|,1); chord distance in the Cartesian embedding
rad=np.empty(len(A)); chord=np.empty(len(A))
for i,(ph,th,pv,tv,cv) in enumerate(zip(A.phiS,A.qS,A.phi_var,A.th_var,A['cov'])):
    S=np.array([[pv,cv],[cv,tv]]); J=np.diag([abs(math.sin(th)),1.0]); lam=float(np.linalg.eigvalsh(J@S@J.T).max())
    rad[i]=1.5*math.sqrt(max(lam,0.0))
    q=_polar_to_cartesian(np.array([th]),np.array([ph]))[0]; hcart=_polar_to_cartesian(np.array([hq[i]]),np.array([hphi[i]]))[0]
    chord[i]=float(np.linalg.norm(q-hcart))
A['sky_in']=chord<=rad; A['chord_over_rad']=chord/rad
sd=np.sqrt(A.delta_luminosity_distance_delta_luminosity_distance.to_numpy()); dL=A.luminosity_distance.to_numpy()
zmin=np.array([0.0 if d-3*s<0 else float(dist_to_redshift(d-3*s,0.50)) for d,s in zip(dL,sd)])
zmax=np.minimum(np.array([float(dist_to_redshift(d+3*s,0.86)) for d,s in zip(dL,sd)]),1.5)
A['z_min_ball']=zmin; A['z_max_ball']=zmax
A['z_in']=(A.z_g+A.sigma_g>=zmin)&(A.z_g-A.sigma_g<=zmax)
A['recovered']=A.sky_in&A.z_in
A['c_wb']=1.0-A.B_num_wbh/(np.exp(A.ln_wb_truth)*A.D_tilde_phi)
A.to_csv(SP/'f5_events.csv')
def st(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x)
    return dict(n=int(n),mean=float(x.mean()) if n else None,sem=float(x.std(ddof=1)/math.sqrt(n)) if n>1 else None,Z=float(x.mean()/(x.std(ddof=1)/math.sqrt(n))) if n>1 and x.std(ddof=1)>0 else None)
M=A[~A.dark]
out={'recovery':{str(s):{'n':int((A.seed==s).sum()),'sky_in':int(A[A.seed==s].sky_in.sum()),'z_in':int(A[A.seed==s].z_in.sum()),'recovered':int(A[A.seed==s].recovered.sum())} for s in [900101,900102,900103,900104]}}
out['recovery_pooled']={'n':len(A),'sky_in_frac':float(A.sky_in.mean()),'z_in_frac':float(A.z_in.mean()),'recovered_frac':float(A.recovered.mean())}
out['z_window_rel']={'zmin_over_zGW_median':float((A.z_min_ball/A.z_GW).median()),'zmax_over_zGW_median':float((A.z_max_ball/A.z_GW).median()),'halfwidth_over_sigma_g_median':float(((A.z_max_ball-A.z_min_ball)/2/A.sigma_g).median())}
for lab,m in (('recovered',M.recovered),('not_recovered',~M.recovered),('sky_out',~M.sky_in),('z_out',~M.z_in)):
    S=M[m.to_numpy()]
    out[f'scores_{lab}']={'n':len(S),'sb_nb':st(S.sb_nb),'ss_nb':st(S.ss_nb),'sb_cat':st(S.sb_cat),'ss_cat':st(S.ss_cat),'c_nb_mean':float(S.c_nb.mean()) if len(S) else None,'z_g_mean':float(S.z_g.mean()) if len(S) else None,'pull_vs_mu_k':st((S.z_true-S.mu_k)/S.sd_k)}
# z_g bins x recovered
zg_edges=[0.0,0.075,0.15,0.25,0.392,2.0]; zb=pd.cut(M.z_g,zg_edges,labels=False)
out['zg_x_recovered']={}
for b in sorted(zb.dropna().unique()):
    for lab,m in (('rec',M.recovered),('not',~M.recovered)):
        S=M[((zb==b)&m).to_numpy()]
        out['zg_x_recovered'][f'{zg_edges[int(b)]}-{zg_edges[int(b)+1]}_{lab}']={'n':len(S),'sb_nb':st(S.sb_nb),'ss_nb':st(S.ss_nb),'sb_cat':st(S.sb_cat),'c':float(S.c_nb.mean()) if len(S) else None}
# contributions to the pooled sums
tot_b=float(np.nansum(M.sb_nb)); tot_s=float(np.nansum(M.ss_nb))
out['sum_share']={'sb_nb_total':tot_b,'sb_nb_from_not_recovered':float(np.nansum(M.sb_nb[~M.recovered])),'sb_nb_from_zg_lt_0.15':float(np.nansum(M.sb_nb[M.z_g<0.15])),'ss_nb_total':tot_s,'ss_nb_from_not_recovered':float(np.nansum(M.ss_nb[~M.recovered])),'ss_nb_from_zg_lt_0.15':float(np.nansum(M.ss_nb[M.z_g<0.15]))}
# realized pull vs kernel mean by z_g bin (generator sanity)
out['pull_mu_k_by_zg']={f'{zg_edges[int(b)]}-{zg_edges[int(b)+1]}':st(((A.z_true-A.mu_k)/A.sd_k)[(pd.cut(A.z_g,zg_edges,labels=False)==b).to_numpy()]) for b in sorted(zb.dropna().unique())}
out['pull_mu_k_all']=st((A.z_true-A.mu_k)/A.sd_k)
out['pull_mu_p_all']=st((A.z_true-A.mu_p)/A.sd_p)
out['c_wb']={'mean':float(M.c_wb.mean()),'median':float(M.c_wb.median()),'frac_gt_0.5':float((M.c_wb>0.5).mean())}
out['tilt_mu_k_minus_zg_over_sigma']={f'{zg_edges[int(b)]}-{zg_edges[int(b)+1]}':st(((A.mu_k-A.z_g)/A.sd_k)[(pd.cut(A.z_g,zg_edges,labels=False)==b).to_numpy()]) for b in sorted(zb.dropna().unique())}
json.dump(out,open(SP/'f5_out.json','w'),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))
