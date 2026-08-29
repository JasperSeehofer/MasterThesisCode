"""Twin experiments: (E0) production ball (1.5 sigma sky, +-1 sigma_g z-widening) [must reproduce f7]; (E1) enlarged ball (3.0 sigma sky,
+-4 sigma_g widening); (E2) E1 + S_bar_phi inside the numerator ('phi' form); (E3) the pool-level normalizer secant C_theta = mean over the
DRAWN hosts of the secant of ln S~_g(theta) (the missing site-2.3 term for the no-BH leg)."""
import numpy as np, pandas as pd, json, math, sys, time
from pathlib import Path
from scipy.stats import norm
sys.path.insert(0,'/home/jasper/Repositories/darksiren-emri')
import darksiren_emri.validation.correspondence_1d as c1d
from darksiren_emri.bayesian_inference.bayesian_statistics import _host_pixels, _completeness_at_host_nodes, _GL_NODES_50, _GL_WEIGHTS_50
from darksiren_emri.physical_relations import dist_vectorized, comoving_volume_element, dist_to_redshift, get_redshift_outer_bounds
from darksiren_emri.emri_rate import R_eff_per_mbh
from darksiren_emri.galaxy_catalogue.handler import _polar_to_cartesian, InternalCatalogColumns as IC
SP=Path('/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/70977a05-4e21-4daa-91f0-d0330808c1ea/scratchpad')
RUN=Path('/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run')
H=c1d.H_TRUE
A=pd.read_csv(SP/'f7_events.csv',index_col=0)
handler=c1d._load_galaxy_catalog_handler(c1d.REDUCED_CATALOGUE_PATH); pool=c1d._host_pool_from_handler(handler)
completeness,phi_table=c1d.build_bsel_selection_objects(h_true=H); zg_grid,s_phi=phi_table[H]
df=handler.reduced_galaxy_catalog
CZ=df[IC.REDSHIFT].to_numpy(float); CZE=df[IC.REDSHIFT_ERROR].to_numpy(float); CPHI=df[IC.PHI_S].to_numpy(float); CQ=df[IC.THETA_S].to_numpy(float); CM=df[IC.BH_MASS].to_numpy(float)
NODES={'truth':(0.0,1.0),'b_plus':(0.02,1.0),'b_minus':(-0.02,1.0),'s_plus':(0.0,math.sqrt(2)),'s_minus':(0.0,1/math.sqrt(2))}
def gl(lo,hi):
    half=0.5*(hi-lo); mid=0.5*(hi+lo); return mid[:,None]+half[:,None]*_GL_NODES_50[None,:], half
def numerators(zg,sg,pix,phis,qs,b,s,ev,cov_inv,lognorm,y_num,half_num,d_num_frac,f_num_shared,sphi_num):
    hz=zg+b*(1+zg); se=s*sg
    lo=np.maximum(hz-4*se,1e-6); hi=hz+4*se
    y,half=gl(lo,hi)
    f_den=_completeness_at_host_nodes(completeness,y,pix,H); zoa=~np.any(f_den>0,axis=1); f_den[zoa,:]=1.0
    wpop=np.asarray(comoving_volume_element(y.reshape(-1),h=H)).reshape(y.shape)/(1+y)*f_den
    g=norm.pdf(y,loc=hz[:,None],scale=se[:,None]); Z=(g*wpop*_GL_WEIGHTS_50[None,:]).sum(axis=1)*half; Z=np.where(Z<=0,1.0,Z)
    f_num=f_num_shared.copy(); f_num[zoa,:]=1.0
    wpop_num=(np.asarray(comoving_volume_element(y_num,h=H))/(1+y_num))[None,:]*f_num
    prior=norm.pdf(y_num[None,:],loc=hz[:,None],scale=se[:,None])*wpop_num/Z[:,None]
    n=len(zg); x=np.empty((n,50,3)); x[:,:,0]=phis[:,None]; x[:,:,1]=qs[:,None]; x[:,:,2]=d_num_frac[None,:]
    diff=x.reshape(-1,3)-ev; maha=np.sum(diff@cov_inv*diff,axis=-1); gw=np.exp(lognorm-0.5*maha).reshape(n,50)
    integ=gw*prior*_GL_WEIGHTS_50[None,:]
    return integ.sum(axis=1)*half_num, (integ*sphi_num[None,:]).sum(axis=1)*half_num
crbs={seed:pd.read_csv(RUN/f's0a_seed{seed}/node_truth_sites2.2_nosmear/simulations/cramer_rao_bounds.csv') for seed in [900101,900102,900103,900104]}
MODES={'E0_prod':(1.5,1.0),'E1_wide':(3.0,4.0)}
results={m:[] for m in MODES}
t0=time.time()
for k,(i,r) in enumerate(A.iterrows()):
    seed=int(r.seed); row=crbs[seed].iloc[int(r.event_idx)]
    d=float(row.luminosity_distance); sd=math.sqrt(row.delta_luminosity_distance_delta_luminosity_distance)
    phi=float(row.phiS); th=float(row.qS); pe=math.sqrt(row.delta_phiS_delta_phiS); te=math.sqrt(row.delta_qS_delta_qS)
    ctp=float(row.delta_phiS_delta_qS); cdp=float(row.delta_phiS_delta_luminosity_distance); cdt=float(row.delta_qS_delta_luminosity_distance)
    cov3=np.array([[pe**2,ctp,cdp/d],[ctp,te**2,cdt/d],[cdp/d,cdt/d,sd**2/d**2]]); cov_inv=np.linalg.pinv(cov3); _,logdet=np.linalg.slogdet(cov3); lognorm=-0.5*(3*math.log(2*math.pi)+logdet)
    ev=np.array([phi,th,1.0])
    zmin,zmax=get_redshift_outer_bounds(d,sd,h_min=0.50,h_max=0.86); zmax=min(zmax,1.5)
    S=np.array([[pe**2,ctp],[ctp,te**2]]); J=np.diag([abs(math.sin(th)),1.0]); lam=float(np.linalg.eigvalsh(J@S@J.T).max())
    qp=_polar_to_cartesian(np.array([th]),np.array([phi]))
    zlo=float(dist_to_redshift(d-4*sd,h=H)); zhi=float(dist_to_redshift(d+4*sd,h=H)); y_num,half_num=gl(np.array([zlo]),np.array([zhi])); y_num=y_num[0]; half_num=float(half_num[0])
    d_num_frac=np.asarray(dist_vectorized(y_num,h=H))/d; sphi_num=np.interp(y_num,zg_grid,s_phi)
    for mode,(mult,kz) in MODES.items():
        R=mult*math.sqrt(max(lam,0.0)); ind=handler.catalog_ball_tree.query_radius(qp,r=R)[0]
        zg=CZ[ind]; sg=CZE[ind]; m=(zg+kz*sg>=zmin)&(zg-kz*sg<=zmax); ind=ind[m]
        out={'n_cand':int(len(ind))}
        if len(ind)==0:
            results[mode].append(out); continue
        zg=CZ[ind]; sg=CZE[ind]; phis=CPHI[ind]; qs=CQ[ind]; Mg=CM[ind]; wg=np.asarray(R_eff_per_mbh(Mg),dtype=float)/(1+zg)
        pix=_host_pixels(completeness,phis,qs)
        f_num_shared=_completeness_at_host_nodes(completeness,np.broadcast_to(y_num[None,:],(len(zg),50)).copy(),pix,H)
        for name,(b,s) in NODES.items():
            Noff,Nphi=numerators(zg,sg,pix,phis,qs,b,s,ev,cov_inv,lognorm,y_num,half_num,d_num_frac,f_num_shared,sphi_num)
            to=float((wg*Noff).sum()); tp=float((wg*Nphi).sum())
            out[f'off_{name}']=math.log(to) if to>0 else float('nan'); out[f'phi_{name}']=math.log(tp) if tp>0 else float('nan')
        results[mode].append(out)
    if k%50==0: print(k,time.time()-t0,{m:results[m][-1]['n_cand'] for m in MODES},flush=True)
res={}
for mode in MODES:
    T=pd.DataFrame(results[mode],index=A.index)
    for form in ('off','phi'):
        T[f'{form}_sb']=(T[f'{form}_b_plus']-T[f'{form}_b_minus'])/0.04; T[f'{form}_ss']=(T[f'{form}_s_plus']-T[f'{form}_s_minus'])/(math.sqrt(2)-1/math.sqrt(2))
    T.to_csv(SP/f'f8_{mode}.csv')
    M=T[~A.dark.to_numpy()]
    def st(x):
        x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x); return dict(n=int(n),mean=float(x.mean()),sem=float(x.std(ddof=1)/math.sqrt(n)))
    res[mode]={'n_cand_median':float(T.n_cand.median()),'off_sb':st(M.off_sb),'off_ss':st(M.off_ss),'phi_sb':st(M.phi_sb),'phi_ss':st(M.phi_ss),
               'off_sb_x_c':st(M.off_sb*A.c_nb[~A.dark]),'off_ss_x_c':st(M.off_ss*A.c_nb[~A.dark]),'phi_sb_x_c':st(M.phi_sb*A.c_nb[~A.dark]),'phi_ss_x_c':st(M.phi_ss*A.c_nb[~A.dark])}
    if mode=='E0_prod':
        res[mode]['check_vs_measured']={'sb_max_abs_diff':float((M.off_sb-A.sb_cat[~A.dark]).abs().max()),'ss_max_abs_diff':float((M.off_ss-A.ss_cat[~A.dark]).abs().max())}
    zg_edges=[0.0,0.075,0.15,0.25,0.392,2.0]; zb=pd.cut(A.z_g[~A.dark],zg_edges,labels=False)
    res[mode]['by_zg']={f'{zg_edges[int(b)]}-{zg_edges[int(b)+1]}':{'n':int((zb==b).sum()),'off_sb':st(M.off_sb[(zb==b).to_numpy()]),'phi_sb':st(M.phi_sb[(zb==b).to_numpy()]),'off_ss':st(M.off_ss[(zb==b).to_numpy()]),'phi_ss':st(M.phi_ss[(zb==b).to_numpy()])} for b in sorted(zb.dropna().unique())}
# E3: pool-level normalizer secant from the drawn hosts (all 800 and the 461 evaluated)
D=pd.read_csv(SP/'f6_alldraws.csv')
idx=D.host.to_numpy().astype(int); z=pool.z[idx]; ze=pool.z_error[idx]; ph=pool.phiS[idx]; q=pool.qS[idx]
lnS={}
for name,(b,s) in NODES.items():
    st_=c1d.kernel_smeared_survival(z+b*(1+z), s*ze, phi_table, completeness, ph, q, h=H); lnS[name]=np.log(st_)
Cb=(lnS['b_plus']-lnS['b_minus'])/0.04; Cs=(lnS['s_plus']-lnS['s_minus'])/(math.sqrt(2)-1/math.sqrt(2))
ev=D.evaluated.to_numpy()
res['E3_pool_normalizer_secant']={'C_b_all800':{'mean':float(Cb.mean()),'sem':float(Cb.std(ddof=1)/math.sqrt(len(Cb)))},'C_s_all800':{'mean':float(Cs.mean()),'sem':float(Cs.std(ddof=1)/math.sqrt(len(Cs)))},
    'C_b_evaluated':{'mean':float(Cb[ev].mean()),'sem':float(Cb[ev].std(ddof=1)/math.sqrt(ev.sum()))},'C_s_evaluated':{'mean':float(Cs[ev].mean()),'sem':float(Cs[ev].std(ddof=1)/math.sqrt(ev.sum()))},
    'check_s_tilde_truth_vs_crb_max_rel':float(np.max(np.abs(np.exp(lnS['truth'])-D.s_tilde_phi_host if 's_tilde_phi_host' in D else 0)))}
json.dump(res,open(SP/'f8_out.json','w'),indent=1,default=float)
print(json.dumps(res,indent=1,default=float))
