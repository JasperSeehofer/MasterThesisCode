"""Zero-re-evaluation venue-law forensic: pull test + per-host kernel moments + mechanism-(A) prediction.
Uses only: banked CRB/diag CSVs, the pinned catalogue (md5 verified), the completeness cache and the S_bar_phi table
built by the harness's own build_bsel_selection_objects (PA-HIER-30 precedent). No evaluate() call."""
import numpy as np, pandas as pd, json, math, time, sys
from pathlib import Path
from scipy.stats import norm
sys.path.insert(0,'/home/jasper/Repositories/darksiren-emri')
import darksiren_emri.validation.correspondence_1d as c1d
from darksiren_emri.bayesian_inference.bayesian_statistics import _host_pixels, _completeness_at_host_nodes
from darksiren_emri.physical_relations import dist_vectorized, comoving_volume_element
SP=Path('/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/70977a05-4e21-4daa-91f0-d0330808c1ea/scratchpad')
A=pd.read_csv(SP/'f1_events.csv')
t0=time.time()
assert c1d.check_reduced_catalogue_pin(), 'catalogue md5 pin mismatch -- STOP'
handler=c1d._load_galaxy_catalog_handler(c1d.REDUCED_CATALOGUE_PATH)
pool=c1d._host_pool_from_handler(handler)
print('handler loaded', time.time()-t0, 'pool n', pool.n, flush=True)
t1=time.time()
completeness, phi_table = c1d.build_bsel_selection_objects(h_true=c1d.H_TRUE)
print('selection objects built', time.time()-t1, flush=True)
zg_grid, s_phi = phi_table[c1d.H_TRUE]
H=c1d.H_TRUE
idx=A.host_galaxy_index.to_numpy().astype(int)
z_g=pool.z[idx]; sig=pool.z_error[idx]; phiS=pool.phiS[idx]; qS=pool.qS[idx]; M_g=pool.M[idx]
A['z_g']=z_g; A['sigma_g']=sig; A['M_g']=M_g
A['pull']=(A.z_true-z_g)/sig
# index-mapping validation: z_true must lie in the host's own +-4 sigma window (floored)
lower,upper=c1d._host_kernel_window(z_g, c1d.host_z_error_eff(z_g,sig))
A['in_window']=(A.z_true>=lower-1e-12)&(A.z_true<=upper+1e-12)
# s_tilde consistency check: recompute S~_g with kernel_smeared_survival and compare to CRB column
st=c1d.kernel_smeared_survival(z_g,sig,phi_table,completeness,phiS,qS,h=H)
A['s_tilde_recomp']=st
print('in_window all', bool(A.in_window.all()), 'max rel diff s_tilde', float(np.max(np.abs(st-A.s_tilde_phi_host)/A.s_tilde_phi_host)), flush=True)
# per-host kernel moments on the host window: k = N*w_pop*f_k ; p = k*S_bar_phi
hp=_host_pixels(completeness,phiS,qS)
NG=4001
rows=[]
rng=np.random.default_rng(20260829)
NMC=400
sigdL=np.sqrt(A.delta_luminosity_distance_delta_luminosity_distance.to_numpy())
dobs=A.luminosity_distance.to_numpy()
def q_g(d, zgrid, kern_norm, s_dl):
    """per-host 'off' numerator: int N(d; d(z), s_dl) k(z) dz on the host grid (k normalized on the grid)."""
    dz=dist_vectorized(zgrid,h=H)
    like=np.exp(-0.5*((d[:,None]-dz[None,:])/s_dl)**2)
    return (like*kern_norm[None,:]).sum(axis=1)
for i in range(len(A)):
    lo,hi=lower[i],upper[i]
    zz=np.linspace(lo,hi,NG)
    w=_completeness_at_host_nodes(completeness, zz[None,:], hp[i:i+1], H)[0]
    zoa = not np.any(w>0)
    if zoa: w=np.ones_like(zz)
    wpop=np.asarray(comoving_volume_element(zz,h=H))/(1+zz)
    gauss=norm.pdf(zz,loc=z_g[i],scale=sig[i])
    k=gauss*wpop*w; k/=np.trapezoid(k,zz)
    sb=np.interp(zz,zg_grid,s_phi)
    p=k*sb; p/=np.trapezoid(p,zz)
    mk=np.trapezoid(zz*k,zz); vk=np.trapezoid((zz-mk)**2*k,zz)
    mp=np.trapezoid(zz*p,zz); vp=np.trapezoid((zz-mp)**2*p,zz)
    # log-slope of S_bar_phi across the kernel
    dlnS=np.gradient(np.log(np.clip(sb,1e-300,None)),zz)
    slope_k=np.trapezoid(dlnS*k,zz)
    # mechanism-(A) prediction for THIS event's realized d_obs: theta-secants of the true-host 'off' numerator
    def lnq(b,s):
        zc=z_g[i]+b*(1+z_g[i]); sc=s*sig[i]
        lo2=max(zc-4*sc,1e-6); hi2=zc+4*sc
        z2=np.linspace(lo2,hi2,NG)
        w2=_completeness_at_host_nodes(completeness, z2[None,:], hp[i:i+1], H)[0]
        if zoa or not np.any(w2>0): w2=np.ones_like(z2)
        k2=norm.pdf(z2,loc=zc,scale=sc)*np.asarray(comoving_volume_element(z2,h=H))/(1+z2)*w2
        k2/=np.trapezoid(k2,z2)
        v=q_g(np.array([dobs[i]]),z2,k2*(z2[1]-z2[0]),sigdL[i])[0]
        return math.log(v) if v>0 else -np.inf
    sb_pred=(lnq(0.02,1)-lnq(-0.02,1))/0.04
    ss_pred=(lnq(0,math.sqrt(2))-lnq(0,1/math.sqrt(2)))/(math.sqrt(2)-1/math.sqrt(2))
    # expectation of the secants under the GENERATOR law (z~p, d~N(d(z),sig_dL)) and under the NULL law (z~k): MC
    cdf_p=np.cumsum(p); cdf_p/=cdf_p[-1]; cdf_k=np.cumsum(k); cdf_k/=cdf_k[-1]
    u=rng.uniform(size=NMC)
    z_p=np.interp(u,cdf_p,zz); z_k=np.interp(u,cdf_k,zz)
    d_p=dist_vectorized(z_p,h=H)+rng.normal(size=NMC)*sigdL[i]
    d_k=dist_vectorized(z_k,h=H)+rng.normal(size=NMC)*sigdL[i]
    def lnq_vec(d,b,s):
        zc=z_g[i]+b*(1+z_g[i]); sc=s*sig[i]
        lo2=max(zc-4*sc,1e-6); hi2=zc+4*sc
        z2=np.linspace(lo2,hi2,801)
        w2=_completeness_at_host_nodes(completeness, z2[None,:], hp[i:i+1], H)[0]
        if zoa or not np.any(w2>0): w2=np.ones_like(z2)
        k2=norm.pdf(z2,loc=zc,scale=sc)*np.asarray(comoving_volume_element(z2,h=H))/(1+z2)*w2
        k2/=np.trapezoid(k2,z2)
        v=q_g(d,z2,k2*(z2[1]-z2[0]),sigdL[i]); return np.log(np.clip(v,1e-300,None))
    Eb_gen=np.mean((lnq_vec(d_p,0.02,1)-lnq_vec(d_p,-0.02,1))/0.04)
    Es_gen=np.mean((lnq_vec(d_p,0,math.sqrt(2))-lnq_vec(d_p,0,1/math.sqrt(2)))/0.70710678)
    Eb_null=np.mean((lnq_vec(d_k,0.02,1)-lnq_vec(d_k,-0.02,1))/0.04)
    Es_null=np.mean((lnq_vec(d_k,0,math.sqrt(2))-lnq_vec(d_k,0,1/math.sqrt(2)))/0.70710678)
    rows.append(dict(mu_k=mk,sd_k=math.sqrt(vk),mu_p=mp,sd_p=math.sqrt(vp),zoa=zoa,slope_lnS_k=slope_k,
                     sb_pred_truehost=sb_pred,ss_pred_truehost=ss_pred,Eb_gen=Eb_gen,Es_gen=Es_gen,Eb_null=Eb_null,Es_null=Es_null,
                     sig_dL_rel=sigdL[i]/dobs[i]))
    if i%50==0: print(i, time.time()-t0, flush=True)
B=pd.DataFrame(rows,index=A.index)
A=pd.concat([A,B],axis=1)
A['b_true_eff']=(A.mu_p-A.mu_k)/(1+A.z_g)
A['s_true_eff']=A.sd_p/A.sd_k
A.to_csv(SP/'f3_events.csv')
M=A[~A.dark]
def st(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; return dict(n=int(len(x)),mean=float(x.mean()),sem=float(x.std(ddof=1)/math.sqrt(len(x))),Z=float(x.mean()/(x.std(ddof=1)/math.sqrt(len(x)))))
out={
 'pull_all':st(A.pull),'pull_matched':st(M.pull),'pull_sd_all':float(A.pull.std(ddof=1)),
 'pull_by_zbin':{},
 'b_true_eff':st(A.b_true_eff),'s_true_eff':st(A.s_true_eff),'s_true_eff_median':float(A.s_true_eff.median()),
 'slope_lnS_k':st(A.slope_lnS_k),
 'pred_truehost_sb_matched':st(M.sb_pred_truehost),'pred_truehost_ss_matched':st(M.ss_pred_truehost),
 'meas_sb_cat_matched':st(M.sb_cat),'meas_ss_cat_matched':st(M.ss_cat),'meas_sb_nb_matched':st(M.sb_nb),'meas_ss_nb_matched':st(M.ss_nb),
 'E_gen_sb':st(A.Eb_gen),'E_gen_ss':st(A.Es_gen),'E_null_sb':st(A.Eb_null),'E_null_ss':st(A.Es_null),
 'corr_sb_pred_vs_meas_cat':float(np.corrcoef(M.sb_pred_truehost,M.sb_cat)[0,1]),
 'corr_ss_pred_vs_meas_cat':float(np.corrcoef(M.ss_pred_truehost,M.ss_cat)[0,1]),
 'sign_agree_sb':float(np.mean(np.sign(M.sb_pred_truehost)==np.sign(M.sb_cat))),
 'sign_agree_ss':float(np.mean(np.sign(M.ss_pred_truehost)==np.sign(M.ss_cat))),
 'n_zoa':int(A.zoa.sum()),'in_window_all':bool(A.in_window.all()),'s_tilde_max_rel':float(np.max(np.abs(A.s_tilde_recomp-A.s_tilde_phi_host)/A.s_tilde_phi_host)),
 'sigma_g_over_1pz_median':float((A.sigma_g/(1+A.z_g)).median()),'sig_dL_rel_median':float(A.sig_dL_rel.median()),
}
edges=[0.0,0.075,0.392,0.559,0.659,0.753,1.018,2.0]
zb=pd.cut(A.z_g,edges,labels=False)
for b in sorted(zb.dropna().unique()):
    m=zb==b
    out['pull_by_zbin'][f'{edges[int(b)]}-{edges[int(b)+1]}']={'n':int(m.sum()),'pull':st(A.pull[m]),'b_true_eff':st(A.b_true_eff[m]),'s_true_eff':st(A.s_true_eff[m]),'slope_lnS':st(A.slope_lnS_k[m]),'E_gen_sb':st(A.Eb_gen[m]),'E_gen_ss':st(A.Es_gen[m])}
json.dump(out,open(SP/'f3_out.json','w'),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))
