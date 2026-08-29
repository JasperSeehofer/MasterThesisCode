"""Per-event mechanism test (zero re-evaluation): catalogue share c_i, GW-precise true-host analytic scores,
deterministic venue-law expectations of the registered secants, z_g-binned prediction vs measurement."""
import numpy as np, pandas as pd, json, math, sys, time
from pathlib import Path
from scipy.stats import norm
sys.path.insert(0,'/home/jasper/Repositories/darksiren-emri')
import darksiren_emri.validation.correspondence_1d as c1d
from darksiren_emri.bayesian_inference.bayesian_statistics import _host_pixels, _completeness_at_host_nodes
from darksiren_emri.physical_relations import dist_vectorized, comoving_volume_element, dist_to_redshift
SP=Path('/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/70977a05-4e21-4daa-91f0-d0330808c1ea/scratchpad')
A=pd.read_csv(SP/'f3_events.csv',index_col=0)
H=c1d.H_TRUE
# catalogue share at truth (no-BH): combined = (beta L_cat + B_num)/D~  => c = 1 - B_num/(combined*D~)
A['c_nb']=1.0-A.B_num/(np.exp(A.ln_nb_truth)*A.D_tilde_phi)
A['c_wb']=np.nan
# z_GW from the observed d_L at h=0.73
A['z_GW']=[float(dist_to_redshift(d,h=H)) for d in A.luminosity_distance]
A['sig_zGW']=[float(dist_to_redshift(d+s,h=H)-dist_to_redshift(d,h=H)) for d,s in zip(A.luminosity_distance,np.sqrt(A.delta_luminosity_distance_delta_luminosity_distance))]
# analytic GW-precise true-host first-order scores at the realized data
A['sb_th_analytic']=(1+A.z_g)*(A.z_GW-A.mu_k)/(A.sd_k**2+A.sig_zGW**2)
A['own_dlnS_b']=(1+A.z_g)*(A.mu_p-A.mu_k)/A.sd_k**2   # = d/db ln S~_g(b) at b=0 (Gaussian-kernel approx)
# deterministic expectations of the registered secants under the venue's own kernel (null) and generator law (gen), GW-precise
assert c1d.check_reduced_catalogue_pin()
handler=c1d._load_galaxy_catalog_handler(c1d.REDUCED_CATALOGUE_PATH); pool=c1d._host_pool_from_handler(handler)
completeness, phi_table = c1d.build_bsel_selection_objects(h_true=H); zg_grid,s_phi=phi_table[H]
idx=A.host_galaxy_index.to_numpy().astype(int); z_g=pool.z[idx]; sig=pool.z_error[idx]; phiS=pool.phiS[idx]; qS=pool.qS[idx]
hp=_host_pixels(completeness,phiS,qS)
NG=4001
def kern(i,b,s,zz):
    zc=z_g[i]+b*(1+z_g[i]); sc=s*sig[i]
    w=_completeness_at_host_nodes(completeness, zz[None,:], hp[i:i+1], H)[0]
    if not np.any(w>0): w=np.ones_like(zz)
    lo=max(zc-4*sc,1e-6); hi=zc+4*sc
    zn=np.linspace(lo,hi,NG)
    wn=_completeness_at_host_nodes(completeness, zn[None,:], hp[i:i+1], H)[0]
    if not np.any(wn>0): wn=np.ones_like(zn)
    Z=np.trapezoid(norm.pdf(zn,loc=zc,scale=sc)*np.asarray(comoving_volume_element(zn,h=H))/(1+zn)*wn, zn)
    k=norm.pdf(zz,loc=zc,scale=sc)*np.asarray(comoving_volume_element(zz,h=H))/(1+zz)*w/Z
    return np.where((zz>=lo)&(zz<=hi),k,0.0)
rows=[]
t0=time.time()
for i in range(len(A)):
    lo=max(z_g[i]-4*sig[i],1e-6); hi=z_g[i]+4*sig[i]
    # evaluation grid covering the truth window (secant nodes' windows are subsets/supersets; ln k outside a node's window = -inf; restrict to the intersection)
    zz=np.linspace(lo,hi,NG)
    k0=kern(i,0,1,zz)
    sb=np.interp(zz,zg_grid,s_phi); p=k0*sb; p/=np.trapezoid(p,zz)
    lk=lambda b,s: np.log(np.clip(kern(i,b,s,zz),1e-300,None))
    secb=(lk(0.02,1)-lk(-0.02,1))/0.04
    secs=(lk(0,math.sqrt(2))-lk(0,1/math.sqrt(2)))/(math.sqrt(2)-1/math.sqrt(2))
    # intersection masks: b-nodes windows both contain z iff z in [lo+0.02(1+zg), hi-0.02(1+zg)] ; s-nodes: s_minus window is the narrow one
    mb=(zz>=max(z_g[i]-0.02*(1+z_g[i])-4*sig[i],1e-6)+0)&(zz<=z_g[i]-0.02*(1+z_g[i])+4*sig[i])&(zz>=max(z_g[i]+0.02*(1+z_g[i])-4*sig[i],1e-6))&(zz<=z_g[i]+0.02*(1+z_g[i])+4*sig[i])
    ms=(zz>=max(z_g[i]-4*sig[i]/math.sqrt(2),1e-6))&(zz<=z_g[i]+4*sig[i]/math.sqrt(2))
    def E(w,f,m):
        w=np.where(m,w,0.0); return float(np.trapezoid(w*f,zz)/np.trapezoid(w,zz))
    rows.append(dict(Eb_null_det=E(k0,secb,mb),Eb_gen_det=E(p,secb,mb),Es_null_det=E(k0,secs,ms),Es_gen_det=E(p,secs,ms),
                     mass_outside_smin=float(1-np.trapezoid(np.where(ms,p,0),zz)/np.trapezoid(p,zz))))
    if i%100==0: print(i,time.time()-t0,flush=True)
B=pd.DataFrame(rows,index=A.index); A=pd.concat([A,B],axis=1)
A.to_csv(SP/'f4_events.csv')
M=A[~A.dark].copy()
def st(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; return dict(n=int(len(x)),mean=float(x.mean()),sem=float(x.std(ddof=1)/math.sqrt(len(x))),Z=float(x.mean()/(x.std(ddof=1)/math.sqrt(len(x)))))
out={}
out['c_nb']={'mean':float(M.c_nb.mean()),'median':float(M.c_nb.median()),'q25':float(M.c_nb.quantile(.25)),'q75':float(M.c_nb.quantile(.75)),'frac_gt_0.5':float((M.c_nb>0.5).mean())}
out['w_G_est']=float(M.w_G.iloc[0])
out['sig_zGW_over_sd_k_median']=float((M.sig_zGW/M.sd_k).median())
# predictions of the pooled combined-channel scores (first order: c_i x true-host expectation)
for lab,col in (('Eb_gen_det','Eb_gen_det'),('Eb_null_det','Eb_null_det'),('own_dlnS_b','own_dlnS_b')):
    out[f'pred_sb_nb_{lab}']=st(M.c_nb*M[col]); out[f'{lab}_unweighted']=st(M[col])
for lab in ('Es_gen_det','Es_null_det'):
    out[f'pred_ss_nb_{lab}']=st(M.c_nb*M[lab]); out[f'{lab}_unweighted']=st(M[lab])
out['meas_sb_nb']=st(M.sb_nb); out['meas_ss_nb']=st(M.ss_nb); out['meas_sb_cat']=st(M.sb_cat); out['meas_ss_cat']=st(M.ss_cat)
# per-event hook-arithmetic check: measured catalogue-leg secant vs analytic true-host score, on true-host-dominated events (c_nb>0.8 & GW-precise)
for tag,sel in (('all',np.ones(len(M),bool)),('c>0.8',(M.c_nb>0.8).to_numpy()),('c>0.95',(M.c_nb>0.95).to_numpy())):
    S=M[sel]; f=np.isfinite(S.sb_th_analytic)&np.isfinite(S.sb_cat)
    if f.sum()>3:
        x=S.sb_th_analytic[f]; y=S.sb_cat[f]
        slope=float(np.polyfit(x,y,1)[0]); r=float(np.corrcoef(x,y)[0,1])
        out[f'hook_check_b_{tag}']={'n':int(f.sum()),'corr':r,'slope_meas_vs_analytic':slope,'mean_meas':float(y.mean()),'mean_analytic':float(x.mean()),'median_abs_ratio':float(np.median(np.abs(y/x)))}
    f=np.isfinite(S.sb_pred_truehost)&np.isfinite(S.sb_cat)&(np.abs(S.sb_pred_truehost)<1e3)
    if f.sum()>3:
        x=S.sb_pred_truehost[f]; y=S.sb_cat[f]
        out[f'hook_check_b_numeric_{tag}']={'n':int(f.sum()),'corr':float(np.corrcoef(x,y)[0,1]),'slope':float(np.polyfit(x,y,1)[0]),'sign_agree':float(np.mean(np.sign(x)==np.sign(y)))}
    f=np.isfinite(S.ss_pred_truehost)&np.isfinite(S.ss_cat)&(np.abs(S.ss_pred_truehost)<1e3)
    if f.sum()>3:
        x=S.ss_pred_truehost[f]; y=S.ss_cat[f]
        out[f'hook_check_s_numeric_{tag}']={'n':int(f.sum()),'corr':float(np.corrcoef(x,y)[0,1]),'slope':float(np.polyfit(x,y,1)[0]),'sign_agree':float(np.mean(np.sign(x)==np.sign(y))),'mean_meas':float(y.mean()),'mean_pred':float(x.mean())}
# z_true-bin selection effect: (z_true - mu_k)/sd_k by z_true bin, and z_g-binned measurement vs prediction
edges=[0.0,0.075,0.392,0.559,0.659,0.753,1.018,2.0]
out['ztrue_bins']={}
zb=pd.cut(M.z_true,edges,labels=False)
for b in sorted(zb.dropna().unique()):
    m=(zb==b).to_numpy(); S=M[m]
    out['ztrue_bins'][f'{edges[int(b)]}-{edges[int(b)+1]}']={'n':int(m.sum()),'pull_vs_mu_k':st((S.z_true-S.mu_k)/S.sd_k),'sb_nb':st(S.sb_nb),'sb_th_analytic_x_c':st(S.c_nb*S.sb_th_analytic),'c_nb_mean':float(S.c_nb.mean()),'ss_nb':st(S.ss_nb),'Es_null_x_c':st(S.c_nb*S.Es_null_det)}
out['zg_bins']={}
zg_edges=[0.0,0.075,0.15,0.25,0.392,2.0]
zb=pd.cut(M.z_g,zg_edges,labels=False)
for b in sorted(zb.dropna().unique()):
    m=(zb==b).to_numpy(); S=M[m]
    out['zg_bins'][f'{zg_edges[int(b)]}-{zg_edges[int(b)+1]}']={'n':int(m.sum()),'sb_nb':st(S.sb_nb),'pred_sb_nb_gen':st(S.c_nb*S.Eb_gen_det),'pred_sb_nb_null':st(S.c_nb*S.Eb_null_det),'ss_nb':st(S.ss_nb),'pred_ss_nb_gen':st(S.c_nb*S.Es_gen_det),'pred_ss_nb_null':st(S.c_nb*S.Es_null_det),'c_nb_mean':float(S.c_nb.mean()),'slope_lnS':st(S.slope_lnS_k)}
# by catalogue share
out['c_bins']={}
cb=pd.cut(M.c_nb,[-0.01,0.2,0.5,0.8,0.95,1.01],labels=False)
for b in sorted(cb.dropna().unique()):
    m=(cb==b).to_numpy(); S=M[m]
    out['c_bins'][str(b)]={'n':int(m.sum()),'c_range':(float(S.c_nb.min()),float(S.c_nb.max())),'sb_nb':st(S.sb_nb),'ss_nb':st(S.ss_nb),'sb_cat':st(S.sb_cat),'ss_cat':st(S.ss_cat),'pred_sb_gen':st(S.c_nb*S.Eb_gen_det),'pred_ss_null':st(S.c_nb*S.Es_null_det),'pred_ss_gen':st(S.c_nb*S.Es_gen_det),'mean_z_g':float(S.z_g.mean())}
out['mass_outside_smin_window']=st(M.mass_outside_smin)
json.dump(out,open(SP/'f4_out.json','w'),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))
