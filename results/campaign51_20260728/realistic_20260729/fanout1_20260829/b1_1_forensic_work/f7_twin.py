"""Independent numpy twin of the no-BH catalogue leg (bc flags: numerator 'off', divisor Sigma^phi point) at the 5 theta-nodes.
Reads: CRB rows, pinned catalogue (via the production handler's own candidate search), completeness cache, S_bar_phi table (for the
Sigma^phi constant only). No evaluate() call. Purpose: (i) per-event hook-arithmetic check, (ii) true-host vs impostor decomposition."""
import numpy as np, pandas as pd, json, math, sys, time
from pathlib import Path
from scipy.stats import norm
sys.path.insert(0,'/home/jasper/Repositories/darksiren-emri')
import darksiren_emri.validation.correspondence_1d as c1d
from darksiren_emri.bayesian_inference.bayesian_statistics import _host_pixels, _completeness_at_host_nodes, _GL_NODES_50, _GL_WEIGHTS_50
from darksiren_emri.physical_relations import dist_vectorized, comoving_volume_element, dist_to_redshift, get_redshift_outer_bounds
from darksiren_emri.emri_rate import R_eff_per_mbh
SP=Path('/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/70977a05-4e21-4daa-91f0-d0330808c1ea/scratchpad')
RUN=Path('/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run')
H=c1d.H_TRUE
A=pd.read_csv(SP/'f5_events.csv',index_col=0)
handler=c1d._load_galaxy_catalog_handler(c1d.REDUCED_CATALOGUE_PATH)
completeness,_=c1d.build_bsel_selection_objects(h_true=H)
NODES={'truth':(0.0,1.0),'b_plus':(0.02,1.0),'b_minus':(-0.02,1.0),'s_plus':(0.0,math.sqrt(2)),'s_minus':(0.0,1/math.sqrt(2))}
def gl(lo,hi):
    half=0.5*(hi-lo); mid=0.5*(hi+lo)
    return mid[:,None]+half[:,None]*_GL_NODES_50[None,:], half
def numerators(zg,sg,pix,phis,qs,b,s,ev,cov_inv,lognorm,y_num,d_num_frac,f_num_shared):
    hz=zg+b*(1+zg); se=s*sg
    lo=np.maximum(hz-4*se,1e-6); hi=hz+4*se
    y,half=gl(lo,hi)
    f_den=_completeness_at_host_nodes(completeness,y,pix,H)
    zoa=~np.any(f_den>0,axis=1); f_den[zoa,:]=1.0
    wpop=np.asarray(comoving_volume_element(y.reshape(-1),h=H)).reshape(y.shape)/(1+y)*f_den
    g=norm.pdf(y,loc=hz[:,None],scale=se[:,None])
    Z=(g*wpop*_GL_WEIGHTS_50[None,:]).sum(axis=1)*half; Z=np.where(Z<=0,1.0,Z)
    f_num=f_num_shared.copy(); f_num[zoa,:]=1.0
    wpop_num=(np.asarray(comoving_volume_element(y_num,h=H))/(1+y_num))[None,:]*f_num
    prior=norm.pdf(y_num[None,:],loc=hz[:,None],scale=se[:,None])*wpop_num/Z[:,None]
    n=len(zg); x=np.empty((n,50,3)); x[:,:,0]=phis[:,None]; x[:,:,1]=qs[:,None]; x[:,:,2]=d_num_frac[None,:]
    diff=x.reshape(-1,3)-ev; maha=np.sum(diff@cov_inv*diff,axis=-1); gw=np.exp(lognorm-0.5*maha).reshape(n,50)
    return (gw*prior*_GL_WEIGHTS_50[None,:]).sum(axis=1)*half_num[0]
rows=[]; t0=time.time()
sigma_phi={}
for seed in [900101,900102,900103,900104]:
    sigma_phi[seed]=json.load(open(RUN/f's0a_seed{seed}/node_truth_sites2.2_nosmear/selection_tables_h_0_73.json'))['sigma_phi']
crbs={seed:pd.read_csv(RUN/f's0a_seed{seed}/node_truth_sites2.2_nosmear/simulations/cramer_rao_bounds.csv') for seed in sigma_phi}
for k,(i,r) in enumerate(A.iterrows()):
    seed=int(r.seed); row=crbs[seed].iloc[int(r.event_idx)]
    d=float(row.luminosity_distance); sd=math.sqrt(row.delta_luminosity_distance_delta_luminosity_distance)
    phi=float(row.phiS); th=float(row.qS); pe=math.sqrt(row.delta_phiS_delta_phiS); te=math.sqrt(row.delta_qS_delta_qS)
    ctp=float(row.delta_phiS_delta_qS); cdp=float(row.delta_phiS_delta_luminosity_distance); cdt=float(row.delta_qS_delta_luminosity_distance)
    M=float(row.M); Mu=math.sqrt(row.delta_M_delta_M)
    cov3=np.array([[pe**2,ctp,cdp/d],[ctp,te**2,cdt/d],[cdp/d,cdt/d,sd**2/d**2]])
    cov_inv=np.linalg.pinv(cov3); sgn,logdet=np.linalg.slogdet(cov3); lognorm=-0.5*(3*math.log(2*math.pi)+logdet)
    ev=np.array([phi,th,1.0])
    zmin,zmax=get_redshift_outer_bounds(d,sd,h_min=0.50,h_max=0.86,sigma_multiplier=2.0); zmax=min(zmax,1.5)
    res=handler.get_possible_hosts_from_ball_tree(phi=phi,phi_sigma=pe,theta=th,theta_sigma=te,M_z=M,M_z_sigma=Mu,z_min=zmin,z_max=zmax,sigma_multiplier=1.5,cov_theta_phi=ctp)
    if res is None:
        rows.append(dict(n_cand=0)); continue
    hosts=res[0]
    zg=np.array([h.z for h in hosts]); sg=np.array([h.z_error for h in hosts]); phis=np.array([h.phiS for h in hosts]); qs=np.array([h.qS for h in hosts]); Mg=np.array([h.M for h in hosts]); cidx=np.array([h.catalog_index for h in hosts])
    wg=np.asarray(R_eff_per_mbh(Mg),dtype=float)/(1+zg)
    pix=_host_pixels(completeness,phis,qs)
    zlo=float(dist_to_redshift(d-4*sd,h=H)); zhi=float(dist_to_redshift(d+4*sd,h=H))
    y_num,half_num=gl(np.array([zlo]),np.array([zhi])); y_num=y_num[0]
    d_num_frac=np.asarray(dist_vectorized(y_num,h=H))/d
    f_num_shared=_completeness_at_host_nodes(completeness,np.broadcast_to(y_num[None,:],(len(zg),50)).copy(),pix,H)
    out={'n_cand':len(zg),'true_in':bool(int(r.host_galaxy_index) in set(cidx.tolist()))}
    N={}
    for name,(b,s) in NODES.items():
        N[name]=numerators(zg,sg,pix,phis,qs,b,s,ev,cov_inv,lognorm,y_num,d_num_frac,f_num_shared)
        tot=float((wg*N[name]).sum()); out[f'lnLcat_{name}']=math.log(tot/sigma_phi[seed]) if tot>0 else float('nan')
    # true-host share and decomposition at truth (first-order secant split)
    w0=wg*N['truth']; tot0=w0.sum()
    if out['true_in'] and tot0>0:
        j=int(np.where(cidx==int(r.host_galaxy_index))[0][0]); out['pi_true']=float(w0[j]/tot0)
        # exact secant decomposition: ln L(+) - ln L(-) = ln[sum_g w N_g(+)] - ln[sum_g w N_g(-)]; split by true vs impostor contributions to the ratio
        for ax,(p,m,den) in {'b':('b_plus','b_minus',0.04),'s':('s_plus','s_minus',math.sqrt(2)-1/math.sqrt(2))}.items():
            wp=wg*N[p]; wm=wg*N[m]
            out[f'sec_{ax}_true_only']=(math.log(wp[j])-math.log(wm[j]))/den if wp[j]>0 and wm[j]>0 else float('nan')
            imp=np.ones(len(zg),bool); imp[j]=False
            out[f'sec_{ax}_imp_only']=(math.log(wp[imp].sum())-math.log(wm[imp].sum()))/den if imp.sum()>0 and wp[imp].sum()>0 and wm[imp].sum()>0 else float('nan')
    else:
        out['pi_true']=0.0
    rows.append(out)
    if k%50==0: print(k,time.time()-t0,len(zg),flush=True)
T=pd.DataFrame(rows,index=A.index); A=pd.concat([A,T],axis=1)
A['tw_sb']=(A.lnLcat_b_plus-A.lnLcat_b_minus)/0.04; A['tw_ss']=(A.lnLcat_s_plus-A.lnLcat_s_minus)/(math.sqrt(2)-1/math.sqrt(2))
A.to_csv(SP/'f7_events.csv')
M=A[(~A.dark)&np.isfinite(A.lnLcat_truth)]
def st(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x); return dict(n=int(n),mean=float(x.mean()),sem=float(x.std(ddof=1)/math.sqrt(n)) if n>1 else None,Z=float(x.mean()/(x.std(ddof=1)/math.sqrt(n))) if n>1 else None)
dl=M.lnLcat_truth-M.ln_cat_truth
out={'n_twin':len(M),'lnLcat_truth_residual':{'mean':float(dl.mean()),'sd':float(dl.std()),'max_abs':float(dl.abs().max()),'median_abs':float(dl.abs().median())},
     'sec_b':{'corr':float(np.corrcoef(M.tw_sb,M.sb_cat)[0,1]),'slope':float(np.polyfit(M.tw_sb,M.sb_cat,1)[0]),'max_abs_diff':float((M.tw_sb-M.sb_cat).abs().max()),'median_abs_diff':float((M.tw_sb-M.sb_cat).abs().median()),'mean_twin':float(M.tw_sb.mean()),'mean_meas':float(M.sb_cat.mean())},
     'sec_s':{'corr':float(np.corrcoef(M.tw_ss,M.ss_cat)[0,1]),'slope':float(np.polyfit(M.tw_ss,M.ss_cat,1)[0]),'max_abs_diff':float((M.tw_ss-M.ss_cat).abs().max()),'median_abs_diff':float((M.tw_ss-M.ss_cat).abs().median()),'mean_twin':float(M.tw_ss.mean()),'mean_meas':float(M.ss_cat.mean())},
     'true_in_frac':float(A.true_in.mean()),'n_cand':{'median':float(A.n_cand.median()),'max':int(A.n_cand.max()),'min':int(A.n_cand.min())},
     'pi_true':{'mean':float(M.pi_true.mean()),'median':float(M.pi_true.median()),'frac_gt_0.5':float((M.pi_true>0.5).mean())},
     'decomp_b':{'true_only':st(M.sec_b_true_only),'imp_only':st(M.sec_b_imp_only),'pi_weighted_true':st(M.pi_true*M.sec_b_true_only),'full':st(M.tw_sb)},
     'decomp_s':{'true_only':st(M.sec_s_true_only),'imp_only':st(M.sec_s_imp_only),'pi_weighted_true':st(M.pi_true*M.sec_s_true_only),'full':st(M.tw_ss)}}
zg_edges=[0.0,0.075,0.15,0.25,0.392,2.0]; zb=pd.cut(M.z_g,zg_edges,labels=False)
out['by_zg']={}
for b in sorted(zb.dropna().unique()):
    S=M[(zb==b).to_numpy()]
    out['by_zg'][f'{zg_edges[int(b)]}-{zg_edges[int(b)+1]}']={'n':len(S),'pi_true_mean':float(S.pi_true.mean()),'n_cand_median':float(S.n_cand.median()),'sec_b_true_only':st(S.sec_b_true_only),'sec_b_imp_only':st(S.sec_b_imp_only),'full_b':st(S.tw_sb),'meas_b':st(S.sb_cat),'sec_s_true_only':st(S.sec_s_true_only),'sec_s_imp_only':st(S.sec_s_imp_only),'full_s':st(S.tw_ss),'meas_s':st(S.ss_cat)}
json.dump(out,open(SP/'f7_out.json','w'),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))
