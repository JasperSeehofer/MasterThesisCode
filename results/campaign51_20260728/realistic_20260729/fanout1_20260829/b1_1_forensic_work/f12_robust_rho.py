"""Robust pool-normalizer ratio rho(theta) (excluding hosts whose b_minus-shifted kernel window inverts / centre goes negative) and the
exact divisor-corrected registered scores; plus a census of the negative-centre edge case of the b-hook at b=-0.02."""
import numpy as np, pandas as pd, json, math, sys
from pathlib import Path
sys.path.insert(0,'/home/jasper/Repositories/darksiren-emri')
import darksiren_emri.validation.correspondence_1d as c1d
from darksiren_emri.emri_rate import R_eff_per_mbh
SP=Path('/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/70977a05-4e21-4daa-91f0-d0330808c1ea/scratchpad')
H=c1d.H_TRUE
handler=c1d._load_galaxy_catalog_handler(c1d.REDUCED_CATALOGUE_PATH); pool=c1d._host_pool_from_handler(handler)
completeness,phi_table=c1d.build_bsel_selection_objects(h_true=H)
NODES={'truth':(0.0,1.0),'b_plus':(0.02,1.0),'b_minus':(-0.02,1.0),'s_plus':(0.0,math.sqrt(2)),'s_minus':(0.0,1/math.sqrt(2))}
# edge-case census over the whole pool at b=-0.02, s=1: negative centre / inverted window
zc=pool.z-0.02*(1+pool.z); neg_centre=zc<0; inverted=(zc+4*pool.z_error)<=1e-6
out={'pool_n':int(pool.n),'pool_neg_centre_at_bminus':int(neg_centre.sum()),'pool_inverted_window_at_bminus':int(inverted.sum()),'pool_z_min':float(pool.z.min()),'pool_frac_z_lt_0.03':float((pool.z<0.03).mean())}
D=pd.read_csv(SP/'f6_alldraws.csv'); idx=D.host.to_numpy().astype(int); z=pool.z[idx]; ze=pool.z_error[idx]; ph=pool.phiS[idx]; q=pool.qS[idx]
S={n:c1d.kernel_smeared_survival(z+b*(1+z), s*ze, phi_table, completeness, ph, q, h=H) for n,(b,s) in NODES.items()}
bad=(z-0.02*(1+z)<0)|~np.isfinite(S['b_minus'])|(S['b_minus']<=0)|(S['truth']<=0)
out['drawn_n']=int(len(z)); out['drawn_bad_hosts']=int(bad.sum()); out['drawn_bad_hosts_detail']=[{'z_g':float(z[i]),'sigma_g':float(ze[i]),'S_truth':float(S['truth'][i]),'S_bminus':float(S['b_minus'][i]),'S_bplus':float(S['b_plus'][i])} for i in np.where(bad)[0][:10]]
good=~bad
rho_drawn={n:float(np.mean(S[n][good]/S['truth'][good])) for n in NODES}
rng=np.random.default_rng(7); sub=rng.choice(pool.n,size=200000,replace=False)
zs=pool.z[sub]; zes=pool.z_error[sub]; phs=pool.phiS[sub]; qs=pool.qS[sub]; ws=np.asarray(R_eff_per_mbh(pool.M[sub]),float)/(1+zs)
Ss={n:c1d.kernel_smeared_survival(zs+b*(1+zs), s*zes, phi_table, completeness, phs, qs, h=H) for n,(b,s) in NODES.items()}
bads=(zs-0.02*(1+zs)<0)|~np.isfinite(Ss['b_minus'])|(Ss['b_minus']<=0)
out['pool200k_bad']=int(bads.sum()); out['pool200k_bad_weight_share']=float((ws*Ss['truth'])[bads].sum()/(ws*Ss['truth']).sum())
g=~bads
rho_pool={n:float((ws[g]*Ss[n][g]).sum()/(ws[g]*Ss['truth'][g]).sum()) for n in NODES}
out['rho_drawn_robust']=rho_drawn; out['rho_pool200k_robust']=rho_pool
out['C_from_rho']={lab:{'C_b':(math.log(r['b_plus'])-math.log(r['b_minus']))/0.04,'C_s':(math.log(r['s_plus'])-math.log(r['s_minus']))/(math.sqrt(2)-1/math.sqrt(2))} for lab,r in (('drawn',rho_drawn),('pool',rho_pool))}
Cb_h=(np.log(S['b_plus'][good])-np.log(S['b_minus'][good]))/0.04; Cs_h=(np.log(S['s_plus'][good])-np.log(S['s_minus'][good]))/(math.sqrt(2)-1/math.sqrt(2))
out['C_perhost_mean_drawn_good']={'C_b':{'mean':float(Cb_h.mean()),'sem':float(Cb_h.std(ddof=1)/math.sqrt(len(Cb_h)))},'C_s':{'mean':float(Cs_h.mean()),'sem':float(Cs_h.std(ddof=1)/math.sqrt(len(Cs_h)))},'n':int(good.sum())}
A=pd.read_csv(SP/'f7_events.csv',index_col=0); M=A[~A.dark].copy()
def corr_ln(node,rh):
    comb=np.exp(M[f'ln_nb_{node}']); betaL=np.clip(comb*M.D_tilde_phi-M.B_num,0,None)
    return np.log((betaL/rh+M.B_num)/M.D_tilde_phi)
st=lambda x:dict(n=int(len(x)),mean=float(x.mean()),sem=float(x.std(ddof=1)/math.sqrt(len(x))),Z=float(x.mean()/(x.std(ddof=1)/math.sqrt(len(x)))))
for lab,rh in (('drawn',rho_drawn),('pool',rho_pool)):
    sb=(corr_ln('b_plus',rh['b_plus'])-corr_ln('b_minus',rh['b_minus']))/0.04
    ss=(corr_ln('s_plus',rh['s_plus'])-corr_ln('s_minus',rh['s_minus']))/(math.sqrt(2)-1/math.sqrt(2))
    out[f'corrected_combined_{lab}']={'score_b':st(sb),'score_s':st(ss),'per_seed':{str(s):{'score_b':st(sb[M.seed==s]),'score_s':st(ss[M.seed==s])} for s in [900101,900102,900103,900104]}}
    zg_edges=[0.0,0.075,0.15,0.25,0.392,2.0]; zb=pd.cut(M.z_g,zg_edges,labels=False)
    out[f'corrected_combined_{lab}_by_zg']={f'{zg_edges[int(b)]}-{zg_edges[int(b)+1]}':{'score_b':st(sb[(zb==b).to_numpy()]),'score_s':st(ss[(zb==b).to_numpy()])} for b in sorted(zb.dropna().unique())}
    out[f'corrected_combined_{lab}_wb']=None
json.dump(out,open(SP/'f12_out.json','w'),indent=1,default=float)
print(json.dumps({k:v for k,v in out.items() if 'by_zg' not in k and 'detail' not in k},indent=1,default=float))
print('detail',out['drawn_bad_hosts_detail'][:5])
print({k:{kk:(round(vv['score_b']['mean'],3),round(vv['score_b']['Z'],2),round(vv['score_s']['mean'],4),round(vv['score_s']['Z'],2)) for kk,vv in v.items()} for k,v in out.items() if 'by_zg' in k})
