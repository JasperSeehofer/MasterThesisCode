"""Exact (not first-order) registered combined-channel scores after restoring the theta-dependence of the no-BH catalogue divisor:
combined_corr(theta) = (beta L_cat(theta)/rho(theta) + B_num)/D~, rho(theta) = Sigma^phi(theta)/Sigma^phi(0,1) estimated as the
draw-weighted mean over the 800 drawn hosts of S~_g(theta)/S~_g(0,1) (the draw weight is w_g S~_g(0,1))."""
import numpy as np, pandas as pd, json, math, sys
from pathlib import Path
sys.path.insert(0,'/home/jasper/Repositories/darksiren-emri')
import darksiren_emri.validation.correspondence_1d as c1d
SP=Path('/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/70977a05-4e21-4daa-91f0-d0330808c1ea/scratchpad')
H=c1d.H_TRUE
handler=c1d._load_galaxy_catalog_handler(c1d.REDUCED_CATALOGUE_PATH); pool=c1d._host_pool_from_handler(handler)
completeness,phi_table=c1d.build_bsel_selection_objects(h_true=H)
NODES={'truth':(0.0,1.0),'b_plus':(0.02,1.0),'b_minus':(-0.02,1.0),'s_plus':(0.0,math.sqrt(2)),'s_minus':(0.0,1/math.sqrt(2))}
D=pd.read_csv(SP/'f6_alldraws.csv'); idx=D.host.to_numpy().astype(int); z=pool.z[idx]; ze=pool.z_error[idx]; ph=pool.phiS[idx]; q=pool.qS[idx]
S={n:c1d.kernel_smeared_survival(z+b*(1+z), s*ze, phi_table, completeness, ph, q, h=H) for n,(b,s) in NODES.items()}
rho={n:float(np.mean(S[n]/S['truth'])) for n in NODES}
# cross-check rho by a uniform random pool subsample with explicit w_g S~ weights
from darksiren_emri.emri_rate import R_eff_per_mbh
rng=np.random.default_rng(7); sub=rng.choice(pool.n,size=200000,replace=False)
zs=pool.z[sub]; zes=pool.z_error[sub]; phs=pool.phiS[sub]; qs=pool.qS[sub]; ws=np.asarray(R_eff_per_mbh(pool.M[sub]),float)/(1+zs)
Ss={n:c1d.kernel_smeared_survival(zs+b*(1+zs), s*zes, phi_table, completeness, phs, qs, h=H) for n,(b,s) in NODES.items()}
rho_pool={n:float((ws*Ss[n]).sum()/(ws*Ss['truth']).sum()) for n in NODES}
A=pd.read_csv(SP/'f7_events.csv',index_col=0)
M=A[~A.dark].copy()
def corr_ln(node,rh):
    comb=np.exp(M[f'ln_nb_{node}']); betaL=comb*M.D_tilde_phi-M.B_num
    return np.log((betaL/rh+M.B_num)/M.D_tilde_phi)
out={'rho_drawn':rho,'rho_pool200k':rho_pool}
for lab,rh in (('rho_drawn',rho),('rho_pool200k',rho_pool)):
    sb=(corr_ln('b_plus',rh['b_plus'])-corr_ln('b_minus',rh['b_minus']))/0.04
    ss=(corr_ln('s_plus',rh['s_plus'])-corr_ln('s_minus',rh['s_minus']))/(math.sqrt(2)-1/math.sqrt(2))
    st=lambda x:dict(n=int(len(x)),mean=float(x.mean()),sem=float(x.std(ddof=1)/math.sqrt(len(x))),Z=float(x.mean()/(x.std(ddof=1)/math.sqrt(len(x)))))
    out[lab+'_corrected_combined']={'score_b':st(sb),'score_s':st(ss)}
    out[lab+'_per_seed']={str(s):{'score_b':st(sb[M.seed==s]),'score_s':st(ss[M.seed==s])} for s in [900101,900102,900103,900104]}
# uncorrected for reference
sb0=(M.ln_nb_b_plus-M.ln_nb_b_minus)/0.04; ss0=(M.ln_nb_s_plus-M.ln_nb_s_minus)/(math.sqrt(2)-1/math.sqrt(2))
out['uncorrected']={'score_b':{'mean':float(sb0.mean()),'sem':float(sb0.std(ddof=1)/math.sqrt(len(sb0)))},'score_s':{'mean':float(ss0.mean()),'sem':float(ss0.std(ddof=1)/math.sqrt(len(ss0)))}}
json.dump(out,open(SP/'f11_out.json','w'),indent=1)
print(json.dumps(out,indent=1))
