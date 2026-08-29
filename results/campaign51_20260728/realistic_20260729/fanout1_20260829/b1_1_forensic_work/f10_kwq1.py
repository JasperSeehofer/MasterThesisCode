"""KW-Q1 contamination estimate: the h-dependence of the un-normalized no-BH divisor factor rho(s;h)=Sigma^phi(s;h)/Sigma^phi(1;h)
at the KW-Q1 h-nodes 0.725/0.735, on a 200k uniform pool subsample with explicit w_g weights (FT config: 'phi' numerator, 2.2/unsmeared)."""
import numpy as np, pandas as pd, json, math, sys
from pathlib import Path
sys.path.insert(0,'/home/jasper/Repositories/darksiren-emri')
import darksiren_emri.validation.correspondence_1d as c1d
from darksiren_emri.emri_rate import R_eff_per_mbh
SP=Path('/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/70977a05-4e21-4daa-91f0-d0330808c1ea/scratchpad')
handler=c1d._load_galaxy_catalog_handler(c1d.REDUCED_CATALOGUE_PATH); pool=c1d._host_pool_from_handler(handler)
rng=np.random.default_rng(7); sub=rng.choice(pool.n,size=200000,replace=False)
zs=pool.z[sub]; zes=pool.z_error[sub]; phs=pool.phiS[sub]; qs=pool.qS[sub]; ws=np.asarray(R_eff_per_mbh(pool.M[sub]),float)/(1+zs)
SN={'s_minus':1/math.sqrt(2),'truth':1.0,'s_plus':math.sqrt(2)}
out={}
lnSig={}
for h in (0.725,0.735,0.73):
    completeness,phi_table=c1d.build_bsel_selection_objects(h_true=h)
    for n,s in SN.items():
        St=c1d.kernel_smeared_survival(zs, s*zes, phi_table, completeness, phs, qs, h=h)
        lnSig[(h,n)]=math.log(float((ws*St).sum()))
for h in (0.725,0.735,0.73):
    out[f'lnrho_splus_h{h}']=lnSig[(h,'s_plus')]-lnSig[(h,'truth')]; out[f'lnrho_sminus_h{h}']=lnSig[(h,'s_minus')]-lnSig[(h,'truth')]
    out[f'C_s_h{h}']=(lnSig[(h,'s_plus')]-lnSig[(h,'s_minus')])/(math.sqrt(2)-1/math.sqrt(2))
# KW-Q1 contamination: s_imp,i(s) uses Delta_h ln[(beta L_cat(s)+B)/B]; the properly normalized catalogue leg divides L_cat(s) by rho(s;h),
# so the un-normalized form carries an extra Delta_h ln rho(s;h) (times the catalogue share c_i, first order) in s_imp(s) - s_imp(1).
dh=0.735-0.725
out['dh_lnrho_splus']=(out['lnrho_splus_h0.735']-out['lnrho_splus_h0.725'])/dh
out['dh_lnrho_sminus']=(out['lnrho_sminus_h0.735']-out['lnrho_sminus_h0.725'])/dh
out['R_numerator_contamination_per_unit_c']=out['dh_lnrho_splus']-out['dh_lnrho_sminus']
out['dh_lnSigma_truth']=(lnSig[(0.735,'truth')]-lnSig[(0.725,'truth')])/dh
json.dump(out,open(SP/'f10_out.json','w'),indent=1)
print(json.dumps(out,indent=1))
