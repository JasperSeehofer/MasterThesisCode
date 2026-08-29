import numpy as np, pandas as pd, json, math, sys
from pathlib import Path
sys.path.insert(0,'/home/jasper/Repositories/darksiren-emri')
import darksiren_emri.validation.correspondence_1d as c1d
SP=Path('/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/70977a05-4e21-4daa-91f0-d0330808c1ea/scratchpad')
H=c1d.H_TRUE
handler=c1d._load_galaxy_catalog_handler(c1d.REDUCED_CATALOGUE_PATH); pool=c1d._host_pool_from_handler(handler)
completeness,phi_table=c1d.build_bsel_selection_objects(h_true=H)
NODES={'truth':(0.0,1.0),'b_plus':(0.02,1.0),'b_minus':(-0.02,1.0),'s_plus':(0.0,math.sqrt(2)),'s_minus':(0.0,1/math.sqrt(2))}
D=pd.read_csv(SP/'f6_alldraws.csv')
idx=D.host.to_numpy().astype(int); z=pool.z[idx]; ze=pool.z_error[idx]; ph=pool.phiS[idx]; q=pool.qS[idx]
lnS={}
for name,(b,s) in NODES.items():
    lnS[name]=np.log(c1d.kernel_smeared_survival(z+b*(1+z), s*ze, phi_table, completeness, ph, q, h=H))
Cb=(lnS['b_plus']-lnS['b_minus'])/0.04; Cs=(lnS['s_plus']-lnS['s_minus'])/(math.sqrt(2)-1/math.sqrt(2))
ev=D.evaluated.to_numpy()
st=lambda x:{'n':int(len(x)),'mean':float(x.mean()),'sem':float(x.std(ddof=1)/math.sqrt(len(x)))}
zg_edges=[0.0,0.075,0.15,0.25,0.392,2.0]; zb=pd.cut(D.z_g,zg_edges,labels=False).to_numpy()
res={'C_b_all800':st(Cb),'C_s_all800':st(Cs),'C_b_evaluated461':st(Cb[ev]),'C_s_evaluated461':st(Cs[ev]),
     'by_zg_evaluated':{f'{zg_edges[int(b)]}-{zg_edges[int(b)+1]}':{'C_b':st(Cb[ev&(zb==b)]),'C_s':st(Cs[ev&(zb==b)])} for b in sorted(set(zb[~np.isnan(zb)]))},
     'lnS_truth_stats':st(lnS['truth'])}
D['C_b']=Cb; D['C_s']=Cs; D.to_csv(SP/'f9_alldraws_C.csv',index=False)
json.dump(res,open(SP/'f9_out.json','w'),indent=1)
print(json.dumps(res,indent=1))
