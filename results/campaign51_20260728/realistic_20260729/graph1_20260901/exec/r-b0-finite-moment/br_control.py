"""B-R design-gate control for r-b0-finite-moment (zero compute; read-only on banked data)."""
import json, glob, os, sys
import numpy as np, pandas as pd
ROOT='/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729'
H=0.73; R_H=1.515548762178686; NDRAWN=200
seeds=list(range(900101,900113))
out={}
def rows_at_h(csv):
    df=pd.read_csv(csv)
    return df[np.isclose(df['h'],H,atol=1e-9)].copy()
def beta(sel_json):
    return json.load(open(sel_json))['beta_G_phi'], json.load(open(sel_json))['beta_Gbar_phi']
# --- mass companion C* ---
mc=json.load(open(f'{ROOT}/p3_b0_identity_test_output.json'))['mass_companion_at_h_gen']
rho=mc['rho']
bG,bGb=beta(f'{ROOT}/p3_b0_work/bc_900101_work/seed900101/selection_tables_h_0_73.json')
Cstar=bG*rho/bGb
out['C_star']=Cstar; out['rho']=rho; out['beta_G_phi']=bG; out['beta_Gbar_phi']=bGb
# --- LHS per seed ---
lhs={'B-T':[], 'B-C':[], 'B-R':[]}; lam=[]; nrows={}
betas=set()
for s in seeds:
    d={}
    for arm,tag in (('B-T','bt'),('B-C','bc')):
        w=f'{ROOT}/p3_b0_work/{tag}_{s}_work/seed{s}'
        bg,_=beta(f'{w}/selection_tables_h_0_73.json'); betas.add(round(bg,3))
        df=rows_at_h(f'{w}/simulations/diagnostics/event_likelihoods.csv')
        A=bg*df['L_cat_no_bh'].to_numpy(); B=df['B_num'].to_numpy()
        with np.errstate(invalid='ignore',divide='ignore'):
            wv=np.where(A+B>0, A/(A+B), 0.0)
        d[arm]=(df,wv)
        lhs[arm].append(Cstar/NDRAWN*np.sum(1-wv))
        nrows[(s,arm)]=len(df)
    wt=d['B-T'][1]
    lhs['B-R'].append(Cstar/NDRAWN*np.sum((1-wt)/(1+(R_H-1)*wt)))
    # C-B Lambda on paired live rows
    t=d['B-T'][0].set_index('event_idx'); c=d['B-C'][0].set_index('event_idx')
    j=t.join(c,lsuffix='_t',rsuffix='_c',how='inner')
    live=(j['L_cat_no_bh_t']>0)&(j['L_cat_no_bh_c']>0)
    lam.append(np.mean(np.log(j.loc[live,'L_cat_no_bh_t']/j.loc[live,'L_cat_no_bh_c'])))
out['beta_G_phi_distinct_across_seeds']=sorted(betas)
out['n_rows_at_h073']={f'{s}_{a}':n for (s,a),n in nrows.items()}
def fleet(v): v=np.asarray(v); return float(v.mean()), float(v.std(ddof=1)/np.sqrt(len(v)))
for k,v in lhs.items(): out[f'LHS_{k}']=fleet(v); out[f'LHS_{k}_per_seed']=[float(x) for x in v]
out['paired_delta_BT_minus_BC']=fleet(np.array(lhs['B-T'])-np.array(lhs['B-C']))
lam=np.asarray(lam); out['Lambda_raw_mean_paired_live']=fleet(lam)
out['Lambda_center_ln_Sigma_w_over_Sigma_phi_tilde']=float(np.log(mc['Sigma_w']/mc['Sigma_phi_tilde']))
out['Lambda_bar_BT']=[out['Lambda_raw_mean_paired_live'][0]+out['Lambda_center_ln_Sigma_w_over_Sigma_phi_tilde'], out['Lambda_raw_mean_paired_live'][1]]
out['Lambda_bar_BR_shift']=float(np.log(R_H))
# --- RHS from the clean C-A chunks 5..224 ---
taus=[30,100,300,1000]
acc={'twin':[], 'coded':[], 'br':[], 'dc':[], 'n_acc_twin':[], 'n_acc_coded':[],
     'tci_rhs_twinR':{t:[] for t in taus}, 'tci_rhs_brR':{t:[] for t in taus}}
bad=[]
for k in range(5,225):
    wt_dir=f'{ROOT}/ca_rhs_work/score_chunk{k}_twin_work'; wc_dir=f'{ROOT}/ca_rhs_work/score_chunk{k}_coded_work'
    try:
        bt_,_=beta(f'{wt_dir}/selection_tables_h_0_73.json'); bc_,_=beta(f'{wc_dir}/selection_tables_h_0_73.json')
        dt=rows_at_h(f'{wt_dir}/simulations/diagnostics/event_likelihoods.csv').set_index('event_idx')
        dc=rows_at_h(f'{wc_dir}/simulations/diagnostics/event_likelihoods.csv').set_index('event_idx')
    except Exception as e:
        bad.append((k,str(e)[:80])); continue
    if len(dt)>NDRAWN or len(dc)>NDRAWN or dt.index.duplicated().any(): bad.append((k,'contaminated')); continue
    At=bt_*dt['L_cat_no_bh'].to_numpy(); Bt=dt['B_num'].to_numpy()
    with np.errstate(invalid='ignore',divide='ignore'):
        wt=np.where(At+Bt>0,At/(At+Bt),0.0)
    Ac=bc_*dc['L_cat_no_bh'].to_numpy(); Bc=dc['B_num'].to_numpy()
    with np.errstate(invalid='ignore',divide='ignore'):
        wc=np.where(Ac+Bc>0,Ac/(Ac+Bc),0.0)
    acc['twin'].append(wt.sum()/NDRAWN); acc['coded'].append(wc.sum()/NDRAWN)
    acc['br'].append(np.sum(wt/(1+(R_H-1)*wt))/NDRAWN)
    j=dt.join(dc,lsuffix='_t',rsuffix='_c',how='inner')
    Wt=np.where(j['L_cat_no_bh_c']>0, j['L_cat_no_bh_t']/j['L_cat_no_bh_c'], 1.0)
    Acj=bc_*j['L_cat_no_bh_c'].to_numpy(); Bcj=j['B_num_c'].to_numpy()
    with np.errstate(invalid='ignore',divide='ignore'):
        wcj=np.where(Acj+Bcj>0,Acj/(Acj+Bcj),0.0)
    acc['dc'].append(np.sum(Wt*wcj)/NDRAWN)
    acc['n_acc_twin'].append(len(dt)); acc['n_acc_coded'].append(len(dc))
    with np.errstate(invalid='ignore',divide='ignore'):
        Rt=np.where(wt>0,(1-wt)/wt,np.inf); Rbr=Rt/R_H
    for t in taus:
        acc['tci_rhs_twinR'][t].append(np.sum(Rt<=t)/NDRAWN); acc['tci_rhs_brR'][t].append(np.sum(Rbr<=t)/NDRAWN)
out['rhs_chunks_used']=len(acc['twin']); out['rhs_chunks_skipped']=bad
for k in ('twin','coded','br','dc'): out[f'RHS_{k}']=fleet(acc[k])
out['RHS_accepted_total']=int(np.sum(acc['n_acc_twin']))
out['RHS_tci_twin']={t:fleet(acc['tci_rhs_twinR'][t]) for t in taus}
out['RHS_tci_brR']={t:fleet(acc['tci_rhs_brR'][t]) for t in taus}
# --- design-gate statistics ---
def comb(a,b): return float(np.sqrt(a**2+b**2))
LBR,sLBR=out['LHS_B-R']; RBR,sRBR=out['RHS_br']; LBT,sLBT=out['LHS_B-T']; RT,sRT=out['RHS_twin']; LBC,sLBC=out['LHS_B-C']; RC,sRC=out['RHS_coded']; DC,sDC=out['RHS_dc']
g={}
g['G_BR_exact']={'value':LBR-RBR,'sigma':comb(sLBR,sRBR)}
g['G_BR_power_naive']={'value':LBR-R_H*RBR,'sigma':comb(sLBR,R_H*sRBR),'predicted_under_identity':RBR*(1-R_H)}
g['T_w_BT']={'value':LBT-RT,'sigma':comb(sLBT,sRT)}
g['BC_vs_DC']={'value':LBC-DC,'sigma':comb(sLBC,sDC)}
g['BC_naive']={'value':LBC-RC,'sigma':comb(sLBC,sRC)}
for kk,v in g.items(): v['Z']=v['value']/v['sigma']; v['abs_gt_band_0.005']=abs(v['value'])>0.005
out['design_gate']=g
# C-TCI LHS for B-T and B-R (indicator member) per tau
tci={'BT':{}, 'BR_naive':{}}
for t in taus:
    lt=[];lb=[]
    for s in seeds:
        w=f'{ROOT}/p3_b0_work/bt_{s}_work/seed{s}'
        bg,_=beta(f'{w}/selection_tables_h_0_73.json'); df=rows_at_h(f'{w}/simulations/diagnostics/event_likelihoods.csv')
        A=bg*df['L_cat_no_bh'].to_numpy(); B=df['B_num'].to_numpy()
        with np.errstate(invalid='ignore',divide='ignore'):
            Rr=np.where(A>0,B/A,np.inf); Rb=Rr/R_H
        lt.append(Cstar/NDRAWN*np.sum(np.where(Rr<=t,Rr,0.0))); lb.append(Cstar/NDRAWN*np.sum(np.where(Rb<=t,Rb,0.0)))
    tci['BT'][t]={'LHS':fleet(lt),'RHS':out['RHS_tci_twin'][t]}
    tci['BR_naive'][t]={'LHS':fleet(lb),'RHS':out['RHS_tci_brR'][t]}
    for key in ('BT','BR_naive'):
        L,sL=tci[key][t]['LHS']; Rv,sR=tci[key][t]['RHS']; tci[key][t]['T']=L-Rv; tci[key][t]['sigma']=comb(sL,sR); tci[key][t]['Z']=(L-Rv)/comb(sL,sR)
out['C_TCI']=tci
json.dump(out,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'br_control_output.json'),'w'),indent=1,default=float)
print(json.dumps({k:v for k,v in out.items() if 'per_seed' not in k},indent=1,default=float))
