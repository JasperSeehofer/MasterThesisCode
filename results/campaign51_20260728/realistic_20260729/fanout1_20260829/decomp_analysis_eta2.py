import pandas as pd, numpy as np

df = pd.read_csv('/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729/fanout1_20260829/b4_imp_decomposition.csv')

# outcome: log10(combined_no_bh) as the catalogue-leg's per-event posterior contribution
df['log_combined_no_bh'] = np.log10(df['combined_no_bh'].clip(lower=1e-300))
df['log_combined_with_bh'] = np.log10(df['combined_with_bh'].clip(lower=1e-300))
df['log_candcount_no_bh'] = np.log10(df['candidate_count_no_bh'].clip(lower=1))
df['log_candcount_with_bh'] = np.log10(df['candidate_count_with_bh'].clip(lower=1))

def eta_squared(sub, covariate, outcome, q=4):
    s = sub.dropna(subset=[covariate, outcome])
    if len(s) < q*3:
        return np.nan, np.nan
    try:
        s = s.copy()
        s['bin'] = pd.qcut(s[covariate], q=q, duplicates='drop', labels=False)
    except Exception:
        return np.nan, np.nan
    if s['bin'].nunique() < 2:
        return np.nan, np.nan
    grand_mean = s[outcome].mean()
    ss_between = s.groupby('bin')[outcome].apply(lambda x: len(x)*(x.mean()-grand_mean)**2).sum()
    ss_total = ((s[outcome]-grand_mean)**2).sum()
    eta2 = ss_between/ss_total if ss_total>0 else np.nan
    bin_means = s.groupby('bin')[outcome].mean()
    top_bin = bin_means.idxmax()
    return eta2, top_bin

covariates = {
    'z_true': 'log_combined_no_bh',
    'log_candcount_no_bh': 'log_combined_no_bh',
}
covariates_wbh = {
    'z_true': 'log_combined_with_bh',
    'log_candcount_with_bh': 'log_combined_with_bh',
}

rows = []
for arm in ['bc','bt']:
    for seed in sorted(df['seed'].unique()):
        sub = df[(df['arm']==arm)&(df['seed']==seed)]
        for cov, out in covariates.items():
            eta2, top_bin = eta_squared(sub, cov, out)
            rows.append({'arm':arm,'seed':seed,'channel':'no_bh','covariate':cov,'eta2':eta2,'top_quartile':top_bin,'n':len(sub)})
        for cov, out in covariates_wbh.items():
            eta2, top_bin = eta_squared(sub, cov, out)
            rows.append({'arm':arm,'seed':seed,'channel':'with_bh','covariate':cov,'eta2':eta2,'top_quartile':top_bin,'n':len(sub)})

res = pd.DataFrame(rows)
res.to_csv('/tmp/eta_by_seed.csv', index=False)

print("=== Mean eta^2 (variance explained) by covariate, channel, arm, across 12 seeds ===")
summary = res.groupby(['channel','covariate','arm'])['eta2'].agg(['mean','std','count'])
print(summary)

print()
print("=== Paired bc vs bt: per-seed eta2 side by side (no_bh channel) ===")
piv = res[res['channel']=='no_bh'].pivot_table(index=['seed','covariate'], columns='arm', values='eta2')
print(piv)

print()
print("=== Paired bc vs bt: per-seed eta2 side by side (with_bh channel) ===")
piv2 = res[res['channel']=='with_bh'].pivot_table(index=['seed','covariate'], columns='arm', values='eta2')
print(piv2)
