"""Regional campaign diagnostics; local read-only inputs and aggregate JSON only.
No customer IDs are available: all counts are monthly observations, not clients.
"""
from pathlib import Path
import sys, argparse, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pipeline import prepare, CORE, HOME, _standardize
from run_analysis import load_inputs
REGIONS=['대경','부산광역시','경상남도','울산광역시']
METRICS=['총디지털거래액','총오프라인거래액','총카드소비','총외환실적','자동이체금액','총예금잔액','총대출잔액']
USE={'digital_use':'총디지털거래액','card_use':'총카드소비','fx_use':'총외환실적','auto_use':'자동이체금액'}

def matched_frame(frame):
    f=frame.replace({c:{np.inf:np.nan,-np.inf:np.nan} for c in CORE}).dropna(subset=CORE)
    f=f[f['예대마진']>=-2500]
    h=_standardize(f[f['사업장_시도'].isin(HOME)].copy())
    o=_standardize(f[~f['사업장_시도'].isin(HOME)].dropna(subset=['사업장_시도','사업장_시군구']).copy())
    z=['z_'+c for c in CORE]
    h['cluster']=KMeans(n_clusters=4,random_state=42,n_init='auto').fit_predict(h[z])
    selected=h[h['cluster']==2]
    both=pd.concat([selected,o],ignore_index=True)
    both['final']=KMeans(n_clusters=3,random_state=42,n_init='auto').fit_predict(both[z])
    m=both[both['final']==2].copy()
    m['region']=m['사업장_시도'].where(~m['사업장_시도'].isin(HOME),'대경')
    counts={'reference_rows':len(selected),'combined_rows':len(both),'matched_rows':len(m)}
    assert counts=={'reference_rows':70825,'combined_rows':131541,'matched_rows':88926},counts
    assert m.groupby('region').size().loc[REGIONS[1:]].tolist()==[5828,3301,1190]
    return m[m['region'].isin(REGIONS)].copy(),counts

def describe(g):
    industries=g['업종_대분류'].fillna('미상').value_counts()
    return {'rows':len(g),'medians':{c:float(g[c].median()) for c in METRICS},
            'use_rates':{k:float(g[k].mean()) for k in USE},
            'industries':[{'industry':str(k),'rows':int(v),'share':float(v/len(g))} for k,v in industries.items() if v>=30]}

def analyze(frame):
    m,counts=matched_frame(frame)
    for k,c in USE.items():m[k]=(m[c]>0).astype(float)
    m['year']=(m['기준년월']//100).astype(int)
    m['size']=m['총예금잔액']+m['총대출잔액']
    m['size_bin'],edges=pd.qcut(m['size'],4,labels=False,retbins=True,duplicates='drop')
    m['industry']=m['업종_대분류'].fillna('미상')
    strata=['year','industry','size_bin']
    sizes=m.groupby(strata+['region']).size().unstack('region',fill_value=0).reindex(columns=REGIONS,fill_value=0)
    common=sizes[(sizes>=30).all(axis=1)]
    assert len(common)>0
    idx=pd.MultiIndex.from_frame(m[strata])
    kept=m[idx.isin(common.index)].copy()
    weights=common.sum(axis=1)/common.to_numpy().sum()
    adjusted={}
    for r in REGIONS:
        g=kept[kept['region']==r]
        means=g.groupby(strata)[METRICS+list(USE)].mean().reindex(common.index)
        adjusted[r]={'rows':len(g),'means':{c:float((means[c]*weights).sum()) for c in means},'unadjusted_same_support':{c:float(g[c].mean()) for c in means}}
    annual={str(y):{r:describe(g) for r,g in a.groupby('region')} for y,a in m.groupby('year')}
    # Hypothetical equal-budget policy alternatives, NOT measured bank costs.
    scenarios=[]
    for channel,n,cost,fixed in [('remote',500,20000,2000000),('consultation',100,100000,2000000)]:
        total=n*cost+fixed
        scenarios.append({'policy':channel,'hypothetical_unique_firms':n,'contact_cost_won':cost,'fixed_cost_won':fixed,'total_cost_won':total,'contribution_per_incremental_conversion_won':1000000,'break_even_incremental_conversion_rate':total/n/1000000,'net_won_at_uplift':{str(u):n*u*1000000-total for u in [.01,.03,.05,.15]}})
    assert scenarios[0]['total_cost_won']==scenarios[1]['total_cost_won']
    return {'scope':'Exploratory regional monthly-observation comparison; no causal effects or independent-client inference','historical_reproduction':counts,'profiles':{r:describe(g) for r,g in m.groupby('region')},'adjustment':{'method':'Direct standardization, pooled weights, year x broad industry x pooled concurrent deposit+loan quartile, >=30 rows in every region per stratum','common_strata':len(common),'common_industries':sorted(set(common.index.get_level_values('industry'))),'included_rows':len(kept),'candidate_rows':len(m),'size_edges':edges.tolist(),'groups':adjusted},'annual_profiles':annual,'hypothetical_policy_scenarios':scenarios}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--rates',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.output.suffix!='.json' or a.output.resolve() in [a.data.resolve(),a.rates.resolve()]:raise ValueError('Choose new aggregate JSON output')
    df,rates=load_inputs(a.data,a.rates)
    result=analyze(prepare(df,rates))
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('Regional aggregate analysis complete',flush=True)
if __name__=='__main__':main()
