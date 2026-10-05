"""Supplementary robustness checks for the ML portfolio.

Reads private local inputs, writes one aggregate JSON, and never writes rows,
models, identifiers, or plots containing observations.
"""
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import prepare, CORE, PROFILE, REGRESSION, HOME


def _scaled(frame, cols, fit=None):
    vals = np.log1p(frame[cols].clip(lower=0))
    scaler = StandardScaler() if fit is None else fit
    return scaler.fit_transform(vals) if fit is None else scaler.transform(vals)


def time_split_regression(frame):
    """Train through 2023-12 and test 2024 observations; aggregate only."""
    home = frame[frame['사업장_시도'].isin(HOME)].copy()
    home = home[home['예대마진'].notna() & (home['예대마진'] >= -2500)]
    train = home[home['기준년월'] <= 202312]
    test = home[home['기준년월'] > 202312]
    if len(train) > 30000:
        train = train.sample(30000, random_state=42)
    xtrain = np.log1p(train[REGRESSION].clip(lower=0))
    xtest = np.log1p(test[REGRESSION].clip(lower=0))
    # A lightweight time-split diagnostic is used here. The stored RF result
    # remains the main model; this check asks whether a simple relationship
    # transfers to the later period without a long retraining run.
    model = Ridge(alpha=1.0)
    model.fit(xtrain, train['예대마진'])
    pred = model.predict(xtest)
    return {'model': 'Ridge time-split diagnostic on 30k train sample', 'train_rows': len(train), 'available_train_rows': int((home['기준년월'] <= 202312).sum()), 'test_rows': len(test), 'train_period': 'through 202312', 'test_period': '202401-202412', 'r2': float(r2_score(test['예대마진'], pred)), 'rmse': float(np.sqrt(mean_squared_error(test['예대마진'], pred)))}


def cluster_stability(frame):
    home = frame[frame['사업장_시도'].isin(HOME)].copy()
    home = home[home[CORE].notna().all(axis=1) & (home['예대마진'] >= -2500)]
    x = _scaled(home, CORE)
    sample = np.random.default_rng(42).choice(len(x), size=min(50000, len(x)), replace=False)
    x = x[sample]
    result = {'sample_rows': len(x), 'silhouette_by_k': {}, 'seed_ari_proxy': {}}
    labels_by_k = {}
    for k in [2,3,4,5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(x)
        labels_by_k[k] = km.labels_
        result['silhouette_by_k'][str(k)] = float(silhouette_score(x, km.labels_))
    # Same-k agreement across seeds, measured as label-invariant adjusted Rand.
    from sklearn.metrics import adjusted_rand_score
    for k in [2,3,4,5]:
        a = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(x)
        b = KMeans(n_clusters=k, random_state=7, n_init=10).fit_predict(x)
        result['seed_ari_proxy'][str(k)] = float(adjusted_rand_score(a, b))
    return result


def similarity_sensitivity(frame):
    core = frame[frame['사업장_시도'].isin(HOME)].copy().reset_index(drop=True)
    core = core[core[CORE].notna().all(axis=1) & (core['예대마진'] >= -2500)]
    other = frame.dropna(subset=['사업장_시도','사업장_시군구']).copy().reset_index(drop=True)
    other = other[~other['사업장_시도'].isin(HOME)]
    other = other[other[CORE].notna().all(axis=1) & (other['예대마진'] >= -2500)]
    other = other.reset_index(drop=True)
    # Use the saved 1st-stage criterion: explicit median profile, no refit.
    # This sensitivity compares scaling choices for the same rows.
    combined = pd.concat([core, other], ignore_index=True)
    cols = PROFILE
    outputs = {}
    for mode in ['global', 'home_reference', 'regional']:
        if mode == 'global':
            vals = _scaled(combined, cols)
            allv = pd.DataFrame(vals, columns=cols)
            homev, otherv = allv.iloc[:len(core)].reset_index(drop=True), allv.iloc[len(core):].reset_index(drop=True)
        elif mode == 'home_reference':
            scaler = StandardScaler().fit(np.log1p(core[cols].clip(lower=0)))
            homev = pd.DataFrame(scaler.transform(np.log1p(core[cols].clip(lower=0))), columns=cols)
            otherv = pd.DataFrame(scaler.transform(np.log1p(other[cols].clip(lower=0))), columns=cols).reset_index(drop=True)
        else:
            homev = pd.DataFrame(_scaled(core, cols), columns=cols)
            otherv = pd.DataFrame(_scaled(other, cols), columns=cols).reset_index(drop=True)
        ref = homev.median().to_numpy().reshape(1,-1)
        rows=[]
        for region, group in other.groupby('사업장_시도', sort=False):
            if len(group) < 30: continue
            idx=group.index
            vec=otherv.loc[idx].median().to_numpy().reshape(1,-1)
            rows.append((region, float(cosine_similarity(ref,vec)[0,0])))
        outputs[mode] = [{'region': r, 'cosine': v} for r,v in sorted(rows,key=lambda z:z[1],reverse=True)[:5]]
    ranks = {m:[r['region'] for r in rows] for m,rows in outputs.items()}
    return {'top5_by_scaling': outputs, 'top3_overlap_global_home_reference': len(set(ranks['global'][:3]) & set(ranks['home_reference'][:3])), 'top3_overlap_global_regional': len(set(ranks['global'][:3]) & set(ranks['regional'][:3]))}


def business_priority(sensitivity, frame):
    """Illustrative screening score; not an observed profit model."""
    home=frame[frame['사업장_시도'].isin(HOME)]
    other=frame[~frame['사업장_시도'].isin(HOME)].dropna(subset=['사업장_시도'])
    by=other.groupby('사업장_시도').size().to_dict()
    out=[]
    for row in sensitivity['top5_by_scaling']['global'][:5]:
        region=row['region']; count=by.get(region,0)
        # volume is log-scaled only for a screening score; weights are explicit assumptions
        score=.7*row['cosine']+.3*(np.log1p(count)/np.log1p(max(by.values())))
        out.append({'region':region,'cosine':row['cosine'],'observation_rows':int(count),'screening_score':float(score)})
    return sorted(out,key=lambda r:r['screening_score'],reverse=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--rates',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    raw,rates=pd.read_csv(a.data,encoding='cp949'),pd.read_csv(a.rates,encoding='utf-8-sig')
    # Robustness checks use a fixed, reproducible 100k-row audit sample to keep
    # this optional script practical. The main stored results use all rows.
    available_rows = len(raw)
    if len(raw) > 100000:
        raw = raw.sample(100000, random_state=42).reset_index(drop=True)
    frame=prepare(raw,rates)
    sensitivity=similarity_sensitivity(frame)
    result={'data':{'audit_sample_rows':len(raw),'available_rows':available_rows,'periods':int(raw['기준년월'].nunique())},'time_split_regression':time_split_regression(frame),'cluster_stability':cluster_stability(frame),'similarity_sensitivity':sensitivity,'illustrative_priority_screen':business_priority(sensitivity,frame),'interpretation':'The priority screen is a sensitivity illustration with assumed weights, not observed profit or ROI.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('PASS supplementary aggregate analysis')

if __name__=='__main__': main()
