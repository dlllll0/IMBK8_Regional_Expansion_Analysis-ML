"""Notebook-derived preprocessing and the historical clustering specification.

Historical mode intentionally keeps separate regional standardization. It is a
reproduction aid, not a validated deployment model. No row-level files are saved.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

CORE = ['예대마진', '총예금좌수', '여신한도금액', '총디지털거래액']
PROFILE = CORE + ['총오프라인거래액', '자동이체금액', '자동이체거래건수', '총외환실적', '총카드소비']
REGRESSION = ['여신한도금액', '총디지털거래액', '총오프라인거래액', '총카드소비', '총외환실적', '총예금좌수', '자동이체거래건수']
HOME = ['대구광역시', '경상북도']

def prepare(df, df_b):
    """Mappings and feature sums from main notebook cell 0 (zero-based)."""
    df = df.copy()
    count_mapping = {'0건': 0, '1건': 1, '2건': 2, '2건초과 5건이하': 3.5, '5건초과 10건이하': 7.5, '10건초과 20건이하': 15, '20건초과 30건이하': 25, '30건초과 40건이하': 35, '40건초과 50건이하': 45, '50건 초과': 55}
    channel_count_cols = ['인터넷뱅킹거래건수', '스마트뱅킹거래건수', '폰뱅킹거래건수', '자동이체거래건수', '창구거래건수', 'ATM거래건수', '외환_수출실적거래건수', '외환_수입실적거래건수']
    for col in channel_count_cols:
        if col in df.columns:
            df[col] = df[col].map(count_mapping)
    account_mapping = {'0개': 0, '1개': 1, '2개': 2, '2개초과 5개이하': 4, '5개초과 10개이하': 8, '10개초과 20개이하': 15.5, '20개초과 30개이하': 25.5, '30개초과 40개이하': 35.5, '40개초과 50개이하': 45.5, '50개 초과': 55}
    account_count_cols = ['요구불예금좌수', '거치식예금좌수', '적립식예금좌수', '수익증권좌수', '신탁좌수', '퇴직연금좌수', '여신_운전자금대출좌수', '여신_시설자금대출좌수', '신용카드개수']
    for col in account_count_cols:
        if col in df.columns:
            df[col] = df[col].map(account_mapping)
    deposit_bal_cols = ['요구불예금잔액', '거치식예금잔액', '적립식예금잔액']
    product_bal_cols = ['수익증권잔액', '신탁잔액']
    deposit_cnt_cols = ['요구불예금좌수', '거치식예금좌수', '적립식예금좌수']
    product_cnt_cols = ['수익증권좌수', '신탁좌수', '퇴직연금좌수']
    flow_cols = ['요구불입금금액', '요구불출금금액']
    loan_bal_cols = ['여신_운전자금대출잔액', '여신_시설자금대출잔액']
    loan_cnt_cols = ['여신_운전자금대출좌수', '여신_시설자금대출좌수']
    rjfo_total = ['인터넷뱅킹거래금액', '폰뱅킹거래금액', '스마트뱅킹거래금액', '창구거래금액', 'ATM거래금액']
    rjfo_digital = ['인터넷뱅킹거래금액', '폰뱅킹거래금액', '스마트뱅킹거래금액']
    rjfo_offline = ['창구거래금액', 'ATM거래금액']
    rjfo_card_total = ['신용카드사용금액', '체크카드사용금액']
    channel_cnt_cols = ['창구거래건수', '인터넷뱅킹거래건수', '스마트뱅킹거래건수', '폰뱅킹거래건수', 'ATM거래건수']
    dhlghks = ['외환_수출실적금액', '외환_수입실적금액']
    df['총예금잔액'] = df[deposit_bal_cols].sum(axis=1)
    df['자산관리잔액'] = df[product_bal_cols].sum(axis=1)
    df['총대출잔액'] = df[loan_bal_cols].sum(axis=1)
    df['총예금좌수'] = df[deposit_cnt_cols].sum(axis=1)
    df['총대출좌수'] = df[loan_cnt_cols].sum(axis=1)
    df['자산관리좌수'] = df[product_cnt_cols].sum(axis=1)
    df['총요구불입출금'] = df[flow_cols].sum(axis=1)
    df['전체거래액'] = df[rjfo_total].sum(axis=1)
    df['총디지털거래액'] = df[rjfo_digital].sum(axis=1)
    df['총오프라인거래액'] = df[rjfo_offline].sum(axis=1)
    df['총카드소비'] = df[rjfo_card_total].sum(axis=1)
    df['총외환실적'] = df[dhlghks].sum(axis=1)
    cols_to_use = ['기준년월', '평균대출금리(%)', '평균예금금리(%)', '예대마진(%)']
    df_b_subset = df_b[cols_to_use]
    df2 = pd.merge(df, df_b_subset, on='기준년월', how='left', validate='many_to_one')
    df2['총자산'] = df2['총예금잔액'] + df2['총대출잔액'] + df2['자산관리잔액']
    df2['예금이자'] = df2['총예금잔액'] * df2['평균예금금리(%)'] * 0.01
    df2['대출이자'] = df2['총대출잔액'] * df2['평균대출금리(%)'] * 0.01
    df2['예대마진'] = df2['대출이자'] - df2['예금이자']
    if df2[['평균대출금리(%)', '평균예금금리(%)']].isna().any().any():
        raise ValueError('Missing monthly interest rates')
    return df2

def _standardize(frame):
    frame = frame.copy()
    minimum = frame['예대마진'].min()
    for col in PROFILE:
        value = frame[col]
        log_value = np.log1p(value - minimum + 1) if col == '예대마진' else np.log1p(value)
        if not np.isfinite(log_value).all():
            raise ValueError('Nonfinite transformed feature: ' + col)
        frame['z_' + col] = StandardScaler().fit_transform(log_value.to_numpy().reshape(-1, 1)).ravel()
    return frame

def historical_clusters(frame):
    """Main cells 2,4,6,7,23,26,31,32,35,39,43,60, without row export.

    Cluster labels 2 and 2 follow the saved notebook. Inspect profiles before
    interpreting labels after changes to data, libraries or initialization.
    """
    home = frame[frame['사업장_시도'].isin(HOME)].copy()
    home[CORE] = home[CORE].replace([np.inf, -np.inf], np.nan)
    home = home.dropna(subset=CORE)
    home = home[home['예대마진'] >= -2500].copy()
    other = frame.dropna(subset=['사업장_시도', '사업장_시군구']).copy()
    other = other[~other['사업장_시도'].isin(HOME)].copy()
    other[CORE] = other[CORE].replace([np.inf, -np.inf], np.nan)
    other = other.dropna(subset=CORE)
    other = other[other['예대마진'] >= -2500].copy()
    home, other = _standardize(home), _standardize(other)
    zcore = ['z_' + col for col in CORE]
    home['cluster'] = KMeans(n_clusters=4, random_state=42, n_init='auto').fit_predict(home[zcore])
    selected = home[home['cluster'] == 2].copy()
    combined = pd.concat([selected, other], axis=0).reset_index(drop=True)
    combined['Final_Cluster'] = KMeans(n_clusters=3, random_state=42, n_init='auto').fit_predict(combined[zcore])
    matched = combined[combined['Final_Cluster'] == 2].copy()
    matched['지역'] = matched['사업장_시도'].where(~matched['사업장_시도'].isin(HOME), '대경')
    profiles = matched.groupby('지역')[['z_' + col for col in PROFILE]].median()
    if '대경' not in profiles.index:
        raise ValueError('Reference cluster no longer contains home observations')
    reference = profiles.loc[['대경']].to_numpy()
    rows = []
    for region in profiles.index:
        if region == '대경':
            continue
        count = int((matched['지역'] == region).sum())
        # Suppress all results for small groups, not just the count.
        if count < 30:
            continue
        rows.append({'region': region, 'cosine': float(cosine_similarity(reference, profiles.loc[[region]].to_numpy())[0, 0]), 'observation_rows': count})
    rows.sort(key=lambda row: row['cosine'], reverse=True)
    medians = home.groupby('cluster')[CORE].median()
    return {
        'home_rows': len(home), 'other_rows': len(other), 'reference_rows': len(selected),
        'combined_rows': len(combined), 'matched_rows': len(matched),
        'stage1_counts': {str(k): int(v) for k, v in home['cluster'].value_counts().items()},
        'stage2_counts': {str(k): int(v) for k, v in combined['Final_Cluster'].value_counts().items()},
        'stage1_medians': {str(k): {col: float(v) for col, v in row.items()} for k, row in medians.iterrows()},
        'region_ranking': rows,
    }

def baseline_regression(frame, n_jobs=2):
    """Main cells 10,13,14,15: seven log1p features; random 80/20 split."""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    home = frame[frame['사업장_시도'].isin(HOME)].copy()
    x, y = np.log1p(home[REGRESSION]), home['예대마진']
    if not np.isfinite(x.to_numpy()).all() or not np.isfinite(y).all():
        raise ValueError('Regression inputs contain nonfinite values')
    train, test, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=n_jobs)
    model.fit(train, ytrain)
    prediction = model.predict(test)
    return {
        'model': 'RandomForest baseline (100 trees)', 'train_rows': len(train), 'test_rows': len(test),
        'r2': float(r2_score(ytest, prediction)), 'rmse': float(np.sqrt(mean_squared_error(ytest, prediction))),
        'importance': dict(zip(REGRESSION, map(float, model.feature_importances_))),
    }
