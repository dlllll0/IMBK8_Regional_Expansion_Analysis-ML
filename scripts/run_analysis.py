"""Read two approved local CSVs and write aggregate-only historical validation.

No network calls, row export, model persistence, deletion or overwrite.
"""
from pathlib import Path
import argparse
import json
import platform
import sys
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import prepare, historical_clusters, baseline_regression
import pandas as pd
import numpy as np


def load_inputs(data, rates):
    schema = json.loads((ROOT / 'docs/input-schema.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(data, encoding='cp949')
    rate = pd.read_csv(rates, encoding='utf-8-sig')
    for table, required in [(frame, schema['data_columns']), (rate, schema['rate_columns'])]:
        missing = sorted(set(required) - set(table.columns))
        if missing:
            raise ValueError('Missing required columns: ' + ', '.join(missing))
    if rate['기준년월'].isna().any() or rate['기준년월'].duplicated().any():
        raise ValueError('Monthly rate keys must be unique and nonmissing')
    for col in schema['count_columns'] + schema['account_columns']:
        allowed = schema['count_categories'] if col in schema['count_columns'] else schema['account_categories']
        if frame[col].isna().any() or not frame[col].isin(allowed).all():
            raise ValueError('Missing or unknown bucket in ' + col)
    numeric = [c for c in schema['data_columns'] if c not in schema['categorical_columns']]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite numeric input')
    if (frame[numeric] < 0).any().any():
        raise ValueError('Negative raw input; review before applying log transforms')
    return frame, rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--rates', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--with-regression', action='store_true')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Output already exists; choose a new file')
    if args.output.resolve() in [args.data.resolve(), args.rates.resolve()]:
        raise ValueError('Output cannot be an input')
    if args.output.suffix != '.json':
        raise ValueError('Only aggregate JSON output is supported')
    frame, rates = load_inputs(args.data, args.rates)
    print('Input guards passed', flush=True)
    prepared = prepare(frame, rates)
    result = {
        'status': 'historical-specification-recalculation',
        'data': {'rows': len(frame), 'columns': len(frame.columns), 'first_month': int(frame['기준년월'].min()), 'last_month': int(frame['기준년월'].max()), 'months': int(frame['기준년월'].nunique())},
        'environment': {'python': platform.python_version(), **{p: version(p) for p in ['pandas', 'numpy', 'scikit-learn']}},
        'clusters': historical_clusters(prepared),
    }
    print('Historical clustering complete', flush=True)
    if args.with_regression:
        result['baseline_regression'] = baseline_regression(prepared, n_jobs=args.jobs)
        print('Baseline regression complete', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
    print('Aggregate-only JSON saved', flush=True)


if __name__ == '__main__':
    main()
