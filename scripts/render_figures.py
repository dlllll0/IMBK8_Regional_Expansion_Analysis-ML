"""Render a regional ranking from stored aggregate results. Never reads raw data."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output filename')
    if args.output.suffix.lower() != '.png':
        raise ValueError('PNG output is required')
    root = Path(__file__).resolve().parents[1]
    results = json.loads((root / 'assets/stored-results.json').read_text(encoding='utf-8'))
    available = {f.name for f in font_manager.fontManager.ttflist}
    korean_font = next((f for f in ['Malgun Gothic', 'Noto Sans CJK KR', 'NanumGothic', 'AppleGothic'] if f in available), None)
    if not korean_font:
        raise RuntimeError('Install a Korean font before rendering')
    plt.rcParams.update({'font.family': korean_font, 'axes.unicode_minus': False, 'font.size': 12})
    fig, ax = plt.subplots(figsize=(10, 4.5))
    rows = results['region_top3']
    labels = [r['region'] for r in rows]
    values = [r['cosine'] for r in rows]
    ax.barh(labels, values, color=['#008754', '#28a877', '#77c4a5'], height=.52)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xlabel('코사인 유사도 · 기존 분석의 표준화 좌표 기준')
    ax.set_title('대구·경북 기준집단과 유사한 상위 3개 지역', loc='left', pad=20, weight='bold')
    for i, row in enumerate(rows):
        ax.text(row['cosine']+.012, i, f"{row['cosine']:.4f}", va='center', weight='bold')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.grid(axis='x', alpha=.16)
    ax.set_axisbelow(True)
    fig.text(.12, .035, '노트북 저장 집계값 · 로컬 재계산에서 소수 넷째 자리 일치 · 진출 성공확률을 의미하지 않음', fontsize=10, color='#53616c')
    fig.tight_layout(rect=(0, .08, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as file:
        fig.savefig(file, format='png', dpi=170, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
