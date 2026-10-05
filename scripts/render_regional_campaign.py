"""Render aggregate regional use rates; overwrite only the named public figure."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False
j=json.loads((ROOT/'assets/regional-campaign-results.json').read_text(encoding='utf-8'))
g=j['adjustment']['groups'];regions=['대경','부산광역시','경상남도','울산광역시']
fig,ax=plt.subplots(figsize=(10,5.5))
for i,(key,label,color) in enumerate([('digital_use','디지털 거래','#277f99'),('auto_use','자동이체','#659b76'),('card_use','카드 사용','#c99446')]):
 vals=[g[r]['means'][key]*100 for r in regions]
 bars=ax.bar([x+(i-1)*.25 for x in range(4)],vals,width=.23,color=color,label=label)
 for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.1f}',ha='center',fontsize=10)
ax.set_xticks(range(4),['대경 기준','부산','경남','울산'])
ax.set(ylim=(0,118),ylabel='거래금액 > 0인 월별 관측 비율 (%)',title='구성을 맞춰도 남아 있는 지역별 이용 차이')
ax.legend(loc='upper right',ncol=3,frameon=False)
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.text(.04,.06,'연도·업종·동월 예금+대출 규모 조정 | 공통 제조업·도소매업 층 27,428행',fontsize=10)
fig.text(.04,.02,'고유 고객 이용률·잠재 수요가 아님 | 공통 표본은 네 지역 후보 관측의 34.7%',fontsize=10,color='#555555')
fig.tight_layout(rect=(0,.10,1,1))
fig.savefig(ROOT/'assets/regional-campaign-gaps.png',dpi=160)
plt.close(fig)
print('Regional aggregate figure rendered.')
