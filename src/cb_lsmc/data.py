"""전환사채(CB) LSMC 평가용 시장데이터 수집 (KRX 요청 없음).

- 기초주식: 네이버 금융 차트(fchart.stock.naver.com) 일별 종가
- 무위험금리: 한국은행 ECOS 국고채 3년(817Y002 / 010200000)
- 신용스프레드: ECOS 회사채 3년 AA-(010300000)·BBB-(010320000) - 국고채 3년
  ECOS 'sample' 키는 요청당 10행이라 10일 단위로 나눠 받는다.

결과: data/stock.csv (date, close), data/rates.json
실행: python data.py [종목코드]
"""
import json
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
DATA.mkdir(exist_ok=True)

UA = {'User-Agent': 'Mozilla/5.0'}
ECOS_URL = 'https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/10/{stat}/D/{b}/{e}/{item}'
TICKER = '035900'          # JYP Ent. (가상 CB의 기초주식 예시, 2014-08 이후 데이터)
NAME = 'JYP Ent.'


def ecos_daily(stat, item, bgn, end, key='sample'):
    d0 = date(int(bgn[:4]), int(bgn[4:6]), int(bgn[6:]))
    d1 = date(int(end[:4]), int(end[4:6]), int(end[6:]))
    rows = []
    while d0 <= d1:
        e = min(d0 + timedelta(days=9), d1)
        r = requests.get(ECOS_URL.format(key=key, stat=stat, b=d0.strftime('%Y%m%d'), e=e.strftime('%Y%m%d'), item=item),
                         timeout=30).json()
        for x in r.get('StatisticSearch', {}).get('row', []):
            rows.append(dict(date=x['TIME'], value=float(x['DATA_VALUE'])))
        d0 = e + timedelta(days=1)
        time.sleep(0.2)
    return pd.DataFrame(rows).drop_duplicates('date').sort_values('date')


def stock(ticker):
    r = requests.get('https://fchart.stock.naver.com/sise.nhn',
                     params={'symbol': ticker, 'timeframe': 'day', 'count': 4000, 'requestType': 0},
                     headers=UA, timeout=30)
    recs = []
    for row in re.findall(r'data="([^"]+)"', r.text):
        d, o, h, l, c, v = row.split('|')
        recs.append((pd.Timestamp(d), float(c)))
    return pd.Series(dict(recs), name='close').sort_index()


def rates():
    end = date.today()
    b, e = (end - timedelta(days=20)).strftime('%Y%m%d'), end.strftime('%Y%m%d')
    out = {}
    for key, item in (('ktb3', '010200000'), ('aa3', '010300000'), ('bbb3', '010320000')):
        df = ecos_daily('817Y002', item, b, e)
        last = df.iloc[-1]
        out[key] = float(last['value']) / 100
        out[key + '_date'] = str(last['date'])
    return out


if __name__ == '__main__':
    tk = sys.argv[1] if len(sys.argv) > 1 else TICKER
    s = stock(tk)
    s.index.name = 'date'
    s.to_csv(DATA / 'stock.csv', encoding='utf-8-sig')
    rt = rates()
    rt['ticker'] = tk
    rt['name'] = NAME if tk == TICKER else tk
    (DATA / 'rates.json').write_text(json.dumps(rt, ensure_ascii=False, indent=1), encoding='utf-8')
    print(tk, len(s), s.index.min().date(), '~', s.index.max().date(), s.iloc[-1])
    print(rt)
