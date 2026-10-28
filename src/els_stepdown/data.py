"""ELS 평가용 시장데이터 수집 (KRX 요청 없음).

- KOSPI200: 네이버 금융 차트(fchart.stock.naver.com, 심볼 KPI200) 일별 종가
- S&P500, 원/달러: Yahoo Finance(^GSPC, KRW=X)
- 원화 무위험금리: 한국은행 ECOS 국고채 3년(817Y002 / 010200000), sample 키
- 달러 무위험금리: FRED 미 국채 3년(DGS3)

결과: data/market.csv (날짜, KS200, SPX, USDKRW), data/rates.json
실행: python data.py
"""
import io
import json
import re
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
DATA.mkdir(exist_ok=True)

UA = {'User-Agent': 'Mozilla/5.0'}
ECOS_URL = 'https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/10/{stat}/D/{b}/{e}/{item}'


def ecos_daily(stat, item, bgn, end, key='sample'):
    """한국은행 ECOS 일별 시계열. 'sample' 키는 요청당 10행이라 10일 단위로 나눠 받는다."""
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
    return pd.DataFrame(rows).drop_duplicates('date')


def kospi200():
    r = requests.get('https://fchart.stock.naver.com/sise.nhn',
                     params={'symbol': 'KPI200', 'timeframe': 'day', 'count': 8000, 'requestType': 0},
                     headers=UA, timeout=30)
    rows = re.findall(r'data="([^"]+)"', r.text)
    recs = []
    for row in rows:
        d, o, h, l, c, v = row.split('|')
        recs.append((pd.Timestamp(d), float(c)))
    return pd.Series(dict(recs), name='KS200').sort_index()


def yahoo(ticker, name):
    d = yf.download(ticker, start='1995-01-01', progress=False, auto_adjust=False)
    s = d['Close'].iloc[:, 0] if isinstance(d['Close'], pd.DataFrame) else d['Close']
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.rename(name).dropna()


def rates():
    end = date.today()
    krw = ecos_daily('817Y002', '010200000', (end - timedelta(days=20)).strftime('%Y%m%d'), end.strftime('%Y%m%d'))
    krw = krw.sort_values('date').iloc[-1]
    fred = pd.read_csv(io.StringIO(requests.get('https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3',
                                                timeout=90).text))
    fred = fred[pd.to_numeric(fred['DGS3'], errors='coerce').notna()].iloc[-1]
    return {'r_krw_3y': float(krw['value']) / 100, 'r_krw_date': str(krw['date']),
            'r_usd_3y': float(fred['DGS3']) / 100, 'r_usd_date': str(fred['observation_date'])}


if __name__ == '__main__':
    ks = kospi200()
    spx = yahoo('^GSPC', 'SPX')
    fx = yahoo('KRW=X', 'USDKRW')
    m = pd.concat([ks, spx, fx], axis=1)
    m.index.name = 'date'
    m.to_csv(DATA / 'market.csv', encoding='utf-8-sig')
    rt = rates()
    (DATA / 'rates.json').write_text(json.dumps(rt, ensure_ascii=False, indent=1), encoding='utf-8')
    for c in m.columns:
        s = m[c].dropna()
        print(c, len(s), s.index.min().date(), '~', s.index.max().date(), round(s.iloc[-1], 2))
    print(rt)
