"""데이터 수집 (KRX 요청 없음). 이미 받은 파일은 건너뛰어 끊겨도 이어받는다.

- 일별 시세: MySQL stock_db.us_price (종가=분할·스핀오프 보정, 수정종가=배당까지 보정). 없으면 yfinance.
- 분할·스핀오프 배율: yfinance splits (스핀오프도 소수 배율로 기록됨)
- 월말 시가총액: companiesmarketcap.com (시가총액 가중 비중의 발행주식수 계산용)
- WBA: 일별 시세가 없어 companiesmarketcap 월말 총수익 가격·시가총액·발행주식수
- 다우 지수(^DJI): yfinance (가격가중 복제 검증용), DIA: DB
- 환율·금리: FRED DEXKOUS(원/달러), DTB3(미 3개월 국채), IR3TIB01KRM156N(한국 3개월 금리, 월)

결과: data/*.parquet, data/cmc/*.json, data/splits.json
진행 상황: quant/logs/dow_price_vs_cap.log
실행: python data.py
"""
import io
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import pymysql
import requests
import yfinance as yf
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
QUANT = HERE.parents[1]
sys.path.insert(0, str(QUANT))
sys.path.insert(0, str(HERE))                 # 저장소 사본은 같은 폴더의 price_adjust.py를 쓴다
import price_adjust as pa  # noqa: E402
import universe as U  # noqa: E402

DATA = HERE / 'data'
CMC = DATA / 'cmc'
CMC.mkdir(parents=True, exist_ok=True)
LOGS = QUANT / 'logs' if (QUANT / 'logs').exists() else HERE / 'logs'
LOGS.mkdir(exist_ok=True)
logging.basicConfig(filename=LOGS / 'dow_price_vs_cap.log', level=logging.INFO,
                    format='%(asctime)s %(message)s', encoding='utf-8')
log = logging.getLogger()
UA = {'User-Agent': 'Mozilla/5.0'}
FROM = '2009-06-01'


def db():
    load_dotenv(QUANT / '.env')
    return pymysql.connect(host=os.getenv('DB_HOST', '127.0.0.1'), port=int(os.getenv('DB_PORT', 3306)),
                           user=os.getenv('DB_USER'), password=os.getenv('DB_PASSWORD'), db=os.getenv('DB_NAME', 'stock_db'))


def load_prices():
    out = DATA / 'prices_raw.parquet'
    if out.exists():
        return pd.read_parquet(out)
    tickers = sorted({U.price_ticker(m) for m in U.all_members() if m not in U.EXCLUDED | U.MONTHLY_ONLY} | {'DIA'})
    try:
        conn = db()
        q = ("select 날짜 as date, 티커 as ticker, 종가 as close, 수정종가 as adj from us_price "
             "where 티커 in (%s) and 날짜 >= %%s" % ','.join(['%s'] * len(tickers)))
        df = pd.read_sql(q, conn, params=tickers + [FROM])
        conn.close()
    except Exception as e:                           # DB가 없으면 전부 yfinance로 받는다
        log.info(f'DB 접속 실패 → yfinance 사용: {e}')
        df = pd.DataFrame(columns=['date', 'ticker', 'close', 'adj'])
    df['date'] = pd.to_datetime(df['date'])
    have = set(df['ticker'])
    for t in sorted(set(tickers) - have):           # DB에 없는 종목은 yfinance
        h = yf.Ticker(t).history(start=FROM, auto_adjust=False)
        h.index = h.index.tz_localize(None)
        df = pd.concat([df, pd.DataFrame({'date': h.index, 'ticker': t, 'close': h['Close'].values,
                                          'adj': h['Adj Close'].values})])
        log.info(f'yfinance 보강 {t} {len(h)}행')
    df.to_parquet(out)
    log.info(f'DB 시세 {df.ticker.nunique()}종목 {len(df)}행')
    return df


def load_splits():
    out = DATA / 'splits.json'
    if out.exists():
        return json.loads(out.read_text(encoding='utf-8'))
    res = {}
    for m in U.all_members():
        if m in U.EXCLUDED | U.MONTHLY_ONLY:
            continue
        t = U.price_ticker(m)
        s = yf.Ticker(t).splits
        s.index = s.index.tz_localize(None)
        res[t] = {str(d.date()): float(v) for d, v in s[s.index >= FROM].items()}
        time.sleep(0.3)
    out.write_text(json.dumps(res, indent=1), encoding='utf-8')
    log.info(f'분할·스핀오프 기록 {sum(len(v) for v in res.values())}건')
    return res


def cmc(slug, page):
    """companiesmarketcap 페이지의 차트 데이터(JSON) → DataFrame(date, value). 캐시 사용."""
    f = CMC / f'{slug}__{page}.json'
    if not f.exists():
        r = requests.get(f'https://companiesmarketcap.com/{slug}/{page}/', headers=UA, timeout=30)
        r.raise_for_status()
        m = re.search(r'data = (\[.*?\]);', r.text)
        f.write_text(m.group(1) if m else '[]', encoding='utf-8')
        log.info(f'cmc {slug} {page} 받음')
        time.sleep(1.5)
    a = json.loads(f.read_text(encoding='utf-8'))
    if not a:
        return pd.Series(dtype=float)
    key = [k for k in a[0] if k != 'd'][0]
    s = pd.Series({pd.Timestamp(x['d'], unit='s').normalize(): float(x[key]) for x in a}).sort_index()
    return s[~s.index.duplicated(keep='last')]


def load_mcap():
    out = DATA / 'mcap.parquet'
    if out.exists():
        return pd.read_parquet(out)
    cols = {}
    for m, slug in U.CMC_SLUG.items():
        cols[m] = cmc(slug, 'marketcap') * 1e5      # 단위: 10만 달러 → 달러 (AAPL 2010말 2,971억 달러로 검증)
    df = pd.DataFrame(cols).sort_index()
    df.to_parquet(out)
    return df


def load_wba():
    out = DATA / 'wba_monthly.parquet'
    if out.exists():
        return pd.read_parquet(out)
    tr = cmc('walgreens-boots-alliance', 'stock-price-history')          # 월말 배당 재투자 가격
    mc = cmc('walgreens-boots-alliance', 'marketcap') * 1e5
    sh = cmc('walgreens-boots-alliance', 'shares-outstanding')
    df = pd.DataFrame({'tr': tr, 'mcap': mc, 'shares': sh}).sort_index()
    df['shares'] = df['shares'].interpolate(limit_direction='both')
    df['raw'] = df['mcap'] / df['shares']                                 # 실제 주가 = 시가총액 / 발행주식수
    df = df.dropna(subset=['tr', 'raw'])
    df.to_parquet(out)
    return df


def fred(series):
    f = DATA / f'fred_{series}.parquet'
    if f.exists():
        return pd.read_parquet(f)[series]
    t = requests.get(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}', timeout=90).text
    d = pd.read_csv(io.StringIO(t))
    d.columns = ['date', series]
    d['date'] = pd.to_datetime(d['date'])
    d[series] = pd.to_numeric(d[series], errors='coerce')
    d = d.dropna().set_index('date')
    d.to_parquet(f)
    log.info(f'FRED {series} {len(d)}행')
    return d[series]


def load_index():
    out = DATA / 'dji.parquet'
    if out.exists():
        return pd.read_parquet(out)
    h = yf.Ticker('^DJI').history(start=FROM, auto_adjust=False)
    h.index = h.index.tz_localize(None)
    df = h[['Close']].rename(columns={'Close': 'DJI'})
    df.to_parquet(out)
    return df


def build_panel():
    """보정까지 끝난 종목별 일별 패널: close(보정), raw(실제가), div(배당수익률)"""
    px = load_prices()
    sp = load_splits()
    close, raw, div, notes = {}, {}, {}, []
    n_fix = 0
    for t, g in px.groupby('ticker'):
        g = g.set_index('date').sort_index()[['close', 'adj']].dropna()
        s = pd.Series({pd.Timestamp(k): v for k, v in sp.get(t, {}).items()}, dtype=float)
        g, n, fx = pa.fix_split_gaps(g, s)
        n_fix += n
        notes += [(t,) + x for x in fx]
        close[t] = g['close']
        raw[t] = pa.raw_price(g['close'], s)
        div[t] = pa.div_yield(g['close'], g['adj'])
    return (pd.DataFrame(close).sort_index(), pd.DataFrame(raw).sort_index(), pd.DataFrame(div).sort_index(),
            n_fix, notes)


if __name__ == '__main__':
    log.info('수집 시작')
    px = load_prices()
    sp = load_splits()
    mc = load_mcap()
    wba = load_wba()
    dji = load_index()
    for s in ['DEXKOUS', 'DTB3', 'IR3TIB01KRM156N']:
        fred(s)
    close, raw, div, n_fix, notes = build_panel()
    print('시세', close.shape, close.index.min().date(), '~', close.index.max().date())
    print('분할 단절 보정', n_fix, '건', notes)
    print('분할·스핀오프 기록', sum(len(v) for v in sp.values()), '건')
    print('시가총액', mc.shape, mc.index.min().date(), '~', mc.index.max().date())
    print('WBA', wba.loc['2018':'2024'].head(3))
    for t in close.columns:
        j = pa.detect_jumps(close[t].dropna())
        if len(j):
            print('남은 단절', t, [(str(d.date()), round(v, 3)) for d, v in j.items()])
    log.info('수집 끝')
