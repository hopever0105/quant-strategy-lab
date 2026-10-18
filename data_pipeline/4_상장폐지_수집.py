#%% [상폐-0] 안내
"""
상장폐지 종목 수집 (생존편향 완화)   파일명: 4_상장폐지_수집.py

[왜 필요한가]
kor_ticker 는 지금 상장된 종목뿐이라, 백테스트 기간 중에 상장폐지된 종목이 빠져 있다.
이러면 망한 종목이 사라져 수익률이 실제보다 좋게 나온다(생존편향).

[이 파일이 하는 일]
FinanceDataReader 의 상장폐지 종목 목록과 가격을 받아 DB 에 넣는다.
  kor_delisted        : 종목코드, 종목명, 상장폐지일, 사유
  kor_price_delisted  : 날짜, 종목코드, 종가, 시가총액

[실행 순서]
[상폐-1] 설정 -> [상폐-2] 점검 -> [상폐-3] 목록 저장 -> [상폐-4] 가격 수집 -> [상폐-5] 확인
그리고 3_장기데이터_수집.py 를 실행하면 폐지 종목의 재무도 같이 받는다.

[패키지]
  conda activate ta_env
  python -m pip install finance-datareader

[출처와 확인 안 된 것]
호출 방식은 공식 문서가 아니라 제3자 블로그(algolab, 2026-09-03)와 FinanceDataReader 소스에서 가져왔다.
  fdr.StockListing('KRX-DELISTING')                     상장폐지 목록 (열: Symbol Name Market SecuGroup ListingDate DelistingDate Reason ...)
  fdr.DataReader('KRX-DELISTING:종목코드', 시작, 끝)       가격 (열: Open High Low Close Volume Amount MarCap ...)
- 가격 호출은 KRX 에 로그인 없이 요청한다. 로그인이 필요해져서 400 오류가 나면 아래 get_delist_price 가
  1번 파일처럼 pykrx 로그인 세션(KRX_ID / KRX_PW)으로 같은 요청을 보내는 방식으로 넘어간다. (KRX 가 실제로 어떤 방식을 받는지는 확인 안 됨)
- MarCap 의 단위(원인지), 종가가 수정주가인지는 확인 안 됐다. [상폐-2] 에서 MarCap / 종가 를 목록의 ListingShares 와 비교해 본다.
- 폐지 종목 종가가 수정주가가 아니면 액면분할이 있었던 종목의 수익률이 틀어진다.
- 목록에는 주식이 아닌 것(펀드, ELW 등)과 1960년대 폐지 종목까지 4천 개 넘게 들어 있어서, SecuGroup 이 '주권'인 것만 쓴다.
"""


#%% [상폐-1] 설정 + 공통 도구 + 테이블 준비
import os
import re
import time
import random
import getpass
from datetime import date, timedelta
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from tqdm import tqdm
import FinanceDataReader as fdr

PRICE_FROM = '2015-01-01'     # 이 날짜 이후에 상장폐지된 종목만 대상으로 한다
SLEEP_DL = (0.5, 1.2)
MAX_CONSEC_FAIL = 10

# 목록의 열 이름을 자동으로 찾지 못하면 [상폐-2] 출력을 보고 여기에 직접 적는다
CODE_COL = None
NAME_COL = None
DATE_COL = None
REASON_COL = None

DB_USER = 'root'
DB_HOST = '127.0.0.1'
DB_NAME = 'stock_db'
DB_PW = os.environ.get('DB_PW') or getpass.getpass('DB password: ')


def db_connect():
    engine = create_engine(
        f'mysql+pymysql://{DB_USER}:{quote_plus(DB_PW)}@{DB_HOST}:3306/{DB_NAME}')
    con = pymysql.connect(user=DB_USER, password=DB_PW, host=DB_HOST, database=DB_NAME, charset='utf8')
    return engine, con


class BlockedError(Exception):
    pass


class NoData(Exception):
    pass


class Breaker:
    def __init__(self, limit=MAX_CONSEC_FAIL):
        self.n = 0
        self.limit = limit

    def ok(self):
        self.n = 0

    def fail(self):
        self.n += 1
        if self.n >= self.limit:
            raise BlockedError(f'연속 {self.n}번 실패 -> 중단합니다. 시간을 두고 다시 실행하면 이어서 받습니다.')


def nap(rng):
    time.sleep(random.uniform(*rng))


def safe_call(fn, tries=3, base=3):
    for k in range(tries):
        try:
            return fn()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(base * (2 ** k) + random.uniform(0, 1))


def log_done(cur, con, kind, ticker, status):
    cur.execute("""
        insert into crawl_log (대상, 종목코드, 완료일, 상태)
        values (%s,%s,%s,%s) as new
        on duplicate key update 완료일 = new.완료일, 상태 = new.상태
    """, (kind, ticker, date.today().strftime('%Y-%m-%d'), status))
    con.commit()


# ---- KRX 가격 호출 (FinanceDataReader 먼저, 400 등으로 실패하면 로그인 세션 방식으로 전환) ----
KRX_URL = 'https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd'
KRX_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd',
}
DL_METHOD = 'auto'            # 'auto': fdr 먼저 / 'fdr': fdr 만 / 'session': 로그인 세션만
_krx_sess = None
_full_code = None


def krx_session():
    """1번 파일과 같은 방식의 KRX 로그인 세션 (환경변수 KRX_ID, KRX_PW 사용)."""
    global _krx_sess
    if _krx_sess is None:
        from pykrx.website.comm.auth import get_auth_session
        _krx_sess = get_auth_session()
    return _krx_sess


def krx_post(data):
    r = krx_session().post(KRX_URL, data=data, headers=KRX_HEADERS, timeout=30)
    r.raise_for_status()
    if r.text.lstrip()[:1] == '<' or 'LOGOUT' in r.text[:50]:
        raise BlockedError('KRX 가 JSON 대신 HTML 을 보냈습니다 (로그인 풀림 또는 차단 의심)')
    return r.json()


def get_full_code(short_code):
    """상장폐지 종목의 짧은 코드 -> KRX 내부 전체 코드 (FinanceDataReader 소스와 같은 조회)."""
    global _full_code
    if _full_code is None:
        j = krx_post({'mktsel': 'ALL', 'searchText': '', 'bld': 'dbms/comm/finder/finder_listdelisu'})
        _full_code = {b['short_code']: b['full_code'] for b in j['block1']}
    return _full_code.get(short_code)


def to_n(sr):
    return pd.to_numeric(sr.astype(str).str.replace(',', ''), errors='coerce')


def delist_price_session(ticker, start, end):
    """KRX [MDCSTAT23902] 를 로그인 세션으로 2년씩 나눠 요청한다. 결과: Date 인덱스, Close, MarCap 열."""
    full = get_full_code(ticker)
    if full is None:
        raise NoData(ticker)
    s = pd.Timestamp(start)
    e_all = pd.Timestamp(end)
    parts = []
    while True:
        e = min(s + pd.DateOffset(years=2) - pd.Timedelta(days=1), e_all)
        j = krx_post({'bld': 'dbms/MDC/STAT/issue/MDCSTAT23902', 'isuCd': full, 'isuCd2': '',
                      'strtDd': s.strftime('%Y%m%d'), 'endDd': e.strftime('%Y%m%d'),
                      'share': '1', 'money': '1', 'csvxls_isNo': 'false'})
        out = j.get('output') or []
        if out:
            parts.append(pd.DataFrame(out))
        if e >= e_all:
            break
        s = e + pd.Timedelta(days=1)
        nap((0.3, 0.7))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts)
    res = pd.DataFrame({'Date': pd.to_datetime(df['TRD_DD']),
                        'Close': to_n(df['TDD_CLSPRC']),
                        'MarCap': to_n(df['MKTCAP'])})
    return res.set_index('Date').sort_index()


def get_delist_price(ticker, start, end):
    global DL_METHOD
    if DL_METHOD in ('auto', 'fdr'):
        try:
            p = fdr.DataReader(f'KRX-DELISTING:{ticker}', start, end)
            if DL_METHOD == 'auto':
                DL_METHOD = 'fdr'
            return p
        except Exception as e:
            if DL_METHOD == 'fdr':
                raise
            print('FinanceDataReader 호출 실패 -> 로그인 세션 방식으로 전환:', type(e).__name__, str(e)[:80])
            DL_METHOD = 'session'
    return delist_price_session(ticker, start, end)


DDL = [
"""
create table if not exists kor_delisted
(
    종목코드 varchar(6) not null,
    종목명 varchar(50),
    상장폐지일 date,
    사유 varchar(100),
    primary key(종목코드)
);
""",
"""
create table if not exists kor_price_delisted
(
    날짜 date,
    종목코드 varchar(6),
    종가 double,
    시가총액 double,
    primary key(날짜, 종목코드)
);
""",
"""
create table if not exists crawl_log
(
    대상 varchar(10),
    종목코드 varchar(6),
    완료일 date,
    상태 varchar(10),
    primary key(대상, 종목코드)
);
""",
]

engine, con = db_connect()
mycursor = con.cursor()
for q in DDL:
    mycursor.execute(q)
con.commit()
engine.dispose()
con.close()
print('테이블 준비 완료')


#%% [상폐-2] 점검: 상장폐지 목록과 가격이 실제로 오는지 확인
"""
여기서 멈추고 출력을 먼저 확인한다.
- 목록은 4천 개가 넘고 펀드, 1960년대 폐지 종목, 8자리 코드(주식이 아닌 것)까지 섞여 있다. 그래서
  SecuGroup 이 '주권', 코드 6자리, 폐지일이 PRICE_FROM 이후인 것만 골라 그중 하나로 가격을 시험한다.
- 가격은 두 방식을 모두 시험한다: (1) FinanceDataReader  (2) KRX 로그인 세션
- 시가총액 / 종가 가 목록의 ListingShares(상장주식수) 와 비슷하면 MarCap 은 원 단위, 종가는 그 시점 주가다.
"""
dl_raw = fdr.StockListing('KRX-DELISTING')
print('상장폐지 목록 행 수:', len(dl_raw))
print('열 이름:', list(dl_raw.columns))


def find_col(df, keys, manual=None):
    if manual:
        return manual
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in keys):
            return c
    return None


code_col = find_col(dl_raw, ['symbol', 'code', '종목코드'], CODE_COL)
name_col = find_col(dl_raw, ['name', '종목명'], NAME_COL)
date_col = find_col(dl_raw, ['delisting', '폐지일'], DATE_COL)
reason_col = find_col(dl_raw, ['reason', '사유'], REASON_COL)
secu_col = find_col(dl_raw, ['secugroup', '증권구분'])
shares_col = find_col(dl_raw, ['listingshares', '상장주식수'])
print('\n자동 탐지:', dict(code=code_col, name=name_col, date=date_col, reason=reason_col,
                          secu=secu_col, shares=shares_col))
if code_col is None or date_col is None:
    raise RuntimeError('종목코드/폐지일 열을 못 찾았습니다. 위 열 이름을 보고 CODE_COL, DATE_COL 을 적어 주세요.')

if secu_col:
    print('\n증권구분(SecuGroup) 종류')
    print(dl_raw[secu_col].value_counts().to_string())

cand = dl_raw.copy()
cand['_code'] = cand[code_col].astype(str).str.strip()
cand['_dt'] = pd.to_datetime(cand[date_col], errors='coerce')
mask = (cand['_code'].str.fullmatch(r'[0-9A-Z]{6}')
        & (cand['_dt'] >= pd.Timestamp(PRICE_FROM))
        & (cand['_code'].str[-1:] == '0'))
if secu_col:
    mask &= cand[secu_col].astype(str).str.contains('주권')
cand = cand[mask].sort_values('_dt', ascending=False).reset_index(drop=True)
print(f'\n조건(주권, 6자리 코드, 보통주, 폐지일 >= {PRICE_FROM}) 통과: {len(cand)}개')
show = [c for c in [code_col, name_col, 'Market', secu_col, date_col, reason_col, shares_col] if c]
print(cand[show].head(8).to_string())
if len(cand) == 0:
    raise RuntimeError('조건을 통과한 종목이 없습니다. 위 열 이름과 SecuGroup 종류를 붙여 주세요.')

# 폐지된 지 1~2년 된 종목 하나로 시험한다 (너무 오래되면 데이터가 없을 수 있다)
row = cand.iloc[min(len(cand) - 1, 20)]
sample = row['_code']
sample_end = row['_dt'].strftime('%Y-%m-%d')
print('\n점검 종목:', sample, row[name_col], '/ 폐지일', sample_end)


def show_price(tag, p):
    print(f'\n[{tag}] 행 수: {len(p)} / 열: {list(p.columns)}')
    if len(p) == 0:
        return
    print(p[[c for c in ['Close', 'MarCap'] if c in p.columns]].tail(5).to_string())
    if 'MarCap' in p.columns and 'Close' in p.columns:
        implied = (p['MarCap'] / p['Close']).iloc[-1]
        print('마지막 날 MarCap / Close =', f'{implied:,.0f}',
              '/ 목록의 상장주식수 =', row[shares_col] if shares_col else '확인 불가')


try:
    show_price('1) FinanceDataReader', fdr.DataReader(f'KRX-DELISTING:{sample}', PRICE_FROM, sample_end))
except Exception as e:
    print('\n[1) FinanceDataReader] 실패:', type(e).__name__, str(e)[:150])

try:
    show_price('2) KRX 로그인 세션', delist_price_session(sample, PRICE_FROM, sample_end))
except Exception as e:
    print('\n[2) KRX 로그인 세션] 실패:', type(e).__name__, str(e)[:150])


#%% [상폐-3] 목록 정리 + kor_delisted 저장
if secu_col:
    dl_raw = dl_raw[dl_raw[secu_col].astype(str).str.contains('주권')]         # 주식(주권)만. 펀드, ELW 등 제외

dl = pd.DataFrame({
    '종목코드': dl_raw[code_col].astype(str).str.strip(),
    '종목명': dl_raw[name_col].astype(str).str.strip() if name_col else '',
    '상장폐지일': pd.to_datetime(dl_raw[date_col], errors='coerce'),
    '사유': dl_raw[reason_col].astype(str).str.strip() if reason_col else '',
})

n0 = len(dl)
dl = dl.dropna(subset=['상장폐지일'])
dl = dl[dl['상장폐지일'] >= pd.Timestamp(PRICE_FROM)]                 # 기간 안에 폐지된 종목만
dl = dl[dl['종목코드'].str.fullmatch(r'[0-9A-Z]{6}')]                   # 6자리 코드만 (8자리는 주식이 아닌 상품)
dl = dl[dl['종목코드'].str[-1:] == '0']                                # 우선주 제외 (끝자리 0 이 아님)
dl = dl[~dl['종목명'].str.contains('스팩|제[0-9]+호')]                  # 스팩 제외
dl = dl[~dl['종목명'].str.endswith('리츠')]                             # 리츠 제외
dl = dl.drop_duplicates('종목코드', keep='last').reset_index(drop=True)
print(f'목록 {n0}개 -> 조건 통과 {len(dl)}개')
print('폐지 연도별 종목 수')
print(dl['상장폐지일'].dt.year.value_counts().sort_index().to_string())
if reason_col:
    print('\n폐지 사유 상위')
    print(dl['사유'].value_counts().head(10).to_string())

engine, con = db_connect()
mycursor = con.cursor()
query = """
    insert into kor_delisted (종목코드, 종목명, 상장폐지일, 사유)
    values (%s,%s,%s,%s) as new
    on duplicate key update 종목명 = new.종목명, 상장폐지일 = new.상장폐지일, 사유 = new.사유
"""
args = dl[['종목코드', '종목명', '상장폐지일', '사유']].copy()
args['상장폐지일'] = args['상장폐지일'].dt.strftime('%Y-%m-%d')
args['사유'] = args['사유'].str.slice(0, 100)
mycursor.executemany(query, args.values.tolist())
con.commit()
engine.dispose()
con.close()
print('kor_delisted 저장 완료:', len(dl))


#%% [상폐-4] 가격 수집 (이어받기)
"""
폐지일 이전 가격을 PRICE_FROM 부터 받는다. 오늘 처리한 종목은 crawl_log 를 보고 건너뛴다.
FinanceDataReader 가 내부에서 KRX 에 여러 번 요청할 수 있어서 종목 사이에 쉬고, 연속 실패하면 멈춘다.
"""
def pick_col(df, names):
    low = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


engine, con = db_connect()
mycursor = con.cursor()

targets = pd.read_sql('select 종목코드, 상장폐지일 from kor_delisted;', con=engine)
today_str = date.today().strftime('%Y-%m-%d')
done = set(pd.read_sql(f"""
select 종목코드 from crawl_log where 대상 = 'dprice' and 완료일 >= '{(date.today() - timedelta(days=180)).strftime('%Y-%m-%d')}';
""", con=engine)['종목코드'])
todo = targets[~targets['종목코드'].isin(done)].reset_index(drop=True)
print(f'대상 {len(targets)}개 중 최근 처리 {len(done)}개 -> 이번에 받을 종목 {len(todo)}개')

query_price = """
    insert into kor_price_delisted (날짜, 종목코드, 종가, 시가총액)
    values (%s,%s,%s,%s) as new
    on duplicate key update 종가 = new.종가, 시가총액 = new.시가총액
"""

error_list = []
br = Breaker()

try:
    for i in tqdm(range(len(todo))):
        ticker = todo['종목코드'][i]
        end = pd.to_datetime(todo['상장폐지일'][i]).strftime('%Y-%m-%d')
        try:
            p = safe_call(lambda: get_delist_price(ticker, PRICE_FROM, end))
            if p is None or len(p) == 0:
                raise NoData(ticker)

            c_close = pick_col(p, ['Close'])
            c_mcap = pick_col(p, ['MarCap', 'Marcap'])
            if c_close is None:
                raise ValueError('Close 열이 없습니다: ' + str(list(p.columns)))

            out = pd.DataFrame({
                '날짜': pd.to_datetime(p.index).strftime('%Y-%m-%d'),
                '종목코드': ticker,
                '종가': p[c_close].values,
                '시가총액': p[c_mcap].values if c_mcap else np.nan,
            })
            out = out.replace([np.inf, -np.inf], np.nan)
            out = out.astype(object).where(out.notna(), None)

            mycursor.executemany(query_price, out[['날짜', '종목코드', '종가', '시가총액']].values.tolist())
            con.commit()
            log_done(mycursor, con, 'dprice', ticker, 'ok')
            br.ok()

        except BlockedError:
            raise
        except NoData:
            log_done(mycursor, con, 'dprice', ticker, 'nodata')
        except Exception as e:
            print(ticker, type(e).__name__, str(e)[:80])
            error_list.append(ticker)
            br.fail()

        nap(SLEEP_DL)

except BlockedError as e:
    print('중단:', e)

finally:
    engine.dispose()
    con.close()

print('실패 종목 수:', len(error_list))


#%% [상폐-5] 수집 결과 확인
engine, con = db_connect()

print(pd.read_sql("""
select 'kor_delisted' as 테이블, count(*) as 행수, max(상장폐지일) as 최근일 from kor_delisted
union all
select 'kor_price_delisted', count(*), max(날짜) from kor_price_delisted;
""", con=engine))

print(pd.read_sql("""
select 상태, count(*) as 종목수 from crawl_log where 대상 = 'dprice' group by 상태;
""", con=engine))

n_with_price = pd.read_sql('select count(distinct 종목코드) as n from kor_price_delisted;', con=engine)['n'][0]
print('가격이 들어온 폐지 종목 수:', n_with_price)

engine.dispose()
con.close()