# -*- coding: utf-8 -*-
#%% [린치-0] 안내
"""
피터 린치 전략 백테스트: 버전 A(알아온 7조건) / 버전 B(문헌 기반)

먼저 '1_자료수집.py' 를 보고 오세요.
이미 DB(stock_db)에 kor_ticker, kor_price, dart_fs 가 있으면 여기서부터 진행하세요.

[이 파일이 하는 일]
1) 린치 전략 조건으로 종목 스크리닝
2) 버전 A, 버전 B 를 같은 방식으로 백테스트해서 비교

[DB 비밀번호]
코드에 쓰지 않는다. 환경변수 DB_PW 에 넣거나 실행할 때 입력한다.

[패키지]
기존 + matplotlib.   conda activate ta_env  ->  python -m pip install matplotlib
Spyder 에서는 도구 > 환경설정 > Python 인터프리터 > 다음 인터프리터 사용 에서
ta_env 의 python.exe 를 선택한 뒤 콘솔을 다시 시작한다.
"""

#%% [린치-1] 설정 + DB 연결 + 테이블 확인
import os
import getpass
import runpy
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect

RUN_DOWNLOAD = False   # True: 자료수집 파일을 실행한다 / False: 건너뛰고 DB 에서 바로 읽는다
DOWNLOAD_FILE = r'C:\경로\1_자료수집.py'   # 본인 파일 위치로 고칠 것

FS_SOURCE = 'dart'     # 'fnguide': kor_fs / 'dart': dart_fs

if RUN_DOWNLOAD:
    runpy.run_path(DOWNLOAD_FILE)

DB_USER = 'root'
DB_HOST = '127.0.0.1'
DB_NAME = 'stock_db'
DB_PW = os.environ.get('DB_PW') or getpass.getpass('DB password: ')

engine = create_engine(
    f'mysql+pymysql://{DB_USER}:{quote_plus(DB_PW)}@{DB_HOST}:3306/{DB_NAME}')

need_tables = ['kor_ticker', 'kor_price', 'kor_fs' if FS_SOURCE == 'fnguide' else 'dart_fs']
have_tables = inspect(engine).get_table_names()

missing = []
for t in need_tables:
    if t not in have_tables:
        missing.append(t)
    else:
        n = pd.read_sql(f'select count(*) as n from {t};', con=engine)['n'][0]
        if n == 0:
            missing.append(t + ' (비어 있음)')

if missing:
    raise RuntimeError('DB 에 없는 테이블: ' + ', '.join(missing) +
                       '\n-> RUN_DOWNLOAD = True 로 바꿔 다시 실행하거나, 자료수집 파일을 먼저 실행하세요.')
print('테이블 확인 완료')

#%% [린치-2] 린치 전략에 필요한 계정이 DB 에 있는지 확인
# 필요한 값: 순이익(PEG 성장률), 재고자산(재고/매출, 재고증가율), 부채총계와 자본총계(부채비율),
#            자산총계(ROA), 매출액, 배당(배당수익률)
# 아직 확인된 계정은 매출액(sales), 당기순이익(ni), 자본총계(equity), 유동자산(ca), 유동부채(cl), 비유동부채(ltd) 뿐이다.
if FS_SOURCE == 'fnguide':
    acc_in_db = pd.read_sql("select distinct 계정 from kor_fs where 공시구분 = 'y';",
                            con=engine)['계정'].tolist()
else:
    acc_in_db = pd.read_sql('select distinct 계정 from dart_fs;', con=engine)['계정'].tolist()

print('전체 계정 수:', len(acc_in_db))
print('재고 관련   :', [a for a in acc_in_db if ('재고' in a) or ('inv' in a.lower())])
print('부채 관련   :', [a for a in acc_in_db if ('부채' in a) or ('liab' in a.lower()) or ('debt' in a.lower())])
print('자산 관련   :', [a for a in acc_in_db if ('자산' in a) or ('asset' in a.lower())])
print('배당 관련   :', [a for a in acc_in_db if ('배당' in a) or ('div' in a.lower())])
print('지배/이익   :', [a for a in acc_in_db if ('지배' in a) or ('이익' in a)])

# 배당수익률은 과거 시점 값이 필요해서 kor_ticker 의 컬럼도 같이 본다
print('\nkor_ticker 컬럼:', pd.read_sql('select * from kor_ticker limit 1;', con=engine).columns.tolist())

#%% [린치-2b] dart_fs 계정 목록과 연도별 종목 수 확인
print('dart_fs 계정 11개:', sorted(acc_in_db))

cov = pd.read_sql("""
select 계정, year(기준일) as 연도, count(distinct 종목코드) as 종목수
from dart_fs
group by 계정, year(기준일);
""", con=engine)
print(cov.pivot(index='계정', columns='연도', values='종목수').fillna(0).astype(int).to_string())

#%% [린치-3] 기준값
"""
버전 A : 알아온 7조건
         PER < 30, 0 < PEG < 1.8, 재고/매출 < 5%, 부채비율 < 150%, ROE > 5%, ROA > 1%, 배당수익률 > 3%
버전 B : 문헌에서 확인된 기준
         3년 순이익 CAGR 20~50%, 0 < PEG < 1, PER < 40, 부채비율 < 80%, 재고증가율 < 매출증가율

값을 None 으로 두면 그 조건은 쓰지 않는다.
PEG, ROE 등의 정의
  PER        : 시가총액 / 최근 연간 순이익
  growth     : 최근 PEG_YEARS 년 순이익 CAGR (시작·끝 순이익이 모두 양수일 때만 계산)
  PEG        : PER / (growth x 100)
  부채비율   : 부채총계 / 자본총계 x 100
  ROE, ROA   : 순이익 / 자본총계, 순이익 / 자산총계 (x 100)
  재고/매출  : 재고자산 / 매출액
  배당수익률 : 사업보고서에 공시된 현금배당수익률(%) (보통주). 수정주가와 명목 배당금을 섞으면 액면분할 종목이 틀어져서 공시값을 쓴다
"""
acc_in_db = pd.read_sql('select distinct 계정 from dart_fs;', con=engine)['계정'].tolist()
HAVE_INV = 'inv' in acc_in_db                                        # dart_fs 에 재고자산(inv)이 있으면 재고 조건을 쓴다
HAVE_DIV = 'dart_div_raw' in inspect(engine).get_table_names()       # 배당 원본 테이블이 있으면 배당 조건을 쓴다
NOINV_AS_ZERO = True         # 재고자산 계정이 아예 없는 회사(crawl_log 'noinv')를 재고 0 으로 보고 재고 조건을 통과시킨다. False 면 제외

PEG_YEARS = 3
DISCLOSURE_LAG = 90          # 결산일 + 90일 이후에야 재무를 알 수 있다고 본다
STALE_DAYS = 450             # 가장 최근 재무가 이보다 오래됐으면 제외 (상장폐지 종목 등)
REBAL_MD = '04-01'
TRADING_COST = 0.003         # 왕복 비용 가정치

MODES = {
    'A': dict(MAX_PER=30, MAX_PEG=1.8, MIN_GROWTH=None, MAX_GROWTH=None,
              MAX_DEBT=150, MIN_ROE=5, MIN_ROA=1,
              MAX_INV_SALES=0.05, INV_LT_SALES=False, MIN_DY=3.0),
    'B': dict(MAX_PER=40, MAX_PEG=1.0, MIN_GROWTH=0.20, MAX_GROWTH=0.50,
              MAX_DEBT=80, MIN_ROE=None, MIN_ROA=None,
              MAX_INV_SALES=None, INV_LT_SALES=True, MIN_DY=None),
}

INCLUDE_DELISTED = True      # True: 상장폐지 종목 포함 (kor_delisted, kor_price_delisted 필요)

ACC_LIST = ['sales', 'ni', 'equity', 'assets', 'liab', 'inv']
print('재고 조건 사용:', HAVE_INV, '/ 배당 조건 사용:', HAVE_DIV, '/ 재고 계정 없는 회사를 재고 0 으로 처리:', NOINV_AS_ZERO)

#%% [린치-4] 데이터 불러오기
if INCLUDE_DELISTED:
    for t in ['kor_delisted', 'kor_price_delisted']:
        if t not in inspect(engine).get_table_names():
            raise RuntimeError(f'{t} 테이블이 없습니다. INCLUDE_DELISTED = False 로 바꾸거나 상장폐지 수집 파일을 먼저 실행하세요.')

ticker_list = pd.read_sql("""
select 종목코드, 종목명, 시장구분, 종가, 시가총액, 기준일
from kor_ticker
where 기준일 = (select max(기준일) from kor_ticker)
  and 종목구분 = '보통주';
""", con=engine)

acc_sql = ', '.join(f"'{a}'" for a in ACC_LIST if a in acc_in_db)
fs_y = pd.read_sql(f"""
select 종목코드, 계정, 기준일, 값
from dart_fs
where 계정 in ({acc_sql});
""", con=engine)
fs_y['기준일'] = pd.to_datetime(fs_y['기준일'])

price = pd.read_sql("select 날짜, 종목코드, 종가 from kor_price;", con=engine)
price['날짜'] = pd.to_datetime(price['날짜'])
price_wide = price.pivot(index='날짜', columns='종목코드', values='종가').sort_index()
del price

# 과거 시가총액 = 현재 주식수 x 과거 수정주가 (유상증자·자사주 소각이 많은 종목은 오차가 생긴다)
ticker_list['shares'] = (ticker_list['시가총액'] / ticker_list['종가']).replace([np.inf, -np.inf], np.nan)
shares = ticker_list.set_index('종목코드')['shares']
universe = set(ticker_list['종목코드'])
name_map = ticker_list.set_index('종목코드')[['종목명', '시장구분']]
mcap_wide = price_wide.mul(shares.reindex(price_wide.columns), axis=1)

if INCLUDE_DELISTED:
    dl_info = pd.read_sql('select 종목코드, 종목명, 상장폐지일 from kor_delisted;', con=engine)
    dl_price = pd.read_sql('select 날짜, 종목코드, 종가, 시가총액 from kor_price_delisted;', con=engine)
    dl_price['날짜'] = pd.to_datetime(dl_price['날짜'])
    dl_price = dl_price[~dl_price['종목코드'].isin(price_wide.columns)]    # 겹치면 현재 상장 쪽 데이터를 쓴다
    dl_p = dl_price.pivot(index='날짜', columns='종목코드', values='종가')
    dl_m = dl_price.pivot(index='날짜', columns='종목코드', values='시가총액')
    all_idx = price_wide.index.union(dl_p.index)
    price_wide = price_wide.reindex(all_idx).join(dl_p)
    mcap_wide = mcap_wide.reindex(all_idx).join(dl_m)
    name_map = pd.concat([name_map, dl_info.set_index('종목코드')[['종목명']].assign(시장구분='폐지')])
    universe |= set(dl_p.columns)
    print('상장폐지 종목 포함:', dl_p.shape[1], '개')

fs_y = fs_y[fs_y['종목코드'].isin(universe)]

# 재고자산 계정이 없는 회사 목록 (수집 단계에서 'noinv' 로 기록해 둔 종목)
NOINV_SET = set()
if HAVE_INV:
    NOINV_SET = set(pd.read_sql("select 종목코드 from crawl_log where 대상 = 'dart_inv' and 상태 = 'noinv';",
                                con=engine)['종목코드'])
    print('재고자산 계정이 없는 회사:', len(NOINV_SET), '개')

# 배당수익률: 사업보고서 하나에 당기/전기/전전기가 들어 있어서 연도별로 펼친다
# stock_knd 가 '보통주', '보통주식' 이면 보통주, '-' 는 종류 구분이 없는 회사. 우선주와 종류주는 쓰지 않는다
div_long = pd.DataFrame(columns=['종목코드', '연도', 'dy', 'fy_end'])
if HAVE_DIV:
    dv = pd.read_sql("""
    select 종목코드, 보고서연도, stock_knd, thstrm, frmtrm, lwfr, stlm_dt
    from dart_div_raw
    where se = '현금배당수익률(%%)';
    """, con=engine)
    dv = dv[dv['stock_knd'].isin(['보통주', '보통주식', '-'])].copy()
    dv['kind_rank'] = np.where(dv['stock_knd'] == '-', 1, 0)         # 보통주 표기가 있으면 그 행을 우선한다
    dv['stlm_year'] = pd.to_datetime(dv['stlm_dt'], errors='coerce').dt.year

    parts = []
    for col, back in (('thstrm', 0), ('frmtrm', 1), ('lwfr', 2)):
        t = dv[['종목코드', '보고서연도', 'kind_rank']].copy()
        t['연도'] = dv['stlm_year'] - back
        # '-' 는 배당이 없다는 뜻이라 0 으로 둔다
        t['dy'] = pd.to_numeric(dv[col].astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0)
        parts.append(t)
    div_long = pd.concat(parts).dropna(subset=['연도'])
    div_long = div_long.sort_values(['종목코드', '연도', 'kind_rank', '보고서연도'],
                                    ascending=[True, True, False, True])
    div_long = div_long.drop_duplicates(['종목코드', '연도'], keep='last')     # 보통주 우선, 같으면 최근 보고서
    div_long['연도'] = div_long['연도'].astype(int)
    div_long['fy_end'] = pd.to_datetime(div_long['연도'].astype(str) + '-12-31')
    div_long = div_long[div_long['종목코드'].isin(universe)]
    print('배당수익률 데이터: 종목', div_long['종목코드'].nunique(), '개 /', len(div_long), '행')

n_years = fs_y[fs_y['계정'] == 'ni'].groupby('종목코드')['기준일'].nunique()
print('대상 종목 수:', len(universe))
print('재무 기준일 범위:', fs_y['기준일'].min().date(), '~', fs_y['기준일'].max().date())
print('주가 범위       :', price_wide.index.min().date(), '~', price_wide.index.max().date())
print('종목별 연간 재무 연수 분포 (연수: 종목 수)')
print(n_years.value_counts().sort_index())

#%% [린치-5] 지표 계산 + 조건 적용 함수
def lynch_metrics(as_of):
    """as_of 시점에 알 수 있었던 재무만으로 종목별 지표를 계산한다. 조건 적용은 apply_conds 에서 한다."""
    as_of = pd.Timestamp(as_of)
    fs = fs_y[fs_y['기준일'] + pd.Timedelta(days=DISCLOSURE_LAG) <= as_of]
    if fs.empty:
        return pd.DataFrame()

    wide = fs.pivot_table(index=['종목코드', '기준일'], columns='계정', values='값').reset_index()
    wide.columns.name = None
    wide = wide.sort_values(['종목코드', '기준일'])
    for c in ACC_LIST:
        if c not in wide.columns:
            wide[c] = np.nan

    # PEG_YEARS 년 전 값 (연도가 빠진 종목은 아래 span 검사에서 걸러진다)
    g = wide.groupby('종목코드')
    for c in ['sales', 'ni', 'inv']:
        wide[c + '_prev'] = g[c].shift(PEG_YEARS)
    wide['d_prev'] = g['기준일'].shift(PEG_YEARS)
    last = wide.groupby('종목코드').tail(1).set_index('종목코드')

    span = (last['기준일'] - last['d_prev']).dt.days
    span_ok = span.between(365 * PEG_YEARS - 10, 366 * PEG_YEARS + 10)
    fresh = last['기준일'] >= as_of - pd.Timedelta(days=STALE_DAYS)

    ni, eq = last['ni'], last['equity']
    valid_g = span_ok & (ni > 0) & (last['ni_prev'] > 0)
    growth = ((ni / last['ni_prev']) ** (1 / PEG_YEARS) - 1).where(valid_g)

    mcap = mcap_wide.loc[:as_of].iloc[-1] / 100000000        # 억원

    df = pd.DataFrame(index=last.index)
    df['sales'] = last['sales']
    df['mcap'] = mcap.reindex(df.index)
    df['PER'] = (df['mcap'] / ni).where(ni > 0)
    df['growth'] = growth
    df['PEG'] = (df['PER'] / (growth * 100)).where(growth > 0)
    df['debt_ratio'] = (last['liab'] / eq * 100).where(eq > 0)
    df['ROE'] = (ni / eq * 100).where(eq > 0)
    df['ROA'] = (ni / last['assets'] * 100).where(last['assets'] > 0)
    df['inv_sales'] = (last['inv'] / last['sales']).where(last['sales'] > 0)
    inv_g = (last['inv'] / last['inv_prev'] - 1).where((last['inv_prev'] > 0) & span_ok)
    sales_g = (last['sales'] / last['sales_prev'] - 1).where((last['sales_prev'] > 0) & span_ok)
    df['inv_growth'] = inv_g
    df['sales_growth'] = sales_g
    df['no_inv'] = df.index.isin(NOINV_SET)

    # 배당수익률: as_of 에 공시가 끝난 가장 최근 사업연도의 값
    if HAVE_DIV:
        dvx = div_long[(div_long['fy_end'] + pd.Timedelta(days=DISCLOSURE_LAG) <= as_of) &
                       (div_long['fy_end'] >= as_of - pd.Timedelta(days=STALE_DAYS))]
        dy_last = dvx.sort_values(['종목코드', '연도']).groupby('종목코드').tail(1).set_index('종목코드')['dy']
        df['DY'] = dy_last.reindex(df.index)
    else:
        df['DY'] = np.nan

    df = df[fresh.reindex(df.index).fillna(False)]
    df = df.join(name_map)
    return df[df.index.isin(universe)]


def apply_conds(df, p):
    """버전별 기준값 p 를 적용해 조건별 True/False 표를 만든다. 데이터가 없는 조건은 표에 넣지 않는다."""
    c = pd.DataFrame(index=df.index)
    c['per_ok'] = (df['PER'] > 0) & (df['PER'] < p['MAX_PER'])
    c['peg_ok'] = (df['PEG'] > 0) & (df['PEG'] < p['MAX_PEG'])
    if p['MIN_GROWTH'] is not None:
        c['growth_ok'] = (df['growth'] >= p['MIN_GROWTH']) & (df['growth'] <= p['MAX_GROWTH'])
    if p['MAX_DEBT'] is not None:
        c['debt_ok'] = df['debt_ratio'] < p['MAX_DEBT']
    if p['MIN_ROE'] is not None:
        c['roe_ok'] = df['ROE'] > p['MIN_ROE']
    if p['MIN_ROA'] is not None:
        c['roa_ok'] = df['ROA'] > p['MIN_ROA']
    # 재고자산 계정이 없는 회사는 NOINV_AS_ZERO 가 True 일 때 재고 조건을 통과시킨다
    pass_noinv = df['no_inv'] if NOINV_AS_ZERO else pd.Series(False, index=df.index)
    if HAVE_INV and p['MAX_INV_SALES'] is not None:
        c['inv_sales_ok'] = (df['inv_sales'] < p['MAX_INV_SALES']) | pass_noinv
    if HAVE_INV and p['INV_LT_SALES']:
        c['inv_growth_ok'] = (df['inv_growth'] < df['sales_growth']) | pass_noinv
    if HAVE_DIV and p['MIN_DY'] is not None:
        c['dy_ok'] = df['DY'] > p['MIN_DY']
    return c.fillna(False).astype(bool)


def lynch_picks(df, mode, drop=()):
    """drop 에 넣은 조건은 빼고 나머지를 모두 통과한 종목."""
    if df is None or len(df) == 0:
        return []
    c = apply_conds(df, MODES[mode]).drop(columns=list(drop), errors='ignore')
    return list(c.index[c.all(axis=1)])

#%% [린치-6] 지금 시점 스크리닝
as_of_now = price_wide.index.max()
now_df = lynch_metrics(as_of_now)
print('기준일', as_of_now.date(), '/ 평가 대상', len(now_df), '개')

for m in MODES:
    c = apply_conds(now_df, MODES[m])
    print(f'\n===== 버전 {m} =====')
    print('조건별 통과 종목 수')
    print(c.sum().to_string())
    ok = now_df[c.all(axis=1)]
    print('모든 조건 통과:', len(ok))
    show = ['종목명', '시장구분', 'PER', 'growth', 'PEG', 'debt_ratio', 'ROE', 'ROA', 'inv_sales', 'DY']
    print(ok[show].sort_values('PEG').head(30).round(3).to_string())
if not HAVE_INV:
    print('\n[참고] 재고자산 계정이 없어 재고 조건은 적용하지 않았다.')
if not HAVE_DIV:
    print('[참고] 배당 데이터가 없어 배당수익률 조건은 적용하지 않았다.')

#%% [린치-7] 백테스트 함수
"""
매년 REBAL_MD 이후 첫 거래일에 스크리닝 -> 통과 종목 동일가중 매수 -> 다음 리밸런싱까지 보유.
통과 종목이 없으면 현금(수익률 0)으로 둔다. 비교 대상은 같은 방식의 전 종목 동일가중.

한계 (결과를 볼 때 감안할 것)
- 시가총액 : 현재 주식수로 근사한 값
- 이익      : 주당이익이 아니라 당기순이익
- 거래 제약 : 거래정지, 호가 단위, 슬리피지는 반영하지 않았다
- 조건      : 재고, 배당 조건은 데이터가 없으면 빠진다
"""
idx = price_wide.index
bench_picks = [c for c in price_wide.columns if c in universe]


def make_rebal_dates():
    out = []
    for y in range(idx.min().year, idx.max().year + 1):
        t = pd.Timestamp(f'{y}-{REBAL_MD}')
        pos = idx.searchsorted(t)
        if t > idx.min() and pos < len(idx) - 20:      # 마지막 20거래일 안이면 보유 기간이 너무 짧아 제외
            out.append(idx[pos])
    return out


def run_backtest(dates, get_picks, cost=TRADING_COST):
    """get_picks(날짜) -> 종목코드 리스트. 일별 가치(NAV)와 매수 종목 수 기록을 돌려준다."""
    value = 1.0
    navs = []
    log = []
    for k, d0 in enumerate(dates):
        d1 = dates[k + 1] if k + 1 < len(dates) else idx[-1]

        picks = [p for p in get_picks(d0)
                 if p in price_wide.columns and pd.notna(price_wide.at[d0, p])]

        if len(picks) == 0:
            seg_nav = pd.Series(value, index=price_wide.loc[d0:d1].index)      # 현금 보유
        else:
            value *= (1 - cost)                                                # 전량 교체 비용
            seg = price_wide.loc[d0:d1, picks].ffill()
            seg_nav = value * (seg / seg.iloc[0]).mean(axis=1)

        value = seg_nav.iloc[-1]
        navs.append(seg_nav if k == 0 else seg_nav.iloc[1:])                   # 경계일 중복 제거
        log.append((d0.date(), len(picks)))

    return pd.concat(navs), pd.DataFrame(log, columns=['리밸런싱일', '매수종목수'])

#%% [린치-8] 백테스트 실행 (버전 A / 버전 B)
dates_all = make_rebal_dates()
metrics_cache = {d: lynch_metrics(d) for d in dates_all}

# 성장률을 계산할 수 있는 날짜부터 시작한다 (재무가 PEG_YEARS + 1 개 연도 쌓인 뒤)
dates = [d for d in dates_all
         if len(metrics_cache[d]) > 0 and metrics_cache[d]['growth'].notna().any()]
if not dates:
    raise RuntimeError('성장률을 계산할 수 있는 리밸런싱 날짜가 없습니다. [린치-4] 의 주가·재무 범위를 확인하세요.')
print('리밸런싱', len(dates), '회:', dates[0].date(), '~', dates[-1].date())

results = {}
for m in MODES:
    nav, log = run_backtest(dates, lambda d, m=m: lynch_picks(metrics_cache[d], m))
    results[m] = dict(nav=nav, log=log)
nav_m, log_m = run_backtest(dates, lambda d: bench_picks)
results['bench'] = dict(nav=nav_m, log=log_m)

tbl = results['A']['log'].merge(results['B']['log'], on='리밸런싱일', suffixes=('_A', '_B'))
print(tbl.to_string(index=False))

#%% [린치-9] 성과 비교
def perf(nav):
    ret = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return pd.Series({
        '누적수익률': nav.iloc[-1] - 1,
        'CAGR': nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        'MDD': (nav / nav.cummax() - 1).min(),
        '변동성(연)': ret.std() * np.sqrt(252),
        'Sharpe(무위험 0)': ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan,
        '기간(년)': years,
    })


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

table = pd.concat([perf(results['A']['nav']), perf(results['B']['nav']), perf(results['bench']['nav'])],
                  axis=1, keys=['버전 A', '버전 B', '전종목 동일가중'])
print(table.round(4).to_string())

fig, ax = plt.subplots(figsize=(9, 5))
results['A']['nav'].plot(ax=ax, label='버전 A (알아온 조건)')
results['B']['nav'].plot(ax=ax, label='버전 B (문헌 기반)')
results['bench']['nav'].plot(ax=ax, label='전종목 동일가중')
ax.set_title('피터 린치 전략 백테스트')
ax.legend()
plt.tight_layout()
plt.show()

#%% [린치-10] 결과의 한계 자동 점검
def yearly(nav):
    y = nav.groupby(nav.index.year).last()
    r = y.pct_change()
    r.iloc[0] = y.iloc[0] / nav.iloc[0] - 1
    return r


years = (results['A']['nav'].index[-1] - results['A']['nav'].index[0]).days / 365.25
print(f'기간 {years:.1f}년 / 리밸런싱 {len(dates)}회')

for m in MODES:
    lg = results[m]['log']
    avg_pick = lg['매수종목수'].mean()
    zero_pick = int((lg['매수종목수'] == 0).sum())
    print(f'버전 {m}: 평균 매수종목 {avg_pick:.1f}개 / 현금 기간 {zero_pick}회')
    if avg_pick < 10:
        print('  - 평균 매수 종목이 10개 미만이라 개별 종목 한두 개가 결과를 좌우한다')

warn = []
if years < 5:
    warn.append('기간이 5년 미만이라 한두 해 시장 흐름이 결과를 좌우한다')
if len(dates) < 5:
    warn.append('리밸런싱이 5회 미만이라 통계적 결론을 내기 어렵다')
if not HAVE_INV:
    warn.append('재고 조건이 빠져 있다 (린치의 재고 기준은 두 버전 모두에 있다)')
if not HAVE_DIV:
    warn.append('배당수익률 조건이 빠져 있다 (버전 A 의 7조건 중 하나)')
warn.append('이익은 당기순이익(주당이익 아님), 시가총액은 현재 주식수 근사')
warn.append('거래비용은 가정치, 슬리피지와 거래정지는 미반영')
if INCLUDE_DELISTED:
    warn.append('상장폐지 종목을 넣었지만 재무를 받지 못한 폐지 종목은 빠져 있어 생존편향이 일부 남는다')
else:
    warn.append('생존편향: 상장폐지 종목이 빠져 있어 수익률이 높게 나올 수 있다')
for w in warn:
    print(' -', w)

yr = pd.concat([yearly(results['A']['nav']), yearly(results['B']['nav']), yearly(results['bench']['nav'])],
               axis=1, keys=['버전 A', '버전 B', '전종목 동일가중'])
yr['A-벤치'] = yr['버전 A'] - yr['전종목 동일가중']
yr['B-벤치'] = yr['버전 B'] - yr['전종목 동일가중']
print('\n연도별 수익률')
print(yr.round(4).to_string())
print('A 가 벤치마크보다 나은 해:', int((yr['A-벤치'] > 0).sum()), '/', len(yr),
      '| B:', int((yr['B-벤치'] > 0).sum()), '/', len(yr))

#%% [린치-11] 조건별 기여도: 조건을 하나씩 빼면 결과가 어떻게 바뀌는가
rows = []
for m in MODES:
    conds_now = list(apply_conds(next(d for d in metrics_cache.values() if len(d) > 0), MODES[m]).columns)
    for drop in [None] + conds_now:
        dr = () if drop is None else (drop,)
        nav, log = run_backtest(dates, lambda d, m=m, dr=dr: lynch_picks(metrics_cache[d], m, drop=dr))
        p = perf(nav)
        rows.append((m, '전체 조건' if drop is None else f'{drop} 제외',
                     p['CAGR'], p['MDD'], log['매수종목수'].mean()))

abl = pd.DataFrame(rows, columns=['버전', '구성', 'CAGR', 'MDD', '평균종목수'])
print(abl.round(4).to_string(index=False))
print('벤치마크 CAGR:', round(perf(results['bench']['nav'])['CAGR'], 4))
# 특정 조건을 빼면 CAGR 이 크게 오르내리면 그 조건이 결과를 좌우한다는 뜻이다.
# 평균 종목 수가 너무 적게 나오는 구성은 결과의 신뢰도가 낮다.

#%% [린치-12] 민감도: PEG 상한과 PER 상한을 바꿔 본다
base_modes = {k: dict(v) for k, v in MODES.items()}
rows = []
for m in MODES:
    for peg in [1.0, 1.5, 1.8, 2.5]:
        for per in [20, 30, 40]:
            MODES[m]['MAX_PEG'] = peg
            MODES[m]['MAX_PER'] = per
            nav, log = run_backtest(dates, lambda d, m=m: lynch_picks(metrics_cache[d], m))
            p = perf(nav)
            rows.append((m, peg, per, p['CAGR'], p['MDD'], log['매수종목수'].mean()))
    MODES[m].update(base_modes[m])

sens = pd.DataFrame(rows, columns=['버전', 'PEG 상한', 'PER 상한', 'CAGR', 'MDD', '평균종목수'])
print(sens.round(4).to_string(index=False))
# 기준값을 조금 바꿀 때 결과가 크게 흔들리면 그 전략은 신뢰하기 어렵다.

#%% [린치-13] 강건성 점검 준비: 공통 도구
"""
지금까지 결과(버전 A 7.52%, 버전 B 8.57%, 벤치마크 6.10%)가 비용, 소형주 쏠림, 일회성 이익, 시작 시점,
리밸런싱 날짜, 우연 중 어느 것 때문은 아닌지 확인하는 점검 모음이다.
전략이 좋다는 것을 증명하는 코드가 아니라 반증을 시도하는 코드이다. 통과 기준(숫자)은 이 코드의 가정이라 바꿔도 된다.

  [린치-14] 거래비용 3배/5배, 시가총액 하한, 영업이익 대비 순이익 필터
  [린치-15] 시작 시점을 바꿔 가며
  [린치-16] 리밸런싱 날짜를 4월 1일이 아닌 다른 날로
  [린치-17] 같은 개수를 무작위로 뽑은 포트폴리오와 비교 (전체 / 시가총액 하위 절반)
  [린치-18] 월간 초과수익 t값, 알파, 블록 부트스트랩 신뢰구간
  [린치-19] 위 결과를 모아 통과/탈락 표
  [린치-20] 보고서용 그림과 CSV 저장

[린치-8] 까지 실행되어 results, metrics_cache, dates 가 있어야 한다.
"""
def nav_cagr(nav):
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return nav.iloc[-1] ** (1 / years) - 1


def nav_mdd(nav):
    return (nav / nav.cummax() - 1).min()


def month_end_ret(nav):
    try:
        m = nav.resample('ME').last()
    except ValueError:                    # 오래된 pandas 는 'ME' 를 모른다
        m = nav.resample('M').last()
    return m.pct_change().dropna()


# 영업이익: 순이익이 영업이익보다 지나치게 크면 자산 매각익 같은 일회성 이익을 의심한다
op_raw = pd.read_sql("select 종목코드, 기준일, 값 from dart_fs where 계정 = 'op';", con=engine)
op_raw['기준일'] = pd.to_datetime(op_raw['기준일'])
op_raw = op_raw.sort_values(['종목코드', '기준일'])
op_cache = {}


def op_at(as_of):
    """as_of 에 공시가 끝난 가장 최근 연간 영업이익(억원)"""
    key = pd.Timestamp(as_of)
    if key not in op_cache:
        x = op_raw[op_raw['기준일'] + pd.Timedelta(days=DISCLOSURE_LAG) <= key]
        op_cache[key] = x.groupby('종목코드').tail(1).set_index('종목코드')['값']
    return op_cache[key]


NI_OP_MAX = 1.5      # 순이익 / 영업이익 이 이 값 이하이고 영업이익이 양수인 종목만 남긴다 (이 값은 임의의 가정)


def adv_picks(cache, d, mode, mcap_floor=0, op_filter=False, drop=()):
    """lynch_picks 결과에 시가총액 하한(억원)과 영업이익 필터를 더한다."""
    df = cache[d]
    picks = lynch_picks(df, mode, drop=drop)
    if len(picks) == 0:
        return picks
    sub = df.loc[picks]
    ok = pd.Series(True, index=sub.index)
    if mcap_floor:
        ok &= sub['mcap'] >= mcap_floor
    if op_filter:
        op = op_at(d).reindex(sub.index)
        ni_last = sub['mcap'] / sub['PER']                 # PER = 시가총액 / 순이익 이므로 순이익을 되살린다
        ok &= (op > 0) & (ni_last / op <= NI_OP_MAX)
    return list(sub.index[ok.fillna(False)])


def bench_floor(d, floor):
    """전 종목 동일가중 중 시가총액 하한을 넘는 종목 (하한을 둔 전략과 같은 조건의 비교 대상)"""
    mc = mcap_wide.loc[:d].iloc[-1] / 100000000
    return list(mc.index[(mc >= floor) & mc.index.isin(universe)])


print('준비 완료')

#%% [린치-14] 거래비용 / 시가총액 하한 / 영업이익 필터
cfgs = [('기본', dict()),
        ('비용 x3', dict(cost=TRADING_COST * 3)),
        ('비용 x5', dict(cost=TRADING_COST * 5)),
        ('시총 500억 이상', dict(mcap_floor=500)),
        ('시총 1000억 이상', dict(mcap_floor=1000)),
        ('시총 3000억 이상', dict(mcap_floor=3000)),
        ('영업이익 필터', dict(op_filter=True)),
        ('시총 1000억 + 영업이익', dict(mcap_floor=1000, op_filter=True))]

rows = []
for name, kw in cfgs:
    cost = kw.get('cost', TRADING_COST)
    floor = kw.get('mcap_floor', 0)
    sel = {k: v for k, v in kw.items() if k != 'cost'}

    if floor:
        bnav, _ = run_backtest(dates, lambda d, f=floor: bench_floor(d, f), cost=cost)
    else:
        bnav, _ = run_backtest(dates, lambda d: bench_picks, cost=cost)
    b_cagr = nav_cagr(bnav)

    for m in MODES:
        nav, log = run_backtest(dates, lambda d, m=m, s=sel: adv_picks(metrics_cache, d, m, **s), cost=cost)
        rows.append((name, m, nav_cagr(nav), b_cagr, nav_cagr(nav) - b_cagr, nav_mdd(nav),
                     log['매수종목수'].mean(), int((log['매수종목수'] == 0).sum())))

cfg_table = pd.DataFrame(rows, columns=['구성', '버전', 'CAGR', '벤치 CAGR', '초과', 'MDD', '평균종목수', '현금기간(회)'])
print(cfg_table.round(4).to_string(index=False))
# 시가총액 하한을 둔 구성의 벤치마크는 같은 하한을 적용한 전 종목 동일가중이다.
# 초과가 0 아래로 내려가는 구성이 있으면 그 조건에서는 벤치마크를 이기지 못한다는 뜻이다.

#%% [린치-15] 시작 시점을 바꿔 가며
rows = []
for k in range(0, len(dates) - 2):          # 최소 3번의 리밸런싱이 남도록
    dts = dates[k:]
    na, _ = run_backtest(dts, lambda d: adv_picks(metrics_cache, d, 'A'))
    nb, _ = run_backtest(dts, lambda d: adv_picks(metrics_cache, d, 'B'))
    nm, _ = run_backtest(dts, lambda d: bench_picks)
    rows.append((dts[0].date(), len(dts), nav_cagr(na), nav_cagr(nb), nav_cagr(nm)))

start_table = pd.DataFrame(rows, columns=['시작일', '리밸런싱 횟수', 'CAGR A', 'CAGR B', 'CAGR 벤치'])
start_table['A-벤치'] = start_table['CAGR A'] - start_table['CAGR 벤치']
start_table['B-벤치'] = start_table['CAGR B'] - start_table['CAGR 벤치']
print(start_table.round(4).to_string(index=False))
print('A 가 벤치마크보다 높은 시작 시점:', int((start_table['A-벤치'] > 0).sum()), '/', len(start_table),
      '| B:', int((start_table['B-벤치'] > 0).sum()), '/', len(start_table))
# 뒤쪽 시작일일수록 기간이 짧아 숫자가 크게 흔들린다. 앞쪽 시작일의 결과를 더 무겁게 본다.

#%% [린치-16] 리밸런싱 날짜를 바꿔 가며 (시간이 걸린다: 날짜마다 스크리닝을 다시 계산)
base_md = REBAL_MD
rows = []
for md in ['04-01', '06-01', '07-01', '10-01']:
    REBAL_MD = md
    dts_all = make_rebal_dates()
    cache = {d: lynch_metrics(d) for d in dts_all}
    dts = [d for d in dts_all if len(cache[d]) > 0 and cache[d]['growth'].notna().any()]
    if len(dts) < 3:
        print(md, ': 리밸런싱 날짜가 3개 미만이라 건너뜁니다.')
        continue
    na, la = run_backtest(dts, lambda d: adv_picks(cache, d, 'A'))
    nb, lb = run_backtest(dts, lambda d: adv_picks(cache, d, 'B'))
    nm, _ = run_backtest(dts, lambda d: bench_picks)
    rows.append((md, dts[0].date(), len(dts), nav_cagr(na), nav_cagr(nb), nav_cagr(nm),
                 la['매수종목수'].mean(), lb['매수종목수'].mean()))
REBAL_MD = base_md

reb_table = pd.DataFrame(rows, columns=['리밸런싱일', '시작일', '횟수', 'CAGR A', 'CAGR B', 'CAGR 벤치',
                                        'A 평균종목', 'B 평균종목'])
reb_table['A-벤치'] = reb_table['CAGR A'] - reb_table['CAGR 벤치']
reb_table['B-벤치'] = reb_table['CAGR B'] - reb_table['CAGR 벤치']
print(reb_table.round(4).to_string(index=False))
# 04-01 은 12월 결산 사업보고서가 나온 직후다. 나머지 날짜도 같은 공시 지연(90일)을 적용한다.

#%% [린치-17] 무작위 대조: 같은 개수를 무작위로 뽑으면 얼마나 나오는가
N_SIM = 1000
rng = np.random.default_rng(42)


def period_returns(dts):
    """리밸런싱 구간별 종목 수익률. 시작일에 가격이 없으면 NaN, 구간 중 폐지/정지는 마지막 가격으로 계산"""
    out = []
    for k, d0 in enumerate(dts):
        d1 = dts[k + 1] if k + 1 < len(dts) else idx[-1]
        seg = price_wide.loc[d0:d1].ffill()
        out.append(seg.iloc[-1] / seg.iloc[0] - 1)
    return out


def chain(period_means, cost=TRADING_COST):
    v = 1.0
    for r in period_means:
        if r is None or np.isnan(r):          # 종목이 없으면 현금
            continue
        v *= (1 - cost) * (1 + r)
    return v


pr = period_returns(dates)
years_total = (idx[-1] - dates[0]).days / 365.25

# 무작위로 뽑을 수 있는 종목: 그날 재무가 계산되고 구간 수익률이 있는 종목
elig_all, elig_small = [], []
for k, d in enumerate(dates):
    df = metrics_cache[d]
    r = pr[k].reindex(df.index)
    ok = r.notna() & df['PER'].notna()
    elig_all.append(r[ok])
    med = df.loc[ok, 'mcap'].median()
    elig_small.append(r[ok & (df['mcap'] <= med)])      # 시가총액 하위 절반

rows = []
for m in MODES:
    n_k, strat_means = [], []
    for k, d in enumerate(dates):
        p = [x for x in adv_picks(metrics_cache, d, m) if pd.notna(pr[k].get(x, np.nan))]
        n_k.append(len(p))
        strat_means.append(pr[k][p].mean() if p else np.nan)
    strat_cagr = chain(strat_means) ** (1 / years_total) - 1

    for label, elig in [('전체', elig_all), ('시총 하위 절반', elig_small)]:
        sims = np.empty(N_SIM)
        arrs = [e.values for e in elig]
        for s in range(N_SIM):
            means = [rng.choice(a, size=min(n, len(a)), replace=False).mean() if n > 0 else np.nan
                     for a, n in zip(arrs, n_k)]
            sims[s] = chain(means) ** (1 / years_total) - 1
        rows.append((m, label, strat_cagr, sims.mean(), np.percentile(sims, 5), np.percentile(sims, 50),
                     np.percentile(sims, 95), (sims < strat_cagr).mean(), (sims >= strat_cagr).mean()))

rand_table = pd.DataFrame(rows, columns=['버전', '무작위 모집단', '전략 CAGR', '무작위 평균', '5%', '50%', '95%',
                                         '전략 백분위', 'p값(무작위 >= 전략)'])
print(rand_table.round(4).to_string(index=False))
# 전략 CAGR 은 구간 수익률을 이어 붙인 값이라 [린치-9] 와 소수점 아래 조금 다를 수 있다.
# 백분위가 높을수록 같은 개수를 무작위로 뽑은 것보다 나았다는 뜻이다. '시총 하위 절반' 은 소형주 효과를 걷어 내려는 비교다.

#%% [린치-18] 통계 검정: 월간 초과수익의 t값, 알파, 블록 부트스트랩
rng_boot = np.random.default_rng(7)


def excess_stats(nav_s, nav_b, label, block=3, n_boot=5000):
    ms, mb = month_end_ret(nav_s), month_end_ret(nav_b)
    j = pd.concat([ms, mb], axis=1, keys=['s', 'b']).dropna()
    n = len(j)
    d = (j['s'] - j['b']).values
    t_ex = d.mean() / (d.std(ddof=1) / np.sqrt(n))

    x, y = j['b'].values, j['s'].values            # 벤치마크 수익률에 대한 회귀: 알파와 베타
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    beta = ((x - xm) * (y - ym)).sum() / sxx
    alpha = ym - beta * xm
    resid = y - (alpha + beta * x)
    s_e = np.sqrt((resid ** 2).sum() / (n - 2))
    t_alpha = alpha / (s_e * np.sqrt(1 / n + xm ** 2 / sxx))

    nb = int(np.ceil(n / block))                    # 월간 초과수익을 3개월 블록으로 다시 뽑는다
    boot = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng_boot.integers(0, n - block + 1, nb)
        samp = np.concatenate([d[s:s + block] for s in starts])[:n]
        boot[i] = samp.mean() * 12
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(버전=label, 월수=n, 연환산초과=d.mean() * 12, t_초과=t_ex, 베타=beta,
                연환산알파=alpha * 12, t_알파=t_alpha, 부트하한=lo, 부트상한=hi)


stat_table = pd.DataFrame([excess_stats(results[m]['nav'], results['bench']['nav'], m) for m in MODES])
print(stat_table.round(4).to_string(index=False))
# t 값이 2 보다 작으면 월간 수익률 기준으로는 초과수익이 0 이라는 가설을 기각하기 어렵다.
# 부트스트랩 구간이 0 을 포함하면 마찬가지다. 월 100여 개 표본이라 검정력이 크지 않다.
# 두 버전 + 민감도 표를 여러 번 보았으므로, 가장 좋아 보이는 값만 골라 쓰면 과대평가가 된다 (다중 비교).
# 보고서에는 미리 정한 A(알아온 조건)와 B(문헌 기준)의 기본 결과를 주 결과로 쓰고 나머지는 점검으로 싣는 편이 안전하다.

#%% [린치-19] 종합: 점검별 통과 / 탈락
# 아래 통과선은 이 코드의 가정이다.
checks = []


def add(name, m, passed, detail):
    checks.append((name, m, '통과' if passed else '탈락', detail))


for m in MODES:
    c = cfg_table[cfg_table['버전'] == m].set_index('구성')
    add('비용 3배에서 벤치마크 초과', m, c.loc['비용 x3', '초과'] > 0, f"초과 {c.loc['비용 x3', '초과']:+.4f}")
    add('시총 1000억 이상에서 초과', m, c.loc['시총 1000억 이상', '초과'] > 0, f"초과 {c.loc['시총 1000억 이상', '초과']:+.4f}")
    add('영업이익 필터 후 초과', m, c.loc['영업이익 필터', '초과'] > 0, f"초과 {c.loc['영업이익 필터', '초과']:+.4f}")

    col = f'{m}-벤치'
    share = (start_table[col] > 0).mean()
    add('시작 시점 70% 이상에서 초과', m, share >= 0.70, f'{share:.0%}')

    share = (reb_table[col] > 0).mean() if len(reb_table) else np.nan
    add('리밸런싱 날짜 75% 이상에서 초과', m, share >= 0.75, f'{share:.0%}')

    r = rand_table[rand_table['버전'] == m].set_index('무작위 모집단')
    add('무작위 대비 백분위 90% 이상(전체)', m, r.loc['전체', '전략 백분위'] >= 0.90, f"{r.loc['전체', '전략 백분위']:.1%}")
    add('무작위 대비 백분위 90% 이상(시총 하위)', m, r.loc['시총 하위 절반', '전략 백분위'] >= 0.90,
        f"{r.loc['시총 하위 절반', '전략 백분위']:.1%}")

    s = stat_table.set_index('버전').loc[m]
    add('알파 t값 2 이상', m, s['t_알파'] >= 2.0, f"t {s['t_알파']:.2f}")
    add('초과수익 부트스트랩 하한 > 0', m, s['부트하한'] > 0, f"하한 {s['부트하한']:+.4f}")

chk = pd.DataFrame(checks, columns=['점검', '버전', '결과', '값'])
print(chk.to_string(index=False))
print()
for m in MODES:
    sub = chk[chk['버전'] == m]
    n_ok = int((sub['결과'] == '통과').sum())
    print(f'버전 {m}: {n_ok} / {len(sub)} 통과')
print('통과 개수가 적으면 "이 표본에서는 그렇게 보인다" 정도로만 해석한다.')
print('통과 개수가 많아도 표본 기간(약 9.5년)과 생존편향, 순이익 근사 같은 [린치-10] 의 한계는 그대로 남는다.')

#%% [린치-20] 보고서용 그림과 CSV 저장
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

for key, label in [('A', '버전 A (알아온 조건)'), ('B', '버전 B (문헌 기반)'), ('bench', '전종목 동일가중')]:
    results[key]['nav'].plot(ax=axes[0], label=label)
axes[0].set_yscale('log')
axes[0].set_title('누적 성과 (로그 눈금)')
axes[0].legend()

for key, label in [('A', '버전 A'), ('B', '버전 B'), ('bench', '전종목 동일가중')]:
    nav = results[key]['nav']
    (nav / nav.cummax() - 1).plot(ax=axes[1], label=label)
axes[1].set_title('낙폭')
axes[1].legend()

yr = pd.concat([yearly(results['A']['nav']) - yearly(results['bench']['nav']),
                yearly(results['B']['nav']) - yearly(results['bench']['nav'])], axis=1, keys=['A-벤치', 'B-벤치'])
yr.plot(kind='bar', ax=axes[2])
axes[2].axhline(0, color='black', linewidth=0.8)
axes[2].set_title('연도별 초과수익 (전략 - 벤치마크)')
plt.tight_layout()
plt.show()

# 표 저장 (엑셀에서 열 수 있게 utf-8-sig). 저장 위치는 Spyder 의 현재 작업 폴더
for name, tbl in [('cfg_table', cfg_table), ('start_table', start_table), ('reb_table', reb_table),
                  ('rand_table', rand_table), ('stat_table', stat_table), ('checks', chk)]:
    tbl.to_csv(f'lynch_{name}.csv', index=False, encoding='utf-8-sig')
print('CSV 저장 완료:', os.getcwd())

#%% [린치-21] 종목별 기여도 / 쏠림 점검
"""
수익이 소수 종목이나 특정 업종에서 나온 것은 아닌지 확인한다. [린치-13], [린치-17] 이 먼저 실행되어 있어야 한다.

  1) 종목별 누적 기여도 상위 10개와 상위 1/3/5개가 전체 기여에서 차지하는 비중
  2) 기여 상위 N개 종목을 사후적으로 빼고 다시 계산한 CAGR (N = 1, 3, 5, 10)
     주의: 사후에 가장 잘 번 종목을 빼는 것이라 어떤 전략이든 성과가 내려간다. 그래서 '빼고도 빼지 않은 벤치마크보다 높은가' 로 본다.
  3) 리밸런싱 구간별로 가장 많이 번 종목 1개가 그 구간 기여에서 차지한 비중
  4) 업종(WICS) 쏠림. kor_sector 테이블이 있을 때만 계산한다.
     업종은 가장 최근 기준일의 분류를 모든 연도에 그대로 쓰고, 상장폐지 종목은 '미분류' 로 나온다.

기여도 = 구간 시작 금액 x 종목 구간수익률 / 편입 종목 수 (동일가중, 비용 반영 후 금액). 합계는 최종 NAV 증가분에서 거래비용을 뺀 값과 같다.
"""
TOP_DROP = [0, 1, 3, 5, 10]
name_series = name_map['종목명'].groupby(level=0).first()

# 벤치마크: 같은 구간수익률을 이어 붙인 값 ([린치-17] 의 chain 과 같은 방식)
bench_means = [pr[k].reindex(bench_picks).mean() for k in range(len(dates))]
bench_cagr_chain = chain(bench_means) ** (1 / years_total) - 1
print(f'벤치마크 CAGR(구간 수익률 이어 붙임): {bench_cagr_chain:.4f}')

sector_map = None
if 'kor_sector' in inspect(engine).get_table_names():
    sec = pd.read_sql("""
    select CMP_CD, SEC_NM_KOR from kor_sector
    where 기준일 = (select max(기준일) from kor_sector);
    """, con=engine)
    sector_map = sec.drop_duplicates('CMP_CD').set_index('CMP_CD')['SEC_NM_KOR']
else:
    print('kor_sector 테이블이 없어 업종 쏠림은 건너뜁니다.')


def valid_picks(mode, **kw):
    """구간별로 실제 매수 가능했던 종목(시작일 가격과 구간 수익률이 있는 종목)"""
    out = []
    for k, d in enumerate(dates):
        out.append([x for x in adv_picks(metrics_cache, d, mode, **kw) if pd.notna(pr[k].get(x, np.nan))])
    return out


def contrib_table(picks_k):
    v = 1.0
    rows = []
    for k, p in enumerate(picks_k):
        if not p:
            continue
        v_start = v * (1 - TRADING_COST)
        r = pr[k][p]
        for code, ri in r.items():
            rows.append((dates[k], code, ri, v_start * ri / len(p)))
        v = v_start * (1 + r.mean())
    ct = pd.DataFrame(rows, columns=['리밸런싱일', '종목코드', '구간수익률', '기여'])
    ct['종목명'] = ct['종목코드'].map(name_series)
    return ct


universe_share = None
if sector_map is not None:
    universe_share = pd.Series(bench_picks).map(sector_map).fillna('미분류').value_counts(normalize=True)

conc_rows = []
for label, mode, kw in [('버전 A', 'A', {}), ('버전 B', 'B', {}), ('버전 B + 시총 1000억 이상', 'B', dict(mcap_floor=1000))]:
    print('\n' + '=' * 70)
    print(label)
    picks_k = valid_picks(mode, **kw)
    ct = contrib_table(picks_k)
    tot = ct['기여'].sum()

    by = (ct.groupby(['종목코드', '종목명'])
            .agg(편입횟수=('기여', 'size'), 누적기여=('기여', 'sum'))
            .reset_index().sort_values('누적기여', ascending=False))
    by['비중'] = by['누적기여'] / tot
    print(f'편입 종목 수 {len(by)}개 / 편입 건수 {len(ct)}건 / 기여 합계 {tot:.3f} (NAV 포인트)')
    print('\n[누적 기여 상위 10]')
    print(by.head(10).round(4).to_string(index=False))
    top1, top3, top5 = (by['누적기여'].head(n).sum() / tot for n in (1, 3, 5))
    print(f'\n상위 1개 {top1:.1%} / 상위 3개 {top3:.1%} / 상위 5개 {top5:.1%} (기여 합계 대비)')

    print('\n[기여 상위 N개를 사후에 뺀 CAGR]')
    rank = list(by['종목코드'])
    drop_rows = []
    for n_drop in TOP_DROP:
        drop = set(rank[:n_drop])
        means = []
        for k, p in enumerate(picks_k):
            q = [x for x in p if x not in drop]
            means.append(pr[k][q].mean() if q else np.nan)
        cg = chain(means) ** (1 / years_total) - 1
        drop_rows.append((n_drop, cg, cg - bench_cagr_chain))
        conc_rows.append((label, n_drop, cg, cg - bench_cagr_chain))
    print(pd.DataFrame(drop_rows, columns=['제외 종목 수', 'CAGR', '벤치마크 대비']).round(4).to_string(index=False))

    prow = []
    for d, g in ct.groupby('리밸런싱일'):
        tot_p = g['기여'].sum()
        top = g.loc[g['기여'].idxmax()]
        prow.append((d.date(), len(g), g['구간수익률'].mean(), top['종목명'], top['구간수익률'],
                     top['기여'] / tot_p if tot_p > 0 else np.nan))
    print('\n[구간별 최대 기여 종목] (비중은 그 구간 기여 합계가 양수일 때만 계산)')
    print(pd.DataFrame(prow, columns=['리밸런싱일', '종목수', '구간 평균수익률', '최대 기여 종목', '그 종목 수익률',
                                      '구간 기여 비중']).round(4).to_string(index=False))

    if sector_map is not None:
        ct['섹터'] = ct['종목코드'].map(sector_map).fillna('미분류')
        sec_tab = ct.groupby('섹터').agg(편입건수=('기여', 'size'), 기여=('기여', 'sum'))
        sec_tab['편입비중'] = sec_tab['편입건수'] / sec_tab['편입건수'].sum()
        sec_tab['기여비중'] = sec_tab['기여'] / tot
        sec_tab['유니버스 비중'] = universe_share.reindex(sec_tab.index).fillna(0)
        print('\n[업종별 편입 / 기여 비중] (유니버스 비중은 전 종목 중 그 업종 종목 수 비중)')
        print(sec_tab.sort_values('기여', ascending=False).round(4).to_string())

conc_table = pd.DataFrame(conc_rows, columns=['구성', '제외 종목 수', 'CAGR', '벤치마크 대비'])
conc_table.to_csv('lynch_conc_table.csv', index=False, encoding='utf-8-sig')
# 해석 기준
# - 상위 5개 종목이 기여 합계의 대부분(예: 50% 이상)이면 소수 종목 의존이다.
# - 상위 3~5개를 빼고도 벤치마크 CAGR 보다 높으면 쏠림에 강한 편이다. 낮아지는 것 자체는 정상이다.
# - 한 구간의 최대 기여 비중이 크면(예: 구간 기여의 절반 이상) 그 해 성과는 한 종목이 좌우한 것이다.
# - 한 업종이 편입과 기여 비중에서 모두 유니버스 비중을 크게 웃돌면 업종 베팅이 섞여 있다.

#%% [린치-22] 버전 B 운용 전략: 설정과 엔진
"""
버전 B 를 통과한 종목을 실제 매매 규칙으로 바꾸는 부분이다. [린치-8] 까지 실행되어 있어야 하고 [린치-13] 도 먼저 실행한다.
(price_wide, metrics_cache, dates, results, nav_cagr, nav_mdd 를 쓴다.)

규칙 요약
  신호  매월 첫 거래일에 버전 B 스크리닝을 한다. 통과한 종목은 워치리스트에 올라가고 WATCH_DAYS 동안 유효하다.
        그 사이 또 통과하면 기간이 연장된다. 신호가보다 STOP_LOSS 이상 빠진 종목은 워치리스트에서 빠진다.
  진입  bulk      신호 다음 거래일에 바로 매수 (기존 연 1회 방식과 가장 가까운 기준선)
        pullback  워치 기간 중 신호가보다 PULLBACK 이상 눌렸고 손절선 위에서 5일선을 회복하면 매수
        ha        워치 기간 중 하이킨아시 양봉으로 바뀌면 매수 (손절선 위일 때만)
        random    워치 기간 중 무작위 하루에 매수 (진입 타이밍 규칙이 의미가 있는지 보는 대조군)
  청산  진입가 대비 -STOP_LOSS 손절 / 보유 중 고점 대비 -TRAIL_STOP 이탈 / MAX_HOLD 거래일 도달, 먼저 걸리는 것
  비중  equal 동일 / invvol 변동성 역수 / peg PEG 가 낮을수록 크게
        한 종목 기본 비중은 1/MAX_POS 이고 변동성, PEG 로 조정해도 0.5배 ~ 2배 안에 둔다.
체결  판단은 당일 종가까지의 정보로 하고 체결은 다음 거래일 종가로 한다.
      비용 TRADING_COST 는 매수와 매도에 절반씩 붙인다. 현금에는 이자가 없다고 본다.
      보유 중 가격이 사라지면(상장폐지, 장기 거래정지) 마지막 가격에 청산한다. 실제로는 이보다 나쁠 수 있다.
"""
HOLD_OPTS = {'단기60': 60, '중기120': 120, '장기250': 250}      # 최대 보유 거래일
ENTRY_OPTS = ['bulk', 'pullback', 'ha']
WEIGHT_OPTS = ['equal', 'invvol', 'peg']

MAX_POS = 20          # 동시 보유 종목 수 상한
WATCH_DAYS = 63       # 신호 유효 기간 (거래일, 약 3개월)
STOP_LOSS = 0.10      # 손절폭 (진입가 기준, 워치리스트 제외는 신호가 기준)
TRAIL_STOP = 0.20     # 고점 대비 하락 청산폭. None 이면 사용 안 함
PULLBACK = 0.03       # pullback 진입: 신호가 대비 최소 눌림
CONFIRM_MA = 5        # pullback 진입: 이 일수 이동평균 위로 회복
VOL_WIN = 60          # 변동성 계산 일수
MCAP_FLOOR = 0        # 시가총액 하한(억원). 0 이면 버전 B 그대로
COST_SIDE = TRADING_COST / 2
RANDOM_SEEDS = range(5)

# 하이킨아시 전환 신호용 시가/고가/저가/종가 (kor_price). 상장폐지 종목 테이블은 컬럼이 같다고 보고 읽어 본다.
def _load_ohlc(table):
    x = pd.read_sql(f"select 날짜, 종목코드, 시가, 고가, 저가, 종가 from {table};", con=engine)
    x['날짜'] = pd.to_datetime(x['날짜'])
    return x


ohlc = [_load_ohlc('kor_price')]
try:
    ohlc.append(_load_ohlc('kor_price_delisted'))
except Exception as e:
    print('kor_price_delisted 에서 시가/고가/저가를 읽지 못해 상장폐지 종목은 하이킨아시 신호에서 빠집니다:', repr(e)[:120])
ohlc = pd.concat(ohlc).drop_duplicates(['날짜', '종목코드'])


def _pv(col):
    return ohlc.pivot(index='날짜', columns='종목코드', values=col).reindex(index=price_wide.index, columns=price_wide.columns)


def ha_turn_matrix():
    """하이킨아시 캔들이 음봉 -> 양봉으로 바뀐 날 True.
    HA종가 = (시+고+저+종)/4, HA시가 = 전날 (HA시가 + HA종가)/2, 양봉 = HA종가 > HA시가"""
    o, h, l, c = (_pv(k).values for k in ['시가', '고가', '저가', '종가'])
    ha_c = (o + h + l + c) / 4
    ha_o = np.full_like(ha_c, np.nan)
    ha_o[0] = (o[0] + c[0]) / 2
    for t in range(1, len(ha_c)):
        prev_ok = ~(np.isnan(ha_o[t - 1]) | np.isnan(ha_c[t - 1]))
        ha_o[t] = np.where(prev_ok, (ha_o[t - 1] + ha_c[t - 1]) / 2, (o[t] + c[t]) / 2)   # 값이 끊기면 다시 시작
    bull = ha_c > ha_o
    turn = bull & ~np.roll(bull, 1, axis=0)
    turn[0] = False
    return turn


HAT = ha_turn_matrix()
del ohlc

idx = price_wide.index
n = len(idx)
cols = list(price_wide.columns)
cidx = {c: i for i, c in enumerate(cols)}
Pdf = price_wide.ffill()
P = price_wide.values                  # 원래 값 (결측 = 그날 거래 없음)
Pf = Pdf.values                        # 앞 값으로 채운 값 (평가용)
MA = Pdf.rolling(CONFIRM_MA).mean().values
VOLM = price_wide.pct_change(fill_method=None).rolling(VOL_WIN, min_periods=VOL_WIN // 2).std().values

i0 = idx.get_loc(dates[0])
_m = np.asarray(idx.year) * 12 + np.asarray(idx.month)
is_sig = np.zeros(n, dtype=bool)
is_sig[1:] = _m[1:] != _m[:-1]         # 달이 바뀐 첫 거래일
is_sig[i0] = True

sig_cache = {}


def get_signals(d):
    """날짜 d 의 버전 B 통과 종목: [(열 번호, PEG, 시가총액), ...]"""
    if d not in sig_cache:
        df = metrics_cache[d] if d in metrics_cache else lynch_metrics(d)
        picks = lynch_picks(df, 'B')
        if MCAP_FLOOR:
            picks = [x for x in picks if df.at[x, 'mcap'] >= MCAP_FLOOR]
        sig_cache[d] = [(cidx[x], df.at[x, 'PEG'], df.at[x, 'mcap']) for x in picks if x in cidx]
    return sig_cache[d]


def add_signals(watch, t, sigs, rng=None):
    for j, peg, mc in sigs:
        if np.isnan(P[t, j]):
            continue
        w = watch.get(j)
        if w is None:
            watch[j] = dict(sig_t=t, sig_px=P[t, j], last_t=t, peg=peg,
                            rnd_t=t + (int(rng.integers(0, WATCH_DAYS)) if rng is not None else 0))
        else:
            w['last_t'] = t
            w['peg'] = peg


def prune_watch(watch, t):
    for j in list(watch):
        w = watch[j]
        if (t - w['last_t'] > WATCH_DAYS) or np.isnan(P[t, j]) or (Pf[t, j] <= w['sig_px'] * (1 - STOP_LOSS)):
            del watch[j]


def entry_ok(mode, w, j, t):
    px = P[t, j]
    if np.isnan(px) or px <= w['sig_px'] * (1 - STOP_LOSS):
        return False
    if mode == 'bulk':
        return w['last_t'] == t
    if mode == 'pullback':
        return bool(px <= w['sig_px'] * (1 - PULLBACK) and px > MA[t, j])
    if mode == 'ha':
        return bool(HAT[t, j])
    if mode == 'random':
        return t >= w['rnd_t']
    raise ValueError(mode)


def weight_score(kind, j, t, w, ref_vol):
    if kind == 'equal':
        return 1.0
    if kind == 'invvol':
        v = VOLM[t, j]
        return 1.0 if (np.isnan(v) or v <= 0) else float(np.clip(ref_vol / v, 0.5, 2.0))
    if kind == 'peg':
        peg = w['peg']
        return 1.0 if (not peg > 0) else float(np.clip(0.5 / peg, 0.5, 2.0))
    raise ValueError(kind)


def _sell(pos, trades, j, t, why):
    p = pos.pop(j)
    px = Pf[t, j]
    trades.append((cols[j], idx[p['t0']], idx[t], p['px0'], px, px / p['px0'] - 1, why))
    return p['units'] * px * (1 - COST_SIDE)


def run_strategy(hold, entry, weight, seed=0):
    max_hold = HOLD_OPTS[hold] if isinstance(hold, str) else int(hold)
    rng = np.random.default_rng(seed) if entry == 'random' else None
    watch, pos, block, trades = {}, {}, {}, []
    cash = 1.0
    sell_orders, buy_orders = [], []
    eq = np.full(n - i0, np.nan)
    inv = np.zeros(n - i0)
    for t in range(i0, n):
        # 1) 전날 낸 매도 주문 체결, 가격이 사라진 종목은 마지막 가격에 정리
        for j, why in sell_orders:
            if j in pos:
                cash += _sell(pos, trades, j, t, why)
        for j in list(pos):
            if np.isnan(P[t, j]):
                cash += _sell(pos, trades, j, t, '가격 소멸')
                block[j] = t + WATCH_DAYS
        # 2) 전날 낸 매수 주문 체결
        eq_pre = cash + sum(p['units'] * Pf[t, j] for j, p in pos.items())
        for j, sc in buy_orders:
            if j in pos or len(pos) >= MAX_POS or np.isnan(P[t, j]):
                continue
            amt = min(sc * eq_pre / MAX_POS, cash)
            if amt < 1e-6:
                continue
            cash -= amt
            pos[j] = dict(units=amt * (1 - COST_SIDE) / P[t, j], px0=P[t, j], t0=t, peak=P[t, j])
        # 3) 종가 기준 평가
        val = sum(p['units'] * Pf[t, j] for j, p in pos.items())
        eq[t - i0] = cash + val
        inv[t - i0] = val / eq[t - i0]
        for j, p in pos.items():
            p['peak'] = max(p['peak'], Pf[t, j])
        # 4) 워치리스트 갱신
        if is_sig[t]:
            add_signals(watch, t, get_signals(idx[t]), rng)
        prune_watch(watch, t)
        if t == n - 1:
            break
        # 5) 내일 체결할 주문 만들기
        sell_orders = []
        for j, p in pos.items():
            px = Pf[t, j]
            if px <= p['px0'] * (1 - STOP_LOSS):
                sell_orders.append((j, '손절'))
            elif TRAIL_STOP and px <= p['peak'] * (1 - TRAIL_STOP):
                sell_orders.append((j, '고점대비하락'))
            elif t - p['t0'] >= max_hold:
                sell_orders.append((j, '보유기간'))
        sold = {j for j, _ in sell_orders}
        for j in sold:
            block[j] = t + 1 + WATCH_DAYS          # 청산한 종목은 한동안 다시 사지 않는다
        slots = MAX_POS - len(pos) + len(sold)
        buy_orders = []
        if slots > 0:
            cands = []
            for j, w in watch.items():
                if j in pos or j in sold or block.get(j, -1) > t:
                    continue
                if entry_ok(entry, w, j, t):
                    cands.append((w['peg'] if (w['peg'] == w['peg'] and w['peg'] > 0) else 9.9, j))
            cands.sort()                              # PEG 낮은 종목부터 빈 자리를 채운다
            if cands:
                ref = np.nanmedian(VOLM[t]) if weight == 'invvol' else None
                for _, j in cands[:slots]:
                    buy_orders.append((j, weight_score(weight, j, t, watch[j], ref)))
    nav = pd.Series(eq, index=idx[i0:])
    tr = pd.DataFrame(trades, columns=['종목코드', '진입일', '청산일', '진입가', '청산가', '수익률', '사유'])
    return nav, pd.Series(inv, index=nav.index), tr


def seg_cagr(nav):
    return nav_cagr(nav / nav.iloc[0])


def strat_stats(nav, inv, tr):
    vol = nav.pct_change().std() * np.sqrt(252)
    cg = nav_cagr(nav)
    mid = nav.index[len(nav) // 2]
    return dict(CAGR=cg, MDD=nav_mdd(nav), 변동성=vol, 수익위험=cg / vol,
                전반부=seg_cagr(nav.loc[:mid]), 후반부=seg_cagr(nav.loc[mid:]),
                투자비중=inv.mean() if inv is not None else np.nan,
                거래수=len(tr) if tr is not None else np.nan,
                승률=(tr['수익률'] > 0).mean() if (tr is not None and len(tr)) else np.nan,
                평균수익률=tr['수익률'].mean() if (tr is not None and len(tr)) else np.nan)


def ref_nav(nav):
    x = nav.loc[nav.index >= idx[i0]]
    return x / x.iloc[0]


print(f'엔진 준비 완료. 백테스트 구간 {idx[i0].date()} ~ {idx[-1].date()} ({n - i0} 거래일), 월간 신호일 {int(is_sig[i0:].sum())}개')

#%% [린치-23] 보유기간 x 진입 방식 x 비중 방식 비교 (27가지 + 대조군)
"""
세 가지를 모두 돌려 비교한다. 오래 걸릴 수 있다 (처음에는 월별 스크리닝을 새로 계산하고, 이후에는 캐시를 쓴다).
최고 CAGR 하나를 고르는 용도가 아니다. 27개 중 가장 좋은 것은 우연히 좋게 나왔을 가능성이 크다.
확인할 것: (1) 전반부와 후반부 모두에서 기준선을 이기는가 (2) 해당 요소를 바꿨을 때 평균적으로 일관된 방향인가
          (3) 같은 보유기간의 무작위 진입(대조군)보다 나은가
"""
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

rows, navs = [], {}
for h in HOLD_OPTS:
    for e in ENTRY_OPTS:
        for w in WEIGHT_OPTS:
            nav, inv, tr = run_strategy(h, e, w)
            navs[(h, e, w)] = nav
            rows.append(dict(보유=h, 진입=e, 비중=w, **strat_stats(nav, inv, tr)))
            print(f'{h} / {e} / {w}: CAGR {rows[-1]["CAGR"]:.4f}, MDD {rows[-1]["MDD"]:.3f}, 투자비중 {rows[-1]["투자비중"]:.2f}')
grid = pd.DataFrame(rows)

# 대조군: 진입 날짜만 무작위 (비중 동일), 시드 5개 평균
ctrl_rows = []
for h in HOLD_OPTS:
    sts = []
    for s in RANDOM_SEEDS:
        nav, inv, tr = run_strategy(h, 'random', 'equal', seed=s)
        sts.append(strat_stats(nav, inv, tr))
    m = pd.DataFrame(sts).mean()
    ctrl_rows.append(dict(보유=h, 진입='random(5회평균)', 비중='equal', **m.to_dict()))
ctrl = pd.DataFrame(ctrl_rows)

# 기준선: 버전 B 연 1회 리밸런싱, 전 종목 동일가중
refs = []
for name, key in [('버전 B 연1회 리밸런싱', 'B'), ('벤치마크(전 종목 동일가중)', 'bench')]:
    nv = ref_nav(results[key]['nav'])
    refs.append(dict(보유='-', 진입=name, 비중='-', **strat_stats(nv, None, None)))
refs = pd.DataFrame(refs)

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 30)
show = ['보유', '진입', '비중', 'CAGR', 'MDD', '변동성', '수익위험', '전반부', '후반부', '투자비중', '거래수', '승률', '평균수익률']
print('\n[기준선]')
print(refs[show].round(4).to_string(index=False))
print('\n[무작위 진입 대조군]')
print(ctrl[show].round(4).to_string(index=False))
print('\n[27가지 전체: 수익위험 순]')
print(grid.sort_values('수익위험', ascending=False)[show].round(4).to_string(index=False))

print('\n[요소별 평균] 다른 두 요소는 평균에 섞여 있다')
for k in ['보유', '진입', '비중']:
    print(f'\n-- {k}')
    print(grid.groupby(k)[['CAGR', 'MDD', '수익위험', '전반부', '후반부', '투자비중']].mean().round(4).to_string())

# 같은 보유기간, 비중 동일일 때 진입 방식 비교 (무작위 대조군 포함)
print('\n[진입 방식 비교: 비중 equal 고정]')
cmp_ = pd.concat([grid[grid['비중'] == 'equal'], ctrl])[['보유', '진입', 'CAGR', 'MDD', '수익위험', '전반부', '후반부', '투자비중']]
print(cmp_.sort_values(['보유', '진입']).round(4).to_string(index=False))

grid.to_csv('lynch_strategy_grid.csv', index=False, encoding='utf-8-sig')

# 수익위험 1등 설정 차트 (참고용, 1등이 최선이라는 뜻은 아니다)
best = grid.sort_values('수익위험', ascending=False).iloc[0]
bk = (best['보유'], best['진입'], best['비중'])
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(navs[bk], label=f'{bk[0]} / {bk[1]} / {bk[2]}')
ax.plot(ref_nav(results['B']['nav']), label='버전 B 연1회')
ax.plot(ref_nav(results['bench']['nav']), label='벤치마크', color='gray')
ax.set_title('운용 전략 vs 기준선 (순자산, 시작=1)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%% [린치-24] 일일 리포트: 오늘 살 종목 / 기다리는 종목 / 보유 종목 점검 + 엑셀 저장
"""
매일 실행해서 오늘 기준의 매수 후보, 대기 목록, 보유 종목의 청산 신호를 보고 엑셀로 저장한다.
[린치-22] 까지 실행되어 있어야 한다. 데이터를 최신으로 쓰려면 주가 수집(kor_price)을 먼저 갱신하고 [린치-4] 부터 다시 실행한다.

CHOSEN : 사용할 설정. [린치-23] 에서 가장 설득력 있던 장기250 / bulk / equal 로 두었다.
my_holdings.csv : 내가 가진 종목을 직접 적는 파일 (열: 종목코드, 매수일, 매수가). 종목코드는 6자리 문자열로 읽는다.
CAPITAL : 투자 원금(원). 0 이면 금액/수량은 계산하지 않고 비중만 보여 준다.
OUT_DIR : 엑셀 저장 폴더. '.' 이면 Spyder 작업 폴더.
LIVE_ADD_TODAY : True 면 오늘 날짜의 스크리닝 결과도 신호로 넣는다. 백테스트는 월 1회 신호만 썼으므로 True 는 검증하지 않은 방식이다.
bulk 에서 '오늘 매수 후보' 는 오늘 B 를 통과한 종목 중 내가 안 가진 것을 PEG 낮은 순으로 빈 자리만큼 보여 준 것이다.
이미 산 종목은 my_holdings.csv 에 적어야 다시 후보로 나오지 않는다.
워치리스트는 최근 4 x WATCH_DAYS 거래일 동안의 월간 신호로 다시 만든다 (그 이전에 시작된 신호가 이어지고 있으면 신호가가 조금 다를 수 있다).
"""
import os

CHOSEN = dict(hold='장기250', entry='bulk', weight='equal')
HOLDINGS_FILE = 'my_holdings.csv'
CAPITAL = 0
OUT_DIR = '.'
LIVE_ADD_TODAY = True

name_s = name_map['종목명'].groupby(level=0).first()


def build_watch(t):
    watch = {}
    for tt in range(max(i0, t - 4 * WATCH_DAYS), t + 1):
        if is_sig[tt]:
            add_signals(watch, tt, get_signals(idx[tt]))
        prune_watch(watch, tt)
    if LIVE_ADD_TODAY and not is_sig[t]:
        df = lynch_metrics(idx[t])
        picks = lynch_picks(df, 'B')
        if MCAP_FLOOR:
            picks = [x for x in picks if df.at[x, 'mcap'] >= MCAP_FLOOR]
        add_signals(watch, t, [(cidx[x], df.at[x, 'PEG'], df.at[x, 'mcap']) for x in picks if x in cidx])
        prune_watch(watch, t)
    return watch


def get_signals_today(t):
    """보유 종목의 B 유지 여부 확인용: 기준일 스크리닝 결과"""
    df = lynch_metrics(idx[t])
    return [(cidx[x], 0, 0) for x in lynch_picks(df, 'B') if x in cidx]


def save_excel(path, sheets, pct_cols=('신호가대비', '수익률', '비중')):
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            for ci, c in enumerate(df.columns, start=1):
                width = max([len(str(c)) * 2] + [len(str(v)) for v in df[c].head(200)]) + 2
                ws.column_dimensions[ws.cell(row=1, column=ci).column_letter].width = min(width, 40)
                if c in pct_cols:
                    for r in range(2, len(df) + 2):
                        ws.cell(row=r, column=ci).number_format = '0.0%'


def daily_report(as_of=None, chosen=CHOSEN, capital=CAPITAL):
    t = n - 1 if as_of is None else int(idx.searchsorted(pd.Timestamp(as_of), side='right') - 1)
    d = idx[t]
    max_hold = HOLD_OPTS[chosen['hold']]
    print(f'기준일 {d.date()}  설정 {chosen}  (최대 {MAX_POS}종목, 손절 {STOP_LOSS:.0%}, 고점대비 {TRAIL_STOP:.0%}, 최대보유 {max_hold}거래일)')
    if (idx[-1] - d).days == 0 and (pd.Timestamp.today().normalize() - d).days > 4:
        print('주의: 가격 데이터가 오늘보다 4일 넘게 오래됐습니다. kor_price 를 갱신하고 [린치-4] 부터 다시 실행하세요.')

    watch = build_watch(t)

    # 보유 종목
    hold = pd.DataFrame(columns=['종목코드', '매수일', '매수가'])
    hold_df = pd.DataFrame()
    try:
        hold = pd.read_csv(HOLDINGS_FILE, dtype={'종목코드': str})
        hold['종목코드'] = hold['종목코드'].str.zfill(6)
        hold['매수일'] = pd.to_datetime(hold['매수일'])
    except FileNotFoundError:
        print(f'\n{HOLDINGS_FILE} 이 없어 보유 종목 점검은 건너뜁니다. (열: 종목코드, 매수일, 매수가)')
    held_idx = set()
    if len(hold):
        b_now = {cols[j] for j, _, _ in get_signals_today(t)}
        out = []
        for _, r in hold.iterrows():
            if r['종목코드'] not in cidx:
                out.append((r['종목코드'], '가격 데이터 없음', None, None, None, None, None, '확인 필요'))
                continue
            j = cidx[r['종목코드']]
            held_idx.add(j)
            px = Pf[t, j]
            i_buy = int(idx.searchsorted(r['매수일']))
            peak = max(np.nanmax(Pf[i_buy:t + 1, j]), r['매수가'])
            days = t - i_buy
            why = '유지'
            if px <= r['매수가'] * (1 - STOP_LOSS):
                why = '손절'
            elif TRAIL_STOP and px <= peak * (1 - TRAIL_STOP):
                why = '고점대비하락'
            elif days >= max_hold:
                why = '보유기간'
            note = '' if r['종목코드'] in b_now else ' (이번 스크리닝에서 B 탈락)'
            out.append((r['종목코드'], name_s.get(r['종목코드'], ''), r['매수가'], px, px / r['매수가'] - 1, days,
                        r['매수가'] * (1 - STOP_LOSS), why + note))
        hold_df = pd.DataFrame(out, columns=['종목코드', '종목명', '매수가', '현재가', '수익률', '보유일', '손절가', '판단']).round(4)
        print('\n[보유 종목 점검]')
        print(hold_df.to_string(index=False))

    # 매수 후보 / 대기
    slots = MAX_POS - len(held_idx)
    cand, wait = [], []
    for j, w in watch.items():
        if j in held_idx:
            continue
        ok = entry_ok(chosen['entry'], w, j, t)
        row = dict(종목코드=cols[j], 종목명=name_s.get(cols[j], ''), 신호일=idx[w['sig_t']].date(), 신호가=w['sig_px'],
                   현재가=Pf[t, j], 신호가대비=Pf[t, j] / w['sig_px'] - 1, PEG=w['peg'],
                   손절가=Pf[t, j] * (1 - STOP_LOSS))
        (cand if ok else wait).append((row, j, w))
    cand.sort(key=lambda x: x[0]['PEG'] if x[0]['PEG'] > 0 else 9.9)
    ref = np.nanmedian(VOLM[t])
    sel = cand[:max(slots, 0)]
    buy_df = pd.DataFrame()
    print(f'\n[오늘 매수 후보] 진입 방식 {chosen["entry"]}, 빈 자리 {max(slots, 0)}개, 후보 {len(cand)}개 (PEG 낮은 순으로 빈 자리만큼)')
    if sel:
        sc = np.array([weight_score(chosen['weight'], j, t, w, ref) for _, j, w in sel])
        wt = sc / MAX_POS
        if wt.sum() > slots / MAX_POS:
            wt = wt * (slots / MAX_POS) / wt.sum()
        buy_df = pd.DataFrame([r for r, _, _ in sel])
        buy_df['비중'] = wt
        if capital:
            buy_df['금액(원)'] = (wt * capital).round(-3)
            buy_df['수량'] = (wt * capital / buy_df['현재가']).astype(int)
        buy_df = buy_df.round(4)
        print(buy_df.to_string(index=False))
        print('체결은 다음 거래일이라는 가정으로 검증했습니다. 위 가격은 오늘 종가입니다.')
    else:
        print('없음')

    wait_df = pd.DataFrame()
    print(f'\n[대기 목록] 워치리스트 중 아직 진입 조건이 안 된 종목 {len(wait)}개')
    if wait:
        wait_df = pd.DataFrame([r for r, _, _ in wait]).drop(columns=['손절가'])
        w_px = [w['sig_px'] for _, _, w in wait]
        if chosen['entry'] == 'pullback':
            wait_df['진입가(이하)'] = np.array(w_px) * (1 - PULLBACK)
            wait_df['신호폐기가'] = np.array(w_px) * (1 - STOP_LOSS)
        wait_df = wait_df.sort_values('신호가대비').round(4)
        print(wait_df.to_string(index=False))

    info = pd.DataFrame({'항목': ['기준일', '설정', '최대 종목 수', '손절', '고점대비 청산', '최대 보유(거래일)', '원금', '주의'],
                         '값': [str(d.date()), str(chosen), MAX_POS, f'{STOP_LOSS:.0%}', f'{TRAIL_STOP:.0%}', max_hold,
                                capital, '백테스트 기반 참고 자료이며 수익을 보장하지 않습니다. 손절/청산 규칙은 가정입니다.']})
    sheets = {'요약': info}
    if len(buy_df):
        sheets['매수후보'] = buy_df
    if len(hold_df):
        sheets['보유점검'] = hold_df
    if len(wait_df):
        sheets['대기목록'] = wait_df
    path = os.path.abspath(os.path.join(OUT_DIR, f'lynch_daily_{d:%Y%m%d}.xlsx'))
    save_excel(path, sheets)
    print(f'\n엑셀 저장: {path}')
    return dict(buy=buy_df, hold=hold_df, wait=wait_df)


report = daily_report()
