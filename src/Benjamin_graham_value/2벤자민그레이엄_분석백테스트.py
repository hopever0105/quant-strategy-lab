#%% [그레이엄-0] 안내
"""
벤저민 그레이엄 방어적 투자자 기준: 종목 분석 + 백테스트 (엄격 버전 / 현실 버전)

먼저 '1_자료수집.py' 를 보고 오세요.
이미 DB(stock_db)에 kor_ticker, kor_price, kor_fs 가 있으면 여기서부터 진행하세요.

[데이터 선택 - [그레이엄-1] 맨 위 두 줄]
FS_SOURCE = 'fnguide' : 1_자료수집.py 가 받은 kor_fs (연간 4~5년)
FS_SOURCE = 'dart'    : 3_장기데이터_수집.py 가 받은 dart_fs (2013년~, 엄격 버전용)
INCLUDE_DELISTED = True : 4_상장폐지_수집.py 가 받은 상장폐지 종목도 대상에 넣는다 (생존편향 완화)

[이 파일이 하는 일]
1) 그레이엄 7가지 기준으로 종목 스크리닝 (지금 시점)
2) 재무 조건은 통과했지만 가격이 아직 비싼 종목의 '매수 상한가' 계산 (언제 사야 하나)
3) 과거 백테스트 두 가지
   - 정기 방식 : 매년 4월 1일에 통과 종목을 동일가중으로 사서 1년 보유
   - 신호 방식 : 재무 조건 통과 종목을 지켜보다가 가격이 매수 상한가 아래로 내려온 날 매수
4) 결과의 한계 자동 점검 + 기준값을 조금 바꿨을 때도 결과가 유지되는지(민감도)

[엄격 버전 / 현실 버전]
엄격 : 원전에 가까운 기간(10년 흑자, 10년 성장). 연간 재무가 10년치 있어야 계산된다.
현실 : DB 에 있는 연간 재무 연수에 맞춘 버전(기본 4년).
DB 에 연간 재무가 몇 년치뿐이면 엄격 버전은 통과 종목이 0개로 나오는 게 정상이다.

[DB 비밀번호]
코드에 쓰지 않는다. 환경변수 DB_PW 에 넣거나 실행할 때 입력한다.

[패키지]
기존 + matplotlib.   conda activate ta_env  ->  python -m pip install matplotlib
"""


#%% [그레이엄-1] 설정 + DB 연결 + 테이블 확인
import os
import getpass
import runpy
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect

RUN_DOWNLOAD = False   # True: 자료수집 파일을 실행한다 / False: 건너뛰고 DB 에서 바로 읽는다
DOWNLOAD_FILE = r'C:\경로\1_자료수집.py'   # 본인 파일 위치로 고칠 것

FS_SOURCE = 'dart'        # 'fnguide': kor_fs (연간 4~5년) / 'dart': dart_fs (2013년~, 3_장기데이터_수집.py)
INCLUDE_DELISTED = True     # True: 상장폐지 종목 포함 (4_상장폐지_수집.py 를 먼저 실행해야 한다)

# True 로 하면 파일 전체가 실행되고 KRX 에도 접속한다. 차단이 걱정되면 False 로 둔다.
if RUN_DOWNLOAD:
    runpy.run_path(DOWNLOAD_FILE)

DB_USER = 'root'
DB_HOST = '127.0.0.1'
DB_NAME = 'stock_db'
DB_PW = os.environ.get('DB_PW') or getpass.getpass('DB password: ')

engine = create_engine(
    f'mysql+pymysql://{DB_USER}:{quote_plus(DB_PW)}@{DB_HOST}:3306/{DB_NAME}')

# 필요한 테이블이 없거나 비어 있으면 어떤 게 문제인지 알려주고 멈춘다
need_tables = ['kor_ticker', 'kor_price', 'kor_fs' if FS_SOURCE == 'fnguide' else 'dart_fs']
if INCLUDE_DELISTED:
    need_tables += ['kor_delisted', 'kor_price_delisted']
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
                       '\n-> RUN_DOWNLOAD = True 로 바꿔 다시 실행하거나, 1_자료수집.py(dart_fs 는 3번, 폐지 종목은 4번 파일)를 먼저 실행하세요.')
print('테이블 확인 완료')


#%% [그레이엄-2] 계정명 확인
# 앞 이름은 이 코드에서 쓰는 짧은 이름, 뒤는 kor_fs 에 들어 있는 계정명이다.
# 확인된 건 매출액(수익), 당기순이익, 자본총계 뿐이고 유동자산/유동부채/비유동부채는 아직 확인 안 됨.
ACC = {
    'sales':  '매출액(수익)',
    'ni':     '당기순이익',
    'equity': '자본총계',
    'ca':     '유동자산',
    'cl':     '유동부채',
    'ltd':    '비유동부채',     # 장기부채 대용. 장기차입금+사채 계정이 있으면 그걸로 바꾸는 게 더 정확하다
}

if FS_SOURCE == 'fnguide':
    acc_in_db = pd.read_sql("""
    select distinct 계정 from kor_fs where 공시구분 = 'y';
    """, con=engine)['계정'].tolist()

    no_acc = [v for v in ACC.values() if v not in acc_in_db]
    if no_acc:
        print('kor_fs 에 없는 계정:', no_acc)
        print('비슷한 이름 후보:')
        print([a for a in acc_in_db if ('유동' in a) or ('부채' in a) or ('차입' in a)])
        raise RuntimeError('후보를 보고 ACC 의 계정명을 DB 에 있는 이름으로 바꾸세요.')
else:
    # dart_fs 는 3_장기데이터_수집.py 가 이미 짧은 이름(sales, ni ...)으로 저장해 둔다
    acc_in_db = pd.read_sql('select distinct 계정 from dart_fs;', con=engine)['계정'].tolist()
    no_acc = [k for k in ACC.keys() if k not in acc_in_db]
    if no_acc:
        print('dart_fs 에 없는 계정:', no_acc, '/ 있는 계정:', acc_in_db)
        raise RuntimeError('3_장기데이터_수집.py 의 [DART-5] 커버리지를 확인하고 MAP 을 고치세요.')
print('계정 확인 완료 (', FS_SOURCE, ')')


#%% [그레이엄-3] 기준값
"""
그레이엄 방어적 투자자 기준 (현명한 투자자). 기억 기준이라 원문과 한 번 더 대조해 볼 것.

1) 적정 규모      : 원전은 매출 1억 달러(제조업). 원화 환산은 가정이라 직접 정한 값이다.
2) 유동비율       : 유동자산 / 유동부채 가 2 이상
3) 장기부채       : 장기부채 <= 순유동자산 (유동자산 - 유동부채)
4) 이익 안정성    : 원전은 10년 연속 흑자
5) 이익 성장      : 10년간 주당이익 1/3 이상 증가 (처음 3년 평균 vs 마지막 3년 평균)
6) PER            : 최근 3년 평균이익 기준 15 이하
7) PBR            : 1.5 이하. 또는 PER x PBR 이 22.5 이하
                    PER 15 이하를 이미 요구하므로 실제로는 PER x PBR <= 22.5 가 PBR 조건을 포함한다
(배당 20년 연속은 DB 에 과거 배당이 없어서 뺐다)

[매수 상한가]
PER <= 15 와 PER x PBR <= 22.5 를 시가총액 기준으로 풀면
  시가총액 <= min( 15 x 3년 평균 순이익 ,  루트(22.5 x 3년 평균 순이익 x 자본총계) )
가 된다. 주식수가 약분되어 사라지므로 주식수 추정 오차의 영향을 받지 않는다.
이 값보다 시가총액이 낮아지는 날이 '진입 시점'이다.
"""
MIN_SALES = 1000            # 억원
MIN_CURRENT_RATIO = 2.0
EPS_GROWTH_MIN = 1 / 3
MAX_PER = 15
MAX_PER_PBR = 22.5          # 1.5 x 15
DISCLOSURE_LAG = 90         # 결산일 + 90일 이후에야 재무를 알 수 있다고 본다 (미래 정보 사용 방지)

REBAL_MD = '04-01'          # 매년 리밸런싱 날짜(월-일). 12월 결산 사업보고서가 나온 뒤
TRADING_COST = 0.003        # 한 번 사고팔 때 드는 비용 가정치(왕복)
TARGET_GAIN = 0.5           # 신호 방식 청산 기준. 매수가 대비 +50% 도달 시 매도
                            # (그레이엄이 저평가주에 제시한 규칙으로 기억하지만 원문 확인은 못 했다. 변수로 두고 바꿔 본다)

# 엄격 / 현실 버전에서 다른 값들
MODES = {
    'strict': dict(PROFIT_YEARS=10, GROWTH_WINDOW=10, GROWTH_AVG_YEARS=3, GROWTH_MIN_YEARS=10, PER_AVG_YEARS=3),
    'real':   dict(PROFIT_YEARS=4,  GROWTH_WINDOW=10, GROWTH_AVG_YEARS=1, GROWTH_MIN_YEARS=4,  PER_AVG_YEARS=3),
}
if FS_SOURCE == 'fnguide':
    # 확인된 사실: FnGuide 연간 재무는 결산 3개 연도(2023~2025)뿐이라, 현실 버전을 3년으로 줄여야 통과 종목이 나온다
    # 이 경우 백테스트는 2026-04 리밸런싱 한 번뿐이라 '지금 시점 후보'만 의미가 있다 (장기 백테스트는 FS_SOURCE = 'dart')
    MODES['real'].update(PROFIT_YEARS=3, GROWTH_MIN_YEARS=3)
# PROFIT_YEARS    : 연속 흑자 연수
# GROWTH_WINDOW   : 이익 성장 비교 구간(연). 데이터가 짧으면 있는 만큼만 쓴다
# GROWTH_AVG_YEARS: 시작/끝 평균에 쓰는 연수 (원전 3)
# GROWTH_MIN_YEARS: 이익 성장 계산에 필요한 최소 연수
# PER_AVG_YEARS   : PER 에 쓰는 평균이익 연수


def set_mode(mode):
    """선택한 버전의 값을 전역 변수로 덮어쓴다. (스크리닝 함수가 전역 변수를 읽는다)"""
    globals().update(MODES[mode])


#%% [그레이엄-4] 데이터 불러오기
ticker_list = pd.read_sql("""
select 종목코드, 종목명, 시장구분, 종가, 시가총액, 기준일, 주당배당금
from kor_ticker
where 기준일 = (select max(기준일) from kor_ticker)
  and 종목구분 = '보통주';
""", con=engine)

if FS_SOURCE == 'fnguide':
    acc_sql = ', '.join(f"'{v}'" for v in ACC.values())
    fs_y = pd.read_sql(f"""
    select 종목코드, 계정, 기준일, 값
    from kor_fs
    where 공시구분 = 'y'
      and 계정 in ({acc_sql});
    """, con=engine)
    fs_y['계정'] = fs_y['계정'].map({v: k for k, v in ACC.items()})    # DB 계정명 -> 짧은 이름
else:
    keys_sql = ', '.join(f"'{k}'" for k in ACC.keys())
    fs_y = pd.read_sql(f"""
    select 종목코드, 계정, 기준일, 값
    from dart_fs
    where 계정 in ({keys_sql});
    """, con=engine)
fs_y['기준일'] = pd.to_datetime(fs_y['기준일'])

price = pd.read_sql("""
select 날짜, 종목코드, 종가 from kor_price;
""", con=engine)
price['날짜'] = pd.to_datetime(price['날짜'])
price_wide = price.pivot(index='날짜', columns='종목코드', values='종가').sort_index()
del price

# 과거 시가총액은 (현재 주식수 x 과거 수정주가) 로 근사한다.
# 수정주가는 액면분할 등이 반영되어 있어서 현재 주식수와 곱하면 대체로 맞지만,
# 유상증자나 자사주 소각이 많은 종목은 오차가 생긴다.
ticker_list['shares'] = (ticker_list['시가총액'] / ticker_list['종가']).replace([np.inf, -np.inf], np.nan)
shares = ticker_list.set_index('종목코드')['shares']
universe = set(ticker_list['종목코드'])
name_map = ticker_list.set_index('종목코드')[['종목명', '시장구분']]

# 시가총액(원) 표. 현재 상장 종목은 (수정주가 x 현재 주식수)
mcap_wide = price_wide.mul(shares.reindex(price_wide.columns), axis=1)

# 상장폐지 종목을 포함하면 가격표와 시가총액표에 폐지 종목 열을 붙인다 (시가총액은 FinanceDataReader 의 MarCap)
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
    print('상장폐지 종목 포함:', dl_p.shape[1], '개 (시가총액이 있는 종목', int(dl_m.notna().any().sum()), '개)')

fs_y = fs_y[fs_y['종목코드'].isin(universe)]

# 데이터 범위 확인 (엄격/현실 버전 중 어느 쪽이 가능한지 여기서 판단한다)
n_years = fs_y[fs_y['계정'] == 'ni'].groupby('종목코드')['기준일'].nunique()
print('재무 출처:', FS_SOURCE, '/ 대상 종목 수:', len(universe))
print('재무 기준일 범위:', fs_y['기준일'].min().date(), '~', fs_y['기준일'].max().date())
print('주가 범위       :', price_wide.index.min().date(), '~', price_wide.index.max().date())
print('종목별 연간 재무 연수 분포 (연수: 종목 수)')
print(n_years.value_counts().sort_index())
print('-> 가장 많은 종목의 연수가', int(n_years.mode().iloc[0]), '년. '
      'PROFIT_YEARS(현실)가 이보다 크면 통과 종목이 없으니 3번 셀 MODES 의 real 값을 낮출 것')


#%% [그레이엄-5] 스크리닝 함수
FIN_COLS = ['sales_ok', 'cr_ok', 'ltd_ok', 'profit_ok', 'growth_ok']     # 가격과 무관한 재무 조건
PRICE_COLS = ['per_ok', 'pbr_ok']                                        # 가격에 따라 매일 바뀌는 조건
OK_COLS = FIN_COLS + PRICE_COLS


def calc_growth(s):
    """구간 안에서 처음 평균 대비 마지막 평균의 증가율. 이익이 음수/0 이면 계산하지 않는다."""
    s = s.dropna().values
    if len(s) < GROWTH_MIN_YEARS:
        return np.nan
    first = s[:GROWTH_AVG_YEARS].mean()
    last_ = s[-GROWTH_AVG_YEARS:].mean()
    return last_ / first - 1 if first > 0 else np.nan


def graham_screen(as_of):
    """as_of 시점에 알 수 있었던 정보만으로 7가지 기준을 평가한다."""
    as_of = pd.Timestamp(as_of)
    empty = pd.DataFrame({'pass_all': pd.Series(dtype=bool),
                          'pass_fin': pd.Series(dtype=bool),
                          'ceil_mcap': pd.Series(dtype=float)})

    # 1) 아직 공시되지 않았을 재무는 제외
    fs = fs_y[fs_y['기준일'] + pd.Timedelta(days=DISCLOSURE_LAG) <= as_of]
    if fs.empty:
        return empty

    wide = fs.pivot_table(index=['종목코드', '기준일'], columns='계정', values='값').reset_index()
    wide.columns.name = None
    wide = wide.sort_values(['종목코드', '기준일'])
    for c in ACC.keys():
        if c not in wide.columns:
            wide[c] = np.nan

    g = wide.groupby('종목코드')
    last = g.tail(1).set_index('종목코드')                      # 종목별 가장 최근 연간 재무

    # 2) 최근 PROFIT_YEARS 년이 모두 흑자인가 (데이터가 모자라면 탈락)
    profit_ok = (g.tail(PROFIT_YEARS).groupby('종목코드')['ni']
                 .agg(lambda s: s.count() == PROFIT_YEARS and bool((s > 0).all())))

    # 3) 이익 성장률 (주당이익 대신 당기순이익, 주식수 변화는 무시)
    growth = g.tail(GROWTH_WINDOW).groupby('종목코드')['ni'].agg(calc_growth)

    # 4) 최근 PER_AVG_YEARS 년 평균 당기순이익
    avg_ni = (g.tail(PER_AVG_YEARS).groupby('종목코드')['ni']
              .agg(lambda s: s.mean() if s.count() == PER_AVG_YEARS else np.nan))

    # 5) as_of 이전 마지막 거래일의 시가총액(억원). 상장폐지된 종목은 폐지 뒤에 값이 없어서 자동으로 빠진다
    mcap = mcap_wide.loc[:as_of].iloc[-1] / 100000000

    df = pd.DataFrame(index=last.index)
    df['sales'] = last['sales']
    df['current_ratio'] = last['ca'] / last['cl']
    df['nca'] = last['ca'] - last['cl']                          # 순유동자산
    df['ltd'] = last['ltd']
    df['growth'] = growth
    df['avg_ni'] = avg_ni
    df['mcap'] = mcap.reindex(df.index)
    df['PER3'] = df['mcap'] / df['avg_ni']
    df['PBR'] = df['mcap'] / last['equity']
    df['PER_PBR'] = df['PER3'] * df['PBR']

    # 매수 상한가(시가총액 기준). 이익이나 자본이 0 이하이면 계산하지 않는다
    pos = (df['avg_ni'] > 0) & (last['equity'] > 0)
    ceil = np.minimum(MAX_PER * df['avg_ni'],
                      np.sqrt(MAX_PER_PBR * df['avg_ni'].clip(lower=0) * last['equity'].clip(lower=0)))
    df['ceil_mcap'] = ceil.where(pos)
    df['gap'] = df['mcap'] / df['ceil_mcap'] - 1                # 0 이하이면 상한가 아래(= 매수 가능)

    # NaN 이 들어간 비교는 False 가 되어 자동으로 탈락한다 (금융업은 유동자산 계정이 없어 여기서 빠진다)
    df['sales_ok'] = df['sales'] >= MIN_SALES
    df['cr_ok'] = df['current_ratio'] >= MIN_CURRENT_RATIO
    df['ltd_ok'] = df['ltd'] <= df['nca']
    df['profit_ok'] = profit_ok.reindex(df.index, fill_value=False).astype(bool)
    df['growth_ok'] = df['growth'] >= EPS_GROWTH_MIN
    df['per_ok'] = (df['PER3'] > 0) & (df['PER3'] <= MAX_PER)
    df['pbr_ok'] = (df['PBR'] > 0) & (df['PER_PBR'] <= MAX_PER_PBR)

    df['score'] = df[OK_COLS].sum(axis=1)
    df['pass_fin'] = df[FIN_COLS].all(axis=1)
    df['pass_all'] = df[OK_COLS].all(axis=1)

    df = df.join(name_map)
    return df[df.index.isin(universe)]


#%% [그레이엄-6] 지금 시점 스크리닝 + 진입 시점 후보
as_of_now = price_wide.index.max()
cols_show = ['종목명', '시장구분', 'sales', 'current_ratio', 'growth', 'PER3', 'PBR', 'PER_PBR', 'gap']

res_now = {}
for m in MODES:
    set_mode(m)
    r = graham_screen(as_of_now)
    res_now[m] = r
    print(f'===== {m} 버전 / 기준일 {as_of_now.date()} / 대상 {len(r)}개 =====')
    if len(r) == 0:
        print('재무 데이터가 없어 계산할 수 없습니다.')
        continue
    print('기준별 통과 종목 수')
    print(r[OK_COLS].sum().to_string())
    print('재무 조건 통과:', int(r['pass_fin'].sum()), '/ 모든 조건 통과:', int(r['pass_all'].sum()))

set_mode('real')
r = res_now['real']

# (1) 지금 사도 되는 종목 : 모든 조건 통과
print('\n[지금 매수 조건 충족] (현실 버전)')
print(r[r['pass_all']][cols_show].sort_values('PER_PBR').to_string())

# (2) 재무는 통과했지만 아직 비싼 종목 : gap 이 0 에 가까울수록 가격이 조금만 내려와도 진입 신호
watch = r[r['pass_fin'] & ~r['pass_all']].sort_values('gap')
print('\n[관심 목록: 재무 통과, 가격 대기] gap = 시가총액이 매수 상한가보다 몇 % 높은지')
print(watch[cols_show].head(30).to_string())

# (3) 참고 : 배당 (그레이엄은 20년 연속 배당을 요구하지만 DB 에는 현재 주당배당금만 있다)
res_div = r[r['pass_all']].join(ticker_list.set_index('종목코드')['주당배당금'])
print('\n[참고] 모든 조건 통과 종목 중 현재 배당이 있는 종목 수:', int((res_div['주당배당금'] > 0).sum()))

#%% [확인-A] 자본·순이익이 지배주주 기준인지 자동 점검
# DART 의 자본총계/당기순이익은 비지배지분을 포함한 값이고, 시가총액은 지배주주 몫만 반영한다.
# 지주회사처럼 비지배지분이 큰 회사는 PBR 이 실제보다 낮게 나올 수 있어서 FnGuide 값과 대조한다.
acc_fn = pd.read_sql("select distinct 계정 from kor_fs where 공시구분 = 'y';", con=engine)['계정'].tolist()
print('kor_fs 의 지배/비지배 관련 계정:', [a for a in acc_fn if '지배' in a])

# 1) 같은 종목, 같은 기준일에서 DART 값과 FnGuide 값이 얼마나 다른가
chk = {'equity': '자본총계', 'ni': '당기순이익'}
fn = pd.read_sql("""
select 종목코드, 계정, 기준일, 값 from kor_fs
where 공시구분 = 'y' and 계정 in ('자본총계', '당기순이익');
""", con=engine)
fn['계정'] = fn['계정'].map({v: k for k, v in chk.items()})
fn['기준일'] = pd.to_datetime(fn['기준일'])

m = fs_y[fs_y['계정'].isin(chk.keys())].merge(
    fn, on=['종목코드', '계정', '기준일'], suffixes=('_dart', '_fn'))
m = m[m['값_fn'] != 0].copy()
m['ratio'] = m['값_dart'] / m['값_fn']
print('\n[DART / FnGuide 비율] 중앙값이 1 근처면 단위와 기준이 같다는 뜻')
for a in chk:
    x = m[m['계정'] == a]['ratio']
    if len(x) == 0:
        print(a, ': 겹치는 (종목, 기준일)이 없음 -> 기준일 형식 확인 필요')
        continue
    print(f'{a}: 겹친 행 {len(x)} / 중앙값 {x.median():.3f} / '
          f'0.97~1.03 안 {((x > 0.97) & (x < 1.03)).mean():.1%} / '
          f'1.03 초과(DART가 큼) {(x >= 1.03).mean():.1%} / 0.97 미만 {(x <= 0.97).mean():.1%}')

# 2) 지배주주 기준으로 다시 계산한 PBR, PER x PBR (FnGuide 에 해당 계정이 있을 때만)
P_EQ, P_NI = '지배주주지분', '지배주주순이익'
r = res_now['real']
ok = r[r['pass_all']].copy()

if P_EQ in acc_fn:
    pe = pd.read_sql(f"select 종목코드, 기준일, 값 from kor_fs where 공시구분 = 'y' and 계정 = '{P_EQ}';", con=engine)
    pe['기준일'] = pd.to_datetime(pe['기준일'])
    pe_last = pe.sort_values('기준일').groupby('종목코드').tail(1).set_index('종목코드')['값']
    ok['PBR_지배'] = ok['mcap'] / pe_last.reindex(ok.index)
    ok['PERPBR_지배'] = ok['PER3'] * ok['PBR_지배']

    if P_NI in acc_fn:
        pn = pd.read_sql(f"select 종목코드, 기준일, 값 from kor_fs where 공시구분 = 'y' and 계정 = '{P_NI}';", con=engine)
        pn['기준일'] = pd.to_datetime(pn['기준일'])
        pn3 = pn.sort_values('기준일').groupby('종목코드').tail(3).groupby('종목코드')['값'].agg(
            lambda s: s.mean() if s.count() == 3 else np.nan)
        ok['PER3_지배'] = ok['mcap'] / pn3.reindex(ok.index)
        ok['PERPBR_지배'] = ok['PER3_지배'] * ok['PBR_지배']
        ok['여전히통과'] = (ok['PER3_지배'] > 0) & (ok['PER3_지배'] <= MAX_PER) & \
                       (ok['PBR_지배'] > 0) & (ok['PERPBR_지배'] <= MAX_PER_PBR)
    else:
        ok['여전히통과'] = (ok['PBR_지배'] > 0) & (ok['PERPBR_지배'] <= MAX_PER_PBR)

    print(f'\n현재 통과 {len(ok)}개 중 지배주주 기준으로도 통과: {int(ok["여전히통과"].sum())}개 '
          f'(FnGuide 값이 없는 종목은 탈락으로 센다)')
    show = ['종목명', 'PBR', 'PBR_지배', 'PER_PBR', 'PERPBR_지배', '여전히통과']
    ok['PBR차이'] = ok['PBR_지배'] / ok['PBR']
    print('\n[PBR 차이가 큰 종목 20개]')
    print(ok.sort_values('PBR차이', ascending=False)[show].head(20).to_string())
    print('\n[PBR 0.2 미만 종목]')
    print(ok[ok['PBR'] < 0.2][show].to_string())
else:
    print(f'\nkor_fs 에 "{P_EQ}" 계정이 없어 지배주주 기준 재계산은 건너뜁니다. 위 지배/비지배 목록에서 이름을 확인하세요.')
#%% [그레이엄-7] 백테스트 함수
"""
방식 A (정기): 매년 REBAL_MD 이후 첫 거래일에 스크리닝 -> 통과 종목 동일가중 매수 -> 다음 리밸런싱까지 보유.
방식 B (신호): 같은 날 '재무 조건만' 통과한 종목을 후보로 두고, 시가총액이 매수 상한가 아래로 내려온 첫 날 매수.
              매수가 대비 TARGET_GAIN 에 닿으면 매도, 아니면 다음 리밸런싱 날 정리한다.
              후보 한 종목에 배정한 돈은 매수 전까지 현금(수익률 0)으로 둔다.
비교 대상: 같은 방식으로 굴린 '전 종목 동일가중'.

한계 (결과를 볼 때 감안할 것)
- 생존편향 : kor_ticker 는 현재 상장 종목뿐이라 상장폐지 종목이 빠져 있다. 수익률이 실제보다 좋게 나올 수 있다.
- 시가총액 : 현재 주식수로 근사한 값
- 이익      : 주당이익이 아니라 당기순이익
- 기간      : 주가와 연간 재무가 동시에 있는 구간만 의미가 있고, 보통 표본이 몇 번 안 된다
- 거래 제약 : 거래정지, 호가 단위, 슬리피지는 반영하지 않았다
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
    """방식 A. get_picks(날짜) -> 종목코드 리스트. 일별 가치(NAV)와 매수 종목 수 기록을 돌려준다."""
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


def run_signal_backtest(dates, cands, ceils, target=TARGET_GAIN, cost=TRADING_COST):
    """방식 B. cands[d]: 재무 조건 통과 종목, ceils[d][종목]: 그 시점에 계산한 매수 상한가(시가총액, 억원)."""
    value = 1.0
    navs = []
    log = []
    entries = []
    for k, d0 in enumerate(dates):
        d1 = dates[k + 1] if k + 1 < len(dates) else idx[-1]
        seg_all = price_wide.loc[d0:d1]
        mc_all = mcap_wide.loc[d0:d1]
        cl = [c for c in cands[d0] if c in price_wide.columns]

        slots = pd.DataFrame(1.0, index=seg_all.index, columns=cl)             # 종목별 배정금의 가치. 처음엔 현금 1.0
        n_entry = 0
        for c in cl:
            px = seg_all[c].ffill()
            mc = mc_all[c].ffill() / 100000000
            hit = ((mc <= ceils[d0].get(c, np.nan)) & px.notna()).values
            if not hit.any():
                continue

            i = int(hit.argmax())                                              # 상한가 아래로 처음 내려온 날 = 진입일
            rel = (px / px.iloc[i]).values.copy()
            rel[:i] = 1.0                                                      # 진입 전에는 현금

            up = np.where(rel[i:] >= 1 + target)[0]                            # 목표 수익 도달 시 매도, 이후 그 값 유지
            exit_date = seg_all.index[-1]
            exit_reason = '기간종료'
            if len(up):
                j = i + int(up[0])
                rel[j:] = rel[j]
                exit_date = seg_all.index[j]
                exit_reason = '목표도달'

            rel[i:] *= (1 - cost)
            slots[c] = rel
            n_entry += 1
            entries.append((d0.date(), c, seg_all.index[i].date(), float(px.iloc[i]),
                            exit_date.date(), exit_reason, float(rel[-1] - 1)))

        if len(cl) == 0:
            seg_nav = pd.Series(value, index=seg_all.index)
        else:
            seg_nav = value * slots.mean(axis=1)

        value = seg_nav.iloc[-1]
        navs.append(seg_nav if k == 0 else seg_nav.iloc[1:])
        log.append((d0.date(), len(cl), n_entry))

    entries = pd.DataFrame(entries, columns=['리밸런싱일', '종목코드', '진입일', '진입가',
                                             '청산일', '청산사유', '기간수익률'])
    entries = entries.merge(name_map['종목명'].reset_index(), on='종목코드', how='left')
    return (pd.concat(navs),
            pd.DataFrame(log, columns=['리밸런싱일', '후보종목수', '실제진입수']),
            entries)


def run_all(mode):
    """한 버전(strict / real)의 스크리닝 + 두 방식 백테스트 + 비교 대상. 통과 종목이 없으면 None."""
    set_mode(mode)
    dates = make_rebal_dates()

    scr = {d: graham_screen(d) for d in dates}
    picks = {d: list(s.index[s['pass_all']]) for d, s in scr.items()}
    cands = {d: list(s.index[s['pass_fin']]) for d, s in scr.items()}
    ceils = {d: s['ceil_mcap'].to_dict() for d, s in scr.items()}

    # 재무 데이터가 아직 없어서 후보가 없던 앞쪽 날짜는 버린다
    while dates and len(cands[dates[0]]) == 0:
        dates.pop(0)
    if not dates:
        return None

    nav_a, log_a = run_backtest(dates, lambda d: picks[d])
    nav_b, log_b, entries = run_signal_backtest(dates, cands, ceils)
    nav_m, _ = run_backtest(dates, lambda d: bench_picks)

    return dict(dates=dates, nav_a=nav_a, log_a=log_a, nav_b=nav_b, log_b=log_b,
                entries=entries, nav_m=nav_m)


#%% [그레이엄-8] 백테스트 실행 (엄격 / 현실)
results = {}
for m in MODES:
    out = run_all(m)
    if out is None:
        print(f'[{m}] 재무 조건을 통과하는 종목이 있는 리밸런싱 날짜가 없어 백테스트를 건너뜁니다. '
              '(필요한 재무 연수가 DB 보다 길거나 기준이 너무 엄격함)')
        continue
    results[m] = out
    print(f'\n[{m}] 리밸런싱 {len(out["dates"])}회')
    print(out['log_a'].merge(out['log_b'], on='리밸런싱일').to_string(index=False))

set_mode('real')
if 'real' in results:
    print('\n[현실 버전] 신호 방식 진입 내역 (언제 샀고 언제 팔았나)')
    print(results['real']['entries'].to_string(index=False))


#%% [그레이엄-9] 성과 비교
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

if results:
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), squeeze=False)
    for ax, (m, o) in zip(axes[0], results.items()):
        table = pd.concat([perf(o['nav_a']), perf(o['nav_b']), perf(o['nav_m'])], axis=1,
                          keys=['정기(A)', '신호(B)', '전종목 동일가중'])
        print(f'\n===== {m} 버전 =====')
        print(table.round(4).to_string())

        o['nav_a'].plot(ax=ax, label='정기(A)')
        o['nav_b'].plot(ax=ax, label='신호(B)')
        o['nav_m'].plot(ax=ax, label='전종목 동일가중')
        ax.set_title(f'그레이엄 {m} 버전')
        ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print('백테스트 결과가 없습니다. [그레이엄-8] 의 메시지를 확인하세요.')


#%% [그레이엄-10] 결과의 한계 자동 점검
def yearly(nav):
    y = nav.groupby(nav.index.year).last()
    r = y.pct_change()
    r.iloc[0] = y.iloc[0] / nav.iloc[0] - 1
    return r


for m, o in results.items():
    print(f'\n===== {m} 버전 점검 =====')
    years = (o['nav_a'].index[-1] - o['nav_a'].index[0]).days / 365.25
    n_reb = len(o['dates'])
    avg_pick = o['log_a']['매수종목수'].mean()
    zero_pick = int((o['log_a']['매수종목수'] == 0).sum())
    n_entry = int(o['log_b']['실제진입수'].sum())

    print(f'기간 {years:.1f}년 / 리밸런싱 {n_reb}회 / 정기 방식 평균 매수종목 {avg_pick:.1f}개 / '
          f'현금 기간 {zero_pick}회 / 신호 방식 진입 {n_entry}건')

    warn = []
    if years < 5:
        warn.append('기간이 5년 미만이라 한두 해 시장 흐름이 결과를 좌우한다')
    if n_reb < 5:
        warn.append('리밸런싱이 5회 미만이라 통계적 결론을 내기 어렵다')
    if avg_pick < 10:
        warn.append('평균 매수 종목이 10개 미만이라 개별 종목 한두 개가 결과를 좌우한다')
    if n_entry < 20:
        warn.append('신호 방식 진입이 20건 미만이라 표본이 부족하다')
    if INCLUDE_DELISTED:
        warn.append('상장폐지 종목을 넣었다. MarCap 단위(원)는 확인했지만 폐지 종목 종가가 수정주가인지는 확인되지 않았다. '
            '재무를 받지 못한 폐지 종목(감사의견 거절, 합병 등)은 빠져 있어 생존편향이 일부 남는다')
    else:
        warn.append('생존편향: 상장폐지 종목이 빠져 있어 수익률이 높게 나올 수 있다 (INCLUDE_DELISTED = True 로 완화)')
    if FS_SOURCE == 'fnguide':
        warn.append('재무가 4~5년치뿐이라 엄격 버전은 계산되지 않는다 (FS_SOURCE = "dart" 로 장기 재무를 쓴다)')
    warn.append('현재 상장 종목의 과거 시가총액은 현재 주식수 근사, 이익은 순이익(주당이익 아님)')
    warn.append('거래비용은 가정치, 슬리피지/거래정지는 미반영')
    for w in warn:
        print(' -', w)

    # 연도별 수익률: 초과수익이 한 해에만 몰려 있는지 본다
    yr = pd.concat([yearly(o['nav_a']), yearly(o['nav_b']), yearly(o['nav_m'])], axis=1,
                   keys=['정기(A)', '신호(B)', '전종목 동일가중'])
    yr['A-벤치'] = yr['정기(A)'] - yr['전종목 동일가중']
    print('\n연도별 수익률')
    print(yr.round(4).to_string())
    print('정기(A)가 벤치마크보다 나은 해:', int((yr['A-벤치'] > 0).sum()), '/', len(yr))

"""
[가능한 전략인지 보는 기준]
1) 초과수익이 여러 해에 나뉘어 있는가 (한 해에 몰려 있으면 우연일 가능성이 크다)
2) 매수 종목 수와 리밸런싱 횟수가 충분한가 (위 경고가 없는가)
3) 거래비용을 TRADING_COST 의 2~3배로 올려도 벤치마크보다 나은가
4) 기준값을 조금 바꿔도 결과가 비슷한가 -> 다음 셀(민감도)
이 중 하나라도 안 되면 '전략이 통한다'가 아니라 '이 표본에서는 그렇게 보인다' 정도로만 해석한다.
"""


#%% [그레이엄-11] 민감도: 기준값을 조금 바꿔도 결과가 유지되는가 (현실 버전)
base = dict(MAX_PER_PBR=MAX_PER_PBR, MIN_CURRENT_RATIO=MIN_CURRENT_RATIO)
rows = []
for pbr in [18, 22.5, 30]:
    for cr in [1.5, 2.0]:
        globals().update(MAX_PER_PBR=pbr, MIN_CURRENT_RATIO=cr)
        o = run_all('real')
        if o is None:
            rows.append((pbr, cr, np.nan, np.nan, np.nan, np.nan))
            continue
        pa, pb, pm = perf(o['nav_a']), perf(o['nav_b']), perf(o['nav_m'])
        rows.append((pbr, cr, pa['CAGR'], pb['CAGR'], pm['CAGR'], o['log_a']['매수종목수'].mean()))
globals().update(base)

sens = pd.DataFrame(rows, columns=['PER_PBR 상한', '유동비율 하한', 'CAGR 정기(A)', 'CAGR 신호(B)',
                                   'CAGR 벤치', 'A 평균종목수'])
print(sens.round(4).to_string(index=False))
# 기준값이 조금 바뀔 때 결과가 크게 흔들리면 그 전략은 신뢰하기 어렵다.