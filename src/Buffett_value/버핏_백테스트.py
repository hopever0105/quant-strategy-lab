#%% [버핏-0] 안내
"""
버핏식 우량 기업 기준: 종목 분석 + 백테스트 (엄격 버전 / 현실 버전)

먼저 아래 순서로 데이터를 모아 오세요. 이미 DB(stock_db)에 해당 테이블이 있으면 여기서부터 진행하세요.
  1) 1_자료수집.py          -> kor_ticker, kor_price            ([수집-0] ~ [수집-9])
  2) 3_장기데이터_수집.py    -> dart_corp, dart_fs, crawl_log     ([DART-1] ~ [DART-6])
     [DART-4] 는 하루 호출 한도에 걸리면 멈춘다. 다음 날 같은 셀을 다시 실행하면 이어받는다.
  3) 4_상장폐지_수집.py      -> kor_delisted, kor_price_delisted  ([상폐-0] ~ [상폐-5]) (INCLUDE_DELISTED = True 일 때)
  4) 이 파일

[이 파일이 하는 일]
1) 버핏식 기준으로 종목 스크리닝 (지금 시점)
2) 재무는 통과했지만 아직 비싼 종목의 '매수 상한가' 계산
3) 과거 백테스트 두 가지
   - 정기 방식 : 매년 4월 1일에 통과 종목을 동일가중으로 사서 1년 보유
   - 신호 방식 : 재무 조건 통과 종목을 지켜보다가 가격이 매수 상한가 아래로 내려온 날 매수
4) 구간별 성과(특정 연도를 빼면 어떤가) + 결과의 한계 자동 점검 + 기준값을 바꿨을 때의 민감도

[엄격 버전 / 현실 버전]
엄격 : 10년 치 재무로 평가한다. DART 재무가 2013년부터라 2023년 4월 이후 4번 정도밖에 계산되지 않는다.
현실 : 5년 치 재무로 평가한다. 2018년 4월부터 계산된다.

[재무 출처]
dart_fs 만 쓴다. 영업현금흐름, 설비투자, 영업이익, 부채가 필요한데 kor_fs 는 3개 연도뿐이다.

[DB 비밀번호]
코드에 쓰지 않는다. 환경변수 DB_PW 에 넣거나 실행할 때 입력한다.

[패키지 / 가상환경]
추가로 필요한 건 matplotlib 뿐이다.
  Anaconda Prompt 에서:  conda activate ta_env   ->   python -m pip install matplotlib
  Spyder 에서: 도구 > 환경설정 > Python 인터프리터 > '다음 인터프리터 사용' 을 고르고
               ta_env 의 python.exe (보통 C:\\Users\\사용자\\anaconda3\\envs\\ta_env\\python.exe) 를 지정한 뒤
               콘솔을 다시 시작한다.
"""


#%% [버핏-1] 설정 + DB 연결 + 테이블 확인
import os
import getpass
import runpy
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect

RUN_DOWNLOAD = False   # True: 자료수집 파일을 실행한다 / False: 건너뛰고 DB 에서 바로 읽는다
DOWNLOAD_FILE = r'C:\경로\3_장기데이터_수집.py'   # 본인 파일 위치로 고칠 것 (재무 수집 파일)

INCLUDE_DELISTED = True      # True: 상장폐지 종목 포함 (4_상장폐지_수집.py 를 먼저 실행해야 한다)

# True 로 하면 파일 전체가 실행되고 DART 에 접속한다. 하루 호출 한도가 있어서 이어받기로 며칠 걸릴 수 있다.
if RUN_DOWNLOAD:
    runpy.run_path(DOWNLOAD_FILE)

DB_USER = 'root'
DB_HOST = '127.0.0.1'
DB_NAME = 'stock_db'
DB_PW = os.environ.get('DB_PW') or getpass.getpass('DB password: ')

engine = create_engine(
    f'mysql+pymysql://{DB_USER}:{quote_plus(DB_PW)}@{DB_HOST}:3306/{DB_NAME}')

# 필요한 테이블이 없거나 비어 있으면 어떤 게 문제인지 알려주고 멈춘다
need_tables = ['kor_ticker', 'kor_price', 'dart_fs']
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
                       '\n-> RUN_DOWNLOAD = True 로 바꿔 다시 실행하거나, 1_자료수집.py / 3_장기데이터_수집.py / '
                       '4_상장폐지_수집.py 를 먼저 실행하세요.')

# DART 수집이 끝까지 갔는지 본다 (남은 종목은 재무가 없어 후보에서 빠진다)
if 'crawl_log' in have_tables:
    # [DART-4] 와 같은 기준으로 대상을 센다 (현재 상장 보통주 + 상장폐지 종목)
    target = set(pd.read_sql("""
    select 종목코드 from kor_ticker
    where 기준일 = (select max(기준일) from kor_ticker) and 종목구분 = '보통주';
    """, con=engine)['종목코드'])
    if 'kor_delisted' in have_tables:
        target |= set(pd.read_sql('select 종목코드 from kor_delisted;', con=engine)['종목코드'])
    done = set(pd.read_sql("select 종목코드 from crawl_log where 대상 = 'dart';", con=engine)['종목코드'])
    n_left = len(target - done)
    print(f'DART 재무 처리: {len(target & done)} / 대상 {len(target)} (남은 종목 {n_left})')
    if n_left > 0:
        print('-> 아직 안 받은 종목이 있다. 3_장기데이터_수집.py 의 [DART-4] 를 다시 실행해 끝낸 뒤 이 파일을 다시 돌리는 게 좋다.')


#%% [버핏-2] 계정명 확인
# dart_fs 에는 3_장기데이터_수집.py 가 짧은 이름으로 저장해 둔다. 값의 단위는 억원.
#   capex 는 부호 표기가 회사마다 달라서 절댓값으로 저장되어 있다 (설비투자 지출액)
ACC = ['sales', 'op', 'ni', 'equity', 'liab', 'ca', 'cl', 'cfo', 'capex']

acc_in_db = pd.read_sql('select distinct 계정 from dart_fs;', con=engine)['계정'].tolist()
no_acc = [k for k in ACC if k not in acc_in_db]
if no_acc:
    print('dart_fs 에 없는 계정:', no_acc, '/ 있는 계정:', acc_in_db)
    raise RuntimeError('3_장기데이터_수집.py 의 [DART-5] 커버리지를 확인하고 MAP 을 고치세요.')
print('계정 확인 완료')


#%% [버핏-3] 기준값
"""
버핏식 기준. 버핏이 숫자로 못 박은 규칙은 없어서, 그의 투자 방식(높은 자기자본이익률, 꾸준한 현금 창출,
적은 부채, 합리적인 가격)을 숫자로 옮긴 것이다. 아래 값은 전부 직접 정한 가정이다.

1) ROE         : 최근 BUF_YEARS 년 평균 ROE >= ROE_AVG_MIN, 어느 해도 ROE_MIN_YEAR 아래로 내려가지 않는다
                 ROE = 당기순이익 / 자본총계
2) 현금 창출   : 자유현금흐름(FCF = 영업현금흐름 - 설비투자)이 BUF_YEARS 년 내내 흑자
3) 이익의 질   : BUF_YEARS 년 FCF 합계 / 순이익 합계 >= FCF_NI_MIN  (이익이 장부에만 있는 게 아닌지)
4) 이익 성장   : 순이익 연평균 증가율 >= NI_CAGR_MIN (처음/끝 GROWTH_AVG_YEARS 년 평균으로 비교)
5) 영업이익률  : 평균 >= OPM_AVG_MIN, 어느 해도 OPM_MIN_YEAR 아래로 내려가지 않는다
6) 부채        : 부채비율(부채총계 / 자본총계) <= DEBT_MAX
7) 유동비율    : MIN_CURRENT_RATIO 이상
8) 가격        : 평균 FCF / 시가총액 (FCF 수익률) >= MIN_FCF_YIELD,  선택적으로 PBR <= MAX_PBR
                 (ROE 가 높으면 PBR 도 높게 나오는 게 정상이라 MAX_PBR 는 기본 None, 즉 쓰지 않는다)
9) 규모        : 최근 매출 >= MIN_SALES (너무 작은 회사는 거래가 어렵다)

[매수 상한가]
FCF 수익률 조건을 시가총액으로 풀면
  시가총액 <= 평균 FCF / MIN_FCF_YIELD      (MAX_PBR 를 쓰면 MAX_PBR x 자본총계 와 비교해 작은 쪽)
이 값보다 시가총액이 낮아지는 날이 '진입 시점'이다.

[한계]
- ROE, FCF, 순이익은 비지배지분을 포함한 연결 값이고 시가총액은 지배주주 몫이라 지주회사류는 FCF 수익률이 부풀 수 있다
- 설비투자 계정이 없는 회사(금융, 일부 서비스업)는 FCF 를 못 구해 후보에서 빠진다
"""
MIN_SALES = 500             # 억원
ROE_AVG_MIN = 0.12
ROE_MIN_YEAR = 0.08
FCF_NI_MIN = 0.7
NI_CAGR_MIN = 0.07
OPM_AVG_MIN = 0.10
OPM_MIN_YEAR = 0.05
DEBT_MAX = 1.0              # 부채비율 100%
MIN_CURRENT_RATIO = 1.5
MIN_FCF_YIELD = 0.05        # 평균 FCF / 시가총액
MAX_PBR = None              # None 이면 PBR 상한을 쓰지 않는다

STALE_DAYS = 540            # 가장 최근 재무가 이 일수보다 오래됐으면 제외 (공시를 멈춘 회사 방지)
DISCLOSURE_LAG = 90         # 결산일 + 90일 이후에야 재무를 알 수 있다고 본다 (미래 정보 사용 방지)

REBAL_MD = '04-01'          # 매년 리밸런싱 날짜(월-일). 12월 결산 사업보고서가 나온 뒤
TRADING_COST = 0.003        # 한 번 사고팔 때 드는 비용 가정치(왕복)
TARGET_GAIN = None          # 신호 방식 청산 기준. None 이면 목표 수익 매도 없이 다음 리밸런싱까지 보유한다
                            # (버핏식은 오래 들고 가는 방식이라 기본은 쓰지 않는다. 0.5 처럼 넣으면 +50% 에서 매도)

# 엄격 / 현실 버전에서 다른 값들
MODES = {
    'strict': dict(BUF_YEARS=10, GROWTH_AVG_YEARS=2),
    'real':   dict(BUF_YEARS=5,  GROWTH_AVG_YEARS=2),
}
# BUF_YEARS        : 평가에 쓰는 연간 재무 연수
# GROWTH_AVG_YEARS : 이익 성장률에서 처음/끝 평균에 쓰는 연수 (한 해의 일회성 이익에 흔들리지 않게)


def set_mode(mode):
    """선택한 버전의 값을 전역 변수로 덮어쓴다. (스크리닝 함수가 전역 변수를 읽는다)"""
    globals().update(MODES[mode])


#%% [버핏-4] 데이터 불러오기
ticker_list = pd.read_sql("""
select 종목코드, 종목명, 시장구분, 종가, 시가총액, 기준일, 주당배당금
from kor_ticker
where 기준일 = (select max(기준일) from kor_ticker)
  and 종목구분 = '보통주';
""", con=engine)

keys_sql = ', '.join(f"'{k}'" for k in ACC)
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
print('대상 종목 수:', len(universe))
print('재무 기준일 범위:', fs_y['기준일'].min().date(), '~', fs_y['기준일'].max().date())
print('주가 범위       :', price_wide.index.min().date(), '~', price_wide.index.max().date())
print('종목별 연간 재무 연수 분포 (연수: 종목 수)')
print(n_years.value_counts().sort_index())
for m, v in MODES.items():
    print(f'-> {m} 버전은 재무가 {v["BUF_YEARS"]}년 이상인 종목이 {int((n_years >= v["BUF_YEARS"]).sum())}개')


#%% [버핏-5] 스크리닝 함수
FIN_COLS = ['data_ok', 'sales_ok', 'roe_ok', 'fcf_ok', 'conv_ok', 'growth_ok', 'opm_ok', 'debt_ok', 'cr_ok']  # 가격과 무관
PRICE_COLS = ['yield_ok', 'pbr_ok']                                                                           # 가격에 따라 매일 바뀜
OK_COLS = FIN_COLS + PRICE_COLS


def buffett_screen(as_of):
    """as_of 시점에 알 수 있었던 정보만으로 기준을 평가한다."""
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
    for c in ACC:
        if c not in wide.columns:
            wide[c] = np.nan

    # 연도별 지표. 분모가 0 이하이면 계산하지 않는다
    wide['roe'] = wide['ni'] / wide['equity'].where(wide['equity'] > 0)
    wide['opm'] = wide['op'] / wide['sales'].where(wide['sales'] > 0)
    wide['fcf'] = wide['cfo'] - wide['capex']                    # capex 는 절댓값(지출액)으로 저장되어 있다

    g = wide.groupby('종목코드')
    last = g.tail(1).set_index('종목코드')                        # 종목별 가장 최근 연간 재무
    tail = g.tail(BUF_YEARS)                                      # 종목별 최근 BUF_YEARS 개 연도
    gt = tail.groupby('종목코드')

    # 2) 최근 BUF_YEARS 년이 빠짐없이 이어져 있는가 (중간에 빠진 해가 있으면 탈락)
    span = gt['기준일'].agg(['min', 'max'])
    consecutive = (span['max'].dt.year - span['min'].dt.year) == (BUF_YEARS - 1)
    full = (gt['ni'].count() == BUF_YEARS) & (gt['cfo'].count() == BUF_YEARS) & \
           (gt['capex'].count() == BUF_YEARS)
    fresh = last['기준일'] >= as_of - pd.Timedelta(days=STALE_DAYS)

    # 3) ROE, 영업이익률, FCF
    roe_ok = (gt['roe'].count() == BUF_YEARS) & (gt['roe'].mean() >= ROE_AVG_MIN) & (gt['roe'].min() >= ROE_MIN_YEAR)
    opm_ok = (gt['opm'].count() == BUF_YEARS) & (gt['opm'].mean() >= OPM_AVG_MIN) & (gt['opm'].min() >= OPM_MIN_YEAR)
    fcf_ok = (gt['fcf'].count() == BUF_YEARS) & (gt['fcf'].min() > 0)

    fcf_sum = gt['fcf'].sum(min_count=BUF_YEARS)
    ni_sum = gt['ni'].sum(min_count=BUF_YEARS)
    fcf_ni = fcf_sum / ni_sum.where(ni_sum > 0)                   # 순이익 합계가 0 이하이면 계산하지 않는다
    avg_fcf = fcf_sum / BUF_YEARS
    avg_ni = ni_sum / BUF_YEARS

    # 4) 순이익 연평균 증가율: 처음/끝 GROWTH_AVG_YEARS 년 평균 사이의 연평균 증가율
    first_ni = tail.groupby('종목코드').head(GROWTH_AVG_YEARS).groupby('종목코드')['ni'].mean()
    last_ni = tail.groupby('종목코드').tail(GROWTH_AVG_YEARS).groupby('종목코드')['ni'].mean()
    gap_years = BUF_YEARS - GROWTH_AVG_YEARS                      # 두 평균의 가운데 지점 사이 햇수
    ok_g = (first_ni > 0) & (last_ni > 0)
    cagr = (last_ni.where(ok_g) / first_ni.where(ok_g)) ** (1 / gap_years) - 1

    # 5) as_of 이전 마지막 거래일의 시가총액(억원). 상장폐지된 종목은 폐지 뒤에 값이 없어서 자동으로 빠진다
    mcap = mcap_wide.loc[:as_of].iloc[-1] / 100000000

    df = pd.DataFrame(index=last.index)
    df['sales'] = last['sales']
    df['roe_avg'] = gt['roe'].mean()
    df['roe_min'] = gt['roe'].min()
    df['opm_avg'] = gt['opm'].mean()
    df['fcf_ni'] = fcf_ni
    df['cagr'] = cagr
    df['debt_ratio'] = last['liab'] / last['equity'].where(last['equity'] > 0)
    df['current_ratio'] = last['ca'] / last['cl']
    df['avg_fcf'] = avg_fcf
    df['avg_ni'] = avg_ni
    df['mcap'] = mcap.reindex(df.index)
    df['FCF_yield'] = df['avg_fcf'] / df['mcap']
    df['PER'] = df['mcap'] / df['avg_ni'].where(df['avg_ni'] > 0)
    df['PBR'] = df['mcap'] / last['equity'].where(last['equity'] > 0)

    # 매수 상한가(시가총액 기준). 평균 FCF 나 자본이 0 이하이면 계산하지 않는다
    ceil = df['avg_fcf'] / MIN_FCF_YIELD
    if MAX_PBR is not None:
        ceil = np.minimum(ceil, MAX_PBR * last['equity'])
    df['ceil_mcap'] = ceil.where((df['avg_fcf'] > 0) & (last['equity'] > 0))
    df['gap'] = df['mcap'] / df['ceil_mcap'] - 1                  # 0 이하이면 상한가 아래(= 매수 가능)

    # NaN 이 들어간 비교는 False 가 되어 자동으로 탈락한다 (금융업은 유동자산 계정이 없어 여기서 빠진다)
    df['data_ok'] = (full & consecutive & fresh).reindex(df.index, fill_value=False).astype(bool)
    df['sales_ok'] = df['sales'] >= MIN_SALES
    df['roe_ok'] = roe_ok.reindex(df.index, fill_value=False).astype(bool)
    df['fcf_ok'] = fcf_ok.reindex(df.index, fill_value=False).astype(bool)
    df['conv_ok'] = df['fcf_ni'] >= FCF_NI_MIN
    df['growth_ok'] = df['cagr'] >= NI_CAGR_MIN
    df['opm_ok'] = opm_ok.reindex(df.index, fill_value=False).astype(bool)
    df['debt_ok'] = df['debt_ratio'] <= DEBT_MAX
    df['cr_ok'] = df['current_ratio'] >= MIN_CURRENT_RATIO
    df['yield_ok'] = df['FCF_yield'] >= MIN_FCF_YIELD
    df['pbr_ok'] = True if MAX_PBR is None else (df['PBR'] > 0) & (df['PBR'] <= MAX_PBR)

    df['score'] = df[OK_COLS].sum(axis=1)
    df['pass_fin'] = df[FIN_COLS].all(axis=1)
    df['pass_all'] = df[OK_COLS].all(axis=1)

    df = df.join(name_map)
    return df[df.index.isin(universe)]


#%% [버핏-6] 지금 시점 스크리닝 + 진입 시점 후보
as_of_now = price_wide.index.max()
cols_show = ['종목명', '시장구분', 'roe_avg', 'roe_min', 'opm_avg', 'fcf_ni', 'cagr',
             'debt_ratio', 'current_ratio', 'PER', 'PBR', 'FCF_yield', 'gap']

res_now = {}
for m in MODES:
    set_mode(m)
    r = buffett_screen(as_of_now)
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
print(r[r['pass_all']][cols_show].sort_values('FCF_yield', ascending=False).round(3).to_string())

# (2) 재무는 통과했지만 아직 비싼 종목 : gap 이 0 에 가까울수록 가격이 조금만 내려와도 진입 신호
watch = r[r['pass_fin'] & ~r['pass_all']].sort_values('gap')
print('\n[관심 목록: 재무 통과, 가격 대기] gap = 시가총액이 매수 상한가보다 몇 % 높은지')
print(watch[cols_show].head(30).round(3).to_string())

# (3) 어느 조건에서 가장 많이 탈락하는가 (기준값을 손볼 때 참고)
print('\n[재무 조건별 통과 종목 수] 현실 버전 / 데이터가 있는 종목 중')
print(r[FIN_COLS].sum().to_string())

# (4) 참고 : 배당
res_div = r[r['pass_all']].join(ticker_list.set_index('종목코드')['주당배당금'])
print('\n[참고] 모든 조건 통과 종목 중 현재 배당이 있는 종목 수:', int((res_div['주당배당금'] > 0).sum()))


#%% [버핏-7] 백테스트 함수
"""
방식 A (정기): 매년 REBAL_MD 이후 첫 거래일에 스크리닝 -> 통과 종목 동일가중 매수 -> 다음 리밸런싱까지 보유.
방식 B (신호): 같은 날 '재무 조건만' 통과한 종목을 후보로 두고, 시가총액이 매수 상한가 아래로 내려온 첫 날 매수.
              TARGET_GAIN 이 None 이 아니면 매수가 대비 그만큼 오른 날 매도, 아니면 다음 리밸런싱 날 정리한다.
              후보 한 종목에 배정한 돈은 매수 전까지 현금(수익률 0)으로 둔다.
비교 대상: 같은 방식으로 굴린 '전 종목 동일가중' (코스피 지수가 아니다).

한계 (결과를 볼 때 감안할 것)
- 생존편향 : 상장폐지 종목을 넣어도 재무를 받지 못한 폐지 종목은 빠진다
- 시가총액 : 현재 주식수로 근사한 값
- 이익      : 연결 순이익(비지배지분 포함)
- 기간      : 5년 치 재무가 필요해서 2018년 이후만 계산된다
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


def run_signal_backtest(dates, cands, ceils, target=None, cost=TRADING_COST):
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

            exit_date = seg_all.index[-1]
            exit_reason = '기간종료'
            if target is not None:                                             # 목표 수익 도달 시 매도, 이후 그 값 유지
                up = np.where(rel[i:] >= 1 + target)[0]
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

    scr = {d: buffett_screen(d) for d in dates}
    picks = {d: list(s.index[s['pass_all']]) for d, s in scr.items()}
    cands = {d: list(s.index[s['pass_fin']]) for d, s in scr.items()}
    ceils = {d: s['ceil_mcap'].to_dict() for d, s in scr.items()}

    # 재무 데이터가 아직 없어서 후보가 없던 앞쪽 날짜는 버린다
    while dates and len(cands[dates[0]]) == 0:
        dates.pop(0)
    if not dates:
        return None

    nav_a, log_a = run_backtest(dates, lambda d: picks[d])
    nav_b, log_b, entries = run_signal_backtest(dates, cands, ceils, target=TARGET_GAIN)
    nav_m, _ = run_backtest(dates, lambda d: bench_picks)

    return dict(dates=dates, nav_a=nav_a, log_a=log_a, nav_b=nav_b, log_b=log_b,
                entries=entries, nav_m=nav_m)


#%% [버핏-8] 백테스트 실행 (엄격 / 현실)
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
    print('\n[현실 버전] 신호 방식 진입 내역 (언제 샀고 언제 팔았나) 처음 60건')
    print(results['real']['entries'].head(60).to_string())


#%% [버핏-9] 성과 비교
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
        ax.set_title(f'버핏 {m} 버전')
        ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print('백테스트 결과가 없습니다. [버핏-8] 의 메시지를 확인하세요.')


#%% [버핏-10] 구간별 성과: 특정 기간을 빼면 결과가 어떻게 달라지는가
# 2025년 이후 시장이 크게 올랐을 때, 그 구간을 빼도 같은 결론인지 본다.
# 일별 가치(NAV)를 구간으로 잘라 1.0 으로 다시 맞춘 뒤 성과를 계산한다. 벤치마크는 코스피 지수가 아니라 전종목 동일가중이다.
PERIODS = [
    ('전체',          None,         None),
    ('2025년 이전',   None,         '2024-12-31'),
    ('2025년 이후',   '2025-01-01', None),
]


def perf_range(nav, start=None, end=None):
    s = nav.loc[start:end]
    if len(s) < 60:
        return None
    s = s / s.iloc[0]
    ret = s.pct_change().dropna()
    years = (s.index[-1] - s.index[0]).days / 365.25
    return dict(CAGR=s.iloc[-1] ** (1 / years) - 1, 누적=s.iloc[-1] - 1, MDD=(s / s.cummax() - 1).min(),
                Sharpe=ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan, 기간=years)


for m, o in results.items():
    rows = []
    for label, a, b in PERIODS:
        pa, pb, pm = (perf_range(o[k], a, b) for k in ('nav_a', 'nav_b', 'nav_m'))
        if pa is None or pm is None:
            continue
        rows.append(dict(구간=label, 기간_년=round(pm['기간'], 1),
                         CAGR_A=pa['CAGR'], CAGR_B=pb['CAGR'], CAGR_벤치=pm['CAGR'],
                         초과_A=pa['CAGR'] - pm['CAGR'], 초과_B=pb['CAGR'] - pm['CAGR'],
                         MDD_A=pa['MDD'], MDD_벤치=pm['MDD'],
                         Sharpe_A=pa['Sharpe'], Sharpe_B=pb['Sharpe'], Sharpe_벤치=pm['Sharpe']))
    print(f'\n===== {m} 버전 구간별 성과 =====')
    print(pd.DataFrame(rows).round(4).to_string(index=False))
# 구간이 짧을수록 우연의 영향이 크다. 위 표는 '그 구간에서의 수치'이고 통계적 검정은 아니다.


#%% [버핏-11] 결과의 한계 자동 점검
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
    if zero_pick > 0:
        warn.append(f'통과 종목이 없어 현금으로 보낸 해가 {zero_pick}번 있다 (그 구간은 전략 성과가 아니라 현금 0%)')
    if n_entry < 20:
        warn.append('신호 방식 진입이 20건 미만이라 표본이 부족하다')
    if INCLUDE_DELISTED:
        warn.append('상장폐지 종목을 넣었다. MarCap 단위(원)는 확인했지만 폐지 종목 종가가 수정주가인지는 확인되지 않았다. '
                    '재무를 받지 못한 폐지 종목(감사의견 거절, 합병 등)은 빠져 있어 생존편향이 일부 남는다')
    else:
        warn.append('생존편향: 상장폐지 종목이 빠져 있어 수익률이 높게 나올 수 있다 (INCLUDE_DELISTED = True 로 완화)')
    warn.append('현재 상장 종목의 과거 시가총액은 현재 주식수 근사, 이익과 자본은 비지배지분을 포함한 연결 값')
    warn.append('설비투자 계정이 없는 회사는 FCF 를 못 구해 빠진다 (금융업 등)')
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
5) 특정 구간(예: 2025년 이후)을 빼도 같은 결론인가 -> [버핏-10]
이 중 하나라도 안 되면 '전략이 통한다'가 아니라 '이 표본에서는 그렇게 보인다' 정도로만 해석한다.
"""


#%% [버핏-12] 민감도: 기준값을 조금 바꿔도 결과가 유지되는가 (현실 버전)
base = dict(ROE_AVG_MIN=ROE_AVG_MIN, MIN_FCF_YIELD=MIN_FCF_YIELD)
rows = []
for roe in [0.10, 0.12, 0.15]:
    for fy in [0.04, 0.05, 0.07]:
        globals().update(ROE_AVG_MIN=roe, MIN_FCF_YIELD=fy)
        o = run_all('real')
        if o is None:
            rows.append((roe, fy, np.nan, np.nan, np.nan, np.nan))
            continue
        pa, pb, pm = perf(o['nav_a']), perf(o['nav_b']), perf(o['nav_m'])
        rows.append((roe, fy, pa['CAGR'], pb['CAGR'], pm['CAGR'], o['log_a']['매수종목수'].mean()))
globals().update(base)

sens = pd.DataFrame(rows, columns=['ROE 평균 하한', 'FCF 수익률 하한', 'CAGR 정기(A)', 'CAGR 신호(B)',
                                   'CAGR 벤치', 'A 평균종목수'])
print(sens.round(4).to_string(index=False))
# 기준값이 조금 바뀔 때 결과가 크게 흔들리면 그 전략은 신뢰하기 어렵다.
# 비교 대상(벤치)은 기준값과 무관하게 같은 값이다. 종목 수가 너무 적게 나온 칸은 해석하지 않는다.