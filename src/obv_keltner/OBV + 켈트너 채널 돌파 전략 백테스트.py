#%% 0. 안내
"""
OBV + 켈트너 채널 돌파 전략 백테스트 (니라지 조시 매매법을 규칙화)

먼저 데이터 수집 파일을 보고 오세요.
  - 한국: '국내 주식 데이터 수집.py' (kor_ticker, kor_price 테이블)
  - 미국: '전세계 주식 데이터 수집.py' (global_ticker, global_price 테이블)
이미 DB에 데이터가 있으면 여기서부터 진행하세요.

실행 순서
  1) 0~2번 셀: 설정, DB 연결, 테이블 확인 (RUN_DOWNLOAD=True 면 가격 수집)
  2) 3~4번 셀: 유니버스 선택, 가격 불러오기
  3) 5~7번 셀: 지표, 신호, 매매 시뮬레이션 함수 정의
  4) 8번 셀: 기본 백테스트
  5) 9번 셀: OBV 있는 버전과 없는 버전 비교
  6) 10번 셀: 파라미터 격자 + 표본 내/외 분리
  7) 11번 셀: 그래프

필요 패키지 (Anaconda Prompt)
  conda activate ta_env
  python -m pip install pandas numpy matplotlib sqlalchemy pymysql tqdm yfinance requests python-dateutil

Spyder 에서 ta_env 쓰기
  Tools > Preferences > Python interpreter > Use the following Python interpreter
  경로: (아나콘다 설치폴더)\envs\ta_env\python.exe  선택 후 콘솔 재시작(Restart kernel)

DB 비밀번호는 코드에 쓰지 않는다. 환경변수 DB_PW 를 쓰거나 실행 시 입력창에 입력한다.
  (윈도우 영구 저장: 명령 프롬프트에서  setx DB_PW "비밀번호"  후 Spyder 재시작)
"""

RUN_DOWNLOAD = False      # True: 가격 수집 후 DB에 저장 / False: 건너뛰고 DB에서 바로 읽기
MARKET = 'KR'             # 'KR' 또는 'US'
N_TICKERS = 2500          # 시가총액 상위 몇 종목으로 테스트할지
START = '2015-01-01'      # 백테스트 시작일


#%% 1. DB 연결 함수
import os
import time
import getpass
import itertools
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from tqdm import tqdm

DB_USER = 'root'
DB_HOST = '127.0.0.1'
DB_NAME = 'stock_db'
DB_PW = os.environ.get('DB_PW') or getpass.getpass('DB password: ')


def db_engine():
    # 읽기용 sqlalchemy 엔진 (비밀번호에 특수문자가 있어도 되도록 quote_plus)
    return create_engine(
        f'mysql+pymysql://{DB_USER}:{quote_plus(DB_PW)}@{DB_HOST}:3306/{DB_NAME}')


def db_con():
    # 쓰기용 pymysql 연결
    return pymysql.connect(user=DB_USER, passwd=DB_PW, host=DB_HOST,
                           db=DB_NAME, charset='utf8')


#%% 2. 테이블 확인 + (선택) 가격 수집
REQ = {'KR': ['kor_ticker', 'kor_price'],
       'US': ['global_ticker', 'global_price']}


def missing_tables(names):
    engine = db_engine()
    have = pd.read_sql(
        f"select table_name from information_schema.tables where table_schema = '{DB_NAME}'",
        con=engine).iloc[:, 0].str.lower().tolist()
    engine.dispose()
    return [n for n in names if n.lower() not in have]


ticker_tbl, price_tbl = REQ[MARKET]

# 티커 테이블은 수집 파일에서 만든다. 이 파일에서는 못 만든다.
miss = missing_tables([ticker_tbl])
if miss:
    raise RuntimeError(f'{miss} 테이블이 DB에 없습니다. 먼저 데이터 수집 파일에서 티커를 수집하세요.')

if RUN_DOWNLOAD:
    if MARKET == 'KR':
        # 네이버 차트 크롤링 (국내 주식 데이터 수집.py 와 같은 방식, 상위 N_TICKERS 종목만)
        # 사이트 구조가 지금도 같은지는 확인하지 못했다. 안 되면 수집 파일의 해당 셀을 쓸 것.
        import requests as rq
        from io import BytesIO
        from datetime import date
        from dateutil.relativedelta import relativedelta

        con = db_con()
        mycursor = con.cursor()
        mycursor.execute("""
            create table if not exists kor_price
            (
                날짜 date, 시가 double, 고가 double, 저가 double,
                종가 double, 거래량 double, 종목코드 varchar(6),
                primary key(날짜, 종목코드)
            )""")
        con.commit()

        engine = db_engine()
        try:
            tl = pd.read_sql(f"""
                select 종목코드 from kor_ticker
                where 기준일 = (select max(기준일) from kor_ticker) and 종목구분 = '보통주'
                order by 시가총액 desc limit {N_TICKERS}""", con=engine)
        except Exception:
            # 시가총액 열 이름이 다르면 정렬 없이 앞에서부터 가져온다.
            tl = pd.read_sql(f"""
                select 종목코드 from kor_ticker
                where 기준일 = (select max(기준일) from kor_ticker) and 종목구분 = '보통주'
                limit {N_TICKERS}""", con=engine)
        engine.dispose()

        query = """
            insert into kor_price (날짜, 시가, 고가, 저가, 종가, 거래량, 종목코드)
            values (%s,%s,%s,%s,%s,%s,%s) as new
            on duplicate key update
            시가=new.시가, 고가=new.고가, 저가=new.저가, 종가=new.종가, 거래량=new.거래량
        """
        fr = (date.today() + relativedelta(years=-10)).strftime('%Y%m%d')
        to = date.today().strftime('%Y%m%d')
        error_list = []

        for ticker in tqdm(tl['종목코드'].tolist()):
            try:
                url = (f'https://fchart.stock.naver.com/siseJson.nhn?symbol={ticker}'
                       f'&requestType=1&startTime={fr}&endTime={to}&timeframe=day')
                data = rq.get(url).content
                price = pd.read_csv(BytesIO(data)).iloc[:, 0:6]
                price.columns = ['날짜', '시가', '고가', '저가', '종가', '거래량']
                price = price.dropna()
                price['날짜'] = price['날짜'].str.extract(r'(\d+)')
                price['날짜'] = pd.to_datetime(price['날짜'])
                price['종목코드'] = ticker
                price = price.astype(object).where(price.notna(), None)
                mycursor.executemany(query, price.values.tolist())
                con.commit()
            except Exception:
                print(ticker)
                error_list.append(ticker)
            time.sleep(2)
        con.close()

    else:
        # 미국: yfinance (전세계 주식 데이터 수집.py 와 같은 방식)
        import yfinance as yf

        con = db_con()
        mycursor = con.cursor()
        mycursor.execute("""
            create table if not exists global_price
            (
                Date date, High double, Low double, Open double, Close double,
                Volume double, `Adj Close` double, ticker varchar(20),
                primary key (Date, ticker)
            )""")
        con.commit()

        engine = db_engine()
        tl = pd.read_sql(f"""
            select Symbol from global_ticker
            where date = (select max(date) from global_ticker) and country = 'United States'
            order by `Market Cap` desc limit {N_TICKERS}""", con=engine)
        engine.dispose()

        query = """
            insert into global_price (Date, High, Low, Open, Close, Volume, `Adj Close`, ticker)
            values (%s,%s,%s,%s,%s,%s,%s,%s) as new
            on duplicate key update
            High=new.High, Low=new.Low, Open=new.Open, Close=new.Close,
            Volume=new.Volume, `Adj Close`=new.`Adj Close`
        """
        error_list = []
        for ticker in tqdm(tl['Symbol'].tolist()):
            try:
                data = yf.download(ticker, start=START, progress=False, auto_adjust=False)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data = data.reset_index()
                data = data[['Date', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']]
                data['ticker'] = ticker
                data = data.dropna()
                data = data.astype(object).where(data.notna(), None)
                mycursor.executemany(query, data.values.tolist())
                con.commit()
            except Exception:
                print(ticker)
                error_list.append(ticker)
            time.sleep(1)
        con.close()

else:
    # 플래그가 False 일 때 가격 테이블 확인
    miss = missing_tables([price_tbl])
    if miss:
        raise RuntimeError(f'{miss} 테이블이 DB에 없습니다. RUN_DOWNLOAD = True 로 바꾸고 다시 실행하세요.')
    print('필요한 테이블이 모두 있습니다. DB에서 바로 읽습니다.')


#%% 3. 유니버스 선택 (시가총액 상위)
def get_universe(n):
    engine = db_engine()
    if MARKET == 'KR':
        q = f"""
            select 종목코드 as tk from kor_ticker
            where 기준일 = (select max(기준일) from kor_ticker) and 종목구분 = '보통주'
            order by 시가총액 desc limit {n}"""
    else:
        q = f"""
            select Symbol as tk from global_ticker
            where date = (select max(date) from global_ticker) and country = 'United States'
            order by `Market Cap` desc limit {n}"""
    out = pd.read_sql(q, con=engine)['tk'].tolist()
    engine.dispose()
    return out


universe = get_universe(N_TICKERS)
print(len(universe), universe[:5])


#%% 4. 가격 불러오기 -> {티커: DataFrame(open, high, low, close, volume)}
def load_prices(tickers, start=START):
    engine = db_engine()
    tk = ",".join(f"'{t}'" for t in tickers)
    if MARKET == 'KR':
        q = f"""
            select 날짜 as `date`, 시가 as `open`, 고가 as high, 저가 as low,
                   종가 as `close`, 거래량 as volume, 종목코드 as ticker
            from kor_price
            where 종목코드 in ({tk}) and 날짜 >= '{start}'"""
    else:
        q = f"""
            select Date as `date`, Open as `open`, High as high, Low as low,
                   Close as `close`, Volume as volume, `Adj Close` as adj, ticker
            from global_price
            where ticker in ({tk}) and Date >= '{start}'"""
    df = pd.read_sql(q, con=engine)
    engine.dispose()

    df['date'] = pd.to_datetime(df['date'])
    if MARKET == 'US':
        # 배당까지 반영한 수정주가 비율을 시고저종에 같이 곱한다.
        f = df['adj'] / df['close']
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] * f

    price_dict = {}
    for t, g in df.groupby('ticker'):
        g = g.set_index('date').sort_index()[['open', 'high', 'low', 'close', 'volume']].dropna()
        g = g[g['volume'] > 0]
        if len(g) >= 250:                  # 데이터가 너무 짧은 종목은 제외
            price_dict[t] = g
    return price_dict


price_dict = load_prices(universe)
print(len(price_dict), '종목 로드')


#%% 5. 파라미터
P = dict(
    # 켈트너 채널: 중심선 EMA, 상하단은 ATR 배수
    KC_EMA=20, KC_ATR=10, KC_MULT=2.0,
    # 박스: 직전 BOX_N 봉 고저폭이 ATR 의 BOX_ATR_MAX 배 이하일 때 에너지 응축으로 본다. (가정값)
    BOX_N=20, BOX_ATR_MAX=6.0,
    # 돌파 캔들: 몸통이 ATR 의 BODY_ATR 배 이상, 거래량이 평균의 VOL_MULT 배 이상 (가정값)
    BODY_ATR=0.8, VOL_N=20, VOL_MULT=1.2,
    # OBV: 'candle' 은 영상처럼 양봉이면 더하고 음봉이면 뺀다. 'close' 는 전일 종가 대비.
    OBV_MODE='candle',
    USE_OBV=True,            # 진입 필터에 OBV 조건을 쓸지
    USE_OBV_STOP=True,       # OBV 기반 청산(전저점 이탈, 켈트너 이탈+OBV 확인, 다이버전스)을 쓸지
    USE_KC_EXIT=True,        # 켈트너 이탈 청산을 쓸지
    KC_EXIT_LINE='mid',      # OBV 확인이 붙을 때 기준선: 'mid' 중심선 / 'lo' 하단선
    OBV_NEAR=0.1,            # 전저점 수준 판정: OBV 박스 범위의 하위 10% 이내면 저점 부근으로 본다
    USE_DIV_EXIT=True,       # 하락 다이버전스 익절을 쓸지
    DIV_N=20,                # 다이버전스 비교 구간(봉 수)
    ALLOW_SHORT=False,       # 숏 허용 (한국 개별종목은 공매도 제약이 있어 기본 False)
    # 손절: 박스 반대편과 (진입가 -+ STOP_ATR_CAP * ATR) 중 가까운 쪽
    STOP_ATR_CAP=3.0,
    # 분할 익절: R 배수와 비중 (비중 합은 1). 목표가 거리는 영상에 없어 1R/2R/3R 로 가정
    TARGET_R=(1.0, 2.0, 3.0), TARGET_W=(0.5, 0.25, 0.25),
    BE_AFTER=1,              # 몇 번째 목표 도달 후 손절선을 본전으로 올릴지 (99면 사용 안 함)
    MAX_HOLD=60,             # 최대 보유 봉 수
    COST=0.0015,             # 편도 비용(수수료+세금+슬리피지 가정). 직접 조정할 것
)


#%% 6. 지표와 신호
def add_signals(df, P):
    d = df.copy()
    pc = d['close'].shift(1)

    # ATR (Wilder 방식 지수평균)
    tr = pd.concat([d['high'] - d['low'],
                    (d['high'] - pc).abs(),
                    (d['low'] - pc).abs()], axis=1).max(axis=1)
    d['atr'] = tr.ewm(alpha=1 / P['KC_ATR'], adjust=False).mean()

    # 켈트너 채널
    d['kc_mid'] = d['close'].ewm(span=P['KC_EMA'], adjust=False).mean()
    d['kc_up'] = d['kc_mid'] + P['KC_MULT'] * d['atr']
    d['kc_lo'] = d['kc_mid'] - P['KC_MULT'] * d['atr']

    # OBV
    if P['OBV_MODE'] == 'candle':
        sgn = np.sign(d['close'] - d['open'])
    else:
        sgn = np.sign(d['close'].diff())
    d['obv'] = (sgn * d['volume']).cumsum()

    # 박스 (오늘을 제외한 직전 BOX_N 봉)
    n = P['BOX_N']
    h = n // 2
    d['box_hi'] = d['high'].rolling(n).max().shift(1)
    d['box_lo'] = d['low'].rolling(n).min().shift(1)
    d['box_w'] = (d['box_hi'] - d['box_lo']) / d['atr'].shift(1)

    # OBV 의 박스 구간 고저
    d['obv_hi'] = d['obv'].rolling(n).max().shift(1)
    d['obv_lo'] = d['obv'].rolling(n).min().shift(1)

    # OBV 가 전저점/전고점 "부근"인지 (켈트너 이탈 청산의 확인 조건)
    rng = d['obv_hi'] - d['obv_lo']
    d['obv_near_lo'] = d['obv_lo'] + P['OBV_NEAR'] * rng
    d['obv_near_hi'] = d['obv_hi'] - P['OBV_NEAR'] * rng

    # 다이버전스: 가격은 고점을 높였는데 OBV 고점은 낮아짐 (하락), 반대는 상승
    dn = P['DIV_N']
    p_hi = d['high'].rolling(dn).max()
    p_hi_prev = d['high'].shift(dn).rolling(dn).max()
    o_hi = d['obv'].rolling(dn).max()
    o_hi_prev = d['obv'].shift(dn).rolling(dn).max()
    p_lo = d['low'].rolling(dn).min()
    p_lo_prev = d['low'].shift(dn).rolling(dn).min()
    o_lo = d['obv'].rolling(dn).min()
    o_lo_prev = d['obv'].shift(dn).rolling(dn).min()
    d['div_bear'] = ((p_hi > p_hi_prev) & (o_hi < o_hi_prev)).fillna(False)
    d['div_bull'] = ((p_lo < p_lo_prev) & (o_lo > o_lo_prev)).fillna(False)

    # OBV 매집 확인: 박스 후반 OBV 의 저점과 고점이 전반보다 낮아지지 않았는가
    rec_min = d['obv'].rolling(h).min().shift(1)
    ear_min = d['obv'].shift(h + 1).rolling(h).min()
    rec_max = d['obv'].rolling(h).max().shift(1)
    ear_max = d['obv'].shift(h + 1).rolling(h).max()
    acc_long = (rec_min >= ear_min) & (rec_max >= ear_max)
    acc_short = (rec_min <= ear_min) & (rec_max <= ear_max)

    d['vol_ma'] = d['volume'].rolling(P['VOL_N']).mean().shift(1)
    body = d['close'] - d['open']
    squeeze = d['box_w'] <= P['BOX_ATR_MAX']
    vol_ok = d['volume'] >= P['VOL_MULT'] * d['vol_ma']

    sig_long = (squeeze & vol_ok
                & (d['close'] > d['box_hi']) & (d['close'] > d['kc_up'])
                & (body >= P['BODY_ATR'] * d['atr']))
    sig_short = (squeeze & vol_ok
                 & (d['close'] < d['box_lo']) & (d['close'] < d['kc_lo'])
                 & (-body >= P['BODY_ATR'] * d['atr']))

    if P['USE_OBV']:
        sig_long = sig_long & acc_long & (d['obv'] > d['obv_hi'])
        sig_short = sig_short & acc_short & (d['obv'] < d['obv_lo'])

    d['sig_long'] = sig_long.fillna(False)
    d['sig_short'] = sig_short.fillna(False)
    return d


#%% 7. 매매 시뮬레이션 (신호 다음 봉 시가 진입)
def simulate(d, P, ticker):
    o = d['open'].values
    h = d['high'].values
    l = d['low'].values
    c = d['close'].values
    atr = d['atr'].values
    kc_mid = d['kc_mid'].values
    kc_up = d['kc_up'].values
    kc_lo = d['kc_lo'].values
    obv = d['obv'].values
    box_hi = d['box_hi'].values
    box_lo = d['box_lo'].values
    obv_hi = d['obv_hi'].values
    obv_lo = d['obv_lo'].values
    near_lo = d['obv_near_lo'].values
    near_hi = d['obv_near_hi'].values
    div_bear = d['div_bear'].values
    div_bull = d['div_bull'].values
    idx = d.index
    n = len(d)

    sl = d['sig_long'].values
    ss = d['sig_short'].values if P['ALLOW_SHORT'] else np.zeros(n, dtype=bool)
    T = P['TARGET_R']
    W = P['TARGET_W']
    assert abs(sum(W) - 1) < 1e-9 and len(T) == len(W)

    # 켈트너 이탈 기준선: OBV 확인이 있으면 설정값, OBV 를 아예 안 쓰는 버전은 하단선(무조건 청산)
    kc_line = P['KC_EXIT_LINE'] if P['USE_OBV_STOP'] else 'lo'

    trades = []
    last_exit = -1

    for i in np.flatnonzero(sl | ss):
        if i < last_exit or i >= n - 1:      # 보유 중이거나 마지막 봉이면 건너뜀
            continue

        dr = 1 if sl[i] else -1              # 1: 롱, -1: 숏
        entry = o[i + 1]

        if dr == 1:
            stop = max(box_lo[i], entry - P['STOP_ATR_CAP'] * atr[i])
            obv_stop = obv_lo[i]
        else:
            stop = min(box_hi[i], entry + P['STOP_ATR_CAP'] * atr[i])
            obv_stop = obv_hi[i]

        risk = dr * (entry - stop)           # 1R (가격 단위)
        if not risk > 0:                     # 갭으로 이미 손절선 밖이면 진입 안 함
            continue

        targets = [entry + dr * r * risk for r in T]
        stop_cur = stop
        k = 0                                # 도달한 목표 개수
        rem = 1.0                            # 남은 비중
        fills = []                           # (비중, 체결가)
        jend = min(n - 1, i + 1 + P['MAX_HOLD'])
        exit_idx, reason = jend, 'time'

        for j in range(i + 1, jend + 1):
            adv = l[j] if dr == 1 else h[j]  # 불리한 방향 극값
            fav = h[j] if dr == 1 else l[j]  # 유리한 방향 극값

            # 1) 손절 먼저 확인 (같은 봉에서 겹치면 손절 우선), 갭이면 시가 체결
            if dr * o[j] <= dr * stop_cur:
                fills.append((rem, o[j]))
                exit_idx, reason, rem = j, ('stop' if stop_cur == stop else 'be'), 0
                break
            if dr * adv <= dr * stop_cur:
                fills.append((rem, stop_cur))
                exit_idx, reason, rem = j, ('stop' if stop_cur == stop else 'be'), 0
                break

            # 2) 목표가 확인 (갭으로 넘으면 시가 체결)
            while k < len(T) and dr * fav >= dr * targets[k]:
                px = targets[k] if dr * o[j] < dr * targets[k] else o[j]
                fills.append((W[k], px))
                rem -= W[k]
                k += 1
                if k >= P['BE_AFTER'] and dr * entry > dr * stop_cur:
                    stop_cur = entry         # 이후 본전 청산
            if k == len(T):
                exit_idx, reason, rem = j, 'target', 0
                break

            # 3) 마지막 봉은 종가 청산
            if j == jend:
                fills.append((rem, c[j]))
                exit_idx = j
                reason = 'time' if j - (i + 1) >= P['MAX_HOLD'] else 'end'
                rem = 0
                break

            # 4) 종가 기준 이탈 신호 -> 다음 봉 시가 청산
            #    - 켈트너 이탈은 OBV 가 전저점(숏은 전고점) 부근일 때만 청산 (영상: 이탈만으로는 보유)
            #    - OBV 가 진입 당시 박스의 저점(고점)을 이탈하면 청산
            #    - 하락(상승) 다이버전스가 나오면 익절
            why = None
            if dr == 1:
                line = kc_mid[j] if kc_line == 'mid' else kc_lo[j]
                if P['USE_KC_EXIT'] and c[j] < line and (not P['USE_OBV_STOP'] or obv[j] <= near_lo[j]):
                    why = 'kc'
                elif P['USE_OBV_STOP'] and obv[j] < obv_stop:
                    why = 'obv'
                elif P['USE_OBV_STOP'] and P['USE_DIV_EXIT'] and div_bear[j]:
                    why = 'div'
            else:
                line = kc_mid[j] if kc_line == 'mid' else kc_up[j]
                if P['USE_KC_EXIT'] and c[j] > line and (not P['USE_OBV_STOP'] or obv[j] >= near_hi[j]):
                    why = 'kc'
                elif P['USE_OBV_STOP'] and obv[j] > obv_stop:
                    why = 'obv'
                elif P['USE_OBV_STOP'] and P['USE_DIV_EXIT'] and div_bull[j]:
                    why = 'div'
            if why:
                fills.append((rem, o[j + 1]))
                exit_idx, reason, rem = j + 1, why, 0
                break

        ret = sum(w * dr * (px / entry - 1) for w, px in fills) - 2 * P['COST']
        R = ret / (risk / entry)             # 손절폭 대비 수익 (R 배수)
        trades.append(dict(ticker=ticker, dir=dr,
                           signal_date=idx[i], entry_date=idx[i + 1], exit_date=idx[exit_idx],
                           entry=entry, stop=stop, ret=ret, R=R,
                           hold=exit_idx - (i + 1), reason=reason))
        last_exit = exit_idx

    return pd.DataFrame(trades)


def run_backtest(price_dict, P):
    out = []
    for t, df in price_dict.items():
        d = add_signals(df, P)
        tr = simulate(d, P, t)
        if len(tr):
            out.append(tr)
    if not out:
        return pd.DataFrame(columns=['ticker', 'dir', 'signal_date', 'entry_date', 'exit_date',
                                     'entry', 'stop', 'ret', 'R', 'hold', 'reason'])
    return pd.concat(out, ignore_index=True)


def perf_stats(tr, risk_per_trade=0.01):
    # 거래별 R 로 성과를 요약한다. 자본 제약과 포지션 겹침은 반영하지 않는다.
    if len(tr) == 0:
        return dict(n=0, win=np.nan, avg_R=np.nan, PF=np.nan, t=np.nan, mdd=np.nan, hold=np.nan)
    win = tr['R'] > 0
    loss_sum = abs(tr.loc[~win, 'R'].sum())
    pf = tr.loc[win, 'R'].sum() / loss_sum if loss_sum > 0 else np.nan
    t = tr.sort_values('exit_date')
    eq = (1 + risk_per_trade * t['R']).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    tstat = tr['R'].mean() / tr['R'].std() * np.sqrt(len(tr)) if len(tr) > 1 else np.nan
    return dict(n=len(tr), win=win.mean(), avg_R=tr['R'].mean(), PF=pf,
                t=tstat, mdd=mdd, hold=tr['hold'].mean())
#%% 8. 기본 백테스트
trades = run_backtest(price_dict, P)
stats = perf_stats(trades)
print(pd.Series(stats))

print('\n[청산 사유별]')
print(trades.groupby('reason')['R'].agg(['count', 'mean', 'sum']))

print('\n[연도별]')
print(trades.groupby(trades['exit_date'].dt.year)['R'].agg(['count', 'mean', 'sum']))


#%% 9. OBV 가 실제로 도움이 되는가 (같은 종목, 같은 기간에서 비교)
variants = {
    '켈트너+박스만 (OBV 없음)': dict(USE_OBV=False, USE_OBV_STOP=False),
    '진입 OBV 필터만':          dict(USE_OBV=True,  USE_OBV_STOP=False),
    '청산 OBV 이탈만':          dict(USE_OBV=False, USE_OBV_STOP=True),
    '영상 전체 (OBV 진입+청산)': dict(USE_OBV=True,  USE_OBV_STOP=True),
}
rows = {}
for name, kw in variants.items():
    tr_v = run_backtest(price_dict, {**P, **kw})
    rows[name] = perf_stats(tr_v)
print(pd.DataFrame(rows).T)


#%% 10. 파라미터 격자 + 표본 내/외 분리
# 표본 내(SPLIT_DATE 이전)와 표본 외 결과를 나란히 본다.
# 표본 내에서 좋았던 조합이 표본 외에서도 유지되는지가 핵심이다.
SPLIT_DATE = '2023-01-01'
grid = {'BOX_ATR_MAX': [4, 6, 8],
        'BODY_ATR': [0.5, 0.8, 1.2],
        'KC_MULT': [1.5, 2.0]}

rows = []
for vals in itertools.product(*grid.values()):
    params = dict(zip(grid.keys(), vals))
    tr_g = run_backtest(price_dict, {**P, **params})
    s_in = perf_stats(tr_g[tr_g['entry_date'] < SPLIT_DATE])
    s_out = perf_stats(tr_g[tr_g['entry_date'] >= SPLIT_DATE])
    rows.append({**params,
                 'n_in': s_in['n'], 'avgR_in': s_in['avg_R'], 'PF_in': s_in['PF'],
                 'n_out': s_out['n'], 'avgR_out': s_out['avg_R'], 'PF_out': s_out['PF']})

grid_df = pd.DataFrame(rows).sort_values('avgR_in', ascending=False)
print(grid_df.round(3).to_string(index=False))


#%% 11. 그래프
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

t = trades.sort_values('exit_date').reset_index(drop=True)
fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))

ax[0].plot(t['exit_date'], t['R'].cumsum())
ax[0].set_title('누적 R (거래 종료일 기준)')
ax[0].axhline(0, color='gray', lw=0.5)

ax[1].hist(t['R'].clip(-3, 6), bins=40)
ax[1].set_title('거래별 R 분포 (-3 ~ 6 로 자름)')
ax[1].axvline(0, color='gray', lw=0.5)

eq = (1 + 0.01 * t['R']).cumprod()
ax[2].plot(t['exit_date'], eq)
ax[2].set_title('거래당 위험 1% 복리 곡선 (겹침 무시)')

plt.tight_layout()
plt.show()


#%% 12. 파워존(Pine 코드) 파라미터와 신호
# 원본 Pine 의 입력값 이름을 그대로 가져온다.
Q = dict(
    PERIODS=5,            # periods: 직전 음봉 뒤에 이어지는 연속 양봉(음봉) 개수
    THRESHOLD=0.0,        # threshold: 최소 이동률(%)
    USEWICKS=False,       # usewicks: True 면 존의 경계를 꼬리 포함 고저로 잡는다
    TP_FACTOR=1.5,        # tp_factor
    SL_FACTOR=1.0,        # sl_factor
    ALLOW_SHORT=True,     # 원본은 숏도 같이 진입한다. 롱 전용으로 보려면 False
    PZ_OBV=False,         # 추가 실험: 진입 시 OBV 가 존 형성 전보다 높아야(숏은 낮아야) 진입
    PZ_KC=False,          # 추가 실험: 켈트너 상단(숏은 하단) 돌파도 함께 요구
    MAX_HOLD=None,        # 원본은 시간 청산이 없다. 숫자를 넣으면 그 봉 수에서 종가 청산
    COST=P['COST'],
)


def add_pz_signals(d, Q):
    n = Q['PERIODS']
    ob = n + 1                                  # ob_period = periods + 1

    o_ob = d['open'].shift(ob)
    c_ob = d['close'].shift(ob)
    h_ob = d['high'].shift(ob)
    l_ob = d['low'].shift(ob)

    # 직전 n 개 봉(t-n ~ t-1)이 모두 양봉(음봉)인가
    up = (d['close'] > d['open']).astype(float)
    dn = (d['close'] < d['open']).astype(float)
    up_run = up.shift(1).rolling(n).sum() == n
    dn_run = dn.shift(1).rolling(n).sum() == n

    # absmove: 존 형성 봉 종가 대비 직전 봉 종가까지의 이동률
    relmove = ((c_ob - d['close'].shift(1)).abs() / c_ob * 100) >= Q['THRESHOLD']

    # 상승 파워존: ob 봉이 음봉, 그 뒤 n 개가 모두 양봉
    pz_bull = (c_ob < o_ob) & up_run & relmove
    d['bull_hi'] = h_ob if Q['USEWICKS'] else o_ob
    d['bull_lo'] = l_ob

    # 하락 파워존: ob 봉이 양봉, 그 뒤 n 개가 모두 음봉
    pz_bear = (c_ob > o_ob) & dn_run & relmove
    d['bear_hi'] = h_ob
    d['bear_lo'] = l_ob if Q['USEWICKS'] else o_ob

    # 진입 신호: 종가가 존 상단을 넘으면 롱, 존 하단 밑이면 숏
    sig_l = pz_bull & (d['close'] > d['bull_hi'])
    sig_s = pz_bear & (d['close'] < d['bear_lo'])

    if Q['PZ_OBV']:
        sig_l = sig_l & (d['obv'] > d['obv'].shift(ob))
        sig_s = sig_s & (d['obv'] < d['obv'].shift(ob))
    if Q['PZ_KC']:
        sig_l = sig_l & (d['close'] > d['kc_up'])
        sig_s = sig_s & (d['close'] < d['kc_lo'])

    d['pz_long'] = sig_l.fillna(False)
    d['pz_short'] = sig_s.fillna(False)
    return d


def simulate_pz(d, Q, ticker):
    o = d['open'].values
    h = d['high'].values
    l = d['low'].values
    c = d['close'].values
    bull_hi = d['bull_hi'].values
    bull_lo = d['bull_lo'].values
    bear_hi = d['bear_hi'].values
    bear_lo = d['bear_lo'].values
    idx = d.index
    n = len(d)

    sl_sig = d['pz_long'].values
    ss_sig = d['pz_short'].values if Q['ALLOW_SHORT'] else np.zeros(n, dtype=bool)

    trades = []
    last_exit = -1

    for i in np.flatnonzero(sl_sig | ss_sig):
        if i < last_exit or i >= n - 1:         # 포지션 보유 중(원본의 position_size == 0 조건)
            continue

        dr = 1 if sl_sig[i] else -1
        if dr == 1:
            rng = bull_hi[i] - bull_lo[i]
            tp = c[i] + rng * Q['TP_FACTOR']                # 원본: 신호 봉 종가 기준
            sl = bull_lo[i] - rng * Q['SL_FACTOR']
        else:
            rng = bear_hi[i] - bear_lo[i]
            tp = c[i] - rng * Q['TP_FACTOR']
            sl = bear_hi[i] + rng * Q['SL_FACTOR']

        entry = o[i + 1]                         # strategy.entry 는 다음 봉 시가에 체결된다고 본다
        risk = dr * (entry - sl)
        if not risk > 0:                         # 시가가 이미 손절선 밖이면 진입 안 함
            continue

        jend = n - 1 if Q['MAX_HOLD'] is None else min(n - 1, i + 1 + Q['MAX_HOLD'])
        exit_idx, px, reason = jend, c[jend], 'end'

        for j in range(i + 1, jend + 1):
            adv = l[j] if dr == 1 else h[j]
            fav = h[j] if dr == 1 else l[j]

            # 같은 봉에서 손절과 목표가 겹치면 손절 우선 (보수적), 갭이면 시가 체결
            if dr * o[j] <= dr * sl:
                exit_idx, px, reason = j, o[j], 'sl'
                break
            if dr * adv <= dr * sl:
                exit_idx, px, reason = j, sl, 'sl'
                break
            if dr * fav >= dr * tp:
                exit_idx, px, reason = j, (tp if dr * o[j] < dr * tp else o[j]), 'tp'
                break
            if j == jend:
                exit_idx, px = j, c[j]
                reason = 'time' if (Q['MAX_HOLD'] is not None and j - (i + 1) >= Q['MAX_HOLD']) else 'end'

        ret = dr * (px / entry - 1) - 2 * Q['COST']
        R = ret / (risk / entry)
        rr = dr * (tp - entry) / risk            # 진입 시점의 보상 대 위험 비율
        trades.append(dict(ticker=ticker, dir=dr,
                           signal_date=idx[i], entry_date=idx[i + 1], exit_date=idx[exit_idx],
                           entry=entry, stop=sl, ret=ret, R=R, rr=rr,
                           hold=exit_idx - (i + 1), reason=reason))
        last_exit = exit_idx

    return pd.DataFrame(trades)


def run_backtest_pz(price_dict, Q):
    out = []
    for t, df in price_dict.items():
        d = add_signals(df, P)                   # obv, kc_up, kc_lo 컬럼을 쓰려고 재사용 (신호는 안 씀)
        d = add_pz_signals(d, Q)
        tr = simulate_pz(d, Q, t)
        if len(tr):
            out.append(tr)
    if not out:
        return pd.DataFrame(columns=['ticker', 'dir', 'signal_date', 'entry_date', 'exit_date',
                                     'entry', 'stop', 'ret', 'R', 'rr', 'hold', 'reason'])
    return pd.concat(out, ignore_index=True)


#%% 13. 파워존 기본 결과 (Pine 기본값 그대로)
trades_pz = run_backtest_pz(price_dict, Q)
print(pd.Series(perf_stats(trades_pz)))

# 이 전략의 구조: 보상/위험 비율과 그에 따른 손익분기 승률
rr_med = trades_pz['rr'].median()
print('\n보상/위험 중앙값: %.2f  -> 손익분기 승률(비용 제외) 약 %.1f%%' % (rr_med, 100 / (1 + rr_med)))

print('\n[청산 사유별]')
print(trades_pz.groupby('reason')['R'].agg(['count', 'mean', 'sum']))

print('\n[롱/숏별]  (1: 롱, -1: 숏)')
print(trades_pz.groupby('dir')['R'].agg(['count', 'mean', 'sum']))

print('\n[연도별]')
print(trades_pz.groupby(trades_pz['exit_date'].dt.year)['R'].agg(['count', 'mean', 'sum']))


#%% 14. 파워존 변형 비교 + 파라미터 격자(표본 내/외)
# (1) OBV, 켈트너를 붙이면 좋아지는지: 영상의 주장을 이 지표에 적용해 본다
variants = {
    '원본 (롱+숏)':          dict(),
    '원본 롱만':             dict(ALLOW_SHORT=False),
    '+OBV 필터 (롱+숏)':     dict(PZ_OBV=True),
    '+켈트너 필터 (롱+숏)':  dict(PZ_KC=True),
    '+OBV +켈트너 (롱+숏)':  dict(PZ_OBV=True, PZ_KC=True),
    '+OBV +켈트너 롱만':     dict(PZ_OBV=True, PZ_KC=True, ALLOW_SHORT=False),
}
rows = {}
for name, kw in variants.items():
    rows[name] = perf_stats(run_backtest_pz(price_dict, {**Q, **kw}))
print(pd.DataFrame(rows).T)

# (2) 파라미터 격자: 표본 내(SPLIT_DATE 이전)와 표본 외를 나란히 본다
SPLIT_DATE = '2023-01-01'
grid = {'PERIODS': [3, 5, 7],
        'TP_FACTOR': [1.0, 1.5, 2.5],
        'SL_FACTOR': [0.5, 1.0]}

rows = []
for vals in itertools.product(*grid.values()):
    params = dict(zip(grid.keys(), vals))
    tr_g = run_backtest_pz(price_dict, {**Q, **params})
    s_in = perf_stats(tr_g[tr_g['entry_date'] < SPLIT_DATE])
    s_out = perf_stats(tr_g[tr_g['entry_date'] >= SPLIT_DATE])
    rows.append({**params,
                 'n_in': s_in['n'], 'avgR_in': s_in['avg_R'], 'PF_in': s_in['PF'],
                 'n_out': s_out['n'], 'avgR_out': s_out['avg_R'], 'PF_out': s_out['PF']})

grid_pz = pd.DataFrame(rows).sort_values('avgR_in', ascending=False)
print(grid_pz.round(3).to_string(index=False))

#%% [확인] kor_price 에 들어있는 종목 수와 기간
engine = db_engine()
print(pd.read_sql("""
    select count(distinct 종목코드) as n_tk, min(날짜) as first_day, max(날짜) as last_day
    from kor_price""", con=engine))
engine.dispose()


#%% 15. 시가총액 구간별 비교 (N_TICKERS 를 크게 늘린 뒤 실행)
# universe 는 시가총액 내림차순이므로 앞에서부터 잘라 구간을 나눈다.
tiers = {'대형 1~200': universe[:200],
         '중형 201~600': universe[200:600],
         '소형 601~': universe[600:]}

variants_t = {'OBV 없음': dict(USE_OBV=False, USE_OBV_STOP=False),
              '영상 전체': dict()}

rows = {}
for name, tks in tiers.items():
    sub = {t: price_dict[t] for t in tks if t in price_dict}
    if not sub:
        continue
    for label, kw in variants_t.items():
        rows[(name, label)] = perf_stats(run_backtest(sub, {**P, **kw}))

print(pd.DataFrame(rows).T)