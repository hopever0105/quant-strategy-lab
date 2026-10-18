#%%
import pandas as pd
from sqlalchemy import create_engine
# 비밀번호는 코드에 직접 쓰지 말고 환경변수로 받는 것을 권합니다
import os
engine = create_engine(f"mysql+pymysql://root:{os.environ['DB_PW']}@127.0.0.1/stock_db")
tables = pd.read_sql("SHOW TABLES", engine)
print(tables)
for t in tables.iloc[:, 0]:
    print(t)
    print(pd.read_sql(f"SHOW COLUMNS FROM `{t}`", engine)[['Field', 'Type']])
    
    #%%
import os
import pandas as pd
from sqlalchemy import create_engine

# 비밀번호는 코드에 직접 쓰지 말고 환경변수로 받습니다
engine = create_engine(f"mysql+pymysql://root:{os.environ['DB_PW']}@127.0.0.1/stock_db")

# 1) kor_ticker가 시점별 이력인지, 한 시점뿐인지
q1 = """
SELECT COUNT(DISTINCT 기준일) AS n_dates, MIN(기준일) AS first_d, MAX(기준일) AS last_d
FROM kor_ticker
"""
print(pd.read_sql(q1, engine))

# 2) kor_price 기간과 종목 수
q2 = """
SELECT MIN(날짜) AS first_d, MAX(날짜) AS last_d, COUNT(DISTINCT 종목코드) AS n_tickers
FROM kor_price
"""
print(pd.read_sql(q2, engine))

# 3) kor_value 지표 종류와 kor_sector 업종 종류
print(pd.read_sql("SELECT DISTINCT 지표 FROM kor_value", engine))
print(pd.read_sql("SELECT DISTINCT IDX_CD, SEC_NM_KOR FROM kor_sector", engine))

# 4) 에너지·나스닥·환율 후보가 있는지 (fdr_us_price, global_price, tv_us_price)
syms = "('USO','UNG','CL=F','NG=F','QQQ','SOXX','SMH','^IXIC','KRW=X','XLE')"
for t, col in [('fdr_us_price', 'Symbol'), ('global_price', 'ticker'), ('tv_us_price', 'Symbol')]:
    q = f"SELECT {col} AS sym, MIN(Date) AS first_d, MAX(Date) AS last_d FROM {t} WHERE {col} IN {syms} GROUP BY {col}"
    print(t)
    print(pd.read_sql(q, engine))

# 5) 이름으로 찾기 (% 문자는 %%로 씁니다)
q5 = """
SELECT Symbol, Name FROM global_ticker
WHERE Name LIKE '%%Oil%%' OR Name LIKE '%%Natural Gas%%' OR Name LIKE '%%Nasdaq%%'
"""
print(pd.read_sql(q5, engine))

#%% 지수 규칙 초안 (가칭: AI·반도체 에너지 페어 지수)
"""
[출처 표기] KEDI = KEDI 한국AI소프트웨어 방법론, WISE = WISE 삼성그룹밸류, 자체 = 우리가 직접 정한 규칙

1. 기준일/기준값   : 자체 (예: 2022-06-30 = 1,000. 시작일은 데이터 기간 보고 확정)
2. 유니버스        : 자체 (kor_sector의 IT, 산업재, 유틸리티 중 AI·반도체·전력기기 후보)
3. 선정 필터       : KEDI (시총 3,000억 이상, 3개월 일평균 거래대금 10억 이상)
4. 편입 종목 수    : KEDI (최대 15개) - 자체로 10개로 바꿀지 결정
5. 비중            : KEDI (시총 가중, 단일 종목 25% 상한, 초과분은 나머지에 비례 재분배)
6. 정기변경        : KEDI (분기 마지막 영업일 D에 선정, D+3에 시행)
7. 수시변경(선택)  : KEDI (3영업일 연속 30% 초과 시 25%로 조정)
8. 지수 연속성     : WISE (기준시가총액 B 조정) / KEDI (보유수량 Q 고정 방식)
9. 에너지 바스켓   : 자체 (CL, NG 선물. 롤 규칙 A 근월물 / B 차월물 / C 분산 롤을 비교)
10. 결합 비중      : 자체 (주식 : 에너지 = 고정 비중 vs 변동성 기준, 실험)
"""

#%% 셀 B. 트레이딩뷰 에너지·환율·지수 심볼 테스트 (호출 7번, 종목마다 2초 대기)
import time
from tvDatafeed import TvDatafeed, Interval

tv = TvDatafeed()   # 익명 접속. 로그인이 필요하면 getpass로 입력받아 사용합니다

# 심볼과 거래소 표기는 확인 안 됨. 실패하는 것은 빼면 됩니다
tests = [
    ('CL1!', 'NYMEX'), ('CL2!', 'NYMEX'),
    ('NG1!', 'NYMEX'), ('NG2!', 'NYMEX'),
    ('USDKRW', 'FX_IDC'), ('IXIC', 'NASDAQ'), ('KOSPI', 'KRX'),
]
for sym, ex in tests:
    try:
        d = tv.get_hist(symbol=sym, exchange=ex, interval=Interval.in_daily, n_bars=1500)
        if d is None or len(d) == 0:
            print(sym, ex, '데이터 없음')
        else:
            print(sym, ex, len(d), d.index.min().date(), d.index.max().date())
    except Exception as e:
        print(sym, ex, 'FAIL', str(e)[:80])
    time.sleep(2)
    
    #%% 셀 C-1. 에너지 선물·환율·지수 수집
# 먼저 [국내 주식 데이터 수집 (TradingView).py]를 보고 오세요. 이미 DB에 데이터가 있으면 여기서부터 진행하세요.
RUN_DOWNLOAD = True    # True: tvDatafeed로 수집 / False: 건너뛰고 DB에서 바로 읽음

import os
import time
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine

DB_PW = os.environ['DB_PW']      # 비밀번호는 코드에 직접 쓰지 않습니다
engine = create_engine(f"mysql+pymysql://root:{quote_plus(DB_PW)}@127.0.0.1:3306/stock_db")
con = pymysql.connect(user='root', passwd=DB_PW, host='127.0.0.1', db='stock_db', charset='utf8')
cur = con.cursor()

fut_list = [
    ('CL1!', 'NYMEX'), ('CL2!', 'NYMEX'),      # WTI 원유 근월물, 차월물
    ('NG1!', 'NYMEX'), ('NG2!', 'NYMEX'),      # 천연가스 근월물, 차월물
    ('USDKRW', 'FX_IDC'),                      # 원/달러
    ('IXIC', 'NASDAQ'), ('KOSPI', 'KRX'),      # 나스닥 종합, 코스피
]
need = [s for s, _ in fut_list]


def clean_for_db(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.astype(object).where(df.notna(), None)


create_sql = """
create table if not exists tv_futures_price
(
    Date date,
    Open double,
    High double,
    Low double,
    Close double,
    Volume double,
    Symbol varchar(20),
    primary key (Date, Symbol)
)
"""
cur.execute(create_sql)
con.commit()

if RUN_DOWNLOAD:
    from tvDatafeed import TvDatafeed, Interval
    tv = TvDatafeed()      # 익명 접속

    done = set(pd.read_sql("SELECT DISTINCT Symbol FROM tv_futures_price", engine)['Symbol'])

    query_price = """
        insert into tv_futures_price (Date, Open, High, Low, Close, Volume, Symbol)
        values (%s,%s,%s,%s,%s,%s,%s) as new
        on duplicate key update
        Open=new.Open, High=new.High, Low=new.Low, Close=new.Close, Volume=new.Volume
    """

    for sym, ex in fut_list:
        if sym in done:
            print(sym, '이미 있음 - 건너뜀')
            continue
        try:
            price = tv.get_hist(symbol=sym, exchange=ex, interval=Interval.in_daily, n_bars=1500)
            if price is None or len(price) == 0:
                raise ValueError('데이터 없음')

            price = price.reset_index()
            price['Date'] = pd.to_datetime(price['datetime']).dt.strftime('%Y-%m-%d')
            price['Symbol'] = sym
            price = price.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                                          'close': 'Close', 'volume': 'Volume'})
            price = price[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']]

            cur.executemany(query_price, clean_for_db(price).values.tolist())
            con.commit()
            print(sym, len(price), '저장')
        except Exception as e:
            print(sym, 'FAIL', str(e)[:80])
        time.sleep(2)

# 필요한 심볼이 DB에 다 있는지 확인
have = set(pd.read_sql("SELECT DISTINCT Symbol FROM tv_futures_price", engine)['Symbol'])
missing = [s for s in need if s not in have]
if missing:
    print('DB에 없는 심볼:', missing, '-> RUN_DOWNLOAD = True 로 바꿔 다시 실행하세요')
else:
    print('필요한 심볼 모두 있음')

con.close()
