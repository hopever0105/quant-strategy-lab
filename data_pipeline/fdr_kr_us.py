#%% 패키지(FinanceDataReader)로 한국/해외 주식 데이터 수집
"""패키지 기반 수집 (FinanceDataReader + yahooquery, 독립 실행 파일)

[하는 일]
- 한국 : 티커(KRX 상장목록) -> 일봉 주가(네이버 차트) -> 요약 재무(네이버 금융, 연간/분기) -> DB
- 해외 : 티커(네이버 해외주식 목록) -> 일봉 주가(야후) -> 계정별 재무제표(yahooquery) -> DB  [선택]

[왜 이 방법인가]
- 로그인/키 없이 무료로 돌아가고, 책에서 쓴 네이버 데이터를 그대로 패키지가 대신 받아 준다.
- 증권사 API(한국투자증권 KIS Open API 등)는 계좌 개설 + appkey 발급이 필요하다.
  계좌가 있으면 주가 수집 부분(fdr.DataReader)만 KIS 호출로 바꿔 끼우면 된다. (이 파일에는 미포함)

[한계]
- 모두 비공식 경로(웹페이지/비공개 API)라 사이트가 바뀌면 패키지 업데이트 전까지 일시적으로 안 될 수 있다.
- 한국 재무는 네이버가 주는 "요약 재무"(매출액, 영업이익, 당기순이익, EPS, PER, BPS, PBR 등)이다.
  FnGuide 처럼 손익/재무상태/현금흐름 계정 전체를 주지는 않는다. (금액 단위: 억원)
- 한국 종목 목록 중 시가총액(Marcap)은 KRX 서버 상태에 따라 안 올 수 있다. 그 경우 캐시 목록을 대신 쓴다.

[설치] (Spyder 콘솔 / 프롬프트에서 한 번만)
  python -m pip install finance-datareader yahooquery sqlalchemy pymysql pandas tqdm lxml
  (패키지 이름은 finance-datareader, import 할 때는 FinanceDataReader)

[DB 비밀번호]
- 코드에 비밀번호를 쓰지 않는다. 환경변수 DB_PW 에 넣어두거나, 실행 시 입력창에 입력한다.
  (윈도우 영구 저장: 명령 프롬프트에서  setx DB_PW "비밀번호"  후 Spyder 재시작)
"""

#%% 0. 설정 + DB 연결 함수
import os
import time
import getpass
from datetime import date
from dateutil.relativedelta import relativedelta
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from tqdm import tqdm
import FinanceDataReader as fdr

DB_USER = 'root'
DB_HOST = '127.0.0.1'
DB_NAME = 'stock_db'
DB_PW = os.environ.get('DB_PW') or getpass.getpass('DB password: ')

START = (date.today() + relativedelta(years=-5)).strftime('%Y-%m-%d')   # 주가 시작일 (5년 전)
FS_LIMIT = None      # 재무 수집 종목 수 제한. None = 전 종목 (전 종목은 몇 시간 걸림). 테스트는 20 정도로


def db_connect():
    """sqlalchemy 엔진(읽기용)과 pymysql 연결(쓰기용)을 같이 만든다."""
    engine = create_engine(
        f'mysql+pymysql://{DB_USER}:{quote_plus(DB_PW)}@{DB_HOST}:3306/{DB_NAME}')
    con = pymysql.connect(user=DB_USER, passwd=DB_PW, host=DB_HOST,
                          db=DB_NAME, charset='utf8')
    return engine, con


def clean_for_db(df):
    """inf / NaN 을 None 으로 바꿔서 MySQL 에 null 로 반환"""
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.astype(object).where(df.notna(), None)


# ============================================================================
#  A. 한국
# ============================================================================

#%% A-1. 티커 수집 (KRX 상장 목록)
# fdr.StockListing('KRX') : Code, Name, Market, Close, Marcap(시가총액) ...
try:
    krx = fdr.StockListing('KRX')
except Exception as e:
    print('KRX 목록 실패 -> 캐시 목록(KRX-DESC)으로 대체:', e)
    krx = fdr.StockListing('KRX-DESC')       # 시가총액은 없고 Code, Name, Market, Sector, Industry ...

if 'Marcap' not in krx.columns:
    krx['Marcap'] = np.nan

# 업종/산업은 상세 목록(KRX-DESC)에서 붙인다 (실패해도 진행)
try:
    desc = fdr.StockListing('KRX-DESC')[['Code', 'Sector', 'Industry']]
    krx = krx.drop(columns=[c for c in ['Sector', 'Industry'] if c in krx.columns])
    krx = krx.merge(desc, on='Code', how='left')
except Exception as e:
    print('업종 정보 못 받음:', e)
    krx['Sector'] = None
    krx['Industry'] = None

krx.head()


#%% A-2. 티커 정리 
kor_ticker = krx.rename(columns={'Code': '종목코드', 'Name': '종목명', 'Market': '시장구분',
                                 'Marcap': '시가총액', 'Sector': '업종', 'Industry': '산업'})
kor_ticker = kor_ticker[['종목코드', '종목명', '시장구분', '시가총액', '업종', '산업']].copy()

# 코스피 / 코스닥만 (코넥스 제외)
kor_ticker = kor_ticker[kor_ticker['시장구분'].astype(str).str.startswith(('KOSPI', 'KOSDAQ'))]
kor_ticker['종목명'] = kor_ticker['종목명'].str.strip()

# 스팩 -> 우선주(코드 끝이 0이 아님) -> 리츠 -> 나머지는 보통주
kor_ticker['종목구분'] = np.where(kor_ticker['종목명'].str.contains('스팩|제[0-9]+호'), '스팩',
                         np.where(kor_ticker['종목코드'].str[-1:] != '0', '우선주',
                         np.where(kor_ticker['종목명'].str.endswith('리츠'), '리츠',
                                  '보통주')))
kor_ticker['기준일'] = date.today().strftime('%Y-%m-%d')
kor_ticker = kor_ticker.reset_index(drop=True)

kor_ticker['종목구분'].value_counts()


#%% A-3. 티커 테이블 생성 SQL
"""
use stock_db;

create table fdr_kr_ticker
(
    종목코드 varchar(6) not null,
    종목명 varchar(50),
    시장구분 varchar(20),
    시가총액 double,
    업종 varchar(100),
    산업 varchar(255),
    기준일 date,
    종목구분 varchar(10),
    primary key (종목코드, 기준일)
);
"""

#%% A-4. 티커 저장
engine, con = db_connect()
mycursor = con.cursor()

query = """
    insert into fdr_kr_ticker (종목코드, 종목명, 시장구분, 시가총액, 업종, 산업, 기준일, 종목구분)
    values (%s,%s,%s,%s,%s,%s,%s,%s) as new
    on duplicate key update
    종목명=new.종목명, 시장구분=new.시장구분, 시가총액=new.시가총액,
    업종=new.업종, 산업=new.산업, 종목구분=new.종목구분
"""
args = clean_for_db(kor_ticker[['종목코드', '종목명', '시장구분', '시가총액', '업종', '산업',
                                '기준일', '종목구분']]).values.tolist()
mycursor.executemany(query, args)
con.commit()

engine.dispose()
con.close()


#%% A-5. 주가: 한 종목 테스트
# 한국 종목코드를 넣으면 네이버 차트에서 일봉을 받기 
price = fdr.DataReader('005930', START)

# 결과: 인덱스=Date, 열=Open, High, Low, Close, Volume, Change
price.head()


#%% A-6. 주가 테이블 생성 SQL
"""
use stock_db;

create table fdr_kr_price
(
    날짜 date,
    시가 double,
    고가 double,
    저가 double,
    종가 double,
    거래량 double,
    종목코드 varchar(6),
    primary key (날짜, 종목코드)
);
"""

#%% A-7. 전 종목(보통주) 주가 저장
engine, con = db_connect()
mycursor = con.cursor()

ticker_list = pd.read_sql("""
select * from fdr_kr_ticker
where 기준일 = (select max(기준일) from fdr_kr_ticker)
and 종목구분 = '보통주';
""", con=engine)

query_price = """
    insert into fdr_kr_price (날짜, 시가, 고가, 저가, 종가, 거래량, 종목코드)
    values (%s,%s,%s,%s,%s,%s,%s) as new
    on duplicate key update
    시가=new.시가, 고가=new.고가, 저가=new.저가, 종가=new.종가, 거래량=new.거래량
"""

error_list = []

for i in tqdm(range(0, len(ticker_list))):
    ticker = ticker_list['종목코드'][i]

    try:
        price = fdr.DataReader(ticker, START)
        if price is None or len(price) == 0:
            raise ValueError('데이터 없음')

        price = price.reset_index()
        price = price[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        price.columns = ['날짜', '시가', '고가', '저가', '종가', '거래량']
        price['날짜'] = pd.to_datetime(price['날짜']).dt.strftime('%Y-%m-%d')
        price['종목코드'] = ticker

        mycursor.executemany(query_price, clean_for_db(price).values.tolist())
        con.commit()

    except Exception:
        print(ticker)
        error_list.append(ticker)

    time.sleep(0.5)

engine.dispose()
con.close()
print('실패 종목 수:', len(error_list))


#%% A-8. 재무: 한 종목 테스트 (네이버 금융 요약 재무)
# 'NAVER/FINSTATE-Y/종목코드' = 연간, 'NAVER/FINSTATE-Q/종목코드' = 분기

fs_y = fdr.SnapDataReader('NAVER/FINSTATE-Y/005930')

# 결과: 인덱스=날짜(결산월), 열=계정(매출액, 영업이익, 당기순이익, EPS, PER, BPS, PBR ...)

# 'E' 가 붙은 열은 증권사 추정치라서 미래 날짜로 들어온다 (아래 함수에서 제외)

fs_y.head()


#%% A-9. 재무 정리 함수 (가로 -> 세로)
def finstate(code, freq):
    """freq: 'Y'=연간, 'Q'=분기. 결과: 계정, 기준일, 값, 종목코드, 공시구분"""
    df = fdr.SnapDataReader(f'NAVER/FINSTATE-{freq}/{code}')
    if df is None or len(df) == 0:
        return None

    df = df[df.index.notna()]                         # 날짜로 못 바꾼 열 제거
    df = df.loc[:, ~df.columns.duplicated()]          # 같은 이름의 계정이 있으면 첫 번째만
    df.index.name = '날짜'
    df = df.reset_index()
    df = df.melt(id_vars='날짜', var_name='계정', value_name='값')
    df['값'] = pd.to_numeric(df['값'], errors='coerce')
    df = df[~pd.isnull(df['값'])]

    df['기준일'] = pd.to_datetime(df['날짜']) + pd.tseries.offsets.MonthEnd(0)   # 월말로 맞춤
    df = df[df['기준일'] <= pd.Timestamp.today()]                                 # 추정치(미래) 제외
    df['기준일'] = df['기준일'].dt.strftime('%Y-%m-%d')
    df['종목코드'] = code
    df['공시구분'] = freq.lower()
    return df[['계정', '기준일', '값', '종목코드', '공시구분']]


sample_y = finstate('005930', 'Y')
sample_q = finstate('005930', 'Q')
pd.concat([sample_y, sample_q]).head()


#%% A-10. 재무 테이블 생성 SQL
"""
use stock_db;

create table fdr_kr_fs
(
    계정 varchar(60),
    기준일 date,
    값 double,
    종목코드 varchar(6),
    공시구분 varchar(1),
    primary key (계정, 기준일, 종목코드, 공시구분)
);
"""

#%% A-11. 전 종목 재무 저장
# 종목당 요청이 4번(연간/분기 x 2) 나가서 전 종목은 오래 걸린다. 테스트는 FS_LIMIT 을 20 정도로.
engine, con = db_connect()
mycursor = con.cursor()

ticker_list = pd.read_sql("""
select * from fdr_kr_ticker
where 기준일 = (select max(기준일) from fdr_kr_ticker)
and 종목구분 = '보통주'
order by 시가총액 desc;
""", con=engine)
if FS_LIMIT:
    ticker_list = ticker_list.head(FS_LIMIT)

query_fs = """
    insert into fdr_kr_fs (계정, 기준일, 값, 종목코드, 공시구분)
    values (%s,%s,%s,%s,%s) as new
    on duplicate key update
    값 = new.값
"""

error_list = []

for i in tqdm(range(0, len(ticker_list))):
    ticker = ticker_list['종목코드'][i]

    try:
        parts = [finstate(ticker, 'Y'), finstate(ticker, 'Q')]
        parts = [p for p in parts if p is not None]
        if not parts:
            raise ValueError('데이터 없음')
        data_fs = pd.concat(parts)

        mycursor.executemany(query_fs, clean_for_db(data_fs).values.tolist())
        con.commit()

    except Exception:
        print(ticker)
        error_list.append(ticker)

    time.sleep(1)

engine.dispose()
con.close()
print('실패 종목 수:', len(error_list))


# ============================================================================
#  B. 해외 (미국)  
# ============================================================================

#%% B-1. 티커 수집 (네이버 해외주식 목록)
# fdr.StockListing('NASDAQ' / 'NYSE' / 'AMEX') : Symbol, Name(영문), IndustryCode, Industry
us_list = []
for exch in ['NASDAQ', 'NYSE', 'AMEX']:
    d = fdr.StockListing(exch)
    d['Exchange'] = exch
    us_list.append(d)

us_ticker = pd.concat(us_list, ignore_index=True)
us_ticker = us_ticker[['Symbol', 'Name', 'Exchange', 'Industry']].drop_duplicates('Symbol')
us_ticker['date'] = date.today().strftime('%Y-%m-%d')
us_ticker = us_ticker.reset_index(drop=True)

us_ticker.head()


#%% B-2. 해외 테이블 생성 SQL
"""
use stock_db;

create table fdr_us_ticker
(
    Symbol varchar(20),
    Name varchar(255),
    Exchange varchar(20),
    Industry varchar(255),
    date date,
    primary key (Symbol, date)
);

create table fdr_us_price
(
    Date date,
    Open double,
    High double,
    Low double,
    Close double,
    `Adj Close` double,
    Volume double,
    Symbol varchar(20),
    primary key (Date, Symbol)
);

create table fdr_us_fs
(
    ticker varchar(20),
    date date,
    account varchar(100),
    value double,
    freq varchar(1),
    primary key (ticker, date, account, freq)
);
"""

#%% B-3. 해외 티커 저장
engine, con = db_connect()
mycursor = con.cursor()

query = """
    insert into fdr_us_ticker (Symbol, Name, Exchange, Industry, date)
    values (%s,%s,%s,%s,%s) as new
    on duplicate key update
    Name=new.Name, Exchange=new.Exchange, Industry=new.Industry
"""
mycursor.executemany(query, clean_for_db(us_ticker).values.tolist())
con.commit()

engine.dispose()
con.close()


#%% B-4. 해외 주가: 한 종목 테스트 (야후 출처)
# 야후는 BRK.B 대신 BRK-B 로 쓴다.
price_us = fdr.DataReader('AAPL', START)

# 결과: 인덱스=Date, 열=Open, High, Low, Close, Adj Close, Volume
price_us.head()


#%% B-5. 해외 전 종목 주가 저장
engine, con = db_connect()
mycursor = con.cursor()

ticker_list_us = pd.read_sql("""
select * from fdr_us_ticker
where date = (select max(date) from fdr_us_ticker);
""", con=engine)

query_price = """
    insert into fdr_us_price (Date, Open, High, Low, Close, `Adj Close`, Volume, Symbol)
    values (%s,%s,%s,%s,%s,%s,%s,%s) as new
    on duplicate key update
    Open=new.Open, High=new.High, Low=new.Low, Close=new.Close,
    `Adj Close`=new.`Adj Close`, Volume=new.Volume
"""

error_list = []

for i in tqdm(range(0, len(ticker_list_us))):
    symbol = ticker_list_us['Symbol'][i]

    try:
        price = fdr.DataReader(symbol.replace('.', '-'), START)
        if price is None or len(price) == 0:
            raise ValueError('데이터 없음')

        price = price.reset_index()
        price['Date'] = pd.to_datetime(price['Date']).dt.strftime('%Y-%m-%d')
        price['Symbol'] = symbol
        
        # 열 순서를 insert 문 순서에 맞춤
        price = price[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Symbol']]

        mycursor.executemany(query_price, clean_for_db(price).values.tolist())
        con.commit()

    except Exception:
        print(symbol)
        error_list.append(symbol)

    time.sleep(1)

engine.dispose()
con.close()
print('실패 종목 수:', len(error_list))


#%% B-6. 해외 재무제표 (yahooquery: 손익/재무상태/현금흐름 전 계정, 연간 + 분기)
from yahooquery import Ticker


def get_fs(yahoo_symbol, frequency, freq_label):
    """yahooquery 재무제표를 세로로 긴 형태로 반환. 데이터가 없으면 None."""
    data = Ticker(yahoo_symbol).all_financial_data(frequency=frequency)
    if isinstance(data, str) or len(data) == 0:      # 데이터 없으면 에러 문자열이 옴
        return None
    data = data.reset_index()
    data = data[data['periodType'] != 'TTM']          # TTM 행이 섞이면 같은 날짜의 분기값을 덮어쓰므로 제외
    data = data.loc[:, ~data.columns.isin(['periodType', 'currencyCode'])]
    data = data.melt(id_vars=['symbol', 'asOfDate'])
    data = data[~pd.isnull(data['value'])]
    data['freq'] = freq_label
    data.columns = ['ticker', 'date', 'account', 'value', 'freq']
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
    return data


engine, con = db_connect()
mycursor = con.cursor()

query_fs = """
    insert into fdr_us_fs (ticker, date, account, value, freq)
    values (%s,%s,%s,%s,%s) as new
    on duplicate key update
    value = new.value
"""

error_list = []
targets = ticker_list_us.head(FS_LIMIT) if FS_LIMIT else ticker_list_us
targets = targets.reset_index(drop=True)

for i in tqdm(range(0, len(targets))):
    symbol = targets['Symbol'][i]

    try:
        y = get_fs(symbol.replace('.', '-'), 'a', 'y')
        q = get_fs(symbol.replace('.', '-'), 'q', 'q')
        parts = [d for d in (y, q) if d is not None]
        if not parts:
            raise ValueError('데이터 없음')
        data_fs = pd.concat(parts, axis=0)
        data_fs['ticker'] = symbol

        mycursor.executemany(query_fs, clean_for_db(data_fs).values.tolist())
        con.commit()

    except Exception:
        print(symbol)
        error_list.append(symbol)

    time.sleep(2)

engine.dispose()
con.close()
print('실패 종목 수:', len(error_list))