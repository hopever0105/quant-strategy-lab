#%% 전세계 주식 데이터 수집 - 미국 (TradingView)
"""
전세계 주식 데이터 수집 - 미국 (TradingView)  (TradingView 스크리너 + tvDatafeed 기반, 독립 실행 파일)

[하는 일]
1) 티커 수집      : TradingView 스크리너로 미국 종목 리스트 + 시가총액 + 업종
2) 주가 수집      : tvDatafeed 로 종목별 일봉(시가/고가/저가/종가/거래량) -> DB
3) 재무 지표 수집 : 스크리너가 주는 TTM / 최근 분기 지표(PER, PBR, 매출, 순이익 등) -> DB
4) [선택] 재무제표: 계정별 연간/분기 시계열은 TradingView 에서 못 받아서 yahooquery 로 보완

[한계 - 꼭 알고 쓸 것]
- TradingView 는 공식 무료 API 가 없다. 이 코드는 비공식 패키지 2개를 쓰므로
  사이트가 바뀌면 갑자기 안 될 수 있고, 사이트 이용약관상 대량 수집은 권장되지 않는다. (개인 연구용)
- 주가는 액면분할만 반영된 가격이다. 배당까지 반영한 수정주가(Adj Close)는 없다.
- 로그인 없이(익명) 쓰면 받을 수 있는 봉 개수나 속도가 제한될 수 있다.
- 재무 지표는 "현재 시점 1개 값"이다. 분기별 과거 시계열이 아니다.

[설치] (Spyder 프롬프트))
  python -m pip install tradingview-screener tvdatafeed-enhanced sqlalchemy pymysql pandas tqdm yahooquery
  (tvdatafeed-enhanced 는 import 할 때 이름이 tvDatafeed 임)

[DB 비밀번호]
- 코드에 비밀번호를 쓰지 않는다. 환경변수 DB_PW 에 넣어두거나, 실행 시 입력창에 입력한다.
  (윈도우 영구 저장: 명령 프롬프트에서  setx DB_PW "비밀번호"  후 Spyder 재시작)
"""

#%% 0. 설정 + DB 연결 함수
import os
import time
import getpass
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

N_BARS = 1250        # 일봉 개수 (약 5년)
N_FS = 100           # [선택] 재무제표 시계열을 받을 종목 수 (시가총액 상위)


def db_connect():
    """sqlalchemy 엔진(읽기용)과 pymysql 연결(쓰기용)을 같이 만들기."""
    engine = create_engine(
        f'mysql+pymysql://{DB_USER}:{quote_plus(DB_PW)}@{DB_HOST}:3306/{DB_NAME}')
    con = pymysql.connect(user=DB_USER, passwd=DB_PW, host=DB_HOST,
                          db=DB_NAME, charset='utf8')
    return engine, con


def clean_for_db(df):
    """inf / NaN 을 None 으로 바꿔서 MySQL 에 null로 저장"""
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.astype(object).where(df.notna(), None)


#%% 1. 사용 가능한 재무 지표 필드 확인
# 스크리너 필드 이름은 TradingView 가 바꿀 수 있다. 통과된 것만 사용 가능.
from tradingview_screener import Query, Column

fund_candidates = [
    'price_earnings_ttm',            # PER
    'price_book_fq',                 # PBR
    'price_sales_current',           # PSR
    'price_free_cash_flow_ttm',      # P/FCF
    'dividend_yield_recent',         # 배당수익률
    'earnings_per_share_basic_ttm',  # EPS
    'earnings_per_share_diluted_ttm',
    'total_revenue_ttm',             # 매출 (TTM)
    'net_income_ttm',                # 순이익 (TTM)
    'ebitda_ttm',
    'free_cash_flow_ttm',
    'gross_margin_ttm',
    'operating_margin_ttm',
    'net_margin_ttm',
    'return_on_equity_fq',           # ROE
    'return_on_assets_fq',           # ROA
    'debt_to_equity_fq',
    'current_ratio_fq',
    'total_assets_fq',
    'total_debt_fq',
    'book_value_per_share_fq',
]

fund_fields = []
for f in fund_candidates:
    try:
        Query().set_markets('america').select(f).limit(1).get_scanner_data()
        fund_fields.append(f)
    except Exception:
        print('사용 불가 필드:', f)

print('사용 가능:', fund_fields)


#%% 2. 티커 + 재무 지표 한 번에 수집
base_fields = ['name', 'description', 'exchange', 'sector', 'industry', 'market_cap_basic', 'close']

count, all_data_df = (Query()
    .set_markets('america')
    .select(*base_fields, *fund_fields)
    .where(Column('type') == 'stock', Column('exchange').isin(['NASDAQ', 'NYSE', 'AMEX']))
    .order_by('market_cap_basic', ascending=False)
    .limit(10000)
    .get_scanner_data())

print('전체 종목 수:', count, '/ 받은 행:', len(all_data_df))   # 두 숫자가 많이 다르면 limit 을 키울 것
all_data_df.head()


#%% 3. 티커 테이블 정리
ticker_list = all_data_df.rename(columns={
    'name': 'Symbol', 'description': 'Name', 'exchange': 'Exchange',
    'sector': 'Sector', 'industry': 'Industry', 'market_cap_basic': 'MarketCap'})
ticker_list = ticker_list[['Symbol', 'Name', 'Exchange', 'Sector', 'Industry', 'MarketCap']].copy()

# TradingView 티커 표기(BRK.B)를 그대로 쓴다. 야후용(BRK-B) 변환은 12번 셀에서만 적용
ticker_list = ticker_list.dropna(subset=['Symbol']).drop_duplicates('Symbol').reset_index(drop=True)
ticker_list['date'] = pd.Timestamp.today().strftime('%Y-%m-%d')

ticker_list.head()


#%% 4. 티커 테이블 생성 SQL
"""
use stock_db;

create table tv_us_ticker
(
    Symbol varchar(20),
    Name varchar(255),
    Exchange varchar(20),
    Sector varchar(255),
    Industry varchar(255),
    MarketCap double,
    date date,
    primary key (Symbol, date)
);
"""

#%% 5. 티커 저장
engine, con = db_connect()
mycursor = con.cursor()

query = """
    insert into tv_us_ticker (Symbol, Name, Exchange, Sector, Industry, MarketCap, date)
    values (%s,%s,%s,%s,%s,%s,%s) as new
    on duplicate key update
    Name=new.Name, Exchange=new.Exchange, Sector=new.Sector,
    Industry=new.Industry, MarketCap=new.MarketCap
"""
args = clean_for_db(ticker_list).values.tolist()
mycursor.executemany(query, args)
con.commit()

engine.dispose()
con.close()


#%% 6. 주가: 한 종목 테스트 (tvDatafeed)
from tvDatafeed import TvDatafeed, Interval

tv = TvDatafeed()   # 익명 접속. 계정이 있으면 TvDatafeed('아이디', '비밀번호')

sample = ticker_list.iloc[0]
price = tv.get_hist(symbol=sample['Symbol'], exchange=sample['Exchange'],
                    interval=Interval.in_daily, n_bars=N_BARS)

# 결과: 인덱스=datetime, 열=symbol, open, high, low, close, volume  (실패하면 None)
price.head()


#%% 7. 주가 테이블 생성 SQL
"""
use stock_db;

create table tv_us_price
(
    Date date,
    Open double,
    High double,
    Low double,
    Close double,
    Volume double,
    Symbol varchar(20),
    primary key (Date, Symbol)
);
"""

#%% 8. 전 종목 주가 저장
from tvDatafeed import TvDatafeed, Interval

tv = TvDatafeed()
engine, con = db_connect()
mycursor = con.cursor()

query_price = """
    insert into tv_us_price (Date, Open, High, Low, Close, Volume, Symbol)
    values (%s,%s,%s,%s,%s,%s,%s) as new
    on duplicate key update
    Open=new.Open, High=new.High, Low=new.Low, Close=new.Close, Volume=new.Volume
"""

error_list = []

for i in tqdm(range(0, len(ticker_list))):
    symbol = ticker_list['Symbol'][i]
    exchange = ticker_list['Exchange'][i]

    try:
        price = tv.get_hist(symbol=symbol, exchange=exchange,
                            interval=Interval.in_daily, n_bars=N_BARS)
        if price is None or len(price) == 0:
            raise ValueError('데이터 없음')

        price = price.reset_index()
        price['Date'] = pd.to_datetime(price['datetime']).dt.strftime('%Y-%m-%d')
        price['Symbol'] = symbol
        price = price.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                                      'close': 'Close', 'volume': 'Volume'})
        # 열 순서를 insert 문 순서에 맞춤
        price = price[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']]

        mycursor.executemany(query_price, clean_for_db(price).values.tolist())
        con.commit()

    except Exception:
        print(symbol)
        error_list.append(symbol)

    time.sleep(1)    # 너무 빠르게 요청하면 차단될 수 있음

engine.dispose()
con.close()
print('실패 종목 수:', len(error_list))


#%% 9. 재무 지표 정리 (가로 -> 세로)
# 스크리너에서 같이 받은 지표를 (Symbol, date, indicator, value) 형태로 바꿈. 
fund_df = all_data_df.rename(columns={'name': 'Symbol'})[['Symbol'] + fund_fields]
fund_df = fund_df.dropna(subset=['Symbol']).drop_duplicates('Symbol')
fund_df = fund_df.melt(id_vars='Symbol', var_name='indicator', value_name='value')
fund_df = fund_df[~pd.isnull(fund_df['value'])]
fund_df['date'] = pd.Timestamp.today().strftime('%Y-%m-%d')
fund_df = fund_df[['Symbol', 'date', 'indicator', 'value']]

fund_df.head()


#%% 10. 재무 지표 테이블 생성 SQL
"""
use stock_db;

create table tv_us_fundamental
(
    Symbol varchar(20),
    date date,
    indicator varchar(60),
    value double,
    primary key (Symbol, date, indicator)
);
"""

#%% 11. 재무 지표 저장
engine, con = db_connect()
mycursor = con.cursor()

query_fund = """
    insert into tv_us_fundamental (Symbol, date, indicator, value)
    values (%s,%s,%s,%s) as new
    on duplicate key update
    value = new.value
"""
mycursor.executemany(query_fund, clean_for_db(fund_df).values.tolist())
con.commit()

engine.dispose()
con.close()


#%% 12. [선택] 계정별 재무제표 시계열 (yahooquery)
# TradingView 는 연간/분기 재무제표 원본을 주지 않는다. 같은 종목을 야후에서 받아 보완한다.
# 시가총액 상위 N_FS 개만 받는다. (전 종목은 시간이 오래 걸림)
from yahooquery import Ticker

"""
use stock_db;

create table tv_us_fs
(
    ticker varchar(20),
    date date,
    account varchar(100),
    value double,
    freq varchar(1),
    primary key (ticker, date, account, freq)
);
"""

engine, con = db_connect()
mycursor = con.cursor()

query_fs = """
    insert into tv_us_fs (ticker, date, account, value, freq)
    values (%s,%s,%s,%s,%s) as new
    on duplicate key update
    value = new.value
"""


def get_fs(yahoo_symbol, frequency, freq_label):
    """yahooquery 재무제표(손익/재무상태/현금흐름 전부)를 세로로 긴 형태로 반환."""
    
    data = Ticker(yahoo_symbol).all_financial_data(frequency=frequency)
    if isinstance(data, str) or len(data) == 0:      # 데이터 없으면 에러 반환
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


error_fs = []
fs_targets = ticker_list.head(N_FS).reset_index(drop=True)

for i in tqdm(range(0, len(fs_targets))):
    symbol = fs_targets['Symbol'][i]
    yahoo_candidates = [symbol.replace('.', '-')]    # 야후는 BRK.B 대신 BRK-B

    try:
        data_fs = None
        for ys in yahoo_candidates:
            y = get_fs(ys, 'a', 'y')
            q = get_fs(ys, 'q', 'q')
            parts = [d for d in (y, q) if d is not None]
            if parts:
                data_fs = pd.concat(parts, axis=0)
                data_fs['ticker'] = symbol          # 저장은 원래 TradingView 티커로 통일
                break
        if data_fs is None:
            raise ValueError('데이터 없음')

        mycursor.executemany(query_fs, clean_for_db(data_fs).values.tolist())
        con.commit()

    except Exception:
        print(symbol)
        error_fs.append(symbol)

    time.sleep(2)

engine.dispose()
con.close()
print('실패 종목 수:', len(error_fs))