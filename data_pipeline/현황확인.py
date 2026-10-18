#%% [현황-0] 안내
"""
DB 현황 확인   파일명: 0_DB현황확인.py

지금 DB 에 어떤 데이터가 몇 년부터 몇 년까지 들어 있는지, 어디에 저장돼 있는지 보여 준다.
읽기만 하고 아무것도 바꾸지 않는다. 언제든 다시 실행해도 된다.

[저장 위치]
1) MySQL stock_db : 모든 테이블. 실제 파일은 MySQL 데이터 폴더(아래 [현황-1] 이 출력)에 있고,
   MySQL Workbench 같은 프로그램으로 보거나 이 파일처럼 파이썬에서 SQL 로 읽는다.
2) 홈폴더\\quant_cache : KRX, WICS 에서 받은 원본 csv, 최근 영업일 메모 (1번 파일이 만든다)
"""


#%% [현황-1] 연결 + 저장 위치 + 테이블 요약
import os
import getpass
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, inspect

DB_USER = 'root'
DB_HOST = '127.0.0.1'
DB_NAME = 'stock_db'
DB_PW = os.environ.get('DB_PW') or getpass.getpass('DB password: ')

engine = create_engine(
    f'mysql+pymysql://{DB_USER}:{quote_plus(DB_PW)}@{DB_HOST}:3306/{DB_NAME}')

# MySQL 이 파일을 저장하는 폴더 (stock_db 라는 하위 폴더 안에 테이블 파일이 있다)
try:
    print('MySQL 데이터 폴더:', pd.read_sql('select @@datadir as d;', con=engine)['d'][0])
except Exception as e:
    print('데이터 폴더 확인 실패:', type(e).__name__)

cache_dir = os.path.join(os.path.expanduser('~'), 'quant_cache')
print('원본 csv 캐시 폴더:', cache_dir, '/ 파일 수:', len(os.listdir(cache_dir)) if os.path.isdir(cache_dir) else '폴더 없음')

tables = inspect(engine).get_table_names()
print('\nDB 에 있는 테이블:', tables)

# 테이블마다 (행 수, 종목 수, 날짜 범위)를 보여 준다. 없는 테이블은 건너뛴다.
spec = [
    ('kor_ticker',         '종목코드', '기준일'),
    ('kor_sector',         'CMP_CD',  '기준일'),
    ('kor_price',          '종목코드', '날짜'),
    ('kor_fs',             '종목코드', '기준일'),
    ('dart_corp',          '종목코드', None),
    ('dart_fs',            '종목코드', '기준일'),
    ('kor_delisted',       '종목코드', '상장폐지일'),
    ('kor_price_delisted', '종목코드', '날짜'),
]
rows = []
for t, code, dt in spec:
    if t not in tables:
        rows.append((t, '없음', None, None, None))
        continue
    q = f'select count(*) as n, count(distinct {code}) as k' + (f', min({dt}) as a, max({dt}) as b' if dt else '') + f' from {t};'
    r = pd.read_sql(q, con=engine).iloc[0]
    rows.append((t, int(r['n']), int(r['k']), r['a'] if dt else None, r['b'] if dt else None))
print()
print(pd.DataFrame(rows, columns=['테이블', '행수', '종목수', '가장 이른 날짜', '가장 늦은 날짜']).to_string(index=False))


#%% [현황-2] 주가: 연도별 행 수와 종목 수
if 'kor_price' in tables:
    print(pd.read_sql("""
    select substr(날짜, 1, 4) as 연도, count(*) as 행수, count(distinct 종목코드) as 종목수
    from kor_price group by substr(날짜, 1, 4) order by 연도;
    """, con=engine).to_string(index=False))


#%% [현황-3] 재무: 계정 이름 목록 (연간) + 연도별 종목 수
# 그레이엄 파일의 ACC 에 적은 이름이 여기 있는 이름과 같아야 한다.
acc = pd.read_sql("""
select 계정, count(distinct 종목코드) as 종목수 from kor_fs where 공시구분 = 'y' group by 계정 order by 종목수 desc;
""", con=engine)
print('연간 계정 수:', len(acc))
print(acc.head(60).to_string(index=False))

# % 문자를 SQL 에 쓰지 않으려고 이름 검색은 pandas 로 한다
for kw in ['자본', '유동', '부채', '순이익', '매출', '영업', '현금']:
    hit = acc[acc['계정'].str.contains(kw)]
    print(f'\n[{kw}] 가 들어간 계정 {len(hit)}개')
    print(hit.head(15).to_string(index=False))

print('\n연간 재무: 결산 연도별 종목 수 (아무 계정이나 하나라도 있으면 1)')
print(pd.read_sql("""
select substr(기준일, 1, 4) as 연도, count(distinct 종목코드) as 종목수
from kor_fs where 공시구분 = 'y' group by substr(기준일, 1, 4) order by 연도;
""", con=engine).to_string(index=False))

#%% [현황-4] 폐지 종목 가격 점검: 단위(원), 종목 겹침, 수집 진행
import FinanceDataReader as fdr

if 'kor_price_delisted' in tables and 'kor_delisted' in tables:
    dl_p = pd.read_sql('select 날짜, 종목코드, 종가, 시가총액 from kor_price_delisted;', con=engine)
    dl_i = pd.read_sql('select 종목코드, 종목명, 상장폐지일 from kor_delisted;', con=engine)
    print(f'폐지 종목 {len(dl_i)}개 중 가격이 들어온 종목 {dl_p["종목코드"].nunique()}개 / 가격 행 {len(dl_p)}')

    # 1) 시가총액 / 종가 가 상장주식수와 비슷한지 (비슷하면 시가총액은 원 단위)
    shares = fdr.StockListing('KRX-DELISTING')[['Symbol', 'ListingShares']].drop_duplicates('Symbol').set_index('Symbol')['ListingShares']
    dl_p['날짜'] = pd.to_datetime(dl_p['날짜'])
    last = dl_p.sort_values('날짜').groupby('종목코드').tail(1).set_index('종목코드')
    last['시총/종가'] = last['시가총액'] / last['종가']
    last['목록 상장주식수'] = shares.reindex(last.index)
    last['비율'] = last['시총/종가'] / last['목록 상장주식수']
    last = last.join(dl_i.set_index('종목코드')[['종목명', '상장폐지일']])
    print('\n마지막 거래일 기준 (비율이 1 근처면 시가총액은 원 단위, 100000000 근처면 억원 단위)')
    print(last[['종목명', '날짜', '상장폐지일', '종가', '시가총액', '시총/종가', '목록 상장주식수', '비율']].head(15).to_string())
    print('\n비율 요약:', last['비율'].describe().round(3).to_dict())

    # 2) 현재 상장 종목과 코드가 겹치는지 (겹치면 그레이엄 파일은 현재 상장 쪽 데이터를 쓴다)
    now_codes = set(pd.read_sql('select distinct 종목코드 from kor_ticker;', con=engine)['종목코드'])
    both = sorted(set(dl_i['종목코드']) & now_codes)
    print('\n폐지 목록과 현재 상장 종목에 같이 있는 코드:', len(both), both[:10])

    # 3) 가격 수집 진행: 폐지일 기준 연도별 (가격이 들어온 종목 / 전체)
    dl_i['연도'] = pd.to_datetime(dl_i['상장폐지일']).dt.year
    dl_i['가격있음'] = dl_i['종목코드'].isin(dl_p['종목코드'])
    print('\n폐지 연도별 수집 진행')
    print(dl_i.groupby('연도')['가격있음'].agg(['sum', 'count']).rename(columns={'sum': '가격 있음', 'count': '전체'}).to_string())
    if 'crawl_log' in tables:
        print('\ncrawl_log 상태')
        print(pd.read_sql("select 상태, count(*) as 종목수 from crawl_log where 대상 = 'dprice' group by 상태;", con=engine).to_string(index=False))
else:
    print('kor_delisted / kor_price_delisted 가 아직 없습니다. 4_상장폐지_수집.py 를 먼저 실행하세요.')