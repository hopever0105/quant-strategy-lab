#%% [수집-0] 안내
"""
국내 주식 데이터 수집 (차단 대비 버전)   파일명: 1_자료수집.py

[하는 일]
KRX 업종분류 + 개별종목 지표 -> kor_ticker / WICS 섹터 -> kor_sector
네이버 수정주가 -> kor_price / FnGuide 재무제표 -> kor_fs

[실행 순서]
[수집-1] 설정, 테이블 준비 -> [수집-2] 최근 영업일 -> [수집-3]~[수집-6] KRX, WICS
-> [수집-7] 주가 -> [수집-8] 재무 -> [수집-9] 결과 확인

[저장 위치]
1) MySQL stock_db           : 최종 저장
2) 홈폴더\\quant_cache        : KRX, WICS 원본 csv 를 날짜별로 보관 (같은 날 재실행하면 접속 안 함)
3) stock_db.crawl_log 테이블  : 종목별 진행 상황 (중간에 멈춰도 이어받기)

[KRX 로그인]
KRX 데이터 사이트가 로그인을 요구해서 pykrx 도 KRX 회원 아이디/비밀번호가 필요하다.
pykrx 는 환경변수 KRX_ID, KRX_PW 를 읽는다. (윈도우 영구 저장: 명령 프롬프트에서
  setx KRX_ID "아이디"
  setx KRX_PW "비밀번호"
후 Spyder 를 완전히 껐다 켠다.)

[DB 비밀번호]
코드에 직접 쓰지 않는다. 환경변수 DB_PW 에 넣거나 실행할 때 입력한다.
  setx DB_PW "비밀번호"   (이후 Spyder 재시작)

[새 컴퓨터에서]
MySQL 에서 먼저 create database stock_db; 를 한 번 실행한다. 테이블은 [수집-1] 이 만든다.

[패키지] 기존에 쓰던 것과 같다 (pykrx, requests, beautifulsoup4, tqdm, sqlalchemy, pymysql, pandas)
없는 게 있으면 Anaconda Prompt 에서:
  conda activate ta_env
  python -m pip install pykrx beautifulsoup4 tqdm sqlalchemy pymysql
Spyder 에서 ta_env 를 쓰려면 Tools > Preferences > Python interpreter 에서
ta_env 의 python.exe 를 지정하거나, 위처럼 activate 한 뒤 spyder 를 실행한다.

[실행 습관]
- KRX 는 장 마감 후(저녁)에 하루 한 번만. 같은 날 다시 돌려도 캐시를 읽으니 접속하지 않는다.
- 주가/재무 셀은 중간에 멈춰도 된다. 다시 실행하면 한 종목부터 이어서 받는다.
"""


#%% [수집-1] 공통 설정 + 안전 요청 함수 + DB 테이블 준비
import os
import re
import time
import random
import getpass
from io import BytesIO
from datetime import datetime, date, timedelta
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests as rq
import pymysql
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine
from tqdm import tqdm

# ---- 설정값 ----
CACHE_DIR = os.path.join(os.path.expanduser('~'), 'quant_cache')   # 원본 csv 보관 폴더
os.makedirs(CACHE_DIR, exist_ok=True)

SLEEP_PRICE = (0.3, 0.8)      # 주가: 종목 사이 랜덤 대기(초)
SLEEP_FS = (1.0, 2.0)         # 재무: 종목 사이 랜덤 대기(초)
MAX_CONSEC_FAIL = 10          # 연속 실패가 이 횟수에 닿으면 차단으로 보고 멈춘다
PRICE_YEARS = 5               # 주가를 몇 년치 받을지
FS_REFRESH_DAYS = 30          # 재무는 이 기간 안에 받은 종목이면 건너뛴다

DB_USER = 'root'
DB_HOST = '127.0.0.1'
DB_NAME = 'stock_db'
DB_PW = os.environ.get('DB_PW') or getpass.getpass('DB password: ')


def db_connect():
    """읽기용 sqlalchemy 엔진과 쓰기용 pymysql 연결을 같이 만든다."""
    engine = create_engine(
        f'mysql+pymysql://{DB_USER}:{quote_plus(DB_PW)}@{DB_HOST}:3306/{DB_NAME}')
    con = pymysql.connect(user=DB_USER, passwd=DB_PW, host=DB_HOST,
                          db=DB_NAME, charset='utf8')
    return engine, con


# ---- 차단 대비 도구 ----
class BlockedError(Exception):
    """연속 실패/로그인 풀림 등 차단이 의심될 때"""


class NoData(Exception):
    """응답은 정상인데 그 종목의 데이터가 없는 경우 (신규상장 등). 차단으로 세지 않는다."""


class Breaker:
    """연속 실패 횟수를 세다가 한도에 닿으면 BlockedError 를 낸다."""
    def __init__(self, limit=MAX_CONSEC_FAIL):
        self.n = 0
        self.limit = limit

    def ok(self):
        self.n = 0

    def fail(self):
        self.n += 1
        if self.n >= self.limit:
            raise BlockedError(f'연속 {self.n}번 실패 -> 차단이 의심되어 중단합니다. '
                               '시간을 두고 다시 실행하면 이어서 받습니다.')


def nap(rng):
    """랜덤 대기. 일정한 간격으로 요청하면 기계처럼 보이기 쉬워서 흔든다."""
    time.sleep(random.uniform(*rng))


def safe_call(fn, tries=3, base=3):
    """fn 을 실행하고, 실패하면 3초, 6초... 늘려 가며 재시도한다. 끝까지 실패하면 예외를 그대로 올린다."""
    for k in range(tries):
        try:
            return fn()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(base * (2 ** k) + random.uniform(0, 1))


def log_done(cur, con, kind, ticker, status):
    """종목별 진행 상황 기록. kind: 'price' / 'fs', status: 'ok' / 'nodata'"""
    cur.execute("""
        insert into crawl_log (대상, 종목코드, 완료일, 상태)
        values (%s,%s,%s,%s) as new
        on duplicate key update 완료일 = new.완료일, 상태 = new.상태
    """, (kind, ticker, date.today().strftime('%Y-%m-%d'), status))
    con.commit()


# ---- 테이블 만들기 (없을 때만) ----
DDL = [
"""
create table if not exists kor_ticker
(
    종목코드 varchar(6) not null,
    종목명 varchar(20),
    시장구분 varchar(6),
    종가 float,
    시가총액 float,
    기준일 date,
    EPS float,
    PER float,
    BPS float,
    주당배당금 float,
    종목구분 varchar(5),
    primary key(종목코드, 기준일)
);
""",
"""
create table if not exists kor_sector
(
    IDX_CD varchar(3),
    CMP_CD varchar(6),
    CMP_KOR varchar(20),
    SEC_NM_KOR varchar(10),
    기준일 date,
    primary key(CMP_CD, 기준일)
);
""",
"""
create table if not exists kor_price
(
    날짜 date,
    시가 double,
    고가 double,
    저가 double,
    종가 double,
    거래량 double,
    종목코드 varchar(6),
    primary key(날짜, 종목코드)
);
""",
"""
create table if not exists kor_fs
(
    계정 varchar(30),
    기준일 date,
    값 float,
    종목코드 varchar(6),
    공시구분 varchar(1),
    primary key(계정, 기준일, 종목코드, 공시구분)
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


#%% [수집-2] 최근 영업일 구하기
"""
- 오늘이 휴일/주말이면 직전 영업일, 영업일이면 오늘을 돌려준다. 결과는 'yyyymmdd' 문자열.
- 하루에 한 번만 KRX 에 물어보고 결과를 캐시 파일에 적어 둔다.
- 장 마감 전에 돌리면 오늘 데이터가 아직 없어서 뒤쪽 셀에서 빈 결과나 오류가 날 수 있다. 마감 후에 실행한다.
"""
def get_last_biz():
    f = os.path.join(CACHE_DIR, f'last_biz_{date.today():%Y%m%d}.txt')
    if os.path.exists(f):
        with open(f) as fp:
            return fp.read().strip()

    from pykrx import stock            # 가져올 때 KRX 로그인이 일어난다 (KRX_ID / KRX_PW 사용)
    today = datetime.today().strftime('%Y%m%d')
    try:
        # 방법 1: pykrx 가 가까운 영업일을 계산
        v = safe_call(lambda: stock.get_nearest_business_day_in_a_week(today))
    except Exception:
        # 방법 2: 삼성전자 일봉의 마지막 날짜. 실제 거래 데이터가 있는 마지막 날이라 가장 확실하다
        start = (datetime.today() - timedelta(days=10)).strftime('%Y%m%d')
        df = safe_call(lambda: stock.get_market_ohlcv(start, today, '005930'))
        v = df.index[-1].strftime('%Y%m%d')

    with open(f, 'w') as fp:
        fp.write(v)
    return v


last_biz = get_last_biz()
print('최근 영업일:', last_biz)


#%% [수집-3] KRX 업종분류 현황 (코스피 / 코스닥)
"""
KRX 정보데이터시스템에서 csv 를 받는 과정은 2단계다.
(1) 원하는 항목을 쿼리로 보내 OTP(generate.cmd)를 받는다.
    -> 개발자 도구(F12) Network 탭의 generate.cmd 에서 request URL 이 제출 주소, payload 가 쿼리 항목이다.
(2) 받은 OTP 를 download.cmd 에 제출하면 csv 가 내려온다.
Referer 를 같이 보내야 한다. (1)의 과정을 거치지 않고 바로 (2)에 보내면 봇으로 본다.

아래 krx_csv() 는 이 두 단계를 묶은 함수다.
- 같은 이름의 캐시 파일이 있으면 KRX 에 접속하지 않고 그 파일을 읽는다.
- OTP 대신 LOGOUT 이나 HTML 이 오면 로그인이 풀렸거나 차단된 것이므로 멈춘다.
- 받은 내용이 HTML 이면 캐시에 저장하지 않는다 (잘못된 파일이 캐시에 남는 것을 막기 위해).
"""
gen_otp_url = 'https://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd'
down_url = 'https://data.krx.co.kr/comm/fileDn/download_csv/download.cmd'
headers = {'Referer': 'https://data.krx.co.kr/contents/MDC/MDI/mdiLoader'}

krx = None            # pykrx 가 로그인해 둔 연결. 캐시가 없어서 접속해야 할 때만 만든다


def krx_csv(query, cache_name, **read_kw):
    global krx
    path = os.path.join(CACHE_DIR, cache_name + '.csv')

    if not os.path.exists(path):
        if krx is None:
            from pykrx.website.comm.auth import get_auth_session
            krx = get_auth_session()

        def _get():
            otp = krx.post(gen_otp_url, headers=headers, data=query, timeout=30).text
            if 'LOGOUT' in otp or '<' in otp:
                raise BlockedError('OTP 를 못 받았습니다 (로그인 풀림 또는 차단 의심): ' + otp[:100])
            res = krx.post(down_url, headers=headers, data={'code': otp}, timeout=60)
            res.raise_for_status()
            if res.content.lstrip()[:1] == b'<':
                raise BlockedError('csv 대신 HTML 이 왔습니다 (차단 의심)')
            return res.content

        content = safe_call(_get)
        with open(path, 'wb') as f:
            f.write(content)
        time.sleep(random.uniform(1, 2))            # 연속 요청 사이에 잠깐 쉬기

    return pd.read_csv(path, encoding='EUC-KR', **read_kw)   # KRX csv 는 EUC-KR


# 코스피
gen_otp_stk = {
    'mktId': 'STK',            # STK 는 코스피, KSQ 는 코스닥
    'trdDd': last_biz,         # 최근 영업일
    'money': '1',
    'csvxls_isNo': 'false',
    'name': 'fileDown',
    'url': 'dbms/MDC/STAT/standard/MDCSTAT03901'
}
sector_stk = krx_csv(gen_otp_stk, f'krx_sector_STK_{last_biz}', dtype={'종목코드': str})

# 코스닥 (코스피와 같은 코드, mktId 만 KSQ)
gen_otp_ksq = dict(gen_otp_stk, mktId='KSQ')
sector_ksq = krx_csv(gen_otp_ksq, f'krx_sector_KSQ_{last_biz}', dtype={'종목코드': str})

krx_sector = pd.concat([sector_stk, sector_ksq], ignore_index=True)   # 두 데이터 합치기
krx_sector['종목명'] = krx_sector['종목명'].str.strip()                  # 종목명 공백 제거
krx_sector['기준일'] = last_biz

print(sector_stk.shape, sector_ksq.shape, krx_sector.shape)
krx_sector.head()


#%% [수집-4] KRX 개별종목 지표 (PER / PBR / 배당수익률 - [12021])
"""
위와 같은 방식이고 요청하는 쿼리 값만 다르다.
다른 항목이 필요하면 개발자 도구에서 csv 버튼을 눌렀을 때 나오는 generate.cmd 의 payload 를 보고 맞춘다.
"""
gen_otp_ind = {
    'searchType': '1',
    'mktId': 'ALL',
    'trdDd': last_biz,
    'csvxls_isNo': 'false',
    'name': 'fileDown',
    'url': 'dbms/MDC/STAT/standard/MDCSTAT03501'
}
krx_ind = krx_csv(gen_otp_ind, f'krx_ind_{last_biz}', dtype={'종목코드': str})
krx_ind['종목명'] = krx_ind['종목명'].str.strip()
krx_ind['기준일'] = last_biz

krx_ind.head()


#%% [수집-5] 데이터 정리 + kor_ticker 저장
diff = list(set(krx_sector['종목명']).symmetric_difference(set(krx_ind['종목명'])))
print(diff)
# 하나의 데이터에만 있는 종목이다. 리츠, 선박/광물 펀드, 해외종목처럼 일반적이지 않은 경우가 많다.
# 그래서 두 데이터를 합친 뒤 '기타'로 구분한다.

kor_ticker = pd.merge(krx_sector,
                      krx_ind,
                      on=krx_sector.columns.intersection(krx_ind.columns).tolist(),
                      how='outer')
# 두 표에 공통으로 있는 열(종목코드, 종목명, 종가, 대비, 등락률 등)을 기준으로 합친다.

# 스팩: '스팩' 이 들어가거나 '제n호' 형태. 우선주: 종목코드 끝자리가 0 이 아님. 리츠: 이름이 리츠로 끝남(중간에 들어간 경우 제외)
kor_ticker['종목구분'] = np.where(kor_ticker['종목명'].str.contains('스팩|제[0-9]+호'), '스팩',
                              np.where(kor_ticker['종목코드'].str[-1:] != '0', '우선주',
                                       np.where(kor_ticker['종목명'].str.endswith('리츠'), '리츠',
                                                np.where(kor_ticker['종목명'].isin(diff), '기타',
                                                         '보통주'))))

kor_ticker = kor_ticker.reset_index(drop=True)
kor_ticker.columns = kor_ticker.columns.str.replace(' ', '')       # 열 이름 공백 삭제
kor_ticker = kor_ticker[['종목코드', '종목명', '시장구분', '종가', '시가총액', '기준일', 'EPS',
                         'PER', 'BPS', '주당배당금', '종목구분']]
kor_ticker = kor_ticker.replace({np.nan: None})                    # SQL 에는 NaN 이 안 들어가므로 None 으로
kor_ticker['기준일'] = pd.to_datetime(kor_ticker['기준일'])           # yyyymmdd -> yyyy-mm-dd

print(kor_ticker['종목구분'].value_counts())

engine, con = db_connect()
mycursor = con.cursor()

query = """
    insert into kor_ticker(종목코드,종목명,시장구분,종가,시가총액,기준일,EPS,PER,BPS,주당배당금,종목구분)
    values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) as new
    on duplicate key update
    종목명 = new.종목명, 시장구분 = new.시장구분, 종가 = new.종가, 시가총액 = new.시가총액,
    EPS = new.EPS, PER = new.PER, BPS = new.BPS, 주당배당금 = new.주당배당금, 종목구분 = new.종목구분;
"""
mycursor.executemany(query, kor_ticker.values.tolist())
con.commit()

engine.dispose()
con.close()
print('kor_ticker 저장 완료:', len(kor_ticker))


#%% [수집-6] WICS 기준 섹터 정보
"""
GICS 는 KRX 가 공개하지만 독점적 지적재산이라 쓰기 곤란하다. 그래서 FnGuide 의 WICS 를 쓴다.
페이지: http://www.wiseindex.com/Index  (WISE Sector Index -> WICS -> 섹터 선택 -> Components)
개발자 도구 Network 탭의 getindexcomponent 요청이 데이터 주소다.
  https://www.wiseindex.com/Index/GetIndexComponets?ceil_yn=0&dt=20261008&sec_cd=G10
  ceil_yn=0 : 실링 여부(0 은 비실링) / dt : 조회일자 / sec_cd : 섹터 코드
결과가 JSON 이라 requests 의 json() 으로 바로 받을 수 있다.
날짜는 위에서 구한 last_biz 를 쓴다. 장이 끝나기 전에는 데이터가 없어서 오류가 날 수 있다.
같은 날 다시 실행하면 캐시 파일을 읽는다.
"""
def get_wics(base_date):
    path = os.path.join(CACHE_DIR, f'wics_{base_date}.csv')
    if os.path.exists(path):
        return pd.read_csv(path, dtype={'IDX_CD': str, 'CMP_CD': str})

    sector_code = ['G25', 'G35', 'G50', 'G40', 'G10', 'G20', 'G55', 'G30', 'G15', 'G45']
    data_sector = []                                   # 섹터 정보가 들어갈 빈 리스트

    for i in tqdm(sector_code):
        url = f'https://www.wiseindex.com/Index/GetIndexComponets?ceil_yn=0&dt={base_date}&sec_cd={i}'
        data = safe_call(lambda: rq.get(url, timeout=30).json())
        data_sector.append(pd.json_normalize(data['list']))
        nap((1.5, 2.5))

    sector = pd.concat(data_sector, axis=0)
    if sector.empty:
        raise ValueError('WICS 데이터가 비어 있습니다 (장 마감 전이거나 날짜 문제)')
    sector = sector[['IDX_CD', 'CMP_CD', 'CMP_KOR', 'SEC_NM_KOR']].copy()
    sector.to_csv(path, index=False, encoding='utf-8-sig')
    return sector


sector = get_wics(last_biz)
sector['기준일'] = pd.to_datetime(last_biz)

engine, con = db_connect()
mycursor = con.cursor()

query = """
    insert into kor_sector (IDX_CD, CMP_CD, CMP_KOR, SEC_NM_KOR, 기준일)
    values(%s,%s,%s,%s,%s) as new
    on duplicate key update
    IDX_CD = new.IDX_CD, CMP_KOR = new.CMP_KOR, SEC_NM_KOR = new.SEC_NM_KOR
"""
mycursor.executemany(query, sector.values.tolist())
con.commit()

engine.dispose()
con.close()
print('kor_sector 저장 완료:', len(sector))


#%% [수집-7] 수정주가 (증분 + 이어받기)
"""
주가는 퀀트 투자와 백테스트에 꼭 필요하고, 백테스트에는 수정주가가 필요하다. (네이버 차트 사용)

처음에는 PRICE_YEARS 년치를 받고, 그 뒤로는 DB 의 마지막 날짜 이후만 받는다.
수정주가는 액면분할/배당 등이 생기면 과거 값이 바뀐다. 그래서 마지막 날짜보다 10일 앞부터 받아서
DB 에 있는 값과 비교하고, 다르면 그 종목만 전체 기간을 다시 받는다.
오늘 이미 처리한 종목은 crawl_log 를 보고 건너뛴다. 중간에 멈췄다가 다시 실행하면 이어서 받는다.
"""
engine, con = db_connect()
mycursor = con.cursor()

# 티커 리스트 (최근일 기준 보통주)
ticker_list = pd.read_sql("""
select * from kor_ticker
where 기준일 = (select max(기준일) from kor_ticker)
    and 종목구분 = '보통주';
""", con=engine)

today_str = date.today().strftime('%Y-%m-%d')
fr_full = (date.today() + relativedelta(years=-PRICE_YEARS)).strftime('%Y%m%d')   # 시작일
to = date.today().strftime('%Y%m%d')                                              # 종료일

# 오늘 이미 처리한 종목
done_today = set(pd.read_sql(f"""
select 종목코드 from crawl_log where 대상 = 'price' and 완료일 = '{today_str}';
""", con=engine)['종목코드'])

# 종목별 DB 의 마지막 날짜
last_map = pd.read_sql("""
select 종목코드, max(날짜) as 마지막 from kor_price group by 종목코드;
""", con=engine)
last_map['마지막'] = pd.to_datetime(last_map['마지막'])
last_map = last_map.set_index('종목코드')['마지막'].to_dict()

# 수정주가 비교용: 최근 40일 종가
recent = pd.read_sql(f"""
select 종목코드, 날짜, 종가 from kor_price
where 날짜 >= '{(date.today() - timedelta(days=40)).strftime('%Y-%m-%d')}';
""", con=engine)
recent['날짜'] = pd.to_datetime(recent['날짜'])
recent_map = recent.set_index(['종목코드', '날짜'])['종가'].to_dict()


def fetch_price(ticker, fr, to):
    """네이버 차트에서 일봉을 받아 정리한다. 결과 열: 날짜, 시가, 고가, 저가, 종가, 거래량, 종목코드"""
    url = f'https://fchart.stock.naver.com/siseJson.nhn?symbol={ticker}&requestType=1&startTime={fr}&endTime={to}&timeframe=day'

    def _get():
        r = rq.get(url, timeout=20)
        r.raise_for_status()
        return r.content

    data = safe_call(_get)
    data_price = pd.read_csv(BytesIO(data))

    price = data_price.iloc[:, 0:6].copy()                     # 날짜, 가격, 거래량 해당 열만
    price.columns = ['날짜', '시가', '고가', '저가', '종가', '거래량']
    price = price.dropna()
    price['날짜'] = price['날짜'].str.extract(r'(\d+)')          # 날짜 열에서 숫자만 추출
    price['날짜'] = pd.to_datetime(price['날짜'])
    price['종목코드'] = ticker
    if len(price) == 0:
        raise NoData(ticker)
    return price


query_price = """
    insert into kor_price(날짜, 시가, 고가, 저가, 종가, 거래량, 종목코드)
    values(%s,%s,%s,%s,%s,%s,%s) as new
    on duplicate key update
    시가 = new.시가, 고가 = new.고가, 저가 = new.저가,
    종가 = new.종가, 거래량 = new.거래량;
"""

todo = [t for t in ticker_list['종목코드'] if t not in done_today]
print(f'대상 {len(ticker_list)}개 중 오늘 처리 완료 {len(done_today)}개 -> 이번에 받을 종목 {len(todo)}개')

error_list = []
br = Breaker()

try:
    for ticker in tqdm(todo):
        try:
            last = last_map.get(ticker)

            if last is None:                                           # DB 에 처음 들어오는 종목
                price = fetch_price(ticker, fr_full, to)
            else:
                fr = (last - timedelta(days=10)).strftime('%Y%m%d')
                price = fetch_price(ticker, fr, to)

                # 겹치는 구간의 종가가 DB 와 다르면 수정주가가 바뀐 것이므로 전체 기간을 다시 받는다
                changed = False
                ov = price[price['날짜'] <= last]
                for d, c in zip(ov['날짜'], ov['종가']):
                    old = recent_map.get((ticker, d))
                    if old is not None and abs(old - c) > abs(old) * 0.001:
                        changed = True
                        break
                if changed:
                    nap(SLEEP_PRICE)
                    price = fetch_price(ticker, fr_full, to)

            mycursor.executemany(query_price, price.values.tolist())
            con.commit()
            log_done(mycursor, con, 'price', ticker, 'ok')
            br.ok()

        except BlockedError:
            raise
        except NoData:
            log_done(mycursor, con, 'price', ticker, 'nodata')         # 데이터가 없는 종목은 오늘 다시 시도하지 않는다
        except Exception as e:
            print(ticker, type(e).__name__, str(e)[:80])
            error_list.append(ticker)
            br.fail()                                                  # 연속 실패가 쌓이면 BlockedError

        nap(SLEEP_PRICE)

except BlockedError as e:
    print('중단:', e)

finally:
    engine.dispose()
    con.close()

print('실패 종목 수:', len(error_list))


#%% [수집-8] 재무제표 (FnGuide, 이어받기)
"""
가치투자의 핵심 데이터다. FnGuide 의 Company Guide (http://comp.fnguide.com/) 에서 받는다.
재무제표 탭에는 포괄손익계산서, 재무상태표, 현금흐름표가 있고, 주소 뒤쪽은 필요 없어서 티커만 바꿔 쓴다.

최근 FS_REFRESH_DAYS 일 안에 받은 종목은 crawl_log 를 보고 건너뛴다.
전 종목을 처음 받으면 몇 시간 걸린다. 중간에 멈춰도(Ctrl+C) 다시 실행하면 이어서 받는다.
"""
fnguide_headers = {'User-Agent': 'Mozilla/5.0'}


def fnguide_fs(name, ticker, freq):
    """name: getFinIncome(손익) / getFinBalance(재무상태) / getFinCashFlow(현금흐름), freq: 'Y'=연간, 'Q'=분기
       결과: 첫 열 'IFRS(연결)' + 기간 열들"""
    nap((0.2, 0.5))                                        # 한 종목에 6번 요청하므로 요청 사이에도 잠깐 쉰다
    u = f'https://wcomp.fnguide.com/CompanyInfo/{name}'

    def _get():
        r = rq.get(u, params={'cmp_cd': ticker, 'freq_typ': freq, 'consol_typ': 'C'},
                   headers=fnguide_headers, timeout=20)
        r.raise_for_status()
        return r.json()['dataset']

    o = safe_call(_get)

    if isinstance(o, list) and len(o) > 0 and 'data' in o[0]:
        o = o[0]

    if not (isinstance(o, dict) and 'data' in o and 'header' in o):   # 구조가 다르면 보여주고 멈춤
        print(type(o), (list(o.keys()) if isinstance(o, dict) else str(o)[:300]))
        raise ValueError(f'{name} {freq}: 구조 확인 필요')

    df = pd.DataFrame(o['data'])
    rename = {h['CD']: h['YYMM'].replace(' (최근분기)', '') for h in o['header']}   # VAL1, VAL2... -> 2023/12 ...
    df = df.rename(columns=rename)
    df['NAME'] = df['NAME'].str.strip()
    df = df[['NAME'] + list(rename.values())]
    df = df.rename(columns={'NAME': 'IFRS(연결)'})
    for c in rename.values():
        df[c] = pd.to_numeric(df[c], errors='coerce')      # 숫자 변환 (글자는 NaN)
    return df


def fnguide_tables(ticker):
    """표 10.1 순서대로 6개: 0 손익(연간) 1 손익(분기) 2 재무상태(연간) 3 재무상태(분기) 4 현금흐름(연간) 5 현금흐름(분기)"""
    return [fnguide_fs('getFinIncome', ticker, 'Y'),
            fnguide_fs('getFinIncome', ticker, 'Q'),
            fnguide_fs('getFinBalance', ticker, 'Y'),
            fnguide_fs('getFinBalance', ticker, 'Q'),
            fnguide_fs('getFinCashFlow', ticker, 'Y'),
            fnguide_fs('getFinCashFlow', ticker, 'Q')]


def clean_fs(df, ticker, frequency):
    """입력: 데이터프레임, 티커, 공시구분(연간/분기)"""
    df = df[~df.loc[:, ~df.columns.isin(['계정'])].isna().all(axis=1)]         # 모든 연도가 NaN 인 항목 제외
    df = df.drop_duplicates(['계정'], keep='first')                            # 계정명이 중복되면 첫 번째만
    df = pd.melt(df, id_vars='계정', var_name='기준일', value_name='값')         # 열로 긴 데이터를 행으로 긴 데이터로
    df = df[~pd.isnull(df['값'])]                                              # 계정값이 없는 항목 제외
    df['계정'] = df['계정'].replace({'계산에 참여한 계정 펼치기': ''}, regex=True)   # 페이지의 [+] 부분 글자 제거
    df['기준일'] = pd.to_datetime(df['기준일'],
                               format='%Y/%m') + pd.tseries.offsets.MonthEnd()  # 월말 일자를 붙인다
                                                # (책은 format='%Y-%m' 이지만 이 페이지의 열 이름은 '2023/12' 형태라 '%Y/%m')
    df['종목코드'] = ticker
    df['공시구분'] = frequency
    return df


def collect_fs(ticker):
    """한 종목의 연간 + 분기 재무제표를 정리해서 돌려준다."""
    url = f'https://wcomp.fnguide.com/CompanyInfo/Finance?c_id=AA&menu_type=01&cmp_cd={ticker}'

    data = fnguide_tables(ticker)

    # 연간 데이터 (포괄손익계산서에서는 '전년동기' 열을 뺀다)
    data_fs_y = pd.concat([
        data[0].iloc[:, ~data[0].columns.str.contains('전년동기')], data[2], data[4]])
    data_fs_y = data_fs_y.rename(columns={data_fs_y.columns[0]: '계정'})

    # 결산월 찾기: 결산 전에 크롤링하면 연간 표에 분기 열이 섞여 들어오므로 결산월 열만 남긴다
    page_data = safe_call(lambda: rq.get(url, headers=fnguide_headers, timeout=20))
    page_data_html = BeautifulSoup(page_data.content, 'html.parser')

    fiscal_data = page_data_html.select('div.corp_group1 > h2')
    if len(fiscal_data) > 1:
        fiscal_data_text = re.findall('[0-9]+', fiscal_data[1].text)    # 'n월 결산' 에서 숫자만
    else:                                                                # (예비) 재무상태표(연간)의 'yyyy/mm' 열에서 월만 뽑기
        fiscal_data_text = list({c[-2:] for c in data[2].columns[1:] if re.fullmatch(r'\d{4}/\d{2}', str(c))})

    data_fs_y = data_fs_y.loc[:, (data_fs_y.columns == '계정') | (
        data_fs_y.columns.str[-2:].isin(fiscal_data_text))]
    data_fs_y_clean = clean_fs(data_fs_y, ticker, 'y')

    # 분기 데이터 (결산월 선택은 필요 없다)
    data_fs_q = pd.concat([
        data[1].iloc[:, ~data[1].columns.str.contains('전년동기')], data[3], data[5]])
    data_fs_q = data_fs_q.rename(columns={data_fs_q.columns[0]: '계정'})
    data_fs_q_clean = clean_fs(data_fs_q, ticker, 'q')

    data_fs_bind = pd.concat([data_fs_y_clean, data_fs_q_clean])
    if len(data_fs_bind) == 0:
        raise NoData(ticker)
    return data_fs_bind


engine, con = db_connect()
mycursor = con.cursor()

# 티커 리스트 (시가총액 큰 종목부터. 중간에 멈춰도 큰 종목은 먼저 들어와 있다)
ticker_list = pd.read_sql("""
select * from kor_ticker
where 기준일 = (select max(기준일) from kor_ticker)
    and 종목구분 = '보통주'
order by 시가총액 desc;
""", con=engine)

# 최근 FS_REFRESH_DAYS 일 안에 정상 수집한 종목 / 데이터가 없다고 확인된 종목
since = (date.today() - timedelta(days=FS_REFRESH_DAYS)).strftime('%Y-%m-%d')
done_fs = set(pd.read_sql(f"""
select 종목코드 from crawl_log
where 대상 = 'fs' and 완료일 >= '{since}';
""", con=engine)['종목코드'])

query = """
    insert into kor_fs (계정, 기준일, 값, 종목코드, 공시구분)
    values (%s,%s,%s,%s,%s) as new
    on duplicate key update
    값=new.값
"""

todo = [t for t in ticker_list['종목코드'] if t not in done_fs]
print(f'대상 {len(ticker_list)}개 중 최근 {FS_REFRESH_DAYS}일 내 처리 {len(done_fs)}개 -> 이번에 받을 종목 {len(todo)}개')

error_list = []
br = Breaker()

try:
    for ticker in tqdm(todo):
        try:
            data_fs_bind = collect_fs(ticker)

            mycursor.executemany(query, data_fs_bind.values.tolist())
            con.commit()
            log_done(mycursor, con, 'fs', ticker, 'ok')
            br.ok()

        except BlockedError:
            raise
        except NoData:
            log_done(mycursor, con, 'fs', ticker, 'nodata')
        except Exception as e:
            print(ticker, type(e).__name__, str(e)[:80])
            error_list.append(ticker)
            br.fail()

        nap(SLEEP_FS)

except BlockedError as e:
    print('중단:', e)

finally:
    engine.dispose()
    con.close()

print('실패 종목 수:', len(error_list))
# 시간이 지나 다시 실행하면 upsert 라서 수정된 재무는 update, 새로 나온 재무는 insert 된다.


#%% [수집-9] 수집 결과 확인
engine, con = db_connect()

print(pd.read_sql("""
select 'kor_ticker' as 테이블, count(*) as 행수, max(기준일) as 최근일 from kor_ticker
union all
select 'kor_sector', count(*), max(기준일) from kor_sector
union all
select 'kor_price', count(*), max(날짜) from kor_price
union all
select 'kor_fs', count(*), max(기준일) from kor_fs;
""", con=engine))

print(pd.read_sql("""
select 대상, 상태, count(*) as 종목수 from crawl_log group by 대상, 상태;
""", con=engine))

engine.dispose()
con.close()