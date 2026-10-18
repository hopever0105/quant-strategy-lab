#%% [DART-0] 안내
"""
장기 데이터 수집: OpenDART 재무제표(2015년 이후) + 주가 과거 보강   파일명: 3_장기데이터_수집.py

[왜 필요한가]
FnGuide 페이지는 연간 재무가 4~5년치뿐이라 그레이엄 엄격 버전(10년 흑자/성장)이 계산되지 않는다.
OpenDART 는 2015년 이후 사업보고서를 주고, 사업보고서 하나에 당기/전기/전전기 3개 연도가 들어 있다.
그래서 3년 간격으로 요청하면 2013년분까지 얻을 수 있다. (예: 2025, 2022, 2019, 2016 보고서 + 2015 보고서)

[실행 전에]
1) https://opendart.fss.or.kr 에서 회원가입 후 인증키(40자리)를 신청한다.
2) 명령 프롬프트에서  setx DART_KEY "인증키"  후 Spyder 를 완전히 껐다 켠다. (또는 실행할 때 입력)
3) 1_자료수집.py 의 [수집-5] 까지 끝나 kor_ticker 가 있어야 한다.
   상장폐지 종목도 같이 받으려면 4_상장폐지_수집.py 의 [상폐-3] 까지 먼저 실행한다.

[저장 위치]
- dart_fs   : 종목코드, 계정(짧은 이름), 기준일, 값(억원)   -> 2_그레이엄 파일의 FS_SOURCE = 'dart' 가 읽는다
- dart_corp : 종목코드 <-> OpenDART 고유번호(corp_code), 결산월
- crawl_log : 종목별 진행 상황 (중간에 멈춰도 이어받기)

[공식 문서(opendart.fss.or.kr 가이드)로 확인한 것]
- fnlttSinglAcntAll.json : 인자 crtfc_key, corp_code, bsns_year, reprt_code(사업보고서 11011), fs_div(CFS 연결 / OFS 별도)
- bsns_year 는 2015년 이후부터 제공
- 응답 항목 : sj_div(BS, IS, CIS, CF, SCE), account_id, account_nm, thstrm_amount, frmtrm_amount, bfefrmtrm_amount
  (bfefrmtrm = 전전기, 사업보고서에만 나옴)
- 상태코드 : 000 정상 / 013 데이터 없음 / 020 요청 제한 초과(보통 2만 건 이상) / 010,011,012,901 키 문제
- corpCode.xml 은 zip 으로 오고 corp_code, corp_name, stock_code, modify_date 가 들어 있다
- company.json 에 acc_mt(결산월) 가 있다

[확인 안 된 것]
- 하루 호출 한도의 정확한 기준 (20,000건은 문서의 오류 설명에서 가져온 숫자)
- corpCode.xml 에 상장폐지 기업이 들어 있는지, 폐지 기업의 stock_code 가 남아 있는지
- 아래 MAP 의 account_id (IFRS 표준 이름을 기억해서 쓴 것). 회사마다 다르게 쓸 수 있어서 [DART-5] 로 커버리지를 확인한다
- 12월 결산이 아닌 회사의 bsns_year 가 어느 연도를 가리키는지. 그래서 12월 결산 회사만 받는다
- 현금흐름표에서 CAPEX 의 부호. 절댓값으로 저장한다
"""


#%% [DART-1] 설정 + 공통 도구 + 테이블 준비
import os
import io
import re
import time
import random
import getpass
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from urllib.parse import quote_plus
from io import BytesIO

import numpy as np
import pandas as pd
import requests as rq
import pymysql
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine, inspect
from tqdm import tqdm

CACHE_DIR = os.path.join(os.path.expanduser('~'), 'quant_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

FIRST_YEAR = 2015            # OpenDART 가 재무제표를 주는 첫 사업연도
DAILY_LIMIT = 19000          # 하루에 보낼 최대 호출 수(코드가 스스로 멈추는 선). 서버 한도는 확인 안 됨
SLEEP_DART = (0.1, 0.3)
MAX_CONSEC_FAIL = 10
DART_REFRESH_DAYS = 120      # 이 기간 안에 받은 종목은 건너뛴다 (연간 재무는 1년에 한 번 바뀐다)

DB_USER = 'root'
DB_HOST = '127.0.0.1'
DB_NAME = 'stock_db'
DB_PW = os.environ.get('DB_PW') or getpass.getpass('DB password: ')
DART_KEY = os.environ.get('DART_KEY') or getpass.getpass('OpenDART 인증키: ')


def db_connect():
    engine = create_engine(
        f'mysql+pymysql://{DB_USER}:{quote_plus(DB_PW)}@{DB_HOST}:3306/{DB_NAME}')
    con = pymysql.connect(user=DB_USER, password=DB_PW, host=DB_HOST, database=DB_NAME, charset='utf8')
    return engine, con


class BlockedError(Exception):
    """연속 실패 등 차단이 의심될 때"""


class NoData(Exception):
    """응답은 정상인데 데이터가 없는 경우"""


class NonDecember(Exception):
    """12월 결산이 아닌 회사"""


class DartLimit(Exception):
    """하루 호출 한도에 닿았을 때"""


class DartFatal(Exception):
    """인증키/접근 문제처럼 계속 시도해도 소용없는 오류"""


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


# ---- 오늘 보낸 호출 수를 파일에 적어 둔다 (같은 날 다시 실행해도 이어서 센다) ----
CALL_FILE = os.path.join(CACHE_DIR, f'dart_calls_{date.today():%Y%m%d}.txt')


def read_calls():
    try:
        with open(CALL_FILE) as f:
            return int(f.read().strip())
    except Exception:
        return 0


n_calls = read_calls()
FATAL_STATUS = {'010', '011', '012', '101', '800', '901'}


def dart_json(name, **params):
    """OpenDART JSON 호출. 정상(000)이면 dict, 데이터 없음(013)이면 None."""
    global n_calls
    if n_calls >= DAILY_LIMIT:
        raise DartLimit(f'오늘 호출 {n_calls}건에 닿아 멈춥니다. 내일 같은 셀을 다시 실행하면 이어받습니다.')

    def _get():
        r = rq.get(f'https://opendart.fss.or.kr/api/{name}.json',
                   params=dict(crtfc_key=DART_KEY, **params), timeout=30)
        r.raise_for_status()
        return r.json()

    js = safe_call(_get)
    n_calls += 1
    with open(CALL_FILE, 'w') as f:
        f.write(str(n_calls))
    nap(SLEEP_DART)

    st = js.get('status')
    if st == '000':
        return js
    if st == '013':
        return None
    if st == '020':
        raise DartLimit('요청 제한 초과(020). 내일 다시 실행하세요: ' + str(js.get('message')))
    if st in FATAL_STATUS:
        raise DartFatal(f'{st}: {js.get("message")}')
    raise RuntimeError(f'DART 오류 {st}: {js.get("message")}')


DDL = [
"""
create table if not exists dart_corp
(
    corp_code varchar(8) not null,
    종목코드 varchar(6),
    corp_name varchar(100),
    acc_mt varchar(2),
    primary key(corp_code)
);
""",
"""
create table if not exists dart_fs
(
    종목코드 varchar(6) not null,
    계정 varchar(20) not null,
    기준일 date not null,
    값 double,
    primary key(종목코드, 계정, 기준일)
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
print('테이블 준비 완료 / 오늘 보낸 호출 수:', n_calls)


#%% [DART-2] 종목코드 <-> 고유번호(corp_code) 매핑
"""
corpCode.xml 은 zip 파일로 온다. 압축 안의 xml 에서 corp_code, corp_name, stock_code 를 읽는다.
안의 태그 구조는 문서에 자세히 없어서, corp_code 태그를 가진 요소를 모두 한 행으로 읽게 했다.
"""
def load_corp_codes():
    path = os.path.join(CACHE_DIR, f'dart_corp_{date.today():%Y%m%d}.xml')
    if not os.path.exists(path):
        r = safe_call(lambda: rq.get('https://opendart.fss.or.kr/api/corpCode.xml',
                                     params={'crtfc_key': DART_KEY}, timeout=60))
        r.raise_for_status()
        try:
            z = zipfile.ZipFile(io.BytesIO(r.content))
        except zipfile.BadZipFile:
            raise RuntimeError('corpCode 응답이 zip 이 아닙니다 (인증키 문제일 수 있음): ' + r.text[:200])
        with open(path, 'wb') as f:
            f.write(z.read(z.namelist()[0]))

    rows = []
    for e in ET.parse(path).getroot().iter():
        if e.find('corp_code') is not None:
            rows.append({k: (e.findtext(k) or '').strip()
                         for k in ['corp_code', 'corp_name', 'stock_code', 'modify_date']})
    return pd.DataFrame(rows)


corp_all = load_corp_codes()
print('corpCode 전체 행:', len(corp_all))

corp = corp_all[corp_all['stock_code'].str.len() == 6].copy()
corp = corp.sort_values('modify_date').drop_duplicates('stock_code', keep='last')   # 같은 종목코드면 최신 것
print('종목코드가 있는 회사:', len(corp))

engine, con = db_connect()
mycursor = con.cursor()

query = """
    insert into dart_corp (corp_code, 종목코드, corp_name)
    values (%s,%s,%s) as new
    on duplicate key update 종목코드 = new.종목코드, corp_name = new.corp_name
"""
mycursor.executemany(query, corp[['corp_code', 'stock_code', 'corp_name']].values.tolist())
con.commit()

# 우리 종목이 얼마나 매핑되는지
tk = pd.read_sql("""
select 종목코드 from kor_ticker
where 기준일 = (select max(기준일) from kor_ticker) and 종목구분 = '보통주';
""", con=engine)
print(f'kor_ticker 보통주 {len(tk)}개 중 고유번호가 있는 종목: {tk["종목코드"].isin(corp["stock_code"]).sum()}개')

if 'kor_delisted' in inspect(engine).get_table_names():
    dl = pd.read_sql('select 종목코드 from kor_delisted;', con=engine)
    n_ok = dl['종목코드'].isin(corp['stock_code']).sum()
    print(f'kor_delisted {len(dl)}개 중 고유번호가 있는 종목: {n_ok}개')
    print('-> 0 에 가까우면 corpCode.xml 에 상장폐지 기업의 종목코드가 없다는 뜻이라 폐지 종목 재무는 받을 수 없습니다.')

engine.dispose()
con.close()


#%% [DART-3] 계정 매핑 + 파싱 함수
"""
이 코드가 쓰는 짧은 계정 이름 -> (재무제표 구분, XBRL 표준계정ID 후보, 계정명 후보)
먼저 표준계정ID 로 찾고, 없으면 계정명으로 찾는다. ID 는 IFRS 이름을 기억해 쓴 것이라 확인이 필요하다.
"""
MAP = {
    'sales':  (('IS', 'CIS'), ['ifrs-full_Revenue'],
               ['매출액', '수익(매출액)', '영업수익', '매출']),
    'op':     (('IS', 'CIS'), ['dart_OperatingIncomeLoss'],
               ['영업이익', '영업이익(손실)']),
    'ni':     (('IS', 'CIS'), ['ifrs-full_ProfitLoss'],
               ['당기순이익', '당기순이익(손실)', '연결당기순이익', '당기순손익']),
    'assets': (('BS',), ['ifrs-full_Assets'], ['자산총계']),
    'liab':   (('BS',), ['ifrs-full_Liabilities'], ['부채총계']),
    'equity': (('BS',), ['ifrs-full_Equity'], ['자본총계']),
    'ca':     (('BS',), ['ifrs-full_CurrentAssets'], ['유동자산']),
    'cl':     (('BS',), ['ifrs-full_CurrentLiabilities'], ['유동부채']),
    'ltd':    (('BS',), ['ifrs-full_NoncurrentLiabilities'], ['비유동부채']),
    'cfo':    (('CF',), ['ifrs-full_CashFlowsFromUsedInOperatingActivities'],
               ['영업활동현금흐름', '영업활동으로인한현금흐름', '영업활동순현금흐름']),
    'capex':  (('CF',), ['ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities',
                         'ifrs-full_PurchaseOfPropertyPlantAndEquipment'],
               ['유형자산의취득', '유형자산취득']),
}


def norm(s):
    return str(s).replace(' ', '').replace('\u00a0', '')


AMT_COLS = ('thstrm_amount', 'frmtrm_amount', 'bfefrmtrm_amount')


def to_num(v):
    """'9,999,999,999' 같은 문자열을 숫자로. 비어 있거나 '-' 면 NaN, '(1,000)' 은 음수로 본다."""
    if v is None:
        return np.nan
    s = str(v).replace(',', '').strip()
    if s in ('', '-'):
        return np.nan
    neg = s.startswith('(') and s.endswith(')')
    if neg:
        s = s[1:-1]
    try:
        return -float(s) if neg else float(s)
    except ValueError:
        return np.nan


def has_value(it):
    return any(not np.isnan(to_num(it.get(c))) for c in AMT_COLS)


def pick_item(items, stmts, ids, names):
    """재무제표 구분 순서대로, 표준계정ID -> 계정명 순으로 찾는다. 금액이 하나라도 있는 첫 번째 항목을 쓴다."""
    names = {norm(n) for n in names}
    for st in stmts:
        for it in items:
            if it.get('sj_div') == st and it.get('account_id') in ids and has_value(it):
                return it
    for st in stmts:
        for it in items:
            if it.get('sj_div') == st and norm(it.get('account_nm', '')) in names and has_value(it):
                return it
    return None


def parse_report(items, bsns_year):
    """사업보고서 하나에서 당기/전기/전전기 3개 연도의 값을 꺼낸다. 금액은 억원으로 바꾼다."""
    out = []
    for key, (stmts, ids, names) in MAP.items():
        it = pick_item(items, stmts, ids, names)
        if it is None:
            continue
        for col, back in (('thstrm_amount', 0), ('frmtrm_amount', 1), ('bfefrmtrm_amount', 2)):
            v = to_num(it.get(col))
            if np.isnan(v):
                continue
            if key == 'capex':
                v = abs(v)                      # 부호 표기가 확인되지 않아 절댓값으로 저장
            out.append((key, date(int(bsns_year) - back, 12, 31), v / 100000000))
    return pd.DataFrame(out, columns=['계정', '기준일', '값'])


def request_years():
    """3년 간격으로 요청할 사업연도. 마지막에 FIRST_YEAR 를 붙여 2013년분까지 채운다."""
    today = date.today()
    end = today.year - 1 if today.month >= 4 else today.year - 2
    ys = list(range(end, FIRST_YEAR - 1, -3))
    if ys and ys[-1] - 2 > FIRST_YEAR - 2:
        ys.append(FIRST_YEAR)
    return ys


print('요청할 사업연도:', request_years())


#%% [DART-4] 전 종목 재무 수집 (이어받기)
"""
종목당 호출: 기업개황 1번(결산월) + 사업보고서 약 5번(연결 재무제표). 연결이 없으면 별도(OFS)로 다시 요청한다.
2,400 종목이면 하루에 다 못 끝날 수 있다. DAILY_LIMIT 에 닿으면 멈추고, 내일 같은 셀을 다시 실행하면 이어받는다.
같은 연도가 여러 보고서에 나오면 가장 최근 보고서의 값을 쓴다 (정정/재작성 반영).
"""
def collect_dart(ticker, corp_code, acc_mt):
    """한 종목의 연간 재무를 모아 (결산월, 데이터프레임) 으로 돌려준다."""
    if not acc_mt:
        js = dart_json('company', corp_code=corp_code)
        if js is None:
            raise NoData(ticker)
        acc_mt = str(js.get('acc_mt') or '').strip()
    if acc_mt != '12':
        raise NonDecember(ticker)

    parts = []
    for yr in request_years():
        js = dart_json('fnlttSinglAcntAll', corp_code=corp_code, bsns_year=str(yr),
                       reprt_code='11011', fs_div='CFS')
        if js is None:                                              # 연결 재무제표가 없으면 별도로
            js = dart_json('fnlttSinglAcntAll', corp_code=corp_code, bsns_year=str(yr),
                           reprt_code='11011', fs_div='OFS')
        if js is None:
            continue
        d = parse_report(js['list'], yr)
        d['rpt_year'] = yr
        parts.append(d)

    if not parts:
        raise NoData(ticker)
    df = pd.concat(parts).sort_values('rpt_year')
    df = df.drop_duplicates(['계정', '기준일'], keep='last')        # 최근 보고서 값이 남는다
    return acc_mt, df[['계정', '기준일', '값']]


engine, con = db_connect()
mycursor = con.cursor()

codes = set(pd.read_sql("""
select 종목코드 from kor_ticker
where 기준일 = (select max(기준일) from kor_ticker) and 종목구분 = '보통주';
""", con=engine)['종목코드'])
if 'kor_delisted' in inspect(engine).get_table_names():
    codes |= set(pd.read_sql('select 종목코드 from kor_delisted;', con=engine)['종목코드'])

corp_map = pd.read_sql('select corp_code, 종목코드, acc_mt from dart_corp;', con=engine)
corp_map = corp_map.dropna(subset=['종목코드']).drop_duplicates('종목코드').set_index('종목코드')

since = (date.today() - timedelta(days=DART_REFRESH_DAYS)).strftime('%Y-%m-%d')
done = set(pd.read_sql(f"""
select 종목코드 from crawl_log where 대상 = 'dart' and 완료일 >= '{since}';
""", con=engine)['종목코드'])

no_corp = sorted(c for c in codes if c not in corp_map.index)
todo = [c for c in sorted(codes) if c in corp_map.index and c not in done]
print(f'대상 {len(codes)}개 / 고유번호 없음 {len(no_corp)}개 / 최근 처리 완료 {len(done)}개 -> 이번에 받을 종목 {len(todo)}개')

query = """
    insert into dart_fs (종목코드, 계정, 기준일, 값)
    values (%s,%s,%s,%s) as new
    on duplicate key update 값 = new.값
"""

error_list = []
br = Breaker()

try:
    for ticker in tqdm(todo):
        corp_code = corp_map.loc[ticker, 'corp_code']
        acc_mt = corp_map.loc[ticker, 'acc_mt']
        acc_mt = acc_mt if isinstance(acc_mt, str) else ''
        try:
            acc_mt, df = collect_dart(ticker, corp_code, acc_mt)
            df['종목코드'] = ticker
            mycursor.executemany(query, df[['종목코드', '계정', '기준일', '값']].values.tolist())
            mycursor.execute('update dart_corp set acc_mt = %s where corp_code = %s', (acc_mt, corp_code))
            con.commit()
            log_done(mycursor, con, 'dart', ticker, 'ok')
            br.ok()

        except (DartLimit, DartFatal):
            raise
        except NonDecember:
            mycursor.execute('update dart_corp set acc_mt = %s where corp_code = %s', ('xx', corp_code))
            con.commit()
            log_done(mycursor, con, 'dart', ticker, 'nondec')
        except NoData:
            log_done(mycursor, con, 'dart', ticker, 'nodata')
        except Exception as e:
            print(ticker, type(e).__name__, str(e)[:80])
            error_list.append(ticker)
            br.fail()

except (DartLimit, DartFatal, BlockedError) as e:
    print('중단:', e)

finally:
    engine.dispose()
    con.close()

print('실패 종목 수:', len(error_list), '/ 오늘 보낸 호출 수:', n_calls)


#%% [DART-5] 수집 결과 + 계정 매핑 커버리지 점검
engine, con = db_connect()

cov = pd.read_sql("""
select 계정, count(distinct 종목코드) as 종목수, min(기준일) as 시작, max(기준일) as 끝, count(*) as 행수
from dart_fs group by 계정 order by 종목수 desc;
""", con=engine)
print(cov.to_string(index=False))

n_tk = pd.read_sql('select count(distinct 종목코드) as n from dart_fs;', con=engine)['n'][0]
print('\n재무가 들어온 종목 수:', n_tk)
print('계정별 종목 수가 종목 수보다 많이 모자라면 해당 계정의 MAP 후보(account_id/계정명)가 맞지 않는 것입니다.')

yrs = pd.read_sql("""
select 종목코드, count(*) as 연수 from dart_fs where 계정 = 'ni' group by 종목코드;
""", con=engine)
print('\n순이익 기준 종목별 연간 재무 연수 분포')
print(yrs['연수'].value_counts().sort_index().to_string())

print(pd.read_sql("""
select 상태, count(*) as 종목수 from crawl_log where 대상 = 'dart' group by 상태;
""", con=engine))

engine.dispose()
con.close()


def show_accounts(ticker, bsns_year):
    """MAP 이 안 맞는 회사의 실제 계정 구성을 보고 싶을 때 쓴다. (호출 1~2건)"""
    engine, con = db_connect()
    cc = pd.read_sql('select corp_code from dart_corp where 종목코드 = %(t)s;',
                     con=engine, params={'t': ticker})['corp_code'][0]
    engine.dispose()
    con.close()
    js = dart_json('fnlttSinglAcntAll', corp_code=cc, bsns_year=str(bsns_year),
                   reprt_code='11011', fs_div='CFS')
    if js is None:
        print('데이터 없음')
        return None
    d = pd.DataFrame(js['list'])[['sj_div', 'account_id', 'account_nm', 'thstrm_amount']]
    print(d.to_string())
    return d
# 사용 예:  show_accounts('005930', 2023)


#%% [DART-6] (별도) 주가 과거 보강
"""
재무를 2013년부터 받아도 kor_price 가 5년치면 백테스트 기간이 5년에 묶인다.
DB 의 첫 날짜가 PRICE_FROM 보다 늦은 종목은 PRICE_FROM 부터 오늘까지 전체를 다시 받는다.
(일부만 이어 붙이면 그 사이에 액면분할 등이 있을 때 수정주가가 어긋나기 때문이다)
네이버 차트가 얼마나 먼 과거까지 주는지는 확인 안 됨. 먼저 한 종목으로 확인한다.
"""
PRICE_FROM = '2015-01-01'
SLEEP_PRICE = (0.3, 0.8)

test = None
try:
    r = safe_call(lambda: rq.get(
        'https://fchart.stock.naver.com/siseJson.nhn?symbol=005930&requestType=1'
        f'&startTime={PRICE_FROM.replace("-", "")}&endTime={date.today():%Y%m%d}&timeframe=day', timeout=20))
    test = pd.read_csv(BytesIO(r.content))
    first = re.findall(r'\d+', str(test.iloc[0, 0]))[0]
    print('삼성전자 테스트: 받은 첫 날짜', first, '/ 행 수', len(test))
except Exception as e:
    print('테스트 실패:', type(e).__name__, str(e)[:100])


def fetch_price(ticker, fr, to):
    """네이버 차트에서 일봉을 받아 정리한다. 결과 열: 날짜, 시가, 고가, 저가, 종가, 거래량, 종목코드"""
    url = (f'https://fchart.stock.naver.com/siseJson.nhn?symbol={ticker}&requestType=1'
           f'&startTime={fr}&endTime={to}&timeframe=day')

    def _get():
        r = rq.get(url, timeout=20)
        r.raise_for_status()
        return r.content

    data_price = pd.read_csv(BytesIO(safe_call(_get)))
    price = data_price.iloc[:, 0:6].copy()
    price.columns = ['날짜', '시가', '고가', '저가', '종가', '거래량']
    price = price.dropna()
    price['날짜'] = price['날짜'].astype(str).str.extract(r'(\d+)')
    price['날짜'] = pd.to_datetime(price['날짜'])
    price['종목코드'] = ticker
    if len(price) == 0:
        raise NoData(ticker)
    return price


engine, con = db_connect()
mycursor = con.cursor()

first_map = pd.read_sql('select 종목코드, min(날짜) as 첫날 from kor_price group by 종목코드;', con=engine)
first_map['첫날'] = pd.to_datetime(first_map['첫날'])
first_map = first_map.set_index('종목코드')['첫날']

since = (date.today() - timedelta(days=365)).strftime('%Y-%m-%d')
done_back = set(pd.read_sql(f"""
select 종목코드 from crawl_log where 대상 = 'pback' and 완료일 >= '{since}';
""", con=engine)['종목코드'])

limit = pd.Timestamp(PRICE_FROM) + pd.Timedelta(days=40)
todo = [t for t, d in first_map.items() if d > limit and t not in done_back]
print(f'과거 보강 대상 {len(todo)}개 (DB 첫 날짜가 {limit.date()} 보다 늦은 종목)')

query_price = """
    insert into kor_price(날짜, 시가, 고가, 저가, 종가, 거래량, 종목코드)
    values(%s,%s,%s,%s,%s,%s,%s) as new
    on duplicate key update
    시가 = new.시가, 고가 = new.고가, 저가 = new.저가, 종가 = new.종가, 거래량 = new.거래량;
"""

to = date.today().strftime('%Y%m%d')
fr = PRICE_FROM.replace('-', '')
error_list = []
br = Breaker()

try:
    for ticker in tqdm(todo):
        try:
            price = fetch_price(ticker, fr, to)
            mycursor.executemany(query_price, price.values.tolist())
            con.commit()
            log_done(mycursor, con, 'pback', ticker, 'ok')
            br.ok()
        except BlockedError:
            raise
        except NoData:
            log_done(mycursor, con, 'pback', ticker, 'nodata')
        except Exception as e:
            print(ticker, type(e).__name__, str(e)[:80])
            error_list.append(ticker)
            br.fail()
        nap(SLEEP_PRICE)
except BlockedError as e:
    print('중단:', e)
finally:
    engine.dispose()
    con.close()

print('실패 종목 수:', len(error_list))
#%% [확인] 삼성전자 액면분할 전후 종가
engine, con = db_connect()
print(pd.read_sql("""
select 날짜, 종목코드, 종가 from kor_price
where 종목코드 = '005930' and 날짜 in ('2015-01-02', '2018-04-27', '2018-05-04', '2018-05-08')
order by 날짜;
""", con=engine))
engine.dispose()
con.close()
#%% [DART-5b] nodata / nondec 종목 확인
# 재무가 안 들어온 종목이 어떤 종목인지 이름과 함께 본다. (읽기만 하고 DB는 바꾸지 않는다.)
engine, con = db_connect()

chk = pd.read_sql("""
select c.상태, c.종목코드,
       coalesce(t.종목명, d.종목명) as 종목명,
       t.시장구분,
       d.상장폐지일, d.사유
from crawl_log c
left join (select 종목코드, 종목명, 시장구분
           from kor_ticker
           where 기준일 = (select max(기준일) from kor_ticker)) t on c.종목코드 = t.종목코드
left join kor_delisted d on c.종목코드 = d.종목코드
where c.대상 = 'dart' and c.상태 in ('nodata', 'nondec')
order by c.상태, c.종목코드;
""", con=engine)

chk['구분'] = chk['상장폐지일'].notna().map({True: '상폐', False: '상장'})
print(chk.groupby(['상태', '구분']).size())
print()
pd.set_option('display.max_rows', 300)
pd.set_option('display.width', 200)
print(chk.to_string(index=False))

engine.dispose()
con.close()