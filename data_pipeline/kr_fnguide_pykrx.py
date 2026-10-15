"""1. 최근 영업일 기준 데이터 받기 """
"""
[pykrx란]
- 한국거래소(KRX) 데이터 사이트에서 주가, 시가총액, 영업일 등을 받아오는 파이썬 라이브러리

[왜 로그인이 필요한가]
- KRX가 데이터 사이트(data.krx.co.kr)를 로그인해야 쓸 수 있음. 
- 그래서 pykrx도 KRX 회원 아이디/비밀번호가 있어야 데이터를 받아옴
  (함수별로 필요 여부를 일일이 확인하지는 못했고, 같은 KRX 데이터라서 둘 다 필요하다고 가정)
- 로그인 정보가 없으면: 빈 결과가 나오거나 'Length mismatch' 같은 에러가 나옴

[pykrx가 로그인 정보를 읽는 방법]
- 환경변수 2개를 읽음:  KRX_ID (아이디),  KRX_PW (비밀번호)
- 설정이 되어 있으면 'from pykrx import stock' 할 때 콘솔에
  'KRX 로그인 시도... KRX 로그인 완료' 라고 뜸

[새 컴퓨터에서 설정하는 방법 3가지 - 하나만 하면 됨]
A) 윈도우에 영구 저장 : 명령 프롬프트(cmd)에서 한 번만 실행 후, Spyder를 완전히 껐다 켜기
      setx KRX_ID "내아이디"
      setx KRX_PW "내비밀번호"
B) 코드에서 임시 설정: 반드시 'from pykrx import stock' 보다 "위"에 쓰기
      import os
      os.environ['KRX_ID'] = '내아이디'
      os.environ['KRX_PW'] = '내비밀번호'
   (주의: 이 코드를 남에게 보내거나 인터넷에 올릴 때는 이 줄을 지울 것)
C) .env 파일 + python-dotenv 패키지 (.env 파일은 깃허브 등에 올리지 말 것)

[설치]
- pykrx가 없으면:  python -m pip install pykrx
- 설치는 Spyder 등의 python 환경에서 할 것.  (확인: import sys; print(sys.executable))
"""

#%% [1] 로그인 정보 설정 확인 (비밀번호는 출력하지 않고 설정 여부만 확인)
import os
print('KRX_ID 설정됨:', os.environ.get('KRX_ID') is not None)
print('KRX_PW 설정됨:', os.environ.get('KRX_PW') is not None)

#%% [2] 영업일 기준 데이터 받기 - 최근 영업일 구하기 방법 1: pykrx가 가까운 영업일을 계산
"""
- 오늘 날짜를 넣으면, 오늘이 휴일/주말이면 직전 영업일을, 영업일이면 오늘을 돌려줌
- 결과는 'yyyymmdd' 형식 문자열 (예: '20261009')
- 이 날짜를 다른 조회(시가총액, 공시 selDate 등)의 날짜에 넣어서 씀
"""
from pykrx import stock
from datetime import datetime, timedelta

today = datetime.today().strftime('%Y%m%d')
last_biz = stock.get_nearest_business_day_in_a_week(today)
print('방법1:', last_biz)

#%% [3] 최근 영업일 구하기 방법 2: 삼성전자 일봉의 마지막 날짜
"""
- 최근 10일치 삼성전자(005930) 일봉을 받아서 마지막 날짜를 꺼냄
- 실제 거래 데이터가 있는 마지막 날이라, 공휴일/임시휴장에도 틀리지 않아 가장 확실함
- 방법 1이 에러가 나면 이 방법을 쓰면 됨
"""
start = (datetime.today() - timedelta(days=10)).strftime('%Y%m%d')
df = stock.get_market_ohlcv(start, today, '005930')
last_biz2 = df.index[-1].strftime('%Y%m%d')
print('방법2:', last_biz2)



#%%
"""2. 한국거래소의 업종분류 현황 크롤링"""
""" KRX 정보데이터시스템에서 기본통계 -> 주식 -> 세부안내 부분으로 들어갈 것. """
#[1] 업종분류 현황 코스피 크롤링
# f12를 눌러 도구화면을 열로 다운로드 버튼을 클릭한 후 [csv]를 누른다. 
# 그 이후 generate cmd와, download.cmd 두 항목이 생긴다. 

# (1) 원하는 항목을 쿼리로 발송하면 해당 쿼리에 해당하는 OTP(generate.cmd) 를 받는다. 
# => [headers] 탭의 'general' 항목 중 'request URL' 부분이 원하는 항목을 제출할 주소이고, [payload] 탭의 form data에는 원하는 항목이 존재함. 
# => 이를 통해 potst 방식으로 데이터를 요청하면 opt를 받을 수 있다.

# (2) 부여받은 opt를 제출하면 이에 해당하는 데이터(download.cmd)를 다운로드한다. 
# -> 'general' 항목의 request url'은 otp를 제출할 주소이다. 
# -> 'form data'의 opt는 1번 단계를 통해 부여받은 opt에 해당하며, 이 역시 post 방식으로 데이터를 요청하여 csv 파일을 받아온다.

import requests as rq
from io import BytesIO
import pandas as pd 
from pykrx.website.comm.auth import get_auth_session

krx = get_auth_session()      # pykrx가 이미 로그인해둔 연결 (KRX_ID / KRX_PW 사용)
print('로그인 세션:', krx is not None)


gen_otp_url = 'https://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd'
gen_otp_stk = { #generate.cmd의 payload 탭의 payload 부분의 쿼리를 딕셔너리 형태로 입력한다.
    'mktId': 'STK',            # STK는 코스피, KSQ는 코스닥
    'trdDd': last_biz,         # 앞선 최근 영업일 데이터를 불러옴
    'money': '1',
    'csvxls_isNo': 'false',
    'name': 'fileDown',
    'url': 'dbms/MDC/STAT/standard/MDCSTAT03901'
}

# header 부분에 referer 추가
headers = {'Referer': 'https://data.krx.co.kr/contents/MDC/MDI/mdiLoader'}

# 리퍼러란 링크를 통해서 각각의 웹사이트 방문시 남는 흔적
# 거래소 데이터 다운로드 과정은 (1) url에서 opt 부여받고 ,(2)  이를 두번재 url 제출임.
# 상기 과정을 보여주지 않고, opt를 바로 (2)의 url에 제출하면 데이터를 주지않음(봇으로 인식)
# 그러므로 리퍼러 과정을 흔적으로 남겨야 한다. 
#리퍼러 위치 : generate.cmd-> headers 탭, request headers 칸에서 referer가 있음. 
# -> https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201 라고 나와있으며 index 전까지 입력.


# post() 함수를 통해 해당 url에 쿼리ㅡㄹㄹ 전송하면 이에 해당하는 데이터를 받으며, 이 중 텍스트에 해당하는 내용만 불러온다. 

# 1) OTP(일회용 코드) 받기  krx.post사용
otp_stk = krx.post(gen_otp_url, headers=headers, data=gen_otp_stk).text
print(otp_stk[:100])           # 영문+숫자 코드가 길게 나오면 성공 (LOGOUT / <HTML> 이면 실패)

# 2) OTP를 제출해서 csv 받기  ←krx.post
down_url = 'https://data.krx.co.kr/comm/fileDn/download_csv/download.cmd'# download.cmd의 [general] 탭의 request URL이다. 
down_sector_stk = krx.post(down_url, headers=headers, data={'code': otp_stk}) # krx.post () 함수를 통해 위에서 부여받은 opt코드를 해당 URL에 제출한다 (OTP로 파일 내려받기))

# 3) 받은 csv를 표로 읽기
sector_stk = pd.read_csv(BytesIO(down_sector_stk.content), encoding='EUC-KR')
# #받은 데이터의 content 부분을 bytesIO를 이용해 바이너리 스트림 형태로 만든 후, read_csv 함수를 통해 데이터 읽어옴. 해당 데이터 EUC-KR 형태인 바, 이를 선언해준다.( 내려받은 csv를 표로 읽기
sector_stk.head()



#%% [4-2] 업종분류 현황 - 코스닥 (코스피와 같은 코드, mktId만 KSQ)
gen_otp_ksq = {
    'mktId': 'KSQ',            # KSQ는 코스닥
    'trdDd': last_biz,
    'money': '1',
    'csvxls_isNo': 'false',
    'name': 'fileDown',
    'url': 'dbms/MDC/STAT/standard/MDCSTAT03901'
}

otp_ksq = krx.post(gen_otp_url, headers=headers, data=gen_otp_ksq).text
down_sector_ksq = krx.post(down_url, headers=headers, data={'code': otp_ksq})
sector_ksq = pd.read_csv(BytesIO(down_sector_ksq.content), encoding='EUC-KR', dtype={'종목코드': str})
sector_ksq.head()

#%% [4-3] 코스피 + 코스닥 합치기
krx_sector = pd.concat([sector_stk, sector_ksq], ignore_index=True) # concat 함수 통해 두 테이터 합침.
krx_sector['종목명'] = krx_sector['종목명'].str.strip() # 종목명에 공백 있는 경우 strip() 메서드로 이를 제거
krx_sector['기준일'] = last_biz # 데이터 기준일에 해당하는 [기준일] 열을 추가 

krx_sector.head()


#%% [4-4] 함수 방식 - 현재는 ''' ''' 안에 들어 있어서 실행되지 않음
# 쓰고 싶으면 아래 ''' 두 줄(시작/끝)을 지우면 위의 [4]~[4-3] 대신 이 방식으로 코스피/코스닥을 받음
# (위 [4-1] 안의 krx, gen_otp_url, headers, down_url 이 먼저 만들어져 있어야 함)
"""
import time

def download_sector(mkt_id, base_date):
    """mkt_id: 'STK'=코스피, 'KSQ'=코스닥 / base_date: 'yyyymmdd'"""
    query = {
        'mktId': mkt_id,
        'trdDd': base_date,
        'money': '1',
        'csvxls_isNo': 'false',
        'name': 'fileDown',
        'url': 'dbms/MDC/STAT/standard/MDCSTAT03901',
    }
    otp = krx.post(gen_otp_url, headers=headers, data=query).text
    if 'LOGOUT' in otp or '<' in otp:      # 로그인/차단 문제면 여기서 멈추고 알려줌
        raise RuntimeError('OTP를 못 받았습니다: ' + otp[:100])
    res = krx.post(down_url, headers=headers, data={'code': otp})
    return pd.read_csv(BytesIO(res.content), encoding='EUC-KR', dtype={'종목코드': str})

sector_stk = download_sector('STK', last_biz)
time.sleep(1)                                   # 연속 요청 사이에 잠깐 쉬기
sector_ksq = download_sector('KSQ', last_biz)
print('함수 방식 실행 완료')

krx_sector = pd.concat([sector_stk, sector_ksq], ignore_index=True)
print(sector_stk.shape, sector_ksq.shape, sector_all.shape)
sector_all.head()"""



#%% 
"""3. 개별종목 지표 크롤링"""
# 개별종목 데이터([12021] PER/PBR/배당수익률(개별종목)을 의미) 크롤링은 위와 유사, 요청하는 쿼리값만 차이가 있음. 
# 개발자 도구 확인 시, csv 버튼을 클릭하여 어떤 쿼리 요청하는지 확인 피룡. 
# generate.cmd의 payload 탭의 필요한 항목만을 가지고와 다운로드 한다. 

import requests as rq
from io import BytesIO
import pandas as pd
from datetime import datetime, timedelta

# krx, gen_otp_url, headers, down_url, last_biz 는 위 [2], [4]에서 이미 만들어 둔 것을 그대로 씀

# strt_day = (datetime.today() - timedelta(days=7)).strftime('%Y%m%d')   # 조회 시작일: 7일 전 (기간은 마음대로 바꿔도 됨)

gen_otp_ind = {
    'searchType': '1',
    'mktId': 'ALL',
    'trdDd': last_biz,                                   # 기준일: 최근 영업일
    #'tboxisuCd_finder_stkisu0_0': '005930/삼성전자',
    #'strtDd': strt_day,                                  # 시작일
    #'endDd': last_biz,                                   # 종료일: 최근 영업일
    'csvxls_isNo': 'false',
    'name': 'fileDown',
    'url': 'dbms/MDC/STAT/standard/MDCSTAT03501'
}

headers = {'Referer': 'https://data.krx.co.kr/contents/MDC/MDI/mdiLoader'}


# 1) OTP 받기
otp_ind = krx.post(gen_otp_url, headers=headers, data=gen_otp_ind).text
print(otp_ind[:100])           # 영문+숫자 코드가 길게 나오면 성공 (LOGOUT / <HTML> 이면 실패)

# 2) OTP를 제출해서 csv 받기
down_ind = krx.post(down_url, headers=headers, data={'code': otp_ind})

# 3) 받은 csv를 표로 읽기
krx_ind = pd.read_csv(BytesIO(down_ind.content), encoding='EUC-KR')
krx_ind['종목명'] = krx_ind['종목명'].str.strip()
krx_ind['기준일'] = last_biz

krx_ind.head()

#%%
"""4. 데이터 정리하기"""
diff = list(set(krx_sector['종목명']).symmetric_difference(set(krx_ind['종목명'])))
print(diff)
# 이는 하나의 데이터에만 존재하는 종목을 말해준다. 
# 두 데이터의 종목명 열을 세트 형태로 바꾼 후 synnetric difference 메서드를 통해 하나의 데이터에만 있는 종목을 나타낸 것이다. 
# 해당 종목들은 일반적으로 리츠나 선박 광물 펀드 및 해외종목 등 일반적이지 않은 경우에 해당. 고로, 다음 두 데이터를 합쳐준다. 

kor_ticker = pd.merge(krx_sector,
                      krx_ind,
                      on= krx_sector.columns.intersection(
                          krx_ind.columns).tolist(),
                      how = 'outer')
# merge 함수는 on 조건을 기준으로 두 데이터를 합친다.
# intersection() 메서드를 이용해 공통 존재 [종목코드,종목명,종가,대비,등락률 열을 기준으로 입력 
# 방법 outer 는 how 에 입력

kor_ticker.head()


# 이하 일반적 종목과 스팩, 우선주, 리츠, 기타 주식등을 구분하자 
# 스팩이란 (special purpose acquisition company) 의 약자로 기업인수를 목적으로 하는 페이퍼컴퍼니

print(kor_ticker[kor_ticker['종목명'].str.contains('스팩|제[0-9]+호')]['종목명'].values)
# 스팩 종목은 스팩 또는 제 n호라는 말이 들어가는바, contains()통해 스팩이 들어가거나 정규 표현식을 이용해 제n호라는 문자가 들어가는 종목명을 찾는다. 

print(kor_ticker[kor_ticker['종목코드'].str[-1:]!='0']['종목명'].values)
# 국내 종목 중 종목코드 끝이 0이 아닌 종목은 우선주이다. 

print(kor_ticker[kor_ticker['종목명'].str.endswith('리츠')]['종목명'].values)
# 메리츠투자와 같이 중간에 리츠가 들어가는 종목도 있는바, contain이 아닌 endswith  사용해야 한다. 

# 이하 해당 종목 구분하여 표기하기

import numpy as np

kor_ticker['종목구분'] = np.where(kor_ticker['종목명'].str.contains('스팩|제[0-9]+호'),'스팩',
                              np.where(kor_ticker['종목코드'].str[-1:]!='0','우선주',
                                     np.where(kor_ticker['종목명'].str.endswith('리츠'),'리츠',
                                              np.where(kor_ticker['종목명'].isin(diff),'기타',
                                                       '보통주')))) 

kor_ticker = kor_ticker.reset_index(drop=True) #reset_index() 메서드 통해 인덱스 초기화
kor_ticker.colums = kor_ticker.columns.str.replace(' ','') # replace()메서드 통해 열 이름의 공백 삭제
kor_ticker = kor_ticker[['종목코드','종목명','시장구분','종가','시가총액','기준일','EPS',
                         'PER','BPS','주당배당금','종목구분']] # 필요한 열만 선택 
kor_ticker = kor_ticker.replace({np.nan:None}) # SQL에는 NAN이 입력되지 않는 바 None으로 변경
kor_ticker['기준일'] = pd.to_datetime(kor_ticker['기준일']) # 기준일을 to_datetime() 이용하여 yyyymmdd에서 yyyy-mm-dd 형태로 변경

#%%
""" 그다음 sql을 이용하며 sql 코드는 아래와 같다.(해당 정보 db에 저장 )

create database stock_db;

use stock_db;

create table kor_ticker

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

"""

#%%
""" sql에 파일 저장하기 """
import pymysql

con = pymysql.connect(user='root',
                      passwd='handdmom1!',
                      host='127.0.0.1',
                      db='stock_db',
                      charset='utf8')

mycursor = con.cursor()
query = f"""
    insert into kor_ticker(종목코드,종목명,시장구분,종가,시가총액,기준일,EPS,PER,BPS,주당배당금, 종목구분)
    values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) as new
    on duplicate key update
    종목명 = new.종목명, 시장구분 = new.시장구분, 종가 = new.종가, 시가총액 = new.시가총액, EPS =new.EPS, PER= new.PER, BPS = new.BPS, 주당배당금=new.주당배당금, 종목구분=new.종목구분;
"""

args = kor_ticker.values.tolist()

mycursor.executemany(query,args)
con.commit()

con.close()

# SELECT * FROM kor_ticker; sql 에서 보면 데이터 들어옴 확인 가능.

#%%
"""3. WICS 기준 섹터 정보 크롤링"""
"""MSCI, GICS를 많이사용하고 GICS를 한국 거래소에서 공개중이나, 독점적 지적 재산으로 명시한 바 사용에 무리가 있다.
고로, fnguide index에서 GICS와 비슷한 WICS 산업분류를 발표하는 바, 이를 크롤링하여 정보를 수집한다. 
해당 페이지는 http://www.wiseindex.com/Index 이다. 
wise sector index -> wics -> 섹터선택 -> component는 구성종목까지 확인이 가능하다. 

개발자 도구화면 통해 페이지의 데이터 전송 과정을 살필 수 있으며, 일자를 선택시 
[network] 탭의 'getindexcomponent'항목에 데이터 전송과정이 나타난다. 
request URL은 데이터를 가져오는 주소이며 이를 분석할 수 있다. 

headers의  request url : https://www.wiseindex.com/Index/GetIndexComponets?ceil_yn=0&dt=20261008&sec_cd=G10  이는 데이터를 요청하는 url이다. 

payload에서는 아래와 같이 확인할 수 있다. 
ceil_yn= 0: 실링 여부 나내며, 0은 비실링을 의미한다. 
dt= 20261008: 조회일자를 나타낸다. 
sec_cd= G10: 섹터 코드를 나타낸다. 

request url 페이지를 열었을 때 json 형식의 데이터로 표현되어있다. 
json 방식은 문법이 단순하고 데이터 용량이 작아 빠른 속도로 데이터 교환이 가능하다. 
파이썬에서 json 패키지를 사용해 매우 쉽게 json 형식의 데이터를 크롤링할 수 있다. """

#%%
import json 
import requests as rq
import pandas as pd

url = f'''https://www.wiseindex.com/Index/GetIndexComponets?ceil_yn=0&dt=20261008&sec_cd=G10 '''
# f-string 포매팅 이용 dt 부분에서는 위에서 구한 최근 영업일 데이터를 입력하여 url을 생성한다. 

data = rq.get(url).json()
# get() 함수를 통해 페이지의 내용을 받아 오며, json() 메서드를 통해 JSON 데이터만 불러올 수 있다. 

type(data)


# key 확인
print(data.keys())

# 각각 확인
data['list'][0]
data['sector']

#%%
data_pd = pd.json_normalize(data['list']) # list부분의 데이터를 데이터프레임 형태로 변경
data_pd.head()

# %%
# for 문을 이용하여 URL의 sec_cd 부분만 변경하면 구성종목 쉽게 얻을 수 있다. 
import time
import json
import requests as rq
import pandas as pd
from tqdm import tqdm

sector_code = [
    'G25','G35','G50','G40','G10','G20','G55','G30','G15','G45'
    ]

data_sector = []    # 섹터 정보가 들어갈 빈 리스트 만들기

for i in tqdm(sector_code): # tqdm() 함수 통해 진행 상황 출력
    url = f'''https://www.wiseindex.com/Index/GetIndexComponets?ceil_yn=0&dt=20261008&sec_cd={i}'''
    # 기존 url에서 날짜부분은 앞서 만든 last biz로 가능 다만 장이 끝나기 전등 데이터가 없으면 오류발생 가능, cd= g10 이 아닌i로 설정
    data = rq.get(url).json()
    data_pd = pd.json_normalize(data['list'])

    data_sector.append(data_pd) # for 문의 i에 섹터 코드를 입력하여 모든 섹터 구성종목 다운 후 append() 메서드를 통해 리스트에 추가

    time.sleep(2)
    
sector = pd.concat(data_sector, axis=0)  # 리스트 내 데이터프레임 합치기
sector = sector[['IDX_CD', 'CMP_CD', 'CMP_KOR', 'SEC_NM_KOR']].copy()  # 필요한 열만 선택
sector['기준일'] = last_biz
sector['기준일'] = pd.to_datetime(sector['기준일'])


#%% sql 코드
"""use stock_db;

create table kor_sector 
( 
    IDX_CD varchar(3),
    CMP_CD varchar(6),
    CMP_KOR varchar(20),
    SEC_NM_KOR varchar(10),
    기준일 date,
    primary key(CMP_CD, 기준일)
);"""

#%% 국내 섹터 들어갈 테이블 만들기
import pymysql

con = pymysql.connect(user='root',
                      passwd='handdmom1!',
                      host='127.0.0.1',
                      db='stock_db',
                      charset = 'utf8')

mycursor=con.cursor()
query = f"""
    insert into kor_sector (IDX_CD, CMP_CD, CMP_KOR, SEC_NM_KOR, 기준일)
    values(%s,%s,%s,%s,%s) as new
    on duplicate key update
    IDX_CD = new.IDX_CD, CMP_KOR = new.CMP_KOR, SEC_NM_KOR = new.SEC_NM_KOR
"""
                        
args = sector.values.tolist()

mycursor.executemany(query,args)
con.commit()

con.close()


#%%
"""4. 수정주가 크롤링

주가 데이터는 투자를 함에 있어 반드시 필요하며, 퀀트 투자 및 백테스트 위해서는 수정주가 필요하다. """

# db에서 티커 데이터 불러오기  
from sqlalchemy import create_engine 
import pandas as pd 

engine = create_engine('mysql+pymysql://root:handdmom1!@127.0.0.1:3306/stock_db')
# create_engine() 함수를 통해 데이터베이스 접속을 위한 엔진 만들기

query = """
select * from kor_ticker 
where 기준일 = (select max(기준일) from kor_ticker)
    and 종목구분 = '보통주' ; 
"""
# 티커 데이터 불로기위한 쿼리로, 가장 최근일자 불러오기 위해 select max(기준일) from kor_ticker를 입력함. 또한, 종목구분에서 보통주에 해당하는 종목만 선택

ticker_list = pd.read_sql(query, con=engine) # read sql() 함수 통해 해당 쿼리 보낸 후 데이터 받아오기
engine.dispose() # 접속 종료

ticker_list.head()


#%% : 위에서 살펴본 주가 데이터 페이지 크롤링 하기 
from dateutil.relativedelta import relativedelta
import requests as rq
from io import BytesIO
from datetime import date

i=0 # for 문 통해 i 값만 변경하면 모든 종목 주가 다운로드 가능
ticker = ticker_list['종목코드'][i] #원하는 종목의 티커 선택

fr = (date.today() + relativedelta(years=-5)).strftime("%Y%m%d") 
to = (date.today()).strftime("%Y%m%d") 
# fr=시작일, to= 종료일 해당하는 날짜 만들어줌 today() 메서드 통해 불러온 오늘 날짜를 불러온후 시작일은 원하는 만큼 빼줄 수 있으며 이 코드에서는 5년을 뺌.
# 그후 strftime() 메서드를 통해 'yyyymmdd' 형식 만들어줌 종료일은 오늘 날짜 그대로 사용

url = f'''https://fchart.stock.naver.com/siseJson.nhn?symbol={ticker}&requestType=1&startTime={fr}&endTime={to}&timeframe=day'''
# 티커, 시작일, 종료일 이용해 주가 데이터 있는 url 생성

data = rq.get(url).content # get() 함수 통해 페이지 데이터 불러오고 content 부분 추출

data_price = pd.read_csv(BytesIO(data)) # byterIO() 이용해 바이너리 스트림 만들고 데이터 읽어온다. 
data_price.head()

# print(ticker)                          # 종목코드
# ticker_list.loc[i, ['종목코드', '종목명']]   # 해당 종목의 이름까지 확인

"""위 결괄h 날자, 주가 ,외국인 소진율 등 추출 이하 클랜징 작업 실시한다"""

#%%
import re
import pandas as pd
price = data_price.iloc[:, 0:6] # iloc 통해 날짜, 가격, 거래량 해당 데이터만 선택
price.columns = ['날짜','시가','고가','저가','종가','거래량'] # 열 이름 변경
price = price.dropna() # dronpna() 통해 NA 데이터 삭제
price['날짜'] = price['날짜'].str.extract('(\d+)') # extract() 메서드 내에 정규 표현식 이용 날짜 열에서 숫자만 추출
price['날짜'] = pd.to_datetime(price['날짜']) # '날짜' 열을 datetime 형태로 변경
price['종목코드'] = ticker # 종목코드 열에 티커를 입력한다. 

price.head()


#%% 전종목 주가 크롤링
""" SQL: 주가가 저될 테이블 만들기
use stock_db;

create table kor_price
(
     날짜 date,
     시가 double,
     고가 double,
     저가 double,
     종가 double,
     거래량 double,
     종목코드 varchar(6),
     primary key(날짜, 종목코드)   
) ; 

이하 파이썬 코드로 저장하기 
"""

#패키지 불러오기
import pymysql
from sqlalchemy import create_engine
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
import requests as rq
import time
from tqdm import tqdm
from io import BytesIO

#DB연결
engine = create_engine('mysql+pymysql://root:handdmom1!@127.0.0.1:3306/stock_db')
con= pymysql.connect (user= 'root',
                      passwd= 'handdmom1!',
                      host = '127.0.0.1',
                      db='stock_db',
                      charset = 'utf8')

mycursor = con.cursor()

#티커 리스트 불러오기(최근일 기준 보통주 해당 티커 리스트)
ticker_list = pd.read_sql("""
select * from kor_ticker
where 기준일 = (select max(기준일) from kor_ticker)
    and 종목구분 = '보통주';
""", con=engine)


#DB저장쿼리
query = """
    insert into kor_price(날짜, 시가, 고가, 저가, 종가, 거래량, 종목코드)
    values(%s,%s,%s,%s,%s,%s,%s) as new
    on duplicate key update
    시가 = new.시가, 고가 = new.고가, 저가 = new.저가, 
    종가 = new.종가, 거래량 = new.거래량; 
"""

# 오류 발생 시 저장할 리스트 생성
error_list = []

# 전 종목 주가 다운로드 및 저장
for i in tqdm(range(0,len(ticker_list))): # 진행상황 표시
    
    
    # 티커 선택
    ticker = ticker_list['종목코드'][i]
    
    
    # 시작일과 종료일
    fr = (date.today() + relativedelta(years=-5)).strftime("%Y%m%d") 
    to = (date.today()).strftime("%Y%m%d") 
    
    
    # 오류 발생시 이를 무시하고 다음 루프로 진행
    try: 
        
        #url 생성
        url = f'''https://fchart.stock.naver.com/siseJson.nhn?symbol={ticker}&requestType=1&startTime={fr}&endTime={to}&timeframe=day'''
        
        
        #데이터 다운로드
        data= rq.get(url).content
        data_price = pd.read_csv(BytesIO(data))
        
        
        #데이터 클랜징
        price = data_price.iloc[:, 0:6] # iloc 통해 날짜, 가격, 거래량 해당 데이터만 선택
        price.columns = ['날짜','시가','고가','저가','종가','거래량'] # 열 이름 변경
        price = price.dropna() # dronpna() 통해 NA 데이터 삭제
        price['날짜'] = price['날짜'].str.extract('(\d+)') # extract() 메서드 내에 정규 표현식 이용 날짜 열에서 숫자만 추출
        price['날짜'] = pd.to_datetime(price['날짜']) # '날짜' 열을 datetime 형태로 변경
        price['종목코드'] = ticker # 종목코드 열에 티커를 입력한다.
        
        
        #주가 데이터를 DB에 저장
        args = price.values.tolist()
        mycursor.executemany(query,args)
        con.commit()
        
        
        
    except:
        
        # 오류 발생 시 error_list에 티커 저장하고 넘어가기
        print(ticker)
        error_list.append(ticker)
    
    
    #타임슬립 적용 for 무한 크롤링 방지 
    #time.sleep(1) 

# DB 연결 종료 
engine.dispose()
con.close()       
                    



#%%
"""5. 재무제표 크롤링
가치투자의 핵심으로 fnguide에서 운영하는 company guide 웹사이트에서 구하기 (http://comp.fnguide.com/)

웹페이지에서 개별종목 재무제표 탭 선택시 포괄손익계산서, 재무상태표, 현금흐름표 항목 존재하고 
그 밖에 티커에 해당하는 뒤에 주소는 불필요한 바, 이를 제거한 주소로 접속한다. 티커만 바꾸면 다른 재무제표 페이지 이동 가능
"""


from sqlalchemy import create_engine
import pandas as pd
import requests as rq
from bs4 import BeautifulSoup
import re
import time
from tqdm import tqdm

engine = create_engine('mysql+pymysql://root:handdmom1!@127.0.0.1:3306/stock_db')
query = """
select * from kor_ticker 
where 기준일 = (select max(기준일) from kor_ticker)
    and 종목구분 = '보통주';
"""
ticker_list = pd.read_sql(query, con=engine)      # 1) 티커 리스트를 불러와 첫 번째 티커를 선택한다.
engine.dispose()

i = 0
ticker = ticker_list['종목코드'][i]

url = f'https://wcomp.fnguide.com/CompanyInfo/Finance?c_id=AA&menu_type=01&cmp_cd={ticker}'   # 2) 재무제표 페이지에 해당하는 URL 생성

fnguide_headers = {'User-Agent': 'Mozilla/5.0'}

def fnguide_fs(name, ticker, freq):
    """name: getFinIncome(손익) / getFinBalance(재무상태) / getFinCashFlow(현금흐름), freq: 'Y'=연간, 'Q'=분기
       결과: 첫 열 'IFRS(연결)' + 기간 열들 """
    u = f'https://wcomp.fnguide.com/CompanyInfo/{name}'
    r = rq.get(u, params={'cmp_cd': ticker, 'freq_typ': freq, 'consol_typ': 'C'}, headers=fnguide_headers)
    o = r.json()['dataset']      
                                   
    if isinstance(o, list) and len(o) > 0 and 'data' in o[0]:
        o = o[0]
        
    if not (isinstance(o, dict) and 'data' in o and 'header' in o):  # 구조가 다르면 보여주고 멈춤
        print(type(o), (list(o.keys()) if isinstance(o, dict) else str(o)[:300]))
        raise ValueError(f'{name} {freq}: 구조 확인 필요')
        
    df = pd.DataFrame(o['data'])
    rename = {h['CD']: h['YYMM'].replace(' (최근분기)', '') for h in o['header']}   # VAL1, VAL2... -> 2023/12 ... 로 열 이름 변경 (' (최근분기)' 글자는 제거)
    df = df.rename(columns=rename)
    df['NAME'] = df['NAME'].str.strip()
    df = df[['NAME'] + list(rename.values())]
    df = df.rename(columns={'NAME': 'IFRS(연결)'})
    for c in rename.values():
        df[c] = pd.to_numeric(df[c], errors='coerce')               # 숫자 변환 (글자는 NaN)
    return df

def fnguide_tables(ticker):
    """표 10.1 순서대로 6개: 0 손익(연간) 1 손익(분기) 2 재무상태(연간) 3 재무상태(분기) 4 현금흐름(연간) 5 현금흐름(분기)"""
    return [fnguide_fs('getFinIncome', ticker, 'Y'),
            fnguide_fs('getFinIncome', ticker, 'Q'),
            fnguide_fs('getFinBalance', ticker, 'Y'),
            fnguide_fs('getFinBalance', ticker, 'Q'),
            fnguide_fs('getFinCashFlow', ticker, 'Y'),
            fnguide_fs('getFinCashFlow', ticker, 'Q')]
# ----------------------------------------------------------------------------------------------

data = fnguide_tables(ticker)       

[item.head(3) for item in data]
# 위의 과정을 거치면 총 6개의 표가 들어온다. (표 10.1)
# 0 포괄손익계산서(연간) / 1 포괄손익계산서(분기) / 2 재무상태표(연간) / 3 재무상태표(분기) / 4 현금흐름표(연간) / 5 현금흐름표(분기)


#%% 연간 재무제표 합치기
# 연간 기준 포괄손익계산서, 재무상태표, 현금흐름표의 열 이름을 살펴본다.
print(data[0].columns.tolist(), '\n',
      data[2].columns.tolist(), '\n',
      data[4].columns.tolist()
      )

# 포괄손익계산서 표에는 '전년동기', '전년동기(%)' 열이 있으며, 이는 필요하지 않은 내용이므로 삭제해 주어야 한다.
data_fs_y = pd.concat(
    [data[0].iloc[:, ~data[0].columns.str.contains('전년동기')], data[2], data[4]])   # 1) 포괄손익계산서 중 '전년동기'라는 글자가 들어간 열을 제외한 데이터를 선택한다.
                                                                                      # 2) concat() 함수를 이용해 포괄손익계산서, 재무상태표, 현금흐름표 3개 표를 하나로 묶는다.
data_fs_y = data_fs_y.rename(columns={data_fs_y.columns[0]: "계정"})                   # 3) rename() 메서드를 통해 첫 번째 열 이름(IFRS 또는 IFRS(연결))을 '계정'으로 변경한다.

data_fs_y.head()


#%% 결산월 찾기
# 결산마감 이전에 해당 페이지를 크롤링할 경우 연간 재무제표 데이터에 분기 재무제표 데이터가 들어오기도 하므로,
# 연간 재무제표에 해당하는 열만을 선택해야 한다. 각 종목별 결산월은 해당 페이지의 상단에서 확인할 수 있다.

page_data = rq.get(url, headers=fnguide_headers)                           # 1) get() 함수를 통해 페이지의 데이터를 불러온 후, content 부분을 BeautifulSoup 객체로 만든다.
page_data_html = BeautifulSoup(page_data.content, 'html.parser')

fiscal_data = page_data_html.select('div.corp_group1 > h2')                # 2) 결산월 항목은 corp_group1 클래스의 div 태그 하부의 h2 태그에 존재하므로 select() 함수를 이용해 추출한다.

if len(fiscal_data) > 1:
    fiscal_data_text = fiscal_data[1].text                                 # 3) fiscal_data 중 첫 번째는 종목코드에 해당하고, 두 번째가 결산 데이터에 해당하므로 해당 부분을 선택해 텍스트만 추출한다.
    fiscal_data_text = re.findall('[0-9]+', fiscal_data_text)              # 4) 'n월 결산' 형태로 텍스트가 구성되어 있으므로, 정규 표현식을 이용해 숫자에 해당하는 부분만 추출한다.
else:
    fiscal_data_text = list({c[-2:] for c in data[2].columns[1:] if re.fullmatch(r'\d{4}/\d{2}', str(c))})   # (예비) 재무상태표(연간)의 'yyyy/mm' 열에서 월만 뽑기
# ------------------------------------------------------------------------------------------------------------------------
print(fiscal_data_text)        # 책 결과: ['12']

# 이를 통해 결산월에 해당하는 부분만이 선택된다. 이를 이용해 연간 재무제표에 해당하는 열만 선택해 보자.
data_fs_y = data_fs_y.loc[:, (data_fs_y.columns == '계정') | (
    data_fs_y.columns.str[-2:].isin(fiscal_data_text))]

data_fs_y.head()
# 열 이름이 '계정', 그리고 재무제표의 월이 결산월과 같은 부분만 선택한다. 이제 추가적으로 클렌징해야 하는 사항은 다음과 같다.


#%% 클렌징이 필요한 부분 확인
# 먼저, 재무제표 값 중에서 모든 연도의 데이터가 NaN인 항목이 있다. 이는 재무제표 계정은 있으나 해당 종목들은 데이터가 없는 것들이므로 삭제해도 된다.
data_fs_y[data_fs_y.loc[:, ~data_fs_y.columns.isin(['계정'])].isna().all(axis=1)].head()

# 또한, 동일한 계정명이 여러 번 반복된다. 이러한 계정은 대부분 중요하지 않은 것들이므로, 하나만 남겨 두도록 한다.
data_fs_y['계정'].value_counts(ascending=False).head()

# 이 외에도 클렌징이 필요한 내용들을 함수로 구성하면 다음과 같다.


#%% 재무제표 클렌징 함수
def clean_fs(df, ticker, frequency):                                          # 1) 입력값으로는 데이터프레임, 티커, 공시구분(연간/분기)이 필요하다.

    df = df[~df.loc[:, ~df.columns.isin(['계정'])].isna().all(axis=1)]         # 2) 먼저, 모든 연도의 데이터가 NaN인 항목은 제외한다.
    df = df.drop_duplicates(['계정'], keep='first')                            # 3) 계정명이 중복되는 경우 drop_duplicates() 함수를 이용해 첫 번째에 위치하는 데이터만 남긴다.
    df = pd.melt(df, id_vars='계정', var_name='기준일', value_name='값')         # 4) melt() 함수를 이용해 열로 긴 데이터를 행으로 긴 데이터로 변경한다.
    df = df[~pd.isnull(df['값'])]                                              # 5) 계정값이 없는 항목은 제외한다.
    df['계정'] = df['계정'].replace({'계산에 참여한 계정 펼치기': ''}, regex=True)   # 6) [계산에 참여한 계정 펼치기]라는 글자는 페이지의 [+]에 해당하는 부분이므로 replace() 메서드를 통해 제거한다.
    df['기준일'] = pd.to_datetime(df['기준일'],
                               format='%Y/%m') + pd.tseries.offsets.MonthEnd()  # 7) to_datetime() 메서드를 통해 기준일을 'yyyy-mm' 형태로 바꾼 후, MonthEnd()를 통해 월말에 해당하는 일을 붙인다.
                                                                                # ([책과 다른 부분 3] 책은 format='%Y-%m' 이지만, 이 페이지의 열 이름이 '2023/12' 형태라서 '%Y/%m' 으로 맞춤)
    df['종목코드'] = ticker                                                    # 8) '종목코드' 열에는 티커를 입력한다.
    df['공시구분'] = frequency                                                 # 9) '공시구분' 열에는 연간 또는 분기에 해당하는 값을 입력한다.

    return df


#%% 연간 / 분기 클렌징 후 합치기
# 연간 재무제표 항목에 위 함수를 적용하면 다음과 같은 결과를 확인할 수 있다.
data_fs_y_clean = clean_fs(data_fs_y, ticker, 'y')
data_fs_y_clean.head()

# 클렌징 처리가 된 데이터가 세로로 긴 형태로 변경되었다. 이제 분기 재무제표도 클렌징 처리를 해보자.
# 분기 데이터
data_fs_q = pd.concat(
    [data[1].iloc[:, ~data[1].columns.str.contains('전년동기')], data[3], data[5]])
data_fs_q = data_fs_q.rename(columns={data_fs_q.columns[0]: "계정"})
data_fs_q_clean = clean_fs(data_fs_q, ticker, 'q')

data_fs_q_clean.head()
# 분기 데이터는 결산월에 해당하는 부분을 선택할 필요가 없으며, 이를 제외하고는 모든 과정이 연간 재무제표의 항목과 동일하다.

# concat() 함수를 통해 두 테이블을 하나로 묶어 준다.
data_fs_bind = pd.concat([data_fs_y_clean, data_fs_q_clean])


#%% 10.5.2 전 종목 재무제표 크롤링
""" 위 과정을 응용해 모든 종목의 재무제표를 크롤링한 후 DB에 저장하는 과정을 살펴보겠다.
먼저, SQL에서 재무제표가 저장될 테이블(kor_fs)을 만들어 준다.

use stock_db;

create table kor_fs
(
    계정 varchar(30),
    기준일 date,
    값 float,
    종목코드 varchar(6),
    공시구분 varchar(1),
    primary key(계정, 기준일, 종목코드, 공시구분)
);
"""

#%% 전 종목 재무제표 크롤링 (이제 파이썬에서 아래 코드를 실행하면 for문을 통해 전 종목 재무제표가 DB에 저장된다.)
# 패키지 불러오기
import pymysql
from sqlalchemy import create_engine
import pandas as pd
import requests as rq
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import time

# DB 연결                                                                      # 1) DB에 연결한다.
engine = create_engine('mysql+pymysql://root:handdmom1!@127.0.0.1:3306/stock_db')
con = pymysql.connect(user='root',
                      passwd='handdmom1!',
                      host='127.0.0.1',
                      db='stock_db',
                      charset='utf8')
mycursor = con.cursor()

# 티커 리스트 불러오기                                                         # 2) 기준일이 최대, 즉 최근일 기준 보통주에 해당하는 티커 리스트(ticker_list)만 불러온다.
ticker_list = pd.read_sql("""
select * from kor_ticker
where 기준일 = (select max(기준일) from kor_ticker)
    and 종목구분 = '보통주';
""", con=engine)

# DB 저장 쿼리                                                                 # 3) DB에 저장할 쿼리(query)를 입력한다.
query = """
    insert into kor_fs (계정, 기준일, 값, 종목코드, 공시구분)
    values (%s,%s,%s,%s,%s) as new
    on duplicate key update
    값=new.값
"""

# 오류 발생 시 저장할 리스트 생성                                               # 4) 오류 발생 시 저장할 리스트(error_list)를 만든다.
error_list = []

# 재무제표 클렌징 함수 (위에서 만든 clean_fs 를 그대로 사용 - 셀을 처음부터 새로 실행했다면 위 clean_fs 셀을 먼저 실행)

# for loop                                                                     # 5) for문을 통해 전 종목 재무제표를 다운로드받으며, 진행 상황을 알기 위해 tqdm() 함수를 이용한다.
for i in tqdm(range(0, len(ticker_list))):

    # 티커 선택
    ticker = ticker_list['종목코드'][i]

    # 오류 발생 시 이를 무시하고 다음 루프로 진행                                   # 6) URL 생성, 데이터 다운로드 및 데이터 클렌징 및 DB에 저장은 위와 동일하며, try except문을 통해 오류 발생 시 티커를 출력 후 error_list에 저장한다.
    try:

        # url 생성
        url = f'https://wcomp.fnguide.com/CompanyInfo/Finance?c_id=AA&menu_type=01&cmp_cd={ticker}'

        # 데이터 받아 오기
        data = fnguide_tables(ticker)                    # 책: pd.read_html(url, displayed_only=False)

        # 연간 데이터
        data_fs_y = pd.concat([
            data[0].iloc[:, ~data[0].columns.str.contains('전년동기')], data[2],
            data[4]
        ])
        data_fs_y = data_fs_y.rename(columns={data_fs_y.columns[0]: "계정"})

        # 결산년 찾기
        page_data = rq.get(url, headers=fnguide_headers)
        page_data_html = BeautifulSoup(page_data.content, 'html.parser')

        fiscal_data = page_data_html.select('div.corp_group1 > h2')
        if len(fiscal_data) > 1:
            fiscal_data_text = fiscal_data[1].text
            fiscal_data_text = re.findall('[0-9]+', fiscal_data_text)
        else:                                            # (예비) 위에서 설명한 [책과 다른 부분 2]
            fiscal_data_text = list({c[-2:] for c in data[2].columns[1:] if re.fullmatch(r'\d{4}/\d{2}', str(c))})

        # 결산년에 해당하는 계정만 남기기
        data_fs_y = data_fs_y.loc[:, (data_fs_y.columns == '계정') | (
            data_fs_y.columns.str[-2:].isin(fiscal_data_text))]

        # 클렌징
        data_fs_y_clean = clean_fs(data_fs_y, ticker, 'y')

        # 분기 데이터
        data_fs_q = pd.concat([
            data[1].iloc[:, ~data[1].columns.str.contains('전년동기')], data[3],
            data[5]
        ])
        data_fs_q = data_fs_q.rename(columns={data_fs_q.columns[0]: "계정"})

        data_fs_q_clean = clean_fs(data_fs_q, ticker, 'q')

        # 2개 합치기
        data_fs_bind = pd.concat([data_fs_y_clean, data_fs_q_clean])

        # 재무제표 데이터를 DB에 저장
        args = data_fs_bind.values.tolist()
        mycursor.executemany(query, args)
        con.commit()

    except:

        # 오류 발생 시 해당 종목명을 저장하고 다음 루프로 이동
        print(ticker)
        error_list.append(ticker)

    # 타임슬립 적용                                                               # 7) 무한 크롤링을 방지하기 위해 한 번의 루프가 끝날 때마다 타임슬립을 적용한다.
    #time.sleep(2)

# DB 연결 종료                                                                   # 8) 모든 작업이 끝나면 DB와의 연결을 종료한다.
engine.dispose()
con.close()

# 작업이 끝난 후 SQL의 kor_fs 테이블을 확인해 보면 전 종목의 재무제표가 저장되어 있다.
# 시간이 지나 앞의 코드를 다시 실행하면 upsert 형식을 통해 수정된 재무제표는 update를, 새로 입력된 재무제표는 insert를 한다.

#%% 6. 가치지표 계산
"""6. 가치지표 계산
위에서 구한 재무제표 데이터를 이용해 가치지표를 계산할 수 있다. 흔히 가치지표로는 'PER', 'PBR', 'PCR', 'PSR', 'DY'가 사용된다.

[표 10.2 가치지표의 종류]
PER : Price to Earnings Ratio    / 필요한 재무제표 데이터: Earnings(순이익)
PBR : Price to Book Ratio        / Book Value(순자산)
PCR : Price to Cash Flow Ratio   / Cash Flow(영업활동현금흐름)
PSR : Price to Sales Ratio       / Sales(매출액)
DY  : Dividend Yield             / Dividend(배당)

가치지표의 경우 연간 재무제표 기준으로 계산할 경우 다음 재무제표가 발표될 때까지 1년이나 기다려야 한다.
반면 분기 재무제표는 3개월마다 발표되므로 최신 정보를 훨씬 빠르게 반영할 수 있다는 장점이 있으므로
일반적으로 최근 4분기 데이터를 이용해 계산하는 TTM(Trailing Twelve Months) 방법을 많이 사용한다.
먼저, 예제로 삼성전자의 가치지표를 계산해 보자.
"""


#%%  삼성전자 가치지표 - 데이터 불러오기
# 패키지 불러오기
from sqlalchemy import create_engine
import pandas as pd

# DB 연결                                                                         # 1) DB에 연결한다.
engine = create_engine('mysql+pymysql://root:handdmom1!@127.0.0.1:3306/stock_db')

# 티커 리스트                                                                     # 2) 티커 리스트를 불러온다.
ticker_list = pd.read_sql("""
select * from kor_ticker
where 기준일 = (select max(기준일) from kor_ticker)
      and 종목구분 = '보통주';
""", con=engine)

# 삼성전자 분기 재무제표                                                            # 3) 삼성전자의 분기공시 자료 중 가치지표 계산에 필요한 항목들만 불러온다.
sample_fs = pd.read_sql("""
select * from kor_fs
where 공시구분 = 'q'
and 종목코드 = '005930'
and 계정 in ('당기순이익', '자본총계', '영업활동으로인한현금흐름', '매출액(수익)');
""", con=engine)

sample_fs['계정'] = sample_fs['계정'].replace({'매출액(수익)': '매출액', '자본총계': '자본'})

engine.dispose()                                                                  # 4) DB와의 연결을 종료한다.


#%% 재무제표 정렬 (종목코드, 계정, 기준일 순)
sample_fs = sample_fs.sort_values(['종목코드', '계정', '기준일'])
# 먼저, sort_values() 함수를 통해 재무제표 데이터를 종목코드, 계정, 기준일 순으로 정렬한다.

sample_fs.head()


#%% TTM 구하기
sample_fs['ttm'] = sample_fs.groupby(
    ['종목코드', '계정'], as_index=False)['값'].rolling(window=4,
                                                      min_periods=4).sum()['값']
# 1) 종목코드와 계정을 기준으로 groupby() 함수를 통해 그룹을 묶으며, as_index=False를 통해 그룹 라벨을 인덱스로 사용하지 않는다.
# 2) rolling() 메서드를 통해 4개 기간씩 합계를 구하며, min_periods 인자를 통해 데이터가 최소 4개는 있을 경우에만 값을 구한다.
#    즉 4개 분기 데이터를 통해 TTM 값을 계산하며, 12개월치 데이터가 없을 경우는 계산을 하지 않는다.

sample_fs


#%% 자본은 평균, 최근 데이터만 선택
import numpy as np

sample_fs['ttm'] = np.where(sample_fs['계정'] == '자본',
                            sample_fs['ttm'] / 4, sample_fs['ttm'])        # 1) '자본' 항목은 재무상태표에 해당하는 항목이므로 합이 아닌 4로 나누어 평균을 구하며, 타 항목은 4분기 기준 합을 그대로 사용한다.
sample_fs = sample_fs.groupby(['계정', '종목코드']).tail(1)                  # 2) 계정과 종목코드별 그룹을 나눈 후 tail(1) 함수를 통해 가장 최근 데이터만 선택한다.

sample_fs.head()


#%% 시가총액과 합치기
# 가치지표 중 분모에 해당하는 재무제표 값들을 계산했으며, 분자에 해당하는 시가총액은 티커 리스트에서 구할 수 있다.
sample_fs_merge = sample_fs[['계정', '종목코드', 'ttm']].merge(
    ticker_list[['종목코드', '시가총액', '기준일']], on='종목코드')
sample_fs_merge['시가총액'] = sample_fs_merge['시가총액'] / 100000000

sample_fs_merge.head()
# 위에서 계산한 테이블과 티커 리스트 중 필요한 열만 선택해 테이블을 합친다.
# 재무제표 데이터의 경우 단위가 억 원인 반면 시가총액은 원이므로, 시가총액을 억으로 나눠 단위를 맞춰 준다.


#%% 가치지표 계산
sample_fs_merge['value'] = sample_fs_merge['시가총액'] / sample_fs_merge['ttm']
sample_fs_merge['지표'] = np.where(
    sample_fs_merge['계정'] == '매출액', 'PSR',
    np.where(
        sample_fs_merge['계정'] == '영업활동으로인한현금흐름', 'PCR',
        np.where(sample_fs_merge['계정'] == '자본', 'PBR',
                 np.where(sample_fs_merge['계정'] == '당기순이익', 'PER', None))))

sample_fs_merge
# 투자(시가총액)를 분모(TTM 기준 재무제표 데이터)로 나누어 가치지표를 계산한 후, 각 지표명을 입력한다.
# 마지막으로, 배당수익률의 경우 티커 리스트의 데이터를 통해 계산할 수 있다.


#%% 배당수익률 (삼성전자)
# [주의] 이 셀은 사진에서 코드가 잘려 있어 결과(DY 열, 0.0235 = 주당배당금/종가)를 보고 추정해 작성한 것
ticker_list_sample = ticker_list[ticker_list['종목코드'] == '005930'].copy()
ticker_list_sample['DY'] = ticker_list_sample['주당배당금'] / ticker_list_sample['종가']

ticker_list_sample.head()
# 티커 리스트의 각종 데이터 중 주당배당금을 종가로 나누면 현재시점 기준 배당수익률을 쉽게 계산할 수 있다.


#%% 10.6.1 전 종목 가치지표 계산 - SQL
""" 위 코드를 응용해 전 종목 가치지표를 계산해 보자. 먼저, SQL에서 가치지표가 저장될 테이블(kor_value)을 만들어 준다.

use stock_db;

create table kor_value
(
    종목코드 varchar(6),
    기준일 date,
    지표 varchar(3),
    값 double,
    primary key (종목코드, 기준일, 지표)
);
"""


#%% 전 종목: 재무 데이터를 이용해 가치지표 계산
# 패키지 불러오기
import pymysql
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

# DB 연결                                                                      # 1) DB에 연결한다.
engine = create_engine('mysql+pymysql://root:handdmom1!@127.0.0.1:3306/stock_db')
con = pymysql.connect(user='root',
                      passwd='handdmom1!',
                      host='127.0.0.1',
                      db='stock_db',
                      charset='utf8')
mycursor = con.cursor()

# 분기 재무제표 불러오기                                                         # 2) 분기 재무제표와 티커 리스트를 불러온다.
kor_fs = pd.read_sql("""
select * from kor_fs
where 공시구분 = 'q'
and 계정 in ('당기순이익', '자본총계', '영업활동으로인한현금흐름', '매출액(수익)');
""", con=engine)
kor_fs['계정'] = kor_fs['계정'].replace({'매출액(수익)': '매출액', '자본총계': '자본'})

# 티커 리스트 불러오기
ticker_list = pd.read_sql("""
select * from kor_ticker
where 기준일 = (select max(기준일) from kor_ticker)
and 종목구분 = '보통주';
""", con=engine)

engine.dispose()


#%% TTM 구하기
# TTM 구하기
kor_fs = kor_fs.sort_values(['종목코드', '계정', '기준일'])                        # 1) sort_values() 함수를 통해 종목코드, 계정, 기준일순으로 정렬을 한다.
kor_fs['ttm'] = kor_fs.groupby(['종목코드', '계정'], as_index=False)['값'].rolling(
    window=4, min_periods=4).sum()['값']                                         # 2) 종목코드와 계정별 그룹을 묶은 후, 롤링 합을 통해 TTM 값을 구한다.

# 자본은 평균 구하기                                                              # 3) '자본' 항목은 재무상태표에 해당하는 항목이므로 합이 아닌 평균을 구하며, 타 항목은 4분기 기준 합을 그대로 사용한다.
kor_fs['ttm'] = np.where(kor_fs['계정'] == '자본', kor_fs['ttm'] / 4,
                         kor_fs['ttm'])
kor_fs = kor_fs.groupby(['계정', '종목코드']).tail(1)                           # 4) tail(1)을 통해 최근 데이터를 선택한다.

# 이제 티커 리스트의 시가총액 데이터를 이용해 가치지표를 계산해 주도록 한다.


#%% 가치지표 계산
kor_fs_merge = kor_fs[['계정', '종목코드',
                       'ttm']].merge(ticker_list[['종목코드', '시가총액', '기준일']],
                                     on='종목코드')                                 # 1) TTM 기준으로 계산된 재무제표 테이블과 티커 리스트 테이블을 합친다.
kor_fs_merge['시가총액'] = kor_fs_merge['시가총액'] / 100000000                    # 2) 시가총액을 억 원으로 나눈다.

kor_fs_merge['value'] = kor_fs_merge['시가총액'] / kor_fs_merge['ttm']             # 3) 시가총액을 재무데이터 값으로 나누어 가치지표를 계산한 후, 반올림을 한다.
kor_fs_merge['value'] = kor_fs_merge['value'].round(4)
kor_fs_merge['지표'] = np.where(                                                  # 4) 각 계정에 맞게 계정명(PSR, PCR, PER, PBR)을 적는다.
    kor_fs_merge['계정'] == '매출액', 'PSR',
    np.where(
        kor_fs_merge['계정'] == '영업활동으로인한현금흐름', 'PCR',
        np.where(kor_fs_merge['계정'] == '자본', 'PBR',
                 np.where(kor_fs_merge['계정'] == '당기순이익', 'PER', None))))

kor_fs_merge.rename(columns={'value': '값'}, inplace=True)                         # 5) rename() 메서드를 통해 'value'라는 열 이름을 '값'으로 변경한다.
kor_fs_merge = kor_fs_merge[['종목코드', '기준일', '지표', '값']]

kor_fs_merge = kor_fs_merge.replace([np.inf, -np.inf], np.nan)                     # 6) 필요한 열만 선택한 후, inf와 nan을 None으로 변경한다.
kor_fs_merge = kor_fs_merge.astype(object).where(kor_fs_merge.notna(), None)

kor_fs_merge.head(4)


#%% 계산된 가치지표를 데이터베이스에 저장
query = """
    insert into kor_value (종목코드, 기준일, 지표, 값)
    values (%s,%s,%s,%s) as new
    on duplicate key update
    값=new.값
"""

args_fs = kor_fs_merge.values.tolist()
mycursor.executemany(query, args_fs)
con.commit()

# 가치지표를 kor_value 테이블에 upsert 방식으로 저장한다. 마지막으로, 배당수익률의 경우 티커 리스트를 통해 한 번에 계산할 수 있다.


#%% 배당수익률(DY) 계산 후 저장
ticker_list['값'] = ticker_list['주당배당금'] / ticker_list['종가']                 # 1) 주당배당금을 종가로 나누어 배당수익률을 계산한 후, 반올림을 한다.
ticker_list['값'] = ticker_list['값'].round(4)
ticker_list['지표'] = 'DY'                                                          # 2) '지표'열에 'DY'라는 글자를 입력한다.
dy_list = ticker_list[['종목코드', '기준일', '지표', '값']]                          # 3) 원하는 열만 선택한다.
dy_list = dy_list.replace([np.inf, -np.inf], np.nan)                               # 4) inf와 nan은 None으로 변경한다.
dy_list = dy_list.astype(object).where(dy_list.notna(), None)
dy_list = dy_list[dy_list['값'] != 0]                                               # 5) 주당배당금이 0원인 종목은 값이 0으로 계산되므로, 이를 제외한 종목만 선택한다.

dy_list.head()

args_dy = dy_list.values.tolist()
mycursor.executemany(query, args_dy)
con.commit()

engine.dispose()
con.close()
# 배당수익률 역시 kor_value 테이블에 upsert 방식으로 저장한 후, DB와의 연결을 종료한다.

