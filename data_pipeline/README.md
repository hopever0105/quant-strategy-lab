# 🗄️ data_pipeline

국내·미국 주식의 **티커, 주가, 재무 데이터**를 수집해 MySQL(`stock_db`)에 저장하는 코드입니다.

---

## 📚 출처 표기

- 테이블 설계, 처리 흐름(티커 → 주가 → 재무 → 가치지표), upsert 저장 방식은 파이썬을 이용한 퀀트 투자 포트폴리오 만들기 (이한열)을 참고했습니다. 
- 상기 서적은 2022년 버전으로 현재는 구현불가능한 영역이 많은 바,  데이터 소스와 정리 로직은 직접 교체·수정했습니다.

---

## 📂 파일

| 파일 | 소스 | 내용 |
|---|---|---|
| `kr_fnguide_pykrx.py` | KRX(pykrx), WICS, 네이버 차트, FnGuide | 국내 티커·섹터·주가·재무제표·가치지표(PER/PBR/PCR/PSR/DY) 수집, DB 저장 |
| `kr_tradingview.py` | TradingView 스크리너, tvDatafeed, yahooquery | 국내 티커, 주가, 현재 시점 재무 지표 |
| `us_tradingview.py` | 동일 | 미국 티커, 주가, 현재 시점 재무 지표 |
| `fdr_kr_us.py` | FinanceDataReader, 네이버 요약재무, yahooquery | 국내·미국 티커, 주가, 재무 |
| kr_collect_core.py | 국내 종목 목록, 시가총액, 수정주가, 업종, FnGuide 재무를 MySQL에 저장|
| kr_dart_long.py| OpenDART 2013~2025 연간 재무와 2015년 이전 가격 보강| 
|  kr_delisted.py | 2015년 이후 상장폐지 종목 목록과 가격 | KRX 로그인 세션 사용| 

---

## 🔧 책 코드에서 직접 바꾼 부분 (`kr_fnguide_pykrx.py`)

- FnGuide 재무제표를 `read_html` 대신 JSON 호출 방식으로 교체
- KRX 로그인이 필요해져 pykrx 인증 세션 사용
- 결산월을 못 찾을 때의 예비 로직 추가
- 재무제표 열 이름이 `YYYY/MM` 형태라 날짜 변환 형식 수정

---

## ⚠️ 한계 (꼭 알고 쓸 것)

- TradingView 코드는 비공식 패키지 기반이라 사이트가 바뀌면 동작하지 않을 수 있고, 대량 수집은 약관상 권장되지 않습니다. 개인 연구용입니다.
- TradingView 재무 지표는 시계열이 아니라 현재 시점 1개 값입니다.
- 모든 코드는 DB 비밀번호를 환경변수 `DB_PW`로 읽습니다.
- 이 코드는 데이터 수집용이며, 수집한 데이터로 검증한 결과는 포함하지 않습니다.