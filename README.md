<div align="center">

# ⚙️ quant-strategy-lab

**규칙 기반 매매 전략을 코드로 구현하고 검증하는 저장소**

![Scanner](https://img.shields.io/badge/Scanner-완료-2da44e)
![Backtest](https://img.shields.io/badge/Backtest-진행%20예정-bf8700)
![Python](https://img.shields.io/badge/Python-3-3776ab)

</div>

---

## 🗂️ 저장소 구성

전략 하나는 `src/` 아래 폴더 하나로 추가합니다.

| 폴더 | 내용 | 상태 |
|---|---|---|
| [📈 src/heikin_ashi](src/heikin_ashi) | 하이킨아시 신호 스캐너 + 보유종목 점검 | ✅ 스캐너 ,백테스트 완료 |
| [📈 src/buttett_value | 한국주식 워렌버핏식 가치평가 검증  | ✅ 스캐너 ,백테스트 완료 || [📈 src/heikin_ashi]
| (src/Benjamin_graham_value) | 하이킨아시 신호 스캐너 + 보유종목 점검 | ✅ 스캐너 ,백테스트 완료 
| [🗄️ data_pipeline](data_pipeline) | 국내·미국 시장 데이터 수집 코드 (MySQL 저장) | ✅ 수집 코드 완료 |
| [📄 data](data) | 스캐너 입력 파일 설명 (실제 데이터는 미포함) | - |


---


---

## ▶️ 실행 방법

```
pip install -r requirements.txt
python src/heikin_ashi/heikin_scanner.py
```

입력 파일은 직접 준비해야 합니다. → [data/README.md](data/README.md)

---
### 그레이엄·버핏 백테스트

실행 순서 (Spyder, conda 환경 ta_env, #%% 셀 단위):

1. data_pipeline/kr_collect_core.py : 종목·가격·재무 수집 (MySQL stock_db)
2. data_pipeline/kr_dart_long.py : OpenDART 장기 재무와 가격 보강 (하루 호출 한도에서 멈추면 다시 실행)
3. data_pipeline/kr_delisted.py : 상장폐지 종목
4. src/value_screen/graham_screen_backtest.py
5. src/value_screen/buffett_screen_backtest.py

환경변수 (코드에 직접 쓰지 않는다): DB_PW, DART_KEY, KRX_ID, KRX_PW
윈도우: setx DB_PW "비밀번호" 후 Spyder 재시작

DB 테이블: kor_ticker, kor_price, kor_fs, kor_sector, crawl_log,
dart_corp, dart_fs, kor_delisted, kor_price_delisted
---

## 🔒 보안·데이터 원칙

비밀번호, 계좌번호, API 키, 실제 보유종목 파일, 결과 엑셀은 저장소에 올리지 않습니다 (`.gitignore`).