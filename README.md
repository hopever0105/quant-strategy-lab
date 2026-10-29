<div align="center">

# ⚙️ quant-strategy-lab

**규칙 기반 매매·가치투자 전략을 코드로 구현하고 검증하는 저장소**

![Scanner](https://img.shields.io/badge/Scanner-완료-2da44e)
![Python](https://img.shields.io/badge/Python-3-3776ab)
![MySQL](https://img.shields.io/badge/MySQL-stock__db-4479a1)

</div>

---

## 🗂️ 저장소 구성

전략 하나는 `src/` 아래 폴더 하나로 추가합니다.

| 폴더 | 내용 | 상태 |
|---|---|---|
| [📈 src/heikin_ashi](src/heikin_ashi) | 하이킨아시 신호 스캐너 + 보유종목 점검 | ✅ 스캐너 · 🔨 백테스트 진행 중 |
| [💎 src/value_screen](src/value_screen) | 한국 주식 그레이엄·버핏식 가치 스크리닝과 백테스트 | ✅ 코드 작성 |
| [📉 src/macd_rsi_swing](src/macd_rsi_swing) | MACD 골든크로스·RSI 다이버전스 스윙 전략 백테스트 (한국 1999~2026, 상장폐지 포함) | ✅ 백테스트·보고서 — 비용 후 우위 없음 |
| [🧮 src/els_stepdown](src/els_stepdown) | KOSPI200·S&P500 2자산 스텝다운 ELS 몬테카를로 평가 (콴토, 해석해 검증, 민감도, 과거 경로 비교) | ✅ 평가·보고서 — 공정 쿠폰 연 8.07%(기본) |
| [🔁 src/cb_lsmc](src/cb_lsmc) | 전환사채 LSMC 평가 (미국형 전환권·조기상환청구권·리픽싱, 해석해·이항트리 검증, TF 신용위험) | ✅ 평가·보고서 — CB 가치 134.13(액면 100) |
| [🗄️ data_pipeline](data_pipeline) | 국내·미국 시장 데이터 수집 코드 (MySQL 저장) | ✅ 수집 코드 완료 |
| [📄 data](data) | 스캐너 입력 파일 설명 (실제 데이터는 미포함) | - |

---

## ▶️ 실행 방법

**하이킨아시 스캐너**

```bash
pip install -r requirements.txt
python src/heikin_ashi/heikin_scanner.py
```

입력 파일은 직접 준비해야 합니다. → [data/README.md](data/README.md)

**그레이엄·버핏 백테스트** (Spyder, conda 환경 `ta_env`, `#%%` 셀 단위)

1. `data_pipeline/kr_collect_core.py`: 종목·가격·재무 수집 (MySQL `stock_db`)
2. `data_pipeline/kr_dart_long.py`: OpenDART 장기 재무와 가격 보강 (하루 호출 한도에서 멈추면 다시 실행)
3. `data_pipeline/kr_delisted.py`: 상장폐지 종목
4. `src/value_screen/graham_screen_backtest.py`
5. `src/value_screen/buffett_screen_backtest.py`

**환경변수** (코드에 직접 쓰지 않습니다): `DB_PW`, `DART_KEY`, `KRX_ID`, `KRX_PW`
윈도우: `setx DB_PW "비밀번호"` 실행 후 Spyder를 다시 켭니다.

**DB 테이블**: `kor_ticker`, `kor_price`, `kor_fs`, `kor_sector`, `crawl_log`, `dart_corp`, `dart_fs`, `kor_delisted`, `kor_price_delisted`

---

## 🔒 보안·데이터 원칙

비밀번호, 계좌번호, API 키, 실제 보유종목 파일, 결과 엑셀은 저장소에 올리지 않습니다 (`.gitignore`).

## 🔗 관련 저장소

- 지수 개발·ETF 기획·ETF 중복 분석: [etf-solution-lab](https://github.com/hopever0105/etf-solution-lab)
- 책 실습 코드 정리: study-notes (비공개)