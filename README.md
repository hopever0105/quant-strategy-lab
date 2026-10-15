# quant-strategy-lab

규칙 기반 매매 전략을 코드로 구현하고 검증하는 저장소입니다.
전략 하나는 `src/` 아래 폴더 하나로 추가합니다.

## 저장소 구성

| 폴더 | 내용 | 상태 |
|---|---|---|
| [src/heikin_ashi](src/heikin_ashi) | 하이킨아시 신호 스캐너 + 보유종목 점검 | 스캐너 완료, 백테스트 진행 예정 |
| [data_pipeline](data_pipeline) | 국내·미국 시장 데이터 수집 코드 (MySQL 저장) | 수집 코드 완료 |
| [data](data) | 스캐너 입력 파일 설명 (실제 데이터는 미포함) | - |

## 현재 상태
- 백테스트·통계 검증 결과는 아직 없습니다.
- 결과가 나오기 전까지 수익률 수치는 기재하지 않습니다.

## 실행 방법
```
pip install -r requirements.txt
python src/heikin_ashi/heikin_scanner.py
```
입력 파일은 직접 준비해야 합니다. [data/README.md](data/README.md) 참고.

## 보안·데이터 원칙
비밀번호, 계좌번호, API 키, 실제 보유종목 파일, 결과 엑셀은 저장소에 올리지 않습니다 (`.gitignore`).