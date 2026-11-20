# -*- coding: utf-8 -*-
"""매년 4월 말 전 종목 EPS·BPS·PER·PBR (pykrx, KRX) → ROE = EPS/BPS. 27건(2000~2026).
KRX 규칙: krx_guard.install()을 pykrx import 전에 먼저. 실패 1번이면 재시도 없이 즉시 중지.
이어받기: quant/cache/larry/krx_fund/<날짜>.csv 가 있으면 건너뜀."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOB = next((p for p in HERE.parents if (p / "quant" / "krx_guard.py").exists()), HERE.parents[1])  # 저장소에서는 상위 폴더 기준
sys.path.insert(0, str(JOB / "quant"))
import krx_guard  # noqa: E402
krx_guard.install()

import pandas as pd  # noqa: E402
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402

OUT = D.CACHE / "krx_fund"
OUT.mkdir(parents=True, exist_ok=True)


def dates():
    k, _ = D.kospi()
    idx = k.index
    return [idx[(idx.year == y) & (idx.month == 4)][-1] for y in range(2000, 2027)]


RUN_CAP = 40                       # 이번 실행 요청 상한(27건 + 로그인). 가드 한도와 별개로 이 스크립트만의 안전장치
LOG = JOB / "quant" / "logs" / "larry_krx.log"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {m}\n")
    print(m, flush=True)


def stop(note, code):
    """실패 1번이면 일시중지 파일을 다시 만들고 끝낸다 → 원인 확인 전까지 KRX 요청이 다시 막힘"""
    log(f"중지: {note}")
    if not krx_guard.paused():
        krx_guard._pause(f"larry krx_fund 실패 - {note}")
        log("krx_paused.txt 다시 생성(원인 확인 후 직접 지울 것)")
    sys.exit(code)


def main():
    start = krx_guard.status()["count"]
    log(f"시작. 오늘 KRX 사용량 {krx_guard.status()}")
    try:
        from pykrx import stock          # 가드 설치 후 import (import 시 로그인 요청 발생)
    except krx_guard.KRXBlocked as e:
        stop(f"로그인 단계 KRXBlocked: {e}", 2)
    except Exception as e:
        stop(f"로그인 단계 오류: {e}", 3)
    ds = dates()
    for d in ds:
        fn = OUT / f"{d:%Y%m%d}.csv"
        if fn.exists():
            continue
        if krx_guard.status()["count"] - start >= RUN_CAP:
            stop(f"이번 실행 요청 상한 {RUN_CAP}건 도달", 5)
        try:
            df = stock.get_market_fundamental(d.strftime("%Y%m%d"), market="ALL")
        except krx_guard.KRXBlocked as e:
            stop(f"{d.date()} KRXBlocked: {e}", 2)
        except Exception as e:
            stop(f"{d.date()} 오류: {e}", 3)
        if df is None or len(df) == 0:
            stop(f"{d.date()} 빈 응답", 4)
        df.index.name = "code"
        df.to_csv(fn, encoding="utf-8-sig")
        log(f"{d.date()} {len(df)}종목")
    log(f"완료. 이번 실행 요청 {krx_guard.status()['count'] - start}건, 오늘 사용량 {krx_guard.status()}")


if __name__ == "__main__":
    main()
