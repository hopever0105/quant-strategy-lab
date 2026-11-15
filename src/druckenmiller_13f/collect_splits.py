# -*- coding: utf-8 -*-
"""3-2) 야후 분할 이력 수집 → 분할 전 실제 종가 복원용 (시가총액 = 공시 주식 수 × 그날 실제 종가).
야후 'Close'는 이후 분할까지 반영된 값이라 그대로 쓰면 과거 시가총액이 틀린다.
저장: cache/druck13f/splits/<티커>.pkl (Series: 날짜 → 분할비). 이어받기 가능.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.stdout.reconfigure(encoding="utf-8")
QUANT = Path(r"C:\Users\arlo0\OneDrive\바탕 화면\취업\quant")
C = QUANT / "cache" / "druck13f"
PX = C / "px"
SP = C / "splits"
SP.mkdir(exist_ok=True)
LOG = QUANT / "logs" / "druck13f.log"


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + "[splits] " + msg + "\n")


def main():
    have = [p.stem for p in PX.glob("*.pkl") if len(pd.read_pickle(p))]
    todo = [t for t in have if not (SP / f"{t}.pkl").exists() and not t.startswith("^") and "=" not in t]
    log(f"분할 이력 남은 것 {len(todo)}")
    for i in range(0, len(todo), 40):
        b = todo[i:i + 40]
        try:
            d = yf.download(b, start="2009-01-01", actions=True, auto_adjust=False, group_by="ticker",
                            threads=True, progress=False)
        except Exception as e:  # noqa
            log(f"실패 {e}")
            time.sleep(30)
            continue
        for t in b:
            try:
                x = d[t] if len(b) > 1 else d
                s = x["Stock Splits"]
                s = s[s.fillna(0) > 0]
            except Exception:
                s = pd.Series(dtype=float)
            s.to_pickle(SP / f"{t}.pkl")
        log(f"분할 {min(i + 40, len(todo))}/{len(todo)}")
    print("완료")


if __name__ == "__main__":
    main()
