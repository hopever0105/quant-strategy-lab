# -*- coding: utf-8 -*-
"""수주잔고 전략 — 1) 데이터 로드. 실제 주문·매매 없음. KRX 요청 없음.

미국: SEC XBRL 수주잔고(RevenueRemainingPerformanceObligation, 없으면 계약부채·이연수익) + S&P500 그 시점 구성종목 분기 패널
      (만드는 코드: ../druckenmiller_13f/backlog2.build, fetch_extra.py us). 시세: 야후 수정종가(분할·배당 보정).
한국: DART 정기보고서 원문 '수주상황' 표의 수주잔고(../druckenmiller_13f/kr_orders.py → kr_backlog.load_backlog)
      + 코스피 상위 200 대용 분기 패널(qpanel.py kr). 시세: quant/cache/ohlc_clean.pkl(상장폐지 포함, 가격 단절 보정본).
결과: data/us_panel.pkl, data/kr_panel.pkl, data/check.txt(수정주가 점검)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
DR = HERE.parent / "druckenmiller_13f"
sys.path.insert(0, str(DR))
QUANT = Path(r"C:\Users\arlo0\OneDrive\바탕 화면\취업\quant")
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)


def build_us():
    from backlog2 import build
    P, R, M = build()
    P = P.dropna(subset=["gap"]).copy()
    P = P[["t0", "t1", "code", "sector", "gap", "fwd_g", "sales_g", "ratio", "ret", "filed", "is_rpo"]]
    P.to_pickle(DATA / "us_panel.pkl")
    R.to_pickle(DATA / "us_qret.pkl")
    M.to_pickle(DATA / "us_qmkt.pkl")
    return P


def build_kr():
    import kr_backlog as K
    from common_q import load
    B = K.load_backlog()
    P, R, M = load("kr")
    P = P[P["code"].isin(set(B["code"]))].copy()
    vals = []
    for code, g in B.dropna(subset=["bl_g"]).groupby("code"):
        g = g.sort_values("filed")
        for q in P.loc[P["code"] == code, "t0"].unique():
            k = g[g["filed"] <= q]
            if len(k) and (q - k["pdate"].iloc[-1]).days < 200:
                vals.append(dict(code=code, t0=q, bl_g=k["bl_g"].iloc[-1], bl=k["bl"].iloc[-1], filed=k["filed"].iloc[-1]))
    V = pd.DataFrame(vals)
    P = P.merge(V, on=["code", "t0"], how="left")
    P["gap"] = P["bl_g"] - P["sales_g"]
    P["sector"] = P["code"].map(B.drop_duplicates("code").set_index("code")["ind"])
    P = P.dropna(subset=["gap"])[["t0", "t1", "code", "sector", "gap", "bl_g", "sales_g", "ret", "filed", "mcap"]]
    P.to_pickle(DATA / "kr_panel.pkl")
    R.to_pickle(DATA / "kr_qret.pkl")
    M.to_pickle(DATA / "kr_qmkt.pkl")
    B.to_csv(DATA / "kr_backlog_reports.csv", index=False, encoding="utf-8-sig")
    return P


def check_prices(us, kr):
    """수정주가 점검: 미국은 야후 수정종가(분할 반영)에서 하루 ±50% 넘는 움직임, 한국은 ohlc_clean의 단절 보정 건수."""
    from features import load_px
    lines = []
    n_jump, n_t = 0, 0
    for t in us["code"].unique():
        x = load_px(t)
        if x is None:
            continue
        r = x["adj"].astype(float).pct_change()["2018-01-01":]
        j = (r.abs() > 0.5).sum()
        n_t += 1
        n_jump += int(j)
    lines.append(f"미국 {n_t}종목: 야후 수정종가(분할·배당 보정). 2018~ 하루 ±50% 넘는 움직임 {n_jump}건(실제 급등락으로 보고 그대로 둠)")
    o = pd.read_pickle(QUANT / "cache" / "ohlc_clean.pkl")
    b = o["breaks"]
    kb = b[b["code"].isin(set(kr["code"]))]
    lines.append(f"한국 {kr['code'].nunique()}종목: ohlc_clean 가격 단절 보정(액면분할·감자·병합) 전체 {len(b)}건 중 이 유니버스 {len(kb)}건")
    (DATA / "check.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    us = build_us()
    kr = build_kr()
    print(f"미국 {len(us)}행 {us['t0'].nunique()}분기 분기당 {us.groupby('t0').size().median():.0f}종목 / "
          f"한국 {len(kr)}행 {kr['t0'].nunique()}분기 분기당 {kr.groupby('t0').size().median():.0f}종목")
    check_prices(us, kr)
