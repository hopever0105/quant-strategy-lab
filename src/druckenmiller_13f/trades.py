# -*- coding: utf-8 -*-
"""4) 분기별 매매 분류: 신규·확대·유지·축소·청산 (주식 수 기준, 분할 보정, 주식/콜/풋 분리).

- 같은 CUSIP·같은 종류(주식/CALL/PUT) 기준으로 직전 분기와 비교.
- 분할 보정: 13F 내재가격(평가액/주식수)의 변화와 야후 분할반영 종가의 변화가 어긋나면 분할로 판단,
  비율을 흔한 분할비(2,3,4,5,10,20,1/2..)에 맞춰 직전 분기 주식 수를 환산.
- 확대/축소 기준: 주식 수 ±10% (보정 후). 13F는 분기말 스냅샷이라 분기 중 사고판 것은 보이지 않음.
결과: results/positions.csv(분기×종목 보유), results/trades.csv(분기×종목 변화), results/splits.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
QUANT = Path(r"C:\Users\arlo0\OneDrive\바탕 화면\취업\quant")
PX = QUANT / "cache" / "druck13f" / "px"
OUT = HERE / "results"
RATIOS = np.array([2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 8, 1 / 10, 1 / 15, 1 / 20, 1.5, 2 / 3])


def px_close(t):
    p = PX / f"{t}.pkl"
    if not isinstance(t, str) or not p.exists():
        return None
    s = pd.read_pickle(p)["close"]
    return s if len(s) else None


def main():
    h = pd.read_csv(OUT / "holdings_raw.csv", dtype={"cusip": str})
    sec = pd.read_csv(OUT / "securities.csv", dtype={"cusip": str}).set_index("cusip")
    h["putcall"] = h["putcall"].fillna("")
    h["leg"] = np.where(h["putcall"] == "", "주식", h["putcall"])
    pos = (h.groupby(["period", "filed", "cusip", "leg"], as_index=False)
           .agg(shares=("shares", "sum"), value=("value_usd", "sum"), name=("name", "first")))
    pos["period"] = pd.to_datetime(pos["period"])
    tot = pos[pos["leg"] == "주식"].groupby("period")["value"].sum()
    pos["weight"] = pos["value"] / pos["period"].map(tot)
    pos["px13f"] = pos["value"] / pos["shares"]
    periods = sorted(pos["period"].unique())
    # 2013Q2 보고서(원본·정정 모두)는 주식 수 칸에 평가액(천 달러)을 적는 기재 오류 → 주식 수 추정
    #   이웃 분기(2013Q1·Q3) 13F 내재가격 × 야후 주가 변화율로 2013Q2 가격을 만들고 평가액/가격
    bad = pos["period"] == pd.Timestamp("2013-06-30")
    est = []
    for i in pos[bad].index:
        c, leg, v = pos.at[i, "cusip"], pos.at[i, "leg"], pos.at[i, "value"]
        t = sec["yf"].get(c) if c in sec.index else None
        s = px_close(t)
        price = np.nan
        for nb in (pd.Timestamp("2013-03-31"), pd.Timestamp("2013-09-30")):
            m = (pos["period"] == nb) & (pos["cusip"] == c) & (pos["leg"] == leg)
            if m.any() and s is not None:
                a, b = s[:pd.Timestamp("2013-06-30")], s[:nb]
                if len(a) and len(b):
                    price = pos.loc[m, "px13f"].iloc[0] * a.iloc[-1] / b.iloc[-1]
                    break
        if not np.isfinite(price) and s is not None and len(s[:pd.Timestamp("2013-06-30")]):
            price = s[:pd.Timestamp("2013-06-30")].iloc[-1]  # 이웃 분기 없음: 야후 가격(이후 분할 반영 주의)
        est.append(v / price if np.isfinite(price) else np.nan)
        pos.at[i, "shares"] = est[-1]
    pos.loc[bad, "px13f"] = pos.loc[bad, "value"] / pos.loc[bad, "shares"]
    print("2013Q2 주식 수 추정", int(np.isfinite(est).sum()), "/", int(bad.sum()))
    filed = pos.groupby("period")["filed"].first()
    pos.to_csv(OUT / "positions.csv", index=False, encoding="utf-8-sig")

    cache = {}
    rows, splits = [], []
    for i, p in enumerate(periods):
        cur = pos[pos["period"] == p].set_index(["cusip", "leg"])
        prev = pos[pos["period"] == periods[i - 1]].set_index(["cusip", "leg"]) if i > 0 else cur.iloc[0:0]
        keys = cur.index.union(prev.index)
        for k in keys:
            c, leg = k
            s1 = cur["shares"].get(k, 0.0)
            s0 = prev["shares"].get(k, 0.0)
            f = 1.0
            if s0 > 0 and s1 > 0 and leg == "주식":
                ip = cur.at[k, "px13f"] / prev.at[k, "px13f"]  # 13F 내재가격 비율(분할 미반영)
                t = sec["yf"].get(c) if c in sec.index else None
                if t not in cache:
                    cache[t] = px_close(t)
                s = cache[t]
                rr = None
                if s is not None:
                    a = s[:p].iloc[-1] if len(s[:p]) else np.nan
                    b = s[:periods[i - 1]].iloc[-1] if len(s[:periods[i - 1]]) else np.nan
                    if np.isfinite(a) and np.isfinite(b) and b > 0:
                        rr = a / b  # 분할 반영된 실제 가격 변화
                g = (rr / ip) if rr else None  # 분할비 추정: 실제변화/내재변화
                if g is None:
                    # 야후 없음: 내재가격이 크게 변하고 주식 수가 반대로 비슷하게 변하면 분할
                    if (ip < 0.6 or ip > 1.7) and abs(np.log((s1 / s0) * ip)) < 0.15:
                        g = 1 / ip
                if g is not None and (g > 1.4 or g < 0.7):
                    j = np.argmin(np.abs(np.log(RATIOS) - np.log(g)))
                    if abs(np.log(RATIOS[j]) - np.log(g)) < 0.12:
                        f = RATIOS[j]
                        splits.append(dict(period=p, cusip=c, name=cur.at[k, "name"], factor=f,
                                           shares_prev=s0, shares_now=s1))
            s0a = s0 * f
            if not (np.isfinite(s0a) and np.isfinite(s1)):
                # 2013Q2 주식 수 추정 불가: 보유 여부로만 판단
                p0 = k in prev.index
                p1 = k in cur.index
                act = "신규" if (p1 and not p0) else ("청산" if (p0 and not p1) else "유지")
            elif s0a == 0 and s1 > 0:
                act = "신규"
            elif s0a > 0 and s1 == 0:
                act = "청산"
            elif s1 > s0a * 1.1:
                act = "확대"
            elif s1 < s0a * 0.9:
                act = "축소"
            else:
                act = "유지"
            w1 = cur["weight"].get(k, 0.0)
            w0 = prev["weight"].get(k, 0.0)
            rows.append(dict(period=p, filed=filed[p], cusip=c, leg=leg,
                             name=(cur["name"].get(k) if k in cur.index else prev["name"].get(k)),
                             shares_prev_adj=s0a, shares=s1, chg_pct=(s1 / s0a - 1) if s0a > 0 else np.nan,
                             split=f, weight_prev=w0, weight=w1, value=cur["value"].get(k, 0.0),
                             px13f=cur["px13f"].get(k, np.nan), action=act))
    tr = pd.DataFrame(rows)
    # 첫 보고서(2011Q4)는 직전 분기가 없으므로 '기존 보유'로 따로 표시
    tr.loc[tr["period"] == periods[0], "action"] = "최초보유"
    # 보유 연속 기간(신규부터 청산까지 분기 수)
    tr = tr.sort_values(["cusip", "leg", "period"]).reset_index(drop=True)
    tr.to_csv(OUT / "trades.csv", index=False, encoding="utf-8-sig")
    sp = pd.DataFrame(splits)
    sp.to_csv(OUT / "splits.csv", index=False, encoding="utf-8-sig")
    print("분할 보정", len(sp), "건")
    if len(sp):
        print(sp[["period", "name", "factor", "shares_prev", "shares_now"]].to_string())
    print(pd.crosstab(tr["leg"], tr["action"]).to_string())


if __name__ == "__main__":
    main()
