# -*- coding: utf-8 -*-
"""수주잔고 전략 — 3) 일별 백테스트 엔진·성과 지표. 실제 주문·매매 없음.

분기말 t0 종가 매수 → H분기 보유(겹치는 묶음 H개 평균), 분기 중엔 가격 따라 비중 변동.
시세가 없거나 끊긴 종목은 그 기간 벤치마크(SPY·코스피)로 대체(대체 비중 기록).
공매도 버전: 상위 묶음 수익 − 하위 묶음 수익 − 대차 비용(연 0.5%).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
DR = HERE.parent / "druckenmiller_13f"
sys.path.insert(0, str(DR))
from engine import simulate  # noqa: E402
from features import load_px  # noqa: E402

QUANT = Path(r"C:\Users\arlo0\OneDrive\바탕 화면\취업\quant")
BAD_US = {"PARA"}  # 합병 후 티커 재사용으로 야후 시세가 깨짐(2024년 1주 16,600달러 등) → 제외
COST = {"us": 0.001, "kr": 0.0025}
BORROW = 0.005


class Prices:
    def __init__(self, mk):
        self.mk = mk
        self._c = {}
        if mk == "kr":
            o = pd.read_pickle(QUANT / "cache" / "ohlc_clean.pkl")
            self.kc = o["c"].astype("float64").where(lambda x: x > 0)
            ks = pd.read_csv(QUANT / "cache" / "kullamaggie" / "KS11.csv", index_col=0, parse_dates=True)["c"].astype(float)
            self.bench = ks
            self.bench_name = "KS11"
        else:
            self.bench = load_px("SPY")["adj"].astype(float)
            self.bench_name = "SPY"

    def __call__(self, t):
        if t in self._c:
            return self._c[t]
        s = None
        if t == self.bench_name:
            s = self.bench
        elif self.mk == "us":
            if t not in BAD_US:
                x = load_px(t)
                s = None if x is None or len(x) == 0 else x["adj"].astype(float)
        else:
            if t in self.kc.columns:
                s = self.kc[t].dropna()
                s = s if len(s) else None
        self._c[t] = s
        return s


def schedule(pk, P):
    t1 = P.groupby("t0")["t1"].first()
    return [(q, t1[q], {c: 1.0 for c in pk.get(q, [])}) for q in sorted(t1.index)]


def run(pk, P, px, H=2, cost=None, mk="us"):
    cost = COST[mk] if cost is None else cost
    sch = schedule(pk, P)
    r, _ = simulate(sch, px, px.bench, cost=cost, sleeves=H, fallback=px.bench_name)
    # 회전율·대체 비중은 1묶음 기준 기록으로 계산
    _, log = simulate(sch, px, px.bench, cost=cost, sleeves=1, fallback=px.bench_name)
    first = sch[0][0]
    return r[r.index > first], log


def metrics(r, rf, a, b):
    x = r[a:b].dropna()
    if len(x) < 60:
        return {}
    rfx = rf.reindex(x.index).fillna(0)
    nav = (1 + x).cumprod()
    yrs = len(x) / 252
    return dict(CAGR=nav.iloc[-1] ** (1 / yrs) - 1, 변동성=x.std() * np.sqrt(252),
                샤프=(x - rfx).mean() / x.std() * np.sqrt(252), MDD=(nav / nav.cummax() - 1).min(), 일수=len(x))


def alpha(r, fac, a, b, cols):
    d = pd.concat([r.rename("r"), fac], axis=1, join="inner")[a:b].dropna()
    if len(d) < 60:
        return {}
    y = d["r"] - d["RF"]
    m = sm.OLS(y, sm.add_constant(d[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    return {"알파(연)": m.params["const"] * 252, "알파_t": m.tvalues["const"], "시장베타": m.params[cols[0]]}


def factors(mk, bench):
    if mk == "us":
        f = pd.read_pickle(QUANT / "cache" / "druck13f" / "ff6_daily.pkl")
        return f, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    rf = pd.read_csv(QUANT / "cache" / "druck13f" / "fred" / "IR3TIB01KRM156N.csv")
    rf.columns = ["date", "v"]
    rf = pd.to_numeric(rf.set_index(pd.to_datetime(rf["date"]))["v"], errors="coerce") / 100 / 252
    br = bench.pct_change()
    f = pd.DataFrame({"RF": rf.reindex(br.index, method="ffill")})
    f["Mkt-RF"] = br - f["RF"]
    return f.dropna(), ["Mkt-RF"]


def rf_series(mk, idx):
    if mk == "us":
        s = load_px("^IRX")["close"] / 100 / 252
    else:
        s = pd.read_csv(QUANT / "cache" / "druck13f" / "fred" / "IR3TIB01KRM156N.csv")
        s.columns = ["date", "v"]
        s = pd.to_numeric(s.set_index(pd.to_datetime(s["date"]))["v"], errors="coerce") / 100 / 252
    return s.reindex(idx, method="ffill").fillna(0)
