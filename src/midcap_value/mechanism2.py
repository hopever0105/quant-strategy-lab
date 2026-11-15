# -*- coding: utf-8 -*-
"""대안 설명: 금리가 오르던 3개월 동안 성장주가 앞섰고(같은 달 SPY−QQQ 민감도 음수), 다음 달 그 반작용(되돌림)이 온 것인가.
→ 직전 3개월 상대수익을 회귀에 넣어도 금리 신호가 남는지, 직전 3개월 상대수익만으로도 되돌림이 있는지 확인.
안정성: 2001~2026 안에서 5년 블록별 신호 효과 방향."""
import io, sys
import numpy as np, pandas as pd
import statsmodels.api as sm
import data as DT
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
pd.set_option("display.width", 220)
R = DT.us_returns()
M = pd.DataFrame({t: (1 + R[t].dropna()).resample("ME").prod() - 1 for t in R.columns})
y = DT.dgs10().resample("ME").last()
up3 = (y.diff(3) > 0).astype(float)
def nw(yv, X):
    d = pd.concat([yv.rename("y"), X], axis=1).dropna()
    return sm.OLS(d.y, sm.add_constant(d.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": 3}), len(d)
rows, blk = [], []
for nm, x in {"IWS−SPY": M.IWS - M.SPY, "IWS−QQQ": M.IWS - M.QQQ, "IWD−SPY": M.IWD - M.SPY}.items():
    x = x["2001":].dropna()
    past3 = x.rolling(3).sum().rename("직전3개월상대")
    X = pd.concat([up3.reindex(x.index).rename("신호"), past3], axis=1)
    mA, _ = nw(x.shift(-1), past3.to_frame())
    mB, n = nw(x.shift(-1), X)
    rows.append(dict(대상=nm, 달=n, 되돌림만_계수=mA.params["직전3개월상대"], 되돌림만_t=mA.tvalues["직전3개월상대"],
                     신호_통제후_연율=mB.params["신호"] * 12, 신호_t=mB.tvalues["신호"], 직전3개월_t=mB.tvalues["직전3개월상대"],
                     신호중_직전3개월평균=past3[X.신호 == 1].mean(), 나머지_직전3개월평균=past3[X.신호 == 0].mean()))
    nx = x.shift(-1)
    for y0 in range(2001, 2026, 5):
        d = pd.DataFrame({"n": nx, "s": up3.reindex(x.index)})[f"{y0}":f"{y0+4}"].dropna()
        blk.append(dict(대상=nm, 블록=f"{y0}~{y0+4}", 차이_연율=(d.n[d.s == 1].mean() - d.n[d.s == 0].mean()) * 12, 상승달=int(d.s.sum()), 달=len(d)))
o, b = pd.DataFrame(rows), pd.DataFrame(blk)
o.to_csv("results/m4_reversal.csv", encoding="utf-8-sig"); b.to_csv("results/m5_blocks.csv", encoding="utf-8-sig")
print(o.round(4).to_string()); print(b.pivot(index="블록", columns="대상", values="차이_연율").round(3).to_string())
