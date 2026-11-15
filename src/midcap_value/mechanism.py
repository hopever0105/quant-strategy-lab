# -*- coding: utf-8 -*-
"""금리 신호의 원인 검증: '금리가 오르면 이익이 먼 미래에 있는 성장주가 더 불리하다(듀레이션)'
예측 ① 같은 달: 금리가 오른 달에 중형가치−S&P500·HML이 플러스 (2001년 이후), 2000년 이전엔 약하거나 없음
예측 ② 금리 추세 지속: 3개월 상승 뒤 다음 달에도 금리가 오르는 경향
예측 ③ ①×② 로 다 설명됨: 다음 달 실제 금리 변화를 회귀에 넣으면 신호 계수가 0 근처로 줄어듦
예측 ④ 크기 순서: 성장 쏠림이 클수록(나스닥100 > S&P500) 같은 달 금리 민감도와 신호 효과가 함께 큼
"""
import io, sys
import numpy as np, pandas as pd
import statsmodels.api as sm
import data as DT, backtest as B
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
pd.set_option("display.width", 220)
ff = DT.ff6_daily(); R = DT.us_returns()
fm = (1 + ff).resample("ME").prod() - 1
y = DT.dgs10().resample("ME").last(); dy = y.diff()
rv, rs = B.factor_reg(R["VOE"], ff), B.factor_reg(R["SPY"], ff)
syn = sum((rv[c] - rs[c]) * fm[c] for c in B.FF6)
M = pd.DataFrame({t: (1 + R[t].dropna()).resample("ME").prod() - 1 for t in R.columns})
up3 = (y.diff(3) > 0).astype(float)
def nw(yv, X, lags=3):
    d = pd.concat([yv.rename("y"), X], axis=1).dropna()
    return sm.OLS(d.y, sm.add_constant(d.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": lags}), len(d)
ser = {"합성 중형가치−S&P500": syn, "HML": fm["HML"], "IWS−SPY": M.IWS - M.SPY, "IWS−QQQ": M.IWS - M.QQQ,
       "SPY−QQQ": M.SPY - M.QQQ, "IWD−SPY": M.IWD - M.SPY, "VO−SPY": M.VO - M.SPY}
P = [("1963~2000", "1963", "2000"), ("2001~2026", "2001", "2026")]
r1, r3 = [], []
for nm, x in ser.items():
    for per, a, z in P:
        xs = x[a:z].dropna()
        if len(xs) < 60: continue
        # ① 같은 달 금리 민감도 (금리 1%p 오를 때 상대수익 %p)
        m, n = nw(xs, dy.reindex(xs.index).rename("dy"))
        r1.append(dict(대상=nm, 구간=per, 달=n, 민감도=m.params["dy"], t=m.tvalues["dy"]))
        # ③ 다음 달 상대수익 ~ 신호 (+ 다음 달 실제 금리 변화)
        nx = xs.shift(-1).rename("y")
        X0 = up3.reindex(xs.index).rename("신호").to_frame()
        X1 = X0.join(dy.shift(-1).reindex(xs.index).rename("다음달dy"))
        m0, _ = nw(nx, X0); m1, n1 = nw(nx, X1)
        r3.append(dict(대상=nm, 구간=per, 달=n1, 신호만_연율=m0.params["신호"] * 12, 신호만_t=m0.tvalues["신호"],
                       통제후_연율=m1.params["신호"] * 12, 통제후_t=m1.tvalues["신호"], 다음달dy_계수=m1.params["다음달dy"], 다음달dy_t=m1.tvalues["다음달dy"]))
# ② 금리 추세 지속
r2 = []
for per, a, z in P:
    d = pd.DataFrame({"s": up3, "n": dy.shift(-1)})[a:z].dropna()
    dif, t, n = B.diff_t(d.n, d.s.astype(bool), 3)
    r2.append(dict(구간=per, 상승뒤_다음달dy_bp=d.n[d.s == 1].mean() * 100, 나머지_bp=d.n[d.s == 0].mean() * 100, 차이_bp=dif * 100, t=t,
                   다음달도_상승비율=(d.n[d.s == 1] > 0).mean()))
o1, o2, o3 = pd.DataFrame(r1), pd.DataFrame(r2), pd.DataFrame(r3)
for o, f in [(o1, "m1_same_month"), (o2, "m2_rate_trend"), (o3, "m3_explained")]:
    o.to_csv(f"results/{f}.csv", encoding="utf-8-sig")
print("① 같은 달 민감도(금리 1%p 상승당 상대수익)"); print(o1.round(3).to_string())
print("② 금리 추세 지속"); print(o2.round(3).to_string())
print("③ 신호 효과가 다음 달 실제 금리 변화로 설명되나"); print(o3.round(3).to_string())
# ④ 2001~ 크기 순서: 같은 달 민감도 vs 신호 효과 순위상관
s = o1[o1.구간 == "2001~2026"].set_index("대상").민감도; e = o3[o3.구간 == "2001~2026"].set_index("대상").신호만_연율
print("④ 2001~ 민감도-신호효과 순위상관", round(s.rank().corr(e.reindex(s.index).rank()), 2), "대상 수", len(s))
