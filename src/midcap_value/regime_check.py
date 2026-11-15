# -*- coding: utf-8 -*-
"""금리 신호가 '시장 성격 변화' 때문에 2001년 이후에만 나타나는지 확인.
성격 = 직전 36개월 '주식 월수익 vs 10년 금리 월변화' 상관(월말에 아는 값).
  상관 > 0: 금리가 오를 때 주식도 오름(성장 기대가 금리를 움직이는 시기, 2000년 이후 흔함)
  상관 < 0: 금리가 오를 때 주식이 빠짐(물가 충격 시기, 1970~90년대 흔함)
가설이 맞다면 1963~2000에서도 상관 > 0 인 달에는 금리 신호가 2001년 이후처럼 작동해야 한다."""
import io, sys
import numpy as np, pandas as pd
import data as DT, backtest as B
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ff = DT.ff6_daily(); R = DT.us_returns()
fm = (1 + ff).resample("ME").prod() - 1
mkt = fm["Mkt-RF"] + fm["RF"]
y = DT.dgs10().resample("ME").last()
dy = y.diff()
corr = mkt.rolling(36).corr(dy.reindex(mkt.index))
rv, rs = B.factor_reg(R["VOE"], ff), B.factor_reg(R["SPY"], ff)
syn = sum((rv[c] - rs[c]) * fm[c] for c in B.FF6)
M = pd.DataFrame({t: (1 + R[t].dropna()).resample("ME").prod() - 1 for t in ["IWS", "SPY", "QQQ"]})
up = (y.diff(3) > 0)
rows = []
def test(name, x, per, a, z):
    nx = x.shift(-1)
    for reg, m in [("상관>0", corr > 0), ("상관<0", corr <= 0)]:
        d = pd.DataFrame({"n": nx, "u": up.reindex(nx.index), "m": m.reindex(nx.index)})[a:z].dropna()
        d = d[d.m.astype(bool)]
        dif, t, n = B.diff_t(d.n, d.u.astype(bool), 3)
        rows.append(dict(대상=name, 구간=per, 성격=reg, 달=len(d), 금리상승달=n, 차이_연율=dif * 12 if dif == dif else np.nan, t=t))
for per, a, z in [("1963~2000", "1963-01-01", "2000-12-31"), ("2001~2026", "2001-01-01", "2026-12-31"), ("전체", "1963-01-01", "2026-12-31")]:
    test("합성 중형가치−S&P500", syn, per, a, z)
    test("HML", fm["HML"], per, a, z)
test("IWS−SPY(실제)", (M.IWS - M.SPY).dropna(), "2001~2026", "2001-01-01", "2026-12-31")
test("IWS−QQQ(실제)", (M.IWS - M.QQQ).dropna(), "2001~2026", "2001-01-01", "2026-12-31")
out = pd.DataFrame(rows); out.to_csv("results/r_regime_corr.csv", encoding="utf-8-sig")
pd.set_option("display.width", 200); print(out.round(3).to_string())
c = corr.dropna(); print("상관>0 비율: 1963~2000", round((c[:"2000"] > 0).mean(), 2), " 2001~", round((c["2001":] > 0).mean(), 2))
print((c > 0).groupby(c.index.year // 10 * 10).mean().round(2).to_string())
