# -*- coding: utf-8 -*-
"""순환매 가설: 금리 상승 → 기술주 하락 → 돈이 저평가 가치주로 이동.
맞다면 ① 신호 다음 달 기술주(QQQ) 절대수익이 낮고 가치주(IWS) 절대수익은 높아야 함(돈이 옮겨감)
       ② 금리 오르던 3개월 동안 QQQ가 실제로 빠진 경우에 효과가 더 커야 함(순환매 방아쇠)
       ③ 가치주 쪽 거래가 늘어야 함 → 자료 없음(펀드 자금흐름 미보유), 대신 IWS 거래량 대용은 생략"""
import io, sys
import numpy as np, pandas as pd
import data as DT, backtest as B
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
pd.set_option("display.width", 200)
R = DT.us_returns()
M = pd.DataFrame({t: (1 + R[t].dropna()).resample("ME").prod() - 1 for t in ["IWS", "SPY", "QQQ"]})["2001-09":]
y = DT.dgs10().resample("ME").last()
sig = (y.diff(3) > 0).reindex(M.index)
q3 = (1 + M.QQQ).rolling(3).apply(np.prod, raw=True) - 1
nx = M.shift(-1)
rows = []
for nm, m in [("신호 있음", sig == True), ("신호 없음", sig == False),
              ("신호 + 그 3개월 QQQ 하락", (sig == True) & (q3 < 0)), ("신호 + 그 3개월 QQQ 상승", (sig == True) & (q3 >= 0))]:
    d = nx[m].dropna()
    rows.append(dict(경우=nm, 달=len(d), 다음달_QQQ=d.QQQ.mean() * 12, 다음달_SPY=d.SPY.mean() * 12, 다음달_IWS=d.IWS.mean() * 12,
                     IWS_QQQ=(d.IWS - d.QQQ).mean() * 12, IWS_SPY=(d.IWS - d.SPY).mean() * 12))
o = pd.DataFrame(rows)
# t값: 신호 있음 vs 없음 (절대수익), QQQ 하락 vs 상승 (신호 안에서)
for c in ["QQQ", "IWS"]:
    dif, t, _ = B.diff_t(nx[c][sig.notna()].dropna(), sig[sig.notna()].reindex(nx[c].dropna().index).astype(bool), 3)
    print(f"신호 있음−없음 다음달 {c} 절대수익 차이 연 {dif*12:+.3f}, t {t:.2f}")
s = sig == True
x = (nx.IWS - nx.QQQ)[s].dropna()
dif, t, _ = B.diff_t(x, (q3[s].reindex(x.index) < 0), 3)
print(f"신호 안에서 QQQ 하락 경우 − 상승 경우 (IWS−QQQ) 연 {dif*12:+.3f}, t {t:.2f}")
o.to_csv("results/m6_rotation.csv", encoding="utf-8-sig")
print(o.round(3).to_string())
