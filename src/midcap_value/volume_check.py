# -*- coding: utf-8 -*-
"""거래량으로 순환매·리밸런싱 확인 (2001~2026, 야후 일봉 거래량)
- 거래대금 비중: 가치 ETF(IWS+IWD) 거래대금 ÷ (가치 + QQQ) — 다음 달 비중이 신호 후 늘어나나(관심·자금 이동)
- 매수 우위 대용: (오른 날 거래량 − 내린 날 거래량) ÷ 전체 거래량, 월별 — 가치 ETF는 올라가고 QQQ는 내려가나
거래량에는 사는 쪽과 파는 쪽이 같아 실제 매수량은 알 수 없음. ETF 설정·환매(실제 자금 유입) 이력은 무료 자료 없음."""
import io, sys
import numpy as np, pandas as pd
import yfinance as yf
import data as DT, backtest as B
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
fn = DT.CACHE / "us_ohlcv.pkl"
if fn.exists():
    d = pd.read_pickle(fn)
else:
    d = yf.download(["IWS", "IWD", "QQQ", "SPY"], start="2001-01-01", end=DT.END, auto_adjust=True, progress=False, threads=False)
    d.index = pd.to_datetime(d.index).tz_localize(None); d.to_pickle(fn); DT.log("ETF 거래량 저장")
C, V = d["Close"], d["Volume"]
dv = (C * V).resample("ME").sum()
share = ((dv.IWS + dv.IWD) / (dv.IWS + dv.IWD + dv.QQQ))["2001-09":]
ret = C.pct_change()
sv = (np.sign(ret) * V).resample("ME").sum() / V.resample("ME").sum()
y = DT.dgs10().resample("ME").last()
sig = (y.diff(3) > 0).reindex(share.index)
rows = []
def comp(nm, x):
    nx = x.shift(-1)
    m = sig.notna() & nx.notna()
    dif, t, n = B.diff_t(nx[m], sig[m].astype(bool), 3)
    rows.append(dict(지표=nm, 신호뒤_다음달=nx[m][sig[m].astype(bool)].mean(), 그밖=nx[m][~sig[m].astype(bool)].mean(), 차이=dif, t=t))
comp("가치 ETF 거래대금 비중(수준)", share)
comp("가치 ETF 거래대금 비중 변화(다음달−이번달)", share.diff())
for t in ["IWS", "IWD", "QQQ"]:
    comp(f"{t} 매수 우위 대용", sv[t]["2001-09":])
comp("매수 우위 차이 (IWS+IWD)/2 − QQQ", ((sv.IWS + sv.IWD) / 2 - sv.QQQ)["2001-09":])
# 리밸런싱 가설: 신호 기간에 QQQ가 더 많이 오른 경우일수록 효과 커야 함
M = (1 + ret).resample("ME").prod() - 1
rel3 = ((1 + M.QQQ).rolling(3).apply(np.prod, raw=True) - (1 + M.IWS).rolling(3).apply(np.prod, raw=True))
nxt = (M.IWS - M.QQQ).shift(-1)
s = (sig == True).reindex(nxt.index).fillna(False).astype(bool); rel3 = rel3.reindex(nxt.index)
x = nxt[s].dropna(); hi = rel3[s].reindex(x.index) > rel3[s].median()
dif, t, _ = B.diff_t(x, hi, 3)
rows.append(dict(지표="리밸런싱: 신호 중 QQQ가 직전 3개월 더 많이 앞선 절반 − 나머지 (다음달 IWS−QQQ, 연율)", 신호뒤_다음달=x[hi].mean()*12, 그밖=x[~hi].mean()*12, 차이=dif*12, t=t))
o = pd.DataFrame(rows); o.to_csv("results/m7_volume.csv", encoding="utf-8-sig")
pd.set_option("display.width", 220); print(o.round(4).to_string())
