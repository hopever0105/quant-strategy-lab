"""미국(S&P500 현재 구성종목, 생존편향!) 참고 백테스트: 같은 규칙, 가격제한·거래세 없음, 왕복 0.1%, 20일 평균 거래대금 500만달러 이상.
실행: python quant/strategies/macd_rsi_swing/us_run.py  → results/metrics_us.json"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core, run

c = run.db()
us = pd.read_sql("SELECT `날짜` d,`티커` code,`시가` o,`고가` h,`저가` l,`종가` c,`수정종가` ac,`거래량` v FROM us_price", c, parse_dates=["d"])
ETF = {"SPY","QQQ","IWM","DIA","TLT","IEF","SHY","LQD","HYG","GLD","SLV","DBC","EEM","EFA","VNQ","XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLB","XLU","VTI","VOO"}
spy = us[us.code == "SPY"].set_index("d")["ac"]
us = us[~us.code.isin(ETF)].dropna(subset=["c", "ac", "o", "h", "l"])
us = us[(us.c > 0) & (us.ac > 0)]
f = us.ac / us.c                               # 수정계수(배당·분할 반영)
us = us.assign(o=us.o * f, h=us.h * f, l=us.l * f, c=us.ac, v=us.v / f)
us["amt"] = us.c / f * us.v * f / f if False else us.c * us.v * f / f      # 원 종가×거래량(분할 조정 전 명목금액 근사)
us["amt"] = (us.c / f) * (us.v * f)                                          # = 원종가 × 원거래량
us["src"] = "U"
P = dict(core.P0); P.update(us=True, liq_amt=5e6, comm=0.0005, slip=0.0005)
dates, tks = core.prepare(us[["d", "code", "o", "h", "l", "c", "v", "amt", "src"]], P)
cd = run.db()
rf = pd.read_sql("SELECT `날짜` d, `금리` r FROM cd91_rate", cd, parse_dates=["d"]).set_index("d")["r"] * 0.0     # 미국 무위험 대용 없음 → 0
bm = spy.reindex(pd.DatetimeIndex(dates)).ffill()
run.EVAL_START = pd.Timestamp("1996-01-04")
out = dict(meta=dict(n_tickers=len(tks), start=str(pd.Timestamp(dates[0]).date()), end=str(pd.Timestamp(dates[-1]).date()), stats=core.STATS))
trs = run.run_variant(dates, tks, {}, P)
res = {}
eqs = {}
for k, tr in trs.items():
    eq, expo, acc = core.portfolio(tr, tks, dates, {}, P)
    eqs[k] = eq
    res[k] = dict(trade=run.tstats(tr), exposure=float(expo.mean()),
                  periods={"전체": run.metrics(eq, dates, rf, bm), "1996~2012": run.metrics(eq, dates, rf, bm, "1996-01-04", "2012-12-31"), "2013~2026": run.metrics(eq, dates, rf, bm, "2013-01-01", None)})
out["base"] = res
pd.DataFrame(eqs, index=pd.DatetimeIndex(dates)).assign(SPY=bm.values).to_csv(HERE / "results" / "equity_us.csv", encoding="utf-8-sig")
(HERE / "results" / "metrics_us.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
for k, v in res.items():
    m = v["periods"]["전체"]; t = v["trade"]
    print(k, "trades", t["n"], "avgR %.3f win %.3f" % (t["avg_R"], t["win"]), "CAGR %.3f Sh %.2f MDD %.3f | SPY CAGR %.3f MDD %.3f" % (m["CAGR"], m["sharpe"], m["MDD"], m["bm_CAGR"], m["bm_MDD"]))
