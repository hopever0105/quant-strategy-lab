"""MACD·RSI 스윙 백테스트 실행: 매매법 1 / 매매법 2 / 합친 것 각각 + 비용 2배 + 기간 분할 + 파라미터 민감도 + 무작위 진입 대조.
실행: python quant/strategies/macd_rsi_swing/run.py   (결과: results/*.json, *.csv, *.png)
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core  # noqa: E402

RES = HERE / "results"; RES.mkdir(exist_ok=True)
(core.BASE / "logs").mkdir(exist_ok=True)
LOG = open(core.BASE / "logs" / "macd_rsi_swing_run.log", "a", encoding="utf-8")


def say(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    LOG.write(s + "\n"); LOG.flush()


def db():
    env = dict(l.split("=", 1) for l in (core.BASE / ".env").read_text(encoding="utf-8").splitlines() if "=" in l and not l.startswith("#"))
    return pymysql.connect(host=env["DB_HOST"], user=env["DB_USER"], password=env["DB_PASSWORD"], database=env["DB_NAME"], charset="utf8mb4")


EVAL_START = pd.Timestamp("2000-01-04")
PERIODS = {"전체(2000~)": (None, None), "개발 2000~2012": ("2000-01-04", "2012-12-31"), "검증 2013~2026": ("2013-01-01", None),
           "2000~2007": ("2000-01-04", "2007-12-31"), "2008~2016": ("2008-01-01", "2016-12-31"), "2017~2026": ("2017-01-01", None)}


def load_ctx():
    c = db()
    mk = pd.read_sql("SELECT `종목코드` code, `시장구분` m FROM kor_ticker WHERE `시장구분` IS NOT NULL AND `기준일`=(SELECT MAX(`기준일`) FROM kor_ticker WHERE `시장구분` IS NOT NULL)", c)
    mkt = dict(zip(mk.code, mk.m))
    cd = pd.read_sql("SELECT `날짜` d, `금리` r FROM cd91_rate", c, parse_dates=["d"]).set_index("d")["r"]
    return mkt, cd


def benchmark(dates):
    import FinanceDataReader as fdr
    k = fdr.DataReader("KS11", "1999-01-01", str(pd.Timestamp(dates[-1]).date()))["Close"]
    k.index = pd.DatetimeIndex(k.index)
    return k.reindex(pd.DatetimeIndex(dates)).ffill()


def metrics(eq, dates, rf, bm, lo=None, hi=None):
    import statsmodels.api as sm
    s = pd.Series(eq, index=pd.DatetimeIndex(dates))
    lo = EVAL_START if lo is None else pd.Timestamp(lo); hi = s.index[-1] if hi is None else pd.Timestamp(hi)
    s = s[(s.index >= lo) & (s.index <= hi)]
    b = bm.reindex(s.index)
    r = s.pct_change().dropna(); rb = b.pct_change().reindex(r.index).fillna(0)
    rfd = (rf.reindex(r.index).ffill().fillna(0) / 100 / 252)
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    cb = (b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    sh = (r - rfd).mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    mdd = (s / s.cummax() - 1).min()
    bmdd = (b / b.cummax() - 1).min()
    X = sm.add_constant((rb - rfd).values); y = (r - rfd).values
    try:
        ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        alpha, beta, ta = ols.params[0] * 252, ols.params[1], ols.tvalues[0]
    except Exception:
        alpha = beta = ta = np.nan
    return dict(CAGR=cagr, vol=vol, sharpe=sh, MDD=mdd, alpha=alpha, alpha_t=ta, beta=beta, bm_CAGR=cb, bm_MDD=bmdd,
                bm_sharpe=((rb - rfd).mean() / rb.std() * np.sqrt(252)))


def tstats(df):
    if len(df) == 0:
        return dict(n=0)
    w = df.ret > 0
    gp = df.ret[w].sum(); gl = -df.ret[~w].sum()
    return dict(n=int(len(df)), win=float(w.mean()), avg_ret=float(df.ret.mean()), avg_R=float(df.rmult.mean()), med_R=float(df.rmult.median()),
                avg_hold=float(df.hold.mean()), PF=float(gp / gl) if gl > 0 else np.nan, t_R=float(df.rmult.mean() / (df.rmult.std() / np.sqrt(len(df)))),
                by_why=df.why.value_counts().to_dict(), by_src=df.src.value_counts().to_dict())


def yearly(eq, dates):
    s = pd.Series(eq, index=pd.DatetimeIndex(dates)); s = s[s.index >= EVAL_START]
    y = s.groupby(s.index.year).last(); first = s.iloc[0]
    prev = pd.concat([pd.Series([first]), y.iloc[:-1]]).values
    return pd.Series(y.values / prev - 1, index=y.index)


def combine(d1, d2):
    """매매법 1·2 합치기: 종목별로 진입일 순서대로, 앞 거래 청산일 이후에 진입하는 것만 유지"""
    d = pd.concat([d1.assign(m=1), d2.assign(m=2)], ignore_index=True).sort_values(["code", "e"])
    keep = []; last = {}
    for i, r in enumerate(d.itertuples(index=False)):
        if r.e > last.get(r.code, -1):
            keep.append(i); last[r.code] = r.fi
    return d.iloc[keep]


def run_variant(dates, tks, mkt, P):
    t1 = core.add_costs(core.all_trades(dates, tks, 1, P), tks, dates, mkt, P)
    t2 = core.add_costs(core.all_trades(dates, tks, 2, P), tks, dates, mkt, P)
    t3 = combine(t1, t2)
    return {"매매법1": t1, "매매법2": t2, "합친것": t3}


def main():
    t0 = time.time()
    say("로드")
    pn = core.load_panel()
    dates, tks = core.prepare(pn)
    del pn
    mkt, cd = load_ctx()
    bm = benchmark(dates)
    say("종목", len(tks), "거래일", len(dates), f"{time.time()-t0:.0f}s")
    P = dict(core.P0)
    out = dict(meta=dict(n_tickers=len(tks), n_days=len(dates), start=str(pd.Timestamp(dates[0]).date()), end=str(pd.Timestamp(dates[-1]).date()), params=P, stats=core.STATS))

    # 1) 기본
    base = run_variant(dates, tks, mkt, P)
    say("기본 거래 수", {k: len(v) for k, v in base.items()}, f"{time.time()-t0:.0f}s")
    eqs = {}; res = {}
    for k, tr in base.items():
        eq, expo, acc = core.portfolio(tr, tks, dates, mkt, P)
        eqs[k] = eq
        res[k] = dict(trade=tstats(tr), periods={p: metrics(eq, dates, cd, bm, lo, hi) for p, (lo, hi) in PERIODS.items()},
                      exposure=float(expo[dates >= np.datetime64(EVAL_START)].mean()), n_accepted=len(acc),
                      turnover=float(2 * sum(a for _, _, a in acc) / eq.mean() / ((pd.Timestamp(dates[-1]) - EVAL_START).days / 365.25)))
        res[k]["yearly"] = {int(y): float(v) for y, v in yearly(eq, dates).items()}
        tr.to_pickle(RES / f"trades_{k}.pkl")
        say(k, "완료", f"CAGR {res[k]['periods']['전체(2000~)']['CAGR']:.3f}", f"{time.time()-t0:.0f}s")
    out["base"] = res
    pd.DataFrame(eqs, index=pd.DatetimeIndex(dates)).assign(KOSPI=bm.values).to_csv(RES / "equity.csv", encoding="utf-8-sig")

    # 2) 소스별(원본 보정 R / kor_price K / 상장폐지 종가-only D) 거래 통계
    out["by_src"] = {k: {s: tstats(v[v.src == s]) for s in v.src.unique()} for k, v in base.items()}

    # 3) 비용 2배
    P2 = dict(P); P2["cost_mult"] = 2.0
    cost2 = {}
    for k, tr in base.items():
        tr2 = core.add_costs(tr[core.COLS], tks, dates, mkt, P2)
        eq, _, _ = core.portfolio(tr2, tks, dates, mkt, P2)
        cost2[k] = dict(trade=tstats(tr2), periods={p: metrics(eq, dates, cd, bm, lo, hi) for p, (lo, hi) in PERIODS.items() if p in ("전체(2000~)", "개발 2000~2012", "검증 2013~2026")})
    out["cost2x"] = cost2
    say("비용 2배 완료", f"{time.time()-t0:.0f}s")

    # 4) 파라미터 민감도 (하나씩 변경)
    var = {"RSI 기준선 45": dict(rsi_gate=45.0), "RSI 기준선 55": dict(rsi_gate=55.0), "과열선 65": dict(rsi_hot=65.0), "과열선 75": dict(rsi_hot=75.0),
           "목표 1.5R": dict(tgt=1.5), "목표 3R": dict(tgt=3.0), "손절 5일저점": dict(stop_lb=5), "손절 20일저점": dict(stop_lb=20),
           "손절폭상한 10%": dict(max_risk=0.10), "유동성 30억": dict(liq_amt=3e9)}
    sens = {}
    for name, ch in var.items():
        Pv = dict(P); Pv.update(ch)
        if "liq_amt" in ch or True:
            pass
        if "liq_amt" in ch:
            d2, t2k = core.prepare(core.load_panel(), Pv)
        else:
            d2, t2k = dates, tks
            # rsi/macd 는 동일, 신호 규칙만 바뀜
        trs = run_variant(d2, t2k, mkt, Pv)
        sens[name] = {}
        for k, tr in trs.items():
            eq, _, _ = core.portfolio(tr, t2k, d2, mkt, Pv)
            m_ = metrics(eq, d2, cd, bm)
            m_o = metrics(eq, d2, cd, bm, "2013-01-01", None)
            sens[name][k] = dict(n=len(tr), avg_R=float(tr.rmult.mean()) if len(tr) else np.nan, CAGR=m_["CAGR"], sharpe=m_["sharpe"], MDD=m_["MDD"], CAGR_oos=m_o["CAGR"], sharpe_oos=m_o["sharpe"])
        say("민감도", name, f"{time.time()-t0:.0f}s")
    out["sensitivity"] = sens

    # 5) 무작위 진입 대조 (같은 종목·같은 해에서 진입일만 무작위, 청산·손절 규칙 동일)
    N = 10
    rnd = {}
    for mode, name in ((1, "매매법1"), (2, "매매법2")):
        sigs = {}
        for code, x in tks.items():
            if mode == 1:
                s = core.m1_signals(x["rsi"], x["macd"], x["sg"], x["liq"], 60, P["rsi_gate"], P["rsi_hot"], P["wait_low"], P["wait_hot"], P["hot_lo"], P["hot_hi"])
            else:
                s = core.m2_signals(x["h"], x["c"], x["rsi"], x["macd"], x["sg"], x["liq"], 60, P["sw"], P["gap_lo"], P["gap_hi"], P["pb"], P["pb_wait"])
            if len(s):
                sigs[code] = s
        yr = {c: pd.DatetimeIndex(dates[tks[c]["gd"]]).year.values for c in sigs}
        runs = []
        for seed in range(N):
            rng = np.random.default_rng(1000 + seed)
            pool = {}
            for code, s in sigs.items():
                x = tks[code]; ys = yr[code]; okd = np.flatnonzero(x["liq"] & (np.arange(x["n"]) >= 60))
                if len(okd) == 0:
                    continue
                by_year = {}
                for t in okd:
                    by_year.setdefault(ys[t], []).append(t)
                pick = []
                for t in s:
                    cand = by_year.get(ys[t])
                    if cand:
                        pick.append(cand[rng.integers(len(cand))])
                if pick:
                    pool[code] = np.array(sorted(pick), dtype=np.int64)
            tr = core.add_costs(core.all_trades(dates, tks, mode, P, rand=pool), tks, dates, mkt, P)
            eq, _, _ = core.portfolio(tr, tks, dates, mkt, P)
            m_ = metrics(eq, dates, cd, bm)
            runs.append(dict(seed=seed, n=len(tr), win=float((tr.ret > 0).mean()), avg_R=float(tr.rmult.mean()), avg_ret=float(tr.ret.mean()),
                             CAGR=m_["CAGR"], sharpe=m_["sharpe"], MDD=m_["MDD"], alpha_t=m_["alpha_t"]))
            say("무작위", name, seed, f"{time.time()-t0:.0f}s")
        rnd[name] = runs
    out["random"] = rnd

    (RES / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    say("전체 완료", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()


