"""종목 선택 기준 점검: 동시 보유 20종목 제한에서 같은 날 신호가 자리보다 많을 때 고르는 기준을 바꿔 결과가 달라지는지,
그리고 보유 제한 없이 모든 신호를 같은 비중으로 담았을 때와 비교. 결과: results/select.json
실행: python quant/strategies/macd_rsi_swing/select_check.py"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core, run  # noqa: E402

RES = HERE / "results"
pn = core.load_panel(); dates, tks = core.prepare(pn); del pn
mkt, cd = run.load_ctx(); bm = run.benchmark(dates)
P = dict(core.P0)
ALL = "전체(2000~)"


def met(eq):
    a = run.metrics(eq, dates, cd, bm); o = run.metrics(eq, dates, cd, bm, "2013-01-01", None)
    return dict(CAGR=a["CAGR"], sharpe=a["sharpe"], MDD=a["MDD"], vol=a["vol"], alpha_t=a["alpha_t"], CAGR_oos=o["CAGR"], sharpe_oos=o["sharpe"])


def equal_weight_all(tr):
    """보유 제한·현금 제약 없이 모든 거래를 같은 비중: 매일 열려 있는 포지션의 일간 수익률 단순평균(포지션이 없으면 현금 0%)"""
    nd = len(dates); tk_k, tk_q = core.tax_schedule(dates, False)
    m = P["cost_mult"]; fee = (P["comm"] + P["slip"]) * m
    s = np.zeros(nd); n = np.zeros(nd)
    for r in tr.itertuples(index=False):
        x = tks[r.code]; tx = tk_k if mkt.get(x["code"]) == "KOSPI" else tk_q
        days, cum = core.trade_path(r, x, tx, m, fee)
        v = 1.0 + cum; prev = np.r_[1.0, v[:-1]]
        s[days] += v / prev - 1.0; n[days] += 1
    ret = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    eq = np.cumprod(1 + ret)
    return eq, float((n > 0).mean()), float(n[n > 0].mean())


out = {}
for name in ("매매법1", "매매법2", "합친것"):
    tr = pd.read_pickle(RES / f"trades_{name}.pkl")
    d = {}
    # 같은 날 신호 수 vs 20종목 제한
    per_day = tr.groupby("ed").size()
    d["signal_days"] = int(len(per_day)); d["days_gt20"] = int((per_day > 20).sum()); d["max_per_day"] = int(per_day.max()); d["n_trades"] = int(len(tr))
    eq, ex, _ = core.portfolio(tr, tks, dates, mkt, P)
    d["liq_desc"] = dict(**met(eq), exposure=float(ex.mean()))
    asc = tr.assign(liq=-tr.liq); eq, ex, _ = core.portfolio(asc, tks, dates, mkt, P); d["liq_asc"] = dict(**met(eq), exposure=float(ex.mean()))
    code_order = tr.assign(liq=-pd.factorize(tr.code)[0].astype(float)); eq, ex, _ = core.portfolio(code_order, tks, dates, mkt, P); d["code_order"] = dict(**met(eq), exposure=float(ex.mean()))
    runs = []
    for seed in range(10):
        rr = tr.assign(liq=np.random.default_rng(500 + seed).random(len(tr)))
        eq, ex, _ = core.portfolio(rr, tks, dates, mkt, P); runs.append(dict(**met(eq), exposure=float(ex.mean())))
    d["random10"] = {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs], ddof=1))] for k in runs[0]}
    d["random10_runs"] = runs
    P60 = dict(P); P60["maxpos"] = 60; eq, ex, _ = core.portfolio(tr, tks, dates, mkt, P60); d["maxpos60"] = dict(**met(eq), exposure=float(ex.mean()))
    eq, frac_days, avg_pos = equal_weight_all(tr)
    d["equal_unlimited"] = dict(**met(eq), invested_days=frac_days, avg_positions=avg_pos)
    out[name] = d
    print(name, {k: round(d[k]["CAGR"], 3) for k in ("liq_desc", "liq_asc", "code_order", "maxpos60", "equal_unlimited")}, "random10", [round(x, 3) for x in d["random10"]["CAGR"]])
(RES / "select.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

