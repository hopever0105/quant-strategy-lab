# -*- coding: utf-8 -*-
"""쿨라마기 회전율 줄인 버전(cost_cut.py 학습 1위) 최종 판정. 기준은 실행 전에 정함(2026-11-13):
  ① Deflated Sharpe(후보 128개 보정, 전체 기간 일간 현금 초과수익) 확률 ≥ 0.95
  ② 3년 블록 9개 중 7개 이상에서 연수익률 > 현금 금리
  ③ 같은 날 같은 후보(모멘텀 상위 2%·ADR·국면, 돌파·정배열·횡보 조건 없음) 중 무작위 종목 진입 20회보다 거래당 평균 R이 높음(백분위 ≥ 95%)
  ④ 거래비용 2배(편도 0.5%)에서 전체 기간 연수익률 > 현금 금리
  ⑤ 시봉으로 진입일 확인(2023-11-14~2026-10-22, 실행 거래) 거래당 평균 R > 0
다섯 개 모두 통과 = 통과. 하나라도 실패 = 탈락.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import backtest as bt
import data
from cost_cut import gen2
from run import Lab

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SEED = 20261113
BEST = dict(top=0.02, n=40, ema=10, partial=True, stop_k=0.0, max_new=1, adr=0.04)


def setup(L, top, n, adr, reg2):
    s = L.I.setup(adr_min=adr, top=top, n=n, regime=True)
    return s & reg2[:, None]


def run_cfg(L, s, cost=bt.COST):
    D, I = L.D, L.I
    arr = gen2(D["o"], D["h"], D["l"], D["c"], I.ema[BEST["ema"]], s, I.hh[BEST["n"]], I.adr, D["lim"], BEST["partial"], BEST["stop_k"])
    tr = bt.to_frame(arr, L.dates, D["codes"], I.mom, cost=cost)
    tr = tr[tr.date >= data.START].reset_index(drop=True)
    r, taken, st = bt.portfolio(tr, D["c"], L.rf, L.dates, cost=cost, max_new=BEST["max_new"])
    return tr, r.loc[data.START:], taken, st


def deflated_sharpe(r_ex, sr_trials_ann, n_trials):
    """Bailey & López de Prado (2014). r_ex: 일간 초과수익, sr_trials_ann: 후보들의 연율 샤프"""
    x = r_ex.dropna()
    T = len(x)
    sr = x.mean() / x.std()
    v = np.var(np.asarray(sr_trials_ann) / np.sqrt(252), ddof=1)
    g = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - g) * stats.norm.ppf(1 - 1 / n_trials) + g * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    sk, ku = stats.skew(x), stats.kurtosis(x, fisher=False)
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf(z)), float(sr * np.sqrt(252)), float(sr0 * np.sqrt(252))


def main():
    L = Lab()
    D, I = L.D, L.I
    ks = D["kospi"]
    e20, e50 = ks.ewm(span=20, adjust=False).mean(), ks.ewm(span=50, adjust=False).mean()
    reg2 = ((ks > e20) & (e20 > e50)).to_numpy()
    s = setup(L, BEST["top"], BEST["n"], BEST["adr"], reg2)
    tr, r, taken, st = run_cfg(L, s)
    rf = L.rf.loc[data.START:]
    out = {}

    # ① Deflated Sharpe
    G = pd.read_csv(RES / "costcut_grid.csv")
    sr_trials = G[G.구간 == "전체"].샤프.values
    dsr, sr_ann, sr0_ann = deflated_sharpe(r - rf.reindex(r.index).fillna(0), sr_trials, len(sr_trials))
    out["①DSR"] = dict(확률=dsr, 샤프=sr_ann, 운으로_기대되는_최고샤프=sr0_ann, 통과=dsr >= 0.95)

    # ② 3년 블록
    blocks = []
    for y0 in range(2000, 2027, 3):
        a, b = f"{y0}-01-01", f"{min(y0 + 2, 2026)}-12-31"
        x, c = r.loc[a:b], rf.loc[a:b]
        if len(x) < 100:
            continue
        ann = (1 + x).prod() ** (252 / len(x)) - 1
        cash = (1 + c).prod() ** (252 / len(c)) - 1
        kos = (1 + L.kospi.loc[a:b]).prod() ** (252 / len(x)) - 1
        blocks.append(dict(구간=f"{y0}~{min(y0 + 2, 2026)}", 연수익률=ann, 현금=cash, 코스피=kos, 현금초과=ann - cash))
        B = pd.DataFrame(blocks)
    nb = int((B.현금초과 > 0).sum())
    out["②블록"] = dict(현금초과_블록수=nb, 블록수=len(B), 통과=nb >= 7)
    B.to_csv(RES / "final_blocks.csv", index=False, encoding="utf-8-sig")

    # ③ 같은 날 같은 후보 무작위
    tk = tr[taken].reset_index(drop=True)
    pool = I.setup(adr_min=BEST["adr"], top=BEST["top"], n=BEST["n"], regime=True, use_align=False, use_tight=False) & reg2[:, None]
    rng = np.random.default_rng(SEED)
    ns, ts, dr = [], [], []
    for row in tk.itertuples():
        cand = np.where(pool[row.t - 1])[0]
        if not len(cand):
            continue
        pick = rng.choice(cand, size=20, replace=True)
        ns += pick.tolist(); ts += [row.t] * 20; dr += list(range(20))
    arr = bt.random_paths(D["o"], D["h"], D["l"], D["c"], I.e10, I.adr, np.array(ns, dtype=np.int64), np.array(ts, dtype=np.int64), D["lim"], 1)
    ok = ~np.isnan(arr[:, 2])
    rt = bt.to_frame(np.column_stack([arr, np.zeros(len(arr))]), L.dates, D["codes"])
    rt["draw"] = np.array(dr)[ok]
    dm = rt.groupby("draw").R.mean()
    pct = float((dm < tk.R.mean()).mean())
    out["③무작위"] = dict(실제평균R=float(tk.R.mean()), 무작위평균R=float(dm.mean()), 무작위_95pct=float(dm.quantile(0.95)), 백분위=pct, 통과=pct >= 0.95)

    # ④ 비용 2배
    tr2, r2, tk2, _ = run_cfg(L, s, cost=bt.COST * 2)
    ann2 = (1 + r2).prod() ** (252 / len(r2)) - 1
    cash = (1 + rf).prod() ** (252 / len(rf)) - 1
    out["④비용2배"] = dict(연수익률=float(ann2), 현금=float(cash), 통과=ann2 > cash)

    # ⑤ 시봉 진입일 확인
    sys.path.insert(0, str(HERE.parent / "hourly_youtube"))
    import data_h
    import kullamaggie_h as kh
    Bh = data_h.build()
    hd = kh.hourly_days(Bh)
    hcodes = set(Bh["codes"])
    dates = pd.DatetimeIndex(L.dates)
    t0 = dates.searchsorted(pd.Timestamp("2023-11-14"))
    O, Hh, Lo, C, EM = D["o"], D["h"], D["l"], D["c"], I.e10
    rows, nfb = [], 0
    T = len(dates)
    for ni, code in enumerate(D["codes"]):
        if code not in hcodes:
            continue
        t = t0
        while t < T:
            if not s[t - 1, ni] or np.isnan(C[t, ni]) or np.isnan(O[t, ni]):
                t += 1; continue
            lev = I.hh[BEST["n"]][t - 1, ni]
            if not (Hh[t, ni] > lev):
                t += 1; continue
            a = I.adr[t - 1, ni]
            bars = hd.get((code, dates[t]))
            if bars is None:
                nfb += 1
                stt, px = bt._entry(O, Lo, C, ni, t, max(O[t, ni], lev), a, O[t, ni] >= lev, 1)
                fill = max(O[t, ni], lev)
            else:
                stt, fill, px = kh.resolve(bars, lev, a)
            if stt == 0 or np.isnan(fill) or fill >= C[t - 1, ni] * (1 + D["lim"][t]) * 0.995:
                t += 1; continue
            if stt == 1:
                tp, pp, te, pe, op = -1, np.nan, t, px, 0
            else:
                tp, pp, te, pe, op = bt._path(O, Hh, Lo, C, EM, ni, t, fill, px, T)
            rows.append([ni, t, fill, px, a, tp, pp, te, pe, op, 0.0])
            t = te + 1
    th = bt.to_frame(np.array(rows, dtype=float), L.dates, D["codes"], I.mom)
    rh, tkh, _ = bt.portfolio(th, D["c"], L.rf, L.dates, max_new=BEST["max_new"])
    thk = th[tkh]
    daily_same = tk[(tk.date >= "2023-11-14") & tk.code.isin(hcodes)]
    out["⑤시봉"] = dict(시봉_실행거래=int(len(thk)), 시봉_평균R=float(thk.R.mean()), 일봉가정_같은기간_평균R=float(daily_same.R.mean()),
                      일봉가정_거래=int(len(daily_same)), 시봉없음=nfb, 통과=bool(thk.R.mean() > 0))

    out["판정"] = "통과" if all(v["통과"] for k, v in out.items() if isinstance(v, dict)) else "탈락"
    (RES / "final_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print(B.round(3).to_string())


if __name__ == "__main__":
    main()
