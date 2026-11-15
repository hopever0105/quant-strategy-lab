# -*- coding: utf-8 -*-
"""16) 분기 패널 만들기 (미국 S&P500 / 코스피200 대용). 실제 매매 없음. KRX 요청 없음.

행 = (분기말 t0, 종목). 특징은 t0까지 아는 값, 결과는 다음 분기 수익률 ret(t0→t1).
공통 특징: r1m r3m r6m r12m, vs60 vs120 vs200(종가/이동평균−1), dist52h, vol60, beta(1년), mcap,
           매출 성장(sales_g), 영업이익률(op_margin), ROE, 이익수익률(ey), 장부가/시총(bm)
미국: S&P500 그 시점 구성종목(panel_ranked + 야후 시세로 60·120일선 추가), 매출 성장 = 최근 4분기(TTM)
코스피200 대용: 매 분기말 코스피 보통주 시가총액 상위 200(과거 코스피200 명단이 없어 대신),
           DART 연간 재무(fy 재무는 fy+1년 4월부터), 상장폐지 포함 종가(ohlc_clean)
결과: results/combo/qpanel_us.pkl, qpanel_kr.pkl, qret_us.pkl(분기×종목 수익률), qret_kr.pkl
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from features import load_px  # noqa: E402
from rule_bt import load_panel  # noqa: E402

QUANT = Path(r"C:\Users\arlo0\OneDrive\바탕 화면\취업\quant")
CA = QUANT / "cache"
OUT = HERE / "results" / "combo"
OUT.mkdir(parents=True, exist_ok=True)


def us():
    S = load_panel()
    S = S[["t0", "t1", "yf", "r1m", "r3m", "r6m", "r12m", "vs200", "dist52h", "vol60", "beta", "mcap",
           "rev_g_ttm", "op_margin", "roe", "ey", "bm", "gross_margin"]].rename(columns={"rev_g_ttm": "sales_g"})
    q = sorted(S["t0"].unique()) + [pd.Timestamp(S["t1"].max())]
    tick = S["yf"].unique()
    v60, v120, rets = {}, {}, {}
    for t in tick:
        d = load_px(t)
        if d is None:
            continue
        a = d["adj"].astype(float)
        m60 = a / a.rolling(60, min_periods=55).mean() - 1
        m120 = a / a.rolling(120, min_periods=110).mean() - 1
        for t0 in q[:-1]:
            s = a[:t0]
            if len(s) == 0 or (t0 - s.index[-1]).days > 7:
                continue
            v60[(t0, t)] = m60[:t0].iloc[-1]
            v120[(t0, t)] = m120[:t0].iloc[-1]
        # 분기 수익률(시세 끝나면 마지막 가격 유지 = 현금화)
        r = {}
        for i in range(len(q) - 1):
            s0 = a[:q[i]]
            if len(s0) == 0 or (q[i] - s0.index[-1]).days > 7:
                continue
            s1 = a[:q[i + 1]]
            r[q[i]] = s1.iloc[-1] / s0.iloc[-1] - 1
        rets[t] = r
    S["vs60"] = [v60.get((a, b)) for a, b in zip(S["t0"], S["yf"])]
    S["vs120"] = [v120.get((a, b)) for a, b in zip(S["t0"], S["yf"])]
    R = pd.DataFrame(rets)  # index t0, columns ticker
    S["ret"] = [R.at[a, b] if (b in R.columns and a in R.index) else np.nan for a, b in zip(S["t0"], S["yf"])]
    spy = load_px("SPY")["adj"]
    mk = {q[i]: spy[:q[i + 1]].iloc[-1] / spy[:q[i]].iloc[-1] - 1 for i in range(len(q) - 1)}
    S = S.rename(columns={"yf": "code"})
    S.to_pickle(OUT / "qpanel_us.pkl")
    R.to_pickle(OUT / "qret_us.pkl")
    pd.Series(mk).to_pickle(OUT / "qmkt_us.pkl")
    print("미국 패널", S.shape, "분기", S["t0"].nunique(), "분기당", round(len(S) / S["t0"].nunique()))


def kr():
    o = pd.read_pickle(CA / "ohlc_clean.pkl")
    c = o["c"].astype("float64").where(lambda x: x > 0)
    amt = o["amt"].astype("float64")
    L = pd.read_csv(CA / "larry" / "listing_info_v4.csv", dtype=str).drop_duplicates("code").set_index("code")
    nm = L["name"].reindex(c.columns).fillna("")
    keep = (c.columns.str[-1] == "0") & ~nm.str.contains("스팩|SPAC|기업인수목적", regex=True).values \
        & (L["market"].reindex(c.columns) == "KOSPI").values
    c = c.loc[:, keep]
    amt = amt[c.columns]
    ks = pd.read_csv(CA / "kullamaggie" / "KS11.csv", index_col=0, parse_dates=True)["c"].reindex(c.index).ffill()
    r = c.pct_change(fill_method=None)
    kr_ = ks.pct_change()
    feats = {
        "r1m": c / c.shift(21) - 1, "r3m": c / c.shift(63) - 1, "r6m": c / c.shift(126) - 1, "r12m": c / c.shift(252) - 1,
        "vs60": c / c.rolling(60, min_periods=55).mean() - 1, "vs120": c / c.rolling(120, min_periods=110).mean() - 1,
        "vs200": c / c.rolling(200, min_periods=180).mean() - 1, "dist52h": c / c.rolling(252, min_periods=200).max() - 1,
        "vol60": r.rolling(60, min_periods=45).std() * np.sqrt(252),
        "beta": r.rolling(252, min_periods=200).cov(kr_).div(kr_.rolling(252, min_periods=200).var(), axis=0),
    }
    fp = pd.read_pickle(CA / "larry" / "factor_panel_v2.pkl")[["code", "date", "mcap"]].dropna()
    mc_a = fp.pivot_table(index="date", columns="code", values="mcap").reindex(columns=c.columns)
    fs = pd.read_csv(CA / "larry" / "dart_fs.csv", dtype={"code": str}).drop_duplicates(["code", "fy"]).set_index(["code", "fy"])
    qe = c.index.to_series().groupby(c.index.to_period("Q")).max()
    qe = qe[qe >= "2015-03-31"].tolist()
    if qe[-1] < c.index[-1]:
        qe.append(c.index[-1])
    fys = set(fs.index.get_level_values(1))
    rows, rets = [], {}
    for i, t0 in enumerate(qe[:-1]):
        t1 = qe[i + 1]
        fy = t0.year - 1 if t0.month >= 4 else t0.year - 2
        ad = mc_a.index[mc_a.index <= t0][-1]
        mcap = mc_a.loc[ad] * (c.loc[t0] / c.loc[:ad].iloc[-1])
        g = pd.DataFrame({k: v.loc[t0] for k, v in feats.items()})
        g["mcap"] = mcap
        g["amt20"] = amt.rolling(20, min_periods=10).mean().loc[t0] if i == 0 else amt.loc[:t0].iloc[-20:].mean()
        g = g[c.loc[t0].notna() & g["mcap"].notna() & (g["amt20"] > 0)]
        g = g.sort_values("mcap", ascending=False).head(200)
        f1 = fs.xs(fy, level="fy") if fy in fys else pd.DataFrame()
        f0 = fs.xs(fy - 1, level="fy") if (fy - 1) in fys else pd.DataFrame()
        if len(f1):
            s1, s0 = f1["sales"].reindex(g.index), f0["sales"].reindex(g.index) if len(f0) else np.nan
            g["sales_g"] = (s1 / s0 - 1).where(s0 > 0) if len(f0) else np.nan
            g["op_margin"] = (f1["op"] / f1["sales"]).reindex(g.index).where(f1["sales"].reindex(g.index) > 0)
            eq = f1["equity"].reindex(g.index)
            g["roe"] = (f1["ni"].reindex(g.index) / eq).where(eq > 0)
            g["ey"] = f1["ni"].reindex(g.index) / g["mcap"]
            g["bm"] = eq / g["mcap"]
        # 다음 분기 수익(상장폐지면 마지막 가격 유지)
        p0 = c.loc[t0, g.index]
        p1 = c.loc[t0:t1, g.index].ffill().iloc[-1]
        g["ret"] = p1 / p0 - 1
        g["t0"], g["t1"] = t0, t1
        g["code"] = g.index
        rows.append(g.reset_index(drop=True))
        # 보유 연장(2분기)용: 이번 분기 상위 200 + 직전 분기 종목들의 이번 분기 수익
        rets[t0] = (c.loc[t0:t1].ffill().iloc[-1] / c.loc[t0] - 1)
    P = pd.concat(rows, ignore_index=True)
    P = P[P["t0"] >= pd.Timestamp("2015-03-31")]
    R = pd.DataFrame(rets).T
    mk = {qe[i]: ks.loc[qe[i + 1]] / ks.loc[qe[i]] - 1 for i in range(len(qe) - 1)}
    P.to_pickle(OUT / "qpanel_kr.pkl")
    R.to_pickle(OUT / "qret_kr.pkl")
    pd.Series(mk).to_pickle(OUT / "qmkt_kr.pkl")
    print("코스피200 대용 패널", P.shape, "분기", P["t0"].nunique(), "매출성장 있음", round(P["sales_g"].notna().mean(), 2))


if __name__ == "__main__":
    us()
    kr()
