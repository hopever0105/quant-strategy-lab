# -*- coding: utf-8 -*-
"""20~23 공통: 분기 패널(미국 S&P500 / 코스피 상위 200) 위에서 '그 프로필' 기본 규칙 + 신호 덧씌우기.

기본 규칙(add_rule.py에서 고른 그대로, 다시 고르지 않음):
  12개월 수익률 분기 내 백분위 ≥ 60 · 200일선 위 · 매출 성장 백분위 ≥ 70 · 베타 ≥ 1.0 → 동일가중, 2분기 보유
현금 수익: 미국 3개월 국채(DTB3), 한국 3개월 금리(IR3TIB01KRM156N) — 그 분기 시작 시점 값 / 4
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
QUANT = Path(r"C:\Users\arlo0\OneDrive\바탕 화면\취업\quant")
C = QUANT / "cache" / "druck13f"
CB = HERE / "results" / "combo"
SEG = {"us": {"학습": ("2011-12-31", "2018-12-31"), "검증": ("2019-01-01", "2021-12-31"), "테스트": ("2022-01-01", "2026-12-31")},
       "kr": {"학습": ("2015-03-31", "2019-12-31"), "검증": ("2020-01-01", "2022-12-31"), "테스트": ("2023-01-01", "2026-12-31")}}
COST = {"us": 0.001, "kr": 0.0025}


def load(mk):
    P = pd.read_pickle(CB / f"qpanel_{mk}.pkl")
    R = pd.read_pickle(CB / f"qret_{mk}.pkl")
    M = pd.read_pickle(CB / f"qmkt_{mk}.pkl")
    P = P.copy()
    P["t0"] = pd.to_datetime(P["t0"])
    for c in ["r12m", "sales_g"]:
        P["p_" + c] = P.groupby("t0")[c].rank(pct=True) * 100
    return P, R, M


def base_pick(g, M=60, G=70, B=1.0):
    m = (g["p_r12m"] >= M) & (g["vs200"] > 0) & (g["p_sales_g"] >= G) & (g["beta"] >= B)
    return g.loc[m.fillna(False), "code"].tolist()


def fred(sid, lag_days=5):
    d = pd.read_csv(C / "fred" / f"{sid}.csv")
    d.columns = ["date", "v"]
    d["date"] = pd.to_datetime(d["date"]) + pd.Timedelta(days=lag_days)  # 발표 지연만큼 늦게 안다고 가정
    d["v"] = pd.to_numeric(d["v"], errors="coerce")
    return d.dropna().set_index("date")["v"]


def asof(s, t):
    x = s[:t]
    return x.iloc[-1] if len(x) else np.nan


def cash(mk, Q):
    s = fred("DTB3", 1) if mk == "us" and (C / "fred" / "DTB3.csv").exists() else None
    if mk == "us":
        if s is None:
            s = fred("DGS2", 1)  # 대체
    else:
        s = fred("IR3TIB01KRM156N", 45)
    return pd.Series({q: asof(s, q) / 100 / 4 for q in Q}).fillna(0)


def qsim(picks, H, R, mkt, cost, expo=None, cashr=None):
    """picks {t0:[codes]} → 분기 수익. expo {t0: 0~1} 주식 비중(나머지 현금)."""
    Q = sorted(picks)
    prev_w = pd.Series(dtype=float)
    out, ns = {}, {}
    for i, q in enumerate(Q):
        sleeves = [picks[Q[i - k]] for k in range(H) if i - k >= 0]
        w = pd.Series(dtype=float)
        rets = []
        for names in sleeves:
            if len(names) == 0:
                rets.append(mkt.get(q, 0.0))
                w = w.add(pd.Series(1.0 / len(sleeves), index=["__MKT__"]), fill_value=0)
                continue
            rr = R.loc[q].reindex(names).fillna(0.0) if q in R.index else pd.Series(0.0, index=names)
            rets.append(rr.mean())
            w = w.add(pd.Series(1.0 / len(names) / len(sleeves), index=names), fill_value=0)
        e = 1.0 if expo is None else expo.get(q, 1.0)
        w = w * e
        allk = w.index.union(prev_w.index)
        to = (w.reindex(allk, fill_value=0) - prev_w.reindex(allk, fill_value=0)).abs().sum()
        cr = 0.0 if cashr is None else cashr.get(q, 0.0)
        out[q] = e * np.mean(rets) + (1 - e) * cr - to * cost
        ns[q] = len(picks[q])
        prev_w = w
    return pd.Series(out), pd.Series(ns)


def qmetrics(r, mkt, seg, drop2025=False):
    res = {}
    segs = dict(seg)
    segs["전체"] = (min(a for a, _ in seg.values()), max(b for _, b in seg.values()))
    for s, (a, b) in segs.items():
        x = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))].dropna()
        if drop2025:
            x = x[x.index.year != 2025]
        if len(x) < 4:
            continue
        m = mkt.reindex(x.index)
        nav = (1 + x).cumprod()
        res[s] = dict(CAGR=nav.iloc[-1] ** (4 / len(x)) - 1, 샤프=x.mean() / x.std() * 2,
                      MDD=(nav / nav.cummax() - 1).min(), 시장CAGR=(1 + m).prod() ** (4 / len(x)) - 1,
                      시장샤프=m.mean() / m.std() * 2, 시장MDD=((1 + m).cumprod() / (1 + m).cumprod().cummax() - 1).min(),
                      분기수=len(x))
    return res


def table(rows):
    out = []
    for nm, mk, r, mkt, drop in rows:
        for s, m in qmetrics(r, mkt, SEG[mk], drop).items():
            out.append(dict(시장=mk, 전략=nm + (" (2025 제외)" if drop else ""), 구간=s, **m))
    return pd.DataFrame(out)
