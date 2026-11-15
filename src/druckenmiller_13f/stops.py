# -*- coding: utf-8 -*-
"""23) 손절 규칙 (미국·코스피, 일별). 실제 매매 없음. KRX 요청 없음.

대상: '그 프로필' 기본 규칙(common_q.base_pick), 분기말 t0 종가 매수, 2분기 보유(겹치는 묶음 2개, 묶음마다 자본 1/2)
손절(숫자는 미리 정한 값, 고르지 않음). 걸리면 그 종목은 그 묶음 끝날 때까지 현금(수익 0):
  L1 매수가 대비 −15% 종가
  L2 50일선 아래 종가
  L3 시간 손절: 1분기 지났을 때 시장보다 못 올랐으면 정리
  L4 L1+L2+L3 아무거나
비용: 사고팔 때 편도 (미국 0.1%, 코스피 0.25%)
결과: results/stops/*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common_q import load, base_pick, SEG, COST  # noqa: E402
from features import load_px  # noqa: E402

OUT = HERE / "results" / "stops"
OUT.mkdir(parents=True, exist_ok=True)
CA = Path(r"C:\Users\arlo0\OneDrive\바탕 화면\취업\quant\cache")


def prices(mk, codes):
    if mk == "us":
        d = {}
        for t in codes:
            x = load_px(t)
            if x is not None and len(x):
                d[t] = x["adj"].astype(float)
        px = pd.DataFrame(d).sort_index()
        bench = load_px("SPY")["adj"].astype(float)
    else:
        o = pd.read_pickle(CA / "ohlc_clean.pkl")
        px = o["c"].astype("float64").where(lambda x: x > 0)[[c for c in codes if c in o["c"].columns]]
        bench = pd.read_csv(CA / "kullamaggie" / "KS11.csv", index_col=0, parse_dates=True)["c"].astype(float)
    px = px[px.index >= "2010-01-01"]
    bench = bench.reindex(px.index).ffill()
    return px, bench


def sleeve(names, t0, t1, px, ma50, bench, rule, cost):
    """한 묶음의 일별 수익(t0 다음날~t1). 종목 동일가중, 손절 시 현금."""
    idx = px.index[(px.index > t0) & (px.index <= t1)]
    if len(idx) == 0:
        return None
    names = [n for n in names if n in px.columns and pd.notna(px.at[t0, n] if t0 in px.index else np.nan)]
    if not names:
        b = bench.loc[idx] / bench[:t0].iloc[-1]
        v = b
        r = v.pct_change()
        r.iloc[0] = v.iloc[0] - 1
        return r
    p0 = px.loc[t0, names]
    P = px.loc[idx, names].ffill().fillna(p0) / p0  # 상대 가격
    M = ma50.loc[idx, names]
    raw = px.loc[idx, names].ffill()
    half = t0 + pd.DateOffset(months=3)
    bh = bench[:half].iloc[-1] / bench[:t0].iloc[-1] - 1 if len(bench[:half]) else np.nan
    val = pd.DataFrame(index=idx, columns=names, dtype=float)
    for n in names:
        s = P[n].values.copy()
        stop_i = None
        for i, d in enumerate(idx):
            if rule in ("L1", "L4") and s[i] <= 0.85:
                stop_i = i
            elif rule in ("L2", "L4") and pd.notna(M[n].iloc[i]) and raw[n].iloc[i] < M[n].iloc[i]:
                stop_i = i
            elif rule in ("L3", "L4") and d >= half and (i == 0 or idx[i - 1] < half) and s[i] - 1 < bh:
                stop_i = i
            if stop_i is not None:
                break
        if stop_i is not None:
            s[stop_i:] = s[stop_i] * (1 - cost)
        val[n] = s
    v = val.mean(axis=1) * (1 - cost)  # 매수 비용
    r = v.pct_change()
    r.iloc[0] = v.iloc[0] - 1
    return r


def run(mk):
    P, R, M = load(mk)
    Q = sorted(P["t0"].unique())
    picks = {q: base_pick(g) for q, g in P.groupby("t0")}
    codes = sorted({c for v in picks.values() for c in v})
    px, bench = prices(mk, codes)
    ma50 = px.rolling(50, min_periods=45).mean()
    last = px.index[-1]
    res = {}
    for rule in ["없음", "L1", "L2", "L3", "L4"]:
        sl = {0: [], 1: []}
        for i, q in enumerate(Q):
            t0 = px.index[px.index <= q][-1]
            t1 = Q[i + 2] if i + 2 < len(Q) else last
            t1 = px.index[px.index <= t1][-1]
            r = sleeve(picks[q], t0, t1, px, ma50, bench, rule, COST[mk])
            if r is not None:
                sl[i % 2].append(r)
        a = pd.concat(sl[0]).sort_index()
        b = pd.concat(sl[1]).sort_index()
        br = bench.pct_change()
        df = pd.concat([a, b], axis=1)
        df = df.apply(lambda c: c.fillna(br.reindex(df.index)))
        res[rule] = df.mean(axis=1)
    res["시장"] = bench.pct_change().reindex(res["없음"].index)
    rows = []
    segs = dict(SEG[mk])
    segs["전체"] = (min(a for a, _ in SEG[mk].values()), max(b for _, b in SEG[mk].values()))
    for k, r in res.items():
        for s, (a, b) in segs.items():
            for drop in ([False, True] if mk == "kr" else [False]):
                x = r[a:b].dropna()
                if drop:
                    x = x[x.index.year != 2025]
                if len(x) < 60:
                    continue
                nav = (1 + x).cumprod()
                rows.append(dict(시장=mk, 손절=k + (" (2025 제외)" if drop else ""), 구간=s,
                                 CAGR=nav.iloc[-1] ** (252 / len(x)) - 1, 변동성=x.std() * np.sqrt(252),
                                 샤프=x.mean() / x.std() * np.sqrt(252), MDD=(nav / nav.cummax() - 1).min()))
    return pd.DataFrame(rows)


def main():
    T = pd.concat([run("us"), run("kr")], ignore_index=True)
    T.to_csv(OUT / "results.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
