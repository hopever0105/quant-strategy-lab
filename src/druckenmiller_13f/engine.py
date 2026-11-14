# -*- coding: utf-8 -*-
"""공통 시뮬레이션 엔진 (아이디어 1~4). 실제 매매 없음.

schedule: [(t0, t1, {종목: 목표비중(상대값)}), ...]  t0 종가에 매수 → t1 종가까지 보유(가격 따라 비중 변동)
pyramid=True ('오른 종목 더 사기'): 직전 기간에 들고 있던 종목 중 기준지수보다 많이 오른 것은
   - 이번 목표에 있으면 비중 ×1.5
   - 목표에서 빠졌으면 1.0으로 한 기간 더 보유
sleeves=H: 같은 목표를 H기간 동안 들고 가는 겹치는 묶음 H개를 평균(보유기간 H기간 효과)
비용: 매매 금액 × 편도 비용
"""
import numpy as np
import pandas as pd


def simulate(schedule, price_fn, bench, cost=0.001, pyramid=False, sleeves=1, fallback="SPY"):
    if sleeves > 1:
        parts = []
        for k in range(sleeves):
            # k번째 묶음: k, k+H, k+2H ... 기간에만 새로 사고 H기간 보유
            sch = []
            for i in range(k, len(schedule), sleeves):
                t0 = schedule[i][0]
                j = min(i + sleeves, len(schedule)) - 1
                t1 = schedule[j][1]
                sch.append((t0, t1, schedule[i][2]))
            # 첫 묶음 시작 전 기간은 기준지수로 채움
            r, _ = simulate(sch, price_fn, bench, cost, pyramid, 1, fallback)
            parts.append(r)
        idx = parts[0].index
        for p in parts[1:]:
            idx = idx.union(p.index)
        br = bench.pct_change()
        df = pd.concat([p.reindex(idx) for p in parts], axis=1)
        df = df.apply(lambda c: c.fillna(br.reindex(idx)))
        return df.mean(axis=1), None
    daily, log = [], []
    prev_w = pd.Series(dtype=float)
    prev_perf = {}
    for t0, t1, tgt in schedule:
        t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
        w = pd.Series(tgt, dtype=float)
        w = w[w > 0]
        if pyramid and prev_perf:
            for t, beat in prev_perf.items():
                if not beat:
                    continue
                if t in w.index:
                    w[t] = w[t] * 1.5
                else:
                    w[t] = 1.0
        if len(w) == 0:
            w = pd.Series({fallback: 1.0})
        w = w / w.sum()
        idx = bench[(bench.index > t0) & (bench.index <= t1)].index
        if len(idx) == 0:
            continue
        val = pd.DataFrame(index=idx)
        sub = 0.0
        for t, wt in w.items():
            s = price_fn(t)
            ok = s is not None and len(s[:t0]) > 0 and (t0 - s[:t0].index[-1]).days <= 7
            if not ok:
                s = price_fn(fallback)
                sub = sub + wt
            b0 = s[:t0].iloc[-1]
            val[t] = s[(s.index > t0) & (s.index <= t1)].reindex(idx).ffill().fillna(b0) / b0 * wt
        v = val.sum(axis=1)
        r = v.pct_change()
        r.iloc[0] = v.iloc[0] - 1
        allk = w.index.union(prev_w.index)
        traded = (w.reindex(allk, fill_value=0) - prev_w.reindex(allk, fill_value=0)).abs().sum()
        r.iloc[0] = r.iloc[0] - traded * cost
        daily.append(r)
        ev = val.iloc[-1]
        prev_w = ev / ev.sum()
        b_ret = bench[:t1].iloc[-1] / bench[:t0].iloc[-1] - 1
        prev_perf = {t: (ev[t] / w[t] - 1) > b_ret for t in w.index if t != fallback}
        log.append(dict(t0=t0, t1=t1, n=len(w), turnover=traded / 2, sub=sub, ret=v.iloc[-1] - 1))
    return pd.concat(daily).sort_index(), pd.DataFrame(log)


def perf(r, rf, a, b):
    r = r[a:b].dropna()
    if len(r) < 20:
        return {}
    rfx = rf.reindex(r.index).fillna(0)
    yrs = len(r) / 252
    nav = (1 + r).cumprod()
    return dict(CAGR=nav.iloc[-1] ** (1 / yrs) - 1, 변동성=r.std() * np.sqrt(252),
                샤프=(r - rfx).mean() / r.std() * np.sqrt(252), MDD=(nav / nav.cummax() - 1).min())
