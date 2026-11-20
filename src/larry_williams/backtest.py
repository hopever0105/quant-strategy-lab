# -*- coding: utf-8 -*-
"""래리 윌리엄스 검증 - 백테스트 엔진과 성과 지표

- 시기 신호(매수/현금): 오늘 종가에 정한 비중을 내일 수익률에 적용. 현금은 단기금리를 받는다.
- 단기 시스템: 시가·고가·저가·종가로 체결가를 정하는 거래 단위 시뮬레이션(한 번에 한 포지션).
- 종목 포트폴리오: 리밸런싱 날짜마다 종목을 고르고, 조건(진입 신호)이 나온 날부터 보유, 다음 리밸런싱 종가에 정리.
거래비용: 편도 수수료 0.015% + 슬리피지 0.05% = 0.065% (cost 인자, 2배 민감도는 0.0013)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

COST = 0.00065
ANN = 252


# ---------------------------------------------------------------- 성과 지표
def perf(r, rf=None):
    r = r.dropna()
    if len(r) < 20:
        return dict(CAGR=np.nan, 변동성=np.nan, 샤프=np.nan, MDD=np.nan, 일수=len(r))
    ex = r - (rf.reindex(r.index).fillna(0) if rf is not None else 0)
    eq = (1 + r).cumprod()
    yrs = len(r) / ANN
    vol = r.std() * np.sqrt(ANN)
    return dict(CAGR=eq.iloc[-1] ** (1 / yrs) - 1, 변동성=vol,
                샤프=ex.mean() / ex.std() * np.sqrt(ANN) if ex.std() > 0 else np.nan,
                MDD=(eq / eq.cummax() - 1).min(), 일수=len(r))


def alpha_t(r, b, rf=None, lags=5):
    """초과수익 회귀 r-rf = a + β(b-rf), Newey-West t값. 반환 (연율 알파, t, 베타)"""
    df = pd.concat([r, b], axis=1).dropna()
    if len(df) < 60:
        return np.nan, np.nan, np.nan
    f = rf.reindex(df.index).fillna(0) if rf is not None else 0
    y, x = df.iloc[:, 0] - f, sm.add_constant(df.iloc[:, 1] - f)
    m = sm.OLS(y.values, x.values).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return m.params[0] * ANN, m.tvalues[0], m.params[1]


def mean_t(x, lags):
    """평균의 Newey-West t값 (겹치는 표본용)"""
    x = pd.Series(x).dropna()
    if len(x) < 10:
        return np.nan, np.nan
    m = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": max(1, lags)})
    return m.params[0], m.tvalues[0]


def diff_t(x, mask, lags):
    """mask 구간 평균 − 나머지 평균, Newey-West t값 (더미 회귀)"""
    df = pd.DataFrame({"y": x, "d": mask.astype(float)}).dropna()
    if df.d.sum() < 5 or (1 - df.d).sum() < 5:
        return np.nan, np.nan
    m = sm.OLS(df.y.values, sm.add_constant(df.d.values)).fit(cov_type="HAC", cov_kwds={"maxlags": max(1, lags)})
    return m.params[1], m.tvalues[1]


def yearly(r):
    return (1 + r.dropna()).groupby(r.dropna().index.year).prod() - 1


def segment(r, a, b):
    return r.loc[a:b]


# ---------------------------------------------------------------- 시기 신호 (매수/현금)
def timing(pos, ret, cash=None, cost=COST):
    """pos: t일 종가에 정한 비중(0~1), ret: 자산 일간 수익률. 반환 (전략 수익률, 거래 수, 연 회전율)"""
    pos = pos.reindex(ret.index).fillna(0.0)
    c = cash.reindex(ret.index).fillna(0.0) if cash is not None else 0.0
    p = pos.shift(1).fillna(0.0)
    dp = pos.diff().abs().fillna(pos.abs())
    r = p * ret + (1 - p) * c - dp.shift(1).fillna(0.0) * cost
    entries = int(((pos > 0) & (pos.shift(1).fillna(0) == 0)).sum())
    turn = dp.sum() / (len(ret) / ANN)
    return r, entries, turn


# ---------------------------------------------------------------- ② 단기 매매 시스템
def short_trades(df, allow, trig, k, T, stop_m):
    """df: o,h,l,c (index 날짜). allow: 진입일 t 에 진입 허용(bool, t-1 종가까지 정보 + t일 달력).
    trig: 'oops' 또는 'vb'. 반환 거래 목록 DataFrame"""
    o, h, l, c = (df[x].values for x in ("o", "h", "l", "c"))
    al = allow.reindex(df.index).fillna(False).values
    n = len(df)
    rng = h - l
    out = []
    i = 1
    while i < n:
        if not al[i]:
            i += 1
            continue
        prev_rng = rng[i - 1]
        if prev_rng <= 0:
            i += 1
            continue
        if trig == "oops":
            if not (o[i] < l[i - 1] and h[i] >= l[i - 1]):
                i += 1
                continue
            entry = l[i - 1]
        else:
            lvl = o[i] + k * prev_rng
            if h[i] < lvl:
                i += 1
                continue
            entry = lvl
        stop = entry - stop_m * prev_rng
        j, ex, why = i, None, None
        if c[i] <= stop:                       # 진입일 종가가 손절선 아래면 종가 정리
            ex, why = c[i], "손절"
        while ex is None:
            j += 1
            if j >= n:
                j = n - 1
                ex, why = c[j], "기간끝"
                break
            if o[j] > entry:
                ex, why = o[j], "Bailout"
            elif o[j] <= stop:
                ex, why = o[j], "손절"
            elif l[j] <= stop:
                ex, why = stop, "손절"
            elif j - i >= T:
                ex, why = c[j], "시간"
        out.append((df.index[i], df.index[j], i, j, entry, ex, entry - stop, why))
        i = j + 1
    return pd.DataFrame(out, columns=["진입일", "청산일", "i", "j", "진입가", "청산가", "손절폭", "사유"])


def trades_to_returns(df, tr, risk, cash=None, cost=COST):
    """위험% 기준 비중 f = min(1, risk ÷ (손절폭/진입가)). 보유 외 자금은 현금 금리."""
    c = df["c"].values
    n = len(df)
    cr = cash.reindex(df.index).fillna(0.0).values if cash is not None else np.zeros(n)
    f_arr = np.zeros(n)
    pr = np.zeros(n)
    for _, t in tr.iterrows():
        i, j = int(t.i), int(t.j)
        f = min(1.0, risk / (t.손절폭 / t.진입가))
        if i == j:
            pr[i] += f * (t.청산가 / t.진입가 - 1) - 2 * f * cost
            f_arr[i] = f
            continue
        pr[i] += f * (c[i] / t.진입가 - 1) - f * cost
        f_arr[i] = f
        for d in range(i + 1, j):
            pr[d] += f * (c[d] / c[d - 1] - 1)
            f_arr[d] = f
        pr[j] += f * (t.청산가 / c[j - 1] - 1) - f * cost
        f_arr[j] = f
    r = pr + (1 - f_arr) * cr
    return pd.Series(r, index=df.index), f_arr


# ---------------------------------------------------------------- ③ 종목 포트폴리오
def stock_portfolio(rebal, picks, entry_sig, R, cash, cost=COST):
    """rebal: 리밸런싱 날짜 목록(마지막은 끝 날짜). picks: {날짜: [종목]}. entry_sig: 날짜×종목 bool(그날 종가에 진입 조건)
    또는 None(즉시 진입). R: 일간 수익률 와이드. 반환 (일간 수익률, 연 회전율, 보유 종목 수 평균, 진입률)"""
    out = []
    traded = 0.0
    held, filled = [], []
    for a, b in zip(rebal[:-1], rebal[1:]):
        days = R.index[(R.index > a) & (R.index <= b)]
        if len(days) == 0:
            continue
        cs = cash.reindex(days).fillna(0.0).values
        sel = [x for x in picks.get(a, []) if x in R.columns]
        if not sel:
            out.append(pd.Series(cs, index=days))
            held.append(0)
            continue
        r = R.loc[days, sel].fillna(0.0).values.astype(float)
        if entry_sig is None:
            E = np.ones_like(r, dtype=bool)
        else:
            sig = entry_sig.reindex(index=R.index[(R.index >= a) & (R.index < b)], columns=sel).fillna(False)
            # 신호가 난 날 종가에 매수 → 다음 날부터 수익. 리밸런싱일 당일 신호도 포함
            first = sig.values.cumsum(axis=0) > 0
            E = np.zeros_like(r, dtype=bool)
            m = min(len(first), len(days))
            E[:m] = first[:m]
        v = np.full(len(sel), 1.0 / len(sel))
        vals = []
        prev_in = np.zeros(len(sel), dtype=bool)
        tot_prev = 1.0
        for t in range(len(days)):
            ent = E[t] & ~prev_in
            v = v - ent * v * cost
            traded += (ent * v).sum() / max(tot_prev, 1e-12)
            v = v * np.where(E[t], 1 + r[t], 1 + cs[t])
            prev_in = E[t]
            tot = v.sum()
            vals.append(tot / tot_prev - 1)
            tot_prev = tot
        # 기간 끝 정리 비용
        exitv = (v * prev_in).sum()
        vals[-1] = vals[-1] - exitv / tot_prev * cost
        traded += exitv / tot_prev
        out.append(pd.Series(vals, index=days))
        held.append(len(sel))
        filled.append(E.any(axis=0).mean())
    ret = pd.concat(out).sort_index()
    yrs = len(ret) / ANN
    return ret, traded / 2 / yrs, float(np.mean(held)) if held else 0.0, float(np.mean(filled)) if filled else np.nan
