# -*- coding: utf-8 -*-
"""쿨라마기 EMA 돌파 - 거래 생성(numba)과 포트폴리오 백테스트, 성과 지표

체결 규칙 (일봉 근사)
- 진입: t-1일 셋업 → t일 고가 > 돌파 기준가이면 매수. 체결가 = max(t일 시가, 기준가).
        체결가가 상한가(전일 종가 × (1+가격제한폭))의 99.5% 이상이면 살 수 없다고 보고 건너뜀.
- 손절: 진입 시점까지의 당일 저가(쿨라마기 원칙). 그 거리가 ADR보다 멀면 진입하지 않는다. 진입일이 지나면 손절가 = 진입일 저가.
  일봉에는 장중 순서가 없으므로 진입일 처리 가정을 셋 둔다(mode).
  · 1 봉 방향(기본): 양봉(종가 ≥ 시가)은 시가→저가→고가→종가, 음봉은 시가→고가→저가→종가로 움직였다고 본다.
      양봉: 진입 전 저가 = 당일 저가 → 거리 > ADR이면 진입 안 함, 아니면 손절가 = 당일 저가(진입일 생존).
      음봉: 진입 전 저가 = 시가 → 거리 > ADR이면 진입 안 함, 아니면 그날 시가까지 밀려 손절(시가에 매도).
      갭 상승으로 시가에 산 경우: 진입 전 저가가 없으므로 손절가 = 체결가 × (1 − ADR), 당일 저가가 그 아래면 손절.
  · 0 보수: 진입일 저가가 체결가 × (1 − ADR) 아래면 무조건 진입일에 그 값으로 손절.
  · 2 낙관: 진입일 손절 없음, 손절가 = 진입일 저가.
- 3거래일째(진입일 = 1일째) 종가에 1/3 매도, 남은 물량 손절가를 진입가로 올림.
- 3거래일째부터 종가 < EMA(10 또는 20)이면 다음 날 시가에 전량 매도.
- 다음 날부터 시가 ≤ 손절가면 시가, 저가 ≤ 손절가면 손절가에 매도.
- 상장폐지로 자료가 끝나면 마지막 종가(정리매매 포함)에 매도.
- 같은 종목은 앞 거래가 끝난 다음 날부터 다시 진입 가능.
R = 비용 차감 거래 수익률 ÷ 진입 시 계획 위험(ADR). 포지션 크기도 같은 ADR로 정한다(자본 × 0.5% ÷ ADR, 최대 20%).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numba import njit

COST = 0.0025          # 편도 (수수료 + 거래세 + 슬리피지)
RISK = 0.005           # 거래당 위험 (자본 대비)
MAXW = 0.20            # 종목당 최대 비중
ANN = 252


@njit(cache=True)
def _entry(O, L, C, n, t, fill, a, gap, mode):
    """진입일 처리. 반환 (status, stop_or_exit_px): status 0 진입 안 함, 1 진입일 손절(가격), 2 생존(이후 손절가)"""
    capp = fill * (1 - a)
    if mode == 2:
        return 2, L[t, n]
    if gap or mode == 0:
        if L[t, n] < capp:
            return 1, capp
        return 2, L[t, n]
    if C[t, n] >= O[t, n]:
        if fill - L[t, n] > a * fill:
            return 0, np.nan
        return 2, L[t, n]
    if fill - O[t, n] > a * fill:
        return 0, np.nan
    return 1, O[t, n]


@njit(cache=True)
def _path(O, H, L, C, EM, n, t, fill, stop, T):
    """진입일 생존 후 경로. 반환 (t_part, p_part, t_exit, p_exit, still_open)"""
    cur = stop
    days = 1
    t_part = -1
    p_part = np.nan
    pend = False
    last = t
    k = t + 1
    while k < T:
        if np.isnan(C[k, n]):
            k += 1
            continue
        days += 1
        if pend:
            return t_part, p_part, k, O[k, n], 0
        if O[k, n] <= cur:
            return t_part, p_part, k, O[k, n], 0
        if L[k, n] <= cur:
            return t_part, p_part, k, cur, 0
        if days == 3 and t_part < 0:
            t_part = k
            p_part = C[k, n]
            if fill > cur:
                cur = fill
        if days >= 3 and C[k, n] < EM[k, n]:
            pend = True
        last = k
        k += 1
    return t_part, p_part, last, C[last, n], 1 if last == T - 1 else 0


@njit(cache=True)
def gen_trades(O, H, L, C, EM, setup, level, adr, lim, use_trigger, mode=1):
    T, N = C.shape
    cap = 400000
    out = np.full((cap, 11), np.nan)
    m = 0
    for n in range(N):
        t = 1
        while t < T:
            if not setup[t - 1, n] or np.isnan(C[t, n]) or np.isnan(O[t, n]):
                t += 1
                continue
            lev = level[t - 1, n]
            if use_trigger:
                if not (H[t, n] > lev):
                    t += 1
                    continue
                fill = max(O[t, n], lev)
            else:
                fill = O[t, n]
            if fill >= C[t - 1, n] * (1 + lim[t]) * 0.995:
                t += 1
                continue
            a = adr[t - 1, n]
            gap = (not use_trigger) or O[t, n] >= lev
            st, px = _entry(O, L, C, n, t, fill, a, gap, mode)
            if st == 0:
                t += 1
                continue
            if st == 1:
                tp, pp, te, pe, op = -1, np.nan, t, px, 0
                stop = px
            else:
                stop = px
                tp, pp, te, pe, op = _path(O, H, L, C, EM, n, t, fill, stop, T)
            if m >= cap:
                return out
            out[m, 0] = n
            out[m, 1] = t
            out[m, 2] = fill
            out[m, 3] = stop
            out[m, 4] = a
            out[m, 5] = tp
            out[m, 6] = pp
            out[m, 7] = te
            out[m, 8] = pe
            out[m, 9] = op
            m += 1
            t = te + 1
    return out[:m]


@njit(cache=True)
def random_paths(O, H, L, C, EM, adr, ns, ts, lim, mode=1):
    """주어진 (종목, 진입일)에 시가로 진입해 같은 청산 규칙 적용. 반환 trades 배열(gen_trades 형식)"""
    T = C.shape[0]
    M = len(ns)
    out = np.full((M, 11), np.nan)
    for i in range(M):
        n, t = ns[i], ts[i]
        fill = O[t, n]
        if np.isnan(fill) or np.isnan(adr[t - 1, n]) or np.isnan(C[t - 1, n]):
            continue
        if fill >= C[t - 1, n] * (1 + lim[t]) * 0.995:
            continue
        a = adr[t - 1, n]
        st, px = _entry(O, L, C, n, t, fill, a, True, mode)
        if st == 1:
            tp, pp, te, pe, op = -1, np.nan, t, px, 0
            stop = px
        else:
            stop = px
            tp, pp, te, pe, op = _path(O, H, L, C, EM, n, t, fill, stop, T)
        out[i, 0] = n
        out[i, 1] = t
        out[i, 2] = fill
        out[i, 3] = stop
        out[i, 4] = a
        out[i, 5] = tp
        out[i, 6] = pp
        out[i, 7] = te
        out[i, 8] = pe
        out[i, 9] = op
    return out


def to_frame(arr, dates, codes, prio=None, cost=COST):
    df = pd.DataFrame(arr[:, :10], columns=["n", "t", "fill", "stop", "adr", "t_part", "p_part", "t_exit", "p_exit", "open"])
    df = df.dropna(subset=["fill"])
    for c in ["n", "t", "t_part", "t_exit", "open"]:
        df[c] = df[c].fillna(-1).astype(int)
    part = df.t_part >= 0
    gross = np.where(part, (df.p_part / 3 + df.p_exit * 2 / 3) / df.fill, df.p_exit / df.fill)
    df["ret"] = gross * (1 - cost) / (1 + cost) - 1
    df["R"] = df.ret / df.adr
    df["date"] = dates[df.t.values]
    df["exit_date"] = dates[df.t_exit.values]
    df["code"] = codes[df.n.values]
    df["hold"] = df.t_exit - df.t + 1
    if prio is not None:
        df["prio"] = prio[df.t.values - 1, df.n.values]
    return df.reset_index(drop=True)


def portfolio(tr, C, rf, dates, cost=COST, risk=RISK, maxw=MAXW, min_w=0.01, max_new=None):
    """거래 목록 → 일간 수익률. 같은 날 후보는 prio(63일 수익률) 높은 순. 반환 (ret Series, taken bool array, stats dict)"""
    tr = tr.reset_index(drop=True)
    T = len(dates)
    rfv = np.nan_to_num(np.asarray(rf, dtype=float))
    by_entry = {}
    order = tr.sort_values(["t", "prio"], ascending=[True, False]).index if "prio" in tr else tr.sort_values("t").index
    for i in order:
        by_entry.setdefault(int(tr.at[i, "t"]), []).append(i)
    tv = tr[["n", "t", "fill", "adr", "t_part", "p_part", "t_exit", "p_exit"]].to_numpy()
    taken = np.zeros(len(tr), dtype=bool)
    cash, E_prev = 1.0, 1.0
    pos = {}                                   # idx → [shares, lastpx]
    eq = np.empty(T)
    expo = np.empty(T)
    traded = 0.0
    for d in range(T):
        cash *= 1 + rfv[d]
        # 시가·장중 청산(진입일 이후)
        for i in [i for i in pos if tv[i, 6] == d and tv[i, 1] != d]:
            sh = pos[i][0]
            cash += sh * tv[i, 7] * (1 - cost)
            traded += sh * tv[i, 7]
            del pos[i]
        for i in (by_entry.get(d, []) if max_new is None else by_entry.get(d, [])[:max_new]):
            amt = min(risk * E_prev / tv[i, 3], maxw * E_prev, cash / (1 + cost))
            if amt < min_w * E_prev:
                continue
            sh = amt / tv[i, 2]
            cash -= amt * (1 + cost)
            traded += amt
            pos[i] = [sh, tv[i, 2]]
            taken[i] = True
        # 3일째 종가 1/3 매도(t_part ≥ 0일 때만), 진입일 손절
        for i in list(pos):
            if tv[i, 4] == d and tv[i, 4] >= 0:
                sh = pos[i][0] / 3
                cash += sh * tv[i, 5] * (1 - cost)
                traded += sh * tv[i, 5]
                pos[i][0] -= sh
            if tv[i, 6] == d and tv[i, 1] == d:
                sh = pos[i][0]
                cash += sh * tv[i, 7] * (1 - cost)
                traded += sh * tv[i, 7]
                del pos[i]
        val = 0.0
        for i, p in pos.items():
            px = C[d, int(tv[i, 0])]
            if not np.isnan(px):
                p[1] = px
            val += p[0] * p[1]
        E = cash + val
        eq[d] = E
        expo[d] = val / E if E > 0 else 0
        E_prev = E
    eq = pd.Series(eq, index=dates)
    r = eq.pct_change().fillna(0.0)
    yrs = T / ANN
    st = dict(회전율=traded / 2 / eq.mean() / yrs, 평균투자비중=float(expo.mean()), 실행거래=int(taken.sum()), 후보거래=len(tr))
    return r, taken, st


# ---------------------------------------------------------------- 성과 지표
def perf(r, rf=None):
    r = r.dropna()
    if len(r) < 20:
        return dict(CAGR=np.nan, 변동성=np.nan, 샤프=np.nan, MDD=np.nan)
    ex = r - (rf.reindex(r.index).fillna(0) if rf is not None else 0)
    eq = (1 + r).cumprod()
    yrs = len(r) / ANN
    return dict(CAGR=eq.iloc[-1] ** (1 / yrs) - 1, 변동성=r.std() * np.sqrt(ANN),
                샤프=ex.mean() / ex.std() * np.sqrt(ANN) if ex.std() > 0 else np.nan,
                MDD=(eq / eq.cummax() - 1).min())


def alpha_t(r, b, rf=None, lags=5):
    df = pd.concat([r, b], axis=1).dropna()
    if len(df) < 60:
        return np.nan, np.nan, np.nan
    f = rf.reindex(df.index).fillna(0) if rf is not None else 0
    y, x = df.iloc[:, 0] - f, sm.add_constant(df.iloc[:, 1] - f)
    m = sm.OLS(y.values, x.values).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return m.params[0] * ANN, m.tvalues[0], m.params[1]


def mean_t(x):
    x = pd.Series(x).dropna()
    if len(x) < 5:
        return np.nan, np.nan
    return x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def yearly(r):
    r = r.dropna()
    return (1 + r).groupby(r.index.year).prod() - 1


def trade_stats(tr):
    if not len(tr):
        return dict(거래수=0)
    m, t = mean_t(tr.R)
    s = tr.R.sort_values(ascending=False)
    k = max(1, int(len(s) * 0.05))
    win = tr.ret > 0
    return dict(거래수=len(tr), 승률=win.mean(), 평균수익=tr.ret.mean(), 평균R=m, R_t=t,
                평균이익=tr.ret[win].mean() if win.any() else np.nan, 평균손실=tr.ret[~win].mean() if (~win).any() else np.nan,
                평균보유일=tr.hold.mean(), 상위5pct제외평균R=s.iloc[k:].mean(),
                상위5pct이익비중=s.iloc[:k].clip(lower=0).sum() / s.clip(lower=0).sum() if s.clip(lower=0).sum() > 0 else np.nan)
