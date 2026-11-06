"""비중 계산과 포트폴리오 시뮬레이션, 성과 지표.

비중(자산군별 위험 한도 + 변동성 역수):
  월말 m에 투자 가능한 자산으로 계산 → m+1일 종가부터 적용.
  자산군 g(주식·채권·실물·통화)의 위험 몫 = 1 / (자산이 있는 자산군 수), 그 안에서 자산마다 똑같이 나눔.
  자산 i 원비중 = 위험 몫 / σᵢ(60일) → 전체 합이 1이 되게 나눔 = 리스크 패리티(항상 100% 투자).
  추세 전략은 이 비중에 '보유 여부 × ML 배수'를 곱한다. 들고 있지 않은 몫은 현금(미 3개월 국채).
  합이 1을 넘으면 1로 줄인다(레버리지 없음).
매매 시점: 비중이 바뀌는 날(월초 재조정, 진입·청산일) 종가에 목표 비중으로 맞추고, 바뀐 양 × 비용을 그날 수익에서 뺀다.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from signals import month_end_idx


def base_weights(F, groups_list, start, asset_mask=None):
    vol = F["vol60"].values
    el = F["elig"].values.copy()
    if asset_mask is not None:
        el &= asset_mask[None, :]
    T, A = vol.shape
    g = np.array(groups_list)
    B = np.zeros((T, A))
    me = month_end_idx(F["vol60"].index)
    me = me[me >= start - 25]
    cur = np.zeros(A)
    last = 0
    for m in me:
        if m + 1 >= T:
            break
        B[last:m + 1] = cur
        ok = el[m] & np.isfinite(vol[m]) & (vol[m] > 0)
        w = np.zeros(A)
        act = [x for x in dict.fromkeys(g[ok])]
        for grp in act:
            sel = ok & (g == grp)
            w[sel] = (1.0 / len(act)) / sel.sum() / vol[m, sel]
        cur = w / w.sum() if w.sum() > 0 else w
        last = m + 1
    B[last:] = cur
    return B


def fixed_weights(index, cols, wmap):
    B = np.zeros((len(index), len(cols)))
    for k, v in wmap.items():
        B[:, cols.index(k)] = v
    return B


def mult_matrix(trades, T, A, mult=None):
    M = np.zeros((T, A))
    m = np.ones(len(trades)) if mult is None else np.asarray(mult)
    for (ai, e, x), v in zip(trades[["ai", "e", "x"]].values, m):
        M[e:x, ai] = v
    return M


def simulate(R, cash, B, M, cost, start, end, force=None):
    """R: (T,A) 일간 수익률, cash: (T,) 일간 현금 수익률. force: 목표가 같아도 재조정할 날(bool, 60/40 월 재조정용).
    반환 ndarray (ret, turnover, exposure)"""
    tgt = B * M
    s = tgt.sum(1)
    tgt = np.where(s[:, None] > 1, tgt / np.maximum(s, 1e-12)[:, None], tgt)
    chg = np.ones(len(tgt), bool)
    chg[1:] = np.abs(tgt[1:] - tgt[:-1]).sum(1) > 1e-12
    if force is not None:
        chg |= force
    Rf = np.nan_to_num(R)
    w = np.zeros(R.shape[1])
    out = np.zeros((end - start + 1, 3))
    for k, d in enumerate(range(start, end + 1)):
        rp = 0.0
        if k > 0:
            rp = w @ Rf[d] + (1 - w.sum()) * cash[d]
            w = w * (1 + Rf[d]) / (1 + rp)
        tc = 0.0
        if chg[d] or k == 0:
            tc = np.abs(tgt[d] - w).sum()
            rp -= tc * cost
            w = tgt[d].copy()
        out[k] = (rp, tc, w.sum())
    return out


def metrics(r, cash, turn=None, expo=None):
    r = pd.Series(r); cash = pd.Series(cash, index=r.index)
    yrs = len(r) / 252
    nav = (1 + r).cumprod()
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    ex = r - cash
    sharpe = ex.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    mdd = (nav / nav.cummax() - 1).min()
    out = dict(CAGR=cagr, vol=vol, sharpe=sharpe, MDD=mdd, calmar=cagr / -mdd if mdd < 0 else np.nan)
    if turn is not None:
        out["turnover"] = float(np.sum(turn) / yrs / 2)      # 연간 편도 회전율
    if expo is not None:
        out["exposure"] = float(np.mean(expo))
    return out


def alpha_t(r, rb, cash):
    """전략 초과수익 = α + β × 비교대상 초과수익, 뉴이-웨스트(10일) 표준오차"""
    y = np.asarray(r) - np.asarray(cash); x = sm.add_constant(np.asarray(rb) - np.asarray(cash))
    f = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    return dict(alpha=float(f.params[0] * 252), alpha_t=float(f.tvalues[0]), beta=float(f.params[1]))


def yearly(r, index):
    s = pd.Series(r, index=index)
    return (1 + s).groupby(s.index.year).prod() - 1
