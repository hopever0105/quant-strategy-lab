# -*- coding: utf-8 -*-
"""성과 지표·회귀·포트폴리오 계산 (중형 가치주 검증)"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

ANN = 252
SEG = {"학습": ("2000-01-01", "2014-12-31"), "검증": ("2015-01-01", "2019-12-31"),
       "테스트": ("2020-01-01", "2026-12-31"), "전체": ("1990-01-01", "2026-12-31")}
FF6 = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]


def perf(r, rf=None, ann=ANN):
    r = r.dropna()
    if len(r) < 20:
        return dict(CAGR=np.nan, 변동성=np.nan, 샤프=np.nan, MDD=np.nan, 기간=len(r))
    ex = r - (rf.reindex(r.index).ffill().fillna(0) if rf is not None else 0)
    eq = (1 + r).cumprod()
    yrs = len(r) / ann
    return dict(CAGR=eq.iloc[-1] ** (1 / yrs) - 1, 변동성=r.std() * np.sqrt(ann),
                샤프=ex.mean() / ex.std() * np.sqrt(ann) if ex.std() > 0 else np.nan,
                MDD=(eq / eq.cummax() - 1).min(), 기간=f"{r.index[0]:%Y-%m}~{r.index[-1]:%Y-%m}")


def seg(r, name):
    a, b = SEG[name]
    return r[(r.index >= a) & (r.index <= b)]


def yearly(r):
    r = r.dropna()
    return (1 + r).groupby(r.index.year).prod() - 1


def ols_nw(y, X, lags=5):
    m = sm.OLS(y.values, sm.add_constant(X).values).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return m


def factor_reg(r, ff, cols=FF6, ann=ANN, lags=5):
    """초과수익을 팩터에 회귀. 반환 dict(알파 연율, 알파 t, 각 베타, 각 t, R2)"""
    d = pd.concat([r.rename("r"), ff], axis=1, join="inner").dropna()
    if len(d) < 60:
        return {}
    m = ols_nw(d["r"] - d["RF"], d[cols], lags)
    out = {"알파(연)": m.params[0] * ann, "알파t": m.tvalues[0], "R2": m.rsquared, "표본": len(d)}
    for i, c in enumerate(cols):
        out[c] = m.params[i + 1]
        out[c + "_t"] = m.tvalues[i + 1]
    return out


def capm_alpha(r, b, rf, ann=ANN, lags=5):
    d = pd.concat([r.rename("r"), b.rename("b")], axis=1).dropna()
    if len(d) < 30:
        return np.nan, np.nan, np.nan
    f = rf.reindex(d.index).ffill().fillna(0)
    m = ols_nw(d["r"] - f, (d["b"] - f).to_frame(), lags)
    return m.params[0] * ann, m.tvalues[0], m.params[1]


def mean_t(x, lags=3):
    x = pd.Series(x).dropna()
    if len(x) < 10:
        return np.nan, np.nan
    m = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return m.params[0], m.tvalues[0]


def diff_t(x, mask, lags=3):
    df = pd.DataFrame({"y": x, "d": mask.astype(float)}).dropna()
    if df.d.sum() < 5 or (1 - df.d).sum() < 5:
        return np.nan, np.nan, int(df.d.sum())
    m = sm.OLS(df.y.values, sm.add_constant(df.d.values)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return m.params[1], m.tvalues[1], int(df.d.sum())


def weight_portfolio(rebal, weights, R, cost):
    """rebal: 리밸런싱 날짜(마지막은 끝 날짜). weights: {날짜: Series(종목→비중, 합 1)}.
    리밸런싱일 종가에 비중을 맞추고 다음 날부터 비중이 가격 따라 움직임(시가총액 가중 ETF와 같은 방식).
    비용 = 편도 cost × 거래한 비중 합(첫 매수·마지막 정리 포함). 상장폐지 후 수익률 결측은 0(현금 보관)으로 둔다.
    반환 (일간 수익률, 연 회전율(편도), 평균 종목 수, 기간별 종목 기여 DataFrame)"""
    out, traded, n_hold, contrib = [], 0.0, [], []
    prev_w = pd.Series(dtype=float)
    for a, b in zip(rebal[:-1], rebal[1:]):
        days = R.index[(R.index > a) & (R.index <= b)]
        w = weights.get(a, pd.Series(dtype=float))
        w = w[w.index.isin(R.columns)]
        w = w / w.sum() if len(w) and w.sum() > 0 else w
        allc = prev_w.index.union(w.index)
        trade = (w.reindex(allc).fillna(0) - prev_w.reindex(allc).fillna(0)).abs().sum()
        traded += trade
        if len(days) == 0 or len(w) == 0:
            out.append(pd.Series(0.0, index=days))
            prev_w = pd.Series(dtype=float)
            continue
        r = R.loc[days, w.index].fillna(0.0)
        growth = (1 + r).cumprod()
        v = growth * w.values                         # 각 종목 가치(시작 1)
        tot = v.sum(axis=1)
        pr = tot / tot.shift(1).fillna(1.0) - 1
        pr.iloc[0] = pr.iloc[0] - trade * cost
        out.append(pr)
        n_hold.append(len(w))
        end_v = v.iloc[-1]
        contrib.append(pd.DataFrame({"start": a, "code": w.index, "w0": w.values,
                                     "ret": growth.iloc[-1].values - 1, "contrib": (end_v - w).values}))
        prev_w = end_v / end_v.sum()
    # 마지막 정리
    traded += prev_w.abs().sum()
    ret = pd.concat(out).sort_index()
    ret.iloc[-1] = ret.iloc[-1] - prev_w.abs().sum() * cost
    yrs = len(ret) / ANN
    return ret, traded / 2 / yrs, float(np.mean(n_hold)) if n_hold else 0.0, pd.concat(contrib, ignore_index=True)


def switch_portfolio(sig, ra, rb, cost):
    """sig: 월말 기준 1이면 다음 달 a, 0이면 b. 일간 수익률 반환(전환일 비용 차감)"""
    s = sig.reindex(ra.index, method="ffill").shift(1)          # 월말 신호 → 다음 거래일부터
    r = ra.where(s == 1, rb).where(s.notna())
    sw = s.diff().abs().fillna(0) > 0
    return (r - sw * cost).dropna(), int(sw.sum())
