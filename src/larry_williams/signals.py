# -*- coding: utf-8 -*-
"""래리 윌리엄스 검증 - 신호 (모든 신호는 그날 종가까지 알 수 있는 정보만 쓴다. 적용은 backtest 에서 하루 뒤)"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- ① 매크로
def cot_index(net, n):
    """래리 상업자 지수 = (순포지션 − N주 최저)/(N주 최고 − N주 최저)×100 (주간, 보고 기준일 index)"""
    lo, hi = net.rolling(n).min(), net.rolling(n).max()
    return ((net - lo) / (hi - lo) * 100).where(hi > lo)


def cot_daily(net, n, cal):
    """기준일(화) +6일(다음 주 월요일)부터 사용할 수 있게 일간 달력에 맞춘다"""
    idx = cot_index(net, n).dropna()
    idx.index = idx.index + pd.Timedelta(days=6)
    return idx.reindex(idx.index.union(cal)).ffill().reindex(cal)


def cot_state(idx_daily, lo, hi):
    """상업자 지수 ≥ hi → 1(매수), ≤ lo → 0(현금), 그 사이는 직전 상태 유지"""
    s = pd.Series(np.where(idx_daily >= hi, 1.0, np.where(idx_daily <= lo, 0.0, np.nan)), index=idx_daily.index)
    return s.ffill().fillna(0.0)


def above_ma(px, n):
    return (px > px.rolling(n).mean()).astype(float).where(px.rolling(n).mean().notna(), 0.0)


def to_kr(sig_us, kr_cal):
    """미국 날짜 신호 → 한국 달력. 한국 d일 종가 시점에 알 수 있는 미국 자료는 d-1일(미국) 종가까지"""
    s = sig_us.copy()
    s.index = s.index + pd.Timedelta(days=1)            # 미국 d일 종가 → 한국 d+1일 장중에 앎
    return s.reindex(s.index.union(kr_cal)).ffill().reindex(kr_cal)


def oil_signal(oil_kr, lag, m):
    """시차(lag)만큼 앞당긴 유가의 m일 수익률 > 0 → 1"""
    r = oil_kr.shift(lag) / oil_kr.shift(lag + m) - 1
    return (r > 0).astype(float).where(r.notna(), 0.0)


def panic_signal(close, drop, hold, win=20):
    """종가가 최근 win일 최고 대비 drop 이상 떨어진 날 매수 → hold일 보유(보유 중 새 신호는 보유 연장)"""
    dd = close / close.rolling(win).max() - 1
    trig = (dd <= -drop).astype(float)
    pos = trig.rolling(hold, min_periods=1).max()
    return pos, trig


def first_events(trig, hold):
    """보유 기간 안에 겹치지 않는 첫 신호만 (사건 연구용)"""
    out, last = [], -10 ** 9
    t = np.flatnonzero(trig.values > 0)
    for i in t:
        if i - last >= hold:
            out.append(i)
            last = i
    return out


# ---------------------------------------------------------------- ② 단기 시스템
def williams_r(h, l, c, n=10):
    hh, ll = h.rolling(n).max(), l.rolling(n).min()
    return (hh - c) / (hh - ll) * -100


def wr_state(h, l, c, n=10, level=-80, look=5):
    """%R(10)이 −80 아래에서 위로 올라온 날이 최근 look거래일 안에 있으면 True (상태 고정값, 튜닝 안 함)"""
    wr = williams_r(h, l, c, n)
    cross = (wr > level) & (wr.shift(1) <= level)
    return cross.astype(float).rolling(look, min_periods=1).max() > 0


def tdm_window(idx, first_n):
    """월말 마지막 거래일 + 다음 달 첫 first_n 거래일이면 True"""
    s = pd.Series(idx, index=idx)
    ym = idx.year * 100 + idx.month
    pos_start = s.groupby(ym).cumcount() + 1
    pos_end = s.groupby(ym).cumcount(ascending=False) + 1
    return pd.Series((pos_start.values <= first_n) | (pos_end.values == 1), index=idx)


ELECTIONS = {
    "미국 대선": ["2000-11-07", "2004-11-02", "2008-11-04", "2012-11-06", "2016-11-08", "2020-11-03", "2024-11-05"],
    "미국 중간선거": ["2002-11-05", "2006-11-07", "2010-11-02", "2014-11-04", "2018-11-06", "2022-11-08", "2026-11-03"],
    "한국 대선": ["2002-12-19", "2007-12-19", "2012-12-19", "2017-05-09", "2022-03-09", "2025-06-03"],
    "한국 총선": ["2000-04-13", "2004-04-15", "2008-04-09", "2012-04-11", "2016-04-13", "2020-04-15", "2024-04-10"],
}


def election_window(idx, dates, days=21):
    """선거일 전후 ±days 거래일(약 ±1개월)"""
    w = pd.Series(False, index=idx)
    pos = idx.searchsorted(pd.to_datetime(dates))
    for p in pos:
        if p < len(idx):
            w.iloc[max(0, p - days): p + days + 1] = True
    return w
