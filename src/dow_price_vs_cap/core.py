"""신호(목표 비중)와 백테스트 엔진.

포트폴리오 A(가격가중): 목표 비중 ∝ 실제 주가. 편입 변경일·분할/스핀오프일에만 다시 맞추고, 그 사이에는 지수처럼 저절로 움직인다.
포트폴리오 B(시총가중): 목표 비중 ∝ 주가 × 발행주식수. 분기 말 + 편입 변경일에 다시 맞춘다.
  발행주식수 = 직전 월말 시가총액 / 그날 종가 (당시 공개된 값만 사용)

국내 상장 분배형 ETF 가정
- 배당: 미국 원천징수 15% 포함 투자자 기준 15.4% 과세 후, 분배일(매월 말)에 재투자 (TR형 불가)
- 총보수: 매일 NAV에서 차감, 거래비용: 매매금액 × 편도 비용
- 청산: 매매차익(원화 기준)에 15.4% (국내 상장 해외주식형 ETF의 매매차익은 배당소득 과세)
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import universe as U


@dataclass
class Params:
    fee: float = 0.0035            # 총보수 연 0.35% (TIGER 미국다우존스30 운용보수 기준)
    cost: float = 0.0005           # 편도 거래비용 5bp
    div_tax: float = 0.154         # 분배금 과세
    gain_tax: float = 0.154        # 청산 시 매매차익 과세
    cap_freq: str = 'Q'            # 시총가중 정기 리밸런싱: 'M', 'Q', 'H', 'Y'
    start: str = '2010-01-01'
    end: str = '2026-10-31'
    cap_lag_days: int = 1          # 시가총액 관측치를 며칠 전 것까지 쓰는가(1 = 전일 이전 월말)
    tag: str = ''


def wba_daily(dates, close, wba):
    """WBA 월말 실제값을 일별로 잇는다. 월 안의 경로는 다른 편입 종목 동일가중 평균 움직임을 월말 값에 맞게 조정.
    반환: close(실제가와 같은 단위), div(월말일에 그 달 배당수익률)"""
    mkt = close.pct_change().mean(axis=1).fillna(0)
    m = wba[['raw', 'tr']].copy()
    m.index = [dates[dates.searchsorted(d, side='right') - 1] if d >= dates[0] else pd.NaT for d in m.index]
    m = m[m.index.notna()]
    m = m[~m.index.duplicated(keep='last')]
    px = pd.Series(np.nan, index=dates)
    dv = pd.Series(0.0, index=dates)
    pts = list(m.index)
    for a, b in zip(pts[:-1], pts[1:]):
        seg = mkt.loc[a:b].iloc[1:]
        if len(seg) == 0:
            continue
        target = np.log(m.loc[b, 'raw'] / m.loc[a, 'raw'])
        base = np.log1p(seg).cumsum()
        adj = target - base.iloc[-1]
        path = base + adj * np.arange(1, len(seg) + 1) / len(seg)
        px.loc[seg.index] = m.loc[a, 'raw'] * np.exp(path.values)
        px.loc[a] = m.loc[a, 'raw']
        tr_r = m.loc[b, 'tr'] / m.loc[a, 'tr']
        pr_r = m.loc[b, 'raw'] / m.loc[a, 'raw']
        dv.loc[b] = max(tr_r / pr_r - 1, 0.0)
    return px, dv


def build_inputs(close, raw, div, mcap, wba, dates):
    """종목(편입 기준 이름) × 날짜 행렬: 보정종가, 실제가, 배당수익률, 시가총액"""
    cols = [m for m in U.all_members() if m not in U.EXCLUDED]
    C = pd.DataFrame(index=dates, columns=cols, dtype=float)
    R = C.copy()
    D = pd.DataFrame(0.0, index=dates, columns=cols)
    for m in cols:
        if m in U.MONTHLY_ONLY:
            continue
        t = U.price_ticker(m)
        C[m] = close[t].reindex(dates)
        R[m] = raw[t].reindex(dates)
        D[m] = div[t].reindex(dates).fillna(0)
    wpx, wdv = wba_daily(dates, C.drop(columns=list(U.MONTHLY_ONLY)), wba)
    C['WBA'] = wpx
    R['WBA'] = wpx
    D['WBA'] = wdv
    # 시가총액: 월말 시가총액(관측일 이후에만) × (그날 보정종가 / 관측일 보정종가)
    CAP = pd.DataFrame(index=dates, columns=cols, dtype=float)
    for m in cols:
        s = mcap[m].dropna() if m in mcap else pd.Series(dtype=float)
        if s.empty:
            continue
        # 월말 관측일의 보정종가(휴장일이면 직전 거래일) → 암묵 발행주식수(보정 기준) = 시가총액 / 보정종가
        c_full = C[m].dropna()
        if c_full.empty:
            continue
        c_obs = c_full.reindex(s.index, method='ffill')
        shares = (s / c_obs).dropna()
        sh = shares.reindex(dates, method='ffill')                   # 그날까지 나온 마지막 월말 값만 사용
        CAP[m] = sh.values * C[m].values
    return C, R, D, CAP


def rebalance_dates(dates, kind, freq, splits_by_member):
    """리밸런싱 날짜(그날 종가에 실행). kind='price' or 'cap'"""
    dates = pd.DatetimeIndex(dates)
    rb = set()
    rb.add(dates[0])
    for e, _, _ in U.CHANGES:
        i = dates.searchsorted(pd.Timestamp(e)) - 1              # 발효일 직전 거래일 종가
        if 0 <= i < len(dates):
            rb.add(dates[i])
    if kind == 'price':
        for m, sp in splits_by_member.items():
            for d in sp:
                i = dates.searchsorted(pd.Timestamp(d))              # 분할일 종가(분할 반영 가격)
                if 0 <= i < len(dates):
                    rb.add(dates[i])
    else:
        s = pd.Series(dates, index=dates)
        rule = {'M': 'ME', 'Q': 'QE', 'H': '6ME', 'Y': 'YE'}[freq]
        if freq == 'H':
            last = s.groupby([dates.year, (dates.month - 1) // 6]).last()
        else:
            last = s.resample(rule).last()
        rb |= set(last.dropna().values)
    return sorted(pd.Timestamp(x) for x in rb if x in set(dates))


def target_weights(day, kind, mem_row, R, CAP):
    names = [m for m in mem_row.index[mem_row.values] if m not in U.EXCLUDED]
    if kind == 'price':
        v = R.loc[day, names]
    elif kind == 'cap':
        v = CAP.loc[day, names]
    elif kind == 'equal':
        v = pd.Series(1.0, index=names)
    v = v.dropna()
    return v / v.sum()


@dataclass
class Result:
    nav_usd: pd.Series                      # 투자자 자산(USD): 보수·비용 차감, 분배금 15.4% 과세 후 재투자
    inv_usd: pd.Series                      # nav_usd와 같음(호환용)
    basis_usd: pd.Series                    # 취득원가 누계: 투자원금 + 재투자액(USD)
    weights: pd.DataFrame                   # 리밸런싱 직후 비중
    turnover: float                         # 연 회전율(편도)
    n_trades: int
    trades: pd.Series                       # 날짜별 매매금액 / NAV
    contrib: pd.DataFrame = field(default=None)   # 종목별 일별 기여(수익률 단위)
    div_gross: pd.Series = field(default=None)    # 날짜별 총 배당 / NAV


def simulate(kind, C, R, D, CAP, mem, p: Params, splits_by_member):
    dates = C.loc[p.start:p.end].index
    rbs = set(rebalance_dates(dates, kind, p.cap_freq, splits_by_member))
    ret = C.pct_change()
    pos = pd.Series(0.0, index=C.columns)        # 종목별 보유 금액
    cash_div = 0.0                               # 분배 대기 배당금(펀드 안)
    nav_l, inv_l, bas_l, tr_l, dg_l = [], [], [], [], []
    wlog = {}
    contrib = []
    basis = 1.0                                  # 취득원가(USD): 최초 투자 + 분배금 재투자액
    n_trades = 0
    month = dates.to_series().dt.to_period('M')
    last_of_month = set(dates.to_series().groupby(month).last().values)
    fee_d = p.fee / 252
    prev = None
    for d in dates:
        if prev is None:
            w = target_weights(d, kind, mem.loc[d], R, CAP)
            pos[:] = 0.0
            pos[w.index] = w.values * (1 - p.cost)
            n_trades += len(w)
            nav0 = pos.sum()
            wlog[d] = w
            nav_l.append(nav0); inv_l.append(nav0); bas_l.append(1.0); tr_l.append(1.0); dg_l.append(0.0)
            prev = d
            continue
        r = ret.loc[d].reindex(pos.index).fillna(0.0)
        held = pos[pos != 0]
        nav_prev = pos.sum() + cash_div
        dv = (held * D.loc[d, held.index].fillna(0)).sum()   # 배당락일 배당금(전일 보유 금액 × D/P)
        c = (held * r[held.index]) / nav_prev
        contrib.append(c.rename(d))
        pos = pos * (1 + r)
        cash_div += dv
        nav = pos.sum() + cash_div
        fee = nav * fee_d
        pos = pos * (1 - fee / nav)
        cash_div = cash_div * (1 - fee / nav)
        traded = 0.0
        # 분배(월말): 배당금 지급 → 투자자가 15.4% 세금 내고 같은 펀드를 다시 산다
        paid = 0.0
        if d in last_of_month and cash_div > 0:
            paid = cash_div
            reinv = paid * (1 - p.div_tax)
            # 펀드는 재투자 자금으로 목표 비중대로 매수 (비용 차감). 시총가중은 현재 비중 그대로.
            w_now = target_weights(d, kind, mem.loc[d], R, CAP) if kind != 'cap' else pos[pos > 0] / pos[pos > 0].sum()
            buy = w_now * reinv
            pos[buy.index] += buy.values * (1 - p.cost)
            traded += reinv
            basis += reinv                       # 재투자액은 청산 시 취득원가에 더해진다
            cash_div = 0.0
        if d in rbs:
            w = target_weights(d, kind, mem.loc[d], R, CAP)
            tot = pos.sum()
            tgt = pd.Series(0.0, index=pos.index)
            tgt[w.index] = w.values * tot
            diff = (tgt - pos)
            tv = diff.abs().sum()
            pos = tgt - p.cost * tv * (tgt / tot)
            traded += tv
            n_trades += int((diff.abs() > 1e-9 * tot).sum())
            wlog[d] = w
        nav = pos.sum() + cash_div
        nav_l.append(nav)
        inv_l.append(nav * 1.0)
        bas_l.append(basis)
        tr_l.append(traded / nav)
        dg_l.append(dv / nav_prev)
        prev = d
    nav = pd.Series(nav_l, index=dates)          # 투자자 자산 = 펀드 총액(분배금은 세후 전액 재투자되므로 외부 유입 없음)
    tr = pd.Series(tr_l, index=dates)
    years = (dates[-1] - dates[0]).days / 365.25
    turnover = tr.iloc[1:].sum() / 2 / years
    W = pd.DataFrame(wlog).T.fillna(0)
    return Result(nav_usd=nav, inv_usd=nav, basis_usd=pd.Series(bas_l, index=dates), weights=W, turnover=turnover,
                  n_trades=n_trades, trades=tr, contrib=pd.DataFrame(contrib).fillna(0),
                  div_gross=pd.Series(dg_l, index=dates))


def price_index(C, R, mem, dates):
    """다우 방식 가격지수 복제(배당·보수·비용 없음). 날짜 t의 수익률 = Σ 전일 실제가 × 수익률 / Σ 전일 실제가.
    전일 실제가는 t일 분할을 반영한 기준(= t일 실제가 / (1 + t일 수익률))으로 써서 제수 조정과 같게 만든다.
    편입 종목은 t일 기준(변경은 전일 종가에 반영)."""
    ret = C.pct_change()
    lvl = [1.0]
    for i in range(1, len(dates)):
        d1 = dates[i]
        names = [m for m in mem.columns[mem.loc[d1].values] if m not in U.EXCLUDED]
        r = ret.loc[d1, names]
        p0 = R.loc[d1, names] / (1 + r)
        ok = r.notna() & p0.notna()
        lvl.append(lvl[-1] * (1 + (p0[ok] * r[ok]).sum() / p0[ok].sum()))
    return pd.Series(lvl, index=dates)
