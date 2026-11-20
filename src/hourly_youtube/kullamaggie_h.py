# -*- coding: utf-8 -*-
"""쿨라마기 EMA 돌파 - 진입일을 시봉으로 확인 (일봉 셋업·청산 규칙은 kullamaggie_breakout 그대로)

진입일 처리(시봉):
- 그날 시봉을 시간 순서로 보며 처음 고가가 돌파 기준가를 넘는 봉에서 매수. 체결가 = max(그 봉 시가, 기준가).
- 진입 전 저가 = 앞선 봉들의 저가와 진입 봉 시가 중 최저. 첫 봉 시가에 갭으로 산 경우 진입 전 저가가 없으므로 손절가 = 체결가×(1−ADR).
- 진입 전 저가가 체결가에서 ADR보다 멀면 진입하지 않음(쿨라마기 원칙).
- 진입 후: 진입 봉이 음봉이면 그 봉 저가는 진입 뒤에 나온 것으로 보고, 이후 봉 저가와 함께 손절가 이하면 그날 손절가에 매도.
  (진입 봉 안의 순서만 봉 방향 가정이 남는다)
- 생존하면 다음 날부터 일봉 규칙(진입일 저가 손절, 3일째 1/3, EMA10 이탈) 그대로.
- 그날 시봉이 없으면 일봉 봉 방향 가정으로 처리하고 개수를 센다.
기간: 시봉이 있는 2023-11-14 ~ 2026-10-22에 진입한 거래. 같은 기간 일봉 가정 3가지(봉 방향·보수·진입일 손절 없음)와 비교.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "kullamaggie_breakout"))
import backtest as kbt  # noqa: E402
import data as kdata  # noqa: E402
import signals as ksig  # noqa: E402

BASE = dict(ema=10, adr=0.04, top=0.05, n=20, regime=True)


def hourly_days(B):
    """(code, day) → (o,h,l,c) numpy (시간 순, 15:30 종가 봉 포함)"""
    H = B["H"]
    out = {}
    for (code, d), g in H.groupby(["code", "d"], sort=False):
        out[(code, d)] = (g.o.to_numpy(), g.h.to_numpy(), g.l.to_numpy(), g.c.to_numpy())
    return out


def resolve(bars, lev, a):
    """반환 (status, fill, stop_or_exit): status 0 진입 없음(그날 돌파 못 함/거리 초과), 1 당일 손절, 2 생존"""
    o, h, l, c = bars
    for j in range(len(o)):
        if h[j] > lev:
            fill = max(o[j], lev)
            pre = min(l[:j].min() if j > 0 else np.inf, o[j])
            cap = fill * (1 - a)
            if j == 0 and o[j] >= lev:          # 갭 상승 시가 체결
                stop = cap
            else:
                if fill - pre > a * fill:
                    return 0, fill, np.nan
                stop = min(pre, fill)
                if stop >= fill:
                    stop = cap
            post = list(l[j + 1:])
            if c[j] < o[j]:                     # 음봉: 진입 봉 저가는 진입 뒤
                post.append(l[j])
            if j == 0 and o[j] >= lev:
                post.append(l[j])
            if len(post) and min(post) <= stop:
                return 1, fill, stop
            return 2, fill, stop
    return 0, np.nan, np.nan


def run(B, ema=10, adr=0.04, top=0.05, n=20, regime=True, cost=kbt.COST):
    D = kdata.load()
    I = ksig.Ind(D)
    s = I.setup(adr_min=adr, top=top, n=n, regime=regime)
    O, Hh, L, C, EM = D["o"], D["h"], D["l"], D["c"], I.ema[ema]
    lev_a, adr_a = I.hh[n], I.adr
    dates = pd.DatetimeIndex(D["dates"])
    codes = D["codes"]
    hd = hourly_days(B)
    hcodes = set(B["codes"])
    t0 = dates.searchsorted(pd.Timestamp("2023-11-14"))
    T = len(dates)
    lim = D["lim"]
    rows = []
    n_fallback = 0
    for ni, code in enumerate(codes):
        if code not in hcodes:
            continue
        t = t0
        while t < T:
            if not s[t - 1, ni] or np.isnan(C[t, ni]) or np.isnan(O[t, ni]):
                t += 1
                continue
            lev = lev_a[t - 1, ni]
            if not (Hh[t, ni] > lev):
                t += 1
                continue
            a = adr_a[t - 1, ni]
            bars = hd.get((code, dates[t]))
            if bars is None:
                n_fallback += 1
                fill = max(O[t, ni], lev)
                gap = O[t, ni] >= lev
                st, px = kbt._entry(O, L, C, ni, t, fill, a, gap, 1)
            else:
                st, fill, px = resolve(bars, lev, a)
            if st == 0 or np.isnan(fill) or fill >= C[t - 1, ni] * (1 + lim[t]) * 0.995:
                t += 1
                continue
            if st == 1:
                tp, pp, te, pe, op = -1, np.nan, t, px, 0
            else:
                tp, pp, te, pe, op = kbt._path(O, Hh, L, C, EM, ni, t, fill, px, T)
            rows.append([ni, t, fill, px, a, tp, pp, te, pe, op, 0.0])
            t = te + 1
    arr = np.array(rows, dtype=float) if rows else np.zeros((0, 11))
    tr = kbt.to_frame(arr, dates.values, codes, I.mom, cost=cost)
    return tr, n_fallback


def daily_modes(B, cost=kbt.COST):
    """같은 기간·같은 종목(시봉 있는 종목) 일봉 가정 3가지"""
    D = kdata.load()
    I = ksig.Ind(D)
    s = I.setup(**{k: v for k, v in dict(adr_min=0.04, top=0.05, n=20, regime=True).items()})
    hcodes = set(B["codes"])
    out = {}
    for mode, nm in [(1, "일봉: 봉 방향 가정"), (0, "일봉: 보수 가정"), (2, "일봉: 진입일 손절 없음")]:
        arr = kbt.gen_trades(D["o"], D["h"], D["l"], D["c"], I.e10, s, I.hh[20], I.adr, D["lim"], True, mode)
        tr = kbt.to_frame(arr, D["dates"], D["codes"], I.mom, cost=cost)
        tr = tr[(tr.date >= "2023-11-14") & tr.code.isin(hcodes)]
        out[nm] = tr
    return out
