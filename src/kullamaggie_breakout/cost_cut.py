# -*- coding: utf-8 -*-
"""쿨라마기 EMA 돌파 - 비용 줄이기 탐색 (2026-11-13, 기본 검증 이후 추가 실험)

문제: 비용 전 거래당 +0.07R(t 4.6)인데 연 회전율 약 26배라 왕복 0.5% 비용이 우위를 다 먹음.
결과를 보기 전에 정한 후보(회전율을 줄이는 손잡이 6개, 2^6 × 2 = 128개):
  ① 종목 선택  : 모멘텀 상위 5%·횡보 20일 / 상위 2%·횡보 40일
  ② 청산 추적선: EMA10 / EMA20
  ③ 3일째 1/3 매도: 함 / 안 함
  ④ 손절       : 진입일 저가(원칙, 거리 ≤ ADR) / 진입가 − 1.5×ADR 고정(넓은 손절 → 잦은 손절·재진입 줄이기)
  ⑤ 하루 신규 진입: 제한 없음 / 63일 수익률 1위 1종목만
  ⑥ 시장 국면  : 코스피 > EMA20 / 코스피 > EMA20 이고 EMA20 > EMA50
  ⑦ ADR 하한   : 4% / 6% (같은 비용이 R 단위로 더 작아짐)
선택 규칙: 학습(2000~2014) 샤프 1위(실행 거래 100건 이상). 검증(2015~2019)으로 확인.
테스트(2020~)는 기본 검증에서 이미 한 번 봤으므로 '확인용(재사용)'으로만 적는다.
일봉 진입일 처리 = 봉 방향 가정(기본과 같음). 고정 손절은 진입 후 저가만 손절에 쓴다(양봉: 당일 생존, 음봉·갭: 당일 저가로 확인).
"""
import json
import time
from itertools import product

import numpy as np
import pandas as pd
from numba import njit

import backtest as bt
import data
import signals
from run import Lab, SEGS

RES = data.HERE / "results"


@njit(cache=True)
def _path2(O, H, L, C, EM, n, t, fill, stop, T, partial, fixed):
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
        if days == 3 and t_part < 0 and partial:
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
def gen2(O, H, L, C, EM, setup, level, adr, lim, partial, stop_k):
    """stop_k = 0: 진입일 저가 원칙(봉 방향 가정), > 0: 진입가 − stop_k×ADR 고정. 반환 trades 배열(열 4 = 계획 위험 비율)"""
    T, N = C.shape
    out = np.full((400000, 11), np.nan)
    m = 0
    for n in range(N):
        t = 1
        while t < T:
            if not setup[t - 1, n] or np.isnan(C[t, n]) or np.isnan(O[t, n]):
                t += 1
                continue
            lev = level[t - 1, n]
            if not (H[t, n] > lev):
                t += 1
                continue
            fill = max(O[t, n], lev)
            if fill >= C[t - 1, n] * (1 + lim[t]) * 0.995:
                t += 1
                continue
            a = adr[t - 1, n]
            gap = O[t, n] >= lev
            bull = C[t, n] >= O[t, n]
            if stop_k > 0:
                stop = fill * (1 - stop_k * a)
                risk = stop_k * a
                post_low = L[t, n] if (gap or not bull) else (C[t, n] if C[t, n] < fill else fill)
                if post_low <= stop:
                    st, px = 1, stop
                else:
                    st, px = 2, stop
            else:
                risk = a
                capp = fill * (1 - a)
                if gap:
                    if L[t, n] < capp:
                        st, px = 1, capp
                    else:
                        st, px = 2, L[t, n]
                elif bull:
                    if fill - L[t, n] > a * fill:
                        st, px = 0, np.nan
                    else:
                        st, px = 2, L[t, n]
                else:
                    if fill - O[t, n] > a * fill:
                        st, px = 0, np.nan
                    else:
                        st, px = 1, O[t, n]
            if st == 0:
                t += 1
                continue
            if st == 1:
                tp, pp, te, pe, op = -1, np.nan, t, px, 0
            else:
                tp, pp, te, pe, op = _path2(O, H, L, C, EM, n, t, fill, px, T, partial, stop_k > 0)
            out[m, 0] = n
            out[m, 1] = t
            out[m, 2] = fill
            out[m, 3] = px
            out[m, 4] = risk
            out[m, 5] = tp
            out[m, 6] = pp
            out[m, 7] = te
            out[m, 8] = pe
            out[m, 9] = op
            m += 1
            t = te + 1
    return out[:m]


def main():
    t0 = time.time()
    L_ = Lab()
    D, I = L_.D, L_.I
    ks = D["kospi"]
    reg2 = ((ks > ks.ewm(span=20, adjust=False).mean()) & (ks.ewm(span=20, adjust=False).mean() > ks.ewm(span=50, adjust=False).mean())).to_numpy()
    rows, keep = [], {}
    grid = list(product([(0.05, 20), (0.02, 40)], [10, 20], [True, False], [0.0, 1.5], [None, 1], [1, 2], [0.04, 0.06]))
    for sel, ema, part, sk, mx, rg, adr in grid:
        top, n = sel
        s = I.setup(adr_min=adr, top=top, n=n, regime=True)
        if rg == 2:
            s = s & reg2[:, None]
        arr = gen2(D["o"], D["h"], D["l"], D["c"], I.ema[ema], s, I.hh[n], I.adr, D["lim"], part, sk)
        tr = bt.to_frame(arr, L_.dates, D["codes"], I.mom)
        tr = tr[tr.date >= data.START].reset_index(drop=True)
        r, taken, st = bt.portfolio(tr, D["c"], L_.rf, L_.dates, max_new=mx)
        r = r.loc[data.START:]
        tk = tr[taken]
        trg = bt.to_frame(arr, L_.dates, D["codes"], I.mom, cost=0.0)
        trg = trg[trg.date >= data.START].reset_index(drop=True)[taken]
        key = dict(선택=f"상위{top:.0%}·{n}일", EMA=ema, 부분매도=part, 손절="1.5×ADR 고정" if sk else "진입일 저가",
                   하루신규=("1종목" if mx else "제한없음"), 국면=("EMA20>50 추가" if rg == 2 else "EMA20"), ADR=adr)
        for sg, (a, b) in SEGS.items():
            x = r.loc[a:b]
            pf = bt.perf(x, L_.rf)
            al, t, _ = bt.alpha_t(x, L_.kospi, L_.rf)
            tt = tk[(tk.date >= a) & (tk.date <= b)]
            gg = trg[(trg.date >= a) & (trg.date <= b)]
            rows.append(dict(**key, 구간=sg, **pf, 알파t=t, 실행거래=len(tt), 평균R=tt.R.mean() if len(tt) else np.nan,
                             비용전평균R=gg.R.mean() if len(gg) else np.nan, 회전율=st["회전율"], 평균투자비중=st["평균투자비중"]))
        keep[tuple(key.values())] = r
    G = pd.DataFrame(rows)
    G.to_csv(RES / "costcut_grid.csv", index=False, encoding="utf-8-sig")
    keys = list(G.columns[:7])
    tr_ = G[(G.구간 == "학습") & (G.실행거래 >= 100)].sort_values("샤프", ascending=False)
    best = tr_.iloc[0][keys].to_dict()
    sel = G.merge(pd.DataFrame([best]), on=keys)
    sel.to_csv(RES / "costcut_selected.csv", index=False, encoding="utf-8-sig")
    # 손잡이별 효과: 다른 조건 평균으로 본 학습·검증 샤프 차이
    eff = []
    for kcol in keys:
        for sg in ("학습", "검증", "테스트", "전체"):
            g = G[G.구간 == sg].groupby(kcol).샤프.median()
            for v, x in g.items():
                eff.append(dict(손잡이=kcol, 값=str(v), 구간=sg, 샤프중앙값=x))
    pd.DataFrame(eff).to_csv(RES / "costcut_knobs.csv", index=False, encoding="utf-8-sig")
    full = G[G.구간 == "전체"]
    summ = dict(후보수=len(grid), 선택=best, 전체기간_코스피초과CAGR=int((full.CAGR > 0.066).sum()),
                전체기간_샤프양수=int((full.샤프 > 0).sum()), 시간=round(time.time() - t0))
    (RES / "costcut_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, default=str))
    print(sel[["구간", "CAGR", "샤프", "MDD", "알파t", "실행거래", "평균R", "비용전평균R", "회전율"]].round(3).to_string())


if __name__ == "__main__":
    main()
