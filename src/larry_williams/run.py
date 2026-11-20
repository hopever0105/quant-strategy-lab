# -*- coding: utf-8 -*-
"""래리 윌리엄스 검증 - 실행 (전략_명세.md 그대로)

구간: 학습 2000~2014 / 검증 2015~2019 / 테스트 2020-01~ (테스트는 학습에서 고른 값으로 한 번만 계산)
파라미터: 명세의 후보 격자만, 학습 구간 샤프로 선택. 시도한 조합 수를 results/grid_counts.csv 에 기록.
실행: python run.py  →  results/ 에 csv·json·png
"""
import io
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D          # noqa: E402
import signals as S       # noqa: E402
import backtest as B      # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

SEG = {"학습": ("2000-01-01", "2014-12-31"), "검증": ("2015-01-01", "2019-12-31"), "테스트": ("2020-01-01", "2026-12-31")}
SEG3 = {"학습": ("2014-05-01", "2017-12-31"), "검증": ("2018-01-01", "2019-12-31"), "테스트": ("2020-01-01", "2026-12-31")}
GRID_COUNT = {}
SUMMARY = {}
NCOL = ["CAGR", "변동성", "샤프", "MDD"]


def P(*a):
    print(*a, flush=True)


def seg_table(name, r, rf, bench=None, segs=SEG, extra=None):
    rows = []
    for k, (a, b) in list(segs.items()) + [("전체", (list(segs.values())[0][0], "2026-12-31"))]:
        rr = r.loc[a:b]
        row = dict(전략=name, 구간=k, **B.perf(rr, rf))
        if bench is not None:
            bb = bench.loc[a:b]
            pb = B.perf(bb, rf)
            row.update({"BM_CAGR": pb["CAGR"], "BM_샤프": pb["샤프"], "BM_MDD": pb["MDD"]})
            al, t, be = B.alpha_t(rr, bb, rf)
            row.update(알파=al, 알파t=t, 베타=be)
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def train_sharpe(r, rf, segs=SEG):
    a, b = segs["학습"]
    return B.perf(r.loc[a:b], rf)["샤프"]


def walk_forward(cands, rf, first=2005, last=2026, start="2000-01-01"):
    """cands: {키: 일간 수익률}. 매년 그해 이전 데이터(확장 창)로 샤프 최대 후보를 골라 그해만 이어 붙임"""
    parts, picks = [], []
    for y in range(first, last + 1):
        best, bk = -np.inf, None
        for k, r in cands.items():
            s = B.perf(r.loc[start:f"{y - 1}-12-31"], rf)["샤프"]
            if s is not None and np.isfinite(s) and s > best:
                best, bk = s, k
        if bk is None:
            continue
        parts.append(cands[bk].loc[f"{y}-01-01":f"{y}-12-31"])
        picks.append((y, str(bk)))
    return pd.concat(parts), picks


def main():
    P("== 데이터")
    R_us, P_us, fixes_us = D.us_assets()
    kos, kos_same = D.kospi()
    kos = kos.loc[:D.END]
    kcal, ucal = kos.index, R_us.index
    rf_kr, rf_us = D.cash_rates(kcal, ucal)
    r_kos = kos["c"].pct_change().fillna(0.0)
    SUMMARY["data"] = dict(kospi_start=str(kcal[0].date()), kospi_end=str(kcal[-1].date()), us_start=str(ucal[0].date()),
                           us_end=str(ucal[-1].date()), fixes_us=fixes_us,
                           kospi_open_eq_prevclose_max=float(kos_same.max()))

    # ================================================================ ① 매크로
    P("== ① COT 상업자 지수")
    cot_rows, cot_ev, cot_choice, cot_series = [], [], {}, {}
    for code, (nm, key) in D.COT_MARKETS.items():
        net = D.cot(code)
        ret = R_us[key]
        rf = rf_us if key in ("SPX", "BOND") else None       # 선물(금·원유)은 초과수익률이라 현금 0
        cands = {}
        for n, (lo, hi) in product([26, 156], [(20, 80), (25, 75)]):
            idx = S.cot_daily(net, n, ucal)
            st = S.cot_state(idx, lo, hi)
            r, ntr, turn = B.timing(st, ret, rf)
            cands[(n, lo, hi)] = (r, st, idx, ntr, turn)
            # 사건 연구: 63거래일(13주) 이후 수익률, 지수≥hi / ≤lo vs 나머지
            fwd = (1 + ret).rolling(63).apply(np.prod, raw=True).shift(-63) - 1
            for seg_name, (a, b) in SEG.items():
                f = fwd.loc[a:b]
                ii = idx.reindex(f.index)
                for lab, msk in [(f"≥{hi}", ii >= hi), (f"≤{lo}", ii <= lo)]:
                    d, t = B.diff_t(f, msk, 63)
                    cot_ev.append(dict(시장=nm, N주=n, 기준=f"{lo}/{hi}", 구간=seg_name, 조건=lab, 표본일=int(msk.sum()),
                                       조건평균=f[msk].mean(), 나머지평균=f[~msk].mean(), 차이=d, NW_t=t))
        GRID_COUNT[f"COT {nm}"] = len(cands)
        best = max(cands, key=lambda k: train_sharpe(cands[k][0], rf))
        cot_choice[key] = best
        r, st, idx, ntr, turn = cands[best]
        cot_series[key] = (r, st, idx)
        bh = ret
        rows = seg_table(f"COT {nm} N={best[0]} {best[1]}/{best[2]}", r, rf, bh, extra=dict(거래수=ntr, 회전율=turn))
        cot_rows += rows
        for k, v in cands.items():
            cot_rows.append(dict(전략=f"[후보] COT {nm} N={k[0]} {k[1]}/{k[2]}", 구간="학습", **B.perf(v[0].loc[SEG['학습'][0]:SEG['학습'][1]], rf)))
        P(nm, best, f"학습샤프 {train_sharpe(r, rf):.2f}")
    pd.DataFrame(cot_rows).to_csv(OUT / "m1_cot.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cot_ev).to_csv(OUT / "m1_cot_events.csv", index=False, encoding="utf-8-sig")
    SUMMARY["cot_choice"] = {k: list(v) for k, v in cot_choice.items()}

    P("== ① 채권 → 주식")
    bond_px = P_us["BOND"]
    bond_rows, bond_choice, bond_sig = [], {}, {}
    for tgt, ret, rf, cal in [("SPY", R_us["SPX"], rf_us, ucal), ("코스피", r_kos, rf_kr, kcal)]:
        cands = {}
        for n in [10, 20, 50, 200]:
            sig = S.above_ma(bond_px, n)
            if tgt == "코스피":
                sig = S.to_kr(sig, kcal).fillna(0.0)
            r, ntr, turn = B.timing(sig, ret, rf)
            cands[n] = (r, sig, ntr, turn)
            bond_rows.append(dict(전략=f"[후보] 채권>{n}일선 → {tgt}", 구간="학습", **B.perf(r.loc[SEG['학습'][0]:SEG['학습'][1]], rf)))
        GRID_COUNT[f"채권 필터 → {tgt}"] = len(cands)
        best = max(cands, key=lambda k: train_sharpe(cands[k][0], rf))
        bond_choice[tgt] = best
        bond_sig[tgt] = cands[best][1]
        r, sig, ntr, turn = cands[best]
        bond_rows += seg_table(f"채권>{best}일선 → {tgt}", r, rf, ret, extra=dict(거래수=ntr, 회전율=turn))
        # 신호 켜짐/꺼짐 날의 다음 날 평균 수익률 차이
        d, t = B.diff_t(ret.shift(-1), sig > 0, 5)
        bond_rows.append(dict(전략=f"채권>{best}일선 → {tgt} (켜짐-꺼짐 다음날 수익 차, 전체)", 구간="전체", 차이=d, NW_t=t))
        P(tgt, best)
    pd.DataFrame(bond_rows).to_csv(OUT / "m1_bond.csv", index=False, encoding="utf-8-sig")
    SUMMARY["bond_choice"] = bond_choice

    P("== ① 유가 → 코스피")
    oil_kr = S.to_kr(P_us["OIL"], kcal)
    fwd20 = kos["c"].shift(-20) / kos["c"] - 1
    lag_rows = []
    for m in [20, 60]:
        for L in range(0, 131, 10):
            x = oil_kr.shift(L) / oil_kr.shift(L + m) - 1
            for seg_name, (a, b) in SEG.items():
                df = pd.concat([x, fwd20], axis=1).loc[a:b].dropna()
                c = df.iloc[:, 0].corr(df.iloc[:, 1])
                # 기울기 NW t
                import statsmodels.api as sm
                mdl = sm.OLS(df.iloc[:, 1].values, sm.add_constant(df.iloc[:, 0].values)).fit(cov_type="HAC", cov_kwds={"maxlags": 20})
                lag_rows.append(dict(m일=m, 시차=L, 구간=seg_name, 상관=c, 기울기t=mdl.tvalues[1], 표본=len(df)))
    lag_df = pd.DataFrame(lag_rows)
    lag_df.to_csv(OUT / "m1_oil_lagcorr.csv", index=False, encoding="utf-8-sig")
    cands = {}
    for L, m in product(range(20, 131, 10), [20, 60]):
        sig = S.oil_signal(oil_kr, L, m)
        r, ntr, turn = B.timing(sig, r_kos, rf_kr)
        cands[(L, m)] = (r, sig, ntr, turn)
    GRID_COUNT["유가 → 코스피"] = len(cands)
    oil_best = max(cands, key=lambda k: train_sharpe(cands[k][0], rf_kr))
    r_oil, sig_oil, ntr, turn = cands[oil_best]
    oil_rows = seg_table(f"유가 시차{oil_best[0]}일·{oil_best[1]}일 추세 → 코스피", r_oil, rf_kr, r_kos, extra=dict(거래수=ntr, 회전율=turn))
    for k, v in cands.items():
        oil_rows.append(dict(전략=f"[후보] 유가 시차{k[0]}·{k[1]}일", 구간="학습", **B.perf(v[0].loc[SEG['학습'][0]:SEG['학습'][1]], rf_kr)))
        oil_rows.append(dict(전략=f"[후보] 유가 시차{k[0]}·{k[1]}일", 구간="검증", **B.perf(v[0].loc[SEG['검증'][0]:SEG['검증'][1]], rf_kr)))
    pd.DataFrame(oil_rows).to_csv(OUT / "m1_oil.csv", index=False, encoding="utf-8-sig")
    SUMMARY["oil_choice"] = list(oil_best)
    P("유가", oil_best)

    # 업종 확장(탐색): 화학·운송·창고(한국거래소 업종명), 정유·항공(대표 종목) 동일가중
    W = D.panel_wide()
    info = D.listing_info()
    groups = {
        "화학(업종)": info.index[(info.sector == "화학") & (info.sector_src == "KRX업종")].tolist(),
        "운송·창고(업종)": info.index[(info.sector == "운송·창고") & (info.sector_src == "KRX업종")].tolist(),
        "정유": ["010950", "096770", "078930"],
        "항공": ["003490", "020560", "089590", "272450", "091810"],
    }
    sec_rows = []
    for g, codes in groups.items():
        codes = [c for c in codes if c in W["ret"].columns]
        gr = W["ret"][codes].mean(axis=1).reindex(kcal).fillna(0.0)
        gp = (1 + gr).cumprod()
        f20 = gp.shift(-20) / gp - 1
        for L in [0, 20, 40, 60, 80, 100, 120]:
            x = oil_kr.shift(L) / oil_kr.shift(L + 60) - 1
            for seg_name, (a, b) in SEG.items():
                df = pd.concat([x, f20], axis=1).loc[a:b].dropna()
                sec_rows.append(dict(업종=g, 종목수=len(codes), 시차=L, 구간=seg_name, 상관=df.iloc[:, 0].corr(df.iloc[:, 1])))
        r, _, _ = B.timing(sig_oil, gr, rf_kr)
        for seg_name, (a, b) in SEG.items():
            sec_rows.append(dict(업종=g, 종목수=len(codes), 시차=f"전략({oil_best[0]},{oil_best[1]})", 구간=seg_name,
                                 전략샤프=B.perf(r.loc[a:b], rf_kr)["샤프"], 보유샤프=B.perf(gr.loc[a:b], rf_kr)["샤프"]))
    pd.DataFrame(sec_rows).to_csv(OUT / "m1_oil_sectors.csv", index=False, encoding="utf-8-sig")

    P("== ① 패닉")
    cands, pan_rows = {}, []
    for drop, hold in product([0.10, 0.15, 0.20], [60, 120]):
        pos, trig = S.panic_signal(kos["c"], drop, hold)
        r, ntr, turn = B.timing(pos, r_kos, rf_kr)
        cands[(drop, hold)] = (r, pos, ntr, turn)
        fwd = kos["c"].shift(-hold) / kos["c"] - 1
        ev = S.first_events(trig, hold)
        evd = kcal[ev]
        for seg_name, (a, b) in SEG.items():
            f = fwd.loc[a:b]
            msk = pd.Series(f.index.isin(evd), index=f.index)
            d, t = B.diff_t(f, msk, hold)
            pan_rows.append(dict(전략=f"[사건] 낙폭{int(drop * 100)}%·보유{hold}일", 구간=seg_name, 사건수=int(msk.sum()),
                                 사건후평균=f[msk].mean(), 평소평균=f[~msk].mean(), 차이=d, NW_t=t,
                                 사건일=", ".join(str(x.date()) for x in evd[(evd >= a) & (evd <= b)])))
    GRID_COUNT["패닉 매수"] = len(cands)
    pan_best = max(cands, key=lambda k: train_sharpe(cands[k][0], rf_kr))
    r_pan, pos_pan, ntr, turn = cands[pan_best]
    pan_rows = seg_table(f"패닉 낙폭{int(pan_best[0] * 100)}%·보유{pan_best[1]}일", r_pan, rf_kr, r_kos, extra=dict(거래수=ntr, 회전율=turn)) + pan_rows
    for k, v in cands.items():
        pan_rows.append(dict(전략=f"[후보] 패닉 {int(k[0] * 100)}%·{k[1]}일", 구간="학습", **B.perf(v[0].loc[SEG['학습'][0]:SEG['학습'][1]], rf_kr)))
    pd.DataFrame(pan_rows).to_csv(OUT / "m1_panic.csv", index=False, encoding="utf-8-sig")
    SUMMARY["panic_choice"] = list(pan_best)

    # 매크로 워크포워드(확장 창, 연 단위)
    wf_rows, wf_series = [], {}
    # 후보별 일간 수익률 다시 모음(워크포워드용)
    net = D.cot("13874A")
    c_spx = {}
    for n, (lo, hi) in product([26, 156], [(20, 80), (25, 75)]):
        c_spx[(n, lo, hi)] = B.timing(S.cot_state(S.cot_daily(net, n, ucal), lo, hi), R_us["SPX"], rf_us)[0]
    c_bond = {n: B.timing(S.to_kr(S.above_ma(bond_px, n), kcal).fillna(0.0), r_kos, rf_kr)[0] for n in [10, 20, 50, 200]}
    c_oil = {}
    for L, m in product(range(20, 131, 10), [20, 60]):
        c_oil[(L, m)] = B.timing(S.oil_signal(oil_kr, L, m), r_kos, rf_kr)[0]
    c_pan = {(d_, h_): B.timing(S.panic_signal(kos["c"], d_, h_)[0], r_kos, rf_kr)[0] for d_, h_ in product([0.10, 0.15, 0.20], [60, 120])}
    for nm, cset, rf, bench in [("COT S&P500 → SPY", c_spx, rf_us, R_us["SPX"]), ("채권 필터 → 코스피", c_bond, rf_kr, r_kos),
                                 ("유가 → 코스피", c_oil, rf_kr, r_kos), ("패닉 → 코스피", c_pan, rf_kr, r_kos)]:
        wr, picks = walk_forward(cset, rf)
        wf_series[nm] = wr
        pb = B.perf(bench.loc[wr.index[0]:wr.index[-1]], rf)
        al, t, be = B.alpha_t(wr, bench, rf)
        wf_rows.append(dict(전략=nm, 기간=f"{wr.index[0].date()}~{wr.index[-1].date()}", **B.perf(wr, rf), BM_CAGR=pb["CAGR"],
                            BM_샤프=pb["샤프"], BM_MDD=pb["MDD"], 알파=al, 알파t=t,
                            선택이력="; ".join(f"{y}:{k}" for y, k in picks)))

    # ================================================================ ② 단기 시스템
    P("== ② 단기 시스템 (코스피 일봉)")
    n_c, lo_c, hi_c = cot_choice["SPX"]
    cot_kr = S.to_kr(S.cot_state(S.cot_daily(D.cot("13874A"), n_c, ucal), lo_c, hi_c), kcal).fillna(0.0)
    cond = (cot_kr > 0) & (bond_sig["코스피"] > 0)
    state = S.wr_state(kos["h"], kos["l"], kos["c"])
    # TDM 분석(코스피·S&P)
    tdm_rows = []
    spy_o = D.ohlc("YAHOO:SPY")
    for nm, ret, idx in [("코스피", r_kos, kcal), ("S&P500(SPY)", R_us["SPX"], ucal)]:
        for dn in [3, 4, 5]:
            w = S.tdm_window(idx, dn)
            for seg_name, (a, b) in list(SEG.items()) + [("전체", ("2000-01-01", "2026-12-31"))]:
                rr, ww = ret.loc[a:b], w.loc[a:b]
                d, t = B.diff_t(rr, ww, 5)
                tdm_rows.append(dict(시장=nm, 월초일수=dn, 구간=seg_name, TDM일평균=rr[ww].mean(), 나머지일평균=rr[~ww].mean(),
                                     차이=d, NW_t=t, TDM비중=ww.mean()))
    pd.DataFrame(tdm_rows).to_csv(OUT / "m2_tdm.csv", index=False, encoding="utf-8-sig")
    # 선거
    el_rows = []
    for nm, ret, idx in [("코스피", r_kos, kcal), ("S&P500(SPY)", R_us["SPX"], ucal)]:
        for ek, dates in S.ELECTIONS.items():
            w = S.election_window(idx, dates)
            d, t = B.diff_t(ret, w, 21)
            el_rows.append(dict(시장=nm, 선거=ek, 횟수=len(dates), 창일평균=ret[w].mean(), 나머지일평균=ret[~w].mean(),
                                연율차이=d * 252 if d == d else np.nan, NW_t=t))
    pd.DataFrame(el_rows).to_csv(OUT / "m2_election.csv", index=False, encoding="utf-8-sig")

    trig_list = [("oops", 0.0)] + [("vb", k) for k in [0.4, 0.5, 0.6, 0.8, 1.0]]
    cands, trade_store = {}, {}
    for dn, (tg, k), T, sm_ in product([3, 4, 5], trig_list, [3, 5, 10], [1, 2]):
        allow = cond.shift(1).fillna(False) & S.tdm_window(kcal, dn) & state.shift(1).fillna(False)
        tr = B.short_trades(kos, allow, tg, k, T, sm_)
        trade_store[(dn, tg, k, T, sm_)] = tr
        for risk in [0.01, 0.02, 0.03]:
            r, f = B.trades_to_returns(kos, tr, risk, rf_kr)
            cands[(dn, tg, k, T, sm_, risk)] = r
    GRID_COUNT["② 단기 시스템"] = len(cands)

    def pick_sys(cset):
        best, bk = -np.inf, None
        for key, r in cset.items():
            p = B.perf(r.loc[SEG["학습"][0]:SEG["학습"][1]], rf_kr)
            if p["MDD"] >= -0.30 and np.isfinite(p["샤프"]) and p["샤프"] > best:
                best, bk = p["샤프"], key
        return bk
    sys_best = pick_sys(cands)
    dn, tg, k, T, sm_, risk = sys_best
    tr_best = trade_store[(dn, tg, k, T, sm_)]
    r_sys = cands[sys_best]
    lab = f"깔때기 TDM{dn}·{'Oops' if tg == 'oops' else f'변동성돌파 k={k}'}·시간{T}일·손절{sm_}×·위험{int(risk * 100)}%"
    sys_rows = seg_table(lab, r_sys, rf_kr, r_kos, extra=dict(거래수=len(tr_best)))
    for seg_name, (a, b) in SEG.items():
        t_ = tr_best[(tr_best.진입일 >= a) & (tr_best.진입일 <= b)]
        rr = t_.청산가 / t_.진입가 - 1
        sys_rows.append(dict(전략="거래 통계", 구간=seg_name, 거래수=len(t_), 승률=(rr > 0).mean() if len(t_) else np.nan,
                             평균거래수익=rr.mean() if len(t_) else np.nan, 사유=t_.사유.value_counts().to_dict()))
    # 비용 2배
    r2, _ = B.trades_to_returns(kos, tr_best, risk, rf_kr, cost=B.COST * 2)
    sys_rows += seg_table(lab + " [비용 2배]", r2, rf_kr, r_kos)
    # 조건 하나씩 빼기 (강건성)
    abl = {
        "조건(COT·채권) 뺌": S.tdm_window(kcal, dn) & state.shift(1).fillna(False),
        "시기(TDM) 뺌": cond.shift(1).fillna(False) & state.shift(1).fillna(False),
        "상태(%R) 뺌": cond.shift(1).fillna(False) & S.tdm_window(kcal, dn),
        "방아쇠만(매일)": pd.Series(True, index=kcal),
    }
    abl_series = {}
    for nm, allow in abl.items():
        tr = B.short_trades(kos, allow, tg, k, T, sm_)
        r, _ = B.trades_to_returns(kos, tr, risk, rf_kr)
        abl_series[nm] = r
        sys_rows += seg_table(f"[뺌] {nm}", r, rf_kr, r_kos, extra=dict(거래수=len(tr)))
    # 파라미터 이웃(한 칸씩 바꾸기): 학습/테스트 샤프
    nb_rows = []
    for key, r in cands.items():
        diff = sum(1 for x, y in zip(key, sys_best) if x != y)
        if diff <= 1:
            nb_rows.append(dict(조합=str(key), 다른칸수=diff, 학습샤프=B.perf(r.loc[SEG['학습'][0]:SEG['학습'][1]], rf_kr)["샤프"],
                                검증샤프=B.perf(r.loc[SEG['검증'][0]:SEG['검증'][1]], rf_kr)["샤프"],
                                테스트샤프=B.perf(r.loc[SEG['테스트'][0]:], rf_kr)["샤프"]))
    pd.DataFrame(nb_rows).to_csv(OUT / "m2_neighbors.csv", index=False, encoding="utf-8-sig")
    # 전체 후보 학습 샤프 분포 (과최적화 점검)
    allv = pd.DataFrame([dict(조합=str(kk), 학습샤프=B.perf(v.loc[SEG['학습'][0]:SEG['학습'][1]], rf_kr)["샤프"],
                              검증샤프=B.perf(v.loc[SEG['검증'][0]:SEG['검증'][1]], rf_kr)["샤프"]) for kk, v in cands.items()])
    allv.to_csv(OUT / "m2_all_candidates.csv", index=False, encoding="utf-8-sig")
    SUMMARY["m2_rank_corr_train_valid"] = float(allv[["학습샤프", "검증샤프"]].corr(method="spearman").iloc[0, 1])
    # KODEX 200 (2014-08~) 같은 규칙
    kdx = D.kodex200()
    kdx = kdx[(kdx.o > 0)].loc[:D.END]
    kdx_cal = kdx.index
    cond_k = cond.reindex(kdx_cal).fillna(False)
    st_k = S.wr_state(kdx["h"], kdx["l"], kdx["c"])
    allow_k = cond_k.shift(1).fillna(False) & S.tdm_window(kdx_cal, dn) & st_k.shift(1).fillna(False)
    tr_k = B.short_trades(kdx, allow_k, tg, k, T, sm_)
    rf_k = rf_kr.reindex(kdx_cal).ffill()
    r_k, _ = B.trades_to_returns(kdx, tr_k, risk, rf_k)
    r_kdx = kdx["c"].pct_change().fillna(0.0)
    sys_rows += seg_table("KODEX 200(069500)로 같은 규칙", r_k.loc["2015-01-01":], rf_k, r_kdx,
                          segs={"검증": ("2015-01-01", "2019-12-31"), "테스트": ("2020-01-01", "2026-12-31")}, extra=dict(거래수=len(tr_k)))
    pd.DataFrame(sys_rows).to_csv(OUT / "m2_system.csv", index=False, encoding="utf-8-sig")
    tr_best.to_csv(OUT / "m2_trades.csv", index=False, encoding="utf-8-sig")
    SUMMARY["sys_filter_share"] = dict(조건=float(cond.mean()), 상태=float(state.mean()), TDM=float(S.tdm_window(kcal, dn).mean()),
                                       전부=float((cond.shift(1).fillna(False) & S.tdm_window(kcal, dn) & state.shift(1).fillna(False)).mean()))
    SUMMARY["sys_choice"] = dict(TDM월초=dn, 방아쇠=tg, k=k, 시간청산=T, 손절배수=sm_, 위험=risk)
    # 워크포워드 ②
    wr2, picks2 = walk_forward(cands, rf_kr)
    pb = B.perf(r_kos.loc[wr2.index[0]:wr2.index[-1]], rf_kr)
    al, t, be = B.alpha_t(wr2, r_kos, rf_kr)
    wf_rows.append(dict(전략="② 단기 시스템", 기간=f"{wr2.index[0].date()}~{wr2.index[-1].date()}", **B.perf(wr2, rf_kr),
                        BM_CAGR=pb["CAGR"], BM_샤프=pb["샤프"], BM_MDD=pb["MDD"], 알파=al, 알파t=t,
                        선택이력="; ".join(f"{y}:{kk}" for y, kk in picks2)))
    wf_series["② 단기 시스템"] = wr2
    pd.DataFrame(wf_rows).to_csv(OUT / "walk_forward.csv", index=False, encoding="utf-8-sig")
    P("시스템", sys_best)

    # ================================================================ ③ 우량주 필터
    P("== ③ 우량주 필터")
    import quality as QM
    QQ = QM.Quality(W, info)
    C, RET = QQ.C, QQ.R
    cal3 = C.index
    rebal = QQ.month_ends(4)
    end3 = cal3[-1]
    mcs = QQ.mcaps(rebal)
    sec_counts = info.loc[info.index.isin(C.columns)].sector_src.value_counts().to_dict()
    U = {d: QQ.universe(d, mcs[d]) for d in rebal}
    # 분포 기준선: 학습 구간 리밸런싱(2014~2017) 비금융 종목 풀링
    QQ.set_thresholds(pd.concat([U[d] for d in rebal if d.year <= 2017]))
    qs = QQ.qs
    SUMMARY["dist_thresholds"] = {str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in qs.items()}
    pick = QQ.pick
    entries = QQ.entries
    rf3 = rf_kr.reindex(cal3).ffill().fillna(0.0)
    r_kos3 = r_kos.reindex(cal3).fillna(0.0)
    # 벤치마크: 동일가중 전 종목(같은 유니버스, 즉시 진입)
    all_picks = {d: U[d].index.tolist() for d in rebal}
    r_ew, _, _, _ = B.stock_portfolio(rebal + [end3], all_picks, None, RET, rf3)
    rebal = rebal + [end3]
    scen_list = QM.SCEN
    cands3, m3_rows, info3 = {}, [], {}
    for sc in scen_list:
        picks = {d: pick(U[d], sc) for d in rebal[:-1]}
        npk = {str(d.date()): len(v) for d, v in picks.items()}
        for en, sig in entries.items():
            r, turn, hold, fill = B.stock_portfolio(rebal, picks, sig, RET, rf3)
            cands3[(sc, en)] = r
            info3[(sc, en)] = dict(회전율=turn, 평균종목수=hold, 진입률=fill, 연도별종목수=npk, picks=picks)
            m3_rows.append(dict(전략=f"[후보] {sc}·{en}", 구간="학습", **B.perf(r.loc[SEG3['학습'][0]:SEG3['학습'][1]], rf3),
                                평균종목수=hold, 진입률=fill))
        r0, turn0, hold0, _ = B.stock_portfolio(rebal, picks, None, RET, rf3)
        m3_rows.append(dict(전략=f"[참고] {sc}·즉시 진입", 구간="전체", **B.perf(r0, rf3), 평균종목수=hold0))
    GRID_COUNT["③ 우량주 필터"] = len(cands3)
    best3 = max(cands3, key=lambda kk: train_sharpe(cands3[kk], rf3, SEG3))
    r3 = cands3[best3]
    i3 = info3[best3]
    m3_rows = seg_table(f"{best3[0]}·{best3[1]}", r3, rf3, r_kos3, SEG3,
                        extra=dict(회전율=i3["회전율"], 평균종목수=i3["평균종목수"])) + \
        seg_table("벤치마크: 동일가중 전 종목", r_ew, rf3, r_kos3, SEG3) + \
        seg_table(f"{best3[0]}·{best3[1]} vs 동일가중", r3, rf3, r_ew, SEG3) + m3_rows
    r3c, _, _, _ = B.stock_portfolio(rebal, i3["picks"], entries[best3[1]], RET, rf3, cost=B.COST * 2)
    m3_rows += seg_table(f"{best3[0]}·{best3[1]} [비용 2배]", r3c, rf3, r_kos3, SEG3)
    kos_codes = set(info.index[info.market == "KOSPI"])
    picks_k = {d: pick(U[d][U[d].index.isin(kos_codes)], best3[0]) for d in rebal[:-1]}
    r3k, _, hk, _ = B.stock_portfolio(rebal, picks_k, entries[best3[1]], RET, rf3)
    m3_rows += seg_table(f"{best3[0]}·{best3[1]} [코스피 확인 종목만]", r3k, rf3, r_kos3, SEG3, extra=dict(평균종목수=hk))
    pd.DataFrame(m3_rows).to_csv(OUT / "m3_quality.csv", index=False, encoding="utf-8-sig")
    yr3 = pd.DataFrame({"전략": B.yearly(r3), "코스피": B.yearly(r_kos3.loc[r3.index[0]:]), "동일가중": B.yearly(r_ew)})
    yr3.to_csv(OUT / "m3_yearly.csv", encoding="utf-8-sig")
    SUMMARY["m3_fill_rows"] = QQ.n_fill
    SUMMARY["m3_missing_paid"] = {str(d.date()): int((U[d].tax_exp_ok & U[d].tax_paid.isna()).sum()) if "tax_paid" in U[d] else 0 for d in rebal[:-1]}
    pd.to_pickle(dict(r3=r3, r_ew=r_ew, r_kos3=r_kos3, rf3=rf3, best3=best3, picks=i3["picks"], cands3=cands3),
                 OUT / "m3_series.pkl")
    hold_last = U[rebal[-2]].loc[i3["picks"][rebal[-2]]][["sector", "de", "per", "psr", "mcap"]] if i3["picks"][rebal[-2]] else pd.DataFrame()
    hold_last = hold_last.join(info[["name"]])
    hold_last.to_csv(OUT / "m3_latest_picks.csv", encoding="utf-8-sig")
    SUMMARY["m3_choice"] = list(best3)
    SUMMARY["m3_counts"] = i3["연도별종목수"]
    SUMMARY["m3_tax_src"] = {str(d.date()): U[d].tax_src.iloc[0] for d in rebal[:-1]}
    SUMMARY["m3_universe_n"] = {str(d.date()): len(U[d]) for d in rebal[:-1]}
    SUMMARY["sector_src_counts"] = sec_counts
    SUMMARY["panel_breaks"] = int(len(W["breaks"]))
    SUMMARY["grid_counts"] = GRID_COUNT
    SUMMARY["grid_total"] = int(sum(GRID_COUNT.values()))
    pd.Series(GRID_COUNT).to_csv(OUT / "grid_counts.csv", encoding="utf-8-sig", header=["시도 조합 수"])
    P("우량주", best3)

    # ================================================================ 연도별·차트
    yr = pd.DataFrame({
        "코스피": B.yearly(r_kos), "SPY": B.yearly(R_us["SPX"]),
        "COT S&P→SPY": B.yearly(cot_series["SPX"][0]), "채권→코스피": B.yearly(B.timing(bond_sig["코스피"], r_kos, rf_kr)[0]),
        "유가→코스피": B.yearly(r_oil), "패닉→코스피": B.yearly(r_pan), "② 단기 시스템": B.yearly(r_sys),
    })
    yr.to_csv(OUT / "yearly.csv", encoding="utf-8-sig")
    pd.to_pickle(dict(cot_spx=cot_series["SPX"][0], spy=R_us["SPX"], rf_us=rf_us, r_kos=r_kos, rf_kr=rf_kr,
                      bond_kos=B.timing(bond_sig["코스피"], r_kos, rf_kr)[0], oil=r_oil, panic=r_pan, sys=r_sys,
                      abl=abl_series, wf=wf_series, kos=kos), OUT / "daily.pkl")
    charts(kos, r_kos, R_us, rf_kr, cot_series, bond_sig, r_oil, r_pan, r_sys, abl_series, lag_df, r3, r_ew, r_kos3, wf_series,
           tdm_rows, best3)
    (OUT / "summary.json").write_text(json.dumps(SUMMARY, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    P("완료")


def _log_y(ax):
    """로그 축 눈금을 일반 숫자로 (유니코드 마이너스 글꼴 경고 방지)"""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _eq(r):
    return (1 + r.fillna(0)).cumprod()


def _shade(ax):
    for nm, (a, b) in SEG.items():
        if nm != "학습":
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#f2f2f2" if nm == "검증" else "#e6eef7", zorder=0)


def charts(kos, r_kos, R_us, rf_kr, cot_series, bond_sig, r_oil, r_pan, r_sys, abl, lag_df, r3, r_ew, r_kos3, wf, tdm_rows, best3):
    # 1) 매크로 자산곡선
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    r_cot, st, idx = cot_series["SPX"]
    for ax, (ttl, a, b, la, lb) in zip(axes.flat, [
        ("COT 상업자 지수 → SPY", r_cot, R_us["SPX"], "전략", "SPY 보유"),
        ("채권 추세 필터 → 코스피", B.timing(bond_sig["코스피"], r_kos, rf_kr)[0], r_kos, "전략", "코스피 보유"),
        ("유가 시차 신호 → 코스피", r_oil, r_kos, "전략", "코스피 보유"),
        ("패닉 매수 → 코스피", r_pan, r_kos, "전략", "코스피 보유")]):
        _shade(ax)
        ax.plot(_eq(a.loc["2000":]), label=la, color="#1f4e79", lw=1.3)
        ax.plot(_eq(b.loc["2000":]), label=lb, color="#a6a6a6", lw=1.0)
        _log_y(ax); ax.set_title(ttl, fontsize=11); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("① 매크로 신호 누적 성과 (회색 배경=검증, 파란 배경=테스트)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "fig1_macro.png", dpi=160); plt.close(fig)
    # 2) 유가 시차 상관
    fig, ax = plt.subplots(figsize=(9, 4))
    for (m, seg), g in lag_df.groupby(["m일", "구간"]):
        if seg in ("학습", "테스트"):
            ax.plot(g.시차, g.상관, marker="o", ms=3, label=f"{m}일 수익률·{seg}", ls="-" if seg == "학습" else "--")
    ax.axhline(0, color="k", lw=0.6); ax.axvspan(30, 130, color="#fff2cc", alpha=0.6, label="래리 주장 1.5~6개월")
    ax.set_xlabel("유가 시차(거래일)"); ax.set_ylabel("코스피 이후 20일 수익률과 상관")
    ax.set_title("유가 → 코스피 시차 상관"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig2_oil_lag.png", dpi=160); plt.close(fig)
    # 3) COT 지수와 S&P
    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax1.plot(idx.loc["2000":], color="#c55a11", lw=0.6, label="상업자 지수")
    ax1.set_ylabel("상업자 지수(0~100)")
    ax2 = ax1.twinx(); ax2.plot(_eq(R_us["SPX"]).loc["2000":], color="#1f4e79", lw=1.0, label="SPY"); _log_y(ax2)
    ax1.set_title("S&P500 상업자 지수와 SPY"); ax1.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig3_cot_spx.png", dpi=160); plt.close(fig)
    # 4) 단기 시스템
    fig, ax = plt.subplots(figsize=(11, 4.5))
    _shade(ax)
    ax.plot(_eq(r_sys.loc["2000":]), label="깔때기 시스템(전부)", color="#1f4e79", lw=1.5)
    for (nm, r), col in zip(abl.items(), ["#70ad47", "#ffc000", "#c55a11", "#7f7f7f"]):
        ax.plot(_eq(r.loc["2000":]), label=nm, lw=0.9, color=col)
    ax.plot(_eq(r_kos.loc["2000":]), label="코스피 보유", color="k", lw=0.7, ls=":")
    _log_y(ax); ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3); ax.set_title("② 단기 매매 시스템과 조건을 하나씩 뺀 버전")
    fig.tight_layout(); fig.savefig(OUT / "fig4_system.png", dpi=160); plt.close(fig)
    # 5) 우량주
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for nm, (a, b) in SEG3.items():
        if nm != "학습":
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#f2f2f2" if nm == "검증" else "#e6eef7", zorder=0)
    ax.plot(_eq(r3), label=f"우량주 필터({best3[0]}·{best3[1]})", color="#1f4e79", lw=1.5)
    ax.plot(_eq(r_ew), label="동일가중 전 종목", color="#c55a11", lw=1.0)
    ax.plot(_eq(r_kos3.loc[r3.index[0]:]), label="코스피", color="#7f7f7f", lw=1.0)
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_title("③ 우량주 필터 누적 성과 (2014-04 리밸런싱부터)")
    fig.tight_layout(); fig.savefig(OUT / "fig5_quality.png", dpi=160); plt.close(fig)
    # 6) TDM
    t = pd.DataFrame(tdm_rows)
    t = t[(t.구간 == "전체")]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    xs = np.arange(len(t))
    ax.bar(xs - 0.2, t.TDM일평균 * 1e4, 0.4, label="TDM 구간", color="#1f4e79")
    ax.bar(xs + 0.2, t.나머지일평균 * 1e4, 0.4, label="나머지", color="#a6a6a6")
    ax.set_xticks(xs); ax.set_xticklabels([f"{a}\n월초{b}일" for a, b in zip(t.시장, t.월초일수)], fontsize=8)
    ax.set_ylabel("일평균 수익률(bp)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y"); ax.set_title("월말·월초(TDM) 효과, 2000~2026")
    fig.tight_layout(); fig.savefig(OUT / "fig6_tdm.png", dpi=160); plt.close(fig)
    # 7) 워크포워드
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for nm, r in wf.items():
        ax.plot(_eq(r), label=nm, lw=1.1)
    ax.plot(_eq(r_kos.loc["2005":]), label="코스피 보유", color="k", lw=0.7, ls=":")
    _log_y(ax); ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("연 단위 워크포워드(확장 창) 2005~2026")
    fig.tight_layout(); fig.savefig(OUT / "fig7_walkforward.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
