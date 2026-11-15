# -*- coding: utf-8 -*-
"""래리 윌리엄스 추천 중형 가치주 ETF 검증 - 실행 (python run.py → results/*.csv, charts/*.png)

A. 미국 ETF·펀드 성과: VOE·IWS·Fidelity(FSMVX·FLPSX)·IJJ vs SPY·QQQ (구간: 학습 ~2014 / 검증 2015~2019 / 테스트 2020~)
B. 특성 추출: 파마-프렌치 6팩터 노출, SPY·QQQ와의 수익 차이 분해, 중형×가치 2×2, 언제 이기나(금리·추세·상대 모멘텀)
C. 상대 모멘텀 전환 규칙(사전 등록 6개 조합, 학습에서 선택 → 검증·테스트 1회)
D. 한국판 중형 가치(코스피, 상장폐지 포함): 장기판(B/P, 2006~) 8개 + CRSP판(B/P·E/P·S/P, 2014~) 8개
실매매 코드 없음.
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import data as DT
import backtest as B
import signals as S

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
FUNDS = ["VOE", "IWS", "FSMVX", "FLPSX", "IJJ"]
COMP = ["VO", "IJH", "IWD", "SPY", "QQQ"]
COST_SWITCH = 0.0005          # ETF 전환 편도 0.05%(수수료+스프레드)
COST_KR = 0.0025              # 국내 편도 0.25%(수수료·세금·슬리피지, 중형주)


def save(df, name):
    df.to_csv(OUT / f"{name}.csv", encoding="utf-8-sig")
    return df


def monthly(r):
    return (1 + r.dropna()).resample("ME").prod() - 1


# ---------------------------------------------------------------- A. 미국
def part_a(R, rf):
    rows = []
    for t in FUNDS + COMP:
        r = R[t].dropna()
        rows.append(dict(종목=t, 구간="상장 이후", **B.perf(r, rf)))
        rows.append(dict(종목=t, 구간="공통(2006-09~)", **B.perf(r[r.index >= "2006-09-01"], rf)))
        for s in ["학습", "검증", "테스트"]:
            rows.append(dict(종목=t, 구간=s, **B.perf(B.seg(r, s), rf)))
    perf = save(pd.DataFrame(rows), "a_perf")

    yr = pd.DataFrame({t: B.yearly(R[t]) for t in FUNDS + COMP})
    first = R.apply(lambda s: s.first_valid_index())
    for t in yr:                                         # 첫해 일부 기간은 비움
        yr.loc[yr.index == first[t].year, t] = np.nan
    save(yr, "a_yearly")

    M = pd.DataFrame({t: monthly(R[t]) for t in FUNDS + COMP})
    roll = []
    for t in FUNDS:
        for b in ["SPY", "QQQ"]:
            for n in [36, 60]:
                g = (1 + M[[t, b]].dropna()).rolling(n).apply(np.prod, raw=True).dropna() - 1
                win = (g[t] > g[b]).mean() if len(g) else np.nan
                rel = (g[t] - g[b]).median() if len(g) else np.nan
                roll.append(dict(종목=t, 비교=b, 창=f"{n // 12}년", 창수=len(g), 이긴비율=win, 누적차이중앙값=rel))
    save(pd.DataFrame(roll), "a_rolling_win")

    cap = []
    for t in FUNDS + ["VO", "IWD", "QQQ"]:
        d = M[[t, "SPY"]].dropna()
        up, dn = d[d.SPY > 0], d[d.SPY < 0]
        worst = d.SPY.nsmallest(max(1, len(d) // 10)).index
        cap.append(dict(종목=t, 개월=len(d), 상승포착=up[t].mean() / up.SPY.mean(), 하락포착=dn[t].mean() / dn.SPY.mean(),
                        SPY최악10퍼센트달_평균=d.loc[worst, t].mean(), SPY최악10퍼센트달_SPY=d.loc[worst, "SPY"].mean()))
    save(pd.DataFrame(cap), "a_capture")
    return perf, yr, M


# ---------------------------------------------------------------- B. 특성
def part_b(R, ff, M):
    rows = []
    for t in FUNDS + COMP:
        for s in ["전체", "학습", "검증", "테스트"]:
            o = B.factor_reg(B.seg(R[t], s), ff)
            if o:
                rows.append(dict(종목=t, 구간=s, **o))
    reg = save(pd.DataFrame(rows), "b_ff6")

    # 수익 차이 분해: (VOE − X) 연 차이 = 알파 차이 + Σ (베타 차이 × 팩터 평균)
    dec = []
    for t, x in [("VOE", "SPY"), ("VOE", "QQQ"), ("IWS", "SPY"), ("IWS", "QQQ")]:
        for nm, a, b in [("앞(~2014)", "2001-01-01", "2014-12-31"), ("뒤(2015~)", "2015-01-01", "2026-12-31"),
                         ("전체", "2001-01-01", "2026-12-31")]:
            d = pd.concat([R[t].rename("a"), R[x].rename("b"), ff], axis=1, join="inner")[a:b].dropna()
            if len(d) < 250:
                continue
            ra = B.factor_reg(d["a"], ff.loc[d.index])
            rb = B.factor_reg(d["b"], ff.loc[d.index])
            row = dict(펀드=t, 비교=x, 구간=nm, 시작=f"{d.index[0]:%Y-%m}",
                       실제차이=(d["a"].mean() - d["b"].mean()) * B.ANN, 알파차이=ra["알파(연)"] - rb["알파(연)"])
            for c in B.FF6:
                row[c] = (ra[c] - rb[c]) * ff.loc[d.index, c].mean() * B.ANN
            dec.append(row)
    save(pd.DataFrame(dec), "b_gap_decomp")

    # 2×2: 중형 효과(VO−SPY), 중형 안 가치(VOE−VO), 대형 안 가치(IWD−SPY), 가치 안 중형(VOE−IWD)
    pairs = [("중형−대형(전체)", "VO", "SPY"), ("가치−전체(중형 안)", "VOE", "VO"),
             ("가치−전체(대형 안)", "IWD", "SPY"), ("중형−대형(가치 안)", "VOE", "IWD"), ("중형가치−나스닥100", "VOE", "QQQ")]
    st = []
    for nm, a, b in pairs:
        for sg, s0, s1 in [("2006-09~2014", "2006-09-01", "2014-12-31"), ("2015~2026", "2015-01-01", "2026-12-31"),
                           ("전체(2006-09~)", "2006-09-01", "2026-12-31")]:
            d = (M[a] - M[b])[s0:s1].dropna()
            m, t = B.mean_t(d, 3)
            st.append(dict(비교=nm, 구간=sg, 월수=len(d), 연평균차이=m * 12, t=t, 플러스달비율=(d > 0).mean()))
    save(pd.DataFrame(st), "b_style_2x2")

    # 파마-프렌치 장기: 10년 단위 가치(HML)·소형(SMB) 프리미엄
    dd = []
    for dec0 in range(1960, 2030, 10):
        f = ff[(ff.index.year >= dec0) & (ff.index.year < dec0 + 10)]
        if len(f) < 250:
            continue
        for c in ["SMB", "HML", "RMW", "CMA", "MOM"]:
            m, t = B.mean_t(f[c], 5)
            dd.append(dict(연대=f"{dec0}년대", 팩터=c, 연평균=m * B.ANN, t=t))
    save(pd.DataFrame(dd).pivot(index="연대", columns="팩터", values="연평균"), "b_ff_decades")
    save(pd.DataFrame(dd), "b_ff_decades_t")

    # 3년 이동 노출(VOE·IWS)
    rl = []
    for t in ["IWS", "VOE"]:
        d = pd.concat([R[t].rename("r"), ff], axis=1, join="inner").dropna()
        ends = d.resample("QE").last().index
        for e in ends:
            w = d[(d.index > e - pd.DateOffset(years=3)) & (d.index <= e)]
            if len(w) < 600:
                continue
            o = B.factor_reg(w["r"], w[ff.columns])
            rl.append(dict(종목=t, 날짜=e, **{c: o[c] for c in B.FF6}, 알파=o["알파(연)"]))
    save(pd.DataFrame(rl), "b_rolling_beta")
    return reg


def part_b_regime(R, M):
    """다음 달 IWS − SPY / IWS − QQQ 상대수익을 월말 상태별로 비교 (IWS가 가장 김: 2001-08~)"""
    y10 = DT.dgs10().resample("ME").last()
    spy = (1 + R["SPY"].fillna(0)).cumprod()
    above = (spy > spy.rolling(200).mean()).resample("ME").last()
    rows = []
    for b in ["SPY", "QQQ"]:
        rel = (M["IWS"] - M[b]).dropna()
        nxt = rel.shift(-1)
        trail12 = (1 + M["IWS"]).rolling(12).apply(np.prod, raw=True) / (1 + M[b]).rolling(12).apply(np.prod, raw=True) - 1
        conds = {
            "10년 금리 3개월 상승": (y10.diff(3) > 0).reindex(rel.index),
            "SPY 200일선 아래": (~above.astype(bool)).reindex(rel.index),
            f"직전 12개월 IWS가 {b}보다 높음": (trail12 > 0).reindex(rel.index),
        }
        for nm, c in conds.items():
            ok = c.notna() & nxt.notna()
            dif, t, n = B.diff_t(nxt[ok], c[ok].astype(bool), 3)
            rows.append(dict(비교=f"IWS−{b}", 상태=nm, 해당달=n, 전체달=int(ok.sum()),
                             해당달_평균=nxt[ok][c[ok].astype(bool)].mean() * 12, 나머지_평균=nxt[ok][~c[ok].astype(bool)].mean() * 12,
                             차이_연율=dif * 12, t=t))
    return save(pd.DataFrame(rows), "b_regime")


# ---------------------------------------------------------------- C. 전환 규칙
def part_c(R, rf, M):
    rows, keep = [], {}
    for b in ["SPY", "QQQ"]:
        ra, rb = R["IWS"], R[b]
        d = pd.concat([ra, rb], axis=1).dropna()
        ra, rb = d.iloc[:, 0], d.iloc[:, 1]
        for L in [3, 6, 12]:
            ga = (1 + M["IWS"]).rolling(L).apply(np.prod, raw=True)
            gb = (1 + M[b]).rolling(L).apply(np.prod, raw=True)
            sig = (ga > gb).astype(float).where(ga.notna() & gb.notna())
            r, nsw = B.switch_portfolio(sig, ra, rb, COST_SWITCH)
            r2, _ = B.switch_portfolio(sig, ra, rb, COST_SWITCH * 2)
            keep[(b, L)] = r
            for s in ["학습", "검증", "테스트"]:
                p = B.perf(B.seg(r, s), rf)
                a, t, _ = B.capm_alpha(B.seg(r, s), rb, rf)
                rows.append(dict(비교=b, 기간L=L, 구간=s, 전환수=nsw, **p, 알파_vs비교=a, t=t,
                                 샤프_비용2배=B.perf(B.seg(r2, s), rf)["샤프"],
                                 샤프_IWS보유=B.perf(B.seg(ra[r.index], s), rf)["샤프"], 샤프_비교보유=B.perf(B.seg(rb[r.index], s), rf)["샤프"]))
    df = save(pd.DataFrame(rows), "c_switch")
    tr = df[df.구간 == "학습"]
    chosen = {b: int(tr[tr.비교 == b].sort_values("샤프", ascending=False).iloc[0].기간L) for b in ["SPY", "QQQ"]}
    save(pd.Series(chosen, name="학습에서 고른 기간(개월)").to_frame(), "c_switch_chosen")
    return df, chosen, keep


# ---------------------------------------------------------------- D. 한국
def part_d():
    K = DT.kr_inputs()
    KM = S.KrMidValue(K)
    R = K["W"]["ret"]
    cal = R.index
    kospi = K["kospi"]["c"].pct_change().reindex(cal)
    rf = pd.Series(0.0, index=cal)
    try:
        rf = K["LD"].cash_rates(cal, cal)[0].reindex(cal).fillna(0.0)
    except Exception:
        pass
    end = cal[-1]
    tables = {}
    res, rets, contribs, info_rows = [], {}, {}, []

    def run_one(name, dates, fn, cost=COST_KR):
        W = {d: fn(d) for d in dates}
        r, turn, nh, ct = B.weight_portfolio(list(dates) + [end], W, R, cost)
        return r, turn, nh, ct, W

    for ver, y0 in [("장기", 2006), ("CRSP", 2014)]:
        dates = S.april_ends(cal, y0, 2026)
        for d in dates:
            if d not in tables:
                tables[d] = KM.table(d)
                t = tables[d]
                info_rows.append(dict(날짜=d.date(), 코스피비금융=len(t), BP있음=int(t.bp.notna().sum()),
                                      BP재무보완=int((t.bp_src == "재무").sum()), EP있음=int(t.ep.notna().sum()),
                                      CRSP중형수=len(KM.mid(t, "CRSP")), KRX중형수=len(KM.mid(t, "KRX"))))
        combos = [(m, v, w) for m in S.MID for v in S.VAL for w in S.WGT]
        bench = [("CRSP", "전체", "시총"), ("KRX", "전체", "시총"), ("대형", "절반", "시총"), ("대형", "전체", "시총")]
        for m, v, w in combos + bench:
            key = f"{ver}|{m}|{v}|{w}"
            r, turn, nh, ct, Wt = run_one(key, dates, lambda d: KM.pick(d, m, v, w, ver, tables[d]))
            r2, *_ = run_one(key, dates, lambda d: KM.pick(d, m, v, w, ver, tables[d]), cost=COST_KR * 2)
            rets[key] = r
            contribs[key] = ct
            segs = ["학습", "검증", "테스트"] if ver == "장기" else ["검증", "테스트"]
            for s in segs + ["전체"]:
                rs = B.seg(r, s) if s != "전체" else r
                if s == "학습":
                    rs = rs[rs.index > dates[0]]
                a, t, beta = B.capm_alpha(rs, kospi, rf)
                p = B.perf(rs, rf)
                res.append(dict(판=ver, 중형=m, 가치=v, 비중=w, 구간=s, **p, 알파_vs코스피=a, t=t, 베타=beta,
                                회전율=turn, 평균종목수=nh, 샤프_비용2배=B.perf(B.seg(r2, s) if s != "전체" else r2, rf)["샤프"],
                                기준="벤치" if (m, v, w) in bench else "후보"))
    # 비교 지수
    eq_all = None
    for ver, y0 in [("장기", 2006), ("CRSP", 2014)]:
        st = S.april_ends(cal, y0, 2026)[0]
        k = kospi[kospi.index > st]
        for s in (["학습", "검증", "테스트", "전체"] if ver == "장기" else ["검증", "테스트", "전체"]):
            rs = B.seg(k, s) if s != "전체" else k
            res.append(dict(판=ver, 중형="코스피지수", 가치="", 비중="", 구간=s, **B.perf(rs, rf), 기준="벤치"))
        rets[f"{ver}|코스피지수"] = k
    df = save(pd.DataFrame(res), "d_kr_results")
    save(pd.DataFrame(info_rows), "d_kr_universe")
    pd.to_pickle(rets, OUT / "d_kr_returns.pkl")
    pd.to_pickle(contribs, OUT / "d_kr_contrib.pkl")

    # 선택: 장기판은 학습 샤프, CRSP판은 검증 샤프(학습 구간 없음)
    cand = df[df.기준 == "후보"]
    pick_l = cand[(cand.판 == "장기") & (cand.구간 == "학습")].sort_values("샤프", ascending=False).iloc[0]
    pick_c = cand[(cand.판 == "CRSP") & (cand.구간 == "검증")].sort_values("샤프", ascending=False).iloc[0]
    chosen = {"장기": f"장기|{pick_l.중형}|{pick_l.가치}|{pick_l.비중}", "CRSP": f"CRSP|{pick_c.중형}|{pick_c.가치}|{pick_c.비중}"}
    save(pd.Series(chosen, name="선택").to_frame(), "d_kr_chosen")

    # 연도별, 2025 제외, 기여 집중도
    yr = pd.DataFrame({k: B.yearly(v) for k, v in rets.items()})
    save(yr, "d_kr_yearly")
    ex = []
    for k, r in rets.items():
        for lab, rr in [("2025 제외", r[r.index.year != 2025]), ("2025·2026 제외", r[r.index.year < 2025])]:
            p = B.perf(rr, rf)
            a, t, _ = B.capm_alpha(rr, kospi, rf)
            ex.append(dict(포트=k, 조건=lab, CAGR=p["CAGR"], 샤프=p["샤프"], 알파=a, t=t))
    save(pd.DataFrame(ex), "d_kr_ex2025")
    conc = []
    for k in chosen.values():
        ct = contribs[k].copy()
        ct["year"] = pd.to_datetime(ct.start).dt.year
        tot = ct.groupby("code").contrib.sum().sort_values(ascending=False)
        top5 = tot.head(5)
        conc.append(dict(포트=k, 총기여합=tot.sum(), 상위5합=top5.sum(), 상위5=", ".join(
            f"{K['info'].name.get(c, c)}({v:+.2f})" for c, v in top5.items()), 종목수=len(tot)))
    save(pd.DataFrame(conc), "d_kr_concentration")
    return df, chosen, rets, kospi


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    R = DT.us_returns()
    ff = DT.ff6_daily()
    rf = ff["RF"].reindex(R.index).ffill()
    perf, yr, M = part_a(R, rf)
    print(perf[perf.구간.isin(["공통(2006-09~)", "테스트"])].round(3).to_string())
    reg = part_b(R, ff, M)
    print(reg[reg.구간 == "전체"].round(3).to_string())
    print(part_b_regime(R, M).round(3).to_string())
    c, chosen, _ = part_c(R, rf, M)
    print(chosen)
    print(c.round(3).to_string())
    d, kchosen, _, _ = part_d()
    print(kchosen)
    print(d.round(3).to_string())


if __name__ == "__main__":
    main()
