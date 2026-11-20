# -*- coding: utf-8 -*-
"""약점 보완 분석 (규칙·구간·후보 격자는 run.py와 같음). run.py 실행 뒤에 실행.

  python robust.py candidates : 법인세 납부액을 받을 후보 (종목, 연도) 목록 → quant/cache/larry/dart/tax_candidates.csv
  python robust.py            : 보완 분석 전체 → results/robust_*.csv, summary_robust.json, fig8~10

보완 항목
 1) 검정력: t=2를 보려면 몇 년이 필요한가 (테스트 구간 길이 문제)
 2) 블록 부트스트랩(12개월 블록, 5,000회): 초과수익·샤프 차이의 95% 구간, 0보다 클 확률
 3) 2025~2026 제외 구간(2020-01~2024-12) - 결과를 본 뒤 정한 구간이라 참고로만
 4) ③ 리밸런싱 달 분산: 1~12월 각각 시작한 12개 포트폴리오 평균(4월 한 번에 따른 우연 제거, 판단 횟수 12배)
 5) ③ 워크포워드: 매년 그 전 데이터만으로 12개 후보 중 다시 선택
 6) ③ 한 해씩 빼기: 특정 해(예: 2016) 의존도
 7) ③ 섹터 분석: 섹터 비중 차이(배분 효과) vs 같은 섹터 안 종목 선택 효과, 섹터별 적중률
 8) ③ 업종 분류를 한 가지(kor_sector)로 통일했을 때
 9) COT(S&P500) 신호를 코스피에 위험 관리용으로 적용했을 때
"""
import io
import json
import sys
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
import quality as QM      # noqa: E402

OUT = HERE / "results"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
SEG3 = {"학습": ("2014-05-01", "2017-12-31"), "검증": ("2018-01-01", "2019-12-31"), "테스트": ("2020-01-01", "2026-12-31")}
RS = {}


def P(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 후보 목록 (법인세 납부액 수집용)
def candidates():
    W = D.panel_wide()
    info = D.listing_info()
    Q = QM.Quality(W, info)
    dates = sorted(set(d for m in range(1, 13) for d in Q.month_ends(m)))
    dates = [d for d in dates if 2015 <= QM.fy_for(d) <= 2022]
    mcs = Q.mcaps(dates)
    apr = [d for d in Q.month_ends(4) if d.year <= 2017]
    Q.set_thresholds(pd.concat([Q.universe(d, Q.mcaps([d])[d], tax_mode="proxy") for d in apr]))
    rows = set()
    for d in dates:
        u = Q.universe(d, mcs[d], tax_mode="expense_only")
        for sc in QM.SCEN:
            for c in Q.pick(u, sc, n=60):
                rows.add((c, QM.fy_for(d)))
    df = pd.DataFrame(sorted(rows), columns=["code", "fy"])
    df.to_csv(D.DART_DIR / "tax_candidates.csv", index=False, encoding="utf-8-sig")
    P("후보", len(df), df.groupby("fy").size().to_dict())


# ---------------------------------------------------------------- 공통 도구
def monthly(r):
    return (1 + r.dropna()).resample("ME").prod() - 1


def boot(x_list, fn, n=5000, block=12, seed=0):
    """월간 수익률 묶음(같은 날짜)을 12개월 블록으로 다시 뽑아 fn 값의 분포를 구함"""
    rng = np.random.default_rng(seed)
    X = pd.concat(x_list, axis=1).dropna().values
    T = len(X)
    nb = int(np.ceil(T / block))
    vals = []
    for _ in range(n):
        st = rng.integers(0, T - block + 1, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:T]
        vals.append(fn(X[idx]))
    v = np.array(vals)
    return np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5), float(np.nanmean(v > 0))


def ann_excess(X):
    return (X[:, 0] - X[:, 1]).mean() * 12


def sharpe_diff(X):
    s = lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else np.nan
    return s(X[:, 0] - X[:, 2]) - s(X[:, 1] - X[:, 2])


def years_needed(ex_m, bm_ex=None):
    """월간 초과수익 → t=2에 필요한 연수 = (2×연 잔차변동 ÷ 연 알파)².
    bm_ex(벤치마크 초과수익)를 주면 시장 민감도(베타)를 뺀 알파·잔차로 계산(현금을 오래 들고 있는 시기 전략용)"""
    if bm_ex is not None:
        df = pd.concat([ex_m, bm_ex], axis=1).dropna()
        X = np.column_stack([np.ones(len(df)), df.iloc[:, 1].values])
        coef, *_ = np.linalg.lstsq(X, df.iloc[:, 0].values, rcond=None)
        resid = df.iloc[:, 0].values - X @ coef
        a, te, n = coef[0] * 12, resid.std(ddof=2) * np.sqrt(12), len(df)
    else:
        a, te, n = ex_m.mean() * 12, ex_m.std() * np.sqrt(12), len(ex_m)
    t = a / te * np.sqrt(n / 12)
    return a, te, t, (2 * te / a) ** 2 if a > 0 else np.inf


def perf_row(name, seg, r, rf, b=None):
    p = B.perf(r, rf)
    row = dict(전략=name, 구간=seg, **p)
    if b is not None:
        pb = B.perf(b.loc[r.index[0]:r.index[-1]], rf)
        al, t, _ = B.alpha_t(r, b, rf)
        row.update(BM_CAGR=pb["CAGR"], BM_샤프=pb["샤프"], BM_MDD=pb["MDD"], 알파=al, 알파t=t)
    return row


# ---------------------------------------------------------------- 본 분석
def main():
    dly = pd.read_pickle(OUT / "daily.pkl")
    m3 = pd.read_pickle(OUT / "m3_series.pkl")
    S0 = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    r_kos, rf_kr, spy, rf_us = dly["r_kos"], dly["rf_kr"], dly["spy"], dly["rf_us"]
    r3, r_ew, r_kos3, rf3, best3 = m3["r3"], m3["r_ew"], m3["r_kos3"], m3["rf3"], m3["best3"]
    kos = dly["kos"]

    # TDM 단독 시기 전략(첫 3거래일 + 월말, 코스피 보유, 나머지 현금) - 효과 크기 확인용
    tdm_pos = S.tdm_window(kos.index, 3).astype(float).shift(-1).fillna(0.0)   # 다음 날이 TDM이면 오늘 종가에 보유
    r_tdm, _, _ = B.timing(tdm_pos, r_kos, rf_kr)

    # 9) COT(S&P500) → 코스피 위험 관리
    n, lo, hi = S0["cot_choice"]["SPX"]
    ucal = spy.index
    cot_kr = S.to_kr(S.cot_state(S.cot_daily(D.cot("13874A"), n, ucal), lo, hi), kos.index).fillna(0.0)
    r_cot_kos, ntr, _ = B.timing(cot_kr, r_kos, rf_kr)

    strategies = {
        "COT S&P500 → SPY": (dly["cot_spx"], spy, rf_us),
        "COT S&P500 → 코스피(추가)": (r_cot_kos, r_kos, rf_kr),
        "패닉 매수 → 코스피": (dly["panic"], r_kos, rf_kr),
        "TDM(월말+첫 3일)만 코스피 보유": (r_tdm, r_kos, rf_kr),
        f"③ 우량주({best3[0]}·{best3[1]}) vs 코스피": (r3, r_kos3, rf3),
        f"③ 우량주({best3[0]}·{best3[1]}) vs 동일가중": (r3, r_ew, rf3),
    }

    # 1) 검정력 + 2) 부트스트랩 + 3) 2020~2024
    rows_p, rows_b, rows_x = [], [], []
    for nm, (r, b, rf) in strategies.items():
        rm, bm, fm = monthly(r), monthly(b), monthly(rf.reindex(r.index).fillna(0) if rf is not None else r * 0)
        df = pd.concat([rm, bm, fm], axis=1).dropna()
        a, te, t, yn = years_needed(df.iloc[:, 0] - df.iloc[:, 2], df.iloc[:, 1] - df.iloc[:, 2])
        test_yrs = len(df.loc["2020":]) / 12
        rows_p.append(dict(전략=nm, 기간=f"{df.index[0]:%Y-%m}~{df.index[-1]:%Y-%m}", 연알파=a, 연잔차변동=te, 전체t=t,
                           t2필요연수=yn, 테스트연수=test_yrs, 테스트기대t=a / te * np.sqrt(test_yrs) if te > 0 else np.nan))
        lo1, hi1, p1 = boot([df.iloc[:, 0], df.iloc[:, 1]], ann_excess)
        lo2, hi2, p2 = boot([df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]], sharpe_diff)
        mdd = lambda x: ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min()
        rows_b.append(dict(전략=nm, 연초과수익=ann_excess(df.values), 초과수익_하한=lo1, 초과수익_상한=hi1, 초과수익_양수확률=p1,
                           샤프차이=sharpe_diff(df.values), 샤프차이_하한=lo2, 샤프차이_상한=hi2, 샤프차이_양수확률=p2,
                           MDD=mdd(df.iloc[:, 0]), BM_MDD=mdd(df.iloc[:, 1])))
        for seg, (a_, b_) in {"테스트 2020~2026": ("2020-01-01", "2026-12-31"), "2020~2024(2025·26 제외, 참고)": ("2020-01-01", "2024-12-31"),
                              "2025~2026": ("2025-01-01", "2026-12-31"), "전체": ("2000-01-01", "2026-12-31")}.items():
            rows_x.append(perf_row(nm, seg, r.loc[a_:b_], rf, b))
    pd.DataFrame(rows_p).to_csv(OUT / "robust_power.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_b).to_csv(OUT / "robust_bootstrap.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_x).to_csv(OUT / "robust_ex2526.csv", index=False, encoding="utf-8-sig")
    P("검정력·부트스트랩 완료")

    # ③ 재구성
    W = D.panel_wide()
    info = D.listing_info()
    Q = QM.Quality(W, info)
    cal = Q.C.index
    end = cal[-1]
    apr = Q.month_ends(4)
    Uapr = {d: Q.universe(d, Q.mcaps([d])[d]) for d in apr}
    Q.set_thresholds(pd.concat([Uapr[d] for d in apr if d.year <= 2017]))
    sc, en = best3

    # 4) 리밸런싱 달 분산
    tr_s, tr_e, tr_rows = [], [], []
    for m in range(1, 13):
        ds = Q.month_ends(m)
        mcs = Q.mcaps(ds)
        U = {d: Q.universe(d, mcs[d]) for d in ds}
        r, _, hold, _ = Q.run(ds, end, sc, en, rf3, U=U)
        re_, _, _, _ = B.stock_portfolio(ds + [end], {d: U[d].index.tolist() for d in ds}, None, Q.R, rf3)
        tr_s.append(r.rename(m)); tr_e.append(re_.rename(m))
        for seg, (a_, b_) in SEG3.items():
            tr_rows.append(dict(시작월=m, 구간=seg, **B.perf(r.loc[a_:b_], rf3), 평균종목수=hold))
    TS, TE = pd.concat(tr_s, axis=1), pd.concat(tr_e, axis=1)
    start = TS.apply(lambda c: c.first_valid_index()).max()
    r_stag, r_ew_stag = TS.loc[start:].mean(axis=1), TE.loc[start:].mean(axis=1)
    stag = []
    for seg, (a_, b_) in list(SEG3.items()) + [("2020~2024(참고)", ("2020-01-01", "2024-12-31")), ("전체", ("2014-01-01", "2026-12-31"))]:
        a_ = max(pd.Timestamp(a_), start)
        stag.append(perf_row("달 분산 12개 평균", seg, r_stag.loc[a_:b_], rf3, r_kos3))
        stag.append(perf_row("달 분산 12개 평균 vs 동일가중(달 분산)", seg, r_stag.loc[a_:b_], rf3, r_ew_stag))
        stag.append(perf_row("4월 한 번(본 결과)", seg, r3.loc[a_:b_], rf3, r_kos3))
    pd.DataFrame(stag).to_csv(OUT / "robust_stagger.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tr_rows).to_csv(OUT / "robust_stagger_tranches.csv", index=False, encoding="utf-8-sig")
    fm = monthly(rf3.loc[start:])
    ms, me, mk = monthly(r_stag), monthly(r_ew_stag), monthly(r_kos3.loc[start:])
    RS["stagger_vs_ew"] = dict(zip(["연알파", "잔차변동", "t", "필요연수"], map(float, years_needed(ms - fm, me - fm))))
    lo1, hi1, p1 = boot([ms, me], ann_excess)
    RS["stagger_vs_ew_boot"] = dict(하한=lo1, 상한=hi1, 양수확률=p1)
    RS["stagger_vs_kospi"] = dict(zip(["연알파", "잔차변동", "t", "필요연수"], map(float, years_needed(ms - fm, mk - fm))))
    lo1, hi1, p1 = boot([ms, mk], ann_excess)
    RS["stagger_vs_kospi_boot"] = dict(하한=lo1, 상한=hi1, 양수확률=p1)
    RS["stagger_start"] = str(start.date())
    RS["stagger_month_test_sharpe"] = {int(r_["시작월"]): r_["샤프"] for r_ in tr_rows if r_["구간"] == "테스트"}
    P("달 분산 완료")

    # 5) 워크포워드 (12개 후보, 매년 재선택)
    cands = m3["cands3"]
    parts, hist = [], []
    for y in range(2016, 2027):
        best, bk = -np.inf, None
        for k_, r_ in cands.items():
            s_ = B.perf(r_.loc["2014-05-01":f"{y - 1}-12-31"], rf3)["샤프"]
            if np.isfinite(s_) and s_ > best:
                best, bk = s_, k_
        parts.append(cands[bk].loc[f"{y}-01-01":f"{y}-12-31"])
        hist.append(f"{y}:{bk[0]}·{bk[1]}")
    r_wf = pd.concat(parts)
    wf_rows = [perf_row("③ 워크포워드", "2016~2026", r_wf, rf3, r_kos3), perf_row("③ 워크포워드 vs 동일가중", "2016~2026", r_wf, rf3, r_ew)]
    wf_rows[0]["선택이력"] = "; ".join(hist)
    pd.DataFrame(wf_rows).to_csv(OUT / "robust_m3_walkforward.csv", index=False, encoding="utf-8-sig")

    # 6) 한 해씩 빼기
    yr = B.yearly(r3)
    yk = B.yearly(r_kos3.loc[r3.index[0]:])
    ye = B.yearly(r_ew)
    loo = []
    for y in yr.index:
        k_ = yr.drop(y)
        g = lambda s: (1 + s).prod() ** (1 / len(s)) - 1
        loo.append(dict(뺀해=y, 그해전략=yr[y], 전략연평균=g(k_), 코스피연평균=g(yk.drop(y)), 동일가중연평균=g(ye.drop(y))))
    pd.DataFrame(loo).to_csv(OUT / "robust_m3_leave_one_year.csv", index=False, encoding="utf-8-sig")

    # 7) 섹터 분석 (즉시 진입 기준: 섹터 효과를 진입 시점과 섞지 않기 위해)
    picks = m3["picks"]
    rb = list(picks.keys())
    att, secrows = [], []
    for a_, b_ in zip(rb, rb[1:] + [end]):
        u = Uapr[a_]
        days = Q.R.index[(Q.R.index > a_) & (Q.R.index <= b_)]
        pr = (1 + Q.R.loc[days, u.index].fillna(0).astype(float)).prod() - 1
        u = u.assign(ret=pr.reindex(u.index))
        u["sector"] = u["sector"].fillna("미상")
        sel = u.loc[[c for c in picks[a_] if c in u.index]]
        if not len(sel):
            continue
        sec_ret_u = u.groupby("sector").ret.mean()
        w_sel = sel.sector.value_counts(normalize=True)
        w_u = u.sector.value_counts(normalize=True)
        r_sel = sel.ret.mean()
        r_match = (w_sel * sec_ret_u.reindex(w_sel.index)).sum()
        r_all = u.ret.mean()
        att.append(dict(시작=a_.date(), 종목수=len(sel), 전략=r_sel, 같은섹터비중벤치=r_match, 동일가중=r_all,
                        선택효과=r_sel - r_match, 배분효과=r_match - r_all,
                        최대섹터=w_sel.index[0], 최대섹터비중=w_sel.iloc[0]))
        for s_, g in sel.groupby("sector"):
            secrows.append(dict(시작=a_.date(), 섹터=s_, 종목수=len(g), 선택종목수익=g.ret.mean(), 섹터전체수익=sec_ret_u[s_],
                                비중_전략=w_sel[s_], 비중_전체=w_u.get(s_, 0)))
    AT, SR = pd.DataFrame(att), pd.DataFrame(secrows)
    AT.to_csv(OUT / "robust_m3_sector_attr.csv", index=False, encoding="utf-8-sig")
    SR["초과"] = SR.선택종목수익 - SR.섹터전체수익
    bysec = SR.groupby("섹터").agg(등장횟수=("종목수", "size"), 종목수합=("종목수", "sum"), 평균초과=("초과", "mean"),
                                   적중률=("초과", lambda x: (x > 0).mean()), 평균비중_전략=("비중_전략", "mean"),
                                   평균비중_전체=("비중_전체", "mean")).sort_values("종목수합", ascending=False)
    bysec.to_csv(OUT / "robust_m3_by_sector.csv", encoding="utf-8-sig")
    tt = lambda x: x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan
    RS["sector_attr"] = dict(기간수=len(AT), 선택효과평균=float(AT.선택효과.mean()), 선택효과t=float(tt(AT.선택효과)),
                             선택효과양수비율=float((AT.선택효과 > 0).mean()), 배분효과평균=float(AT.배분효과.mean()),
                             배분효과t=float(tt(AT.배분효과)), 섹터적중률=float((SR.초과 > 0).mean()), 섹터관측수=int(len(SR)))
    P("섹터 완료")

    # 8) 업종 분류 통일(kor_sector)
    ks = D._sql("SELECT CMP_CD code, SEC_NM_KOR sec FROM kor_sector ORDER BY `기준일`").drop_duplicates("code", keep="last").set_index("code")
    info2 = info.copy()
    info2["sector_ks"] = ks.sec.reindex(info2.index)
    Q2 = QM.Quality(W, info2, sector_col="sector_ks")
    Q2.qs = Q.qs
    U2 = {d: Q2.universe(d, Q.mcaps([d])[d]) for d in apr}
    rows8 = []
    for en_ in QM.ENTRY:
        r8, _, h8, _ = Q2.run(apr, end, "업종상대", en_, rf3, U=U2)
        for seg, (a_, b_) in SEG3.items():
            rows8.append(dict(전략=f"업종상대(kor_sector 통일)·{en_}", 구간=seg, **B.perf(r8.loc[a_:b_], rf3), 평균종목수=h8))
    rows8.append(dict(전략="kor_sector 업종 있는 종목 수(2026-04)", 구간="", 일수=int(U2[apr[-1]].sector.notna().sum())))
    pd.DataFrame(rows8).to_csv(OUT / "robust_m3_sector_ks.csv", index=False, encoding="utf-8-sig")

    (OUT / "summary_robust.json").write_text(json.dumps(RS, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    # 차트
    fig, ax = plt.subplots(figsize=(11, 4.3))
    for m in range(1, 13):
        ax.plot((1 + TS[m].loc[start:].fillna(0)).cumprod(), color="#9dc3e6", lw=0.6)
    ax.plot((1 + r_stag).cumprod(), color="#1f4e79", lw=1.8, label="시작월 12개 평균")
    ax.plot((1 + r_ew_stag).cumprod(), color="#c55a11", lw=1.2, label="동일가중 전 종목(달 분산)")
    ax.plot((1 + r_kos3.loc[start:]).cumprod(), color="#7f7f7f", lw=1.0, label="코스피")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_title("③ 리밸런싱 시작월을 1~12월로 바꾼 12개 포트폴리오(옅은 선)와 평균")
    fig.tight_layout(); fig.savefig(OUT / "fig8_stagger.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(AT))
    ax.bar(x - 0.2, AT.선택효과 * 100, 0.4, label="같은 섹터 안 종목 선택 효과", color="#1f4e79")
    ax.bar(x + 0.2, AT.배분효과 * 100, 0.4, label="섹터 비중 차이(배분) 효과", color="#ffc000")
    ax.set_xticks(x); ax.set_xticklabels([str(d)[:4] for d in AT.시작], fontsize=8); ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("%p (1년 보유)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y"); ax.set_title("③ 초과수익 분해: 종목 선택 vs 섹터 배분")
    fig.tight_layout(); fig.savefig(OUT / "fig9_sector_attr.png", dpi=160); plt.close(fig)
    P("완료")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "candidates":
        candidates()
    else:
        main()
