# -*- coding: utf-8 -*-
"""17) 조합 750개 + 머신러닝(워크포워드) — 미국 S&P500 / 코스피200 대용. 실제 매매 없음.

[조합] 분기말에 아래를 모두 만족하는 종목 동일가중, 보유 H분기(겹치는 묶음)
  ① 12개월 수익률 상위 M%  M∈{20,30,40,50,60}
  ② 종가 > N일 이동평균     N∈{60,120,200}
  ③ 매출 성장 상위 G%        G∈{10,20,30,40,50}
  ④ 베타 ≥ B                 B∈{없음,0.8,1.0,1.2,1.4}
  ⑤ 보유 H분기               H∈{1,2}                         → 750개
  '1등'은 학습 구간 샤프로만 고른다(평균 종목 수 하한 있음). 검증·테스트는 확인만.
  과적합 점검: 학습 순위와 검증·테스트 순위의 상관, 시장을 이긴 조합 비율, 1등의 무작위 대조 백분위.
[머신러닝] 특징 = 같은 분기 안 백분위(0~1). 목표 = 다음 분기 수익률의 같은 분기 안 백분위.
  Ridge(선형) / Gradient Boosting(깊이 3, 고정 설정) / 가짜(목표를 분기 안에서 섞어 학습한 GB)
  매년 1월 기준 그 전까지(결과가 확정된 행만)로 학습 → 그해 4개 분기 예측 → 예측 상위 10% 동일가중 1·2분기 보유
  순위 정보계수(IC: 예측 순위와 실제 수익 순위의 상관) 분기 평균·t값.
분기 수익률 근사(분기 중 비중 변동 무시), 비용 = 편도 비용 × 비중 변화 합. 미국 편도 0.1%, 코스피 0.25%.
구간: 미국 학습 2012~2018 / 검증 2019~2021 / 테스트 2022~ , 코스피 학습 2015~2019 / 검증 2020~2022 / 테스트 2023~ (+2025 제외)
결과: results/combo/{us,kr}_*.csv
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "combo"

CFG = {
    "us": dict(cost=0.001, seg={"학습": ("2011-12-31", "2018-12-31"), "검증": ("2019-01-01", "2021-12-31"),
                                "테스트": ("2022-01-01", "2026-12-31")}, min_n=5, ml_start=2015,
               feats=["r1m", "r3m", "r6m", "r12m", "vs60", "vs120", "vs200", "dist52h", "vol60", "beta", "mcap",
                      "sales_g", "op_margin", "roe", "ey", "bm", "gross_margin"], mkt="SPY"),
    "kr": dict(cost=0.0025, seg={"학습": ("2015-03-31", "2019-12-31"), "검증": ("2020-01-01", "2022-12-31"),
                                 "테스트": ("2023-01-01", "2026-12-31")}, min_n=3, ml_start=2017,
               feats=["r1m", "r3m", "r6m", "r12m", "vs60", "vs120", "vs200", "dist52h", "vol60", "beta", "mcap",
                      "sales_g", "op_margin", "roe", "ey", "bm"], mkt="코스피"),
}


def seg_of(t, seg):
    for s, (a, b) in seg.items():
        if pd.Timestamp(a) <= t <= pd.Timestamp(b):
            return s
    return None


def qsim(picks, H, R, mkt, cost):
    """picks: {t0: [codes]} → 분기 수익 Series(index t0), 평균 종목 수."""
    Q = sorted(picks)
    prev_w = pd.Series(dtype=float)
    out, ns = {}, {}
    for i, q in enumerate(Q):
        sleeves = [picks[Q[i - k]] for k in range(H) if i - k >= 0]
        w = pd.Series(dtype=float)
        rets = []
        for names in sleeves:
            if len(names) == 0:
                rets.append(mkt.get(q, 0.0))
                continue
            rr = R.loc[q].reindex(names).fillna(0.0) if q in R.index else pd.Series(0.0, index=names)
            rets.append(rr.mean())
            w = w.add(pd.Series(1.0 / len(names) / len(sleeves), index=names), fill_value=0)
        allk = w.index.union(prev_w.index)
        to = (w.reindex(allk, fill_value=0) - prev_w.reindex(allk, fill_value=0)).abs().sum()
        out[q] = np.mean(rets) - to * cost
        ns[q] = len(picks[q])
        prev_w = w
    return pd.Series(out), pd.Series(ns)


def qmetrics(r, mkt, seg, drop2025=False):
    res = {}
    segs = dict(seg)
    segs["전체"] = (min(a for a, _ in seg.values()), max(b for _, b in seg.values()))
    for s, (a, b) in segs.items():
        x = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))]
        if drop2025:
            x = x[x.index.year != 2025]
        if len(x) < 4:
            continue
        m = mkt.reindex(x.index)
        nav = (1 + x).cumprod()
        res[s] = dict(CAGR=nav.iloc[-1] ** (4 / len(x)) - 1, 샤프=x.mean() / x.std() * 2,
                      MDD=(nav / nav.cummax() - 1).min(), 시장초과=(x - m).mean() * 4, 이긴분기=(x > m).mean(),
                      시장CAGR=(1 + m).prod() ** (4 / len(x)) - 1, 시장샤프=m.mean() / m.std() * 2)
    return res


def ranks(P, cols):
    return P.groupby("t0")[cols].rank(pct=True)


def run_market(mk):
    cfg = CFG[mk]
    P = pd.read_pickle(OUT / f"qpanel_{mk}.pkl").reset_index(drop=True)
    R = pd.read_pickle(OUT / f"qret_{mk}.pkl")
    if mk == "us":
        R.index = pd.to_datetime(R.index)
    M_ = pd.read_pickle(OUT / f"qmkt_{mk}.pkl")
    M_.index = pd.to_datetime(M_.index)
    P = P[P["ret"].notna()]
    pr = ranks(P, ["r12m", "sales_g"]) * 100
    P["p_r12"], P["p_sg"] = pr["r12m"], pr["sales_g"]
    Q = sorted(P["t0"].unique())
    by = {q: P[P["t0"] == q] for q in Q}
    seg = cfg["seg"]
    # ---------- 조합 ----------
    rows = []
    series = {}
    for M, N, G, B, H in itertools.product([20, 30, 40, 50, 60], [60, 120, 200], [10, 20, 30, 40, 50],
                                           [None, 0.8, 1.0, 1.2, 1.4], [1, 2]):
        picks = {}
        for q in Q:
            g = by[q]
            m = (g["p_r12"] >= 100 - M) & (g[f"vs{N}"] > 0) & (g["p_sg"] >= 100 - G)
            if B is not None:
                m &= g["beta"] >= B
            picks[q] = g.loc[m.fillna(False), "code"].tolist()
        r, ns = qsim(picks, H, R, M_, cfg["cost"])
        key = (M, N, G, B, H)
        series[key] = (r, ns, picks)
        mt = qmetrics(r, M_, seg)
        for s, v in mt.items():
            a, b = (seg[s] if s in seg else (min(x for x, _ in seg.values()), max(y for _, y in seg.values())))
            nn = ns[(ns.index >= pd.Timestamp(a)) & (ns.index <= pd.Timestamp(b))].mean()
            rows.append(dict(상위M=M, 이평N=N, 매출상위G=G, 베타=("없음" if B is None else B), 보유H=H, 구간=s, 평균종목수=nn, **v))
    G_ = pd.DataFrame(rows)
    G_.to_csv(OUT / f"{mk}_grid.csv", index=False, encoding="utf-8-sig")
    keycols = ["상위M", "이평N", "매출상위G", "베타", "보유H"]
    piv = G_.pivot_table(index=keycols, columns="구간", values="샤프")
    nmin = G_[G_["구간"] == "학습"].set_index(keycols)["평균종목수"]
    ok = nmin[nmin >= cfg["min_n"]].index
    best_key = piv.loc[ok, "학습"].idxmax()
    bk = tuple(None if x == "없음" else x for x in best_key)
    mkt_sh = {s: v["시장샤프"] for s, v in qmetrics(M_, M_, seg).items()}
    diag = dict(조합수=len(piv), 종목수하한통과=len(ok))
    for s in ["학습", "검증", "테스트", "전체"]:
        diag[f"{s}_시장샤프"] = mkt_sh.get(s)
        diag[f"{s}_시장보다샤프높은조합비율"] = (piv[s] > mkt_sh.get(s, np.nan)).mean()
    sub = piv.loc[ok]
    diag["순위상관_학습vs검증"] = stats.spearmanr(sub["학습"], sub["검증"]).correlation
    diag["순위상관_학습vs테스트"] = stats.spearmanr(sub["학습"], sub["테스트"]).correlation
    diag["1등_학습순위"] = 1
    diag["1등_검증순위"] = int(sub["검증"].rank(ascending=False)[best_key])
    diag["1등_테스트순위"] = int(sub["테스트"].rank(ascending=False)[best_key])
    top10 = sub.sort_values("학습", ascending=False).head(10)
    top10.to_csv(OUT / f"{mk}_grid_top10.csv", encoding="utf-8-sig")
    # 조건별 평균 샤프(각 값을 고정했을 때 나머지 조합 평균) → 어느 값이 일관되게 좋은가
    marg = []
    for col in keycols:
        for v, x in G_[G_["구간"].isin(["학습", "검증", "테스트"])].groupby([col, "구간"]):
            marg.append(dict(조건=col, 값=v[0], 구간=v[1], 평균샤프=x["샤프"].mean(), 평균CAGR=x["CAGR"].mean()))
    MG = pd.DataFrame(marg).pivot_table(index=["조건", "값"], columns="구간", values="평균샤프")
    MG.to_csv(OUT / f"{mk}_grid_marginal.csv", encoding="utf-8-sig")
    # 1등 무작위 대조
    rb, nb, pb = series[bk]
    rng = np.random.default_rng(3)
    sims = []
    for _ in range(500):
        rp = {q: list(rng.choice(by[q]["code"].values, min(len(pb[q]), len(by[q])), replace=False)) if len(pb[q]) else [] for q in Q}
        sims.append(qsim(rp, bk[4], R, M_, cfg["cost"])[0])
    rc = {}
    for s, (a, b) in list(seg.items()) + [("전체", (min(x for x, _ in seg.values()), max(y for _, y in seg.values())))]:
        sel = lambda x: x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))]
        st = (1 + sel(rb)).prod()
        sm_ = np.array([(1 + sel(x)).prod() for x in sims])
        rc[s] = (sm_ < st).mean() * 100
    best = dict(시장=mk, 조합=str(best_key), **{f"무작위백분위_{s}": v for s, v in rc.items()}, **diag)
    pd.DataFrame([best]).to_csv(OUT / f"{mk}_grid_best.csv", index=False, encoding="utf-8-sig")
    bm = qmetrics(rb, M_, seg)
    out_rows = [dict(전략=f"조합 1등 {best_key}", 구간=s, **v, 평균종목수=None) for s, v in bm.items()]
    if mk == "kr":
        for s, v in qmetrics(rb, M_, seg, drop2025=True).items():
            out_rows.append(dict(전략=f"조합 1등 (2025 제외)", 구간=s, **v))

    # ---------- 머신러닝 ----------
    feats = cfg["feats"]
    X = ranks(P, feats).fillna(0.5)
    y = P.groupby("t0")["ret"].rank(pct=True)
    years = sorted({q.year for q in Q if q.year >= cfg["ml_start"]})
    preds = {nm: pd.Series(np.nan, index=P.index) for nm in ["Ridge", "GB", "가짜GB"]}
    imp = []
    rng = np.random.default_rng(1)
    for Y in years:
        cut = pd.Timestamp(f"{Y}-01-01")
        tr = (P["t0"] < cut) & (P["t1"] <= cut + pd.Timedelta(days=95))
        te = (P["t0"] >= cut) & (P["t0"] < pd.Timestamp(f"{Y + 1}-01-01"))
        if tr.sum() < 500 or te.sum() == 0:
            continue
        Xtr, ytr = X[tr], y[tr]
        rd = Ridge(alpha=1.0).fit(Xtr, ytr)
        preds["Ridge"][te] = rd.predict(X[te])
        gb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50,
                                           random_state=0).fit(Xtr, ytr)
        preds["GB"][te] = gb.predict(X[te])
        ysh = ytr.groupby(P.loc[tr, "t0"]).transform(lambda s: pd.Series(rng.permutation(s.values), index=s.index))
        gf = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50,
                                           random_state=0).fit(Xtr, ysh)
        preds["가짜GB"][te] = gf.predict(X[te])
        imp.append(dict(연도=Y, **{f"Ridge_{f}": c for f, c in zip(feats, rd.coef_)}))
    pd.DataFrame(imp).to_csv(OUT / f"{mk}_ml_ridge_coef.csv", index=False, encoding="utf-8-sig")
    ic_rows = []
    ml_series = {}
    for nm, pz in preds.items():
        d = P.assign(pred=pz).dropna(subset=["pred"])
        ic = d.groupby("t0").apply(lambda g: stats.spearmanr(g["pred"], g["ret"]).correlation)
        ic_rows.append(dict(모델=nm, IC평균=ic.mean(), IC_t=ic.mean() / ic.std() * np.sqrt(len(ic)), 양수비율=(ic > 0).mean(), 분기=len(ic)))
        picks = {q: g.sort_values("pred", ascending=False)["code"].head(max(1, int(round(len(g) * 0.1)))).tolist()
                 for q, g in d.groupby("t0")}
        for H in [1, 2]:
            r, _ = qsim(picks, H, R, M_, cfg["cost"])
            ml_series[(nm, H)] = r
            for s, v in qmetrics(r, M_, seg).items():
                out_rows.append(dict(전략=f"ML {nm} 상위10% 보유{H}분기", 구간=s, **v))
            if mk == "kr":
                for s, v in qmetrics(r, M_, seg, drop2025=True).items():
                    out_rows.append(dict(전략=f"ML {nm} 상위10% 보유{H}분기 (2025 제외)", 구간=s, **v))
    IC = pd.DataFrame(ic_rows)
    IC.to_csv(OUT / f"{mk}_ml_ic.csv", index=False, encoding="utf-8-sig")
    # 시장·동일가중
    ew = {q: by[q]["code"].tolist() for q in Q}
    rew, _ = qsim(ew, 1, R, M_, cfg["cost"])
    for s, v in qmetrics(rew, M_, seg).items():
        out_rows.append(dict(전략="대조: 유니버스 동일가중", 구간=s, **v))
    for s, v in qmetrics(M_[M_.index.isin(Q)], M_, seg).items():
        out_rows.append(dict(전략=f"시장({cfg['mkt']})", 구간=s, **v))
    O = pd.DataFrame(out_rows)
    O.to_csv(OUT / f"{mk}_summary.csv", index=False, encoding="utf-8-sig")
    pd.to_pickle(dict(best=series[bk], ml=ml_series, best_key=bk), OUT / f"{mk}_series.pkl")
    return best, O, IC, top10, MG


def main():
    pd.set_option("display.width", 260)
    for mk in ["us", "kr"]:
        print(f"\n################ {mk} ################", flush=True)
        best, O, IC, top10, MG = run_market(mk)
        print(pd.Series(best).to_string())
        print("\n학습 상위 10 조합(샤프)\n", top10.round(2).to_string())
        print("\n조건별 평균 샤프\n", MG.round(2).to_string())
        print("\nML IC\n", IC.round(3).to_string(index=False))
        print("\n요약\n", O[O["구간"].isin(["학습", "검증", "테스트", "전체"])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
