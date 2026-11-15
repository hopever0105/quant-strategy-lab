# -*- coding: utf-8 -*-
"""17) 그가 고른 종목 하나하나 vs '그때 똑같이 생긴데 안 고른 종목' 구분. 실제 매매 없음. KRX 요청 없음.

매수(신규·확대) 건마다, 같은 분기말(t0) S&P500에서
  - 같은 업종
  - 아래 8개 특징의 S&P500 백분위가 가장 가까운 5종목 (유클리드 거리, 빈칸은 50)
    12개월 수익률, 3개월 수익률, 60일 변동성, 베타, 시가총액, 매출 성장(4분기), 영업이익률, PSR
  - 그가 그 분기에 이미 보유했거나 산 종목은 제외
를 '닮은꼴'로 잡는다. (같은 업종 후보가 5개 미만이면 업종 무시)

1) 건별 표: 그가 고른 종목 vs 닮은꼴 5종목, 이후 6·12개월 SPY 초과수익, 차이(고른 것 − 닮은꼴 중앙값)
2) 집계: 신규/확대, 비중 크기, 연도, 업종별 '닮은꼴보다 나은 비율'·차이 중앙값
3) 닮은꼴과 '무엇이 달랐나': 맞추지 않은 특징(최근 분기 매출 성장, 성장 가속, 1개월 수익률, 52주 고점 거리,
   R&D 비중, 이익률, 밸류, 부채, 거래대금 등)을 짝별로 빼서 부호 검정 → 그가 보는 추가 정보 후보
4) 그 분기에 안 고른 닮은꼴을 그가 나중에 산 비율
결과: results/lookalike/*.csv, charts/lookalike_*.png
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "lookalike"
OUT.mkdir(parents=True, exist_ok=True)
CH = HERE / "charts"
CACHE = Path(r"C:\Users\arlo0\OneDrive\바탕 화면\취업\quant\cache\druck13f")

MATCH = ["pr_r12m", "pr_r3m", "pr_vol60", "pr_beta", "pr_mcap", "pr_rev_g_ttm", "pr_op_margin", "pr_ps"]
K = 5
# 맞추지 않은 특징(백분위) — 닮은꼴과 무엇이 달랐나
EXTRA = [("pr_rev_g_q", "최근 분기 매출 성장"), ("pr_rev_accel", "매출 성장 가속"), ("pr_op_g_ttm", "영업이익 성장"),
         ("pr_op_accel", "영업이익 성장 가속"), ("pr_ni_g_ttm", "순이익 성장"), ("pr_r1m", "1개월 수익률"),
         ("pr_r6m", "6개월 수익률"), ("pr_dist52h", "52주 고점 근접"), ("pr_vs200", "200일선 대비"),
         ("pr_ma50_200", "50일선/200일선"), ("pr_ivol", "고유 변동성"), ("pr_mdd1y", "1년 최대낙폭(클수록 덜 빠짐)"),
         ("pr_gross_margin", "매출총이익률"), ("pr_net_margin", "순이익률"), ("pr_fcf_margin", "FCF 마진"),
         ("pr_roe", "ROE"), ("pr_rnd_sales", "R&D/매출"), ("pr_ey", "이익수익률(싸다)"), ("pr_bm", "장부가/시총(싸다)"),
         ("pr_fcf_yield", "FCF 수익률(싸다)"), ("pr_debt_equity", "부채/자본"), ("pr_dollar_vol", "거래대금")]
OUTC = [("r_inq_x", "산 분기 안(SPY 초과)"), ("f6m_t1_x", "이후 6개월(SPY 초과)"), ("f12m_t1_x", "이후 12개월(SPY 초과)")]


def names():
    d = {}
    try:
        j = json.load(open(CACHE / "company_tickers.json", encoding="utf-8"))
        for v in j.values():
            d[v["ticker"].replace("-", ".")] = v["title"]
            d[v["ticker"]] = v["title"]
    except Exception:
        pass
    sec = pd.read_csv(RES / "securities.csv", dtype=str)
    for _, r in sec.dropna(subset=["yf"]).iterrows():
        d.setdefault(r["yf"], r.get("name", r["yf"]))
    return d


def main():
    P = pd.read_pickle(RES / "panel_ranked.pkl")
    NM = names()
    E = P[(P["src"] == "드러켄밀러") & (P["leg"] == "주식") & (P["kind"] != "ETF")
          & P["action"].isin(["신규", "확대"])].copy()
    S = P[P["src"] == "S&P500"].copy()
    # 그가 그 분기(period)에 들고 있던 종목 전부 — 닮은꼴에서 제외
    D = P[(P["src"] == "드러켄밀러") & (P["leg"] == "주식")]
    held = D[D["action"] != "청산"].groupby("t0")["yf"].apply(set).to_dict()
    bought_later = D[D["action"].isin(["신규", "확대"])].groupby("yf")["t0"].apply(sorted).to_dict()

    rows, pairs = [], []
    skipped = 0
    for t0, g in E.groupby("t0"):
        pool = S[(S["t0"] == t0) & ~S["yf"].isin(held.get(t0, set()))].copy()
        pool = pool[pool["pr_r12m"].notna() & pool["pr_vol60"].notna()]
        if len(pool) < 50:
            skipped += len(g)
            continue
        X = pool[MATCH].fillna(50).values
        for _, e in g.iterrows():
            if pd.isna(e["pr_r12m"]) or pd.isna(e["pr_vol60"]):
                skipped += 1
                continue
            same = (pool["sector"] == e["sector"]).values
            idx = np.where(same)[0] if same.sum() >= K else np.arange(len(pool))
            x = e[MATCH].astype(float).fillna(50).values
            dist = np.sqrt(((X[idx] - x) ** 2).sum(1))
            o = idx[np.argsort(dist)[:K]]
            pk = pool.iloc[o]
            r = dict(t0=t0, 분기=str(e["period"])[:10], 종목=e["yf"], 이름=e["name"], 업종=e["sector"],
                     구분=e["action"], 비중=e["weight"], 업종맞춤=bool(same.sum() >= K),
                     거리평균=float(np.sort(dist)[:K].mean()),
                     닮은꼴=", ".join(pk["yf"]), 닮은꼴이름=" / ".join(str(NM.get(t, t))[:28] for t in pk["yf"]))
            for c, _ in OUTC:
                r[f"{c}_고른것"] = e[c]
                r[f"{c}_닮은꼴중앙"] = pk[c].median()
                r[f"{c}_차이"] = e[c] - pk[c].median() if pd.notna(e[c]) and pk[c].notna().sum() >= 3 else np.nan
                r[f"{c}_닮은꼴1등"] = pk[c].max()
            # 고른 것이 닮은꼴 5개 중 몇 등(1=제일 좋음)
            if pd.notna(e["f6m_t1_x"]) and pk["f6m_t1_x"].notna().sum() >= 3:
                r["6개월순위(6개중)"] = int(1 + (pk["f6m_t1_x"].dropna() > e["f6m_t1_x"]).sum())
            # 안 고른 닮은꼴을 나중에 샀나
            later = [t for t in pk["yf"] if any(q > t0 for q in bought_later.get(t, []))]
            r["닮은꼴중_나중에산것"] = ", ".join(later)
            rows.append(r)
            for c, _ in EXTRA:
                pairs.append(dict(t0=t0, 종목=e["yf"], 구분=e["action"], 비중=e["weight"], 특징=c,
                                  차이=e[c] - pk[c].median() if pd.notna(e[c]) else np.nan))
    R = pd.DataFrame(rows)
    R["확신"] = np.where(R["비중"] >= 0.03, "확신(3%+)", np.where(R["비중"] < 0.01, "작은(1%-)", "중간"))
    R["연도"] = pd.to_datetime(R["분기"]).dt.year
    R["결과"] = np.where(R["f6m_t1_x_차이"].isna(), "결과 없음(최근)",
                       np.where(R["f6m_t1_x_차이"] > 0, "닮은꼴보다 나음", "닮은꼴보다 못함"))
    R.sort_values(["t0", "비중"], ascending=[True, False]).to_csv(OUT / "picks_vs_lookalikes.csv", index=False, encoding="utf-8-sig")
    print(f"매수 건 {len(E)} 중 비교 {len(R)} (제외 {skipped}), 업종까지 맞춘 비율 {R['업종맞춤'].mean():.0%}, "
          f"닮은꼴 평균 거리 {R['거리평균'].median():.1f} (백분위 8개 합성)")

    # ---- 2) 집계 ----
    def agg(df, key):
        out = []
        for k, x in df.groupby(key):
            for c, lb in OUTC:
                v = x[f"{c}_차이"].dropna()
                if len(v) < 8:
                    continue
                q = x.dropna(subset=[f"{c}_차이"]).groupby("t0")[f"{c}_차이"].mean()
                t = stats.ttest_1samp(q, 0).statistic if len(q) > 3 else np.nan
                out.append({key: k, "기간": lb, "건수": len(v), "나은비율": (v > 0).mean(), "차이중앙": v.median(),
                            "차이평균": v.mean(), "상하위5%자른평균": v.clip(v.quantile(.05), v.quantile(.95)).mean(),
                            "분기t": t})
        return pd.DataFrame(out)

    R["전체"] = "매수 전체"
    A = pd.concat([agg(R, "전체").rename(columns={"전체": "묶음"}),
                   agg(R, "구분").rename(columns={"구분": "묶음"}),
                   agg(R, "확신").rename(columns={"확신": "묶음"})], ignore_index=True)
    A.to_csv(OUT / "summary_groups.csv", index=False, encoding="utf-8-sig")
    Y = agg(R, "연도")
    Y.to_csv(OUT / "summary_by_year.csv", index=False, encoding="utf-8-sig")
    SEC = agg(R, "업종")
    SEC.to_csv(OUT / "summary_by_sector.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("\n== 묶음별: 고른 것 − 닮은꼴 5개 중앙값 ==")
    print(A.round(3).to_string(index=False))
    print("\n== 연도별(이후 6개월) ==")
    print(Y[Y["기간"].str.contains("6개월")].round(3).to_string(index=False))
    print("\n== 업종별(이후 6개월) ==")
    print(SEC[SEC["기간"].str.contains("6개월")].sort_values("건수", ascending=False).round(3).to_string(index=False))
    rk = R["6개월순위(6개중)"].dropna().value_counts(normalize=True).sort_index()
    print("\n6개 중 순위 분포(무작위면 각 17%):", rk.round(3).to_dict())

    # ---- 3) 무엇이 달랐나 ----
    PR = pd.DataFrame(pairs)
    out = []
    for grp, m in [("매수 전체", PR["구분"].notna()), ("확신(3%+)", PR["비중"] >= 0.03), ("신규", PR["구분"] == "신규"),
                   ("확대", PR["구분"] == "확대")]:
        for c, lb in EXTRA:
            x = PR[m & (PR["특징"] == c)].dropna(subset=["차이"])
            if len(x) < 20:
                continue
            q = x.groupby("t0")["차이"].mean()
            out.append(dict(묶음=grp, 특징=lb, 건수=len(x), 차이중앙=x["차이"].median(), 높은비율=(x["차이"] > 0).mean(),
                            부호검정p=stats.binomtest(int((x["차이"] > 0).sum()), int((x["차이"] != 0).sum())).pvalue,
                            분기t=stats.ttest_1samp(q, 0).statistic))
    F = pd.DataFrame(out)
    F.to_csv(OUT / "what_differs.csv", index=False, encoding="utf-8-sig")
    print("\n== 닮은꼴과 다른 점(백분위 차이, 맞추지 않은 특징) — 매수 전체·확신, |분기t|≥2 ==")
    print(F[F["분기t"].abs() >= 2].sort_values(["묶음", "분기t"]).round(3).to_string(index=False))

    # 다른 점이 성과와 이어지나: 매수 전체에서 차이 특징 상위 vs 하위 절반의 6개월 차이
    W = PR.pivot_table(index=["t0", "종목"], columns="특징", values="차이").reset_index()
    W = W.merge(R[["t0", "종목", "f6m_t1_x_차이"]], on=["t0", "종목"], how="left").dropna(subset=["f6m_t1_x_차이"])
    lk = []
    for c, lb in EXTRA:
        if c not in W:
            continue
        x = W.dropna(subset=[c])
        if len(x) < 40:
            continue
        hi, lo = x[x[c] > 0]["f6m_t1_x_차이"], x[x[c] <= 0]["f6m_t1_x_차이"]
        lk.append(dict(특징=lb, 닮은꼴보다높을때_6개월차이중앙=hi.median(), 낮을때=lo.median(), 높을때나은비율=(hi > 0).mean(),
                       낮을때나은비율=(lo > 0).mean(), 순위상관=stats.spearmanr(x[c], x["f6m_t1_x_차이"]).statistic, 건수=len(x)))
    LK = pd.DataFrame(lk).sort_values("순위상관", ascending=False)
    LK.to_csv(OUT / "differs_vs_outcome.csv", index=False, encoding="utf-8-sig")
    print("\n== 닮은꼴보다 그 특징이 높았던 매수가 결과도 좋았나 ==")
    print(LK.round(3).to_string(index=False))

    # ---- 4) 닮은꼴을 나중에 산 비율 ----
    later = (R["닮은꼴중_나중에산것"] != "").mean()
    print(f"\n안 고른 닮은꼴 5개 중 하나라도 나중에 산 비율: {later:.0%}")

    # 확신 매수 건별 표(보고서용)
    C = R[R["확신"] == "확신(3%+)"].sort_values("t0")
    cols = ["분기", "종목", "이름", "구분", "비중", "닮은꼴", "f6m_t1_x_고른것", "f6m_t1_x_닮은꼴중앙", "f6m_t1_x_차이", "결과"]
    C[cols].to_csv(OUT / "conviction_picks_table.csv", index=False, encoding="utf-8-sig")
    print(f"\n확신 매수 {len(C)}건 표 저장. 최근 5건:\n", C[cols].tail(5).round(3).to_string(index=False))

    # ---- 그림 ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for grp, col in [("확신(3%+)", "#1f3864"), ("중간", "#7f9cc7"), ("작은(1%-)", "#c9c9c9")]:
        v = R.loc[R["확신"] == grp, "f6m_t1_x_차이"].dropna().clip(-1, 1)
        ax[0].hist(v, bins=40, alpha=0.6, label=f"{grp} (n={len(v)}, 나음 {(v > 0).mean():.0%})", color=col, density=True)
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set_title("고른 것 − 닮은꼴 5개 중앙값 (이후 6개월, SPY 초과)")
    ax[0].legend(fontsize=8)
    y6 = Y[Y["기간"].str.contains("6개월")]
    ax[1].bar(y6["연도"].astype(str), y6["나은비율"] * 100, color="#1f3864")
    ax[1].axhline(50, color="gray", ls="--", lw=0.8)
    ax[1].set_title("연도별 '닮은꼴보다 나은' 비율(%)")
    plt.setp(ax[1].get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(CH / "lookalike_outcome.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
