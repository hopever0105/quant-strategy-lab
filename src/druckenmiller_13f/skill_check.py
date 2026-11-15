# -*- coding: utf-8 -*-
"""8) 실제 수치 표 + '실력' 확인.

1. 매수 종목 vs S&P500: 특징의 실제 중앙값(백분위 말고 원래 단위)
   확신 매수(비중 3% 이상) / 작은 매수(1% 미만)도 따로
2. 실력 확인(짝 비교): 매수 종목마다 같은 분기 S&P500에서
   60일 변동성 5분위 × 12개월 수익률 3분위 × 시총 3분위가 같은 종목들을 '짝'으로 잡고
   이후 12개월 SPY 초과수익 차이(매수 - 짝 평균)를 본다. 0보다 크면 특징 말고 종목 고르는 실력.
3. 매도 종목도 같은 방식 — 판 게 이후 짝보다 못했으면 파는 실력.
결과: results/analysis/raw_medians.csv, skill_matched.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
AN = OUT / "analysis"

RAW = [("vol60", "60일 변동성(연율)", "pct"), ("beta", "베타", "x"), ("r12m", "직전 12개월 수익률", "pct"),
       ("r6m", "직전 6개월 수익률", "pct"), ("dist52h", "52주 고점 대비", "pct"), ("vs200", "200일선 대비", "pct"),
       ("mcap", "시가총액(십억$)", "b"), ("rev_g_ttm", "매출 성장(최근 4분기, 전년 대비)", "pct"),
       ("rev_g_q", "매출 성장(최근 분기, 전년 동기 대비)", "pct"), ("gross_margin", "매출총이익률", "pct"),
       ("op_margin", "영업이익률", "pct"), ("roe", "ROE", "pct"), ("rnd_sales", "R&D/매출", "pct"),
       ("ps", "PSR(배)", "x"), ("ey", "이익수익률(1/PER)", "pct"), ("debt_equity", "부채/자본(배)", "x"),
       ("profitable", "흑자 종목 비율", "mean")]


def fmt(v, k):
    if pd.isna(v):
        return "-"
    if k == "pct" or k == "mean":
        return f"{v * 100:.0f}%"
    if k == "b":
        return f"{v / 1e9:.0f}"
    return f"{v:.2f}"


def main():
    P = pd.read_pickle(OUT / "panel_ranked.pkl")
    E = P[(P["src"] == "드러켄밀러") & (P["leg"] == "주식") & (P["kind"] != "ETF")].copy()
    S = P[P["src"] == "S&P500"].copy()
    buy = E["action"].isin(["신규", "확대"])
    big = buy & (E["weight"] >= 0.03)
    small = buy & (E["weight"] < 0.01)
    rows = []
    for c, lb, k in RAW:
        f = (lambda s: s.mean()) if k == "mean" else (lambda s: s.median())
        rows.append({"특징": lb, "매수 전체": fmt(f(E.loc[buy, c]), k), "확신 매수": fmt(f(E.loc[big, c]), k),
                     "작은 매수": fmt(f(E.loc[small, c]), k), "S&P500": fmt(f(S[c]), k)})
    d = pd.DataFrame(rows)
    d.to_csv(AN / "raw_medians.csv", index=False, encoding="utf-8-sig")
    print(d.to_string(index=False))

    # ---- 짝 비교 ----
    for col, nq in [("vol60", 5), ("r12m", 3), ("mcap", 3)]:
        P[f"b_{col}"] = np.nan
        for p, g in P.groupby("period"):
            ref = g.loc[g["src"] == "S&P500", col].dropna()
            if len(ref) < 50:
                continue
            edges = np.quantile(ref, np.linspace(0, 1, nq + 1)[1:-1])
            v = g[col]
            P.loc[v.dropna().index, f"b_{col}"] = np.searchsorted(edges, v.dropna().values)
    key = ["period", "b_vol60", "b_r12m", "b_mcap"]
    S = P[P["src"] == "S&P500"]
    pool = S.dropna(subset=key + ["f12m_t1_x"]).groupby(key)["f12m_t1_x"].agg(["mean", "count"]).reset_index()
    E = P[(P["src"] == "드러켄밀러") & (P["leg"] == "주식") & (P["kind"] != "ETF")].copy()
    E = E.merge(pool, on=key, how="left")
    E["diff"] = E["f12m_t1_x"] - E["mean"]
    out = []
    print("\n== 짝 비교: 이후 12개월 SPY 초과수익, 그가 고른 종목 - 같은 분기·같은 변동성/모멘텀/시총 칸의 S&P500 평균 ==")
    groups = {"매수 전체": E["action"].isin(["신규", "확대"]), "신규": E["action"] == "신규", "확대": E["action"] == "확대",
              "확신 매수(3%+)": E["action"].isin(["신규", "확대"]) & (E["weight"] >= 0.03),
              "작은 매수(1%-)": E["action"].isin(["신규", "확대"]) & (E["weight"] < 0.01),
              "매도(축소·청산)": E["action"].isin(["축소", "청산"]), "계속 보유(유지)": E["action"] == "유지"}
    for nm, m in groups.items():
        x = E.loc[m & E["diff"].notna() & (E["count"] >= 3)]
        if len(x) < 10:
            continue
        q = x.groupby("period")["diff"].mean()
        t = stats.ttest_1samp(q, 0).statistic
        trim = x["diff"].clip(x["diff"].quantile(.05), x["diff"].quantile(.95))
        out.append(dict(구분=nm, 표본=len(x), 평균차이=x["diff"].mean(), 중앙차이=x["diff"].median(),
                        상하위5퍼_잘라낸평균=trim.mean(), 짝보다나은비율=(x["diff"] > 0).mean(), 분기t값=t))
    o = pd.DataFrame(out)
    o.to_csv(AN / "skill_matched.csv", index=False, encoding="utf-8-sig")
    for _, r in o.iterrows():
        print(f"  {r['구분']:<12} n={r['표본']:4d} 평균 {r['평균차이']:+.1%} 중앙 {r['중앙차이']:+.1%} "
              f"잘라낸평균 {r['상하위5퍼_잘라낸평균']:+.1%} 짝보다나음 {r['짝보다나은비율']:.0%} t={r['분기t값']:+.1f}")
    # 기간별 확신 매수
    x = E.loc[groups["확신 매수(3%+)"] & E["diff"].notna() & (E["count"] >= 3)]
    x = x.assign(half=np.where(pd.to_datetime(x["period"]) <= "2018-12-31", "2012~2018", "2019~2026"))
    print("  확신 매수 기간별:", x.groupby("half")["diff"].agg(["count", "mean", "median"]).round(3).to_dict("index"))
    y = E.loc[groups["매수 전체"] & E["diff"].notna() & (E["count"] >= 3)]
    y = y.assign(half=np.where(pd.to_datetime(y["period"]) <= "2018-12-31", "2012~2018", "2019~2026"))
    print("  매수 전체 기간별:", y.groupby("half")["diff"].agg(["count", "mean", "median"]).round(3).to_dict("index"))


if __name__ == "__main__":
    main()
