# -*- coding: utf-8 -*-
"""7) 강건성 점검.

1. 기간 나누기: 2012~2018 vs 2019~2026 매수 특징이 같은가
2. 확신 매수(비중 3% 이상 신규·확대) vs 작은 매수(1% 미만)
3. 신규 매수 vs 매도(축소·청산) 직접 비교 — 무엇으로 갈리나
4. 겹치기 조합의 '이후 12개월 초과'를 기간별로 — 특정 시기 효과인지
5. 매수 후 성과에서 상위 5% 빼면
6. 비교군 생존편향: S&P500 비교군 중 시세 없는 비율(상장폐지 추정)
결과: results/analysis/robust_*.csv, results/robust.txt
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
sys.path.insert(0, str(HERE))
from analyze import FEATS  # noqa: E402

KEY = ["vol60", "beta", "ivol", "r12m", "r6m", "dist52h", "vs200", "rev_g_ttm", "rev_g_q", "rev_accel",
       "op_margin", "roe", "ey", "ps", "bm", "mcap", "rnd_sales", "mdd1y"]
LBL = dict(FEATS)


def main():
    P = pd.read_pickle(OUT / "panel_ranked.pkl")
    E = P[(P["src"] == "드러켄밀러") & (P["leg"] == "주식") & (P["kind"] != "ETF")].copy()
    S = P[P["src"] == "S&P500"].copy()
    buy = E["action"].isin(["신규", "확대"])
    L = []

    def say(s=""):
        print(s)
        L.append(str(s))

    # 1. 기간
    say("== 1. 기간 나누기 (매수 평균 백분위) ==")
    early = E["period"] <= "2018-12-31"
    rows = []
    for c in KEY:
        pc = f"pr_{c}"
        a = E.loc[buy & early, pc].dropna()
        b = E.loc[buy & ~early, pc].dropna()
        rows.append(dict(특징=LBL[c], 전반_2012_18=a.mean(), 후반_2019_26=b.mean(), n전=len(a), n후=len(b),
                         같은방향=np.sign(a.mean() - 50) == np.sign(b.mean() - 50)))
    d = pd.DataFrame(rows)
    d.to_csv(AN / "robust_period.csv", index=False, encoding="utf-8-sig")
    for _, r in d.iterrows():
        say(f"  {r['특징']:<14} 전반 {r['전반_2012_18']:5.1f} 후반 {r['후반_2019_26']:5.1f} {'' if r['같은방향'] else '← 방향 바뀜'}")

    # 2. 확신 매수
    say("\n== 2. 확신 매수(분기말 비중 3% 이상) vs 작은 매수(1% 미만) ==")
    big = buy & (E["weight"] >= 0.03)
    small = buy & (E["weight"] < 0.01)
    say(f"  확신 {big.sum()}건, 작은 {small.sum()}건")
    rows = []
    for c in KEY:
        pc = f"pr_{c}"
        a = E.loc[big, pc].dropna()
        b = E.loc[small, pc].dropna()
        t = stats.ttest_ind(a, b, equal_var=False).statistic if len(a) > 10 and len(b) > 10 else np.nan
        rows.append(dict(특징=LBL[c], 확신=a.mean(), 작은=b.mean(), t값=t))
    d = pd.DataFrame(rows).sort_values("t값", key=lambda s: -s.abs())
    d.to_csv(AN / "robust_conviction.csv", index=False, encoding="utf-8-sig")
    for _, r in d.iterrows():
        say(f"  {r['특징']:<14} 확신 {r['확신']:5.1f} 작은 {r['작은']:5.1f} (t={r['t값']:+.1f})")
    for nm, m in [("확신", big), ("작은", small)]:
        x = E.loc[m, "f12m_t1_x"].dropna()
        say(f"  {nm} 매수 12개월 SPY 초과: 평균 {x.mean():+.1%} 중앙 {x.median():+.1%} 승률 {(x > 0).mean():.0%} n={len(x)}")
    # 확신 매수 목록
    bl = E.loc[big, ["period", "name", "action", "weight", "sector", "f12m_t1_x"]].sort_values("period")
    bl.to_csv(AN / "conviction_buys.csv", index=False, encoding="utf-8-sig")

    # 3. 신규 vs 매도
    say("\n== 3. 신규 매수 vs 매도(축소·청산) ==")
    new = E["action"] == "신규"
    sell = E["action"].isin(["축소", "청산"])
    rows = []
    for c in KEY:
        pc = f"pr_{c}"
        a = E.loc[new, pc].dropna()
        b = E.loc[sell, pc].dropna()
        t = stats.ttest_ind(a, b, equal_var=False).statistic
        rows.append(dict(특징=LBL[c], 신규=a.mean(), 매도=b.mean(), t값=t))
    d = pd.DataFrame(rows).sort_values("t값", key=lambda s: -s.abs())
    d.to_csv(AN / "robust_new_vs_sell.csv", index=False, encoding="utf-8-sig")
    for _, r in d.iterrows():
        say(f"  {r['특징']:<14} 신규 {r['신규']:5.1f} 매도 {r['매도']:5.1f} (t={r['t값']:+.1f})")
    # 매도 직전 분기(보유 중) 수익률: 오른 걸 파는가 떨어진 걸 파는가
    x = E.loc[E["action"] == "청산", "r_inq_x"].dropna()
    y = E.loc[E["action"] == "신규", "r_inq_x"].dropna()
    say(f"  분기 중 SPY 초과수익: 청산한 분기 평균 {x.mean():+.1%}(중앙 {x.median():+.1%}), 신규로 산 분기 {y.mean():+.1%}(중앙 {y.median():+.1%})")

    # 4. 조합 기간별
    say("\n== 4. 겹치기 조합(시총 상위 30% + 베타>1.2 + 변동성 상위 30%) 기간별 이후 12개월 SPY 초과 ==")
    m = (S["pr_mcap"] > 70) & (S["beta"] > 1.2) & (S["pr_vol60"] > 70)
    S["yr"] = pd.to_datetime(S["period"]).dt.year
    g = S[m].groupby("yr")["f12m_t1_x"].agg(["count", "mean"])
    g2 = S[~m].groupby("yr")["f12m_t1_x"].mean()
    g["나머지"] = g2
    g["차이"] = g["mean"] - g["나머지"]
    g.to_csv(AN / "robust_combo_by_year.csv", encoding="utf-8-sig")
    for y, r in g.iterrows():
        if r["count"] > 0 and np.isfinite(r["mean"]):
            say(f"  {y}: 조합 {r['mean']:+.1%} (n={int(r['count'])}) 나머지 {r['나머지']:+.1%} 차이 {r['차이']:+.1%}")
    pos = (g["차이"] > 0).sum()
    say(f"  조합이 나머지를 이긴 해 {pos}/{g['차이'].notna().sum()}")

    # 5. 상위 5% 제외
    say("\n== 5. 매수 후 12개월 SPY 초과에서 대박 종목 제외 ==")
    x = E.loc[buy, "f12m_t1_x"].dropna()
    for q in [1.0, 0.99, 0.95, 0.90]:
        y = x[x <= x.quantile(q)]
        say(f"  상위 {100 - q * 100:.0f}% 제외: 평균 {y.mean():+.1%} (n={len(y)})")
    top = E.loc[buy].sort_values("f12m_t1_x", ascending=False)[["period", "name", "weight", "f12m_t1_x"]].head(15)
    top.to_csv(AN / "top_winners.csv", index=False, encoding="utf-8-sig")
    say("  상위 10: " + ", ".join(f"{r['name']}({r['period'].year}, {r['f12m_t1_x']:+.0%})" for _, r in top.head(10).iterrows()))

    # 6. 생존편향
    say("\n== 6. 비교군 생존편향 ==")
    miss = S["r6m"].isna()
    say(f"  S&P500 비교군 중 시세 없는 행 {miss.mean():.0%} (대부분 이후 상장폐지·인수된 종목 추정)")
    yr = S.groupby("yr")["r6m"].apply(lambda s: s.isna().mean())
    say("  연도별: " + ", ".join(f"{y} {v:.0%}" for y, v in yr.items() if y % 2 == 0))
    em = E.loc[buy, "r6m"].isna()
    say(f"  드러켄밀러 매수 중 시세 없는 비율 {em.mean():.0%}")
    (OUT / "robust.txt").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
