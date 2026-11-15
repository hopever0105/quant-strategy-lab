# -*- coding: utf-8 -*-
"""14) 그가 '비중을 늘린(확대)' 종목의 공통점 → 그 없이 쓸 수 있는 규칙.

A. 그의 보유 종목 안에서: 확대 vs 유지 vs 축소·청산 — 늘리기 직전(t0) 무엇이 달랐나 (같은 분기 S&P500 백분위)
   + 로지스틱 회귀(확대=1, 유지·축소·청산=0)로 서로 겹치는 정보 정리
B. 확대 중 성공(12개월 SPY 초과 상위 1/3) vs 실패(하위 1/3)
C. 확대 종목 vs S&P500 실제 수치(중앙값)
결과: results/adds/*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "adds"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
from analyze import FEATS  # noqa: E402

LBL = dict(FEATS)
KEY = ["r1m", "r3m", "r6m", "r12m", "dist52h", "vs200", "ma50_200", "vol60", "beta", "mcap",
       "rev_g_ttm", "rev_g_q", "rev_accel", "op_g_ttm", "gross_margin", "op_margin", "roe", "rnd_sales",
       "ps", "ey", "bm", "fcf_yield", "debt_equity", "mdd1y"]


def main():
    P = pd.read_pickle(HERE / "results" / "panel_ranked.pkl")
    bad = (P["mcap"] > 5e12) | (P["mcap"] < 1e8)
    P.loc[bad, "pr_mcap"] = np.nan
    E = P[(P["src"] == "드러켄밀러") & (P["leg"] == "주식") & (P["kind"] != "ETF")].copy()
    S = P[P["src"] == "S&P500"]
    add = E["action"] == "확대"
    hold = E["action"] == "유지"
    cut = E["action"].isin(["축소", "청산"])

    # A. 보유 종목 안 비교
    rows = []
    for c in KEY:
        pc = f"pr_{c}"
        a, h, s = E.loc[add, pc].dropna(), E.loc[hold, pc].dropna(), E.loc[cut, pc].dropna()
        t_ah = stats.ttest_ind(a, h, equal_var=False).statistic
        t_as = stats.ttest_ind(a, s, equal_var=False).statistic
        q = E.loc[add & E[pc].notna()].groupby("period")[pc].mean()
        rows.append(dict(특징=LBL[c], 코드=c, 확대=a.mean(), 유지=h.mean(), 축소청산=s.mean(),
                         t_확대vs유지=t_ah, t_확대vs축소=t_as, t_확대vsSP500=stats.ttest_1samp(q, 50).statistic, n확대=len(a)))
    A = pd.DataFrame(rows).sort_values("t_확대vs축소", key=lambda x: -x.abs())
    A.to_csv(OUT / "add_vs_hold_cut.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("== A. 그가 들고 있던 종목 안에서: 늘린 것 vs 그대로 둔 것 vs 줄인·판 것 (S&P500 백분위) ==")
    print(A.drop(columns="코드").round(1).to_string(index=False))

    # 로지스틱: 확대 vs 나머지(유지·축소·청산)
    X = E.loc[add | hold | cut, [f"pr_{c}" for c in ["r12m", "r1m", "vs200", "dist52h", "vol60", "mcap", "rev_g_ttm", "ey", "gross_margin"]]]
    y = add[X.index].astype(float)
    d = pd.concat([X, y.rename("y")], axis=1).dropna()
    m = sm.Logit(d["y"], sm.add_constant((d.drop(columns="y") - 50) / 10)).fit(disp=0,
                                                                                  cov_type="cluster", cov_kwds={"groups": E.loc[d.index, "period"].astype("int64")})
    L = pd.DataFrame({"계수(백분위10당)": m.params, "t": m.tvalues}).round(2)
    L.index = ["상수"] + [LBL[i.replace("pr_", "")] for i in L.index[1:]]
    L.to_csv(OUT / "add_logit.csv", encoding="utf-8-sig")
    print("\n로지스틱(확대=1 vs 유지·축소·청산=0, 분기 군집 표준오차) n=", len(d))
    print(L.to_string())

    # B. 성공 vs 실패 확대
    B_ = E[add & E["f12m_t1_x"].notna()].copy()
    lo, hi = B_["f12m_t1_x"].quantile([1 / 3, 2 / 3])
    rows = []
    for c in KEY:
        pc = f"pr_{c}"
        w = B_.loc[B_["f12m_t1_x"] >= hi, pc].dropna()
        l_ = B_.loc[B_["f12m_t1_x"] <= lo, pc].dropna()
        rows.append(dict(특징=LBL[c], 성공=w.mean(), 실패=l_.mean(), t=stats.ttest_ind(w, l_, equal_var=False).statistic))
    Bt = pd.DataFrame(rows).sort_values("t", key=lambda x: -x.abs())
    Bt.to_csv(OUT / "add_win_vs_loss.csv", index=False, encoding="utf-8-sig")
    print(f"\n== B. 늘린 종목 중 성공(12개월 초과 ≥ {hi:+.0%}) vs 실패(≤ {lo:+.0%}) ==")
    print(Bt.round(1).to_string(index=False))

    # C. 실제 수치
    raw = [("r12m", "pct"), ("r6m", "pct"), ("r1m", "pct"), ("vs200", "pct"), ("dist52h", "pct"), ("vol60", "pct"), ("beta", "x"),
           ("mcap", "b"), ("rev_g_ttm", "pct"), ("gross_margin", "pct"), ("op_margin", "pct"), ("ps", "x"), ("ey", "pct")]
    f = lambda v, k: "-" if pd.isna(v) else (f"{v * 100:.0f}%" if k == "pct" else (f"{v / 1e9:.0f}십억$" if k == "b" else f"{v:.2f}"))
    C = pd.DataFrame([dict(특징=LBL[c], 늘린종목=f(E.loc[add, c].median(), k), 그대로둔종목=f(E.loc[hold, c].median(), k),
                           줄인판종목=f(E.loc[cut, c].median(), k), SP500=f(S[c].median(), k)) for c, k in raw])
    C.to_csv(OUT / "add_raw.csv", index=False, encoding="utf-8-sig")
    print("\n== C. 실제 수치(중앙값) ==")
    print(C.to_string(index=False))


if __name__ == "__main__":
    main()
