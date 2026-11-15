# -*- coding: utf-8 -*-
"""19) '작게 시작 → 크게 키우기' 경로 분석. 실제 매매 없음. KRX 요청 없음.

1) 보유 한 번(신규 ~ 청산)을 '에피소드'로 묶고, 처음 비중 1% 미만으로 시작한 것 중
   나중에 3% 이상으로 키운 비율 / 1~3%까지만 / 안 키우고 판 비율. 확신 포지션(3%+)이 작게 시작했는지 바로 크게 샀는지.
2) 키운 것 vs 안 키운 것: 처음 산 분기 직전(t0) 특징, 산 분기 안 주가(r_inq_x), 그다음 분기 주가(f3m_t1_x), 매출 성장 변화
3) 따라 하기(공시일 기준, 2분기 보유):
   S1 작은 신규 전부 / S2 작은 신규 중 산 분기에 SPY보다 오른 것 / S3 S2 + 공시일까지도 SPY보다 오른 것
   S4 그가 3% 이상으로 키운 걸 공시일에 사기(기다렸다 타기)
결과: results/scaleup/*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine import simulate, perf  # noqa: E402
from rule_bt import adj, alpha, ff_factors, SEG  # noqa: E402
from features import load_px  # noqa: E402

OUT = HERE / "results" / "scaleup"
OUT.mkdir(parents=True, exist_ok=True)


def episodes():
    tr = pd.read_csv(HERE / "results" / "trades.csv", dtype={"cusip": str}, parse_dates=["period", "filed"])
    sec = pd.read_csv(HERE / "results" / "securities.csv", dtype={"cusip": str})[["cusip", "yf", "kind"]]
    tr = tr[tr["leg"] == "주식"].merge(sec, on="cusip", how="left")
    tr = tr[tr["kind"] != "ETF"].sort_values(["cusip", "period"])
    eps = []
    for cu, g in tr.groupby("cusip"):
        cur = None
        for _, r in g.iterrows():
            if r["action"] == "신규":
                cur = dict(cusip=cu, yf=r["yf"], name=r["name"], start=r["period"], filed=r["filed"], w0=r["weight"],
                           wmax=r["weight"], q_max=0, nq=1, q_to3=np.nan, end=None)
                continue
            if cur is None:
                continue
            if r["action"] == "청산":
                cur["end"] = r["period"]
                eps.append(cur)
                cur = None
                continue
            cur["nq"] += 1
            if r["weight"] > cur["wmax"]:
                cur["wmax"], cur["q_max"] = r["weight"], cur["nq"] - 1
            if np.isnan(cur["q_to3"]) and r["weight"] >= 0.03:
                cur["q_to3"] = cur["nq"] - 1
        if cur is not None:
            eps.append(cur)
    E = pd.DataFrame(eps)
    E["시작"] = np.where(E["w0"] < 0.01, "작게(1%-)", np.where(E["w0"] < 0.03, "중간(1~3%)", "바로 크게(3%+)"))
    E["경로"] = np.where(E["wmax"] >= 0.03, "3%+로 키움", np.where(E["wmax"] >= 0.01, "1~3%까지", "안 키움"))
    E["진행중"] = E["end"].isna()
    return E


def main():
    E = episodes()
    E.to_csv(OUT / "episodes.csv", index=False, encoding="utf-8-sig")
    print(f"보유 에피소드 {len(E)} (진행중 {E['진행중'].sum()})")
    t = pd.crosstab(E["시작"], E["경로"], margins=True)
    print("\n== 시작 비중 × 최대 비중 (건수) ==\n", t.to_string())
    print("\n== 시작 비중별 경로 비율 ==\n", pd.crosstab(E["시작"], E["경로"], normalize="index").round(3).to_string())
    big = E[E["wmax"] >= 0.03]
    print(f"\n3% 이상이 된 포지션 {len(big)}개 중: 작게 시작 {(big['w0'] < 0.01).mean():.0%}, 중간 시작 "
          f"{((big['w0'] >= 0.01) & (big['w0'] < 0.03)).mean():.0%}, 바로 크게 {(big['w0'] >= 0.03).mean():.0%}")
    s = E[(E["w0"] < 0.01) & (E["wmax"] >= 0.03)]
    print(f"작게 시작해 3%+로 키운 것: 몇 분기 뒤에 3% 넘었나 {s['q_to3'].value_counts().sort_index().to_dict()}")
    print("에피소드 길이(분기) 중앙값: ", E.groupby("경로")["nq"].median().to_dict())

    # 2) 키운 것 vs 안 키운 것: 처음 산 때 특징
    P = pd.read_pickle(HERE / "results" / "panel_ranked.pkl")
    D = P[(P["src"] == "드러켄밀러") & (P["leg"] == "주식") & (P["action"] == "신규")].copy()
    D["period"] = pd.to_datetime(D["period"])
    S = E[E["w0"] < 0.01].merge(D, left_on=["cusip", "start"], right_on=["cusip", "period"], how="inner", suffixes=("", "_p"))
    S = S[~S["진행중"] | (S["wmax"] >= 0.03)]
    S["키움"] = S["wmax"] >= 0.03
    feats = [("r_inq_x", "산 분기 안 주가(SPY 초과)"), ("f3m_t1_x", "다음 분기 주가(SPY 초과)"), ("f6m_t1_x", "다음 6개월(SPY 초과)"),
             ("pr_r12m", "12개월 수익률 백분위"), ("pr_r3m", "3개월 수익률 백분위"), ("pr_vs200", "200일선 대비 백분위"),
             ("pr_dist52h", "52주 고점 근접 백분위"), ("pr_vol60", "변동성 백분위"), ("pr_beta", "베타 백분위"),
             ("pr_mcap", "시총 백분위"), ("pr_rev_g_ttm", "매출 성장 백분위"), ("pr_rev_g_q", "최근 분기 매출 성장 백분위"),
             ("pr_rev_accel", "매출 성장 가속 백분위"), ("pr_gross_margin", "매출총이익률 백분위"), ("pr_roe", "ROE 백분위"),
             ("pr_ps", "PSR 백분위"), ("pr_rnd_sales", "R&D/매출 백분위"), ("pr_dollar_vol", "거래대금 백분위")]
    rows = []
    for c, lb in feats:
        a, b = S.loc[S["키움"], c].dropna(), S.loc[~S["키움"], c].dropna()
        if len(a) < 10 or len(b) < 10:
            continue
        rows.append(dict(특징=lb, 키운것_중앙=a.median(), 안키운것_중앙=b.median(), 키운것_n=len(a), 안키운것_n=len(b),
                         p값=stats.mannwhitneyu(a, b).pvalue))
    F = pd.DataFrame(rows)
    F.to_csv(OUT / "grow_vs_not.csv", index=False, encoding="utf-8-sig")
    print(f"\n== 작게 시작한 신규: 3%+로 키운 것({S['키움'].sum()}) vs 안 키운 것({(~S['키움']).sum()}) — 처음 산 때 ==")
    print(F.round(3).to_string(index=False))

    # 3) 따라 하기 (공시일 기준)
    tr = pd.read_csv(HERE / "results" / "trades.csv", dtype={"cusip": str}, parse_dates=["period", "filed"])
    sec = pd.read_csv(HERE / "results" / "securities.csv", dtype={"cusip": str})[["cusip", "yf", "kind"]]
    tr = tr[tr["leg"] == "주식"].merge(sec, on="cusip", how="left")
    tr = tr[tr["kind"] != "ETF"]
    spy = adj("SPY")
    per = sorted(tr["period"].unique())
    filed = tr.groupby("period")["filed"].min()
    rq = P[(P["src"] == "드러켄밀러")][["period", "cusip", "r_inq_x"]].copy()
    rq["period"] = pd.to_datetime(rq["period"])
    tr = tr.merge(rq.drop_duplicates(["period", "cusip"]), on=["period", "cusip"], how="left")

    def since(t, a, b):
        s = adj(t)
        if s is None or len(s[:a]) == 0:
            return np.nan
        return s[:b].iloc[-1] / s[:a].iloc[-1] - 1

    sch = {k: [] for k in ["S1 작은 신규 전부", "S2 작은 신규·산 분기 SPY보다 오름", "S3 S2+공시일까지도 SPY보다 오름",
                           "S4 그가 3%+로 키운 것(확대)"]}
    for i, p in enumerate(per):
        g = tr[tr["period"] == p]
        f0 = pd.Timestamp(filed[p])
        f1 = pd.Timestamp(filed[per[i + 1]]) if i + 1 < len(per) else spy.index[-1]
        sm = g[(g["action"] == "신규") & (g["weight"] < 0.01)].dropna(subset=["yf"])
        s2 = sm[sm["r_inq_x"] > 0]
        spy_since = spy[:f0].iloc[-1] / spy[:pd.Timestamp(p)].iloc[-1] - 1
        s3 = s2[[since(t, pd.Timestamp(p), f0) > spy_since for t in s2["yf"]]] if len(s2) else s2
        s4 = g[(g["action"] == "확대") & (g["weight"] >= 0.03)].dropna(subset=["yf"])
        for k, d in zip(sch, [sm, s2, s3, s4]):
            sch[k].append((f0, f1, {t: 1.0 for t in d["yf"].unique()}))
    rf = (load_px("^IRX")["close"] / 100 / 252).reindex(spy.index).ffill().fillna(0)
    ff = ff_factors()
    res = []
    for k, s in sch.items():
        n = np.mean([len(x[2]) for x in s])
        for h in [1, 2]:
            r, _ = simulate(s, adj, spy, sleeves=h)
            for seg in SEG:
                a, b = SEG[seg]
                m = perf(r, rf, a, b)
                if not m:
                    continue
                m.update(alpha(r, ff, seg))
                res.append(dict(전략=k, 보유분기=h, 구간=seg, 평균종목수=n, **m))
    sp = spy.pct_change().dropna()["2012-02-15":]
    for seg in SEG:
        a, b = SEG[seg]
        res.append(dict(전략="SPY", 보유분기=0, 구간=seg, **perf(sp, rf, a, b)))
    R = pd.DataFrame(res)
    R.to_csv(OUT / "follow_results.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    show = ["전략", "보유분기", "구간", "평균종목수", "CAGR", "샤프", "MDD", "FF6_알파(연)", "FF6_t"]
    print("\n== 따라 하기(공시일) ==\n", R[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
