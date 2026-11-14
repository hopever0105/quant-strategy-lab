# -*- coding: utf-8 -*-
"""26) 미국 수주잔고 신호 깊게 파기. 실제 매매 없음. KRX 요청 없음.

신호 gap = 수주잔고(없으면 계약부채) 전년 대비 증가율 − 매출(최근 4분기) 증가율  (분기말 t0에 공시된 값만)
A 진짜인가
  A1 공시 직후 효과: 신호가 담긴 공시일(f) 기준 앞 3개월·뒤 1/3/6개월 SPY 초과수익, 상위 30% − 하위 30%
  A2 다음 매출: gap이 2·4분기 뒤 매출 증가율 변화를 맞히나
  A3 누가 벌었나: 상위 5% 수익 잘라낸 뒤, 업종별 IC
B 돈이 되나
  B1 업종 안 상위 30%(업종 중립) / 산업재·IT만 상위 30% vs 업종 ETF(XLI·XLK)
  B2 상위 30% 매수 − 하위 30% 공매도(비용 양쪽)
  B3 보유 1·2·4분기
C 언제 잘 되나
  C1 수주잔고 증가율 가속(이번 − 직전 분기) / C2 잔고 ÷ 최근 4분기 매출 / C3 드러켄밀러 확신 매수 때 신호 위치
결과: results/backlog/deep_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common_q import load, qsim, table, COST, C  # noqa: E402
from features import load_px  # noqa: E402

OUT = HERE / "results" / "backlog"
pd.set_option("display.width", 250)


def tstat(x):
    x = pd.Series(x).dropna()
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan


def ic_series(d, col, y="ret"):
    return d.dropna(subset=[col, y]).groupby("t0").apply(
        lambda g: stats.spearmanr(g[col], g[y]).statistic if len(g) > 20 else np.nan).dropna()


def build():
    P, R, M = load("us")
    Pn = pd.read_pickle(HERE / "results" / "panel_ranked.pkl")
    S = Pn[Pn["src"] == "S&P500"].copy()
    S["t0"] = pd.to_datetime(S["t0"])
    y2c = S[["yf", "cik"]].dropna().drop_duplicates("yf")
    y2c = dict(zip(y2c["yf"], y2c["cik"].astype(int)))
    Q = sorted(P["t0"].unique())
    rows = []
    for cik in sorted(set(y2c.values())):
        fn = C / "fs_rpo" / f"{cik}.csv.gz"
        if not fn.exists():
            continue
        d = pd.read_csv(fn)
        if len(d) == 0:
            continue
        d = d[(d["unit"] == "USD") & d["start"].isna()].copy()
        d["end"] = pd.to_datetime(d["end"])
        d["filed"] = pd.to_datetime(d["filed"])
        for kind, cons in [("rpo", ["RevenueRemainingPerformanceObligation"]),
                           ("cl", ["ContractWithCustomerLiability", "DeferredRevenue"]),
                           ("clc", ["ContractWithCustomerLiabilityCurrent", "DeferredRevenueCurrent"])]:
            x = d[d["concept"].isin(cons)].sort_values("filed")
            if len(x) == 0:
                continue
            for q in Q:
                k = x[x["filed"] <= q]
                if len(k) == 0:
                    continue
                last = k.groupby("end").last()
                e1 = last.index.max()
                if (q - e1).days > 200:
                    continue
                prev = last[(last.index >= e1 - pd.Timedelta(days=410)) & (last.index <= e1 - pd.Timedelta(days=320))]
                if len(prev) == 0 or prev["val"].iloc[-1] <= 0:
                    continue
                rows.append(dict(cik=cik, t0=q, kind=kind, g=last.at[e1, "val"] / prev["val"].iloc[-1] - 1,
                                 lvl=last.at[e1, "val"], filed=last.at[e1, "filed"]))
    G = pd.DataFrame(rows)
    W = G.pivot_table(index=["cik", "t0"], columns="kind", values=["g", "lvl"], aggfunc="first")
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    F = G.pivot_table(index=["cik", "t0"], columns="kind", values="filed", aggfunc="first")
    F.columns = [f"filed_{b}" for b in F.columns]
    W = W.join(F).reset_index()
    for c in ["g_cl", "lvl_cl", "filed_cl"]:
        if c in W and c.replace("cl", "clc") in W:
            W[c] = W[c].fillna(W[c.replace("cl", "clc")])
    W["fwd_g"] = W["g_rpo"].fillna(W["g_cl"])
    W["lvl"] = W["lvl_rpo"].fillna(W["lvl_cl"])
    W["filed"] = W["filed_rpo"].fillna(W["filed_cl"])
    W["is_rpo"] = W["g_rpo"].notna()
    P["cik"] = P["code"].map(y2c)
    P = P.merge(W[["cik", "t0", "fwd_g", "lvl", "filed", "is_rpo"]], on=["cik", "t0"], how="left")
    P = P.merge(S[["t0", "yf", "rev_ttm", "sector"]].rename(columns={"yf": "code"}), on=["t0", "code"], how="left")
    P["gap"] = P["fwd_g"] - P["sales_g"]
    P = P.sort_values(["code", "t0"])
    P["fwd_g_prev"] = P.groupby("code")["fwd_g"].shift(1)
    P["accel"] = P["fwd_g"] - P["fwd_g_prev"]
    P["ratio"] = P["lvl"] / P["rev_ttm"]
    P["sales_g_f2"] = P.groupby("code")["sales_g"].shift(-2)
    P["sales_g_f4"] = P.groupby("code")["sales_g"].shift(-4)
    P["d_sales_f2"] = P["sales_g_f2"] - P["sales_g"]
    P["d_sales_f4"] = P["sales_g_f4"] - P["sales_g"]
    P = P[P["t0"] >= "2019-03-31"]
    P["p_gap"] = P.groupby("t0")["gap"].rank(pct=True) * 100
    return P, R, M


def main():
    P, R, M = build()
    P.to_pickle(OUT / "deep_panel_us.pkl")
    d = P.dropna(subset=["gap"])
    print(f"표본 {len(d)}행, 분기당 {d.groupby('t0').size().mean():.0f}종목, 수주잔고(RPO) 비율 {d['is_rpo'].mean():.0%}")

    # ---------- A1 공시 직후 ----------
    spy = load_px("SPY")["adj"]
    ev = d.dropna(subset=["filed"]).drop_duplicates(["code", "filed"]).copy()
    ev["fq"] = ev["filed"].dt.to_period("Q")
    ev["rk"] = ev.groupby("fq")["gap"].rank(pct=True)
    win = {"앞 3개월": (-63, 0), "뒤 1개월": (1, 21), "뒤 3개월": (1, 63), "뒤 6개월": (1, 126)}
    px = {}
    for t in ev["code"].unique():
        x = load_px(t)
        if x is not None:
            px[t] = x["adj"].astype(float)
    idx = spy.index
    for nm, (a, b) in win.items():
        vals = []
        for _, e in ev.iterrows():
            s = px.get(e["code"])
            if s is None:
                vals.append(np.nan)
                continue
            i0 = idx.searchsorted(e["filed"])  # 공시일(장 마감 후 공시 가정 → 다음날부터)
            ia, ib = i0 + a, i0 + b
            if ia < 0 or ib >= len(idx):
                vals.append(np.nan)
                continue
            da, db = idx[ia], idx[ib]
            sa, sb = s[:da], s[:db]
            if len(sa) == 0 or (da - sa.index[-1]).days > 7:
                vals.append(np.nan)
                continue
            vals.append((sb.iloc[-1] / sa.iloc[-1] - 1) - (spy[:db].iloc[-1] / spy[:da].iloc[-1] - 1))
        ev[nm] = vals
    out = []
    for nm in win:
        q = ev.groupby("fq").apply(lambda g: g.loc[g["rk"] >= 0.7, nm].mean() - g.loc[g["rk"] <= 0.3, nm].mean())
        out.append(dict(기간=nm, 상위30_평균=ev.loc[ev["rk"] >= 0.7, nm].mean(), 하위30_평균=ev.loc[ev["rk"] <= 0.3, nm].mean(),
                        차이_분기평균=q.mean(), t=tstat(q), 분기수=q.notna().sum()))
    A1 = pd.DataFrame(out)
    A1.to_csv(OUT / "deep_A1_event.csv", index=False, encoding="utf-8-sig")
    print("\n== A1 공시일 기준 SPY 초과수익 (상위 30% vs 하위 30%) ==\n", A1.round(4).to_string(index=False))

    # ---------- A2 다음 매출 ----------
    out = []
    for y, lb in [("d_sales_f2", "2분기 뒤 매출 증가율 변화"), ("d_sales_f4", "4분기 뒤 매출 증가율 변화")]:
        s = ic_series(d, "gap", y)
        s2 = ic_series(d, "fwd_g", y)
        out.append(dict(맞힐것=lb, IC_gap=s.mean(), t_gap=tstat(s), IC_증가율만=s2.mean(), t_증가율만=tstat(s2), 분기=len(s)))
    hi = d[d["p_gap"] >= 70]
    lo = d[d["p_gap"] <= 30]
    A2 = pd.DataFrame(out)
    A2.to_csv(OUT / "deep_A2_sales.csv", index=False, encoding="utf-8-sig")
    print("\n== A2 신호가 앞으로의 매출을 맞히나 ==\n", A2.round(3).to_string(index=False))
    print(f"   상위 30%: 지금 매출 증가율 중앙 {hi['sales_g'].median():.1%} → 4분기 뒤 {hi['sales_g_f4'].median():.1%} / "
          f"하위 30%: {lo['sales_g'].median():.1%} → {lo['sales_g_f4'].median():.1%}")

    # ---------- A3 누가 벌었나 ----------
    dd = d.dropna(subset=["ret"]).copy()
    dd["ret_w"] = dd.groupby("t0")["ret"].transform(lambda x: x.clip(x.quantile(0.05), x.quantile(0.95)))
    base = ic_series(dd, "gap")
    w = ic_series(dd, "gap", "ret_w")
    q = dd.groupby("t0").apply(lambda g: g.loc[g["p_gap"] >= 70, "ret_w"].mean() - g.loc[g["p_gap"] <= 30, "ret_w"].mean())
    print(f"\n== A3 상·하위 5% 수익 잘라낸 뒤: IC {w.mean():+.3f} t {tstat(w):+.2f} (원래 {base.mean():+.3f} t {tstat(base):+.2f}), "
          f"상위30−하위30 {q.mean():+.2%}/분기 t {tstat(q):+.2f}")
    sec = []
    for s_, g in dd.groupby("sector"):
        s = ic_series(g.assign(t0=g["t0"]), "gap") if g.groupby("t0").size().median() > 20 else \
            g.groupby("t0").apply(lambda x: stats.spearmanr(x["gap"], x["ret"]).statistic if len(x) >= 6 else np.nan).dropna()
        sec.append(dict(업종=s_, 분기당종목=g.groupby("t0").size().median(), IC=s.mean(), t=tstat(s), 분기=len(s)))
    A3 = pd.DataFrame(sec).sort_values("분기당종목", ascending=False)
    A3.to_csv(OUT / "deep_A3_sector.csv", index=False, encoding="utf-8-sig")
    print(A3.round(3).to_string(index=False))

    # ---------- B 돈이 되나 ----------
    by = {q_: g for q_, g in d.groupby("t0")}
    d["p_gap_sec"] = d.groupby(["t0", "sector"])["gap"].rank(pct=True) * 100
    by = {q_: g for q_, g in d.groupby("t0")}
    ew = d.dropna(subset=["ret"]).groupby("t0")["ret"].mean()
    picks = {"상위 30%": {q_: g.loc[g["p_gap"] >= 70, "code"].tolist() for q_, g in by.items()},
             "업종 안 상위 30%": {q_: g.loc[g["p_gap_sec"] >= 70, "code"].tolist() for q_, g in by.items()},
             "하위 30%": {q_: g.loc[g["p_gap"] <= 30, "code"].tolist() for q_, g in by.items()}}
    rows = []
    res = {}
    for H in [1, 2, 4]:
        for nm, pk in picks.items():
            r, n = qsim(pk, H, R, M, COST["us"])
            res[(nm, H)] = r
            rows.append((f"{nm} {H}분기 보유 (평균 {n.mean():.0f}종목)", "us", r, M, False))
        ls = res[("상위 30%", H)] - res[("하위 30%", H)] - 0.0005  # 공매도 쪽 추가 비용 근사(분기 0.05%)
        rows.append((f"상위30 매수−하위30 공매도 {H}분기", "us", ls, M, False))
        lsn = res[("업종 안 상위 30%", H)] - ew.reindex(res[("업종 안 상위 30%", H)].index)
        rows.append((f"업종 안 상위 30% − 동일가중 {H}분기", "us", lsn, M, False))
    rows.append(("신호 있는 종목 동일가중", "us", ew, M, False))
    rows.append(("SPY", "us", M, M, False))
    # 산업재·IT만 vs 업종 ETF
    etf = {}
    for t in ["XLI", "XLK"]:
        a = load_px(t)["adj"]
        Q = sorted(by)
        etf[t] = pd.Series({Q[i]: a[:(Q[i + 1] if i + 1 < len(Q) else a.index[-1])].iloc[-1] / a[:Q[i]].iloc[-1] - 1 for i in range(len(Q))})
    sub = {q_: g[g["sector"].isin(["산업재", "IT"])] for q_, g in by.items()}
    for H in [2]:
        pk = {q_: g.loc[g.groupby("sector")["gap"].rank(pct=True) >= 0.7, "code"].tolist() for q_, g in sub.items()}
        r, n = qsim(pk, H, R, M, COST["us"])
        bench = (etf["XLI"] + etf["XLK"]) / 2
        rows.append((f"산업재·IT 안 상위 30% {H}분기 (평균 {n.mean():.0f}종목)", "us", r, M, False))
        rows.append(("XLI·XLK 반반", "us", bench, M, False))
        rows.append((f"산업재·IT 상위 30% − XLI·XLK", "us", r - bench.reindex(r.index), M, False))
    B = table(rows)
    B.to_csv(OUT / "deep_B_money.csv", index=False, encoding="utf-8-sig")
    print("\n== B 돈이 되나 (2019~, 검증 2019~2021 / 테스트 2022~) ==")
    print(B[B["구간"].isin(["검증", "테스트", "전체"])][["전략", "구간", "CAGR", "샤프", "MDD", "시장CAGR", "시장샤프"]].round(3).to_string(index=False))

    # ---------- C 언제 잘 되나 ----------
    out = []
    for col, lb in [("gap", "기본: 증가율 − 매출 증가율"), ("fwd_g", "수주잔고 증가율만"), ("accel", "C1 증가율 가속(이번−직전)"),
                    ("ratio", "C2 잔고 ÷ 최근 4분기 매출")]:
        s = ic_series(d, col)
        out.append(dict(신호=lb, IC=s.mean(), t=tstat(s), 분기=len(s), 분기당종목=d.dropna(subset=[col]).groupby("t0").size().median()))
    # 결합: gap 상위 30% 안에서 가속도 + 인 것
    d["both"] = (d["p_gap"] >= 70) & (d["accel"] > 0)
    sp = d.dropna(subset=["ret"]).groupby(["t0", "both"])["ret"].mean().unstack().dropna()
    out.append(dict(신호="gap 상위30% ∩ 가속>0 − 나머지", IC=(sp[True] - sp[False]).mean(), t=tstat(sp[True] - sp[False]), 분기=len(sp),
                    분기당종목=d[d["both"]].groupby("t0").size().median()))
    d["rq"] = d.groupby("t0")["ratio"].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) if x.notna().sum() >= 10 else np.nan)
    rq = d.dropna(subset=["ret", "rq"]).groupby("rq")["ret"].mean()
    C3 = pd.DataFrame(out)
    C3.to_csv(OUT / "deep_C_when.csv", index=False, encoding="utf-8-sig")
    print("\n== C 언제 잘 되나 ==\n", C3.round(3).to_string(index=False))
    print("   잔고/매출 5분위별 다음 분기 평균 수익(0=낮음):", rq.round(4).to_dict())
    # 상위 30% 안에서 잔고/매출 높은 쪽 vs 낮은 쪽
    t3 = d[d["p_gap"] >= 70].copy()
    t3["rhi"] = t3.groupby("t0")["ratio"].transform(lambda x: x >= x.median())
    sp = t3.dropna(subset=["ret"]).groupby(["t0", "rhi"])["ret"].mean().unstack().dropna()
    print(f"   gap 상위 30% 안: 잔고/매출 높은 절반 − 낮은 절반 {(sp[True] - sp[False]).mean():+.2%}/분기 t {tstat(sp[True] - sp[False]):+.2f}")

    # 드러켄밀러 연결
    Pn = pd.read_pickle(HERE / "results" / "panel_ranked.pkl")
    E = Pn[(Pn["src"] == "드러켄밀러") & (Pn["leg"] == "주식") & Pn["action"].isin(["신규", "확대"]) & (Pn["weight"] >= 0.03)].copy()
    E["t0"] = pd.to_datetime(E["t0"])
    E = E.merge(d[["t0", "code", "p_gap", "gap", "fwd_g", "sales_g", "accel"]], left_on=["t0", "yf"], right_on=["t0", "code"], how="inner")
    E["p_gap_prev"] = np.nan
    # 그가 사기 한 분기 전 신호
    prev = d[["t0", "code", "p_gap"]].copy()
    prev["t0"] = prev["t0"] + pd.offsets.QuarterEnd(1)
    E = E.drop(columns="p_gap_prev").merge(prev.rename(columns={"p_gap": "p_gap_prev"}), on=["t0", "code"], how="left")
    print(f"\n== 드러켄밀러 확신 매수(2019~, 신호 있는 {len(E)}건) ==")
    print(f"   살 때 신호 백분위 중앙 {E['p_gap'].median():.0f} (무작위면 50), 상위 30% 안 비율 {(E['p_gap'] >= 70).mean():.0%} (무작위 30%), "
          f"한 분기 전에 이미 상위 30% {(E['p_gap_prev'] >= 70).mean():.0%}")
    cols = ["period", "yf", "name", "weight", "p_gap", "p_gap_prev", "fwd_g", "sales_g", "f6m_t1_x"]
    E[cols].sort_values("period").to_csv(OUT / "deep_druck.csv", index=False, encoding="utf-8-sig")
    print(E[cols].sort_values("period").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
