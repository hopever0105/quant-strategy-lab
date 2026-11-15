# -*- coding: utf-8 -*-
"""24) '앞으로 들어올 매출' 신호 (미국·코스피). 실제 매매 없음. KRX 요청 없음.

미국: SEC XBRL 수주잔고(RevenueRemainingPerformanceObligation, 2018년 회계기준 변경 후 공시) +
      계약부채·이연수익(ContractWithCustomerLiability, DeferredRevenue: 유동+비유동, 없으면 유동만)
      분기말 t0에 이미 공시된(filed ≤ t0) 가장 최근 값 vs 그 1년 전 같은 시점 값 → 전년 대비 증가율
코스피: DART 사업보고서 재무상태표 계약부채·선수금·선수수익 합(당기 vs 전기), fy 재무는 fy+1년 4월부터 사용
신호: 증가율 > 매출 증가율 ('쌓이는 속도가 매출보다 빠름'), 그리고 증가율 > 0
테스트(분기, 2분기 보유):
  B1 신호 종목 전부 / B2 '증가율 − 매출 증가율' 상위 30% / B3 기본 규칙 ∩ 신호 / 기본 규칙 / 시장
  + 순위상관(IC): 다음 분기 수익 vs (증가율 − 매출 증가율)
결과: results/backlog/*.csv
"""
import glob
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common_q import load, base_pick, qsim, table, COST, C  # noqa: E402

OUT = HERE / "results" / "backlog"
OUT.mkdir(parents=True, exist_ok=True)


def us_signal(P):
    S = pd.read_pickle(HERE / "results" / "panel_ranked.pkl")
    S = S[S["src"] == "S&P500"][["yf", "cik"]].dropna().drop_duplicates("yf")
    y2c = dict(zip(S["yf"], S["cik"].astype(int)))
    Q = sorted(P["t0"].unique())
    rows = []
    for cik in sorted(set(y2c.values())):
        fn = C / "fs_rpo" / f"{cik}.csv.gz"
        if not fn.exists():
            continue
        d = pd.read_csv(fn)
        if len(d) == 0:
            continue
        d = d[d["unit"] == "USD"].copy()
        d["end"] = pd.to_datetime(d["end"])
        d["filed"] = pd.to_datetime(d["filed"])
        d = d[d["start"].isna()]  # 시점 값만
        for kind, cons in [("rpo", ["RevenueRemainingPerformanceObligation"]),
                           ("cl", ["ContractWithCustomerLiability", "DeferredRevenue"]),
                           ("cl_cur", ["ContractWithCustomerLiabilityCurrent", "DeferredRevenueCurrent"])]:
            x = d[d["concept"].isin(cons)].sort_values("filed")
            if len(x) == 0:
                continue
            # 같은 end에 여러 값 → 가장 최근 공시값(그 시점까지)
            for q in Q:
                k = x[x["filed"] <= q]
                if len(k) == 0:
                    continue
                k = k.groupby("end")["val"].last()
                e1 = k.index.max()
                if (q - e1).days > 200:
                    continue
                prev = k[(k.index >= e1 - pd.Timedelta(days=410)) & (k.index <= e1 - pd.Timedelta(days=320))]
                if len(prev) == 0 or prev.iloc[-1] <= 0:
                    continue
                rows.append(dict(cik=cik, t0=q, kind=kind, g=k[e1] / prev.iloc[-1] - 1))
    G = pd.DataFrame(rows).pivot_table(index=["cik", "t0"], columns="kind", values="g").reset_index()
    G["cl"] = G["cl"].fillna(G.get("cl_cur")) if "cl_cur" in G else G["cl"]
    P = P.copy()
    P["cik"] = P["code"].map(y2c)
    P = P.merge(G, on=["cik", "t0"], how="left")
    P["fwd_g"] = P["rpo"].fillna(P["cl"])   # 수주잔고 우선, 없으면 계약부채
    return P


def kr_signal(P):
    rows = []
    for fn in glob.glob(str(C / "kr_contract" / "*.json")):
        code, fy = Path(fn).stem.split("_")
        L = json.loads(Path(fn).read_text(encoding="utf-8"))
        if not L:
            continue
        def num(v):
            try:
                return float(str(v).replace(",", ""))
            except Exception:
                return np.nan
        bs = [x for x in L if not x["acc"].startswith("__sales__")
              and not any(w in x["acc"] for w in ("확정", "보험", "투자계약", "재보험"))]
        # 상위 합계 줄과 하위 줄이 같이 있으면 이중 계산 → 이름별 최댓값 하나씩, 유동/비유동만 합산
        cur = {}
        prv = {}
        for x in bs:
            nm = x["acc"]
            cur[nm] = max(cur.get(nm, 0), num(x["cur"]) if pd.notna(num(x["cur"])) else 0)
            prv[nm] = max(prv.get(nm, 0), num(x["prev"]) if pd.notna(num(x["prev"])) else 0)
        c, p = sum(cur.values()), sum(prv.values())
        rows.append(dict(code=code, fy=int(fy), cl_cur=c, cl_prev=p))
    K = pd.DataFrame(rows)
    K["cl"] = np.where(K["cl_prev"] > 0, K["cl_cur"] / K["cl_prev"] - 1, np.nan)
    P = P.copy()
    P["fy"] = np.where(P["t0"].dt.month >= 4, P["t0"].dt.year - 1, P["t0"].dt.year - 2)
    P = P.merge(K[["code", "fy", "cl"]], on=["code", "fy"], how="left")
    P["fwd_g"] = P["cl"]
    return P


def test(mk, P, R, M):
    P = P.copy()
    P["gap"] = P["fwd_g"] - P["sales_g"]
    P["sig"] = (P["fwd_g"] > P["sales_g"]) & (P["fwd_g"] > 0)
    P["p_gap"] = P.groupby("t0")["gap"].rank(pct=True) * 100
    cov = P.groupby("t0")["fwd_g"].apply(lambda s: s.notna().mean())
    start = cov[cov >= 0.3].index.min()
    print(f"[{mk}] 신호 자료 있는 비율(분기 평균) {cov.mean():.0%}, 30% 넘는 첫 분기 {start.date() if pd.notna(start) else '-'}")
    P = P[P["t0"] >= start]
    ic = P.dropna(subset=["gap", "ret"]).groupby("t0").apply(lambda g: stats.spearmanr(g["gap"], g["ret"]).statistic if len(g) > 20 else np.nan).dropna()
    ic2 = P.dropna(subset=["fwd_g", "ret"]).groupby("t0").apply(lambda g: stats.spearmanr(g["fwd_g"], g["ret"]).statistic if len(g) > 20 else np.nan).dropna()
    print(f"[{mk}] IC(증가율−매출증가율) 평균 {ic.mean():+.3f} t {ic.mean() / ic.std() * np.sqrt(len(ic)):+.2f} / "
          f"IC(증가율) {ic2.mean():+.3f} t {ic2.mean() / ic2.std() * np.sqrt(len(ic2)):+.2f} ({len(ic)}분기)")
    d = P.dropna(subset=["ret"])
    sp = d.groupby(["t0", "sig"])["ret"].mean().unstack()
    print(f"[{mk}] 신호 켜진 종목 − 꺼진 종목 다음 분기 수익 평균 {(sp[True] - sp[False]).mean():+.2%} (이긴 분기 {(sp[True] > sp[False]).mean():.0%})")
    Q = sorted(P["t0"].unique())
    by = {q: g for q, g in P.groupby("t0")}
    pk = {"B1 신호 종목 전부": {q: g.loc[g["sig"], "code"].tolist() for q, g in by.items()},
          "B2 차이 상위 30%": {q: g.loc[g["p_gap"] >= 70, "code"].tolist() for q, g in by.items()},
          "B3 기본 규칙 ∩ 신호": {q: [c for c in base_pick(g) if c in set(g.loc[g["sig"], "code"])] for q, g in by.items()},
          "기본 규칙": {q: base_pick(g) for q, g in by.items()}}
    rows = []
    for nm, p in pk.items():
        r, n = qsim(p, 2, R, M, COST[mk])
        rows.append((f"{nm} (평균 {n.mean():.0f}종목)", mk, r, M, False))
        if mk == "kr":
            rows.append((f"{nm} (평균 {n.mean():.0f}종목)", mk, r, M, True))
    lo, _ = qsim({q: g.loc[g["p_gap"] <= 30, "code"].tolist() for q, g in by.items()}, 2, R, M, COST[mk])
    hi, _ = qsim(pk["B2 차이 상위 30%"], 2, R, M, COST[mk])
    ew = P.groupby("t0")["ret"].mean()
    ls = (hi - lo).dropna()
    print(f"[{mk}] 상위30% − 하위30% 분기 평균 {ls.mean():+.2%} t {ls.mean() / ls.std() * np.sqrt(len(ls)):+.2f}, "
          f"상위30% − 동일가중 시장 {(hi - ew).mean():+.2%}/분기")
    rows.append(("하위 30% (비교)", mk, lo, M, False))
    rows.append(("동일가중 시장", mk, ew.reindex(Q), M, False))
    rows.append(("시장지수", mk, M.reindex(Q), M, False))
    if mk == "kr":
        rows.append(("시장지수", mk, M.reindex(Q), M, True))
    T = table(rows)
    T["IC"] = ic.mean()
    return T


def main():
    out = []
    which = sys.argv[1:] or ["us", "kr"]
    for mk in which:
        P, R, M = load(mk)
        P = us_signal(P) if mk == "us" else kr_signal(P)
        P[["t0", "code", "fwd_g", "sales_g"]].to_pickle(OUT / f"signal_{mk}.pkl")
        out.append(test(mk, P, R, M))
    T = pd.concat(out, ignore_index=True)
    T.to_csv(OUT / f"results_{'_'.join(which)}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
