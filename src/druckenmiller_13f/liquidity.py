# -*- coding: utf-8 -*-
"""20) 유동성 점수로 주식 비중 정하기 (미국·코스피). 실제 매매 없음. KRX 요청 없음.

점수(분기말 t0에 아는 값, 3개월 전과 비교, 각 +1점):
  미국: 연준 대차대조표(WALCL) 증가 · 2년 국채금리(DGS2) 하락 · 신용 스프레드(BAA10Y) 축소 · 달러지수(DTWEXBGS) 하락
  한국: 연준 대차대조표 증가 · 한국 3개월 금리 하락(월 자료, 45일 늦게 안다고 가정) · 원화 강세(DEXKOUS 하락) · 신용 스프레드 축소
비중: 3~4점 100% / 2점 70% / 0~1점 40% (나머지 현금 = 3개월 금리). 숫자는 미리 정한 값, 고르지 않음.
적용: ① 시장지수(SPY·코스피) ② '그 프로필' 기본 규칙 ③ 바꿔 타기: 3점 이상이면 기본 규칙, 아니면 시장지수
확인: 드러켄밀러 본인이 점수 높을 때 종목 수·베타를 늘렸나(13F)
결과: results/liquidity/*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common_q import load, base_pick, fred, asof, cash, qsim, table, COST  # noqa: E402

OUT = HERE / "results" / "liquidity"
OUT.mkdir(parents=True, exist_ok=True)
SIG = {"us": [("WALCL", +1, 5, "연준 자산 증가"), ("DGS2", -1, 1, "2년 금리 하락"), ("BAA10Y", -1, 1, "신용 스프레드 축소"),
              ("DTWEXBGS", -1, 1, "달러 약세")],
       "kr": [("WALCL", +1, 5, "연준 자산 증가"), ("IR3TIB01KRM156N", -1, 45, "한국 3개월 금리 하락"),
              ("DEXKOUS", -1, 1, "원화 강세"), ("BAA10Y", -1, 1, "신용 스프레드 축소")]}
EXPO = {4: 1.0, 3: 1.0, 2: 0.7, 1: 0.4, 0: 0.4}


def score(mk, Q):
    rows = {}
    ser = {s: fred(s, lag) for s, _, lag, _ in SIG[mk]}
    for q in Q:
        d = {}
        for s, sg, lag, lb in SIG[mk]:
            a, b = asof(ser[s], q), asof(ser[s], q - pd.Timedelta(days=91))
            d[lb] = int(np.sign(a - b) == sg) if pd.notna(a) and pd.notna(b) else 0
        d["점수"] = sum(d.values())
        rows[q] = d
    return pd.DataFrame(rows).T


def main():
    res, sc_all = [], {}
    for mk in ["us", "kr"]:
        P, R, M = load(mk)
        Q = sorted(P["t0"].unique())
        S = score(mk, Q)
        sc_all[mk] = S
        S.to_csv(OUT / f"score_{mk}.csv", encoding="utf-8-sig")
        ex = S["점수"].map(EXPO).to_dict()
        cr = cash(mk, Q)
        picks = {q: base_pick(g) for q, g in P.groupby("t0")}
        mk_picks = {q: [] for q in Q}  # 빈 목록 → 시장지수 수익
        r_base, _ = qsim(picks, 2, R, M, COST[mk])
        r_base_l, _ = qsim(picks, 2, R, M, COST[mk], expo=ex, cashr=cr)
        r_mkt = M.reindex(Q)
        r_mkt_l = pd.Series({q: ex[q] * M.get(q, 0) + (1 - ex[q]) * cr[q] for q in Q})
        sw = {q: (picks[q] if S.loc[q, "점수"] >= 3 else []) for q in Q}
        r_sw, _ = qsim(sw, 2, R, M, COST[mk])
        rows = [("시장지수", mk, r_mkt, M, False), ("시장지수 + 유동성 비중", mk, r_mkt_l, M, False),
                ("기본 규칙", mk, r_base, M, False), ("기본 규칙 + 유동성 비중", mk, r_base_l, M, False),
                ("바꿔 타기(3점+ 규칙, 아니면 지수)", mk, r_sw, M, False)]
        if mk == "kr":
            rows += [(a, b, c, d, True) for a, b, c, d, _ in rows]
        res.append(table(rows))
        # 점수별 다음 분기 시장 수익
        d = pd.DataFrame({"점수": S["점수"], "시장": M.reindex(Q).values, "규칙": r_base.reindex(Q).values})
        g = d.groupby("점수").agg(분기수=("시장", "size"), 시장평균=("시장", "mean"), 규칙평균=("규칙", "mean"),
                                시장하락비율=("시장", lambda x: (x < 0).mean()))
        g.to_csv(OUT / f"by_score_{mk}.csv", encoding="utf-8-sig")
        print(f"\n== {mk} 점수별 다음 분기 수익 ==\n", g.round(3).to_string())
        print(f"{mk} 신호별 다음 분기 시장 수익 차이(켜짐 − 꺼짐):",
              {c: round(d.loc[S[c].values == 1, '시장'].mean() - d.loc[S[c].values == 0, '시장'].mean(), 3)
               for c in S.columns if c != "점수"})
    T = pd.concat(res, ignore_index=True)
    T.to_csv(OUT / "results.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("\n", T.round(3).to_string(index=False))

    # 드러켄밀러 본인 행동
    Pn = pd.read_pickle(HERE / "results" / "panel_ranked.pkl")
    D = Pn[(Pn["src"] == "드러켄밀러") & (Pn["leg"] == "주식") & (Pn["action"] != "청산")].copy()
    D["t0"] = pd.to_datetime(D["t0"])
    beh = D.groupby("t0").apply(lambda g: pd.Series(dict(종목수=len(g), 가중베타=np.nansum(g["weight"] * g["beta"]) / g.loc[g["beta"].notna(), "weight"].sum(),
                                                       IT비중=g.loc[g["sector"] == "IT", "weight"].sum())))
    S = sc_all["us"]
    j = beh.join(S["점수"], how="inner")
    print("\n== 드러켄밀러 행동 vs 미국 유동성 점수(분기말, 다음 분기 보유) 순위상관 ==")
    for c in ["종목수", "가중베타", "IT비중"]:
        rr = stats.spearmanr(j["점수"], j[c])
        print(f"  {c}: {rr.statistic:+.2f} (p {rr.pvalue:.2f})  점수 3+ 평균 {j.loc[j['점수'] >= 3, c].mean():.2f} / 0~1 평균 {j.loc[j['점수'] <= 1, c].mean():.2f}")
    j.to_csv(OUT / "druck_behavior.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
