# -*- coding: utf-8 -*-
"""10) 무작위 대조: 매 분기 '대형 고베타' 후보(시총 상위 30%·베타≥1.0) 안에서
확신 프로필이 고른 것과 같은 개수를 무작위로 1,000번 뽑아 분기 수익(동일가중, 분기 보유)을 비교.
→ 재무·추세 필터가 '대형 고베타'라는 성격 이상으로 더해주는지. 시드 고정.
결과: results/rule/random_control.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rule_bt import load_panel, select, SEG  # noqa: E402

OUT = HERE / "results" / "rule"


def main():
    S = load_panel()
    grid = pd.read_csv(OUT / "grid.csv")
    tr = grid[(grid["구간"] == "학습") & (grid["평균종목수"] >= 5)].sort_values("샤프", ascending=False).iloc[0]
    p = dict(B=tr["B"], H=tr["H"], G=tr["G"], M=tr["M"])
    rng = np.random.default_rng(42)
    N = 1000
    rows = []
    for q0, g in S.groupby("t0"):
        g = g[g["r_inq"].notna()]
        pick = select(g, p)
        pool = g[(g["pr_mcap"] > 70) & (g["beta"] >= p["B"])]
        if len(pick) == 0 or len(pool) < len(pick) + 3:
            continue
        rr = g.set_index("yf")["r_inq"]
        strat = rr.reindex(pick).mean()
        pr = pool["r_inq"].values
        k = len(pick)
        sims = np.array([pr[rng.choice(len(pr), k, replace=False)].mean() for _ in range(N)])
        rows.append(dict(t0=q0, k=k, strat=strat, sim_mean=sims.mean(), pct=(sims < strat).mean(), sims=sims))
    D = pd.DataFrame(rows)
    out = []
    for seg, (a, b) in SEG.items():
        d = D[(D["t0"] >= pd.Timestamp(a) - pd.Timedelta(days=95)) & (D["t0"] <= b)]
        if len(d) == 0:
            continue
        sims = np.vstack(d["sims"].values)  # 분기 × N
        s_cum = np.prod(1 + d["strat"].values)
        sim_cum = np.prod(1 + sims, axis=0)
        out.append(dict(구간=seg, 분기=len(d), 전략누적=s_cum - 1, 무작위누적중앙=np.median(sim_cum) - 1,
                        무작위대비백분위=(sim_cum < s_cum).mean() * 100,
                        분기평균초과=(d["strat"] - d["sim_mean"]).mean(), 이긴분기비율=(d["strat"] > d["sim_mean"]).mean()))
    O = pd.DataFrame(out)
    O.to_csv(OUT / "random_control.csv", index=False, encoding="utf-8-sig")
    print(O.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
