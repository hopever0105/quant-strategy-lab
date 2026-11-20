# -*- coding: utf-8 -*-
"""③ 우량주 필터가 고른 종목 전체(m3_all_picks.csv)와 종목별 초과기여(m3_contribution.csv)
기여 = (그 종목 1년 보유 수익 − 같은 기간 동일가중 수익) ÷ 그해 종목 수.
'성과를 만든 종목' = 기여 큰 순서로 그 종목 수익을 동일가중 수익으로 바꿔 넣었을 때 전략 연수익이 동일가중 이하가 되는 최소 개수.
실행: python picks.py (run.py, robust.py 뒤)"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D          # noqa: E402
import quality as QM      # noqa: E402

OUT = HERE / "results"


def main():
    m3 = pd.read_pickle(OUT / "m3_series.pkl")
    picks = m3["picks"]
    W = D.panel_wide()
    info = D.listing_info()
    Q = QM.Quality(W, info)
    apr = Q.month_ends(4)
    end = Q.C.index[-1]
    Q.set_thresholds(pd.concat([Q.universe(d, Q.mcaps([d])[d]) for d in apr if d.year <= 2017]))
    ent = Q.entries[m3["best3"][1]]
    rows, ew = [], {}
    ds = list(picks.keys())
    for a, b in zip(ds, ds[1:] + [end]):
        u = Q.universe(a, Q.mcaps([a])[a])
        days = Q.R.index[(Q.R.index > a) & (Q.R.index <= b)]
        ew[str(a.date())] = ((1 + Q.R.loc[days, u.index].fillna(0).astype(float)).prod() - 1).mean()
        for c in picks[a]:
            sig = ent.loc[a:b, c].fillna(False) if c in ent.columns else pd.Series(dtype=bool)
            first = sig[sig].index.min() if sig.any() else pd.NaT
            hold = days[days > first] if pd.notna(first) else days[:0]
            r = (1 + Q.R.loc[hold, c].fillna(0).astype(float)).prod() - 1 if len(hold) else np.nan
            rows.append(dict(리밸런싱=str(a.date()), 코드=c, 종목명=info.name.get(c, ""), 업종=u.loc[c, "sector"],
                             부채비율=u.loc[c, "de"], PER=u.loc[c, "per"], PSR=u.loc[c, "psr"], 시총억=u.loc[c, "mcap"],
                             매수일=str(first.date()) if pd.notna(first) else "미진입", 보유수익=r))
    p = pd.DataFrame(rows)
    p.to_csv(OUT / "m3_all_picks.csv", index=False, encoding="utf-8-sig")
    p["r"] = p.보유수익.fillna(0.0)
    p["n"] = p.groupby("리밸런싱").코드.transform("size")
    p["ew"] = p.리밸런싱.map(ew)
    p["초과기여"] = (p.r - p.ew) / p.n
    yrs = (end - ds[0]).days / 365.25
    e = pd.Series(ew)

    def cagr(df):
        per = df.groupby("리밸런싱")[["r", "n"]].apply(lambda g: (g.r / g.n).sum())
        return (1 + per).prod() ** (1 / yrs) - 1, (1 + e.reindex(per.index)).prod() ** (1 / yrs) - 1
    base, ewc = cagr(p)
    s = p.sort_values("초과기여", ascending=False).reset_index(drop=True)
    q = p.copy()
    n_need = None
    for i, row in s.iterrows():
        mk = (q.리밸런싱 == row.리밸런싱) & (q.코드 == row.코드)
        q.loc[mk, "r"] = q.loc[mk, "ew"]
        c, _ = cagr(q)
        s.loc[i, "바꾼뒤_전략연수익"] = c
        if c <= ewc:
            n_need = i + 1
            break
    s.to_csv(OUT / "m3_contribution.csv", index=False, encoding="utf-8-sig")
    s["바이오제약"] = s.업종.isin(["건강관리", "제약"])
    summ = dict(전략연수익=base, 동일가중연수익=ewc, 픽수=len(p), 필요종목수=n_need,
                상위합=float(s.head(n_need or 0).초과기여.sum()) if n_need else np.nan,
                나머지합=float(s.iloc[n_need or 0:].초과기여.sum()),
                나머지승률=float((s.iloc[n_need or 0:].r > s.iloc[n_need or 0:].ew).mean()),
                바이오비중=float(s.바이오제약.mean()), 바이오기여=float(s[s.바이오제약].초과기여.sum()),
                비바이오기여=float(s[~s.바이오제약].초과기여.sum()))
    pd.Series(summ).to_csv(OUT / "m3_contribution_summary.csv", encoding="utf-8-sig")
    print(summ)
    print(s.head(n_need or 10)[["리밸런싱", "종목명", "업종", "PER", "PSR", "시총억", "보유수익", "ew", "초과기여", "바꾼뒤_전략연수익"]].round(3).to_string(index=False))


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
