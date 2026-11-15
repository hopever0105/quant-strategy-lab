# -*- coding: utf-8 -*-
"""11) 13F 공시일에 따라 사기. 실제 매매 없음.

13F 공시일(분기말 + 약 45일) 종가에 사서 다음 공시일까지 보유.
  F1 전체 복제: 13F 주식 보유 비중 그대로(ETF 포함, 옵션 제외)
  F2 확신·확대: 그 분기 '확대' 종목 + 비중 3% 이상 신규 종목, 동일가중
  F3 상위 10: 비중 상위 10종목, 동일가중
  F4 작은 신규: 비중 1% 미만 신규 종목, 동일가중 (대조: 실력 안 보인 쪽)
비교: 같은 포트폴리오를 분기말(그가 실제로 들고 있던 날)에 샀다고 가정한 경우 → 공시 지연 비용
시세 없는 종목은 SPY로 대체(비중 기록). 비용 편도 0.1%.
결과: results/follow/*.csv, charts/follow_equity.png
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rule_bt import adj, metrics, alpha, ff_factors, SEG, COST  # noqa: E402
from features import load_px  # noqa: E402

OUT = HERE / "results" / "follow"
OUT.mkdir(parents=True, exist_ok=True)
CH = HERE / "charts"


def books():
    tr = pd.read_csv(HERE / "results" / "trades.csv", dtype={"cusip": str}, parse_dates=["period", "filed"])
    sec = pd.read_csv(HERE / "results" / "securities.csv", dtype={"cusip": str})[["cusip", "yf", "kind"]]
    tr = tr[tr["leg"] == "주식"].merge(sec, on="cusip", how="left")
    out = {}
    for p, g in tr.groupby("period"):
        h = g[g["shares"] > 0]
        f = g["filed"].iloc[0]
        b = {}
        b["F1 전체 복제"] = h.groupby("yf")["value"].sum()
        m2 = (g["action"] == "확대") | ((g["action"] == "신규") & (g["weight"] >= 0.03))
        b["F2 확신·확대"] = pd.Series(1.0, index=g.loc[m2, "yf"].dropna().unique())
        b["F3 상위 10"] = pd.Series(1.0, index=h.sort_values("weight", ascending=False)["yf"].dropna().head(10).unique())
        m4 = (g["action"] == "신규") & (g["weight"] < 0.01) & (g["kind"] != "ETF")
        b["F4 작은 신규"] = pd.Series(1.0, index=g.loc[m4, "yf"].dropna().unique())
        out[p] = (f, b)
    return out


def run(bk, name, start_key, spy):
    """start_key: 'filed' 또는 'period'. 반환 일별 수익률, SPY 대체 비중 기록."""
    keys = sorted(bk)
    daily, miss = [], []
    prev_w = pd.Series(dtype=float)
    for i, p in enumerate(keys):
        f, b = bk[p]
        w = b[name].copy()
        w = w[w.index.notna()]
        if w.sum() <= 0:
            continue
        t0 = pd.Timestamp(f if start_key == "filed" else p)
        if i + 1 < len(keys):
            nf, _ = bk[keys[i + 1]]
            t1 = pd.Timestamp(nf if start_key == "filed" else keys[i + 1])
        else:
            t1 = spy.index[-1]
        idx = spy[(spy.index > t0) & (spy.index <= t1)].index
        if len(idx) == 0:
            continue
        w = w / w.sum()
        val = pd.DataFrame(index=idx)
        missing = 0.0
        for t, wt in w.items():
            s = adj(t)
            ok = s is not None and len(s[:t0]) and (t0 - s[:t0].index[-1]).days <= 7
            if not ok:
                s = adj("SPY")
                missing = missing + wt
            base = s[:t0].iloc[-1]
            val[t] = s[(s.index > t0) & (s.index <= t1)].reindex(idx).ffill().fillna(base) / base * wt
        v = val.sum(axis=1)
        r = v.pct_change()
        r.iloc[0] = v.iloc[0] - 1
        allk = w.index.union(prev_w.index)
        r.iloc[0] = r.iloc[0] - (w.reindex(allk, fill_value=0) - prev_w.reindex(allk, fill_value=0)).abs().sum() * COST
        ev = val.iloc[-1]
        prev_w = ev / ev.sum()
        daily.append(r)
        miss.append(dict(period=p, t0=t0, n=len(w), spy_sub=missing))
    return pd.concat(daily).sort_index(), pd.DataFrame(miss)


def main():
    bk = books()
    spy = adj("SPY")
    rf = (load_px("^IRX")["close"] / 100 / 252).reindex(spy.index).ffill().fillna(0)
    ff = ff_factors()
    res, curves = [], {}
    for name in ["F1 전체 복제", "F2 확신·확대", "F3 상위 10", "F4 작은 신규"]:
        for sk, lab in [("filed", "공시일"), ("period", "분기말(불가능, 참고)")]:
            r, ms = run(bk, name, sk, spy)
            curves[(name, lab)] = r
            ms.to_csv(OUT / f"subs_{name.split()[0]}_{sk}.csv", index=False, encoding="utf-8-sig")
            for seg in SEG:
                m = metrics(r, rf, seg)
                if not m:
                    continue
                m.update(alpha(r, ff, seg))
                res.append(dict(전략=name, 기준=lab, 구간=seg, **m, 평균종목수=ms["n"].mean(), SPY대체비중=ms["spy_sub"].mean()))
    sp = spy.pct_change().dropna()["2012-02-15":]
    for seg in SEG:
        m = metrics(sp, rf, seg)
        m.update(alpha(sp, ff, seg))
        res.append(dict(전략="SPY", 기준="-", 구간=seg, **m))
    R = pd.DataFrame(res)
    R.to_csv(OUT / "results.csv", index=False, encoding="utf-8-sig")
    yr = pd.DataFrame({f"{k[0]}|{k[1][:3]}": (1 + v).groupby(v.index.year).prod() - 1 for k, v in curves.items()})
    yr["SPY"] = (1 + sp).groupby(sp.index.year).prod() - 1
    yr.to_csv(OUT / "yearly.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(11, 5))
    for name in ["F1 전체 복제", "F2 확신·확대", "F3 상위 10", "F4 작은 신규"]:
        v = curves[(name, "공시일")]
        ax.plot((1 + v).cumprod(), label=name + " (공시일)", lw=1.3)
    ax.plot((1 + sp).cumprod(), label="SPY", color="k", lw=1)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("13F 공시일에 따라 사기 누적 수익 (로그)")
    plt.tight_layout()
    plt.savefig(CH / "follow_equity.png", dpi=150)
    plt.close()
    pd.set_option("display.width", 250)
    show = ["전략", "기준", "구간", "CAGR", "변동성", "샤프", "MDD", "CAPM_알파(연)", "CAPM_t", "FF6_알파(연)", "FF6_t", "SPY대체비중"]
    print(R[R["구간"].isin(["전체", "학습", "검증", "테스트"])][show].round(3).to_string(index=False))
    print(yr.round(3).to_string())


if __name__ == "__main__":
    main()
