# -*- coding: utf-8 -*-
"""13) 아이디어 4: 미국에서 정한 '확신 프로필'을 코스피에 그대로 적용 (조건 재조정 없음 = 전 기간 표본 밖). 실제 매매 없음. KRX 요청 없음.

데이터(로컬 캐시만): 종가 ohlc_clean.pkl(1999~, 상장폐지 포함, 분할·감자 보정), 거래대금, 상장정보 listing_info_v4,
  시가총액 factor_panel_v2(매년 4월 말) × 이후 주가 변화, DART 연간 재무 dart_fs(2013~) — fy 재무는 fy+1년 4월부터 사용.
규칙(분기말):
  코스피 보통주 · 20일 평균 거래대금 3억 이상 · 시총 코스피 상위 30% · 베타(코스피 대비 1년) ≥ 1.0
  · 12개월 수익률 > 0 · 52주 고점 −20% 이내 · 매출 성장(최근 연도) ≥ 5% · 영업이익 흑자(미국의 매출총이익률 40% 대신, 매출총이익 자료 없음)
  → 동일가중, 다음 분기말까지 보유. 해당 없으면 코스피 지수.
변형: 오른 종목 더 사기 / 흑자 조건 빼기
대조: 코스피 지수, 대형(상위 30%) 동일가중, 대형 고베타 동일가중, 대형 고베타 중 무작위 같은 개수 1,000회
비용: 편도 0.25%. 기간: 2015Q2~ (매출 성장 계산 가능 시점), 구간 2015~2020 / 2021~2026
결과: results/ideas/idea4_kr*.csv, charts/idea4_kr.png
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
from engine import simulate, perf  # noqa: E402

QUANT = Path(r"C:\Users\arlo0\OneDrive\바탕 화면\취업\quant")
CA = QUANT / "cache"
OUT = HERE / "results" / "ideas"
OUT.mkdir(parents=True, exist_ok=True)
CH = HERE / "charts"
COST = 0.0025
SEGK = {"2015~2020": ("2015-04-01", "2020-12-31"), "2021~2026": ("2021-01-01", "2026-12-31"), "전체": ("2015-04-01", "2026-12-31")}


def main():
    o = pd.read_pickle(CA / "ohlc_clean.pkl")
    c = o["c"].astype("float64").where(lambda x: x > 0)
    amt = o["amt"].astype("float64")
    L = pd.read_csv(CA / "larry" / "listing_info_v4.csv", dtype=str).drop_duplicates("code").set_index("code")
    nm = L["name"].reindex(c.columns).fillna("")
    common = (c.columns.str[-1] == "0") & ~nm.str.contains("스팩|SPAC|기업인수목적", regex=True).values
    kospi_mask = (L["market"].reindex(c.columns) == "KOSPI").values
    c = c.loc[:, common & kospi_mask]
    amt = amt[c.columns]
    ks = pd.read_csv(CA / "kullamaggie" / "KS11.csv", index_col=0, parse_dates=True)["c"].reindex(c.index).ffill()
    rf = pd.read_csv(CA / "larry" / "FRED_IR3TIB01KRM156N.csv", index_col=0, parse_dates=True).iloc[:, 0]
    rf = (rf / 100 / 252).reindex(c.index, method="ffill").fillna(0)
    print("코스피 보통주", c.shape[1], flush=True)

    r = c.pct_change(fill_method=None)
    kr = ks.pct_change()
    beta = r.rolling(252, min_periods=200).cov(kr).div(kr.rolling(252, min_periods=200).var(), axis=0)
    r12 = c / c.shift(252) - 1
    d52 = c / c.rolling(252, min_periods=200).max() - 1
    amt20 = amt.rolling(20, min_periods=10).mean()
    fp = pd.read_pickle(CA / "larry" / "factor_panel_v2.pkl")[["code", "date", "mcap"]].dropna()
    mc_a = fp.pivot_table(index="date", columns="code", values="mcap").reindex(columns=c.columns)
    fs = pd.read_csv(CA / "larry" / "dart_fs.csv", dtype={"code": str})
    fs = fs.drop_duplicates(["code", "fy"]).set_index(["code", "fy"])[["sales", "op"]]

    qe = c.index.to_series().groupby(c.index.to_period("Q")).max()
    qe = qe[(qe >= "2015-03-31")].tolist()
    if qe[-1] < c.index[-1]:
        qe.append(c.index[-1])
    snaps = []
    for t0 in qe[:-1]:
        fy = t0.year - 1 if t0.month >= 4 else t0.year - 2  # fy 재무는 fy+1년 4월부터
        a_dates = mc_a.index[mc_a.index <= t0]
        if len(a_dates) == 0:
            continue
        ad = a_dates[-1]
        mcap = mc_a.loc[ad] * (c.loc[t0] / c.loc[:ad].iloc[-1])
        g = pd.DataFrame({"mcap": mcap, "beta": beta.loc[t0], "r12": r12.loc[t0], "d52": d52.loc[t0],
                          "amt20": amt20.loc[t0], "px": c.loc[t0]})
        g = g[g["px"].notna() & (g["amt20"] >= 3e8) & g["mcap"].notna()]
        s1 = fs["sales"].xs(fy, level="fy") if fy in fs.index.get_level_values(1) else pd.Series(dtype=float)
        s0 = fs["sales"].xs(fy - 1, level="fy") if (fy - 1) in fs.index.get_level_values(1) else pd.Series(dtype=float)
        op = fs["op"].xs(fy, level="fy") if fy in fs.index.get_level_values(1) else pd.Series(dtype=float)
        g["sales_g"] = (s1 / s0 - 1).reindex(g.index).where(s0.reindex(g.index) > 0)
        g["op"] = op.reindex(g.index)
        g["pr_mcap"] = g["mcap"].rank(pct=True) * 100
        g["t0"] = t0
        snaps.append(g)
    print("분기", len(snaps), flush=True)
    t1_of = {qe[i]: qe[i + 1] for i in range(len(qe) - 1)}

    def sched(fn):
        return [(g["t0"].iloc[0], t1_of[g["t0"].iloc[0]], {t: 1.0 for t in fn(g)}) for g in snaps]

    large = lambda g: g["pr_mcap"] > 70
    lhb = lambda g: large(g) & (g["beta"] >= 1.0)
    prof = lambda g: lhb(g) & (g["r12"] > 0) & (g["d52"] >= -0.20) & (g["sales_g"] >= 0.05) & (g["op"] > 0)
    prof_noop = lambda g: lhb(g) & (g["r12"] > 0) & (g["d52"] >= -0.20) & (g["sales_g"] >= 0.05)

    def pfn(t):
        if t == "KOSPI":
            return ks
        return c[t].dropna() if t in c.columns else None

    strategies = {
        "확신 프로필(코스피)": dict(fn=prof),
        "확신 프로필 + 오른 종목 더 사기": dict(fn=prof, pyramid=True),
        "확신 프로필(흑자 조건 뺌)": dict(fn=prof_noop),
        "대조: 대형 고베타 동일가중": dict(fn=lhb),
        "대조: 대형 고베타 + 더 사기": dict(fn=lhb, pyramid=True),
        "대조: 대형 동일가중": dict(fn=large),
    }
    res, curves, logs = [], {}, {}
    for nm_, st in strategies.items():
        sch = sched(lambda g: g.index[st["fn"](g).fillna(False)].tolist())
        rr, lg = simulate(sch, pfn, ks, cost=COST, pyramid=st.get("pyramid", False), fallback="KOSPI")
        curves[nm_], logs[nm_] = rr, lg
        for seg, (a, b) in SEGK.items():
            m = perf(rr, rf, a, b)
            lgs = lg[(lg["t0"] >= pd.Timestamp(a) - pd.Timedelta(days=95)) & (lg["t0"] <= b)]
            x = rr[a:b]
            kx = ks.pct_change()[a:b].reindex(x.index)
            res.append(dict(전략=nm_, 구간=seg, **m, 평균종목수=lgs["n"].mean(), 지수대체분기=(lgs["sub"] > 0.99).sum(),
                            회전율=lgs["turnover"].mean(), 연초과=(x.mean() - kx.mean()) * 252))
    kr_ = ks.pct_change().dropna()["2015-04-01":]
    curves["코스피 지수"] = kr_
    for seg, (a, b) in SEGK.items():
        res.append(dict(전략="코스피 지수(가격지수)", 구간=seg, **perf(kr_, rf, a, b)))
    R = pd.DataFrame(res)
    R.to_csv(OUT / "idea4_kr.csv", index=False, encoding="utf-8-sig")
    logs["확신 프로필(코스피)"].to_csv(OUT / "idea4_kr_log.csv", index=False, encoding="utf-8-sig")
    # 보유 종목 이름
    names = []
    for g in snaps:
        ix = g.index[prof(g).fillna(False)]
        names.append(dict(t0=g["t0"].iloc[0], n=len(ix), 종목=",".join(L["name"].reindex(ix).fillna("").tolist())))
    pd.DataFrame(names).to_csv(OUT / "idea4_kr_names.csv", index=False, encoding="utf-8-sig")

    # 무작위 대조(분기 수익, 대형 고베타 후보 안)
    rng = np.random.default_rng(11)
    rows = []
    for g in snaps:
        t0 = g["t0"].iloc[0]
        t1 = t1_of[t0]
        qr = (c.loc[:t1].iloc[-1] / c.loc[t0] - 1).reindex(g.index)
        qr = qr.fillna(c.loc[t0:t1].ffill().iloc[-1].reindex(g.index) / c.loc[t0].reindex(g.index) - 1)
        pick = g.index[prof(g).fillna(False)]
        pool = qr[lhb(g).fillna(False)].dropna().values
        if len(pick) == 0 or len(pool) <= len(pick):
            continue
        st = qr.reindex(pick).mean()
        sims = np.array([pool[rng.choice(len(pool), len(pick), replace=False)].mean() for _ in range(1000)])
        rows.append(dict(t0=t0, st=st, sims=sims))
    D = pd.DataFrame(rows)
    rc = []
    for seg, (a, b) in SEGK.items():
        d = D[(D["t0"] >= pd.Timestamp(a) - pd.Timedelta(days=95)) & (D["t0"] <= b)]
        if len(d) == 0:
            continue
        sc = np.prod(1 + d["st"].values)
        sm_ = np.prod(1 + np.vstack(d["sims"].values), axis=0)
        rc.append(dict(구간=seg, 분기=len(d), 전략누적=sc - 1, 무작위중앙=np.median(sm_) - 1, 백분위=(sm_ < sc).mean() * 100))
    RC = pd.DataFrame(rc)
    RC.to_csv(OUT / "idea4_kr_random.csv", index=False, encoding="utf-8-sig")
    yr = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in curves.items()})
    yr.to_csv(OUT / "idea4_kr_yearly.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k, v in curves.items():
        if k in ("확신 프로필(코스피)", "확신 프로필 + 오른 종목 더 사기", "대조: 대형 고베타 동일가중", "대조: 대형 동일가중", "코스피 지수"):
            ax.plot((1 + v["2015-04-01":]).cumprod(), label=k, lw=1.3 if "프로필" in k else 1.0)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("아이디어 4: 확신 프로필을 코스피에 적용 (조건은 미국에서 고정)")
    plt.tight_layout()
    plt.savefig(CH / "idea4_kr.png", dpi=150)
    plt.close()
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))
    print(RC.round(3).to_string(index=False))
    print(yr.round(3).to_string())


if __name__ == "__main__":
    main()
