# -*- coding: utf-8 -*-
"""12) 아이디어 1~3 검증 (미국). 실제 매매 없음. 시간 분할: 학습 2012~2018 / 검증 2019~2021 / 테스트 2022~

아이디어 1. 그가 '크게 산' 종목 따라 사기 (13F 공시일 매수)
   A 확대 + 비중 3% 이상 신규 / B 비중 3% 이상 신규만 / C 확대만  × 보유 1·2·4분기(겹치는 묶음)
   → 보유기간은 학습 구간 샤프로 고르고 검증·테스트 확인. 매개변수가 적어 과적합 위험 낮음.
아이디어 2. '오른 종목 더 사기'를 여러 전략에 붙여 보기
   기본 4개(13F 확신·확대 / 확신 프로필 / 대형 고베타 / S&P500 대형 동일가중) 각각 붙이기 전·후 비교
   → 기본마다·구간마다 좋아졌는지 센다(부품으로 쓸 만한지).
아이디어 3. 덜 오른 걸 사고 많이 오른 걸 팔기 (S&P500 그 시점 구성종목, 분기)
   12개월 수익률 백분위가 중간 띠 안 + 최근(1개월 또는 3개월) 수익률 상위 → 동일가중 1분기
   격자(띠 3 × 최근 지표 2 × 문턱 2 × 대상 3 = 36) 학습 구간에서 고르고, 같은 대상 무작위 1,000회 대조.
결과: results/ideas/*.csv, charts/idea*.png
"""
import itertools
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
from rule_bt import adj, alpha, ff_factors, load_panel, select, SEG  # noqa: E402
from features import load_px  # noqa: E402

OUT = HERE / "results" / "ideas"
OUT.mkdir(parents=True, exist_ok=True)
CH = HERE / "charts"
SEGS = ["학습", "검증", "테스트", "전체"]


def rows_for(name, r, rf, ff, log=None, extra=None):
    out = []
    for seg in SEGS:
        a, b = SEG[seg]
        m = perf(r, rf, a, b)
        if not m:
            continue
        m.update(alpha(r, ff, seg))
        if log is not None and len(log):
            lg = log[(log["t0"] >= pd.Timestamp(a) - pd.Timedelta(days=95)) & (log["t0"] <= b)]
            m["평균종목수"] = lg["n"].mean()
            m["회전율"] = lg["turnover"].mean()
        out.append(dict(전략=name, 구간=seg, **(extra or {}), **m))
    return out


def schedule_13f(kind):
    tr = pd.read_csv(HERE / "results" / "trades.csv", dtype={"cusip": str}, parse_dates=["period", "filed"])
    sec = pd.read_csv(HERE / "results" / "securities.csv", dtype={"cusip": str})[["cusip", "yf", "kind"]]
    tr = tr[tr["leg"] == "주식"].merge(sec, on="cusip", how="left")
    tr = tr[tr["kind"] != "ETF"]
    per = sorted(tr["period"].unique())
    filed = tr.groupby("period")["filed"].first()
    sch = []
    for i, p in enumerate(per):
        g = tr[tr["period"] == p]
        if kind == "A":
            m = (g["action"] == "확대") | ((g["action"] == "신규") & (g["weight"] >= 0.03))
        elif kind == "B":
            m = (g["action"] == "신규") & (g["weight"] >= 0.03)
        else:
            m = g["action"] == "확대"
        names = g.loc[m, "yf"].dropna().unique()
        t1 = filed[per[i + 1]] if i + 1 < len(per) else pd.Timestamp("2026-11-12")
        sch.append((filed[p], t1, {t: 1.0 for t in names}))
    return sch


def schedule_sp(S, fn):
    sch = []
    for q0, g in S.groupby("t0"):
        sch.append((q0, g["t1"].iloc[0], {t: 1.0 for t in fn(g)}))
    return sch


def main():
    spy = adj("SPY")
    rf = (load_px("^IRX")["close"] / 100 / 252).reindex(spy.index).ffill().fillna(0)
    ff = ff_factors()
    S = load_panel()
    res = []

    # ---------- 아이디어 1 ----------
    print("== 아이디어 1: 크게 산 종목 따라 사기 ==", flush=True)
    curves1 = {}
    for kind, lab in [("A", "확대+3%이상 신규"), ("B", "3%이상 신규만"), ("C", "확대만")]:
        sch = schedule_13f(kind)
        for H in [1, 2, 4]:
            r, lg = simulate(sch, adj, spy, pyramid=False, sleeves=H)
            if lg is None:
                lg = pd.DataFrame({"t0": [s[0] for s in sch], "n": [len(s[2]) for s in sch], "turnover": np.nan})
            res += rows_for(f"1-{kind} {lab} / 보유 {H}분기", r, rf, ff, lg, dict(아이디어=1, 묶음=kind, 보유=H))
            curves1[(kind, H)] = r
    sp = spy.pct_change().dropna()["2012-02-15":]
    res += rows_for("SPY", sp, rf, ff, None, dict(아이디어=0))

    # ---------- 아이디어 2 ----------
    print("== 아이디어 2: 오른 종목 더 사기 ==", flush=True)
    grid = pd.read_csv(HERE / "results" / "rule" / "grid.csv")
    tr_ = grid[(grid["구간"] == "학습") & (grid["평균종목수"] >= 5)].sort_values("샤프", ascending=False).iloc[0]
    P0 = dict(B=tr_["B"], H=tr_["H"], G=tr_["G"], M=tr_["M"])
    bases = {
        "13F 확신·확대": schedule_13f("A"),
        "확신 프로필": schedule_sp(S, lambda g: select(g, P0)),
        "대형 고베타": schedule_sp(S, lambda g: g.loc[(g["pr_mcap"] > 70) & (g["beta"] >= 1.0), "yf"].tolist()),
        "S&P500 대형 동일가중": schedule_sp(S, lambda g: g.loc[g["pr_mcap"] > 70, "yf"].tolist()),
    }
    for bn, sch in bases.items():
        for pyr in [False, True]:
            r, lg = simulate(sch, adj, spy, pyramid=pyr)
            res += rows_for(f"2 {bn} / {'더 사기' if pyr else '기본'}", r, rf, ff, lg, dict(아이디어=2, 기본=bn, 더사기=pyr))

    # ---------- 아이디어 3 ----------
    print("== 아이디어 3: 덜 오른 걸 사고 많이 오른 걸 팔기 ==", flush=True)
    U = {"S&P500 전체": lambda g: pd.Series(True, index=g.index),
         "대형 고베타": lambda g: (g["pr_mcap"] > 70) & (g["beta"] >= 1.0),
         "성장주(매출성장5%·총이익률40%)": lambda g: (g["rev_g_ttm"] >= 0.05) & (g["gross_margin"] >= 0.40)}

    def sel3(g, band, rec, th, uni):
        u = U[uni](g).fillna(False)
        p12 = g["pr_r12m"]
        pr = g[f"pr_{rec}"]
        m = u & (p12 >= band[0]) & (p12 <= band[1]) & (pr >= th)
        return g.loc[m.fillna(False), "yf"].tolist()

    g3 = []
    cache3 = {}
    for band, rec, th, uni in itertools.product([(30, 70), (20, 60), (40, 80)], ["r1m", "r3m"], [70, 80], list(U)):
        sch = schedule_sp(S, lambda g: sel3(g, band, rec, th, uni))
        r, lg = simulate(sch, adj, spy)
        key = (band, rec, th, uni)
        cache3[key] = (r, lg)
        for seg in SEGS:
            a, b = SEG[seg]
            m = perf(r, rf, a, b)
            lgs = lg[(lg["t0"] >= pd.Timestamp(a) - pd.Timedelta(days=95)) & (lg["t0"] <= b)]
            g3.append(dict(띠=f"{band[0]}~{band[1]}", 최근=rec, 문턱=th, 대상=uni, 구간=seg, **m, 평균종목수=lgs["n"].mean()))
    G3 = pd.DataFrame(g3)
    G3.to_csv(OUT / "idea3_grid.csv", index=False, encoding="utf-8-sig")
    best3 = G3[(G3["구간"] == "학습") & (G3["평균종목수"] >= 5)].sort_values("샤프", ascending=False).iloc[0]
    k3 = (tuple(int(x) for x in best3["띠"].split("~")), best3["최근"], int(best3["문턱"]), best3["대상"])
    r3, lg3 = cache3[k3]
    res += rows_for(f"3 선택: 12개월 백분위 {best3['띠']} + {best3['최근']} 상위 {100 - int(best3['문턱'])}% / {best3['대상']}",
                    r3, rf, ff, lg3, dict(아이디어=3))
    # 같은 대상 동일가중(대조)
    ru, lgu = simulate(schedule_sp(S, lambda g: g.loc[U[k3[3]](g).fillna(False), "yf"].tolist()), adj, spy)
    res += rows_for(f"3 대조: {k3[3]} 동일가중", ru, rf, ff, lgu, dict(아이디어=3))
    # 무작위 1,000회 (분기 수익률 기준)
    rng = np.random.default_rng(7)
    rr = []
    for q0, g in S.groupby("t0"):
        g = g[g["r_inq"].notna()]
        pick = sel3(g, *k3)
        pool = g.loc[U[k3[3]](g).fillna(False), "r_inq"].values
        if len(pick) == 0 or len(pool) <= len(pick):
            continue
        st = g.set_index("yf")["r_inq"].reindex(pick).mean()
        sims = np.array([pool[rng.choice(len(pool), len(pick), replace=False)].mean() for _ in range(1000)])
        rr.append(dict(t0=q0, st=st, sims=sims))
    D = pd.DataFrame(rr)
    rc = []
    for seg in SEGS:
        a, b = SEG[seg]
        d = D[(D["t0"] >= pd.Timestamp(a) - pd.Timedelta(days=95)) & (D["t0"] <= b)]
        sc = np.prod(1 + d["st"].values)
        sm_ = np.prod(1 + np.vstack(d["sims"].values), axis=0)
        rc.append(dict(구간=seg, 분기=len(d), 전략누적=sc - 1, 무작위중앙=np.median(sm_) - 1, 백분위=(sm_ < sc).mean() * 100))
    RC = pd.DataFrame(rc)
    RC.to_csv(OUT / "idea3_random.csv", index=False, encoding="utf-8-sig")

    R = pd.DataFrame(res)
    R.to_csv(OUT / "ideas_us.csv", index=False, encoding="utf-8-sig")

    # 차트
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for (kind, H), r in curves1.items():
        if kind == "A":
            ax.plot((1 + r["2012-02-15":]).cumprod(), label=f"확대+3%이상 신규, 보유 {H}분기", lw=1.2)
    ax.plot((1 + sp).cumprod(), label="SPY", color="k", lw=1)
    ax.plot((1 + r3["2012-02-15":]).cumprod(), label="아이디어 3 선택 규칙", lw=1.2, ls="--")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("아이디어 1·3 누적 수익 (로그)")
    plt.tight_layout()
    plt.savefig(CH / "ideas_us.png", dpi=150)
    plt.close()

    pd.set_option("display.width", 260)
    show = ["전략", "구간", "CAGR", "샤프", "MDD", "FF6_알파(연)", "FF6_t", "평균종목수"]
    for i in [1, 0, 2, 3]:
        x = R[R["아이디어"] == i]
        print(x[x["구간"].isin(SEGS)][show].round(3).to_string(index=False))
    print("\n아이디어 3 무작위 대조\n", RC.round(3).to_string(index=False))
    print("\n아이디어 3 격자 상위(학습)\n", G3[G3["구간"] == "학습"].sort_values("샤프", ascending=False).head(8).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
