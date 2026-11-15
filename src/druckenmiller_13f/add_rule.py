# -*- coding: utf-8 -*-
"""15) '그가 비중을 늘린 종목' 프로필을 그 사람 없이 S&P500에 적용. 실제 매매 없음.

프로필(add_traits.py 결과): 1년 강한 상승 · 200일선 위 · 매출 고성장 · 베타 높음 (성공한 확대일수록 매출 성장·베타 높음)
규칙(분기말 t0, S&P500 그 시점 구성종목):
   12개월 수익률 백분위 ≥ M · 200일선 위 · 매출 성장(최근 4분기) 백분위 ≥ G · (베타 ≥ 1.0 선택) → 동일가중
   보유 H분기(겹치는 묶음; 그의 확대 따라 사기도 2~4분기가 나았음)
격자: M {60,70,80} × G {60,70,80} × 베타 {조건 없음, ≥1.0} × H {1,2,4} = 54
  → 분기 수익률로 빠르게 학습 샤프 계산·선택 → 선택안만 일별 엔진으로 정밀 계산
대조: ① 같은 분기 S&P500 전체에서 무작위 같은 개수 ② '모멘텀만'(12개월 ≥ M·200일선 위) 후보 안 무작위 → 매출 성장 조건이 더해주는지
시간 분할: 학습 2012~2018 / 검증 2019~2021 / 테스트 2022~ (테스트 한 번만). 비용 편도 0.1%.
결과: results/adds/rule_*.csv, charts/add_rule.png
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
from rule_bt import adj, alpha, ff_factors, load_panel, SEG  # noqa: E402
from features import load_px  # noqa: E402

OUT = HERE / "results" / "adds"
CH = HERE / "charts"
SEGS = ["학습", "검증", "테스트", "전체"]


def pick(g, M, G, beta):
    m = (g["pr_r12m"] >= M) & (g["vs200"] > 0) & (g["pr_rev_g_ttm"] >= G)
    if beta:
        m &= g["beta"] >= 1.0
    return g.loc[m.fillna(False), "yf"].tolist()


def seg_of(t):
    for s in ["학습", "검증", "테스트"]:
        a, b = SEG[s]
        if pd.Timestamp(a) - pd.Timedelta(days=95) <= t <= pd.Timestamp(b):
            return s
    return None


def quarterly_grid(S):
    """분기 수익(r_inq)으로 빠른 근사: H분기 보유 = 최근 H개 분기에 고른 묶음의 평균."""
    Q = sorted(S["t0"].unique())
    by = {q: S[S["t0"] == q].set_index("yf") for q in Q}
    rows = []
    for M, G, beta, H in itertools.product([60, 70, 80], [60, 70, 80], [False, True], [1, 2, 4]):
        picks = {q: pick(by[q].reset_index(), M, G, beta) for q in Q}
        rets = []
        for i, q in enumerate(Q):
            sl = []
            for k in range(H):
                if i - k < 0:
                    continue
                names = picks[Q[i - k]]
                r = by[q]["r_inq"].reindex(names).dropna()
                if len(r):
                    sl.append(r.mean())
            spy_q = by[q]["r_inq"].mean()  # 대체: 동일가중 시장
            rets.append(dict(t0=q, r=np.mean(sl) if sl else np.nan, n=len(picks[q])))
        d = pd.DataFrame(rets).dropna()
        d["seg"] = d["t0"].map(seg_of)
        for s, x in d.groupby("seg"):
            rows.append(dict(M=M, G=G, 베타=beta, H=H, 구간=s, 분기샤프=x["r"].mean() / x["r"].std() * 2,
                             분기평균=x["r"].mean(), 평균종목수=x["n"].mean()))
    return pd.DataFrame(rows)


def main():
    S = load_panel()
    spy = adj("SPY")
    rf = (load_px("^IRX")["close"] / 100 / 252).reindex(spy.index).ffill().fillna(0)
    ff = ff_factors()
    Gq = quarterly_grid(S)
    Gq.to_csv(OUT / "rule_grid_quarterly.csv", index=False, encoding="utf-8-sig")
    tr = Gq[(Gq["구간"] == "학습") & (Gq["평균종목수"] >= 8)].sort_values("분기샤프", ascending=False)
    b = tr.iloc[0]
    M, G, beta, H = int(b["M"]), int(b["G"]), bool(b["베타"]), int(b["H"])
    print(f"학습 선택: 12개월 백분위≥{M}, 매출성장 백분위≥{G}, 베타조건 {beta}, 보유 {H}분기", flush=True)
    piv = Gq.pivot_table(index=["M", "G", "베타", "H"], columns="구간", values="분기샤프")
    print("격자 학습 상위 10과 검증·테스트(분기 근사 샤프)\n", piv.sort_values("학습", ascending=False).head(10).round(2).to_string())
    print("격자 54개 중 검증>학습중앙 & 테스트>0 비율:", round(((piv["검증"] > 0) & (piv["테스트"] > 0)).mean(), 2))

    def sch(fn):
        return [(q0, g["t1"].iloc[0], {t: 1.0 for t in fn(g)}) for q0, g in S.groupby("t0")]

    res = []
    curves = {}
    variants = {
        "선택 규칙": (lambda g: pick(g, M, G, beta), H),
        "선택 규칙 1분기 보유": (lambda g: pick(g, M, G, beta), 1),
        "대조: 모멘텀만(12개월·200일선)": (lambda g: g.loc[((g["pr_r12m"] >= M) & (g["vs200"] > 0)).fillna(False), "yf"].tolist(), H),
        "대조: 매출 성장만": (lambda g: g.loc[(g["pr_rev_g_ttm"] >= G).fillna(False), "yf"].tolist(), H),
        "대조: S&P500 동일가중": (lambda g: g["yf"].tolist(), 1),
    }
    for nm, (fn, h) in variants.items():
        r, _ = simulate(sch(fn), adj, spy, sleeves=h)
        curves[nm] = r
        for s in SEGS:
            a, b_ = SEG[s]
            m = perf(r, rf, a, b_)
            m.update(alpha(r, ff, s))
            res.append(dict(전략=nm, 구간=s, **m))
    sp = spy.pct_change().dropna()["2011-10-01":]
    curves["SPY"] = sp
    for s in SEGS:
        a, b_ = SEG[s]
        m = perf(sp, rf, a, b_)
        res.append(dict(전략="SPY", 구간=s, **m))
    # 비용 2배
    r2, _ = simulate(sch(lambda g: pick(g, M, G, beta)), adj, spy, cost=0.002, sleeves=H)
    for s in SEGS:
        a, b_ = SEG[s]
        m = perf(r2, rf, a, b_)
        m.update(alpha(r2, ff, s))
        res.append(dict(전략="선택 규칙(비용 2배)", 구간=s, **m))
    R = pd.DataFrame(res)
    R.to_csv(OUT / "rule_results.csv", index=False, encoding="utf-8-sig")

    # 무작위 대조 (분기 수익, H분기 묶음 근사)
    rng = np.random.default_rng(5)
    Q = sorted(S["t0"].unique())
    by = {q: S[S["t0"] == q] for q in Q}
    out = []
    for pool_nm, pool_fn in [("S&P500 전체", lambda g: g.index), ("모멘텀만 후보", lambda g: g.index[((g["pr_r12m"] >= M) & (g["vs200"] > 0)).fillna(False)])]:
        rows = []
        for q in Q:
            g = by[q][by[q]["r_inq"].notna()]
            pk = g.index[g["yf"].isin(pick(g, M, G, beta))]
            pool = g.loc[pool_fn(g), "r_inq"].values
            if len(pk) == 0 or len(pool) <= len(pk):
                continue
            st = g.loc[pk, "r_inq"].mean()
            sims = np.array([pool[rng.choice(len(pool), len(pk), replace=False)].mean() for _ in range(1000)])
            rows.append(dict(t0=q, st=st, sims=sims))
        D = pd.DataFrame(rows)
        for s in SEGS:
            a, b_ = SEG[s]
            d = D[(D["t0"] >= pd.Timestamp(a) - pd.Timedelta(days=95)) & (D["t0"] <= b_)]
            sc = np.prod(1 + d["st"].values)
            sm_ = np.prod(1 + np.vstack(d["sims"].values), axis=0)
            out.append(dict(대조=pool_nm, 구간=s, 분기=len(d), 전략누적=sc - 1, 무작위중앙=np.median(sm_) - 1, 백분위=(sm_ < sc).mean() * 100))
    RC = pd.DataFrame(out)
    RC.to_csv(OUT / "rule_random.csv", index=False, encoding="utf-8-sig")
    yr = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in curves.items()})
    yr.to_csv(OUT / "rule_yearly.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k, v in curves.items():
        ax.plot((1 + v["2012-01-01":]).cumprod(), label=k, lw=1.5 if k == "선택 규칙" else 1.0)
    for s in ["검증", "테스트"]:
        ax.axvline(pd.Timestamp(SEG[s][0]), color="gray", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("'그가 비중을 늘린 종목' 프로필 규칙 (S&P500, 점선: 검증·테스트 시작)")
    plt.tight_layout()
    plt.savefig(CH / "add_rule.png", dpi=150)
    plt.close()
    pd.set_option("display.width", 250)
    show = ["전략", "구간", "CAGR", "변동성", "샤프", "MDD", "FF6_알파(연)", "FF6_t"]
    print(R[R["구간"].isin(SEGS)][show].round(3).to_string(index=False))
    print(RC.round(3).to_string(index=False))
    print(yr.round(3).to_string())


if __name__ == "__main__":
    main()
