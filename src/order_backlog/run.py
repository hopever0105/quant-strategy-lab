# -*- coding: utf-8 -*-
"""수주잔고 전략 — 4) 실행: 본 규칙 + 비교 + 강건성 → results/, charts/. 실제 주문·매매 없음. KRX 요청 없음.

순서: python data.py → python run.py → python build_report.py
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
from signals import picks, ratio_hi, random_picks  # noqa: E402
from backtest import Prices, run, metrics, alpha, factors, rf_series, COST, BORROW  # noqa: E402

RES = HERE / "results"
CH = HERE / "charts"
RES.mkdir(exist_ok=True)
CH.mkdir(exist_ok=True)
DATA = HERE / "data"
SEG = {"us": {"검증 2019~2021": ("2019-04-01", "2021-12-31"), "테스트 2022~": ("2022-01-01", "2026-12-31"),
              "전체": ("2019-04-01", "2026-12-31")},
       "kr": {"학습 2016~2019": ("2016-04-01", "2019-12-31"), "검증 2020~2022": ("2020-01-01", "2022-12-31"),
              "테스트 2023~": ("2023-01-01", "2026-12-31"), "전체": ("2016-04-01", "2026-12-31")}}
NEUTRAL = {"us": True, "kr": False}


def summarize(mk, name, r, log, rf, fac, cols, drop2025=False):
    rows = []
    if name.startswith("B "):  # 롱숏은 자기자본이 거의 안 드는 구조 → 무위험 이자 빼지 않음
        rf = rf * 0
        fac = fac.assign(RF=0.0)
    x = r[r.index.year != 2025] if drop2025 else r
    for s, (a, b) in SEG[mk].items():
        m = metrics(x, rf, a, b)
        if not m:
            continue
        m.update(alpha(x, fac, a, b, cols))
        if log is not None and len(log):
            lg = log[(log["t0"] >= pd.Timestamp(a) - pd.Timedelta(days=95)) & (log["t0"] <= pd.Timestamp(b))]
            m["평균종목수"] = lg["n"].mean()
            m["분기회전율"] = lg["turnover"].mean()
            m["시세없어대체비중"] = lg["sub"].mean()
        rows.append(dict(시장=mk, 전략=name + (" (2025 제외)" if drop2025 else ""), 구간=s, **m))
    return rows


def market(mk):
    P = pd.read_pickle(DATA / f"{mk}_panel.pkl")
    P["t0"] = pd.to_datetime(P["t0"])
    P["t1"] = pd.to_datetime(P["t1"])
    px = Prices(mk)
    rf = rf_series(mk, px.bench.index)
    fac, cols = factors(mk, px.bench)
    nt = NEUTRAL[mk]
    out, curves, logs = [], {}, {}
    top = picks(P, 0.3, "top", nt)
    bot = picks(P, 0.3, "bottom", nt)
    allp = {q: g["code"].tolist() for q, g in P.groupby("t0")}
    rA, lA = run(top, P, px, 2, mk=mk)
    rB, lB = run(bot, P, px, 2, mk=mk)
    rE, lE = run(allp, P, px, 2, mk=mk)
    rLS = (rA - rB).dropna() - BORROW / 252
    bench = px.bench.pct_change().reindex(rA.index)
    curves = {"A 상위 30% 매수": rA, "하위 30% 매수(비교)": rB, "신호 있는 종목 동일가중": rE,
              "B 상위 매수 − 하위 공매도": rLS, ("SPY" if mk == "us" else "코스피"): bench}
    logs = {"A 상위 30% 매수": lA, "하위 30% 매수(비교)": lB, "신호 있는 종목 동일가중": lE}
    drops = [False, True] if mk == "kr" else [False]
    for nm, r in curves.items():
        for dp in drops:
            out += summarize(mk, nm, r, logs.get(nm), rf, fac, cols, dp)
    # 거래 수(매수 건수)
    n_buy = sum(len(v) for v in top.values())
    # 연도별
    yr = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in curves.items()})
    yr.to_csv(RES / f"{mk}_yearly.csv", encoding="utf-8-sig")

    # ---- 강건성 ----
    rob = []
    for tp in [0.2, 0.3, 0.4]:
        for H in [1, 2, 4]:
            ra, la = run(picks(P, tp, "top", nt), P, px, H, mk=mk)
            rb, _ = run(picks(P, tp, "bottom", nt), P, px, H, mk=mk)
            ls = (ra - rb).dropna() - BORROW / 252
            for dp in drops:
                for nm, r in [("A 매수", ra), ("B 롱숏", ls)]:
                    x = r[r.index.year != 2025] if dp else r
                    rf_, fac_ = (rf * 0, fac.assign(RF=0.0)) if nm.startswith("B") else (rf, fac)
                    m = metrics(x, rf_, *SEG[mk]["전체"])
                    m.update(alpha(x, fac_, *SEG[mk]["전체"], cols))
                    rob.append(dict(시장=mk, 버전=nm, 상위비율=tp, 보유분기=H, 제외2025=dp, **m))
    # 비용 2배
    r2, _ = run(top, P, px, 2, cost=COST[mk] * 2, mk=mk)
    for dp in drops:
        out += summarize(mk, "A 상위 30% 매수 (비용 2배)", r2, None, rf, fac, cols, dp)
    # 참고 변형: 잔고/매출 높은 절반 (미국만 — 한국은 잔고 금액 단위 잡음)
    if mk == "us":
        rR, lR = run(picks(P, 0.3, "top", nt, extra=ratio_hi), P, px, 2, mk=mk)
        out += summarize(mk, "참고: 상위 30% ∩ 잔고/매출 높은 절반", rR, lR, rf, fac, cols)
        curves["참고: 잔고/매출 높은 절반"] = rR
    # 무작위 대조 (분기 수익, 같은 분기·같은 유니버스에서 같은 개수 500회, 2분기 보유 근사)
    R = pd.read_pickle(DATA / f"{mk}_qret.pkl")
    rng = np.random.default_rng(7)
    nq = {q: len(v) for q, v in top.items()}

    def qret(pk):
        Q = sorted(pk)
        s = {}
        for i, q in enumerate(Q):
            sl = [pk[Q[i - k]] for k in range(2) if i - k >= 0]
            vals = []
            for names in sl:
                if names and q in R.index:
                    vals.append(R.loc[q].reindex(names).fillna(0).mean())
            s[q] = np.mean(vals) if vals else 0.0
        return pd.Series(s)

    st = qret(top)
    sims = np.vstack([qret(random_picks(P, nq, rng)).reindex(st.index).values for _ in range(500)])
    rc = []
    for s, (a, b) in SEG[mk].items():
        m = (st.index >= pd.Timestamp(a) - pd.Timedelta(days=95)) & (st.index <= pd.Timestamp(b))
        if m.sum() < 3:
            continue
        sc = np.prod(1 + st.values[m])
        sm_ = np.prod(1 + sims[:, m], axis=1)
        rc.append(dict(시장=mk, 구간=s, 분기=int(m.sum()), 전략누적=sc - 1, 무작위중앙=np.median(sm_) - 1, 백분위=(sm_ < sc).mean() * 100))
    pd.DataFrame(rc).to_csv(RES / f"{mk}_random.csv", index=False, encoding="utf-8-sig")
    print(f"[{mk}] 무작위 대조:", pd.DataFrame(rc)[["구간", "백분위"]].to_dict("records"))

    # 그림
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for k, v in curves.items():
        if k.startswith("참고"):
            continue
        ax.plot((1 + v.fillna(0)).cumprod(), label=k, lw=1.8 if k.startswith("A ") else 1.0,
                ls="--" if k.startswith("B ") else "-")
    if mk == "us":
        ax.axvline(pd.Timestamp("2022-01-01"), color="gray", ls=":", lw=0.8)
    else:
        for d in ["2020-01-01", "2023-01-01"]:
            ax.axvline(pd.Timestamp(d), color="gray", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title(("미국 S&P500" if mk == "us" else "한국 수주 업종") + " — 수주잔고 전략 누적 수익(로그, 점선: 구간 경계)")
    plt.tight_layout()
    plt.savefig(CH / f"{mk}_equity.png", dpi=150)
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    cols_y = ["A 상위 30% 매수", "신호 있는 종목 동일가중", "SPY" if mk == "us" else "코스피"]
    yr[cols_y].plot.bar(ax=ax, width=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("연도별 수익률")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CH / f"{mk}_yearly.png", dpi=150)
    plt.close()
    return out, rob, n_buy


def main():
    allo, allr = [], []
    info = []
    for mk in ["us", "kr"]:
        o, r, n = market(mk)
        allo += o
        allr += r
        info.append(f"{mk} 매수 건수(본 규칙, 묶음 기준) {n}")
    T = pd.DataFrame(allo)
    T.to_csv(RES / "main_results.csv", index=False, encoding="utf-8-sig")
    RB = pd.DataFrame(allr)
    RB.to_csv(RES / "robustness.csv", index=False, encoding="utf-8-sig")
    (RES / "info.txt").write_text("\n".join(info), encoding="utf-8")
    pd.set_option("display.width", 250)
    show = ["시장", "전략", "구간", "CAGR", "변동성", "샤프", "MDD", "알파(연)", "알파_t", "시장베타", "평균종목수", "분기회전율", "시세없어대체비중"]
    print(T[[c for c in show if c in T]].round(3).to_string(index=False))
    print(RB[["시장", "버전", "상위비율", "보유분기", "제외2025", "CAGR", "샤프", "MDD", "알파_t"]].round(3).to_string(index=False))
    print("\n".join(info))


if __name__ == "__main__":
    main()
