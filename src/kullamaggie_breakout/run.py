# -*- coding: utf-8 -*-
"""쿨라마기 EMA 돌파 검증 - 전체 실행 (같은 입력 → 같은 결과, 난수 시드 고정)

1. 기본 설정(사전 합의): EMA10 청산, ADR 4%·5% 둘 다, 모멘텀 상위 5%, 횡보 20일, 코스피 EMA20 국면 필터
2. 변형: EMA20 청산
3. 파라미터 격자 108개(EMA 2 × ADR 3 × 모멘텀 3 × 횡보 기간 3 × 국면 2) → 학습 구간 샤프로 1개 선택 → 검증 확인 → 테스트 1회
4. 조건 분해: 모멘텀+ADR만 ~ 전체 조건 (각 조건이 성과를 바꾸는가)
5. 무작위 진입 대조: (가) 같은 종목·같은 해, 진입일만 무작위 (나) 같은 날 모멘텀·ADR 후보 중 무작위 종목. 청산 규칙 동일
6. 강건성: 진입일 가정 3가지, 거래비용 2배, 2025~2026 제외, 상위 5% 거래 제외, 시장·유동성별
결과: results/*.csv, results/*.png, results/summary.json
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import backtest as bt
import data
import signals

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

RES = data.HERE / "results"
RES.mkdir(exist_ok=True)
SEED = 20261113
SEGS = data.SEGS
BASE = dict(ema=10, adr=0.04, top=0.05, n=20, regime=True)


def P(*a):
    print(*a, flush=True)
    data.log(" ".join(str(x) for x in a))


class Lab:
    def __init__(self):
        self.D = data.load()
        self.I = signals.Ind(self.D)
        D = self.D
        self.dates = D["dates"]
        self.rf = D["rf"]
        C = D["c"]
        ret = C / pd.DataFrame(C).ffill().shift(1).to_numpy() - 1
        lq = np.vstack([np.zeros((1, C.shape[1]), bool), self.I.liq[:-1]])
        ew = np.where(lq & ~np.isnan(ret), ret, np.nan)
        self.ew = pd.Series(np.nanmean(ew, axis=1), index=self.dates).fillna(0.0)
        self.kospi = D["kospi"].pct_change().fillna(0.0)
        self.kosdaq = D["kosdaq"].pct_change()

    def trades(self, ema=10, adr=0.04, top=0.05, n=20, regime=True, mode=1, cost=bt.COST, use_trigger=True, **kw):
        I, D = self.I, self.D
        s = I.setup(adr_min=adr, top=top, n=n, regime=regime, **kw)
        arr = bt.gen_trades(D["o"], D["h"], D["l"], D["c"], I.ema[ema], s, I.hh[n], I.adr, D["lim"], use_trigger, mode)
        tr = bt.to_frame(arr, self.dates, D["codes"], I.mom, cost=cost)
        tr = tr[tr.date >= data.START].reset_index(drop=True)
        tr["amt20"] = I.amt20[tr.t.values - 1, tr.n.values]
        return tr

    def run(self, cost=bt.COST, **p):
        tr = self.trades(cost=cost, **p)
        r, taken, st = bt.portfolio(tr, self.D["c"], self.rf, self.dates, cost=cost)
        r = r.loc[data.START:]
        tr["taken"] = taken
        return tr, r, st

    def seg_table(self, r, tr, name):
        rows = []
        for sg, (a, b) in SEGS.items():
            x = r.loc[a:b]
            pf = bt.perf(x, self.rf)
            al, t, be = bt.alpha_t(x, self.kospi, self.rf)
            al2, t2, _ = bt.alpha_t(x, self.ew, self.rf)
            tt = tr[(tr.date >= a) & (tr.date <= b)]
            ts = bt.trade_stats(tt)
            rows.append(dict(설정=name, 구간=sg, **pf, 알파_코스피=al, t_코스피=t, 베타=be, 알파_동일가중=al2, t_동일가중=t2,
                             거래수=ts.get("거래수"), 승률=ts.get("승률"), 평균수익=ts.get("평균수익"), 평균R=ts.get("평균R"), R_t=ts.get("R_t"),
                             평균보유일=ts.get("평균보유일"), 상위5pct제외평균R=ts.get("상위5pct제외평균R"), 상위5pct이익비중=ts.get("상위5pct이익비중")))
        return rows


def bench_table(L):
    rows = []
    for nm, r in [("코스피", L.kospi), ("코스닥", L.kosdaq), ("동일가중(유동성 통과 종목)", L.ew)]:
        for sg, (a, b) in SEGS.items():
            rows.append(dict(설정=nm, 구간=sg, **bt.perf(r.loc[a:b], L.rf)))
    return pd.DataFrame(rows)


def random_control(L, tr, k=20):
    """실제 거래와 같은 종목·같은 해에서 유동성 통과일 중 무작위 진입일 k개씩"""
    rng = np.random.default_rng(SEED)
    I, D = L.I, L.D
    years = pd.DatetimeIndex(L.dates).year.values
    liq = I.liq
    ns, ts, draw = [], [], []
    for row in tr.itertuples():
        yr = years[row.t]
        idx = np.where((years == yr) & liq[:, row.n])[0] + 1
        idx = idx[(idx < len(L.dates) - 1)]
        if not len(idx):
            continue
        pick = rng.choice(idx, size=k, replace=True)
        ns.extend([row.n] * k)
        ts.extend(pick.tolist())
        draw.extend(range(k))
    ns, ts, draw = np.array(ns, dtype=np.int64), np.array(ts, dtype=np.int64), np.array(draw)
    arr = bt.random_paths(D["o"], D["h"], D["l"], D["c"], I.e10, I.adr, ns, ts, D["lim"], 1)
    rt = bt.to_frame(np.column_stack([arr, np.zeros(len(arr))]), L.dates, D["codes"])
    ok = ~np.isnan(arr[:, 2])
    rt["draw"] = draw[ok]
    return rt


def main():
    t0 = time.time()
    L = Lab()
    D = L.D
    P(f"데이터 {len(L.dates)}일 × {len(D['codes'])}종목, 가격 단절 보정 {len(D['breaks'])}건")
    summary = dict(단절보정=int(len(D["breaks"])), 단절보정종목=int(D["breaks"].code.nunique()),
                   종목수=int(len(D["codes"])), 기간=[data.START, data.END])

    # 1·2. 기본 설정, ADR 5%, EMA20 변형
    seg_rows, curves, trades = [], {}, {}
    mains = {"기본(ADR4%·EMA10)": dict(BASE), "ADR5%·EMA10": dict(BASE, adr=0.05),
             "ADR4%·EMA20": dict(BASE, ema=20), "ADR5%·EMA20": dict(BASE, adr=0.05, ema=20)}
    for nm, p in mains.items():
        tr, r, st = L.run(**p)
        curves[nm], trades[nm] = r, tr
        seg_rows += [dict(x, 회전율=st["회전율"], 평균투자비중=st["평균투자비중"]) for x in L.seg_table(r, tr, nm)]
        P(nm, {k: round(float(v), 3) for k, v in bt.perf(r, L.rf).items()}, "거래", len(tr))
    main_tab = pd.DataFrame(seg_rows)
    main_tab.to_csv(RES / "main_segments.csv", index=False, encoding="utf-8-sig")
    bench = bench_table(L)
    bench.to_csv(RES / "benchmarks.csv", index=False, encoding="utf-8-sig")
    yr = pd.DataFrame({nm: bt.yearly(r) for nm, r in curves.items()})
    yr["코스피"] = bt.yearly(L.kospi.loc[data.START:])
    yr["코스닥"] = bt.yearly(L.kosdaq.loc[data.START:])
    yr["동일가중"] = bt.yearly(L.ew.loc[data.START:])
    yr.index.name = "연도"
    yr.to_csv(RES / "yearly.csv", encoding="utf-8-sig")
    trades["기본(ADR4%·EMA10)"].to_csv(RES / "trades_base.csv", index=False, encoding="utf-8-sig")
    trades["ADR5%·EMA10"].to_csv(RES / "trades_adr5.csv", index=False, encoding="utf-8-sig")

    # 3. 격자
    grid = []
    for ema in (10, 20):
        for adr in (0.03, 0.04, 0.05):
            for top in (0.02, 0.05, 0.10):
                for n in (10, 20, 40):
                    for rg in (True, False):
                        p = dict(ema=ema, adr=adr, top=top, n=n, regime=rg)
                        tr, r, st = L.run(**p)
                        for sg, (a, b) in SEGS.items():
                            x = r.loc[a:b]
                            al, t, _ = bt.alpha_t(x, L.kospi, L.rf)
                            tt = tr[(tr.date >= a) & (tr.date <= b)]
                            ts = bt.trade_stats(tt)
                            grid.append(dict(**p, 구간=sg, **bt.perf(x, L.rf), 알파=al, t=t, 거래수=ts.get("거래수", 0),
                                             평균R=ts.get("평균R"), R_t=ts.get("R_t"), 승률=ts.get("승률")))
    G = pd.DataFrame(grid)
    full = G[G.구간 == "전체"].copy()
    pv = 2 * (1 - stats.norm.cdf(full.t.abs()))
    order = np.argsort(pv)
    q = np.empty(len(pv))
    q[order] = np.minimum.accumulate((pv[order] * len(pv) / np.arange(1, len(pv) + 1))[::-1])[::-1]
    full["p"], full["BH_q"] = pv, np.minimum(q, 1)
    G = G.merge(full[["ema", "adr", "top", "n", "regime", "p", "BH_q"]], on=["ema", "adr", "top", "n", "regime"], how="left")
    G.to_csv(RES / "grid.csv", index=False, encoding="utf-8-sig")
    tr_g = G[(G.구간 == "학습") & (G.거래수 >= 200)].sort_values("샤프", ascending=False)
    best = tr_g.iloc[0][["ema", "adr", "top", "n", "regime"]].to_dict()
    best = dict(ema=int(best["ema"]), adr=float(best["adr"]), top=float(best["top"]), n=int(best["n"]), regime=bool(best["regime"]))
    tr, r, st = L.run(**best)
    curves["격자 선택"] = r
    sel = pd.DataFrame(L.seg_table(r, tr, "격자 선택"))
    sel.to_csv(RES / "grid_selected.csv", index=False, encoding="utf-8-sig")
    summary["격자선택"] = best
    summary["격자_전체기간_샤프양수"] = int((G[G.구간 == "전체"].샤프 > 0).sum())
    summary["격자_전체기간_BHq05"] = int((full.BH_q < 0.05).sum())
    summary["격자_테스트_평균R양수"] = int((G[G.구간 == "테스트"].평균R > 0).sum())
    summary["격자수"] = int(len(full))
    P("격자 선택", best, sel[["구간", "CAGR", "샤프", "평균R"]].round(3).to_dict("records"))

    # 4. 조건 분해
    abl = []
    steps = [("① 모멘텀+ADR만 (다음 날 시가 매수)", dict(use_trigger=False, use_align=False, use_tight=False, regime=False)),
             ("② + 돌파 진입", dict(use_trigger=True, use_align=False, use_tight=False, regime=False)),
             ("③ + 이동평균 정배열", dict(use_trigger=True, use_align=True, use_tight=False, regime=False)),
             ("④ + 횡보 압축", dict(use_trigger=True, use_align=True, use_tight=True, regime=False)),
             ("⑤ + 시장 국면 = 전체 전략", dict(use_trigger=True, use_align=True, use_tight=True, regime=True)),
             ("(변형) ⑤ + 엄격한 횡보", dict(use_trigger=True, use_align=True, use_tight=True, regime=True, strict=True))]
    for nm, kw in steps:
        p = dict(ema=10, adr=0.04, top=0.05, n=20)
        p.update(kw)
        tr_a, r_a, st_a = L.run(**p)
        for sg, (a, b) in SEGS.items():
            tt = tr_a[(tr_a.date >= a) & (tr_a.date <= b)]
            abl.append(dict(단계=nm, 구간=sg, **bt.trade_stats(tt), **{"포트_" + k: v for k, v in bt.perf(r_a.loc[a:b], L.rf).items()}))
    A = pd.DataFrame(abl)
    A.to_csv(RES / "ablation.csv", index=False, encoding="utf-8-sig")
    P("조건 분해", A[A.구간 == "전체"][["단계", "거래수", "평균R", "R_t", "포트_CAGR"]].round(3).to_dict("records"))

    # 5. 무작위 진입 대조
    base_tr = trades["기본(ADR4%·EMA10)"]
    rt = random_control(L, base_tr, k=20)
    dm = rt.groupby("draw").R.mean()
    act = base_tr.R.mean()
    rc = dict(실제평균R=act, 무작위평균R=dm.mean(), 무작위_5pct=dm.quantile(0.05), 무작위_95pct=dm.quantile(0.95),
              백분위=float((dm < act).mean()), 실제승률=(base_tr.ret > 0).mean(), 무작위승률=(rt.ret > 0).mean(),
              실제평균수익=base_tr.ret.mean(), 무작위평균수익=rt.ret.mean())
    # 진입 신호 + 무작위 : 차이의 t값 (거래별 같은 종목·해의 무작위 평균과 짝)
    pd.Series(rc).to_csv(RES / "random_control.csv", encoding="utf-8-sig")
    rt.groupby("draw").agg(평균R=("R", "mean"), 평균수익=("ret", "mean"), 거래=("R", "size")).to_csv(RES / "random_draws.csv", encoding="utf-8-sig")
    # (나) 같은 날 후보 풀: 실제 거래일마다 그날 진입 가능한 모멘텀·ADR 후보(국면 필터 통과일) 중 무작위 종목을 시가에 매수
    rng = np.random.default_rng(SEED + 1)
    pool = L.I.setup(adr_min=0.04, top=0.05, n=20, regime=True, use_align=False, use_tight=False)
    ns2, ts2, dr2 = [], [], []
    for row in base_tr.itertuples():
        cand = np.where(pool[row.t - 1])[0]
        if not len(cand):
            continue
        pick = rng.choice(cand, size=20, replace=True)
        ns2.extend(pick.tolist()); ts2.extend([row.t] * 20); dr2.extend(range(20))
    arr2 = bt.random_paths(D["o"], D["h"], D["l"], D["c"], L.I.e10, L.I.adr, np.array(ns2, dtype=np.int64),
                           np.array(ts2, dtype=np.int64), D["lim"], 1)
    ok2 = ~np.isnan(arr2[:, 2])
    rt2 = bt.to_frame(np.column_stack([arr2, np.zeros(len(arr2))]), L.dates, D["codes"])
    rt2["draw"] = np.array(dr2)[ok2]
    dm2 = rt2.groupby("draw").R.mean()
    rc.update(같은날후보_평균R=dm2.mean(), 같은날후보_5pct=dm2.quantile(0.05), 같은날후보_95pct=dm2.quantile(0.95),
              같은날후보_백분위=float((dm2 < act).mean()), 같은날후보_승률=(rt2.ret > 0).mean(), 같은날후보_평균수익=rt2.ret.mean())
    pd.Series(rc).to_csv(RES / "random_control.csv", encoding="utf-8-sig")
    summary["무작위대조"] = {k: float(v) for k, v in rc.items()}
    P("무작위 대조", {k: round(float(v), 4) for k, v in rc.items()})

    # 6. 강건성
    rob = []

    def add(nm, tr_x, r_x):
        for sg in ("학습", "검증", "테스트", "전체"):
            a, b = SEGS[sg]
            tt = tr_x[(tr_x.date >= a) & (tr_x.date <= b)]
            rob.append(dict(검증=nm, 구간=sg, **bt.perf(r_x.loc[a:b], L.rf), **bt.trade_stats(tt)))

    for mode, nm in [(1, "진입일 가정: 봉 방향(기본)"), (0, "진입일 가정: 보수"), (2, "진입일 가정: 진입일 손절 없음")]:
        tr_x, r_x, _ = L.run(**BASE, mode=mode)
        add(nm, tr_x, r_x)
    tr_x, r_x, _ = L.run(**BASE, strict=True)
    add("엄격한 횡보 정의(변형)", tr_x, r_x)
    tr_x, r_x, _ = L.run(**BASE, cost=bt.COST * 2)
    add("거래비용 2배(편도 0.5%)", tr_x, r_x)
    tr_x, r_x, _ = L.run(**BASE, cost=0.0)
    add("거래비용 0(참고)", tr_x, r_x)
    btr, br = trades["기본(ADR4%·EMA10)"], curves["기본(ADR4%·EMA10)"]
    ex = br.copy()
    ex.loc["2025-01-01":] = np.nan
    add("2025~2026 제외", btr[btr.date < "2025-01-01"], ex.dropna())
    k = max(1, int(len(btr) * 0.05))
    top_idx = btr.R.sort_values(ascending=False).index[:k]
    tr_nt = btr.drop(top_idx)
    r_nt, _, _ = bt.portfolio(tr_nt, D["c"], L.rf, L.dates)
    add("상위 5% 거래 제외", tr_nt, r_nt.loc[data.START:])
    R = pd.DataFrame(rob)
    R.to_csv(RES / "robust.csv", index=False, encoding="utf-8-sig")

    # 거래 단위 나눠 보기 (기본 설정)
    info = D["info"]
    btr = btr.copy()
    btr["시장"] = btr.code.map(info["market"]).fillna("정보 없음(옛 상장폐지)")
    btr["유동성"] = pd.qcut(btr.amt20.rank(method="first"), 3, labels=["하위1/3", "중간", "상위1/3"])
    btr["국면"] = np.where(L.I.regime[btr.t.values - 1], "코스피>EMA20", "코스피≤EMA20")
    btr["연도"] = btr.date.dt.year
    parts = []
    for col in ("시장", "유동성", "연도"):
        for v, g in btr.groupby(col, observed=True):
            parts.append(dict(구분=col, 값=str(v), **bt.trade_stats(g)))
    S = pd.DataFrame(parts)
    S.to_csv(RES / "breakdown.csv", index=False, encoding="utf-8-sig")
    # 종목별 기여
    cc = btr.groupby("code").ret.sum().sort_values(ascending=False)
    summary["상위5종목_이익합_비중"] = float(cc.head(5).clip(lower=0).sum() / cc.clip(lower=0).sum())
    summary["상위5pct거래_이익비중"] = float(bt.trade_stats(btr)["상위5pct이익비중"])

    # t=2 필요 연수 (테스트 구간 알파)
    tt = main_tab[(main_tab.설정 == "기본(ADR4%·EMA10)") & (main_tab.구간 == "테스트")].iloc[0]
    yrs_test = len(curves["기본(ADR4%·EMA10)"].loc[SEGS["테스트"][0]:]) / 252
    summary["테스트_알파t"] = float(tt.t_코스피)
    summary["t2_필요연수"] = float((2 / tt.t_코스피) ** 2 * yrs_test) if tt.t_코스피 > 0 else None
    summary["테스트연수"] = yrs_test

    charts(L, curves, yr, G, A, dm, act, btr, dm2)
    (RES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    P(f"완료 {time.time() - t0:.0f}초")


def charts(L, curves, yr, G, A, dm, act, btr, dm2):
    # 누적 수익
    fig, ax = plt.subplots(figsize=(10, 5))
    for nm, c in [("기본(ADR4%·EMA10)", "#1f4e79"), ("ADR5%·EMA10", "#2e86c1"), ("ADR4%·EMA20", "#7fb3d5")]:
        ax.plot((1 + curves[nm]).cumprod(), label=nm, color=c, lw=1.4)
    for nm, r, c in [("코스피", L.kospi, "#c0392b"), ("코스닥", L.kosdaq.fillna(0), "#e67e22"), ("동일가중", L.ew, "#7f8c8d")]:
        ax.plot((1 + r.loc[data.START:]).cumprod(), label=nm, color=c, lw=1.0, alpha=0.8)
    ax.set_yscale("log")
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    for d in ("2015-01-01", "2020-01-01"):
        ax.axvline(pd.Timestamp(d), color="k", ls=":", lw=0.8)
    ax.set_title("누적 수익 (로그, 2000-01 = 1, 점선: 학습|검증|테스트)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "equity.png", dpi=150)
    plt.close(fig)

    # 연도별
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(yr))
    ax.bar(x - 0.2, yr["기본(ADR4%·EMA10)"] * 100, 0.4, label="기본 전략", color="#1f4e79")
    ax.bar(x + 0.2, yr["코스피"] * 100, 0.4, label="코스피", color="#c0392b", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(yr.index, rotation=90, fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("%")
    ax.set_title("연도별 수익률")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(RES / "yearly.png", dpi=150)
    plt.close(fig)

    # R 분포
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(btr.R.clip(-3, 10), bins=80, color="#1f4e79")
    ax.axvline(btr.R.mean(), color="#c0392b", ls="--", label=f"평균 {btr.R.mean():.3f}R")
    ax.set_title("거래별 R 분포 (기본 설정, 10R 이상은 10으로 표시)")
    ax.set_xlabel("R (계획 위험 대비 손익)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "r_hist.png", dpi=150)
    plt.close(fig)

    # 격자 히트맵 (평균R, 구간별, EMA10·국면 필터)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, sg in zip(axes, ("학습", "검증", "테스트")):
        g = G[(G.구간 == sg) & (G.ema == 10) & (G.regime)]
        m = g.pivot_table(index="adr", columns=["top", "n"], values="평균R")
        im = ax.imshow(m.values, cmap="RdBu", vmin=-0.3, vmax=0.3, aspect="auto")
        ax.set_yticks(range(len(m.index)))
        ax.set_yticklabels([f"ADR {v:.0%}" for v in m.index])
        ax.set_xticks(range(len(m.columns)))
        ax.set_xticklabels([f"상위{t:.0%}\n{n}일" for t, n in m.columns], fontsize=7)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"{sg}: 평균 R")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle("파라미터 격자 (EMA10, 국면 필터 적용) — 모멘텀 상위% × 횡보 기간")
    fig.savefig(RES / "grid_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 조건 분해
    a = A[A.구간 == "전체"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(range(len(a)), a.평균R, color=["#c0392b" if v < 0 else "#1f4e79" for v in a.평균R])
    ax.set_yticks(range(len(a)))
    ax.set_yticklabels(a.단계, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6)
    for i, (v, n) in enumerate(zip(a.평균R, a.거래수)):
        ax.text(v, i, f" {v:+.3f}R ({n:,}건)", va="center", fontsize=8)
    ax.set_title("조건을 하나씩 더할 때 거래당 평균 R (2000~2026, 비용 차감)")
    fig.tight_layout()
    fig.savefig(RES / "ablation.png", dpi=150)
    plt.close(fig)

    # 무작위 대조
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dm, bins=12, color="#7f8c8d", alpha=0.8, label="(가) 같은 종목·같은 해, 날짜 무작위")
    ax.hist(dm2, bins=12, color="#e67e22", alpha=0.8, label="(나) 같은 날 모멘텀·ADR 후보 중 종목 무작위")
    ax.axvline(act, color="#1f4e79", lw=2, label=f"실제 신호 {act:.3f}R")
    ax.set_title("무작위 진입 20회 평균 R (청산 규칙 동일)")
    ax.set_xlabel("거래당 평균 R")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "random_control.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
