# -*- coding: utf-8 -*-
"""6) 드러켄밀러가 산 종목의 공통 특징 찾기.

A. 포트폴리오 개요: 분기 보유 수·집중도·회전·보유기간·옵션·ETF/ADR
B. 업종: 매수 업종 비중 vs S&P500 업종 비중, 연도별 업종 비중
C. 특징: 매수(신규·확대, 주식, ETF 제외)를 같은 분기 S&P500 안에서 백분위로 → 평균 백분위,
   분기별 평균의 t값(분기 간 독립 가정), 상위 20%에 속하는 비율(기준 20%)
   + 매도(축소·청산)와 비교
D. 승자/패자: 매수 후 12개월 SPY 초과수익 상위 1/3 vs 하위 1/3 특징 차이
E. 매크로: 분기 업종 순매수와 유가·달러·금리·S&P500 추세의 관계
F. 조합: 특징을 겹쳤을 때 S&P500 중 '그가 산 종목'일 확률 배율(lift)
결과: results/analysis/*.csv, charts/*.png, results/summary.txt
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
AN = OUT / "analysis"
CH = HERE / "charts"
AN.mkdir(exist_ok=True)
CH.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
from features import load_px  # noqa: E402

FEATS = [
    ("r1m", "1개월 수익률"), ("r3m", "3개월 수익률"), ("r6m", "6개월 수익률"), ("r12m", "12개월 수익률"),
    ("r12_1", "12-1 모멘텀"), ("dist52h", "52주 고점 대비"), ("dist52l", "52주 저점 대비"),
    ("vs200", "200일선 대비"), ("ma50_200", "50일선/200일선"),
    ("vol60", "60일 변동성"), ("beta", "베타"), ("ivol", "잔차 변동성"), ("mdd1y", "1년 최대낙폭"),
    ("mcap", "시가총액"), ("dollar_vol", "거래대금"),
    ("rev_g_ttm", "매출 성장(TTM)"), ("rev_g_q", "매출 성장(최근 분기)"), ("rev_accel", "매출 가속"),
    ("op_g_ttm", "영업이익 성장(TTM)"), ("ni_g_ttm", "순이익 성장(TTM)"), ("op_accel", "영업이익 가속"),
    ("gross_margin", "매출총이익률"), ("op_margin", "영업이익률"), ("net_margin", "순이익률"),
    ("fcf_margin", "FCF 마진"), ("roe", "ROE"), ("roa", "ROA"), ("rnd_sales", "R&D/매출"),
    ("debt_equity", "부채/자본"), ("liab_assets", "부채/자산"),
    ("ps", "PSR"), ("ey", "이익수익률(1/PER)"), ("bm", "장부가/시총(1/PBR)"), ("fcf_yield", "FCF 수익률"),
]
LBL = dict(FEATS)
BUY = ["신규", "확대"]
SELL = ["축소", "청산"]


def pct_rank(panel, col):
    """같은 분기 S&P500 분포 안에서의 백분위(0~100). 드러켄밀러 종목도 S&P500 분포 기준."""
    out = pd.Series(np.nan, index=panel.index)
    for p, g in panel.groupby("period"):
        ref = g.loc[g["src"] == "S&P500", col].dropna().values
        if len(ref) < 50:
            continue
        ref = np.sort(ref)
        v = g[col]
        m = v.notna()
        out.loc[v[m].index] = np.searchsorted(ref, v[m].values, side="right") / len(ref) * 100
    return out


def feat_table(panel, mask, name):
    rows = []
    mask = pd.Series(np.asarray(mask), index=panel.index)
    for c, lb in FEATS:
        if c not in panel:
            continue
        pr = panel.loc[mask, f"pr_{c}"]
        n = pr.notna().sum()
        if n < 15:
            continue
        q = panel.loc[mask & panel[f"pr_{c}"].notna()].groupby("period")[f"pr_{c}"].mean()
        t = stats.ttest_1samp(q, 50).statistic if len(q) > 5 else np.nan
        raw = panel.loc[mask, c]
        rows.append(dict(특징=lb, 코드=c, 표본=n, 평균백분위=pr.mean(), 중앙백분위=pr.median(),
                         상위20비율=(pr > 80).mean() * 100, 하위20비율=(pr <= 20).mean() * 100,
                         분기수=len(q), t값=t, 그룹중앙값=raw.median(),
                         SP500중앙값=panel.loc[panel["src"] == "S&P500", c].median()))
    d = pd.DataFrame(rows).sort_values("t값", key=lambda s: -s.abs())
    d.to_csv(AN / f"features_{name}.csv", index=False, encoding="utf-8-sig")
    return d


def main():
    P = pd.read_pickle(OUT / "panel.pkl")
    P["period"] = pd.to_datetime(P["period"])
    # ADR 등 주식 수 단위가 어긋난 시총(5조$ 초과·1억$ 미만)은 시총 기반 지표 제외
    badm = (P["mcap"] > 5e12) | (P["mcap"] < 1e8)
    P.loc[badm, ["mcap", "ps", "ey", "bm", "fcf_yield"]] = np.nan
    for c, _ in FEATS:
        if c in P:
            P[f"pr_{c}"] = pct_rank(P, c)
    E = P[P["src"] == "드러켄밀러"].copy()
    S = P[P["src"] == "S&P500"].copy()
    stock = (E["leg"] == "주식") & (E["kind"] != "ETF")
    buy = stock & E["action"].isin(BUY)
    new = stock & (E["action"] == "신규")
    add = stock & (E["action"] == "확대")
    sell = stock & E["action"].isin(SELL)
    L = []

    def say(s=""):
        print(s)
        L.append(str(s))

    # ---------- A. 개요 ----------
    pos = pd.read_csv(OUT / "positions.csv", parse_dates=["period"])
    ps = pos[pos["leg"] == "주식"]
    g = ps.groupby("period")
    ov = pd.DataFrame({
        "보유수": g.size(), "평가액(백만$)": g["value"].sum() / 1e6,
        "상위10비중": g["weight"].apply(lambda w: w.nlargest(10).sum()),
        "최대비중": g["weight"].max(),
    })
    trs = E[E["leg"] == "주식"]
    cnt = trs.groupby(["period", "action"]).size().unstack(fill_value=0)
    ov = ov.join(cnt)
    # 회전율: 분기 매수 비중 합(신규+확대분)
    trs = trs.assign(dw=(trs["weight"] - trs["weight_prev"]))
    ov["매수비중"] = trs[trs["action"].isin(BUY)].groupby("period")["dw"].sum()
    ov["매도비중"] = -trs[trs["action"].isin(SELL)].groupby("period")["dw"].sum()
    ov.to_csv(AN / "overview_by_quarter.csv", encoding="utf-8-sig")
    say("== A. 포트폴리오 개요 (2011Q4~2026Q2, 59분기) ==")
    say(f"분기 평균 보유 {ov['보유수'].mean():.0f}종목(최소 {ov['보유수'].min()}, 최대 {ov['보유수'].max()}), "
        f"상위10 비중 평균 {ov['상위10비중'].mean():.0%}, 최대 단일비중 평균 {ov['최대비중'].mean():.0%}")
    say(f"분기 평균 신규 {ov.get('신규', pd.Series(0)).mean():.1f}, 확대 {ov.get('확대', pd.Series(0)).mean():.1f}, "
        f"축소 {ov.get('축소', pd.Series(0)).mean():.1f}, 청산 {ov.get('청산', pd.Series(0)).mean():.1f}")
    say(f"분기 매수 비중(포트폴리오 대비) 평균 {ov['매수비중'].mean():.0%}, 매도 {ov['매도비중'].mean():.0%}")
    # 보유기간: 신규→청산 분기 수
    hold = []
    for (c, leg), gg in E[E["leg"] == "주식"].sort_values("period").groupby(["cusip", "leg"]):
        start = None
        for _, r in gg.iterrows():
            if r["action"] in ("신규", "최초보유"):
                start = r["period"]
            if r["action"] == "청산" and start is not None:
                hold.append(dict(cusip=c, name=r["name"], start=start, end=r["period"],
                                 q=round((r["period"] - start).days / 91.3)))
                start = None
    hold = pd.DataFrame(hold)
    hold.to_csv(AN / "holding_periods.csv", index=False, encoding="utf-8-sig")
    say(f"보유기간(신규→청산, {len(hold)}건): 중앙값 {hold['q'].median():.0f}분기, "
        f"1분기 만에 청산 {(hold['q'] <= 1).mean():.0%}, 4분기 이상 {(hold['q'] >= 4).mean():.0%}, 8분기 이상 {(hold['q'] >= 8).mean():.0%}")
    kinds = E[E["action"].isin(BUY)].groupby(["leg", "kind"]).size()
    say("매수 건수(종류별): " + ", ".join(f"{a}/{b} {n}" for (a, b), n in kinds.items()))
    say(f"매수 중 S&P500 종목 비율 {E.loc[buy, 'in_sp500'].mean():.0%}")

    # 비중 상위 종목(역대 최대 비중)
    top = ps.sort_values("weight", ascending=False).drop_duplicates("cusip").head(25)
    top[["period", "name", "weight", "value"]].to_csv(AN / "top_positions.csv", index=False, encoding="utf-8-sig")

    # ---------- B. 업종 ----------
    say("\n== B. 업종 ==")
    bs = E.loc[buy, "sector"].value_counts(normalize=True)
    ss = S["sector"].value_counts(normalize=True)
    # 금액 기준 업종 비중(보유)
    hv = E[(E["leg"] == "주식") & (E["shares"] > 0)]
    hv_sec = hv.groupby("sector")["value"].sum() / hv["value"].sum()
    sec_tab = pd.DataFrame({"매수건수비중": bs, "보유금액비중": hv_sec, "S&P500종목비중": ss}).fillna(0)
    sec_tab["배율(매수/S&P)"] = sec_tab["매수건수비중"] / sec_tab["S&P500종목비중"].replace(0, np.nan)
    sec_tab = sec_tab.sort_values("보유금액비중", ascending=False)
    sec_tab.to_csv(AN / "sector_share.csv", encoding="utf-8-sig")
    for s, r in sec_tab.iterrows():
        say(f"  {s}: 매수 {r['매수건수비중']:.0%} / 보유금액 {r['보유금액비중']:.0%} / S&P500 {r['S&P500종목비중']:.0%}")
    hv = hv.assign(year=hv["period"].dt.year)
    yr = hv.groupby(["year", "sector"])["value"].sum().unstack(fill_value=0)
    yr = yr.div(yr.sum(1), axis=0)
    yr.to_csv(AN / "sector_by_year.csv", encoding="utf-8-sig")
    ax = yr[sec_tab.index[:9]].plot(kind="bar", stacked=True, figsize=(11, 5), colormap="tab10", width=0.85)
    ax.set_ylabel("보유 금액 비중")
    ax.set_xlabel("")
    ax.set_title("연도별 업종 비중 (13F 보유 금액, ETF 포함)")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    plt.savefig(CH / "sector_by_year.png", dpi=150)
    plt.close()

    # ---------- C. 특징 ----------
    say("\n== C. 매수 직전 특징 (같은 분기 S&P500 안 백분위, 50=중간) ==")
    tb = feat_table(P, P.index.isin(E[buy].index), "buy")
    tn = feat_table(P, P.index.isin(E[new].index), "new")
    ta = feat_table(P, P.index.isin(E[add].index), "add")
    tsl = feat_table(P, P.index.isin(E[sell].index), "sell")
    comp = tb.set_index("코드")[["특징", "표본", "평균백분위", "상위20비율", "t값"]].join(
        tn.set_index("코드")[["평균백분위"]].rename(columns={"평균백분위": "신규"})).join(
        ta.set_index("코드")[["평균백분위"]].rename(columns={"평균백분위": "확대"})).join(
        tsl.set_index("코드")[["평균백분위", "t값"]].rename(columns={"평균백분위": "매도", "t값": "매도t"}))
    comp.to_csv(AN / "features_compare.csv", encoding="utf-8-sig")
    insp = E["in_sp500"].astype(bool)
    say(f"  흑자(최근 4분기 순이익>0) 비율: 매수 {E.loc[buy, 'profitable'].mean():.0%} vs S&P500 {S['profitable'].mean():.0%}")
    say(f"  매수 중 S&P500 밖 종목 {1 - insp[buy].mean():.0%}, 그 시총 중앙값 {E.loc[buy & ~insp, 'mcap'].median() / 1e9:.1f}십억$ "
        f"(S&P500 중앙값 {S['mcap'].median() / 1e9:.1f}십억$)")
    for c, r in comp.iterrows():
        say(f"  {r['특징']:<14} 매수 {r['평균백분위']:5.1f} (t={r['t값']:+.1f}, 상위20% {r['상위20비율']:.0f}%) | "
            f"신규 {r['신규']:5.1f} 확대 {r['확대']:5.1f} 매도 {r['매도']:5.1f} | n={r['표본']}")
    # 차트: 매수 평균 백분위
    cc = comp.dropna(subset=["t값"]).sort_values("평균백분위")
    fig, ax = plt.subplots(figsize=(9, 10))
    col = ["#c0392b" if abs(t) >= 3 else ("#e67e22" if abs(t) >= 2 else "#95a5a6") for t in cc["t값"]]
    ax.barh(cc["특징"], cc["평균백분위"] - 50, left=50, color=col)
    ax.axvline(50, color="k", lw=0.8)
    ax.set_xlim(20, 80)
    ax.set_xlabel("S&P500 안 평균 백분위 (빨강 |t|≥3, 주황 |t|≥2)")
    ax.set_title("드러켄밀러 매수 종목의 매수 직전 특징")
    plt.tight_layout()
    plt.savefig(CH / "buy_feature_percentiles.png", dpi=150)
    plt.close()

    # 분포 비교 차트(대표 특징 4개)
    keyf = ["dist52h", "r6m", "rev_g_ttm", "mcap"]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, c in zip(axs, keyf):
        a = E.loc[buy, c].dropna()
        b = S[c].dropna()
        if c == "mcap":
            a, b = np.log10(a[a > 0]), np.log10(b[b > 0])
            ax.set_xlabel("log10 시가총액($)")
        lo, hi = np.nanpercentile(np.r_[a, b], [2, 98])
        bins = np.linspace(lo, hi, 40)
        ax.hist(b.clip(lo, hi), bins=bins, density=True, alpha=0.5, label="S&P500", color="#7f8c8d")
        ax.hist(a.clip(lo, hi), bins=bins, density=True, alpha=0.6, label="드러켄밀러 매수", color="#c0392b")
        ax.set_title(LBL[c], fontsize=10)
    axs[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(CH / "buy_vs_sp500_dist.png", dpi=150)
    plt.close()

    # ---------- D. 매수 후 성과, 승자/패자 ----------
    say("\n== D. 매수 후 성과 (SPY 초과, 분기말 기준 / 공시일 기준) ==")
    perf = []
    for h in ["3m", "6m", "12m"]:
        for tag, nm in [("t1", "분기말"), ("tf", "공시일")]:
            x = E.loc[buy, f"f{h}_{tag}_x"].dropna()
            q = E.loc[buy].groupby("period")[f"f{h}_{tag}_x"].mean().dropna()
            perf.append(dict(기간=h, 기준=nm, 표본=len(x), 평균=x.mean(), 중앙=x.median(), 승률=(x > 0).mean(),
                             분기평균t=stats.ttest_1samp(q, 0).statistic if len(q) > 5 else np.nan))
    for lab, msk in [("신규", new), ("확대", add), ("매도", sell)]:
        x = E.loc[msk, "f12m_t1_x"].dropna()
        perf.append(dict(기간="12m", 기준=f"분기말·{lab}", 표본=len(x), 평균=x.mean(), 중앙=x.median(), 승률=(x > 0).mean()))
    perf = pd.DataFrame(perf)
    perf.to_csv(AN / "post_buy_performance.csv", index=False, encoding="utf-8-sig")
    for _, r in perf.iterrows():
        say(f"  {r['기간']} {r['기준']}: 평균 {r['평균']:+.1%} 중앙 {r['중앙']:+.1%} 승률 {r['승률']:.0%} n={r['표본']}"
            + (f" t={r['분기평균t']:+.1f}" if pd.notna(r.get('분기평균t')) else ""))
    B = E[buy & E["f12m_t1_x"].notna()].copy()
    B["tercile"] = B.groupby("period")["f12m_t1_x"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["하", "중", "상"]) if len(s) >= 3 else np.nan)
    wl = []
    for c, lb in FEATS:
        pc = f"pr_{c}"
        if pc not in B:
            continue
        a = B.loc[B["tercile"] == "상", pc].dropna()
        b = B.loc[B["tercile"] == "하", pc].dropna()
        if len(a) < 15 or len(b) < 15:
            continue
        t = stats.ttest_ind(a, b, equal_var=False).statistic
        wl.append(dict(특징=lb, 코드=c, 승자백분위=a.mean(), 패자백분위=b.mean(), 차이=a.mean() - b.mean(), t값=t, n승=len(a), n패=len(b)))
    wl = pd.DataFrame(wl).sort_values("t값", key=lambda s: -s.abs())
    wl.to_csv(AN / "winners_vs_losers.csv", index=False, encoding="utf-8-sig")
    say("\n  승자(12개월 초과 상위 1/3) vs 패자(하위 1/3) 매수 직전 특징 차이 (|t|≥2):")
    for _, r in wl[wl["t값"].abs() >= 2].iterrows():
        say(f"   {r['특징']}: 승자 {r['승자백분위']:.0f} vs 패자 {r['패자백분위']:.0f} (t={r['t값']:+.1f})")
    # 업종별 승률
    sec_perf = E[buy].groupby("sector")["f12m_t1_x"].agg(["count", "mean", "median", lambda s: (s > 0).mean()])
    sec_perf.columns = ["표본", "평균초과", "중앙초과", "승률"]
    sec_perf.to_csv(AN / "sector_performance.csv", encoding="utf-8-sig")

    # ---------- E. 매크로 ----------
    say("\n== E. 매크로 환경과 업종 매수 ==")
    mk = {}
    for t, nm in [("SPY", "spy"), ("CL=F", "oil"), ("DX-Y.NYB", "dxy"), ("^TNX", "us10y"), ("^VIX", "vix"), ("GC=F", "gold"), ("HG=F", "copper")]:
        d = load_px(t)
        if d is not None:
            mk[nm] = d["close"]
    periods = sorted(E["period"].unique())
    mrows = []
    for i in range(1, len(periods)):
        p, p0 = periods[i], periods[i - 1]
        r = {"period": p}
        for nm, s in mk.items():
            a, b = s[:p0], s[:p0 - pd.Timedelta(days=91)]
            if len(a) and len(b):
                r[f"{nm}_3m"] = (a.iloc[-1] - b.iloc[-1]) if nm in ("us10y", "vix") else a.iloc[-1] / b.iloc[-1] - 1
            if nm == "vix" and len(a):
                r["vix_lvl"] = a.iloc[-1]
        # 업종별 순매수(비중 변화, 주식 leg)
        g = trs[trs["period"] == p].groupby("sector")["dw"].sum()
        for s, v in g.items():
            r[f"dw_{s}"] = v
        tv0 = ps[ps["period"] == p0]["value"].sum()
        r["n_hold"] = (ps["period"] == p).sum()
        r["top10"] = ov.loc[p, "상위10비중"] if p in ov.index else np.nan
        mrows.append(r)
    M = pd.DataFrame(mrows).set_index("period")
    M.to_csv(AN / "macro_quarter.csv", encoding="utf-8-sig")
    pairs = [("dw_에너지", "oil_3m", "유가 3개월 변화"), ("dw_소재", "copper_3m", "구리 3개월 변화"),
             ("dw_소재", "dxy_3m", "달러 3개월 변화"), ("dw_금융", "us10y_3m", "10년 금리 3개월 변화(%p)"),
             ("dw_IT", "spy_3m", "S&P500 3개월 수익률"), ("dw_ETF", "vix_3m", "VIX 3개월 변화"),
             ("dw_헬스케어", "spy_3m", "S&P500 3개월 수익률"), ("n_hold", "vix_lvl", "VIX 수준")]
    mc = []
    for a, b, lb in pairs:
        if a in M and b in M:
            x = M[[a, b]].dropna()
            if len(x) > 10:
                rho, pv = stats.spearmanr(x[a], x[b])
                mc.append(dict(대상=a.replace("dw_", "") + " 순매수", 환경=lb, 순위상관=rho, p값=pv, 분기수=len(x)))
    mc = pd.DataFrame(mc)
    mc.to_csv(AN / "macro_corr.csv", index=False, encoding="utf-8-sig")
    for _, r in mc.iterrows():
        say(f"  {r['대상']} ~ 직전 {r['환경']}: 순위상관 {r['순위상관']:+.2f} (p={r['p값']:.2f}, {r['분기수']}분기)")

    # ---------- F. 조합 ----------
    say("\n== F. 특징 겹치기: S&P500 중 '그가 산 종목'일 확률 배율 ==")
    S2 = S.copy()
    S2["y"] = S2["druck_bought"].astype(float)
    base_rate = S2["y"].mean()
    conds = {
        "52주 고점 -10% 이내": S2["dist52h"] >= -0.10,
        "6개월 수익률 상위 30%": S2["pr_r6m"] > 70,
        "200일선 위": S2["vs200"] > 0,
        "매출 성장 상위 30%": S2["pr_rev_g_ttm"] > 70,
        "매출 가속(+)": S2["rev_accel"] > 0,
        "변동성 상위 30%": S2["pr_vol60"] > 70,
        "베타 > 1.2": S2["beta"] > 1.2,
        "시총 상위 30%": S2["pr_mcap"] > 70,
        "영업이익률 상위 30%": S2["pr_op_margin"] > 70,
        "PSR 상위 30%(비쌈)": S2["pr_ps"] > 70,
        "이익수익률 하위 30%(PER 높음)": S2["pr_ey"] <= 30,
        "부채/자본 하위 50%": S2["pr_debt_equity"] <= 50,
    }
    cr = []
    for k, m in conds.items():
        m = m.fillna(False)
        cr.append(dict(조건=k, 해당비율=m.mean(), 매수확률=S2.loc[m, "y"].mean(), 배율=S2.loc[m, "y"].mean() / base_rate,
                       포함매수비율=S2.loc[m, "y"].sum() / S2["y"].sum()))
    cr = pd.DataFrame(cr).sort_values("배율", ascending=False)
    cr.to_csv(AN / "single_condition_lift.csv", index=False, encoding="utf-8-sig")
    say(f"  기준: S&P500 종목이 그 분기에 그가 산 종목일 확률 {base_rate:.2%}")
    for _, r in cr.iterrows():
        say(f"  {r['조건']}: 배율 {r['배율']:.2f} (해당 {r['해당비율']:.0%}, 그가 산 S&P 종목의 {r['포함매수비율']:.0%} 포함)")
    # 상위 조건 2~3개 겹치기
    good = [k for k in cr["조건"] if cr.set_index("조건").loc[k, "배율"] > 1.2][:6]
    combo = []
    from itertools import combinations
    for n in (2, 3):
        for ks in combinations(good, n):
            m = np.logical_and.reduce([conds[k].fillna(False) for k in ks])
            if m.sum() < 100:
                continue
            combo.append(dict(조합=" + ".join(ks), 해당비율=m.mean(), 배율=S2.loc[m, "y"].mean() / base_rate,
                              포함매수비율=S2.loc[m, "y"].sum() / S2["y"].sum(),
                              이후12개월초과=S2.loc[m, "f12m_t1_x"].mean(),
                              나머지12개월초과=S2.loc[~m, "f12m_t1_x"].mean()))
    combo = pd.DataFrame(combo).sort_values("배율", ascending=False)
    combo.to_csv(AN / "combo_lift.csv", index=False, encoding="utf-8-sig")
    for _, r in combo.head(8).iterrows():
        say(f"  {r['조합']}: 배율 {r['배율']:.2f}, 포함 {r['포함매수비율']:.0%}, "
            f"S&P 안에서 이후 12개월 초과 {r['이후12개월초과']:+.1%} (나머지 {r['나머지12개월초과']:+.1%})")

    # 커버리지
    say("\n== 데이터 커버리지 ==")
    say(f"매수 이벤트(주식, ETF 제외) {buy.sum()}건: 주가 특징 {E.loc[buy, 'r6m'].notna().mean():.0%}, "
        f"재무 {E.loc[buy, 'rev_ttm'].notna().mean():.0%}, 시총 {E.loc[buy, 'mcap'].notna().mean():.0%}, "
        f"12개월 후 수익률 {E.loc[buy, 'f12m_t1'].notna().mean():.0%}")
    say(f"S&P500 비교군 {len(S)}행: 주가 {S['r6m'].notna().mean():.0%}, 재무 {S['rev_ttm'].notna().mean():.0%}")
    (OUT / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    P.to_pickle(OUT / "panel_ranked.pkl")


if __name__ == "__main__":
    main()
