# -*- coding: utf-8 -*-
"""전체 시장 지표 검증 (급등 종목에서 보인 특징이 코스피·코스닥 전체에서도 다음 1년 수익과 관련 있는지)

대상: 매년 4월 마지막 거래일(2000~2025) 상장 종목 중 비금융·보통주, 상장 1년 이상, 시가총액 확인 가능. 상장폐지 포함.
수익: 그날 종가 → 다음 해 4월 마지막 거래일 종가(상장폐지되면 마지막 가격에서 멈춤).
지표(그날까지 알 수 있는 값만)
  회전율: 20일 평균 거래대금 ÷ 시가총액     베타: 최근 252일 코스피 대비
  MAX: 최근 21일 중 최대 하루 상승률        ROE: KRX EPS ÷ BPS (그날 기준 최근 사업연도)
  200일선 대비: 종가 ÷ 200일 평균 − 1       변동성60일: 최근 60일 일간 수익률 표준편차 연율화
  %R(10): Williams %R
통제: 시가총액(로그), PBR(KRX), 12-1개월 모멘텀
방법
  A) 5분위: 지표 순서대로 5개 구간 → 구간별 동일가중 수익. 시총 5분위 안에서 다시 5분위(시총 통제)도.
  B) Fama-MacBeth: 매년 다음 1년 수익을 지표 순위(0~1)들에 회귀 → 연도별 계수 평균·t. 단변량(시총 통제)·다변량.
  C) Fama-French 2×3: 시총 중앙값 × 지표 30/70% 경계(Fama-French 1993 관례) → (소형 상위+대형 상위)/2 − (소형 하위+대형 하위)/2
  D) 조합: 지표마다 '급등 종목 쪽'(상위 20%, ROE는 10% 이상)과 반대쪽(하위 20%, ROE 10% 미만)을 1·2·3개 겹침.
     시총 통제 수익 = 종목 수익 − 같은 시총 5분위 평균. 126개 비교 → Benjamini-Hochberg로 다중검정 보정.
     무작위 대조: 해마다 같은 수의 종목을 무작위로 1,000번 뽑아 같은 계산 → 실제가 무작위보다 나은 비율.
  E) 기간 분할 2000~2014 / 2015~2025, F) 코스피 상승 해 / 하락 해, G) 플라시보(1년 전 지표),
  H) 바이오·제약 제외, I) 거래대금 하위 20% 제외(매매 가능성)
실행: python factors.py → results/factors_*.csv
"""
import io
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D          # noqa: E402

OUT = HERE / "results"
KRXF = D.CACHE / "krx_fund"
IND = ["회전율", "베타", "MAX", "ROE", "200일선대비", "변동성60일", "WR10"]
CTRL = ["log시총", "PBR", "모멘텀12_1"]
RNG = np.random.default_rng(0)


def build_panel():
    fn = D.CACHE / ("factor_panel_v2.pkl" if D.USE_DELIST else "factor_panel.pkl")
    if fn.exists():
        return pd.read_pickle(fn)
    W = D.panel_wide()
    C, R = W["c"], W["ret"].astype("float64")
    info = D.listing_info()
    pan = D.panel_long()[["d", "code", "amt"]] if D.USE_DELIST else pd.read_pickle(D.PANEL)[["d", "code", "amt"]]
    AMT = pan.pivot(index="d", columns="code", values="amt").reindex(C.index)
    del pan
    kos, _ = D.kospi()
    kr = kos["c"].pct_change().reindex(C.index).fillna(0.0)
    cal = C.index
    dates = [cal[(cal.year == y) & (cal.month == 4)][-1] for y in range(2000, 2027)]
    mc = D.market_caps(dates, C)
    rows = []
    for i, d in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        pos = cal.get_loc(d)
        c = C.iloc[: pos + 1]
        p0 = c.iloc[-1]
        alive = p0.notna() & c.iloc[-5:].notna().any() & (c.iloc[-252:].notna().sum() >= 240)
        codes = alive.index[alive]
        codes = codes[[x.endswith("0") for x in codes]]                      # 보통주(우선주 코드 끝자리 5·7·9 등 제외)
        fin = info["fin"].reindex(codes).fillna(False).astype(bool)
        codes = codes[~fin.values]
        f = pd.DataFrame(index=codes)
        f["mcap"] = mc.loc[d].reindex(codes)
        f = f[f.mcap > 0]
        codes = f.index
        cc = c[codes]
        rr = R.iloc[pos - 251: pos + 1][codes]
        amt = AMT.iloc[pos - 19: pos + 1][codes].mean() / 1e8
        f["회전율"] = amt / f.mcap
        k = kr.iloc[pos - 251: pos + 1].values
        X = rr.fillna(0.0).values
        f["베타"] = ((X - X.mean(0)) * (k - k.mean())[:, None]).sum(0) / ((k - k.mean()) ** 2).sum()
        f["MAX"] = rr.iloc[-21:].max()
        f["200일선대비"] = p0[codes] / cc.iloc[-200:].mean() - 1
        f["변동성60일"] = rr.iloc[-60:].std() * np.sqrt(252)
        hh, ll = cc.iloc[-10:].max(), cc.iloc[-10:].min()
        f["WR10"] = ((hh - p0[codes]) / (hh - ll) * -100).where(hh > ll)
        f["모멘텀12_1"] = cc.iloc[-22] / cc.iloc[-253] - 1 if len(cc) > 253 else np.nan
        kf = pd.read_csv(KRXF / f"{d:%Y%m%d}.csv", dtype={"code": str}).set_index("code")
        kf = kf.reindex(codes)
        f["ROE"] = (kf.EPS / kf.BPS).where(kf.BPS > 0)
        f["PBR"] = kf.PBR.where(kf.PBR > 0)
        f["log시총"] = np.log(f.mcap)
        f["거래대금억"] = amt
        days = R.index[(R.index > d) & (R.index <= nxt)]
        f["fwd"] = (1 + R.loc[days, codes].fillna(0.0)).prod() - 1
        f["bio"] = info["sector"].reindex(codes).isin(["건강관리", "제약"]).values
        f["date"] = d
        f["year"] = d.year
        rows.append(f)
        print(d.date(), len(f), flush=True)
    P = pd.concat(rows)
    P.index.name = "code"
    P = P.reset_index()
    # 플라시보: 1년 전 지표
    prev = P.set_index(["code", "year"])[IND]
    prev.index = pd.MultiIndex.from_arrays([prev.index.get_level_values(0), prev.index.get_level_values(1) + 1])
    lag = prev.reindex(pd.MultiIndex.from_arrays([P.code, P.year]))
    for col in IND:
        P[f"lag_{col}"] = lag[col].values
    # 시총 5분위·시총 통제 수익
    P["size_q"] = P.groupby("year").log시총.transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
    P["fwd_w"] = P.groupby("year").fwd.transform(lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))
    P["fwd_adj"] = P.fwd - P.groupby(["year", "size_q"]).fwd.transform("mean")
    P.to_pickle(fn)
    return P


def tstat(x):
    x = pd.Series(x).dropna()
    return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan


def quintiles(P, col, by_size=False, ret="fwd"):
    """반환: 연도×구간 평균 수익 표"""
    def q(x):
        return pd.qcut(x.rank(method="first"), 5, labels=False) if x.notna().sum() >= 25 else pd.Series(np.nan, index=x.index)
    df = P[["year", "size_q", col, ret]].dropna(subset=[col])
    if by_size:
        df = df.assign(q=df.groupby(["year", "size_q"])[col].transform(q))
        t = df.groupby(["year", "size_q", "q"])[ret].mean().groupby(["year", "q"]).mean().unstack()
    else:
        df = df.assign(q=df.groupby("year")[col].transform(q))
        t = df.groupby(["year", "q"])[ret].mean().unstack()
    t.columns = [f"Q{int(c) + 1}" for c in t.columns]
    return t


def q_summary(t, label, col, extra=None):
    sp = t["Q5"] - t["Q1"]
    m = t.mean()
    row = dict(지표=col, 방식=label, **{k: m[k] for k in ["Q1", "Q2", "Q3", "Q4", "Q5"]}, Q5_Q1=sp.mean(), t=tstat(sp),
               Q5가이긴해비율=(sp > 0).mean(), 단조성=stats.spearmanr(range(5), m.values)[0], 연수=len(sp))
    if extra:
        row.update(extra)
    return row


def fama_macbeth(P, cols, ret="fwd_w"):
    coefs = []
    for y, g in P.groupby("year"):
        g = g[cols + [ret]].dropna()
        if len(g) < 100:
            continue
        X = g[cols].rank(pct=True).values
        X = np.column_stack([np.ones(len(X)), X])
        b, *_ = np.linalg.lstsq(X, g[ret].values, rcond=None)
        coefs.append(pd.Series(b[1:], index=cols, name=y))
    Cf = pd.DataFrame(coefs)
    return Cf


def ff_factor(P, col):
    rows = {}
    for y, g in P.groupby("year"):
        g = g[["log시총", col, "fwd"]].dropna()
        if len(g) < 100:
            continue
        big = g.log시총 > g.log시총.median()
        lo, hi = g[col].quantile(0.3), g[col].quantile(0.7)
        def m(sz, sel):
            s = g[sz & sel].fwd
            return s.mean() if len(s) else np.nan
        H, L = g[col] >= hi, g[col] <= lo
        rows[y] = 0.5 * (m(~big, H) + m(big, H)) - 0.5 * (m(~big, L) + m(big, L))
    return pd.Series(rows)


def flags(P):
    """급등 쪽(상위 20%, ROE ≥ 10%)·반대쪽(하위 20%, ROE < 10%) 표시"""
    F = pd.DataFrame(index=P.index)
    for col in IND:
        if col == "ROE":
            F[f"{col}+"] = P.ROE >= 0.10
            F[f"{col}-"] = P.ROE < 0.10
        else:
            pr = P.groupby("year")[col].rank(pct=True)
            F[f"{col}+"] = pr > 0.8
            F[f"{col}-"] = pr <= 0.2
    return F


def bh(p):
    p = np.asarray(p, float)
    n = np.sum(~np.isnan(p))
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    q = np.full(len(p), np.nan)
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        if np.isnan(p[i]):
            continue
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def combos(P, F, sign, sub=None, n_rand=1000):
    names = [f"{c}{sign}" for c in IND]
    years = sorted(P.year.unique())
    adj_by_year = {y: P.loc[P.year == y, "fwd_adj"].values for y in years}
    rows = []
    for k in (1, 2, 3):
        for cmb in combinations(names, k):
            m = F[list(cmb)].all(axis=1)
            if sub is not None:
                m &= sub
            g = P[m]
            ex_raw = g.groupby("year").fwd.mean() - P.groupby("year").fwd.mean().reindex(g.year.unique())
            ex_adj = g.groupby("year").fwd_adj.mean()
            cnt = g.groupby("year").size()
            ok = cnt[cnt >= 3].index
            ex_adj, ex_raw, cnt = ex_adj.reindex(ok), ex_raw.reindex(ok), cnt.reindex(ok)
            if len(ok) < 8:
                continue
            # 무작위 대조: 같은 해 같은 수 무작위 → 연도 평균의 평균
            sims = np.zeros(n_rand)
            for y, n in cnt.items():
                pool = adj_by_year[y]
                idx = RNG.integers(0, len(pool), size=(n_rand, int(n)))
                sims += pool[idx].mean(axis=1)
            sims /= len(cnt)
            t = tstat(ex_adj)
            rows.append(dict(조합=" & ".join(cmb), 지표수=k, 해수=len(ok), 연평균종목수=cnt.mean(),
                             초과수익_단순=ex_raw.mean(), 초과수익_시총통제=ex_adj.mean(), t=t,
                             p=2 * stats.t.sf(abs(t), len(ok) - 1) if t == t else np.nan,
                             이긴해비율=(ex_adj > 0).mean(), 무작위보다나은비율=(ex_adj.mean() > sims).mean(),
                             전기_2000_2014=ex_adj[ex_adj.index <= 2014].mean(), 후기_2015_2025=ex_adj[ex_adj.index >= 2015].mean(),
                             t전기=tstat(ex_adj[ex_adj.index <= 2014]), t후기=tstat(ex_adj[ex_adj.index >= 2015])))
    return pd.DataFrame(rows)


def main():
    P = build_panel()
    P = P[P.year <= 2025]
    kos, _ = D.kospi()
    kc = kos["c"]
    up = {}
    for y in sorted(P.year.unique()):
        a = kc[(kc.index.year == y) & (kc.index.month == 4)].iloc[-1]
        b = kc[(kc.index.year == y + 1) & (kc.index.month == 4)]
        b = b.iloc[-1] if len(b) else kc.iloc[-1]
        up[y] = b / a - 1
    P["kospi_up"] = P.year.map(lambda y: up[y] > 0)
    info = dict(연수=P.year.nunique(), 종목연=len(P), 연평균종목=len(P) / P.year.nunique(),
                ROE있는비율=P.ROE.notna().mean(), ROE0비율=(P.ROE == 0).mean())
    print(info, flush=True)

    # A) 5분위
    QA = []
    for col in IND:
        QA.append(q_summary(quintiles(P, col), "5분위", col))
        QA.append(q_summary(quintiles(P, col, by_size=True), "5분위(시총 통제)", col))
        QA.append(q_summary(quintiles(P, col, by_size=True)[lambda t: t.index <= 2014], "시총통제·2000~2014", col))
        QA.append(q_summary(quintiles(P, col, by_size=True)[lambda t: t.index >= 2015], "시총통제·2015~2025", col))
        qs = quintiles(P, col, by_size=True)
        QA.append(q_summary(qs[[up[y] > 0 for y in qs.index]], "시총통제·코스피 상승 해", col))
        QA.append(q_summary(qs[[up[y] <= 0 for y in qs.index]], "시총통제·코스피 하락 해", col))
        QA.append(q_summary(quintiles(P.assign(**{col: P[f"lag_{col}"]}), col, by_size=True), "플라시보(1년 전 지표)·시총통제", col))
        QA.append(q_summary(quintiles(P[~P.bio], col, by_size=True), "바이오·제약 제외·시총통제", col))
        liq = P.groupby("year").거래대금억.rank(pct=True) > 0.2
        QA.append(q_summary(quintiles(P[liq], col, by_size=True), "거래대금 하위20% 제외·시총통제", col))
    # ROE 10% 기준 (사용자 지정)
    r10 = P.dropna(subset=["ROE"]).groupby(["year"]).apply(lambda g: g[g.ROE >= 0.1].fwd_adj.mean() - g[g.ROE < 0.1].fwd_adj.mean())
    QA.append(dict(지표="ROE", 방식="ROE≥10% − ROE<10% (시총 통제)", Q5_Q1=r10.mean(), t=tstat(r10), Q5가이긴해비율=(r10 > 0).mean(), 연수=len(r10)))
    QA = pd.DataFrame(QA)
    QA.to_csv(OUT / "factors_quintiles.csv", index=False, encoding="utf-8-sig")

    # B) Fama-MacBeth
    FM = []
    for col in IND:
        cf = fama_macbeth(P, [col, "log시총"])
        FM.append(dict(지표=col, 모형="단변량+시총", 계수=cf[col].mean(), t=tstat(cf[col]), 양수해비율=(cf[col] > 0).mean(),
                       전기=cf[col][cf.index <= 2014].mean(), t전기=tstat(cf[col][cf.index <= 2014]),
                       후기=cf[col][cf.index >= 2015].mean(), t후기=tstat(cf[col][cf.index >= 2015])))
    cf = fama_macbeth(P, IND + CTRL)
    for col in IND + CTRL:
        FM.append(dict(지표=col, 모형="다변량(7개+시총·PBR·모멘텀)", 계수=cf[col].mean(), t=tstat(cf[col]), 양수해비율=(cf[col] > 0).mean(),
                       전기=cf[col][cf.index <= 2014].mean(), t전기=tstat(cf[col][cf.index <= 2014]),
                       후기=cf[col][cf.index >= 2015].mean(), t후기=tstat(cf[col][cf.index >= 2015])))
    cfb = fama_macbeth(P[~P.bio], IND + CTRL)
    for col in IND:
        FM.append(dict(지표=col, 모형="다변량·바이오 제외", 계수=cfb[col].mean(), t=tstat(cfb[col]), 양수해비율=(cfb[col] > 0).mean()))
    FM = pd.DataFrame(FM)
    FM.to_csv(OUT / "factors_fama_macbeth.csv", index=False, encoding="utf-8-sig")

    # C) Fama-French 2×3
    FF = []
    for col in IND:
        s = ff_factor(P, col)
        FF.append(dict(지표=col, 연평균=s.mean(), t=tstat(s), 양수해비율=(s > 0).mean(), 전기=s[s.index <= 2014].mean(),
                       t전기=tstat(s[s.index <= 2014]), 후기=s[s.index >= 2015].mean(), t후기=tstat(s[s.index >= 2015])))
    FF = pd.DataFrame(FF)
    FF.to_csv(OUT / "factors_ff2x3.csv", index=False, encoding="utf-8-sig")

    # E2) 구간 단위 교차검증(3년 블록, 검증 블록 앞뒤 1년은 학습에서 뺌)
    #  학습 블록들로 효과 방향(+/−)을 정하고, 검증 블록에서 같은 방향이 나오는지 본다. 블록마다 반복.
    years = sorted(P.year.unique())
    blocks = [years[i:i + 3] for i in range(0, len(years), 3)]
    CV = []
    series = {}
    for col in IND:
        series[(col, "5분위 시총통제 Q5−Q1")] = (lambda t: t["Q5"] - t["Q1"])(quintiles(P, col, by_size=True))
        series[(col, "Fama-MacBeth 다변량 계수")] = cf[col]
        series[(col, "Fama-French 2×3")] = ff_factor(P, col)
    for (col, how), s in series.items():
        hits, tests = [], []
        for b in blocks:
            purge = set(b) | {b[0] - 1, b[-1] + 1}
            tr = s[~s.index.isin(purge)]
            te = s[s.index.isin(b)]
            if len(tr) < 5 or not len(te):
                continue
            sign = np.sign(tr.mean())
            hits.append(np.sign(te.mean()) == sign)
            tests.append(te.mean() * sign)
        CV.append(dict(지표=col, 방식=how, 블록수=len(hits), 방향일치블록=int(np.sum(hits)), 방향일치비율=np.mean(hits),
                       검증블록_평균효과=np.mean(tests), 블록별=", ".join(f"{b[0]}~{b[-1]}:{s[s.index.isin(b)].mean():+.3f}" for b in blocks)))
    CV = pd.DataFrame(CV)
    CV.to_csv(OUT / "factors_cv.csv", index=False, encoding="utf-8-sig")

    # D) 조합 + 다중검정 + 무작위
    F = flags(P)
    CB = pd.concat([combos(P, F, "+").assign(방향="급등 쪽"), combos(P, F, "-").assign(방향="반대쪽")], ignore_index=True)
    CB["q_BH"] = bh(CB.p.values)
    CB = CB.sort_values("t", ascending=False)
    CB.to_csv(OUT / "factors_combos.csv", index=False, encoding="utf-8-sig")
    CBb = combos(P, F, "+", sub=~P.bio).assign(방향="급등 쪽·바이오 제외")
    CBb["q_BH"] = bh(CBb.p.values)
    CBb.sort_values("t", ascending=False).to_csv(OUT / "factors_combos_nobio.csv", index=False, encoding="utf-8-sig")
    pd.Series(info).to_csv(OUT / "factors_info.csv", encoding="utf-8-sig")

    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30); pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(QA.to_string(index=False))
    print(FM.to_string(index=False))
    print(FF.to_string(index=False))
    print(CV.to_string(index=False))
    print(CB.head(20).to_string(index=False))
    print(CB.tail(10).to_string(index=False))
    print("BH q<0.05:", int((CB.q_BH < 0.05).sum()), "/", len(CB), " q<0.10:", int((CB.q_BH < 0.10).sum()))
    print(CBb.sort_values("t", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
