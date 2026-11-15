# -*- coding: utf-8 -*-
"""9) '확신 매수 프로필' 규칙 백테스트 (미국, S&P500 그 시점 구성종목). 실제 매매 없음.

규칙(분기말 t0에 아는 정보만):
  시총 S&P500 상위 30% (pr_mcap > 70) · 베타 ≥ B · 12개월 수익률 > 0 · 52주 고점 대비 ≥ H
  · 매출 성장(최근 4분기) ≥ G · 매출총이익률 ≥ M
  → 분기말 종가에 동일가중 매수, 다음 분기말까지 보유(분기 중엔 가격 따라 비중 변동).
  해당 종목 없으면 SPY.
변형:
  V1 1분기 보유 / V2 '오른 종목 더 사기': 직전 분기 SPY보다 오른 보유 종목은 조건(시총·베타)만 맞으면 남기고 비중 1.5배
대조:
  C1 대형 고베타만(시총 상위 30% + 베타 ≥ B), C2 S&P500 동일가중(시세 있는 종목), SPY
시간 분할: 학습 2012~2018(격자 선택) / 검증 2019~2021 / 테스트 2022~2026Q2(한 번만)
비용: 편도 0.1% × 매매 금액
결과: results/rule/*.csv, charts/rule_*.png
"""
import io
import itertools
import sys
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from features import load_px  # noqa: E402

QUANT = Path(r"C:\Users\arlo0\OneDrive\바탕 화면\취업\quant")
C = QUANT / "cache" / "druck13f"
OUT = HERE / "results" / "rule"
OUT.mkdir(parents=True, exist_ok=True)
CH = HERE / "charts"
COST = 0.001
SEG = {"학습": ("2012-01-01", "2018-12-31"), "검증": ("2019-01-01", "2021-12-31"), "테스트": ("2022-01-01", "2026-12-31"),
       "전체": ("2012-01-01", "2026-12-31")}

_px = {}


def adj(t):
    if t not in _px:
        d = load_px(t)
        _px[t] = None if d is None else d["adj"].astype(float)
    return _px[t]


def load_panel():
    P = pd.read_pickle(HERE / "results" / "panel_ranked.pkl")
    S = P[P["src"] == "S&P500"].copy()
    S["t0"] = pd.to_datetime(S["t0"])
    S["t1"] = pd.to_datetime(S["t1"])
    bad = (S["mcap"] > 5e12) | (S["mcap"] < 1e8)
    S.loc[bad, ["mcap", "pr_mcap"]] = np.nan
    return S[S["r12m"].notna()]


def select(g, p, keep=None):
    m = (g["pr_mcap"] > 70) & (g["beta"] >= p["B"])
    if p.get("ctrl"):
        return g.loc[m, "yf"].tolist()
    m &= (g["r12m"] > 0) & (g["dist52h"] >= p["H"]) & (g["rev_g_ttm"] >= p["G"]) & (g["gross_margin"] >= p["M"])
    return g.loc[m.fillna(False), "yf"].tolist()


def run(S, p, spy):
    """분기별 보유 → 일별 수익률. 반환: 일별 수익률 Series, 분기 기록 DataFrame."""
    quarters = sorted(S["t0"].unique())
    daily = []
    log = []
    prev_w = pd.Series(dtype=float)  # 직전 분기말 (가격 변동 반영된) 비중
    prev_ret = {}
    for q0 in quarters:
        g = S[S["t0"] == q0]
        q1 = g["t1"].iloc[0]
        names = select(g, p)
        w = pd.Series(1.0, index=names)
        if p.get("pyramid") and len(prev_ret):
            gi = g.set_index("yf")
            spy_r = prev_ret.get("__SPY__", 0)
            for t, r in prev_ret.items():
                if t == "__SPY__" or t not in gi.index:
                    continue
                ok = (gi.at[t, "pr_mcap"] > 70) and (gi.at[t, "beta"] >= p["B"] - 0.1)
                if r > spy_r and ok:
                    w[t] = 1.5
        if len(w) == 0:
            w = pd.Series({"SPY": 1.0})
        w = w / w.sum()
        # 일별 경로
        idx = spy[(spy.index > q0) & (spy.index <= q1)].index
        if len(idx) == 0:
            continue
        val = pd.DataFrame(index=idx)
        for t in w.index:
            s = adj(t)
            if s is None:
                s = adj("SPY")
            a = s[:q1]
            base = a[:q0]
            if len(base) == 0 or (q0 - base.index[-1]).days > 7:
                a = adj("SPY")[:q1]
                base = a[:q0]
            path = a[(a.index > q0)].reindex(idx).ffill().fillna(base.iloc[-1]) / base.iloc[-1]
            val[t] = path * w[t]
        v = val.sum(axis=1)
        r = v.pct_change()
        r.iloc[0] = v.iloc[0] - 1
        # 비용: 직전 비중(가격 변동 후) → 새 비중 매매 금액
        allk = w.index.union(prev_w.index)
        traded = (w.reindex(allk, fill_value=0) - prev_w.reindex(allk, fill_value=0)).abs().sum()
        r.iloc[0] = r.iloc[0] - traded * COST * p.get("cost_mult", 1)
        daily.append(r)
        end_val = val.iloc[-1]
        prev_w = end_val / end_val.sum()
        qret = (val.iloc[-1] / w).to_dict()
        prev_ret = {t: qret[t] - 1 for t in w.index}
        prev_ret["__SPY__"] = spy[:q1].iloc[-1] / spy[:q0].iloc[-1] - 1
        log.append(dict(t0=q0, t1=q1, n=len(w) if "SPY" not in w.index or len(w) > 1 else 0,
                        turnover=traded / 2, qret=v.iloc[-1] - 1, names=",".join(w.index[:40])))
    return pd.concat(daily).sort_index(), pd.DataFrame(log)


def metrics(r, rf, seg=None):
    if seg:
        a, b = SEG[seg]
        r = r[a:b]
    if len(r) < 20:
        return {}
    rfx = rf.reindex(r.index).fillna(0)
    yrs = len(r) / 252
    cum = (1 + r).prod()
    cagr = cum ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    sh = (r - rfx).mean() / r.std() * np.sqrt(252)
    nav = (1 + r).cumprod()
    mdd = (nav / nav.cummax() - 1).min()
    return dict(CAGR=cagr, 변동성=vol, 샤프=sh, MDD=mdd)


def ff_factors():
    p = C / "ff6_daily.pkl"
    if p.exists():
        return pd.read_pickle(p)
    H = {"User-Agent": "Mozilla/5.0"}
    out = []
    for u in ["https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
              "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"]:
        z = zipfile.ZipFile(io.BytesIO(requests.get(u, headers=H, timeout=60).content))
        txt = z.read(z.namelist()[0]).decode("latin-1").splitlines()
        rows = [l.split(",") for l in txt if l[:8].strip().isdigit() and len(l.split(",")) > 1]
        hdr_i = [i for i, l in enumerate(txt) if l.startswith(",")][0]
        cols = [c.strip() for c in txt[hdr_i].split(",")[1:]]
        d = pd.DataFrame([r[1:] for r in rows], index=pd.to_datetime([r[0] for r in rows]), columns=cols).astype(float) / 100
        out.append(d)
    f = out[0].join(out[1], how="inner")
    f.columns = [c.replace("Mom", "MOM").strip() for c in f.columns]
    f.to_pickle(p)
    return f


def alpha(r, ff, seg="전체"):
    a, b = SEG[seg]
    d = pd.concat([r.rename("r"), ff], axis=1, join="inner")[a:b].dropna()
    if len(d) < 60:
        return {}
    y = d["r"] - d["RF"]
    out = {}
    for nm, cols in [("CAPM", ["Mkt-RF"]), ("FF6", ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"])]:
        X = sm.add_constant(d[cols])
        m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
        out[f"{nm}_알파(연)"] = m.params["const"] * 252
        out[f"{nm}_t"] = m.tvalues["const"]
        if nm == "FF6":
            for c in cols:
                out[f"β_{c}"] = m.params[c]
    return out


def summarize(name, r, log, rf, ff):
    rows = []
    for seg in SEG:
        m = metrics(r, rf, seg)
        if not m:
            continue
        m.update(alpha(r, ff, seg))
        a, b = SEG[seg]
        lg = log[(log["t0"] >= pd.Timestamp(a) - pd.Timedelta(days=95)) & (log["t0"] <= b)] if log is not None else None
        if lg is not None and len(lg):
            m["평균종목수"] = lg["n"].mean()
            m["분기회전율"] = lg["turnover"].mean()
            m["SPY대체분기"] = (lg["n"] == 0).sum()
        rows.append(dict(전략=name, 구간=seg, **m))
    return rows


def main():
    S = load_panel()
    spy = adj("SPY")
    irx = load_px("^IRX")["close"] / 100 / 252
    rf = irx.reindex(spy.index).ffill().fillna(0)
    ff = ff_factors()
    spy_r = spy.pct_change().dropna()
    spy_r = spy_r["2011-09-30":]

    # ---- 1. 격자(학습 구간에서만 선택) ----
    grid = []
    for B, H, G, M in itertools.product([1.0, 1.1, 1.2], [-0.10, -0.15, -0.20], [0.05, 0.10, 0.15], [0.40, 0.50]):
        p = dict(B=B, H=H, G=G, M=M)
        r, lg = run(S, p, spy)
        for seg in ["학습", "검증", "테스트", "전체"]:
            m = metrics(r, rf, seg)
            a, b = SEG[seg]
            lgs = lg[(lg["t0"] >= pd.Timestamp(a) - pd.Timedelta(days=95)) & (lg["t0"] <= b)]
            grid.append(dict(B=B, H=H, G=G, M=M, 구간=seg, **m, 평균종목수=lgs["n"].mean(), SPY대체분기=(lgs["n"] == 0).sum()))
        print(f"격자 B={B} H={H} G={G} M={M} 학습샤프 {metrics(r, rf, '학습')['샤프']:.2f}", flush=True)
    G_ = pd.DataFrame(grid)
    G_.to_csv(OUT / "grid.csv", index=False, encoding="utf-8-sig")
    tr = G_[(G_["구간"] == "학습") & (G_["평균종목수"] >= 5)].sort_values("샤프", ascending=False)
    best = tr.iloc[0]
    P0 = dict(B=best["B"], H=best["H"], G=best["G"], M=best["M"])
    # 학습 상위 격자의 중앙값 설정(과적합 점검용)
    print("학습 선택:", P0)

    # ---- 2. 본 전략·변형·대조 ----
    res = []
    curves = {}
    logs = {}
    for name, p in [("V1 확신 프로필(1분기)", dict(P0)), ("V2 확신 프로필 + 오른 종목 더 사기", dict(P0, pyramid=True)),
                    ("C1 대형 고베타만(대조)", dict(P0, ctrl=True)), ("V1 비용 2배", dict(P0, cost_mult=2))]:
        r, lg = run(S, p, spy)
        curves[name] = r
        logs[name] = lg
        res += summarize(name, r, lg, rf, ff)
    # S&P500 동일가중(시세 있는 종목 전부)
    ew = []
    for q0, g in S.groupby("t0"):
        q1 = g["t1"].iloc[0]
        idx = spy[(spy.index > q0) & (spy.index <= q1)].index
        vals = []
        for t in g["yf"]:
            s = adj(t)
            if s is None or len(s[:q0]) == 0 or (q0 - s[:q0].index[-1]).days > 7:
                continue
            b0 = s[:q0].iloc[-1]
            vals.append(s[(s.index > q0) & (s.index <= q1)].reindex(idx).ffill().fillna(b0) / b0)
        v = pd.concat(vals, axis=1).mean(axis=1)
        rr = v.pct_change()
        rr.iloc[0] = v.iloc[0] - 1
        ew.append(rr)
    ew = pd.concat(ew).sort_index()
    curves["C2 S&P500 동일가중"] = ew
    res += summarize("C2 S&P500 동일가중", ew, None, rf, ff)
    curves["SPY"] = spy_r
    res += summarize("SPY", spy_r, None, rf, ff)
    R = pd.DataFrame(res)
    R.to_csv(OUT / "results.csv", index=False, encoding="utf-8-sig")
    for k, lg in logs.items():
        lg.to_csv(OUT / f"log_{k.split()[0]}.csv", index=False, encoding="utf-8-sig")

    # 연도별
    yr = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in curves.items()})
    yr.to_csv(OUT / "yearly.csv", encoding="utf-8-sig")

    # 파라미터 이웃: 선택값 주변 격자의 검증·테스트 성과
    nb = G_[(G_["B"].sub(P0["B"]).abs() <= 0.1) & (G_["H"].sub(P0["H"]).abs() <= 0.05)
            & (G_["G"].sub(P0["G"]).abs() <= 0.05)]
    nb.to_csv(OUT / "neighbors.csv", index=False, encoding="utf-8-sig")

    # 차트
    fig, ax = plt.subplots(figsize=(11, 5))
    for k in ["V1 확신 프로필(1분기)", "V2 확신 프로필 + 오른 종목 더 사기", "C1 대형 고베타만(대조)", "C2 S&P500 동일가중", "SPY"]:
        v = curves[k]["2012-01-01":]
        ax.plot((1 + v).cumprod(), label=k, lw=1.4 if k.startswith("V") else 1.0)
    for seg in ["검증", "테스트"]:
        ax.axvline(pd.Timestamp(SEG[seg][0]), color="gray", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("확신 매수 프로필 규칙 누적 수익 (로그, 점선: 검증·테스트 시작)")
    plt.tight_layout()
    plt.savefig(CH / "rule_equity.png", dpi=150)
    plt.close()

    pd.set_option("display.width", 250)
    show = ["전략", "구간", "CAGR", "변동성", "샤프", "MDD", "CAPM_알파(연)", "CAPM_t", "FF6_알파(연)", "FF6_t", "평균종목수", "분기회전율"]
    print(R[show].round(3).to_string(index=False))
    print("\n연도별\n", yr.round(3).to_string())
    print("\n이웃 격자(검증·테스트 샤프)")
    print(nb[nb["구간"].isin(["학습", "검증", "테스트"])].pivot_table(index=["B", "H", "G", "M"], columns="구간", values="샤프").round(2).to_string())


if __name__ == "__main__":
    main()
