# -*- coding: utf-8 -*-
"""5) 특징 패널: 드러켄밀러 매매(분기×종목) + 비교군(그 시점 S&P500 구성종목).

기준 시점
- t0 = 매수 분기의 '직전 분기말' (매수는 t0~t1 사이 어느 날. 그가 산 '직전 상태'를 본다)
- t1 = 매수 분기말(13F 기준일), tf = 13F 공시일(따라 사는 사람이 알 수 있는 날)
특징(전부 t0 이전 정보)
- 주가: 1·3·6·12개월 수익률, 12-1 모멘텀, 52주 고점 대비, 52주 저점 대비, 200일선 대비, 50/200일선,
        60일 변동성, 1년 베타, 잔차 변동성, 6개월 상대강도(SPY 대비), 1년 최대낙폭, 60일 거래대금
- 재무(SEC XBRL, t0 이전 공시분): 매출·영업이익·순이익 성장(TTM·분기), 매출 가속, 이익률, ROE, 부채, R&D, FCF
- 가치: 시가총액, PSR, 이익수익률, 장부가/시총, FCF수익률 (USD 재무만)
- 업종: SEC SIC → 11개 섹터(근사)
결과(사후, 특징 아님): 분기 중 수익률(t0→t1), t1·tf 이후 3·6·12개월 수익률과 SPY 초과
결과 파일: results/panel.pkl
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fundamentals import Company, snapshot, valuation  # noqa: E402

QUANT = Path(r"C:\Users\arlo0\OneDrive\바탕 화면\취업\quant")
C = QUANT / "cache" / "druck13f"
PX = C / "px"
SUBS = C / "subs"
OUT = HERE / "results"
H = {"User-Agent": "QuantResearch research-bot admin@example.com"}


# ---------------- 업종 ----------------
def sic_sector(sic):
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "미분류"
    if s in (1311, 1381, 1382, 1389, 1220, 1221, 1222) or 2900 <= s <= 2999 or s in (4922, 4923, 4924, 4925, 4610, 4612, 4613):
        return "에너지"
    if 1000 <= s <= 1499 or 2400 <= s <= 2699 or 2800 <= s <= 2829 or 2850 <= s <= 2899 or 3000 <= s <= 3299 or 3300 <= s <= 3399:
        return "소재"
    if 2830 <= s <= 2836 or 3840 <= s <= 3851 or 8000 <= s <= 8099 or s in (5122, 8731, 8734):
        return "헬스케어"
    if 2000 <= s <= 2199 or 2840 <= s <= 2844 or 5400 <= s <= 5499 or s in (5912, 5141, 5140):
        return "필수소비재"
    if 3570 <= s <= 3579 or 3661 <= s <= 3679 or 3670 <= s <= 3679 or 7370 <= s <= 7379 or 3820 <= s <= 3839 or s in (3600, 3612, 3620, 3621, 3690, 3695, 3577, 3578):
        return "IT"
    if 4800 <= s <= 4899 or 2700 <= s <= 2799 or 7810 <= s <= 7829 or s in (7812, 7819, 7822, 7841):
        return "커뮤니케이션"
    if 4900 <= s <= 4999:
        return "유틸리티"
    if 6000 <= s <= 6499 or 6700 <= s <= 6799 and s not in (6798, 6770):
        return "금융"
    if 6500 <= s <= 6599 or s == 6798:
        return "부동산"
    if s == 6770:
        return "SPAC"
    if 2200 <= s <= 2399 or 3630 <= s <= 3652 or 3710 <= s <= 3716 or 3900 <= s <= 3999 or 5200 <= s <= 5999 \
            or 7000 <= s <= 7299 or 7900 <= s <= 7999 or 8200 <= s <= 8299 or s in (3751, 3711, 3714, 3716, 2510, 2511, 2520, 2531):
        return "경기소비재"
    if 1500 <= s <= 1799 or 3400 <= s <= 3569 or 3580 <= s <= 3599 or 3700 <= s <= 3799 or 3800 <= s <= 3819 or 3860 <= s <= 3899 \
            or 4000 <= s <= 4799 or 5000 <= s <= 5199 or 7300 <= s <= 7369 or 7380 <= s <= 7399 or 8700 <= s <= 8799 or 3600 <= s <= 3699:
        return "산업재"
    return "기타"


def sub_info(cik):
    p = SUBS / f"{int(cik)}.json"
    if not p.exists():
        for k in range(3):
            r = requests.get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", headers=H, timeout=60)
            time.sleep(0.12)
            if r.status_code == 200:
                p.write_bytes(r.content)
                break
            if r.status_code == 404:
                return {}
            time.sleep(3)
    if not p.exists():
        return {}
    j = json.loads(p.read_bytes())
    return {"sic": j.get("sic"), "sic_desc": j.get("sicDescription")}


# ---------------- 주가 특징 ----------------
def load_px(t):
    p = PX / f"{t}.pkl"
    if not isinstance(t, str) or not p.exists():
        return None
    d = pd.read_pickle(p)
    if d.empty:
        return None
    d.index = pd.to_datetime(d.index).tz_localize(None)
    d = d[~d.index.duplicated()].sort_index()
    # 분할 전 실제 종가(raw): 야후 close × (그 날 이후 일어난 분할비의 곱)
    sp = C / "splits" / f"{t}.pkl"
    fac = pd.Series(1.0, index=d.index)
    if sp.exists():
        s = pd.read_pickle(sp)
        if len(s):
            s.index = pd.to_datetime(s.index).tz_localize(None)
            for dt, ratio in s.items():
                fac[fac.index < dt] *= float(ratio)
    d["raw"] = d["close"] * fac
    return d


def price_feats(d, spy):
    """일별 특징 표(일별 index). d: adj/close/vol."""
    a = d["adj"].astype(float)
    r = a.pct_change(fill_method=None)
    rs = spy.reindex(a.index).pct_change(fill_method=None)
    f = pd.DataFrame(index=a.index)
    for n, k in [(21, "r1m"), (63, "r3m"), (126, "r6m"), (252, "r12m")]:
        f[k] = a / a.shift(n) - 1
    f["r12_1"] = a.shift(21) / a.shift(252) - 1
    hi = a.rolling(252, min_periods=200).max()
    lo = a.rolling(252, min_periods=200).min()
    f["dist52h"] = a / hi - 1
    f["dist52l"] = a / lo - 1
    ma200 = a.rolling(200, min_periods=180).mean()
    ma50 = a.rolling(50, min_periods=45).mean()
    f["vs200"] = a / ma200 - 1
    f["ma50_200"] = ma50 / ma200 - 1
    f["vol60"] = r.rolling(60, min_periods=45).std() * np.sqrt(252)
    cov = r.rolling(252, min_periods=200).cov(rs)
    var = rs.rolling(252, min_periods=200).var()
    f["beta"] = cov / var
    resid = r - f["beta"] * rs
    f["ivol"] = resid.rolling(126, min_periods=100).std() * np.sqrt(252)
    spa = spy.reindex(a.index)
    f["rs6m"] = (a / a.shift(126)) / (spa / spa.shift(126)) - 1
    f["mdd1y"] = (a / a.rolling(252, min_periods=200).max() - 1).rolling(252, min_periods=200).min()
    f["dollar_vol"] = (d["close"] * d["vol"]).rolling(60, min_periods=40).mean()
    f["close"] = d["raw"]
    f["adj"] = a
    return f


def at(f, t):
    s = f[:t]
    if len(s) == 0 or (pd.Timestamp(t) - s.index[-1]).days > 7:
        return None
    return s.iloc[-1]


def fwd(a, t, n):
    s = a[t:]
    if len(s) == 0 or (s.index[0] - pd.Timestamp(t)).days > 7:
        return np.nan  # 기준일 근처 시세 없음(상장 전 등)
    i0 = 1 if s.index[0] == pd.Timestamp(t) else 0
    s = s.iloc[i0:]
    if len(s) <= n:
        return np.nan
    return s.iloc[n] / s.iloc[0] - 1


# ---------------- 재무 (병렬) ----------------
def fs_job(args):
    cik, dates = args
    co = Company(cik)
    out = {}
    if not co.ok:
        return cik, out
    for t in dates:
        try:
            out[t] = snapshot(co, t)
        except Exception as e:  # noqa
            out[t] = {"fs_err": str(e)[:80]}
    return cik, out


def main():
    sec = pd.read_csv(OUT / "securities.csv", dtype={"cusip": str})
    tr = pd.read_csv(OUT / "trades.csv", dtype={"cusip": str}, parse_dates=["period", "filed"])
    periods = sorted(tr["period"].unique())
    prevp = {periods[i]: periods[i - 1] for i in range(1, len(periods))}
    prevp[periods[0]] = periods[0] - pd.offsets.QuarterEnd(1)

    spy = load_px("SPY")["adj"]
    spy.index = pd.to_datetime(spy.index)

    # 드러켄밀러 이벤트
    ev = tr.merge(sec[["cusip", "yf", "cik", "kind", "sic", "sic_desc"]], on="cusip", how="left")
    ev["t1"] = ev["period"]
    ev["t0"] = ev["period"].map(prevp)
    ev["src"] = "드러켄밀러"

    # 비교군: t0 시점 S&P500
    sp = pd.read_csv(C / "sp500_ticker_start_end.csv")
    sp["start_date"] = pd.to_datetime(sp["start_date"])
    sp["end_date"] = pd.to_datetime(sp["end_date"]).fillna(pd.Timestamp("2100-01-01"))
    sp["yf"] = sp["ticker"].str.replace(".", "-", regex=False)
    ct = json.loads((C / "company_tickers.json").read_text())
    t2c = {v["ticker"].upper(): int(v["cik_str"]) for v in ct.values()}
    base = []
    for p in periods:
        t0 = prevp[p]
        m = sp[(sp["start_date"] <= t0) & (sp["end_date"] > t0)]
        for t in m["yf"].unique():
            base.append(dict(period=p, t0=t0, t1=p, yf=t, cik=t2c.get(t), src="S&P500"))
    base = pd.DataFrame(base)
    base["filed"] = base["period"].map(tr.groupby("period")["filed"].first())
    print("비교군 행", len(base), "분기당 평균", round(len(base) / len(periods)))

    # 업종: SIC
    all_cik = pd.concat([ev["cik"], base["cik"]]).dropna().astype(int).unique()
    sic = {}
    for c in all_cik:
        sic[c] = sub_info(c)
    for df in (ev, base):
        df["sic"] = [sic.get(int(c), {}).get("sic") if pd.notna(c) else None for c in df["cik"]]
        df["sector"] = [sic_sector(s) for s in df["sic"]]
    ev.loc[ev["kind"] == "ETF", "sector"] = "ETF"

    # 주가 특징
    feats = {}
    for t in pd.concat([ev["yf"], base["yf"]]).dropna().unique():
        d = load_px(t)
        if d is not None and len(d) > 30:
            feats[t] = price_feats(d, spy)
    print("시세 있는 티커", len(feats))
    pcols = ["r1m", "r3m", "r6m", "r12m", "r12_1", "dist52h", "dist52l", "vs200", "ma50_200", "vol60",
             "beta", "ivol", "rs6m", "mdd1y", "dollar_vol", "close"]

    def add_price(df):
        out = []
        for t, t0, t1, tf in zip(df["yf"], df["t0"], df["t1"], df["filed"]):
            row = {}
            f = feats.get(t)
            if f is not None:
                x = at(f, t0)
                if x is not None:
                    row.update({k: x[k] for k in pcols})
                    row["px_t0"] = x["close"]
                a = f["adj"]
                x1 = at(f, t1)
                if x is not None and x1 is not None:
                    row["r_inq"] = x1["adj"] / x["adj"] - 1
                    row["r_inq_x"] = row["r_inq"] - (at_s(spy, t1) / at_s(spy, t0) - 1)
                for n, k in [(63, "3m"), (126, "6m"), (252, "12m")]:
                    for base_t, tag in [(t1, "t1"), (tf, "tf")]:
                        v = fwd(a, base_t, n)
                        row[f"f{k}_{tag}"] = v
                        row[f"f{k}_{tag}_x"] = v - fwd(spy, base_t, n)
            out.append(row)
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(out)], axis=1)

    def at_s(s, t):
        x = s[:t]
        return x.iloc[-1] if len(x) else np.nan

    ev = add_price(ev)
    base = add_price(base)

    # 재무(병렬)
    need = {}
    for df in (ev, base):
        for c, t0 in zip(df["cik"], df["t0"]):
            if pd.notna(c):
                need.setdefault(int(c), set()).add(pd.Timestamp(t0))
    jobs = [(c, sorted(v)) for c, v in need.items()]
    print("재무 계산 회사", len(jobs))
    res = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for k, (c, out) in enumerate(ex.map(fs_job, jobs, chunksize=4), 1):
            res[c] = out
            if k % 200 == 0:
                print("  재무", k, "/", len(jobs), flush=True)

    def add_fs(df):
        rows = []
        for c, t0, px in zip(df["cik"], df["t0"], df.get("px_t0", pd.Series([np.nan] * len(df)))):
            r = {}
            if pd.notna(c):
                r = dict(res.get(int(c), {}).get(pd.Timestamp(t0), {}))
                r.update(valuation(r, px if px is not None else np.nan))
            rows.append(r)
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    ev = add_fs(ev)
    base = add_fs(base)
    # 같은 분기에 드러켄밀러가 산 종목은 비교군에서 표시
    bought = set(zip(ev.loc[ev["action"].isin(["신규", "확대"]), "period"], ev.loc[ev["action"].isin(["신규", "확대"]), "yf"]))
    base["druck_bought"] = [(p, t) in bought for p, t in zip(base["period"], base["yf"])]
    held = set(zip(ev.loc[ev["shares"] > 0, "period"], ev.loc[ev["shares"] > 0, "yf"]))
    base["druck_held"] = [(p, t) in held for p, t in zip(base["period"], base["yf"])]
    sp_set = set(zip(base["period"], base["yf"]))
    ev["in_sp500"] = [(p, t) in sp_set for p, t in zip(ev["period"], ev["yf"])]
    panel = pd.concat([ev, base], ignore_index=True)
    panel.to_pickle(OUT / "panel.pkl")
    print("패널", panel.shape)
    e = panel[panel["src"] == "드러켄밀러"]
    print("드러켄밀러 행", len(e), "주가 있음", e["r6m"].notna().sum(), "재무 있음", e["rev_ttm"].notna().sum())
    b = panel[panel["src"] == "S&P500"]
    print("비교군 행", len(b), "주가 있음", b["r6m"].notna().sum(), "재무 있음", b["rev_ttm"].notna().sum())


if __name__ == "__main__":
    main()
