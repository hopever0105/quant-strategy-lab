# -*- coding: utf-8 -*-
"""21) 다른 유명 펀드들의 13F → '여러 펀드가 같은 분기에 새로 산 종목' 신호. 실제 매매 없음. KRX 요청 없음.

펀드(집중 투자형 주식 펀드, 2012~ 13F 있음): Appaloosa, Tiger Global, Coatue, Viking, Lone Pine, Third Point,
  Pershing Square, Baupost, Greenlight, Maverick, Soros, Berkshire, Glenview, Akre, Whale Rock, ValueAct  (드러켄밀러 제외)
1단계 fetch: SEC에서 13F 받기 (cache/druck13f/raw_others/, 이어받기)  — python others13f.py fetch
2단계 build: 펀드·분기별 보유 → 신규 매수(직전 분기에 없던 CUSIP) → 분기별 '새로 산 펀드 수'
            2곳 이상 겹친 CUSIP만 OpenFIGI로 티커 → results/others/overlap.csv  — python others13f.py build
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from lxml import etree

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sec13f import parse_xml_table, parse_txt_table, strip_ns  # noqa: E402
from map_ids import figi_map  # noqa: E402

QUANT = Path(r"C:\Users\arlo0\OneDrive\바탕 화면\취업\quant")
RAW = QUANT / "cache" / "druck13f" / "raw_others"
LOG = QUANT / "logs" / "druck13f.log"
OUT = HERE / "results" / "others"
OUT.mkdir(parents=True, exist_ok=True)
H = {"User-Agent": "QuantResearch research-bot admin@example.com"}
FUNDS = {"Appaloosa": [1656456, 1006438], "Tiger Global": [1167483], "Coatue": [1135730], "Viking": [1103804],
         "Lone Pine": [1061165], "Third Point": [1040273], "Pershing Square": [1336528], "Baupost": [1061768],
         "Greenlight": [1079114], "Maverick": [934639], "Soros": [1029160], "Berkshire": [1067983],
         "Glenview": [1138995], "Akre": [1112520], "Whale Rock": [1387322], "ValueAct": [1351069]}


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + "[others] " + m + "\n")


def get(url, path=None):
    if path is not None and path.exists():
        return path.read_bytes()
    for k in range(4):
        try:
            r = requests.get(url, headers=H, timeout=60)
        except Exception as e:  # noqa
            log(f"오류 {url} {e}")
            time.sleep(5)
            continue
        time.sleep(0.15)
        if r.status_code == 200:
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(r.content)
            return r.content
        if r.status_code == 404:
            return None
        time.sleep(3 * (k + 1))
    return None


def filings(cik):
    j = json.loads(get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", RAW / str(cik) / "submissions.json"))
    blocks = [j["filings"]["recent"]]
    for f in j["filings"].get("files", []):
        b = get(f"https://data.sec.gov/submissions/{f['name']}", RAW / str(cik) / f["name"])
        if b:
            blocks.append(json.loads(b))
    rows = []
    for rec in blocks:
        for f, d, a, rd in zip(rec["form"], rec["filingDate"], rec["accessionNumber"], rec["reportDate"]):
            if f in ("13F-HR", "13F-HR/A") and rd and rd >= "2011-09-30":
                rows.append(dict(form=f, filed=d, acc=a, period=rd))
    return pd.DataFrame(rows).sort_values(["period", "filed"]) if rows else pd.DataFrame()


def fetch_one(cik, acc):
    accn = acc.replace("-", "")
    d = RAW / str(cik) / acc
    b = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/index.json", d / "index.json")
    if not b:
        return [], ""
    items = [x["name"] for x in json.loads(b)["directory"]["item"]]
    at = ""
    if "primary_doc.xml" in [x.lower() for x in items]:
        nm = [x for x in items if x.lower() == "primary_doc.xml"][0]
        pb = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/{nm}", d / nm)
        if pb:
            try:
                at = (strip_ns(etree.fromstring(pb)).findtext(".//amendmentType") or "").strip().upper()
            except Exception:
                pass
    xmls = [x for x in items if x.lower().endswith(".xml") and x.lower() != "primary_doc.xml"]
    recs = []
    if xmls:
        for nm in xmls:
            xb = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/{nm}", d / nm)
            if xb and b"infoTable" in xb:
                recs += parse_xml_table(xb)
    else:
        for nm in [x for x in items if x.lower().endswith(".txt")]:
            tb = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/{nm}", d / nm)
            if tb:
                recs += parse_txt_table(tb.decode("latin-1"))
                break
    return recs, at


def fetch():
    for fund, ciks in FUNDS.items():
        out = RAW / f"{fund}.csv"
        if out.exists():
            continue
        allr = []
        for cik in ciks:
            fl = filings(cik)
            per = {}
            for _, r in fl.iterrows():
                recs, at = fetch_one(cik, r["acc"])
                df = pd.DataFrame(recs)
                if len(df) == 0:
                    continue
                p = r["period"]
                if r["form"] == "13F-HR":
                    per[p] = (df, r["filed"])
                elif at.startswith("NEW HOLDINGS") and p in per:
                    per[p] = (pd.concat([per[p][0], df]), per[p][1])
                elif p in per:
                    per[p] = (df, per[p][1])
            for p, (df, f) in per.items():
                df = df.copy()
                df["period"], df["filed"], df["cik"] = p, f, cik
                allr.append(df)
        if allr:
            h = pd.concat(allr, ignore_index=True)
            h["fund"] = fund
            h.to_csv(out, index=False, encoding="utf-8-sig")
            log(f"{fund} {h['period'].nunique()}분기 {len(h)}행")
        else:
            pd.DataFrame().to_csv(out, index=False)
            log(f"{fund} 없음")
    log("others fetch 끝")


def build():
    hs = []
    for fund in FUNDS:
        p = RAW / f"{fund}.csv"
        if p.exists() and p.stat().st_size > 10:
            h = pd.read_csv(p, dtype={"cusip": str})
            hs.append(h)
    H_ = pd.concat(hs, ignore_index=True)
    H_ = H_[(H_["putcall"].fillna("") == "") & (H_["shares"] > 0)]
    H_["cusip"] = H_["cusip"].str.upper().str.zfill(9)
    H_["period"] = pd.to_datetime(H_["period"])
    # 같은 펀드에 CIK 둘(Appaloosa)이면 합침
    pos = H_.groupby(["fund", "period", "cusip"])["shares"].sum().reset_index()
    rows = []
    for fund, g in pos.groupby("fund"):
        ps = sorted(g["period"].unique())
        prev = set()
        for i, p in enumerate(ps):
            cur = set(g.loc[g["period"] == p, "cusip"])
            if i > 0 and (p - ps[i - 1]).days < 100:  # 직전 분기 자료가 있어야 '신규' 판단
                for c in cur - prev:
                    rows.append(dict(fund=fund, period=p, cusip=c))
            prev = cur
    N = pd.DataFrame(rows)
    cnt = N.groupby(["period", "cusip"])["fund"].agg(["nunique", lambda x: ", ".join(sorted(x))]).reset_index()
    cnt.columns = ["period", "cusip", "n_funds", "funds"]
    cov = pos.groupby("period")["fund"].nunique()
    print("분기별 자료 있는 펀드 수:", cov.describe().round(1).to_dict())
    ov = cnt[cnt["n_funds"] >= 2].copy()
    print(f"2곳 이상 같은 분기 신규: {len(ov)}건, CUSIP {ov['cusip'].nunique()}개 → FIGI 매핑")
    fm = figi_map(sorted(ov["cusip"].unique()))
    ov["ticker"] = [(fm.get(c) or {}).get("ticker") for c in ov["cusip"]]
    ov["sectype"] = [(fm.get(c) or {}).get("securityType") for c in ov["cusip"]]
    ov.to_csv(OUT / "overlap.csv", index=False, encoding="utf-8-sig")
    N.to_csv(OUT / "new_buys.csv", index=False, encoding="utf-8-sig")
    print(ov["n_funds"].value_counts().sort_index().to_dict(), "티커 매핑률", round(ov["ticker"].notna().mean(), 2))


def test():
    import numpy as np
    from common_q import load, base_pick, qsim, table, COST
    ov = pd.read_csv(OUT / "overlap.csv", dtype={"cusip": str}, parse_dates=["period"])
    ov = ov[ov["ticker"].notna()]
    ov["code"] = ov["ticker"].str.replace("/", "-", regex=False)
    P, R, M = load("us")
    # period p의 13F는 p+45일까지 공시 → 다음 분기말 t0 = p + 1분기에 사용
    ov["t0"] = (ov["period"].dt.to_period("Q") + 1).dt.end_time.dt.normalize()
    k = ov.groupby(["t0", "code"])["n_funds"].max().reset_index()
    P = P.merge(k, on=["t0", "code"], how="left")
    P["n_funds"] = P["n_funds"].fillna(0)
    d = P.dropna(subset=["ret"])
    for th in [2, 3]:
        sp = d.groupby(["t0", d["n_funds"] >= th])["ret"].mean().unstack().dropna()
        x = sp[True] - sp[False]
        print(f"{th}곳+ 같이 새로 산 S&P500 종목 − 나머지 다음 분기 {x.mean():+.2%} t {x.mean() / x.std() * np.sqrt(len(x)):+.2f} "
              f"({len(x)}분기, 분기당 평균 {d[d['n_funds'] >= th].groupby('t0').size().mean():.1f}종목)")
    by = {q: g for q, g in P.groupby("t0")}
    pk = {"O1 2곳+ 새로 삼": {q: g.loc[g["n_funds"] >= 2, "code"].tolist() for q, g in by.items()},
          "O2 3곳+ 새로 삼": {q: g.loc[g["n_funds"] >= 3, "code"].tolist() for q, g in by.items()},
          "O3 기본 규칙 ∩ 1곳+": {q: [c for c in base_pick(g) if c in set(g.loc[g["n_funds"] >= 1, "code"])] for q, g in by.items()},
          "기본 규칙": {q: base_pick(g) for q, g in by.items()}}
    rows = []
    for nm, p in pk.items():
        r, n = qsim(p, 2, R, M, COST["us"])
        rows.append((f"{nm} (평균 {n.mean():.1f}종목, 빈 분기 {(n == 0).mean():.0%})", "us", r, M, False))
    rows.append(("동일가중 시장", "us", d.groupby("t0")["ret"].mean(), M, False))
    rows.append(("시장지수", "us", M, M, False))
    T = table(rows)
    T.to_csv(OUT / "results.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False))
    # 드러켄밀러 확신 매수와 겹침: 다른 펀드가 먼저(직전 분기)/같은 분기/다음 분기에 새로 샀나
    Pn = pd.read_pickle(HERE / "results" / "panel_ranked.pkl")
    E = Pn[(Pn["src"] == "드러켄밀러") & (Pn["leg"] == "주식") & Pn["action"].isin(["신규", "확대"]) & (Pn["weight"] >= 0.03)].copy()
    E["period"] = pd.to_datetime(E["period"])
    N = pd.read_csv(OUT / "new_buys.csv", dtype={"cusip": str}, parse_dates=["period"])
    nb = N.groupby(["period", "cusip"])["fund"].nunique()
    out = {}
    for lag, nm in [(-1, "직전 분기"), (0, "같은 분기"), (1, "다음 분기")]:
        hits = []
        for _, e in E.iterrows():
            p = (e["period"].to_period("Q") + lag).end_time.normalize()
            hits.append(nb.get((p, str(e["cusip"]).zfill(9)), 0))
        out[nm] = np.mean(np.array(hits) >= 1)
    print("확신 매수 종목을 다른 펀드(16곳 중 1곳+)가 새로 산 비율:", {k: f"{v:.0%}" for k, v in out.items()})


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    {"fetch": fetch, "build": build, "test": test}[sys.argv[1]]()
