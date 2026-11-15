# -*- coding: utf-8 -*-
"""2) CUSIP → 티커(OpenFIGI) · CIK(SEC) · SIC 업종 · 증권 종류(보통주/ADR/ETF).

- OpenFIGI: 키 없이 1요청 10건, 분당 25요청. 결과 캐시 cache/druck13f/figi.json (이어받기).
- 상장폐지 등으로 FIGI가 못 찾으면 SEC 회사명 목록(cik-lookup-data.txt)과 이름 일치로 CIK.
- CIK마다 SEC submissions JSON → SIC 코드, 티커, 거래소 (cache/druck13f/subs/).
결과: results/securities.csv
"""
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
QUANT = Path(r"C:\Users\arlo0\OneDrive\바탕 화면\취업\quant")
C = QUANT / "cache" / "druck13f"
SUBS = C / "subs"
SUBS.mkdir(parents=True, exist_ok=True)
OUT = HERE / "results"
H = {"User-Agent": "QuantResearch research-bot admin@example.com"}

STOP = {"INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "PLC", "LLC", "LP", "L P",
        "HOLDINGS", "HLDGS", "HLDG", "GROUP", "GRP", "THE", "NEW", "DEL", "SA", "NV", "AG", "SE", "ADR",
        "SPONSORED", "SPON", "CL", "CLASS", "A", "B", "COM", "ORD", "SHS", "INTL", "INTERNATIONAL"}


def norm(s):
    s = str(s).upper().replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    toks = [t for t in s.split() if t not in STOP]
    return " ".join(toks)


def figi_map(cusips):
    path = C / "figi.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [c for c in cusips if c not in cache]
    for i in range(0, len(todo), 10):
        batch = todo[i:i + 10]
        for k in range(5):
            r = requests.post("https://api.openfigi.com/v3/mapping",
                              json=[{"idType": "ID_CUSIP", "idValue": c} for c in batch], timeout=60)
            if r.status_code == 200:
                break
            time.sleep(15 * (k + 1))
        for c, x in zip(batch, r.json()):
            d = [y for y in x.get("data", []) if y.get("exchCode") == "US"] or x.get("data", [])
            cache[c] = d[0] if d else None
        path.write_text(json.dumps(cache))
        time.sleep(2.6)
    return cache


def sec_get(url, path):
    if path.exists():
        return path.read_bytes()
    for k in range(4):
        r = requests.get(url, headers=H, timeout=60)
        time.sleep(0.15)
        if r.status_code == 200:
            path.write_bytes(r.content)
            return r.content
        if r.status_code == 404:
            return None
        time.sleep(3 * (k + 1))
    return None


# 이름 일치가 틀린 것(옛 법인·자회사로 잡힌 것) → 직접 지정한 CIK. None = 확인 못 함(재무 제외)
CIK_FIX = {
    "Celgene Corp": 816284, "Chubb Limited": 896159, "Linde Plc": 1707925, "Broadcom Ltd": 1649338,
    "Monsanto Co": 1110783, "Eaton Corp Plc": 1551182, "Herbalife Ltd.": 1180262, "Transocean Ltd": 1451505,
    "Ingersoll-Rand Plc": 1466258, "NABORS INDUSTRIES LTD": 1163739, "Ubs Group Ag": 1610520,
    "Noble Corp": None, "Flex Ltd": 866374, "Elan Corp": None, "Liberty Media Corp Del": 1560385,
    "Limited Brands, Inc.": 701985, "Kraft Foods Inc.": 1103982, "Alcoa Inc.": 4281, "Nordstrom Inc": 72333,
    "Fisker Inc": 1720990, "New Fortress Energy Inc": 1749723, "Jacobs Engr Group Inc.": 52988,
    "Confluent Inc": 1699838, "Rice Energy Inc": 1584952, "WhiteWave Foods Co": None,
    "Standard Pacific Corp.": None, "Tronox Limited": 1530804, "Intelsat S A": 1525773,
    "Time Warner Inc": 1105705, "Carnival Corp": 815097, "Baker Hughes Inc.": 808362,
}
# FIGI·이름으로 못 찾은 것 → 현재 티커(있으면 SEC 목록으로 CIK)
TICKER_FIX = {
    "881624209": "TEVA", "G7997R103": "STX", "G3643J108": "FLUT", "N62509109": "NAMS", "N53745100": "LYB",
    "G25508105": "CRH", "228227104": "CCI", "78388J106": "SBAC", "16117M305": "CHTR", "V7780T103": "RCL",
    "13645T100": "CP", "22663K107": "CRNX", "N7902X106": "ST", "647581107": "EDU", "M3760D101": "ESLT",
    "450047204": "IRS",
}


def main():
    h = pd.read_csv(OUT / "holdings_raw.csv", dtype={"cusip": str})
    sec = (h.sort_values("period").groupby("cusip")
           .agg(name=("name", "last"), cls=("cls", "last"), first=("period", "min"), last=("period", "max"),
                n_q=("period", "nunique"), max_val=("value_usd", "max")).reset_index())
    print("고유 CUSIP", len(sec))
    fm = figi_map(sec["cusip"].tolist())
    sec["ticker"] = [(fm.get(c) or {}).get("ticker") for c in sec["cusip"]]
    sec["figi_type"] = [(fm.get(c) or {}).get("securityType") for c in sec["cusip"]]
    sec["figi_name"] = [(fm.get(c) or {}).get("name") for c in sec["cusip"]]
    print("FIGI 매핑", sec["ticker"].notna().sum())

    # SEC 현재 티커 → CIK
    ct = json.loads(sec_get("https://www.sec.gov/files/company_tickers.json", C / "company_tickers.json"))
    t2c = {v["ticker"].upper(): int(v["cik_str"]) for v in ct.values()}
    # 전체 회사명 → CIK
    lk = sec_get("https://www.sec.gov/Archives/edgar/cik-lookup-data.txt", C / "cik-lookup-data.txt")
    n2c = {}
    for line in lk.decode("latin-1").splitlines():
        m = re.match(r"(.*):(\d{10}):$", line)
        if m:
            k = norm(m.group(1))
            if k and k not in n2c:
                n2c[k] = int(m.group(2))
    for c, t in TICKER_FIX.items():
        sec.loc[sec["cusip"] == c, "ticker"] = t
    ciks, how = [], []
    for _, r in sec.iterrows():
        cik, w = None, ""
        if r["name"] in CIK_FIX:
            ciks.append(CIK_FIX[r["name"]])
            how.append("fix")
            continue
        t = r["ticker"]
        if isinstance(t, str):
            tt = t.replace("/", "-").upper()
            if tt in t2c:
                cik, w = t2c[tt], "ticker"
        if cik is None:
            for nm in [r["name"], r["figi_name"]]:
                if isinstance(nm, str) and norm(nm) in n2c:
                    cik, w = n2c[norm(nm)], "name"
                    break
        ciks.append(cik)
        how.append(w)
    sec["cik"] = ciks
    sec["cik_how"] = how
    # submissions → SIC, 티커, 거래소, 회사 종류
    info = []
    for cik in sec["cik"]:
        if pd.isna(cik):
            info.append({})
            continue
        b = sec_get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", SUBS / f"{int(cik)}.json")
        if not b:
            info.append({})
            continue
        j = json.loads(b)
        info.append(dict(sic=j.get("sic"), sic_desc=j.get("sicDescription"), sec_name=j.get("name"),
                         sec_tickers=",".join(j.get("tickers") or []), exch=",".join(j.get("exchanges") or []),
                         entity_type=j.get("entityType"), country=(j.get("addresses", {}).get("business", {}) or {}).get("stateOrCountry")))
    sec = pd.concat([sec, pd.DataFrame(info)], axis=1)
    nm = sec["name"].str.upper().fillna("") + " " + sec["cls"].astype(str).str.upper()
    etf = nm.str.contains(r"\bETF\b|SPDR|ISHARES|SELECT SECTOR|TRUST UNIT|POWERSHARES|VANGUARD|INVESCO QQQ|PROSHARES|VANECK|GLOBAL X|DIREXION|INDEX FD|MARKET VECTORS")
    etf = etf | sec["figi_type"].isin(["ETP", "Closed-End Fund", "Open-End Fund", "Unit"])
    adr = nm.str.contains(r"\bADR\b|\bADS\b|SPONSORED|SPON ") | sec["figi_type"].eq("ADR")
    sec["kind"] = "보통주"
    sec.loc[adr, "kind"] = "ADR"
    sec.loc[etf, "kind"] = "ETF"
    # 야후 시세용 티커: 미국 거래소 FIGI 티커 > SEC 현재 티커 첫 번째
    fx = [(fm.get(c) or {}).get("exchCode") for c in sec["cusip"]]
    yt = []
    for t, e, st, c in zip(sec["ticker"], fx, sec["sec_tickers"], sec["cusip"]):
        if c in TICKER_FIX:
            yt.append(TICKER_FIX[c])
        elif isinstance(t, str) and e == "US":
            yt.append(t.replace("/", "-"))
        elif isinstance(st, str) and st:
            yt.append(st.split(",")[0])
        else:
            yt.append(None)
    sec["yf"] = yt
    sec.to_csv(OUT / "securities.csv", index=False, encoding="utf-8-sig")
    print(sec["kind"].value_counts().to_string())
    print("CIK 찾음", sec["cik"].notna().sum(), "/", len(sec), sec["cik_how"].value_counts().to_dict())
    miss = sec[sec["cik"].isna() & (sec["kind"] != "ETF")]
    print("CIK 못 찾은 주식", len(miss))
    print(miss[["cusip", "name", "ticker", "first", "max_val"]].sort_values("max_val", ascending=False).head(40).to_string())


if __name__ == "__main__":
    main()
