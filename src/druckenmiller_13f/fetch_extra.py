# -*- coding: utf-8 -*-
"""18) 추가 데이터 받기 (백그라운드, 끊겨도 이어받기). KRX 요청 없음. 진행은 quant/logs/druck13f.log 에만.

us   : SEC XBRL companyfacts에서 '앞으로 들어올 매출' 계정만 → cache/druck13f/fs_rpo/<cik>.csv.gz
       수주잔고 RevenueRemainingPerformanceObligation, 계약부채(선수금) ContractWithCustomerLiability(Current),
       이연수익 DeferredRevenue(Current)
kr   : OpenDART 전체 재무제표(fnlttSinglAcntAll, 사업보고서)에서 계약부채·선수금·선수수익 계정
       → cache/druck13f/kr_contract/<code>_<fy>.json (코스피 상위 200 대용 패널에 나온 376종목, 2016~2025)
fred : 유동성 점수용 FRED 시리즈 → cache/druck13f/fred/<id>.csv
실행: python fetch_extra.py us | kr | fred
"""
import gzip
import io
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
QUANT = Path(r"C:\Users\arlo0\OneDrive\바탕 화면\취업\quant")
JOB = QUANT.parent
C = QUANT / "cache" / "druck13f"
LOG = QUANT / "logs" / "druck13f.log"
H = {"User-Agent": "QuantResearch research-bot admin@example.com"}

RPO = {"us-gaap": ["RevenueRemainingPerformanceObligation", "ContractWithCustomerLiability",
                   "ContractWithCustomerLiabilityCurrent", "DeferredRevenue", "DeferredRevenueCurrent",
                   "ContractWithCustomerLiabilityNoncurrent", "DeferredRevenueNoncurrent"]}
FRED = ["WALCL", "DGS2", "DGS10", "BAA10Y", "DTWEXBGS", "DEXKOUS", "IR3TIB01KRM156N", "IRLTLT01KRM156N",
        "T10Y2Y", "M2SL"]


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + "[extra] " + m + "\n")


def us():
    out = C / "fs_rpo"
    out.mkdir(exist_ok=True)
    ciks = [int(p.name.split(".")[0]) for p in (C / "fs").glob("*.csv.gz")]
    todo = [c for c in ciks if not (out / f"{c}.csv.gz").exists()]
    log(f"us 수주잔고 대상 {len(ciks)}, 남은 것 {len(todo)}")
    for n, cik in enumerate(todo, 1):
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
        j = None
        for k in range(4):
            try:
                r = requests.get(url, headers=H, timeout=90)
            except Exception as e:  # noqa
                log(f"us 오류 {cik} {e}")
                time.sleep(5)
                continue
            time.sleep(0.15)
            if r.status_code == 200:
                j = r.json()
                break
            if r.status_code == 404:
                break
            time.sleep(5 * (k + 1))
        rows = []
        if j:
            for tax, names in RPO.items():
                for nm in names:
                    node = j.get("facts", {}).get(tax, {}).get(nm)
                    if not node:
                        continue
                    for unit, arr in node.get("units", {}).items():
                        for f in arr:
                            rows.append((nm, unit, f.get("start"), f.get("end"), f.get("val"), f.get("filed"), f.get("form")))
        df = pd.DataFrame(rows, columns=["concept", "unit", "start", "end", "val", "filed", "form"])
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        (out / f"{cik}.csv.gz").write_bytes(gzip.compress(buf.getvalue()))
        if n % 100 == 0:
            log(f"us {n}/{len(todo)}")
    log("us 끝")


def kr():
    VAL = JOB / "valuation"
    key = next(l.split("=", 1)[1].strip() for l in (VAL / ".env").read_text(encoding="utf-8").splitlines()
               if l.startswith("DART_API_KEY="))
    root = ET.parse(VAL / "tools" / "corpCode.xml").getroot()
    cm = {(e.findtext("stock_code") or "").strip(): e.findtext("corp_code") for e in root if (e.findtext("stock_code") or "").strip()}
    out = C / "kr_contract"
    out.mkdir(exist_ok=True)
    codes = sorted(pd.read_pickle(HERE / "results" / "combo" / "qpanel_kr.pkl")["code"].unique())
    jobs = [(c, y) for c in codes for y in range(2016, 2026)]
    log(f"kr 계약부채 대상 {len(jobs)}")
    n = 0
    for c, y in jobs:
        fn = out / f"{c}_{y}.json"
        if fn.exists():
            continue
        if c not in cm:
            fn.write_text("[]", encoding="utf-8")
            continue
        rows = []
        for fd in ("CFS", "OFS"):
            j = {}
            for k in range(3):
                try:
                    j = requests.get("https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json",
                                     params=dict(crtfc_key=key, corp_code=cm[c], bsns_year=y, reprt_code="11011", fs_div=fd),
                                     timeout=60).json()
                    break
                except Exception as e:  # noqa
                    log(f"kr 오류 {c} {y} {e}")
                    time.sleep(5)
            time.sleep(0.3)
            if j.get("status") == "020":
                log("DART 요청 한도 초과(020) - 중지")
                sys.exit(2)
            if j.get("list"):
                for x in j["list"]:
                    nm = (x.get("account_nm") or "").replace(" ", "")
                    aid = x.get("account_id") or ""
                    if x.get("sj_div") == "BS" and ("계약부채" in nm or "선수금" in nm or "선수수익" in nm or "ContractLiab" in aid):
                        rows.append(dict(acc=nm, aid=aid, cur=x.get("thstrm_amount"), prev=x.get("frmtrm_amount"), fs=fd))
                    if x.get("sj_div") in ("IS", "CIS") and (aid == "ifrs-full_Revenue" or nm in ("매출액", "수익(매출액)", "영업수익", "매출")):
                        rows.append(dict(acc="__sales__" + nm, aid=aid, cur=x.get("thstrm_amount"), prev=x.get("frmtrm_amount"), fs=fd))
                break
        fn.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        n += 1
        if n % 200 == 0:
            log(f"kr {n}")
    log("kr 끝")


def fred():
    out = C / "fred"
    out.mkdir(exist_ok=True)
    for s in FRED:
        r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}", timeout=60)
        (out / f"{s}.csv").write_bytes(r.content)
        log(f"fred {s} {len(r.content)}")
        time.sleep(1)


if __name__ == "__main__":
    {"us": us, "kr": kr, "fred": fred}[sys.argv[1]]()
