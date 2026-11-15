# -*- coding: utf-8 -*-
"""3) 시세(야후)·재무(SEC XBRL companyfacts) 수집. 백그라운드용, 이어받기 가능. KRX 요청 없음.

대상
- 드러켄밀러 보유 종목(securities.csv의 yf·cik)
- 비교군: S&P500 역대 구성종목(fja05680/sp500 ticker_start_end, 편입·편출일 포함)
- 시장 변수: SPY QQQ IWM ^VIX ^TNX ^IRX DX-Y.NYB CL=F GC=F HG=F, 섹터 ETF 11개
저장
- cache/druck13f/px/<티커>.pkl  (일별 수정종가 adj, 종가 close, 거래량 vol). 실패는 빈 표로 기록 → 다시 안 받음
- cache/druck13f/fs/<cik>.csv.gz (필요 계정만 추린 XBRL 사실: concept, start, end, val, filed, form, fp)
진행 로그: quant/logs/druck13f.log
"""
import gzip
import io
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
QUANT = Path(r"C:\Users\arlo0\OneDrive\바탕 화면\취업\quant")
C = QUANT / "cache" / "druck13f"
PX = C / "px"
FS = C / "fs"
PX.mkdir(parents=True, exist_ok=True)
FS.mkdir(parents=True, exist_ok=True)
LOG = QUANT / "logs" / "druck13f.log"
H = {"User-Agent": "QuantResearch research-bot admin@example.com"}
START = "2009-01-01"

MARKET = ["SPY", "QQQ", "IWM", "^VIX", "^TNX", "^IRX", "DX-Y.NYB", "CL=F", "GC=F", "HG=F",
          "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]

CONCEPTS = {
    "us-gaap": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet",
                "GrossProfit", "OperatingIncomeLoss", "NetIncomeLoss", "StockholdersEquity",
                "Assets", "Liabilities", "LongTermDebt", "LongTermDebtNoncurrent", "DebtCurrent",
                "CashAndCashEquivalentsAtCarryingValue", "NetCashProvidedByUsedInOperatingActivities",
                "PaymentsToAcquirePropertyPlantAndEquipment", "EarningsPerShareDiluted",
                "WeightedAverageNumberOfDilutedSharesOutstanding", "ResearchAndDevelopmentExpense",
                "CommonStockSharesOutstanding"],
    "dei": ["EntityCommonStockSharesOutstanding"],
    "ifrs-full": ["Revenue", "GrossProfit", "ProfitLossFromOperatingActivities", "ProfitLoss", "Equity",
                  "Assets", "Liabilities", "CashFlowsFromUsedInOperatingActivities"],
}


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + "[collect] " + msg + "\n")


def sp500_list():
    p = C / "sp500_ticker_start_end.csv"
    if not p.exists():
        u = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
        p.write_bytes(requests.get(u, timeout=60).content)
    return pd.read_csv(p)


def get_prices(tickers):
    todo = [t for t in dict.fromkeys(tickers) if t and not (PX / f"{t}.pkl").exists()]
    log(f"시세 대상 {len(set(tickers))}, 남은 것 {len(todo)}")
    for i in range(0, len(todo), 40):
        b = todo[i:i + 40]
        try:
            d = yf.download(b, start=START, auto_adjust=False, actions=False, group_by="ticker",
                            threads=True, progress=False)
        except Exception as e:  # noqa
            log(f"yf 실패 {b[:3]}.. {e}")
            time.sleep(30)
            continue
        for t in b:
            try:
                x = d[t] if len(b) > 1 else d
                df = pd.DataFrame({"adj": x["Adj Close"], "close": x["Close"], "vol": x["Volume"]}).dropna(subset=["close"])
            except Exception:
                df = pd.DataFrame(columns=["adj", "close", "vol"])
            df.to_pickle(PX / f"{t}.pkl")
        log(f"시세 {min(i + 40, len(todo))}/{len(todo)}")
        time.sleep(1)


def get_facts(ciks):
    todo = [int(c) for c in dict.fromkeys(ciks) if pd.notna(c) and not (FS / f"{int(c)}.csv.gz").exists()]
    log(f"재무 대상 {len(set(ciks))}, 남은 것 {len(todo)}")
    for n, cik in enumerate(todo, 1):
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
        j = None
        for k in range(4):
            try:
                r = requests.get(url, headers=H, timeout=90)
            except Exception as e:  # noqa
                log(f"재무 오류 {cik} {e}")
                time.sleep(5)
                continue
            time.sleep(0.12)
            if r.status_code == 200:
                j = r.json()
                break
            if r.status_code == 404:
                break
            time.sleep(5 * (k + 1))
        rows = []
        if j:
            facts = j.get("facts", {})
            for tax, names in CONCEPTS.items():
                for nm in names:
                    node = facts.get(tax, {}).get(nm)
                    if not node:
                        continue
                    for unit, arr in node.get("units", {}).items():
                        for f in arr:
                            rows.append((tax + ":" + nm, unit, f.get("start"), f.get("end"), f.get("val"),
                                         f.get("filed"), f.get("form"), f.get("fp"), f.get("fy"), f.get("frame")))
        df = pd.DataFrame(rows, columns=["concept", "unit", "start", "end", "val", "filed", "form", "fp", "fy", "frame"])
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        (FS / f"{cik}.csv.gz").write_bytes(gzip.compress(buf.getvalue()))
        if n % 50 == 0:
            log(f"재무 {n}/{len(todo)}")


def main():
    sec = pd.read_csv(HERE / "results" / "securities.csv", dtype={"cusip": str})
    sp = sp500_list()
    sp = sp[pd.to_datetime(sp["end_date"].fillna("2100-01-01")) >= "2011-01-01"]
    ct = json.loads((C / "company_tickers.json").read_text())
    t2c = {v["ticker"].upper(): int(v["cik_str"]) for v in ct.values()}
    sp_t = [t.replace(".", "-") for t in sp["ticker"]]
    tick = MARKET + [t for t in sec["yf"].dropna()] + sp_t
    get_prices(tick)
    ciks = list(sec["cik"].dropna().astype(int)) + [t2c[t] for t in sp_t if t in t2c]
    get_facts(ciks)
    log("완료")
    print("완료")


if __name__ == "__main__":
    main()
