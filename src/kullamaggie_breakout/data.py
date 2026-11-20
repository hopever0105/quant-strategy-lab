# -*- coding: utf-8 -*-
"""쿨라마기 EMA 돌파 검증 - 데이터 로드 (KRX 요청 없음)

- 국내 일봉: quant/ohlc_panel.py (MySQL stock_db 1999~ + 2018-07 이후 상장폐지 KRX 보강, 가격 단절 보정, 상장폐지 포함)
- 종목 정보: 래리 검증에서 만든 listing_info_v4.csv (시장·종목명). 정보가 없는 코드는 대부분 2018년 이전 상장폐지 종목 → 보통주 코드면 포함
- 지수: 야후 ^KS11(코스피), ^KQ11(코스닥) — FinanceDataReader, 캐시 quant/cache/kullamaggie/
- 현금 금리: FRED 한국 3개월 금리(래리 검증 캐시)
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
JOB = next((p for p in [Path.cwd(), *Path.cwd().parents, *HERE.parents] if (p / "quant" / "ohlc_panel.py").exists()), None)
if JOB is None:
    JOB = Path(r"C:\Users\arlo0\OneDrive\바탕 화면\취업")
QUANT = JOB / "quant"
sys.path.insert(0, str(QUANT))
import ohlc_panel  # noqa: E402

CACHE = QUANT / "cache" / "kullamaggie"
CACHE.mkdir(parents=True, exist_ok=True)
LARRY = QUANT / "cache" / "larry"
LOG = QUANT / "logs" / "kullamaggie.log"

START, END = "2000-01-01", "2026-10-22"
SEGS = {"학습": ("2000-01-01", "2014-12-31"), "검증": ("2015-01-01", "2019-12-31"),
        "테스트": ("2020-01-01", END), "전체": ("2000-01-01", END)}


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def index_ohlc(sym):
    """야후 지수 일봉. 반환 DataFrame[o,h,l,c]"""
    fn = CACHE / (sym.replace("^", "") + ".csv")
    if fn.exists():
        return pd.read_csv(fn, index_col=0, parse_dates=True)
    if sym == "^KS11" and (LARRY / "YAHOO_KS11.csv").exists():
        d = pd.read_csv(LARRY / "YAHOO_KS11.csv", index_col=0, parse_dates=True)[["o", "h", "l", "c"]]
    else:
        import FinanceDataReader as fdr
        d = None
        for k in range(3):
            try:
                x = fdr.DataReader("YAHOO:" + sym, "1996-01-01")
                if len(x):
                    d = pd.DataFrame({"o": x["Open"], "h": x["High"], "l": x["Low"], "c": x["Close"]}).astype(float)
                    break
            except Exception as e:  # 네트워크 재시도 (KRX 아님)
                log(f"{sym} 실패 {k + 1}: {e}")
                time.sleep(3)
        if d is None:
            raise RuntimeError(sym)
        d.index = pd.to_datetime(d.index).tz_localize(None) if getattr(d.index, "tz", None) else pd.to_datetime(d.index)
    d = d[d.c > 0]
    d.to_csv(fn)
    return d


def cash_rate(cal):
    s = pd.read_csv(LARRY / "FRED_IR3TIB01KRM156N.csv", index_col=0, parse_dates=True).iloc[:, 0]
    return s.reindex(s.index.union(cal)).ffill().bfill().reindex(cal) / 100 / 252


def universe_codes(cols):
    """코스피·코스닥 보통주(우선주·스팩·리츠·코넥스 제외). 반환 (codes, info)"""
    info = pd.read_csv(LARRY / "listing_info_v4.csv", dtype={"code": str}).set_index("code")
    name = info["name"].fillna("")
    bad = name.str.contains("스팩|기업인수목적|리츠|REIT|인프라|선박투자|맥쿼리") | (info["market"] == "KONEX")
    ok = [c for c in cols if c.isdigit() and c.endswith("0") and not bad.get(c, False)]
    return ok, info


def load():
    """반환 dict: o,h,l,c,amt (numpy float64, 날짜×종목), dates, codes, lim(np), kospi, kosdaq, rf, info, breaks"""
    fn = CACHE / "arrays.pkl"
    if fn.exists() and fn.stat().st_mtime >= ohlc_panel.OUT.stat().st_mtime:
        return pd.read_pickle(fn)
    P = ohlc_panel.build()
    codes, info = universe_codes(P["c"].columns)
    dates = P["c"].index
    out = {k: P[k][codes].to_numpy(dtype="float64") for k in ["o", "h", "l", "c", "amt"]}
    out.update(dates=dates, codes=np.array(codes), lim=P["lim"].to_numpy(), info=info,
               breaks=P["breaks"][P["breaks"].code.isin(codes)])
    ks, kq = index_ohlc("^KS11"), index_ohlc("^KQ11")
    out["kospi"] = ks.c.reindex(dates).ffill()
    out["kosdaq"] = kq.c.reindex(dates).ffill()
    out["rf"] = cash_rate(dates)
    pd.to_pickle(out, fn)
    log(f"arrays 저장: {len(dates)}일 × {len(codes)}종목, 단절 보정 {len(out['breaks'])}건")
    return out


if __name__ == "__main__":
    D = load()
    print(D["c"].shape, len(D["breaks"]), D["kosdaq"].first_valid_index())
