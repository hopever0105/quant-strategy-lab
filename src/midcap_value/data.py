# -*- coding: utf-8 -*-
"""래리 윌리엄스 추천 중형 가치주 ETF 검증 - 데이터 로드

미국: 야후 일봉(수정종가, 분배금 재투자) - VOE·Fidelity 펀드·비교 ETF
장기: Ken French 25 포트폴리오(시가총액 5 × 장부가/시가 5, 월간, 1926~) - 중형·가치 칸을 ETF 이전까지 연장
국내: larry_williams 의 상장폐지 포함 패널(수정종가)·시가총액·PBR·DART 재무를 그대로 재사용
캐시: quant/cache/midcap_value/  (같은 입력 → 같은 결과)
KRX 요청 없음.
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
JOB = next((p for p in HERE.parents if (p / "quant" / "krx_guard.py").exists()), HERE.parents[1])
QUANT = JOB / "quant"
CACHE = QUANT / "cache" / "midcap_value"
CACHE.mkdir(parents=True, exist_ok=True)
LOG = QUANT / "logs" / "midcap_value.log"
LARRY = HERE.parent / "larry_williams"


def _larry_data():
    """larry_williams/data.py 를 다른 이름으로 불러옴(같은 파일명 충돌 방지)"""
    import importlib.util
    if "larry_data" in sys.modules:
        return sys.modules["larry_data"]
    src = LARRY / "data.py"
    if not src.exists():                                   # 저장소: 같은 폴더의 사본(larry_panel.py)
        src = HERE / "larry_panel.py"
    spec = importlib.util.spec_from_file_location("larry_data", src)
    m = importlib.util.module_from_spec(spec)
    sys.modules["larry_data"] = m
    spec.loader.exec_module(m)
    return m

END = "2026-11-13"

# 이름: (티커, 설명)
US = {
    "VOE": "Vanguard Mid-Cap Value ETF (CRSP 미국 중형 가치, 2006-08~)",
    "VO": "Vanguard Mid-Cap ETF (CRSP 미국 중형 전체, 2004-01~)",
    "IWS": "iShares Russell Midcap Value ETF (2001-07~)",
    "FSMVX": "Fidelity Mid Cap Value Fund (액티브, 2001-11~)",
    "FLPSX": "Fidelity Low-Priced Stock Fund (액티브 중소형 가치, 1989~)",
    "IJJ": "iShares S&P MidCap 400 Value ETF (2000-07~)",
    "IJH": "iShares S&P MidCap 400 ETF (중형 전체, 2000-05~)",
    "IWD": "iShares Russell 1000 Value ETF (대형 가치, 2000-05~)",
    "SPY": "SPDR S&P 500 ETF",
    "QQQ": "Invesco QQQ (나스닥100)",
}


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def us_prices(refresh=False):
    """수정종가(분배금 재투자) 와이드. 캐시 우선"""
    fn = CACHE / "us_adjclose.pkl"
    if fn.exists() and not refresh:
        return pd.read_pickle(fn)
    import yfinance as yf
    d = yf.download(list(US), start="1989-01-01", end=END, auto_adjust=True, actions=False, progress=False, threads=False)
    px = d["Close"].copy()
    px.index = pd.to_datetime(px.index).tz_localize(None)
    px = px[list(US)]
    px.to_pickle(fn)
    log(f"야후 {len(US)}개 저장 {px.shape}, 시작일 " + ", ".join(f"{c} {px[c].first_valid_index():%Y-%m}" for c in px))
    return px


def us_returns():
    px = us_prices()
    r = px.pct_change(fill_method=None)
    # 하루 ±40% 넘는 값은 오류로 보고 0 처리(펀드 분배 기준가 오류 대비). 건수 기록
    bad = r.abs() > 0.40
    if bad.any().any():
        log(f"하루 ±40% 초과 {int(bad.sum().sum())}건 0 처리: " + ", ".join(f"{c}:{int(bad[c].sum())}" for c in r if bad[c].any()))
    return r.mask(bad, 0.0)


def ff6_daily():
    return pd.read_pickle(QUANT / "cache" / "druck13f" / "ff6_daily.pkl")


def _french_zip(name):
    fn = CACHE / f"{name}.csv"
    if fn.exists():
        return fn.read_text(encoding="latin-1")
    url = f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
    z = zipfile.ZipFile(io.BytesIO(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60).content))
    txt = z.read(z.namelist()[0]).decode("latin-1")
    fn.write_text(txt, encoding="latin-1")
    log(f"French {name} 저장")
    return txt


def _first_monthly_block(txt):
    """French CSV 첫 블록(가치가중 월간 수익률) → DataFrame(소수)"""
    lines = txt.splitlines()
    out, hdr, started = [], None, False
    for ln in lines:
        s = ln.strip()
        if not started:
            if s.startswith(",") and hdr is None and len(s) > 5:
                hdr = [x.strip() for x in s.split(",")[1:]]
                started = True
            continue
        parts = [x.strip() for x in s.split(",")]
        if len(parts[0]) == 6 and parts[0].isdigit():
            out.append(parts)
        elif out:
            break
    df = pd.DataFrame([p[1:] for p in out], index=[p[0] for p in out], columns=hdr).astype(float) / 100
    df.index = pd.to_datetime(df.index, format="%Y%m") + pd.offsets.MonthEnd(0)
    return df.replace([-99.99 / 100, -999 / 100], np.nan)


def ff25_monthly():
    """25 포트폴리오(시가총액 5 × B/M 5) 가치가중 월간 + FF 3팩터 월간(시장·무위험)"""
    p = _first_monthly_block(_french_zip("25_Portfolios_5x5"))
    f = _first_monthly_block(_french_zip("F-F_Research_Data_Factors"))
    return p, f


def dgs10():
    """미국 10년 국채 금리(일간, 1962~). FRED 공식 CSV, 캐시 우선"""
    fn = CACHE / "FRED_DGS10_full.csv"
    if not fn.exists():
        try:
            import FinanceDataReader as fdr
            d = fdr.DataReader("FRED:DGS10", "1962-01-01")
            d.iloc[:, :1].to_csv(fn, encoding="utf-8")
        except Exception as e:                           # FRED 접속 실패 → 야후 ^TNX(10년 금리, 1962~)
            import yfinance as yf
            d = yf.download("^TNX", start="1962-01-01", end=END, auto_adjust=False, progress=False)["Close"]
            d.index = pd.to_datetime(d.index).tz_localize(None)
            d.iloc[:, :1].to_csv(fn, encoding="utf-8")
            log(f"FRED 실패({type(e).__name__}) → 야후 ^TNX 사용")
        log("10년 금리 전체 저장")
    d = pd.read_csv(fn)
    d.columns = ["d", "v"]
    s = pd.to_numeric(d.v, errors="coerce")
    s.index = pd.to_datetime(d.d)
    return s.dropna()


# ---------------- 국내 ----------------
def kr_inputs():
    """larry_williams 패널 재사용. 반환 dict(W, info, fs, pbr, kospi)"""
    LD = _larry_data()
    W = LD.panel_wide()
    info = LD.listing_info()
    fs, n_fill = LD.fundamentals_filled()
    fp = pd.read_pickle(LD.CACHE / "factor_panel_v2.pkl")[["code", "date", "PBR", "mcap"]]
    k, _ = LD.kospi()
    return dict(W=W, info=info, fs=fs, n_fill=n_fill, fp=fp, kospi=k, LD=LD)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    px = us_prices(refresh="--refresh" in sys.argv)
    print(px.apply(lambda s: s.first_valid_index()).to_string())
    print(px.tail(3))
    p, f = ff25_monthly()
    print(p.shape, p.index[0], p.index[-1], list(p.columns)[:6])
    print(f.tail(2))
    print(dgs10().tail(2))
