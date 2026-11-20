# -*- coding: utf-8 -*-
"""래리 윌리엄스 검증 - 데이터 로드 (KRX 요청 없음)

출처
- 국내 주가: MySQL stock_db 를 이어 붙인 패널 quant/cache/kor_panel.pkl (1999~, 상장폐지 포함, 수정주가)
             + kor_ohlcv_raw(2018-07까지 시가총액) + kor_price_delisted(상장폐지 시가총액) + kor_ticker(최근 시가총액)
- 국내 재무: dart_fs (연간, 2013~2025, 단위 억원), kor_fs (법인세비용·법인세 납부, 2023~)
- 업종: 한국거래소 업종 분류 파일(2026-10-05 기준 코스피 942종목, '업종명') → 없으면 kor_sector 로 보완
- 코스피 일봉(시가·고가·저가·종가): 야후 ^KS11 (코스피200 일봉은 KRX 외 출처에 2014년 이전 자료가 없어 코스피로 대체)
- KODEX 200(069500): 네이버 (2014-08~, 보조 확인용)
- 미국·원자재: 야후 (SPY, TLT, ZB=F, GC=F, CL=F, USO), FRED (DTB3 미국 3개월 국채, IR3TIB01KRM156N 한국 3개월 금리)
- COT: CFTC 공식 API (Legacy Futures Only) 캐시 cot_<코드>.csv

모든 외부 자료는 quant/cache/larry/ 에 csv로 저장해 다시 받지 않는다(이어받기). 같은 입력 → 같은 결과.
"""
import io
import json
import sys
import time
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
JOB = next((p for p in HERE.parents if (p / "quant" / "krx_guard.py").exists()), HERE.parents[1])  # 저장소에서는 상위 폴더 기준   # 취업 폴더
QUANT = JOB / "quant"
CACHE = QUANT / "cache" / "larry"
CACHE.mkdir(parents=True, exist_ok=True)
COT_CACHE = HERE.parent.parent / "cache" / "cftc"           # probe.py 가 받아 둔 위치(04_.../01_퀀트/cache/cftc)
LOG = QUANT / "logs" / "larry_williams_data.log"
PANEL = QUANT / "cache" / "kor_panel.pkl"
TIMEFOLIO = CACHE / "krx_sector_20261005.csv"          # 한국거래소 업종 분류(코스피, 2026-10-05) 내려받은 파일

START = "1999-01-01"
END = "2026-11-11"          # 미국 자료 마지막 공통일 (국내 패널은 2026-10-22까지)

COT_MARKETS = {             # CFTC 코드: (이름, 가격 키)
    "13874A": ("S&P500", "SPX"),
    "020601": ("미국 30년 국채", "BOND"),
    "088691": ("금", "GOLD"),
    "067651": ("원유(WTI)", "OIL"),
}


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {msg}\n")


# ---------------------------------------------------------------- 외부 시세
def _fdr(sym, start="1990-01-01"):
    import FinanceDataReader as fdr
    for k in range(3):
        try:
            d = fdr.DataReader(sym, start)
            if len(d):
                return d
        except Exception as e:      # 네트워크 오류 재시도 (KRX 아님)
            log(f"{sym} 실패 {k + 1}회: {e}")
            time.sleep(3)
    raise RuntimeError(f"{sym} 다운로드 실패")


def ohlc(sym, refresh=False):
    """야후·네이버 일봉. 반환 DataFrame[o,h,l,c,adj]"""
    fn = CACHE / (sym.replace(":", "_").replace("=", "_").replace("^", "") + ".csv")
    if fn.exists() and not refresh:
        return pd.read_csv(fn, index_col=0, parse_dates=True)
    d = _fdr(sym)
    d.index = pd.to_datetime(d.index)
    if getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_localize(None)
    out = pd.DataFrame({"o": d["Open"], "h": d["High"], "l": d["Low"], "c": d["Close"]}).astype(float)
    out["adj"] = d["Adj Close"].astype(float) if "Adj Close" in d.columns else out["c"]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.to_csv(fn, encoding="utf-8")
    log(f"{sym} 저장 {out.index.min().date()}~{out.index.max().date()} {len(out)}행")
    return out


def fred(code, refresh=False):
    fn = CACHE / f"FRED_{code}.csv"
    if fn.exists() and not refresh:
        return pd.read_csv(fn, index_col=0, parse_dates=True).iloc[:, 0]
    d = _fdr(f"FRED:{code}", "1990-01-01")
    s = d.iloc[:, 0].astype(float).dropna()
    s.to_frame(code).to_csv(fn, encoding="utf-8")
    log(f"FRED {code} 저장 {len(s)}행")
    return s


def cot(code, refresh=False):
    """CFTC 상업자 순포지션(주간, 화요일 기준). 캐시 우선"""
    fn = COT_CACHE / f"cot_{code}.csv"
    if fn.exists() and not refresh:
        df = pd.read_csv(fn, parse_dates=["date"])
    else:
        q = {"$select": "report_date_as_yyyy_mm_dd,comm_positions_long_all,comm_positions_short_all,open_interest_all",
             "$where": f"cftc_contract_market_code='{code}'", "$order": "report_date_as_yyyy_mm_dd", "$limit": "50000"}
        url = "https://publicreporting.cftc.gov/resource/6dca-aqww.json?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(url, timeout=60) as r:
            rows = json.loads(r.read().decode("utf-8"))
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).dt.normalize()
        for c in ["comm_positions_long_all", "comm_positions_short_all", "open_interest_all"]:
            df[c] = pd.to_numeric(df[c])
        df = df.groupby("date", as_index=False).sum(numeric_only=True)
        df["comm_net"] = df["comm_positions_long_all"] - df["comm_positions_short_all"]
        df = df[["date", "comm_net", "open_interest_all"]]
        COT_CACHE.mkdir(parents=True, exist_ok=True)
        df.to_csv(fn, index=False, encoding="utf-8-sig")
    return df.set_index("date")["comm_net"].sort_index()


def _ret_clean(px, alt=None):
    """종가 → 일간 수익률. 0 이하 가격(2020-04 원유 마이너스)이 낀 날은 대체 자산(alt) 수익률로 채운다."""
    r = px.pct_change()
    bad = (px <= 0) | (px.shift(1) <= 0)
    if alt is not None:
        r[bad] = alt.pct_change().reindex(r.index)[bad]
    r[bad & r.isna()] = 0.0
    return r.fillna(0.0), int(bad.sum())


def us_assets():
    """미국·원자재 일간 수익률 DataFrame + 가격 지수(1에서 시작) + 보정 내역"""
    spy = ohlc("YAHOO:SPY")["adj"]
    tlt = ohlc("YAHOO:TLT")["adj"]
    zb = ohlc("YAHOO:ZB=F")["c"]
    gc = ohlc("YAHOO:GC=F")["c"]
    cl = ohlc("YAHOO:CL=F")["c"]
    uso = ohlc("YAHOO:USO")["adj"]
    cal = spy.index[(spy.index >= "1999-01-01") & (spy.index <= END)]
    f = lambda s: s.reindex(s.index.union(cal)).ffill(limit=5).reindex(cal)
    R = pd.DataFrame(index=cal)
    R["SPX"] = f(spy).pct_change()
    # 30년 국채: TLT 상장(2002-07-30) 전은 ZB=F 수익률로 이어 붙임
    rb_t, rb_z = f(tlt).pct_change(), f(zb).pct_change()
    R["BOND"] = rb_t.where(rb_t.index > tlt.index[0], rb_z)
    R["GOLD"] = f(gc).pct_change()
    r_oil, n_bad = _ret_clean(f(cl), f(uso))
    R["OIL"] = r_oil
    # 연속선물 만기 교체 보정: 하루 ±25% 넘는 움직임 중 다음 날 반대로 되돌린 것은 교체 단절로 보고 0 처리
    fixes = {"OIL_neg_price_days": n_bad}
    for k in ["BOND", "GOLD", "OIL"]:
        r = R[k]
        jump = (r.abs() > 0.25) & ((r + r.shift(-1)).abs() < 0.05)
        fixes[f"{k}_roll_jumps"] = int(jump.sum())
        R.loc[jump, k] = 0.0
    R = R.fillna(0.0)
    P = (1 + R).cumprod()
    return R, P, fixes


def cash_rates(kr_cal, us_cal):
    """연율(%) 금리를 일간 수익률로. 미국 DTB3(일간), 한국 3개월 금리(월간, OECD/FRED)"""
    us = fred("DTB3")
    kr = fred("IR3TIB01KRM156N")
    us_d = us.reindex(us.index.union(us_cal)).ffill().reindex(us_cal).fillna(method="bfill") / 100 / 252
    kr_d = kr.reindex(kr.index.union(kr_cal)).ffill().reindex(kr_cal).fillna(method="bfill") / 100 / 252
    return kr_d, us_d


def kospi():
    """코스피 일봉 (야후 ^KS11). 시가가 전일 종가와 같은(시가 미기록) 날 비율도 돌려준다"""
    d = ohlc("YAHOO:^KS11")
    d = d[(d.index >= "1999-01-01")]
    d = d[(d.h > 0) & (d.l > 0) & (d.c > 0)].copy()
    bad_open = (d.o <= 0) | d.o.isna()
    d.loc[bad_open, "o"] = d["c"].shift(1)[bad_open]
    d["h"] = d[["h", "o", "c"]].max(axis=1)
    d["l"] = d[["l", "o", "c"]].min(axis=1)
    same = (d.o == d.c.shift(1)).groupby(d.index.year).mean()
    return d.dropna(), same


def kodex200():
    d = ohlc("NAVER:069500")
    return d


# ---------------------------------------------------------------- 국내 주식 패널
def _db():
    import pymysql
    env = dict(l.split("=", 1) for l in (QUANT / ".env").read_text(encoding="utf-8").splitlines()
               if "=" in l and not l.startswith("#"))
    return pymysql.connect(host=env["DB_HOST"], user=env["DB_USER"], password=env.get("DB_PASSWORD", env.get("DB_PASS")),
                           database=env["DB_NAME"], charset="utf8mb4")


def _sql(q):
    c = _db()
    try:
        return pd.read_sql(q, c)
    finally:
        c.close()


USE_DELIST = True          # 2018-07 이후 상장폐지 종목을 KRX 일봉으로 보강한 패널을 쓸지 (2026-11-12 보완)


def panel_long():
    """기존 패널(kor_panel.pkl) + KRX 상장폐지 일봉 보강. 반환 long[d, code, h, l, c, amt, src]
    보강 방법: 상장폐지 종목별로 KRX 일봉 첫날 이후 구간을 KRX 값으로 바꾼다(기존은 종가만 있던 구간).
    가격 수준은 겹치는 날(KRX 첫날 이전 마지막 기존 종가 ↔ 그날 KRX 종가) 비율로 맞춰 이어 붙인다."""
    fn = CACHE / "panel_long_v2.pkl"
    Y = delist_px() if USE_DELIST and (CACHE / "delist").exists() else None
    if fn.exists() and (Y is None or fn.stat().st_mtime >= (CACHE / "delist_px.pkl").stat().st_mtime):
        return pd.read_pickle(fn)
    P = pd.read_pickle(PANEL)[["d", "code", "h", "l", "c", "amt", "src"]]
    P = P[P.c > 0]
    if Y is None or not len(Y):
        return P
    keep, add, n_scale = [], [], 0
    first = Y.groupby("code").d.min()
    P = P.set_index("code")
    for code, g in Y.groupby("code"):
        d0 = first[code]
        if code in P.index:
            old = P.loc[[code]]
            base = old[old.d <= d0]
            if len(base):
                ref_d = base.d.max()
                ref_c = base.loc[base.d == ref_d, "c"].iloc[0]
                y_at = g[g.d >= ref_d].iloc[0].c
                k = ref_c / y_at if y_at > 0 else 1.0
                n_scale += 1
            else:
                k = 1.0
        else:
            k = 1.0
        gg = g.assign(h=g.h * k, l=g.l * k, c=g.c * k, src="X")[["d", "code", "h", "l", "c", "amt", "src"]]
        add.append(gg[gg.d >= d0])
    drop = P.reset_index()
    drop = drop.merge(first.rename("d0"), left_on="code", right_index=True, how="left")
    drop = drop[drop.d0.isna() | (drop.d < drop.d0)].drop(columns="d0")
    out = pd.concat([drop] + add, ignore_index=True).sort_values(["code", "d"])
    out.to_pickle(fn)
    log(f"panel_long_v2: 상장폐지 보강 {len(add)}종목(기존과 이어 붙임 {n_scale}), 행 {len(out)}")
    return out


def panel_wide(refresh=False):
    """종가·고가·저가 와이드 표(float32). 하루 가격제한폭을 넘는 수익률은 가격 단절로 보고 그날 수익률을 0으로 둔다.
    반환: dict(c, h, l, ret, breaks(DataFrame), last_date)"""
    fn = CACHE / ("panel_wide_v2.pkl" if USE_DELIST else "panel_wide.pkl")
    if fn.exists() and not refresh and (not USE_DELIST or fn.stat().st_mtime >= (CACHE / "panel_long_v2.pkl").stat().st_mtime):
        return pd.read_pickle(fn)
    P = panel_long() if USE_DELIST else pd.read_pickle(PANEL)[["d", "code", "h", "l", "c", "src"]]
    P = P[P.c > 0]
    C = P.pivot(index="d", columns="code", values="c").astype("float32")
    H = P.pivot(index="d", columns="code", values="h").astype("float32")
    L = P.pivot(index="d", columns="code", values="l").astype("float32")
    # 상장폐지 종가만 있는 행(src D)은 고가·저가가 없으므로 종가로 채움
    H = H.where(H > 0, C)
    L = L.where(L > 0, C)
    R = C.pct_change(fill_method=None)
    lim = pd.Series(np.where(R.index < "2015-06-15", 0.155, 0.305), index=R.index)
    brk = R.abs().gt(lim, axis=0)
    if USE_DELIST and "src" in P.columns:
        # KRX 상장폐지 일봉(src X)은 등락률로 이미 분할·감자를 보정했고, 정리매매(가격제한 없음) 폭락은 실제 손실이라 0 처리하지 않음
        X = (P.assign(x=(P.src == "X")).pivot(index="d", columns="code", values="x")
             .reindex(index=brk.index, columns=brk.columns).fillna(False).astype(bool))
        brk = brk & ~X
    st = brk.stack()
    st = st[st]
    breaks = pd.DataFrame({"date": st.index.get_level_values(0), "code": st.index.get_level_values(1)})
    breaks["ret"] = [float(R.at[d, c]) for d, c in zip(breaks.date, breaks.code)]
    R = R.mask(brk, 0.0).astype("float32")
    out = dict(c=C, h=H, l=L, ret=R, breaks=breaks, last_date=C.index.max())
    pd.to_pickle(out, fn)
    log(f"panel_wide 저장 {C.shape}, 가격제한폭 초과(단절) {len(breaks)}건 0 처리")
    return out


def delist_px():
    """KRX에서 받은 2018-07 이후 상장폐지 종목 일봉(krx_delist.py prices). 원본 가격 → 등락률로 연결한 수정가격.
    반환 long DataFrame[d, code, o, h, l, c, amt, mcap(억원)]"""
    fn = CACHE / "delist_px.pkl"
    files = sorted((CACHE / "delist").glob("px_*.csv"))
    if fn.exists() and fn.stat().st_mtime >= max((f.stat().st_mtime for f in files), default=0):
        return pd.read_pickle(fn)
    num = lambda s: pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")
    parts = []
    for f in files:
        try:
            x = pd.read_csv(f, dtype=str)
        except pd.errors.EmptyDataError:                    # 그 구간 거래 없음(거래정지 등)
            continue
        if not len(x):
            continue
        parts.append(x)
    X = pd.concat(parts, ignore_index=True)
    X["d"] = pd.to_datetime(X.TRD_DD, format="%Y/%m/%d")
    X["code"] = X.ISU_CD.str[-6:]
    for a, b in [("c", "TDD_CLSPRC"), ("o", "TDD_OPNPRC"), ("h", "TDD_HGPRC"), ("l", "TDD_LWPRC"), ("amt", "ACC_TRDVAL"),
                 ("mcap", "MKTCAP"), ("rt", "FLUC_RT")]:
        X[a] = num(X[b])
    X = X.drop_duplicates(["code", "d"]).sort_values(["code", "d"])
    X = X[X.c > 0]
    out = []
    for code, g in X.groupby("code"):
        g = g.copy()
        # 연결 수정가격: 첫날 원본 종가에서 시작해 KRX 등락률(기준가 대비, 분할·감자 반영)로 이어 감
        rt = g.rt.fillna(0) / 100
        adj = g.c.iloc[0] * (1 + rt).cumprod() / (1 + rt.iloc[0])
        f = adj / g.c
        g["o"], g["h"], g["l"], g["c"] = g.o * f, g.h * f, g.l * f, adj
        g["mcap"] = g.mcap / 1e8
        out.append(g[["d", "code", "o", "h", "l", "c", "amt", "mcap"]])
    Y = pd.concat(out, ignore_index=True)
    Y.to_pickle(fn)
    return Y


def listing_info():
    """종목명·시장·업종 정보. 반환 DataFrame(index=code; name, market, sector, sector_src, fin)"""
    fn = CACHE / ("listing_info_v4.csv" if USE_DELIST else "listing_info_v2.csv")
    if fn.exists():
        return pd.read_csv(fn, dtype={"code": str}).set_index("code")
    tf = pd.read_csv(TIMEFOLIO, encoding="cp949", dtype=str)
    tf = tf.rename(columns={"종목코드": "code", "종목명": "name", "시장구분": "market", "업종명": "sector"})[["code", "name", "market", "sector"]]
    tf["sector_src"] = "KRX업종"
    kt = _sql("SELECT `종목코드` code, `종목명` name, `시장구분` market FROM kor_ticker ORDER BY `기준일`").drop_duplicates("code", keep="last")
    ks = _sql("SELECT CMP_CD code, CMP_KOR name, SEC_NM_KOR sector FROM kor_sector ORDER BY `기준일`").drop_duplicates("code", keep="last")
    kd = _sql("SELECT `종목코드` code, `종목명` name, `상장폐지일` delist FROM kor_delisted")
    info = tf.set_index("code")
    rest = pd.concat([kt.set_index("code"), ks.set_index("code")[["sector"]].rename(columns={"sector": "sector_ks"})], axis=1)
    rest = rest.join(kd.set_index("code")[["name"]].rename(columns={"name": "name_d"}), how="outer")
    rest = rest.join(ks.set_index("code")[["name"]].rename(columns={"name": "name_ks"}), how="left")
    rest["name"] = rest["name"].fillna(rest["name_d"]).fillna(rest["name_ks"])
    rest = rest[~rest.index.isin(info.index)]
    rest["sector"] = rest["sector_ks"]
    rest["sector_src"] = np.where(rest["sector"].notna(), "kor_sector", "없음")
    info = pd.concat([info, rest[["name", "market", "sector", "sector_src"]]])
    info = info.join(kd.set_index("code")[["delist"]])
    dl = CACHE / "delist" / "delist_list.csv"
    if USE_DELIST and dl.exists():                            # KRX 상장폐지 목록(2015~)으로 이름·시장·상장폐지일 보완
        k = pd.read_csv(dl, dtype=str)
        k["code"] = k.ISU_CD.str[-6:]
        k = k[k.SECUGRP_NM == "주권"].drop_duplicates("code", keep="last").set_index("code")
        new = k.index.difference(info.index)
        add = pd.DataFrame({"name": k.loc[new, "ISU_NM"], "market": k.loc[new, "MKT_NM"], "sector": np.nan, "sector_src": "없음"})
        info = pd.concat([info, add])
        info["name"] = info["name"].fillna(k.ISU_NM.reindex(info.index))
        info["market"] = info["market"].fillna(k.MKT_NM.reindex(info.index))
        info["delist"] = info["delist"].fillna(pd.to_datetime(k.DELIST_DD, format="%Y/%m/%d", errors="coerce").reindex(info.index))
    fin_sec = {"은행", "증권", "보험", "금융"}
    name = info["name"].fillna("")
    info["fin"] = info["sector"].isin(fin_sec) | name.str.contains("은행|증권|보험|화재|생명|캐피탈|카드|금융지주|투자금융|종금|저축")
    info.index.name = "code"
    info.reset_index().to_csv(fn, index=False, encoding="utf-8-sig")
    return info


def fundamentals():
    """연간 재무(억원). 반환 DataFrame(index=[code, fy]) 열: liab, equity, ni, sales, op, cfo, assets"""
    fn = CACHE / "dart_fs.csv"
    if fn.exists():
        return pd.read_csv(fn, dtype={"code": str}).set_index(["code", "fy"])
    d = _sql("SELECT `종목코드` code, `계정` acc, YEAR(`기준일`) fy, `값` v FROM dart_fs")
    w = d.pivot_table(index=["code", "fy"], columns="acc", values="v", aggfunc="last")
    w.reset_index().to_csv(fn, index=False, encoding="utf-8-sig")
    return w


DART_DIR = CACHE / "dart"


def _amt(x):
    try:
        return float(str(x).replace(",", "")) / 1e8          # 원 → 억원
    except Exception:
        return np.nan


def dart_multi():
    """OpenDART 다중회사 주요계정(2015~2025, dart_fill.py multi 결과). 연결(CFS) 우선, 없으면 별도(OFS).
    반환 DataFrame(index=[code, fy]) 열: liab, equity, sales, op, pretax, ni"""
    fn = CACHE / "dart_multi.csv"
    files = sorted(DART_DIR.glob("multi_*.json"))
    if fn.exists() and fn.stat().st_mtime >= max((f.stat().st_mtime for f in files), default=0):
        return pd.read_csv(fn, dtype={"code": str}).set_index(["code", "fy"])
    rows = []
    for f in files:
        for x in json.loads(f.read_text(encoding="utf-8")):
            rows.append((x.get("stock_code"), int(x.get("bsns_year")), x.get("fs_div"), x.get("sj_div"), x.get("account_nm", "").replace(" ", ""),
                         _amt(x.get("thstrm_amount"))))
    d = pd.DataFrame(rows, columns=["code", "fy", "fs", "sj", "acc", "v"])
    amap = {"부채총계": "liab", "자본총계": "equity", "매출액": "sales", "영업이익": "op", "법인세차감전순이익": "pretax", "당기순이익": "ni"}
    d["k"] = d.acc.map(lambda a: next((v for k, v in amap.items() if a.startswith(k)), None))
    d = d[d.k.notna()]
    d["pri"] = (d.fs == "CFS").astype(int)
    d = d.sort_values("pri").drop_duplicates(["code", "fy", "k"], keep="last")
    w = d.pivot_table(index=["code", "fy"], columns="k", values="v", aggfunc="last")
    w.reset_index().to_csv(fn, index=False, encoding="utf-8-sig")
    return w


def fundamentals_filled():
    """dart_fs(기존 DB)를 기본으로, 없는 (종목, 연도)는 DART 다중회사 주요계정으로 채움. pretax 열 추가.
    반환 (DataFrame, 채운 건수)"""
    base = fundamentals().copy()
    m = dart_multi()
    base["src"] = "dart_fs"
    add = m.loc[~m.index.isin(base.index)].copy()
    add["src"] = "DART보강"
    out = pd.concat([base, add])
    out["pretax"] = m["pretax"].reindex(out.index)
    return out.sort_index(), int(len(add))


def tax_table():
    """(종목, 연도)별 법인세비용·납부액(억원).
    - 2023~: kor_fs(법인세비용, 법인세 납부)
    - 2015~2022: 법인세비용 = DART 법인세차감전순이익 − 당기순이익, 납부액 = DART 전체 재무제표 현금흐름표 '법인세' 납부 계정(후보 종목만 수집)"""
    k = tax_fs()[["tax_exp", "tax_paid", "pretax"]]
    m = dart_multi()
    t = pd.DataFrame(index=m.index)
    t["pretax"] = m["pretax"]
    t["tax_exp"] = m["pretax"] - m["ni"]
    paid = {}
    for f in DART_DIR.glob("tax_*.json"):
        _, code, fy = f.stem.split("_")
        rows = json.loads(f.read_text(encoding="utf-8"))
        # 회사마다 부호 표기가 달라서('법인세환입(납부)' 양수가 납부인 회사도 있음) 크기로 판단:
        # '납부·지급'이 들어간 현금흐름표 줄의 절댓값 = 납부액, 환급 줄만 있으면 음수(미납부)
        cf = [r for r in rows if r.get("sj") == "CF"]
        pay = [r for r in cf if ("납부" in r["acc"] or "지급" in r["acc"]) and _amt(r["v"]) == _amt(r["v"])]
        ref = [r for r in cf if "환급" in r["acc"] and "납부" not in r["acc"] and _amt(r["v"]) == _amt(r["v"])]
        v = np.nan
        if pay:
            v = abs(_amt(pay[0]["v"]))
        elif ref:
            v = -abs(_amt(ref[0]["v"]))
        elif rows is not None and len(rows) == 0:
            v = np.nan
        else:
            v = 0.0                                           # 재무제표는 있는데 납부 줄이 없음
        paid[(code, int(fy))] = v
    t["tax_paid"] = pd.Series(paid).reindex(t.index) if paid else np.nan
    t = t[t.index.get_level_values(1) < 2023]
    return pd.concat([t, k[k.index.get_level_values(1) >= 2023]]).sort_index()


def tax_fs():
    """법인세비용·법인세 납부(현금흐름표) - kor_fs 연간(2023~). 납부는 음수가 납부라서 부호를 뒤집어 '납부액'으로"""
    fn = CACHE / "kor_fs_tax.csv"
    if fn.exists():
        return pd.read_csv(fn, dtype={"code": str}).set_index(["code", "fy"])
    d = _sql("SELECT `종목코드` code, `계정` acc, YEAR(`기준일`) fy, `값` v FROM kor_fs "
             "WHERE `공시구분`='y' AND `계정` IN ('법인세비용','법인세환입(납부)','법인세비용차감전계속사업이익')")
    w = d.pivot_table(index=["code", "fy"], columns="acc", values="v", aggfunc="last")
    w = w.rename(columns={"법인세비용": "tax_exp", "법인세환입(납부)": "tax_cf", "법인세비용차감전계속사업이익": "pretax"})
    w["tax_paid"] = -w["tax_cf"]
    w.reset_index().to_csv(fn, index=False, encoding="utf-8-sig")
    return w


def market_caps(dates, closes):
    """리밸런싱 날짜별 시가총액(억원).
    - 2018-07-03 이전: kor_ohlcv_raw 시가총액(원본)
    - 상장폐지 종목: kor_price_delisted 시가총액
    - 그 밖(2018-07 이후 생존 종목): 주식 수 = 시가총액/종가 를 2018-07-03 과 최근(kor_ticker) 두 시점에서 구해
      날짜 비율로 선형 보간 × 그날 수정종가 (수정종가라 분할은 자동 반영, 그 사이 증자는 보간으로 근사)"""
    fn = CACHE / ("mcap_rebal_v2.pkl" if USE_DELIST else "mcap_rebal.pkl")
    key = tuple(str(pd.Timestamp(x).date()) for x in dates)
    if fn.exists():
        obj = pd.read_pickle(fn)
        if obj["key"] == key:
            return obj["mcap"]
    out = {}
    raw_end = pd.Timestamp("2018-07-03")
    ds = ",".join(f"'{pd.Timestamp(x).date()}'" for x in dates if pd.Timestamp(x) <= raw_end)
    raw = _sql(f"SELECT `날짜` d, `종목코드` code, `시가총액` m FROM kor_ohlcv_raw WHERE `날짜` IN ({ds})") if ds else pd.DataFrame(columns=["d", "code", "m"])
    raw["d"] = pd.to_datetime(raw["d"])
    s18 = _sql("SELECT `종목코드` code, `시가총액` m, `종가` c FROM kor_ohlcv_raw WHERE `날짜`='2018-07-03'").set_index("code")
    kt = _sql("SELECT `종목코드` code, `시가총액` m, `종가` c, `기준일` d FROM kor_ticker ORDER BY `기준일`").drop_duplicates("code", keep="last").set_index("code")
    dl = _sql("SELECT `날짜` d, `종목코드` code, `시가총액` m FROM kor_price_delisted")
    dl["d"] = pd.to_datetime(dl["d"])
    dlw = dl.pivot(index="d", columns="code", values="m")
    # 주식 수(수정주가 기준): 시가총액 / 그날 수정종가
    adj18 = closes.loc[:raw_end].iloc[-1]
    sh18 = (s18["m"] / adj18.reindex(s18.index).astype(float)).replace([np.inf, -np.inf], np.nan)
    last_adj = closes.ffill().iloc[-1]
    shnow = (kt["m"] / last_adj.reindex(kt.index).astype(float)).replace([np.inf, -np.inf], np.nan)
    tnow = pd.to_datetime(kt["d"]).max()
    Yd = None
    if USE_DELIST and (CACHE / "delist").exists():
        Yd = delist_px().pivot(index="d", columns="code", values="mcap")
    for d in dates:
        d = pd.Timestamp(d)
        px = closes.loc[:d].iloc[-1].astype(float)
        if d <= raw_end:
            m = raw[raw.d == raw.d[raw.d <= d].max()].set_index("code")["m"] if (raw.d <= d).any() else pd.Series(dtype=float)
        else:
            w = (d - raw_end) / (tnow - raw_end)
            sh = pd.concat([sh18.rename("a"), shnow.rename("b")], axis=1)
            sh["s"] = np.where(sh.a.notna() & sh.b.notna(), sh.a * (1 - w) + sh.b * w, sh.a.fillna(sh.b))
            m = (sh["s"] * px.reindex(sh.index))
        # 상장폐지 종목은 실제 시가총액으로 덮어씀
        if d >= dlw.index.min():
            row = dlw.loc[:d]
            if len(row):
                m = m.copy()
                v = row.iloc[-1].dropna()
                m = m.reindex(m.index.union(v.index))
                m.loc[v.index] = v
        m = (m / 1e8).astype(float)
        if USE_DELIST and Yd is not None:                     # KRX 상장폐지 일봉의 실제 시가총액(억원)으로 덮어씀
            row = Yd.loc[:d]
            if len(row):
                v = row.iloc[-5:].ffill().iloc[-1].dropna()
                v = v[v.index.isin(closes.columns) & closes.loc[:d].iloc[-1].reindex(v.index).notna()]
                m = m.reindex(m.index.union(v.index))
                m.loc[v.index] = v
        out[d] = m
    mc = pd.DataFrame(out).T
    pd.to_pickle({"key": key, "mcap": mc}, fn)
    return mc


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    R, P, fx = us_assets(); print(R.tail(), fx)
    k, same = kospi(); print(k.tail(), same.tail())
    print(kodex200().tail())
    W = panel_wide(); print(W["c"].shape, len(W["breaks"]))
    print(listing_info().sector_src.value_counts())
    print(fundamentals().shape, tax_fs().shape)
    for c in COT_MARKETS:
        print(c, cot(c).tail(2))
