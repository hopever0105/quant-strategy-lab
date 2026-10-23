"""국내·미국 일별 시세를 MySQL stock_db에 upsert.

- 국내: kor_price(날짜, 시가, 고가, 저가, 종가, 거래량, 종목코드), kor_ticker(기준일 스냅샷) -- 기존 구조 그대로
- 미국: us_price(날짜, 티커, 시가, 고가, 저가, 종가, 수정종가, 거래량) -- 없으면 생성
- 휴장일이면 아무것도 넣지 않는다. 로그: quant/logs/collect_YYYYMMDD.log
"""
import logging
import os
import re
import sys
import warnings
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import pymysql

warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("QUANT_HOME", Path(__file__).resolve().parent))   # .env·logs·cache 위치(기본: 이 폴더)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import krx_guard            # KRX 요청 제한(간격 2초·하루 500건·연속 실패 3회 잠금). pykrx 등 import 전에 install
krx_guard.install()
LOG_DIR = BASE / "logs"
LOG_DIR.mkdir(exist_ok=True)
TODAY = date.today()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"collect_{TODAY:%Y%m%d}.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("collect")

US_ETFS = ["SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "SHY", "LQD", "HYG", "GLD", "SLV", "DBC",
           "EEM", "EFA", "VNQ", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "VTI", "VOO"]
US_FIRST_START = date(2015, 1, 1)  # 국내(kor_price 2015~)와 기간 맞춤


def load_env():
    env = {}
    for line in (BASE / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def connect():
    e = load_env()
    return pymysql.connect(host=e["DB_HOST"], port=int(e.get("DB_PORT", 3306)), user=e["DB_USER"],
                           password=e["DB_PASSWORD"], database=e["DB_NAME"], charset="utf8mb4", autocommit=False)


def clean(v):
    """NaN/inf -> None, numpy -> python."""
    if v is None or pd.isna(v) or v in (float("inf"), float("-inf")):
        return None
    return v.item() if hasattr(v, "item") else v


def upsert(conn, sql, rows, chunk=5000):
    n = 0
    with conn.cursor() as cur:
        for i in range(0, len(rows), chunk):
            cur.executemany(sql, rows[i:i + chunk])
            n += len(rows[i:i + chunk])
    conn.commit()
    return n


# ---------------------------------------------------------------- 국내
def collect_kr(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(`날짜`) FROM kor_price")
        last = cur.fetchone()[0]
    start = last + timedelta(days=1)
    pend = BASE / "logs" / "fallback_days.txt"
    if pend.exists() and pend.read_text(encoding="utf-8").split():
        start = min(start, date.fromisoformat(min(pend.read_text(encoding="utf-8").split())))   # 대체 출처 날짜를 KRX 값으로 교체
    try:
        from pykrx import stock          # import 시 KRX 로그인을 시도함 → 차단되면 여기서 KRXBlocked
    except krx_guard.KRXBlocked as ex:
        log.error("KRX 로그인/조회 불가(%s) → 대체 출처(FinanceDataReader/네이버)로 전환", ex)
        n, ld, fl = fdr_fallback(conn, start, TODAY)
        return n, ld or last, fl
    total, fail, days_done, last_day = 0, 0, 0, last
    sql = ("INSERT INTO kor_price (`날짜`,`시가`,`고가`,`저가`,`종가`,`거래량`,`종목코드`) VALUES (%s,%s,%s,%s,%s,%s,%s) "
           "ON DUPLICATE KEY UPDATE `시가`=VALUES(`시가`),`고가`=VALUES(`고가`),`저가`=VALUES(`저가`),"
           "`종가`=VALUES(`종가`),`거래량`=VALUES(`거래량`)")
    d = start
    blocked = False
    while d <= TODAY:
        if d.weekday() >= 5:
            d += timedelta(days=1)
            continue
        if d == TODAY and datetime.now().time() < time(15, 45):
            log.info("KR 오늘(%s)은 장 마감 전 - 건너뜀", d)
            break
        ds = d.strftime("%Y%m%d")
        try:
            df = stock.get_market_ohlcv(ds, market="ALL")
            if df is None or "거래량" not in df.columns:      # KRX가 빈 응답을 주면 pykrx는 열 없는 빈 표를 돌려줌 = 차단 의심
                raise krx_guard.KRXBlocked("KRX 빈 응답(열 없음)")
        except krx_guard.KRXBlocked as ex:
            log.error("KR %s KRX 조회 불가(%s) → 대체 출처(FinanceDataReader/네이버)로 전환", ds, ex)
            n, ld, fl = fdr_fallback(conn, d, TODAY)
            total += n; fail += fl; last_day = ld or last_day
            blocked = True
            break
        except Exception as ex:
            log.error("KR %s 수집 실패: %s", ds, ex)
            fail += 1
            d += timedelta(days=1)
            continue
        if df is None or df.empty or df["거래량"].sum() == 0:
            log.info("KR %s 휴장 또는 미집계 - 건너뜀", ds)
            d += timedelta(days=1)
            continue
        rows = [(d, clean(r["시가"]), clean(r["고가"]), clean(r["저가"]), clean(r["종가"]),
                 clean(r["거래량"]), code) for code, r in df.iterrows()]
        n = upsert(conn, sql, rows)
        total += n
        days_done += 1
        last_day = d
        log.info("KR %s kor_price %d행", ds, n)
        try:
            update_kr_ticker(conn, stock, d, df)
        except krx_guard.KRXBlocked as ex:
            log.error("KR %s kor_ticker 갱신 중단(KRX 제한): %s", ds, ex)
        except Exception as ex:
            log.error("KR %s kor_ticker 갱신 실패: %s", ds, ex)
            fail += 1
        d += timedelta(days=1)
    if not blocked and pend.exists():
        pend.write_text("", encoding="utf-8")          # KRX로 다시 받았으니 대체 출처 표시 해제
    return total, last_day, fail


def fdr_fallback(conn, d0, d1):
    """KRX를 못 쓸 때: 최근 kor_ticker 스냅샷의 종목을 FinanceDataReader(네이버, 수정주가)로 받아 kor_price에 upsert.
    kor_ticker(시총·EPS 등)는 갱신하지 않는다. 휴장일은 데이터가 없어 자연히 건너뛴다. 장 마감 전 오늘은 제외."""
    import FinanceDataReader as fdr
    from concurrent.futures import ThreadPoolExecutor
    end = d1 if datetime.now().time() >= time(15, 45) else d1 - timedelta(days=1)
    if end < d0:
        return 0, None, 0
    with conn.cursor() as cur:
        cur.execute("SELECT `종목코드` FROM kor_ticker WHERE `기준일`=(SELECT MAX(`기준일`) FROM kor_ticker)")
        codes = [r[0] for r in cur.fetchall()]
    sql = ("INSERT INTO kor_price (`날짜`,`시가`,`고가`,`저가`,`종가`,`거래량`,`종목코드`) VALUES (%s,%s,%s,%s,%s,%s,%s) "
           "ON DUPLICATE KEY UPDATE `시가`=VALUES(`시가`),`고가`=VALUES(`고가`),`저가`=VALUES(`저가`),"
           "`종가`=VALUES(`종가`),`거래량`=VALUES(`거래량`)")

    import requests
    sess = requests.Session()
    pat = re.compile(r'\["(\d{8})",\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)')

    def one(code):
        try:
            r = sess.get("https://api.finance.naver.com/siseJson.naver", timeout=10, headers={"User-Agent": "Mozilla/5.0"},
                         params=dict(symbol=code, requestType=1, startTime=d0.strftime("%Y%m%d"), endTime=end.strftime("%Y%m%d"), timeframe="day"))
            if r.status_code != 200:
                return code, None
            return code, [(datetime.strptime(m[0], "%Y%m%d").date(), float(m[1]), float(m[2]), float(m[3]), float(m[4]), float(m[5])) for m in pat.findall(r.text)]
        except Exception:
            return code, None

    rows, bad = [], 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, (code, x) in enumerate(ex.map(one, codes)):
            if i % 500 == 0:
                log.info("KR 대체 출처 진행 %d/%d", i, len(codes))
            if x is None:
                bad += 1; continue
            for dt, o_, h_, l_, c_, v_ in x:
                if c_ <= 0 or (v_ == 0 and o_ == 0):
                    continue
                rows.append((dt, o_, h_, l_, c_, v_, code))
    n = upsert(conn, sql, rows) if rows else 0
    ld = max((r[0] for r in rows), default=None)
    if rows:
        days = sorted({r[0] for r in rows})
        pend = BASE / "logs" / "fallback_days.txt"
        old = set(pend.read_text(encoding="utf-8").split()) if pend.exists() else set()
        pend.write_text("\n".join(sorted(old | {x.isoformat() for x in days})), encoding="utf-8")
    log.info("KR 대체 출처(네이버; KRX 공식 종가와 NXT 반영 등으로 소폭 다를 수 있음) %d행, 조회 실패 %d종목 (kor_ticker 미갱신, 기간 %s~%s)", n, bad, d0, end)
    return n, ld, bad


def update_kr_ticker(conn, stock, d, ohlcv):
    """기준일 d 스냅샷을 kor_ticker에 upsert (상장폐지·신규상장은 당일 전종목 목록 기준으로 자연 반영)."""
    ds = d.strftime("%Y%m%d")
    cap = stock.get_market_cap(ds, market="ALL")
    fund = stock.get_market_fundamental(ds, market="ALL")
    market = {}
    for m in ("KOSPI", "KOSDAQ"):
        for t in stock.get_market_ticker_list(ds, market=m):
            market[t] = m
    sql = ("INSERT INTO kor_ticker (`종목코드`,`종목명`,`시장구분`,`종가`,`시가총액`,`기준일`,`EPS`,`PER`,`BPS`,`주당배당금`,`종목구분`) "
           "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE `종목명`=VALUES(`종목명`),"
           "`시장구분`=VALUES(`시장구분`),`종가`=VALUES(`종가`),`시가총액`=VALUES(`시가총액`),`EPS`=VALUES(`EPS`),"
           "`PER`=VALUES(`PER`),`BPS`=VALUES(`BPS`),`주당배당금`=VALUES(`주당배당금`),`종목구분`=VALUES(`종목구분`)")
    rows = []
    for code, m in market.items():
        if code not in ohlcv.index:
            continue
        try:
            name = stock.get_market_ticker_name(code)
        except Exception:
            name = None
        c = cap.loc[code] if code in cap.index else None
        f = fund.loc[code] if code in fund.index else None
        kind = "보통주" if code.endswith("0") else "우선주"
        rows.append((code, name if isinstance(name, str) else None, m, clean(ohlcv.loc[code, "종가"]),
                     clean(c["시가총액"]) if c is not None else None, d,
                     clean(f["EPS"]) if f is not None else None, clean(f["PER"]) if f is not None else None,
                     clean(f["BPS"]) if f is not None else None, clean(f["DPS"]) if f is not None else None, kind))
    n = upsert(conn, sql, rows)
    log.info("KR %s kor_ticker %d행", ds, n)


# ---------------------------------------------------------------- 미국
def sp500_tickers():
    import FinanceDataReader as fdr
    df = fdr.StockListing("S&P500")
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    fix = {"BRKB": "BRK-B", "BFB": "BF-B"}
    return [fix.get(s, s) for s in (str(s).replace(".", "-") for s in df[col].dropna().unique())]


def collect_us(conn):
    import yfinance as yf

    with conn.cursor() as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS us_price (
            `날짜` date NOT NULL, `티커` varchar(20) NOT NULL,
            `시가` double DEFAULT NULL, `고가` double DEFAULT NULL, `저가` double DEFAULT NULL,
            `종가` double DEFAULT NULL, `수정종가` double DEFAULT NULL, `거래량` double DEFAULT NULL,
            PRIMARY KEY (`날짜`,`티커`)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
        cur.execute("SELECT MIN(`날짜`), MAX(`날짜`) FROM us_price")
        first, last = cur.fetchone()
    conn.commit()
    end = TODAY + timedelta(days=1)
    if first and first > US_FIRST_START + timedelta(days=7):
        # 시작일을 앞으로 늘린 경우: 기존 최초일까지 과거 구간을 다시 채운다(전체 구간 upsert)
        start = US_FIRST_START
        log.info("US 과거 구간 보강 %s ~ 현재 (기존 최초일 %s)", start, first)
    else:
        start = last + timedelta(days=1) if last else US_FIRST_START
    if start > TODAY:
        log.info("US 이미 최신(%s)", last)
        return 0, last, 0
    log.warning("US 유니버스 = 현재 S&P500 구성종목 + 주요 ETF 뿐 → 생존편향(과거 편출·상장폐지 종목 없음). "
                "미국 백테스트 결과는 상향 편향 가능. 자세한 내용 quant/README.md")
    try:
        tickers = sorted(set(sp500_tickers()) | set(US_ETFS))
    except Exception as ex:
        log.error("S&P500 목록 실패(%s) - ETF만 수집", ex)
        tickers = list(US_ETFS)

    sql = ("INSERT INTO us_price (`날짜`,`티커`,`시가`,`고가`,`저가`,`종가`,`수정종가`,`거래량`) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
           "ON DUPLICATE KEY UPDATE `시가`=VALUES(`시가`),`고가`=VALUES(`고가`),`저가`=VALUES(`저가`),"
           "`종가`=VALUES(`종가`),`수정종가`=VALUES(`수정종가`),`거래량`=VALUES(`거래량`)")
    total, failed, last_day = 0, [], last
    batch = 100
    for i in range(0, len(tickers), batch):
        sub = tickers[i:i + batch]
        try:
            df = yf.download(sub, start=start.isoformat(), end=end.isoformat(),
                             auto_adjust=False, group_by="ticker", progress=False, threads=True)
        except Exception as ex:
            log.error("US 배치 %d 실패: %s", i // batch, ex)
            failed += sub
            continue
        rows = []
        for t in sub:
            try:
                x = df[t].dropna(subset=["Close"])
            except Exception:
                x = pd.DataFrame()
            if x.empty:
                failed.append(t)
                continue
            for idx, r in x.iterrows():
                rows.append((idx.date(), t, clean(r["Open"]), clean(r["High"]), clean(r["Low"]),
                             clean(r["Close"]), clean(r["Adj Close"]), clean(r["Volume"])))
        if rows:
            total += upsert(conn, sql, rows)
            last_day = max([last_day] + [r[0] for r in rows]) if last_day else max(r[0] for r in rows)
    log.info("US us_price %d행, 데이터 없음/실패 %d종목 %s", total, len(failed), failed[:30])
    return total, last_day, len(failed)


def main():
    log.info("=== 수집 시작 %s ===", datetime.now())
    conn = connect()
    try:
        kr = collect_kr(conn)
        us = collect_us(conn)
    finally:
        conn.close()
    log.info("국내: %d행, 마지막 %s, 실패 %d / 미국: %d행, 마지막 %s, 실패 %d", *kr, *us)
    print(f"국내 {kr[0]}행(마지막 {kr[1]}, 실패 {kr[2]}) / 미국 {us[0]}행(마지막 {us[1]}, 실패 {us[2]})")


if __name__ == "__main__":
    main()
