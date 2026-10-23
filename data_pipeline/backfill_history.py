"""장기 이력 이어받기 수집 (백그라운드용). 중간에 끊겨도 다시 실행하면 이어서 진행한다.

1) 국내: pykrx 날짜별 전체 시장(KOSPI·KOSDAQ·KONEX, 상장폐지 종목 포함) 원본(비수정) 시세를
   `kor_ohlcv_raw` 에 저장. 1999-01-04 ~ 오늘(장 마감 후). 완료일은 `kor_ohlcv_raw_done` 에 기록(휴장일 포함).
2) 미국: us_price 를 1995-01-01 부터 보강(collect_daily.collect_us 재사용). 생존편향은 README 참고.
진행 상황은 quant/logs/backfill_YYYYMMDD.log 에만 남긴다. 기존 테이블(kor_price 등)은 건드리지 않는다.
"""
import logging
import os
import time as _time
import warnings
from datetime import date, datetime, time, timedelta
from pathlib import Path

import collect_daily as cd

warnings.filterwarnings("ignore")
KR_START = date(1999, 1, 4)   # 거래량이 있는 KRX 전종목 시세의 최초일(1996~98은 거래량 0)
BASE = Path(os.environ.get("QUANT_HOME", Path(__file__).resolve().parent))   # .env·logs·cache 위치(기본: 이 폴더)
log = logging.getLogger("backfill")
log.setLevel(logging.INFO)
fh = logging.FileHandler(BASE / "logs" / f"backfill_{date.today():%Y%m%d}.log", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
log.addHandler(fh)
cd.log = log   # collect_daily 의 로그도 같은 파일로

DDL1 = """CREATE TABLE IF NOT EXISTS kor_ohlcv_raw (
  `날짜` date NOT NULL, `종목코드` varchar(6) NOT NULL,
  `시가` double, `고가` double, `저가` double, `종가` double,
  `거래량` double, `거래대금` double, `등락률` double, `시가총액` double,
  PRIMARY KEY (`날짜`,`종목코드`)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""
DDL2 = "CREATE TABLE IF NOT EXISTS kor_ohlcv_raw_done (`날짜` date NOT NULL PRIMARY KEY, n int) ENGINE=InnoDB"
SQL = ("INSERT INTO kor_ohlcv_raw (`날짜`,`종목코드`,`시가`,`고가`,`저가`,`종가`,`거래량`,`거래대금`,`등락률`,`시가총액`) "
       "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE `시가`=VALUES(`시가`),`고가`=VALUES(`고가`),"
       "`저가`=VALUES(`저가`),`종가`=VALUES(`종가`),`거래량`=VALUES(`거래량`),`거래대금`=VALUES(`거래대금`),"
       "`등락률`=VALUES(`등락률`),`시가총액`=VALUES(`시가총액`)")


def kr_raw(conn):
    from pykrx import stock
    with conn.cursor() as cur:
        cur.execute(DDL1); cur.execute(DDL2)
        cur.execute("SELECT `날짜` FROM kor_ohlcv_raw_done")
        done = {r[0] for r in cur.fetchall()}
    conn.commit()
    end = date.today() if datetime.now().time() >= time(15, 45) else date.today() - timedelta(days=1)
    days = [KR_START + timedelta(days=i) for i in range((end - KR_START).days + 1)]
    days = [d for d in days if d.weekday() < 5 and d not in done]
    log.info("KR 원본 시세 수집 대상 %d일 (%s ~ %s)", len(days), days[0] if days else "-", days[-1] if days else "-")
    fails, total = [], 0
    for k, d in enumerate(days):
        ds = d.strftime("%Y%m%d")
        df = None
        for attempt in range(4):
            try:
                df = stock.get_market_ohlcv(ds, market="ALL")
                break
            except Exception as ex:
                log.warning("KR %s 재시도 %d: %s", ds, attempt + 1, str(ex)[:80])
                _time.sleep(3 * (attempt + 1))
        if df is None:
            fails.append(ds); continue
        n = 0
        if len(df) and df["거래량"].sum() > 0:
            rows = [(d, code, cd.clean(r["시가"]), cd.clean(r["고가"]), cd.clean(r["저가"]), cd.clean(r["종가"]),
                     cd.clean(r["거래량"]), cd.clean(r.get("거래대금")), cd.clean(r.get("등락률")), cd.clean(r.get("시가총액")))
                    for code, r in df.iterrows()]
            n = cd.upsert(conn, SQL, rows); total += n
        with conn.cursor() as cur:
            cur.execute("INSERT IGNORE INTO kor_ohlcv_raw_done (`날짜`,n) VALUES (%s,%s)", (d, n))
        conn.commit()
        if k % 100 == 0:
            log.info("KR 진행 %d/%d (%s) 누적 %d행", k + 1, len(days), ds, total)
    log.info("KR 원본 시세 완료: %d행, 실패일 %d %s", total, len(fails), fails[:20])
    return total, fails


def main():
    log.info("=== backfill 시작 ===")
    conn = cd.connect()
    try:
        kr_raw(conn)
        cd.US_FIRST_START = date(1995, 1, 1)
        us = cd.collect_us(conn)
        log.info("US 보강 완료: %s", us)
    finally:
        conn.close()
    (BASE / "logs" / "backfill_DONE.flag").write_text(str(datetime.now()), encoding="utf-8")
    log.info("=== backfill 종료 ===")


if __name__ == "__main__":
    main()
