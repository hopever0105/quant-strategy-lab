"""1999~2018-07-03 원본(kor_ohlcv_raw)을 수정주가로 보정하고 kor_price(2015~, 수정주가)에 이어 붙인다. KRX 요청 없음(DB만).

보정 방법
1) 가격 단절(액면분할·병합·감자·권리락 등) 탐지: KRX 등락률은 '기준가' 대비라서 분할일에도 정상 등락률로 나온다.
   기준가 = 종가 / (1 + 등락률/100),  f = 전일종가 / 기준가.  |f-1| > 1% 이면 사건으로 보고 그 이전 가격에 1/f 를 곱한다.
   (전일 행과 20일 넘게 떨어진 행, 등락률 결측은 제외)
2) 연결 지점 재조정: kor_price(또는 상장폐지는 kor_price_delisted)와 겹치는 첫 날(보통 2015-01-02)의
   기준 종가 / 보정한 종가 = g 를 종목별로 구해 앞 구간 전체에 곱한다(2018-07 이후 사건까지 반영되는 효과).
3) 연결 지점에서 마지막 2014 종가 → 첫 2015 종가가 ±30% 넘게 튀는 종목 수를 보고서에 기록.
결과: (a) kor_price 에 2015-01-02 이전 행 INSERT IGNORE (kor_price 에 있는 종목만),
      (b) quant/cache/kor_panel.pkl : 상장폐지 포함 전체 패널(src R=원본보정, K=kor_price, D=상장폐지 종가만)
"""
import os
import sys
import time as _t
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql

BASE = Path(os.environ.get("QUANT_HOME", Path(__file__).resolve().parent))   # .env·logs·cache 위치(기본: 이 폴더)
(BASE / "cache").mkdir(exist_ok=True)
EV_TOL = 0.01
JUNC = date(2015, 1, 2)
RAW_END = date(2018, 7, 3)
OUT = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s); OUT.append(s)


def conn():
    env = dict(l.split("=", 1) for l in (BASE / ".env").read_text(encoding="utf-8").splitlines() if "=" in l and not l.startswith("#"))
    return pymysql.connect(host=env["DB_HOST"], user=env["DB_USER"], password=env["DB_PASSWORD"], database=env["DB_NAME"], charset="utf8mb4")


def main():
    c = conn(); t0 = _t.time()
    raw = pd.read_sql("SELECT `날짜` d, `종목코드` code, `시가` o, `고가` h, `저가` l, `종가` c, `거래량` v, `거래대금` amt, `등락률` r "
                      "FROM kor_ohlcv_raw WHERE `종가`>0", c, parse_dates=["d"])
    raw.sort_values(["code", "d"], inplace=True, ignore_index=True)
    p("raw rows", len(raw), "tickers", raw.code.nunique(), "기간", raw.d.min().date(), raw.d.max().date(), f"({_t.time()-t0:.0f}s)")

    # 1) 사건 탐지
    g = raw.groupby("code", sort=False)
    prev_c = g["c"].shift(1); prev_d = g["d"].shift(1)
    base = raw.c / (1 + raw.r / 100)
    f = prev_c / base
    gap = (raw.d - prev_d).dt.days
    ev = prev_c.notna() & raw.r.notna() & (base > 0) & ((f - 1).abs() > EV_TOL) & (gap <= 20)
    p("가격 단절 사건 수", int(ev.sum()), "영향 종목", raw.loc[ev, "code"].nunique(),
      "| f<0.5 또는 f>2 (분할·병합급)", int(((f < 0.5) | (f > 2))[ev].sum()))
    inv = np.where(ev, 1.0 / f, 1.0)
    m = np.ones(len(raw))
    codes = raw.code.values
    bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        x = inv[a:b]
        rc = np.cumprod(x[::-1])[::-1]
        m[a:b - 1] = rc[1:]
    for col in ("o", "h", "l", "c"):
        raw[col] = raw[col] * m
    raw["v"] = raw["v"] / m
    p("보정 완료", f"({_t.time()-t0:.0f}s)")

    # 2) 연결 지점 재조정 g
    kp = pd.read_sql("SELECT `날짜` d, `종목코드` code, `시가` o, `고가` h, `저가` l, `종가` c, `거래량` v FROM kor_price WHERE `날짜`>=%s", c,
                     params=[JUNC], parse_dates=["d"])
    surv = set(kp[kp.d < pd.Timestamp('2026-10-01')].code.unique())   # 최근 며칠만 들어간 종목(우선주 등)은 이력 없는 것으로 간주
    dl = pd.read_sql("SELECT `날짜` d, `종목코드` code, `종가` c, `시가총액` mcap FROM kor_price_delisted WHERE `날짜`>=%s", c,
                     params=[JUNC], parse_dates=["d"])
    dl = dl[~dl.code.isin(surv)]
    ref = pd.concat([kp[["d", "code", "c"]], dl[["d", "code", "c"]]], ignore_index=True)
    ov = raw[raw.d >= pd.Timestamp(JUNC)][["d", "code", "c"]].merge(ref, on=["d", "code"], suffixes=("_mine", "_ref"))
    ov["ratio"] = ov.c_ref / ov.c_mine
    ov.sort_values(["code", "d"], inplace=True)
    anchor = ov.groupby("code").first()["ratio"]
    spread = ov.groupby("code")["ratio"].agg(lambda s: s.max() / s.min() - 1)
    p("겹침 비교 종목", len(anchor), "| 앵커비율 g 중앙값", round(float(anchor.median()), 4),
      "| g가 1과 1% 이상 다른 종목", int(((anchor - 1).abs() > 0.01).sum()),
      "| 겹침 구간 비율이 2% 넘게 변하는 종목(사건 탐지 불일치)", int((spread > 0.02).sum()))
    gmap = anchor.reindex(raw.code.unique()).fillna(1.0)
    graw = raw.code.map(gmap).values
    for col in ("o", "h", "l", "c"):
        raw[col] = raw[col] * graw
    raw["v"] = raw["v"] / graw

    # 3) 연결 지점 점프 (마지막 2014 vs 첫 2015)
    pre = raw[raw.d < pd.Timestamp(JUNC)]
    last14 = pre.sort_values("d").groupby("code").last()[["d", "c"]]
    first15 = ref.sort_values("d").groupby("code").first()
    j = last14.join(first15, lsuffix="_14", rsuffix="_15", how="inner")
    j = j[(j.d_15 - j.d_14).dt.days <= 14]
    j["jump"] = j.c_15 / j.c_14 - 1
    p("연결 지점 비교 종목", len(j), "| |점프|>30%:", int((j.jump.abs() > 0.30).sum()), "| >15%:", int((j.jump.abs() > 0.15).sum()),
      "| 중앙값", f"{j.jump.median():+.4f}")
    big = j[j.jump.abs() > 0.30].sort_values("jump")
    p("±30% 초과 종목(최대 30개):", ", ".join(f"{k}({v:+.0%})" for k, v in big.jump.head(30).items()))

    # 4) kor_price 앞 구간 삽입 (kor_price 에 있는 종목만)
    ins = raw[(raw.d < pd.Timestamp(JUNC)) & raw.code.isin(surv)]
    rows = [(d.date(), o, h, l, cc, v, code) for d, o, h, l, cc, v, code in zip(ins.d, ins.o, ins.h, ins.l, ins.c, ins.v, ins.code)]
    cur = c.cursor(); n = 0
    for i in range(0, len(rows), 20000):
        cur.executemany("INSERT IGNORE INTO kor_price (`날짜`,`시가`,`고가`,`저가`,`종가`,`거래량`,`종목코드`) VALUES (%s,%s,%s,%s,%s,%s,%s)", rows[i:i + 20000])
        n += len(rows[i:i + 20000])
    c.commit()
    p("kor_price 앞 구간 INSERT IGNORE 시도", n, "행 (", len(set(ins.code)), "종목 )", f"({_t.time()-t0:.0f}s)")

    # 5) 전체 패널 캐시 (상장폐지 포함)
    R = raw[(raw.d < pd.Timestamp(JUNC)) | (~raw.code.isin(surv))].copy()
    R["src"] = "R"
    # 거래대금: 원본(비수정, 원) — 보정하지 않음
    K = kp.copy()
    amt_raw = pd.read_sql("SELECT `날짜` d, `종목코드` code, `거래대금` amt FROM kor_ohlcv_raw WHERE `날짜`>=%s", c, params=[JUNC], parse_dates=["d"])
    K = K.merge(amt_raw, on=["d", "code"], how="left")
    est = K.c * K.v
    chk = K.dropna(subset=["amt"]); chk = chk[(chk.amt > 0) & (est[chk.index] > 0)]
    p("kor_price 종가×거래량 / 원본 거래대금 중앙값(겹침)", round(float((est[chk.index] / chk.amt).median()), 3))
    K["amt"] = K.amt.fillna(est)
    K["src"] = "K"
    D = dl[dl.d > pd.Timestamp(RAW_END)].copy()
    D["o"] = D["h"] = D["l"] = D["c"]; D["v"] = np.nan; D["amt"] = np.nan; D["src"] = "D"
    panel = pd.concat([R[["d", "code", "o", "h", "l", "c", "v", "amt", "src"]], K[["d", "code", "o", "h", "l", "c", "v", "amt", "src"]],
                       D[["d", "code", "o", "h", "l", "c", "v", "amt", "src", "mcap"]]], ignore_index=True)
    panel.sort_values(["code", "d"], inplace=True, ignore_index=True)
    panel.to_pickle(BASE / "cache" / "kor_panel.pkl")
    p("패널", len(panel), "행,", panel.code.nunique(), "종목,", panel.d.min().date(), "~", panel.d.max().date(), "| src별", panel.src.value_counts().to_dict())
    (BASE / "logs" / "stitch_report.txt").write_text("\n".join(OUT), encoding="utf-8")


if __name__ == "__main__":
    main()
