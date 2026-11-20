# -*- coding: utf-8 -*-
"""유튜브 전략 시봉 재검증 - 데이터 (KRX 요청 없음)

- 시봉: stock_db.kor_price_60m (quant/collect_hourly.py, 야후 60분봉, 하루 6개 봉 09~14시, 14시 봉에 장 마감까지 포함)
- 일봉: quant/ohlc_panel.py (수정주가, 상장폐지 포함) — 일봉 비교와 유동성(전날까지의 거래대금 평균) 계산에 사용
- 기간: 시봉과 일봉이 겹치는 2023-11-14 ~ 2026-10-22
- 같은 종목 집합: 시봉이 있는 종목만 일봉에서도 쓴다(일봉·시봉 비교를 같은 조건으로)
- 시봉 가격 단절(분할 등 야후 미보정)은 일봉 수정주가 비율로 맞춘다: 날짜별 (일봉 수정종가 ÷ 시봉 마지막 종가)를 그날 시봉 전체에 곱함.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
JOB = next((p for p in [Path.cwd(), *Path.cwd().parents, *HERE.parents] if (p / "quant" / "ohlc_panel.py").exists()),
           Path(r"C:\Users\arlo0\OneDrive\바탕 화면\취업"))
QUANT = JOB / "quant"
sys.path.insert(0, str(QUANT))
import ohlc_panel  # noqa: E402

CACHE = QUANT / "cache" / "hourly"
CACHE.mkdir(parents=True, exist_ok=True)
LOG = QUANT / "logs" / "hourly_youtube.log"
H_START, H_END = "2023-11-14", "2026-10-22"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {m}\n")


def _db():
    import pymysql
    env = dict(l.split("=", 1) for l in (QUANT / ".env").read_text(encoding="utf-8").splitlines()
               if "=" in l and not l.startswith("#"))
    return pymysql.connect(host=env["DB_HOST"], user=env["DB_USER"], password=env.get("DB_PASSWORD", env.get("DB_PASS")),
                           database=env["DB_NAME"], charset="utf8mb4")


def hourly_raw(refresh=False):
    fn = CACHE / "h60_raw.pkl"
    if fn.exists() and not refresh:
        return pd.read_pickle(fn)
    c = _db()
    try:
        h = pd.read_sql(f"SELECT `일시` t, `종목코드` code, `시가` o, `고가` h, `저가` l, `종가` c, `거래량` v FROM kor_price_60m "
                        f"WHERE `일시` >= '{H_START}' AND `일시` < DATE_ADD('{H_END}', INTERVAL 1 DAY)", c)
    finally:
        c.close()
    h["t"] = pd.to_datetime(h.t)
    h = h[(h.c > 0) & (h.h >= h.l)].sort_values(["code", "t"]).reset_index(drop=True)
    h.to_pickle(fn)
    return h


def daily():
    P = ohlc_panel.build()
    return P


def build(refresh=False):
    """반환 dict:
       H: 시봉 DataFrame[t, d, code, o,h,l,c,v] — 일봉 수정주가 수준으로 맞추고, 매일 15:30 동시호가 봉(공식 종가)을 붙인 것
       Dd: 일봉 와이드 dict(o,h,l,c,amt) (기간 + 400일 워밍업), codes, check(대조 통계)
    야후 시봉은 14시 봉이 15:20 전에 끝나 장 마감 동시호가(가격·거래량)가 빠져 있다 → 마지막에 종가 봉을 붙인다.
    가격 배율 k = 일봉 수정 시가 ÷ 시봉 첫 봉 시가 (시가는 99% 일치)."""
    fn = CACHE / "h60_built.pkl"
    if fn.exists() and not refresh:
        return pd.read_pickle(fn)
    h = hourly_raw(refresh)
    P = daily()
    codes = sorted(set(h.code) & set(P["c"].columns))
    h = h[h.code.isin(codes)].copy()
    h["d"] = h.t.dt.normalize()
    sl = lambda k: P[k][codes].loc[H_START:H_END].stack().rename(k).reset_index().set_axis(["d", "code", k], axis=1)
    D = sl("o").merge(sl("h"), on=["d", "code"]).merge(sl("l"), on=["d", "code"]).merge(sl("c"), on=["d", "code"])
    D.columns = ["d", "code", "do", "dh", "dl", "dc"]
    kp = pd.read_pickle(QUANT / "cache" / "kor_panel.pkl")
    kp = kp[(kp.d >= H_START) & kp.code.isin(codes)][["d", "code", "v"]].rename(columns={"v": "dv"})
    g = h.groupby(["code", "d"]).agg(ho=("o", "first"), hh=("h", "max"), hl=("l", "min"), hc=("c", "last"), hv=("v", "sum")).reset_index()
    m = g.merge(D, on=["code", "d"], how="inner").merge(kp, on=["code", "d"], how="left")
    m["k"] = m["do"] / m.ho
    m = m.sort_values(["code", "d"])
    m["k_chg"] = m.groupby("code").k.pct_change().abs()
    check = dict(일수=int(len(m)), 종목=int(m.code.nunique()),
                 시가배율변화_2pct초과_일수=int((m.k_chg > 0.02).sum()),
                 시가배율변화_10pct초과_일수=int((m.k_chg > 0.10).sum()),
                 고가일치_0_5pct=float(((m.hh * m.k / m.dh - 1).abs() < 0.005).mean()),
                 저가일치_0_5pct=float(((m.hl * m.k / m.dl - 1).abs() < 0.005).mean()),
                 마지막봉종가_공식종가_차이중앙값=float((m.hc * m.k / m.dc - 1).abs().median()),
                 시봉거래량_합_대비_일봉=float((m.hv / m.dv).median()))
    # 배율이 하루 사이 10% 넘게 뛰는 날은 시가 자체가 이상한 경우가 있어 그 종목·날짜는 뺀다
    bad = m.k_chg > 0.10
    m = m[~bad]
    h = h.merge(m[["code", "d", "k", "dc", "dv", "hv"]], on=["code", "d"], how="inner")
    for col in ["o", "h", "l", "c"]:
        h[col] = h[col] * h.k
    auc = m[["code", "d", "dc", "dv", "hv"]].copy()
    auc["t"] = auc.d + pd.Timedelta(hours=15, minutes=30)
    auc["o"] = auc["h"] = auc["l"] = auc["c"] = auc.dc
    auc["v"] = (auc.dv - auc.hv).clip(lower=0).fillna(0)
    h = pd.concat([h[["t", "d", "code", "o", "h", "l", "c", "v"]], auc[["t", "d", "code", "o", "h", "l", "c", "v"]]], ignore_index=True)
    h = h.sort_values(["code", "t"]).reset_index(drop=True)
    check["뺀_종목일"] = int(bad.sum())
    check["09시봉_거래량0_비율"] = float((h[h.t.dt.hour == 9].v == 0).mean())
    out = dict(H=h, codes=codes, check=check)
    a = pd.Timestamp(H_START) - pd.Timedelta(days=400)
    out["Dd"] = {k: P[k][codes].loc[a:H_END] for k in ["o", "h", "l", "c", "amt"]}
    kv = pd.read_pickle(QUANT / "cache" / "kor_panel.pkl")
    kv = kv[(kv.d >= a) & kv.code.isin(codes)]
    out["Dd"]["v"] = kv.pivot(index="d", columns="code", values="v").reindex(index=out["Dd"]["c"].index, columns=codes)
    out["lim"] = P["lim"].loc[a:H_END]
    pd.to_pickle(out, fn)
    log(f"시봉 준비: {len(codes)}종목, {len(h):,}봉, 대조 {check}")
    return out


if __name__ == "__main__":
    B = build(refresh=True)
    print(B["check"], len(B["codes"]), len(B["H"]))
