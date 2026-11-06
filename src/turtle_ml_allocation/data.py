"""데이터: FinanceDataReader(야후·FRED)로 받아 quant/cache/turtle_ml 에 csv로 저장(있으면 재사용 = 이어받기).
KRX 요청은 하나도 없다.

- 가격은 배당·분할을 반영한 수정종가(Adj Close)를 쓴다.
- ETF 상장 전 구간은 대용 펀드·지수의 일간 수익률로 이어 붙인다(첫 ETF 가격에 맞춰 뒤로 늘림).
- 기준 달력은 미국 증시 거래일(SPY, 1993년 이전은 VFINX). 환율처럼 휴일이 다른 자료는 직전 값으로 채운다(최대 5일).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import FinanceDataReader as fdr

HERE = Path(__file__).resolve().parent
QUANT = HERE.parents[1]
sys.path.insert(0, str(QUANT))
try:
    from price_adjust import detect_jumps  # noqa: E402  (공용: 보정 후 남은 하루 ±35% 단절 점검)
except ImportError:                          # 저장소에 공용 모듈이 없을 때
    def detect_jumps(s, thresh=0.35):
        r = s.pct_change()
        return r[r.abs() > thresh].dropna()

sys.path.insert(0, str(HERE))
from universe import UNIVERSE, MACRO  # noqa: E402

CACHE = QUANT / "cache" / "turtle_ml"
CACHE.mkdir(parents=True, exist_ok=True)
LOG = QUANT / "logs" / "turtle_ml_data.log"
END = "2026-11-04"          # 모든 자료가 있는 마지막 날 (FRED DTB3 기준)


def _log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def fetch(sym, refresh=False):
    """심볼 하나를 받아 Series(수정종가 또는 FRED 값)로 돌려준다. 캐시가 있으면 다시 받지 않는다."""
    fn = CACHE / (sym.replace(":", "_").replace("=", "_").replace("^", "_").replace(".", "_") + ".csv")
    if fn.exists() and not refresh:
        s = pd.read_csv(fn, index_col=0, parse_dates=True).iloc[:, 0]
        return s
    for k in range(3):
        try:
            d = fdr.DataReader(sym, "1980-01-01")
            break
        except Exception as e:  # 네트워크 오류만 재시도 (KRX 아님)
            _log(f"{sym} 실패 {k+1}회: {e}")
            time.sleep(3)
    else:
        raise RuntimeError(f"{sym} 다운로드 실패")
    col = "Adj Close" if "Adj Close" in d.columns else d.columns[0]
    s = d[col].astype(float).dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None) if getattr(s.index, "tz", None) else pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = sym
    s.to_frame().to_csv(fn, encoding="utf-8")
    _log(f"{sym} 저장 {s.index.min().date()}~{s.index.max().date()} {len(s)}행")
    return s


def calendar():
    spy = fetch("SPY")
    vf = fetch("VFINX")
    cal = vf.index[vf.index < spy.index[0]].union(spy.index)
    return cal[cal <= END]


def _on_cal(s, cal):
    return s.reindex(s.index.union(cal)).ffill(limit=5).reindex(cal)


def build(refresh=False):
    """반환: prices(DataFrame, 이어 붙인 가격), info(DataFrame: 자산별 시작일·이어붙임 정보), macro(DataFrame)"""
    cal = calendar()
    prices, rows = {}, []
    for key, name, grp, etf, proxy, tf in UNIVERSE:
        e = _on_cal(fetch(etf, refresh), cal)
        e_first = e.first_valid_index()
        row = dict(key=key, name=name, group=grp, etf=etf, etf_start=e_first.date(), proxy=proxy or "-")
        if proxy:
            p = fetch(proxy, refresh)
            if tf == "inv":
                p = 1.0 / p
            p = _on_cal(p, cal)
            pr, er = p.pct_change(fill_method=None), e.pct_change(fill_method=None)
            ov = pd.concat([pr, er], axis=1).dropna()
            ov = ov[ov.index > e_first]
            # 겹치는 기간 추적 오차
            yrs = len(ov) / 252
            # 환율(런던 시각 고시)·채권펀드(장 마감 후 평가)는 하루 시차가 있어 일간 상관이 낮게 나온다 → 주간 상관도 본다
            wk = (1 + ov).resample("W-FRI").prod() - 1
            row.update(
                corr_w=round(wk.iloc[:, 0].corr(wk.iloc[:, 1]), 4),
                proxy_start=p.first_valid_index().date(),
                overlap_years=round(yrs, 1),
                corr=round(ov.iloc[:, 0].corr(ov.iloc[:, 1]), 4),
                ann_diff=round(((1 + ov.iloc[:, 0]).prod() ** (1 / yrs) - (1 + ov.iloc[:, 1]).prod() ** (1 / yrs)), 4),
                te=round((ov.iloc[:, 0] - ov.iloc[:, 1]).std() * np.sqrt(252), 4),
            )
            # 이어 붙이기: ETF 첫날 이전은 대용 수익률로 뒤로 늘린다
            r = er.copy()
            pre = cal < e_first
            r[pre] = pr[pre]
            r[cal == e_first] = 0.0
            r = r.fillna(0.0)
            first = p.first_valid_index()
            r[cal <= first] = np.nan
            px = (1 + r.fillna(0)).cumprod()
            px = px / px.loc[e_first] * e.loc[e_first]
            px[cal < first] = np.nan
            px[cal >= e_first] = e[cal >= e_first]
            s = px
        else:
            s = e
            row.update(proxy_start=None, overlap_years=None, corr_w=None, corr=None, ann_diff=None, te=None)
        s = s.ffill(limit=5)
        jumps = detect_jumps(s.dropna(), 0.25)
        row["start"] = s.first_valid_index().date()
        row["jumps_25pct"] = len(jumps)
        row["jump_dates"] = ", ".join(f"{d.date()}({v:+.0%})" for d, v in jumps.items())
        prices[key] = s
        rows.append(row)
    P = pd.DataFrame(prices)
    info = pd.DataFrame(rows).set_index("key")
    mac = pd.DataFrame({k: _on_cal(fetch(v, refresh), cal) for k, v in MACRO.items()})
    _log(f"build 완료 {P.shape}")
    return P, info, mac


if __name__ == "__main__":
    P, info, mac = build()
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print(info.drop(columns=["jump_dates"]))
    print(info[info.jumps_25pct > 0][["jump_dates"]])
    print(mac.describe())
