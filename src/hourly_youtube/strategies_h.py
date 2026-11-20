# -*- coding: utf-8 -*-
"""유튜브 전략 3개를 일봉·시봉에 같은 규칙(봉 개수 기준 같은 숫자)으로 적용.

1. 로스 캐머런 MACD·RSI (macd_rsi_swing/core.py 의 신호·청산 함수를 그대로 import)
   매매법 1: MACD 골든크로스 + RSI 50~70 (범위 밖이면 대기), 2R 절반 익절 후 본전 손절, 잔량은 데드크로스·RSI≤50 다음 봉 시가
   매매법 2: RSI 하락 다이버전스 → 눌림(고점 85% 이상) → MACD 재골든크로스, 2R 전량 익절, 도달 전 데드크로스면 다음 봉 시가
   손절 = 직전 10봉 최저가(손절폭 3~15%만), 유동성 = 전날까지 20일 평균 거래대금 10억 이상, 비용 편도 0.115% + 매도세(연도·시장별)
2. 니라지 조시 OBV + 켈트너 (전략_하이킨아시_OBV켈트너/backtest_obv_keltner_v1.py 규칙)
   박스(직전 20봉 범위/종가 ≤ 5% 등) 첫 돌파 + OBV > OBV EMA20 → 다음 봉 시가, 손절 박스 하단(위험 1~25%),
   1R/2R/3R 분할(1/2·1/4·1/4), 1R 후 본전, 60봉 만료. 변형 A2·A5·A8·B·C·D·E. 왕복 비용 0.3%.
3. 하이킨아시 도지 반전 + EMA20 (backtest_heikin_v2.py + heikin_scanner_v2.py 규칙)
   도지(들) 뒤 평평한 바닥·긴 윗꼬리 양봉 + 거래량 증가 + 직전 12봉 음봉 비율 ≥ 65% + EMA20 조건(N 없음/R 위/S 돌파)
   → 다음 봉 시가, 손절 = 도지 최저가×0.995, 익절 2R, 스무스 HA 양→음 전환 다음 봉 시가, 60봉 만료. 왕복 비용 0.3%.
유동성(OBV·하이킨아시): 전날까지 5일 평균 거래대금 30억 이상인 신호만 (원본은 '5봉' 평균. 시봉에서 5봉=5시간이 되지 않게 일 단위로 고정)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "macd_rsi_swing"))
import core as ross  # noqa: E402

WARM_START = "2023-12-01"      # 일봉·시봉 모두 이 날 이후 신호만 (시봉 지표 워밍업)


# ───────────────────────── 종목별 배열
def stock_arrays(B, freq, info):
    """반환 dict code → dict(o,h,l,c,v,vt,day(Timestamp 배열),t(시각), liq10, liq30, uplim, tax, n)"""
    Dd = B["Dd"]
    C = Dd["c"]
    amt20 = Dd["amt"].rolling(20, min_periods=15).mean().shift(1)          # 전날까지
    amt5 = Dd["amt"].rolling(5, min_periods=5).mean().shift(1)
    prevc = C.ffill().shift(1)
    lim = B["lim"].reindex(C.index)
    kospi_tax, kosdaq_tax = ross.tax_schedule(C.index)
    out = {}
    if freq == "H":
        H = B["H"]
        groups = H.groupby("code", sort=False)
    for code in B["codes"]:
        if freq == "D":
            g = pd.DataFrame({"o": Dd["o"][code], "h": Dd["h"][code], "l": Dd["l"][code], "c": Dd["c"][code],
                              "v": Dd["v"][code]}).dropna(subset=["c", "o"])
            g["d"] = g.index
            g["t"] = g.index
        else:
            try:
                g = groups.get_group(code)
            except KeyError:
                continue
        if len(g) < 150:
            continue
        d = pd.DatetimeIndex(g["d"].values)
        di = C.index.get_indexer(d)
        ok = di >= 0
        g = g[ok]
        d, di = d[ok], di[ok]
        mk = info.get(code, "KOSDAQ")
        tx = (kospi_tax if mk == "KOSPI" else kosdaq_tax)[di]
        v = g["v"].to_numpy(dtype=float)
        dv = Dd["v"][code].to_numpy()[di] if code in Dd["v"] else np.ones(len(g))
        vt = np.where(np.nan_to_num(dv, nan=1.0) > 0, 1.0, 0.0)            # 체결 가능(그날 거래정지 아님)
        out[code] = dict(o=g["o"].to_numpy(float), h=g["h"].to_numpy(float), l=g["l"].to_numpy(float), c=g["c"].to_numpy(float),
                         v=np.nan_to_num(v), vt=vt, day=d.values, t=pd.DatetimeIndex(g["t"].values).values,
                         liq10=(amt20[code].to_numpy()[di] >= 1e9), liq30=(amt5[code].to_numpy()[di] >= 3e9),
                         uplim=prevc[code].to_numpy()[di] * (1 + lim.to_numpy()[di]), tax=tx, n=len(g))
    return out


def start_idx(x):
    return int(np.searchsorted(x["day"], np.datetime64(WARM_START)))


# ───────────────────────── 1. 로스 MACD·RSI
P_ROSS = dict(ross.P0)


def ross_trades(S, mode, rand=None, rng=None, k=20):
    P = P_ROSS
    fee = P["comm"] + P["slip"]
    rows, rrows = [], []
    for code, x in S.items():
        c, o, h, l = x["c"], x["o"], x["h"], x["l"]
        f, s_, sgn = P["macd"]
        macd = ross.ema(c, f) - ross.ema(c, s_)
        sg = ross.ema(macd, sgn)
        rsi = ross.rsi_w(c, P["rsi_n"])
        ok = x["liq10"].astype(np.bool_)
        st = max(start_idx(x), 60)
        if mode == 1:
            sigs = ross.m1_signals(rsi, macd, sg, ok, st, P["rsi_gate"], P["rsi_hot"], P["wait_low"], P["wait_hot"], P["hot_lo"], P["hot_hi"])
        else:
            sigs = ross.m2_signals(h, c, rsi, macd, sg, ok, st, P["sw"], P["gap_lo"], P["gap_hi"], P["pb"], P["pb_wait"])
        last = -1
        for t in sigs:
            t = int(t)
            e = t + 1
            if e >= x["n"] or e <= last or x["vt"][e] == 0:
                continue
            stop0 = l[max(0, t - P["stop_lb"] + 1):t + 1].min()
            rk = (c[t] - stop0) / c[t]
            if rk < P["min_risk"] or rk > P["max_risk"]:
                continue
            entry = o[e]
            if entry <= stop0 or entry >= x["uplim"][e] * (1 - 0.005):
                continue
            r = _ross_one(x, e, stop0, macd, sg, rsi, mode, fee)
            last = r[0]
            rows.append((code, x["t"][e], x["day"][e]) + r[1:])
            if rand is not None:
                rrows += _ross_rand(x, e, (entry - stop0) / entry, macd, sg, rsi, mode, fee, rng, k, st, code)
    cols = ["code", "t", "day", "entry", "stop", "fi", "ret", "R", "hold_bars", "hold_days"]
    return pd.DataFrame(rows, columns=cols), pd.DataFrame(rrows, columns=cols + ["draw"]) if rand is not None else None


def _ross_one(x, e, stop0, macd, sg, rsi, mode, fee):
    P = P_ROSS
    fi, fp, li, lp, fr, why = ross.sim_trade(e, x["o"], x["h"], x["l"], x["c"], x["vt"], macd, sg, rsi, stop0, mode, P["tgt"], P["rsi_gate"])
    entry = x["o"][e]
    if li >= 0:
        gross = fr * (lp / entry - 1) + (1 - fr) * (fp / entry - 1)
        cost = fee + fr * (fee + x["tax"][li]) + (1 - fr) * (fee + x["tax"][fi])
    else:
        gross = fp / entry - 1
        cost = 2 * fee + x["tax"][fi]
    net = gross - cost
    hd = int((x["day"][fi] - x["day"][e]) / np.timedelta64(1, "D"))
    return fi, entry, stop0, fi, net, net / ((entry - stop0) / entry), fi - e + 1, hd


def _rand_bars(x, e, st, rng, k, need):
    """같은 종목·같은 달 무작위 진입 봉(조건 need 배열이 참인 봉의 다음 봉)"""
    mon = x["day"].astype("datetime64[M]")
    idx = np.where((mon == mon[e]) & need)[0] + 1
    idx = idx[(idx >= st + 1) & (idx < x["n"] - 1)]
    if not len(idx):
        return []
    return rng.choice(idx, size=k, replace=True)


def _ross_rand(x, e, rk, macd, sg, rsi, mode, fee, rng, k, st, code):
    out = []
    for j, ee in enumerate(_rand_bars(x, e, st, rng, k, x["liq10"])):
        if x["vt"][ee] == 0:
            continue
        stop0 = x["o"][ee] * (1 - rk)
        r = _ross_one(x, ee, stop0, macd, sg, rsi, mode, fee)
        out.append((code, x["t"][ee], x["day"][ee]) + r[1:] + (j,))
    return out


# ───────────────────────── 2. OBV + 켈트너
OBV_VARIANTS = {
    "A2": dict(box="pct", thr=0.02, obv=True, kc=False, big=False, hold=False, reentry=False),
    "A5": dict(box="pct", thr=0.05, obv=True, kc=False, big=False, hold=False, reentry=False),
    "A8": dict(box="pct", thr=0.08, obv=True, kc=False, big=False, hold=False, reentry=False),
    "B": dict(box="pct", thr=0.05, obv=True, kc=True, big=True, hold=True, reentry=False),
    "C": dict(box="pct", thr=0.05, obv=False, kc=False, big=False, hold=False, reentry=False),
    "D": dict(box="atr", thr=2.5, obv=True, kc=False, big=False, hold=False, reentry=False),
    "E": dict(box="pct", thr=0.05, obv=True, kc=False, big=False, hold=False, reentry=True),
}
OBV_COST = 0.003


def obv_features(x):
    o, h, l, c, v = (pd.Series(x[k]) for k in ("o", "h", "l", "c", "v"))
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr14, atr10 = tr.rolling(14).mean(), tr.rolling(10).mean()
    hh, ll = h.shift(1).rolling(20).max(), l.shift(1).rolling(20).min()
    obv = (np.sign(c - o) * v).cumsum()
    f = dict(hh=hh.to_numpy(), ll=ll.to_numpy(), box_pct=((hh - ll) / pc).to_numpy(), box_atr=((hh - ll) / atr14.shift(1)).to_numpy(),
             obv_up=(obv > obv.ewm(span=20, adjust=False).mean()).to_numpy(), obv_hold=(obv.shift(1) >= obv.shift(20)).to_numpy(),
             kc_break=(c > c.ewm(span=20, adjust=False).mean() + 1.5 * atr10).to_numpy(),
             big=((c - o) >= 0.8 * atr14.shift(1)).to_numpy())
    brk = (c > hh)
    f["first"] = (brk & ~brk.shift(1, fill_value=False)).to_numpy()
    return f


def obv_mask(f, spec, x):
    m = f["first"] & (f["hh"] > f["ll"]) & x["liq30"]
    m = m & ((f["box_pct"] <= spec["thr"]) if spec["box"] == "pct" else (f["box_atr"] <= spec["thr"]))
    for key, col in (("obv", "obv_up"), ("kc", "kc_break"), ("big", "big"), ("hold", "obv_hold")):
        if spec[key]:
            m = m & f[col]
    return np.nan_to_num(m, nan=0).astype(bool)


@njit(cache=True)
def obv_sim(o, h, l, c, e, stop, max_hold):
    n = len(o)
    entry = o[e]
    R = entry - stop
    tg = np.array([entry + R, entry + 2 * R, entry + 3 * R])
    wt = np.array([0.5, 0.25, 0.25])
    last = min(n - 1, e + max_hold - 1)
    cur = stop
    nxt = 0
    rem = 1.0
    gross = 0.0
    for j in range(e, last + 1):
        if l[j] <= cur:
            px = o[j] if o[j] <= cur else cur
            gross += rem * (px / entry - 1)
            return j, gross, 1 if nxt == 0 else 0
        hit = False
        while nxt < 3 and h[j] >= tg[nxt]:
            px = o[j] if (j > e and o[j] >= tg[nxt]) else tg[nxt]
            w = rem if nxt == 2 else wt[nxt]
            gross += w * (px / entry - 1)
            rem -= w
            if nxt == 0:
                hit = True
            nxt += 1
        if hit:
            cur = entry
        if rem <= 1e-9:
            return j, gross, 0
    gross += rem * (c[last] / entry - 1)
    return last, gross, 0


def obv_trades(S, vname, rand=False, rng=None, k=20):
    spec = OBV_VARIANTS[vname]
    rows, rrows = [], []
    for code, x in S.items():
        f = obv_features(x)
        m = obv_mask(f, spec, x)
        st = max(start_idx(x), 60)
        m[:st] = False
        idxs = np.flatnonzero(m)
        busy = -1
        o, h, l, c = x["o"], x["h"], x["l"], x["c"]
        for i in idxs:
            if i + 1 <= busy or i + 1 >= x["n"]:
                continue
            e = i + 1
            stop = f["ll"][i]
            risk = (o[e] - stop) / o[e] if o[e] > 0 else np.nan
            if not (0.01 <= risk <= 0.25) or x["vt"][e] == 0:
                continue
            j, gross, stopped = obv_sim(o, h, l, c, e, stop, 60)
            busy = j
            rows.append(_obv_row(x, code, e, j, gross, stop))
            if spec["reentry"] and stopped:
                # 원본 E: 손절 뒤 10봉 안에 박스 안(손절가~박스 상단)으로 돌아왔다가 다시 상단 위로 마감 + OBV 상승이면 재진입
                box_hi = f["hh"][i]
                inside = False
                for q in range(j + 1, min(x["n"] - 2, j + 10) + 1):
                    if stop <= c[q] <= box_hi:
                        inside = True
                    elif inside and c[q] > box_hi and f["obv_up"][q]:
                        e2 = q + 1
                        r2 = (o[e2] - stop) / o[e2]
                        if 0.01 <= r2 <= 0.25 and x["vt"][e2] != 0:
                            j2, g2, _ = obv_sim(o, h, l, c, e2, stop, 60)
                            rows.append(_obv_row(x, code, e2, j2, g2, stop))
                            busy = j2
                        break
            if rand:
                for dj, ee in enumerate(_rand_bars(x, e, st, rng, k, x["liq30"])):
                    if x["vt"][ee] == 0:
                        continue
                    s2 = o[ee] * (1 - risk)
                    jj, gg, _ = obv_sim(o, h, l, c, ee, s2, 60)
                    rrows.append(_obv_row(x, code, ee, jj, gg, s2) + (dj,))
    cols = ["code", "t", "day", "entry", "stop", "fi", "ret", "R", "hold_bars", "hold_days"]
    return pd.DataFrame(rows, columns=cols), (pd.DataFrame(rrows, columns=cols + ["draw"]) if rand else None)


def _obv_row(x, code, e, j, gross, stop):
    entry = x["o"][e]
    net = gross - OBV_COST
    hd = int((x["day"][j] - x["day"][e]) / np.timedelta64(1, "D"))
    return (code, x["t"][e], x["day"][e], entry, stop, j, net, net / ((entry - stop) / entry), j - e + 1, hd)


# ───────────────────────── 3. 하이킨아시 도지 반전
HA_COST = 0.003


def ha_features(x):
    o, h, l, c = (pd.Series(x[k]) for k in ("o", "h", "l", "c"))
    hc = (o + h + l + c) / 4
    ho = np.empty(len(o))
    ho[0] = (o.iloc[0] + c.iloc[0]) / 2
    hcv = hc.to_numpy()
    for i in range(1, len(o)):
        ho[i] = (ho[i - 1] + hcv[i - 1]) / 2
    ho = pd.Series(ho)
    hh = pd.concat([h, ho, hc], axis=1).max(axis=1)
    hl = pd.concat([l, ho, hc], axis=1).min(axis=1)
    # 스무스 HA (len 10, len2 10)
    eo, ec, eh, el = (s.ewm(span=10, adjust=False).mean() for s in (o, c, h, l))
    hc2 = (eo + eh + el + ec) / 4
    ho2 = np.empty(len(o))
    ho2[0] = (eo.iloc[0] + ec.iloc[0]) / 2
    hc2v = hc2.to_numpy()
    for i in range(1, len(o)):
        ho2[i] = (ho2[i - 1] + hc2v[i - 1]) / 2
    ho2 = pd.Series(ho2)
    o2, c2 = ho2.ewm(span=10, adjust=False).mean(), hc2.ewm(span=10, adjust=False).mean()
    bull2, bear2 = (c2 > o2), (o2 > c2)
    flip = (bull2.shift(1, fill_value=False) & bear2).to_numpy()
    ema20 = c.ewm(span=20, adjust=False).mean().to_numpy()
    return dict(o1=ho.to_numpy(), h1=hh.to_numpy(), l1=hl.to_numpy(), c1=hc.to_numpy(), flip=flip, ema20=ema20)


def ha_signals(x, f, ema_mode, st):
    o1, h1, l1, c1 = f["o1"], f["h1"], f["l1"], f["c1"]
    rng_ = h1 - l1
    body = np.abs(c1 - o1)
    lw = np.minimum(o1, c1) - l1
    uw = h1 - np.maximum(o1, c1)
    valid = rng_ > 0
    doji = valid & (body <= 0.10 * rng_)
    flat = valid & (lw <= 0.10 * rng_)
    longup = valid & (uw >= 0.30 * rng_)
    bull, bear = c1 > o1, c1 < o1
    c, l, v, ema = x["c"], x["l"], x["v"], f["ema20"]
    n = len(c)
    run = np.zeros(n, dtype=int)
    for i in range(n):
        if doji[i]:
            run[i] = run[i - 1] + 1 if i > 0 else 1
    out = []
    for i in range(max(1, st), n):
        if not (bull[i] and flat[i] and longup[i]):
            continue
        r = run[i - 1]
        if r == 0:
            continue
        s = i - r
        if not v[i] > v[i - 1]:
            continue
        w0 = s - 12
        if w0 < 0 or bear[w0:s].mean() < 0.65:
            continue
        if ema_mode == "S" and not (c[i - 1] < ema[i - 1] and c[i] > ema[i]):
            continue
        if ema_mode == "R" and not (c[i] > ema[i]):
            continue
        stop = l[s:i].min() * (1 - 0.005)
        out.append((i, stop))
    return out


@njit(cache=True)
def ha_sim(o, h, l, c, flip, e, stop, max_hold):
    n = len(o)
    entry = o[e]
    target = entry + 2.0 * (entry - stop)
    last = min(n - 1, e + max_hold - 1)
    for k in range(e, last + 1):
        if k > e and flip[k - 1]:                  # 전환 봉 다음 봉 시가
            if k - 1 >= e:
                return k, o[k]
        if k > e and o[k] <= stop:
            return k, o[k]
        if k > e and o[k] >= target:
            return k, o[k]
        if l[k] <= stop:
            return k, stop
        if h[k] >= target:
            return k, target
    return last, c[last]


def ha_trades(S, ema_mode, rand=False, rng=None, k=20):
    rows, rrows = [], []
    for code, x in S.items():
        f = ha_features(x)
        st = max(start_idx(x), 60)
        busy = -1
        o, h, l, c = x["o"], x["h"], x["l"], x["c"]
        for i, stop in ha_signals(x, f, ema_mode, st):
            e = i + 1
            if e >= x["n"] or e <= busy or not x["liq30"][i] or x["vt"][e] == 0:
                continue
            if not (stop < o[e]):
                continue
            j, px = ha_sim(o, h, l, c, f["flip"], e, stop, 60)
            busy = j
            rows.append(_ha_row(x, code, e, j, px, stop))
            if rand:
                rk = (o[e] - stop) / o[e]
                for dj, ee in enumerate(_rand_bars(x, e, st, rng, k, x["liq30"])):
                    if x["vt"][ee] == 0:
                        continue
                    s2 = o[ee] * (1 - rk)
                    jj, pp = ha_sim(o, h, l, c, f["flip"], ee, s2, 60)
                    rrows.append(_ha_row(x, code, ee, jj, pp, s2) + (dj,))
    cols = ["code", "t", "day", "entry", "stop", "fi", "ret", "R", "hold_bars", "hold_days"]
    return pd.DataFrame(rows, columns=cols), (pd.DataFrame(rrows, columns=cols + ["draw"]) if rand else None)


def _ha_row(x, code, e, j, px, stop):
    entry = x["o"][e]
    net = px / entry - 1 - HA_COST
    hd = int((x["day"][j] - x["day"][e]) / np.timedelta64(1, "D"))
    return (code, x["t"][e], x["day"][e], entry, stop, j, net, net / ((entry - stop) / entry), j - e + 1, hd)
