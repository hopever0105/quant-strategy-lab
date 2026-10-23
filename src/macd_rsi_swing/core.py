"""MACD·RSI 스윙(매매법 1: MACD 골든크로스, 매매법 2: RSI 하락 다이버전스 + MACD 재골든크로스) 백테스트 핵심.
- 데이터: quant/cache/kor_panel.pkl (stitch_history.py 가 DB에서 만든 상장폐지 포함 수정주가 패널). 실매매 코드 없음.
- 신호는 종가 기준, 체결은 다음 거래일 시가. 같은 입력 → 같은 결과(난수는 seed 고정).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

BASE = Path(os.environ.get("QUANT_HOME", Path(__file__).resolve().parents[2]))   # .env·cache·logs 위치(기본 quant/)
CACHE = BASE / "cache" / "kor_panel.pkl"
STATS = {}

P0 = dict(  # 기본 파라미터 (이해 확인 단계에서 합의한 값)
    macd=(12, 26, 9), rsi_n=14, rsi_gate=50.0, rsi_hot=70.0,
    wait_low=20, wait_hot=40, hot_lo=50.0, hot_hi=56.0,
    stop_lb=10, min_risk=0.03, max_risk=0.15, tgt=2.0,
    sw=5, gap_lo=5, gap_hi=60, pb=0.85, pb_wait=40,
    liq_amt=1e9, liq_mcap=5e10,               # 20일 평균 거래대금 10억원 / 상장폐지 종가-only 구간은 시총 500억 대용
    risk=0.005, maxw=0.10, maxpos=20, comm=0.00015, slip=0.001, cost_mult=1.0)


# ───────────────────────── 지표 (numba)
@njit(cache=True)
def ema(x, n):
    a = 2.0 / (n + 1.0)
    out = np.empty(len(x)); out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


@njit(cache=True)
def rsi_w(c, n):
    m = len(c); out = np.full(m, np.nan)
    if m <= n:
        return out
    g = 0.0; l = 0.0
    for i in range(1, n + 1):
        d = c[i] - c[i - 1]
        if d > 0: g += d
        else: l -= d
    g /= n; l /= n
    out[n] = 100.0 if l == 0 else 100.0 - 100.0 / (1.0 + g / l)
    for i in range(n + 1, m):
        d = c[i] - c[i - 1]
        gg = d if d > 0 else 0.0
        ll = -d if d < 0 else 0.0
        g = (g * (n - 1) + gg) / n; l = (l * (n - 1) + ll) / n
        out[i] = 100.0 if l == 0 else 100.0 - 100.0 / (1.0 + g / l)
    return out


# ───────────────────────── 신호 (numba)
@njit(cache=True)
def m1_signals(rsi, macd, sg, ok, start, rsi_gate, rsi_hot, wait_low, wait_hot, hot_lo, hot_hi):
    n = len(rsi); out = np.empty(n, np.int64); k = 0
    state = 0; age = 0
    for t in range(start, n):
        if np.isnan(rsi[t]) or np.isnan(rsi[t - 1]):
            continue
        gc = macd[t - 1] <= sg[t - 1] and macd[t] > sg[t]
        if state > 0:
            age += 1
            if macd[t] <= sg[t] or (state == 1 and age > wait_low) or (state == 2 and age > wait_hot):
                state = 0
        if gc:
            if rsi[t] >= rsi_gate and rsi[t] < rsi_hot:
                if ok[t]: out[k] = t; k += 1
                state = 0
            elif rsi[t] < rsi_gate:
                state = 1; age = 0
            else:
                state = 2; age = 0
        elif state == 1:
            if rsi[t - 1] < rsi_gate and rsi[t] >= rsi_gate:
                if ok[t]: out[k] = t; k += 1
                state = 0
        elif state == 2:
            if rsi[t] >= hot_lo and rsi[t] <= hot_hi and rsi[t] > rsi[t - 1]:
                if ok[t]: out[k] = t; k += 1
                state = 0
    return out[:k]


@njit(cache=True)
def m2_signals(h, c, rsi, macd, sg, ok, start, sw, gap_lo, gap_hi, pb, pb_wait):
    n = len(c); out = np.empty(n, np.int64); k = 0
    prev_i = -1; armed = False; H2 = 0.0; age = 0; dead_seen = False
    for t in range(start, n):
        i = t - sw
        if i - sw >= 0 and not np.isnan(rsi[i]):
            mx_l = h[i - sw]
            for j in range(i - sw, i):
                if h[j] > mx_l: mx_l = h[j]
            mx_r = h[i + 1]
            for j in range(i + 1, i + sw + 1):
                if h[j] > mx_r: mx_r = h[j]
            if h[i] > mx_l and h[i] >= mx_r:                      # t 시점에 확정된 스윙 고점 i (미래정보 없음: 우측 sw일 확인 후)
                if prev_i >= 0 and gap_lo <= i - prev_i <= gap_hi and h[i] > h[prev_i] and rsi[i] < rsi[prev_i]:
                    armed = True; H2 = h[i]; age = 0; dead_seen = macd[t] < sg[t]
                prev_i = i
        if armed:
            age += 1
            if c[t] < pb * H2 or age > pb_wait:
                armed = False
            else:
                if macd[t] < sg[t]:
                    dead_seen = True
                elif dead_seen and macd[t - 1] <= sg[t - 1] and macd[t] > sg[t]:
                    if ok[t]: out[k] = t; k += 1
                    armed = False
    return out[:k]


# ───────────────────────── 한 건 시뮬레이션 (numba)
@njit(cache=True)
def sim_trade(e, o, h, l, c, v, macd, sg, rsi, stop0, mode, tgt, rsi_gate):
    """mode 1: 2R 절반 익절 후 잔량(손절=본전, MACD 데드크로스/RSI<=50 → 다음날 시가).
       mode 2: 2R 전량 익절, 도달 전 MACD 데드크로스 → 다음날 시가 전량.  손절·목표 같은 날 동시 도달 시 손절 우선.
       거래정지(거래량 0)인 날은 체결 불가. 반환 (최종일, 최종가, 1차일, 1차가, 1차비중, 사유)."""
    n = len(c)
    entry = o[e]; R = entry - stop0
    target = entry + tgt * R
    stop = stop0
    hit1 = False; leg_i = -1; leg_p = np.nan
    frac1 = 0.5 if mode == 1 else 1.0
    pend = -1
    for k in range(e, n):
        if k > e and v[k] == 0:
            continue
        if pend >= 0 and k >= pend and k > e:
            return k, o[k], leg_i, leg_p, frac1, 3
        if l[k] <= stop:
            px = stop if k == e else min(o[k], stop)
            return k, px, leg_i, leg_p, frac1, 1
        if (not hit1) and h[k] >= target:
            px = target if k == e else max(o[k], target)
            if mode == 2:
                return k, px, -1, np.nan, 1.0, 2
            hit1 = True; leg_i = k; leg_p = px; stop = entry
        if k + 1 < n:
            dc = macd[k] < sg[k] and macd[k - 1] >= sg[k - 1]
            if mode == 2:
                if dc: pend = k + 1
            elif hit1:
                if dc or rsi[k] <= rsi_gate: pend = k + 1
    return n - 1, c[n - 1], leg_i, leg_p, frac1, 4


# ───────────────────────── 데이터
def tax_schedule(dates, us=False):
    """매도 시 거래세+농특세 (소수). 근거: 나무위키 '증권거래세'(2026-07-31 판) 표 + 각주. 1999~2018 구간은 각주 기반 추정(확인 필요).
    KOSPI 0.30%(1996~2019-05-29) → 0.25%(2019-05-30) → 0.23%(2021) → 0.20%(2023) → 0.18%(2024) → 0.15%(2025) → 0.20%(2026)
    KOSDAQ 0.50%(~2013-06-30) → 0.30%(2013-07-01~2019-05-29) → 이후 KOSPI와 동일."""
    d = pd.DatetimeIndex(dates)
    if us:
        z = np.zeros(len(d)); return z, z
    kospi = np.full(len(d), 0.0030)
    for start, r in [("2019-05-30", .0025), ("2021-01-01", .0023), ("2023-01-01", .0020), ("2024-01-01", .0018),
                     ("2025-01-01", .0015), ("2026-01-01", .0020)]:
        kospi[d >= pd.Timestamp(start)] = r
    kosdaq = kospi.copy()
    kosdaq[d < pd.Timestamp("2013-07-01")] = 0.0050
    return kospi, kosdaq


def load_panel():
    pn = pd.read_pickle(CACHE)
    pn = pn[(pn.c > 0) & (pn.h >= pn.l) & pn.o.notna()].drop_duplicates(["code", "d"])
    return pn


def prepare(pn, P=P0):
    """종목별 numpy 묶음. 반환: dates, tickers(dict)"""
    dates = np.sort(pn.d.unique())
    gidx = np.searchsorted(dates, pn.d.values)
    pn = pn.assign(gd=gidx)
    out = {}
    # 구간 분리: (1) 전일 대비 달력 15일 초과 결측, (2) 가격제한폭(2015-06-15 전 ±15%, 후 ±30%)을 넘는 일간 변동 = 보정 안 된 가격 단절
    pn = pn.sort_values(["code", "d"]).reset_index(drop=True)
    cc = pn.code.values; dd = pn.d.values; cl = pn.c.values
    same = np.r_[False, cc[1:] == cc[:-1]]
    gapd = np.r_[0, (dd[1:] - dd[:-1]).astype("timedelta64[D]").astype(int)]
    ret = np.r_[0.0, cl[1:] / cl[:-1] - 1.0]
    limv = np.where(dd < np.datetime64("2015-06-15"), 0.15, 0.30)
    if P.get("us"):
        limv = np.full(len(dd), 0.45)            # 미국은 가격제한 없음: 수정주가가 이어지지 않는 50% 초과 변동만 단절로 본다
    brk_gap = same & (gapd > 15)
    brk_px = same & (np.abs(ret) > limv * 1.1 + 0.01)
    newseg = (~same) | brk_gap | brk_px
    seg = np.cumsum(newseg)
    pn = pn.assign(seg=seg)
    STATS["breaks_gap"] = int(brk_gap.sum()); STATS["breaks_px"] = int(brk_px.sum())
    for sid, g in pn.groupby("seg", sort=False):
        code = g.code.iloc[0] + ("" if (g.code.iloc[0], 0) is None else "")
        if len(g) < 120:
            continue
        c = g.c.values.astype(np.float64)
        o = g.o.values.astype(np.float64); h = g.h.values.astype(np.float64); l = g.l.values.astype(np.float64)
        v = g.v.values.astype(np.float64); amt = g.amt.values.astype(np.float64)
        v = np.where(np.isnan(v), 1.0, v)                       # 거래량 미상(상장폐지 종가-only)은 체결 가능으로 취급
        src = g.src.values
        mc = g["mcap"].values.astype(np.float64) if "mcap" in g.columns else np.full(len(g), np.nan)
        a20 = pd.Series(amt).rolling(20, min_periods=15).mean().values
        liq = np.where(src == "D", (mc >= P["liq_mcap"]), (a20 >= P["liq_amt"]))
        if P.get("us"):
            liq = a20 >= P["liq_amt"]
        f, s, sg_n = P["macd"]
        macd = ema(c, f) - ema(c, s); sg = ema(macd, sg_n)
        r = rsi_w(c, P["rsi_n"])
        gd = g.gd.values
        d_ts = dates[gd]
        lim = np.where(d_ts < np.datetime64("2015-06-15"), 0.15, 0.30)
        if P.get("us"):
            lim = np.full(len(lim), 9.0)
        key = g.code.iloc[0] if g.code.iloc[0] not in out else f"{g.code.iloc[0]}#{sid}"
        out[key] = dict(code=g.code.iloc[0], o=o, h=h, l=l, c=c, v=v, macd=macd, sg=sg, rsi=r, liq=liq.astype(np.bool_), gd=gd, lim=lim,
                         src=src, a20=a20, n=len(c))
    return dates, out


def lowmin(l, t, lb):
    return l[max(0, t - lb + 1):t + 1].min()


def gen_trades(tk, code, mode, P, rng=None, rand_pool=None):
    """한 종목의 거래 목록 (종목별 동시 보유 1개). rand_pool: 무작위 진입용 {연도: 후보 t 배열}"""
    x = tk
    start = 60
    ok = x["liq"]
    if rand_pool is None:
        if mode == 1:
            sigs = m1_signals(x["rsi"], x["macd"], x["sg"], ok, start, P["rsi_gate"], P["rsi_hot"], P["wait_low"], P["wait_hot"], P["hot_lo"], P["hot_hi"])
        else:
            sigs = m2_signals(x["h"], x["c"], x["rsi"], x["macd"], x["sg"], ok, start, P["sw"], P["gap_lo"], P["gap_hi"], P["pb"], P["pb_wait"])
    else:
        sigs = rand_pool
    rows = []; last_exit = -1
    n = x["n"]
    for t in sigs:
        t = int(t); e = t + 1
        if e >= n or e <= last_exit or x["v"][e] == 0:
            continue
        stop0 = lowmin(x["l"], t, P["stop_lb"]); rk = (x["c"][t] - stop0) / x["c"][t]
        if rk < P["min_risk"] or rk > P["max_risk"]:
            continue
        entry = x["o"][e]
        if entry <= stop0 or entry >= x["c"][t] * (1 + x["lim"][e] - 0.005):      # 시가 갭하락(손절가 이하) / 상한가 근접 시가 → 매수 불가
            continue
        fi, fp, li, lp, fr, why = sim_trade(e, x["o"], x["h"], x["l"], x["c"], x["v"], x["macd"], x["sg"], x["rsi"], stop0, mode, P["tgt"], P["rsi_gate"])
        last_exit = fi
        rows.append((code, t, e, entry, stop0, fi, fp, li, lp, fr, why, x["src"][e], x["a20"][t] if not np.isnan(x["a20"][t]) else 0.0))
    return rows


COLS = ["code", "sig", "e", "entry", "stop", "fi", "fpx", "li", "lpx", "frac1", "why", "src", "liq"]


def all_trades(dates, tks, mode, P, rand=None):
    rows = []
    for code, x in tks.items():
        rows += gen_trades(x, code, mode, P, rand_pool=None if rand is None else rand.get(code))
    df = pd.DataFrame(rows, columns=COLS)
    return df


def add_costs(df, tks, dates, mkt, P):
    """거래비용·수익률 계산 (R배수 포함). mkt: code->'KOSPI'/'KOSDAQ'/None(알 수 없으면 코스닥 세율=보수적)"""
    tk_k, tk_q = tax_schedule(dates, P.get("us", False))
    m = P["cost_mult"]
    fee = (P["comm"] + P["slip"]) * m
    ret = np.empty(len(df)); rmult = np.empty(len(df)); hold = np.empty(len(df)); ed = np.empty(len(df), np.int64); xd = np.empty(len(df), np.int64)
    for i, r in enumerate(df.itertuples(index=False)):
        x = tks[r.code]; gd = x["gd"]
        tx = tk_k if mkt.get(x["code"]) == "KOSPI" else tk_q
        g_final = (r.fpx / r.entry - 1.0)
        sell_final = fee + tx[gd[r.fi]] * m
        if r.li >= 0:
            g_leg = r.lpx / r.entry - 1.0; sell_leg = fee + tx[gd[r.li]] * m
            gross = r.frac1 * g_leg + (1 - r.frac1) * g_final
            cost = fee + r.frac1 * sell_leg + (1 - r.frac1) * sell_final
        else:
            gross = g_final; cost = fee + sell_final
        net = gross - cost
        ret[i] = net; rmult[i] = net / ((r.entry - r.stop) / r.entry)
        hold[i] = r.fi - r.e + 1; ed[i] = gd[r.e]; xd[i] = gd[r.fi]
    df = df.assign(ret=ret, rmult=rmult, hold=hold, ed=ed, xd=xd)
    return df


def trade_path(r, x, tx_k, m, fee):
    """한 거래의 일별 누적수익(투입금 대비, 종가 평가, 비용 반영). 반환 (global day 배열, 값 배열)"""
    gd = x["gd"]; ks = np.arange(r.e, r.fi + 1)
    ent = r.entry
    cum = np.empty(len(ks))
    legged = r.li >= 0
    for j, k in enumerate(ks):
        if k == r.fi:
            fin = r.fpx / ent - 1.0 - (fee + tx_k[gd[k]] * m)
            if legged and k > r.li:
                cum[j] = r.frac1 * (r.lpx / ent - 1.0 - (fee + tx_k[gd[r.li]] * m)) + (1 - r.frac1) * fin - fee
            elif legged and k == r.li:
                cum[j] = r.frac1 * (r.lpx / ent - 1.0) + (1 - r.frac1) * fin - fee - r.frac1 * tx_k[gd[r.li]] * m
            else:
                cum[j] = fin - fee
        elif legged and k >= r.li:
            cum[j] = (r.frac1 * (r.lpx / ent - 1.0 - (fee + tx_k[gd[r.li]] * m)) + (1 - r.frac1) * (x["c"][k] / ent - 1.0) - fee)
        else:
            cum[j] = x["c"][k] / ent - 1.0 - fee
    return gd[ks], cum


def portfolio(df, tks, dates, mkt, P, eq0=1.0):
    """거래 목록 → 포트폴리오 일별 자산곡선. 거래당 위험 P['risk'], 종목당 최대 P['maxw'], 동시 최대 P['maxpos'], 레버리지 없음."""
    nd = len(dates)
    tk_k, tk_q = tax_schedule(dates, P.get("us", False))
    m = P["cost_mult"]; fee = (P["comm"] + P["slip"]) * m
    contrib = np.zeros(nd + 1); delta = np.zeros(nd + 2); expo = np.zeros(nd + 1)
    df = df.sort_values(["ed", "liq"], ascending=[True, False])
    open_ = []          # (exit_day, A)
    real = 0.0; ptr = -1
    acc = []
    for r in df.itertuples(index=False):
        e = r.ed
        while ptr < e:
            ptr += 1; real += delta[ptr]
        open_ = [(xd_, A_) for xd_, A_ in open_ if xd_ >= e]
        if len(open_) >= P["maxpos"]:
            continue
        eq = eq0 + contrib[e] + real
        rk = (r.entry - r.stop) / r.entry
        A = min(P["risk"] * eq / rk, P["maxw"] * eq)
        if sum(a for _, a in open_) + A > eq:
            continue
        x = tks[r.code]; tx = tk_k if mkt.get(x["code"]) == "KOSPI" else tk_q
        days, cum = trade_path(r, x, tx, m, fee)
        # 같은 global day 중복 방지(종목 행이 하루 1개라 중복 없음). 구간 사이 빈 날은 직전 값 유지
        full = np.arange(days[0], days[-1] + 1)
        idx = np.searchsorted(days, full, side="right") - 1
        contrib[full] += A * cum[idx]
        delta[days[-1] + 1] += A * cum[-1]
        expo[full] += A
        open_.append((int(days[-1]), A)); acc.append((r.code, e, A))
    eqs = eq0 + contrib[:nd] + np.cumsum(delta[:nd])
    expo = expo[:nd] / eqs
    return eqs, expo, acc


