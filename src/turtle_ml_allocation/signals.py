"""입력 변수와 매매 신호(1차 모델). 모든 값은 그날 종가까지의 정보만 쓴다.

- 터틀: 종가가 직전 n일 최고 종가를 넘으면 매수 결정 → 다음 날 종가에 진입.
        종가가 직전 n일 최저 종가 아래 또는 진입가 − 2N 아래면 청산 결정 → 다음 날 종가에 청산.
        N = 20일 평균 종가 변동폭(|ΔP|). 펀드 대용 구간에는 고가·저가가 없어 종가로 통일했다.
- 200일선: 월말 종가가 200일 이동평균 위면 보유(다음 날 진입), 아래면 청산.
- 12개월 모멘텀: 월말 기준 최근 252거래일 수익률이 0보다 크면 보유.
"""
import numpy as np
import pandas as pd

FEATS = ["dist200", "align", "rsi", "bbw", "vix", "t10y2y"]
FEAT_KO = {"dist200": "200일선 대비 위치", "align": "EMA 21·55·89 정배열", "rsi": "RSI(14)",
           "bbw": "볼린저 밴드폭", "vix": "VIX(전일)", "t10y2y": "장단기 금리차(전일)"}
HORIZON = 20

# 변수 확장(사후 점검): 기본 6개 + 아래. 결과(기본 6개 = 효과 없음)를 본 뒤 추가했으므로 본 결과와 구분해 보고한다.
FEATS_EXT = FEATS + ["baa", "baa_chg", "dgs10", "dgs10_chg", "avg_corr", "mom252", "vol_ratio", "brk_n",
                     "g_주식", "g_채권", "g_실물", "g_통화"]
FEAT_KO.update({"baa": "신용 스프레드", "baa_chg": "신용 스프레드 3개월 변화", "dgs10": "미국채 10년 금리",
                "dgs10_chg": "10년 금리 3개월 변화", "avg_corr": "자산 간 평균 상관(60일)", "mom252": "12개월 수익률",
                "vol_ratio": "변동성 국면(20일÷252일)", "brk_n": "돌파 강도(N 단위)",
                "g_주식": "자산군: 주식", "g_채권": "자산군: 채권", "g_실물": "자산군: 실물", "g_통화": "자산군: 통화"})


def avg_pairwise_corr(P, elig, win=60):
    """그날까지 60거래일, 투자 가능한 자산끼리 일간 수익률 상관의 평균(시장 전체가 한 방향으로 움직이는 정도)"""
    R = P.pct_change(fill_method=None).values
    el = elig.values
    out = np.full(len(P), np.nan)
    for t in range(win, len(P)):
        ok = el[t]
        if ok.sum() < 3:
            continue
        x = R[t - win + 1:t + 1][:, ok]
        x = x[:, np.isfinite(x).all(0)]
        if x.shape[1] < 3:
            continue
        c = np.corrcoef(x.T)
        n = c.shape[0]
        out[t] = (c.sum() - n) / (n * (n - 1))
    return pd.Series(out, index=P.index)


def add_ext(F, P, mac, groups):
    bc = lambda s: pd.DataFrame({c: s for c in P.columns})
    baa, dgs = mac["BAA10Y"].shift(1), mac["DGS10"].shift(1)
    F["baa"], F["baa_chg"] = bc(baa), bc(baa - baa.shift(63))
    F["dgs10"], F["dgs10_chg"] = bc(dgs), bc(dgs - dgs.shift(63))
    F["avg_corr"] = bc(avg_pairwise_corr(P, F["elig"]))
    lr = np.log(P).diff()
    F["vol_ratio"] = lr.rolling(20).std() / lr.rolling(252).std()
    hi = P.shift(1).rolling(20).max()
    F["brk_n"] = (P - hi) / F["N"]
    for g in ("주식", "채권", "실물", "통화"):
        F[f"g_{g}"] = pd.DataFrame({c: float(groups[c] == g) for c in P.columns}, index=P.index)
    return F


def features(P, mac):
    sma200 = P.rolling(200).mean()
    ema = lambda n: P.ewm(span=n, adjust=False, min_periods=n).mean()
    e21, e55, e89 = ema(21), ema(55), ema(89)
    d = P.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    F = {
        "dist200": P / sma200 - 1,
        "align": ((e21 > e55) & (e55 > e89)).astype(float).where(e89.notna()),
        "rsi": 100 - 100 / (1 + up / dn),
        "bbw": 4 * P.rolling(20).std() / P.rolling(20).mean(),
        "N": d.abs().rolling(20).mean(),
        "vol60": np.log(P).diff().rolling(60).std() * np.sqrt(252),
        "mom252": P / P.shift(252) - 1,
        "sma200": sma200,
    }
    F["vix"] = pd.DataFrame({c: mac["VIX"].shift(1) for c in P.columns})
    F["t10y2y"] = pd.DataFrame({c: mac["T10Y2Y"].shift(1) for c in P.columns})
    # 투자 가능: 가격이 252일 이상 쌓이고 변동성·200일선 계산이 되는 날부터
    F["elig"] = (P.notna().cumsum() >= 252) & P.notna() & F["vol60"].notna() & sma200.notna()
    return F


def month_end_idx(index):
    s = pd.Series(np.arange(len(index)), index=index)
    return s.groupby([index.year, index.month]).max().values


def _rec(rows, a, d, e, x, reason):
    rows.append((a, d, e, x, reason))


def turtle_trades(P, F, start, n_in=20, n_out=20, stop=2.0):
    hi = P.shift(1).rolling(n_in).max().values
    lo = P.shift(1).rolling(n_out).min().values
    p, N, el = P.values, F["N"].values, F["elig"].values
    T = len(P)
    rows = []
    for a in range(P.shape[1]):
        inpos, d0, e, stp = False, 0, 0, 0.0
        for t in range(start, T - 1):
            if not inpos:
                if el[t, a] and p[t, a] > hi[t, a]:
                    inpos, d0, e = True, t, t + 1
                    stp = p[e, a] - stop * N[t, a]
            elif t >= e:
                if p[t, a] < lo[t, a]:
                    _rec(rows, a, d0, e, t + 1, "채널 이탈"); inpos = False
                elif p[t, a] < stp:
                    _rec(rows, a, d0, e, t + 1, "2N 손절"); inpos = False
        if inpos:
            _rec(rows, a, d0, e, T - 1, "기간 끝")
    return rows


def monthly_trades(P, F, start, kind="ma200"):
    me = month_end_idx(P.index)
    me = me[me >= start]
    p, el = P.values, F["elig"].values
    cond = (P > F["sma200"]).values if kind == "ma200" else (F["mom252"] > 0).values
    T = len(P)
    rows = []
    for a in range(P.shape[1]):
        inpos, d0, e = False, 0, 0
        for m in me:
            if m >= T - 1:
                break
            on = bool(el[m, a] and cond[m, a])
            if on and not inpos:
                inpos, d0, e = True, m, m + 1
            elif inpos and not on:
                _rec(rows, a, d0, e, m + 1, "조건 해제"); inpos = False
        if inpos:
            _rec(rows, a, d0, e, T - 1, "기간 끝")
    return rows


def to_frame(rows, P, F, groups):
    cols = list(P.columns)
    df = pd.DataFrame(rows, columns=["ai", "d", "e", "x", "reason"])
    idx = P.index
    df["asset"] = [cols[i] for i in df.ai]
    df["group"] = [groups[c] for c in df.asset]
    df["d_date"], df["e_date"], df["x_date"] = idx[df.d], idx[df.e], idx[df.x]
    pv = P.values
    df["ret"] = pv[df.x, df.ai] / pv[df.e, df.ai] - 1
    df["hold_days"] = df.x - df.e
    T = len(P)
    ok = df.e + HORIZON < T
    le = np.where(ok, df.e + HORIZON, T - 1)
    df["fwd20"] = np.where(ok, pv[le, df.ai] / pv[df.e, df.ai] - 1, np.nan)
    df["label"] = np.where(ok, (df["fwd20"] > 0).astype(float), np.nan)
    df["label_date"] = np.where(ok, idx[le], pd.NaT)
    df["label_date"] = pd.to_datetime(df["label_date"])
    for f in FEATS_EXT:
        if f in F:
            df[f] = F[f].values[df.d, df.ai]
    return df.sort_values(["d", "ai"]).reset_index(drop=True)
