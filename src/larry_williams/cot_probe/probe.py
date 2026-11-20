# -*- coding: utf-8 -*-
"""
래리 윌리엄스 COT 상업자 지수 - 시험 검증 (probe)
- COT: CFTC 공식 API (publicreporting.cftc.gov, Legacy Futures Only, 6dca-aqww). KRX 사용 없음.
- 가격: FinanceDataReader (야후 연속선물 CL=F, GC=F, ES=F, ZB=F)
- 상업자 지수 = (상업자 순포지션 - N주 최저) / (N주 최고 - N주 최저) * 100   (Williams, N=26주 기본)
- 미래정보 방지: COT 기준일(화요일) 데이터는 금요일 발표 -> 다음 주 월요일(+6일) 시가부터 사용, 종가 기준으로 근사
"""
import os, json, time, urllib.request, urllib.parse
import numpy as np
import pandas as pd
import FinanceDataReader as fdr

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.abspath(os.path.join(BASE, '..', '..', 'cache', 'cftc'))
OUT = os.path.join(BASE, 'results')
os.makedirs(CACHE, exist_ok=True); os.makedirs(OUT, exist_ok=True)

# CFTC 계약 코드 : (이름, 야후 가격 심볼)
MARKETS = {
    '067651': ('원유(WTI)', 'CL=F'),
    '088691': ('금', 'GC=F'),
    '13874A': ('S&P500 E-mini', 'ES=F'),
    '020601': ('미국 30년 국채', 'ZB=F'),
}
LOOKBACKS = [26, 156]      # 26주(6개월), 156주(3년)
HI, LO = 80, 20            # 극단 기준
HORIZONS = [4, 8, 13]      # 주 단위 이후 수익률


def fetch_cot(code):
    path = os.path.join(CACHE, f'cot_{code}.csv')
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < 86400:
        return pd.read_csv(path, parse_dates=['date'])
    q = {
        '$select': 'report_date_as_yyyy_mm_dd,comm_positions_long_all,comm_positions_short_all,open_interest_all',
        '$where': f"cftc_contract_market_code='{code}'",
        '$order': 'report_date_as_yyyy_mm_dd',
        '$limit': '50000',
    }
    url = 'https://publicreporting.cftc.gov/resource/6dca-aqww.json?' + urllib.parse.urlencode(q)
    with urllib.request.urlopen(url, timeout=60) as r:
        rows = json.loads(r.read().decode('utf-8'))
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['report_date_as_yyyy_mm_dd']).dt.normalize()
    for c in ['comm_positions_long_all', 'comm_positions_short_all', 'open_interest_all']:
        df[c] = pd.to_numeric(df[c])
    df = df.groupby('date', as_index=False).sum(numeric_only=True)   # 같은 날 중복 행 합산
    df['comm_net'] = df['comm_positions_long_all'] - df['comm_positions_short_all']
    df = df[['date', 'comm_net', 'open_interest_all']]
    df.to_csv(path, index=False, encoding='utf-8-sig')
    return df


def fetch_price(sym):
    path = os.path.join(CACHE, f'px_{sym.replace("=", "_")}.csv')
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < 86400:
        return pd.read_csv(path, index_col=0, parse_dates=True)['Close']
    px = fdr.DataReader(sym, '1998-01-01')
    px[['Close']].to_csv(path, encoding='utf-8-sig')
    return px['Close']


def analyze(code, name, sym):
    cot = fetch_cot(code).set_index('date').sort_index()
    px = fetch_price(sym).dropna()
    px = px[~px.index.duplicated()].sort_index()
    rows = []
    for N in LOOKBACKS:
        lo = cot['comm_net'].rolling(N).min(); hi = cot['comm_net'].rolling(N).max()
        idx = (cot['comm_net'] - lo) / (hi - lo) * 100
        sig = pd.DataFrame({'idx': idx}).dropna()
        sig['avail'] = sig.index + pd.Timedelta(days=6)              # 다음 주 월요일
        # 사용 가능일 이후 첫 거래일 종가를 진입가로
        pos = px.index.searchsorted(sig['avail'])
        ok = pos < len(px)
        sig = sig[ok]; pos = pos[ok]
        sig['p0'] = px.values[pos]
        for h in HORIZONS:
            p_h = px.index.searchsorted(sig['avail'] + pd.Timedelta(weeks=h))
            v = np.where(p_h < len(px), px.values[np.minimum(p_h, len(px) - 1)], np.nan)
            sig[f'r{h}'] = v / sig['p0'] - 1
        sig = sig[sig.index >= px.index[0]]
        for grp, mask in [('전체', sig['idx'].notna()), (f'지수≥{HI} (상업자 극단 매수)', sig['idx'] >= HI),
                          (f'지수≤{LO} (상업자 극단 매도)', sig['idx'] <= LO)]:
            s = sig[mask]
            row = {'시장': name, 'N주': N, '구분': grp, '주수': len(s)}
            for h in HORIZONS:
                r = s[f'r{h}'].dropna()
                row[f'{h}주 평균%'] = round(r.mean() * 100, 2)
                row[f'{h}주 상승비율%'] = round((r > 0).mean() * 100, 1)
            # 13주 기준 '극단 vs 나머지' 차이의 t값 (주간 겹침 있음 -> 과대평가 주의)
            if grp != '전체':
                a = s['r13'].dropna(); b = sig.loc[~mask, 'r13'].dropna()
                se = np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 1 else np.nan
                row['13주 차이 t(겹침 미보정)'] = round((a.mean() - b.mean()) / se, 2) if se else np.nan
            rows.append(row)
        # 간단 전략: 지수≥HI면 롱, ≤LO면 숏, 그 사이는 직전 포지션 유지 (주간, 비용 미반영)
        w = px.resample('W-FRI').last()
        sw = sig['idx'].copy(); sw.index = sig['avail']
        sw = sw.resample('W-FRI').last().reindex(w.index).ffill()
        pos_ = pd.Series(np.where(sw >= HI, 1, np.where(sw <= LO, -1, np.nan)), index=w.index).ffill().fillna(0)
        ret = w.pct_change()
        strat = (pos_.shift(1) * ret).dropna()
        bh = ret.loc[strat.index]
        def stat(r):
            cagr = (1 + r).prod() ** (52 / len(r)) - 1
            vol = r.std() * np.sqrt(52)
            eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
            return cagr, vol, cagr / vol if vol else np.nan, mdd
        c1, v1, s1, m1 = stat(strat); c2, v2, s2, m2 = stat(bh)
        rows.append({'시장': name, 'N주': N, '구분': '전략(롱/숏 유지) vs 보유',
                     '주수': len(strat),
                     '4주 평균%': f'CAGR {c1*100:.1f} / {c2*100:.1f}',
                     '4주 상승비율%': f'샤프 {s1:.2f} / {s2:.2f}',
                     '8주 평균%': f'MDD {m1*100:.0f} / {m2*100:.0f}',
                     '8주 상승비율%': f'{strat.index[0].date()}~{strat.index[-1].date()}'})
    return rows


if __name__ == '__main__':
    allrows = []
    for code, (name, sym) in MARKETS.items():
        try:
            allrows += analyze(code, name, sym)
            print('완료', name)
        except Exception as e:
            print('실패', name, e)
    res = pd.DataFrame(allrows)
    res.to_csv(os.path.join(OUT, 'cot_probe.csv'), index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
    print(res.to_string(index=False))
