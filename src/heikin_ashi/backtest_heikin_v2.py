# -*- coding: utf-8 -*-
"""
하이킨아시 스캐너 v2 백테스트 (heikin_scanner_v2.py 와 같은 폴더(BASE_DIR)에 두고 실행)
=================================================================
heikin_scanner_v2.py 의 신호 탐지 함수(find_signals)와 설정값을 그대로 가져다 쓴다.

[규칙]
  진입: 필수 4조건 (도지 후 평평한바닥+긴윗꼬리 양봉 / 거래량 / 직전 하락추세 / EMA20 돌파) -> 다음날 시가 진입 가정
  청산: 손절가, 익절가, 스무스 하이킨아시(히말라야) 양봉->음봉 전환 (전환 다음날 시가 매도), 최대보유일 만료 중 먼저 오는 것

[3가지 EMA 기준을 동시에 비교]
  N) EMA 조건 없음 (3조건만)  <- EMA20 돌파가 실제로 도움이 되는지 보는 기준선
  R) 완화: 신호봉 종가가 EMA20 위
  S) 확실: 전날 종가는 EMA20 아래, 신호봉 종가는 EMA20 위 (아래->위 돌파)

[포함된 것]
  - 상장폐지/합병으로 사라진 종목도 함께 백테스트 (생존편향 완화, pykrx 과거 상장목록 활용)
  - 5일 평균 거래대금 조건(대회 유동성) 통과 여부 표시 및 따로 집계
  - 기간 분할: 앞 구간(조정용) / 뒤 구간(검증용) 따로 집계
  - 포트폴리오 시뮬레이션(동시 보유 제한), 랜덤 진입 대조군
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# 이 프로젝트 폴더. heikin_scanner_v2.py 의 BASE_DIR 과 반드시 같은 경로로 맞출 것
BASE_DIR = r'C:\Users\arlo0\OneDrive\바탕 화면\포트폴리오용 코드\heikinashi_method'
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)           # 작업 폴더가 어디든 heikin_scanner_v2.py 를 찾을 수 있게

import heikin_scanner_v2 as hs   # BASE_DIR 안의 heikin_scanner_v2.py (신호 탐지 함수, 설정값 재사용)


def P(filename):
    return os.path.join(BASE_DIR, filename)


# =========================== [설정] ===========================
UNIVERSE_PATH = P('stock_universe.csv')
CACHE_DIR = P('data_cache')
CACHE_MAX_AGE_DAYS = 1      # 캐시 파일이 이 일수보다 오래되면 다시 다운로드
OUT_TRADES = P('trades_v2.csv')
DELISTED_CACHE = P('delisted_codes.csv')   # 상장폐지/소멸 종목 목록 캐시 (지우면 다시 만듦)
EXCLUDE_SPAC = True         # 이름에 '스팩' 들어간 사라진 종목 제외 (합병 전 껍데기 회사)
DELIST_GAP_DAYS = 10        # 데이터가 오늘보다 이만큼 일찍 끝나면 '상장폐지'로 간주

START = '2021-01-01'
END = None                  # None 이면 오늘까지
N_STOCKS = None            # 처음엔 30으로 오류 확인 -> 300 -> 전체는 None (시간 오래 걸림, 수십 분~)
SEED = 42

MAX_HOLD = 60               # 최대 보유 거래일 (지나면 마지막 날 종가 청산)
COST_RT = 0.003             # 왕복 거래비용 가정 0.3% (수수료+세금+슬리피지, 직접 확인/조정 필요)
N_RANDOM = 200              # 랜덤 대조군 반복 횟수
WARMUP = 60                 # 랜덤 진입일은 데이터 시작 후 이 봉수 이후에서만 뽑음

SPLIT_DATE = '2024-01-01'   # 이 날짜 이전 신호 = 조정용 구간, 이후 = 검증용 구간
MAX_POS = 10                # 포트폴리오 시뮬: 동시 최대 보유 종목 수 (1종목 비중 = 1/MAX_POS)
MIN_TRADES_FOR_CONTROL = 20 # 거래가 이보다 적으면 랜덤 대조군은 건너뜀
USE_FLIP_EXIT = True        # 히말라야(스무스HA 양->음) 전환 시 매도. False 로 하면 손절/익절/기간만료만
# ==============================================================

VARIANTS = {
    'N': 'EMA 조건 없음 (3조건만, 기준선)',
    'R': 'EMA20 완화 (종가가 EMA20 위)',
    'S': 'EMA20 확실 (아래->위 돌파)',
}
EMA_MODE = {'N': 'none', 'R': 'relaxed', 'S': 'strict'}


def _today_str():
    return pd.Timestamp.today().strftime('%Y%m%d')


def load_prices(code):
    """pykrx 로 일봉(수정주가) 다운로드 + 하루짜리 캐시. 상장폐지 종목도 폐지 전까지 가져옴."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f'{code}.csv')
    fresh = os.path.exists(path) and (time.time() - os.path.getmtime(path)) < CACHE_MAX_AGE_DAYS * 86400
    if fresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        end = (END or _today_str()).replace('-', '')
        try:
            df = hs.stock.get_market_ohlcv(START.replace('-', ''), end, code)
        except Exception as e:
            print(f'  [다운로드 실패] {code}: {e}')
            return None
        if df is None or len(df) == 0:
            return None
        df = df.rename(columns={'시가': 'open', '고가': 'high', '저가': 'low',
                                '종가': 'close', '거래량': 'volume'})
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.to_csv(path)
        time.sleep(0.2)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df = df[df['volume'] > 0]
    return df if len(df) >= 150 else None


def _list_tickers(date_str):
    out = []
    for mk in ('KOSPI', 'KOSDAQ'):
        out += hs.stock.get_market_ticker_list(date_str, market=mk)
    return out


def _tickers_near(date, forward=True):
    """date 가 휴장일이면 앞(또는 뒤)으로 하루씩 옮겨 가며 종목 목록이 나오는 날을 찾는다."""
    for k in range(8):
        d = date + pd.Timedelta(days=k if forward else -k)
        if d > pd.Timestamp.today():
            continue
        try:
            t = _list_tickers(d.strftime('%Y%m%d'))
        except Exception:
            t = []
        if t:
            return t
    return []


def get_delisted(universe_codes):
    """
    START 이후 분기마다 상장 종목 목록을 모아서, 지금 목록과 대회 종목 목록에 없는 종목 = 사라진 종목을 찾는다.
    결과는 DELISTED_CACHE 에 저장해 두고 다음부터는 그 파일을 쓴다.
    """
    if os.path.exists(DELISTED_CACHE):
        d = pd.read_csv(DELISTED_CACHE, dtype=str)
        print(f'[상장폐지 종목] 캐시 사용: {len(d)}개 ({DELISTED_CACHE})')
        return d
    print('[상장폐지 종목] 과거 상장목록 조회 중 (처음 한 번만, 몇 분 걸림)...')
    seen = set()
    for d0 in pd.date_range(pd.Timestamp(START), pd.Timestamp.today(), freq='QS'):
        seen.update(_tickers_near(d0, forward=True))
    now = set(_tickers_near(pd.Timestamp.today().normalize(), forward=False))
    gone = sorted(seen - now - set(universe_codes))
    rows = []
    for code in gone:
        try:
            nm = hs.stock.get_market_ticker_name(code)
        except Exception:
            nm = ''
        if not isinstance(nm, str):
            nm = ''
        if EXCLUDE_SPAC and '스팩' in nm:
            continue
        rows.append((code, nm))
    d = pd.DataFrame(rows, columns=['code', '종목명'])
    d.to_csv(DELISTED_CACHE, index=False, encoding='utf-8-sig')
    print(f'[상장폐지 종목] 사라진 종목 {len(gone)}개 중 스팩 제외 {len(d)}개 저장 -> {DELISTED_CACHE}')
    return d


def prepare(raw):
    """신호 탐지 + 히말라야 전환 + 유동성 판단에 필요한 열을 모두 계산."""
    raw = hs.add_indicators(raw)
    ha1 = hs.heikin_ashi_raw(raw)
    ha2 = hs.smoothed_heikin_ashi(raw, hs.HA_LEN, hs.HA_LEN2)
    df = raw.join(ha1).join(ha2)          # 지표가 아직 계산 안 된 초반 봉은 NaN 그대로 둠 (비교는 False 처리됨)
    bull2 = (df['c2'] > df['o2']).fillna(False)
    bear2 = (df['o2'] > df['c2']).fillna(False)
    df['flip'] = bull2.shift(1, fill_value=False) & bear2     # 스무스HA 양봉 -> 음봉 전환 봉
    value = df['close'] * df['volume']                         # 거래대금 근사 (종가 x 거래량)
    df['liq_ok'] = value.rolling(hs.TRADING_VALUE_AVG_DAYS).mean() >= hs.MIN_AVG_TRADING_VALUE
    return df


def sim_trade(o, h, l, c, flip, i_sig, stop=None, stop_pct=None, use_flip=False):
    """
    신호일 i_sig 종가 후 -> 다음날 시가 진입.
    청산: 손절 / 익절 / (옵션) 히말라야 전환 다음날 시가 / 기간만료 종가 중 먼저 오는 것.
    같은 날 손절과 익절이 모두 가능하면 손절 먼저 (보수적). 갭으로 시가가 이미 손절가 아래/익절가 위면 시가 청산.
    반환: (진입가, 청산봉 위치, 청산가, 사유, 손절가, 익절가) 또는 None
    """
    n = len(o)
    e = i_sig + 1
    if e >= n:
        return None
    entry = o[e]
    if not entry > 0:
        return None
    if stop is None:
        stop = entry * (1 - stop_pct)
    if stop >= entry:
        return None
    target = entry + hs.RR_RATIO * (entry - stop)

    last = min(n - 1, e + MAX_HOLD - 1)
    oo, hh, ll = o[e:last + 1], h[e:last + 1], l[e:last + 1]
    k = len(oo)
    first = np.arange(k) == 0
    gap_stop = (~first) & (oo <= stop)
    gap_tgt = (~first) & (oo >= target)
    hit_stop = ll <= stop
    hit_tgt = hh >= target
    any_ev = gap_stop | gap_tgt | hit_stop | hit_tgt
    ev = int(np.argmax(any_ev)) if any_ev.any() else None

    fl = None
    if use_flip:
        fz = flip[e:last + 1]
        if fz.any():
            f = int(np.argmax(fz))
            if f + 1 < k:               # 전환 다음날 시가에 청산 (기간 안에 들어올 때만)
                fl = f

    if fl is not None and (ev is None or ev > fl):
        d = e + fl + 1
        return entry, d, o[d], '히말라야', stop, target

    if ev is not None:
        d = e + ev
        if gap_stop[ev]:
            return entry, d, o[d], '손절(갭)', stop, target
        if gap_tgt[ev]:
            return entry, d, o[d], '익절(갭)', stop, target
        if hit_stop[ev]:
            return entry, d, stop, '손절', stop, target
        return entry, d, target, '익절', stop, target

    return entry, last, c[last], '기간만료', stop, target


def run_real(df, code, name, variant, delisted=False):
    sigs = hs.find_signals(df, hs.TREND_WINDOW, EMA_MODE[variant])
    o, h, l, c = (df[k].to_numpy() for k in ('open', 'high', 'low', 'close'))
    flip = df['flip'].to_numpy()
    liq = df['liq_ok'].to_numpy()
    trades = []
    busy_until = -1
    for sg in sigs:
        i = sg['idx']
        if i + 1 <= busy_until:           # 이미 보유 중이면 새 신호는 건너뜀
            continue
        r = sim_trade(o, h, l, c, flip, i, stop=sg['stop'], use_flip=USE_FLIP_EXIT)
        if r is None:
            continue
        entry, d_exit, px_exit, reason, stop, target = r
        busy_until = d_exit
        if delisted and reason == '기간만료' and d_exit == len(o) - 1:
            reason = '상폐청산'            # 데이터가 끝날 때(상장폐지/합병)까지 들고 있다가 마지막 종가로 정리
        trades.append({
            '전략': variant, '종목코드': code, '종목명': name,
            '신호일': df.index[i].date(), '진입일': df.index[i + 1].date(),
            '청산일': df.index[d_exit].date(),
            '진입가': entry, '손절가': stop, '익절가': target, '청산가': px_exit,
            '청산사유': reason,
            '수익률': px_exit / entry - 1 - COST_RT,
            'R배수': (px_exit - entry) / (entry - stop),
            '보유일': d_exit - (i + 1) + 1,
            '손절폭': (entry - stop) / entry,
            'EMA확실': bool(sg['ema_cross']),
            '유동성통과': bool(liq[i]),
            '사라진종목': bool(delisted),
        })
    return trades


def summarize(t, title):
    if len(t) == 0:
        print(f'\n[{title}] 거래 0건')
        return
    r = t['수익률']
    win = r[r > 0]
    lose = r[r <= 0]
    pf = win.sum() / -lose.sum() if len(lose) and lose.sum() != 0 else float('nan')
    print(f'\n[{title}] 거래 {len(t)}건')
    print(f'  승률 {len(win) / len(t) * 100:.1f}% | 평균수익률 {r.mean() * 100:.2f}% | 중앙값 {r.median() * 100:.2f}%')
    print(f'  평균이익 {win.mean() * 100 if len(win) else 0:.2f}% | 평균손실 {lose.mean() * 100 if len(lose) else 0:.2f}% | Profit Factor {pf:.2f}')
    print(f'  평균 R배수 {t["R배수"].mean():.2f} | 평균 보유 {t["보유일"].mean():.1f}일')
    print('  청산사유별 건수:', t['청산사유'].value_counts().to_dict())


def portfolio_sim(t, title):
    """
    동시 보유 MAX_POS 종목 제한, 종목당 비중 1/MAX_POS 로 거래를 순서대로 받아 수익률 / MDD 를 근사.
    (청산 시점에 실현된 손익만 반영하는 근사치. 같은 날 신호가 넘치면 EMA 확실 돌파를 먼저 채움)
    """
    if len(t) == 0:
        return
    t = t.sort_values(['진입일', 'EMA확실'], ascending=[True, False])
    w = 1.0 / MAX_POS
    open_exits, acc, skipped = [], [], 0
    for r in t.itertuples(index=False):
        open_exits = [x for x in open_exits if x >= r.진입일]
        if len(open_exits) >= MAX_POS:
            skipped += 1
            continue
        acc.append(r)
        open_exits.append(r.청산일)
    acc.sort(key=lambda r: r.청산일)
    eq, curve = 1.0, []
    for r in acc:
        eq *= 1 + w * r.수익률
        curve.append(eq)
    curve = np.array(curve)
    mdd = (curve / np.maximum.accumulate(curve) - 1).min()
    years = max((acc[-1].청산일 - t['진입일'].min()).days / 365.25, 1e-9)
    cagr = eq ** (1 / years) - 1 if years >= 0.5 else float('nan')
    print(f'\n[포트폴리오 근사 - {title}] 동시보유 최대 {MAX_POS}종목, 종목당 {w * 100:.0f}%')
    print(f'  체결 {len(acc)}건 / 자리 없어 건너뜀 {skipped}건')
    print(f'  누적수익률 {(eq - 1) * 100:.1f}% | 연환산 {cagr * 100:.1f}% | 최대낙폭(실현 기준 근사) {mdd * 100:.1f}%')


def random_control(trades_df, store, use_flip, use_liq_pool, title):
    """
    실제 거래 한 건마다 같은 종목의 무작위 날짜에 같은 손절폭 / 같은 손익비 / 같은 청산규칙으로 진입한 가상 거래를 만들고,
    실제 거래 수만큼의 평균수익률을 N_RANDOM번 반복해 '운으로 얻을 평균수익률 분포'와 비교한다.
    use_liq_pool=True 면 무작위 날짜도 유동성 조건을 통과한 날 중에서만 뽑는다.
    """
    if len(trades_df) < MIN_TRADES_FOR_CONTROL:
        print(f'\n[랜덤 대조군 - {title}] 거래 {len(trades_df)}건 < {MIN_TRADES_FOR_CONTROL}건이라 건너뜀')
        return
    rng = np.random.default_rng(SEED)
    T = len(trades_df)
    rets = np.full((T, N_RANDOM), np.nan)
    for t_i, row in enumerate(trades_df.itertuples(index=False)):
        o, h, l, c, flip, liq = store[row.종목코드]
        n = len(o)
        ok = np.arange(n)
        mask = (ok >= WARMUP) & (ok <= n - 2)     # 실제 신호와 똑같이 데이터 끝(상장폐지 포함)에서 잘리는 것도 허용
        if use_liq_pool:
            mask &= liq
        pool = ok[mask]
        if len(pool) == 0:
            continue
        days = rng.choice(pool, size=N_RANDOM)
        for k, d in enumerate(days):
            r = sim_trade(o, h, l, c, flip, int(d), stop_pct=row.손절폭, use_flip=use_flip)
            if r is None:
                continue
            rets[t_i, k] = r[2] / r[0] - 1 - COST_RT
    null_means = np.nanmean(rets, axis=0)
    real_mean = trades_df['수익률'].mean()
    p = (np.sum(null_means >= real_mean) + 1) / (N_RANDOM + 1)
    print(f'\n[랜덤 대조군 - {title}]')
    print(f'  실제 신호 평균수익률 {real_mean * 100:.2f}%')
    print(f'  랜덤 진입 평균수익률 분포: 평균 {np.nanmean(null_means) * 100:.2f}% | 5~95% 구간 '
          f'{np.nanpercentile(null_means, 5) * 100:.2f}% ~ {np.nanpercentile(null_means, 95) * 100:.2f}%')
    print(f'  p값(랜덤이 실제 이상일 비율) {p:.3f}  (작을수록 운이 아닐 가능성이 큼, 0.05 이하면 유의)')


def report(t, variant):
    name = f'기준 {variant}: {VARIANTS[variant]}'
    print('\n' + '=' * 70)
    print(name)
    print('=' * 70)
    if len(t) == 0:
        print('거래 없음')
        return
    summarize(t, f'{variant} | 전체 (유동성 무관)')
    summarize(t[t['유동성통과']], f'{variant} | 유동성 통과(5일 평균 거래대금 조건)만')
    summarize(t[t['사라진종목']], f'{variant} | 그중 사라진(상장폐지/합병) 종목 거래만')

    split = pd.Timestamp(SPLIT_DATE).date()
    summarize(t[t['신호일'] < split], f'{variant} | 조정용 구간 (신호일 < {SPLIT_DATE})')
    summarize(t[t['신호일'] >= split], f'{variant} | 검증용 구간 (신호일 >= {SPLIT_DATE})')

    portfolio_sim(t[t['유동성통과']], f'{variant} | 유동성 통과 거래만')


def main():
    if not os.path.isdir(BASE_DIR):
        print(f'[오류] BASE_DIR 폴더를 찾을 수 없어요: {BASE_DIR}\n  코드 위쪽 BASE_DIR 경로를 확인하세요.')
        return
    uni = pd.read_csv(UNIVERSE_PATH, dtype=str)
    uni['code'] = uni['종목코드'].str.replace('^A', '', regex=True)
    base = pd.DataFrame({'code': uni['code'], 'name': uni['종목명'], 'gone': False})

    gone = get_delisted(set(uni['code']))
    gone_df = pd.DataFrame({'code': gone['code'], 'name': gone['종목명'], 'gone': True})
    allu = pd.concat([base, gone_df], ignore_index=True).drop_duplicates('code')
    print(f'백테스트 대상: 현재 종목 {len(base)}개 + 사라진 종목 {int(allu["gone"].sum())}개')

    if N_STOCKS is not None:
        allu = allu.sample(min(N_STOCKS, len(allu)), random_state=SEED)
        print(f'  무작위 {len(allu)}개 사용 (그중 사라진 종목 {int(allu["gone"].sum())}개)')

    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=DELIST_GAP_DAYS)
    trades = {v: [] for v in VARIANTS}
    store = {}
    for k, row in enumerate(allu.itertuples(index=False), 1):
        code, name = row.code, row.name
        raw = load_prices(code)
        if raw is None:
            continue
        delisted = bool(raw.index[-1] < cutoff)
        try:
            df = prepare(raw)
        except Exception as e:
            print(f'  [계산 실패] {name}({code}): {e}')
            continue
        store[code] = tuple(df[x].to_numpy() for x in ('open', 'high', 'low', 'close')) + (
            df['flip'].to_numpy(), df['liq_ok'].to_numpy())
        for v in VARIANTS:
            trades[v].extend(run_real(df, code, name, v, delisted))
        if k % 20 == 0 or k == len(allu):
            print(f'  진행 {k}/{len(allu)}')

    dfs = {}
    for v in VARIANTS:
        dfs[v] = pd.DataFrame(trades[v])
        if len(dfs[v]):
            dfs[v] = dfs[v].sort_values('진입일').reset_index(drop=True)

    if all(len(d) == 0 for d in dfs.values()):
        print('거래가 하나도 없습니다. 종목 수를 늘리거나 기간을 확인하세요.')
        return

    all_t = pd.concat([d for d in dfs.values() if len(d)], ignore_index=True)
    all_t.to_csv(OUT_TRADES, index=False, encoding='utf-8-sig')
    print(f'\n{OUT_TRADES} 저장 (N+R+S 합계 {len(all_t)}건, 같은 거래가 기준별로 겹쳐 들어있으니 전략 열로 구분)')

    for v in VARIANTS:
        report(dfs[v], v)
        if len(dfs[v]):
            use_flip = USE_FLIP_EXIT
            random_control(dfs[v], store, use_flip, False, f'{v} | 전체')
            random_control(dfs[v][dfs[v]['유동성통과']], store, use_flip, True, f'{v} | 유동성 통과')


if __name__ == '__main__':
    main()