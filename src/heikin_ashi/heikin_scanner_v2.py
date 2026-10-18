# -*- coding: utf-8 -*-
"""
하이킨아시 스윙 스캐너 v2 + 보유종목 관리   (v1 heikin_scanner.py 는 그대로 두고 따로 쓰는 버전)
=================================================
필수 4조건 (모두 만족해야 신호, 가점 없음)
  (1) 도지 연속 후 평평한 바닥 + 긴 윗꼬리 양봉 (일반 하이킨아시)
  (2) 신호봉 거래량 > 직전 도지 거래량
  (3) 도지 직전 하락추세 (음봉 비율)
  (4) 신호봉에서 EMA20 돌파  -> 기준 2가지를 모두 표시 (설정 SCAN_EMA_MODE 로 통과 기준 선택)
        확실: 전날 종가는 EMA20 아래, 신호봉 종가는 EMA20 위 (아래->위 돌파)
        완화: 신호봉 종가가 EMA20 위
매도: 손절가 / 익절가 도달, 또는 스무스 하이킨아시(히말라야) 양봉->음봉 전환 즉시 매도
      (익절가에 못 가도 전환되면 매도)
=================================================
사용법 (Spyder 기준):
  1. 아래 [설정]의 BASE_DIR 을 이 프로젝트 폴더 경로로 맞춘다.
  2. 그 폴더에 stock_universe.csv, sector1.csv (보유종목 점검을 쓰면 holdings.xlsx 도) 를 둔다.
  3. Run File(F5)로 돌린다. 작업 폴더가 어디든 상관없이 BASE_DIR 안의 파일만 읽고 쓴다.
  4. 결과는 BASE_DIR 안의 signals_result_v2.xlsx 로 저장되고, 콘솔에도 요약이 출력된다.
처음엔 TEST_MODE=True 로 소수 종목만 테스트해보는 걸 추천.
"""

import os
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

from pykrx import stock
import pandas_ta as ta

# =========================== [설정] ===========================
# 이 프로젝트 폴더 (파일 읽기/쓰기는 전부 이 폴더 안에서 함). 폴더 위치를 옮기면 이 줄만 고치면 됨
BASE_DIR = r'C:\Users\arlo0\OneDrive\바탕 화면\포트폴리오용 코드\heikinashi_method'


def P(filename):
    return os.path.join(BASE_DIR, filename)


UNIVERSE_PATH = P('stock_universe.csv')
SECTOR_PATH = P('sector1.csv')
OUTPUT_PATH = P('signals_result_v2.xlsx')

LOOKBACK_DAYS = 1500       # 데이터 조회 기간 (주봉 지표 워밍업까지 고려해 약 5년)
HA_LEN = 10                 # 1차 하이킨아시 평활화 길이 (파인스크립트 len)
HA_LEN2 = 10                # 2차 평활화 길이 (파인스크립트 len2)

DOJI_BODY_PCT = 0.10        # 몸통 <= 레인지의 10% 이면 도지 (일반 하이킨아시 기준)
FLAT_WICK_PCT = 0.10        # 아래꼬리 <= 레인지의 10% 이면 "평평" (일반 하이킨아시 기준)
LONG_UPPER_WICK_PCT = 0.30  # 윗꼬리 >= 레인지의 30% 이면 "길게 위로 찌름" (일반 하이킨아시 기준)

EMA_TREND_LEN = 20          # 돌파 기준 EMA
# 신호 통과 기준 (필수 4번째 조건): 'strict' = 신호봉에서 아래->위 돌파만 / 'relaxed' = 종가가 EMA20 위면 통과 / 'none' = EMA 조건 없음
# 어느 쪽이든 결과 엑셀에는 '확실 돌파' 여부가 O/- 로 같이 표시되므로 relaxed 로 두고 확실한 것만 골라 봐도 됨
SCAN_EMA_MODE = 'relaxed'

USE_DAILY = True            # 일봉 스캔 켜기/끄기
USE_WEEKLY = True           # 주봉 스캔 켜기/끄기 (이번 주 진행 중인 봉도 포함)
TREND_WINDOW = 12           # [일봉] 첫 도지 직전 몇 개 봉을 보고 '하락추세였는지' 판단할지
TREND_WINDOW_W = 8          # [주봉] 같은 기준 (8주)
TREND_BEARISH_RATIO = 0.65  # 그 구간에서 음봉 비율이 이 값 이상이면 하락추세로 인정
STOP_BUFFER = 0.005         # 손절가 = 도지들 최저 저가 x (1 - 0.5%)
RR_RATIO = 2.0              # 손익비 (익절 = 진입 + 2 x (진입-손절))
HOLDINGS_PATH = P('holdings.xlsx')  # 보유종목 엑셀 (직접 입력)
HOLDINGS_ONLY = False       # True면 신규신호 스캔은 건너뛰고 보유종목만 점검 (결과: holdings_result_v2.xlsx)

RECHECK_WINDOW_W = 2        # [주봉] 최근 몇 주 내 신호를 눌림목 재탐색 대상으로 볼지
RECHECK_WINDOW = 5          # 최근 며칠 내 신호를 "눌림목 재탐색" 대상으로 볼지
SIGNAL_LOOKBACK_FOR_SCAN = 1  # 오늘(가장 최근 봉) 신규 신호만 볼지(1), 더 늘리면 최근 n봉 전부 스캔

# --- 매수 불가 종목 필터 (대회 규정) ---
MIN_MARKET_CAP = 100_000_000_000       # 시가총액 1,000억원 미만 제외
MIN_AVG_TRADING_VALUE = 3_000_000_000  # 최근 5거래일 평균 거래대금 30억원 이하 제외
TRADING_VALUE_AVG_DAYS = 5             # 거래대금 평균 낼 거래일 수

# True  -> 대회 조건(시총/거래대금) 미달 종목은 아예 스캔 대상에서 빼버림 (대회용, 결과에 대회 조건 미달 종목 없음)
# False -> 전부 다 스캔하고, 결과 엑셀에 '대회매수가능' 컬럼(적합/부적합)만 추가로 붙여줌 (개인투자 겸용)
EXCLUDE_INELIGIBLE = False

TEST_MODE = True            # True면 앞의 TEST_N개 종목만 테스트
TEST_N = 30

SLEEP_SEC = 0.15            # pykrx 과호출 방지용 딜레이(초)
# ================================================================


def load_universe():
    df = pd.read_csv(UNIVERSE_PATH, dtype=str)
    df['pykrx_code'] = df['종목코드'].str.replace('^A', '', regex=True)
    return df


def load_sector_limits():
    df = pd.read_csv(SECTOR_PATH)
    return df[['섹터코드', '섹터명', '시장 비중', '한도 비중']]


def get_eligible_codes(end):
    """
    '매수 불가 종목' 중 시가총액 / 거래대금 조건을 코드로 거를 수 있는 종목코드 집합을 반환.
    - 시가총액 1,000억원 미만 제외
    - 최근 5거래일 평균 거래대금 30억원 이하 제외
    종목 단위로 하나씩 조회하지 않고, '날짜 하나당 전체 시장 스냅샷'을 받아오는 방식이라
    종목이 2,600개가 넘어도 API 호출은 며칠치(기본 5번)만 하면 됨 -> 훨씬 빠름.
    실패하면 (필터를 건너뛴다는 의미로) None 을 반환.
    """
    base_date = None
    cap_today = None
    d = datetime.strptime(end, '%Y%m%d')
    for _ in range(10):
        ds = d.strftime('%Y%m%d')
        try:
            tmp = stock.get_market_cap_by_ticker(ds, market="ALL")
            if tmp is not None and len(tmp) > 0:
                cap_today = tmp
                base_date = d
                break
        except Exception as e:
            print(f"  [시총조회 실패] {ds}: {e}")
        d -= timedelta(days=1)

    if cap_today is None:
        print("[경고] 시가총액 데이터를 가져오지 못해 시총/거래대금 필터를 건너뜁니다.")
        return None

    value_frames = []
    d = base_date
    tries = 0
    while len(value_frames) < TRADING_VALUE_AVG_DAYS and tries < 15:
        ds = d.strftime('%Y%m%d')
        try:
            tmp = stock.get_market_cap_by_ticker(ds, market="ALL")
            if tmp is not None and len(tmp) > 0:
                value_frames.append(tmp['거래대금'])
        except Exception as e:
            print(f"  [거래대금조회 실패] {ds}: {e}")
        d -= timedelta(days=1)
        tries += 1

    eligible = set(cap_today.index[cap_today['시가총액'] >= MIN_MARKET_CAP])

    if value_frames:
        avg_value = pd.concat(value_frames, axis=1).mean(axis=1)
        eligible &= set(avg_value.index[avg_value >= MIN_AVG_TRADING_VALUE])
    else:
        print("[경고] 거래대금 데이터를 가져오지 못해 거래대금 필터는 건너뜁니다.")

    return eligible


def fetch_ohlcv(code, start, end):
    """pykrx로 일봉 OHLCV 가져오기. 실패하면 None 반환."""
    try:
        df = stock.get_market_ohlcv(start, end, code)
        if df is None or len(df) < 60:
            return None
        df = df.rename(columns={
            '시가': 'open', '고가': 'high', '저가': 'low',
            '종가': 'close', '거래량': 'volume'
        })
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception:
        return None


def heikin_ashi_raw(df):
    """일반(평활화 없는) 하이킨아시. 도지/평평한바닥 패턴 탐지에 사용."""
    o = df['open']
    h = df['high']
    l = df['low']
    c = df['close']

    haclose = (o + h + l + c) / 4
    haopen = pd.Series(index=df.index, dtype=float)
    haopen.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2
    for i in range(1, len(df)):
        haopen.iloc[i] = (haopen.iloc[i - 1] + haclose.iloc[i - 1]) / 2

    hahigh = pd.concat([h, haopen, haclose], axis=1).max(axis=1)
    halow = pd.concat([l, haopen, haclose], axis=1).min(axis=1)

    out = pd.DataFrame({'o1': haopen, 'h1': hahigh, 'l1': halow, 'c1': haclose}, index=df.index)
    return out


def smoothed_heikin_ashi(df, len1=HA_LEN, len2=HA_LEN2):
    """파인스크립트 'Smoothed Heiken Ashi Candles' 그대로 복제."""
    o = ta.ema(df['open'], length=len1)
    c = ta.ema(df['close'], length=len1)
    h = ta.ema(df['high'], length=len1)
    l = ta.ema(df['low'], length=len1)

    haclose = (o + h + l + c) / 4
    haopen = pd.Series(index=df.index, dtype=float)
    haopen.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2 if not np.isnan(o.iloc[0]) else np.nan
    for i in range(1, len(df)):
        if np.isnan(haopen.iloc[i - 1]):
            haopen.iloc[i] = (o.iloc[i] + c.iloc[i]) / 2
        else:
            haopen.iloc[i] = (haopen.iloc[i - 1] + haclose.iloc[i - 1]) / 2

    hahigh = pd.concat([h, haopen, haclose], axis=1).max(axis=1)
    halow = pd.concat([l, haopen, haclose], axis=1).min(axis=1)

    o2 = ta.ema(haopen, length=len2)
    c2 = ta.ema(haclose, length=len2)
    h2 = ta.ema(hahigh, length=len2)
    l2 = ta.ema(halow, length=len2)

    out = pd.DataFrame({'o2': o2, 'h2': h2, 'l2': l2, 'c2': c2}, index=df.index)
    return out


def add_indicators(df):
    df = df.copy()
    df['ema20'] = ta.ema(df['close'], length=EMA_TREND_LEN)
    return df


def tick_size(price):
    """한국거래소 주식 호가가격단위 (2023.1.25~ 코스피/코스닥 공통, ETF 제외)."""
    if price < 2000:
        return 1
    if price < 5000:
        return 5
    if price < 20000:
        return 10
    if price < 50000:
        return 50
    if price < 200000:
        return 100
    if price < 500000:
        return 500
    return 1000


def round_to_tick(price):
    """가격을 호가단위에 맞춰 반올림 (주문창에 그대로 입력할 수 있게)."""
    t = tick_size(price)
    return int(round(price / t) * t)


def find_signals(df, trend_window, ema_mode=None):
    """
    필수 4조건을 모두 만족하는 봉을 전부 찾아서 리스트로 반환.
      ① 직전 봉(들)이 도지 -> 현재 봉이 양봉 + 아래꼬리 평평 + 윗꼬리 김 (일반 하이킨아시)
      ② 현재 봉(신호 봉) 거래량 > 바로 직전 도지 봉 거래량
      ③ 연속 도지가 시작되기 직전 trend_window 봉 중 음봉 비율 >= TREND_BEARISH_RATIO (하락 추세)
      ④ EMA20 돌파  (ema_mode: 'strict' 아래->위 돌파 / 'relaxed' 종가가 EMA20 위 / 'none' 조건 없음)
    진입가 = 신호일 실제 종가
    손절가 = 연속 도지 전체의 실제 최저가 x (1 - STOP_BUFFER)
    익절가 = 진입가 + RR_RATIO x (진입가 - 손절가)
    """
    if ema_mode is None:
        ema_mode = SCAN_EMA_MODE
    n = len(df)
    rng = (df['h1'] - df['l1']).to_numpy()
    body = (df['c1'] - df['o1']).abs().to_numpy()
    lower_wick = (df[['o1', 'c1']].min(axis=1) - df['l1']).to_numpy()
    upper_wick = (df['h1'] - df[['o1', 'c1']].max(axis=1)).to_numpy()

    valid = rng > 0
    is_doji = valid & (body <= DOJI_BODY_PCT * rng)
    is_flat = valid & (lower_wick <= FLAT_WICK_PCT * rng)
    is_long_up = valid & (upper_wick >= LONG_UPPER_WICK_PCT * rng)
    is_bull = (df['c1'] > df['o1']).to_numpy()
    is_bear = (df['c1'] < df['o1']).to_numpy()

    close = df['close'].to_numpy()
    low = df['low'].to_numpy()
    volume = df['volume'].to_numpy()
    ema20 = df['ema20'].to_numpy()

    # 각 봉까지 이어진 연속 도지 개수
    doji_run = np.zeros(n, dtype=int)
    for i in range(n):
        if is_doji[i]:
            doji_run[i] = doji_run[i - 1] + 1 if i > 0 else 1

    signals = []
    for i in range(1, n):
        if not (is_bull[i] and is_flat[i] and is_long_up[i]):
            continue
        run = doji_run[i - 1]
        if run == 0:                      # 직전 봉이 도지가 아님
            continue
        s = i - run                       # 연속 도지의 첫 봉 위치
        if not volume[i] > volume[i - 1]:  # 거래량: 신호 봉 > 바로 직전 도지
            continue
        w0 = s - trend_window
        if w0 < 0:
            continue
        bear_ratio = is_bear[w0:s].mean()  # 도지 구간 직전 trend_window 봉의 음봉 비율
        if bear_ratio < TREND_BEARISH_RATIO:
            continue

        ema_above = bool(close[i] > ema20[i])                         # 완화: 종가가 EMA20 위
        ema_cross = bool(close[i - 1] < ema20[i - 1] and close[i] > ema20[i])   # 확실: 아래->위 돌파
        if ema_mode == 'strict' and not ema_cross:
            continue
        if ema_mode == 'relaxed' and not ema_above:
            continue

        entry = int(close[i])
        stop = round_to_tick(low[s:i].min() * (1 - STOP_BUFFER))
        if stop >= entry:
            continue
        target = round_to_tick(entry + RR_RATIO * (entry - stop))

        signals.append({
            'idx': i, 'entry': entry, 'stop': stop, 'target': target,
            'doji_n': int(run), 'bear_ratio': float(bear_ratio),
            'ema_cross': ema_cross, 'ema_above': ema_above,
        })
    return signals


def to_weekly(raw):
    """일봉 -> 주봉 (금요일 기준 주 단위). 시가=첫날, 고가=최대, 저가=최소, 종가=마지막날, 거래량=합계."""
    w = raw[['open', 'high', 'low', 'close', 'volume']].resample('W-FRI').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    return w.dropna(subset=['open', 'close'])


def analyze_timeframe(raw, tf, trend_window, recheck_window, code, name, sector_code):
    """한 종목의 한 시간프레임(일봉/주봉)을 분석해서 결과 dict 리스트(또는 None) 반환."""
    raw = add_indicators(raw)
    ha1 = heikin_ashi_raw(raw)                          # 패턴/추세 판단용 (일반 하이킨아시)
    ha2 = smoothed_heikin_ashi(raw, HA_LEN, HA_LEN2)     # 매도 타이밍용 (스무스 하이킨아시)
    df = raw.join(ha1).join(ha2)
    df = df.dropna().copy()
    if len(df) < trend_window + 20:
        return None

    signals = find_signals(df, trend_window)
    if not signals:
        return None

    n = len(df)
    cur_close = int(df['close'].iloc[-1])
    # 스무스 하이킨아시: 직전 봉 양봉 -> 현재 봉 음봉 (오늘 막 전환됐으면 매도)
    ha2_color = '양봉(상승)' if df['c2'].iloc[-1] > df['o2'].iloc[-1] else '음봉(하락)'
    sell_now = bool((df['c2'].iloc[-2] > df['o2'].iloc[-2]) and (df['o2'].iloc[-1] > df['c2'].iloc[-1]))

    results = []
    for sg in signals:
        i = sg['idx']
        kind = None
        if i >= n - SIGNAL_LOOKBACK_FOR_SCAN:
            kind = '신규신호'
        elif (n - 1 - recheck_window) <= i <= n - 2:
            # 눌림목 재탐색: 최근 신호 후 손절가를 안 깼고, 현재가가 진입가 아래로 눌려있는 경우
            hit_stop = bool((df['low'].iloc[i:] <= sg['stop']).any())
            pulled_back = sg['stop'] < cur_close < sg['entry']
            if (not hit_stop) and pulled_back:
                kind = '눌림목재탐색'
        if kind is None:
            continue

        results.append({
            '종목코드': code, '종목명': name, '섹터코드': sector_code, '봉': tf,
            '신호일': df.index[i].date(), '구분': kind,
            'EMA20확실돌파': 'O' if sg['ema_cross'] else '-',
            'EMA20종가위': 'O' if sg['ema_above'] else '-',
            '도지개수': sg['doji_n'],
            '하락봉비율(%)': round(sg['bear_ratio'] * 100),
            '진입가': sg['entry'], '손절가': sg['stop'], '익절가': sg['target'],
            '현재가': cur_close,
            '스무스HA현재색': ha2_color,
            '스무스HA매도전환': sell_now,
        })

    return results if results else None


def analyze_one_stock(code, name, sector_code, start, end):
    raw = fetch_ohlcv(code, start, end)          # 일봉은 한 번만 내려받고 주봉은 여기서 변환
    if raw is None:
        return None
    out = []
    if USE_DAILY:
        r = analyze_timeframe(raw, '일봉', TREND_WINDOW, RECHECK_WINDOW, code, name, sector_code)
        if r:
            out.extend(r)
    if USE_WEEKLY:
        r = analyze_timeframe(to_weekly(raw), '주봉', TREND_WINDOW_W, RECHECK_WINDOW_W, code, name, sector_code)
        if r:
            out.extend(r)
    return out if out else None


# ======================= 보유 종목 점검 =======================
def load_holdings():
    """holdings.xlsx 를 읽어서 DataFrame 반환. 파일이 없거나 비어있으면 None."""
    if not os.path.exists(HOLDINGS_PATH):
        print(f"[보유종목] {HOLDINGS_PATH} 파일이 없어 보유종목 점검은 건너뜁니다.")
        return None

    h = pd.read_excel(HOLDINGS_PATH, dtype=str)
    h.columns = [str(c).strip() for c in h.columns]
    if '종목코드' not in h.columns:
        print(f"[보유종목] {HOLDINGS_PATH} 에 '종목코드' 열이 없어 건너뜁니다.")
        return None

    h = h.dropna(subset=['종목코드']).copy()
    if len(h) == 0:
        print("[보유종목] 보유종목 파일이 비어있어 점검은 건너뜁니다.")
        return None

    h['종목코드'] = (h['종목코드'].astype(str).str.strip()
                     .str.replace(r'\.0$', '', regex=True).str.upper().str.zfill(6))
    for col in ['종목명', '매수일', '매수가', '손절가', '익절가']:
        if col not in h.columns:
            h[col] = np.nan
    h['매수일'] = pd.to_datetime(h['매수일'], errors='coerce')
    for col in ['매수가', '손절가', '익절가']:
        h[col] = pd.to_numeric(h[col], errors='coerce')
    return h.reset_index(drop=True)


def check_holding(row, start, end):
    """보유 종목 1개 점검: 손절/익절 도달 또는 스무스 하이킨아시(히말라야) 양->음 전환이면 매도."""
    base = {
        '종목코드': row['종목코드'], '종목명': row['종목명'],
        '매수일': row['매수일'].date() if pd.notna(row['매수일']) else '',
        '매수가': row['매수가'], '손절가': row['손절가'], '익절가': row['익절가'],
    }
    raw = fetch_ohlcv(row['종목코드'], start, end)
    if raw is None:
        base.update({'현재가': '', '수익률(%)': '', '스무스HA': '', '최근전환일': '', '상태': '데이터 조회 실패'})
        return base

    ha2 = smoothed_heikin_ashi(raw, HA_LEN, HA_LEN2)
    df = raw.join(ha2).dropna().copy()

    bull = df['c2'] > df['o2']
    flip = bull.shift(1, fill_value=False) & (df['o2'] > df['c2'])   # 양 -> 음 전환 봉

    # 매수일이 있으면 그날부터, 비어 있으면 오늘 봉 하나만 본다 (예전 데이터로 잘못 걸리는 것 방지)
    since = df[df.index >= row['매수일']] if pd.notna(row['매수일']) else df.iloc[-1:]

    cur = int(df['close'].iloc[-1])
    buy = row['매수가']
    ret = round((cur / buy - 1) * 100, 2) if pd.notna(buy) and buy > 0 else ''

    stop, target = row['손절가'], row['익절가']
    stop_date = None
    target_date = None
    if pd.notna(stop):
        hit = since.index[(since['low'] <= stop).to_numpy()]
        stop_date = hit[0] if len(hit) else None
    if pd.notna(target):
        hit = since.index[(since['high'] >= target).to_numpy()]
        target_date = hit[0] if len(hit) else None

    flip_dates = since.index[flip.loc[since.index].to_numpy()]
    last_flip = flip_dates[-1] if len(flip_dates) else None
    flip_today = bool(flip.iloc[-1])
    now_bull = bool(bull.iloc[-1])

    if stop_date is not None and (target_date is None or stop_date <= target_date):
        status = f"손절가 도달 ({stop_date.date()}) -> 매도"
    elif target_date is not None:
        status = f"익절가 도달 ({target_date.date()}) -> 매도"
    elif flip_today:
        status = "매도: 스무스HA 양봉->음봉 전환 (오늘)"
    elif last_flip is not None:
        status = f"매도: 스무스HA 음봉 전환 이미 발생 (전환일 {last_flip.date()})"
    else:
        status = "보유 유지"

    base.update({
        '현재가': cur, '수익률(%)': ret,
        '스무스HA': '양봉(상승)' if now_bull else '음봉(하락)',
        '최근전환일': last_flip.date() if last_flip is not None else '',
        '상태': status,
    })
    return base


def main():
    if not os.path.isdir(BASE_DIR):
        print(f"[오류] BASE_DIR 폴더를 찾을 수 없어요: {BASE_DIR}\n  코드 위쪽 [설정]의 BASE_DIR 경로를 확인하세요.")
        return
    universe = load_universe()
    sectors = load_sector_limits()

    end = datetime.today().strftime('%Y%m%d')
    start = (datetime.today() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y%m%d')

    if HOLDINGS_ONLY:
        print("[보유종목만 점검 모드] 신규신호 스캔은 건너뜁니다.")
        holdings = load_holdings()
        if holdings is None:
            return
        rows = []
        for _, hr in holdings.iterrows():
            try:
                rows.append(check_holding(hr, start, end))
            except Exception as e:
                print(f"  [보유종목 에러] {hr['종목명']}({hr['종목코드']}): {e}")
            time.sleep(SLEEP_SEC)
        if not rows:
            print("점검 결과가 없습니다.")
            return
        holding_df = pd.DataFrame(rows)
        try:
            holding_df.to_excel(P('holdings_result_v2.xlsx'), sheet_name='보유종목', index=False)
        except PermissionError:
            print("\n[저장 실패] holdings_result_v2.xlsx 가 엑셀에서 열려있어요. 닫고 다시 실행해주세요.")
            return
        print(f"\n완료! 보유종목 {len(holding_df)}건 점검 -> {P('holdings_result_v2.xlsx')} 저장됨\n")
        print(holding_df[['종목명', '현재가', '수익률(%)', '스무스HA', '상태']].to_string(index=False))
        return

    eligible = get_eligible_codes(end)

    if EXCLUDE_INELIGIBLE:
        if eligible is not None:
            before_n = len(universe)
            universe = universe[universe['pykrx_code'].isin(eligible)].reset_index(drop=True)
            print(f"[매수가능 필터] {before_n}개 -> {len(universe)}개 "
                  f"(시총 {MIN_MARKET_CAP/1e8:.0f}억원 이상 & "
                  f"{TRADING_VALUE_AVG_DAYS}일평균거래대금 {MIN_AVG_TRADING_VALUE/1e8:.0f}억원 이상)")
    else:
        print("[매수가능 필터] EXCLUDE_INELIGIBLE=False -> 전종목 스캔, "
              "결과에 '대회매수가능' 컬럼으로만 표시합니다.")

    if TEST_MODE:
        universe = universe.head(TEST_N)
        print(f"[테스트 모드] {TEST_N}개 종목만 스캔합니다.")

    # ---------- 1) 신규 신호 스캔 ----------
    all_results = []
    n = len(universe)
    for i, r in enumerate(universe.itertuples(), 1):
        code, name, sector_code = r.pykrx_code, r.종목명, r.섹터코드
        try:
            res = analyze_one_stock(code, name, sector_code, start, end)
            if res:
                all_results.extend(res)
        except Exception as e:
            print(f"  [에러] {name}({code}): {e}")
        if i % 50 == 0 or i == n:
            print(f"진행 {i}/{n} ...")
        time.sleep(SLEEP_SEC)

    base_cols = ['종목코드', '종목명', '섹터코드', '봉', '신호일', '구분', 'EMA20확실돌파', 'EMA20종가위',
                 '도지개수', '하락봉비율(%)',
                 '진입가', '손절가', '익절가', '현재가', '스무스HA현재색', '스무스HA매도전환']
    if all_results:
        result_df = pd.DataFrame(all_results)
        result_df = result_df.merge(sectors, on='섹터코드', how='left')
        if eligible is not None:
            result_df['대회매수가능'] = np.where(result_df['종목코드'].isin(eligible), '적합', '부적합')
        else:
            result_df['대회매수가능'] = '확인불가'
        result_df['_k'] = (result_df['구분'] != '신규신호')
        result_df['_e'] = (result_df['EMA20확실돌파'] != 'O')     # 확실 돌파가 위로
        result_df = result_df.sort_values(['_k', '_e', '신호일'], ascending=[True, True, False]).drop(columns=['_k', '_e'])
    else:
        print("조건에 맞는 신규 신호가 없습니다.")
        result_df = pd.DataFrame(columns=base_cols + ['섹터명', '시장 비중', '한도 비중', '대회매수가능'])

    # ---------- 2) 보유 종목 점검 ----------
    holding_df = None
    holdings = load_holdings()
    if holdings is not None:
        rows = []
        for _, hr in holdings.iterrows():
            try:
                rows.append(check_holding(hr, start, end))
            except Exception as e:
                print(f"  [보유종목 에러] {hr['종목명']}({hr['종목코드']}): {e}")
            time.sleep(SLEEP_SEC)
        if rows:
            holding_df = pd.DataFrame(rows)

    # ---------- 3) 저장 ----------
    try:
        with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
            for tf in ['일봉', '주봉']:
                part = result_df[result_df['봉'] == tf] if '봉' in result_df.columns else result_df.iloc[0:0]
                part.drop(columns='봉', errors='ignore').to_excel(writer, sheet_name=f'신규신호({tf})', index=False)
            if holding_df is not None:
                holding_df.to_excel(writer, sheet_name='보유종목', index=False)
    except PermissionError:
        print(f"\n[저장 실패] {OUTPUT_PATH} 가 엑셀에서 열려있어요. 닫고 다시 실행해주세요.")
        return

    n_d = int((result_df['봉'] == '일봉').sum()) if len(result_df) else 0
    n_w = int((result_df['봉'] == '주봉').sum()) if len(result_df) else 0
    print(f"\n완료! 신규신호 일봉 {n_d}건 / 주봉 {n_w}건"
          + (f", 보유종목 {len(holding_df)}건 점검" if holding_df is not None else "")
          + f" -> {OUTPUT_PATH} 저장됨")
    if len(result_df):
        print(result_df.head(20))
    if holding_df is not None:
        print("\n[보유종목 상태]")
        print(holding_df[['종목명', '현재가', '수익률(%)', '스무스HA', '상태']].to_string(index=False))


if __name__ == '__main__':
    main()