"""수정주가 공용 도구 (/quant, /index 공용).

Yahoo(yfinance) 계열 시세는 '종가'가 액면분할·스핀오프 보정가, '수정종가'가 여기에 배당까지 보정한 값이다.
DB(us_price)는 처음 받은 날짜 기준으로 보정돼 있어서, 그 뒤에 생긴 분할은 과거 행에 반영되지 않아 가격이 끊길 수 있다.

- fix_split_gaps : 분할·스핀오프일에 가격이 1/배율만큼 뚝 떨어진(=과거 행이 보정 안 된) 구간을 찾아 과거 행을 나눠 맞춘다.
- raw_price      : 보정 종가 × (그 날 이후 분할·스핀오프 배율의 곱) = 그 날 실제 거래 가격 (가격가중 지수용)
- div_yield      : 종가·수정종가 비율로 배당락일의 배당수익률(D / 전일 종가)을 역산
"""
import numpy as np
import pandas as pd


def fix_split_gaps(df, splits, close='close', adj='adj', tol=0.12):
    """df: 날짜 인덱스, close/adj 열. splits: 날짜→배율(예: 4:1 분할이면 4.0, 스핀오프는 1.2 같은 소수).
    분할일 수익률이 1/배율에 가까우면(=과거 행 미보정) 그 이전 행을 배율로 나눈다. 반환: (보정된 df, 보정 건수, 보정 내역)"""
    df = df.copy()
    fixed = []
    for d, f in splits.sort_index().items():
        if abs(f - 1) < 0.15 or d not in df.index:
            continue
        i = df.index.get_loc(d)
        if i == 0:
            continue
        ret = df[close].iloc[i] / df[close].iloc[i - 1]
        if abs(ret * f - 1) < tol:          # 실제 가격이 1/f로 떨어짐 → 과거 행이 보정 안 됨
            df.iloc[:i, df.columns.get_loc(close)] /= f
            df.iloc[:i, df.columns.get_loc(adj)] /= f
            fixed.append((str(pd.Timestamp(d).date()), float(f), float(ret)))
    return df, len(fixed), fixed


def raw_price(close_adj, splits):
    """보정 종가 → 실제 거래 가격. 각 날짜 t에 대해 t보다 뒤에 있는 분할·스핀오프 배율을 모두 곱한다."""
    fac = pd.Series(1.0, index=close_adj.index)
    for d, f in splits.items():
        fac[fac.index < d] *= f
    return close_adj * fac


def div_yield(close_adj, adj_close):
    """배당락일 배당수익률 D/P(t-1). 수정종가 비율 = 종가 비율 / (1 - D/P) 를 이용한다."""
    cr = close_adj / close_adj.shift(1)
    ar = adj_close / adj_close.shift(1)
    d = 1 - cr / ar
    d = d.where(d > 1e-5, 0.0).clip(upper=0.2)
    return d.fillna(0.0)


def detect_jumps(close_adj, thresh=0.35):
    """보정 후에도 남은 하루 ±35% 이상 단절(데이터 오류 의심) 목록"""
    r = close_adj.pct_change()
    return r[(r.abs() > thresh)].dropna()
