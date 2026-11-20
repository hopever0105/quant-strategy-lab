# -*- coding: utf-8 -*-
"""쿨라마기 EMA 돌파 - 지표와 셋업 조건 (t일 종가까지 아는 값만)

셋업(t일 종가 기준, 진입은 t+1일):
  ① 유동성: 20일 평균 거래대금 ≥ 10억 원, 종가 ≥ 1,000원, 상장 후 63거래일 이상
  ② 모멘텀: 63일 수익률이 그날 유동성 통과 종목 중 상위 top% 이고 +30% 이상
  ③ ADR: 20일 평균 (고가÷저가 − 1) ≥ adr_min
  ④ 횡보 압축: 1 − 종가 ÷ (최근 n일 최고가) ≤ 2.5 × ADR, 종가 > EMA10 > EMA20 > EMA50
  ⑤ 시장 국면: 코스피 종가 > 코스피 EMA20
  (변형) 엄격한 횡보: 최근 10일 (최고가÷최저가 − 1) ≤ 2 × ADR 이고, 20일 최고가가 3거래일 이상 전에 나옴(고점 뒤 쉬는 중)
돌파 기준가 = 최근 n일 최고가(t일 포함). t+1일 고가가 이 값을 넘으면 매수.
"""
import numpy as np
import pandas as pd

MIN_AMT = 1e9
MIN_PX = 1000.0
MOM_N = 63
MOM_MIN = 0.30
ADR_N = 20
TIGHT_K = 2.5


def ema(a, span):
    """열별 EMA (NaN 구간은 이어서 계산). a: numpy T×N"""
    return pd.DataFrame(a).ewm(span=span, adjust=False, ignore_na=True).mean().to_numpy()


class Ind:
    """파라미터와 무관한 지표를 한 번만 계산"""

    def __init__(self, D):
        O, H, L, C, A = D["o"], D["h"], D["l"], D["c"], D["amt"]
        self.D = D
        Cdf = pd.DataFrame(C)
        valid = ~np.isnan(C)
        age = np.cumsum(valid, axis=0)
        amt20 = pd.DataFrame(np.where(valid, np.nan_to_num(A), np.nan)).rolling(20, min_periods=15).mean().to_numpy()
        self.liq = (amt20 >= MIN_AMT) & (C >= MIN_PX) & (age > MOM_N) & valid
        self.amt20 = amt20
        Cff = Cdf.ffill().to_numpy()
        self.mom = Cff / pd.DataFrame(Cff).shift(MOM_N).to_numpy() - 1
        momq = np.where(self.liq, self.mom, np.nan)
        self.mom_pct = pd.DataFrame(momq).rank(axis=1, pct=True, ascending=True).to_numpy()  # 1 = 최상위
        dr = np.where(valid, H / L - 1, np.nan)
        self.adr = pd.DataFrame(dr).rolling(ADR_N, min_periods=15).mean().to_numpy()
        self.e10, self.e20, self.e50 = ema(C, 10), ema(C, 20), ema(C, 50)
        self.ema = {10: self.e10, 20: self.e20}
        self.align = (C > self.e10) & (self.e10 > self.e20) & (self.e20 > self.e50)
        self.hh = {n: pd.DataFrame(H).rolling(n, min_periods=n).max().to_numpy() for n in (10, 20, 40)}
        ll10 = pd.DataFrame(L).rolling(10, min_periods=10).min().to_numpy()
        hh3 = pd.DataFrame(H).rolling(3, min_periods=3).max().to_numpy()
        self.strict = ((self.hh[10] / ll10 - 1) <= 2 * self.adr) & (hh3 < self.hh[20])
        ks = D["kospi"]
        self.regime = (ks > ks.ewm(span=20, adjust=False).mean()).to_numpy()
        self.C = C

    def setup(self, adr_min=0.04, top=0.05, n=20, regime=True, use_align=True, use_tight=True, use_mom=True, use_adr=True, strict=False):
        s = self.liq.copy()
        if use_mom:
            s &= (self.mom_pct >= 1 - top) & (self.mom >= MOM_MIN)
        if use_adr:
            s &= self.adr >= adr_min
        if use_align:
            s &= self.align
        if use_tight:
            s &= (1 - self.C / self.hh[n]) <= TIGHT_K * self.adr
        if strict:
            s &= self.strict
        if regime:
            s &= self.regime[:, None]
        return np.nan_to_num(s, nan=0).astype(np.bool_)
