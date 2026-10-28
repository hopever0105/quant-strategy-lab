"""2자산 스텝다운 ELS 몬테카를로 평가 엔진.

모형: 위험중립 하에서 두 기초자산이 상관된 기하 브라운 운동(Black-Scholes)을 따른다.
- KOSPI200 (원화 자산): drift = r_krw - q
- S&P500 (달러 자산, 원화 지급 = 콴토): drift = r_usd - q - rho(SPX, 원/달러) * sigma_SPX * sigma_FX
할인은 원화 무위험금리로 한다. 낙인은 일별 종가로 관찰한다(1년 252일).

상품 구조(가상의 예시 상품):
- 6개월마다 조기상환 평가. 두 지수 중 낮은 쪽 성과(worst-of)가 그 회차 배리어 이상이면 원금 + 연 c x 경과기간 지급 후 종료
- 만기에 상환되지 않았으면: 만기 배리어 이상이거나 기간 중 낙인이 한 번도 없었으면 원금 + 연 c x 3년,
  낙인이 있었고 만기 배리어 미만이면 worst-of 성과만큼 지급(손실)

payoff가 쿠폰 c에 선형(= P + c x Y)이라 한 번의 시뮬레이션으로 공정 쿠폰(가격 = 액면)을 바로 푼다.
"""
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.stats import norm

DAYS = 252


@dataclass
class Product:
    T: float = 3.0
    obs_years: tuple = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    barriers: tuple = (0.90, 0.90, 0.85, 0.85, 0.80, 0.75)
    ki: float = 0.50
    coupon: float = 0.07          # 연 쿠폰 (가격 계산용, 공정 쿠폰은 별도로 푼다)


@dataclass
class Market:
    sigma: np.ndarray             # [KS200, SPX] 연 변동성
    rho: float                    # KS200-SPX 상관
    r_krw: float
    r_usd: float
    q: np.ndarray                 # [KS200, SPX] 배당수익률 (가정값)
    sigma_fx: float               # 원/달러 연 변동성
    rho_spx_fx: float             # SPX-원/달러 상관 (콴토 조정)
    s0: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))   # 기준가 대비 현재 수준

    def drift(self):
        quanto = self.rho_spx_fx * self.sigma[1] * self.sigma_fx
        return np.array([self.r_krw - self.q[0], self.r_usd - self.q[1] - quanto])


def simulate(prod, mkt, n_paths, seed=0, batch=10_000):
    """경로를 배치로 생성해 (P, Y, 상환 회차, 낙인 여부, 만기 worst) 를 돌려준다. 대조변수(antithetic) 사용."""
    steps = int(round(prod.T * DAYS))
    dt = 1.0 / DAYS
    obs_idx = [int(round(t * DAYS)) for t in prod.obs_years]
    mu = mkt.drift()
    sig = np.asarray(mkt.sigma)
    L = np.linalg.cholesky(np.array([[1.0, mkt.rho], [mkt.rho, 1.0]]))
    inc_mean = (mu - 0.5 * sig ** 2) * dt
    rng = np.random.default_rng(seed)

    P_all, Y_all, k_all, ki_all, wT_all = [], [], [], [], []
    done = 0
    while done < n_paths:
        half = min(batch, n_paths - done) // 2
        z = rng.standard_normal((half, steps, 2)) @ L.T
        z = np.concatenate([z, -z])                           # 대조변수
        logret = np.cumsum(inc_mean + sig * np.sqrt(dt) * z, axis=1)
        perf = np.asarray(mkt.s0) * np.exp(logret)            # 기준가 대비 성과
        worst = perf.min(axis=2)                              # (경로, 일)
        m = worst.shape[0]

        P = np.zeros(m); Y = np.zeros(m)
        k_red = np.full(m, -1)                                # 조기·만기 쿠폰상환 회차(0~5), -1 = 미상환
        alive = np.ones(m, bool)
        for k, (ix, b, t) in enumerate(zip(obs_idx, prod.barriers, prod.obs_years)):
            hit = alive & (worst[:, ix - 1] >= b)
            P[hit] = 1.0; Y[hit] = t; k_red[hit] = k
            alive &= ~hit
        ki = (worst.min(axis=1) < prod.ki) | (mkt.s0.min() < prod.ki)
        wT = worst[:, -1]
        safe = alive & ~ki                                    # 낙인 없음 → 원금 + 전체 쿠폰
        P[safe] = 1.0; Y[safe] = prod.T; k_red[safe] = len(obs_idx) - 1
        loss = alive & ki
        P[loss] = wT[loss]; Y[loss] = 0.0
        P_all.append(P); Y_all.append(Y); k_all.append(k_red); ki_all.append(ki); wT_all.append(wT)
        done += m
    return (np.concatenate(P_all), np.concatenate(Y_all), np.concatenate(k_all),
            np.concatenate(ki_all), np.concatenate(wT_all))


def price(prod, mkt, n_paths=100_000, seed=0, coupon=None):
    """가격(액면 1 기준), 표준오차, 공정 쿠폰, 상환 확률 분포."""
    P, Y, k_red, ki, wT = simulate(prod, mkt, n_paths, seed)
    times = np.array(prod.obs_years)
    t_pay = np.where(k_red >= 0, times[np.clip(k_red, 0, None)], prod.T)
    df = np.exp(-mkt.r_krw * t_pay)
    c = prod.coupon if coupon is None else coupon
    # 대조변수 쌍 평균으로 표준오차 계산 (쌍은 배치마다 앞 절반/뒤 절반)
    v = df * (P + c * Y)
    A, B = (df * P).mean(), (df * Y).mean()
    se = pair_se(v)
    loss = (k_red < 0)
    return {
        'price': float(v.mean()), 'se': float(se),
        'fair_coupon': float((1.0 - A) / B), 'A': float(A), 'B': float(B),
        'p_redeem': [float((k_red == k).mean()) for k in range(len(times))],
        'p_ki': float(ki.mean()), 'p_loss': float(loss.mean()),
        'avg_loss_given_loss': float(1 - wT[loss].mean()) if loss.any() else 0.0,
    }


def pair_se(v, batch=10_000):
    """대조변수 쌍을 하나의 관측으로 묶어 표준오차를 낸다(쌍끼리 음의 상관이라 그냥 표준편차를 쓰면 과대추정)."""
    out, i = [], 0
    while i < len(v):
        b = v[i:i + batch]
        h = len(b) // 2
        out.append(0.5 * (b[:h] + b[h:2 * h]))
        i += batch
    pairs = np.concatenate(out)
    return pairs.std(ddof=1) / np.sqrt(len(pairs))


# ---------------------------------------------------------------- 검증용 해석해
def bs_call(s, k, T, r, q, sigma):
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return s * np.exp(-q * T) * norm.cdf(d1) - k * np.exp(-r * T) * norm.cdf(d2)


def mc_check(mkt, T=3.0, n_paths=100_000, seed=1, strike=1.0):
    """같은 경로 생성기로 (1) 두 자산의 선도가격, (2) 각 자산 콜옵션을 해석해와 대조한다."""
    steps = int(round(T * DAYS)); dt = 1.0 / DAYS
    mu = mkt.drift(); sig = np.asarray(mkt.sigma)
    L = np.linalg.cholesky(np.array([[1.0, mkt.rho], [mkt.rho, 1.0]]))
    rng = np.random.default_rng(seed)
    ST = []
    done = 0
    while done < n_paths:
        half = min(10_000, n_paths - done) // 2
        z = rng.standard_normal((half, steps, 2)) @ L.T
        z = np.concatenate([z, -z])
        ST.append(np.exp(np.sum((mu - 0.5 * sig ** 2) * dt + sig * np.sqrt(dt) * z, axis=1)))
        done += 2 * half
    ST = np.concatenate(ST)
    df = np.exp(-mkt.r_krw * T)
    rows = []
    for i, name in enumerate(['KOSPI200', 'S&P500(콴토)']):
        q_eff = mkt.r_krw - mu[i]                                  # 원화 측도에서의 실효 배당률
        fwd_mc, fwd_se = ST[:, i].mean(), pair_se(ST[:, i])
        call = df * np.maximum(ST[:, i] - strike, 0)
        rows.append({'자산': name,
                     '선도 해석해': float(np.exp(mu[i] * T)), '선도 MC': float(fwd_mc), '선도 SE': float(fwd_se),
                     '콜 해석해': float(bs_call(1.0, strike, T, mkt.r_krw, q_eff, sig[i])),
                     '콜 MC': float(call.mean()), '콜 SE': float(pair_se(call))})
    return rows


def bump(mkt, **kw):
    return replace(mkt, **kw)
