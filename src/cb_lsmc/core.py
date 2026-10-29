"""전환사채(CB) LSMC(Least Squares Monte Carlo) 평가 엔진.

왜 LSMC인가: 국내 CB는 (1) 투자자가 1년 뒤부터 언제든 전환할 수 있고(미국형), (2) 18개월 뒤부터 3개월마다
조기상환청구(풋)를 할 수 있으며, (3) 주가가 내려가면 전환가액을 낮추는 리픽싱이 있다.
(1)(2)는 트리로도 풀리지만 (3)은 전환가액이 지나온 경로에 따라 달라져 트리 노드가 재결합하지 않는다.
그래서 경로를 시뮬레이션하고, 각 행사 시점의 '계속 보유 가치'를 회귀로 추정하는 LSMC를 쓴다.

모형
- 주가: 위험중립 GBM, 주 단위(1년 52주) 정확한 이산화. drift = r - q
- 신용위험: Tsiveriotis-Fernandes(1998) 방식. 주식으로 받는 현금흐름(전환)은 무위험금리 r로,
  발행사가 현금으로 갚는 현금흐름(만기상환·풋)은 r + 신용스프레드 s로 할인한다.
- 회귀 기저: 패리티 x = 전환가치/액면, 리픽싱 상태 c = 현재 전환가액/최초 전환가액
  [1, x, x^2, x^3, c, c·x]
- 편향 줄이기: 회귀계수는 별도 경로(훈련용)로 구하고, 가격은 다른 난수 경로(평가용)로 계산한다(out-of-sample).

상품(가상의 예시, 액면 100 기준)
- 만기 3년, 표면이자 0%, 만기보장수익률 2%(3개월 복리) → 만기상환 100 x (1 + 0.02/4)^12
- 전환: 1년 뒤부터 만기 1개월 전까지 매주 가능. 전환주식수 = 100 / 전환가액
- 조기상환청구(풋): 1.5년 뒤부터 3개월마다, 조기상환수익률 2%(3개월 복리)
- 리픽싱: 3개월마다 주가가 전환가액보다 낮으면 그 주가로 낮춤. 최초 전환가액의 70%가 하한
- 발행사 콜(매도청구권), 희석 효과, 리픽싱 기준가의 가중평균 산식은 넣지 않았다(단순화).
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import norm

WEEKS = 52


@dataclass
class CB:
    T: float = 3.0
    ytm: float = 0.02               # 만기보장수익률 (3개월 복리)
    ytp: float = 0.02               # 조기상환수익률 (3개월 복리)
    conv_start: float = 1.0         # 전환 시작 (년)
    conv_end_gap: float = 1 / 12    # 만기 몇 년 전까지 전환 가능한지
    put_start: float = 1.5          # 조기상환청구 시작 (년), None이면 풋 없음
    put_every: float = 0.25
    refix: bool = True
    refix_every: float = 0.25
    refix_floor: float = 0.70
    premium: float = 0.0            # 최초 전환가액 = 발행 시 주가 x (1 + 할증률)

    def redemption(self):
        return 100 * (1 + self.ytm / 4) ** (4 * self.T)

    def put_price(self, t):
        return 100 * (1 + self.ytp / 4) ** (4 * t)

    def steps(self):
        return int(round(self.T * WEEKS))

    def conv_steps(self):
        a = int(round(self.conv_start * WEEKS)); b = int(round((self.T - self.conv_end_gap) * WEEKS))
        return set(range(a, b + 1))

    def put_steps(self):
        if self.put_start is None:
            return {}
        out, t = {}, self.put_start
        while t < self.T - 1e-9:
            out[int(round(t * WEEKS))] = self.put_price(t)
            t += self.put_every
        return out

    def refix_steps(self):
        if not self.refix:
            return set()
        n = int(round(self.refix_every * WEEKS))
        return set(range(n, self.steps(), n))


@dataclass
class Mkt:
    s0: float          # 평가 시점 주가 / 발행 시점 주가 (신규 발행이면 1)
    sigma: float
    r: float
    s: float           # 신용스프레드
    q: float = 0.0


def paths(prod, mkt, n, seed):
    """주 단위 GBM 경로 (n x (steps+1)), 발행 시점 주가 = 1 기준. 대조변수: 앞 절반과 뒤 절반이 쌍."""
    N = prod.steps(); dt = 1 / WEEKS
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n // 2, N))
    z = np.concatenate([z, -z])
    inc = (mkt.r - mkt.q - 0.5 * mkt.sigma ** 2) * dt + mkt.sigma * np.sqrt(dt) * z
    S = np.empty((len(z), N + 1))
    S[:, 0] = mkt.s0
    S[:, 1:] = mkt.s0 * np.exp(np.cumsum(inc, axis=1))
    return S


def conv_price_path(prod, S):
    """리픽싱을 반영한 전환가액 경로 (발행 시점 주가 = 1 기준)."""
    cp0 = 1.0 + prod.premium
    CP = np.full(S.shape, cp0)
    rf = prod.refix_steps()
    cur = np.full(S.shape[0], cp0)
    for k in range(1, S.shape[1]):
        if k in rf:
            cur = np.maximum(np.minimum(cur, S[:, k]), prod.refix_floor * cp0)
        CP[:, k] = cur
    return CP


BASIS = 'euro'


def euro_value(x, tau, prod, mkt):
    """남은 기간 tau 동안 만기에만 전환 가능하다고 볼 때의 CB 가치(해석해). 회귀 기저로 쓴다."""
    tau = max(tau, 1e-6)
    R = prod.redemption()
    S = 100 * np.maximum(x, 1e-12)
    d1 = (np.log(S / R) + (mkt.r - mkt.q + 0.5 * mkt.sigma ** 2) * tau) / (mkt.sigma * np.sqrt(tau))
    d2 = d1 - mkt.sigma * np.sqrt(tau)
    return S * np.exp(-mkt.q * tau) * norm.cdf(d1) + R * np.exp(-(mkt.r + mkt.s) * tau) * norm.cdf(-d2)


def basis(x, c, tau=None, prod=None, mkt=None):
    """'cubic': 다항식만 / 'euro': 다항식 + 같은 상태의 유럽형 CB 해석해(계속 보유 가치의 모양을 잘 따라감)."""
    cols = [np.ones_like(x), x, x * x, x ** 3, c, c * x]
    if BASIS == 'euro':
        e = euro_value(x, tau, prod, mkt) / 100
        cols += [e, e * e, e * x]
    return np.column_stack(cols)


def lsmc(prod, mkt, S, betas=None):
    """역방향 LSMC. betas가 없으면 이 경로로 회귀계수를 추정해 돌려준다(훈련), 있으면 그 계수로 결정만 한다(평가).
    돌려주는 값: (경로별 할인 현금흐름, 회귀계수 dict, 경로별 행사 정보)"""
    N = prod.steps(); dt = 1 / WEEKS
    CP = conv_price_path(prod, S)
    cp0 = 1.0 + prod.premium
    CV = 100 * S / CP                                     # 전환가치
    conv_ok, puts = prod.conv_steps(), prod.put_steps()

    R = prod.redemption()
    cf = np.where(CV[:, N] >= R, CV[:, N], R) if N in conv_ok else np.full(S.shape[0], R)
    eq = (CV[:, N] >= R) if N in conv_ok else np.zeros(S.shape[0], bool)   # True = 주식(전환)으로 받음
    tk = np.full(S.shape[0], N)                                              # 현금흐름 발생 시점(주)
    kind = np.where(eq, 1, 0)                                                # 0 만기상환, 1 전환, 2 풋
    fit = {} if betas is None else betas

    for k in range(N - 1, 0, -1):
        c_ok, p_ok = k in conv_ok, k in puts
        if not (c_ok or p_ok):
            continue
        tau = (tk - k) * dt
        cont = cf * np.exp(-(mkt.r + mkt.s * (~eq)) * tau)                   # TF 할인: 전환분은 r, 현금분은 r+s
        cv = CV[:, k]
        pp = puts[k] if p_ok else -np.inf
        ev = np.maximum(cv if c_ok else -np.inf, pp)
        X = basis(cv / 100, CP[:, k] / cp0, (N - k) * dt, prod, mkt)
        if betas is None:
            fit[k] = np.linalg.lstsq(X, cont, rcond=None)[0]
        ex = ev > X @ fit[k]
        to_eq = ex & c_ok & (cv >= pp)
        to_put = ex & ~to_eq
        cf = np.where(ex, ev, cf); tk = np.where(ex, k, tk)
        eq = np.where(to_eq, True, np.where(to_put, False, eq))
        kind = np.where(to_eq, 1, np.where(to_put, 2, kind))

    v = cf * np.exp(-(mkt.r + mkt.s * (~eq)) * tk * dt)
    return v, fit, {'k': tk, 'kind': kind}


def pair_se(v):
    h = len(v) // 2
    p = 0.5 * (v[:h] + v[h:2 * h])
    return p.std(ddof=1) / np.sqrt(len(p))


def price(prod, mkt, n=200_000, seed=1, train_seed=101, n_train=100_000):
    """out-of-sample 가격: 훈련 경로로 회귀계수 → 평가 경로로 가격. in-sample 가격도 함께 돌려준다."""
    St = paths(prod, mkt, n_train, train_seed)
    v_in, betas, _ = lsmc(prod, mkt, St)
    S = paths(prod, mkt, n, seed)
    v, _, info = lsmc(prod, mkt, S, betas)
    dt = 1 / WEEKS
    kinds = info['kind']
    return {'price': float(v.mean()), 'se': float(pair_se(v)), 'in_sample': float(v_in.mean()),
            'p_conv': float((kinds == 1).mean()), 'p_put': float((kinds == 2).mean()), 'p_mat': float((kinds == 0).mean()),
            'avg_conv_year': float((info['k'][kinds == 1] * dt).mean()) if (kinds == 1).any() else None,
            'avg_put_year': float((info['k'][kinds == 2] * dt).mean()) if (kinds == 2).any() else None}


# ---------------------------------------------------------------- 검증 1: 만기에만 전환하는 CB의 해석해
def european_cb(prod, mkt):
    """전환이 만기에만 가능하고 풋·리픽싱이 없으면 TF 할인 아래 해석해가 있다.
    V = n·S0·e^{-qT}·N(d1) + R·e^{-(r+s)T}·N(-d2),  n = 100/CP, 행사가 K = R/n"""
    T, R = prod.T, prod.redemption()
    n_sh = 100 / (1 + prod.premium)
    K = R / n_sh
    d1 = (np.log(mkt.s0 / K) + (mkt.r - mkt.q + 0.5 * mkt.sigma ** 2) * T) / (mkt.sigma * np.sqrt(T))
    d2 = d1 - mkt.sigma * np.sqrt(T)
    return float(n_sh * mkt.s0 * np.exp(-mkt.q * T) * norm.cdf(d1) + R * np.exp(-(mkt.r + mkt.s) * T) * norm.cdf(-d2))


# ---------------------------------------------------------------- 검증 2·3: 이항트리 (재결합, 리픽싱 없음)
def tree_cb(prod, mkt, m=10):
    """TF 이항트리(CRR). 한 주를 m스텝으로 나누고 행사는 주 단위 시점에만 허용(LSMC와 같은 행사 시점).
    노드마다 전체 가치 V와 현금 부분 B를 함께 들고 간다: B는 r+s, V-B는 r로 할인."""
    N = prod.steps() * m; dt = 1 / WEEKS / m
    u = np.exp(mkt.sigma * np.sqrt(dt)); d = 1 / u
    p = (np.exp((mkt.r - mkt.q) * dt) - d) / (u - d)
    j = np.arange(N + 1)
    S = mkt.s0 * u ** (N - 2 * j)
    n_sh = 100 / (1 + prod.premium)
    conv_ok, puts = prod.conv_steps(), prod.put_steps()
    R = prod.redemption()
    cv = n_sh * S
    if prod.steps() in conv_ok:
        V = np.maximum(cv, R); B = np.where(cv >= R, 0.0, R)
    else:
        V = np.full(N + 1, R); B = V.copy()
    da, dc = np.exp(-(mkt.r + mkt.s) * dt), np.exp(-mkt.r * dt)
    for i in range(N - 1, -1, -1):
        Bn = da * (p * B[:-1] + (1 - p) * B[1:])
        En = dc * (p * (V - B)[:-1] + (1 - p) * (V - B)[1:])
        V, B = Bn + En, Bn
        if i % m == 0 and i > 0:
            k = i // m
            S = mkt.s0 * u ** (i - 2 * np.arange(i + 1))
            if k in puts:
                hit = puts[k] > V
                V = np.where(hit, puts[k], V); B = np.where(hit, puts[k], B)
            if k in conv_ok:
                cvk = n_sh * S
                hit = cvk >= V
                V = np.where(hit, cvk, V); B = np.where(hit, 0.0, B)
    return float(V[0])


def lsmc_put(mkt, K, T, n, seed, train_seed=101, n_train=100_000):
    """주식 미국형(주 단위 버뮤단) 풋 LSMC: Longstaff-Schwartz(2001)의 기본 예제로 엔진의 회귀·행사 로직을 따로 검증."""
    prod = CB(T=T)
    dt = 1 / WEEKS

    def run(S, betas):
        N = S.shape[1] - 1
        cf = np.maximum(K - S[:, N], 0); tk = np.full(S.shape[0], N)
        fit = {} if betas is None else betas
        for k in range(N - 1, 0, -1):
            ex_v = np.maximum(K - S[:, k], 0)
            itm = ex_v > 0
            if not itm.any():
                continue
            cont = cf * np.exp(-mkt.r * (tk - k) * dt)
            x = S[:, k] / K
            X = np.column_stack([np.ones_like(x), x, x * x, x ** 3])
            if betas is None:
                fit[k] = np.linalg.lstsq(X[itm], cont[itm], rcond=None)[0]
            ex = itm & (ex_v > X @ fit[k])
            cf = np.where(ex, ex_v, cf); tk = np.where(ex, k, tk)
        return cf * np.exp(-mkt.r * tk * dt), fit

    _, betas = run(paths(prod, mkt, n_train, train_seed), None)
    v, _ = run(paths(prod, mkt, n, seed), betas)
    return float(v.mean()), float(pair_se(v))


def tree_put(mkt, K, T, m=10):
    N = int(round(T * WEEKS)) * m; dt = 1 / WEEKS / m
    u = np.exp(mkt.sigma * np.sqrt(dt)); d = 1 / u
    p = (np.exp((mkt.r - mkt.q) * dt) - d) / (u - d)
    disc = np.exp(-mkt.r * dt)
    V = np.maximum(K - mkt.s0 * u ** (N - 2 * np.arange(N + 1)), 0)
    for i in range(N - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if i % m == 0 and i > 0:
            V = np.maximum(V, K - mkt.s0 * u ** (i - 2 * np.arange(i + 1)))
    return float(V[0])


def bs_put(s, k, T, r, q, sigma):
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(k * np.exp(-r * T) * norm.cdf(-d2) - s * np.exp(-q * T) * norm.cdf(-d1))


def bump(x, **kw):
    return replace(x, **kw)
