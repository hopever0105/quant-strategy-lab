"""2자산 스텝다운 ELS 평가 실행: 파라미터 산출 → 엔진 검증 → 공정 쿠폰·민감도 → 과거 실제 경로 점검 → 그림.

먼저 python data.py 로 data/ 를 채운 뒤 실행.
실행:  python run.py
결과:  results/results.json, results/fig_*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import Product, Market, price, mc_check, bump

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
RES.mkdir(exist_ok=True)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

Q_ASSUMED = np.array([0.015, 0.012])   # 배당수익률: 가정값(확인 필요). 민감도로 영향 크기를 따로 보인다.
N_MAIN = 200_000


# ---------------------------------------------------------------- 1. 파라미터 산출
def params():
    m = pd.read_csv(HERE / 'data' / 'market.csv', index_col=0, parse_dates=True)
    rt = json.loads((HERE / 'data' / 'rates.json').read_text(encoding='utf-8'))
    val_date = min(m['KS200'].dropna().index.max(), m['SPX'].dropna().index.max())
    m = m.loc[:val_date]
    start = val_date - pd.DateOffset(years=1)

    def daily_vol(s):
        r = np.log(s.dropna()).diff().dropna()
        r = r[r.index > start]
        return float(r.std(ddof=1) * np.sqrt(252)), len(r)

    # 상관은 주간 수익률로: 한국·미국 장 마감 시각이 달라 일간 수익률 상관은 실제보다 낮게 나온다(비동시성)
    wk = m.ffill().resample('W-FRI').last()
    wr = np.log(wk).diff().dropna()
    wr = wr[wr.index > start]
    d_corr = np.log(m.dropna()).diff().dropna()
    d_corr = d_corr[d_corr.index > start]

    s_ks, n_ks = daily_vol(m['KS200']); s_sp, n_sp = daily_vol(m['SPX']); s_fx, n_fx = daily_vol(m['USDKRW'])
    info = {
        'valuation_date': str(val_date.date()),
        'window': f'{(start + pd.Timedelta(days=1)).date()} ~ {val_date.date()}',
        'sigma_ks200': s_ks, 'sigma_spx': s_sp, 'sigma_fx': s_fx, 'n_daily': [n_ks, n_sp, n_fx],
        'rho_weekly': float(wr['KS200'].corr(wr['SPX'])),
        'rho_daily': float(d_corr['KS200'].corr(d_corr['SPX'])),
        'rho_spx_fx_weekly': float(wr['SPX'].corr(wr['USDKRW'])),
        'n_weekly': int(len(wr)),
        'r_krw': rt['r_krw_3y'], 'r_krw_src': f"국고채 3년 (ECOS, {rt['r_krw_date']})",
        'r_usd': rt['r_usd_3y'], 'r_usd_src': f"미 국채 3년 (FRED DGS3, {rt['r_usd_date']})",
        'q_assumed': Q_ASSUMED.tolist(),
        'last': {c: float(m[c].dropna().iloc[-1]) for c in m.columns},
    }
    mkt = Market(sigma=np.array([s_ks, s_sp]), rho=info['rho_weekly'], r_krw=info['r_krw'], r_usd=info['r_usd'],
                 q=Q_ASSUMED, sigma_fx=s_fx, rho_spx_fx=info['rho_spx_fx_weekly'])

    # 장기 시나리오: KOSPI200 데이터가 있는 2014-08 이후 전체 기간으로 같은 방식 산출 (최근 1년이 극단적 변동 구간이라 비교용)
    lstart = m['KS200'].dropna().index.min()

    def lvol(s):
        r = np.log(s.dropna()).diff().dropna()
        return float(r[r.index > lstart].std(ddof=1) * np.sqrt(252))

    lwr = np.log(wk).diff().dropna()
    lwr = lwr[lwr.index > lstart]
    info['long'] = {'window': f'{lstart.date()} ~ {val_date.date()}',
                    'sigma_ks200': lvol(m['KS200']), 'sigma_spx': lvol(m['SPX']), 'sigma_fx': lvol(m['USDKRW']),
                    'rho_weekly': float(lwr['KS200'].corr(lwr['SPX'])),
                    'rho_spx_fx_weekly': float(lwr['SPX'].corr(lwr['USDKRW']))}
    L = info['long']
    mkt_long = Market(sigma=np.array([L['sigma_ks200'], L['sigma_spx']]), rho=L['rho_weekly'], r_krw=info['r_krw'],
                      r_usd=info['r_usd'], q=Q_ASSUMED, sigma_fx=L['sigma_fx'], rho_spx_fx=L['rho_spx_fx_weekly'])
    return mkt, mkt_long, info, m


# ---------------------------------------------------------------- 2. 엔진 검증
def convergence(prod, mkt, c):
    rows = []
    for n in (2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 400_000):
        r = price(prod, mkt, n, seed=11, coupon=c)
        rows.append({'n': n, 'price': r['price'], 'se': r['se']})
    return rows


def greeks(prod, mkt, c, n=N_MAIN):
    """같은 난수(seed 고정)로 입력만 바꿔 재평가: 민감도의 몬테카를로 잡음을 줄인다."""
    base = price(prod, mkt, n, seed=7, coupon=c)['price']
    out = {'base': base}
    for i, name in enumerate(['KOSPI200', 'S&P500']):
        up = mkt.s0.copy(); up[i] *= 1.01
        dn = mkt.s0.copy(); dn[i] *= 0.99
        vu = price(prod, bump(mkt, s0=up), n, seed=7, coupon=c)['price']
        vd = price(prod, bump(mkt, s0=dn), n, seed=7, coupon=c)['price']
        sg = mkt.sigma.copy(); sg[i] += 0.01
        vv = price(prod, bump(mkt, sigma=sg), n, seed=7, coupon=c)['price']
        out[name] = {'delta_1pct': (vu - vd) / 2, 'gamma_1pct': vu - 2 * base + vd, 'vega_1pp': vv - base}
    vr_u = price(prod, bump(mkt, rho=mkt.rho + 0.05), n, seed=7, coupon=c)['price']
    vr_d = price(prod, bump(mkt, rho=mkt.rho - 0.05), n, seed=7, coupon=c)['price']
    out['corr_0p05'] = (vr_u - vr_d) / 2
    sg = mkt.sigma + 0.01
    out['vega_both_1pp'] = price(prod, bump(mkt, sigma=sg), n, seed=7, coupon=c)['price'] - base
    out['rate_1bp'] = price(prod, bump(mkt, r_krw=mkt.r_krw + 0.0001), n, seed=7, coupon=c)['price'] - base
    return out


def sensitivities(prod, mkt, n=50_000):
    grid = {}
    grid['sigma_scale'] = [(s, price(prod, bump(mkt, sigma=mkt.sigma * s), n, seed=3)['fair_coupon'])
                           for s in (0.8, 0.9, 1.0, 1.1, 1.2)]
    grid['rho'] = [(r, price(prod, bump(mkt, rho=r), n, seed=3)['fair_coupon']) for r in (0.0, 0.2, 0.4, 0.6, 0.8)]
    grid['ki'] = [(k, price(Product(ki=k), mkt, n, seed=3)['fair_coupon']) for k in (0.40, 0.45, 0.50, 0.55, 0.60)]
    grid['q_ks200'] = [(q, price(prod, bump(mkt, q=np.array([q, mkt.q[1]])), n, seed=3)['fair_coupon'])
                       for q in (0.0, 0.01, 0.015, 0.02, 0.03)]
    return grid


# ---------------------------------------------------------------- 3. 과거 실제 경로 점검
def historical(prod, m, c):
    """매 거래일에 같은 상품을 발행했다고 보고 실제 KOSPI200·S&P500 종가로 결과를 낸다(환율 효과 없음, 겹치는 표본)."""
    px = m[['KS200', 'SPX']].copy()
    px['SPX'] = px['SPX'].ffill()
    px = px[px['KS200'].notna()].dropna()
    dates = px.index
    vals = px.values
    out = []
    for i, d0 in enumerate(dates):
        obs = [d0 + pd.DateOffset(months=int(round(t * 12))) for t in prod.obs_years]
        if obs[-1] > dates[-1]:
            break
        j_obs = [dates.searchsorted(o) for o in obs]
        perf = vals[i:j_obs[-1] + 1] / vals[i]
        worst = perf.min(axis=1)
        res = None
        for k, (j, b, t) in enumerate(zip(j_obs, prod.barriers, prod.obs_years)):
            if worst[j - i] >= b:
                res = (k, 1 + c * t, t); break
        if res is None:
            ki = worst.min() < prod.ki
            res = (5, 1 + c * prod.T, prod.T) if not ki else (-1, worst[-1], prod.T)
        k, pay, t = res
        out.append({'start': d0, 'k': k, 'payoff': pay, 'years': t, 'ann_ret': pay ** (1 / t) - 1})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 4. 그림
def figures(conv, sens, base, base_long, hist, prod):
    # 수렴
    df = pd.DataFrame(conv)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.errorbar(df['n'], df['price'] * 100, yerr=1.96 * df['se'] * 100, fmt='o-', color='#1F3A5F', capsize=3, lw=1.2)
    ax.axhline(conv[-1]['price'] * 100, color='gray', ls='--', lw=0.8)
    ax.set_xscale('log'); ax.set_xlabel('경로 수 (로그 축)'); ax.set_ylabel('가격 (액면 대비 %)')
    ax.set_title('경로 수에 따른 가격 수렴 (95% 신뢰구간)', fontsize=11)
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(RES / 'fig_convergence.png', dpi=200); plt.close(fig)

    # 상환 시점 분포: 모형 vs 과거
    labels = ['6개월', '1년', '1.5년', '2년', '2.5년', '3년\n(만기 수익)', '손실']
    mc = base['p_redeem'] + [base['p_loss']]
    mcl = base_long['p_redeem'] + [base_long['p_loss']]
    hk = hist['k'].value_counts(normalize=True)
    hs = [float(hk.get(k, 0)) for k in range(6)] + [float(hk.get(-1, 0))]
    x = np.arange(len(labels)); w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(x - w, np.array(mc) * 100, w, label='몬테카를로: 최근 1년 변동성', color='#1F3A5F')
    ax.bar(x, np.array(mcl) * 100, w, label='몬테카를로: 2014년 이후 변동성', color='#4F77AA')
    ax.bar(x + w, np.array(hs) * 100, w, label=f'과거 실제 경로 ({len(hist):,}개 발행일)', color='#B4C7E7')
    ax.set_xticks(x, labels, fontsize=8.5); ax.set_ylabel('비율 (%)')
    ax.set_title('상환 시점 분포: 모형과 과거 실제 경로', fontsize=11)
    ax.legend(fontsize=8.5); ax.grid(axis='y', alpha=0.3); fig.tight_layout()
    fig.savefig(RES / 'fig_redeem.png', dpi=200); plt.close(fig)

    # 민감도: 공정 쿠폰
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for ax, key, xl in zip(axs, ('sigma_scale', 'rho', 'ki'), ('변동성 배수', 'KOSPI200-S&P500 상관', '낙인 배리어')):
        xs, ys = zip(*sens[key])
        ax.plot(xs, np.array(ys) * 100, 'o-', color='#1F3A5F', lw=1.2)
        ax.set_xlabel(xl, fontsize=8.5); ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
    axs[0].set_ylabel('공정 쿠폰 (연 %)', fontsize=8.5)
    fig.suptitle('입력값에 따른 공정 쿠폰 변화', fontsize=11); fig.tight_layout()
    fig.savefig(RES / 'fig_sens.png', dpi=200); plt.close(fig)


if __name__ == '__main__':
    mkt, mkt_long, info, m = params()
    prod = Product()
    print('파라미터', json.dumps(info, ensure_ascii=False, indent=1))

    chk = mc_check(mkt)
    base = price(prod, mkt, N_MAIN, seed=7)
    c = base['fair_coupon']
    base = price(prod, mkt, N_MAIN, seed=7, coupon=c)
    print('공정 쿠폰', round(c * 100, 3), '% / 가격', base['price'], 'SE', base['se'])
    base_long = price(prod, mkt_long, N_MAIN, seed=7)
    c_long = base_long['fair_coupon']
    base_long = price(prod, mkt_long, N_MAIN, seed=7, coupon=c_long)
    print('장기 변동성 공정 쿠폰', round(c_long * 100, 3), '%')
    # 기본 시나리오 = 장기 변동성(내재변동성을 구할 수 없어 대신 씀). 최근 1년 변동성(KOSPI200 60%)은 스트레스 시나리오로 둔다.
    conv = convergence(prod, mkt_long, c_long)
    grk = greeks(prod, mkt_long, c_long)
    sens = sensitivities(prod, mkt_long)
    hist = historical(prod, m, c_long)    # 과거 경로의 수익률은 장기 변동성 기준 공정 쿠폰으로 계산
    hist.to_csv(RES / 'historical.csv', index=False, encoding='utf-8-sig')
    hsum = {
        'n_starts': int(len(hist)), 'first': str(hist['start'].min().date()), 'last': str(hist['start'].max().date()),
        'p_k': {int(k): float(v) for k, v in hist['k'].value_counts(normalize=True).sort_index().items()},
        'p_loss': float((hist['k'] == -1).mean()),
        'avg_loss_given_loss': float(1 - hist.loc[hist['k'] == -1, 'payoff'].mean()) if (hist['k'] == -1).any() else 0.0,
        'loss_start_range': [str(hist.loc[hist['k'] == -1, 'start'].min()), str(hist.loc[hist['k'] == -1, 'start'].max())]
        if (hist['k'] == -1).any() else None,
        'mean_ann_ret': float(hist['ann_ret'].mean()),
    }
    figures(conv, sens, base, base_long, hist, prod)
    out = {'params': info, 'product': prod.__dict__, 'check': chk, 'base': base, 'fair_coupon': c,
           'base_long': base_long, 'fair_coupon_long': c_long,
           'convergence': conv, 'greeks': grk, 'sensitivity': sens, 'historical': hsum}
    (RES / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('check', 'base', 'base_long', 'greeks', 'sensitivity', 'historical')},
                     ensure_ascii=False, indent=1, default=str))
