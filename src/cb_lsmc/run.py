"""전환사채 LSMC 평가 실행: 파라미터 산출 → 엔진 검증(해석해·이항트리) → 가치 분해 → 민감도 → 그림.

먼저 python data.py 로 data/ 를 채운 뒤 실행.
실행:  python run.py
결과:  results/results.json, results/fig_*.png
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import core
from core import CB, Mkt, price, european_cb, tree_cb, lsmc_put, tree_put, bs_put, bump, paths, conv_price_path, WEEKS

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
RES.mkdir(exist_ok=True)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

DPS = 877               # JYP Ent. 2025 사업연도 주당 현금배당금(원), DART 사업보고서 '배당에 관한 사항'
DPS_SRC = 'DART 2025 사업보고서 주당 현금배당금 877원 ÷ 기준일 종가'
N_MAIN = 200_000
N_GRID = 100_000
NAVY, BLUE, LIGHT, ORANGE = '#1F3A5F', '#4F77AA', '#B4C7E7', '#C55A11'


# ---------------------------------------------------------------- 1. 파라미터 산출
def params():
    s = pd.read_csv(HERE / 'data' / 'stock.csv', index_col=0, parse_dates=True)['close']
    rt = json.loads((HERE / 'data' / 'rates.json').read_text(encoding='utf-8'))
    val_date = s.index.max()
    r = np.log(s).diff().dropna()

    def vol(years):
        x = r[r.index > val_date - pd.DateOffset(years=years)] if years else r
        return float(x.std(ddof=1) * np.sqrt(252)), int(len(x))

    v3, n3 = vol(3); v1, n1 = vol(1); vl, nl = vol(None)
    q = DPS / float(s.iloc[-1])                 # 직전 연도 배당이 유지된다고 본 배당수익률
    info = {
        'valuation_date': str(val_date.date()), 'ticker': rt['ticker'], 'name': rt['name'],
        's0_krw': float(s.iloc[-1]),
        'sigma_3y': v3, 'sigma_1y': v1, 'sigma_all': vl, 'n_daily': [n3, n1, nl],
        'window_all': f'{r.index.min().date()} ~ {val_date.date()}',
        'r': rt['ktb3'], 'r_src': f"국고채 3년 (ECOS, {rt['ktb3_date']})",
        'aa3': rt['aa3'], 'bbb3': rt['bbb3'],
        's_aa': rt['aa3'] - rt['ktb3'], 's_bbb': rt['bbb3'] - rt['ktb3'],
        'spread_src': f"회사채 3년 AA-·BBB- − 국고채 3년 (ECOS, {rt['aa3_date']})",
        'q': q, 'dps': DPS, 'q_src': DPS_SRC,
    }
    # 기본: 만기(3년)와 같은 기간의 과거 변동성, AA- 스프레드. 스트레스: 최근 1년 변동성 / BBB- 스프레드
    mkt = Mkt(s0=1.0, sigma=v3, r=info['r'], s=info['s_aa'], q=q)
    return mkt, info, s


# ---------------------------------------------------------------- 2. 엔진 검증
def checks(mkt):
    out = {}
    # (1) 만기에만 전환: 해석해 vs 몬테카를로 vs 이항트리
    e = CB(conv_start=3.0, conv_end_gap=0.0, put_start=None, refix=False)
    r = price(e, mkt, N_MAIN, seed=5)
    out['european'] = {'closed': european_cb(e, mkt), 'mc': r['price'], 'mc_se': r['se'], 'tree': tree_cb(e, mkt, m=10)}
    # (2) 주식 버뮤단 풋(1년, 주 단위 행사): LSMC vs 이항트리 (Longstaff-Schwartz 기본 예제)
    pv, pse = lsmc_put(mkt, 1.0, 1.0, N_MAIN, seed=5)
    out['put'] = {'lsmc': pv, 'lsmc_se': pse, 'tree': tree_put(mkt, 1.0, 1.0, m=10),
                  'european': bs_put(1.0, 1.0, 1.0, mkt.r, mkt.q, mkt.sigma)}
    # (3) 리픽싱 없는 CB(전환 + 풋): 트리로도 풀리는 경우에서 LSMC 기저함수 비교
    nr = CB(refix=False)
    rows = {'tree': tree_cb(nr, mkt, m=10)}
    for b in ('cubic', 'euro'):
        core.BASIS = b
        rr = price(nr, mkt, N_MAIN, seed=5)
        rows[b] = {'oos': rr['price'], 'se': rr['se'], 'in_sample': rr['in_sample']}
    core.BASIS = 'euro'
    out['no_refix'] = rows
    # 트리 스텝 수 수렴
    out['tree_steps'] = [(m, tree_cb(nr, mkt, m=m)) for m in (2, 5, 10, 20)]
    return out


def convergence(prod, mkt):
    rows = []
    for n in (5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 400_000):
        r = price(prod, mkt, n, seed=11)
        rows.append({'n': n, 'price': r['price'], 'se': r['se']})
    return rows


# ---------------------------------------------------------------- 3. 가치 분해와 시나리오
def decompose(mkt):
    R = CB().redemption()
    out = {'bond_no_put': float(R * np.exp(-(mkt.r + mkt.s) * 3.0))}
    out['bond_with_put'] = price(CB(conv_start=99, refix=False), mkt, N_GRID, seed=3)['price']   # 전환권 없는 채권 + 풋
    out['conv_only'] = price(CB(put_start=None, refix=False), mkt, N_MAIN, seed=7)['price']
    out['conv_put'] = price(CB(refix=False), mkt, N_MAIN, seed=7)['price']
    full = price(CB(), mkt, N_MAIN, seed=7)
    out['full'] = full['price']; out['full_detail'] = full
    out['v_conversion'] = out['conv_only'] - out['bond_no_put']
    out['v_put'] = out['conv_put'] - out['conv_only']
    out['v_refix'] = out['full'] - out['conv_put']
    return out


def greeks(prod, mkt, n=N_MAIN):
    """같은 난수로 입력만 바꿔 재평가(회귀계수도 같은 훈련 난수로 다시 추정)."""
    base = price(prod, mkt, n, seed=7)['price']
    up = price(prod, bump(mkt, s0=1.01), n, seed=7)['price']
    dn = price(prod, bump(mkt, s0=0.99), n, seed=7)['price']
    return {
        'base': base,
        'delta_1pct': (up - dn) / 2, 'gamma_1pct': up - 2 * base + dn,
        'vega_1pp': price(prod, bump(mkt, sigma=mkt.sigma + 0.01), n, seed=7)['price'] - base,
        'spread_50bp': price(prod, bump(mkt, s=mkt.s + 0.005), n, seed=7)['price'] - base,
        'rate_10bp': price(prod, bump(mkt, r=mkt.r + 0.001), n, seed=7)['price'] - base,
        'q_1pp': price(prod, bump(mkt, q=mkt.q + 0.01), n, seed=7)['price'] - base,
    }


def scenarios(mkt, info):
    out = {}
    for name, m in (('기본 (3년 변동성, AA-)', mkt),
                    ('변동성 스트레스 (최근 1년)', bump(mkt, sigma=info['sigma_1y'])),
                    ('신용 스트레스 (BBB-)', bump(mkt, s=info['s_bbb'])),
                    ('동시 스트레스', bump(mkt, sigma=info['sigma_1y'], s=info['s_bbb']))):
        r = price(CB(), m, N_MAIN, seed=7)
        r['bond_with_put'] = price(CB(conv_start=99, refix=False), m, N_GRID, seed=3)['price']
        out[name] = r
    return out


def grids(mkt):
    g = {}
    xs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0]
    g['stock'] = []
    for x in xs:
        m = bump(mkt, s0=x)
        g['stock'].append({'s0': x,
                           'full': price(CB(), m, N_GRID, seed=3)['price'],
                           'no_refix': price(CB(refix=False), m, N_GRID, seed=3)['price'],
                           'bond': price(CB(conv_start=99, refix=False), m, N_GRID, seed=3)['price'],
                           'parity': 100 * x})
    g['sigma'] = [(s, price(CB(), bump(mkt, sigma=s), N_GRID, seed=3)['price'],
                   price(CB(refix=False), bump(mkt, sigma=s), N_GRID, seed=3)['price'])
                  for s in (0.25, 0.35, 0.45, 0.55, 0.65)]
    g['spread'] = [(s, price(CB(), bump(mkt, s=s), N_GRID, seed=3)['price'])
                   for s in (0.0, 0.0068, 0.02, 0.04, 0.065, 0.08)]
    g['floor'] = [(f, price(CB(refix_floor=f), mkt, N_GRID, seed=3)['price']) for f in (0.6, 0.7, 0.8, 0.9, 1.0)]
    g['q'] = [(q, price(CB(), bump(mkt, q=q), N_GRID, seed=3)['price']) for q in (0.0, 0.01, 0.02, 0.03)]
    return g


# ---------------------------------------------------------------- 4. 그림
def figures(conv, g, mkt, info):
    # 주가에 따른 CB 가치: 채권 하한, 패리티, 리픽싱 유무
    df = pd.DataFrame(g['stock'])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(df['s0'] * 100, df['full'], 'o-', color=NAVY, lw=1.6, ms=4, label='CB 가치 (전환 + 풋 + 리픽싱)')
    ax.plot(df['s0'] * 100, df['no_refix'], 's--', color=BLUE, lw=1.2, ms=3.5, label='리픽싱 없는 CB')
    ax.plot(df['s0'] * 100, df['bond'], color=ORANGE, lw=1.2, label='채권 가치 (전환권 없음, 풋 포함)')
    ax.plot(df['s0'] * 100, df['parity'], color='gray', lw=0.9, ls=':', label='패리티 (지금 전환하면 받는 가치)')
    ax.set_xlabel('현재 주가 (최초 전환가액 대비 %)'); ax.set_ylabel('가치 (액면 100 기준)')
    ax.set_title('주가에 따른 CB 가치: 주가가 낮을수록 리픽싱 가치가 커진다', fontsize=10.5)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RES / 'fig_parity.png', dpi=200); plt.close(fig)

    # 수렴
    c = pd.DataFrame(conv)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.errorbar(c['n'], c['price'], yerr=1.96 * c['se'], fmt='o-', color=NAVY, capsize=3, lw=1.2)
    ax.axhline(conv[-1]['price'], color='gray', ls='--', lw=0.8)
    ax.set_xscale('log'); ax.set_xlabel('평가 경로 수 (로그 축)'); ax.set_ylabel('CB 가치')
    ax.set_title('경로 수에 따른 가격 수렴 (95% 신뢰구간, 회귀용 경로 10만 개 별도)', fontsize=10.5)
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(RES / 'fig_convergence.png', dpi=200); plt.close(fig)

    # 민감도: 변동성, 신용스프레드, 리픽싱 하한
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7))
    xs, full, nr = zip(*g['sigma'])
    axs[0].plot(np.array(xs) * 100, full, 'o-', color=NAVY, lw=1.2, label='리픽싱 있음')
    axs[0].plot(np.array(xs) * 100, nr, 's--', color=BLUE, lw=1.0, ms=3.5, label='없음')
    axs[0].set_xlabel('변동성 (%)', fontsize=8.5); axs[0].legend(fontsize=7, frameon=False)
    xs, ys = zip(*g['spread'])
    axs[1].plot(np.array(xs) * 100, ys, 'o-', color=NAVY, lw=1.2)
    axs[1].axvline(info['s_aa'] * 100, color='gray', lw=0.7, ls=':'); axs[1].axvline(info['s_bbb'] * 100, color='gray', lw=0.7, ls=':')
    axs[1].set_xlabel('신용스프레드 (%p)', fontsize=8.5)
    xs, ys = zip(*g['floor'])
    axs[2].plot(np.array(xs) * 100, ys, 'o-', color=NAVY, lw=1.2)
    axs[2].set_xlabel('리픽싱 하한 (최초 전환가 대비 %)', fontsize=8.5)
    axs[0].set_ylabel('CB 가치', fontsize=8.5)
    for ax in axs:
        ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
    fig.suptitle('입력값에 따른 CB 가치 변화', fontsize=11); fig.tight_layout()
    fig.savefig(RES / 'fig_sens.png', dpi=200); plt.close(fig)

    # 표본 경로와 리픽싱된 전환가액
    S = paths(CB(), mkt, 6, seed=2026)[[0, 1, 2]]
    CP = conv_price_path(CB(), S)
    t = np.arange(S.shape[1]) / WEEKS
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for i, col in enumerate((NAVY, ORANGE, '#548235')):
        ax.plot(t, S[i] * 100, color=col, lw=0.9, alpha=0.85)
        ax.step(t, CP[i] * 100, where='post', color=col, lw=1.4, ls='--')
    ax.axhline(70, color='gray', lw=0.8, ls=':'); ax.text(0.05, 72, '리픽싱 하한 70%', fontsize=7.5)
    ax.axvline(1.0, color='gray', lw=0.6); ax.text(1.02, ax.get_ylim()[1] * 0.92, '전환 시작', fontsize=7.5)
    ax.axvline(1.5, color='gray', lw=0.6); ax.text(1.52, ax.get_ylim()[1] * 0.84, '풋 시작', fontsize=7.5)
    ax.set_xlabel('경과 연수'); ax.set_ylabel('최초 전환가액 대비 %')
    ax.set_title('표본 경로 3개: 실선 = 주가, 점선 = 리픽싱된 전환가액', fontsize=10.5)
    ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(RES / 'fig_paths.png', dpi=200); plt.close(fig)


if __name__ == '__main__':
    t0 = time.time()
    mkt, info, s = params()
    print('파라미터', json.dumps(info, ensure_ascii=False, indent=1))
    chk = checks(mkt); print('검증', json.dumps(chk, ensure_ascii=False, indent=1), round(time.time() - t0))
    dec = decompose(mkt); print('가치 분해', json.dumps(dec, ensure_ascii=False, indent=1), round(time.time() - t0))
    conv = convergence(CB(), mkt)
    grk = greeks(CB(), mkt); print('민감도', grk)
    scen = scenarios(mkt, info); print('시나리오', json.dumps(scen, ensure_ascii=False, indent=1))
    g = grids(mkt)
    figures(conv, g, mkt, info)
    prod = CB()
    out = {'params': info, 'product': {**prod.__dict__, 'redemption': prod.redemption(),
                                       'puts': {f'{k / WEEKS:.2f}y': v for k, v in prod.put_steps().items()}},
           'check': chk, 'decompose': dec, 'convergence': conv, 'greeks': grk, 'scenarios': scen, 'grids': g,
           'runtime_sec': round(time.time() - t0)}
    (RES / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print('완료', round(time.time() - t0), '초')
