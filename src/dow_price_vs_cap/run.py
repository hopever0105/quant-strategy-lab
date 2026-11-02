"""백테스트 실행: 가격가중(DJIA 방식) vs 시가총액가중, 같은 편입 종목.
결과: results/results.json, results/*.csv, results/fig_*.png
실행: python data.py → python run.py → python build_report.py
"""
import json
import sys
from dataclasses import replace, asdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core  # noqa: E402
import data as D  # noqa: E402
import universe as U  # noqa: E402

RES = HERE / 'results'
RES.mkdir(exist_ok=True)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
COL = {'price': '#C55A11', 'cap': '#2F5597', 'equal': '#7F7F7F', 'dji': '#000000'}
NAME = {'price': '가격가중(다우 방식)', 'cap': '시가총액가중', 'equal': '동일가중(참고)'}


def load():
    close, raw, div, n_fix, notes = D.build_panel()
    mc, wba, dji = D.load_mcap(), D.load_wba(), D.load_index()['DJI']
    all_dates = close.loc[:'2026-10-31'].index                   # 2009-06부터: 2009년 말 시가총액 관측치를 쓰기 위해
    C, R, Dv, CAP = core.build_inputs(close, raw, div, mc, wba, all_dates)
    mem = U.membership(all_dates)
    dates = close.loc[U.START:'2026-10-31'].index
    sp = D.load_splits()
    splits_by_member = {m: list(sp.get(U.price_ticker(m), {}).keys()) for m in mem.columns}
    fx = D.fred('DEXKOUS').reindex(dates).ffill().bfill()
    us = (D.fred('DTB3') / 100).reindex(dates).ffill().bfill()
    kr = (D.fred('IR3TIB01KRM156N') / 100)
    kr = kr.reindex(pd.date_range(kr.index.min(), dates[-1])).ffill().reindex(dates).ffill().bfill()
    dia = close['DIA'].reindex(dates)
    dia_adj = D.load_prices().query("ticker=='DIA'").set_index('date')['adj'].reindex(dates)
    return dict(C=C, R=R, Dv=Dv, CAP=CAP, mem=mem, sp=splits_by_member, fx=fx, us=us, kr=kr, dji=dji.reindex(dates).ffill(),
                dia_adj=dia_adj, n_fix=n_fix, fix_notes=notes, n_split=sum(len(v) for v in sp.values()), dates=dates)


def run(kind, X, p):
    return core.simulate(kind, X['C'], X['R'], X['Dv'], X['CAP'], X['mem'], p, X['sp'])


def ccy_paths(res, X):
    """USD, 원화 비헤지, 원화 환헤지 자산 경로와 취득원가 누계(청산 과세용).
    분배금은 펀드 밖으로 나갔다가 세후 전액 재투자되므로 자산 경로에는 외부 유입이 없다.
    취득원가 = 최초 투자 + 재투자액(그날 환율로 원화 환산)."""
    nav = res.nav_usd
    d = nav.index
    reinv = res.basis_usd.diff().fillna(0)                               # USD 재투자액
    fx = X['fx'].loc[d]
    carry = ((X['kr'] - X['us']).loc[d] / 252).shift(1).fillna(0)        # 환헤지 손익 ≈ (원화금리 − 달러금리)
    r = nav.pct_change().fillna(0)
    hed = (1 + r + carry).cumprod() * nav.iloc[0] * fx.iloc[0]
    hed.iloc[0] = nav.iloc[0] * fx.iloc[0]
    return {
        'USD': (nav, nav.iloc[0] + reinv.cumsum()),
        'KRW': (nav * fx, nav.iloc[0] * fx.iloc[0] + (reinv * fx).cumsum()),
        'KRW_H': (hed, hed.iloc[0] + (reinv / nav * hed).cumsum()),
    }


def metrics(v, basis, rf, gain_tax):
    d = v.index
    years = (d[-1] - d[0]).days / 365.25
    r = v.pct_change().iloc[1:]
    tw = v.iloc[-1] / v.iloc[0]
    cagr = tw ** (1 / years) - 1
    vol = r.std() * np.sqrt(252)
    ex = r - rf.loc[r.index] / 252
    sharpe = ex.mean() / r.std() * np.sqrt(252)
    mdd = (v / v.cummax() - 1).min()
    # 청산: (최종자산 − 취득원가 누계) × 15.4% 세금
    gain = v.iloc[-1] - basis.iloc[-1]
    after = v.iloc[-1] - gain_tax * max(gain, 0)
    tw_after = after / v.iloc[0]
    return dict(CAGR=cagr, CAGR_after_liq=tw_after ** (1 / years) - 1, Vol=vol, Sharpe=sharpe, MDD=mdd,
                Final=tw, Final_after_liq=tw_after)


def clean_returns(res):
    return res.nav_usd.pct_change().iloc[1:]


def alpha_test(r_cap, r_price):
    """시총가중 일별 수익률 = a + b × 가격가중 + e, Newey-West(lag 5). 월별 초과수익 평균 t값도."""
    X = sm.add_constant(r_price.values)
    m = sm.OLS(r_cap.values, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    diff_m = ((1 + r_cap).resample('ME').prod() - (1 + r_price).resample('ME').prod())
    t_m = diff_m.mean() / diff_m.std() * np.sqrt(len(diff_m))
    return dict(alpha_ann=m.params[0] * 252, alpha_t=m.tvalues[0], beta=m.params[1], beta_t=m.tvalues[1],
                ex_month_mean=diff_m.mean(), ex_month_t=t_m, n_months=len(diff_m),
                hit=(diff_m > 0).mean())


def yearly(r):
    return (1 + r).groupby(r.index.year).prod() - 1


def main():
    X = load()
    p = core.Params()
    out = {'params': asdict(p), 'n_fix': X['n_fix'], 'n_split': X['n_split'], 'fix_notes': X['fix_notes']}

    # 1) 복제 검증: 가격가중 가격지수 vs 실제 다우(^DJI)
    pi = core.price_index(X['C'], X['R'], X['mem'], X['dates'])
    a, b = pi.pct_change().dropna(), X['dji'].pct_change().dropna()
    e = a - b
    ya, yb = yearly(a), yearly(b)
    out['replication'] = dict(corr=float(a.corr(b)), te=float(e.std() * np.sqrt(252)),
                              yearly={int(y): [float(ya[y]), float(yb[y])] for y in ya.index})
    # 배당 포함 검증: 보수·비용·세금 0인 가격가중 총수익 vs DIA 수정종가(보수 0.16% 차감된 실제 ETF)
    g = run('price', X, replace(p, fee=0, cost=0, div_tax=0))
    rg = clean_returns(g)
    dia = X['dia_adj'].pct_change().dropna()
    out['dia_check'] = dict(ours_cagr=float((1 + rg).prod() ** (252 / len(rg)) - 1),
                            dia_cagr=float((1 + dia).prod() ** (252 / len(dia)) - 1),
                            corr=float(rg.corr(dia.reindex(rg.index))),
                            yearly={int(y): [float(v1), float(v2)] for (y, v1), v2 in zip(yearly(rg).items(), yearly(dia).values)})

    # 2) 기본 백테스트
    R_ = {k: run(k, X, p) for k in ['price', 'cap', 'equal']}
    rf = {'USD': X['us'], 'KRW': X['kr'], 'KRW_H': X['kr']}
    base = {}
    paths = {}
    for k, res in R_.items():
        cp = ccy_paths(res, X)
        paths[k] = cp
        base[k] = {c: metrics(v, bs, rf[c], p.gain_tax) for c, (v, bs) in cp.items()}
        base[k]['turnover'] = res.turnover
        base[k]['n_trades'] = res.n_trades
        base[k]['div_yield'] = float(res.div_gross.sum() / ((res.nav_usd.index[-1] - res.nav_usd.index[0]).days / 365.25))
    out['base'] = base
    rp, rc, re_ = clean_returns(R_['price']), clean_returns(R_['cap']), clean_returns(R_['equal'])
    out['yearly'] = {int(y): dict(price=float(v1), cap=float(v2), equal=float(v3))
                     for (y, v1), v2, v3 in zip(yearly(rp).items(), yearly(rc).values, yearly(re_).values)}
    out['alpha'] = alpha_test(rc, rp)
    out['alpha_equal_vs_price'] = alpha_test(re_, rp)

    # 3) 비중 비교·기여도
    W = {k: R_[k].weights for k in ['price', 'cap']}
    last = {k: W[k].iloc[-1] for k in W}
    top = pd.DataFrame(last).fillna(0).sort_values('cap', ascending=False)
    top.to_csv(RES / 'weights_last.csv', encoding='utf-8-sig')
    out['weights_last'] = {m: [float(top.loc[m, 'price']), float(top.loc[m, 'cap'])] for m in top.index}
    # 연도별 평균 비중(리밸런싱 직후 기준) 상위 5종목 비중 합 = 집중도
    conc = {}
    for k in W:
        w = W[k]
        conc[k] = {int(y): float(g_.iloc[-1].sort_values(ascending=False).head(5).sum()) for y, g_ in w.groupby(w.index.year)}
    out['top5_conc'] = conc
    ctb = {k: R_[k].contrib.sum() for k in ['price', 'cap']}
    cdf = pd.DataFrame(ctb).fillna(0)
    cdf['diff'] = cdf['cap'] - cdf['price']
    cdf = cdf.sort_values('diff')
    cdf.to_csv(RES / 'contrib.csv', encoding='utf-8-sig')
    out['contrib_top'] = {m: [float(cdf.loc[m, 'price']), float(cdf.loc[m, 'cap']), float(cdf.loc[m, 'diff'])]
                          for m in list(cdf.index[:6]) + list(cdf.index[-8:])}

    # 4) 강건성
    rob = {}

    def summary(pp, label, kinds=('price', 'cap')):
        rr = {k: run(k, X, pp) for k in kinds}
        row = {}
        for k in kinds:
            cp = ccy_paths(rr[k], X)
            v, bs = cp['KRW']
            row[k] = metrics(v, bs, X['kr'], pp.gain_tax)
            row[k]['turnover'] = rr[k].turnover
        row['alpha'] = alpha_test(clean_returns(rr['cap']), clean_returns(rr['price']))
        rob[label] = row

    summary(replace(p, start='2010-01-01', end='2017-12-31'), '2010~2017')
    summary(replace(p, start='2018-01-01', end='2026-10-31'), '2018~2026')
    summary(replace(p, start='2013-09-23', end='2026-10-31'), '2013.09~ (알코아 제외 영향 없는 구간)')
    summary(replace(p, cost=p.cost * 2), '거래비용 2배(10bp)')
    summary(replace(p, cost=p.cost * 4), '거래비용 4배(20bp)')
    summary(replace(p, cap_freq='M'), '시총 리밸런싱 매월')
    summary(replace(p, cap_freq='H'), '시총 리밸런싱 반기')
    summary(replace(p, cap_freq='Y'), '시총 리밸런싱 연 1회')
    summary(replace(p, fee=0.0001), '총보수 0.01%')
    summary(replace(p, div_tax=0, gain_tax=0), '세금 없음(참고: 세금 영향 측정)')
    out['robust'] = rob
    # 시작 시점을 바꿔 가며(매년 1월 시작 → 끝까지) 시총-가격 CAGR 차이
    roll = {}
    for y in range(2010, 2024):
        pp = replace(p, start=f'{y}-01-01')
        rr = {k: run(k, X, pp) for k in ['price', 'cap']}
        cg = {k: metrics(*ccy_paths(rr[k], X)['KRW'], X['kr'], pp.gain_tax)['CAGR'] for k in rr}
        roll[y] = [cg['price'], cg['cap']]
    out['start_year'] = roll
    # 3년 롤링 초과수익
    lv_p, lv_c = (1 + rp).cumprod(), (1 + rc).cumprod()
    roll3 = (lv_c / lv_c.shift(756)) ** (1 / 3) - (lv_p / lv_p.shift(756)) ** (1 / 3)
    out['roll3_pos_share'] = float((roll3.dropna() > 0).mean())

    (RES / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    pd.DataFrame({k: paths[k]['KRW'][0] for k in paths}).to_csv(RES / 'nav_krw.csv', encoding='utf-8-sig')

    # ───────── 차트
    fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=170)
    for k in ['price', 'cap', 'equal']:
        v, bs = paths[k]['KRW']
        r = clean_returns(R_[k])
        lv = (1 + r).cumprod()
        ax.plot(lv.index, lv * 100, color=COL[k], lw=1.4 if k != 'equal' else 0.9, ls='-' if k != 'equal' else '--',
                label=NAME[k] + ' (USD 기준)')
    ax.set_yscale('log'); ax.set_ylabel('누적 가치 (시작=100, 로그)')
    ax.set_yticks([100, 150, 200, 300, 400, 600, 800])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title('다우 30종목: 가격가중 vs 시가총액가중 (분배금 15.4% 과세 후 재투자, 보수·비용 차감)', fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=0.25)
    [ax.spines[s].set_visible(False) for s in ('top', 'right')]
    plt.tight_layout(); plt.savefig(RES / 'fig_cum.png'); plt.close()

    fig, ax = plt.subplots(figsize=(7.4, 2.6), dpi=170)
    for k in ['price', 'cap']:
        lv = (1 + clean_returns(R_[k])).cumprod()
        ax.fill_between(lv.index, (lv / lv.cummax() - 1) * 100, 0, color=COL[k], alpha=0.35, label=NAME[k])
    ax.set_ylabel('고점 대비 하락 (%)'); ax.set_title('낙폭(Drawdown), USD 기준', fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc='lower left'); ax.grid(alpha=0.25)
    [ax.spines[s].set_visible(False) for s in ('top', 'right')]
    plt.tight_layout(); plt.savefig(RES / 'fig_dd.png'); plt.close()

    yy = pd.DataFrame(out['yearly']).T
    fig, ax = plt.subplots(figsize=(7.4, 2.9), dpi=170)
    x = np.arange(len(yy))
    ax.bar(x - 0.2, yy['price'] * 100, 0.4, color=COL['price'], label=NAME['price'])
    ax.bar(x + 0.2, yy['cap'] * 100, 0.4, color=COL['cap'], label=NAME['cap'])
    ax.set_xticks(x); ax.set_xticklabels([str(i)[2:] for i in yy.index], fontsize=8)
    ax.axhline(0, color='black', lw=0.6); ax.set_ylabel('연 수익률 (%)')
    ax.set_title('연도별 수익률 (USD, 세후 분배 재투자; 2026은 10월까지)', fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=0.25, axis='y')
    [ax.spines[s].set_visible(False) for s in ('top', 'right')]
    plt.tight_layout(); plt.savefig(RES / 'fig_yearly.png'); plt.close()

    t = top.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=170)
    y_ = np.arange(len(t))
    ax.barh(y_ - 0.2, t['price'] * 100, 0.4, color=COL['price'], label=NAME['price'])
    ax.barh(y_ + 0.2, t['cap'] * 100, 0.4, color=COL['cap'], label=NAME['cap'])
    ax.set_yticks(y_); ax.set_yticklabels(t.index, fontsize=8)
    ax.set_xlabel('비중 (%)'); ax.set_title(f'같은 30종목, 다른 비중: 마지막 리밸런싱({W["cap"].index[-1].date()}) 기준 상위 15', fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc='lower right'); ax.grid(alpha=0.25, axis='x')
    [ax.spines[s].set_visible(False) for s in ('top', 'right')]
    plt.tight_layout(); plt.savefig(RES / 'fig_weights.png'); plt.close()

    cc = cdf['diff'].copy()
    cc = pd.concat([cc.head(6), cc.tail(8)])
    fig, ax = plt.subplots(figsize=(7.4, 3.2), dpi=170)
    ax.barh(range(len(cc)), cc * 100, color=['#C00000' if v < 0 else '#2F5597' for v in cc])
    ax.set_yticks(range(len(cc))); ax.set_yticklabels(cc.index, fontsize=8)
    ax.axvline(0, color='black', lw=0.6)
    ax.set_xlabel('누적 기여 차이 (시총가중 - 가격가중, %p, 일별 기여 단순합)')
    ax.set_title('어떤 종목이 차이를 만들었나', fontsize=9.5); ax.grid(alpha=0.25, axis='x')
    [ax.spines[s].set_visible(False) for s in ('top', 'right')]
    plt.tight_layout(); plt.savefig(RES / 'fig_contrib.png'); plt.close()

    fig, ax = plt.subplots(figsize=(7.4, 2.6), dpi=170)
    ax.plot(roll3.index, roll3 * 100, color='#2F5597', lw=1.1)
    ax.axhline(0, color='black', lw=0.6)
    ax.set_ylabel('%p / 년'); ax.set_title('3년 롤링 연환산 초과수익 (시총가중 - 가격가중, USD)', fontsize=9.5)
    ax.grid(alpha=0.25); [ax.spines[s].set_visible(False) for s in ('top', 'right')]
    plt.tight_layout(); plt.savefig(RES / 'fig_roll3.png'); plt.close()

    fig, ax = plt.subplots(figsize=(7.4, 2.6), dpi=170)
    for k in ['price', 'cap']:
        s = pd.Series(out['top5_conc'][k])
        ax.plot(s.index, s.values * 100, marker='o', ms=3, color=COL[k], label=NAME[k])
    ax.set_ylabel('상위 5종목 비중 합 (%)'); ax.set_title('집중도: 연말 상위 5종목 비중 합', fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=0.25)
    [ax.spines[s].set_visible(False) for s in ('top', 'right')]
    plt.tight_layout(); plt.savefig(RES / 'fig_conc.png'); plt.close()

    print(json.dumps({k: {c: {m: round(v, 4) for m, v in base[k][c].items()} for c in ['USD', 'KRW', 'KRW_H']} for k in base},
                     ensure_ascii=False, indent=1))
    print('turnover', {k: round(base[k]['turnover'], 3) for k in base}, 'trades', {k: base[k]['n_trades'] for k in base},
          'div_yield', {k: round(base[k]['div_yield'], 4) for k in base})
    print('alpha', out['alpha'])
    print('replication', {k: v for k, v in out['replication'].items() if k != 'yearly'})
    print('dia', {k: v for k, v in out['dia_check'].items() if k != 'yearly'})
    for k, v in rob.items():
        print(k, {kk: round(vv['CAGR'], 4) for kk, vv in v.items() if kk != 'alpha'}, 'alpha_t', round(v['alpha']['alpha_t'], 2))
    print('start_year', {y: [round(a_, 4), round(b_, 4)] for y, (a_, b_) in roll.items()})
    print('roll3 positive share', out['roll3_pos_share'])
    print(top.head(12).round(4))
    print(cdf.round(4).head(6)); print(cdf.round(4).tail(8))


if __name__ == '__main__':
    main()
