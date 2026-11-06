"""실행: python quant/strategies/turtle_ml_allocation/run.py
같은 입력(캐시)이면 같은 결과가 나온다(난수 시드 고정). 결과는 results/ 에 저장한다.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data  # noqa: E402
from universe import GROUP, NAME  # noqa: E402
from signals import features, add_ext, turtle_trades, monthly_trades, to_frame, month_end_idx, FEATS, FEATS_EXT  # noqa: E402
from backtest import base_weights, fixed_weights, mult_matrix, simulate, metrics, alpha_t, yearly  # noqa: E402
from ml import walk_forward, to_mult, diagnostics  # noqa: E402

RES = HERE / "results"
RES.mkdir(exist_ok=True)
COST = 0.0005
THR = 0.3
N_PLACEBO = 20
t0 = time.time()

# ───────── 데이터
P, info, mac = data.build()
idx = P.index
cols = list(P.columns)
groups = [GROUP[c] for c in cols]
T, A = P.shape
R = P.pct_change(fill_method=None).values
cash = (mac["TBILL"].shift(1) / 100 / 252).fillna(0).values
F = add_ext(features(P, mac), P, mac, GROUP)
start = idx.searchsorted(pd.Timestamp("1998-01-02"))
oos = idx.searchsorted(pd.Timestamp("2008-01-02"))
end = T - 1
PER = {
    "표본 밖 2008-01~2026-11": (oos, end),
    "전반 2008~2016": (oos, idx.searchsorted(pd.Timestamp("2017-01-01")) - 1),
    "후반 2017~2026": (idx.searchsorted(pd.Timestamp("2017-01-01")), end),
    "전체 1998~2026": (start, end),
}
first_elig = {c: idx[F["elig"][c].values.argmax()] for c in cols}
new_assets = [c for c in cols if first_elig[c] > pd.Timestamp("2007-12-31")]
fixed_mask = np.array([first_elig[c] <= idx[start] for c in cols])

me = month_end_idx(idx)
force = np.zeros(T, bool); force[np.clip(me + 1, 0, T - 1)] = True


def run(B, M, cost=COST, frc=None):
    out = simulate(R, cash, B, M, cost, start, end, frc)
    return pd.DataFrame(out, columns=["ret", "turn", "expo"], index=idx[start:end + 1])


def per_metrics(df, ref=None):
    o = {}
    for k, (a, b) in PER.items():
        s = df.loc[idx[a]:idx[b]]
        c = pd.Series(cash, index=idx).loc[s.index]
        m = metrics(s.ret.values, c.values, s.turn.values, s.expo.values)
        if ref is not None:
            m.update(alpha_t(s.ret.values, ref.loc[s.index].ret.values, c.values))
        o[k] = m
    return o


def oos_metrics(df, ref=None, ref2=None):
    a, b = PER["표본 밖 2008-01~2026-11"]
    s = df.loc[idx[a]:idx[b]]; c = pd.Series(cash, index=idx).loc[s.index]
    m = metrics(s.ret.values, c.values, s.turn.values, s.expo.values)
    if ref is not None:
        at = alpha_t(s.ret.values, ref.loc[s.index].ret.values, c.values)
        m.update(alpha_vs_rp=at["alpha"], t_vs_rp=at["alpha_t"])
    if ref2 is not None:
        at = alpha_t(s.ret.values, ref2.loc[s.index].ret.values, c.values)
        m.update(alpha_vs_rule=at["alpha"], t_vs_rule=at["alpha_t"])
    return m


def trade_stats(tr, mult=None):
    d = tr[tr.e_date >= idx[oos]].copy()
    if mult is not None:
        d = d[np.asarray(mult)[d.index] > 0]
    return dict(trades=int(len(d)), win=float((d.ret > 0).mean()), avg_ret=float(d.ret.mean()),
                hold=float(d.hold_days.mean()), per_year=float(len(d) / ((idx[end] - idx[oos]).days / 365.25)))


# ───────── 비교 대상
B_rp = base_weights(F, groups, start)
ONE = np.ones((T, A))
res = {}
res["리스크 패리티"] = run(B_rp, ONE)
B_6040 = fixed_weights(idx, cols, {"SPY": 0.6, "IEF": 0.4})
res["60/40"] = run(B_6040, ONE, frc=force)

# ───────── 1차 신호 + ML
def strat(rows, kind="logit", thr=THR, cost=COST, mask=None, shuffle=None, B=None, label="fwd20", rel=False, feats=FEATS):
    """label='fwd20'(사용자 규격: 진입 후 20일 수익>0) / 'trade'(그 거래의 실제 손익>0, 청산일에 정답 확정)
    rel=True: 배수 = p ÷ 학습 기간 평균 승률"""
    tr = to_frame(rows, P, F, GROUP)
    if mask is not None:
        tr = tr[mask[tr.ai.values]].reset_index(drop=True)
    if label == "trade":
        done = tr.reason != "기간 끝"
        tr["label"] = np.where(done, (tr.ret > 0).astype(float), np.nan)
        tr["label_date"] = tr.x_date.where(done)
    B = B_rp if B is None else B
    rule = run(B, mult_matrix(tr, T, A), cost)
    p, coefs, ntr, base = walk_forward(tr, kind, shuffle, feats)
    mult = to_mult(p.reindex(tr.index).values, thr, base.reindex(tr.index).values if rel else None)
    mlr = run(B, mult_matrix(tr, T, A, mult), cost)
    tr = tr.assign(p=p.reindex(tr.index), mult=mult)
    return tr, rule, mlr, p, coefs, ntr


rows20 = turtle_trades(P, F, start, 20, 20)
tr, res["터틀 규칙"], res["터틀 + ML"], p, coefs, ntr = strat(rows20)
print(f"기본 실행 {time.time()-t0:.0f}초")

main = {k: per_metrics(v, res["리스크 패리티"] if k != "리스크 패리티" else None) for k, v in res.items()}
main["터틀 + ML"]["표본 밖 2008-01~2026-11"].update(
    {f"{k}_vs_rule": v for k, v in alpha_t(*(res[n].loc[idx[oos]:].ret.values for n in ("터틀 + ML", "터틀 규칙")),
                                             cash[oos:end + 1]).items()})
tstats = {"터틀 규칙": trade_stats(tr), "터틀 + ML": trade_stats(tr, tr["mult"].values)}
diag, cal, grp, nw = diagnostics(tr, p, new_assets)

# ───────── 강건성
rob = []


def add(label, df, ref_rule=None, note=""):
    m = oos_metrics(df, res["리스크 패리티"], ref_rule)
    m.update(label=label, note=note); rob.append(m)


add("기본: 터틀 규칙", res["터틀 규칙"])
add("기본: 터틀 + ML", res["터틀 + ML"], res["터틀 규칙"])
for c2 in (0.001,):
    add("비용 2배: 리스크 패리티", run(B_rp, ONE, c2))
    _, r1, r2, *_ = strat(rows20, cost=c2)
    add("비용 2배: 터틀 규칙", r1); add("비용 2배: 터틀 + ML", r2, r1)
for n in (10, 55):
    _, r1, r2, *_ = strat(turtle_trades(P, F, start, n, n))
    add(f"채널 {n}일: 터틀 규칙", r1); add(f"채널 {n}일: 터틀 + ML", r2, r1)
for th in (0.4, 0.5):
    _, _, r2, *_ = strat(rows20, thr=th)
    add(f"기준 {th}: 터틀 + ML", r2, res["터틀 규칙"])
tr_g, _, r2, p_g, *_ = strat(rows20, kind="gbm")
add("부스팅 모델: 터틀 + ML", r2, res["터틀 규칙"])
diag_g, *_ = diagnostics(tr_g, p_g, new_assets)
# 사후 추가 점검(결과를 본 뒤 설계 문제를 확인하려고 넣음, 본 결과는 위 사용자 규격)
_, _, r2, *_ = strat(rows20, rel=True)
add("상대 배수(p÷학습 승률): 터틀 + ML", r2, res["터틀 규칙"], "사후 점검")
tr_t, _, r2, p_t, *_ = strat(rows20, label="trade")
add("정답=거래 실제 손익: 터틀 + ML", r2, res["터틀 규칙"], "사후 점검")
diag_t, cal_t, *_ = diagnostics(tr_t, p_t, new_assets)
_, _, r2, *_ = strat(rows20, label="trade", rel=True)
add("정답=거래 손익 + 상대 배수: 터틀 + ML", r2, res["터틀 규칙"], "사후 점검")
sig_diag = {}
for kind, ko in (("ma200", "200일선"), ("mom12", "12개월 모멘텀")):
    rows_k = monthly_trades(P, F, start, kind)
    trk, r1, r2, pk, *_ = strat(rows_k)
    add(f"{ko}: 규칙", r1); add(f"{ko}: + ML", r2, r1)
    sig_diag[ko], *_ = diagnostics(trk, pk, new_assets)
    _, _, r3, *_ = strat(rows_k, rel=True)
    add(f"{ko}: + ML 상대 배수", r3, r1, "사후 점검")
# 변수 확장(사후 점검): 신용 스프레드·금리·자산 간 상관·12개월 수익률·변동성 국면·돌파 강도·자산군 (총 18개)
ext_diag, ext_cal = {}, None
for kind, ko in (("logit", "로지스틱"), ("gbm", "부스팅")):
    tr_e, _, r2, p_e, coef_e, *_ = strat(rows20, kind=kind, feats=FEATS_EXT)
    add(f"변수 확장 {ko}: 터틀 + ML", r2, res["터틀 규칙"], "사후 점검")
    ext_diag[ko], c_, g_, n_ = diagnostics(tr_e, p_e, new_assets)
    if kind == "logit":
        coef_e.to_csv(RES / "ml_coefs_ext.csv", encoding="utf-8-sig"); ext_cal = c_
    _, _, r2, p_e2, *_ = strat(rows20, kind=kind, feats=FEATS_EXT, label="trade")
    add(f"변수 확장 {ko} + 정답=거래 손익: 터틀 + ML", r2, res["터틀 규칙"], "사후 점검")
    tr_t2 = tr_e.copy(); done = tr_t2.reason != "기간 끝"
    tr_t2["label"] = np.where(done, (tr_t2.ret > 0).astype(float), np.nan)
    ext_diag[ko + "_거래손익"], *_ = diagnostics(tr_t2, p_e2, new_assets)
for kind, ko in (("ma200", "200일선"), ("mom12", "12개월 모멘텀")):
    rows_k = monthly_trades(P, F, start, kind)
    trk, r1, r2, pk, *_ = strat(rows_k, feats=FEATS_EXT)
    add(f"변수 확장 {ko}: + ML", r2, r1, "사후 점검")
    ext_diag[ko], *_ = diagnostics(trk, pk, new_assets)
ext_cal.to_csv(RES / "ml_calibration_ext.csv", encoding="utf-8-sig", index=False)
plc_ext = []
for kind in ("logit", "gbm"):
    for s in range(N_PLACEBO):
        _, _, r2, pp, *_ = strat(rows20, kind=kind, shuffle=s, feats=FEATS_EXT)
        m = oos_metrics(r2, None, res["터틀 규칙"]); m.update(seed=s, model=kind)
        d_, *_ = diagnostics(tr, pp, new_assets); m["auc"] = d_["auc"]
        plc_ext.append(m)
plc_ext = pd.DataFrame(plc_ext)
plc_ext.to_csv(RES / "placebo_ext.csv", encoding="utf-8-sig", index=False)
print(f"변수 확장 {time.time()-t0:.0f}초")

B_fix = base_weights(F, groups, start, fixed_mask)
add("고정 유니버스: 리스크 패리티", run(B_fix, ONE))
_, r1, r2, *_ = strat(rows20, mask=fixed_mask, B=B_fix)
add("고정 유니버스: 터틀 규칙", r1); add("고정 유니버스: 터틀 + ML", r2, r1)
print(f"강건성 {time.time()-t0:.0f}초")

plc = []
for s in range(N_PLACEBO):
    _, _, r2, pp, *_ = strat(rows20, shuffle=s)
    m = oos_metrics(r2, None, res["터틀 규칙"]); m["seed"] = s
    d_, *_ = diagnostics(tr, pp, new_assets); m["auc"] = d_["auc"]
    plc.append(m)
plc = pd.DataFrame(plc)
# 200일선 + ML도 같은 가짜 학습으로 확인 (AUC가 0.5 아래인데 성과가 좋아진 이유가 예측력인지 비중 효과인지)
rows_ma = monthly_trades(P, F, start, "ma200")
tr_ma, r_ma_rule, *_ = strat(rows_ma)
plc_ma = []
for s in range(N_PLACEBO):
    _, _, r2, pp, *_ = strat(rows_ma, shuffle=s)
    m = oos_metrics(r2, None, r_ma_rule); m["seed"] = s
    d_, *_ = diagnostics(tr_ma, pp, new_assets); m["auc"] = d_["auc"]
    plc_ma.append(m)
plc_ma = pd.DataFrame(plc_ma)
plc_ma.to_csv(RES / "placebo_ma200.csv", encoding="utf-8-sig", index=False)
print(f"가짜 학습 {time.time()-t0:.0f}초")

# ───────── 저장
daily = pd.concat({k: v.ret for k, v in res.items()}, axis=1)
daily = pd.concat([daily, pd.concat({k + "_노출": v.expo for k, v in res.items()}, axis=1)], axis=1)
daily.to_csv(RES / "daily.csv", encoding="utf-8-sig")
yr = pd.DataFrame({k: yearly(v.ret.values, v.index) for k, v in res.items()})
yr.to_csv(RES / "yearly.csv", encoding="utf-8-sig")
tr.to_csv(RES / "trades_turtle.csv", encoding="utf-8-sig", index=False)
info.assign(first_elig=[first_elig[c].date() for c in info.index]).to_csv(RES / "universe_info.csv", encoding="utf-8-sig")
pd.DataFrame(rob).to_csv(RES / "robustness.csv", encoding="utf-8-sig", index=False)
plc.to_csv(RES / "placebo.csv", encoding="utf-8-sig", index=False)
cal.to_csv(RES / "ml_calibration.csv", encoding="utf-8-sig", index=False)
grp.to_csv(RES / "ml_by_group.csv", encoding="utf-8-sig", index=False)
nw.to_csv(RES / "ml_new_assets.csv", encoding="utf-8-sig", index=False)
coefs.to_csv(RES / "ml_coefs.csv", encoding="utf-8-sig")
wk = (1 + pd.DataFrame(R, index=idx, columns=cols).iloc[oos:]).resample("W-FRI").prod() - 1
wk.corr().round(2).to_csv(RES / "corr_weekly_oos.csv", encoding="utf-8-sig")
# 평균 비중(리스크 패리티, 표본 밖)
pd.Series(B_rp[oos:].mean(0), index=cols).to_csv(RES / "rp_avg_weight.csv", encoding="utf-8-sig")
meta = dict(
    run_at=str(pd.Timestamp.now()), data_end=str(idx[end].date()), sim_start=str(idx[start].date()),
    oos_start=str(idx[oos].date()), cost=COST, thr=THR, n_assets=A, new_assets=new_assets,
    fixed_assets=[c for c, m in zip(cols, fixed_mask) if m], n_train=ntr,
    n_assets_by_year={str(y): int(F["elig"].loc[str(y)].iloc[0].sum()) for y in (1998, 2003, 2008, 2013, 2026)},
    price_breaks_fixed=0, jumps=info[info.jumps_25pct > 0].jump_dates.to_dict(),
)
cal_t.to_csv(RES / "ml_calibration_tradelabel.csv", encoding="utf-8-sig", index=False)
json.dump(dict(meta=meta, main=main, trades=tstats, diag=diag, diag_gbm=diag_g, diag_tradelabel=diag_t, sig_diag=sig_diag,
               ext_diag=ext_diag),
          open(RES / "metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
for k, v in main.items():
    m = v["표본 밖 2008-01~2026-11"]
    print(f"{k:10s} CAGR {m['CAGR']:.2%} vol {m['vol']:.2%} 샤프 {m['sharpe']:.2f} MDD {m['MDD']:.1%} 노출 {m['exposure']:.0%} 회전 {m['turnover']:.1f}"
          + (f" α {m['alpha']:.2%} t {m['alpha_t']:.2f}" if "alpha" in m else ""))
print(tstats)
print({k: round(v, 3) if isinstance(v, float) else v for k, v in diag.items()})
print(cal)
print(grp); print(nw)
print(pd.DataFrame(rob)[["label", "CAGR", "sharpe", "MDD", "exposure", "t_vs_rp", "t_vs_rule"]].round(3))
print(plc[["sharpe", "CAGR", "MDD", "auc", "t_vs_rule"]].describe().round(3))
print(coefs.mean().round(3))
print("가짜 학습 200일선", plc_ma[["sharpe", "auc", "t_vs_rule"]].describe().round(3))
for k in ("터틀 규칙", "터틀 + ML", "리스크 패리티", "60/40"):
    print(k, {p_: (round(v["sharpe"], 2), round(v["CAGR"], 4), round(v["MDD"], 3)) for p_, v in main[k].items()})
print("ntrain", ntr)
print("변수 확장", {k: {a: round(b, 3) for a, b in v.items()} for k, v in ext_diag.items()})
print(plc_ext.groupby("model")[["auc", "sharpe", "t_vs_rule"]].describe().round(3).T)
print(pd.read_csv(RES / "ml_coefs_ext.csv", index_col=0).agg(["mean", "std"]).T.round(3))
print("거래손익 정답", {k: round(v, 3) for k, v in diag_t.items()})
print("월간 신호", {k: {a: round(b, 3) for a, b in v.items()} for k, v in sig_diag.items()})
print(yr.loc[2007:].round(3))
print(pd.Series(B_rp[oos:].mean(0), index=cols).groupby(groups).sum().round(3))
print(pd.Series(B_rp[oos:].mean(0), index=cols).round(3).sort_values(ascending=False).head(8))
print(f"끝 {time.time()-t0:.0f}초")
