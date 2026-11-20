# -*- coding: utf-8 -*-
"""유튜브 전략 시봉 재검증 - 전체 실행 (같은 입력 → 같은 결과, 시드 고정)
결과: results/*.csv, *.png, summary.json
"""
import json
import time

import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data_h
import strategies_h as sh
import kullamaggie_h as kh

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
RES = data_h.HERE / "results"
RES.mkdir(exist_ok=True)
SEED = 20261113
SPLIT = "2025-01-01"


def P(*a):
    print(*a, flush=True)
    data_h.log(" ".join(str(x) for x in a))


def tstat(x):
    x = pd.Series(x).dropna()
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan


def stats(tr, rt=None):
    if tr is None or not len(tr):
        return dict(거래수=0)
    d = dict(거래수=len(tr), 승률=(tr.ret > 0).mean(), 평균수익=tr.ret.mean(), 평균R=tr.R.mean(), R_t=tstat(tr.R),
             평균보유봉=tr.hold_bars.mean(), 평균보유일=tr.hold_days.mean(),
             앞구간평균R=tr[tr.day < np.datetime64(SPLIT)].R.mean(), 뒤구간평균R=tr[tr.day >= np.datetime64(SPLIT)].R.mean())
    s = tr.R.sort_values(ascending=False)
    k = max(1, int(len(s) * 0.05))
    d["상위5pct제외평균R"] = s.iloc[k:].mean()
    if rt is not None and len(rt):
        dm = rt.groupby("draw").R.mean()
        d.update(무작위평균R=dm.mean(), 무작위_5pct=dm.quantile(0.05), 무작위_95pct=dm.quantile(0.95),
                 백분위=float((dm < d["평균R"]).mean()), 무작위평균수익=rt.ret.mean())
    return d


def main():
    t0 = time.time()
    B = data_h.build()
    P("시봉 대조", B["check"])
    info = pd.read_csv(data_h.QUANT / "cache" / "larry" / "listing_info_v4.csv", dtype={"code": str}).set_index("code")["market"].to_dict()
    S = {"일봉": sh.stock_arrays(B, "D", info), "시봉": sh.stock_arrays(B, "H", info)}
    P("종목", {k: len(v) for k, v in S.items()})
    rows, trades = [], {}
    for fq, SS in S.items():
        rng = np.random.default_rng(SEED)
        for mode in (1, 2):
            tr, rt = sh.ross_trades(SS, mode, rand=True, rng=rng)
            nm = f"로스 매매법 {mode}"
            rows.append(dict(전략=nm, 봉=fq, **stats(tr, rt)))
            trades[(nm, fq)] = tr
        for v in sh.OBV_VARIANTS:
            tr, rt = sh.obv_trades(SS, v, rand=True, rng=rng)
            nm = f"OBV·켈트너 {v}"
            rows.append(dict(전략=nm, 봉=fq, **stats(tr, rt)))
            trades[(nm, fq)] = tr
        for m in ("N", "R", "S"):
            tr, rt = sh.ha_trades(SS, m, rand=True, rng=rng)
            nm = f"하이킨아시 {m}"
            rows.append(dict(전략=nm, 봉=fq, **stats(tr, rt)))
            trades[(nm, fq)] = tr
        P(fq, "끝", f"{time.time() - t0:.0f}초")
    T = pd.DataFrame(rows)
    T.to_csv(RES / "youtube_daily_vs_hourly.csv", index=False, encoding="utf-8-sig")
    for (nm, fq), tr in trades.items():
        tr.to_csv(RES / f"trades_{nm.replace(' ', '_').replace('·', '')}_{fq}.csv", index=False, encoding="utf-8-sig")
    P(T[["전략", "봉", "거래수", "승률", "평균수익", "평균R", "R_t", "무작위평균R", "백분위"]].round(3).to_string())

    # 쿨라마기 진입일 시봉 확인
    ktr, nfb = kh.run(B)
    km = kh.daily_modes(B)
    krows = [dict(처리="시봉으로 진입일 확인", 시봉없어_일봉가정=nfb, **_kstats(ktr))]
    for nm, t in km.items():
        krows.append(dict(처리=nm, **_kstats(t)))
    ktr_g, _ = kh.run(B, cost=0.0)
    krows.append(dict(처리="시봉 확인, 비용 0(참고)", **_kstats(ktr_g)))
    K = pd.DataFrame(krows)
    K.to_csv(RES / "kullamaggie_entry_check.csv", index=False, encoding="utf-8-sig")
    ktr.to_csv(RES / "kullamaggie_hourly_trades.csv", index=False, encoding="utf-8-sig")
    P(K.round(3).to_string())

    summary = dict(check=B["check"], 종목수={k: len(v) for k, v in S.items()}, 기간=[sh.WARM_START, data_h.H_END], 쿨라마기_시봉없음=nfb)
    (RES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    charts(T, K)
    P(f"완료 {time.time() - t0:.0f}초")


def _kstats(t):
    if not len(t):
        return dict(거래수=0)
    s = t.R.sort_values(ascending=False)
    k = max(1, int(len(s) * 0.05))
    return dict(거래수=len(t), 승률=(t.ret > 0).mean(), 평균수익=t.ret.mean(), 평균R=t.R.mean(), R_t=tstat(t.R),
                진입일손절비율=(t.t_exit == t.t).mean(), 평균보유일=t.hold.mean(), 상위5pct제외평균R=s.iloc[k:].mean())


def charts(T, K):
    main = ["로스 매매법 1", "로스 매매법 2", "OBV·켈트너 A5", "OBV·켈트너 B", "하이킨아시 R", "하이킨아시 S"]
    t = T[T.전략.isin(main)]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(main))
    for k, (fq, col) in enumerate([("일봉", "#7f8c8d"), ("시봉", "#1f4e79")]):
        v = [t[(t.전략 == m) & (t.봉 == fq)].평균R.iloc[0] for m in main]
        rnd = [t[(t.전략 == m) & (t.봉 == fq)].무작위평균R.iloc[0] for m in main]
        ax.bar(x + (k - 0.5) * 0.38, v, 0.36, color=col, label=f"{fq} 신호")
        ax.scatter(x + (k - 0.5) * 0.38, rnd, color="#e67e22", marker="_", s=300, zorder=3,
                   label="무작위 진입(같은 종목·같은 달)" if k == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(main, fontsize=9)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("거래당 평균 R (비용 차감)")
    ax.set_title("일봉 vs 시봉: 같은 규칙, 같은 기간(2023-12~2026-10), 같은 종목")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(RES / "daily_vs_hourly.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    k = K[~K.처리.str.contains("비용 0")]
    ax.barh(range(len(k)), k.평균R, color=["#1f4e79"] + ["#7f8c8d"] * (len(k) - 1))
    ax.set_yticks(range(len(k)))
    ax.set_yticklabels(k.처리, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6)
    for i, (v, n) in enumerate(zip(k.평균R, k.거래수)):
        ax.text(v, i, f" {v:+.3f}R ({int(n):,}건)", va="center", fontsize=8)
    ax.set_title("쿨라마기: 진입일 처리별 거래당 평균 R (2023-11~2026-10)")
    fig.tight_layout()
    fig.savefig(RES / "kullamaggie_entry_check.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
