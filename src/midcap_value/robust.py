# -*- coding: utf-8 -*-
"""강건성 검증 (python robust.py, run.py 다음에)

1. 사후 발견 '10년 금리가 오른 뒤 다음 달엔 중형 가치가 이긴다' 검증
   - 결과를 본 뒤 생긴 가설이므로 사전 등록 조합이 아님 → 보고서에 '사후 발견'으로 표시
   - ① 구간 나누기(2001~2014 / 2015~2026) ② 금리 변화 기간 {1, 3, 6}개월 ③ 진짜 표본 밖: 1963~2000 파마-프렌치 팩터로
     만든 '중형 가치 − S&P500' 합성 수익(VOE·SPY 팩터 노출 차이 × 팩터) ④ 전환 규칙(IWS ↔ SPY/QQQ)·비용 2배
2. Fidelity 액티브 vs 패시브: 구간별 6팩터 알파, 같은 지수 ETF(IWS) 대비 알파
3. 한국: 중형 구간 경계 바꾸기, 3년 블록 방향, 금리(미국 10년·한국 3개월) 상태별 다음 달 상대수익
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import data as DT
import backtest as B
import signals as S

OUT = Path(__file__).resolve().parent / "results"


def save(df, name):
    df.to_csv(OUT / f"{name}.csv", encoding="utf-8-sig")
    return df


def mret(r):
    return (1 + r.dropna()).resample("ME").prod() - 1


def rate_regime(R, ff, rf):
    y10 = DT.dgs10().resample("ME").last()
    M = pd.DataFrame({t: mret(R[t]) for t in ["IWS", "VOE", "SPY", "QQQ"]})
    rows = []
    for b in ["SPY", "QQQ"]:
        rel = (M["IWS"] - M[b]).dropna()
        nxt = rel.shift(-1)
        for k in [1, 3, 6]:
            up = (y10.diff(k) > 0).reindex(rel.index)
            for nm, a, z in [("2001~2014", "2001-01-01", "2014-12-31"), ("2015~2026", "2015-01-01", "2026-12-31"),
                             ("전체", "2001-01-01", "2026-12-31")]:
                m = nxt[a:z].notna() & up[a:z].notna()
                x, c = nxt[a:z][m], up[a:z][m].astype(bool)
                d, t, n = B.diff_t(x, c, 3)
                rows.append(dict(대상=f"IWS−{b}", 금리기간=f"{k}개월", 구간=nm, 상승달=n, 전체달=len(x),
                                 상승_다음달=x[c].mean() * 12, 하락_다음달=x[~c].mean() * 12, 차이=d * 12, t=t))
    # 표본 밖: 1963~2000 합성 '중형 가치 − S&P500' = (VOE 노출 − SPY 노출) × 팩터 (일간 → 월간)
    rv = B.factor_reg(R["VOE"], ff)
    rs = B.factor_reg(R["SPY"], ff)
    fm = (1 + ff[B.FF6 + ["RF"]]).resample("ME").prod() - 1
    syn = sum((rv[c] - rs[c]) * fm[c] for c in B.FF6)
    for k in [1, 3, 6]:
        up = (y10.diff(k) > 0).reindex(syn.index)
        nxt = syn.shift(-1)
        for nm, a, z in [("1963~2000(표본 밖)", "1963-01-01", "2000-12-31"), ("2001~2026", "2001-01-01", "2026-12-31")]:
            m = nxt[a:z].notna() & up[a:z].notna()
            x, c = nxt[a:z][m], up[a:z][m].astype(bool)
            d, t, n = B.diff_t(x, c, 3)
            rows.append(dict(대상="합성(VOE−SPY 팩터 노출)", 금리기간=f"{k}개월", 구간=nm, 상승달=n, 전체달=len(x),
                             상승_다음달=x[c].mean() * 12, 하락_다음달=x[~c].mean() * 12, 차이=d * 12, t=t))
        for f in ["HML", "SMB"]:
            nx = fm[f].shift(-1)
            m = nx["1963":"2000"].notna() & up["1963":"2000"].notna()
            x, c = nx["1963":"2000"][m], up["1963":"2000"][m].astype(bool)
            d, t, n = B.diff_t(x, c, 3)
            rows.append(dict(대상=f"{f} 팩터", 금리기간=f"{k}개월", 구간="1963~2000(표본 밖)", 상승달=n, 전체달=len(x),
                             상승_다음달=x[c].mean() * 12, 하락_다음달=x[~c].mean() * 12, 차이=d * 12, t=t))
    reg = save(pd.DataFrame(rows), "r_rate_regime")
    syn_w = pd.Series({c: rv[c] - rs[c] for c in B.FF6}, name="VOE−SPY 노출 차이")
    save(syn_w.to_frame(), "r_synthetic_weights")

    # 전환 규칙: 월말에 10년 금리가 3개월 전보다 높으면 다음 달 IWS, 아니면 비교 ETF
    sw = []
    up3 = (y10.diff(3) > 0).astype(float).where(y10.diff(3).notna())
    for b in ["SPY", "QQQ"]:
        d = pd.concat([R["IWS"], R[b]], axis=1).dropna()
        for cost in [0.0005, 0.001]:
            r, n = B.switch_portfolio(up3[up3.index >= d.index[0]], d["IWS"], d[b], cost)
            for s in ["학습", "검증", "테스트", "전체"]:
                rs_ = B.seg(r, s)
                a, t, _ = B.capm_alpha(rs_, d[b], rf)
                p = B.perf(rs_, rf)
                pb = B.perf(B.seg(d[b][r.index], s), rf)
                sw.append(dict(비교=b, 비용=cost, 구간=s, 전환수=n, CAGR=p["CAGR"], 샤프=p["샤프"], MDD=p["MDD"],
                               비교_CAGR=pb["CAGR"], 비교_샤프=pb["샤프"], 알파=a, t=t))
    save(pd.DataFrame(sw), "r_rate_switch")
    return reg


def fidelity(R, ff, rf):
    rows = []
    for t in ["FSMVX", "FLPSX"]:
        for s in ["전체", "학습", "검증", "테스트"]:
            o = B.factor_reg(B.seg(R[t], s), ff)
            a, tt, beta = B.capm_alpha(B.seg(R[t], s), R["IWS"], rf)
            rows.append(dict(펀드=t, 구간=s, 알파6팩터=o.get("알파(연)"), t6=o.get("알파t"),
                             알파_vsIWS=a, t_vsIWS=tt, 베타_vsIWS=beta))
    return save(pd.DataFrame(rows), "r_fidelity")


def korea():
    K = DT.kr_inputs()
    KM = S.KrMidValue(K)
    R = K["W"]["ret"]
    cal = R.index
    end = cal[-1]
    kospi = K["kospi"]["c"].pct_change().reindex(cal)
    rf = K["LD"].cash_rates(cal, cal)[0].reindex(cal).fillna(0.0)
    dates = S.april_ends(cal, 2006, 2026)
    tabs = {d: KM.table(d) for d in dates}
    rows, keep = [], {}
    # 중형 경계 바꾸기 (CRSP 70~85% 기본 → 65~85, 70~90, 60~80)
    orig = KM.mid
    for lo, hi in [(0.70, 0.85), (0.65, 0.85), (0.70, 0.90), (0.60, 0.80)]:
        def mid(df, how, lo=lo, hi=hi):
            s = df.mcap.sort_values(ascending=False)
            prev = (s.cumsum() / s.sum()).shift(1).fillna(0)
            return df.loc[s.index[((prev >= lo) & (prev < hi)).values]]
        KM.mid = mid
        for v in ["상위30", "전체"]:
            W = {d: KM.pick(d, "CRSP", v, "동일", "장기", tabs[d]) for d in dates}
            r, turn, nh, _ = B.weight_portfolio(list(dates) + [end], W, R, 0.0025)
            keep[(lo, hi, v)] = r
            for s in ["학습", "검증", "테스트", "전체"]:
                rs = B.seg(r, s)
                a, t, _ = B.capm_alpha(rs, kospi, rf)
                p = B.perf(rs, rf)
                rows.append(dict(경계=f"{int(lo*100)}~{int(hi*100)}%", 가치=v, 구간=s, CAGR=p["CAGR"], 샤프=p["샤프"],
                                 알파=a, t=t, 평균종목수=nh))
    KM.mid = orig
    save(pd.DataFrame(rows), "r_kr_band")

    # 3년 블록: 선택 조합(장기|CRSP|상위30|동일) − 코스피, − 중형 전체
    rets = pd.read_pickle(OUT / "d_kr_returns.pkl")
    ch = rets["장기|CRSP|상위30|동일"]
    mid_all = rets["장기|CRSP|전체|시총"]
    blk = []
    for y0 in range(2006, 2027, 3):
        a, z = f"{y0}-05-01", f"{y0 + 2}-12-31"
        for nm, b in [("코스피", kospi), ("중형 전체", mid_all)]:
            x = ch[a:z]
            bb = b.reindex(x.index)
            if len(x) < 100:
                continue
            blk.append(dict(블록=f"{y0}~{min(y0 + 2, 2026)}", 비교=nm,
                            연차이=((1 + x).prod() ** (252 / len(x)) - (1 + bb.fillna(0)).prod() ** (252 / len(x)))))
    save(pd.DataFrame(blk), "r_kr_blocks")

    # 금리 상태별 다음 달 상대수익 (한국 중형 가치 − 코스피)
    LD = K["LD"]
    y10 = DT.dgs10().resample("ME").last()
    kr3 = LD.fred("IR3TIB01KRM156N").resample("ME").last()
    rel = (mret(ch) - mret(kospi)).dropna()
    nxt = rel.shift(-1)
    out = []
    for nm, s in [("미국 10년 3개월 상승", y10), ("한국 3개월 금리 3개월 상승", kr3)]:
        up = (s.diff(3) > 0).reindex(rel.index)
        m = nxt.notna() & up.notna()
        d, t, n = B.diff_t(nxt[m], up[m].astype(bool), 3)
        out.append(dict(상태=nm, 상승달=n, 전체달=int(m.sum()), 상승_다음달=nxt[m][up[m].astype(bool)].mean() * 12,
                        하락_다음달=nxt[m][~up[m].astype(bool)].mean() * 12, 차이=d * 12, t=t))
    save(pd.DataFrame(out), "r_kr_rate")


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    R = DT.us_returns()
    ff = DT.ff6_daily()
    rf = ff["RF"].reindex(R.index).ffill()
    print(rate_regime(R, ff, rf).round(3).to_string())
    print(pd.read_csv(OUT / "r_rate_switch.csv").round(3).to_string())
    print(fidelity(R, ff, rf).round(3).to_string())
    korea()
    for f in ["r_kr_band", "r_kr_blocks", "r_kr_rate"]:
        print(pd.read_csv(OUT / f"{f}.csv").round(3).to_string())


if __name__ == "__main__":
    main()
