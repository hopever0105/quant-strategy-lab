# -*- coding: utf-8 -*-
"""③ 우량주 필터에서 성과를 만든 종목 vs 떨어진 종목 비교 (매수 시점에 알 수 있던 값만)

그룹
  급등(기여 상위): 동일가중 이하로 떨어뜨리는 데 필요한 7건 (m3_contribution.csv 위에서 7건)
  하락: 1년 보유 수익 < 0
  나머지: 그 사이
지표
  재무: PER, PBR, PSR, 부채비율, ROE, 영업이익률, 매출·영업이익·순이익 성장률, 영업현금흐름/순이익, 유동비율, 재고/자산, 설비투자/매출, 시총
  기술: 12-1개월 모멘텀, 6개월·1개월 수익, 52주 고점 대비, 200일선 대비, 60일 변동성, 1년 베타, 최근 1개월 최대 일간 수익(MAX),
        20일 평균 거래대금, 거래대금/시총(회전율), %R(10)
검증: 그룹 차이가 이 7건만의 우연인지 보려고, 같은 지표가 전체 유니버스(매년 4월 말 비금융 1,300~1,900종목)에서
      다음 1년 수익을 예측했는지 순위상관(IC) 13년 평균과 t값을 함께 낸다.
실행: python winners.py → results/winners_*.csv
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D          # noqa: E402
import quality as QM      # noqa: E402

OUT = HERE / "results"


def tech_features(C, R, AMT, kos_r, d):
    """d일 기준 기술적 지표 (d일 종가까지)"""
    c = C.loc[:d]
    r = R.loc[:d]
    p0 = c.iloc[-1]
    f = pd.DataFrame(index=C.columns)
    f["모멘텀12_1"] = c.iloc[-22] / c.iloc[-253] - 1
    f["수익6개월"] = p0 / c.iloc[-127] - 1
    f["수익1개월"] = p0 / c.iloc[-22] - 1
    f["52주고점대비"] = p0 / c.iloc[-252:].max() - 1
    f["200일선대비"] = p0 / c.iloc[-200:].mean() - 1
    f["변동성60일"] = r.iloc[-60:].std() * np.sqrt(252)
    rr = r.iloc[-252:].astype(float)
    kk = kos_r.reindex(rr.index).fillna(0).values
    f["베타1년"] = rr.apply(lambda x: np.cov(x.fillna(0), kk)[0, 1] / kk.var() if x.notna().sum() > 120 else np.nan)
    f["최근1개월최대일간"] = r.iloc[-21:].max()
    a = AMT.loc[:d].iloc[-20:].mean() / 1e8
    f["거래대금20일_억"] = a
    hh, ll = C.loc[:d].iloc[-10:].max(), C.loc[:d].iloc[-10:].min()
    f["WR10"] = (hh - p0) / (hh - ll) * -100
    return f


def fund_features(fs, d, u):
    fy = QM.fy_for(d)
    cur = fs.xs(fy, level="fy") if fy in fs.index.get_level_values("fy") else pd.DataFrame()
    prv = fs.xs(fy - 1, level="fy") if fy - 1 in fs.index.get_level_values("fy") else pd.DataFrame(columns=fs.columns)
    f = pd.DataFrame(index=u.index)
    cu, pv = cur.reindex(u.index), prv.reindex(u.index)
    f["PER"], f["PSR"] = u.per, u.psr
    f["PBR"] = u.mcap / cu.equity
    f["부채비율"] = u.de
    f["ROE"] = cu.ni / cu.equity
    f["영업이익률"] = cu.op / cu.sales
    f["매출성장"] = cu.sales / pv.sales - 1
    f["영업이익성장"] = (cu.op - pv.op) / pv.op.abs()
    f["순이익성장"] = (cu.ni - pv.ni) / pv.ni.abs()
    f["현금흐름_순이익"] = cu.cfo / cu.ni
    f["유동비율"] = cu.ca / cu.cl
    f["재고_자산"] = cu.inv / cu.assets
    f["설비투자_매출"] = cu.capex.abs() / cu.sales
    f["시총_억"] = u.mcap
    f["거래대금_시총"] = np.nan
    return f


def main():
    W = D.panel_wide()
    info = D.listing_info()
    Q = QM.Quality(W, info)
    C, R = Q.C, Q.R
    pan = D.panel_long()[["d", "code", "amt"]] if D.USE_DELIST else pd.read_pickle(D.PANEL)[["d", "code", "amt"]]
    AMT = pan.pivot(index="d", columns="code", values="amt").reindex(C.index).astype("float32")
    del pan
    kos, _ = D.kospi()
    kos_r = kos["c"].pct_change()
    fs = Q.fs
    apr = Q.month_ends(4)
    end = C.index[-1]
    Q.set_thresholds(pd.concat([Q.universe(d, Q.mcaps([d])[d]) for d in apr if d.year <= 2017]))
    con = pd.read_csv(OUT / "m3_contribution.csv", dtype={"코드": str})
    k = int(pd.read_csv(OUT / "m3_contribution_summary.csv", index_col=0).iloc[:, 0]["필요종목수"])   # 동일가중 이하로 떨어뜨리는 데 필요한 수
    top = set(zip(con.head(k).리밸런싱, con.head(k).코드))

    picks_rows, ic_rows = [], []
    for a, b in zip(apr, apr[1:] + [end]):
        u = Q.universe(a, Q.mcaps([a])[a])
        tf = tech_features(C, R, AMT, kos_r, a).reindex(u.index)
        ff = fund_features(fs, a, u)
        ff["거래대금_시총"] = tf["거래대금20일_억"] / u.mcap
        F = pd.concat([ff, tf], axis=1)
        days = R.index[(R.index > a) & (R.index <= b)]
        fwd = (1 + R.loc[days, u.index].fillna(0).astype(float)).prod() - 1
        F["다음1년수익"] = fwd
        F["바이오제약"] = u.sector.isin(["건강관리", "제약"]).astype(float)
        # 전체 유니버스 IC (지표 → 다음 1년 수익, 순위상관)
        for col in F.columns.drop("다음1년수익"):
            x = F[[col, "다음1년수익"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(x) > 50:
                ic_rows.append(dict(시작=a.date(), 지표=col, IC=stats.spearmanr(x.iloc[:, 0], x.iloc[:, 1])[0]))
        sub = con[con.리밸런싱 == str(a.date())]
        for _, rw in sub.iterrows():
            if rw.코드 in F.index:
                row = F.loc[rw.코드].to_dict()
                row.update(리밸런싱=rw.리밸런싱, 종목명=rw.종목명, 업종=rw.업종, 보유수익=rw.보유수익, 초과기여=rw.초과기여,
                           그룹="급등(기여 상위)" if (rw.리밸런싱, rw.코드) in top else ("하락" if rw.보유수익 < 0 else "나머지"))
                picks_rows.append(row)
    PK = pd.DataFrame(picks_rows)
    PK.to_csv(OUT / "winners_features.csv", index=False, encoding="utf-8-sig")
    feats = [c for c in PK.columns if c not in ("리밸런싱", "종목명", "업종", "보유수익", "초과기여", "그룹", "다음1년수익")]
    PK[feats] = PK[feats].replace([np.inf, -np.inf], np.nan)
    med = PK.groupby("그룹")[feats].median().T
    med["전체 픽 중앙값"] = PK[feats].median()
    # 픽 안에서 지표와 초과기여의 순위상관(171건)
    sp = {f: stats.spearmanr(PK[f], PK["초과기여"], nan_policy="omit") for f in feats}
    med["픽 내 순위상관"] = [sp[f][0] for f in feats]
    med["픽 내 p값"] = [sp[f][1] for f in feats]
    # 급등 7 vs 하락: 순위합 검정
    g1, g2 = PK[PK.그룹 == "급등(기여 상위)"], PK[PK.그룹 == "하락"]
    med["급등vs하락 p값"] = [stats.mannwhitneyu(g1[f].dropna(), g2[f].dropna()).pvalue if g1[f].notna().sum() > 2 and g2[f].notna().sum() > 2 else np.nan for f in feats]
    IC = pd.DataFrame(ic_rows)
    ics = IC.groupby("지표").IC.agg(["mean", "std", "count"])
    ics["t"] = ics["mean"] / (ics["std"] / np.sqrt(ics["count"]))
    ics["양수해비율"] = IC.groupby("지표").IC.apply(lambda x: (x > 0).mean())
    med = med.join(ics[["mean", "t", "양수해비율"]].rename(columns={"mean": "전체시장 IC평균", "t": "전체시장 IC t", "양수해비율": "IC 양수 해 비율"}))
    med.to_csv(OUT / "winners_compare.csv", encoding="utf-8-sig")
    IC.to_csv(OUT / "winners_ic_by_year.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(PK.그룹.value_counts().to_dict())
    print(med.to_string())
    print(PK[PK.그룹.str.startswith("급등")][["리밸런싱", "종목명"] + feats].T.to_string())


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
