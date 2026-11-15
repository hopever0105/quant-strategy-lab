# -*- coding: utf-8 -*-
"""한국판 중형 가치 - 종목 선정 (매년 4월 말, 연간 재무 공시 후)

사전 등록한 후보(결과 보고 나서 추가 금지):
- 중형 정의  CRSP: 시가총액 큰 순 누적 비중 70~85% 구간 (VOE가 따르는 CRSP 방식)
             KRX : 시가총액 101~300위 (한국거래소 코스피 중형주 지수 방식)
- 가치 기준  절반: 중형 안 가치 점수 상위 50% (CRSP 가치/성장 절반 나누기)
             상위30: 상위 30%
- 비중       시총: 시가총액 가중(ETF와 같음)  /  동일: 동일가중
- 가치 점수  장기판(2006~): 장부가/가격(B/P) 한 개 - PBR이 2006년부터 95% 이상 채워져 있음
             CRSP판(2014~): B/P·이익/가격(E/P)·매출/가격(S/P) 순위 평균 (CRSP 방법서 지표 중 국내 공시로 만들 수 있는 3개.
                            예상 이익/가격·배당/가격은 자료 없음)
대상: 그 시점 코스피 상장 비금융 종목(상장폐지 포함). 재무는 4월 말 이후 전년도만 사용.
"""
import numpy as np
import pandas as pd

MID = ["CRSP", "KRX"]
VAL = ["절반", "상위30"]
WGT = ["시총", "동일"]


def fy_for(d):
    d = pd.Timestamp(d)
    return d.year - 1 if d.month >= 4 else d.year - 2


def april_ends(cal, y0, y1):
    out = []
    for y in range(y0, y1 + 1):
        m = cal[(cal.year == y) & (cal.month == 4)]
        if len(m):
            out.append(m[-1])
    return out


class KrMidValue:
    def __init__(self, K):
        self.W, self.info, self.fs, self.fp = K["W"], K["info"], K["fs"], K["fp"]
        self.C = self.W["c"]
        o = pd.read_pickle(K["LD"].CACHE / "mcap_rebal_v2.pkl")    # 2014~2026 4월 말 시총(억원, 상장폐지 포함)
        self.mc26 = o["mcap"]
        self.kospi = set(self.info.index[self.info.market == "KOSPI"])
        self.fin = set(self.info.index[self.info.fin.fillna(False).astype(bool)])

    def table(self, d):
        """d(4월 말)의 코스피 비금융 생존 종목 표: mcap, bp, ep, sp"""
        d = pd.Timestamp(d)
        C = self.C
        live = C.loc[:d].iloc[-5:].notna().any() & C.loc[d].notna()
        codes = [c for c in live.index[live] if c in self.kospi and c not in self.fin]
        fp = self.fp[self.fp.date == d].set_index("code")
        if len(fp):
            df = pd.DataFrame(index=codes)
            df["mcap"] = fp.mcap.reindex(codes)
            df["bp"] = (1 / fp.PBR).where(fp.PBR > 0).reindex(codes)
        else:                                               # 2026-04: factor_panel 없음 → 시총 캐시 + 재무 자본
            df = pd.DataFrame(index=codes)
            df["mcap"] = self.mc26.loc[d].reindex(codes) if d in self.mc26.index else np.nan
            df["bp"] = np.nan
        fy = fy_for(d)
        if fy in self.fs.index.get_level_values("fy"):
            f = self.fs.xs(fy, level="fy").reindex(codes)
            m = df.mcap * 1e8
            df["ep"] = f.ni / m
            df["sp"] = (f.sales / m).where(f.sales > 0)
            bp_fs = (f.equity / m).where(f.equity > 0)
            df["bp_src"] = np.where(df.bp.notna(), "PBR", np.where(bp_fs.notna(), "재무", "없음"))
            df["bp"] = df.bp.fillna(bp_fs)
        else:
            df["ep"], df["sp"], df["bp_src"] = np.nan, np.nan, np.where(df.bp.notna(), "PBR", "없음")
        df = df[df.mcap.notna() & (df.mcap > 0)]
        return df.sort_values("mcap", ascending=False)

    @staticmethod
    def mid(df, how):
        s = df.mcap.sort_values(ascending=False)
        if how == "CRSP":
            cum = s.cumsum() / s.sum()
            prev = cum.shift(1).fillna(0)
            m = (prev >= 0.70) & (prev < 0.85)
            return df.loc[s.index[m.values]]
        rk = np.arange(1, len(s) + 1)
        return df.loc[s.index[(rk >= 101) & (rk <= 300)]]

    @staticmethod
    def large(df):
        s = df.mcap.sort_values(ascending=False)
        prev = (s.cumsum() / s.sum()).shift(1).fillna(0)
        return df.loc[s.index[(prev < 0.70).values]]

    @staticmethod
    def score(df, ver):
        if ver == "장기":
            return df.bp.rank(pct=True)
        r = pd.concat([df.bp.rank(pct=True), df.ep.rank(pct=True), df.sp.rank(pct=True)], axis=1)
        return r.mean(axis=1, skipna=False)

    def pick(self, d, mid, val, wgt, ver, base=None):
        df = self.table(d) if base is None else base
        g = self.mid(df, mid) if mid != "대형" else self.large(df)
        sc = self.score(g, ver).dropna()
        q = 0.5 if val == "절반" else 0.7
        if val == "전체":
            sel = g.index[g.mcap.notna()]
        else:
            sel = sc.index[sc >= sc.quantile(q)]
        w = g.mcap.reindex(sel) if wgt == "시총" else pd.Series(1.0, index=sel)
        return w / w.sum()
