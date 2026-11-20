# -*- coding: utf-8 -*-
"""③ 우량주 필터 - 유니버스·필터·진입 (run.py와 robust.py가 같이 씀, 규칙은 전략_명세.md 그대로)

재무 사용 시점: d일(월말)에 쓸 수 있는 연간 재무 = d가 4월 말 이후면 전년도, 그 전이면 전전년도(공시 이후만).
세금 납부: 2015 회계연도부터 실제 계정(법인세비용 > 0, 현금흐름표 법인세 납부 > 0, 유효세율 5~40%),
           2013~2014는 세금 계정이 없어 대용(순이익·영업이익·영업현금흐름 > 0).
"""
import numpy as np
import pandas as pd

import data as D
import backtest as B

SCEN = ["통상", "분포30", "분포50", "분포70", "업종상대", "CRSP가치"]
ENTRY = ["200일선 위", "%R 과매도 탈출"]


def fy_for(d):
    d = pd.Timestamp(d)
    return d.year - 1 if d.month >= 4 else d.year - 2


class Quality:
    def __init__(self, W, info, sector_col="sector"):
        self.C, self.H, self.L, self.R = W["c"], W["h"], W["l"], W["ret"]
        self.info = info
        self.sector_col = sector_col
        self.fs, self.n_fill = D.fundamentals_filled()
        self.tx = D.tax_table()
        C, H, L = self.C, self.H, self.L
        ma200 = C > C.rolling(200, min_periods=150).mean()
        hh, ll = H.rolling(10).max(), L.rolling(10).min()
        wr = (hh - C) / (hh - ll) * -100
        self.entries = {"200일선 위": ma200, "%R 과매도 탈출": (wr > -80) & (wr.shift(1) <= -80)}
        self._mc = {}
        self.qs = None
        self.missing_paid = 0

    def month_ends(self, month, y0=2014, y1=2026):
        cal = self.C.index
        out = []
        for y in range(y0, y1 + 1):
            m = cal[(cal.year == y) & (cal.month == month)]
            if len(m) and fy_for(m[-1]) >= 2013:
                out.append(m[-1])
        return out

    def mcaps(self, dates):
        need = [d for d in dates if d not in self._mc]
        if need:
            mc = D.market_caps(sorted(set(need)), self.C)
            for d in mc.index:
                self._mc[d] = mc.loc[d]
        return {d: self._mc[d] for d in dates}

    def universe(self, d, mc_row, tax_mode="real"):
        fy = fy_for(d)
        C = self.C
        live = C.loc[:d].iloc[-5:].notna().any() & C.loc[d].notna()
        codes = live.index[live]
        fs = self.fs
        f = fs.xs(fy, level="fy") if fy in fs.index.get_level_values("fy") else pd.DataFrame(columns=fs.columns)
        df = pd.DataFrame(index=codes)
        df["mcap"] = mc_row.reindex(codes)
        df = df.join(f[["liab", "equity", "ni", "sales", "op", "cfo", "pretax", "src"]], how="inner")
        sc = self.info[[self.sector_col, "sector_src", "fin", "market"]].rename(columns={self.sector_col: "sector"})
        df = df.join(sc, how="left")
        df["fin"] = df["fin"].fillna(False).astype(bool)
        df = df[~df.fin & df.mcap.notna() & (df.mcap > 0)]
        df = df[df.mcap > df.mcap.quantile(0.20)]
        df["de"] = (df.liab / df.equity).where(df.equity > 0)
        df["per"] = df.mcap / df.ni
        df["psr"] = (df.mcap / df.sales).where(df.sales > 0)
        df["bp"], df["ep"], df["sp"] = df.equity / df.mcap, df.ni / df.mcap, df.sales / df.mcap
        df["fy"] = fy
        if fy >= 2015 and tax_mode != "proxy":
            t = self.tx.xs(fy, level="fy").reindex(df.index) if fy in self.tx.index.get_level_values("fy") else pd.DataFrame(index=df.index, columns=["tax_exp", "tax_paid", "pretax"])
            eff = (t.tax_exp / t.pretax).astype(float)
            exp_ok = (t.tax_exp > 0) & eff.between(0.05, 0.40)
            df["tax_exp_ok"] = exp_ok.fillna(False)
            df["tax_paid"] = t.tax_paid
            if tax_mode == "expense_only":                   # 후보 고를 때(납부액 수집 전)
                df["tax_ok"] = df.tax_exp_ok
            else:
                # 납부액은 후보 종목만 받았으므로, 없는 종목은 비용 조건으로 판단(순위 비교 대상이 줄지 않게).
                # 후보에 들 수 있는 종목은 모두 납부액을 받았기 때문에 실제 선택에는 납부 조건이 그대로 적용된다.
                paid_ok = (t.tax_paid > 0) | t.tax_paid.isna()
                df["tax_ok"] = df.tax_exp_ok & paid_ok.fillna(True)
            df["tax_src"] = "실제"
        else:
            df["tax_ok"] = (df.ni > 0) & (df.op > 0) & (df.cfo > 0)
            df["tax_exp_ok"] = df["tax_ok"]
            df["tax_src"] = "대용"
        df["crsp"] = df[["bp", "ep", "sp"]].rank(pct=True).mean(axis=1)
        return df

    def set_thresholds(self, pool):
        self.qs = {q: dict(de=pool.de.quantile(q), per=pool.per[pool.per > 0].quantile(q), psr=pool.psr.quantile(q))
                   for q in [0.3, 0.5, 0.7]}

    def pick(self, df, scen, n=30, use_tax=True):
        if use_tax:
            df = df[df.tax_ok]
        df = df[df.de.notna() & df.psr.notna()]
        if scen == "통상":
            m = (df.de <= 1.0) & (df.per > 0) & (df.per <= 10) & (df.psr <= 1.0)
        elif scen.startswith("분포"):
            t = self.qs[{"분포30": 0.3, "분포50": 0.5, "분포70": 0.7}[scen]]
            m = (df.de <= t["de"]) & (df.per > 0) & (df.per <= t["per"]) & (df.psr <= t["psr"])
        elif scen == "업종상대":
            g = df[df.sector.notna()].copy()
            g["per_pos"] = g.per.where(g.per > 0)
            rk = g.groupby("sector")[["de", "per_pos", "psr"]].rank(pct=True)
            ok = (rk.de <= 0.3) & (rk.per_pos <= 0.3) & (rk.psr <= 0.3)
            m = pd.Series(False, index=df.index)
            m.loc[ok.index[ok]] = True
        elif scen == "CRSP가치":
            m = df.crsp >= df.crsp.quantile(0.70)
        return df[m].sort_values("crsp", ascending=False).head(n).index.tolist()

    def run(self, rebal_dates, end, scen, entry, rf, U=None, cost=B.COST, picks=None):
        """rebal_dates: 리밸런싱 날짜 목록, end: 마지막 날. 반환 (수익률, 회전율, 평균 종목 수, picks)"""
        if picks is None:
            picks = {d: self.pick(U[d], scen) for d in rebal_dates}
        sig = None if entry == "즉시" else self.entries[entry]
        r, turn, hold, _ = B.stock_portfolio(list(rebal_dates) + [end], picks, sig, self.R, rf, cost=cost)
        return r, turn, hold, picks
