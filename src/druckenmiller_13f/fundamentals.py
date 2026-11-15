# -*- coding: utf-8 -*-
"""XBRL 사실 → 시점 t0에 '이미 공시된' 재무 지표 (미래정보 금지).

규칙
- filed <= t0 인 사실만 사용. 같은 기간(start,end)이 여러 번 공시되면 t0 이전 가장 늦은 공시값(정정 반영).
- 손익·현금흐름(기간 값): 분기(80~100일)·반기·3분기 누적·연간 값을 조합해 분기 값을 만든다.
  Q4 = 연간 - 9개월 누적, 2분기 = 6개월 누적 - 1분기 등.
- 최근 4분기 합계(TTM), 전년 같은 4분기 대비 성장률, 최근 분기 전년 동기 대비 성장률과 그 직전 분기와의 차이(가속).
- 재무상태(시점 값): t0 이전 공시된 가장 최근 값.
- 통화: 비율(성장률·이익률·ROE·부채비율)은 통화와 무관하게 계산. 시가총액 기반 지표는 USD일 때만.
"""
import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd

QUANT = Path(r"C:\Users\arlo0\OneDrive\바탕 화면\취업\quant")
FS = QUANT / "cache" / "druck13f" / "fs"

REV = ["us-gaap:Revenues", "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
       "us-gaap:SalesRevenueNet", "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
       "us-gaap:SalesRevenueGoodsNet", "ifrs-full:Revenue"]
GP = ["us-gaap:GrossProfit", "ifrs-full:GrossProfit"]
OP = ["us-gaap:OperatingIncomeLoss", "ifrs-full:ProfitLossFromOperatingActivities"]
NI = ["us-gaap:NetIncomeLoss", "ifrs-full:ProfitLoss"]
CFO = ["us-gaap:NetCashProvidedByUsedInOperatingActivities", "ifrs-full:CashFlowsFromUsedInOperatingActivities"]
CAPEX = ["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"]
RND = ["us-gaap:ResearchAndDevelopmentExpense"]
EQ = ["us-gaap:StockholdersEquity", "ifrs-full:Equity"]
AS = ["us-gaap:Assets", "ifrs-full:Assets"]
LI = ["us-gaap:Liabilities", "ifrs-full:Liabilities"]
DEBT = ["us-gaap:LongTermDebt", "us-gaap:LongTermDebtNoncurrent"]
DEBTC = ["us-gaap:DebtCurrent"]
CASH = ["us-gaap:CashAndCashEquivalentsAtCarryingValue"]
SHS = ["dei:EntityCommonStockSharesOutstanding", "us-gaap:CommonStockSharesOutstanding"]


def load_facts(cik):
    p = FS / f"{int(cik)}.csv.gz"
    if not p.exists():
        return None
    d = pd.read_csv(io.BytesIO(gzip.decompress(p.read_bytes())))
    if d.empty:
        return None
    d = d[d["form"].astype(str).str.match(r"^(10-K|10-Q|20-F|40-F|10-KT|10-QT|6-K|8-K)")]
    d["filed"] = pd.to_datetime(d["filed"], errors="coerce")
    d["end"] = pd.to_datetime(d["end"], errors="coerce")
    d["start"] = pd.to_datetime(d["start"], errors="coerce")
    d = d.dropna(subset=["filed", "end", "val"])
    d["dur"] = (d["end"] - d["start"]).dt.days
    return d


class Company:
    def __init__(self, cik):
        self.d = load_facts(cik)
        self.ok = self.d is not None and len(self.d) > 0
        if self.ok:
            # 손익 계정은 주 통화 한 가지만
            money = self.d[self.d["unit"].str.len() == 3]
            self.cur = money["unit"].mode().iloc[0] if len(money) else "USD"

    def _pick(self, concepts, t0, duration):
        d = self.d
        m = d["concept"].isin(concepts) & (d["filed"] <= t0)
        if duration:
            m &= d["dur"].notna() & (d["unit"] == self.cur)
        else:
            m &= d["dur"].isna()
        x = d[m]
        if x.empty:
            return x
        x = x.assign(pri=x["concept"].map({c: i for i, c in enumerate(concepts)}))
        key = ["concept", "start", "end"] if duration else ["concept", "end"]
        # 같은 계정·같은 기간은 t0 이전 가장 늦은 공시값(정정 반영)
        x = x.sort_values("filed").groupby(key, dropna=False).tail(1)
        # 같은 기간에 여러 계정이 있으면 우선순위 높은 계정
        x = x.sort_values("pri").drop_duplicates(["start", "end"] if duration else ["end"], keep="first")
        return x

    def quarters(self, concepts, t0):
        """분기 값 시리즈(index=분기말). 누적값 차감으로 분기 복원."""
        x = self._pick(concepts, t0, True)
        if x.empty:
            return pd.Series(dtype=float)
        q = {}
        x = x.sort_values("end")
        direct = x[(x["dur"] >= 75) & (x["dur"] <= 105)]
        for _, r in direct.iterrows():
            q[r["end"]] = r["val"]
        # 누적값(6·9·12개월)에서 분기 복원: 같은 시작일의 더 짧은 누적과 차이
        cum = x[(x["dur"] > 105) & (x["dur"] <= 380)]
        for _, r in cum.iterrows():
            if r["end"] in q:
                continue
            same = x[(x["start"] == r["start"]) & (x["end"] < r["end"])]
            if len(same):
                prev = same.sort_values("end").iloc[-1]
                if 75 <= (r["end"] - prev["end"]).days <= 105:
                    q[r["end"]] = r["val"] - prev["val"]
                    continue
            # 분기 3개로 연간에서 4분기 복원
            if 350 <= r["dur"] <= 380:
                ins = [k for k in q if r["start"] < k < r["end"]]
                if len(ins) == 3:
                    q[r["end"]] = r["val"] - sum(q[k] for k in ins)
        s = pd.Series(q).sort_index()
        return s

    def annual(self, concepts, t0):
        x = self._pick(concepts, t0, True)
        if x.empty:
            return pd.Series(dtype=float)
        a = x[(x["dur"] >= 350) & (x["dur"] <= 380)]
        return pd.Series(a["val"].values, index=a["end"].values).sort_index()

    def instant(self, concepts, t0, unit_money=True):
        x = self._pick(concepts, t0, False)
        if unit_money:
            x = x[x["unit"] == self.cur]
        if x.empty:
            return np.nan, None
        r = x.sort_values("end").iloc[-1]
        return r["val"], r["end"]


def ttm(qs, end=None):
    """최근 4개 연속 분기 합(분기 간격 75~105일). end 지정 시 그 분기까지."""
    if len(qs) < 4:
        return np.nan, None
    s = qs if end is None else qs[:end]
    if len(s) < 4:
        return np.nan, None
    last4 = s.iloc[-4:]
    gaps = np.diff(last4.index.values).astype("timedelta64[D]").astype(int)
    if (gaps < 70).any() or (gaps > 110).any():
        return np.nan, None
    return last4.sum(), last4.index[-1]


def flow_metrics(co, concepts, t0, tag):
    out = {}
    q = co.quarters(concepts, t0)
    v, e = ttm(q)
    if not np.isfinite(v):
        a = co.annual(concepts, t0)
        if len(a):
            v, e = a.iloc[-1], a.index[-1]
    out[f"{tag}_ttm"] = v
    out[f"{tag}_end"] = e
    if e is not None and len(q) >= 8:
        prev_end = q.index[q.index <= e - pd.Timedelta(days=330)]
        if len(prev_end):
            pv, _ = ttm(q, prev_end[-1])
            if np.isfinite(pv) and pv != 0:
                out[f"{tag}_g_ttm"] = v / abs(pv) - 1 if pv > 0 else np.nan
    # 최근 분기 전년동기 대비, 직전 분기 전년동기 대비(가속)
    if len(q) >= 5:
        def yoy(k):
            e1 = q.index[k]
            cand = q[(q.index >= e1 - pd.Timedelta(days=380)) & (q.index <= e1 - pd.Timedelta(days=350))]
            if len(cand) and cand.iloc[-1] > 0:
                return q.iloc[k] / cand.iloc[-1] - 1
            return np.nan
        out[f"{tag}_g_q"] = yoy(-1)
        if len(q) >= 6:
            out[f"{tag}_g_q_prev"] = yoy(-2)
    return out


def snapshot(co, t0):
    """t0 시점 재무 스냅샷(통화 무관 비율 + USD 금액)."""
    if not co.ok:
        return {}
    t0 = pd.Timestamp(t0)
    r = {"fs_cur": co.cur}
    for concepts, tag in [(REV, "rev"), (GP, "gp"), (OP, "op"), (NI, "ni"), (CFO, "cfo"), (CAPEX, "capex"), (RND, "rnd")]:
        r.update(flow_metrics(co, concepts, t0, tag))
    for concepts, tag in [(EQ, "eq"), (AS, "assets"), (LI, "liab"), (DEBT, "debt"), (DEBTC, "debtc"), (CASH, "cash")]:
        r[tag], _ = co.instant(concepts, t0)
    sh, _ = co.instant(SHS, t0, unit_money=False)
    r["shares_out"] = sh
    if r.get("rev_end") is not None:
        r["fs_age_days"] = (t0 - pd.Timestamp(r["rev_end"])).days
    rev = r.get("rev_ttm", np.nan)

    def div(a, b):
        return a / b if (np.isfinite(a) and np.isfinite(b) and b != 0) else np.nan

    r["gross_margin"] = div(r.get("gp_ttm", np.nan), rev)
    r["op_margin"] = div(r.get("op_ttm", np.nan), rev)
    r["net_margin"] = div(r.get("ni_ttm", np.nan), rev)
    r["fcf_margin"] = div(r.get("cfo_ttm", np.nan) - np.nan_to_num(r.get("capex_ttm", np.nan)), rev)
    r["rnd_sales"] = div(r.get("rnd_ttm", np.nan), rev)
    r["roe"] = div(r.get("ni_ttm", np.nan), r["eq"]) if np.isfinite(r["eq"]) and r["eq"] > 0 else np.nan
    r["roa"] = div(r.get("ni_ttm", np.nan), r["assets"])
    r["liab_assets"] = div(r["liab"], r["assets"])
    debt = np.nansum([r["debt"], r["debtc"]]) if np.isfinite(r["debt"]) or np.isfinite(r["debtc"]) else np.nan
    r["debt_equity"] = div(debt, r["eq"]) if np.isfinite(r["eq"]) and r["eq"] > 0 else np.nan
    r["profitable"] = float(r.get("ni_ttm", np.nan) > 0) if np.isfinite(r.get("ni_ttm", np.nan)) else np.nan
    if np.isfinite(r.get("rev_g_q", np.nan)) and np.isfinite(r.get("rev_g_q_prev", np.nan)):
        r["rev_accel"] = r["rev_g_q"] - r["rev_g_q_prev"]
    if np.isfinite(r.get("op_g_q", np.nan)) and np.isfinite(r.get("op_g_q_prev", np.nan)):
        r["op_accel"] = r["op_g_q"] - r["op_g_q_prev"]
    return r


def valuation(r, price):
    """USD 재무 + t0 주가 → 시가총액·PS·PE·PB·FCF수익률."""
    out = {}
    sh = r.get("shares_out", np.nan)
    if not (np.isfinite(sh) and np.isfinite(price) and sh > 0 and price > 0):
        return out
    mcap = sh * price
    out["mcap"] = mcap
    if r.get("fs_cur") != "USD":
        return out
    rev = r.get("rev_ttm", np.nan)
    ni = r.get("ni_ttm", np.nan)
    eq = r.get("eq", np.nan)
    fcf = r.get("cfo_ttm", np.nan) - np.nan_to_num(r.get("capex_ttm", np.nan))
    out["ps"] = mcap / rev if np.isfinite(rev) and rev > 0 else np.nan
    out["ey"] = ni / mcap if np.isfinite(ni) else np.nan  # 이익수익률(1/PER, 적자도 비교 가능)
    out["bm"] = eq / mcap if np.isfinite(eq) else np.nan  # 장부가/시총(1/PBR)
    out["fcf_yield"] = fcf / mcap if np.isfinite(fcf) else np.nan
    return out
