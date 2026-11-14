# -*- coding: utf-8 -*-
"""27) 한국 수주잔고 신호 테스트 (수주 업종: 건설·운송장비·기계·전기전자·IT서비스). 실제 매매 없음. KRX 요청 없음.

원문 표(kr_orders.py)에서 보고서마다 '수주잔고 합계'를 읽음:
  표에 '잔고'가 있고, 행 이름이 합계/총계/계/수주잔고인 행의 마지막 숫자 칸. 단위(원·천원·백만원·억원) 맞춤.
  한 보고서에서 여러 표가 나오면 가장 큰 값(전체 합계). 문장 칸은 제외.
전년 같은 달 보고서와 비교 → 수주잔고 증가율. 공시일(접수일)부터 사용.
신호 gap = 수주잔고 증가율 − 매출 증가율(DART 연간, 패널 값)
테스트(분기, 그 업종 안): IC, 상위 30% − 하위 30%, 상위 30% vs 업종 동일가중 / 코스피, 2025 제외
결과: results/backlog/kr_orders_*.csv
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common_q import load, qsim, table, COST, C  # noqa: E402

OUT = HERE / "results" / "backlog"
NUM = re.compile(r"^\(?-?[\d,]+(\.\d+)?\)?$")
UNIT = [("억원", 1e8), ("백만원", 1e6), ("천원", 1e3), ("원", 1.0)]


def mult(u):
    u = (u or "").replace(" ", "")
    for k, v in UNIT:
        if k in u:
            return v
    return np.nan


def parse_num(c):
    c = c.replace(" ", "")
    if not NUM.match(c):
        return None
    neg = c.startswith("(") or c.startswith("-")
    v = float(c.strip("()").replace(",", "").replace("-", "") or 0)
    return -v if neg else v


def backlog_of(j):
    """① 합계·총계·수주잔고 행의 가장 오른쪽 숫자 ② 없으면 머리에 '잔고'가 있는 표의 숫자 행들 오른쪽 숫자 합. 표 여러 개면 최댓값."""
    tot, alt = None, None
    for t in j.get("tables", []):
        m = mult(t.get("unit")) if t.get("unit") else mult(j.get("unit"))
        if np.isnan(m):
            m = 1e6  # 단위 못 읽으면 백만원 가정(같은 회사 안 증가율이라 대부분 상쇄)
        rows = (t.get("rows") or []) + (t.get("next_rows") or [])
        head = " ".join(" ".join(r) for r in rows[:3]).replace(" ", "")
        datas = []
        for row in rows:
            if not row or len(row[0]) > 40:
                continue
            nums = [parse_num(c) for c in row[1:] if len(c) <= 25]
            nums = [x for x in nums if x is not None]
            lab = row[0].replace(" ", "")
            if nums and (re.search(r"^(합계|총계|계|총합계|수주잔고|총수주잔고)", lab)):
                v = nums[-1] * m
                if v > 0 and (tot is None or v > tot):
                    tot = v
            elif len(nums) >= 2:
                datas.append(nums[-1])
        if ("잔고" in head or "잔액" in head) and datas:
            v = sum(x for x in datas if x > 0) * m
            if v > 0 and (alt is None or v > alt):
                alt = v
    c = [x for x in (tot, alt) if x is not None]
    return max(c) if c else None


def load_backlog():
    rows = []
    for f in glob.glob(str(C / "kr_orders2" / "*.json")):
        if Path(f).name.startswith("_list"):
            continue
        j = json.loads(Path(f).read_text(encoding="utf-8"))
        v = backlog_of(j)
        rows.append(dict(code=j["code"], name=j["name"], ind=j["ind"], period=j["period"], filed=pd.Timestamp(j["rcept_dt"]), bl=v))
    B = pd.DataFrame(rows).dropna(subset=["bl"])
    B["pdate"] = pd.to_datetime(B["period"].str.replace(".", "-") + "-01") + pd.offsets.MonthEnd(0)
    B = B.sort_values(["code", "pdate"])
    prev = B[["code", "pdate", "bl"]].copy()
    prev["pdate"] = prev["pdate"] + pd.offsets.MonthEnd(12)
    B = B.merge(prev.rename(columns={"bl": "bl_prev"}), on=["code", "pdate"], how="left")
    B["bl_g"] = B["bl"] / B["bl_prev"] - 1
    B.loc[B["bl_g"].abs() > 5, "bl_g"] = np.nan  # 단위 바뀜 등 이상치
    return B


def main():
    B = load_backlog()
    B.to_csv(OUT / "kr_orders_backlog.csv", index=False, encoding="utf-8-sig")
    print(f"수주잔고 읽은 보고서 {len(B)}건, 회사 {B['code'].nunique()}, 증가율 계산 {B['bl_g'].notna().sum()}건")
    print(B.groupby("ind")["code"].nunique().to_dict())
    P, R, M = load("kr")
    codes = set(B["code"])
    P = P[P["code"].isin(codes)].copy()
    # t0에 공시된 가장 최근 보고서
    vals = []
    for (code), g in B.dropna(subset=["bl_g"]).groupby("code"):
        g = g.sort_values("filed")
        for q in P.loc[P["code"] == code, "t0"].unique():
            k = g[g["filed"] <= q]
            if len(k) and (q - k["pdate"].iloc[-1]).days < 200:
                vals.append(dict(code=code, t0=q, bl_g=k["bl_g"].iloc[-1]))
    V = pd.DataFrame(vals)
    P = P.merge(V, on=["code", "t0"], how="left")
    P["gap"] = P["bl_g"] - P["sales_g"]
    ind = B.drop_duplicates("code").set_index("code")["ind"]
    P["ind"] = P["code"].map(ind)
    d = P.dropna(subset=["gap", "ret"])
    print(f"신호 있는 종목 분기당 {d.groupby('t0').size().median():.0f}, 분기 {d['t0'].nunique()} ({d['t0'].min().date()}~)")

    def ic(col):
        s = d.dropna(subset=[col]).groupby("t0").apply(lambda g: stats.spearmanr(g[col], g["ret"]).statistic if len(g) >= 10 else np.nan).dropna()
        return s

    out = []
    for col, lb in [("gap", "수주잔고 증가율 − 매출 증가율"), ("bl_g", "수주잔고 증가율만")]:
        for drop in [False, True]:
            s = ic(col)
            if drop:
                s = s[s.index.year != 2025]
            out.append(dict(신호=lb + (" (2025 제외)" if drop else ""), IC=s.mean(), t=s.mean() / s.std() * np.sqrt(len(s)), 분기=len(s),
                            플러스분기=(s > 0).mean()))
    T1 = pd.DataFrame(out)
    print(T1.round(3).to_string(index=False))
    d = d.copy()
    d["p"] = d.groupby("t0")["gap"].rank(pct=True)
    sp = d.groupby("t0").apply(lambda g: g.loc[g["p"] >= 0.7, "ret"].mean() - g.loc[g["p"] <= 0.3, "ret"].mean())
    for drop in [False, True]:
        x = sp[sp.index.year != 2025] if drop else sp
        print(f"상위30% − 하위30% 분기 평균 {x.mean():+.2%} t {x.mean() / x.std() * np.sqrt(len(x)):+.2f}" + (" (2025 제외)" if drop else ""))
    by = {q: g for q, g in d.groupby("t0")}
    rows = []
    pk = {"상위 30%": {q: g.loc[g["p"] >= 0.7, "code"].tolist() for q, g in by.items()},
          "하위 30%": {q: g.loc[g["p"] <= 0.3, "code"].tolist() for q, g in by.items()},
          "수주 업종 신호 있는 종목 전부": {q: g["code"].tolist() for q, g in by.items()}}
    for nm, p in pk.items():
        r, n = qsim(p, 2, R, M, COST["kr"])
        rows += [(f"{nm} 2분기 (평균 {n.mean():.0f}종목)", "kr", r, M, False), (f"{nm} 2분기 (평균 {n.mean():.0f}종목)", "kr", r, M, True)]
    rows += [("코스피", "kr", M, M, False), ("코스피", "kr", M, M, True)]
    T = table(rows)
    T.to_csv(OUT / "kr_orders_results.csv", index=False, encoding="utf-8-sig")
    T1.to_csv(OUT / "kr_orders_ic.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(T[T["구간"].isin(["검증", "테스트", "전체"])][["전략", "구간", "CAGR", "샤프", "MDD", "시장CAGR"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
