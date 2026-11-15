# -*- coding: utf-8 -*-
"""22) 공시 속 단어 세기 → '말이 많아지기 시작한 테마'. 미국만(SEC 전문검색). 실제 매매 없음. KRX 요청 없음.

단어 목록은 미리 정함 — 뜬 테마만 고르면 사후 선택이 되므로, 꺼진 유행어도 같이 넣음:
  뜬 것·꺼진 것 섞어서 26개 (아래 TERMS)
1단계 fetch: 단어 × 분기(2011Q3~) 10-K·10-Q 문서 수, 그리고 언급한 회사 CIK 목록(최대 1,000문서) → cache/druck13f/efts/
2단계 test (python themes.py test):
  테마 '뜨는 중' = 최근 2분기 문서 수 ≥ 그 1년 전 2분기의 2배, 그리고 최근 2분기 ≥ 40건  (분기말 t0에 아는 값)
  종목 = S&P500 중 최근 2분기 공시에 그 단어를 쓴 회사
  T1 뜨는 테마 언급 종목 전부 / T2 T1 중 매출 성장 백분위 ≥ 70 / T3 T2 + 200일선 위 / 비교: 기본 규칙, 시장
  보유 2분기, 비용 0.1%
  확인: 드러켄밀러 확신 매수가 '뜨는 테마' 언급 종목이었던 비율 vs S&P500 평균
"""
import json
import sys
import time
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
QUANT = Path(r"C:\Users\arlo0\OneDrive\바탕 화면\취업\quant")
EF = QUANT / "cache" / "druck13f" / "efts"
EF.mkdir(parents=True, exist_ok=True)
LOG = QUANT / "logs" / "druck13f.log"
OUT = HERE / "results" / "themes"
OUT.mkdir(parents=True, exist_ok=True)
H = {"User-Agent": "QuantResearch research-bot admin@example.com"}
TERMS = ["artificial intelligence", "generative AI", "large language model", "machine learning", "data center",
         "cloud computing", "cybersecurity", "electricity demand", "nuclear power", "liquid cooling", "GLP-1",
         "obesity", "gene therapy", "CRISPR", "electric vehicle", "autonomous driving", "lithium", "hydrogen",
         "blockchain", "cryptocurrency", "metaverse", "virtual reality", "3D printing", "cannabis", "5G", "Internet of Things"]


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + "[themes] " + m + "\n")


def quarters():
    q = pd.period_range("2011Q1", pd.Timestamp.today().to_period("Q"), freq="Q")
    return [(p.start_time.date().isoformat(), p.end_time.date().isoformat(), str(p)) for p in q]


def q_one(term, a, b, frm=0):
    u = ("https://efts.sec.gov/LATEST/search-index?q=" + urllib.parse.quote(f'"{term}"') +
         f"&forms=10-K,10-Q&dateRange=custom&startdt={a}&enddt={b}&from={frm}")
    for k in range(4):
        try:
            r = requests.get(u, headers=H, timeout=60)
            time.sleep(0.2)
            if r.status_code == 200:
                return r.json()
        except Exception as e:  # noqa
            log(f"오류 {term} {a} {e}")
        time.sleep(5 * (k + 1))
    return None


def fetch():
    for term in TERMS:
        for a, b, q in quarters():
            fn = EF / f"{term.replace(' ', '_')}_{q}.json"
            if fn.exists():
                continue
            j = q_one(term, a, b)
            if j is None or "hits" not in j:
                log(f"응답 이상 {term} {q}")
                continue
            tot = j["hits"]["total"]["value"]
            ciks = set()
            for h in j["hits"]["hits"]:
                ciks.update(h["_source"].get("ciks", []))
            frm = 100
            while frm < min(tot, 1000):
                jj = q_one(term, a, b, frm)
                if not jj or "hits" not in jj:
                    log(f"페이지 끊김 {term} {q} from={frm}")
                    break
                for h in jj["hits"]["hits"]:
                    ciks.update(h["_source"].get("ciks", []))
                frm += 100
            fn.write_text(json.dumps(dict(total=tot, rel=j["hits"]["total"]["relation"], ciks=sorted(ciks))))
        log(f"{term} 끝")
    log("themes fetch 끝")


def test():
    sys.path.insert(0, str(HERE))
    from common_q import load, base_pick, qsim, table, COST
    from scipy import stats
    tot, who = {}, {}
    for term in TERMS:
        for a, b, q in quarters():
            fn = EF / f"{term.replace(' ', '_')}_{q}.json"
            if fn.exists():
                j = json.loads(fn.read_text())
                tot[(term, q)] = j["total"]
                who[(term, q)] = set(int(c) for c in j["ciks"])
    T = pd.Series(tot).unstack(0)  # index 분기, columns 단어
    T.index = pd.PeriodIndex(T.index, freq="Q")
    T = T.sort_index()
    T.to_csv(OUT / "term_counts.csv", encoding="utf-8-sig")
    last2 = T + T.shift(1)
    prev2 = T.shift(4) + T.shift(5)
    hot = (last2 >= 2 * prev2) & (last2 >= 40)
    hot.to_csv(OUT / "hot_flags.csv", encoding="utf-8-sig")
    print("뜨는 테마로 잡힌 분기(단어별 처음 3개):")
    for t in TERMS:
        if t in hot and hot[t].any():
            print(f"  {t}: {[str(x) for x in hot.index[hot[t]][:3]]} … 총 {int(hot[t].sum())}분기")
    P, R, M = load("us")
    Pn = pd.read_pickle(HERE / "results" / "panel_ranked.pkl")
    y2c = Pn[Pn["src"] == "S&P500"][["yf", "cik"]].dropna().drop_duplicates("yf")
    y2c = dict(zip(y2c["yf"], y2c["cik"].astype(int)))
    P["cik"] = P["code"].map(y2c)
    P["theme"] = False
    P["themes"] = ""
    for q in sorted(P["t0"].unique()):
        pq = pd.Period(q, freq="Q")
        if pq not in hot.index:
            continue
        hs = [t for t in TERMS if t in hot and bool(hot.at[pq, t])]
        if not hs:
            continue
        m = P["t0"] == q
        for t in hs:
            cs = who.get((t, str(pq)), set()) | who.get((t, str(pq - 1)), set())
            hit = m & P["cik"].isin(cs)
            P.loc[hit, "theme"] = True
            P.loc[hit, "themes"] = P.loc[hit, "themes"] + t + "; "
    d = P.dropna(subset=["ret"])
    sp = d.groupby(["t0", "theme"])["ret"].mean().unstack().dropna()
    x = sp[True] - sp[False]
    print(f"뜨는 테마 언급 종목 − 나머지 다음 분기 수익 {x.mean():+.2%} t {x.mean() / x.std() * np.sqrt(len(x)):+.2f} ({len(x)}분기, 이긴 분기 {(x > 0).mean():.0%})")
    by = {q: g for q, g in P.groupby("t0")}
    pk = {"T1 뜨는 테마 언급 전부": {q: g.loc[g["theme"], "code"].tolist() for q, g in by.items()},
          "T2 T1 ∩ 매출 성장 상위 30%": {q: g.loc[g["theme"] & (g["p_sales_g"] >= 70), "code"].tolist() for q, g in by.items()},
          "T3 T2 ∩ 200일선 위": {q: g.loc[g["theme"] & (g["p_sales_g"] >= 70) & (g["vs200"] > 0), "code"].tolist() for q, g in by.items()},
          "기본 규칙": {q: base_pick(g) for q, g in by.items()}}
    rows = []
    for nm, p in pk.items():
        r, n = qsim(p, 2, R, M, COST["us"])
        rows.append((f"{nm} (평균 {n.mean():.0f}종목, 빈 분기 {(n == 0).mean():.0%})", "us", r, M, False))
    rows.append(("동일가중 시장", "us", d.groupby("t0")["ret"].mean(), M, False))
    rows.append(("시장지수", "us", M, M, False))
    Tb = table(rows)
    Tb.to_csv(OUT / "results.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(Tb.round(3).to_string(index=False))
    # 드러켄밀러 확신 매수가 뜨는 테마 종목이었나
    E = Pn[(Pn["src"] == "드러켄밀러") & (Pn["leg"] == "주식") & Pn["action"].isin(["신규", "확대"]) & (Pn["weight"] >= 0.03)].copy()
    E["t0"] = pd.to_datetime(E["t0"])
    E = E.merge(P[["t0", "code", "theme", "themes"]], left_on=["t0", "yf"], right_on=["t0", "code"], how="inner")
    base = P.groupby("t0")["theme"].mean()
    print(f"확신 매수 중 뜨는 테마 언급 종목 비율 {E['theme'].mean():.0%} (S&P500 평균 {base.mean():.0%}, 같은 분기 기준 {base.reindex(E['t0']).mean():.0%})")
    E[["t0", "yf", "name", "weight", "theme", "themes"]].to_csv(OUT / "druck_conviction_themes.csv", index=False, encoding="utf-8-sig")
    P[["t0", "code", "theme", "themes"]].to_pickle(OUT / "signal_us.pkl")


if __name__ == "__main__":
    {"fetch": fetch, "test": test}[sys.argv[1]]()
