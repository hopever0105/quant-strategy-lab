# -*- coding: utf-8 -*-
"""OpenDART로 재무 보강 (KRX 아님). 끊겨도 이어받기: 배치별 결과를 quant/cache/larry/dart/ 에 저장하고 있으면 건너뜀.

1단계 multi : 다중회사 주요계정(fnlttMultiAcnt, 한 번에 100개사) 2015~2025 회계연도.
              상장폐지 종목처럼 dart_fs에 없는 재무를 채우고, 법인세비용 = 법인세차감전순이익 − 당기순이익 을 구한다.
2단계 tax   : 전체 재무제표(fnlttSinglAcntAll)에서 현금흐름표 '법인세 납부액' - 후보 종목(run.py가 만든 목록)만.
진행 상황은 quant/logs/larry_dart.log 에만 남긴다.
실행: python dart_fill.py multi   /   python dart_fill.py tax
"""
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
JOB = next((p for p in HERE.parents if (p / "quant" / "krx_guard.py").exists()), HERE.parents[1])  # 저장소에서는 상위 폴더 기준
VAL = JOB / "valuation"
OUT = JOB / "quant" / "cache" / "larry" / "dart"
OUT.mkdir(parents=True, exist_ok=True)
LOG = JOB / "quant" / "logs" / "larry_dart.log"
API = "https://opendart.fss.or.kr/api"
YEARS = range(2015, 2026)


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {m}\n")


def key():
    for l in (VAL / ".env").read_text(encoding="utf-8").splitlines():
        if l.startswith("DART_API_KEY="):
            return l.split("=", 1)[1].strip()


def corp_map():
    r = ET.parse(VAL / "tools" / "corpCode.xml").getroot()
    return {(e.findtext("stock_code") or "").strip(): e.findtext("corp_code") for e in r if (e.findtext("stock_code") or "").strip()}


def get(url, params, tries=3):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60).json()
            st = r.get("status")
            if st == "020":                      # 요청 한도 초과 → 멈추고 기록(재시도 안 함)
                log("DART 요청 한도 초과(020) - 중지")
                sys.exit(2)
            return r
        except Exception as e:
            log(f"오류 {k + 1}회: {e}")
            time.sleep(5)
    return {}


def multi():
    sys.path.insert(0, str(HERE))
    import data as D
    cm = corp_map()
    C = D.panel_wide()["c"]
    alive = C.loc["2015-04-01":].notna().any()
    codes = sorted(c for c in alive.index[alive] if c in cm)
    corps = [cm[c] for c in codes]
    batches = [corps[i:i + 100] for i in range(0, len(corps), 100)]
    log(f"multi 시작: {len(codes)}종목, {len(batches)}배치 × {len(YEARS)}년")
    k = key()
    for y in YEARS:
        for bi, b in enumerate(batches):
            fn = OUT / f"multi_{y}_{bi:03d}.json"
            if fn.exists():
                continue
            r = get(f"{API}/fnlttMultiAcnt.json", dict(crtfc_key=k, corp_code=",".join(b), bsns_year=y, reprt_code="11011"))
            fn.write_text(json.dumps(r.get("list", []), ensure_ascii=False), encoding="utf-8")
            time.sleep(0.3)
        log(f"multi {y} 완료")
    log("multi 끝")


def tax():
    cand = pd.read_csv(OUT / "tax_candidates.csv", dtype={"code": str})
    cm = corp_map()
    k = key()
    log(f"tax 시작: {len(cand)}건")
    for i, r in cand.iterrows():
        fn = OUT / f"tax_{r.code}_{r.fy}.json"
        if fn.exists() or r.code not in cm:
            continue
        rows = []
        for fd in ("CFS", "OFS"):
            j = get(f"{API}/fnlttSinglAcntAll.json", dict(crtfc_key=k, corp_code=cm[r.code], bsns_year=int(r.fy), reprt_code="11011", fs_div=fd))
            time.sleep(0.3)
            if j.get("list"):
                rows = [dict(sj=x.get("sj_div"), acc=x.get("account_nm"), v=x.get("thstrm_amount"), fs=fd) for x in j["list"]
                        if "법인세" in (x.get("account_nm") or "")]
                break
        fn.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        if i % 200 == 0:
            log(f"tax {i}/{len(cand)}")
    log("tax 끝")


if __name__ == "__main__":
    {"multi": multi, "tax": tax}[sys.argv[1]]()
