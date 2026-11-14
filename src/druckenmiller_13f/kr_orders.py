# -*- coding: utf-8 -*-
"""25) 한국 수주잔고: DART 정기보고서 원문 '수주상황' 표 받기 (백그라운드, 이어받기). KRX 요청 없음.

대상: 코스피 상위 200 대용 패널에 나온 종목 중 수주 업종(KRX 업종명 — 건설·운송장비·부품·기계·장비·전기·전자·IT 서비스)
보고서: 2015~ 사업·반기·분기보고서(정정 보고서는 원본 대신 가장 마지막 정정본)
저장: cache/druck13f/kr_orders/<code>_<rcept_no>.json = {보고서명, 접수일, 표들(행×칸 글자)} — 원문 zip은 저장 안 함
진행: quant/logs/druck13f.log 에만. DART 요청 한도(020)면 즉시 중지.
실행: python kr_orders.py fetch
"""
import io
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
QUANT = Path(r"C:\Users\arlo0\OneDrive\바탕 화면\취업\quant")
JOB = QUANT.parent
OUT = QUANT / "cache" / "druck13f" / "kr_orders2"
OUT.mkdir(parents=True, exist_ok=True)
LOG = QUANT / "logs" / "druck13f.log"
API = "https://opendart.fss.or.kr/api"
INDS = ["건설", "운송장비·부품", "기계·장비", "전기·전자", "IT 서비스"]


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + "[kr_orders] " + m + "\n")


def key():
    for l in (JOB / "valuation" / ".env").read_text(encoding="utf-8").splitlines():
        if l.startswith("DART_API_KEY="):
            return l.split("=", 1)[1].strip()


def targets():
    s = pd.read_csv(QUANT / "cache" / "larry" / "krx_sector_20261005.csv", dtype=str, encoding="cp949")
    P = pd.read_pickle(HERE / "results" / "combo" / "qpanel_kr.pkl")
    s = s[s["종목코드"].isin(set(P["code"])) & s["업종명"].isin(INDS)]
    return s[["종목코드", "종목명", "업종명"]].values.tolist()


def get(url, params, binary=False):
    for k in range(3):
        try:
            r = requests.get(url, params=params, timeout=90)
            time.sleep(0.4)
            if binary:
                if r.content[:2] == b"PK":
                    return r.content
                try:
                    j = r.json()
                except Exception:
                    j = {}
                if j.get("status") == "020":
                    log("DART 요청 한도 초과(020) - 중지")
                    sys.exit(2)
                return None
            j = r.json()
            if j.get("status") == "020":
                log("DART 요청 한도 초과(020) - 중지")
                sys.exit(2)
            return j
        except SystemExit:
            raise
        except Exception as e:  # noqa
            log(f"오류 {e}")
            time.sleep(5)
    return None


def reports(k, corp):
    rows, page = [], 1
    while True:
        j = get(f"{API}/list.json", dict(crtfc_key=k, corp_code=corp, bgn_de="20150101", end_de="20261231",
                                         pblntf_ty="A", page_no=page, page_count=100))
        if not j or j.get("status") != "000":
            break
        rows += j["list"]
        if page >= int(j.get("total_page", 1)):
            break
        page += 1
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df = df[df["report_nm"].str.contains("사업보고서|반기보고서|분기보고서")]
    df["period"] = df["report_nm"].str.extract(r"\((\d{4}\.\d{2})\)")[0]
    df = df.dropna(subset=["period"]).sort_values("rcept_dt")
    return df.groupby("period").tail(1)  # 정정본 있으면 마지막 것


TAG = re.compile(r"<[^>]+>")


def tables_in_section(xml):
    """문서 전체에서 '잔고'가 들어간 표 + 그 표 바로 앞 글(단위 표기)만 저장. 수주잔고 표가 여기에 있음."""
    """표 머리와 숫자가 다른 표로 나뉜 경우가 많아, 조건 맞는 표 + 바로 다음 표까지 저장.
    조건: 표 글자에 수주잔고/수주총액/수주잔액이 있거나, 표 앞 300자에 수주상황/수주현황/수주잔고가 있음(그리고 표에 잔고·잔액)."""
    out = []
    ms = list(re.finditer(r"<TABLE[^>]*>(.*?)</TABLE>", xml, flags=re.S | re.I))

    def rows_of(t):
        rows = []
        for tr in re.findall(r"<TR[^>]*>(.*?)</TR>", t, flags=re.S | re.I):
            cells = [re.sub(r"\s+", " ", TAG.sub("", c)).strip() for c in re.findall(r"<T[DHE][^>]*>(.*?)</T[DHE]>", tr, flags=re.S | re.I)]
            if cells:
                rows.append(cells)
        return rows

    for i, m in enumerate(ms):
        t = m.group(1)
        t2 = TAG.sub("", t).replace(" ", "")
        before = re.sub(r"\s+", " ", TAG.sub(" ", xml[max(0, m.start() - 800): m.start()]))[-300:]
        b2 = before.replace(" ", "")
        ok = any(w in t2 for w in ("수주잔고", "수주총액", "수주잔액")) or \
            (any(w in b2 for w in ("수주상황", "수주현황", "수주잔고")) and ("잔고" in t2 or "잔액" in t2))
        if not ok:
            continue
        nxt = rows_of(ms[i + 1].group(1)) if i + 1 < len(ms) and ms[i + 1].start() - m.end() < 3000 else []
        unit = re.findall(r"단위\s*[:：]?\s*([^)\]]{1,12})", before + " " + TAG.sub("", t)[:300])
        out.append(dict(rows=rows_of(t), next_rows=nxt, before=before, unit=unit[-1].strip() if unit else ""))
        if len(out) >= 8:
            break
    return dict(tables=out)


def fetch():
    k = key()
    T = targets()
    log(f"수주 원문 대상 {len(T)}개사")
    root = ET.parse(JOB / "valuation" / "tools" / "corpCode.xml").getroot()
    cm = {(e.findtext("stock_code") or "").strip(): e.findtext("corp_code") for e in root if (e.findtext("stock_code") or "").strip()}
    n = 0
    for code, name, ind in T:
        if code not in cm:
            continue
        lf = OUT / f"_list_{code}.csv"
        if lf.exists():
            R = pd.read_csv(lf, dtype=str)
        else:
            R = reports(k, cm[code])
            R.to_csv(lf, index=False, encoding="utf-8-sig")
        for _, r in R.iterrows():
            fn = OUT / f"{code}_{r['rcept_no']}.json"
            if fn.exists():
                old = json.loads(fn.read_text(encoding="utf-8"))
                if old.get("v") == 4:
                    continue
            b = get(f"{API}/document.xml", dict(crtfc_key=k, rcept_no=r["rcept_no"]), binary=True)
            res = dict(code=code, name=name, ind=ind, report=r["report_nm"], period=r["period"], rcept_dt=r["rcept_dt"], tables=[], unit="", v=4)
            if b:
                try:
                    z = zipfile.ZipFile(io.BytesIO(b))
                    for nm in sorted(z.namelist(), key=lambda x: -z.getinfo(x).file_size):
                        raw = z.read(nm)
                        for enc in ("utf-8", "cp949", "euc-kr"):
                            try:
                                xml = raw.decode(enc)
                                break
                            except Exception:
                                continue
                        sec = tables_in_section(xml)
                        if sec.get("tables"):
                            res.update(sec)
                            break
                except Exception as e:  # noqa
                    log(f"zip 오류 {code} {r['rcept_no']} {e}")
            fn.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
            n += 1
            if n % 100 == 0:
                log(f"수주 원문 {n}")
    log("kr_orders 끝")


if __name__ == "__main__":
    {"fetch": fetch}[sys.argv[1]]()
