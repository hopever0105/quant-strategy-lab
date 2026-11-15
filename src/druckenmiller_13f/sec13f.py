# -*- coding: utf-8 -*-
"""1) Duquesne Family Office(CIK 0001536411) 13F 전체 수집·파싱.

- 원본은 quant/cache/druck13f/raw/<accession>/ 에 저장(있으면 다시 받지 않음 → 이어받기).
- 2013Q1까지 텍스트 표, 이후 XML infotable.
- 정정(13F-HR/A): XML의 amendmentType이 RESTATEMENT면 해당 분기 전체 교체, NEW HOLDINGS면 추가.
  텍스트 정정은 내용 보고 판단(로그에 남김).
결과: results/holdings_raw.csv (분기·CUSIP·종목명·클래스·평가액·주식수·PUT/CALL)
"""
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from lxml import etree

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
QUANT = Path(r"C:\Users\arlo0\OneDrive\바탕 화면\취업\quant")
RAW = QUANT / "cache" / "druck13f" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
LOG = QUANT / "logs" / "druck13f.log"
H = {"User-Agent": "QuantResearch research-bot admin@example.com"}
CIK = "1536411"


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def get(url, path=None):
    if path is not None and path.exists():
        return path.read_bytes()
    for k in range(4):
        r = requests.get(url, headers=H, timeout=60)
        time.sleep(0.25)
        if r.status_code == 200:
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(r.content)
            return r.content
        log(f"HTTP {r.status_code} {url} (try {k})")
        time.sleep(3 * (k + 1))
    raise RuntimeError(url)


def filings():
    j = json.loads(get(f"https://data.sec.gov/submissions/CIK{CIK.zfill(10)}.json",
                       RAW / "submissions.json"))
    rec = j["filings"]["recent"]
    rows = []
    for f, d, a, p, rd in zip(rec["form"], rec["filingDate"], rec["accessionNumber"],
                              rec["primaryDocument"], rec["reportDate"]):
        if f.startswith("13F"):
            rows.append(dict(form=f, filed=d, acc=a, primary=p, period=rd))
    return pd.DataFrame(rows).sort_values(["period", "filed"]).reset_index(drop=True)


def strip_ns(root):
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def parse_xml_table(b):
    root = strip_ns(etree.fromstring(b))
    out = []
    for it in root.iter("infoTable"):
        g = lambda t: (it.findtext(t) or "").strip()
        out.append(dict(
            name=g("nameOfIssuer"), cls=g("titleOfClass"), cusip=g("cusip").upper(),
            value=float(g("value") or 0), shares=float(it.findtext("shrsOrPrnAmt/sshPrnamt") or 0),
            sh_type=(it.findtext("shrsOrPrnAmt/sshPrnamtType") or "").strip(),
            putcall=g("putCall").upper(), discretion=g("investmentDiscretion"),
        ))
    return out


CUSIP_RE = re.compile(r"\b([0-9A-Z]{6}[0-9A-Z]{2}[0-9])\b")


def parse_txt_table(text):
    """2011~2013Q1 텍스트 13F. 한 줄: NAME  CLASS  CUSIP  VALUE(천달러)  SHARES  SH/PRN  PUT/CALL ..."""
    out = []
    body = text
    m = re.search(r"<TABLE>(.*?)</TABLE>", text, re.S | re.I)
    if m:
        body = m.group(1)
    for line in body.splitlines():
        mc = CUSIP_RE.search(line)
        if not mc:
            continue
        cus = mc.group(1)
        if not any(ch.isdigit() for ch in cus[:6]) and not cus[:6].isalnum():
            continue
        left = line[:mc.start()].rstrip()
        right = line[mc.end():].split()
        nums = []
        for tok in right:
            t = tok.replace(",", "").replace("$", "")
            if re.fullmatch(r"\d+(\.\d+)?", t):
                nums.append(float(t))
            else:
                break
        if len(nums) < 2:
            continue
        rest = " ".join(right[len(nums):]).upper()
        sh_type = "PRN" if re.search(r"\bPRN\b", rest) else "SH"
        pc = "PUT" if re.search(r"\bPUT\b", rest) else ("CALL" if re.search(r"\bCALL\b", rest) else "")
        parts = re.split(r"\s{2,}", left.strip())
        name = parts[0] if parts else left.strip()
        cls = parts[1] if len(parts) > 1 else ""
        out.append(dict(name=name, cls=cls, cusip=cus, value=nums[0], shares=nums[1],
                        sh_type=sh_type, putcall=pc, discretion=""))
    return out


def fetch_one(row):
    acc = row["acc"]
    accn = acc.replace("-", "")
    d = RAW / acc
    idx = json.loads(get(f"https://www.sec.gov/Archives/edgar/data/{CIK}/{accn}/index.json", d / "index.json"))
    items = [x["name"] for x in idx["directory"]["item"]]
    amend_type = ""
    # primary doc (XML) → amendmentType
    for nm in items:
        if nm.lower() == "primary_doc.xml":
            b = get(f"https://www.sec.gov/Archives/edgar/data/{CIK}/{accn}/{nm}", d / nm)
            root = strip_ns(etree.fromstring(b))
            amend_type = (root.findtext(".//amendmentType") or "").strip().upper()
    xmls = [nm for nm in items if nm.lower().endswith(".xml") and nm.lower() != "primary_doc.xml"]
    recs = []
    if xmls:
        for nm in xmls:
            b = get(f"https://www.sec.gov/Archives/edgar/data/{CIK}/{accn}/{nm}", d / nm)
            if b"infoTable" in b:
                recs += parse_xml_table(b)
        fmt = "xml"
    else:
        txts = [nm for nm in items if nm.lower().endswith(".txt") and not nm.startswith(accn[:0] + acc)]
        if not txts:
            txts = [nm for nm in items if nm.lower().endswith(".txt")]
        for nm in txts:
            b = get(f"https://www.sec.gov/Archives/edgar/data/{CIK}/{accn}/{nm}", d / nm)
            recs += parse_txt_table(b.decode("latin-1"))
        fmt = "txt"
        if row["form"].endswith("/A"):
            amend_type = "RESTATEMENT?"
    return recs, fmt, amend_type


def main():
    fl = filings()
    fl.to_csv(OUT / "filings.csv", index=False, encoding="utf-8-sig")
    print(f"13F 공시 {len(fl)}건 (정정 {fl.form.str.endswith('/A').sum()})")
    per = {}
    meta = []
    for _, row in fl.iterrows():
        recs, fmt, at = fetch_one(row)
        n = len(recs)
        meta.append(dict(**row, fmt=fmt, amend_type=at, n=n, total_value=sum(r["value"] for r in recs)))
        df = pd.DataFrame(recs)
        p = row["period"]
        if not row["form"].endswith("/A"):
            per[p] = dict(df=df, filed=row["filed"])
        else:
            if n == 0:
                continue
            if at.startswith("NEW HOLDINGS"):
                per[p]["df"] = pd.concat([per[p]["df"], df], ignore_index=True)
            else:  # RESTATEMENT (또는 텍스트 정정): 교체
                per[p] = dict(df=df, filed=per.get(p, {}).get("filed", row["filed"]), amended=row["filed"])
        log(f"{row['form']} {p} {row['acc']} {fmt} {at} n={n}")
    pd.DataFrame(meta).to_csv(OUT / "filings_parsed.csv", index=False, encoding="utf-8-sig")
    allr = []
    for p, v in per.items():
        df = v["df"].copy()
        df["period"] = p
        df["filed"] = v["filed"]
        allr.append(df)
    h = pd.concat(allr, ignore_index=True)
    # 2022Q4까지 value는 천 달러, 2023Q1부터 달러 단위(SEC 규정 변경)
    per_dt = pd.to_datetime(h["period"])
    h["value_usd"] = h["value"] * 1000
    newunit = per_dt >= "2022-12-31"
    # 보고서마다 단위 확인: 가격(value/shares)이 비정상적으로 작으면 달러 단위
    for p in h["period"].unique():
        m = (h["period"] == p) & (h["shares"] > 0) & (h["putcall"] == "")
        px = (h.loc[m, "value"] / h.loc[m, "shares"]).median()
        if px > 1:  # 달러 단위로 보고
            h.loc[h["period"] == p, "value_usd"] = h.loc[h["period"] == p, "value"]
    h.to_csv(OUT / "holdings_raw.csv", index=False, encoding="utf-8-sig")
    s = h.groupby("period").agg(n=("cusip", "size"), val=("value_usd", "sum"))
    s["val"] = (s["val"] / 1e6).round(0)
    print(s.to_string())


if __name__ == "__main__":
    main()
