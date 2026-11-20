# -*- coding: utf-8 -*-
"""상장폐지 종목 보강 (KRX, 로그인 세션 + krx_guard)

  python krx_delist.py list    : 2015~ 상장폐지 목록(MDCSTAT23801, 2년 단위) + 종목코드→표준코드 표 → cache/larry/delist/
  python krx_delist.py prices  : 목록 중 패널에 일봉·거래대금이 없는 종목을 종목별로(MDCSTAT23902, 2년 단위) 받음

안전장치: krx_guard 먼저 설치, 실패 1번이면 즉시 중지 + krx_paused.txt 다시 생성, 실행당 요청 상한 RUN_CAP,
이어받기(종목·구간별 파일이 있으면 건너뜀). 진행은 quant/logs/larry_krx.log 에만.
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOB = next((p for p in HERE.parents if (p / "quant" / "krx_guard.py").exists()), HERE.parents[1])  # 저장소에서는 상위 폴더 기준
sys.path.insert(0, str(JOB / "quant"))
import krx_guard  # noqa: E402
krx_guard.install()
import os  # noqa: E402
if os.environ.get("LARRY_KRX_DAILY_LIMIT"):            # 사용자가 2026-11-12 이번 수집에 한해 허락한 하루 한도(가드 파일은 그대로)
    krx_guard.DAILY_LIMIT = int(os.environ["LARRY_KRX_DAILY_LIMIT"])

import pandas as pd  # noqa: E402
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402

OUT = D.CACHE / "delist"
OUT.mkdir(parents=True, exist_ok=True)
LOG = JOB / "quant" / "logs" / "larry_krx.log"
URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"   # pykrx와 같은 https (로그인 쿠키가 https 전용)
RUN_CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 300
START = datetime(2015, 1, 1)


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {m}\n")
    print(m, flush=True)


def stop(note, code):
    log(f"중지: {note}")
    if not krx_guard.paused():
        krx_guard._pause(f"larry krx_delist 실패 - {note}")
        log("krx_paused.txt 다시 생성(원인 확인 후 직접 지울 것)")
    sys.exit(code)


def session():
    try:
        from pykrx.website.comm.webio import get_session      # import 시 로그인
    except krx_guard.KRXBlocked as e:
        stop(f"로그인 단계 KRXBlocked: {e}", 2)
    except Exception as e:
        stop(f"로그인 단계 오류: {e}", 3)
    return get_session()          # KRXSession: post 때 로그인 헤더를 자동으로 붙임


def post(s, data, start_count):
    if krx_guard.status()["count"] - start_count >= RUN_CAP:
        log(f"이번 실행 요청 상한 {RUN_CAP}건 도달 - 정상 종료(다음에 이어받기)")
        sys.exit(0)
    try:
        r = s.post(URL, headers={"Referer": "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd"}, data=data, timeout=30)
        try:
            j = r.json()
        except Exception:
            stop(f"JSON 아님 status={r.status_code} len={len(r.content)} 앞부분={r.text[:150]!r}", 3)
    except krx_guard.KRXBlocked as e:
        stop(f"KRXBlocked: {e}", 2)
    except Exception as e:
        stop(f"요청/해석 오류: {e}", 3)
    return j


def windows(a, b):
    out, s = [], a
    while s <= b:
        e = min(datetime(s.year + 2, s.month, s.day) - timedelta(days=1), b)
        out.append((s, e))
        s = e + timedelta(days=1)
    return out


def do_list():
    start = krx_guard.status()["count"]
    s = session()
    rows = []
    for a, b in windows(START, datetime(2026, 11, 12)):
        j = post(s, {"bld": "dbms/MDC/STAT/issue/MDCSTAT23801", "mktId": "ALL", "isuCd": "ALL", "isuCd2": "ALL",
                     "strtDd": a.strftime("%Y%m%d"), "endDd": b.strftime("%Y%m%d"), "share": "1", "csvxls_isNo": "true"}, start)
        out = j.get("output")
        if out is None:
            stop(f"목록 응답 형식 이상: {str(j)[:200]}", 4)
        rows += out
        log(f"목록 {a.date()}~{b.date()} {len(out)}건")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "delist_list.csv", index=False, encoding="utf-8-sig")
    j = post(s, {"bld": "dbms/comm/finder/finder_listdelisu", "mktsel": "ALL", "searchText": ""}, start)
    fd = pd.DataFrame(j.get("block1", []))
    if not len(fd):
        stop(f"표준코드 표 응답 이상: {str(j)[:200]}", 4)
    fd.to_csv(OUT / "finder.csv", index=False, encoding="utf-8-sig")
    log(f"목록 완료: 상장폐지 {len(df)}건, 표준코드 표 {len(fd)}건, 이번 실행 요청 {krx_guard.status()['count'] - start}건")


def targets():
    """패널에 2018-07 이후 일봉·거래대금이 없는 상장폐지 보통주 (2018-01부터 상장폐지일까지)"""
    df = pd.read_csv(OUT / "delist_list.csv", dtype=str)
    fd = pd.read_csv(OUT / "finder.csv", dtype=str)
    df["code"] = df["ISU_CD"].str[-6:] if df["ISU_CD"].str.len().max() > 6 else df["ISU_CD"]
    df["dd"] = pd.to_datetime(df["DELIST_DD"], format="%Y/%m/%d", errors="coerce")
    df["ld"] = pd.to_datetime(df["LIST_DD"], format="%Y/%m/%d", errors="coerce")
    # 분석 대상과 같게: 코스피·코스닥 주권 보통주, 스팩(기업인수목적회사) 제외
    df = df[(df.dd >= "2018-07-04") & (df.KIND_STKCERT_TP_NM.fillna("").str.contains("보통"))]
    df = df[(df.SECUGRP_NM == "주권") & df.MKT_NM.isin(["KOSPI", "KOSDAQ"]) & ~df.ISU_NM.fillna("").str.contains("스팩")]
    sc = "short_code" if "short_code" in fd else fd.columns[1]
    fc = "full_code" if "full_code" in fd else fd.columns[0]
    m = dict(zip(fd[sc].str[-6:], fd[fc]))
    df["full"] = df.code.map(m)
    return df


def do_prices():
    start = krx_guard.status()["count"]
    T = targets()
    log(f"가격 대상 {len(T)}종목(2018-07-04 이후 상장폐지 보통주), 표준코드 있음 {T.full.notna().sum()}")
    s = session()
    for _, r in T.iterrows():
        if pd.isna(r.full):
            continue
        a = max(datetime(2018, 1, 1), r.ld.to_pydatetime() if pd.notna(r.ld) else datetime(2018, 1, 1))   # 원본 시세가 2018-07까지 있고 1년 지표용 여유
        for w0, w1 in windows(a, r.dd.to_pydatetime()):
            fn = OUT / f"px_{r.code}_{w0:%Y%m%d}.csv"
            if fn.exists():
                continue
            j = post(s, {"bld": "dbms/MDC/STAT/issue/MDCSTAT23902", "isuCd": r.full, "isuCd2": "", "strtDd": w0.strftime("%Y%m%d"),
                         "endDd": w1.strftime("%Y%m%d"), "share": "1", "money": "1", "csvxls_isNo": "false"}, start)
            out = j.get("output")
            if out is None:
                stop(f"{r.code} 응답 형식 이상: {str(j)[:200]}", 4)
            pd.DataFrame(out).to_csv(fn, index=False, encoding="utf-8-sig")
        log(f"{r.code} {r.ISU_NM if 'ISU_NM' in r else ''} 완료")
    log(f"가격 전부 완료. 이번 실행 요청 {krx_guard.status()['count'] - start}건")


if __name__ == "__main__":
    {"list": do_list, "prices": do_prices}[sys.argv[1]]()
