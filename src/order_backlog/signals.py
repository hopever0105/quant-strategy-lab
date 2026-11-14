# -*- coding: utf-8 -*-
"""수주잔고 전략 — 2) 신호·종목 고르기. 실제 주문·매매 없음.

gap = 수주잔고 전년 대비 증가율 − 매출 증가율 (분기말 t0에 공시된 값만)
미국: 업종(11개) 안 순위 → 상위 top 비율 매수(하위는 공매도 버전용)
한국: 수주 업종 61개사 전체 안 순위(업종별로 나누면 분기당 2~8종목뿐이라)
"""
import numpy as np
import pandas as pd


def rank(P, neutral):
    P = P.copy()
    if neutral:
        P["pr"] = P.groupby(["t0", "sector"])["gap"].rank(pct=True)
        # 업종 종목 수 3개 미만이면 전체 순위로
        n = P.groupby(["t0", "sector"])["gap"].transform("size")
        allr = P.groupby("t0")["gap"].rank(pct=True)
        P.loc[n < 3, "pr"] = allr[n < 3]
    else:
        P["pr"] = P.groupby("t0")["gap"].rank(pct=True)
    return P


def picks(P, top=0.3, side="top", neutral=True, extra=None):
    """{t0: [code]} — side 'top' 상위 top, 'bottom' 하위 top. extra: 추가 조건 함수(g → bool Series)."""
    P = rank(P, neutral)
    out = {}
    for q, g in P.groupby("t0"):
        m = g["pr"] > 1 - top if side == "top" else g["pr"] <= top
        if extra is not None:
            m &= extra(g)
        out[q] = g.loc[m, "code"].tolist()
    return out


def ratio_hi(g):
    """참고 변형: 상위 안에서 잔고 ÷ 최근 4분기 매출이 그 분기 중앙값 이상 (결과를 보고 찾은 조건이라 본 규칙 아님)"""
    return g["ratio"] >= g["ratio"].median()


def random_picks(P, n_by_q, rng):
    out = {}
    for q, g in P.groupby("t0"):
        k = n_by_q.get(q, 0)
        codes = g["code"].values
        out[q] = list(rng.choice(codes, size=min(k, len(codes)), replace=False)) if k else []
    return out
