# -*- coding: utf-8 -*-
"""보고서 생성: results/ → charts/*.png + 딥러닝_전략_검증_보고서.docx (+pdf)
실행: python build_report.py"""
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
CH = HERE / "charts"
CH.mkdir(exist_ok=True)
OUT = HERE / "딥러닝_전략_검증_보고서.docx"
NAVY = "1F3864"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def rdir(prefix):
    return next(p for p in RES.iterdir() if p.name.startswith(prefix + "_"))


def rd(prefix, name, **k):
    return pd.read_csv(rdir(prefix) / f"{name}.csv", encoding="utf-8-sig", **k)


def pc(x, d=1):
    return "-" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def f2(x):
    return "-" if x is None or not np.isfinite(x) else f"{x:.2f}"


# ───────────────────────── 자료
C33_COST = rd("추가3-3", "비용민감도", index_col=0)
C33_ST = rd("추가3-3", "보완성과", index_col=0)
C33_D = rd("추가3-3", "일별수익", index_col=0, parse_dates=True)
C33_Y = rd("추가3-3", "연도별수익", index_col=0)
C33_V = rd("추가3-3", "변동성구간", index_col=0)
C33_H = rd("추가3-3", "고정검증", index_col=0)
C33_STK = rd("추가3-3", "주식참고", index_col=0)
C33_NAV = rd("추가3-3", "NAV괴리", index_col=0)
C32_B = rd("추가3-2", "구간_진단", index_col=0)
C32_US = rd("추가3-2", "미국갭_일별", index_col=0, parse_dates=True)
A13_M = rd("추가1-3", "월별수익", index_col=0, parse_dates=True)
A13_SPLIT = rd("추가1-3", "기간나눠보기", index_col=0)
A12_ST = rd("추가1-2", "성과", index_col=0)
D42_R = rd("추가4-2", "규칙비교", index_col=0)
D43_L = rd("추가4-3", "단계별", index_col=0)
D43_AE = rd("추가4-3", "오토인코더보완", index_col=0)
B22_D = rd("추가2-2", "일별수익", index_col=0, parse_dates=True)

K = "KODEX 반도체"
T = "TIGER 반도체"
cost_k = C33_COST.loc[f"{K} · 로지스틱 상위20%"]
cost_t = C33_COST.loc[f"{T} · 로지스틱 상위20%"]
st_k = C33_ST.loc[f"{K} 로지스틱 상위20%"]
st_kp = C33_ST.loc[f"{K} 로지스틱 상위20% 확률비례"]
st_t = C33_ST.loc[f"{T} 로지스틱 상위20%"]
st_hold = C33_ST.loc[f"{K} 보유(참고)"]


def perf(r, ppy):
    r = r.dropna()
    cum = (1 + r).cumprod()
    yrs = len(r) / ppy
    return {"CAGR": cum.iloc[-1] ** (1 / yrs) - 1, "샤프": r.mean() / r.std() * np.sqrt(ppy),
            "MDD": (cum / cum.cummax() - 1).min()}


# ───────────────────────── 차트
fig, ax = plt.subplots(figsize=(10, 4.2))
for c, col, lw in [(f"{K} 로지스틱 상위20%", "#1f4e79", 2.2), (f"{K} 로지스틱 상위20% 확률비례", "#2e75b6", 1.6),
                   (f"{T} 로지스틱 상위20%", "#70ad47", 1.6), (f"{K} 보유(참고)", "#7f7f7f", 1.2)]:
    ax.plot((1 + C33_D[c].fillna(0)).cumprod(), label=c, color=col, lw=lw)
ax.set_yscale("log"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
ax.set_title("반도체 ETF 장중 전략 누적수익 (2014~, 왕복 비용 0.05%, 로그 눈금)", fontsize=10)
fig.tight_layout(); fig.savefig(CH / "fig1_semis_cum.png", dpi=160); plt.close(fig)

costs = [0.05, 0.10, 0.15, 0.20]
fig, ax = plt.subplots(figsize=(10, 3.4))
for nm, row, col in [(f"{K} 상위20%", cost_k, "#1f4e79"), (f"{T} 상위20%(학습 안 씀)", cost_t, "#70ad47"),
                     (f"{K} 상위10%", C33_COST.loc[f"{K} · 로지스틱 상위10%"], "#c55a11")]:
    ax.plot(costs, [row[f"샤프@{c:.2f}%"] for c in costs], marker="o", color=col, label=nm)
ax.axhline(st_hold["샤프"], color="#7f7f7f", ls="--", lw=1, label=f"{K} 보유 샤프 {st_hold['샤프']:.2f}")
ax.set_xticks(costs); ax.set_xticklabels([f"{c:.2f}%" for c in costs]); ax.set_xlabel("왕복 비용 가정")
ax.set_ylabel("샤프"); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.set_title("비용을 올려도 남는가 (로지스틱 확률 상위 날만 매수)", fontsize=10)
fig.tight_layout(); fig.savefig(CH / "fig2_cost.png", dpi=160); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, nm in zip(axes, ["삼성전자", "SK하이닉스"]):
    sub = C32_B[C32_B.index.str.startswith(f"{nm} · SOXX구간")]
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub["평균 갭"] * 100, 0.4, color="#9dc3e6", label="시가 갭")
    ax.bar(x + 0.2, sub["평균 장중"] * 100, 0.4, color="#1f4e79", label="장중(시가→종가)")
    ax.set_xticks(x); ax.set_xticklabels(["1\n가장 하락", "2", "3", "4", "5\n가장 상승"], fontsize=8)
    ax.axhline(0, color="k", lw=.7); ax.grid(alpha=.3, axis="y"); ax.set_title(f"{nm}: 전날 밤 SOXX 5구간", fontsize=10)
    ax.set_ylabel("평균(%)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(CH / "fig3_overnight.png", dpi=160); plt.close(fig)

yy = C33_Y[[f"{K} 로지스틱 상위20%", f"{K} 로지스틱 상위20% 확률비례"]]
fig, ax = plt.subplots(figsize=(10, 3.2))
x = np.arange(len(yy))
ax.bar(x - 0.2, yy.iloc[:, 0] * 100, 0.4, color="#1f4e79", label="상위20% 같은 금액")
ax.bar(x + 0.2, yy.iloc[:, 1] * 100, 0.4, color="#9dc3e6", label="상위20% 확률 비례 금액")
ax.set_xticks(x); ax.set_xticklabels(yy.index, fontsize=8); ax.axhline(0, color="k", lw=.7); ax.grid(alpha=.3, axis="y")
ax.legend(fontsize=8); ax.set_title(f"{K} 연도별 수익률 (2026은 11월까지)", fontsize=10)
fig.tight_layout(); fig.savefig(CH / "fig4_yearly.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3.6))
for c, col, lw in [("앙상블(MLP기본+모멘텀) · 매달", "#1f4e79", 2.0), ("모멘텀 · 매달", "#c55a11", 1.4),
                   ("MLP 기본 · 매달", "#70ad47", 1.2), ("10개 같은 비중", "#7f7f7f", 1.2)]:
    ax.plot((1 + A13_M[c].fillna(0)).cumprod(), label=c, color=col, lw=lw)
ax.axvline(pd.Timestamp("2012-01-31"), color="k", ls=":", lw=1)
ax.set_yscale("log"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
ax.set_title("ETF 10개 자산배분 (2006~, 월간, 비용 차감, 점선 = 처음 시험한 2012년)", fontsize=10)
fig.tight_layout(); fig.savefig(CH / "fig5_allocation.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3.2))
lab = [i.replace(" · ", "\n") for i in D42_R.index]
cols = ["#1f4e79" if "공적분" in i else "#bfbfbf" for i in D42_R.index]
ax.bar(range(len(D42_R)), D42_R["샤프"], color=cols)
ax.set_xticks(range(len(D42_R))); ax.set_xticklabels(lab, fontsize=7); ax.axhline(0, color="k", lw=.7)
ax.grid(alpha=.3, axis="y"); ax.set_ylabel("샤프"); ax.set_title("페어 트레이딩: 짝 고르는 방법 × 규칙 (진한 색 = 공적분 검정 통과 짝만)", fontsize=10)
fig.tight_layout(); fig.savefig(CH / "fig6_pairs.png", dpi=160); plt.close(fig)

# ───────────────────────── 문서
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(1.8)
sec.top_margin = sec.bottom_margin = Cm(1.5)
stl = doc.styles["Normal"]
stl.font.name = "Malgun Gothic"
stl.font.size = Pt(9.5)
stl.element.rPr.rFonts.set(qn("w:eastAsia"), "Malgun Gothic")
stl.paragraph_format.space_after = Pt(3)


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    s_ = OxmlElement("w:shd")
    s_.set(qn("w:val"), "clear"); s_.set(qn("w:color"), "auto"); s_.set(qn("w:fill"), color)
    tcPr.append(s_)


def para(text, bold=False, size=None, color=None, after=None, italic=False, align=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = RGBColor.from_string(color)
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    if align:
        p.alignment = align
    return p


def h(text):
    p = para(text, bold=True, size=12, color=NAVY, after=3)
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.keep_with_next = True
    pPr = p._p.get_or_add_pPr()
    b = OxmlElement("w:pBdr"); bt = OxmlElement("w:bottom")
    for k, v in (("val", "single"), ("sz", "6"), ("space", "1"), ("color", NAVY)):
        bt.set(qn(f"w:{k}"), v)
    b.append(bt); pPr.append(b)


def h3(text):
    p = para(text, bold=True, size=10, color=NAVY, after=2)
    p.paragraph_format.keep_with_next = True


def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(1)
    p.add_run(text).font.size = Pt(9.5)


def table(rows, widths=None, size=8, hl=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, rw in enumerate(rows):
        for j, v in enumerate(rw):
            c = t.cell(i, j); c.text = ""
            p = c.paragraphs[0]; r = p.add_run(str(v)); r.font.size = Pt(size)
            p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            if i == 0:
                r.bold = True; r.font.color.rgb = RGBColor(255, 255, 255); shade(c, NAVY)
            elif hl and i in hl:
                shade(c, "DEEAF6")
    if widths:
        for rw in t.rows:
            for j, w_ in enumerate(widths):
                rw.cells[j].width = Cm(w_)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def pic(name, w=17):
    doc.add_picture(str(CH / name), width=Cm(w))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


# 표지 요약
para("딥러닝 전략 4종 검증 보고서", bold=True, size=18, color=NAVY, after=2)
para("MLP 자산배분 · 자산 히트맵 CNN · 한미 시차 LSTM · 오토인코더 페어 트레이딩 — 워크포워드, 비용 차감, 표본 밖 검증", size=10, color="595959", after=6)
para("2026-11-21 · stock_db(국내·미국 일별 시세, ETF, 거시 지표) · 포트폴리오용 연구, 실제 매매 없음", size=8.5, color="7F7F7F", after=8)
table([
    ["전략", "판정", "핵심 숫자"],
    ["한미 시차 → 반도체 ETF 장중 매수\n(미국 밤사이 정보로 확률 상위 날만)", "살아남음",
     f"KODEX 반도체 1회 +{cost_k['1회 평균(비용 전)'] * 100:.2f}% (t {cost_k['t값']:.2f}, {int(cost_k['매매 수'])}회), 샤프 {cost_k['샤프@0.05%']:.2f}\n"
     f"학습에 안 쓴 TIGER 반도체 t {cost_t['t값']:.2f}, 손익분기 비용 {cost_k['손익분기 왕복비용'] * 100:.2f}%"],
    ["MLP + 모멘텀 앙상블 (ETF 10개 자산배분)", "보조",
     f"2012~ 샤프 {A13_SPLIT.loc['2012~ (1-2와 같은 기간) · 앙상블(MLP기본+모멘텀) · 매달', '샤프']:.2f} vs 모멘텀 {A13_SPLIT.loc['2012~ (1-2와 같은 기간) · 모멘텀 · 매달', '샤프']:.2f}, "
     f"2006~2011엔 {A13_SPLIT.loc['2006~2011 (2012 전, 금융위기 포함) · 앙상블(MLP기본+모멘텀) · 매달', '샤프']:.2f} vs {A13_SPLIT.loc['2006~2011 (2012 전, 금융위기 포함) · 모멘텀 · 매달', '샤프']:.2f}"],
    ["자산 히트맵 CNN (방향·급락 위험)", "버림", "방향 AUC 0.49, 급락 AUC 0.51 < 모델 없는 20일 변동성 0.70"],
    ["오토인코더 페어 트레이딩", "버림(공적분 필터만 남김)",
     f"공적분 짝만: 샤프 −0.07 → {D42_R.loc['오토인코더 · 업종+공적분', '샤프']:.2f}, 하지만 연 1~2.5%, 자금 활용 보완 모두 개선 없음"],
], widths=[5.5, 2.6, 9.3], size=8, hl=[1])

h("1. 무엇을 했나")
bullet("공부한 딥러닝 모델 4가지(MLP·CNN·LSTM·오토인코더)를 가진 데이터로 실제 투자 문제에 적용했다. 각 전략은 '처음 실험 → 결과에서 나온 할 일 → 검증·보완' 3단계로 돌렸다.")
bullet("공통 규칙: ① 매년 그 전까지의 자료로만 다시 학습(워크포워드) ② 아침 시가 시점·월말 시점에 알 수 있는 값만 입력 ③ 거래비용 차감 ④ 신경망과 같은 입력의 단순 모델(로지스틱·모멘텀·변동성)을 반드시 함께 비교 ⑤ 결과는 실행할 때마다 결과요약 파일로 자동 저장.")
bullet("여러 규칙을 시험한 만큼 우연히 좋은 것이 나올 수 있어, 45개 규칙 기준 t > 3.3(0.05/45 양쪽)인 것만 '통과'로 판정했다.")

h("2. 한미 시차: 미국 밤사이 정보로 반도체 장중 방향 맞히기")
h3("아이디어와 사전 진단")
bullet("미국 장은 한국 새벽에 끝난다. 전날 밤 SOXX(미국 반도체)가 크게 오르면 삼성전자·SK하이닉스는 시가가 높게 뜬다(SOXX-갭 상관 평균 0.57). 질문은 '시가 이후 장중(시가→종가)에도 이어지나'였다.")
bullet("결과는 반대였다. 미국 반도체가 가장 많이 오른 5분위 날, 시가는 평균 +1~2% 높게 떴지만 장중엔 삼성전자 −0.19%, SK하이닉스 −0.29%(t −3.3)로 되돌렸다. 가장 많이 빠진 날은 장중 반등(+0.15~0.24%). 즉 미국 정보는 시가에 과하게 반영된다.")
pic("fig3_overnight.png", 16.5)
h3("모델과 매매 규칙")
bullet("입력: 최근 20거래일의 미국 SPY·QQQ·SOXX 밤사이 수익, SOXX−SPY, SOXX 5일 수익, 한국 시가 갭, 전날 장중·종가 수익, 5일 수익, 거래량 z, 종목 구분. 정답: 그날 시가→종가 > 0.")
bullet("모델: LSTM(32) vs 로지스틱(마지막 날 값만). 2014년부터 매년 재학습, 삼성전자·SK하이닉스·KODEX 반도체를 한 모델로 학습.")
bullet("매매: 예측확률이 그해 학습구간 확률 분포의 상위 20%(또는 10%)에 들면 시가 매수 → 종가 매도, 아니면 현금.")
h3("첫 실험이 실패한 이유와 고친 것")
bullet("주식 2종목을 '평균보다 높은 날' 사면 비용 전 1회 +0.13%(t 4.4)였지만 왕복 비용 0.25%(매도세 포함)를 못 넘어 샤프 −0.96.")
bullet("고친 것 두 가지: ① 매도세가 없는 반도체 ETF(왕복 0.05% 가정) ② 확신 높은 날(상위 20%·10%)만. 이 둘로 1회 수익이 비용보다 커졌다.")
h3("결과")
table([
    ["규칙 (2014~2026-11)", "매매 수", "1회(비용 전)", "t", "샤프", "연수익", "MDD", "플러스 해"],
    [f"{K} 로지스틱 상위20%", int(cost_k["매매 수"]), pc(cost_k["1회 평균(비용 전)"], 2), f2(cost_k["t값"]), f2(st_k["샤프"]), pc(st_k["CAGR"]), pc(st_k["MDD"]), st_k["플러스 해"]],
    [f"{K} 로지스틱 상위20% 확률 비례 금액", int(cost_k["매매 수"]), pc(cost_k["1회 평균(비용 전)"], 2), f2(cost_k["t값"]), f2(st_kp["샤프"]), pc(st_kp["CAGR"]), pc(st_kp["MDD"]), st_kp["플러스 해"]],
    [f"{T} 로지스틱 상위20% (학습 안 씀)", int(cost_t["매매 수"]), pc(cost_t["1회 평균(비용 전)"], 2), f2(cost_t["t값"]), f2(st_t["샤프"]), pc(st_t["CAGR"]), pc(st_t["MDD"]), st_t["플러스 해"]],
    [f"{K} 그냥 보유(참고)", "-", "-", "-", f2(st_hold["샤프"]), pc(st_hold["CAGR"]), pc(st_hold["MDD"]), st_hold["플러스 해"]],
], widths=[5.6, 1.4, 1.7, 1.0, 1.2, 1.5, 1.5, 1.6], size=7.5, hl=[2])
pic("fig1_semis_cum.png", 16.5)
bullet(f"보유와 연수익은 비슷하지만(13.7% vs 14.4%) 시장에 있는 날이 25%뿐이라 변동성이 34.5% → 14.7%, 최대 낙폭이 −54.7% → −19.2%로 줄었다.")
bullet("SK하이닉스는 주식인데도 LSTM 상위 20% 날 1회 +0.44%(t 4.53)로 왕복 0.25%를 넘었다(비용 후 +0.19%). 삼성전자는 통과 못 함.")

h("3. 검증: 이 결과를 믿어도 되나")
h3("① 비용을 올려도 남는가")
bullet(f"손익분기 왕복 비용(= 비용 전 1회 평균): 상위 20% {cost_k['손익분기 왕복비용'] * 100:.2f}%, 상위 10% {C33_COST.loc[f'{K} · 로지스틱 상위10%', '손익분기 왕복비용'] * 100:.2f}%.")
bullet(f"왕복 0.15%로 올리면 샤프 {cost_k['샤프@0.15%']:.2f}(연 {pc(cost_k['CAGR@0.15%'])}), 0.20%면 {cost_k['샤프@0.20%']:.2f}. 상위 10%는 0.20%에서도 {C33_COST.loc[f'{K} · 로지스틱 상위10%', '샤프@0.20%']:.2f}.")
bullet(f"실제 비용의 단서인 종가 NAV 괴리율은 평균 {C33_NAV.loc[K, '괴리율 절댓값 평균'] * 100:.2f}%(최근 1년 {C33_NAV.loc[K, '최근 1년 평균'] * 100:.2f}%). 시가 체결 비용이 0.1~0.2%라면 기대 샤프는 0.5~0.7 수준으로 본다.")
pic("fig2_cost.png", 15.5)
h3("② 학습에 안 쓴 상품에서도 통하는가")
bullet(f"같은 모델을 학습에 넣지 않은 TIGER 반도체에 그대로 적용: 1회 +{cost_t['1회 평균(비용 전)'] * 100:.2f}%, t {cost_t['t값']:.2f}, 샤프 {cost_t['샤프@0.05%']:.2f}. KODEX와 거의 같아 특정 상품에 맞춘 결과가 아니다.")
h3("③ 미래를 모르는 상태로 고정해도 통하는가")
hk = C33_H.loc[f"{K} · 로지스틱 상위20% · 2024년까지 고정"]
ht = C33_H.loc[f"{T} · 로지스틱 상위20% · 2024년까지 고정"]
bullet(f"2024년까지 자료로 한 번만 학습하고 2025년~ 예측: {K} 1회 +{hk['1회 평균(비용 전)'] * 100:.2f}%(t {hk['t값']:.2f}, {int(hk['매매 수'])}회), {T} +{ht['1회 평균(비용 전)'] * 100:.2f}%(t {ht['t값']:.2f}). 매년 재학습과 비슷.")
h3("④ 특정 해에 기대는가")
bullet(f"2020년과 2026년을 빼면 샤프 {st_k['2020·2026 뺀 샤프']:.2f}, 연 {pc(st_k['2020·2026 뺀 CAGR'])}. 상위 2개 해가 로그 수익의 {pc(st_k['상위 2개 해 비중'], 0)}를 차지한다 → 효과는 있지만 변동성이 큰 해에 몰린다.")
bullet(f"변동성 구간별: 최근 20일 장중 변동성이 높은 3분의 1 구간에서만 1회 +{C33_V.loc[f'{K} · 3 높음', '1회 평균(비용 전)'] * 100:.2f}%(t {C33_V.loc[f'{K} · 3 높음', 't값']:.2f}), 낮은 구간은 +{C33_V.loc[f'{K} · 1 낮음', '1회 평균(비용 전)'] * 100:.2f}%. 반도체가 크게 흔들리는 시기에 쓰는 신호다.")
pic("fig4_yearly.png", 16.5)
h3("⑤ 보완 규칙")
table([["보완", "샤프", "연수익", "MDD", "플러스 해", "2020·2026 뺀 샤프"]] + [
    [n.replace(f"{K} ", ""), f2(C33_ST.loc[n, "샤프"]), pc(C33_ST.loc[n, "CAGR"]), pc(C33_ST.loc[n, "MDD"]), C33_ST.loc[n, "플러스 해"], f2(C33_ST.loc[n, "2020·2026 뺀 샤프"])]
    for n in [f"{K} 로지스틱 상위20%", f"{K} 로지스틱 상위20% 확률비례", f"{K} 로지스틱 상위20% + SOXX 200일선 위",
              f"{K} 두 모델 동의", f"{K} 로지스틱 상위20% 변동성맞춤"]
], widths=[6.5, 1.5, 1.8, 1.8, 1.8, 2.8], size=8, hl=[2])
bullet("확률에 비례해 금액을 조절(백분위 80% → 0.5, 100% → 1.0)한 것이 가장 나았다: 샤프 1.00, 최대 낙폭 −13.4%, 13년 중 11년 플러스.")
bullet("LSTM은 재학습할 때마다 결과가 흔들렸다(같은 설정에서 t 3.4 → 1.7). 로지스틱이 더 안정적이라 최종 모델은 로지스틱, LSTM은 '두 모델 동의' 확인용으로만 둔다.")
h3("⑥ 되돌림·미국판")
bullet("SOXX가 크게 오른 다음 날 장중 매도는 삼성전자 +0.18%(t 2.75), SK하이닉스 +0.32%(t 3.71)지만 공매도가 필요하다. 반도체 인버스 ETF는 국내에 없어 코스피200 인버스로 대신 보면 +0.09%(t 2.04)로 약해진다.")
us = {c: perf(C32_US[c], 252) for c in C32_US.columns}
bullet(f"미국 대형주 150개 갭 반전(갭 하위 10% 매수·상위 10% 매도, 장중): 신호는 있으나(순위상관 t −9.0) 비용 전 샤프 {us[C32_US.columns[0]]['샤프']:.2f} → 비용 후 {us[C32_US.columns[1]]['샤프']:.2f}, 다음 날 종가→종가 {us[C32_US.columns[2]]['샤프']:.2f}. 탈락.")

h("4. MLP 자산배분: ETF 10개 중 다음 달 상위 3개")
bullet("입력: ETF별 1·3·6·12개월 수익, 3개월 변동성, 10개월선 위치와 같은 달 안 순위. 정답: 다음 달 수익 − 그달 평균. 매년 재학습, 시드 3개 평균.")
bullet("MLP 단독은 12개월 모멘텀보다 못했다(2012~ 샤프 0.60 vs 0.68, 회전율 97%). 순위 정답·거시 특성(금리·금리차·신용가산금리·VIX·달러)도 개선 없음(거시 0.45).")
bullet("MLP 순위 + 모멘텀 순위를 더한 앙상블이 2012~ 샤프 0.79~0.80으로 가장 좋았고 3개월 보유로 월 회전율을 73% → 36%로 줄여도 유지됐다(샤프 0.77).")
split = A13_SPLIT
rows = [["기간 · 규칙", "연수익", "샤프", "MDD"]]
for p_ in ["2006~2011 (2012 전, 금융위기 포함)", "2012~ (1-2와 같은 기간)"]:
    for k_ in ["앙상블(MLP기본+모멘텀) · 매달", "모멘텀 · 매달", "MLP 기본 · 매달", "10개 같은 비중", "SPY 보유"]:
        r_ = split.loc[f"{p_} · {k_}"]
        rows.append([f"{p_.split(' ')[0]} · {k_}", pc(r_["CAGR"]), f2(r_["샤프"]), pc(r_["MDD"])])
table(rows, widths=[8.5, 2.2, 2.2, 2.2], size=8, hl=[1, 6])
pic("fig5_allocation.png", 16.5)
bullet("그러나 기간을 2006년으로 늘리자 앙상블의 우위가 사라졌다(2006~2011 샤프 0.40 vs 모멘텀 0.43, MLP 단독 0.53). 전체 2006~2026은 0.62 vs 0.60. → 앙상블 효과는 2012년 이후 구간에 맞춘 결과일 수 있어 '보조'로 둔다.")

h("5. 자산 히트맵 CNN: 버림")
bullet("10개 자산의 최근 20일 표준화 수익률을 20×10 그림으로 만들어 CNN이 SPY 다음 5일을 맞히게 했다. 방향 AUC 0.49(로지스틱 0.51) → 예측력 없음.")
bullet("정답을 '5일 안 −3% 급락'으로 바꿔도 CNN AUC 0.51. 모델 없이 최근 20일 변동성만 본 것이 0.70으로 훨씬 나았다. 200일선에 CNN을 더하면 샤프 1.00 → 0.88로 오히려 나빠짐.")
b22 = {c: perf(B22_D[c], 252) for c in ["SPY 보유", "200일선 위에서만 보유", "200일선 위 + CNN 위험 낮을 때만", "200일선 위 + 변동성 낮을 때만(모델 없음)"]}
table([["규칙 (2012~)", "샤프", "연수익", "MDD"]] + [[k_, f2(v["샤프"]), pc(v["CAGR"]), pc(v["MDD"])] for k_, v in b22.items()],
      widths=[8.5, 2.2, 2.2, 2.2], size=8)
bullet("결론: 여러 자산 그림으로 시장 타이밍을 잡는 데 CNN이 더하는 것이 없고, 급락 위험은 단순 변동성이 대부분 설명한다.")

h("6. 오토인코더 페어 트레이딩: 버림 (공적분 필터만 남김)")
bullet("종목 하나 = 1년치 수익률 252개를 오토인코더로 6차원으로 압축해 잠재공간에서 가까운 두 종목을 짝으로, 가격 차이가 ±2 표준편차 벌어지면 되돌림에 베팅(직전 해 거래대금 상위 100, 6개월마다 짝 10쌍).")
bullet("처음 설정은 세 방법(오토인코더·PCA·상관) 모두 비용 후 마이너스, 손절 40%. 같은 업종 제한은 효과 없고, 공적분 검정(p < 0.05) 통과 짝만 쓰자 모두 플러스로 돌아섰다(오토인코더+업종+공적분 샤프 0.47).")
pic("fig6_pairs.png", 16.5)
bullet(f"하지만 통과 짝이 기간당 4~5개라 자금이 놀아 연 1~2.5%. 짝 후보 늘리기·열린 짝에만 배분·매달 겹치기·놀린 돈 이자·변동성 맞춤을 하나씩 더해도 샤프가 {D43_L['샤프'].iloc[0]:.2f}에서 {D43_L['샤프'].min():.2f}~{D43_L['샤프'].iloc[1:].max():.2f} 사이로 나아지지 않았다.")
bullet(f"오토인코더 보완(시장 움직임 빼고 학습·시드 3개 평균·잡음 제거)도 기본({D43_AE.loc['오토인코더', '샤프']:.2f})보다 낮았다({D43_AE.loc['오토인코더(시장제거)', '샤프']:.2f}~{D43_AE.loc['오토인코더(시드3평균)', '샤프']:.2f}). 남길 것은 '짝은 공적분 검정으로 거른다'는 원칙 하나.")

h("7. 한계와 다음 단계")
bullet("시가 동시호가에 그대로 체결된다고 가정했다. 실제 시가 체결 비용은 확인하지 못했고 괴리율로 크기만 짐작했다 → 비용 0.15%·0.20% 결과를 같이 본다.")
bullet("반도체 효과가 변동성 큰 해(2020·2026)에 몰려 있다. 앞으로 2026-11 이후 매일 신호를 기록해(실제 매매 없이) 진짜 표본 밖 성과를 쌓는다.")
bullet("되돌림(장중 매도)은 개인 공매도가 어려워, 삼성전자·SK하이닉스 주식선물 매도로 할 수 있는지 다음 단계에서 확인한다.")
bullet("오토인코더에 재무 특성(시가총액·PER·PBR·PSR)을 넣은 실험은 재무 자료가 2022년 이후만 있어 8개 기간뿐이었다(샤프 0.26 vs 0.15). DART 재무 수집이 끝나면 약 20개 기간으로 다시 본다.")
bullet("국내 종목은 현재 상장 종목 위주(생존 편향), 업종은 현재 시점 하나만 있다. 미국 갭 반전은 현재 S&P500 구성 상위 종목이다.")

h("8. 결론")
bullet("딥러닝 모델 4개 중 살아남은 것은 '미국 밤사이 정보 → 반도체 ETF 장중' 하나이고, 살아남은 모델도 LSTM이 아니라 로지스틱이었다. 핵심은 모델이 아니라 ① 미국 정보가 시가에 과하게 반영된다는 관찰 ② 비용이 낮은 상품 ③ 확신 높은 날만 매매였다.")
bullet("검증 5가지(비용 0.15%, 학습 안 쓴 상품, 2024년 고정 모델, 특정 해 제외, 여러 번 시험 보정)를 모두 통과했지만, 수익이 변동성 큰 시기에 몰리므로 확률 비례 금액으로 운용하는 것이 가장 안정적이었다.")
bullet("CNN 시장 타이밍·오토인코더 페어·미국 갭 반전은 버리고, 공적분 필터와 '단순 변동성이 급락 위험을 더 잘 맞힌다'는 관찰은 신호 장부에 남긴다.")

doc.save(OUT)
print(OUT)
try:
    from docx2pdf import convert
    convert(str(OUT), str(OUT.with_suffix(".pdf")))
    print(OUT.with_suffix(".pdf"))
except Exception as e:
    print("PDF 변환 실패:", e)
