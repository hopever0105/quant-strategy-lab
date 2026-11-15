# -*- coding: utf-8 -*-
"""보고서 생성: results/ → 차트(charts/*.png) + 보고서_초안.docx (+pdf)
실행: python build_report.py [최종]   ('최종'이면 중형가치주_ETF_검증_보고서.docx)"""
import sys
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

import data as DT

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
CH = HERE / "charts"
CH.mkdir(exist_ok=True)
FINAL = len(sys.argv) > 1 and sys.argv[1] == "최종"
OUT = HERE / ("중형가치주_ETF_검증_보고서.docx" if FINAL else "보고서_초안.docx")
NAVY = "1F3864"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
COL = {"VOE": "#1f4e79", "IWS": "#5b9bd5", "FSMVX": "#70ad47", "FLPSX": "#a9d18e", "SPY": "#7f7f7f", "QQQ": "#c55a11",
       "IJJ": "#9dc3e6", "VO": "#2e75b6", "IWD": "#bf9000"}


def rd(f, **k):
    return pd.read_csv(RES / f"{f}.csv", encoding="utf-8-sig", **k).drop(columns=["Unnamed: 0"], errors="ignore")


def pc(x, d=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x * 100:.{d}f}%"


def pp(x, d=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x * 100:+.{d}f}%p"


def f2(x):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"


A = rd("a_perf")
AY = pd.read_csv(RES / "a_yearly.csv", encoding="utf-8-sig", index_col=0)
AW = rd("a_rolling_win")
AC = rd("a_capture")
FF = rd("b_ff6")
GAP = rd("b_gap_decomp")
ST = rd("b_style_2x2")
DEC = pd.read_csv(RES / "b_ff_decades.csv", encoding="utf-8-sig", index_col=0)
RB = rd("b_rolling_beta", parse_dates=["날짜"])
REG = rd("b_regime")
SW = rd("c_switch")
SWC = pd.read_csv(RES / "c_switch_chosen.csv", encoding="utf-8-sig", index_col=0).iloc[:, 0]
KR = rd("d_kr_results")
KCH = pd.read_csv(RES / "d_kr_chosen.csv", encoding="utf-8-sig", index_col=0).iloc[:, 0]
KY = pd.read_csv(RES / "d_kr_yearly.csv", encoding="utf-8-sig", index_col=0)
KEX = rd("d_kr_ex2025")
KCON = rd("d_kr_concentration")
KU = rd("d_kr_universe")
RR = rd("r_rate_regime")
RSW = rd("r_rate_switch")
RFD = rd("r_fidelity")
RBAND = rd("r_kr_band")
RBLK = rd("r_kr_blocks")
RKR = rd("r_kr_rate")
RCOR = rd("r_regime_corr")
MM1, MM2, MM3, MM4, MM5 = rd("m1_same_month"), rd("m2_rate_trend"), rd("m3_explained"), rd("m4_reversal"), rd("m5_blocks")
MM6, MM7 = rd("m6_rotation"), rd("m7_volume")
KRET = pd.read_pickle(RES / "d_kr_returns.pkl")


def ap(t, s):
    return A[(A.종목 == t) & (A.구간 == s)].iloc[0]


def ffr(t, s="전체"):
    return FF[(FF.종목 == t) & (FF.구간 == s)].iloc[0]


def kr(ver, m, v, w, s):
    r = KR[(KR.판 == ver) & (KR.중형 == m) & (KR.가치.fillna("") == v) & (KR.비중.fillna("") == w) & (KR.구간 == s)]
    return r.iloc[0] if len(r) else None


# ───────────────────────── 차트
R = DT.us_returns()
st0 = "2006-09-01"
fig, ax = plt.subplots(figsize=(10, 4.2))
for t in ["VOE", "IWS", "FSMVX", "FLPSX", "SPY", "QQQ"]:
    r = R[t][st0:].fillna(0)
    ax.plot((1 + r).cumprod(), label=t, color=COL[t], lw=2.2 if t in ("VOE", "SPY", "QQQ") else 1.2)
ax.set_yscale("log"); ax.grid(alpha=.3); ax.legend(ncol=6, fontsize=8, loc="upper left")
ax.set_title("1달러가 얼마가 됐나 (2006-09 = 1, 분배금 재투자, 로그 눈금)", fontsize=10)
fig.tight_layout(); fig.savefig(CH / "fig1_us_growth.png", dpi=160); plt.close(fig)

M = pd.DataFrame({t: (1 + R[t].dropna()).resample("ME").prod() - 1 for t in ["VOE", "IWS", "SPY", "QQQ"]})
fig, ax = plt.subplots(figsize=(10, 3.6))
for t, b, c, ls in [("IWS", "SPY", "#1f4e79", "-"), ("IWS", "QQQ", "#c55a11", "-")]:
    g = (1 + M[[t, b]].dropna()).rolling(36).apply(np.prod, raw=True) ** (1 / 3) - 1
    ax.plot((g[t] - g[b]) * 100, color=c, ls=ls, label=f"{t} - {b}")
ax.axhline(0, color="k", lw=.7); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.set_ylabel("연율 차이(%p)"); ax.set_title("3년 이동 연수익 차이: 중형 가치(IWS) - SPY / QQQ", fontsize=10)
fig.tight_layout(); fig.savefig(CH / "fig2_rolling_gap.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3.4))
g = RB[RB.종목 == "IWS"].set_index("날짜")
for c, col in [("HML", "#1f4e79"), ("SMB", "#c55a11"), ("RMW", "#70ad47"), ("CMA", "#bf9000"), ("MOM", "#7f7f7f")]:
    ax.plot(g[c], label=c, color=col)
ax.axhline(0, color="k", lw=.7); ax.grid(alpha=.3); ax.legend(ncol=5, fontsize=8)
ax.set_title("IWS(러셀 중형 가치)의 파마-프렌치 팩터 노출, 3년 이동 회귀", fontsize=10)
fig.tight_layout(); fig.savefig(CH / "fig3_rolling_beta.png", dpi=160); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
parts = ["알파차이", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
pcol = ["#7f7f7f", "#1f4e79", "#c55a11", "#5b9bd5", "#70ad47", "#bf9000", "#a5a5a5"]
for a, b in zip(axs, ["SPY", "QQQ"]):
    d = GAP[(GAP.펀드 == "VOE") & (GAP.비교 == b) & (GAP.구간 != "전체")].set_index("구간")
    x = np.arange(len(d))
    pos, neg = np.zeros(len(d)), np.zeros(len(d))
    for p_, c_ in zip(parts, pcol):
        v = d[p_].values * 100
        a.bar(x, np.where(v > 0, v, 0), bottom=pos, color=c_, label=p_ if b == "SPY" else None)
        a.bar(x, np.where(v < 0, v, 0), bottom=neg, color=c_)
        pos += np.where(v > 0, v, 0); neg += np.where(v < 0, v, 0)
    a.scatter(x, d["실제차이"] * 100, color="k", zorder=5, marker="D", label="실제 차이" if b == "SPY" else None)
    a.set_xticks(x); a.set_xticklabels(d.index); a.axhline(0, color="k", lw=.7); a.grid(alpha=.3, axis="y")
    a.set_title(f"VOE - {b} 연 수익 차이 분해", fontsize=10)
axs[0].set_ylabel("%p / 년"); fig.legend(ncol=8, fontsize=7.5, loc="lower center")
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(CH / "fig4_gap_decomp.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3.4))
lab, val, cc = [], [], []
for _, r in RR[(RR.금리기간 == "3개월")].iterrows():
    lab.append(f"{r.대상}\n{r.구간}".replace("−", "-"))
    val.append(r.차이 * 100)
    cc.append("#c55a11" if "표본 밖" in r.구간 else "#1f4e79")
ax.bar(range(len(val)), val, color=cc)
ax.set_xticks(range(len(val))); ax.set_xticklabels(lab, fontsize=6.5); ax.axhline(0, color="k", lw=.7); ax.grid(alpha=.3, axis="y")
ax.set_ylabel("다음 달 차이(연율 %p)")
ax.set_title("10년 금리가 3개월 동안 오른 뒤 다음 달 − 내린 뒤 다음 달 (파랑: 2001~2026, 주황: 1963~2000 표본 밖)".replace("−", "-"), fontsize=9)
fig.tight_layout(); fig.savefig(CH / "fig5_rate_regime.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4))
for k, nm, c, lw in [(KCH["장기"], "선택: 중형(CRSP 70~85%) 가치 상위30% 동일가중", "#1f4e79", 2.2),
                     ("장기|CRSP|전체|시총", "중형 전체(시총가중)", "#5b9bd5", 1.2),
                     ("장기|대형|절반|시총", "대형 가치 절반(시총가중)", "#bf9000", 1.2),
                     ("장기|코스피지수", "코스피 지수", "#7f7f7f", 2.0)]:
    r = KRET[k].fillna(0)
    ax.plot((1 + r).cumprod(), label=nm, color=c, lw=lw)
ax.set_yscale("log"); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.set_title("한국판 중형 가치: 1원이 얼마가 됐나 (2006-05 ~ 2026-10, 편도 0.25% 비용 차감)", fontsize=10)
fig.tight_layout(); fig.savefig(CH / "fig6_kr_growth.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3.2))
yy = KY[[KCH["장기"], "장기|코스피지수"]].loc[2007:]
x = np.arange(len(yy))
ax.bar(x - 0.2, yy.iloc[:, 0] * 100, 0.4, color="#1f4e79", label="중형 가치(선택)")
ax.bar(x + 0.2, yy.iloc[:, 1] * 100, 0.4, color="#7f7f7f", label="코스피")
ax.set_xticks(x); ax.set_xticklabels(yy.index, fontsize=7); ax.axhline(0, color="k", lw=.7); ax.grid(alpha=.3, axis="y")
ax.legend(fontsize=8); ax.set_title("연도별 수익률 (2026은 10월까지)", fontsize=10)
fig.tight_layout(); fig.savefig(CH / "fig7_kr_yearly.png", dpi=160); plt.close(fig)

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


def para(text, bold=False, size=None, color=None, after=None, italic=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = RGBColor.from_string(color)
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def h(text):
    p = para(text, bold=True, size=12, color=NAVY, after=3)
    p.paragraph_format.space_before = Pt(8)
    pPr = p._p.get_or_add_pPr()
    b = OxmlElement("w:pBdr"); bt = OxmlElement("w:bottom")
    for k, v in (("val", "single"), ("sz", "6"), ("space", "1"), ("color", NAVY)):
        bt.set(qn(f"w:{k}"), v)
    b.append(bt); pPr.append(b)


def h3(text):
    para(text, bold=True, size=10, color=NAVY, after=2)


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


# 주요 숫자
voe, spy, qqq = ap("VOE", "공통(2006-09~)"), ap("SPY", "공통(2006-09~)"), ap("QQQ", "공통(2006-09~)")
iws_all, spy_te, qqq_te, voe_te = ap("IWS", "상장 이후"), ap("SPY", "테스트"), ap("QQQ", "테스트"), ap("VOE", "테스트")
fv = ffr("VOE")
fl = ffr("FLPSX")
g_spy_back = GAP[(GAP.펀드 == "VOE") & (GAP.비교 == "SPY") & (GAP.구간 == "뒤(2015~)")].iloc[0]
g_spy_front = GAP[(GAP.펀드 == "VOE") & (GAP.비교 == "SPY") & (GAP.구간 == "앞(~2014)")].iloc[0]
g_qqq_all = GAP[(GAP.펀드 == "VOE") & (GAP.비교 == "QQQ") & (GAP.구간 == "전체")].iloc[0]
st = ST.set_index(["비교", "구간"])
vin = st.loc[("가치−전체(중형 안)", "전체(2006-09~)")]
mid = st.loc[("중형−대형(전체)", "전체(2006-09~)")]
vq = st.loc[("중형가치−나스닥100", "전체(2006-09~)")]
w_spy = AW[(AW.종목 == "VOE") & (AW.비교 == "SPY") & (AW.창 == "5년")].iloc[0]
w_qqq = AW[(AW.종목 == "VOE") & (AW.비교 == "QQQ") & (AW.창 == "5년")].iloc[0]
w_iws = AW[(AW.종목 == "IWS") & (AW.비교 == "SPY") & (AW.창 == "5년")].iloc[0]
rr = RR.set_index(["대상", "금리기간", "구간"])
r_in = rr.loc[("IWS−SPY", "3개월", "전체")]
r_inq = rr.loc[("IWS−QQQ", "3개월", "전체")]
r_oos = rr.loc[("합성(VOE−SPY 팩터 노출)", "3개월", "1963~2000(표본 밖)")]
r_oos6 = rr.loc[("합성(VOE−SPY 팩터 노출)", "6개월", "1963~2000(표본 밖)")]
r_hml = rr.loc[("HML 팩터", "3개월", "1963~2000(표본 밖)")]
ml, mc = KCH["장기"].split("|"), KCH["CRSP"].split("|")
k_tr, k_va, k_te, k_all = (kr("장기", ml[1], ml[2], ml[3], s) for s in ["학습", "검증", "테스트", "전체"])
ks = {s: kr("장기", "코스피지수", "", "", s) for s in ["학습", "검증", "테스트", "전체"]}
kc_va, kc_te, kc_all = (kr("CRSP", mc[1], mc[2], mc[3], s) for s in ["검증", "테스트", "전체"])
kcs = {s: kr("CRSP", "코스피지수", "", "", s) for s in ["검증", "테스트", "전체"]}
midall = {s: kr("장기", "CRSP", "전체", "시총", s) for s in ["학습", "검증", "테스트", "전체"]}
lgv = {s: kr("장기", "대형", "절반", "시총", s) for s in ["학습", "검증", "테스트", "전체"]}
n_cand = len(KR[(KR.기준 == "후보")][["판", "중형", "가치", "비중"]].drop_duplicates())
ex = KEX.set_index(["포트", "조건"])
kcon = KCON.iloc[0]
rkr = RKR.set_index("상태")

title = "래리 윌리엄스 추천 중형 가치주 ETF 검증" + ("" if FINAL else " (초안)")
para(title, bold=True, size=17, color=NAVY, after=2)
para("VOE·Fidelity 중형 가치 펀드 vs SPY·QQQ → 무엇이 성과를 갈랐나 → 한국판 중형 가치 지수", size=10.5, color="595959", after=1)
para(f"작성 2026-11-15 · 미국 2001~2026-11, 한국 2006-05~2026-10 · 실제 매매 없음(포트폴리오 연구용)", size=8.5, color="7f7f7f", after=6)

h("1. 요약")
bullet(f"중형 가치 ETF는 2006-09 이후 시장을 이기지 못했다. VOE 연 {pc(voe.CAGR)}·샤프 {f2(voe.샤프)} vs SPY {pc(spy.CAGR)}·{f2(spy.샤프)}, "
       f"QQQ {pc(qqq.CAGR)}·{f2(qqq.샤프)}. 최대 낙폭도 VOE {pc(voe.MDD, 0)}로 SPY {pc(spy.MDD, 0)}보다 깊었다.")
bullet(f"5년 보유로 쪼개 봐도 VOE가 SPY를 이긴 구간은 {pc(w_spy.이긴비율, 0)}, QQQ를 이긴 구간은 {pc(w_qqq.이긴비율, 0)}뿐이다. "
       f"2014년까지는 VOE가 SPY보다 연 {pp(g_spy_front.실제차이)} 앞섰고, 2015년 이후 {pp(g_spy_back.실제차이)} 뒤졌다.")
bullet(f"특징(파마-프렌치 6팩터): 시장 베타 {f2(fv['Mkt-RF'])}, 소형 {f2(fv.SMB)}, 가치 {f2(fv.HML)}, 수익성 {f2(fv.RMW)}, "
       f"보수적 투자 {f2(fv.CMA)}, 모멘텀 {f2(fv.MOM)}. 즉 '약간 작고, 싸고, 투자를 덜 하는' 시장 포트폴리오다. 6팩터 알파 {pp(fv['알파(연)'])}(t {f2(fv.알파t)}).")
bullet(f"SPY·QQQ와의 차이는 가치 팩터 때문이 아니다. 같은 중형 안에서 가치−전체 차이는 연 {pp(vin.연평균차이)}(t {f2(vin.t)})로 거의 0이고, "
       f"차이 대부분은 팩터로 설명되지 않는 몫(2015년 이후 SPY 대비 {pp(g_spy_back.알파차이)})과 시장 베타 차이 — 대형 기술주 쏠림 — 에서 나왔다.")
bullet(f"사후 발견 '10년 금리가 3개월 오른 뒤 다음 달엔 중형 가치가 이긴다': 2001~2026 IWS−SPY 연 {pp(r_in.차이)}(t {f2(r_in.t)}), "
       f"5년 블록 5개 모두 플러스. 1963~2000엔 없음({pp(r_oos.차이)}). 원인으로 의심한 '성장주가 금리에 더 약하다'와 '성장주 되돌림'은 검정에서 둘 다 기각 "
       "순환매·리밸런싱도 거래량으로 확인되지 않았고, 효과는 기술주가 한 달 늦게 약해지는 데서 나온다 → 2001년 이후 시장에서만 쓰는 보조 신호로 판정.")
bullet(f"한국판(코스피 비금융, 상장폐지 포함): 학습 구간에서 고른 '중형 가치 상위 30% 동일가중'은 학습 연 {pc(k_tr.CAGR)}(코스피 {pc(ks['학습'].CAGR)}, 알파 t {f2(k_tr.t)})였지만 "
       f"검증 {pc(k_va.CAGR)}(코스피 {pc(ks['검증'].CAGR)}), 테스트 {pc(k_te.CAGR)}(코스피 {pc(ks['테스트'].CAGR)})로 시장에 졌다.")

h3("래리 주장별 판정")
table([["주장(이 검증에서 쓴 형태)", "결과", "판정"],
       ["중형 가치 ETF를 사 두면 시장보다 낫다", f"VOE {pc(voe.CAGR)} vs SPY {pc(spy.CAGR)}, 5년 승률 {pc(w_spy.이긴비율, 0)}", "안 맞음"],
       ["대형 성장(나스닥100)보다 낫다", f"연 {pp(vq.연평균차이)}, t {f2(vq.t)}", "안 맞음"],
       ["중형이 대형보다 낫다", f"2014년까지 +, 2015년 이후 −, 전체 {pp(mid.연평균차이)}(t {f2(mid.t)})", "시기에 따라 다름"],
       ["가치주가 낫다(같은 중형 안)", f"{pp(vin.연평균차이)}, t {f2(vin.t)}", "안 맞음(차이 없음)"],
       ["액티브(Fidelity)가 지수보다 낫다", f"FLPSX 6팩터 알파 {pp(fl['알파(연)'])}(t {f2(fl.알파t)}), 2020 이후는 마이너스", "과거엔 맞음, 최근엔 아님"],
       ["한국에서도 중형 가치가 낫다", f"테스트 {pc(k_te.CAGR)} vs 코스피 {pc(ks['테스트'].CAGR)}", "안 맞음"]],
      widths=[6, 8, 3.4])

h("2. 검증 규칙 (사전 등록)")
bullet("래리 윌리엄스가 장기 보유용으로 권한 중형 가치주 상품 → 미국 Vanguard Mid-Cap Value(VOE, CRSP 미국 중형 가치 지수 추종)와 "
       "Fidelity 펀드. Fidelity는 어느 펀드인지 원문 확인이 안 되어 중형 가치 펀드(FSMVX)와 저가주 펀드(FLPSX) 둘 다 넣었다.")
bullet("비교: SPY(S&P500), QQQ(나스닥100). 원인 나누기용: VO(중형 전체), IWD(대형 가치), IWS(러셀 중형 가치, 2001년부터라 VOE보다 김), IJJ·IJH(S&P400).")
bullet("구간(래리 검증 공통): 학습 ~2014 / 검증 2015~2019 / 테스트 2020~. 상품마다 상장일이 달라 학습 구간 시작이 다르다(표에 기간 표시).")
bullet(f"전환 규칙(C): 매달 말 직전 L개월 수익이 높은 쪽(IWS vs SPY, IWS vs QQQ)을 다음 달 보유, L ∈ {{3, 6, 12}} → 6개 조합. 학습 샤프로 고르고 테스트는 한 번. 전환 편도 0.05%.")
bullet(f"한국판(D): 매년 4월 말(연간 재무 공시 후) 코스피 비금융 종목, 상장폐지 포함. 중형 2개(CRSP 누적 시총 70~85% / KRX 101~300위) × "
       f"가치 2개(중형 안 점수 상위 50% / 30%) × 비중 2개(시총 / 동일) = 8개, 두 판(장기 B/P 2006~ / CRSP 3지표 2014~) 합 {n_cand}개. "
       "장기판은 학습 샤프로, CRSP판은 학습 구간이 없어 검증 샤프로 고름. 편도 0.25%(수수료·세금·슬리피지), 2배 민감도.")
bullet("사후 가설: 결과를 본 뒤 생긴 '금리 상승 → 중형 가치 유리'는 사전 등록 조합이 아니라서 구간 나누기·다른 기간(1·3·6개월)·1963~2000 표본 밖으로만 판정했다.")

h("3. 데이터")
table([["항목", "출처·처리"],
       ["미국 상품 일봉", "야후 수정종가(분배금 재투자). 하루 ±40% 넘는 값 0건. 상장일 VOE 2006-08, IWS 2001-08, FSMVX 2001-11, FLPSX 1989-12"],
       ["팩터", "Ken French 5팩터+모멘텀 일간(1963-07~2026-09). 25 포트폴리오는 French 사이트 접속 차단으로 못 받음 → 1963~2000 검증은 팩터 노출로 합성"],
       ["금리", "미국 10년 국채(FRED DGS10, 1962~), 한국 3개월 금리(FRED/OECD 월간)"],
       ["한국 시세", "stock_db 수정종가 + 상장폐지 종목(kor_price_delisted, KRX 보강), 1999~2026-10, 래리 검증 패널 그대로"],
       ["한국 시총·가치", f"4월 말 시가총액(억원), PBR(2006년부터 95%+ 채움), DART 연간 재무(2013 회계연도~, 보완 {int(KU.BP재무보완.sum())}건). 금융업 제외"],
       ["한국 코스피 지수", "야후 ^KS11 (가격지수, 배당 제외 — 포트폴리오도 배당 제외 가격 기준이라 같은 조건)"]],
      widths=[3.2, 14.2], size=7.5)

h("4. 결과 — 미국")
pic("fig1_us_growth.png")
rows = [["상품", "기간", "CAGR", "변동성", "샤프", "MDD", "테스트 CAGR", "테스트 샤프"]]
for t in ["VOE", "IWS", "FSMVX", "FLPSX", "IJJ", "VO", "IWD", "SPY", "QQQ"]:
    a, te = ap(t, "공통(2006-09~)"), ap(t, "테스트")
    rows.append([t, a.기간, pc(a.CAGR), pc(a.변동성), f2(a.샤프), pc(a.MDD, 0), pc(te.CAGR), f2(te.샤프)])
table(rows, widths=[1.8, 3.4, 1.7, 1.7, 1.4, 1.6, 2.2, 2.2], hl=[1, 8, 9])
rows = [["연도"] + ["VOE", "IWS", "FSMVX", "FLPSX", "SPY", "QQQ"]]
for y in AY.index:
    if y < 2007:
        continue
    rows.append([str(y)] + [pc(AY.loc[y, t], 0) for t in ["VOE", "IWS", "FSMVX", "FLPSX", "SPY", "QQQ"]])
table(rows, size=7)
pic("fig2_rolling_gap.png")
cap = AC.set_index("종목")
bullet(f"하락장에서 덜 빠지지도 않았다: SPY가 내린 달 VOE 하락 포착 {f2(cap.loc['VOE','하락포착'])}(1보다 크면 더 빠짐), "
       f"SPY 최악 10% 달 평균 VOE {pc(cap.loc['VOE','SPY최악10퍼센트달_평균'])} vs SPY {pc(cap.loc['VOE','SPY최악10퍼센트달_SPY'])}. "
       f"방어가 된 건 FLPSX뿐(하락 포착 {f2(cap.loc['FLPSX','하락포착'])}).")
bullet(f"IWS(2001~)까지 넓혀도 5년 보유 승률 {pc(w_iws.이긴비율, 0)} — VOE보다 시작이 빨라(2001-08) 조금 높을 뿐 절반 아래.")

h("5. 특성 추출 — 무엇이 차이를 만들었나")
rows = [["상품", "알파(연)", "알파 t", "시장", "소형(SMB)", "가치(HML)", "수익성(RMW)", "보수투자(CMA)", "모멘텀", "R²"]]
for t in ["VOE", "IWS", "FSMVX", "FLPSX", "VO", "IWD", "SPY", "QQQ"]:
    r = ffr(t)
    rows.append([t, pp(r["알파(연)"]), f2(r.알파t), f2(r["Mkt-RF"]), f2(r.SMB), f2(r.HML), f2(r.RMW), f2(r.CMA), f2(r.MOM), f2(r.R2)])
table(rows, size=7.5, hl=[1])
bullet("VOE의 가치 노출(HML 0.28)은 대형 가치 IWD(0.29)와 같은 수준이고, 소형 노출(SMB 0.20)은 중형 전체 VO와 같다. "
       "즉 VOE = '중형' + '가치'를 단순히 더한 상품이며, 수익성·보수적 투자 노출이 플러스라 질 나쁜 싸구려 주식만 담지는 않는다.")
bullet(f"QQQ는 정반대(가치 {f2(ffr('QQQ').HML)}, 수익성 {f2(ffr('QQQ').RMW)}, 보수투자 {f2(ffr('QQQ').CMA)}, 알파 {pp(ffr('QQQ')['알파(연)'])} t {f2(ffr('QQQ').알파t)}). "
       "6팩터로 설명되지 않는 대형 기술주 초과수익이 2015년 이후 격차의 본체다.")
pic("fig4_gap_decomp.png")
bullet(f"분해: VOE−SPY 2015년 이후 {pp(g_spy_back.실제차이)} 중 팩터로 설명 안 되는 몫 {pp(g_spy_back.알파차이)}, 시장 베타 {pp(g_spy_back['Mkt-RF'])}, "
       f"소형 {pp(g_spy_back.SMB)}, 가치 {pp(g_spy_back.HML)}. VOE−QQQ 전체 {pp(g_qqq_all.실제차이)} 중 알파 몫 {pp(g_qqq_all.알파차이)}.")
rows = [["비교(같은 기간 월간)", "2006-09~2014", "2015~2026", "전체", "전체 t"]]
for nm in ["중형−대형(전체)", "가치−전체(중형 안)", "가치−전체(대형 안)", "중형−대형(가치 안)", "중형가치−나스닥100"]:
    rows.append([nm] + [pp(st.loc[(nm, g), "연평균차이"]) for g in ["2006-09~2014", "2015~2026", "전체(2006-09~)"]] + [f2(st.loc[(nm, "전체(2006-09~)"), "t"])])
table(rows, widths=[5, 3, 3, 3, 2])
pic("fig3_rolling_beta.png")
d = DEC
bullet(f"가치 프리미엄 자체가 약해졌다: HML 연평균 1970년대 {pc(d.loc['1970년대','HML'])}, 2000년대 {pc(d.loc['2000년대','HML'])} → 2010년대 {pc(d.loc['2010년대','HML'])}, "
       f"2020년대 {pc(d.loc['2020년대','HML'])}. 소형(SMB)도 2010년대 이후 마이너스. 중형 가치 상품이 기대는 가치·소형 프리미엄이 최근 15년 동안 거의 없었다.")

h3("언제 이기나 (월말 상태 → 다음 달 IWS 상대수익)")
rows = [["비교", "상태", "해당 달", "해당 달 다음 달(연율)", "나머지(연율)", "차이", "t"]]
for _, r in REG.iterrows():
    rows.append([r.비교, r.상태, f"{int(r.해당달)}/{int(r.전체달)}", pp(r.해당달_평균), pp(r.나머지_평균), pp(r.차이_연율), f2(r.t)])
table(rows, size=7.5, widths=[2, 5.2, 1.6, 2.8, 2.2, 1.8, 1.2])
bullet("SPY가 200일선 아래(하락장)일 때 중형 가치가 더 낫지 않았고, 상대 모멘텀(직전 12개월 이긴 쪽이 계속 이기나)도 없었다. 금리 상태만 차이가 컸다 → 6장에서 따로 검증.")

h3("상대 모멘텀 전환 규칙 (사전 등록 6개)")
rows = [["비교", "L(개월)", "구간", "CAGR", "샤프", "비교 ETF 샤프", "IWS 보유 샤프", "알파 t", "전환 수"]]
for _, r in SW.iterrows():
    rows.append([r.비교, int(r.기간L), r.구간, pc(r.CAGR), f2(r.샤프), f2(r.샤프_비교보유), f2(r.샤프_IWS보유), f2(r.t), int(r.전환수)])
hl = [i + 1 for i, (_, r) in enumerate(SW.iterrows()) if r.기간L == SWC.get(r.비교)]
table(rows, size=7, hl=hl)
s_te = SW[(SW.비교 == "SPY") & (SW.기간L == SWC["SPY"]) & (SW.구간 == "테스트")].iloc[0]
q_te = SW[(SW.비교 == "QQQ") & (SW.기간L == SWC["QQQ"]) & (SW.구간 == "테스트")].iloc[0]
bullet(f"학습에서 둘 다 12개월이 뽑혔지만 테스트 샤프 {f2(s_te.샤프)}(SPY 보유 {f2(s_te.샤프_비교보유)}), {f2(q_te.샤프)}(QQQ 보유 {f2(q_te.샤프_비교보유)}) → "
       f"그냥 SPY·QQQ를 들고 있는 것보다 못함. 학습 구간의 SPY 대비 알파(t {f2(SW[(SW.비교 == 'SPY') & (SW.기간L == SWC['SPY']) & (SW.구간 == '학습')].iloc[0].t)})는 검증·테스트에서 이어지지 않았다.")

h("6. 강건성 검증")
h3("① 사후 발견: 금리 상승 → 다음 달 중형 가치 유리?")
pic("fig5_rate_regime.png")
rows = [["대상", "금리 기간", "구간", "상승 뒤(연율)", "하락 뒤(연율)", "차이", "t"]]
for _, r in RR.iterrows():
    rows.append([r.대상, r.금리기간, r.구간, pp(r.상승_다음달), pp(r.하락_다음달), pp(r.차이), f2(r.t)])
table(rows, size=6.8, widths=[4.2, 1.5, 3.2, 2.2, 2.2, 1.8, 1.2])
bullet(f"2001~2026에서는 3개월 기준이 두 구간 모두 플러스(IWS−SPY 연 {pp(r_in.차이)} t {f2(r_in.t)}, IWS−QQQ {pp(r_inq.차이)} t {f2(r_inq.t)})지만, "
       "1개월·6개월로 바꾸면 한쪽 구간에서 사라진다(기간 선택에 민감).")
bullet(f"진짜 표본 밖(1963~2000): VOE와 SPY의 팩터 노출 차이로 만든 합성 수익은 금리 상승 뒤 오히려 나빴다(3개월 {pp(r_oos.차이)} t {f2(r_oos.t)}, "
       f"6개월 {pp(r_oos6.차이)} t {f2(r_oos6.t)}). 가치 팩터(HML)도 {pp(r_hml.차이)}(t {f2(r_hml.t)}).")
rc = RCOR.set_index(["대상", "구간", "성격"])
bullet(f"그러나 1963~2000은 비교 대상이 못 된다: 직전 3년 '주가 vs 금리' 상관이 플러스인 달이 1963~2000엔 {int(rc.loc[('HML','1963~2000','상관>0'),'달'])}개월(2%), "
       "2001년 이후 84%. 즉 2000년 무렵 시장 성격이 바뀌어(금리가 오를 때 주가가 빠지던 시장 → 같이 오르는 시장) 옛 자료로는 지금 시장을 검정할 수 없다.")
h3("①-2 원인 검정: 왜 2001년 이후 금리 상승 뒤 중형 가치가 이기나")
m1 = MM1.set_index(["대상", "구간"]); m2 = MM2.set_index("구간"); m3 = MM3.set_index(["대상", "구간"]); m4 = MM4.set_index("대상")
table([["원인 가설이 맞다면", "결과(2001~2026)", "판정"],
       ["(가) 금리가 오른 같은 달에 중형 가치가 앞서야 함", f"IWS−SPY 금리 1%p당 {pp(m1.loc[('IWS−SPY','2001~2026'),'민감도'])}(t {f2(m1.loc[('IWS−SPY','2001~2026'),'t'])}), "
        f"SPY−QQQ {pp(m1.loc[('SPY−QQQ','2001~2026'),'민감도'])}(t {f2(m1.loc[('SPY−QQQ','2001~2026'),'t'])}) — 오히려 성장주가 앞섬", "기각"],
       ["(나) 금리 오름세가 다음 달에도 이어져야 함", f"3개월 상승 뒤 다음 달 금리 {m2.loc['2001~2026','상승뒤_다음달dy_bp']:+.1f}bp, 다시 오른 달 {pc(m2.loc['2001~2026','다음달도_상승비율'], 0)}", "기각"],
       ["(다) 다음 달 실제 금리 변화를 넣으면 신호가 사라져야 함", f"IWS−SPY 신호 {pp(m3.loc[('IWS−SPY','2001~2026'),'신호만_연율'])} → 넣은 뒤 {pp(m3.loc[('IWS−SPY','2001~2026'),'통제후_연율'])}(t {f2(m3.loc[('IWS−SPY','2001~2026'),'통제후_t'])})", "기각"],
       ["(라) 성장주 되돌림이면 직전 3개월 상대수익이 설명해야 함", f"직전 3개월 상대수익 넣어도 신호 t {f2(m4.loc['IWS−SPY','신호_t'])}, 되돌림 자체 t {f2(m4.loc['IWS−SPY','되돌림만_t'])}", "기각"],
       ["안정성: 5년 블록 방향", "IWS−SPY·IWS−QQQ·IWD−SPY 모두 5/5 블록 플러스", "통과"]],
      widths=[6, 8.6, 1.8], size=7.5)
bullet("결론: '금리가 오르면 성장주가 불리하다(듀레이션)'는 설명은 틀렸다 — 금리가 오르는 달엔 오히려 성장주가 앞섰고, 금리 추세도 이어지지 않았다. "
       "되돌림도 아니다. 그런데 효과는 2001년 이후 다섯 블록과 세 비교에서 모두 같은 방향이다. → 지금 시장(2001~)에서 반복된 규칙이며, 원인은 다음 절(①-3)에서 좁힌다. "
       "대형 가치(IWD−SPY)에도 같은 크기로 나타나므로 '중형'이 아니라 '가치 vs 성장' 전체의 현상이다.")
h3("①-3 순환매인가, 리밸런싱인가 (다음 달 절대수익·거래량)")
m6 = MM6.set_index("경우"); m7 = MM7.set_index("지표")
table([["확인한 것", "신호 뒤 다음 달", "그 밖의 달", "t", "뜻"],
       ["기술주(QQQ) 수익(연율)", pc(m6.loc["신호 있음", "다음달_QQQ"]), pc(m6.loc["신호 없음", "다음달_QQQ"]), "-1.68", "기술주가 한 달 늦게 약해짐"],
       ["가치주(IWS) 수익(연율)", pc(m6.loc["신호 있음", "다음달_IWS"]), pc(m6.loc["신호 없음", "다음달_IWS"]), "0.57", "가치주가 크게 오르진 않음"],
       ["가치 ETF 거래대금 비중", pc(m7.iloc[0].신호뒤_다음달), pc(m7.iloc[0].그밖), f2(m7.iloc[0].t), "가치주로 거래가 몰리지 않음"],
       ["QQQ 매수 우위 대용(오른 날−내린 날 거래량)", pc(m7.loc["QQQ 매수 우위 대용", "신호뒤_다음달"]), pc(m7.loc["QQQ 매수 우위 대용", "그밖"]), f2(m7.loc["QQQ 매수 우위 대용", "t"]), "기술주는 파는 쪽 우세"],
       ["가치 − QQQ 매수 우위 대용 차이", pc(m7.loc["매수 우위 차이 (IWS+IWD)/2 − QQQ", "신호뒤_다음달"]), pc(m7.loc["매수 우위 차이 (IWS+IWD)/2 − QQQ", "그밖"]), f2(m7.loc["매수 우위 차이 (IWS+IWD)/2 − QQQ", "t"]), "주가 결과와 같은 방향"],
       ["리밸런싱: 직전 3개월 QQQ가 더 앞선 경우 효과가 더 큰가", pp(m7.iloc[-1].신호뒤_다음달), pp(m7.iloc[-1].그밖), f2(m7.iloc[-1].t), "기각"]],
      widths=[5.6, 2.2, 2.2, 1.2, 5.2], size=7.5)
bullet("정리: 효과의 대부분은 '금리 상승의 타격이 기술주에 한 달 늦게 오는 것'이다. 기술주에서 돈이 빠지는 흔적(매수 우위 하락)은 있지만, "
       "가치 ETF 거래대금 비중은 그대로라 그 돈이 저평가 가치주로 옮겨 갔다는 증거는 없다(현금·채권 등 다른 곳일 수 있음). 리밸런싱(오른 기술주 비중 줄이기)도 아니다. "
       "매수 우위 대용은 오른 날·내린 날로 나눈 값이라 주가와 독립된 증거는 아니며, 실제 자금 유입(ETF 설정·환매 이력)은 무료 자료가 없어 확인하지 못했다.")
rs = RSW[(RSW.비용 == 0.0005)].set_index(["비교", "구간"])
bullet(f"쓰는 법의 한계: 이 규칙으로 매달 IWS↔SPY를 갈아타면 2015~2019 SPY 대비 알파 t {f2(rs.loc[('SPY','검증'),'t'])}, 2020~ {f2(rs.loc[('SPY','테스트'),'t'])}. "
       "다음 달 상대수익 차이는 일관되지만 전체를 바꿔 타면 시장 베타 차이에 묻힌다 → 단독 매매 신호가 아니라 가치·성장 비중을 조금 기울이는 보조 신호로만 쓴다. "
       "결과를 보고 찾은 신호(6개 상태 중 하나)이므로 2026-12 이후 새 데이터에서 다시 확인한다.")
h3("② Fidelity 액티브는 지수보다 나았나")
rows = [["펀드", "구간", "6팩터 알파", "t", "IWS 대비 알파", "t"]]
for _, r in RFD.iterrows():
    rows.append([r.펀드, r.구간, pp(r.알파6팩터), f2(r.t6), pp(r.알파_vsIWS), f2(r.t_vsIWS)])
table(rows, size=7.5)
bullet("FLPSX는 2014년까지 6팩터 알파 연 4.0%(t 3.3)로 실력이 숫자로 보이지만 2015년 이후 0 근처~마이너스. "
       f"FSMVX는 지수(IWS)보다 나은 구간이 없다(2015~2019엔 6팩터 알파 {pp(RFD[(RFD.펀드 == 'FSMVX') & (RFD.구간 == '검증')].iloc[0].알파6팩터)}, t {f2(RFD[(RFD.펀드 == 'FSMVX') & (RFD.구간 == '검증')].iloc[0].t6)}). 래리 추천을 따랐다면 '언제 샀나'가 결과를 갈랐다.")

h("7. 한국판 중형 가치 (코스피)")
pic("fig6_kr_growth.png")
rows = [["포트폴리오", "학습 06-05~14", "검증 15~19", "테스트 20~26-10", "전체 CAGR", "전체 샤프", "MDD", "알파 t(전체)", "종목 수"]]
order = [("장기", m, v, w) for m in ["CRSP", "KRX"] for v in ["상위30", "절반"] for w in ["동일", "시총"]] + \
        [("장기", "CRSP", "전체", "시총"), ("장기", "KRX", "전체", "시총"), ("장기", "대형", "절반", "시총"), ("장기", "대형", "전체", "시총")]
hl = []
for i, (ver, m, v, w) in enumerate(order):
    ss = [kr(ver, m, v, w, s) for s in ["학습", "검증", "테스트", "전체"]]
    nm = f"{'중형' if m != '대형' else '대형'}({m if m != '대형' else '70% 안'}) {v} {w}"
    rows.append([nm] + [pc(x.CAGR) for x in ss] + [f2(ss[3].샤프), pc(ss[3].MDD, 0), f2(ss[3].t), f"{ss[3].평균종목수:.0f}"])
    if f"{ver}|{m}|{v}|{w}" == KCH["장기"]:
        hl.append(i + 1)
rows.append(["코스피 지수"] + [pc(ks[s].CAGR) for s in ["학습", "검증", "테스트", "전체"]] + [f2(ks["전체"].샤프), pc(ks["전체"].MDD, 0), "-", "-"])
table(rows, size=7, hl=hl, widths=[4.4, 1.8, 1.7, 2, 1.6, 1.4, 1.3, 1.6, 1.2])
bullet(f"장기판(B/P 하나) 8개 중 학습 샤프 1등 = 중형(CRSP 70~85%) 가치 상위 30% 동일가중(평균 {k_all.평균종목수:.0f}종목). "
       f"학습 연 {pc(k_tr.CAGR)}·알파 t {f2(k_tr.t)} → 검증 {pc(k_va.CAGR)}(t {f2(k_va.t)}) → 테스트 {pc(k_te.CAGR)}(t {f2(k_te.t)}). 테스트에서 코스피({pc(ks['테스트'].CAGR)})에 짐.")
bullet(f"CRSP판(B/P·E/P·S/P, 2014~)도 검증에서 고른 {mc[1]} {mc[2]} {mc[3]}가 테스트 {pc(kc_te.CAGR)} vs 코스피 {pc(kcs['테스트'].CAGR)}, "
       f"전체 알파 t {f2(kc_all.t)}. 지표를 늘려도 달라지지 않았다.")
bullet(f"한국에서 오히려 나았던 건 대형 가치(시총 상위 70% 안 B/P 상위 절반): 전체 연 {pc(lgv['전체'].CAGR)} vs 코스피 {pc(ks['전체'].CAGR)}, 알파 t {f2(lgv['전체'].t)}. "
       f"중형 전체는 2015년 이후 대형에 밀렸다(검증 {pc(midall['검증'].CAGR)}, 코스피 {pc(ks['검증'].CAGR)}).")
pic("fig7_kr_yearly.png")
rows = [["블록(3년)", "선택 − 코스피(연)", "선택 − 중형 전체(연)"]]
b = RBLK.pivot(index="블록", columns="비교", values="연차이")
for k_ in b.index:
    rows.append([k_, pp(b.loc[k_, "코스피"]), pp(b.loc[k_, "중형 전체"])])
table(rows, widths=[4, 4, 4])
bullet(f"3년 블록으로 보면 2014년까지 3블록 모두 코스피 대비 플러스, 2015년 이후 4블록 모두 마이너스 → 미국과 같은 '2015년 단절'. "
       f"2025년(반도체 쏠림)을 빼도 알파 t {f2(ex.loc[(KCH['장기'], '2025 제외'), 't'])}.")
bullet(f"기여 집중: {kcon.종목수}개 종목 기여 합 {kcon.총기여합:.2f} 중 상위 5종목 {kcon.상위5합:.2f}({kcon.상위5합 / kcon.총기여합 * 100:.0f}%) — {kcon.상위5}.")
h3("한국 강건성")
rows = [["중형 경계(누적 시총)", "가치", "학습 CAGR", "검증 CAGR", "테스트 CAGR", "전체 알파 t", "종목 수"]]
bb = RBAND.set_index(["경계", "가치", "구간"])
for lo in ["70~85%", "65~85%", "70~90%", "60~80%"]:
    for v in ["상위30", "전체"]:
        rows.append([lo, v] + [pc(bb.loc[(lo, v, s), "CAGR"]) for s in ["학습", "검증", "테스트"]] +
                    [f2(bb.loc[(lo, v, "전체"), "t"]), f"{bb.loc[(lo, v, '전체'), '평균종목수']:.0f}"])
table(rows, size=7.5)
bullet("경계를 바꿔도 모양이 같다: 학습 플러스 → 검증 마이너스. 가치 상위 30%와 중형 전체의 차이도 구간마다 들쭉날쭉.")
bullet(f"비용 2배(편도 0.5%): 선택 조합 테스트 샤프 {f2(k_te.샤프)} → {f2(k_te.샤프_비용2배)}. 회전율이 연 {k_te.회전율:.0%}라 비용은 결론을 바꾸지 않는다.")
bullet(f"금리(탐색): 한국 3개월 금리가 3개월 오른 뒤 다음 달 중형 가치 − 코스피 연 {pp(rkr.loc['한국 3개월 금리 3개월 상승','차이'])}(t {f2(rkr.loc['한국 3개월 금리 3개월 상승','t'])}), "
       f"미국 10년 금리 기준은 {pp(rkr.loc['미국 10년 3개월 상승','차이'])}(t {f2(rkr.loc['미국 10년 3개월 상승','t'])}). 한국에선 미국 금리 기준으로는 안 나타남 → 미국 2001년 이후 현상으로 한정.")

h("8. 한계")
bullet("상품 비교는 실제 상품 수익률이라 생존편향이 없지만, VOE 상장(2006-08) 이후 20년 하나의 경로다. 가치 프리미엄이 약했던 시기와 겹쳐 '중형 가치는 원래 안 된다'까지 말할 수는 없다(1970~2000년대 HML은 플러스).")
bullet("Fidelity는 래리가 말한 펀드가 어느 것인지 확인하지 못해 두 개를 넣었다. 펀드 수익률은 보수 차감 후 기준가(ETF와 같은 조건).")
bullet("French 25 포트폴리오를 못 받아 1963~2000 '중형 가치' 수익은 팩터 노출 × 팩터 합성으로 대신했다(실제 중형 가치 포트폴리오와 다를 수 있음).")
bullet("한국: 시장 구분은 마지막 상장 시장 기준(코스닥→코스피 이전 종목은 과거에도 코스피로 분류). 2006~2013은 KRX PBR, 2026년은 DART 자본으로 B/P를 만들어 출처가 섞임. "
       "상장폐지 뒤 정리매매 이후 가치는 0으로 보지 않고 마지막 가격에서 멈춤(부실 종목 손실이 약간 덜 반영될 수 있음).")
bullet("한국 지수·포트폴리오 모두 배당 제외 가격수익률이다. 배당을 넣으면 가치주 쪽이 약간 유리해질 수 있다(배당 자료로 확인하지 않음).")

h("9. 결론")
bullet("'중형 가치 ETF를 사서 들고 있으라'는 2006년 이후 미국·한국 어디서도 SPY·QQQ·코스피를 이기지 못했다. 이긴 시기는 2014년 이전에 몰려 있다.")
bullet("중형 가치 ETF의 특징 = 시장 베타 1 근처 + 약한 소형·가치·수익성·보수투자 노출. 이 노출들의 프리미엄이 2010년대 이후 거의 0이었고, "
       "대형 기술주의 '팩터로 설명 안 되는 초과수익'이 격차를 만들었다. 가치 노출 자체는 중형 안에서도 성과를 더하지 못했다.")
bullet("상대 모멘텀 전환은 테스트에서 무너졌고, 한국판도 학습 구간에서만 좋았다 → 신호 장부 '효과 없음'. 금리 신호는 2001년 이후 5/5 블록 같은 방향이고, 듀레이션·되돌림·순환매·리밸런싱 설명은 기각, 남는 건 '금리 상승의 타격이 기술주에 한 달 늦게 온다' → 보조 신호로 기록.")
bullet("남길 것: '2015년 단절' 진단(미국·한국 같은 시기), 6팩터로 상품의 특징을 뽑아 원인을 나누는 틀, 한국 대형 가치가 중형 가치보다 나았다는 관찰(검증 대상 후보).")

doc.save(OUT)
print(OUT)
try:
    from docx2pdf import convert
    convert(str(OUT), str(OUT.with_suffix(".pdf")))
    print(OUT.with_suffix(".pdf"))
except Exception as e:
    print("PDF 변환 실패:", e)
