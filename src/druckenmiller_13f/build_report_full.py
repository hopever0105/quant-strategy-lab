# -*- coding: utf-8 -*-
"""종합 보고서 생성: 오전 특징 분석(build_report.py) + 오후 검증(실력 확인·따라 사기·규칙·조합·보조 재료) + 수주잔고 결론
→ 드러켄밀러_13F_종합보고서_초안.docx (+pdf). '최종'이면 드러켄밀러_13F_종합보고서.docx
실행: python build_report_full.py [최종]
숫자는 results/ 와 ../order_backlog/results/ 의 csv에서 읽는다. csv로 남지 않은 몇 개(진행상황.md·신호 장부에 기록된 값)는 출처를 주석으로 단다."""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
AN = RES / "analysis"
CH = HERE / "charts"
FINAL = len(sys.argv) > 1 and sys.argv[1] == "최종"
OUT = HERE / ("드러켄밀러_13F_종합보고서.docx" if FINAL else "드러켄밀러_13F_종합보고서_초안.docx")
OB = HERE.parent / "order_backlog"
NAVY = "1F3864"

rd = lambda f, **k: pd.read_csv(AN / f, **k)
pc = lambda x, d=0: "-" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"
pcs = lambda x, d=1: "-" if x is None or pd.isna(x) else f"{x * 100:+.{d}f}%"
f1 = lambda x: "-" if x is None or pd.isna(x) else f"{x:.1f}"
ft = lambda x: "-" if x is None or pd.isna(x) else f"{x:+.1f}"

# ---------------- 데이터 ----------------
P = pd.read_pickle(RES / "panel_ranked.pkl")
E = P[P["src"] == "드러켄밀러"]
S = P[P["src"] == "S&P500"]
stock = (E["leg"] == "주식") & (E["kind"] != "ETF")
buy = stock & E["action"].isin(["신규", "확대"])
fl = pd.read_csv(RES / "filings_parsed.csv")
OV = rd("overview_by_quarter.csv", index_col=0)
HP = rd("holding_periods.csv")
SEC = rd("sector_share.csv", index_col=0)
CMP = rd("features_compare.csv", index_col=0)
PERF = rd("post_buy_performance.csv")
WL = rd("winners_vs_losers.csv")
MC = rd("macro_corr.csv")
LIFT = rd("single_condition_lift.csv")
COMBO = rd("combo_lift.csv")
RP = rd("robust_period.csv")
RCV = rd("robust_conviction.csv")
RNS = rd("robust_new_vs_sell.csv")
RCY = rd("robust_combo_by_year.csv", index_col=0)
TOPW = rd("top_winners.csv")
CB = rd("conviction_buys.csv")
SPL = pd.read_csv(RES / "splits.csv")

n_q = len(OV)
n_buy = int(buy.sum())
big = buy & (E["weight"] >= 0.03)
small = buy & (E["weight"] < 0.01)
new = stock & (E["action"] == "신규")
sell = stock & E["action"].isin(["축소", "청산"])
x12 = E.loc[buy, "f12m_t1_x"].dropna()
inq_new = E.loc[new, "r_inq_x"].dropna()
inq_exit = E.loc[stock & (E["action"] == "청산"), "r_inq_x"].dropna()
prof_b = E.loc[buy, "profitable"].mean()
prof_s = S["profitable"].mean()
insp = E["in_sp500"].astype(bool)
miss_sp = S["r6m"].isna().mean()


def cmpv(code, col="평균백분위"):
    return CMP.loc[code, col] if code in CMP.index else np.nan


def rcv(code, col):
    r = RCV[RCV["특징"] == dict(zip(CMP.index, CMP["특징"])).get(code, code)]
    return r[col].iloc[0] if len(r) else np.nan


# ---------------- 추가 차트 ----------------
def chart_conviction():
    d = RCV.copy().sort_values("t값")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    y = np.arange(len(d))
    ax.barh(y + 0.2, d["확신"] - 50, left=50, height=0.4, color="#1F3864", label="확신 매수(비중 3% 이상)")
    ax.barh(y - 0.2, d["작은"] - 50, left=50, height=0.4, color="#c0a060", label="작은 매수(비중 1% 미만)")
    ax.set_yticks(y)
    ax.set_yticklabels(d["특징"], fontsize=8)
    ax.axvline(50, color="k", lw=0.8)
    ax.set_xlim(20, 85)
    ax.set_xlabel("S&P500 안 평균 백분위")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("확신 매수와 작은 매수의 매수 직전 특징")
    plt.tight_layout()
    plt.savefig(CH / "conviction_vs_small.png", dpi=150)
    plt.close()


def chart_new_sell():
    d = RNS.copy().sort_values("t값")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    y = np.arange(len(d))
    ax.barh(y + 0.2, d["신규"] - 50, left=50, height=0.4, color="#2e7d32", label="신규 매수")
    ax.barh(y - 0.2, d["매도"] - 50, left=50, height=0.4, color="#c62828", label="매도(축소·청산)")
    ax.set_yticks(y)
    ax.set_yticklabels(d["특징"], fontsize=8)
    ax.axvline(50, color="k", lw=0.8)
    ax.set_xlim(20, 85)
    ax.set_xlabel("S&P500 안 평균 백분위 (매수·매도 직전 분기말)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("새로 산 종목과 판 종목의 직전 상태")
    plt.tight_layout()
    plt.savefig(CH / "new_vs_sell.png", dpi=150)
    plt.close()


def chart_overview():
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
    o = OV.copy()
    o.index = pd.to_datetime(o.index)
    axs[0].plot(o.index, o["평가액(백만$)"] / 1000, color="#1F3864")
    axs[0].set_ylabel("13F 보유 평가액(십억 달러)")
    ax2 = axs[0].twinx()
    ax2.bar(o.index, o["보유수"], width=60, alpha=0.3, color="#c0a060")
    ax2.set_ylabel("보유 종목 수")
    axs[0].set_title("분기말 보유 규모와 종목 수")
    q = HP["q"].clip(upper=12)
    axs[1].hist(q, bins=np.arange(0.5, 13.5, 1), color="#1F3864")
    axs[1].set_xlabel("신규 매수부터 청산까지 분기 수 (12 이상은 12)")
    axs[1].set_ylabel("건수")
    axs[1].set_title("보유 기간 분포")
    plt.tight_layout()
    plt.savefig(CH / "overview.png", dpi=150)
    plt.close()


def chart_combo_year():
    d = RCY.dropna(subset=["mean"])
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.bar(d.index.astype(str), d["차이"] * 100, color=["#1F3864" if v > 0 else "#c62828" for v in d["차이"]])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("%p")
    ax.set_title("겹치기 조합(대형·고베타·고변동성) S&P500 종목의 이후 12개월 초과수익 - 나머지 S&P500 (해당 연도 분기말 기준)")
    plt.tight_layout()
    plt.savefig(CH / "combo_by_year.png", dpi=150)
    plt.close()


chart_conviction()
chart_new_sell()
chart_overview()
chart_combo_year()

# ---------------- 문서 ----------------
doc = Document()
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.0)
    s.top_margin = s.bottom_margin = Cm(1.8)
st = doc.styles["Normal"]
st.font.name = "Malgun Gothic"
st.font.size = Pt(9.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Malgun Gothic")
st.paragraph_format.space_after = Pt(3)


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    s_ = OxmlElement("w:shd")
    s_.set(qn("w:val"), "clear")
    s_.set(qn("w:color"), "auto")
    s_.set(qn("w:fill"), color)
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
    b = OxmlElement("w:pBdr")
    bt_ = OxmlElement("w:bottom")
    for k, v in (("val", "single"), ("sz", "6"), ("space", "1"), ("color", NAVY)):
        bt_.set(qn(f"w:{k}"), v)
    b.append(bt_)
    pPr.append(b)


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
            c = t.cell(i, j)
            c.text = ""
            p = c.paragraphs[0]
            r = p.add_run(str(v))
            r.font.size = Pt(size)
            p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            if i == 0:
                r.bold = True
                r.font.color.rgb = RGBColor(255, 255, 255)
                shade(c, NAVY)
            elif hl and i in hl:
                shade(c, "DEEAF6")
    if widths:
        for rw in t.rows:
            for j, w_ in enumerate(widths):
                rw.cells[j].width = Cm(w_)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def pic(name, w=16.5):
    doc.add_picture(str(CH / name), width=Cm(w))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


# ---------------- 오후 검증·수주잔고 데이터 ----------------
def rr(p, **k):
    return pd.read_csv(RES / p, **k)


def row(df, **kw):
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k] == v
    return df[m].iloc[0]


def rowc(df, col, text, **kw):
    """col에 text가 들어간 행 중 kw 조건 맞는 첫 행"""
    d = df[df[col].astype(str).str.contains(text, regex=False)]
    return row(d, **kw) if kw else d.iloc[0]


SK = rd("skill_matched.csv").set_index("구분")
LG = rr("lookalike/summary_groups.csv")
LG6 = LG[LG["기간"].str.contains("6개월")].set_index("묶음")
GV = rr("scaleup/grow_vs_not.csv").set_index("특징")
FO = rr("follow/results.csv")
IU = rr("ideas/ideas_us.csv")
RR = rr("rule/results.csv")
RRC = rr("rule/random_control.csv").set_index("구간")
AR = rr("adds/rule_results.csv")
ARR = rr("adds/rule_random.csv")
UML = rr("combo/us_ml_ic.csv").set_index("모델")
KML = rr("combo/kr_ml_ic.csv").set_index("모델")
IK = rr("ideas/idea4_kr.csv")
LQ = rr("liquidity/results.csv")
STP = rr("stops/results.csv")
OTH = rr("others/results.csv")
THM = rr("themes/results.csv")
BU = rr("backlog/results_us.csv")
A1 = rr("backlog/deep_A1_event.csv")
A2 = rr("backlog/deep_A2_sales.csv")
A3 = rr("backlog/deep_A3_sector.csv")
BM = rr("backlog/deep_B_money.csv")
CW = rr("backlog/deep_C_when.csv")
DD = rr("backlog/deep_druck.csv")
KIC = rr("backlog/kr_orders_ic.csv")
KOR = rr("backlog/kr_orders_results.csv")
OBM = pd.read_csv(OB / "results" / "main_results.csv")
OBR = pd.read_csv(OB / "results" / "robustness.csv")
OKR = pd.read_csv(OB / "results" / "kr_random.csv").set_index("구간")
OUR = pd.read_csv(OB / "results" / "us_random.csv").set_index("구간")
TC = rr("themes/druck_conviction_themes.csv")

us_ic = row(CW, 신호="기본: 증가율 − 매출 증가율")
kr_ic = KIC.iloc[0]
kr_ic_x = KIC.iloc[1]
druck_top30 = (DD["p_gap"] >= 70).mean()
okr = lambda name, seg: row(OBM, 시장="kr", 전략=name, 구간=seg)
ous = lambda name, seg: row(OBM, 시장="us", 전략=name, 구간=seg)
kr_ls_min_t = OBR[(OBR["시장"] == "kr") & (OBR["버전"] == "B 롱숏")]["알파_t"].min()


def chart_backlog_event():
    d = A1.copy()
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.bar(d["기간"], d["차이_분기평균"] * 100, color=["#9aa7bd", "#1F3864", "#1F3864", "#1F3864"])
    for i, (v, t) in enumerate(zip(d["차이_분기평균"] * 100, d["t"])):
        ax.text(i, v + 0.05, f"{v:+.1f}%p\n(t {t:.1f})", ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("상위 30% - 하위 30% (%p)")
    ax.set_ylim(0, max(d["차이_분기평균"] * 100) * 1.45)
    ax.set_title("미국: 신호가 담긴 공시일 앞뒤의 SPY 초과수익 차이")
    plt.tight_layout()
    plt.savefig(CH / "backlog_event.png", dpi=150)
    plt.close()


def chart_backlog_sector():
    d = A3.dropna(subset=["IC"]).sort_values("IC")
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.barh(d["업종"], d["IC"], color=["#1F3864" if t >= 2 else ("#c62828" if t <= -1.5 else "#9aa7bd") for t in d["t"]])
    for i, (v, t) in enumerate(zip(d["IC"], d["t"])):
        ax.text(v + (0.004 if v >= 0 else -0.004), i, f"t {t:.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("순위상관 IC (다음 분기 수익률)")
    ax.set_title("미국: 업종별 신호 효과 (진한 색 = t 2 이상)")
    plt.tight_layout()
    plt.savefig(CH / "backlog_sector.png", dpi=150)
    plt.close()


def chart_kr_orders():
    segs = [("학습", "학습 2016~2019"), ("검증", "검증 2020~2022"), ("테스트", "테스트 2023~ (2025 제외)")]
    grp = [("상위 30% 2분기", "상위 30%", "#1F3864"), ("수주 업종 신호 있는 종목 전부", "업종 전부", "#9aa7bd"),
           ("하위 30% 2분기", "하위 30%", "#c62828"), ("코스피", "코스피", "#c0a060")]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    x = np.arange(len(segs))
    for j, (key, lab, col) in enumerate(grp):
        vals = []
        for seg, _ in segs:
            d = KOR[KOR["전략"].str.startswith(key) & KOR["전략"].str.contains("2025 제외") & (KOR["구간"] == seg)]
            vals.append(d["CAGR"].iloc[0] * 100)
        ax.bar(x + (j - 1.5) * 0.2, vals, width=0.2, color=col, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([s[1] for s in segs])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("연 수익률(%)")
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    ax.set_title("한국 수주 업종: 신호 순위별 연 수익률 (분기 리밸런싱, 2분기 보유)")
    plt.tight_layout()
    plt.savefig(CH / "kr_orders_segments.png", dpi=150)
    plt.close()


chart_backlog_event()
chart_backlog_sector()
chart_kr_orders()


def pic2(path, w=16.5):
    doc.add_picture(str(path), width=Cm(w))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


# 표지
para("스탠리 드러켄밀러 13F 분석 — 종합", bold=True, size=18, color=NAVY, after=2)
para("그가 산 종목의 공통점에서 출발해, 숫자로 남는 신호(수주잔고)를 찾기까지 — 2011년 4분기~2026년 2분기", size=11, after=2)
para("Duquesne Family Office 13F 보고서 62건 · 비교군: 같은 시점 S&P500 구성종목 · 한국 확장: 코스피 상위 200 · 실제 매매 없음(분석용)",
     size=8.5, color="666666", after=8)

# 1. 요약
h("1. 요약")
pr = lambda c: cmpv(c)
h3("이 보고서의 흐름")
para("① 그가 산 종목은 어떤 종목이었나(4장) → ② 그 선택이 실력인가, 그냥 '그런 종목군' 효과인가(5장) → "
     "③ 따라 사거나 규칙으로 옮기면 돈이 되나(6장) → ④ 안 된다면 그가 보는 것 중 숫자로 남는 재료는 무엇인가(7장) → "
     "⑤ 살아남은 하나, 수주잔고(8장).", size=9)
_c3 = SK.loc["확신 매수(3%+)"]
_l3 = LG6.loc["확신(3%+)"]
_ls = LG6.loc["작은(1%-)"]
bullet(f"무엇을 샀나: 변동성·베타가 높은 고성장 종목(60일 변동성 백분위 {f1(pr('vol60'))}, 베타 {f1(pr('beta'))}, 매출 성장 {f1(pr('rev_g_ttm'))}). "
       f"이익률·밸류에이션은 S&P500보다 낮다. 보유는 짧다(중앙값 {HP['q'].median():.0f}분기).")
bullet(f"실력은 '크게 산 종목'에만 보인다. 비중 3% 이상 확신 매수는 변동성·추세·시총이 같은 짝보다 이후 12개월 중앙 {pcs(_c3['중앙차이'])}"
       f"(분기 t {_c3['분기t값']:.1f}), 특징이 가장 닮은 같은 업종 5종목보다 6개월 뒤 {pc(_l3['나은비율'])}가 나았다. "
       f"1% 미만 작은 매수는 차이가 없다({pc(_ls['나은비율'])}). 이 차이를 설명하는 공개 숫자는 없었다 — 그의 성과는 "
       "앞을 보는 판단, 하락장에서 크게 거는 결단, 맞으면 키우고 틀리면 바로 자르는 운용에서 나왔다(5-4).")
_v1t = row(RR, 전략="V1 확신 프로필(1분기)", 구간="테스트")
bullet(f"그러나 규칙으로 옮기면 깨진다. 확신 매수 특징으로 만든 규칙은 학습·검증에서 무작위 대조 백분위 {RRC.loc['학습', '무작위대비백분위']:.0f}·"
       f"{RRC.loc['검증', '무작위대비백분위']:.0f}였지만 테스트(2022~)에서 {RRC.loc['테스트', '무작위대비백분위']:.0f}로 떨어졌고, "
       f"조합 750개·머신러닝(미국 IC {UML.loc['Ridge', 'IC평균']:+.3f})도 표본 밖에서 시장을 못 이겼다. 공시일 따라 사기는 2022년 이후에만 알파가 있었다.")
bullet("그래서 '그가 보는데 숫자로 남는 것'을 찾았다. 유동성 점수·손절·다른 펀드 동시 매수·공시 단어 테마는 무효 또는 약했고, "
       f"하나만 살아남았다: 수주잔고 증가율 − 매출 증가율. 미국 IC {us_ic['IC']:.3f}(t {us_ic['t']:.2f}), "
       f"한국 수주 업종 IC {kr_ic['IC']:.3f}(t {kr_ic['t']:.2f}, 2025년 빼도 t {kr_ic_x['t']:.2f}). 그의 확신 매수 중 {pc(druck_top30)}가 이 신호 상위 30%였다.")
_ka = okr("A 상위 30% 매수", "전체")
_kk = okr("코스피", "전체")
_kax = okr("A 상위 30% 매수 (2025 제외)", "전체")
_ua = ous("A 상위 30% 매수", "전체")
_us = ous("SPY", "전체")
bullet(f"일별 백테스트: 한국 상위 30% 매수 연 {pc(_ka['CAGR'], 1)}(코스피 {pc(_kk['CAGR'], 1)}), 코스피 대비 알파 t {_ka['알파_t']:.2f}"
       f"(2025년 제외 {_kax['알파_t']:.2f}), 상위 비율·보유기간을 바꾼 18개 조합 모두 롱숏 알파 t {kr_ls_min_t:.2f} 이상. "
       f"미국은 매수만으로는 SPY와 비슷(연 {pc(_ua['CAGR'], 1)} vs {pc(_us['CAGR'], 1)}) → 한국이 주력, 미국은 보조 신호.")
h3("특징 분석 요약 (4장)")
bullet(f"매수 {n_buy:,}건(주식, ETF 제외)을 매수 직전 분기말 상태로 같은 분기 S&P500 종목들과 비교했다. "
       f"가장 뚜렷한 공통점은 '많이 움직이는 종목'이다. 60일 변동성 평균 백분위 {f1(pr('vol60'))}, 베타 {f1(pr('beta'))} (50이 중간).")
bullet(f"재무로는 매출이 빨리 늘지만(매출 성장 백분위 {f1(pr('rev_g_ttm'))}) 아직 이익률이 낮고 비싼 성장주를 샀다 "
       f"(영업이익률 {f1(pr('op_margin'))}, ROE {f1(pr('roe'))}, 이익수익률 {f1(pr('ey'))}, R&D/매출 {f1(pr('rnd_sales'))}). "
       f"흑자 종목 비율은 {pc(prof_b)}로 S&P500({pc(prof_s)})보다 낮다.")
bullet(f"장기 보유자가 아니다. 신규 매수부터 청산까지 중앙값 {HP['q'].median():.0f}분기, "
       f"{pc((HP['q'] <= 1).mean())}가 한 분기 만에 사라졌다. 분기마다 포트폴리오의 약 {pc(OV['매수비중'].mean())}를 새로 산다.")
bullet(f"두 종류의 매수가 섞여 있다. 비중 3% 이상의 '확신 매수'는 대형·저변동·52주 고점 근처·이익률 높은 주도주이고 "
       f"(시총 백분위 {f1(rcv('mcap', '확신'))}), 1% 미만의 '작은 매수'는 소형·고변동·적자 성장주다(시총 {f1(rcv('mcap', '작은'))}).")
bullet(f"새로 사는 종목은 직전 1년 동안 덜 오른 상태(12개월 수익률 백분위 {f1(RNS.set_index('특징').loc['12개월 수익률', '신규'])})였고, "
       f"파는 종목은 이미 많이 오른 상태({f1(RNS.set_index('특징').loc['12개월 수익률', '매도'])})였다. "
       f"다만 산 분기 안에서는 평균 {pcs(inq_new.mean())} SPY를 이겼다 — 추세가 막 돌아서는 구간에서 들어간다.")
bullet(f"매수 후 12개월 SPY 초과수익은 평균 {pcs(x12.mean())}이지만 중앙값은 {pcs(x12.median())}, 승률 {pc((x12 > 0).mean())}이다. "
       f"상위 5% 종목을 빼면 평균이 {pcs(x12[x12 <= x12.quantile(.95)].mean())}로, 소수의 큰 성공이 성과를 만든다.")

# 2. 분석 방법
h("2. 분석 방법")
h3("2-1. 무엇을 '매수'로 보았나")
table([["구분", "정의 (같은 종목·같은 분기 비교, 주식 수 기준)"],
       ["신규", "직전 분기 0주 → 이번 분기 보유"],
       ["확대", "주식 수 10% 넘게 증가 (액면분할 보정 후)"],
       ["유지", "±10% 이내"],
       ["축소", "10% 넘게 감소"],
       ["청산", "이번 분기 0주"],
       ["콜·풋옵션", "주식과 따로 집계 (이번 특징 분석에서는 제외)"]], widths=[3, 13.5])
para("13F는 분기말 하루의 보유 명단이다. 분기 안에서 사고판 것은 보이지 않고, 정확한 매수일과 가격도 알 수 없다. "
     "그래서 '매수 직전 상태'는 매수 분기의 직전 분기말(t0) 값으로 본다. 이 값은 그가 매수를 결정하기 전에 누구나 알 수 있던 정보다.", size=9)
h3("2-2. 특징과 비교 방법")
bullet("주가: 1·3·6·12개월 수익률, 52주 고점·저점 대비, 200일선 대비, 50일/200일선, 60일 변동성, 베타(1년), 잔차 변동성, 1년 최대낙폭, 거래대금.")
bullet("재무: SEC XBRL에서 t0 이전에 공시된 숫자만 사용. 최근 4분기 매출·영업이익·순이익 성장, 최근 분기 전년 동기 대비 성장과 그 가속, 이익률, ROE·ROA, 부채, R&D/매출, FCF.")
bullet("가치: 시가총액(공시 주식 수 × 분할 전 실제 종가), PSR, 이익수익률(1/PER), 장부가/시총(1/PBR), FCF 수익률.")
bullet("비교: 같은 분기 S&P500 구성종목(그 시점 기준, 분기당 약 500개) 안에서의 백분위. 50이면 S&P500 중간, 80이면 상위 20%.")
bullet("t값: 분기마다 매수 종목의 평균 백분위를 구하고, 분기 평균들이 50과 다른지 검정. 같은 분기 종목끼리 서로 닮는 문제를 줄인다.")

# 3. 데이터
h("3. 데이터")
n_am = int(fl["form"].str.endswith("/A").sum())
table([["항목", "내용"],
       ["13F", f"Duquesne Family Office(SEC CIK 1536411), {fl['period'].min()}~{fl['period'].max()}, {len(fl)}건(정정 {n_am}건 반영). 보고서에 적힌 종목 수와 파싱 결과 {len(fl)}건 모두 일치"],
       ["종목 식별", "CUSIP 697개 → 티커(OpenFIGI)·SEC 회사번호·업종코드(SIC). 이름 오매칭은 직접 수정"],
       ["주가", "야후 일별 시세(수정주가), 분할 이력으로 분할 전 종가 복원"],
       ["재무", "SEC XBRL companyfacts, 공시일(filed) 기준으로 그 시점에 알 수 있던 값만"],
       ["비교군", "S&P500 역대 구성종목(편입·편출일 포함), 분기당 약 500개"],
       ["분할 보정", f"{len(SPL)}건: " + ", ".join(f"{r['name'].split()[0]}({str(r['period'])[:4]})" for _, r in SPL.iterrows())],
       ["자료 보정", "2013년 2분기 보고서(원본·정정 모두)는 주식 수 칸에 평가액을 적은 기재 오류 → 앞뒤 분기 13F 가격과 주가 변화로 주식 수 추정"],
       ["업종", "SEC SIC 코드를 11개 섹터로 묶은 근사치(구글·메타는 IT로 분류됨)"]], widths=[3, 13.5])
cov = lambda c: pc(E.loc[buy, c].notna().mean())
para(f"커버리지: 매수 {n_buy:,}건 중 주가 특징 {cov('r6m')}, 재무 {cov('rev_ttm')}, 시가총액 {cov('mcap')}, 12개월 후 수익률 {cov('f12m_t1')}. "
     f"빠진 것은 대부분 이후 상장폐지·인수돼 야후 시세가 없는 종목이다.", size=9)

# 4. 결과
h("4. 결과")
h3("4-1. 포트폴리오 모습")
pic("overview.png", 16.5)
table([["항목", "값"],
       ["분기 평균 보유 종목", f"{OV['보유수'].mean():.0f}개 (최소 {OV['보유수'].min():.0f}, 최대 {OV['보유수'].max():.0f})"],
       ["상위 10종목 비중", f"평균 {pc(OV['상위10비중'].mean())}"],
       ["최대 단일 종목 비중", f"평균 {pc(OV['최대비중'].mean())}"],
       ["분기 평균 신규 / 확대 / 축소 / 청산", f"{OV['신규'].mean():.1f} / {OV['확대'].mean():.1f} / {OV['축소'].mean():.1f} / {OV['청산'].mean():.1f}"],
       ["보유 기간(신규→청산)", f"중앙값 {HP['q'].median():.0f}분기, 4분기 이상 {pc((HP['q'] >= 4).mean())}, 8분기 이상 {pc((HP['q'] >= 8).mean())}"],
       ["매수 중 S&P500 밖 종목", f"{pc(1 - insp[buy].mean())} (그 시가총액 중앙값 {E.loc[buy & ~insp, 'mcap'].median() / 1e9:.1f}십억 달러)"]],
      widths=[6, 10.5])
para("상위 10종목에 65% 안팎을 몰아 두고, 나머지 수십 종목은 작게 담았다가 빠르게 교체한다. 집중과 회전이 동시에 일어난다.", size=9)

h3("4-2. 업종")
sec_rows = [["업종", "매수 건수 비중", "보유 금액 비중", "S&P500 종목 비중"]]
for s_, r in SEC.iterrows():
    if s_ in ("기타",):
        continue
    sec_rows.append([s_, pc(r["매수건수비중"]), pc(r["보유금액비중"]), pc(r["S&P500종목비중"])])
table(sec_rows, widths=[4, 4, 4, 4.5])
pic("sector_by_year.png", 16)
para("IT·헬스케어·커뮤니케이션은 S&P500보다 훨씬 많이, 필수소비재·유틸리티·부동산은 거의 사지 않았다. "
     "S&P500의 '미분류'는 상장폐지돼 업종코드를 찾지 못한 종목이다.", size=9)

h3("4-3. 매수 직전 특징 (핵심)")
pic("buy_feature_percentiles.png", 13.5)
rows = [["특징", "매수 평균 백분위", "t값", "상위 20%에 든 비율", "신규", "확대", "매도"]]
for c, r in CMP.sort_values("t값", key=lambda s: -s.abs()).iterrows():
    rows.append([r["특징"], f1(r["평균백분위"]), ft(r["t값"]), pc(r["상위20비율"] / 100), f1(r["신규"]), f1(r["확대"]), f1(r["매도"])])
table(rows, widths=[3.6, 2.4, 1.5, 2.4, 1.6, 1.6, 1.6], size=7.5)
para("읽는 법: 백분위 50이면 S&P500과 같은 수준. 상위 20%에 든 비율의 기준은 20%다. |t|가 3 이상이면 우연으로 보기 어렵다.", size=8.5, italic=True)
pic("buy_vs_sp500_dist.png", 16.5)
h3("해석")
bullet(f"움직임: 변동성·베타·잔차 변동성이 모두 상위권이다. 매수의 {pc(CMP.loc['vol60', '상위20비율'] / 100)}가 S&P500 변동성 상위 20%에 들었다.")
bullet(f"추세: 12개월 수익률({f1(pr('r12m'))})·200일선 대비({f1(pr('vs200'))})·52주 저점 대비({f1(pr('dist52l'))})가 중간보다 높다. "
       f"바닥에서 많이 올라온 종목이지만, 52주 고점 대비는 {f1(pr('dist52h'))}로 고점에 붙어 있지는 않다.")
bullet(f"성장과 가격: 매출 성장은 높고 PSR({f1(pr('ps'))})은 높으며 이익수익률({f1(pr('ey'))})·장부가/시총({f1(pr('bm'))})은 낮다. 싼 주식을 고르지 않았다.")
bullet(f"수익성: 영업이익률·ROE·ROA가 모두 하위권이다. R&D 비중이 높은 기술·바이오 기업이 많아서다. "
       f"이익 성장(TTM)과 매출 가속은 S&P500과 차이가 없다(t {ft(CMP.loc['op_g_ttm', 't값'])}, {ft(CMP.loc['rev_accel', 't값'])}).")
bullet("매도 종목도 비슷한 특징을 가진다. 즉 이 특징들은 '그가 사는 순간'보다 '그가 노는 종목군'을 보여준다. 사고파는 시점의 차이는 4-4에서 본다.")

h3("4-4. 언제 사고 언제 파나 — 신규 매수 vs 매도")
pic("new_vs_sell.png", 13.5)
rns = RNS.set_index("특징")
bullet(f"새로 산 종목은 직전 12개월 수익률 백분위 {f1(rns.loc['12개월 수익률', '신규'])}, 판 종목은 {f1(rns.loc['12개월 수익률', '매도'])} (t {ft(rns.loc['12개월 수익률', 't값'])}). "
       f"200일선 대비도 {f1(rns.loc['200일선 대비', '신규'])} vs {f1(rns.loc['200일선 대비', '매도'])}. 오래 오른 것을 팔고, 덜 오른 것을 산다.")
bullet(f"산 분기 안의 SPY 초과수익은 평균 {pcs(inq_new.mean())}(중앙 {pcs(inq_new.median())}), 청산한 분기는 {pcs(inq_exit.mean())}(중앙 {pcs(inq_exit.median())}). "
       "덜 오른 종목이 분기 안에 강하게 반등하는 구간에 들어가고, 힘이 빠지면 나온다.")
bullet("주의: 13F는 분기말 명단이라, 분기 안에 사서 떨어지자 바로 판 종목은 명단에 안 남는다. 그래서 '산 분기 수익률'은 실제보다 좋게 보인다(분기 내 생존 편향).")

h3("4-5. 확신 매수 vs 작은 매수")
pic("conviction_vs_small.png", 13.5)
rows = [["특징", "확신(비중 3% 이상)", "작은(1% 미만)", "t값"]]
for _, r in RCV.iterrows():
    rows.append([r["특징"], f1(r["확신"]), f1(r["작은"]), ft(r["t값"])])
table(rows, widths=[4.5, 3.5, 3.5, 2.5], size=7.5)
xb = E.loc[big, "f12m_t1_x"].dropna()
xs = E.loc[small, "f12m_t1_x"].dropna()
bullet(f"확신 매수 {int(big.sum())}건: 대형주, 변동성 낮은 편, 52주 고점 근처, 이익률·ROE가 높은 주도주. 12개월 SPY 초과 중앙값 {pcs(xb.median())}, 승률 {pc((xb > 0).mean())}.")
bullet(f"작은 매수 {int(small.sum())}건: 소형, 고변동, 적자 성장주. 평균 {pcs(xs.mean())}이지만 중앙값 {pcs(xs.median())}, 승률 {pc((xs > 0).mean())} — 몇 개가 크게 터지길 기대하는 옵션 같은 베팅.")
cb = CB.sort_values("weight", ascending=False).drop_duplicates("name").head(12)
para("비중이 컸던 확신 매수 예: " + ", ".join(f"{r['name']}({str(r['period'])[:4]}, {r['weight'] * 100:.0f}%)" for _, r in cb.iterrows()), size=8.5)

h3("4-6. 매수 후 성과 (참고)")
rows = [["기간", "기준일", "표본", "평균", "중앙값", "승률", "t값(분기 평균)"]]
for _, r in PERF.iterrows():
    rows.append([r["기간"], r["기준"], int(r["표본"]), pcs(r["평균"]), pcs(r["중앙"]), pc(r["승률"]), ft(r.get("분기평균t"))])
table(rows, widths=[1.5, 3, 1.5, 2, 2, 1.6, 2.5], size=7.5)
para("분기말 = 13F 기준일(그의 보유 시점에 가까움), 공시일 = 13F가 공개된 날(따라 사는 사람이 알 수 있는 날, 약 45일 뒤). "
     "평균은 플러스지만 중앙값·승률은 0 근처다. 상위 10: " +
     ", ".join(f"{r['name'].split()[0]}({str(r['period'])[:4]}, {r['f12m_t1_x'] * 100:+.0f}%)" for _, r in TOPW.head(10).iterrows()) + ".", size=8.5)
if len(WL):
    bullet("승자(12개월 초과 상위 1/3)와 패자(하위 1/3)의 매수 직전 차이: " +
           "; ".join(f"{r['특징']} {r['승자백분위']:.0f} vs {r['패자백분위']:.0f}(t {r['t값']:+.1f})" for _, r in WL[WL['t값'].abs() >= 2].iterrows()) +
           ". 베타가 높을수록 결과가 좋았다 — 다만 2019~2025 상승장 영향이 크다.")

h3("4-7. 시장 환경과 업종 매수")
rows = [["대상", "직전 3개월 환경", "순위상관", "p값", "분기 수"]]
for _, r in MC.iterrows():
    rows.append([r["대상"].replace("n_hold 순매수", "보유 종목 수"), r["환경"], f"{r['순위상관']:+.2f}", f"{r['p값']:.2f}", int(r["분기수"])])
table(rows, widths=[3.5, 5, 2, 2, 2], size=7.5)
bullet("통계적으로 뚜렷한 것은 두 가지다. 직전 3개월 금리가 오르면 금융주를 덜 샀고(-0.40), VIX가 높을 때 보유 종목 수가 많았다(+0.31, 불안할 때 작게 여러 개로 나눔).")
bullet("유가·구리·달러와 에너지·소재 매수의 관계는 약했다. 매크로 판단이 업종 선택에 쓰였더라도 '직전 3개월 가격 변화'로는 드러나지 않는다.")

h3("4-8. 특징 겹치기 — S&P500 중 그가 산 종목을 얼마나 잘 골라내나")
base_rate = S["druck_bought"].mean()
rows = [["조건", "S&P500 중 해당", "배율", "그가 산 S&P 종목 중 포함"]]
for _, r in LIFT.iterrows():
    rows.append([r["조건"], pc(r["해당비율"]), f"{r['배율']:.2f}", pc(r["포함매수비율"])])
table(rows, widths=[6, 3, 2, 4], size=7.5)
rows = [["조합", "배율", "포함", "이후 12개월 SPY 초과", "나머지 S&P500"]]
for _, r in COMBO.head(6).iterrows():
    rows.append([r["조합"], f"{r['배율']:.2f}", pc(r["포함매수비율"]), pcs(r["이후12개월초과"]), pcs(r["나머지12개월초과"])])
table(rows, widths=[7.5, 1.5, 1.5, 3, 3], size=7.5)
para(f"배율 = 조건에 맞는 S&P500 종목이 그 분기 그의 매수 종목일 확률 ÷ 전체 확률({pc(base_rate, 1)}). "
     "그의 S&P500 매수는 '대형 + 고베타 + 고변동 + 비싼 성장주'에 몰려 있다.", size=8.5)

# ================= 5. 실력 =================
h("5. 실력인가, 그런 종목군 효과인가")
para("4장의 특징은 '그가 노는 종목군'을 보여줄 뿐이다. 같은 종목군 안에서 그가 고른 종목이 안 고른 종목보다 나았는지를 두 가지 방법으로 확인했다.", size=9)
h3("5-1. 짝 비교 — 변동성·추세·시총이 같은 종목과")
para("매수 종목마다 같은 분기 S&P500에서 60일 변동성 5분위 × 12개월 수익률 3분위 × 시총 3분위가 같은 종목들을 '짝'으로 잡고, 이후 12개월 SPY 초과수익 차이를 쟀다.", size=9)
rows = [["구분", "표본", "평균 차이", "중앙 차이", "짝보다 나은 비율", "분기 t값"]]
for k, r in SK.iterrows():
    rows.append([k, int(r["표본"]), pcs(r["평균차이"]), pcs(r["중앙차이"]), pc(r["짝보다나은비율"]), ft(r["분기t값"])])
table(rows, widths=[4, 1.6, 2.3, 2.3, 3, 2], size=7.5, hl=[4])
para("평균은 모두 플러스지만 소수 대박 영향이 크다. 중앙값까지 플러스이고 t가 3을 넘는 것은 확대와 확신 매수뿐이다. 작은 매수·신규·유지는 중앙 0 근처.", size=9)
h3("5-2. 닮은꼴 비교 — 같은 업종에서 8개 특징이 가장 닮은 5종목과")
para("매수 건마다 같은 분기·같은 업종 S&P500에서 12개월·3개월 수익률, 변동성, 베타, 시총, 매출 성장, 영업이익률, PSR 백분위가 가장 가까운 5종목"
     "(그가 그 분기에 보유했거나 산 종목 제외)을 골라 이후 수익률을 비교했다.", size=9)
pic("lookalike_outcome.png", 14)
rows = [["묶음", "건수", "닮은꼴보다 나은 비율", "차이 중앙", "상·하위 5% 자른 평균", "분기 t"]]
for k in ["매수 전체", "신규", "확대", "작은(1%-)", "중간", "확신(3%+)"]:
    r = LG6.loc[k]
    rows.append([k, int(r["건수"]), pc(r["나은비율"]), pcs(r["차이중앙"]), pcs(r["상하위5%자른평균"]), ft(r["분기t"])])
table(rows, widths=[3, 1.6, 3.2, 2.2, 3.4, 1.8], size=7.5, hl=[6])
para("이후 6개월 SPY 초과수익 기준. 확신 매수만 뚜렷하게 닮은꼴을 이긴다. 다만 닮은꼴과 다른 점(고유 변동성·R&D·최근 분기 매출 성장·200일선 위·거래대금이 높고 "
     "ROE·이익률·싼 정도는 낮음) 중 어느 것도 '닮은꼴을 이긴 매수'를 가르지 못했다(순위상관 절댓값 0.08 미만). "
     # 아래 24%·19%·63%는 lookalike.py 실행 출력(진행상황.md 기록)
     "그는 닮은꼴보다 극단적인 종목을 골랐고(6개 중 1등 24%, 꼴찌 19%), 그 분기에 안 고른 닮은꼴의 63%는 나중에 그가 샀다. "
     "즉 그의 추가 판단은 공개된 숫자로는 보이지 않는다.", size=9)
h3("5-3. 작게 시작해서 키우나")
_g1 = GV.loc["다음 분기 주가(SPY 초과)"]
# 605·3.6%·59%는 scaleup.py 실행 출력(진행상황.md 기록)
para(f"처음 1% 미만으로 시작한 605건 중 나중에 3% 이상으로 키운 것은 3.6%뿐이고, 3% 이상 포지션의 59%는 처음부터 크게 샀다. "
     f"키운 종목은 산 다음 분기 SPY 초과 중앙 {pcs(_g1['키운것_중앙'])}(안 키운 것 {pcs(_g1['안키운것_중앙'])}, p {_g1['p값']:.3f}) — "
     "산 뒤 오른 것을 키운 것이지, 작은 매수가 큰 매수의 예고는 아니었다. '작은 매수 먼저 따라 타기'는 무효.", size=9)

h3("5-4. 숫자 밖의 판단력 — 그의 성과를 만든 '눈'")
para("5-1·5-2에서 확신 매수는 닮은꼴을 이겼지만, 그 차이를 설명하는 공개 숫자는 없었다. 남는 설명은 정성적인 판단이다. "
     "13F 기록과 그가 공개적으로 밝힌 원칙을 맞춰 보면 그 판단은 네 가지로 정리된다.", size=9)
CP = rr("lookalike/conviction_picks_table.csv").dropna(subset=["f6m_t1_x_차이"])
POS = rr("positions.csv")
POS = POS[POS["leg"] == "주식"]


PERIODS = sorted(rr("positions.csv")["period"].unique())
TK = rr("lookalike/conviction_picks_table.csv")[["종목", "이름"]].drop_duplicates("종목").set_index("종목")["이름"]


def after(tk, period, name=None):
    """확신 매수 다음 분기 비중 변화 → 라벨 (13F 같은 이름 기준)"""
    key = str(name if name else TK.get(tk, tk)).lower().split()[0].strip(".,")
    i = PERIODS.index(period)
    if i + 1 >= len(PERIODS):
        return "다음 분기 미공시"
    nxt = PERIODS[i + 1]
    d = POS[POS["name"].str.lower().str.contains(key, regex=False)]
    w0 = d.loc[d["period"] == period, "weight"].sum()
    w1 = d.loc[d["period"] == nxt, "weight"].sum()
    if w1 == 0:
        return "다음 분기 전량 정리"
    if w1 >= w0 * 1.1:
        return f"다음 분기 키움({w1 * 100:.0f}%)"
    if w1 <= w0 * 0.9:
        return f"다음 분기 축소({w1 * 100:.0f}%)"
    return f"다음 분기 유지({w1 * 100:.0f}%)"


def wpath(key, start, end):
    d = POS[POS["name"].str.lower().str.contains(key) & (POS["period"] >= start) & (POS["period"] <= end)]
    return " → ".join(f"{p[:4]}.{int(p[5:7]) // 3}Q {w * 100:.0f}%" for p, w in zip(d["period"], d["weight"]))


bullet("① 지금이 아니라 앞을 본다. 그는 2015년 Lost Tree Club 강연에서 스승에게 '18개월 뒤의 상황을 그리고, 가격은 지금이 아니라 그쪽으로 간다'고 배웠다고 말했다. "
       "13F에서도 새로 산 종목은 직전 1년 동안 덜 오른 상태였고(4-4), 크게 산 종목은 닮은꼴보다 최근 분기 매출 성장·R&D 비중이 높았다 — 이미 보이는 이익보다 앞으로의 매출을 샀다.")
_q = CP[CP["분기"] == "2022-12-31"].sort_values("f6m_t1_x_차이", ascending=False).head(3)
bullet("② 시장 전체는 유동성으로 판단하고, 하락장에서 크게 건다. 같은 강연에서 그는 '이익이 아니라 연준과 유동성이 전체 시장을 움직인다', "
       "큰돈의 대부분은 주식 하락장에서 벌었다고 했다. 2022년 하락장 막바지인 4분기에 그는 " +
       ", ".join(f"{r['종목']}({r['구분']}, 비중 {r['비중'] * 100:.0f}%)" for _, r in _q.iterrows()) +
       "를 샀고, 이후 6개월 SPY 초과수익은 " + ", ".join(f"{r['f6m_t1_x_고른것'] * 100:+.0f}%" for _, r in _q.iterrows()) +
       "(같은 업종 닮은꼴 중앙 " + ", ".join(f"{r['f6m_t1_x_닮은꼴중앙'] * 100:+.0f}%" for _, r in _q.iterrows()) +
       "). 6-1에서 따라 사기 알파가 2022년 이후에만 나온 이유가 이 판단이다.")
bullet(f"③ 확신이 서면 처음부터 크게, 맞으면 더 크게. 3% 이상 포지션의 59%는 처음부터 크게 샀다(5-3). 엔비디아는 {wpath('nvidia', '2022-12-31', '2024-06-30')}로 "
       "2022년 4분기 신규 매수 뒤 두 분기 만에 14%까지 키웠다. 그는 젊은 동료의 추천으로 샀고, 한 달 뒤 ChatGPT가 나오자 비중을 크게 늘렸다고 인터뷰에서 밝혔다. "
       "2024년 1분기에 대부분 판 것은 스스로 '큰 실수'라고 했다.")
bullet(f"④ 틀리면 바로 인정한다. 크게 산 뒤 닮은꼴보다 크게 뒤진 종목 — 카바나(2021년 4분기 비중 4%, {after('CVNA', '2021-12-31')}), "
       f"데이터독(2022년 3분기 4%, {after('DDOG', '2022-09-30')}) — 은 오래 붙들지 않았다(아래 표 '그 뒤'). 신규 매수의 55%가 한 분기 안에 사라진다(4-1). "
       "평균 승률은 51%로 동전 던지기와 비슷하지만(4-6), 맞은 것은 키우고 틀린 것은 빨리 잘라 소수의 큰 성공이 성과를 만든다.")
rows = [["확신 매수(비중 3%+)", "분기", "구분", "비중", "6개월 SPY 초과", "닮은꼴 중앙", "그 뒤"]]
best = CP.sort_values("f6m_t1_x_차이", ascending=False).drop_duplicates("종목").head(5)
worst = CP.sort_values("f6m_t1_x_차이").drop_duplicates("종목").head(3)
for tag, d in [("맞힌 것", best), ("틀린 것", worst)]:
    for i, (_, r) in enumerate(d.iterrows()):
        rows.append([f"{tag}: {r['종목']}" if i == 0 else r["종목"], r["분기"][:7], r["구분"], pc(r["비중"], 1),
                     pcs(r["f6m_t1_x_고른것"], 0), pcs(r["f6m_t1_x_닮은꼴중앙"], 0), after(r["종목"], r["분기"], r["이름"])])
table(rows, widths=[3.2, 1.8, 1.2, 1.4, 2.6, 2.4, 3.4], size=7.5)
para("정리: 숫자 규칙으로 옮길 수 있었던 것은 그가 고르는 '종목군'과, 그가 보는 것 중 하나인 '앞으로 들어올 매출'(8장 수주잔고)뿐이다. "
     "언제 시장 전체에 크게 걸지(유동성 판단), 얼마나 크게 살지, 언제 틀렸다고 인정할지는 사람의 판단으로 남았다. "
     "이 보고서의 신호는 그 판단을 대신하는 것이 아니라, 판단할 후보를 좁혀 주는 도구로 쓴다.", size=9)
para("출처(정성 부분): 2015년 1월 Lost Tree Club 강연 녹취(Cove Street Capital 배포본 등 2차 정리본, 문구는 요지), "
     "엔비디아 매수·매도 경위는 CNBC 인터뷰 보도(2024). 13F 비중과 수익률은 이 분석의 데이터.", size=7.5, color="666666")

# ================= 6. 따라 하기·규칙 =================
h("6. 따라 사거나 규칙으로 옮기면")
para("시간 분할(미국): 학습 2012~2018(설정 고르기) / 검증 2019~2021(확인) / 테스트 2022~2026년 2분기(한 번만). 한국: 학습 2015~2019 / 검증 2020~2022 / 테스트 2023~. "
     "비용 편도 미국 0.1%, 한국 0.25%. 알파는 Fama-French 5팩터 + 모멘텀(6팩터) 회귀.", size=9)
h3("6-1. 13F 공시일에 따라 사기")
rows = [["전략(공시일 종가 매수)", "구간", "연 수익률", "샤프", "MDD", "6팩터 알파(연)", "알파 t"]]
for s in ["F1 전체 복제", "F2 확신·확대", "F3 상위 10", "F4 작은 신규"]:
    for seg in ["학습", "검증", "테스트"]:
        r = row(FO, 전략=s, 기준="공시일", 구간=seg)
        rows.append([s if seg == "학습" else "", seg, pc(r["CAGR"], 1), f"{r['샤프']:.2f}", pc(r["MDD"], 0), pcs(r["FF6_알파(연)"]), ft(r["FF6_t"])])
for seg in ["학습", "검증", "테스트"]:
    r = row(RR, 전략="SPY", 구간=seg)
    rows.append(["SPY" if seg == "학습" else "", seg, pc(r["CAGR"], 1), f"{r['샤프']:.2f}", pc(r["MDD"], 0), "-", "-"])
table(rows, widths=[3.6, 1.4, 2, 1.4, 1.6, 2.6, 1.6], size=7)
pic("follow_equity.png", 15)
para("2012~2021년에는 어느 방식도 알파가 없고, 2022년 이후 테스트 구간에서만 t 2 이상이 나온다. 작은 신규만 따라 사면 MDD가 −60%대다. "
     "약 45일 늦게 공개되는 13F를 그대로 따라 사는 것은 안정적인 전략이 아니다.", size=9)
h3("6-2. 확대 종목을 따라 사서 오래 들기")
rows = [["전략", "구간", "연 수익률", "샤프", "알파 t"]]
for s in ["1-A 확대+3%이상 신규 / 보유 2분기", "1-C 확대만 / 보유 4분기", "SPY"]:
    for seg in ["학습", "검증", "테스트"]:
        r = row(IU, 전략=s, 구간=seg)
        rows.append([s if seg == "학습" else "", seg, pc(r["CAGR"], 1), f"{r['샤프']:.2f}", ft(r["FF6_t"])])
table(rows, widths=[6, 1.6, 2.2, 1.6, 1.6], size=7.5)
para("그가 비중을 늘린 종목을 2~4분기 들면 검증·테스트에서 알파 t 1.7~2.7로 가장 나은 따라 사기였다. 그러나 학습 구간(2012~2018)에는 알파가 없고, 평균 4~10종목에 집중된다.", size=9)
h3("6-3. 확신 매수 특징을 규칙으로 — 그 없이 S&P500에 적용")
para("규칙: 시총 상위 30% · 베타 ≥ 1.0 · 12개월 수익 > 0 · 52주 고점 −20% 이내 · 매출 성장 ≥ 5% · 매출총이익률 ≥ 40% → 동일가중 1분기 보유(학습 구간 격자 72개 중 선택).", size=9)
rows = [["전략", "구간", "연 수익률", "샤프", "MDD", "알파 t"]]
for s in ["V1 확신 프로필(1분기)", "C1 대형 고베타만(대조)", "SPY"]:
    for seg in ["학습", "검증", "테스트"]:
        r = row(RR, 전략=s, 구간=seg)
        rows.append([s if seg == "학습" else "", seg, pc(r["CAGR"], 1), f"{r['샤프']:.2f}", pc(r["MDD"], 0), ft(r["FF6_t"])])
table(rows, widths=[5, 1.6, 2.2, 1.6, 1.8, 1.6], size=7.5)
pic("rule_equity.png", 15)
para(f"같은 '대형 고베타' 후보 안에서 같은 개수를 무작위로 1,000번 고른 것과 비교한 백분위: 학습 {RRC.loc['학습', '무작위대비백분위']:.0f} / "
     f"검증 {RRC.loc['검증', '무작위대비백분위']:.0f} / 테스트 {RRC.loc['테스트', '무작위대비백분위']:.0f}. 테스트에서 무작위와 구별되지 않는다. "
     f"2022년에는 {pc(row(pd.read_csv(RES / 'rule' / 'yearly.csv').rename(columns={'Date': 'yr'}), yr=2022)['V1 확신 프로필(1분기)'], 0)}(SPY −18%).", size=9)
h3("6-4. 비중을 늘린 종목의 특징을 규칙으로")
_ar = row(AR, 전략="선택 규칙", 구간="전체")
para(f"늘린 종목은 S&P500보다 12개월 수익률·매출 성장·베타가 높았고, 성공한 확대일수록 매출 성장이 더 높았다(t 3.3). 이를 규칙(12개월 백분위 ≥ 60 · 200일선 위 · "
     f"매출 성장 백분위 ≥ 70 · 베타 ≥ 1, 2분기 보유)으로 만들면 연 {pc(_ar['CAGR'], 1)}로 SPY보다 높고, 모멘텀만 맞춘 후보 안 무작위 대비 백분위 "
     f"{' / '.join(f'{v:.0f}' for v in ARR[ARR['대조'] == '모멘텀만 후보'].set_index('구간').loc[['학습', '검증', '테스트'], '백분위'])}(학습/검증/테스트)이다. "
     f"그러나 6팩터 알파 t는 전체 {_ar['FF6_t']:.2f}로, 이미 알려진 팩터(시장·모멘텀·성장) 노출로 설명된다. 이 규칙은 이후 장에서 '기본 규칙'으로 쓴다.", size=9)
pic("add_rule.png", 15)
h3("6-5. 조합 750개와 머신러닝")
# 순위상관 −0.33/0.08, 0.06/0.15 는 combo_ml.py 실행 출력(진행상황.md 기록)
para("12개월 수익률 상위 M% · N일선 위 · 매출 성장 상위 G% · 베타 ≥ B · 보유 H분기의 750개 조합을 학습 구간 샤프로 줄 세웠다. "
     "학습 순위와 검증·테스트 순위의 상관은 미국 −0.33 / 0.08, 코스피 0.06 / 0.15 — 학습 1등은 표본 밖에서 중간 이하였다.", size=9)
rows = [["모델(워크포워드, 다음 분기 순위 예측)", "미국 IC", "미국 t", "코스피 IC", "코스피 t"]]
for m, lab in [("Ridge", "Ridge(선형)"), ("GB", "Gradient Boosting"), ("가짜GB", "가짜(정답 섞어 학습)")]:
    rows.append([lab, f"{UML.loc[m, 'IC평균']:+.3f}", ft(UML.loc[m, "IC_t"]), f"{KML.loc[m, 'IC평균']:+.3f}", ft(KML.loc[m, "IC_t"])])
table(rows, widths=[6, 2, 2, 2, 2], size=7.5)
para("미국은 정답을 섞어 학습한 가짜 모델보다도 낮다. 코스피 Ridge는 IC 0.048(t 2.0)이지만 예측 상위 10% 포트폴리오는 코스피를 못 이겼다.", size=9)
h3("6-6. 코스피에 그대로 옮기면")
_ik = row(IK, 전략="확신 프로필(코스피)", 구간="전체")
_ik1 = row(IK, 전략="확신 프로필(코스피)", 구간="2015~2020")
para(f"확신 프로필을 코스피 상위 200에 적용하면 2015~2026 연 {pc(_ik['CAGR'], 1)}(코스피 {pc(row(IK, 전략='코스피 지수(가격지수)', 구간='전체')['CAGR'], 1)})지만 "
     f"2015~2020은 연 {pc(_ik1['CAGR'], 1)}, MDD {pc(_ik['MDD'], 0)}이고 성과 대부분이 2025년 한 해(전력기기·조선·반도체)에서 나왔다. 무효.", size=9)

# ================= 7. 보조 재료 =================
h("7. 그가 보는 것 중 숫자로 남는 재료 찾기")
para("6장까지의 결론은 '그의 종목 선택을 공개된 숫자 규칙으로는 재현할 수 없다'이다. 그래서 방향을 바꿔, 그가 투자 판단에 쓴다고 알려진 재료 가운데 "
     "숫자로 측정할 수 있는 것을 하나씩 6장의 기본 규칙이나 시장에 붙여 표본 밖에서 확인했다. 숫자는 모두 미리 정한 값이고 고르지 않았다.", size=9)
_lq = rowc(LQ, "전략", "기본 규칙 + 유동성", 시장="us", 구간="전체")
_lq0 = rowc(LQ, "전략", "기본 규칙", 시장="us", 구간="전체")
_st = row(STP, 시장="us", 손절="L1", 구간="전체")
_st0 = row(STP, 시장="us", 손절="없음", 구간="전체")
_o2 = rowc(OTH, "전략", "O2", 시장="us", 구간="전체")
_t1 = rowc(THM, "전략", "T1", 시장="us", 구간="전체")
rows = [["재료", "방법", "결과(미국 전체 구간)", "판정"],
        ["유동성 점수", "연준 자산·2년 금리·신용 스프레드·달러, 점수별 주식 비중 100/70/40%",
         f"샤프 {_lq['샤프']:.2f} vs 기본 규칙 {_lq0['샤프']:.2f}, 낙폭만 줄고 수익 감소. 그의 종목 수·베타와도 무관", "무효"],
        ["손절", "−15% / 50일선 이탈 / 1분기 시장보다 못하면 정리",
         f"−15% 손절 샤프 {_st['샤프']:.2f} vs 없음 {_st0['샤프']:.2f}, 50일선은 대부분 일찍 털림", "무효"],
        ["다른 펀드 16곳", "Appaloosa·Tiger·Coatue 등이 같은 분기에 새로 산 종목",
         f"3곳 이상: 연 {pc(_o2['CAGR'], 1)}이나 평균 1종목·빈 분기 41%·MDD {pc(_o2['MDD'], 0)}. 그의 확신 매수를 먼저 알려주지 않음(직전 분기 33%)", "약함"],
        ["공시 단어 테마", "10-K·10-Q에서 언급이 1년 새 2배 된 단어(AI·GLP-1 등 26개 미리 정함)",
         f"언급 종목 다음 분기 +2.3%p(t 1.8), 연 {pc(_t1['CAGR'], 1)}. 그의 확신 매수 20%가 뜨는 테마(S&P500 평균 8%)", "보조"],
        ["수주잔고", "수주잔고 증가율 − 매출 증가율, 순위 상위 30%",
         f"IC {us_ic['IC']:.3f}(t {us_ic['t']:.2f}), 다른 특징 통제 후 t 2.9, 전·후반 모두 플러스", "통과 → 8장"]]
table(rows, widths=[2.4, 5, 7, 1.6], size=7, hl=[5])
# 33%·20%·8%·t 2.9 는 others13f.py·themes.py·backlog.py 실행 출력(검증된_신호_장부.md 2장 16·17, 3장)

# ================= 8. 수주잔고 =================
h("8. 수주잔고 — 살아남은 신호")
h3("8-1. 왜 수주잔고인가")
para(f"5·6장에서 성공한 확대 매수일수록 매출 성장이 높았고, 그는 이익보다 매출이 빨리 느는 종목을 샀다. 그렇다면 '앞으로 들어올 매출'을 미리 보여주는 숫자가 있으면 "
     f"그가 보는 것에 더 가까워진다. 기업이 이미 계약했지만 아직 매출로 잡지 않은 금액이 수주잔고다. 실제로 그의 확신 매수 {len(DD)}건(2019~, 신호가 있는 종목) 중 "
     f"{pc(druck_top30)}가 신호 상위 30%였다(무작위면 30%).", size=9)
h3("8-2. 신호 정의")
table([["항목", "내용"],
       ["신호", "수주잔고 전년 대비 증가율 − 매출 증가율. 잔고가 매출보다 빨리 쌓이면 플러스"],
       ["미국 수주잔고", "SEC XBRL 남은 이행의무(RevenueRemainingPerformanceObligation, 2018년 회계기준 변경 후 공시), 없으면 계약부채·이연수익. 분기말에 이미 공시된 값만"],
       ["한국 수주잔고", "DART 사업·반기·분기보고서 원문 '수주상황' 표의 수주잔고 합계. 전년 같은 달과 비교, 공시일(접수일)부터 사용. 코스피 상위 200 중 건설·운송장비·기계·전기전자·IT서비스 61개사"],
       ["매출 증가율", "미국 최근 4분기 합계 전년 대비, 한국 연간(다음 해 4월부터)"],
       ["포트폴리오", "분기말 순위(미국은 업종 안 순위) 상위 30% 동일가중, 2분기 보유. 매개변수는 고르지 않음"]], widths=[3, 13.5], size=8)
h3("8-3. 미국 — 진짜 신호인가")
pic("backlog_event.png", 13)
_a2 = A2.set_index("맞힐것")
bullet(f"공시 직후 한 번에 반영되지 않는다. 공시 뒤 3개월 상위−하위 차이 {pcs(row(A1, 기간='뒤 3개월')['차이_분기평균'])}(t {row(A1, 기간='뒤 3개월')['t']:.1f}) — 따라 할 시간이 있다.")
bullet(f"실제로 매출이 따라온다. 신호는 4분기 뒤 매출 증가율 변화와 순위상관 {_a2.iloc[1]['IC_gap']:.2f}(t {_a2.iloc[1]['t_gap']:.1f}). "
       f"수주잔고 증가율만으로는 {_a2.iloc[1]['IC_증가율만']:+.2f} — 매출 대비 차이가 핵심이다.")
bullet("상·하위 5% 수익을 잘라도 IC 0.041(t 4.0)로 그대로 — 소수 대박이 아니다.")
pic("backlog_sector.png", 13)
bullet("산업재·소재·유틸리티처럼 수주로 일하는 업종에서 강하고, 헬스케어(계약부채 성격이 다름)에서는 반대다.")
rows = [["변형", "IC", "t"]]
for _, r in CW.iterrows():
    rows.append([r["신호"].replace("gap", "신호"), f"{r['IC']:+.3f}", ft(r["t"])])
table(rows, widths=[8, 2, 2], size=7.5, hl=[1])
para("증가율만, 가속(이번 − 직전 분기), 잔고 ÷ 매출 단독은 모두 약하다. '매출보다 빨리 쌓이는가'가 신호다. 다만 상위 30% 안에서는 잔고 ÷ 매출이 높은 절반이 분기 +2.2%p(t 2.8) 더 좋았다.", size=9)
h3("8-4. 한국 — 같은 생각이 통하나")
para(f"한국 재무제표의 계약부채·선수금으로는 효과가 없었다(IC 0.013). 대신 정기보고서 원문의 수주상황 표 4천여 건에서 수주잔고를 직접 읽었다. "
     f"IC {kr_ic['IC']:.3f}(t {kr_ic['t']:.2f}, {kr_ic['분기']:.0f}분기, {pc(kr_ic['플러스분기'])} 분기 플러스), 2025년 제외 {kr_ic_x['IC']:.3f}(t {kr_ic_x['t']:.2f}).", size=9)
pic("kr_orders_segments.png", 15)
para("세 구간 모두 상위 30% > 업종 전부 > 하위 30% 순서다. 업종 전부도 코스피보다 높아(2020년 이후 조선·전력기기 장세) 업종 효과가 섞여 있으므로 상위와 하위의 차이로 봐야 한다.", size=9)
h3("8-5. 일별 백테스트 (order_backlog 전략)")
rows = [["시장 · 전략", "구간", "연 수익률", "변동성", "샤프", "MDD", "알파(연)", "알파 t"]]
for name in ["A 상위 30% 매수", "B 상위 매수 − 하위 공매도", "하위 30% 매수(비교)", "코스피"]:
    for seg in ["학습 2016~2019", "검증 2020~2022", "테스트 2023~", "전체"]:
        r = okr(name, seg)
        rows.append([("한국 " + name) if seg.startswith("학습") else "", seg, pc(r["CAGR"], 1), pc(r["변동성"], 0), f"{r['샤프']:.2f}",
                     pc(r["MDD"], 0), "-" if name == "코스피" else pcs(r["알파(연)"]), "-" if name == "코스피" else ft(r["알파_t"])])
for name in ["A 상위 30% 매수", "B 상위 매수 − 하위 공매도", "SPY"]:
    for seg in ["검증 2019~2021", "테스트 2022~", "전체"]:
        r = ous(name, seg)
        rows.append([("미국 " + name) if seg.startswith("검증") else "", seg, pc(r["CAGR"], 1), pc(r["변동성"], 0), f"{r['샤프']:.2f}",
                     pc(r["MDD"], 0), "-" if name == "SPY" else pcs(r["알파(연)"]), "-" if name == "SPY" else ft(r["알파_t"])])
table(rows, widths=[4.6, 2.6, 1.8, 1.5, 1.3, 1.5, 1.8, 1.4], size=7, hl=[4])
pic2(OB / "charts" / "kr_equity.png", 15)
bullet(f"한국 상위 30% 매수: 연 {pc(_ka['CAGR'], 1)}, 코스피 대비 알파 t {_ka['알파_t']:.2f}. 2025년 빼도 연 {pc(_kax['CAGR'], 1)}, t {_kax['알파_t']:.2f}. "
       f"비용 2배에도 t {okr('A 상위 30% 매수 (비용 2배)', '전체')['알파_t']:.2f}.")
bullet(f"같은 후보 안 무작위 대조 백분위: 학습 {OKR.loc['학습 2016~2019', '백분위']:.0f} / 검증 {OKR.loc['검증 2020~2022', '백분위']:.0f} / 테스트 {OKR.loc['테스트 2023~', '백분위']:.0f}.")
bullet(f"상위 비율 20·30·40% × 보유 1·2·4분기 × 2025 포함·제외 18개 조합에서 롱숏 알파 t 최솟값 {kr_ls_min_t:.2f}. 검증 구간(2020~2022)은 롱숏이 0 근처였다.")
bullet(f"미국은 매수만으로 SPY와 비슷하다(알파 t {_ua['알파_t']:.2f}). 롱숏 알파 t {ous('B 상위 매수 − 하위 공매도', '전체')['알파_t']:.2f}, "
       f"무작위 대조 백분위 {OUR.loc['전체', '백분위']:.0f}. 2019년 이후 대형 기술주 장세에서 동일가중 바구니가 시총가중 SPY를 이기기 어려웠다. 미국은 다른 전략에 붙이는 보조 신호로 쓴다.")

# 9. 강건성
h("9. 강건성 검증")
h3("9-1. 특징 분석 — 기간을 나누면")
rows = [["특징", "2012~2018", "2019~2026", "방향"]]
for _, r in RP.iterrows():
    rows.append([r["특징"], f1(r["전반_2012_18"]), f1(r["후반_2019_26"]), "같음" if r["같은방향"] else "바뀜"])
table(rows, widths=[4.5, 3, 3, 2], size=7.5)
para("시가총액만 방향이 바뀌었다(전반 대형주 위주 → 후반 중소형 늘어남). 변동성·베타·성장·저수익성 성향은 후반에 더 강해졌다.", size=9)
h3("9-2. 특징 분석 — 겹치기 조합이 특정 시기 효과인가")
pic("combo_by_year.png", 16.5)
wins = int((RCY["차이"] > 0).sum())
tot = int(RCY["차이"].notna().sum())
para(f"'시총 상위 30% + 베타 1.2 초과 + 변동성 상위 30%' S&P500 종목은 {tot}개 연도 중 {wins}개 연도에 나머지 S&P500을 이겼다. "
     "다만 큰 차이는 2019~2025년(대형 기술주 장세)에 몰려 있고, 2015·2018·2021년에는 졌다. "
     "이 숫자는 특징을 찾은 같은 기간에서 잰 것이라 전략 성과로 볼 수 없다. 기간을 나눈 검증은 6장에 있다.", size=9)
h3("9-3. 대박 종목 의존도")
rows = [["제외", "12개월 SPY 초과 평균"]]
for q in [1.0, 0.99, 0.95, 0.90]:
    y = x12[x12 <= x12.quantile(q)]
    rows.append(["없음" if q == 1 else f"상위 {100 - q * 100:.0f}%", pcs(y.mean())])
table(rows, widths=[4, 4])
h3("9-4. 생존편향")
yrm = S.assign(yr=pd.to_datetime(S["period"]).dt.year).groupby("yr")["r6m"].apply(lambda s: s.isna().mean())
para(f"S&P500 비교군 중 야후 시세가 없는 행이 {pc(miss_sp)}다(2012년 {pc(yrm.get(2012))} → 2024년 {pc(yrm.get(2024))}). "
     "대부분 이후 상장폐지·인수된 종목이라, 앞쪽 기간의 비교군은 살아남은 종목 쪽으로 치우쳐 있다. "
     "특징 백분위는 살아남은 비교군 기준이므로, 초기 기간의 '변동성 높음'은 실제보다 조금 약하게 잡혔을 가능성이 있다.", size=9)
h3("9-5. 과적합 점검 — 결과가 좋을 때 먼저 의심한 것")
bullet("학습 구간에서 고른 설정은 검증·테스트에서 한 번만 확인했다. 테스트를 보고 설정을 바꾸지 않았다(확신 프로필 규칙은 테스트에서 실패한 그대로 기록).")
bullet("모든 규칙은 같은 후보 안 무작위 1,000회와 비교했다. 학습에서 99 백분위였던 규칙이 테스트에서 61로 떨어진 것이 이 점검으로 드러났다.")
bullet("머신러닝은 정답을 섞어 학습한 가짜 모델과 비교했다. 미국 모델은 가짜보다 낮아 무효로 판정했다.")
bullet("수주잔고 신호는 매개변수를 고르지 않았고(상위 30%·2분기는 미리 정함), 미국·한국 두 시장에서 같은 방향, 한국은 2025년(조선·전력기기 급등)을 뺀 결과를 따로 냈다.")
bullet("미래정보: 재무·수주잔고는 분기말 이전에 공시된 값만(한국 연간 매출은 다음 해 4월부터), 수주상황은 보고서 접수일부터 사용했다.")

# 10. 한계
h("10. 한계")
bullet("13F는 분기말 미국 상장 주식·옵션 명단만 보여준다. 그의 주력인 채권·통화·원자재·선물 포지션과 공매도는 안 보인다. 매크로 판단의 상당 부분이 빠져 있다.")
bullet("분기 안의 매매는 보이지 않는다. 분기 안에 사서 판 종목은 분석에서 빠지고, 그 결과 '산 분기 수익률'은 좋게 보인다.")
bullet("콜옵션은 평가액이 기초 주식 기준으로 보고돼 실제 투자 금액과 다르다. 이번 특징 분석은 주식만 사용했다.")
bullet(f"시세·재무가 없는 상장폐지 종목(매수의 약 {pc(E.loc[buy, 'r6m'].isna().mean())})은 특징 계산에서 빠졌다.")
bullet("업종은 SEC SIC 코드 근사치라 GICS와 다르다(구글·메타가 IT로 분류됨).")
bullet("4장의 특징 수치는 특징을 찾은 같은 기간 안에서 잰 것이다. 전략 성과는 6·8장의 기간 분할 결과로 판단한다.")
bullet("미국 수주잔고는 2018년 회계기준 변경 뒤에야 공시돼 학습 구간이 없고(2019~ 29분기), 공시 회사가 분기당 약 41%로 산업재·IT에 쏠린다.")
bullet("한국 수주잔고는 원문 표를 정해진 규칙으로 읽은 값이라 회사마다 잡음이 있다(보고서 종류별 표 범위가 다른 회사). 분기당 신호 종목 약 23개, 상위 30%는 약 8종목으로 집중된다.")
bullet("한국 결과에는 2020년 이후 조선·전력기기 업종 장세가 섞여 있다. 그래서 업종 전부·하위 30%·롱숏과 함께 비교했다.")

# 11. 결론
h("11. 결론")
bullet("드러켄밀러가 산 종목은 '변동성 높은 고성장주'라는 뚜렷한 공통점이 있지만, 그 공통점만으로는 돈을 벌 수 없다. 같은 종목군 안에서의 실력은 크게 산 종목에만 보이고, 그 추가 판단은 공개 숫자로 재현되지 않았다.")
bullet("그의 성과를 만든 것은 정성적인 눈이다. 18개월 뒤를 보고, 유동성으로 시장 전체를 판단해 하락장 막바지에 크게 걸고(2022년 4분기 엔비디아·메타·버티브), "
       "맞으면 키우고 틀리면 한 분기 안에 자른다. 숫자로 옮길 수 있는 것은 그 눈의 일부 — 앞으로 들어올 매출 — 뿐이었다.")
bullet("따라 사기(13F 공시일)·특징 규칙·조합 750개·머신러닝은 모두 표본 밖에서 무작위와 구별되지 않거나 특정 시기에만 통했다. 이 결과를 그대로 기록했다.")
bullet(f"남은 하나는 '수주잔고가 매출보다 빨리 쌓이는 회사'다. 그가 산 성장주의 핵심인 '앞으로 들어올 매출'을 숫자로 측정한 신호로, 미국 IC {us_ic['IC']:.3f}(t {us_ic['t']:.2f})·"
       f"한국 IC {kr_ic['IC']:.3f}(t {kr_ic['t']:.2f}), 공시 뒤에도 천천히 반영되고 실제로 4분기 뒤 매출을 맞혔다.")
bullet(f"전략으로는 한국 수주 업종 상위 30%(연 {pc(_ka['CAGR'], 1)}, 알파 t {_ka['알파_t']:.2f}, 2025년 제외 {_kax['알파_t']:.2f})를 주력 후보로, "
       "미국 신호는 다른 전략(6-4 기본 규칙 등)에 붙이는 보조 재료로 쓴다.")
bullet("다음 단계: 한국 신호 규칙을 지금 상태로 고정하고 2026년 4분기부터 매 분기 결과를 그대로 쌓는다(포워드 검증). 원문 표 읽기 오류를 표본 점검으로 줄인다.")

doc.save(OUT)
print(OUT)
try:
    sys.path.insert(0, r"C:\Users\arlo0\OneDrive\바탕 화면\취업\jobapp")
    from jobkit.common import to_pdf
    print(to_pdf(OUT))
except Exception as e:  # noqa
    print("PDF 변환 실패:", e)
