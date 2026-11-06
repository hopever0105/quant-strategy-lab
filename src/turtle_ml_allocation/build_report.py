"""보고서 생성: results/ → 보고서_초안.docx (+ results/fig_*.png)
실행: python quant/strategies/turtle_ml_allocation/build_report.py [최종]
  인자 '최종'을 주면 터틀_ML_자산배분_보고서.docx 로 저장한다."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from signals import FEAT_KO  # noqa: E402

RES = HERE / "results"
FINAL = len(sys.argv) > 1 and sys.argv[1] == "최종"
OUT = HERE / ("터틀_ML_자산배분_보고서.docx" if FINAL else "보고서_초안.docx")

M = json.loads((RES / "metrics.json").read_text(encoding="utf-8"))
D = pd.read_csv(RES / "daily.csv", index_col=0, parse_dates=True)
YR = pd.read_csv(RES / "yearly.csv", index_col=0)
ROB = pd.read_csv(RES / "robustness.csv")
PLC = pd.read_csv(RES / "placebo.csv")
PLC_MA = pd.read_csv(RES / "placebo_ma200.csv")
CAL = pd.read_csv(RES / "ml_calibration.csv")
GRP = pd.read_csv(RES / "ml_by_group.csv")
NEW = pd.read_csv(RES / "ml_new_assets.csv")
COEF = pd.read_csv(RES / "ml_coefs.csv", index_col=0)
UNI = pd.read_csv(RES / "universe_info.csv", index_col=0)
TR = pd.read_csv(RES / "trades_turtle.csv", parse_dates=["e_date"])
CORR = pd.read_csv(RES / "corr_weekly_oos.csv", index_col=0)
RPW = pd.read_csv(RES / "rp_avg_weight.csv", index_col=0).iloc[:, 0]
PLC_EXT = pd.read_csv(RES / "placebo_ext.csv")
COEF_EXT = pd.read_csv(RES / "ml_coefs_ext.csv", index_col=0)

plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
S4 = ["리스크 패리티", "60/40", "터틀 규칙", "터틀 + ML"]
COL = {"리스크 패리티": "#7F7F7F", "60/40": "#BF9000", "터틀 규칙": "#2F5597", "터틀 + ML": "#C00000"}
NAVY = "1F3864"
OOSK = "표본 밖 2008-01~2026-11"
meta = M["meta"]; main = M["main"]; dg = M["diag"]
oos0 = pd.Timestamp(meta["oos_start"])
pc = lambda x, d=1: f"{(0.0 if abs(x) < 0.5 * 10 ** (-d - 2) else x)*100:.{d}f}%"   # -0.0% 표기 방지
f2 = lambda x: f"{x:.2f}"


def clean(a):
    a.grid(alpha=.25); [a.spines[k].set_visible(False) for k in ("top", "right")]


# ───────── 차트
o = D.loc[oos0:]
fig, ax = plt.subplots(2, 1, figsize=(7.4, 5.0), dpi=170, sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
for c in S4:
    nav = (1 + o[c]).cumprod()
    ax[0].plot(nav.index, nav, label=c, color=COL[c], lw=1.1)
    ax[1].plot(nav.index, nav / nav.cummax() - 1, color=COL[c], lw=0.8)
ax[0].set_ylabel("자산(2008-01-02=1)"); ax[0].legend(ncol=4, fontsize=8, frameon=False)
ax[1].set_ylabel("낙폭")
for a in ax: clean(a)
plt.tight_layout(); plt.savefig(RES / "fig_equity.png"); plt.close()

yr = YR.loc[YR.index >= 2008, S4] * 100
fig, ax = plt.subplots(figsize=(7.4, 2.8), dpi=170)
x = np.arange(len(yr)); w = .2
for i, c in enumerate(S4):
    ax.bar(x + (i - 1.5) * w, yr[c], w, color=COL[c], label=c)
ax.set_xticks(x); ax.set_xticklabels(yr.index, rotation=90, fontsize=7); ax.set_ylabel("연 수익률(%)")
ax.axhline(0, color="k", lw=.6); ax.legend(ncol=4, fontsize=7, frameon=False); clean(ax)
plt.tight_layout(); plt.savefig(RES / "fig_yearly.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.7), dpi=170)
c_ = CAL[CAL.n > 0]
ax[0].bar(range(len(c_)), c_.n, color="#9DC3E6")
ax[0].set_xticks(range(len(c_))); ax[0].set_xticklabels(c_["bin"], fontsize=7); ax[0].set_ylabel("신호 수")
ax[0].set_title("예측 확률 분포 (표본 밖 신호)", fontsize=9); clean(ax[0])
ax[1].plot(c_.p_mean, c_.win, "o-", color=COL["터틀 + ML"], label="실제 승률")
ax[1].plot([0.4, 0.75], [0.4, 0.75], "--", color="#7F7F7F", lw=1, label="예측 = 실제")
ax[1].axhline(dg["base_rate"], color=COL["터틀 규칙"], lw=1, ls=":", label=f"전체 평균 {dg['base_rate']:.0%}")
ax[1].set_xlabel("예측 확률 평균"); ax[1].set_ylabel("20일 뒤 수익>0 비율"); ax[1].legend(fontsize=7, frameon=False)
ax[1].set_title("예측 확률 vs 실제 결과", fontsize=9); clean(ax[1])
plt.tight_layout(); plt.savefig(RES / "fig_calib.png"); plt.close()

r_ml = ROB.set_index("label").loc["기본: 터틀 + ML"]; r_ma = ROB.set_index("label").loc["200일선: + ML"]
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5), dpi=170)
for a, (vals, act, ttl) in zip(ax, [
        (PLC.auc, dg["auc"], "터틀: AUC"),
        (PLC.t_vs_rule, r_ml.t_vs_rule, "터틀: 규칙 대비 알파 t"),
        (PLC_MA.t_vs_rule, r_ma.t_vs_rule, "200일선: 규칙 대비 알파 t")]):
    a.hist(vals, bins=10, color="#A6A6A6", label="가짜 학습 20회")
    a.axvline(act, color=COL["터틀 + ML"], lw=1.8, label=f"실제 {act:.2f}")
    a.set_title(ttl, fontsize=8.5); a.legend(fontsize=6.5, frameon=False); clean(a)
plt.tight_layout(); plt.savefig(RES / "fig_placebo.png"); plt.close()

rb = ROB[ROB.note.isna()].copy()
fig, ax = plt.subplots(figsize=(7.4, 3.3), dpi=170)
cl = ["#C00000" if "ML" in l else ("#7F7F7F" if "패리티" in l else "#2F5597") for l in rb.label]
ax.barh(range(len(rb)), rb.sharpe, color=cl)
ax.set_yticks(range(len(rb))); ax.set_yticklabels(rb.label, fontsize=7); ax.invert_yaxis()
ax.axvline(main["리스크 패리티"][OOSK]["sharpe"], color="#7F7F7F", ls="--", lw=1, label="리스크 패리티(기본)")
ax.set_xlabel("표본 밖 샤프 지수"); ax.legend(fontsize=7, frameon=False); clean(ax)
plt.tight_layout(); plt.savefig(RES / "fig_robust.png"); plt.close()

cm = COEF.mean()
fig, ax = plt.subplots(figsize=(7.4, 2.2), dpi=170)
ax.bar([FEAT_KO[k] for k in cm.index], cm.values, yerr=COEF.std().values, color="#2F5597", capsize=3)
ax.axhline(0, color="k", lw=.6); ax.set_ylabel("표준화 계수"); ax.tick_params(axis="x", labelsize=7); clean(ax)
plt.tight_layout(); plt.savefig(RES / "fig_coef.png"); plt.close()

EX = M["ext_diag"]
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5), dpi=170, gridspec_kw={"width_ratios": [1, 1, 1.6]})
for a, (mdl, ko) in zip(ax[:2], (("logit", "로지스틱"), ("gbm", "부스팅"))):
    a.hist(PLC_EXT[PLC_EXT.model == mdl].auc, bins=10, color="#A6A6A6", label="가짜 학습 20회")
    a.axvline(EX[ko]["auc"], color=COL["터틀 + ML"], lw=1.8, label=f"실제 {EX[ko]['auc']:.3f}")
    a.set_title(f"변수 확장 {ko}: AUC", fontsize=8.5); a.legend(fontsize=6.5, frameon=False); clean(a)
ce = COEF_EXT.mean().sort_values()
ax[2].barh([FEAT_KO[k] for k in ce.index], ce.values, xerr=COEF_EXT.std()[ce.index].values, color="#2F5597", capsize=2)
ax[2].axvline(0, color="k", lw=.6); ax[2].tick_params(axis="y", labelsize=6); ax[2].set_title("변수 확장 로지스틱 계수", fontsize=8.5)
clean(ax[2])
plt.tight_layout(); plt.savefig(RES / "fig_ext.png"); plt.close()

# ───────── 문서
doc = Document()
sec = doc.sections[0]; sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(1.8); sec.top_margin = Cm(1.5); sec.bottom_margin = Cm(1.5)
st = doc.styles["Normal"]; st.font.name = "Malgun Gothic"; st.font.size = Pt(9.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Malgun Gothic"); st.paragraph_format.space_after = Pt(3)


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr(); s_ = OxmlElement("w:shd")
    s_.set(qn("w:val"), "clear"); s_.set(qn("w:color"), "auto"); s_.set(qn("w:fill"), color); tcPr.append(s_)


def para(text, bold=False, size=None, color=None, after=None, italic=False):
    p = doc.add_paragraph(); r = p.add_run(text); r.bold = bold; r.italic = italic
    if size: r.font.size = Pt(size)
    if color: r.font.color.rgb = RGBColor.from_string(color)
    if after is not None: p.paragraph_format.space_after = Pt(after)
    return p


def h(text):
    p = para(text, bold=True, size=12, color=NAVY, after=3); p.paragraph_format.space_before = Pt(8)
    pPr = p._p.get_or_add_pPr(); b = OxmlElement("w:pBdr"); bt = OxmlElement("w:bottom")
    for k, v in (("val", "single"), ("sz", "6"), ("space", "1"), ("color", NAVY)): bt.set(qn(f"w:{k}"), v)
    b.append(bt); pPr.append(b)


def h3(text):
    para(text, bold=True, size=10, color=NAVY, after=2)


def bullet(text):
    p = doc.add_paragraph(style="List Bullet"); p.paragraph_format.space_after = Pt(1)
    p.add_run(text).font.size = Pt(9.5)


def table(rows, widths=None, size=8, first_left=True, hl=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = ""; p = c.paragraphs[0]; r = p.add_run(str(v)); r.font.size = Pt(size)
            p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if (j == 0 and first_left) else WD_ALIGN_PARAGRAPH.CENTER
            if i == 0:
                r.bold = True; r.font.color.rgb = RGBColor(255, 255, 255); shade(c, NAVY)
            elif hl and i in hl:
                shade(c, "DEEAF6")
    if widths:
        for row in t.rows:
            for j, w_ in enumerate(widths): row.cells[j].width = Cm(w_)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def pic(name, w=16.5):
    doc.add_picture(str(RES / name), width=Cm(w)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


mk = {k: main[k][OOSK] for k in S4}
rob = ROB.set_index("label")
pl_auc_m, pl_auc_s = PLC.auc.mean(), PLC.auc.std()
y08, y22 = YR.loc[2008], YR.loc[2022]

# 표제
para("다자산 ETF 추세추종(터틀 트레이딩) + 머신러닝 메타 라벨링", bold=True, size=16, color=NAVY, after=1)
para(f"백테스트 보고서{'' if FINAL else ' (초안)'} · 송호빈 · 데이터 기준일 {meta['data_end']} · 실제 매매 없음(포트폴리오용 검증)",
     size=9, color="595959", after=6)

h("1. 요약")
bullet(f"대상: 4대 자산군(주식 10·채권 6·실물 6·통화 3) 미국 상장 ETF {meta['n_assets']}개. ETF 상장 전 구간은 같은 자산을 "
       f"추종하는 인덱스 펀드로 이어 붙이고, 각 자산은 데이터가 쌓인 시점부터 편입(1998년 {meta['n_assets_by_year']['1998']}개 → "
       f"2013년 초부터 {meta['n_assets_by_year']['2026']}개).")
bullet(f"방법: 터틀 20일 돌파 신호(1차 모델) + '이 신호가 20거래일 뒤 수익일 확률'을 예측하는 로지스틱 회귀(2차 모델). "
       f"1998~2007년으로 학습하고 매년 다시 학습, 2008-01~2026-11을 처음 보는 데이터로 검증.")
bullet(f"표본 밖 결과: 터틀 규칙 샤프 {f2(mk['터틀 규칙']['sharpe'])}·최대 낙폭 {pc(mk['터틀 규칙']['MDD'])}, "
       f"터틀+ML 샤프 {f2(mk['터틀 + ML']['sharpe'])}·최대 낙폭 {pc(mk['터틀 + ML']['MDD'])}, "
       f"리스크 패리티 샤프 {f2(mk['리스크 패리티']['sharpe'])}·최대 낙폭 {pc(mk['리스크 패리티']['MDD'])}, "
       f"60/40 샤프 {f2(mk['60/40']['sharpe'])}·최대 낙폭 {pc(mk['60/40']['MDD'])}.")
bullet(f"위기 구간: 2008년 터틀 {pc(y08['터틀 규칙'])} (60/40 {pc(y08['60/40'])}), 2022년 터틀 {pc(y22['터틀 규칙'])} "
       f"(60/40 {pc(y22['60/40'])}). 추세추종은 큰 하락장에서 손실을 막았지만 횡보·반등장에서 수익이 작았다.")
bullet(f"ML 예측력: 표본 밖 AUC {dg['auc']:.3f}. 정답을 무작위로 섞어 학습한 '가짜 학습' 20회 평균 {pl_auc_m:.3f}"
       f"(표준편차 {pl_auc_s:.3f})과 구별되지 않았다. 규칙 대비 알파 t값 {f2(rob.loc['기본: 터틀 + ML','t_vs_rule'])}. "
       "결론: 이 입력 변수들로는 터틀 신호의 성패를 예측할 수 없었다.")
bullet(f"강건성: 채널을 55일로 늘리면 샤프 {f2(rob.loc['채널 55일: 터틀 규칙','sharpe'])}로 좋아지고 10일이면 "
       f"{f2(rob.loc['채널 10일: 터틀 규칙','sharpe'])}로 나빠졌다. 거래비용 2배에서 터틀 샤프 {f2(rob.loc['비용 2배: 터틀 규칙','sharpe'])}. "
       f"월간 200일선 신호가 가장 좋았다(샤프 {f2(rob.loc['200일선: 규칙','sharpe'])}). 200일선에 ML을 붙이면 t {f2(rob.loc['200일선: + ML','t_vs_rule'])}였지만 "
       f"가짜 학습도 t {PLC_MA.t_vs_rule.min():.1f}~{PLC_MA.t_vs_rule.max():.1f}가 나와 예측력의 증거로 보기 어렵다.")
bullet(f"변수 확장(신용 스프레드·금리·자산 간 상관·변동성 국면·돌파 강도·자산군 등 18개): 부스팅 AUC {EX['부스팅']['auc']:.3f}로 "
       f"가짜 학습 최고값({PLC_EXT[PLC_EXT.model=='gbm'].auc.max():.3f})을 조금 넘었지만, 포트폴리오에서는 규칙 대비 알파 t "
       f"{f2(rob.loc['변수 확장 부스팅: 터틀 + ML','t_vs_rule'])}로 개선이 없었다. 약한 신호가 있어도 비용·비중 조절을 넘을 만큼은 아니었다.")

h("2. 전략 규칙")
table([
    ["항목", "내용"],
    ["유니버스", "주식(미국 대형·나스닥100·소형, 유럽, 일본, 신흥국, 한국, 중국, 인도, 중남미) · 채권(미국채 단기·중기·장기, 물가연동채, "
               "투자등급 회사채, 하이일드) · 실물(금, 은, 원유, 원자재 종합, 농산물, 리츠) · 통화(달러 인덱스, 유로, 엔화). "
               "결과를 보기 전에 목록 고정. 다우존스는 S&P500과 사실상 같은 자산이라 제외, VIX는 살 수 있는 자산이 아니라 입력 변수로만 사용"],
    ["1차 신호(터틀)", "종가가 직전 20일 최고 종가를 넘으면 매수 결정 → 다음 날 종가에 진입. 종가가 직전 20일 최저 종가 아래 또는 "
                     "진입가 − 2N(N = 20일 평균 종가 변동폭) 아래면 청산 결정 → 다음 날 종가에 청산"],
    ["비중", "월말에 자산군 4개가 위험을 똑같이 나눠 갖고(자산이 있는 자산군 수로 나눔), 자산군 안에서는 자산마다 똑같이 나눈 뒤 "
            "60일 변동성으로 나눔(변동성 역수). 합이 1 = 리스크 패리티. 터틀은 들고 있는 자산만 이 비중, 나머지는 현금(미 3개월 국채)"],
    ["2차 모델(ML)", "입력: " + ", ".join(FEAT_KO.values()) + ". 정답: 진입 후 20거래일 수익 > 0. 로지스틱 회귀(변수 표준화). "
                    "매년 1월 1일에 그때까지 정답이 확정된 신호로만 다시 학습(확정 안 된 마지막 20일치 신호 제외)"],
    ["ML 비중 배수", "배수 = 2 × 예측 확률(0.7 → 1.4배), 0.3 미만이면 진입 안 함. 합이 100%를 넘으면 100%로 줄임(레버리지 없음)"],
    ["거래비용", "바뀐 비중의 0.05%(편도: 수수료 + 호가 차이)"],
    ["기간", f"모의 운용 {meta['sim_start']}~{meta['data_end']}, 학습 1998~2007, 표본 밖 검증 {meta['oos_start']}~{meta['data_end']}"],
    ["비교 대상", "리스크 패리티(같은 비중을 항상 100% 투자), 60/40(S&P500 60 + 미국채 7~10년 40, 월 재조정)"],
], [3.0, 14.4], size=7.5)

h("3. 데이터")
para("시세는 FinanceDataReader(야후)의 배당·분할 반영 수정종가, VIX·장단기 금리차·3개월 국채 금리·환율은 FRED. 한국거래소(KRX) 요청은 없다. "
     "ETF 상장 전 구간은 아래 대용으로 이어 붙였고, 둘이 겹치는 기간의 차이를 점검했다(환율 대용은 처음 쓴 야후 환율의 주간 상관이 "
     "0.83에 그쳐 FRED 뉴욕 정오 고시 환율로 바꿨다).", after=3)
rows = [["ETF", "자산", "상장일", "상장 전 대용", "겹친 기간", "주간 상관", "연 수익 차이"]]
for k, r in UNI.iterrows():
    if r.proxy != "-":
        rows.append([k, r["name"], r.etf_start, r.proxy.replace("FRED:", ""), f"{r.overlap_years:.0f}년", f"{r.corr_w:.3f}", pc(r.ann_diff, 2)])
table(rows, [1.4, 3.8, 2.2, 2.4, 1.6, 1.7, 2.0], size=7)
nb = ", ".join(f"{k} {v}" for k, v in meta["jumps"].items())
para(f"가격 단절 점검: 하루 ±25% 넘는 움직임은 {nb}. 모두 실제 시장 사건(2008년 반등, 2020년 유가 폭락, 2026년 은 급락)이라 "
     f"보정하지 않았다(보정 건수 {meta['price_breaks_fixed']}건). 대용이 없는 자산(나스닥100·일본·한국·중국·인도·중남미·은·원유·농산물)은 "
     "ETF 상장 252거래일 뒤부터 편입했다. 원유·은·농산물은 선물 연속가격을 이어 붙이면 만기 교체 때 가격이 튀어 가짜 돌파가 생기므로 쓰지 않았다.",
     size=8.5)

h("4. 결과 (표본 밖 2008-01 ~ 2026-11)")
rows = [["전략", "연 수익률", "변동성", "샤프", "최대 낙폭", "평균 투자 비중", "연 회전율", "리스크 패리티 대비 알파(t)"]]
for k in S4:
    m = mk[k]
    rows.append([k, pc(m["CAGR"], 2), pc(m["vol"]), f2(m["sharpe"]), pc(m["MDD"]), pc(m["exposure"], 0), f"{m['turnover']:.1f}배",
                 "-" if k == "리스크 패리티" else f"{pc(m['alpha'], 2)} ({f2(m['alpha_t'])})"])
table(rows, [2.8, 1.9, 1.7, 1.3, 1.9, 2.3, 1.9, 3.6], size=7.5, hl=[3, 4])
pic("fig_equity.png")
pw = RPW.groupby(UNI.group).sum()
para(f"읽는 법: 리스크 패리티는 위험을 똑같이 나눠서 변동성이 낮은 자산에 돈이 몰린다. 표본 밖 평균 비중은 채권 {pc(pw['채권'],0)}"
     f"(미국채 1~3년만 {pc(RPW['SHY'],0)}), 통화 {pc(pw['통화'],0)}, 실물 {pc(pw['실물'],0)}, 주식 {pc(pw['주식'],0)}로, 레버리지 없이 쓰면 "
     "연 수익률이 낮다. 60/40과는 위험 수준이 달라 수익률보다 샤프·최대 낙폭으로 비교하는 것이 맞다.", size=8.5)
pic("fig_yearly.png")

h3("기간을 나눠 보면")
rows = [["전략"] + list(main["터틀 규칙"].keys())]
for k in S4:
    rows.append([k] + [f"{f2(v['sharpe'])} / {pc(v['MDD'])}" for v in main[k].values()])
table(rows, [2.8, 3.6, 3.6, 3.6, 3.6], size=7.5)
para("칸 안 값: 샤프 / 최대 낙폭. 터틀은 학습 기간이 포함된 전체 기간에서 샤프가 더 높고, 2017년 이후에는 거의 0이다. "
     "추세추종이 최근 10년 횡보·급반등 장세에서 약했다는 뜻이다.", size=8.5)
ts = M["trades"]["터틀 규칙"]
para(f"거래: 표본 밖 {ts['trades']:,}건(연 {ts['per_year']:.0f}건, 자산당 연 {ts['per_year']/meta['n_assets']:.1f}건), "
     f"이긴 거래 {pc(ts['win'],0)}, 거래당 평균 {pc(ts['avg_ret'],2)}, 평균 보유 {ts['hold']:.0f}거래일. 승률이 낮고 소수의 큰 추세로 메우는 "
     "터틀의 구조가 그대로 나타났다.", size=8.5)

h("5. 머신러닝 진단")
para(f"표본 밖 신호 {dg['n']:,}개 중 20거래일 뒤 수익이 난 비율은 {pc(dg['base_rate'],1)}이다. 모델의 예측 확률은 대부분 0.5~0.7에 몰려 "
     f"'진입 안 함(0.3 미만)'이 한 번도 나오지 않았고, 배수 2p가 평균 1.2배 정도라 ML 버전은 사실상 투자 비중만 늘어났다"
     f"(평균 투자 비중 {pc(mk['터틀 규칙']['exposure'],0)} → {pc(mk['터틀 + ML']['exposure'],0)}).", after=3)
pic("fig_calib.png")
rows = [["구분", "신호 수", "실제 승률", "AUC"]]
for _, r in GRP.iterrows():
    rows.append([r.group, int(r.n), pc(r.base, 1), f"{r.auc:.3f}"])
for _, r in NEW.iterrows():
    rows.append(["학습 때 없던 자산(2008년 이후 편입)" if r.new else "학습 때 있던 자산", int(r.n), pc(r.base, 1), f"{r.auc:.3f}"])
table(rows, [6.5, 2.5, 2.5, 2.5], size=7.5)
para("AUC는 '이긴 신호에 더 높은 확률을 줬는가'를 0~1로 잰 값이다. 0.5면 동전 던지기와 같다. 자산군별로도, 처음 보는 자산에서도 0.5 근처였다.",
     size=8.5)
pic("fig_coef.png", 15)
pos = (COEF > 0).mean()
para(f"변수별 평균 계수(막대)와 해마다 다시 학습할 때의 흔들림(선). 정배열·RSI는 19번 모두 +, 200일선 대비 위치는 19번 모두 −로 "
     f"부호가 일관됐다(이미 많이 오른 자산의 돌파는 덜 이긴다). 하지만 크기가 작다: 표준화 계수 {cm['align']:.2f}는 그 변수가 1표준편차 "
     f"달라질 때 확률이 약 {cm['align']*0.25*100:.0f}%p 움직이는 정도다. 밴드폭·VIX·금리차는 해마다 부호가 바뀌었다"
     f"(+로 나온 해 {pos['bbw']:.0%}·{pos['vix']:.0%}·{pos['t10y2y']:.0%}).", size=8.5)
pic("fig_placebo.png")
para(f"가짜 학습: 정답(20일 뒤 수익 여부)을 무작위로 섞어 같은 절차로 20번 학습했다. 실제 AUC {dg['auc']:.3f}는 가짜 학습 분포"
     f"({PLC.auc.min():.3f}~{PLC.auc.max():.3f}) 안에 있다. 규칙 대비 알파 t값도 가짜 학습(평균 {PLC.t_vs_rule.mean():.2f})이 오히려 높았다. "
     f"200일선 + ML의 t {f2(r_ma.t_vs_rule)}도 가짜 학습 범위({PLC_MA.t_vs_rule.min():.1f}~{PLC_MA.t_vs_rule.max():.1f}) 안이어서, "
     "성과 차이는 예측력이 아니라 비중이 달라진 효과로 본다.", size=8.5)

h("6. 강건성 검증")
para("조건을 바꿔도 결과가 유지되는지 확인했다. 아래 표의 모든 값은 표본 밖(2008-01~2026-11) 기준이다.", after=3)
rows = [["조건", "연 수익률", "샤프", "최대 낙폭", "평균 투자 비중", "규칙 대비 알파 t"]]
for _, r in ROB.iterrows():
    rows.append([r.label + (" *" if r.note == "사후 점검" else ""), pc(r.CAGR, 2), f2(r.sharpe), pc(r.MDD), pc(r.exposure, 0),
                 "-" if pd.isna(r.t_vs_rule) else f2(r.t_vs_rule)])
table(rows, [6.4, 2.0, 1.5, 2.0, 2.4, 2.6], size=7)
para("* 사후 점검: 결과를 본 뒤 설계 문제(① 2p 배수가 투자 비중만 늘리는 문제, ② 정답을 '20일 뒤 수익' 대신 그 거래의 실제 손익으로 두는 "
     "일반적인 메타 라벨링 방식)를 확인하려고 추가했다. 어느 쪽도 결론을 바꾸지 않았다.", size=8)
pic("fig_robust.png")
bullet(f"채널 길이: 10일 {f2(rob.loc['채널 10일: 터틀 규칙','sharpe'])} → 20일 {f2(rob.loc['기본: 터틀 규칙','sharpe'])} → "
       f"55일 {f2(rob.loc['채널 55일: 터틀 규칙','sharpe'])}. 짧을수록 거래가 많아 비용과 가짜 돌파에 약하다.")
bullet(f"거래비용 2배(0.1%): 터틀 샤프 {f2(rob.loc['비용 2배: 터틀 규칙','sharpe'])}. 연 회전율이 {mk['터틀 규칙']['turnover']:.1f}배라 비용에 민감하다.")
bullet(f"유니버스: 1998년 초에 이미 있던 {len(meta['fixed_assets'])}개만 쓰면 터틀 샤프 {f2(rob.loc['고정 유니버스: 터틀 규칙','sharpe'])}, "
       f"리스크 패리티 {f2(rob.loc['고정 유니버스: 리스크 패리티','sharpe'])}. 늦게 들어온 자산에 기댄 결과가 아니다.")
bullet(f"모델: 로지스틱 대신 그래디언트 부스팅도 AUC {M['diag_gbm']['auc']:.3f}, 샤프 {f2(rob.loc['부스팅 모델: 터틀 + ML','sharpe'])}로 같았다.")
bullet(f"신호 종류: 월간 200일선 {f2(rob.loc['200일선: 규칙','sharpe'])}, 12개월 모멘텀 {f2(rob.loc['12개월 모멘텀: 규칙','sharpe'])}로 "
       "일간 20일 돌파보다 좋았다. 월 1회만 판단해서 가짜 돌파와 비용이 적은 것이 이유로 보인다.")

h("7. 변수 확장 후 재검증 (사후 점검)")
para("기본 6개 변수로 효과가 없다는 결과를 본 뒤, 다음 단계로 정한 변수 확장을 같은 가짜 학습 기준으로 다시 검증했다. "
     "결과를 본 뒤 추가한 것이라 본 결과와 구분한다. 추가 변수: 신용 스프레드(무디스 Baa − 미국채 10년, 수준·3개월 변화), "
     "미국채 10년 금리(수준·3개월 변화), 자산 간 평균 상관(60일), 12개월 수익률, 변동성 국면(20일÷252일), 돌파 강도(돌파 폭 ÷ N), "
     "자산군 구분 4개. 하이일드 스프레드는 FRED가 최근 3년치만 제공해 쓰지 못했다.", after=3)
rows = [["모델 · 정답", "AUC", "가짜 학습 AUC(평균 / 최고)", "샤프", "평균 투자 비중", "규칙 대비 알파 t"]]
for ko, mdl in (("로지스틱", "logit"), ("부스팅", "gbm")):
    pe = PLC_EXT[PLC_EXT.model == mdl]
    rr = rob.loc[f"변수 확장 {ko}: 터틀 + ML"]
    rows.append([f"{ko} · 20일 뒤 수익", f"{EX[ko]['auc']:.3f}", f"{pe.auc.mean():.3f} / {pe.auc.max():.3f}", f2(rr.sharpe),
                 pc(rr.exposure, 0), f2(rr.t_vs_rule)])
    rr = rob.loc[f"변수 확장 {ko} + 정답=거래 손익: 터틀 + ML"]
    rows.append([f"{ko} · 거래 실제 손익", f"{EX[ko + '_거래손익']['auc']:.3f}", "-", f2(rr.sharpe), pc(rr.exposure, 0), f2(rr.t_vs_rule)])
for ko in ("200일선", "12개월 모멘텀"):
    rr = rob.loc[f"변수 확장 {ko}: + ML"]
    rows.append([f"{ko} 신호 · 로지스틱", f"{EX[ko]['auc']:.3f}", "-", f2(rr.sharpe), pc(rr.exposure, 0), f2(rr.t_vs_rule)])
rows.append(["(참고) 기본 6개 변수 · 로지스틱", f"{dg['auc']:.3f}", f"{PLC.auc.mean():.3f} / {PLC.auc.max():.3f}",
             f2(mk['터틀 + ML']['sharpe']), pc(mk['터틀 + ML']['exposure'], 0), f2(rob.loc['기본: 터틀 + ML', 't_vs_rule'])])
table(rows, [5.0, 1.5, 3.6, 1.4, 2.4, 2.6], size=7.5)
pic("fig_ext.png")
cme = COEF_EXT.mean()
bullet(f"부스팅 AUC {EX['부스팅']['auc']:.3f}는 가짜 학습 20회 중 가장 높은 값보다 조금 높아, 약한 예측력이 있을 가능성은 있다. "
       f"그러나 포트폴리오 성과는 개선되지 않았다(샤프 {f2(rob.loc['변수 확장 부스팅: 터틀 + ML','sharpe'])} vs 규칙 "
       f"{f2(mk['터틀 규칙']['sharpe'])}).")
bullet(f"계수가 해마다 일관된 변수는 자산군 구분(통화 {cme['g_통화']:+.2f}, 주식 {cme['g_주식']:+.2f}), RSI({cme['rsi']:+.2f}), "
       f"200일선 대비 위치({cme['dist200']:+.2f}), 돌파 강도({cme['brk_n']:+.2f}), 금리 3개월 변화({cme['dgs10_chg']:+.2f})였다. "
       "자산군 계수가 가장 안정적이라는 것은, 모델이 신호의 질보다 '주식은 원래 20일 뒤 오를 확률이 높다'는 자산군별 평균 차이를 주로 배웠다는 뜻이다.")
bullet("정답을 거래 실제 손익(승률 34%)으로 바꾸면 모델이 신호를 많이 걸러 투자 비중이 30% 안팎으로 줄고 샤프도 낮아졌다. "
       "터틀은 낮은 승률을 소수의 큰 추세로 메우는 구조라, 이기는 거래를 맞히려는 필터가 큰 추세까지 걸러 버린 것으로 본다.")
bullet(f"12개월 모멘텀 + 변수 확장은 규칙 대비 t {f2(rob.loc['변수 확장 12개월 모멘텀: + ML','t_vs_rule'])}였지만 2에 못 미치고, "
       "200일선 가짜 학습에서 t가 2를 넘는 경우도 있었던 만큼 근거로 삼지 않는다.")

h("8. 한계")
bullet("달러 기준 수익률이다. 원화 투자자의 환율 효과와 세금은 넣지 않았다.")
bullet("ETF 상장 전 구간은 대용 펀드로 이어 붙였다. 회사채·하이일드 대용은 주간 상관 0.84·0.83으로 차이가 있다(만기·신용등급 구성 차이).")
bullet("리스크 패리티를 레버리지 없이 써서 채권·현금성 자산 비중이 높다. 수익률 수준은 레버리지 여부에 따라 크게 달라진다.")
bullet("ETF 목록은 2026년 현재 상장된 상품에서 골랐다. 자산군을 빠짐없이 넣고 결과를 보기 전에 고정해서 사후 선택을 줄였지만 완전히 없앤 것은 아니다.")
bullet("입력 변수는 기본 6개, 확장 18개까지 봤다. 거래량·수급·밸류에이션 같은 다른 종류의 변수와, 신호 성패 대신 다음 달 자산별 수익률을 "
       "직접 예측하는 방식은 다루지 않았다.")
bullet("종가 기준 체결, 호가 차이는 고정 0.05%로 가정했다. 2008년 같은 위기에는 실제 비용이 더 컸을 수 있다.")

h("9. 결론")
para("1) 다자산 터틀 추세추종은 2008·2022년 같은 큰 하락장에서 손실을 거의 막았지만, 표본 밖 샤프는 단순 리스크 패리티보다 낮았다. "
     "특히 2017년 이후 성과가 거의 없었다.")
para("2) 머신러닝으로 신호를 걸러 비중을 조절하는 방식은 기본 6개 변수로도, 18개로 늘려도 포트폴리오 성과를 개선하지 못했다. "
     "예측력(AUC)은 정답을 섞은 가짜 학습과 거의 구별되지 않았고, 성과가 좋아 보인 경우(200일선 + ML)도 가짜 학습 범위 안이었다.")
para("3) 같은 틀에서 신호만 바꿔 보면, 월 1회 판단하는 200일선·12개월 모멘텀이 일간 돌파보다 안정적이었다. 신호를 거르는 것보다 "
     "어떤 신호를 어떤 주기로 쓰느냐가 성과에 더 큰 영향을 줬다.")
para("4) 이 프로젝트에서 가장 쓸모 있던 것은 검증 절차다: 상장 시점 기준 편입, 정답이 확정된 신호만 학습, 매년 재학습, 가짜 학습 대조. "
     "이 절차가 있었기 때문에 '좋아 보이는 결과'(200일선 + ML t 1.92, 부스팅 AUC 0.524)를 성과로 착각하지 않을 수 있었다.")
para("코드: quant-strategy-lab/turtle_ml_allocation (universe·data·signals·backtest·ml·run). 같은 데이터면 같은 결과가 나온다(난수 고정).",
     size=8.5, color="595959")
doc.save(OUT)
print(OUT)
