# -*- coding: utf-8 -*-
"""보고서 생성: results/ → 보고서_초안.docx
실행: python build_report.py [최종]   ('최종'이면 래리_윌리엄스_검증_보고서.docx)"""
import json
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

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FINAL = len(sys.argv) > 1 and sys.argv[1] == "최종"
OUT = HERE / ("래리_윌리엄스_검증_보고서.docx" if FINAL else "보고서_초안.docx")
NAVY = "1F3864"

S = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
rd = lambda f: pd.read_csv(RES / f)
COT, COTE, BOND, OIL, LAG, SEC, PAN = (rd(f) for f in ["m1_cot.csv", "m1_cot_events.csv", "m1_bond.csv", "m1_oil.csv",
                                                       "m1_oil_lagcorr.csv", "m1_oil_sectors.csv", "m1_panic.csv"])
TDM, ELE, SYS, NB, ALLC = (rd(f) for f in ["m2_tdm.csv", "m2_election.csv", "m2_system.csv", "m2_neighbors.csv", "m2_all_candidates.csv"])
Q, QY, QP, WF = rd("m3_quality.csv"), pd.read_csv(RES / "m3_yearly.csv", index_col=0), \
    pd.read_csv(RES / "m3_latest_picks.csv", dtype={"code": str}), rd("walk_forward.csv")
TR = rd("m2_trades.csv")
# 보완 분석·종목 분석·전체 시장 검증 (robust.py, picks.py, winners.py, factors.py)
RP, RB, RX, RST = rd("robust_power.csv"), rd("robust_bootstrap.csv"), rd("robust_ex2526.csv"), rd("robust_stagger.csv")
RWF, RLO, RSA, RBS, RKS = rd("robust_m3_walkforward.csv"), rd("robust_m3_leave_one_year.csv"), rd("robust_m3_sector_attr.csv"),     rd("robust_m3_by_sector.csv"), rd("robust_m3_sector_ks.csv")
SR = json.loads((RES / "summary_robust.json").read_text(encoding="utf-8"))
CON = pd.read_csv(RES / "m3_contribution.csv", dtype={"코드": str})
CONS = pd.read_csv(RES / "m3_contribution_summary.csv", index_col=0).iloc[:, 0]
WIN = pd.read_csv(RES / "winners_compare.csv", index_col=0)
FQ, FMB, FFF, FCB, FCV = rd("factors_quintiles.csv"), rd("factors_fama_macbeth.csv"), rd("factors_ff2x3.csv"),     rd("factors_combos.csv"), rd("factors_cv.csv")
FINFO = pd.read_csv(RES / "factors_info.csv", index_col=0).iloc[:, 0]


def pc(x, d=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    x = 0.0 if abs(x) < 0.5 * 10 ** (-d - 2) else x
    return f"{x * 100:.{d}f}%"


def f2(x):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"


def row(df, name_prefix, seg):
    r = df[(df["전략"].astype(str).str.startswith(name_prefix)) & (df["구간"] == seg)]
    return r.iloc[0] if len(r) else None


# ───────── 전체 시장 검증 차트
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
_q = FQ[FQ.방식 == "5분위(시총 통제)"].set_index("지표")
_fig, _ax = plt.subplots(1, 7, figsize=(13, 2.8), sharey=True)
for _a, _f in zip(_ax, ["회전율", "변동성60일", "MAX", "200일선대비", "ROE", "베타", "WR10"]):
    _v = [_q.loc[_f, k] * 100 for k in ["Q1", "Q2", "Q3", "Q4", "Q5"]]
    _a.bar(range(5), _v, color=["#9dc3e6", "#9dc3e6", "#9dc3e6", "#9dc3e6", "#1f4e79"])
    _a.set_xticks(range(5)); _a.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"], fontsize=7); _a.set_title(_f, fontsize=9); _a.grid(alpha=.3, axis="y")
_ax[0].set_ylabel("다음 1년 수익(%)")
_fig.suptitle("지표 5분위별 다음 1년 수익 (시가총액 통제, 2000~2025 평균)", fontsize=10)
_fig.tight_layout(); _fig.savefig(RES / "fig10_factor_quintiles.png", dpi=160); plt.close(_fig)
_cv = FCV[FCV.방식 == "Fama-MacBeth 다변량 계수"].set_index("지표")
_fig, _ax = plt.subplots(figsize=(10, 3.4))
_w = 0.25
for _i, (_f, _c) in enumerate([("회전율", "#c55a11"), ("변동성60일", "#7f7f7f"), ("200일선대비", "#1f4e79")]):
    _parts = [x.split(":") for x in _cv.loc[_f, "블록별"].split(", ")]
    _ax.bar([j + (_i - 1) * _w for j in range(len(_parts))], [float(v) * 100 for _, v in _parts], _w, color=_c, label=_f)
_ax.set_xticks(range(len(_parts))); _ax.set_xticklabels([k for k, _ in _parts], fontsize=8); _ax.axhline(0, color="k", lw=.6)
_ax.set_ylabel("Fama-MacBeth 계수(%p)"); _ax.legend(fontsize=8); _ax.grid(alpha=.3, axis="y")
_ax.set_title("3년 블록별 효과 (다변량, 순위 0→1일 때 다음 1년 수익 차이)", fontsize=10)
_fig.tight_layout(); _fig.savefig(RES / "fig11_factor_cv.png", dpi=160); plt.close(_fig)

# ───────── 문서
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(1.8)
sec.top_margin = Cm(1.5)
sec.bottom_margin = Cm(1.5)
st = doc.styles["Normal"]
st.font.name = "Malgun Gothic"
st.font.size = Pt(9.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Malgun Gothic")
st.paragraph_format.space_after = Pt(3)


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


def table(rows, widths=None, size=8, first_left=True, hl=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, rw in enumerate(rows):
        for j, v in enumerate(rw):
            c = t.cell(i, j); c.text = ""
            p = c.paragraphs[0]; r = p.add_run(str(v)); r.font.size = Pt(size)
            p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if (j == 0 and first_left) else WD_ALIGN_PARAGRAPH.CENTER
            if i == 0:
                r.bold = True; r.font.color.rgb = RGBColor(255, 255, 255); shade(c, NAVY)
            elif hl and i in hl:
                shade(c, "DEEAF6")
    if widths:
        for rw in t.rows:
            for j, w_ in enumerate(widths):
                rw.cells[j].width = Cm(w_)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def pic(name, w=16.5):
    doc.add_picture(str(RES / name), width=Cm(w))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def seg_rows(df, prefix, label=None, segs=("학습", "검증", "테스트", "전체")):
    out = []
    for sg in segs:
        r = row(df, prefix, sg)
        if r is None:
            continue
        out.append([label or prefix, sg, pc(r.CAGR), f2(r.샤프), pc(r.MDD), pc(r.get("BM_CAGR")), f2(r.get("BM_샤프")),
                    pc(r.get("BM_MDD")), f2(r.get("알파t"))])
        label = ""
    return out


HDR = ["전략", "구간", "CAGR", "샤프", "MDD", "BM CAGR", "BM 샤프", "BM MDD", "알파 t"]

# ───────── 주요 숫자
cc = S["cot_choice"]
spx_name = f"COT S&P500 N={cc['SPX'][0]} {cc['SPX'][1]}/{cc['SPX'][2]}"
wf = WF.set_index("전략")
oil_L, oil_m = S["oil_choice"]
pan_d, pan_h = S["panic_choice"]
sc = S["sys_choice"]
sys_name = SYS["전략"].iloc[0]
q_name = f"{S['m3_choice'][0]}·{S['m3_choice'][1]}"
tdm_k = TDM[(TDM.시장 == "코스피") & (TDM.월초일수 == 3)].set_index("구간")
tdm_s = TDM[(TDM.시장 == "S&P500(SPY)") & (TDM.월초일수 == 4)].set_index("구간")
ev_spx = COTE[(COTE.시장 == "S&P500") & (COTE.N주 == cc["SPX"][0]) & (COTE.기준 == f"{cc['SPX'][1]}/{cc['SPX'][2]}")]
ev_bond = COTE[(COTE.시장 == "미국 30년 국채") & (COTE.N주 == cc["BOND"][0]) & (COTE.기준 == f"{cc['BOND'][1]}/{cc['BOND'][2]}")]
evb_tr = ev_bond[(ev_bond.구간 == "학습") & ev_bond.조건.str.startswith("≥")].iloc[0]
evb_te = ev_bond[(ev_bond.구간 == "테스트") & ev_bond.조건.str.startswith("≥")].iloc[0]
evs_tr = ev_spx[(ev_spx.구간 == "학습") & ev_spx.조건.str.startswith("≤")].iloc[0]
evs_te = ev_spx[(ev_spx.구간 == "테스트") & ev_spx.조건.str.startswith("≤")].iloc[0]
pan_ev = PAN[PAN.전략.astype(str).str.startswith(f"[사건] 낙폭{int(pan_d * 100)}%·보유{pan_h}일")].set_index("구간")
lag_tr = LAG[(LAG.구간 == "학습") & (LAG.시차 >= 30)]
neg_share = (lag_tr.상관 < 0).mean()
sys_all = row(SYS, sys_name, "전체")
sys_te = row(SYS, sys_name, "테스트")
trig_only = row(SYS, "[뺌] 방아쇠만", "전체")
trig_only_te = row(SYS, "[뺌] 방아쇠만", "테스트")
q_vs_ew = row(Q, f"{q_name} vs 동일가중", "전체")
q_te = row(Q, q_name, "테스트")
q_tr = row(Q, q_name, "학습")
ew_te = row(Q, "벤치마크: 동일가중", "테스트")
el_k = ELE[ELE.시장 == "코스피"]
n_tr = int(len(TR))

VERDICT = [
    ["래리의 주장", "검증 방법", "결과", "판정"],
    ["상업자(헤지 주체) 순매수가 극단이면 S&P500이 오른다", "COT 상업자 지수 26·156주, 극단 20/80·25/75",
     f"학습: 지수 ≤{cc['SPX'][1]}일 때 13주 수익이 평소보다 {pc(evs_tr.차이)}(t {f2(evs_tr.NW_t)}), 테스트 t {f2(evs_te.NW_t)}. "
     f"워크포워드 샤프 {f2(wf.loc['COT S&P500 → SPY','샤프'])}(보유 {f2(wf.loc['COT S&P500 → SPY','BM_샤프'])}), 최대 낙폭 "
     f"{pc(wf.loc['COT S&P500 → SPY','MDD'],0)}(보유 {pc(wf.loc['COT S&P500 → SPY','BM_MDD'],0)})", "일부 맞음"],
    ["상업자 지수는 채권에서도 통한다", "30년 국채 COT",
     f"학습 t {f2(evb_tr.NW_t)} → 테스트 t {f2(evb_te.NW_t)}", "일부 맞음(학습 구간만)"],
    ["상업자 지수는 금·원유에서도 통한다", "금·WTI COT", "학습에서 고른 조합 기준, 학습·테스트 모두 유의하지 않음(|t|<1.3)", "안 맞음"],
    ["채권 가격이 주식을 선행한다", "채권 가격 > n일 평균일 때만 주식 보유",
     f"코스피: 전체 알파 t {f2(BOND[(BOND.전략.str.contains('코스피')) & (BOND.구간 == '전체')].iloc[0].알파t)}, "
     f"워크포워드 샤프 {f2(wf.loc['채권 필터 → 코스피','샤프'])}(보유 {f2(wf.loc['채권 필터 → 코스피','BM_샤프'])})", "안 맞음"],
    ["유가가 주가를 1.5~6개월 앞선다", f"유가 시차 20~130일 × 추세 20·60일(24개)",
     f"학습에서 고른 시차 {oil_L}일은 범위 안이지만, 시차 30일 이상 상관의 {pc(neg_share,0)}가 음(−). "
     f"워크포워드 알파 t {f2(wf.loc['유가 → 코스피','알파t'])}", "안 맞음(방향 반대)"],
    ["패닉(급락) 때 사라", "20일 고점 대비 10·15·20% 하락 후 60·120일 보유",
     f"고른 값({int(pan_d*100)}%·{pan_h}일) 학습 사건 {int(pan_ev.loc['학습','사건수'])}회, 평소 대비 +{pc(pan_ev.loc['학습','차이'])}"
     f"(t {f2(pan_ev.loc['학습','NW_t'])}); 테스트 사건 {int(pan_ev.loc['테스트','사건수'])}회 평균 {pc(pan_ev.loc['테스트','사건후평균'])}",
     "일부 맞음(표본 적음)"],
    ["월말·월초(TDM)에 주가가 오른다", "월 마지막 거래일 + 첫 3·4·5거래일",
     f"코스피(첫 3일) 일평균 차이 t: 학습 {f2(tdm_k.loc['학습','NW_t'])}, 검증 {f2(tdm_k.loc['검증','NW_t'])}, 테스트 "
     f"{f2(tdm_k.loc['테스트','NW_t'])}, 전체 {f2(tdm_k.loc['전체','NW_t'])}. S&P500은 전체 t {f2(tdm_s.loc['전체','NW_t'])}",
     "일부 맞음(코스피가 더 강함)"],
    ["선거 전후에 주가가 다르게 움직인다", "미국·한국 대선·총선 ±1개월", f"8개 조합 모두 |t| < 1.6", "안 맞음"],
    ["조건→시기→상태→방아쇠 깔때기로 거르면 좋아진다", f"후보 {S['grid_counts']['② 단기 시스템']}개 조합 중 학습 샤프 최고",
     f"26년간 거래 {n_tr}회, 테스트 샤프 {f2(sys_te.샤프)}. 학습·검증 샤프 순위상관 {f2(S['m2_rank_corr_train_valid'])}", "안 맞음"],
    ["변동성 돌파 + Bailout 청산", "깔때기 없이 방아쇠만 매일",
     f"전체 알파 t {f2(trig_only.알파t)}, 테스트 알파 t {f2(trig_only_te.알파t)}", "일부 맞음(최근 약화)"],
    ["부채 적고 싸고 세금 내는 회사를, 추세 확인 후 산다", "통상·분포·업종·CRSP 6개 × 진입 2개",
     f"고른 조합({q_name}): 동일가중 대비 전체 알파 t {f2(q_vs_ew.알파t)}, 코스피 대비 테스트 CAGR {pc(q_te.CAGR)} vs {pc(q_te.BM_CAGR)}. "
     f"동일가중 대비 우위는 {int(CONS['픽수'])}건 중 {int(CONS['필요종목수']) if CONS['필요종목수'] == CONS['필요종목수'] else '-'}건에서 나왔고 나머지는 동일가중보다 나빴음",
     "안 맞음(소수 종목 의존)"],
]

# ───────── 표제
para("래리 윌리엄스 매매 원칙 검증: 시장 간 관계·단기 매매 시스템·우량주 필터", bold=True, size=16, color=NAVY, after=1)
para(f"백테스트 보고서{'' if FINAL else ' (초안)'} · 송호빈 · 데이터 기준일 {S['data']['kospi_end']} · 실제 매매 없음(포트폴리오용 검증)",
     size=9, color="595959", after=6)

h("1. 요약")
bullet("목적: 래리 윌리엄스가 책·강연에서 말한 원칙 11가지를 국내·미국 데이터로 하나씩 검증했다. ① 시장 간 관계(COT 상업자 지수, "
       "채권·유가 선행, 패닉 매수), ② 코스피 단기 매매 시스템(월말·월초, 선거, %R, Oops!·변동성 돌파, Bailout 청산, 위험% 포지션), "
       "③ 우량주 필터(부채·PER·PSR·세금 납부 + 추세 확인).")
bullet(f"검증 방식: 시간 순서로 학습 2000~2014 / 검증 2015~2019 / 테스트 2020-01~2026-11로 나누고, 파라미터는 미리 정한 후보 격자 "
       f"안에서 학습 구간 샤프로만 골랐다. 시도한 조합은 모두 {S['grid_total']}개. 테스트 구간은 마지막에 한 번만 계산했다. "
       "보조로 매년 그 이전 데이터만으로 다시 고르는 워크포워드를 2005년부터 돌렸다.")
bullet(f"가장 쓸 만했던 것: COT 상업자 지수(S&P500)는 수익률을 높이지는 못했지만 위험을 크게 줄였다. 워크포워드 샤프 "
       f"{f2(wf.loc['COT S&P500 → SPY','샤프'])}(보유 {f2(wf.loc['COT S&P500 → SPY','BM_샤프'])}), 최대 낙폭 "
       f"{pc(wf.loc['COT S&P500 → SPY','MDD'],0)}(보유 {pc(wf.loc['COT S&P500 → SPY','BM_MDD'],0)}). 코스피 월말·월초 효과는 "
       f"전체 기간 t {f2(tdm_k.loc['전체','NW_t'])}로 통계적으로 뚜렷했지만 검증 구간(2015~2019)에는 반대로 나타났다.")
bullet(f"맞지 않은 것: 유가 선행(학습에서 고른 시차 {oil_L}일은 래리가 말한 1.5~6개월 안이었지만 관계 방향이 반대, 워크포워드 알파 t "
       f"{f2(wf.loc['유가 → 코스피','알파t'])}), 채권 선행, 선거 주기, 금·원유 COT. 깔때기 시스템은 조건을 겹칠수록 거래가 사라져 "
       f"26년간 {n_tr}회만 거래했고, 학습에서 좋았던 조합이 검증에서 좋지 않았다(순위상관 {f2(S['m2_rank_corr_train_valid'])}).")
bullet(f"우량주 필터({q_name}): 동일가중 전 종목 대비 연 알파 {pc(q_vs_ew.알파)}(t {f2(q_vs_ew.알파t)})였지만, 종목별로 나눠 보면 "
       f"{int(CONS['픽수'])}건 중 {int(CONS['필요종목수'])}건만 빼도 우위가 사라지고 나머지의 동일가중 대비 승률은 {pc(CONS['나머지승률'],0)}였다. "
       f"테스트(2020~) CAGR {pc(q_te.CAGR)}로 코스피 {pc(q_te.BM_CAGR)}에 못 미쳤다. 필터보다는 소형주 몇 개의 급등에 기댄 결과로 판단한다.")
bullet("약점 보완(같은 규칙으로 재실행): ① 상장폐지 종목 재무를 OpenDART로 2,748건 채움 ② 법인세 비용·현금 납부액을 2015 회계연도부터 실제 값으로 "
       "③ 2018-07 이후 상장폐지 230종목의 일봉·거래대금·시가총액을 KRX에서 받아 넣음 ④ 리밸런싱 달 분산, 워크포워드, 블록 부트스트랩, "
       "검정력(필요 연수), 섹터 분해, 한 해씩 빼기 추가.")
cvr = FCV.set_index(["지표", "방식"])
bullet(f"전체 시장 검증(2000~2025, 종목×연도 {int(FINFO['종목연']):,}건): 급등 종목에서 보인 특징 7개를 시가총액 통제 후 5분위·Fama-MacBeth·"
       "Fama-French 2×3·3년 블록 교차검증으로 확인했다. 26년 동안 9개 블록 대부분에서 같은 방향이었던 것은 회전율(높으면 나쁨, "
       f"{int(cvr.loc[('회전율','Fama-MacBeth 다변량 계수'),'방향일치블록'])}/9), 변동성(높으면 나쁨, "
       f"{int(cvr.loc[('변동성60일','5분위 시총통제 Q5−Q1'),'방향일치블록'])}/9), 200일선 위(좋음, "
       f"{int(cvr.loc[('200일선대비','Fama-MacBeth 다변량 계수'),'방향일치블록'])}/9) 세 가지였다. ROE와 %R은 효과가 없었고, "
       "급등 종목의 특징(고회전·고변동·최근 급등)은 전체 시장에서는 평균적으로 지는 종목의 특징이었다.")
h3("래리 주장별 판정")
table(VERDICT, widths=[4.2, 3.6, 7.0, 2.4], size=7.5)

# ───────── 2. 규칙
h("2. 전략 규칙")
h3("공통")
bullet("구간: 학습 2000-01~2014-12, 검증 2015-01~2019-12, 테스트 2020-01~2026-11. ③ 우량주 필터는 국내 재무 자료가 2013 회계연도부터 "
       "있어서 학습 2014-05~2017-12, 검증 2018~2019, 테스트 2020~로 줄였다.")
bullet("신호는 그날 종가까지 알 수 있는 정보로 만들고 다음 날 수익률에 적용한다. 미국 자료는 한국 시각으로 하루 늦게 반영한다. "
       "COT는 기준일(화요일) 엿새 뒤(다음 주 월요일)부터 쓴다. 연간 재무는 다음 해 4월 말부터 쓴다.")
bullet("거래비용: 편도 수수료 0.015% + 슬리피지 0.05%. 비용 2배도 따로 계산. 현금은 단기금리(한국 3개월 금리, 미국 3개월 국채)를 받는다.")
bullet("유의성: 겹치는 표본(13주·60일 이후 수익률 등)은 Newey-West t값. 알파는 초과수익을 벤치마크 초과수익에 회귀한 절편(연율).")
h3("① 시장 간 관계")
table([["항목", "규칙", "후보 격자"],
       ["COT 상업자 지수", "(상업자 순포지션 − N주 최저)/(N주 최고 − N주 최저)×100. 상단 이상이면 매수, 하단 이하면 현금, 그 사이는 유지",
        "N 26·156주 × 극단 20/80·25/75, 시장별(S&P500·30년 국채·금·WTI) 따로"],
       ["채권 → 주식", "채권 가격(TLT, 상장 전 30년 국채 선물)이 n일 평균 위일 때만 주식 보유", "n 10·20·50·200일, SPY·코스피 따로"],
       ["유가 → 코스피", "L일 전 유가의 m일 수익률 > 0이면 코스피 보유", "L 20~130일(10일 간격) × m 20·60일"],
       ["패닉 매수", "코스피 종가가 20일 고점 대비 x 이상 하락하면 매수, h일 보유", "x 10·15·20% × h 60·120일"]],
      widths=[2.8, 9.0, 5.4], size=7.5)
h3("② 코스피 단기 매매 시스템 (깔때기, 조합은 하나로 고정)")
table([["단계", "규칙", "후보"],
       ["조건", "①에서 고른 COT(S&P500) 매수 상태 그리고 채권 필터(코스피) 켜짐", "①의 선택값"],
       ["시기", "월말 마지막 거래일 + 다음 달 첫 D거래일(TDM)", "D 3·4·5"],
       ["상태", "Williams %R(10일)이 −80 아래에서 위로 올라온 날이 최근 5거래일 안에 있음", "고정"],
       ["방아쇠", "Oops!(시가 < 전일 저가 → 전일 저가 회복 시 매수) 또는 변동성 돌파(시가 + k×전일 변동폭)", "Oops!, k 0.4·0.5·0.6·0.8·1.0"],
       ["청산", "Bailout(처음 이익 난 시가에 청산) · 시간 청산 · 손절(진입가 − 전일 변동폭×s)", "시간 3·5·10일 × s 1·2"],
       ["포지션", "투자 비중 = 위험% ÷ (손절폭/진입가), 최대 100%. 학습 MDD 30% 이내에서 샤프 최대", "위험 1·2·3%"]],
      widths=[1.8, 11.0, 4.4], size=7.5)
h3("③ 우량주 필터 (국내 전 종목, 상장폐지 포함)")
dt = S["dist_thresholds"]
table([["시나리오", "부채비율", "PER", "PSR"],
       ["통상 기준", "≤ 100%", "0 < PER ≤ 10", "≤ 1.0"],
       ["분포 30% (타이트)", pc(dt["0.3"]["de"], 0), f"0 < PER ≤ {dt['0.3']['per']:.1f}", f"≤ {dt['0.3']['psr']:.2f}"],
       ["분포 50% (중간)", pc(dt["0.5"]["de"], 0), f"0 < PER ≤ {dt['0.5']['per']:.1f}", f"≤ {dt['0.5']['psr']:.2f}"],
       ["분포 70% (넓게)", pc(dt["0.7"]["de"], 0), f"0 < PER ≤ {dt['0.7']['per']:.1f}", f"≤ {dt['0.7']['psr']:.2f}"],
       ["업종 상대", "같은 업종 하위 30%", "같은 업종 하위 30%(양수)", "같은 업종 하위 30%"],
       ["CRSP 가치 점수", "장부가/가격·이익/가격·매출/가격 순위 평균 상위 30%", "", ""]],
      widths=[3.6, 4.6, 4.6, 4.4], size=7.5)
bullet("분포 기준선은 학습 구간(2014~2017 리밸런싱) 비금융 종목의 실제 분포에서 정했다. 모든 시나리오에 세금 납부 조건을 붙였다: "
       "2023 회계연도부터는 법인세비용 > 0, 현금흐름표 법인세 납부 > 0, 유효세율 5~40%. 그 이전은 세금 계정이 없어 "
       "순이익·영업이익·영업현금흐름이 모두 양수인 것으로 대신했다.")
bullet("금융업(은행·증권·보험) 제외, 시가총액 하위 20% 제외. 통과 종목이 30개를 넘으면 가치 점수 순으로 30개. 동일가중, 매년 4월 말 리밸런싱.")
bullet("진입: 필터 통과 종목을 바로 사지 않고, 종가 > 200일 평균이 되거나 %R이 −80 위로 올라온 날 매수(래리의 '조건부 매매'). "
       "신호가 없는 종목 몫은 현금. 다음 리밸런싱 날 전부 정리.")

# ───────── 3. 데이터
h("3. 데이터")
ss = S["sector_src_counts"]
table([["자료", "출처", "기간·비고"],
       ["코스피 일봉(시가·고가·저가·종가)", "야후 ^KS11", f"{S['data']['kospi_start']}~{S['data']['kospi_end']}. 코스피200 일봉은 KRX 외 출처에 2014년 이전 자료가 없어 코스피로 대체"],
       ["KODEX 200(069500)", "네이버", "2014-08~. 실제 거래 가능한 상품으로 같은 규칙 재확인"],
       ["국내 개별 종목", "stock_db(kor_price·kor_ohlcv_raw·kor_price_delisted) + KRX 상장폐지 일봉", "1999~2026-10, 상장폐지 포함, 수정주가. 2018-07 이후 상장폐지 230종목은 KRX에서 보강"],
       ["국내 재무", "stock_db dart_fs + OpenDART 다중회사 주요계정, 전체 재무제표(법인세 납부)", "2013~2025 회계연도. 법인세 비용·납부 2015~"],
       ["EPS·BPS·PBR", "KRX(pykrx), 매년 4월 말 27회", "2000~2026, 전체 시장 검증의 ROE·PBR"],
       ["업종", "한국거래소 업종 분류(2026-10-05, 코스피), 없으면 kor_sector", f"거래소 업종 {ss.get('KRX업종',0)}종목, kor_sector 보완 {ss.get('kor_sector',0)}종목, 업종 없음 {ss.get('없음',0)}종목"],
       ["미국·원자재", "야후(SPY·TLT·30년 국채·금·WTI 선물·USO)", "1999~2026-11"],
       ["COT", "CFTC 공식 API(Legacy, 선물만)", "S&P500 E-mini·30년 국채·금·WTI 주간"],
       ["단기금리", "FRED(DTB3, 한국 3개월 금리)", "현금 수익률"]],
      widths=[4.2, 5.3, 7.7], size=7.5)
bullet(f"가격 단절 보정: 국내 패널은 기존 공용 코드(stitch_history.py)로 액면분할·감자 등 7,302건을 이미 보정한 상태다. 여기에 더해 하루 수익률이 "
       f"가격제한폭(2015-06 전 ±15%, 이후 ±30%)을 넘는 {S['panel_breaks']:,}건(신규 상장 첫날·권리락 등)을 단절로 보고 그날 수익률을 0으로 두었다. "
       f"WTI 선물은 2020-04 마이너스 가격 {S['data']['fixes_us']['OIL_neg_price_days']}일을 USO 수익률로 대체했다.")
bullet("상장폐지 보강: 기존 DB는 2018-07 이후 상장폐지 종목의 종가만 있어 거래대금·고가·저가가 없었다. KRX 상장폐지 목록(2015~) "
       "1,676건 중 코스피·코스닥 보통주(스팩 제외) 230종목의 일봉을 받아, KRX 등락률로 분할·감자를 이은 수정가격을 기존 가격과 이어 붙였다. "
       "정리매매(가격제한 없음) 폭락은 실제 손실이라 단절 처리하지 않았다.")
bullet("재무 보강: 나중에 상장폐지된 종목 중 재무가 있는 비율이 약 63%에서 약 72%로 올랐다(전체 종목은 약 90%). 남은 종목은 DART에 사업보고서가 "
       "없는 경우(2015년 이전 상장폐지 등)라 더 채울 수 없었다.")
bullet("시가총액: 2018-07까지는 원본 시가총액, 상장폐지 종목은 상장폐지 시세표의 시가총액, 그 밖은 2018-07과 최근 두 시점의 주식 수를 "
       "날짜 비율로 이어 수정종가에 곱해 추정했다. 보강한 상장폐지 종목은 KRX 실제 시가총액을 썼다.")
bullet("코스피 시가: 야후 코스피 시가가 전일 종가와 같은(시가 미기록) 날은 연도별 최대 "
       f"{pc(S['data']['kospi_open_eq_prevclose_max'])}로 시가 기반 방아쇠 계산에 문제가 없었다.")

# ───────── 4. 결과
h("4. 결과")
h3("① COT 상업자 지수 (학습 샤프로 시장별 선택)")
rows = [HDR]
for key, nm in [("SPX", "S&P500"), ("BOND", "미국 30년 국채"), ("GOLD", "금"), ("OIL", "원유(WTI)")]:
    n, lo, hi = cc[key]
    rows += seg_rows(COT, f"COT {nm} N={n} {lo}/{hi}", f"{nm} N={n} {lo}/{hi}", ("학습", "검증", "테스트"))
table(rows, widths=[3.6, 1.2, 1.5, 1.2, 1.5, 1.6, 1.4, 1.6, 1.2], size=7.5)
para("BM은 각 자산을 계속 보유한 경우. 금·원유는 선물 수익률(현금 이자 제외)이라 현금 0으로 계산.", size=8, color="595959")
pic("fig3_cot_spx.png", 16)
h3("① 채권·유가·패닉 → 주식")
rows = [HDR]
rows += seg_rows(BOND, f"채권>{S['bond_choice']['SPY']}일선 → SPY", f"채권 > {S['bond_choice']['SPY']}일선 → SPY", ("학습", "검증", "테스트"))
rows += seg_rows(BOND, f"채권>{S['bond_choice']['코스피']}일선 → 코스피", f"채권 > {S['bond_choice']['코스피']}일선 → 코스피", ("학습", "검증", "테스트"))
rows += seg_rows(OIL, "유가 시차", f"유가 시차 {oil_L}일·{oil_m}일 추세", ("학습", "검증", "테스트"))
rows += seg_rows(PAN, "패닉 낙폭", f"패닉 {int(pan_d*100)}%·{pan_h}일", ("학습", "검증", "테스트"))
table(rows, widths=[3.6, 1.2, 1.5, 1.2, 1.5, 1.6, 1.4, 1.6, 1.2], size=7.5)
para("패닉 검증 구간(2015~2019)은 20일 안에 15% 넘게 떨어진 날이 없어 계속 현금이었다.", size=8, color="595959")
pic("fig1_macro.png", 16.5)
h3("유가 시차 상관과 업종 확장")
lt = LAG.pivot_table(index="시차", columns=["m일", "구간"], values="상관")
rows = [["시차(거래일)", "20일·학습", "20일·검증", "20일·테스트", "60일·학습", "60일·검증", "60일·테스트"]]
for L in [0, 20, 30, 50, 60, 80, 100, 130]:
    rows.append([L] + [f2(lt.loc[L, (m, s)]) for m in (20, 60) for s in ("학습", "검증", "테스트")])
table(rows, size=7.5)
para(f"값은 'L일 전 유가의 m일 수익률'과 '코스피 이후 20일 수익률'의 상관. 학습 구간 시차 30일 이상에서 {pc(neg_share,0)}가 음수로, "
     "유가가 오른 뒤 1.5~6개월 뒤 코스피는 오히려 약했다. 래리 주장(유가 상승 → 주가 상승)과 방향이 반대다.", size=8.5)
sec_t = SEC[SEC.시차.astype(str).str.startswith("전략")]
rows = [["업종(동일가중)", "종목 수", "학습 전략 샤프", "학습 보유 샤프", "테스트 전략 샤프", "테스트 보유 샤프"]]
for g, gg in sec_t.groupby("업종", sort=False):
    a, b = gg[gg.구간 == "학습"].iloc[0], gg[gg.구간 == "테스트"].iloc[0]
    rows.append([g, int(a.종목수), f2(a.전략샤프), f2(a.보유샤프), f2(b.전략샤프), f2(b.보유샤프)])
table(rows, size=7.5)
para("업종 확장은 탐색용이다. 원가·매출이 유가에 묶인 업종에서도 코스피에서 고른 유가 신호는 보유보다 나빴다. "
     "업종 목록은 현재 상장 종목 기준이라 생존편향이 있다.", size=8, color="595959")
pic("fig2_oil_lag.png", 14)

h3("② 월말·월초(TDM)와 선거")
rows = [["시장", "월초 일수", "구간", "TDM 일평균", "나머지 일평균", "차이 t"]]
for _, r in TDM[TDM.구간.isin(["학습", "검증", "테스트", "전체"]) & (TDM.월초일수.isin([3, 4]))].iterrows():
    rows.append([r.시장, r.월초일수, r.구간, pc(r.TDM일평균, 2), pc(r.나머지일평균, 2), f2(r.NW_t)])
table(rows, size=7.5)
pic("fig6_tdm.png", 12)
rows = [["시장", "선거", "횟수", "±1개월 일평균", "나머지 일평균", "연율 차이", "t"]]
for _, r in ELE.iterrows():
    rows.append([r.시장, r.선거, r.횟수, pc(r.창일평균, 3), pc(r.나머지일평균, 3), pc(r.연율차이), f2(r.NW_t)])
table(rows, size=7.5)

h3("② 깔때기 단기 시스템")
fs_ = S["sys_filter_share"]
trig_txt = "Oops!" if sc["방아쇠"] == "oops" else f"변동성 돌파 k={sc['k']}"
para(f"선택된 조합: TDM 첫 {sc['TDM월초']}일, {trig_txt}, 시간 청산 {sc['시간청산']}일, "
     f"손절 전일 변동폭×{sc['손절배수']}, 위험 {int(sc['위험']*100)}%. 진입 허용 날의 비율: 조건 {pc(fs_['조건'],0)} × 시기 {pc(fs_['TDM'],0)} × "
     f"상태 {pc(fs_['상태'],0)} → 전부 겹치는 날 {pc(fs_['전부'],1)}.", size=9)
rows = [HDR]
rows += seg_rows(SYS, sys_name, "깔때기(전부)")
rows += seg_rows(SYS, sys_name + " [비용 2배]", "비용 2배", ("학습", "검증", "테스트"))
for nm in ["조건(COT·채권) 뺌", "시기(TDM) 뺌", "상태(%R) 뺌", "방아쇠만(매일)"]:
    rows += seg_rows(SYS, f"[뺌] {nm}", nm, ("학습", "검증", "테스트"))
rows += seg_rows(SYS, "KODEX 200", "KODEX 200 같은 규칙", ("검증", "테스트"))
table(rows, widths=[3.6, 1.2, 1.5, 1.2, 1.5, 1.6, 1.4, 1.6, 1.2], size=7.5)
ts = SYS[SYS.전략 == "거래 통계"].set_index("구간")
para("거래 통계: " + ", ".join(f"{k} {int(ts.loc[k,'거래수'])}회(승률 {pc(ts.loc[k,'승률'],0)}, 평균 {pc(ts.loc[k,'평균거래수익'],2)})"
                               for k in ["학습", "검증", "테스트"]) + ". 거래가 연 1회 안팎이라 대부분의 날이 현금이고, "
     "그래서 변동성과 최대 낙폭이 아주 작게 나온다.", size=8.5)
pic("fig4_system.png", 16.5)

h3("③ 우량주 필터")
rows = [HDR]
rows += seg_rows(Q, q_name, f"{q_name} (BM 코스피)")
rows += seg_rows(Q, f"{q_name} vs 동일가중", "같은 전략 (BM 동일가중)")
rows += seg_rows(Q, "벤치마크: 동일가중", "동일가중 전 종목 (BM 코스피)")
rows += seg_rows(Q, f"{q_name} [비용 2배]", "비용 2배", ("테스트", "전체"))
rows += seg_rows(Q, f"{q_name} [코스피", "코스피 확인 종목만", ("학습", "검증", "테스트"))
table(rows, widths=[3.6, 1.2, 1.5, 1.2, 1.5, 1.6, 1.4, 1.6, 1.2], size=7.5)
cand = Q[Q.전략.astype(str).str.startswith("[후보]")]
ref = Q[Q.전략.astype(str).str.startswith("[참고]")].set_index("전략")
rows = [["시나리오", "200일선 진입 학습 샤프", "%R 진입 학습 샤프", "즉시 진입 전체 CAGR", "평균 종목 수"]]
for scn in ["통상", "분포30", "분포50", "분포70", "업종상대", "CRSP가치"]:
    a = cand[cand.전략 == f"[후보] {scn}·200일선 위"].iloc[0]
    b = cand[cand.전략 == f"[후보] {scn}·%R 과매도 탈출"].iloc[0]
    c = ref.loc[f"[참고] {scn}·즉시 진입"]
    rows.append([scn, f2(a.샤프), f2(b.샤프), pc(c.CAGR), f"{a.평균종목수:.1f}"])
table(rows, size=7.5)
pic("fig5_quality.png", 16.5)
rows = [["연도"] + list(QY.columns)]
for y, r in QY.iterrows():
    rows.append([y] + [pc(v) for v in r.values])
table(rows, size=7.5)
para("2014년은 4월 말 이후, 2026년은 10월까지.", size=8, color="595959")
if len(QP):
    rows = [["종목", "업종", "부채비율", "PER", "PSR", "시가총액(억원)"]]
    for _, r in QP.iterrows():
        rows.append([f"{r['name']}" if isinstance(r["name"], str) else str(r["code"]).zfill(6), r.sector, pc(r.de, 0), f"{r.per:.1f}",
                     f"{r.psr:.2f}", f"{r.mcap:,.0f}"])
    para("2026-04 리밸런싱에서 필터를 통과한 종목(이후 200일선 위일 때만 보유):", size=8.5)
    table(rows, size=7.5)

# ───────── 5. 강건성
h("5. 강건성 검증")
h3("연 단위 워크포워드 (2005~2026, 매년 그 전 데이터만으로 다시 선택)")
rows = [["전략", "CAGR", "샤프", "MDD", "BM CAGR", "BM 샤프", "BM MDD", "알파 t"]]
for nm, r in wf.iterrows():
    rows.append([nm, pc(r.CAGR), f2(r.샤프), pc(r.MDD), pc(r.BM_CAGR), f2(r.BM_샤프), pc(r.BM_MDD), f2(r.알파t)])
table(rows, size=7.5)
pic("fig7_walkforward.png", 16)
h3("스스로 던진 질문과 답")
nb = NB.copy()
nb1 = nb[nb.다른칸수 == 1]
bullet(f"파라미터를 한 칸 바꾸면? 깔때기 시스템에서 한 칸만 다른 {len(nb1)}개 조합의 테스트 샤프는 {f2(nb1.테스트샤프.min())}~"
       f"{f2(nb1.테스트샤프.max())}(중앙값 {f2(nb1.테스트샤프.median())}). 학습 샤프 1위가 우연히 고른 값이라는 뜻이다.")
bullet(f"후보를 많이 보면 생기는 착시는? 깔때기 {S['grid_counts']['② 단기 시스템']}개 조합의 학습 샤프와 검증 샤프 순위상관이 "
       f"{f2(S['m2_rank_corr_train_valid'])}로, 학습 성적이 검증 성적을 전혀 예측하지 못했다.")
bullet(f"거래비용 2배면? 깔때기는 테스트 샤프 {f2(row(SYS, sys_name + ' [비용 2배]', '테스트').샤프)}, 우량주 필터는 테스트 CAGR "
       f"{pc(row(Q, q_name + ' [비용 2배]', '테스트').CAGR)}로 결론이 바뀌지 않는다(거래가 적어 비용 영향이 작다).")
bullet(f"결과가 좋은 곳(우량주 학습 CAGR {pc(q_tr.CAGR)})은 미래정보·생존편향부터 의심했다: 재무는 4월 말 이후만, 상장폐지 종목 포함, "
       "시가총액 하위 20% 제외, 상장폐지 종목 재무·시세 보강. 보강 뒤에도 나중에 상장폐지된 종목의 재무 보유율이 약 72%라서 "
       "부실 종목 일부가 빠져 있다.")
bullet(f"코스피 대신 실제 상품(KODEX 200)이면? 2015년 이후 거래 수와 결론이 같다(테스트 샤프 {f2(row(SYS,'KODEX 200','테스트').샤프)}).")
bullet(f"코스피 상장 확인 종목만 쓰면? 평균 종목 수가 크게 줄어 학습 CAGR {pc(row(Q, q_name+' [코스피','학습').CAGR)}. 상장폐지 종목의 "
       "시장 구분을 알 수 없어 본 결과는 전 시장(코스피+코스닥) 기준을 기본으로 했다.")

exec((HERE / "report_sections_extra.py").read_text(encoding="utf-8"))

doc.save(OUT)
print(OUT)
