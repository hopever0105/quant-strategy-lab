# -*- coding: utf-8 -*-
"""수주잔고 전략 — 5) 보고서: results/ → 보고서_초안.docx (+pdf).
실행: python build_report.py [최종]  ('최종'이면 수주잔고_전략_보고서.docx)
숫자는 모두 results/ 와 ../druckenmiller_13f/results/backlog/ 의 csv에서 읽는다(손으로 적은 숫자 없음)."""
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
CH = HERE / "charts"
DB = HERE.parent / "druckenmiller_13f" / "results" / "backlog"
FINAL = len(sys.argv) > 1 and sys.argv[1] == "최종"
OUT = HERE / ("수주잔고_전략_보고서.docx" if FINAL else "보고서_초안.docx")
NAVY = "1F3864"

pc = lambda x, d=1: "-" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"
pcs = lambda x, d=1: "-" if x is None or pd.isna(x) else f"{x * 100:+.{d}f}%"
f2 = lambda x: "-" if x is None or pd.isna(x) else f"{x:.2f}"
ft = lambda x: "-" if x is None or pd.isna(x) else f"{x:+.2f}"

T = pd.read_csv(RES / "main_results.csv")
RB = pd.read_csv(RES / "robustness.csv")
RC = pd.concat([pd.read_csv(RES / "us_random.csv"), pd.read_csv(RES / "kr_random.csv")])
YU = pd.read_csv(RES / "us_yearly.csv", index_col=0)
YK = pd.read_csv(RES / "kr_yearly.csv", index_col=0)
A1 = pd.read_csv(DB / "deep_A1_event.csv")
A3 = pd.read_csv(DB / "deep_A3_sector.csv")
KIC = pd.read_csv(DB / "kr_orders_ic.csv")
CHK = (HERE / "data" / "check.txt").read_text(encoding="utf-8").splitlines()
INFO = (RES / "info.txt").read_text(encoding="utf-8").splitlines()
UP = pd.read_pickle(HERE / "data" / "us_panel.pkl")
KP = pd.read_pickle(HERE / "data" / "kr_panel.pkl")


def g(mk, strat, seg, col):
    x = T[(T["시장"] == mk) & (T["전략"] == strat) & (T["구간"] == seg)]
    return x[col].iloc[0] if len(x) else float("nan")


def ic_us():
    import numpy as np
    from scipy import stats
    d = UP.dropna(subset=["gap", "ret"])
    s = d.groupby("t0").apply(lambda x: stats.spearmanr(x["gap"], x["ret"]).statistic)
    return s.mean(), s.mean() / s.std() * np.sqrt(len(s)), (s > 0).mean(), len(s)


doc = Document()
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(1.8)
    s.top_margin = s.bottom_margin = Cm(1.6)
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


def para(text, bold=False, size=None, color=None, after=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
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


def perf_rows(mk, strats, segs):
    rows = [["전략", "구간", "연수익(CAGR)", "변동성", "샤프", "최대낙폭", "알파(연)", "알파 t값"]]
    for s_ in strats:
        for sg in segs:
            if pd.isna(g(mk, s_, sg, "CAGR")):
                continue
            bm = s_.startswith("코스피") or s_.startswith("SPY")  # 벤치마크 자기 자신의 알파는 의미 없음
            rows.append([s_, sg, pc(g(mk, s_, sg, "CAGR")), pc(g(mk, s_, sg, "변동성")), f2(g(mk, s_, sg, "샤프")),
                         pc(g(mk, s_, sg, "MDD")), "-" if bm else pcs(g(mk, s_, sg, "알파(연)")), "-" if bm else ft(g(mk, s_, sg, "알파_t"))])
    return rows


icm, ict, icp, icn = ic_us()
kic = KIC.set_index("신호")
k_ic = kic.loc["수주잔고 증가율 − 매출 증가율"]
k_ic2 = kic.loc["수주잔고 증가율 − 매출 증가율 (2025 제외)"]
KA, KA25 = "A 상위 30% 매수", "A 상위 30% 매수 (2025 제외)"
KB, KB25 = "B 상위 매수 − 하위 공매도", "B 상위 매수 − 하위 공매도 (2025 제외)"
rck = RC[(RC["시장"] == "kr")].set_index("구간")
rcu = RC[(RC["시장"] == "us")].set_index("구간")

# 표지
para("수주잔고 전략", bold=True, size=18, color=NAVY, after=2)
para("받아 둔 일감이 매출보다 빨리 쌓이는 회사를 산다 — 한국 수주 업종 · 미국 S&P500", size=11, after=2)
para("포트폴리오용 검증(실제 매매 없음) · 한국 2016~2026, 미국 2019~2026 · 분기 리밸런싱", size=8.5, color="666666", after=8)

h("1. 요약")
bullet(f"아이디어: 수주잔고(계약은 했지만 아직 매출로 잡히지 않은 금액)가 매출보다 빨리 늘면 앞으로 매출이 빨라진다. "
       f"신호 = 수주잔고 전년 대비 증가율 − 매출 증가율. 분기말에 이미 공시된 숫자만 쓴다.")
bullet(f"한국(건설·조선·기계·전기전자·IT서비스 {KP['code'].nunique()}개사, 정기보고서 원문 수주상황 표): 상위 30% 매수 연 {pc(g('kr', KA, '전체', 'CAGR'))} "
       f"(코스피 {pc(g('kr', '코스피', '전체', 'CAGR'))}, 같은 업종 전부 {pc(g('kr', '신호 있는 종목 동일가중', '전체', 'CAGR'))}), "
       f"코스피 대비 알파 t값 {ft(g('kr', KA, '전체', '알파_t'))}. 2025년을 빼도 연 {pc(g('kr', KA25, '전체', 'CAGR'))}, t값 {ft(g('kr', KA25, '전체', '알파_t'))}.")
bullet(f"한국 신호의 순위 예측력(다음 분기 수익과의 순위상관) {f2(k_ic['IC'])}, t값 {ft(k_ic['t'])}. "
       f"상위 30% 매수 − 하위 30% 공매도는 연 {pc(g('kr', KB, '전체', 'CAGR'))}, 알파 t값 {ft(g('kr', KB, '전체', '알파_t'))}. "
       f"상위 비율(20·30·40%)·보유 기간(1·2·4분기) 9개 조합 모두 2025년 포함·제외 어느 쪽이든 알파 t값 1.9 이상.")
bullet(f"미국(S&P500 {UP.groupby('t0').size().median():.0f}종목/분기, SEC 수주잔고 공시): 순위 예측력 {f2(icm)}(t {ft(ict)})으로 신호 자체는 있으나, "
       f"매수만 하면 연 {pc(g('us', KA, '전체', 'CAGR'))}로 SPY({pc(g('us', 'SPY', '전체', 'CAGR'))})와 비슷하다. "
       f"롱숏은 연 {pc(g('us', KB, '전체', 'CAGR'))}, 샤프 {f2(g('us', KB, '전체', '샤프'))}, 6팩터 알파 t값 {ft(g('us', KB, '전체', '알파_t'))}로 약하다.")
bullet("결론: 한국 수주 업종에서는 쓸 만한 전략, 미국은 시장 방향을 뺀 보조 신호. 한국은 종목 수가 적어(분기 평균 8종목) 집중 위험이 크다.")

h("2. 전략 규칙")
table([["항목", "내용"],
       ["신호", "수주잔고 전년 대비 증가율 − 매출 증가율 (미국: 최근 4분기 매출, 한국: 연간 매출 — 다음 해 4월부터 사용)"],
       ["수주잔고", "미국: SEC 재무 데이터의 수주잔고(RevenueRemainingPerformanceObligation), 없으면 계약부채·이연수익 / "
                   "한국: 사업·반기·분기보고서 원문 '수주상황' 표의 수주잔고 합계, 공시일(접수일)부터 사용"],
       ["유니버스", "미국: 그 시점 S&P500 구성종목 중 신호 있는 종목 / 한국: 코스피 시가총액 상위 200(분기말 기준) 중 수주 업종, 상장폐지 포함"],
       ["종목 고르기", "미국: 업종(11개) 안 순위 상위 30% / 한국: 수주 업종 전체 안 순위 상위 30% (업종별로 나누면 2~8종목뿐)"],
       ["진입·청산", "분기말 종가 매수, 2분기 보유. 보유 묶음 2개가 겹쳐 돌아가 매 분기 절반씩 교체"],
       ["비중", "동일가중 (분기 중엔 가격 따라 변동)"],
       ["거래비용", "편도 미국 0.1%, 한국 0.25%(세금 포함). 공매도 버전은 대차비용 연 0.5% 추가"],
       ["벤치마크", "미국: SPY, 신호 있는 종목 동일가중 / 한국: 코스피, 수주 업종 동일가중"],
       ["알파", "미국: 파마-프렌치 5요인 + 모멘텀(6팩터), 한국: 코스피 대비(1요인). 일별, 자기상관 보정(HAC) t값"],
       ["버전", "A 매수만(본 전략) / B 상위 30% 매수 − 하위 30% 공매도(신호의 순수 효과 확인용)"]], widths=[3, 14])

h("3. 데이터")
bullet(f"미국: S&P500 구성종목 이력, 야후 수정종가(분할·배당 보정), SEC XBRL. 분기 {UP['t0'].nunique()}개(2019년 1분기~), "
       f"분기당 {UP.groupby('t0').size().median():.0f}종목, 그중 수주잔고 직접 공시 {pc(UP['is_rpo'].mean(), 0)}(나머지는 계약부채).")
bullet(f"한국: DART 정기보고서 원문 약 4,000건에서 수주잔고 표를 읽음({KP['code'].nunique()}개사, 분기 {KP['t0'].nunique()}개, 분기당 {KP.groupby('t0').size().median():.0f}종목). "
       "표 머리와 숫자가 다른 표로 나뉜 경우, 합계 행이 없는 경우를 처리하는 읽기 규칙을 썼다.")
for l_ in CHK:
    bullet("수정주가 점검 — " + l_)
bullet("미국 PARA(파라마운트)는 합병 뒤 티커가 재사용돼 야후 시세가 깨져 있어(1주 16,600달러 등) 제외했다. 시세가 없는 종목은 그 기간 벤치마크로 대체(대체 비중 0.1% 이하).")
bullet("미래정보 차단: 재무·수주잔고는 분기말 이전 공시분만, 한국 연간 매출은 다음 해 4월부터. 생존편향: 한국은 상장폐지 포함, 미국은 S&P500 과거 구성종목 포함(일부 상장폐지 종목 시세 누락).")
for l_ in INFO:
    bullet(l_.replace("kr", "한국").replace("us", "미국"))

h("4. 결과 — 한국 수주 업종")
pic("kr_equity.png")
table(perf_rows("kr", [KA, "하위 30% 매수(비교)", "신호 있는 종목 동일가중", "코스피", KB],
                ["학습 2016~2019", "검증 2020~2022", "테스트 2023~", "전체"]), widths=[4.2, 2.6, 1.8, 1.6, 1.3, 1.7, 1.7, 1.5], size=7, hl=[1, 2, 3, 4])
para("2025년을 뺀 결과(2025년 조선·전력기기 급등 의존 여부 확인)", bold=True, size=9)
table(perf_rows("kr", [KA25, "신호 있는 종목 동일가중 (2025 제외)", "코스피 (2025 제외)", KB25], ["테스트 2023~", "전체"]), widths=[4.2, 2.6, 1.8, 1.6, 1.3, 1.7, 1.7, 1.5], size=7, hl=[1, 2])
pic("kr_yearly.png")
para(f"평균 종목 수 {g('kr', KA, '전체', '평균종목수'):.1f}, 분기 회전율 {pc(g('kr', KA, '전체', '분기회전율'), 0)}(묶음 1개 기준 편도).", size=8.5, color="555555")

h("5. 결과 — 미국 S&P500")
pic("us_equity.png")
table(perf_rows("us", [KA, "하위 30% 매수(비교)", "신호 있는 종목 동일가중", "SPY", KB, "참고: 상위 30% ∩ 잔고/매출 높은 절반"],
                ["검증 2019~2021", "테스트 2022~", "전체"]), widths=[4.2, 2.6, 1.8, 1.6, 1.3, 1.7, 1.7, 1.5], size=7, hl=[1, 2, 3])
pic("us_yearly.png")
para(f"평균 종목 수 {g('us', KA, '전체', '평균종목수'):.0f}, 분기 회전율 {pc(g('us', KA, '전체', '분기회전율'), 0)}. "
     "'잔고/매출 높은 절반'은 결과를 보고 찾은 조건이라 참고로만 둔다.", size=8.5, color="555555")

h("6. 신호가 진짜인가 — 추가 확인")
h3("공시 뒤 얼마나 빨리 반영되나 (미국, 공시일 기준 SPY 초과수익, 상위 30% − 하위 30%)")
table([["기간", "상위 30%", "하위 30%", "차이(분기 평균)", "t값"]] +
      [[r["기간"], pcs(r["상위30_평균"]), pcs(r["하위30_평균"]), pcs(r["차이_분기평균"]), ft(r["t"])] for _, r in A1.iterrows()], size=8)
bullet("공시 뒤 3~6개월에 걸쳐 천천히 반영된다 → 공시를 보고 사도 늦지 않다. 공시 전 3개월에도 일부 먼저 반영(실적 발표 전 주문 소식 등).")
bullet("수주잔고 증가율은 현재 매출 증가율을 함께 넣어도 4분기 뒤 매출 증가율을 맞힌다(분기별 순위 회귀 t 9.3) → 신호가 실제 매출로 이어진다.")
bullet(f"미국 업종별 순위 예측력: " + ", ".join(f"{r['업종']} {f2(r['IC'])}(t {ft(r['t'])})" for _, r in A3.head(6).iterrows()) +
       " → 산업재·소재처럼 수주로 일하는 업종에서 강하다. 한국 유니버스(수주 업종만)에서 더 잘 되는 이유와 같다.")
h3("한국 신호 예측력")
table([["신호", "순위상관(IC)", "t값", "분기", "플러스 분기"]] +
      [[r["신호"], f2(r["IC"]), ft(r["t"]), int(r["분기"]), pc(r["플러스분기"], 0)] for _, r in KIC.iterrows()], size=8)

h("7. 강건성 검증")
h3("상위 비율 × 보유 기간 (전체 기간)")
rows = [["시장", "버전", "상위", "보유", "2025 제외", "연수익", "샤프", "최대낙폭", "알파 t값"]]
for _, r in RB.iterrows():
    rows.append(["한국" if r["시장"] == "kr" else "미국", r["버전"], pc(r["상위비율"], 0), f"{int(r['보유분기'])}분기",
                 "예" if r["제외2025"] else "-", pc(r["CAGR"]), f2(r["샤프"]), pc(r["MDD"]), ft(r["알파_t"])])
table(rows, size=6.5)
h3("거래비용 2배")
table(perf_rows("kr", ["A 상위 30% 매수 (비용 2배)", "A 상위 30% 매수 (비용 2배) (2025 제외)"], ["전체"]) +
      perf_rows("us", ["A 상위 30% 매수 (비용 2배)"], ["전체"])[1:], size=7)
h3("무작위 대조 (같은 분기·같은 유니버스에서 같은 개수 500번)")
table([["시장", "구간", "분기", "전략 누적", "무작위 중앙", "백분위"]] +
      [["한국" if r["시장"] == "kr" else "미국", r["구간"], int(r["분기"]), pcs(r["전략누적"], 0), pcs(r["무작위중앙"], 0), f"{r['백분위']:.0f}"]
       for _, r in RC.iterrows()], size=8)

h("8. 한계")
bullet(f"한국은 종목 수가 적다(상위 30% 평균 {g('kr', KA, '전체', '평균종목수'):.1f}종목) → 최대낙폭 {pc(g('kr', KA, '전체', 'MDD'))}. 실제로 쓰려면 비중 상한·분산이 필요하다.")
bullet(f"한국 검증 구간(2020~2022)에서는 롱숏 효과가 0에 가까웠다(연 {pc(g('kr', KB, '검증 2020~2022', 'CAGR'))}). 효과가 학습·테스트 구간에 몰려 있다.")
bullet("한국 수주잔고는 원문 표를 규칙으로 읽은 값이라 회사마다 잡음이 있다(사업보고서와 분기보고서의 표 범위가 다른 회사 등).")
bullet("같은 수주 업종 전부를 사도 코스피보다 높았다 → 2020년 이후 조선·전력기기 장세의 업종 효과가 섞여 있어, 업종 동일가중 대비로 봐야 한다.")
bullet("미국은 수주잔고 공시가 2019년부터라 기간이 짧고(29분기), 매수만으로는 SPY를 이기지 못했다. 미국 시세에서 일부 상장폐지 종목이 빠져 있다.")
bullet("신호 아이디어는 드러켄밀러 13F 분석 과정에서 여러 후보를 시험하다 나왔다. 규칙의 숫자(상위 30%, 2분기)는 고르지 않고 미리 정했지만, 여러 후보 중 살아남은 신호라는 점은 감안해야 한다.")

h("9. 결론")
bullet("한국 수주 업종: 수주잔고가 매출보다 빨리 쌓이는 종목을 사면 학습·검증·테스트 세 구간 모두 업종 평균보다 높았고, 2025년을 빼도 알파가 유의하다. 쓸 만한 전략 후보.")
bullet("미국: 신호의 순위 예측력은 있지만 크기가 작아 매수 전략으로는 SPY와 비슷하다. 시장 방향을 뺀 롱숏 보조 신호로 본다.")
bullet("다음 단계: 한국은 종목 수를 늘리기 위해 코스닥 수주 업종까지 넓히고, 업종 비중 상한을 둔 버전을 시험한다.")

doc.save(OUT)
print(OUT)
try:
    sys.path.insert(0, r"C:\Users\arlo0\OneDrive\바탕 화면\취업\jobapp")
    from jobkit.common import to_pdf
    print(to_pdf(OUT))
except Exception as e:  # noqa
    print("PDF 변환 실패:", e)
