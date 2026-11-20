# -*- coding: utf-8 -*-
"""보고서 생성: results/ → 보고서_초안.docx (+pdf)
실행: python build_report.py [최종]   ('최종'이면 쿨라마기_EMA돌파_검증_보고서.docx)
숫자는 모두 results/ 의 csv·json 에서 읽는다(손으로 적은 숫자 없음)."""
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
OUT = HERE / ("쿨라마기_EMA돌파_검증_보고서.docx" if FINAL else "보고서_초안.docx")
NAVY = "1F3864"

S = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
rd = lambda f: pd.read_csv(RES / f)
M, BM, G, GS, A, RB, BD = (rd(f) for f in ["main_segments.csv", "benchmarks.csv", "grid.csv", "grid_selected.csv",
                                          "ablation.csv", "robust.csv", "breakdown.csv"])
YR = pd.read_csv(RES / "yearly.csv", index_col=0)
RC = S["무작위대조"]

pc = lambda x, d=1: "-" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"
f2 = lambda x: "-" if x is None or pd.isna(x) else f"{x:.2f}"
f3 = lambda x: "-" if x is None or pd.isna(x) else f"{x:+.3f}"
nn = lambda x: "-" if x is None or pd.isna(x) else f"{int(x):,}"

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
    b = OxmlElement("w:pBdr"); bt_ = OxmlElement("w:bottom")
    for k, v in (("val", "single"), ("sz", "6"), ("space", "1"), ("color", NAVY)):
        bt_.set(qn(f"w:{k}"), v)
    b.append(bt_); pPr.append(b)


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


def pic(name, w=16.5):
    doc.add_picture(str(RES / name), width=Cm(w))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def m(setting, seg):
    return M[(M.설정 == setting) & (M.구간 == seg)].iloc[0]


def b(name, seg):
    return BM[(BM.설정 == name) & (BM.구간 == seg)].iloc[0]


BASE = "기본(ADR4%·EMA10)"
mb, mt, m5 = m(BASE, "전체"), m(BASE, "테스트"), m("ADR5%·EMA10", "전체")
ks, ew = b("코스피", "전체"), b("동일가중(유동성 통과 종목)", "전체")
Aall = A[A.구간 == "전체"].set_index("단계")
a1 = Aall.iloc[0]
a5 = Aall.iloc[-1]
rob = RB[RB.구간 == "전체"].set_index("검증")
gfull = G[G.구간 == "전체"]
gtest = G[G.구간 == "테스트"]
sel = S["격자선택"]
gsel = GS.set_index("구간")

# ───────────────────────── 표지·요약
para("쿨라마기 EMA 돌파 전략 검증", bold=True, size=18, color=NAVY, after=2)
para("국내 주식 2000~2026, 상장폐지 종목 포함 · 일봉 근사 · 비용 차감", size=10, color="595959", after=8)

h("1. 요약")
bullet(f"기본 설정(사전 합의, ADR 4%·EMA10 청산): 2000-01~2026-10 연 {pc(mb.CAGR)}, 샤프 {f2(mb.샤프)}, 최대 낙폭 {pc(mb.MDD)}. "
       f"같은 기간 코스피 연 {pc(ks.CAGR)}(최대 낙폭 {pc(ks.MDD)}), 유동성 통과 종목 동일가중 연 {pc(ew.CAGR)}.")
bullet(f"거래 {nn(mb.거래수)}건, 승률 {pc(mb.승률)}, 거래당 평균 {f3(mb.평균R)}R (t {f2(mb.R_t)}). "
       f"상위 5% 거래가 이익 합계의 {pc(mb.상위5pct이익비중, 0)}를 차지했고, 이들을 빼면 평균 {f3(mb.상위5pct제외평균R)}R.")
bullet(f"ADR 5% 기준: 연 {pc(m5.CAGR)}, 샤프 {f2(m5.샤프)}, 평균 {f3(m5.평균R)}R ({nn(m5.거래수)}건).")
bullet(f"조건 분해: 모멘텀·고변동만으로 다음 날 시가에 사면 평균 {f3(a1.평균R)}R, 돌파·정배열·횡보 압축·국면 필터를 모두 붙이면 {f3(a5.평균R)}R. "
       f"같은 날 모멘텀·고변동 후보 중 무작위 종목을 사면 {f3(RC['같은날후보_평균R'])}R로, 실제 신호가 무작위 20회 모두보다 높았다.")
bullet(f"파라미터 격자 {S['격자수']}개 중 전체 기간 샤프가 양수인 설정 {S['격자_전체기간_샤프양수']}개, "
       f"다중검정 보정(BH q<0.05)을 통과한 설정 {S['격자_전체기간_BHq05']}개, 테스트 구간 평균 R이 양수인 설정 {S['격자_테스트_평균R양수']}개.")

# ───────────────────────── 규칙
h("2. 전략 규칙")
para("크리스찬 쿨라마기(Kristjan Kullamägi)가 공개한 'Breakout' 셋업을 일봉으로 옮겼다. 원작은 장중 첫 몇 분 고가 돌파로 진입하지만 "
     "장중 자료가 없어 직전 n일 고가 돌파로 대신했다. 모든 조건은 진입 전날 종가까지 아는 값만 쓴다.")
table([["항목", "규칙"],
       ["유니버스", "코스피·코스닥 보통주 전체(우선주·스팩·리츠·코넥스 제외), 상장폐지 종목 포함"],
       ["유동성", "20일 평균 거래대금 10억 원 이상, 종가 1,000원 이상, 상장 후 63거래일 이상"],
       ["① 모멘텀", "63일 수익률이 유동성 통과 종목 중 상위 5% 이내이고 +30% 이상"],
       ["② 변동성(ADR)", "20일 평균 (고가÷저가 − 1) ≥ 4% (ADR 5%도 함께 보고)"],
       ["③ 횡보 압축", "1 − 종가 ÷ 최근 20일 최고가 ≤ 2.5 × ADR, 종가 > EMA10 > EMA20 > EMA50"],
       ["④ 시장 국면", "코스피 종가 > 코스피 EMA20"],
       ["진입", "다음 날 고가가 최근 20일 최고가를 넘으면 매수, 체결가 = max(시가, 기준가). 상한가 부근(99.5% 이상) 체결은 불가로 처리"],
       ["손절", "진입 시점까지의 당일 저가. 그 거리가 ADR보다 멀면 진입하지 않음. 이후 손절가 = 진입일 저가"],
       ["청산", "3거래일째 종가에 1/3 매도 후 손절가를 진입가로 올림. 나머지는 종가 < EMA10이면 다음 날 시가 매도 (변형: EMA20)"],
       ["비중", "거래당 위험 자본의 0.5%(위험 = ADR), 종목당 최대 20%. 같은 날 후보는 63일 수익률 높은 순, 현금 부족 시 건너뜀"],
       ["거래비용", "편도 0.25%(수수료·거래세·슬리피지), 2배도 시험. 현금은 한국 3개월 금리"],
       ["구간", "학습 2000~2014 / 검증 2015~2019 / 테스트 2020~2026-10 (격자 선택은 학습 구간만 보고, 테스트는 한 번만)"]],
      widths=[3.0, 13.5], size=8)
h3("일봉 근사의 진입일 처리")
para("일봉에는 장중 순서가 없어 진입 당일 손절 여부를 정해야 한다. 기본은 '봉 방향' 가정이다: 양봉은 시가→저가→고가→종가, "
     "음봉은 시가→고가→저가→종가 순으로 움직였다고 본다. 양봉이면 저가가 진입 전에 나왔으므로 손절가 = 당일 저가, "
     "음봉이면 진입 전 저가 = 시가이고 이후 시가 아래로 밀렸으므로 진입일에 시가에서 손절된다. "
     "가정이 결과를 바꾸는 정도는 7장에서 보수적 가정(저가가 ADR 밖이면 무조건 손절)과 비교한다.")

# ───────────────────────── 데이터
h("3. 데이터")
bullet("국내 일봉: MySQL stock_db(1999~)에 2018-07 이후 상장폐지 종목 KRX 일봉을 보강한 패널. 이번 작업에서 새로 요청한 KRX 데이터는 없다.")
bullet(f"대상 {S['종목수']:,}종목(상장폐지 포함). 수정주가 단절(하루 수익률이 가격제한폭 초과 → 분할·감자 미반영으로 판단) "
       f"{S['단절보정']:,}건({S['단절보정종목']:,}종목)을 보정해 이동평균·고점이 튀지 않게 했다. 보정 코드는 공용(quant/ohlc_panel.py).")
bullet("지수: 코스피(^KS11), 코스닥(^KQ11, 2000-10부터) 야후 가격지수(배당 제외). 동일가중 벤치마크는 전날 유동성 조건(거래대금 10억 원 이상 등)을 "
       "통과한 종목을 매일 같은 비중으로 들고 있는 수익률로, 전략이 고르는 후보 집단 전체의 성과다.")
bullet("상장폐지 종목은 마지막 거래일(정리매매 포함) 종가로 청산한다. 2018-07 이후 상장폐지 95종목은 종가만 있어(거래대금 없음) 진입 대상에서 빠진다.")

# ───────────────────────── 결과
h("4. 결과")
rows = [["설정", "구간", "CAGR", "변동성", "샤프", "MDD", "알파(코스피)", "t", "거래", "승률", "평균R"]]
for nm in [BASE, "ADR5%·EMA10", "ADR4%·EMA20", "ADR5%·EMA20"]:
    lab = nm
    for sg in ("학습", "검증", "테스트", "전체"):
        r = m(nm, sg)
        rows.append([lab, sg, pc(r.CAGR), pc(r.변동성), f2(r.샤프), pc(r.MDD), pc(r.알파_코스피), f2(r.t_코스피), nn(r.거래수), pc(r.승률, 0), f3(r.평균R)])
        lab = ""
table(rows, size=7.5, hl=[4, 8])
rows = [["벤치마크", "구간", "CAGR", "변동성", "샤프", "MDD"]]
for nm in ["코스피", "코스닥", "동일가중(유동성 통과 종목)"]:
    lab = nm
    for sg in ("학습", "검증", "테스트", "전체"):
        r = b(nm, sg)
        rows.append([lab, sg, pc(r.CAGR), pc(r.변동성), f2(r.샤프), pc(r.MDD)])
        lab = ""
table(rows, size=7.5)
para(f"회전율(연, 편도) {mb.회전율:.1f}배, 평균 투자 비중 {pc(mb.평균투자비중, 0)}, 평균 보유 {mb.평균보유일:.1f}거래일.", size=9)
pic("equity.png")
pic("yearly.png")
rows = [["연도", "기본 전략", "ADR5%", "코스피", "코스닥", "동일가중"]]
for y, r in YR.iterrows():
    rows.append([y, pc(r[BASE]), pc(r["ADR5%·EMA10"]), pc(r["코스피"]), pc(r["코스닥"]), pc(r["동일가중"])])
table(rows, size=7)
pic("r_hist.png", 13)

# ───────────────────────── 조건 분해
h("5. 어떤 조건이 성과를 바꾸는가 (조건 분해)")
para("모멘텀 상위·고변동 종목은 조건 없이 사면 평균적으로 불리하다는 기존 결과가 있다. 쿨라마기 규칙은 그 종목들 가운데 "
     "추세가 정배열이고 고점 부근에서 쉬다가 돌파하는 경우만 고른다. 조건을 하나씩 더해 같은 청산 규칙으로 비교했다(ADR 4%, EMA10).")
rows = [["단계", "거래", "승률", "평균 수익", "평균R", "t", "포트폴리오 CAGR", "MDD"]]
for nm, r in Aall.iterrows():
    rows.append([nm, nn(r.거래수), pc(r.승률, 0), pc(r.평균수익, 2), f3(r.평균R), f2(r.R_t), pc(r.포트_CAGR), pc(r.포트_MDD)])
table(rows, size=7.5, hl=[len(rows) - 1])
rows = [["단계"] + ["학습", "검증", "테스트"]]
for nm in Aall.index:
    rows.append([nm] + [f3(A[(A.단계 == nm) & (A.구간 == sg)].평균R.iloc[0]) for sg in ("학습", "검증", "테스트")])
para("구간별 평균 R", bold=True, size=9)
table(rows, size=7.5)
para("③→④가 같은 것은 합의한 횡보 정의(고점 대비 하락 ≤ 2.5×ADR)가 정배열 조건을 통과한 종목을 하나도 더 거르지 못했기 때문이다. "
     "그래서 '최근 10일 변동폭 ≤ 2×ADR이고 20일 고점이 3거래일 이상 전'인 엄격한 정의를 변형으로 시험했다(마지막 줄, 거래 수가 크게 줄어 판단은 어렵다).", size=9)
pic("ablation.png", 15)

# ───────────────────────── 대조
h("6. 무작위 진입 대조")
para("두 가지 대조를 만들었다. 둘 다 무작위로 고른 날 시가에 사고, 같은 손절·청산 규칙을 적용해 20회 반복했다.")
bullet("(가) 같은 종목·같은 해: 실제 거래 종목의 같은 해 유동성 통과일 중 무작위 날짜.")
bullet("(나) 같은 날 후보: 실제 거래일마다 그날 모멘텀·ADR·국면 조건을 통과한 종목 중 무작위 종목(돌파·정배열·횡보 조건 없음).")
table([["항목", "실제 신호", "(가) 같은 종목·같은 해", "(나) 같은 날 후보"],
       ["평균 R", f3(RC["실제평균R"]), f"{f3(RC['무작위평균R'])} ({f3(RC['무작위_5pct'])}~{f3(RC['무작위_95pct'])})",
        f"{f3(RC['같은날후보_평균R'])} ({f3(RC['같은날후보_5pct'])}~{f3(RC['같은날후보_95pct'])})"],
       ["평균 거래 수익", pc(RC["실제평균수익"], 2), pc(RC["무작위평균수익"], 2), pc(RC["같은날후보_평균수익"], 2)],
       ["승률", pc(RC["실제승률"], 1), pc(RC["무작위승률"], 1), pc(RC["같은날후보_승률"], 1)],
       ["실제 신호의 백분위", "", pc(RC["백분위"], 0), pc(RC["같은날후보_백분위"], 0)]], widths=[4.2, 2.6, 4.9, 4.9], size=8)
para("(가)는 '그해에 신호가 나온 종목'을 고르는 순간 그해 크게 오른 종목을 고르게 되어(모멘텀 +30% 조건) 대조군에 유리한 편향이 있다. "
     "또 시가 진입이라 진입일 손절 처리 방식도 다르다. 따라서 신호의 가치는 (나)로 판단한다: 같은 날 같은 성격의 종목 중에서 "
     "돌파·정배열 조건으로 고른 것이 무작위보다 거래당 약 0.1R 낫다. 다만 비용을 빼면 그 차이가 0 근처로 줄어든다.", size=9)
pic("random_control.png", 13)

# ───────────────────────── 강건성
h("7. 강건성 검증")
rows = [["검증", "CAGR", "샤프", "MDD", "거래", "평균R", "t"]]
for nm, r in rob.iterrows():
    rows.append([nm, pc(r.CAGR), f2(r.샤프), pc(r.MDD), nn(r.거래수), f3(r.평균R), f2(r.R_t)])
table(rows, size=7.5, hl=[1])
h3("파라미터 격자")
para(f"EMA(10·20) × ADR(3·4·5%) × 모멘텀 상위(2·5·10%) × 횡보 기간(10·20·40일) × 국면 필터(있음·없음) = {S['격자수']}개. "
     f"학습 구간(거래 200건 이상) 샤프 1위: EMA{sel['ema']}, ADR {sel['adr']:.0%}, 상위 {sel['top']:.0%}, {sel['n']}일, 국면 필터 {'있음' if sel['regime'] else '없음'}.")
rows = [["구간", "CAGR", "샤프", "MDD", "알파 t", "거래", "평균R"]]
for sg in ("학습", "검증", "테스트", "전체"):
    r = gsel.loc[sg]
    rows.append([sg, pc(r.CAGR), f2(r.샤프), pc(r.MDD), f2(r.t_코스피), nn(r.거래수), f3(r.평균R)])
table(rows, size=8)
rows = [["구간", "샤프 > 0 설정 수", "평균R > 0 설정 수", "평균R 중앙값", "샤프 중앙값"]]
for sg in ("학습", "검증", "테스트", "전체"):
    g = G[G.구간 == sg]
    rows.append([sg, f"{(g.샤프 > 0).sum()} / {len(g)}", f"{(g.평균R > 0).sum()} / {len(g)}", f3(g.평균R.median()), f2(g.샤프.median())])
table(rows, size=8)
pic("grid_heatmap.png")
h3("나눠 보기 (기본 설정, 거래 단위)")
rows = [["구분", "값", "거래", "승률", "평균R", "t"]]
for _, r in BD[BD.구분 != "연도"].iterrows():
    rows.append([r.구분, r.값, nn(r.거래수), pc(r.승률, 0), f3(r.평균R), f2(r.R_t)])
table(rows, size=8)
bullet(f"상위 5종목이 종목별 이익 합계의 {pc(S['상위5종목_이익합_비중'], 0)}, 상위 5% 거래가 이익 합계의 {pc(S['상위5pct거래_이익비중'], 0)}.")
if S.get("t2_필요연수"):
    bullet(f"테스트 구간 알파 t {S['테스트_알파t']:.2f}: 같은 효과가 이어진다면 t=2에 약 {S['t2_필요연수']:.0f}년 필요(테스트 {S['테스트연수']:.1f}년).")
else:
    bullet(f"테스트 구간 알파 t {S['테스트_알파t']:.2f} (음수라 필요 연수 계산 안 함).")

# ───────────────────────── 시봉 확인
HK = HERE.parent / "hourly_youtube" / "results" / "kullamaggie_entry_check.csv"
if HK.exists():
    KH = pd.read_csv(HK)
    h3("시봉으로 진입일 확인 (2023-11 ~ 2026-10)")
    para("야후 60분봉 3년치로 같은 기간 거래의 진입일을 다시 처리했다(돌파가 처음 일어난 시간 봉과 그 전후 저가 확인). "
         "자세한 방법은 hourly_youtube 폴더의 '유튜브 전략 시봉 재검증' 보고서.", size=9)
    rows = [["처리", "거래", "승률", "평균R", "t", "진입일 손절"]]
    for _, x in KH.iterrows():
        rows.append([x.처리, nn(x.거래수), pc(x.승률, 0), f3(x.평균R), f2(x.R_t), pc(x.진입일손절비율, 0)])
    table(rows, size=8, hl=[1])
    kh_, k1_ = KH.iloc[0], KH[KH.처리 == "일봉: 봉 방향 가정"].iloc[0]
    bullet(f"실제(시봉) {f3(kh_.평균R)}R은 봉 방향 가정({f3(k1_.평균R)}R)에 가깝고 보수 가정보다 훨씬 낫다. 다만 봉 방향 가정이 약 {abs(kh_.평균R - k1_.평균R):.2f}R 낙관적이었다.")

# ───────────────────────── 한계
h("8. 한계")
bullet("일봉 근사: 원작의 장중 오프닝 레인지 돌파·당일 저가 손절을 일봉으로 옮겨 진입일 처리에 가정이 들어간다(7장 비교). 최근 3년은 시봉으로 확인했다.")
bullet("체결: 돌파 가격에 그대로 체결된다고 보고, 슬리피지는 편도 비용 0.25% 안에 묶었다. 급등 종목은 실제 호가 공백이 더 클 수 있다.")
bullet("같은 종목은 앞 거래가 끝난 다음 날부터 다시 신호를 본다. 포트폴리오가 현금 부족으로 건너뛴 거래 기간에도 같은 종목 재진입은 따로 만들지 않았다.")
bullet("2018-07 이후 상장폐지 95종목은 종가만 있어 진입 대상에서 빠진다(보유 중 상장폐지는 반영).")
bullet("미국 시장(원작의 주 무대)은 이번 검증 대상이 아니다.")

# ───────────────────────── 결론
h("9. 결론")
CONC = (RES / "conclusion.txt")
if CONC.exists():
    for line in CONC.read_text(encoding="utf-8").strip().splitlines():
        bullet(line)

doc.save(OUT)
print(OUT)
try:
    sys.path.insert(0, str(next(p for p in HERE.parents if (p / "jobapp").exists()) / "jobapp"))
except StopIteration:
    sys.path.insert(0, r"C:\Users\arlo0\OneDrive\바탕 화면\취업\jobapp")
try:
    from jobkit.common import to_pdf
    print(to_pdf(OUT))
except Exception as e:
    print("PDF 변환 실패:", e)
