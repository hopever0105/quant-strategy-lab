# -*- coding: utf-8 -*-
"""보고서: results/ → 유튜브_전략_시봉_재검증_보고서_초안.docx (+pdf). 숫자는 모두 results/ 에서 읽는다."""
import json
import sys
from pathlib import Path

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
OUT = HERE / ("유튜브_전략_시봉_재검증_보고서.docx" if FINAL else "유튜브_전략_시봉_재검증_보고서_초안.docx")
NAVY = "1F3864"
S = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
T = pd.read_csv(RES / "youtube_daily_vs_hourly.csv")
K = pd.read_csv(RES / "kullamaggie_entry_check.csv")
CH = S["check"]

pc = lambda x, d=1: "-" if pd.isna(x) else f"{x * 100:.{d}f}%"
f2 = lambda x: "-" if pd.isna(x) else f"{x:.2f}"
f3 = lambda x: "-" if pd.isna(x) else f"{x:+.3f}"
nn = lambda x: "-" if pd.isna(x) else f"{int(x):,}"

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
    b = OxmlElement("w:pBdr"); bt_ = OxmlElement("w:bottom")
    for k, v in (("val", "single"), ("sz", "6"), ("space", "1"), ("color", NAVY)):
        bt_.set(qn(f"w:{k}"), v)
    b.append(bt_); pPr.append(b)


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


def r(nm, fq):
    x = T[(T.전략 == nm) & (T.봉 == fq)]
    return x.iloc[0] if len(x) else None


H = T[T.봉 == "시봉"]
n_pos = int((H.평균R > 0).sum())
n_beat = int((H.백분위 >= 0.95).sum())
kh, k1, k0 = (K[K.처리 == n].iloc[0] for n in ("시봉으로 진입일 확인", "일봉: 봉 방향 가정", "일봉: 보수 가정"))

para("유튜브 매매법 시봉 재검증", bold=True, size=18, color=NAVY, after=2)
para("로스 MACD·RSI, OBV·켈트너, 하이킨아시, 쿨라마기 · 국내 주식 2023-12 ~ 2026-10 · 같은 규칙으로 일봉과 시봉 비교",
     size=10, color="595959", after=8)

h("1. 요약")
bullet(f"영상들은 시봉·분봉 차트를 쓰지만 기존 검증은 일봉이었다. 야후 60분봉 3년치({CH['종목']:,}종목)를 받아 각 전략의 원래 규칙과 숫자(봉 개수 기준)를 "
       f"그대로 두고 일봉·시봉을 같은 기간·같은 종목에서 비교했다.")
bullet(f"시봉으로 바꿔도 이기는 전략은 없었다. 시봉 {len(H)}개 설정 중 거래당 평균 R이 양수인 것은 {n_pos}개였고, "
       f"같은 종목·같은 달 무작위 진입보다 뚜렷이 나은(무작위 20회 중 95% 이상 위) 설정은 {n_beat}개였는데, "
       f"그 {n_beat}개(로스 매매법 1, 하이킨아시 N)도 평균 R이 음수라 '무작위보다 덜 잃었다'는 뜻이다.")
r1h, r1d = r("로스 매매법 1", "시봉"), r("로스 매매법 1", "일봉")
bullet(f"로스 매매법 1: 일봉 {f3(r1d.평균R)}R({nn(r1d.거래수)}건) → 시봉 {f3(r1h.평균R)}R({nn(r1h.거래수)}건, t {f2(r1h.R_t)}). "
       f"시봉은 거래가 6배로 늘고 한 번의 움직임이 작아 같은 비용이 더 크게 작용한다.")
bh = r("OBV·켈트너 B", "시봉")
bullet(f"OBV·켈트너 B(영상 규칙에 가장 가까운 변형)는 시봉에서 {f3(bh.평균R)}R(t {f2(bh.R_t)})로 양수였지만, 같은 종목·같은 달 무작위 진입이 {f3(bh.무작위평균R)}R로 더 높았다. "
       "이 기간은 상승장이라 아무 때나 사도 이익이었고, 신호가 그보다 낫지 않았다.")
bullet(f"쿨라마기: 일봉 백테스트에서 결과를 크게 흔들던 '진입일 장중 순서'를 시봉으로 확인했다. 같은 기간 거래당 평균 R은 시봉 확인 {f3(kh.평균R)}, "
       f"일봉 봉 방향 가정 {f3(k1.평균R)}, 일봉 보수 가정 {f3(k0.평균R)}. 실제는 봉 방향 가정에 가깝지만 약 {abs(kh.평균R - k1.평균R):.2f}R 더 나쁘다.")

h("2. 데이터")
bullet(f"시봉: 야후 60분봉(무료, 계좌 불필요), 하루 6개 봉(09~14시 시작). {S['기간'][0]} ~ {S['기간'][1]} 신호 대상. KRX 요청 없음.")
bullet("야후 14시 봉은 15:20 전에 끝나 장 마감 동시호가가 빠져 있다. 그래서 매일 15:30 '종가 봉'(공식 종가, 빠진 거래량)을 붙였다. "
       f"이후 하루로 묶은 고가·저가가 일봉과 0.5% 안에서 맞는 비율은 {pc(CH['고가일치_0_5pct'])}·{pc(CH['저가일치_0_5pct'])}, 거래량 합은 일봉의 {pc(CH['시봉거래량_합_대비_일봉'], 0)}다.")
bullet(f"가격 수준은 시가 기준 배율로 일봉 수정주가에 맞췄다. 배율이 하루에 10% 넘게 바뀐 {nn(CH['뺀_종목일'])}개 종목·일(시가 오류 의심)은 뺐다.")
bullet(f"09시 봉의 {pc(CH['09시봉_거래량0_비율'], 0)}는 야후에서 거래량이 0으로 온다. OBV 누적과 하이킨아시 '거래량 증가' 조건이 그만큼 부정확하다.")
bullet("생존편향: 현재 상장 종목 95%의 시봉은 있지만, 2023-11 이후 상장폐지 141종목 중 야후에 남은 것은 9종목뿐이다. 3년 구간이라 영향은 작지만 결과를 조금 좋게 만든다.")
bullet("일봉 비교도 같은 종목 집합·같은 기간으로 다시 돌렸다(기존 보고서의 27년 결과와는 기간이 다르다).")

h("3. 같은 규칙을 봉만 바꿔 적용")
bullet("로스 MACD·RSI: 기존 macd_rsi_swing 코드의 신호·청산 함수를 그대로 가져왔다. 손절 = 직전 10봉 최저가(손절폭 3~15%), 비용 편도 0.115% + 매도세.")
bullet("OBV·켈트너: 기존 7개 변형 그대로(박스 20봉, OBV EMA20, 켈트너 20/10/1.5, 1R·2R·3R 분할, 60봉 만료, 왕복 0.3%).")
bullet("하이킨아시: 도지 반전 + EMA20(N 없음 / R 종가 위 / S 돌파), 손절 도지 저점×0.995, 2R 익절, 스무스 HA 전환 청산, 60봉 만료, 왕복 0.3%.")
bullet("유동성 조건은 일 단위로 고정했다(시봉에서 '5봉 평균'이 5시간이 되지 않게): 로스 전날까지 20일 평균 거래대금 10억, OBV·하이킨아시 5일 평균 30억.")
bullet("무작위 대조: 거래마다 같은 종목·같은 달에서 무작위 봉 20개를 골라, 같은 손절폭·같은 청산 규칙으로 진입했다.")

h("4. 결과: 일봉 vs 시봉")
rows = [["전략", "봉", "거래", "승률", "평균 수익", "평균R", "t", "무작위 평균R", "백분위", "'24 / '25~'26 평균R"]]
hl = []
for nm in T.전략.unique():
    for fq in ("일봉", "시봉"):
        x = r(nm, fq)
        if x is None:
            continue
        rows.append([nm if fq == "일봉" else "", fq, nn(x.거래수), pc(x.승률, 0), pc(x.평균수익, 2), f3(x.평균R), f2(x.R_t),
                     f3(x.무작위평균R), pc(x.백분위, 0), f"{f3(x.앞구간평균R)} / {f3(x.뒤구간평균R)}"])
        if fq == "시봉":
            hl.append(len(rows) - 1)
table(rows, size=7, hl=hl)
para("백분위 = 무작위 진입 20회 평균 R 중 실제 신호보다 낮은 비율(95% 이상이면 무작위보다 뚜렷이 나음). 거래가 0~50건인 일봉 OBV 변형은 판단하기 어렵다.", size=8.5)
pic("daily_vs_hourly.png")

h("5. 쿨라마기 진입일 확인")
para("일봉에는 장중 순서가 없어, 돌파한 날 저가가 진입 전에 나왔는지 후에 나왔는지 가정해야 했다. 같은 기간 거래를 시봉으로 다시 처리했다. "
     "시봉으로는 돌파가 처음 일어난 시간 봉과 그 전후 저가를 볼 수 있어, 남는 가정은 진입 봉 안의 순서뿐이다.")
rows = [["처리", "거래", "승률", "평균 수익", "평균R", "t", "진입일 손절", "상위 5% 제외 평균R"]]
for _, x in K.iterrows():
    rows.append([x.처리, nn(x.거래수), pc(x.승률, 0), pc(x.평균수익, 2), f3(x.평균R), f2(x.R_t), pc(x.진입일손절비율, 0), f3(x.상위5pct제외평균R)])
table(rows, size=8, hl=[1])
pic("kullamaggie_entry_check.png", 14)
bullet(f"시봉이 없어 일봉 가정으로 처리한 날: {S['쿨라마기_시봉없음']}건.")

h("6. 해석")
bullet("시봉으로 바꾸면 신호가 많아지지만 한 번에 먹는 폭이 작아져, 같은 왕복 비용(0.3~0.5%)의 비중이 커진다. 로스·하이킨아시는 시봉에서 더 나빠졌다.")
bullet("2023-12~2026-10은 코스피가 크게 오른 구간이라, 아무 때나 사도 같은 청산 규칙으로 이익이 났다. 그래서 신호의 가치는 무작위 진입과의 차이로 판단했고, 차이가 뚜렷한 설정은 거의 없었다.")
bullet("쿨라마기 일봉 결과는 '봉 방향' 가정이 실제에 가까웠다(보수 가정은 지나치게 나빴다). 다만 실제는 조금 더 나빠, 일봉 기본 결과(비용 후 손익분기)를 낮춰 읽어야 한다.")

h("7. 한계")
bullet("기간 3년(야후 시봉 한도). 상승장 한 번이 대부분이라 통계적 결론에는 짧다.")
bullet("09시 봉 거래량 결측, 마감 동시호가 보정, 상장폐지 종목 부족(9/141).")
bullet("영상들은 15분봉·5분봉을 쓰는 경우도 있다. 분봉은 무료 자료가 7~60일뿐이라 시험하지 못했다.")
bullet("터틀 자산배분은 원래 일봉 ETF 시스템이라 이번 대상에서 뺐다.")

doc.save(OUT)
print(OUT)
sys.path.insert(0, r"C:\Users\arlo0\OneDrive\바탕 화면\취업\jobapp")
try:
    from jobkit.common import to_pdf
    print(to_pdf(OUT))
except Exception as e:
    print("PDF 변환 실패:", e)
