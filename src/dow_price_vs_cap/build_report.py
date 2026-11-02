"""보고서 생성: results/results.json + 그림 → 보고서_초안.docx
실행 순서: python data.py → python run.py → python build_report.py"""
import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
R = json.loads((RES / 'results.json').read_text(encoding='utf-8'))
OUT = HERE / (sys.argv[1] if len(sys.argv) > 1 else '보고서_초안.docx')
B, P = R['base'], R['params']
NAVY = '1F3864'


def pct(x, d=1):
    return f'{x * 100:.{d}f}%'


def pp(x, d=1):
    return f'{x * 100:+.{d}f}%p'


doc = Document()
sec = doc.sections[0]; sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(1.8); sec.top_margin = Cm(1.5); sec.bottom_margin = Cm(1.5)
st = doc.styles['Normal']; st.font.name = 'Malgun Gothic'; st.font.size = Pt(9.5)
st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Malgun Gothic')
st.paragraph_format.space_after = Pt(3)


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr(); s_ = OxmlElement('w:shd')
    s_.set(qn('w:val'), 'clear'); s_.set(qn('w:color'), 'auto'); s_.set(qn('w:fill'), color); tcPr.append(s_)


def para(text, bold=False, size=None, color=None, after=None, italic=False, align=None):
    p = doc.add_paragraph(); r = p.add_run(text); r.bold = bold; r.italic = italic
    if size: r.font.size = Pt(size)
    if color: r.font.color.rgb = RGBColor.from_string(color)
    if after is not None: p.paragraph_format.space_after = Pt(after)
    if align is not None: p.alignment = align
    return p


def h(text):
    p = para(text, bold=True, size=12, color=NAVY, after=3); p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.keep_with_next = True
    pPr = p._p.get_or_add_pPr(); b = OxmlElement('w:pBdr'); bt = OxmlElement('w:bottom')
    for k, v in (('val', 'single'), ('sz', '6'), ('space', '1'), ('color', NAVY)): bt.set(qn(f'w:{k}'), v)
    b.append(bt); pPr.append(b)


def h2(text):
    p = para(text, bold=True, size=10.5, color=NAVY, after=2); p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.keep_with_next = True


def bullet(text):
    p = doc.add_paragraph(style='List Bullet'); p.paragraph_format.space_after = Pt(1)
    p.add_run(text).font.size = Pt(9.5)


def table(rows, widths=None, size=8, first_left=True, hl=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0])); t.style = 'Table Grid'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = ''; p = c.paragraphs[0]; r = p.add_run(str(v)); r.font.size = Pt(size)
            p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if (j == 0 and first_left) else WD_ALIGN_PARAGRAPH.CENTER
            if i == 0:
                r.bold = True; r.font.color.rgb = RGBColor(255, 255, 255); shade(c, NAVY)
            elif hl and i in hl:
                shade(c, 'DEEAF6')
    if widths:
        for row in t.rows:
            for j, w in enumerate(widths): row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def pic(name, w=16.5, cap=None):
    doc.add_picture(str(RES / name), width=Cm(w)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if cap:
        para(cap, size=8, color='595959', after=6, align=WD_ALIGN_PARAGRAPH.CENTER)


pk, ck, ek = B['price']['KRW'], B['cap']['KRW'], B['equal']['KRW']
pu, cu = B['price']['USD'], B['cap']['USD']
A = R['alpha']
rob = R['robust']
rep = R['replication']
W = R['weights_last']
conc = R['top5_conc']
y_last = max(conc['cap'], key=int)
top5_cap = conc['cap'][y_last]
top5_price = conc['price'][max(conc['price'], key=int)]

# ───────── 표지·요약
para('다우존스 산업평균 30종목, 가격가중 vs 시가총액가중', bold=True, size=17, color=NAVY, after=2)
para('같은 종목을 비중만 바꿔 국내 상장 분배형 ETF로 운용했다면 (2010.01 ~ 2026.10 백테스트)', size=10.5, color='404040', after=8)

h('1. 요약')
table([
    ['', '가격가중(다우 방식)', '시가총액가중', '차이'],
    ['연 수익률(CAGR, 원화·환노출)', pct(pk['CAGR'], 2), pct(ck['CAGR'], 2), pp(ck['CAGR'] - pk['CAGR'], 2)],
    ['연 수익률, 청산 시 매매차익 과세 후', pct(pk['CAGR_after_liq'], 2), pct(ck['CAGR_after_liq'], 2), pp(ck['CAGR_after_liq'] - pk['CAGR_after_liq'], 2)],
    ['연 수익률(USD 기준)', pct(pu['CAGR'], 2), pct(cu['CAGR'], 2), pp(cu['CAGR'] - pu['CAGR'], 2)],
    ['변동성(원화)', pct(pk['Vol']), pct(ck['Vol']), pp(ck['Vol'] - pk['Vol'])],
    ['최대낙폭 MDD(원화)', pct(pk['MDD']), pct(ck['MDD']), pp(ck['MDD'] - pk['MDD'])],
    ['샤프지수(원화)', f"{pk['Sharpe']:.2f}", f"{ck['Sharpe']:.2f}", f"{ck['Sharpe'] - pk['Sharpe']:+.2f}"],
    [f'상위 5종목 비중 합({y_last}년 말)', pct(top5_price), pct(top5_cap), pp(top5_cap - top5_price)],
    ['연 회전율(편도)', pct(B['price']['turnover']), pct(B['cap']['turnover']), pp(B['cap']['turnover'] - B['price']['turnover'])],
], widths=[6.0, 3.6, 3.6, 3.0], hl=[1, 2])
bullet(f"시가총액가중이 연 {pp(ck['CAGR'] - pk['CAGR'], 2).replace('%p', '%p')} 높았다(원화, 분배금 15.4% 과세 후 재투자, 총보수 0.35% 차감). "
       f"1억 원을 넣었다면 청산 후 가격가중 {pk['Final_after_liq']:.2f}억 원, 시가총액가중 {ck['Final_after_liq']:.2f}억 원이다.")
bullet(f"다만 차이는 통계적으로 유의하지 않다. 일별 회귀 알파 연 {pct(A['alpha_ann'], 2)}, t값 {A['alpha_t']:.2f}이고, "
       f"시가총액가중이 이긴 달은 {pct(A['hit'], 0)}로 절반 수준이다.")
bullet(f"기간을 나누면 결과가 뒤집힌다. 2010~2017년은 가격가중이 연 {pct(rob['2010~2017']['price']['CAGR'], 1)}로 "
       f"시가총액가중 {pct(rob['2010~2017']['cap']['CAGR'], 1)}보다 높았다. "
       f"2018~2026년은 시가총액가중이 {pct(rob['2018~2026']['cap']['CAGR'], 1)}로 가격가중 {pct(rob['2018~2026']['price']['CAGR'], 1)}보다 높았다. "
       "시가총액가중의 우위는 Apple·Microsoft·NVIDIA 세 종목에서 거의 다 나왔다.")
bullet(f"시가총액가중 다우는 {y_last}년 말 상위 5종목(NVIDIA·Apple·Microsoft·Alphabet·Amazon) 비중이 {pct(top5_cap, 0)}이다. "
       f"30종목 지수라기보다 대형 기술주 5개 펀드에 가까워진다. 가격가중은 {pct(top5_price, 0)}로 분산이 유지된다.")
bullet('결론: 가격가중이 수익률에서 불리했던 건 맞지만, 그 원인은 가중 방식 자체보다 지난 8년의 대형 기술주 장세였다. '
       '가격가중은 집중도를 낮추는 장치로 작동했다.')

# ───────── 2. 질문과 설계
h('2. 질문과 비교 설계')
para('다우존스 산업평균은 주가(1주 가격)에 비례해 비중을 매긴다. 주가가 높은 종목이 회사 규모와 상관없이 큰 비중을 갖는다. '
     '예를 들어 주가 400달러·발행주식 10억 주(시총 4,000억 달러)인 A와 주가 100달러·100억 주(시총 1조 달러)인 B가 있다고 하자. '
     '가격가중이면 A 80% : B 20%이고, 시가총액가중이면 A 29% : B 71%이다. '
     '이 보고서는 편입 종목을 그대로 두고 비중 규칙만 바꿨을 때 수익률이 어떻게 달라지는지 확인한다.')
table([
    ['항목', '내용'],
    ['종목', '각 시점의 실제 다우 30종목. 편입·편출 이력 반영(9회 변경, 탈퇴 종목 포함)'],
    ['A. 가격가중', '비중 ∝ 실제 주가. 편입 변경일·분할·스핀오프일에만 다시 맞추고, 그 사이에는 지수처럼 저절로 움직인다'],
    ['B. 시가총액가중', '비중 ∝ 주가 × 발행주식수. 분기 말과 편입 변경일에 다시 맞춘다. 발행주식수는 직전 월말까지 공개된 시가총액 ÷ 그날 주가'],
    ['C. 동일가중(참고)', '30종목 각 1/30, 분기 말 재조정'],
    ['상품 구조', '국내 상장 해외주식형 ETF, 분배형(월말 분배). 2025.7 시행령 개정으로 해외주식형 TR(분배금 재투자형) 불가'],
    ['세금', '분배금 15.4%(미국 원천 15% 포함), 분배 후 남은 돈을 같은 ETF에 재투자. 매도 시 매매차익(원화)에도 15.4%(배당소득)'],
    ['비용', '총보수 연 0.35%(TIGER 미국다우존스30 운용보수와 같은 수준, 두 상품 동일), 거래비용 편도 0.05%'],
    ['통화', 'USD / 원화 환노출 / 원화 환헤지(헤지손익 ≈ 한국 3개월 금리 − 미국 3개월 국채 금리)'],
    ['기간', '2010-01-04 ~ 2026-10-29 (16.8년, 일별)'],
    ['평가지표', 'CAGR, 청산 후 CAGR, 변동성, 샤프, MDD, 회전율, 연도별 수익률, 알파 t값(Newey-West), 시작 연도·기간 분할 강건성'],
], widths=[3.4, 13.4], size=8)

# ───────── 3. 데이터
h('3. 데이터와 기간 선택: 왜 2010년부터인가')
para('처음 계획은 1995년부터였다. 다음 두 가지 이유로 시작을 2010년으로 바꿨다.')
bullet('빠진 종목이 너무 많다. 1995~1999년 다우 30종목 중 최대 10종목(코닥, 옛 GM, 시어스, 베들레헴스틸, 텍사코, 유니언카바이드, '
       '웨스팅하우스, 울워스, 옛 AT&T, J.P.모건)이 상장폐지나 합병으로 무료 일별 시세가 없다. '
       '이 구간을 넣으면 지수의 최대 3분의 1을 평균 수익률 같은 대체값으로 채워야 하는데, 그러면 가중 방식의 차이가 아니라 대체 방식의 차이를 재게 된다.')
bullet('1995~2009년에는 발행주식수를 구할 수 없다. 시가총액가중의 핵심 입력이 과거 발행주식수인데, '
       '무료로 쓸 수 있는 월별 시가총액 시계열은 대부분 1996년부터 있지만 탈퇴 종목은 빠져 있다. 공시 원문(SEC EDGAR)에서 직접 추출하는 방법은 이번 작업 환경에서 접속이 막혀 쓰지 않았다.')
bullet('2010년 이후로는 대체값이 필요한 곳이 거의 없다. 옛 Alcoa(2010.01~2013.09)는 시세가 없어 두 포트폴리오 모두에서 같이 빼고 29종목으로 운용했다. '
       'Walgreens(2018.06~2024.02)는 2025년 상장폐지로 일별 시세가 없어, 월말 실제 시가총액·배당 재투자 가격을 쓰고 월 안의 일별 경로만 나머지 종목 평균 움직임으로 이었다. '
       'Walgreens의 월 수익률은 실제값과 같다.')
table([
    ['데이터', '출처', '용도'],
    ['일별 종가·수정종가', 'stock_db.us_price (Yahoo 기반, 분할·스핀오프 보정가 + 배당 보정가)', '수익률, 배당락 배당수익률 역산'],
    ['분할·스핀오프 배율', 'Yahoo(yfinance) 기록 35건', '실제 거래가격 복원(가격가중 비중)'],
    ['월말 시가총액', 'companiesmarketcap.com 월말 시가총액', '발행주식수 = 시가총액 ÷ 그 달 종가'],
    ['Walgreens 월말값', 'companiesmarketcap.com 시가총액·발행주식수·총수익 가격', '상장폐지 종목 보완'],
    ['다우지수(^DJI), DIA', 'Yahoo, stock_db', '복제 검증'],
    ['환율·금리', 'FRED DEXKOUS, DTB3, IR3TIB01KRM156N', '원화 환산, 환헤지 손익, 샤프 무위험금리'],
], widths=[3.4, 8.2, 5.2], size=8)
h2('수정주가 처리')
para(f"Yahoo 종가는 분할·스핀오프가 반영된 값이라, 가격가중에 필요한 '그날 실제 주가'는 그 이후의 분할·스핀오프 배율을 곱해 되살렸다(기록 {R['n_split']}건: "
     "Apple 7:1·4:1, NVIDIA 4:1·10:1, Walmart 3:1, 3M의 Solventum 분리, UTX의 Carrier·Otis 분리 등). "
     f"DB 시세에서 분할 후 과거 행이 보정되지 않아 생긴 가격 단절은 공용 코드(quant/price_adjust.py)로 검사했고, 보정이 필요한 건은 {R['n_fix']}건이었다. "
     "배당은 수정종가와 종가의 비율에서 배당락일 배당수익률을 역산했다.")
h2('복제 검증: 이 방법으로 실제 다우를 재현할 수 있는가')
yr = rep['yearly']
rows = [['연도', '복제 가격지수', '실제 다우(^DJI)', '차이']]
for y in sorted(yr, key=int):
    a_, b_ = yr[y]
    rows.append([y, pct(a_, 2), pct(b_, 2), pp(a_ - b_, 2)])
table(rows, widths=[2.4, 3.4, 3.4, 3.0], size=7.5)
dc = R['dia_check']
para(f"일별 수익률 상관계수 {rep['corr']:.4f}, 연환산 추적오차 {pct(rep['te'], 2)}이다. 2010~2013년 차이는 Alcoa를 뺀 영향이고, 그 뒤로는 연 0.4%p 안쪽이다."
     f"배당까지 넣은 총수익(보수·비용·세금 0)은 연 {pct(dc['ours_cagr'], 2)}로, 실제 다우 ETF DIA(보수 0.16%)의 {pct(dc['dia_cagr'], 2)}와 "
     f"{pp(dc['ours_cagr'] - dc['dia_cagr'], 2)} 차이다. 보수 차이를 빼면 거의 같다. 가격가중 쪽 엔진은 실제 다우를 충실히 재현한다.")

# ───────── 4. 결과
h('4. 결과')
rows = [['', '통화', 'CAGR', '청산 후 CAGR', '변동성', '샤프', 'MDD', '1 → 최종(청산 후)']]
for k, nm in [('price', '가격가중'), ('cap', '시가총액가중'), ('equal', '동일가중(참고)')]:
    for c, cn in [('USD', 'USD'), ('KRW', '원화 환노출'), ('KRW_H', '원화 환헤지')]:
        m = B[k][c]
        rows.append([nm, cn, pct(m['CAGR'], 2), pct(m['CAGR_after_liq'], 2), pct(m['Vol']), f"{m['Sharpe']:.2f}",
                     pct(m['MDD']), f"{m['Final']:.2f} ({m['Final_after_liq']:.2f})"])
table(rows, widths=[2.8, 2.4, 1.7, 2.2, 1.6, 1.3, 1.6, 3.2], size=7.5, hl=[4, 5, 6])
para(f"세 방식 모두 분배금 과세 후 재투자, 총보수·거래비용 차감 후다. 청산 후 CAGR은 마지막 날 전량 매도하며 매매차익(취득원가 = 최초 투자 + 재투자액)에 15.4%를 낸 값이다. "
     f"연 배당수익률(세전)은 가격가중 {pct(B['price']['div_yield'], 2)}, 시가총액가중 {pct(B['cap']['div_yield'], 2)}로 가격가중이 조금 높다. "
     f"거래 건수는 가격가중 {B['price']['n_trades']}건, 시가총액가중 {B['cap']['n_trades']}건이고, 회전율은 둘 다 연 10% 미만이라 거래비용 영향이 작다.")
para('원화 환노출과 환헤지의 연 수익률은 거의 같다. 기간 전체의 원화 약세 이득과 환헤지 비용(원화 금리 − 달러 금리)이 비슷했기 때문이다. '
     '환노출은 위기 때 달러가 강해져 원화 기준 낙폭이 더 작았다.')
pic('fig_cum.png', cap='그림 1. 누적 가치(USD, 세후 분배 재투자, 보수·비용 차감)')
pic('fig_dd.png', cap='그림 2. 낙폭')
pic('fig_yearly.png', cap='그림 3. 연도별 수익률')
yy = R['yearly']
rows = [['연도', '가격가중', '시가총액가중', '차이', '동일가중']]
for y in sorted(yy, key=int):
    v = yy[y]
    rows.append([y, pct(v['price']), pct(v['cap']), pp(v['cap'] - v['price']), pct(v['equal'])])
table(rows, widths=[2.0, 3.0, 3.0, 2.6, 3.0], size=7.5)

h2('차이는 어디서 왔나')
para('같은 30종목이라도 비중이 크게 다르다. 마지막 리밸런싱 기준으로 시가총액가중은 NVIDIA·Apple·Microsoft·Alphabet·Amazon 다섯 종목에 70% 넘게 몰린다. '
     '가격가중에서는 Goldman Sachs·Caterpillar·Microsoft처럼 주가가 높은 종목이 크고, 주가를 분할로 낮춰 온 기술주는 작다.')
pic('fig_weights.png', cap='그림 4. 마지막 리밸런싱 비중 비교 (상위 15종목)')
ct = R['contrib_top']
neg = [k for k in ct if ct[k][2] < 0][:4]
pos_ = sorted([k for k in ct if ct[k][2] > 0], key=lambda k: -ct[k][2])[:3]
para('종목별 기여(일별 비중 × 수익률의 합)를 비교하면, 시가총액가중 우위는 '
     + ', '.join(f"{k} {pp(ct[k][2], 0)}" for k in pos_) + '에서 나왔다. 반대로 가격가중은 '
     + ', '.join(f"{k} {pp(-ct[k][2], 0)}" for k in neg) + ' 같은 고가주에서 앞섰다.')
pic('fig_contrib.png', cap='그림 5. 종목별 누적 기여 차이 (양수 = 시가총액가중이 더 벌었음)')
pic('fig_conc.png', cap='그림 6. 집중도: 상위 5종목 비중 합')

# ───────── 5. 통계적 유의성·강건성
h('5. 통계적 유의성과 강건성')
para(f"시가총액가중 일별 수익률을 가격가중 수익률에 회귀하면(Newey-West 표준오차, 시차 5) 알파는 연 {pct(A['alpha_ann'], 2)}, t값 {A['alpha_t']:.2f}, "
     f"베타 {A['beta']:.2f}이다. 월별 초과수익 평균은 {pct(A['ex_month_mean'], 2)}, t값 {A['ex_month_t']:.2f}({A['n_months']}개월)이다. "
     "5% 유의수준(|t| ≥ 1.96)에 한참 못 미치므로, 16.8년 표본으로는 시가총액가중이 체계적으로 낫다고 말할 수 없다.")
rows = [['검증', '가격가중 CAGR', '시가총액가중 CAGR', '차이', '알파 t값']]
for k, v in rob.items():
    rows.append([k, pct(v['price']['CAGR'], 2), pct(v['cap']['CAGR'], 2), pp(v['cap']['CAGR'] - v['price']['CAGR'], 2),
                 f"{v['alpha']['alpha_t']:.2f}"])
table(rows, widths=[6.0, 2.8, 3.0, 2.4, 2.0], size=7.5)
para('원화 환노출 기준이다. 거래비용을 4배로 올리거나 시가총액가중의 재조정 주기를 바꿔도 결론은 그대로다. '
     '비용·주기보다 기간 선택의 영향이 훨씬 크다. 2010~2017년만 보면 가격가중이 앞서고(t = −1.76), 2018년 이후만 보면 시가총액가중이 앞선다(t = 1.49). '
     '세금은 두 방식에 거의 같은 크기(연 0.4%p 안팎)로 작용해서 순위를 바꾸지 않는다.')
sy = R['start_year']
rows = [['투자 시작', '가격가중', '시가총액가중', '차이']]
for y in sorted(sy, key=int):
    a_, b_ = sy[y]
    rows.append([f'{y}년 1월 → 2026.10', pct(a_, 2), pct(b_, 2), pp(b_ - a_, 2)])
table(rows, widths=[4.2, 3.0, 3.0, 2.6], size=7.5)
para(f"3년 롤링 초과수익이 양수였던 구간은 전체의 {pct(R['roll3_pos_share'], 0)}이다. 어느 해에 시작했든 2026년까지 들고 있었다면 시가총액가중이 앞섰지만, "
     '이는 모두 같은 2018~2025년 기술주 장세를 포함하기 때문이다.')
pic('fig_roll3.png', cap='그림 7. 3년 롤링 초과수익 (시가총액가중 − 가격가중)')

# ───────── 6. 한계
h('6. 한계')
bullet('기간이 2010년 이후 16.8년이다. 1990년대 후반~2000년대 초 기술주 버블과 붕괴(시가총액가중에 불리했던 구간)는 데이터 제약으로 포함하지 못했다.')
bullet('시가총액은 월말 시가총액을 그날 주가 변화로 늘리거나 줄여 쓴다. 월 중 자사주 매입·증자는 다음 달 값에 반영된다. 유동주식(float) 조정은 하지 않았다.')
bullet('옛 Alcoa(2010.01~2013.09)를 뺀 29종목 구간이 있다. 두 포트폴리오에 똑같이 적용했고, 이 구간을 뺀 2013.09 이후 결과도 결론이 같다.')
bullet('Walgreens는 월말 실제값 사이의 일별 경로를 보간했다. 월·연 수익률은 실제와 같지만 일별 변동성은 조금 작게 나온다(최대 비중 2% 안팎).')
bullet('세금은 단순화했다. 국내 상장 해외 ETF 매매차익의 과세표준(과표기준가 증가분과 실제 차익 중 작은 값), 금융소득 종합과세(연 2천만 원 초과), 연금·ISA 계좌는 반영하지 않았다.')
bullet('스핀오프로 받은 주식은 Yahoo 보정가 방식대로 모회사에 재투자한 것으로 처리했다(두 방식 동일).')
bullet('환헤지는 한·미 3개월 금리차로 근사했다. 실제 선물환 스왑포인트와 롤오버 비용은 다를 수 있다.')

# ───────── 7. 결론
h('7. 결론')
para(f"질문: 다우의 가격가중이 과연 좋은가? 같은 30종목을 시가총액가중으로 바꿨다면 2010~2026년 원화 기준 연 {pct(ck['CAGR'], 1)}로, "
     f"가격가중 {pct(pk['CAGR'], 1)}보다 연 {pp(ck['CAGR'] - pk['CAGR'], 1)} 높았다. 하지만 이 우위는 세 가지 이유로 '가중 방식이 원래 더 좋다'는 근거가 되기 어렵다.")
bullet(f"통계적으로 유의하지 않다(알파 t = {A['alpha_t']:.2f}, 이긴 달 {pct(A['hit'], 0)}).")
bullet('앞선 8년(2010~2017)은 가격가중이 앞섰고, 뒤집힌 건 2018년 이후 Apple·Microsoft·NVIDIA 세 종목 덕분이다.')
bullet(f"시가총액가중 다우는 상위 5종목 비중이 {pct(top5_cap, 0)}까지 올라 나스닥100과 비슷한 대형 기술주 집중 상품이 된다. "
       '다우 고유의 업종 분산(금융·산업재·헬스케어)이 사라진다.')
para('상품 기획 측면의 시사점은 이렇다. 가격가중은 주가 수준이라는 경제적 의미 없는 기준을 쓰지만, 결과적으로 초대형주 쏠림을 막는 상한 장치처럼 작동했다. '
     '시가총액 기반으로 바꾸려면 종목당 비중 상한(예: 10% 캡)을 함께 두는 설계를 다음 단계로 검증할 만하다. 세금(15.4%)과 보수는 두 방식에 똑같이 작용해 선택에 영향을 주지 않는다.')
para('모든 수치는 과거 백테스트이며 실제 매매·주문 코드는 포함하지 않는다.', size=8, color='595959')

doc.save(OUT)
print('saved', OUT)
