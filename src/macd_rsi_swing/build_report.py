"""보고서 초안 생성: results/metrics.json, equity.csv, metrics_us.json → 보고서_초안.docx + 차트(results/*.png)
실행: python quant/strategies/macd_rsi_swing/build_report.py"""
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
import core  # noqa: E402

RES = HERE / "results"
M = json.loads((RES / "metrics.json").read_text(encoding="utf-8"))
S = json.loads((RES / "select.json").read_text(encoding="utf-8"))
U = json.loads((RES / "metrics_us.json").read_text(encoding="utf-8"))
EQ = pd.read_csv(RES / "equity.csv", index_col=0, parse_dates=True)
EQ = EQ[EQ.index >= "2000-01-04"]
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
NAMES = ["매매법1", "매매법2", "합친것"]
COL = {"매매법1": "#2F5597", "매매법2": "#C55A11", "합친것": "#548235", "KOSPI": "#7F7F7F"}
NAVY = "1F3864"
ALL = "전체(2000~)"

# ───────── 총비용 영향: 비용 0 가정 시 거래당 평균 R (총수익 vs 순수익)
pn = core.load_panel(); dates, tks = core.prepare(pn); del pn
P0 = dict(core.P0); Pz = dict(P0); Pz["cost_mult"] = 0.0
gross = {}
for k in ("매매법1", "매매법2"):
    tr = pd.read_pickle(RES / f"trades_{k}.pkl")
    g = core.add_costs(tr[core.COLS], tks, dates, {}, Pz)
    gross[k] = dict(avg_R=float(g.rmult.mean()), avg_ret=float(g.ret.mean()), win=float((g.ret > 0).mean()))

# ───────── 차트
s = EQ[NAMES + ["KOSPI"]]; s = s / s.iloc[0]
fig, ax = plt.subplots(2, 1, figsize=(7.4, 5.4), dpi=170, sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
for c in NAMES + ["KOSPI"]:
    ax[0].plot(s.index, s[c], label=c, color=COL[c], lw=1.1)
    ax[1].plot(s.index, s[c] / s[c].cummax() - 1, color=COL[c], lw=0.8)
ax[0].set_yscale("log"); ax[0].set_ylabel("자산(2000-01-04=1, 로그)"); ax[0].legend(ncol=4, fontsize=8, frameon=False)
ax[1].set_ylabel("낙폭"); ax[1].set_ylim(-1, 0)
for a in ax:
    a.grid(alpha=.25); [a.spines[x].set_visible(False) for x in ("top", "right")]
plt.tight_layout(); plt.savefig(RES / "fig_equity.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.6), dpi=170)
for a, name in zip(ax, ("매매법1", "매매법2")):
    runs = M["random"][name]; cg = np.array([r["CAGR"] for r in runs]) * 100
    act = M["base"][name]["periods"][ALL]["CAGR"] * 100
    a.scatter(np.zeros(len(cg)) + np.random.default_rng(0).normal(0, .04, len(cg)), cg, color="#A6A6A6", s=22, label="무작위 진입 10회")
    a.axhline(act, color=COL[name], lw=1.6, label=f"실제 신호 {act:.1f}%")
    a.axhline(cg.mean(), color="#7F7F7F", ls="--", lw=1, label=f"무작위 평균 {cg.mean():.1f}%")
    a.set_title(name, fontsize=9); a.set_xticks([]); a.set_ylabel("CAGR(%)"); a.legend(fontsize=7, frameon=False)
    [a.spines[x].set_visible(False) for x in ("top", "right")]
plt.tight_layout(); plt.savefig(RES / "fig_random.png"); plt.close()

names = list(M["sensitivity"].keys())
fig, ax = plt.subplots(figsize=(7.4, 2.8), dpi=170)
x = np.arange(len(names)); w = .38
ax.bar(x - w / 2, [M["sensitivity"][n]["매매법1"]["CAGR"] * 100 for n in names], w, color=COL["매매법1"], label="매매법1")
ax.bar(x + w / 2, [M["sensitivity"][n]["매매법2"]["CAGR"] * 100 for n in names], w, color=COL["매매법2"], label="매매법2")
ax.axhline(M["base"]["매매법1"]["periods"][ALL]["CAGR"] * 100, color=COL["매매법1"], ls=":", lw=1); ax.axhline(M["base"]["매매법2"]["periods"][ALL]["CAGR"] * 100, color=COL["매매법2"], ls=":", lw=1)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7); ax.set_ylabel("CAGR(%)"); ax.legend(fontsize=7, frameon=False)
[ax.spines[k].set_visible(False) for k in ("top", "right")]; ax.grid(axis="y", alpha=.25)
plt.tight_layout(); plt.savefig(RES / "fig_sens.png"); plt.close()

# ───────── 문서 유틸
doc = Document()
sec = doc.sections[0]; sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(1.8); sec.top_margin = Cm(1.5); sec.bottom_margin = Cm(1.5)
st = doc.styles["Normal"]; st.font.name = "Malgun Gothic"; st.font.size = Pt(9.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Malgun Gothic")
st.paragraph_format.space_after = Pt(3)


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
            for j, w in enumerate(widths): row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def pic(path, w=16.5):
    doc.add_picture(str(path), width=Cm(w)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


pc = lambda x, d=1: f"{x*100:.{d}f}%"
B = M["base"]; meta = M["meta"]; st_ = meta["stats"]
bm = B["매매법1"]["periods"][ALL]
rnd = {}
for n in ("매매법1", "매매법2"):
    runs = M["random"][n]
    rnd[n] = {k: (np.mean([r[k] for r in runs]), np.std([r[k] for r in runs], ddof=1)) for k in ("avg_R", "CAGR", "sharpe", "win")}

# ───────── 표제 + 요약
para("MACD·RSI 스윙 전략 백테스트 보고서", bold=True, size=16, color=NAVY, after=0)
para("MACD 골든크로스(매매법 1) · RSI 하락 다이버전스 + MACD 재골든크로스(매매법 2) | 한국 주식 2000~2026-10-22, 상장폐지 종목 포함 | 포트폴리오용 검증이며 투자 권유·실매매 아님", size=8.5, color="595959", after=4)
h("1. 요약")
bullet(f"결론: 두 매매법 모두 거래비용 후 우위가 확인되지 않았다. 매매법 1은 CAGR {pc(B['매매법1']['periods'][ALL]['CAGR'])}, MDD {pc(B['매매법1']['periods'][ALL]['MDD'],0)}, 알파 t값 {B['매매법1']['periods'][ALL]['alpha_t']:.1f}로 일관되게 손실이고, 매매법 2는 CAGR {pc(B['매매법2']['periods'][ALL]['CAGR'])}(개발기간 2000~2012는 +{B['매매법2']['periods']['개발 2000~2012']['CAGR']*100:.1f}%, 검증기간 2013~2026은 {pc(B['매매법2']['periods']['검증 2013~2026']['CAGR'])})로 사실상 0이며 검증기간에 악화됐다. KOSPI는 같은 기간 CAGR {pc(bm['bm_CAGR'])}.")
bullet(f"우연 검증: 같은 종목·같은 해에서 진입일만 무작위로 바꾸고 손절·익절 규칙은 그대로 둔 10회 대조에서 신호 진입이 우연보다 낫지 않았다. 거래당 평균 R은 매매법 1 {B['매매법1']['trade']['avg_R']:.3f} 대 무작위 {rnd['매매법1']['avg_R'][0]:.3f}±{rnd['매매법1']['avg_R'][1]:.3f}, 매매법 2 {B['매매법2']['trade']['avg_R']:.3f} 대 {rnd['매매법2']['avg_R'][0]:.3f}±{rnd['매매법2']['avg_R'][1]:.3f}. 신호 자체의 우위는 없다.")
bullet(f"강건성: 비용 2배(CAGR {pc(M['cost2x']['매매법1']['periods'][ALL]['CAGR'])} / {pc(M['cost2x']['매매법2']['periods'][ALL]['CAGR'])}), 기간 3분할, 파라미터 10종 변경(RSI 기준선·과열선·목표배수·손절 기준·손절폭 상한·유동성) 모두에서 CAGR이 음수 또는 0 부근이었다(매매법 1 −5.1%~−13.1%, 매매법 2 −4.2%~−0.7%).")
bullet("종목 선택 기준을 바꿔도 결론 동일: 같은 날 신호가 자리보다 많을 때의 선택 기준(거래대금 큰 순·작은 순·무작위 10회·종목코드 순)과 동시 보유 한도(20→60종목, 한도 없음·동일비중)를 바꿔도 세 가지 모두 CAGR이 음수였다(5장 ⑥).")
bullet(f"비용 전에는 우위가 거의 없고, 비용이 그 작은 우위를 지운다: 비용을 0으로 놓은 거래당 평균 R은 매매법 1 {gross['매매법1']['avg_R']:+.3f}, 매매법 2 {gross['매매법2']['avg_R']:+.3f}(0 부근)이고, 비용 반영 후 {B['매매법1']['trade']['avg_R']:+.3f} / {B['매매법2']['trade']['avg_R']:+.3f}이다(손절폭 약 10% 기준 왕복 비용이 약 0.06R). 승률은 34%·32%다. 여기에 자본 제약(동시 20종목)이 겹쳐 포트폴리오 손실은 거래 평균보다 크다(5장 ①).")
bullet(f"미국(S&P500 현재 구성종목, 생존편향) 참고 결과는 매매법 1이 CAGR {pc(U['base']['매매법1']['periods']['전체']['CAGR'])}로 SPY({pc(U['base']['매매법1']['periods']['전체']['bm_CAGR'])})보다 낮고, 생존편향으로 부풀려진 값이므로 위 한국 결론을 뒤집지 못한다.")

# ───────── 전략 규칙
h("2. 전략 규칙 (구현 정의)")
table([["항목", "정의 (모두 종가 신호 → 다음 거래일 시가 체결)"],
       ["지표", "MACD(12,26,9), RSI(14, Wilder). 일봉, 수정주가."],
       ["매매법 1 진입", "MACD 골든크로스일 RSI 50 이상 70 미만이면 진입. RSI<50이면 보류(MACD 정배열 유지 중 RSI가 50을 상향 돌파하면 진입, 최대 20일). RSI≥70이면 보류(정배열 유지 + RSI가 50~56에서 반등하면 진입, 최대 40일)."],
       ["매매법 2 진입", "스윙 고점(좌우 5일 최고가, 우측 5일 확인 후 사용)이 직전 고점보다 높은데 RSI는 낮은 하락 다이버전스(고점 간격 5~60일) 발생 → 가격이 다이버전스 고점의 85% 아래로 무너지지 않고 40일 안에 MACD가 데드 후 다시 골든크로스하면 진입. RSI 수준 무관."],
       ["손절", "신호일 포함 직전 10일 최저가. 종가 대비 손절폭이 3% 미만이거나 15% 초과면 신호 제외. 시가가 손절가 이하면 진입 안 함."],
       ["매매법 1 청산", "손절폭(R)의 2배(2R) 도달 시 절반 익절, 이후 손절선을 진입가로 상향. 잔량은 MACD 데드크로스·RSI≤50·본전 이탈 중 먼저 오는 것(신호는 종가 기준 → 다음 날 시가)."],
       ["매매법 2 청산", "2R 도달 시 전량 익절. 도달 전 MACD 데드크로스면 다음 날 시가에 전량 매도. 손절 동시 도달일은 손절 우선."],
       ["포트폴리오", "거래당 위험 = 자산의 0.5%, 종목당 최대 10%, 동시 보유 최대 20종목, 레버리지 없음, 같은 날 신호가 많으면 20일 거래대금 큰 순. 유휴 현금 수익률 0%(보수적)."],
       ["거래비용", "수수료 0.015% + 슬리피지 0.10% (편도) + 매도 시 증권거래세·농특세(연도·시장별, 아래). 비용 2배 시나리오는 모두 2배."],
       ["체결 제약", "거래량 0(거래정지)인 날은 매수·매도 불가. 다음 날 시가가 상한가 근접(전일 종가 +제한폭−0.5%p 이상: 2015-06-15 전 ±15%, 후 ±30%)이면 매수 불가. 손절은 시가 갭하락 시 시가로 체결."],
       ["유니버스", "KOSPI·KOSDAQ·KONEX 전 종목(상장폐지 포함, 우선주 포함), 20일 평균 거래대금 10억원 이상만 신호 인정."]],
      widths=[2.6, 14.8], size=8)

# ───────── 데이터
h("3. 데이터와 보정")
para(f"데이터: MySQL stock_db. 국내 2015-01-02~2026-10-22는 기존 kor_price(수정주가), 1999-01-04~2018-07-03은 KRX 전종목 일별 원본(kor_ohlcv_raw, 상장폐지 종목 포함 8,954,783행, 3,706종목), 상장폐지 종목의 2018-07 이후는 kor_price_delisted(종가만). KRX가 2018-07-04부터 빈 응답을 주어 원본은 그 날짜까지만 받았다. 최종 패널 13,747,276행·4,624종목, 분석 단위(구간 분리 후) {meta['n_tickers']:,}개.", size=9)
bullet("수정주가 보정: KRX 등락률은 기준가 대비라서 액면분할·병합·감자·권리락일에도 정상 값으로 나온다. 기준가 = 종가/(1+등락률), 전일 종가/기준가가 1%를 넘게 어긋나면 가격 단절로 보고 그 이전 가격을 보정 — 7,302건 보정(분할·병합급 2배 이상 2,238건, 2,413종목).")
bullet("연결 지점 재조정: 보정한 1999~2014 구간을 kor_price(상장폐지는 kor_price_delisted)와 겹치는 첫날(2015-01-02) 종가 비율로 종목별 재조정(2018-07 이후 사건 반영). 비교 가능 1,837종목 중 연결 지점(2014 마지막 거래일→2015 첫 거래일)에서 ±30%를 넘게 튄 종목 0개(±15% 초과도 0개, 중앙값 +0.35%). 겹침 구간에서 비율이 2% 넘게 변해 사건 탐지가 kor_price와 어긋난 종목 157개.")
bullet(f"보정되지 않은 가격 단절 방어: 일간 변동이 가격제한폭을 넘거나(4,329건) 15일 넘게 데이터가 끊긴(187건) 곳에서 종목을 구간으로 나눠 거래가 단절을 가로지르지 못하게 했다(이 과정에서 최근 3일만 kor_price에 들어 있던 우선주 등의 가짜 장기 보유 거래를 발견·제거).")
bullet("거래대금: 원본(1999~2018-07-03)은 KRX 거래대금, 이후는 수정종가×거래량(겹치는 구간에서 원본 거래대금의 0.999배로 검증). 상장폐지 종가-only 구간(2018-07 이후, 336,817행)은 거래대금이 없어 시가총액 500억원 이상을 유동성 대용으로 썼다.")
para("증권거래세·농특세(매도, %): 근거는 나무위키 '증권거래세' 표(2026-07-31 판)와 각주이며 법령 원문은 직접 확인하지 못했다(확인 필요). 1999~2018 구간은 각주 기반 추정이다. 시장 구분은 현재 상장 종목만 알 수 있어 모르는 종목(대부분 상장폐지)은 코스닥 세율(보수적)을 적용했다.", size=8.5)
table([["기간", "KOSPI", "KOSDAQ", "비고"],
       ["~2013-06-30", "0.30", "0.50", "KOSPI 0.15+농특 0.15 / KOSDAQ 각주상 이전 0.50%"],
       ["2013-07-01~2019-05-29", "0.30", "0.30", ""],
       ["2019-05-30~2020-12-31", "0.25", "0.25", ""], ["2021~2022", "0.23", "0.23", ""], ["2023", "0.20", "0.20", ""],
       ["2024", "0.18", "0.18", ""], ["2025", "0.15", "0.15", "KOSPI 거래세 폐지, 농특세 0.15만"], ["2026~", "0.20", "0.20", "KOSPI 0.05+0.15"]],
      widths=[4.2, 2.2, 2.2, 8.8], size=8)

para("※ 각주(거래세 근거): 나무위키 기준, 법령 원문 미확인.", size=8, color="595959")

# ───────── 결과
h("4. 결과 (비용 반영, 2000-01-04~2026-10-22)")
rows = [["", "CAGR", "변동성", "샤프", "MDD", "알파(연)", "알파 t", "베타", "거래 수", "승률", "평균 R", "평균 보유일", "회전율(연)", "평균 투자비중"]]
for n in NAMES:
    m = B[n]["periods"][ALL]; t = B[n]["trade"]
    rows.append([n, pc(m["CAGR"]), pc(m["vol"]), f"{m['sharpe']:.2f}", pc(m["MDD"], 0), pc(m["alpha"]), f"{m['alpha_t']:.1f}", f"{m['beta']:.2f}", f"{t['n']:,}", pc(t["win"], 0),
                 f"{t['avg_R']:.3f}", f"{t['avg_hold']:.0f}", f"{B[n]['turnover']:.0f}배", pc(B[n]["exposure"], 0)])
rows.append(["KOSPI", pc(bm["bm_CAGR"]), "", f"{bm['bm_sharpe']:.2f}", pc(bm["bm_MDD"], 0), "", "", "1.00", "", "", "", "", "", ""])
table(rows, size=7)
pic(RES / "fig_equity.png", 15.5)
para("기간별 성과 (CAGR / 샤프 / MDD)", bold=True, size=9.5)
prow = [["기간"] + NAMES + ["KOSPI"]]
for p in B["매매법1"]["periods"]:
    prow.append([p] + [f"{pc(B[n]['periods'][p]['CAGR'])} / {B[n]['periods'][p]['sharpe']:.2f} / {pc(B[n]['periods'][p]['MDD'],0)}" for n in NAMES] +
                [f"{pc(B['매매법1']['periods'][p]['bm_CAGR'])} / {B['매매법1']['periods'][p]['bm_sharpe']:.2f} / {pc(B['매매법1']['periods'][p]['bm_MDD'],0)}"])
table(prow, size=7.5)
yk = EQ["KOSPI"].groupby(EQ.index.year).last(); ykr = pd.concat([pd.Series([EQ["KOSPI"].iloc[0]]), yk.iloc[:-1]]).values
ky = pd.Series(yk.values / ykr - 1, index=yk.index)
yr = [["연도"] + NAMES + ["KOSPI"]]
for y in sorted(B["매매법1"]["yearly"], key=int):
    yr.append([y] + [pc(B[n]["yearly"][y]) for n in NAMES] + [pc(ky[int(y)])])
para("연도별 수익률 (2026은 10-22까지)", bold=True, size=9.5)
table(yr, size=7)
para("원천별 거래 통계 (평균 R / 거래 수): R=1999~2018-07 원본 보정, K=kor_price, D=상장폐지 종가-only(체결 근사)", bold=True, size=9.5)
table([["", "R", "K", "D"]] + [[n] + [f"{M['by_src'][n][s]['avg_R']:+.3f} / {M['by_src'][n][s]['n']:,}" for s in ("R", "K", "D")] for n in NAMES], size=8)

# ───────── 강건성
h("5. 강건성 검증")
para("① 우연 검증 — 같은 종목·같은 해에서 진입일만 무작위(10회), 손절·익절·청산 규칙 동일", bold=True, size=9.5)
table([["", "지표", "실제 신호", "무작위 평균 ± 표준편차", "신호−무작위 (표준편차 단위)"]] +
      [[n, k_name, f"{act:.3f}", f"{rnd[n][k][0]:.3f} ± {rnd[n][k][1]:.3f}", f"{(act - rnd[n][k][0]) / rnd[n][k][1]:+.1f}"]
       for n in ("매매법1", "매매법2") for k, k_name, act in (("avg_R", "거래당 평균 R", B[n]["trade"]["avg_R"]), ("CAGR", "CAGR", B[n]["periods"][ALL]["CAGR"]),
                                                              ("sharpe", "샤프", B[n]["periods"][ALL]["sharpe"]), ("win", "승률", B[n]["trade"]["win"]))], size=8)
pic(RES / "fig_random.png", 14.5)
para("해석: 신호로 진입한 쪽이 무작위보다 좋지 않다(양의 값이 우위). 손익은 진입 신호보다 손절·익절·비용 구조에서 결정되고, 신호는 우위를 더하지 못한다. 무작위 진입도 평균 R이 음수라 이 청산 규칙 자체가 비용 후 기대값이 음이다. 포트폴리오 CAGR 차이(신호 쪽이 더 나쁨)에는 자본 제약 효과가 섞여 있다: 골든크로스는 시장 전반에 몰려 나오는데, 동시 20종목 제한 때문에 포트폴리오는 신호가 몰린 날의 일부와 한산한 날의 신호를 주로 담게 되고, 채택된 거래의 평균 순수익률은 약 −1.2%로 전체 신호 평균(약 −0.3%~−0.4%)보다 낮았다(우선순위를 무작위·유동성 오름차순으로 바꾸거나 동시 보유를 60종목으로 늘려도 같은 결과; 거래당 위험을 0.2%로 낮추면 CAGR −4.1%로 손실만 줄어듦). 무작위 표본 10회라 표준편차 추정은 거칠다.", size=9)
para("② 비용 2배", bold=True, size=9.5)
table([["", "전체 CAGR / 샤프", "개발 2000~2012", "검증 2013~2026", "거래당 평균 R"]] + [[n, f"{pc(M['cost2x'][n]['periods'][ALL]['CAGR'])} / {M['cost2x'][n]['periods'][ALL]['sharpe']:.2f}", pc(M['cost2x'][n]['periods']['개발 2000~2012']['CAGR']), pc(M['cost2x'][n]['periods']['검증 2013~2026']['CAGR']), f"{M['cost2x'][n]['trade']['avg_R']:.3f}"] for n in NAMES], size=8)
para("③ 파라미터 한 개씩 변경 (CAGR / 거래당 평균 R / 검증기간 CAGR)", bold=True, size=9.5)
table([["변경", "매매법1", "매매법2"]] + [[n, *[f"{pc(M['sensitivity'][n][k]['CAGR'])} / {M['sensitivity'][n][k]['avg_R']:+.3f} / {pc(M['sensitivity'][n][k]['CAGR_oos'])}" for k in ("매매법1", "매매법2")]] for n in names], widths=[3.5, 7, 7], size=7.5)
pic(RES / "fig_sens.png", 14.5)
para("④ 개발·검증 분리: 개발(2000~2012)과 검증(2013~2026)에서 부호가 같다(매매법 1은 둘 다 음수). 매매법 2의 개발기간 양(+)은 유의하지 않고(알파 t 0.1) 검증기간에 음수(알파 t −3.1)로 바뀌어, 개발기간 성과가 표본 밖에서 유지되지 않았다.", size=9)
para("⑥ 종목 선택 기준 점검 — 같은 날 신호가 남는 자리보다 많을 때 고르는 기준과 보유 한도", bold=True, size=9.5)
d1 = S["매매법1"]
para(f"기본 규칙은 같은 진입일의 후보를 20일 평균 거래대금이 큰 순으로 정렬해 자리가 있는 만큼 담고 나머지는 건너뛴다(종목코드순이나 거래대금 작은 순이 아니다). 매매법 1 신호 {d1['n_trades']:,}건 중 진입 신호가 있던 날은 {d1['signal_days']:,}일이고 하루 20개를 넘긴 날이 {d1['days_gt20']:,}일(하루 최대 {d1['max_per_day']}개)이라 선택 기준이 결과에 영향을 줄 수 있어 아래와 같이 바꿔 다시 돌렸다.", size=9)
def cell(v): return f"{pc(v['CAGR'])} / {v['sharpe']:.2f} / {pc(v['MDD'],0)}"
rows6 = [["선택·보유 기준 (CAGR / 샤프 / MDD)"] + NAMES]
rows6.append(["거래대금 큰 순 (기본, 동시 20종목)"] + [cell(S[n]["liq_desc"]) for n in NAMES])
rows6.append(["거래대금 작은 순"] + [cell(S[n]["liq_asc"]) for n in NAMES])
rows6.append(["종목코드 순 (임의 순서 참고)"] + [cell(S[n]["code_order"]) for n in NAMES])
rows6.append(["무작위 (시드 고정 10회 평균, ±표준편차)"] + [f"{pc(S[n]['random10']['CAGR'][0])}±{pc(S[n]['random10']['CAGR'][1])} / {S[n]['random10']['sharpe'][0]:.2f} / {pc(S[n]['random10']['MDD'][0],0)}" for n in NAMES])
rows6.append(["동시 60종목 (거래대금 큰 순)"] + [cell(S[n]["maxpos60"]) for n in NAMES])
rows6.append(["보유 제한 없음 · 모든 신호 동일비중 · 현금 제약 없음"] + [cell(S[n]["equal_unlimited"]) for n in NAMES])
table(rows6, widths=[6.2, 3.7, 3.7, 3.7], size=7.5)
eu = S["매매법1"]["equal_unlimited"]
para(f"결과: 모든 기준에서 세 전략 CAGR이 음수이고 부호가 바뀌는 경우가 없다. 종목 선택 기준을 바꿔도 결론 동일. (종목코드 순에서 매매법 1이 덜 나빠 보이는 것은 임의 순서가 우연히 유리했던 것으로 보이며 무작위 분포의 상단(평균보다 약 +1.9표준편차)일 뿐 의미 있는 기준이 아니다. 보유 제한 없는 동일비중에서는 열려 있는 포지션이 평균 {eu['avg_positions']:.0f}개로 거래일의 {eu['invested_days']*100:.0f}%에 투자 상태였고, 현금 제약이 없어 현실 가능한 포트폴리오는 아니다.)", size=9)
um = U["base"]
para("⑤ 미국 참고 (S&P500 현재 구성종목 + 상장폐지 종목 없음 → 생존편향으로 상향 편향, 1995~2026, 왕복 0.1%)", bold=True, size=9.5)
table([["", "CAGR", "샤프", "MDD", "거래 수", "승률", "평균 R", "평균 수익률/거래", "SPY CAGR / MDD"]] + [[n, pc(um[n]["periods"]["전체"]["CAGR"]), f"{um[n]['periods']['전체']['sharpe']:.2f}", pc(um[n]["periods"]["전체"]["MDD"], 0), f"{um[n]['trade']['n']:,}", pc(um[n]["trade"]["win"], 0), f"{um[n]['trade']['avg_R']:+.3f}", pc(um[n]["trade"]["avg_ret"], 2), f"{pc(um[n]['periods']['전체']['bm_CAGR'])} / {pc(um[n]['periods']['전체']['bm_MDD'],0)}"] for n in NAMES], size=7.5)

# ───────── 한계
h("6. 한계")
bullet("상장폐지 종목의 2018-07 이후는 종가만 있어(KRX 차단으로 원본 중단) 장중 저가 기반 손절·상한가 판정이 근사이고 유동성은 시가총액으로 대용했다. 전체 거래의 약 3%(D)이며 원천별 표에서 결론은 같다.")
bullet("수정주가는 등락률 기반 탐지(1% 허용오차)라 작은 권리락·배당 보정은 놓칠 수 있고, kor_price와 2% 넘게 어긋난 종목이 157개 있다. 구간 분리로 단절을 가로지르는 거래는 막았지만 단절 지점에서 정상 신호 일부가 사라졌을 수 있다.")
bullet("거래세·농특세율은 나무위키 기준이며 법령 원문은 확인하지 못했다. 1999~2018 구간은 각주 기반 추정, 시장 구분 미상 종목은 코스닥 세율 적용(보수적). 수수료·슬리피지는 가정값이며 호가 단위·시장충격은 반영하지 않았다.")
bullet("상한가 매수 불가·거래정지는 단순 규칙(시가가 상한가 근접, 거래량 0)으로 처리했고, 하한가 매도 불가는 별도로 반영하지 않았다(손실 과소평가 가능).")
bullet("유휴 현금 수익률을 0%로 두었다(투자비중 평균 매매법 1 89%, 매매법 2 57%). 샤프는 CD 91일 금리 대비, 알파는 KOSPI 대비 OLS(HAC).")
bullet("규칙 해석(다이버전스 정의, 지지 판정 '50~56 반등', 손절폭 3~15% 필터, 2R)은 합의한 기본값이며 최적화하지 않았다. 다른 해석에서는 결과가 달라질 수 있으나 파라미터 10종 변경에서 방향은 같았다.")
h("7. 결론")
para(f"영상에서 소개된 두 매매법을 상장폐지 종목을 포함한 한국 주식 27년 데이터에 그대로 적용하면 비용 후 우위가 없었다. 신호 진입은 무작위 진입보다 낫지 않았고(매매법 1은 오히려 낮음), 비용 2배·기간 분할·파라미터 변경에서도 같은 방향이다. 이 규칙(10일 저점 손절, 2R 목표, 모멘텀 진입)은 비용 전 기대값이 거의 0이고 승률이 약 1/3이라 거래비용과 자본 제약 하의 신호 군집 효과를 이기지 못했다. 영상의 높은 승률 주장은 이 데이터에서 재현되지 않았다. 후속으로는 추세 필터(시장 국면)·청산 규칙 변경을 별도 가설로 사전 정의한 뒤 같은 절차로 검증하는 것이 적절하다.", size=9.5)

out = HERE / "MACD_RSI_스윙_백테스트_보고서.docx"
doc.save(out); print(out)

