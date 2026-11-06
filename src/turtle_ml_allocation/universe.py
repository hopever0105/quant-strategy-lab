"""유니버스 정의 (결과를 보기 전에 고정, 2026-11-06).

- 4대 자산군(주식·채권·실물·통화)을 빠짐없이 넣는다. 성과가 나빴던 자산군도 뺀 것이 없다.
- 다우존스는 SPY와 사실상 같은 자산이라 넣지 않았다. VIX는 살 수 있는 자산이 아니라 ML 입력 변수로만 쓴다.
- ETF 상장 전 구간은 같은 자산을 추종하는 인덱스 펀드·지수로 이어 붙인다(proxy). 이어 붙일 대상이 없으면
  ETF 상장 후부터 유니버스에 들어온다(시점 기준 편입).
- 원유·은·농산물은 선물 연속가격(CL=F 등)을 이어 붙이면 만기 교체 때 가격이 튀어 가짜 돌파가 생기므로 쓰지 않았다.
  금(GC=F)은 보유 비용이 작아 이어 붙였고, 겹치는 기간의 추적 오차를 보고서에 적는다.
"""

# key, 이름, 자산군, ETF, 상장 전 대용(없으면 None), 대용 변환('inv' = 1/가격)
UNIVERSE = [
    # 주식 (10)
    ("SPY",  "미국 대형주(S&P500)",   "주식", "SPY",  "VFINX",    None),
    ("QQQ",  "미국 나스닥100",         "주식", "QQQ",  None,       None),
    ("IWM",  "미국 소형주(러셀2000)",  "주식", "IWM",  "NAESX",    None),
    ("VGK",  "유럽 주식",              "주식", "VGK",  "VEURX",    None),
    ("EWJ",  "일본 주식",              "주식", "EWJ",  None,       None),
    ("EEM",  "신흥국 주식",            "주식", "EEM",  "VEIEX",    None),
    ("EWY",  "한국 주식(MSCI Korea)",  "주식", "EWY",  None,       None),
    ("FXI",  "중국 대형주(홍콩 상장)", "주식", "FXI",  None,       None),
    ("INDA", "인도 주식",              "주식", "INDA", None,       None),
    ("ILF",  "중남미 주식",            "주식", "ILF",  None,       None),
    # 채권 (6)
    ("SHY",  "미국채 1~3년",           "채권", "SHY",  "VFISX",    None),
    ("IEF",  "미국채 7~10년",          "채권", "IEF",  "VFITX",    None),
    ("TLT",  "미국채 20년 이상",       "채권", "TLT",  "VUSTX",    None),
    ("TIP",  "미국 물가연동채",        "채권", "TIP",  "VIPSX",    None),
    ("LQD",  "미국 투자등급 회사채",   "채권", "LQD",  "VWESX",    None),
    ("HYG",  "미국 하이일드 회사채",   "채권", "HYG",  "VWEHX",    None),
    # 실물 (6)
    ("GLD",  "금",                     "실물", "GLD",  "GC=F",     None),
    ("SLV",  "은",                     "실물", "SLV",  None,       None),
    ("USO",  "원유(WTI)",              "실물", "USO",  None,       None),
    ("DBC",  "원자재 종합",            "실물", "DBC",  "PCRIX",    None),
    ("DBA",  "농산물",                 "실물", "DBA",  None,       None),
    ("VNQ",  "미국 리츠",              "실물", "VNQ",  "VGSIX",    None),
    # 통화 (3) — 미국 상장 ETF라 롱만 해도 달러 약세 추세는 유로·엔 매수로 잡힌다
    ("UUP",  "달러 인덱스",            "통화", "UUP",  "DX-Y.NYB", None),
    # 환율 대용은 FRED 뉴욕 정오 고시 환율(야후 EURUSD=X·JPY=X는 주간 상관 0.83으로 품질이 낮아 바꿈)
    ("FXE",  "유로",                   "통화", "FXE",  "FRED:DEXUSEU", None),
    ("FXY",  "엔화",                   "통화", "FXY",  "FRED:DEXJPUS", "inv"),
]

GROUPS = ["주식", "채권", "실물", "통화"]
KEYS = [u[0] for u in UNIVERSE]
NAME = {u[0]: u[1] for u in UNIVERSE}
GROUP = {u[0]: u[2] for u in UNIVERSE}

MACRO = {"VIX": "FRED:VIXCLS", "T10Y2Y": "FRED:T10Y2Y", "TBILL": "FRED:DTB3",
         # 변수 확장(사후 점검, 2026-11-06 추가): 신용 스프레드(무디스 Baa − 미국채 10년), 10년 금리 수준
         # (하이일드 스프레드 BAMLH0A0HYM2는 FRED에서 최근 3년치만 제공돼 쓰지 못함)
         "BAA10Y": "FRED:BAA10Y", "DGS10": "FRED:DGS10"}
