"""다우존스 산업평균(DJIA) 편입 이력 2010-01 ~ 2026-10.

출처: Wikipedia 'Historical components of the Dow Jones Industrial Average' (편입·편출 발효일).
발효일 e의 변경은 e 직전 거래일 종가에 반영한다(지수도 e 개장 전에 바뀜).

티커는 Yahoo 시세 이력 기준이다.
- KFT(Kraft Foods Inc, 2008~2012)는 법적 승계회사 Mondelez(MDLZ) 이력을 쓴다.
- UTX(United Technologies)는 2020-04 Raytheon과 합병 후 RTX로 이름만 바뀌어 같은 RTX 이력을 쓴다.
- DuPont(2017-09부터 DowDuPont)은 Yahoo 'DD' 이력이 이어져 있다(2019-04-02 Dow Inc로 교체).
- 옛 Alcoa(2010-01 ~ 2013-09)는 무료 일별 시세가 없어 두 포트폴리오 모두에서 제외한다(그 기간 29종목).
- Walgreens(WBA, 2018-06 ~ 2024-02)는 2025년 상장폐지로 일별 시세가 없어 월말 실제값(시가총액·총수익)을 쓰고,
  월 안의 일별 경로만 나머지 종목 평균 움직임으로 잇는다.
"""
import pandas as pd

START = '2010-01-01'

INITIAL = ['MMM', 'AA_OLD', 'AXP', 'T', 'BAC', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DD', 'XOM', 'GE', 'HPQ', 'HD',
           'INTC', 'IBM', 'JNJ', 'JPM', 'KFT', 'MCD', 'MRK', 'MSFT', 'PFE', 'PG', 'TRV', 'RTX', 'VZ', 'WMT', 'DIS']

# (발효일, 편출, 편입)
CHANGES = [
    ('2012-09-24', ['KFT'], ['UNH']),
    ('2013-09-23', ['AA_OLD', 'BAC', 'HPQ'], ['GS', 'NKE', 'V']),
    ('2015-03-19', ['T'], ['AAPL']),
    ('2018-06-26', ['GE'], ['WBA']),
    ('2019-04-02', ['DD'], ['DOW']),
    ('2020-08-31', ['XOM', 'PFE', 'RTX'], ['AMGN', 'HON', 'CRM']),
    ('2024-02-26', ['WBA'], ['AMZN']),
    ('2024-11-08', ['INTC', 'DOW'], ['NVDA', 'SHW']),
    ('2026-06-29', ['VZ'], ['GOOGL']),
]

# 시세 출처 티커 (Yahoo/DB)
PRICE_TICKER = {'KFT': 'MDLZ'}
EXCLUDED = {'AA_OLD'}            # 일별 시세 없음 → 제외
MONTHLY_ONLY = {'WBA'}           # 월말 실제값 + 월 안 경로 보간

# companiesmarketcap.com 페이지 이름 (월말 시가총액)
CMC_SLUG = {
    'MMM': '3m', 'AXP': 'american-express', 'T': 'att', 'BAC': 'bank-of-america', 'BA': 'boeing', 'CAT': 'caterpillar',
    'CVX': 'chevron', 'CSCO': 'cisco', 'KO': 'coca-cola', 'DD': 'dupont-de-nemours', 'XOM': 'exxon-mobil',
    'GE': 'general-electric', 'HPQ': 'hp', 'HD': 'home-depot', 'INTC': 'intel', 'IBM': 'ibm',
    'JNJ': 'johnson-and-johnson', 'JPM': 'jp-morgan-chase', 'KFT': 'mondelez', 'MCD': 'mcdonald', 'MRK': 'merck',
    'MSFT': 'microsoft', 'PFE': 'pfizer', 'PG': 'procter-and-gamble', 'TRV': 'the-travelers-companies',
    'RTX': 'raytheon-technologies', 'VZ': 'verizon', 'WMT': 'walmart', 'DIS': 'walt-disney', 'UNH': 'united-health',
    'GS': 'goldman-sachs', 'NKE': 'nike', 'V': 'visa', 'AAPL': 'apple', 'WBA': 'walgreens-boots-alliance',
    'DOW': 'dow', 'AMGN': 'amgen', 'HON': 'honeywell', 'CRM': 'salesforce', 'AMZN': 'amazon', 'NVDA': 'nvidia',
    'SHW': 'sherwin-williams', 'GOOGL': 'alphabet-google',
}


def all_members():
    s = set(INITIAL)
    for _, out, inn in CHANGES:
        s |= set(inn)
    return sorted(s)


def price_ticker(m):
    return PRICE_TICKER.get(m, m)


def membership(dates):
    """날짜 × 종목 True/False 편입 여부 (제외 종목 포함, 30종목 검증용)"""
    dates = pd.DatetimeIndex(dates)
    mem = pd.DataFrame(False, index=dates, columns=all_members())
    cur = set(INITIAL)
    bounds = [(pd.Timestamp(START), None, None)] + [(pd.Timestamp(e), o, i) for e, o, i in CHANGES]
    for k, (e, out, inn) in enumerate(bounds):
        if out is not None:
            cur = (cur - set(out)) | set(inn)
        assert len(cur) == 30, (e, len(cur))
        nxt = bounds[k + 1][0] if k + 1 < len(bounds) else dates[-1] + pd.Timedelta(days=1)
        mask = (dates >= e) & (dates < nxt)
        mem.loc[mask, sorted(cur)] = True
    return mem
