# -*- coding: utf-8 -*-
"""
MLP 자산배분: ETF 10개 중 다음 달 상위 3개 (추가 전략 1, 1-2, 1-3)
- 딥러닝 전략 실험 묶음(deep_learning_lab)의 코드. 공부 파일 끝에 붙여 돌린 '추가 전략' 셀만 모았다.
- 실행: Spyder(conda ta_env), #%% 셀 단위. 맨 위 [1] 환경 설정 셀 → 원하는 [추가 전략] 셀.
- 결과: ../results/<전략 이름>/ 에 결과요약.md·지표.json·csv·png 자동 저장 (다시 돌리면 덮어씀).
- DB 접속 정보는 .env 파일(환경변수 QUANT_ENV_PATH 로 경로 지정 가능)에서 읽고 코드에 쓰지 않는다.
- 포트폴리오용 연구 코드. 실제 주문·매매 기능 없음.
"""
#%% [1] 환경 설정, 공통 라이브러리 임포트 및 로컬 MySQL DB 연결
import os
import math
import random
import urllib.parse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
#%%
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Input, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

warnings.filterwarnings("ignore")

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
if TF_AVAILABLE:
    tf.random.set_seed(SEED)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

CODE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
ENV_PATH = os.environ.get("QUANT_ENV_PATH", os.path.join(CODE_DIR, "..", "..", "..", "..", "..", "quant", ".env"))
load_dotenv(dotenv_path=ENV_PATH)

DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = urllib.parse.quote_plus(os.getenv("DB_PASSWORD", ""))
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "")

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4",
    pool_pre_ping=True,
)


def load_sql(query: str, fallback_df: pd.DataFrame = None) -> pd.DataFrame:
    """SQL 쿼리 실행 및 빈 데이터·중복 컬럼 방어 함수"""
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        if df is None or df.empty:
            return fallback_df.copy() if fallback_df is not None else pd.DataFrame()
        df = df.loc[:, list(dict.fromkeys(df.columns))]
        return df
    except Exception as e:
        print(f"[DB 연결/쿼리 오류] {e} -> 폴백 데이터 사용")
        return fallback_df.copy() if fallback_df is not None else pd.DataFrame()


print(f"[초기화 완료] TensorFlow/Keras 사용 가능 여부: {TF_AVAILABLE}")


#%% [추가 전략 1-2] MLP 자산배분 개선: 회전율 줄이기·모멘텀 앙상블·거시 특성·순위 정답 (추가전략_다음할일.md 의 추가1)
# 무엇: 추가 전략 1 결과(샤프 0.60 < 모멘텀 0.68, 월 회전율 97%)에서 나온 할 일을 한 번에 실험한다.
#  - 모델 4개: MLP 기본 / 순위 정답 / 거시 특성(금리·금리차·신용가산금리·VIX·달러 + 자산 구분) / 거시 + 순위 정답
#  - 점수 7개(MLP 4 + 모멘텀 + 앙상블 2) x 규칙 3개(매달 / 3개월 평균 점수 / 3개월 보유) = 21개 조합 + 같은 비중·SPY
# 실행: 맨 위 [1] 환경 설정 셀을 먼저 실행한 뒤 이 셀 실행 (Ctrl+Enter). 모델 4개 x 15년 x 시드 3개라 20~30분
# 결과: 06_퀀트공부/딥러닝/결과/추가1-2_MLP_회전율_앙상블_거시_순위정답/ 에 결과요약.md·지표.json·csv·그림 자동 저장
# ----- 결과 자동 저장 도구 (각 추가 전략 셀이 혼자 실행돼도 되도록 셀마다 같은 함수를 둠) -----
import json
import datetime

RESULT_ROOT = os.path.join(CODE_DIR, "..", "results")


def perf_stats(ret, periods_per_year):
    """수익률 시계열 -> CAGR·변동성·샤프·MDD·누적수익·플러스 비율"""
    ret = pd.Series(ret).dropna()
    if len(ret) < 2 or ret.std() == 0:
        return {"CAGR": np.nan, "변동성": np.nan, "샤프": np.nan, "MDD": np.nan,
                "누적수익": np.nan, "플러스비율": np.nan, "기간수": int(len(ret))}
    cum = (1 + ret).cumprod()
    years = len(ret) / periods_per_year
    return {
        "CAGR": float(cum.iloc[-1] ** (1 / years) - 1),
        "변동성": float(ret.std() * np.sqrt(periods_per_year)),
        "샤프": float(ret.mean() / ret.std() * np.sqrt(periods_per_year)),
        "MDD": float((cum / cum.cummax() - 1).min()),
        "누적수익": float(cum.iloc[-1] - 1),
        "플러스비율": float((ret > 0).mean()),
        "기간수": int(len(ret)),
    }


def yearly_table(ret_dict):
    """{이름: 수익률 시계열} -> 연도별 수익률 표"""
    out = {}
    for k, r in ret_dict.items():
        r = pd.Series(r).dropna()
        out[k] = (1 + r).groupby(r.index.year).prod() - 1
    tab = pd.DataFrame(out)
    tab.index.name = "연도"
    return tab


def md_table(df, pct_cols=(), digits=3):
    """tabulate 없이 마크다운 표 만들기 (pct_cols="all"이면 전부 % 표시)"""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([str(df.index.name or "")] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = []
        for c, v in zip(df.columns, row.values):
            if isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    cells.append("-")
                elif pct_cols == "all" or c in pct_cols:
                    cells.append(f"{v * 100:.2f}%")
                elif float(v).is_integer() and abs(v) >= 1:
                    cells.append(str(int(v)))          # 기간수·매매 수 같은 개수
                else:
                    cells.append(f"{v:.{digits}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def stats_frame(ret_dict, periods_per_year):
    """여러 전략의 성과 지표를 한 표로"""
    tab = pd.DataFrame({k: perf_stats(v, periods_per_year) for k, v in ret_dict.items()}).T
    tab.index.name = "전략"
    return tab


def save_result(tag, title, sections, tables_csv, figs, metrics_json):
    """결과요약.md + 지표.json + csv + png 를 결과/<tag>/ 에 저장 (다시 돌리면 덮어씀)"""
    out_dir = os.path.join(RESULT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    md = [f"# {title}", "",
          f"- 실행 시각: {now}",
          f"- TensorFlow 사용: {TF_AVAILABLE} (False면 신경망 대신 대체 모델로 낸 결과)", ""]
    for head, body in sections:
        md += [f"## {head}", "", body, ""]
    md += ["## 같이 저장된 파일", ""]
    md += [f"- {n}.csv" for n in tables_csv] + [f"- {n}.png" for n in figs] + ["- 지표.json"]
    with open(os.path.join(out_dir, "결과요약.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    for n, df in tables_csv.items():
        df.to_csv(os.path.join(out_dir, f"{n}.csv"), encoding="utf-8-sig")
    for n, fig in figs.items():
        fig.savefig(os.path.join(out_dir, f"{n}.png"), dpi=120, bbox_inches="tight")
    metrics_json = dict(metrics_json)
    metrics_json["실행시각"] = now
    metrics_json["TF_AVAILABLE"] = bool(TF_AVAILABLE)
    with open(os.path.join(out_dir, "지표.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics_json, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n[결과 저장 완료] {out_dir}")
    print("  -> 결과요약.md 를 열면 이 실행의 설정·성과·연도별·진단이 전부 들어 있습니다.")
# ----- 결과 저장 도구 끝 -----

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from scipy.stats import spearmanr

# ----- 설정 (바꿔 보면서 다시 돌려 보기) -----
A2_TAG = "추가1-2_MLP_회전율_앙상블_거시_순위정답"
A2_ASSETS = ["SPY", "IEV", "EWJ", "EEM", "TLT", "IEF", "IYR", "RWX", "GLD", "DBC"]
A2_TOPK = 3
A2_COST = 0.001
A2_TEST_START = 2012
A2_SEEDS = [SEED, SEED + 1, SEED + 2]
# 거시 특성 (macro_series 지표ID): 금리 수준·변화, 장단기 금리차, 신용 가산금리, VIX, 달러
A2_MACRO = {"DGS10": "미국채10년", "T10Y2Y": "장단기금리차", "BAA10Y": "신용가산금리", "VIXCLS": "VIX", "DTWEXBGS": "달러지수"}

# ----- 1) 데이터·특성 (추가 전략 1과 같음) -----
a2_raw = load_sql("SELECT * FROM sample_etf")
a2_raw["Date"] = pd.to_datetime(a2_raw["Date"])
a2_px = a2_raw.set_index("Date").sort_index()[A2_ASSETS].astype(float)
a2_ret_d = a2_px.pct_change(fill_method=None)


def a2_month_end(df):
    try:
        return df.resample("ME").last()
    except ValueError:
        return df.resample("M").last()


a2_px_m = a2_month_end(a2_px)
if (a2_px_m.index[-1] - a2_px.index.max()).days > 5:
    a2_px_m = a2_px_m.iloc[:-1]
a2_feat = {
    "ret1": a2_px_m.pct_change(1, fill_method=None),
    "ret3": a2_px_m.pct_change(3, fill_method=None),
    "ret6": a2_px_m.pct_change(6, fill_method=None),
    "ret12": a2_px_m.pct_change(12, fill_method=None),
    "vol3": a2_month_end(a2_ret_d.rolling(63, min_periods=50).std() * np.sqrt(252)),
    "ma10_gap": a2_px_m / a2_px_m.rolling(10).mean() - 1,
}
a2_fwd = a2_px_m.pct_change(1, fill_method=None).shift(-1)


def a2_to_long(df, name):
    s = df.copy()
    s.index.name = "date"
    s.columns.name = None
    return s.reset_index().melt(id_vars="date", var_name="asset", value_name=name).set_index(["date", "asset"])[name]


a2_panel = pd.concat([a2_to_long(v, k) for k, v in a2_feat.items()] + [a2_to_long(a2_fwd, "fwd")], axis=1)
A2_FEATS = list(a2_feat.keys())
a2_panel = a2_panel.dropna(subset=A2_FEATS).sort_index()
a2_panel = a2_panel[a2_panel.groupby(level="date")["ret1"].transform("count") >= 5]
for f in A2_FEATS:
    a2_panel[f + "_rank"] = a2_panel.groupby(level="date")[f].rank(pct=True) - 0.5
A2_XBASE = A2_FEATS + [f + "_rank" for f in A2_FEATS]
a2_panel["target"] = a2_panel["fwd"] - a2_panel.groupby(level="date")["fwd"].transform("mean")
a2_panel["target_rank"] = a2_panel.groupby(level="date")["fwd"].rank(pct=True) - 0.5     # 순위 정답

# 거시 특성: 그달 말까지 나온 값 (월말 마지막 값), 변화는 3개월 차이. 자산마다 반응이 달라서 자산 구분(원-핫)도 같이 넣음
a2_mac_raw = load_sql(
    f"SELECT 지표ID, 날짜, 값 FROM macro_series WHERE 지표ID IN ({', '.join(repr(k) for k in A2_MACRO)})"
)
a2_mac_raw["날짜"] = pd.to_datetime(a2_mac_raw["날짜"])
a2_mac_raw["값"] = pd.to_numeric(a2_mac_raw["값"], errors="coerce")
a2_mac_d = a2_mac_raw.pivot_table(index="날짜", columns="지표ID", values="값").sort_index().ffill()
a2_mac_m = a2_month_end(a2_mac_d).reindex(a2_px_m.index).ffill()
a2_mac = pd.DataFrame(index=a2_px_m.index)
a2_mac["m_DGS10"] = a2_mac_m.get("DGS10")
a2_mac["m_DGS10_chg3"] = a2_mac_m.get("DGS10").diff(3)
a2_mac["m_T10Y2Y"] = a2_mac_m.get("T10Y2Y")
a2_mac["m_BAA10Y"] = a2_mac_m.get("BAA10Y")
a2_mac["m_VIX"] = a2_mac_m.get("VIXCLS")
a2_mac["m_USD_chg3"] = np.log(a2_mac_m.get("DTWEXBGS")).diff(3)
a2_mac["m_USD_있음"] = a2_mac["m_USD_chg3"].notna().astype(float)   # 달러지수는 2006년부터 -> 그 전은 0으로 채우고 표시
a2_mac["m_USD_chg3"] = a2_mac["m_USD_chg3"].fillna(0.0)
A2_XMAC = list(a2_mac.columns)
a2_panel = a2_panel.join(a2_mac, on="date")
for a in A2_ASSETS:
    a2_panel["is_" + a] = (a2_panel.index.get_level_values("asset") == a).astype(float)
A2_XID = ["is_" + a for a in A2_ASSETS]
a2_mac_ok = a2_panel[A2_XMAC].notna().all(axis=1)
print(f"[추가1-2] 패널 {len(a2_panel)}개, 거시 특성 있는 표본 {int(a2_mac_ok.sum())}개 ({a2_panel[a2_mac_ok].index.get_level_values('date').min().date()}~)")


# ----- 2) MLP (추가 전략 1과 같은 구조) -----
def a2_fit_predict(x_tr, y_tr, x_te, seed):
    if TF_AVAILABLE:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Input, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        m = Sequential([Input(shape=(x_tr.shape[1],)), Dense(32, activation="relu"), Dropout(0.2),
                        Dense(16, activation="relu"), Dense(1)])
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        m.fit(x_tr, y_tr, epochs=100, batch_size=32, validation_split=0.15, verbose=0,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])
        return m.predict(x_te, verbose=0).ravel()
    m = MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, learning_rate_init=1e-3, max_iter=500,
                     early_stopping=True, validation_fraction=0.15, n_iter_no_change=10, random_state=seed)
    m.fit(x_tr, y_tr)
    return m.predict(x_te)


# 모델 4가지: (특성, 정답)
A2_MODELS = {
    "MLP 기본": (A2_XBASE, "target"),
    "MLP 순위정답": (A2_XBASE, "target_rank"),
    "MLP 거시": (A2_XBASE + A2_XMAC + A2_XID, "target"),
    "MLP 거시+순위정답": (A2_XBASE + A2_XMAC + A2_XID, "target_rank"),
}
a2_dates = a2_panel.index.get_level_values("date")
a2_lab = a2_panel["fwd"].notna().values
a2_train_log = []
for mname, (cols, tgt) in A2_MODELS.items():
    preds = []
    for yr in range(A2_TEST_START, a2_dates.max().year + 1):
        use = a2_panel[cols].notna().all(axis=1).values
        tr = a2_panel[(a2_dates.year < yr) & a2_lab & use]
        te = a2_panel[(a2_dates.year == yr) & use]
        if len(tr) < 200 or len(te) == 0:
            continue
        sc = StandardScaler().fit(tr[cols])
        y_sd = tr[tgt].std()
        p = np.mean([a2_fit_predict(sc.transform(tr[cols]), tr[tgt].values / y_sd, sc.transform(te[cols]), s)
                     for s in A2_SEEDS], axis=0)
        preds.append(pd.Series(p, index=te.index))
        a2_train_log.append({"모델": mname, "연도": yr, "학습표본": len(tr), "예측표본": len(te)})
    a2_panel["p_" + mname] = pd.concat(preds)
    print(f"  {mname} 완료")

a2_test = a2_panel.dropna(subset=["p_MLP 기본"]).copy()
# 점수 만들기: 같은 달 안 순위(-0.5~0.5)로 바꿔서 합치기 쉽게
for mname in A2_MODELS:
    a2_test["r_" + mname] = a2_test.groupby(level="date")["p_" + mname].rank(pct=True) - 0.5
a2_test["r_모멘텀"] = a2_test.groupby(level="date")["ret12"].rank(pct=True) - 0.5
a2_test["r_앙상블(MLP기본+모멘텀)"] = a2_test["r_MLP 기본"] + a2_test["r_모멘텀"]
a2_test["r_앙상블(MLP거시+모멘텀)"] = a2_test["r_MLP 거시"] + a2_test["r_모멘텀"]
A2_SCORES = [c[2:] for c in a2_test.columns if c.startswith("r_")]
# 3개월 평균 점수: 자산마다 최근 3개월 점수 평균 (점수가 천천히 바뀌게 = 회전율 줄이기)
for s in A2_SCORES:
    w = a2_test["r_" + s].unstack("asset").rolling(3, min_periods=1).mean()
    a2_test["s3_" + s] = w.stack().reindex(a2_test.index)


# ----- 3) 포트폴리오 규칙: 매달 상위3 / 3개월 평균 점수 상위3 / 3개월 보유(매달 1/3씩 교체) -----
def a2_topk(col):
    w = {}
    for d, g in a2_test.groupby(level="date"):
        top = g[col].nlargest(A2_TOPK).index.get_level_values("asset")
        w[d] = pd.Series(1.0 / A2_TOPK, index=top)
    return pd.DataFrame(w).T.reindex(columns=A2_ASSETS).fillna(0.0)


a2_next_me = pd.Series(a2_px_m.index[1:], index=a2_px_m.index[:-1])


def a2_port(w):
    fwd = a2_fwd.reindex(w.index)[A2_ASSETS]
    has = fwd.notna().any(axis=1)
    gross = (w * fwd.fillna(0.0)).sum(axis=1)
    # 보유 중 가격 변화로 비중이 움직인 뒤 다시 맞추는 것까지 회전율에 포함
    drift = w.shift(1) * (1 + fwd.shift(1).fillna(0.0))
    drift = drift.div(drift.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    turnover = (w - drift).abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()
    net = (gross - turnover * A2_COST)[has]
    net.index = pd.DatetimeIndex(a2_next_me.reindex(net.index).values)
    return net, float(turnover[has].mean())


a2_rets, a2_to = {}, {}
for s in A2_SCORES:
    w1 = a2_topk("r_" + s)
    a2_rets[f"{s} · 매달"], a2_to[f"{s} · 매달"] = a2_port(w1)
    a2_rets[f"{s} · 3개월평균점수"], a2_to[f"{s} · 3개월평균점수"] = a2_port(a2_topk("s3_" + s))
    w3 = w1.rolling(3, min_periods=1).mean()        # 매달 산 묶음을 3개월 들고 감 = 최근 3번 고른 것의 평균
    a2_rets[f"{s} · 3개월보유"], a2_to[f"{s} · 3개월보유"] = a2_port(w3)
w_ew = pd.DataFrame({d: pd.Series(1.0 / len(g), index=g.index.get_level_values("asset"))
                     for d, g in a2_test.groupby(level="date")}).T.reindex(columns=A2_ASSETS).fillna(0.0)
a2_rets["10개 같은 비중"], a2_to["10개 같은 비중"] = a2_port(w_ew)
a2_spy = a2_fwd["SPY"].reindex(w_ew.index).dropna()
a2_spy.index = pd.DatetimeIndex(a2_next_me.reindex(a2_spy.index).values)
a2_rets["SPY 보유"], a2_to["SPY 보유"] = a2_spy, 0.0
a2_stats = stats_frame(a2_rets, 12)
a2_stats["월평균 회전율"] = [a2_to[k] for k in a2_rets]
a2_ew_r = a2_rets["10개 같은 비중"]
a2_stats["같은비중 이긴 달 비율"] = [float((v - a2_ew_r.reindex(v.index)).gt(0).mean()) for v in a2_rets.values()]
a2_stats = a2_stats.sort_values("샤프", ascending=False)
a2_stats.index.name = "점수 · 규칙"


# ----- 4) 진단: 점수별 IC, 거시 모델이 시기에 따라 다르게 고르는지 -----
def a2_ic(col):
    ics = {}
    for d, g in a2_test.dropna(subset=["fwd"]).groupby(level="date"):
        if len(g) >= 5:
            ics[d] = spearmanr(g[col], g["fwd"])[0]
    return pd.Series(ics).dropna()


a2_ic_tab = {}
a2_ic_ser = {}
for s in A2_SCORES:
    for pre, lab in [("r_", "매달 점수"), ("s3_", "3개월 평균 점수")]:
        ic = a2_ic(pre + s)
        a2_ic_ser[f"{s} · {lab}"] = ic
        a2_ic_tab[f"{s} · {lab}"] = {"IC 평균": ic.mean(), "IC t값": ic.mean() / ic.std() * np.sqrt(len(ic)), "IC>0 비율": (ic > 0).mean(), "달 수": len(ic)}
a2_ic_tab = pd.DataFrame(a2_ic_tab).T.sort_values("IC t값", ascending=False)
a2_ic_tab.index.name = "점수"
a2_best = a2_stats.index[0]
a2_show = [a2_best, "MLP 기본 · 매달", "모멘텀 · 매달", "10개 같은 비중", "SPY 보유"]
a2_show = list(dict.fromkeys(k for k in a2_show if k in a2_rets))
a2_year = yearly_table({k: a2_rets[k] for k in a2_show})

# ----- 5) 그림 -----
a2_fig1, ax = plt.subplots(figsize=(11, 4.5))
for k in a2_show:
    ax.plot((1 + a2_rets[k]).cumprod(), label=k, linewidth=1.8 if k == a2_best else 1.1)
ax.set_title("추가1-2: 회전율 줄이기·앙상블·거시·순위정답 (비용 차감)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
a2_fig1.tight_layout()
plt.show()
print(md_table(a2_stats.head(12)[["CAGR", "샤프", "MDD", "월평균 회전율"]], pct_cols=("CAGR", "MDD", "월평균 회전율")))

# ----- 6) 결과 자동 저장 -----
a2_auto = [
    f"- 샤프 최고: {a2_best} ({a2_stats.loc[a2_best, '샤프']:.2f}, 회전율 {a2_stats.loc[a2_best, '월평균 회전율'] * 100:.0f}%)",
    f"- 처음 방식 'MLP 기본 · 매달' 샤프 {a2_stats.loc['MLP 기본 · 매달', '샤프']:.2f} (회전율 {a2_stats.loc['MLP 기본 · 매달', '월평균 회전율'] * 100:.0f}%) vs "
    f"'MLP 기본 · 3개월보유' {a2_stats.loc['MLP 기본 · 3개월보유', '샤프']:.2f} ({a2_stats.loc['MLP 기본 · 3개월보유', '월평균 회전율'] * 100:.0f}%) vs "
    f"'모멘텀 · 매달' {a2_stats.loc['모멘텀 · 매달', '샤프']:.2f}",
    "- 같은 규칙(매달)에서 점수별 샤프: " + " / ".join(f"{s} {a2_stats.loc[f'{s} · 매달', '샤프']:.2f}" for s in A2_SCORES),
    f"- IC t값 최고: {a2_ic_tab.index[0]} ({a2_ic_tab.iloc[0]['IC t값']:.2f})",
]
save_result(
    A2_TAG, "추가 전략 1-2: MLP 자산배분 개선 - 회전율 줄이기, 모멘텀 앙상블, 거시 특성, 순위 정답",
    [
        ("한눈에", "\n".join(a2_auto)),
        ("설정", "\n".join([
            f"- 자산·기본 특성·워크포워드는 추가 전략 1과 같음 ({A2_TEST_START}년부터 매년 1월 재학습, 시드 {len(A2_SEEDS)}개 평균)",
            "- 모델 4가지: 기본 / 순위정답(다음 달 수익 '순위'를 맞힘) / 거시(+금리·금리차·신용가산금리·VIX·달러, +자산 구분) / 거시+순위정답",
            f"- 거시 지표: {', '.join(f'{v}({k})' for k, v in A2_MACRO.items())}. 달러지수는 2006년부터라 그 전은 0으로 채우고 '있음' 표시 특성 추가",
            "- 점수 7가지: MLP 4 + 12개월 모멘텀 + 앙상블 2(MLP 순위 + 모멘텀 순위)",
            f"- 규칙 3가지: 매달 상위{A2_TOPK} / 최근 3개월 평균 점수 상위{A2_TOPK} / 3개월 보유(매달 1/3씩 교체)",
            f"- 회전율은 가격 변화로 비중이 흐트러진 것을 다시 맞추는 매매까지 포함, 한쪽 비용 {A2_COST * 100:.2f}%",
        ])),
        ("성과 전체 (샤프 높은 순, 월간, 비용 차감)", md_table(a2_stats, pct_cols=("CAGR", "변동성", "MDD", "누적수익", "플러스비율", "월평균 회전율", "같은비중 이긴 달 비율"))),
        ("점수별 순위 예측력 (IC)", md_table(a2_ic_tab, pct_cols=("IC>0 비율",))),
        ("연도별 수익률 (대표)", md_table(a2_year, pct_cols="all")),
        ("재학습 기록", md_table(pd.DataFrame(a2_train_log).set_index("모델"))),
        ("주의", "\n".join([
            "- 21개 조합을 시험했다. 최고 하나보다 같은 점수의 세 규칙이 고르게 좋은지를 봐야 과최적화를 피한다.",
            "- ETF 10개뿐이라 IC가 많이 흔들린다. t값으로 판단.",
            "- 거래비용은 가정값. 세금·환전 미반영.",
        ])),
    ],
    {"월별수익": pd.DataFrame(a2_rets), "성과": a2_stats, "IC": a2_ic_tab, "IC_월별": pd.DataFrame(a2_ic_ser),
     "연도별수익": a2_year, "점수패널": a2_test[[c for c in a2_test.columns if c.startswith(("p_", "r_", "s3_"))] + ["fwd"]]},
    {"누적수익": a2_fig1},
    {"성과": a2_stats.to_dict(orient="index"), "IC": a2_ic_tab.to_dict(orient="index")},
)


#%% [추가 전략 1-3] 앙상블 기간 늘려 확인: 1-2 셀을 2005년부터(금융위기 포함) 다시 돌리고 2012 전·후로 나눠 보기
# 무엇: 1-2에서 'MLP + 모멘텀 앙상블'이 샤프 0.79로 가장 좋았다. 2012년 이전에도 통하는지 확인한다.
# 실행: 맨 위 [1] 환경 설정 셀 먼저, 그다음 이 셀 (1-2 셀은 안 돌려도 됨 - 이 셀이 파일에서 1-2 코드를 읽어 실행). 30~40분
# 결과: 06_퀀트공부/딥러닝/결과/추가1-3_앙상블_2005년부터/ (결과요약.md 끝에 '기간 나눠 보기' 표)
# 이 셀은 바로 위 [추가 전략 1-2] 셀 코드를 이 파일에서 그대로 읽어 와, 아래 설정 4개만 바꿔 실행한다.
# (같은 코드를 두 번 붙이지 않으려고. 1-2 셀을 고치면 1-3에도 그대로 반영된다)
A3_FILE = os.path.join(CODE_DIR, "01_mlp_etf_allocation.py")
A3_CHANGES = [
    ('A2_TAG = "추가1-2_MLP_회전율_앙상블_거시_순위정답"', 'A2_TAG = "추가1-3_앙상블_2005년부터"'),
    ("A2_TEST_START = 2012", "A2_TEST_START = 2005"),       # 학습 자료가 200개 넘는 해부터 자동 시작 (2006년 무렵)
    ('"추가 전략 1-2: MLP 자산배분 개선', '"추가 전략 1-3: (2005년부터 다시) MLP 자산배분 개선'),
    ('A2_TEST_START}년부터 매년 1월 재학습', 'A2_TEST_START}년부터(학습 자료 200개 넘는 해부터) 매년 1월 재학습'),
]
A3_SPLIT = [("2006~2011 (2012 전, 금융위기 포함)", "2005-01-01", "2011-12-31"), ("2012~ (1-2와 같은 기간)", "2012-01-01", "2100-01-01")]

with open(A3_FILE, encoding="utf-8") as fh:
    a3_src = fh.read()
a3_s = a3_src.index("#%% [추가 전략 1-2]")
a3_e = a3_src.index("#%% [추가 전략 1-3]")
a3_code = a3_src[a3_s:a3_e]
for a, b in A3_CHANGES:
    assert a in a3_code, f"1-2 셀에서 못 찾음: {a}"
    a3_code = a3_code.replace(a, b)
print("[추가1-3] 1-2 셀 코드를 2005년부터로 바꿔 실행합니다 (모델 4개 x 약 21년 x 시드 3개라 30~40분)")
exec(compile(a3_code, "추가전략_1-2_2005년부터", "exec"), globals())

# 기간을 둘로 나눠 성과 비교 -> 결과요약.md 끝에 덧붙임
a3_rows = {}
for lab, s, e in A3_SPLIT:
    for k, v in a2_rets.items():
        vv = v[(v.index >= s) & (v.index <= e)]
        if len(vv) > 12:
            st = perf_stats(vv, 12)
            a3_rows[f"{lab} · {k}"] = {"CAGR": st["CAGR"], "샤프": st["샤프"], "MDD": st["MDD"], "달 수": st["기간수"]}
a3_split = pd.DataFrame(a3_rows).T
a3_split.index.name = "기간 · 점수 · 규칙"
a3_key = ["앙상블(MLP기본+모멘텀) · 매달", "앙상블(MLP기본+모멘텀) · 3개월보유", "앙상블(MLP거시+모멘텀) · 3개월보유",
          "모멘텀 · 매달", "MLP 기본 · 매달", "10개 같은 비중", "SPY 보유"]
a3_short = a3_split[[any(i.endswith(k) for k in a3_key) for i in a3_split.index]]
a3_md = os.path.join(RESULT_ROOT, A2_TAG, "결과요약.md")
with open(a3_md, "a", encoding="utf-8") as fh:
    fh.write("\n## 기간 나눠 보기 (대표 조합)\n\n" + md_table(a3_short, pct_cols=("CAGR", "MDD")) + "\n")
    fh.write("\n## 기간 나눠 보기 (전체)\n\n" + md_table(a3_split, pct_cols=("CAGR", "MDD")) + "\n")
a3_split.to_csv(os.path.join(RESULT_ROOT, A2_TAG, "기간나눠보기.csv"), encoding="utf-8-sig")
print(md_table(a3_short, pct_cols=("CAGR", "MDD")))
print(f"[추가1-3] 기간 나눠 보기를 {a3_md} 끝에 덧붙였습니다.")
