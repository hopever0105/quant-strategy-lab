# -*- coding: utf-8 -*-
"""
한미 시차 LSTM·로지스틱: 반도체 장중(시가→종가) 매매 (추가 전략 3, 3-2, 3-3)
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Input, Dense, LSTM, Activation, Dropout, BatchNormalization
    from tensorflow.keras import backend as K, regularizers
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


print(f"[초기화 완료] TA-Lib 사용 가능: {TALIB_AVAILABLE} | TensorFlow/Keras 사용 가능: {TF_AVAILABLE}")


#%% [추가 전략 3] 한미 시차 LSTM: 미국 밤사이 정보로 삼성전자·SK하이닉스 장중(시가->종가) 방향 맞히기 
# 무엇: 미국 장은 한국 새벽에 끝난다. 전날 밤 SPY·QQQ·SOXX(반도체) 움직임 + 한국 시가 갭을 보고
#  '오늘 시가에 사서 종가에 팔면 오를까'를 LSTM이 최근 20거래일 흐름으로 판단
#  - 핵심 질문: 미국 정보가 시가(갭)에 이미 다 반영되나, 장중까지 이어지나
#  - 매년 재학습, 두 종목을 한 모델로 (종목 구분 특성 포함)
# 비교: 로지스틱(마지막 날 정보만), 갭 하락 날만 매수, SOXX 상승 날만 매수, 매일 매수 / 비용 전·후 둘 다
# 실행: 맨 위 [1] 환경 설정 셀 먼저, 그다음 이 셀. TF 있으면 10분 안팎
# 결과: 06_퀀트공부/딥러닝/결과/추가3_LSTM_한미시차_장중/ 에 자동 저장
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# ----- 설정 -----
C_TAG = "추가3_LSTM_한미시차_장중"
C_KR = {"005930": "삼성전자", "000660": "SK하이닉스"}
C_US = ["SPY", "QQQ", "SOXX"]
C_SEQ = 20                 # LSTM이 보는 과거 거래일 수
C_COST_RT = 0.0025         # 시가 매수 -> 종가 매도 한 번 왕복 비용 0.25% (세금·수수료·슬리피지 가정, 바꿔 보기)
C_TEST_START = 2014
C_EPOCHS = 15

# ----- 1) 데이터 -----
c_us_raw = load_sql(
    "SELECT 날짜, 티커, 종가, 수정종가 FROM us_price "
    f"WHERE 티커 IN ({', '.join(repr(t) for t in C_US)}) AND 날짜 >= '2002-01-01'"
)
c_us_raw["날짜"] = pd.to_datetime(c_us_raw["날짜"])
c_us_raw["가격"] = c_us_raw["수정종가"].fillna(c_us_raw["종가"])
c_us_px = c_us_raw.pivot(index="날짜", columns="티커", values="가격").sort_index()[C_US].dropna()
c_us = pd.DataFrame(index=c_us_px.index)
for t in C_US:
    c_us[f"us_{t}"] = c_us_px[t].pct_change(fill_method=None)
c_us["us_SOXX_minus_SPY"] = c_us["us_SOXX"] - c_us["us_SPY"]
c_us["us_SOXX_5d"] = c_us_px["SOXX"].pct_change(5, fill_method=None)
c_us = c_us.dropna()
C_US_COLS = list(c_us.columns)

c_kr_raw = load_sql(
    "SELECT 날짜, 종목코드, 시가, 고가, 저가, 종가, 거래량 FROM kor_price "
    f"WHERE 종목코드 IN ({', '.join(repr(k) for k in C_KR)}) AND 날짜 >= '2002-01-01'"
)
c_kr_raw["날짜"] = pd.to_datetime(c_kr_raw["날짜"])
print(f"[추가3] 미국 {c_us.index.min().date()}~{c_us.index.max().date()}, 한국 {len(c_kr_raw)}행")

# ----- 2) 종목별 특성: 전부 '그날 아침 9시 시가 시점에 알 수 있는 값' -----
C_FEATS = C_US_COLS + ["gap", "prev_intra", "prev_ret", "prev_ret5", "prev_volz", "is_hynix"]
c_frames = []
for code, name in C_KR.items():
    k = c_kr_raw[c_kr_raw["종목코드"] == code].sort_values("날짜").set_index("날짜")
    k = k[(k["시가"] > 0) & (k["거래량"] > 0)].copy()      # 거래정지일(시가 0) 제외
    k["gap"] = k["시가"] / k["종가"].shift(1) - 1           # 밤사이 갭 (시가에 이미 앎)
    k["intra"] = k["종가"] / k["시가"] - 1                  # 시가 -> 종가 (맞힐 대상)
    k["ret"] = k["종가"].pct_change(fill_method=None)
    k["prev_intra"] = k["intra"].shift(1)
    k["prev_ret"] = k["ret"].shift(1)
    k["prev_ret5"] = k["종가"].pct_change(5, fill_method=None).shift(1)
    lv = np.log(k["거래량"])
    k["prev_volz"] = ((lv - lv.rolling(20).mean()) / lv.rolling(20).std()).shift(1)
    k["is_hynix"] = 1.0 if code == "000660" else 0.0
    # 한국 날짜 d 에는 '날짜가 d보다 앞선 마지막 미국 거래일' 종가 정보 (미국 장은 한국 새벽에 끝남)
    us_aligned = pd.merge_asof(
        k.reset_index()[["날짜"]], c_us.reset_index().rename(columns={c_us.index.name or "index": "날짜"}),
        on="날짜", direction="backward", allow_exact_matches=False,
    ).set_index("날짜")
    for col in C_US_COLS:
        k[col] = us_aligned[col]
    k = k[(k["ret"].abs() < 0.35) | k["ret"].isna()]       # 액면분할 등 이상치 하루 제거
    k["code"] = code
    c_frames.append(k)


# ----- 3) 시퀀스 만들기: (최근 20거래일 x 특성) -> 그날 시가->종가 상승 여부 -----
def c_make_seq(k):
    f = k[C_FEATS].values
    xs, ys, rs, ds = [], [], [], []
    for i in range(C_SEQ - 1, len(k)):
        win = f[i - C_SEQ + 1: i + 1]
        if np.isnan(win).any() or np.isnan(k["intra"].iloc[i]):
            continue
        xs.append(win)
        ys.append(int(k["intra"].iloc[i] > 0))
        rs.append(k["intra"].iloc[i])
        ds.append(k.index[i])
    return np.array(xs), np.array(ys), np.array(rs), pd.DatetimeIndex(ds)


c_parts = [c_make_seq(k) for k in c_frames]
c_X = np.concatenate([p[0] for p in c_parts])
c_y = np.concatenate([p[1] for p in c_parts])
c_r = np.concatenate([p[2] for p in c_parts])
c_d = pd.DatetimeIndex(np.concatenate([p[3].values for p in c_parts]))
c_code = np.concatenate([[code] * len(p[1]) for code, p in zip(C_KR, c_parts)])
c_order = np.argsort(c_d.values, kind="stable")          # 날짜순 정렬 (검증 구간이 최근이 되게)
c_X, c_y, c_r, c_d, c_code = c_X[c_order], c_y[c_order], c_r[c_order], c_d[c_order], c_code[c_order]
print(f"[추가3] 시퀀스 {c_X.shape}, 장중 상승 비율 {c_y.mean():.3f}")

# 사전 진단: 미국 반도체(SOXX) 전날 수익이 갭과 장중에 각각 얼마나 이어지는지
c_diag = {}
for k in c_frames:
    kk = k.dropna(subset=["us_SOXX", "gap", "intra"])
    c_diag[C_KR[k["code"].iloc[0]]] = {
        "SOXX-갭 상관": kk["us_SOXX"].corr(kk["gap"]),
        "SOXX-장중 상관": kk["us_SOXX"].corr(kk["intra"]),
        "갭-장중 상관": kk["gap"].corr(kk["intra"]),
        "장중 평균수익": kk["intra"].mean(),
    }
c_diag = pd.DataFrame(c_diag).T
c_diag.index.name = "종목"


# ----- 4) 모델: LSTM (TF 없으면 로지스틱으로 대체) + 비교용 로지스틱(마지막 날 정보만) -----
def c_build_lstm(n_feat, seed):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    tf.random.set_seed(seed)
    inp = Input(shape=(C_SEQ, n_feat))
    x = LSTM(32)(inp)
    x = Dropout(0.2)(x)
    x = Dense(16, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)
    m = Model(inp, out, name="KR_US_Lag_LSTM")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m


c_prob_lstm = np.full(len(c_y), np.nan)
c_prob_lr = np.full(len(c_y), np.nan)
c_thr_lstm = np.full(len(c_y), np.nan)
c_thr_lr = np.full(len(c_y), np.nan)
c_train_log = []
c_lr_coef = None
n_feat = c_X.shape[2]
for yr in range(C_TEST_START, c_d.max().year + 1):
    te = np.where(c_d.year == yr)[0]
    tr = np.where(c_d < pd.Timestamp(f"{yr}-01-01"))[0]
    if len(te) == 0 or len(tr) < 1000:
        continue
    sc = StandardScaler().fit(c_X[tr].reshape(-1, n_feat))
    x_tr = sc.transform(c_X[tr].reshape(-1, n_feat)).reshape(-1, C_SEQ, n_feat)
    x_te = sc.transform(c_X[te].reshape(-1, n_feat)).reshape(-1, C_SEQ, n_feat)

    lr = LogisticRegression(C=0.1, max_iter=2000)
    lr.fit(x_tr[:, -1, :], c_y[tr])
    c_prob_lr[te] = lr.predict_proba(x_te[:, -1, :])[:, 1]
    c_thr_lr[te] = lr.predict_proba(x_tr[:, -1, :])[:, 1].mean()
    c_lr_coef = pd.Series(lr.coef_[0], index=C_FEATS)

    if TF_AVAILABLE:
        from tensorflow.keras.callbacks import EarlyStopping
        m = c_build_lstm(n_feat, SEED)
        m.fit(x_tr, c_y[tr], epochs=C_EPOCHS, batch_size=128, validation_split=0.15, verbose=0,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
        c_prob_lstm[te] = m.predict(x_te, verbose=0).ravel()
        c_thr_lstm[te] = m.predict(x_tr, verbose=0).ravel().mean()
    else:
        c_prob_lstm[te] = c_prob_lr[te]
        c_thr_lstm[te] = c_thr_lr[te]
    c_train_log.append({"연도": yr, "학습표본": len(tr), "예측표본": len(te)})
    print(f"  {yr}년 모델: 학습 {len(tr)}개 -> {len(te)}개 예측")

# ----- 5) 분류 성능 -----
c_mask = ~np.isnan(c_prob_lstm)
c_name = "LSTM" if TF_AVAILABLE else "로지스틱(대체)"
c_sig_lstm = (c_prob_lstm > c_thr_lstm) & c_mask
c_sig_lr = (c_prob_lr > c_thr_lr) & c_mask
c_cls = {}
for nm, p, s in [(c_name, c_prob_lstm, c_sig_lstm), ("로지스틱(마지막 날만)", c_prob_lr, c_sig_lr)]:
    c_cls[nm] = {"AUC": roc_auc_score(c_y[c_mask], p[c_mask]),
                 "정확도": accuracy_score(c_y[c_mask], s[c_mask].astype(int)),
                 "매수 신호 비율": s[c_mask].mean(), "실제 상승 비율": c_y[c_mask].mean()}
c_cls = pd.DataFrame(c_cls).T
c_cls.index.name = "모델"

# ----- 6) 매매: 신호 1이면 그날 시가 매수 -> 종가 매도, 두 종목에 자금 반씩 -----
c_days = pd.DatetimeIndex(sorted(set(c_d[c_mask])))


def c_daily(sig, cost):
    df = pd.DataFrame({"d": c_d[c_mask], "code": c_code[c_mask],
                       "r": np.where(sig[c_mask], c_r[c_mask] - cost, 0.0)})
    wide = df.pivot_table(index="d", columns="code", values="r", aggfunc="sum").reindex(c_days).fillna(0.0)
    return wide.mean(axis=1)    # 종목마다 자금 절반


c_always = np.ones(len(c_y), dtype=bool)
c_gapdown = np.zeros(len(c_y), dtype=bool)
c_gapdown[c_mask] = c_X[c_mask][:, -1, C_FEATS.index("gap")] < 0      # 갭 하락한 날만 (단순 반전 규칙)
c_soxx_up = np.zeros(len(c_y), dtype=bool)
c_soxx_up[c_mask] = c_X[c_mask][:, -1, C_FEATS.index("us_SOXX")] > 0  # 전날 미국 반도체 상승한 날만
c_rules = {f"{c_name} 신호": c_sig_lstm, "로지스틱 신호": c_sig_lr, "갭 하락 날만": c_gapdown,
           "SOXX 상승 날만": c_soxx_up, "매일 시가→종가": c_always}
c_rets_net = {k: c_daily(v, C_COST_RT) for k, v in c_rules.items()}
c_rets_gross = {k: c_daily(v, 0.0) for k, v in c_rules.items()}
c_stats = stats_frame(c_rets_net, 252)
c_stats["비용 전 CAGR"] = [perf_stats(c_rets_gross[k], 252)["CAGR"] for k in c_rules]
c_trade_tab = {}
for k, v in c_rules.items():
    tr_r = c_r[v & c_mask]
    c_trade_tab[k] = {"매매 수": int(len(tr_r)), "1회 평균(비용 전)": tr_r.mean(),
                      "1회 평균(비용 후)": tr_r.mean() - C_COST_RT,
                      "t값(비용 전)": tr_r.mean() / tr_r.std() * np.sqrt(len(tr_r)) if len(tr_r) > 1 else np.nan,
                      "승률": (tr_r > 0).mean()}
c_trade_tab = pd.DataFrame(c_trade_tab).T
c_trade_tab.index.name = "규칙"
c_year = yearly_table(c_rets_net)
c_coef_tab = pd.DataFrame({"로지스틱 계수(마지막 모델)": c_lr_coef}).sort_values(
    "로지스틱 계수(마지막 모델)", key=lambda s: -s.abs()) if c_lr_coef is not None else pd.DataFrame()
c_coef_tab.index.name = "특성"

# ----- 7) 그림 -----
c_fig1, ax = plt.subplots(figsize=(11, 4.5))
for k, r in c_rets_net.items():
    ax.plot((1 + r).cumprod(), label=k, linewidth=1.8 if k.startswith(c_name) else 1.2)
ax.set_title("추가3: 미국 밤사이 정보로 삼성전자·SK하이닉스 장중(시가→종가) 매매 (비용 차감)")
ax.legend()
ax.grid(True, alpha=0.3)
c_fig1.tight_layout()
plt.show()

print(md_table(c_cls, pct_cols=("매수 신호 비율", "실제 상승 비율", "정확도")))
print(md_table(c_trade_tab, pct_cols=("1회 평균(비용 전)", "1회 평균(비용 후)", "승률"), digits=2))

# ----- 8) 결과 자동 저장 -----
c_best = c_trade_tab["1회 평균(비용 전)"].idxmax()
c_auto = [
    f"- 표본 밖 AUC: {c_name} {c_cls.iloc[0]['AUC']:.3f}, 로지스틱 {c_cls.iloc[1]['AUC']:.3f}",
    f"- 비용 후 샤프: " + " / ".join(f"{k} {c_stats.loc[k, '샤프']:.2f}" for k in c_rules),
    f"- 1회 평균 비용 전 수익이 가장 큰 규칙: {c_best} ({c_trade_tab.loc[c_best, '1회 평균(비용 전)'] * 100:.3f}%, t {c_trade_tab.loc[c_best, 't값(비용 전)']:.2f}) vs 왕복 비용 {C_COST_RT * 100:.2f}%",
    f"- 진단: SOXX-갭 상관 평균 {c_diag['SOXX-갭 상관'].mean():.2f}, SOXX-장중 상관 평균 {c_diag['SOXX-장중 상관'].mean():.2f} (미국 정보가 시가에 이미 반영됐는지)",
]
save_result(
    C_TAG, "추가 전략 3: 미국 밤사이 정보 LSTM으로 삼성전자·SK하이닉스 장중 방향 맞히기",
    [
        ("한눈에", "\n".join(c_auto)),
        ("설정", "\n".join([
            f"- 종목: {', '.join(f'{v}({k})' for k, v in C_KR.items())}, 미국: {', '.join(C_US)} 수정종가",
            f"- 특성(그날 시가 시점에 아는 값만): {', '.join(C_FEATS)}",
            f"- 미국 정보는 한국 날짜보다 앞선 마지막 미국 거래일 종가 (미국 장 마감 = 한국 새벽)",
            f"- 정답: 그날 시가->종가 수익 > 0, 입력: 최근 {C_SEQ}거래일 시퀀스",
            f"- 모델: LSTM 32 - Dropout 0.2 - Dense 16 - sigmoid, 최대 {C_EPOCHS}회·조기 종료, {C_TEST_START}년부터 매년 재학습",
            f"- 매매: 확률 > 학습 구간 평균 확률이면 시가 매수·종가 매도, 두 종목 자금 반씩, 왕복 비용 {C_COST_RT * 100:.2f}%",
        ])),
        ("사전 진단: 미국 정보가 어디에 반영되나", md_table(c_diag, digits=3)),
        ("분류 성능 (표본 밖)", md_table(c_cls, pct_cols=("매수 신호 비율", "실제 상승 비율", "정확도"))),
        ("매매 1회 기준 (두 종목 합산)", md_table(c_trade_tab, pct_cols=("1회 평균(비용 전)", "1회 평균(비용 후)", "승률"), digits=2)),
        ("일간 성과 (비용 차감)", md_table(c_stats, pct_cols=("CAGR", "변동성", "MDD", "누적수익", "플러스비율", "비용 전 CAGR"))),
        ("연도별 수익률 (비용 차감)", md_table(c_year, pct_cols="all")),
        ("마지막 로지스틱 계수 (표준화 기준, 부호 = 장중 상승 쪽)", md_table(c_coef_tab) if len(c_coef_tab) else "-"),
        ("재학습 기록", md_table(pd.DataFrame(c_train_log).set_index("연도"))),
        ("주의", "\n".join([
            "- 비용은 가정값. 실제 매도 세율은 연도마다 달랐다 -> C_COST_RT 바꿔서 다시 확인.",
            "- 시가 체결 가정 (동시호가 가격 그대로). 실제로는 미끄러짐이 더 있다.",
            "- 두 종목뿐이라 종목 특유의 사건에 결과가 좌우될 수 있다.",
        ])),
    ],
    {"일별수익_비용후": pd.DataFrame(c_rets_net), "일별수익_비용전": pd.DataFrame(c_rets_gross), "연도별수익": c_year,
     "예측확률": pd.DataFrame({"날짜": c_d[c_mask], "종목": c_code[c_mask], c_name: c_prob_lstm[c_mask],
                              "로지스틱": c_prob_lr[c_mask], "장중수익": c_r[c_mask]}).set_index("날짜")},
    {"누적수익": c_fig1},
    {"분류": c_cls.to_dict(orient="index"), "성과": c_stats.to_dict(orient="index"),
     "매매": c_trade_tab.to_dict(orient="index"), "진단": c_diag.to_dict(orient="index")},
)


#%% [추가 전략 3-2] 한미 시차 개선: 반도체 ETF로 비용 줄이기·확신 높은 날만·되돌림 확인·미국 갭 반전 (추가전략_다음할일.md 의 추가3)
# 무엇: 추가 전략 3은 비용 전 1회 +0.11~0.13%(t 3.6~4.2)인데 왕복 비용 0.25%에 못 미쳤다. 할 일 4개를 한 번에:
#  1) KODEX 반도체 ETF(매도세 없음, 왕복 0.05% 가정)를 주식 2개와 함께 학습·비교
#  2) 학습 구간 확률의 상위 20%·10% 넘는 날만 매수  3) SOXX·갭이 크게 오른 날 장중 되돌림(매도) 확인
#  4) 미국 대형주 갭 반전 롱숏(BRAIN 알파 후보): 갭 하위 10% 매수·상위 10% 매도, 장중 + 다음 날 종가→종가
# 실행: 맨 위 [1] 환경 설정 셀 먼저, 그다음 이 셀. LSTM 학습 10분 + 미국 자료 1~2분
# 결과: 06_퀀트공부/딥러닝/결과/추가3-2_한미시차_비용줄이기_반전_미국갭/ 에 자동 저장
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

# ----- 설정 (바꿔 보면서 다시 돌려 보기) -----
C2_TAG = "추가3-2_한미시차_비용줄이기_반전_미국갭"
C2_KR = {"005930": "삼성전자", "000660": "SK하이닉스"}
C2_ETF = {"091160": "KODEX 반도체"}         # 매도 거래세가 없는 반도체 ETF (krx_etf_daily)
C2_NAMES = {**C2_KR, **C2_ETF}
C2_US = ["SPY", "QQQ", "SOXX"]
C2_SEQ = 20
C2_COST = {"005930": 0.0025, "000660": 0.0025,   # 주식 왕복: 매도세 + 수수료 + 미끄러짐 (가정)
           "091160": 0.0005}                     # ETF 왕복: 매도세 없음, 수수료 + 미끄러짐만 (가정)
C2_TEST_START = 2014
C2_EPOCHS = 15
C2_TOPQ = {"상위20%": 0.8, "상위10%": 0.9}       # 학습 구간 예측확률 분포의 분위 = 기준선 (미래 정보 없음)
C2_US_TOP = 150            # 미국 갭 반전: S&P500(IVV 보유) 비중 상위 종목 수
C2_US_START = "2016-01-01"
C2_US_COST = 0.0005        # 미국 한 번 사고팔 때 한쪽 0.05% (가정)

# ----- 1) 데이터: 미국 밤사이 정보 -----
c2_us_raw = load_sql(
    "SELECT 날짜, 티커, 종가, 수정종가 FROM us_price "
    f"WHERE 티커 IN ({', '.join(repr(t) for t in C2_US)}) AND 날짜 >= '2002-01-01'"
)
c2_us_raw["날짜"] = pd.to_datetime(c2_us_raw["날짜"])
c2_us_raw["가격"] = c2_us_raw["수정종가"].fillna(c2_us_raw["종가"])
c2_us_px = c2_us_raw.pivot(index="날짜", columns="티커", values="가격").sort_index()[C2_US].dropna()
c2_us = pd.DataFrame(index=c2_us_px.index)
for t in C2_US:
    c2_us[f"us_{t}"] = c2_us_px[t].pct_change(fill_method=None)
c2_us["us_SOXX_minus_SPY"] = c2_us["us_SOXX"] - c2_us["us_SPY"]
c2_us["us_SOXX_5d"] = c2_us_px["SOXX"].pct_change(5, fill_method=None)
c2_us = c2_us.dropna()
C2_US_COLS = list(c2_us.columns)

# 한국: 주식 2개(kor_price) + 반도체 ETF(krx_etf_daily, 같은 열 이름으로 맞춤)
c2_kr_raw = load_sql(
    "SELECT 날짜, 종목코드, 시가, 고가, 저가, 종가, 거래량 FROM kor_price "
    f"WHERE 종목코드 IN ({', '.join(repr(k) for k in C2_KR)}) AND 날짜 >= '2002-01-01'"
)
c2_etf_raw = load_sql(
    "SELECT BAS_DD AS 날짜, ISU_CD AS 종목코드, TDD_OPNPRC AS 시가, TDD_HGPRC AS 고가, TDD_LWPRC AS 저가, "
    "TDD_CLSPRC AS 종가, ACC_TRDVOL AS 거래량 FROM krx_etf_daily "
    f"WHERE ISU_CD IN ({', '.join(repr(k) for k in C2_ETF)})"
)
c2_etf_raw = c2_etf_raw.drop_duplicates(subset=["날짜", "종목코드"], keep="first")   # 출처(SRC)가 둘인 날 중복 제거
c2_all_raw = pd.concat([c2_kr_raw, c2_etf_raw], ignore_index=True)
c2_all_raw["날짜"] = pd.to_datetime(c2_all_raw["날짜"])
for col in ["시가", "고가", "저가", "종가", "거래량"]:
    c2_all_raw[col] = pd.to_numeric(c2_all_raw[col], errors="coerce")
print(f"[추가3-2] 미국 {c2_us.index.min().date()}~{c2_us.index.max().date()}, 한국 {len(c2_all_raw)}행 "
      f"(ETF {len(c2_etf_raw)}행)")

# ----- 2) 종목별 특성 (그날 시가 시점에 아는 값만) -----
C2_FEATS = C2_US_COLS + ["gap", "prev_intra", "prev_ret", "prev_ret5", "prev_volz", "is_hynix", "is_etf"]
c2_frames = []
for code in C2_NAMES:
    k = c2_all_raw[c2_all_raw["종목코드"] == code].sort_values("날짜").set_index("날짜")
    k = k[(k["시가"] > 0) & (k["거래량"] > 0)].copy()
    k["gap"] = k["시가"] / k["종가"].shift(1) - 1
    k["intra"] = k["종가"] / k["시가"] - 1
    k["ret"] = k["종가"].pct_change(fill_method=None)
    k["prev_intra"] = k["intra"].shift(1)
    k["prev_ret"] = k["ret"].shift(1)
    k["prev_ret5"] = k["종가"].pct_change(5, fill_method=None).shift(1)
    lv = np.log(k["거래량"])
    k["prev_volz"] = ((lv - lv.rolling(20).mean()) / lv.rolling(20).std()).shift(1)
    k["is_hynix"] = 1.0 if code == "000660" else 0.0
    k["is_etf"] = 1.0 if code in C2_ETF else 0.0
    us_aligned = pd.merge_asof(
        k.reset_index()[["날짜"]], c2_us.reset_index().rename(columns={c2_us.index.name or "index": "날짜"}),
        on="날짜", direction="backward", allow_exact_matches=False,
    ).set_index("날짜")
    for col in C2_US_COLS:
        k[col] = us_aligned[col]
    k = k[(k["ret"].abs() < 0.35) | k["ret"].isna()]
    k = k[k["gap"].abs() < 0.3]          # 분할·병합 흔적 제거
    k["code"] = code
    c2_frames.append(k)


def c2_make_seq(k):
    f = k[C2_FEATS].values
    xs, ys, rs, ds = [], [], [], []
    for i in range(C2_SEQ - 1, len(k)):
        win = f[i - C2_SEQ + 1: i + 1]
        if np.isnan(win).any() or np.isnan(k["intra"].iloc[i]):
            continue
        xs.append(win)
        ys.append(int(k["intra"].iloc[i] > 0))
        rs.append(k["intra"].iloc[i])
        ds.append(k.index[i])
    return np.array(xs), np.array(ys), np.array(rs), pd.DatetimeIndex(ds)


c2_parts = [c2_make_seq(k) for k in c2_frames]
c2_X = np.concatenate([p[0] for p in c2_parts])
c2_y = np.concatenate([p[1] for p in c2_parts])
c2_r = np.concatenate([p[2] for p in c2_parts])
c2_d = pd.DatetimeIndex(np.concatenate([p[3].values for p in c2_parts]))
c2_code = np.concatenate([[code] * len(p[1]) for code, p in zip(C2_NAMES, c2_parts)])
c2_order = np.argsort(c2_d.values, kind="stable")
c2_X, c2_y, c2_r, c2_d, c2_code = c2_X[c2_order], c2_y[c2_order], c2_r[c2_order], c2_d[c2_order], c2_code[c2_order]
print(f"[추가3-2] 시퀀스 {c2_X.shape} (주식 2 + ETF 1을 한 모델로)")


# ----- 3) 모델: LSTM + 로지스틱, 매년 재학습. 기준선은 학습 구간 확률의 평균·80%·90% 분위 -----
def c2_build_lstm(n_feat, seed):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    tf.random.set_seed(seed)
    inp = Input(shape=(C2_SEQ, n_feat))
    x = LSTM(32)(inp)
    x = Dropout(0.2)(x)
    x = Dense(16, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)
    m = Model(inp, out, name="KR_US_Lag_LSTM_v2")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return m


c2_prob = {"LSTM": np.full(len(c2_y), np.nan), "로지스틱": np.full(len(c2_y), np.nan)}
c2_thr = {m: {nm: np.full(len(c2_y), np.nan) for nm in ["평균", *C2_TOPQ, "하위10%"]} for m in c2_prob}
c2_train_log = []
c2_nf = c2_X.shape[2]
for yr in range(C2_TEST_START, c2_d.max().year + 1):
    te = np.where(c2_d.year == yr)[0]
    tr = np.where(c2_d < pd.Timestamp(f"{yr}-01-01"))[0]
    if len(te) == 0 or len(tr) < 1000:
        continue
    sc = StandardScaler().fit(c2_X[tr].reshape(-1, c2_nf))
    x_tr = sc.transform(c2_X[tr].reshape(-1, c2_nf)).reshape(-1, C2_SEQ, c2_nf)
    x_te = sc.transform(c2_X[te].reshape(-1, c2_nf)).reshape(-1, C2_SEQ, c2_nf)
    lr = LogisticRegression(C=0.1, max_iter=2000).fit(x_tr[:, -1, :], c2_y[tr])
    p_tr = {"로지스틱": lr.predict_proba(x_tr[:, -1, :])[:, 1]}
    c2_prob["로지스틱"][te] = lr.predict_proba(x_te[:, -1, :])[:, 1]
    if TF_AVAILABLE:
        from tensorflow.keras.callbacks import EarlyStopping
        m = c2_build_lstm(c2_nf, SEED)
        m.fit(x_tr, c2_y[tr], epochs=C2_EPOCHS, batch_size=128, validation_split=0.15, verbose=0,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
        c2_prob["LSTM"][te] = m.predict(x_te, verbose=0).ravel()
        p_tr["LSTM"] = m.predict(x_tr, verbose=0).ravel()
    else:
        c2_prob["LSTM"][te] = c2_prob["로지스틱"][te]
        p_tr["LSTM"] = p_tr["로지스틱"]
    for mname in c2_prob:
        c2_thr[mname]["평균"][te] = p_tr[mname].mean()
        for qn, q in C2_TOPQ.items():
            c2_thr[mname][qn][te] = np.quantile(p_tr[mname], q)
        c2_thr[mname]["하위10%"][te] = np.quantile(p_tr[mname], 0.1)
    c2_train_log.append({"연도": yr, "학습표본": len(tr), "예측표본": len(te)})
    print(f"  {yr}년 모델: 학습 {len(tr)}개 -> {len(te)}개 예측")

c2_mask = ~np.isnan(c2_prob["로지스틱"])
C2_MNAME = {"LSTM": "LSTM" if TF_AVAILABLE else "로지스틱(대체)", "로지스틱": "로지스틱"}

# ----- 4) 규칙들: 모델 기준선별 + 단순 규칙 + 반대 방향(되돌림) 규칙 -----
c2_gap = c2_X[:, -1, C2_FEATS.index("gap")]
c2_soxx = c2_X[:, -1, C2_FEATS.index("us_SOXX")]
# 'SOXX가 크게 오른 날' 기준 = 그 전날까지의 SOXX 일간수익 80% 분위 (확장 창, 미래 정보 없음)
c2_soxx_q80 = c2_us["us_SOXX"].expanding(250).quantile(0.8).shift(1)
c2_soxx_q20 = c2_us["us_SOXX"].expanding(250).quantile(0.2).shift(1)
c2_us_day = pd.merge_asof(pd.DataFrame({"날짜": c2_d}), pd.DataFrame({"날짜": c2_us.index, "us날짜": c2_us.index}),
                          on="날짜", direction="backward", allow_exact_matches=False)["us날짜"]
c2_q80 = c2_soxx_q80.reindex(c2_us_day.values).values
c2_q20 = c2_soxx_q20.reindex(c2_us_day.values).values
c2_gap_q80 = np.full(len(c2_y), np.nan)
c2_gap_q20 = np.full(len(c2_y), np.nan)
for code in C2_NAMES:                          # 갭 기준도 종목별 확장 창 분위 (전날까지)
    idx = np.where(c2_code == code)[0]
    g = pd.Series(c2_gap[idx])
    c2_gap_q80[idx] = g.expanding(250).quantile(0.8).shift(1).values
    c2_gap_q20[idx] = g.expanding(250).quantile(0.2).shift(1).values

c2_long_rules = {}
for mname in c2_prob:
    for tn in ["평균", *C2_TOPQ]:
        c2_long_rules[f"{C2_MNAME[mname]} {tn} 이상"] = c2_mask & (c2_prob[mname] > c2_thr[mname][tn])
c2_long_rules["매일 시가→종가"] = c2_mask.copy()
c2_long_rules["SOXX 상승 날"] = c2_mask & (c2_soxx > 0)
c2_long_rules["SOXX 하락 날"] = c2_mask & (c2_soxx < 0)
c2_long_rules["SOXX 크게 하락(하위20%) 날"] = c2_mask & (c2_soxx < c2_q20)
c2_long_rules["갭 하락 날"] = c2_mask & (c2_gap < 0)
c2_long_rules["갭 크게 하락(하위20%) 날"] = c2_mask & (c2_gap < c2_gap_q20)
# 반대 방향: 장중 '공매도'(시가에 팔고 종가에 되삼) 가정 -> 되돌림 확인용 (실제 개인 공매도는 어려움, ETF는 인버스로 대체 가능성만)
c2_short_rules = {
    "SOXX 크게 상승(상위20%) 날 장중 매도": c2_mask & (c2_soxx > c2_q80),
    "갭 크게 상승(상위20%) 날 장중 매도": c2_mask & (c2_gap > c2_gap_q80),
    f"{C2_MNAME['LSTM']} 하위10% 날 장중 매도": c2_mask & (c2_prob["LSTM"] < c2_thr["LSTM"]["하위10%"]),
}


def c2_t(x):
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 and x.std() > 0 else np.nan


c2_rows = []
for code, name in C2_NAMES.items():
    is_c = c2_code == code
    cost = C2_COST[code]
    for rn, sig in c2_long_rules.items():
        x = c2_r[sig & is_c]
        c2_rows.append({"종목": name, "규칙": rn, "방향": "매수", "매매 수": len(x),
                        "1회 평균(비용 전)": x.mean() if len(x) else np.nan, "t값": c2_t(x),
                        "1회 평균(비용 후)": x.mean() - cost if len(x) else np.nan, "승률": (x > 0).mean() if len(x) else np.nan})
    for rn, sig in c2_short_rules.items():
        x = -c2_r[sig & is_c]
        c2_rows.append({"종목": name, "규칙": rn, "방향": "매도", "매매 수": len(x),
                        "1회 평균(비용 전)": x.mean() if len(x) else np.nan, "t값": c2_t(x),
                        "1회 평균(비용 후)": x.mean() - cost if len(x) else np.nan, "승률": (x > 0).mean() if len(x) else np.nan})
c2_rule_tab = pd.DataFrame(c2_rows).set_index(["종목", "규칙"])
c2_rule_tab_flat = c2_rule_tab.reset_index()
c2_rule_tab_flat.index = c2_rule_tab_flat["종목"] + " · " + c2_rule_tab_flat["규칙"]
c2_rule_tab_flat = c2_rule_tab_flat.drop(columns=["종목", "규칙"])
c2_rule_tab_flat.index.name = "종목 · 규칙"

# ----- 5) 되돌림 진단: SOXX 밤사이 수익 5구간·갭 5구간별 갭·장중 평균 -----
c2_bucket_rows = []
for code, name in C2_NAMES.items():
    is_c = (c2_code == code) & c2_mask
    df = pd.DataFrame({"soxx": c2_soxx[is_c], "gap": c2_gap[is_c], "intra": c2_r[is_c]})
    df["SOXX구간"] = pd.qcut(df["soxx"], 5, labels=["1(가장 하락)", "2", "3", "4", "5(가장 상승)"])
    df["갭구간"] = pd.qcut(df["gap"].rank(method="first"), 5, labels=["1(가장 하락)", "2", "3", "4", "5(가장 상승)"])
    for bcol in ["SOXX구간", "갭구간"]:
        for b, g in df.groupby(bcol, observed=True):
            c2_bucket_rows.append({"종목": name, "기준": bcol, "구간": str(b), "일수": len(g),
                                   "평균 갭": g["gap"].mean(), "평균 장중": g["intra"].mean(),
                                   "장중 t값": c2_t(g["intra"].values)})
c2_bucket = pd.DataFrame(c2_bucket_rows)
c2_bucket.index = c2_bucket["종목"] + " · " + c2_bucket["기준"] + " " + c2_bucket["구간"]
c2_bucket = c2_bucket.drop(columns=["종목", "기준", "구간"])
c2_bucket.index.name = "종목 · 구간"

# ----- 6) 일간 성과: ETF 단독 / 주식 2개 반반 -----
c2_days = pd.DatetimeIndex(sorted(set(c2_d[c2_mask])))


def c2_daily(sig, codes, sign=1.0, net=True):
    pick = sig & c2_mask & np.isin(c2_code, codes)
    cost = np.array([C2_COST[c] for c in c2_code]) if net else 0.0
    r = np.where(pick, sign * c2_r - cost, 0.0)
    df = pd.DataFrame({"d": c2_d, "code": c2_code, "r": r})[c2_mask & np.isin(c2_code, codes)]
    wide = df.pivot_table(index="d", columns="code", values="r", aggfunc="sum").reindex(c2_days).fillna(0.0)
    return wide.mean(axis=1)


C2_ETF_CODES = list(C2_ETF)
C2_STK_CODES = list(C2_KR)
c2_lstm_nm = C2_MNAME["LSTM"]
c2_rets = {
    "ETF 매일": c2_daily(c2_long_rules["매일 시가→종가"], C2_ETF_CODES),
    f"ETF {c2_lstm_nm} 평균 이상": c2_daily(c2_long_rules[f"{c2_lstm_nm} 평균 이상"], C2_ETF_CODES),
    f"ETF {c2_lstm_nm} 상위20%": c2_daily(c2_long_rules[f"{c2_lstm_nm} 상위20% 이상"], C2_ETF_CODES),
    f"ETF {c2_lstm_nm} 상위10%": c2_daily(c2_long_rules[f"{c2_lstm_nm} 상위10% 이상"], C2_ETF_CODES),
    "ETF 로지스틱 상위20%": c2_daily(c2_long_rules["로지스틱 상위20% 이상"], C2_ETF_CODES),
    f"주식 {c2_lstm_nm} 평균 이상": c2_daily(c2_long_rules[f"{c2_lstm_nm} 평균 이상"], C2_STK_CODES),
    f"주식 {c2_lstm_nm} 상위10%": c2_daily(c2_long_rules[f"{c2_lstm_nm} 상위10% 이상"], C2_STK_CODES),
    "ETF SOXX 크게 상승 날 장중 매도(가정)": c2_daily(c2_short_rules["SOXX 크게 상승(상위20%) 날 장중 매도"], C2_ETF_CODES, sign=-1.0),
}
c2_stats = stats_frame(c2_rets, 252)
c2_stats["시장에 있던 날 비율"] = [float((v != 0).mean()) for v in c2_rets.values()]
c2_year = yearly_table(c2_rets)

c2_cls = {}
for mname in c2_prob:
    for code, name in C2_NAMES.items():
        mk = c2_mask & (c2_code == code)
        c2_cls[f"{C2_MNAME[mname]} · {name}"] = {"AUC": roc_auc_score(c2_y[mk], c2_prob[mname][mk]), "표본": int(mk.sum())}
c2_cls = pd.DataFrame(c2_cls).T
c2_cls.index.name = "모델 · 종목"

# ----- 7) 미국판 갭 반전 (BRAIN 알파 후보 검증): 갭이 크게 뜬 종목이 장중에 되돌리나 -----
print("[추가3-2] 미국 대형주 시가·종가 불러오는 중... (DB에서 1~2분)")
c2_hold = load_sql(      # S&P500 구성(IVV 보유종목, 최신일) 비중 상위
    "SELECT 티커, 비중 FROM us_etf_holdings_daily WHERE ETF티커 = 'IVV' AND 종류 = 'EQUITY' "
    "AND 기준일 = (SELECT MAX(기준일) FROM us_etf_holdings_daily WHERE ETF티커 = 'IVV')"
)
c2_us_tickers = (c2_hold.sort_values("비중", ascending=False)["티커"].astype(str)
                 .str.replace(".", "-", regex=False).head(C2_US_TOP).tolist()) if len(c2_hold) else []
c2_gap_ok = len(c2_us_tickers) >= 30
if c2_gap_ok:
    c2_ud = load_sql(
        "SELECT 날짜, 티커, 시가, 종가 FROM us_price "
        f"WHERE 날짜 >= '{C2_US_START}' AND 티커 IN ({', '.join(repr(t) for t in c2_us_tickers)})"
    )
    c2_ud["날짜"] = pd.to_datetime(c2_ud["날짜"])
    c2_o = c2_ud.pivot(index="날짜", columns="티커", values="시가").sort_index()
    c2_c = c2_ud.pivot(index="날짜", columns="티커", values="종가").sort_index()
    c2_ugap = c2_o / c2_c.shift(1) - 1
    c2_uintra = c2_c / c2_o - 1
    bad = (c2_ugap.abs() > 0.25) | (c2_uintra.abs() > 0.25) | (c2_o <= 0)
    c2_ugap, c2_uintra = c2_ugap.mask(bad), c2_uintra.mask(bad)
    c2_ugap_rel = c2_ugap.sub(c2_ugap.mean(axis=1), axis=0)          # 시장 전체 갭을 뺀 '종목만의 갭'
    c2_next_cc = (c2_c.shift(-1) / c2_c - 1).mask(bad)               # 다음 날 종가->종가 (지연0 근사)
    c2_next2_cc = (c2_c.shift(-2) / c2_c.shift(-1) - 1).mask(bad)    # 그다음 날 종가->종가 (지연1 근사)
    c2_rk = c2_ugap_rel.rank(axis=1, pct=True)
    c2_n = c2_rk.notna().sum(axis=1)
    c2_lo = (c2_rk <= 0.1) & (c2_n >= 30).values[:, None]
    c2_hi = (c2_rk >= 0.9) & (c2_n >= 30).values[:, None]

    def c2_ls(ret):
        """갭 하위 10% 매수 - 갭 상위 10% 매도 (각 0.5씩)"""
        lo = ret.where(c2_lo).mean(axis=1)
        hi = ret.where(c2_hi).mean(axis=1)
        return (0.5 * lo - 0.5 * hi).dropna()

    c2_us_gross = c2_ls(c2_uintra)
    c2_us_net = c2_us_gross - 2 * C2_US_COST          # 시가에 열고 종가에 닫음: 매일 양쪽 전부 회전
    c2_us_rets = {"장중 갭반전 롱숏(비용 전)": c2_us_gross, "장중 갭반전 롱숏(비용 후)": c2_us_net,
                  "같은 신호 다음 날 종가→종가(비용 전)": c2_ls(c2_next_cc),
                  "같은 신호 그다음 날 종가→종가(비용 전)": c2_ls(c2_next2_cc)}
    c2_us_stats = stats_frame(c2_us_rets, 252)
    c2_us_year = yearly_table(c2_us_rets)
    c2_ic = []
    for d in c2_ugap_rel.index:
        a, b = c2_ugap_rel.loc[d], c2_uintra.loc[d]
        ok = a.notna() & b.notna()
        if ok.sum() >= 30:
            c2_ic.append(spearmanr(a[ok], b[ok])[0])
    c2_ic = pd.Series(c2_ic)
    c2_us_q = pd.DataFrame({"gap": c2_ugap_rel.stack(), "intra": c2_uintra.stack()}).dropna()
    c2_us_q["갭구간"] = pd.qcut(c2_us_q["gap"].rank(method="first"), 10, labels=[f"{i}" for i in range(1, 11)])
    c2_us_dec = c2_us_q.groupby("갭구간", observed=True).agg(평균갭=("gap", "mean"), 평균장중=("intra", "mean"), 표본=("intra", "size"))
    c2_us_dec.index.name = "갭 10분위(1=가장 하락)"
else:
    c2_us_stats = pd.DataFrame()
    c2_us_year = pd.DataFrame()
    c2_ic = pd.Series(dtype=float)
    c2_us_dec = pd.DataFrame()
    c2_us_rets = {}
    print("[추가3-2] IVV 보유종목 목록이 없어 미국 갭 반전은 건너뜀")

# ----- 8) 그림 -----
c2_fig1, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for k, r in c2_rets.items():
    axes[0].plot((1 + r).cumprod(), label=k, linewidth=1.8 if "상위" in k else 1.0)
axes[0].set_title("추가3-2: 반도체 ETF·주식 장중 매매 (비용 차감)")
axes[0].legend(fontsize=7)
axes[0].grid(True, alpha=0.3)
for k, r in c2_us_rets.items():
    axes[1].plot((1 + r).cumprod(), label=k)
axes[1].set_title("미국 대형주 갭 반전 롱숏")
axes[1].legend(fontsize=7)
axes[1].grid(True, alpha=0.3)
c2_fig1.tight_layout()
plt.show()

# ----- 9) 결과 자동 저장 -----
c2_etf_rows = c2_rule_tab_flat[c2_rule_tab_flat.index.str.startswith(C2_ETF["091160"])]
c2_best_etf = c2_etf_rows[c2_etf_rows["방향"] == "매수"]["1회 평균(비용 후)"].idxmax()
c2_auto = [
    f"- ETF 왕복 비용 {C2_COST['091160'] * 100:.2f}% vs 주식 {C2_COST['005930'] * 100:.2f}%: "
    f"ETF 1회 평균(비용 후)이 가장 큰 규칙 = {c2_best_etf} ({c2_etf_rows.loc[c2_best_etf, '1회 평균(비용 후)'] * 100:.3f}%, "
    f"t {c2_etf_rows.loc[c2_best_etf, 't값']:.2f}, {int(c2_etf_rows.loc[c2_best_etf, '매매 수'])}회)",
    "- 일간 샤프(비용 후): " + " / ".join(f"{k} {c2_stats.loc[k, '샤프']:.2f}" for k in c2_rets),
    "- 되돌림: 'SOXX 크게 상승 날 장중 매도' 1회 평균(비용 전) " + " / ".join(
        f"{n} {c2_rule_tab.loc[(n, 'SOXX 크게 상승(상위20%) 날 장중 매도'), '1회 평균(비용 전)'] * 100:.3f}% "
        f"(t {c2_rule_tab.loc[(n, 'SOXX 크게 상승(상위20%) 날 장중 매도'), 't값']:.2f})" for n in C2_NAMES.values()),
]
if c2_gap_ok:
    c2_auto.append(
        f"- 미국 갭 반전(대형주 {len(c2_us_tickers)}개): 갭-장중 순위상관 평균 {c2_ic.mean():.3f} (t {c2_t(c2_ic.values):.2f}), "
        f"롱숏 샤프 비용 전 {c2_us_stats.iloc[0]['샤프']:.2f} / 비용 후 {c2_us_stats.iloc[1]['샤프']:.2f} / "
        f"다음 날 종가→종가 {c2_us_stats.iloc[2]['샤프']:.2f}")
save_result(
    C2_TAG, "추가 전략 3-2: 한미 시차 - 반도체 ETF로 비용 줄이기, 확신 높은 날만, 되돌림, 미국 갭 반전",
    [
        ("한눈에", "\n".join(c2_auto)),
        ("설정", "\n".join([
            f"- 한국: {', '.join(f'{v}({k})' for k, v in C2_NAMES.items())} 를 한 모델로 (종목 구분 특성 is_hynix·is_etf)",
            f"- 미국 밤사이: {', '.join(C2_US)} 수정종가 수익, 한국 날짜보다 앞선 마지막 미국 거래일",
            f"- 정답: 그날 시가->종가 > 0, 입력: 최근 {C2_SEQ}거래일, {C2_TEST_START}년부터 매년 재학습",
            f"- 기준선: 학습 구간 예측확률의 평균 / 80% 분위(상위20%) / 90% 분위(상위10%) -> 확률이 그보다 높은 날만 매수",
            "- 'SOXX 크게 상승'·'갭 크게 상승' 기준도 그 전날까지의 확장 창 분위(미래 정보 없음)",
            f"- 왕복 비용(가정): " + ", ".join(f"{C2_NAMES[c]} {v * 100:.2f}%" for c, v in C2_COST.items()),
            f"- 미국 갭 반전: S&P500(IVV 보유종목) 비중 상위 {C2_US_TOP}종목(현재 구성), {C2_US_START}~, 시장 평균 갭을 뺀 종목 갭의 순위 하위 10% 매수·상위 10% 매도, 시가 진입·종가 청산, 한쪽 비용 {C2_US_COST * 100:.2f}%",
        ])),
        ("종목별 분류 성능 (표본 밖 AUC)", md_table(c2_cls)),
        ("규칙별 매매 1회 기준 (종목별)", md_table(c2_rule_tab_flat, pct_cols=("1회 평균(비용 전)", "1회 평균(비용 후)", "승률"), digits=2)),
        ("되돌림 진단: 미국 반도체 밤사이 수익·갭 5구간별 평균", md_table(c2_bucket, pct_cols=("평균 갭", "평균 장중"), digits=2)),
        ("일간 성과 (비용 차감)", md_table(c2_stats, pct_cols=("CAGR", "변동성", "MDD", "누적수익", "플러스비율", "시장에 있던 날 비율"))),
        ("연도별 수익률", md_table(c2_year, pct_cols="all")),
        ("미국 갭 반전 롱숏 성과", md_table(c2_us_stats, pct_cols=("CAGR", "변동성", "MDD", "누적수익", "플러스비율")) if len(c2_us_stats) else "-"),
        ("미국 갭 반전 연도별", md_table(c2_us_year, pct_cols="all") if len(c2_us_year) else "-"),
        ("미국 갭 10분위별 장중 평균", md_table(c2_us_dec, pct_cols=("평균갭", "평균장중")) if len(c2_us_dec) else "-"),
        ("BRAIN으로 옮길 때", "\n".join([
            "- 후보 식(검증 필요): -rank(open / ts_delay(close, 1) - 1) 를 업종·시장 중립으로.",
            "- BRAIN은 종가 기준 보유라 '장중' 되돌림은 그대로 안 잡힌다. 위 표의 '다음 날 종가→종가' 샤프가 BRAIN에서 기대할 수 있는 쪽에 가깝다.",
            "- '그다음 날' 샤프까지 양수면 지연 1에서도 살아 있을 가능성.",
        ])),
        ("재학습 기록", md_table(pd.DataFrame(c2_train_log).set_index("연도"))),
        ("주의", "\n".join([
            "- 비용은 가정값(C2_COST 바꿔서 확인). ETF는 시가 동시호가 호가 폭이 주식보다 넓을 수 있다.",
            "- '장중 매도'는 공매도 가정. 개인은 국내 공매도가 어려움 -> 연구용. ETF는 인버스 ETF로 비슷하게 할 수 있는지 따로 확인 필요.",
            "- 미국 종목은 '현재' S&P500 구성 상위라 생존 편향이 있다.",
        ])),
    ],
    {"규칙별_매매": c2_rule_tab_flat, "구간_진단": c2_bucket, "일별수익": pd.DataFrame(c2_rets), "연도별수익": c2_year,
     "미국갭_일별": pd.DataFrame(c2_us_rets) if c2_us_rets else pd.DataFrame(), "미국갭_10분위": c2_us_dec,
     "예측확률": pd.DataFrame({"날짜": c2_d[c2_mask], "종목": c2_code[c2_mask], "LSTM": c2_prob["LSTM"][c2_mask],
                              "로지스틱": c2_prob["로지스틱"][c2_mask], "장중수익": c2_r[c2_mask]}).set_index("날짜")},
    {"누적수익": c2_fig1},
    {"성과": c2_stats.to_dict(orient="index"), "AUC": c2_cls.to_dict(orient="index"),
     "미국갭": c2_us_stats.to_dict(orient="index") if len(c2_us_stats) else {},
     "미국갭_IC평균": float(c2_ic.mean()) if len(c2_ic) else None},
)


#%% [추가 전략 3-3] 한미 시차 반도체 ETF 전략 검증·보완: 비용 올리기·특정 연도 빼기·TIGER 반도체·고정 검증·보완 규칙
# 무엇: 3-2에서 살아남은 'KODEX 반도체 확률 상위 날 장중 매수'를 보고서에 넣기 전에 약점을 확인하고 보완한다.
#  1) 왕복 비용 0.05·0.10·0.15·0.20% + 손익분기 비용  2) 2020·2026 빼고, 플러스 해 수, 상위 2개 해 비중
#  3) 학습에 안 쓴 TIGER 반도체에 그대로 적용  4) 2024년까지 한 번만 학습한 모델로 2025~ 고정 검증
#  5) 보완: 두 모델 동의, SOXX 200일선 위일 때만, 확률 비례 금액, 변동성 맞춤 + 진 해·변동성 구간 진단
#  6) NAV 괴리율(실제 비용 단서), 되돌림을 인버스 ETF로, 여러 번 시험한 효과(t > 3.3)
# 실행: 맨 위 [1] 환경 설정 셀 먼저, 그다음 이 셀. LSTM 학습 10~15분
# 결과: 06_퀀트공부/딥러닝/결과/추가3-3_한미시차_검증_보완/ 에 자동 저장
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
from sklearn.linear_model import LogisticRegression

# ----- 설정 (바꿔 보면서 다시 돌려 보기) -----
C3_TAG = "추가3-3_한미시차_검증_보완"
C3_TRAIN = {"005930": "삼성전자", "000660": "SK하이닉스", "091160": "KODEX 반도체"}   # 추가 전략 3-2와 같은 학습 대상
C3_EXTRA = {"091230": "TIGER 반도체"}       # 학습에는 안 넣고 예측만 -> 다른 상품에서도 통하나
C3_ETF = ["091160", "091230"]
C3_NAMES = {**C3_TRAIN, **C3_EXTRA}
C3_US = ["SPY", "QQQ", "SOXX"]
C3_SEQ = 20
C3_TEST_START = 2014
C3_HOLD_START = 2025          # 고정 검증: 2024년까지 자료로 한 번만 학습 -> 2025년부터 예측
C3_EPOCHS = 15
C3_COSTS = [0.0005, 0.0010, 0.0015, 0.0020]   # ETF 왕복 비용 민감도
C3_COST_STOCK = 0.0025
C3_COST = 0.0005              # 기본 ETF 왕복 비용
C3_TARGET_VOL = 0.15          # 변동성 맞춤: 연 15% 목표 (넘으면 투자 금액 줄임, 1배 초과 안 함)
C3_DROP_YEARS = [2020, 2026]
C3_INVERSE = ["114800", "123310"]   # 반도체 인버스는 DB에 없어 코스피200 인버스로 되돌림 확인 (앞에서부터 있는 것 사용)
C3_T_CUT = 3.3                # 규칙을 45개쯤 시험했을 때 우연을 걸러내는 t 기준 (0.05/45 양쪽 ≈ 3.3)

# ----- 1) 데이터 (추가 전략 3-2와 같음 + TIGER 반도체·NAV·인버스) -----
c3_us_raw = load_sql(
    "SELECT 날짜, 티커, 종가, 수정종가 FROM us_price "
    f"WHERE 티커 IN ({', '.join(repr(t) for t in C3_US)}) AND 날짜 >= '2002-01-01'"
)
c3_us_raw["날짜"] = pd.to_datetime(c3_us_raw["날짜"])
c3_us_raw["가격"] = c3_us_raw["수정종가"].fillna(c3_us_raw["종가"])
c3_us_px = c3_us_raw.pivot(index="날짜", columns="티커", values="가격").sort_index()[C3_US].dropna()
c3_us = pd.DataFrame(index=c3_us_px.index)
for t in C3_US:
    c3_us[f"us_{t}"] = c3_us_px[t].pct_change(fill_method=None)
c3_us["us_SOXX_minus_SPY"] = c3_us["us_SOXX"] - c3_us["us_SPY"]
c3_us["us_SOXX_5d"] = c3_us_px["SOXX"].pct_change(5, fill_method=None)
c3_us = c3_us.dropna()
C3_US_COLS = list(c3_us.columns)
# 시장 국면: SOXX가 200일선 위인가 (미국 종가 기준, 한국 아침에 이미 앎)
c3_soxx_up200 = (c3_us_px["SOXX"] > c3_us_px["SOXX"].rolling(200).mean()).astype(float)
c3_soxx_q80 = c3_us["us_SOXX"].expanding(250).quantile(0.8).shift(1)

c3_kr_raw = load_sql(
    "SELECT 날짜, 종목코드, 시가, 고가, 저가, 종가, 거래량 FROM kor_price "
    f"WHERE 종목코드 IN ({', '.join(repr(k) for k in C3_NAMES if k not in C3_ETF)}) AND 날짜 >= '2002-01-01'"
)
c3_etf_all = C3_ETF + C3_INVERSE
c3_etf_raw = load_sql(
    "SELECT BAS_DD AS 날짜, ISU_CD AS 종목코드, TDD_OPNPRC AS 시가, TDD_HGPRC AS 고가, TDD_LWPRC AS 저가, "
    "TDD_CLSPRC AS 종가, ACC_TRDVOL AS 거래량, NAV FROM krx_etf_daily "
    f"WHERE ISU_CD IN ({', '.join(repr(k) for k in c3_etf_all)})"
).drop_duplicates(subset=["날짜", "종목코드"], keep="first")
c3_all_raw = pd.concat([c3_kr_raw, c3_etf_raw], ignore_index=True)
c3_all_raw["날짜"] = pd.to_datetime(c3_all_raw["날짜"])
for col in ["시가", "고가", "저가", "종가", "거래량", "NAV"]:
    c3_all_raw[col] = pd.to_numeric(c3_all_raw[col], errors="coerce")


def c3_align_us(dates, frame):
    """한국 날짜마다 '그 날짜보다 앞선 마지막 미국 거래일' 값"""
    left = pd.DataFrame({"날짜": pd.DatetimeIndex(dates)})
    right = frame.copy()
    right.index.name = "날짜"
    return pd.merge_asof(left, right.reset_index(), on="날짜", direction="backward",
                         allow_exact_matches=False).set_index("날짜")


C3_FEATS = C3_US_COLS + ["gap", "prev_intra", "prev_ret", "prev_ret5", "prev_volz", "is_hynix", "is_etf"]
c3_frames = {}
for code in C3_NAMES:
    k = c3_all_raw[c3_all_raw["종목코드"] == code].sort_values("날짜").set_index("날짜")
    k = k[(k["시가"] > 0) & (k["거래량"] > 0)].copy()
    k["gap"] = k["시가"] / k["종가"].shift(1) - 1
    k["intra"] = k["종가"] / k["시가"] - 1
    k["ret"] = k["종가"].pct_change(fill_method=None)
    k["prev_intra"] = k["intra"].shift(1)
    k["prev_ret"] = k["ret"].shift(1)
    k["prev_ret5"] = k["종가"].pct_change(5, fill_method=None).shift(1)
    lv = np.log(k["거래량"])
    k["prev_volz"] = ((lv - lv.rolling(20).mean()) / lv.rolling(20).std()).shift(1)
    k["is_hynix"] = 1.0 if code == "000660" else 0.0
    k["is_etf"] = 1.0 if code in C3_ETF else 0.0
    k["vol20"] = k["intra"].rolling(20).std().shift(1) * np.sqrt(252)   # 아침에 아는 최근 20일 장중 변동성
    ua = c3_align_us(k.index, c3_us)
    for col in C3_US_COLS:
        k[col] = ua[col].values
    k["soxx_up200"] = c3_align_us(k.index, c3_soxx_up200.to_frame("v"))["v"].values
    k["soxx_q80"] = c3_align_us(k.index, c3_soxx_q80.to_frame("v"))["v"].values
    k = k[(k["ret"].abs() < 0.35) | k["ret"].isna()]
    k = k[k["gap"].abs() < 0.3]
    c3_frames[code] = k
print(f"[추가3-3] 종목별 일수: " + ", ".join(f"{C3_NAMES[c]} {len(v)}" for c, v in c3_frames.items()))


def c3_make_seq(code, k):
    f = k[C3_FEATS].values
    out = {"X": [], "y": [], "r": [], "d": [], "vol": [], "up200": [], "soxx": [], "q80": []}
    for i in range(C3_SEQ - 1, len(k)):
        win = f[i - C3_SEQ + 1: i + 1]
        if np.isnan(win).any() or np.isnan(k["intra"].iloc[i]):
            continue
        out["X"].append(win)
        out["y"].append(int(k["intra"].iloc[i] > 0))
        out["r"].append(k["intra"].iloc[i])
        out["d"].append(k.index[i])
        out["vol"].append(k["vol20"].iloc[i])
        out["up200"].append(k["soxx_up200"].iloc[i])
        out["soxx"].append(k["us_SOXX"].iloc[i])
        out["q80"].append(k["soxx_q80"].iloc[i])
    out["code"] = [code] * len(out["y"])
    return out


c3_parts = [c3_make_seq(c, k) for c, k in c3_frames.items()]
c3_X = np.concatenate([np.array(p["X"]) for p in c3_parts])
c3_y = np.concatenate([p["y"] for p in c3_parts])
c3_r = np.concatenate([p["r"] for p in c3_parts])
c3_d = pd.DatetimeIndex(np.concatenate([np.array(p["d"], dtype="datetime64[ns]") for p in c3_parts]))
c3_vol = np.concatenate([p["vol"] for p in c3_parts]).astype(float)
c3_up200 = np.concatenate([p["up200"] for p in c3_parts]).astype(float)
c3_soxx = np.concatenate([p["soxx"] for p in c3_parts]).astype(float)
c3_q80 = np.concatenate([p["q80"] for p in c3_parts]).astype(float)
c3_code = np.concatenate([p["code"] for p in c3_parts])
c3_is_train = np.isin(c3_code, list(C3_TRAIN))
c3_nf = c3_X.shape[2]
print(f"[추가3-3] 시퀀스 {c3_X.shape} (TIGER 반도체는 예측만)")


# ----- 2) 모델: 매년 재학습(워크포워드) + 2024년까지 한 번만 학습(고정 검증) -----
def c3_build_lstm(seed):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    tf.random.set_seed(seed)
    inp = Input(shape=(C3_SEQ, c3_nf))
    x = LSTM(32)(inp)
    x = Dropout(0.2)(x)
    x = Dense(16, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)
    m = Model(inp, out, name="KR_US_Lag_LSTM_v3")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return m


def c3_fit_predict(tr, te):
    """반환: {모델: 예측구간 확률}, {모델: 예측구간 '학습구간 확률 분포 안 백분위'(0~1)}"""
    sc = StandardScaler().fit(c3_X[tr].reshape(-1, c3_nf))
    x_tr = sc.transform(c3_X[tr].reshape(-1, c3_nf)).reshape(-1, C3_SEQ, c3_nf)
    x_te = sc.transform(c3_X[te].reshape(-1, c3_nf)).reshape(-1, C3_SEQ, c3_nf)
    lr = LogisticRegression(C=0.1, max_iter=2000).fit(x_tr[:, -1, :], c3_y[tr])
    p_tr = {"로지스틱": lr.predict_proba(x_tr[:, -1, :])[:, 1]}
    p_te = {"로지스틱": lr.predict_proba(x_te[:, -1, :])[:, 1]}
    if TF_AVAILABLE:
        from tensorflow.keras.callbacks import EarlyStopping
        m = c3_build_lstm(SEED)
        m.fit(x_tr, c3_y[tr], epochs=C3_EPOCHS, batch_size=128, validation_split=0.15, verbose=0,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
        p_tr["LSTM"] = m.predict(x_tr, verbose=0).ravel()
        p_te["LSTM"] = m.predict(x_te, verbose=0).ravel()
    else:
        p_tr["LSTM"], p_te["LSTM"] = p_tr["로지스틱"], p_te["로지스틱"]
    pct = {k: np.searchsorted(np.sort(p_tr[k]), p_te[k], side="right") / len(p_tr[k]) for k in p_te}
    return p_te, pct


c3_pct = {"로지스틱": np.full(len(c3_y), np.nan), "LSTM": np.full(len(c3_y), np.nan)}
for yr in range(C3_TEST_START, c3_d.max().year + 1):
    te = np.where(c3_d.year == yr)[0]
    tr = np.where((c3_d < pd.Timestamp(f"{yr}-01-01")) & c3_is_train)[0]
    if len(te) == 0 or len(tr) < 1000:
        continue
    _, pct = c3_fit_predict(tr, te)
    for k in pct:
        c3_pct[k][te] = pct[k]
    print(f"  {yr}년 모델: 학습 {len(tr)}개 -> {len(te)}개 예측")
c3_pct_hold = {"로지스틱": np.full(len(c3_y), np.nan), "LSTM": np.full(len(c3_y), np.nan)}
te = np.where(c3_d >= pd.Timestamp(f"{C3_HOLD_START}-01-01"))[0]
tr = np.where((c3_d < pd.Timestamp(f"{C3_HOLD_START}-01-01")) & c3_is_train)[0]
_, pct = c3_fit_predict(tr, te)
for k in pct:
    c3_pct_hold[k][te] = pct[k]
print(f"  고정 검증 모델: {C3_HOLD_START - 1}년까지 {len(tr)}개로 한 번 학습 -> {len(te)}개 예측")

c3_mask = ~np.isnan(c3_pct["로지스틱"])
C3_LN = "LSTM" if TF_AVAILABLE else "로지스틱(대체)"


def c3_rules(pct):
    L, S = pct["로지스틱"], pct["LSTM"]
    return {"로지스틱 상위20%": L > 0.8, "로지스틱 상위10%": L > 0.9,
            f"{C3_LN} 상위20%": S > 0.8, f"{C3_LN} 상위10%": S > 0.9,
            "두 모델 동의 상위20%": (L > 0.8) & (S > 0.8)}


c3_R = c3_rules(c3_pct)


def c3_t(x):
    x = np.asarray(x, dtype=float)
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 and x.std() > 0 else np.nan


def c3_daily(sig, code, cost, w=None, mask=None):
    m = (c3_mask if mask is None else mask) & (c3_code == code)
    ww = np.ones(len(c3_r)) if w is None else np.nan_to_num(w, nan=0.0)
    r = np.where(sig & m, ww * (c3_r - cost), 0.0)
    return pd.Series(r[m], index=c3_d[m]).sort_index()


# ----- 3) 비용 민감도 + 다른 ETF(TIGER) + 여러 번 시험한 효과 -----
c3_cost_rows = []
for code in C3_ETF:
    for rn, sig in c3_R.items():
        x = c3_r[sig & c3_mask & (c3_code == code)]
        row = {"매매 수": len(x), "1회 평균(비용 전)": x.mean() if len(x) else np.nan, "t값": c3_t(x),
               f"t>{C3_T_CUT}": "예" if c3_t(x) > C3_T_CUT else "아니오",
               "손익분기 왕복비용": x.mean() if len(x) else np.nan}
        for cst in C3_COSTS:
            s = perf_stats(c3_daily(sig, code, cst), 252)
            row[f"샤프@{cst * 100:.2f}%"] = s["샤프"]
            row[f"CAGR@{cst * 100:.2f}%"] = s["CAGR"]
        c3_cost_rows.append({"상품 · 규칙": f"{C3_NAMES[code]} · {rn}", **row})
c3_cost_tab = pd.DataFrame(c3_cost_rows).set_index("상품 · 규칙")
c3_stock_rows = []
for code in ["005930", "000660"]:
    for rn, sig in c3_R.items():
        x = c3_r[sig & c3_mask & (c3_code == code)]
        s = perf_stats(c3_daily(sig, code, C3_COST_STOCK), 252)
        c3_stock_rows.append({"종목 · 규칙": f"{C3_NAMES[code]} · {rn}", "매매 수": len(x),
                              "1회 평균(비용 전)": x.mean(), "t값": c3_t(x), f"t>{C3_T_CUT}": "예" if c3_t(x) > C3_T_CUT else "아니오",
                              "1회 평균(비용 후)": x.mean() - C3_COST_STOCK, "샤프(비용 후)": s["샤프"]})
c3_stock_tab = pd.DataFrame(c3_stock_rows).set_index("종목 · 규칙")

# ----- 4) 약점 보완 규칙: 두 모델 동의, 반도체 상승장만, 확률 비례 금액, 변동성 맞춤 -----
c3_L = c3_pct["로지스틱"]
c3_w_prob = np.clip(0.5 + 2.5 * (c3_L - 0.8), 0.5, 1.0)         # 백분위 80% -> 0.5, 100% -> 1.0
c3_w_vol = np.minimum(1.0, C3_TARGET_VOL / np.where(c3_vol > 0, c3_vol, np.nan))
c3_up = c3_up200 > 0.5
c3_strats = {}
for code in C3_ETF:
    n = C3_NAMES[code]
    c3_strats[f"{n} 로지스틱 상위20%"] = c3_daily(c3_R["로지스틱 상위20%"], code, C3_COST)
    c3_strats[f"{n} 두 모델 동의"] = c3_daily(c3_R["두 모델 동의 상위20%"], code, C3_COST)
    c3_strats[f"{n} 로지스틱 상위20% + SOXX 200일선 위"] = c3_daily(c3_R["로지스틱 상위20%"] & c3_up, code, C3_COST)
    c3_strats[f"{n} 두 모델 동의 + SOXX 200일선 위"] = c3_daily(c3_R["두 모델 동의 상위20%"] & c3_up, code, C3_COST)
    c3_strats[f"{n} 로지스틱 상위20% 확률비례"] = c3_daily(c3_R["로지스틱 상위20%"], code, C3_COST, w=c3_w_prob)
    c3_strats[f"{n} 로지스틱 상위20% 변동성맞춤"] = c3_daily(c3_R["로지스틱 상위20%"], code, C3_COST, w=c3_w_vol)
kd = c3_frames["091160"]["ret"]
c3_strats["KODEX 반도체 보유(참고)"] = kd.reindex(c3_strats["KODEX 반도체 로지스틱 상위20%"].index).fillna(0.0)


def c3_extra(r):
    r = r.dropna()
    yr = (1 + r).groupby(r.index.year).prod() - 1
    keep = r[~r.index.year.isin(C3_DROP_YEARS)]
    lg = np.log1p(yr)
    top2 = lg.sort_values(ascending=False).head(2).sum() / lg.sum() if lg.sum() > 0 else np.nan
    s2 = perf_stats(keep, 252)
    return {"플러스 해": f"{int((yr > 0).sum())}/{len(yr)}", "연도 중앙값": yr.median(),
            f"{'·'.join(map(str, C3_DROP_YEARS))} 뺀 샤프": s2["샤프"], f"{'·'.join(map(str, C3_DROP_YEARS))} 뺀 CAGR": s2["CAGR"],
            "상위 2개 해 비중": top2}


c3_stats = stats_frame(c3_strats, 252)
c3_ex = pd.DataFrame({k: c3_extra(v) for k, v in c3_strats.items()}).T
c3_stats = pd.concat([c3_stats, c3_ex], axis=1)
c3_stats.index.name = "전략"
c3_year = yearly_table(c3_strats)

# ----- 5) 진 해 진단 + 변동성 구간 -----
c3_soxx_y = (1 + c3_us["us_SOXX"]).groupby(c3_us.index.year).prod() - 1
c3_soxx_v = c3_us["us_SOXX"].groupby(c3_us.index.year).std() * np.sqrt(252)
c3_base = c3_strats["KODEX 반도체 로지스틱 상위20%"]
c3_diag_year = pd.DataFrame({
    "SOXX 연수익": c3_soxx_y, "SOXX 변동성": c3_soxx_v,
    "KODEX 반도체 보유": (1 + kd).groupby(kd.index.year).prod() - 1,
    "전략(로지스틱 상위20%)": (1 + c3_base).groupby(c3_base.index.year).prod() - 1,
    "SOXX 200일선 위 날 비율": c3_soxx_up200.groupby(c3_soxx_up200.index.year).mean(),
}).loc[lambda x: x.index >= C3_TEST_START].dropna(subset=["전략(로지스틱 상위20%)"])
c3_diag_year.index.name = "연도"
c3_vol_rows = []
for code in C3_ETF:
    m = c3_mask & (c3_code == code) & ~np.isnan(c3_vol)
    cuts = np.nanquantile(c3_vol[m], [1 / 3, 2 / 3])
    lab = np.where(c3_vol < cuts[0], "1 낮음", np.where(c3_vol < cuts[1], "2 중간", "3 높음"))
    for b in ["1 낮음", "2 중간", "3 높음"]:
        x = c3_r[m & c3_R["로지스틱 상위20%"] & (lab == b)]
        allx = c3_r[m & (lab == b)]
        c3_vol_rows.append({"상품 · 변동성 구간": f"{C3_NAMES[code]} · {b}", "매매 수": len(x),
                            "1회 평균(비용 전)": x.mean() if len(x) else np.nan, "t값": c3_t(x),
                            "그 구간 매일 평균": allx.mean()})
c3_vol_tab = pd.DataFrame(c3_vol_rows).set_index("상품 · 변동성 구간")

# ----- 6) 실제 비용 단서: 종가와 NAV의 차이 (괴리율) -----
c3_nav_rows = []
for code in C3_ETF:
    k = c3_frames[code]
    prem = (k["종가"] / k["NAV"] - 1).replace([np.inf, -np.inf], np.nan).dropna()
    prem = prem[prem.abs() < 0.05]
    c3_nav_rows.append({"상품": C3_NAMES[code], "괴리율 절댓값 평균": prem.abs().mean(),
                        "중앙값": prem.abs().median(), "90% 분위": prem.abs().quantile(0.9),
                        "최근 1년 평균": prem[prem.index >= prem.index.max() - pd.Timedelta(days=365)].abs().mean()})
c3_nav_tab = pd.DataFrame(c3_nav_rows).set_index("상품")

# ----- 7) 되돌림을 실제 상품으로: SOXX 크게 오른 다음 날 인버스 ETF 장중 매수 -----
c3_inv_tab = pd.DataFrame()
c3_inv_code = None
for ic in C3_INVERSE:
    k = c3_all_raw[c3_all_raw["종목코드"] == ic].sort_values("날짜").set_index("날짜")
    k = k[(k["시가"] > 0) & (k["종가"] > 0)]
    if len(k) >= 500:
        c3_inv_code = ic
        k = k.copy()
        k["intra"] = k["종가"] / k["시가"] - 1
        k = k[k["intra"].abs() < 0.15]
        ua = c3_align_us(k.index, c3_us[["us_SOXX"]])
        q = c3_align_us(k.index, c3_soxx_q80.to_frame("v"))["v"].values
        big = ua["us_SOXX"].values > q
        x, allx = k["intra"].values[big], k["intra"].values
        c3_inv_tab = pd.DataFrame([
            {"규칙": f"인버스({ic}) · SOXX 크게 오른 다음 날 장중 매수", "매매 수": len(x), "1회 평균(비용 전)": x.mean(),
             "t값": c3_t(x), "1회 평균(비용 후)": x.mean() - C3_COST, "승률": (x > 0).mean()},
            {"규칙": f"인버스({ic}) · 매일 장중 매수(비교)", "매매 수": len(allx), "1회 평균(비용 전)": allx.mean(),
             "t값": c3_t(allx), "1회 평균(비용 후)": allx.mean() - C3_COST, "승률": (allx > 0).mean()},
        ]).set_index("규칙")
        break

# ----- 8) 고정 검증: 2024년까지로 정한 모델이 2025~에도 통하나 -----
c3_R_hold = c3_rules(c3_pct_hold)
c3_hold_rows = []
hm = c3_d >= pd.Timestamp(f"{C3_HOLD_START}-01-01")
for code in C3_NAMES:
    cost = C3_COST if code in C3_ETF else C3_COST_STOCK
    for rn in ["로지스틱 상위20%", f"{C3_LN} 상위20%", "두 모델 동의 상위20%"]:
        for lab, R in [("매년 재학습", c3_R), ("2024년까지 고정", c3_R_hold)]:
            x = c3_r[R[rn] & hm & (c3_code == code)]
            c3_hold_rows.append({"상품 · 규칙 · 방식": f"{C3_NAMES[code]} · {rn} · {lab}", "매매 수": len(x),
                                 "1회 평균(비용 전)": x.mean() if len(x) else np.nan, "t값": c3_t(x),
                                 "1회 평균(비용 후)": x.mean() - cost if len(x) else np.nan})
c3_hold_tab = pd.DataFrame(c3_hold_rows).set_index("상품 · 규칙 · 방식")

# ----- 9) 그림 -----
c3_fig1, axes = plt.subplots(1, 2, figsize=(14, 4.8))
for k in ["KODEX 반도체 로지스틱 상위20%", "KODEX 반도체 두 모델 동의", "KODEX 반도체 두 모델 동의 + SOXX 200일선 위",
          "KODEX 반도체 로지스틱 상위20% 변동성맞춤", "TIGER 반도체 로지스틱 상위20%", "KODEX 반도체 보유(참고)"]:
    axes[0].plot((1 + c3_strats[k]).cumprod(), label=k, linewidth=1.0 if "참고" in k else 1.5)
axes[0].set_yscale("log")
axes[0].set_title("추가3-3: 보완 규칙 누적수익 (왕복 0.05%, 로그 눈금)")
axes[0].legend(fontsize=7)
axes[0].grid(True, alpha=0.3)
for cst in C3_COSTS:
    axes[1].plot((1 + c3_daily(c3_R["로지스틱 상위20%"], "091160", cst)).cumprod(), label=f"KODEX 로지스틱 상위20% 비용 {cst * 100:.2f}%")
axes[1].set_yscale("log")
axes[1].set_title("비용을 올리면")
axes[1].legend(fontsize=7)
axes[1].grid(True, alpha=0.3)
c3_fig1.tight_layout()
plt.show()

# ----- 10) 결과 자동 저장 -----
k1, k2 = "KODEX 반도체 · 로지스틱 상위20%", "TIGER 반도체 · 로지스틱 상위20%"
c3_auto = [
    f"- 손익분기 왕복비용: KODEX 로지스틱 상위20% {c3_cost_tab.loc[k1, '손익분기 왕복비용'] * 100:.3f}%, "
    f"비용 0.15%일 때 샤프 {c3_cost_tab.loc[k1, '샤프@0.15%']:.2f} (0.05%일 때 {c3_cost_tab.loc[k1, '샤프@0.05%']:.2f})",
    f"- 학습에 안 쓴 TIGER 반도체: 로지스틱 상위20% 1회 {c3_cost_tab.loc[k2, '1회 평균(비용 전)'] * 100:.3f}% (t {c3_cost_tab.loc[k2, 't값']:.2f}), 샤프 {c3_cost_tab.loc[k2, '샤프@0.05%']:.2f}",
    "- 2020·2026 뺀 샤프: " + " / ".join(f"{k} {c3_stats.loc[k, '2020·2026 뺀 샤프']:.2f}" for k in
                                        ["KODEX 반도체 로지스틱 상위20%", "KODEX 반도체 두 모델 동의", "KODEX 반도체 두 모델 동의 + SOXX 200일선 위", "KODEX 반도체 로지스틱 상위20% 변동성맞춤"]),
    "- 플러스 해: " + " / ".join(f"{k} {c3_stats.loc[k, '플러스 해']}" for k in
                                ["KODEX 반도체 로지스틱 상위20%", "KODEX 반도체 두 모델 동의", "KODEX 반도체 두 모델 동의 + SOXX 200일선 위"]),
    f"- t > {C3_T_CUT} 통과 규칙(ETF): " + (", ".join(c3_cost_tab.index[c3_cost_tab[f't>{C3_T_CUT}'] == '예']) or "없음"),
    f"- KODEX 반도체 종가 NAV 괴리율 평균 {c3_nav_tab.loc['KODEX 반도체', '괴리율 절댓값 평균'] * 100:.3f}% (호가 비용의 대략적 크기)",
]
save_result(
    C3_TAG, "추가 전략 3-3: 한미 시차 반도체 ETF 전략 검증·보완 (비용, 특정 연도, 다른 ETF, 고정 검증)",
    [
        ("한눈에", "\n".join(c3_auto)),
        ("설정", "\n".join([
            f"- 모델·특성·학습 대상은 추가 전략 3-2와 같음({', '.join(C3_TRAIN.values())}), {C3_TEST_START}년부터 매년 재학습",
            f"- {', '.join(C3_EXTRA.values())}: 학습에 넣지 않고 예측만 (다른 상품 재확인)",
            "- 규칙 기준선: 예측확률이 그해 학습구간 확률 분포에서 몇 % 위치인지(백분위). 상위20% = 80% 넘는 날",
            f"- 보완: 두 모델 동의 / SOXX 200일선 위일 때만 / 확률 비례 금액(0.5~1) / 변동성 맞춤(연 {C3_TARGET_VOL * 100:.0f}% 목표, 1배 이하)",
            f"- 고정 검증: {C3_HOLD_START - 1}년까지 자료로 한 번만 학습해 {C3_HOLD_START}년~ 예측, 매년 재학습과 비교",
            f"- 여러 번 시험한 효과: t > {C3_T_CUT} 인 것만 '통과'로 표시",
        ])),
        ("비용 민감도 (ETF, 손익분기 왕복비용 = 비용 전 1회 평균)", md_table(c3_cost_tab, pct_cols=("1회 평균(비용 전)", "손익분기 왕복비용") + tuple(f"CAGR@{c * 100:.2f}%" for c in C3_COSTS), digits=2)),
        ("주식 참고 (왕복 0.25%)", md_table(c3_stock_tab, pct_cols=("1회 평균(비용 전)", "1회 평균(비용 후)"), digits=2)),
        ("보완 규칙 성과 (일간, 왕복 0.05%)", md_table(c3_stats, pct_cols=("CAGR", "변동성", "MDD", "누적수익", "플러스비율", "연도 중앙값", "2020·2026 뺀 CAGR", "상위 2개 해 비중"), digits=2)),
        ("연도별 수익률", md_table(c3_year, pct_cols="all")),
        ("진 해 진단 (반도체 시장 국면과 같이 보기)", md_table(c3_diag_year, pct_cols="all")),
        ("변동성 구간별 1회 수익 (로지스틱 상위20%)", md_table(c3_vol_tab, pct_cols=("1회 평균(비용 전)", "그 구간 매일 평균"), digits=2)),
        ("ETF 종가 NAV 괴리율 (실제 비용 단서)", md_table(c3_nav_tab, pct_cols="all")),
        ("되돌림을 인버스 ETF로", md_table(c3_inv_tab, pct_cols=("1회 평균(비용 전)", "1회 평균(비용 후)", "승률"), digits=2) if len(c3_inv_tab) else "인버스 ETF 자료 없음"),
        ("고정 검증 (2025년~)", md_table(c3_hold_tab, pct_cols=("1회 평균(비용 전)", "1회 평균(비용 후)"), digits=2)),
        ("주의", "\n".join([
            "- 시가 체결 가정. 괴리율은 종가 기준이라 시가 동시호가 비용의 대략적 크기만 보여 준다.",
            "- 인버스는 코스피200 인버스라 반도체만의 되돌림이 아니라 시장 전체 되돌림을 본다.",
            "- 보완 규칙도 여러 개 시험했다. 한 규칙만 좋은지, 여러 보완이 같은 방향인지 같이 볼 것.",
        ])),
    ],
    {"비용민감도": c3_cost_tab, "주식참고": c3_stock_tab, "보완성과": c3_stats, "일별수익": pd.DataFrame(c3_strats),
     "연도별수익": c3_year, "진해진단": c3_diag_year, "변동성구간": c3_vol_tab, "NAV괴리": c3_nav_tab,
     "인버스": c3_inv_tab, "고정검증": c3_hold_tab},
    {"누적수익": c3_fig1},
    {"비용민감도": c3_cost_tab.to_dict(orient="index"), "보완성과": c3_stats.to_dict(orient="index"),
     "고정검증": c3_hold_tab.to_dict(orient="index")},
)
