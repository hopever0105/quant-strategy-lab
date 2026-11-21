# -*- coding: utf-8 -*-
"""
자산 히트맵 CNN: 시장 타이밍·급락 위험 (추가 전략 2, 2-2)
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
import shutil
import subprocess
import urllib.parse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
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


#%% [추가 전략 2] 자산 히트맵 CNN: 10개 자산 20일 수익률 그림으로 SPY 넣고 빼기 
# 무엇: 위 캔들차트 CNN은 '한 종목 캔들차트' 그림이었다. 여기서는 '여러 자산의 최근 20일 움직임'을 그림 한 장으로 본다.
#  - 세로 20일 x 가로 10자산(주식·채권·리츠·금·원자재) 표준화 수익률 = 20x10 흑백 그림
#  - CNN이 '채권은 오르고 주식은 빠지는' 같은 자산 간 패턴을 찾아 SPY 다음 5일 상승 확률을 냄
#  - 확률이 평소보다 높으면 SPY 보유, 아니면 현금. 매년 재학습
# 비교: 로지스틱(같은 그림을 펴서 선형 모델), 200일선 규칙, SPY 보유 / 가리기 진단으로 CNN이 본 자산 확인
# 실행: 맨 위 [1] 환경 설정 셀 먼저, 그다음 이 셀. TF 있으면 10분 안팎
# 결과: 06_퀀트공부/딥러닝/결과/추가2_CNN_자산히트맵_타이밍/ 에 자동 저장
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef

# ----- 설정 -----
B_TAG = "추가2_CNN_자산히트맵_타이밍"
B_ASSETS = ["SPY", "IEV", "EWJ", "EEM", "TLT", "IEF", "IYR", "RWX", "GLD", "DBC"]
B_LOOKBACK = 20            # 그림 세로: 최근 20거래일
B_HORIZON = 5              # 맞힐 것: SPY 다음 5거래일 수익률 > 0
B_COST = 0.0005            # SPY 들어가고 나올 때 한쪽 0.05% (가정)
B_TEST_START = 2012
B_EPOCHS = 15

# ----- 1) 데이터: 10개 ETF 일별 수익률 -> 자기 변동성으로 나눈 '표준화 수익률' -----
b_raw = load_sql("SELECT * FROM sample_etf")
b_raw["Date"] = pd.to_datetime(b_raw["Date"])
b_px = b_raw.set_index("Date").sort_index()[B_ASSETS].astype(float)
b_ret = b_px.pct_change(fill_method=None)
# 과거 252일 변동성으로만 나눔 (미래 정보 없음) -> 채권·주식·원자재가 같은 크기로 보이게
b_z = (b_ret / b_ret.rolling(252, min_periods=126).std()).clip(-4, 4)
b_z = b_z.dropna()          # 10개 모두 값이 있는 날만 (RWX·DBC 상장 이후)
b_spy_fwd = b_px["SPY"].shift(-B_HORIZON) / b_px["SPY"] - 1

# ----- 2) 그림 만들기: (20일 x 10자산 x 1채널) 히트맵 한 장 = 하루 -----
b_vals = b_z.values
b_X = np.stack([b_vals[i - B_LOOKBACK + 1: i + 1] for i in range(B_LOOKBACK - 1, len(b_vals))])[..., None]
b_dates = b_z.index[B_LOOKBACK - 1:]
b_fwd = b_spy_fwd.reindex(b_dates)
b_y = (b_fwd > 0).astype(int).values
b_has_y = b_fwd.notna().values
print(f"[추가2] 히트맵 {b_X.shape} ({b_dates.min().date()} ~ {b_dates.max().date()}), SPY 5일 상승 비율 {b_y[b_has_y].mean():.3f}")


# ----- 3) 모델: 작은 CNN (위 캔들차트 CNN 구조를 20x10 그림에 맞게 축소) + 비교용 로지스틱 -----
def b_build_cnn(seed):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
    tf.random.set_seed(seed)
    inp = Input(shape=(B_LOOKBACK, len(B_ASSETS), 1))
    x = Conv2D(16, (3, 3), padding="same", activation="relu")(inp)   # 3일 x 3자산 패턴
    x = MaxPooling2D((2, 1))(x)                                       # 시간축만 절반 (자산 수는 유지)
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation="sigmoid")(x)
    m = Model(inp, out, name="Asset_Heatmap_CNN")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m


b_prob_cnn = pd.Series(np.nan, index=b_dates)
b_prob_lr = pd.Series(np.nan, index=b_dates)
b_thr_cnn, b_thr_lr = {}, {}
# 가리기(occlusion) 진단: 자산 하나를 0으로 가렸을 때 예측이 얼마나 나빠지나 -> CNN이 본 자산
b_occ_names = B_ASSETS + ["최근 5일 전부", "6~20일 전 전부"]
b_prob_occ = {n: pd.Series(np.nan, index=b_dates) for n in b_occ_names}
b_lr_coef_last = None
b_train_log = []

for yr in range(B_TEST_START, b_dates.max().year + 1):
    te_pos = np.where(b_dates.year == yr)[0]
    if len(te_pos) == 0:
        continue
    first = te_pos[0]
    # 학습 표본의 정답(5일 뒤 가격)이 첫 예측일 전에 다 나와 있어야 함 -> 5일 비우기(엠바고)
    tr_pos = np.arange(0, max(first - B_HORIZON, 0))
    tr_pos = tr_pos[b_has_y[tr_pos]]
    if len(tr_pos) < 500:
        continue
    x_tr, y_tr, x_te = b_X[tr_pos], b_y[tr_pos], b_X[te_pos]

    # 로지스틱 (그림을 한 줄로 펴서 선형 모델) - 항상 계산해서 CNN과 비교
    lr = LogisticRegression(C=0.05, max_iter=3000)
    lr.fit(x_tr.reshape(len(x_tr), -1), y_tr)
    b_prob_lr.iloc[te_pos] = lr.predict_proba(x_te.reshape(len(x_te), -1))[:, 1]
    b_thr_lr[yr] = lr.predict_proba(x_tr.reshape(len(x_tr), -1))[:, 1].mean()
    b_lr_coef_last = lr.coef_.reshape(B_LOOKBACK, len(B_ASSETS))

    if TF_AVAILABLE:
        from tensorflow.keras.callbacks import EarlyStopping
        cnn = b_build_cnn(SEED)
        cnn.fit(x_tr, y_tr, epochs=B_EPOCHS, batch_size=64, validation_split=0.15, verbose=0,
                callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
        predict = lambda arr: cnn.predict(arr, verbose=0).ravel()
    else:
        # TF 없으면 CNN 자리에 sklearn MLP(펼친 그림 입력)로 대체
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(hidden_layer_sizes=(64, 16), alpha=1e-3, max_iter=300, early_stopping=True,
                            validation_fraction=0.15, random_state=SEED)
        mlp.fit(x_tr.reshape(len(x_tr), -1), y_tr)
        predict = lambda arr: mlp.predict_proba(arr.reshape(len(arr), -1))[:, 1]
    b_prob_cnn.iloc[te_pos] = predict(x_te)
    b_thr_cnn[yr] = predict(x_tr).mean()   # 기준선 = 학습 구간 평균 확률 (평소보다 낙관적일 때만 매수)

    for j, n in enumerate(B_ASSETS):
        xo = x_te.copy()
        xo[:, :, j, 0] = 0.0
        b_prob_occ[n].iloc[te_pos] = predict(xo)
    xo = x_te.copy()
    xo[:, -5:, :, 0] = 0.0
    b_prob_occ["최근 5일 전부"].iloc[te_pos] = predict(xo)
    xo = x_te.copy()
    xo[:, :-5, :, 0] = 0.0
    b_prob_occ["6~20일 전 전부"].iloc[te_pos] = predict(xo)
    b_train_log.append({"연도": yr, "학습표본": len(tr_pos), "예측일수": len(te_pos),
                        "CNN 기준확률": b_thr_cnn[yr], "로지스틱 기준확률": b_thr_lr[yr]})
    print(f"  {yr}년 모델: 학습 {len(tr_pos)}일 -> {len(te_pos)}일 예측")

# ----- 4) 분류 성능 -----
b_mask = b_prob_cnn.notna().values & b_has_y
b_auc_cnn = roc_auc_score(b_y[b_mask], b_prob_cnn.values[b_mask])
b_auc_lr = roc_auc_score(b_y[b_mask], b_prob_lr.values[b_mask])
b_thr_series_cnn = pd.Series([b_thr_cnn.get(d.year, np.nan) for d in b_dates], index=b_dates)
b_thr_series_lr = pd.Series([b_thr_lr.get(d.year, np.nan) for d in b_dates], index=b_dates)
b_sig_cnn = (b_prob_cnn > b_thr_series_cnn).astype(int)
b_sig_lr = (b_prob_lr > b_thr_series_lr).astype(int)
b_cls_tab = pd.DataFrame({
    "AUC": [b_auc_cnn, b_auc_lr],
    "정확도": [accuracy_score(b_y[b_mask], b_sig_cnn.values[b_mask]), accuracy_score(b_y[b_mask], b_sig_lr.values[b_mask])],
    "MCC": [matthews_corrcoef(b_y[b_mask], b_sig_cnn.values[b_mask]), matthews_corrcoef(b_y[b_mask], b_sig_lr.values[b_mask])],
    "실제 상승 비율": [b_y[b_mask].mean()] * 2,
    "매수 신호 비율": [b_sig_cnn.values[b_mask].mean(), b_sig_lr.values[b_mask].mean()],
}, index=["CNN" if TF_AVAILABLE else "MLP(대체)", "로지스틱"])
b_cls_tab.index.name = "모델"

b_auc_year = {}
for yr in sorted(set(b_dates[b_mask].year)):
    mk = b_mask & (b_dates.year == yr)
    if len(np.unique(b_y[mk])) > 1:
        b_auc_year[yr] = {"CNN AUC": roc_auc_score(b_y[mk], b_prob_cnn.values[mk]),
                          "로지스틱 AUC": roc_auc_score(b_y[mk], b_prob_lr.values[mk])}
b_auc_year = pd.DataFrame(b_auc_year).T
b_auc_year.index.name = "연도"

b_occ_tab = pd.DataFrame({"가렸을 때 AUC": {n: roc_auc_score(b_y[b_mask], b_prob_occ[n].values[b_mask]) for n in b_occ_names}})
b_occ_tab["AUC 하락폭"] = b_auc_cnn - b_occ_tab["가렸을 때 AUC"]
b_occ_tab = b_occ_tab.sort_values("AUC 하락폭", ascending=False)
b_occ_tab.index.name = "가린 부분"

# ----- 5) 매매: 신호 1이면 다음 날 SPY 보유, 0이면 현금 (신호는 그날 종가에 계산 -> 다음 날 수익에 적용) -----
b_test_days = b_dates[b_prob_cnn.notna().values]
b_spy_ret = b_px["SPY"].pct_change(fill_method=None)


def b_backtest(sig):
    pos = sig.reindex(b_test_days).astype(float)
    r = pos.shift(1) * b_spy_ret.reindex(b_test_days)
    cost = pos.diff().abs().fillna(pos.iloc[0]) * B_COST
    return (r - cost.shift(1).fillna(0.0)).iloc[1:], float(pos.mean()), int(pos.diff().abs().sum())


b_ma200 = (b_px["SPY"] > b_px["SPY"].rolling(200).mean()).astype(int)
b_r_cnn, b_in_cnn, b_sw_cnn = b_backtest(b_sig_cnn)
b_r_lr, b_in_lr, b_sw_lr = b_backtest(b_sig_lr)
b_r_ma, b_in_ma, b_sw_ma = b_backtest(b_ma200)
b_r_spy = b_spy_ret.reindex(b_test_days).iloc[1:]
b_name_cnn = "CNN 신호" if TF_AVAILABLE else "MLP(대체) 신호"
b_rets = {b_name_cnn: b_r_cnn, "로지스틱 신호": b_r_lr, "200일선 규칙": b_r_ma, "SPY 보유": b_r_spy}
b_stats = stats_frame(b_rets, 252)
b_stats["시장 보유 비율"] = [b_in_cnn, b_in_lr, b_in_ma, 1.0]
b_stats["매매 횟수"] = [b_sw_cnn, b_sw_lr, b_sw_ma, 0]
b_year = yearly_table(b_rets)
b_years_beat = int((b_year[b_name_cnn] > b_year["SPY 보유"]).sum())
# 신호 1인 날 vs 0인 날, 다음 날 SPY 평균 수익 (신호가 정말 갈라 주는지)
b_next = b_spy_ret.shift(-1).reindex(b_test_days)
b_split = pd.DataFrame({
    "신호1 다음날 평균": [b_next[b_sig_cnn.reindex(b_test_days) == 1].mean(), b_next[b_sig_lr.reindex(b_test_days) == 1].mean()],
    "신호0 다음날 평균": [b_next[b_sig_cnn.reindex(b_test_days) == 0].mean(), b_next[b_sig_lr.reindex(b_test_days) == 0].mean()],
}, index=[b_name_cnn, "로지스틱 신호"])
b_split["차이"] = b_split["신호1 다음날 평균"] - b_split["신호0 다음날 평균"]
b_split.index.name = "신호"

# ----- 6) 그림 -----
b_fig1, ax = plt.subplots(figsize=(11, 4.5))
for k, r in b_rets.items():
    ax.plot((1 + r).cumprod(), label=k, linewidth=1.8 if k == b_name_cnn else 1.2)
ax.set_title("추가2: 자산 히트맵 CNN으로 SPY 넣고 빼기 (표본 밖, 비용 차감)")
ax.legend()
ax.grid(True, alpha=0.3)
b_fig1.tight_layout()

b_fig2, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].imshow(b_X[-1, :, :, 0].T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
axes[0].set_yticks(range(len(B_ASSETS)))
axes[0].set_yticklabels(B_ASSETS)
axes[0].set_title(f"입력 그림 예시 ({b_dates[-1].date()}, 가로=20일 세로=자산)")
if b_lr_coef_last is not None:
    lim = np.abs(b_lr_coef_last).max()
    axes[1].imshow(b_lr_coef_last.T, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[1].set_yticks(range(len(B_ASSETS)))
    axes[1].set_yticklabels(B_ASSETS)
    axes[1].set_title("마지막 로지스틱 가중치 (빨강=상승 쪽)")
b_fig2.tight_layout()
plt.show()

print(md_table(b_cls_tab, pct_cols=("정확도", "실제 상승 비율", "매수 신호 비율")))
print(md_table(b_stats, pct_cols=("CAGR", "변동성", "MDD", "누적수익", "플러스비율", "시장 보유 비율")))

# ----- 7) 결과 자동 저장 -----
b_auto = [
    f"- 표본 밖 AUC: {b_name_cnn.split()[0]} {b_auc_cnn:.3f}, 로지스틱 {b_auc_lr:.3f} (0.5 = 동전 던지기)",
    f"- 샤프: {b_name_cnn} {b_stats.loc[b_name_cnn, '샤프']:.2f} / 로지스틱 {b_stats.loc['로지스틱 신호', '샤프']:.2f} / 200일선 {b_stats.loc['200일선 규칙', '샤프']:.2f} / SPY {b_stats.loc['SPY 보유', '샤프']:.2f}",
    f"- MDD: {b_name_cnn} {b_stats.loc[b_name_cnn, 'MDD'] * 100:.1f}% vs SPY {b_stats.loc['SPY 보유', 'MDD'] * 100:.1f}%, SPY를 이긴 해 {b_years_beat}/{len(b_year)}",
    f"- 가렸을 때 가장 나빠진 부분: {b_occ_tab.index[0]} (AUC -{b_occ_tab.iloc[0]['AUC 하락폭']:.3f})",
]
save_result(
    B_TAG, "추가 전략 2: 10개 자산 수익률 히트맵 CNN으로 SPY 시장 타이밍",
    [
        ("한눈에", "\n".join(b_auto)),
        ("설정", "\n".join([
            f"- 입력: 10개 ETF({', '.join(B_ASSETS)}) 최근 {B_LOOKBACK}일 표준화 수익률 = {B_LOOKBACK}x{len(B_ASSETS)} 그림 한 장",
            f"- 정답: SPY 다음 {B_HORIZON}거래일 수익률 > 0, 학습과 예측 사이 {B_HORIZON}일 비움",
            f"- 모델: Conv 16 - Pool(2,1) - Conv 32 - Pool(2,2) - Dense 32 - Dropout 0.3 - sigmoid, 최대 {B_EPOCHS}회 학습·조기 종료",
            f"- 워크포워드: {B_TEST_START}년부터 매년 재학습, 표본 밖 {len(b_test_days)}일 ({b_test_days.min().date()} ~ {b_test_days.max().date()})",
            f"- 매매: 예측 확률 > 학습 구간 평균 확률이면 다음 날 SPY 보유, 아니면 현금. 한쪽 비용 {B_COST * 100:.2f}%",
        ])),
        ("분류 성능 (표본 밖)", md_table(b_cls_tab, pct_cols=("정확도", "실제 상승 비율", "매수 신호 비율"))),
        ("연도별 AUC", md_table(b_auc_year)),
        ("매매 성과 (일간, 비용 차감)", md_table(b_stats, pct_cols=("CAGR", "변동성", "MDD", "누적수익", "플러스비율", "시장 보유 비율"))),
        ("연도별 수익률", md_table(b_year, pct_cols="all")),
        ("신호가 수익을 갈라 주는지 (다음 날 SPY 평균 수익)", md_table(b_split, pct_cols="all")),
        ("CNN이 본 곳: 가리기 진단 (그 부분을 0으로 가렸을 때 AUC가 많이 떨어질수록 중요)", md_table(b_occ_tab)),
        ("재학습 기록", md_table(pd.DataFrame(b_train_log).set_index("연도"))),
        ("주의", "\n".join([
            "- 정답 기간(5일)이 겹쳐 표본끼리 독립이 아니다. AUC 0.5 근처면 의미 없음으로 본다.",
            "- 현금 구간 이자 수익은 0으로 둠 (보수적).",
            "- sample_etf 는 DB 마지막 날짜까지만.",
        ])),
    ],
    {"일별수익": pd.DataFrame(b_rets), "연도별수익": b_year,
     "예측확률": pd.DataFrame({"CNN": b_prob_cnn, "로지스틱": b_prob_lr, "SPY_5일수익": b_fwd}).dropna(subset=["CNN"]),
     "가리기진단": b_occ_tab},
    {"누적수익": b_fig1, "입력그림_로지스틱가중치": b_fig2},
    {"분류": b_cls_tab.to_dict(orient="index"), "성과": b_stats.to_dict(orient="index"),
     "가리기진단": b_occ_tab["AUC 하락폭"].to_dict(), "SPY_이긴해": b_years_beat},
)


#%% [추가 전략 2-2] 자산 히트맵 CNN 개선: 방향 대신 '5일 안 -3% 급락'을 맞히고 200일선과 묶기 (추가전략_다음할일.md 의 추가2)
# 무엇: 추가 전략 2는 방향 AUC 0.51로 예측력이 없었다. 같은 그림으로 '급락 위험'을 맞혀 본다.
#  - 모델: CNN(10자산 전부) / CNN(가리기 진단으로 고른 5자산만, 매년 학습 구간에서만 고름) / 로지스틱 / 모델 없는 20일 변동성
#  - 매매: 200일선 위에서만 보유 + 위험 높으면 빼기, 또는 200일선 아래라도 위험 낮으면 빨리 재진입
# 실행: 맨 위 [1] 환경 설정 셀 먼저, 그다음 이 셀. CNN을 해마다 두 번 학습해 15~25분
# 결과: 06_퀀트공부/딥러닝/결과/추가2-2_CNN_급락위험_200일선결합/ 에 자동 저장
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ----- 설정 (바꿔 보면서 다시 돌려 보기) -----
B2_TAG = "추가2-2_CNN_급락위험_200일선결합"
B2_ASSETS = ["SPY", "IEV", "EWJ", "EEM", "TLT", "IEF", "IYR", "RWX", "GLD", "DBC"]
B2_LOOKBACK = 20
B2_HORIZON = 5             # 다음 5거래일 안에
B2_DROP = -0.03            # 종가 기준 -3% 이상 빠진 날이 한 번이라도 있으면 '급락 위험' = 1
B2_COST = 0.0005
B2_TEST_START = 2012
B2_EPOCHS = 15
B2_RISK_Q = 0.8            # 학습 구간 위험확률 80% 분위를 넘으면 '위험 높음'
B2_KEEP = 5                # 가리기 진단으로 중요한 자산 몇 개만 남길지

# ----- 1) 데이터 (추가 전략 2와 같은 그림) + 새 정답: 5일 안 -3% 급락 -----
b2_raw = load_sql("SELECT * FROM sample_etf")
b2_raw["Date"] = pd.to_datetime(b2_raw["Date"])
b2_px = b2_raw.set_index("Date").sort_index()[B2_ASSETS].astype(float)
b2_ret = b2_px.pct_change(fill_method=None)
b2_z = (b2_ret / b2_ret.rolling(252, min_periods=126).std()).clip(-4, 4).dropna()
b2_spy = b2_px["SPY"]
b2_fwd_min = pd.concat([b2_spy.shift(-h) for h in range(1, B2_HORIZON + 1)], axis=1).min(axis=1) / b2_spy - 1
b2_fwd_min[b2_spy.shift(-B2_HORIZON).isna()] = np.nan
b2_vals = b2_z.values
b2_X = np.stack([b2_vals[i - B2_LOOKBACK + 1: i + 1] for i in range(B2_LOOKBACK - 1, len(b2_vals))])[..., None]
b2_dates = b2_z.index[B2_LOOKBACK - 1:]
b2_fm = b2_fwd_min.reindex(b2_dates)
b2_y = (b2_fm <= B2_DROP).astype(int).values
b2_has_y = b2_fm.notna().values
# 모델 없는 기준 점수: SPY 최근 20일 실현 변동성 (높을수록 급락 위험이 높다고 보는 단순 규칙)
b2_vol20 = b2_ret["SPY"].rolling(20).std().reindex(b2_dates)
b2_ma200 = b2_spy.rolling(200).mean()
b2_above = (b2_spy > b2_ma200).reindex(b2_dates)
print(f"[추가2-2] 그림 {b2_X.shape}, 5일 안 {B2_DROP * 100:.0f}% 급락 비율 {b2_y[b2_has_y].mean():.3f}")


def b2_build_cnn(n_assets, seed):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
    tf.random.set_seed(seed)
    inp = Input(shape=(B2_LOOKBACK, n_assets, 1))
    x = Conv2D(16, (3, 3), padding="same", activation="relu")(inp)
    x = MaxPooling2D((2, 1))(x)
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D((2, 2 if n_assets >= 4 else 1))(x)
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation="sigmoid")(x)
    m = Model(inp, out, name="Asset_Heatmap_Risk_CNN")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return m


def b2_fit(x_tr, y_tr, seed):
    """급락이 드물어(약 1할) 클래스 가중치를 줌. TF 없으면 sklearn MLP"""
    w1 = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    if TF_AVAILABLE:
        from tensorflow.keras.callbacks import EarlyStopping
        m = b2_build_cnn(x_tr.shape[2], seed)
        m.fit(x_tr, y_tr, epochs=B2_EPOCHS, batch_size=64, validation_split=0.15, verbose=0,
              class_weight={0: 1.0, 1: float(w1)}, callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
        return lambda arr: m.predict(arr, verbose=0).ravel()
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(64, 16), alpha=1e-3, max_iter=300, early_stopping=True,
                        validation_fraction=0.15, random_state=seed)
    mlp.fit(x_tr.reshape(len(x_tr), -1), y_tr)
    return lambda arr: mlp.predict_proba(arr.reshape(len(arr), -1))[:, 1]


# ----- 2) 매년 재학습: CNN(10자산 전부) / 가리기로 고른 자산만 CNN / 로지스틱 -----
b2_prob = {k: pd.Series(np.nan, index=b2_dates) for k in ["CNN 전체", "CNN 고른 자산", "로지스틱"]}
b2_thr = {k: {} for k in b2_prob}
b2_keep_log = []
b2_train_log = []
for yr in range(B2_TEST_START, b2_dates.max().year + 1):
    te_pos = np.where(b2_dates.year == yr)[0]
    if len(te_pos) == 0:
        continue
    tr_pos = np.arange(0, max(te_pos[0] - B2_HORIZON, 0))
    tr_pos = tr_pos[b2_has_y[tr_pos]]
    if len(tr_pos) < 500 or b2_y[tr_pos].sum() < 20:
        continue
    x_tr, y_tr, x_te = b2_X[tr_pos], b2_y[tr_pos], b2_X[te_pos]

    lr = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced")
    lr.fit(x_tr.reshape(len(x_tr), -1), y_tr)
    b2_prob["로지스틱"].iloc[te_pos] = lr.predict_proba(x_te.reshape(len(x_te), -1))[:, 1]
    b2_thr["로지스틱"][yr] = np.quantile(lr.predict_proba(x_tr.reshape(len(x_tr), -1))[:, 1], B2_RISK_Q)

    pred_full = b2_fit(x_tr, y_tr, SEED)
    b2_prob["CNN 전체"].iloc[te_pos] = pred_full(x_te)
    b2_thr["CNN 전체"][yr] = np.quantile(pred_full(x_tr), B2_RISK_Q)

    # 가리기 진단을 '학습 구간의 마지막 15%'에서만 해서 중요한 자산 고르기 (예측 구간은 안 봄)
    va = np.arange(int(len(x_tr) * 0.85), len(x_tr))
    if len(np.unique(y_tr[va])) > 1:
        base_auc = roc_auc_score(y_tr[va], pred_full(x_tr[va]))
        drops = {}
        for j, n in enumerate(B2_ASSETS):
            xo = x_tr[va].copy()
            xo[:, :, j, 0] = 0.0
            drops[n] = base_auc - roc_auc_score(y_tr[va], pred_full(xo))
        keep = sorted(drops, key=lambda n: -drops[n])[:B2_KEEP]
    else:
        keep = B2_ASSETS[:B2_KEEP]
    kj = [B2_ASSETS.index(n) for n in keep]
    pred_keep = b2_fit(x_tr[:, :, kj, :], y_tr, SEED)
    b2_prob["CNN 고른 자산"].iloc[te_pos] = pred_keep(x_te[:, :, kj, :])
    b2_thr["CNN 고른 자산"][yr] = np.quantile(pred_keep(x_tr[:, :, kj, :]), B2_RISK_Q)
    b2_keep_log.append({"연도": yr, "고른 자산": ", ".join(keep)})
    b2_train_log.append({"연도": yr, "학습표본": len(tr_pos), "학습 급락 비율": float(y_tr.mean()), "예측일수": len(te_pos)})
    print(f"  {yr}년 모델: 학습 {len(tr_pos)}일, 고른 자산 {keep}")

# ----- 3) 위험 예측력: AUC, 위험 높음 신호일 때 실제 급락 비율 -----
b2_mask = b2_prob["CNN 전체"].notna().values & b2_has_y
b2_scores = dict(b2_prob)
b2_scores["SPY 20일 변동성(모델 없음)"] = b2_vol20
b2_cls = {}
for k, sc in b2_scores.items():
    v = sc.values[b2_mask]
    ok = ~np.isnan(v)
    hi = np.zeros(len(b2_dates), dtype=bool)
    if k in b2_thr:
        th = np.array([b2_thr[k].get(d.year, np.nan) for d in b2_dates])
        hi = sc.values > th
    else:                                             # 변동성: 그 전날까지 확장 창 80% 분위
        hi = (sc > sc.expanding(250).quantile(B2_RISK_Q).shift(1)).values
    hm = hi & b2_mask
    b2_cls[k] = {"AUC": roc_auc_score(b2_y[b2_mask][ok], v[ok]),
                 "위험 높음 날 비율": hm.sum() / b2_mask.sum(),
                 "위험 높음 날 실제 급락 비율": b2_y[hm].mean() if hm.sum() else np.nan,
                 "나머지 날 실제 급락 비율": b2_y[b2_mask & ~hi].mean()}
    b2_scores[k] = pd.Series(hi, index=b2_dates)       # 아래 매매용으로 신호(참/거짓)로 바꿔 둠
b2_cls = pd.DataFrame(b2_cls).T
b2_cls.index.name = "점수"
b2_auc_year = {}
for yr in sorted(set(b2_dates[b2_mask].year)):
    mk = b2_mask & (b2_dates.year == yr)
    if len(np.unique(b2_y[mk])) > 1:
        b2_auc_year[yr] = {k: roc_auc_score(b2_y[mk], b2_prob[k].values[mk]) for k in b2_prob}
b2_auc_year = pd.DataFrame(b2_auc_year).T
b2_auc_year.index.name = "연도"

# ----- 4) 매매: 200일선과 묶기 (신호는 그날 종가 -> 다음 날 수익) -----
b2_test_days = b2_dates[b2_prob["CNN 전체"].notna().values]
b2_spy_ret = b2_ret["SPY"]


def b2_bt(hold):
    pos = pd.Series(hold, index=b2_dates).reindex(b2_test_days).astype(float)
    r = pos.shift(1) * b2_spy_ret.reindex(b2_test_days)
    cost = pos.diff().abs().fillna(pos.iloc[0]) * B2_COST
    return (r - cost.shift(1).fillna(0)).iloc[1:], float(pos.mean())


b2_ab = b2_above.fillna(False).values
b2_rules = {
    "SPY 보유": np.ones(len(b2_dates), dtype=bool),
    "200일선 위에서만 보유": b2_ab,
    "CNN 위험 높으면 빼기": ~b2_scores["CNN 전체"].values,
    "200일선 위 + CNN 위험 낮을 때만": b2_ab & ~b2_scores["CNN 전체"].values,
    "200일선 위 + 고른자산 CNN 위험 낮을 때만": b2_ab & ~b2_scores["CNN 고른 자산"].values,
    "200일선 위 + 로지스틱 위험 낮을 때만": b2_ab & ~b2_scores["로지스틱"].values,
    "200일선 위 + 변동성 낮을 때만(모델 없음)": b2_ab & ~b2_scores["SPY 20일 변동성(모델 없음)"].values,
    "200일선 위 또는 CNN 위험 낮음(빨리 재진입)": b2_ab | ~b2_scores["CNN 전체"].values,
}
b2_rets, b2_inmkt = {}, {}
for k, h in b2_rules.items():
    b2_rets[k], b2_inmkt[k] = b2_bt(h)
b2_stats = stats_frame(b2_rets, 252)
b2_stats["시장 보유 비율"] = [b2_inmkt[k] for k in b2_rules]
b2_year = yearly_table(b2_rets)
b2_keep_tab = pd.DataFrame(b2_keep_log).set_index("연도") if b2_keep_log else pd.DataFrame()
b2_keep_freq = pd.Series(", ".join(b2_keep_tab["고른 자산"]).split(", ")).value_counts() if len(b2_keep_tab) else pd.Series(dtype=int)
b2_keep_freq = pd.DataFrame({"고른 해 수": b2_keep_freq})
b2_keep_freq.index.name = "자산"

# ----- 5) 그림 -----
b2_fig1, ax = plt.subplots(figsize=(11, 4.5))
for k, r in b2_rets.items():
    ax.plot((1 + r).cumprod(), label=k, linewidth=1.8 if "CNN 위험 낮을 때만" in k else 1.0)
ax.set_title("추가2-2: 급락 위험 CNN + 200일선 (비용 차감)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
b2_fig1.tight_layout()
plt.show()
print(md_table(b2_cls, pct_cols=("위험 높음 날 비율", "위험 높음 날 실제 급락 비율", "나머지 날 실제 급락 비율")))

# ----- 6) 결과 자동 저장 -----
b2_auto = [
    "- 급락 위험 AUC: " + " / ".join(f"{k} {b2_cls.loc[k, 'AUC']:.3f}" for k in b2_cls.index),
    f"- CNN '위험 높음' 날 실제 급락 비율 {b2_cls.loc['CNN 전체', '위험 높음 날 실제 급락 비율'] * 100:.1f}% vs 나머지 날 {b2_cls.loc['CNN 전체', '나머지 날 실제 급락 비율'] * 100:.1f}%",
    "- 샤프: " + " / ".join(f"{k} {b2_stats.loc[k, '샤프']:.2f}" for k in b2_rules),
    f"- 200일선만 대비 '200일선 + CNN' 샤프 차이 {b2_stats.loc['200일선 위 + CNN 위험 낮을 때만', '샤프'] - b2_stats.loc['200일선 위에서만 보유', '샤프']:+.2f}, "
    f"'200일선 + 변동성(모델 없음)' 차이 {b2_stats.loc['200일선 위 + 변동성 낮을 때만(모델 없음)', '샤프'] - b2_stats.loc['200일선 위에서만 보유', '샤프']:+.2f} (CNN이 단순 변동성보다 나은지)",
]
save_result(
    B2_TAG, "추가 전략 2-2: 방향 대신 급락 위험을 맞히는 CNN + 200일선",
    [
        ("한눈에", "\n".join(b2_auto)),
        ("설정", "\n".join([
            f"- 입력: 추가 전략 2와 같은 {B2_LOOKBACK}일 x {len(B2_ASSETS)}자산 표준화 수익률 그림",
            f"- 정답: 다음 {B2_HORIZON}거래일 안에 SPY 종가가 오늘보다 {B2_DROP * 100:.0f}% 이상 낮은 날이 있으면 1 (급락 위험)",
            f"- '위험 높음' = 예측확률이 학습 구간 확률의 {B2_RISK_Q * 100:.0f}% 분위 초과. 급락이 드물어 학습 때 클래스 가중치 사용",
            f"- 고른 자산 CNN: 매년 학습 구간 마지막 15%에서 자산을 하나씩 가려 AUC가 많이 떨어지는 {B2_KEEP}개만 남겨 다시 학습",
            f"- 비교: 모델 없는 SPY 20일 변동성 (높으면 위험), 200일선, SPY 보유. 한쪽 비용 {B2_COST * 100:.2f}%",
        ])),
        ("급락 위험 예측력 (표본 밖)", md_table(b2_cls, pct_cols=("위험 높음 날 비율", "위험 높음 날 실제 급락 비율", "나머지 날 실제 급락 비율"))),
        ("연도별 AUC", md_table(b2_auc_year)),
        ("매매 성과 (일간, 비용 차감)", md_table(b2_stats, pct_cols=("CAGR", "변동성", "MDD", "누적수익", "플러스비율", "시장 보유 비율"))),
        ("연도별 수익률", md_table(b2_year, pct_cols="all")),
        ("해마다 고른 자산", md_table(b2_keep_tab) if len(b2_keep_tab) else "-"),
        ("자산별 고른 해 수", md_table(b2_keep_freq) if len(b2_keep_freq) else "-"),
        ("재학습 기록", md_table(pd.DataFrame(b2_train_log).set_index("연도"))),
        ("주의", "\n".join([
            "- 정답 기간(5일)이 겹쳐 표본끼리 독립이 아니다.",
            "- 현금 구간 이자 0 (보수적). sample_etf 마지막 날짜까지만.",
            "- 급락 기준(-3%)·위험 분위(80%)를 바꾸면 결과가 달라진다 -> 설정 바꿔 재확인.",
        ])),
    ],
    {"일별수익": pd.DataFrame(b2_rets), "연도별수익": b2_year,
     "위험확률": pd.DataFrame({**{k: v for k, v in b2_prob.items()}, "SPY_5일최저": b2_fm, "급락": pd.Series(b2_y, index=b2_dates)}).dropna(subset=["CNN 전체"]),
     "고른자산": b2_keep_tab},
    {"누적수익": b2_fig1},
    {"위험예측": b2_cls.to_dict(orient="index"), "성과": b2_stats.to_dict(orient="index")},
)
