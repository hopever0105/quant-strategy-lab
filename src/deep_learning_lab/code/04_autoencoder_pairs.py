# -*- coding: utf-8 -*-
"""
오토인코더 페어 트레이딩 (추가 전략 4, 4-2, 4-3)
- 딥러닝 전략 실험 묶음(deep_learning_lab)의 코드. 공부 파일 끝에 붙여 돌린 '추가 전략' 셀만 모았다.
- 실행: Spyder(conda ta_env), #%% 셀 단위. 맨 위 [1] 환경 설정 셀 → 원하는 [추가 전략] 셀.
- 결과: ../results/<전략 이름>/ 에 결과요약.md·지표.json·csv·png 자동 저장 (다시 돌리면 덮어씀).
- DB 접속 정보는 .env 파일(환경변수 QUANT_ENV_PATH 로 경로 지정 가능)에서 읽고 코드에 쓰지 않는다.
- 포트폴리오용 연구 코드. 실제 주문·매매 기능 없음.
"""
#%% [1] 환경 설정, 공통 라이브러리 임포트 및 로컬 MySQL DB 연결
import os
import random
import urllib.parse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, BatchNormalization
    from tensorflow.keras import backend as K
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


#%% [추가 전략 4] 오토인코더 페어 트레이딩: 잠재공간에서 닮은 종목 짝 찾아 벌어지면 되돌림 매매 
# 무엇: 위 오토인코더 예제는 지수 하나의 수익률 창을 압축했다. 여기서는 '종목 하나 = 1년치 수익률 252개'를 6차원으로 압축해
#  잠재공간에서 가까운 두 종목을 '닮은 짝'으로 본다 -> 두 종목 가격 차이(스프레드)가 평소보다 벌어지면 되돌아올 쪽에 베팅
#  - 유니버스: 직전 해 거래대금 상위 100종목, 1월·7월마다 짝 10쌍 새로 고르고 6개월 매매
# 비교: PCA로 고른 짝, 단순 상관계수로 고른 짝 (오토인코더가 상관과 다른 짝을 찾는지, 그게 더 나은지)
# 실행: 맨 위 [1] 환경 설정 셀 먼저, 그다음 이 셀. DB 집계 1분 + 학습 몇 분
# 결과: 06_퀀트공부/딥러닝/결과/추가4_오토인코더_페어트레이딩/ 에 자동 저장
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

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

# ----- 설정 -----
D_TAG = "추가4_오토인코더_페어트레이딩"
D_START = "2012-01-01"
D_UNIVERSE = 100           # 매 기간 직전 해 평균 거래대금 상위 종목 수
D_FORM = 252               # 짝 찾는 기간 (형성 기간, 거래일)
D_TRADE = 126              # 매매 기간 (약 6개월, 1·7월 시작)
D_NPAIR = 10               # 기간마다 매매하는 짝 수
D_LATENT = 6               # 오토인코더 잠재 차원
D_ENTRY, D_STOP = 2.0, 4.0 # 스프레드 z가 ±2 넘으면 진입, ±4 넘으면 손절, 0 지나면 청산
D_COST_LEG = 0.002         # 한 종목 한쪽 거래비용 0.2% (매도세·수수료·대차 가정)

# ----- 1) 유니버스: 해마다 평균 거래대금 상위 100 (직전 해 값만 사용) -----
print("[추가4] 연도별 거래대금 집계 중... (DB에서 1분 안팎)")
d_liq = load_sql(
    "SELECT 종목코드, YEAR(날짜) AS 연도, AVG(종가 * 거래량) AS 평균거래대금, COUNT(*) AS 일수 "
    f"FROM kor_price WHERE 날짜 >= '{pd.Timestamp(D_START).year - 1}-01-01' GROUP BY 종목코드, YEAR(날짜)"
)
d_liq = d_liq[d_liq["일수"] >= 200]
d_top = {int(y): g.nlargest(D_UNIVERSE, "평균거래대금")["종목코드"].tolist() for y, g in d_liq.groupby("연도")}
d_codes = sorted(set(c for v in d_top.values() for c in v))
print(f"[추가4] 후보 종목 {len(d_codes)}개 가격 불러오는 중...")
d_raw = load_sql(
    "SELECT 날짜, 종목코드, 종가 FROM kor_price "
    f"WHERE 날짜 >= '{pd.Timestamp(D_START) - pd.DateOffset(years=1):%Y-%m-%d}' "
    f"AND 종목코드 IN ({', '.join(repr(c) for c in d_codes)})"
)
d_raw["날짜"] = pd.to_datetime(d_raw["날짜"])
d_px = d_raw.pivot(index="날짜", columns="종목코드", values="종가").sort_index()
d_ret = d_px.pct_change(fill_method=None)
d_ret = d_ret.where(d_ret.abs() < 0.35)       # 30% 가격제한 넘는 값 = 분할·병합 흔적 -> 결측 처리
d_names = load_sql("SELECT 종목코드, 종목명 FROM kor_ticker WHERE 기준일 = (SELECT MAX(기준일) FROM kor_ticker)")
d_names = dict(zip(d_names["종목코드"], d_names["종목명"])) if len(d_names) else {}


# ----- 2) 세 가지 '닮은 종목' 찾기: 오토인코더 잠재공간 / PCA / 단순 상관 -----
def d_autoencoder_latent(x, seed):
    """x: 종목 x 252일 표준화 수익률 -> 종목마다 D_LATENT 차원 좌표.
    위 오토인코더 예제와 같은 대칭 구조(252-32-잠재-32-252). 위 예제 실행에서 relu 잠재 하나가 전부 0(죽은 ReLU)이었던 걸 피하려고 tanh 사용"""
    if TF_AVAILABLE:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense
        tf.random.set_seed(seed)
        inp = Input(shape=(x.shape[1],))
        h = Dense(32, activation="tanh")(inp)
        z = Dense(D_LATENT, activation="tanh", name="latent")(h)
        h2 = Dense(32, activation="tanh")(z)
        out = Dense(x.shape[1], activation="linear")(h2)
        ae = Model(inp, out)
        ae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        ae.fit(x, x, epochs=300, batch_size=16, verbose=0)
        enc = Model(inp, z)
        lat = enc.predict(x, verbose=0)
        rec_mse = float(np.mean((ae.predict(x, verbose=0) - x) ** 2))
        return lat, rec_mse
    m = MLPRegressor(hidden_layer_sizes=(32, D_LATENT, 32), activation="tanh", alpha=1e-4,
                     max_iter=2000, random_state=seed)
    m.fit(x, x)
    h = np.tanh(x @ m.coefs_[0] + m.intercepts_[0])
    lat = np.tanh(h @ m.coefs_[1] + m.intercepts_[1])        # 인코더 부분만 직접 계산 = 잠재 좌표
    rec_mse = float(np.mean((m.predict(x) - x) ** 2))
    return lat, rec_mse


def d_pick_pairs(dist, codes, n):
    """거리 가까운 순으로 짝 고르기 (한 종목은 한 번만)"""
    iu = np.triu_indices(len(codes), 1)
    order = np.argsort(dist[iu])
    used, pairs = set(), []
    for o in order:
        a, b = iu[0][o], iu[1][o]
        if a in used or b in used:
            continue
        pairs.append((codes[a], codes[b], float(dist[a, b])))
        used.update([a, b])
        if len(pairs) == n:
            break
    return pairs


def d_euclid(z):
    z = (z - z.mean(0)) / (z.std(0) + 1e-9)
    sq = (z ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * z @ z.T, 0))


# ----- 3) 짝 하나 매매: 형성 기간에서 헤지비율·평균·표준편차 -> 매매 기간에 z로 진입·청산 -----
def d_trade_pair(a, b, form_idx, trade_idx):
    la_f, lb_f = np.log(d_adj[a].loc[form_idx]), np.log(d_adj[b].loc[form_idx])
    ok = la_f.notna() & lb_f.notna()
    beta, alpha = np.polyfit(lb_f[ok].values, la_f[ok].values, 1)
    resid = la_f - (alpha + beta * lb_f)
    mu, sd = resid.mean(), resid.std()
    la, lb = np.log(d_adj[a].loc[trade_idx]), np.log(d_adj[b].loc[trade_idx])
    z = ((la - (alpha + beta * lb)) - mu) / sd
    ra = d_ret_clean[a].loc[trade_idx].fillna(0.0).values
    rb = d_ret_clean[b].loc[trade_idx].fillna(0.0).values
    zv = z.values
    pos = np.zeros(len(zv))
    cur, blocked, entry_i = 0, False, None
    trades = []
    for i in range(len(zv) - 1):        # 마지막 날은 무조건 청산
        if cur == 0 and not blocked:
            if zv[i] > D_ENTRY and zv[i] < D_STOP:
                cur, entry_i = -1, i        # 스프레드 비쌈: a 공매도·b 매수
            elif zv[i] < -D_ENTRY and zv[i] > -D_STOP:
                cur, entry_i = 1, i         # 스프레드 쌈: a 매수·b 공매도
        elif cur != 0:
            reason = None
            if (cur == -1 and zv[i] <= 0) or (cur == 1 and zv[i] >= 0):
                reason = "수렴"
            elif abs(zv[i]) >= D_STOP:
                reason, blocked = "손절", True
            if reason:
                trades.append({"a": a, "b": b, "진입": trade_idx[entry_i], "청산": trade_idx[i],
                               "보유일": i - entry_i, "사유": reason, "_i0": entry_i, "_i1": i})
                cur = 0
        pos[i] = cur
    if cur != 0:
        trades.append({"a": a, "b": b, "진입": trade_idx[entry_i], "청산": trade_idx[-1],
                       "보유일": len(zv) - 1 - entry_i, "사유": "기간끝", "_i0": entry_i, "_i1": len(zv) - 1})
    # 오늘 종가 신호 -> 내일 수익. 자금 1 = 매수 0.5 + 공매도 0.5
    held = np.concatenate([[0.0], pos[:-1]])
    gross = held * 0.5 * (ra - rb)
    cost = np.abs(np.diff(np.concatenate([[0.0], pos]))) * D_COST_LEG   # 두 종목 x 0.5 x 비용
    cost = np.concatenate([[0.0], cost[:-1]])
    net = pd.Series(gross - cost, index=trade_idx)
    for t in trades:
        seg = net.iloc[t.pop("_i0"): t.pop("_i1") + 2]   # 진입 날 ~ 청산 다음 날(청산 비용 포함), 첫 날은 빼고 곱함
        t["수익(비용후)"] = float((1 + seg.iloc[1:]).prod() - 1) if len(seg) > 1 else 0.0
    return net, trades


# 결측(거래정지 등)은 수익 0으로 이어 붙인 '수정 가격' (분할 흔적 제거된 수익률로 다시 만든 가격)
d_ret_clean = d_ret.copy()
d_adj = (1 + d_ret_clean.fillna(0.0)).cumprod()
d_adj = d_adj.where(d_px.ffill().notna())

# ----- 4) 6개월마다 반복 -----
d_all_days = d_px.index
d_starts = [d_all_days[d_all_days >= pd.Timestamp(f"{y}-{m:02d}-01")][0]
            for y in range(pd.Timestamp(D_START).year + 1, d_all_days.max().year + 1) for m in (1, 7)
            if (d_all_days >= pd.Timestamp(f"{y}-{m:02d}-01")).any()]
D_METHODS = ["오토인코더", "PCA", "상관계수"]
d_rets = {m: [] for m in D_METHODS}
d_trades = {m: [] for m in D_METHODS}
d_pairs_log = []
d_period_log = []
for s in d_starts:
    s_pos = d_all_days.get_loc(s)
    if s_pos < D_FORM:
        continue
    form_idx = d_all_days[s_pos - D_FORM: s_pos]
    trade_idx = d_all_days[s_pos: s_pos + D_TRADE]
    if len(trade_idx) < 20:
        continue
    univ = [c for c in d_top.get(s.year - 1, []) if c in d_ret.columns]
    fr = d_ret.loc[form_idx, univ]
    univ = [c for c in univ if fr[c].notna().mean() > 0.98]          # 형성 기간 데이터가 거의 다 있는 종목만
    fr = d_ret.loc[form_idx, univ].fillna(0.0)
    univ = [c for c in univ if fr[c].std() > 0]                       # 1년 내내 거래정지 등 변동 없는 종목 제외
    fr = fr[univ]
    x =((fr - fr.mean()) / fr.std()).T.values                        # 행 = 종목, 열 = 날짜
    corr = np.corrcoef(x)
    lat, rec = d_autoencoder_latent(x, SEED)
    pcs = PCA(n_components=D_LATENT, random_state=SEED).fit_transform(x)
    dists = {"오토인코더": d_euclid(lat), "PCA": d_euclid(pcs), "상관계수": 1 - corr}
    chosen = {}
    for meth in D_METHODS:
        pairs = d_pick_pairs(dists[meth], univ, D_NPAIR)
        chosen[meth] = set(frozenset(p[:2]) for p in pairs)
        pr = []
        for a, b, dist in pairs:
            net, tr = d_trade_pair(a, b, form_idx, trade_idx)
            pr.append(net)
            for t in tr:
                t.update({"방법": meth, "기간시작": s})
            d_trades[meth] += tr
            d_pairs_log.append({"기간시작": s.date(), "방법": meth, "종목A": f"{d_names.get(a, '')}({a})",
                                "종목B": f"{d_names.get(b, '')}({b})", "거리": dist,
                                "형성기간 상관": float(np.corrcoef(fr[a], fr[b])[0, 1]),
                                "매매기간 수익": float((1 + net).prod() - 1)})
        d_rets[meth].append(pd.concat(pr, axis=1).mean(axis=1))      # 짝 10개에 자금 똑같이
    d_period_log.append({"기간시작": s.date(), "종목수": len(univ), "AE 복원오차": rec,
                         "AE-상관 같은 짝": len(chosen["오토인코더"] & chosen["상관계수"]),
                         "AE-PCA 같은 짝": len(chosen["오토인코더"] & chosen["PCA"])})
    print(f"  {s.date()} 시작: 종목 {len(univ)}개, AE 복원오차 {rec:.3f}, AE·상관 같은 짝 {d_period_log[-1]['AE-상관 같은 짝']}개")

d_rets = {m: pd.concat(v).sort_index() for m, v in d_rets.items()}
d_rets = {m: v[~v.index.duplicated(keep="first")] for m, v in d_rets.items()}

# 비교 기준: 코스피 대용 = 유니버스 같은 비중 (매매 기간마다 직전 해 상위 100 단순 평균)
d_bm = []
for s in d_starts:
    s_pos = d_all_days.get_loc(s)
    trade_idx = d_all_days[s_pos: s_pos + D_TRADE]
    univ = [c for c in d_top.get(s.year - 1, []) if c in d_ret.columns]
    if s_pos >= D_FORM and len(trade_idx) >= 20:
        d_bm.append(d_ret.loc[trade_idx, univ].mean(axis=1))
d_bm = pd.concat(d_bm).sort_index()
d_bm = d_bm[~d_bm.index.duplicated(keep="first")]

d_all_rets = {f"{m} 짝": d_rets[m] for m in D_METHODS}
d_all_rets["유동성 상위100 같은 비중(참고)"] = d_bm
d_stats = stats_frame(d_all_rets, 252)
d_corr_bm = {k: float(v.corr(d_bm.reindex(v.index))) for k, v in d_all_rets.items()}
d_stats["시장과 상관"] = [d_corr_bm[k] for k in d_all_rets]
d_year = yearly_table(d_all_rets)

d_trade_tab = {}
for m in D_METHODS:
    t = pd.DataFrame(d_trades[m])
    if len(t) == 0:
        d_trade_tab[m] = {"매매 수": 0}
        continue
    d_trade_tab[m] = {"매매 수": len(t), "승률": (t["수익(비용후)"] > 0).mean(),
                      "1회 평균수익": t["수익(비용후)"].mean(), "평균 보유일": t["보유일"].mean(),
                      "수렴 비율": (t["사유"] == "수렴").mean(), "손절 비율": (t["사유"] == "손절").mean(),
                      "기간끝 청산 비율": (t["사유"] == "기간끝").mean()}
d_trade_tab = pd.DataFrame(d_trade_tab).T
d_trade_tab.index.name = "방법"
d_pairs_df = pd.DataFrame(d_pairs_log)
d_pair_quality = d_pairs_df.groupby("방법")[["형성기간 상관", "매매기간 수익"]].mean()
d_period_df = pd.DataFrame(d_period_log).set_index("기간시작")

# ----- 5) 그림 -----
d_fig1, ax = plt.subplots(figsize=(11, 4.5))
for k, r in d_all_rets.items():
    ax.plot((1 + r).cumprod(), label=k, linewidth=1.8 if k.startswith("오토인코더") else 1.2,
            linestyle="--" if "참고" in k else "-")
ax.set_title("추가4: 오토인코더 vs PCA vs 상관계수로 고른 짝 매매 (비용 차감)")
ax.legend()
ax.grid(True, alpha=0.3)
d_fig1.tight_layout()
plt.show()

print(md_table(d_stats, pct_cols=("CAGR", "변동성", "MDD", "누적수익", "플러스비율")))

# ----- 6) 결과 자동 저장 -----
d_auto = [
    "- 샤프: " + " / ".join(f"{k} {d_stats.loc[k, '샤프']:.2f}" for k in d_all_rets),
    "- 1회 평균수익(비용후): " + " / ".join(f"{m} {d_trade_tab.loc[m, '1회 평균수익'] * 100:.2f}%" for m in D_METHODS if "1회 평균수익" in d_trade_tab.columns),
    f"- 오토인코더가 고른 짝 중 상관계수 방법과 같은 짝: 기간 평균 {d_period_df['AE-상관 같은 짝'].mean():.1f}/{D_NPAIR}개 (적을수록 AE가 다른 걸 봄)",
    f"- 짝의 형성기간 평균 상관: " + " / ".join(f"{m} {d_pair_quality.loc[m, '형성기간 상관']:.2f}" for m in D_METHODS),
]
save_result(
    D_TAG, "추가 전략 4: 오토인코더 잠재공간에서 닮은 종목 짝 찾아 페어 트레이딩",
    [
        ("한눈에", "\n".join(d_auto)),
        ("설정", "\n".join([
            f"- 유니버스: 직전 해 평균 거래대금 상위 {D_UNIVERSE} (kor_price, {D_START}~), 형성 기간 데이터 98% 이상 있는 종목",
            f"- 형성 {D_FORM}일 -> 매매 {D_TRADE}일, 1월·7월마다 새로 짝 찾기 ({len(d_period_df)}개 기간)",
            f"- 짝 찾기: 종목마다 252일 표준화 수익률을 (1) 오토인코더 {D_LATENT}차원 잠재 좌표 (2) PCA {D_LATENT}차원 (3) 상관계수로 거리 계산 -> 가까운 {D_NPAIR}쌍",
            f"- 매매: 로그가격 회귀 잔차의 z가 ±{D_ENTRY} 넘으면 진입, 0 지나면 청산, ±{D_STOP} 손절(그 기간 재진입 금지), 기간 끝 청산",
            f"- 자금: 짝마다 매수 0.5 + 공매도 0.5, 짝 10개 같은 비중(놀고 있는 짝 자금 포함). 한 종목 한쪽 비용 {D_COST_LEG * 100:.2f}%",
        ])),
        ("성과 비교 (일간, 비용 차감)", md_table(d_stats, pct_cols=("CAGR", "변동성", "MDD", "누적수익", "플러스비율"))),
        ("연도별 수익률", md_table(d_year, pct_cols="all")),
        ("매매 1회 기준", md_table(d_trade_tab, pct_cols=("승률", "1회 평균수익", "수렴 비율", "손절 비율", "기간끝 청산 비율"), digits=1)),
        ("짝의 질 (방법별 평균)", md_table(d_pair_quality, pct_cols=("매매기간 수익",))),
        ("기간별 기록", md_table(d_period_df)),
        ("최근 기간에 고른 짝", md_table(d_pairs_df[d_pairs_df["기간시작"] == d_pairs_df["기간시작"].max()].set_index("방법"),
                                     pct_cols=("매매기간 수익",))),
        ("주의", "\n".join([
            "- kor_price 는 현재 상장 종목 위주라 중간에 상장폐지된 종목이 빠져 있다(생존 편향). 짝 매매는 롱숏이라 영향이 롱온리보다 작지만 0은 아님.",
            "- 개인은 국내 공매도가 사실상 어렵고 2023-11~2025-03 전면 금지 기간도 있었다. 연구용 검증.",
            "- 대차 수수료는 비용 가정에 대충 포함. 실제보다 낙관적일 수 있음.",
        ])),
    ],
    {"일별수익": pd.DataFrame(d_all_rets), "연도별수익": d_year, "짝목록": d_pairs_df.set_index("기간시작"),
     "매매내역": pd.concat([pd.DataFrame(v) for v in d_trades.values() if len(v)], ignore_index=True)
     if any(len(v) for v in d_trades.values()) else pd.DataFrame(),
     "기간별기록": d_period_df},
    {"누적수익": d_fig1},
    {"성과": d_stats.to_dict(orient="index"), "매매": d_trade_tab.to_dict(orient="index"),
     "짝의질": d_pair_quality.to_dict(orient="index"),
     "AE_상관_같은짝_평균": float(d_period_df["AE-상관 같은 짝"].mean())},
)


#%% [추가 전략 4-2] 페어 트레이딩 개선: 같은 업종·공적분 검정·매매 기준 격자·재무 특성 오토인코더 (추가전략_다음할일.md 의 추가4)
# 무엇: 추가 전략 4는 손절 40%, 수렴 25%로 고른 짝이 진짜로 같이 움직이지 않았다. 할 일 4개를 한 번에:
#  1) 같은 업종(kor_sector)끼리만  2) 공적분 검정(Engle-Granger p<0.05) 통과한 짝만  3) 업종+공적분
#  4) 형성 252·126일 x 진입 z 2·1.5 x 손절 4·없음 격자, 5) 재무(시가총액·PER·PBR·PSR)를 붙인 오토인코더 (재무 자료 있는 2023년 이후)
# 실행: 맨 위 [1] 환경 설정 셀 먼저, 그다음 이 셀. DB 집계 1분 + 기간마다 오토인코더 2~3번 학습해 15~30분
# 결과: 06_퀀트공부/딥러닝/결과/추가4-2_페어_업종_공적분_파라미터_재무/ 에 자동 저장
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

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
try:
    from statsmodels.tsa.stattools import coint as d2_coint
    D2_HAS_SM = True
except ImportError:
    D2_HAS_SM = False

# ----- 설정 (바꿔 보면서 다시 돌려 보기) -----
D2_TAG = "추가4-2_페어_업종_공적분_파라미터_재무"
D2_START = "2012-01-01"
D2_UNIVERSE = 100
D2_TRADE = 126
D2_NPAIR = 10
D2_LATENT = 6
D2_COST_LEG = 0.002
D2_FORMS = [252, 126]                    # 짝 찾는 기간 비교
D2_ENTRIES = [2.0, 1.5]                  # 진입 z 비교
D2_STOPS = [4.0, None]                   # 손절 z 비교 (None = 손절 없음)
D2_BASE = (252, 2.0, 4.0)                # 처음 실험(추가 전략 4)과 같은 설정
D2_COINT_P = 0.05                        # 공적분 검정 p값 기준
D2_COINT_TRY = 60                        # 거리 가까운 짝부터 이만큼만 검정 (시간 절약)
D2_FIN_COLS = ["시가총액", "PER", "PBR", "PSR"]   # 재무 특성 (kor_value_daily, 2022-11부터 있음)
D2_FIN_W = 0.5                           # 재무 묶음이 수익률 묶음 대비 차지하는 크기

# ----- 1) 데이터: 유니버스·가격 (추가 전략 4와 같음) + 업종 + 재무 -----
print("[추가4-2] 연도별 거래대금 집계 중... (DB에서 1분 안팎)")
d2_liq = load_sql(
    "SELECT 종목코드, YEAR(날짜) AS 연도, AVG(종가 * 거래량) AS 평균거래대금, COUNT(*) AS 일수 "
    f"FROM kor_price WHERE 날짜 >= '{pd.Timestamp(D2_START).year - 1}-01-01' GROUP BY 종목코드, YEAR(날짜)"
)
d2_liq = d2_liq[d2_liq["일수"] >= 200]
d2_top = {int(y): g.nlargest(D2_UNIVERSE, "평균거래대금")["종목코드"].tolist() for y, g in d2_liq.groupby("연도")}
d2_codes = sorted(set(c for v in d2_top.values() for c in v))
d2_raw = load_sql(
    "SELECT 날짜, 종목코드, 종가 FROM kor_price "
    f"WHERE 날짜 >= '{pd.Timestamp(D2_START) - pd.DateOffset(years=1):%Y-%m-%d}' "
    f"AND 종목코드 IN ({', '.join(repr(c) for c in d2_codes)})"
)
d2_raw["날짜"] = pd.to_datetime(d2_raw["날짜"])
d2_px = d2_raw.pivot(index="날짜", columns="종목코드", values="종가").sort_index()
d2_ret = d2_px.pct_change(fill_method=None)
d2_ret = d2_ret.where(d2_ret.abs() < 0.35)
d2_adj = (1 + d2_ret.fillna(0.0)).cumprod().where(d2_px.ffill().notna())
d2_logp = np.log(d2_adj)
d2_names = load_sql("SELECT 종목코드, 종목명 FROM kor_ticker WHERE 기준일 = (SELECT MAX(기준일) FROM kor_ticker)")
d2_names = dict(zip(d2_names["종목코드"], d2_names["종목명"])) if len(d2_names) else {}
d2_sec = load_sql("SELECT CMP_CD, SEC_NM_KOR FROM kor_sector WHERE 기준일 = (SELECT MAX(기준일) FROM kor_sector)")
d2_sector = dict(zip(d2_sec["CMP_CD"], d2_sec["SEC_NM_KOR"])) if len(d2_sec) else {}
d2_fin = load_sql(
    f"SELECT 종목코드, 날짜, {', '.join(D2_FIN_COLS)} FROM kor_value_daily "
    f"WHERE 종목코드 IN ({', '.join(repr(c) for c in d2_codes)})"
)
if len(d2_fin):
    d2_fin["날짜"] = pd.to_datetime(d2_fin["날짜"])
    d2_fin = d2_fin.sort_values("날짜")
print(f"[추가4-2] 후보 {len(d2_codes)}종목, 업종 정보 {len(d2_sector)}종목, 재무 {len(d2_fin)}행 "
      f"({d2_fin['날짜'].min().date() if len(d2_fin) else '-'}~), statsmodels: {D2_HAS_SM}")


# ----- 2) 잠재공간 / 거리 / 공적분 -----
def d2_latent(x, seed):
    if TF_AVAILABLE:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense
        tf.random.set_seed(seed)
        inp = Input(shape=(x.shape[1],))
        h = Dense(32, activation="tanh")(inp)
        z = Dense(D2_LATENT, activation="tanh")(h)
        h2 = Dense(32, activation="tanh")(z)
        out = Dense(x.shape[1], activation="linear")(h2)
        ae = Model(inp, out)
        ae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        ae.fit(x, x, epochs=300, batch_size=16, verbose=0)
        return Model(inp, z).predict(x, verbose=0)
    m = MLPRegressor(hidden_layer_sizes=(32, D2_LATENT, 32), activation="tanh", alpha=1e-4, max_iter=2000, random_state=seed)
    m.fit(x, x)
    h = np.tanh(x @ m.coefs_[0] + m.intercepts_[0])
    return np.tanh(h @ m.coefs_[1] + m.intercepts_[1])


def d2_euclid(z):
    z = (z - z.mean(0)) / (z.std(0) + 1e-9)
    sq = (z ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * z @ z.T, 0))


def d2_coint_p(a, b, form_idx):
    if not D2_HAS_SM:
        return np.nan
    la, lb = d2_logp[a].loc[form_idx], d2_logp[b].loc[form_idx]
    ok = la.notna() & lb.notna()
    if ok.sum() < 60:
        return 1.0
    try:
        return float(d2_coint(la[ok].values, lb[ok].values)[1])
    except Exception:
        return 1.0


def d2_pick(dist, codes, n, rule, form_idx):
    """거리 가까운 순으로 보면서 규칙(업종·공적분)을 통과한 짝만, 한 종목은 한 번만"""
    iu = np.triu_indices(len(codes), 1)
    order = np.argsort(dist[iu])
    used, pairs, tried = set(), [], 0
    for o in order:
        a, b = iu[0][o], iu[1][o]
        if a in used or b in used:
            continue
        ca, cb = codes[a], codes[b]
        if rule in ("업종", "업종+공적분"):
            sa, sb = d2_sector.get(ca), d2_sector.get(cb)
            if sa is None or sa != sb:
                continue
        p = np.nan
        if rule in ("공적분", "업종+공적분"):
            if tried >= D2_COINT_TRY:
                break
            tried += 1
            p = d2_coint_p(ca, cb, form_idx)
            if not (p < D2_COINT_P):
                continue
        pairs.append((ca, cb, float(dist[a, b]), p))
        used.update([a, b])
        if len(pairs) == n:
            break
    return pairs


# ----- 3) 짝 매매 (진입·손절을 바꿔 가며 돌리기 쉽게 배열로) -----
def d2_pair_arrays(a, b, form_idx, trade_idx):
    la_f, lb_f = d2_logp[a].loc[form_idx], d2_logp[b].loc[form_idx]
    ok = la_f.notna() & lb_f.notna()
    beta, alpha = np.polyfit(lb_f[ok].values, la_f[ok].values, 1)
    resid = la_f - (alpha + beta * lb_f)
    mu, sd = resid.mean(), resid.std()
    z = ((d2_logp[a].loc[trade_idx] - (alpha + beta * d2_logp[b].loc[trade_idx])) - mu) / sd
    ra = d2_ret[a].loc[trade_idx].fillna(0.0).values
    rb = d2_ret[b].loc[trade_idx].fillna(0.0).values
    return z.ffill().fillna(0.0).values, ra, rb


def d2_sim(zv, ra, rb, entry, stop):
    """반환: 일별 순수익 배열, 매매 목록(사유·보유일·수익)"""
    stop = np.inf if stop is None else stop
    pos = np.zeros(len(zv))
    cur, blocked, e_i = 0, False, None
    trades = []
    for i in range(len(zv) - 1):
        if cur == 0 and not blocked:
            if entry < zv[i] < stop:
                cur, e_i = -1, i
            elif -stop < zv[i] < -entry:
                cur, e_i = 1, i
        elif cur != 0:
            reason = None
            if (cur == -1 and zv[i] <= 0) or (cur == 1 and zv[i] >= 0):
                reason = "수렴"
            elif abs(zv[i]) >= stop:
                reason, blocked = "손절", True
            if reason:
                trades.append([e_i, i, reason])
                cur = 0
        pos[i] = cur
    if cur != 0:
        trades.append([e_i, len(zv) - 1, "기간끝"])
    held = np.concatenate([[0.0], pos[:-1]])
    gross = held * 0.5 * (ra - rb)
    cost = np.abs(np.diff(np.concatenate([[0.0], pos]))) * D2_COST_LEG
    cost = np.concatenate([[0.0], cost[:-1]])
    net = gross - cost
    out = []
    for i0, i1, reason in trades:
        seg = net[i0 + 1: i1 + 2]
        out.append({"사유": reason, "보유일": i1 - i0, "수익": float(np.prod(1 + seg) - 1) if len(seg) else 0.0})
    return net, out


# ----- 4) 6개월마다: 형성 기간 2가지 x 짝 고르는 방법 3가지 x 규칙 4가지 x (진입, 손절) 4가지 -----
d2_days = d2_px.index
d2_starts = [d2_days[d2_days >= pd.Timestamp(f"{y}-{m:02d}-01")][0]
             for y in range(pd.Timestamp(D2_START).year + 1, d2_days.max().year + 1) for m in (1, 7)
             if (d2_days >= pd.Timestamp(f"{y}-{m:02d}-01")).any()]
D2_METHODS = ["오토인코더", "PCA", "상관계수"]
D2_RULES = ["기본", "업종", "공적분", "업종+공적분"]
d2_res = {}          # (형성, 방법, 규칙, 진입, 손절) -> [기간별 일별수익 Series]
d2_tr = {}           # 같은 키 -> 매매 목록
d2_pairs_log = []
d2_fin_rows = []     # 재무 특성 오토인코더 비교 (재무 자료가 있는 기간만)
for s in d2_starts:
    s_pos = d2_days.get_loc(s)
    trade_idx = d2_days[s_pos: s_pos + D2_TRADE]
    if len(trade_idx) < 20:
        continue
    for F in D2_FORMS:
        if s_pos < max(D2_FORMS):          # 두 형성 기간 모두 같은 매매 기간에서 비교되게
            continue
        form_idx = d2_days[s_pos - F: s_pos]
        univ = [c for c in d2_top.get(s.year - 1, []) if c in d2_ret.columns]
        fr = d2_ret.loc[form_idx, univ]
        univ = [c for c in univ if fr[c].notna().mean() > 0.98]
        fr = d2_ret.loc[form_idx, univ].fillna(0.0)
        univ = [c for c in univ if fr[c].std() > 0]
        fr = fr[univ]
        x = ((fr - fr.mean()) / fr.std()).T.values
        lat = d2_latent(x, SEED)
        dists = {"오토인코더": d2_euclid(lat),
                 "PCA": d2_euclid(PCA(n_components=D2_LATENT, random_state=SEED).fit_transform(x)),
                 "상관계수": 1 - np.corrcoef(x)}
        arr_cache = {}
        for meth in D2_METHODS:
            for rule in D2_RULES:
                pairs = d2_pick(dists[meth], univ, D2_NPAIR, rule, form_idx)
                for a, b, dist, p in pairs:
                    if (a, b) not in arr_cache:
                        arr_cache[(a, b)] = d2_pair_arrays(a, b, form_idx, trade_idx)
                for entry in D2_ENTRIES:
                    for stop in D2_STOPS:
                        key = (F, meth, rule, entry, stop)
                        nets = []
                        for a, b, dist, p in pairs:
                            net, tl = d2_sim(*arr_cache[(a, b)], entry, stop)
                            nets.append(net)
                            d2_tr.setdefault(key, []).extend(tl)
                            if (F, entry, stop) == D2_BASE:
                                d2_pairs_log.append({"기간시작": s.date(), "형성": F, "방법": meth, "규칙": rule,
                                                     "종목A": f"{d2_names.get(a, '')}({a})", "종목B": f"{d2_names.get(b, '')}({b})",
                                                     "업종": d2_sector.get(a, "?") + ("" if d2_sector.get(a) == d2_sector.get(b) else "/" + d2_sector.get(b, "?")),
                                                     "공적분p": p, "형성기간 상관": float(np.corrcoef(fr[a], fr[b])[0, 1]),
                                                     "매매기간 수익": float(np.prod(1 + net) - 1)})
                        # 짝 10개 자리에 자금 똑같이 (못 채운 자리는 현금 0)
                        tot = np.sum(nets, axis=0) / D2_NPAIR if nets else np.zeros(len(trade_idx))
                        d2_res.setdefault(key, []).append(pd.Series(tot, index=trade_idx))
        # 재무 특성 오토인코더: 형성 끝 날짜까지 나온 최신 재무값을 수익률 묶음 옆에 붙임
        if F == D2_BASE[0] and len(d2_fin) and form_idx[-1] >= d2_fin["날짜"].min() + pd.Timedelta(days=30):
            snap = d2_fin[d2_fin["날짜"] <= form_idx[-1]].groupby("종목코드").last()
            fin = snap.reindex(univ)[D2_FIN_COLS].astype(float)
            fin["시가총액"] = np.log(fin["시가총액"].where(fin["시가총액"] > 0))
            fin = fin.rank(pct=True) - 0.5                    # 순위로 바꿔 이상치(적자 PER 등) 영향 줄이기
            fin = fin.fillna(0.0)
            fz = fin.values / (fin.values.std(0) + 1e-9) * np.sqrt(x.shape[1] / len(D2_FIN_COLS)) * D2_FIN_W
            lat_f = d2_latent(np.hstack([x, fz]), SEED)
            dist_f = d2_euclid(lat_f)
            for rule in ["기본", "업종"]:
                for lab, dist_use in [("오토인코더(수익률만)", dists["오토인코더"]), ("오토인코더(수익률+재무)", dist_f)]:
                    pairs = d2_pick(dist_use, univ, D2_NPAIR, rule, form_idx)
                    nets = [d2_sim(*d2_pair_arrays(a, b, form_idx, trade_idx), D2_BASE[1], D2_BASE[2])[0] for a, b, _, _ in pairs]
                    tot = np.sum(nets, axis=0) / D2_NPAIR if nets else np.zeros(len(trade_idx))
                    d2_fin_rows.append({"기간시작": s.date(), "규칙": rule, "방법": lab,
                                        "기간수익": float(np.prod(1 + tot) - 1), "_ser": pd.Series(tot, index=trade_idx)})
    print(f"  {s.date()} 시작 완료 (짝 기록 {len(d2_pairs_log)}개)")


def d2_concat(lst):
    r = pd.concat(lst).sort_index()
    return r[~r.index.duplicated(keep="first")]


d2_ret_all = {k: d2_concat(v) for k, v in d2_res.items()}


def d2_label(k):
    F, meth, rule, entry, stop = k
    return f"{meth} · {rule} · 형성{F} · 진입{entry} · 손절{'없음' if stop is None else stop}"


def d2_row(k):
    st = perf_stats(d2_ret_all[k], 252)
    t = pd.DataFrame(d2_tr.get(k, []))
    st.update({"매매 수": len(t),
               "승률": (t["수익"] > 0).mean() if len(t) else np.nan,
               "1회 평균수익": t["수익"].mean() if len(t) else np.nan,
               "수렴 비율": (t["사유"] == "수렴").mean() if len(t) else np.nan,
               "손절 비율": (t["사유"] == "손절").mean() if len(t) else np.nan,
               "평균 보유일": t["보유일"].mean() if len(t) else np.nan})
    return st


# 표 1: 규칙 비교 (처음 실험 설정 그대로)
d2_tab_rule = pd.DataFrame({f"{m} · {r}": d2_row((D2_BASE[0], m, r, D2_BASE[1], D2_BASE[2]))
                            for m in D2_METHODS for r in D2_RULES}).T
d2_tab_rule.index.name = "방법 · 규칙"
# 표 2: 파라미터 전체 격자
d2_tab_grid = pd.DataFrame({d2_label(k): d2_row(k) for k in d2_ret_all}).T.sort_values("샤프", ascending=False)
d2_tab_grid.index.name = "방법 · 규칙 · 형성 · 진입 · 손절"
# 표 3: 짝의 질
d2_pairs_df = pd.DataFrame(d2_pairs_log)
d2_quality = d2_pairs_df.groupby(["방법", "규칙"])[["형성기간 상관", "공적분p", "매매기간 수익"]].mean() if len(d2_pairs_df) else pd.DataFrame()
if len(d2_quality):
    d2_quality["짝 수(기간 평균)"] = d2_pairs_df.groupby(["방법", "규칙", "기간시작"]).size().groupby(["방법", "규칙"]).mean()
    d2_quality.index = [f"{a} · {b}" for a, b in d2_quality.index]
    d2_quality.index.name = "방법 · 규칙"
# 표 4: 재무 특성
if d2_fin_rows:
    d2_fin_df = pd.DataFrame(d2_fin_rows)
    d2_fin_tab = {}
    for (rule, lab), g in d2_fin_df.groupby(["규칙", "방법"]):
        st = perf_stats(d2_concat(list(g["_ser"])), 252)
        st["기간 수"] = len(g)
        d2_fin_tab[f"{lab} · {rule}"] = st
    d2_fin_tab = pd.DataFrame(d2_fin_tab).T
    d2_fin_tab.index.name = "방법 · 규칙"
    d2_fin_period = d2_fin_df.drop(columns="_ser").pivot_table(index="기간시작", columns=["규칙", "방법"], values="기간수익")
    d2_fin_period.columns = [f"{a} · {b}" for a, b in d2_fin_period.columns]
else:
    d2_fin_tab = pd.DataFrame()
    d2_fin_period = pd.DataFrame()

d2_best = d2_tab_grid.index[0]
d2_base_key = (D2_BASE[0], "오토인코더", "기본", D2_BASE[1], D2_BASE[2])
d2_show = {d2_label(d2_base_key): d2_ret_all[d2_base_key], d2_best + " (격자 최고)": d2_ret_all[[k for k in d2_ret_all if d2_label(k) == d2_best][0]]}
for m in D2_METHODS:
    k = (D2_BASE[0], m, "업종+공적분", D2_BASE[1], D2_BASE[2])
    d2_show[d2_label(k)] = d2_ret_all[k]
d2_year = yearly_table(d2_show)

# ----- 5) 그림 -----
d2_fig1, ax = plt.subplots(figsize=(12, 4.8))
for k, r in d2_show.items():
    ax.plot((1 + r).cumprod(), label=k, linewidth=1.8 if "최고" in k else 1.1)
ax.set_title("추가4-2: 짝 고르는 규칙·매매 기준 비교 (비용 차감)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
d2_fig1.tight_layout()
plt.show()
print(md_table(d2_tab_rule[["CAGR", "샤프", "MDD", "매매 수", "승률", "손절 비율"]], pct_cols=("CAGR", "MDD", "승률", "손절 비율")))

# ----- 6) 결과 자동 저장 -----
d2_pct = ("CAGR", "변동성", "MDD", "누적수익", "플러스비율", "승률", "1회 평균수익", "수렴 비율", "손절 비율")
d2_auto = [
    "- 규칙별 샤프(처음 설정): " + " / ".join(f"{i} {d2_tab_rule.loc[i, '샤프']:.2f}" for i in d2_tab_rule.index),
    f"- 격자 최고: {d2_best} (샤프 {d2_tab_grid.iloc[0]['샤프']:.2f}, CAGR {d2_tab_grid.iloc[0]['CAGR'] * 100:.2f}%, 손절 {d2_tab_grid.iloc[0]['손절 비율'] * 100:.0f}%)",
    f"- 격자 {len(d2_tab_grid)}개 중 샤프 > 0 인 조합: {int((d2_tab_grid['샤프'] > 0).sum())}개 (많이 시험할수록 우연히 좋은 조합이 나옴 -> 최고 하나보다 '대부분이 양수인지'를 봐야 함)",
]
if len(d2_fin_tab):
    d2_auto.append("- 재무 특성 오토인코더(재무 자료 있는 기간만): " + " / ".join(f"{i} 샤프 {d2_fin_tab.loc[i, '샤프']:.2f}" for i in d2_fin_tab.index))
save_result(
    D2_TAG, "추가 전략 4-2: 페어 트레이딩 개선 - 같은 업종, 공적분, 매매 기준, 재무 특성",
    [
        ("한눈에", "\n".join(d2_auto)),
        ("설정", "\n".join([
            f"- 유니버스·매매 방식은 추가 전략 4와 같음 (직전 해 거래대금 상위 {D2_UNIVERSE}, 1·7월 시작 {D2_TRADE}일 매매, 짝 {D2_NPAIR}개 같은 비중, 한쪽 비용 {D2_COST_LEG * 100:.2f}%)",
            f"- 규칙: 기본(거리만) / 업종(kor_sector 같은 업종끼리) / 공적분(Engle-Granger p < {D2_COINT_P}, 가까운 짝부터 {D2_COINT_TRY}개까지 검정) / 업종+공적분",
            f"- 격자: 형성 {D2_FORMS}일 x 진입 z {D2_ENTRIES} x 손절 z {['없음' if s is None else s for s in D2_STOPS]} x 방법 3 x 규칙 4 = {len(d2_tab_grid)}개",
            f"- 재무 특성: {', '.join(D2_FIN_COLS)} 순위를 수익률 252개 옆에 붙여 오토인코더 입력 (크기 비중 {D2_FIN_W}), 재무 자료가 있는 기간만 비교",
        ])),
        ("규칙 비교 (형성 252, 진입 2, 손절 4 = 처음 설정)", md_table(d2_tab_rule, pct_cols=d2_pct, digits=2)),
        ("짝의 질 (규칙별 평균)", md_table(d2_quality, pct_cols=("매매기간 수익",), digits=3) if len(d2_quality) else "-"),
        ("파라미터 격자 전체 (샤프 높은 순)", md_table(d2_tab_grid, pct_cols=d2_pct, digits=2)),
        ("연도별 수익률 (대표 조합)", md_table(d2_year, pct_cols="all")),
        ("재무 특성 오토인코더 비교", md_table(d2_fin_tab, pct_cols=("CAGR", "변동성", "MDD", "누적수익", "플러스비율")) if len(d2_fin_tab) else "재무 자료가 있는 기간이 없어 건너뜀"),
        ("재무 비교 기간별 수익", md_table(d2_fin_period, pct_cols="all") if len(d2_fin_period) else "-"),
        ("주의", "\n".join([
            "- kor_sector 는 현재 시점 업종 하나만 있다(과거 업종 변경 반영 안 됨).",
            "- 격자에서 최고 조합만 고르면 과최적화. 이웃 조합(진입·손절만 다른 것)도 비슷하게 좋은지 같이 볼 것.",
            "- 생존 편향·공매도 제약은 추가 전략 4와 같음 (연구용).",
        ])),
    ],
    {"규칙비교": d2_tab_rule, "격자": d2_tab_grid, "짝목록": d2_pairs_df.set_index("기간시작") if len(d2_pairs_df) else pd.DataFrame(),
     "대표_일별수익": pd.DataFrame(d2_show), "연도별수익": d2_year, "재무비교": d2_fin_tab, "재무_기간별": d2_fin_period},
    {"누적수익": d2_fig1},
    {"규칙비교": d2_tab_rule.to_dict(orient="index"), "격자_상위10": d2_tab_grid.head(10).to_dict(orient="index"),
     "재무비교": d2_fin_tab.to_dict(orient="index") if len(d2_fin_tab) else {}},
)


#%% [추가 전략 4-3] 페어 트레이딩 보완: 놀고 있는 자금 줄이기 + 오토인코더 보완 (재무 특성은 11-26에 따로)
# 무엇: 4-2에서 공적분 필터만 효과가 있었지만 짝이 4~5개뿐이라 자금이 놀았다(연 2.5%). 하나씩 더해 본다.
#  1) 짝 후보 늘리기(검정 60->300개, 유니버스 100->200)  2) 열린 짝에만 자금 배분(한 짝 최대 20%)
#  3) 매달 시작하는 묶음 6개 겹치기  4) 놀린 돈에 3개월 금리 이자  5) 변동성 맞춤(연 8%)
#  6) 오토인코더 보완: 시장 움직임 빼고 학습 / 시드 3개 평균 / 잡음 제거 오토인코더
# 실행: 맨 위 [1] 환경 설정 셀 먼저, 그다음 이 셀. 오토인코더를 많이 학습해 40분~1시간
# 결과: 06_퀀트공부/딥러닝/결과/추가4-3_페어_자금활용_오토인코더보완/ 에 자동 저장
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

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
try:
    from statsmodels.tsa.stattools import coint as d3_coint
    D3_HAS_SM = True
except ImportError:
    D3_HAS_SM = False

# ----- 설정 (바꿔 보면서 다시 돌려 보기) -----
D3_TAG = "추가4-3_페어_자금활용_오토인코더보완"
D3_START = "2012-01-01"
D3_FORM = 252
D3_TRADE = 126
D3_NPAIR = 10
D3_LATENT = 6
D3_COST_LEG = 0.002
D3_ENTRY = 2.0
D3_STOPS = [None, 4.0]              # 4-2에서 손절 없음이 대체로 나았음 -> 둘 다 확인
D3_COINT_P = 0.05
D3_TRYS = [60, 300]                 # 공적분 검정을 가까운 짝 몇 개까지 할지 (많을수록 10쌍을 채우기 쉬움)
D3_UNIVS = [100, 200]               # 유니버스 크기
D3_MIN_OPEN = 5                     # '열린 짝에만 배분': 한 짝 최대 1/5 = 20%
D3_TARGET_VOL = 0.08                # 변동성 맞춤 목표(연)
D3_MAX_LEV = 3.0
D3_SEEDS = [SEED, SEED + 1, SEED + 2]
D3_NOISE = 0.3                      # 잡음 제거 오토인코더: 입력에 섞는 잡음 크기(표준화 단위)
D3_MONTHLY_TRY = 150                # 매달 겹치기에서 공적분 검정 개수 (시간 때문에 300보다 작게)
D3_RF_ID = "IR3TIB01KRM156N"        # 한국 3개월 금리(월, %) -> 놀고 있는 돈 이자

# ----- 1) 데이터 -----
print("[추가4-3] 연도별 거래대금 집계 중... (DB에서 1분 안팎)")
d3_liq = load_sql(
    "SELECT 종목코드, YEAR(날짜) AS 연도, AVG(종가 * 거래량) AS 평균거래대금, COUNT(*) AS 일수 "
    f"FROM kor_price WHERE 날짜 >= '{pd.Timestamp(D3_START).year - 1}-01-01' GROUP BY 종목코드, YEAR(날짜)"
)
d3_liq = d3_liq[d3_liq["일수"] >= 200]
d3_top = {int(y): g.nlargest(max(D3_UNIVS), "평균거래대금")["종목코드"].tolist() for y, g in d3_liq.groupby("연도")}
d3_codes = sorted(set(c for v in d3_top.values() for c in v))
print(f"[추가4-3] 후보 {len(d3_codes)}종목 가격 불러오는 중...")
d3_raw = load_sql(
    "SELECT 날짜, 종목코드, 종가 FROM kor_price "
    f"WHERE 날짜 >= '{pd.Timestamp(D3_START) - pd.DateOffset(years=1):%Y-%m-%d}' "
    f"AND 종목코드 IN ({', '.join(repr(c) for c in d3_codes)})"
)
d3_raw["날짜"] = pd.to_datetime(d3_raw["날짜"])
d3_px = d3_raw.pivot(index="날짜", columns="종목코드", values="종가").sort_index()
d3_ret = d3_px.pct_change(fill_method=None)
d3_ret = d3_ret.where(d3_ret.abs() < 0.35)
d3_logp = np.log((1 + d3_ret.fillna(0.0)).cumprod().where(d3_px.ffill().notna()))
d3_sec = load_sql("SELECT CMP_CD, SEC_NM_KOR FROM kor_sector WHERE 기준일 = (SELECT MAX(기준일) FROM kor_sector)")
d3_sector = dict(zip(d3_sec["CMP_CD"], d3_sec["SEC_NM_KOR"])) if len(d3_sec) else {}
d3_rf_raw = load_sql(f"SELECT 날짜, 값 FROM macro_series WHERE 지표ID = '{D3_RF_ID}'")
d3_days = d3_px.index
if len(d3_rf_raw):
    d3_rf_raw["날짜"] = pd.to_datetime(d3_rf_raw["날짜"])
    rf = d3_rf_raw.set_index("날짜")["값"].astype(float).sort_index()
    d3_rf = (rf.reindex(rf.index.union(d3_days)).ffill().reindex(d3_days) / 100 / 252).fillna(0.0)
else:
    d3_rf = pd.Series(0.0, index=d3_days)
print(f"[추가4-3] 금리 자료 {len(d3_rf_raw)}개월, statsmodels {D3_HAS_SM}")


# ----- 2) 오토인코더(보완 3가지 포함) / 거리 / 공적분 -----
def d3_latent(x, seed, noise=0.0):
    """noise > 0 이면 잡음 제거 오토인코더: 잡음 섞은 입력으로 원래 입력을 복원하게 학습"""
    if TF_AVAILABLE:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, GaussianNoise
        tf.random.set_seed(seed)
        inp = Input(shape=(x.shape[1],))
        h = GaussianNoise(noise)(inp) if noise > 0 else inp      # 학습 때만 잡음, 예측 때는 꺼짐
        h = Dense(32, activation="tanh")(h)
        z = Dense(D3_LATENT, activation="tanh")(h)
        h2 = Dense(32, activation="tanh")(z)
        out = Dense(x.shape[1], activation="linear")(h2)
        ae = Model(inp, out)
        ae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        ae.fit(x, x, epochs=300, batch_size=16, verbose=0)
        return Model(inp, z).predict(x, verbose=0)
    rng = np.random.default_rng(seed)
    xin = x + rng.normal(0, noise, x.shape) if noise > 0 else x
    m = MLPRegressor(hidden_layer_sizes=(32, D3_LATENT, 32), activation="tanh", alpha=1e-4, max_iter=2000, random_state=seed)
    m.fit(xin, x)
    h = np.tanh(x @ m.coefs_[0] + m.intercepts_[0])
    return np.tanh(h @ m.coefs_[1] + m.intercepts_[1])


def d3_euclid(z):
    z = (z - z.mean(0)) / (z.std(0) + 1e-9)
    sq = (z ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * z @ z.T, 0))


d3_pcache = {}


def d3_coint_p(a, b, form_idx):
    key = (a, b, form_idx[0], form_idx[-1])
    if key in d3_pcache:
        return d3_pcache[key]
    p = 1.0
    if D3_HAS_SM:
        la, lb = d3_logp[a].loc[form_idx], d3_logp[b].loc[form_idx]
        ok = la.notna() & lb.notna()
        if ok.sum() >= 60:
            try:
                p = float(d3_coint(la[ok].values, lb[ok].values)[1])
            except Exception:
                p = 1.0
    d3_pcache[key] = p
    return p


def d3_pick(dist, codes, rule, ntry, form_idx):
    iu = np.triu_indices(len(codes), 1)
    order = np.argsort(dist[iu])
    used, pairs, tried = set(), [], 0
    for o in order:
        a, b = iu[0][o], iu[1][o]
        if a in used or b in used:
            continue
        ca, cb = codes[a], codes[b]
        if rule == "업종+공적분" and (d3_sector.get(ca) is None or d3_sector.get(ca) != d3_sector.get(cb)):
            continue
        if tried >= ntry:
            break
        tried += 1
        if not (d3_coint_p(ca, cb, form_idx) < D3_COINT_P):
            continue
        pairs.append((ca, cb))
        used.update([a, b])
        if len(pairs) == D3_NPAIR:
            break
    return pairs


def d3_pair_arrays(a, b, form_idx, trade_idx):
    la_f, lb_f = d3_logp[a].loc[form_idx], d3_logp[b].loc[form_idx]
    ok = la_f.notna() & lb_f.notna()
    beta, alpha = np.polyfit(lb_f[ok].values, la_f[ok].values, 1)
    resid = la_f - (alpha + beta * lb_f)
    z = ((d3_logp[a].loc[trade_idx] - (alpha + beta * d3_logp[b].loc[trade_idx])) - resid.mean()) / resid.std()
    return (z.ffill().fillna(0.0).values, d3_ret[a].loc[trade_idx].fillna(0.0).values,
            d3_ret[b].loc[trade_idx].fillna(0.0).values)


def d3_sim(zv, ra, rb, stop):
    """반환: 짝 하나에 자금 1을 넣었을 때 일별 순수익, 그날 포지션 보유 여부, 매매 목록"""
    stop = np.inf if stop is None else stop
    pos = np.zeros(len(zv))
    cur, blocked, e_i, trades = 0, False, None, []
    for i in range(len(zv) - 1):
        if cur == 0 and not blocked:
            if D3_ENTRY < zv[i] < stop:
                cur, e_i = -1, i
            elif -stop < zv[i] < -D3_ENTRY:
                cur, e_i = 1, i
        elif cur != 0:
            reason = None
            if (cur == -1 and zv[i] <= 0) or (cur == 1 and zv[i] >= 0):
                reason = "수렴"
            elif abs(zv[i]) >= stop:
                reason, blocked = "손절", True
            if reason:
                trades.append((e_i, i, reason))
                cur = 0
        pos[i] = cur
    if cur != 0:
        trades.append((e_i, len(zv) - 1, "기간끝"))
    held = np.concatenate([[0.0], pos[:-1]])
    cost = np.abs(np.diff(np.concatenate([[0.0], pos]))) * D3_COST_LEG
    net = held * 0.5 * (ra - rb) - np.concatenate([[0.0], cost[:-1]])
    tl = [{"사유": rs, "수익": float(np.prod(1 + net[i0 + 1: i1 + 2]) - 1)} for i0, i1, rs in trades]
    return net, (held != 0).astype(float), tl


def d3_combine(sims, alloc):
    """짝 여러 개 -> (일별수익, 놀린 자금 비율). 균등10 = 짝 자리 10개에 1/10씩, 열린짝 = 열린 짝끼리 나눔(최대 1/5)"""
    T = len(sims[0][0]) if sims else 0
    if not sims:
        return None, None
    net = np.array([s[0] for s in sims])
    held = np.array([s[1] for s in sims])
    n_open = held.sum(0)
    if alloc == "균등10":
        return net.sum(0) / D3_NPAIR, 1 - n_open / D3_NPAIR
    active = (held > 0) | (net != 0)
    w = 1.0 / np.maximum(n_open, D3_MIN_OPEN)
    return (net * active).sum(0) * w, 1 - n_open * w


def d3_universe(s_pos, form_idx, n):
    univ = [c for c in d3_top.get(d3_days[s_pos].year - 1, [])[:n] if c in d3_ret.columns]
    fr = d3_ret.loc[form_idx, univ]
    univ = [c for c in univ if fr[c].notna().mean() > 0.98]
    fr = d3_ret.loc[form_idx, univ].fillna(0.0)
    univ = [c for c in univ if fr[c].std() > 0]
    return univ, fr[univ]


def d3_dists(fr, kinds):
    x = ((fr - fr.mean()) / fr.std()).T.values
    out = {}
    if "오토인코더" in kinds:
        out["오토인코더"] = d3_euclid(d3_latent(x, SEED))
    if "오토인코더(시장제거)" in kinds:          # 그날 시장 평균을 빼서 '시장과 같이 움직임'을 지우고 종목끼리의 닮음만
        frm = fr.sub(fr.mean(axis=1), axis=0)
        xm = ((frm - frm.mean()) / (frm.std() + 1e-12)).T.values
        out["오토인코더(시장제거)"] = d3_euclid(d3_latent(xm, SEED))
    if "오토인코더(시드3평균)" in kinds:          # 시드마다 잠재 좌표가 달라 거리를 3번 평균
        ds = [out["오토인코더"]] if "오토인코더" in out else []
        ds += [d3_euclid(d3_latent(x, s)) for s in D3_SEEDS[len(ds):]]
        out["오토인코더(시드3평균)"] = np.mean(ds, axis=0)
    if "오토인코더(잡음제거)" in kinds:
        out["오토인코더(잡음제거)"] = d3_euclid(d3_latent(x, SEED, noise=D3_NOISE))
    if "PCA" in kinds:
        out["PCA"] = d3_euclid(PCA(n_components=D3_LATENT, random_state=SEED).fit_transform(x))
    if "상관계수" in kinds:
        out["상관계수"] = 1 - np.corrcoef(x)
    return out


# ----- 3) 1·7월 시작: 유니버스 2 x 방법 6 x 규칙 2 x 검정 개수 2 x 손절 2 x 배분 2 -----
D3_K100 = ["오토인코더", "오토인코더(시장제거)", "오토인코더(시드3평균)", "오토인코더(잡음제거)", "PCA", "상관계수"]
D3_K200 = ["오토인코더", "PCA", "상관계수"]
d3_starts = [d3_days[d3_days >= pd.Timestamp(f"{y}-{m:02d}-01")][0]
             for y in range(pd.Timestamp(D3_START).year + 1, d3_days.max().year + 1) for m in (1, 7)
             if (d3_days >= pd.Timestamp(f"{y}-{m:02d}-01")).any()]
d3_res, d3_idle, d3_tr = {}, {}, {}
for s in d3_starts:
    s_pos = d3_days.get_loc(s)
    trade_idx = d3_days[s_pos: s_pos + D3_TRADE]
    if s_pos < D3_FORM or len(trade_idx) < 20:
        continue
    form_idx = d3_days[s_pos - D3_FORM: s_pos]
    for U in D3_UNIVS:
        univ, fr = d3_universe(s_pos, form_idx, U)
        dists = d3_dists(fr, D3_K100 if U == 100 else D3_K200)
        cache = {}
        for meth, dist in dists.items():
            for rule in ["공적분", "업종+공적분"]:
                for ntry in D3_TRYS:
                    pairs = d3_pick(dist, univ, rule, ntry, form_idx)
                    for stop in D3_STOPS:
                        sims = []
                        for a, b in pairs:
                            if (a, b) not in cache:
                                cache[(a, b)] = d3_pair_arrays(a, b, form_idx, trade_idx)
                            sims.append(d3_sim(*cache[(a, b)], stop))
                        for alloc in ["균등10", "열린짝"]:
                            key = (U, meth, rule, ntry, stop, alloc)
                            r, idle = d3_combine(sims, alloc)
                            if r is None:
                                r, idle = np.zeros(len(trade_idx)), np.ones(len(trade_idx))
                            d3_res.setdefault(key, []).append(pd.Series(r, index=trade_idx))
                            d3_idle.setdefault(key, []).append(pd.Series(idle, index=trade_idx))
                            if alloc == "균등10":
                                d3_tr.setdefault(key, []).extend(t for sm in sims for t in sm[2])
    print(f"  {s.date()} 시작 완료 (공적분 검정 누적 {len(d3_pcache)}개)")


def d3_cat(lst):
    r = pd.concat(lst).sort_index()
    return r[~r.index.duplicated(keep="first")]


def d3_lab(k):
    U, meth, rule, ntry, stop, alloc = k
    return f"상위{U} · {meth} · {rule} · 검정{ntry} · 손절{'없음' if stop is None else stop} · {alloc}"


def d3_row(r, idle, trades=None):
    st = perf_stats(r, 252)
    st["평균 놀린 자금"] = float(idle.mean())
    if trades is not None:
        t = pd.DataFrame(trades)
        st.update({"매매 수": len(t), "승률": (t["수익"] > 0).mean() if len(t) else np.nan,
                   "손절 비율": (t["사유"] == "손절").mean() if len(t) else np.nan})
    return st


d3_R = {k: d3_cat(v) for k, v in d3_res.items()}
d3_I = {k: d3_cat(v) for k, v in d3_idle.items()}
d3_grid = pd.DataFrame({d3_lab(k): d3_row(d3_R[k], d3_I[k], d3_tr.get((*k[:5], "균등10"))) for k in d3_R}).T
d3_grid = d3_grid.sort_values("샤프", ascending=False)
d3_grid.index.name = "유니버스 · 방법 · 규칙 · 검정 · 손절 · 배분"
# 오토인코더 보완 비교 (같은 조건에서 방법만 바꿈)
d3_ae_keys = [(100, m, "공적분", max(D3_TRYS), None, "균등10") for m in D3_K100]
d3_ae_tab = pd.DataFrame({m: d3_row(d3_R[k], d3_I[k], d3_tr.get(k)) for m, k in zip(D3_K100, d3_ae_keys)}).T
d3_ae_tab.index.name = f"방법 (상위100 · 공적분 · 검정{max(D3_TRYS)} · 손절없음 · 균등10)"

# ----- 4) 매달 시작하는 묶음 6개 겹치기 (방법 3개, 공적분, 손절 없음) -----
print("[추가4-3] 매달 겹치기 계산 중... (방법 3개 x 달마다 오토인코더 학습)")
d3_mstarts = [d3_days[d3_days >= pd.Timestamp(f"{y}-{m:02d}-01")][0]
              for y in range(pd.Timestamp(D3_START).year + 1, d3_days.max().year + 1) for m in range(1, 13)
              if (d3_days >= pd.Timestamp(f"{y}-{m:02d}-01")).any()]
d3_coh = {}            # (방법, 배분) -> [(일별수익, 놀린비율)]
for s in d3_mstarts:
    s_pos = d3_days.get_loc(s)
    trade_idx = d3_days[s_pos: s_pos + D3_TRADE]
    if s_pos < D3_FORM or len(trade_idx) < 20:
        continue
    form_idx = d3_days[s_pos - D3_FORM: s_pos]
    univ, fr = d3_universe(s_pos, form_idx, 100)
    dists = d3_dists(fr, ["오토인코더", "PCA", "상관계수"])
    for meth, dist in dists.items():
        pairs = d3_pick(dist, univ, "공적분", D3_MONTHLY_TRY, form_idx)
        sims = [d3_sim(*d3_pair_arrays(a, b, form_idx, trade_idx), None) for a, b in pairs]
        for alloc in ["균등10", "열린짝"]:
            r, idle = d3_combine(sims, alloc)
            if r is None:
                r, idle = np.zeros(len(trade_idx)), np.ones(len(trade_idx))
            d3_coh.setdefault((meth, alloc), []).append((pd.Series(r, index=trade_idx), pd.Series(idle, index=trade_idx)))
    if s.month == 12:
        print(f"  {s.year}년 완료")


def d3_stack(lst):
    """겹친 묶음들 평균: 그날 살아 있는 묶음끼리 자금을 나눔 (최대 6개)"""
    r = pd.concat([x[0] for x in lst], axis=1).mean(axis=1)
    idle = pd.concat([x[1] for x in lst], axis=1).mean(axis=1)
    return r.sort_index(), idle.sort_index()


d3_M = {k: d3_stack(v) for k, v in d3_coh.items()}


# ----- 5) 놀린 돈 이자 + 변동성 맞춤 -----
def d3_with_cash(r, idle):
    return r + idle * d3_rf.reindex(r.index).fillna(0.0)


def d3_voltarget(r):
    vol = r.rolling(60).std().shift(1) * np.sqrt(252)
    lev = (D3_TARGET_VOL / vol).clip(upper=D3_MAX_LEV).fillna(1.0)
    return r * lev


d3_mtab = {}
for (meth, alloc), (r, idle) in d3_M.items():
    d3_mtab[f"{meth} · {alloc} · 매달겹치기"] = d3_row(r, idle)
    d3_mtab[f"{meth} · {alloc} · 매달겹치기 + 이자"] = d3_row(d3_with_cash(r, idle), idle)
    d3_mtab[f"{meth} · {alloc} · 매달겹치기 + 이자 + 변동성맞춤"] = d3_row(d3_voltarget(d3_with_cash(r, idle)), idle)
d3_mtab = pd.DataFrame(d3_mtab).T.sort_values("샤프", ascending=False)
d3_mtab.index.name = "방법 · 배분 · 보완"

# 단계별 사다리: 4-2 최고 근처 설정에서 하나씩 보완을 더함 (방법은 매달 겹치기에서 가장 좋은 것)
d3_best_meth = d3_mtab.index[0].split(" · ")[0]
d3_ladder = {
    f"1) 4-2 설정 (상위100·검정{min(D3_TRYS)}·손절없음·균등10, 1·7월)": d3_R[(100, d3_best_meth, "공적분", min(D3_TRYS), None, "균등10")],
    f"2) + 검정 {max(D3_TRYS)}개": d3_R[(100, d3_best_meth, "공적분", max(D3_TRYS), None, "균등10")],
    "3) + 열린 짝에만 배분": d3_R[(100, d3_best_meth, "공적분", max(D3_TRYS), None, "열린짝")],
    "4) + 매달 겹치기": d3_M[(d3_best_meth, "열린짝")][0],
    "5) + 놀린 돈 이자": d3_with_cash(*d3_M[(d3_best_meth, "열린짝")]),
    "6) + 변동성 맞춤(연 8%)": d3_voltarget(d3_with_cash(*d3_M[(d3_best_meth, "열린짝")])),
}
d3_ladder_tab = stats_frame(d3_ladder, 252)
d3_ladder_tab.index.name = f"단계 (방법: {d3_best_meth})"
d3_year = yearly_table(d3_ladder)

# ----- 6) 그림 -----
d3_fig1, ax = plt.subplots(figsize=(12, 4.8))
for k, r in d3_ladder.items():
    ax.plot((1 + r).cumprod(), label=k, linewidth=1.8 if k.startswith("6)") else 1.1)
ax.set_title(f"추가4-3: 자금을 놀리지 않게 하나씩 보완 ({d3_best_meth}, 비용 차감)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
d3_fig1.tight_layout()
plt.show()
print(md_table(d3_ladder_tab[["CAGR", "샤프", "MDD"]], pct_cols=("CAGR", "MDD")))

# ----- 7) 결과 자동 저장 -----
d3_pct = ("CAGR", "변동성", "MDD", "누적수익", "플러스비율", "평균 놀린 자금", "승률", "손절 비율")
d3_auto = [
    "- 단계별 샤프: " + " / ".join(f"{k.split(')')[0]}) {d3_ladder_tab.loc[k, '샤프']:.2f}" for k in d3_ladder),
    f"- 단계별 CAGR: " + " / ".join(f"{k.split(')')[0]}) {d3_ladder_tab.loc[k, 'CAGR'] * 100:.1f}%" for k in d3_ladder),
    "- 오토인코더 보완 샤프: " + " / ".join(f"{i} {d3_ae_tab.loc[i, '샤프']:.2f}" for i in d3_ae_tab.index),
    f"- 1·7월 격자 {len(d3_grid)}개 중 샤프 > 0: {int((d3_grid['샤프'] > 0).sum())}개, 최고 {d3_grid.index[0]} ({d3_grid.iloc[0]['샤프']:.2f})",
]
save_result(
    D3_TAG, "추가 전략 4-3: 페어 트레이딩 - 놀고 있는 자금 줄이기 + 오토인코더 보완 (재무 특성 제외)",
    [
        ("한눈에", "\n".join(d3_auto)),
        ("설정", "\n".join([
            f"- 공통: 형성 {D3_FORM}일, 매매 {D3_TRADE}일, 진입 z {D3_ENTRY}, 공적분 p < {D3_COINT_P}, 한쪽 비용 {D3_COST_LEG * 100:.2f}%",
            f"- 짝 후보 늘리기: 공적분 검정 개수 {D3_TRYS}, 유니버스 {D3_UNIVS}",
            f"- 열린 짝에만 배분: 그날 열린 짝끼리 자금을 나눔, 한 짝 최대 1/{D3_MIN_OPEN}",
            f"- 매달 겹치기: 매달 새 묶음(짝 최대 {D3_NPAIR}, 검정 {D3_MONTHLY_TRY}개)을 만들어 {D3_TRADE}일 굴림 -> 약 6개 묶음이 겹침",
            f"- 놀린 돈 이자: 한국 3개월 금리(macro_series {D3_RF_ID}, 월별)",
            f"- 변동성 맞춤: 최근 60일 변동성으로 연 {D3_TARGET_VOL * 100:.0f}% 목표, 최대 {D3_MAX_LEV}배 (연구용: 실제로는 공매도·대차 제약)",
            f"- 오토인코더 보완: 시장제거(그날 평균 수익을 빼고 학습), 시드3평균(시드 3개 거리 평균), 잡음제거(잡음 {D3_NOISE})",
        ])),
        ("단계별로 하나씩 보완", md_table(d3_ladder_tab, pct_cols=("CAGR", "변동성", "MDD", "누적수익", "플러스비율"))),
        ("단계별 연도 수익", md_table(d3_year, pct_cols="all")),
        ("매달 겹치기 + 이자 + 변동성 맞춤", md_table(d3_mtab, pct_cols=d3_pct, digits=2)),
        ("오토인코더 보완 비교", md_table(d3_ae_tab, pct_cols=d3_pct, digits=2)),
        ("1·7월 시작 격자 전체 (샤프 높은 순)", md_table(d3_grid, pct_cols=d3_pct, digits=2)),
        ("주의", "\n".join([
            "- 재무 특성 오토인코더는 DART 재무 수집이 끝난 뒤(11-26 예정) 따로 다시 돌린다.",
            "- 변동성 맞춤은 레버리지를 쓴다. 공매도 제약 때문에 개인이 그대로 하긴 어렵다(연구용).",
            "- 업종은 현재 시점 하나, 종목은 생존 편향 있음 (추가 전략 4와 같음).",
        ])),
    ],
    {"단계별_일별수익": pd.DataFrame(d3_ladder), "단계별": d3_ladder_tab, "연도별": d3_year,
     "매달겹치기": d3_mtab, "오토인코더보완": d3_ae_tab, "격자": d3_grid},
    {"누적수익": d3_fig1},
    {"단계별": d3_ladder_tab.to_dict(orient="index"), "오토인코더보완": d3_ae_tab.to_dict(orient="index"),
     "매달겹치기": d3_mtab.to_dict(orient="index")},
)
