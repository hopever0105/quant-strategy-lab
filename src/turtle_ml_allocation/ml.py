"""2차 모델(메타 라벨링): 1차 신호가 뜰 때 '진입 후 20거래일 수익 > 0' 확률을 예측한다.

워크포워드: 매년 1월 1일에 그 전날까지 '결과(20일 뒤 수익)가 이미 확정된' 신호만으로 다시 학습한다.
  → 학습 1998~2007, 첫 예측 2008년. 결과가 아직 안 나온 마지막 20일치 신호는 학습에서 빠진다(정답 미리보기 방지).
비중 배수 = 2p (p<기준이면 0 = 진입 안 함).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from signals import FEATS

TRAIN_START = pd.Timestamp("1998-01-01")
OOS_YEARS = range(2008, 2027)


def make_model(kind):
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                              min_samples_leaf=40, random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def walk_forward(tr, kind="logit", shuffle_seed=None, feats=FEATS):
    """반환: p(Series, OOS 신호만), coefs(DataFrame, 로지스틱일 때 해마다 표준화 계수), n_train(dict),
    base(Series, 그 신호를 예측한 모델의 학습 기간 평균 승률 — 상대 배수용)"""
    tr = tr[tr.d_date >= TRAIN_START]
    p = pd.Series(np.nan, index=tr.index)
    base = pd.Series(np.nan, index=tr.index)
    coefs, ntr = [], {}
    for Y in OOS_YEARS:
        cut = pd.Timestamp(Y, 1, 1)
        trn = tr[(tr.label_date < cut) & tr.label.notna()]
        tst = tr[tr.d_date.dt.year == Y]
        if len(tst) == 0:
            continue
        X = trn[feats]
        med = X.median()
        X = X.fillna(med)
        y = trn.label.values.astype(int)
        if shuffle_seed is not None:
            y = np.random.default_rng(shuffle_seed * 1000 + Y).permutation(y)
        mdl = make_model(kind).fit(X.values, y)
        p.loc[tst.index] = mdl.predict_proba(tst[feats].fillna(med).values)[:, 1]
        base.loc[tst.index] = trn.label.mean()
        ntr[Y] = len(trn)
        if kind == "logit":
            coefs.append(pd.Series(mdl[-1].coef_[0], index=feats, name=Y))
    return p, (pd.DataFrame(coefs) if coefs else None), ntr, base


def to_mult(p, thr=0.3, base=None):
    """기본(사용자 규격): 배수 = 2p, p<thr이면 0.
    base를 주면 상대 배수 = p ÷ 학습 기간 평균 승률(평균 배수가 약 1이 되어 '투자 비중 증가' 효과를 뺀다)."""
    p = np.asarray(p, float)
    m = 2 * p if base is None else p / np.asarray(base, float)
    return np.where(np.isnan(p), 1.0, np.where(p < thr, 0.0, m))


def diagnostics(tr, p, new_assets):
    """OOS 신호에 대한 예측력 점검"""
    d = tr.loc[p.dropna().index].copy()
    d["p"] = p.dropna()
    d = d[d.label.notna()]
    out = dict(n=len(d), base_rate=float(d.label.mean()), auc=float(roc_auc_score(d.label, d.p)),
               acc=float(((d.p >= 0.5) == (d.label == 1)).mean()))
    bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    d["bin"] = pd.cut(d.p, bins, right=False)
    cal = d.groupby("bin", observed=False).agg(n=("label", "size"), p_mean=("p", "mean"),
                                               win=("label", "mean"), fwd20=("fwd20", "mean"))
    f = lambda g: pd.Series(dict(n=len(g), base=g.label.mean(),
                                 auc=roc_auc_score(g.label, g.p) if g.label.nunique() > 1 else np.nan))
    grp = d.groupby("group").apply(f, include_groups=False)
    d["new"] = d.asset.isin(new_assets)
    nw = d.groupby("new").apply(f, include_groups=False)
    taken, skipped = d[d.p >= 0.3], d[d.p < 0.3]
    out.update(taken_n=len(taken), skipped_n=len(skipped),
               taken_win=float(taken.label.mean()), skipped_win=float(skipped.label.mean()) if len(skipped) else np.nan,
               taken_fwd=float(taken.fwd20.mean()), skipped_fwd=float(skipped.fwd20.mean()) if len(skipped) else np.nan)
    return out, cal.reset_index(), grp.reset_index(), nw.reset_index()
