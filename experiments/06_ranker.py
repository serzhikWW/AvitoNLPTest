"""Эксперимент 6: LightGBM-ранкер над объединёнными кандидатами (val v2).

Обучающих запросов с «выдачей» у нас только 3000 (val), поэтому качество ранкера
меряем 5-fold кросс-валидацией по запросам: учим на 4/5 запросов, top-50 строим на 1/5.
Recall считается по ВСЕМ релевантным (в т.ч. не попавшим в кандидаты), как в метрике.

Запуск: .venv/bin/python -m experiments.06_ranker [модель эмбеддингов]
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data import CACHE_DIR
from src.dense import DenseRetriever
from src.ranker import LGB_PARAMS, STATIC_ITEM, FeatureBuilder
from src.validation import build_hard_corpus, load_split, recall_at_k

model_name = sys.argv[1] if len(sys.argv) > 1 else "intfloat/multilingual-e5-small"
tag = model_name.rstrip("/").split("/")[-1]
feat_path = CACHE_DIR / f"ranker_feats_val_{tag}.parquet"

train_fit, queries, qrels, _ = load_split()
corpus = build_hard_corpus()
ids = corpus.item_id.values

if feat_path.exists():
    X = pd.read_parquet(feat_path)
else:
    dr = DenseRetriever(model_name)
    fb = FeatureBuilder(corpus, train_fit, dr.item_embeddings(corpus), dr.query_embeddings)
    X = fb.build(queries, qrels)
    X.to_parquet(feat_path)

FEATURES = [c for c in X.columns if c not in ("query_id", "item_idx", "label")]
print(f"rows={len(X)}, cand/query={len(X) / X.query_id.nunique():.0f}, "
      f"candidate recall={X.groupby('query_id').label.sum().clip(upper=1).mean():.4f} (доля запросов с позитивом)")


def top50(df: pd.DataFrame, score: np.ndarray) -> dict:
    df = df.assign(s=score).sort_values(["query_id", "s"], ascending=[True, False])
    top = df.groupby("query_id").head(50)
    return {q: list(ids[g.item_idx.values]) for q, g in top.groupby("query_id")}


def show(name, preds):
    r = recall_at_k(preds, qrels)
    seen = queries.set_index("query_id").seen.reindex(r.index)
    print(f"{name:<32s} R@50={r.mean():.4f}  seen={r[seen].mean():.4f}  unseen={r[~seen].mean():.4f}")


show("fuse (без ранкера)", top50(X, X.fuse.values))


def cv(features):
    oof = np.zeros(len(X))
    imp = pd.Series(0.0, index=features)
    for tr_idx, te_idx in GroupKFold(5).split(X, groups=X.query_id):
        tr = X.iloc[tr_idx].sort_values("query_id", kind="stable")
        m = lgb.LGBMRanker(**LGB_PARAMS)
        m.fit(tr[features], tr.label, group=tr.groupby("query_id", sort=True).size().values)
        oof[te_idx] = m.predict(X.iloc[te_idx][features])
        imp += pd.Series(m.booster_.feature_importance("gain"), index=features)
    return oof, imp


oof, _ = cv(FEATURES)
show("LightGBM, все признаки", top50(X, oof))
# финальный набор: без статичных признаков объявления (см. комментарий в src/ranker.py)
final = [f for f in FEATURES if f not in STATIC_ITEM]
oof, imp = cv(final)
show("LightGBM, без статичных (финал)", top50(X, oof))
print("\nважность признаков (gain, доля):")
print((imp / imp.sum()).sort_values(ascending=False).round(3).to_string())
