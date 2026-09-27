"""Эксперимент 6: LightGBM-ранкер над объединёнными кандидатами (val v2).

Обучающих запросов с «выдачей» у нас только 3000 (val), поэтому качество ранкера
меряем 5-fold кросс-валидацией по запросам: учим на 4/5 запросов, top-50 строим на 1/5.
Recall считается по ВСЕМ релевантным (в т.ч. не попавшим в кандидаты), как в метрике.

Запуск: .venv/bin/python -m experiments.06_ranker [модель] [--fuse a,b,w] [--geo] [--noloc N] [--profile] [--extra-dense M ...]
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data import CACHE_DIR
from src.pipeline import make_builder
from src.ranker import FUSE, LGB_PARAMS, STATIC_ITEM, feats_tag
from src.validation import build_hard_corpus, load_split, recall_at_k

ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?", default="intfloat/multilingual-e5-small")
ap.add_argument("--fuse", default=None, help="веса a,b,w: a*bm25_norm + b*cos + w*logP(loc)")
ap.add_argument("--geo", action="store_true", help="гео-признаки (см. FeatureBuilder)")
ap.add_argument("--noloc", type=int, default=0, help="доп. кандидаты фьюжна без сильного приора локации")
ap.add_argument("--profile", action="store_true", help="профиль кликов похожих запросов")
ap.add_argument("--extra-dense", nargs="*", default=[], help="доп. bi-encoder'ы (косинус как признак)")
args = ap.parse_args()
model_name = args.model
fuse = tuple(float(x) for x in args.fuse.split(",")) if args.fuse else FUSE
opts = dict(fuse=fuse, geo=args.geo, n_noloc=args.noloc, profile=args.profile, extra_dense=tuple(args.extra_dense))
feat_path = CACHE_DIR / f"ranker_feats_val_{feats_tag(model_name, **opts)}.parquet"

train_fit, queries, qrels, _ = load_split()
corpus = build_hard_corpus()
ids = corpus.item_id.values

if feat_path.exists():
    X = pd.read_parquet(feat_path)
else:
    X = make_builder(corpus, train_fit, model_name, **opts).build(queries, qrels)
    X.to_parquet(feat_path)

FEATURES = [c for c in X.columns if c not in ("query_id", "item_idx", "label")]
print(f"rows={len(X)}, cand/query={len(X) / X.query_id.nunique():.0f}, "
      f"candidate recall={X.groupby('query_id').label.sum().clip(upper=1).mean():.4f} (доля запросов с позитивом)")


def top50(df: pd.DataFrame, score: np.ndarray) -> dict:
    df = df.assign(s=score).sort_values(["query_id", "s"], ascending=[True, False])
    top = df.groupby("query_id").head(50)
    return {q: list(ids[g.item_idx.values]) for q, g in top.groupby("query_id")}


# сегменты из анализа ошибок: поиск по региону и мегаполисы (>10k объявлений в корпусе)
loc_n = queries.set_index("query_id").search_location_id.map(corpus.item_location_id.value_counts()).fillna(0)


def show(name, preds):
    r = recall_at_k(preds, qrels)
    seen = queries.set_index("query_id").seen.reindex(r.index)
    n = loc_n.reindex(r.index)
    print(f"{name:<32s} R@50={r.mean():.4f}  seen={r[seen].mean():.4f}  unseen={r[~seen].mean():.4f}  "
          f"region={r[n == 0].mean():.4f}  megacity={r[n > 10000].mean():.4f}")


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
