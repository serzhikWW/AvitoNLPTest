"""Эксперимент 9: диагностика промахов в мегаполисах (>10k объявлений локации в корпусе).

Финальная конфигурация (--geo --noloc 100), OOF-скоры 5-fold. Смотрим:
  * судьбу релевантных (не в кандидатах / вытеснен / найден) и место вытесненных;
  * сколько кандидатов у запроса и сколько из них «очень похожи» (тот же microcat, что у позитива);
  * держал ли позитив хоть один отдельный сигнал в top-50 (BM25, косинус, фьюжн) — т.е. ранкер его «потерял»,
    или он был далеко по всем сигналам — т.е. признаков для него просто нет.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data import CACHE_DIR
from src.ranker import LGB_PARAMS, feats_tag, ranker_features
from src.validation import build_hard_corpus, load_split

_, queries, qrels, _ = load_split()
corpus = build_hard_corpus()
X = pd.read_parquet(CACHE_DIR / f"ranker_feats_val_{feats_tag('models/multilingual-e5-base-ft-val', (0.5, 1.0, 0.1), True, 100)}.parquet")
feats = ranker_features(X)
oof = np.zeros(len(X))
for tr_idx, te_idx in GroupKFold(5).split(X, groups=X.query_id):
    tr = X.iloc[tr_idx].sort_values("query_id", kind="stable")
    m = lgb.LGBMRanker(**LGB_PARAMS).fit(tr[feats], tr.label, group=tr.groupby("query_id", sort=True).size().values)
    oof[te_idx] = m.predict(X.iloc[te_idx][feats])
X["score"] = oof
X["rank"] = X.groupby("query_id").score.rank(ascending=False, method="first")
X["item_id"] = corpus.item_id.values[X.item_idx.values]
X["mc"] = corpus.item_microcat_id.values[X.item_idx.values]

loc_n = queries.set_index("query_id").search_location_id.map(corpus.item_location_id.value_counts()).fillna(0)
seg = np.where(loc_n > 10000, "megacity", np.where(loc_n == 0, "region", "city"))
seg = pd.Series(seg, index=loc_n.index)
rel = qrels.merge(X[["query_id", "item_id", "rank", "bm25_rank", "cos_rank", "fuse_rank", "mc"]], how="left")
rel["seg"] = rel.query_id.map(seg)
rel["reason"] = np.where(rel["rank"] <= 50, "found", np.where(rel["rank"].isna(), "not_candidate", "ranked_out"))
print("судьба релевантных по сегментам:")
print(pd.crosstab(rel.seg, rel.reason, normalize="index").round(3).to_string(), "\n")

ro = rel[rel.reason == "ranked_out"]
ro = ro.assign(any_top50=(ro[["bm25_rank", "cos_rank", "fuse_rank"]].min(axis=1) <= 50))
print("ranked_out: доля, где позитив был в top-50 хотя бы по одному сигналу (ранкер «потерял»):")
print(ro.groupby("seg").any_top50.mean().round(3).to_string(), "\n")

# конкуренция: сколько кандидатов той же подкатегории, что позитив, и на каком месте позитив среди них
same_mc = X.merge(rel[["query_id", "mc"]].rename(columns={"mc": "pos_mc"}), on="query_id")
same_mc = same_mc[same_mc.mc == same_mc.pos_mc].groupby("query_id").size()
rel["n_same_mc"] = rel.query_id.map(same_mc)
print("кандидатов той же подкатегории, что и позитив (медиана):")
print(rel.groupby("seg").n_same_mc.median().to_string(), "\n")
print("вытесненные в мегаполисах, место:", pd.cut(ro[ro.seg == "megacity"]["rank"], [50, 75, 100, 200, 1000]).value_counts(normalize=True).sort_index().round(2).to_dict())
