"""Выгрузка пар «запрос — кандидат» для скоринга cross-encoder'ом (запускать на маке/CPU).

Первый уровень — финальный ранкер (отправка 4: ft-e5, фьюжн 0.5/1/0.1, --geo --noloc 100).
Cross-encoder дорогой, поэтому переранжирует только top-TOP_K первого уровня:
  * val:   OOF-скоры (5-fold по запросам) — каждый запрос скорится ранкером, его не видевшим;
  * bench: ранкер, обученный на всех val-запросах.
Сохраняет:
  cache/stage1_{val,bench}.parquet   — признаки + скор первого уровня для top-TOP_K (для второго уровня);
  cache/ce_pairs_{val,bench}.parquet — (query_id, item_id, query, doc): вход для score_cross_encoder.py.
Файлы ce_pairs_* нужно перенести на машину с GPU в ту же папку cache/.
"""
import pandas as pd

from predict import bench_features, val_features
from src.data import CACHE_DIR, load_bench_items, load_bench_queries
from src.dense import item_text
from src.ranker import fit_ranker, oof_scores, ranker_features
from src.validation import build_hard_corpus, load_split

TOP_K = 100
STAGE1 = dict(fuse=(0.5, 1.0, 0.1), geo=True, n_noloc=100)
DENSE_VAL = "models/multilingual-e5-base-ft-val"
DENSE_FULL = "models/multilingual-e5-base-ft-full"


def top_k(X: pd.DataFrame, score) -> pd.DataFrame:
    X = X.assign(stage1=score)
    X["stage1_rank"] = X.groupby("query_id").stage1.rank(ascending=False, method="first")
    return X[X.stage1_rank <= TOP_K].reset_index(drop=True)


def export(name: str, top: pd.DataFrame, queries: pd.DataFrame, corpus: pd.DataFrame):
    top = top.assign(item_id=corpus.item_id.values[top.item_idx.values])
    top.to_parquet(CACHE_DIR / f"stage1_{name}.parquet")
    docs = pd.Series(item_text(corpus), index=corpus.item_id.values)
    qtext = queries.set_index("query_id").search_query
    pairs = pd.DataFrame({
        "query_id": top.query_id.values,
        "item_id": top.item_id.values,
        "query": qtext.loc[top.query_id.values].astype(str).values,
        "doc": docs.loc[top.item_id.values].astype(str).values,
    })
    pairs.to_parquet(CACHE_DIR / f"ce_pairs_{name}.parquet")
    r = top.groupby("query_id").label.max().mean() if "label" in top else float("nan")
    print(f"{name}: {len(pairs)} pairs, {pairs.query_id.nunique()} queries, "
          f"доля запросов с позитивом в top-{TOP_K}: {r:.4f}")


X = val_features(DENSE_VAL, STAGE1)
feats = ranker_features(X)
_, vq, _, _ = load_split()
export("val", top_k(X, oof_scores(X, feats)), vq, build_hard_corpus())

model = fit_ranker(X, feats)
del X
B = bench_features(DENSE_FULL, STAGE1)
export("bench", top_k(B, model.predict(B[feats])), load_bench_queries(), load_bench_items())
