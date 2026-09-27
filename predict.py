"""Генерация answer.csv для бенчмарка.

Методы:
  v1      BM25 + 2 * log P(item_loc | search_loc) (src/baseline.py).
          val v2: 0.817, бенчмарк: 0.809.
  ranker  кандидаты BM25 ∪ dense ∪ фьюжн (~540 на запрос) + LightGBM lambdarank (src/ranker.py).
          Ранкер обучается на val-запросах (признаки считаются по train-fit и корпусу val v2
          моделью --dense-val), затем применяется к бенчмарку: признаки считаются по ВСЕМУ train
          и корпусу бенчмарка моделью --dense. --dense-val и --dense должны быть «одной» моделью
          (одна и та же, либо её версии ft-val / ft-full), иначе распределения косинусов разные.

Примеры:
  python predict.py --method v1
  python predict.py --method ranker --dense intfloat/multilingual-e5-small
  python predict.py --method ranker --dense models/multilingual-e5-base-ft-full \\
                    --dense-val models/multilingual-e5-base-ft-val
Решение детерминировано: повторный запуск даёт тот же answer.csv.
"""
import argparse
import gc

import lightgbm as lgb
import pandas as pd

from src.baseline import predict_v1
from src.data import CACHE_DIR, ROOT, load_bench_items, load_bench_queries, load_train
from src.ranker import LGB_PARAMS, FeatureBuilder, ranker_features


def val_features(dense_name: str) -> pd.DataFrame:
    """Признаки + метки на val-запросах (с кэшем): обучающая выборка ранкера."""
    from src.dense import DenseRetriever
    from src.validation import build_hard_corpus, load_split

    tag = dense_name.rstrip("/").split("/")[-1]
    path = CACHE_DIR / f"ranker_feats_val_{tag}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    train_fit, queries, qrels, _ = load_split()
    corpus = build_hard_corpus()
    dr = DenseRetriever(dense_name)
    X = FeatureBuilder(corpus, train_fit, dr.item_embeddings(corpus), dr.query_embeddings).build(queries, qrels)
    X.to_parquet(path)
    return X


def predict_ranker(dense_name: str, dense_val_name: str) -> dict:
    from src.dense import DenseRetriever

    # 1. обучаем ранкер на val
    X = val_features(dense_val_name)
    feats = ranker_features(X)
    X = X.sort_values("query_id", kind="stable")
    model = lgb.LGBMRanker(**LGB_PARAMS)
    model.fit(X[feats], X.label, group=X.groupby("query_id", sort=True).size().values)
    del X
    gc.collect()

    # 2. признаки на бенчмарке: статистики по всему train, корпус бенчмарка
    items = load_bench_items()
    queries = load_bench_queries()
    train = load_train(columns=["search_query", "search_location_id", "item_location_id",
                                "item_microcat_id", "item_id"])
    dr = DenseRetriever(dense_name)
    fb = FeatureBuilder(items, train, dr.item_embeddings(items), dr.query_embeddings)
    B = fb.build(queries)

    # 3. скор ранкера и top-50 внутри кандидатов каждого запроса
    B["score"] = model.predict(B[feats])
    top = B.sort_values(["query_id", "score"], ascending=[True, False]).groupby("query_id").head(50)
    ids = items.item_id.values
    return {q: list(ids[g.item_idx.values]) for q, g in top.groupby("query_id")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["v1", "ranker"], default="ranker")
    ap.add_argument("--dense", default="intfloat/multilingual-e5-small", help="модель эмбеддингов для бенчмарка")
    ap.add_argument("--dense-val", default=None, help="модель для признаков val (по умолчанию = --dense)")
    ap.add_argument("--out", default="answer.csv")
    args = ap.parse_args()

    queries = load_bench_queries()
    if args.method == "v1":
        train = load_train(columns=["search_location_id", "item_location_id"])
        preds = predict_v1(queries, load_bench_items(), train)
    else:
        preds = predict_ranker(args.dense, args.dense_val or args.dense)

    answer = pd.DataFrame({
        "query_id": queries.query_id.values,
        "answer": [" ".join(preds[q]) for q in queries.query_id.values],
    })
    answer.to_csv(ROOT / args.out, index=False)
    print("saved", len(answer), "rows ->", args.out)


if __name__ == "__main__":
    main()
