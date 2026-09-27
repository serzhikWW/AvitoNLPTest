"""Генерация answer.csv для бенчмарка.

Быстрый старт — финальное решение (отправка 5, Recall@50 = 0.899 на бенчмарке):
    python predict.py                    # = python predict.py --preset sub5
Любую прошлую отправку можно воспроизвести пресетом:
    python predict.py --preset sub3 --out answer_sub3.csv

Пресеты (val v2 -> бенчмарк):
  sub1  BM25 + 2*log P(item_loc | search_loc)                                   0.817 -> 0.809
  sub2  кандидаты BM25 ∪ e5-small ∪ фьюжн + LightGBM lambdarank                  0.883 -> 0.867
  sub3  то же, но дообученная e5-base (ft-val для val, ft-full для бенчмарка)     0.898 -> 0.881
  sub4  + гео-признаки + 100 кандидатов без сильного приора локации              0.913 -> 0.894
  sub5  + профиль кликов похожих запросов + косинус e5-small (ансамбль)           0.919 -> 0.899

Как устроен ранкер (sub2-sub5): он обучается на val-запросах — признаки считаются по train-fit
и корпусу val v2 моделью dense_val, — а затем применяется к бенчмарку, где признаки считаются по
ВСЕМУ train и корпусу бенчмарка моделью dense. dense_val и dense должны быть «одной» моделью
(одна и та же или её версии ft-val / ft-full), иначе распределения косинусов разойдутся.

Без --preset (--preset none) настройки задаются флагами --method/--dense/--fuse/... вручную.
Решение детерминировано: повторный запуск даёт побитово тот же answer.csv.
"""
import argparse
import gc

import pandas as pd

from src.baseline import predict_v1
from src.data import CACHE_DIR, ROOT, load_bench_items, load_bench_queries, load_train
from src.pipeline import TRAIN_COLS, make_builder
from src.ranker import FUSE, feats_tag, fit_ranker, ranker_features


def val_features(dense_name: str, opts: dict) -> pd.DataFrame:
    """Признаки + метки на val-запросах (с кэшем): обучающая выборка ранкера."""
    from src.validation import build_hard_corpus, load_split

    path = CACHE_DIR / f"ranker_feats_val_{feats_tag(dense_name, **opts)}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    train_fit, queries, qrels, _ = load_split()
    X = make_builder(build_hard_corpus(), train_fit, dense_name, **opts).build(queries, qrels)
    X.to_parquet(path)
    return X


def bench_features(dense_name: str, opts: dict) -> pd.DataFrame:
    """Признаки кандидатов бенчмарка (с кэшем): статистики по всему train, корпус бенчмарка."""
    path = CACHE_DIR / f"ranker_feats_bench_{feats_tag(dense_name, **opts)}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    B = make_builder(load_bench_items(), load_train(columns=TRAIN_COLS), dense_name, **opts).build(load_bench_queries())
    B.to_parquet(path)
    return B


def predict_ranker(dense_name: str, dense_val_name: str, opts: dict) -> dict:
    """opts — настройки FeatureBuilder (fuse, geo, n_noloc), одинаковые для val и бенчмарка."""
    # 1. обучаем ранкер на val
    X = val_features(dense_val_name, opts)
    feats = ranker_features(X)
    model = fit_ranker(X, feats)
    del X
    gc.collect()

    # 2. скор ранкера на кандидатах бенчмарка и top-50 внутри кандидатов каждого запроса
    B = bench_features(dense_name, opts)
    B["score"] = model.predict(B[feats])
    top = B.sort_values(["query_id", "score"], ascending=[True, False]).groupby("query_id").head(50)
    ids = load_bench_items(columns=["item_id"]).item_id.values
    return {q: list(ids[g.item_idx.values]) for q, g in top.groupby("query_id")}


FT_VAL = "models/multilingual-e5-base-ft-val"
FT_FULL = "models/multilingual-e5-base-ft-full"
E5_SMALL = "intfloat/multilingual-e5-small"
PRESETS = {
    "sub1": dict(method="v1"),
    "sub2": dict(method="ranker", dense=E5_SMALL, dense_val=None, fuse="0.5,1.0,0.05"),
    "sub3": dict(method="ranker", dense=FT_FULL, dense_val=FT_VAL, fuse="0.5,1.0,0.1"),
    "sub4": dict(method="ranker", dense=FT_FULL, dense_val=FT_VAL, fuse="0.5,1.0,0.1", geo=True, noloc=100),
    "sub5": dict(method="ranker", dense=FT_FULL, dense_val=FT_VAL, fuse="0.5,1.0,0.1", geo=True, noloc=100,
                 profile=True, extra_dense=[E5_SMALL]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="sub5", choices=[*PRESETS, "none"],
                    help="готовая конфигурация отправки; none — задать флагами ниже")
    ap.add_argument("--method", choices=["v1", "ranker"], default="ranker")
    ap.add_argument("--dense", default="intfloat/multilingual-e5-small", help="модель эмбеддингов для бенчмарка")
    ap.add_argument("--dense-val", default=None, help="модель для признаков val (по умолчанию = --dense)")
    ap.add_argument("--fuse", default=",".join(map(str, FUSE)),
                    help="веса фьюжна a,b,w: a*bm25_norm + b*cos + w*logP(loc)")
    ap.add_argument("--geo", action="store_true", help="гео-признаки (по итогам анализа ошибок)")
    ap.add_argument("--noloc", type=int, default=0, help="доп. кандидаты без сильного приора локации")
    ap.add_argument("--profile", action="store_true", help="профиль кликов похожих запросов (src/profile.py)")
    ap.add_argument("--extra-dense", nargs="*", default=[], help="доп. bi-encoder'ы, их косинус как признак")
    ap.add_argument("--out", default="answer.csv")
    args = ap.parse_args()
    if args.preset != "none":
        for k, v in PRESETS[args.preset].items():
            setattr(args, k, v)
        print(f"preset {args.preset}: {PRESETS[args.preset]}")

    queries = load_bench_queries()
    if args.method == "v1":
        train = load_train(columns=["search_location_id", "item_location_id"])
        preds = predict_v1(queries, load_bench_items(), train)
    else:
        opts = dict(fuse=tuple(float(x) for x in args.fuse.split(",")), geo=args.geo, n_noloc=args.noloc,
                    profile=args.profile, extra_dense=tuple(args.extra_dense))
        preds = predict_ranker(args.dense, args.dense_val or args.dense, opts)

    answer = pd.DataFrame({
        "query_id": queries.query_id.values,
        "answer": [" ".join(preds[q]) for q in queries.query_id.values],
    })
    answer.to_csv(ROOT / args.out, index=False)
    print("saved", len(answer), "rows ->", args.out)


if __name__ == "__main__":
    main()
