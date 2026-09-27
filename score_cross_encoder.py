"""Скоринг пар «запрос — кандидат» дообученным cross-encoder'ом (запускать на GPU).

Вход:  cache/ce_pairs_{val,bench}.parquet (готовит export_ce_pairs.py).
Выход: cache/ce_scores_{val,bench}.parquet с колонками (query_id, item_id, ce_score).

Val-пары скорятся моделью ce-val (не видела val-запросов), bench-пары — моделью ce-full,
так же как для bi-encoder'а (ft-val / ft-full).

Запуск:
  python score_cross_encoder.py --split val   --model models/cross-encoder-russian-msmarco-ce-val
  python score_cross_encoder.py --split bench --model models/cross-encoder-russian-msmarco-ce-full
"""
import argparse

import numpy as np
import pandas as pd
import torch
from sentence_transformers.cross_encoder import CrossEncoder

from src.data import CACHE_DIR
from src.dense import device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "bench"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="для отладки: только первые N пар")
    args = ap.parse_args()

    pairs = pd.read_parquet(CACHE_DIR / f"ce_pairs_{args.split}.parquet")
    if args.limit:
        pairs = pairs.head(args.limit)
    kw = {"dtype": torch.float16} if torch.cuda.is_available() else {}
    model = CrossEncoder(args.model, device=device(), model_kwargs=kw)
    # сортировка по длине документа: меньше паддинга в батчах -> быстрее; результат возвращаем в исходный порядок
    order = np.argsort(pairs.doc.str.len().values)
    scores = model.predict(
        [(pairs["query"].values[i], pairs.doc.values[i]) for i in order],
        batch_size=args.batch, show_progress_bar=True, convert_to_numpy=True,
    )
    out = np.empty(len(pairs), dtype=np.float32)
    out[order] = np.asarray(scores, dtype=np.float32).ravel()
    res = pairs[["query_id", "item_id"]].assign(ce_score=out)
    path = CACHE_DIR / f"ce_scores_{args.split}{'_debug' if args.limit else ''}.parquet"
    res.to_parquet(path)
    print(f"saved {len(res)} scores -> {path}")


if __name__ == "__main__":
    main()
