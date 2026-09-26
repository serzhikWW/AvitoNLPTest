"""Генерация answer.csv для бенчмарка.

Версия v1: BM25 (лемматизация pymorphy3, заголовок x3 + параметры + описание)
+ 2 * log P(item_loc | search_loc), приор выучен на ВСЁМ train.
Валидация этой же конфигурации: Recall@50 = 0.870 (experiments/02_location.py).
Решение детерминировано: повторный запуск даёт тот же answer.csv.
"""
import numpy as np
import pandas as pd

from src.bm25 import SparseBM25
from src.data import ROOT, load_bench_items, load_bench_queries, load_train
from src.location import LocationPrior
from src.retrieval import batched, topk_indices
from src.text import normalize
from src.tokens import item_tokens

LOC_WEIGHT = 2.0
K = 50

items = load_bench_items()
queries = load_bench_queries()
train = load_train(columns=["search_location_id", "item_location_id"])

# 1. Лексический индекс по корпусу бенчмарка
tok = item_tokens(items, "lemma")
docs = [(t.split() * 3) + p.split() + d.split() for t, p, d in zip(tok.title, tok.params, tok.desc)]
bm = SparseBM25().fit(docs)

# 2. Локационный приор по всему train
lp = LocationPrior().fit(train)
item_locs = items.item_location_id.values
priors = {l: lp.log_prior(l, item_locs) for l in queries.search_location_id.unique()}

# 3. Скоринг батчами и top-50 (всегда ровно 50 — пустые слоты метрике не помогают)
qt = [normalize(t) for t in queries.search_query]
ids = items.item_id.values
answers = []
for sl in batched(len(qt)):
    S = bm.scores(qt[sl])
    S += LOC_WEIGHT * np.vstack([priors[l] for l in queries.search_location_id.values[sl]])
    answers += [" ".join(ids[r]) for r in topk_indices(S, K)]

answer = pd.DataFrame({"query_id": queries.query_id.values, "answer": answers})
answer.to_csv(ROOT / "answer.csv", index=False)
print("saved", len(answer), "rows")
