"""Эксперимент 5: bi-encoder эмбеддинги и фьюжн с BM25 на val v2.

Сигналы (для каждого запроса — вектор по всему корпусу):
  bm  — BM25, нормированный на максимум по запросу (к [0, 1], чтобы сопоставить с косинусом);
  cos — косинус эмбеддингов запроса и объявления;
  loc — log P(item_loc | search_loc).
Итоговый скор = a * bm + b * cos + w * loc; веса перебираем по сетке.
Дополнительно: Recall@300 объединения top-300 BM25 и top-300 dense — потолок
для следующего этапа (ранкер над кандидатами).
Запуск: .venv/bin/python -m experiments.05_dense intfloat/multilingual-e5-small
"""
import sys
from itertools import product

import numpy as np

from src.baseline import build_bm25
from src.dense import DenseRetriever
from src.location import LocationPrior
from src.retrieval import batched, topk_indices
from src.text import normalize
from src.validation import build_hard_corpus, load_split, recall_at_k

model_name = sys.argv[1] if len(sys.argv) > 1 else "intfloat/multilingual-e5-small"
train_fit, queries, qrels, _ = load_split()
corpus = build_hard_corpus()
ids = corpus.item_id.values

bm = build_bm25(corpus)
dr = DenseRetriever(model_name)
E = dr.item_embeddings(corpus).astype(np.float32)
Qe = dr.query_embeddings(queries.search_query.tolist()).astype(np.float32)
lp = LocationPrior().fit(train_fit)
priors = {l: lp.log_prior(l, corpus.item_location_id.values) for l in queries.search_location_id.unique()}
qt = [normalize(t) for t in queries.search_query]

configs = {"bm25+loc (v1)": (1, 0, None)}  # None = сырой bm25 как в v1, w=2
configs["dense only"] = (0, 1, 0)
for w in [0.02, 0.05, 0.1, 0.2]:
    configs[f"dense+loc w={w}"] = (0, 1, w)
for a, b, w in product([0.3, 0.5, 1.0], [1.0], [0.05, 0.1, 0.2]):
    configs[f"fuse a={a} b={b} w={w}"] = (a, b, w)

preds = {c: {} for c in configs}
union300, bm300, de300 = {}, {}, {}
qids, qlocs = queries.query_id.values, queries.search_location_id.values
for sl in batched(len(qt), 128):
    B = bm.scores(qt[sl])
    C = Qe[sl] @ E.T
    L = np.vstack([priors[l] for l in qlocs[sl]])
    Bn = B / np.maximum(B.max(1, keepdims=True), 1e-6)
    for name, (a, b, w) in configs.items():
        S = B + 2.0 * L if w is None else a * Bn + b * C + w * L
        for q, r in zip(qids[sl], topk_indices(S, 50)):
            preds[name][q] = list(ids[r])
    # потолок для ранкера: объединение top-300 двух источников (оба с локацией)
    tb = topk_indices(B + 2.0 * L, 300)
    td = topk_indices(C + 0.1 * L, 300)
    for q, x, y in zip(qids[sl], tb, td):
        bm300[q], de300[q] = list(ids[x]), list(ids[y])
        union300[q] = list(dict.fromkeys(list(ids[x]) + list(ids[y])))

seen = queries.set_index("query_id").seen
for name, p in list(preds.items()) + [("R@300 bm25+loc", bm300), ("R@300 dense+loc", de300), ("R@600 union", union300)]:
    k = 600 if "union" in name else (300 if name.startswith("R@300") else 50)
    r = recall_at_k(p, qrels, k)
    s = seen.reindex(r.index)
    print(f"{name:<32s} R={r.mean():.4f}  seen={r[s].mean():.4f}  unseen={r[~s].mean():.4f}")
