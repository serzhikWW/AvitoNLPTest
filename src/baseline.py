"""Бейзлайн v1: BM25 (леммы, заголовок x3 + параметры + описание) + w * log P(item_loc | search_loc).

Одна функция для валидации и для бенчмарка, чтобы они гарантированно считались одинаково.
"""
import numpy as np
import pandas as pd

from .bm25 import SparseBM25
from .location import LocationPrior
from .retrieval import batched, topk_indices
from .text import normalize
from .tokens import item_tokens


def build_bm25(corpus: pd.DataFrame) -> SparseBM25:
    tok = item_tokens(corpus, "lemma")
    # заголовок x3 + параметры + описание (веса полей в tf)
    return SparseBM25().fit_fields([tok.title.values, tok.params.values, tok.desc.values], [3.0, 1.0, 1.0])


def predict_v1(queries: pd.DataFrame, corpus: pd.DataFrame, train: pd.DataFrame,
               loc_weight: float = 2.0, k: int = 50) -> dict:
    """{query_id: [item_id x k]} для набора запросов."""
    bm = build_bm25(corpus)
    lp = LocationPrior().fit(train)
    item_locs = corpus.item_location_id.values
    priors = {l: lp.log_prior(l, item_locs) for l in queries.search_location_id.unique()}
    qt = [normalize(t) for t in queries.search_query]
    ids = corpus.item_id.values
    qids, qlocs = queries.query_id.values, queries.search_location_id.values
    preds = {}
    for sl in batched(len(qt)):
        S = bm.scores(qt[sl])
        S += loc_weight * np.vstack([priors[l] for l in qlocs[sl]])
        for qid, row in zip(qids[sl], topk_indices(S, k)):
            preds[qid] = list(ids[row])
    return preds
