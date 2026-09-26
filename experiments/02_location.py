"""Эксперимент 2: BM25 (lemma, title x3) + локационный приор.

Скор = bm25 + w * log P(item_loc | search_loc). w=inf эквивалентно жёсткому
фильтру по локации (с бэкоффом на остальные, если своих кандидатов < 50).
"""
import numpy as np

from src.bm25 import SparseBM25
from src.location import LocationPrior
from src.retrieval import batched, topk_indices
from src.text import normalize
from src.tokens import item_tokens
from src.validation import load_split, report

train_fit, queries, qrels, corpus = load_split()
ids = corpus.item_id.values
tok = item_tokens(corpus, "lemma")
docs = [(t.split() * 3) + p.split() + d.split() for t, p, d in zip(tok.title, tok.params, tok.desc)]
bm = SparseBM25().fit(docs)
qt = [normalize(t) for t in queries.search_query]
lp = LocationPrior().fit(train_fit)
item_locs = corpus.item_location_id.values
qlocs = queries.search_location_id.values
priors = {l: lp.log_prior(l, item_locs) for l in np.unique(qlocs)}

S = np.vstack([bm.scores(qt[sl]) for sl in batched(len(qt))])
P = np.vstack([priors[l] for l in qlocs])
for w in [0, 0.5, 1, 2, 4, 1000]:
    top = topk_indices(S + w * P)
    preds = {q: list(ids[r]) for q, r in zip(queries.query_id, top)}
    report(preds, queries, qrels, f"bm25 + {w} * logP(loc)")
# доля релевантных, чья локация вообще имеет P>floor
rel = qrels.merge(corpus[["item_id", "item_location_id"]]).merge(queries[["query_id", "search_location_id"]])
ok = [lp.probs(s).get(l, 0) > 0 for s, l in zip(rel.search_location_id, rel.item_location_id)]
print("relevant items covered by location prior:", np.mean(ok))
