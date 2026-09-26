"""Эксперимент 1: BM25-бейзлайн и влияние нормализации текста.

Сравниваем raw / stem / lemma и разные наборы полей объявления.
Запуск: .venv/bin/python -m experiments.01_bm25  (из корня репозитория)
"""
import time

from src.bm25 import SparseBM25
from src.retrieval import batched, topk_indices
from src.text import normalize
from src.tokens import item_tokens
from src.validation import load_split, report

_, queries, qrels, corpus = load_split()
ids = corpus.item_id.values


def run(mode, fields, name, use_params=False, title_boost=1):
    tok = item_tokens(corpus, mode)
    docs = []
    for row in tok[list(fields)].itertuples(index=False):
        parts = []
        for f, text in zip(fields, row):
            parts += text.split() * (title_boost if f == "title" else 1)
        docs.append(parts)
    bm = SparseBM25().fit(docs)
    qtexts = queries.search_query + (" " + queries.search_infm_params_text if use_params else "")
    qt = [normalize(t, mode) for t in qtexts]
    preds = {}
    for sl in batched(len(qt)):
        top = topk_indices(bm.scores(qt[sl]))
        for qid, row in zip(queries.query_id.values[sl], top):
            preds[qid] = list(ids[row])
    report(preds, queries, qrels, name)


for mode in ["raw", "stem", "lemma"]:
    t = time.time()
    run(mode, ["title", "params", "desc"], f"{mode}: title+params+desc")
    print(f"  {time.time()-t:.0f}s")
run("lemma", ["title"], "lemma: title only")
run("lemma", ["title", "params"], "lemma: title+params")
run("lemma", ["title", "params", "desc"], "lemma: all, title x3", title_boost=3)
run("lemma", ["title", "params", "desc"], "lemma: all + query params", use_params=True)
