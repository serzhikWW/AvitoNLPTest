"""Эксперимент 3: почему бенчмарк сложнее валидации.

Гипотеза: корпус бенчмарка собран из выдач по его же запросам, поэтому у каждого
запроса бенчмарка в корпусе есть «родная» пачка похожих объявлений из той же
локации (конкуренты), а у val-запроса — только сам вставленный позитив.
Мера: сколько объявлений корпуса из той же локации содержат в тексте (заголовок+параметры+описание) ВСЕ леммы запроса.
"""
import numpy as np
import pandas as pd

from src.data import load_bench_queries
from src.text import normalize
from src.tokens import item_tokens
from src.validation import load_split

_, vq, _, corpus = load_split()
bq = load_bench_queries()
tok = item_tokens(corpus, "lemma")
title_sets = [set(f"{t} {p} {d}".split()) for t, p, d in zip(tok.title, tok.params, tok.desc)]
by_loc = pd.Series(np.arange(len(corpus))).groupby(corpus.item_location_id.values).apply(list).to_dict()


def competitors(q):
    out = []
    for text, loc in zip(q.search_query, q.search_location_id):
        terms = set(normalize(text))
        idx = by_loc.get(loc, [])
        out.append(sum(terms <= title_sets[i] for i in idx) if terms else 0)
    return pd.Series(out)


for name, q in [("val", vq), ("bench", bq)]:
    c = competitors(q)
    loc_hit = q.search_location_id.isin(by_loc.keys()).values
    c = c[loc_hit]
    print(f"{name:5s} (город есть в корпусе: {loc_hit.mean():.0%})  "
          f"медиана={c.median():.0f}  среднее={c.mean():.1f}  "
          f"доля с 0 конкурентов={(c == 0).mean():.0%}  с >=10={(c >= 10).mean():.0%}")
