"""Эксперимент 4: калибровка val v2 («сложный» корпус с имитацией выдачи).

Подбираем число дистракторов на запрос так, чтобы бейзлайн v1 на val давал
примерно тот же Recall@50, что и на бенчмарке (0.809).
"""
from src.baseline import predict_v1
from src.validation import build_hard_corpus, load_split, report

train_fit, queries, qrels, corpus = load_split()
report(predict_v1(queries, corpus, train_fit), queries, qrels, "v1 @ val v1 (без дистракторов)")
for n in [30, 60, 100]:
    hard = build_hard_corpus(n)
    report(predict_v1(queries, hard, train_fit), queries, qrels, f"v1 @ val v2, {n} дистракторов")
print("бенчмарк v1: 0.8093")
