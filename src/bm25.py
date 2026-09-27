"""Разреженный BM25 на scipy.

Готовые библиотеки (bm25s, rank_bm25) возвращают только top-k. Нам же нужен
полный вектор скоров по всему корпусу, чтобы потом складывать его с другими
сигналами (локация, эмбеддинги, история кликов) до отсечения top-50.

Реализация: матрица W [n_docs x vocab] с весами BM25 для каждого (doc, term).
Скор запроса q = сумма W[:, t] по уникальным термам t запроса = Q @ W.T,
где Q — бинарная матрица запросов. Всё считается одним sparse-dense умножением.
"""
import itertools

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


def _split(doc):
    return doc.split() if isinstance(doc, str) else doc


class SparseBM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, docs):
        """docs: список документов (строки токенов через пробел или списки токенов)."""
        return self.fit_fields([docs], [1.0])

    def fit_fields(self, fields, weights):
        """Документ = взвешенная сумма полей: tf = sum_f weight_f * tf_f.

        fields: список полей, каждое — список длины n_docs (строки токенов через пробел).
        Вес 3 у заголовка эквивалентен повторению заголовка 3 раза, но без копирования токенов.
        CountVectorizer строит разреженную матрицу сразу, без промежуточных
        Python-списков: на 340k длинных документах это экономит ~10 ГБ памяти.
        """
        n = len(fields[0])
        cv = CountVectorizer(analyzer=_split, lowercase=False, dtype=np.float32)
        X = cv.fit_transform(itertools.chain.from_iterable(fields)).tocsr()
        tf = sum(w * X[i * n:(i + 1) * n] for i, w in enumerate(weights)).tocsr()
        self._fit_tf(tf, cv.vocabulary_)
        return self

    def _fit_tf(self, tf: sp.csr_matrix, vocab: dict):
        """tf[i, j] = (взвешенное) число вхождений терма j в документ i."""
        tf.sort_indices()
        n = tf.shape[0]
        dl = np.asarray(tf.sum(1)).ravel()
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        # нормировка tf по длине документа (стандартная формула BM25)
        norm = self.k1 * (1 - self.b + self.b * dl / dl.mean())
        tf_row_norm = np.repeat(norm, np.diff(tf.indptr)).astype(np.float32)
        data = tf.data * (self.k1 + 1) / (tf.data + tf_row_norm)
        data *= idf[tf.indices]
        self.W_T = sp.csr_matrix((data, tf.indices, tf.indptr), shape=tf.shape).T.tocsr()
        self.vocab, self.idf, self.n_docs = vocab, idf, n

    def query_matrix(self, queries: list[list[str]]) -> sp.csr_matrix:
        rows, cols = [], []
        for i, toks in enumerate(queries):
            for t in set(toks):
                j = self.vocab.get(t)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        return sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(queries), len(self.vocab)),
        )

    def scores(self, queries: list[list[str]]) -> np.ndarray:
        """Плотная матрица скоров [n_queries x n_docs]. Вызывать батчами."""
        Q = self.query_matrix(queries)
        return (Q @ self.W_T).toarray()
