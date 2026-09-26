"""Разреженный BM25 на scipy.

Готовые библиотеки (bm25s, rank_bm25) возвращают только top-k. Нам же нужен
полный вектор скоров по всему корпусу, чтобы потом складывать его с другими
сигналами (локация, эмбеддинги, история кликов) до отсечения top-50.

Реализация: матрица W [n_docs x vocab] с весами BM25 для каждого (doc, term).
Скор запроса q = сумма W[:, t] по уникальным термам t запроса = Q @ W.T,
где Q — бинарная матрица запросов. Всё считается одним sparse-dense умножением.
"""
import numpy as np
import scipy.sparse as sp


class SparseBM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, docs: list[list[str]]):
        vocab: dict[str, int] = {}
        rows, cols = [], []
        for i, toks in enumerate(docs):
            for t in toks:
                j = vocab.setdefault(t, len(vocab))
                rows.append(i)
                cols.append(j)
        n = len(docs)
        tf = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, len(vocab))
        )
        tf.sum_duplicates()  # tf[i, j] = сколько раз терм j встретился в документе i
        dl = np.asarray(tf.sum(1)).ravel()
        df = np.bincount(tf.indices, minlength=len(vocab))
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        # нормировка tf по длине документа (стандартная формула BM25)
        norm = self.k1 * (1 - self.b + self.b * dl / dl.mean())
        tf_row_norm = np.repeat(norm, np.diff(tf.indptr)).astype(np.float32)
        data = tf.data * (self.k1 + 1) / (tf.data + tf_row_norm)
        data *= idf[tf.indices]
        self.W_T = sp.csr_matrix((data, tf.indices, tf.indptr), shape=tf.shape).T.tocsr()
        self.vocab, self.idf, self.n_docs = vocab, idf, n
        return self

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
