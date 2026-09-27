"""«Профиль кликов» похожих запросов — коллаборативный сигнал для ранкера.

Идея: для запроса q находим K самых похожих запросов из train (по эмбеддингам запросов
той же bi-encoder модели) и собираем объявления, которые по ним реально выбирали.
Их взвешенный средний вектор — «профиль» того, что люди выбирают по такой потребности.
Признак кандидата — косинус его эмбеддинга с профилем. Пример: «тонировка стёкол в квартире»
близка к запросам, по которым выбирали плёнку для окон, а не автотонировку.
Тот же набор соседей даёт распределение подкатегорий выбранных объявлений (p_mc_nn) —
более устойчивую к формулировкам замену статистике по леммам.

Защита от утечки (leave-item-out): если сам кандидат входит в профиль (его выбирали по
соседнему запросу), его вклад вычитается. Иначе признак превратился бы в «объявление
кликали раньше» — а такая история есть у 40% позитивов val, но лишь у ~10% корпуса бенчмарка.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from .retrieval import batched, topk_indices


class ClickProfile:
    def __init__(self, k: int = 30, max_items: int = 20, temp: float = 0.05, seed: int = 42):
        self.k, self.max_items, self.temp, self.seed = k, max_items, temp, seed

    def fit(self, train: pd.DataFrame, query_encoder, item_embeddings):
        """item_embeddings(df с колонками объявления) -> матрица эмбеддингов в порядке df."""
        pairs = (train.drop_duplicates(["search_query", "item_id"])
                 .sample(frac=1.0, random_state=self.seed)
                 .groupby("search_query").head(self.max_items))
        items = pairs.drop_duplicates("item_id").reset_index(drop=True)
        self.E = item_embeddings(items).astype(np.float32)
        self.item_ids = items.item_id.values
        self.item_mc = items.item_microcat_id.values
        pos = pd.Series(np.arange(len(items)), index=items.item_id.values)
        grp = pairs.assign(ix=pos.loc[pairs.item_id.values].values).groupby("search_query").ix.apply(np.array)
        self.texts = grp.index.to_numpy()
        self.text_items = grp.values  # индексы объявлений, выбранных по каждому тексту
        self.T = query_encoder(self.texts.tolist()).astype(np.float32)
        return self

    def query_profiles(self, Qe: np.ndarray):
        """Для каждого запроса: (вектор профиля V, {item_id: вес}, {microcat: доля}, max сходство с соседом)."""
        out = []
        for sl in batched(len(Qe), 256):
            S = Qe[sl] @ self.T.T
            nn = topk_indices(S, self.k)
            for j in range(len(nn)):
                sims = S[j, nn[j]]
                w = np.exp((sims - sims.max()) / self.temp)
                w /= w.sum()
                item_w = defaultdict(float)
                for n, wn in zip(nn[j], w):
                    its = self.text_items[n]
                    for i in its:
                        item_w[i] += wn / len(its)
                idx = np.fromiter(item_w.keys(), int)
                a = np.fromiter(item_w.values(), float)
                V = (a[:, None] * self.E[idx]).sum(0)
                mc = defaultdict(float)
                for i, ai in zip(idx, a):
                    mc[self.item_mc[i]] += ai
                out.append((V.astype(np.float32), dict(zip(self.item_ids[idx], a)), dict(mc), float(sims.max())))
        return out

    @staticmethod
    def candidate_features(prof, cand_emb: np.ndarray, cand_ids: np.ndarray, cand_mc: np.ndarray):
        """prof_cos (косинус с профилем без вклада самого кандидата) и p_mc_nn (тоже leave-item-out)."""
        V, item_w, mc, _ = prof
        a = np.array([item_w.get(i, 0.0) for i in cand_ids], dtype=np.float32)
        dot = cand_emb @ V  # эмбеддинги нормированы: |e| = 1
        norm = np.sqrt(np.maximum(V @ V - 2 * a * dot + a * a, 1e-12))
        prof_cos = (dot - a) / norm
        p_mc = np.array([mc.get(m, 0.0) for m in cand_mc], dtype=np.float32) - a
        return prof_cos, np.maximum(p_mc, 0.0)
