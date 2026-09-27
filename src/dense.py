"""Плотный (семантический) ретривал на bi-encoder'е.

Трансформер кодирует запрос и объявление по отдельности в векторы; релевантность —
косинус (векторы нормированы, поэтому это просто скалярное произведение).
Эмбеддинги объявлений считаются один раз и кэшируются в cache/emb_<model>.npy
(порядок строк — cache/emb_<model>_ids.parquet), так что val и бенчмарк используют
один и тот же кэш: val-корпус v2 — надмножество корпуса бенчмарка.

Модели open-source и работают локально (sentence-transformers); после первой загрузки
веса лежат в кэше HuggingFace, обращений к внешним API нет.
"""
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from .data import CACHE_DIR

# Префиксы, с которыми обучались модели семейства E5 (без них качество заметно падает).
PREFIXES = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
}
DESC_CHARS = 400  # начало описания; дальше в 128 токенов всё равно не влезет


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def item_text(corpus: pd.DataFrame) -> list[str]:
    """Заголовок + начало описания. Параметры (item_infm_params_text) не берём:
    они длинные и шаблонные («Тип стоимости за услугу ...») и съедают окно в 128 токенов."""
    return (corpus.item_title_raw + ". " + corpus.item_description_raw.str.slice(0, DESC_CHARS)).tolist()


class DenseRetriever:
    def __init__(self, model_name: str, max_len: int = 128, batch_size: int = 128):
        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device())
        self.model.max_seq_length = max_len
        self.batch_size = batch_size
        # дообученные модели лежат локально (models/<имя>) и наследуют E5-префиксы
        self.q_prefix, self.d_prefix = PREFIXES.get(
            model_name, ("query: ", "passage: ") if "e5" in model_name else ("", "")
        )
        self.tag = model_name.rstrip("/").split("/")[-1]

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=True,
        ).astype(np.float16)

    def item_embeddings(self, corpus: pd.DataFrame) -> np.ndarray:
        """Эмбеддинги объявлений в порядке corpus (с кэшем по item_id)."""
        emb_path = CACHE_DIR / f"emb_{self.tag}.npy"
        ids_path = CACHE_DIR / f"emb_{self.tag}_ids.parquet"
        if emb_path.exists():
            emb, ids = np.load(emb_path), pd.read_parquet(ids_path).item_id
        else:
            emb, ids = np.zeros((0, self.model.get_embedding_dimension()), np.float16), pd.Series([], dtype=str)
        todo = corpus[~corpus.item_id.isin(set(ids))]
        if len(todo):
            # сортируем по длине текста: меньше паддинга в батчах -> заметно быстрее
            texts = [self.d_prefix + t for t in item_text(todo)]
            order = np.argsort([len(t) for t in texts])
            new = np.empty((len(texts), emb.shape[1]), np.float16)
            new[order] = self._encode([texts[i] for i in order])
            emb = np.vstack([emb, new])
            ids = pd.concat([ids, todo.item_id], ignore_index=True)
            np.save(emb_path, emb)
            pd.DataFrame({"item_id": ids}).to_parquet(ids_path)
        pos = pd.Series(np.arange(len(ids)), index=ids.values)
        return emb[pos.loc[corpus.item_id.values].values]

    def query_embeddings(self, texts) -> np.ndarray:
        return self._encode([self.q_prefix + t for t in texts])
