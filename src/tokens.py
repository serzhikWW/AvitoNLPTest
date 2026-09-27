"""Кэш нормализованных текстов объявлений по полям.

Токенизация + лемматизация ~190k объявлений с длинными описаниями занимает
минуты, поэтому результат сохраняется в cache/tokens_<mode>.parquet
(ключ — item_id; токены хранятся строкой через пробел).
Val-корпус — надмножество корпуса бенчмарка, так что один кэш обслуживает оба.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

from .data import CACHE_DIR
from .text import normalize

FIELDS = {
    "title": "item_title_raw",
    "desc": "item_description_raw",
    "params": "item_infm_params_text",
}


def _read_subset(path, need) -> pd.DataFrame:
    """Потоковое чтение кэша батчами с векторной фильтрацией по item_id.

    Пиковая память = нужные строки + один батч, а не весь кэш. Строки остаются
    Arrow-буферами (string[pyarrow]), а не миллионами Python-объектов.
    """
    value_set = pa.array(need, type=pa.large_string())
    parts = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000):
        batch = batch.cast(pa.schema([(f.name, pa.large_string()) for f in batch.schema]))
        parts.append(batch.filter(pc.is_in(batch.column("item_id"), value_set=value_set)))
    return pa.Table.from_batches(parts).to_pandas(types_mapper=pd.ArrowDtype)


def item_tokens(corpus: pd.DataFrame, mode: str = "lemma") -> pd.DataFrame:
    """DataFrame [item_id, title, desc, params] с нормализованными токенами (строки)."""
    path = CACHE_DIR / f"tokens_{mode}.parquet"
    need = corpus.item_id.values
    # читаем из кэша только нужные строки: кэш общий для всех корпусов (~0.5M объявлений),
    # целиком в памяти он занимает несколько ГБ
    cached = _read_subset(path, need) if path.exists() else None
    have = set() if cached is None else set(cached.item_id)
    todo = corpus[~corpus.item_id.isin(have)]
    if len(todo):
        new = pd.DataFrame({"item_id": todo.item_id.values})
        for f, col in FIELDS.items():
            new[f] = [" ".join(normalize(t, mode)) for t in tqdm(todo[col].values, desc=f"{mode}:{f}")]
        full = pd.read_parquet(path) if path.exists() else None
        pd.concat([full, new], ignore_index=True).to_parquet(path)
        del full
        new = new.astype("string[pyarrow]")
        cached = new if cached is None else pd.concat([cached, new], ignore_index=True)
    # переупорядочиваем под corpus через позиционный take (без копирования в Python-объекты)
    pos = pd.Series(np.arange(len(cached)), index=cached.item_id.astype(str).values)
    return cached.take(pos.loc[need].values).reset_index(drop=True)
