"""Кэш нормализованных текстов объявлений по полям.

Токенизация + лемматизация ~190k объявлений с длинными описаниями занимает
минуты, поэтому результат сохраняется в cache/tokens_<mode>.parquet
(ключ — item_id; токены хранятся строкой через пробел).
Val-корпус — надмножество корпуса бенчмарка, так что один кэш обслуживает оба.
"""
import pandas as pd
from tqdm import tqdm

from .data import CACHE_DIR
from .text import normalize

FIELDS = {
    "title": "item_title_raw",
    "desc": "item_description_raw",
    "params": "item_infm_params_text",
}


def item_tokens(corpus: pd.DataFrame, mode: str = "lemma") -> pd.DataFrame:
    """DataFrame [item_id, title, desc, params] с нормализованными токенами (строки)."""
    path = CACHE_DIR / f"tokens_{mode}.parquet"
    cached = pd.read_parquet(path) if path.exists() else None
    todo = corpus if cached is None else corpus[~corpus.item_id.isin(set(cached.item_id))]
    if len(todo):
        new = pd.DataFrame({"item_id": todo.item_id.values})
        for f, col in FIELDS.items():
            new[f] = [" ".join(normalize(t, mode)) for t in tqdm(todo[col].values, desc=f"{mode}:{f}")]
        cached = new if cached is None else pd.concat([cached, new], ignore_index=True)
        cached.to_parquet(path)
    return cached.set_index("item_id").loc[corpus.item_id.values].reset_index()
