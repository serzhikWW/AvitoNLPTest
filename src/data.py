"""Загрузка данных.

Все идентификаторы (query_id, item_id) читаются и хранятся строго как строки:
по ТЗ приведение к числу/смена регистра молча портит метрику.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "NLP_avito_interns-dataset"
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

QUERY_COLS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
ITEM_COLS = [
    "item_id",
    "item_title_raw",
    "item_description_raw",
    "item_infm_params_text",
    "item_category_id",
    "item_microcat_id",
    "item_price",
    "item_rating",
    "item_rating_reviews_count",
    "item_location_id",
    "item_latitude",
    "item_longitude",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
]


def _fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит типы: id -> str, decimal -> float, пропуски в тексте -> ''."""
    for c in ("item_id", "query_id"):
        if c in df:
            df[c] = df[c].astype(str)
    for c in ("item_price", "item_latitude", "item_longitude"):
        if c in df:
            df[c] = df[c].astype(float)
    return compact_text(df)


def compact_text(df: pd.DataFrame) -> pd.DataFrame:
    """Текстовые колонки -> string[pyarrow] (пропуски -> "").

    Arrow-строки лежат в одном буфере, а не миллионами Python-объектов: на корпусе
    в 340k объявлений с длинными описаниями это экономит несколько ГБ памяти.
    """
    for c in df.columns:
        if c.endswith(("_raw", "_text")) or c == "search_query":
            df[c] = df[c].astype("string[pyarrow]").fillna("")
    return df


def load_train(columns=None) -> pd.DataFrame:
    return _fix_types(pd.read_parquet(DATA_DIR / "train.parquet", columns=columns))


def load_bench_queries() -> pd.DataFrame:
    return _fix_types(pd.read_parquet(DATA_DIR / "benchmark_queries.parquet"))


def load_bench_items(columns=None) -> pd.DataFrame:
    return _fix_types(pd.read_parquet(DATA_DIR / "benchmark_items.parquet", columns=columns))
