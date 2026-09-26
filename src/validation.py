"""Офлайн-валидация, имитирующая бенчмарк.

Что мы знаем о бенчмарке из EDA:
  * все 2452 запроса имеют уникальный текст (т.е. это не выборка событий
    пропорционально популярности, а скорее выборка уникальных запросов);
  * ~37% текстов бенчмарка встречаются в train;
  * «запрос» = поисковое событие: текст + локация + фильтры + категория,
    релевантные — выбранные по нему объявления (в среднем ~1.4 шт.).

Поэтому валидация строится так:
  1. Единица валидации — группа строк train с одинаковыми
     (search_query, search_location_id, search_infm_params_text, search_category).
     Её объявления — релевантные для этой группы.
  2. Берём одну группу на уникальный текст. ~63% групп — «невиденные»:
     их текст целиком удаляется из train-fit. ~37% — «виденные»: удаляется только
     сама группа, другие группы с этим текстом остаются в train-fit.
  3. Корпус для валидации = корпус бенчмарка + релевантные объявления валидации
     (без них искать было бы нечего). Все статистики, которые строятся по train
     (история кликов, локационные приоры и т.п.), считаются только по train-fit.
"""
import numpy as np
import pandas as pd

from .data import CACHE_DIR, ITEM_COLS, QUERY_COLS, load_bench_items, load_train

GROUP_KEY = ["search_query", "search_location_id", "search_infm_params_text", "search_category"]
SEED = 42
N_VAL = 3000
SEEN_SHARE = 0.37  # доля запросов бенчмарка, чей текст встречается в train


def build_split(n_val: int = N_VAL, seen_share: float = SEEN_SHARE, seed: int = SEED):
    """Строит сплит и кэширует его в cache/val_*.parquet."""
    rng = np.random.default_rng(seed)
    train = load_train()
    train["gid"] = train.groupby(GROUP_KEY, sort=False).ngroup()

    # Тексты, у которых больше одной группы, — кандидаты в «виденные»:
    # после удаления одной группы текст останется в train-fit.
    groups_per_text = train.drop_duplicates("gid").groupby("search_query").size()
    multi_texts = groups_per_text.index[groups_per_text > 1].to_numpy()
    all_texts = groups_per_text.index.to_numpy()

    n_seen = int(round(n_val * seen_share))
    n_unseen = n_val - n_seen
    seen_texts = rng.choice(multi_texts, n_seen, replace=False)
    rest = np.setdiff1d(all_texts, seen_texts)
    unseen_texts = rng.choice(rest, n_unseen, replace=False)

    gids = train[["gid", "search_query"]].drop_duplicates("gid")
    # по одной случайной группе на каждый выбранный текст
    pick = (
        gids[gids.search_query.isin(np.concatenate([seen_texts, unseen_texts]))]
        .sample(frac=1.0, random_state=seed)
        .drop_duplicates("search_query")
    )
    val_gids = set(pick.gid)

    is_val = train.gid.isin(val_gids)
    # train-fit: всё, кроме val-групп и всех строк «невиденных» текстов
    is_fit = ~is_val & ~train.search_query.isin(set(unseen_texts))

    val_rows = train[is_val]
    queries = val_rows.drop_duplicates("gid")[["gid"] + QUERY_COLS].copy()
    queries["query_id"] = "val_" + queries.gid.astype(str)
    queries["seen"] = queries.search_query.isin(set(seen_texts))
    qrels = val_rows[["gid", "item_id"]].drop_duplicates()
    qrels["query_id"] = "val_" + qrels.gid.astype(str)

    # корпус: бенчмарк + позитивы валидации (их признаки берём из train)
    bench = load_bench_items()
    pos_items = val_rows[ITEM_COLS].drop_duplicates("item_id")
    pos_items = pos_items[~pos_items.item_id.isin(set(bench.item_id))]
    corpus = pd.concat([bench, pos_items], ignore_index=True)

    train[is_fit].drop(columns=["gid"]).to_parquet(CACHE_DIR / "val_train_fit.parquet")
    queries.drop(columns=["gid"]).to_parquet(CACHE_DIR / "val_queries.parquet")
    qrels[["query_id", "item_id"]].to_parquet(CACHE_DIR / "val_qrels.parquet")
    corpus.to_parquet(CACHE_DIR / "val_corpus.parquet")
    print(
        f"val queries={len(queries)} (seen={queries.seen.sum()}), qrels={len(qrels)}, "
        f"train_fit rows={is_fit.sum()}, corpus={len(corpus)} (+{len(pos_items)} val positives)"
    )


def load_split():
    """Возвращает (train_fit, val_queries, val_qrels, val_corpus)."""
    if not (CACHE_DIR / "val_corpus.parquet").exists():
        build_split()
    return tuple(
        pd.read_parquet(CACHE_DIR / f"val_{n}.parquet")
        for n in ("train_fit", "queries", "qrels", "corpus")
    )


def recall_at_k(predictions: dict, qrels: pd.DataFrame, k: int = 50) -> pd.Series:
    """Recall@k по каждому запросу, ровно как в метрике бенчмарка.

    predictions: {query_id: [item_id, ...]} (берутся первые k, дубликаты не считаются)
    qrels: DataFrame с колонками query_id, item_id.
    """
    rel = qrels.groupby("query_id").item_id.apply(set)
    out = {}
    for qid, items in rel.items():
        top = set(predictions.get(qid, [])[:k])
        out[qid] = len(top & items) / len(items)
    return pd.Series(out, name=f"recall@{k}")


def report(predictions: dict, queries: pd.DataFrame, qrels: pd.DataFrame, name: str = "") -> float:
    """Печатает Recall@50 целиком и по сегментам seen/unseen."""
    r = recall_at_k(predictions, qrels)
    seen = queries.set_index("query_id").seen.reindex(r.index)
    print(
        f"{name:<40s} R@50={r.mean():.4f}  seen={r[seen].mean():.4f}  unseen={r[~seen].mean():.4f}"
    )
    return r.mean()


if __name__ == "__main__":
    build_split()
