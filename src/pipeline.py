"""Фабрика FeatureBuilder: одинаковая сборка признаков для val, бенчмарка и экспериментов.

opts — настройки признаков (все, кроме fuse, по умолчанию выключены):
  fuse         веса фьюжна a,b,w;
  geo          гео-признаки;
  n_noloc      доп. кандидаты без сильного приора локации;
  profile      профиль кликов похожих запросов (src/profile.py);
  extra_dense  доп. bi-encoder'ы (имена моделей) — их косинус как признак.
"""
# ranker (lightgbm) импортируем раньше dense (torch): на macOS при обратном порядке две копии
# OpenMP конфликтуют и обучение LightGBM падает с segfault
from .ranker import FeatureBuilder  # noqa: I001
from .dense import DenseRetriever
from .profile import ClickProfile

TRAIN_COLS = ["search_query", "search_location_id", "item_location_id", "item_microcat_id", "item_id",
              "item_latitude", "item_longitude", "item_title_raw", "item_description_raw"]


def short_name(model: str) -> str:
    return model.rstrip("/").split("/")[-1].replace("multilingual-", "")


def make_builder(corpus, train, dense_name: str, fuse, geo=False, n_noloc=0, profile=False, extra_dense=()):
    dr = DenseRetriever(dense_name)
    prof = ClickProfile().fit(train, dr.query_embeddings, dr.item_embeddings) if profile else None
    extras = []
    for m in extra_dense:
        xr = DenseRetriever(m)
        extras.append((short_name(m), xr.item_embeddings(corpus), xr.query_embeddings))
    return FeatureBuilder(corpus, train, dr.item_embeddings(corpus), dr.query_embeddings,
                          fuse=fuse, geo=geo, n_noloc=n_noloc, profile=prof, extra_dense=extras)
