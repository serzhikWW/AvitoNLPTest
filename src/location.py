"""Локационный приор P(локация объявления | локация поиска).

EDA: если локация поиска — город, в 93% случаев выбранное объявление из этого же
города. Если это регион/страна (напр. 107620 — Московская обл., 621540 — вся РФ),
в корпусе объявлений с таким location_id нет вовсе, а выбранные объявления лежат
в «дочерних» городах (для 107620 — в основном Москва 637640).

Распределение оцениваем по train-fit (счётчики пар search_loc -> item_loc)
со сглаживанием: к наблюдаемым парам добавляется псевдосчётчик для «своей»
локации, чтобы редкие локации без истории всё равно тянули к себе свои объявления.
"""
import numpy as np
import pandas as pd


class LocationPrior:
    def __init__(self, self_pseudo: float = 5.0, floor: float = 1e-4):
        self.self_pseudo = self_pseudo
        self.floor = floor  # вероятность «любой другой» локации

    def fit(self, train: pd.DataFrame):
        pairs = train.groupby(["search_location_id", "item_location_id"]).size().rename("n").reset_index()
        self.pairs = pairs
        self.dist = {s: dict(zip(g.item_location_id, g.n)) for s, g in pairs.groupby("search_location_id")}
        return self

    def probs(self, search_loc: int) -> dict:
        d = dict(self.dist.get(search_loc, {}))
        d[search_loc] = d.get(search_loc, 0) + self.self_pseudo
        tot = sum(d.values())
        return {k: v / tot for k, v in d.items()}

    def log_prior(self, search_loc: int, item_locs: np.ndarray) -> np.ndarray:
        """log P(item_loc | search_loc) для каждого объявления корпуса."""
        p = self.probs(search_loc)
        vals = pd.Series(item_locs).map(p).fillna(0).to_numpy()
        return np.log(np.maximum(vals, self.floor)).astype(np.float32)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class GeoPrior:
    """Геометрия поиска: «центр» и «радиус» локации поиска по координатам выбранных объявлений.

    Зачем (см. experiments/07_error_analysis.py): при поиске по региону (Московская обл.,
    вся РФ) у четверти релевантных объявлений P(item_loc | search_loc) <= 0.02 — исполнитель
    из небольшого города внутри региона, и лог-приор его «топит». Приор знает только
    id локаций, а расстояние показывает, что городок лежит внутри региона, а не на другом
    конце страны.

    Центр локации поиска = медиана координат объявлений, выбранных в train при поиске
    в этой локации; радиус = медиана расстояний от них до центра. Для локаций поиска
    без истории центр — медиана координат объявлений корпуса с тем же location_id.
    """

    def fit(self, train: pd.DataFrame, corpus: pd.DataFrame):
        t = train[["search_location_id", "item_latitude", "item_longitude"]].astype(float)
        center = t.groupby("search_location_id")[["item_latitude", "item_longitude"]].median()
        t = t.join(center, on="search_location_id", rsuffix="_c")
        t["d"] = haversine_km(t.item_latitude, t.item_longitude, t.item_latitude_c, t.item_longitude_c)
        radius = t.groupby("search_location_id").d.median()
        own = corpus.assign(lat=corpus.item_latitude.astype(float), lon=corpus.item_longitude.astype(float)) \
            .groupby("item_location_id")[["lat", "lon"]].median()
        own.columns = ["item_latitude", "item_longitude"]
        center = pd.concat([center, own[~own.index.isin(center.index)]])
        self.center = {int(k): (v.item_latitude, v.item_longitude) for k, v in center.iterrows()}
        self.radius = radius.to_dict()
        return self

    def features(self, search_loc: int, lat: np.ndarray, lon: np.ndarray):
        """(расстояние до центра в км, расстояние / (радиус + 1)); NaN, если центр неизвестен."""
        c = self.center.get(int(search_loc))
        if c is None:
            nan = np.full(len(lat), np.nan, np.float32)
            return nan, nan
        d = haversine_km(lat, lon, c[0], c[1]).astype(np.float32)
        return d, d / (self.radius.get(search_loc, 0.0) + 1.0)
